"""MCPTriageSurface — the L1 alert triage tool surface.

Wraps FastMCP (3.x API) into a reusable scaffold. Each domain (the conveyor,
the ROC workflow) subclass this and register their own tools. The surface
gives the L1 triage agent a strict, validated interface to the system.

Why this abstraction exists:
- The JD asks for "tools for our globally staffed Remote Operations Center".
- An L1 triage agent needs a strict, validated interface — not a free-form
  REST API the LLM can call with any payload.
- FastMCP gives us Pydantic-validated tool contracts at the boundary.
- Subclassing means each module owns its own tools, not a god class.

The 5 reference tools (defined by the AlertSource protocol):
1. get_recent_alerts(limit) — pull the last N perception events
2. classify_alert(alert_id) — return routine/attention/escalate
3. escalate_alert(alert_id, reason) — mark as escalated, log reason
4. get_system_health() — throughput, latency, drift indicators
5. log_resolution(alert_id, action) — record the action taken

The triage agent (an LLM) calls these tools to walk the alert queue.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Protocol

logger = logging.getLogger(__name__)


@dataclass
class Alert:
    """A single perception event that needs triage.

    Attributes:
        id: Unique alert ID (UUID or hash of timestamp+class+track_id).
        timestamp: When the event was detected (UTC).
        class_name: Detected object class.
        confidence: Detection confidence in [0, 1].
        track_id: Stable ID from TrackingPipeline (None if not tracked).
        severity: Computed severity (routine, attention, escalate).
        bbox: Optional bounding box in original image coordinates.
        metadata: Optional extra context (camera_id, line_id, etc.).
    """

    id: str
    timestamp: datetime
    class_name: str
    confidence: float
    track_id: int | None = None
    severity: str = "routine"
    bbox: tuple[float, float, float, float] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "timestamp": self.timestamp.isoformat(),
            "class_name": self.class_name,
            "confidence": self.confidence,
            "track_id": self.track_id,
            "severity": self.severity,
            "bbox": list(self.bbox) if self.bbox else None,
            "metadata": self.metadata,
        }


class AlertSource(Protocol):
    """Protocol for the alert source backing the triage surface.

    Subclass this to provide your own alert queue. The default implementation
    is `InMemoryAlertQueue` (a thread-safe queue for the prototype).
    """

    def get_recent(self, limit: int) -> list[Alert]:
        """Return the last `limit` alerts, newest first."""
        ...

    def classify(self, alert_id: str) -> str:
        """Classify an alert. Returns one of: routine, attention, escalate."""
        ...

    def escalate(self, alert_id: str, reason: str) -> None:
        """Mark an alert as escalated. Logs the reason for audit."""
        ...

    def get_health(self) -> dict[str, Any]:
        """Return system health: throughput, latency, drift indicators."""
        ...

    def log_resolution(self, alert_id: str, action: str) -> None:
        """Record the action taken on an alert."""
        ...


class InMemoryAlertQueue:
    """Thread-safe in-memory alert queue for the prototype.

    Production swap: this is the seam. A real ROC integration would replace
    this with a Kafka subscription, a Postgres-backed queue, or a ROS topic
    subscription. The protocol stays the same.
    """

    def __init__(self, max_size: int = 10000):
        import threading
        from collections import deque

        self._alerts: deque[Alert] = deque(maxlen=max_size)
        self._resolutions: dict[str, str] = {}
        self._lock = threading.Lock()

    def push(self, alert: Alert) -> None:
        with self._lock:
            self._alerts.append(alert)

    def get_recent(self, limit: int) -> list[Alert]:
        with self._lock:
            return list(reversed(list(self._alerts)))[:limit]

    def classify(self, alert_id: str) -> str:
        with self._lock:
            for alert in self._alerts:
                if alert.id == alert_id:
                    return alert.severity
        return "unknown"

    def escalate(self, alert_id: str, reason: str) -> None:
        with self._lock:
            for alert in self._alerts:
                if alert.id == alert_id:
                    alert.severity = "escalate"
                    alert.metadata["escalation_reason"] = reason
                    alert.metadata["escalated_at"] = datetime.now(
                        tz=UTC
                    ).isoformat()
                    return

    def get_health(self) -> dict[str, Any]:
        with self._lock:
            n = len(self._alerts)
            if n == 0:
                return {
                    "queue_size": 0,
                    "throughput_per_min": 0.0,
                    "avg_inference_ms": 0.0,
                    "drift_indicators": {},
                }
            # Naive: estimate throughput from the time span of alerts
            oldest = self._alerts[0].timestamp
            newest = self._alerts[-1].timestamp
            span_sec = max(1.0, (newest - oldest).total_seconds())
            throughput_per_min = (n / span_sec) * 60
            return {
                "queue_size": n,
                "throughput_per_min": round(throughput_per_min, 2),
                "avg_inference_ms": 0.0,  # Set by the pipeline
                "drift_indicators": {},  # Set by the DriftMonitor
            }

    def log_resolution(self, alert_id: str, action: str) -> None:
        with self._lock:
            self._resolutions[alert_id] = action
            for alert in self._alerts:
                if alert.id == alert_id:
                    alert.metadata["resolution_action"] = action
                    alert.metadata["resolved_at"] = datetime.now(
                        tz=UTC
                    ).isoformat()
                    return


class MCPTriageSurface:
    """FastMCP-based triage surface. Subclass to add domain tools.

    The 5 reference tools (get_recent_alerts, classify_alert, escalate_alert,
    get_system_health, log_resolution) are registered by default. Subclass
    to add more tools relevant to your domain.

    Example:
        >>> from conveyor_perception.core.triage_surface import (
        ...     MCPTriageSurface, InMemoryAlertQueue,
        ... )
        >>> queue = InMemoryAlertQueue()
        >>> surface = MCPTriageSurface("l1-triage", queue)
        >>> surface.run()  # starts the MCP server (stdio by default)
    """

    def __init__(self, name: str, alert_source: AlertSource):
        self.name = name
        self.alert_source = alert_source
        self._server = None  # FastMCP instance, lazy-init

    def _ensure_server(self):
        if self._server is not None:
            return
        try:
            from fastmcp import FastMCP  # type: ignore

            self._server = FastMCP(self.name)
            self._register_default_tools()
        except ImportError as e:
            raise RuntimeError(
                "FastMCP not installed. Install with: pip install fastmcp>=3.4.7"
            ) from e

    def _register_default_tools(self) -> None:
        """Register the 5 reference tools on the MCP server."""
        assert self._server is not None

        @self._server.tool()
        def get_recent_alerts(limit: int = 10) -> list[dict[str, Any]]:
            """Return the last `limit` perception events, newest first.

            Args:
                limit: Maximum number of alerts to return (default 10, max 1000).
            """
            limit = max(1, min(1000, int(limit)))
            return [a.to_dict() for a in self.alert_source.get_recent(limit)]

        @self._server.tool()
        def classify_alert(alert_id: str) -> dict[str, str]:
            """Classify an alert. Returns routine, attention, or escalate.

            Args:
                alert_id: The alert ID returned by get_recent_alerts.
            """
            severity = self.alert_source.classify(alert_id)
            return {"alert_id": alert_id, "severity": severity}

        @self._server.tool()
        def escalate_alert(alert_id: str, reason: str) -> dict[str, str]:
            """Mark an alert as escalated. Logs the reason for audit.

            Args:
                alert_id: The alert ID to escalate.
                reason: Why this alert is being escalated (free text, but
                    validated: max 500 chars, no special characters).
            """
            reason = self._validate_reason(reason)
            self.alert_source.escalate(alert_id, reason)
            return {"alert_id": alert_id, "status": "escalated", "reason": reason}

        @self._server.tool()
        def get_system_health() -> dict[str, Any]:
            """Return system health: throughput, latency, drift indicators."""
            return self.alert_source.get_health()

        @self._server.tool()
        def log_resolution(alert_id: str, action: str) -> dict[str, str]:
            """Record the action taken on an alert.

            Args:
                alert_id: The alert ID being resolved.
                action: The action taken (e.g., 'auto-resolved', 'paged-on-call').
            """
            action = self._validate_action(action)
            self.alert_source.log_resolution(alert_id, action)
            return {"alert_id": alert_id, "status": "logged", "action": action}

    @staticmethod
    def _validate_reason(reason: str) -> str:
        """Sanitize the escalation reason. Prompt-injection resistance."""
        reason = (reason or "").strip()
        if not reason:
            raise ValueError("escalation reason cannot be empty")
        if len(reason) > 500:
            raise ValueError("escalation reason must be <= 500 chars")
        # Strip control characters; allow printable text + common punctuation
        return "".join(c for c in reason if c.isprintable() or c in "\n\t")

    @staticmethod
    def _validate_action(action: str) -> str:
        """Validate the resolution action against an enum-like list."""
        allowed = {
            "auto-resolved",
            "paged-on-call",
            "escalated-to-l2",
            "deferred",
            "false-positive",
        }
        action = (action or "").strip().lower()
        if action not in allowed:
            raise ValueError(
                f"action must be one of {sorted(allowed)}, got: {action!r}"
            )
        return action

    def run(self, transport: str = "stdio", **kwargs: Any) -> None:
        """Start the MCP server.

        Args:
            transport: "stdio" (default) or "http" or "sse".
            **kwargs: Passed to FastMCP.run() (e.g., host, port for http).
        """
        self._ensure_server()
        assert self._server is not None
        logger.info("Starting MCPTriageSurface: %s (transport=%s)", self.name, transport)
        self._server.run(transport=transport, **kwargs)
