"""Monitoring dashboard.

Aggregates real-time metrics from the perception + triage + drift layers
into a single dashboard snapshot. The ROC operator (or a downstream alerting
system) reads this snapshot every minute to:

- Track per-class detection volume per shift
- Catch P95/P99 latency regressions before they hit the SLA
- Spot the alert queue growing faster than operators can resolve it
- Trigger auto-retraining when drift exceeds threshold

This is the "per-shift ROC report" — the single thing the supervisor checks
at the start of every shift to know the system is healthy.

Data sources:
- Detector: per-class counts (from `perception/detector.Detector`)
- Tracker: track lifetimes, track losses
- DriftMonitor: drift event counts
- L1TriageAgent: alert counts, resolution times
- MultitaskPipeline: end-to-end inference latency

The dashboard is JSON-serializable so it can be:
- Pushed to a Prometheus / OpenMetrics exporter
- POSTed to a ROC web dashboard
- Logged to a file for later analysis
"""

from __future__ import annotations

import logging
import statistics
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class ShiftReport:
    """A per-shift report, the supervisor's 8am Monday-morning view.

    All times in UTC. All counts since the shift started. The report is
    JSON-serializable for ingestion by external dashboards.
    """

    shift_start: datetime
    shift_end: datetime
    frames_processed: int = 0
    total_detections: int = 0
    detections_per_class: dict[str, int] = field(default_factory=dict)
    tracks_opened: int = 0
    tracks_closed: int = 0
    alerts_pushed: int = 0
    alerts_routine: int = 0
    alerts_attention: int = 0
    alerts_escalated: int = 0
    alerts_resolved: int = 0
    alerts_unresolved: int = 0
    drift_events_total: int = 0
    drift_events_by_type: dict[str, int] = field(default_factory=dict)
    inference_ms_p50: float = 0.0
    inference_ms_p95: float = 0.0
    inference_ms_p99: float = 0.0
    inference_ms_max: float = 0.0
    retrain_recommended: bool = False
    retrain_reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "shift_start": self.shift_start.isoformat(),
            "shift_end": self.shift_end.isoformat(),
            "frames_processed": self.frames_processed,
            "total_detections": self.total_detections,
            "detections_per_class": self.detections_per_class,
            "tracks": {
                "opened": self.tracks_opened,
                "closed": self.tracks_closed,
            },
            "alerts": {
                "pushed": self.alerts_pushed,
                "routine": self.alerts_routine,
                "attention": self.alerts_attention,
                "escalated": self.alerts_escalated,
                "resolved": self.alerts_resolved,
                "unresolved": self.alerts_unresolved,
            },
            "drift": {
                "total": self.drift_events_total,
                "by_type": self.drift_events_by_type,
            },
            "inference_ms": {
                "p50": round(self.inference_ms_p50, 2),
                "p95": round(self.inference_ms_p95, 2),
                "p99": round(self.inference_ms_p99, 2),
                "max": round(self.inference_ms_max, 2),
            },
            "retrain": {
                "recommended": self.retrain_recommended,
                "reason": self.retrain_reason,
            },
        }


class MonitoringDashboard:
    """Real-time monitoring dashboard. Collects metrics from all layers.

    Add samples via `record_frame(result)` after each MultitaskPipeline.step().
    Get the current snapshot via `snapshot()`. Get a full shift report via
    `shift_report(start, end)`.

    Thread-safe enough for one writer + one reader. Production swap:
    push to Prometheus / StatsD / OpenTelemetry.
    """

    def __init__(
        self,
        retrain_drift_threshold: int = 5,
        retrain_alert_ratio_threshold: float = 0.30,
        history_size: int = 10000,
    ):
        self._retrain_drift_threshold = retrain_drift_threshold
        self._retrain_alert_ratio_threshold = retrain_alert_ratio_threshold
        # Per-frame latency history (for P50/P95/P99)
        self._latencies: deque[float] = deque(maxlen=history_size)
        # Per-class detection counts
        self._class_counts: dict[str, int] = {}
        # Track lifecycle events
        self._tracks_opened = 0
        self._tracks_closed = 0
        # Track IDs seen in the previous frame (for opened/closed accounting)
        self._last_track_ids: set[int] = set()
        # Drift event log
        self._drift_events: deque[dict[str, Any]] = deque(maxlen=history_size)
        # Alert stats
        self._alerts_pushed = 0
        self._alerts_routine = 0
        self._alerts_attention = 0
        self._alerts_escalated = 0
        self._alerts_resolved = 0
        # Frame counter
        self._frames = 0
        self._total_detections = 0
        # Snapshot start (for the running shift)
        self._start = datetime.now(tz=timezone.utc)

    def record_frame(self, result: Any) -> None:
        """Record metrics from one MultitaskPipeline.step() result.

        The result is the FrameResult dataclass. We read fields directly;
        no duck-typing issues if the shape changes.
        """
        self._frames += 1
        self._latencies.append(result.inference_ms)
        for d in result.detections:
            cn = d.get("class_name", "unknown")
            self._class_counts[cn] = self._class_counts.get(cn, 0) + 1
            self._total_detections += 1
        # Track open/close: a "new" track has track_id we haven't seen; a
        # "closed" track is one from the previous frame that's missing now.
        current_track_ids = {t.get("track_id") for t in result.tracks}
        for tid in current_track_ids:
            if tid is not None and tid not in self._last_track_ids:
                self._tracks_opened += 1
        for tid in self._last_track_ids:
            if tid is not None and tid not in current_track_ids:
                self._tracks_closed += 1
        self._last_track_ids = current_track_ids
        # Alerts
        for a in result.alerts:
            self._alerts_pushed += 1
            sev = a.get("severity", "routine")
            if sev == "routine":
                self._alerts_routine += 1
            elif sev == "attention":
                self._alerts_attention += 1
            elif sev == "escalate":
                self._alerts_escalated += 1
        # Drift events
        if result.drift_signals and result.drift_signals.get("active"):
            self._drift_events.append(
                {
                    "frame_idx": result.frame_idx,
                    "drift_type": result.drift_signals.get("drift_type", "unknown"),
                    "severity": result.drift_signals.get("severity", "info"),
                    "message": result.drift_signals.get("message", ""),
                }
            )

    def snapshot(self) -> dict[str, Any]:
        """Return a current snapshot. Cheap; no aggregation of history."""
        return {
            "frames": self._frames,
            "total_detections": self._total_detections,
            "class_counts": dict(self._class_counts),
            "tracks": {"opened": self._tracks_opened, "closed": self._tracks_closed},
            "alerts": {
                "pushed": self._alerts_pushed,
                "routine": self._alerts_routine,
                "attention": self._alerts_attention,
                "escalated": self._alerts_escalated,
            },
            "drift_events_recent": list(self._drift_events)[-10:],
            "inference_ms": self._latency_percentiles(),
        }

    def _latency_percentiles(self) -> dict[str, float]:
        if not self._latencies:
            return {"p50": 0.0, "p95": 0.0, "p99": 0.0, "max": 0.0}
        sorted_lat = sorted(self._latencies)
        n = len(sorted_lat)
        return {
            "p50": sorted_lat[min(n - 1, int(n * 0.50))],
            "p95": sorted_lat[min(n - 1, int(n * 0.95))],
            "p99": sorted_lat[min(n - 1, int(n * 0.99))],
            "max": sorted_lat[-1],
        }

    def _check_retrain_recommendation(self) -> tuple[bool, str]:
        """Decide if retraining is recommended.

        Rules:
        1. >= retrain_drift_threshold drift events → "drift events"
        2. alert ratio (attention + escalated) / pushed > threshold → "alert surge"
        3. P95 latency > some_threshold → skip (handled by predictive_maintenance)
        """
        reasons = []
        if len(self._drift_events) >= self._retrain_drift_threshold:
            reasons.append(
                f"{len(self._drift_events)} drift events observed (threshold "
                f"{self._retrain_drift_threshold})"
            )
        if self._alerts_pushed > 0:
            alert_ratio = (
                self._alerts_attention + self._alerts_escalated
            ) / self._alerts_pushed
            if alert_ratio > self._retrain_alert_ratio_threshold:
                reasons.append(
                    f"alert ratio {alert_ratio:.0%} above threshold "
                    f"{self._retrain_alert_ratio_threshold:.0%}"
                )
        return (bool(reasons), "; ".join(reasons))

    def shift_report(
        self,
        shift_start: Optional[datetime] = None,
        shift_end: Optional[datetime] = None,
    ) -> ShiftReport:
        """Build a per-shift report. Default: from the dashboard's start to now."""
        end = shift_end or datetime.now(tz=timezone.utc)
        start = shift_start or self._start
        pcts = self._latency_percentiles()
        retrain, reason = self._check_retrain_recommendation()
        # Aggregate drift events by type
        by_type: dict[str, int] = {}
        for e in self._drift_events:
            by_type[e["drift_type"]] = by_type.get(e["drift_type"], 0) + 1
        return ShiftReport(
            shift_start=start,
            shift_end=end,
            frames_processed=self._frames,
            total_detections=self._total_detections,
            detections_per_class=dict(self._class_counts),
            tracks_opened=self._tracks_opened,
            tracks_closed=self._tracks_closed,
            alerts_pushed=self._alerts_pushed,
            alerts_routine=self._alerts_routine,
            alerts_attention=self._alerts_attention,
            alerts_escalated=self._alerts_escalated,
            alerts_resolved=self._alerts_resolved,
            alerts_unresolved=self._alerts_pushed - self._alerts_resolved,
            drift_events_total=len(self._drift_events),
            drift_events_by_type=by_type,
            inference_ms_p50=pcts["p50"],
            inference_ms_p95=pcts["p95"],
            inference_ms_p99=pcts["p99"],
            inference_ms_max=pcts["max"],
            retrain_recommended=retrain,
            retrain_reason=reason,
        )

    def reset(self) -> None:
        """Reset all counters. Use at shift change."""
        self._latencies.clear()
        self._class_counts.clear()
        self._tracks_opened = 0
        self._tracks_closed = 0
        self._last_track_ids = set()
        self._drift_events.clear()
        self._alerts_pushed = 0
        self._alerts_routine = 0
        self._alerts_attention = 0
        self._alerts_escalated = 0
        self._alerts_resolved = 0
        self._frames = 0
        self._total_detections = 0
        self._start = datetime.now(tz=timezone.utc)
