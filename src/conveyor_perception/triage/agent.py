"""L1 triage agent — wires detection events into the alert queue.

The agent owns the rules engine + the alert queue. The perception layer
calls `triage.push_detection(...)` with the detection context; the agent
runs the rules, builds an Alert, and pushes it to the queue. The MCP
surface then exposes the queue to the L1 triage human/agent.

This is the "everest-labs-style L1 triage" pattern:
- L0: perception (detector + tracker + drift monitor)
- L1: this agent (rules-based severity + alert queue) — the seam
- L2: human + escalation tools (the ROC operator)

Replacing the rules engine with an LLM is straightforward: the public
API (`push_detection`, `get_pending`, `resolve`) stays the same. The
rules engine is the policy; this agent is the wiring.
"""

from __future__ import annotations

import logging
import uuid
from collections import deque
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from conveyor_perception.core.triage_surface import Alert, InMemoryAlertQueue
from conveyor_perception.triage.severity import (
    DetectionContext,
    SeverityDecision,
    SeverityRules,
)

logger = logging.getLogger(__name__)


@dataclass
class TriageStats:
    """Running counters for the triage agent. The L1 operator can read
    these to see how many alerts are routine vs attention vs escalated.
    """

    pushed: int = 0
    routine: int = 0
    attention: int = 0
    escalated: int = 0
    resolved: int = 0
    last_decision_rule: str = ""
    last_decision_at: datetime | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "pushed": self.pushed,
            "routine": self.routine,
            "attention": self.attention,
            "escalated": self.escalated,
            "resolved": self.resolved,
            "last_decision_rule": self.last_decision_rule,
            "last_decision_at": (
                self.last_decision_at.isoformat() if self.last_decision_at else None
            ),
        }


class L1TriageAgent:
    """The L1 triage agent. Owns the rules + the queue + the stats.

    Thread-safe (the InMemoryAlertQueue is thread-safe; this class adds
    no shared mutable state of its own).
    """

    def __init__(
        self,
        rules: SeverityRules | None = None,
        queue: InMemoryAlertQueue | None = None,
    ):
        self.rules = rules or SeverityRules()
        self.queue = queue or InMemoryAlertQueue()
        self._stats = TriageStats()
        # Local in-memory audit log of recent decisions (separate from queue)
        self._decision_log: deque[dict[str, Any]] = deque(maxlen=1000)

    def push_detection(self, ctx: DetectionContext) -> Alert:
        """Run the rules on a detection context, build an Alert, push it.

        Returns the Alert that was pushed (callers can use it for follow-up
        actions like immediate escalation).
        """
        decision = self.rules.decide(ctx)
        alert = self._build_alert(ctx, decision)
        self.queue.push(alert)
        self._update_stats(decision)
        self._log_decision(alert, decision)
        return alert

    def _build_alert(self, ctx: DetectionContext, decision: SeverityDecision) -> Alert:
        """Build the Alert dataclass from a context + decision."""
        return Alert(
            id=str(uuid.uuid4()),
            timestamp=datetime.now(tz=UTC),
            class_name=ctx.class_name,
            confidence=ctx.confidence,
            track_id=ctx.track_id,
            severity=decision.severity,
            metadata={
                "reason": decision.reason,
                "rule_fired": decision.rule_fired,
                "bbox_area_ratio": ctx.bbox_area_ratio,
                "track_age_sec": ctx.track_age_sec,
                "track_motion_px": ctx.track_motion_px,
                "drift_signals": list(ctx.drift_signals),
                **ctx.metadata,  # user-supplied metadata wins on collision
            },
        )

    def _update_stats(self, decision: SeverityDecision) -> None:
        self._stats.pushed += 1
        if decision.severity == "routine":
            self._stats.routine += 1
        elif decision.severity == "attention":
            self._stats.attention += 1
        elif decision.severity == "escalate":
            self._stats.escalated += 1
        self._stats.last_decision_rule = decision.rule_fired
        self._stats.last_decision_at = datetime.now(tz=UTC)

    def _log_decision(self, alert: Alert, decision: SeverityDecision) -> None:
        self._decision_log.append(
            {
                "alert_id": alert.id,
                "timestamp": alert.timestamp.isoformat(),
                "severity": decision.severity,
                "rule_fired": decision.rule_fired,
                "class_name": alert.class_name,
                "confidence": alert.confidence,
                "track_id": alert.track_id,
            }
        )

    def get_pending(self, limit: int = 10) -> list[Alert]:
        """Return the most recent N alerts, newest first."""
        return self.queue.get_recent(limit)

    def get_decision_log(self, limit: int = 100) -> list[dict[str, Any]]:
        """Return the recent decision log (for audit / debugging)."""
        return list(self._decision_log)[-limit:]

    def get_stats(self) -> TriageStats:
        return self._stats

    def resolve(
        self, alert_id: str, action: str = "auto-resolved"
    ) -> bool:
        """Resolve an alert. Returns True only if the alert existed and was
        resolved. The underlying queue accepts the action either way, so we
        verify presence first to give the caller a meaningful result.
        """
        # Check presence: scan the queue for this id
        if not any(a.id == alert_id for a in self.queue.get_recent(limit=10000)):
            return False
        self.queue.log_resolution(alert_id, action)
        self._stats.resolved += 1
        return True

    def escalate(self, alert_id: str, reason: str) -> bool:
        """Manually escalate an alert (called by the L1 operator).
        Returns True only if the alert existed and was escalated.
        """
        if not any(a.id == alert_id for a in self.queue.get_recent(limit=10000)):
            return False
        self.queue.escalate(alert_id, reason)
        return True

    def get_health(self) -> dict[str, Any]:
        """Surface health: stats + queue + decision log summary."""
        health = self.queue.get_health()
        health["triage_stats"] = self._stats.to_dict()
        return health
