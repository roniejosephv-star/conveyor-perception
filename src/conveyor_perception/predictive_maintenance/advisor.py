"""Predictive maintenance advisor.

Reads the drift signals from the core DriftMonitor (or any compatible source)
and produces actionable maintenance hints. This is the "EverestLabs aspirational
predictive maintenance" piece: turn a statistical signal into a concrete
action the ROC operator can take.

Design:
- Pure rules engine. No model. Auditable, explainable, fast.
- One signal in → zero or more hints out. A hint is a (severity, action, why)
  triple. The L1 triage layer can route hints the same way it routes alerts.
- Hints are not just "model is bad". They say "model is bad because of X,
  fix by doing Y". That's the difference between observability and actionability.

Sources of drift (in priority order):
1. Latency spike (MAD on recent inference time) → "GPU overloaded or process
   stuck. Check `nvidia-smi` (CUDA) or Activity Monitor (MPS). Reduce concurrent
   streams or batch size."
2. Confidence distribution shift (KS test) → "Model is seeing unfamiliar
   material. Collect 50-100 recent frames, retrain or fine-tune on them."
3. Class count shift (z-score on per-class counts) → "New material class
   appearing. Check the upstream sorting line for new input."
4. Stuck tracks (covered in the triage layer) → "Conveyor jam, check
   downstream camera section."

Each hint has:
- severity: info | warn | critical
- action: short, imperative (e.g., "Reduce batch size from 16 to 8")
- why: longer explanation
- signal: the source signal that triggered this hint
- confidence: 0-1, how sure we are this hint applies
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Iterable, Optional

logger = logging.getLogger(__name__)


@dataclass
class MaintenanceHint:
    """A single maintenance hint. Routes through the same L1 triage as alerts.

    The L1 triage surface treats hints and alerts uniformly. The hint_id is
    a stable identifier for audit/UI purposes.
    """

    hint_id: str
    timestamp: datetime
    severity: str  # "info" | "warn" | "critical"
    action: str
    why: str
    signal: str
    confidence: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "hint_id": self.hint_id,
            "timestamp": self.timestamp.isoformat(),
            "severity": self.severity,
            "action": self.action,
            "why": self.why,
            "signal": self.signal,
            "confidence": round(self.confidence, 3),
            "metadata": self.metadata,
        }


@dataclass
class DriftSignal:
    """A single drift signal from any source (DriftMonitor, manual, or test).

    The advisor is source-agnostic: it just looks at the fields. The
    `extra` dict carries signal-specific context.
    """

    name: str  # e.g., "ks_confidence", "z_class", "mad_latency"
    active: bool
    p_value: Optional[float] = None  # for KS test
    z_score: Optional[float] = None  # for z-score test
    mad_value: Optional[float] = None  # for MAD test
    current: Optional[float] = None
    baseline: Optional[float] = None
    extra: dict[str, Any] = field(default_factory=dict)


class MaintenanceAdvisor:
    """The advisor. Reads DriftSignals, emits MaintenanceHints.

    Stateless and thread-safe. Same input → same output (modulo timestamp
    + hint_id, which are intentionally per-call to avoid collision).
    """

    def __init__(self, hint_id_prefix: str = "hint"):
        self._hint_id_counter = 0
        self._hint_id_prefix = hint_id_prefix

    def _next_id(self) -> str:
        self._hint_id_counter += 1
        return f"{self._hint_id_prefix}-{self._hint_id_counter:06d}"

    def advise(self, signals: Iterable[DriftSignal]) -> list[MaintenanceHint]:
        """Convert a batch of signals into a list of hints.

        Returns a list (not a generator) so the caller can iterate multiple
        times. Inactive signals are ignored.
        """
        hints: list[MaintenanceHint] = []
        for s in signals:
            if not s.active:
                continue
            h = self._signal_to_hint(s)
            if h is not None:
                hints.append(h)
        return hints

    def _signal_to_hint(self, s: DriftSignal) -> MaintenanceHint | None:
        """Route a single signal to its hint template."""
        if s.name == "ks_confidence":
            return self._hint_ks_confidence(s)
        if s.name == "z_class":
            return self._hint_z_class(s)
        if s.name == "mad_latency":
            return self._hint_mad_latency(s)
        # Unknown signal: emit a generic info hint so nothing is silently dropped
        return MaintenanceHint(
            hint_id=self._next_id(),
            timestamp=datetime.now(tz=timezone.utc),
            severity="info",
            action="Review the unknown drift signal in the audit log",
            why=f"Signal '{s.name}' is active but the advisor has no specific guidance",
            signal=s.name,
            confidence=0.30,
            metadata={
                "p_value": s.p_value,
                "z_score": s.z_score,
                "mad_value": s.mad_value,
                "current": s.current,
                "baseline": s.baseline,
            },
        )

    def _hint_ks_confidence(self, s: DriftSignal) -> MaintenanceHint:
        """Confidence distribution has shifted (KS test p-value < 0.05)."""
        # Confidence in the hint = how far below p=0.05 we are
        p = s.p_value if s.p_value is not None else 0.05
        conf = max(0.50, min(0.99, 1.0 - p * 10))  # p=0.005 → 0.95, p=0.05 → 0.50
        return MaintenanceHint(
            hint_id=self._next_id(),
            timestamp=datetime.now(tz=timezone.utc),
            severity="warn",
            action=(
                "Collect 50-100 recent frames; fine-tune the model on them or "
                "trigger a retraining pipeline"
            ),
            why=(
                f"Confidence distribution has shifted (KS p={p:.4f} < 0.05). "
                f"The model is likely seeing unfamiliar material — drift toward "
                f"lower confidence or higher confidence is the early warning."
            ),
            signal=s.name,
            confidence=conf,
            metadata={"p_value": s.p_value},
        )

    def _hint_z_class(self, s: DriftSignal) -> MaintenanceHint:
        """Class count is far from baseline (z-score)."""
        z = s.z_score if s.z_score is not None else 0.0
        severity = "critical" if abs(z) > 3.0 else "warn"
        # Direction: positive z = more objects of this class than usual
        direction = "more" if z > 0 else "fewer"
        class_name = s.extra.get("class_name", "unknown")
        return MaintenanceHint(
            hint_id=self._next_id(),
            timestamp=datetime.now(tz=timezone.utc),
            severity=severity,
            action=(
                f"Check the upstream sorting line — {direction} '{class_name}' than usual. "
                f"Verify it's not a new input stream, contamination, or a sensor shift."
            ),
            why=(
                f"Per-class count for '{class_name}' is at z={z:+.2f} from baseline. "
                f"|z| > 2 means a real shift, not noise. "
                f"Positive z = more than usual, negative z = fewer than usual."
            ),
            signal=s.name,
            confidence=min(0.99, abs(z) / 5.0),
            metadata={"z_score": s.z_score, "class_name": class_name, **s.extra},
        )

    def _hint_mad_latency(self, s: DriftSignal) -> MaintenanceHint:
        """Latency has spiked (MAD test)."""
        mad = s.mad_value if s.mad_value is not None else 0.0
        current = s.current if s.current is not None else 0.0
        baseline = s.baseline if s.baseline is not None else 0.0
        # Severity scales with how many MAD units above baseline
        sev = "critical" if mad > 5.0 else ("warn" if mad > 2.0 else "info")
        action_options = [
            "Reduce batch size by 50%",
            "Drop to a smaller model variant (e.g., yolo26n instead of yolo26s)",
            "Check GPU/CPU utilization — another process may be starving the inference loop",
        ]
        # Pick the most relevant action based on the magnitude
        if mad > 5.0:
            action = action_options[2]  # other process starving
        elif mad > 2.0:
            action = action_options[0]  # reduce batch
        else:
            action = action_options[1]  # smaller model
        return MaintenanceHint(
            hint_id=self._next_id(),
            timestamp=datetime.now(tz=timezone.utc),
            severity=sev,
            action=action,
            why=(
                f"Inference latency is {mad:.1f} MAD units above baseline "
                f"(current={current:.1f}ms, baseline={baseline:.1f}ms). "
                f"Either the system is overloaded or the input distribution has "
                f"shifted to harder frames."
            ),
            signal=s.name,
            confidence=min(0.99, 0.50 + mad / 10.0),
            metadata={
                "mad_value": s.mad_value,
                "current_ms": s.current,
                "baseline_ms": s.baseline,
            },
        )
