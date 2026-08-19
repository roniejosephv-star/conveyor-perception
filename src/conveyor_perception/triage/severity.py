"""Severity rules for the L1 triage agent.

A deterministic rules engine that turns a detection + context into a severity
(routine / attention / escalate). The L1 triage agent (a human or an LLM
calling this via the MCP surface) uses this as the first pass; ambiguous
cases go to attention for human review, clear bad cases escalate.

This is the seam between raw perception output and the ROC alert queue.
Replacing this with an ML model is straightforward (return the same severity
enum), but rules are auditable, fast, and explainable — which is what the
ROC actually needs.

Rules (in priority order):
1. ESCALATE: drift signals are active → system is misbehaving, halt and review
2. ESCALATE: confidence below 0.30 → likely misclass, needs human confirmation
3. ESCALATE: track is "stuck" (motion < threshold for > N seconds) → conveyor jam
4. ATTENTION: confidence 0.30-0.60 → low but plausible, watch this one
5. ATTENTION: class is in the attention list (e.g., battery, hazardous)
6. ATTENTION: bbox area > 60% of frame → object too close, camera might need check
7. ROUTINE: everything else
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


# Class names that warrant human attention even at high confidence
DEFAULT_ATTENTION_CLASSES: frozenset[str] = frozenset(
    {
        "battery",  # hazardous waste
        "chemicals",  # hazardous
        "medical",  # medical waste
        "paint",  # hazardous
        "syringe",  # medical
        "e-waste",  # electronics
    }
)


@dataclass
class SeverityThresholds:
    """All tunable knobs in one place. The defaults are conservative —
    they err on the side of "attention" (more human review, fewer missed
    bad cases) for an MRF where the cost of a missed misclass is high.
    """

    confidence_escalate: float = 0.30
    confidence_attention: float = 0.60
    bbox_area_attention: float = 0.60  # 60% of frame
    stuck_seconds_escalate: float = 5.0  # object stationary this long = jam
    attention_classes: frozenset[str] = DEFAULT_ATTENTION_CLASSES


@dataclass
class DetectionContext:
    """The full context a single detection carries into triage.

    Built by the perception layer + the tracking layer + the drift monitor.
    The severity engine uses all of these signals.
    """

    class_name: str
    confidence: float
    bbox_area_ratio: float = 0.0  # bbox area / frame area, in [0, 1]
    track_id: int | None = None
    track_age_sec: float = 0.0  # how long this track has existed
    track_motion_px: float = 0.0  # recent motion, in pixels/sec
    drift_active: bool = False  # any drift signal currently active
    drift_signals: tuple[str, ...] = ()
    metadata: dict[str, Any] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.metadata is None:
            self.metadata = {}


@dataclass
class SeverityDecision:
    """The output of the rules engine: a severity, the reason, and which
    rule fired. The reason is what the L1 triage agent displays in its
    review UI; the fired rule is what we log for the audit trail.
    """

    severity: str  # "routine" | "attention" | "escalate"
    reason: str
    rule_fired: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "severity": self.severity,
            "reason": self.reason,
            "rule_fired": self.rule_fired,
        }


class SeverityRules:
    """Deterministic severity rules engine.

    Stateless and thread-safe. Same input → same output, always.
    The 7 rules in priority order are documented in the module docstring.
    """

    def __init__(self, thresholds: SeverityThresholds | None = None):
        self.thresholds = thresholds or SeverityThresholds()

    def decide(self, ctx: DetectionContext) -> SeverityDecision:
        """Run the 7 rules in priority order and return the first match."""
        t = self.thresholds

        # Rule 1: drift active → escalate, system is misbehaving
        if ctx.drift_active:
            signals = ", ".join(ctx.drift_signals) if ctx.drift_signals else "unspecified"
            return SeverityDecision(
                severity="escalate",
                reason=f"Drift signal active ({signals}); halt review and check pipeline",
                rule_fired="drift_active",
            )

        # Rule 2: very low confidence → escalate, likely misclass
        if ctx.confidence < t.confidence_escalate:
            return SeverityDecision(
                severity="escalate",
                reason=f"Confidence {ctx.confidence:.2f} below {t.confidence_escalate}; likely misclass",
                rule_fired="low_confidence_escalate",
            )

        # Rule 3: stuck track → escalate, conveyor jam
        if (
            ctx.track_id is not None
            and ctx.track_age_sec > t.stuck_seconds_escalate
            and ctx.track_motion_px < 1.0  # < 1 px/sec = effectively stationary
        ):
            return SeverityDecision(
                severity="escalate",
                reason=(
                    f"Track {ctx.track_id} stationary for {ctx.track_age_sec:.1f}s "
                    f"(motion {ctx.track_motion_px:.2f} px/s); possible conveyor jam"
                ),
                rule_fired="track_stuck",
            )

        # Rule 4: low confidence but plausible → attention
        if ctx.confidence < t.confidence_attention:
            return SeverityDecision(
                severity="attention",
                reason=(
                    f"Confidence {ctx.confidence:.2f} in attention band "
                    f"[{t.confidence_escalate}, {t.confidence_attention})"
                ),
                rule_fired="low_confidence_attention",
            )

        # Rule 5: hazardous class → attention
        if ctx.class_name.lower() in t.attention_classes:
            return SeverityDecision(
                severity="attention",
                reason=f"Class '{ctx.class_name}' is on the attention list (hazardous material)",
                rule_fired="attention_class",
            )

        # Rule 6: bbox too large → attention
        if ctx.bbox_area_ratio > t.bbox_area_attention:
            return SeverityDecision(
                severity="attention",
                reason=(
                    f"Bbox covers {ctx.bbox_area_ratio:.0%} of frame "
                    f"(>{t.bbox_area_attention:.0%}); object too close"
                ),
                rule_fired="bbox_too_large",
            )

        # Rule 7: default
        return SeverityDecision(
            severity="routine",
            reason=f"Confidence {ctx.confidence:.2f} and class '{ctx.class_name}' both normal",
            rule_fired="default_routine",
        )

    def decide_batch(
        self, contexts: Iterable[DetectionContext]
    ) -> list[SeverityDecision]:
        """Convenience: run decide() on a batch of contexts."""
        return [self.decide(c) for c in contexts]
