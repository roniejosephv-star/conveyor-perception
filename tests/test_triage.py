"""Tests for the L1 triage module.

Covers:
- Each of the 7 severity rules fires correctly
- Default routine is returned when no rule matches
- L1TriageAgent.push_detection creates + queues + logs alerts
- L1TriageAgent.resolve + escalate work
- Stats are updated correctly
- Decision log is bounded
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from conveyor_perception.triage.severity import (
    DEFAULT_ATTENTION_CLASSES,
    DetectionContext,
    SeverityRules,
    SeverityThresholds,
)
from conveyor_perception.triage.agent import L1TriageAgent


# ---------- SeverityRules tests ----------


def _ctx(**overrides) -> DetectionContext:
    """Build a DetectionContext with sensible defaults."""
    defaults = dict(
        class_name="plastic",
        confidence=0.90,
        bbox_area_ratio=0.10,
        track_id=1,
        track_age_sec=1.0,
        track_motion_px=50.0,
        drift_active=False,
        drift_signals=(),
    )
    defaults.update(overrides)
    return DetectionContext(**defaults)


class TestSeverityRules:
    def setup_method(self):
        self.rules = SeverityRules()

    def test_default_routine(self):
        d = self.rules.decide(_ctx())
        assert d.severity == "routine"
        assert d.rule_fired == "default_routine"

    def test_drift_active_escalates(self):
        d = self.rules.decide(
            _ctx(drift_active=True, drift_signals=("ks_confidence",))
        )
        assert d.severity == "escalate"
        assert d.rule_fired == "drift_active"
        assert "ks_confidence" in d.reason

    def test_low_confidence_escalates(self):
        d = self.rules.decide(_ctx(confidence=0.20))
        assert d.severity == "escalate"
        assert d.rule_fired == "low_confidence_escalate"

    def test_low_confidence_threshold_boundary(self):
        # Exactly at the threshold is NOT escalate (rule is strict-less-than)
        d = self.rules.decide(_ctx(confidence=0.30))
        assert d.rule_fired != "low_confidence_escalate"

    def test_track_stuck_escalates(self):
        d = self.rules.decide(
            _ctx(track_id=42, track_age_sec=10.0, track_motion_px=0.5)
        )
        assert d.severity == "escalate"
        assert d.rule_fired == "track_stuck"
        assert "42" in d.reason

    def test_track_stuck_does_not_fire_when_moving(self):
        # Same age, but moving — should NOT be stuck
        d = self.rules.decide(
            _ctx(track_id=42, track_age_sec=10.0, track_motion_px=50.0)
        )
        assert d.rule_fired != "track_stuck"

    def test_track_stuck_does_not_fire_when_young(self):
        # Short-lived track — not stuck yet
        d = self.rules.decide(
            _ctx(track_id=42, track_age_sec=1.0, track_motion_px=0.5)
        )
        assert d.rule_fired != "track_stuck"

    def test_low_confidence_attention(self):
        d = self.rules.decide(_ctx(confidence=0.45))
        assert d.severity == "attention"
        assert d.rule_fired == "low_confidence_attention"

    def test_attention_class(self):
        d = self.rules.decide(_ctx(class_name="battery", confidence=0.95))
        assert d.severity == "attention"
        assert d.rule_fired == "attention_class"

    def test_attention_class_case_insensitive(self):
        d = self.rules.decide(_ctx(class_name="Battery", confidence=0.95))
        assert d.severity == "attention"
        assert d.rule_fired == "attention_class"

    def test_bbox_too_large(self):
        d = self.rules.decide(_ctx(bbox_area_ratio=0.75, confidence=0.95))
        assert d.severity == "attention"
        assert d.rule_fired == "bbox_too_large"

    def test_bbox_at_threshold_is_routine(self):
        # Exactly at threshold is NOT attention (rule is strict-greater-than)
        d = self.rules.decide(_ctx(bbox_area_ratio=0.60, confidence=0.95))
        assert d.rule_fired != "bbox_too_large"

    def test_priority_drift_wins_over_low_confidence(self):
        # Drift + low confidence → drift rule fires (priority 1)
        d = self.rules.decide(
            _ctx(drift_active=True, drift_signals=("z_class",), confidence=0.10)
        )
        assert d.rule_fired == "drift_active"

    def test_priority_low_confidence_escalate_wins_over_stuck(self):
        # Both qualify; low_confidence_escalate has higher priority
        d = self.rules.decide(
            _ctx(confidence=0.20, track_id=1, track_age_sec=10.0, track_motion_px=0.5)
        )
        assert d.rule_fired == "low_confidence_escalate"

    def test_priority_stuck_wins_over_low_confidence_attention(self):
        d = self.rules.decide(
            _ctx(confidence=0.50, track_id=1, track_age_sec=10.0, track_motion_px=0.5)
        )
        assert d.rule_fired == "track_stuck"

    def test_custom_thresholds(self):
        t = SeverityThresholds(
            confidence_escalate=0.50,
            confidence_attention=0.80,
            bbox_area_attention=0.30,
        )
        r = SeverityRules(t)
        # 0.40 would be routine by default; with custom threshold, escalate
        d = r.decide(_ctx(confidence=0.40))
        assert d.severity == "escalate"
        # 0.75 is routine by default; with custom threshold 0.80, attention
        d = r.decide(_ctx(confidence=0.75))
        assert d.severity == "attention"
        # 0.40 bbox is attention with custom threshold 0.30
        d = r.decide(_ctx(bbox_area_ratio=0.40, confidence=0.95))
        assert d.severity == "attention"

    def test_decide_batch(self):
        contexts = [
            _ctx(confidence=0.95),  # routine
            _ctx(confidence=0.45),  # attention
            _ctx(confidence=0.10),  # escalate
        ]
        decisions = self.rules.decide_batch(contexts)
        assert [d.severity for d in decisions] == ["routine", "attention", "escalate"]

    def test_severity_to_dict(self):
        d = self.rules.decide(_ctx())
        d_dict = d.to_dict()
        assert set(d_dict.keys()) == {"severity", "reason", "rule_fired"}
        assert d_dict["severity"] == "routine"

    def test_default_attention_classes_is_frozenset(self):
        # Make sure we don't accidentally mutate the default
        assert isinstance(DEFAULT_ATTENTION_CLASSES, frozenset)
        assert "battery" in DEFAULT_ATTENTION_CLASSES


# ---------- L1TriageAgent tests ----------


class TestL1TriageAgent:
    def setup_method(self):
        self.agent = L1TriageAgent()

    def test_push_creates_alert_in_queue(self):
        alert = self.agent.push_detection(_ctx())
        pending = self.agent.get_pending(limit=1)
        assert pending[0].id == alert.id
        assert pending[0].class_name == "plastic"
        assert pending[0].severity == "routine"

    def test_push_runs_rules(self):
        alert = self.agent.push_detection(_ctx(confidence=0.10))
        assert alert.severity == "escalate"
        assert alert.metadata["rule_fired"] == "low_confidence_escalate"
        assert "0.10" in alert.metadata["reason"]

    def test_push_increments_stats(self):
        self.agent.push_detection(_ctx())  # routine
        self.agent.push_detection(_ctx(confidence=0.45))  # attention
        self.agent.push_detection(_ctx(confidence=0.10))  # escalate
        s = self.agent.get_stats()
        assert s.pushed == 3
        assert s.routine == 1
        assert s.attention == 1
        assert s.escalated == 1
        assert s.last_decision_rule == "low_confidence_escalate"
        assert s.last_decision_at is not None
        assert s.last_decision_at.tzinfo is not None  # UTC

    def test_alert_id_is_unique(self):
        ids = set()
        for _ in range(50):
            a = self.agent.push_detection(_ctx())
            ids.add(a.id)
        assert len(ids) == 50

    def test_alert_metadata_includes_drift(self):
        a = self.agent.push_detection(
            _ctx(drift_active=True, drift_signals=("z_class", "ks_confidence"))
        )
        assert "drift_signals" in a.metadata
        assert "z_class" in a.metadata["drift_signals"]
        assert "ks_confidence" in a.metadata["drift_signals"]

    def test_alert_metadata_merges_user_metadata(self):
        a = self.agent.push_detection(
            _ctx(metadata={"camera_id": "cam-01", "line_id": "line-A"})
        )
        assert a.metadata["camera_id"] == "cam-01"
        assert a.metadata["line_id"] == "line-A"
        # Default fields should also be present
        assert "rule_fired" in a.metadata
        assert "reason" in a.metadata

    def test_resolve_marks_action(self):
        a = self.agent.push_detection(_ctx())
        ok = self.agent.resolve(a.id, action="false-positive")
        assert ok
        s = self.agent.get_stats()
        assert s.resolved == 1

    def test_resolve_unknown_id_returns_false(self):
        ok = self.agent.resolve("nonexistent-id", action="auto-resolved")
        assert ok is False

    def test_escalate_marks_severity(self):
        a = self.agent.push_detection(_ctx(confidence=0.95))  # routine
        assert a.severity == "routine"
        ok = self.agent.escalate(a.id, reason="supervisor noticed something off")
        assert ok
        # Look it up again
        pending = self.agent.get_pending(limit=100)
        escalated = next(x for x in pending if x.id == a.id)
        assert escalated.severity == "escalate"
        assert escalated.metadata["escalation_reason"] == "supervisor noticed something off"

    def test_get_decision_log_bounded(self):
        # Push more than the log's maxlen (1000)
        for _ in range(1050):
            self.agent.push_detection(_ctx())
        log = self.agent.get_decision_log(limit=10000)
        assert len(log) == 1000  # bounded by deque maxlen
        # And the most recent is the last one pushed
        assert log[-1]["severity"] == "routine"

    def test_get_health_includes_stats(self):
        self.agent.push_detection(_ctx())
        self.agent.push_detection(_ctx(confidence=0.10))
        h = self.agent.get_health()
        assert "triage_stats" in h
        assert h["triage_stats"]["pushed"] == 2
        assert h["triage_stats"]["escalated"] == 1
        assert "queue_size" in h
        assert h["queue_size"] == 2

    def test_metadata_default_is_independent_per_alert(self):
        # Bug check: ensure that mutating one alert's metadata doesn't
        # leak to the next via a shared dict default.
        a1 = self.agent.push_detection(_ctx())
        a1.metadata["extra_field"] = "set-after-push"
        a2 = self.agent.push_detection(_ctx(metadata={"camera_id": "x"}))
        # a2's metadata should not contain a1's extra_field
        assert "extra_field" not in a2.metadata
        # And a2's camera_id should be set
        assert a2.metadata["camera_id"] == "x"

    def test_decision_log_records_correct_fields(self):
        a = self.agent.push_detection(
            _ctx(track_id=99, class_name="metal", confidence=0.88)
        )
        log = self.agent.get_decision_log(limit=10)
        entry = log[0]
        assert entry["alert_id"] == a.id
        assert entry["class_name"] == "metal"
        assert entry["confidence"] == 0.88
        assert entry["track_id"] == 99
        assert entry["severity"] == "routine"
        # ISO 8601 format check (contains 'T' and 'Z' or '+00:00')
        assert "T" in entry["timestamp"]
        assert ("Z" in entry["timestamp"] or "+00:00" in entry["timestamp"])
