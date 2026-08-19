"""Tests for the predictive maintenance advisor."""

from __future__ import annotations

from conveyor_perception.predictive_maintenance.advisor import (
    DriftSignal,
    MaintenanceAdvisor,
)


class TestMaintenanceAdvisor:
    def setup_method(self):
        self.advisor = MaintenanceAdvisor()

    def test_inactive_signals_are_ignored(self):
        s = DriftSignal(name="ks_confidence", active=False, p_value=0.001)
        hints = self.advisor.advise([s])
        assert hints == []

    def test_ks_confidence_produces_warn(self):
        s = DriftSignal(name="ks_confidence", active=True, p_value=0.001)
        hints = self.advisor.advise([s])
        assert len(hints) == 1
        h = hints[0]
        assert h.severity == "warn"
        assert h.signal == "ks_confidence"
        assert "fine-tune" in h.action.lower() or "retrain" in h.action.lower()
        assert "0.0010" in h.why or "0.001" in h.why
        # Confidence should be high when p is very low
        assert h.confidence > 0.90

    def test_ks_confidence_at_threshold_has_lower_confidence(self):
        s = DriftSignal(name="ks_confidence", active=True, p_value=0.04)
        hints = self.advisor.advise([s])
        h = hints[0]
        # p=0.04 → confidence = max(0.50, min(0.99, 1.0 - 0.4)) = 0.60
        assert h.confidence < 0.90

    def test_z_class_more_than_baseline(self):
        s = DriftSignal(
            name="z_class",
            active=True,
            z_score=2.5,
            extra={"class_name": "plastic"},
        )
        hints = self.advisor.advise([s])
        h = hints[0]
        assert h.severity == "warn"
        assert "plastic" in h.action
        assert "more" in h.action
        assert "2.5" in h.why

    def test_z_class_fewer_than_baseline(self):
        s = DriftSignal(
            name="z_class",
            active=True,
            z_score=-2.5,
            extra={"class_name": "metal"},
        )
        hints = self.advisor.advise([s])
        h = hints[0]
        assert "metal" in h.action
        assert "fewer" in h.action
        assert "-2.5" in h.why

    def test_z_class_critical_above_3(self):
        s = DriftSignal(
            name="z_class",
            active=True,
            z_score=4.0,
            extra={"class_name": "glass"},
        )
        hints = self.advisor.advise([s])
        h = hints[0]
        assert h.severity == "critical"

    def test_z_class_critical_below_negative_3(self):
        s = DriftSignal(
            name="z_class",
            active=True,
            z_score=-3.5,
            extra={"class_name": "vinyl"},
        )
        hints = self.advisor.advise([s])
        h = hints[0]
        assert h.severity == "critical"

    def test_mad_latency_info_below_2(self):
        s = DriftSignal(
            name="mad_latency",
            active=True,
            mad_value=1.5,
            current=85.0,
            baseline=80.0,
        )
        hints = self.advisor.advise([s])
        h = hints[0]
        assert h.severity == "info"
        assert "smaller model" in h.action.lower() or "model" in h.action.lower()

    def test_mad_latency_warn_at_2_to_5(self):
        s = DriftSignal(
            name="mad_latency",
            active=True,
            mad_value=3.0,
            current=120.0,
            baseline=80.0,
        )
        hints = self.advisor.advise([s])
        h = hints[0]
        assert h.severity == "warn"
        assert "batch" in h.action.lower()

    def test_mad_latency_critical_above_5(self):
        s = DriftSignal(
            name="mad_latency",
            active=True,
            mad_value=7.0,
            current=250.0,
            baseline=80.0,
        )
        hints = self.advisor.advise([s])
        h = hints[0]
        assert h.severity == "critical"
        # Should suggest checking GPU/CPU starvation
        assert "process" in h.action.lower() or "gpu" in h.action.lower() or "cpu" in h.action.lower()

    def test_mad_latency_metadata_includes_ms(self):
        s = DriftSignal(
            name="mad_latency",
            active=True,
            mad_value=2.5,
            current=100.0,
            baseline=80.0,
        )
        hints = self.advisor.advise([s])
        h = hints[0]
        assert h.metadata["current_ms"] == 100.0
        assert h.metadata["baseline_ms"] == 80.0
        assert h.metadata["mad_value"] == 2.5

    def test_unknown_signal_emits_generic_info(self):
        s = DriftSignal(name="custom_signal", active=True, current=42.0)
        hints = self.advisor.advise([s])
        h = hints[0]
        assert h.severity == "info"
        assert h.signal == "custom_signal"
        assert h.confidence < 0.50
        assert "audit log" in h.action.lower()

    def test_multiple_signals_produce_multiple_hints(self):
        signals = [
            DriftSignal(name="ks_confidence", active=True, p_value=0.001),
            DriftSignal(name="z_class", active=True, z_score=3.5, extra={"class_name": "x"}),
            DriftSignal(name="mad_latency", active=True, mad_value=2.0, current=100.0, baseline=80.0),
            DriftSignal(name="ks_confidence", active=False, p_value=0.001),  # ignored
        ]
        hints = self.advisor.advise(signals)
        assert len(hints) == 3
        # All hint_ids should be unique
        ids = [h.hint_id for h in hints]
        assert len(set(ids)) == 3

    def test_hint_ids_are_stable_format(self):
        advisor = MaintenanceAdvisor(hint_id_prefix="unit-test")
        signals = [DriftSignal(name="ks_confidence", active=True, p_value=0.001)]
        for _ in range(3):
            hints = advisor.advise(signals)
        # Hint ids should be sequential with the prefix
        assert hints[0].hint_id.startswith("unit-test-")
        # Sequential numbering
        h1 = advisor.advise(signals)[0]
        h2 = advisor.advise(signals)[0]
        assert int(h1.hint_id.split("-")[-1]) < int(h2.hint_id.split("-")[-1])

    def test_to_dict_is_json_safe(self):
        s = DriftSignal(name="ks_confidence", active=True, p_value=0.001)
        hints = self.advisor.advise([s])
        d = hints[0].to_dict()
        # Verify all values are JSON-safe (str, int, float, list, dict, None, bool)
        import json
        json.dumps(d)  # would raise if not serializable
        assert "timestamp" in d
        assert d["severity"] == "warn"

    def test_timestamp_is_utc(self):
        s = DriftSignal(name="ks_confidence", active=True, p_value=0.001)
        hints = self.advisor.advise([s])
        assert hints[0].timestamp.tzinfo is not None
