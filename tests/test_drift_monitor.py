"""Tests for DriftMonitor.

The KS test is data-driven, so we generate synthetic confidence streams
with known drift to verify the detector fires correctly.
"""

from __future__ import annotations

import time

import pytest

from conveyor_perception.core.drift_monitor import (
    ClassBaseline,
    DriftMonitor,
    DriftAlert,
    ProductionSignal,
)


def _sig(class_id: int, conf: float, latency_ms: float = 10.0, ts: float | None = None):
    return ProductionSignal(
        class_id=class_id,
        confidence=conf,
        inference_time_ms=latency_ms,
        timestamp=ts or time.time(),
    )


class TestDriftMonitor:
    def test_empty_monitor_no_alert(self):
        m = DriftMonitor(min_samples_for_drift=50)
        assert m.check_drift() is None

    def test_no_drift_when_distribution_stable(self):
        m = DriftMonitor(baseline_window=200, min_samples_for_drift=50)
        # Stable distribution: confidence around 0.85 with small noise
        import random

        random.seed(42)
        for _ in range(200):
            m.update(_sig(0, conf=0.85 + random.gauss(0, 0.02)))
        m.reset_count_window()
        alert = m.check_drift()
        assert alert is None  # no drift in stable distribution

    def test_confidence_drift_fires_alert(self):
        m = DriftMonitor(baseline_window=400, min_samples_for_drift=100, drift_threshold=0.05)
        import random

        random.seed(42)
        # Baseline: confidence around 0.85
        for _ in range(200):
            m.update(_sig(0, conf=0.85 + random.gauss(0, 0.02)))
        # Drift: confidence drops to 0.65
        for _ in range(200):
            m.update(_sig(0, conf=0.65 + random.gauss(0, 0.02)))
        alert = m.check_drift()
        assert alert is not None
        assert alert.drift_type == "confidence"
        assert alert.severity in ("warn", "critical")
        assert alert.details["class_id"] == 0
        assert alert.details["p_value"] < 0.05
        # The recent mean should be much lower than the baseline
        assert alert.details["recent_mean"] < alert.details["baseline_mean"]

    def test_count_anomaly_fires_alert(self):
        import random

        random.seed(42)
        m = DriftMonitor(count_z_threshold=2.0, min_samples_for_drift=10)
        # Build up a baseline of ~5 per window with some natural variance.
        # If all baseline counts are identical, std=0 and the z-score is
        # undefined — so we need variance for the detector to fire.
        for _ in range(15):
            for _ in range(random.randint(3, 7)):
                m.update(_sig(0, conf=0.9))
            m.reset_count_window()
        # Suddenly: 25 in this window (huge spike; z = (25-mean)/std, well > 2)
        for _ in range(25):
            m.update(_sig(0, conf=0.9))
        m.check_drift()
        count_alerts = [a for a in m._alerts if a.drift_type == "class_count"]
        assert len(count_alerts) > 0
        assert count_alerts[0].details["class_id"] == 0
        assert count_alerts[0].details["z_score"] > 2.0

    def test_get_health_snapshot(self):
        m = DriftMonitor()
        for i in range(10):
            m.update(_sig(i % 2, conf=0.8))
        health = m.get_health()
        assert "num_classes" in health
        assert "total_signals" in health
        assert "alerts_fired" in health
        assert health["total_signals"] == 10
        assert health["num_classes"] == 2
