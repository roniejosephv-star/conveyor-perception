"""Tests for the MonitoringDashboard."""

from __future__ import annotations

import numpy as np

from conveyor_perception.core.drift_monitor import DriftMonitor
from conveyor_perception.core.tracking_pipeline import TrackingPipeline
from conveyor_perception.monitoring.dashboard import MonitoringDashboard, ShiftReport
from conveyor_perception.multitask.pipeline import FrameResult, MultitaskPipeline
from conveyor_perception.triage.agent import L1TriageAgent


class FakeDetector:
    def __init__(self, plan):
        self.plan = plan
        self.call_count = 0

    def detect(self, frame, conf_threshold=None):
        idx = min(self.call_count, len(self.plan) - 1)
        self.call_count += 1
        return self.plan[idx] if idx < len(self.plan) else []


def _det(class_id=0, class_name="plastic", conf=0.9, bbox=(10, 10, 50, 50)):
    from conveyor_perception.core.detection_pipeline import Detection

    return Detection(
        class_id=class_id,
        class_name=class_name,
        confidence=conf,
        bbox=bbox,
        track_id=None,
    )


def _build_pipeline(detections_per_frame: int = 1):
    """Build a MultitaskPipeline whose detector returns the same N detections each frame."""

    plan = [[_det(class_name="plastic", conf=0.9)] for _ in range(detections_per_frame)]
    detector = FakeDetector(plan)
    tracker = TrackingPipeline()
    drift = DriftMonitor(baseline_window=10, min_samples_for_drift=5)
    triage = L1TriageAgent()
    return MultitaskPipeline(detector, tracker, drift, triage)


class TestMonitoringDashboard:
    def test_empty_snapshot(self):
        d = MonitoringDashboard()
        s = d.snapshot()
        assert s["frames"] == 0
        assert s["total_detections"] == 0
        assert s["alerts"]["pushed"] == 0

    def test_record_frame_increments_counters(self):
        d = MonitoringDashboard()
        pipeline = _build_pipeline()
        r = pipeline.step(np.zeros((480, 640, 3), dtype=np.uint8))
        d.record_frame(r)
        s = d.snapshot()
        assert s["frames"] == 1
        assert s["total_detections"] == 1
        assert s["class_counts"].get("plastic") == 1
        assert s["alerts"]["pushed"] == 1
        assert s["alerts"]["routine"] == 1

    def test_record_multiple_frames_aggregates(self):
        d = MonitoringDashboard()
        pipeline = _build_pipeline()
        for _ in range(5):
            r = pipeline.step(np.zeros((480, 640, 3), dtype=np.uint8))
            d.record_frame(r)
        s = d.snapshot()
        assert s["frames"] == 5
        assert s["total_detections"] == 5
        assert s["class_counts"]["plastic"] == 5

    def test_latency_percentiles(self):
        d = MonitoringDashboard()
        # Inject fake results with known latencies
        for lat in [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0]:
            r = FrameResult(
                frame_idx=0,
                timestamp=0.0,
                inference_ms=lat,
                detections=[],
                tracks=[],
                drift_signals={"active": False},
                alerts=[],
            )
            d.record_frame(r)
        s = d.snapshot()
        assert s["inference_ms"]["max"] == 100.0
        # P50 should be around 50
        assert 40 <= s["inference_ms"]["p50"] <= 60
        # P95 should be around 95
        assert 80 <= s["inference_ms"]["p95"] <= 100

    def test_drift_events_recorded(self):

        d = MonitoringDashboard()
        r = FrameResult(
            frame_idx=1,
            timestamp=0.0,
            inference_ms=10.0,
            detections=[],
            tracks=[],
            drift_signals={
                "active": True,
                "drift_type": "ks_confidence",
                "severity": "warn",
                "message": "test",
            },
            alerts=[],
        )
        d.record_frame(r)
        s = d.snapshot()
        assert len(s["drift_events_recent"]) == 1
        assert s["drift_events_recent"][0]["drift_type"] == "ks_confidence"

    def test_no_drift_when_inactive(self):
        d = MonitoringDashboard()
        r = FrameResult(
            frame_idx=1,
            timestamp=0.0,
            inference_ms=10.0,
            detections=[],
            tracks=[],
            drift_signals={"active": False},
            alerts=[],
        )
        d.record_frame(r)
        s = d.snapshot()
        assert s["drift_events_recent"] == []

    def test_retrain_recommended_on_many_drift_events(self):
        d = MonitoringDashboard(retrain_drift_threshold=3)
        # Inject 4 drift events
        for i in range(4):
            r = FrameResult(
                frame_idx=i,
                timestamp=0.0,
                inference_ms=10.0,
                detections=[],
                tracks=[],
                drift_signals={
                    "active": True,
                    "drift_type": "z_class",
                    "severity": "warn",
                },
                alerts=[],
            )
            d.record_frame(r)
        report = d.shift_report()
        assert report.retrain_recommended is True
        assert "drift events" in report.retrain_reason.lower() or "drift" in report.retrain_reason.lower()

    def test_retrain_recommended_on_alert_surge(self):
        d = MonitoringDashboard(
            retrain_drift_threshold=1000,  # disable drift rule
            retrain_alert_ratio_threshold=0.20,
        )
        # Inject 10 attention alerts (out of 20) — 50% > 20% threshold
        for i in range(20):
            r = FrameResult(
                frame_idx=i,
                timestamp=0.0,
                inference_ms=10.0,
                detections=[],
                tracks=[],
                drift_signals={"active": False},
                alerts=[{"severity": "attention"} if i < 10 else {"severity": "routine"}],
            )
            d.record_frame(r)
        report = d.shift_report()
        assert report.retrain_recommended is True
        assert "alert" in report.retrain_reason.lower()

    def test_retrain_not_recommended_when_healthy(self):
        d = MonitoringDashboard()
        # Inject 5 frames, all clean
        for i in range(5):
            r = FrameResult(
                frame_idx=i,
                timestamp=0.0,
                inference_ms=10.0,
                detections=[],
                tracks=[],
                drift_signals={"active": False},
                alerts=[{"severity": "routine"}],
            )
            d.record_frame(r)
        report = d.shift_report()
        assert report.retrain_recommended is False
        assert report.retrain_reason == ""

    def test_shift_report_includes_all_fields(self):
        d = MonitoringDashboard()
        pipeline = _build_pipeline(detections_per_frame=3)
        for _ in range(3):
            r = pipeline.step(np.zeros((480, 640, 3), dtype=np.uint8))
            d.record_frame(r)
        report = d.shift_report()
        assert isinstance(report, ShiftReport)
        assert report.frames_processed == 3
        assert report.total_detections >= 3
        assert report.alerts_pushed >= 3
        assert report.shift_start is not None
        assert report.shift_end is not None
        # Times in UTC
        assert report.shift_start.tzinfo is not None
        assert report.shift_end.tzinfo is not None

    def test_to_dict_is_json_safe(self):
        import json

        d = MonitoringDashboard()
        pipeline = _build_pipeline()
        r = pipeline.step(np.zeros((480, 640, 3), dtype=np.uint8))
        d.record_frame(r)
        report = d.shift_report()
        # JSON serialization should not raise
        json.dumps(report.to_dict())

    def test_reset_clears_counters(self):
        d = MonitoringDashboard()
        pipeline = _build_pipeline()
        for _ in range(3):
            r = pipeline.step(np.zeros((480, 640, 3), dtype=np.uint8))
            d.record_frame(r)
        assert d.snapshot()["frames"] == 3
        d.reset()
        s = d.snapshot()
        assert s["frames"] == 0
        assert s["total_detections"] == 0
        assert s["alerts"]["pushed"] == 0

    def test_track_lifecycle_opened_closed(self):
        d = MonitoringDashboard()
        # Frame 1: a new track
        r1 = FrameResult(
            frame_idx=1,
            timestamp=1.0,
            inference_ms=10.0,
            detections=[],
            tracks=[{"track_id": 42}],
            drift_signals={"active": False},
            alerts=[],
        )
        d.record_frame(r1)
        # Frame 2: same track (no new open, no close)
        r2 = FrameResult(
            frame_idx=2,
            timestamp=2.0,
            inference_ms=10.0,
            detections=[],
            tracks=[{"track_id": 42}],
            drift_signals={"active": False},
            alerts=[],
        )
        d.record_frame(r2)
        # Frame 3: track gone (closed)
        r3 = FrameResult(
            frame_idx=3,
            timestamp=3.0,
            inference_ms=10.0,
            detections=[],
            tracks=[],
            drift_signals={"active": False},
            alerts=[],
        )
        d.record_frame(r3)
        s = d.snapshot()
        assert s["tracks"]["opened"] == 1
        assert s["tracks"]["closed"] == 1
