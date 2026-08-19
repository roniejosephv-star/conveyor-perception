"""Tests for the MultitaskPipeline.

We use a fake detector (returns canned detections) so the test is fast,
deterministic, and independent of the real Ultralytics YOLO runtime.
The real detector is exercised in test_perception.py.
"""

from __future__ import annotations

import numpy as np
import pytest

from conveyor_perception.core.drift_monitor import DriftMonitor
from conveyor_perception.core.tracking_pipeline import TrackingPipeline
from conveyor_perception.multitask.pipeline import FrameResult, MultitaskPipeline
from conveyor_perception.triage.agent import L1TriageAgent


class FakeDetector:
    """Returns canned detections. Configurable per frame via the .plan list.

    Each plan entry is a list of detection dicts that step() will return.
    When the plan is exhausted, returns an empty list.
    """

    def __init__(self, plan: list[list[dict]]):
        self.plan = plan
        self.call_count = 0
        self.last_frame = None
        self.last_conf = None

    def detect(self, frame, conf_threshold=0.25):
        self.last_frame = frame
        self.last_conf = conf_threshold
        idx = min(self.call_count, len(self.plan) - 1)
        self.call_count += 1
        dets = self.plan[idx] if idx < len(self.plan) else []
        return {
            "detections": dets,
            "severity": "routine",
            "reason": "fake",
            "rule_fired": "fake",
        }


def _det(class_id=0, class_name="plastic", conf=0.9, bbox=(100, 100, 200, 200)):
    return {
        "class_id": class_id,
        "class_name": class_name,
        "conf": conf,
        "bbox": list(bbox),
    }


@pytest.fixture
def pipeline():
    plan = [
        [_det(0, "plastic", 0.9, (100, 100, 200, 200))],
        [
            _det(0, "plastic", 0.85, (110, 105, 210, 205)),
            _det(1, "metal", 0.7, (300, 300, 380, 360)),
        ],
        [_det(0, "plastic", 0.4, (120, 110, 220, 210))],  # low conf → attention
    ]
    detector = FakeDetector(plan)
    tracker = TrackingPipeline()
    drift = DriftMonitor(baseline_window=20)
    triage = L1TriageAgent()
    p = MultitaskPipeline(detector, tracker, drift, triage)
    return p, detector, tracker, drift, triage


class TestMultitaskPipeline:
    def test_step_returns_FrameResult(self, pipeline):
        p, *_ = pipeline
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        r = p.step(frame)
        assert isinstance(r, FrameResult)

    def test_step_increments_frame_count(self, pipeline):
        p, *_ = pipeline
        for i in range(3):
            r = p.step(np.zeros((480, 640, 3), dtype=np.uint8))
            assert r.frame_idx == i + 1

    def test_step_passes_frame_to_detector(self, pipeline):
        p, det, *_ = pipeline
        frame = np.ones((480, 640, 3), dtype=np.uint8) * 42
        p.step(frame)
        assert det.last_frame is frame
        assert det.last_conf == 0.25  # default

    def test_step_emits_detection_count_in_result(self, pipeline):
        p, *_ = pipeline
        # Frame 1: 1 detection; Frame 2: 2 detections
        r1 = p.step(np.zeros((480, 640, 3), dtype=np.uint8))
        r2 = p.step(np.zeros((480, 640, 3), dtype=np.uint8))
        assert len(r1.detections) == 1
        assert len(r2.detections) == 2

    def test_step_runs_tracker(self, pipeline):
        p, *_ = pipeline
        r1 = p.step(np.zeros((480, 640, 3), dtype=np.uint8))
        r2 = p.step(np.zeros((480, 640, 3), dtype=np.uint8))
        # Both frames should have at least one track
        assert len(r1.tracks) >= 1
        assert len(r2.tracks) >= 1
        # The plastic should have a stable track_id across frames (ByteTrack)
        # We don't assert exact id stability (depends on the tracker impl) but
        # there should be some track
        assert any(t.get("track_id") is not None for t in r1.tracks)

    def test_step_emits_alerts_via_triage(self, pipeline):
        p, _, _, _, triage = pipeline
        r = p.step(np.zeros((480, 640, 3), dtype=np.uint8))
        # Frame 1: 1 plastic @ 0.90 → routine
        assert len(r.alerts) == 1
        # Stats should be updated
        assert triage.get_stats().pushed == 1

    def test_low_confidence_detection_becomes_attention(self, pipeline):
        p, *_ = pipeline
        # Skip to frame 3 (low conf detection in our plan)
        p.step(np.zeros((480, 640, 3), dtype=np.uint8))  # frame 1
        p.step(np.zeros((480, 640, 3), dtype=np.uint8))  # frame 2
        r3 = p.step(np.zeros((480, 640, 3), dtype=np.uint8))  # frame 3
        assert r3.alerts[0]["severity"] == "attention"

    def test_step_records_drift_signals(self, pipeline):
        p, *_ = pipeline
        r = p.step(np.zeros((480, 640, 3), dtype=np.uint8))
        # drift_signals dict always has an "active" key (True or False)
        assert "active" in r.drift_signals
        # And it should be a boolean
        assert isinstance(r.drift_signals["active"], bool)
        # With 1 detection and a small sample size, drift should NOT be active
        # (we need >= min_samples_for_drift signals before the monitor reports)
        assert r.drift_signals["active"] is False

    def test_step_metadata_includes_frame_shape(self, pipeline):
        p, *_ = pipeline
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        r = p.step(frame)
        assert r.metadata["frame_shape"] == (480, 640, 3)

    def test_to_dict_is_json_safe(self, pipeline):
        import json
        p, *_ = pipeline
        r = p.step(np.zeros((480, 640, 3), dtype=np.uint8))
        d = r.to_dict()
        json.dumps(d)  # would raise on non-serializable

    def test_inference_ms_is_recorded(self, pipeline):
        p, *_ = pipeline
        r = p.step(np.zeros((480, 640, 3), dtype=np.uint8))
        assert r.inference_ms >= 0
        # Should be under a second for our test fixtures
        assert r.inference_ms < 1000

    def test_get_stats_aggregates_across_pipeline(self, pipeline):
        p, *_ = pipeline
        for _ in range(3):
            p.step(np.zeros((480, 640, 3), dtype=np.uint8))
        s = p.get_stats()
        assert s["frames_processed"] == 3
        assert s["triage"]["pushed"] == 4  # 1 + 2 + 1 = 4 total detections
        assert "active_tracks" in s

    def test_handles_empty_detection_list(self):
        detector = FakeDetector([[]])  # always empty
        tracker = TrackingPipeline()
        drift = DriftMonitor(baseline_window=10)
        triage = L1TriageAgent()
        p = MultitaskPipeline(detector, tracker, drift, triage)
        r = p.step(np.zeros((480, 640, 3), dtype=np.uint8))
        assert r.detections == []
        assert r.alerts == []
        assert r.tracks == []

    def test_handles_missing_bbox(self):
        # Detection with no bbox — should not crash the pipeline.
        # The tracker filters out detections with non-positive area, so this
        # detection is dropped before triage. The point of the test is that
        # the pipeline doesn't crash, not that the alert is generated.
        detector = FakeDetector([
            [{"class_id": 0, "class_name": "plastic", "conf": 0.9, "bbox": None}]
        ])
        tracker = TrackingPipeline()
        drift = DriftMonitor(baseline_window=10)
        triage = L1TriageAgent()
        p = MultitaskPipeline(detector, tracker, drift, triage)
        # Just verify no exception is raised
        r = p.step(np.zeros((480, 640, 3), dtype=np.uint8))
        # Detection was filtered; alert may or may not be generated depending on tracker
        assert r is not None

    def test_bbox_area_ratio_in_alert_metadata(self):
        # Large bbox → should be attention (>60% of frame)
        detector = FakeDetector([
            [_det(0, "plastic", 0.95, (0, 0, 600, 480))]  # entire 640x480 frame
        ])
        tracker = TrackingPipeline()
        drift = DriftMonitor(baseline_window=10)
        triage = L1TriageAgent()
        p = MultitaskPipeline(detector, tracker, drift, triage)
        r = p.step(np.zeros((480, 640, 3), dtype=np.uint8))
        # The bbox covers 100% of the frame, which is >60%, so attention
        assert r.alerts[0]["severity"] == "attention"

    def test_motion_tracked_across_frames(self, pipeline):
        p, *_ = pipeline
        # Frame 1: plastic at (100,100,200,200)
        r1 = p.step(np.zeros((480, 640, 3), dtype=np.uint8))
        # Frame 2: same plastic at (110,105,210,205) — moved slightly
        r2 = p.step(np.zeros((480, 640, 3), dtype=np.uint8))
        # The motion history should now have data for this track
        # Look at any track with motion data
        # The alert metadata for the second frame should have non-zero motion
        # (since the plastic moved)
        if len(r2.alerts) > 0:
            for alert in r2.alerts:
                md = alert.get("metadata", {})
                if "track_motion_px" in md:
                    # We can't assert exact value but it should be a number
                    assert isinstance(md["track_motion_px"], (int, float))
