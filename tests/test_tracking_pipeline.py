"""Tests for TrackingPipeline (IoU fallback path; the supervision path is
integration-tested on Day 2)."""

from __future__ import annotations

from conveyor_perception.core.detection_pipeline import Detection
from conveyor_perception.core.tracking_pipeline import TrackingPipeline


def _det(class_id: int, conf: float, bbox: tuple, track_id: int | None = None):
    return Detection(
        class_id=class_id,
        class_name=f"cls_{class_id}",
        confidence=conf,
        bbox=bbox,
        track_id=track_id,
    )


class TestTrackingPipelineFallback:
    """Test the IoU fallback tracker (no supervision dependency).

    Uses `force_fallback=True` to skip the supervision.ByteTrack import
    path, so these tests run in lightweight environments too.
    """

    def test_first_detection_gets_new_id(self):
        t = TrackingPipeline(frame_rate=30, match_thresh=0.3, force_fallback=True)
        out = t.update([_det(0, 0.9, (0, 0, 100, 100))])
        assert out[0].track_id is not None
        assert out[0].track_id >= 1

    def test_same_position_keeps_id(self):
        t = TrackingPipeline(frame_rate=30, match_thresh=0.3, force_fallback=True)
        first = t.update([_det(0, 0.9, (0, 0, 100, 100))])
        first_id = first[0].track_id
        second = t.update([_det(0, 0.9, (5, 5, 105, 105))])  # high IoU
        assert second[0].track_id == first_id

    def test_different_position_gets_new_id(self):
        t = TrackingPipeline(frame_rate=30, match_thresh=0.3, force_fallback=True)
        first = t.update([_det(0, 0.9, (0, 0, 100, 100))])
        second = t.update([_det(0, 0.9, (500, 500, 600, 600))])  # no overlap
        assert second[0].track_id != first[0].track_id

    def test_tiny_detections_filtered(self):
        t = TrackingPipeline(frame_rate=30, min_box_area=1000, force_fallback=True)
        out = t.update([_det(0, 0.9, (0, 0, 5, 5))])  # 25 px^2 < 1000
        assert out == []

    def test_iou_helper(self):
        # Identical boxes → IoU 1.0
        assert TrackingPipeline._iou((0, 0, 100, 100), (0, 0, 100, 100)) == 1.0
        # Disjoint boxes → IoU 0.0
        assert TrackingPipeline._iou((0, 0, 100, 100), (200, 200, 300, 300)) == 0.0
        # Partial overlap
        iou = TrackingPipeline._iou((0, 0, 100, 100), (50, 50, 150, 150))
        # Intersection: 50*50=2500; Union: 10000+10000-2500=17500; IoU=2500/17500≈0.143
        assert abs(iou - 2500 / 17500) < 1e-6
