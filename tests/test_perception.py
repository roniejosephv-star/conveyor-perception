"""Tests for the perception module (Detector + Tracker)."""

from __future__ import annotations

import urllib.request
from pathlib import Path

import cv2
import numpy as np
import pytest

from conveyor_perception.perception.detector import (
    COCO_CLASSES,
    Detector,
    _class_color,
    _parse_yolo_classes,
)
from conveyor_perception.perception.track import Tracker


SAMPLE_IMAGE_URL = "https://ultralytics.com/images/bus.jpg"
SAMPLE_IMAGE_PATH = Path("data/sample/test_bus.jpg")


@pytest.fixture(scope="module", autouse=True)
def _download_sample_image():
    """Download a sample image once for the whole module."""
    SAMPLE_IMAGE_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not SAMPLE_IMAGE_PATH.exists():
        urllib.request.urlretrieve(SAMPLE_IMAGE_URL, SAMPLE_IMAGE_PATH)


class TestCOCOClasses:
    def test_coco_has_80_classes(self):
        assert len(COCO_CLASSES) == 80

    def test_coco_includes_common_objects(self):
        assert "person" in COCO_CLASSES
        assert "car" in COCO_CLASSES
        assert "bicycle" in COCO_CLASSES


class TestClassColor:
    def test_returns_bgr_tuple(self):
        color = _class_color(0)
        assert isinstance(color, tuple)
        assert len(color) == 3
        assert all(0 <= c <= 255 for c in color)

    def test_different_classes_different_colors(self):
        c0 = _class_color(0)
        c1 = _class_color(1)
        # With 37-step hue, class 0 and 1 should differ
        assert c0 != c1


class TestDetectorCOCOPretrained:
    """Integration test using the actual YOLO26s COCO model."""

    def test_load_and_detect(self, _download_sample_image):
        """Load YOLO26s COCO pretrained, detect objects in the bus image."""
        models_dir = Path("models")
        onnx_path = models_dir / "yolo26s.onnx"
        if not onnx_path.exists():
            pytest.skip(f"yolo26s.onnx not found at {onnx_path} — run scripts/download_dataset.py first")
        detector = Detector(model_path=str(onnx_path), class_names=COCO_CLASSES, conf_threshold=0.3)
        frame = cv2.imread(str(SAMPLE_IMAGE_PATH))
        assert frame is not None
        detections = detector.detect(frame)
        # The bus.jpg should have a bus + people
        assert len(detections) > 0
        classes_found = {d.class_name for d in detections}
        assert "bus" in classes_found, f"Expected 'bus' in {classes_found}"
        assert "person" in classes_found, f"Expected 'person' in {classes_found}"

    def test_detect_and_draw_returns_annotated_frame(self, _download_sample_image):
        """Verify the annotated frame has the same shape as input."""
        models_dir = Path("models")
        onnx_path = models_dir / "yolo26s.onnx"
        if not onnx_path.exists():
            pytest.skip(f"yolo26s.onnx not found at {onnx_path}")
        detector = Detector(model_path=str(onnx_path), class_names=COCO_CLASSES, conf_threshold=0.3)
        frame = cv2.imread(str(SAMPLE_IMAGE_PATH))
        detections, annotated = detector.detect_and_draw(frame)
        assert annotated.shape == frame.shape
        assert len(detections) == len(detections)  # sanity


class TestParseYoloClasses:
    def test_parses_list_format(self, tmp_path):
        yaml_path = tmp_path / "data.yaml"
        yaml_path.write_text(
            "names:\n"
            "  - PET\n"
            "  - HDPE\n"
            "  - PVC\n"
        )
        classes = _parse_yolo_classes(yaml_path)
        assert classes == ["PET", "HDPE", "PVC"]

    def test_parses_dict_format(self, tmp_path):
        yaml_path = tmp_path / "data.yaml"
        yaml_path.write_text(
            "names:\n"
            "  0: PET\n"
            "  1: HDPE\n"
            "  2: PVC\n"
        )
        classes = _parse_yolo_classes(yaml_path)
        assert classes == ["PET", "HDPE", "PVC"]


class TestTracker:
    def test_tracker_returns_detections_with_ids(self):
        """Use the IoU fallback path for determinism in tests."""
        from conveyor_perception.core.detection_pipeline import Detection

        tracker = Tracker(use_supervision=False)  # forces IoU fallback
        det1 = Detection(class_id=0, class_name="PET", confidence=0.9, bbox=(0, 0, 100, 100))
        out1 = tracker.update([det1])
        assert out1[0].track_id is not None

        det2 = Detection(class_id=0, class_name="PET", confidence=0.9, bbox=(5, 5, 105, 105))
        out2 = tracker.update([det2])
        # Same object, same ID
        assert out2[0].track_id == out1[0].track_id
