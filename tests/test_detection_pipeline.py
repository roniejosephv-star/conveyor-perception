"""Tests for DetectionPipeline.

These tests use a mock model (no real YOLO file required) so they run
without any external dependencies. The mock simulates YOLO26's NMS-free
output format.

For real end-to-end tests, see tests/integration/ (added on Day 2).
"""

from __future__ import annotations

import numpy as np
import pytest

from conveyor_perception.core.detection_pipeline import (
    Detection,
    DetectionPipeline,
)


def _make_mock_model(
    output_shape: tuple[int, ...],
    tmp_path,
    output_values: list[list[float]] | None = None,
) -> str:
    """Create a tiny ONNX file that returns a fixed output.

    NOTE: For unit tests we don't need a real ONNX. The tests below patch
    the pipeline's _net attribute to a mock that returns the desired output.
    """
    return str(tmp_path / "mock.onnx")


def _patch_net(pipeline: DetectionPipeline, output: np.ndarray) -> None:
    """Replace the pipeline's network with a mock that returns a fixed output.

    OpenCV's net.forward() returns a single np.ndarray for a single-output
    model. The mock matches that contract.
    """

    class MockNet:
        def __init__(self, out):
            self._out = out

        def setInput(self, _blob):
            pass

        def setPreferableBackend(self, _b):
            pass

        def setPreferableTarget(self, _t):
            pass

        def forward(self):
            return self._out

    pipeline._net = MockNet(output)


class TestDetectionPipeline:
    """Test the DetectionPipeline abstraction."""

    def test_preprocess_letterboxes_correctly(self):
        """640x480 → 640x640 padded with 114 on top+bottom."""
        pipeline = DetectionPipeline(
            model_path="dummy.onnx",
            class_names=["PET", "HDPE"],
            input_size=640,
        )
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = pipeline.preprocess(frame)
        assert result.blob.shape == (1, 3, 640, 640)
        assert result.scale == pytest.approx(640 / 640)  # longer side is 640
        assert result.padding == (0.0, 80.0)  # (640-480)/2 = 80
        assert result.original_shape == (480, 640)
        # BGR→RGB swap: the blob's channels should differ if frame has color
        # (we test with zeros so this is trivially true)

    def test_preprocess_preserves_aspect_ratio(self):
        """1280x720 → 640x360 + 140 padding (top+bottom)."""
        pipeline = DetectionPipeline(
            model_path="dummy.onnx",
            class_names=["x"],
            input_size=640,
        )
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        result = pipeline.preprocess(frame)
        assert result.scale == pytest.approx(640 / 1280)  # 0.5
        assert result.padding == (0.0, 140.0)  # (640 - 360) / 2 = 140
        assert result.original_shape == (720, 1280)

    def test_infer_e2e_returns_detections(self):
        """YOLO26 NMS-free output: shape (1, 300, 6)."""
        pipeline = DetectionPipeline(
            model_path="dummy.onnx",
            class_names=["PET_bottle", "HDPE_container"],
            conf_threshold=0.3,
            end_to_end=True,
        )
        # Mock output: 1 detection with high conf, 1 with low conf (filtered)
        output = np.zeros((1, 300, 6), dtype=np.float32)
        # [x1, y1, x2, y2, conf, cls_id] in letterboxed coords (centered)
        output[0, 0] = [100, 100, 200, 200, 0.92, 0]
        output[0, 1] = [300, 300, 400, 400, 0.10, 1]  # below threshold
        _patch_net(pipeline, output)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        detections = pipeline.infer(frame)
        assert len(detections) == 1
        d = detections[0]
        assert d.class_id == 0
        assert d.class_name == "PET_bottle"
        assert d.confidence == pytest.approx(0.92)
        # Bbox scaled back to original (480x640) coords
        assert d.bbox[0] == pytest.approx(100.0, abs=1.0)
        assert d.bbox[1] == pytest.approx(20.0, abs=1.0)  # 100 - 80 padding
        assert d.bbox[2] == pytest.approx(200.0, abs=1.0)
        assert d.bbox[3] == pytest.approx(120.0, abs=1.0)  # 200 - 80 padding

    def test_infer_legacy_with_nms(self):
        """YOLOv8/YOLO11 legacy output: shape (1, 4+nc, 8400)."""
        pipeline = DetectionPipeline(
            model_path="dummy.onnx",
            class_names=["PET", "HDPE"],
            conf_threshold=0.3,
            end_to_end=False,
        )
        # Mock output: 1 detection at the center of the letterboxed image
        nc = 2
        output = np.zeros((1, 4 + nc, 8400), dtype=np.float32)
        # Anchor 0: cx=320, cy=320, w=100, h=100, class_scores=[0.9, 0.1]
        output[0, 0, 0] = 320  # cx
        output[0, 1, 0] = 320  # cy
        output[0, 2, 0] = 100  # w
        output[0, 3, 0] = 100  # h
        output[0, 4, 0] = 0.9
        output[0, 5, 0] = 0.1
        _patch_net(pipeline, output)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        detections = pipeline.infer(frame)
        assert len(detections) == 1
        d = detections[0]
        assert d.class_id == 0
        assert d.class_name == "PET"
        assert d.confidence == pytest.approx(0.9)

    def test_detection_dataclass(self):
        """Detection dataclass fields and defaults."""
        d = Detection(
            class_id=0,
            class_name="PET",
            confidence=0.85,
            bbox=(10, 20, 110, 120),
        )
        assert d.track_id is None  # default
        d_with_id = Detection(
            class_id=0, class_name="PET", confidence=0.85, bbox=(0, 0, 100, 100), track_id=42
        )
        assert d_with_id.track_id == 42
