"""Smoke tests for UltralyticsDetector.

The UltralyticsDetector wraps Ultralytics YOLO. We don't want this test
to require a real model file (CI doesn't have one), so we mock the
YOLO class via sys.modules injection.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np
import pytest


def _install_fake_ultralytics(monkeypatch, predict_returns=None) -> None:
    """Install a fake `ultralytics` module in sys.modules with a FakeYOLO class."""
    pred_returns = predict_returns or []

    class _FakeBox:
        """One detection box — shim for result.boxes[i] iteration."""

        def __init__(self, item):
            self._item = item  # (cls_id, conf, x1, y1, x2, y2)

        @property
        def cls(self):
            captured = self._item  # capture into closure

            class _Cls:
                def item(self_inner):
                    return captured[0]

            return _Cls()

        @property
        def conf(self):
            captured = self._item

            class _Conf:
                def item(self_inner):
                    return captured[1]

            return _Conf()

        @property
        def xyxy(self):
            captured = self._item

            class _Box4:
                """Shim for one box's 4 xyxy values — supports .tolist()."""

                def __init__(self):
                    self._vals = list(captured[2:])

                def tolist(self_inner):
                    return list(self_inner._vals)

                def __iter__(self_inner):
                    return iter(self_inner._vals)

            class _Xyxy:
                def __getitem__(self_inner, i):
                    return _Box4()

                def tolist(self_inner):
                    return [list(captured[2:])]

            return _Xyxy()

    class FakeBoxes:
        def __init__(self, items):
            self._items = items  # list of (cls_id, conf, x1, y1, x2, y2)

        @property
        def cls(self):
            class ClsProxy:
                def __init__(self, items):
                    self._items = items

                def item(self):
                    return self._items[0][0] if self._items else 0

                def __iter__(self):
                    return iter([t[0] for t in self._items])

            return ClsProxy(self._items)

        @property
        def conf(self):
            class ConfProxy:
                def __init__(self, items):
                    self._items = items

                def item(self):
                    return self._items[0][1] if self._items else 0.0

                def __iter__(self):
                    return iter([t[1] for t in self._items])

            return ConfProxy(self._items)

        @property
        def xyxy(self):
            class XyxyProxy:
                def __init__(self, items):
                    self._items = items

                def __getitem__(self, i):
                    # Real Ultralytics: xyxy[0] is a 1D tensor of 4 values for box 0
                    return list(self._items[i][2:])

                def tolist(self):
                    return [list(t[2:]) for t in self._items]

            return XyxyProxy(self._items)

        def __bool__(self):
            return len(self._items) > 0

        def __iter__(self):
            # Each iterated box is a tiny shim exposing cls/conf/xyxy per item.
            for it in self._items:
                yield _FakeBox(it)

    class FakeResult:
        def __init__(self, items, model_names):
            self.boxes = FakeBoxes(items) if items else None
            self._model_names = model_names

        def plot(self, line_width=2):
            # Return a fake annotated image (just an empty black image)
            return np.zeros((100, 100, 3), dtype=np.uint8)

    class FakeYOLO:
        def __init__(self, path):
            self.path = path
            self.names = {0: "plastic", 1: "metal", 2: "glass", 3: "vinyl"}
            self.task = "detect"  # what the detector logs

        def predict(self, frame, imgsz, device, conf, verbose):
            return [FakeResult(pred_returns, self.names)]

    fake_mod = types.ModuleType("ultralytics")
    fake_mod.YOLO = FakeYOLO
    monkeypatch.setitem(sys.modules, "ultralytics", fake_mod)


class TestUltralyticsDetector:
    def test_init_loads_model(self, tmp_path, monkeypatch):
        from conveyor_perception.perception import UltralyticsDetector

        _install_fake_ultralytics(monkeypatch)
        model_path = tmp_path / "fake.pt"
        model_path.write_bytes(b"0")
        det = UltralyticsDetector(
            model_path=str(model_path),
            class_names=["plastic", "metal", "glass", "vinyl"],
            conf_threshold=0.25,
        )
        assert det.class_names == ["plastic", "metal", "glass", "vinyl"]
        assert det.conf_threshold == 0.25

    def test_init_raises_on_missing_model(self, monkeypatch):
        from conveyor_perception.perception import UltralyticsDetector

        _install_fake_ultralytics(monkeypatch)
        with pytest.raises(FileNotFoundError):
            UltralyticsDetector(
                model_path="/nonexistent/path/model.pt",
                class_names=["x"],
                conf_threshold=0.25,
            )

    def test_detect_returns_list_of_detections(self, tmp_path, monkeypatch):
        from conveyor_perception.perception import UltralyticsDetector

        # 2 detections: 1 plastic, 1 metal
        _install_fake_ultralytics(
            monkeypatch,
            predict_returns=[
                (0, 0.85, 10, 20, 100, 200),
                (1, 0.65, 300, 400, 500, 600),
            ],
        )
        model_path = tmp_path / "fake.pt"
        model_path.write_bytes(b"0")
        det = UltralyticsDetector(
            model_path=str(model_path),
            class_names=["plastic", "metal", "glass", "vinyl"],
            conf_threshold=0.25,
        )
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        detections = det.detect(frame)
        assert len(detections) == 2
        # First detection
        assert detections[0].class_id == 0
        assert detections[0].class_name == "plastic"
        assert detections[0].confidence == 0.85
        assert detections[0].bbox == (10.0, 20.0, 100.0, 200.0)
        # Second detection
        assert detections[1].class_id == 1
        assert detections[1].class_name == "metal"
        assert detections[1].confidence == 0.65

    def test_detect_filters_by_confidence(self, tmp_path, monkeypatch):
        """Detections below the threshold should be filtered out."""
        from conveyor_perception.perception import UltralyticsDetector

        # 2 detections, one with conf=0.10 (below 0.25 threshold)
        _install_fake_ultralytics(
            monkeypatch,
            predict_returns=[
                (0, 0.85, 10, 20, 100, 200),
                (1, 0.10, 300, 400, 500, 600),  # below threshold
            ],
        )
        model_path = tmp_path / "fake.pt"
        model_path.write_bytes(b"0")
        det = UltralyticsDetector(
            model_path=str(model_path),
            class_names=["plastic", "metal"],
            conf_threshold=0.25,
        )
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        detections = det.detect(frame)
        # The low-confidence detection should be filtered
        assert len(detections) == 1
        assert detections[0].class_id == 0

    def test_detect_handles_no_boxes(self, tmp_path, monkeypatch):
        """When the model returns no boxes, detect() returns []."""
        from conveyor_perception.perception import UltralyticsDetector

        _install_fake_ultralytics(monkeypatch, predict_returns=[])
        model_path = tmp_path / "fake.pt"
        model_path.write_bytes(b"0")
        det = UltralyticsDetector(
            model_path=str(model_path),
            class_names=["plastic"],
            conf_threshold=0.25,
        )
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        detections = det.detect(frame)
        assert detections == []

    def test_detect_handles_class_id_out_of_range(self, tmp_path, monkeypatch):
        """If a class_id is past the end of class_names, fall back to 'class_N'."""
        from conveyor_perception.perception import UltralyticsDetector

        # cls_id=99 is out of range
        _install_fake_ultralytics(
            monkeypatch, predict_returns=[(99, 0.5, 0, 0, 50, 50)]
        )
        model_path = tmp_path / "fake.pt"
        model_path.write_bytes(b"0")
        det = UltralyticsDetector(
            model_path=str(model_path),
            class_names=["plastic"],
            conf_threshold=0.25,
        )
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        detections = det.detect(frame)
        assert len(detections) == 1
        assert detections[0].class_name == "class_99"

    def test_detect_and_draw_returns_both(self, tmp_path, monkeypatch):
        from conveyor_perception.perception import UltralyticsDetector

        _install_fake_ultralytics(
            monkeypatch, predict_returns=[(0, 0.9, 10, 20, 100, 200)]
        )
        model_path = tmp_path / "fake.pt"
        model_path.write_bytes(b"0")
        det = UltralyticsDetector(
            model_path=str(model_path),
            class_names=["plastic"],
            conf_threshold=0.25,
        )
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        dets, annotated = det.detect_and_draw(frame)
        assert len(dets) == 1
        assert isinstance(annotated, np.ndarray)
        assert annotated.shape == (100, 100, 3)  # from FakeResult.plot

    def test_satisfies_detector_interface(self, tmp_path, monkeypatch):
        """UltralyticsDetector is a drop-in replacement for Detector.

        Both expose `detect(frame) -> list[Detection]`. The MultitaskPipeline
        uses the same interface, so this test just confirms UltralyticsDetector
        produces the right Detection dataclass shape.
        """
        from conveyor_perception.core.detection_pipeline import Detection
        from conveyor_perception.perception import UltralyticsDetector

        _install_fake_ultralytics(
            monkeypatch, predict_returns=[(0, 0.7, 10, 20, 100, 200)]
        )
        model_path = tmp_path / "fake.pt"
        model_path.write_bytes(b"0")
        det = UltralyticsDetector(
            model_path=str(model_path),
            class_names=["plastic"],
            conf_threshold=0.25,
        )
        dets = det.detect(np.zeros((100, 100, 3), dtype=np.uint8))
        # Each result must be a Detection dataclass
        for d in dets:
            assert isinstance(d, Detection)
            assert isinstance(d.class_id, int)
            assert isinstance(d.class_name, str)
            assert isinstance(d.confidence, float)
            assert isinstance(d.bbox, tuple)
            assert len(d.bbox) == 4
            assert d.track_id is None  # tracking is downstream
