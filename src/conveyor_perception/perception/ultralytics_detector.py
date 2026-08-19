"""Ultralytics-based Detector — for models that OpenCV DNN can't parse.

Why this exists:
- The `Detector` class wraps `DetectionPipeline` which uses OpenCV DNN.
  OpenCV DNN is fast and portable, but it doesn't support every ONNX node.
- YOLO26 models trained on segmentation datasets include Reshape/Attention
  nodes that OpenCV can't infer through.
- Ultralytics YOLO handles the full model graph (including the
  segmentation nodes), so it works on every YOLO26 variant.

When to use this vs the OpenCV DNN Detector:
- Use UltralyticsDetector when:
  - The model was trained on a segmentation dataset
  - The ONNX fails to load in OpenCV DNN with a "Reshape" assertion error
  - You need both bboxes + segmentation masks out
- Use the regular Detector when:
  - The model is plain detection (most pretrained + finetuned models)
  - You need pure ONNX portability (no Ultralytics dependency at runtime)

The interface is identical: `detector.detect(frame)` returns a list of
Detection dataclasses. Swap is one line in your script.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, cast

import numpy as np

from conveyor_perception.core.detection_pipeline import Detection

logger = logging.getLogger(__name__)


class UltralyticsDetector:
    """A drop-in Detector that uses Ultralytics YOLO for inference.

    Same public interface as `conveyor_perception.perception.detector.Detector`
    (the `detect(frame)` method returning list[Detection]). Use this when
    the OpenCV DNN path can't parse your ONNX.
    """

    def __init__(
        self,
        model_path: str,
        class_names: list[str],
        conf_threshold: float = 0.25,
        device: str = "cpu",
        imgsz: int = 640,
    ):
        from ultralytics import YOLO

        if not Path(model_path).exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        self._model = YOLO(model_path)
        self.class_names = class_names
        self.conf_threshold = conf_threshold
        self.device = device
        self.imgsz = imgsz
        logger.info(
            "UltralyticsDetector loaded %s (task=%s, classes=%d, device=%s)",
            model_path,
            self._model.task,
            len(class_names),
            device,
        )

    def detect(self, frame: np.ndarray) -> list[Detection]:
        """Run detection on a single frame. Returns list[Detection]."""
        # Ultralytics' predict() is typed as Iterator | list[Results]. In
        # practice it returns a list when stream=False, but we coerce for
        # type-narrowing (mypy can't index into an Iterator).
        results = list(
            self._model.predict(
                frame,
                imgsz=self.imgsz,
                device=self.device,
                conf=self.conf_threshold,
                verbose=False,
            )
        )
        if not results:
            return []
        # Ultralytics types results[0] as Results | Tensor; the Tensor variant
        # has no .boxes attribute, so we cast to narrow for mypy.
        first = cast(Any, results[0])
        if first.boxes is None:
            return []
        boxes = first.boxes
        detections: list[Detection] = []
        for box in boxes:
            cls_id = int(box.cls.item())
            conf = float(box.conf.item())
            if conf < self.conf_threshold:
                continue
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            class_name = (
                self.class_names[cls_id] if cls_id < len(self.class_names) else f"class_{cls_id}"
            )
            detections.append(
                Detection(
                    class_id=cls_id,
                    class_name=class_name,
                    confidence=conf,
                    bbox=(x1, y1, x2, y2),
                    track_id=None,
                )
            )
        return detections

    def detect_and_draw(
        self, frame: np.ndarray, thickness: int = 2
    ) -> tuple[list[Detection], np.ndarray]:
        """Run detection and return both detections and the annotated frame."""
        # Coerce to list for the same Iterator | list union reason as detect()
        results = list(
            self._model.predict(
                frame,
                imgsz=self.imgsz,
                device=self.device,
                conf=self.conf_threshold,
                verbose=False,
            )
        )
        if not results:
            return [], frame
        # Use Ultralytics' built-in plotting (returns BGR numpy array).
        # Same Tensor | Results union as detect() — cast for mypy.
        first = cast(Any, results[0])
        annotated = first.plot(line_width=thickness)
        dets = self.detect(frame)
        return dets, annotated
