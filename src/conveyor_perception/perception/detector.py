"""YOLO26 detector — the production entry point for the perception layer.

Wraps the core `DetectionPipeline` with the specific config for this
project: YOLO26s ONNX model, COCO 80 class names by default, swap to
recycling class names when trained on Roboflow data.

Two ways to use:
1. YOLO26s COCO pretrained (default) — 80 classes, downloads on first use
2. Custom-trained ONNX + class_names from the Roboflow dataset

Example:
    >>> from conveyor_perception.perception.detector import Detector
    >>> detector = Detector.from_coco_pretrained()  # downloads if needed
    >>> detections = detector.detect(frame)  # List[Detection]
    >>> for d in detections:
    ...     print(f"{d.class_name}: {d.confidence:.2f}")
"""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np

from ..core.detection_pipeline import Detection, DetectionPipeline

logger = logging.getLogger(__name__)


# COCO 80 class names. Matches the YOLO26s COCO pretrained model.
# If you train on a custom dataset, replace this list with your class names.
COCO_CLASSES: list[str] = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote",
    "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book",
    "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush",
]


class Detector:
    """YOLO26 detector with a clean public API.

    The detector is a thin wrapper around `DetectionPipeline`. It owns the
    model loading + class names config so callers don't have to.

    Attributes:
        pipeline: The underlying DetectionPipeline.
        class_names: List of class names matching the model's output indices.
    """

    def __init__(
        self,
        model_path: str,
        class_names: list[str],
        conf_threshold: float = 0.5,
        iou_threshold: float = 0.4,
        device: str = "cpu",
    ):
        """Initialize the detector.

        Args:
            model_path: Path to the YOLO26 ONNX model.
            class_names: List of class names (length must match the model's nc).
            conf_threshold: Minimum confidence to keep a detection.
            iou_threshold: IoU threshold for NMS (legacy path only).
            device: "cpu" or "cuda".
        """
        self.class_names = class_names
        self.pipeline = DetectionPipeline(
            model_path=model_path,
            class_names=class_names,
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold,
            end_to_end=True,  # YOLO26 NMS-free by default
            device=device,
        )
        self.pipeline.load()

    @classmethod
    def from_coco_pretrained(
        cls,
        conf_threshold: float = 0.3,
        device: str = "cpu",
    ) -> "Detector":
        """Create a detector using YOLO26s COCO pretrained weights.

        Downloads yolo26s.pt + exports to ONNX if not present.
        Use this for quick demos + when you don't have a custom dataset yet.
        """
        models_dir = Path(__file__).resolve().parent.parent.parent.parent / "models"
        onnx_path = models_dir / "yolo26s.onnx"
        pt_path = models_dir / "yolo26s.pt"
        if not onnx_path.exists():
            if not pt_path.exists():
                logger.info("Downloading YOLO26s COCO pretrained...")
                from ultralytics import YOLO  # type: ignore

                models_dir.mkdir(parents=True, exist_ok=True)
                YOLO("yolo26s.pt")
                # Ultralytics saves to cwd; move to models/
                import shutil

                src = Path("yolo26s.pt")
                if src.exists():
                    shutil.move(str(src), str(pt_path))
            logger.info("Exporting to ONNX...")
            from ultralytics import YOLO  # type: ignore

            model = YOLO(str(pt_path))
            model.export(format="onnx", imgsz=640, simplify=True)
            src = Path("yolo26s.onnx")
            if src.exists():
                import shutil

                shutil.move(str(src), str(onnx_path))
        return cls(
            model_path=str(onnx_path),
            class_names=COCO_CLASSES,
            conf_threshold=conf_threshold,
            device=device,
        )

    @classmethod
    def from_roboflow_dataset(
        cls,
        onnx_path: str,
        dataset_meta_path: str,
        conf_threshold: float = 0.3,
        device: str = "cpu",
    ) -> "Detector":
        """Create a detector from a Roboflow-trained model.

        Reads the class names from data.yaml in the dataset directory
        (so the detector knows the recycling classes, not COCO).
        """
        import json

        meta = json.loads(Path(dataset_meta_path).read_text())
        # Find data.yaml inside the dataset
        ds_dir = Path(meta["location"])
        yaml_files = list(ds_dir.rglob("data.yaml"))
        if not yaml_files:
            raise FileNotFoundError(f"No data.yaml found in {ds_dir}")
        # Parse class names from data.yaml
        class_names = _parse_yolo_classes(yaml_files[0])
        return cls(
            model_path=onnx_path,
            class_names=class_names,
            conf_threshold=conf_threshold,
            device=device,
        )

    def detect(self, frame: np.ndarray) -> list[Detection]:
        """Run detection on a single frame.

        Args:
            frame: BGR image as numpy array (H, W, 3).

        Returns:
            List of Detection objects, sorted by confidence descending.
        """
        return self.pipeline.infer(frame)

    def detect_and_draw(
        self, frame: np.ndarray, thickness: int = 2
    ) -> tuple[list[Detection], np.ndarray]:
        """Run detection and return both the detections and the annotated frame.

        Args:
            frame: BGR image as numpy array.
            thickness: Bounding box line thickness in pixels.

        Returns:
            (detections, annotated_frame) — the frame has boxes + labels drawn.
        """
        detections = self.detect(frame)
        annotated = frame.copy()
        for d in detections:
            x1, y1, x2, y2 = [int(round(v)) for v in d.bbox]
            color = _class_color(d.class_id)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)
            label = f"{d.class_name} {d.confidence:.2f}"
            (tw, th), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
            )
            cv2.rectangle(
                annotated, (x1, y1 - th - baseline - 2), (x1 + tw, y1), color, -1
            )
            cv2.putText(
                annotated,
                label,
                (x1, y1 - baseline - 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
        return detections, annotated


def _class_color(class_id: int) -> tuple[int, int, int]:
    """Deterministic color per class (BGR)."""
    # HSV with fixed saturation/value gives distinct colors
    hsv_hue = (class_id * 37) % 180
    hsv = np.array([[[hsv_hue, 200, 230]]], dtype=np.uint8)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
    return int(bgr[0]), int(bgr[1]), int(bgr[2])


def _parse_yolo_classes(data_yaml_path: Path) -> list[str]:
    """Parse YOLO data.yaml to extract class names."""
    import yaml  # type: ignore

    with open(data_yaml_path) as f:
        data = yaml.safe_load(f)
    names = data.get("names", [])
    if isinstance(names, dict):
        # names can be {0: 'PET', 1: 'HDPE', ...}
        return [names[k] for k in sorted(names.keys())]
    return list(names)
