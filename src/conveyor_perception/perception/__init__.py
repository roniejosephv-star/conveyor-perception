"""L1 perception layer for the conveyor perception system.

The L1 perception layer is the first stage of the pipeline. It turns
raw camera frames into structured Detection objects via the Detector
class. Tracking is layered on top via the Tracker wrapper.

Two detector backends are available:
- `Detector`: OpenCV DNN, fastest on CPU, no Ultralytics at runtime
- `UltralyticsDetector`: Ultralytics YOLO, handles segmentation-trained
  models and any YOLO26 variant OpenCV can't parse
"""

from conveyor_perception.perception.detector import COCO_CLASSES, Detector
from conveyor_perception.perception.track import Tracker
from conveyor_perception.perception.ultralytics_detector import UltralyticsDetector

__all__ = [
    "COCO_CLASSES",
    "Detector",
    "Tracker",
    "UltralyticsDetector",
]
