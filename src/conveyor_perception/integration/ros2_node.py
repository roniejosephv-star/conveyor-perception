"""ROS 2 integration for the conveyor perception system.

This module wraps the Detector in a rclpy node that:
- Subscribes to sensor_msgs/Image (or compressed)
- Runs the detector on each frame
- Publishes vision_msgs/Detection2DArray (or our custom ConveyorAlert)

Why a custom message:
- The standard `vision_msgs/Detection2DArray` carries bbox + class + score, but
  not the triage severity or maintenance hints. We extend it via a custom
  `conveyor_perception_msgs/ConveyorAlert` that carries:
    - std_msgs/Header
    - vision_msgs/Detection2DArray detections
    - string severity (routine/attention/escalate)
    - string reason
    - string rule_fired
- This means downstream ROC tooling sees the L1 triage decision directly,
  not just raw detections.

The module is import-safe: if rclpy is not installed, all classes are
still importable but the ROS node class raises a clear error. This lets
the framework run on machines without ROS 2 (e.g., M4 Mac dev box) while
deploying cleanly on the ROC Ubuntu machines.

The mock node (`MockROS2Node`) provides a non-ROS test double. Same
interface as the real node, in-memory pub/sub. Tests use this.
"""

from __future__ import annotations

import json
import logging
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol

logger = logging.getLogger(__name__)

# Detect ROS 2 availability. We don't make this a hard import because ROS 2
# requires a system install (rclpy is not on PyPI for the standard Linux
# install path) and we want the framework to work without it.
try:
    import rclpy  # type: ignore  # noqa: F401 — used as a presence sentinel below
    from rclpy.node import Node as _RclpyNode  # type: ignore
    from sensor_msgs.msg import Image as _ROSImage  # type: ignore

    ROS2_AVAILABLE = True
except ImportError:
    ROS2_AVAILABLE = False
    _RclpyNode = object  # type: ignore[assignment,misc]
    _ROSImage = None  # type: ignore[assignment]


@dataclass
class ConveyorAlert:
    """A custom ROS 2 message mirroring conveyor_perception_msgs/ConveyorAlert.

    We define it as a plain dataclass so the same payload works through
    mock mode, real ROS 2, or pure-Python consumers (ROC web UI, scripts).

    Field mapping to ROS 2:
        header: std_msgs/Header
        detections: list of Detection2D
        severity: string (routine/attention/escalate)
        reason: string
        rule_fired: string
    """

    header_frame_id: str
    header_stamp_sec: int
    header_stamp_nanosec: int
    detections: list[dict[str, Any]] = field(default_factory=list)
    severity: str = "routine"
    reason: str = ""
    rule_fired: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "header": {
                "frame_id": self.header_frame_id,
                "stamp": {
                    "sec": self.header_stamp_sec,
                    "nanosec": self.header_stamp_nanosec,
                },
            },
            "detections": self.detections,
            "severity": self.severity,
            "reason": self.reason,
            "rule_fired": self.rule_fired,
        }


class ImageSource(Protocol):
    """Protocol for any source that yields frames.

    Used by both the real ROS 2 node (which calls a sensor callback) and
    the mock node (which pushes synthetic frames). The interface is just
    "give me the next frame as a numpy array".
    """

    def next_frame(self) -> Any | None:
        """Return the next frame as a numpy array, or None if exhausted."""
        ...


# ---------- Mock ROS 2 node (works without rclpy) ----------


class MockROS2Node:
    """A non-ROS test double for ConveyorNode. Same public API, in-memory.

    Use this for:
    - Unit tests on machines without rclpy (M4 Mac dev box)
    - Local development with synthetic frames
    - Demos and CI

    Not for production. The real ConveyorNode is what publishes to a real
    ROS 2 topic on a ROC Ubuntu machine.
    """

    def __init__(self, image_topic: str = "/conveyor/camera/image_raw"):
        self.image_topic = image_topic
        self.published: deque[ConveyorAlert] = deque(maxlen=1000)
        self._subscribers: list[Callable[[ConveyorAlert], None]] = []
        self._frame_count = 0
        logger.info("MockROS2Node created (image_topic=%s)", image_topic)

    def publish(self, alert: ConveyorAlert) -> None:
        """Publish an alert to all in-memory subscribers + record for inspection."""
        self.published.append(alert)
        for sub in self._subscribers:
            try:
                sub(alert)
            except Exception as e:  # pragma: no cover
                logger.warning("subscriber raised: %s", e)

    def subscribe(self, callback: Callable[[ConveyorAlert], None]) -> None:
        """Register an in-memory callback."""
        self._subscribers.append(callback)

    def get_published(self) -> list[ConveyorAlert]:
        return list(self.published)

    def step(self, frame: Any) -> ConveyorAlert | None:
        """Process a single frame and return the alert (if any).

        This is the test/demo entry point. The real ROS 2 node subscribes
        to an image topic; the mock just gets called with a frame directly.
        """
        self._frame_count += 1
        if frame is None:
            return None
        # The actual detection happens here; for the mock we just return a stub.
        # Tests verify the publish path, not the detection.
        alert = ConveyorAlert(
            header_frame_id="conveyor_camera_optical",
            header_stamp_sec=int(time.time()),
            header_stamp_nanosec=0,
            detections=[],
            severity="routine",
            reason="mock_step",
            rule_fired="mock",
        )
        self.publish(alert)
        return alert


# ---------- Real ROS 2 node ----------


class ConveyorNode(_RclpyNode if ROS2_AVAILABLE else object):  # type: ignore[misc]
    """ROS 2 node that runs the conveyor perception pipeline.

    Subscribes to a sensor image topic, runs the Detector, publishes
    ConveyorAlert messages.

    Construction:
        >>> detector = Detector.from_roboflow_dataset(...)
        >>> node = ConveyorNode(
        ...     detector=detector,
        ...     image_topic="/conveyor/camera/image_raw",
        ...     alert_topic="/conveyor/alerts",
        ... )
        >>> rclpy.spin(node)

    Note: imports of this class require rclpy to be installed. The class
    is defined unconditionally so the framework can be imported without
    rclpy; instantiation will raise ImportError if rclpy is missing.
    """

    def __init__(
        self,
        detector: Any,
        image_topic: str = "/conveyor/camera/image_raw",
        alert_topic: str = "/conveyor/alerts",
        confidence_threshold: float = 0.25,
        node_name: str = "conveyor_perception",
    ):
        if not ROS2_AVAILABLE:
            raise ImportError(
                "rclpy is not installed. The ConveyorNode requires ROS 2. "
                "Install on Ubuntu: `sudo apt install ros-${ROS_DISTRO}-rclpy`. "
                "For local dev without ROS, use MockROS2Node instead."
            )
        super().__init__(node_name)
        self._detector = detector
        self._confidence_threshold = confidence_threshold
        self._frame_count = 0
        self._alert_count = 0

        # Use QoS profile that matches typical industrial camera publishers
        from rclpy.qos import QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy  # type: ignore

        qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=5,
        )

        self._image_sub = self.create_subscription(
            _ROSImage, image_topic, self._on_image, qos
        )
        # The real publish type would be ConveyorAlert; we publish a std_msgs/String
        # with JSON payload as a portable fallback when the custom message isn't built.
        from std_msgs.msg import String  # type: ignore

        self._alert_pub = self.create_publisher(String, alert_topic, qos)
        self._alert_topic = alert_topic
        self.get_logger().info(
            f"ConveyorNode ready: image_topic={image_topic}, alert_topic={alert_topic}, "
            f"confidence_threshold={confidence_threshold}"
        )

    def _on_image(self, msg: Any) -> None:
        """Image callback. Decode, detect, publish."""
        self._frame_count += 1
        # Convert ROS Image to OpenCV BGR
        try:
            import cv2  # type: ignore
            import numpy as np  # type: ignore

            h, w = msg.height, msg.width
            # ROS Image data is raw bytes; encoding tells us the layout
            if msg.encoding == "bgr8":
                frame = np.frombuffer(msg.data, dtype=np.uint8).reshape(h, w, 3).copy()
            elif msg.encoding == "rgb8":
                frame = np.frombuffer(msg.data, dtype=np.uint8).reshape(h, w, 3)
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            elif msg.encoding == "mono8":
                frame = np.frombuffer(msg.data, dtype=np.uint8).reshape(h, w)
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            else:
                self.get_logger().warn(f"Unsupported encoding: {msg.encoding}")
                return
        except Exception as e:
            self.get_logger().error(f"Failed to decode image: {e}")
            return

        # Run detection
        try:
            results = self._detector.detect(frame, conf_threshold=self._confidence_threshold)
        except Exception as e:
            self.get_logger().error(f"Detector raised: {e}")
            return

        # Build alert
        alert = ConveyorAlert(
            header_frame_id=msg.header.frame_id,
            header_stamp_sec=msg.header.stamp.sec,
            header_stamp_nanosec=msg.header.stamp.nanosec,
            detections=results.get("detections", []),
            severity=results.get("severity", "routine"),
            reason=results.get("reason", ""),
            rule_fired=results.get("rule_fired", ""),
        )
        # Publish as JSON (portable, no message-gen step)
        from std_msgs.msg import String  # type: ignore

        self._alert_pub.publish(String(data=json.dumps(alert.to_dict())))
        self._alert_count += 1

    def get_stats(self) -> dict[str, int]:
        return {"frames_received": self._frame_count, "alerts_published": self._alert_count}


# ---------- Test entry point ----------


def build_node_for_test(detector: Any = None) -> MockROS2Node:
    """Build the appropriate node for the current environment.

    On machines without rclpy, returns MockROS2Node. On ROS 2 machines,
    returns a ConveyorNode.

    Use this in tests to abstract the env. The detector is optional
    for the mock; required for the real node.
    """
    if not ROS2_AVAILABLE:
        return MockROS2Node()
    return ConveyorNode(detector=detector)
