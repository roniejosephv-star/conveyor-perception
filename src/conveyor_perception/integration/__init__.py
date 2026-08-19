"""Integration layer.

Wraps the conveyor perception pipeline in deployment surfaces:
- ROS 2 node for the ROC Ubuntu machines (real ConveyorNode)
- Mock ROS 2 node for dev/test/CI on machines without rclpy
- ConveyorAlert: a portable dataclass that mirrors the ROS 2 custom message

The package is import-safe: even without rclpy, all classes load. The real
ConveyorNode raises ImportError on construction if rclpy is missing.
"""

from conveyor_perception.integration.ros2_node import (
    ROS2_AVAILABLE,
    ConveyorAlert,
    ConveyorNode,
    ImageSource,
    MockROS2Node,
    build_node_for_test,
)

__all__ = [
    "ConveyorAlert",
    "ConveyorNode",
    "ImageSource",
    "MockROS2Node",
    "ROS2_AVAILABLE",
    "build_node_for_test",
]
