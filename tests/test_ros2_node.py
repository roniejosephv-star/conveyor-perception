"""Tests for the ROS 2 integration.

Tests focus on:
- The mock node (works on any machine, no rclpy required)
- The ConveyorAlert dataclass
- The ROS 2 availability flag
- The real ConveyorNode raises ImportError when rclpy is missing
"""

from __future__ import annotations

import json

import pytest

from conveyor_perception.integration.ros2_node import (
    ConveyorAlert,
    ConveyorNode,
    MockROS2Node,
    ROS2_AVAILABLE,
    build_node_for_test,
)


class TestConveyorAlert:
    def test_to_dict_shape(self):
        a = ConveyorAlert(
            header_frame_id="cam_01",
            header_stamp_sec=12345,
            header_stamp_nanosec=678,
            detections=[{"class": "plastic", "conf": 0.9}],
            severity="attention",
            reason="low conf",
            rule_fired="low_confidence_attention",
        )
        d = a.to_dict()
        assert d["header"]["frame_id"] == "cam_01"
        assert d["header"]["stamp"]["sec"] == 12345
        assert d["header"]["stamp"]["nanosec"] == 678
        assert d["detections"] == [{"class": "plastic", "conf": 0.9}]
        assert d["severity"] == "attention"
        assert d["reason"] == "low conf"
        assert d["rule_fired"] == "low_confidence_attention"

    def test_to_dict_is_json_safe(self):
        a = ConveyorAlert(
            header_frame_id="x",
            header_stamp_sec=0,
            header_stamp_nanosec=0,
        )
        json.dumps(a.to_dict())  # would raise on non-serializable

    def test_defaults(self):
        a = ConveyorAlert(header_frame_id="x", header_stamp_sec=0, header_stamp_nanosec=0)
        assert a.detections == []
        assert a.severity == "routine"
        assert a.reason == ""
        assert a.rule_fired == ""


class TestMockROS2Node:
    def test_construct_with_default_topic(self):
        node = MockROS2Node()
        assert node.image_topic == "/conveyor/camera/image_raw"
        assert list(node.published) == []

    def test_construct_with_custom_topic(self):
        node = MockROS2Node(image_topic="/cam/test")
        assert node.image_topic == "/cam/test"

    def test_subscribe_registers_callback(self):
        node = MockROS2Node()
        calls = []
        node.subscribe(lambda a: calls.append(a))
        a = ConveyorAlert(header_frame_id="x", header_stamp_sec=0, header_stamp_nanosec=0)
        node.publish(a)
        assert len(calls) == 1
        assert calls[0] is a

    def test_subscriber_exception_does_not_break_publish(self):
        node = MockROS2Node()

        def bad_cb(_):
            raise ValueError("test")

        def good_cb(alert):
            good_cb.calls.append(alert)

        good_cb.calls = []
        node.subscribe(bad_cb)
        node.subscribe(good_cb)
        a = ConveyorAlert(header_frame_id="x", header_stamp_sec=0, header_stamp_nanosec=0)
        node.publish(a)
        # The good callback should still have received the alert
        assert len(good_cb.calls) == 1

    def test_publish_records_in_published(self):
        node = MockROS2Node()
        a1 = ConveyorAlert(header_frame_id="a", header_stamp_sec=0, header_stamp_nanosec=0)
        a2 = ConveyorAlert(header_frame_id="b", header_stamp_sec=0, header_stamp_nanosec=0)
        node.publish(a1)
        node.publish(a2)
        published = node.get_published()
        assert len(published) == 2
        assert published[0] is a1
        assert published[1] is a2

    def test_published_is_bounded(self):
        node = MockROS2Node()
        for i in range(1500):
            node.publish(
                ConveyorAlert(
                    header_frame_id=str(i),
                    header_stamp_sec=0,
                    header_stamp_nanosec=0,
                )
            )
        # maxlen=1000
        assert len(node.get_published()) == 1000

    def test_step_with_none_returns_none(self):
        node = MockROS2Node()
        assert node.step(None) is None
        assert node.get_published() == []

    def test_step_with_frame_publishes_stub(self):
        node = MockROS2Node()
        import numpy as np

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        alert = node.step(frame)
        assert alert is not None
        assert alert.severity == "routine"
        assert alert.rule_fired == "mock"
        assert alert.header_frame_id == "conveyor_camera_optical"
        # The published list should have one entry
        assert len(node.get_published()) == 1


class TestRclpyHandling:
    def test_ros2_availability_is_bool(self):
        assert isinstance(ROS2_AVAILABLE, bool)

    def test_conveyor_node_import_does_not_require_rclpy(self):
        # The class should be importable regardless of rclpy
        from conveyor_perception.integration import ConveyorNode as CN
        assert CN is ConveyorNode

    def test_conveyor_node_construction_without_rclpy_raises(self):
        if ROS2_AVAILABLE:
            pytest.skip("rclpy is installed; this test only runs without rclpy")
        with pytest.raises(ImportError, match="rclpy is not installed"):
            ConveyorNode(detector=None)


class TestBuildNodeForTest:
    def test_returns_mock_when_rclpy_missing(self):
        if ROS2_AVAILABLE:
            pytest.skip("rclpy is installed; mock is not used")
        node = build_node_for_test()
        assert isinstance(node, MockROS2Node)

    def test_returns_conveyor_node_when_rclpy_available(self):
        if not ROS2_AVAILABLE:
            pytest.skip("rclpy is not installed")
        # Just check the import path; full construction needs a real detector
        from conveyor_perception.integration import ConveyorNode as CN

        assert CN is not None
