#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
from pathlib import Path

import rclpy
from rclpy.node import Node

from interfaces.msg import TrafficSignDetection, TrafficSignDetections


DEFAULT_SEQUENCE = [
    {"class_name": "red_light", "legacy_sign_id": 5, "confidence": 0.95, "duration_sec": 3.0},
    {"class_name": "", "legacy_sign_id": -1, "confidence": 0.0, "duration_sec": 2.0},
    {"class_name": "green_light", "legacy_sign_id": 3, "confidence": 0.95, "duration_sec": 2.0},
    {"class_name": "", "legacy_sign_id": -1, "confidence": 0.0, "duration_sec": 2.0},
    {"class_name": "turn_left", "legacy_sign_id": 4, "confidence": 0.95, "duration_sec": 2.0},
    {"class_name": "", "legacy_sign_id": -1, "confidence": 0.0, "duration_sec": 2.0}
]


class MockTrafficSignDetector(Node):
    """Publish synthetic traffic sign detections for adapter and rule-chain validation."""

    def __init__(self) -> None:
        super().__init__("mock_traffic_sign_detector")

        default_sequence_path = (
            Path(__file__).resolve().parent.parent / "config" / "mock_traffic_sign_sequence.json"
        )

        self.detections_topic = str(
            self.declare_parameter("detections_topic", "/traffic_sign/detections").value
        )
        self.publish_hz = float(self.declare_parameter("publish_hz", 5.0).value)
        self.loop_sequence = bool(self.declare_parameter("loop_sequence", True).value)
        self.source_name = str(self.declare_parameter("source_name", "mock_traffic_sign_detector").value)
        self.sequence_path = Path(
            str(self.declare_parameter("sequence_path", str(default_sequence_path)).value)
        )

        self.sequence = self._load_sequence(self.sequence_path)
        self.index = 0
        self.step_started_ns = self.get_clock().now().nanoseconds
        self.pub = self.create_publisher(TrafficSignDetections, self.detections_topic, 10)

        period = 1.0 / self.publish_hz if self.publish_hz > 0 else 0.2
        self.create_timer(period, self._tick)

        self.get_logger().info(
            f"mock_traffic_sign_detector started: detections_topic={self.detections_topic}, "
            f"sequence_path={self.sequence_path}, steps={len(self.sequence)}"
        )

    @staticmethod
    def _load_sequence(path: Path) -> list[dict]:
        if not path.exists():
            return list(DEFAULT_SEQUENCE)
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, list) or not payload:
            raise ValueError(f"mock sequence must be a non-empty list: {path}")
        return payload

    def _tick(self) -> None:
        if not self.sequence:
            return

        now_ns = self.get_clock().now().nanoseconds
        step = self.sequence[self.index]
        duration_sec = float(step.get("duration_sec", 1.0))
        if (now_ns - self.step_started_ns) / 1e9 >= duration_sec:
            self.index += 1
            if self.index >= len(self.sequence):
                if self.loop_sequence:
                    self.index = 0
                else:
                    self.get_logger().info("mock sequence finished")
                    rclpy.shutdown()
                    return
            self.step_started_ns = now_ns
            step = self.sequence[self.index]

        msg = TrafficSignDetections()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.source_name

        class_name = str(step.get("class_name", "")).strip()
        if class_name:
            det = TrafficSignDetection()
            det.header.stamp = msg.header.stamp
            det.header.frame_id = self.source_name
            det.class_name = class_name
            det.legacy_sign_id = int(step.get("legacy_sign_id", -1))
            det.confidence = float(step.get("confidence", 0.95))
            det.xmin = int(step.get("xmin", 100))
            det.ymin = int(step.get("ymin", 100))
            det.xmax = int(step.get("xmax", 200))
            det.ymax = int(step.get("ymax", 200))
            det.area_ratio = float(step.get("area_ratio", 0.02))
            det.confirmed = bool(step.get("confirmed", False))
            det.source = self.source_name
            msg.detections = [det]
        else:
            msg.detections = []

        self.pub.publish(msg)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = MockTrafficSignDetector()
    try:
        rclpy.spin(node)
    finally:
        if rclpy.ok():
            node.destroy_node()
            rclpy.shutdown()


if __name__ == "__main__":
    main()
