#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
from pathlib import Path

import rclpy
from rclpy.node import Node

from interfaces.msg import TrafficRuleInput, TrafficSignDetections


class TrafficSignRecorder(Node):
    def __init__(self) -> None:
        super().__init__("traffic_sign_record")

        default_output = Path.home() / "yahboom_ws" / "log" / "traffic_sign_record.jsonl"
        self.output_path = Path(
            str(self.declare_parameter("output_path", str(default_output)).value)
        )
        self.detections_topic = str(
            self.declare_parameter("detections_topic", "/traffic_sign/detections").value
        )
        self.rule_input_topic = str(
            self.declare_parameter("rule_input_topic", "/traffic_sign/rule_input").value
        )

        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.fp = self.output_path.open("a", encoding="utf-8")

        self.create_subscription(
            TrafficSignDetections,
            self.detections_topic,
            self.detections_callback,
            10,
        )
        self.create_subscription(
            TrafficRuleInput,
            self.rule_input_topic,
            self.rule_input_callback,
            10,
        )

        self.get_logger().info(
            f"traffic_sign_record started: output_path={self.output_path}, "
            f"detections_topic={self.detections_topic}, rule_input_topic={self.rule_input_topic}"
        )

    def detections_callback(self, msg: TrafficSignDetections) -> None:
        record = {
            "type": "detections",
            "stamp": {"sec": int(msg.header.stamp.sec), "nanosec": int(msg.header.stamp.nanosec)},
            "frame_id": msg.header.frame_id,
            "detections": [
                {
                    "frame_id": det.header.frame_id,
                    "class_name": det.class_name,
                    "legacy_sign_id": int(det.legacy_sign_id),
                    "confidence": float(det.confidence),
                    "xmin": int(det.xmin),
                    "ymin": int(det.ymin),
                    "xmax": int(det.xmax),
                    "ymax": int(det.ymax),
                    "area_ratio": float(det.area_ratio),
                    "confirmed": bool(det.confirmed),
                    "source": det.source,
                }
                for det in msg.detections
            ],
        }
        self._write_record(record)

    def rule_input_callback(self, msg: TrafficRuleInput) -> None:
        record = {
            "type": "rule_input",
            "stamp": {"sec": int(msg.header.stamp.sec), "nanosec": int(msg.header.stamp.nanosec)},
            "frame_id": msg.header.frame_id,
            "legacy_sign_id": int(msg.legacy_sign_id),
            "class_name": msg.class_name,
            "confidence": float(msg.confidence),
            "area_ratio": float(msg.area_ratio),
            "confirmed": bool(msg.confirmed),
            "source": msg.source,
        }
        self._write_record(record)

    def _write_record(self, record: dict) -> None:
        self.fp.write(json.dumps(record, ensure_ascii=True) + "\n")
        self.fp.flush()

    def destroy_node(self) -> bool:
        try:
            self.fp.close()
        except Exception:
            pass
        return super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = TrafficSignRecorder()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
