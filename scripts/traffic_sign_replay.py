#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
from pathlib import Path

import rclpy
from rclpy.node import Node

from interfaces.msg import TrafficSignDetection, TrafficSignDetections


class TrafficSignReplay(Node):
    def __init__(self) -> None:
        super().__init__("traffic_sign_replay")

        default_input = Path.home() / "yahboom_ws" / "log" / "traffic_sign_record.jsonl"
        self.input_path = Path(str(self.declare_parameter("input_path", str(default_input)).value))
        self.detections_topic = str(
            self.declare_parameter("detections_topic", "/traffic_sign/detections").value
        )
        self.publish_hz = float(self.declare_parameter("publish_hz", 5.0).value)
        self.loop_replay = bool(self.declare_parameter("loop_replay", False).value)

        if not self.input_path.exists():
            raise FileNotFoundError(f"replay input file not found: {self.input_path}")

        self.records = self._load_records(self.input_path)
        self.index = 0
        self.pub = self.create_publisher(TrafficSignDetections, self.detections_topic, 10)
        period = 1.0 / self.publish_hz if self.publish_hz > 0 else 0.2
        self.create_timer(period, self._tick)

        self.get_logger().info(
            f"traffic_sign_replay started: input_path={self.input_path}, "
            f"detections_topic={self.detections_topic}, records={len(self.records)}"
        )

    @staticmethod
    def _load_records(path: Path) -> list[dict]:
        records = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            payload = json.loads(line)
            if payload.get("type") == "detections":
                records.append(payload)
        if not records:
            raise ValueError(f"no detection records found in {path}")
        return records

    def _tick(self) -> None:
        if self.index >= len(self.records):
            if self.loop_replay:
                self.index = 0
            else:
                self.get_logger().info("traffic_sign_replay finished")
                rclpy.shutdown()
                return

        payload = self.records[self.index]
        msg = TrafficSignDetections()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = str(payload.get("frame_id", "replay"))

        detections = []
        for det in payload.get("detections", []):
            item = TrafficSignDetection()
            item.header.stamp = msg.header.stamp
            item.header.frame_id = str(det.get("frame_id", msg.header.frame_id))
            item.class_name = str(det.get("class_name", ""))
            item.legacy_sign_id = int(det.get("legacy_sign_id", -1))
            item.confidence = float(det.get("confidence", 0.0))
            item.xmin = int(det.get("xmin", 0))
            item.ymin = int(det.get("ymin", 0))
            item.xmax = int(det.get("xmax", 0))
            item.ymax = int(det.get("ymax", 0))
            item.area_ratio = float(det.get("area_ratio", 0.0))
            item.confirmed = bool(det.get("confirmed", False))
            item.source = str(det.get("source", "replay"))
            detections.append(item)

        msg.detections = detections
        self.pub.publish(msg)
        self.index += 1


def main(args=None) -> None:
    rclpy.init(args=args)
    node = TrafficSignReplay()
    try:
        rclpy.spin(node)
    finally:
        if rclpy.ok():
            node.destroy_node()
            rclpy.shutdown()


if __name__ == "__main__":
    main()
