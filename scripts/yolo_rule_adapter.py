#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
from pathlib import Path
from typing import Optional

import rclpy
from rclpy.node import Node

from interfaces.msg import TrafficRuleInput, TrafficSignDetection, TrafficSignDetections


def load_class_map(path: Path) -> dict[str, int]:
    if not path.exists():
        raise FileNotFoundError(f"class_map_path not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    classes = payload.get("classes", {})
    mapping: dict[str, int] = {}
    for key, value in classes.items():
        try:
            mapping[str(key).strip().lower()] = int(value["legacy_sign_id"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"invalid mapping entry for {key!r}: {value!r}") from exc
    return mapping


class YoloRuleAdapter(Node):
    """Bridge future YOLOv8 detections to the legacy rule input contract."""

    def __init__(self) -> None:
        super().__init__("yolo_rule_adapter")

        default_internal_memory = (
            "/opt/nvidia/deepstream/deepstream-6.0/"
            "sources/DeepStream-Yolo/nvdsinfer_custom_impl_Yolo/internal_memory.txt"
        )
        default_class_map = (
            Path(__file__).resolve().parent.parent / "config" / "yolov8_rule_map.json"
        )

        self.detections_topic = str(
            self.declare_parameter("detections_topic", "/traffic_sign/detections").value
        )
        self.rule_input_topic = str(
            self.declare_parameter("rule_input_topic", "/traffic_sign/rule_input").value
        )
        self.internal_memory_path = Path(
            str(self.declare_parameter("internal_memory_path", default_internal_memory).value)
        )
        self.write_internal_memory = bool(
            self.declare_parameter("write_internal_memory", True).value
        )
        self.min_confidence = float(self.declare_parameter("min_confidence", 0.5).value)
        self.confirm_frames = int(self.declare_parameter("confirm_frames", 2).value)
        self.reset_after_sec = float(self.declare_parameter("reset_after_sec", 0.75).value)
        self.idle_write_hz = float(self.declare_parameter("idle_write_hz", 10.0).value)
        self.source_name = str(self.declare_parameter("source_name", "yolov8_adapter").value)
        self.class_map_path = Path(
            str(self.declare_parameter("class_map_path", str(default_class_map)).value)
        )
        self.class_map = load_class_map(self.class_map_path)

        self.rule_pub = self.create_publisher(TrafficRuleInput, self.rule_input_topic, 10)
        self.create_subscription(
            TrafficSignDetections,
            self.detections_topic,
            self.detections_callback,
            10,
        )

        self.last_sign_id = -1
        self.last_area_ratio = 0.0
        self.last_class_name = ""
        self.streak_sign_id = -1
        self.streak_count = 0
        self.last_seen_ns = 0
        self.last_published_sign_id = -2

        period = 1.0 / self.idle_write_hz if self.idle_write_hz > 0 else 0.1
        self.create_timer(period, self._idle_tick)
        self._write_legacy("", 0.0)

        self.get_logger().info(
            "yolo_rule_adapter started: "
            f"detections_topic={self.detections_topic}, rule_input_topic={self.rule_input_topic}, "
            f"internal_memory_path={self.internal_memory_path}, class_map_path={self.class_map_path}"
        )

    def detections_callback(self, msg: TrafficSignDetections) -> None:
        best = self._select_best_detection(msg)
        now_ns = self.get_clock().now().nanoseconds

        if best is None:
            self._maybe_clear(now_ns)
            return

        sign_id = int(best.legacy_sign_id) if int(best.legacy_sign_id) >= 0 else self._map_class(best)
        if sign_id < 0:
            return

        if sign_id == self.streak_sign_id:
            self.streak_count += 1
        else:
            self.streak_sign_id = sign_id
            self.streak_count = 1

        self.last_seen_ns = now_ns
        self.last_sign_id = sign_id
        self.last_area_ratio = float(best.area_ratio)
        self.last_class_name = best.class_name

        confirmed = self.streak_count >= self.confirm_frames
        if not confirmed:
            return

        self._publish_rule_input(sign_id, best.class_name, float(best.confidence), float(best.area_ratio))
        self._write_legacy(str(sign_id), float(best.area_ratio))

    def _select_best_detection(
        self, msg: TrafficSignDetections
    ) -> Optional[TrafficSignDetection]:
        best: Optional[TrafficSignDetection] = None
        for det in msg.detections:
            if float(det.confidence) < self.min_confidence:
                continue
            mapped_id = int(det.legacy_sign_id) if int(det.legacy_sign_id) >= 0 else self._map_class(det)
            if mapped_id < 0:
                continue
            if best is None or float(det.confidence) > float(best.confidence):
                best = det
        return best

    def _map_class(self, det: TrafficSignDetection) -> int:
        key = det.class_name.strip().lower()
        return self.class_map.get(key, -1)

    def _publish_rule_input(
        self, sign_id: int, class_name: str, confidence: float, area_ratio: float
    ) -> None:
        msg = TrafficRuleInput()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.source_name
        msg.legacy_sign_id = int(sign_id)
        msg.class_name = class_name
        msg.confidence = float(confidence)
        msg.area_ratio = float(area_ratio)
        msg.confirmed = True
        msg.source = self.source_name
        self.rule_pub.publish(msg)

        if sign_id != self.last_published_sign_id:
            self.get_logger().info(
                f"rule input confirmed: sign_id={sign_id}, class_name={class_name}, "
                f"confidence={confidence:.3f}, area_ratio={area_ratio:.5f}"
            )
            self.last_published_sign_id = sign_id

    def _idle_tick(self) -> None:
        now_ns = self.get_clock().now().nanoseconds
        self._maybe_clear(now_ns)

    def _maybe_clear(self, now_ns: int) -> None:
        if self.last_seen_ns <= 0:
            return
        if (now_ns - self.last_seen_ns) / 1e9 < self.reset_after_sec:
            return

        self.last_seen_ns = 0
        self.last_sign_id = -1
        self.last_area_ratio = 0.0
        self.last_class_name = ""
        self.streak_sign_id = -1
        self.streak_count = 0
        self.last_published_sign_id = -2
        self._publish_clear_rule_input()
        self._write_legacy("", 0.0)

    def _publish_clear_rule_input(self) -> None:
        msg = TrafficRuleInput()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.source_name
        msg.legacy_sign_id = -1
        msg.class_name = ""
        msg.confidence = 0.0
        msg.area_ratio = 0.0
        msg.confirmed = False
        msg.source = self.source_name
        self.rule_pub.publish(msg)
        self.get_logger().info("rule input cleared")

    def _write_legacy(self, sign: str, area_ratio: float) -> None:
        if not self.write_internal_memory:
            return
        try:
            self.internal_memory_path.parent.mkdir(parents=True, exist_ok=True)
            self.internal_memory_path.write_text(
                f"{sign}\n{float(area_ratio)}\n",
                encoding="utf-8",
            )
        except Exception as exc:
            self.get_logger().warn(f"failed to write legacy rule input: {exc}")


def main(args=None) -> None:
    rclpy.init(args=args)
    node = YoloRuleAdapter()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
