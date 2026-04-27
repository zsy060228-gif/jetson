#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path

import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node

import PID


class VthToRosNode(Node):
    def __init__(self) -> None:
        super().__init__("vth2ros")

        self.speed_straight = float(self.declare_parameter("Speed_stright", 0.3).value)
        self.k_curve = float(self.declare_parameter("K_curve", 0.1).value)
        self.max_angular_curve = float(self.declare_parameter("Max_angular_curve", 4.0).value)
        p_gain = float(self.declare_parameter("P", 0.1).value)
        i_gain = float(self.declare_parameter("I", 0.01).value)
        d_gain = float(self.declare_parameter("D", 0.01).value)
        dead_area = float(self.declare_parameter("DeadArea", 4.0).value)
        publish_hz = float(self.declare_parameter("publish_hz", 30.0).value)
        turn_slowdown_gain = float(self.declare_parameter("TurnSlowdownGain", 0.70).value)
        turn_angular_boost = float(self.declare_parameter("TurnAngularBoost", 0.40).value)
        min_speed_ratio = float(self.declare_parameter("MinSpeedRatio", 0.40).value)
        default_note_file = Path(__file__).resolve().with_name("note.txt")
        self.note_file = str(self.declare_parameter("note_file", str(default_note_file)).value)
        cmd_vel_topic = str(self.declare_parameter("cmd_vel_topic", "/cmd_vel_1").value)

        self.pub = self.create_publisher(Twist, cmd_vel_topic, 10)

        self.pid = PID.PID(p_gain, i_gain, d_gain)
        self.pid.SetPoint = 0.0
        self.pid.setSampleTime(0.05)
        self.pid.setWindup(20.0)
        self.pid.setDeadArea(dead_area)
        self.turn_slowdown_gain = max(0.0, turn_slowdown_gain)
        self.turn_angular_boost = max(0.0, turn_angular_boost)
        self.min_speed_ratio = min(1.0, max(0.05, min_speed_ratio))

        period = 1.0 / publish_hz if publish_hz > 0 else 1.0 / 30.0
        self.timer = self.create_timer(period, self._tick)

    def _read_error(self):
        path = Path(self.note_file)
        if not path.exists():
            self.get_logger().warn(f"note file not found: {path}")
            return None
        text = path.read_text(encoding="utf-8", errors="ignore").strip()
        if not text:
            return None
        try:
            parts = [part.strip() for part in text.split(",") if part.strip()]
            deviation = float(parts[0])
            severity = float(parts[1]) if len(parts) >= 2 else 0.0
            return deviation, severity
        except ValueError:
            self.get_logger().warn(f"invalid vtherror in note file: {text}")
            return None

    def _tick(self) -> None:
        state = self._read_error()
        if state is None:
            return
        vtherror, turn_severity = state
        turn_severity = max(0.0, min(1.0, float(turn_severity)))

        msg = Twist()
        speed_ratio = max(
            self.min_speed_ratio,
            1.0 - (turn_severity * self.turn_slowdown_gain),
        )
        msg.linear.x = self.speed_straight * speed_ratio

        self.pid.update(vtherror)
        msg.angular.z = self.pid.output * (1.0 + (turn_severity * self.turn_angular_boost))

        if msg.angular.z > self.max_angular_curve:
            msg.angular.z = self.max_angular_curve
        elif msg.angular.z < -self.max_angular_curve:
            msg.angular.z = -self.max_angular_curve

        if vtherror == -128:
            msg.linear.x = 0.0
            msg.angular.z = 0.0

        self.pub.publish(msg)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = VthToRosNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
