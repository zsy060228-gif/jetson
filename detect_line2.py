#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import socket
import struct
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node
from std_msgs.msg import Float32


def recv_all(sock: socket.socket, length: int) -> bytes:
    data = b""
    while len(data) < length:
        chunk = sock.recv(length - len(data))
        if not chunk:
            return b""
        data += chunk
    return data


class LaneKeepBridgeNode(Node):
    """Jetson side client: capture image -> send to laptop -> receive steering angle."""

    def __init__(self) -> None:
        super().__init__("detect_line")

        default_note_file = Path(__file__).resolve().with_name("note.txt")

        self.laptop_ip = str(self.declare_parameter("laptop_ip", "10.139.124.70").value)
        self.port = int(self.declare_parameter("port", 8000).value)
        self.video_device = str(self.declare_parameter("video_device", "/dev/video0").value)
        self.frame_width = int(self.declare_parameter("frame_width", 640).value)
        self.frame_height = int(self.declare_parameter("frame_height", 480).value)
        self.jpeg_quality = int(self.declare_parameter("jpeg_quality", 80).value)
        self.loop_hz = float(self.declare_parameter("loop_hz", 20.0).value)
        self.reconnect_interval_sec = float(
            self.declare_parameter("reconnect_interval_sec", 2.0).value
        )

        self.steering_topic = str(
            self.declare_parameter("steering_topic", "/lane/steering_angle").value
        )
        self.cmd_vel_topic = str(self.declare_parameter("cmd_vel_topic", "/cmd_vel").value)
        self.publish_cmd_vel = bool(self.declare_parameter("publish_cmd_vel", False).value)
        self.linear_speed = float(self.declare_parameter("linear_speed", 0.15).value)
        self.steering_gain = float(self.declare_parameter("steering_gain", 0.01).value)

        self.note_file = str(self.declare_parameter("note_file", str(default_note_file)).value)
        self.write_note_file = bool(self.declare_parameter("write_note_file", True).value)
        self.response_floats = int(self.declare_parameter("response_floats", 2).value)
        self.severity_topic = str(
            self.declare_parameter("severity_topic", "/lane/turn_severity").value
        )

        self.angle_pub = self.create_publisher(Float32, self.steering_topic, 10)
        self.severity_pub = self.create_publisher(Float32, self.severity_topic, 10)
        self.cmd_vel_pub = self.create_publisher(Twist, self.cmd_vel_topic, 10)

        self.client: Optional[socket.socket] = None
        self.cap: Optional[cv2.VideoCapture] = None
        self.last_connect_attempt_time = 0.0
        self.last_camera_error_time = 0.0
        self._note_file_error_logged = False

        self._open_camera()
        period = 1.0 / self.loop_hz if self.loop_hz > 0 else 0.05
        self.timer = self.create_timer(period, self._tick)

        self.get_logger().info(
            f"detect_line started: video={self.video_device}, server={self.laptop_ip}:{self.port}"
        )

    def _log_camera_open_failed(self) -> None:
        now = self.get_clock().now().nanoseconds / 1e9
        if now - self.last_camera_error_time >= 2.0:
            self.last_camera_error_time = now
            self.get_logger().error(f"无法打开摄像头: {self.video_device}")

    @staticmethod
    def _try_parse_video_index(video_device: str) -> Optional[int]:
        if video_device.isdigit():
            return int(video_device)
        if video_device.startswith("/dev/video"):
            suffix = video_device[len("/dev/video") :]
            if suffix.isdigit():
                return int(suffix)
        return None

    def _make_capture(self, source, backend: Optional[int]) -> cv2.VideoCapture:
        if backend is None:
            return cv2.VideoCapture(source)
        return cv2.VideoCapture(source, backend)

    def _open_camera(self) -> None:
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass

        video_index = self._try_parse_video_index(self.video_device)
        candidates = []
        if video_index is not None:
            candidates.append((video_index, cv2.CAP_V4L2, f"index {video_index} + V4L2"))
            candidates.append((video_index, None, f"index {video_index}"))
        candidates.append((self.video_device, cv2.CAP_V4L2, f"path {self.video_device} + V4L2"))
        candidates.append((self.video_device, None, f"path {self.video_device}"))

        self.cap = None
        for source, backend, tag in candidates:
            cap = self._make_capture(source, backend)
            if not cap.isOpened():
                try:
                    cap.release()
                except Exception:
                    pass
                continue

            cap.set(6, cv2.VideoWriter_fourcc("M", "J", "P", "G"))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)
            self.cap = cap
            self.get_logger().info(f"摄像头打开成功: {tag}")
            return

        self._log_camera_open_failed()

    def _close_client(self) -> None:
        if self.client is not None:
            try:
                self.client.close()
            except Exception:
                pass
            self.client = None

    def _maybe_connect(self) -> bool:
        now = self.get_clock().now().nanoseconds / 1e9
        if now - self.last_connect_attempt_time < self.reconnect_interval_sec:
            return False
        self.last_connect_attempt_time = now

        self._close_client()
        try:
            self.client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.client.settimeout(5.0)
            self.client.connect((self.laptop_ip, self.port))
            self.client.settimeout(None)
            self.get_logger().info(f"已连接到服务器 {self.laptop_ip}:{self.port}")
            return True
        except Exception as e:
            self.get_logger().warn(f"连接失败: {e}")
            self._close_client()
            return False

    def _tick(self) -> None:
        if self.cap is None or not self.cap.isOpened():
            self._open_camera()
            return

        if self.client is None and not self._maybe_connect():
            return

        ok, frame = self.cap.read()
        if not ok:
            self.get_logger().warn("摄像头取帧失败")
            return

        try:
            encoded_ok, encoded_img = cv2.imencode(
                ".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality]
            )
            if not encoded_ok:
                self.get_logger().warn("图像编码失败")
                return

            payload = np.asarray(encoded_img).tobytes()
            self.client.sendall(struct.pack(">I", len(payload)))
            self.client.sendall(payload)

            response_len = 8 if self.response_floats >= 2 else 4
            response_data = recv_all(self.client, response_len)
            if not response_data:
                self.get_logger().warn("服务器断开连接")
                self._close_client()
                return

            if self.response_floats >= 2:
                deviation, severity = struct.unpack(">ff", response_data)
            else:
                deviation = struct.unpack(">f", response_data)[0]
                severity = 0.0
            self._publish(deviation, severity)
        except Exception as e:
            self.get_logger().error(f"通信异常: {e}")
            self._close_client()

    def _publish(self, deviation: float, severity: float) -> None:
        if self.write_note_file:
            try:
                Path(self.note_file).write_text(
                    f"{float(deviation)},{float(severity)}", encoding="utf-8"
                )
                self._note_file_error_logged = False
            except Exception as e:
                if not self._note_file_error_logged:
                    self.get_logger().warn(f"写入 note_file 失败: {self.note_file}, {e}")
                    self._note_file_error_logged = True

        angle_msg = Float32()
        angle_msg.data = float(deviation)
        self.angle_pub.publish(angle_msg)

        severity_msg = Float32()
        severity_msg.data = float(severity)
        self.severity_pub.publish(severity_msg)

        if not self.publish_cmd_vel:
            return

        twist = Twist()
        speed_scale = max(0.35, 1.0 - (0.65 * max(0.0, min(1.0, float(severity)))))
        steering_scale = 1.0 + (0.35 * max(0.0, min(1.0, float(severity))))
        twist.linear.x = self.linear_speed * speed_scale
        twist.angular.z = -float(deviation) * self.steering_gain * steering_scale
        self.cmd_vel_pub.publish(twist)

    def destroy_node(self) -> bool:
        try:
            if rclpy.ok():
                stop_msg = Twist()
                self.cmd_vel_pub.publish(stop_msg)
        except Exception:
            pass

        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
        self._close_client()
        return super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = LaneKeepBridgeNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
