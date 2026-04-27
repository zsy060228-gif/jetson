#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import socket
import sys
import threading
import time
from http import server
from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import rclpy
import torch
from geometry_msgs.msg import Twist
from rclpy.node import Node
from std_msgs.msg import Float32

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from lane_local_core import (  # noqa: E402
    LaneTracker,
    build_debug_visualization,
    load_unet_model,
    postprocess_lane_mask,
    run_unet_inference,
)


class StreamBuffer:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._condition = threading.Condition(self._lock)
        self._frames: Dict[str, bytes] = {}
        self._versions: Dict[str, int] = {}

    def update(self, name: str, frame, quality: int) -> None:
        ok, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        if not ok:
            return
        with self._condition:
            self._frames[name] = encoded.tobytes()
            self._versions[name] = self._versions.get(name, 0) + 1
            self._condition.notify_all()

    def wait_next(
        self,
        name: str,
        last_version: int,
        timeout: float = 1.0,
    ) -> Tuple[Optional[bytes], int]:
        with self._condition:
            self._condition.wait_for(
                lambda: self._versions.get(name, 0) > last_version,
                timeout=timeout,
            )
            return self._frames.get(name), self._versions.get(name, last_version)


class ThreadedHTTPServer(server.ThreadingHTTPServer):
    daemon_threads = True

    def __init__(
        self,
        server_address,
        handler_class,
        stream_buffer: StreamBuffer,
        *,
        yolo_embed_enabled: bool = False,
        yolo_stream_port: int = 8091,
        yolo_stream_path: str = "/yolo.mjpg",
    ):
        super().__init__(server_address, handler_class)
        self.stream_buffer = stream_buffer
        self.yolo_embed_enabled = yolo_embed_enabled
        self.yolo_stream_port = yolo_stream_port
        self.yolo_stream_path = yolo_stream_path


class StreamHandler(server.BaseHTTPRequestHandler):
    server_version = "detect_line3/1.0"

    STREAM_MAP = {
        "/original.mjpg": "original",
        "/mask.mjpg": "mask",
        "/fit.mjpg": "fit",
    }

    def log_message(self, format: str, *args) -> None:
        return

    def do_GET(self) -> None:
        if self.path == "/":
            self._serve_index()
            return
        if self.path in self.STREAM_MAP:
            self._serve_mjpeg(self.STREAM_MAP[self.path])
            return
        self.send_error(404, "Not Found")

    def _serve_index(self) -> None:
        yolo_card = ""
        if getattr(self.server, "yolo_embed_enabled", False):
            yolo_card = """
    <div class="card">
      <h2>YOLO</h2>
      <img id="yolo-stream" alt="YOLO stream" />
      <p class="hint">YOLO stream served independently. Browser disconnect does not affect local detection.</p>
    </div>
"""

        html_template = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>detect_line3 streams</title>
  <style>
    body { font-family: sans-serif; margin: 16px; background: #111; color: #eee; }
    .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 16px; }
    .card { background: #1b1b1b; padding: 12px; border-radius: 10px; }
    img { width: 100%; height: auto; display: block; background: #000; border-radius: 8px; }
    h2 { margin: 0 0 8px 0; font-size: 18px; }
    .hint { margin: 8px 0 0 0; font-size: 12px; color: #aaa; }
  </style>
</head>
<body>
  <div class="grid">
    <div class="card"><h2>Original</h2><img src="/original.mjpg" /></div>
    <div class="card"><h2>Mask</h2><img src="/mask.mjpg" /></div>
    <div class="card"><h2>Fit</h2><img src="/fit.mjpg" /></div>
    __YOLO_CARD__
  </div>
  <script>
    (function() {
      const yoloImg = document.getElementById("yolo-stream");
      if (!yoloImg) return;
      const host = window.location.hostname || "127.0.0.1";
      const port = __YOLO_PORT__;
      const path = "__YOLO_PATH__";
      yoloImg.src = `http://${host}:${port}${path}`;
    })();
  </script>
</body>
</html>
""".strip()
        html = (
            html_template.replace("__YOLO_CARD__", yolo_card)
            .replace("__YOLO_PORT__", str(int(getattr(self.server, "yolo_stream_port", 8091))))
            .replace("__YOLO_PATH__", str(getattr(self.server, "yolo_stream_path", "/yolo.mjpg")))
            .encode("utf-8")
        )
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(html)))
        self.end_headers()
        self.wfile.write(html)

    def _serve_mjpeg(self, stream_name: str) -> None:
        self.send_response(200)
        self.send_header("Age", "0")
        self.send_header("Cache-Control", "no-cache, private")
        self.send_header("Pragma", "no-cache")
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.end_headers()

        version = 0
        while True:
            frame, version = self.server.stream_buffer.wait_next(stream_name, version, timeout=1.0)
            if frame is None:
                continue
            try:
                self.wfile.write(b"--frame\r\n")
                self.send_header("Content-Type", "image/jpeg")
                self.send_header("Content-Length", str(len(frame)))
                self.end_headers()
                self.wfile.write(frame)
                self.wfile.write(b"\r\n")
            except (BrokenPipeError, ConnectionResetError, socket.timeout):
                break


class LaneKeepLocalNode(Node):
    """Jetson side local lane node: local inference + local streaming server."""

    def __init__(self) -> None:
        super().__init__("detect_line3")

        default_note_file = SCRIPT_DIR / "note.txt"
        default_weights = SCRIPT_DIR / "params" / "min_loss.pth"

        self.video_device = str(self.declare_parameter("video_device", "/dev/video0").value)
        self.frame_width = int(self.declare_parameter("frame_width", 640).value)
        self.frame_height = int(self.declare_parameter("frame_height", 480).value)
        self.loop_hz = float(self.declare_parameter("loop_hz", 20.0).value)
        self.note_file = str(self.declare_parameter("note_file", str(default_note_file)).value)
        self.write_note_file = bool(self.declare_parameter("write_note_file", True).value)
        self.weights_path = str(self.declare_parameter("weights_path", str(default_weights)).value)

        self.steering_topic = str(
            self.declare_parameter("steering_topic", "/lane/steering_angle").value
        )
        self.severity_topic = str(
            self.declare_parameter("severity_topic", "/lane/turn_severity").value
        )
        self.cmd_vel_topic = str(self.declare_parameter("cmd_vel_topic", "/cmd_vel").value)
        self.publish_cmd_vel = bool(self.declare_parameter("publish_cmd_vel", False).value)
        self.linear_speed = float(self.declare_parameter("linear_speed", 0.15).value)
        self.steering_gain = float(self.declare_parameter("steering_gain", 0.01).value)

        self.stream_host = str(self.declare_parameter("stream_host", "0.0.0.0").value)
        self.stream_port = int(self.declare_parameter("stream_port", 8090).value)
        self.stream_quality = int(self.declare_parameter("stream_quality", 75).value)
        self.enable_stream_server = bool(
            self.declare_parameter("enable_stream_server", True).value
        )
        self.yolo_embed_enabled = bool(
            self.declare_parameter("yolo_embed_enabled", True).value
        )
        self.yolo_stream_port = int(self.declare_parameter("yolo_stream_port", 8091).value)
        self.yolo_stream_path = str(
            self.declare_parameter("yolo_stream_path", "/yolo.mjpg").value
        )

        self.angle_pub = self.create_publisher(Float32, self.steering_topic, 10)
        self.severity_pub = self.create_publisher(Float32, self.severity_topic, 10)
        self.cmd_vel_pub = self.create_publisher(Twist, self.cmd_vel_topic, 10)

        self.cap: Optional[cv2.VideoCapture] = None
        self.last_camera_error_time = 0.0
        self.stream_buffer = StreamBuffer()
        self.http_server: Optional[ThreadedHTTPServer] = None
        self.http_thread: Optional[threading.Thread] = None

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net = load_unet_model(self.device, weights_path=self.weights_path)
        self.tracker = LaneTracker()

        self._open_camera()
        if self.enable_stream_server:
            self._start_stream_server()

        period = 1.0 / self.loop_hz if self.loop_hz > 0 else 0.05
        self.timer = self.create_timer(period, self._tick)
        self.get_logger().info(
            f"detect_line3 started: video={self.video_device}, stream={self._stream_urls_text()}"
        )

    def _stream_urls_text(self) -> str:
        hosts = self._discover_stream_hosts()
        if not hosts:
            return f"http://127.0.0.1:{self.stream_port}/"
        return ", ".join(f"http://{host}:{self.stream_port}/" for host in hosts)

    def _discover_stream_hosts(self) -> list[str]:
        if self.stream_host not in ("", "0.0.0.0", "::"):
            return [self.stream_host]

        hosts: list[str] = []
        try:
            hostname = socket.gethostname()
            infos = socket.getaddrinfo(hostname, None, family=socket.AF_INET, type=socket.SOCK_STREAM)
            for info in infos:
                addr = info[4][0]
                if addr.startswith("127."):
                    continue
                if addr not in hosts:
                    hosts.append(addr)
        except OSError:
            pass

        try:
            probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            probe.connect(("8.8.8.8", 80))
            addr = probe.getsockname()[0]
            probe.close()
            if addr and not addr.startswith("127.") and addr not in hosts:
                hosts.insert(0, addr)
        except OSError:
            pass

        return hosts

    def _start_stream_server(self) -> None:
        self.http_server = ThreadedHTTPServer(
            (self.stream_host, self.stream_port),
            StreamHandler,
            self.stream_buffer,
            yolo_embed_enabled=self.yolo_embed_enabled,
            yolo_stream_port=self.yolo_stream_port,
            yolo_stream_path=self.yolo_stream_path,
        )
        self.http_thread = threading.Thread(
            target=self.http_server.serve_forever,
            daemon=True,
        )
        self.http_thread.start()

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

    def _tick(self) -> None:
        if self.cap is None or not self.cap.isOpened():
            self._open_camera()
            return

        ok, frame = self.cap.read()
        if not ok:
            self.get_logger().warn("摄像头取帧失败")
            return

        t_start = time.time()
        mask = run_unet_inference(frame, self.net, self.device)
        mask = postprocess_lane_mask(mask)
        deviation, severity = self.tracker.process_lane_lock_on(mask)
        fit_vis = build_debug_visualization(mask, self.tracker, deviation, t_start)
        mask_vis = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

        self._publish(deviation, severity)
        if self.enable_stream_server:
            self.stream_buffer.update("original", frame, self.stream_quality)
            self.stream_buffer.update("mask", mask_vis, self.stream_quality)
            self.stream_buffer.update("fit", fit_vis, self.stream_quality)

    def _publish(self, deviation: float, severity: float) -> None:
        if self.write_note_file:
            try:
                Path(self.note_file).write_text(
                    f"{float(deviation)},{float(severity)}",
                    encoding="utf-8",
                )
            except Exception as exc:
                self.get_logger().warn(f"写入 note_file 失败: {self.note_file}, {exc}")

        angle_msg = Float32()
        angle_msg.data = float(deviation)
        self.angle_pub.publish(angle_msg)

        severity_msg = Float32()
        severity_msg.data = float(severity)
        self.severity_pub.publish(severity_msg)

        if not self.publish_cmd_vel:
            return

        severity_clamped = max(0.0, min(1.0, float(severity)))
        speed_scale = max(0.35, 1.0 - (0.65 * severity_clamped))
        steering_scale = 1.0 + (0.35 * severity_clamped)

        twist = Twist()
        twist.linear.x = self.linear_speed * speed_scale
        twist.angular.z = -float(deviation) * self.steering_gain * steering_scale
        self.cmd_vel_pub.publish(twist)

    def destroy_node(self) -> bool:
        try:
            if rclpy.ok():
                self.cmd_vel_pub.publish(Twist())
        except Exception:
            pass

        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass

        if self.http_server is not None:
            self.http_server.shutdown()
            self.http_server.server_close()
        return super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = LaneKeepLocalNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
