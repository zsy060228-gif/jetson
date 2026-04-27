#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import socket
import sys
import threading
import time
from http import server
from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
import rclpy
from PIL import Image, ImageDraw, ImageFont
from rclpy.node import Node

from interfaces.msg import TrafficSignDetection, TrafficSignDetections

try:
    from ultralytics import YOLO
except ImportError as exc:  # pragma: no cover - startup validation path
    YOLO = None
    IMPORT_ERROR = exc
else:
    IMPORT_ERROR = None

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


def pick_cjk_font() -> str:
    candidates = [
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
        "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",
        "/usr/share/fonts/truetype/arphic/ukai.ttc",
        "/usr/share/fonts/truetype/arphic/uming.ttc",
    ]
    for path in candidates:
        if Path(path).exists():
            return path
    raise FileNotFoundError("no usable CJK font found on system")


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


class YoloHTTPServer(server.ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, server_address, handler_class, stream_buffer: StreamBuffer):
        super().__init__(server_address, handler_class)
        self.stream_buffer = stream_buffer


class YoloStreamHandler(server.BaseHTTPRequestHandler):
    server_version = "yolov8_detector/1.0"

    def log_message(self, format: str, *args) -> None:
        return

    def do_GET(self) -> None:
        if self.path in ("/", "/index.html"):
            self._serve_index()
            return
        if self.path == "/yolo.mjpg":
            self._serve_mjpeg("yolo")
            return
        self.send_error(404, "Not Found")

    def _serve_index(self) -> None:
        html = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>YOLO Stream</title>
  <style>
    body { font-family: sans-serif; margin: 16px; background: #111; color: #eee; }
    .card { background: #1b1b1b; padding: 12px; border-radius: 10px; max-width: 960px; }
    img { width: 100%; height: auto; display: block; background: #000; border-radius: 8px; }
    h2 { margin: 0 0 8px 0; font-size: 18px; }
  </style>
</head>
<body>
  <div class="card">
    <h2>YOLO</h2>
    <img src="/yolo.mjpg" />
  </div>
</body>
</html>
""".strip().encode("utf-8")
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


class YoloV8Detector(Node):
    """Independent YOLOv8 detector that only publishes standardized detections."""

    def __init__(self) -> None:
        super().__init__("yolov8_detector")

        if YOLO is None:
            raise RuntimeError(f"ultralytics import failed: {IMPORT_ERROR}")

        default_profile_path = (
            Path(__file__).resolve().parent.parent / "config" / "yolov8_detector_profile.json"
        )
        default_model_path = "/home/jetson/yolov8n.pt"
        default_class_map = (
            Path(__file__).resolve().parent.parent / "config" / "yolov8_rule_map.json"
        )
        self.profile_path = Path(
            str(self.declare_parameter("profile_path", str(default_profile_path)).value)
        )
        profile = self._load_profile(self.profile_path)

        self.model_path = str(
            self.declare_parameter("model_path", str(profile.get("model_path", default_model_path))).value
        )
        self.video_device = str(
            self.declare_parameter("video_device", str(profile.get("video_device", "/dev/video2"))).value
        )
        self.frame_width = int(
            self.declare_parameter("frame_width", int(profile.get("frame_width", 640))).value
        )
        self.frame_height = int(
            self.declare_parameter("frame_height", int(profile.get("frame_height", 480))).value
        )
        self.loop_hz = float(
            self.declare_parameter("loop_hz", float(profile.get("loop_hz", 10.0))).value
        )
        self.confidence = float(
            self.declare_parameter("confidence", float(profile.get("confidence", 0.35))).value
        )
        self.iou = float(self.declare_parameter("iou", float(profile.get("iou", 0.45))).value)
        self.max_det = int(
            self.declare_parameter("max_det", int(profile.get("max_det", 20))).value
        )
        self.source_name = str(self.declare_parameter("source_name", "yolov8_detector").value)
        self.detections_topic = str(
            self.declare_parameter(
                "detections_topic", str(profile.get("detections_topic", "/traffic_sign/detections"))
            ).value
        )
        self.show_debug = bool(self.declare_parameter("show_debug", False).value)
        self.class_map_path = Path(
            str(self.declare_parameter("class_map_path", str(default_class_map)).value)
        )
        self.fallback_video_devices_csv = str(
            self.declare_parameter(
                "fallback_video_devices_csv",
                "/dev/video2,/dev/video0,/dev/video1,/dev/video3",
            ).value
        )
        self.stream_host = str(self.declare_parameter("stream_host", "0.0.0.0").value)
        self.stream_port = int(self.declare_parameter("stream_port", 8091).value)
        self.stream_quality = int(self.declare_parameter("stream_quality", 80).value)
        self.enable_stream_server = bool(
            self.declare_parameter("enable_stream_server", True).value
        )
        self.draw_detections = bool(self.declare_parameter("draw_detections", True).value)
        self.overlay_font_path = str(
            self.declare_parameter("overlay_font_path", pick_cjk_font()).value
        )

        model_file = Path(self.model_path)
        if not model_file.exists():
            raise FileNotFoundError(f"model_path not found: {model_file}")

        self.model = YOLO(str(model_file))
        self.names = getattr(self.model.model, "names", None) or getattr(self.model, "names", {})
        self.class_map = load_class_map(self.class_map_path)
        self.pub = self.create_publisher(TrafficSignDetections, self.detections_topic, 10)

        self.cap: Optional[cv2.VideoCapture] = None
        self.last_camera_error_time = 0.0
        self.stream_buffer = StreamBuffer()
        self.http_server: Optional[YoloHTTPServer] = None
        self.http_thread: Optional[threading.Thread] = None
        self._open_camera()
        if self.enable_stream_server:
            self._start_stream_server()

        period = 1.0 / self.loop_hz if self.loop_hz > 0 else 0.1
        self.create_timer(period, self._tick)
        self.get_logger().info(
            f"yolov8_detector started: model={self.model_path}, video={self.video_device}, "
            f"detections_topic={self.detections_topic}, class_map_path={self.class_map_path}, "
            f"profile_path={self.profile_path}, fallback_video_devices_csv={self.fallback_video_devices_csv}, "
            f"stream=http://{self._stream_host_for_log()}:{self.stream_port}/, "
            f"overlay_font_path={self.overlay_font_path}"
        )

    @staticmethod
    def _format_overlay_text(class_name: str, confidence: float, legacy_sign_id: int) -> str:
        if legacy_sign_id >= 0:
            return f"{class_name} {confidence:.2f} 已映射 ID:{legacy_sign_id}"
        return f"{class_name} {confidence:.2f} 未映射"

    @staticmethod
    def _confidence_color(confidence: float) -> tuple[int, int, int]:
        if confidence >= 0.80:
            return (0, 255, 120)
        if confidence >= 0.60:
            return (0, 220, 255)
        return (0, 80, 255)

    def _load_font(self, size: int) -> ImageFont.FreeTypeFont:
        return ImageFont.truetype(self.overlay_font_path, size=size)

    def _draw_text(
        self,
        image_bgr,
        position: tuple[int, int],
        text: str,
        *,
        font_size: int,
        text_color: tuple[int, int, int],
        bg_color: Optional[tuple[int, int, int]] = None,
        padding: int = 4,
    ):
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        draw = ImageDraw.Draw(pil_img)
        font = self._load_font(font_size)
        x, y = position
        bbox = draw.textbbox((x, y), text, font=font)
        if bg_color is not None:
            bg_rgb = (bg_color[2], bg_color[1], bg_color[0])
            draw.rectangle(
                (
                    bbox[0] - padding,
                    bbox[1] - padding,
                    bbox[2] + padding,
                    bbox[3] + padding,
                ),
                fill=bg_rgb,
            )
        text_rgb = (text_color[2], text_color[1], text_color[0])
        draw.text((x, y), text, fill=text_rgb, font=font)
        return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    def _draw_status_panel(self, frame, best_detection: Optional[dict]) -> None:
        panel_w = 280
        panel_h = 136
        x2 = frame.shape[1] - 16
        x1 = max(0, x2 - panel_w)
        y1 = 16
        y2 = min(frame.shape[0] - 1, y1 + panel_h)

        cv2.rectangle(frame, (x1, y1), (x2, y2), (32, 32, 32), -1)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (180, 180, 180), 1)

        if best_detection is None:
            lines = [
                "当前目标: 无",
                "置信度: -",
                "规则ID: -",
                "规则状态: -",
            ]
            color = (200, 200, 200)
        else:
            rule_id = best_detection["legacy_sign_id"]
            mapped = rule_id >= 0
            lines = [
                f"当前目标: {best_detection['class_name']}",
                f"置信度: {best_detection['confidence']:.2f}",
                f"规则ID: {rule_id if mapped else '-'}",
                f"规则状态: {'已映射' if mapped else '未映射'}",
            ]
            color = self._confidence_color(float(best_detection["confidence"]))

        for idx, line in enumerate(lines):
            frame[:] = self._draw_text(
                frame,
                (x1 + 12, y1 + 14 + idx * 28),
                line,
                font_size=22,
                text_color=color,
            )

    def _annotate_frame(self, frame, detections) -> "cv2.typing.MatLike":
        annotated = frame.copy()
        best_detection = None
        best_score = -1.0

        for det in detections:
            mapped = int(det.legacy_sign_id) >= 0
            color = self._confidence_color(float(det.confidence))
            cv2.rectangle(annotated, (int(det.xmin), int(det.ymin)), (int(det.xmax), int(det.ymax)), color, 2)
            label = self._format_overlay_text(det.class_name, float(det.confidence), int(det.legacy_sign_id))
            text_y = max(8, int(det.ymin) - 28)
            annotated = self._draw_text(
                annotated,
                (int(det.xmin), text_y),
                label,
                font_size=20,
                text_color=(255, 255, 255),
                bg_color=color,
            )

            score = float(det.confidence) + (0.001 if mapped else 0.0)
            if score > best_score:
                best_score = score
                best_detection = {
                    "class_name": det.class_name,
                    "confidence": float(det.confidence),
                    "legacy_sign_id": int(det.legacy_sign_id),
                }

        self._draw_status_panel(annotated, best_detection)
        return annotated

    def _stream_host_for_log(self) -> str:
        if self.stream_host not in ("", "0.0.0.0", "::"):
            return self.stream_host
        try:
            probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            probe.connect(("8.8.8.8", 80))
            addr = probe.getsockname()[0]
            probe.close()
            if addr:
                return addr
        except OSError:
            pass
        return "127.0.0.1"

    def _start_stream_server(self) -> None:
        self.http_server = YoloHTTPServer(
            (self.stream_host, self.stream_port),
            YoloStreamHandler,
            self.stream_buffer,
        )
        self.http_thread = threading.Thread(
            target=self.http_server.serve_forever,
            daemon=True,
        )
        self.http_thread.start()

    @staticmethod
    def _load_profile(path: Path) -> dict:
        if not path.exists():
            return {}
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"invalid detector profile content: {path}")
        return payload

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

        candidates = []
        seen_devices = set()
        devices = [self.video_device]
        for item in self.fallback_video_devices_csv.split(","):
            item = item.strip()
            if item:
                devices.append(item)

        for device in devices:
            if device in seen_devices:
                continue
            seen_devices.add(device)
            video_index = self._try_parse_video_index(device)
            if video_index is not None:
                candidates.append((video_index, cv2.CAP_V4L2, f"index {video_index} + V4L2"))
                candidates.append((video_index, None, f"index {video_index}"))
            candidates.append((device, cv2.CAP_V4L2, f"path {device} + V4L2"))
            candidates.append((device, None, f"path {device}"))

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
            self.get_logger().info(f"camera opened: {tag}")
            return

        self._log_camera_open_failed()

    def _log_camera_open_failed(self) -> None:
        now = self.get_clock().now().nanoseconds / 1e9
        if now - self.last_camera_error_time >= 2.0:
            self.last_camera_error_time = now
            self.get_logger().error(f"failed to open camera: {self.video_device}")

    def _tick(self) -> None:
        if self.cap is None or not self.cap.isOpened():
            self._open_camera()
            return

        ok, frame = self.cap.read()
        if not ok:
            self.get_logger().warn("camera read failed")
            return

        results = self.model.predict(
            source=frame,
            conf=self.confidence,
            iou=self.iou,
            max_det=self.max_det,
            verbose=False,
        )
        if not results:
            self._publish(frame, [])
            return

        result = results[0]
        boxes = result.boxes
        detections = []
        if boxes is not None:
            frame_area = float(frame.shape[0] * frame.shape[1]) if frame.size else 1.0
            for box in boxes:
                cls_idx = int(box.cls.item())
                confidence = float(box.conf.item())
                xmin, ymin, xmax, ymax = [int(v) for v in box.xyxy[0].tolist()]
                width = max(0, xmax - xmin)
                height = max(0, ymax - ymin)
                area_ratio = float(width * height) / frame_area if frame_area > 0 else 0.0
                class_name = str(self.names.get(cls_idx, cls_idx))
                legacy_sign_id = self.class_map.get(class_name.strip().lower(), -1)

                msg = TrafficSignDetection()
                msg.header.stamp = self.get_clock().now().to_msg()
                msg.header.frame_id = self.source_name
                msg.class_name = class_name
                msg.legacy_sign_id = int(legacy_sign_id)
                msg.confidence = float(confidence)
                msg.xmin = int(xmin)
                msg.ymin = int(ymin)
                msg.xmax = int(xmax)
                msg.ymax = int(ymax)
                msg.area_ratio = float(area_ratio)
                msg.confirmed = False
                msg.source = self.source_name
                detections.append(msg)

        self._publish(frame, detections)

    def _publish(self, frame, detections) -> None:
        msg = TrafficSignDetections()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.source_name
        msg.detections = detections
        self.pub.publish(msg)

        display_frame = self._annotate_frame(frame, detections) if self.draw_detections else frame

        if self.enable_stream_server:
            self.stream_buffer.update("yolo", display_frame, self.stream_quality)

        if self.show_debug:
            cv2.imshow("yolov8_detector", display_frame)
            cv2.waitKey(1)

    def destroy_node(self) -> bool:
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
        if self.http_server is not None:
            try:
                self.http_server.shutdown()
                self.http_server.server_close()
            except Exception:
                pass
        if self.show_debug:
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass
        return super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    try:
        node = YoloV8Detector()
    except Exception as exc:
        print(f"yolov8_detector startup failed: {exc}", file=sys.stderr)
        raise
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
