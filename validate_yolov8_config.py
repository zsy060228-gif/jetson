#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import sys
from pathlib import Path


def load_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"file not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"json root must be object: {path}")
    return payload


def main() -> int:
    profile_path = Path(
        sys.argv[1]
        if len(sys.argv) > 1
        else "/home/jetson/yahboom_ws/src/velpub/config/yolov8_detector_profile.json"
    )
    class_map_path = Path(
        sys.argv[2]
        if len(sys.argv) > 2
        else "/home/jetson/yahboom_ws/src/velpub/config/yolov8_rule_map.json"
    )

    try:
        profile = load_json(profile_path)
        class_map = load_json(class_map_path)
    except Exception as exc:
        print(f"[ERROR] {exc}")
        return 1

    model_path = Path(str(profile.get("model_path", "")))
    classes = class_map.get("classes", {})

    errors = []
    warnings = []

    if not model_path:
        errors.append("profile.model_path missing")
    elif not model_path.exists():
        warnings.append(f"model_path does not exist yet: {model_path}")

    for key in ("video_device", "frame_width", "frame_height", "loop_hz", "confidence", "iou", "max_det"):
        if key not in profile:
            warnings.append(f"profile missing key: {key}")

    if not isinstance(classes, dict) or not classes:
        errors.append("class_map.classes missing or empty")
    else:
        for name, entry in classes.items():
            if not isinstance(entry, dict):
                errors.append(f"class_map entry for {name!r} must be object")
                continue
            if "legacy_sign_id" not in entry:
                errors.append(f"class_map entry for {name!r} missing legacy_sign_id")
                continue
            try:
                int(entry["legacy_sign_id"])
            except Exception:
                errors.append(f"class_map entry for {name!r} has invalid legacy_sign_id")

    print("[INFO] profile_path =", profile_path)
    print("[INFO] class_map_path =", class_map_path)
    print("[INFO] model_path =", model_path)
    print("[INFO] mapped_classes =", len(classes))

    for warning in warnings:
        print(f"[WARN] {warning}")
    for error in errors:
        print(f"[ERROR] {error}")

    if errors:
        return 1

    print("[OK] configuration files are structurally valid")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
