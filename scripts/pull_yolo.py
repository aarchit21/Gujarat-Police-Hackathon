"""Download YOLOv8n weights into data/models. One-time, ~6MB."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.config import settings  # noqa: E402


def main() -> None:
    dest = ROOT / "data" / "models"
    dest.mkdir(parents=True, exist_ok=True)
    name = Path(settings.yolo_weights or "yolov8n.pt").name
    target = dest / name
    print("weights", target)
    try:
        from ultralytics import YOLO
    except Exception as exc:
        print("FAIL install ultralytics first:", exc)
        sys.exit(1)
    model = YOLO(name)
    print("loaded", getattr(model, "ckpt_path", name))
    print("ok")


if __name__ == "__main__":
    main()
