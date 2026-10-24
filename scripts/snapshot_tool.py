"""
scripts/snapshot_tool.py
--------------------------
Grab a single frame from a video source and save it as a JPEG.

Use this to get a reference image for drawing table polygons in
config/tables.json (or later, the setup wizard in Phase 4).

Usage:
    python scripts/snapshot_tool.py
    python scripts/snapshot_tool.py --source data/restaurant_sample.mp4 --out output/snapshot.jpg
    python scripts/snapshot_tool.py --source rtsp://192.168.1.100:554/stream
"""

import argparse
import sys
from pathlib import Path

# Allow running from the project root
sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
from loguru import logger

from cv_worker.config import settings
from cv_worker.video_source import VideoSource


def main() -> None:
    parser = argparse.ArgumentParser(description="Capture a snapshot from a video source")
    parser.add_argument("--source", type=str, default=settings.video_source)
    parser.add_argument("--out", type=str, default="output/snapshot.jpg")
    args = parser.parse_args()

    logger.info(f"Capturing snapshot from: {args.source}")
    src = VideoSource(args.source)
    frame = src.get_snapshot()

    if frame is None:
        logger.error("Failed to capture snapshot.")
        sys.exit(1)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), frame)
    h, w = frame.shape[:2]
    logger.info(f"Snapshot saved: {out} ({w}×{h})")
    logger.info(
        "Open the snapshot in any image editor and note the pixel coordinates\n"
        "of each table's corners. Add them to config/tables.json as polygons."
    )


if __name__ == "__main__":
    main()
