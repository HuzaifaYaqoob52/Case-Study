"""
cv_worker/main.py
------------------
Entry point for the CV worker process.

Usage:
    python -m cv_worker.main
    python -m cv_worker.main --video path/to/video.mp4
    python -m cv_worker.main --video rtsp://192.168.1.100:554/stream
    python -m cv_worker.main --video path/to/video.mp4 --output output/annotated.mp4
    python -m cv_worker.main --snapshot              # save one frame, then exit
"""

import argparse
import sys
from pathlib import Path

from loguru import logger

from cv_worker.config import settings


def configure_logging() -> None:
    """Set up loguru with a clean format."""
    logger.remove()
    logger.add(
        sys.stderr,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | {message}",
        level="DEBUG",
        colorize=True,
    )
    logger.add(
        "logs/cv_worker.log",
        rotation="10 MB",
        retention="7 days",
        level="INFO",
        format="{time} | {level} | {message}",
    )


def main() -> None:
    configure_logging()

    parser = argparse.ArgumentParser(
        description="AI Table Monitor — CV Worker (Phase 1)"
    )
    parser.add_argument(
        "--video",
        type=str,
        default=None,
        help="Path to video file or RTSP URL (overrides VIDEO_SOURCE env var)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to table config JSON (overrides TABLE_CONFIG env var)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path for annotated output video (overrides ANNOTATED_OUTPUT env var)",
    )
    parser.add_argument(
        "--snapshot",
        action="store_true",
        help="Save a single frame snapshot (for setup wizard) and exit",
    )
    args = parser.parse_args()

    # ── Snapshot mode ────────────────────────────────────────────────────────
    if args.snapshot:
        from cv_worker.video_source import VideoSource
        import cv2

        src_path = args.video or settings.video_source
        logger.info(f"Taking snapshot from: {src_path}")
        src = VideoSource(src_path)
        frame = src.get_snapshot()
        if frame is None:
            logger.error("Could not capture snapshot.")
            sys.exit(1)
        out = Path("output/snapshot.jpg")
        out.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out), frame)
        logger.info(f"Snapshot saved: {out}")
        return

    # ── Normal pipeline mode ──────────────────────────────────────────────────
    from cv_worker.pipeline import CVPipeline

    pipeline = CVPipeline(
        video_source_path=args.video,
        table_config_path=args.config,
        output_path=args.output,
    )

    try:
        pipeline.run()
    except KeyboardInterrupt:
        logger.info("Interrupted by user. Shutting down.")
    except Exception as exc:
        logger.exception(f"Fatal pipeline error: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
