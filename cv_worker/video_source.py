"""
cv_worker/video_source.py
--------------------------
Handles reading frames from RTSP streams or local video files.

Features:
- Supports both RTSP URLs and local video files (MP4, AVI, etc.)
- Auto-reconnects on stream drops with exponential backoff
- Yields frames at a configured target FPS (skips frames to match rate)
- Emits a brief wait period between reconnect attempts to avoid hammering
- Context manager support for clean resource release
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Iterator, Optional

import cv2
import numpy as np
from loguru import logger


class VideoSource:
    """
    Frame iterator for RTSP streams and local video files.

    Args:
        source:          Path to a video file or an RTSP URL string.
        target_fps:      Desired frames-per-second to yield. Frames are
                         dropped (not processed) to match this rate.
                         Set to None to yield every frame.
        reconnect_delay: Initial delay (seconds) before first reconnect attempt.
                         Doubles on each failure (max: reconnect_max_delay).
        reconnect_max_delay: Maximum backoff delay in seconds.
        max_reconnects:  Maximum reconnect attempts before giving up.
                         None = unlimited (keeps trying forever).
    """

    def __init__(
        self,
        source: str,
        target_fps: Optional[float] = 3.0,
        reconnect_delay: float = 2.0,
        reconnect_max_delay: float = 30.0,
        max_reconnects: Optional[int] = None,
    ) -> None:
        self.source = source
        self.target_fps = target_fps
        self.reconnect_delay = reconnect_delay
        self.reconnect_max_delay = reconnect_max_delay
        self.max_reconnects = max_reconnects

        self._cap: Optional[cv2.VideoCapture] = None
        self._source_fps: float = 30.0  # actual source FPS (read on open)
        self._frame_interval: float = 1.0 / target_fps if target_fps else 0.0

    # ─── Context manager ─────────────────────────────────────────────────────

    def __enter__(self) -> "VideoSource":
        return self

    def __exit__(self, *args) -> None:
        self.release()

    def release(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    # ─── Properties ──────────────────────────────────────────────────────────

    @property
    def is_file(self) -> bool:
        """True if the source is a local video file (not RTSP)."""
        return not str(self.source).startswith("rtsp://")

    @property
    def source_fps(self) -> float:
        return self._source_fps

    # ─── Connection ──────────────────────────────────────────────────────────

    def _open(self) -> bool:
        """Open (or reopen) the video source. Returns True on success."""
        self.release()

        if self.is_file:
            path = Path(self.source)
            if not path.exists():
                logger.error(f"Video file not found: {path}")
                return False

        logger.info(f"Opening video source: {self.source}")
        cap = cv2.VideoCapture(self.source)

        if not cap.isOpened():
            logger.warning(f"Failed to open: {self.source}")
            cap.release()
            return False

        fps = cap.get(cv2.CAP_PROP_FPS)
        self._source_fps = fps if fps > 0 else 30.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        logger.info(
            f"Opened: {self.source} | {w}×{h} @ {self._source_fps:.1f} fps"
            + (f" | {total} frames" if total > 0 else "")
        )

        self._cap = cap
        return True

    # ─── Frame iteration ─────────────────────────────────────────────────────

    def frames(self) -> Iterator[tuple[np.ndarray, float]]:
        """
        Yield (frame, timestamp) pairs at the configured target FPS.

        For local files, timestamp = video position in seconds.
        For RTSP, timestamp = wall-clock time.time().

        On stream failure, attempts to reconnect automatically.
        For local files, iteration stops when the file ends (no reconnect).
        """
        reconnect_attempts = 0
        delay = self.reconnect_delay

        while True:
            if not self._open():
                if self.is_file:
                    logger.error("Cannot open video file. Aborting.")
                    return
                # RTSP reconnect
                reconnect_attempts += 1
                if self.max_reconnects and reconnect_attempts > self.max_reconnects:
                    logger.error(f"Max reconnect attempts ({self.max_reconnects}) reached. Stopping.")
                    return
                logger.warning(f"Reconnecting in {delay:.0f}s… (attempt {reconnect_attempts})")
                time.sleep(delay)
                delay = min(delay * 2, self.reconnect_max_delay)
                continue

            # Successful open — reset backoff
            reconnect_attempts = 0
            delay = self.reconnect_delay

            # FPS throttling: skip source frames to match target_fps
            skip_n = 0
            if self.target_fps and self._source_fps > self.target_fps:
                skip_n = max(1, round(self._source_fps / self.target_fps)) - 1

            frame_index = 0
            last_yield_time = 0.0

            while True:
                ret, frame = self._cap.read()
                if not ret:
                    if self.is_file:
                        logger.info("Video file ended.")
                        return  # Normal end — stop iteration
                    else:
                        logger.warning("RTSP stream lost. Reconnecting…")
                        break  # Break inner loop → reconnect

                frame_index += 1

                # Skip frames to hit target FPS
                if skip_n > 0 and (frame_index % (skip_n + 1)) != 0:
                    continue

                # Wall-clock or video-position timestamp
                if self.is_file:
                    ts = self._cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
                else:
                    ts = time.time()

                # Rate-limit: if we're yielding too fast, sleep a tiny bit
                if self.target_fps:
                    elapsed = ts - last_yield_time if self.is_file else time.time() - last_yield_time
                    if elapsed < self._frame_interval and elapsed > 0:
                        time.sleep(max(0, self._frame_interval - elapsed))
                    last_yield_time = ts if self.is_file else time.time()

                yield frame, ts

    def get_snapshot(self) -> Optional[np.ndarray]:
        """
        Read a single frame from the source (for setup wizard / snapshot tool).
        Does not affect the frame iterator.
        """
        if not self._open():
            return None
        ret, frame = self._cap.read()
        self.release()
        return frame if ret else None
