"""
cv_worker/tracker/bytetrack_wrapper.py
----------------------------------------
Thin wrapper around BoxMOT's ByteTrack implementation.

BoxMOT provides a unified tracker API. We use it here to assign stable
track IDs to person detections across frames.

Each call to update() takes raw detections and returns tracks with IDs.
The wrapper keeps Tracker state internal — callers just pass bboxes in
and get TrackedPerson objects back.
"""

from dataclasses import dataclass

import numpy as np
from loguru import logger


@dataclass
class TrackedPerson:
    """A person with a stable track ID in a single frame."""

    track_id: int
    bbox: tuple[float, float, float, float]  # [x1, y1, x2, y2]
    confidence: float

    @property
    def center(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)

    @property
    def foot_point(self) -> tuple[float, float]:
        """Bottom-centre of bbox — useful for angled cameras."""
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2, y2)


class ByteTrackWrapper:
    """
    Wraps BoxMOT ByteTrack to provide track IDs per frame.

    Args:
        tracker_type: 'bytetrack' or 'botsort' (both supported by BoxMOT).
        frame_rate:   Source video frame rate (used for track age calculations).
    """

    def __init__(self, tracker_type: str = "bytetrack", frame_rate: float = 30.0) -> None:
        self.tracker_type = tracker_type
        self.frame_rate = frame_rate
        self._tracker = None

    def _init_tracker(self, frame_shape: tuple[int, int]) -> None:
        """Lazy-initialise the tracker on first frame."""
        if self._tracker is not None:
            return

        try:
            from boxmot import BYTETracker, BoTSORT  # type: ignore

            h, w = frame_shape
            if self.tracker_type == "bytetrack":
                self._tracker = BYTETracker(
                    track_thresh=0.45,
                    match_thresh=0.8,
                    track_buffer=int(self.frame_rate * 2),  # 2 s buffer
                    frame_rate=int(self.frame_rate),
                )
            elif self.tracker_type == "botsort":
                self._tracker = BoTSORT(frame_rate=int(self.frame_rate))
            else:
                raise ValueError(f"Unknown tracker type: {self.tracker_type}")

            logger.info(f"Tracker initialised: {self.tracker_type}")
        except ImportError as exc:
            logger.error(f"BoxMOT not installed or import failed: {exc}")
            raise

    def update(
        self,
        detections: list,  # list[Detection] from detectors
        frame: np.ndarray,
    ) -> list[TrackedPerson]:
        """
        Update the tracker with new detections from the current frame.

        Args:
            detections: List of Detection objects from a BaseDetector.
            frame:      The current BGR frame (used by some trackers for
                        appearance features — ByteTrack ignores it).

        Returns:
            List of TrackedPerson objects with stable IDs.
        """
        h, w = frame.shape[:2]
        self._init_tracker((h, w))

        if not detections:
            # Still update the tracker with empty input to age out lost tracks
            try:
                self._tracker.update(
                    np.empty((0, 6), dtype=np.float32), frame
                )
            except Exception as exc:
                logger.debug(f"Tracker empty-update error (non-critical): {exc}")
            return []

        # BoxMOT expects: [[x1, y1, x2, y2, conf, class_id], ...]
        det_array = np.array(
            [
                [*d.bbox, d.confidence, 0]  # class_id=0 (person)
                for d in detections
            ],
            dtype=np.float32,
        )

        try:
            tracks = self._tracker.update(det_array, frame)
        except Exception as exc:
            logger.warning(f"Tracker update error: {exc}")
            return []

        if tracks is None or len(tracks) == 0:
            return []

        tracked: list[TrackedPerson] = []
        for track in tracks:
            # BoxMOT output: [x1, y1, x2, y2, track_id, conf, class_id, ...]
            x1, y1, x2, y2 = float(track[0]), float(track[1]), float(track[2]), float(track[3])
            track_id = int(track[4])
            conf = float(track[5]) if len(track) > 5 else 1.0
            tracked.append(
                TrackedPerson(
                    track_id=track_id,
                    bbox=(x1, y1, x2, y2),
                    confidence=conf,
                )
            )

        return tracked

    def reset(self) -> None:
        """Reset tracker state (e.g., after a stream reconnect)."""
        self._tracker = None
        logger.info("Tracker reset.")
