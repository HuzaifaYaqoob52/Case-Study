"""
cv_worker/detectors/base.py
---------------------------
Abstract Detector interface.
Any detection backend (YOLO, RT-DETR, etc.) must implement this.
Swapping models = swap the class, not the pipeline logic.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass
class Detection:
    """A single person detection in one frame."""

    # Bounding box in [x1, y1, x2, y2] pixel coordinates (top-left, bottom-right)
    bbox: tuple[float, float, float, float]
    confidence: float
    # Always "person" for this system — kept for future multi-class use
    class_name: str = "person"

    @property
    def center(self) -> tuple[float, float]:
        """Returns the centre point of the bounding box."""
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)

    @property
    def foot_point(self) -> tuple[float, float]:
        """
        Returns the bottom-centre of the bounding box.
        For overhead cameras this is less important, but for angled
        cameras the foot point is more representative of where a person stands.
        """
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2, y2)


class BaseDetector(ABC):
    """
    Abstract base class for person detectors.

    Implementations must be stateless across frames (tracking state lives
    in the Tracker layer, not here).
    """

    @abstractmethod
    def detect(self, frame: np.ndarray) -> list[Detection]:
        """
        Run person detection on a single BGR frame.

        Args:
            frame: A NumPy array of shape (H, W, 3) in BGR colour order
                   (as returned by OpenCV).

        Returns:
            List of Detection objects. Empty list if no people detected.
        """
        ...

    @abstractmethod
    def warmup(self) -> None:
        """
        Optional warm-up pass (run a dummy frame to load model into GPU/RAM).
        Called once before the main loop. Safe to be a no-op.
        """
        ...
