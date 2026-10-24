"""
cv_worker/detectors/yolo_detector.py
--------------------------------------
YOLO-based person detector (Ultralytics, AGPL-3.0).

NOTE: Ultralytics YOLO is AGPL-3.0 licensed. This implementation is kept
behind the BaseDetector interface so it can be replaced with a permissively
licensed model (e.g. RT-DETR) without changing any downstream code.

Only class 0 ("person") detections are returned.
"""

import numpy as np
from loguru import logger

from cv_worker.detectors.base import BaseDetector, Detection


class YOLODetector(BaseDetector):
    """
    Person detector using YOLOv8 (or any YOLO variant supported by Ultralytics).

    Args:
        model_path: Model weights file, e.g. 'yolov8n.pt'.
                    Downloaded automatically on first run if not present.
        confidence: Minimum confidence threshold (0–1).
        device:     Inference device, e.g. 'cpu', 'cuda', 'mps'.
                    None = auto-select.
    """

    PERSON_CLASS_ID = 0  # COCO class 0 is "person"

    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        confidence: float = 0.35,
        device: str | None = None,
    ) -> None:
        self.model_path = model_path
        self.confidence = confidence
        self.device = device
        self._model = None

    def _load_model(self) -> None:
        """Lazy-load the model on first use."""
        if self._model is not None:
            return
        try:
            from ultralytics import YOLO  # type: ignore

            logger.info(f"Loading YOLO model: {self.model_path}")
            self._model = YOLO(self.model_path)
            if self.device:
                self._model.to(self.device)
            logger.info("YOLO model loaded successfully.")
        except Exception as exc:
            logger.error(f"Failed to load YOLO model '{self.model_path}': {exc}")
            raise

    def warmup(self) -> None:
        """Run inference on a black frame to initialise GPU/RAM."""
        self._load_model()
        dummy = np.zeros((640, 640, 3), dtype=np.uint8)
        self._model.predict(
            dummy,
            classes=[self.PERSON_CLASS_ID],
            conf=self.confidence,
            verbose=False,
        )
        logger.info("YOLO warmup complete.")

    def detect(self, frame: np.ndarray) -> list[Detection]:
        """
        Detect people in a single BGR frame.

        Returns a list of Detection objects, one per person found.
        """
        self._load_model()

        try:
            results = self._model.predict(
                frame,
                classes=[self.PERSON_CLASS_ID],
                conf=self.confidence,
                verbose=False,
            )
        except Exception as exc:
            # Graceful degradation: log and return empty rather than crashing
            logger.warning(f"YOLO inference error: {exc}")
            return []

        detections: list[Detection] = []
        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                # box.xyxy is shape (1, 4) — flatten to a 1D array
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().tolist()
                conf = float(box.conf[0].cpu().numpy())
                detections.append(
                    Detection(
                        bbox=(x1, y1, x2, y2),
                        confidence=conf,
                        class_name="person",
                    )
                )

        return detections
