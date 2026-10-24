"""
cv_worker/detectors/rtdetr_detector.py
----------------------------------------
Stub for a future RT-DETR detector (permissive license).

RT-DETR is available via:
  - Ultralytics (rtdetr-l.pt, rtdetr-x.pt) — same API as YOLO
  - Hugging Face Transformers (PekingU/rtdetr-r50vd)

When you are ready to drop AGPL, implement this class by replacing the body
of detect() and warmup() with the RT-DETR inference code.
No changes needed anywhere else in the pipeline.
"""

import numpy as np

from cv_worker.detectors.base import BaseDetector, Detection


class RTDETRDetector(BaseDetector):
    """
    Placeholder RT-DETR detector.

    To activate:
    1. Install: pip install transformers torch torchvision
    2. Implement warmup() and detect() below.
    3. Change DETECTOR_CLASS in cv_worker/pipeline.py to RTDETRDetector.
    """

    def __init__(self, model_name: str = "PekingU/rtdetr-r50vd", confidence: float = 0.35) -> None:
        self.model_name = model_name
        self.confidence = confidence
        # TODO: Load model here
        raise NotImplementedError(
            "RTDETRDetector is not yet implemented. "
            "See docstring above for instructions."
        )

    def warmup(self) -> None:
        raise NotImplementedError

    def detect(self, frame: np.ndarray) -> list[Detection]:
        raise NotImplementedError
