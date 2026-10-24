"""
cv_worker/detectors/__init__.py
"""
from cv_worker.detectors.base import BaseDetector, Detection
from cv_worker.detectors.yolo_detector import YOLODetector

__all__ = ["BaseDetector", "Detection", "YOLODetector"]
