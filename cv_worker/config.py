"""
cv_worker/config.py
-------------------
Centralised configuration via Pydantic Settings.
All values come from environment variables or a .env file — never hardcoded.
"""

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class CVConfig(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Video source
    video_source: str = Field(
        default="data/restaurant_sample.mp4",
        description="Path to a local video file or an RTSP URL.",
    )
    sample_fps: float = Field(
        default=3.0,
        description="Target frames per second to sample from the source.",
    )

    # Table config
    table_config: str = Field(
        default="config/tables.json",
        description="Path to the JSON file defining table polygons.",
    )

    # Detector
    yolo_model: str = Field(
        default="yolov8n.pt",
        description="YOLO model weights file (downloaded automatically on first run).",
    )
    detection_confidence: float = Field(
        default=0.35,
        description="Minimum confidence threshold for person detections.",
    )

    # Tracker
    tracker_type: str = Field(
        default="bytetrack",
        description="Tracker algorithm: 'bytetrack' or 'botsort'.",
    )

    # State machine thresholds
    seated_debounce_secs: float = Field(
        default=20.0,
        description="Seconds a person must remain in a zone before table becomes SEATED.",
    )
    empty_hold_off_secs: float = Field(
        default=60.0,
        description="Seconds a zone must be empty before table transitions to NEEDS_RESET.",
    )
    smoothing_window_frames: int = Field(
        default=5,
        description="Rolling window size (frames) for zone-occupancy smoothing.",
    )

    # Alert thresholds
    waiting_alert_minutes: float = Field(
        default=5.0,
        description="Minutes before a 'waiting too long' alert fires.",
    )
    bill_grace_minutes: float = Field(
        default=3.0,
        description="Grace period before a 'bill still open' alert fires.",
    )
    alert_cooldown_minutes: float = Field(
        default=5.0,
        description="Minimum minutes between duplicate alerts for the same table.",
    )

    # Output
    annotated_output: str = Field(
        default="output/annotated.mp4",
        description="Path for the annotated output video.",
    )
    backend_ingest_url: str = Field(
        default="http://localhost:8000/api/ingest",
        description="URL of the backend event-ingest endpoint (Phase 3+).",
    )


# Singleton — import this everywhere
settings = CVConfig()
