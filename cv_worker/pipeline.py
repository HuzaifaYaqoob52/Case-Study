"""
cv_worker/pipeline.py
-----------------------
Orchestrates the full CV pipeline:

  VideoSource → Detector → Tracker → ZoneEngine → StateMachine → AlertEngine → Annotator

Phase 2 additions:
- Alert engine integration (waiting, reset, bill-check alerts)
- Bill provider (manual or mock POS)
- Next-table suggestion
- Alert annotations on output video
- State events trigger bill lifecycle (open on SEATED, check on NEEDS_RESET)
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from loguru import logger

from cv_worker.alerts.alert_engine import Alert, AlertEngine, AlertType
from cv_worker.alerts.bill_provider import (
    BaseBillProvider,
    ManualBillProvider,
    MockPOSBillProvider,
)
from cv_worker.alerts.next_table import suggest_next_table
from cv_worker.annotator import draw_alerts, draw_hud, draw_persons, draw_zones
from cv_worker.config import settings
from cv_worker.detectors.base import BaseDetector
from cv_worker.detectors.yolo_detector import YOLODetector
from cv_worker.tracker.bytetrack_wrapper import ByteTrackWrapper
from cv_worker.video_source import VideoSource
from cv_worker.zone_engine.polygon_zone import (
    compute_zone_occupancies,
    load_zones_from_config,
)
from cv_worker.zone_engine.table_state import TableState, build_state_machines


class CVPipeline:
    """
    Full computer-vision pipeline.

    Args:
        video_source_path: Path to video file or RTSP URL.
        table_config_path: Path to tables.json.
        output_path:       Path for annotated output video. None = no video output.
        detector:          Optional custom BaseDetector. Defaults to YOLODetector.
        bill_provider:     Optional custom BillProvider. Defaults to MockPOSBillProvider.
    """

    def __init__(
        self,
        video_source_path: Optional[str] = None,
        table_config_path: Optional[str] = None,
        output_path: Optional[str] = None,
        detector: Optional[BaseDetector] = None,
        bill_provider: Optional[BaseBillProvider] = None,
    ) -> None:
        self.video_source_path = video_source_path or settings.video_source
        self.table_config_path = table_config_path or settings.table_config
        self.output_path = output_path or settings.annotated_output

        # Components (built on run())
        self.detector = detector or YOLODetector(
            model_path=settings.yolo_model,
            confidence=settings.detection_confidence,
        )
        self.tracker = ByteTrackWrapper(
            tracker_type=settings.tracker_type,
        )
        self.bill_provider = bill_provider or MockPOSBillProvider(
            auto_pay_delay_secs=None,  # Bills stay open until manually paid
        )
        self.zones = None
        self.state_machines = None
        self.alert_engine = None

    # ─── Main entry point ────────────────────────────────────────────────────

    def run(self) -> None:
        """
        Run the pipeline to completion.
        For local files: processes until the video ends, then exits.
        For RTSP: runs until interrupted (Ctrl+C) or fatal error.
        """
        logger.info("=== CV Pipeline starting ===")

        # Load table zones
        self.zones = load_zones_from_config(self.table_config_path)
        self.state_machines = build_state_machines(
            zones=self.zones,
            seated_debounce_secs=settings.seated_debounce_secs,
            empty_hold_off_secs=settings.empty_hold_off_secs,
            smoothing_window=settings.smoothing_window_frames,
        )
        logger.info(f"Tables: {[z.id for z in self.zones]}")

        # Initialise alert engine
        self.alert_engine = AlertEngine(
            bill_provider=self.bill_provider,
            waiting_alert_minutes=settings.waiting_alert_minutes,
            bill_grace_minutes=settings.bill_grace_minutes,
            alert_cooldown_minutes=settings.alert_cooldown_minutes,
        )
        logger.info(
            f"Alert engine: waiting={settings.waiting_alert_minutes}min, "
            f"bill_grace={settings.bill_grace_minutes}min, "
            f"cooldown={settings.alert_cooldown_minutes}min"
        )

        # Warmup model
        logger.info("Warming up detector…")
        self.detector.warmup()

        # Set up video writer
        writer: Optional[cv2.VideoWriter] = None
        out_path = Path(self.output_path)

        frame_idx = 0
        fps_counter_start = time.time()
        fps_counter_frames = 0
        current_fps = 0.0

        # Alert check frequency: evaluate alerts every N frames (not every frame)
        alert_check_interval = max(1, int(settings.sample_fps))  # ~once per second

        with VideoSource(self.video_source_path, target_fps=settings.sample_fps) as src:
            for frame, video_ts in src.frames():
                frame_idx += 1

                # ── 1. Detection ──────────────────────────────────────────
                detections = self.detector.detect(frame)

                # ── 2. Tracking ───────────────────────────────────────────
                tracked_persons = self.tracker.update(detections, frame)

                # ── 3. Zone occupancy ─────────────────────────────────────
                occupancies = compute_zone_occupancies(self.zones, tracked_persons)

                # ── 4. State machine updates ──────────────────────────────
                all_state_events = []
                for zone in self.zones:
                    occ = occupancies[zone.id]
                    sm = self.state_machines[zone.id]
                    events = sm.update(occ.count, now=video_ts)
                    all_state_events.extend(events)

                # ── 5. Handle state transitions (bill lifecycle) ──────────
                for event in all_state_events:
                    logger.info(
                        f"[EVENT] {event.table_id}: {event.from_state.value}"
                        f" → {event.to_state.value} | party={event.party_size}"
                    )
                    # Open bill when table becomes SEATED
                    if event.to_state == TableState.SEATED:
                        self.bill_provider.open_bill(event.table_id)
                    # Clear alerts and close bill when table is reset to FREE
                    elif event.to_state == TableState.FREE:
                        self.alert_engine.clear_alerts_for_table(event.table_id)
                        self.bill_provider.close_bill(event.table_id)

                # ── 6. Evaluate alert rules ───────────────────────────────
                new_alerts: list[Alert] = []
                if frame_idx % alert_check_interval == 0:
                    new_alerts = self.alert_engine.evaluate(
                        self.state_machines, now=video_ts
                    )
                    for alert in new_alerts:
                        logger.warning(
                            f"[ALERT] {alert.alert_type.value}: {alert.message}"
                        )

                # ── 7. Next-table suggestion (log periodically) ──────────
                if frame_idx % (alert_check_interval * 10) == 0:
                    suggestion = suggest_next_table(self.state_machines, now=video_ts)
                    if suggestion:
                        logger.info(
                            f"[SUGGESTION] Next table to free up: "
                            f"{suggestion.table_id} "
                            f"(seated {suggestion.minutes_seated:.0f}min, "
                            f"party={suggestion.party_size}, "
                            f"confidence={suggestion.confidence})"
                        )

                # ── 8. Log per-frame summary every 30 frames ──────────────
                if frame_idx % 30 == 0:
                    status = {
                        z.id: f"{self.state_machines[z.id].state.value}({occupancies[z.id].count})"
                        for z in self.zones
                    }
                    logger.debug(f"Frame {frame_idx} | {status}")

                # ── 9. Annotate frame ─────────────────────────────────────
                annotated = frame.copy()
                annotated = draw_zones(annotated, self.zones, self.state_machines, occupancies)
                annotated = draw_persons(annotated, tracked_persons)

                # Draw active alerts on frame
                active_alerts = self.alert_engine.get_active_alerts()
                annotated = draw_alerts(annotated, active_alerts)

                fps_counter_frames += 1
                elapsed = time.time() - fps_counter_start
                if elapsed >= 1.0:
                    current_fps = fps_counter_frames / elapsed
                    fps_counter_frames = 0
                    fps_counter_start = time.time()

                annotated = draw_hud(annotated, current_fps, frame_idx, video_ts)

                # ── 10. Initialise video writer on first frame ────────────
                if writer is None and self.output_path:
                    out_path.parent.mkdir(parents=True, exist_ok=True)
                    h, w = annotated.shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    writer = cv2.VideoWriter(
                        str(out_path),
                        fourcc,
                        settings.sample_fps,
                        (w, h),
                    )
                    logger.info(f"Writing annotated video to: {out_path}")

                if writer:
                    writer.write(annotated)

        if writer:
            writer.release()
            logger.info(f"Annotated video saved: {out_path}")

        logger.info(f"=== Pipeline finished. {frame_idx} frames processed. ===")

        # Final state summary
        logger.info("── Final table states ──")
        for zone in self.zones:
            sm = self.state_machines[zone.id]
            logger.info(f"  {zone.name}: {sm.state.value} | party={sm.party_size}")

        # Final alert summary
        all_alerts = self.alert_engine.get_all_alerts()
        if all_alerts:
            logger.info(f"── Total alerts fired: {len(all_alerts)} ──")
            for alert in all_alerts:
                ack_str = "✓" if alert.acknowledged else "✗"
                logger.info(f"  [{ack_str}] {alert.alert_type.value}: {alert.message}")
        else:
            logger.info("── No alerts fired during this run. ──")
