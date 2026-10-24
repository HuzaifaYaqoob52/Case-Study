"""
cv_worker/annotator.py
------------------------
Draws debug annotations onto frames for the output video.

Annotations:
- Table polygon outlines (colour-coded by state)
- Table label (ID + name + person count)
- Person bounding boxes with track IDs
- Active alerts panel (right side of frame)
- FPS counter and source timestamp

This module is only used by the annotation pipeline (main.py --annotate).
It has no effect on the detection/state logic.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

import cv2
import numpy as np

from cv_worker.zone_engine.polygon_zone import TableZone
from cv_worker.zone_engine.table_state import TableState, TableStateMachine
from cv_worker.tracker.bytetrack_wrapper import TrackedPerson

if TYPE_CHECKING:
    from cv_worker.alerts.alert_engine import Alert, AlertType


# ─── Colour palette (BGR) ────────────────────────────────────────────────────
COLOURS = {
    TableState.FREE: (60, 200, 60),          # Green
    TableState.SEATED: (50, 165, 255),        # Orange
    TableState.NEEDS_RESET: (0, 80, 220),     # Red
}
PERSON_BOX_COLOUR = (255, 200, 50)           # Cyan-ish
TEXT_BG_COLOUR = (20, 20, 20)
TEXT_COLOUR = (240, 240, 240)
ALERT_BG_COLOUR = (30, 30, 60)              # Dark red-ish panel
ALERT_WARNING_COLOUR = (0, 180, 255)         # Orange for warnings
ALERT_URGENT_COLOUR = (0, 50, 255)           # Red for urgent
ALERT_INFO_COLOUR = (200, 200, 50)           # Cyan for info
FONT = cv2.FONT_HERSHEY_SIMPLEX


def draw_zones(
    frame: np.ndarray,
    zones: list[TableZone],
    state_machines: dict[str, TableStateMachine],
    occupancies: dict,  # dict[str, ZoneOccupancy]
) -> np.ndarray:
    """
    Draw table zones onto the frame, colour-coded by state.

    Args:
        frame:          BGR frame to annotate (modified in-place).
        zones:          List of TableZone objects.
        state_machines: Dict mapping zone_id → TableStateMachine.
        occupancies:    Dict mapping zone_id → ZoneOccupancy.

    Returns:
        The annotated frame.
    """
    overlay = frame.copy()

    for zone in zones:
        sm = state_machines.get(zone.id)
        occ = occupancies.get(zone.id)
        state = sm.state if sm else TableState.FREE
        count = occ.count if occ else 0

        colour = COLOURS[state]
        pts = np.array(zone.polygon_points(), dtype=np.int32)

        # Semi-transparent fill
        cv2.fillPoly(overlay, [pts], colour)

        # Polygon outline
        cv2.polylines(frame, [pts], isClosed=True, color=colour, thickness=2)

        # Blend overlay
        alpha = 0.20
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        overlay = frame.copy()

        # Label: table name + state + count
        centroid = pts.mean(axis=0).astype(int)
        label = f"{zone.name} | {state.value}"
        count_label = f"People: {count}"

        # Show seated duration if table is seated
        duration_label = ""
        if sm and sm.state == TableState.SEATED and sm.seated_at is not None:
            # Use approximate video-time duration (sm tracks seated_at in video time)
            duration_label = f"Seated: {_format_duration(sm.seated_at)}"

        _draw_label(frame, label, (centroid[0], centroid[1] - 18), scale=0.55)
        _draw_label(frame, count_label, (centroid[0], centroid[1] + 4), scale=0.50)
        if duration_label:
            _draw_label(frame, duration_label, (centroid[0], centroid[1] + 24), scale=0.45)

    return frame


def draw_persons(frame: np.ndarray, persons: list[TrackedPerson]) -> np.ndarray:
    """Draw bounding boxes and track IDs for detected people."""
    for person in persons:
        x1, y1, x2, y2 = [int(v) for v in person.bbox]
        cv2.rectangle(frame, (x1, y1), (x2, y2), PERSON_BOX_COLOUR, 2)
        _draw_label(frame, f"ID:{person.track_id}", (x1, y1 - 6), scale=0.45)
    return frame


def draw_alerts(frame: np.ndarray, alerts: list) -> np.ndarray:
    """
    Draw an alert panel on the right side of the frame.

    Args:
        frame:  BGR frame to annotate.
        alerts: List of Alert objects (unacknowledged).

    Returns:
        The annotated frame.
    """
    if not alerts:
        return frame

    h, w = frame.shape[:2]
    panel_width = min(350, w // 3)
    panel_x = w - panel_width - 10
    panel_y = 10

    # Draw panel background
    panel_height = min(30 + len(alerts) * 50, h - 20)
    overlay = frame.copy()
    cv2.rectangle(
        overlay,
        (panel_x, panel_y),
        (panel_x + panel_width, panel_y + panel_height),
        ALERT_BG_COLOUR,
        cv2.FILLED,
    )
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

    # Panel title
    _draw_label(
        frame,
        f"ALERTS ({len(alerts)})",
        (panel_x + panel_width // 2, panel_y + 16),
        scale=0.55,
        anchor="center",
    )

    # Draw each alert
    y = panel_y + 36
    for alert in alerts[:6]:  # Cap at 6 visible alerts
        # Choose colour by severity
        from cv_worker.alerts.alert_engine import AlertSeverity
        if alert.severity == AlertSeverity.URGENT:
            colour = ALERT_URGENT_COLOUR
        elif alert.severity == AlertSeverity.WARNING:
            colour = ALERT_WARNING_COLOUR
        else:
            colour = ALERT_INFO_COLOUR

        # Severity indicator dot
        cv2.circle(frame, (panel_x + 12, y), 5, colour, cv2.FILLED)

        # Alert type
        type_text = alert.alert_type.value.replace("_", " ")
        _draw_label(
            frame,
            f"{alert.table_id}: {type_text}",
            (panel_x + 22, y),
            scale=0.40,
            anchor="left",
        )

        # Truncated message
        msg = alert.message
        if len(msg) > 40:
            msg = msg[:37] + "…"
        _draw_label(
            frame,
            msg,
            (panel_x + 22, y + 18),
            scale=0.35,
            anchor="left",
        )

        y += 46
        if y > panel_y + panel_height - 20:
            break

    return frame


def draw_hud(
    frame: np.ndarray,
    fps: float,
    frame_idx: int,
    video_ts: float,
) -> np.ndarray:
    """Draw a small heads-up display in the top-left corner."""
    lines = [
        f"Frame: {frame_idx}",
        f"Video time: {video_ts:.1f}s",
        f"Render FPS: {fps:.1f}",
    ]
    y = 20
    for line in lines:
        _draw_label(frame, line, (10, y), scale=0.48, anchor="left")
        y += 20
    return frame


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _draw_label(
    frame: np.ndarray,
    text: str,
    pos: tuple[int, int],
    scale: float = 0.5,
    anchor: str = "center",
) -> None:
    """Draw a text label with a dark background for readability."""
    thickness = 1
    (tw, th), baseline = cv2.getTextSize(text, FONT, scale, thickness)

    x, y = pos
    if anchor == "center":
        x -= tw // 2
        y += th // 2

    # Background rectangle
    pad = 3
    cv2.rectangle(
        frame,
        (x - pad, y - th - pad),
        (x + tw + pad, y + baseline + pad),
        TEXT_BG_COLOUR,
        cv2.FILLED,
    )
    cv2.putText(frame, text, (x, y), FONT, scale, TEXT_COLOUR, thickness, cv2.LINE_AA)


def _format_duration(seated_at: float) -> str:
    """Format a seated-since timestamp as a human-readable duration label."""
    # This is called during annotation — we just show the raw timestamp
    # since we're working with video time, not wall clock
    mins = int(seated_at // 60)
    secs = int(seated_at % 60)
    return f"since {mins}:{secs:02d}"
