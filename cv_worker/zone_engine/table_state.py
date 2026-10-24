"""
cv_worker/zone_engine/table_state.py
--------------------------------------
Per-table state machine.

States:
    FREE         → No one at the table.
    SEATED       → Guests are present (debounced: must be occupied for N secs).
    NEEDS_RESET  → Guests left, table needs cleaning (hold-off: must be empty for M secs).

Transitions:
    FREE       + people enter zone (for seated_debounce_secs) → SEATED
    SEATED     + zone empties (for empty_hold_off_secs)       → NEEDS_RESET
    NEEDS_RESET + staff resets (external call)                 → FREE

Temporal smoothing:
    Zone occupancy is tracked via a rolling window of the last
    `smoothing_window` frame counts. A zone is considered "occupied"
    if the median count > 0 to filter single-frame glitches.

Party size:
    The state machine tracks the number of people at the table.
    A partial departure (some guests leave) keeps the table SEATED
    as long as ≥1 person remains.

This module is pure Python with no I/O — easy to unit test.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from statistics import median
from typing import Optional


class TableState(str, Enum):
    FREE = "FREE"
    SEATED = "SEATED"
    NEEDS_RESET = "NEEDS_RESET"


@dataclass
class StateEvent:
    """Emitted when a table changes state."""

    table_id: str
    from_state: TableState
    to_state: TableState
    timestamp: float  # Unix timestamp
    party_size: int


class TableStateMachine:
    """
    Manages the lifecycle of a single table.

    Args:
        table_id:              Unique table identifier, e.g. 'T1'.
        seated_debounce_secs:  How long a zone must be continuously occupied
                               before transitioning FREE → SEATED.
        empty_hold_off_secs:   How long a zone must be continuously empty
                               before transitioning SEATED → NEEDS_RESET.
        smoothing_window:      Rolling frame window for occupancy smoothing.
    """

    def __init__(
        self,
        table_id: str,
        seated_debounce_secs: float = 20.0,
        empty_hold_off_secs: float = 60.0,
        smoothing_window: int = 5,
    ) -> None:
        self.table_id = table_id
        self.seated_debounce_secs = seated_debounce_secs
        self.empty_hold_off_secs = empty_hold_off_secs
        self.smoothing_window = smoothing_window

        # Current state
        self.state: TableState = TableState.FREE
        self.party_size: int = 0

        # Timestamps for debounce logic
        self._occupied_since: Optional[float] = None   # when zone first became occupied
        self._empty_since: Optional[float] = None      # when zone first became empty

        # Seated time tracking
        self.seated_at: Optional[float] = None         # when table transitioned → SEATED

        # Rolling window of per-frame person counts for smoothing
        self._count_window: deque[int] = deque(maxlen=smoothing_window)

    # ─── Public interface ────────────────────────────────────────────────────

    def update(self, person_count: int, now: Optional[float] = None) -> list[StateEvent]:
        """
        Feed the current frame's person count into the state machine.

        Args:
            person_count: Number of people detected in this table's zone this frame.
            now:          Current timestamp. Defaults to time.time().

        Returns:
            List of StateEvent objects emitted this frame (usually 0 or 1).
        """
        if now is None:
            now = time.time()

        # Apply smoothing: push count into window, use median as effective count
        self._count_window.append(person_count)
        smoothed_count = int(median(self._count_window))

        events: list[StateEvent] = []

        if self.state == TableState.FREE:
            events += self._handle_free(smoothed_count, now)
        elif self.state == TableState.SEATED:
            events += self._handle_seated(smoothed_count, now)
        elif self.state == TableState.NEEDS_RESET:
            events += self._handle_needs_reset(smoothed_count, now)

        # Update party size (raw count — reflects actual people visible)
        self.party_size = person_count
        return events

    def mark_reset(self, now: Optional[float] = None) -> list[StateEvent]:
        """
        Called by staff (or the API) when a table has been cleaned.
        Transitions NEEDS_RESET → FREE regardless of zone occupancy.
        """
        if now is None:
            now = time.time()
        if self.state == TableState.NEEDS_RESET:
            return [self._transition(TableState.FREE, now)]
        return []

    @property
    def seconds_seated(self) -> float:
        """How long the table has been in SEATED state. 0 if not seated."""
        if self.state == TableState.SEATED and self.seated_at is not None:
            return time.time() - self.seated_at
        return 0.0

    def to_dict(self) -> dict:
        """Serialise state for API responses."""
        return {
            "table_id": self.table_id,
            "state": self.state.value,
            "party_size": self.party_size,
            "seconds_seated": round(self.seconds_seated, 1),
            "seated_at": self.seated_at,
        }

    # ─── State handlers ──────────────────────────────────────────────────────

    def _handle_free(self, count: int, now: float) -> list[StateEvent]:
        if count > 0:
            if self._occupied_since is None:
                self._occupied_since = now
            elif now - self._occupied_since >= self.seated_debounce_secs:
                # Debounce passed — transition to SEATED
                self._occupied_since = None
                self._empty_since = None
                return [self._transition(TableState.SEATED, now)]
        else:
            # Zone is empty — reset debounce timer
            self._occupied_since = None
        return []

    def _handle_seated(self, count: int, now: float) -> list[StateEvent]:
        if count == 0:
            if self._empty_since is None:
                self._empty_since = now
            elif now - self._empty_since >= self.empty_hold_off_secs:
                # Hold-off passed — guests have left
                self._empty_since = None
                self._occupied_since = None
                return [self._transition(TableState.NEEDS_RESET, now)]
        else:
            # Still occupied — reset empty timer
            self._empty_since = None
        return []

    def _handle_needs_reset(self, count: int, now: float) -> list[StateEvent]:
        # In NEEDS_RESET: wait for staff to call mark_reset()
        # If someone sits down again before reset, stay in NEEDS_RESET
        # (avoids the edge case of a brief walkthrough resetting prematurely)
        return []

    # ─── Helpers ─────────────────────────────────────────────────────────────

    def _transition(self, new_state: TableState, now: float) -> StateEvent:
        event = StateEvent(
            table_id=self.table_id,
            from_state=self.state,
            to_state=new_state,
            timestamp=now,
            party_size=self.party_size,
        )
        if new_state == TableState.SEATED:
            self.seated_at = now
        elif new_state in (TableState.FREE, TableState.NEEDS_RESET):
            self.seated_at = None
        self.state = new_state
        return event


def build_state_machines(
    zones: list,  # list[TableZone]
    seated_debounce_secs: float,
    empty_hold_off_secs: float,
    smoothing_window: int,
) -> dict[str, TableStateMachine]:
    """
    Construct one TableStateMachine per zone.

    Returns a dict mapping zone_id → TableStateMachine.
    """
    return {
        zone.id: TableStateMachine(
            table_id=zone.id,
            seated_debounce_secs=seated_debounce_secs,
            empty_hold_off_secs=empty_hold_off_secs,
            smoothing_window=smoothing_window,
        )
        for zone in zones
    }
