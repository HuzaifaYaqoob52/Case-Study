"""
tests/test_state_machine.py
----------------------------
Unit tests for the per-table state machine.

Tests use simulated sequences of person counts fed directly into
TableStateMachine.update() — no video, no model, no camera needed.

All time values are injected explicitly (no time.time() calls) for
deterministic, fast tests.
"""

import pytest
from cv_worker.zone_engine.table_state import TableState, TableStateMachine, StateEvent


# ─── Helpers ─────────────────────────────────────────────────────────────────

def make_sm(
    seated_debounce_secs: float = 10.0,
    empty_hold_off_secs: float = 20.0,
    smoothing_window: int = 1,  # disable smoothing for most tests
) -> TableStateMachine:
    """Create a state machine with tight thresholds for fast tests."""
    return TableStateMachine(
        table_id="T1",
        seated_debounce_secs=seated_debounce_secs,
        empty_hold_off_secs=empty_hold_off_secs,
        smoothing_window=smoothing_window,
    )


def feed(sm: TableStateMachine, count: int, now: float) -> list[StateEvent]:
    """Feed a single count update and return emitted events."""
    return sm.update(count, now=now)


# ─── Basic state transitions ─────────────────────────────────────────────────

class TestBasicTransitions:
    def test_initial_state_is_free(self):
        sm = make_sm()
        assert sm.state == TableState.FREE

    def test_free_stays_free_if_no_one_arrives(self):
        sm = make_sm()
        for t in range(0, 30, 1):
            events = feed(sm, 0, float(t))
            assert not events
        assert sm.state == TableState.FREE

    def test_free_to_seated_after_debounce(self):
        sm = make_sm(seated_debounce_secs=10.0)
        # Person enters at t=0
        events = feed(sm, 1, 0.0)
        assert sm.state == TableState.FREE  # not yet
        assert not events

        # 9 seconds pass — still FREE
        events = feed(sm, 1, 9.0)
        assert sm.state == TableState.FREE

        # 10 seconds — should transition
        events = feed(sm, 1, 10.0)
        assert sm.state == TableState.SEATED
        assert len(events) == 1
        assert events[0].to_state == TableState.SEATED

    def test_debounce_resets_if_person_leaves_early(self):
        sm = make_sm(seated_debounce_secs=10.0)
        feed(sm, 1, 0.0)   # person enters
        feed(sm, 1, 5.0)   # still there at 5s
        feed(sm, 0, 6.0)   # leaves before debounce
        feed(sm, 1, 7.0)   # comes back — debounce should restart
        events = feed(sm, 1, 14.0)  # only 7s since re-entry — still FREE
        assert sm.state == TableState.FREE

    def test_seated_to_needs_reset_after_hold_off(self):
        sm = make_sm(seated_debounce_secs=10.0, empty_hold_off_secs=20.0)
        # Transition to SEATED
        feed(sm, 1, 0.0)
        feed(sm, 1, 10.0)
        assert sm.state == TableState.SEATED

        # Zone empties at t=15
        feed(sm, 0, 15.0)
        assert sm.state == TableState.SEATED  # hold-off not reached

        # After 20s of empty (t=35)
        events = feed(sm, 0, 35.0)
        assert sm.state == TableState.NEEDS_RESET
        assert events[0].to_state == TableState.NEEDS_RESET

    def test_needs_reset_to_free_via_mark_reset(self):
        sm = make_sm(seated_debounce_secs=5.0, empty_hold_off_secs=10.0)
        feed(sm, 1, 0.0)
        feed(sm, 1, 5.0)
        feed(sm, 0, 6.0)
        feed(sm, 0, 16.0)
        assert sm.state == TableState.NEEDS_RESET

        events = sm.mark_reset(now=20.0)
        assert sm.state == TableState.FREE
        assert events[0].to_state == TableState.FREE

    def test_mark_reset_ignored_when_not_needs_reset(self):
        sm = make_sm()
        assert sm.state == TableState.FREE
        events = sm.mark_reset(now=0.0)
        assert not events
        assert sm.state == TableState.FREE


# ─── Party size tracking ─────────────────────────────────────────────────────

class TestPartySize:
    def test_party_size_updates_correctly(self):
        sm = make_sm(seated_debounce_secs=5.0)
        feed(sm, 2, 0.0)
        feed(sm, 2, 5.0)
        assert sm.state == TableState.SEATED
        assert sm.party_size == 2

    def test_partial_departure_keeps_seated(self):
        """If 3 → 1 person, table stays SEATED (partial departure)."""
        sm = make_sm(seated_debounce_secs=5.0, empty_hold_off_secs=20.0)
        feed(sm, 3, 0.0)
        feed(sm, 3, 5.0)
        assert sm.state == TableState.SEATED

        # 2 leave, 1 stays
        feed(sm, 1, 6.0)
        feed(sm, 1, 10.0)
        assert sm.state == TableState.SEATED  # still has a guest


# ─── Smoothing ───────────────────────────────────────────────────────────────

class TestSmoothing:
    def test_single_frame_glitch_does_not_trigger_seated(self):
        """A one-frame spike should not start the debounce timer reliably."""
        sm = make_sm(seated_debounce_secs=5.0, smoothing_window=5)
        # 4 frames of 0, then one spike — median of [0,0,0,0,1] = 0
        for t in range(4):
            feed(sm, 0, float(t))
        events = feed(sm, 1, 4.0)
        assert sm.state == TableState.FREE

    def test_smoothing_confirms_real_occupancy(self):
        """Sustained occupancy over the window should register."""
        sm = make_sm(seated_debounce_secs=5.0, smoothing_window=3)
        # Fill window with 1s → median=1 → debounce starts
        feed(sm, 1, 0.0)
        feed(sm, 1, 1.0)
        feed(sm, 1, 2.0)
        events = feed(sm, 1, 6.0)  # 6s since debounce started
        assert sm.state == TableState.SEATED


# ─── Seated duration ─────────────────────────────────────────────────────────

class TestSeatedDuration:
    def test_seconds_seated_is_zero_when_free(self):
        sm = make_sm()
        assert sm.seconds_seated == 0.0

    def test_seconds_seated_resets_on_leave(self):
        sm = make_sm(seated_debounce_secs=5.0, empty_hold_off_secs=10.0)
        feed(sm, 1, 0.0)
        feed(sm, 1, 5.0)
        assert sm.state == TableState.SEATED
        feed(sm, 0, 6.0)
        feed(sm, 0, 16.0)
        assert sm.state == TableState.NEEDS_RESET
        assert sm.seated_at is None


# ─── Full cycle ───────────────────────────────────────────────────────────────

class TestFullCycle:
    def test_complete_table_lifecycle(self):
        """FREE → SEATED → NEEDS_RESET → FREE full cycle."""
        sm = make_sm(seated_debounce_secs=10.0, empty_hold_off_secs=20.0)

        # Guests arrive
        feed(sm, 2, 0.0)
        feed(sm, 2, 10.0)
        assert sm.state == TableState.SEATED

        # Guests leave
        feed(sm, 0, 20.0)
        feed(sm, 0, 40.0)
        assert sm.state == TableState.NEEDS_RESET

        # Staff cleans
        events = sm.mark_reset(now=45.0)
        assert sm.state == TableState.FREE

        # New guests arrive
        feed(sm, 1, 46.0)
        feed(sm, 1, 56.0)
        assert sm.state == TableState.SEATED
