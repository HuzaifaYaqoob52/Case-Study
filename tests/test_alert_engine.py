"""
tests/test_alert_engine.py
-----------------------------
Unit tests for the alert engine, bill provider, and next-table suggestion.

Tests use simulated state machine states and timestamps — no video,
no model, no camera needed. All times are injected explicitly.
"""

import pytest

from cv_worker.alerts.alert_engine import (
    Alert,
    AlertEngine,
    AlertSeverity,
    AlertType,
)
from cv_worker.alerts.bill_provider import (
    BillStatus,
    ManualBillProvider,
    MockPOSBillProvider,
)
from cv_worker.alerts.next_table import NextTableSuggestion, suggest_next_table
from cv_worker.zone_engine.table_state import (
    TableState,
    TableStateMachine,
    StateEvent,
)


# ─── Helpers ─────────────────────────────────────────────────────────────────

def make_sm(
    table_id: str = "T1",
    seated_debounce_secs: float = 5.0,
    empty_hold_off_secs: float = 10.0,
    smoothing_window: int = 1,
) -> TableStateMachine:
    """Create a state machine with tight thresholds for fast tests."""
    return TableStateMachine(
        table_id=table_id,
        seated_debounce_secs=seated_debounce_secs,
        empty_hold_off_secs=empty_hold_off_secs,
        smoothing_window=smoothing_window,
    )


def seat_table(sm: TableStateMachine, now: float = 0.0) -> None:
    """Force a state machine into SEATED state by feeding it occupancy."""
    sm.update(1, now=now)
    sm.update(1, now=now + sm.seated_debounce_secs)
    assert sm.state == TableState.SEATED


def vacate_table(sm: TableStateMachine, leave_time: float) -> None:
    """Force a state machine into NEEDS_RESET by emptying the zone."""
    sm.update(0, now=leave_time)
    sm.update(0, now=leave_time + sm.empty_hold_off_secs)
    assert sm.state == TableState.NEEDS_RESET


def make_alert_engine(
    bill_provider=None,
    waiting_minutes: float = 5.0,
    bill_grace_minutes: float = 3.0,
    cooldown_minutes: float = 5.0,
) -> AlertEngine:
    """Create an alert engine with configurable thresholds."""
    return AlertEngine(
        bill_provider=bill_provider or ManualBillProvider(),
        waiting_alert_minutes=waiting_minutes,
        bill_grace_minutes=bill_grace_minutes,
        alert_cooldown_minutes=cooldown_minutes,
    )


# ─── WAITING_TOO_LONG alert ──────────────────────────────────────────────────

class TestWaitingTooLongAlert:
    def test_no_alert_when_not_seated(self):
        """No alert when table is FREE."""
        sm = make_sm()
        engine = make_alert_engine(waiting_minutes=5.0)
        alerts = engine.evaluate({"T1": sm}, now=1000.0)
        assert len(alerts) == 0

    def test_no_alert_before_threshold(self):
        """No alert if seated for less than threshold."""
        sm = make_sm()
        seat_table(sm, now=0.0)
        engine = make_alert_engine(waiting_minutes=5.0)
        # 4 minutes seated — no alert
        alerts = engine.evaluate({"T1": sm}, now=4 * 60.0)
        assert len(alerts) == 0

    def test_alert_fires_after_threshold(self):
        """Alert fires when seated time exceeds threshold."""
        sm = make_sm()
        seat_table(sm, now=0.0)
        engine = make_alert_engine(waiting_minutes=5.0)
        # 6 minutes seated — should alert
        alerts = engine.evaluate({"T1": sm}, now=6 * 60.0)
        assert len(alerts) == 1
        assert alerts[0].alert_type == AlertType.WAITING_TOO_LONG
        assert alerts[0].table_id == "T1"
        assert alerts[0].severity == AlertSeverity.WARNING

    def test_alert_cooldown_prevents_duplicate(self):
        """Same alert should not fire again within cooldown period."""
        sm = make_sm()
        seat_table(sm, now=0.0)
        engine = make_alert_engine(waiting_minutes=5.0, cooldown_minutes=5.0)

        # First alert at 6 min
        alerts = engine.evaluate({"T1": sm}, now=6 * 60.0)
        assert len(alerts) == 1

        # 2 minutes later — should be suppressed by cooldown
        alerts = engine.evaluate({"T1": sm}, now=8 * 60.0)
        assert len(alerts) == 0

    def test_alert_fires_again_after_cooldown(self):
        """Alert should fire again after cooldown expires."""
        sm = make_sm()
        seat_table(sm, now=0.0)
        engine = make_alert_engine(waiting_minutes=5.0, cooldown_minutes=5.0)

        # First alert at 6 min
        engine.evaluate({"T1": sm}, now=6 * 60.0)
        # After cooldown (11 min + 1 sec) — should fire again
        alerts = engine.evaluate({"T1": sm}, now=11 * 60.0 + 1)
        assert len(alerts) == 1

    def test_multiple_tables_get_independent_alerts(self):
        """Each table tracks alerts independently."""
        sm1 = make_sm(table_id="T1")
        sm2 = make_sm(table_id="T2")
        seat_table(sm1, now=0.0)    # seated_at = 5.0 (debounce)
        seat_table(sm2, now=10.0)   # seated_at = 15.0 (debounce)

        engine = make_alert_engine(waiting_minutes=5.0)
        # At t=400: T1 seated 395s (>300s), T2 seated 385s (>300s)
        alerts = engine.evaluate({"T1": sm1, "T2": sm2}, now=400.0)
        assert len(alerts) == 2
        table_ids = {a.table_id for a in alerts}
        assert table_ids == {"T1", "T2"}


# ─── TABLE_NEEDS_RESET alert ─────────────────────────────────────────────────

class TestNeedsResetAlert:
    def test_alert_fires_when_table_enters_needs_reset(self):
        """Alert fires when table transitions to NEEDS_RESET."""
        sm = make_sm()
        seat_table(sm, now=0.0)
        vacate_table(sm, leave_time=20.0)

        engine = make_alert_engine()
        alerts = engine.evaluate({"T1": sm}, now=31.0)
        reset_alerts = [a for a in alerts if a.alert_type == AlertType.TABLE_NEEDS_RESET]
        assert len(reset_alerts) == 1
        assert "vacant" in reset_alerts[0].message.lower() or "clean" in reset_alerts[0].message.lower()

    def test_no_alert_when_free(self):
        """No reset alert when table is FREE."""
        sm = make_sm()
        engine = make_alert_engine()
        alerts = engine.evaluate({"T1": sm}, now=100.0)
        reset_alerts = [a for a in alerts if a.alert_type == AlertType.TABLE_NEEDS_RESET]
        assert len(reset_alerts) == 0

    def test_reset_alert_suppressed_by_cooldown(self):
        """Reset alert does not repeat within cooldown."""
        sm = make_sm()
        seat_table(sm, now=0.0)
        vacate_table(sm, leave_time=20.0)

        engine = make_alert_engine(cooldown_minutes=5.0)
        engine.evaluate({"T1": sm}, now=31.0)
        # Immediately again — should be suppressed
        alerts = engine.evaluate({"T1": sm}, now=32.0)
        reset_alerts = [a for a in alerts if a.alert_type == AlertType.TABLE_NEEDS_RESET]
        assert len(reset_alerts) == 0


# ─── BILL_STILL_OPEN alert ───────────────────────────────────────────────────

class TestBillStillOpenAlert:
    def test_no_alert_if_no_bill(self):
        """No bill alert if there's no bill open."""
        sm = make_sm()
        seat_table(sm, now=0.0)
        vacate_table(sm, leave_time=20.0)

        provider = ManualBillProvider()
        # No bill opened for T1
        engine = make_alert_engine(bill_provider=provider, bill_grace_minutes=1.0)
        # Evaluate after grace period
        alerts = engine.evaluate({"T1": sm}, now=100.0)
        bill_alerts = [a for a in alerts if a.alert_type == AlertType.BILL_STILL_OPEN]
        assert len(bill_alerts) == 0

    def test_no_alert_if_bill_is_paid(self):
        """No bill alert if the bill has been paid."""
        sm = make_sm()
        seat_table(sm, now=0.0)

        provider = ManualBillProvider()
        provider.open_bill("T1")
        provider.mark_paid("T1")

        vacate_table(sm, leave_time=20.0)
        engine = make_alert_engine(bill_provider=provider, bill_grace_minutes=1.0)
        # First evaluate to set needs_reset_since
        engine.evaluate({"T1": sm}, now=31.0)
        # Wait past grace
        alerts = engine.evaluate({"T1": sm}, now=31.0 + 120.0)
        bill_alerts = [a for a in alerts if a.alert_type == AlertType.BILL_STILL_OPEN]
        assert len(bill_alerts) == 0

    def test_no_alert_before_grace_period(self):
        """No alert during the grace period after table empties."""
        sm = make_sm()
        seat_table(sm, now=0.0)

        provider = ManualBillProvider()
        provider.open_bill("T1")

        vacate_table(sm, leave_time=20.0)
        engine = make_alert_engine(bill_provider=provider, bill_grace_minutes=3.0)
        # Just entered NEEDS_RESET
        engine.evaluate({"T1": sm}, now=31.0)
        # 1 minute later — still within grace
        alerts = engine.evaluate({"T1": sm}, now=31.0 + 60.0)
        bill_alerts = [a for a in alerts if a.alert_type == AlertType.BILL_STILL_OPEN]
        assert len(bill_alerts) == 0

    def test_alert_fires_after_grace_with_open_bill(self):
        """Alert fires after grace period when bill is unpaid."""
        sm = make_sm()
        seat_table(sm, now=0.0)

        provider = ManualBillProvider()
        provider.open_bill("T1")

        vacate_table(sm, leave_time=20.0)
        engine = make_alert_engine(bill_provider=provider, bill_grace_minutes=1.0)
        # Set needs_reset_since
        engine.evaluate({"T1": sm}, now=31.0)
        # After grace (1 min = 60s)
        alerts = engine.evaluate({"T1": sm}, now=31.0 + 65.0)
        bill_alerts = [a for a in alerts if a.alert_type == AlertType.BILL_STILL_OPEN]
        assert len(bill_alerts) == 1
        assert "bill still open" in bill_alerts[0].message.lower()

    def test_bill_alert_wording_is_neutral(self):
        """Bill alert must never be accusatory."""
        sm = make_sm()
        seat_table(sm, now=0.0)

        provider = ManualBillProvider()
        provider.open_bill("T1")
        vacate_table(sm, leave_time=20.0)

        # bill_grace=0 means alert can fire immediately after needs_reset_since is set
        engine = make_alert_engine(bill_provider=provider, bill_grace_minutes=0.0)
        # First call: sets needs_reset_since for T1
        engine.evaluate({"T1": sm}, now=31.0)
        # Second call: grace elapsed (0s), bill is OPEN → alert should fire
        # But the first call already fires the alert since grace=0 and
        # needs_reset_since was just set at now=31.0 (31.0 - 31.0 = 0 >= 0)
        # So the first evaluate already fired it. Let's check get_all_alerts.
        all_alerts = engine.get_all_alerts()
        bill_alerts = [a for a in all_alerts if a.alert_type == AlertType.BILL_STILL_OPEN]
        assert len(bill_alerts) >= 1
        msg = bill_alerts[0].message.lower()
        # Must not contain accusatory language
        for bad_word in ["steal", "theft", "dine and dash", "fraud", "skip", "run away"]:
            assert bad_word not in msg, f"Alert contains accusatory word: '{bad_word}'"


# ─── Alert management ────────────────────────────────────────────────────────

class TestAlertManagement:
    def test_acknowledge_alert(self):
        sm = make_sm()
        seat_table(sm, now=0.0)
        engine = make_alert_engine(waiting_minutes=5.0)
        alerts = engine.evaluate({"T1": sm}, now=6 * 60.0)
        assert len(alerts) == 1

        # Acknowledge
        assert engine.acknowledge_alert(alerts[0].id)
        active = engine.get_active_alerts()
        assert len(active) == 0

    def test_acknowledge_nonexistent_alert_returns_false(self):
        engine = make_alert_engine()
        assert not engine.acknowledge_alert("bogus_id")

    def test_clear_alerts_for_table(self):
        sm = make_sm()
        seat_table(sm, now=0.0)
        engine = make_alert_engine(waiting_minutes=5.0)
        engine.evaluate({"T1": sm}, now=6 * 60.0)
        assert len(engine.get_active_alerts()) == 1

        cleared = engine.clear_alerts_for_table("T1")
        assert cleared == 1
        assert len(engine.get_active_alerts()) == 0

    def test_get_active_alerts_filters_by_table(self):
        sm1 = make_sm(table_id="T1")
        sm2 = make_sm(table_id="T2")
        seat_table(sm1, now=0.0)
        seat_table(sm2, now=0.0)
        engine = make_alert_engine(waiting_minutes=5.0)
        engine.evaluate({"T1": sm1, "T2": sm2}, now=6 * 60.0)

        t1_alerts = engine.get_active_alerts(table_id="T1")
        assert len(t1_alerts) == 1
        assert t1_alerts[0].table_id == "T1"


# ─── Bill providers ──────────────────────────────────────────────────────────

class TestManualBillProvider:
    def test_no_bill_by_default(self):
        provider = ManualBillProvider()
        info = provider.get_bill_status("T1")
        assert info.status == BillStatus.NO_BILL

    def test_open_and_pay_bill(self):
        provider = ManualBillProvider()
        provider.open_bill("T1")
        assert provider.get_bill_status("T1").status == BillStatus.OPEN
        provider.mark_paid("T1")
        assert provider.get_bill_status("T1").status == BillStatus.PAID

    def test_close_bill_removes_record(self):
        provider = ManualBillProvider()
        provider.open_bill("T1")
        provider.close_bill("T1")
        assert provider.get_bill_status("T1").status == BillStatus.NO_BILL


class TestMockPOSBillProvider:
    def test_auto_pay_after_delay(self):
        import time

        provider = MockPOSBillProvider(auto_pay_delay_secs=0.1)
        provider.open_bill("T1")
        assert provider.get_bill_status("T1").status == BillStatus.OPEN

        time.sleep(0.15)
        assert provider.get_bill_status("T1").status == BillStatus.PAID

    def test_no_auto_pay_when_disabled(self):
        import time

        provider = MockPOSBillProvider(auto_pay_delay_secs=None)
        provider.open_bill("T1")
        time.sleep(0.1)
        assert provider.get_bill_status("T1").status == BillStatus.OPEN


# ─── Next-table suggestion ───────────────────────────────────────────────────

class TestNextTableSuggestion:
    def test_no_suggestion_when_no_tables_seated(self):
        sm = make_sm()
        result = suggest_next_table({"T1": sm}, now=100.0)
        assert result is None

    def test_single_seated_table_is_suggested(self):
        sm = make_sm()
        seat_table(sm, now=0.0)
        result = suggest_next_table({"T1": sm}, now=30 * 60.0)
        assert result is not None
        assert result.table_id == "T1"
        assert result.minutes_seated == pytest.approx(30.0, abs=1.0)

    def test_longest_seated_table_is_suggested_first(self):
        sm1 = make_sm(table_id="T1")
        sm2 = make_sm(table_id="T2")
        seat_table(sm1, now=0.0)    # seated at t=5 (debounce=5)
        seat_table(sm2, now=600.0)  # seated at t=605

        result = suggest_next_table({"T1": sm1, "T2": sm2}, now=60 * 60.0)
        assert result is not None
        # T1 has been seated longer → should be suggested
        assert result.table_id == "T1"

    def test_smaller_party_breaks_tie(self):
        """When seated duration is similar, smaller party should rank higher."""
        sm1 = make_sm(table_id="T1")
        sm2 = make_sm(table_id="T2")

        # Both seat at same time, but different party sizes
        seat_table(sm1, now=0.0)
        sm1.party_size = 4  # Big party
        seat_table(sm2, now=0.0)
        sm2.party_size = 1  # Solo diner

        result = suggest_next_table({"T1": sm1, "T2": sm2}, now=30 * 60.0)
        assert result is not None
        # T2 has smaller party → higher score
        assert result.table_id == "T2"

    def test_confidence_level_reflects_time(self):
        sm = make_sm()
        seat_table(sm, now=0.0)

        # Short seated time — low confidence
        result = suggest_next_table({"T1": sm}, now=10 * 60.0)
        assert result.confidence == "low"

        # Medium seated time
        result = suggest_next_table({"T1": sm}, now=25 * 60.0)
        assert result.confidence == "medium"

        # Long seated time — high confidence
        result = suggest_next_table({"T1": sm}, now=50 * 60.0)
        assert result.confidence == "high"


# ─── Integration: full alert lifecycle ────────────────────────────────────────

class TestAlertLifecycle:
    def test_full_lifecycle_free_seated_reset_with_alerts(self):
        """
        Simulate a complete table lifecycle and verify all alerts fire
        at the correct times.
        """
        sm = make_sm(
            table_id="T1",
            seated_debounce_secs=5.0,
            empty_hold_off_secs=10.0,
        )
        provider = ManualBillProvider()
        engine = AlertEngine(
            bill_provider=provider,
            waiting_alert_minutes=2.0,    # 2 minutes = 120s
            bill_grace_minutes=1.0,       # 1 minute = 60s
            alert_cooldown_minutes=10.0,  # Long cooldown to avoid duplicates
        )

        # 1. Table starts FREE — no alerts
        alerts = engine.evaluate({"T1": sm}, now=0.0)
        assert len(alerts) == 0

        # 2. Guests arrive → SEATED
        sm.update(2, now=0.0)
        sm.update(2, now=5.0)
        assert sm.state == TableState.SEATED
        provider.open_bill("T1")

        # 3. After 1 minute — no waiting alert yet
        alerts = engine.evaluate({"T1": sm}, now=60.0)
        waiting_alerts = [a for a in alerts if a.alert_type == AlertType.WAITING_TOO_LONG]
        assert len(waiting_alerts) == 0

        # 4. After 3 minutes — waiting alert fires
        alerts = engine.evaluate({"T1": sm}, now=3 * 60.0)
        waiting_alerts = [a for a in alerts if a.alert_type == AlertType.WAITING_TOO_LONG]
        assert len(waiting_alerts) == 1

        # 5. Guests leave → NEEDS_RESET
        sm.update(0, now=4 * 60.0)
        sm.update(0, now=4 * 60.0 + 10.0)
        assert sm.state == TableState.NEEDS_RESET

        # 6. Reset alert fires
        alerts = engine.evaluate({"T1": sm}, now=4 * 60.0 + 11.0)
        reset_alerts = [a for a in alerts if a.alert_type == AlertType.TABLE_NEEDS_RESET]
        assert len(reset_alerts) == 1

        # 7. Bill still open — after grace period
        alerts = engine.evaluate({"T1": sm}, now=4 * 60.0 + 11.0 + 65.0)
        bill_alerts = [a for a in alerts if a.alert_type == AlertType.BILL_STILL_OPEN]
        assert len(bill_alerts) == 1

        # 8. Staff marks paid and resets table
        provider.mark_paid("T1")
        engine.clear_alerts_for_table("T1")
        sm.mark_reset(now=6 * 60.0)
        assert sm.state == TableState.FREE
        assert len(engine.get_active_alerts()) == 0
