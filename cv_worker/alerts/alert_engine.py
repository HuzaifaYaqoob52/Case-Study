"""
cv_worker/alerts/alert_engine.py
----------------------------------
Rule-based alert engine for the restaurant table monitoring system.

Alerts:
- WAITING_TOO_LONG:  Table seated for > N minutes with no staff visit.
- TABLE_NEEDS_RESET: Guests left — staff should clean and reset the table.
- BILL_STILL_OPEN:   Table vacated but bill is unpaid after grace period.
                     Wording is neutral; never accusatory.

Design:
- All thresholds and cooldowns are configurable.
- Duplicate alerts are suppressed via cooldown timers per (table, alert_type).
- Alerts have an "acknowledged" flag so staff can dismiss them.
- The engine is stateless except for cooldown/ack tracking — all input
  comes from the state machines and bill provider.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from loguru import logger

from cv_worker.alerts.bill_provider import BaseBillProvider, BillStatus
from cv_worker.zone_engine.table_state import TableState, TableStateMachine


class AlertType(str, Enum):
    WAITING_TOO_LONG = "WAITING_TOO_LONG"
    TABLE_NEEDS_RESET = "TABLE_NEEDS_RESET"
    BILL_STILL_OPEN = "BILL_STILL_OPEN"


class AlertSeverity(str, Enum):
    INFO = "INFO"
    WARNING = "WARNING"
    URGENT = "URGENT"


@dataclass
class Alert:
    """A single alert emitted by the engine."""

    id: str                        # Unique alert ID: "{table_id}_{type}_{timestamp}"
    table_id: str
    alert_type: AlertType
    severity: AlertSeverity
    message: str
    timestamp: float               # Unix timestamp when alert was created
    acknowledged: bool = False
    acknowledged_at: Optional[float] = None

    def acknowledge(self, now: Optional[float] = None) -> None:
        self.acknowledged = True
        self.acknowledged_at = now or time.time()

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "table_id": self.table_id,
            "alert_type": self.alert_type.value,
            "severity": self.severity.value,
            "message": self.message,
            "timestamp": self.timestamp,
            "acknowledged": self.acknowledged,
            "acknowledged_at": self.acknowledged_at,
        }


class AlertEngine:
    """
    Evaluates alert rules against current table states each processing cycle.

    Args:
        bill_provider:          BillProvider for checking bill status.
        waiting_alert_minutes:  Minutes before a WAITING_TOO_LONG alert fires.
        bill_grace_minutes:     Grace period (minutes) after table empties before
                                a BILL_STILL_OPEN alert fires.
        alert_cooldown_minutes: Minimum minutes between duplicate alerts for
                                the same (table, alert_type) pair.
    """

    def __init__(
        self,
        bill_provider: BaseBillProvider,
        waiting_alert_minutes: float = 5.0,
        bill_grace_minutes: float = 3.0,
        alert_cooldown_minutes: float = 5.0,
    ) -> None:
        self.bill_provider = bill_provider
        self.waiting_alert_secs = waiting_alert_minutes * 60.0
        self.bill_grace_secs = bill_grace_minutes * 60.0
        self.alert_cooldown_secs = alert_cooldown_minutes * 60.0

        # Cooldown tracker: (table_id, AlertType) → last alert timestamp
        self._cooldowns: dict[tuple[str, AlertType], float] = {}

        # Active alerts — kept for the dashboard (Phase 4)
        self._active_alerts: list[Alert] = []

        # Track when each table entered NEEDS_RESET for bill grace timing
        self._needs_reset_since: dict[str, float] = {}

    # ─── Main evaluation ────────────────────────────────────────────────────

    def evaluate(
        self,
        state_machines: dict[str, TableStateMachine],
        now: Optional[float] = None,
    ) -> list[Alert]:
        """
        Run all alert rules against the current table states.

        Call this once per processing cycle (e.g., every frame or every N frames).

        Args:
            state_machines: Dict of table_id → TableStateMachine.
            now:            Current timestamp. Defaults to time.time().

        Returns:
            List of newly fired alerts (may be empty).
        """
        if now is None:
            now = time.time()

        new_alerts: list[Alert] = []

        for table_id, sm in state_machines.items():
            # Rule 1: WAITING_TOO_LONG
            alert = self._check_waiting_too_long(table_id, sm, now)
            if alert:
                new_alerts.append(alert)

            # Rule 2: TABLE_NEEDS_RESET
            alert = self._check_needs_reset(table_id, sm, now)
            if alert:
                new_alerts.append(alert)

            # Rule 3: BILL_STILL_OPEN
            alert = self._check_bill_still_open(table_id, sm, now)
            if alert:
                new_alerts.append(alert)

        return new_alerts

    # ─── Alert rules ────────────────────────────────────────────────────────

    def _check_waiting_too_long(
        self, table_id: str, sm: TableStateMachine, now: float
    ) -> Optional[Alert]:
        """
        Fire if table has been SEATED for longer than the waiting threshold.
        """
        if sm.state != TableState.SEATED:
            return None

        if sm.seated_at is None:
            return None

        seated_duration = now - sm.seated_at
        if seated_duration < self.waiting_alert_secs:
            return None

        if self._is_on_cooldown(table_id, AlertType.WAITING_TOO_LONG, now):
            return None

        minutes = int(seated_duration / 60)
        alert = Alert(
            id=f"{table_id}_WAITING_{int(now)}",
            table_id=table_id,
            alert_type=AlertType.WAITING_TOO_LONG,
            severity=AlertSeverity.WARNING,
            message=(
                f"{table_id} has been seated for {minutes} minute(s) "
                f"with no staff visit. Please check on them."
            ),
            timestamp=now,
        )
        self._set_cooldown(table_id, AlertType.WAITING_TOO_LONG, now)
        self._active_alerts.append(alert)
        return alert

    def _check_needs_reset(
        self, table_id: str, sm: TableStateMachine, now: float
    ) -> Optional[Alert]:
        """
        Fire once when a table transitions to NEEDS_RESET.
        """
        if sm.state == TableState.NEEDS_RESET:
            # Track the moment we enter NEEDS_RESET
            if table_id not in self._needs_reset_since:
                self._needs_reset_since[table_id] = now
        else:
            # Clear tracking when table leaves NEEDS_RESET
            self._needs_reset_since.pop(table_id, None)
            return None

        if self._is_on_cooldown(table_id, AlertType.TABLE_NEEDS_RESET, now):
            return None

        alert = Alert(
            id=f"{table_id}_RESET_{int(now)}",
            table_id=table_id,
            alert_type=AlertType.TABLE_NEEDS_RESET,
            severity=AlertSeverity.INFO,
            message=f"{table_id} is now vacant. Please clean and reset the table.",
            timestamp=now,
        )
        self._set_cooldown(table_id, AlertType.TABLE_NEEDS_RESET, now)
        self._active_alerts.append(alert)
        return alert

    def _check_bill_still_open(
        self, table_id: str, sm: TableStateMachine, now: float
    ) -> Optional[Alert]:
        """
        Fire if table is in NEEDS_RESET (guests left), the bill is still
        unpaid, and the grace period has elapsed.

        Alert wording is deliberately neutral — never accusatory.
        """
        if sm.state != TableState.NEEDS_RESET:
            return None

        # Check how long the table has been in NEEDS_RESET
        reset_since = self._needs_reset_since.get(table_id)
        if reset_since is None:
            return None

        # Wait for the grace period before alerting
        if now - reset_since < self.bill_grace_secs:
            return None

        # Check bill status
        bill_info = self.bill_provider.get_bill_status(table_id)
        if bill_info.status != BillStatus.OPEN:
            return None

        if self._is_on_cooldown(table_id, AlertType.BILL_STILL_OPEN, now):
            return None

        alert = Alert(
            id=f"{table_id}_BILL_{int(now)}",
            table_id=table_id,
            alert_type=AlertType.BILL_STILL_OPEN,
            severity=AlertSeverity.URGENT,
            message=f"{table_id} vacated, bill still open. Please verify payment status.",
            timestamp=now,
        )
        self._set_cooldown(table_id, AlertType.BILL_STILL_OPEN, now)
        self._active_alerts.append(alert)
        return alert

    # ─── Alert management ───────────────────────────────────────────────────

    def acknowledge_alert(self, alert_id: str, now: Optional[float] = None) -> bool:
        """
        Acknowledge an alert by ID. Returns True if found and acknowledged.
        """
        for alert in self._active_alerts:
            if alert.id == alert_id:
                alert.acknowledge(now)
                return True
        return False

    def get_active_alerts(self, table_id: Optional[str] = None) -> list[Alert]:
        """
        Get all unacknowledged alerts, optionally filtered by table.
        """
        alerts = [a for a in self._active_alerts if not a.acknowledged]
        if table_id:
            alerts = [a for a in alerts if a.table_id == table_id]
        return alerts

    def get_all_alerts(self) -> list[Alert]:
        """Get all alerts (including acknowledged)."""
        return list(self._active_alerts)

    def clear_alerts_for_table(self, table_id: str) -> int:
        """
        Remove all alerts for a table (e.g., after table is reset to FREE).
        Returns count of cleared alerts.
        """
        before = len(self._active_alerts)
        self._active_alerts = [
            a for a in self._active_alerts if a.table_id != table_id
        ]
        cleared = before - len(self._active_alerts)
        if cleared:
            logger.debug(f"Cleared {cleared} alert(s) for {table_id}")
        return cleared

    # ─── Cooldown helpers ───────────────────────────────────────────────────

    def _is_on_cooldown(
        self, table_id: str, alert_type: AlertType, now: float
    ) -> bool:
        key = (table_id, alert_type)
        last = self._cooldowns.get(key)
        if last is None:
            return False
        return (now - last) < self.alert_cooldown_secs

    def _set_cooldown(
        self, table_id: str, alert_type: AlertType, now: float
    ) -> None:
        self._cooldowns[(table_id, alert_type)] = now
