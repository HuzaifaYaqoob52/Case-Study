"""
cv_worker/alerts/bill_provider.py
-----------------------------------
Abstract bill-status interface and concrete implementations.

Design:
- BaseBillProvider defines a single method: get_bill_status(table_id) → BillStatus.
- ManualBillProvider: staff marks tables paid via the staff web app (Phase 4).
  For now, state is held in memory with a simple dict.
- MockPOSBillProvider: simulated POS data for testing. Configurable to
  auto-mark bills as paid after a delay.

When a real POS integration arrives (Phase 5+), implement a new subclass
(e.g. SquareBillProvider) — no other code changes needed.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from loguru import logger


class BillStatus(str, Enum):
    """Status of a table's bill."""
    OPEN = "OPEN"          # Bill exists and is unpaid
    PAID = "PAID"          # Bill has been paid
    NO_BILL = "NO_BILL"    # No bill opened (table is free / pre-order)


@dataclass
class BillInfo:
    """Snapshot of bill state for a single table."""
    table_id: str
    status: BillStatus
    opened_at: Optional[float] = None   # Unix timestamp when bill was opened
    paid_at: Optional[float] = None     # Unix timestamp when bill was paid
    amount: Optional[float] = None      # Total bill amount (from POS)


class BaseBillProvider(ABC):
    """
    Abstract interface for checking a table's bill status.

    All implementations must be safe to call frequently (once per frame
    or once per alert-check cycle). Heavy lookups should be cached.
    """

    @abstractmethod
    def get_bill_status(self, table_id: str) -> BillInfo:
        """
        Return the current bill status for a table.

        Args:
            table_id: Table identifier, e.g. 'T1'.

        Returns:
            BillInfo with at least the status field populated.
        """
        ...

    @abstractmethod
    def mark_paid(self, table_id: str) -> None:
        """
        Mark a table's bill as paid.

        Args:
            table_id: Table identifier.
        """
        ...

    @abstractmethod
    def open_bill(self, table_id: str) -> None:
        """
        Open a new bill for a table (when guests are seated).

        Args:
            table_id: Table identifier.
        """
        ...

    @abstractmethod
    def close_bill(self, table_id: str) -> None:
        """
        Close / clear a table's bill record (after table is reset).

        Args:
            table_id: Table identifier.
        """
        ...


class ManualBillProvider(BaseBillProvider):
    """
    In-memory bill tracker where staff marks tables paid manually.

    State is lost on restart — this is intentional for MVP.
    Phase 3 will persist bill state in the database.
    """

    def __init__(self) -> None:
        # table_id → BillInfo
        self._bills: dict[str, BillInfo] = {}

    def get_bill_status(self, table_id: str) -> BillInfo:
        if table_id in self._bills:
            return self._bills[table_id]
        return BillInfo(table_id=table_id, status=BillStatus.NO_BILL)

    def open_bill(self, table_id: str) -> None:
        now = time.time()
        self._bills[table_id] = BillInfo(
            table_id=table_id,
            status=BillStatus.OPEN,
            opened_at=now,
        )
        logger.debug(f"Bill opened for {table_id}")

    def mark_paid(self, table_id: str) -> None:
        if table_id in self._bills:
            self._bills[table_id].status = BillStatus.PAID
            self._bills[table_id].paid_at = time.time()
            logger.debug(f"Bill marked PAID for {table_id}")
        else:
            logger.warning(f"mark_paid called for {table_id} but no bill exists")

    def close_bill(self, table_id: str) -> None:
        if table_id in self._bills:
            del self._bills[table_id]
            logger.debug(f"Bill closed for {table_id}")


class MockPOSBillProvider(BaseBillProvider):
    """
    Simulated POS for testing. All seated tables get an open bill.
    Bills auto-pay after `auto_pay_delay_secs` seconds (default: never).

    Args:
        auto_pay_delay_secs: If set, bills are automatically marked paid
                             this many seconds after being opened.
                             None = bills stay open until mark_paid().
    """

    def __init__(self, auto_pay_delay_secs: Optional[float] = None) -> None:
        self.auto_pay_delay_secs = auto_pay_delay_secs
        self._bills: dict[str, BillInfo] = {}

    def get_bill_status(self, table_id: str) -> BillInfo:
        if table_id not in self._bills:
            return BillInfo(table_id=table_id, status=BillStatus.NO_BILL)

        bill = self._bills[table_id]

        # Auto-pay simulation
        if (
            bill.status == BillStatus.OPEN
            and self.auto_pay_delay_secs is not None
            and bill.opened_at is not None
            and time.time() - bill.opened_at >= self.auto_pay_delay_secs
        ):
            bill.status = BillStatus.PAID
            bill.paid_at = time.time()
            logger.debug(f"[MockPOS] Auto-paid bill for {table_id}")

        return bill

    def open_bill(self, table_id: str) -> None:
        now = time.time()
        self._bills[table_id] = BillInfo(
            table_id=table_id,
            status=BillStatus.OPEN,
            opened_at=now,
            amount=round(15.0 + (hash(table_id) % 50), 2),  # Fake amount
        )
        logger.debug(f"[MockPOS] Bill opened for {table_id}")

    def mark_paid(self, table_id: str) -> None:
        if table_id in self._bills:
            self._bills[table_id].status = BillStatus.PAID
            self._bills[table_id].paid_at = time.time()
            logger.debug(f"[MockPOS] Bill marked PAID for {table_id}")

    def close_bill(self, table_id: str) -> None:
        if table_id in self._bills:
            del self._bills[table_id]
            logger.debug(f"[MockPOS] Bill closed for {table_id}")
