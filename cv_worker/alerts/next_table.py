"""
cv_worker/alerts/next_table.py
--------------------------------
"Next table to free up" suggestion logic.

Predicts which currently occupied table is most likely to become available
next, based on how long each table has been seated.

Heuristic (Phase 2 — simple):
- Longest-seated table is most likely to leave next.
- Tables with fewer people (smaller party) are weighted as slightly
  more likely to turn over sooner.

Phase 5+ will improve this with historical averages per table / party size.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from cv_worker.zone_engine.table_state import TableState, TableStateMachine


@dataclass
class NextTableSuggestion:
    """Suggestion for which table will free up next."""

    table_id: str
    minutes_seated: float
    party_size: int
    confidence: str   # "high", "medium", "low" — qualitative for MVP

    def to_dict(self) -> dict:
        return {
            "table_id": self.table_id,
            "minutes_seated": round(self.minutes_seated, 1),
            "party_size": self.party_size,
            "confidence": self.confidence,
        }


def suggest_next_table(
    state_machines: dict[str, TableStateMachine],
    now: Optional[float] = None,
) -> Optional[NextTableSuggestion]:
    """
    Suggest which occupied table is most likely to free up next.

    Args:
        state_machines: Dict of table_id → TableStateMachine.
        now:            Current timestamp. Defaults to time.time().

    Returns:
        NextTableSuggestion for the best candidate, or None if no tables
        are currently seated.
    """
    import time as _time

    if now is None:
        now = _time.time()

    candidates: list[tuple[str, float, int]] = []  # (table_id, minutes_seated, party_size)

    for table_id, sm in state_machines.items():
        if sm.state != TableState.SEATED or sm.seated_at is None:
            continue
        minutes_seated = (now - sm.seated_at) / 60.0
        candidates.append((table_id, minutes_seated, sm.party_size))

    if not candidates:
        return None

    # Score: longer seated + smaller party = more likely to leave
    # Simple heuristic: score = minutes_seated - (party_size * 5)
    # Higher score → more likely to free up next
    def score(c: tuple[str, float, int]) -> float:
        _, minutes, party = c
        return minutes - (party * 5.0)

    candidates.sort(key=score, reverse=True)
    best = candidates[0]
    table_id, minutes_seated, party_size = best

    # Confidence heuristic
    if minutes_seated > 45:
        confidence = "high"
    elif minutes_seated > 20:
        confidence = "medium"
    else:
        confidence = "low"

    return NextTableSuggestion(
        table_id=table_id,
        minutes_seated=minutes_seated,
        party_size=party_size,
        confidence=confidence,
    )
