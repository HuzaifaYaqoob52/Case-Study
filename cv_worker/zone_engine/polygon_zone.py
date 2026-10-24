"""
cv_worker/zone_engine/polygon_zone.py
---------------------------------------
Table zone definitions and person-in-zone detection.

Each TableZone wraps a named polygon and tracks which track IDs are
currently inside it. Uses Shapely for robust point-in-polygon testing.

Key design decisions:
- We test the **centre point** of each bounding box against the polygon.
  For overhead cameras this is ideal. For angled cameras, the foot point
  (bottom-centre) is better — set use_foot_point=True per zone.
- A track ID is "in zone" if its test point falls strictly inside or on
  the polygon boundary.
- Zone hit counts are computed fresh each frame; temporal smoothing is
  handled in the pipeline layer.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from loguru import logger
from shapely.geometry import Point, Polygon


@dataclass
class TableZone:
    """
    A named table zone defined by a convex or concave polygon.

    Attributes:
        id:             Unique table identifier, e.g. 'T1'.
        name:           Human-readable label, e.g. 'Table 1'.
        polygon:        Shapely Polygon representing the table zone.
        use_foot_point: If True, test bottom-centre of bbox instead of centre.
                        Prefer True for angled/side-view cameras.
    """

    id: str
    name: str
    polygon: Polygon
    use_foot_point: bool = False

    @classmethod
    def from_dict(cls, data: dict, use_foot_point: bool = False) -> "TableZone":
        """Build a TableZone from a dict like the ones in config/tables.json."""
        pts = [tuple(p) for p in data["polygon"]]
        if len(pts) < 3:
            raise ValueError(
                f"Table '{data['id']}' polygon must have at least 3 points."
            )
        return cls(
            id=data["id"],
            name=data.get("name", data["id"]),
            polygon=Polygon(pts),
            use_foot_point=use_foot_point,
        )

    def contains(self, person) -> bool:
        """
        Return True if a TrackedPerson is inside this zone.

        Args:
            person: A TrackedPerson with .center and .foot_point properties.
        """
        pt = person.foot_point if self.use_foot_point else person.center
        return self.polygon.contains(Point(pt)) or self.polygon.boundary.contains(Point(pt))

    def polygon_points(self) -> list[tuple[int, int]]:
        """Return polygon vertices as integer (x, y) tuples for OpenCV drawing."""
        return [(int(x), int(y)) for x, y in self.polygon.exterior.coords[:-1]]


def load_zones_from_config(config_path: str | Path, use_foot_point: bool = False) -> list[TableZone]:
    """
    Load table zone definitions from the JSON config file.

    Args:
        config_path:    Path to config/tables.json.
        use_foot_point: Passed to each TableZone.

    Returns:
        List of TableZone objects.

    Raises:
        FileNotFoundError if the file does not exist.
        ValueError if the JSON is malformed.
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Table config not found: {path}. "
            "Copy config/tables.json.example and update polygons to match your camera."
        )

    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    tables = data.get("tables", [])
    if not tables:
        raise ValueError(f"No tables defined in {path}.")

    zones = [TableZone.from_dict(t, use_foot_point=use_foot_point) for t in tables]
    logger.info(f"Loaded {len(zones)} table zone(s) from {path}.")
    return zones


@dataclass
class ZoneOccupancy:
    """
    Result of a zone-check pass for a single frame.

    Attributes:
        zone_id:    Table zone ID.
        track_ids:  Set of track IDs currently inside the zone.
    """

    zone_id: str
    track_ids: set[int] = field(default_factory=set)

    @property
    def count(self) -> int:
        return len(self.track_ids)

    @property
    def is_occupied(self) -> bool:
        return self.count > 0


def compute_zone_occupancies(
    zones: list[TableZone],
    tracked_persons: list,  # list[TrackedPerson]
) -> dict[str, ZoneOccupancy]:
    """
    For each zone, determine which tracked persons are inside it.

    A person is assigned to the zone whose polygon contains their test point.
    If a person is inside multiple overlapping zones, they are counted in all
    of them. (Overlapping zones are not expected in normal restaurant configs.)

    Args:
        zones:           List of TableZone objects.
        tracked_persons: List of TrackedPerson objects from the tracker.

    Returns:
        Dict mapping zone_id → ZoneOccupancy.
    """
    occupancies: dict[str, ZoneOccupancy] = {z.id: ZoneOccupancy(zone_id=z.id) for z in zones}

    for person in tracked_persons:
        for zone in zones:
            if zone.contains(person):
                occupancies[zone.id].track_ids.add(person.track_id)

    return occupancies
