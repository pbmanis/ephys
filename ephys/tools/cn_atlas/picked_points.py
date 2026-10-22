# Claude added 2026-10-22: dedicated store for cursor-picked alignment points.
"""User-picked points on section plates (adjunct for alignment).

A left-click on a section plate (when not drawing a fiducial line, contour, or
landmark) records the 3D world position of the cursor here. These points are
kept separate from ground-truth landmarks and fiducial lines; they are intended
to seed later image/atlas alignment. Each point stores the world coordinate
(um), the structure under the cursor (if any), and the section descriptor of the
plate it was picked on. Persisted as JSON under the Application Support dir.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional


def default_path() -> Path:
    d = Path.home() / "Library" / "Application Support" / "CochlearNucleusAlignment"
    d.mkdir(parents=True, exist_ok=True)
    return d / "picked_points.json"


class PickedPoints:
    """Container of user-picked 3D world points for alignment."""

    def __init__(self):
        self.points: list[dict] = []   # [{"id","point":[x,y,z],"region","section","timestamp"}]
        self._next_id = 1

    def _new_id(self) -> int:
        i = self._next_id
        self._next_id += 1
        return i

    def add_point(self, point, region: str = "", section: Optional[dict] = None) -> dict:
        p = {"id": self._new_id(),
             "point": [float(point[0]), float(point[1]), float(point[2])],
             "region": region or "",
             "section": dict(section) if section else {},
             "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}
        self.points.append(p)
        return p

    def remove_point(self, pid: int):
        self.points = [p for p in self.points if p["id"] != pid]

    def clear(self):
        self.points = []

    # -- persistence -------------------------------------------------------
    def to_dict(self) -> dict:
        return {"saved": time.strftime("%Y-%m-%d %H:%M:%S"),
                "points": self.points, "next_id": self._next_id}

    @classmethod
    def from_dict(cls, d: dict) -> "PickedPoints":
        pp = cls()
        pp.points = d.get("points", []) or []
        ids = [p["id"] for p in pp.points]
        pp._next_id = max(ids, default=0) + 1
        if "next_id" in d:
            pp._next_id = max(pp._next_id, int(d["next_id"]))
        return pp

    def save(self, path: Optional[Path] = None) -> Path:
        path = Path(path) if path else default_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2))
        return path

    @classmethod
    def load(cls, path: Optional[Path] = None) -> "PickedPoints":
        path = Path(path) if path else default_path()
        if not path.exists():
            return cls()
        try:
            return cls.from_dict(json.loads(path.read_text()))
        except Exception:  # noqa: BLE001
            return cls()
