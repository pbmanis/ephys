"""User-drawn ground-truth annotations (adjunct to the Kim atlas).

The Kim atlas is left untouched; corrected region boundaries and landmarks are
stored separately here. A *region* is a set of closed contours, each drawn on a
section plate and stored in 3D world coordinates (plus the section descriptor it
was drawn on, so it can be matched back to that plate). A *landmark* is a single
named 3D point. Persisted as JSON under the Application Support directory.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional

import numpy as np


def default_path() -> Path:
    d = Path.home() / "Library" / "Application Support" / "CochlearNucleusAlignment"
    d.mkdir(parents=True, exist_ok=True)
    return d / "ground_truth.json"


def _ptlist(points) -> list:
    return [[float(p[0]), float(p[1]), float(p[2])] for p in points]


class GroundTruth:
    """Container of user regions (contours) and point landmarks."""

    def __init__(self):
        self.regions: dict[str, dict] = {}   # name -> {"color":[r,g,b], "contours":[...]}
        self.landmarks: list[dict] = []      # [{"id","name","point":[x,y,z],"color"}]
        self._next_id = 1

    # -- ids ---------------------------------------------------------------
    def _new_id(self) -> int:
        i = self._next_id
        self._next_id += 1
        return i

    # -- regions / contours ------------------------------------------------
    def add_region(self, name: str, color=(255, 230, 0)):
        if name and name not in self.regions:
            self.regions[name] = {"color": [int(c) for c in color], "contours": []}

    def remove_region(self, name: str):
        self.regions.pop(name, None)

    def add_contour(self, region: str, points, section: dict) -> dict:
        self.add_region(region)
        c = {"id": self._new_id(), "points": _ptlist(points), "section": dict(section)}
        self.regions[region]["contours"].append(c)
        return c

    def update_contour(self, region: str, cid: int, points):
        for c in self.regions.get(region, {}).get("contours", []):
            if c["id"] == cid:
                c["points"] = _ptlist(points)
                return

    def remove_contour(self, region: str, cid: int):
        cs = self.regions.get(region, {}).get("contours", [])
        self.regions[region]["contours"] = [c for c in cs if c["id"] != cid]

    # -- landmarks ---------------------------------------------------------
    def add_landmark(self, name: str, point, color=(255, 255, 0)) -> dict:
        lm = {"id": self._new_id(), "name": name,
              "point": [float(point[0]), float(point[1]), float(point[2])],
              "color": [int(c) for c in color]}
        self.landmarks.append(lm)
        return lm

    def remove_landmark(self, lid: int):
        self.landmarks = [lm for lm in self.landmarks if lm["id"] != lid]

    # -- persistence -------------------------------------------------------
    def to_dict(self) -> dict:
        return {"saved": time.strftime("%Y-%m-%d %H:%M:%S"),
                "regions": self.regions, "landmarks": self.landmarks,
                "next_id": self._next_id}

    @classmethod
    def from_dict(cls, d: dict) -> "GroundTruth":
        gt = cls()
        gt.regions = d.get("regions", {}) or {}
        gt.landmarks = d.get("landmarks", []) or []
        used = [gt.regions[r]["contours"] for r in gt.regions]
        ids = [c["id"] for cs in used for c in cs] + [lm["id"] for lm in gt.landmarks]
        gt._next_id = max(ids, default=0) + 1
        if "next_id" in d:
            gt._next_id = max(gt._next_id, int(d["next_id"]))
        return gt

    def save(self, path: Optional[Path] = None) -> Path:
        path = Path(path) if path else default_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2))
        return path

    @classmethod
    def load(cls, path: Optional[Path] = None) -> "GroundTruth":
        path = Path(path) if path else default_path()
        if not path.exists():
            return cls()
        try:
            return cls.from_dict(json.loads(path.read_text()))
        except Exception:  # noqa: BLE001
            return cls()
