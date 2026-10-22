# Claude added 2026-10-22: persist recent day dirs + selected mosaics.
"""Recent-history store for the mosaic browser.

Remembers the day directories visited (most-recent first) and the last mosaic
selected in each, so the tool can be restarted and resume where work left off.
Persisted as JSON under the Application Support directory, matching the rest of
the cn_atlas tool (ground_truth / picked_points / alignments).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

MAX_RECENT = 30


def default_path() -> Path:
    d = Path.home() / "Library" / "Application Support" / "CochlearNucleusAlignment"
    d.mkdir(parents=True, exist_ok=True)
    return d / "mosaic_history.json"


class History:
    """Recent day directories + last-selected mosaic per day."""

    def __init__(self):
        self.recent_days: list[str] = []
        self.last_mosaic: dict[str, str] = {}

    # -- updates -----------------------------------------------------------
    def add_day(self, day_dir) -> None:
        d = str(day_dir)
        if d in self.recent_days:
            self.recent_days.remove(d)
        self.recent_days.insert(0, d)
        del self.recent_days[MAX_RECENT:]

    def add_mosaic(self, day_dir, mosaic_path) -> None:
        self.last_mosaic[str(day_dir)] = str(mosaic_path)

    def mosaic_for(self, day_dir) -> Optional[str]:
        return self.last_mosaic.get(str(day_dir))

    # -- persistence -------------------------------------------------------
    def to_dict(self) -> dict:
        return {"recent_days": self.recent_days, "last_mosaic": self.last_mosaic}

    @classmethod
    def from_dict(cls, d: dict) -> "History":
        h = cls()
        h.recent_days = list(d.get("recent_days", []) or [])
        h.last_mosaic = dict(d.get("last_mosaic", {}) or {})
        return h

    def save(self, path: Optional[Path] = None) -> Path:
        path = Path(path) if path else default_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2))
        return path

    @classmethod
    def load(cls, path: Optional[Path] = None) -> "History":
        path = Path(path) if path else default_path()
        if not path.exists():
            return cls()
        try:
            return cls.from_dict(json.loads(path.read_text()))
        except Exception:  # noqa: BLE001
            return cls()
