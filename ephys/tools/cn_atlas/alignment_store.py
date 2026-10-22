"""Persist Muniak->Kim alignments between sessions.

Alignments are appended (with a timestamp) to a JSON file under the macOS
Application Support directory, so a fine-tuned alignment can be recalled later.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional


def store_path() -> Path:
    d = Path.home() / "Library" / "Application Support" / "CochlearNucleusAlignment"
    d.mkdir(parents=True, exist_ok=True)
    return d / "alignments.json"


def _load_all() -> list:
    path = store_path()
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text())
        return data if isinstance(data, list) else []
    except Exception:  # noqa: BLE001
        return []


def save_alignment(record: dict) -> Path:
    """Append ``record`` (with a timestamp) and return the file path."""
    data = _load_all()
    rec = dict(record)
    rec["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")
    data.append(rec)
    path = store_path()
    path.write_text(json.dumps(data, indent=2))
    return path


def load_latest() -> Optional[dict]:
    """Return the most recently saved alignment, or None."""
    data = _load_all()
    return data[-1] if data else None
