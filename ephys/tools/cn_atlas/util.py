"""Small shared helpers for the cn_atlas subpackage."""

from __future__ import annotations

try:  # colored console output, matching the rest of the ephys tools
    import pylibrary.tools.cprint as CP

    def cprint(color: str, msg: str) -> None:
        CP.cprint(color, msg)

except Exception:  # pragma: no cover - pylibrary is normally present

    def cprint(color: str, msg: str) -> None:
        print(msg)
