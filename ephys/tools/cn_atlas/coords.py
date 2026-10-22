# Claude added 2026-10-22: coordinate reference frames for the section readout.
"""Reference frames for reporting a cursor/world position.

The Kim display world is in micrometres with the origin at a *volume corner*:
+X = M-L measured from the left edge of the volume (0 .. ~11.4 mm), +Y = A-P,
+Z = D-V (0 at the dorsal top, negative ventral). Reporting raw X is confusing
because M-L is then referenced to the left side of the brain rather than the
midline. These frames re-reference the readout:

    Kim-native   raw world coordinates (identity).
    Bregma       translated so the origin is bregma; Kim axis *directions* kept
                 (+X right, +Y as-is, +Z dorsal).
    Stereotaxic  translated to bregma with Paxinos mouse sign conventions:
                 M-L +right/-left of midline, A-P + anterior / - posterior,
                 D-V depth (negative ventral, below bregma). The per-axis signs
                 are configurable because the Kim A-P/M-L axis *direction* is a
                 data convention that may differ (see build_frames).

All frames report (M-L, A-P, D-V) with no axis permutation; they differ only in
origin and per-axis sign. Values are returned in millimetres.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Frame:
    """A reference frame: display = sign * (world_um - origin_um), then -> mm."""

    name: str
    origin: np.ndarray            # world micrometres (x, y, z)
    signs: np.ndarray            # per display axis (ML, AP, DV)
    labels: tuple = ("ML", "AP", "DV")

    def to_mm(self, world_um) -> np.ndarray:
        w = np.asarray(world_um, float).reshape(3)
        return (self.signs * (w - self.origin)) / 1000.0

    def format(self, world_um, bregma_set: bool = True) -> str:
        v = self.to_mm(world_um)
        s = (f"{self.labels[0]}={v[0]:.3f} {self.labels[1]}={v[1]:.3f} "
             f"{self.labels[2]}={v[2]:.3f} mm")
        if self.name != "Kim-native" and not bregma_set:
            s += " (bregma not set)"
        return s


def build_frames(midline_x: float, bregma=None,
                 ml_sign: float = 1.0, ap_sign: float = 1.0,
                 dv_sign: float = 1.0) -> tuple[dict, bool]:
    """Build the three reference frames.

    ``midline_x`` (um) is the M-L world coordinate of the midsagittal plane
    (typically the volume centre). ``bregma`` is the bregma world point
    ([x, y, z] um) if known; when it is None, the origin falls back to the
    midline in M-L (so M-L is at least midline-referenced) and 0 in A-P/D-V,
    and the frames are flagged ``bregma_set = False``. The stereotaxic
    ``*_sign`` values flip an axis if the Kim data direction is opposite the
    Paxinos convention.

    Returns (frames_dict, bregma_set).
    """
    if bregma is not None:
        origin = np.asarray(bregma, float).reshape(3)
        bregma_set = True
    else:
        origin = np.array([float(midline_x), 0.0, 0.0])
        bregma_set = False
    frames = {
        "Kim-native": Frame("Kim-native", np.zeros(3), np.ones(3)),
        "Bregma": Frame("Bregma", origin.copy(), np.ones(3)),
        "Stereotaxic": Frame(
            "Stereotaxic", origin.copy(),
            np.array([float(ml_sign), float(ap_sign), float(dv_sign)])),
    }
    return frames, bregma_set
