# Claude added 2026-10-22: Qt-free model for the preliminary slice-stack alignment.
"""Slice-stack model for the preliminary (pre-atlas) alignment step.

Each assembled mosaic is treated as one brain *slice*. Slices are stacked along
an axis orthogonal to the section plane, spaced by a single section thickness
(per day, default 250 um). Each slice carries a preliminary placement — in-plane
translation + planar rotation, an up/down flip (reflection), and a left/right
side — so slices that were cut flipped or rotated relative to each other can be
brought into register before being handed to the atlas aligner.

The whole-stack also records an approximate section-plane preset (cardinal or a
CN-specific tilt). That preset is *not* used to render anything here; it is
transferred to the main atlas program to orient the sections, keyed to the
fiducial points.

This module is Qt-free so the geometry can be tested headless; the 3D view and
controls live in ``alignment.py``.
"""

from __future__ import annotations

import colorsys
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

# Approximate section-plane presets. Cardinal planes carry no tilt; the two
# CN-specific planes carry named tilt angles (degrees) used downstream by the
# atlas aligner, not for display here.
PLANE_PRESETS: dict[str, dict] = {
    "Horizontal": {},
    "Coronal": {},
    "Sagittal": {},
    # tilted caudally on the dorsal end by 25 deg, and off the horizontal plane
    # by 35 deg (lateral positive / up)
    "transstrial-CN": {"caudal_dorsal_tilt": 25.0, "off_horizontal_tilt": 35.0},
    # tilted off the sagittal axis by 25 deg, dorsal aspect toward the midline
    "sagittal-CN": {"off_sagittal_tilt": 25.0},
}

DEFAULT_THICKNESS_UM = 250.0


@dataclass
class SliceEntry:
    """One slice (an assembled mosaic) plus its preliminary placement."""

    asm: object                  # AssembledMosaic
    name: str
    flip_ud: bool = False        # reflect the slice top<->bottom
    rot_deg: float = 0.0         # in-plane (planar) rotation, degrees
    side: str = "Right"          # "Left" or "Right"
    tx: float = 0.0              # in-plane translation, metres
    ty: float = 0.0
    image_on_top: bool = True    # draw the slice image on the top or bottom face
    image_alpha: float = 1.0     # per-slice image opacity (0..1)
    visible: bool = True         # show/hide this slice in the 3D view
    center: tuple = (0.0, 0.0)   # raw-world centroid (metres) for about-centre rot
    fiducials: list = field(default_factory=list)  # [{"label","pos":[x,y,z]}]

    def placement_affine(self) -> np.ndarray:
        """3x3 in-plane affine mapping raw world -> placed world.

        Mirror (side / up-down) and rotation are taken about the slice's own
        ``center`` so the slice spins in place; the user translation is applied
        last in world coordinates. A "Left" side mirrors about x (so left/right
        slices overlay); ``flip_ud`` mirrors about y (top/bottom reflection).
        """
        cx, cy = self.center
        th = np.deg2rad(self.rot_deg)
        c, s = np.cos(th), np.sin(th)
        rot = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
        mx = -1.0 if self.side == "Left" else 1.0
        my = -1.0 if self.flip_ud else 1.0
        mir = np.array([[mx, 0.0, 0.0], [0.0, my, 0.0], [0.0, 0.0, 1.0]])
        to0 = np.array([[1.0, 0.0, -cx], [0.0, 1.0, -cy], [0.0, 0.0, 1.0]])
        back = np.array([[1.0, 0.0, cx], [0.0, 1.0, cy], [0.0, 0.0, 1.0]])
        user = np.array([[1.0, 0.0, self.tx], [0.0, 1.0, self.ty], [0.0, 0.0, 1.0]])
        return user @ back @ rot @ mir @ to0


class SliceStack:
    """Ordered stack of slices with a single thickness and a plane preset."""

    def __init__(self, thickness_um: float = DEFAULT_THICKNESS_UM):
        self.entries: list[SliceEntry] = []
        self.thickness_um: float = float(thickness_um)
        self.reverse_colors: bool = False
        self.plane_preset: str = "Horizontal"

    # -- membership --------------------------------------------------------
    def add(self, asm, name: Optional[str] = None) -> SliceEntry:
        e = SliceEntry(asm=asm, name=name or getattr(asm, "path", "slice"))
        self.entries.append(e)
        return e

    def remove(self, idx: int):
        if 0 <= idx < len(self.entries):
            del self.entries[idx]

    def clear(self):
        self.entries = []

    def move(self, idx: int, delta: int) -> int:
        """Move entry at ``idx`` by ``delta`` positions; returns the new index."""
        j = idx + delta
        if 0 <= idx < len(self.entries) and 0 <= j < len(self.entries):
            self.entries[idx], self.entries[j] = self.entries[j], self.entries[idx]
            return j
        return idx

    # -- geometry ----------------------------------------------------------
    @property
    def thickness_m(self) -> float:
        return self.thickness_um * 1e-6

    def z_of(self, idx: int) -> float:
        """World z (metres) of the lower face of slice ``idx``'s thickness slab."""
        return idx * self.thickness_m

    def color_for(self, idx: int) -> tuple:
        """Spectral RGB (0-255) for slice ``idx`` across the stack (reversible)."""
        n = max(len(self.entries), 1)
        t = idx / max(n - 1, 1)
        if self.reverse_colors:
            t = 1.0 - t
        # violet (dorsal of spectrum) -> red: hue 0.75 .. 0.0
        hue = 0.75 * (1.0 - t)
        r, g, b = colorsys.hsv_to_rgb(hue, 0.9, 1.0)
        return int(r * 255), int(g * 255), int(b * 255)

    def preset_descriptor(self) -> dict:
        """The plane preset + its tilt angles, for hand-off to the atlas aligner."""
        return {"preset": self.plane_preset,
                "angles": dict(PLANE_PRESETS.get(self.plane_preset, {}))}
