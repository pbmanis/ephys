# Claude added 2026-10-22: Qt-free reader/assembler for acq4 .mosaic files.
"""Read and assemble acq4 ``.mosaic`` files.

A ``.mosaic`` file is JSON with a ``rootPath``, a list of ``items`` and a saved
``view``. The items we care about:

    ImageCanvasItem   a tiled image. ``filename`` is relative to ``rootPath``;
                      the real physical placement lives in the tiff's directory
                      ``.index`` metadata (``transform``: pos [m], scale [m/px,
                      y usually negative], angle [deg]). The mosaic item's own
                      ``userTransform`` (pos/scale/angle) is an additional nudge
                      applied on top (usually identity).
    MarkersCanvasItem a named set of anatomical landmarks. ``name`` is the
                      section type (e.g. "Cortex Coronal", "DCN Transstrial");
                      ``markers`` is [label, [x, y, z]] in metres.
    CellCanvasItem    a cell location (``userTransform.pos``, metres).
    RulerCanvasItem   a measurement polyline.

Placement is returned as a 3x3 numpy affine mapping image pixel (col, row) to
world metres, so this module is Qt-free and testable headless. Tile images are
loaded with :class:`ephys.datareaders.acq4_reader.acq4_reader` (tiff + .ma),
which already reads the per-image ``.index`` transform.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from ephys.datareaders import acq4_reader

MOSAIC_EXT = ".mosaic"


def _srt_affine(pos, scale, angle_deg: float) -> np.ndarray:
    """3x3 affine M = T(pos) @ R(angle) @ S(scale) for a 2D SRT transform."""
    px, py = float(pos[0]), float(pos[1])
    sx, sy = float(scale[0]), float(scale[1])
    th = np.deg2rad(float(angle_deg))
    c, s = np.cos(th), np.sin(th)
    T = np.array([[1.0, 0.0, px], [0.0, 1.0, py], [0.0, 0.0, 1.0]])
    R = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    S = np.array([[sx, 0.0, 0.0], [0.0, sy, 0.0], [0.0, 0.0, 1.0]])
    return T @ R @ S


def _apply_affine(M: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Apply 3x3 affine ``M`` to [N,2] points, returning [N,2]."""
    pts = np.atleast_2d(np.asarray(pts, float))
    h = np.column_stack([pts, np.ones(len(pts))])
    return (h @ M.T)[:, :2]


def _robust_levels(img: np.ndarray) -> tuple:
    """(lo, hi) display levels from the 1st/99th percentile."""
    finite = img[np.isfinite(img)]
    if finite.size == 0:
        return 0.0, 1.0
    lo, hi = np.percentile(finite, (1.0, 99.0))
    return float(lo), float(hi) if hi > lo else float(lo) + 1.0


def composite_tiles(tiles: list, placement: Optional[np.ndarray] = None,
                    max_dim: int = 1400) -> tuple:
    """Rasterise overlapping tiles into a single grayscale RGBA image.

    Returns ``(rgba, out_affine)`` where ``rgba`` is (cols, rows, 4) uint8 (the
    axis order GLImageItem wants) and ``out_affine`` is a 3x3 mapping output
    pixel (col, row) -> world metres. Covered pixels are opaque, uncovered ones
    transparent. Each tile is normalised by its own robust levels before being
    laid down (large tiles first, so finer detail tiles overwrite on top), which
    avoids the z-fighting you get from stacking many coplanar GL textures.

    ``placement`` is an optional extra 3x3 in-plane affine (slice repositioning)
    applied on top of each tile's world affine.
    """
    tiles = [t for t in tiles if t.image is not None]
    if not tiles:
        return None, None
    ident = np.eye(3)
    pl = ident if placement is None else np.asarray(placement, float)

    def world_affine(t):
        return pl @ t.affine

    def pixel_corners(t):
        h, w = t.image.shape[0], t.image.shape[1]
        return np.array([[0, 0], [w, 0], [w, h], [0, h]], float)

    # order: largest footprint first (behind), smallest last (on top)
    def area(t):
        c = _apply_affine(world_affine(t), pixel_corners(t))
        return (c[:, 0].max() - c[:, 0].min()) * (c[:, 1].max() - c[:, 1].min())

    tiles = sorted(tiles, key=area, reverse=True)
    allc = np.vstack([_apply_affine(world_affine(t), pixel_corners(t)) for t in tiles])
    x0, x1 = float(allc[:, 0].min()), float(allc[:, 0].max())
    y0, y1 = float(allc[:, 1].min()), float(allc[:, 1].max())
    span = max(x1 - x0, y1 - y0, 1e-9)
    scale = span / max_dim
    ow = max(1, int(np.ceil((x1 - x0) / scale)))
    oh = max(1, int(np.ceil((y1 - y0) / scale)))
    # output pixel (col,row) -> world: wx = x0 + col*scale, wy = y1 - row*scale
    out_affine = np.array([[scale, 0.0, x0], [0.0, -scale, y1], [0.0, 0.0, 1.0]])
    out_inv = np.linalg.inv(out_affine)

    out = np.zeros((oh, ow), float)
    cov = np.zeros((oh, ow), bool)
    for t in tiles:
        a = world_affine(t)
        th, tw = t.image.shape[0], t.image.shape[1]
        # tile footprint in output pixels
        oc = _apply_affine(out_inv, _apply_affine(a, pixel_corners(t)))
        c0 = max(0, int(np.floor(oc[:, 0].min())))
        c1 = min(ow, int(np.ceil(oc[:, 0].max())))
        r0 = max(0, int(np.floor(oc[:, 1].min())))
        r1 = min(oh, int(np.ceil(oc[:, 1].max())))
        if c1 <= c0 or r1 <= r0:
            continue
        gx, gy = np.meshgrid(np.arange(c0, c1), np.arange(r0, r1))
        m = np.linalg.inv(a) @ out_affine        # output(col,row) -> tile(col,row)
        flat = np.column_stack([gx.ravel(), gy.ravel(), np.ones(gx.size)])
        tp = (flat @ m.T)[:, :2]
        tcol = np.round(tp[:, 0]).astype(int)
        trow = np.round(tp[:, 1]).astype(int)
        valid = (tcol >= 0) & (tcol < tw) & (trow >= 0) & (trow < th)
        if not valid.any():
            continue
        lo, hi = _robust_levels(t.image)
        samp = np.clip((t.image[trow[valid], tcol[valid]] - lo) / (hi - lo), 0, 1)
        oyy = gy.ravel()[valid]
        oxx = gx.ravel()[valid]
        out[oyy, oxx] = samp
        cov[oyy, oxx] = True

    g8 = (out * 255).astype(np.ubyte)
    rgba = np.zeros((oh, ow, 4), np.ubyte)
    rgba[..., 0] = rgba[..., 1] = rgba[..., 2] = g8
    rgba[..., 3] = np.where(cov, 255, 0).astype(np.ubyte)
    return np.transpose(rgba, (1, 0, 2)), out_affine


@dataclass
class Tile:
    """One assembled image tile placed in world (metre) coordinates."""

    name: str
    path: Optional[Path]
    image: np.ndarray            # 2D array for display (stacks are projected)
    affine: np.ndarray           # 3x3: pixel (col,row) -> world metres
    alpha: float = 1.0
    visible: bool = True         # the acq4 mosaic's "visible" flag for this image
    include: bool = True         # whether to use this tile in the assembly
    z: int = 0
    is_stack: bool = False       # True if the source was a multi-frame .ma/stack

    def corners_world(self) -> np.ndarray:
        """Four world-metre corners of the tile (pixel bbox through the affine)."""
        h, w = self.image.shape[0], self.image.shape[1]
        px = np.array([[0, 0], [w, 0], [w, h], [0, h]], float)
        return _apply_affine(self.affine, px)


@dataclass
class Marker:
    label: str
    pos: tuple                   # (x, y, z) metres


@dataclass
class MarkerSet:
    name: str                    # section type, e.g. "Cortex Coronal"
    markers: list                # list[Marker]


@dataclass
class Cell:
    name: str
    pos: tuple                   # (x, y) metres


@dataclass
class Ruler:
    name: str
    points: list                 # list of (x, y) metre endpoints


@dataclass
class AssembledMosaic:
    path: Path
    root: Path
    tiles: list = field(default_factory=list)
    marker_sets: list = field(default_factory=list)
    cells: list = field(default_factory=list)
    rulers: list = field(default_factory=list)
    missing: list = field(default_factory=list)   # tiles that could not load

    def included_tiles(self) -> list:
        """Tiles currently flagged for inclusion in the assembly."""
        return [t for t in self.tiles if t.image is not None and t.include]

    @property
    def n_images(self) -> int:
        return sum(1 for t in self.tiles if t.image is not None)

    @property
    def n_included(self) -> int:
        return len(self.included_tiles())

    def extent(self) -> Optional[tuple]:
        """Overall world bbox (xmin, xmax, ymin, ymax) in metres, or None."""
        allc = [t.corners_world() for t in self.included_tiles()]
        if not allc:
            return None
        c = np.vstack(allc)
        return float(c[:, 0].min()), float(c[:, 0].max()), \
            float(c[:, 1].min()), float(c[:, 1].max())

    @property
    def section_type(self) -> Optional[str]:
        """The first marker-set name (acq4's section-type label), if any."""
        return self.marker_sets[0].name if self.marker_sets else None


def find_mosaic_files(day_dir: Path) -> list:
    """All ``.mosaic`` files under ``day_dir`` (day, slice, cell levels), sorted."""
    day_dir = Path(day_dir)
    return sorted(day_dir.rglob(f"*{MOSAIC_EXT}"))


class MosaicReader:
    """Read a ``.mosaic`` file and assemble it into world (metre) coordinates."""

    def __init__(self):
        self.ar = acq4_reader.acq4_reader()

    # -- path resolution ----------------------------------------------------
    def _resolve_tile_path(self, item: dict, root: Path,
                           mosaic_path: Path) -> Optional[Path]:
        rel = item.get("filename")
        if not rel:
            return None
        for base in (root, mosaic_path.parent):
            p = (base / rel)
            if p.exists():
                return p
        return None

    # -- tile loading -------------------------------------------------------
    # Note: acq4_reader's bundled tifffile is incompatible with the installed
    # imagecodecs, so pixels are read with the modern ``tifffile`` package (and
    # MetaArray for .ma). acq4_reader is still used for the ``.index`` metadata,
    # which is where the physical placement transform lives.
    def _read_pixels(self, path: Path) -> tuple:
        """Return (2D float image, is_stack) or (None, False) on failure."""
        suf = path.suffix.lower()
        try:
            if suf in (".tif", ".tiff"):
                import tifffile
                arr = np.asarray(tifffile.imread(str(path)))
            elif suf == ".ma":
                import MetaArray as EM
                arr = np.asarray(EM.MetaArray(file=str(path)))
            else:
                return None, False
        except Exception:  # noqa: BLE001 - skip unreadable tiles
            return None, False
        is_stack = arr.ndim == 3
        if arr.ndim == 3:            # .ma / z-stack -> max projection for display
            arr = arr.max(axis=0)
        return np.asarray(arr, float), is_stack

    def _read_meta(self, path: Path) -> dict:
        """The tiff's ``.index`` metadata entry (transform etc.), or {}."""
        try:
            idx = self.ar._readIndex(path.parent)
            return idx.get(path.name, {}) or {}
        except Exception:  # noqa: BLE001
            return {}

    def _load_tile(self, item: dict, root: Path,
                   mosaic_path: Path) -> Optional[Tile]:
        path = self._resolve_tile_path(item, root, mosaic_path)
        if path is None:
            return None
        img, is_stack = self._read_pixels(path)
        if img is None:
            return None
        # base placement transform from the tiff's .index metadata
        meta = self._read_meta(path)
        tr = meta.get("transform", {}) or {}
        pos = list(tr.get("pos", (0.0, 0.0)))[:2] or [0.0, 0.0]
        scale = list(tr.get("scale", (1.0, 1.0)))[:2] or [1.0, 1.0]
        angle = float(tr.get("angle", 0.0) or 0.0)
        # acq4 adds a per-image .index userTransform position when set (getImage)
        iut = (meta.get("userTransform", {}) or {}).get("pos", (0.0, 0.0))
        if tuple(iut[:2]) != (0.0, 0.0):
            pos = [pos[0] + iut[0], pos[1] + iut[1]]
        base = _srt_affine(pos, scale, angle)
        # mosaic-item userTransform applied on top (usually identity)
        ut = item.get("userTransform", {}) or {}
        user = _srt_affine(ut.get("pos", (0.0, 0.0)), ut.get("scale", (1.0, 1.0)),
                           ut.get("angle", 0.0))
        affine = user @ base
        vis = bool(item.get("visible", True))
        return Tile(
            name=item.get("name", path.name), path=path, image=img,
            affine=affine, alpha=float(item.get("alpha", 1.0)),
            visible=vis, include=vis, z=int(item.get("z", 0)),
            is_stack=is_stack)

    # -- public -------------------------------------------------------------
    def read(self, mosaic_path) -> AssembledMosaic:
        mosaic_path = Path(mosaic_path)
        with open(mosaic_path) as fh:
            doc = json.load(fh)
        root = Path(doc.get("rootPath", mosaic_path.parent))
        asm = AssembledMosaic(path=mosaic_path, root=root)
        for item in doc.get("items", []):
            t = item.get("type")
            if t == "ImageCanvasItem":
                tile = self._load_tile(item, root, mosaic_path)
                if tile is not None:
                    asm.tiles.append(tile)
                else:
                    asm.missing.append(item.get("filename") or item.get("name"))
            elif t == "MarkersCanvasItem":
                marks = [Marker(label=m[0], pos=tuple(m[1]))
                         for m in item.get("markers", [])]
                asm.marker_sets.append(MarkerSet(name=item.get("name", ""),
                                                 markers=marks))
            elif t == "CellCanvasItem":
                pos = (item.get("userTransform", {}) or {}).get("pos", [0.0, 0.0])
                asm.cells.append(Cell(name=item.get("name", "cell"),
                                      pos=(float(pos[0]), float(pos[1]))))
            elif t == "RulerCanvasItem":
                pts = item.get("points") or item.get("markers") or []
                asm.rulers.append(Ruler(name=item.get("name", "ruler"),
                                        points=[tuple(p) for p in pts]))
        asm.tiles.sort(key=lambda tt: tt.z)
        return asm
