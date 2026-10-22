# Claude added 2026-10-22: 3D preliminary slice-stack alignment view + controls.
"""Preliminary (pre-atlas) 3D alignment view for the mosaic importer.

The right-hand pane gets a rotatable ``GLViewWidget`` showing every slice in the
:class:`~ephys.tools.cn_atlas.mosaic.stack.SliceStack`:

  * the slice image in grayscale — composited once per slice into a single
    texture (so overlapping tiles do not z-fight), placed by the slice's
    preliminary in-plane placement (translation / rotation about its own centre /
    side + up-down mirror);
  * a translucent, spectral-coloured slab spanning the slice's thickness extent,
    oriented with the slice, with a solid outline of the same colour;
  * the slice image drawn on the top or bottom face per the up/down setting.

The currently list-selected (and visible) slice is highlighted with a white
outline and corner handles, and can be dragged in the 3D view: drag the body to
translate, grab a corner to rotate about the slice centre. The amounts are
mirrored into the Rot / Δx / Δy boxes. Placement is baked into the GL transform
(not the composite) so dragging never re-rasterises.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pyqtgraph as pg
import pyqtgraph.opengl as gl
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets

from .reader import composite_tiles
from .stack import PLANE_PRESETS, SliceEntry, SliceStack

HANDLE_TOL_PX = 15.0


def _affine3d(a: np.ndarray, z: float) -> QtGui.QMatrix4x4:
    """Embed a 3x3 in-plane affine (maps (col,row)->(x,y)) into a 4x4 at z."""
    return QtGui.QMatrix4x4(
        float(a[0, 0]), float(a[0, 1]), 0.0, float(a[0, 2]),
        float(a[1, 0]), float(a[1, 1]), 0.0, float(a[1, 2]),
        0.0, 0.0, 1.0, float(z),
        0.0, 0.0, 0.0, 1.0)


def _prism(corners_xy, z0, z1) -> np.ndarray:
    """8 vertices (bottom 0-3, top 4-7) of a prism over a quad base."""
    c = np.asarray(corners_xy, float)
    bot = np.column_stack([c, np.full(len(c), z0)])
    top = np.column_stack([c, np.full(len(c), z1)])
    return np.vstack([bot, top])


def _prism_mesh(corners_xy, z0, z1, color, alpha: float) -> gl.GLMeshItem:
    """Translucent filled prism (oriented thickness slab)."""
    v = _prism(corners_xy, z0, z1)
    f = np.array([[0, 1, 2], [0, 2, 3], [4, 6, 5], [4, 7, 6],
                  [0, 1, 5], [0, 5, 4], [1, 2, 6], [1, 6, 5],
                  [2, 3, 7], [2, 7, 6], [3, 0, 4], [3, 4, 7]], int)
    r, g, b = color
    md = gl.MeshData(vertexes=v, faces=f)
    return gl.GLMeshItem(meshdata=md, smooth=False, drawEdges=False,
                         color=(r / 255, g / 255, b / 255, float(alpha)),
                         glOptions="translucent")


def _prism_outline(corners_xy, z0, z1, color, width=2.0) -> gl.GLLinePlotItem:
    """Solid wireframe outline of the oriented slab (12 edges)."""
    v = _prism(corners_xy, z0, z1)
    edges = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4),
             (0, 4), (1, 5), (2, 6), (3, 7)]
    seg = np.array([v[i] for e in edges for i in e], float)
    r, g, b = color
    return gl.GLLinePlotItem(pos=seg, color=(r / 255, g / 255, b / 255, 1.0),
                             width=width, mode="lines", antialias=True)


def _point_in_quad(p, quad) -> bool:
    """Ray-casting point-in-polygon test for a 4-point quad."""
    x, y = float(p[0]), float(p[1])
    inside = False
    n = len(quad)
    j = n - 1
    for i in range(n):
        xi, yi = quad[i]
        xj, yj = quad[j]
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi + 1e-12) + xi):
            inside = not inside
        j = i
    return inside


class _AlignGLView(gl.GLViewWidget):
    """GLViewWidget that lets the controller intercept mouse drags for editing."""

    def __init__(self, controller):
        super().__init__()
        self._ctrl = controller

    def mousePressEvent(self, ev):  # noqa: N802 - Qt signature
        if self._ctrl.on_mouse_press(ev):
            ev.accept()
            return
        super().mousePressEvent(ev)

    def mouseMoveEvent(self, ev):  # noqa: N802
        if self._ctrl.on_mouse_move(ev):
            ev.accept()
            return
        super().mouseMoveEvent(ev)

    def mouseReleaseEvent(self, ev):  # noqa: N802
        if self._ctrl.on_mouse_release(ev):
            ev.accept()
            return
        super().mouseReleaseEvent(ev)


class AlignmentController:
    """Owns the 3D view + control widgets and the slice stack they drive."""

    SIDES = ("Right", "Left")

    def __init__(self):
        self.stack = SliceStack()
        self.get_current = None        # callable -> AssembledMosaic or None
        self.mosaic_source = None      # object with all_/selected_mosaic_asms()
        self.fill_alpha = 0.22
        self.show_outlines = True          # draw the coloured slab bounding lines
        self.slabs_both = False            # draw the image on both faces (slab look)
        self.show_tissue = True            # draw thresholded tissue outlines
        self.tissue_threshold = 0.25       # 0..1 on the normalised composite
        self.netting_open_px = 3           # morphological opening radius (px); 0 = off
        self._thr_user_set = False
        self._composite_cache: dict = {}   # id(entry) -> (ntiles, rgba, out_affine)
        self._contour_cache: dict = {}     # id(entry) -> (sig, [Nx2 col,row arrays])
        self._items: list = []             # per-slice GL items
        self._sel_items: list = []         # selection highlight + handles
        self._entry_items: list = []       # parallel to entries: dict or None
        self._drag = None
        self._updating = False

        self.view = _AlignGLView(self)
        self.view.setBackgroundColor(pg.mkColor(16, 18, 24))
        self.view.setCameraPosition(distance=0.02)
        self.view.addItem(gl.GLAxisItem(size=pg.Vector(0.003, 0.003, 0.003)))

        self.controls = self._build_controls()
        self._refresh_list()
        self._sync_slice_controls()

    # -- controls UI --------------------------------------------------------
    def _build_controls(self) -> QtWidgets.QWidget:
        w = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(w)
        v.setContentsMargins(0, 0, 0, 0)

        trow = QtWidgets.QHBoxLayout()
        trow.addWidget(QtWidgets.QLabel("Thickness (µm):"))
        self.thick_spin = QtWidgets.QSpinBox()
        self.thick_spin.setRange(1, 5000)
        self.thick_spin.setValue(int(self.stack.thickness_um))
        self.thick_spin.valueChanged.connect(self._on_thickness)
        trow.addWidget(self.thick_spin)
        v.addLayout(trow)

        self.add_btn = QtWidgets.QPushButton("Add displayed mosaic as slice")
        self.add_btn.clicked.connect(self._on_add)
        v.addWidget(self.add_btn)
        arow = QtWidgets.QHBoxLayout()
        self.add_sel_btn = QtWidgets.QPushButton("Add selected")
        self.add_sel_btn.clicked.connect(self._on_add_selected)
        self.add_all_btn = QtWidgets.QPushButton("Add all mosaics")
        self.add_all_btn.clicked.connect(self._on_add_all)
        arow.addWidget(self.add_sel_btn)
        arow.addWidget(self.add_all_btn)
        v.addLayout(arow)

        self.slice_list = QtWidgets.QListWidget()
        self.slice_list.setMaximumHeight(140)
        self.slice_list.currentRowChanged.connect(self._on_row_changed)
        self.slice_list.itemChanged.connect(self._on_item_changed)
        v.addWidget(QtWidgets.QLabel("Slices (check = visible):"))
        v.addWidget(self.slice_list)

        brow = QtWidgets.QHBoxLayout()
        for label, cb in (("↑", lambda: self._move(-1)), ("↓", lambda: self._move(1)),
                          ("Remove", self._remove), ("Clear", self._clear)):
            b = QtWidgets.QPushButton(label)
            b.clicked.connect(cb)
            brow.addWidget(b)
        v.addLayout(brow)

        self.reverse_cb = QtWidgets.QCheckBox("Reverse colour order")
        self.reverse_cb.toggled.connect(self._on_reverse)
        v.addWidget(self.reverse_cb)

        frow = QtWidgets.QHBoxLayout()
        frow.addWidget(QtWidgets.QLabel("Fill opacity:"))
        self.fill_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.fill_slider.setRange(0, 100)
        self.fill_slider.setValue(int(self.fill_alpha * 100))
        self.fill_slider.valueChanged.connect(self._on_fill_alpha)
        frow.addWidget(self.fill_slider)
        v.addLayout(frow)

        self.outline_cb = QtWidgets.QCheckBox("Show bounding lines")
        self.outline_cb.setChecked(True)
        self.outline_cb.toggled.connect(self._on_outlines)
        v.addWidget(self.outline_cb)

        self.slabs_cb = QtWidgets.QCheckBox("Show slices as slabs (space-filling)")
        self.slabs_cb.setChecked(False)
        self.slabs_cb.toggled.connect(self._on_slabs)
        v.addWidget(self.slabs_cb)

        tbox = QtWidgets.QGroupBox("Tissue outlines")
        tv = QtWidgets.QVBoxLayout(tbox)
        tv.setContentsMargins(4, 4, 4, 4)
        self.tissue_cb = QtWidgets.QCheckBox("Show tissue outlines")
        self.tissue_cb.setChecked(True)
        self.tissue_cb.toggled.connect(self._on_tissue)
        tv.addWidget(self.tissue_cb)
        thr = QtWidgets.QHBoxLayout()
        thr.addWidget(QtWidgets.QLabel("Threshold:"))
        self.thr_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.thr_slider.setRange(0, 100)
        self.thr_slider.setValue(int(self.tissue_threshold * 100))
        self.thr_slider.valueChanged.connect(self._on_threshold)
        thr.addWidget(self.thr_slider)
        self.thr_auto_btn = QtWidgets.QPushButton("Auto")
        self.thr_auto_btn.clicked.connect(lambda: self._auto_threshold(rebuild=True))
        thr.addWidget(self.thr_auto_btn)
        tv.addLayout(thr)
        nrow = QtWidgets.QHBoxLayout()
        nrow.addWidget(QtWidgets.QLabel("Remove netting (px):"))
        self.netting_spin = QtWidgets.QSpinBox()
        self.netting_spin.setRange(0, 15)
        self.netting_spin.setValue(self.netting_open_px)
        self.netting_spin.setToolTip(
            "Morphological opening radius: removes thin bright lines (netting) "
            "that cross dark/empty regions. 0 = off.")
        self.netting_spin.valueChanged.connect(self._on_netting)
        nrow.addWidget(self.netting_spin)
        tv.addLayout(nrow)
        v.addWidget(tbox)

        self.slice_box = QtWidgets.QGroupBox("Selected slice (drag body=move, corner=rotate)")
        sv = QtWidgets.QVBoxLayout(self.slice_box)
        self.flip_cb = QtWidgets.QCheckBox("Flip up/down (reflect)")
        self.flip_cb.toggled.connect(self._on_slice_edit)
        sv.addWidget(self.flip_cb)
        self.ontop_cb = QtWidgets.QCheckBox("Image on top face")
        self.ontop_cb.setChecked(True)
        self.ontop_cb.toggled.connect(self._on_slice_edit)
        sv.addWidget(self.ontop_cb)
        arow2 = QtWidgets.QHBoxLayout()
        arow2.addWidget(QtWidgets.QLabel("Image opacity:"))
        self.img_alpha_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.img_alpha_slider.setRange(0, 100)
        self.img_alpha_slider.setValue(100)
        self.img_alpha_slider.valueChanged.connect(self._on_slice_edit)
        arow2.addWidget(self.img_alpha_slider)
        sv.addLayout(arow2)
        srow = QtWidgets.QHBoxLayout()
        srow.addWidget(QtWidgets.QLabel("Side:"))
        self.side_combo = QtWidgets.QComboBox()
        self.side_combo.addItems(self.SIDES)
        self.side_combo.currentTextChanged.connect(self._on_slice_edit)
        srow.addWidget(self.side_combo)
        srow.addWidget(QtWidgets.QLabel("Rot°:"))
        self.rot_spin = QtWidgets.QDoubleSpinBox()
        self.rot_spin.setRange(-180.0, 180.0)
        self.rot_spin.setSingleStep(1.0)
        self.rot_spin.valueChanged.connect(self._on_slice_edit)
        srow.addWidget(self.rot_spin)
        sv.addLayout(srow)
        trow2 = QtWidgets.QHBoxLayout()
        trow2.addWidget(QtWidgets.QLabel("Δx,Δy (µm):"))
        self.tx_spin = QtWidgets.QDoubleSpinBox()
        self.ty_spin = QtWidgets.QDoubleSpinBox()
        for sp in (self.tx_spin, self.ty_spin):
            sp.setRange(-20000.0, 20000.0)
            sp.setSingleStep(10.0)
            sp.valueChanged.connect(self._on_slice_edit)
            trow2.addWidget(sp)
        sv.addLayout(trow2)
        v.addWidget(self.slice_box)

        prow = QtWidgets.QHBoxLayout()
        prow.addWidget(QtWidgets.QLabel("Section plane:"))
        self.plane_combo = QtWidgets.QComboBox()
        self.plane_combo.addItems(list(PLANE_PRESETS.keys()))
        self.plane_combo.currentTextChanged.connect(self._on_plane)
        prow.addWidget(self.plane_combo, stretch=1)
        v.addLayout(prow)

        self.info = QtWidgets.QLabel("Add slices to build the stack.")
        self.info.setWordWrap(True)
        v.addWidget(self.info)
        return w

    # -- adding slices ------------------------------------------------------
    def _included_tiles(self, asm) -> list:
        if hasattr(asm, "included_tiles"):
            return asm.included_tiles()
        return [t for t in getattr(asm, "tiles", []) if t.image is not None]

    def _raw_bbox(self, asm) -> Optional[tuple]:
        cs = [t.corners_world() for t in self._included_tiles(asm)]
        if not cs:
            return None
        allc = np.vstack(cs)
        return (float(allc[:, 0].min()), float(allc[:, 0].max()),
                float(allc[:, 1].min()), float(allc[:, 1].max()))

    def _raw_corners(self, asm) -> Optional[np.ndarray]:
        bb = self._raw_bbox(asm)
        if bb is None:
            return None
        x0, x1, y0, y1 = bb
        return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], float)

    def _add_asm(self, asm) -> bool:
        if asm is None or not getattr(asm, "tiles", None):
            return False
        e = self.stack.add(asm, name=asm.path.name)
        bb = self._raw_bbox(asm)
        if bb is not None:
            e.center = ((bb[0] + bb[1]) / 2.0, (bb[2] + bb[3]) / 2.0)
        return True

    def _on_add(self):
        asm = self.get_current() if self.get_current else None
        if not self._add_asm(asm):
            self.info.setText("No assembled mosaic to add — select one first.")
            return
        self._refresh_list()
        self.slice_list.setCurrentRow(len(self.stack.entries) - 1)
        self._maybe_auto_threshold()
        self._rebuild()

    def _on_add_selected(self):
        self._bulk_add(getattr(self.mosaic_source, "selected_mosaic_asms", lambda: [])())

    def _on_add_all(self):
        self._bulk_add(getattr(self.mosaic_source, "all_mosaic_asms", lambda: [])())

    def _bulk_add(self, asms):
        added = sum(1 for asm in asms if self._add_asm(asm))
        self._refresh_list()
        if self.stack.entries:
            self.slice_list.setCurrentRow(len(self.stack.entries) - 1)
        self._maybe_auto_threshold()
        self._rebuild()
        self.info.setText(f"Added {added} mosaic(s); {len(self.stack.entries)} in stack.")

    def add_slice(self, asm):
        """Programmatic add (same as the Add button)."""
        if self._add_asm(asm):
            self._refresh_list()
            self._rebuild()

    # -- control handlers ---------------------------------------------------
    def _on_thickness(self, val: int):
        self.stack.thickness_um = float(val)
        self._rebuild()

    def _current_idx(self) -> int:
        return self.slice_list.currentRow()

    def _move(self, delta: int):
        idx = self._current_idx()
        if idx < 0:
            return
        new = self.stack.move(idx, delta)
        self._refresh_list()
        self.slice_list.setCurrentRow(new)
        self._rebuild()

    def _remove(self):
        idx = self._current_idx()
        if idx < 0:
            return
        self._composite_cache.pop(id(self.stack.entries[idx]), None)
        self.stack.remove(idx)
        self._refresh_list()
        self._rebuild()

    def _clear(self):
        self.stack.clear()
        self._composite_cache.clear()
        self._refresh_list()
        self._rebuild()

    def _on_reverse(self, on: bool):
        self.stack.reverse_colors = bool(on)
        self._rebuild()

    def _on_fill_alpha(self, val: int):
        self.fill_alpha = val / 100.0
        self._rebuild()

    def _on_outlines(self, on: bool):
        self.show_outlines = bool(on)
        self._rebuild()

    def _on_slabs(self, on: bool):
        self.slabs_both = bool(on)
        self._rebuild()

    def _on_tissue(self, on: bool):
        self.show_tissue = bool(on)
        self._rebuild()

    def _on_threshold(self, val: int):
        if self._updating:
            return
        self.tissue_threshold = val / 100.0
        self._thr_user_set = True
        self._contour_cache.clear()
        self._rebuild()

    def _on_netting(self, val: int):
        self.netting_open_px = int(val)
        self._contour_cache.clear()
        self._rebuild()

    def invalidate_asm(self, asm):
        """Drop cached composites/contours for stack entries using ``asm`` (its
        included-image set changed) and redraw."""
        changed = False
        for e in self.stack.entries:
            if e.asm is asm:
                self._composite_cache.pop(id(e), None)
                self._contour_cache.pop(id(e), None)
                bb = self._raw_bbox(asm)
                if bb is not None:
                    e.center = ((bb[0] + bb[1]) / 2.0, (bb[2] + bb[3]) / 2.0)
                changed = True
        if changed:
            self._rebuild()

    def _on_plane(self, name: str):
        self.stack.plane_preset = name
        angles = PLANE_PRESETS.get(name, {})
        self.info.setText(
            f"Section plane '{name}'"
            + (f"  ({angles})" if angles else "  (cardinal)")
            + " — used by the atlas aligner, not shown here.")

    def _on_row_changed(self, *_):
        self._sync_slice_controls()
        self._refresh_selection()

    def _on_item_changed(self, item: QtWidgets.QListWidgetItem):
        if self._updating:
            return
        idx = self.slice_list.row(item)
        if 0 <= idx < len(self.stack.entries):
            self.stack.entries[idx].visible = (
                item.checkState() == QtCore.Qt.CheckState.Checked)
            self._rebuild()

    def _on_slice_edit(self, *_):
        if self._updating:
            return
        idx = self._current_idx()
        if idx < 0:
            return
        e = self.stack.entries[idx]
        e.flip_ud = self.flip_cb.isChecked()
        e.image_on_top = self.ontop_cb.isChecked()
        e.side = self.side_combo.currentText()
        e.rot_deg = self.rot_spin.value()
        e.tx = self.tx_spin.value() * 1e-6
        e.ty = self.ty_spin.value() * 1e-6
        e.image_alpha = self.img_alpha_slider.value() / 100.0
        self._rebuild()

    # -- list / selection sync ---------------------------------------------
    def _refresh_list(self):
        row = self.slice_list.currentRow()
        self._updating = True
        self.slice_list.clear()
        for i, e in enumerate(self.stack.entries):
            it = QtWidgets.QListWidgetItem(f"{i}: {e.name}")
            it.setFlags(it.flags() | QtCore.Qt.ItemFlag.ItemIsUserCheckable)
            it.setCheckState(QtCore.Qt.CheckState.Checked if e.visible
                             else QtCore.Qt.CheckState.Unchecked)
            self.slice_list.addItem(it)
        self._updating = False
        if 0 <= row < len(self.stack.entries):
            self.slice_list.setCurrentRow(row)

    def _sync_slice_controls(self, *_):
        idx = self._current_idx()
        enabled = 0 <= idx < len(self.stack.entries)
        self.slice_box.setEnabled(enabled)
        if not enabled:
            return
        e = self.stack.entries[idx]
        self._updating = True
        self.flip_cb.setChecked(e.flip_ud)
        self.ontop_cb.setChecked(e.image_on_top)
        self.side_combo.setCurrentText(e.side)
        self.rot_spin.setValue(e.rot_deg)
        self.tx_spin.setValue(e.tx * 1e6)
        self.ty_spin.setValue(e.ty * 1e6)
        self.img_alpha_slider.setValue(int(round(e.image_alpha * 100)))
        self._updating = False

    def _selected_visible_idx(self) -> Optional[int]:
        idx = self.slice_list.currentRow()
        if 0 <= idx < len(self.stack.entries) and self.stack.entries[idx].visible:
            return idx
        return None

    # -- geometry helpers ---------------------------------------------------
    def _placed_corners(self, e: SliceEntry) -> Optional[np.ndarray]:
        rc = self._raw_corners(e.asm)
        if rc is None:
            return None
        h = np.column_stack([rc, np.ones(len(rc))])
        return (h @ e.placement_affine().T)[:, :2]

    def _zface(self, idx: int) -> float:
        e = self.stack.entries[idx]
        th = self.stack.thickness_m
        z0 = self.stack.z_of(idx)
        eps = max(th * 0.02, 1e-6)
        return (z0 + th + eps) if e.image_on_top else (z0 - eps)

    def _image_zs(self, idx: int) -> list:
        """z-planes at which to draw the slice image. One face normally; when
        'slabs' is on, several planes through the thickness to fill the volume."""
        th = self.stack.thickness_m
        z0 = self.stack.z_of(idx)
        if self.slabs_both:
            n = int(np.clip(round(self.stack.thickness_um / 30.0), 3, 8))
            return list(np.linspace(z0, z0 + th, n))
        return [self._zface(idx)]

    def _composite_for(self, e: SliceEntry):
        key = id(e)
        tiles = self._included_tiles(e.asm)
        n = len(tiles)
        c = self._composite_cache.get(key)
        if c is not None and c[0] == n:
            return c[1], c[2]
        rgba, aff = composite_tiles(tiles)            # raw world (placement=None)
        self._composite_cache[key] = (n, rgba, aff)
        self._contour_cache.pop(key, None)            # composite changed
        return rgba, aff

    # -- tissue outlines ----------------------------------------------------
    def _maybe_auto_threshold(self):
        if not self._thr_user_set and self.stack.entries:
            self._auto_threshold(rebuild=False)

    def _auto_threshold(self, rebuild: bool = False):
        """Default threshold from the combined intensity histogram (Otsu)."""
        vals = []
        for e in self.stack.entries:
            rgba, _ = self._composite_for(e)
            if rgba is None:
                continue
            g = rgba[..., 0].astype(float) / 255.0
            m = rgba[..., 3] > 0
            if m.any():
                vals.append(g[m])
        if vals:
            allg = np.concatenate(vals)
            try:
                from skimage.filters import threshold_otsu
                thr = float(threshold_otsu(allg))
            except Exception:  # noqa: BLE001
                thr = float(np.clip(np.mean(allg), 0.05, 0.95))
            self.tissue_threshold = thr
            self._updating = True
            self.thr_slider.setValue(int(round(thr * 100)))
            self._updating = False
            self._contour_cache.clear()
        if rebuild:
            self._thr_user_set = False
            self._rebuild()

    def _contours_for(self, e: SliceEntry) -> list:
        """Thresholded tissue-boundary contours in composite pixel (col,row,0)."""
        key = id(e)
        sig = (len(self._included_tiles(e.asm)), round(self.tissue_threshold, 4),
               self.netting_open_px)
        c = self._contour_cache.get(key)
        if c is not None and c[0] == sig:
            return c[1]
        rgba, _ = self._composite_for(e)
        contours: list = []
        if rgba is not None:
            gray = (rgba[..., 0].astype(float) / 255.0).T      # (rows, cols)
            mask = (rgba[..., 3] > 0).T
            binary = (gray > self.tissue_threshold) & mask
            binary = self._clean_mask(binary)
            try:
                from skimage.measure import find_contours
                for cc in find_contours(binary.astype(float), 0.5):
                    if len(cc) < 60:                           # drop speckle
                        continue
                    contours.append(np.column_stack(
                        [cc[:, 1], cc[:, 0], np.zeros(len(cc))]))
            except Exception:  # noqa: BLE001
                contours = []
        self._contour_cache[key] = (sig, contours)
        return contours

    def _clean_mask(self, binary: np.ndarray) -> np.ndarray:
        """Remove thin netting, fill holes, and keep only major connected
        components (tissue pieces), so the outline traces the tissue boundary
        rather than the netting lines or every bright speck.

        The netting (thin bright lines ~1–1.5 mm apart) is removed with a
        morphological opening: it erases bright features thinner than the
        structuring element while leaving the large tissue blob. Netting that
        lies over tissue is harmless (it is inside the filled blob); only netting
        crossing dark/empty regions would otherwise create spurious contours.
        """
        try:
            from scipy.ndimage import binary_fill_holes
            from skimage.measure import label
            from skimage.morphology import disk, opening
            # Fill holes first so the tissue is solid, THEN open — this strips thin
            # netting lines without fragmenting the (now solid) tissue blob.
            work = binary_fill_holes(binary)
            if self.netting_open_px > 0:
                work = opening(work, disk(self.netting_open_px))
            lbl = label(work)
            if lbl.max() == 0:
                return work
            sizes = np.bincount(lbl.ravel())
            sizes[0] = 0
            keep = sizes >= max(sizes.max() * 0.05, 50)   # ≥5% of the largest
            keep[0] = False
            return keep[lbl]
        except Exception:  # noqa: BLE001
            return binary

    # -- 3D rendering -------------------------------------------------------
    def _clear_items(self):
        for it in self._items:
            self.view.removeItem(it)
        self._items = []
        self._clear_selection_items()

    def _clear_selection_items(self):
        for it in self._sel_items:
            self.view.removeItem(it)
        self._sel_items = []

    def _rebuild(self):
        self._clear_items()
        self._entry_items = []
        th = self.stack.thickness_m
        all_bboxes = []
        for i, e in enumerate(self.stack.entries):
            if not e.visible:
                self._entry_items.append(None)
                continue
            z0 = self.stack.z_of(i)
            z1 = z0 + th
            color = self.stack.color_for(i)
            pc = self._placed_corners(e)
            d: dict = {}
            if pc is not None:
                all_bboxes.append((pc[:, 0].min(), pc[:, 0].max(),
                                   pc[:, 1].min(), pc[:, 1].max(), z0, z1))
                slab = _prism_mesh(pc, z0, z1, color, self.fill_alpha)
                self.view.addItem(slab)
                self._items.append(slab)
                d["slab"] = slab
                if self.show_outlines:
                    outline = _prism_outline(pc, z0, z1, color, 2.0)
                    self.view.addItem(outline)
                    self._items.append(outline)
                    d["outline"] = outline
            rgba, aff = self._composite_for(e)
            tr = _affine3d(e.placement_affine() @ aff, self._zface(i))
            if rgba is not None:
                rgba = rgba.copy()
                rgba[..., 3] = (rgba[..., 3].astype(float) * e.image_alpha).astype(np.ubyte)
                faces = self._image_zs(i)
                d["images"] = []
                for zf in faces:
                    img = gl.GLImageItem(rgba, glOptions="translucent")
                    img.setTransform(_affine3d(e.placement_affine() @ aff, zf))
                    self.view.addItem(img)
                    self._items.append(img)
                    d["images"].append(img)
            if self.show_tissue and rgba is not None:
                r, g, b = color
                d["contours"] = []
                for pts in self._contours_for(e):
                    li = gl.GLLinePlotItem(pos=pts, color=(r / 255, g / 255, b / 255, 1.0),
                                           width=2.0, mode="line_strip", antialias=True)
                    li.setTransform(tr)
                    self.view.addItem(li)
                    self._items.append(li)
                    d["contours"].append(li)
            self._entry_items.append(d)
        self._refresh_selection()
        self._update_info()
        self._fit_camera(all_bboxes)

    def _refresh_selection(self):
        self._clear_selection_items()
        idx = self._selected_visible_idx()
        if idx is None:
            return
        e = self.stack.entries[idx]
        pc = self._placed_corners(e)
        if pc is None:
            return
        z0 = self.stack.z_of(idx)
        z1 = z0 + self.stack.thickness_m
        hl = _prism_outline(pc, z0, z1, (255, 255, 255), 3.0)
        zface = self._zface(idx)
        pts = np.column_stack([pc, np.full(len(pc), zface)])
        handles = gl.GLScatterPlotItem(pos=pts, size=14, color=(1, 1, 1, 1),
                                       pxMode=True)
        handles.setGLOptions("translucent")
        self.view.addItem(hl)
        self.view.addItem(handles)
        self._sel_items = [hl, handles]

    def _update_drag_items(self, idx: int):
        """Fast per-drag update: move the slice's items without re-compositing."""
        if idx >= len(self._entry_items):
            return
        d = self._entry_items[idx]
        if not d:
            return
        e = self.stack.entries[idx]
        th = self.stack.thickness_m
        z0 = self.stack.z_of(idx)
        z1 = z0 + th
        color = self.stack.color_for(idx)
        pc = self._placed_corners(e)
        if pc is None:
            return
        _, aff = self._composite_for(e)
        base = e.placement_affine() @ aff
        faces = self._image_zs(idx)
        for item, zf in zip(d.get("images", []), faces):
            item.setTransform(_affine3d(base, zf))
        ctr = _affine3d(base, self._zface(idx))
        for item in d.get("contours", []):
            item.setTransform(ctr)
        for key, maker in (("slab", lambda: _prism_mesh(pc, z0, z1, color, self.fill_alpha)),
                           ("outline", lambda: _prism_outline(pc, z0, z1, color, 2.0))):
            if key in d:
                self.view.removeItem(d[key])
                self._items.remove(d[key])
                d[key] = maker()
                self.view.addItem(d[key])
                self._items.append(d[key])
        self._refresh_selection()

    def _update_info(self):
        n = len(self.stack.entries)
        vis = sum(1 for e in self.stack.entries if e.visible)
        self.info.setText(
            f"{n} slice(s) ({vis} shown), thickness {self.stack.thickness_um:.0f} µm, "
            f"depth {n * self.stack.thickness_um:.0f} µm. Plane: {self.stack.plane_preset}.")

    def _fit_camera(self, bboxes):
        if not bboxes:
            return
        arr = np.array(bboxes, float)
        cx = 0.5 * (arr[:, 0].min() + arr[:, 1].max())
        cy = 0.5 * (arr[:, 2].min() + arr[:, 3].max())
        cz = 0.5 * (arr[:, 4].min() + arr[:, 5].max())
        span = max(arr[:, 1].max() - arr[:, 0].min(),
                   arr[:, 3].max() - arr[:, 2].min(),
                   arr[:, 5].max() - arr[:, 4].min(), 1e-4)
        self.view.opts["center"] = pg.Vector(cx, cy, cz)
        self.view.setCameraPosition(distance=span * 2.2)

    # -- mouse manipulation (translate / rotate the selected slice) ---------
    def _mvp(self) -> np.ndarray:
        proj = np.array(self.view.projectionMatrix().data(), float).reshape(4, 4).T
        view = np.array(self.view.viewMatrix().data(), float).reshape(4, 4).T
        return proj @ view

    def _world_to_screen(self, x, y, z):
        c = self._mvp() @ np.array([x, y, z, 1.0])
        if abs(c[3]) < 1e-12:
            return None
        c = c / c[3]
        w, h = self.view.width(), self.view.height()
        return (c[0] + 1) / 2 * w, (1 - c[1]) / 2 * h

    def _screen_to_plane(self, sx, sy, z):
        w, h = self.view.width(), self.view.height()
        ndx = 2 * sx / w - 1
        ndy = 1 - 2 * sy / h
        inv = np.linalg.inv(self._mvp())

        def un(ndz):
            p = inv @ np.array([ndx, ndy, ndz, 1.0])
            return p[:3] / p[3]

        a, b = un(-1.0), un(1.0)
        d = b - a
        if abs(d[2]) < 1e-12:
            return None
        t = (z - a[2]) / d[2]
        p = a + t * d
        return np.array([p[0], p[1]])

    def on_mouse_press(self, ev) -> bool:
        if ev.button() != QtCore.Qt.MouseButton.LeftButton:
            return False
        idx = self._selected_visible_idx()
        if idx is None:
            return False
        e = self.stack.entries[idx]
        zface = self._zface(idx)
        pos = ev.position()
        sx, sy = pos.x(), pos.y()
        pc = self._placed_corners(e)
        if pc is None:
            return False
        for cx, cy in pc:                       # corner handle -> rotate
            scr = self._world_to_screen(cx, cy, zface)
            if scr and np.hypot(scr[0] - sx, scr[1] - sy) < HANDLE_TOL_PX:
                center = pc.mean(axis=0)
                plane = self._screen_to_plane(sx, sy, zface)
                if plane is None:
                    return False
                self._drag = {"mode": "rotate", "idx": idx, "center": center,
                              "ang0": np.arctan2(plane[1] - center[1], plane[0] - center[0]),
                              "rot0": e.rot_deg}
                return True
        plane = self._screen_to_plane(sx, sy, zface)
        if plane is not None and _point_in_quad(plane, pc):   # body -> translate
            self._drag = {"mode": "translate", "idx": idx, "start": plane,
                          "tx0": e.tx, "ty0": e.ty}
            return True
        return False

    def on_mouse_move(self, ev) -> bool:
        if self._drag is None:
            return False
        idx = self._drag["idx"]
        e = self.stack.entries[idx]
        pos = ev.position()
        plane = self._screen_to_plane(pos.x(), pos.y(), self._zface(idx))
        if plane is None:
            return True
        if self._drag["mode"] == "translate":
            e.tx = self._drag["tx0"] + (plane[0] - self._drag["start"][0])
            e.ty = self._drag["ty0"] + (plane[1] - self._drag["start"][1])
        else:
            ang = np.arctan2(plane[1] - self._drag["center"][1],
                             plane[0] - self._drag["center"][0])
            e.rot_deg = self._drag["rot0"] + np.degrees(ang - self._drag["ang0"])
            e.rot_deg = (e.rot_deg + 180.0) % 360.0 - 180.0
        self._update_drag_items(idx)
        self._sync_slice_controls()
        return True

    def on_mouse_release(self, ev) -> bool:
        if self._drag is None:
            return False
        self._drag = None
        self.view.mousePos = ev.position()     # keep camera from jumping next drag
        return True
