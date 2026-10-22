"""KimViewer: Kim Unified Brain atlas controller.

Manages the brain outline, CN region meshes, region tree, section/slice tool,
and the registered Muniak CN overlay on the shared GLViewWidget.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pyqtgraph as pg
import pyqtgraph.opengl as gl
from pyqtgraph.Qt import QtCore, QtWidgets

from . import alignment_store
from . import coords as coords_mod
from . import ground_truth as gt_mod
from . import kim as kim_mod
from . import muniak as muniak_mod
from . import picked_points as picks_mod
from . import registration as reg_mod
from . import render
from . import section as section_mod
from .widgets import (CollapsibleBox, StepSlider, LATERAL_VIEW as _LATERAL_VIEW,
                      euler_R as _euler_R)


class KimViewer:
    """Kim Unified Brain atlas: translucent brain outline + CN groups + tree."""

    OUTLINE_ALPHA = 0.12

    def __init__(self, view, colorbar, cfg, cache_dir, atlas_root, rebuild=False,
                 section_plate=None, muniak_root=None):
        self.view = view
        self.colorbar = colorbar
        self.cfg = cfg
        self.cache_dir = cache_dir
        self.rebuild = rebuild
        self.atlas = kim_mod.KimAtlas(atlas_root, cfg, cache_dir)
        self.section_plate = section_plate
        self.muniak_root = muniak_root
        self.muniak_items: list = []   # registered Muniak CN mesh item(s)
        self._reg = None               # anatomical base placement dict
        self._muniak_orig = None       # original Muniak frame data (loaded once)
        self._base_verts = None        # Muniak verts after base placement
        self._base_centroid = None     # centroid of base-placed verts (nudge pivot)
        self._tree_on: dict = {}       # sid -> checked (for master visibility)
        self.outline_item: Optional[gl.GLMeshItem] = None
        self.group_items: dict[str, gl.GLMeshItem] = {}
        self.tree_items: dict[int, gl.GLMeshItem] = {}
        self._base_color: dict = {}            # item -> (r,g,b) 0-1
        self._mesh_data: dict = {}             # item -> (vertices, full_faces)
        self._fit_verts: Optional[np.ndarray] = None  # for camera framing
        self._plane_item: Optional[gl.GLMeshItem] = None  # 3D section indicator
        self._section_line_item: Optional[gl.GLLinePlotItem] = None  # 3D drawn line
        self._section_pts_item: Optional[gl.GLScatterPlotItem] = None  # 3D endpoints
        self._line_world = None            # (w0, w1) world endpoints once both placed
        self._line_pts_present = False     # any line endpoint currently placed
        self._oblique_section = None       # {n,u,v,mid} when resliced ⊥ the line
        self.on_section_visibility = None  # window-set callback(bool)
        self.gt = gt_mod.GroundTruth.load()   # adjunct ground-truth annotations
        # Claude added 2026-10-22: cursor-picked alignment points (left-click).
        self.picks = picks_mod.PickedPoints.load()
        self._gt_items: list = []             # 3D GL items for ground truth
        self._built = False
        self.panel = self._build_panel()

    # -- panel --------------------------------------------------------------
    def _build_panel(self) -> QtWidgets.QWidget:
        panel = QtWidgets.QWidget()
        form = QtWidgets.QVBoxLayout(panel)
        form.setContentsMargins(0, 0, 0, 0)

        self._build_model_controls(form)

        self.outline_cb = QtWidgets.QCheckBox("Brain outline (translucent)")
        self.outline_cb.setChecked(True)
        self.outline_cb.toggled.connect(self._toggle_outline)
        form.addWidget(self.outline_cb)

        self.outline_op_label = QtWidgets.QLabel(f"Outline opacity: {int(self.OUTLINE_ALPHA*100)}%")
        form.addWidget(self.outline_op_label)
        self.outline_op = StepSlider()
        self.outline_op.setRange(0, 100)
        self.outline_op.setValue(int(self.OUTLINE_ALPHA * 100))
        self.outline_op.valueChanged.connect(self._on_outline_opacity)
        form.addWidget(self.outline_op)

        # --- Cochlear nucleus regions (collapsible) ---
        regions_box = CollapsibleBox("Cochlear nucleus regions")
        self.group_cbs: dict[str, QtWidgets.QCheckBox] = {}
        for g, info in kim_mod.CN_GROUPS.items():
            cb = QtWidgets.QCheckBox(g)
            cb.setChecked(True)
            r, gc, b = info["color"]
            cb.setStyleSheet(f"QCheckBox{{color: rgb({r},{gc},{b});}}")
            cb.toggled.connect(lambda on, grp=g: self._toggle_group(grp, on))
            self.group_cbs[g] = cb
            regions_box.addWidget(cb)
        self.opacity_label = QtWidgets.QLabel("Region opacity: 100%")
        regions_box.addWidget(self.opacity_label)
        self.opacity = StepSlider()
        self.opacity.setRange(0, 100)
        self.opacity.setValue(100)
        self.opacity.valueChanged.connect(self._on_opacity_change)
        regions_box.addWidget(self.opacity)
        form.addWidget(regions_box)

        self.axis_cb = QtWidgets.QCheckBox("Show axes")
        self.axis_cb.setChecked(False)
        self.axis_cb.toggled.connect(self._toggle_axis)
        form.addWidget(self.axis_cb)

        # Section / slice controls live on the right panel (above Muniak
        # placement), built in _build_right_panel().

        # --- Muniak overlay (display controls only; placement on right panel) ---
        self._build_overlay_controls(form)

        # --- Ground truth (adjunct annotations) ---
        gt_box = CollapsibleBox("Ground truth (your labels)")
        self._build_gt_controls(gt_box)
        form.addWidget(gt_box)

        form.addWidget(QtWidgets.QLabel("<b>All regions</b> (check to add)"))
        self.tree = QtWidgets.QTreeWidget()
        self.tree.setHeaderHidden(True)
        self.tree.itemChanged.connect(self._on_tree_item)
        form.addWidget(self.tree, stretch=1)

        self.status = QtWidgets.QLabel("")
        self.status.setWordWrap(True)
        form.addWidget(self.status)
        self.axis = None

        # right-side panel: Muniak hemisphere + fine-tune placement
        self.right_panel = self._build_right_panel()
        self._restore_overlay()

        # wire the ground-truth editor on the shared 2D plate
        if self.section_plate is not None:
            self.section_plate.set_gt(self.gt)
            self.section_plate.on_contour_drawn = self._gt_contour_drawn
            self.section_plate.on_contour_edited = self._gt_contour_edited
            self.section_plate.on_landmark_placed = self._gt_landmark_placed
            self.section_plate.on_section_line_changed = self._on_section_line_changed
            self.section_plate.set_picks(self.picks)
            self.section_plate.on_point_picked = self._point_picked
            self._push_coord_frames()
        return panel

    # -- coordinate reference frames ---------------------------------------
    # Claude added 2026-10-22: build Kim-native / Bregma / Stereotaxic frames
    # from the atlas geometry + config and hand them to the shared plate.
    def _midline_x(self) -> float:
        try:
            return float(self.atlas.world_center()[0])
        except Exception:  # noqa: BLE001 - volume may be missing
            return 0.0

    def _build_coord_frames(self):
        bregma = self.cfg.get("KimBregmaWorld")
        if bregma is not None:
            bregma = [float(v) for v in bregma]
        return coords_mod.build_frames(
            self._midline_x(), bregma,
            ml_sign=float(self.cfg.get("StereoMLRightSign", 1.0)),
            ap_sign=float(self.cfg.get("StereoAPAnteriorSign", 1.0)),
            dv_sign=float(self.cfg.get("StereoDVSign", 1.0)))

    def _push_coord_frames(self):
        if self.section_plate is None:
            return
        frames, bregma_set = self._build_coord_frames()
        self.section_plate.set_coord_frames(
            frames, self.cfg.get("DefaultCoordFrame", "Kim-native"), bregma_set)

    def _set_bregma_from_pick(self):
        if not self.picks.points:
            self.status.setText("No picked point yet — left-click a slice first.")
            return
        world = self.picks.points[-1]["point"]
        self.cfg["KimBregmaWorld"] = [float(world[0]), float(world[1]), float(world[2])]
        self._push_coord_frames()
        self.status.setText(
            f"Bregma set from last pick @ ML={world[0] / 1000.0:.3f} "
            f"AP={world[1] / 1000.0:.3f} DV={world[2] / 1000.0:.3f} mm (Kim-native) "
            "— Bregma/Stereotaxic readouts now reference it.")

    # -- activation ---------------------------------------------------------
    def activate(self):
        self.colorbar.setVisible(False)
        if not self._built:
            self._build_scene()
        else:
            self._add_items()
        self._fit_camera()
        if self.on_section_visibility is not None:
            self.on_section_visibility(self.section_cb.isChecked())

    def deactivate(self):
        for it in self._all_items():
            if it in self.view.items:
                self.view.removeItem(it)
        if self.on_section_visibility is not None:
            self.on_section_visibility(False)

    def _all_items(self):
        items = []
        if self.outline_item is not None:
            items.append(self.outline_item)
        items += list(self.group_items.values())
        items += list(self.tree_items.values())
        items += list(self.muniak_items)
        items += list(self._gt_items)
        if self.axis is not None:
            items.append(self.axis)
        if self._plane_item is not None:
            items.append(self._plane_item)
        if self._section_line_item is not None:
            items.append(self._section_line_item)
        if self._section_pts_item is not None:
            items.append(self._section_pts_item)
        return items

    def _add_items(self):
        for it in self._all_items():
            if it not in self.view.items:
                self.view.addItem(it)

    def _build_scene(self):
        self.status.setText("Building Kim atlas meshes ...")
        QtWidgets.QApplication.processEvents()
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.CursorShape.WaitCursor)
        try:
            bo = self.atlas.brain_outline(rebuild=self.rebuild)
            self._fit_verts = bo["vertices"]
            self.outline_item = gl.GLMeshItem(
                vertexes=bo["vertices"], faces=bo["faces"], smooth=True,
                color=(0.7, 0.72, 0.8, self._outline_alpha()), shader="shaded")
            # Translucent hull that must not occlude the structures inside it,
            # and is drawn after the opaque regions (higher depthValue).
            self.outline_item.setGLOptions(render.translucent_no_depth_write())
            self.outline_item.setDepthValue(1)
            self._mesh_data[self.outline_item] = (bo["vertices"], bo["faces"])
            self.view.addItem(self.outline_item)
            for g in kim_mod.CN_GROUPS:
                self._ensure_group(g)
            self._populate_tree()
            self._built = True
            self.status.setText("")
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()

    # -- CN groups ----------------------------------------------------------
    def _ensure_group(self, group: str):
        if group in self.group_items:
            return
        m = self.atlas.group_mesh(group, rebuild=self.rebuild)
        if not len(m["vertices"]):
            return
        r, g, b = kim_mod.CN_GROUPS[group]["color"]
        base = (r / 255, g / 255, b / 255)
        item = gl.GLMeshItem(vertexes=m["vertices"], faces=m["faces"], smooth=True,
                             shader=render.BRIGHT_SHADER, glOptions="opaque")
        self._base_color[item] = base
        self._mesh_data[item] = (m["vertices"], m["faces"])
        item.setColor((*base, self._alpha()))
        self.group_items[group] = item
        self.view.addItem(item)
        item.setVisible(self.group_cbs[group].isChecked())
        self._apply_clip(item)

    def _toggle_group(self, group: str, on: bool):
        if on and group not in self.group_items:
            QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.CursorShape.WaitCursor)
            try:
                self._ensure_group(group)
            finally:
                QtWidgets.QApplication.restoreOverrideCursor()
        if group in self.group_items:
            self.group_items[group].setVisible(on and self._kim_master())

    # -- region tree --------------------------------------------------------
    def _populate_tree(self):
        present = self.atlas.present_labels()
        # keep only nodes with at least one present descendant
        keep = set()
        for sid in self.atlas.structures:
            if sid in present:
                node = sid
                while node is not None and node not in keep:
                    keep.add(node)
                    node = self.atlas.structures.get(node, {}).get("parent")
        self.tree.blockSignals(True)
        roots = [s for s in self.atlas.structures if self.atlas.structures[s]["parent"] is None]
        witems: dict[int, QtWidgets.QTreeWidgetItem] = {}

        def add(sid, parent_item):
            s = self.atlas.structures[sid]
            label = self.atlas.display_name(sid) + f"  ({s['name']})"
            it = QtWidgets.QTreeWidgetItem([label])
            it.setFlags(it.flags() | QtCore.Qt.ItemFlag.ItemIsUserCheckable)
            it.setCheckState(0, QtCore.Qt.CheckState.Unchecked)
            it.setData(0, QtCore.Qt.ItemDataRole.UserRole, sid)
            r, g, b = s["rgb"]
            it.setForeground(0, pg.mkColor(r, g, b))
            if parent_item is None:
                self.tree.addTopLevelItem(it)
            else:
                parent_item.addChild(it)
            witems[sid] = it
            for c in sorted(self.atlas.children.get(sid, []),
                            key=lambda x: self.atlas.structures[x]["order"]):
                if c in keep:
                    add(c, it)

        for r in sorted(roots, key=lambda x: self.atlas.structures[x]["order"]):
            if r in keep:
                add(r, None)
        self.tree.blockSignals(False)

    def _on_tree_item(self, item, _col):
        sid = item.data(0, QtCore.Qt.ItemDataRole.UserRole)
        if sid is None:
            return
        on = item.checkState(0) == QtCore.Qt.CheckState.Checked
        if on and sid not in self.tree_items:
            QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.CursorShape.WaitCursor)
            try:
                m = self.atlas.structure_mesh(sid, rebuild=self.rebuild)
            finally:
                QtWidgets.QApplication.restoreOverrideCursor()
            if not len(m["vertices"]):
                self.status.setText(f"{self.atlas.display_name(sid)}: no voxels")
                return
            r, g, b = self.atlas.structures[sid]["rgb"]
            base = (r / 255, g / 255, b / 255)
            gi = gl.GLMeshItem(vertexes=m["vertices"], faces=m["faces"], smooth=True,
                               shader=render.BRIGHT_SHADER, glOptions="opaque")
            self._base_color[gi] = base
            self._mesh_data[gi] = (m["vertices"], m["faces"])
            gi.setColor((*base, self._alpha()))
            self.tree_items[sid] = gi
            self.view.addItem(gi)
            self._apply_clip(gi)
        self._tree_on[sid] = on
        if sid in self.tree_items:
            self.tree_items[sid].setVisible(on and self._kim_master())

    # -- shared controls ----------------------------------------------------
    def _alpha(self) -> float:
        return self.opacity.value() / 100.0

    def _on_opacity_change(self, _v=None):
        a = self._alpha()
        self.opacity_label.setText(f"<b>Region opacity</b>: {self.opacity.value()}%")
        for item, base in self._base_color.items():
            item.setColor((*base, a))
            item.setGLOptions("opaque" if a >= 0.98 else "translucent")

    def _toggle_outline(self, on: bool):
        if self.outline_item is not None:
            self.outline_item.setVisible(on and self._kim_master())

    def _outline_alpha(self) -> float:
        return self.outline_op.value() / 100.0

    def _on_outline_opacity(self, _v=None):
        self.outline_op_label.setText(f"Outline opacity: {self.outline_op.value()}%")
        if self.outline_item is not None:
            self.outline_item.setColor((0.7, 0.72, 0.8, self._outline_alpha()))

    def _toggle_axis(self, on: bool):
        if on and self.axis is None:
            bo = self.atlas.brain_outline()
            lo, hi = bo["vertices"].min(0), bo["vertices"].max(0)
            self.axis = gl.GLAxisItem()
            self.axis.setSize(*(hi - lo))
            self.axis.translate(*lo)
            self.view.addItem(self.axis)
        if self.axis is not None:
            self.axis.setVisible(on and self._kim_master())

    def _fit_camera(self):
        verts = self._fit_verts
        if verts is None:
            return
        center = verts.mean(axis=0)
        diag = float(np.linalg.norm(verts.max(axis=0) - verts.min(axis=0)))
        self.view.opts["center"] = pg.Vector(*center)
        self.view.setCameraPosition(distance=diag * 1.3, **_LATERAL_VIEW)

    # -- section / slicing --------------------------------------------------
    def _build_section_controls(self, form):
        self.section_cb = QtWidgets.QCheckBox("Enable section")
        self.section_cb.toggled.connect(self._on_section_toggle)
        form.addWidget(self.section_cb)

        box = QtWidgets.QWidget()
        bl = QtWidgets.QVBoxLayout(box)
        bl.setContentsMargins(0, 0, 0, 0)

        self.orient_combo = QtWidgets.QComboBox()
        self.orient_combo.addItems(list(section_mod.ORIENTATIONS.keys()))
        self.orient_combo.currentTextChanged.connect(self._on_orient_change)
        bl.addWidget(self.orient_combo)

        self.tilt_u = QtWidgets.QDoubleSpinBox()
        self.tilt_v = QtWidgets.QDoubleSpinBox()
        self.tilt_u_lbl = QtWidgets.QLabel()
        self.tilt_v_lbl = QtWidgets.QLabel()
        for w in (self.tilt_u, self.tilt_v):
            w.setRange(-60, 60)
            w.setSuffix(" deg")
            w.valueChanged.connect(self._on_tilt_change)
        # Claude added 2026-10-21: effective-tilt readout in line with each spin-box.
        # These report the plane normal's actual A-P / M-L tilt vs the cardinal
        # orientation, which is the only place the oblique-reslice angles appear.
        self.tilt_u_val = QtWidgets.QLabel("")
        self.tilt_v_val = QtWidgets.QLabel("")
        bl.addWidget(self.tilt_u_lbl)
        _tu_row = QtWidgets.QHBoxLayout()
        _tu_row.addWidget(self.tilt_u)
        _tu_row.addWidget(self.tilt_u_val)
        bl.addLayout(_tu_row)
        bl.addWidget(self.tilt_v_lbl)
        _tv_row = QtWidgets.QHBoxLayout()
        _tv_row.addWidget(self.tilt_v)
        _tv_row.addWidget(self.tilt_v_val)
        bl.addLayout(_tv_row)

        self.thick_lbl = QtWidgets.QLabel("Thickness: 40 um")
        bl.addWidget(self.thick_lbl)
        self.thick = StepSlider()
        self.thick.setRange(int(self.atlas.voxel_size), 500)
        self.thick.setValue(int(self.atlas.voxel_size * 2))
        self.thick.setSingleStep(int(self.atlas.voxel_size))
        self.thick.valueChanged.connect(self._update_section)
        bl.addWidget(self.thick)

        self.pos_lbl = QtWidgets.QLabel("Position: 0 um")
        bl.addWidget(self.pos_lbl)
        self.pos = StepSlider()
        self.pos.setRange(-7000, 7000)
        self.pos.setSingleStep(int(self.atlas.voxel_size))
        self.pos.valueChanged.connect(self._update_section)
        bl.addWidget(self.pos)

        self.outlines_cb = QtWidgets.QCheckBox("Show region outlines")
        self.outlines_cb.setChecked(True)
        self.outlines_cb.toggled.connect(self._update_section)
        bl.addWidget(self.outlines_cb)

        self.smooth_cb = QtWidgets.QCheckBox("Smooth outlines (vs jagged)")
        self.smooth_cb.setChecked(True)
        self.smooth_cb.toggled.connect(self._update_section)
        bl.addWidget(self.smooth_cb)

        self.clip_cb = QtWidgets.QCheckBox("Clip 3D view to slab")
        self.clip_cb.toggled.connect(self._on_clip_toggle)
        bl.addWidget(self.clip_cb)

        # Claude added 2026-10-22: calibrate the Bregma/Stereotaxic origin by
        # clicking bregma on a slice, then pressing this (uses the last pick).
        self.bregma_btn = QtWidgets.QPushButton("Set bregma = last picked point")
        self.bregma_btn.clicked.connect(self._set_bregma_from_pick)
        bl.addWidget(self.bregma_btn)

        # --- oblique reslice from a drawn line ---
        bl.addWidget(QtWidgets.QLabel("<b>Oblique from line</b>"))
        lrow = QtWidgets.QHBoxLayout()
        self.line_draw_btn = QtWidgets.QPushButton("Draw section line")
        self.line_draw_btn.clicked.connect(self._on_draw_section_line)
        self.line_clear_btn = QtWidgets.QPushButton("Clear")
        self.line_clear_btn.clicked.connect(self._on_clear_section_line)
        lrow.addWidget(self.line_draw_btn)
        lrow.addWidget(self.line_clear_btn)
        bl.addLayout(lrow)
        self.line_reslice_btn = QtWidgets.QPushButton("Reslice ⊥ line")
        self.line_reslice_btn.setEnabled(False)
        self.line_reslice_btn.clicked.connect(self._on_reslice_orthogonal)
        bl.addWidget(self.line_reslice_btn)
        self.line_help = QtWidgets.QLabel(
            "Draw a line on the slice and drag its endpoints, then 'Reslice ⊥ "
            "line' to cut perpendicular to it. The line also shows in 3D.")
        self.line_help.setWordWrap(True)
        bl.addWidget(self.line_help)

        box.setEnabled(False)
        self.section_box = box
        form.addWidget(box)
        self._refresh_orient_labels()
        try:
            self._refresh_pos_range()
        except Exception:  # noqa: BLE001 - volume may be unavailable at build time
            pass

    def _refresh_orient_labels(self):
        o = section_mod.ORIENTATIONS[self.orient_combo.currentText()]
        self.tilt_u_lbl.setText(o["tilt_u"] + " (deg)")
        self.tilt_v_lbl.setText(o["tilt_v"] + " (deg)")

    def _current_section(self) -> section_mod.Section:
        if self._oblique_section is not None:
            o = self._oblique_section
            sec = section_mod.Section(
                orientation="Oblique", position=float(self.pos.value()),
                thickness=float(self.thick.value()), center=o["mid"])
            return sec.set_basis(o["n"], o["u"], o["v"])
        return section_mod.Section(
            orientation=self.orient_combo.currentText(),
            tilt_u=self.tilt_u.value(), tilt_v=self.tilt_v.value(),
            position=float(self.pos.value()), thickness=float(self.thick.value()),
            center=self.atlas.world_center())

    def _refresh_pos_range(self):
        n, _, _ = self._current_section().basis()
        mn, mx = self.atlas.world_bbox()
        center = (mn + mx) / 2.0
        corners = np.array([[x, y, z] for x in (mn[0], mx[0])
                            for y in (mn[1], mx[1]) for z in (mn[2], mx[2])])
        hs = int(np.abs((corners - center) @ n).max())
        cur = self.pos.value()
        self.pos.blockSignals(True)
        self.pos.setRange(-hs, hs)
        self.pos.setValue(max(-hs, min(hs, cur)))
        self.pos.blockSignals(False)

    def _on_orient_change(self, _=None):
        # picking a cardinal orientation leaves any oblique (line) reslice; the
        # drawn line is tied to the old basis, so discard it too
        self._oblique_section = None
        if self.section_plate is not None:
            self.section_plate.clear_section_line()
        self._refresh_orient_labels()
        self._refresh_pos_range()
        self._update_section(fit=True)

    def _on_tilt_change(self, _=None):
        # adjusting a tilt changes the basis: drop the line and leave oblique mode
        if self.section_plate is not None:
            self.section_plate.clear_section_line()
        if self._oblique_section is not None:
            self._exit_oblique()
        else:
            self._update_section(fit=True)

    def _on_section_toggle(self, on: bool):
        self.section_box.setEnabled(on)
        if self._plane_item is not None:
            self._plane_item.setVisible(on)
        if self._section_line_item is not None:
            self._section_line_item.setVisible(on and self._line_world is not None)
        if self._section_pts_item is not None:
            self._section_pts_item.setVisible(on and self._line_pts_present)
        if self.on_section_visibility is not None:
            self.on_section_visibility(on)
        if on:
            self._update_section(fit=True)
        elif self.clip_cb.isChecked():
            self._restore_all_faces()

    # Claude fixed 2026-10-21: fit=True refits the slice view (basis changed or
    # first display); the default keeps the current zoom so position/thickness
    # scrolling no longer snaps back to the whole brain.
    def _update_section(self, _=None, fit: bool = False):
        if not self.section_cb.isChecked():
            return
        self.thick_lbl.setText(f"Thickness: {self.thick.value()} um")
        self.pos_lbl.setText(f"Position: {self.pos.value()} um")
        sec = self._current_section()
        self._update_tilt_readout(sec)
        if self.section_plate is not None:
            res = self.atlas.sample_section(sec)
            name = "Oblique ⊥ line" if self._oblique_section is not None else sec.orientation
            title = (f"{name}  pos={int(sec.position)}um  "
                     f"thick={int(sec.thickness)}um")
            self.section_plate.set_section(res, self.atlas.display_name, title,
                                           show_outlines=self.outlines_cb.isChecked(),
                                           smooth=self.smooth_cb.isChecked(), section=sec,
                                           fit=fit)
        self._update_plane_indicator(sec)
        if self.clip_cb.isChecked():
            self._apply_clip_all(sec)

    def _effective_tilts(self, sec: section_mod.Section) -> tuple[float, float]:
        """(tilt_u_deg, tilt_v_deg) of the plane normal relative to the selected
        cardinal orientation, matching the tilt_u / tilt_v spin-box semantics."""
        o = section_mod.ORIENTATIONS[self.orient_combo.currentText()]
        n0 = np.asarray(o["normal"], float)
        u0 = np.asarray(o["u"], float)
        v0 = np.asarray(o["v"], float)
        n, _, _ = sec.basis()
        tu = np.degrees(np.arctan2(float(n @ v0), float(n @ n0)))  # tip about u0
        tv = np.degrees(np.arctan2(float(n @ u0), float(n @ n0)))  # tip about v0
        return float(tu), float(tv)

    def _update_tilt_readout(self, sec: section_mod.Section):
        tu, tv = self._effective_tilts(sec)
        self.tilt_u_val.setText(f"=  {tu:+.1f}°")
        self.tilt_v_val.setText(f"=  {tv:+.1f}°")

    def _update_plane_indicator(self, sec: section_mod.Section):
        # Claude fixed 2026-10-21: build the quad from asymmetric bounds about the
        # actual plane center so an off-center (oblique) plane is not clipped on
        # the far (anterior) side.
        # hu, hv = self.atlas._plane_halfspan(*sec.basis()[1:])
        # c = sec.corners(hu, hv).astype(np.float32)
        _, u, v = sec.basis()
        c0 = sec.plane_center()
        u0, u1, v0, v1 = self.atlas._plane_bounds(u, v, c0)
        c = np.array([c0 + u0 * u + v0 * v, c0 + u1 * u + v0 * v,
                      c0 + u1 * u + v1 * v, c0 + u0 * u + v1 * v], dtype=np.float32)
        faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
        if self._plane_item is None:
            self._plane_item = gl.GLMeshItem(vertexes=c, faces=faces, smooth=False,
                                             color=(0.95, 0.9, 0.3, 0.22), glOptions="additive")
            self._plane_item.setDepthValue(2)
            self.view.addItem(self._plane_item)
        else:
            self._plane_item.setMeshData(vertexes=c, faces=faces)
        self._plane_item.setVisible(self.section_cb.isChecked())

    def _clip_item(self, item, sec: section_mod.Section):
        if item not in self._mesh_data:
            return
        verts, faces = self._mesh_data[item]
        n, _, _ = sec.basis()
        cf = section_mod.clip_faces(verts, faces, sec.plane_center(), n, sec.thickness)
        item.setMeshData(vertexes=verts, faces=cf)

    def _apply_clip(self, item):
        if self.section_cb.isChecked() and self.clip_cb.isChecked():
            self._clip_item(item, self._current_section())

    def _apply_clip_all(self, sec: section_mod.Section):
        for item in list(self._mesh_data.keys()):
            self._clip_item(item, sec)

    def _restore_all_faces(self):
        for item, (verts, faces) in self._mesh_data.items():
            item.setMeshData(vertexes=verts, faces=faces)

    def _on_clip_toggle(self, on: bool):
        if on and self.section_cb.isChecked():
            self._apply_clip_all(self._current_section())
        else:
            self._restore_all_faces()

    # -- oblique reslice from a drawn line ----------------------------------
    def _on_draw_section_line(self):
        if not self.section_cb.isChecked():
            self.line_help.setText("Enable section first, then draw a line.")
            return
        if self.section_plate is not None and self.section_plate.start_section_line():
            self.line_help.setText(
                "Click the first point on the slice, double-click to lock it, "
                "then scroll to another section and place the second point.")

    def _on_clear_section_line(self):
        if self.section_plate is not None:
            self.section_plate.clear_section_line()
        self._exit_oblique()
        self.line_help.setText("Draw a line on the slice, then 'Reslice ⊥ line'.")

    def _on_section_line_changed(self, w0, w1):
        """Plate reports the line endpoints in world coords. Either may be None
        until placed (the points can live on different sections)."""
        both = w0 is not None and w1 is not None
        self._line_world = (np.asarray(w0, float), np.asarray(w1, float)) if both else None
        self.line_reslice_btn.setEnabled(both)
        self._update_section_line_3d(w0, w1)
        if both:
            self.line_help.setText(self._line_angle_summary(w0, w1))
            if self._oblique_section is not None:   # keep a live reslice in sync
                self._apply_oblique_from_line()
        elif w0 is not None:
            self.line_help.setText(
                "First point set — scroll the Position slider to another section "
                "and click to place the second point.")

    def _line_angle_summary(self, w0, w1) -> str:
        n, u, v = self._current_section().basis()
        d = np.asarray(w1, float) - np.asarray(w0, float)
        du, dv, dn = float(d @ u), float(d @ v), float(d @ n)
        in_plane = np.degrees(np.arctan2(dv, du))
        out_plane = np.degrees(np.arctan2(dn, float(np.hypot(du, dv))))
        return (f"Line {float(np.linalg.norm(d)):.0f} µm — in-plane {in_plane:+.1f}°, "
                f"out-of-plane {out_plane:+.1f}° vs the current plane. "
                f"'Reslice ⊥ line' cuts perpendicular to it.")

    def _update_section_line_3d(self, w0, w1):
        # endpoints (scatter): show a single point as soon as it is placed
        placed = [p for p in (w0, w1) if p is not None]
        self._line_pts_present = bool(placed)
        if placed:
            arr = np.array(placed, dtype=np.float32)
            if self._section_pts_item is None:
                self._section_pts_item = gl.GLScatterPlotItem(
                    pos=arr, color=(0.4, 0.9, 1.0, 1.0), size=11)
                self._section_pts_item.setDepthValue(5)
                self.view.addItem(self._section_pts_item)
            else:
                self._section_pts_item.setData(pos=arr)
                if self._section_pts_item not in self.view.items:
                    self.view.addItem(self._section_pts_item)
            self._section_pts_item.setVisible(self.section_cb.isChecked())
        elif self._section_pts_item is not None:
            self._section_pts_item.setVisible(False)
        # connecting segment: only once both endpoints exist
        if w0 is not None and w1 is not None:
            seg = np.array([w0, w1], dtype=np.float32)
            if self._section_line_item is None:
                self._section_line_item = gl.GLLinePlotItem(
                    pos=seg, color=(0.31, 0.78, 1.0, 1.0), width=3, antialias=True)
                self._section_line_item.setDepthValue(4)
                self.view.addItem(self._section_line_item)
            else:
                self._section_line_item.setData(pos=seg)
                if self._section_line_item not in self.view.items:
                    self.view.addItem(self._section_line_item)
            self._section_line_item.setVisible(self.section_cb.isChecked())
        elif self._section_line_item is not None:
            self._section_line_item.setVisible(False)

    def _oblique_basis_from_line(self):
        """Orthonormal (n, u, v, mid) for a plane ⊥ the drawn line.

        Claude fixed 2026-10-21: the in-plane axes are the SOURCE (cardinal) plate's
        u, v projected onto the new plane (Gram-Schmidt), so the resliced view is a
        minimally-rotated continuation of the plate the line was drawn on (e.g. from
        Horizontal: Anterior stays up, +X/lateral toward the right). The previous
        dorsal cross-product put A-P on the horizontal axis and rotated the view.
        """
        w0, w1 = self._line_world
        d = np.asarray(w1, float) - np.asarray(w0, float)
        norm = float(np.linalg.norm(d))
        if norm < 1e-6:
            return None
        n = d / norm
        # source cardinal plate axes (the plane the line was drawn on)
        src = section_mod.Section(
            orientation=self.orient_combo.currentText(),
            tilt_u=self.tilt_u.value(), tilt_v=self.tilt_v.value(),
            center=self.atlas.world_center())
        _, u_src, v_src = src.basis()
        # keep "up" = source v projected into the plane; fall back to u if parallel
        v = v_src - float(v_src @ n) * n
        if np.linalg.norm(v) < 1e-6:
            v = u_src - float(u_src @ n) * n
        v /= np.linalg.norm(v)
        # u = source u orthogonalized against n and v (so it stays near +X/lateral)
        u = u_src - float(u_src @ n) * n - float(u_src @ v) * v
        if np.linalg.norm(u) < 1e-6:
            u = np.cross(v, n)
        u /= np.linalg.norm(u)
        mid = 0.5 * (np.asarray(w0, float) + np.asarray(w1, float))
        return n, u, v, mid

    def _apply_oblique_from_line(self, reset_pos: bool = False):
        basis = self._oblique_basis_from_line()
        if basis is None:
            self.line_help.setText("The line is too short — drag its endpoints apart.")
            return
        n, u, v, mid = basis
        self._oblique_section = {"n": n, "u": u, "v": v, "mid": mid}
        if reset_pos:
            self.pos.blockSignals(True)
            self.pos.setValue(0)
            self.pos.blockSignals(False)
        self._refresh_pos_range()
        self._update_section(fit=True)

    def _on_reslice_orthogonal(self):
        if self._line_world is None:
            self.line_help.setText("Draw a line first.")
            return
        self._apply_oblique_from_line(reset_pos=True)
        self.line_help.setText(
            "Resliced ⊥ the line. Position/thickness still apply; adjust an "
            "orientation or tilt (or Clear) to leave the oblique plane.")

    def _exit_oblique(self):
        if self._oblique_section is None:
            return
        self._oblique_section = None
        self._refresh_pos_range()
        self._update_section(fit=True)

    # -- model on/off + alternation ----------------------------------------
    def _build_model_controls(self, form):
        form.addWidget(QtWidgets.QLabel("<b>Show models</b>"))
        self.kim_model_cb = QtWidgets.QCheckBox("Kim brain + regions")
        self.kim_model_cb.setChecked(True)
        self.kim_model_cb.toggled.connect(self._on_kim_model_toggle)
        form.addWidget(self.kim_model_cb)

        self.muniak_cb = QtWidgets.QCheckBox("Muniak CN overlay")
        self.muniak_cb.toggled.connect(self._toggle_muniak)
        if self.muniak_root is None:
            self.muniak_cb.setEnabled(False)
            self.muniak_cb.setText("Muniak CN overlay (no atlas found)")
        form.addWidget(self.muniak_cb)

        self.alt_btn = QtWidgets.QPushButton("Alternate (1 s)")
        self.alt_btn.setCheckable(True)
        self.alt_btn.toggled.connect(self._toggle_alternation)
        form.addWidget(self.alt_btn)

        self._alt_timer = QtCore.QTimer()
        self._alt_timer.setInterval(1000)
        self._alt_timer.timeout.connect(self._alt_tick)
        self._alt_phase = 0

    def _kim_master(self) -> bool:
        return self.kim_model_cb.isChecked()

    def _set_kim_visible(self, on: bool):
        if self.outline_item is not None:
            self.outline_item.setVisible(on and self.outline_cb.isChecked())
        for g, it in self.group_items.items():
            it.setVisible(on and self.group_cbs[g].isChecked())
        for sid, it in self.tree_items.items():
            it.setVisible(on and self._tree_on.get(sid, False))
        # Claude fixed 2026-10-21: the section sheet (plane indicator, drawn line,
        # endpoints) is logically independent of the Kim model; gate it on the
        # Section toggle only so it stays visible over a Muniak-only view when the
        # Kim model is switched off.
        # if self._plane_item is not None:
        #     self._plane_item.setVisible(on and self.section_cb.isChecked())
        # sec_on = on and self.section_cb.isChecked()
        if self._plane_item is not None:
            self._plane_item.setVisible(self.section_cb.isChecked())
        sec_on = self.section_cb.isChecked()
        if self._section_line_item is not None:
            self._section_line_item.setVisible(sec_on and self._line_world is not None)
        if self._section_pts_item is not None:
            self._section_pts_item.setVisible(sec_on and self._line_pts_present)

    def _set_muniak_visible(self, on: bool):
        for it in self.muniak_items:
            it.setVisible(on)

    def _on_kim_model_toggle(self, on: bool):
        self._set_kim_visible(on)

    def _toggle_alternation(self, on: bool):
        if on:
            if not self.muniak_items and self.muniak_root is not None:
                self._rebuild_muniak(force=True)
            self._alt_phase = 0
            self._alt_tick()
            self._alt_timer.start()
        else:
            self._alt_timer.stop()
            self._set_kim_visible(self.kim_model_cb.isChecked())
            self._set_muniak_visible(self.muniak_cb.isChecked())

    def _alt_tick(self):
        self._alt_phase ^= 1
        show_kim = self._alt_phase == 0
        self._set_kim_visible(show_kim)
        self._set_muniak_visible(not show_kim)

    # -- Muniak CN overlay (Step 3 co-registration) -------------------------
    def _build_overlay_controls(self, form):
        """Left-panel Muniak overlay display controls (color/opacity/both-sides).
        Hemisphere + fine-tune placement live on the right panel."""
        form.addWidget(QtWidgets.QLabel("<b>Muniak CN overlay</b>"))
        obox = QtWidgets.QWidget()
        ol = QtWidgets.QVBoxLayout(obox)
        ol.setContentsMargins(0, 0, 0, 0)

        self.muniak_color = QtWidgets.QComboBox()
        self.muniak_color.addItems(["Tonotopic", "Cochleotopic", "Structure"])
        self.muniak_color.currentTextChanged.connect(lambda _=None: self._update_muniak_colors())
        ol.addWidget(self.muniak_color)

        self.muniak_op_label = QtWidgets.QLabel("Overlay opacity: 90%")
        ol.addWidget(self.muniak_op_label)
        self.muniak_op = StepSlider()
        self.muniak_op.setRange(0, 100)
        self.muniak_op.setValue(90)
        self.muniak_op.valueChanged.connect(self._on_muniak_opacity)
        ol.addWidget(self.muniak_op)

        self.muniak_mirror_cb = QtWidgets.QCheckBox("Show both sides")
        self.muniak_mirror_cb.toggled.connect(lambda _=None: self._rebuild_muniak())
        ol.addWidget(self.muniak_mirror_cb)

        obox.setEnabled(False)
        self.muniak_box = obox
        form.addWidget(obox)

    def _build_right_panel(self) -> QtWidgets.QWidget:
        """Right-side panel: Section (slice) controls above Muniak hemisphere +
        fine-tune placement, wrapped in a scroll area so it fits short screens."""
        content = QtWidgets.QWidget()
        rl = QtWidgets.QVBoxLayout(content)
        rl.setContentsMargins(0, 0, 0, 0)

        # --- Section / slice (placed above the Muniak controls) ---
        section_box = CollapsibleBox("Section (slice)", expanded=True)
        self._build_section_controls(section_box)
        rl.addWidget(section_box)

        rl.addWidget(QtWidgets.QLabel("<b>Muniak placement</b>"))

        fbox = QtWidgets.QWidget()
        fl = QtWidgets.QVBoxLayout(fbox)
        fl.setContentsMargins(0, 0, 0, 0)

        row = QtWidgets.QHBoxLayout()
        row.addWidget(QtWidgets.QLabel("Hemisphere"))
        self.hemi_combo = QtWidgets.QComboBox()
        self.hemi_combo.addItems(["Right", "Left"])
        self.hemi_combo.currentTextChanged.connect(lambda _=None: self._rebuild_base())
        row.addWidget(self.hemi_combo)
        fl.addLayout(row)
        self.flip_cb = QtWidgets.QCheckBox("Flip L-R (handedness)")
        self.flip_cb.toggled.connect(lambda _=None: self._rebuild_base())
        fl.addWidget(self.flip_cb)

        fl.addWidget(QtWidgets.QLabel("<b>Fine-tune</b> (arrow keys when focused)"))
        self._nudge_rows = []

        def nudge(text, lo, hi, val, step, suffix):
            lbl = QtWidgets.QLabel(f"{text}: {val}{suffix}")
            sl = StepSlider()
            sl.setRange(lo, hi)
            sl.setValue(val)
            sl.setSingleStep(step)
            sl.valueChanged.connect(self._on_nudge)
            fl.addWidget(lbl)
            fl.addWidget(sl)
            self._nudge_rows.append((sl, lbl, text, suffix))
            return sl

        self.nudge_rot_ml = nudge("Rotate ML", -90, 90, 0, 1, " deg")
        self.nudge_rot_ap = nudge("Rotate AP", -90, 90, 0, 1, " deg")
        self.nudge_rot_dv = nudge("Rotate DV", -90, 90, 0, 1, " deg")
        self.nudge_scale = nudge("Scale", 50, 200, 100, 1, " %")
        self.nudge_tx = nudge("Shift ML", -1500, 1500, 0, 20, " um")
        self.nudge_ty = nudge("Shift AP", -1500, 1500, 0, 20, " um")
        self.nudge_tz = nudge("Shift DV", -1500, 1500, 0, 20, " um")

        self.nudge_reset = QtWidgets.QPushButton("Reset fine-tune")
        self.nudge_reset.clicked.connect(self._reset_nudge)
        fl.addWidget(self.nudge_reset)
        self.save_btn = QtWidgets.QPushButton("Save alignment")
        self.save_btn.clicked.connect(self._save_alignment)
        fl.addWidget(self.save_btn)

        self.muniak_status = QtWidgets.QLabel("")
        self.muniak_status.setWordWrap(True)
        fl.addWidget(self.muniak_status)

        fbox.setEnabled(False)
        self.finetune_box = fbox
        rl.addWidget(fbox)
        rl.addStretch(1)

        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFixedWidth(260)
        scroll.setWidget(content)
        return scroll

    def _ensure_registration(self, rebuild_base: bool = False):
        if self._muniak_orig is None:
            md = muniak_mod.load_frame("Paxinos coronal", self.muniak_root, self.cache_dir)
            self._muniak_orig = {"verts": md["vertices"].astype(float), "faces": md["faces"],
                                 "freq": md["freq"], "place": md["place"]}
        if self._reg is None or rebuild_base:
            side = "left" if self.hemi_combo.currentText() == "Left" else "right"
            self._reg = reg_mod.cached_registration(
                self.atlas, self.muniak_root, self.cache_dir,
                side=side, mirror=self.flip_cb.isChecked())
            self._base_verts = reg_mod.apply_transform(
                self._reg["matrix"], self._muniak_orig["verts"]).astype(np.float32)
            self._base_centroid = self._base_verts.mean(0)
        return self._reg

    def _toggle_muniak(self, on: bool):
        self.muniak_box.setEnabled(on)
        self.finetune_box.setEnabled(on)
        if on and not self.muniak_items:
            self._rebuild_muniak()
        else:
            self._set_muniak_visible(on)

    def _rebuild_base(self):
        self._ensure_registration(rebuild_base=True)
        self._update_muniak_transform()

    def _rebuild_muniak(self, force: bool = False):
        for it in self.muniak_items:
            if it in self.view.items:
                self.view.removeItem(it)
            self._mesh_data.pop(it, None)
        self.muniak_items = []
        if not (self.muniak_cb.isChecked() or force):
            return
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.CursorShape.WaitCursor)
        try:
            reg = self._ensure_registration()
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()
        n = 2 if self.muniak_mirror_cb.isChecked() else 1
        for _ in range(n):
            it = gl.GLMeshItem(smooth=True, shader=render.BRIGHT_SHADER, glOptions="opaque")
            self.view.addItem(it)
            self.muniak_items.append(it)
        self._update_muniak_transform()
        self.muniak_status.setText(
            f"hemi={reg['side']}, scale={reg.get('scale', 0):.3f} (+ fine-tune)")

    def _nudge_matrix(self):
        c = self._base_centroid
        rot = _euler_R(self.nudge_rot_ml.value(), self.nudge_rot_ap.value(),
                       self.nudge_rot_dv.value())
        s = self.nudge_scale.value() / 100.0
        t = np.array([self.nudge_tx.value(), self.nudge_ty.value(), self.nudge_tz.value()], float)
        linear = s * rot
        m = np.eye(4)
        m[:3, :3] = linear
        m[:3, 3] = (c + t) - linear @ c
        return m

    def _update_muniak_transform(self):
        if not self.muniak_items or self._base_verts is None:
            return
        v0 = reg_mod.apply_transform(self._nudge_matrix(), self._base_verts).astype(np.float32)
        of = self._muniak_orig["faces"]
        fac = of[:, [0, 2, 1]] if self._reg["mirror"] else of
        sets = [(v0, fac)]
        if self.muniak_mirror_cb.isChecked():
            vm = reg_mod.mirror_x(v0, self._reg["mid_x"]).astype(np.float32)
            sets.append((vm, of if self._reg["mirror"] else of[:, [0, 2, 1]]))
        for it, (v, f) in zip(self.muniak_items, sets):
            self._mesh_data[it] = (v, f)
        self._update_muniak_colors()

    def _on_nudge(self, _v=None):
        for sl, lbl, text, suffix in self._nudge_rows:
            lbl.setText(f"{text}: {sl.value()}{suffix}")
        self._update_muniak_transform()

    def _reset_nudge(self):
        for sl in (self.nudge_rot_ml, self.nudge_rot_ap, self.nudge_rot_dv,
                   self.nudge_tx, self.nudge_ty, self.nudge_tz):
            sl.setValue(0)
        self.nudge_scale.setValue(100)

    def _update_muniak_colors(self):
        if not self.muniak_items or self._muniak_orig is None:
            return
        mode = self.muniak_color.currentText()
        alpha = self.muniak_op.value() / 100.0
        o = self._muniak_orig
        colors = None
        base = (0.85, 0.85, 0.9)
        if mode != "Structure":
            scalar = o["freq"] if mode == "Tonotopic" else o["place"]
            finite = scalar[np.isfinite(scalar)]
            lo, hi = float(finite.min()), float(finite.max())
            colors = render.scalar_to_rgba(scalar, render.spectral_colormap(), lo, hi, alpha)
        glopt = "opaque" if alpha >= 0.98 else "translucent"
        for it in self.muniak_items:
            v, f = self._mesh_data[it]
            it.setMeshData(vertexes=v, faces=f, vertexColors=colors, smooth=True,
                           shader=render.BRIGHT_SHADER)
            if colors is None:
                it.setColor((*base, alpha))
            it.setGLOptions(glopt)

    def _on_muniak_opacity(self, _v=None):
        self.muniak_op_label.setText(f"Overlay opacity: {self.muniak_op.value()}%")
        self._update_muniak_colors()

    def _overlay_params(self) -> dict:
        return {
            "hemi": self.hemi_combo.currentText(),
            "flip": self.flip_cb.isChecked(),
            "both": self.muniak_mirror_cb.isChecked(),
            "frame": "Paxinos coronal",
            "rot_ml": self.nudge_rot_ml.value(), "rot_ap": self.nudge_rot_ap.value(),
            "rot_dv": self.nudge_rot_dv.value(), "scale": self.nudge_scale.value(),
            "tx": self.nudge_tx.value(), "ty": self.nudge_ty.value(),
            "tz": self.nudge_tz.value(),
        }

    def _restore_overlay(self):
        """Apply the most recent saved alignment (Application Support file first,
        then the config), so a fine-tuned placement is recalled between sessions."""
        saved = alignment_store.load_latest() or self.cfg.get("MuniakOverlay", {}) or {}
        if not saved:
            self._on_nudge()
            return
        widgets = [
            (self.nudge_rot_ml, "rot_ml"), (self.nudge_rot_ap, "rot_ap"),
            (self.nudge_rot_dv, "rot_dv"), (self.nudge_scale, "scale"),
            (self.nudge_tx, "tx"), (self.nudge_ty, "ty"), (self.nudge_tz, "tz"),
        ]
        blockers = [self.hemi_combo, self.flip_cb, self.muniak_mirror_cb, *[s for s, _ in widgets]]
        for w in blockers:
            w.blockSignals(True)
        self.hemi_combo.setCurrentText(saved.get("hemi", "Right"))
        self.flip_cb.setChecked(bool(saved.get("flip", False)))
        self.muniak_mirror_cb.setChecked(bool(saved.get("both", False)))
        for sl, key in widgets:
            if key in saved:
                sl.setValue(int(saved[key]))
        for w in blockers:
            w.blockSignals(False)
        self._on_nudge()

    def _save_alignment(self):
        path = alignment_store.save_alignment(self._overlay_params())
        self.muniak_status.setText(f"Saved alignment -> {path.name} ({alignment_store.load_latest()['timestamp']})")

    def remember(self, prefs: dict):
        prefs["MuniakOverlay"] = self._overlay_params()
        # Claude added 2026-10-22: persist a calibrated bregma + chosen frame.
        if self.cfg.get("KimBregmaWorld") is not None:
            prefs["KimBregmaWorld"] = [float(v) for v in self.cfg["KimBregmaWorld"]]
        if self.section_plate is not None:
            prefs["DefaultCoordFrame"] = self.section_plate.frame_combo.currentText()

    # -- ground truth (adjunct annotations) ---------------------------------
    def _build_gt_controls(self, form):
        self.gt_show_cb = QtWidgets.QCheckBox("Show ground truth (2D + 3D)")
        self.gt_show_cb.toggled.connect(self._gt_toggle_show)
        form.addWidget(self.gt_show_cb)

        row = QtWidgets.QHBoxLayout()
        row.addWidget(QtWidgets.QLabel("Region"))
        self.gt_region_combo = QtWidgets.QComboBox()
        row.addWidget(self.gt_region_combo, stretch=1)
        form.addLayout(row)
        brow = QtWidgets.QHBoxLayout()
        new_btn = QtWidgets.QPushButton("New…")
        new_btn.clicked.connect(self._gt_new_region)
        del_btn = QtWidgets.QPushButton("Delete region")
        del_btn.clicked.connect(self._gt_delete_region)
        brow.addWidget(new_btn)
        brow.addWidget(del_btn)
        form.addLayout(brow)

        drow = QtWidgets.QHBoxLayout()
        draw_btn = QtWidgets.QPushButton("Draw contour")
        draw_btn.clicked.connect(self._gt_start_draw)
        lmk_btn = QtWidgets.QPushButton("Add landmark")
        lmk_btn.clicked.connect(self._gt_add_landmark)
        drow.addWidget(draw_btn)
        drow.addWidget(lmk_btn)
        form.addLayout(drow)
        copy_btn = QtWidgets.QPushButton("Copy nearby contour here")
        copy_btn.clicked.connect(self._gt_copy_here)
        form.addWidget(copy_btn)
        undo_btn = QtWidgets.QPushButton("Delete last contour (this plate)")
        undo_btn.clicked.connect(self._gt_delete_last)
        form.addWidget(undo_btn)

        srow = QtWidgets.QHBoxLayout()
        save_btn = QtWidgets.QPushButton("Save")
        save_btn.clicked.connect(self._gt_save)
        load_btn = QtWidgets.QPushButton("Load")
        load_btn.clicked.connect(self._gt_load)
        srow.addWidget(save_btn)
        srow.addWidget(load_btn)
        form.addLayout(srow)

        self.gt_help = QtWidgets.QLabel(
            "Enable Section, pick a region, then Draw contour. Edit: drag vertices; "
            "shift-click a vertex to select it, then Delete key or right-click to "
            "remove it. 'Copy nearby contour here' clones the closest contour of the "
            "selected region onto this plate as an editable template.")
        self.gt_help.setWordWrap(True)
        form.addWidget(self.gt_help)
        self._gt_refresh_region_combo()

    def _gt_refresh_region_combo(self, select: str = None):
        self.gt_region_combo.blockSignals(True)
        self.gt_region_combo.clear()
        self.gt_region_combo.addItems(sorted(self.gt.regions.keys()))
        if select:
            self.gt_region_combo.setCurrentText(select)
        self.gt_region_combo.blockSignals(False)

    def _gt_new_region(self):
        name, ok = QtWidgets.QInputDialog.getText(self.panel, "New region", "Region name:")
        if not ok or not name.strip():
            return
        name = name.strip()
        col = QtWidgets.QColorDialog.getColor(parent=self.panel)
        color = (col.red(), col.green(), col.blue()) if col.isValid() else (255, 230, 0)
        self.gt.add_region(name, color)
        self._gt_refresh_region_combo(select=name)
        self.gt_help.setText(f"Region '{name}' created.")

    def _gt_delete_region(self):
        name = self.gt_region_combo.currentText()
        if not name:
            return
        self.gt.remove_region(name)
        self._gt_refresh_region_combo()
        self._refresh_gt_3d()
        if self.section_plate is not None:
            self.section_plate.refresh_gt()

    def _gt_start_draw(self):
        if not self.section_cb.isChecked():
            self.gt_help.setText("Enable Section first, navigate to a plate, then Draw.")
            return
        region = self.gt_region_combo.currentText()
        if not region:
            self.gt_help.setText("Create/select a region first (New…).")
            return
        if not self.gt_show_cb.isChecked():
            self.gt_show_cb.setChecked(True)
        self.section_plate.start_draw_contour(region)
        self.gt_help.setText("Click points on the plate; double-click (or right-click) to close.")

    def _gt_add_landmark(self):
        if not self.section_cb.isChecked():
            self.gt_help.setText("Enable Section first.")
            return
        if not self.gt_show_cb.isChecked():
            self.gt_show_cb.setChecked(True)
        self.section_plate.start_place_landmark()
        self.gt_help.setText("Click a point on the plate to place a landmark.")

    def _gt_contour_drawn(self, region, world):
        self.gt.add_contour(region, world, section_mod.section_descriptor(self._current_section()))
        self.section_plate.refresh_gt()
        self._refresh_gt_3d()
        self.gt_help.setText(f"Added contour to '{region}'. Drag vertices to edit; Save when done.")

    def _gt_contour_edited(self, region, cid, world):
        self.gt.update_contour(region, cid, world)
        self._refresh_gt_3d()

    def _gt_landmark_placed(self, world):
        name, ok = QtWidgets.QInputDialog.getText(self.panel, "Landmark", "Landmark name:")
        if ok and name.strip():
            self.gt.add_landmark(name.strip(), world)
            self.section_plate.refresh_gt()
            self._refresh_gt_3d()
            self.gt_help.setText(f"Landmark '{name.strip()}' added.")

    # Claude added 2026-10-22: a plain left-click on the plate records the
    # cursor's world position as an alignment pick, saved to its own JSON store.
    def _point_picked(self, world):
        sec = self._current_section()
        rec = self.picks.add_point(
            world, section=section_mod.section_descriptor(sec))
        self.picks.save()
        if self.section_plate is not None:
            self.section_plate.refresh_picks()
            coord = self.section_plate.format_world(world)
        else:
            coord = (f"ML={world[0] / 1000.0:.3f} AP={world[1] / 1000.0:.3f} "
                     f"DV={world[2] / 1000.0:.3f} mm")
        self.status.setText(
            f"Picked point #{rec['id']} @ {coord} — saved "
            f"({len(self.picks.points)} total)")

    def _gt_copy_here(self):
        if not self.section_cb.isChecked():
            self.gt_help.setText("Enable Section first, navigate to the target plate.")
            return
        region = self.gt_region_combo.currentText()
        if not region:
            self.gt_help.setText("Select a region first.")
            return
        if not self.gt_show_cb.isChecked():
            self.gt_show_cb.setChecked(True)
        if self.section_plate is not None and self.section_plate.copy_contour_here(region):
            self.gt_help.setText(
                f"Copied a nearby '{region}' contour onto this plate — edit to fit, "
                "then Save.")

    def _gt_delete_last(self):
        region = self.gt_region_combo.currentText()
        contours = self.gt.regions.get(region, {}).get("contours", [])
        sec = self._current_section()
        on_plate = [c for c in contours if section_mod.descriptor_matches(sec, c.get("section", {}))]
        if on_plate:
            self.gt.remove_contour(region, on_plate[-1]["id"])
            self.section_plate.refresh_gt()
            self._refresh_gt_3d()
            self.gt_help.setText("Deleted last contour on this plate.")

    def _gt_toggle_show(self, on: bool):
        if self.section_plate is not None:
            self.section_plate.set_gt_visible(on)
        self._refresh_gt_3d()

    def _gt_save(self):
        path = self.gt.save()
        self.gt_help.setText(f"Saved ground truth -> {path}")

    def _gt_load(self):
        self.gt = gt_mod.GroundTruth.load()
        if self.section_plate is not None:
            self.section_plate.set_gt(self.gt)
        self._gt_refresh_region_combo()
        self._refresh_gt_3d()
        self.gt_help.setText("Loaded ground truth.")

    def _refresh_gt_3d(self):
        for it in self._gt_items:
            if it in self.view.items:
                self.view.removeItem(it)
        self._gt_items = []
        if self.gt is None or not self.gt_show_cb.isChecked():
            return
        for info in self.gt.regions.values():
            col = tuple(c / 255 for c in info.get("color", [255, 230, 0])) + (1.0,)
            for c in info["contours"]:
                pts = np.array(c["points"], dtype=np.float32)
                if len(pts) < 2:
                    continue
                pts = np.vstack([pts, pts[0]])   # close the loop
                line = gl.GLLinePlotItem(pos=pts, color=col, width=2, antialias=True)
                line.setDepthValue(3)
                self.view.addItem(line)
                self._gt_items.append(line)
        if self.gt.landmarks:
            lp = np.array([lm["point"] for lm in self.gt.landmarks], dtype=np.float32)
            sp = gl.GLScatterPlotItem(pos=lp, color=(1, 1, 0, 1), size=12)
            sp.setDepthValue(3)
            self.view.addItem(sp)
            self._gt_items.append(sp)
