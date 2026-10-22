"""ColorBar + MuniakViewer view controllers.

KimViewer lives in kim_viewer.py; shared helpers are in widgets.py.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pyqtgraph as pg
import pyqtgraph.opengl as gl
from pyqtgraph.Qt import QtCore, QtWidgets

from . import muniak as muniak_mod
from . import render
from .util import cprint
from .widgets import StepSlider, LATERAL_VIEW as _LATERAL_VIEW


class ColorBar(QtWidgets.QWidget):
    """Narrow vertical colorbar: color banded along the value axis."""

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.label = QtWidgets.QLabel("")
        self.label.setAlignment(QtCore.Qt.AlignmentFlag.AlignHCenter)
        layout.addWidget(self.label)
        self.glw = pg.GraphicsLayoutWidget()
        self.glw.setFixedWidth(96)
        self.glw.setFixedHeight(300)
        layout.addWidget(self.glw)
        layout.addStretch(1)  # keep label + bar packed at the top
        self.bar: Optional[pg.ColorBarItem] = None

    def set_map(self, cmap: pg.ColorMap, lo: float, hi: float, label: str):
        self.label.setText(label)
        self.glw.clear()
        self.bar = pg.ColorBarItem(
            values=(lo, hi), colorMap=cmap, interactive=False, width=14, orientation="vertical"
        )
        self.glw.addItem(self.bar)


# ---------------------------------------------------------------------------
# Muniak
# ---------------------------------------------------------------------------


class MuniakViewer:
    """Muniak CN atlas: translucent surface + tonotopic/cochleotopic maps."""

    COLOR_MODES = ("Structure", "Tonotopic", "Cochleotopic")

    def __init__(self, view, colorbar, cfg, cache_dir, atlas_root, rebuild=False):
        self.view = view
        self.colorbar = colorbar
        self.cfg = cfg
        self.cache_dir = cache_dir
        self.atlas_root = atlas_root
        self.rebuild = rebuild
        self.cmap = render.spectral_colormap()
        self.data: Optional[dict] = None
        self.mesh: Optional[gl.GLMeshItem] = None
        self.axis: Optional[gl.GLAxisItem] = None
        self.panel = self._build_panel()

    # -- panel --------------------------------------------------------------
    def _build_panel(self) -> QtWidgets.QWidget:
        panel = QtWidgets.QWidget()
        form = QtWidgets.QVBoxLayout(panel)
        form.setContentsMargins(0, 0, 0, 0)

        form.addWidget(QtWidgets.QLabel("<b>Coordinate frame</b>"))
        self.frame_combo = QtWidgets.QComboBox()
        self.frame_combo.addItems(list(muniak_mod.FRAMES.keys()))
        self.frame_combo.currentTextChanged.connect(self.load_and_show)
        form.addWidget(self.frame_combo)

        form.addWidget(QtWidgets.QLabel("<b>Color by</b>"))
        self.mode_combo = QtWidgets.QComboBox()
        self.mode_combo.addItems(list(self.COLOR_MODES))
        self.mode_combo.currentTextChanged.connect(self._on_mode_change)
        form.addWidget(self.mode_combo)

        self.opacity_label = QtWidgets.QLabel("<b>Surface opacity</b>")
        form.addWidget(self.opacity_label)
        self.opacity = StepSlider()
        self.opacity.setRange(0, 100)
        self.opacity.setValue(int(float(self.cfg.get("SurfaceOpacity", 0.35)) * 100))
        self.opacity.valueChanged.connect(self._on_opacity_change)
        form.addWidget(self.opacity)
        self._update_opacity_label()

        self.wire_cb = QtWidgets.QCheckBox("Draw edges (wireframe)")
        self.wire_cb.toggled.connect(lambda _=None: self.update_colors())
        form.addWidget(self.wire_cb)

        self.axis_cb = QtWidgets.QCheckBox("Show axes")
        self.axis_cb.setChecked(True)
        self.axis_cb.toggled.connect(self._toggle_axis)
        form.addWidget(self.axis_cb)

        self.status = QtWidgets.QLabel("")
        self.status.setWordWrap(True)
        form.addWidget(self.status)
        form.addStretch(1)

        # Apply remembered defaults WITHOUT triggering loads/renders: nothing may
        # be added to the shared view until activate() is called, or stale items
        # leak across atlas switches.
        for w in (self.frame_combo, self.mode_combo, self.opacity):
            w.blockSignals(True)
        f = self.cfg.get("DefaultFrame", "Original")
        if f in muniak_mod.FRAMES:
            self.frame_combo.setCurrentText(f)
        m = self.cfg.get("DefaultColorMode", "Tonotopic")
        if m in self.COLOR_MODES:
            self.mode_combo.setCurrentText(m)
        self.opacity.setValue(30 if m == "Structure" else 100)
        for w in (self.frame_combo, self.mode_combo, self.opacity):
            w.blockSignals(False)
        self._update_opacity_label()
        return panel

    # -- activation ---------------------------------------------------------
    def activate(self):
        self.colorbar.setVisible(True)
        if self.mesh is None:
            self.load_and_show(self.frame_combo.currentText())
        else:
            self._add_items()
            self._fit_camera()
            self.update_colors()

    def deactivate(self):
        for it in (self.mesh, self.axis):
            if it is not None and it in self.view.items:
                self.view.removeItem(it)

    def _add_items(self):
        for it in (self.mesh, self.axis):
            if it is not None and it not in self.view.items:
                self.view.addItem(it)

    # -- data + rendering ---------------------------------------------------
    def load_and_show(self, frame: str):
        if frame not in muniak_mod.FRAMES:
            return
        self.status.setText(f"Loading {frame} ...")
        QtWidgets.QApplication.processEvents()
        try:
            self.data = muniak_mod.load_frame(frame, self.atlas_root, self.cache_dir, self.rebuild)
        except Exception as exc:  # noqa: BLE001
            self.status.setText(f"<span style='color:#f66'>Load failed: {exc}</span>")
            cprint("r", f"cn_alignment: {exc}")
            return
        self._rebuild_mesh()
        self._fit_camera()
        self.update_colors()

    def _rebuild_mesh(self):
        if self.mesh is not None:
            self.view.removeItem(self.mesh)
        verts, faces = self.data["vertices"], self.data["faces"]
        self.mesh = gl.GLMeshItem(vertexes=verts, faces=faces, smooth=True,
                                  shader="shaded", glOptions="translucent")
        self.view.addItem(self.mesh)

        if self.axis is not None:
            self.view.removeItem(self.axis)
        lo, hi = verts.min(axis=0), verts.max(axis=0)
        self.axis = gl.GLAxisItem()
        self.axis.setSize(*(hi - lo))
        self.axis.translate(*lo)
        self.axis.setVisible(self.axis_cb.isChecked())
        self.view.addItem(self.axis)

    def _fit_camera(self):
        verts = self.data["vertices"]
        center = verts.mean(axis=0)
        diag = float(np.linalg.norm(verts.max(axis=0) - verts.min(axis=0)))
        self.view.opts["center"] = pg.Vector(*center)
        self.view.setCameraPosition(distance=diag * 1.3, **_LATERAL_VIEW)

    def update_colors(self):
        if self.mesh is None or self.data is None:
            return
        mode = self.mode_combo.currentText()
        alpha = self.opacity.value() / 100.0
        verts = self.data["vertices"]
        glopt = "opaque" if (mode != "Structure" and alpha >= 0.98) else "translucent"

        if mode == "Structure":
            colors = np.empty((verts.shape[0], 4), dtype=np.float32)
            colors[:] = (0.75, 0.78, 0.85, alpha)
            self.colorbar.setVisible(False)
        else:
            if mode == "Tonotopic":
                scalar, label = self.data["freq"], self.data["freq_label"] or "Freq-CN, kHz"
            else:
                scalar, label = self.data["place"], self.data["place_label"] or "Place-CN"
            finite = scalar[np.isfinite(scalar)]
            if finite.size == 0:
                self.status.setText("No scalar data for this mode.")
                return
            lo, hi = float(np.nanmin(finite)), float(np.nanmax(finite))
            colors = render.scalar_to_rgba(scalar, self.cmap, lo, hi, alpha)
            self.colorbar.set_map(self.cmap, lo, hi, label)
            self.colorbar.setVisible(True)

        shader = "shaded" if mode == "Structure" else render.BRIGHT_SHADER
        self.mesh.setMeshData(vertexes=verts, faces=self.data["faces"], vertexColors=colors,
                              smooth=True, shader=shader, drawEdges=self.wire_cb.isChecked(),
                              edgeColor=(1, 1, 1, 0.15))
        self.mesh.setGLOptions(glopt)
        self._update_status()

    def _update_status(self):
        d = self.data
        fr = d["freq"][np.isfinite(d["freq"])]
        pl = d["place"][np.isfinite(d["place"])]
        msg = [f"<b>{d['frame']}</b>",
               f"{d['vertices'].shape[0]} vertices, {d['faces'].shape[0]} faces"]
        if fr.size:
            msg.append(f"{d['freq_label'] or 'Freq'}: {fr.min():.2f}-{fr.max():.2f}")
        if pl.size:
            msg.append(f"{d['place_label'] or 'Place'}: {pl.min():.2f}-{pl.max():.2f}")
        self.status.setText("<br>".join(msg))

    def _on_mode_change(self, mode: str):
        self.opacity.blockSignals(True)
        self.opacity.setValue(30 if mode == "Structure" else 100)
        self.opacity.blockSignals(False)
        self._update_opacity_label()
        self.update_colors()

    def _on_opacity_change(self, _value=None):
        self._update_opacity_label()
        self.update_colors()

    def _update_opacity_label(self):
        self.opacity_label.setText(f"<b>Surface opacity</b>: {self.opacity.value()}%")

    def _toggle_axis(self, on: bool):
        if self.axis is not None:
            self.axis.setVisible(on)

    def remember(self, prefs: dict):
        prefs["DefaultFrame"] = self.frame_combo.currentText()
        prefs["DefaultColorMode"] = self.mode_combo.currentText()
        prefs["SurfaceOpacity"] = round(self.opacity.value() / 100.0, 3)


# ---------------------------------------------------------------------------
# Kim
# ---------------------------------------------------------------------------
