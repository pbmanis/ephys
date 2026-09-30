#!/usr/bin/python
"""
ephus_viewer: a small PyQt6 + pyqtgraph test GUI for :mod:`ephys.datareaders.ephus_reader`.

Point it at an ephus experiment "day" directory (e.g. ``031609pbm``) and it
builds a tree of cells (``pkNNNN``) and their maps (``mapNN``).  Selecting a map
reads the single ``.xsg`` through ``ephus_reader`` and shows:

  * the cell's video image with the LSPS map grid overlaid (spots placed either
    from a reference ``lsps_result.mat`` when present, or from the imaging-system
    calibration otherwise);
  * the N per-spot sweeps, stacked, with the stimulus window shaded;
  * the photodiode (laser-power monitor) trace.

Clicking a spot in the image highlights its sweep, and vice-versa.  This is a
read-only diagnostic tool; it modifies no data and no other file in the package.

Run:
    python -m ephys.datareaders.ephus_viewer [EXPERIMENT_DIR]
    python ephys/datareaders/ephus_viewer.py  [EXPERIMENT_DIR]

2026, Paul B. Manis.
"""

import sys
from pathlib import Path

import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets
from pyqtgraph.parametertree import Parameter, ParameterTree

from ephys.datareaders import ephus_reader as ER


def charge_color(q, thr):
    """Map a charge value ``q`` (fC) to an (r, g, b) color: grey at 0, grading to
    red for positive charge (outward/IPSC) and blue for negative (inward/EPSC).
    ``thr`` sets the saturation bound; |q| > thr is clipped to white."""
    if thr <= 0 or not np.isfinite(q):
        return (128, 128, 128)
    if abs(q) > thr:
        return (255, 255, 255)  # clipped / off-scale
    grey = np.array([128, 128, 128])
    frac = abs(q) / thr
    end = np.array([220, 40, 40]) if q >= 0 else np.array([50, 110, 255])
    return tuple(int(v) for v in (grey + frac * (end - grey)))


# ---------------------------------------------------------------------------
# color helpers for the spot overlay
# ---------------------------------------------------------------------------
def _reference_matches(reader, celldir, mapdir):
    """Return the reference result dict if an lsps_result.mat exists for this
    exact map, else None."""
    ref = reader.load_reference_result(celldir)
    xsg = reader.find_xsg(mapdir)
    if ref is None or xsg is None:
        return None
    if str(ref.get("datafile", "")).replace("\\", "/").endswith(xsg.name):
        return ref
    return None


def _response_brushes(reader, celldir, mapdir, image_shape, nspots):
    """Return (brushes, detected_mask, source_str), colored per acquisition
    sweep from the reference detection when available (matched by pixel
    proximity, so it stays correct even if the grid is moved), else neutral.
    """
    neutral = pg.mkBrush(80, 160, 255, 35)  # very translucent blue
    brushes = [neutral] * nspots
    detected = np.zeros(nspots, dtype=bool)
    ref = _reference_matches(reader, celldir, mapdir)
    if ref is None or image_shape is None:
        return brushes, detected, "no reference"
    detn = np.asarray(ref["current_detected"]).astype(int).ravel()
    cur = np.asarray(ref["current_maximum"]).astype(float).ravel()
    refpix = np.column_stack(
        [np.asarray(ref["mapper"]["map_locationx"]).ravel(),
         np.asarray(ref["mapper"]["map_locationy"]).ravel()]
    )
    mypix = reader.map_pixels(mapdir, image_shape)  # default placement == reference
    cmax = np.max(np.abs(cur)) if np.any(cur) else 1.0
    brushes = []
    for i in range(nspots):
        j = int(np.argmin(np.hypot(refpix[:, 0] - mypix[i, 0], refpix[:, 1] - mypix[i, 1])))
        if j < len(detn) and detn[j] == 1:
            detected[i] = True
            frac = min(1.0, abs(cur[j]) / cmax)
            brushes.append(pg.mkBrush(255, int(80 * (1 - frac)), 40, 120))  # hot = bigger
        else:
            brushes.append(pg.mkBrush(70, 90, 120, 30))
    return brushes, detected, "reference lsps_result.mat"


def _spot_pixels(reader, celldir, mapdir, image_shape, center_px=None):
    """Return (pixels Nx2, soma_xy or None, source_str) in image pixel coords.

    Placement always uses the reader's lsps-formula ``map_pixels`` (which
    reproduces the reference pixels exactly); ``center_px`` (a user click)
    re-centers the grid there.  The soma marker is the registered
    soma1Coordinates, or None when it was never set.
    """
    pix = reader.map_pixels(mapdir, image_shape, center_px=center_px)
    soma = reader.soma_pixels(mapdir, image_shape)
    if center_px is not None:
        src = "manual center (click)"
    elif _reference_matches(reader, celldir, mapdir) is not None:
        src = "lsps calibration (matches reference)"
    else:
        src = "lsps calibration"
    return pix, soma, src


# ---------------------------------------------------------------------------
class EphusViewer(QtWidgets.QMainWindow):
    def __init__(self, startdir=None):
        super().__init__()
        self.reader = ER.ephus_reader()
        self.setWindowTitle("Ephus / XSG map viewer")
        self.resize(1400, 900)

        # --- left: directory tree ------------------------------------------
        self.tree = QtWidgets.QTreeWidget()
        self.tree.setHeaderLabels(["cell / map"])
        self.tree.setMinimumWidth(260)
        self.tree.itemClicked.connect(self._on_tree_click)

        openbtn = QtWidgets.QPushButton("Open experiment folder…")
        openbtn.clicked.connect(self._choose_dir)
        # full path to the loaded data directory, shown above the picker
        self.path_label = QtWidgets.QLabel("(no folder loaded)")
        self.path_label.setWordWrap(True)
        self.path_label.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
        self.path_label.setStyleSheet("QLabel { color: #446; font-size: 11px; }")
        self.place_cb = QtWidgets.QCheckBox("Click image to set map center")
        self.info = QtWidgets.QLabel("No map loaded")
        self.info.setWordWrap(True)

        # charge-measurement controls (parametertree) below the file/map list
        self.charge_params = Parameter.create(
            name="Charge measure",
            type="group",
            children=[
                {"name": "start_after_stim", "title": "Start after stim (s)",
                 "type": "float", "value": 0.008, "step": 0.001, "limits": (0.0, 10.0)},
                {"name": "end_after_stim", "title": "End after stim (s)",
                 "type": "float", "value": 0.100, "step": 0.005, "limits": (0.0, 10.0)},
                {"name": "threshold_fC", "title": "Charge threshold (fC)",
                 "type": "float", "value": 1.0e4, "step": 1.0e3, "limits": (1.0, 1.0e9)},
                {"name": "spot_opacity", "title": "Spot opacity", "type": "float",
                 "value": 0.4, "step": 0.05, "limits": (0.0, 1.0)},
                {"name": "spot_size_scale", "title": "Spot size (x)", "type": "float",
                 "value": 1.0, "step": 0.1, "limits": (0.1, 2.0)},
            ],
        )
        self.ptree = ParameterTree(showHeader=False)
        self.ptree.setParameters(self.charge_params, showTop=True)
        self.ptree.setMaximumHeight(200)
        self.charge_params.sigTreeStateChanged.connect(self._on_charge_params_changed)

        left = QtWidgets.QWidget()
        ll = QtWidgets.QVBoxLayout(left)
        ll.addWidget(openbtn)
        ll.addWidget(self.path_label)
        ll.addWidget(self.tree, 1)
        ll.addWidget(self.ptree)
        ll.addWidget(self.place_cb)
        ll.addWidget(self.info)

        # --- right: image + traces -----------------------------------------
        self.glw = pg.GraphicsLayoutWidget()
        self.img_vb = self.glw.addViewBox(row=0, col=0, lockAspect=True, invertY=True)
        self.img_item = pg.ImageItem()
        self.img_vb.addItem(self.img_item)
        # pxMode=False -> spot size is in image pixels, so spots stay fixed
        # relative to the image (they scale with zoom).
        self.scatter = pg.ScatterPlotItem(pxMode=False, pen=pg.mkPen((0, 0, 0, 70), width=0.4))
        self.scatter.sigClicked.connect(self._on_spot_click)
        self.img_vb.addItem(self.scatter)
        self.soma_item = pg.ScatterPlotItem(
            size=18, symbol="+", pen=pg.mkPen("y", width=2), brush=None
        )
        self.img_vb.addItem(self.soma_item)

        self.trace_plot = self.glw.addPlot(row=1, col=0)
        self.trace_plot.setLabel("bottom", "Time", "s")
        self.trace_plot.setLabel("left", "sweep (stacked)")
        self.trace_plot.showGrid(x=True, y=False)

        self.pd_plot = self.glw.addPlot(row=2, col=0)
        self.pd_plot.setLabel("bottom", "Time", "s")
        self.pd_plot.setLabel("left", "acquirer (photodiode1)")
        self.pd_plot.setMaximumHeight(160)
        self.pd_plot.setXLink(self.trace_plot)

        self.glw.ci.layout.setRowStretchFactor(0, 5)
        self.glw.ci.layout.setRowStretchFactor(1, 5)
        self.glw.ci.layout.setRowStretchFactor(2, 2)

        splitter = QtWidgets.QSplitter()
        splitter.addWidget(left)
        splitter.addWidget(self.glw)
        splitter.setStretchFactor(1, 1)
        self.setCentralWidget(splitter)

        self._trace_curves = []
        self._offsets = None
        self._nspots = 0
        self._img_shape = None
        self._map_center = None
        self._celldir = self._mapdir = None
        self._brushes = []
        self._laser = None
        self._vc = {}
        self._amp = {}
        self._pix_src = ""
        self._resp_src = ""
        self._spot_size = 14.0
        self._charge = None  # per-sweep charge (fC)
        self.glw.scene().sigMouseClicked.connect(self._on_scene_click)

        if startdir is not None and Path(startdir).is_dir():
            self._load_experiment(Path(startdir))

    # -- directory handling -------------------------------------------------
    def _choose_dir(self):
        d = QtWidgets.QFileDialog.getExistingDirectory(self, "Choose ephus experiment folder")
        if d:
            self._load_experiment(Path(d))

    def _load_experiment(self, expdir: Path):
        """Build the cell/map tree by finding every xsg under the folder."""
        self.tree.clear()
        self.expdir = expdir
        self.path_label.setText(str(expdir))
        self.path_label.setToolTip(str(expdir))
        cells = {}
        for xsg in sorted(expdir.rglob("*.xsg")):
            mapdir = xsg.parent
            celldir = mapdir.parent
            cells.setdefault(celldir, []).append(mapdir)
        for celldir in sorted(cells):
            label = str(celldir.relative_to(expdir)) if celldir != expdir else celldir.name
            citem = QtWidgets.QTreeWidgetItem([label or celldir.name])
            citem.setData(0, QtCore.Qt.ItemDataRole.UserRole, ("cell", str(celldir)))
            for mapdir in sorted(set(cells[celldir])):
                mitem = QtWidgets.QTreeWidgetItem([mapdir.name])
                mitem.setData(
                    0, QtCore.Qt.ItemDataRole.UserRole, ("map", str(celldir), str(mapdir))
                )
                citem.addChild(mitem)
            self.tree.addTopLevelItem(citem)
        self.tree.expandAll()
        self.setWindowTitle(f"Ephus / XSG map viewer — {expdir.name}")

    def _on_tree_click(self, item, _col):
        data = item.data(0, QtCore.Qt.ItemDataRole.UserRole)
        if data and data[0] == "map":
            self._show_map(Path(data[1]), Path(data[2]))

    # -- display ------------------------------------------------------------
    def _show_map(self, celldir: Path, mapdir: Path):
        self.reader.setProtocol(mapdir)
        try:
            if not self.reader.getData():
                self.info.setText(f"Could not read {mapdir}")
                return
        except Exception as exc:  # keep the GUI alive on bad files
            self.info.setText(f"Error reading {mapdir}:\n{exc}")
            return
        self.reader.getScannerPositions()
        self.reader.getPhotodiode()
        self._nspots = self.reader.data_array.shape[0]
        self._celldir, self._mapdir = celldir, mapdir
        self._map_center = None  # reset any manual placement for the new map
        self._laser = self.reader.laser_pulse(mapdir)
        self._vc = self.reader.getStim()
        self._amp = self.reader.amplifier_info(mapdir)

        # image
        img = None
        try:
            img = self.reader.getImage(celldir)
        except FileNotFoundError:
            pass
        if img is not None:
            # ImageItem expects (col, row) orientation; transpose the 2D frame
            self.img_item.setImage(np.asarray(img).T, autoLevels=True)
            self._img_shape = np.asarray(img).shape
        else:
            self.img_item.clear()
            self._img_shape = None

        self._charge = self._compute_charge()  # per-sweep charge (fC)
        self._brushes = self._charge_brushes()
        self._place_spots()  # initial overlay placement
        self._plot_traces()
        self._plot_photodiode()
        self._update_info()

    # -- charge measurement -------------------------------------------------
    def _compute_charge(self):
        """Integrated (baseline-subtracted) charge per sweep, in femtocoulombs,
        over [stim + start, stim + end] where stim is the laser onset."""
        if getattr(self.reader, "data_array", None) is None or self._nspots == 0:
            return None
        data = np.asarray(self.reader.data_array, dtype=float)  # Amperes
        sr = float(self.reader.sample_rate[0])
        laser = self._laser
        t0 = float(laser["start"][0]) if (laser and laser.get("start")) else 0.0
        start = t0 + float(self.charge_params["start_after_stim"])
        end = t0 + float(self.charge_params["end_after_stim"])
        i0 = max(0, int(round(start * sr)))
        i1 = min(data.shape[1], int(round(end * sr)))
        b1 = max(0, int(round(t0 * sr)))  # pre-stimulus baseline
        base = data[:, :b1].mean(axis=1, keepdims=True) if b1 > 1 else 0.0
        if i1 <= i0:
            return np.zeros(self._nspots)
        seg = data[:, i0:i1] - base
        charge_C = seg.sum(axis=1) / sr  # Coulombs
        return charge_C * 1e15  # -> femtocoulombs

    def _charge_brushes(self):
        thr = float(self.charge_params["threshold_fC"])
        alpha = int(round(255 * float(self.charge_params["spot_opacity"])))
        if self._charge is None:
            return [pg.mkBrush(128, 128, 128, min(alpha, 60))] * self._nspots
        return [pg.mkBrush(*charge_color(q, thr), alpha) for q in self._charge]

    def _recolor(self):
        """Recompute charge and recolor spots + traces (on parameter change)."""
        if self._mapdir is None:
            return
        self._charge = self._compute_charge()
        self._brushes = self._charge_brushes()
        self._place_spots(center_px=self._map_center)
        self._plot_traces()
        self._update_info()

    def _on_charge_params_changed(self, *args):
        self._recolor()

    def _place_spots(self, center_px=None):
        """(Re)compute and draw the spot overlay.  ``center_px`` (image pixel
        tuple) places the calibrated grid at a user-chosen point."""
        if self._img_shape is None:
            pix = np.column_stack([np.arange(self._nspots), np.zeros(self._nspots)])
            soma, self._pix_src = None, "no image"
        else:
            pix, soma, self._pix_src = _spot_pixels(
                self.reader, self._celldir, self._mapdir, self._img_shape, center_px=center_px
            )
        self.soma_item.setData([soma[0]] if soma else [], [soma[1]] if soma else [])
        # spot diameter in image pixels (~60% of the grid spacing) so spots read
        # like the map cells and scale with the image (pxMode=False)
        size_px = 14.0
        if self._img_shape is not None and self._nspots > 1:
            finite = pix[np.isfinite(pix).all(axis=1)]
            if len(finite) > 1:
                d = np.hypot(np.diff(finite[:, 0]), np.diff(finite[:, 1]))
                d = d[d > 0]
                if d.size:
                    size_px = 0.5 * float(np.min(d))
        size_px *= float(self.charge_params["spot_size_scale"])  # user size scaling
        self._spot_size = size_px
        spots = [
            {"pos": (pix[i, 0], pix[i, 1]), "data": i, "brush": self._brushes[i], "size": size_px}
            for i in range(self._nspots)
        ]
        self.scatter.setData(spots)

    def _plot_photodiode(self):
        self.pd_plot.clear()
        if hasattr(self.reader, "Photodiode"):
            pd = self.reader.Photodiode
            # use the acquirer's own time base (its rate/length can differ from ephys)
            tb = self.reader.Photodiode_time_base[0]
            for i in range(pd.shape[0]):
                self.pd_plot.plot(tb, pd[i], pen=pg.mkPen((120, 120, 120, 120)))
            self.pd_plot.enableAutoRange(axis="y")  # rescale to the new signal
        if self._laser and self._laser.get("start"):
            self.pd_plot.addItem(
                pg.InfiniteLine(
                    pos=self._laser["start"][0], angle=90, pen=pg.mkPen((220, 30, 30), width=2)
                )
            )

    def _update_info(self):
        laser, vc, amp = self._laser, self._vc, self._amp
        laser_txt = (
            f"laser (shutter): {laser['start'][0] * 1e3:.1f} ms"
            if (laser and laser.get("start"))
            else "laser: not found"
        )
        vc_txt = f" | VC test pulse: {vc['start'][0] * 1e3:.0f} ms" if vc.get("start") else ""
        # amplifier settings read from the file
        amp_txt = ""
        if amp:
            hold_disp = (
                f"{amp['holding'] * 1e3:.0f} mV"
                if amp["mode"] == "VC"
                else f"{amp['holding'] * 1e12:.0f} pA"
            )
            ph = amp.get("putative_holding")
            ph_txt = ""
            if ph is not None:
                ph_txt = (
                    f"<br>putative holding (acquirer/{'%g' % self.reader.defaults['acquirer_holding_gain']}): "
                    f"<b>{ph * 1e3:.0f} mV</b>"
                )
            amp_txt = (
                f"amp: {amp.get('amp_mode', '')} "
                f"gain in {amp['input_gain']:g} / out {amp['output_gain']:g}  "
                f"units {amp['input_units']}/{amp['output_units']}<br>"
                f"<i>defaults (not in file):</i> LPF {amp['LPF'] / 1e3:.1f} kHz, "
                f"holding {hold_disp}{ph_txt}<br>"
            )
        charge_txt = "charge: n/a"
        if self._charge is not None and len(self._charge):
            charge_txt = (
                f"charge (fC): min {np.nanmin(self._charge):.0f} / "
                f"max {np.nanmax(self._charge):.0f}  "
                f"[blue −, red +, |Q| over {self.charge_params['threshold_fC']:.0f} = white]"
            )
        self.info.setText(
            f"<b>{self._celldir.name}/{self._mapdir.name}</b><br>"
            f"mode={self.reader.mode}  sweeps={self._nspots}  "
            f"sr={self.reader.sample_rate[0]:.0f} Hz<br>"
            f"{amp_txt}"
            f"<span style='color:#c00'>{laser_txt}</span>{vc_txt}<br>"
            f"spot placement: {self._pix_src}<br>"
            f"{charge_txt}"
        )

    def _on_scene_click(self, ev):
        """When 'click to set map center' is checked, recenter the overlay grid
        at the clicked image point."""
        if not self.place_cb.isChecked() or self._img_shape is None or self._mapdir is None:
            return
        if not self.img_vb.sceneBoundingRect().contains(ev.scenePos()):
            return
        p = self.img_vb.mapSceneToView(ev.scenePos())
        self._map_center = (p.x(), p.y())
        self._place_spots(center_px=self._map_center)
        self._update_info()

    def _plot_traces(self, highlight=None):
        self.trace_plot.clear()
        self._trace_curves = []
        data = self.reader.data_array * 1e12  # pA
        tb = self.reader.time_base
        span = np.nanpercentile(data, 98) - np.nanpercentile(data, 2)
        offset = 1.3 * (span if span > 0 else 1.0)
        self._offsets = np.arange(self._nspots) * offset
        # red line at the ACTUAL photostimulus (laser/shutter) onset
        laser = self.reader.laser_pulse()
        if laser and laser.get("start"):
            self.trace_plot.addItem(
                pg.InfiniteLine(
                    pos=laser["start"][0], angle=90, pen=pg.mkPen((220, 30, 30), width=2)
                )
            )
        # each sweep is colored by its charge (same mapping as its spot);
        # the highlighted sweep is emphasized in yellow.
        thr = float(self.charge_params["threshold_fC"])
        tek_cyan = (60, 210, 235)
        for i in range(self._nspots):
            highlighted = highlight == i
            if highlighted:
                color = (255, 255, 0)
            elif self._charge is not None:
                color = charge_color(self._charge[i], thr)
            else:
                color = tek_cyan
            c = self.trace_plot.plot(
                tb,
                data[i] + self._offsets[i],
                pen=pg.mkPen(color, width=2 if highlighted else 1),
            )
            self._trace_curves.append(c)
        # reset the view to fit the current map's sweeps
        if self._nspots:
            ymin = float(np.nanmin(data[0])) + self._offsets[0]
            ymax = float(np.nanmax(data[-1])) + self._offsets[-1]
            pad = 0.05 * (ymax - ymin if ymax > ymin else 1.0)
            self.trace_plot.setYRange(ymin - pad, ymax + pad)
            self.trace_plot.setXRange(tb[0], tb[-1])

    def _on_spot_click(self, _scatter, points):
        if not len(points):
            return
        idx = points[0].data()
        self._plot_traces(highlight=idx)
        # emphasize the clicked spot (sizes are in image pixels, pxMode=False)
        base = getattr(self, "_spot_size", 14.0)
        sizes = [1.7 * base if p.data() == idx else base for p in self.scatter.points()]
        self.scatter.setSize(sizes)


def main():
    startdir = sys.argv[1] if len(sys.argv) > 1 else None
    pg.mkQApp("Ephus viewer")
    win = EphusViewer(startdir=startdir)
    win.show()
    pg.exec()


if __name__ == "__main__":
    main()
