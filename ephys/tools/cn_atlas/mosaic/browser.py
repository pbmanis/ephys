# Claude added 2026-10-22: PyQt6 window — mosaic file reader + 2D assembly view.
"""Mosaic browser window (Step 1 of the acq4 mosaic import workflow).

Left: a file reader — pick a day directory (``yyyy.mm.dd_nnn``), see every
``.mosaic`` under it (day / slice / cell levels), with a recent-days history so
the tool resumes where it left off. Right: a pyqtgraph view that assembles and
shows the selected mosaic (tiled grayscale image in physical/metre coordinates)
with its markers, cells and rulers overlaid.

This is the first step; the preliminary 3D alignment/ordering tool and the
export-to-atlas hand-off are added later. The currently assembled mosaic is kept
on ``self.current`` and announced via the ``on_mosaic_selected`` callback so the
main program can pick it up.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets

from .alignment import AlignmentController
from .history import History
from .reader import AssembledMosaic, MosaicReader, find_mosaic_files

# acq4 day directory: 2025.01.10_000
_DAYDIR_RE = re.compile(r"\d{4}\.\d{2}\.\d{2}_\d{3}$")


def _affine_to_qtransform(m: np.ndarray) -> QtGui.QTransform:
    """3x3 numpy affine (maps [x,y,1]->world) -> QTransform (row-vector form)."""
    return QtGui.QTransform(
        float(m[0, 0]), float(m[1, 0]),      # m11, m12
        float(m[0, 1]), float(m[1, 1]),      # m21, m22
        float(m[0, 2]), float(m[1, 2]))      # dx, dy


def _robust_levels(img: np.ndarray) -> tuple:
    """(lo, hi) display levels from the 1st/99th percentile (ignoring NaNs)."""
    finite = img[np.isfinite(img)]
    if finite.size == 0:
        return 0.0, 1.0
    lo, hi = np.percentile(finite, (1.0, 99.0))
    if hi <= lo:
        hi = lo + 1.0
    return float(lo), float(hi)


class MosaicBrowserWindow(QtWidgets.QMainWindow):
    """File reader + assembled-mosaic display."""

    def __init__(self, on_mosaic_selected=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Mosaic Importer — file reader")
        self.resize(1200, 820)

        self.reader = MosaicReader()
        self.history = History.load()
        self.on_mosaic_selected = on_mosaic_selected
        self.day_dir: Optional[Path] = None
        self.mosaics: list[Path] = []
        self.current: Optional[AssembledMosaic] = None
        self._overlay_items: list = []
        self._img_tiles: list = []

        # Step 2: preliminary 3D slice-stack alignment (view + controls)
        self.alignment = AlignmentController()
        self.alignment.get_current = lambda: self.current
        self.alignment.mosaic_source = self

        self._build_ui()
        self._refresh_recent()

    # -- UI -----------------------------------------------------------------
    def _build_ui(self):
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        hbox = QtWidgets.QHBoxLayout(central)

        left = QtWidgets.QWidget()
        lv = QtWidgets.QVBoxLayout(left)
        left_scroll = QtWidgets.QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setFixedWidth(360)
        left_scroll.setWidget(left)
        left_scroll.setHorizontalScrollBarPolicy(
            QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        lv.addWidget(QtWidgets.QLabel("<b>Day directory</b>  (yyyy.mm.dd_nnn)"))
        self.day_label = QtWidgets.QLabel("— none —")
        self.day_label.setWordWrap(True)
        self.day_label.setStyleSheet("QLabel{color:#9cf;}")
        lv.addWidget(self.day_label)
        self.browse_btn = QtWidgets.QPushButton("Browse for day directory…")
        self.browse_btn.clicked.connect(self._on_browse)
        lv.addWidget(self.browse_btn)

        lv.addWidget(QtWidgets.QLabel("<b>Recent days</b>"))
        self.recent_list = QtWidgets.QListWidget()
        self.recent_list.setMaximumHeight(130)
        self.recent_list.itemActivated.connect(self._on_recent_activated)
        lv.addWidget(self.recent_list)

        lv.addWidget(QtWidgets.QLabel("<b>Mosaic files</b>  (multi-select for Add selected)"))
        self.mosaic_list = QtWidgets.QListWidget()
        self.mosaic_list.setSelectionMode(
            QtWidgets.QAbstractItemView.SelectionMode.ExtendedSelection)
        self.mosaic_list.currentRowChanged.connect(self._on_mosaic_row)
        lv.addWidget(self.mosaic_list, stretch=1)

        # per-image include/exclude for the selected mosaic (default = the acq4
        # mosaic's "visible" flags; uncheck images that are mis-placed / wrong mag)
        img_box = QtWidgets.QGroupBox("Images in mosaic (checked = included)")
        iv = QtWidgets.QVBoxLayout(img_box)
        iv.setContentsMargins(4, 4, 4, 4)
        self.image_list = QtWidgets.QListWidget()
        self.image_list.setMaximumHeight(130)
        self.image_list.itemChanged.connect(self._on_image_item_changed)
        iv.addWidget(self.image_list)
        ibtns = QtWidgets.QHBoxLayout()
        for label, cb in (("All", lambda: self._set_all_images(True)),
                          ("None", lambda: self._set_all_images(False)),
                          ("Mosaic-visible", self._reset_images_to_visible)):
            b = QtWidgets.QPushButton(label)
            b.clicked.connect(cb)
            ibtns.addWidget(b)
        iv.addLayout(ibtns)
        self.image_warn = QtWidgets.QLabel("")
        self.image_warn.setWordWrap(True)
        self.image_warn.setStyleSheet("QLabel{color:#f84;}")
        iv.addWidget(self.image_warn)
        lv.addWidget(img_box)

        # preliminary 3D-alignment controls (Step 2)
        self.controls_box = QtWidgets.QGroupBox("Preliminary alignment (3D)")
        cb = QtWidgets.QVBoxLayout(self.controls_box)
        cb.setContentsMargins(4, 4, 4, 4)
        cb.addWidget(self.alignment.controls)
        lv.addWidget(self.controls_box)

        self.status = QtWidgets.QLabel("Pick a day directory to begin.")
        self.status.setWordWrap(True)
        lv.addWidget(self.status)

        hbox.addWidget(left_scroll)

        # right: tabbed display — 2D assembly + 3D preliminary alignment
        self.glw = pg.GraphicsLayoutWidget()
        self.vb = self.glw.addViewBox(lockAspect=True, invertY=True)
        self.vb.setMenuEnabled(True)
        self.title_label = pg.LabelItem(justify="left")
        self.glw.addItem(self.title_label, row=1, col=0)

        self.tabs = QtWidgets.QTabWidget()
        self.tabs.addTab(self.glw, "Assembly (2D)")
        self.tabs.addTab(self.alignment.view, "Alignment (3D)")
        hbox.addWidget(self.tabs, stretch=1)

    # -- day / mosaic selection --------------------------------------------
    @staticmethod
    def _coerce_day_dir(path: Path) -> Optional[Path]:
        """Walk up from ``path`` to the enclosing acq4 day directory, if any."""
        p = Path(path)
        for cand in [p, *p.parents]:
            if _DAYDIR_RE.search(cand.name):
                return cand
        return None

    def _on_browse(self):
        start = str(self.day_dir or Path.home())
        chosen = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Select an acq4 day directory", start)
        if chosen:
            self.load_day(Path(chosen))

    def _on_recent_activated(self, item: QtWidgets.QListWidgetItem):
        self.load_day(Path(item.data(QtCore.Qt.ItemDataRole.UserRole)))

    def load_day(self, path: Path, push_recent: bool = True):
        day = self._coerce_day_dir(path) or path
        if not day.is_dir():
            self.status.setText(f"Not a directory: {day}")
            return
        if self._coerce_day_dir(path) is None:
            self.status.setText(
                f"Warning: '{day.name}' is not a yyyy.mm.dd_nnn day dir; "
                "searching it anyway.")
        self.day_dir = day
        self.day_label.setText(str(day))
        self.mosaics = find_mosaic_files(day)
        self.mosaic_list.blockSignals(True)
        self.mosaic_list.clear()
        for mf in self.mosaics:
            # show the path relative to the day dir (slice/cell/name)
            try:
                rel = mf.relative_to(day)
            except ValueError:
                rel = mf.name
            self.mosaic_list.addItem(str(rel))
        self.mosaic_list.blockSignals(False)
        if push_recent:
            self.history.add_day(day)
            self.history.save()
            self._refresh_recent()
        if not self.mosaics:
            # Claude added 2026-10-22: no mosaics -> tell the user and clear the
            # display so a stale mosaic from a previous day is not left showing.
            self._clear_view()
            self.current = None
            self.title_label.setText("")
            self.status.setText(
                f"No .mosaic files found under {day.name}. "
                "Pick a different day directory.")
            return
        self.status.setText(
            f"{len(self.mosaics)} mosaic file(s) under {day.name}.")
        # auto-select the previously used mosaic, else the first
        prev = self.history.mosaic_for(day)
        target = 0
        if prev is not None:
            for i, mf in enumerate(self.mosaics):
                if str(mf) == prev:
                    target = i
                    break
        self.mosaic_list.setCurrentRow(target)

    def _refresh_recent(self):
        self.recent_list.clear()
        for d in self.history.recent_days:
            p = Path(d)
            it = QtWidgets.QListWidgetItem(p.name if p.name else d)
            it.setData(QtCore.Qt.ItemDataRole.UserRole, d)
            it.setToolTip(d)
            self.recent_list.addItem(it)

    def _on_mosaic_row(self, row: int):
        if row < 0 or row >= len(self.mosaics):
            return
        self._load_and_show(self.mosaics[row])

    # -- assembly + display -------------------------------------------------
    def _load_and_show(self, mosaic_path: Path):
        self.status.setText(f"Reading {mosaic_path.name} …")
        QtWidgets.QApplication.processEvents()
        try:
            asm = self.reader.read(mosaic_path)
        except Exception as exc:  # noqa: BLE001
            self.status.setText(f"Failed to read {mosaic_path.name}: {exc}")
            return
        self.current = asm
        if self.day_dir is not None:
            self.history.add_mosaic(self.day_dir, mosaic_path)
            self.history.save()
        self._populate_image_list(asm)
        self._render(asm)
        msg = (f"{mosaic_path.name}: {asm.n_included}/{asm.n_images} image(s) "
               f"included, {len(asm.cells)} cell(s), section = {asm.section_type or '—'}")
        if asm.missing:
            msg += f"  ({len(asm.missing)} tile(s) could not load)"
        self.status.setText(msg)
        if self.on_mosaic_selected is not None:
            self.on_mosaic_selected(asm)

    # -- per-image include/exclude -----------------------------------------
    def _populate_image_list(self, asm):
        """List the mosaic's image tiles with checkboxes (checked = included)."""
        self._img_tiles = [t for t in asm.tiles if t.image is not None]
        self.image_list.blockSignals(True)
        self.image_list.clear()
        for t in self._img_tiles:
            it = QtWidgets.QListWidgetItem(t.name)
            it.setFlags(it.flags() | QtCore.Qt.ItemFlag.ItemIsUserCheckable)
            it.setCheckState(QtCore.Qt.CheckState.Checked if t.include
                             else QtCore.Qt.CheckState.Unchecked)
            if not t.visible:
                it.setToolTip("not marked visible in the acq4 mosaic")
            self.image_list.addItem(it)
        self.image_list.blockSignals(False)
        self._update_image_warning(asm)

    def _update_image_warning(self, asm):
        if asm.n_images > 1 and asm.n_included <= 1:
            self.image_warn.setText(
                f"Only {asm.n_included} of {asm.n_images} images included. If the "
                "mosaic's visible images are not set up right, fix it in acq4's "
                "Mosaic Editor, or use 'All' here.")
        else:
            self.image_warn.setText("")

    def _apply_image_change(self):
        if self.current is None:
            return
        self._render(self.current)
        self._update_image_warning(self.current)
        self.alignment.invalidate_asm(self.current)

    def _on_image_item_changed(self, item: QtWidgets.QListWidgetItem):
        row = self.image_list.row(item)
        if 0 <= row < len(getattr(self, "_img_tiles", [])):
            self._img_tiles[row].include = (
                item.checkState() == QtCore.Qt.CheckState.Checked)
            self._apply_image_change()

    def _set_all_images(self, on: bool):
        for t in getattr(self, "_img_tiles", []):
            t.include = on
        if self.current is not None:
            self._populate_image_list(self.current)
            self._apply_image_change()

    def _reset_images_to_visible(self):
        for t in getattr(self, "_img_tiles", []):
            t.include = t.visible
        if self.current is not None:
            self._populate_image_list(self.current)
            self._apply_image_change()

    # -- bulk mosaic access for the alignment controller --------------------
    def _read_paths(self, paths) -> list:
        out = []
        for p in paths:
            try:
                out.append(self.reader.read(p))
            except Exception:  # noqa: BLE001 - skip unreadable mosaics
                continue
        return out

    def all_mosaic_asms(self) -> list:
        """Assemble every mosaic under the current day (for 'Add all')."""
        return self._read_paths(self.mosaics)

    def selected_mosaic_asms(self) -> list:
        """Assemble the mosaics multi-selected in the list (for 'Add selected')."""
        rows = sorted(i.row() for i in self.mosaic_list.selectedIndexes())
        return self._read_paths([self.mosaics[r] for r in rows
                                 if 0 <= r < len(self.mosaics)])

    def _clear_view(self):
        for it in self._overlay_items:
            self.vb.removeItem(it)
        self._overlay_items = []

    @staticmethod
    def _tile_area(tile) -> float:
        c = tile.corners_world()
        return (c[:, 0].max() - c[:, 0].min()) * (c[:, 1].max() - c[:, 1].min())

    def _render(self, asm: AssembledMosaic):
        self._clear_view()
        gray = np.repeat(np.arange(256, dtype=np.ubyte)[:, None], 3, axis=1)
        # Assemble ALL tiles (the acq4 ``visible`` flag is just the editor's last
        # UI state; the montage needs every tile). Draw larger overview tiles
        # behind smaller high-magnification detail tiles.
        tiles = list(asm.included_tiles())
        tiles.sort(key=lambda t: (t.z, -self._tile_area(t)))
        for tile in tiles:
            img = pg.ImageItem(tile.image, axisOrder="row-major")
            img.setLookupTable(gray)
            img.setLevels(_robust_levels(tile.image))
            img.setTransform(_affine_to_qtransform(tile.affine))
            img.setZValue(tile.z)
            img.setOpacity(float(tile.alpha))
            self.vb.addItem(img)
            self._overlay_items.append(img)
        self._render_overlays(asm)
        ext = asm.extent()
        if ext is not None:
            self.vb.setRange(QtCore.QRectF(ext[0], ext[2], ext[1] - ext[0],
                                           ext[3] - ext[2]), padding=0.08)
        self.title_label.setText(
            f"{asm.path.name}   |   section: {asm.section_type or '—'}   |   "
            f"{len(asm.tiles)} tiles")

    def _render_overlays(self, asm: AssembledMosaic):
        # anatomical markers (per section-type set)
        for ms in asm.marker_sets:
            spots = [{"pos": (m.pos[0], m.pos[1])} for m in ms.markers]
            if spots:
                sc = pg.ScatterPlotItem(
                    spots, size=10, symbol="o", pen=pg.mkPen((0, 255, 255), width=2),
                    brush=pg.mkBrush(0, 255, 255, 90))
                sc.setZValue(500)
                self.vb.addItem(sc)
                self._overlay_items.append(sc)
            for m in ms.markers:
                lbl = pg.TextItem(m.label, color=(0, 255, 255), anchor=(0, 1))
                lbl.setPos(m.pos[0], m.pos[1])
                lbl.setZValue(501)
                self.vb.addItem(lbl)
                self._overlay_items.append(lbl)
        # cells
        if asm.cells:
            spots = [{"pos": (c.pos[0], c.pos[1])} for c in asm.cells]
            sc = pg.ScatterPlotItem(
                spots, size=14, symbol="+", pen=pg.mkPen((255, 230, 0), width=2),
                brush=None)
            sc.setZValue(510)
            self.vb.addItem(sc)
            self._overlay_items.append(sc)
            for c in asm.cells:
                lbl = pg.TextItem(c.name, color=(255, 230, 0), anchor=(0, 0))
                lbl.setPos(c.pos[0], c.pos[1])
                lbl.setZValue(511)
                self.vb.addItem(lbl)
                self._overlay_items.append(lbl)
        # rulers
        for ruler in asm.rulers:
            if len(ruler.points) >= 2:
                pts = np.array([(p[0], p[1]) for p in ruler.points], float)
                ln = pg.PlotCurveItem(pts[:, 0], pts[:, 1],
                                      pen=pg.mkPen((255, 120, 200), width=1.5))
                ln.setZValue(505)
                self.vb.addItem(ln)
                self._overlay_items.append(ln)


def main(argv=None) -> int:
    app = pg.mkQApp("Mosaic Importer")
    win = MosaicBrowserWindow()
    win.show()
    app.exec()
    return 0
