"""
Fix the objective scale factor
Give "stated objective", "actual objective"

('objective', '4x 0.1na ACHROPLAN')

This is a tool for ephys. 

"""

import copy
import datetime
import html
import inspect
import os
import pprint
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Union

import ephys
import MetaArray as EM  # acq4 video (.ma) reader
import numpy as np
import tifffile  # acq4 image (.tif) reader
import pylibrary.tools.cprint as CP
import pyqtgraph as pg
import pyqtgraph.dockarea as PGD
from pylibrary.tools import fileselector as FS
from pylibrary.tools.utility import seqparse as SQP
from pyqtgraph import configfile
from pyqtgraph.parametertree import Parameter, ParameterTree
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets

# The experiment configuration (config/experiments.cfg) supplies the raw-data
# directory. The configuration is imported under an alias so it does not 
# collide with the local
# objective-config loader get_configuration() below.
from ephys.tools.get_configuration import get_configuration as get_experiment_configuration

pp = pprint.PrettyPrinter(indent=4)
# acq4 "day" directory names are stuctured as 2025.01.10_000. This re is used to
# shorten recent-file labels to the path tail starting at the day directory.
_DATEDIR_RE = re.compile(r"\d{4}\.\d{2}\.\d{2}_\d{3}")


def _ephys_config_dir() -> Path:
    """Return the packaged ephys/config directory (independent of cwd)."""
    return Path(inspect.getfile(ephys)).parent / "config"


# The objective-config loader is independent of the working directory.
def get_configuration(configfile_name: Union[str, Path, None] = None):
    candidates = []
    if configfile_name is not None:
        candidates.append(Path(configfile_name))
        candidates.append(Path(Path().absolute(), configfile_name))
    candidates.append(_ephys_config_dir() / "fix_objective_data.cfg")
    for cpath in candidates:
        try:
            if cpath is not None and Path(cpath).is_file():
                CP.cprint("g", f"fix_objscale: reading objective config from: {cpath!s}")
                cfg = configfile.readConfigFile(str(cpath))
                return cfg, Path(cpath)
        except Exception as exc:  # noqa: BLE001 - fall through to next candidate
            CP.cprint("y", f"fix_objscale: could not read {cpath!s}: {exc}")
    CP.cprint("y", "fix_objscale: objective config not found; using built-in defaults")
    return copy.deepcopy(objective_data), None


# default dict for objective data:
objective_data = {
    "Objectives": {
        # format name: "magnification na type"
        "4x 0.1na ACHROPLAN": 4.0,
        "5x 0.25na ACHROPLAN": 5.0,
        "10x 0.3na W N-ACHROPLAN": 10.0,
        "20x 0.5na W N-ACHROPLAN": 20.0,
        "40x 0.75na W ACHROPLAN": 40.0,
        "40x 0.8na ACHROPLAN": 40.0,
        "63x 0.9na ACHROPLAN": 63.0,
    },
    "CineScales": {
        1.0: 1.0,  # scale factor for cine data
        0.5: 0.5,
        0.3: 0.3,
    },
    # This is the coupler between the camera and microscope.
    # our couplers are 1x or 0.5x.
    "Cameras": {
        # information about the cameras - specifically, the pixel size
        "RetigaR1": {"pixelsize": 6.54e-6},
        "Electro": {"pixelsize": 6.45e-6},
        "Prism95B": {"pixelsize": 11.0e-6},
        "Retiga2000DC": {"pixelsize": 7.40e-6},
    },
    "LastFile": "/Volumes/Pegasus_002/ManisLab_Data3/Kasten_Michael/NF107Ai32_Het/Parasagittal/2020.07.07_000/slice_000/cell_000",
}


@dataclass
class Changer:
    change_type: str = "new"
    filename: str = ""
    from_objective: str = list(objective_data["Objectives"].keys())[0]
    to_objective: str = list(objective_data["Objectives"].keys())[4]
    videos: Union[str, None] = None
    images: Union[str, None] = None


class FixObjective(pg.QtWidgets.QWidget):
    def __init__(self, objective_data=None, objective_config_path=None,
                 datasets=None, experiments=None, app=None):
        super(FixObjective, self).__init__()
        self.objective_data = objective_data
        # save where the objective config was loaded from
        # so the recent-files list can be written back to it.
        self.objective_config_path = objective_config_path
        # Experiment configuration (config/experiments.cfg),
        # is used to derive the raw-data directory.
        self.datasets = list(datasets) if datasets else []
        self.experiments = experiments if experiments else {}
        self.app = app
        self.write = False
        self.objdata = Changer()
        self.last_file = None
        self.selected_camera = "RetigaR1"
        self.selected_cinescale = "1.0"
        self.ptreewid = 300  # params panel width
        self._recent_param = None  # set in build_ptree
        self._suppress_recent = False  # guard against reload on programmatic update
        self.status_path = None  # status-bar label for the protocol path (set in set_window)
        # image/video display state
        self._file_param = None  # the "File" dropdown (set in build_ptree)
        self._suppress_file = False  # guard against display on programmatic update
        self._cur_frames = None  # ndarray [nframes, H, W] for the current file
        self._cur_rect = None  # (width_um, height_um) of the current image
        self._cur_pxum = None  # pixel size in microns (for the FFT frequency axis)
        self.day_tree = None  # day-directory browser (set in set_window)
        self._current_file = None  # file currently shown (for .index highlight)
        # normalize LastFile to a 10-deep most-recent list and
        # resolve the data directory from the experiment configuration.
        self._normalize_recent()
        self.selected_dataset = self.datasets[0] if self.datasets else None
        self.datadir = self._resolve_datadir()
        self.update_camera_settings()

    # ------------------------------------------------------------------
    # recent-files (LastFile) management and data-dir
    # resolution. The recent list livesin fix_objective_data.cfg
    # ------------------------------------------------------------------
    def _normalize_recent(self):
        """Coerce LastFile into a list of at most 10 entries, newest first."""
        lf = self.objective_data.get("LastFile", [])
        if isinstance(lf, str):
            lf = [lf] if lf else []
        elif lf is None:
            lf = []
        else:
            lf = list(lf)
        self.objective_data["LastFile"] = lf[:10]

    def _resolve_datadir(self):
        """Pick the data directory: selected experiment dataset if its
        rawdatapath/directory exists, else the newest existing recent entry,
        else the user's home directory."""
        ds = self.selected_dataset
        if ds is not None and ds in self.experiments:
            expt = self.experiments[ds]
            try:
                d = Path(expt["rawdatapath"], expt["directory"])
                if d.is_dir():
                    return str(d)
            except Exception:  # noqa: BLE001 - missing keys / offline volume
                pass
        for r in self.objective_data.get("LastFile", []):
            rp = Path(r)
            if rp.exists():
                return str(rp if rp.is_dir() else rp.parent)
        return str(Path.home())

    def _save_objective_config(self):
        """Persist the recent-files list back to the config file.

        Only the single ``LastFile:`` line is rewritten; the
        rest of the file (comments, objectives, cameras, cinescales, and any
        hand edits) is preserved. This avoids comment loss and
        churn of pyqtgraph's whole-file writeConfigFile.
        """
        if self.objective_config_path is None:
            CP.cprint("y", "fix_objscale: no config path known; recent list not saved")
            return
        path = Path(self.objective_config_path)
        recents = self.objective_data.get("LastFile", [])
        new_line = f"LastFile: {recents!r}\n"
        try:
            lines = path.read_text().splitlines(keepends=True) if path.is_file() else []
            out, replaced = [], False
            for ln in lines:
                if not replaced and re.match(r"^\s*LastFile\s*:", ln):
                    out.append(new_line)
                    replaced = True
                else:
                    out.append(ln)
            if not replaced:
                if out and not out[-1].endswith("\n"):
                    out[-1] += "\n"
                out.append(new_line)
            path.write_text("".join(out))
        except Exception as exc:  # noqa: BLE001
            CP.cprint("r", f"fix_objscale: failed to save recent list: {exc}")

    def _push_recent(self, path):
        """Push a protocol directory onto the most-recent stack (newest first,
        de-duplicated, capped at 10) and persist it."""
        path = str(path)
        lst = self.objective_data.get("LastFile", [])
        if isinstance(lst, str):
            lst = [lst] if lst else []
        lst = [p for p in lst if p != path]
        lst.insert(0, path)
        self.objective_data["LastFile"] = lst[:10]
        self._update_recent_dropdown()
        self._save_objective_config()

    # shorten the recent-file labels to keep the left panel sane.
    def _recent_label(self, path):
        parts = Path(path).parts
        for i, part in enumerate(parts):
            if _DATEDIR_RE.fullmatch(part):
                return str(Path(*parts[i:]))
        return str(Path(*parts[-3:])) if len(parts) >= 3 else str(path)

    def _recent_items(self):
        """Ordered {display_label: full_path} for the recent stack (unique labels)."""
        items = {}
        for p in self.objective_data.get("LastFile", []):
            label = self._recent_label(p)
            base, n = label, 2
            while label in items:  # disambiguate collisions
                label, n = f"{base} ({n})", n + 1
            items[label] = p
        return items

    def _update_recent_dropdown(self):
        if self._recent_param is None:
            return
        items = self._recent_items()
        limits = items if items else {"(none)": "(none)"}
        self._suppress_recent = True
        try:
            self._recent_param.setLimits(limits)
            self._recent_param.setValue(list(limits.values())[0])
        finally:
            self._suppress_recent = False

    def update_camera_settings(self):
        self.camera = self.objective_data["Cameras"][self.selected_camera]
        self.cinescale = self.objective_data["CineScales"][self.selected_cinescale]
        self.pixelsize = self.objective_data["Cameras"][self.selected_camera]["pixelsize"]
        self.refscale = [
            (1.0 / self.cinescale) * self.pixelsize,
            -(1.0 / self.cinescale) * self.pixelsize,
        ]

    # Image / video display with a scale grid.
    # and a 2-D spectral amplitude spectrum. 
    # ------------------------------------------------------------------
    def _build_image_dock(self):
        pg.setConfigOptions(imageAxisOrder="row-major")  # arrays are [row=y, col=x]
        container = QtWidgets.QWidget()
        vlayout = QtWidgets.QVBoxLayout()
        vlayout.setContentsMargins(0, 0, 0, 0)
        container.setLayout(vlayout)

        # objective / binning / scale for the displayed file,
        # shown above the image.
        self.obj_label = QtWidgets.QLabel("(no image)")
        self.obj_label.setWordWrap(True)
        self.obj_label.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
        vlayout.addWidget(self.obj_label, stretch=0)

        self.image_glw = pg.GraphicsLayoutWidget()
        self.image_plot = self.image_glw.addPlot(row=0, col=0)
        self.image_plot.setTitle("Image")
        self.image_plot.setAspectLocked(True)
        self.image_plot.invertY(True)  # row 0 of the image at the top
        self.image_plot.setLabel("bottom", "X", units="µm")
        self.image_plot.setLabel("left", "Y", units="µm")
        self.image_plot.showGrid(x=True, y=True, alpha=0.6)
        self.image_item = pg.ImageItem()
        self.image_plot.addItem(self.image_item)

        # intensity (min/max level) control to the right of the
        # image - a histogram with draggable min & max level handles.
        self.hist = pg.HistogramLUTItem()
        self.hist.setImageItem(self.image_item)
        self.image_glw.addItem(self.hist, row=0, col=1)

        self.fft_plot = self.image_glw.addPlot(row=0, col=2)
        self.fft_plot.setTitle("2-D amplitude spectrum (log)")
        self.fft_plot.setAspectLocked(True)
        self.fft_plot.setLabel("bottom", "fx", units="1/µm")
        self.fft_plot.setLabel("left", "fy", units="1/µm")
        self.fft_item = pg.ImageItem()
        self.fft_plot.addItem(self.fft_item)

        vlayout.addWidget(self.image_glw, stretch=1)

        # frame slider (enabled only for videos)
        hbox = QtWidgets.QHBoxLayout()
        self.frame_label = QtWidgets.QLabel("Frame: -")
        self.frame_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.frame_slider.setMinimum(0)
        self.frame_slider.setMaximum(0)
        self.frame_slider.setEnabled(False)
        self.frame_slider.valueChanged.connect(self._on_frame_changed)
        hbox.addWidget(self.frame_label)
        hbox.addWidget(self.frame_slider, stretch=1)
        sliderw = QtWidgets.QWidget()
        sliderw.setLayout(hbox)
        vlayout.addWidget(sliderw, stretch=0)

        self.Dock_Image.addWidget(container)

    def _populate_file_dropdown(self, select_name=None):
        """List image_*.tif / video_*.ma in the current protocol directory and
        display one (``select_name`` if present, else the first)."""
        if self._file_param is None:
            return
        d = Path(self.objdata.filename)
        files = []
        if d.is_dir():
            files = sorted(
                p.name
                for p in d.iterdir()
                if p.name.startswith(("image_", "video_"))
                and p.suffix.lower() in (".tif", ".tiff", ".ma")
            )
        vals = files if files else ["(none)"]
        chosen = select_name if (select_name in vals) else (vals[0] if files else "(none)")
        self._suppress_file = True
        try:
            self._file_param.setLimits(vals)
            self._file_param.setValue(chosen)
        finally:
            self._suppress_file = False
        if files:
            self._display_file(chosen)

    def _load_image_data(self, path: Path):
        """Return (frames[nframes, H, W] as float, total_frame_count). Videos are
        truncated to the first 100 frames."""
        suf = path.suffix.lower()
        if suf in (".tif", ".tiff"):
            arr = np.asarray(tifffile.imread(str(path)))
        elif suf == ".ma":
            arr = np.asarray(EM.MetaArray(file=str(path)))
        else:
            raise ValueError(f"unsupported file type: {suf}")
        arr = np.squeeze(arr)
        if arr.ndim == 2:
            return arr.astype(float)[np.newaxis, :, :], 1
        if arr.ndim == 3:
            if arr.shape[-1] in (3, 4) and arr.shape[0] > 4:  # single RGB(A) image
                return arr.mean(axis=-1).astype(float)[np.newaxis, :, :], 1
            total = int(arr.shape[0])  # acq4 stores the frame ('Time') axis first
            frames = arr[:100] if total > 100 else arr
            return frames.astype(float), total
        raise ValueError(f"unexpected image dimensionality: {arr.ndim}")

    def _compute_scale_um(self, name: str, W: int, H: int):
        """Physical extent (width_um, height_um) and x pixel size (um) from the
        .index transform scale (meters/pixel). Falls back to 1 px == 1 unit."""
        try:
            scale = self.index[name]["transform"]["scale"]
            sx, sy = abs(float(scale[0])), abs(float(scale[1]))
            w_um, h_um, px_um = sx * 1e6 * W, sy * 1e6 * H, sx * 1e6
            if w_um > 0 and h_um > 0:
                return w_um, h_um, px_um
        except Exception:  # noqa: BLE001 - missing index entry / unusable scale
            pass
        CP.cprint("y", f"fix_objscale: no usable scale for {name}; showing pixels")
        return float(W), float(H), 1.0

    @staticmethod
    def _nice_step(span: float, target_divs: int = 10) -> float:
        """A 1-2-5-rounded grid step closest to span/target_divs (so e.g. a ~1 mm
        image yields a 100 um grid -> about a 10x10 grid)."""
        if span <= 0:
            return 1.0
        raw = span / float(target_divs)
        mag = 10.0 ** np.floor(np.log10(raw))
        candidates = [m * mag for m in (1.0, 2.0, 5.0, 10.0)]
        return float(min(candidates, key=lambda c: abs(c - raw)))

    def _display_file(self, name: str):
        path = Path(self.objdata.filename, name)
        try:
            frames, total = self._load_image_data(path)
        except Exception as exc:  # noqa: BLE001
            CP.cprint("r", f"fix_objscale: could not load {path!s}: {exc}")
            return
        nframes = frames.shape[0]
        if total > nframes:  # video was truncated
            QtWidgets.QMessageBox.information(
                self.win,
                "Large video truncated",
                f"'{name}' contains {total} frames.\n"
                f"Only the first {nframes} will be loaded and displayed.",
            )
        H, W = int(frames.shape[1]), int(frames.shape[2])
        w_um, h_um, px_um = self._compute_scale_um(name, W, H)
        self._cur_frames = frames
        self._cur_rect = (w_um, h_um)
        self._cur_pxum = px_um
        is_video = nframes > 1
        self.frame_slider.blockSignals(True)
        self.frame_slider.setMaximum(max(0, nframes - 1))
        self.frame_slider.setValue(0)
        self.frame_slider.setEnabled(is_video)
        self.frame_slider.blockSignals(False)
        self.image_plot.setTitle(name)
        # Remember the current file, update the objective
        # banner above the image, auto-scale intensity for the new file, and
        # refresh the .index table so the row is highlighted.
        self._current_file = name
        self._update_objective_label(name)
        self._show_frame(0, auto_levels=True)
        self.show_index()

    def _update_objective_label(self, name: str):
        entry = {}
        try:
            entry = self.index.get(name, {})
        except Exception:  # noqa: BLE001 - index not loaded
            entry = {}
        objn = entry.get("objective", "?")
        binn = entry.get("binning", "?")
        scale = entry.get("transform", {}).get("scale", "?") if isinstance(entry, dict) else "?"
        self.obj_label.setText(
            f"<b>{html.escape(str(name))}</b> &nbsp;|&nbsp; Objective: "
            f"<b>{html.escape(str(objn))}</b> &nbsp;|&nbsp; binning: {html.escape(str(binn))} "
            f"&nbsp;|&nbsp; scale (m/px): {html.escape(str(scale))}"
        )

    def _on_frame_changed(self, value: int):
        # keep the user's intensity levels while scrubbing
        self._show_frame(value, auto_levels=False)

    def _show_frame(self, idx: int, auto_levels: bool = True):
        if self._cur_frames is None:
            return
        idx = int(np.clip(idx, 0, self._cur_frames.shape[0] - 1))
        frame = self._cur_frames[idx]
        w_um, h_um = self._cur_rect
        # use mm once the field of view reaches 1 mm, and
        # microns only below 1 mm. This also avoids pyqtgraph's SI auto-prefix
        # turning 1000 µm into "1 kµm".
        if max(w_um, h_um) >= 1000.0:
            unit, uf = "mm", 1.0e-3  # microns -> millimeters
        else:
            unit, uf = "µm", 1.0
        w_disp, h_disp = w_um * uf, h_um * uf

        # image (row-major: array[y, x]); invertY keeps row 0 at the top.
        # auto_levels only on a new file; preserve the user's min/max while
        # scrubbing video frames.
        self.image_item.setImage(frame, autoLevels=auto_levels)
        self.image_item.setRect(QtCore.QRectF(0.0, 0.0, w_disp, h_disp))
        step = self._nice_step(max(w_disp, h_disp), target_divs=10)
        for axname, label in (("bottom", "X"), ("left", "Y")):
            ax = self.image_plot.getAxis(axname)
            self.image_plot.setLabel(axname, label, units=unit)
            ax.enableAutoSIPrefix(False)  # show the unit verbatim (mm or µm)
            ax.setTickSpacing(step, step / 5.0)
        self.image_plot.setXRange(0.0, w_disp, padding=0.02)
        self.image_plot.setYRange(0.0, h_disp, padding=0.02)

        # 2-D amplitude spectrum (log), centered at zero frequency; frequency in
        # cycles per the same length unit as the image.
        spec = np.log1p(np.abs(np.fft.fftshift(np.fft.fft2(frame))))
        self.fft_item.setImage(spec, autoLevels=True)
        px_disp = (self._cur_pxum or 1.0) * uf  # pixel size in display units
        f_nyq = 1.0 / (2.0 * px_disp)  # Nyquist (cycles per display unit)
        self.fft_item.setRect(QtCore.QRectF(-f_nyq, -f_nyq, 2.0 * f_nyq, 2.0 * f_nyq))
        for axname, label in (("bottom", "fx"), ("left", "fy")):
            ax = self.fft_plot.getAxis(axname)
            self.fft_plot.setLabel(axname, label, units="1/" + unit)
            ax.enableAutoSIPrefix(False)
        self.fft_plot.setXRange(-f_nyq, f_nyq, padding=0.02)
        self.fft_plot.setYRange(-f_nyq, f_nyq, padding=0.02)

        n = self._cur_frames.shape[0]
        self.frame_label.setText(f"Frame: {idx}" + (f" / {n - 1}" if n > 1 else ""))

    # ------------------------------------------------------------------
    # collapsible day-directory browser (below the Quit
    # button). Shows the whole directory tree for the current acq4 "day", pruned
    # to the branches that contain images/videos, so files are easy to pick.
    # ------------------------------------------------------------------
    def _day_dir(self, path):
        """Return the acq4 day directory (…/YYYY.MM.DD_NNN) containing ``path``."""
        parts = Path(path).parts
        for i, part in enumerate(parts):
            if _DATEDIR_RE.fullmatch(part):
                return str(Path(*parts[: i + 1]))
        return None

    def _populate_day_tree(self):
        if self.day_tree is None:
            return
        self.day_tree.clear()
        day = self._day_dir(self.objdata.filename)
        if day is None or not Path(day).is_dir():
            self.day_tree.setHeaderLabels(["Day directory (none)"])
            return
        self.day_tree.setHeaderLabels([Path(day).name])
        root = QtWidgets.QTreeWidgetItem([Path(day).name])
        root.setData(0, QtCore.Qt.ItemDataRole.UserRole, {"type": "dir", "path": str(day)})
        self.day_tree.addTopLevelItem(root)
        self._add_dir_children(root, Path(day), depth=0)
        root.setExpanded(True)

    def _add_dir_children(self, parent_item, dirpath: Path, depth: int = 0) -> bool:
        """Add image/video files and media-bearing subdirectories under
        ``dirpath``. Returns True if anything was added (so empty branches are
        pruned). Subdirectories start collapsed."""
        if depth > 6:
            return False
        try:
            entries = sorted(dirpath.iterdir(), key=lambda p: (not p.is_dir(), p.name))
        except (OSError, PermissionError):
            return False
        added = False
        role = QtCore.Qt.ItemDataRole.UserRole
        for e in entries:
            if e.is_dir():
                if e.name.startswith("."):
                    continue
                item = QtWidgets.QTreeWidgetItem([e.name])
                item.setData(0, role, {"type": "dir", "path": str(e)})
                if self._add_dir_children(item, e, depth + 1):
                    parent_item.addChild(item)
                    item.setExpanded(False)  # closed by default (> toggle)
                    added = True
            elif e.name.startswith(("image_", "video_")) and e.suffix.lower() in (
                ".tif",
                ".tiff",
                ".ma",
            ):
                item = QtWidgets.QTreeWidgetItem([e.name])
                item.setData(0, role, {"type": "file", "path": str(e)})
                parent_item.addChild(item)
                added = True
        return added

    def _on_day_tree_item(self, item, _col):
        """Clicking an image/video in the day tree navigates to its protocol
        directory and displays it (without pushing onto the recent stack)."""
        data = item.data(0, QtCore.Qt.ItemDataRole.UserRole)
        if not data or data.get("type") != "file":
            return
        fpath = Path(data["path"])
        protodir = fpath.parent
        self.objdata.filename = str(protodir)
        try:
            self.read_index(objective=self.objdata, write=False)
        except (FileNotFoundError, KeyError) as exc:
            CP.cprint("r", f"fix_objscale: could not read .index in {protodir!s}: {exc}")
            return
        self._set_status(str(protodir))
        self.show_index()
        self._populate_file_dropdown(select_name=fpath.name)

    def getProtocolDir(self, reload_last=False, recent_path=None):
        # Three entry points now - an explicit recent_path
        # (from the Recent dropdown), reload_last (newest on the stack), or a
        # directory chooser. On a successful .index read the directory is pushed
        # onto the most-recent stack.
        current_filename = None
        if recent_path is not None:
            current_filename = recent_path
        elif reload_last:
            recents = self.objective_data.get("LastFile", [])
            if len(recents) == 0:
                CP.cprint("y", "fix_objscale: no previous files found")
                return
            current_filename = recents[0]  # newest on top
        else:
            sel = FS.FileSelector(dialogtype="dir", startingdir=str(self.datadir))
            current_filename = sel.fileName
        if current_filename is None:
            return
        print("Current Filename: ", current_filename)

        self.objdata.filename = current_filename
        try:
            self.read_index(objective=self.objdata, write=False)
        except (FileNotFoundError, KeyError) as exc:
            CP.cprint("r", f"fix_objscale: could not read .index in {current_filename!s}: {exc}")
            return
        self._push_recent(current_filename)
        self._set_status(current_filename)
        self.show_index()
        self._populate_file_dropdown()
        self._populate_day_tree()

    def rewrite_index(self, index: dict, index_file: Union[str, Path]):
        configfile.writeConfigFile(index, index_file)

    def read_indexes(self, changeList: list = [], write: bool = False):
        """
        To change a list of images/videos
        """
        print("write flag: ", write)
        for objective in changeList:
            self.read_index(objective, write=write)

    def read_index(self, objective: object = None, write: bool = False):
        print("\nfix_objscale: We will be using the following reference scale: ", self.refscale)
        print("   This scale may be specific to your camera!!!!!")
        print("read_index write flag is: ", write)
        print("Objective: ")
        print(f"    Change Type: {objective.change_type!s}")
        print(f"    File: {objective.filename!s}")
        print(f"    From: {objective.from_objective!s}X to {objective.to_objective!s}X")
        print(f"    For images: {str(objective.images)!s}")
        print(f"    For videos: {str(objective.videos)!s}")
        print("=" * 40)
        self.index_file = Path(objective.filename, ".index")
        self.index = configfile.readConfigFile(self.index_file)

    # def show_index(self):
    #     """show_index display the relevant data from the index on the right side.

    #     """

    #     self.textbox.setCurrentFont(QtGui.QFont("Courier New"))
    #     text = []  # ["<font color='red' size='3' font-family='monospace'>Hello PyQt5!\nHello"]
    #     # ["<font-family:'Courier New' color='blue' size='10'>"]
    #     # imagelist = [""]
    #     # videolist = [""]

    #     # I'm sure there is a better way to do this, but parametrees are poorly documented
    #     # and few practical examples
    #     # u = self.ptreedata.children()
    #     # ulist = {}  # hold the parameter objects for the images and videos
    #     # for uu in u:
    #     #     if uu.name() in ["Images", "Videos"]:
    #     #         uu.setValue("")
    #     #         uu.setLimits([])
    #     #         ulist[uu.name()] = uu

    #     for k in self.index.keys():
    #         if k.startswith("image"):
    #             text.append(
    #                 f" {k:16s}: {self.index[k]['objective']:24s}, {str(self.index[k]['binning']):8s}"
    #                 + f" {str(self.index[k]['transform']['scale']):s}<br>"
    #             )
    #             imagelist.append(k)
    #         if k.startswith("video"):
    #             text.append(
    #                 f" {k:16s}: {self.index[k]['objective']:24s}, {str(self.index[k]['binning']):8s}"
    #                 + f" {str(self.index[k]['transform']['scale']):s}<br>"
    #             )
    #             videolist.append(k)

    #     self.textbox.setHtml("\n".join([t for t in text]))
    #     # populate the drop-down lists in the parametertree
    #     ulist["Images"].setValue(imagelist[0])
    #     ulist["Images"].setLimits(imagelist)
    #     ulist["Videos"].setValue(videolist[0])
    #     ulist["Videos"].setLimits(videolist)

    def show_index(self):
        # Remember the current file, update the objective
        col = {"file": 20, "objective": 28, "binning": 12}
        header = (
            f"{'file':<{col['file']}s}{'objective':<{col['objective']}s}"
            f"{'binning':<{col['binning']}s}scale (m/px)"
        )
        rows = [html.escape(header)]
        for k in self.index.keys():
            if not (k.startswith("image") or k.startswith("video")):
                continue
            entry = self.index[k]
            objn = str(entry.get("objective", "") if isinstance(entry, dict) else "")
            binn = str(entry.get("binning", "") if isinstance(entry, dict) else "")
            try:
                scale = str(entry.get("transform", {}).get("scale", ""))
            except Exception:  # noqa: BLE001
                scale = ""
            line = (
                f"{k:<{col['file']}s}{objn:<{col['objective']}s}"
                f"{binn:<{col['binning']}s}{scale}"
            )
            line = html.escape(line.ljust(96))  # pad so the highlight spans the row
            if self._current_file is not None and k == self._current_file:
                line = (
                    "<span style=\"background-color:#ffe680; color:#000;\">"
                    f"{line}</span>"
                )
            rows.append(line)
        body = "\n".join(rows)
        self.textbox.setHtml(
            "<pre style=\"font-family:'Courier New',Menlo,monospace; font-size:10pt; "
            f"line-height:1.1;\">{body}</pre>"
        )

    def set_new_objective(self, data):
        self.objdata.to_objective = data

    def update_from_objective(self):
        fns = self.get_imagefilenames(self.objdata)  # get for the current
        print("image file names: ", fns)
        # print('self.index: ', self.index)
        for fn in fns:
            objname = self.index[fn]["objective"]
            if objname not in list(self.objective_data["Objectives"].keys()):
                print(f"Objective {objname:s} not found in list of known objectives.")
                continue
            print("objective(s) found: ", objname)
            self.ptreedata.param("Original Objective").setValue(objname)

    def get_original_objective(self):
        pass

    def view_proposed_changes(self, write=False):
        print("Proposed changes - new objective data: ", self.objdata)
        self.change_scale(self.objdata, write=write)

    def get_imagefilenames(self, objective: object) -> str:
        imagefiles = []
        print("objective img: ", objective.images)
        if objective.images is not None and len(objective.images) > 0:
            imagefiles.append(objective.images)
            images, target = SQP(objective.images)
            print("images: ", images)
            for img in images[0]:
                print("img: ", img)
                imagefiles.append(f"image_{int(img[0]):03d}.tif")
        print("objective.videos: ", objective.videos)
        if objective.videos is not None:
            videos, target = SQP(objective.videos)
            print('Videls: ', videos)
            for vid in videos[0]:
                imagefiles.append(f"video_{int(vid):03d}.ma")
        else:
            pass
        return imagefiles

    def change_scale(self, objective: object = None, write: bool = False):
        print("change scale objective: ", objective)
        imagefiles = self.get_imagefilenames(objective)
        if len(imagefiles) == 0:
            return
        text = []
        text.append(f"Changescale: imagefiles: {imagefiles}")

        text.append(f"\nindex keys:  {self.index.keys()}")
        for imagefile in imagefiles:
            if imagefile not in self.index.keys():
                text.append(f"File {imagefile:s} not found in {str(list(self.index.keys())):s}")
                continue
            text.append(f"Index imagefile:  {imagefile}")
            text.append(f"Old objective: {self.objdata.from_objective}")  # self.index[imagefile]["objective"]
            # pp.pprint(index[imagefile] )
            old_objective = self.index[imagefile]["objective"]
            pp.pprint("   Old transform: ")
            pp.pprint(self.index[imagefile]["transform"])
            pp.pprint("   Old device transform,: ")
            pp.pprint(self.index[imagefile]["deviceTransform"])
            binning = self.index[imagefile]["binning"]
            new_objective = objective.to_objective  # string name
            magnification_new_objective = float(
                self.objective_data["Objectives"][new_objective]
            )  # new magnification
            self.index[imagefile]["transform"]["scale"] = (
                binning[0] * self.refscale[0] / magnification_new_objective,
                binning[1] * self.refscale[1] / magnification_new_objective,
                1.0,
            )
            self.index[imagefile]["deviceTransform"]["scale"] = (
                binning[0] * self.refscale[0] / magnification_new_objective,
                binning[1] * self.refscale[1] / magnification_new_objective,
                1.0,
            )
            d = datetime.datetime.now()
            dstr = d.strftime("%Y-%m-%d %H:%M:%S")
            self.index[imagefile]["objective"] = objective.to_objective
            self.index[imagefile]["note"] = f"Objective scale corrected from {old_objective:s}"
            self.index[imagefile]["note"] += f" to {new_objective:s} on {dstr:s} by PBM"
            print(
                "New objective: ", self.index[imagefile]["objective"]
            )  # pp.pprint(index[imagfile] )
            print("   New transform: ")
            pp.pprint(self.index[imagefile]["transform"])
            print("   New device transform: ")
            pp.pprint(self.index[imagefile]["deviceTransform"])
            print("   Added Note: ", self.index[imagefile]["note"])
            print("----------------------------")

            print("read_index write: ", write)
            index_filename = Path(objective.filename, ".index")
            if write:
                self.rewrite_index(self.index, index_filename)
                CP.cprint("g", ".index file has been updated")

            else:
                print("Dry Run: .index file was NOT modified")
                print(f" file to modify is: {str(index_filename):s}")
        # then update index file display
        text.append(f"\n\nChanges made to {str(imagefiles):s}")
        self.change_text.setHtml("\n".join([t for t in text]))

        self.show_index()

    def build_ptree(self):
        cameras = list(self.objective_data["Cameras"].keys())
        objectives = list(self.objective_data["Objectives"].keys())
        cinescales = list(self.objective_data["CineScales"].keys())
        # dataset chooser (from experiments.cfg) and a
        # most-recent-protocols dropdown (from fix_objective_data.cfg).
        datasets = self.datasets if self.datasets else ["(none)"]
        # Recent items use {short_label: full_path} so the dropdown
        # shows the day-dir tail while still carrying the full path as its value.
        recent_items = self._recent_items()
        recent_limits = recent_items if recent_items else {"(none)": "(none)"}
        recent_value = list(recent_limits.values())[0]
        self.params = [
            # {"name": "Pick Cell", "type": "list", "values": cellvalues, "value": cellvalues[0]},
            {
                "name": "Dataset",
                "type": "list",
                "limits": datasets,
                "value": datasets[0],
                "renamable": False,
            },
            {"name": "Set Directory/Protocol", "type": "action"},
            {"name": "Reload Last Protocol", "type": "action"},
            {
                "name": "Recent",
                "type": "list",
                "limits": recent_limits,
                "value": recent_value,
                "renamable": False,
            },
            # Three entry points now - an explicit recent_path
            {
                "name": "File",
                "type": "list",
                "limits": ["(none)"],
                "value": "(none)",
                "renamable": False,
            },
            {"name": "Images", "type": "str", "value": ""},
            {"name": "Videos", "type": "str", "value": ""},
            {
                "name": "Camera",
                "type": "list",
                "limits": cameras,
                "value": cameras[0],
                "renamable": False,
            },
            {
                "name": "Cine Scale",
                "type": "list",
                "limits": cinescales,
                "value": cinescales[0],
                "readonly": True,
            },
            {
                "name": "Original Objective",
                "type": "list",
                "limits": objectives,
                "value": objectives[0],
                "renamable": False,
            },
            {
                "name": "New Objective",
                "type": "list",
                "limits": objectives,
                "value": objectives[1],
                "renamable": False,
            },
            {"name": "View .index", "type": "action"},
            {"name": "View proposed changes", "type": "action"},
            {"name": "Apply changes", "type": "action"},
            {'name': "Reload", 'type': 'action'},
            {"name": "Quit", "type": "action"},
        ]
        self.ptree = ParameterTree()
        self.ptreedata = Parameter.create(name="Commands", type="group", children=self.params)

        self.ptree.setParameters(self.ptreedata)
        # Keep a handle to the Recent dropdown so it can be
        # refreshed as the most-recent stack changes.
        self._recent_param = self.ptreedata.param("Recent")
        self._file_param = self.ptreedata.param("File")
        self.ptree.setMinimumWidth(self.ptreewid)
        self.ptree.setColumnWidth(0, 150)

    def command_dispatcher(self, param, changes):
        """
        Dispatcher for the commands from parametertree
        path[0] will be the command name
        path[1] will be the parameter (if there is one)
        path[2] will have the subcommand, if there is one
        data will be the field data (if there is any)
        """
        for param, change, data in changes:
            path = self.ptreedata.childPath(param)

            if path[0] == "Quit":
                self.quit()
            elif path[0] == "View .index":
                self.show_index()
            elif path[0] == "Dataset":
                # switch experiment dataset -> data directory
                self.selected_dataset = data
                self.datadir = self._resolve_datadir()
                CP.cprint("c", f"fix_objscale: data directory set to {self.datadir!s}")
            elif path[0] == "Recent":
                # load a directory chosen from the recent stack
                if not self._suppress_recent and data not in (None, "(none)"):
                    self.getProtocolDir(recent_path=data)
            elif path[0] == "File":
                if not self._suppress_file and data not in (None, "(none)"):
                    self._display_file(data)
            elif path[0] == "Set Directory/Protocol":
                self.getProtocolDir()
            elif path[0] == "Reload Last Protocol":
                self.getProtocolDir(reload_last=True)
            elif path[0] == "Camera":
                self.selected_camera = data
                self.update_camera_settings()  # when changing cameras, make sure pixel size is updated
            elif path[0] == "Cine Scale":
                self.selected_cinescale = data
                self.update_camera_settings()  # when changing the cinescale, also make sure pixel size is updated
            elif path[0] == "Videos":
                self.objdata.videos = data
            elif path[0] == "Images":
                self.objdata.images = data
            elif path[0] == "Original Objective":
                self.objdata.from_objective = data
            elif path[0] == "New Objective":
                self.objdata.to_objective = data
                self.update_from_objective()
            elif path[0] == "View proposed changes":
                self.view_proposed_changes(write=False)
            elif path[0] == "Apply changes":
                self.view_proposed_changes(write=True)

            elif path[0] == "Quit":
                self.quit()

    def set_window(self, parent=None):
        super(FixObjective, self).__init__(parent=parent)
        self.win = pg.QtWidgets.QMainWindow()
        self.win.setWindowTitle("FixObjective")
        self.DockArea = PGD.DockArea()
        self.win.setCentralWidget(self.DockArea)
        self.win.resize(1600, 1024)
        self.fullscreen_widget = None
        self.win.setWindowTitle("Model DataTables/FileSelector")
        win_wid = 1024
        win_ht = 512

        self.win.setWindowTitle("No File")

        # self.buttons = pg.QtGui.QGridLayout()
        self.build_ptree()
        # self.buttons.addWidget(self.ptree)

        # Initial Dock Arrangment

        self.Dock_Params = PGD.Dock("Params", size=(self.ptreewid, win_ht))
        # the Params dock stacks the command tree on top and a
        # collapsible day-directory browser below the Quit button (in a splitter
        # so the divide is draggable).
        params_split = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical)
        params_split.addWidget(self.ptree)
        self.day_tree = QtWidgets.QTreeWidget()
        self.day_tree.setHeaderLabels(["Day directory"])
        self.day_tree.setRootIsDecorated(True)   # show the >/v expand toggles
        self.day_tree.setItemsExpandable(True)
        self.day_tree.setUniformRowHeights(True)
        self.day_tree.setMinimumWidth(self.ptreewid)
        self.day_tree.itemClicked.connect(self._on_day_tree_item)
        params_split.addWidget(self.day_tree)
        params_split.setStretchFactor(0, 0)
        params_split.setStretchFactor(1, 1)
        self.Dock_Params.addWidget(params_split)

        # the image and its 2-D spectrum live in one
        # dock (top-right); the .index text, day-spectra plot, changes and help are
        # tabbed docks beneath it.
        self.Dock_Image = PGD.Dock("Image + Spectrum", size=(win_wid - self.ptreewid, win_ht))
        self._build_image_dock()

        self.Dock_Report = PGD.Dock("Reporting (.index)", size=(win_wid - self.ptreewid, 320))
        self.textbox = QtWidgets.QTextEdit()
        self.textbox.setReadOnly(True)
        self.textbox.setText("(.index file)")
        self.Dock_Report.addWidget(self.textbox)

        # 1-D radial spectra for a day dir (not implemented)
        self.Dock_DaySpectra = PGD.Dock("Day Spectra", size=(win_wid - self.ptreewid, 320))
        self.day_spectra_plot = pg.PlotWidget()
        self.day_spectra_plot.setLabel("bottom", "Spatial frequency", units="1/µm")
        self.day_spectra_plot.setLabel("left", "Power (log)")
        self.day_spectra_plot.addLegend()
        self.Dock_DaySpectra.addWidget(self.day_spectra_plot)

        self.Dock_Changes = PGD.Dock("Changes", size=(win_wid - self.ptreewid, 320))
        self.change_text = QtWidgets.QTextEdit()
        self.change_text.setReadOnly(True)
        self.change_text.setText("Changes")
        self.Dock_Changes.addWidget(self.change_text)

        self.Dock_Help = PGD.Dock("Help", size=(win_wid - self.ptreewid, 320))
        self.help_text = QtWidgets.QTextEdit()
        self.help_text.setReadOnly(True)
        self.help_text.setText("Help")
        self.Dock_Help.addWidget(self.help_text)

        self.DockArea.addDock(self.Dock_Params, "left")
        self.DockArea.addDock(self.Dock_Image, "right", self.Dock_Params)
        self.DockArea.addDock(self.Dock_Report, "bottom", self.Dock_Image)
        self.DockArea.addDock(self.Dock_DaySpectra, "below", self.Dock_Report)
        self.DockArea.addDock(self.Dock_Changes, "below", self.Dock_DaySpectra)
        self.DockArea.addDock(self.Dock_Help, "below", self.Dock_Changes)
        self.Dock_Report.raiseDock()  # show the .index tab first
        self.ptreedata.sigTreeStateChanged.connect(self.command_dispatcher)
        # Status bar at the bottom shows the full protocol
        self.status_path = QtWidgets.QLabel("No protocol selected")
        try:
            self.status_path.setTextInteractionFlags(
                QtCore.Qt.TextInteractionFlag.TextSelectableByMouse
            )
        except Exception:  # noqa: BLE001 - enum path differs across Qt bindings
            pass
        self.win.statusBar().addWidget(self.status_path, 1)
        self.put_help()
        self.win.show()

    def _set_status(self, msg: str):
        """Show a message (typically the current protocol path) in the status bar."""
        if self.status_path is not None:
            self.status_path.setText(str(msg))

    def quit(self):
        exit(0)

    def put_help(self):
        help_text = """
        <h1>Fix Objective</h1>
        <p>This tool is used to correct the objective scale factor in the .index file for a set of images or videos.</p>
        <p>First select the directory that holds the image data you wish to correct by selecting the 'Set Directory/Protocol' button.</p>
        <p>When you have selected this directory, the 'Reporting' window should show the .index file data entries for the images and videos in the directory.</p>
        <p>Identify the images and videos you want to change by using the 'seqparse' format of the corresponding numbers in the 'Images' and 'Videos' fields.</p>
        <p> For example, if you want to change images 1, 2, and 3, you would enter '1;3' in the 'Images' field.
        if you wnat to change images 1, 3, and 5, enter '1,3,5' in the image field. Note that the image and video numbers start at 0.
        </p>
        <p> You can then view the proposed changes by selecting the 'View proposed changes' button.</p>
        <p>It is also important to select the correct objective for the data you are working with.</p>
        <p>Once you have selected the camera, cine scale, and the images or videos you wish to correct, you can select the new objective and apply the changes.</p>
        <p> If you are satisfied with the changes, you can apply them by selecting the 'Apply changes' button.</p>
        <p>It is important to select the correct camera and cine scale for the data you are working with.</p>
        <p>It is important to note that the changes are not reversible, so be sure you have selected the correct data and the correct objective before applying the changes.</p>
        <p>Once you have applied the changes, you can view the changes in the .index file by selecting the 'View .index' button.</p>
        <p>Once you have applied the changes, you can quit the program by selecting the 'Quit' button.</p>
        """
        self.help_text.setHtml(help_text)

def main():
    app = pg.mkQApp()
    app.setStyle("fusion")

    # Load the objective config (working directory-independent)
    # and, if available, the experiment configuration for the raw-data
    # directory. A missing/offline experiments.cfg is non-fatal.
    objective_cfg, objective_cfg_path = get_configuration("ephys/config/fix_objective_data.cfg")
    datasets, experiments = None, None
    try:
        datasets, experiments = get_experiment_configuration(
            "config/experiments.cfg", check_completeness=False
        )
    except Exception as exc:  # noqa: BLE001 - missing file / offline data volume
        CP.cprint(
            "y",
            f"fix_objscale: experiments.cfg not loaded ({exc}); "
            "falling back to the most-recent directory.",
        )

    FO = FixObjective(
        app=app,
        objective_data=objective_cfg,
        objective_config_path=objective_cfg_path,
        datasets=datasets,
        experiments=experiments,
    )
    app.aboutToQuit.connect(
        FO.quit
    )  # prevent python exception when closing window with system control
    FO.set_window()

    if (sys.flags.interactive != 1) or not hasattr(pg.QtCore, "PYQT_VERSION"):
        pg.QtWidgets.QApplication.instance().exec()

if __name__ == "__main__":
    main()
