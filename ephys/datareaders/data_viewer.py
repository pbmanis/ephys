#!/usr/bin/python
"""
data_viewer: a stand-alone PyQt + pyqtgraph tool for *viewing* electrophysiology
data through the common acq4_reader-style interface.

It is deliberately format-agnostic: it opens any supported data source through
:func:`ephys.datareaders.get_reader.get_reader` (which sniffs acq4 / ephus /
Axon ABF / matdatac), lists the protocols inside the file/directory, and plots
the primary (recorded) and command traces for the selected protocol -- labeled
correctly whether the recording is voltage clamp (current + command voltage) or
current clamp (voltage + command current).

This is a viewing/testing tool only: it performs NO analysis and modifies no
data.  The files used while building the readers are offered as presets, and the
"Open file..." / "Open folder..." buttons accept any other matching data.

Run:
    python ephys/datareaders/data_viewer.py [PATH]
    python -m ephys.datareaders.data_viewer  [PATH]

2026, Paul B. Manis.
"""

import os
import sys
from pathlib import Path

import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets

from ephys.datareaders.get_reader import get_reader, sniff_format, _is_datac


# ---------------------------------------------------------------------------
# persistent settings (TOML). Stored per-user under an OS-appropriate app-config
# directory: on macOS ~/Library/Application Support/Ephys/data_viewer/settings.toml.
# The file is structured (tables) so more settings can be added later.
# ---------------------------------------------------------------------------
def _config_dir() -> Path:
    if sys.platform == "darwin":
        base = Path.home() / "Library" / "Application Support"
    elif os.name == "nt":
        base = Path(os.environ.get("APPDATA", str(Path.home())))
    else:
        base = Path(os.environ.get("XDG_CONFIG_HOME", str(Path.home() / ".config")))
    return base / "Ephys" / "data_viewer"


def _config_path() -> Path:
    return _config_dir() / "settings.toml"


def _load_config() -> dict:
    """Read the settings TOML into a dict ({} if missing/unreadable)."""
    p = _config_path()
    if not p.is_file():
        return {}
    try:
        import tomllib  # Python 3.11+
        with open(p, "rb") as fh:
            return tomllib.load(fh)
    except Exception:  # noqa: BLE001 - corrupt/unreadable config is non-fatal
        return {}


def _toml_value(v) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        return repr(v)
    s = str(v).replace("\\", "\\\\").replace('"', '\\"')
    return f'"{s}"'


def _toml_dump(cfg: dict) -> str:
    """Minimal TOML writer for a dict of tables ({section: {key: value}}) with
    string/number/bool values (no external dependency)."""
    lines = []
    for section, values in cfg.items():
        if not isinstance(values, dict):
            continue
        lines.append(f"[{section}]")
        for k, v in values.items():
            lines.append(f"{k} = {_toml_value(v)}")
        lines.append("")
    return "\n".join(lines)


def _save_config(cfg: dict) -> None:
    """Write the settings dict to the TOML file (best-effort, non-fatal)."""
    p = _config_path()
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(_toml_dump(cfg))
    except Exception:  # noqa: BLE001 - never let a settings write crash the GUI
        pass


# ---------------------------------------------------------------------------
# preset test files (only those that exist are shown).
# These live in the package's local test_data/ directory (not tracked in git),
# so the viewer never reaches out to external data volumes for its examples.
# ---------------------------------------------------------------------------
_TESTDATA = Path(__file__).resolve().parent / "test_data"
PRESETS = [
    ("datac — 01DEC95F.JSR (VC, .TXT notes)", _TESTDATA / "01DEC95F.JSR"),
    ("datac — 7DEC90D.PBM (CC, .NOT notes)", _TESTDATA / "7DEC90D.PBM"),
    ("datac — 2DEC88C.PBM (CC, no notes)", _TESTDATA / "2DEC88C.PBM"),
    ("matdatac — 04aug11a.mat", _TESTDATA / "04aug11a.mat"),
    ("abf — 17JUN11a-Bushy-sumRec1_20.ABF", _TESTDATA / "17JUN11a-Bushy-sumRec1_20.ABF"),
]


# ---------------------------------------------------------------------------
# protocol enumeration (per format)
# ---------------------------------------------------------------------------
def enumerate_protocols(path):
    """Return a list of ``(label, protocol_path)`` for the data at ``path``.

    A "protocol" is whatever the reader's ``setProtocol`` accepts and
    ``getData`` then reads: an .abf file, a matdatac block, an ephus map dir,
    or an acq4 protocol dir.  A file/dir may hold several protocols.
    """
    p = Path(path)
    try:
        fmt = sniff_format(p)
    except Exception as exc:  # noqa: BLE001
        return [], f"unrecognized ({exc})"

    if fmt == "abf":
        return [(p.name, p)], fmt

    if fmt == "matdatac":
        r = get_reader(p)
        dc = r._open_datac(p)
        out = []
        for k in r.data_block_indices(p):
            t = r._block_type(dc, k)
            out.append((f"{k:03d}_{t}", p / f"{k:03d}_{t}"))
        return out, fmt

    if fmt == "ephus":
        # a map dir holds the .xsg directly; a cell/day dir holds map subdirs
        if any(p.glob("*.xsg")):
            return [(p.name, p)], fmt
        maps = sorted(
            d for d in p.rglob("map*") if d.is_dir() and any(d.glob("*.xsg"))
        )
        if maps:
            return [(str(m.relative_to(p)), m) for m in maps], fmt
        return [(p.name, p)], fmt

    if fmt == "datac":
        # A datac file is a "cell" of blocks (from its .NOT, or one "000_all"
        # block when absent).  A folder is offered as its (content-validated)
        # datac files, each selectable as a cell.
        if p.is_file():
            r = get_reader(p, fmt="datac")
            return [(lbl, p / lbl) for lbl in r.data_block_labels(p)], fmt
        files = sorted(f for f in p.iterdir() if f.is_file() and _is_datac(f)) if p.is_dir() else []
        return [(f.name, f) for f in files], fmt

    if fmt == "acq4":
        # A protocol dir has numeric subdirs (000, 001, ...) each with a device
        # .ma; a cell dir holds such protocol dirs.  Detect either case.
        def _is_protocol(d):
            return d.is_dir() and any(
                (sd / "MultiClamp1.ma").exists() or (sd / "Clamp1.ma").exists()
                for sd in d.iterdir() if sd.is_dir() and sd.name.isdigit()
            )
        if _is_protocol(p):
            return [(p.name, p)], fmt
        protos = sorted(d for d in p.iterdir() if _is_protocol(d)) if p.is_dir() else []
        if protos:
            return [(d.name, d) for d in protos], fmt
        return [(p.name, p)], fmt

    return [(p.name, p)], fmt


# ---------------------------------------------------------------------------
# display scaling / labels by clamp mode
# ---------------------------------------------------------------------------
def _display_spec(mode):
    """Return ((primary_scale, primary_label), (command_scale, command_label))
    for a clamp ``mode`` ('VC' or 'IC'/'I=0')."""
    if str(mode).upper().startswith("V"):  # voltage clamp: record current
        return (1e12, "Membrane current (pA)"), (1e3, "Command voltage (mV)")
    # current clamp (or I=0): record voltage
    return (1e3, "Membrane potential (mV)"), (1e12, "Command current (pA)")


class DataViewer(QtWidgets.QMainWindow):
    def __init__(self, startpath=None):
        super().__init__()
        self.setWindowTitle("ephys data viewer")
        self.reader = None
        self.current_path = None

        # persistent settings: recall the last opened directory for the file
        # dialogs (if it still exists).
        self._config = _load_config()
        self.last_dir = None
        saved = self._config.get("paths", {}).get("last_dir")
        if saved and Path(saved).is_dir():
            self.last_dir = saved

        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        layout = QtWidgets.QHBoxLayout(central)

        # ---- left control panel ------------------------------------------
        left = QtWidgets.QVBoxLayout()
        layout.addLayout(left, 0)

        self.preset_combo = QtWidgets.QComboBox()
        self.preset_combo.addItem("— preset test files —", None)
        for label, pth in PRESETS:
            if Path(pth).exists():
                self.preset_combo.addItem(label, pth)
        self.preset_combo.currentIndexChanged.connect(self._on_preset)
        left.addWidget(self.preset_combo)

        btns = QtWidgets.QHBoxLayout()
        b_file = QtWidgets.QPushButton("Open file…")
        b_file.clicked.connect(self._open_file)
        b_dir = QtWidgets.QPushButton("Open folder…")
        b_dir.clicked.connect(self._open_dir)
        btns.addWidget(b_file)
        btns.addWidget(b_dir)
        left.addLayout(btns)

        self.path_label = QtWidgets.QLabel("(no file loaded)")
        self.path_label.setWordWrap(True)
        self.path_label.setStyleSheet("font-size: 10px; color: #888;")
        left.addWidget(self.path_label)

        left.addWidget(QtWidgets.QLabel("Protocols:"))
        self.proto_list = QtWidgets.QListWidget()
        self.proto_list.currentItemChanged.connect(self._on_proto_selected)
        left.addWidget(self.proto_list, 1)

        self.info = QtWidgets.QLabel("")
        self.info.setWordWrap(True)
        self.info.setTextFormat(QtCore.Qt.TextFormat.RichText)
        left.addWidget(self.info)

        # ---- right area: plots on top, notes panel below -----------------
        right = QtWidgets.QVBoxLayout()
        layout.addLayout(right, 1)

        self.glw = pg.GraphicsLayoutWidget()
        right.addWidget(self.glw, 1)  # plots flex to fill the space
        self.p_primary = self.glw.addPlot(row=0, col=0)
        self.p_cmd = self.glw.addPlot(row=1, col=0)
        self.p_primary.showGrid(x=True, y=True, alpha=0.2)
        self.p_cmd.showGrid(x=True, y=True, alpha=0.2)
        self.p_cmd.setXLink(self.p_primary)
        self.p_cmd.setLabel("bottom", "Time", units="ms")
        # give the command trace ~1/3 of the vertical space
        self.glw.ci.layout.setRowStretchFactor(0, 2)
        self.glw.ci.layout.setRowStretchFactor(1, 1)

        # notes panel: shows the .NOT / .TXT sidecar (if any), scrollable,
        # monospaced, at least 24 lines tall.
        self.notes_label = QtWidgets.QLabel("Notes: (none)")
        self.notes_label.setStyleSheet("font-size: 11px; color: #888;")
        right.addWidget(self.notes_label, 0)
        self.notes_view = QtWidgets.QPlainTextEdit()
        self.notes_view.setReadOnly(True)
        self.notes_view.setLineWrapMode(QtWidgets.QPlainTextEdit.LineWrapMode.NoWrap)
        mono = QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.SystemFont.FixedFont)
        mono.setPointSize(11)
        self.notes_view.setFont(mono)
        fm = self.notes_view.fontMetrics()
        self.notes_view.setMinimumHeight(fm.lineSpacing() * 24 + 12)  # >= 24 lines
        right.addWidget(self.notes_view, 0)

        self.resize(1200, 1040)  # taller to fit the notes panel under the plots
        if startpath:
            self._load(startpath)

    # -- file loading ------------------------------------------------------
    def _on_preset(self, _idx):
        pth = self.preset_combo.currentData()
        if pth:
            self._load(pth)

    def _open_file(self):
        fn, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Open a data file", self.last_dir or "",
            "Data files (*.abf *.mat *.xsg *.ma);;All files (*)",
        )
        if fn:
            self._remember_dir(fn)
            self._load(fn)

    def _open_dir(self):
        d = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Open a data folder", self.last_dir or "")
        if d:
            self._remember_dir(d)
            self._load(d)

    def _remember_dir(self, path):
        """Persist the directory of the just-opened path as the default location
        for the next session's file dialogs."""
        p = Path(path)
        d = str(p if p.is_dir() else p.parent)
        self.last_dir = d
        self._config.setdefault("paths", {})["last_dir"] = d
        _save_config(self._config)

    def _load(self, path):
        self.current_path = Path(path)
        self.path_label.setText(str(self.current_path))
        self.proto_list.clear()
        self.p_primary.clear()
        self.p_cmd.clear()
        try:
            protos, fmt = enumerate_protocols(self.current_path)
            self.reader = get_reader(self.current_path)
        except Exception as exc:  # noqa: BLE001
            self.info.setText(f"<span style='color:#c00'>Could not open: {exc}</span>")
            self.reader = None
            return
        self._protos = protos
        for label, ppath in protos:
            item = QtWidgets.QListWidgetItem(label)
            item.setData(QtCore.Qt.ItemDataRole.UserRole, str(ppath))
            self.proto_list.addItem(item)
        have_notes = self._load_notes()
        if protos:
            self.info.setText(f"format: <b>{fmt}</b> &nbsp; protocols: <b>{len(protos)}</b>")
            self.proto_list.setCurrentRow(0)
        else:
            # No readable data (e.g. a header-only datac file). Alert clearly and
            # fall back to whatever notes are available.
            extra = " (header only)" if fmt == "datac" else ""
            tail = "see notes below" if have_notes else "no notes file found"
            self.info.setText(
                f"format: <b>{fmt}</b><br>"
                f"<span style='color:#c00'><b>NO DATA RECORDS</b>{extra} — {tail}.</span>"
            )
            self.p_primary.setTitle("NO DATA — header only" if fmt == "datac" else "NO DATA")
            self.p_cmd.setTitle("")

    def _load_notes(self):
        """Show the .NOT / .TXT sidecar (if any) for the current data file in the
        scrollable notes panel."""
        note = None
        p = Path(self.current_path)
        # the datac reader knows how to locate .NOT/.TXT sidecars; fall back to a
        # generic sidecar search next to the file for other formats.
        finder = getattr(self.reader, "_find_note_file", None)
        if finder is not None:
            base = getattr(self.reader, "_file", None) or p
            try:
                note = finder(base)[0]
            except Exception:  # noqa: BLE001
                note = None
        if note is None and p.suffix:
            stem = p.with_suffix("")
            for ext in (".NOT", ".not", ".TXT", ".txt"):
                cand = stem.with_suffix(ext)
                if cand.is_file():
                    note = cand
                    break
        text = ""
        if note is not None and Path(note).is_file():
            try:
                text = Path(note).read_text(errors="replace")
            except Exception:  # noqa: BLE001
                text = ""
        self.notes_view.setPlainText(text)
        self.notes_view.moveCursor(QtGui.QTextCursor.MoveOperation.Start)
        self.notes_label.setText(f"Notes: {Path(note).name}" if note else "Notes: (none)")
        return bool(text)

    # -- plotting ----------------------------------------------------------
    def _on_proto_selected(self, item, _prev=None):
        if item is None or self.reader is None:
            return
        ppath = item.data(QtCore.Qt.ItemDataRole.UserRole)
        self._show_protocol(ppath)

    def _show_protocol(self, ppath):
        self.p_primary.clear()
        self.p_cmd.clear()
        try:
            self.reader.setProtocol(str(ppath))
            ok = self.reader.getData()
        except Exception as exc:  # noqa: BLE001
            self.info.setText(f"<span style='color:#c00'>getData failed: {exc}</span>")
            return
        if not ok:
            self.info.setText(
                "<span style='color:#c00'>getData returned False "
                f"({getattr(self.reader, 'error_info', '') or 'no data'})</span>"
            )
            return

        tb = np.asarray(self.reader.time_base, dtype=float) * 1e3  # ms
        traces = np.asarray(self.reader.traces, dtype=float)
        cmd = np.asarray(getattr(self.reader, "cmd_wave", None), dtype=float)
        if traces.ndim == 1:
            traces = traces[np.newaxis, :]
        if cmd.ndim == 1:
            cmd = cmd[np.newaxis, :]
        mode = getattr(self.reader, "mode", "IC")
        (pscale, plabel), (cscale, clabel) = _display_spec(mode)

        nsweeps = traces.shape[0]
        for i in range(nsweeps):
            pen = pg.intColor(i, hues=max(nsweeps, 6), alpha=200)
            self.p_primary.plot(tb[: traces.shape[1]], traces[i] * pscale, pen=pen)
            if cmd.shape[0] > i and cmd.shape[1] == traces.shape[1]:
                self.p_cmd.plot(tb[: cmd.shape[1]], cmd[i] * cscale, pen=pen)

        self.p_primary.setLabel("left", plabel)
        self.p_cmd.setLabel("left", clabel)
        self.p_primary.setTitle(f"{Path(ppath).name}  —  {mode}  —  {nsweeps} sweep(s)")

        rate = None
        try:
            sr = np.asarray(self.reader.sample_rate, dtype=float)
            rate = float(sr.ravel()[0]) if sr.size else None
        except Exception:  # noqa: BLE001
            rate = None
        holding = getattr(self.reader, "holding", None)
        units = getattr(self.reader, "units", ["", ""])
        hold_txt = ""
        if holding is not None:
            hold_txt = (
                f" &nbsp; holding: <b>{holding*1e3:.0f} mV</b>"
                if str(mode).upper().startswith("V")
                else f" &nbsp; holding: <b>{holding*1e12:.0f} pA</b>"
            )
        self.info.setText(
            f"protocol: <b>{Path(ppath).name}</b><br>"
            f"mode: <b>{mode}</b> &nbsp; sweeps: <b>{nsweeps}</b>"
            + (f" &nbsp; rate: <b>{rate/1e3:.1f} kHz</b>" if rate else "")
            + f"<br>primary units: <b>{units[0]}</b>{hold_txt}"
        )


def main():
    startpath = sys.argv[1] if len(sys.argv) > 1 else None
    app = pg.mkQApp("ephys data viewer")
    win = DataViewer(startpath=startpath)
    win.show()
    if hasattr(app, "exec"):
        app.exec()
    else:
        app.exec_()


if __name__ == "__main__":
    main()
