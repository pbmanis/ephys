from __future__ import print_function

#!/usr/bin/python

"""
ephus_reader: read Ephus / XSG laser-scanning-photostimulation (LSPS) data and
present it through the same public interface as ``acq4_reader.acq4_reader``.

Ephus (Suter et al., 2010; the "ephus"/"pkmirrors" era of the Kanold/Manis labs,
~2007-2015) stores data very differently from acq4:

  * acq4 writes **one sweep per subdirectory**
    (``protocol/000/MultiClamp1.ma``, ``protocol/001/...``) and a ``.index``
    metadata file at every level of the tree.
  * ephus writes **one ``.xsg`` file per map** and concatenates every sweep of
    that map into a single trace (e.g. a 4x4 map = 16 spots x 5000 pts =
    80000 pts).  There are no ``.index`` files; the metadata lives in the xsg
    header struct plus sidecar ``.m`` (MATLAB cell/slice/animal info) and
    ``.txt`` (notes) files.

To let the rest of the ephys package use ephus data unchanged, this reader:

  * splits the single xsg trace back into N per-spot sweeps,
  * presents N synthetic per-spot subdirectories via :meth:`subDirs` so code
    that iterates sweep directories keeps working,
  * synthesizes an acq4-style ``.index`` dict from the xsg header + ``.m``/``.txt``,
  * fills the same output attributes acq4_reader fills (``self.traces`` as a
    MetaArray, ``self.data_array``, ``self.time_base``, ``self.scanner_positions``,
    ``self.mode``, ``self.sample_rate``, ``self.WCComp``/``self.CCComp``, ...).

Directory layout understood by this reader (one experiment "day"):

    031609pbm/                       <- experiment day  (mmddyy + initials)
      pk0001/                        <- a cell           (initials + expNumber)
        pk0001.m                     <- MATLAB metadata (may also sit at day level)
        pk0001.txt                   <- notes / log
        lsps_result.mat              <- OLD analysis output (optional; reference only)
        images/videoImg_pk0001_2.tif <- slice/cell video image
        map02/  map03/  ...          <- each map = one "protocol"
          pk0001MAAB0001.xsg         <- one xsg = one whole map (all sweeps)

This module is READ ONLY with respect to the data and does not modify any other
file in the ephys package.

2026, Paul B. Manis.
"""

import datetime
import gc
import pprint
import re
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import pylibrary.tools.cprint as CP
import scipy.io as sio

try:
    import tifffile as tf
except Exception:  # pragma: no cover - fall back to pylibrary's vendored copy
    import pylibrary.tools.tifffile as tf

import MetaArray as EM

pp = pprint.PrettyPrinter(indent=4)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def mat2dict(obj):
    """Recursively convert the objects returned by ``scipy.io.loadmat``
    (with ``struct_as_record=False, squeeze_me=True``) into plain nested
    dicts / lists, leaving numeric ndarrays and scalars untouched.
    """
    if hasattr(obj, "_fieldnames"):
        return {f: mat2dict(getattr(obj, f)) for f in obj._fieldnames}
    if isinstance(obj, np.ndarray):
        if obj.dtype == object:
            return [mat2dict(x) for x in obj.flat]
        return obj
    return obj


def _first_value(d: dict):
    """Return the single value of a one-key dict (ephus wraps most header
    sections in a struct that has one same-named field)."""
    if isinstance(d, dict) and len(d) >= 1:
        return next(iter(d.values()))
    return d


def _as_str(v) -> str:
    """Coerce an ephus header field to a python string (empty char arrays
    load as 0-length ndarrays)."""
    if v is None:
        return ""
    if isinstance(v, np.ndarray):
        if v.size == 0:
            return ""
        return "".join(str(x) for x in v.ravel())
    return str(v)


def _as_float(v, default: float = 0.0) -> float:
    try:
        if isinstance(v, np.ndarray):
            if v.size == 0:
                return default
            return float(v.ravel()[0])
        return float(v)
    except (TypeError, ValueError):
        return default


def _as_list(v) -> list:
    """Normalize an ephus header field that may be a single struct (dict) or a
    list of structs (object array -> list) into a list."""
    if v is None:
        return []
    if isinstance(v, list):
        return v
    if isinstance(v, np.ndarray):
        return list(v.flat)
    return [v]


# unit strings -> multiplier to SI (Amperes or Volts)
_UNIT_SCALE = {
    "a": 1.0,
    "ma": 1e-3,
    "ua": 1e-6,
    "na": 1e-9,
    "pa": 1e-12,
    "v": 1.0,
    "mv": 1e-3,
    "uv": 1e-6,
}


def parse_dot_m(path: Union[str, Path]) -> dict:
    """Parse an ephus sidecar ``.m`` metadata file.

    These are MATLAB scripts of simple ``name = value; % comment`` assignments
    (see the headerGUI fields).  We evaluate the right-hand side loosely:
    quoted strings become str, bare numbers become float, everything else is
    kept as a stripped string.
    """
    meta: dict = {}
    path = Path(path)
    if not path.is_file():
        return meta
    assign_re = re.compile(r"^\s*([A-Za-z_]\w*)\s*=\s*(.*?);")
    with open(path, "r", errors="replace") as fh:
        for line in fh:
            line = line.split("%", 1)[0]  # strip trailing comment
            m = assign_re.match(line)
            if m is None:
                continue
            key, raw = m.group(1), m.group(2).strip()
            if len(raw) >= 2 and raw[0] in "'\"" and raw[-1] in "'\"":
                meta[key] = raw[1:-1]
            else:
                try:
                    meta[key] = float(raw)
                except ValueError:
                    meta[key] = raw
    return meta


def parse_txt_notes(path: Union[str, Path]) -> str:
    """Return the text of an ephus ``.txt`` notes/log sidecar file."""
    path = Path(path)
    if not path.is_file():
        return ""
    with open(path, "r", errors="replace") as fh:
        return fh.read().strip()


class ephus_reader:
    """Read an ephus/xsg map directory through the acq4_reader interface.

    A "protocol" for this reader is a single map directory (e.g. ``map02``)
    that contains exactly one ``.xsg`` file holding all sweeps of the map.
    """

    #: default values for settings that ephus does NOT store in the xsg file
    #: (Bessel low-pass cutoff and holding potential/current).  Override per
    #: instance via the ``defaults`` constructor argument or :meth:`set_defaults`,
    #: e.g. from an experiment config, since these are rig/experiment specific.
    DEFAULTS = {
        "LPF": 2000.0,        # Hz, primary-signal Bessel low-pass (not in file)
        "holding_VC": -0.060,  # V, default voltage-clamp holding (not in file)
        "holding_IC": 0.0,     # A, default current-clamp holding (not in file)
        # rig alignment shift applied to map->image placement, matching the
        # lab's lsps_readfile.m (S.Xoffset/S.Yoffset; 50/-50 px for the 3/09
        # Kanold-lab data).  Configure per rig/dataset.
        "map_x_offset_px": 50.0,
        "map_y_offset_px": -50.0,
        # the "photodiode1" acquirer channel appears to record ~10x the VC
        # holding voltage (no actual photodiode pulse is present); DC / this gain
        # gives the putative per-map holding.  Verified against the .txt notes
        # for pk0002-pk0008 (clean -60/0 mV); pk0001 is anomalous.
        "acquirer_holding_gain": 10.0,
    }

    def __init__(
        self,
        pathtoprotocol: Union[Path, str, None] = None,
        dataname: Union[str, None] = None,
        defaults: Union[dict, None] = None,
    ) -> None:
        self.protocol = None
        self.last_protocol = None
        self.last_dirs: List[Path] = []
        self.bad_clamp_mode = False
        if pathtoprotocol is not None:
            self.setProtocol(pathtoprotocol)
        if dataname is None:
            dataname = "MultiClamp1.ma"  # accepted for parity; the ephys channel is always used
        self.setDataName(dataname)
        self.clampInfo = {}
        self.lb = "\n"

        # establish known clamp devices (mirrors acq4_reader so that
        # data_summary recognizes the synthesized device names)
        maxclamps = 4
        clamps = []
        self.clampdevices = []
        for nc in range(maxclamps):
            cno = nc + 1
            cname = "Clamp%d" % cno
            mcname = "MultiClamp%d" % cno
            clamps.extend([(cname, "Pulse_amplitude"), (mcname, "Pulse_amplitude")])
            self.clampdevices.extend([cname, mcname])
        aon = "AxoPatch200"
        apn = "AxoProbe"
        clamps.extend([(aon, "Pulse_amplitude"), (apn, "Pulse_amplitude")])
        self.clampdevices.extend([aon, apn])
        self.clamps = clamps
        # ephus files always land on the first (Multi)Clamp name
        self.ephus_device = "MultiClamp1"

        self.tstamp = re.compile(r"\s*(__timestamp__: )([\d.\d]*)")
        self.clampInfo["dirs"] = []
        self.clampInfo["missingData"] = []
        self.traces = []
        self.data_array = []
        self.commandLevels = []
        self.cmd_wave = []
        self.time_base = []
        self.values = []
        self.trace_StartTimes = np.zeros(0)
        self.sample_rate = []
        self.pre_process_filters = {"LPF": None, "Notch": []}

        self.importantFlag = False
        self.error_info = None

        # ephus-specific caches
        self._xsg_cache: dict = {}  # path -> mat2dict(header/data)
        self._warned_multi_xsg: set = set()  # dirs already warned about >1 xsg
        self.mode = None
        # config-driven defaults for values ephus does not store (LPF, holding)
        self.defaults = dict(self.DEFAULTS)
        if defaults:
            self.defaults.update(defaults)

    # -- context manager -----------------------------------------------------
    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        for attr in (
            "time_base",
            "traces",
            "data_array",
            "cmd_wave",
            "values",
            "clamps",
            "sample_rate",
            "clampInfo",
            "clampdevices",
            "_xsg_cache",
        ):
            if hasattr(self, attr):
                delattr(self, attr)
        gc.collect()

    # -----------------------------------------------------------------------
    # configuration
    # -----------------------------------------------------------------------
    def setImportant(self, flag: bool = False) -> None:
        self.importantFlag = flag
        CP.cprint("r", f"Important' flag was set: {flag:b}")

    def setProtocol(self, pathtoprotocol: Union[str, Path, None] = None) -> None:
        self.protocol = pathtoprotocol

    def set_pre_process(self, LPF: Union[None, float] = None, Notch: Union[None, list] = None):
        self.pre_process_filters["LPF"] = LPF
        self.pre_process_filters["Notch"] = Notch

    def set_defaults(self, **kwargs) -> None:
        """Override the defaults for values ephus does not record (``LPF``,
        ``holding_VC``, ``holding_IC``).  Intended to be driven from an
        experiment config, e.g. ``reader.set_defaults(LPF=4000.0,
        holding_VC=-0.070)``."""
        self.defaults.update(kwargs)

    def setDataName(self, dataname: Union[str, Path]) -> None:
        self.dataname = dataname
        self.shortdname = str(Path(self.dataname).stem)

    # -----------------------------------------------------------------------
    # ephus low-level readers
    # -----------------------------------------------------------------------
    def find_xsg(self, mapdir: Union[str, Path]) -> Optional[Path]:
        """Return the single ``.xsg`` file inside a map directory, or None."""
        mapdir = Path(mapdir)
        if not mapdir.is_dir():
            return None
        xsgs = sorted(mapdir.glob("*.xsg"))
        if len(xsgs) == 0:
            return None
        if len(xsgs) > 1 and str(mapdir) not in self._warned_multi_xsg:
            self._warned_multi_xsg.add(str(mapdir))
            CP.cprint(
                "y",
                f"ephus_reader.find_xsg: {len(xsgs):d} xsg files in {str(mapdir):s}; using {xsgs[0].name:s}",
            )
        return xsgs[0]

    def load_xsg(self, mapdir: Union[str, Path]) -> Optional[dict]:
        """Load (and cache) the xsg for a map directory as nested dicts.

        Returns a dict with keys ``header`` and ``data`` (both plain
        dict/ndarray), or None if there is no xsg.
        """
        mapdir = Path(mapdir)
        xsg = self.find_xsg(mapdir)
        if xsg is None:
            return None
        key = str(xsg)
        if key in self._xsg_cache:
            return self._xsg_cache[key]
        mat = sio.loadmat(key, struct_as_record=False, squeeze_me=True)
        out = {
            "path": xsg,
            "header": mat2dict(mat["header"]),
            "data": mat2dict(mat["data"]),
        }
        self._xsg_cache[key] = out
        return out

    # -- convenience accessors into the (double-nested) header ---------------
    def _hdr_section(self, xsg: dict, name: str) -> dict:
        sect = xsg["header"].get(name, {})
        return _first_value(sect) if isinstance(sect, dict) else sect

    def _amp_settings(self, ephys: dict) -> dict:
        amp = ephys.get("amplifierSettings", {})
        return _first_value(amp) if isinstance(amp, dict) else {}

    def _mode_and_units(self, xsg: dict):
        """Return (mode, primary_units, primary_scale, cmd_units) from the amp
        settings.  mode is 'VC' or 'IC'."""
        ephys = self._hdr_section(xsg, "ephys")
        amp = self._amp_settings(ephys)
        amp_mode = _as_str(amp.get("mode", "")).lower()
        if "v-clamp" in amp_mode or amp_mode == "vc":
            mode = "VC"
            in_units = _as_str(amp.get("input_units", "pA")) or "pA"
        else:
            mode = "IC"
            in_units = _as_str(amp.get("input_units", "mV")) or "mV"
        scale = _UNIT_SCALE.get(in_units.lower(), 1.0)
        primary_units = "A" if mode == "VC" else "V"
        cmd_units = "V" if mode == "VC" else "A"
        return mode, primary_units, scale, cmd_units

    def amplifier_info(self, mapdir: Union[str, Path, None] = None) -> dict:
        """Return the amplifier settings for a map: recording mode and gains
        read from the xsg, plus the LPF and holding **defaults** (which ephus
        does not store).  Handy for display."""
        if mapdir is None:
            mapdir = self.protocol
        xsg = self.load_xsg(mapdir)
        if xsg is None:
            return {}
        mode, primary_units, _scale, cmd_units = self._mode_and_units(xsg)
        ephys = self._hdr_section(xsg, "ephys")
        amp = self._amp_settings(ephys)
        holding = _as_float(
            self.defaults.get("holding_VC" if mode == "VC" else "holding_IC", 0.0)
        )
        return {
            "mode": mode,
            "amp_mode": _as_str(amp.get("mode", "")),
            "input_gain": _as_float(amp.get("input_gain", 1.0), 1.0),
            "output_gain": _as_float(amp.get("output_gain", 1.0), 1.0),
            "input_units": _as_str(amp.get("input_units", "")),
            "output_units": _as_str(amp.get("output_units", "")),
            "sample_rate": _as_float(ephys.get("sampleRate", 0.0)),
            "LPF": _as_float(self.defaults.get("LPF", 0.0)),
            "holding": holding,
            "acquirer_dc": self.acquirer_dc(mapdir),
            "putative_holding": self.putative_holding(mapdir),
            "defaults_note": "LPF and holding are defaults (not stored in ephus files)",
        }

    def header_metadata(self, mapdir: Union[str, Path, None] = None) -> dict:
        """Capture the descriptive metadata stored in the xsg header: the
        headerGUI animal/slice/cell fields, the xsg file identity, and the
        autonotes file pointer.  Many fields are often blank in the file (the
        experimenter used the sidecar .m/.txt instead), so for the data_summary
        export these should be merged with parse_dot_m()/parse_txt_notes()."""
        if mapdir is None:
            mapdir = self.protocol
        xsg = self.load_xsg(mapdir)
        if xsg is None:
            return {}
        g = self._hdr_section(xsg, "headerGUI")
        x = self._hdr_section(xsg, "xsg")
        an = self._hdr_section(xsg, "autonotes")
        return {
            # animal / slice
            "species": _as_str(g.get("speciesStrain", "")),
            "age": _as_float(g.get("animalAge", 0.0)),
            "sex": _as_str(g.get("gender", "")),
            "strain": _as_str(g.get("transgenicLine", "")),
            "temperature": _as_float(g.get("temp", 0.0)),
            "experimenter": _as_str(g.get("experimenter", "")),
            "rig": _as_str(g.get("rig", "")),
            "slice_type": _as_str(g.get("sliceType", "")),
            # recorded cell (#1)
            "cell_type": _as_str(g.get("cellType1", "")),
            "brain_area": _as_str(g.get("brainArea1", "")),
            "region": _as_str(g.get("region1", "")),
            "subregion": _as_str(g.get("subregion1", "")),
            "soma_z": _as_float(g.get("somaZ1", 0.0)),
            "v_rest": _as_float(g.get("vRest1", 0.0)),
            # solutions
            "internal": _as_str(g.get("intracellularSolutionType", "")),
            "internal_drugs": _as_str(g.get("intracellularSolutionDrugs", "")),
            "bath": _as_str(g.get("bathSolutionType", "")),
            "bath_drugs": _as_str(g.get("bathSolutionDrugs", "")),
            # xsg file identity
            "initials": _as_str(x.get("initials", "")),
            "experiment_number": _as_str(x.get("experimentNumber", "")),
            "set_id": _as_str(x.get("setID", "")),
            "acquisition_number": _as_str(x.get("acquisitionNumber", "")),
            "timestamp": _as_str(x.get("xsgFileCreationTimestamp", "")),
            "notes_file": _as_str(an.get("filename", "")),
        }

    # -----------------------------------------------------------------------
    # directory navigation
    # -----------------------------------------------------------------------
    def _is_spot_dir(self, p: Path):
        """If ``p`` is a synthetic per-spot subdir (``mapdir/NNN``), return
        (mapdir, spot_index); else (None, None)."""
        p = Path(p)
        if re.fullmatch(r"\d{3}", p.name) and self.find_xsg(p.parent) is not None:
            return p.parent, int(p.name)
        return None, None

    def _sweep_shape(self, xsg: dict):
        """Return (nspots, npts) for a map's ephys trace.

        The number of sweeps is derived from the *actual* trace length divided
        by the expected points-per-sweep (``sampleRate * traceLength``).  This
        is more reliable than the header ``positionNumber``, which can be stale
        for non-map protocols: e.g. a 'flashes' run carried a mapper
        ``positionNumber`` of 169 while only 6 sweeps were acquired.  Falls
        back to ``positionNumber`` if the traceLength is missing, then to a
        single sweep.
        """
        ephys = self._hdr_section(xsg, "ephys")
        sr = _as_float(ephys.get("sampleRate", 10000.0), 10000.0)
        tl = _as_float(ephys.get("traceLength", 0.0))
        trace = self._ephys_trace(xsg)
        if trace is None:
            return 0, 0
        exp_pts = int(round(sr * tl)) if tl > 0 else 0
        if exp_pts > 0 and trace.size % exp_pts == 0:
            return trace.size // exp_pts, exp_pts
        mapper = self._hdr_section(xsg, "mapper")
        npos = int(_as_float(mapper.get("positionNumber", 0)))
        if npos > 0 and trace.size % npos == 0:
            return npos, trace.size // npos
        if exp_pts > 0:
            return max(1, trace.size // exp_pts), exp_pts
        return 1, trace.size

    def n_spots(self, mapdir: Union[str, Path]) -> int:
        """Number of sweeps/spots in a map's xsg (see :meth:`_sweep_shape`)."""
        xsg = self.load_xsg(mapdir)
        if xsg is None:
            return 0
        return self._sweep_shape(xsg)[0]

    def subDirs(self, p: Union[str, Path]) -> List:
        """Return the synthetic per-spot subdirectories for a map directory.

        Mirrors acq4_reader.subDirs, which returns one directory per sweep.
        For ephus, the sweeps live inside a single xsg, so we return synthetic
        paths ``p/000``, ``p/001``, ... (these are not real directories).
        """
        p = Path(p)
        if self.last_protocol == p:
            return self.last_dirs
        nspots = self.n_spots(p)
        if nspots == 0:
            self.last_dirs = []
            self.last_protocol = p
            return []
        dirs = [p / f"{i:03d}" for i in range(nspots)]
        CP.cprint("c", f"ephus_reader: {nspots:d} synthetic spot dirs under {str(p):s}")
        self.last_dirs = dirs
        self.last_protocol = p
        return dirs

    # -----------------------------------------------------------------------
    # synthesized .index handling
    # -----------------------------------------------------------------------
    def _xsg_timestamp(self, xsg: dict) -> float:
        """Return a POSIX timestamp from the ephys triggerTime (a 6-vector
        [Y M D h m s]) or 0.0 if unavailable."""
        ephys = self._hdr_section(xsg, "ephys")
        tt = ephys.get("triggerTime", None)
        if isinstance(tt, np.ndarray) and tt.size >= 6:
            y, mo, d, h, mi, s = [float(x) for x in tt[:6]]
            try:
                return datetime.datetime(
                    int(y), int(mo), int(d), int(h), int(mi), int(s)
                ).timestamp()
            except ValueError:
                return 0.0
        return 0.0

    def _synthesize_protocol_index(self, mapdir: Path) -> dict:
        """Build an acq4-style top-level ``.index`` dict for a map directory."""
        xsg = self.load_xsg(mapdir)
        if xsg is None:
            return None
        mode, primary_units, _scale, cmd_units = self._mode_and_units(xsg)
        ephys = self._hdr_section(xsg, "ephys")
        positions = self.scanner_target_list(mapdir)
        notes = ""
        celldir = mapdir.parent
        for txt in sorted(celldir.glob("*.txt")):
            notes = parse_txt_notes(txt)
            break
        index = {
            ".": {
                "dirType": "ProtocolSequence",
                "__timestamp__": self._xsg_timestamp(xsg),
                "notes": notes,
                "description": _as_str(ephys.get("pulseSetNameArray", "")),
                "devices": {
                    self.ephus_device: {
                        "mode": mode,
                        "holdingCheck": False,
                        "holdingSpin": 0.0,
                        "primaryUnits": primary_units,
                    },
                    "Scanner": {"spotSize": self.spot_size(mapdir)},
                    "Photodiode": {"mode": "photodiode"},
                },
                "sequenceParams": {
                    ("Scanner", "targets"): positions,
                    ("protocol", "repetitions"): [0],
                },
            }
        }
        return index

    def _synthesize_spot_index(self, mapdir: Path, spot: int) -> dict:
        """Build an acq4-style ``.index`` dict for one synthetic spot dir."""
        xsg = self.load_xsg(mapdir)
        positions = self.scanner_target_list(mapdir)
        pos = positions[spot] if spot < len(positions) else [0.0, 0.0]
        return {
            ".": {
                "dirType": "Protocol",
                "__timestamp__": self._xsg_timestamp(xsg) if xsg else 0.0,
                "Scanner": {
                    "position": pos,
                    "spotSize": self.spot_size(mapdir),
                },
            }
        }

    def _synthesize_cell_index(self, celldir: Path) -> dict:
        """Build an acq4-style cell ``.index`` from the sidecar ``.m`` file."""
        celldir = Path(celldir)
        meta = {}
        # .m may live inside the cell folder or at the day level (pkNNNN.m)
        for cand in list(celldir.glob("*.m")) + list(celldir.parent.glob(celldir.name + ".m")):
            meta = parse_dot_m(cand)
            if meta:
                break
        notes = ""
        for txt in list(celldir.glob("*.txt")) + list(celldir.parent.glob(celldir.name + ".txt")):
            notes = parse_txt_notes(txt)
            if notes:
                break
        return {
            ".": {
                "dirType": "Cell",
                "notes": notes,
                "type": str(meta.get("cellType1", "")),
                "location": " ".join(
                    str(meta.get(k, "")) for k in ("brainArea1", "region1", "subregion1")
                ).strip(),
                "important": False,
            }
        }

    def getIndex(self, currdir: Union[str, Path, None] = None, lineend: str = "\n"):
        self.lb = lineend
        self._readIndex(currdir=currdir)
        if self._index is not None:
            return self._index["."]
        return None

    def _readIndex(self, currdir: Union[str, Path, None] = None):
        self._index = self.readDirIndex(currdir if currdir is not None else self.protocol)
        return self._index

    def readDirIndex(self, currdir: Union[str, Path, None] = None):
        """Return a synthesized acq4-style index dict for the given directory.

        Handles the three relevant levels: a synthetic per-spot subdir, a map
        (protocol) directory, and a cell directory.  Returns None if the path
        does not correspond to ephus data.
        """
        self._dirindex = None
        if currdir is None:
            currdir = self.protocol
        currdir = Path(currdir)

        mapdir, spot = self._is_spot_dir(currdir)
        if mapdir is not None:
            self._dirindex = self._synthesize_spot_index(mapdir, spot)
            return self._dirindex

        if self.find_xsg(currdir) is not None:  # a map/protocol directory
            self._dirindex = self._synthesize_protocol_index(currdir)
            return self._dirindex

        # otherwise treat as a cell directory if it holds maps or sidecar files
        if currdir.is_dir() and (
            any(currdir.glob("map*")) or any(currdir.glob("*.m")) or any(currdir.glob("*.xsg"))
        ):
            self._dirindex = self._synthesize_cell_index(currdir)
            return self._dirindex

        CP.cprint("y", f"ephus_reader.readDirIndex: no ephus data recognized at {str(currdir):s}")
        return None

    # -- index pretty-printing (mirrors acq4_reader) -------------------------
    def _parse_timestamp(self, lstr: str):
        ts = self.tstamp.match(lstr)
        if ts is not None:
            fts = float(ts.group(2))
            return datetime.datetime.fromtimestamp(fts).strftime("%Y-%m-%d  %H:%M:%S %z")
        return None

    def convert_timestamp(self, fts) -> str:
        return datetime.datetime.fromtimestamp(fts).strftime("%Y-%m-%d  %H:%M:%S %z")

    def printIndex(self, index):
        self.indent = 0
        self.textline = ""
        self.textline = pp.pformat(index)
        print(self.textline)

    def getIndex_text(self, index):
        return pp.pformat(index)

    def file_cell_protocol(self, filename: Union[str, Path, None] = None) -> tuple:
        assert filename is not None
        filename = Path(filename)
        proto = filename.stem
        cell = filename.parent
        sliceid = cell.parent
        date = sliceid.parent.name
        return (date, sliceid.name, cell.name, proto, sliceid.parent)

    def getClampDevices(self, currdir: Union[str, Path, None] = None, verbose: bool = False) -> list:
        info = self.getIndex(currdir=currdir)
        devs = []
        if info is not None and "devices" in info.keys():
            for d in info["devices"]:
                if d in self.clampdevices:
                    devs.append(d)
        return devs

    # -----------------------------------------------------------------------
    # protocol checking
    # -----------------------------------------------------------------------
    def checkProtocol(self, protocolpath: Union[str, Path, None] = None, allow_partial=False) -> bool:
        if protocolpath is None:
            protocolpath = self.protocol
        return self.find_xsg(protocolpath) is not None

    def checkProtocolImportantFlags(self, protocolpath: Union[str, Path, None] = None) -> bool:
        if protocolpath is None:
            protocolpath = self.protocol
        dirs = self.subDirs(protocolpath)
        self.important = {d.name: True for d in dirs}
        return True

    def checkProtocolImportant(self, protocol):
        self.setProtocol(protocol)
        self.info = self.getIndex(self.protocol)
        return self._getImportant(self.info)

    def listSequenceParams(self, dh):
        info = self.readDirIndex(dh)
        if info is None:
            return {}
        return info["."].get("sequenceParams", {})

    def _getImportant(self, info):
        if info is None:
            return False
        return bool(info.get("important", False)) if isinstance(info, dict) else False

    # -----------------------------------------------------------------------
    # clamp-info parsing (dict shapes identical to acq4_reader)
    # -----------------------------------------------------------------------
    def _clampstate(self, xsg: dict) -> dict:
        mode, primary_units, _scale, cmd_units = self._mode_and_units(xsg)
        ephys = self._hdr_section(xsg, "ephys")
        amp = self._amp_settings(ephys)
        sr = _as_float(ephys.get("sampleRate", 10000.0), 10000.0)
        primary_signal = "Membrane Current" if mode == "VC" else "Membrane Potential"
        secondary_signal = "Command Voltage" if mode == "VC" else "Command Current"
        # LPF and holding are NOT stored by ephus -> use configurable defaults
        lpf = _as_float(self.defaults.get("LPF", 0.0))
        holding = _as_float(
            self.defaults.get("holding_VC" if mode == "VC" else "holding_IC", 0.0)
        )
        clampparams = {
            "OutputZeroEnable": 0,
            "PipetteOffset": 0.0,
            "Holding": holding,
            "PrimarySignalHPF": 0.0,
            "BridgeBalResist": 0.0,
            "PrimarySignalLPF": lpf,
            "RsCompBandwidth": 0.0,
            "WholeCellCompResist": 0.0,
            "WholeCellCompEnable": 0,
            "LeakSubResist": 0.0,
            "HoldingEnable": int(holding != 0.0),
            "FastCompTau": 0.0,
            "SlowCompCap": 0.0,
            "WholeCellCompCap": 0.0,
            "LeakSubEnable": 0,
            "NeutralizationCap": 0.0,
            "BridgeBalEnable": 0,
            "RsCompCorrection": 0.0,
            "NeutralizationEnable": 0,
            "RsCompEnable": 0,
            "OutputZeroAmplitude": 0.0,
            "FastCompCap": 0.0,
            "SlowCompTau": 0.0,
        }
        return {
            "mode": mode,
            "primarySignal": primary_signal,
            "secondarySignal": secondary_signal,
            "primaryUnits": primary_units,
            "secondaryUnits": cmd_units,
            "primaryGain": _as_float(amp.get("input_gain", 1.0), 1.0),
            "secondaryGain": _as_float(amp.get("output_gain", 1.0), 1.0),
            "holding": holding,
            "ClampParams": clampparams,
            "DAQ": {
                "primary": {"rate": sr, "numPts": 0, "startTime": 0.0},
                "command": {"rate": sr, "numPts": 0, "startTime": 0.0},
                "secondary": {"rate": sr, "numPts": 0, "startTime": 0.0},
            },
        }

    def parseClampInfo(self, info: list):
        self.trClampInfo = info
        self.primary_trace_index = 0
        self.secondary_trace_index = 1
        self.command_trace_index = 1
        cs = info[1]["ClampState"]
        self.mode = cs["mode"]
        self.units = [cs["primaryUnits"], cs["secondaryUnits"]]
        self.samp_rate = cs["DAQ"]["primary"]["rate"]

    def parseClampWCCompSettings(self, info: list) -> dict:
        if "ClampState" in info[1].keys() and "ClampParams" in info[1]["ClampState"].keys():
            par = info[1]["ClampState"]["ClampParams"]
            return {
                "WCCompValid": True,
                "WCEnabled": par["WholeCellCompEnable"],
                "WCResistance": par["WholeCellCompResist"],
                "WCCellCap": par["WholeCellCompCap"],
                "RsCompCorrection": par["RsCompCorrection"],
                "CompEnabled": par["RsCompEnable"],
                "CompCorrection": par["RsCompCorrection"],
                "CompBW": par["RsCompBandwidth"],
            }
        return {
            "WCCompValid": False,
            "WCEnable": 0,
            "WCResistance": 0.0,
            "WholeCellCap": 0.0,
            "CompEnable": 0,
            "CompCorrection": 0.0,
            "CompBW": 50000.0,
            "RsCompCorrection": 0.0,
        }

    def parseClampCCCompSettings(self, info: list) -> dict:
        if "ClampState" in info[1].keys() and "ClampParams" in info[1]["ClampState"].keys():
            par = info[1]["ClampState"]["ClampParams"]
            return {
                "CCCompValid": True,
                "CCBridgeEnable": par["BridgeBalEnable"],
                "CCBridgeResistance": par["BridgeBalResist"],
                "CCNeutralizationEnable": par["NeutralizationEnable"],
                "CCNeutralizationCap": par["NeutralizationCap"],
                "CCLPF": par["PrimarySignalLPF"],
                "CCPipetteOffset": par["PipetteOffset"],
            }
        return {
            "CCCompValid": False,
            "CCBridgeEnable": 0,
            "CCBridgeResistance": 0.0,
            "CCNeutralizationEnable": 0.0,
            "CCNeutralizationCap": 0,
            "CCPipetteOffset": 0.0,
            "CCLPF": 10000.0,
        }

    def parseClampHoldingLevel(self, info: list):
        try:
            return info[1]["ClampState"]["holding"]
        except (KeyError, IndexError, TypeError):
            return 0.0

    def getDataInfo(self, filename: Union[str, Path, None] = None, silent: bool = False):
        """Return a 2-element info list ([time-axis, clampstate]) for a map's
        xsg, without loading the trace data.  ``filename`` may be a map dir or
        a synthetic spot dir / file within it."""
        assert filename is not None
        fn = Path(filename)
        mapdir, _spot = self._is_spot_dir(fn.parent if fn.suffix else fn)
        if mapdir is None:
            mapdir = fn if fn.is_dir() else fn.parent
        xsg = self.load_xsg(mapdir)
        if xsg is None:
            if not silent:
                CP.cprint("r", f"ephus_reader.getDataInfo: no xsg at {str(mapdir):s}")
            return None
        clampstate = self._clampstate(xsg)
        info = [{"name": "Time", "units": "s"}, {"ClampState": clampstate, "startTime": self._xsg_timestamp(xsg)}]
        self.parseClampInfo(info)
        return info

    # -----------------------------------------------------------------------
    # data
    # -----------------------------------------------------------------------
    def _ephys_trace(self, xsg: dict) -> Optional[np.ndarray]:
        data = xsg.get("data", {})
        ep = data.get("ephys", {})
        for k in ep:
            if k.startswith("trace"):
                return np.asarray(ep[k], dtype=float).ravel()
        return None

    def _acquirer_trace(self, xsg: dict) -> Optional[np.ndarray]:
        data = xsg.get("data", {})
        acq = data.get("acquirer", {})
        for k in acq:
            if k.startswith("trace"):
                return np.asarray(acq[k], dtype=float).ravel()
        return None

    def acquirer_dc(self, mapdir: Union[str, Path, None] = None):
        """Return the mean (DC) level of the acquirer ('photodiode1') channel in
        volts, or None.  This channel carries no laser pulse; its DC appears to
        be ~10x the VC holding voltage."""
        if mapdir is None:
            mapdir = self.protocol
        xsg = self.load_xsg(mapdir)
        if xsg is None:
            return None
        pd = self._acquirer_trace(xsg)
        return None if pd is None else float(np.mean(pd))

    def putative_holding(self, mapdir: Union[str, Path, None] = None):
        """Best-effort per-map holding voltage (volts) inferred from the acquirer
        DC divided by ``acquirer_holding_gain`` (default 10).  Verified against
        the .txt notes for pk0002-pk0008 (clean -60/0 mV); returns a value even
        when it is implausible (e.g. pk0001), so treat it as putative."""
        dc = self.acquirer_dc(mapdir)
        if dc is None:
            return None
        gain = _as_float(self.defaults.get("acquirer_holding_gain", 10.0), 10.0)
        return dc / gain if gain else None

    def getData(
        self,
        pos: int = 1,
        check: bool = False,
        allow_partial: bool = False,
        record_list: list = [],
        downsample: int = 1,
        silent: bool = True,
    ):
        """Read the map's xsg, split into per-spot sweeps, and fill the same
        attributes acq4_reader.getData fills.  Returns True on success."""
        if len(record_list) > 0 and not allow_partial:
            raise ValueError(
                "ephus_reader.getData: record_list not empty but allow_partial is False"
            )
        self.error_info = None
        mapdir = Path(self.protocol)
        xsg = self.load_xsg(mapdir)
        if xsg is None:
            self.error_info = f"No xsg found in {str(mapdir):s}"
            if check:
                return False
            raise FileNotFoundError(self.error_info)
        if check:
            return True

        mode, primary_units, scale, cmd_units = self._mode_and_units(xsg)
        self.units = [primary_units, cmd_units]
        ephys = self._hdr_section(xsg, "ephys")
        sr = _as_float(ephys.get("sampleRate", 10000.0), 10000.0)
        trace = self._ephys_trace(xsg)
        if trace is None:
            self.error_info = f"No ephys trace in xsg {str(mapdir):s}"
            CP.cprint("r", self.error_info)
            return False
        nspots, npts = self._sweep_shape(xsg)
        sweeps = trace[: npts * nspots].reshape(nspots, npts) * scale  # -> SI units

        self.clampInfo["dirs"] = self.subDirs(mapdir)
        self.clampInfo["missingData"] = []
        self.mode = mode
        self.sample_interval = 1.0 / sr
        self.sample_rate = np.ones(nspots) * sr
        self.time_base = np.arange(npts) / sr

        clampstate = self._clampstate(xsg)
        clampstate["DAQ"]["primary"]["rate"] = sr
        clampstate["DAQ"]["primary"]["numPts"] = npts

        # which sweeps to return
        if allow_partial and isinstance(record_list, list) and len(record_list) > 0:
            keep = [i for i in record_list if 0 <= i < nspots]
        else:
            keep = list(range(nspots))

        self.data_array = sweeps[keep, :]
        self.protoDirs = [f"{i:03d}" for i in keep]
        self.trace_index = list(range(len(keep)))
        self.trace_important = [True] * len(keep)
        self.nprotodirs = nspots
        self.trace_StartTimes = np.zeros(len(keep))
        # a map has no per-sweep clamp command; command levels are 0
        self.values = np.zeros(len(keep))
        self.commandLevels = np.zeros(len(keep))
        self.holding = clampstate.get("holding", 0.0)  # from configurable defaults

        info = [
            {"name": "Command", "units": cmd_units, "values": np.array(self.values)},
            {"name": "Time", "units": "s", "values": self.time_base},
            {
                "ClampState": clampstate,
                "Protocol": {"recordState": True, "mode": mode},
                "DAQ": clampstate["DAQ"],
                "startTime": self._xsg_timestamp(xsg),
            },
        ]
        self.traces = EM.MetaArray(self.data_array, info=info)
        self.cmd_wave = EM.MetaArray(
            np.zeros_like(self.data_array),
            info=[
                {"name": "Command", "units": cmd_units, "values": np.array(self.values)},
                {"name": "Time", "units": "s", "values": self.time_base},
                info[2],
            ],
        )
        self.WCComp = self.parseClampWCCompSettings([None, {"ClampState": clampstate}])
        self.CCComp = self.parseClampCCCompSettings([None, {"ClampState": clampstate}])

        # stimulus timing (from the ephys pulseParameters)
        self.tstart = 0.0
        self.tend = float(np.max(self.time_base))
        stim = self._pulse_from_params(ephys.get("pulseParameters", {}))
        if stim is not None and len(stim.get("start", [])) > 0:
            self.tstart = float(stim["start"][0])
            self.tend = self.tstart + float(stim["duration"][0])
        self.repetitions = 1
        return True

    def getClampCommand(self, data, generateEmpty: bool = True):
        if hasattr(data, "hasColumn") and data.hasColumn("Channel", "Command"):
            return data["Channel":"Command"]
        if generateEmpty:
            tVals = self.time_base
            units = "V" if (self.mode and "v" in self.mode.lower()) else "A"
            return EM.MetaArray(
                np.zeros(tVals.shape),
                info=[{"name": "Time", "values": tVals, "units": "s"}, {"units": units}],
            )
        return None

    def plotClampData(self, all=True):
        import matplotlib.pyplot as mpl

        f, ax = mpl.subplots(2)
        if all:
            for i in range(len(self.data_array)):
                ax[0].plot(self.time_base, self.data_array[i])
                ax[1].plot(self.time_base, self.cmd_wave[i])
        else:
            ax[0].plot(self.time_base, np.array(self.data_array).mean(axis=0))
        mpl.show()

    # -----------------------------------------------------------------------
    # stimulus / light source timing
    # -----------------------------------------------------------------------
    def _pulse_from_params(self, params: dict) -> Optional[dict]:
        """Translate an ephus squarePulseTrain pulseParameters struct into the
        acq4_reader _getPulses dict shape."""
        if not isinstance(params, dict) or "type" not in params:
            return None
        delay = _as_float(params.get("squarePulseTrainDelay", 0.0))
        width = _as_float(params.get("squarePulseTrainWidth", 0.0))
        isi = _as_float(params.get("squarePulseTrainISI", 0.0))
        n = int(_as_float(params.get("squarePulseTrainNumber", 1), 1))
        amp = _as_float(params.get("amplitude", 0.0))
        times = {
            "start": [delay + k * isi for k in range(max(n, 1))],
            "duration": [width] * max(n, 1),
            "amplitude": [amp] * max(n, 1),
            "period": [isi] * max(n, 1),
            "type": _as_str(params.get("type", "squarePulseTrain")),
            "npulses": max(n, 1),
        }
        return times

    def getStim(self, stimname: str = "Stim") -> dict:
        """Return the amplifier COMMAND pulse (the VC/CC test pulse) from
        ``header.ephys.pulseParameters``.  Note: this is the seal-test / access
        monitoring pulse, NOT the photostimulus - for the laser timing use
        :meth:`laser_pulse` / :meth:`getLaserBlueTimes`."""
        xsg = self.load_xsg(self.protocol)
        if xsg is None:
            return {}
        ephys = self._hdr_section(xsg, "ephys")
        stim = self._pulse_from_params(ephys.get("pulseParameters", {}))
        return stim if stim is not None else {}

    def _stimulator_pulse(self, xsg: dict, channel_name: str) -> Optional[dict]:
        """Return the pulse dict for a named stimulator channel (galvo mirrors,
        shutter, siu, ...).  The stimulator holds parallel ``channels`` and
        ``pulseParameters`` lists."""
        st = self._hdr_section(xsg, "stimulator")
        names = [_as_str(c.get("channelName", "")) for c in _as_list(st.get("channels"))]
        params = _as_list(st.get("pulseParameters"))
        for nm, pp in zip(names, params):
            if nm == channel_name and isinstance(pp, dict):
                return self._pulse_from_params(pp)
        return None

    def laser_pulse(self, mapdir: Union[str, Path, None] = None) -> Optional[dict]:
        """Return the actual photostimulus (laser) timing for a map.

        The laser is gated by the shutter channel named in
        ``header.mapper.shutterChannels`` (typically ``shutter0``); its
        squarePulseTrain delay is the laser onset within each sweep (0.098 s in
        the sample data).  This is the meaningful stimulus time for LSPS, and it
        is present even on maps that have no amplifier command (test) pulse.
        """
        if mapdir is None:
            mapdir = self.protocol
        xsg = self.load_xsg(mapdir)
        if xsg is None:
            return None
        mapper = self._hdr_section(xsg, "mapper")
        shutter = _as_str(mapper.get("shutterChannels", "")) or "shutter0"
        pulse = self._stimulator_pulse(xsg, shutter)
        if pulse is None:
            pulse = self._stimulator_pulse(xsg, "shutter0")
        return pulse

    def getLaserBlueTimes(self) -> bool:
        """Fill ``self.LaserBlueTimes`` with the laser (shutter) pulse timing."""
        self.LaserBlueTimes = self.laser_pulse(self.protocol)
        return self.LaserBlueTimes is not None

    def getLEDCommand(self) -> bool:
        return False  # ephus LSPS uses a laser+galvo, not an LED command channel

    def _getPulses(self, stimuli: dict) -> dict:
        return stimuli

    def _getWaveGeneratorWidget(self, parent_device: str = "", channels: str = None, device: str = None):
        return self.getStim()

    def getLaserBlueShutter(self) -> dict:
        return self.getStim()

    # -----------------------------------------------------------------------
    # device data (photodiode)
    # -----------------------------------------------------------------------
    def getPhotodiode(self) -> bool:
        xsg = self.load_xsg(self.protocol)
        if xsg is None:
            return False
        pd = self._acquirer_trace(xsg)
        if pd is None:
            return False
        # the acquirer has its OWN sample rate / trace length and can hold a
        # different number of sweeps than the ephys channel (e.g. pk0007 stores
        # only one acquirer sweep), so use the acquirer header, not the ephys one
        acq = self._hdr_section(xsg, "acquirer")
        sr = _as_float(acq.get("sampleRate", 10000.0), 10000.0)
        tl = _as_float(acq.get("traceLength", 0.0))
        npts = int(round(sr * tl)) if tl > 0 else pd.size
        if npts <= 0 or pd.size % npts != 0:
            npts = pd.size  # fall back to a single sweep
        nsw = max(1, pd.size // npts)
        self.Photodiode = pd[: nsw * npts].reshape(nsw, npts)
        self.Photodiode_time_base = np.tile(np.arange(npts) / sr, (nsw, 1))
        self.Photodiode_sample_rate = np.ones(nsw) * sr
        return True

    def getDeviceData(self, device="Photodiode", devicename="Photodiode", allow_partial=False):
        if device.lower().startswith("photodiode"):
            if not self.getPhotodiode():
                return None
            return {
                "data": self.Photodiode,
                "time_base": self.Photodiode_time_base,
                "sample_rate": self.Photodiode_sample_rate,
            }
        return None

    def getLaserBlueCommand(self) -> bool:
        """No separate raw laser command waveform is stored per spot in ephus."""
        return False

    # -----------------------------------------------------------------------
    # scanner positions / geometry
    # -----------------------------------------------------------------------
    def spot_size(self, mapdir: Union[str, Path]) -> float:
        """Nominal spot size in meters (mapper does not store it directly; use
        the smaller grid spacing as a stand-in)."""
        xsg = self.load_xsg(mapdir)
        if xsg is None:
            return 0.0
        mapper = self._hdr_section(xsg, "mapper")
        xs = _as_float(mapper.get("xSpacing", 0.0))
        ys = _as_float(mapper.get("ySpacing", 0.0))
        s = min([v for v in (xs, ys) if v > 0], default=0.0)
        return s * 1e-6  # um -> m

    def map_grid_um(self, mapdir: Union[str, Path]):
        """Return per-acquisition-sweep (x, y) map positions in microns.

        Construction matches the lab's ``lsps_readfile.m`` (verified to
        reproduce the reference ``lsps_result.mat`` pixel coordinates exactly):

            mp = mapPatternArray'                     (transposed, as in lsps)
            mapx(r, c) = (r - nrows/2) * xSpacing     (r, c are 1-based)
            mapy(r, c) = (ncols/2 - c) * ySpacing
            loc_x = mapx*cos(rot) - mapy*sin(rot) + xPatternOffset
            loc_y = mapx*sin(rot) + mapy*cos(rot) + yPatternOffset

        ``mp`` holds the acquisition order at each grid cell, so sweep k
        (0-based) is placed at the cell where ``mp == k+1``.  Returned microns
        are in the galvo frame (x right, y up).  Cells not described by the
        pattern (e.g. non-map protocols) are NaN.
        """
        xsg = self.load_xsg(mapdir)
        if xsg is None:
            return np.zeros((0, 2))
        mapper = self._hdr_section(xsg, "mapper")
        pattern = np.asarray(mapper.get("mapPatternArray"))
        nspots = self._sweep_shape(xsg)[0]
        if pattern.ndim != 2 or nspots == 0:
            return np.zeros((nspots, 2))
        mp = pattern.T  # lsps_readfile transposes the pattern
        nrows, ncols = mp.shape
        xs = _as_float(mapper.get("xSpacing", 0.0))
        ys = _as_float(mapper.get("ySpacing", 0.0))
        xoff = _as_float(mapper.get("xPatternOffset", 0.0))
        yoff = _as_float(mapper.get("yPatternOffset", 0.0))
        rot = np.deg2rad(_as_float(mapper.get("spatialRotation", 0.0)))
        cos, sin = np.cos(rot), np.sin(rot)
        positions = np.full((nspots, 2), np.nan)
        for r in range(1, nrows + 1):
            for c in range(1, ncols + 1):
                order = int(mp[r - 1, c - 1])  # 1-based acquisition index
                if order < 1 or order > nspots:
                    continue
                mapx = (r - nrows / 2.0) * xs
                mapy = (ncols / 2.0 - c) * ys
                loc_x = mapx * cos - mapy * sin + xoff
                loc_y = mapx * sin + mapy * cos + yoff
                positions[order - 1] = [loc_x, loc_y]
        return positions

    def scanner_target_list(self, mapdir: Union[str, Path]) -> list:
        """Scanner targets (one [x, y] per sweep) in meters, acquisition order.
        Unknown positions (NaN) are reported as [0, 0]."""
        um = np.nan_to_num(self.map_grid_um(mapdir), nan=0.0)
        return (um * 1e-6).tolist()

    def getScannerPositions(self, dataname: str = "Laser-Blue-raw.ma") -> bool:
        mapdir = Path(self.protocol)
        um = self.map_grid_um(mapdir)
        n = um.shape[0]
        self.scanner_positions = np.nan_to_num(um, nan=0.0) * 1e-6  # meters, acq order
        self.scanner_spotsize = self.spot_size(mapdir)
        self.scanner_camera = {}
        self.scanner_info = {}
        self.scanner_targets = [[]] * n
        self.scanner_sequenceparams = {
            "sequence1": {"index": [0]},
            "sequence2": {"index": n},
        }
        for i in range(n):
            self.scanner_info[(0, i)] = {
                "directory": f"{i:03d}",
                "rep": 0,
                "pos": self.scanner_positions[i],
            }
        return n > 0

    # -----------------------------------------------------------------------
    # images
    # -----------------------------------------------------------------------
    def imaging_calibration(self, mapdir: Union[str, Path]) -> dict:
        """Return the imaging-system field-of-view calibration from the xsg
        ``imagingSys`` header: full FOV size in microns and axis inversion
        flags.  Used to place the map grid onto the video image when there is
        no reference ``lsps_result.mat`` to supply pixel coordinates."""
        xsg = self.load_xsg(mapdir)
        if xsg is None:
            return {}
        im = self._hdr_section(xsg, "imagingSys")
        return {
            "xMicrons": _as_float(im.get("xMicrons", 0.0)),
            "yMicrons": _as_float(im.get("yMicrons", 0.0)),
            "xOffset": _as_float(im.get("xOffset", 0.0)),
            "yOffset": _as_float(im.get("yOffset", 0.0)),
            "xInvert": int(_as_float(im.get("xInvert", 0.0))),
            "yInvert": int(_as_float(im.get("yInvert", 0.0))),
        }

    def _imscale(self, mapdir, image_shape):
        """pixels-per-micron (x, y) from the imagingSys FOV calibration.

        Note the x/y name swap between the imaging header and the image axes:
        ``imscalex = imageWidth / yMicrons``, ``imscaley = imageHeight / xMicrons``
        (verified against the reference; see lsps_readfile.m)."""
        H, W = int(image_shape[0]), int(image_shape[1])
        cal = self.imaging_calibration(mapdir)
        imscalex = (W / cal["yMicrons"]) if cal.get("yMicrons") else 1.0
        imscaley = (H / cal["xMicrons"]) if cal.get("xMicrons") else 1.0
        return imscalex, imscaley

    def map_pixels(
        self,
        mapdir: Union[str, Path],
        image_shape,
        center_px=None,
    ) -> np.ndarray:
        """Return the map's spot positions in video-image pixel coordinates.

        Replicates the lab's ``lsps_readfile.m`` mapping (verified to reproduce
        the reference ``lsps_result.mat`` ``map_locationx/y`` exactly)::

            px = imageWidth/2  + loc_x * imscalex - X_Offset
            py = imageHeight/2 - loc_y * imscaley - Y_Offset

        where ``loc_x/loc_y`` are the per-sweep microns from :meth:`map_grid_um`,
        the y-axis is inverted (image rows increase downward), and
        ``X_Offset``/``Y_Offset`` are the configurable rig alignment shift
        (``map_x_offset_px``/``map_y_offset_px``).

        With ``center_px`` given (a user click) the grid is instead centered on
        that pixel (its geometric centroid moved to the click), preserving scale
        and orientation - the reliable way to place maps whose soma was not
        registered.
        """
        H, W = int(image_shape[0]), int(image_shape[1])
        imscalex, imscaley = self._imscale(mapdir, image_shape)
        um = self.map_grid_um(mapdir)
        if center_px is None:
            xo = _as_float(self.defaults.get("map_x_offset_px", 0.0))
            yo = _as_float(self.defaults.get("map_y_offset_px", 0.0))
            px = W / 2.0 + np.nan_to_num(um[:, 0]) * imscalex - xo
            py = H / 2.0 - np.nan_to_num(um[:, 1]) * imscaley - yo
        else:
            umc = np.nan_to_num(um - np.nanmean(um, axis=0), nan=0.0)
            px = float(center_px[0]) + umc[:, 0] * imscalex
            py = float(center_px[1]) - umc[:, 1] * imscaley
        return np.column_stack([px, py])

    def soma_pixels(self, mapdir: Union[str, Path], image_shape):
        """Return the registered soma location (image pixels), or None.

        The soma marker comes from ``mapper.soma1Coordinates`` (set manually via
        the lab's image_register.m).  When it is empty the soma was never
        registered - lsps_readfile.m would default it to the image center, which
        is not meaningful, so we return None instead."""
        xsg = self.load_xsg(mapdir)
        if xsg is None:
            return None
        mapper = self._hdr_section(xsg, "mapper")
        soma = np.asarray(mapper.get("soma1Coordinates"))
        if soma.size < 2:
            return None
        H, W = int(image_shape[0]), int(image_shape[1])
        imscalex, imscaley = self._imscale(mapdir, image_shape)
        sx = W / 2.0 + float(soma.ravel()[0]) * imscalex
        sy = H / 2.0 - float(soma.ravel()[1]) * imscaley
        return (sx, sy)

    def find_video_image(self, celldir: Union[str, Path]) -> Optional[Path]:
        """Return the video image tif for a cell directory, if present."""
        celldir = Path(celldir)
        for sub in (celldir / "images", celldir):
            if sub.is_dir():
                tifs = sorted(sub.glob("videoImg_*.tif")) or sorted(sub.glob("*.tif"))
                if tifs:
                    return tifs[0]
        return None

    def getImage(self, filename: Union[str, Path, None] = None) -> np.ndarray:
        """Read a tif image and set Image_pos/scale/region attributes.

        For an ephus cell, the position/scale come from the imaging-system
        header of a map's xsg (microns-per-pixel field of view) rather than an
        acq4 transform.  ``filename`` may be a tif path or a cell directory.
        """
        assert filename is not None
        filename = Path(filename)
        if filename.is_dir():
            fil1 = self.find_video_image(filename)
            if fil1 is None:
                raise FileNotFoundError(f"No video image found under {str(filename):s}")
            filename = fil1
        self.imageData = tf.imread(str(filename))
        self.Image_filename = filename.name
        self.Image_pos = (0.0, 0.0, 0.0)
        self.Image_scale = (1.0, 1.0)
        self.Image_region = [0, 0, self.imageData.shape[-1], self.imageData.shape[-2]]
        self.Image_binning = (1, 1)
        return self.imageData

    def getAverageScannerImages(self, dataname: str = "Camera/frames.ma", mode: str = "average", **kwargs):
        """Ephus stores no per-spot camera frames; use the cell video image."""
        CP.cprint("y", "ephus_reader.getAverageScannerImages: no per-spot frames; returning video image")
        celldir = Path(self.protocol).parent
        img = self.find_video_image(celldir)
        if img is None:
            return None
        return tf.imread(str(img))

    # -----------------------------------------------------------------------
    # reference results (old lsps_result.mat) - convenience, not acq4 parity
    # -----------------------------------------------------------------------
    def load_reference_result(self, celldir: Union[str, Path]) -> Optional[dict]:
        """Load an ``lsps_result.mat`` (old analysis output) if present, for
        validation / overlay.  Returns the ``result`` struct as nested dicts."""
        celldir = Path(celldir)
        rf = celldir / "lsps_result.mat"
        if not rf.is_file():
            return None
        mat = sio.loadmat(str(rf), struct_as_record=False, squeeze_me=True)
        return mat2dict(mat["result"])
