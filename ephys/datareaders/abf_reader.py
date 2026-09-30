import matplotlib.pyplot as mpl
from pathlib import Path
import pyabf

# Claude added 2026-09-29: acq4_reader-API-compatible reader class (see the
# ``abf_reader`` class below).  The pre-existing plot_one_run/test_abf_reader
# demo functions are left intact.
import datetime
import pprint
from typing import List, Optional, Union

import numpy as np
import pylibrary.tools.cprint as CP
import MetaArray as EM

pp = pprint.PrettyPrinter(indent=4)

# unit strings -> multiplier to SI (Amperes or Volts)
_UNIT_SCALE = {
    "a": 1.0,
    "ma": 1e-3,
    "ua": 1e-6,
    "µa": 1e-6,
    "na": 1e-9,
    "pa": 1e-12,
    "v": 1.0,
    "mv": 1e-3,
    "uv": 1e-6,
    "µv": 1e-6,
}


def _unit_scale(units: str) -> float:
    """Multiplier that converts a pyabf unit string (e.g. 'pA', 'mV') to SI
    (Amperes or Volts).  Unknown units default to 1.0."""
    if units is None:
        return 1.0
    return _UNIT_SCALE.get(str(units).strip().lower(), 1.0)


def _is_current_unit(units: str) -> bool:
    """True if the unit string denotes a current (ends in 'A'): the recorded
    signal is a current, i.e. the amplifier is in voltage clamp."""
    return str(units).strip().lower().endswith("a")


basepath = Path("/Volumes/PBM_007/Example_D-stellate_neuron")
abf_filename = Path(basepath, "20210112a", "2021_01_12_0008.abf")

# print(abf.headerText)

def plot_one_run(subfig, abf, fn):

    nSweeps = abf.sweepCount
    ax = subfig.subplots(2, 1, sharex=True,  gridspec_kw={
                           'width_ratios': [1],
                           'height_ratios': [0.8, 0.2],
                       'wspace': 0.1,
                       'hspace': 0.0}, squeeze=True)
    mpl.subplots_adjust(left=0.2, bottom=0.15, right=0.9, top=0.85)
    ax[0].set_title(str(fn.name), fontsize=9)

    for i in range(nSweeps):
        print(f"Sweep = {i:d}")
        abf.setSweep(i)
        ax[0].plot(abf.sweepX, abf.sweepY, linewidth=0.5)
        ax[1].plot(abf.sweepX, abf.sweepC, linewidth=0.3)
    ax[1].set_ylim((-100, 20))
    for i in range(2):
        ax[i].spines['right'].set_visible(False)
        ax[i].spines['top'].set_visible(False)

def test_abf_reader():
    ephys_path = Path(basepath, "20210112a")
    files = list(ephys_path.glob("*.abf"))
    abf = pyabf.ABF(abf_filename)
    nf = len(files)
    fig= mpl.figure(figsize=(8, 6))
    subfigs = fig.subfigures(3,3) # , wspace=0.2, hspace=0.15)
    subfigs = subfigs.ravel()
    for i, fn in enumerate(files):
        abf = pyabf.ABF(fn)
        plot_one_run(subfigs[i], abf, fn)
    mpl.show()

# ===========================================================================
# Claude added 2026-09-29: abf_reader -- an Axon ABF reader that mirrors the
# acq4_reader public API (duck-typed drop-in), so that data_summary.py and the
# ephys_analysis pipeline can consume .abf files.  Uses pyabf for file access.
#
# Protocol model: one .abf FILE == one acq4 "protocol"; the sweeps inside the
# file are the traces.  Synthetic per-sweep subdirs (<file>/000, /001, ...) and
# a synthesized acq4-style .index dict provide parity with the directory-based
# acq4 layout.  Values ABF does not record cleanly through pyabf's high-level
# API (the recording Bessel LPF) come from configurable defaults; the holding
# level IS read from the file (holdingCommand).
# ===========================================================================
class abf_reader:
    """Read an Axon ``.abf`` file through the acq4_reader interface."""

    #: values not exposed by pyabf's high-level API -> configurable defaults.
    DEFAULTS = {
        "LPF": 10000.0,  # Hz, primary-signal Bessel low-pass (telegraph not read)
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
            dataname = "MultiClamp1.ma"  # accepted for parity
        self.setDataName(dataname)
        self.clampInfo: dict = {}
        self.lb = "\n"

        # known clamp devices (mirrors acq4_reader so callers recognize the
        # synthesized device names)
        maxclamps = 4
        clamps = []
        self.clampdevices = []
        for nc in range(maxclamps):
            cno = nc + 1
            cname = "Clamp%d" % cno
            mcname = "MultiClamp%d" % cno
            clamps.extend([(cname, "Pulse_amplitude"), (mcname, "Pulse_amplitude")])
            self.clampdevices.extend([cname, mcname])
        clamps.extend([("AxoPatch200", "Pulse_amplitude"), ("AxoProbe", "Pulse_amplitude")])
        self.clampdevices.extend(["AxoPatch200", "AxoProbe"])
        self.clamps = clamps
        self.abf_device = "MultiClamp1"  # abf sweeps always land on this name

        self.tstamp = None
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
        self.mode = None
        self._abf_cache: dict = {}
        self.defaults = dict(self.DEFAULTS)
        if defaults:
            self.defaults.update(defaults)

    # -- context manager -----------------------------------------------------
    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        for attr in (
            "time_base", "traces", "data_array", "cmd_wave", "values",
            "clamps", "sample_rate", "clampInfo", "clampdevices", "_abf_cache",
        ):
            if hasattr(self, attr):
                delattr(self, attr)
        import gc

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
        self.defaults.update(kwargs)

    def setDataName(self, dataname: Union[str, Path]) -> None:
        self.dataname = dataname
        self.shortdname = str(Path(self.dataname).stem)

    # -----------------------------------------------------------------------
    # low-level file access
    # -----------------------------------------------------------------------
    def _resolve_abf_path(self, path: Union[str, Path, None]) -> Optional[Path]:
        """Resolve a caller-supplied path to the underlying ``.abf`` file.

        Accepts the .abf file itself, a synthetic per-sweep subdir
        (``<file.abf>/000``), or a directory containing exactly one .abf.
        """
        if path is None:
            return None
        p = Path(path)
        if p.suffix.lower() == ".abf":
            return p if p.is_file() else None
        # synthetic sweep dir: <something.abf>/NNN
        if p.name.isdigit() and p.parent.suffix.lower() == ".abf":
            return p.parent if p.parent.is_file() else None
        if p.is_dir():
            abfs = sorted(p.glob("*.abf"))
            if len(abfs) == 1:
                return abfs[0]
            if len(abfs) > 1:
                CP.cprint(
                    "y",
                    f"abf_reader: {len(abfs):d} .abf files in {str(p):s}; a directory "
                    "is ambiguous as a single protocol -- point setProtocol at one .abf file.",
                )
            return None
        return None

    def _open(self, path: Union[str, Path, None] = None):
        """Return a cached ``pyabf.ABF`` for the protocol path, or None."""
        abf_path = self._resolve_abf_path(self.protocol if path is None else path)
        if abf_path is None:
            return None
        key = str(abf_path)
        if key not in self._abf_cache:
            try:
                self._abf_cache[key] = pyabf.ABF(str(abf_path))
            except Exception as exc:  # noqa: BLE001 - surface as reader error
                self.error_info = f"pyabf failed to open {key}: {exc}"
                CP.cprint("r", f"abf_reader: {self.error_info}")
                return None
        return self._abf_cache[key]

    def _is_sweep_dir(self, p: Path):
        """(abf_file, sweep_index) if ``p`` is a synthetic sweep dir, else (None, None)."""
        p = Path(p)
        if p.name.isdigit() and p.parent.suffix.lower() == ".abf":
            return p.parent, int(p.name)
        return None, None

    def _mode_and_units(self, abf):
        """Return (mode, primary_units, primary_scale, cmd_units, cmd_scale).

        Mode is taken from the recorded (ADC) signal's units when they are
        readable (a current unit -> VC, a voltage unit -> IC).  Some ABF files
        have a corrupt/blank ADC-units field (e.g. '\\x00\\x00\\x00\\x00'); in
        that case fall back to the command (DAC) units, which are unambiguous
        (you command a voltage in VC, a current in IC).  When the ADC unit is
        unreadable, assume the conventional primary unit for the mode (pA for
        VC, mV for IC) so SI scaling stays correct.
        """
        ch = 0
        adc_units = abf.adcUnits[ch] if abf.adcUnits else ""
        dac_units = abf.dacUnits[ch] if (abf.dacUnits and ch < len(abf.dacUnits)) else ""
        au = str(adc_units).strip().lower()
        du = str(dac_units).strip().lower()
        adc_known = au in _UNIT_SCALE
        dac_known = du in _UNIT_SCALE
        if adc_known:
            mode = "VC" if _is_current_unit(adc_units) else "IC"
        elif dac_known:
            # command voltage -> voltage clamp; command current -> current clamp
            mode = "IC" if _is_current_unit(dac_units) else "VC"
        else:
            mode = "IC"  # last-resort default
        if adc_known:
            primary_units, pscale = adc_units, _unit_scale(adc_units)
        else:
            primary_units = "pA" if mode == "VC" else "mV"
            pscale = _unit_scale(primary_units)
        if dac_known:
            cmd_units, cscale = dac_units, _unit_scale(dac_units)
        else:
            cmd_units = "mV" if mode == "VC" else "pA"
            cscale = _unit_scale(cmd_units)
        return mode, primary_units, pscale, cmd_units, cscale

    def _abf_timestamp(self, abf) -> float:
        try:
            return abf.abfDateTime.timestamp()
        except Exception:  # noqa: BLE001
            return 0.0

    @staticmethod
    def _command_level(C: np.ndarray) -> float:
        """Best-effort per-sweep scalar command level: the median of the central
        third of the command waveform (approximates the step amplitude for
        episodic protocols and the holding level for gap-free records)."""
        if C is None or len(C) == 0:
            return 0.0
        n = len(C)
        a, b = n // 3, 2 * n // 3
        seg = C[a:b] if b > a else C
        return float(np.median(seg))

    # -----------------------------------------------------------------------
    # clamp-state / info (mirrors acq4_reader return shapes)
    # -----------------------------------------------------------------------
    def _clampstate(self, abf) -> dict:
        mode, primary_units, _pscale, cmd_units, cmd_scale = self._mode_and_units(abf)
        sr = float(abf.sampleRate)
        lpf = float(self.defaults.get("LPF", 10000.0))
        holding = 0.0
        try:
            hv = float(abf.holdingCommand[0])
            holding = hv * cmd_scale if np.isfinite(hv) else 0.0
        except Exception:  # noqa: BLE001
            holding = 0.0
        primary_signal = "Membrane Current" if mode == "VC" else "Membrane Potential"
        secondary_signal = "Command Voltage" if mode == "VC" else "Command Current"
        clampparams = {
            "OutputZeroEnable": 0, "PipetteOffset": 0.0, "Holding": holding,
            "PrimarySignalHPF": 0.0, "BridgeBalResist": 0.0, "PrimarySignalLPF": lpf,
            "RsCompBandwidth": 0.0, "WholeCellCompResist": 0.0, "WholeCellCompEnable": 0,
            "LeakSubResist": 0.0, "HoldingEnable": int(holding != 0.0), "FastCompTau": 0.0,
            "SlowCompCap": 0.0, "WholeCellCompCap": 0.0, "LeakSubEnable": 0,
            "NeutralizationCap": 0.0, "BridgeBalEnable": 0, "RsCompCorrection": 0.0,
            "NeutralizationEnable": 0, "RsCompEnable": 0, "OutputZeroAmplitude": 0.0,
            "FastCompCap": 0.0, "SlowCompTau": 0.0,
        }
        return {
            "mode": mode,
            "primarySignal": primary_signal,
            "secondarySignal": secondary_signal,
            "primaryUnits": primary_units,
            "secondaryUnits": cmd_units,
            "primaryGain": 1.0,
            "secondaryGain": 1.0,
            "holding": holding,
            "ClampParams": clampparams,
            "DAQ": {
                "primary": {"rate": sr, "numPts": abf.sweepPointCount, "startTime": 0.0},
                "command": {"rate": sr, "numPts": abf.sweepPointCount, "startTime": 0.0},
                "secondary": {"rate": sr, "numPts": abf.sweepPointCount, "startTime": 0.0},
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
                "WCCompValid": True, "WCEnabled": par["WholeCellCompEnable"],
                "WCResistance": par["WholeCellCompResist"], "WCCellCap": par["WholeCellCompCap"],
                "RsCompCorrection": par["RsCompCorrection"], "CompEnabled": par["RsCompEnable"],
                "CompCorrection": par["RsCompCorrection"], "CompBW": par["RsCompBandwidth"],
            }
        return {
            "WCCompValid": False, "WCEnable": 0, "WCResistance": 0.0, "WholeCellCap": 0.0,
            "CompEnable": 0, "CompCorrection": 0.0, "CompBW": 50000.0, "RsCompCorrection": 0.0,
        }

    def parseClampCCCompSettings(self, info: list) -> dict:
        if "ClampState" in info[1].keys() and "ClampParams" in info[1]["ClampState"].keys():
            par = info[1]["ClampState"]["ClampParams"]
            return {
                "CCCompValid": True, "CCBridgeEnable": par["BridgeBalEnable"],
                "CCBridgeResistance": par["BridgeBalResist"],
                "CCNeutralizationEnable": par["NeutralizationEnable"],
                "CCNeutralizationCap": par["NeutralizationCap"],
                "CCLPF": par["PrimarySignalLPF"], "CCPipetteOffset": par["PipetteOffset"],
            }
        return {
            "CCCompValid": False, "CCBridgeEnable": 0, "CCBridgeResistance": 0.0,
            "CCNeutralizationEnable": 0.0, "CCNeutralizationCap": 0,
            "CCPipetteOffset": 0.0, "CCLPF": 10000.0,
        }

    def parseClampHoldingLevel(self, info: list):
        try:
            return info[1]["ClampState"]["holding"]
        except (KeyError, IndexError, TypeError):
            return 0.0

    def getDataInfo(self, filename: Union[str, Path, None] = None, silent: bool = False):
        """Return a 2-element info list ([time-axis, clampstate]) without loading
        the trace data.  ``filename`` may be the .abf file, a synthetic sweep
        dir, or a directory containing one .abf."""
        assert filename is not None
        abf = self._open(filename)
        if abf is None:
            if not silent:
                CP.cprint("r", f"abf_reader.getDataInfo: no abf at {str(filename):s}")
            return None
        clampstate = self._clampstate(abf)
        info = [
            {"name": "Time", "units": "s"},
            {"ClampState": clampstate, "startTime": self._abf_timestamp(abf)},
        ]
        self.parseClampInfo(info)
        return info

    # -----------------------------------------------------------------------
    # synthesized directory / index handling
    # -----------------------------------------------------------------------
    def subDirs(self, p: Union[str, Path]) -> List:
        """Return synthetic per-sweep subdirectories for an .abf protocol
        (one per sweep), mirroring acq4_reader.subDirs."""
        p = Path(p)
        if self.last_protocol == p:
            return self.last_dirs
        abf = self._open(p)
        if abf is None:
            self.last_dirs, self.last_protocol = [], p
            return []
        abf_path = self._resolve_abf_path(p)
        dirs = [abf_path / f"{i:03d}" for i in range(abf.sweepCount)]
        self.last_dirs, self.last_protocol = dirs, p
        return dirs

    def _synthesize_protocol_index(self, abf) -> dict:
        mode, primary_units, _ps, _cu, _cs = self._mode_and_units(abf)
        return {
            ".": {
                "dirType": "ProtocolSequence",
                "__timestamp__": self._abf_timestamp(abf),
                "notes": abf.abfFileComment if hasattr(abf, "abfFileComment") else "",
                "description": str(getattr(abf, "protocol", "")),
                "devices": {
                    self.abf_device: {
                        "mode": mode,
                        "holdingCheck": False,
                        "holdingSpin": 0.0,
                        "primaryUnits": primary_units,
                    },
                },
                "sequenceParams": {("protocol", "repetitions"): [0]},
            }
        }

    def _synthesize_sweep_index(self, abf, sweep: int) -> dict:
        return {
            ".": {
                "dirType": "Protocol",
                "__timestamp__": self._abf_timestamp(abf),
                "sweep": sweep,
            }
        }

    def readDirIndex(self, currdir: Union[str, Path, None] = None):
        """Return a synthesized acq4-style index dict for the given directory:
        a synthetic per-sweep dir or an .abf protocol."""
        self._dirindex = None
        if currdir is None:
            currdir = self.protocol
        currdir = Path(currdir)
        abf_path, sweep = self._is_sweep_dir(currdir)
        if abf_path is not None:
            abf = self._open(abf_path)
            self._dirindex = self._synthesize_sweep_index(abf, sweep) if abf else None
            return self._dirindex
        abf = self._open(currdir)
        if abf is not None:
            self._dirindex = self._synthesize_protocol_index(abf)
            return self._dirindex
        CP.cprint("y", f"abf_reader.readDirIndex: no abf data recognized at {str(currdir):s}")
        return None

    def getIndex(self, currdir: Union[str, Path, None] = None, lineend: str = "\n"):
        self.lb = lineend
        self._index = self.readDirIndex(currdir if currdir is not None else self.protocol)
        return self._index["."] if self._index is not None else None

    def _readIndex(self, currdir: Union[str, Path, None] = None):
        self._index = self.readDirIndex(currdir if currdir is not None else self.protocol)
        return self._index

    def convert_timestamp(self, fts) -> str:
        return datetime.datetime.fromtimestamp(fts).strftime("%Y-%m-%d  %H:%M:%S %z")

    def printIndex(self, index):
        print(pp.pformat(index))

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

    def header_metadata(self, filename: Union[str, Path, None] = None) -> dict:
        """Flat dict of file-level metadata (parity with ephus_reader)."""
        abf = self._open(filename if filename is not None else self.protocol)
        if abf is None:
            return {}
        mode, primary_units, _ps, cmd_units, cmd_scale = self._mode_and_units(abf)
        try:
            hv = float(abf.holdingCommand[0])
            holding = hv * cmd_scale if np.isfinite(hv) else 0.0
        except Exception:  # noqa: BLE001
            holding = 0.0
        return {
            "abfID": abf.abfID,
            "protocol": str(getattr(abf, "protocol", "")),
            "datetime": abf.abfDateTime.isoformat() if abf.abfDateTime else "",
            "version": abf.abfVersionString,
            "adcNames": list(abf.adcNames), "adcUnits": list(abf.adcUnits),
            "dacNames": list(abf.dacNames), "dacUnits": list(abf.dacUnits),
            "sampleRate": float(abf.sampleRate),
            "sweepCount": int(abf.sweepCount), "channelCount": int(abf.channelCount),
            "mode": mode, "holding": holding,
            "comment": getattr(abf, "abfFileComment", ""),
        }

    # -----------------------------------------------------------------------
    # protocol checking (minimal parity)
    # -----------------------------------------------------------------------
    def checkProtocol(self, protocolpath: Union[str, Path, None] = None, allow_partial=False) -> bool:
        return self._open(protocolpath if protocolpath is not None else self.protocol) is not None

    def checkProtocolImportantFlags(self, protocolpath: Union[str, Path, None] = None) -> bool:
        return True

    def checkProtocolImportant(self, protocol):
        return True

    def listSequenceParams(self, dh=None):
        return {}

    # -----------------------------------------------------------------------
    # data
    # -----------------------------------------------------------------------
    def getData(
        self,
        pos: int = 1,
        check: bool = False,
        allow_partial: bool = False,
        record_list: list = [],
        downsample: int = 1,
        silent: bool = True,
    ):
        """Read all sweeps of the .abf protocol and fill the same attributes
        acq4_reader.getData fills.  Returns True on success (or, with
        ``check=True``, whether data is present)."""
        if len(record_list) > 0 and not allow_partial:
            raise ValueError("abf_reader.getData: record_list not empty but allow_partial is False")
        self.error_info = None
        abf = self._open(self.protocol)
        if abf is None:
            self.error_info = f"No abf found at {str(self.protocol):s}"
            if check:
                return False
            raise FileNotFoundError(self.error_info)
        if check:
            return True

        mode, primary_units, pscale, cmd_units, cmd_scale = self._mode_and_units(abf)
        self.units = [primary_units, cmd_units]
        sr = float(abf.sampleRate)
        nsweeps = int(abf.sweepCount)
        npts = int(abf.sweepPointCount)

        # which sweeps to return
        if allow_partial and isinstance(record_list, list) and len(record_list) > 0:
            keep = [i for i in record_list if 0 <= i < nsweeps]
        else:
            keep = list(range(nsweeps))

        data = np.empty((len(keep), npts), dtype=float)
        cmd = np.empty((len(keep), npts), dtype=float)
        levels = np.zeros(len(keep))
        for row, sw in enumerate(keep):
            abf.setSweep(sw, channel=0)
            data[row, :] = np.asarray(abf.sweepY, dtype=float)[:npts] * pscale
            C = np.asarray(abf.sweepC, dtype=float)[:npts] * cmd_scale
            cmd[row, :] = C
            levels[row] = self._command_level(C)

        self.mode = mode
        self.sample_interval = 1.0 / sr
        self.sample_rate = np.ones(len(keep)) * sr
        self.time_base = np.arange(npts) / sr
        self.clampInfo["dirs"] = self.subDirs(self.protocol)
        self.clampInfo["missingData"] = []

        clampstate = self._clampstate(abf)
        clampstate["DAQ"]["primary"]["rate"] = sr
        clampstate["DAQ"]["primary"]["numPts"] = npts

        self.data_array = data
        abf_path = self._resolve_abf_path(self.protocol)
        self.protoDirs = [f"{i:03d}" for i in keep]
        self.trace_index = list(range(len(keep)))
        self.trace_important = [True] * len(keep)
        self.nprotodirs = nsweeps
        self.trace_StartTimes = np.zeros(len(keep))
        self.values = levels
        self.commandLevels = levels
        self.holding = clampstate.get("holding", 0.0)

        info = [
            {"name": "Command", "units": cmd_units, "values": np.array(self.values)},
            {"name": "Time", "units": "s", "values": self.time_base},
            {
                "ClampState": clampstate,
                "Protocol": {"recordState": True, "mode": mode},
                "DAQ": clampstate["DAQ"],
                "startTime": self._abf_timestamp(abf),
            },
        ]
        self.traces = EM.MetaArray(self.data_array, info=info)
        self.cmd_wave = EM.MetaArray(
            cmd,
            info=[
                {"name": "Command", "units": cmd_units, "values": np.array(self.values)},
                {"name": "Time", "units": "s", "values": self.time_base},
                info[2],
            ],
        )
        self.WCComp = self.parseClampWCCompSettings([None, {"ClampState": clampstate}])
        self.CCComp = self.parseClampCCCompSettings([None, {"ClampState": clampstate}])

        self.tstart = 0.0
        self.tend = float(np.max(self.time_base)) if npts else 0.0
        self.tdur = self.tend - self.tstart
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
        f, ax = mpl.subplots(2)
        if all:
            for i in range(len(self.data_array)):
                ax[0].plot(self.time_base, self.data_array[i])
                ax[1].plot(self.time_base, self.cmd_wave[i])
        else:
            ax[0].plot(self.time_base, np.array(self.data_array).mean(axis=0))
        mpl.show()

    # -----------------------------------------------------------------------
    # LSPS / imaging: no ABF equivalent -> honest stubs (parity with acq4/ephus)
    # -----------------------------------------------------------------------
    def getStim(self, stimname: str = "Stim") -> dict:
        return {"start": [], "duration": [], "amplitude": [], "npulses": [], "period": []}

    def getLaserBlueTimes(self) -> bool:
        return False

    def getLaserBlueCommand(self) -> bool:
        return False

    def getLaserBlueShutter(self) -> bool:
        return False

    def getLEDCommand(self) -> bool:
        return False

    def getPhotodiode(self) -> bool:
        return False

    def getScannerPositions(self, dataname: str = "Laser-Blue-raw.ma") -> bool:
        return False

    def getImage(self, filename: Union[str, Path, None] = None):
        return None

    def getAverageScannerImages(self, dataname: str = "Camera/frames.ma", mode: str = "average", **kwargs):
        return None


if __name__ == "__main__":
    test_abf_reader()