# -*- coding: utf-8 -*-
"""
Read the 'old' matdata files - File structure used in the Manis lab from
about 2000 to 2015.

The files are matlab files (basically HDF5 in the later versions).
This code specifically reads those files and turns them into a "clamps" structure
as used by acq4 code. This way we can use the same analysis. 

GetClamps does the transformation.

2018-2019, Paul B. Manis.

"""

import datetime
import os
import pprint
from collections import OrderedDict
from pathlib import Path

import matplotlib.pyplot as mpl
import numpy as np
import pylibrary.plotting.plothelpers as PH
import pylibrary.tools.fileselector as FS
from scipy.io import matlab

import MetaArray as MA

"""
Helper functions
"""


def ndinds(shape):
    ind = np.indices(shape)
    while len(ind.shape) > 2:
        ind = ind.reshape(ind.shape[:-2] + (ind.shape[-2] * ind.shape[-1],))
    return [tuple(x) for x in ind.T]


def flatten(data):
    """Flatten highly nested structures returned by loadmat"""
    if isinstance(data, np.ndarray):
        while data.ndim > 0 and data.shape[0] == 1:
            data = data[0]
        if data.ndim == 0 and data.dtype.names is not None:
            data = dict(zip(data.dtype.names, data))
        elif data.ndim > 0:
            if data.dtype.kind == "O":
                for i in ndinds(data.shape):
                    data[i] = flatten(data[i])
            elif data.dtype.names is not None:
                for fname in data.dtype.names:
                    if data.dtype.fields[fname][0].kind == "O":
                        field = data[fname]
                        for i in ndinds(field.shape):
                            field[i] = flatten(field[i])
    if isinstance(data, dict):
        for k, v in data.items():
            data[k] = flatten(v)
    return data


class GetClamps:
    def __init__(self, datac=None, pathtoprotocol=None, dataname=None, defaults=None):
        """
        Set up clamps.
        This class aspires to provide acq4-API parity (path-based construction,
        synthetic index, extra attributes).  Backward compatible: GetClamps(datac)
        with an integer setProtocol still behaves exactly as before.

        Parameters
        ----------
        datac : object (default None)
            The DatacFile instance of the matdatac file.  Optional: when a
            ``pathtoprotocol`` is given instead, the DatacFile is opened
            internally (see :meth:`setProtocol`), giving this class the same
            path-based construction as acq4_reader/ephus_reader.
        pathtoprotocol : str | Path | int (default None)
            A matdatac protocol path (``<file.mat>/<NNN>_<type>``), a bare
            ``<file.mat>`` (first data block), or -- for backward
            compatibility -- an integer item index into ``datac.items``.
        dataname : str (default None)
            Accepted for acq4 parity; the primary (membrane) channel is always
            used regardless of this value.
        defaults : dict (default None)
            Reserved for parity with the other readers (no matdatac-specific
            defaults at present).
        """

        self.datac = datac
        self.protocol = None            # integer item index into datac.items
        self.protocol_path = None       # the caller-supplied protocol path
        self._datac_cache = {}          # matpath (str) -> DatacFile
        if datac is not None:
            try:
                self._datac_cache[str(Path(datac.fname))] = datac
            except Exception:  # noqa: BLE001
                pass
        self.last_protocol = None
        self.last_dirs = []
        self.error_info = None
        self.importantFlag = False
        self.mode = None
        self.lb = "\n"
        self.pre_process_filters = {"LPF": None, "Notch": []}
        self.defaults = dict(defaults) if defaults else {}
        self.dataname = "MultiClamp1.ma"
        self.shortdname = "MultiClamp1"
        self.clampInfo = {"dirs": [], "missingData": []}

        # known clamp devices (mirrors acq4_reader so callers recognize the
        # synthesized device names)
        maxclamps = 4
        clamps = []
        self.clampdevices = []
        for nc in range(maxclamps):
            cno = nc + 1
            clamps.extend([("Clamp%d" % cno, "Pulse_amplitude"),
                           ("MultiClamp%d" % cno, "Pulse_amplitude")])
            self.clampdevices.extend(["Clamp%d" % cno, "MultiClamp%d" % cno])
        clamps.extend([("AxoPatch200", "Pulse_amplitude"), ("AxoProbe", "Pulse_amplitude")])
        self.clampdevices.extend(["AxoPatch200", "AxoProbe"])
        self.clamps = clamps
        self.datac_device = "MultiClamp1"

        if dataname is not None:
            self.setDataName(dataname)
        if pathtoprotocol is not None:
            self.setProtocol(pathtoprotocol)

    def getData(self, pos=1, check=False, allow_partial=False, record_list=[],
                downsample=1, silent=True, chmap=None, debug=False):
        """
        create a Clamp structure for use in SpikeAnalysis and RMTauAnalysis from acq4.
        Fills in the fields that are returned by PatchEPhys getClamps:
        clampInfo['dirs]
        clampInfo['missingData']
        self.time_base
        self.values
        self.traceStartTimes = np.zeros(0)
        self.sequence
        self.clampValues (sequence)
        self.nclamp = len(self.clmapVlaues
        self.repc
        self.nrepc
        self.data_mode
        self.model_mode = False
        self.command_scale_factor
        self.command_units
        self.devicesUsed
        self.clampDevices
        self.holding
        self.clampState
        self.sample_interval
        self.RSeriesUncomp
        self.amplifierSettings['WCCompValid', 'WCEmabled', 'CompEnabled', 'WCSeriesResistance']
        self.cmd_wave
        self.commandLevels (np.array(self.values))
        self.traces = MetaArray(traces, info=info)
        self.tstart
        self.tdur
        self.tend
        self.spikecount = np.zeros(len...) if in vcmode.

        Info from an example data file:
        [{'name': 'Channel', 'cols': [{'units': 'A', 'name': 'Command'}, {'units': 'V', 'name': 'primary'}, {'units': 'A', 'name': 'secondary'}]},
        {'units': 's', 'values': array([ 0.00000000e+00, 2.50000000e-05, 5.00000000e-05, ..., 6.99925000e-01, 6.99950000e-01, 6.99975000e-01]),
        'name': 'Time'}, {'ClampState': {'primaryGain': 10.0, 'ClampParams': {'OutputZeroEnable': 0, 'PipetteOffset': 0.05197399854660034,
        'Holding': -1.525747063413352e-11, 'PrimarySignalHPF': 0.0, 'BridgeBalResist': 20757020.0, 'PrimarySignalLPF': 20000.0, 'RsCompBandwidth':
        8.413395979806202e-42, 'WholeCellCompResist': 8.413395979806202e-42, 'WholeCellCompEnable': 6004, 'LeakSubResist': 8.413395979806202e-42,
        'HoldingEnable': 1, 'FastCompTau': 8.413395979806202e-42, 'SlowCompCap': 8.413395979806202e-42, 'WholeCellCompCap': 8.413395979806202e-42,
        'LeakSubEnable': 6004, 'NeutralizationCap': 1.9578947837994853e-12, 'BridgeBalEnable': 1, 'RsCompCorrection': 8.413395979806202e-42,
        'NeutralizationEnable': 1, 'RsCompEnable': 6004, 'OutputZeroAmplitude': -0.0009990156395360827, 'FastCompCap': 8.413395979806202e-42,
        'SlowCompTau': 8.413395979806202e-42}, 'secondarySignal': 'Command Current', 'secondaryGain': 1.0, 'secondaryScaleFactor': 2e-09,
        'primarySignal': 'Membrane Potential', 'extCmdScale': 4e-10, 'mode': 'IC', 'holding': 0.0, 'primaryUnits': 'V', 'LPFCutoff': 20000.0,
        'secondaryUnits': 'A', 'primaryScaleFactor': 0.1, 'membraneCapacitance': 0.0}, 'Protocol': {'recordState': True, 'secondary': None,
        'primary': None, 'mode': 'IC'}, 'DAQ': {'command': {'numPts': 28000, 'rate': 40000.0, 'type': 'ao', 'startTime': 1296241556.7347913},
        'primary': {'numPts': 28000, 'rate': 40000.0, 'type': 'ai', 'startTime': 1296241556.7347913}, 'secondary': {'numPts': 28000, 'rate':
        40000.0, 'type': 'ai', 'startTime': 1296241556.7347913}}, 'startTime': 1296241556.7347913}]

        )
        """
        item = self.protocol
        if item is None:
            if debug:
                print("No items in protocol")
            return False
        # Claude added 2026-09-29: acq4-parity 'check' flag -- report whether a
        # readable data block is present without loading it.
        if check:
            try:
                return (
                    self.datac is not None
                    and self.datac.items[item].type not in ("HFILE", "NOTE")
                )
            except (IndexError, AttributeError, TypeError):
                return False
        # print('item: ', item)
        # print('len datac items: ', len(self.datac.items))
        protocol = self.datac.items[item].type
        # print(self.datac.items[item])
        # print('protocol: ', protocol)
        if protocol in ["HFILE", "NOTE"]:
            print("MatdatacRead.Getclamps: Protocol is HFILE or NOTE, not data")
            return False
        try:
            rate, recs = self.datac.items[item].data()
        except:
            print("MatdatacRead.Getclamps: item: ", self.datac.items[item])
            if debug:
                print("MatdatacRead.Getclamps: failed to read data")
            # raise ValueError('failed to read rate, recs')
            return False
        rate = (
            self.datac.items[item].dfile["Sample_Rate"]["v"] * 1e-6
        )  # convert to seconds
        self.dfile = self.datac.items[item].dfile
        (ddir, fname) = os.path.split(self.datac.fname)
        points = self.dfile["Points"]["v"]
        nchannels = len(self.dfile["Channels"]["v"])
        dt = nchannels * float(rate)
        if debug:
            print(
                f"   MatdatacRead.Getclamps: nchannels={nchannels:3d}, points={points:8d}, rate={rate:f}, dt={dt:f}"
            )
        data_mode = self.getDataMode()
        self.data_mode = data_mode[0]
        # Claude added 2026-09-29: expose acq4-style clamp mode ('VC'/'IC') and
        # primary/secondary units (acq4_reader sets these via parseClampInfo).
        self.mode = "VC" if str(self.data_mode).lower() in ("vc",) else "IC"
        self.units = ["A", "V"] if self.mode == "VC" else ["V", "A"]
        self.time_base = np.arange(points) * dt
        datac_traces = np.zeros((len(recs), nchannels + 1, points))
        self.recs = recs
        if debug:
            print("   MatdatacRead.Getclamps: rec shape: ", np.array(recs).shape)
        if len(recs) < 1:
            if debug:
                print("MatdatacRead.Getclamps: too few recs")
            return False
        for i in range(len(recs)):
            for j in range(nchannels):
                if j == 2:  # pull the stimulus only
                    # print len(recs[i][j])
                    # print len(np.arange(0, len(recs[i][j])))
                    cmd = np.interp(
                        self.time_base,
                        np.arange(0, len(recs[i][j])) / 1000.0,
                        recs[i][j],
                    )
                    datac_traces[i, j, :] = cmd
                else:  # channels 0 and 1
                    if debug:
                        print("MatdatacRead: Debugging")
                        print("   i,j", i, j)
                        print("   shape: ", datac_traces.shape)
                        print("   ndim: ", datac_traces.ndim)
                        print("   len recs[i]: ", len(recs[i]))
                        print("   len recs[i][j]: ", len(recs[i][j]))
                    if len(recs[i][j]) != datac_traces.shape[2]:
                        return False  # (cannot match, something is wrong... )
                    datac_traces[i, j, :] = recs[i][j]
                    if j == 1:
                        datac_traces[i, j, :] *= 1e-3  # convert to V from mV
        self.traces = np.array(datac_traces)

        # np.array([[x[chmap[i]] for x in recs] for i in range(len(chmap))])  # remap channels
        # self.values = np.array([np.mean(datac_traces[1,i]) for i in range(len(recs))])
        # print self.values
        self.repc = 1
        self.nrepc = 1
        self.model_mode = False
        self.command_scale_factor = 1
        self.command_units = "A"
        self.devicesUsed = None
        self.clampDevices = None
        self.holding = 0.0
        self.amplfierSettings = {
            "WCCompValid": False,
            "WCEnabled": False,
            "CompEnabled": False,
            "WCSeriesResistance": 0.0,
        }
        self.clampState = None
        self.sample_interval = dt  # in seconds
        self.sample_rate = (1.0 / dt) * np.ones(len(recs))
        self.RSeriesUncomp = 0.0
        # self.cmd_wave = np.squeeze(self.traces[:, 0, :]) * 1e-12
        # Claude fixed 2026-09-29: physical channels are fixed (ch0=current in pA,
        # ch1=voltage in mV) in BOTH modes; only the primary(recorded)-vs-command
        # ROLE flips. For IC/CC the command is the injected current (ch0); for VC
        # the command is the holding/step voltage (ch1). The original code always
        # used ch0 as the command, which switched current/voltage for VC data.
        # Keep CC byte-identical; use the voltage channel for the VC command.
        if self.mode == "VC":
            self.cmd_wave = np.squeeze(self.traces[:, 1, :])  # command = voltage (V, already *1e-3)
        else:
            self.cmd_wave = np.squeeze(self.traces[:, 0, :]) * 1e-12  # command = current (A)
        if self.cmd_wave.ndim == 1:
            self.cmd_wave = np.expand_dims(self.cmd_wave, 0)
        self.protoTimes = {
            "drugtestiv": [0.21, 0.5],
            "ap-iv2": [0.01, 0.5],
            "cciv": [0.005, 0.100],
        }  # in seconds
        self.tstart = 0.01
        self.tdur = 0.500
        if protocol in self.protoTimes:
            self.tstart = self.protoTimes[protocol][0]
            self.tdur = self.protoTimes[protocol][1] - self.tstart

        # check how the command wave is doing:
        cmds = np.mean(np.abs(self.cmd_wave), axis=0)
        if isinstance(cmds, float):
            cmds = np.ones_like(self.time_base) * cmds
        try:
            dcmds = np.diff(cmds)
        except:
            print("MatdatacRead: cmds: ", cmds)
            import matplotlib.pyplot as mpl

            mpl.plot(self.time_base[:-1], dcmds)
            mpl.show()
            exit()
        start = np.where(dcmds > 1e-11)[0]
        if len(start) > 1:
            start = start[0]
        end = np.where(dcmds < -1e-11)[0]
        if len(end) > 1:
            end = end[0]
        print(
            "MatdatacRead: shapes: ",
            self.cmd_wave.shape,
            self.time_base.shape,
            cmds.shape,
            self.traces.shape,
        )
        # print('start, end: ', start, end)
        # print(self.time_base[start], self.time_base[end])
        self.tend = self.tstart + self.tdur
        # print('self.tstart, end, dur: ', self.tstart, self.tend, self.tdur)
        t0 = int(self.tstart / dt)
        t1 = int(self.tend / dt)
        self.values = np.mean(self.cmd_wave[:, t0:t1], axis=1)  # express values in amps
        self.commandLevels = self.values
        info = [
            {"units": "A", "values": self.values, "name": "Command"},
            {"name": "Time", "units": "s", "values": self.time_base},
            {
                "ClampState": {
                    "primaryGain": 10.0,
                    "ClampParams": {
                        "OutputZeroEnable": 0,
                        "PipetteOffset": 0.0,
                        "Holding": 0.0,
                        "PrimarySignalHPF": 0.0,
                        "BridgeBalResist": 0.0,
                        "PrimarySignalLPF": 20000.0,
                        "RsCompBandwidth": 0.0,
                        "WholeCellCompResist": 0.0,
                        "WholeCellCompEnable": 6004,
                        "LeakSubResist": 0.0,
                        "HoldingEnable": 1,
                        "FastCompTau": 0.0,
                        "SlowCompCap": 0.0,
                        "WholeCellCompCap": 0.0,
                        "LeakSubEnable": 6004,
                        "NeutralizationCap": 0.0,
                        "BridgeBalEnable": 0,
                        "RsCompCorrection": 0.0,
                        "NeutralizationEnable": 1,
                        "RsCompEnable": 6004,
                        "OutputZeroAmplitude": 0.0,
                        "FastCompCap": 0.0,
                        "SlowCompTau": 0.0,
                    },
                    "secondarySignal": "Command Current",
                    "secondaryGain": 1.0,
                    "secondaryScaleFactor": 2e-09,
                    "primarySignal": "Membrane Potential",
                    "extCmdScale": 4e-10,
                    "mode": "IC",
                    "holding": 0.0,
                    "primaryUnits": "V",
                    "LPFCutoff": 20000.0,
                    "secondaryUnits": "A",
                    "primaryScaleFactor": 0.1,
                    "membraneCapacitance": 0.0,
                },
                "Protocol": {
                    "recordState": True,
                    "secondary": None,
                    "primary": None,
                    "mode": "IC",
                },
                "DAQ": {
                    "command": {
                        "numPts": 10000,
                        "rate": 10000.0,
                        "type": "ao",
                        "startTime": 0.0,
                    },
                    "primary": {
                        "numPts": 10000,
                        "rate": 10000.0,
                        "type": "ai",
                        "startTime": 0.0,
                    },
                    "secondary": {
                        "numPts": 1000,
                        "rate": 10000.0,
                        "type": "ai",
                        "startTime": 0.0,
                    },
                },
                "startTime": 0.0,
            },
        ]
        self.WCComp = self.parseClampWCCompSettings(info)
        self.CCComp = self.parseClampCCCompSettings(info)
        # self.traces = np.squeeze(self.traces[:, 1, :])
        # Claude fixed 2026-09-29 (two fixes):
        #  (1) primary = the RECORDED signal: voltage (ch1, V) for IC/CC, current
        #      (ch0, A) for VC.  The original code always used ch1, which put the
        #      command voltage (not the recorded current) into traces for VC data
        #      -- i.e. current/voltage appeared switched.  CC path is unchanged.
        #  (2) for single-record blocks (e.g. VCmini) np.squeeze collapses traces
        #      to 1-D, which cannot carry the 3-element MetaArray info list; the
        #      expand_dims below restores a leading axis (mirrors the cmd_wave guard).
        if self.mode == "VC":
            self.traces = np.squeeze(self.traces[:, 0, :]) * 1e-12  # recorded current (A)
        else:
            self.traces = np.squeeze(self.traces[:, 1, :])          # recorded voltage (V)
        if self.traces.ndim == 1:
            self.traces = np.expand_dims(self.traces, 0)
        print("MatdatacRead: Shape of traces: ", self.traces.shape)

        self.traces = MA.MetaArray(self.traces, info=info)
        self.cmd_wave = MA.MetaArray(
            self.cmd_wave,
            info=[
                {"name": "Command", "units": "A", "values": np.array(self.values)},
                self.traces.infoCopy("Time"),
                self.traces.infoCopy(-1),
            ],
        )
        self.spikecount = np.zeros(len(recs))
        self.rgnrmp = [0, 0.005]
        # print('getclamps got it all')
        return True

    def getDataMode(self):
        datamode = self.dfile["Data_Mode"]["v"]
        # print "current block data mode : ", datamode
        if datamode in ["cc", "ic", "CC", "IC"]:
            datamode = "cc"
            chmap = [1, 0, 2]
        elif datamode in ["vc", "VC"]:
            datamode = "vc"
            chmap = [0, 1, 2]
        else:
            raise ValueError("Data mode <%s> is not known" % datamode)
        return datamode, chmap

    def setProtocol(self, pathtoprotocol):
        """
        Set the protocol to read.

        ``pathtoprotocol`` may be:
        * an integer item index into ``self.datac.items`` (legacy behavior;
          requires a DatacFile already assigned to ``self.datac``);
        * a matdatac protocol path ``<file.mat>/<NNN>_<type>``;
        * a bare ``<file.mat>`` (selects the first data block);
        * a synthetic per-sweep path ``<file.mat>/<NNN>_<type>/<sss>``.
        """
        # Claude added 2026-09-29: accept a path (block) in addition to the
        # legacy integer item index.
        self.protocol_path = pathtoprotocol
        self.last_protocol = None
        if isinstance(pathtoprotocol, (int, np.integer)):
            self.protocol = int(pathtoprotocol)  # legacy: index into self.datac
            return
        matpath, item, _sweep, _explicit = self._parse_protocol_path(pathtoprotocol)
        if matpath is None:
            self.protocol = None
            return
        self.datac = self._open_datac(matpath)
        self.protocol = item

    def setDataName(self, dataname):
        """
        Set the type (name) of the data metaarray name that will be read
        """
        self.dataname = dataname

    def parseClampWCCompSettings(self, info):
        """
        Given the .index file for this protocol dir, try to parse the
        clamp state and compensation
        """
        d = {}
        if (
            "ClampState" in info[1].keys()
            and "ClampParams" in info[1]["ClampState"].keys()
        ):
            par = info[1]["ClampState"]["ClampParams"]
            d["WCCompValid"] = True
            d["WCEnabled"] = par["WholeCellCompEnable"]
            d["WCResistance"] = par["WholeCellCompResist"]
            d["WCCellCap"] = par["WholeCellCompCap"]
            d["CompEnabled"] = par["RsCompEnable"]
            d["CompCorrection"] = par["RsCompCorrection"]
            d["CompBW"] = par["RsCompBandwidth"]
            return d
        else:
            return {
                "WCCompValid": False,
                "WCEnable": 0,
                "WCResistance": 0.0,
                "WholeCellCap": 0.0,
                "CompEnable": 0,
                "CompCorrection": 0.0,
                "CompBW": 50000.0,
            }

    def parseClampCCCompSettings(self, info):
        d = {}
        if (
            "ClampState" in info[1].keys()
            and "ClampParams" in info[1]["ClampState"].keys()
        ):
            par = info[1]["ClampState"]["ClampParams"]
            d["CCCompValid"] = True
            d["CCBridgeEnable"] = par["BridgeBalEnable"]
            d["CCBridgeResistance"] = par["BridgeBalResist"]
            d["CCNeutralizationEnable"] = par["NeutralizationEnable"]
            d["CCNeutralizationCap"] = par["NeutralizationCap"]
            d["CCLPF"] = par["PrimarySignalLPF"]
            d["CCPipetteOffset"] = par["PipetteOffset"]
            return d
        else:
            return {
                "CCCompValid": False,
                "CCBridgeEnable": 0,
                "CCBridgeResistance": 0.0,
                "CCNeutralizationEnable": 0.0,
                "CCNeutralizationCap": 0,
                "CCPipetteOffset": 0.0,
                "CCLPF": 10000.0,
            }

    # =======================================================================
    # A matdatac ".mat" file
    # is treated as a CELL that contains N data BLOCKS (protocols); each block
    # is addressed by a synthetic path "<file.mat>/<NNN>_<type>", and its sweeps
    # by "<file.mat>/<NNN>_<type>/<sss>".  These methods let data_summary.py and
    # the ephys_analysis pipeline drive the reader through the same interface as
    # acq4_reader / ephus_reader.  Existing behavior (GetClamps(datac) + integer
    # setProtocol + getData) is unchanged.
    # =======================================================================
    def setImportant(self, flag: bool = False) -> None:
        self.importantFlag = flag

    def set_pre_process(self, LPF=None, Notch=None):
        self.pre_process_filters["LPF"] = LPF
        self.pre_process_filters["Notch"] = Notch

    def set_defaults(self, **kwargs) -> None:
        self.defaults.update(kwargs)

    # -- low-level resolution -----------------------------------------------
    def _open_datac(self, matpath):
        """Open (and cache) a DatacFile for a .mat path, or return None."""
        if matpath is None:
            return None
        key = str(Path(matpath))
        if key not in self._datac_cache:
            try:
                self._datac_cache[key] = DatacFile(key)
            except Exception as exc:  # noqa: BLE001
                self.error_info = f"Failed to open matdatac file {key}: {exc}"
                print(self.error_info)
                return None
        return self._datac_cache[key]

    def data_block_indices(self, matpath=None):
        """Item indices of the data blocks (protocols) in a matdatac file
        (everything that is not an HFILE or NOTE)."""
        datac = self._open_datac(matpath) if matpath is not None else self.datac
        if datac is None:
            return []
        return [k for k, it in enumerate(datac.items) if it.type not in ("HFILE", "NOTE")]

    def _block_type(self, datac, item) -> str:
        try:
            return str(datac.items[item].type)
        except (IndexError, AttributeError, TypeError):
            return ""

    def _block_nrecs(self, datac, item) -> int:
        if datac is None or item is None:
            return 0
        try:
            _rate, recs = datac.items[item].data()
            return len(recs)
        except Exception:  # noqa: BLE001
            return 0

    @staticmethod
    def _item_from_token(token: str):
        """Parse the block index from a path token like 'NNN_type' or 'NNN'."""
        head = str(token).split("_", 1)[0]
        try:
            return int(head)
        except ValueError:
            return None

    def _parse_protocol_path(self, path):
        """Resolve a protocol path into (matpath, item_index, sweep, explicit_block).

        ``explicit_block`` is True when the path names a block after the .mat
        component (i.e. a protocol), False for a bare .mat (a cell).
        """
        p = Path(path)
        parts = p.parts
        mat_idx = None
        for i, part in enumerate(parts):
            if part.lower().endswith(".mat"):
                mat_idx = i
                break
        if mat_idx is None:
            return None, None, None, False
        matpath = Path(*parts[: mat_idx + 1])
        rest = parts[mat_idx + 1:]
        item = None
        sweep = None
        explicit = False
        if len(rest) >= 1:
            item = self._item_from_token(rest[0])
            explicit = item is not None
        if len(rest) >= 2:
            try:
                sweep = int(rest[1])
            except ValueError:
                sweep = None
        if item is None:  # bare .mat -> first data block
            blocks = self.data_block_indices(matpath)
            item = blocks[0] if blocks else None
        return matpath, item, sweep, explicit

    # -- clamp-state helpers -------------------------------------------------
    def _clampparams(self, lpf=20000.0):
        return {
            "OutputZeroEnable": 0, "PipetteOffset": 0.0, "Holding": 0.0,
            "PrimarySignalHPF": 0.0, "BridgeBalResist": 0.0, "PrimarySignalLPF": lpf,
            "RsCompBandwidth": 0.0, "WholeCellCompResist": 0.0, "WholeCellCompEnable": 0,
            "LeakSubResist": 0.0, "HoldingEnable": 0, "FastCompTau": 0.0,
            "SlowCompCap": 0.0, "WholeCellCompCap": 0.0, "LeakSubEnable": 0,
            "NeutralizationCap": 0.0, "BridgeBalEnable": 0, "RsCompCorrection": 0.0,
            "NeutralizationEnable": 0, "RsCompEnable": 0, "OutputZeroAmplitude": 0.0,
            "FastCompCap": 0.0, "SlowCompTau": 0.0,
        }

    def _clampstate_for(self, datac, item):
        mode, lpf, sr = "IC", 20000.0, 10000.0
        try:
            dfile = datac.items[item].dfile
            dm = str(dfile["Data_Mode"]["v"]).lower()
            mode = "VC" if dm in ("vc",) else "IC"
            nch = len(dfile["Channels"]["v"])
            sr = 1.0 / (float(dfile["Sample_Rate"]["v"]) * 1e-6 * max(nch, 1))
        except Exception:  # noqa: BLE001
            pass
        primary_units = "A" if mode == "VC" else "V"
        secondary_units = "V" if mode == "VC" else "A"
        return {
            "mode": mode,
            "primarySignal": "Membrane Current" if mode == "VC" else "Membrane Potential",
            "secondarySignal": "Command Voltage" if mode == "VC" else "Command Current",
            "primaryUnits": primary_units, "secondaryUnits": secondary_units,
            "primaryGain": 1.0, "secondaryGain": 1.0, "holding": 0.0,
            "ClampParams": self._clampparams(lpf),
            "DAQ": {
                "primary": {"rate": sr, "numPts": 0, "startTime": 0.0},
                "command": {"rate": sr, "numPts": 0, "startTime": 0.0},
                "secondary": {"rate": sr, "numPts": 0, "startTime": 0.0},
            },
        }

    def parseClampInfo(self, info):
        self.trClampInfo = info
        cs = info[1]["ClampState"]
        self.mode = cs["mode"]
        self.units = [cs["primaryUnits"], cs["secondaryUnits"]]
        self.samp_rate = cs["DAQ"]["primary"]["rate"]

    def parseClampHoldingLevel(self, info):
        try:
            return info[1]["ClampState"]["holding"]
        except (KeyError, IndexError, TypeError):
            return 0.0

    def getDataInfo(self, filename=None, silent: bool = False):
        assert filename is not None
        if isinstance(filename, (int, np.integer)):
            datac, item = self.datac, int(filename)
        else:
            matpath, item, _sweep, _explicit = self._parse_protocol_path(filename)
            datac = self._open_datac(matpath) if matpath is not None else self.datac
        if datac is None or item is None:
            if not silent:
                print(f"matdatac_reader.getDataInfo: no data block at {str(filename)}")
            return None
        clampstate = self._clampstate_for(datac, item)
        info = [{"name": "Time", "units": "s"}, {"ClampState": clampstate, "startTime": 0.0}]
        self.parseClampInfo(info)
        return info

    # -- synthesized directory / index --------------------------------------
    def subDirs(self, p):
        """Synthetic per-sweep subdirectories for a matdatac protocol block."""
        pp_ = Path(p)
        if self.last_protocol == pp_:
            return self.last_dirs
        matpath, item, _sweep, _explicit = self._parse_protocol_path(p)
        datac = self._open_datac(matpath) if matpath is not None else self.datac
        n = self._block_nrecs(datac, item)
        dirs = [pp_ / f"{i:03d}" for i in range(n)]
        self.last_dirs, self.last_protocol = dirs, pp_
        return dirs

    def _synthesize_protocol_index(self, datac, item):
        cs = self._clampstate_for(datac, item)
        return {
            ".": {
                "dirType": "ProtocolSequence",
                "__timestamp__": 0.0,
                "notes": "",
                "description": self._block_type(datac, item),
                "devices": {
                    self.datac_device: {
                        "mode": cs["mode"], "holdingCheck": False,
                        "holdingSpin": 0.0, "primaryUnits": cs["primaryUnits"],
                    },
                },
                "sequenceParams": {("protocol", "repetitions"): [0]},
            }
        }

    def _synthesize_sweep_index(self, sweep):
        return {".": {"dirType": "Protocol", "__timestamp__": 0.0, "sweep": sweep}}

    def _synthesize_cell_index(self, datac):
        species = age = sex = ""
        for it in getattr(datac, "items", []):
            if it.type == "HFILE":
                species = str(getattr(it, "species", ""))
                age = str(getattr(it, "age", ""))
                sex = str(getattr(it, "sex", ""))
                break
        return {
            ".": {
                "dirType": "Cell", "notes": "", "type": "",
                "species": species, "age": age, "sex": sex, "important": False,
            }
        }

    def readDirIndex(self, currdir=None):
        self._dirindex = None
        if currdir is None:
            currdir = self.protocol_path
        if currdir is None:
            return None
        if isinstance(currdir, (int, np.integer)):
            self._dirindex = self._synthesize_protocol_index(self.datac, int(currdir))
            return self._dirindex
        matpath, item, sweep, explicit = self._parse_protocol_path(currdir)
        if matpath is None:
            print(f"matdatac_reader.readDirIndex: no matdatac data recognized at {str(currdir)}")
            return None
        datac = self._open_datac(matpath)
        if datac is None:
            return None
        if sweep is not None:
            self._dirindex = self._synthesize_sweep_index(sweep)
        elif explicit:
            self._dirindex = self._synthesize_protocol_index(datac, item)
        else:  # bare .mat -> a cell
            self._dirindex = self._synthesize_cell_index(datac)
        return self._dirindex

    def getIndex(self, currdir=None, lineend="\n"):
        self.lb = lineend
        self._index = self.readDirIndex(currdir if currdir is not None else self.protocol_path)
        return self._index["."] if self._index is not None else None

    def _readIndex(self, currdir=None):
        self._index = self.readDirIndex(currdir if currdir is not None else self.protocol_path)
        return self._index

    def convert_timestamp(self, fts) -> str:
        return datetime.datetime.fromtimestamp(fts).strftime("%Y-%m-%d  %H:%M:%S %z")

    def printIndex(self, index):
        print(pprint.pformat(index))

    def getIndex_text(self, index):
        return pprint.pformat(index)

    def file_cell_protocol(self, filename=None):
        assert filename is not None
        matpath, item, sweep, explicit = self._parse_protocol_path(filename)
        proto = Path(filename).name if explicit else ""
        cell = matpath.stem if matpath is not None else Path(filename).stem
        date = matpath.parent.name if matpath is not None else ""
        parent = matpath.parent if matpath is not None else Path(filename).parent
        return (date, "slice_000", cell, proto, parent)

    def getClampDevices(self, currdir=None, verbose: bool = False):
        info = self.getIndex(currdir=currdir)
        devs = []
        if info is not None and "devices" in info:
            for d in info["devices"]:
                if d in self.clampdevices:
                    devs.append(d)
        return devs

    def header_metadata(self, filename=None):
        src = filename if filename is not None else self.protocol_path
        if src is not None and not isinstance(src, (int, np.integer)):
            matpath, _item, _sweep, _explicit = self._parse_protocol_path(src)
        else:
            matpath = None
        datac = self._open_datac(matpath) if matpath is not None else self.datac
        meta = {}
        if datac is not None:
            for it in datac.items:
                if it.type == "HFILE":
                    meta = {
                        "species": str(getattr(it, "species", "")),
                        "age": str(getattr(it, "age", "")),
                        "sex": str(getattr(it, "sex", "")),
                    }
                    break
            meta["nblocks"] = len(
                self.data_block_indices(matpath) if matpath is not None else self.data_block_indices()
            )
            meta["file"] = str(matpath) if matpath is not None else str(
                getattr(datac, "fname", "")
            )
        return meta

    # -- protocol checking (minimal parity) ---------------------------------
    def checkProtocol(self, protocolpath=None, allow_partial=False) -> bool:
        src = protocolpath if protocolpath is not None else self.protocol_path
        if src is None:
            return False
        if isinstance(src, (int, np.integer)):
            return self.datac is not None and 0 <= int(src) < len(self.datac.items)
        matpath, item, _sweep, _explicit = self._parse_protocol_path(src)
        datac = self._open_datac(matpath) if matpath is not None else None
        return datac is not None and item is not None

    def checkProtocolImportantFlags(self, protocolpath=None) -> bool:
        return True

    def checkProtocolImportant(self, protocol) -> bool:
        return True

    def listSequenceParams(self, dh=None):
        return {}

    # -- LSPS / imaging: no matdatac equivalent -> honest stubs -------------
    def getStim(self, stimname: str = "Stim"):
        return {"start": [], "duration": [], "amplitude": [], "npulses": [], "period": []}

    def getLaserBlueTimes(self):
        return False

    def getLaserBlueCommand(self):
        return False

    def getLaserBlueShutter(self):
        return False

    def getLEDCommand(self):
        return False

    def getPhotodiode(self):
        return False

    def getScannerPositions(self, dataname: str = "Laser-Blue-raw.ma"):
        return False

    def getImage(self, filename=None):
        return None

    def getAverageScannerImages(self, dataname: str = "Camera/frames.ma", mode: str = "average", **kwargs):
        return None


class DatacFile(object):
    def __init__(self, fname):
        """
        Open the matdatac file and read the data (all of it)
        Creates an instance with the data, a list of the
        'items' in the file. The items are either
        HFILE, Note, or data (sfile, dfile, data) Block
        structures from the Block, Hfile and Note classes.

        """
        assert Path(fname).is_file()
        self.fname = fname
        self.data = matlab.loadmat(fname)

        # read block records
        self.items = []
        blockid = None
        blockrecs = []
        for rec in self.data["Index"][0]:
            typ = rec["type"][0]
            # print('typ: ', typ)
            if typ == "HFILE":
                self.items.append(HFile(self, rec))
            elif typ == "NOTE":
                self.items.append(Note(self, rec))
            elif typ in ("SFILE", "DFILE", "DATA", "STIM"):
                bid = rec["block"][0][0]
                if blockid != bid:
                    if blockid is not None:
                        self.items.append(Block(self, blockrecs))
                        blockrecs = []
                    blockid = bid
                blockrecs.append(rec)

        if len(blockrecs) > 0:
            self.items.append(Block(self, blockrecs))

    def summary(self):
        """
        Generate a string with a short summary of the file (which blocks, etc)
        """
        summ = "FILE: " + str(self.fname)
        typs = [i.type for i in self.items if i.type not in ("HFILE", "NOTE")]
        summ += "\nBlocks: \n"
        for i, t in enumerate(typs):
            summ += f"    {i:3d}: {t:s}\n"

        for i in self.items:
            if i.type == "HFILE":
                summ += "\n" + i.summary() + "\n"
            if i.type == "NOTE":
                summ += "\n* " + i.note
        return summ


class HFile(object):
    def __init__(self, datac, rec):
        """
        Store the matdatac Hfile structure
        """
        self.datac = datac
        self.rec = rec
        self.data = flatten(self.datac.data[self.rec["MatName"][0]])
        self.type = "HFILE"
        self.age = self.data["Age"]["v"]
        self.species = self.data["Species"]["v"]
        self.sex = self.data["Sex"]["v"]

    def summary(self):
        exp = self.data["Experiment"]["v"]

        # What does it mean to have multiple values in the Experiment field??
        if isinstance(exp, np.ndarray):
            exp = "\n".join(exp)

        summ = exp + "\n"
        summ += "Species: %s  Age: %s" % (
            self.data["Species"]["v"],
            self.data["Age"]["v"],
        )
        return summ


class Note(object):
    def __init__(self, datac, rec):
        """
        Store the note associated with the current record
        from the matdatac file structure

        Parameters
        ----------
        datac : DatacFile object for the data

        rec : int
            Record for the note data to be retrieved
        """
        self.datac = datac
        self.rec = rec
        self.type = "NOTE"
        self.note = flatten(datac.data[rec["MatName"][0]])["v"]

    def summary(self):
        return self.note


class Block(object):
    def __init__(self, datac, block_recs):
        """
        Read the data from the block records

        Parameters
        ----------
        datac : DatacFile object for the data

        block_recs : list
            A list of the records to be read from the file and stored

        """
        self.datac = datac
        self._data = None
        self.recs = block_recs
        self.notes = []
        self.type = None
        self.stims = []
        self.recordings = []
        for rec in self.recs:
            typ = rec["type"][0]
            #    print('typ: ', typ)
            if rec["MatName"][0] not in self.datac.data.keys():
                print("missing data block: ", rec["MatName"][0])
                continue
            data = flatten(self.datac.data[rec["MatName"][0]])
            # print('data len: ', len(data))
            if typ == "DATA":
                self.type = rec["type2"][0]
                self.recordings.append(data)
            elif typ in ["SFILE", "STIM"]:
                #  print('block sfile, stype2, stims: ', rec['type2'])
                self.type = rec["type2"][0]
                self.stims.append(data)
            elif typ == "DFILE":
                self.dfile = data
            else:
                raise ValueError("Unrecognized data type: %s" % typ)

    def summary(self, printnow=True):
        summ = "%d data, %d stims\n" % (len(self.recordings[0]), len(self.stims[0]))
        summ += "---- DFILE: ----\n"
        summ = ""
        for k in self.dfile.keys():
            if k in [
                "Actual_Rate",
                "ChannelList",
                "Data_Mode",
                "Points",
                "Sample_Rate",
                "TraceDur",
            ]:
                fmtstr = "%15s : " + ("%s\n" % self.dfile[k]["f"])
                summ += fmtstr % (k, self.dfile[k]["v"])
        summ += "\n"
        summ += "%15s : %3d - %3d" % (
            "Records",
            self.dfile["Record"],
            len(self.recordings[0]),
        )
        summ += pprint.pformat(self.dfile)

        return summ

    def data(self):
        """Return (sample rate, [traces]) for this block, where [traces] is
        a list of (primary_channel, secondary_channel, stim) tuples.
        """
        if self._data is not None:
            return self._data

        d = self.recordings[0]
        # print('data keys: ', len(d))
        # print(isinstance(d, dict))
        if isinstance(d, dict) and len(d) == 1:
            dkey = list(d.keys())[0]
            dk = 0
            mode = d[dkey]["status"]["Data"]["mode"]
        else:
            dkey = 0
            dk = list(d[0].keys())[0]
            mode = d[dkey][dk]["status"]["Data"]["mode"]
        # print('dk: ', dk, ' dkey: ', dkey)
        # print('d[dkey]: ', d[dkey])
        if isinstance(mode, np.ndarray):
            mode = mode[0]
        x = []

        if len(self.stims) > 0:
            reps = self.stims[0]["Repeats"]["v"]
        else:
            reps = 1
        for i in range(len(d)):
            if isinstance(d, np.ndarray):
                dkx = list(d[i].keys())[0]
                data = d[i][dkx]["data"]
            else:
                dkx = list(d.keys())[0]
                data = d[dkx]["data"]
            if len(self.stims) > 0:
                stim = self.stims[0]["waveform"]
            else:
                stim = np.zeros(data.shape[0])
            if isinstance(stim, np.ndarray):
                # Is this right? when len(stims) < len(recordings), how should
                # they be matched together?
                if len(self.stims) > 0:
                    stim = stim[0]["v2"]  # stim[i//reps]['v2']
                else:
                    stim = np.zeros(data.shape[0])
            else:
                stim = stim["v2"]

            if mode == "V-Clamp":
                x.append((data[:, 1], data[:, 0], stim))
            else:
                x.append((data[:, 0], data[:, 1], stim))
                # x.append((pg.gaussianFilter(data[:, 0], sigma=(2,)), data[:, 1], stim))

        rate = 1e6 / self.dfile["Sample_Rate"]["v"]
        self._data = rate, x
        return self._data


# -------------------------------------------------

# def example_dir_scan(searchstr, protocol='cciv2'):
#     """
#     Scan a directory and plot the traces from the first protocol
#     that matches the search string
#     """
#     fns = glob.glob(searchstr)
#     for fn in fns:
#         df = printNotes(fn, indent=5)
#         if df is None:
#             continue
#         for protn in df.expInfo.keys():
#             if df.expInfo[protn]['Protocol'] == protocol:
#                 df.readrecords(range(df.expInfo[protn]['R'], df.expInfo[protn]['Rend']))
#                 df.plotcurrentrecs(fn)
#                 break
#     mpl.show()


def get_blocks_type(fn, protocol):
    df = DatacFile(Path(fn))
    for item in df.items:
        print(item.type)


def get_blocks(fn, protocol=None):
    blks = []
    prots = []
    df = DatacFile(Path(fn))
    for i, item in enumerate(df.items):
        if protocol is not None:
            if item.type.startswith(protocol):
                blks.append(i)
                prots.append(item.type)
        else:  # take all
            blks.append(i)
            prots.append(item.type)

    return blks, prots


def get_note(fn):
    df = DatacFile(Path(fn))
    for item in df.items:
        # print(item.type)
        if item.type == "HFILE":
            # print('HFILE: ', item.data['Experiment']['v'])
            return " ".join(item.data["Experiment"]["v"]).replace("\t", ",")
        if item.type == "NOTE":
            return item.note
    return ""


def plot_block(ax, fn=None, block=1, protocol=""):

    # matplotlib.use('TkAgg',warn=False, force=True)
    # f, ax = mpl.subplots(2,1)

    df = DatacFile(Path(fn))  # get the matdatac file
    GC = GetClamps(df)  # make an instance of the clamp converter
    GC.setProtocol(block)
    r = GC.getData(debug=True)  # convert the data to acq4 Clamp object
    if not r:
        return
    # print(df.summary())
    dfn = "df%d" % block
    dbn = "db_%d" % block
    print("MatdatacRead: filename fn: ", str(fn))
    print("    block: ", block)
    # print(GC.datac.data.keys()) # [dbn])  # is a numpy recarray
    for rec in range(len(GC.datac.data[dbn][0])):
        if isinstance(GC.datac.data[dbn][0][rec], dict):
            drec = list(GC.datac.data[dbn][0][rec].keys())
            V = GC.datac.data[dbn][0][rec][drec[0]]["data"]
        elif isinstance(GC.datac.data[dbn][0][rec], np.ndarray):
            # print(GC.datac.data[dbn][0][rec])
            # print(len(GC.datac.data[dbn][0][rec]))
            names = GC.datac.data[dbn][0][rec].dtype.names
            V = GC.datac.data[dbn][0][rec][names[0]][0][0][0][0][0]  # ugh - Why?
        T = GC.time_base
        sf2 = GC.datac.data["sf2"]
        # ax[0].plot(T, V[:, 0], 'k-', linewidth=0.5)

        # if len(V) > 1:
        ax.plot(T, V[:, 1], "b-", linewidth=0.5)
        # else:
        #     ax.plot(T, V[0], 'b-', linewidth=0.5)
        if protocol == "anshock":
            ax.set_xlim([0.008, 0.015])
        elif protocol == "antest":
            ax.set_xlim([0.004, 0.010])
        else:
            pass
    print(f"    Block: {block:d} Proto: {protocol:s}")
    ax.set_title(f"Block: {block:d} Proto: {protocol:s}")


def plot_blocks(fn, pdfpath=None):
    df = DatacFile(Path(fn))
    blks, protocols = get_blocks(fn)
    print("MatdatacRead: blocks: ", blks)
    nblk = len(blks)
    (r, c) = PH.getLayoutDimensions(nblk, pref="width")
    P = PH.regular_grid(
        r,
        c,
        order="rowsfirst",
        figsize=(8.0, 10.0),
        showgrid=False,
        verticalspacing=0.08,
        horizontalspacing=0.08,
        margins={
            "leftmargin": 0.08,
            "rightmargin": 0.08,
            "topmargin": 0.1,
            "bottommargin": 0.1,
        },
        labelposition=(0.0, 0.0),
        parent_figure=None,
        panel_labels=None,
    )
    axs = P.axarr.ravel()
    for iblk, blk in enumerate(blks):
        try:
            plot_block(axs[iblk], fn=fn, block=blk, protocol=protocols[iblk])
        except:
            pass
            #raise ValueError("Cannot plot block: ", iblk)
    note = get_note(fn)
    P.figure_handle.suptitle(f"File: {str(fn):s}, \nNote: {note:s}", fontsize=12)
    if pdfpath is not None:
        print("PDF File: ", str(Path(pdfpath, fn.stem).with_suffix(".pdf")))
        mpl.savefig(Path(pdfpath, fn.stem).with_suffix(".pdf"))
    else:
        mpl.show()


# Claude added 2026-09-29: expose GetClamps under the module-name alias that
# datareaders.get_reader.get_reader() dispatches to for matdatac files.  The
# path-based, acq4-API-compatible construction lives in GetClamps itself.
matdatac_reader = GetClamps


def main():

    path = ""
    datadir = "/Volumes/Pegasus/ManisLab_Data3/Rich_Alex/Rich_Acq3/"
    #    fn = '/Users/pbmanis/Documents/data/RX_CCdata/13jul09a.mat'
    fn = "/Volumes/Pegasus/ManisLab_Data3/Rich_Alex/Rich_Acq3/01may09c.mat"
    fn = "/Volumes/Pegasus/ManisLab_Data3/Rich_Alex/Rich_Acq3/19feb09d.mat"
    # fn = '/Volumes/Pegasus/ManisLab_Data3/Rich_Alex/Rich_Acq3/'
    pdfpath = "/Users/pbmanis/Desktop/Python/awrdata"
    if Path(fn).is_dir():
        files = Path(fn).glob("*.mat")
        for f in sorted(list(files)):
            blks, prots = get_blocks(f)
            note = get_note(f)
            print("MatdatacRead (main): blks: ", blks)
            if note.find("PTS") >= 0 and len(blks) > 0:
                print("   PTS: ", f, prots, note)
                plot_blocks(f, Path(pdfpath, "PTS"))
            elif note.find("TTS") >= 0 and len(blks) > 0:
                print("   TTS: ", f, prots, note)
                plot_blocks(f, Path(pdfpath, "TTS"))
            elif len(blks) > 0:
                print("   CTL: ", f, prots, note)
                plot_blocks(f, Path(pdfpath, "CTL"))
            else:
                pass
            # plot_blocks(f, Path(pdfpath, 'CTL'))
    else:
        print("MatdatacRead (main) filename: ", Path(fn).name)
        plot_blocks(fn, pdfpath=None)
    # if fn is None:
    #     sel = FS.FileSelector(dialogtype='file', startingdir=datadir)
    #     print(sel.fileName)
    #     if sel.fileName is None:
    #         return
    #     # fn = '/Volumes/Samsung_T5/data/Rich_Alex/Acq3Files/10dec08a.mat'
    #         show_block(sel.fileName[0], 0)
    # else:
    #     show_block(fn, 4)


if __name__ == "__main__":
    main()
#  df = DatacFile(os.path.join(fn))  # get the matdatac file
#  # for it in df.items:
#  #     if isinstance(it, Block):
#  #         print(it.summary())
#  #
#  blocksel = 2
#  GC = GetClamps(df)  # make an instance of the clamp converter
#  GC.setProtocol(blocksel)
#  GC.getData()  # convert the data to acq4 Clamp object
#  # print(dir(GC))
#  # print(dir(GC.datac))
#  # print(GC.datac.data.keys())
#  block = 2
#  dfn = 'df%d' % block
# # print('dfn: ', GC.datac.data[dfn][0][0])
#  print(len(GC.datac.data[dfn]))
#  dbn = 'db_%d' % block
#  kx = list(GC.datac.data[dbn][0][0].keys())[0]
#  print('kx: ', kx)
#  V = GC.datac.data[dbn][0][0][kx]['data']
#  print(GC.datac.data[dbn][0][0][kx].keys())
#  T = GC.time_base
#  print(GC.protocol)
#  sf = GC.datac.data['sf2']
# # print(sf[0][0])
#  #mpl.ion()
#  f, ax = mpl.subplots(1,1)
#  print(V.shape)
#  print(T.shape)
#  for i in range(int(V[:, 0].shape[0]/T.shape[0])):
#      ax.plot(T, V[i*len(T):i*len(T)+len(T)], 'k-', linewidth=0.5)
#  mpl.draw()
#  mpl.show()
#  print('aaa')# print(GC.tstart, GC.tend)
# IV = IVS.IVSummary(2)
# IV.AR = GC
# IV.compute_iv()
