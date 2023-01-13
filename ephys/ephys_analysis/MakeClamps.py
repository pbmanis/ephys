"""
Create an acq4-style Clamps structure (for the acq4/ephysanalysis routines)
from simple data (time, traces, command)
This is used as a base class for vcnmodel readmodel to read a pickled data file

"""

from dataclasses import dataclass
from typing import Union

import numpy as np
import MetaArray as EM


class MakeClamps:
    """
    Make acq4 clamp structures from data passed from other formats
    """

    def __init__(self):
        self.holding = 0.0  # set a default value for models
        self.WCComp = 0.0
        self.CCComp = 0.0
        self.NWB = False
        self.nwbfile = None

    def set_clamps(
        self,
        time,
        data,
        cmddata=None,
        dmode="CC",
        tstart_tdur=[0.01, 0.100],
        protocol=None,
    ):
        """
        Copy parameters into the Clamps structure.

        Parameters
        ----------
        time: np.array (no default)
            An array holding the time data. Should be same length as the data

        data: np.array or ndarray (no default)
            An array of the data

        cmddata: np.array (default None)
            Command waveform data if available

        dmode : str (default: 'CC')
            data mode: CC for current clamp, vc for voltage clamp

        tatsrt_tdur: 2-element list or np.array (default: [0.01, 0.10]; values in seconds)
            The start and duration for current/voltage pulses and steps.

        """
        self.data = data
        self.time = time
        self.rate_sec = np.diff(self.time)
        self.cmddata = cmddata
        self.tstart_tdur = tstart_tdur
        self.tstart = tstart_tdur[0]
        self.tend = np.sum(tstart_tdur)
        self.mode = dmode
        self.protocol = protocol

    def setNWB(self, nwbmode: bool = False, nwbfile: Union[object, None] = None):
        self.NWB = nwbmode
        self.nwbfile = nwbfile

    def getData(self):
        return self.getClampData()

    def getClampData(self):
        """
        Translates fields as best as we can from the original or an NWB structure
        create a Clamp structure for use in SpikeAnalysis and RMTauAnalysis.
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
        self.amplifeirSettings['WCCompValid', 'WCEmabled', 'CompEnabled', 'WCSeriesResistance']
        self.cmd_wave
        self.commandLevels (np.array(self.values))
        self.traces = EM.MetaArray(traces, info=info)
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

        Parameters
        ----------
        None

        """
        if self.data is None:
            raise ValueError("No data has been set")
        protocol = self.protocol

        # filled, automatically with default values
        self.repc = 1
        self.nrepc = 1
        self.model_mode = False
        self.command_scale_factor = 1
        self.command_units = "A"
        self.devicesUsed = None
        self.clampDevices = None
        self.holding = 0.0
        self.bridgeBalance = 0.0
        self.neutralizingCap = 0.0
        self.primaryGain = 1.0
        self.primarySignalLPF = 20000.
        self.FastCompCap = 0.
        self.SlowCompCap = 0.
        self.RsCompBandwidth = 10000. # hz
        self.RsCompCorrection = 0.
        self.RsPrecition = 0.  # not recorded in acq4, we rarely use this.
        self.WholeCellCompCap = 0.
        self.WholeCellCompResist = 0.
        self.amplfierSettings = {
            "WCCompValid": False,
            "WCEnabled": False,
            "CompEnabled": False,
            "WCSeriesResistance": 0.0,
        }
        self.clampState = None
        self.RSeriesUncomp = 0.0
        if self.NWB:  # fill in as many of the parameters from the nwbfile as we can
            if self.nwbfile.acquisition['Vcs1'].comments == "CC":
                step_times = self.nwbfile.acquisition['Ics1'].control
                step_times = self.nwbfile.acquisition["Ics1"].control
                self.holding = self.nwbfile.acquisition["Vcs1"].bias_current
                self.bridgeBalance = self.nwbfile.acquisition["Vcs1"].bridge_balance
                self.neutralizingCap = self.nwbfile.acquisition[
                    "Vcs1"
                ].capacitance_compensation
                self.primarySignalLPF = self.nwbfile.acquisition["Vcs1"].electrode.filtering
                self.primaryGain = self.nwbfile.acquisition["Vcs1"].gain
            
            elif self.nwbfile.acquisition['Vcs1'].comments == "VC":
                vacq = self.nwbfile.acquisition["Vcs1"]
                iacq = self.nwbfile.acquisition["Ics1"]
                step_times = vacq.control
                self.holding = vacq.offset
                self.FastCompCap = iacq.capacitance_fast
                self.SlowCompCap = iacq.capacitance_slow
                self.RsCompBandwidth = iacq.resistance_comp_bandwidth
                self.RsCompCorrection = iacq.resistance_comp_correction,
                self.RsPrecition = iacq.resistance_comp_prediction  # not recorded in acq4, we rarely use this.
                self.WholeCellCompCap = iacq.whole_cell_capacitance_comp
                self.WholeCellCompResist = iacq.whole_cell_series_resistance_comp
 
                self.primarySignalLPF = self.nwbfile.acquisition["Ics1"].electrode.filtering
                self.primaryGain = self.nwbfile.acquisition["Ics1"].gain

            self.tstart = step_times[0]*self.rate_sec[0]
            self.tend = step_times[1]*self.rate_sec[0]
            self.tdur = self.tend - self.tstart

        else:
            self.tstart = self.tstart_tdur[
                0
            ]  # should be pulled from protocol/stimulus information
            self.tdur = self.tstart_tdur[1]
            self.tend = self.tstart + self.tdur
        
        self.sample_interval = self.rate_sec[0] # express in seconds
        self.traces = np.array(self.data)

        dt = self.sample_interval  # makes assumption that rate is constant in a block
        if self.NWB:
            time_ax = 0
            rec_ax = 1
        else:
            time_ax = 1
            rec_ax = 0
        self.time_base = self.time[: self.traces.shape[time_ax]]  # in seconds
        points = self.data.shape[time_ax]
        recs = range(self.data.shape[rec_ax])

        if self.mode == "CC":  # use first channel
            mainch = 0
            cmdch = 1
        else:  # assumption is swapped - for this data, that means voltage clamp mode.
            mainch = 1
            cmdch = 0

        cmds = self.cmddata  # self.traces[:,cmdch,:]
        self.cmd_wave = self.cmddata  # np.squeeze(self.traces[:, cmdch, :])
        diffpts = self.traces.shape[time_ax] - self.cmd_wave.shape[time_ax]
        ntr = self.cmd_wave.shape[rec_ax]
        if diffpts > 0:
            self.cmd_wave = np.pad(
                self.cmd_wave,
                pad_width=(0, diffpts),
                mode="constant",
                constant_values=0.0,
            )[
                :, :ntr
            ]  # awkward
        self.holding = np.zeros((ntr, 1))
        t0 = int(self.tstart / dt)
        t1 = int(self.tend / dt)
        if cmds.shape[0] > 1:
            if self.NWB:
                self.values = np.nanmean(
                    self.cmd_wave[t0:t1, :], axis=time_ax
                )  # express values in amps
                self.holding = np.nanmean(self.cmd_wave[:t0, :], axis=time_ax)
                self.cmd_wave = self.cmd_wave.T
                self.traces = self.traces.T

            else:
                self.values = np.nanmean(
                    self.cmd_wave[:, t0:t1], axis=time_ax
                )  # express values in amps
                self.holding = np.nanmean(self.cmd_wave[:, :t0], axis=time_ax)
        else:
            self.values = np.zeros_like(self.traces.shape[1:2])
        self.commandLevels = self.values
        # for i in range(self.traces.shape[0]):
        #     mpl.plot(self.time, self.traces[i])
        #     mpl.plot(self.time[:self.cmd_wave[i].shape[0]], self.cmd_wave[i])
        # mpl.show()

        info = [
            {"units": "A", "values": self.values, "name": "Command"},  # dict 0
            {"name": "Time", "units": "s", "values": self.time_base},  # dict 1
            {
                "ClampState": {  # note that many of these values are just defaults and cannot be relied upon
                    "primarySignal": "Membrane Potential",
                    "primaryGain": self.primaryGain,
                    "primaryUnits": "V",
                    "primaryScaleFactor": 0.1,

                    "secondarySignal": "Command Current",
                    "secondaryGain": 1.0,
                    "secondaryScaleFactor": 2e-09,
                    "secondaryUnits": "A",

                    "extCmdScale": 4e-10,
                    "mode": self.mode,
                    "holding": self.holding,
                    "LPFCutoff": 10000.0,
                    "membraneCapacitance": 0.0,
 
                    "ClampParams": {
                        "BridgeValEnable": 6004,
                        "BridgeBalResist": self.bridgeBalance,
                        "FastCompCap": 0.0,
                        "FastCompTau": 0.0,
                        "HoldingEnable": 0,
                        "Holding": self.holding,
                        "LeakSubEnable": 0,
                        "LeakSubResist": 0.0,
                        "NeutralizationCap": self.neutralizingCap,
                        "NeutralizationEnable": 6004,
                        "OutputZeroAmplitude": 0.0,
                        "OutputZeroEnable": 0,
                        "PipetteOffset": 0.0,
                        "PrimarySignalHPF": 0.0,
                        "PrimarySignalLPF": self.primarySignalLPF,
                        "RsCompBandwidth": 0.0,
                        "RsCompCorrection": 0.0,
                        "RsCompEnable": 6004,
                        "SlowCompCap": 0.0,
                        "SlowCompTau": 0.0,
                        "WholeCellCompCap": 0.0,
                        "WholeCellCompEnable": 6004,
                        "WholeCellCompResist": 0.0,
                    },
                },
                "Protocol": {
                    "recordState": True,
                    "secondary": None,
                    "primary": None,
                    "mode": "IC",
                },
                "DAQ": {
                    "command": {
                        "numPts": points,
                        "rate": self.sample_interval,
                        "type": "ao",
                        "startTime": 0.0,
                    },
                    "primary": {
                        "numPts": points,
                        "rate": self.sample_interval,
                        "type": "ai",
                        "startTime": 0.0,
                    },
                    "secondary": {
                        "numPts": points,
                        "rate": self.sample_interval,
                        "type": "ai",
                        "startTime": 0.0,
                    },
                },
                "startTime": 0.0,
            },
        ]

        # if self.traces.shape[0] > 1:
        #     # depending on the mode, select which channel goes to traces
        #     self.traces = self.traces[:,mainch,:]
        # else:
        #     self.traces[0,mainch,:] = self.traces[0,mainch,:]

        self.traces = EM.MetaArray(self.traces, info=info)
        self.sample_rate = np.ones(self.traces.shape[rec_ax]) * self.sample_interval
        self.cmd_wave = EM.MetaArray(
            self.cmd_wave,
            info=[
                {"name": "Command", "units": "A", "values": np.array(self.values)},
                self.traces.infoCopy("Time"),
                self.traces.infoCopy(-1),
            ],
        )

        self.spikecount = np.zeros(len(recs))
        self.rgnrmp = [0.0, self.tstart - self.sample_interval]
        return True

    def setProtocol(self, protocol):
        self.protocol = protocol
