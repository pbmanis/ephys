"""
Create an acq4-style Clamps structure (for the acq4/ephysanalysis routines)
from simple data (time, traces, command)
This is used as a base class for vcnmodel readmodel to read a pickled data file

"""
from pathlib import Path
import numpy as np
import pickle
import matplotlib
from dataclasses import dataclass
import dataclasses
import datetime
import matplotlib.pyplot as mpl
import pylibrary.plotting.plothelpers as PH
from pylibrary.tools.params import Params
import MetaArray as EM 


class MakeClamps:
    """
    Make acq4 clamp structures from data passed from other formats
    """
    
    def __init__(self):
        self.holding = 0.  # set a default value for models
        self.WCComp = 0.
        self.CCComp = 0.
        self.NWB = False

    
    def set_clamps(self, time, data, cmddata=None, dmode='CC', tstart_tdur=[0.01, 0.100], protocol=None):
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
        self.rate = np.diff(self.time)*1e6
        self.cmddata = cmddata
        self.tstart_tdur = tstart_tdur
        self.tstart = tstart_tdur[0]
        self.tend = np.sum(tstart_tdur)
        self.dmode = dmode
        self.protocol = protocol

    def setNWB(self, nwbmode:bool = False):
        self.NWB = nwbmode

    def getData(self):
        return(self.getClampData())

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
            raise ValueError('No data has been set')
        protocol = self.protocol

        self.sample_interval = self.rate[0]*1e-6  # express in seconds
        self.traces = np.array(self.data)
        # nchannels = self.data.shape[0]
        dt = self.sample_interval  # make assumption that rate is constant in a block
        if self.NWB:
            time_ax = 0
            rec_ax = 1
        else:
            time_ax = 1
            rec_ax = 0
        self.time_base = self.time[:self.traces.shape[time_ax]] # in seconds
        points = self.data.shape[time_ax]
        recs = range(self.data.shape[rec_ax])

        if self.dmode == 'CC':  # use first channel
            mainch = 0
            cmdch = 1
        else:  # assumption is swapped - for this data, that means voltage clamp mode.
            mainch = 1
            cmdch = 0

        cmds = self.cmddata # self.traces[:,cmdch,:]
        self.tstart = self.tstart_tdur[0]  # could be pulled from protocol/stimulus information
        self.tdur = self.tstart_tdur[1]
        self.tend = self.tstart + self.tdur
        t0 = int(self.tstart/dt)
        t1 = int(self.tend/dt)
        self.cmd_wave = self.cmddata # np.squeeze(self.traces[:, cmdch, :])
        diffpts = self.traces.shape[time_ax] - self.cmd_wave.shape[time_ax]
        ntr = self.cmd_wave.shape[rec_ax]
        if diffpts > 0:
            self.cmd_wave = np.pad(self.cmd_wave, pad_width=(0, diffpts), 
                    mode='constant', constant_values=0.)[:, :ntr]  # awkward
        self.holding = np.zeros((ntr, 1))
        if cmds.shape[0] > 1:
            if self.NWB:
                self.values = np.nanmean(self.cmd_wave[t0:t1, :], axis=time_ax)  # express values in amps
                self.holding = np.nanmean(self.cmd_wave[:t0, :], axis=time_ax)
                self.cmd_wave = self.cmd_wave.T
                self.traces = self.traces.T
            else:
                self.values = np.nanmean(self.cmd_wave[:, t0:t1], axis=time_ax)  # express values in amps
                self.holding = np.nanmean(self.cmd_wave[:, :t0], axis=time_ax)
        else:
            self.values = np.zeros_like(self.traces.shape[1:2])
        self.commandLevels = self.values        
        # for i in range(self.traces.shape[0]):
        #     mpl.plot(self.time, self.traces[i])
        #     mpl.plot(self.time[:self.cmd_wave[i].shape[0]], self.cmd_wave[i])
        # mpl.show()
        
        info = [{'units': 'A', 'values': self.values, 'name': 'Command'},  # dict 0
                    {'name': 'Time', 'units': 's', 'values': self.time_base},  # dict 1
                    {'ClampState':  # note that many of these values are just defaults and cannot be relied upon
                        {'primaryGain': 1.0, 'ClampParams': 
                            {'OutputZeroEnable': 0, 'PipetteOffset': 0.0,
                            'Holding': 0., 'PrimarySignalHPF': 0.0, 'BridgeBalResist': 0.0, 
                            'PrimarySignalLPF': 20000.0, 'RsCompBandwidth': 0.0, 
                            'WholeCellCompResist': 0.0, 'WholeCellCompEnable': 6004, 'LeakSubResist': 0.0,
                            'HoldingEnable': 1, 'FastCompTau': 0.0, 'SlowCompCap': 0.0, 
                            'WholeCellCompCap': 0.,
                            'LeakSubEnable': 6004, 'NeutralizationCap': 0.,
                            'BridgeBalEnable': 0, 'RsCompCorrection': 0.0,
                            'NeutralizationEnable': 1, 'RsCompEnable': 6004,
                            'OutputZeroAmplitude': 0., 'FastCompCap': 0.,
                            'SlowCompTau': 0.0}, 'secondarySignal': 
                            'Command Current', 'secondaryGain': 1.0,
                            'secondaryScaleFactor': 2e-09,
                            'primarySignal': 'Membrane Potential', 'extCmdScale': 4e-10,
                            'mode': self.dmode, 'holding': self.holding, 'primaryUnits': 'V', 
                            'LPFCutoff': 10000.,
                            'secondaryUnits': 'A', 'primaryScaleFactor': 0.1,
                            'membraneCapacitance': 0.0,
                        }, 
                    'Protocol': {'recordState': True, 'secondary': None,
                                'primary': None, 'mode': 'IC',
                            }, 
                    'DAQ': {'command': {'numPts': points, 'rate': self.sample_interval,
                                'type': 'ao', 'startTime': 0.},
                                'primary': {'numPts': points, 'rate': self.sample_interval,
                                'type': 'ai', 'startTime': 0.}, 
                                'secondary': {'numPts': points, 'rate': self.sample_interval,
                                'type': 'ai', 'startTime': 0.},
                            },
                    'startTime': 0.}
                ]

        # filled, automatically with default values
        self.repc = 1
        self.nrepc = 1
        self.model_mode = False
        self.command_scale_factor = 1
        self.command_units = 'A'
        self.devicesUsed = None
        self.clampDevices = None
        self.holding = 0.
        self.amplfierSettings = {'WCCompValid': False, 'WCEnabled': False, 
                'CompEnabled': False, 'WCSeriesResistance': 0.}
        self.clampState = None
        self.RSeriesUncomp = 0.
            
        self.tend = self.tstart + self.tdur

        # if self.traces.shape[0] > 1:
        #     # depending on the mode, select which channel goes to traces
        #     self.traces = self.traces[:,mainch,:]
        # else:
        #     self.traces[0,mainch,:] = self.traces[0,mainch,:]

        self.traces = EM.MetaArray(self.traces, info=info)
        self.sample_rate = np.ones(self.traces.shape[rec_ax])*self.sample_interval
        self.cmd_wave = EM.MetaArray(self.cmd_wave,
             info=[{'name': 'Command', 'units': 'A',
              'values': np.array(self.values)},
              self.traces.infoCopy('Time'), self.traces.infoCopy(-1)])


        self.spikecount = np.zeros(len(recs))
        self.rgnrmp = [0, 0.004]
        return True
    
    def setProtocol(self, protocol):
        self.protocol=protocol

