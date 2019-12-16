"""
Create a Clamps structure (for the acq4/ephysanalysis routines) from modeling data generated
in VCN_Model (the pickled file)

"""
from pathlib import Path
import numpy as np
import pickle
import matplotlib
# matplotlib.use('Qt4Agg')

import matplotlib.pyplot as mpl
import pylibrary.plotting.plothelpers as PH
from pylibrary.params import Params
import ephysanalysis.metaarray as EM 


class MakeClamps():
    
    def __init__(self):
        self.holding = 0.  # set a default value for models
        self.WCComp = 0.
        self.CCComp = 0.
        pass
    
    def set_clamps(self, dmode='CC', time=None, data=None, cmddata=None, tstart_tdur=[0.01, 0.100]):
        self.data = data
        self.time = time
        self.rate = np.diff(self.time)*1e6
        self.cmddata = cmddata
        self.tstart_tdur = tstart_tdur
        self.tstart = tstart_tdur[0]
        self.tend = np.sum(tstart_tdur)
        self.dmode = dmode
    
    def read_pfile(self, filename, plot=False):
        fh = open(filename, 'rb')
        df = pickle.load(fh)
        r = df['Results'][0]

        if plot:
            P = PH.Plotter((1, 1), figsize=(6, 4))
            cell_ax = list(P.axdict.keys())[0]
            for trial in range(len(df['Results'])):
                ds = df['Results'][trial]
                k0 = list(df['Results'][trial].keys())[0]
                dx = ds[k0]['monitor']
                P.axdict[cell_ax].plot(dx['time'], dx['postsynapticV'], linewidth=1.0)
                P.axdict[cell_ax].set_xlim(0., 150.)
                P.axdict[cell_ax].set_ylim(-200., 50.)
            PH.calbar(P.axdict[cell_ax], calbar=[120., -95., 25., 20.], axesoff=True, orient='left', 
                    unitNames={'x': 'ms', 'y': 'mV'}, font='Arial', fontsize=8)

            # mpl.savefig(outfile)
            mpl.show()
        # print(list(df.keys()))
        # print('\nbasename: ', df['basename'])
        # print('\nruninfo: ', df['runInfo'])
        """
        The runInfo dictionary holds somethign like this:
        runinfo:  {'folder': PosixPath('VCN_Cells/VCN_c08/Simulations/IV'), 'fileName': 'Normal', 'runName': 'Run', 
        'manipulation': 'Canonical', 'preMode': 'cc', 'postMode': 'cc', 'TargetCellType': 'Bushy', 
        'electrodeSection': 'soma', 'dendriticElectrodeSection': 'dendrite', 
        'dendriticSectionDistance': 100.0, 'celsius': 37, 'nStim': 1, 
        'stimFreq': 200.0, 'stimInj': {'pulse': [-1.0, 2.01, 0.2]}, 
        'stimDur': 100.0, 'stimDelay': 5.0, 'stimPost': 3.0, 
        'vnStim': 1, 'vstimFreq': 200.0, 'vstimInj': 50, 
        'vstimDur': 50.0, 'vstimDelay': 2.0, 'vstimPost': 3.0, 'vstimHolding': -60, 
        'gif_i0': 0.0, 'gif_sigma': 0.5, 'gif_fmod': 0.2, 'gif_tau': 3.0, 
        'gif_dur': 10.0, 'gif_skew': 0.0, 
        'runTime': 'Wed Oct  9 13:05:54 2019', 
        'inFile': None, 'inFileRep': 1, 'spikeTimeList': {}, 
        'v_init': -61.0, 'useSaveState': True, 'tstop': 8.0, 'filename': 'VCN_c08_pulse_'}
        """
        # print('\nmodelPars: ', df['modelPars'])
        """
        The modelPars dict holds the following:
        modelPars:  {'species': 'mouse', 'cellClass': 'bushy', 'modelType': 'II', 
        'modelName': 'mGBC', 'soma': True, 'axon': False, 
        'dendrites': False, 'pumps': False, 'hillock': False, 
        'initialsegment': False, 'myelinatedaxon': False, 
        'unmyelinatedaxon': False, 'na': 'nav11', 'ttx': False, 
        'name': 'bushy', 'morphology': 'VCN_Cells/VCN_c08/Morphology/VCN_c08.hoc', 
        'temperature': 34.0}
        
        Note 10/28/2019 changed structure so that runInfo and modelPars are both 
        subdictionaries of Params
        """
        if 'runInfo' not in list(df.keys()):  # handle data structure change 10/28/2019
            dinfo = df['Params']['runInfo']
        else:
            dinfo = df['runInfo']
        if isinstance(dinfo, Params):
            dinfo = dinfo.todict()
        print(dinfo)
        dur = dinfo['stimDur']
        delay = dinfo['stimDelay']
        mode = dinfo['postMode'].upper()
        ntr = len(df['Results'])
        V = [[]]*ntr
        I = [[]]*ntr
        for i in range(len(df['Results'])):
            fk = list(df['Results'][i].keys())[0]
            dfx = df['Results'][i][fk]['monitor']
            timebase = dfx['time']
            V[i] = dfx['postsynapticV']
            I[i] = dfx['i_stim0']
        V = np.array(V)
        I = np.array(I)
        self.set_clamps(dmode=mode, time=timebase, data=V, cmddata=I, tstart_tdur=[delay, dur])
        self.getClampData()

    def getClampData(self, verbose=False):
        """
        Translates fields as best as we can from the original DATAC structure
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
        if self.data is None:
            raise ValueError('No data has been set')
        protocol = ''

        self.sample_interval = self.rate[0]*1e-6  # express in seconds
        self.traces = np.array(self.data)
        points = self.data.shape[1]
        recs = range(self.data.shape[0])
        nchannels = self.data.shape[0]
        dt = self.sample_interval  # make assumption that rate is constant in a block
        self.time_base = self.time[:self.traces.shape[1]] # in seconds
        if self.dmode == 'CC':  # use first channel
            mainch = 0
            cmdch = 1
        else:  # assumption is swapped - for this data, that means voltage clamp mode.
            mainch = 1
            cmdch = 0

        cmds = self.cmddata # elf.traces[:,cmdch,:]
        self.tstart = self.tstart_tdur[0]  # could be pulled from protocol/stimulus information
        self.tdur = self.tstart_tdur[1]
        self.tend = self.tstart + self.tdur
        t0 = int(self.tstart/dt)
        t1 = int(self.tend/dt)
        self.cmd_wave = self.cmddata # np.squeeze(self.traces[:, cmdch, :])
        diffpts = self.traces.shape[1] - self.cmd_wave.shape[1]
        ntr = self.cmd_wave.shape[0]
        self.cmd_wave = np.pad(self.cmd_wave, (0, diffpts), 'constant', constant_values=0.)[:ntr,:]  # awkward
        if cmds.shape[0] > 1:
            self.values = np.nanmean(self.cmd_wave[:, t0:t1], axis=1)  # express values in amps
        else:
            self.values = np.zeros_like(self.traces.shape[1:2])
        self.commandLevels = self.values        
        # for i in range(self.traces.shape[0]):
        #     mpl.plot(self.time, self.traces[i])
        #     mpl.plot(self.time[:self.cmd_wave[i].shape[0]], self.cmd_wave[i])
        # mpl.show()
        
        info = [{'units': 'A', 'values': self.values, 'name': 'Command'},
                    {'name': 'Time', 'units': 's', 'values': self.time_base},
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
                                'mode': self.dmode, 'holding': 0.0, 'primaryUnits': 'V', 
                                'LPFCutoff': 10000.,
                                'secondaryUnits': 'A', 'primaryScaleFactor': 0.1,
                                'membraneCapacitance': 0.0}, 
                            'Protocol': {'recordState': True, 'secondary': None,
                                    'primary': None, 'mode': 'IC'}, 
                            'DAQ': {'command': {'numPts': points, 'rate': self.sample_interval,
                                    'type': 'ao', 'startTime': 0.},
                            '       primary': {'numPts': points, 'rate': self.sample_interval,
                                    'type': 'ai', 'startTime': 0.}, 
                                    'secondary': {'numPts': points, 'rate': self.sample_interval,
                                    'type': 'ai', 'startTime': 0.}
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
        #     # dependiung on the mode, select which channel goes to traces
        #     self.traces = self.traces[:,mainch,:]
        # else:
        #     self.traces[0,mainch,:] = self.traces[0,mainch,:]

        self.traces = EM.MetaArray(self.traces, info=info)
        self.cmd_wave = EM.MetaArray(self.cmd_wave,
             info=[{'name': 'Command', 'units': 'A',
              'values': np.array(self.values)},
              self.traces.infoCopy('Time'), self.traces.infoCopy(-1)])
              
        self.spikecount = np.zeros(len(recs))
        self.rgnrmp = [0, 0.004]
