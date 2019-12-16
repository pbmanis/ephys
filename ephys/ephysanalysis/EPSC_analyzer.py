"""
Analyze EPSCs or IPSCs
Or EPSPs and IPSPs...

This module provides the following analyses:

1. Amplitudes from a train
2. Paired pulse facilitation for pulse pairs, and the first pair in a train.
3. Current-voltage relationship in voltage clamp measured over a time window

The results of the analysis are stored in the class variable analysis_summary


"""

import sys
from pathlib import Path
import os  #legacy
import scipy.signal
import pandas as pd

from cycler import cycler
from itertools import cycle
import numpy as np

import ephysanalysis as EP
import ephysanalysis.metaarray as EM  # need to use this version for Python 3
import ephysanalysis.cursor_plot as CP
import pylibrary.plotting.plothelpers as PH
import matplotlib.pyplot as mpl
import matplotlib.colors

import pyqtgraph as pg
from pyqtgraph.Qt import QtGui, QtCore
from pyqtgraph.Point import Point


def make_key(pathname):
    """
    Make a key string using the date, slice, cell and protocol from the path name
    """
    p = pathname.parts

    return(str('~'.join([p[i] for i in range(-4, 0)])))


class PSCAnalyzer():
    def __init__(self, datapath, plot=True, update_regions=False):
        """
        Analyze PSCs in a few different formats:
        IO - a stimulus sequence with increasing stimulation current,
        all collected at a single holding voltage
        VDEP - a Meausrement of EPSCs across voltage, targeted at obtaining
        an NMDA/AMPA current ratio from currents at +50 and -90 mV. Data may include
        averaging of repetead trials.
        PPF - Paired pulse facilitiation over several intervals; may include repeated
        trials
        
        Parameters
        ----------
        datapath : path to the data protocol (Path or string)
        
        plot : boolean (default: True)
            Flag to control plotting of the data
        
        update_regions: Boolean (default: False)
            A flag that forces the routines to plot data so that a time window for the
            analysis can be defined and saved.
        
        
        """
        self.datapath = datapath
        self.AR = EP.acq4read.Acq4Read()  # make our own private cersion of the analysis and reader
        self.plot = plot
        self.db = None
        self.db_filename = None
        self.update_regions = update_regions
        self.JunctionPotential = -8.0 * 1e-3  # junction potential for correction
        self.NMDA_voltage = 0.050 # in V  positive
        self.AMPA_voltage = -0.0741 # in V  - this is the Cl eq potential to minize GABA interference
        self.NMDA_delay = 0.050 # delay in s to make measurement
        

    def setup(self, clamps=None, spikes=None, baseline=[0, 0.001]):
        """
        Set up for the fitting
        
        Parameters
        ----------
        clamps : A datamodel structure (required)
            Brings the data to the module. This usually will be a PatchEphys object.
        
        spikes : A spikeAnalysis structure (optional)
            Has information about which traces have spikes
            Use this when analyzing events that may be contaminated by spikes
        
        baseline : list (2 elements)
            times over which baseline is measured (in seconds)
        
        """
        
        if clamps is None:
            raise ValueError("VC analysis requires defined clamps ")
        self.Clamps = clamps
        self.spikes = spikes
        self.set_baseline_times(baseline)

        self.analysis_summary = {}  # init the result structure

    def check_protocol(self, protocol):
        """
        Verify that the protocol we are examining is complete.
        Returns True or False
        """
        
        return(self.AR.checkProtocol(protocol))
            
    def read_database(self, filename):
        """
        Read the database that will be used for analysis
        The database is a pandas pickled file with columns
        date, protocol, T0 and T1
        
        Parameters
        ----------
        filename : str or Path 
            The name of the database file (full path or file if in the current
            working directory)
        """
        
        self.db_filename = Path(filename)
        if self.db_filename.is_file():
            with(open(self.db_filename, 'rb')) as fh:
                self.db = pd.read_pickle(fh)
        else:
            self.db = pd.DataFrame(columns=['date', 'protocol', 'T0', 'T1'])

    def update_database(self):
        """
        Write the database
        """
        
        if self.db is not None:
            self.db.to_pickle(self.db_filename)

    def compute_PSC_IV(self, protocolName, plot=True, savetimes=False):
        """
        Direct the analysis
        Uses the beginning of the protocol name to select which analysis to use
        
        Parameters:
        protocolName : str 
            Name of the protocol to analyze, underneath the datapath
        
        plot : boolean (default: True)
            Flag to plot data
        
        """
        self.AR.setProtocol(self.datapath)  # define the protocol path where the data is
        self.setup(clamps=self.AR)
        self.read_database(f"{protocolName:s}.p")

        if self.AR.getData():  # get that data.
            ok = False
            if protocolName.startswith('Stim_IO'):
                ok = self.analyze_IO()
            elif protocolName.startswith('VC-EPSC_3'):
                ok = self.analyze_VDEP()
            elif protocolName.startswith('PPF'):
                ok = self.analyze_PPF()
            if not ok:
                print('Failed on protocol: ', self.datapath, protocolName)
                return False
            if plot:
                self.plot_vciv()
            if savetimes:
                date = make_key(self.datapath)
 
                if date not in self.db['date'].tolist():
                    self.db.loc[len(self.db)] = [date, protocolName, self.T0, self.T1]
                    print('new date added')
                else:
                    self.db.loc[date, 'date'] = date
                    self.db.loc[date, 'protocol'] = protocolName
                    self.db.loc[date, 'T0'] = self.T0
                    self.db.loc[date, 'T1'] = self.T1
                    print('old date data updated')
                self.update_database()
                # print('db head: ', self.db.head())
            return True
        else:
            return False

    def get_stimtimes(self):
        """
        This should get the stimulus times. Right now, it does nothing
        """
        pass

    def set_baseline_times(self, baseline):
        """
        baseline: 2-element list or numpy array
        """
        if len(baseline) != 2:
            raise ValueError('Baseline must be a 2-element array')
        if isinstance(baseline, list):
            baseline = np.array(baseline)
        self.baseline = np.sort(baseline)

    def get_baseline(self):
        """ Return the mean values in the data over the baseline region.
        """
        bl = self.mean_I_analysis(region=self.baseline, reps=[0])
        return bl
        
    def analyze_IO(self, rmpregion=[0., 0.05], protocolName=None, device='Stim0'):
        """Analyze in input=output function for a specific device
        
        """
        ptrain = self.AR.getStim(device)
        # stim dict in ptrain will look like:
        # {'start': [0.05, 0.1], 'duration': [0.0001, 0.0001],
        # 'amplitude': [0.00025, 0.00025], 'npulses': [2], 'period': [0.05], 'type': ['pulseTrain']}
        # try:
        dd  = self.AR.getDeviceData(device=device, devicename='command')
        if dd is None:
            return False
        # except:
        #     print('Failed to get device data from Stim0')
        #     return False
        filekey = Path(make_key(self.datapath))
        # check the db to see if we have parameters already
        dfiles = self.db['date'].tolist()
        if filekey in dfiles:
            delay = self.db.loc[filekey, 'date']['T0']
            t1    = self.db.loc[filekey, 'date']['T1']
            width = t1-delay
        else:
            delay = 1.0*1e-3
            width = 15.0*1e-3            
        reps = self.AR.sequence[('protocol', 'repetitions')]
        stim_dt = np.diff(ptrain['start'])
        stim_I = [ptrain['amplitude'][0]]
        mode = '?'
        if not (device, 'command.PulseTrain_amplitude') in self.AR.sequence.keys():
            raise ValueError('Cannot find PulseTrain_amplitude in stimulus command')
            
        stim_I = self.AR.sequence[(device, 'command.PulseTrain_amplitude')]

        baseline = []
        meani = []
        stimamp = []
        stimintvl = []
        cmdv = []
        self.sign = -1
        self.i_mean = []
        self.set_baseline_times(rmpregion)
        self.analysis_summary['iHold'] = []
        self.i_mean = []
        self.analysis_summary[f'PSP_IO'] = [[]]*len(ptrain['start'])
        bl = self.mean_I_analysis(region=self.baseline, mode='baseline', reps=[0])
        for i in range(len(ptrain['start'])):
            pdelay = ptrain['start'][i] + delay
            if i == 0 and self.update_regions:
                rgn = self.set_region([ptrain['start'][i], ptrain['start'][i]+0.050], baseline=bl)
            else:
                rgn = [delay, delay+width]
            self.T0 = rgn[0]
            self.T1 = rgn[1]
            print('pdelay: ', pdelay, self.baseline)
            i_mean = self.mean_I_analysis(region=np.array(rgn)+ptrain['start'][i], mode='min', baseline=bl, reps=reps)
            if i_mean is None:
                return False
            self.analysis_summary[f'PSP_IO'][i] = self.sign*1e12*i_mean
            cmdv.append(self.i_mean_cmd)
            stimamp.append(ptrain['amplitude'][i])
            stimintvl.append(ptrain['period'][0])

        self.analysis_summary['psc_stim_amplitudes'] = 1e6*np.array(stim_I)
        self.analysis_summary['psc_intervals'] = np.array(stimintvl)
        self.analysis_summary['ppf_dt'] = np.array(stim_dt)
        self.analysis_summary['stim_times'] = ptrain['start']
        self.analysis_summary['window'] = [self.T0, self.T1]
        return True

    def analyze_VDEP(self, rmpregion=[0., 0.05], protocolName=None, device='Stim0'):
        """
        Analyze the voltage-dependence of EPSCs 
        
        When selecting the analysis window, choose a window that encompases
        the peak of the inward EPSC in the negative voltage range.
        Do not try to balance the trace (the slope should be turned off)
        
        Parameters
        ----------
        rmpregion : 2 element list (default: [0., 0.05])
            The region of the trace used to measure the resting membrane potential, 
            in seconds. 
        protocolName : str (default: None)
            The name of the protocol (not used here)
        device : str (default: 'Stim0')
            The name of the stimulus device 
        """
        print('\n'+'******'*4)
        ptrain = self.AR.getStim(device)
        dt = self.Clamps.sample_interval
        # stim dict in ptrain will look like:
        # {'start': [0.05, 0.1], 'duration': [0.0001, 0.0001],
        # 'amplitude': [0.00025, 0.00025], 'npulses': [2], 'period': [0.05], 'type': ['pulseTrain']}
        dd  = self.AR.getDeviceData(device=device, devicename='command')
        # print('ptrain: ', ptrain)
        # print('dd: ', dd)
        reps = self.AR.sequence[('protocol', 'repetitions')]
        stim_dt = np.diff(ptrain['start'])
        stim_I = [ptrain['amplitude'][0]]
        mode = '?'
        if not ('MultiClamp1', 'Pulse_amplitude') in self.AR.sequence.keys():
            raise ValueError('Cannot find (MultiClamp1, Pulse_amplitude) in stimulus command')
            
        stim_V = self.AR.sequence[('MultiClamp1', 'Pulse_amplitude')]
        filekey = make_key(self.datapath)
        # check the db to see if we have parameters already
        dfiles = self.db['date'].tolist()

        if filekey in dfiles:
            # print(self.db.loc[self.db['date'] == filekey])
            # print(self.db.head())
            delays = self.db.loc[self.db['date'] == filekey]['T0'].values
            t1s    = self.db.loc[self.db['date'] == filekey]['T1'].values
            if isinstance(delays, np.ndarray) and len(delays) > 1:
                delay = delays[0]
            else:
                delay = delays
            if isinstance(t1s, np.ndarray) and len(t1s) > 1:
                t1 = t1s[0]
            else:
                t1 = t1s
            print('delay from file', delay, t1)
        else:
            delay = 1.0*1e-3
            t1 = (20-1.0)*1e-3
            print('auto delay', delay, t1)
        ndelay = self.NMDA_delay
        nwidth = 0.0025
        bl_region = [ptrain['start'][0]-0.060, ptrain['start'][0]-0.010]  # time just before stimulus
        baseline = []
        self.baseline = bl_region
        meani = []
        stimamp = []
        stimintvl = []
        cmdv = []
        self.sign = 1
        self.i_mean = []
        # self.set_baseline_times(rmpregion)
        self.analysis_summary['iHold'] = []
        self.analysis_summary[f'PSP_VDEP_AMPA'] = [[]]*len(ptrain['start'])
        self.analysis_summary[f'PSP_VDEP_NMDA'] = [[]]*len(ptrain['start'])
        bl = self.mean_I_analysis(region=bl_region, mode='baseline', reps=[0])
        # print('bl_region: ', bl_region)
        
        rgn = [delay, t1]
        # print('rgn: ', rgn)
        if self.update_regions:
            rgn = self.set_region([ptrain['start'][0], ptrain['start'][0]+self.NMDA_delay+0.010], baseline=bl, slope=True)
        self.T0 = float(rgn[0])
        self.T1 = float(rgn[1])

        if rgn[0] > 0.012:
            rgn[0] = 0.004
        rgn[1] = 0.20
        slope_region = rgn
        self.T0 = float(rgn[0])
        self.T1 = float(rgn[1])
        print('t0, t1: ', self.T0, self.T1)
        # two pass approach:
        # 1 find min, and look at the most negative traces (-100 to -60) to get the time of the peak
        # 2. average those times and make a new window
        # 3. use the new window to do the analysis by taking the mean in a 1msec wide window
        #    around the mean time
        # print(delay, t1)
        slope_region=np.array(slope_region)+ptrain['start'][0]
        print('slope region: ', slope_region)

        cmds = np.array(self.V_cmd)+self.AR.holding+self.JunctionPotential
        bl = self.mean_I_analysis(region=[ptrain['start'][0]+self.T0-0.0005, ptrain['start'][0]+self.T0], mode='baseline', reps=[0])

        data1, tb = self.get_traces(region=slope_region,
            trlist=None, baseline=bl, intno=0, nint=1, reps=reps, slope=False)
        # self.plot_data(tb, data1)
   
        ind = np.argmin(np.fabs(cmds-self.AMPA_voltage))

        self.T1 = self.T0 + 0.010
        print('p1min: ', self.T0)

        p1delay = ptrain['start'][0] + self.T0
        p1end = ptrain['start'][0] + self.T1  # note that this is a narrow
        
        nmdelay = ptrain['start'][0] + ndelay            
        i_mean = self.mean_I_analysis(region=[p1delay, p1end], mode='min',
                                    baseline=bl, reps=reps, slope=False)
        print('IMEAN ARGMIN: ', i_mean, self.i_argmin)
        if i_mean is None:
            return False
        if len(self.i_argmin) < 1:
            return False
        mintime = self.i_argmin[ind]*dt  # get AMPA peak index in the window
        print(f'AMPA mintime @ {self.AMPA_voltage*1e3:.1f} mV: {mintime*1e3:.3f} ms')

        # values for nmda analysis are currently fixed
        i_nmda_mean = self.mean_I_analysis(region=[nmdelay-nwidth, nmdelay+nwidth], mode='mean',
                                    baseline=bl, reps=reps, slope=False)

        self.analysis_summary[f'PSP_VDEP_AMPA'][0] = self.sign*i_mean
        self.analysis_summary[f'PSP_VDEP_NMDA'][0] = self.sign*i_nmda_mean
        stimamp.append(ptrain['amplitude'][0])
        stimintvl.append(ptrain['period'][0])
        
        # print('ampa window & mean: ', [p1delay, p1end], i_mean)
        # print('nmda window & mean: ', [nmdelay-nwidth, nmdelay+nwidth], i_nmda_mean)

        # find -80 and +30 voltage indices (so we can save them and save the data)
        iAMPA = np.argmin(np.fabs(-self.AMPA_voltage+cmds))
        iNMDA = np.argmin(np.fabs(-self.NMDA_voltage+cmds))
        # print(iAMPA, iNMDA)
        # print('-90 mV found closest command: ', cmds[iAMPA])
        # print('+50 mV found closest command: ', cmds[iNMDA])
        if data1 is None or iNMDA >= data1.shape[0]:

            self.analysis_summary['Vindices'] = {'vAMPA': np.nan, 'vNMDA': np.nan}
            self.analysis_summary['NMDAAMPARatio'] = np.nan
            self.analysis_summary['AMPA_NMDA_traces'] = {'T': None, 'iAMPA': None, 'iNMDA': None}
        else:
            # print('data1 shape: ', data1.shape, iAMPA, iNMDA, cmds[iAMPA], cmds[iNMDA])
            # print(self.analysis_summary[f'PSP_VDEP_AMPA'])
            self.analysis_summary['Vindices'] = {'-90': iAMPA, '50': iNMDA}
            self.analysis_summary['NMDAAMPARatio'] = self.analysis_summary[f'PSP_VDEP_NMDA'][0][iNMDA]/self.analysis_summary[f'PSP_VDEP_AMPA'][0][iAMPA]
            self.analysis_summary['AMPA_NMDA_traces'] = {'T': tb, 'iAMPA': data1[iAMPA], 'iNMDA': data1[iNMDA]}
        self.analysis_summary['meas_times'] = {'tAMPA': mintime, 'tNMDA': ndelay}
        self.analysis_summary['psc_stim_amplitudes'] = np.array(stim_I)
        self.analysis_summary['psc_intervals'] = np.array(stimintvl)
        self.analysis_summary['stim_times'] = ptrain['start']
        self.analysis_summary['Vcmd'] = cmds
        self.analysis_summary['Window'] = [self.T0, self.T1]
        return True

    def plot_data(self, tb, data1, title=''):
        """
        Quick plot of data for testing purposes
        
        Parameters
        ----------
        tb : np.array (no default)
            the time base (one dimension)
        
        data1 : np.array (no default)
            The data, can be [m traces x npoints]
        
        title : str (default: '')
            A title to put on the plot
        
        Return
        ------
        Nothing
        """
        
        f, ax = mpl.subplots(1)
        ax = np.array(ax).ravel()
        ie = data1.shape[1]
        it = tb.shape[0]
        if ie > it:
            ie = it
        if it > ie:
            it = ie
        print(it, ie)
        for i in range(data1.shape[0]):
            ax[0].plot(tb[:it], data1[i,:ie])
        ax[0].set_title(str(self.datapath).replace('_', '\_')+' '+title, fontsize=8)
        mpl.show()
        
    def analyze_PPF(self, rmpregion=[0., 0.05], protocolName=None, device='Stim0'):
        """
        Analyze paired-pulse facilitiation

        Parameters
        ----------
        rmpregion : 2 element list (default: [0., 0.05])
            The region of the trace used to measure the resting membrane potential, 
            in seconds. 
        protocolName : str (default: None)
            The name of the protocol (not used here)
        device : str (default: 'Stim0')
            The name of the stimulus device
        """
        # self.rmp_analysis(region=rmpregion)
        #        self.tau_membrane(region=tauregion)
        # r0 = self.Clamps.tstart + 0.9*(self.Clamps.tend-self.Clamps.tstart) #
        # print(dir(self.Clamps))
        ptrain = self.AR.getStim(device)
        # stim dict in ptrain will look like:
        # {'start': [0.05, 0.1], 'duration': [0.0001, 0.0001], 'amplitude': [0.00025, 0.00025], 'npulses': [2], 'period': [0.05], 'type': ['pulseTrain']}
        dd  = self.AR.getDeviceData(device=device, devicename='command')

        reps = self.AR.sequence[('protocol', 'repetitions')]
        stim_I = [ptrain['amplitude'][0]]
        if not (device, 'command.PulseTrain_period') in self.AR.sequence.keys():
            raise ValueError('Cannot find PulseTrain_period in stimulus command')
        
        stim_dt = self.AR.sequence[(device, 'command.PulseTrain_period')]
        mode = 'PPF'
        print(mode)
        delay = 1.0*1e-3
        width = 20.0*1e-3
        ndelay = 1.0*1e-3

        filekey = make_key(self.datapath)
        # check the db to see if we have parameters already
        dfiles = self.db['date'].tolist()
        # print('dfiles: ', dfiles)
        # print('filekey: ', filekey)
        # print('file in dfiles?: ', filekey in dfiles)
        if filekey in dfiles:
            # print(self.db.loc[self.db['date'] == filekey])
            # print(self.db.head())
            delays = self.db.loc[self.db['date'] == filekey]['T0'].values
            t1s    = self.db.loc[self.db['date'] == filekey]['T1'].values
            if isinstance(delays, np.ndarray) and len(delays) > 1:
                delay = delays[0]
            else:
                delay = delays
            if isinstance(t1s, np.ndarray) and len(t1s) > 1:
                t1 = t1s[0]
            else:
                t1 = t1s
            print('delay from file', delay, t1)
        else:
            delay = 1.0*1e-3
            t1 = (20-1.0)*1e-3
            print('auto delay', delay, t1)
        nwidth = width
        baseline = []
        meani = []
        stimamp = []
        stimintvl = []
        cmdv = []
        self.sign = 1
        self.set_baseline_times(rmpregion)
        self.i_mean = []
        self.analysis_summary['iHold'] = []
        self.i_mean = []
        self.analysis_summary[f'PPF'] = [[]]*len(stim_dt)
        # print(ptrain)
        self.i_mean = []
        ppftr = {}
        
        rgn = [delay, t1]
        # print('rgn: ', rgn)
        if self.update_regions:
            rgn = self.set_region([ptrain['start'][0], ptrain['start'][0]+0.020], baseline=bl)
        self.T0 = float(rgn[0])
        self.T1 = float(rgn[1])
        window = self.T1-self.T0
        p1delay = ptrain['start'][0] + self.T0 # first stim of pair
        p1end = ptrain['start'][0] + self.T1
        print('p1delay, p1end: ', p1delay, p1end)
        pulse2 = ptrain['start'][0] + stim_dt[0]
        print('ptrain start+dt: ', p1end, pulse2)
        if p1end >= pulse2:  # would run into next stimulus!
            delt = p1end - pulse2
            self.T1 -= delt  # make it msec shorter
            p1end = ptrain['start'][0] + self.T1
            print('new p1end: ', p1end)
        bl = self.mean_I_analysis(region=self.baseline, mode='baseline', reps=[0])

        for i in range(len(stim_dt)):
            p2delay = p1delay + stim_dt[i] # second stim of pair
            p2end = p2delay + self.T1
            # print('p1, p2, t1: ', p1delay, p2delay, self.T1)
            # print('T1: ', float(self.T1))
            print('p1delay, end: ', p1delay, p1end, stim_dt[i])
            i_mean_ref = self.mean_I_analysis(region=[p1delay, p1end], mode='min', baseline=bl,
                        intno=i, nint=len(stim_dt), reps=reps)
            i_ref = self.i_data
            tb_ref = self.i_tb
            if i_mean_ref is None:
                return False
            i_mean = self.mean_I_analysis(region=[p2delay, p2end], mode='min', baseline=bl,
                intno=i, nint=len(stim_dt), reps=reps)
            i_p2 = self.i_data
            tb_p2 = self.i_tb
            # i_mean -= bl.mean()
            # i_mean_ref -= bl.mean()
            cmdv.extend(self.i_mean_cmd)
            stimamp.extend(ptrain['amplitude'])
            stimintvl.append(stim_dt[i])
            self.analysis_summary[f'PPF'][i] = i_mean/i_mean_ref
            ppftr[stim_dt[i]] = {'TRef': tb_ref, 'IRef': i_ref-i_ref[0], 'TPP2': tb_p2, 'IPP2': i_p2-i_ref[0]}
            print('iref, ip2: ', i, ppftr[stim_dt[i]]['IRef'][0]*1e9, ppftr[stim_dt[i]]['IPP2'][0]*1e9)
        self.analysis_summary['PPF_traces'] = ppftr
        self.analysis_summary['psc_stim_amplitudes'] = np.array(stim_I)
        self.analysis_summary['psc_intervals'] = np.array(stimintvl)
        self.analysis_summary['ppf_dt'] = np.array(stim_dt)
        self.analysis_summary['stim_times'] = ptrain['start']
        self.analysis_summary['window'] = [self.T0, self.T1]
        # f, ax = mpl.subplots(1,1)
        # ax = np.array(ax).ravel()
        # for i in range(len(stim_dt)):
        #     dt = stim_dt[i]
        #     ax[0].plot(ppftr[dt]['TRef'], ppftr[dt]['IRef'], 'k')
        #     ax[0].plot(ppftr[dt]['TPP2'], ppftr[dt]['IPP2'], 'k')
        # mpl.show()
        return True


    def set_region(self, region=None, baseline=None, slope=True):
        if region is None:
            raise ValueError("PSPSummary, set_region requires a region beginning and end to measure the current")
    
        data1 = self.Clamps.traces['Time': region[0]:region[1]]
        if baseline is None:
            baseline = [0.]

        tb = np.arange(0, data1.shape[1]*self.Clamps.sample_interval, self.Clamps.sample_interval)
        data1 = data1.view(np.ndarray)
        newCP = CP.CursorPlot(str(self.datapath))
        setline = True

        # for i in range(data1.shape[0]):
        newCP.plotData(x=tb, y=np.array([data1[i]-baseline[i] for i in range(data1.shape[0])])*1e12, setline=setline, slope=slope)
            # setline = False # only do on first plot
    
        if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
            QtGui.QApplication.instance().exec_()
         
        self.T0, self.T1 = newCP.selectedRegion
        if self.T0 == None:
            return(None)
        return(newCP.selectedRegion)
    

    def mean_I_analysis(self, region=None, t0=0.5, mode='mean', baseline=None, intno=0, nint=1, reps=[0], slope=True, slopewin=None):
        """
        Get the mean current in a window
        Works with the current Clamps object
        
        Parameters
        ----------
        region : tuple, list or numpy array with 2 values (default: None)
            start and end time of a trace used to measure the RMP across
            traces. Note that if slopewin is set, it may replace the region
        
        t0 : float (default=0.5)
            start time for the mean current analysis
        
        mode : str (default='mean')
            How to measure the value (valid values: 'mean', 'baseline' both compute mean,
            'min' gives the minimum current in the window.
        
        baseline: np.array (default None)
            an array of precomputed baseline values to subtract from the data; one value
            per trace
        
        intno : int (default=0)
            first trace to do in a group
        
        nint : int (default=1)
            # of traces to skip (e.g., for repeats or different values across the array)
        
        reps: list (default=[0])
            # of repetitions (used to reshape data in computation)
        
        slope: bool (default=True)
            set to subtract a slope from trace
        
        slopewin: list or np.array of 2 elements (default=None)
            Time window to use to compute slope, [start, stop], in seconds
        
        Return
        ------
        the mean current in the window
        
        Stores computed mean current in the variable "name".
        """
        if region is None:
            raise ValueError("PSPSummary, mean_I_analysis requires a region beginning and end to measure the current")
        analysis_region = region.copy()
        if slope and slopewin is not None:
            region = slopewin
        
        data1 = self.Clamps.traces['Time': region[0]:region[1]]
        # data2 = self.Clamps.traces.view(np.ndarray)
        self.V_cmd = self.Clamps.commandLevels

        tb = np.arange(0, data1.shape[1]*self.Clamps.sample_interval, self.Clamps.sample_interval)
        # tb2 = np.arange(0, data2.shape[1]*self.Clamps.sample_interval, self.Clamps.sample_interval)
        data1 = data1.view(np.ndarray)
        if baseline is not None:
            data1 = np.array([data1[i]-baseline[i] for i in range(data1.shape[0])])
            # data2 = np.array([data2[i]-baseline[i] for i in range(data2.shape[0])])

        # subtract the "baseline" from the beginning of the interval to the end. 
        if slope:
            slrgn = region
            if slopewin is not None:
                slrgn = slopewin
            data1 = self.slope_subtraction(tb, data1, region, mode=mode)
        # if not slope and slopewin is not None: # just first slope point to align current
        #     data1 = self.slope_subtraction(tb, data1, region, mode='point')
        print('slope, slopewin: ', slope, slopewin, mode)
        
        
        sh = data1.shape
        nreps = len(reps)
        if nint > 1:
            dindx = range(intno, sh[0], nint)
            data1 = data1[dindx,:,:]

        self.i_mean_index = None
        # print('imean data shape: ', data1.shape)
        self.i_data = data1.mean(axis=0)
        self.i_tb = tb+region[0]
        
        nx = int(sh[0]/len(reps))
        
        if mode in ['mean', 'baseline']:
            # print('calc mean: ')
            # print('    data1.shape: ', data1.shape)
            i_mean = data1.mean(axis=1)  # all traces, average over specified time window
            # print('    i_mean.shape: ', i_mean.shape)
            if nint == 1:
                nx = int(sh[0]/len(reps))
                try:
                    i_mean = np.reshape(i_mean, (len(reps), nx))  # reshape by repetition
                except:
                    return i_mean
            # print('    _after possible reshape: ', i_mean.shape)
            i_mean = i_mean.mean(axis=0)  # average over reps
            
            # print('     i_mean averaged over reps: ', i_mean)
            # print('mean data windows : ', self.i_data.shape)
            return i_mean
     
        elif mode == 'min':
            dfilt = scipy.signal.savgol_filter(data1, 5, 2, axis=1, mode='nearest')
            ist = int((analysis_region[0]-t0)/self.Clamps.sample_interval)
            ien = int((analysis_region[1]-t0)/self.Clamps.sample_interval)
            # print(ist, ien)

            dfw = [[]]*nreps
            nvs = int(sh[0]/nreps)
            for i in range(nreps):
                dfw[i] = dfilt[i*nvs:i*nvs + nvs,: ]
            dfw = np.array(dfw)
            # dfw = dfw.reshape((nreps, -1, int(sh[0]/nreps)))
            dfw = dfw.mean(axis=0)
            # for i in range(dfw.shape[0]):
            #     mpl.plot(dfw[i])
            # mpl.show()
            
            # print(dfw.shape, ist, ien)
            i_mean = dfw[:, ist:ien].min(axis=1)  # all traces, average over specified time window
            # self.i_argmin = dfw[:, ist:ien].argmin(axis=1) +ist
            # print('imean shape: ', i_mean.shape)
            # print('mean values: ', i_mean)
            # print('iargmin: ', self.i_argmin)

            return(i_mean)
            # except:
            #     return None
        # elif mode == 'getmintime':
        #     cmds = np.array(self.V_cmd)+self.AR.holding+self.JunctionPotential
        #     ind = np.where((cmds >= -0.12) & (cmds <= -0.07))
        #     # f, ax = mpl.subplots(1)
        #     # ax = np.array(ax).ravel()
        #     # for i in range(data1.shape[0]):
        #     #     mpl.plot(tb, data1[i])
        #     # mpl.show()
        #     print('ind: slope ', ind, slope)
        #     # dfilt = scipy.signal.savgol_filter(data1, 5, 2, axis=1, mode='nearest')
        #     try:
        #         print(data1.shape)
        #         print(self.V_cmd + self.AR.holding + self.JunctionPotential)
        #         print(data1[:, 0])
        #         i_mins = data1.argmin(axis=1)  # all traces, average over specified time window
        #         print(ind, i_mins)
        #         # exit()
        #     except:
        #         return None          
        


    def slope_subtraction(self, tb, data1, region, mode='mean'):
        """
        Subtract a slope from the data; the slope is calculated from a time region
        
        Parameters
        ----------
        tb : np.array 
            time base, in seconds. Must be of same size as data1 2nd dimension
        data1 : np.array
            data array; 2 dimensional (traces x time)
        region : 2 element list or np.array
            time region for computation of the slope, in seconds
        mode : str (default: 'mean')
            Either 'point' (does nothing to the data)
                or 'mean' 
        Return
        ------
            slope-subtracted data
        """
        dt = tb[1]-tb[0]
        minX = 0 #int((region[0])/dt)
        maxX = int((region[1]-region[0])/dt)
        if mode is 'point':  # do nothing... 
            # for i in range(data1.shape[0]):
            #     data1[i,:] -=  data1[i,0]
            return data1
            
        for i in range(data1.shape[0]):
            x0 = list(range(minX,minX+3))
            ml = maxX
            x0.extend(list(range(ml-10, ml)))
            fdx = np.array([tb[j] for j in x0])
            fdy = np.array([data1[i][j] for j in x0])
            pf = np.polyfit(fdx, fdy, 1)
            bline = np.polyval(pf, tb)
            if bline.shape[0] > data1[i].shape[0]:
                bline = bline[:data1[i].shape[0]]
            if mode != 'baseline':
                data1[i,:] -= bline
        return data1

    def get_traces(self, region=None, trlist=None, baseline=None, order=0, intno=0, nint=1, reps=[0], mode='baseline', slope=True):
        """
        Get the mean current (averages) in a window
        
        Parameters
        ----------
        region : tuple, list or numpy array with 2 values (default: None)
            start and end time of a trace used to measure the RMP across
            traces.
        
        Return
        ------
        Nothing
        
        Stores computed mean current in the variable "name".
        """
        if region is None:
            raise ValueError("PSPSummary, mean_I_analysis requires a region beginning and end to measure the current")
        
        data1 = self.Clamps.traces['Time': region[0]:region[1]]

        tb = np.arange(0, data1.shape[1]*self.Clamps.sample_interval, self.Clamps.sample_interval)
        data1 = data1.view(np.ndarray)
        nreps = len(reps)
        sh = data1.shape
        # print('data1 initial shape: ', sh)
        # print('tb shape: ', tb.shape)
        # print('nint, nreps, order: ', nint, nreps, order)

        if nint > 1:
            # if order == 0:
                dindx = range(intno, sh[0], nint)
                data1 = data1[dindx]
            # else:
            #     pass
                # dindx = range(sh[0], intno, nreps)
                # data1 = data1[dindx]
        # print('unshaped data1: ', data1.shape)
        # print('slope: ', slope, region)
        # subtract the "baseline" from the beginning of the interval to the end. 
        if slope:
            data1 = self.slope_subtraction(tb, data1, region, mode=mode)
            
        if baseline is not None:
            data1 = np.array([data1[i]-baseline[i] for i in range(data1.shape[0])])
        
        nx = int(sh[0]/nreps)
        if nx < 13:
            nreps = 1

        if order == 0 and nreps > 1:
            try:
                print('gettraces reshaping: data shape, reps, nx, nint: ', data1.shape, nreps, nx, data1.shape[0]/nreps, nint)
                data2 = np.reshape(data1, (len(reps), nx,   -1))
            except:
                print('Failed to reshape: data shape, reps, nx: ', data1.shape, len(reps), nx, data1.shape[0]/len(reps))
                if data1.shape[0] > 1:
                    data2 = data1
                    return data2, tb
                else:
                    return None, None
        elif order == 1 or nreps == 1:
            data2 = data1 # np.reshape(data1, (len(reps), nx,   sh[1]))
        # print('data 1 reshaped: data2: ', data2.shape)
            
        ### check data by plotting
        # prop_cycle = mpl.rcParams['axes.prop_cycle']
        # colors = prop_cycle.by_key()['color']
        #
        # colors = ['r', 'g', 'b', 'c', 'y', 'm', 'k']
        # if len(colors) > data1.shape[0]:
        #     colors = colors[:data1.shape[0]]
        # color_cycle = cycler(c=colors)
        # f, ax = mpl.subplots(1, 1)
        # k = 0
        # print(data1.shape, nx, reps)
        # if order == 0:
        #     for j in range(len(reps)):
        #         for i, sty in zip(range(nx), cycle(color_cycle)):
        #     # for i in range(data1.shape[0]):
        #             ax.plot(tb, data2[k], linewidth=0.5, **sty)
        #             k += 1
        # elif order == 1:
        #     sty = zip(range(nint), cycle(color_cycle))
        #     for i in range(nint):
        #     # for i in range(data1.shape[0]):
        #             ax.plot(tb, data2[k], colors[intno], linewidth=0.5)
        #             k += 1
        # print('nx: ', nx, '  reps: ', reps, '  len reps: ', len(reps))
        ###
        
        

            #data1 = np.reshape(data1, (nx, len(reps), -1))
            
        # print('reshaped data: ', data1.shape)
        data2 = data2.mean(axis=0)
        # print('mean data data: ', data1.shape)
        
        ### check data by plotting
        # print('data2 mean shape: ', data2.shape)
        # if order == 0:
        #     for i, sty in zip(range(data2.shape[0]), cycle(color_cycle)):
        #         ax.plot(tb, data2[i], '--', **sty)
        # elif order == 1:
        #     ax.plot(tb, data2, '--k')
        # mpl.show()

        ###
        
        # if nint == 1:
        #     nx = int(sh[0]/len(reps))
        #     try:
        #         i_mean = np.reshape(i_mean, (len(reps), nx))  # reshape by repetition
        #     except:
        #         return None

        # i_mean = i_mean.mean(axis=0)  # average over reps
        return data2, tb
                

    def vcss_analysis(self, region=None):
        """
        compute steady-state IV curve - from the mean current 
        across the stimulus set over the defined time region 
        (this usually will be the last half or third of the trace)
        
        Parameters
        ----------
        region : list or tuple
            Start and end times for the analysis
        """
        
        data1 = self.Clamps.traces['Time': region[0]:region[1]]
        icmds = EM.MetaArray(self.Clamps.cmd_wave,  # easiest = turn in to a matching metaarray... 
            info=[{'name': 'Command', 'units': 'A',
             'values': np.array(self.Clamps.clampValues)},
             self.Clamps.traces.infoCopy('Time'), self.Clamps.traces.infoCopy(-1)])
        self.vcss_vcmd = icmds['Time': region[0]:region[1]].mean(axis=1)
        self.r_in = np.nan
        self.analysis_summary['Rin'] = np.nan
        self.vcss_v = []
        if data1.shape[1] == 0 or data1.shape[0] == 1:
            return  # skip it

        ntr = len(self.Clamps.traces)

        self.vcss_Im = data1.mean(axis=1)  # steady-state, all traces
        self.analysis_summary['Rin'] = np.NaN
        #        self.Clamps.plotClampData()
        
        isort = np.argsort(self.vcss_vcmd)
        self.vcss_Im= self.vcss_Im[isort]
        self.vcss_vcmd = self.vcss_vcmd[isort]
        bl = self.vcbaseline[isort]
        self.vcss_bl = bl
        # compute Rin from the SS IV:
        # this makes the assumption that:
        # successive trials are in order so we sort above
        # commands are not repeated...
        if len(self.vcss_vcmd) > 1 and len(self.vcss_v) > 1:
            pf = np.polyfit(self.vcss_vcmd, self.vcss_v, 3, rcond=None, full=False, w=None, cov=False)
            pval = np.polyval(pf, self.vcss_vcmd)
            #print('pval: ', pval)
            slope = np.diff(pval) / np.diff(self.vcss_vcmd)  # local slopes
            imids = np.array((self.vcss_vcmd[1:] + self.vcss_vcmd[:-1]) / 2.)
            self.rss_fit ={'I': imids, 'V': np.polyval(pf, imids)}
            #print('fit V: ', self.rss_fit['V'])
            #slope = slope[[slope > 0 ] and [self.vcss_vcmd[:-1] > -0.8] ] # only consider positive slope points
            l = int(len(slope)/2)
            maxloc = np.argmax(slope[l:]) + l
            self.r_in = slope[maxloc]
            self.r_in_loc = [self.vcss_vcmd[maxloc], self.vcss_v[maxloc], maxloc]  # where it was found
            minloc = np.argmin(slope[:l])
            self.r_in_min = slope[minloc]
            self.r_in_minloc = [self.vcss_vcmd[minloc], self.vcss_v[minloc], minloc]  # where it was found
            self.analysis_summary['Rin'] = self.r_in*1.0e-6


    def plot_vciv(self):
        """
        Plot the current voltage-clamp IV function
        """
        P = PH.regular_grid(2 , 2, order='columns', figsize=(8., 6.), showgrid=False,
                        verticalspacing=0.1, horizontalspacing=0.1,
                        margins={'leftmargin': 0.12, 'rightmargin': 0.12, 'topmargin': 0.08, 'bottommargin': 0.1},
                        labelposition=(-0.12, 0.95))
        (date, sliceid, cell, proto, p3) = self.file_cell_protocol(self.datapath)
        
        P.figure_handle.suptitle(os.path.join(date, sliceid, cell, proto).replace('_', '\_'), fontsize=12)
        bl = self.get_baseline()
        if "PPF" in self.analysis_summary.keys():
            maxt = 250.
        else:
            maxt = 150.
        for i in range(self.AR.traces.shape[0]):
            P.axdict['A'].plot(self.AR.time_base*1e3, (self.AR.traces[i,:]-bl[i])*1e12, 'k-', linewidth=0.5)
        if 'PSP_VDEP' in self.analysis_summary.keys():
            P.axdict['A'].set_xlim(self.analysis_summary['stim_times'][0]*1e3-10, self.analysis_summary['stim_times'][0]*1e3+50)
            # P.axdict['A'].set_ylim(-2500, 2500)
        else:
            P.axdict['A'].set_xlim(40, maxt)
            P.axdict['A'].set_ylim(-2000, 500)
        # PH.talbotTicks(P.axdict['A'], tickPlacesAdd={'x': 0, 'y': 0}, floatAdd={'x': 0, 'y': 0})
        P.axdict['A'].set_xlabel('T (ms)')
        P.axdict['A'].set_ylabel('I (pA)')

        if 'PSP_IO' in self.analysis_summary.keys(): # io function
            for i in range(len(self.analysis_summary['stim_times'])):
                try:
                    P.axdict['C'].plot(self.analysis_summary['psc_stim_amplitudes'],
                        np.array(self.analysis_summary[f'PSP_IO'][i]), linewidth=1, markersize=4)
                except:
                    print('Plot Failed on protocol: ', self.datapath, proto)
            P.axdict['C'].set_xlabel('Istim (microAmps)')
            P.axdict['C'].set_ylabel('EPSC I (pA)')
            PH.talbotTicks(P.axdict['C'], tickPlacesAdd={'x': 0, 'y': 0}, floatAdd={'x': 0, 'y': 0})
        elif 'PSP_VDEP' in self.analysis_summary.keys(): # io function
            for i in range(len(self.analysis_summary['stim_times'])):
                P.axdict['C'].plot(self.analysis_summary['Vcmd']*1e3,
                        self.sign*np.array(self.analysis_summary[f'PSP_VDEP'][i])*1e12, marker='o', linewidth=1, markersize=4)
                P.axdict['C'].plot(self.analysis_summary['Vcmd']*1e3,
                        self.sign*np.array(self.analysis_summary[f'PSP_VDEP_NMDA'][i])*1e12, marker='s', linewidth=1, markersize=4)
            P.axdict['C'].set_xlabel('V (mV)')
            P.axdict['C'].set_ylabel('EPSC I (pA)')
            PH.crossAxes(P.axdict['C'], xyzero=(-60., 0.))
            PH.talbotTicks(P.axdict['C'], tickPlacesAdd={'x': 0, 'y': 0}, floatAdd={'x': 0, 'y': 0})
        elif 'PPF' in self.analysis_summary.keys():
            for i in range(len(self.analysis_summary['stim_times'])):
                P.axdict['C'].plot(self.analysis_summary['ppf_dt']*1e3,
                            self.sign*np.array(self.analysis_summary[f'PPF']), linewidth=1, markersize=4)
                P.axdict['C'].set_xlim(0, 200.)
                P.axdict['C'].set_ylim(0, 2.0)
                PH.referenceline(P.axdict['C'], 1.0)
                P.axdict['C'].set_xlabel('Interval (ms)')
                P.axdict['C'].set_ylabel('PPF (R2/R1)')
                PH.talbotTicks(P.axdict['C'], tickPlacesAdd={'x': 0, 'y': 1}, floatAdd={'x': 0, 'y': 1})


        P.axdict['B'].set_xlabel('I (nA)')
        P.axdict['B'].set_ylabel('V (mV)')
        PH.talbotTicks(P.axdict['B'], tickPlacesAdd={'x': 1, 'y': 0}, floatAdd={'x': 2, 'y': 0})

        P.axdict['D'].set_xlabel('I (pA)')
        P.axdict['D'].set_ylabel('Latency (ms)')

        self.IVFigure = P.figure_handle
    
        if self.plot:
             mpl.show()

    def file_cell_protocol(self, filename):
        """
        file_cell_protocol breaks the current filename down and returns a
        tuple: (date, cell, protocol)
        
        Parameters
        ----------
        filename : str
            Name of the protocol to break down
        
        Returns
        -------
        tuple : (date, sliceid, cell, protocol, any other...)
            last argument returned is the rest of the path...
        """
        (p0, proto) = os.path.split(filename)
        (p1, cell) = os.path.split(p0)
        (p2, sliceid) = os.path.split(p1)
        (p3, date) = os.path.split(p2)
        return (date, sliceid, cell, proto, p3)
        

if __name__ == '__main__':
    """
    This is for testing - normally an instance of EPSC_analyzer would be
    created and these values would be filled in.
    """
    import matplotlib
    matplotlib.use('Qt5Agg')
    from matplotlib import rc
    #rc('font',**{'family':'sans-serif','sans-serif':['Arial']})
    #rcParams['font.sans-serif'] = ['Arial']
    #rcParams['font.family'] = 'sans-serif'
    rc('text', usetex=False)
    rcParams = matplotlib.rcParams
    rcParams['svg.fonttype'] = 'none' # No text as paths. Assume font installed.
    rcParams['pdf.fonttype'] = 42
    rcParams['ps.fonttype'] = 42
    rcParams['text.latex.unicode'] = True    
    # disk = '/Volumes/Pegasus/ManisLab_Data3'
    # disk = '/Volumes/PBM_005/data'
    disk = '/Users/pbmanis/Documents/Lab'
    middir = 'Kasten_Michael'
    middir = 'data'
    # middir = ''
    directory = 'Maness_PFC_stim'
    cell = '2019.03.19_000/slice_000/cell_001'
    # cell = '2019.03.19_000/slice_001/cell_000'
    
    ddc = Path(disk, middir, directory, cell)
    # protocol = 'Stim_IO_1_001'
    protocol = 'PPF_2_001'
    # protocol = 'VC-EPSC_3_ver2_003'
    fn = Path(ddc, protocol)
    PSC = PSCAnalyzer(fn)
    PSC.compute_PSC_IV(protocol[:-4], savetimes=True)
    
