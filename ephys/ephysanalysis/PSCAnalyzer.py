"""
Analyze EPSCs or IPSCs
Or EPSPs and IPSPs...

This module provides the following analyses:

1. Amplitudes from a train
2. Paired pulse facilitation for pulse pairs, and the first pair in a train.
3. Current-voltage relationship in voltage clamp measured over a time window

The results of the analysis are stored in the class variable analysis_summary

Note: if the analyzer is called with update_regions set True, then traces will be
sent to cursor_plot to get start and end times. (this might be broken now - need to test)

"""

import sys
from pathlib import Path
import os  #legacy
import scipy.signal
import pandas as pd
import lmfit
from collections import OrderedDict

from cycler import cycler
from itertools import cycle
import numpy as np

import ephysanalysis as EP
import ephysanalysis.metaarray as EM  # need to use this version for Python 3
import ephysanalysis.cursor_plot as CP
import pylibrary.plotting.plothelpers as PH
import matplotlib.pyplot as mpl
import matplotlib.colors
import seaborn as sns

import pyqtgraph as pg
from pyqtgraph.Qt import QtGui, QtCore
from pyqtgraph.Point import Point


def make_key(pathname):
    """
    Make a key string using the date, slice, cell and protocol from the path name
    """
    p = pathname.parts
    return(str('~'.join([p[i] for i in range(-4, 0)])))

class PSC_Fitter():
    """
    Provide fitting functions for PSCs:
    1. decay tau only
    2. PSC full fit (1-exp(tau_rise))^4 * exp(tau_fall)
    
    """
    def __init__(self):
        pass # nothing to do

    def _fcn_tau(self, params, x, data):
        """Model single exponential"""
        v = params.valuesdict()

        model = v['amp'] * np.exp(-x/v['tau_fall']) + v['DC']
        return model - data

    def fitTau(self):

        # create a set of Parameters
        params = lmfit.Parameters()
        params.add('amp', value=self.ptreedata.param('Initial Fit Parameters').param('amp').value(), min=-self.dmax, max=self.dmax)
        params.add('tau_fall', value=self.ptreedata.param('Initial Fit Parameters').param('taufall').value(), min=1e-4, max=1e-1)
        params.add('DC', value=self.ptreedata.param('Initial Fit Parameters').param('DC').value(), min=-1e3, max=1e3)
        
        t0 = self.T0.value()
        t1 = self.T1.value()
        it0 = int(t0/self.dt)
        it1 = int(t1/self.dt)
        if it0 > it1:
            t = it0
            it0 = it1
            it1 = t

        time_zero = int(self.time_zero/self.dt)
        print('timezero: ', time_zero, self.dataX[time_zero])
        # do fit, here with the default leastsq algorithm
        minner = lmfit.Minimizer(self._fcn_tau, params, fcn_args=(self.dataX[it0:it1]-self.dataX[time_zero], self.dataY[it0:it1]))
        self.fitresult = minner.minimize('leastsq')

        # calculate final result
        final = self.dataY[it0:it1] + self.fitresult.residual

        # write error report
        lmfit.report_fit(self.fitresult)


    def _fcn_EPSC(self, params, x, data):
        """Model EPSC"""
        v = params.valuesdict()

        model = v['amp'] * (((1. - np.exp(-x/v['tau_rise']))**4.0)*np.exp(-x/v['tau_fall'])) + v['DC']
        return model - data
    
    def fitEPSC(self):

        # create a set of Parameters
        params = lmfit.Parameters()
        params.add('amp', value=self.ptreedata.param('Initial Fit Parameters').param('amp').value(), min=-self.dmax, max=self.dmax)
        params.add('tau_rise', value=self.ptreedata.param('Initial Fit Parameters').param('taurise').value(), min=1e-4, max=1e-1)
        params.add('tau_fall', value=self.ptreedata.param('Initial Fit Parameters').param('taufall').value(), min=1e-4, max=1e-1)
        params.add('DC', value=self.ptreedata.param('Initial Fit Parameters').param('DC').value(), min=-1e3, max=1e3)
        dc = np.mean(self.dataY[0:10])
        params.add('DC', value=dc, min=dc-dc*1, max=dc+dc*1)
        t0 = self.T0.value()
        t1 = self.T1.value()
        it0 = int(t0/self.dt)
        it1 = int(t1/self.dt)
        if it0 > it1:
            t = it0
            it0 = it1
            it1 = t
        
        # do fit, here with the default leastsq algorithm
        time_zero = int(self.time_zero/self.dt)
        print('timezero: ', time_zero, self.dataX[time_zero])
        print(self.dataX[it0:it1]-self.time_zero)
        print(self.dataY[it0:it1])
        
        minner = lmfit.Minimizer(self._fcn_EPSC, params, fcn_args=(self.dataX[it0:it1]-self.dataX[time_zero], self.dataY[it0:it1]))
        self.fitresult = minner.minimize(method='least_squares', )

        # calculate final result
        final = self.dataY[it0:it1] + self.fitresult.residual

        # write error report
        lmfit.report_fit(self.fitresult)


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
                self.db = pd.read_pickle(fh, compression=None)
        else:
            self.db = pd.DataFrame(columns=['date', 'protocol', 'T0', 'T1'])

    def update_database(self):
        """
        Write the database
        """
        
        if self.db is not None:
            self.db.to_pickle(self.db_filename)

    def measure_PSC(self, protocolName, plot=True, savetimes=False):
        """
        Direct the analysis
        Uses the beginning of the protocol name to select which analysis to use
        
        Parameters:
        protocolName : str 
            Name of the protocol to analyze, underneath the datapath
        
        plot : boolean (default: True)
            Flag to plot data
        
        """
        dp_s = str(self.datapath)
        date, name, cellname, proto, sliceid = self.AR.file_cell_protocol(dp_s)
        dk = list(self.AR.getIndex(dp_s).keys())
        if 'important' in dk:
            print(str(Path(date, name, cellname, proto)), self.AR.getIndex(dp_s)['important'])
        else:
            return False
        
        
        self.AR.setProtocol(self.datapath)  # define the protocol path where the data is
        
        self.setup(clamps=self.AR)
        self.read_database(f"{protocolName:s}.p")

        if self.AR.getData():  # get that data.
            print('Protocol important: ', self.AR.protocol_important)
            if not self.AR.protocol_important:
                return False
            ok = False
            if protocolName.startswith('Stim_IO'):
                ok = self.analyze_IO()
            elif protocolName.startswith('VC-EPSC_3'):
                ok = self.analyze_VDEP()
            elif protocolName.startswith('PPF'):
                ok = self.analyze_PPF()
            if not ok:
                print('Failed on protocol in IV: ', self.datapath, protocolName)
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


    def analyze_IO(self, rmpregion=[0., 0.05], twidth=0.05, deadwin=0.001, protocolName=None, device='Stim0'):
        """Analyze in input=output relationship for a specific driving device
        """
        pulse_train = self.AR.getStim(device)  # get the stimulus information
        # stim dict in pulse_train will look like:
        # {'start': [0.05, 0.1], 'duration': [0.0001, 0.0001],
        # 'amplitude': [0.00025, 0.00025], 'npulses': [2], 'period': [0.05], 'type': ['pulseTrain']}
        # try:
        devicedata  = self.AR.getDeviceData(device=device, devicename='command')
        if devicedata is None:
            print('No device data? name command, ', device)
            return False
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
        self.sign = -1
        stim_io = self.AR.sequence[(device, 'command.PulseTrain_amplitude')]
        reps = self.AR.sequence[('protocol', 'repetitions')]
        Stim_IO = np.tile(stim_io, len(reps))  # stimuli in order
        self.analysis_summary[f'PSP_IO'] = [[]]*len(pulse_train['start'])  # create space for amplitude results, per pulse
        self.analysis_summary[f'psc_stim_amplitudes'] = [[]]*len(pulse_train['start'])  # create space for amplitude results, per pulse
        stimintvl = []
        idat = [None]*len(pulse_train['start'])
        for i in range(len(idat)):   # across each of the pulses in the train
            idat[i] = OrderedDict()  # storage for data for each stimulus level
            pdelay = pulse_train['start'][i] + delay
            if i == 0 and self.update_regions:  # if self.update_region is set, then use cursor plot to get the regions
                rgn = self.set_region([pulse_train['start'][i], pulse_train['start'][i]+twidth], baseline=bl)
            else:  # normal operation, just use stock values
                rgn = [delay, delay+width]
            self.T0 = rgn[0]  # kind of bogus
            self.T1 = rgn[1]
            region = np.array(rgn)+pulse_train['start'][i]  # get region relative to start of this pulse
            for j in range(len(self.AR.traces)):  # for all traces
                mi = self.AR.trace_index[j]  # get index into marked traces then compute the min value minus the baseline
                da = np.min(self.Clamps.traces['Time': region[0]:region[1]][j]) - np.mean(self.Clamps.traces['Time': rmpregion[0]:rmpregion[1]][j])
                if Stim_IO[mi] not in list(idat[i].keys()):
                    idat[i][Stim_IO[mi]] = [da]
                else:
                    idat[i][Stim_IO[mi]].append(da)
            for j in range(len(self.AR.traces)):
                mi = self.AR.trace_index[j]
                idat[i][Stim_IO[mi]] = np.mean(idat[i][Stim_IO[mi]])  # replace with the mean value for that stimulus level within the protocol

            self.analysis_summary[f'PSP_IO'][i] = self.sign*1e12*np.array([idat[i][k] for k in idat[i].keys()])
            self.analysis_summary[f'psc_stim_amplitudes'][i] = 1e6*np.array([k for k in idat[i].keys()])
            stimintvl.append(pulse_train['period'][0])


        stim_dt = np.diff(pulse_train['start'])
        # self.analysis_summary['psc_stim_amplitudes'] = 1e6*np.array(stim_io)
        self.analysis_summary['psc_intervals'] = np.array(stimintvl)
        self.analysis_summary['ppf_dt'] = np.array(stim_dt)
        self.analysis_summary['stim_times'] = pulse_train['start']
        self.analysis_summary['window'] = [self.T0, self.T1]
        return True

    def analyze_PPF(self, rmpregion=[0., 0.05], twidth=0.02, protocolName=None, device='Stim0'):
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
        pulse_train = self.AR.getStim(device)
        # stim dict in pulse_train will look like:
        # {'start': [0.05, 0.1], 'duration': [0.0001, 0.0001], 'amplitude': [0.00025, 0.00025], 'npulses': [2], 'period': [0.05], 'type': ['pulseTrain']}
        dd  = self.AR.getDeviceData(device=device, devicename='command')
        reps = self.AR.sequence[('protocol', 'repetitions')]
        
        stim_I = [pulse_train['amplitude'][0]]

        mode = 'PPF'
        delay = 1.0*1e-3
        width = 25.0*1e-3
        ndelay = 1.0*1e-3

        filekey = make_key(self.datapath)
        # check the db to see if we have parameters already
        dfiles = self.db['date'].tolist()  # protocols matching our prefix
        if filekey in dfiles:
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
        else:
            delay = 0.5*1e-3
            t1 = width-ndelay
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
        self.i_mean = []

        if not (device, 'command.PulseTrain_period') in self.AR.sequence.keys():
            raise ValueError('Cannot find PulseTrain_period in stimulus command')
        stim_dt = self.AR.sequence[(device, 'command.PulseTrain_period')]
        Stim_Intvl = np.tile(stim_dt, len(reps))  # stimuli in order

        self.analysis_summary[f'PPF'] = [[]]*len(stim_dt)
        self.analysis_summary['iHold'] = []
        self.analysis_summary['ppf_dt'] = [None]*len(stim_dt)
        # print(pulse_train)
        self.i_mean = []
        
        rgn = [delay, width]
        # print('rgn: ', rgn)
        if self.update_regions:
            rgn = self.set_region([pulse_train['start'][0], pulse_train['start'][0]+width], baseline=bl)
        self.T0 = float(rgn[0])
        self.T1 = float(rgn[1])
        window = self.T1-self.T0
        p1delay = pulse_train['start'][0] + self.T0 # first stim of pair
        p1end = pulse_train['start'][0] + self.T1
        pulse2 = pulse_train['start'][0] + stim_dt[0]
        bl = self.mean_I_analysis(region=self.baseline, mode='baseline', reps=[0])

        ppf_traces_T1 = OrderedDict([(k, []) for k in stim_dt])
        ppf_traces_R1 = OrderedDict([(k, []) for k in stim_dt])
        ppf_traces_T2 = OrderedDict([(k, []) for k in stim_dt])
        ppf_traces_R2 = OrderedDict([(k, []) for k in stim_dt])
        # ppftr2 = OrderedDict([(k, []) for k in stim_dt])
        ppf_dat = OrderedDict([(k, []) for k in stim_dt])

        for j in range(len(self.AR.traces)):  # for all (accepted) traces
            mi = self.AR.trace_index[j]  # get index into marked/accepted traces then compute the min value minus the baseline
            # print(pulse_train['start'][0])
            # print('stimn_dt[i]: ', Stim_Intvl[mi], delay)
            rgn = [pulse_train['start'][0]+delay, pulse_train['start'][0]+twidth]
            # rgn = [pulse_train['start'][i]+delay, pulse_train['start'][i]+twidth]
            self.T0 = rgn[0]  # kind of bogus
            self.T1 = rgn[1]
            region = np.array(rgn)+Stim_Intvl[mi]  # get region relative to start of this pulse
            region0 = [pulse_train['start'][0]+delay, pulse_train['start'][0]+np.min((Stim_Intvl[mi], twidth))]
            bl = np.mean(self.Clamps.traces['Time': rmpregion[0]:rmpregion[1]][j])
            i_pp2 = self.Clamps.traces['Time': region[0]:region[1]][j] - bl
            tb_p2= self.Clamps.time_base[np.where((self.Clamps.time_base >= region[0]) & (self.Clamps.time_base < region[1]))]
            i_pp1 = self.Clamps.traces['Time': region0[0]:region0[1]][j] - bl
            tb_ref = self.Clamps.time_base[np.where((self.Clamps.time_base >= region0[0]) & (self.Clamps.time_base < region0[1]))]
            # mpl.plot(self.Clamps.time_base*1e3, self.Clamps.traces[j]-bl, 'g-')
            # mpl.plot(tb_p2*1e3, i_pp2, 'r-')
            # mpl.plot(tb_ref*1e3, i_pp1, 'k-')
            # mpl.show()
            # exit()
            da2 = np.min(i_pp2)
            da1 = np.min(i_pp1)
            ppf_tr = da2/da1  # get facilitation for this trace and interval
            ppf_dat[Stim_Intvl[mi]].append(ppf_tr)  # accumulate 
            ppf_traces_T1[Stim_Intvl[mi]].append(tb_ref)
            ppf_traces_R1[Stim_Intvl[mi]].append(i_pp1)
            ppf_traces_T2[Stim_Intvl[mi]].append(tb_p2)
            ppf_traces_R2[Stim_Intvl[mi]].append(i_pp2)
            # self.analysis_summary[f'PPF'][i] = idat[i]
            # print(self.analysis_summary.keys())
            # self.analysis_summary['ppf_dt'][i] = 1e6*np.array([k for k in idat[i].keys()])[0]
            # stimintvl.append(pulse_train['period'][0])
            #
            # cmdv.extend([self.V_cmd[i]])
            # stimamp.extend(pulse_train['amplitude'])
            # stimintvl.append(stim_dt[i])
            # ppftr[stim_dt[i]].append({'TRef': tb_ref, 'IRef': i_pp1, 'TPP2': tb_p2, 'IPP2': i_pp2})
        # print(self.analysis_summary)
        # self.analysis_summary[f'PPF'] = [self.analysis_summary[f'PPF'][i]/self.analysis_summary[f'PPF'][i][0] for i in range(len(self.analysis_summary[f'PPF']))]
            # print('iref, ip2: ', i, ppftr[stim_dt[i]]['IRef'][0]*1e9, ppftr[stim_dt[i]]['IPP2'][0]*1e9)
        self.analysis_summary[f'PPF'] = ppf_dat
        self.analysis_summary['PPF_traces_T1'] = ppf_traces_T1
        self.analysis_summary['PPF_traces_R1'] = ppf_traces_R1
        self.analysis_summary['PPF_traces_T2'] = ppf_traces_T2
        self.analysis_summary['PPF_traces_R2'] = ppf_traces_R2
        self.analysis_summary['psc_stim_amplitudes'] = np.array(stim_I)
        self.analysis_summary['psc_intervals'] = np.array(stim_dt)
        self.analysis_summary['stim_times'] = pulse_train['start']
        self.analysis_summary['window'] = [self.T0, self.T1]
        # f, ax = mpl.subplots(1,1)
        # ax = np.array(ax).ravel()
        # for i in range(len(stim_dt)):
        #     dt = stim_dt[i]
        #     ax[0].plot(ppftr[dt]['TRef'], ppftr[dt]['IRef'], 'k')
        #     ax[0].plot(ppftr[dt]['TPP2'], ppftr[dt]['IPP2'], 'k')
        # mpl.show()
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
        pulse_train = self.AR.getStim(device)
        dt = self.Clamps.sample_interval
        # stim dict in pulse_train will look like:
        # {'start': [0.05, 0.1], 'duration': [0.0001, 0.0001],
        # 'amplitude': [0.00025, 0.00025], 'npulses': [2], 'period': [0.05], 'type': ['pulseTrain']}
        dd  = self.AR.getDeviceData(device=device, devicename='command')
        # print('pulse_train: ', pulse_train)
        # print('dd: ', dd)
        reps = self.AR.sequence[('protocol', 'repetitions')]
        stim_dt = np.diff(pulse_train['start'])
        stim_I = [pulse_train['amplitude'][0]]
        mode = '?'
        if not ('MultiClamp1', 'Pulse_amplitude') in self.AR.sequence.keys():
            raise ValueError('Cannot find (MultiClamp1, Pulse_amplitude) in stimulus command')
            
        stim_V = self.AR.sequence[('MultiClamp1', 'Pulse_amplitude')]
        filekey = make_key(self.datapath)
        # check the db to see if we have parameters already
        dfiles = self.db['date'].tolist()
        width = 20.
        
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
            t1 = (width-1.0)*1e-3
            print('auto delay', delay, t1)
        ndelay = self.NMDA_delay
        nwidth = 0.0025
        bl_region = [pulse_train['start'][0]-0.060, pulse_train['start'][0]-0.010]  # time just before stimulus
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
        self.analysis_summary[f'PSP_VDEP_AMPA'] = [[]]*len(pulse_train['start'])
        self.analysis_summary[f'PSP_VDEP_NMDA'] = [[]]*len(pulse_train['start'])
        bl = self.mean_I_analysis(region=bl_region, mode='baseline', reps=[0])
        # print('bl: ', bl)
        
        rgn = [delay, t1]
        # print('rgn: ', rgn)
        if self.update_regions:
            rgn = self.set_region([pulse_train['start'][0], pulse_train['start'][0]+self.NMDA_delay+0.010], baseline=bl, slope=True)
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
        slope_region=np.array(slope_region)+pulse_train['start'][0]
        # print('slope region: ', slope_region)

        cmds = np.array(self.V_cmd)+self.AR.holding+self.JunctionPotential
        bl = self.mean_I_analysis(region=[pulse_train['start'][0]+self.T0-0.0005, pulse_train['start'][0]+self.T0], mode='baseline', reps=[0])

        data1, tb = self.get_traces(region=slope_region,
            trlist=None, baseline=bl, intno=0, nint=1, reps=reps, slope=False)
        if data1.ndim == 1:
            return False
        # self.plot_data(tb, data1)
   
        ind = np.argmin(np.fabs(cmds-self.AMPA_voltage))

        self.T1 = self.T0 + 0.010
        print('p1min: ', self.T0)

        p1delay = pulse_train['start'][0] + self.T0
        p1end = pulse_train['start'][0] + self.T1  # note that this is a narrow
        
        nmdelay = pulse_train['start'][0] + ndelay            
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
        stimamp.append(pulse_train['amplitude'][0])
        stimintvl.append(pulse_train['period'][0])
        
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
        self.analysis_summary['stim_times'] = pulse_train['start']
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


    def set_region(self, region=None, baseline=None, slope=True):
        print('set region')
        if region is None:
            raise ValueError("PSCAnalyzer, set_region requires a region beginning and end to measure the current")
    
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
        print('done with cp') 
        self.T0, self.T1 = newCP.selectedRegion
        if self.T0 == None:
            return(None)
        return(newCP.selectedRegion)
    

    def mean_I_analysis(self, region=None, t0=0., mode='mean', baseline=None, intno=0, nint=1, reps=[0], slope=False, slopewin=None):
        """
        Get the mean current in a window
        Works with the current Clamps object
        
        Parameters
        ----------
        region : tuple, list or numpy array with 2 values (default: None)
            start and end time of a trace used to measure the RMP across
            traces. Note that if slopewin is set, it may replace the region
        
        t0 : float (default=0.5)
            start time for the mean current analysis, offset from the region ("deadwin")
        
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
        print('data shape: ', data1.shape)
        
        # data2 = self.Clamps.traces.view(np.ndarray)
        rgn = [int(region[i]/self.Clamps.sample_interval) for i in range(len(region))]
        self.V_cmd = self.Clamps.cmd_wave[:,rgn[0]:rgn[1]].mean(axis=1).view(np.ndarray)
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
            if data1.ndim == 3:
                data1 = data1[dindx,:,:]
            elif data1.ndim == 2:
                data1 = data1[dindx,:]
            else:
                raise ValueError('Data must have 2 or 3 dimensions')
        print(sh, data1.shape, nint)
        self.i_mean_index = None
        self.i_data = data1.mean(axis=0)
        self.i_tb = tb+region[0]
        
        nx = int(sh[0]/len(reps))
        
        if mode in ['mean', 'baseline']:  # just return the mean value
            i_mean = data1.mean(axis=1)  # all traces, average over specified time window
            if nint == 1:
                nx = int(sh[0]/len(reps))
                try:
                    i_mean = np.reshape(i_mean, (len(reps), nx))  # reshape by repetition
                except:
                    return i_mean
            i_mean = i_mean.mean(axis=0)  # average over reps
            return i_mean
     
        elif mode == 'min':

            # mpl.plot(data1.T)
            # mpl.show()

            i_mina = data1.min(axis=1)  # all traces, average over specified time window
            if nint == 1:
                nx = int(sh[0]/len(reps))
                try:
                    i_mina = np.reshape(i_mina, (len(reps), nx))  # reshape by repetition
                except:
                    raise ValueError("Reshape failed on min")
            print('imina: ', i_mina)
            i_min = i_mina.min(axis=0)  # average over reps
            self.i_argmin = np.argmin(i_mina, axis=0)
            # return i_min

            # dfilt = data1 # scipy.signal.savgol_filter(data1, 5, 2, axis=1, mode='nearest')
            # print('data1.shpae 0: ', data1.shape)
            # ist = int(t0/self.Clamps.sample_interval) # points in deadwin
            # ien = int((analysis_region[1]-analysis_region[0])/self.Clamps.sample_interval)
            # print('region 0: ', analysis_region[0])
            # print('analysis time: ', ist*self.Clamps.sample_interval, ien*self.Clamps.sample_interval)
            # print('dfilt shape: ', dfilt.shape)
            # print('ist, ien: ', ist, ien)
            # i_min = dfilt[:, ist:ien].min(axis=1)  # all traces, get the minimum over specified time window
            # print('nint: ', nint)
            # print('imin shape: ', i_min.shape)
            # print('reps: ', nreps)
            # if nint == 1:
            #     nx = int(sh[0]/nreps)
            #     print('nx: ', nx)
            #     print(i_min.shape)
            #     print((nreps, nx))
            #     try:
            #         i_min = np.reshape(i_min, (nreps, nx))  # reshape by repetition
            #         print('rehsape ok')
            #     except:
            #         print('reshape failed!!!!')
            # i_min = i_min.mean(axis=0)  # average over reps
            print('imin shape: ', i_min.shape)
            # data2 = np.reshape(data1, (nreps, nx, data1.shape[1])).mean(axis=0)
            # print(data2.shape)
            # f, ax = mpl.subplots(1,1)
            # sns.set_palette("coolwarm_r",data2.shape[0])
            # cp = sns.color_palette("muted",data2.shape[0])
            # device = 'Stim0'
            # # stim_I = np.array(self.AR.sequence[(device, 'command.PulseTrain_amplitude')])*1e6
            # for i in range(data2.shape[0]):
            #     ax.plot(data2[i,:], color = cp[i])
            # data3 = np.reshape(data1, (nreps, nx, data1.shape[1]))
            # for j in range(data3.shape[0]):
            #     for i in range(data3.shape[1]):
            #         csel = i
            #         print('csel: ', csel)
            #         ax.plot(data3[j, i,:], color = cp[csel], linestyle='--', linewidth=1, alpha=1)
            # mpl.show()
            # # print(ist, ien)
            # print(dfilt.shape, nreps)
            # dfw = [[]]*nreps
           #  nvs = int(sh[0]/nreps)
           #  print('nreps, nvs: ', nreps, nvs)
           #  for i in range(nreps):
           #      dfw[i] = data1[i*nvs:i*nvs + nvs,: ]
           #  dfw = np.array(dfw)
           #  print(dfw.shape)
           #  dfw = np.array(dfw).mean(axis=0)
           #  # dfw = dfw.reshape((nreps, -1, int(sh[0]/nreps)))
           #  # dfw = dfw.mean(axis=0)
           #  # for i in range(dfw.shape[0]):
           #  #     mpl.plot(dfw[i])
           #  # mpl.show()
           #
           #  # print(dfw.shape, ist, ien)
           #  self.i_argmin = dfw[:, ist:ien].argmin(axis=1) +ist
            print('iargmin: ', self.i_argmin)
            return i_min




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
            
        print('SLOPE SUBTRACTION')
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
        print('vciv')
        P = PH.regular_grid(2 , 2, order='columnsfirst', figsize=(8., 6.), showgrid=False,
                        verticalspacing=0.1, horizontalspacing=0.1,
                        margins={'leftmargin': 0.12, 'rightmargin': 0.12, 'topmargin': 0.08, 'bottommargin': 0.1},
                        labelposition=(-0.12, 0.95))
        (date, sliceid, cell, proto, p3) = self.file_cell_protocol(self.datapath)
        
        P.figure_handle.suptitle(str(Path(date, sliceid, cell, proto)), fontsize=12)
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
                        np.array(self.analysis_summary[f'PSP_IO'][i]), linewidth=1, markersize=4, marker='s')
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
    disk = '/Volumes/Pegasus/ManisLab_Data3'
    middir = 'Kasten_Michael'
    directory = 'Maness_PFC_stim'
    cell = '2019.03.19_000/slice_000/cell_001'
    # cell = '2019.03.19_000/slice_001/cell_000'
    
    ddc = Path(disk, middir, directory, cell)
    protocol = 'Stim_IO_1'
    # protocol = 'PPF_2_001'
    # protocol = 'VC-EPSC_3_ver2_003'
    fn = Path(ddc, protocol)
    PSC = PSCAnalyzer(fn)
    PSC.measure_PSC(protocol[:-4], savetimes=True)
    
