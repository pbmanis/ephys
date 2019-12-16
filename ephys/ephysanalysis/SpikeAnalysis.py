"""
Analyze spike shapes - pulled out of IVCurve 2/6/2016 pbm.
Allows routine to be used to analyze spike trains independent of acq4's data models.
Create instance, then call setup to define the "Clamps" object and the spike threshold. 
The Clamps object must have the following variables defined:

    commandLevels (current injection levels, list)
    time_base (np.array of times corresponding to traces)
    data_mode (string, indicating current or voltgae clamp)
    tstart (time for start of looking at spikes; ms)
    tend (time to stop looking at spikes; ms)
    trace (the data trace itself, numpy array records x points)
    sample_interval (time between samples, sec)
    values (command waveforms; why it is called this in acq4 is a mystery)

Note that most of the results from this module are accessed either 
as class variables, or through the class variable analysis_summary,
a dictionary with key analysis results. 
IVCurve uses the analysis_summary to post results to an sql database.

Paul B. Manis, Ph.D. 2016-2019
for Acq4 (and beyond)

"""

from collections import OrderedDict
import os
import os.path
from pathlib import Path
import inspect
import sys
import itertools
import functools
import numpy as np
import scipy
from . import Utility # pbm's utilities...
from . import Fitting # pbm's fitting stuff...
import pprint
import time

this_source_file = 'ephysanalysis.SpikeAnalysisrc'

class SpikeAnalysis():
    
    def __init__(self):
        pass
        self.threshold = 0.
        self.Clamps = None
        self.analysis_summary = {}
        self.verbose = False
        self.FIGrowth = 1  # use function FIGrowth1 (can use simpler version FIGrowth 2 also)
        self.analysis_summary['FI_Growth'] = []   # permit analysis of multiple growth functions.


    def setup(self, clamps=None, threshold=None, refractory=0.0007, peakwidth=0.001,
                    verify=False, interpolate=True, verbose=False, mode='peak', min_halfwidth=0.010):
        """
        configure the inputs to the SpikeAnalysis class
        
        Paramters
        ---------
        clamps : class (default: None)
            PatchEphys clamp data holding/accessing all ephys data for this analysis
        
        threshold : float (default: None)
            Voltage threshold for spike detection
        
        refractory : float (default 0.0007)
            Minimum time between detected spikes, in seconds (or units of the clamp
                    time base)
        
        peakwidth : float (default: 0.001)
            When using "peak" as method in findspikes, this is the peak width maximum in sec
        
        min_halfwidth : float (default: 0.010)
            minimum spike half width in seconds. Default value is deliberately large...
                    
        verify : boolean (default: False)
                    
        interpolate : boolean (default: True)
            Use interpolation to get spike threshold time and half-widths
        
        mode : string (default: 'peak')
            if using detector "peak", this is mode passed to findspikes

        verbose : boolean (default: False)
            Set true to get lots of print out while running - used
            mostly for debugging.
        """
        
        if clamps is None or threshold is None:
            raise ValueError("Spike Analysis requires defined clamps and threshold")
        self.Clamps = clamps
        self.threshold = threshold
        self.refractory = refractory
        self.interpolate = interpolate # use interpolation on spike thresholds...
        self.peakwidth = peakwidth
        self.min_halfwidth = min_halfwidth
        self.verify = verify
        self.verbose = verbose
        self.mode = mode
        self.ar_window = 0.1
        self.ar_lastspike = 0.075
        self.min_peaktotrough = 0.010 # change in V on falling phase to be considered a spike
        self.max_spike_look = 0.010  # msec over which to measure spike widths

    def analyzeSpikes(self):
        """
        analyzeSpikes: Using the threshold set in the control panel, count the
        number of spikes in the stimulation window (self.Clamps.tstart, self.Clamps.tend)
        Updates the spike plot(s).

        The following class variables are modified upon successful analysis and return:
        self.spikecount: a 1-D numpy array of spike counts, aligned with the
            current (command)
        self.adapt_ratio: the adaptation ratio of the spike train
        self.fsl: a numpy array of first spike latency for each command level
        self.fisi: a numpy array of first interspike intervals for each
            command level
        self.nospk: the indices of command levels where no spike was detected
        self.spk: the indices of command levels were at least one spike
            was detected
        self.analysis_summary : Dictionary of results.
        
        Parameters
        ----------
        None
        
        Returns
        -------
        Nothing, but see the list of class variables that are modified
        
        """
        twin = self.Clamps.tend - self.Clamps.tstart  # measurements window in seconds
        maxspkrate = 50  # max rate to count  in adaptation is 50 spikes/second
        minspk = 4
        maxspk = int(maxspkrate*twin)  # scale max dount by range of spike counts
        #print('max spike rate: ', maxspk)
        ntr = len(self.Clamps.traces)
        self.spikecount = np.zeros(ntr)
        self.fsl = np.zeros(ntr)
        self.fisi = np.zeros(ntr)
        ar = np.zeros(ntr)
        self.allisi = []
        self.spikes = [[] for i in range(ntr)]
        self.spikeIndices = [[] for i in range(ntr)]
        #print 'clamp start/end: ', self.Clamps.tstart, self.Clamps.tend
        lastspikecount = 0
        U = Utility.Utility()
        for i in range(ntr):  # this is where we should parallelize the analysis for spikes
            spikes = U.findspikes(self.Clamps.time_base, np.array(self.Clamps.traces[i]),
                                              self.threshold, t0=self.Clamps.tstart,
                                              t1=self.Clamps.tend,
                                              dt=self.Clamps.sample_interval,
                                              mode=self.mode,  # mode to use for finding spikes
                                              interpolate=self.interpolate,
                                              detector='argrelmax',
                                              refract=self.refractory,
                                              peakwidth=self.peakwidth,
                                              verify=self.verify,
                                              debug=False)
           # print (ntr, i, self.Clamps.values[i], len(spikes))
            if len(spikes) == 0:
              #  print ('no spikes found')
                continue
            self.spikes[i] = spikes
           # print 'found %d spikes in trace %d' % (len(spikes), i)
            self.spikeIndices[i] = [np.argmin(np.fabs(self.Clamps.time_base-t)) for t in spikes]
            self.spikecount[i] = len(spikes)
            self.fsl[i] = (spikes[0] - self.Clamps.tstart)*1e3
            if len(spikes) > 1:
                self.fisi[i] = (spikes[1] - spikes[0])*1e3  # first ISI
                self.allisi.append(np.diff(spikes)*1e3)
            # for Adaptation ratio analysis: limit spike rate, and also only on monotonic increase in rate
            # 8/2018: 
            #   AR needs to be tethered to time into stimulus
            #   Here we return a standardized ar measured during the first 100 msec
            #  (standard ar)
            if (minspk <= len(spikes)) and (self.spikecount[i] > lastspikecount):
                spx = spikes[np.where(spikes-self.Clamps.tstart < self.ar_window)]  # default is 100 msec
                if len(spx) >= 4: # at least 4 spikes
                    if spx[-1] > self.ar_lastspike+self.Clamps.tstart:  # default 75 msec
                        misi = np.mean(np.diff(spx[-2:]))*1e3  # last ISIs in the interval
                        ar[i] = misi / self.fisi[i]
            lastspikecount = self.spikecount[i]  # update rate (sets max rate)
            
        iAR = np.where(ar > 0)  # valid AR and monotonically rising
        self.adapt_ratio = np.nan
        if len(ar[iAR]) > 0:
            self.adapt_ratio = np.mean(ar[iAR])  # only where we made the measurement
        self.ar = ar  # stores all the ar values
        self.analysis_summary['AdaptRatio'] = self.adapt_ratio  # only the valid values
        self.nospk = np.where(self.spikecount == 0)
        self.spk = np.where(self.spikecount > 0)[0]
        self.analysis_summary['FI_Curve'] = np.array([self.Clamps.values, self.spikecount])
        self.analysis_summary['FiringRate'] = np.max(self.spikecount)/(self.Clamps.tend - self.Clamps.tstart)
        self.spikes_counted = True
#        self.update_SpikePlots()

    def analyzeSpikes_brief(self, mode='baseline'):
        """
        analyzeSpikes_brief: Using the threshold set in the control panel, count the
        number of spikes in a window and fill out ana analysis summary dict with 
        the spike latencies in that window (from 0 time)

        Parameters
        ----------
        mode: str (default : baseline)
            baseline: from 0 to self.Clamps.tstart
            poststimulus : from self.Clamps.tend to end of trace
            evoked : from self.Clamps.start to self.Clamps.end
        
        Returns:

        -------
        Nothing, but see the list of class variables that are modified
        Class variable modified is the
            self.analysis_summary : Dictionary of spike times. Key is
                'spikes_baseline'
                'spikes_poststimulus'
                'spikes_evoked' 
            according to the mode in the call
        """

        if mode == 'baseline':
            twin = [0., self.Clamps.tstart]
        elif mode == 'evoked':
            twin = [self.Clamps.tstart,self.Clamps.tend] 
        elif mode == 'poststimulus':
            twin = [self.Clamps.tend, np.max(self.Clamps.time_base)]
        else:
            raise ValueError(f'{thissourcefile:s}:: analyzeSpikes_brief requires mode to be "baseline", "evoked", or "poststimulus"')

        ntr = len(self.Clamps.traces)
        allspikes = [[] for i in range(ntr)]
        spikeIndices = [[] for i in range(ntr)]
        U = Utility.Utility()
        for i in range(ntr):
            spikes = U.findspikes(self.Clamps.time_base, np.array(self.Clamps.traces[i]),
                                              self.threshold, t0=twin[0],
                                              t1=twin[1],
                                              dt=self.Clamps.sample_interval,
                                              mode=self.mode,  # mode to use for finding spikes
                                              interpolate=self.interpolate,
                                              detector='argrelmax',
                                              refract=self.refractory,
                                              peakwidth=self.peakwidth,
                                              verify=self.verify,
                                              debug=False)
            if len(spikes) == 0:
                #print 'no spikes found'
                continue
            allspikes[i] = spikes
        self.analysis_summary[mode+'_spikes'] = allspikes

    def _timeindex(self, t):
        """
        Find the index into the time_base of the Clamps structure that 
        corresponds to the time closest to t
        
        Parameters
        ----------
        t : float (time, no default)
        
        Returns
        -------
        index : int (index to the closest time)
        """
        return np.argmin(self.Clamps.time_base-t)

    def _initialize_summarymeasures(self):
        self.analysis_summary['AP1_Latency'] = np.inf
        self.analysis_summary['AP1_HalfWidth'] = np.inf
        self.analysis_summary['AP1_HalfWidth_interpolated'] = np.inf
        self.analysis_summary['AP2_Latency'] = np.inf
        self.analysis_summary['AP2_HalfWidth'] = np.inf
        self.analysis_summary['AP2_HalfWidth_interpolated'] = np.inf
        self.analysis_summary['FiringRate_1p5T'] = np.inf
        self.analysis_summary['AHP_Depth'] = np.inf  # convert to mV

    def analyzeSpikeShape(self, printSpikeInfo=False, begin_dV=12.0):
        """analyze the spike shape.
        Does analysis of ONE protocol, all traces.
        Based on the analysis from Druckman et al. Cerebral Cortex, 2013
        
        The results of the analysis are stored in the SpikeAnalysis object
        as SpikeAnalysis.analysis_summary, a dictionary with specific keys.
        Also available are the raw spike measures, in the 'spikes' dictionary
        of the analysis_summary (spike shape dict, with keys by trace number,
        each trace with a dict of values)
        
        Every spike is measured, and a number of points on the waveform
            are defined for each spike, including the peak, the half-width
            on the rising phase, half-width on the falling phase, the
            peak of the AHP, the peak-trough time (AP peak to AHP peak),
            and a beginning, based on the slope (set in begin_dV)
        
        Parameters
        ----------
        printSpikeInfo : Boolean (default: Fase)
            Flag; when set prints arrays, etc, for debugging purposes
        
        begin_dV : float (default: 12 mV/ms)
            Slope used to define onset of the spike. The default value
            is from Druckmann et al; change this at your own peril!
        
        Returns
        -------
        Nothing (but see doc notes above)
        """
        self._initialize_summarymeasures()
        
        self.madeplot = False
        ntr = len(self.Clamps.traces)
#        print 'analyzespikeshape, self.spk: ', self.spk
        self.spikeShape = OrderedDict()
        rmps = np.zeros(ntr)
        self.iHold_i = np.zeros(ntr)
        U = Utility.Utility()
        for i in range(ntr):
           # print('rec nspk: ', i, len(self.spikes[i]))
            if len(self.spikes[i]) == 0:
                continue
            if printSpikeInfo:
                print(f'{this_source_file:s}:: spikes: ', self.spikes[i])
                print((np.array(self.Clamps.values)))
                print((len(self.Clamps.traces)))
            (rmps[i], r2) = U.measure('mean', self.Clamps.time_base, self.Clamps.traces[i],
                                           0.0, self.Clamps.tstart)            
            (self.iHold_i[i], r2) = U.measure('mean', self.Clamps.time_base, self.Clamps.cmd_wave[i],
                                              0.0, self.Clamps.tstart)
            trspikes = OrderedDict()
            for j in range(len(self.spikes[i])):
                # print('i,j,etc: ', i, j, begin_dV)
                thisspike = self.analyze_one_spike(i, j, begin_dV)
                if thisspike is not None:
                    trspikes[j] = thisspike
            self.spikeShape[i] = trspikes

        self.iHold = np.mean(self.iHold_i)
        self.analysis_summary['spikes'] = self.spikeShape  # save in the summary dictionary too       
        self.analysis_summary['iHold'] = self.iHold
        self.analysis_summary['pulseDuration'] = self.Clamps.tend - self.Clamps.tstart
        if len(self.spikeShape.keys()) > 0:  # only try to classify if there are spikes
            self.getClassifyingInfo()  # build analysis summary here as well.
        if printSpikeInfo:
            pp = pprint.PrettyPrinter(indent=4)
            for m in sorted(self.spikeShape.keys()):
                print(('----\nTrace: %d  has %d APs' % (m, len(list(self.spikeShape[m].keys())))))
                for n in sorted(self.spikeShape[m].keys()):
                    pp.pprint(self.spikeShape[m][n])

    def analyze_one_spike(self, i, j, begin_dV):
        thisspike = {'trace': i, 'AP_number': j, 'AP_beginIndex': None, 'AP_endIndex': None, 
                     'AP_peakIndex': None, 'peak_T': None, 'peak_V': None, 'AP_Latency': None,
                     'AP_beginV': None, 'halfwidth': None, 'halfwidth_interpolated': None,
                     'trough_T': None, 'trough_V': None, 'peaktotrough': None,
                     'current': None, 'iHold': None,
                     'pulseDuration': None, 'tstart': self.Clamps.tstart}  # initialize the structure
        thisspike['current'] = self.Clamps.values[i] - self.iHold_i[i]
        thisspike['iHold'] = self.iHold_i[i]
        thisspike['pulseDuration'] = self.Clamps.tend - self.Clamps.tstart  # in seconds
        thisspike['AP_peakIndex'] = self.spikeIndices[i][j]
        thisspike['peak_T'] = self.Clamps.time_base[thisspike['AP_peakIndex']]
        thisspike['peak_V'] = self.Clamps.traces[i][thisspike['AP_peakIndex']]  # max voltage of spike
        thisspike['tstart'] = self.Clamps.tstart

        # find the minimum going forward - that is AHP min
        dt = (self.Clamps.time_base[1]-self.Clamps.time_base[0])
        dv = np.diff(self.Clamps.traces[i])/dt

        # find end of spike (either top of next spike, or end of trace)
        k = self.spikeIndices[i][j] + 1  # point to next spike
        if j < self.spikecount[i] - 1:  #
            kend = self.spikeIndices[i][j+1]
        else:
            kend = int(self.spikeIndices[i][j]+self.max_spike_look/dt)
        if kend > dv.shape[0]:
            return(thisspike)  # end of spike would be past end of trace
        else:
            if kend < k:
                kend = k + 1
            km = np.argmin(dv[k:kend]) + k

        # Find trough after spike and calculate peak to trough
        kmin =  np.argmin(self.Clamps.traces[i][km:kend])+km
        thisspike['AP_endIndex'] = kmin
        thisspike['trough_T'] = self.Clamps.time_base[thisspike['AP_endIndex']]
        thisspike['trough_V'] = self.Clamps.traces[i][kmin]

        if thisspike['AP_endIndex'] is not None:
            thisspike['peaktotrough'] = thisspike['trough_T'] - thisspike['peak_T']

        # find points on spike waveform
        # because index is to peak, we look for previous spike
        k = self.spikeIndices[i][j]-1
        if j > 0:
            kbegin = self.spikeIndices[i][j-1] # index to previous spike start
        else:
            kbegin = k - int(0.002/dt)  # for first spike - 2 msec prior only

        if k <= kbegin:
            k = kbegin + 2
        if k > len(dv):  # end of block of data, so can not measure
            return(thisspike)
        try:
            km = np.argmax(dv[kbegin:k]) + kbegin
        except:
            print(f'{this_source_file:s}:: kbdgin, k: ', kbegin, k)
            print(len(dv))
            raise
        if ((km - kbegin) < 1):
            km = kbegin + int((k - kbegin)/2.) + 1
        kthresh = np.argmin(np.fabs(dv[kbegin:km] - begin_dV)) + kbegin  # point where slope is closest to begin
        # print('kthresh, kbegin: ', kthresh, kbegin)
        # save values in dict here
        thisspike['AP_beginIndex'] = kthresh
        thisspike['AP_Latency'] = self.Clamps.time_base[kthresh]
        thisspike['AP_beginV'] = self.Clamps.traces[i][thisspike['AP_beginIndex']]
        # if successful in defining spike start/end, calculate half widths in two ways:
        # closest points in raw data, and by interpolation
        if (
                (thisspike['AP_beginIndex'] is not None) and
                (thisspike['AP_beginIndex'] > 0) and
                (thisspike['AP_endIndex'] is not None) and
                (thisspike['AP_beginIndex'] < thisspike['AP_peakIndex']) and
                (thisspike['AP_peakIndex'] < thisspike['AP_endIndex'])
            ):
            halfv = 0.5*(thisspike['peak_V'] + thisspike['AP_beginV'])
            tr = np.array(self.Clamps.traces[i])
            xr = self.Clamps.time_base
            kup = np.argmin(np.fabs(tr[thisspike['AP_beginIndex']:thisspike['AP_peakIndex']] - halfv))
            kup += thisspike['AP_beginIndex']
            kdown = np.argmin(np.fabs(tr[thisspike['AP_peakIndex']:thisspike['AP_endIndex']] - halfv))
            kdown += thisspike['AP_peakIndex'] 
            if kup is not None and kdown is not None:
                thisspike['halfwidth'] = xr[kdown] - xr[kup]
                thisspike['hw_up'] = xr[kup] - xr[thisspike['AP_peakIndex']]
                thisspike['hw_down'] = xr[thisspike['AP_peakIndex']] - xr[kdown]
                thisspike['hw_v'] = halfv
                # interpolated spike hwup, down and width
                pkt = xr[thisspike['AP_peakIndex']]
                if tr[kup] <= halfv:
                    vi = tr[kup-1:kup+1]
                    xi = xr[kup-1:kup+1]
                else:
                    vi = tr[kup:kup+2]
                    xi = xr[kup:kup+2]
                m1 = (vi[1]-vi[0])/(xi[1]-xi[0])
                b1 = vi[1] - m1*xi[1]
                if m1 == 0.0 or np.std(tr) == 0.0:
                    # print('a: ', vi[1], vi[0], kup, tr[kup:kup+2], tr[kup-1:kup+1], tr[kup], halfv)
                    return(thisspike)

                t_hwup = (halfv-b1)/m1
                if tr[kdown] <= halfv:
                    vi = tr[kdown:kdown+2]
                    xi = xr[kdown:kdown+2]
                    u='a'
                else:
                    vi = tr[kdown-1:kdown+1]
                    xi = xr[kdown-1:kdown+1]
                    u='b'
                m2 = (vi[1]-vi[0])/(xi[1]-xi[0])
                b2 = vi[1] - m2*xi[1]
                if m2 == 0.0 or np.std(tr) == 0.0:
                    # print('b: ', vi[1], vi[0], kup , tr[kdown-1:kdown+1], tr[kdown:kdown+2], tr[kdown], halfv)
                    return(thisspike)
                t_hwdown = (halfv-b2)/m2
                thisspike['halfwidth'] = t_hwdown-t_hwup
                # import matplotlib.pyplot as mpl
                # fig, ax = mpl.subplots(1,1)
                # ax.plot(xr[kup-10:kdown+10], tr[kup-10:kdown+10])
                # ax.plot(t_hwdown, halfv, 'ro')
                # ax.plot(t_hwup, halfv, 'bx')
                # mpl.show()
                if thisspike['halfwidth'] > self.min_halfwidth:  # too broad to be acceptable
                    if self.verbose:
                        print(f'{this_source_file:s}::\n   spikes > min half width', thisspike['halfwidth'])
                        print('   halfv: ', halfv, thisspike['peak_V'], thisspike['AP_beginV'])
                    thisspike['halfwidth'] = None
                    thisspike['halfwidth_interpolated'] = None

                else:
                    thisspike['halfwidth_interpolated'] = t_hwdown - t_hwup
                pkvI = tr[thisspike['AP_peakIndex']]
                pkvM = np.max(tr[thisspike['AP_beginIndex']:thisspike['AP_endIndex']])
                pkvMa = np.argmax(tr[thisspike['AP_beginIndex']:thisspike['AP_endIndex']])
                if pkvI != pkvM:
                    pktrap = True

        return(thisspike)

    def getIVCurrentThresholds(self):
        """ figure out "threshold" for spike, get 150% and 300% points.
        
        Parameters
        ----------
        None
        
        Returns
        -------
        tuple: (int, int)
            The tuple contains the index to command threshold for spikes, and 150% of that threshold
            The indices are computed to be as close to the command step values
            that are actually used (so, the threshold is absolute; the 150%
            value will be the closest estimate given the step sizes used to
            collect the data)
        """
        icmd = []  # list of command currents that resulted in spikes.
        for m in sorted(self.spikeShape.keys()):
            n = len(list(self.spikeShape[m].keys())) # number of spikes in the trace
            for n in list(self.spikeShape[m].keys()):
                icmd.append(self.spikeShape[m][n]['current'])
        icmd = np.array(icmd)
        try:
            iamin = np.argmin(icmd)
        except:
            print(f'{this_source_file:s}: Problem with command: ')
            print('self.spikeShape.keys(): ', self.spikeShape.keys())
            print('   m = ', m)
            print('   n = ', n)
            print('   current? ', self.spikeShape[m][n]['current'])
            raise ValueError(f'{this_source_file:s}:getIVCurrentThresholds - icmd seems to be ? : ', icmd)
            
        imin = np.min(icmd)
        ia150 = np.argmin(np.abs(1.5*imin-icmd))
        iacmdthr = np.argmin(np.abs(imin-self.Clamps.values))
        ia150cmdthr = np.argmin(np.abs(icmd[ia150] - self.Clamps.values))
        return (iacmdthr, ia150cmdthr)  # return threshold indices into self.Clamps.values array at threshold and 150% point


    
    def getClassifyingInfo(self):
        """
        Adds the classifying information according to Druckmann et al., Cerebral Cortex, 2013
        to the analysis summary
        
        Parameters
        ----------
        None
        
        Returns
        -------
        Nothing
        
        Modifies the class analysis_summary dictionary to contain a number of results
        regarding the AP train, including the first and second spike latency,
        the first and second spike halfwidths, the firing rate at 150% of threshold,
        and the depth of the AHP
        """
 
        (jthr, j150) = self.getIVCurrentThresholds()  # get the indices for the traces we need to pull data from
        jthr = int(jthr)
        j150 = int(j150)
        if j150 not in list(self.spikeShape.keys()):
            return
        if jthr == j150 and self.verbose:
            #print '\n%s:' % self.filename
            print('Threshold current T and 1.5T the same: using next up value for j150')
            print('jthr, j150, len(spikeShape): ', jthr, j150, len(self.spikeShape))
            print('1 ', self.spikeShape[jthr][0]['current']*1e12)
            print('2 ', self.spikeShape[j150+1][0]['current']*1e12)
            print(' >> Threshold current: %8.3f   1.5T current: %8.3f, next up: %8.3f' % (self.spikeShape[jthr][0]['current']*1e12,
                        self.spikeShape[j150][0]['current']*1e12, self.spikeShape[j150+1][0]['current']*1e12))
            j150 = jthr + 1

        spikesfound = False
        if len(self.spikeShape[j150]) >= 1 and (0 in list(self.spikeShape[j150].keys())) and self.spikeShape[j150][0]['halfwidth'] is not None:
            self.analysis_summary['AP1_Latency'] = (self.spikeShape[j150][0]['AP_Latency'] - self.spikeShape[j150][0]['tstart'])*1e3
            self.analysis_summary['AP1_HalfWidth'] = self.spikeShape[j150][0]['halfwidth']*1e3
            if self.spikeShape[j150][0]['halfwidth_interpolated'] is not None:
                self.analysis_summary['AP1_HalfWidth_interpolated'] = self.spikeShape[j150][0]['halfwidth_interpolated']*1e3
            else:
                self.analysis_summary['AP1_HalfWidth_interpolated'] = np.nan
            spikesfound = True

        if len(self.spikeShape[j150]) >= 2 and (1 in list(self.spikeShape[j150].keys())) and self.spikeShape[j150][1]['halfwidth'] is not None:
            self.analysis_summary['AP2_Latency'] = (self.spikeShape[j150][1]['AP_Latency'] - self.spikeShape[j150][1]['tstart'])*1e3
            self.analysis_summary['AP2_HalfWidth'] = self.spikeShape[j150][1]['halfwidth']*1e3
            if self.spikeShape[j150][1]['halfwidth_interpolated'] is not None:
                self.analysis_summary['AP2_HalfWidth_interpolated'] = self.spikeShape[j150][1]['halfwidth_interpolated']*1e3
            else:
                self.analysis_summary['AP2_HalfWidth_interpolated'] = np.nan
        
        if spikesfound:
            rate = len(self.spikeShape[j150])/self.spikeShape[j150][0]['pulseDuration']  # spikes per second, normalized for pulse duration
            # first AHP depth
            print('AHP pars: begin, trough: ', self.spikeShape[j150][0]['AP_beginV'], self.spikeShape[j150][0]['trough_V'])
            AHPDepth = self.spikeShape[j150][0]['AP_beginV'] - self.spikeShape[j150][0]['trough_V']  # from first spike in train
            self.analysis_summary['FiringRate_1p5T'] = rate
            self.analysis_summary['AHP_Depth'] = AHPDepth*1e3  # convert to mV
            print('AHP Depth: ', AHPDepth*1e3)

    def fitOne(self, x=None, yd=None, info='', function=None, fixNonMonotonic=True, excludeNonMonotonic=False):
        """Fit the FI plot to an equation that is piecewise linear up to the threshold
            called Ibreak, then (1-exp(F/Frate)) for higher currents
        
        Parameters
        ---------- 
            x : numpy array (no default)
                The x data to fit (typically an array of current levels)
        
            yd : numpy array (no default)
                The y data to fit (typically an array of spike counts)
        
            if x and yd are none, we extrace from the 'FI_Curve' for this cell.
            info : string (default: '')
                information to add to a fitted plot
        
            fixNonMonotonic : Boolean (default: True)
                If True, only use data up to the maximal firing rate,
                discarding the remainder of the steps under the assumption
                that the cell is entering depolarization block.
        
            excludeNonMonotonic : Boolean (default: False)
                if True, does not even try to fit, and returns None
        
        Returns
        -------
        None if there is no fitting to be done (excluding non-monotonic or no spikes)
        tuple of (fpar, xf, yf, names, error, f, func)
            These are the fit parameters
        """
        # print('fitone called')
        if function is not None:
            self.FIGrowth = function
        if x is None: # use class data
            x = self.analysis_summary['FI_Curve'][0]*1e9
            yd = self.analysis_summary['FI_Curve'][1]/self.analysis_summary['pulseDuration']  # convert to rate in spikes/second
        
        
        if self.FIGrowth == 'fitOneOriginal':
            ymax = np.max(yd)
            ymax_a = 0.8*ymax
            if ymax <= 0.:
                return(None)
            nonmono = 0
            if fixNonMonotonic: # and ymax > yd[-1]:  # clip at max firing rate
                imaxs = [i for i, y in enumerate(yd) if y >= ymax_a]  # handle duplicate firing rates
                # print('imaxs: ', imaxs)
                imax = max(imaxs)  # find highest index
                # print('imax: ', imax)
                dypos = range(0, imax+1)
                x = x[dypos]
                # print('yd: ', yd)
                yd = yd[dypos]
                # print('yd[dypos]: ', yd)
                ymax = np.max(yd)
            if np.max(x) < 0.:  # skip if max rate is < 0 current
                return(None)
            ymin = 5.
            if ymax < ymin:
                ymin = 0.
            if ymax > yd[-1] and excludeNonMonotonic:
                nonmono += 1
                return(None)
#            fpnt = np.where(yd > 0)  # find first point where cell fires
            fire_points = np.where((yd[:-1] > 0) & (yd[1:] > 0))[0]  # limit to positive current injections with successive spikes
            if len(fire_points) == 0:
                return(None)
            fbr = fire_points[0]
            # print('fpnt: ', fire_points)
            # print('yd: ', yd)
            # fbr = fpnt[0][0]
            # print('fbr: ', fbr)
            ibreak0 = x[fbr-1]  # use point before first spike as the initial break point
            dx = np.abs(np.mean(np.diff(x)))  # get current steps
            xp = x[fire_points]
            # print('xp: ', xp)
            xp = xp - ibreak0 - dx
            yp = yd[fire_points]  # save data with responses
            # print('yp: ', yp)
            testMethod = 'SLSQP'  #  'SLSQP'  # L-BFGS-B simplex, SLSQP, 'TNC', 'COBYLA'
    #        print 'ibreak0: ', ibreak0
            if fbr-2 >= 0:
                x0 = fbr-2
            else:
                x0 = 0
            if fbr < len(x):
                x1 = fbr
            else:
                x1 = len(x)-1
            res = []
            err = []
            fitter = Fitting.Fitting()  # make sure we always work with the same instance
            for i in range(-4, 4):  # allow breakpoint to move
                if fbr + i + 1 > len(x)-1:
                    continue
                x0 = fbr+i
                for j in range(0,4):  # then vary the width of the linear region
                    x1 = x0 + j
                    if x1 >= len(x):
                        continue
                    bounds = ((0., 0.), np.sort([x[x0], x[x1]]),
                         (0., 2.*yp[0]), (0., ymax*5.0), (0.0001, 100.))
                    # parameters for FIGrowth 1: ['Fzero', 'Ibreak', 'F1amp', 'F2amp', 'Irate']
                    fitbreak0 = ibreak0
                    initpars = [0., ibreak0, 0., ymax/2., 0.001]
                    func = 'FIGrowthExpBreak'
                    # print( 'bounds: ', bounds)
                    # print( 'initpars: ', initpars)
                    # print('fitbreak0: ', fitbreak0)
                    # print('max x, yd: ', np.max(x[fire_points]), np.max(yd[fire_points]))
                    # f = Fitting.Fitting().fitfuncmap[func]
                    f = fitter.fitfuncmap[func]
                    # now fit the full data set
                    (fpar, xf, yf, names) = fitter.FitRegion(np.array([1]), 0, x, yd, t0=fitbreak0, t1=np.max(x[fire_points]),
                                            fitFunc=func, fitPars=initpars, bounds=bounds,
                                            fixedPars=None, method=testMethod)
                    error = fitter.getFitErr()
                           # print 'xf: ', np.min(xf), np.max(xf)
                    res.append({'fpar': fpar, 'xf': xf, 'yf': yf, 'names': names, 'error': error})
                    err.append(error)
                    # print('xr: ', np.sort([x[x0], x[x1]]), ' err: %f' % error)
            # print('err: ', err)
            minerr = np.argmin(err)
            # print('minerr: ', minerr)
            fpar = res[minerr]['fpar']
            xf = res[minerr]['xf']
            yf = res[minerr]['yf']
            names = res[minerr]['names']
            error = res[minerr]['error']
            # print ('fpar: ', fpar)

        else:  # recompute some stuff
        
        # estimate initial parameters and set region of IV curve to be used for fitting
            ymax = np.max(yd)  # maximum spike rate (not count; see above)
            if ymax == 0:
                return None
            ymax_nm = 0.8*np.max(yd)  # maximum spike rate (not count; see above)
            dypos = range(len(x))
            if fixNonMonotonic and ymax_nm > yd[-1]:  # fix non-monotinic firing - clip fitting to current that generates the max firing rate
                imaxs = [i for i, y in enumerate(yd) if y >= ymax_nm]  # handle duplicate firing rates
                imax = max(imaxs)  # find highest index
                dypos = list(range(0, imax+1))
                x = x[dypos]  # restrict current and response range to those currents
                yd = yd[dypos]
                ymax = np.max(yd)
            if np.max(x) < 0.:  # skip if max rate occurs at negative current level
                return None 

            ymin = 5
            if ymax < ymin:
                ymin = 0.
            if ymax > yd[-1] and excludeNonMonotonic:
                nonmono += 1
                return None 
        
            # Now find first point where cell fires and next step also has cell firing
            fire_points = np.where((yd[:-1] > 0) & (yd[1:] > 0))[0]  # limit to positive current injections with successive spikes
            fbr = fire_points[0]
            testMethod = 'SLSQP'  #  'SLSQP'  # L-BFGS-B simplex, SLSQP, 'TNC', 'COBYLA'
            if fbr - 1 >= 0:  # set start and end of linear fit
                x0 = fbr - 1  # x0 is last point (in current) with no spikes
            else:
                x0 = 0
            if fbr < len(x):  # x1 is the next point, which has a spike
                x1 = fbr
            else:
                x1 = len(x) - 1
            ibreak0 = x[x0]  # use point before first spike as the initial break point
        
        if self.FIGrowth == 'FIGrowthExpBreak':
            # print('Exponential model fit')
            ixb = fbr # np.argwhere(yd > 0)[0][0]
            cons = ( {'type': 'eq', 'fun': lambda xc:  xc[0]},  # lock F0 at >= 0
                     {'type': 'ineq', 'fun': lambda xc: xc[1] - x[ixb-1]},  #  ibreak between last no spike and first spiking level
                     {'type': 'ineq', 'fun': lambda xc: x[ixb] - xc[1]},  #  ibreak between last no spike and first spiking level
                     {'type': 'eq', 'fun': lambda xc: xc[2]}, # F1amp >= 0
                     {'type': 'ineq', 'fun': lambda xc: xc[3] - xc[2]}, # F2amp > F1amp (must be!)
                     {'type': 'ineq', 'fun': lambda xc: xc[4]},
                )
            bounds = ((0., yd[fbr-1]+5), np.sort([x[x0], x[x1]]),
                 (0., 2*yd[fbr]), (0., ymax*10.0), (0, 1e5))
            # print(x, yd)
            # print('fbr: ', fbr)  # first spike
            # print('ymax: ', ymax)
            # print('x0, x1: ', x0, x1)
            # print('xfbr: ', x[fbr])
            # print('ydfbr: ', yd[fbr-1], yd[fbr])
            # print('bounds: ', bounds)  # *(yd[fbr]-yd[fbr-1])/(x[fbr]-x[fbr-1])
            # print(yd[fbr-1], yd[fbr], x[fbr-1], x[fbr])
        # # parameters for FIGrowth 1: ['Fzero', 'Ibreak', 'F1amp', 'F2amp', 'Irate']
            initpars = [0., ibreak0, yd[fbr], ymax*2, 0.01*np.max(np.diff(yd)/np.diff(x))]
            # print('initpars: ', initpars)
            func = 'FIGrowthExpBreak'
            fitbreak0 = x[fbr]
            f = Fitting.Fitting().fitfuncmap[func]
            # now fit the full data set
            # print('breaks/max: ', fitbreak0, np.max(x[fpnt]))
            (fpar, xf, yf, names) = Fitting.Fitting().FitRegion(np.array([1]), 0, x, yd, t0=fitbreak0, t1=x[dypos[-1]],
                                    fitFunc=func, fitPars=initpars, bounds=bounds, constraints=cons, weights=None, #np.sqrt,
                                    fixedPars=None, method=testMethod)
            # print('names: ', names)
            # print('initpars: ', initpars)
            # print('fitbreak0', fitbreak0)
            # print('\n Bounds: ', bounds)
            # print('results: ', fpar)
            error = Fitting.Fitting().getFitErr()
            self.FIKeys = f[6]

        elif self.FIGrowth == 'FIGrowthExp':  # FIGrowth is 2, Exponential from 0 rate
            bounds = (np.sort([x[x0], x[x1]]),
                  (0., ymax*5.0), (0.0001, 1000.))
        # # parameters for FIGrowth 2: [''Ibreak', 'F2amp', 'Irate']
            fitbreak0 = ibreak0
            if fitbreak0 > 0.:
                fitbreak0 = 0.
            initpars = [ibreak0, ymax/2., 0.001]
            func = 'FIGrowthExp'
            f = Fitting.Fitting().fitfuncmap[func]
            # now fit the full data set
            (fpar, xf, yf, names) = Fitting.Fitting().FitRegion(np.array([1]), 0, x, yd, t0=fitbreak0, t1=np.max(x),
                                    fitFunc=func, fitPars=initpars, bounds=bounds,
                                    fixedPars=None, method=testMethod)
            error = Fitting.Fitting().getFitErr()
            self.FIKeys = f[6]
            imap = [-1, 0, -1, 1, 2]
            
        elif self.FIGrowth == 'piecewiselinear3':

            fitbreak0 = ibreak0
            # print('ibreak0: ', ibreak0)
            if fitbreak0 > 0.:
                fitbreak0 = 0.
            x1 = np.argwhere(yd > 0.)
            initpars = (x[x1[0]-1], 0. , x[x1[0]+1], 1., 20., 100.)
            if x[fbr] > 0:
                xn = x[fbr]
            else:
                xn = 0
            bounds = ((0., xn), # Ibreak forced to first spike level almost
                       (0., 20.),  # Rate0 (y0)
                       (0., np.max(x)),  # Ibreak1 (x1)  # spread it out?
                       (0., 100.), # IRate1  (k1, k2, k3)
                       (0., 1000.), #IRate2
                       (0., 1000.), # Irate3
                      )
            cons = ( {'type': 'ineq', 'fun': lambda x: x[0]},
                     {'type': 'ineq', 'fun': lambda x: x[1]},
                     {'type': 'ineq', 'fun': lambda x: x[2] - (x[0] + 0.05 + ibreak0)}, # ibreak1 > 50pA + ibreak0
                     {'type': 'ineq', 'fun': lambda x: x[3]},
                     {'type': 'ineq', 'fun': lambda x: x[4] - x[3]},
                     {'type': 'ineq', 'fun': lambda x: x[5] - x[4]/2.0},
                )
                     
            func = 'piecewiselinear3'
            f = Fitting.Fitting().fitfuncmap[func]
            # now fit the full data set
            (fpar, xf, yf, names) = Fitting.Fitting().FitRegion(np.array([1]), 0, x, yd, t0=fitbreak0, t1=np.max(x),
                                    fitFunc=func, fitPars=initpars, bounds=bounds, constraints=cons,
                                    fixedPars=None, method=testMethod)
            error = Fitting.Fitting().getFitErr()
            self.FIKeys = f[6]
            
        elif self.FIGrowth == 'piecewiselinear3_ugh': # use piecewise linear, 3 segment fit
            # print ('Fitting with 3 segment line')
            # parameters for pwl3 (piecewise linear...): ['Ibreak', 'Rate0', 'Ibreak1', 'Irate1', 'Irate2', 'Irate3']
            fitbreak0 = ibreak0
            # print('ibreak0: ', ibreak0)
            if fitbreak0 > 0.:
                fitbreak0 = 0.
            x1 = np.argwhere(yd > 0.)
            initpars = (x[x1[0]-1], 0. , x[x1[0]+1], 1., 20., 100.)
            if x[fbr] > 0:
                xn = x[fbr]
            else:
                xn = 0
            bounds = ((0., xn), # Ibreak forced to first spike level almost
                       (0., 20.),  # Rate0 (y0)
                       (0., np.max(x)),  # Ibreak1 (x1)  # spread it out?
                       (0., 100.), # IRate1  (k1, k2, k3)
                       (0., 1000.), #IRate2
                       (0., 1000.), # Irate3
                      )
            cons = ( {'type': 'ineq', 'fun': lambda x: x[0]},
                     {'type': 'ineq', 'fun': lambda x: x[1]},
                     {'type': 'ineq', 'fun': lambda x: x[2] - (x[0] + 0.05 + ibreak0)}, # ibreak1 > 50pA + ibreak0
                     {'type': 'ineq', 'fun': lambda x: x[3]},
                     {'type': 'ineq', 'fun': lambda x: x[4] - x[3]},
                     {'type': 'ineq', 'fun': lambda x: x[5] - x[4]/2.0},
                )
                     
            func = 'piecewiselinear3'
            f = Fitting.Fitting().fitfuncmap[func]
            # now fit the full data set
            (fpar, xf, yf, names) = Fitting.Fitting().FitRegion(np.array([1]), 0, x, yd, t0=fitbreak0, t1=np.max(x),
                                    fitFunc=func, fitPars=initpars, bounds=bounds, constraints=cons,
                                    fixedPars=None, method=testMethod)
            error = Fitting.Fitting().getFitErr()
            self.FIKeys = f[6]

        elif self.FIGrowth == 'FIGrowthPower':
            # print ('Fitting with sublinear power function FI')
            # parameters for power (piecewise linear...): [c, s, 'd']
            # data are only fit for the range over which the cell fires

            fitbreak0 = ibreak0*1e9
            if fitbreak0 > 0.:
                fitbreak0 = 0.
            ix1 = np.argwhere(yd > 0.)  # find first point with spikes
            xna = x*1e9
            x1 = xna[ix1[0]][0]
            initpars = (x1, 3., 0.5)  # 
            bds = [(0., 500.), (0.01, 100.), (0.01, 100)]

            # cons = ( {'type': 'ineq', 'fun': lambda x:  x[0]},
           #           {'type': 'ineq', 'fun': lambda x: x[1]},
           #           {'type': 'ineq', 'fun': lambda x: x[2] - [x[0] + 50.]}, # ibreak1 > 100pA + ibreak0
           #           {'type': 'ineq', 'fun': lambda x: x[3]},
           #           {'type': 'ineq', 'fun': lambda x: x[4] - x[3]},
           #           {'type': 'ineq', 'fun': lambda x: x[4]*0.5 - x[5]},
           #      )
           #                    
            func = 'FIGrowthPower'
            f = Fitting.Fitting().fitfuncmap[func]
            # now fit the full data set
            (fpar, xf, yf, names) = Fitting.Fitting().FitRegion(np.array([1]), 0, xna, yd, t0=fitbreak0, t1=np.max(xna[fpnt]),
                                    fitFunc=func, fitPars=initpars, bounds=bds, constraints=None,
                                    fixedPars=None, method=testMethod)
            error = Fitting.Fitting().getFitErr()
            self.FIKeys = f[6]
        elif self.FIGrowth == 'fitOneOriginal':
            pass
        else:
            raise ValueError('SpikeAnalysis: FIGrowth function %s is not known' % self.FIGrowth)
        self.analysis_summary['FI_Growth'].append({'FunctionName': self.FIGrowth, 'function': func,
                'names': names, 'error': error, 'parameters': fpar, 'fit': [np.array(xf)*1e-9, yf]})

