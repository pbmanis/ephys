"""
Analyze spike shapes - pulled out of IVCurve 2/6/2016 pbm.
Allows routine to be used to analyze spike trains independent of acq4's data models.
Create instance, then call setup to define the "Clamps" object and the spike threshold. 
The Clamps object must have the following variables defined::

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

import pprint
from collections import OrderedDict
from dataclasses import dataclass
import datetime
from typing import Union
import multiprocessing as MP
from pyqtgraph import multiprocess as mproc

import numpy as np
import scipy.stats
from numba import jit

from ephys.tools import fitting  # pbm's fitting stuff...
from ephys.tools import utilities  # pbm's utilities...
import pylibrary.tools.cprint as CP

U = utilities.Utility()

this_source_file = "ephysanalysis.SpikeAnalysis"


@dataclass
class OneSpike:
    trace: int
    AP_number: int
    dt: Union[float, None] = None  # sample rate
    dvdt: object = None  # np.array
    V: object = None  # np.array
    Vtime: object = None  # np.array
    pulseDuration: Union[float, None] = None
    tstart: Union[float, None] = None
    tend: Union[float, None] = None
    AP_beginIndex: Union[int, None] = None
    AP_peakIndex: Union[int, None] = None
    AP_endIndex: Union[int, None] = None
    peak_T: Union[int, None] = None
    peak_V: Union[int, None] = None
    AP_latency: Union[float, None] = None
    AP_begin_V: Union[float, None] = None
    halfwidth: Union[float, None] = None
    halfwidth_V: Union[float, None] = None
    halfwidth_up: Union[float, None] = None
    halfwidth_down: Union[float, None] = None
    halfwidth_interpolated: Union[float, None] = None
    left_halfwidth_T: Union[float, None] = None
    left_halfwidth_V: Union[float, None] = None
    right_halfwidth_T: Union[float, None] = None
    right_halfwidth_V: Union[float, None] = None
    trough_T: Union[float, None] = None
    trough_V: Union[float, None] = None
    peaktotrough: Union[float, None] = None
    current: Union[float, None] = None
    iHold: Union[float, None] = None
    dvdt_rising: Union[float, None] = None
    dvdt_falling: Union[float, None] = None


@jit(nopython=True)
def interpolate_halfwidth(tr, xr, kup, halfv, kdown):
    if tr[kup] <= halfv:
        vi = tr[kup - 1 : kup + 1]
        xi = xr[kup - 1 : kup + 1]
    else:
        vi = tr[kup : kup + 2]
        xi = xr[kup : kup + 2]
    m1 = (vi[1] - vi[0]) / (xi[1] - xi[0])
    b1 = vi[1] - m1 * xi[1]
    if m1 == 0.0 or np.std(tr) == 0.0:
        # print('a: ', vi[1], vi[0], kup, tr[kup:kup+2], tr[kup-1:kup+1], tr[kup], halfv)
        # raise ValueError("m1 is 0 or std(tr) is 0, ? ")
        return None, None

    t_hwup = (halfv - b1) / m1
    if tr[kdown] <= halfv:
        vi = tr[kdown : kdown + 2]
        xi = xr[kdown : kdown + 2]
        u = "a"
    else:
        vi = tr[kdown - 1 : kdown + 1]
        xi = xr[kdown - 1 : kdown + 1]
        u = "b"
    m2 = (vi[1] - vi[0]) / (xi[1] - xi[0])
    b2 = vi[1] - m2 * xi[1]
    if m2 == 0.0 or np.std(tr) == 0.0:
        # print('b: ', vi[1], vi[0], kup , tr[kdown-1:kdown+1], tr[kdown:kdown+2], tr[kdown], halfv)
        # raise ValueError("m2 is 0 or std(tr) is 0, ? ")
        return None, None
    t_hwdown = (halfv - b2) / m2
    return t_hwdown, t_hwup


class SpikeAnalysis:
    def __init__(self):
        pass
        self.spike_detector = "Kalluri"  # This seems to be the best method to use.
        self.reset_analysis()

    def reset_analysis(self):
        self.threshold = 0.0
        self.Clamps = None
        self.analysis_summary = {}
        self.verbose = False
        self.FIGrowth = 1  # use function FIGrowth1 (can use simpler version FIGrowth 2 also)
        self.analysis_summary["FI_Growth"] = []  # permit analysis of multiple growth functions.
        self.U = utilities.Utility()

    def setup(
        self,
        clamps=None,
        threshold=None,
        refractory: float = 0.0007,
        peakwidth: float = 0.001,
        verify=False,
        interpolate=True,
        verbose=False,
        mode="peak",
        min_halfwidth=0.010,
        max_spike_look=0.010,
        data_time_units: str = "s",
        data_volt_units: str = "V",
    ):
        """
        configure the inputs to the SpikeAnalysis class

        Parameters
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
        assert data_time_units in ["s", "ms"]
        assert data_volt_units in ["V", "mV"]
        self.time_units = data_time_units
        self.volt_units = data_volt_units  # needed by spike detector for data conversion
        self.threshold = threshold
        self.refractory = refractory
        self.interpolate = interpolate  # use interpolation on spike thresholds...
        self.peakwidth = peakwidth
        self.min_halfwidth = min_halfwidth
        self.verify = verify
        self.verbose = verbose
        self.mode = mode
        self.ar_window = 1.0
        self.ar_lastspike = 0.75 * self.ar_window
        self.min_peaktotrough = 0.010  # change in V on falling phase to be considered a spike
        self.max_spike_look = max_spike_look  # sec over which to measure spike widths

    def set_detector(self, detector: str = "argrelmax"):
        assert detector in [
            "argrelmax",
            "threshold",
            "Kalluri",
            "find_peaks",
            "find_peaks_cwt",
        ]
        self.spike_detector = detector

    def analyzeSpikes(self, reset=True, track:bool=False):
        """
        analyzeSpikes: Using the threshold set in the control panel, count the
        number of spikes in the stimulation window (self.Clamps.tstart, self.Clamps.tend)
        Updates the spike plot(s).
        Operates over all current/voltage trace pairs in the protocol.

        The following class variables are modified upon successful analysis and return::
            self.spikecount: a 1-D numpy array of spike counts, aligned with the current (command)
            self.allisi : a list of numpy arrays of interspike intervals for each command level
            self.adapt_ratio: the adaptation ratio of the spike train
            self.fsl: a numpy array of first spike latency for each command level
            self.fisi: a numpy array of first interspike intervals for each command level
            self.nospk: the indices of command levels where no spike was detected
            self.spk: the indices of command levels were at least one spike was detected
            self.analysis_summary : Dictionary of results.

        Parameters
        ----------
        None

        Returns
        -------
        Nothing, but see the list of class variables that are modified

        """
        if reset:
            self.analysis_summary["FI_Growth"] = []  # permit analysis of multiple growth functions.
        self.analysis_summary['analysistimestamp'] = datetime.datetime.now().strftime('%m/%d/%Y, %H:%M:%S')
        # CP.cprint("r", "AnalyzeSpikes: 1")
        self.U = utilities.Utility()
        maxspkrate = 50  # max rate to count in adaptation is 50 spikes/second
        minspk = 4  # minimum # of spikes when computing adaptation ratio
        ntraces = len(self.Clamps.traces)
        self.spikecount = np.zeros(ntraces)
        self.fsl = np.zeros(ntraces)
        self.fisi = np.zeros(ntraces)
        self.allisi = []
        self.spikes = [[] for i in range(ntraces)]
        self.spikeIndices = [[] for i in range(ntraces)]

        adapt_ratio = np.zeros(ntraces)
        lastspikecount = 0
        twin = self.Clamps.tend - self.Clamps.tstart  # measurements window in seconds
        # maxspk = int(maxspkrate * twin)  # scale max dount by range of spike counts
        if track:
            print("in analyze spikes")
        for trace_number in range(ntraces):  # this is where we would parallelize the analysis for spikes
            # if we could, but can only have ONE parallelization at a time (using top level)
            # The question is whether this would be faster? 
            if track:
                CP.cprint("r", f"AnalyzeSpikes: 2: trace: {trace_number:04d}", end="\r")
            spikes = self.U.findspikes(
                self.Clamps.time_base,
                np.array(self.Clamps.traces[trace_number]),
                self.threshold,
                t0=self.Clamps.tstart,
                t1=self.Clamps.tend,
                dt=self.Clamps.sample_interval,
                mode=self.mode,  # mode to use for finding spikes
                interpolate=self.interpolate,
                detector=self.spike_detector,
                mindip=1e-2,
                refract=self.refractory,
                peakwidth=self.peakwidth,
                data_time_units=self.time_units,
                data_volt_units=self.volt_units,
                verify=self.verify,
                debug=False,
            )
            if len(spikes) == 0:
                CP.cprint("b", f"    No spikes found, tr: {trace_number:04d}", end="\r")
                continue
            spikes = np.array(spikes)
            # if len(spikes) > 1:
            #     print("min diff time between spikes: ", np.min(np.diff(spikes)))
            self.spikes[trace_number] = spikes
            self.spikeIndices[trace_number] = [np.argmin(np.fabs(self.Clamps.time_base - t)) for t in spikes]
            self.spikecount[trace_number] = len(spikes)
            self.fsl[trace_number] = (spikes[0] - self.Clamps.tstart) * 1e3
            if len(spikes) > 1:
                self.fisi[trace_number] = (spikes[1] - spikes[0]) * 1e3  # first ISI
                self.allisi.append(np.diff(spikes) * 1e3)
            # for Adaptation ratio analysis: limit spike rate, and also only on monotonic increase in rate
            # 8/2018:
            #   Adaptation ratio needs to be tethered to time into stimulus
            #   Here we return a standardized ratio measured during the first 100 msec
            #   (standard ar)
            sp_for_ar = spikes[
                np.where(spikes - self.Clamps.tstart < self.ar_window)
            ] 
            if len(sp_for_ar) >= minspk and (self.spikecount[trace_number] > lastspikecount):
                if sp_for_ar[-1] > self.ar_lastspike + self.Clamps.tstart:  # default 75 msec
                    misi = np.mean(np.diff(sp_for_ar[-2:])) * 1e3  # last ISIs in the interval
                    adapt_ratio[trace_number] = misi / self.fisi[trace_number]
            lastspikecount = self.spikecount[trace_number]  # update rate (sets max rate)
        print()
        iAR = np.where(adapt_ratio > 0)  # valid AR and monotonically rising
        self.adapt_ratio = np.nan
        if len(adapt_ratio[iAR]) > 0:
            self.adapt_ratio = np.mean(adapt_ratio[iAR])  # only where we made the measurement
        self.ar = adapt_ratio  # stores *all* the ar values
        self.analysis_summary["AdaptRatio"] = self.adapt_ratio  # only the valid values
        self.nospk = np.where(self.spikecount == 0)
        self.spk = np.where(self.spikecount > 0)[0]
        self.analysis_summary["FI_Curve"] = np.array([self.Clamps.values, self.spikecount])
        self.analysis_summary["FiringRate"] = np.max(self.spikecount) / (
            twin
        )
        self.spikes_counted = True
        print("spike analysis: AnalyzeSpikes finished")

    def analyzeSpikes_brief(self, mode="baseline"):
        """
        analyzeSpikes_brief: Using the threshold, count the
        number of spikes in a window and fill out an analysis summary dict with
        the spike latencies in that window (from 0 time)

        No analysis of spike shape

        Parameters
        ----------
        mode: str (default: baseline)
            "baseline" from 0 to self.Clamps.tstart
            "poststimulus" from self.Clamps.tend to end of trace
            "evoked" from self.Clamps.start to self.Clamps.end

        Returns
        -------
        Nothing, but see the list of class variables that are modified

        Class variable modified is self.analysis_summary

        """

        if mode == "baseline":
            twin = [0.0, self.Clamps.tstart]
        elif mode == "evoked":
            twin = [self.Clamps.tstart, self.Clamps.tend]
        elif mode == "poststimulus":
            twin = [self.Clamps.tend, np.max(self.Clamps.time_base)]
        else:
            raise ValueError(
                f'{this_source_file:s}:: analyzeSpikes_brief requires mode to be "baseline", "evoked", or "poststimulus"'
            )

        ntr = len(self.Clamps.traces)
        allspikes = [[] for i in range(ntr)]

        for trace_number in range(ntr):
            spikes = self.U.findspikes(
                self.Clamps.time_base,
                np.array(self.Clamps.traces[trace_number]),
                self.threshold,
                t0=twin[0],
                t1=twin[1],
                dt=self.Clamps.sample_interval,
                mode=self.mode,  # mode to use for finding spikes
                interpolate=self.interpolate,
                detector=self.spike_detector,
                refract=self.refractory,
                peakwidth=self.peakwidth,
                verify=self.verify,
                debug=False,
            )
            if len(spikes) == 0:
                continue
            allspikes[trace_number] = spikes

        self.analysis_summary[mode + "_spikes"] = allspikes

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
        return np.argmin(self.Clamps.time_base - t)

    def _initialize_summarymeasures(self):
        self.analysis_summary["AP1_Latency"] = np.inf
        self.analysis_summary["AP1_HalfWidth"] = np.inf
        self.analysis_summary["AP1_HalfWidth_interpolated"] = np.inf
        self.analysis_summary["AP2_Latency"] = np.inf
        self.analysis_summary["AP2_HalfWidth"] = np.inf
        self.analysis_summary["AP2_HalfWidth_interpolated"] = np.inf
        self.analysis_summary["FiringRate_1p5T"] = np.inf
        self.analysis_summary["AHP_Depth"] = np.inf
        self.analysis_summary["AHP_Trough_V"] = np.inf  # depth of trough
        self.analysis_summary["AHP_Trough_T"] = np.inf  # time of trough minimum

    def analyze_one_trace(
        self,
        trace_number,
        begin_dV=12.0,
        max_spikeshape: Union[int, None] = 5,
        printSpikeInfo: bool = False,
    ):
        if len(self.spikes[trace_number]) == 0:
            return
        if printSpikeInfo:
            print(f"{this_source_file:s}:: spikes: ", self.spikes[trace_number])
            print((np.array(self.Clamps.values)))
            print((len(self.Clamps.traces)))
        (self.rmps[trace_number], _) = U.measure(
            "mean",
            self.Clamps.time_base,
            self.Clamps.traces[trace_number],
            0.0,
            self.Clamps.tstart,
        )
        (self.iHold_i[trace_number], _) = U.measure(
            "mean",
            self.Clamps.time_base,
            self.Clamps.cmd_wave[trace_number],
            0.0,
            self.Clamps.tstart,
        )
        trspikes = OrderedDict()
        # if max_spikeshape is None:
        #     jmax = len(self.spikes[i])
        # else:
        #     jmax = max_spikeshape
        # print("# in train: ", i, len(self.spikes[i]))
        for spike_number in range(len(self.spikes[trace_number])):
            # if spike_number >= len(self.spikeIndices[trace_number]):
            #     continue
            # print("trace, spike, beginDV: ", trace_number, spike_number, begin_dV)
            thisspike = self.analyze_one_spike(
                trace_number, spike_number, begin_dV, max_spikeshape=max_spikeshape
            )
            # print("thisspike: ", i, j, thisspike)
            if thisspike is not None:
                trspikes[spike_number] = thisspike
        self.spikeShapes[trace_number] = trspikes

    def analyzeSpikeShape(
        self,
        printSpikeInfo=False,
        spike_begin_dV=12.0,
        max_spikeshape: Union[int, None] = 5,
    ):
        """analyze the spike shape.
        Does analysis of ONE protocol, all traces.
        Based loosely on the analysis from Druckman et al. Cerebral Cortex, 2013

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

        spike_begin_dV : float (default: 12 mV/ms)
            Slope used to define onset of the spike. The default value
            is from Druckmann et al; change this at your own peril!

        max_spikeshape : int (default 5) or None
            Maximum number of spikes for detailed analysis of spike shape. Only the first
            maxspikes in a trial (trace) are analyzed. The rest are counted and latency measured,
            but there is no detailed analysis.
            if None, all spikes are analyzed


        Returns
        -------
        Nothing (but see doc notes above)

        """
        self._initialize_summarymeasures()
        self.madeplot = False
        ntr = len(self.Clamps.traces)
        self.spikeShapes = OrderedDict()
        self.rmps = np.zeros(ntr)
        self.iHold_i = np.zeros(ntr)

        # for i in range(ntr):
        #     self.analyze_one_trace(i, begin_dV, max_spikeshape, printSpikeInfo)

        # parallelize the analysis of the traces

        nWorkers = MP.cpu_count()  # get this automatically
        traces = [tr for tr in range(ntr) if len(self.spikes[tr]) > 0]  # only traces with spikes
        ntraces = len(traces)
        # tasks = [s for s in range(ntr)]
        # tresults = [None]*len(tasks)
        # NOTE: not parallelizing spike processing.
        # with mproc.Parallelize(
        #     enumerate(tasks), results=tresults, workers=nWorkers
        # ) as tasker:
        #     for i, x in tasker:
        #         tr = self.analyze_one_trace(traces[i], begin_dV, max_spikeshape, printSpikeInfo)
        #         tasker.results[i] = tr

        for i in range(ntraces):
            self.analyze_one_trace(traces[i], spike_begin_dV, max_spikeshape, printSpikeInfo)

        self.iHold = np.mean(self.iHold_i)
        self.analysis_summary["spikes"] = self.spikeShapes  # save in the summary dictionary too
        self.analysis_summary["iHold"] = self.iHold
        try:
            self.analysis_summary["pulseDuration"] = self.Clamps.tend - self.Clamps.tstart
        except:
            self.analysis_summary["pulseDuration"] = np.max(self.Clamps.time_base)

        if len(self.spikeShapes.keys()) > 0:  # only try to classify if there are spikes
            lcs = self.get_lowest_current_spike()
            self.analysis_summary["LowestCurrentSpike"] = lcs
            self.getClassifyingInfo()  # build analysis summary here as well.
        else:
            self.analysis_summary["LowestCurrentSpike"] = None

        if printSpikeInfo:
            pp = pprint.PrettyPrinter(indent=4)
            for m in sorted(self.spikeShapes.keys()):
                print(("----\nTrace: %d  has %d APs" % (m, len(list(self.spikeShapes[m].keys())))))
                for n in sorted(self.spikeShapes[m].keys()):
                    pp.pprint(self.spikeShapes[m][n])
        # print("a sp s done")

    def get_lowest_current_spike(self):
        """get_lowest_current_spike : get the lowest current spike in the traces
        and save the various measurements in a dictionary

        Returns
        -------
        dict
            a dictionary of the measurements from the lowest current elicited spike

        """
        dvdts = {}
        # print("LCS: len spike traces: ", len(self.spikeShapes))
        for tr in self.spk:  # for each trace with spikes
            # print("trace: ", tr, "# spikes: ", len(self.spikeShapes[tr]))
            if len(self.spikeShapes[tr]) >= 1:  # only if there is at least one spike
                dvdts[tr] = self.spikeShapes[tr][0]  # get the first spike in the trace
                continue
        # print("get_lowest_current_spike: # traces to investigate: ", len(dvdts))
        LCS = {}
        if len(dvdts) > 0:
            currents = []
            itr = []
            for d in dvdts.keys():  # for each first spike, make a list of the currents
                currents.append(dvdts[d].current)
                itr.append(d)
            i_min_current = np.argmin(currents)  # find spike elicited by the minimum current
            min_current = currents[i_min_current]
            sp = self.spikeShapes[itr[i_min_current]][0]  # gets just the first spike in the lowest current trace
            if sp.AP_begin_V is None:
                print("\nSpike empty? \n", sp)
                return None
            LCS["dvdt_rising"] = sp.dvdt_rising
            LCS["dvdt_falling"] = sp.dvdt_falling
            LCS["dvdt_current"] = min_current * 1e12  # put in pA


            LCS["AP_thr_T"] = sp.AP_beginIndex * self.Clamps.sample_interval * 1e3
            LCS["AP_peak_V"] = 1e3 * sp.peak_V
            LCS["AP_peak_T"] = sp.peak_T
            if sp.halfwidth_interpolated is not None:
                LCS["AP_HW"] = sp.halfwidth_interpolated * 1e3
            else:  # if interpolated halfwidth is not available, use the raw halfwidth
                LCS["AP_HW"] = sp.halfwidth * 1e3

            LCS["AP_begin_V"] = 1e3 * sp.AP_begin_V
            LCS["AHP_depth"] = 1e3 * (sp.AP_begin_V - sp.trough_V)
            LCS["AHP_trough_T"] = sp.trough_T
            LCS["trace"] = itr[i_min_current]
            LCS["dt"] = sp.dt
            LCS["spike_no"] = 0
        return LCS

    def analyze_one_spike(
        self,
        trace_number: int,
        spike_number: int,
        spike_begin_dV: float,
        max_spikeshape: Union[int, None] = None,
    ):
        """analyze_one_spike  Make measurements on a single spike in a trace. To the
        extent possible (and it is not always possible), the measurements include:
        spike threshold, spike height, half-width (data), half-width (interpolated),
        AHP depth, time to AHP depth from time of spike threshold, spike latency,
        and rising and falling slopes of the spike.
        Requires that the spike has been detected and that the spike indices are available.

        Parameters
        ----------
        trace_number : int
            The trace that is being analyzed
        spike_number : int
            The spike # within the trace
        spike_begin_dV : float
            dvdt at spike start
        max_spikeshape : Union[int, None], optional
            Maximum number of spikes within a train that will get this
            detaile analysis. Usually the analysis is only useful on the
            first couple of spikes. By default None

        Returns
        -------
        OneSpike class
            The dataclass that holds information for individual spikes

        Raises
        ------
        ValueError
            Insufficient points before spike to perform analysis
        ValueError
            Insufficient points (< 3) before threshold
        ValueError
            Insufficient points (< 3) before threshold
        """
        thisspike = OneSpike(trace=trace_number, AP_number=spike_number)
        thisspike.current = self.Clamps.values[trace_number] - self.iHold_i[trace_number]
        thisspike.iHold = self.iHold_i[trace_number]
        thisspike.pulseDuration = self.Clamps.tend - self.Clamps.tstart
        thisspike.AP_peakIndex = self.spikeIndices[trace_number][spike_number]
        thisspike.peak_T = self.Clamps.time_base[thisspike.AP_peakIndex]
        thisspike.peak_V = self.Clamps.traces[trace_number][thisspike.AP_peakIndex]
        thisspike.tstart = self.Clamps.tstart
        thisspike.tend = self.Clamps.tend

        dt = self.Clamps.time_base[1] - self.Clamps.time_base[0]
        thisspike.dt = dt
        t_step_start = int(self.Clamps.tstart / dt)
        # compute dv/dt
        dvdt = np.diff(self.Clamps.traces[trace_number]) / dt
        kpeak: int = int(self.spikeIndices[trace_number][spike_number])
        # Check whether there is a previous spike, 
        # and find the minimum voltage between this spike and the previous spike.
        # If this is the first spike, then find 
        # the most proximal minimum before this spike to the start of the trace.
        # Use a 2 mV threshold to find the minimum.
        if spike_number > 0:
            kprevious = self.spikeIndices[trace_number][
                spike_number - 1
            ]  # index into voltage array to previous spike peak
        else:  # no previous spike so look back to the beginning of the trace
            #  to minimize current step artifacts, search for first local minimum prior to the spike
            #  in the window from the peak of the spike to the start of the trance

            min_point = kpeak
            band = 1e-3  # 1 mV change ends minimum search
            for km in range(kpeak-1, t_step_start, -1):
                delta = self.Clamps.traces[trace_number][km] - self.Clamps.traces[trace_number][km + 1]
                if delta < 0:
                    min_point = km
                    min_v = self.Clamps.traces[trace_number][km]  # save current minimum
                    continue
                elif delta > 0:
                    if self.Clamps.traces[trace_number][km] > (min_v + band):
                        break  # end of minimum, report the prior minimum point
            kprevious = min_point
        if kpeak-kprevious <= 2:
            print("peak too close to 'previous' spike: ", trace_number, kprevious, kpeak)
            return thisspike
        kbegin = np.argmin(self.Clamps.traces[trace_number][kprevious:kpeak]) + kprevious
        # raise ValueError(
        #         f"k <= kbegin, can't analyze spike: trace {trace_number:d}, #{spike_number:d} kpeak: {kpeak:d}, kbegin: {kbegin:d}"
        #     )
        #     k = kbegin + 2
        # try:
        #     # get the minimum voltage after the peak to use as the kbegin
        #   #  kbegin = kbegin + int((kpeak - kbegin) / 2)
        km: int = (
            np.argmax(dvdt[kbegin:kpeak]) + kbegin
        )  # find max rising slope, but start halfway between last spike and this spike
        # except ValueError:
        #     print(
        #         f"Not enough points before spike to analyze spike in {this_source_file:s}:: kbegin = {kbegin:d}, kpeak = {kpeak:d}"
        #     )
        #     raise ValueError
        #     # return thisspike

        # if (km - kbegin) < 1:
        #     km = kbegin + int((kpeak - kbegin) / 2) + 1
        # find points where slope exceeds the dv/dt defined for spike threshold, but
        # only up to the max slope of the spike
        kthresholds = np.argwhere(dvdt[kbegin:km] < spike_begin_dV)
        if len(kthresholds) == 0:
            # print(f"No spike found: trace: {trace_number:d}")
            # {kthresh!s}\n     {len(kthresh):d}, {len(dvdt):d}, {kbegin:d}, {kprevious:d}, {kpeak:d}, {spike_begin_dV:.3f}")
            # import pyqtgraph as pg
            # pg.plot(self.Clamps.time_base, self.Clamps.traces[trace_number], pen=pg.mkPen('b', width=1))
            # pg.plot(self.Clamps.time_base[kbegin:kpeak+100], self.Clamps.traces[trace_number][kbegin:kpeak+100],
            #         pen=pg.mkPen('r', width=2))

            # import matplotlib.pyplot as mpl
            # f, ax = mpl.subplots(3, 1, figsize=(5, 8))
            # kb = kbegin - int(0.002/dt)
            # k2 = kpeak + int(0.002/dt)
            # ax[0].plot(self.Clamps.time_base[kb:k2], dvdt[kb:k2])
            # ax[0].plot(self.Clamps.time_base[kb:k2], np.ones_like(dvdt[kb:k2]) * spike_begin_dV)
            # ax[1].plot(self.Clamps.time_base[kb:k2], np.array(self.Clamps.traces[trace_number][kb:k2]))
            # # ax[1].plot(self.Clamps.time_base[kthresh], np.array(self.Clamps.traces[i][kthresh]), 'ro')
            # mpl.show()
            # raise ValueError
            return thisspike

        # then get the LAST point that is above that threshold - this avoids
        # having noise in the trace just below threshold determine the threshold.
        # A better way would be to provide logic that ensures that the threshold represents
        # a monotonically increasing region of the values, or to fit a simple function to
        # smooth out the noise in this region.
        # Note that if there is only one point in kthresholds, then the threshold is the first point
        kthresh:int = int(kthresholds[-1][0]) + kbegin
        # if self.Clamps.time_base[kthresh] < 0.25 + self.Clamps.tstart:
        #     CP("y", f"Spike too early - probably artifact?: {self.Clamps.time_base[kthresh]:.3f}")
        #     return thisspike
        # print(f"kthresh:  {kthresh:d}, kbegin: {kbegin:d}, time: {self.Clamps.time_base[kthresh]:.3f}, detection dvdt: {spike_begin_dV:.3f}, spike dvdt at thresho: {dvdt[kthresh]:.3f}")
        # print(dvdt[kbegin:km])
        # raise ValueError
        # save values in dict here
        thisspike.AP_latency = self.Clamps.time_base[kthresh]
        thisspike.AP_beginIndex = kthresh
        thisspike.AP_begin_V = self.Clamps.traces[trace_number][thisspike.AP_beginIndex]
        if spike_number > max_spikeshape:  # no shape measurements on the rest of the spikes for speed
            # print("Reached spike # > max_spike shape")
            # print("returning latency: ", thisspike.AP_latency)
            return thisspike

        # find end of spike (either top of next spike, or end of trace)
        k = self.spikeIndices[trace_number][spike_number] + 1  # point to next spike
        if spike_number < self.spikecount[trace_number] - 1:  #
            kend = self.spikeIndices[trace_number][spike_number + 1]
        else:
            kend = int(self.spikeIndices[trace_number][spike_number] + self.max_spike_look / dt)
        if kend >= dvdt.shape[0]:
            # raise ValueError()
            return thisspike  # end of spike would be past end of trace
        else:
            if kend < k:
                kend = k + 1
            km = np.argmin(dvdt[k:kend]) + k

        # Find trough after spike and calculate peak to trough
        kmin = np.argmin(self.Clamps.traces[trace_number][km:kend]) + km
        thisspike.AP_endIndex = kmin
        thisspike.trough_T = self.Clamps.time_base[thisspike.AP_endIndex]
        thisspike.trough_V = self.Clamps.traces[trace_number][kmin]

        if thisspike.AP_endIndex is not None:
            thisspike.peaktotrough = thisspike.trough_T - thisspike.peak_T

        # compute rising and falling max dv/dt
        five_ms = int(5e-3 / dt)
        four_ms = int(4e-3 / dt)
        # three_ms = int(3e-3 / dt)
        # two_ms = int(2e-3 / dt)
        one_ms = int(1e-3 / dt)
        thisspike.dvdt_rising = np.max(dvdt[thisspike.AP_beginIndex : thisspike.AP_peakIndex])
        thisspike.dvdt_falling = np.min(dvdt[thisspike.AP_peakIndex : thisspike.AP_endIndex])
        thisspike.dvdt = dvdt[thisspike.AP_beginIndex - four_ms : thisspike.AP_endIndex + one_ms]
        thisspike.V = self.Clamps.traces[trace_number][
            thisspike.AP_beginIndex - four_ms : thisspike.AP_endIndex + five_ms
        ].view(np.ndarray)
        thisspike.Vtime = self.Clamps.time_base[
            thisspike.AP_beginIndex - four_ms : thisspike.AP_endIndex + five_ms
        ].view(np.ndarray)
        # if successful in defining spike start/end, calculate half widths in two ways:
        # closest points in raw data, and by interpolation
        if (
            (thisspike.AP_beginIndex is not None)
            and (thisspike.AP_beginIndex > 0)
            and (thisspike.AP_endIndex is not None)
            and (thisspike.AP_beginIndex < thisspike.AP_peakIndex)
            and (thisspike.AP_peakIndex < thisspike.AP_endIndex)
        ):
            halfv = 0.5 * (thisspike.peak_V + thisspike.AP_begin_V)
            tr = np.array(self.Clamps.traces[trace_number])
            xr = self.Clamps.time_base
            kup = np.argmin(np.fabs(tr[thisspike.AP_beginIndex : thisspike.AP_peakIndex] - halfv))
            kup += thisspike.AP_beginIndex
            kdown = np.argmin(np.fabs(tr[thisspike.AP_peakIndex : thisspike.AP_endIndex] - halfv))
            kdown += thisspike.AP_peakIndex
            if kup is not None and kdown is not None:
                thisspike.halfwidth = xr[kdown] - xr[kup]
                thisspike.halfwidth_up = xr[kup] - xr[thisspike.AP_peakIndex]
                thisspike.halfwidth_down = xr[thisspike.AP_peakIndex] - xr[kdown]
                thisspike.halfwidth_V = halfv
                thisspike.left_halfwidth_T = xr[kup]
                thisspike.left_halfwidth_V = tr[kup]
                thisspike.right_halfwidth_T = xr[kdown]
                thisspike.right_halfwidth_V = tr[kdown]

                # interpolated spike hwup, down and width
                t_hwdown, t_hwup = interpolate_halfwidth(tr, xr, kup, halfv, kdown)
                # print("half-width stuff original: ", thisspike.halfwidth_up, thisspike.halfwidth_down)
                # print("interpolated: ", t_hwup, t_hwdown)

                if t_hwdown is None:
                    return thisspike

                thisspike.halfwidth = t_hwdown - t_hwup
                if thisspike.halfwidth > self.min_halfwidth:  # too broad to be acceptable
                    if self.verbose:
                        # print(
                        #     f"{this_source_file:s}::\n   spikes > min half width",
                        #     thisspike.halfwidth,
                        # )
                        print("   halfv: ", halfv, thisspike.peak_V, thisspike.AP_begin_V)
                    thisspike.halfwidth = None
                    thisspike.halfwidth_interpolated = None

                else:
                    thisspike.halfwidth_interpolated = t_hwdown - t_hwup
                pkvI = tr[thisspike.AP_peakIndex]
                pkvM = np.max(tr[thisspike.AP_beginIndex : thisspike.AP_endIndex])
                pkvMa = np.argmax(tr[thisspike.AP_beginIndex : thisspike.AP_endIndex])
                if pkvI != pkvM:
                    pktrap = True

        return thisspike

    def getIVCurrentThresholds(self):
        """figure out "threshold" for spike, get 150% and 300% points.

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
        for m in sorted(self.spikeShapes.keys()):
            n = len(list(self.spikeShapes[m].keys()))  # number of spikes in the trace
            for n in list(self.spikeShapes[m].keys()):
                icmd.append(self.spikeShapes[m][n].current)
        icmd = np.array(icmd)
        try:
            iamin = np.argmin(icmd)
        except:
            print(f"{this_source_file:s}: Problem with command: ")
            print("self.spikeShapes.keys(): ", self.spikeShapes.keys())
            print("   m = ", m)
            print("   n = ", n)
            print("   current? ", self.spikeShapes[m][n].current)
            raise ValueError(
                f"{this_source_file:s}:getIVCurrentThresholds - icmd seems to be ? : ",
                icmd,
            )

        imin = np.min(icmd)
        ia150 = np.argmin(np.abs(1.5 * imin - icmd))
        iacmdthr = np.argmin(np.abs(imin - self.Clamps.values))
        ia150cmdthr = np.argmin(np.abs(icmd[ia150] - self.Clamps.values))
        return (
            iacmdthr,
            ia150cmdthr,
        )  # return threshold indices into self.Clamps.values array at threshold and 150% point

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
        (
            jthr,
            j150,
        ) = (
            self.getIVCurrentThresholds()
        )  # get the indices for the traces we need to pull data from
        jthr = int(jthr)
        j150 = int(j150)
        if j150 not in list(self.spikeShapes.keys()):
            return
        if jthr == j150 and self.verbose:
            # print '\n%s:' % self.filename
            # print('Threshold current T and 1.5T the same: using next up value for j150')
            # print('jthr, j150, len(spikeShape): ', jthr, j150, len(self.spikeShapes))
            # print('1 ', self.spikeShapes[jthr][0].current*1e12)
            # print('2 ', self.spikeShapes[j150+1][0].current*1e12)
            # print(' >> Threshold current: %8.3f   1.5T current: %8.3f, next up: %8.3f' % (self.spikeShapes[jthr][0].current*1e12,
            #             self.spikeShapes[j150][0].current*1e12, self.spikeShapes[j150+1][0].current*1e12))
            j150 = jthr + 1

        spikesfound = False
        if (
            len(self.spikeShapes[j150]) >= 1
            and (0 in list(self.spikeShapes[j150].keys()))
            and self.spikeShapes[j150][0].halfwidth is not None
        ):
            self.analysis_summary["AP1_Latency"] = (
                self.spikeShapes[j150][0].AP_latency - self.spikeShapes[j150][0].tstart
            ) * 1e3
            self.analysis_summary["AP1_HalfWidth"] = self.spikeShapes[j150][0].halfwidth * 1e3
            if self.spikeShapes[j150][0].halfwidth_interpolated is not None:
                self.analysis_summary["AP1_HalfWidth_interpolated"] = (
                    self.spikeShapes[j150][0].halfwidth_interpolated * 1e3
                )
            else:
                self.analysis_summary["AP1_HalfWidth_interpolated"] = np.nan
            spikesfound = True

        if (
            len(self.spikeShapes[j150]) >= 2
            and (1 in list(self.spikeShapes[j150].keys()))
            and self.spikeShapes[j150][1].halfwidth is not None
        ):
            self.analysis_summary["AP2_Latency"] = (
                self.spikeShapes[j150][1].AP_latency - self.spikeShapes[j150][1].tstart
            ) * 1e3
            self.analysis_summary["AP2_HalfWidth"] = self.spikeShapes[j150][1].halfwidth * 1e3
            if self.spikeShapes[j150][1].halfwidth_interpolated is not None:
                self.analysis_summary["AP2_HalfWidth_interpolated"] = (
                    self.spikeShapes[j150][1].halfwidth_interpolated * 1e3
                )
            else:
                self.analysis_summary["AP2_HalfWidth_interpolated"] = np.nan

        if spikesfound:
            rate = (
                len(self.spikeShapes[j150]) / self.spikeShapes[j150][0].pulseDuration
            )  # spikes per second, normalized for pulse duration
            AHPDepth = (
                self.spikeShapes[j150][0].AP_begin_V - self.spikeShapes[j150][0].trough_V
            )  # from first spike             # first AHP depth
            # print(f"AHP: Begin  = {self.spikeShapes[j150][0].AP_begin_V*1e3:.2f} mV")
            # print(f"     Trough = {self.spikeShapes[j150][0].trough_V*1e3:.2f} mV")
            # print(f"     Depth  = {AHPDepth*1e3:.2f} mV")
            self.analysis_summary["FiringRate_1p5T"] = rate
            self.analysis_summary["AHP_Depth"] = AHPDepth * 1e3  # convert to mV
            self.analysis_summary["AHP_Trough_V"] = self.spikeShapes[j150][0].trough_V  # absolute
            self.analysis_summary["AHP_Trough_T"] = self.spikeShapes[j150][0].trough_T

    def getFISlope(
        self,
        i_inj=None,
        spike_count=None,
        pulse_duration=None,
        min_current: float = 0.0e-12,  # A
        max_current: float = 300e-12,  # A
    ):
        """getFISlope Fit a straight line to part of the FI curve,
        and return the slope value (in spikes/second/nanoampere)

        Parameters
        ----------
        i_inj : numpy array, optional
            Current levels for the FI curve, in A, by default None
            if None, then we use the most recent FI_Curve data from the spike analysis.
        spike_count : numpy array, optional
            Spike COUNTS, by default None
            if None, then we use the most recent FI_Curve spike count
        pulse_duration : float, required
            duration of the current pulse over which spikes were counted, in seconds
        min_current : float, optional
            minimum current in range for fit, by default 0.0 (specify in A)
        max_current : float, optional
            maximum current in range for fit, by defaule 300 pA (specify in A)
        """
        if pulse_duration is None:
            pulse_duration = 1.0  # assume already have spike count in RATE
        if i_inj is None:  # use class data
            i_inj = self.analysis_summary["FI_Curve"][0]
            spike_count = self.analysis_summary["FI_Curve"][1]
            if max_current is not None:
                i_inj = i_inj[i_inj <= max_current]
                spike_count = spike_count[i_inj <= max_current]
        spike_rate = spike_count / pulse_duration  # convert to rate in spikes/second
        window = np.where((i_inj >= min_current) & (i_inj <= max_current))[0]
        if len(window) < 2:  # need at least 2 points for the fit
            return None
        result = scipy.stats.linregress(i_inj[window], spike_rate[window])
        return result  # return the full result

    def fitOne(
        self,
        i_inj=None,
        spike_count=None,
        pulse_duration=None,
        info="",
        function=None,
        fixNonMonotonic=True,
        excludeNonMonotonic=False,
        max_current: Union[float, None] = None,
    ):
        """Fit the FI plot to one of several possible equations.
            1: 'FIGrowthExpBreak' - exponential growth with a breakpoint
            2: 'Hill'
            3:
        Parameters
        ----------
            i_inj : numpy array (no default)
                The x data to fit (typically an array of current levels)

            spike_count : numpy array (no default)
                The y data to fit (typically an array of spike counts)
            if i_inj and spike_count are none, then we extract them from the
                'FI_Curve' for this cell.

            pulse_duration: float or none.
                If float, this should be the duration of the pulse in seconds.
                if None, then we will assume that the spike count is actually corrected spike rate

            info : string (default: '')
                information to add to a fitted plot

            fixNonMonotonic : Boolean (default: True)
                If True, only use data up to the maximal firing rate,
                discarding the remainder of the steps under the assumption
                that the cell is entering depolarization block.

            excludeNonMonotonic : Boolean (default: False)
                if True, does not even try to fit, and returns None

            max_current : float, None (default: None)
                The max current level to include in the fitting,
                to minimize effects of rollover/depolarization block
        Returns
        -------
        None if there is no fitting to be done (excluding non-monotonic or no spikes)
        tuple of (fpar, xf, yf, names, error, f, func)
            These are the fit parameters
        """
        if pulse_duration is None:
            pulse_duration = 1.0  # no correction, assumes spike_count is already converted to rate
        if function is not None:
            self.FIGrowth = function
        if i_inj is None:  # use class data
            i_inj = self.analysis_summary["FI_Curve"][0]
            spike_count = self.analysis_summary["FI_Curve"][1]
            if max_current is not None:
                i_inj = i_inj[i_inj <= max_current]
                spike_count = spike_count[i_inj <= max_current]
        spike_rate = spike_count / pulse_duration  # convert to rate in spikes/second
        spike_rate_max = np.max(spike_rate)
        spike_rate_max_index = np.argmax(spike_rate)  # get where the peak rate is located
        if spike_rate_max <= 0.0:  # max firing rate is 0, no fit
            # CP.cprint("r", "Spike analysis fitOne: Max spike rate is 0")
            return None
        nonmono = 0
        dypos: list = [range(len(spike_rate))]

        # clip to max firing rate to remove non-monotonic rates at high current
        if fixNonMonotonic:  # clip at max firing rate once rate is above the peak rate
            spike_rate_slope: np.ndarray = np.gradient(spike_rate, i_inj)
            xnm = np.where(
                (spike_rate_slope[spike_rate_max_index:] < 0.0)  # find where slope is negative
                & (spike_rate[spike_rate_max_index:] < 0.8 * spike_rate_max)
            )[
                0
            ]  # and rate is less than 80% of max
            if len(xnm) > 0:
                imax = xnm[0] + spike_rate_max_index
            else:
                imax = len(spike_rate)
            dypos = list(range(0, imax))
            i_inj = i_inj[dypos]  # clip to monotonic region
            spike_rate = spike_rate[dypos]  # match rate
            spike_rate_max = np.max(spike_rate)
        if np.max(i_inj) < 0.0:  # no fit if max rate occurs at < 0 current
            # CP.cprint("r", "Spike analysis fitOne: Max current inj is 0")
            return None
        if len(i_inj) < 5:
            return None
        # CP.cprint("m", f"after fixnonmono: {len(i_inj):d} points")
        # get max spike rate
        spike_rate_min = 5.0
        if spike_rate_max < spike_rate_min:
            spike_rate_min = 0.0
        if (
            spike_rate_max > spike_rate[-1] and excludeNonMonotonic
        ):  # no fit if max rate does not occur at the highest current and the flag is set
            nonmono += 1
            CP.cprint("r", "Spike analysis fitOne: exclude non monotonic was triggered")
            return None
        fire_points = np.where((spike_rate[:-1] > 0) & (spike_rate[1:] > 0))[0]
        # CP.cprint("m", f"fire_points: {len(fire_points):d} points")
        # limit to positive current injections with successive spikes
        if len(fire_points) == 0:  # no fit if there are no points in the curve where the cell fires
            return None
        firing_rate_break = fire_points[0]  # this is the "break point" where the first spike occurs
        ibreak0 = i_inj[
            firing_rate_break - 1
        ]  # use point before first spike as the initial break point
        dx = np.abs(np.mean(np.diff(i_inj)))  # get current steps
        xp = i_inj[fire_points]
        xp = xp - ibreak0 - dx
        yp = spike_rate[fire_points]  # save data with responses
        testMethod = "simplex"  #  'SLSQP'  # L-BFGS-B simplex, SLSQP, 'TNC', 'COBYLA'
        if firing_rate_break - 2 >= 0:
            x0 = firing_rate_break - 2
        else:
            x0 = 0
        if firing_rate_break < len(i_inj):
            x1 = firing_rate_break
        else:
            x1 = len(i_inj) - 1

        if self.FIGrowth == "fitOneOriginal":
            res = []
            fitter = fitting.Fitting()  # make sure we always work with the same instance

            for i in range(0, int(len(i_inj) / 2)):  # allow breakpoint to move, but only in steps
                if firing_rate_break + i + 1 > len(i_inj) - 1:
                    continue
                x0 = firing_rate_break + i
                for j in range(0, 4):  # then vary the width of the linear region
                    x1 = x0 + j
                    if x1 >= len(i_inj):
                        continue
                    bounds = (
                        (0.0, 0.0),
                        np.sort([i_inj[x0], i_inj[x1]]),
                        (0.0, yp[0]),
                        (0.0, 4.0 * spike_rate_max * pulse_duration),
                        (0.0, 5.0 * spike_rate_max * pulse_duration / np.max(i_inj)),
                    )
                    # parameters for FIGrowth 1: ['Fzero', 'Ibreak', 'F1amp', 'F2amp', 'Irate']
                    # if i == -4 and j == 0:
                    fitbreak0 = ibreak0
                    initpars = [
                        0.0,
                        np.mean(bounds[1]),
                        yp[0],
                        spike_rate_max * pulse_duration,
                        spike_rate_max * pulse_duration / np.max(i_inj),  # 100 spikes/sec/nA
                    ]
                    function = "FIGrowthExpBreak"
                    f = fitter.fitfuncmap[function]

                    (fpar, xf, yf, names) = fitter.FitRegion(
                        [1],
                        0,
                        i_inj,
                        spike_rate,
                        t0=fitbreak0,
                        t1=np.max(i_inj),
                        fitFunc=function,
                        fitPars=initpars,
                        bounds=bounds,
                        fixedPars=None,
                        method=testMethod,
                    )
                    error = fitter.getFitErr()
                    # keep track of different moves
                    res.append(
                        {
                            "fpar": fpar,
                            "xf": xf,
                            "yf": yf,
                            "names": names,
                            "error": error,
                        }
                    )
            minerr = np.argmin([e["error"][0] for e in res])

            # select the fit with the minimum error
            fpar = res[minerr]["fpar"]
            xf = res[minerr]["xf"]
            yf = res[minerr]["yf"]
            names = res[minerr]["names"]
            error = res[minerr]["error"]

            fitter_func = fitting.Fitting().fitfuncmap[function]
            yfit = fitter_func[0](fpar[0], x=i_inj, C=None)

        elif self.FIGrowth == "FIGrowthExp":  # FIGrowth is 2, Exponential from 0 rate
            bounds = (
                np.sort([i_inj[x0], i_inj[x1]]),
                (0.0, 20.0 * spike_rate_max * pulse_duration),
                (0.0, 10.0 * spike_rate_max * pulse_duration / np.max(i_inj)),
            )
            fitbreak0 = ibreak0
            if fitbreak0 > 0.0:
                fitbreak0 = 0.0
            initpars = [
                ibreak0,
                spike_rate_max * pulse_duration,
                spike_rate_max * pulse_duration / np.max(i_inj),
            ]
            function = "FIGrowthExp"
            f = fitting.Fitting().fitfuncmap[function]
            # now fit the full data set
            (fpar, xf, yf, names) = fitting.Fitting().FitRegion(
                [1.0],
                0,
                i_inj,
                spike_rate,
                t0=fitbreak0,
                t1=np.max(i_inj),
                fitFunc=function,
                fitPars=initpars,
                bounds=bounds,
                fixedPars=None,
                method=testMethod,
            )
            error = fitting.Fitting().getFitErr()
            fitter_func = fitting.Fitting().fitfuncmap[function]
            yfit = fitter_func[0](fpar[0], x=i_inj, C=None)
            self.FIKeys = f[6]
            imap = [-1, 0, -1, 1, 2]

        elif self.FIGrowth == "Hill":
            fitbreak0 = ibreak0
            if fitbreak0 > 0.0:
                fitbreak0 = 0.0
            x1 = np.argwhere((spike_rate > 0.0) & (i_inj > fitbreak0))
            initpars = [0.0, spike_rate_max, 0.5 * np.mean(i_inj[x1]), 1.0]
            bounds = [(0.0, 200.0), (0.0, spike_rate_max * 2.0), (0.0, np.max(i_inj)), (0.0, 10.0)]
            function = "Hill"
            f = fitting.Fitting().fitfuncmap[function]
            # now fit the full data set
            (fpar, xf, yf, names) = fitting.Fitting().FitRegion(
                [1],
                0,
                i_inj,
                spike_rate,
                t0=fitbreak0,
                t1=np.max(i_inj),
                fitFunc=function,
                fitPars=initpars,
                bounds=bounds,
                fixedPars=None,
                method=testMethod,
            )
            error = fitting.Fitting().getFitErr()
            fitter_func = fitting.Fitting().fitfuncmap[function]
            yfit = fitter_func[0](fpar[0], x=i_inj, C=None)
            self.FIKeys = f[6]
            print("spikeanalysis: Hill fit results: ", fpar)

        elif self.FIGrowth == "piecewiselinear3":
            fitbreak0 = ibreak0
            # print('ibreak0: ', ibreak0)
            if fitbreak0 > 0.0:
                fitbreak0 = 0.0
            x1 = np.argwhere(spike_rate > 0.0)
            initpars = (i_inj[x1[0] - 1], 0.0, i_inj[x1[0] + 1], 1.0, 20.0, 100.0)
            if i_inj[firing_rate_break] > 0:
                xn = i_inj[firing_rate_break]
            else:
                xn = 0
            bounds = (
                (0.0, xn),  # Ibreak forced to first spike level almost
                (0.0, 20.0),  # Rate0 (y0)
                (0.0, np.max(i_inj)),  # Ibreak1 (x1)  # spread it out?
                (0.0, 100.0),  # IRate1  (k1, k2, k3)
                (0.0, 1000.0),  # IRate2
                (0.0, 1000.0),  # Irate3
            )
            cons = (
                {"type": "ineq", "fun": lambda x: x[0]},
                {"type": "ineq", "fun": lambda x: x[1]},
                {
                    "type": "ineq",
                    "fun": lambda x: x[2] - (x[0] + 0.05 + ibreak0),
                },  # ibreak1 > 50pA + ibreak0
                {"type": "ineq", "fun": lambda x: x[3]},
                {"type": "ineq", "fun": lambda x: x[4] - x[3]},
                {"type": "ineq", "fun": lambda x: x[5] - x[4] / 2.0},
            )

            function = "piecewiselinear3"
            f = fitting.Fitting().fitfuncmap[function]
            # now fit the full data set
            (fpar, xf, yf, names) = fitting.Fitting().FitRegion(
                [1],
                0,
                i_inj,
                spike_rate,
                t0=fitbreak0,
                t1=np.max(i_inj),
                fitFunc=function,
                fitPars=initpars,
                bounds=bounds,
                constraints=cons,
                fixedPars=None,
                method=testMethod,
            )
            error = fitting.Fitting().getFitErr()
            fitter_func = fitting.Fitting().fitfuncmap[function]
            yfit = fitter_func[0](fpar[0], x=i_inj, C=None)
            self.FIKeys = f[6]

        elif self.FIGrowth == "FIGrowthPower":
            # parameters for power
            # data are only fit for the range over which the cell fires

            fitbreak0 = ibreak0 * 1e9
            if fitbreak0 > 0.0:
                fitbreak0 = 0.0
            ix1 = np.argwhere(spike_rate > 0.0)  # find first point with spikes
            xna = i_inj
            x1 = xna[ix1[0]][0]
            initpars = [x1, spike_rate_max * pulse_duration, 1.0]
            bounds = [
                (0.0, 5e-9),
                (0.0, 20.0 * spike_rate_max * pulse_duration),
                (0.0, 5),
            ]
            # cons = ( {'type': 'ineq', 'fun': lambda x:  x[0]},
            #           {'type': 'ineq', 'fun': lambda x: x[1]},
            #           {'type': 'ineq', 'fun': lambda x: x[2] - [x[0] + 50.]}, # ibreak1 > 100pA + ibreak0
            #           {'type': 'ineq', 'fun': lambda x: x[3]},
            #           {'type': 'ineq', 'fun': lambda x: x[4] - x[3]},
            #           {'type': 'ineq', 'fun': lambda x: x[4]*0.5 - x[5]},
            #      )
            #
            function = "FIGrowthPower"
            # now fit the full data set
            (fpar, xf, yf, names) = fitting.Fitting().FitRegion(
                [1],
                0,
                xna,
                spike_rate,
                t0=fitbreak0,
                t1=np.max(xna),
                fitFunc=function,
                fitPars=initpars,
                bounds=bounds,
                constraints=None,
                fixedPars=None,
                method=testMethod,
            )
            error = fitting.Fitting().getFitErr()
            fitter_func = fitting.Fitting().fitfuncmap[function]
            yfit = fitter_func[0](fpar[0], x=i_inj, C=None)

            self.FIKeys = f[6]
        elif self.FIGrowth == "fitOneOriginal":
            pass
        else:
            raise ValueError("SpikeAnalysis: FIGrowth function %s is not known" % self.FIGrowth)

        self.analysis_summary["FI_Growth"].append(
            {
                "FunctionName": self.FIGrowth,
                "function": function,
                "names": names,
                "error": error,
                "parameters": fpar,
                "fit": [np.array(xf), yf],
                "fit_at_data_points": [np.array(i_inj), np.array(yfit)],
            }
        )
    #     print("LIGHT AS A FEATHER")
    # print("RETURN TO FOREVER")
if __name__ == "__main__":
    pass