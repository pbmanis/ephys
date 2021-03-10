from __future__ import print_function

"""
Classes that provide support functions for minis_methods, 
including fitting, smoothing, filtering, and some analysis.

Test run timing:
cb: 0.175 s (with cython version of algorithm); misses overlapping events
aj: 0.028 s, plus gets overlapping events

July 2017

Note: all values are MKS (Seconds, plus Volts, Amps)
per acq4 standards... 
"""

import numpy as np
import scipy.signal
from dataclasses import dataclass, field
from typing import Union, List
import timeit
from scipy.optimize import curve_fit
import lmfit

import pylibrary.tools.digital_filters as dfilt
from pylibrary.tools.cprint import cprint


@dataclass
class Filtering:
    LPF_applied: bool=False
    HPF_applied: bool=False
    LPF_frequency: Union[float, None]= None
    HPF_frequency: Union[float, None]= None

def def_empty_list():
    return [0] # [0.0003, 0.001]  # in seconds (was 0.001, 0.010)

def def_empty_list2():
    return [[None]] # [0.0003, 0.001]  # in seconds (was 0.001, 0.010)


@dataclass
class AverageEvent:
    """
        The AverageEvent class holds the averaged events from all
        traces/trials
    """
    
    averaged : bool= False  # set flags in case of no events found
    avgeventtb:Union[List, np.ndarray] = field(
        default_factory=def_empty_list)
    avgevent: Union[List, np.ndarray] =field(
        default_factory=def_empty_list)
    Nevents: int = 0
    avgnpts: int = 0
    fitted :bool = False
    fitted_tau1 :float = np.nan
    fitted_tau2 :float = np.nan
    Amplitude :float = np.nan
    avg_fiterr :float = np.nan
    risetenninety:float = np.nan
    decaythirtyseven:float = np.nan
       

@dataclass
class Summaries:
    """
        The Summaries dataclass holdes the results of the
        individual events that were detected,
        as well as the results of various fits
        and the averge fit
    """
    onsets: Union[List, np.ndarray] = field(
        default_factory=def_empty_list2)
    peaks: Union[List, np.ndarray] = field(
        default_factory=def_empty_list)
    smpkindex: Union[List, np.ndarray] = field(
        default_factory=def_empty_list)
    smoothed_peaks : Union[List, np.ndarray] = field(
        default_factory=def_empty_list)
    amplitudes : Union[List, np.ndarray] = field(
        default_factory=def_empty_list)
    Qtotal : Union[List, np.ndarray] = field(
        default_factory=def_empty_list)
    individual_events: bool = False
    average: object = AverageEvent()
    allevents: Union[List, np.ndarray] = field(
        default_factory=def_empty_list)
    event_trace_list : Union[List] = field(
        default_factory=def_empty_list)

class MiniAnalyses:
    def __init__(self):
        """
        Base class for Clements-Bekkers and Andrade-Jonas methods
        Provides template generation, and summary analyses
        Allows use of common methods between different algorithms
        """
        self.verbose = False
        self.ntraces = 1
        self.filtering = Filtering()
        self.risepower = 4.0
        self.min_event_amplitude = 5.0e-12  # pA default
        self.Criterion = [None]
        self.template = None
        self.template_tmax = 0.
        self.analysis_window=[None, None]  # specify window or entire data set
        super().__init__()
        
    def setup(
        self,
        ntraces: int = 1,
        tau1: Union[float, None] = None,
        tau2: Union[float, None] = None,
        template_tmax: float = 0.05,
        dt_seconds: Union[float, None] = None,
        delay: float = 0.0,
        sign: int = 1,
        eventstartthr: Union[float, None] = None,
        risepower: float = 4.0,
        min_event_amplitude: float = 5.0e-12,
        threshold:float = 2.5,
        global_SD:Union[float, None] = None,
        analysis_window:[Union[float, None], Union[float, None]] = [None, None],
        lpf:Union[float, None] = None,
        hpf:Union[float, None] = None
    ) -> None:
        """
        Just store the parameters - will compute when needed
        Use of globalSD and threshold: 
        if glboal SD is None, we use the threshold as it.
    
        If Global SD has a value, then we use that rather than the 
        current trace SD for threshold determinations
        """
        cprint('r', 'SETUP***')
        assert sign in [-1, 1]  # must be selective, positive or negative events only
        self.ntraces = ntraces
        self.Criterion = [[] for x in range(ntraces)]
        self.sign = sign
        self.taus = [tau1, tau2]
        self.dt_seconds = dt_seconds
        self.template_tmax = template_tmax
        self.idelay = int(delay / self.dt_seconds)  # points delay in template with zeros
        self.template = None  # reset the template if needed.
        self.eventstartthr = eventstartthr
        self.risepower = risepower
        self.min_event_amplitude = min_event_amplitude
        self.threshold = threshold
        self.sdthr = self.threshold # for starters
        self.analysis_window = analysis_window
        self.lpf = lpf
        self.hpf = hpf

    def set_sign(self, sign: int = 1):
        self.sign = sign

    def set_risepower(self, risepower: float = 4):
        if risepower > 0 and risepower <= 8:
            self.risepower = risepower
        else:
            raise ValueError("Risepower must be 0 < n <= 8")

    def _make_template(self):
        """
        Private function: make template when it is needed
        """
        tau_1, tau_2 = self.taus  # use the predefined taus
        t_psc = np.arange(0, self.template_tmax, self.dt_seconds)
        self.t_template = t_psc
        Aprime = (tau_2 / tau_1) ** (tau_1 / (tau_1 - tau_2))
        self.template = np.zeros_like(t_psc)
        tm = (
            1.0
            / Aprime
            * (
                (1 - (np.exp(-t_psc / tau_1))) ** self.risepower
                * np.exp((-t_psc / tau_2))
            )
        )
        # tm = 1./2. * (np.exp(-t_psc/tau_1) - np.exp(-t_psc/tau_2))
        if self.idelay > 0:
            self.template[self.idelay :] = tm[: -self.idelay]  # shift the template
        else:
            self.template = tm
        if self.sign > 0:
            self.template_amax = np.max(self.template)
        else:
            self.template = -self.template
            self.template_amax = np.min(self.template)


    def reset_filtering(self):
        self.filtering.LPF_applied = False
        self.filtering.HPF_applied = False
        
    def LPFData(
        self, data: np.ndarray, lpf: Union[float, None] = None, NPole: int = 8
    ) -> np.ndarray:
        assert (not self.filtering.LPF_applied)  # block repeated application of filtering
        # cprint('y', f"minis_methods_common, LPF data:  {lpf:f}")
        # old_data = data.copy()
        if lpf is not None : 
            # cprint('y', f"     ... lpf at {lpf:f}")
            if lpf > 0.49 / self.dt_seconds:
                raise ValueError(
                    "lpf > Nyquist: ", lpf, 0.49 / self.dt_seconds, self.dt_seconds, 1.0 / self.dt_seconds
                )
            data = dfilt.SignalFilter_LPFButter(data, lpf, 1./self.dt_seconds, NPole=8)
            self.filtering.LPF = lpf
            self.filtering.LPF_applied = True
        # import matplotlib.pyplot as mpl
        # print(old_data.shape[0]*self.dt_seconds)
        # tb = np.arange(0, old_data.shape[0]*self.dt_seconds, self.dt_seconds)
        # print(tb.shape)
        # mpl.plot(tb,  old_data, 'b-')
        # mpl.plot(tb, data, 'k-')
        # mpl.show()
        # exit()
        return data

    def HPFData(self, data:np.ndarray, hpf: Union[float, None] = None, NPole: int = 8) -> np.ndarray:
        assert (not self.filtering.HPF_applied)  # block repeated application of filtering
        if hpf is None or hpf == 0.0 :
            return data
        if len(data.shape) == 1:
            ndata = data.shape[0]
        else:
            ndata = data.shape[1]
        nyqf = 0.5 * ndata * self.dt_seconds
        # cprint('y', f"minis_methods: hpf at {hpf:f}")
        if hpf < 1.0 / nyqf:  # duration of a trace
            raise ValueError(
                "hpf < Nyquist: ",
                hpf,
                "nyquist",
                1.0 / nyqf,
                "ndata",
                ndata,
                "dt in seconds",
                self.dt_seconds,
                "sampelrate",
                1.0 / self.dt,
            )
        
        data = dfilt.SignalFilter_HPFButter(data-data[0], hpf, 1.0 / self.dt_seconds, NPole=4)
        self.filtering.HPF = hpf
        self.filtering.HPF_applied = True

        return data
    
    def prepare_data(self, data):
        """
        This function prepares the incoming data for the mini analyses.
        1. Clip the data in time (remove sections with current or voltage steps)
        2. Filter the data (LPF, HPF)
        """
        # cprint('r', 'Prepare data')

        self.timebase = np.arange(0.0, data.shape[0] * self.dt_seconds, self.dt_seconds)
        if self.analysis_window[1] is not None:
            jmax = np.argmin(np.fabs(self.timebase - self.analysis_window[1]))
        else:
            jmax = len(self.timebase)
        if self.analysis_window[0] is not None:
            jmin = np.argmin(np.fabs(self.timebase) - self.analysis_window[0])
        else:
            jmin = 0
        data = data[jmin:jmax]
        if self.verbose:
            if self.lpf is not None:
                cprint('y', f"minis_methods_common, prepare_data: LPF: {self.lpf:.1f} Hz")
            else:
                cprint('r', f"minis_methods_common, no LPF applied")
            if self.hpf is not None:
                cprint('y', f"minis_methods_common, prepare_data: HPF: {self.hpf:.1f} Hz")
            else:
                cprint('r', f"minis_methods_common, no HPF applied")
        if isinstance(self.lpf, float):
            data = self.LPFData(data, lpf=self.lpf)
        if isinstance(self.hpf, float):
            data = self.HPFData(data, hpf=self.hpf)
        self.data = data
        self.timebase = self.timebase[jmin:jmax]

    def moving_average(self, a, n: int = 3) -> (np.array, int):
        ret = np.cumsum(a, dtype=float)
        ret[n:] = ret[n:] - ret[:-n]
        # return ret[n - 1 :] / n, n
        return ret[int(n/2):] / n, n  # re-align array
        
    def remove_outliers(self, x:np.ndarray, scale:float=3.0) -> np.ndarray:
        a = np.array(x)
        upper_quartile = np.percentile(a, 75)
        lower_quartile = np.percentile(a, 25)
        IQR = (upper_quartile - lower_quartile) * scale
        quartileSet = (lower_quartile - IQR, upper_quartile + IQR)
        result = np.where(((a >= quartileSet[0]) & (a <= quartileSet[1])), a, np.nan)
        # import matplotlib.pyplot as mpl
        # mpl.plot(x)
        # mpl.plot(result)
        # mpl.show()
        return result
    

    def summarize(self, data, order: int = 11, verbose: bool = False) -> None:
        """
        compute intervals,  peaks and ampitudes for all found events in a
        trace or a group of traces
        filter out events that are less than min_event_amplitude
        """
        i_decay_pts = int(2.0 * self.taus[1] / self.dt_seconds)  # decay window time (points) Units all seconds
        assert i_decay_pts > 5
        
        self.Summary = Summaries()  # a single summary class is created
        ndata = len(data)
        # set up arrays : note construction to avoid "same memory but different index" problem
        self.Summary.onsets = [[] for x in range(ndata)]
        self.Summary.peaks = [[] for x in range(ndata)]
        self.Summary.smoothed_peaks = [[] for x in range(ndata)]
        self.Summary.smpkindex = [[] for x in range(ndata)]
        self.Summary.amplitudes = [[] for x in range(ndata)]
        self.Summary.filtered_traces = [[] for x in range(ndata)]
        avgwin = (
            5  # int(1.0/self.dt_seconds)  # 5 point moving average window for peak detection
        )

        mwin = int((0.50) / self.dt_seconds)
        if self.sign > 0:
            nparg = np.greater
        else:
            nparg = np.less
        self.intervals = []
        self.timebase = np.arange(0., data.shape[1]*self.dt_seconds, self.dt_seconds)

        for itrial, dataset in enumerate(data):  # each trial/trace
            print(len(self.onsets), itrial)
            if len(self.onsets[itrial]) == 0:  # original events
                continue
            cprint('c', f"Onsets found: {len(self.onsets[itrial]):d} in trial {itrial:d}")
            acceptlist_trial = []
            self.intervals.append(np.diff(self.timebase[self.onsets[itrial]]))  # event intervals
            # cprint('y', f"Summarize: trial: {itrial:d} onsets: {len(self.onsets[itrial]):d}")
            # print('onsets: ', self.onsets[itrial])
            ev_accept = []
            for j, onset in enumerate(self.onsets[itrial]):  # for all of the events in this trace
                if self.sign > 0 and self.eventstartthr is not None:
                    if dataset[onset] < self.eventstartthr:
                        continue
                if self.sign < 0 and self.eventstartthr is not None:
                    if dataset[onset] > -self.eventstartthr:
                        continue
                event_data = dataset[onset : (onset + mwin)]  # get this event
                svwinlen = event_data.shape[0]
                if svwinlen > 11:
                    svn = 11
                else:
                    svn = svwinlen
                if (
                    svn % 2 == 0
                ):  # if even, decrease by 1 point to meet ood requirement for savgol_filter
                    svn -= 1
                if svn > 3:  # go ahead and filter
                    p = scipy.signal.argrelextrema(
                        scipy.signal.savgol_filter(
                            event_data, svn, 2
                        ),
                        nparg,
                        order=order,
                    )[0]
                else:  # skip filtering
                    print('mmethods_common summarize order: ', order)
                    p = scipy.signal.argrelextrema(
                        event_data,
                        nparg,
                        order=order,
                    )[0]
                # print('len(p): ', len(p), svn, event_data)
                if len(p) > 0:
                    # print('p, idecay onset: ', len(p), i_decay_pts, onset)
                    i_end = i_decay_pts + onset # distance from peak to end
                    i_end = min(dataset.shape[0], i_end)  # keep within the array limits
                    if j < len(self.onsets[itrial]) - 1:
                        if i_end > self.onsets[itrial][j + 1]:
                            i_end = (
                                self.onsets[itrial][j + 1] - 1
                            )  # only go to next event start
                    windowed_data = dataset[onset : i_end]
                    # print('onset, iend: ', onset, i_end)
                    # import matplotlib.pyplot as mpl
                    # fx, axx = mpl.subplots(1,1)
                    # axx.plot(self.timebase[onset:i_end], dataset[onset:i_end], 'g-')
                    # mpl.show()
                    move_avg, n = self.moving_average(
                        windowed_data,
                        n=min(avgwin, len(windowed_data)),
                    )
                    # print('moveavg: ', move_avg)
                    # print(avgwin, len(windowed_data))
                    # print('windowed_data: ', windowed_data)
                    if self.sign > 0:
                        smpk = np.argmax(move_avg)  # find peak of smoothed data
                        rawpk = np.argmax(windowed_data)  # non-smoothed
                    else:
                        smpk = np.argmin(move_avg)
                        rawpk = np.argmin(windowed_data)
                    if self.sign*(move_avg[smpk] - windowed_data[0]) < self.min_event_amplitude:
                        print('too small: ', self.sign*(move_avg[smpk] - windowed_data[0]), 'vs. ', self.min_event_amplitude)
                        continue  # filter out events smaller than the amplitude
                    else:
                        # print('accept: ', j)
                        ev_accept.append(j)
                    # cprint('m', f"Extending for trial: {itrial:d}, {len(self.Summary.onsets[itrial]):d}, onset={onset}")
                    self.Summary.onsets[itrial].append(onset)
                    self.Summary.peaks[itrial].append(onset + rawpk)
                    self.Summary.amplitudes[itrial].append(windowed_data[rawpk])
                    self.Summary.smpkindex[itrial].append(onset + smpk)
                    self.Summary.smoothed_peaks[itrial].append(move_avg[smpk])
                    acceptlist_trial.append(j)

            self.onsets[itrial] = self.onsets[itrial][ev_accept]  # reduce to the accepted values only
        # self.Summary.smoothed_peaks = np.array(self.Summary.smoothed_peaks)
        # self.Summary.amplitudes = np.array(self.Summary.amplitudes)
        
        self.average_events(
            data,
        )
        if self.Summary.average.averaged:
            self.fit_average_event(
                tb=self.Summary.average.avgeventtb,
                avgevent=self.Summary.average.avgevent,
                initdelay=0.,
                debug=False)

        else:
            if verbose:
                print("No events found")
            return

    def measure_events(self, data:object, eventlist: list) -> dict:
        # compute simple measurements of events (area, amplitude, half-width)
        #
        # cprint('r', 'MEASURE EVENTS')
        assert data.ndim == 1
        self.measured = False
        # treat like averaging
        tdur = np.max((np.max(self.taus) * 5.0, 0.010))  # go 5 taus or 10 ms past event
        tpre = 0.0  # self.taus[0]*10.
        self.avgeventdur = tdur
        self.tpre = tpre
        self.avgnpts = int((tpre + tdur) / self.dt_seconds)  # points for the average
        npre = int(tpre / self.dt_seconds)  # points for the pre time
        npost = int(tdur / self.dt_seconds)
        avg = np.zeros(self.avgnpts)
        avgeventtb = np.arange(self.avgnpts) * self.dt_seconds
        # assert True == False
        allevents = np.zeros((len(eventlist), self.avgnpts))
        k = 0
        pkt = 0  # np.argmax(self.template)  # accumulate
        meas = {"Q": [], "A": [], "HWup": [], "HWdown": [], "HW": []}
        for j, i in enumerate(eventlist):
            ix = i + pkt  # self.idelay
            if (ix + npost) < len(self.data) and (ix - npre) >= 0:
                allevents[k, :] = data[ix - npre : ix + npost]
                k = k + 1
        if k > 0:
            allevents = allevents[0:k, :]  # trim unused
            for j in range(k):
                ev_j = scipy.signal.savgol_filter(
                    self.sign * allevents[j, :], 7, 2, mode="nearest"
                )  # flip sign if negative
                ai = np.argmax(ev_j)
                if ai == 0:
                    continue  # skip events where max is first point
                q = np.sum(ev_j) * tdur
                meas["Q"].append(q)
                meas["A"].append(ev_j[ai])
                hw_up = self.dt_seconds * np.argmin(np.fabs((ev_j[ai] / 2.0) - ev_j[:ai]))
                hw_down = self.dt_seconds * np.argmin(np.fabs(ev_j[ai:] - (ev_j[ai] / 2.0)))
                meas["HWup"].append(hw_up)
                meas["HWdown"].append(hw_down)
                meas["HW"].append(hw_up + hw_down)
            self.measured = True
            self.Summary.allevents = allevents
        else:
            self.measured = False
            self.Summary.allevents = None
        return meas

    def average_events(self, data: np.ndarray) -> tuple:
        """
        compute average event with length of template
        Parameters
        ----------
        eventlist : list
            List of event onset indices into the arrays
            Expect a 2-d list (traces x onsets)
        """ 
        # cprint('r', 'AVERAGE EVENTS')
        self.Summary.average.averaged = False
        tdur = np.max((np.max(self.taus) * 5.0, 0.010))  # go 5 taus or 10 ms past event
        tpre = 0.0  # self.taus[0]*10.
        avgeventdur = tdur
        self.tpre = tpre
        avgnpts = int((tpre + tdur) / self.dt_seconds)  # points for the average
        npre = int(tpre / self.dt_seconds)  # points for the pre time
        npost = int(tdur / self.dt_seconds)
        avg = np.zeros(avgnpts)
        avgeventtb = np.arange(avgnpts) * self.dt_seconds
        n_events = sum([len(events) for events in self.Summary.onsets])
        allevents = np.zeros((n_events, avgnpts))
        event_trace = [[]]*n_events
        k = 0
        pkt = 0
        for itrace, onsets in enumerate(self.Summary.onsets):
            # cprint('c', f"Trace: {itrace: d}, # onsets: {len(onsets):d}")
            for j, event_onset in enumerate(onsets):
                ix = event_onset + pkt  # self.idelay
                # print('itrace, ix, npre, npost: ', itrace, ix, npre, npost)
                if (ix + npost) < data[itrace].shape[0] and (ix - npre) >= 0:
                    allevents[k, :] = data[itrace, (ix - npre) : (ix + npost)]
                else:
                    allevents[k, :] = np.nan*allevents[k,:]
                event_trace[k] = [itrace, j]
                k = k + 1
        # tr_incl = [u[0] for u in event_trace]
        # print(set(tr_incl), len(set(tr_incl)), len(event_trace))
        # exit()
        if k > 0:
            self.Summary.average.averaged = True
            self.Summary.average.avgnpts = avgnpts
            self.Summary.average.Nevents = k
            self.Summary.allevents = allevents
            avgevent = allevents.mean(axis=0)
            self.Summary.average.avgevent = avgevent - np.mean(avgevent[:3])
            self.Summary.average.avgeventtb = avgeventtb
            self.Summary.event_trace_list = event_trace
            return
        else:
            self.Summary.average.avgnpts = 0
            self.Summary.average.avgevent = []
            self.Summary.average.allevents = []
            self.Summary.average.avgeventtb = []
            self.Summary.average.averaged = False
            self.Summary.event_trace_list = []
            return

    def average_events_subset(self, data: np.ndarray, eventlist:list) -> tuple:
        """
        compute average event with length of template
        Parameters
        ----------
        data:
            1-d numpy array of the data
            eventlist : list
            List of event onset indices into the arrays
            Expect a 1-d list (traces x onsets)
        """ 
        assert data.ndim == 1
        # cprint('r', 'AVERAGE EVENTS')
        tdur = np.max((np.max(self.taus) * 5.0, 0.010))  # go 5 taus or 10 ms past event
        tpre = 0.0  # self.taus[0]*10.
        avgeventdur = tdur
        self.tpre = tpre
        avgnpts = int((tpre + tdur) / self.dt_seconds)  # points for the average
        npre = int(tpre / self.dt_seconds)  # points for the pre time
        npost = int(tdur / self.dt_seconds)
        avg = np.zeros(avgnpts)
        avgeventtb = np.arange(avgnpts) * self.dt_seconds
        n_events = sum([len(events) for events in self.Summary.onsets])
        allevents = np.zeros((n_events, avgnpts))
        event_trace = [None]*n_events
        k = 0
        pkt = 0
        for itrace, event_onset in enumerate(eventlist):
            # cprint('c', f"Trace: {itrace: d}, # onsets: {len(onsets):d}")
            ix = event_onset + pkt  # self.idelay
            # print('itrace, ix, npre, npost: ', itrace, ix, npre, npost)
            if (ix + npost) < data.shape[0] and (ix - npre) >= 0:
                allevents[k, :] = data[(ix - npre) : (ix + npost)]
                k = k + 1
        return np.mean(allevents, axis=0), avgeventtb, allevents


    def doubleexp(
        self,
        p: list,
        x: np.ndarray,
        y: Union[None, np.ndarray],
        risepower: float,
        fixed_delay: float = 0.0,
        mode: int = 0,
    ) -> np.ndarray:
        """
        Calculate a double expoential EPSC-like waveform with the rise to a power
        to make it sigmoidal
        """
        # fixed_delay = p[3]  # allow to adjust; ignore input value
        ix = np.argmin(np.fabs(x - fixed_delay))
        tm = np.zeros_like(x)
        tm[ix:] = p[0] * (1.0 - np.exp(-(x[ix:] - fixed_delay) / p[1])) ** risepower
        tm[ix:] *= np.exp(-(x[ix:] - fixed_delay) / p[2])

        if mode == 0:
            return tm - y
        elif mode == 1:
            return np.linalg.norm(tm - y)
        elif mode == -1:
            return tm
        else:
            raise ValueError(
                "doubleexp: Mode must be 0 (diff), 1 (linalg.norm) or -1 (just value)"
            )

    def risefit(
        self,
        p: list,
        x: np.ndarray,
        y: Union[None, np.ndarray],
        risepower: float,
        mode: int = 0,
    ) -> np.ndarray:
        """
        Calculate a delayed EPSC-like waveform rise shape with the rise to a power
        to make it sigmoidal, and an adjustable delay
        input data should only be the rising phase.
        p is in order: [amplitude, tau, delay]
        """
        assert mode in [-1, 0, 1]
        ix = np.argmin(np.fabs(x - p[2]))
        tm = np.zeros_like(x)
        expf = (x[ix:] - p[2]) / p[1]
        pclip = 1.0e3
        nclip = 0.0
        expf[expf > pclip] = pclip
        expf[expf < -nclip] = -nclip
        tm[ix:] = p[0] * (1.0 - np.exp(-expf)) ** risepower
        if mode == 0:
            return tm - y
        elif mode == 1:
            return np.linalg.norm(tm - y)
        elif mode == -1:
            return tm
        else:
            raise ValueError(
                "doubleexp: Mode must be 0 (diff), 1 (linalg.norm) or -1 (just value)"
            )

    def decayexp(
        self,
        p: list,
        x: np.ndarray,
        y: Union[None, np.ndarray],
        fixed_delay: float = 0.0,
        mode: int = 0,
    ):
        """
        Calculate an exponential decay (falling phase fit)
        """
        tm = p[0] * np.exp(-(x - fixed_delay) / p[1])
        if mode == 0:
            return tm - y
        elif mode == 1:
            return np.linalg.norm(tm - y)
        elif mode == -1:
            return tm
        else:
            raise ValueError(
                "doubleexp: Mode must be 0 (diff), 1 (linalg.norm) or -1 (just value)"
            )

    def fit_average_event(
        self,
        tb,
        avgevent,
        debug: bool = False,
        label: str = "",
        inittaus: List = [0.001, 0.005],
        initdelay: Union[float, None] = None,
    ) -> None:
        """
        Fit the averaged event to a double exponential epsc-like function
        Operates on the AverageEvent data structure
        """
        # tsel = np.argwhere(self.avgeventtb > self.tpre)[0]  # only fit data in event,  not baseline
        tsel = 0  # use whole averaged trace
        self.tsel = tsel
        self.tau1 = inittaus[0]
        self.tau2 = inittaus[1]
        self.tau2_range = 10.0
        self.tau1_minimum_factor = 5.0
        time_past_peak = 2.5e-4
        self.fitted_tau1 = np.nan
        self.fitted_tau2 = np.nan
        self.Amplitude = np.nan
        # peak_pos = np.argmax(self.sign*self.avgevent[self.tsel:])
        # decay_fit_start = peak_pos + int(time_past_peak/self.dt_seconds)
        # init_vals = [self.sign*10.,  1.0,  4., 0.]
        # init_vals_exp = [20.,  5.0]
        # bounds_exp  = [(0., 0.5), (10000., 50.)]

        res, rdelay = self.event_fitter(
            tb,
            avgevent,
            time_past_peak=time_past_peak,
            initdelay=initdelay,
            debug=debug,
            label=label,
        )
        # print('rdelay: ', rdelay)
        if res is None:
            self.fitted = False
            return
        self.fitresult = res.x
        self.Amplitude = self.fitresult[0]
        self.fitted_tau1 = self.fitresult[1]
        self.fitted_tau2 = self.fitresult[2]
        self.bfdelay = rdelay
        self.avg_best_fit = self.doubleexp(
            self.fitresult,
            tb[self.tsel :],
            np.zeros_like(tb[self.tsel :]),
            risepower=self.risepower,
            mode=0,
            fixed_delay=self.bfdelay,
        )
        self.avg_best_fit = self.sign * self.avg_best_fit
        fiterr = np.linalg.norm(self.avg_best_fit - 
            avgevent[self.tsel :])
        self.avg_fiterr = fiterr
        ave = self.sign * avgevent
        ipk = np.argmax(ave)
        pk = ave[ipk]
        p10 = 0.1 * pk
        p90 = 0.9 * pk
        p37 = 0.37 * pk
        try:
            i10 = np.argmin(np.fabs(ave[:ipk] - p10))
        except:
            self.fitted = False
            return
        i90 = np.argmin(np.fabs(ave[:ipk] - p90))
        i37 = np.argmin(np.fabs(ave[ipk:] - p37))
        self.risetenninety = self.dt_seconds * (i90 - i10)
        self.decaythirtyseven = self.dt_seconds * (i37 - ipk)
        self.Qtotal = self.dt_seconds * np.sum(avgevent[self.tsel :])
        self.fitted = True

    def fit_individual_events(self, onsets: np.ndarray) -> None:
        """
        Fitting individual events
        Events to be fit are selected from the entire event pool as:
        1. events that are completely within the trace, AND
        2. events that do not overlap other events
        
        Fit events are further classified according to the fit error
        
        """
        if (
            not self.averaged or not self.fitted
        ):  # averaging should be done first: stores events for convenience and gives some tau estimates
            print("Require fit of averaged events prior to fitting individual events")
            raise (ValueError)
        time_past_peak = 0.75  # msec - time after peak to start fitting

        # allocate arrays for results. Arrays have space for ALL events
        # okevents, notok, and evok are indices
        nevents = len(self.Summary.allevents)  # onsets.shape[0]
        self.ev_fitamp = np.zeros(nevents)  # measured peak amplitude from the fit
        self.ev_A_fitamp = np.zeros(
            nevents
        )  # fit amplitude - raw value can be quite different than true amplitude.....
        self.ev_tau1 = np.zeros(nevents)
        self.ev_tau2 = np.zeros(nevents)
        self.ev_1090 = np.zeros(nevents)
        self.ev_2080 = np.zeros(nevents)
        self.ev_amp = np.zeros(nevents)  # measured peak amplitude from the event itself
        self.ev_Qtotal = np.zeros(
            nevents
        )  # measured charge of the event (integral of current * dt)
        self.fiterr = np.zeros(nevents)
        self.bfdelay = np.zeros(nevents)
        self.best_fit = np.zeros((nevents, self.avgeventtb.shape[0]))
        self.best_decay_fit = np.zeros((nevents, self.avgeventtb.shape[0]))
        self.tsel = 0
        self.tau2_range = 10.0
        self.tau1_minimum_factor = 5.0

        # prescreen events
        minint = self.avgeventdur  # msec minimum interval between events.
        self.fitted_events = (
            []
        )  # events that can be used (may not be all events, but these are the events that were fit)
        for i in range(nevents):
            te = self.timebase[onsets[i]]  # get current event
            try:
                tn = self.timebase[onsets[i + 1]]  # check time to next event
                if tn - te < minint:  # event is followed by too soon by another event
                    continue
            except:
                pass  # just handle trace end condition
            try:
                tp = self.timebase[onsets[i - 1]]  # check previous event
                if (
                    te - tp < minint
                ):  # if current event too close to a previous event, skip
                    continue
                self.fitted_events.append(i)  # passes test, include in ok events
            except:
                pass

        for n, i in enumerate(self.fitted_events):
            try:
                max_event = np.max(self.sign * self.Summary.allevents[i, :])
            except:
                print("minis_methods eventfitter")
                print("fitted: ", self.fitted_events)
                print("i: ", i)
                print("allev: ", self.Summary.allevents)
                print("len allev: ", len(self.Summary.allevents), len(onsets))
                raise ValueError('Fit failed)')
            res, rdelay = self.event_fitter(
                self.avgeventtb, self.Summmary.allevents[i, :], time_past_peak=time_past_peak
            )
            if res is None:   # skip events that won't fit
                continue
            self.fitresult = res.x

            # lmfit version - fails for odd reason
            # dexpmodel = Model(self.doubleexp)
            # params = dexpmodel.make_params(A=-10.,  tau_1=0.5,  tau_2=4.0,  dc=0.)
            # self.fitresult = dexpmodel.fit(self.avgevent[tsel:],  params,  x=self.avgeventtb[tsel:])
            self.ev_A_fitamp[i] = self.fitresult[0]
            self.ev_tau1[i] = self.fitresult[1]
            self.ev_tau2[i] = self.fitresult[2]
            self.bfdelay[i] = rdelay
            self.fiterr[i] = self.doubleexp(
                self.fitresult,
                self.avgeventtb,
                self.sign * self.Summary.allevents[i, :],
                risepower=self.risepower,
                fixed_delay=self.bfdelay[i],
                mode=1,
            )
            self.best_fit[i] = self.doubleexp(
                self.fitresult,
                self.avgeventtb,
                np.zeros_like(self.avgeventtb),
                risepower=self.risepower,
                fixed_delay=self.bfdelay[i],
                mode=0,
            )
            self.best_decay_fit[i] = self.decay_fit  # from event_fitter
            self.ev_fitamp[i] = np.max(self.best_fit[i])
            self.ev_Qtotal[i] = self.dt_seconds * np.sum(self.sign * self.Summary.allevents[i, :])
            self.ev_amp[i] = np.max(self.sign * self.Summary.allevents[i, :])
        self.individual_event_screen(fit_err_limit=2000.0, tau2_range=10.0)
        self.individual_events = True  # we did this step

    def event_fitter(
        self,
        timebase: np.ndarray,
        event: np.ndarray,
        time_past_peak: float = 0.0001,
        initdelay: Union[float, None] = None,
        debug: bool = False,
        label: str = "",
    ) -> (dict, float):
        """
        Fit the event
        Procedure:
        First we fit the rising phase (to the peak) with (1-exp(t)^n), allowing
        the onset of the function to slide in time. This onset time is locked after this step
        to minimize trading in the error surface between the onset and the tau values.
        Second, we fit the decay phase, starting just past the peak (and accouting for the fixed delay)
        Finally, we combine the parameters and do a final optimization with somewhat narrow
        bounds.
        Fits are good on noiseless test data. 
        Fits are affected by noise on the events (of course), but there is no "systematic"
        variation that is present in terms of rise-fall tau tradeoffs.
        
        """
        debug = False
        if debug:
            import matplotlib.pyplot as mpl


        ev_bl = np.mean(event[: int(initdelay / self.dt_seconds)])  # just first point...
        evfit = self.sign * (event - ev_bl)
        maxev = np.max(evfit)
        if maxev == 0:
            maxev = 1
        # if peak_pos == 0:
        #     peak_pos = int(0.001/self.dt_seconds) # move to 1 msec later
        evfit = evfit / maxev  # scale to max of 1
        peak_pos = np.argmax(evfit) + 1
        amp_bounds = [0.0, 1.0]
        # set reasonable, but wide bounds, and make sure init values are within bounds
        # (and off center, but not at extremes)

        bounds_rise = [amp_bounds, (self.dt_seconds, 4.0 * self.dt_seconds * peak_pos), (0.0, 0.005)]
        if initdelay is None or initdelay < self.dt_seconds:
            fdelay = 0.2 * np.mean(bounds_rise[2])
        else:
            fdelay = initdelay
        if fdelay > self.dt_seconds * peak_pos:
            fdelay = 0.2 * self.dt_seconds * peak_pos
        init_vals_rise = [0.9, self.dt_seconds * peak_pos, fdelay]

        try:
            res_rise = scipy.optimize.minimize(
                self.risefit,
                init_vals_rise,
                bounds=bounds_rise,
                method="SLSQP",  # x_scale=[1e-12, 1e-3, 1e-3],
                args=(
                    timebase[:peak_pos],  # x
                    evfit[:peak_pos],  # 'y
                    self.risepower,
                    1,
                ),  # risepower, mode
            )
        except:
            # mpl.plot(self.timebase[:peak_pos], evfit[:peak_pos], 'k-')
  #           mpl.show()
  #           print('risefit: ', self.risefit)
  #           print('init_vals_rise: ', init_vals_rise)
  #           print('bounds rise: ', bounds_rise)
  #           print('peak_pos: ', peak_pos)
            return None, None
            # raise ValueError()
            
        if debug:
            import matplotlib.pyplot as mpl

            f, ax = mpl.subplots(2, 1)
            ax[0].plot(timebase, evfit, "-k")
            ax[1].plot(timebase[:peak_pos], evfit[:peak_pos], "-k")
            print("\nrise fit:")
            ax[1].set_title('To peak (black), to end (red)')
            print("dt: ", self.dt_second, " maxev: ", maxev, " peak_pos: ", peak_pos)
            print("bounds: ", bounds_rise)
            print("init values: ", init_vals_rise)
            print("result: ", res_rise.x)
            rise_tb = timebase[:peak_pos]
            rise_yfit = self.risefit(
                res_rise.x, rise_tb, np.zeros_like(rise_tb), self.risepower, -1
            )
            ax[0].plot(rise_tb, rise_yfit, "r-")
            ax[1].plot(rise_tb, rise_yfit, "r-")
            # mpl.show()

        self.res_rise = res_rise
        # fit decay exponential next:
        bounds_decay = [
            amp_bounds,
            (self.dt_seconds, self.tau2 * 20.0),
        ]  # be sure init values are inside bounds
        init_vals_decay = [0.9 * np.mean(amp_bounds), self.tau2]
        # print('peak, tpast, tdel',  peak_pos , int(time_past_peak/self.dt_seconds) , int(res_rise.x[2]/self.dt_seconds))
        decay_fit_start = peak_pos + int(
            time_past_peak / self.dt_seconds
        )  # + int(res_rise.x[2]/self.dt_seconds)
        # print('decay start: ', decay_fit_start, decay_fit_start*self.dt_seconds, len(event[decay_fit_start:]))

        res_decay = scipy.optimize.minimize(
            self.decayexp,
            init_vals_decay,
            bounds=bounds_decay,
            method="L-BFGS-B",
            #  bounds=bounds_decay, method='L-BFGS-B',
            args=(
                timebase[decay_fit_start:] - decay_fit_start * self.dt_seconds,
                evfit[decay_fit_start:],
                res_rise.x[2],
                1,
            ),
        )  # res_rise.x[2], 1))
        self.res_decay = res_decay

        if debug:
            decay_tb = timebase[decay_fit_start:]
            decay_ev = evfit[decay_fit_start:]
            # f, ax = mpl.subplots(2, 1)
            # ax[0].plot(timebase, evfit)
            ax[1].plot(decay_tb, decay_ev, "g-")
            ax[1].set_title('Decay fit (green)')
            print("\ndecay fit:")
            print("dt: ", self.dt_seconds, " maxev: ", maxev, " peak_pos: ", peak_pos)
            print("bounds: ", bounds_decay)
            print("init values: ", init_vals_decay)
            print("result: ", res_decay.x)
            y = self.decayexp(
                res_decay.x,
                decay_tb,
                np.zeros_like(decay_tb),
                fixed_delay=decay_fit_start * self.dt_seconds,
                mode=-1,
            )
            # print(y)
            # ax[1].plot(decay_tb, y, 'bo', markersize=3)
            ax[1].plot(decay_tb, y, "g-")
        if res_rise.x[2] == 0.0:
            res_rise.x[2] = 2.0*dt
        # now tune by fitting the whole trace, allowing some (but not too much) flexibility
        bounds_full = [
            [a * 10.0 for a in amp_bounds],  # overall amplitude
            (0.2 * res_rise.x[1], 5.0 * res_rise.x[1]),  # rise tau
            (0.2 * res_decay.x[1], 50.0 * res_decay.x[1]),  # decay tau
            (0.3 * res_rise.x[2], 20.0 * res_rise.x[2]),  # delay
            # (0, 1), # amplitude of decay component
        ]
        init_vals = [
            amp_bounds[1],
            res_rise.x[1],
            res_decay.x[1],
            res_rise.x[2],
        ]  # be sure init values are inside bounds
        # if len(label) > 0:
        #     print('Label: ', label)
        #     print('bounds full: ', bounds_full)
        #     print('init_vals: ', init_vals)
        try:
            res = scipy.optimize.minimize(
            self.doubleexp,
            init_vals,
            method="L-BFGS-B",
            # method="Nelder-Mead",
            args=(timebase, evfit, self.risepower, res_rise.x[2], 1),
            bounds=bounds_full,
            options={"maxiter": 100000},
        )
        except:
            print('Fitting failed in event fitter')
            print('evfit: ', evfit)
            return None, None
            # print('timebase: ', timebase)
            # import matplotlib.pyplot as mpl
            # mpl.plot(timebase, evfit)
            # mpl.show()
            # print('risepower: ', self.risepower, res_rise.x[2], bounds_full)
            # raise ValueError()
            
        if debug:
            print("\nFull fit:")
            print("dt: ", self.dt_seconds, " maxev: ", maxev, " peak_pos: ", peak_pos)
            print("bounds: ", bounds_full)
            print("init values: ", init_vals)
            print("result: ", res.x, res_rise.x[2])
            f, ax = mpl.subplots(2, 1)
            ax[0].plot(timebase, evfit, "k-")
            ax[1].plot(timebase, evfit, "k-")
            y = self.doubleexp(
                res.x,
                timebase,
                event,
                risepower=self.risepower,
                fixed_delay=res_rise.x[2],
                mode=-1,
            )
            ax[1].plot(timebase, y, "bo", markersize=3)
            f.suptitle("Full fit")
            mpl.show()

        self.rise_fit = self.risefit(
            res_rise.x, timebase, np.zeros_like(timebase), self.risepower, mode=0
        )
        self.rise_fit[peak_pos:] = 0
        self.rise_fit = self.rise_fit * maxev

        self.decay_fit = self.decayexp(
            self.res_decay.x,
            timebase,
            np.zeros_like(timebase),
            fixed_delay=self.res_rise.x[2],
            mode=0,
        )
        self.decay_fit[:decay_fit_start] = 0  # clip the initial part
        self.decay_fit = self.decay_fit * maxev

        self.bferr = self.doubleexp(
            res.x,
            timebase,
            event,
            risepower=self.risepower,
            fixed_delay=decay_fit_start * self.dt_seconds,
            mode=1,
        )
        # print('fit result: ', res.x, res_rise.x[2])
        res.x[0] = res.x[0] * maxev  # correct for factor
        self.peak_val = maxev
        return res, res_rise.x[2]

    def individual_event_screen(
        self, fit_err_limit: float = 2000.0, tau2_range: float = 2.5
    ) -> None:
        """
        Screen events:
        error of the fit must be less than a limit,
        and
        tau2 must fall within a range of the default tau2
        and
        tau1 must be breater than a minimum tau1
        sets:
        self.events_ok : the list of fitted events that pass
        self.events_notok : the list of fitted events that did not pass
        """
        self.events_ok = []
        for i in self.fitted_events:  # these are the events that were fit
            if self.fiterr[i] <= fit_err_limit:
                if self.ev_tau2[i] <= self.tau2_range * self.tau2:
                    if self.ev_fitamp[i] > self.min_event_amplitude:
                        if self.ev_tau1[i] > self.tau1 / self.tau1_minimum_factor:
                            self.events_ok.append(i)
        self.events_notok = list(set(self.fitted_events).difference(self.events_ok))

    def plot_individual_events(
        self, fit_err_limit: float = 1000.0, tau2_range: float = 2.5, show: bool = True
    ) -> None:
        if not self.individual_events:
            raise
        P = PH.regular_grid(
            3,
            3,
            order="columns",
            figsize=(8.0, 8.0),
            showgrid=False,
            verticalspacing=0.1,
            horizontalspacing=0.12,
            margins={
                "leftmargin": 0.12,
                "rightmargin": 0.12,
                "topmargin": 0.03,
                "bottommargin": 0.1,
            },
            labelposition=(-0.12, 0.95),
        )
        self.P = P
        #        evok, notok = self.individual_event_screen(fit_err_limit=fit_err_limit, tau2_range=tau2_range)
        evok = self.events_ok
        notok = self.events_notok

        P.axdict["A"].plot(self.ev_tau1[evok], self.ev_amp[evok], "ko", markersize=4)
        P.axdict["A"].set_xlabel(r"$tau_1$ (ms)")
        P.axdict["A"].set_ylabel(r"Amp (pA)")
        P.axdict["B"].plot(self.ev_tau2[evok], self.ev_amp[evok], "ko", markersize=4)
        P.axdict["B"].set_xlabel(r"$tau_2$ (ms)")
        P.axdict["B"].set_ylabel(r"Amp (pA)")
        P.axdict["C"].plot(self.ev_tau1[evok], self.ev_tau2[evok], "ko", markersize=4)
        P.axdict["C"].set_xlabel(r"$\tau_1$ (ms)")
        P.axdict["C"].set_ylabel(r"$\tau_2$ (ms)")
        P.axdict["D"].plot(self.ev_amp[evok], self.fiterr[evok], "ko", markersize=3)
        P.axdict["D"].plot(self.ev_amp[notok], self.fiterr[notok], "ro", markersize=3)
        P.axdict["D"].set_xlabel(r"Amp (pA)")
        P.axdict["D"].set_ylabel(r"Fit Error (cost)")
        for i in notok:
            ev_bl = np.mean(self.Summary.allevents[i, 0:5])
            P.axdict["E"].plot(
                self.avgeventtb, self.Summary.allevents[i] - ev_bl, "b-", linewidth=0.75
            )
            # P.axdict['E'].plot()
            P.axdict["F"].plot(
                self.avgeventtb, self.Summary.allevents[i] - ev_bl, "r-", linewidth=0.75
            )
        P2 = PH.regular_grid(
            1,
            1,
            order="columns",
            figsize=(8.0, 8.0),
            showgrid=False,
            verticalspacing=0.1,
            horizontalspacing=0.12,
            margins={
                "leftmargin": 0.12,
                "rightmargin": 0.12,
                "topmargin": 0.03,
                "bottommargin": 0.1,
            },
            labelposition=(-0.12, 0.95),
        )
        P3 = PH.regular_grid(
            1,
            5,
            order="columns",
            figsize=(12, 8.0),
            showgrid=False,
            verticalspacing=0.1,
            horizontalspacing=0.12,
            margins={
                "leftmargin": 0.12,
                "rightmargin": 0.12,
                "topmargin": 0.03,
                "bottommargin": 0.1,
            },
            labelposition=(-0.12, 0.95),
        )
        idx = [a for a in P3.axdict.keys()]
        ncol = 5
        offset2 = 0.0
        k = 0
        for i in evok:
            #  print(self.ev_tau1, self.ev_tau2)
            offset = i * 3.0
            ev_bl = np.mean(self.Summary.allevents[i, 0:5])
            P2.axdict["A"].plot(
                self.avgeventtb,
                self.Summary.allevents[i] + offset - ev_bl,
                "k-",
                linewidth=0.35,
            )
            # p = [self.ev_amp[i], self.ev_tau1[i],self.ev_tau2[i]]
            # x = self.avgeventtb
            # y = self.doubleexp(p, x, np.zeros_like(x), self.risepower, mode=-1)
            # y = p[0] * (((np.exp(-x/p[1]))) - np.exp(-x/p[2]))
            P2.axdict["A"].plot(
                self.avgeventtb,
                self.sign * self.best_fit[i] + offset,
                "c--",
                linewidth=0.3,
            )
            P2.axdict["A"].plot(
                self.avgeventtb,
                self.sign * self.best_decay_fit[i] + offset,
                "r--",
                linewidth=0.3,
            )
            P3.axdict[idx[k]].plot(
                self.avgeventtb, self.Summary.allevents[i] + offset2, "k--", linewidth=0.3
            )
            P3.axdict[idx[k]].plot(
                self.avgeventtb,
                self.sign * self.best_fit[i] + offset2,
                "r--",
                linewidth=0.3,
            )
            if k == 4:
                k = 0
                offset2 += 10.0
            else:
                k += 1

        if show:
            mpl.show()

    def plots(self, 
        data, events: Union[np.ndarray, None] = None, title: Union[str, None] = None,
        testmode:bool=False, index:int=0,
    ) -> object:
        """
        Plot the results from the analysis and the fitting
        """

        import matplotlib.pyplot as mpl
        import pylibrary.plotting.plothelpers as PH


        P = PH.regular_grid(
            3,
            1,
            order="columnsfirst",
            figsize=(8.0, 6),
            showgrid=False,
            verticalspacing=0.08,
            horizontalspacing=0.08,
            margins={
                "leftmargin": 0.07,
                "rightmargin": 0.20,
                "topmargin": 0.03,
                "bottommargin": 0.1,
            },
            labelposition=(-0.12, 0.95),
        )
        self.P = P

        ax = P.axarr
        ax = ax.ravel()
        PH.nice_plot(ax)
        for i in range(1, 2):
            ax[i].get_shared_x_axes().join(ax[i], ax[0])
        # raw traces, marked with onsets and peaks
        ax[0].set_ylabel("I (pA)")
        ax[0].set_xlabel("T (s)")
        ax[0].legend(fontsize=8, loc=2, bbox_to_anchor=(1.0, 1.0))
        ax[1].set_ylabel("Deconvolution")
        ax[1].set_xlabel("T (s)")
        ax[1].legend(fontsize=8, loc=2, bbox_to_anchor=(1.0, 1.0))
        ax[2].set_ylabel("Averaged I (pA)")
        ax[2].set_xlabel("T (s)")
        ax[2].legend(fontsize=8, loc=2, bbox_to_anchor=(1.0, 1.0))
        if title is not None:
            P.figure_handle.suptitle(title)
        print(self.P.axarr)
        for i, d in enumerate(data):
            self.plot_trial(self.P.axarr.ravel(), i, d, events, index=index)


        if testmode:  # just display briefly
            mpl.show(block=False)
            mpl.pause(2)
            mpl.close()
            return None
        else:    
            return self.P.figure_handle
 
                  

    def plot_trial(self, ax, i, data, events, markersonly:bool=False, index:int=0):
        onset_marks = {0: "k^", 1:"b^", 2:"m^", 3:"c^"}
        peak_marks = {0: "+", 1:"g+", 2:"y+", 3:"k+"}
        scf = 1e12
        tb = self.timebase[: data.shape[0]]
        label = 'Data'
        ax[0].plot(tb, scf * data, "k-", linewidth=0.75, label=label)  # original data
        label = 'Onsets'
        ax[0].plot(
            tb[self.onsets[i]],
            scf * data[self.onsets[i]],
            onset_marks[index],
            markersize=6,
            markerfacecolor=(1, 1, 0, 0.8),
            label=label,
        )
        if len(self.onsets[i]) is not None:
            #            ax[0].plot(tb[events],  data[events],  'go',  markersize=5, label='Events')
            #        ax[0].plot(tb[self.peaks],  self.data[self.peaks],  'r^', label=)
            if i == 0:
                label = 'Smoothed Peaks'
            else:
                label = ''
            ax[0].plot(
                tb[self.Summary.smpkindex[i]],
                scf * np.array(self.Summary.smoothed_peaks[i]),
                peak_marks[index],
                label=label,
            )
        if markersonly:
            return
            
        # deconvolution trace, peaks marked (using onsets), plus threshold)
        if i == 0:
            label = "Deconvolution"
        else:
            label = ''
        ax[1].plot(tb[: self.Criterion[i].shape[0]], self.Criterion[i], label=label)
        
        if i == 0:
            label="Threshold ({0:4.2f}) SD".format(self.sdthr)
        else:
            label = ''    
        ax[1].plot(
            [tb[0], tb[-1]],
            [self.sdthr, self.sdthr],
            "r--",
            linewidth=0.75,
            label=label,
        )
        if i == 0:
            label = "Deconv. Peaks"
        else:
            label = ''
        ax[1].plot(
            tb[self.onsets[i]] - self.idelay,
            self.Criterion[i][self.onsets[i]],
            onset_marks[index],
            label=label,
        )
        if events is not None:  # original events
            ax[1].plot(
                tb[: self.Criterion[i].shape[0]][events],
                self.Criterion[i][events],
                "ro",
                markersize=5.0,
            )
 
        # averaged events, convolution template, and fit
        if self.Summary.average.averaged:
            if i == 0:
                nev = sum([len(x) for x in self.onsets])
                label = f'Average Event (N={nev:d})'
            else:
                label=''
            evlen = len(self.Summary.average.avgevent)
            ax[2].plot(
                self.Summary.average.avgeventtb[: evlen],
                scf * self.Summary.average.avgevent,
                "k",
                label=label,
            )
            maxa = np.max(self.sign * self.Summary.average.avgevent)
            # tpkmax = np.argmax(self.sign*self.template)
            if self.template is not None:
                maxl = int(np.min([len(self.template), 
                    len(self.Summary.average.avgeventtb)]))
                temp_tb = np.arange(0, maxl * self.dt_seconds, self.dt_seconds)
                # print(len(self.avgeventtb[:len(self.template)]), len(self.template))
                if i == 0:
                    label = "Template"
                else:
                    label = ''
                ax[2].plot(
                    self.Summary.average.avgeventtb[:maxl],
                    scf * self.sign * self.template[:maxl] * maxa / self.template_amax,
                    "r-",
                    label=label,
                )
           
           
            # compute double exp based on rise and decay alone
            # print('res rise: ', self.res_rise)
            # p = [self.res_rise.x[0], self.res_rise.x[1], self.res_decay.x[1], self.res_rise.x[2]]
            # x = self.avgeventtb[:len(self.avg_best_fit)]
            # y = self.doubleexp(p, x, np.zeros_like(x), risepower=4, fixed_delay=0, mode=0)
            # ax[2].plot(x, y, 'b--', linewidth=1.5)
            sf = 1.0
            tau1 = np.power(
                10, (1.0 / self.risepower) * np.log10(self.tau1 * sf)
            )  # correct for rise power
            tau2 = self.tau2 * sf
            if i == 0:
               label = "Best Fit:\nRise Power={0:.2f}\nTau1={1:.3f} ms\nTau2={2:.3f} ms\ndelay: {3:.3f} ms".format(
                                    self.risepower,
                                    self.res_rise.x[1] * sf,
                                    self.res_decay.x[1] * sf,
                                    self.bfdelay * sf,
                                )
            else:
                label = ''
            ax[2].plot(
                self.Summary.average.avgeventtb[: len(self.avg_best_fit)],
                scf * self.avg_best_fit,
                "c--",
                linewidth=2.0,
                label=label,
            )
            # ax[2].plot(self.avgeventtb[:len(self.decay_fit)],  self.sign*scf*self.rise_fit,  'g--', linewidth=1.0,
            #     label='Rise tau  {0:.2f} ms'.format(self.res_rise.x[1]*1e3))
            # ax[2].plot(self.avgeventtb[:len(self.decay_fit)],  self.sign*scf*self.decay_fit,  'm--', linewidth=1.0,
            #     label='Decay tau {0:.2f} ms'.format(self.res_decay.x[1]*1e3))


        # if self.fitted:
        #     print('measures: ', self.risetenninety, self.decaythirtyseven)
