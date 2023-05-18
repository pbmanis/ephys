"""
Classes that provide support functions for minis_methods, 
including fitting, smoothing, filtering, and some analysis.

Test run timing:
cb: 0.175 s (with cython version of algorithm); misses overlapping events
aj: 0.028 s, plus gets overlapping events

July 2017

Note: all values are MKS (Seconds, plus Volts, Amps)
per acq4 standards... 

2022: Cleaned up again; changed to lmfit rather than
multiple passes with scipy.optimize with various algorithms.

"""

from dataclasses import dataclass, field
from typing import List, Tuple, Union

import lmfit
import matplotlib.pyplot as mpl
import numpy as np
import pylibrary.plotting.plothelpers as PH
import pylibrary.tools.cprint as CP
import scipy.signal as SPS
import scipy.special
from scipy.optimize import curve_fit

import ephys.tools.digital_filters as dfilt


@dataclass
class Filtering:
    LPF_applied: bool = False
    HPF_applied: bool = False
    LPF_frequency: Union[float, None] = None
    HPF_frequency: Union[float, None] = None


def def_empty_list():
    return [0]  # [0.0003, 0.001]  # in seconds (was 0.001, 0.010)


def def_empty_list2():
    return [[None]]  # [0.0003, 0.001]  # in seconds (was 0.001, 0.010)


@dataclass
class AverageEvent:
    """
    The AverageEvent class holds the averaged events
    from all traces and trials
    """

    averaged: bool = False  # set flags in case of no events found
    avgeventtb: Union[List, np.ndarray] = field(  # time base for the event
        default_factory=def_empty_list
    )
    avgevent: Union[List, np.ndarray] = field(  # the event
        default_factory=def_empty_list
    )
    avgevent25: Union[List, np.ndarray] = field(  # the event
        default_factory=def_empty_list
    )
    avgevent75: Union[List, np.ndarray] = field(  # the event
        default_factory=def_empty_list
    )
    Nevents: int = 0  # number of events that were avetaged
    avgnpts: int = 0  # number of points in the array
    fitted: bool = False  # Set True if data has been fitted
    fitted_tau1: float = np.nan  # rising time constant for 2-exp fit
    fitted_tau2: float = np.nan  # falling time constant for 2-exp fit
    best_fit: object = None  # best fit trace
    Amplitude: float = np.nan  # amplitude from the fit
    avg_fiterr: float = np.nan  # fit error
    risetenninety: float = np.nan  # rise time (seconds), 10-90 %
    decaythirtyseven: float = np.nan  # fall time to 37% of peak


@dataclass
class Summaries:
    """
    The Summaries dataclass holdes the results of the
    individual events that were detected,
    as well as the results of various fits
    and the averge fit
    """

    onsets: Union[List, np.ndarray] = field(  # onset times for detected events
        default_factory=def_empty_list2
    )
    peaks: Union[
        List, np.ndarray
    ] = field(  # peak times (not denoised) for detected events
        default_factory=def_empty_list
    )
    smpkindex: Union[List, np.ndarray] = field(  # peak indices for smoothed peaks
        default_factory=def_empty_list
    )
    smoothed_peaks: Union[List, np.ndarray] = field(  # smoothed peaks
        default_factory=def_empty_list
    )
    amplitudes: Union[List, np.ndarray] = field(  # event amplitudes
        default_factory=def_empty_list
    )
    Qtotal: Union[List, np.ndarray] = field(  # charge for each event
        default_factory=def_empty_list
    )
    individual_events: bool = False  # hmmm.
    average: object = AverageEvent()  # average
    average25: object = AverageEvent()  # average of lower 25th percentile
    average75: object = AverageEvent()  # average of upper 25th percentile
    allevents: Union[List, np.ndarray] = field(  # list of all events
        default_factory=def_empty_list
    )
    event_trace_list: Union[List, None] = field(  # list linking events to parent trace
        default_factory=def_empty_list
    )


class MiniAnalyses:
    def __init__(self):
        """
        Base class for Clements-Bekkers and Andrade-Jonas methods
        Provides template generation, and summary analyses
        Allows use of common methods between different algorithms
        """
        self.verbose = False  # flag to control extra printing for debugging
        self.datasource = ""  # name of day/slice/cell/protocol being analyzed
        self.ntraces = 1  # nubmer of traces
        self.filtering = Filtering()  # filtering class
        self.risepower = 4.0  # power for sigmoidal rise when fitting events
        self.min_event_amplitude = 5.0e-12  # pA default for minimum event size
        self.eventstartthr = None
        self.Criterion = [None]  # array for C-B
        self.template = None  # template for C-B
        self.template_tmax = 0.0
        self.analysis_window = [None, None]  # specify window or entire data set
        self.datatype = None
        self.events_ok = []
        self.events_notok = []
        super().__init__()

    def setup(
        self,
        datasource: str = "",
        ntraces: int = 1,
        tau1: Union[float, None] = None,
        tau2: Union[float, None] = None,
        template_tmax: float = 0.05,
        template_pre_time: float = 0.0,
        dt_seconds: Union[float, None] = None,
        delay: float = 0.0,  # into start of each trace for analysis
        sign: int = 1,
        eventstartthr: Union[float, None] = None,
        risepower: Union[float, None] = None,
        min_event_amplitude: float = 5.0e-12,
        threshold: float = 2.5,
        global_SD: Union[float, None] = None,
        analysis_window: List[Union[float, None]] = [None, None],
        lpf: Union[float, None] = None,
        hpf: Union[float, None] = None,
        notch: Union[float, None] = None,
        notch_Q: float = 30.0,
    ) -> None:
        """
        Just store the parameters - will compute when needed
        Use of globalSD and threshold:
        if global SD is None, we use the threshold as it.

        If global SD has a value, then we use that rather than the
        current trace SD for threshold determinations
        """
        CP.cprint("c", "MiniAnalysis SETUP")
        assert sign in [-1, 1]  # must be selective, positive or negative events only
        self.datasource = datasource
        self.ntraces = ntraces
        self.Criterion = [[None] for x in range(ntraces)]
        self.sign = sign
        self.taus = [tau1, tau2]
        self.dt_seconds = dt_seconds
        self.template_tmax = template_tmax
        self.idelay = int(delay / self.dt_seconds)
        self.template_pre_time = template_pre_time
        self.template = None  # reset the template if needed.
        if eventstartthr is not None:
            self.eventstartthr = eventstartthr
        if risepower is not None:
            self.risepower = risepower
        self.min_event_amplitude = min_event_amplitude
        self.threshold = threshold
        self.sdthr = self.threshold  # for starters
        self.analysis_window = analysis_window
        self.lpf = lpf
        self.hpf = hpf
        self.notch = notch
        self.notch_Q = notch_Q
        self.reset_filtering()
        self.set_datatype("VC")

    def set_datatype(self, datatype: str):
        CP.cprint("c", f"data type: {datatype:s}")
        self.datatype = datatype

    def set_sign(self, sign: int = 1):
        self.sign = sign

    def set_dt_seconds(self, dt_seconds: Union[None, float] = None):
        self.dt_seconds = dt_seconds

    def set_risepower(self, risepower: float = 4):
        if risepower > 0 and risepower <= 8:
            self.risepower = risepower
        else:
            raise ValueError("Risepower must be 0 < n <= 8")

    # def set_notch(self, notches):
    #     if isinstance(nothce, float):
    #         notches = [notches]
    #     elif isinstance(notches, None):
    #         self.notch = None
    #         self.Notch_applied = False
    #         return
    #     elif isinstance(notches, list):
    #         self.notch = notches
    #     else:
    #         raise ValueError("set_notch: Notch must be list, float or None")

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
        if self.template_pre_time > 0:
            t_delay = int(self.template_pre_time / self.dt_seconds)
            self.template[t_delay:] = tm[:-t_delay]  # shift the template
            self.template[0:t_delay] = np.zeros(t_delay)
        else:
            self.template = tm
        if self.sign > 0:
            self.template_amax = np.max(self.template)
        else:
            self.template = -self.template
            self.template_amax = np.min(self.template)

    def reset_filtering(self):
        """
        Reset the filtering flags so we know which have been done.
        The purpose of this is to keep from applying filters repeatedly
        """
        self.filtering.LPF_applied = False
        self.filtering.HPF_applied = False
        self.filtering.Notch_applied = False

    def LPFData(
        self, data: np.ndarray, lpf: Union[float, None] = None, NPole: int = 8
    ) -> np.ndarray:
        """
        Parameters
        ----------
        data : the  array of data
        lpf: float : low pass filter frequency in Hz
        NPole : number of poles for designing the filter

        Return
        ------
        data : the filtered data (as a copy)
        """
        assert not self.filtering.LPF_applied  # block repeated application of filtering
        CP.cprint("y", f"minis_methods_common, LPF data:  {lpf:f}")
        if lpf is not None:
            # CP.cprint('y', f"     ... lpf at {lpf:f}")
            if lpf > 0.49 / self.dt_seconds:
                raise ValueError(
                    "lpf > Nyquist: ",
                    lpf,
                    0.49 / self.dt_seconds,
                    self.dt_seconds,
                    1.0 / self.dt_seconds,
                )
            # data = dfilt.SignalFilter_LPFButter(data, lpf, 1./self.dt_seconds, NPole=8)
            data = dfilt.SignalFilter_LPFBessel(
                data,
                LPF=lpf,
                samplefreq=1.0 / self.dt_seconds,
                NPole=4,
                filtertype="low",
            )
            self.filtering.LPF = lpf
            self.filtering.LPF_applied = True
        return data.copy()

    def HPFData(
        self, data: np.ndarray, hpf: Union[float, None] = None, NPole: int = 8
    ) -> np.ndarray:
        """
        Parameters
        ----------
        data : the  array of data
        hpf: float : high pass filter frequency in Hz
        NPole : number of poles for designing the filter

        Return
        ------
        data : the filtered data (as a copy)
        """
        assert not self.filtering.HPF_applied  # block repeated application of filtering
        if hpf is None or hpf == 0.0:
            return data
        if len(data.shape) == 1:
            ndata = data.shape[0]
        else:
            ndata = data.shape[1]
        nyqf = 0.5 * ndata * self.dt_seconds
        CP.cprint("y", f"minis_methods: hpf at {hpf:f}")
        if hpf < 1.0 / nyqf:  # duration of a trace
            CP.cprint("r", "unable to apply HPF, trace too short")
            return data
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
                1.0 / self.dt_seconds,
            )
        data = dfilt.SignalFilter_HPFButter(
            data - data[0], hpf, 1.0 / self.dt_seconds, NPole=4
        )
        self.filtering.HPF = hpf
        self.filtering.HPF_applied = True
        return data.copy()

    def NotchData(
        self, data: np.ndarray, notch: Union[list, None] = None, notch_Q=30.0
    ) -> np.ndarray:
        """
        Notch filter the data
        This routine can apply multiple notch filters to the data at once.

        Parameters
        ----------
        data : the  array of data
        notch: list : list of notch frequencies, in Hz
        notch_Q : the Q of the filter (higher is narrower)

        Return
        ------
        data : the filtered data (as a copy)
        """

        if notch is None:
            return data
        if len(data.shape) == 1:
            ndata = data.shape[0]
        else:
            ndata = data.shape[1]
        data = dfilt.NotchFilterZP(
            data,
            notchf=notch,
            Q=notch_Q,
            QScale=False,
            samplefreq=1.0 / self.dt_seconds,
        )
        self.filtering.notch = notch
        self.filtering.Notch_applied = True
        return data.copy()

    def prepare_data(self, data: np.array):
        """
        This function prepares the incoming data for the mini analyses.
        1. Clip the data in time (remove sections with current or voltage steps)
        2. Filter the data (LPF, HPF)

        Parameters
        ----------
        data : np array

        Returns
        -------
        nothing. The result is held in the class variable "data", along with a
        corresponding timebase.
        """
        self.timebase = np.arange(0.0, data.shape[0] * self.dt_seconds, self.dt_seconds)
        # print(
        #     "preparedata timebase shape, data shape: ", self.timebase.shape, data.shape
        # )
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
                CP.cprint(
                    "y", f"minis_methods_common, prepare_data: LPF: {self.lpf:.1f} Hz"
                )
            else:
                CP.cprint("r", f"**** minis_methods_common, no LPF applied")
            if self.hpf is not None:
                CP.cprint(
                    "y",
                    f"    minis_methods_common, prepare_data: HPF: {self.hpf:.1f} Hz",
                )
            else:
                CP.cprint("r", f"    minis_methods_common, no HPF applied")
        if self.lpf is not None and isinstance(self.lpf, float):
            data = self.LPFData(data, lpf=self.lpf)
        if self.hpf is not None and isinstance(self.hpf, float):
            data = self.HPFData(data, hpf=self.hpf)
        if self.notch is not None and isinstance(self.notch, list):
            data = self.NotchData(data, notch=self.notch, notch_Q=self.notch_Q)
        self.data = data
        self.timebase = self.timebase[jmin:jmax]
        # self.template_tmax = np.max(self.timebase)

    def moving_average(self, a, n: int = 3) -> Tuple[np.array, int]:
        ret = np.cumsum(a, dtype=float)
        ret[n:] = ret[n:] - ret[:-n]
        return ret[int(n / 2) :] / n, n  # re-align array

    def remove_outliers(self, x: np.ndarray, scale: float = 3.0) -> np.ndarray:
        a = np.array(x)
        upper_quartile = np.percentile(a, 75)
        lower_quartile = np.percentile(a, 25)
        IQR = (upper_quartile - lower_quartile) * scale
        quartileSet = (lower_quartile - IQR, upper_quartile + IQR)
        result = np.where(((a >= quartileSet[0]) & (a <= quartileSet[1])), a, np.nan)
        return result

    def summarize(self, data, order: int = 11, verbose: bool = False) -> None:
        """
        compute intervals,  peaks and ampitudes for all found events in a
        trace or a group of traces
        filter out events that are less than min_event_amplitude
        and events where the charge is of the wrong sign.
        """
        CP.cprint("c", "    Summarizing data")
        i_decay_pts = int(
            2.0 * self.taus[1] / self.dt_seconds
        )  # decay window time (points) Units all seconds
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
        avgwin = 5  # 5 point moving average window for peak detection
        mwin = int((0.50) / self.dt_seconds)
        if self.sign > 0:
            nparg = np.greater
        else:
            nparg = np.less
        self.intervals = []
        self.timebase = np.arange(0.0, data.shape[1] * self.dt_seconds, self.dt_seconds)

        nrejected_too_small = 0
        for itrial, dataset in enumerate(data):  # each trial/trace
            if len(self.onsets[itrial]) == 0:  # original events
                continue
            acceptlist_trial = []
            ons = np.where(self.onsets[itrial] < len(self.timebase))
            self.intervals.append(np.diff(ons))  # event intervals
            ev_accept = []
            for j, onset in enumerate(
                self.onsets[itrial]
            ):  # for all of the events in this trace
                if self.sign > 0 and self.eventstartthr is not None:
                    if dataset[onset] < self.eventstartthr:
                        continue
                if self.sign < 0 and self.eventstartthr is not None:
                    if dataset[onset] > -self.eventstartthr:
                        continue
                event_data = dataset[onset : (onset + mwin)]  # get this event
                svwinlen = event_data.shape[0]
                if svwinlen > 11:  # savitz-golay window length
                    svn = 11
                else:
                    svn = svwinlen
                if (
                    svn % 2 == 0
                ):  # if even, decrease by 1 point to meet ood requirement for savgol_filter
                    svn -= 1
                if svn > 3:  # go ahead and filter
                    p = scipy.signal.argrelextrema(
                        scipy.signal.savgol_filter(event_data, svn, 2),
                        nparg,
                        order=order,
                    )[0]
                else:  # skip filtering
                    p = scipy.signal.argrelextrema(
                        event_data,
                        nparg,
                        order=order,
                    )[0]
                if len(p) > 0:
                    i_end = i_decay_pts + onset  # distance from peak to end
                    i_end = min(dataset.shape[0], i_end)  # keep within the array limits
                    if j < len(self.onsets[itrial]) - 1:
                        if i_end > self.onsets[itrial][j + 1]:
                            i_end = (
                                self.onsets[itrial][j + 1] - 1
                            )  # only go to next event start
                    windowed_data = dataset[onset:i_end]
                    move_avg, n = self.moving_average(
                        windowed_data,
                        n=min(avgwin, len(windowed_data)),
                    )
                    if self.sign > 0:
                        smpk = np.argmax(move_avg)  # find peak of smoothed data
                        rawpk = np.argmax(windowed_data)  # non-smoothed
                    else:
                        smpk = np.argmin(move_avg)
                        rawpk = np.argmin(windowed_data)
                    # if self.sign*(move_avg[smpk] - windowed_data[0]) >= self.min_event_amplitude:
                    if nparg(
                        move_avg[smpk] - windowed_data[0], self.min_event_amplitude
                    ):
                        ev_accept.append(j)
                        self.Summary.onsets[itrial].append(onset)
                        self.Summary.peaks[itrial].append(onset + rawpk)
                        self.Summary.amplitudes[itrial].append(windowed_data[rawpk])
                        self.Summary.smpkindex[itrial].append(onset + smpk)
                        self.Summary.smoothed_peaks[itrial].append(move_avg[smpk])
                        acceptlist_trial.append(j)
                    else:
                        nrejected_too_small += 1

                        # CP.cprint(
                        #     "y",
                        #     f"     Event too small: {1e12*(move_avg[smpk]-windowed_data[0]):6.1f} vs. hardthreshold: {1e12*self.min_event_amplitude:6.1f} pA",
                        # )
                        continue  # filter out events smaller than the amplitude
            self.onsets[itrial] = self.onsets[itrial][
                ev_accept
            ]  # reduce to the accepted values
        CP.cprint(
            "y",
            f"    {nrejected_too_small:5d} events were smaller than threshold of {1e12*self.min_event_amplitude:6.1f} pA",
        )

        self.average_events(
            traces=range(len(data)), eventlist=self.Summary.onsets, data=data
        )

    def re_average_events(self):
        self.average_events(
            traces=self.traces_for_average,
            eventlist=self.Summary.onsets,
            data=self.data_for_average,
        )

    def measure_events(self, data: object, eventlist: list) -> dict:
        """
        compute simple measurements of events (area, amplitude, half-width)
        """
        assert data.ndim == 1
        self.measured = False
        # treat like averaging
        tdur = np.max((np.max(self.taus) * 5.0, 0.010))  # go 5 taus or 10 ms past event
        npost = int(tdur / self.dt_seconds)
        self.avgeventdur = tdur
        tpre = self.template_pre_time  # self.taus[0]*10.
        npre = int(tpre / self.dt_seconds)  # points for the pre time
        self.avgnpts = int((tpre + tdur) / self.dt_seconds)  # points for the average
        allevents = np.zeros((len(eventlist), self.avgnpts))
        k = 0
        pkt = 0  # np.argmax(self.template)  # accumulate
        meas = {"Q": [], "A": [], "HWup": [], "HWdown": [], "HW": []}
        for j, i in enumerate(eventlist):
            ix = i
            if (
                (ix + npost) < len(self.data)
                and (ix - npre) >= 0
                and np.count_nonzero(np.isnan(self.data) == npre + npost)
            ):
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
                hw_up = self.dt_seconds * np.argmin(
                    np.fabs((ev_j[ai] / 2.0) - ev_j[:ai])
                )
                hw_down = self.dt_seconds * np.argmin(
                    np.fabs(ev_j[ai:] - (ev_j[ai] / 2.0))
                )
                meas["HWup"].append(hw_up)
                meas["HWdown"].append(hw_down)
                meas["HW"].append(hw_up + hw_down)
            self.measured = True
            self.Summary.allevents = allevents
            # assert 1 == 0  # force trap here
        else:
            self.measured = False
            self.Summary.allevents = None
        return meas

    def average_events(
        self,
        traces: list,
        eventlist: Union[list, None] = None,
        data: Union[list, object, None] = None,
    ) -> tuple:
        """
        compute average event with length of template
        For the average, exclude events in which a second event
        occurs within the duration of the template.

        Parameters
        ----------
        traces:
            list of traces to go thorugh when computing average.
                may be a single trace or a group
        eventlist : list
            List of event onset indices into the arrays
            Expect a 2-d list (traces x onsets)
        data : expect 2d list matching the eventlist.

        """
        if eventlist is None and data is None:
            raise ValueError(
                "minis_methods_common.average_events requires an eventlist and the original data"
            )
        if not self.Summary.onsets:
            raise ValueError("No onstes identified")
        self.data_for_average = data
        self.traces_for_average = traces
        self.Summary.average.averaged = False
        tdur = np.max((np.max(self.taus) * 5.0, 0.010))  # go 5 taus or 10 ms past event
        tpre = self.template_pre_time
        npre = int(tpre / self.dt_seconds)  # points for the pre time
        npost = int(tdur / self.dt_seconds)
        avgnpts = npre + npost  # points for the average
        avgeventtb = np.arange(avgnpts) * self.dt_seconds
        n_events = sum([len(events) for events in self.Summary.onsets])
        allevents = np.zeros((n_events, avgnpts))
        event_trace = [[]] * n_events
        k = 0
        pkt = 0
        n_incomplete_events = 0
        n_overlaps = 0
        n_charge_sign = 0
        onsetlist = self.Summary.onsets

        # select traces for averaging
        for itrace in traces:
            for j, event_onset in enumerate(onsetlist[itrace]):
                ix = event_onset + pkt  # self.idelay
                accept = (
                    True  # assume trace is ok, then set rejection based on criteria
                )
                if ((ix + npost) >= data[itrace].shape[0]) or ((ix - npre) < 0):
                    accept = False  # outside bounds
                    n_incomplete_events += 1
                    # CP.cprint("y", f"        trace: {itrace:d}, event: {j:d} event would be outside data window")
                if (
                    accept and j < len(onsetlist[itrace]) - 1
                ):  # check for next event and reject if in the window
                    if (onsetlist[itrace][j + 1] - event_onset) < avgnpts:
                        accept = False
                        n_overlaps += 1
                        # CP.cprint("y", f"        trace: {itrace:d}, event: {j:d} has overlaps")

                if (
                    accept
                    and np.sum(
                        data[itrace][ix : (ix + npost)]
                        - np.mean(data[itrace][(ix - npre) : ix])
                    )
                    > 0.0
                ):  # check sign of charge
                    accept = False
                    n_charge_sign += 1
                    # CP.cprint("y", f"        trace: {itrace:d}, event: {j:d} has wrong charge")

                if accept:  # ok to add
                    allevents[k, :] = data[
                        itrace, (ix - npre) : (ix + npost)
                    ]  # save event, reject later
                    allevents[k, :] -= np.mean(allevents[k, 0:npre])
                    # padding sometimes needed to fit output array shape
                    # pad = allevents.shape[1] - ((ix - npre) - (ix + npost))
                    event_trace[k] = [
                        itrace,
                        j,
                    ]  # only add to the event list if data in window
                    k = k + 1
                else:  # reject by setting data to nans
                    allevents[k, :] = np.nan * allevents[k, :]
                    k = k + 1

        if n_incomplete_events > 0:
            CP.cprint(
                "y",
                f"    {n_incomplete_events:d} event(s) excluded from average because they were too close to the end of the trace\n",
            )
        if n_overlaps > 0:
            CP.cprint(
                "y",
                f"    {n_overlaps:d} event(s) excluded from average because they overlapped in the window with another event\n",
            )
        if n_charge_sign > 0:
            CP.cprint(
                "y",
                f"    {n_charge_sign:d} event(s) excluded the sign of the charge was wrong\n",
            )
        self.Summary.average.avgeventtb = avgeventtb
        self.Summary.average.avgnpts = avgnpts
        self.Summary.average.avgevent = np.nanmean(
            allevents, axis=0
        )  # - np.mean(avgevent[:3])
        self.Summary.average.stdevent = np.nanstd(allevents, axis=0)
        print("allevents shape: ", allevents.shape)
        if self.sign < 0:
            evamps = self.sign * np.nanmin(allevents, axis=1)
            print(1e12 * np.nanmin(np.nanmin(allevents, axis=1)))
        else:
            evamps = self.sign * np.nanmax(allevents, axis=1)
        print("evamps: ", 1e12 * evamps[:100])
        ev25 = np.nanpercentile(
            evamps,
            q=25,
        )  #  method="median_unbiased")
        ev75 = np.nanpercentile(
            evamps,
            q=75,
        )  #  method="median_unbiased")
        print(
            "len evamps, allevents, timebase: ",
            len(evamps),
            len(allevents),
            len(self.Summary.average.avgeventtb),
        )
        print("ev25: ", ev25 * 1e12)
        print("ev75: ", ev75 * 1e12)
        print(
            "nanmean events25 shape: ",
            np.nanmean(allevents[evamps < ev25], axis=1).shape,
        )
        print(
            "nanmean events75 shape: ",
            np.nanmean(allevents[evamps > ev75], axis=1).shape,
        )
        avgevent25 = np.nanmean(allevents[evamps < ev25], axis=0)
        avgevent75 = np.nanmean(allevents[evamps > ev75], axis=0)
        print(np.min(avgevent25), np.max(avgevent25), len(avgevent25))
        print(np.min(avgevent75), np.max(avgevent75), len(avgevent75))

        if k > 0:
            self.Summary.average.averaged = True
            self.Summary.average.Nevents = k
            self.Summary.average.avgevent25 = avgevent25
            self.Summary.average.avgevent75 = avgevent75
            self.Summary.allevents = allevents
            self.Summary.clean_event_trace_list = event_trace
        else:
            self.Summary.average.averaged = False
            self.Summary.average.avgevent = []
            self.Summary.average.avgevent25 = []
            self.Summary.average.avgevent75 = []
            self.Summary.average.allevents = []
            self.Summary.clean_event_trace_list = []

        if self.Summary.average.averaged:
            CP.cprint("m", "    Fitting averaged event")
            self.fit_average_event(
                tb=self.Summary.average.avgeventtb,
                avgevent=self.Summary.average.avgevent,
                initdelay=self.template_pre_time,
                debug=False,
            )
            self.Summary.average.fitted_tau1 = self.fitted_tau1
            self.Summary.average.fitted_tau2 = self.fitted_tau2
            self.Summary.average.best_fit = self.avg_best_fit
            self.Summary.average.Amplitude = self.Amplitude
            self.Summary.average.avg_fiterr = self.avg_fiterr
            self.Summary.average.risetenninety = self.risetenninety
            self.Summary.average.decaythirtyseven = self.decaythirtyseven
        else:
            CP.cprint("r", "**** No events found")
        return

    def average_events_subset(self, eventlist: list, data: np.ndarray) -> tuple:
        """
        compute average event with length of template
        Parameters
        ----------
        eventlist : list
            1-d numpy array of the data
            List of event onset indices into the arrays
            Expect a 1-d list (traces x onsets)
        data : numpy array of the data
        """
        assert data.ndim == 1
        tdur = np.max((np.max(self.taus) * 5.0, 0.010))  # go 5 taus or 10 ms past event
        npost = int(tdur / self.dt_seconds)
        tpre = self.template_pre_time  # self.taus[0]*10.
        npre = int(tpre / self.dt_seconds)  # points for the pre time
        avgnpts = int((tpre + tdur) / self.dt_seconds)  # points for the average
        avgeventtb = np.arange(avgnpts) * self.dt_seconds
        n_events = sum([len(events) for events in self.Summary.onsets])
        allevents = np.zeros((n_events, avgnpts))
        k = 0
        pkt = 0
        for itrace, event_onset in enumerate(eventlist):
            # CP.cprint('c', f"Trace: {itrace: d}, # onsets: {len(onsets):d}")
            ix = event_onset + pkt  # self.idelay
            # print('itrace, ix, npre, npost: ', itrace, ix, npre, npost)
            if (ix + npost) < data.shape[0] and (ix - npre) >= 0:
                allevents[k, :] = data[(ix - npre) : (ix + npost)]
                k = k + 1
        return np.mean(allevents, axis=0), avgeventtb, allevents

    def fit_average_event(
        self,
        tb,
        avgevent,
        debug: bool = False,
        label: str = "",
        inittaus: List = [0.001, 0.005],
        initdelay: Union[float, None] = 0.0,
    ) -> None:
        """
        Fit the averaged event to a double exponential epsc-like function
        Operates on the AverageEvent data structure
        """
        # self.numpyerror = np.geterr()
        # np.seterr(all="raise")
        # self.scipyerror = scipy.special.geterr()
        # scipy.special.seterr(all="raise")
        tsel = 0  # use whole averaged trace
        self.tsel = tsel
        self.tau1 = inittaus[0]
        self.tau2 = inittaus[1]
        self.tau_ratio = self.tau2 / self.tau1
        self.tau2_range = 10.0
        self.tau1_minimum_factor = 5.0
        time_past_peak = 2.5e-4
        self.fitted_tau1 = np.nan
        self.fitted_tau2 = np.nan
        self.Amplitude = np.nan
        self.avg_fiterr = np.nan
        self.bfdelay = np.nan
        self.avg_best_fit = None
        self.Qtotal = np.nan
        self.risetenninety = np.nan
        self.decaythirtyseven = np.nan
        CP.cprint("c", f"        Fitting average event, fixed_delay={initdelay:f}")
        res = self.event_fitter_lm(
            tb,
            avgevent,
            time_past_peak=time_past_peak,
            fixed_delay=initdelay,
            debug=debug,
            label=label,
        )
        if res is None:
            CP.cprint(
                "r",
                f"**** Average event fit result is None [data source = {str(self.datasource):s}]",
            )
            self.fitted = False
            np.seterr(**self.numpyerror)  # reset error detection
            scipy.special.seterr(**self.scipyerror)
            return
        # print(res)
        self.Amplitude = res.values["amp"]
        self.fitted_tau1 = res.values["tau_1"]
        self.fitted_tau2 = res.values["tau_2"]
        self.fitted_tau_ratio = (
            res.values["tau_2"] / res.values["tau_1"]
        )  # res.values["tau_ratio"]
        self.bfdelay = res.values["fixed_delay"]
        self.avg_best_fit = self.sign * res.best_fit  # self.fitresult.best_fit
        # f, ax = mpl.subplots(1,1)
        # ax.plot(tb, avgevent)
        # ax.plot(tb, self.avg_best_fit, "r--")
        # mpl.show()
        # print(self.fitted_tau1, self.fitted_tau2)
        # exit()
        fiterr = np.linalg.norm(self.avg_best_fit - avgevent[self.tsel :])
        self.avg_fiterr = fiterr
        ave = self.sign * avgevent
        move_avg, n = self.moving_average(  # apply 5 point moving average
            ave,
            n=min(5, len(ave)),
        )
        ipk = np.argmax(move_avg)
        pk = move_avg[ipk]
        p10 = 0.10 * pk
        p90 = 0.90 * pk
        p37 = 0.37 * pk

        i10 = np.nonzero(move_avg[:ipk] <= p10)[0]
        if len(i10) == 0:
            self.risetenninety = np.nan
        else:
            i10 = i10[-1]  # get the last point where this condition was met
            ix10 = np.interp(p10, move_avg[i10 : i10 + 2], [0, self.dt_seconds])
            i90 = np.nonzero(move_avg[: ipk + 1] >= p90)[0]
            ix90 = np.interp(p90, move_avg[ipk - 2 : ipk], [0, self.dt_seconds])
            it10 = ix10 + i10 * self.dt_seconds
            it90 = ix90 + (ipk - 2) * self.dt_seconds
            try:
                i90 = i90[0]  # get the first point where this condition was met
            except:
                CP.cprint("r", "**** Error in fit_average computing 10-90 RT")
                print(move_avg[:ipk], p90)
                print(i90, ix90, ix90 + ipk - 2)
                print(i10, ix10, ix10 + i10)
                exit()
            self.risetenninety = it90 - it10

        i37 = np.nonzero(move_avg[ipk:] >= p37)[-1]
        if len(i37) == 0:
            self.decaythirtyseven = np.nan
        else:
            i37 = i37[0]  # first point
            self.decaythirtyseven = self.dt_seconds * i37
        self.Qtotal = self.dt_seconds * np.sum(avgevent[self.tsel :])
        self.fitted = True
        # np.seterr(**self.numpyerror)
        # scipy.special.seterr(**self.scipyerror)

    def fit_individual_events(self) -> None:
        """
        Fitting individual events
        Events to be fit are selected from the entire event pool as:
        1. events that are completely within the trace, AND
        2. events that do not overlap other events

        Fit events are further classified according to the fit error

        """
        if (
            not self.Summary.average.averaged or not self.fitted
        ):  # averaging should be done first: stores events for convenience and gives some tau estimates
            print(
                "Require fit of averaged events prior to fitting individual events",
                self.Summary.average.averaged,
            )
            raise (ValueError)
        onsets = self.Summary.onsets
        time_past_peak = 0.1  # msec - time after peak to start fitting
        fixed_delay = self.template_pre_time
        # allocate arrays for results. Arrays have space for ALL events
        # okevents, notok, and self.events_ok are indices
        nevents = np.sum(np.sum(len(o)) for o in onsets)
        self.ev_fitamp = np.zeros(nevents)  # measured peak amplitude from the fit
        self.ev_A_fitamp = np.zeros(
            nevents
        )  # fit amplitude - raw value can be quite different than true amplitude.....

        def nanfill(n):
            a = np.empty(n)
            a.fill(np.nan)
            return a

        self.ev_tau1 = nanfill(nevents)
        self.ev_tau2 = nanfill(nevents)
        self.ev_1090 = nanfill(nevents)
        self.ev_2080 = nanfill(nevents)
        self.ev_amp = nanfill(nevents)  # measured peak amplitude from the event itself
        self.ev_Qtotal = nanfill(
            nevents
        )  # measured charge of the event (integral of current * dt)
        self.ev_Q_end = nanfill(
            nevents
        )  # measured charge of last half of event (integral of current * dt)
        self.fiterr = nanfill(nevents)
        self.bfdelay = nanfill(nevents)
        self.best_fit = (
            np.zeros((nevents, self.Summary.average.avgeventtb.shape[0])) * np.nan
        )
        self.best_decay_fit = (
            np.zeros((nevents, self.Summary.average.avgeventtb.shape[0])) * np.nan
        )
        self.tsel = 0
        self.tau2_range = 10.0
        self.tau1_minimum_factor = 5.0

        # prescreen events
        minint = np.max(
            self.Summary.average.avgeventtb
        )  # msec minimum interval between events.
        self.fitted_events = (
            []
        )  # events that can be used (may not be all events, but these are the events that were fit)

        # only use "well-isolated" events in time to make the fit measurements.
        for j, ev_tr in enumerate(
            self.Summary.clean_event_trace_list
        ):  # trace list of events
            # print('onsetsj: ', len(onsets[j]))
            if not ev_tr:  # event in this trace could be outside data window, so skip
                continue
            i_tr, j_tr = ev_tr
            te = self.timebase[onsets[i_tr][j_tr]]  # get current event time
            try:
                tn = self.timebase[onsets[i_tr][j_tr + 1]]  # check time to next event
                if (tn - te) < minint:  # event is followed by too soon by another event
                    continue
            except:
                pass  # just handle trace end condition
            try:
                tp = self.timebase[onsets[i_tr][j_tr - 1]]  # check previous event
                if (
                    te - tp
                ) < minint:  # if current event too close to a previous event, skip
                    continue
            except:
                pass
            j_nan = np.count_nonzero(np.isnan(self.Summary.allevents[j, :]))
            if j_nan > 0:
                raise ValueError(
                    f"Event array {j:d} has {j_nan:d} nan values in it, array length = {len(self.Summary.allevents[j, :]):d} and {len(onsets[i_tr]):d} onset values"
                )

            try:
                max_event = np.max(self.sign * self.Summary.allevents[j, :])
            except:
                CP.cprint("r", "FITTING FAILED: ")
                print("  minis_methods eventfitter")
                print("  j: ", j)
                print("  allev: ", self.Summary.allevents)
                print("  len allev: ", len(self.Summary.allevents), len(onsets))
                raise ValueError("  Fit failed)")
            res = self.event_fitter_lm(
                timebase=self.Summary.average.avgeventtb,
                event=self.Summary.allevents[j, :],
                time_past_peak=time_past_peak,
                fixed_delay=fixed_delay,
                label=f"Fitting event in trace: {str(ev_tr):s}  j = {j:d}",
            )
            if res is None:  # skip events that won't fit
                continue

            self.bfdelay[j] = res.values["fixed_delay"]
            self.avg_best_fit = res.best_fit
            self.ev_A_fitamp[j] = res.values["amp"]
            self.ev_tau1[j] = res.values["tau_1"]
            self.ev_tau2[j] = res.values["tau_2"]

            # print(self.fitresult.params)
            # print('amp: ', self.fitresult.params['amp'].value)
            # print('tau_1: ', self.fitresult.params['tau_1'].value)
            # print('tau_2: ', self.fitresult.params['tau_2'].value)
            # print('risepower: ', self.fitresult.params['risepower'].value)
            # print('fixed_delay: ', self.fitresult.params['fixed_delay'].value)
            # print('y shape: ', np.shape(self.Summary.allevents[j,:]))
            # print('x shape: ', np.shape(self.Summary.average.avgeventtb))
            self.fiterr[j] = self.doubleexp_lm(
                y=self.Summary.allevents[j],
                time=self.Summary.average.avgeventtb,
                amp=self.fitresult.params["amp"].value,
                tau_1=self.fitresult.params["tau_1"].value,
                tau_2=self.fitresult.params["tau_2"].value,
                # self.sign * self.Summary.allevents[j, :],
                risepower=self.fitresult.params["risepower"].value,
                fixed_delay=self.fitresult.params[
                    "fixed_delay"
                ].value,  # self.bfdelay[j],
            )
            self.best_fit[j] = res.best_fit
            # self.best_decay_fit[j] = self.decay_fit  # from event_fitter
            self.ev_fitamp[j] = np.max(self.best_fit[j])
            self.ev_Qtotal[j] = self.dt_seconds * np.sum(
                self.sign * self.Summary.allevents[j, :]
            )
            last_half = int(self.Summary.allevents.shape[1] / 2)
            self.ev_Q_end[j] = self.dt_seconds * np.sum(
                self.Summary.allevents[j, last_half:]
            )
            self.ev_amp[j] = np.max(self.sign * self.Summary.allevents[j, :])
            self.fitted_events.append(j)
        self.individual_event_screen(
            fit_err_limit=2000.0, tau2_range=10.0, verbose=False
        )
        self.individual_events = True  # we did this step

    @staticmethod
    def doubleexp_lm(
        time: np.ndarray,
        amp: float,
        tau_1: float,
        tau_2: float,
        risepower: float,
        fixed_delay: float = 0.0,
        y: np.ndarray = None,
    ) -> np.ndarray:
        """
        Calculate a double exponential EPSC-like waveform. The rising phase
        is taken to a power to make it sigmoidal
        Note that the parameters p[1] and p[2] are multiplied, rather
        than divided here, so the values must be inverted in the caller
        from the "time constants"
        """
        # fixed_delay = p[3]  # allow to adjust; ignore input value
        ix = np.argmin(np.fabs(time - fixed_delay))
        tm = np.zeros_like(time)
        tx = time[ix:] - fixed_delay
        exp_arg1 = tx / tau_1
        exp_arg1[exp_arg1 > 30] = 30  # limit values
        # exp_arg1 = np.clip(exp_arg1, np.log(1e-16), 0)
        # print(f"Fitting function with: {amp:.3e}, tau1: {tau_1:.4e}, risepower: {risepower:.2f}")
        try:
            tm[ix:] = amp * (1.0 - np.exp(-exp_arg1)) ** risepower
        except:
            print(
                f"Fitting function failed: {amp:.3e}, tau1: {tau_1:.4e}, risepower: {risepower:.2f}"
            )
            raise ValueError()
        tm[ix:] *= np.exp(-tx / tau_2)
        if y is not None:  # return error - single value
            tm = np.sqrt(np.sum((tm - y) * (tm - y)))
        return tm

    def event_fitter_lm(
        self,
        timebase: np.ndarray,
        event: np.ndarray,
        time_past_peak: float = 1e-4,
        tau1: float = None,
        tau2: float = None,
        init_amp: float = None,
        fixed_delay: float = 0.0,
        debug: bool = False,
        label: str = "",
        j: int = 0,
    ) -> Tuple[dict, float]:
        """
        Fit the event using lmfit and LevenbergMarquardt
        Using lmfit is a bit more disciplined approach than just using scipy.optimize

        """
        evfit, peak_pos, maxev = self.set_fit_delay(event, initdelay=fixed_delay)
        if peak_pos == len(event):
            peak_pos = len(event) - 10
        dexpmodel = lmfit.Model(self.doubleexp_lm)
        params = lmfit.Parameters()
        # get some logical initial parameters
        init_amp = maxev
        if tau1 is None:
            tau1 = 0.67 * peak_pos * self.dt_seconds
        if tau2 is None:
            # first smooth the decay a bit
            decay_data = SPS.savgol_filter(
                self.sign * event[peak_pos:], 11, 3, mode="nearest"
            )
            # try to look for the first point at 0.37 height
            itau2 = np.nonzero(decay_data < self.sign * 0.37 * event[peak_pos])[0]
            if len(itau2) == 0:
                itau2 = np.nonzero(decay_data < self.sign * 0.37 * event[peak_pos])[-1]
                if len(itau2) == 0:
                    itau2 = [
                        int(0.005 - fixed_delay) / self.dt_seconds
                    ]  # print("itau2: ", itau2)
            tau2 = (itau2[0] - fixed_delay) * self.dt_seconds
        if tau2 <= 2 * tau1:
            tau2 = 5 * tau1  # move it further out
        amp = event[peak_pos]
        # print("initial tau1: ", tau1, "tau2: ", tau2)

        if self.datatype in ["V", "VC"]:
            tau1min = tau1 / 10.0
            if tau1min < 1e-4:
                tau1min = 1e-4
            tau1_maxfac = 3.0
            tau2_maxfac = 5.0
            tau2_minfac = 1.0 / 20
            params["amp"] = lmfit.Parameter(
                name="amp", value=25.0e-12, min=0.0, max=50e-9, vary=True
            )
        elif self.datatype in ["I", "IC"]:
            tau1min = tau1 / 10.0
            if tau1min < 1e-4:
                tau1min = 1e-4
            tau1_maxfac = 10.0
            tau2_maxfac = 20.0
            tau2_minfac = 1.0 / 20.0
            # params["amp"] = lmfit.Parameter(
            #     name="amp", value=1.0e-3, min=0.0, max=50e-3, vary=True
            # )
            params["amp"] = lmfit.Parameter(
                name="amp",
                value=amp,
                min=0.0,
                max=5 * amp,
                vary=True,
            )
        else:
            raise ValueError("Data type must be VC or IC: got", self.datatype)
        params["tau_1"] = lmfit.Parameter(
            name="tau_1",
            value=tau1,
            min=tau1min,
            max=5 * tau1,  # tau1 * tau1_maxfac,
            vary=True,
        )
        # params["tau_ratio"] = lmfit.Parameter(
        #     name="tau_ratio",
        #     value = self.tau2/self.tau1,
        #     min = 1.0,
        #     max = 50.0,
        #     vary = True,
        # )
        params["tau_2"] = lmfit.Parameter(
            name="tau_2",
            # expr="tau_1*tau_ratio"
            value=tau2,
            min=tau1,  # tau2 * tau2_minfac,
            max=15 * tau2,  # tau2 * tau2_maxfac,
            vary=True,
        )
        params["fixed_delay"] = lmfit.Parameter(
            name="fixed_delay",
            value=fixed_delay,
            vary=False,
        )
        params["risepower"] = lmfit.Parameter(
            name="risepower", value=self.risepower, vary=False
        )

        self.fitresult = dexpmodel.fit(evfit, params, nan_policy="raise", time=timebase)

        self.peak_val = maxev
        self.evfit = self.fitresult.best_fit  # handy right out of the result

        if debug:
            import matplotlib.pyplot as mpl

            mpl.figure()
            mpl.plot(timebase, evfit, "k-")
            mpl.plot(timebase, self.fitresult.best_fit, "r--")
            print(self.fitresult.fit_report())
            mpl.show()

        return self.fitresult

    def set_fit_delay(self, event, initdelay: Union[float, None] = None):
        if initdelay is None:
            init_delay = 0.0  # in seconds
        else:
            init_delay = int(initdelay / self.dt_seconds)
        ev_bl = 0.0  # np.mean(event[: init_delay])  # just first point...
        evfit = self.sign * (event - ev_bl)
        maxev = np.max(evfit)
        if maxev == 0:
            maxev = 1
        peak_pos = np.argmax(evfit) + 1
        if peak_pos <= init_delay:
            peak_pos = init_delay + 5
        return evfit, peak_pos, maxev

    def individual_event_screen(
        self,
        fit_err_limit: float = 2000.0,
        tau2_range: float = 2.5,
        verbose: bool = False,
    ) -> None:
        """
        Screen events:
        error of the fit must be less than a limit,
        and
        tau2 must fall within a range of the default tau2
        and
        tau1 must be greater than a minimum tau1
        and
        sign of total charge must match sign of event
        sets:
        self.events_ok : the list of fitted events that pass
        self.events_notok : the list of fitted events that did not pass
        """
        self.events_ok = []
        failed_charge = 0
        for i in self.fitted_events:  # these are the events that were fit
            if verbose:
                print(f" fiterr: {i:d}: {self.fiterr[i]:.3e}", end="")
            if self.sign * self.ev_Q_end[i] < 0.0:
                failed_charge += 1
                # print(f"Event {i:d} Failed charge screen: Q = {self.ev_Q_end[i]:.3e}")
                continue

            if self.fiterr[i] <= fit_err_limit:
                if self.ev_tau2[i] <= self.tau2_range * self.tau2:
                    if verbose:
                        print(
                            f"  passed tau: {self.ev_tau2[i]:.3e} <= {self.tau2_range * self.tau2:.3e}",
                            end="",
                        )
                    if self.ev_fitamp[i] > self.min_event_amplitude:
                        if verbose:
                            print(f"  > min amplitude. ", end="")
                        if self.ev_tau1[i] > self.tau1 / self.tau1_minimum_factor:
                            if verbose:
                                print(
                                    f" tau1 > {self.tau1/self.tau1_minimum_factor:.3f}, ** all passed",
                                    end="",
                                )
                            self.events_ok.append(i)

            if verbose:
                print()
        if failed_charge > 0:
            CP.cprint("y", f"{failed_charge:d} events failed charge screening")

        self.events_notok = list(set(self.fitted_events).difference(self.events_ok))
        if verbose:
            print("Events ok: ", self.events_ok)
            print("Events not ok: ", self.events_notok)

    def plot_individual_events(
        self, fit_err_limit: float = 1000.0, tau2_range: float = 2.5, show: bool = True
    ) -> None:
        """
        Plot individual events
        """
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
        #        self.events_ok, notok = self.individual_event_screen(fit_err_limit=fit_err_limit, tau2_range=tau2_range)
        P.axdict["A"].plot(
            self.ev_tau1[self.events_ok],
            self.ev_amp[self.events_ok],
            "ko",
            markersize=4,
        )
        P.axdict["A"].set_xlabel(r"$tau_1$ (ms)")
        P.axdict["A"].set_ylabel(r"Amp (pA)")
        P.axdict["B"].plot(
            self.ev_tau2[self.events_ok],
            self.ev_amp[self.events_ok],
            "ko",
            markersize=4,
        )
        P.axdict["B"].set_xlabel(r"$tau_2$ (ms)")
        P.axdict["B"].set_ylabel(r"Amp (pA)")
        P.axdict["C"].plot(
            self.ev_tau1[self.events_ok],
            self.ev_tau2[self.events_ok],
            "ko",
            markersize=4,
        )
        P.axdict["C"].set_xlabel(r"$\tau_1$ (ms)")
        P.axdict["C"].set_ylabel(r"$\tau_2$ (ms)")
        P.axdict["D"].plot(
            self.ev_amp[self.events_ok], self.fiterr[self.events_ok], "go", markersize=3
        )
        P.axdict["D"].plot(
            self.ev_amp[self.events_notok],
            self.fiterr[self.events_notok],
            "yo",
            markersize=3,
        )
        P.axdict["D"].set_xlabel(r"Amp (pA)")
        P.axdict["D"].set_ylabel(r"Fit Error (cost)")
        for i in self.events_ok:
            ev_bl = np.mean(self.Summary.allevents[i, 0:5])
            P.axdict["E"].plot(
                self.Summary.average.avgeventtb,
                self.Summary.allevents[i] - ev_bl,
                "b-",
                linewidth=0.75,
            )
            # P.axdict['E'].plot()
            P.axdict["F"].plot(
                self.Summary.average.avgeventtb,
                self.Summary.allevents[i] - ev_bl,
                "r-",
                linewidth=0.75,
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
        for i in self.events_ok:
            offset = i * 3.0
            ev_bl = np.mean(self.Summary.allevents[i, 0:5])
            P2.axdict["A"].plot(
                self.Summary.average.avgeventtb,
                self.Summary.allevents[i] + offset - ev_bl,
                "k-",
                linewidth=0.35,
            )
            P2.axdict["A"].plot(
                self.Summary.average.avgeventtb,
                self.best_fit[i] + offset,
                "c--",
                linewidth=0.3,
            )
            # P2.axdict["A"].plot(
            #     self.Summary.average.avgeventtb,
            #     self.sign * self.best_decay_fit[i] + offset,
            #     "r--",
            #     linewidth=0.3,
            # )
            P3.axdict[idx[k]].plot(
                self.Summary.average.avgeventtb,
                self.Summary.allevents[i] + offset2,
                "k--",
                linewidth=0.3,
            )
            P3.axdict[idx[k]].plot(
                self.Summary.average.avgeventtb,
                self.best_fit[i] + offset2,
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

    # note: this is only called from cs.py, not otherwised used.
    def plots(
        self,
        data,
        events: Union[np.ndarray, None] = None,
        title: Union[str, None] = None,
        testmode: bool = False,
        index: int = 0,
    ) -> object:
        """
        Plot the results from the analysis and the fitting
        """
        if testmode is False:
            return
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
        for i in range(2):
            ax[i].sharex(ax[i + 1])
        # raw traces, marked with onsets and peaks
        ax[0].set_ylabel("I (pA)")
        ax[0].set_xlabel("T (s)")
        ax[1].set_ylabel("Deconvolution")
        ax[1].set_xlabel("T (s)")
        ax[2].set_ylabel("Averaged I (pA)")
        ax[2].set_xlabel("T (s)")
        if title is not None:
            P.figure_handle.suptitle(title)
        for i, d in enumerate(data):
            self.plot_trial(self.P.axarr.ravel(), i, d, events, index=index)

        if testmode:  # just display briefly
            mpl.show(block=False)
            mpl.pause(1)
            mpl.close()
            return None
        else:
            return self.P.figure_handle

    def plot_trial(
        self, ax, i, data, events, markersonly: bool = False, index: int = 0
    ):
        onset_marks = {0: "k^", 1: "b^", 2: "m^", 3: "c^"}
        peak_marks = {0: "+", 1: "g+", 2: "y+", 3: "k+"}
        scf = 1e12
        tb = self.timebase[: data.shape[0]]
        label = "Data"
        ax[0].plot(tb, scf * data, "k-", linewidth=0.5, label=label)  # original data
        label = "Onsets"
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
                label = "Smoothed Peaks"
            else:
                label = ""
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
            label = ""
        ax[1].plot(tb[: self.Criterion[i].shape[0]], self.Criterion[i], label=label)

        if i == 0:
            label = "Threshold ({0:4.2f}) SD".format(self.sdthr)
        else:
            label = ""
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
            label = ""
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
                label = f"Average Event (N={nev:d})"
            else:
                label = ""
            evlen = len(self.Summary.average.avgevent)
            ax[2].plot(
                self.Summary.average.avgeventtb[:evlen],
                scf * self.Summary.average.avgevent,
                "k-",
                label=label,
            )
            maxa = np.max(self.sign * self.Summary.average.avgevent)
            if self.template is not None:
                maxl = int(
                    np.min([len(self.template), len(self.Summary.average.avgeventtb)])
                )
                temp_tb = np.arange(0, maxl * self.dt_seconds, self.dt_seconds)
                if i == 0:
                    label = "Template"
                else:
                    label = ""
                ax[2].plot(
                    self.Summary.average.avgeventtb[:maxl] + self.bfdelay,
                    scf * self.sign * self.template[:maxl] * maxa / self.template_amax,
                    "m-",
                    label=label,
                )

            sf = 1.0
            tau1 = np.power(
                10, (1.0 / self.risepower) * np.log10(self.tau1 * sf)
            )  # correct for rise power
            tau2 = self.tau2 * sf

            if i == 0:
                label = "Best Fit:\nRise Power={0:.2f}\nTau1={1:.3f} ms\nTau2={2:.3f} ms\ndelay: {3:.3f} ms".format(
                    self.risepower,
                    self.tau1 * sf,
                    self.tau2 * sf,
                    self.bfdelay * sf,
                )
            else:
                label = ""
            ax[2].plot(
                self.Summary.average.avgeventtb[: len(self.avg_best_fit)],
                scf * self.avg_best_fit,
                "c--",
                linewidth=2.0,
                label=label,
            )
