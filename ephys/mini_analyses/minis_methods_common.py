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
import itertools
import lmfit
import matplotlib.pyplot as mpl
import matplotlib.colors
import numpy as np
import pandas as pd
import pylibrary.plotting.plothelpers as PH
import pylibrary.tools.cprint as CP
import scipy.signal as SPS
import scipy.special
import meegkit as MEK
from scipy.optimize import curve_fit
import ephys.tools.functions as FUNCS

import ephys.tools.digital_filters as dfilt
import ephys.mini_analyses.mini_event_dataclasses as MEDC  # get result datastructure


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
        self.filters = MEDC.Filtering()  # filtering class
        self.risepower = 4.0  # power for sigmoidal rise when fitting events
        self.min_event_amplitude = 5.0e-12  # pA default for minimum event size
        self.eventstartthr = None
        self.Criterion = [None]  # array for C-B
        self.template = None  # template for C-B
        self.template_tmax = 0.0
        self.analysis_window = [None, None]  # specify window or entire data set
        self.datatype = None
        self.events_ok = []
        self.individual_events = False
        self.do_individual_fits = False
        self.events_notok = []
        self.filters_enabled = True  # enable filtering
        self.data_prepared = False
        self.summary = None  # This gets set when we run summarize
        super().__init__()

    def setup(
        self,
        datasource: str = "",
        ntraces: int = 1,
        tau1: Union[float, None] = None,
        tau2: Union[float, None] = None,
        tau3: Union[float, None] = 2e-3,
        tau4: Union[float, None] = 30e-3,
        template_tmax: float = 0.05,
        template_pre_time: float = 0.0,
        dt_seconds: Union[float, None] = None,
        delay: float = 0.0,  # into start of each trace for analysis, seconds
        sign: int = 1,
        eventstartthr: Union[float, None] = None,
        risepower: Union[float, None] = None,
        min_event_amplitude: float = 5.0e-12,
        threshold: float = 2.5,
        global_SD: Union[float, None] = None,
        analysis_window: List[Union[float, None]] = [None, None],
        filters: Union[MEDC.Filtering, None] = None,
        do_individual_fits: bool = False,
    ) -> None:
        """
        Just store the parameters - will compute when needed
        Use of globalSD and threshold:
        if global SD is None, we use the threshold as it.

        If global SD has a value, then we use that rather than the
        current trace SD for threshold determinations
        """
        CP.cprint("c", f"MiniAnalysis SETUP {str(self):s}")

        assert sign in [-1, 1]  # must be selective, positive or negative events only
        self.datasource = datasource
        self.ntraces = ntraces
        self.Criterion = [[None] for x in range(ntraces)]
        self.sign = sign
        self.taus = [tau1, tau2, tau3, tau4]
        self.dt_seconds = dt_seconds
        self.template_tmax = template_tmax
        self.template_pre_time = template_pre_time
        # self.idelay = int(delay / self.dt_seconds)
        self.template = None  # reset the template if needed.
        if eventstartthr is not None:
            self.eventstartthr = eventstartthr
        if risepower is not None:
            self.risepower = risepower
        self.min_event_amplitude = min_event_amplitude
        self.threshold = threshold
        self.sdthr = self.threshold  # for starters
        self.analysis_window = analysis_window
        self.do_individual_fits = do_individual_fits
        self.set_filters(filters)
        self.set_datatype("VC")
        print("MiniAnalysis - template pre time: ", self.template_pre_time)

    
    def set_filters(self, filters):
        """Set the filtering arguments

        Args:
            filters (dataclass): Class structure with filtering information
        """
        if filters is None:
            self.filters = MEDC.Filtering()
        else:
            self.filters = filters
        self.reset_filters()
        # print("Filter set in MMC: \n", self.filters)

    def set_datatype(self, datatype: str):
        CP.cprint("c", f"data type: {datatype:s}")
        self.datatype = datatype

    def set_sign(self, sign: int = 1):
        self.sign = sign

    def set_dt_seconds(self, dt_seconds: Union[None, float] = None):
        self.dt_seconds = dt_seconds

    def set_timebase(self, timebase: np.ndarray):
        """Set the timebase - this should usually be called from
        somewhere outside this module. The timebase will be set
        automatically if we are using the routines here

        Args:
            timebase (np.ndarray): timebase array corresponding to the data
        """
        self.timebase = timebase

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

    def _make_template(self, timebase: np.array):
        """Private function: make template when it is needed

        Args:
            timebase (np.array): time points for computation of the template array
        """

        assert timebase is not None
        tau_1, tau_2 = self.taus[:2]  # use the predefined taus
        tmax = np.min((self.template_tmax + self.template_pre_time, timebase.max()))
        # make template time base
        t_psc = np.arange(0, tmax, self.dt_seconds)
        # give the template a possible pre-event baseline
        template_delay = int(
            self.template_pre_time / self.dt_seconds
        )  # delay to template waveform start
        self.template = np.zeros_like(t_psc)
        i_pscend = len(t_psc) - template_delay
        Aprime = (tau_2 / tau_1) ** (tau_1 / (tau_1 - tau_2))
        # calculate template waveform - just one rising/falling tau.
        self.template[template_delay:] = (
            1.0
            / Aprime
            * (
                (1 - (np.exp(-t_psc[:i_pscend] / tau_1))) ** self.risepower
                * np.exp((-t_psc[:i_pscend] / tau_2))
            )
        )
        if self.sign > 0:
            self.template_amax = np.max(self.template)
        else:
            self.template = -self.template
            self.template_amax = np.min(self.template)

    def reset_filters(self):
        """
        Reset the filtering flags so we know which have been done.
        The purpose of this is to keep from applying filters repeatedly
        """
        self.filters.Detrend_applied = False
        self.filters.LPF_applied = False
        self.filters.HPF_applied = False
        self.filters.Notch_applied = False
        self.data_prepared = False

    def filters_on(self):
        """Turn the filtering OFF (e.g., data is pre-filtered)"""
        self.filters.enabled = True

    def filters_off(self):
        """Turn the filtering OFF (e.g., data is pre-filtered)"""
        self.filters.enabled = False

    def LPFData(self, data: np.ndarray, NPole: int = 8) -> np.ndarray:
        """
        Low pass filter the data. This uses the LPF_frequency in
        the self.filters dataclass for the value

        Parameters
        ----------
        data : the  array of data
        NPole : number of poles for designing the filter

        Return
        ------
        data : the filtered data (as a copy)
        """
        if not self.filters.enabled:
            return data
        if self.verbose:
            CP.cprint(
                "y",
                f"        minis_methods_common, LPF data:  {self.filters.LPF_frequency:f}",
            )
        if self.filters.LPF_frequency is not None:
            # CP.cprint('y', f"     ... lpf at {lpf:f}")
            if self.filters.LPF_frequency > 0.49 / self.dt_seconds:
                raise ValueError(
                    "lpf > Nyquist: ",
                    self.filters.LPF_frequency,
                    0.49 / self.dt_seconds,
                    self.dt_seconds,
                    1.0 / self.dt_seconds,
                )
            # data = dfilt.SignalFilter_LPFButter(data, lpf, 1./self.dt_seconds, NPole=8)
            if self.filters.LPF_type == "ba":
                fdata = dfilt.SignalFilter_LPFBessel(
                    data,
                    LPF=self.filters.LPF_frequency,
                    samplefreq=1.0 / self.dt_seconds,
                    NPole=4,
                    filtertype="low",
                )
            elif self.filters.LPF_type == "sos":
                fdata = dfilt.SignalFilterLPF_SOS(
                    data,
                    LPF=self.filters.LPF_frequency,
                    samplefreq=1.0 / self.dt_seconds,
                    NPole=8,
                )
            else:
                raise ValueError(
                    f"Signal filter type must be 'ba' or 'sos': got {self.filters.LPF_type:s}"
                )
            self.filters.LPF_applied = True
            return fdata

    def HPFData(self, data: np.ndarray, NPole: int = 8) -> np.ndarray:
        """
        High pass filter the data. This uses the HPF_frequency in
        the self.filters dataclass for the value

        Parameters
        ----------
        data : the  array of data
        NPole : number of poles for designing the filter

        Return
        ------
        data : the filtered data (as a copy)
        """
        if not self.filters.enabled:
            return data
        if self.verbose:
            CP.cprint(
                "y",
                f"        minis_methods_common, HPF data:  {self.filters.HPF_frequency:f}",
            )
        if self.filters.HPF_frequency is None or self.filters.HPF_frequency == 0.0:
            return data
        if len(data.shape) == 1:
            ndata = data.shape[0]
        else:
            ndata = data.shape[1]
        nyqf = 0.5 * ndata * self.dt_seconds
        if self.verbose:
            CP.cprint("y", f"minis_methods: hpf at {self.filters.HPF_frequency:f}")
        if self.filters.HPF_frequency < 1.0 / nyqf:  # duration of a trace
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
        if self.filters.HPF_type == "ba":
            fdata = dfilt.SignalFilter_HPFButter(
                data - data[0],
                self.filters.HPF_frequency,
                1.0 / self.dt_seconds,
                NPole=4,
            )
        elif self.filters.LPF_type == "sos":
            fdata = dfilt.SignalFilterLPF_SOS(
                data,
                HPF=self.filters.HPF_frequency,
                samplefreq=1.0 / self.dt_seconds,
                NPole=8,
            )
        else:
            raise ValueError(
                f"Signal filter type must be 'ba' or 'sos': got {self.filters.HPF_type:s}"
            )
        self.filters.HPF_applied = True
        return fdata

    def NotchData(
        self,
        data: np.ndarray,
    ) -> np.ndarray:
        """
        Notch filter the data
        This routine can apply multiple notch filters to the data at once.
        The filter values
            notch: list : list of notch frequencies, in Hz
            notch_Q : the Q of the filter (higher is narrower)
        are in the self.filters dataclass

        Parameters
        ----------
        data : the  array of data

        Return
        ------
        data : the filtered data (as a copy)
        """
        if not self.filters.enabled:
            return data
        CP.cprint(
            "y",
            f"         minis_methods_common, Notch filter data:  {str(self.filters.Notch_frequencies):s}",
        )

        if (
            self.filters.Notch_frequencies is None
            or len(self.filters.Notch_frequencies) == 0
        ):
            return data
        fdata = dfilt.NotchFilterZP(
            data,
            notchf=self.filters.Notch_frequencies,
            Q=self.filters.Notch_Q,
            QScale=False,
            samplefreq=1.0 / self.dt_seconds,
        )
        self.filters.Notch_applied = True
        return fdata

    def NotchFilterComb(
        self,
        data: np.ndarray,
    ) -> np.ndarray:
        """
        Notch filter the data with a comb filter
        This routine does a comb filter with the first listed notch frequency.

        Parameters
        ----------
        data : the  array of data
        notch: list : list of notch frequencies, in Hz
        notch_Q : the Q of the filter (higher is narrower)

        Return
        ------
        data : the filtered data (as a copy)
        """
        if not self.filters.enabled:
            return data
        if self.verbose:
            CP.cprint(
                "y",
                f"minis_methods_common, Notch comb filter data:  {self.filters.Notch_frequencies[0]:f}",
            )

        if (
            self.filters.Notch_frequencies is None
        ):
            return data

        fdata = dfilt.NotchFilterComb(
            data,
            notchf=self.filters.Notch_frequencies,
            Q=self.filters.Notch_Q,
            QScale=False,
            samplefreq=1.0 / self.dt_seconds,
        )
        self.filters.Notch_applied = True
        return fdata

    def prepare_data(self, data: np.array):
        """
        This function prepares the incoming data for the mini analyses.
        1. Clip the data in time (remove sections with current or voltage steps)
        2. detrend (either with meegkit or scipy)
        3. remove mean using first 10 msec of remaining trace
        4. Filter the data: order is LPF, HPF (if set)
        5. Apply a notch filter (if set)

        Parameters
        ----------
        data : np array

        Returns
        -------
        Nothing. The result is held in the class variable "data", along with a
        corresponding timebase.
        """
        if self.data_prepared:
            return
        if data.ndim != 2:
            raise ValueError

        CP.cprint("c", "Preparing data")
        timebase = np.arange(0.0, data.shape[1] * self.dt_seconds, self.dt_seconds)
        print(
            "original data : ",
            data.shape,
            " timebase: ",
            timebase.shape,
            np.max(timebase),
        )
        print("analysis window: ", self.analysis_window)
        #
        # 1. Clip the timespan of the data to the values in analysis_window
        #
        if self.analysis_window[1] is not None:
            jmax = np.argmin(np.fabs(timebase - self.analysis_window[1]))
        else:
            jmax = len(self.timebase)
        if self.analysis_window[0] is not None:
            jmin = np.argmin(np.fabs(timebase - self.analysis_window[0]))
        else:
            jmin = 0
        data = data[:, jmin:jmax]
        self.timebase = timebase[jmin:jmax]
        print(
            "    Preparing data: Window clipped: ",
            data.shape,
            jmin,
            jmax,
            np.min(timebase),
            np.max(timebase),
            self.analysis_window,
        )
        print(f"    Preparing Data: Filters Enabled = {str(self.filters.enabled):s}")
        if not self.filters.enabled:
            self.data = data
            self.data_prepared = False
            return
        print(f"    Preparing data: LPF = {str(self.filters.LPF_frequency):s}")
        print(f"    Preparing data: HPF = {str(self.filters.HPF_frequency):s}")
        print(f"    Preparing data: Notch = {str(self.filters.Notch_frequencies):s}")
        print(f"    Preparing data: detrend: {str(self.filters.Detrend_type):s}")
        if self.verbose:
            if self.filters.LPF_frequency is not None:
                CP.cprint(
                    "y",
                    f"minis_methods_common, prepare_data: LPF: {str(self.filters.LPF_frequency):s} Hz",
                )
            else:
                CP.cprint("r", f"**** minis_methods_common, no LPF applied")
            if self.filters.HPF_frequency is not None:
                CP.cprint(
                    "y",
                    f"    minis_methods_common, prepare_data: HPF: {str(self.filters.HPF_frequency):s} Hz",
                )
            else:
                CP.cprint("r", f"    minis_methods_common, no HPF applied")
        filters_applied = ""
        #
        # 2. detrend. Method depends on self.filters.Detrend_type
        #

        if self.filters.Detrend_type == "meegkit":
            for itrace in range(data.shape[0]):
                data[itrace], _, _ = MEK.detrend.detrend(
                    data[itrace], order=self.filters.Detrend_order
                )
                self.filters.Detrend_applied = True
            filters_applied += "Detrend: meegkit  "
        elif self.filters.Detrend_type == "scipy":
            for itrace in range(data.shape[0]):
                data[itrace] = FUNCS.adaptiveDetrend(
                    data[itrace], x=self.timebase[jmin:jmax], threshold=3.0
                )
            self.filters.Detrend_applied = True
            filters_applied += "Detrend: scipy adaptive  "
        elif self.filters.Detrend_type == "None" or self.filters.Detrend_type == None:
            pass
        else:
            raise ValueError(
                f"minis_methods_common: detrending filter type not known: got {self.filters.Detrend_type:s}"
            )

        #
        # 4. Perform LPF and HPF filtering
        #
        if (
            (self.filters.LPF_frequency is not None)
            and isinstance(self.filters.LPF_frequency, float)
            and self.filters.enabled
        ):
            for itrace in range(data.shape[0]):
                data[itrace] = self.LPFData(data[itrace])
            filters_applied += f"LPF={self.filters.LPF_frequency:.1f} "

        if (
            (self.filters.HPF_frequency is not None)
            and isinstance(self.filters.HPF_frequency, float)
            and self.filters.enabled
        ):
            for itrace in range(data.shape[0]):
                data[itrace] = self.HPFData(data[itrace])
            filters_applied += f"HPF={self.filters.HPF_frequency:.1f} "
        #
        # 3. Apply notch filtering to remove periodic noise (60 Hz + harmonics, and some other junk in the system)
        #
        if (
            (self.filters.Notch_frequencies is not None)
            and (
                isinstance(self.filters.Notch_frequencies, list)
                or isinstance(self.filters.Notch_frequencies, np.ndarray)
            )
            and self.filters_enabled
        ):
            if self.verbose:
                CP.cprint("r", "Comb filter notch")
            for itrace in range(data.shape[0]):
                data[itrace] = self.NotchFilterComb(data[itrace])
            filters_applied += f"Comb Notch={str(self.filters.Notch_frequencies):s} "

        self.data = data.copy()

        # f, ax = mpl.subplots(1,1)
        # for i in range(self.data.shape[0]):
        #     mpl.plot(self.timebase, self.data[i], linewidth = 0.35)
        # mpl.show()
        self.data_prepared = True
        CP.cprint("g", f"Filters applied = {filters_applied:s}")


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

    def get_data_cleaned_of_stimulus_artifacts(
        self, data: object, summary: object = None, pars: dict = None
    ):
        """
        After the traces have been analyzed, and amplitudes, peaks and onsets identified,
        we genetrate a list of events that are free of stimulus artifacts.
        This list is used to filter the trace data later.
        """
        # build array of artifact times first
        assert data.ndim == 2
        ntraces = data.shape[0]
        # set up parameters for artifact exclusion
        art_starts = []
        art_durs = []
        art_starts = [
            pars.analysis_window[1],
            pars.shutter_artifact,
        ]
        # generic artifact times and arrays
        art_durs = [2, 2 * self.dt_seconds]
        if pars.artifact_suppress:
            for si, s in enumerate(pars.stimtimes["start"]):
                if s in art_starts:
                    continue
                art_starts.append(s)
                if isinstance(pars.stimtimes["duration"], float):
                    if pd.isnull(pars.stimdur):  # allow override
                        art_starts.append(s + pars.stimtimes["duration"])
                    else:
                        art_starts.append(s + pars.stimdur)

                else:
                    if pd.isnull(pars.stimdur):
                        art_starts.append(s + pars.stimtimes["duration"][si])
                    else:
                        art_starts.append(s + pars.stimdur)
                art_durs.append(2.0 * self.dt_seconds)
                art_durs.append(2.0 * self.dt_seconds)

        ok_onsets = [[]] * ntraces
        for i in range(ntraces):
            # CP.cprint("r", f"analyzing trace: {i:d}")
            npk0 = self.select_events(
                summary.smpkindex[i],
                art_starts,
                art_durs,
                self.dt_seconds,
                mode="reject",
            )
            # npk4 = self.select_by_sign(
            #     itrace=i, npks=npk0, data=data[i], min_event=5e-12
            # )  # events must have correct sign and a minimum amplitude
            npk = list(set(npk0))
            #     set(npk0).intersection(set(npk4))
            # )  # make a list of all peaks that pass all tests (logical AND)
            #  if not self.artifact_suppress:
            #     npk = npk4  # only suppress shutter artifacts  .. exception

            if len(npk) == 0:
                ok_onsets[i] = []
            else:  # store the ok events
                ok_onsets[i] = [summary.onsets[i][n] for n in npk]
        return ok_onsets

    def summarize(self, data, order: int = 11, verbose: bool = False) -> None:
        """
        Compute peaks, smoothed peaks, and ampitudes for all found events in a
        trace or a group of traces.
        Filter out events that are less than min_event_amplitude
        and events where the charge is of the wrong sign.
        """
        CP.cprint("c", "    Summarizing data")
        i_decay_pts = int(
            2.0 * self.taus[1] / self.dt_seconds
        )  # decay window time (points) Units all seconds
        assert i_decay_pts > 5

        summary = (
            MEDC.Mini_Event_Summary()
        )  # a single summary class is created each time we are called
        ndata = len(data)
        # set up arrays : note construction to avoid "same memory but different index" problem
        summary.dt_seconds = self.dt_seconds
        summary.onsets = [[] for x in range(ndata)]
        summary.peakindices = [[] for x in range(ndata)]
        summary.smoothed_peaks = [[] for x in range(ndata)]
        summary.smpkindex = [[] for x in range(ndata)]
        summary.amplitudes = [[] for x in range(ndata)]

        avgwin = 5  # 5 point moving average window for peak detection
        mwin = int((0.50) / self.dt_seconds)
        if self.sign > 0:
            nparg = np.greater
        else:
            nparg = np.less
        self.intervals = []

        nrejected_too_small = 0
        for itrial, dataset in enumerate(data):  # each trial/trace
            if len(self.onsets[itrial]) == 0:  # original events
                continue
            acceptlist_trial = []
            ons = np.where(self.onsets[itrial] < data.shape[1])
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
                        summary.onsets[itrial].append(onset)
                        summary.peakindices[itrial].append(onset + rawpk)
                        summary.amplitudes[itrial].append(windowed_data[rawpk])
                        summary.smpkindex[itrial].append(onset + smpk)
                        summary.smoothed_peaks[itrial].append(move_avg[smpk])
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
            f"    methods common: identify_events: {nrejected_too_small:5d} events were smaller than threshold of {1e12*self.min_event_amplitude:6.1f} pA",
        )
        self.summary = summary  # keep a local copy, but only for resting.
        return summary

    def select_events(
        self,
        pkt: Union[list, np.ndarray],
        tstarts: list,  # window starts
        tdurs: list,  # window durations
        rate: float,
        mode: str = "reject",
        thr: float = 5e-12,
        data: Union[np.ndarray, None] = None,
        first_only: bool = False,
        debug: bool = False,
    ) -> list:
        """
        return indices where the input index is outside (or inside) a set of time windows.
        tstarts is a list of window starts
        twin is the duration of each window
        rate is the data sample rate (in msec...)
        pkt is the list of times to compare against.
        """

        # print('rate: ', rate)
        debug = False

        if mode in ["reject", "threshold_reject"]:
            npk = list(range(len(pkt)))  # assume all
        else:
            npk = []
        for itw, tw in enumerate(tstarts):  # and for each stimulus
            first = False
            if isinstance(tdurs, list) or isinstance(
                tdurs, np.ndarray
            ):  # either use array parallel to tstarts, or
                ttwin = tdurs[itw]
            else:
                ttwin = tdurs  # or use just a single value
            ts = int(tw / rate)
            te = ts + int(ttwin / rate)
            for k, pk in enumerate(pkt):  # over each index
                if (
                    mode == "reject" and npk[k] is None
                ):  # means we have already rejected the n'th one
                    continue
                if mode == "reject":
                    if pk >= ts and pk < te:
                        npk[k] = None
                elif (mode == "threshold_reject") and (data is not None):
                    if (pk >= ts) and (pk < te) and (np.fabs(data[k]) < thr):
                        # print("np.fabs: ", np.fabs(data[k]), thr)
                        npk[k] = None
                elif mode == "accept":
                    if debug:
                        print("accepting ?: ", ts, k, pk, te, rate)
                    if pk >= ts and pk < te and not first:
                        if debug:
                            print("    ok")
                        if k not in npk:
                            npk.append(k)
                        if first_only and not first:
                            first = True
                            break

                else:
                    raise ValueError(
                        "analyzeMapData:select_times: mode must be accept, threshold_reject, or reject; got: %s"
                        % mode
                    )
        if debug:
            print("npk: ", npk)
        npks = [
            n for n in npk if n is not None
        ]  # return the truncated list of indices into pkt
        return npks

    def select_by_sign(
        self,
        method: object,
        itrace: int,
        npks: int,
        data: np.ndarray,
        min_event: float = 5e-12,
    ) -> Union[list, np.ndarray]:
        """
        Screen events for correct sign and minimum amplitude.
        Here we use the onsets and smoothed peak to select
        for events satisfying criteria.

        Parameters
        ----------
        method : object (mini_analysis object)
            result of the mini analysis. The object must contain
            at least two lists, one of onsets and one of the smoothed peaks.
            The lists must be of the same length.

        data : array
            a 1-D array of the data to be screened. This is the entire
            trace.

        event_min : float (default 5e-12)
            The smallest size event that will be considered acceptable.
        """

        pkt = []
        if len(method.onsets) == 0 or not method.Summary.average.averaged:
            return pkt
        tb = method.timebase  # full time base
        smpks = np.array(method.Summary.smpkindex[itrace])
        # events[trial]['aveventtb']
        rate = np.mean(np.diff(tb))
        tb_event = method.Summary.average.avgeventtb  # event time base
        tpre = 0.002  # 0.1*np.max(tb0)
        tpost = np.max(tb_event) - tpre
        ipre = int(tpre / self.rate)
        ipost = int(tpost / self.rate)
        pt_fivems = int(0.0005 / self.rate)
        pk_width = int(0.0005 / self.rate / 2.0)

        for npk, jevent in enumerate(np.array(method.Summary.onsets[itrace])[npks]):
            jstart = jevent - ipre
            jpeak = method.Summary.smpkindex[itrace][npk]
            jend = jevent + ipost + 1
            evdata = data[jstart:jend].copy()
            l_expect = jend - jstart
            if evdata.shape[0] == 0 or evdata.shape[0] < l_expect:
                # print('nodata', evdata.shape[0], l_expect)
                continue
            bl = np.mean(evdata[:pt_fivems])
            evdata -= bl

            # next we make a window over which the data will be averaged to test the ampltiude
            left = jpeak - pk_width
            right = jpeak + pk_width
            left = max(0, left)
            right = min(right, len(data))
            if right - left == 0:  # peak and onset cannot be the same
                # print('r - l = 0')
                continue
            # if (self.Pars.sign < 0) and (
            #     np.mean(data[left:right]-zero) > (self.Pars.sign * min_event)
            # ):  # filter events by amplitude near peak
            #      print('data pos, sign neg', np.mean(data[left:right]))
            #      continue
            # if (self.Pars.sign >= 0) and (
            #     np.mean(data[left:right]-zero) < (self.Pars.sign * min_event)
            # ):
            #     print('data neg, sign pos', np.mean(data[left:right]))
            #     continue
            pkt.append(npk)  # build array through acceptance.
        return pkt

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
            self.summary.allevents = allevents
            # assert 1 == 0  # force trap here
        else:
            self.measured = False
            self.summary.allevents = None
        return meas

    def average_events(
        self,
        traces: list,
        data: Union[list, object, None] = None,
        summary: object = None,  # dataclass Summary
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
        data : expect 2d list matching the eventlist.

        """
        if data is None:
            raise ValueError(
                "minis_methods_common.average_events requires access to the original data array"
            )
        assert summary is not None
        if not summary.onsets:
            raise ValueError("No onsets identified")
        CP.cprint("c", "mmc: average_events")
        self.data_for_average = data
        self.traces_for_average = traces
        summary.average.averaged = False
        npre = int(self.template_pre_time / self.dt_seconds)  # points for the pre time
        npost = int(self.template_tmax / self.dt_seconds)
        avgnpts = npre + npost  # points for the average
        avgeventtb = np.arange(avgnpts) * self.dt_seconds
        n_events = sum([len(events) for events in summary.onsets])
        allevents = {}  # np.zeros((n_events, avgnpts))
        allevents_onsets = {}
        allevents_baseline = {}
        event_onset_times = {}

        all_event_indices = []
        incomplete_event_list = []
        artifact_event_list = []
        overlapping_event_list = []
        wrong_charge_sign_event_list = []
        clean_event_list = []

        CP.cprint(
            "y", "methods common: average_events: Categorize then average clean events"
        )
        print("npre: ", npre, " npost: ", npost)
        # tag traces and get a clean set for averaging

        overlap_dur = 0.005
        overlap_window = int(overlap_dur / summary.dt_seconds)  # 5 msec window
        # print(f"overlap_window ({overlap_dur:f} sec): {overlap_window:d}")
        # print("npre: ", npre, "template: ", self.template_pre_time, "npost: ", npost)
        for itrace in traces:
            for j, event_onset in enumerate(summary.onsets[itrace]):
                thisev = (itrace, j)
                event_start = event_onset - npre # define cut-out region re: onset point of event
                event_end = event_onset + npost
                if (event_end >= data[itrace].shape[0]) or (event_start < 0):
                    incomplete_event_list.append(thisev)
                    continue
                else:
                    allevents[thisev] = data[
                        itrace, event_start : event_end
                    ].copy()  # save event. Tag for rejection/inclusion later
                    all_event_indices.append(thisev)
                    allevents_onsets[thisev] = event_onset
                    if npre > 0:
                        allevents_baseline[thisev] = np.mean(allevents[thisev][0:npre])
                    else:
                        allevents_baseline[thisev] = 0.
                    # allevents[thisev] -= allevents_baseline[thisev]
                    event_onset_times[thisev] = event_onset * self.dt_seconds # keep track of where event came from

                # Test for sign of the charge of the event
                if npre > 0:
                    baseline =  np.nanmean(allevents[thisev][0:event_onset-event_start])
                else:
                    baseline = 0.0
                td = allevents[thisev][event_onset + int(0.8 * float(npost)) : event_end]
                if len(td) > 0 and (
                    self.sign
                    * np.mean(
                        td
                        # - baseline
                    )
                    < -5e-12
                ):
                    # CP.cprint("y", f"        trace: {itrace:d}, event: {j:d} has wrong charge")
                    wrong_charge_sign_event_list.append(thisev)

                # test for overlap with next event
                if (
                    j < len(summary.onsets[itrace]) - 1
                ):  # check for next event and reject if in the window
                    if (summary.onsets[itrace][j + 1] - event_onset) < overlap_window:
                        # CP.cprint("y", f"        trace: {itrace:d}, event: {j:d} has overlaps")
                        # print("      ", summary.onsets[itrace][j+1], event_onset, overlap_window)
                        overlapping_event_list.append((itrace, j + 1))

        tarnished_events = list(
            set(wrong_charge_sign_event_list).union(
                set(tuple(incomplete_event_list)), set(tuple(overlapping_event_list))
            )
        )

        print("    # Included Events found: ", len(all_event_indices))
        print("    # Incomplete Events found: ", len(incomplete_event_list))
        print("    # Overlapping Events found: ", len(overlapping_event_list))
        print("    # Wrong chage Events found: ", len(wrong_charge_sign_event_list))
        print("    # Tarnished events: ", len(tarnished_events))
        # get the clean events (non overlapping, correct charge, complete in trace)
        clean_event_list = tuple(
            [ev for ev in all_event_indices if ev not in tarnished_events]
        )
        wrong_event_list = tuple(
            [ev for ev in all_event_indices if ev in wrong_charge_sign_event_list]
        )
        clean_event_traces = np.array([allevents[ev] for ev in clean_event_list])
        clean_event_onsets = np.array([allevents_onsets[ev] for ev in clean_event_list])
        wrong_event_traces = np.array([allevents[ev] for ev in wrong_event_list])
        overlapping_event_traces = np.array([allevents[ev] for ev in overlapping_event_list])
        
        if self.verbose:
            f, ax = mpl.subplots(2, 1)
            tx = np.array(range(clean_event_traces.shape[1]))*summary.dt_seconds
            for i, ev in enumerate(clean_event_list):
                ax[0].plot(tx+(clean_event_onsets[i]-npre)*summary.dt_seconds, clean_event_traces[i], linewidth=0.35, color='m')
                ax[0].plot(event_onset_times[ev], data[ev[0]][summary.onsets[ev[0]][ev[1]]], 'ro', markersize=3)
            for i, ev in enumerate(overlapping_event_list):
                ax[0].plot(tx+(allevents_onsets[ev]-npre)*summary.dt_seconds, overlapping_event_traces[i], linewidth=0.35, color='c')
                ax[0].plot(event_onset_times[ev], data[ev[0]][summary.onsets[ev[0]][ev[1]]], 'ko', markersize=3)

            mpl.show()
        
        print("    # Clean event array: ", clean_event_traces.shape[0], " events found")

        # get all events that are within the trace, whether overlappingn or charge is wrong
        event_indices = [i for i in all_event_indices if i not in incomplete_event_list]
        # allevents = allevents[event_indices]
        # np.array([x for i, x in enumerate(allevents) if not all(np.isnan(allevents[i,:]))])

        if len(incomplete_event_list) > 0:
            CP.cprint(
                "y",
                f"    {len(incomplete_event_list):d} event(s) excluded from average because they were too close to the end of the trace\n",
            )
        if len(overlapping_event_list) > 0:
            CP.cprint(
                "y",
                f"    {len(overlapping_event_list):d} event(s) excluded from average because they overlapped in the window with another event\n",
            )
        if len(wrong_charge_sign_event_list) > 0:
            CP.cprint(
                "y",
                f"    {len(wrong_charge_sign_event_list):d} event(s) excluded the sign of the charge was wrong\n",
            )
        CP.cprint("m", f"# clean event traces: {len(clean_event_list):d}")
        summary.allevents = allevents
        summary.all_event_indices = all_event_indices
        summary.isolated_event_trace_list = clean_event_list
        clean_event_traces = []
        for ev in clean_event_list:
            if ev in allevents.keys():
                clean_event_traces.append(allevents[ev])
        clean_event_traces = np.array(clean_event_traces)
        # generate the average and some stats on the "clean" events:
        if len(clean_event_list) > 0:
            if self.sign < 0:
                evamps = self.sign * np.nanmin(clean_event_traces, axis=1)
            else:
                evamps = self.sign * np.nanmax(clean_event_traces, axis=1)
            ev25 = np.nanpercentile(
                evamps,
                q=25,
            )  #  method="median_unbiased")
            ev75 = np.nanpercentile(
                evamps,
                q=75,
            )  #  method="median_unbiased")

            summary.average.avgevent = np.nanmean(clean_event_traces, axis=0)
            # print("Clean event traces: ", clean_event_traces)
            # print(summary.average.avgevent)
            # print(np.min(summary.average.avgevent), np.max(summary.average.avgevent), summary.average.avgevent.shape)
            summary.average.stdevent = np.nanstd(clean_event_traces, axis=0)
            summary.average.averaged = True
            summary.average.avgeventtb = avgeventtb
            summary.average.avgnpts = avgnpts
            summary.average.Nevents = len(clean_event_traces)
            summary.average.avgevent25 = np.nanmean(
                clean_event_traces[evamps < ev25], axis=0
            )
            summary.average.avgevent75 = np.nanmean(
                clean_event_traces[evamps > ev75], axis=0
            )
            summary.isolated_event_trace_list = clean_event_list
        else:
            summary.average.averaged = False
            summary.average.Nevents = 0
            summary.average.avgeventtb = []
            summary.average.avgnpts = None
            summary.isolated_event_trace_list = []
            summary.average.avgevent = None
            summary.average.stdevent = None
            summary.average.avgevent = []
            summary.average.avgevent25 = []
            summary.average.avgevent75 = []

        if summary.average.averaged:
            CP.cprint("m", "    Fitting averaged event")
            self.fit_average_event(
                tb=summary.average.avgeventtb,
                avgevent=summary.average.avgevent,
                initdelay=self.template_pre_time,
                debug=False,
            )
            # print("fit: ", self.fitted_tau1, self.fitted_tau2, self.amplitude, self.amplitude2, self.avg_fiterr)
            summary.average.fitted_tau1 = self.fitted_tau1
            summary.average.fitted_tau2 = self.fitted_tau2
            summary.average.fitted_tau3 = self.fitted_tau3
            summary.average.fitted_tau4 = self.fitted_tau4
            summary.average.fitted_tau_ratio = self.fitted_tau_ratio
            summary.average.best_fit = self.avg_best_fit
            summary.average.amplitude = self.amplitude
            summary.average.amplitude2 = self.amplitude2
            summary.average.avg_fiterr = self.avg_fiterr
            summary.average.risetenninety = self.risetenninety
            summary.average.t10 = self.t10
            summary.average.t90 = self.t90
            summary.average.risepower = self.risepower
            summary.average.decaythirtyseven = self.decaythirtyseven
        else:
            CP.cprint(
                "r", "    average_events: **** No events found that meet criteria ****"
            )
            return
        # for testing, plot out the clean events
        # fig, ax = mpl.subplots(1,2)
        # for itrace in traces:
        #     for j, onset in enumerate(summary.onsets[itrace]):
        #         ev = (itrace, j)
        #         if ev in self.summary.isolated_event_trace_list:
        #             ax[0].plot(avgeventtb,
        #                 self.summary.allevents[ev], # -self.summary.allevents[ev][0],
        #                 'b-', lw=0.5, )
        # # rejected:
        # for itrace in traces:
        #     for j, onset in enumerate(summary.onsets[itrace]):
        #         ev = (itrace, j)
        #         if ev not in self.summary.isolated_event_trace_list and ev in self.summary.allevents.keys():
        #             ax[1].plot(avgeventtb,
        #                 self.summary.allevents[ev], # -self.summary.allevents[ev][0],
        #                 'r-', lw=0.5, )

        # # mpl.plot(avgeventtb, self.summary.average.avgevent, 'k-', lw=3)
        # mpl.show()
        # CP.cprint("r", f"Isolated event tr list: {str(summary.isolated_event_trace_list):s}")
        return summary

    def average_events_subset(
        self, data: np.ndarray, eventlist: list = None, minisummary: object = None
    ) -> tuple:
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
        npost = int(self.template_tmax / self.dt_seconds)
        npre = int(self.template_pre_time / self.dt_seconds)  # points for the pre time
        avgnpts = int((self.template_pre_time + self.template_tmax) / self.dt_seconds)  # points for the average
        avgeventtb = np.arange(avgnpts) * self.dt_seconds
        n_events = sum([len(events) for events in minisummary.onsets])
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

    def measure_psc_shape(self, ave):
        move_avg, n = self.moving_average(  # apply 3 point moving average
            ave,
            n=min(3, len(ave)),
        )
        move_avg = ave
        ipk = np.argmax(move_avg)
        pk = move_avg[ipk]
        p10 = 0.20 * pk
        p90 = 0.80 * pk
        p37 = 0.37 * pk
        print("ipk: ", ipk, " pk: ", pk, " p10, p90, p37: ", p10, p90, p37)
        i10 = np.nonzero(move_avg[:ipk] <= p10)[0]
        print(move_avg[:ipk]*1e12)
        print(i10)
        if len(i10) == 0:
            self.risetenninety = np.nan
            self.t10 = np.nan
            self.t90 = np.nan
        else:
            i10 = i10[-1]  # get the last point where this condition was met
            ix10 = np.interp(p10, move_avg[i10 : i10 + 2], [0, self.dt_seconds])
            i90 = np.where(move_avg[:ipk] <= p90)[0][-1]
            ix90 = np.interp(p90, move_avg[i90 - 1 : i90 + 1], [0, self.dt_seconds])
            t10 = ix10 + i10 * self.dt_seconds
            t90 = ix90 + (i90 - 1) * self.dt_seconds
            # try:
            #     i90 = i90  # get the first point where this condition was met
            # except:
            #     CP.cprint("r", "**** Error in fit_average computing 10-90 RT")
            #     print(move_avg[:ipk], p90)
            #     print(t90, ix90, ix90 + ipk - 2)
            #     print(t10, ix10, ix10 + i10)
            #     exit()
            self.risetenninety = t90 - t10
            self.t10 = t10
            self.t90 = t90

        i37 = np.nonzero(move_avg[ipk:] >= p37)[
            0
        ]  # find last point greater than 37% of peak
        if len(i37) == 0:
            self.decaythirtyseven = np.nan
        else:
            i37 = i37[-1]  # last point
            self.decaythirtyseven = self.dt_seconds * i37
        print(
            "Measures t10, 90, t1 init, t37: ",
            self.t10,
            self.t90,
            (self.t90 - self.t10) / 2.0,
            self.decaythirtyseven,
        )

    def fit_average_event(
        self,
        tb: np.ndarray,
        avgevent: np.ndarray,
        debug: bool = False,
        label: str = "",
        inittaus: List = [0.001, 0.005, 0.005, 0.030],
        initdelay: Union[float, None] = 0.0,
    ) -> None:
        """
        Fit the averaged event to a double exponential epsc-like function
        Operates on the AverageEvent data structure
        """
        CP.cprint("c", f"        Fitting average event, fixed_delay={initdelay:f}")

        self.risetenninety = np.nan
        self.t10 = np.nan
        self.t90 = np.nan
        self.decaythirtyseven = np.nan

        ave = self.sign * avgevent
        self.measure_psc_shape(ave)  # first get 10-90 times and t37

        tsel = 0  # use whole averaged trace
        self.tsel = tsel
        self.tau1 = (self.t90 - self.t10) / 2.0  # inittaus[0]
        self.tau2 = self.decaythirtyseven  # inittaus[1]
        self.tau3 = self.tau1 * 3.0
        self.tau4 = self.tau2 * 5
        self.tau_ratio = self.tau2 / self.tau1
        self.tau2_range = 10.0
        self.tau1_minimum_factor = 5.0
        time_past_peak = 2.5e-4
        self.fitted_tau1 = np.nan
        self.fitted_tau2 = np.nan
        self.fitted_tau3 = np.nan
        self.fitted_tau4 = np.nan

        self.fitted_tau_ratio = np.nan
        self.amplitude = np.nan
        self.amplitude2 = np.nan
        self.avg_fiterr = np.nan
        self.bfdelay = np.nan
        self.avg_best_fit = None
        self.Qtotal = np.nan

        # print("init delay: ", initdelay)
        res = self.event_fitter_lm(
            tb,
            avgevent,
            time_past_peak=time_past_peak,
            tau1=self.tau1,
            tau2=self.tau2,
            tau3=self.tau3,
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

        amp = np.max(res.best_fit)
        # print("amp, res amp: ", amp, res.values["amp"])
        self.amplitude = res.values["amp"]
        self.amplitude2 = res.values["amp2"]
        self.fitted_tau1 = res.values["tau_1"]
        self.fitted_tau2 = res.values["tau_2"]
        self.fitted_tau3 = res.values["tau_3"]
        self.fitted_tau4 = res.values["tau_4"]
        self.fitted_tau_ratio = np.nan  # res.values["tau_ratio"] #(
        # res.values["tau_2"] / res.values["tau_1"]
        # )  # res.values["tau_ratio"]
        self.bfdelay = res.values["fixed_delay"]

        self.avg_best_fit = self.sign * res.best_fit  # self.fitresult.best_fit
        # print("tb max: ", np.max(tb))
        # f, ax = mpl.subplots(1,1)
        # ax.plot(tb, avgevent)
        # ax.plot(tb, self.avg_best_fit, "r--")
        # mpl.show()
        # print(self.fitted_tau1, self.fitted_tau2)
        fiterr = np.linalg.norm(self.avg_best_fit - avgevent[self.tsel :])
        self.avg_fiterr = fiterr

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
            not self.summary.average.averaged or not self.fitted
        ):  # averaging should be done first: stores events for convenience and gives some tau estimates
            print(
                "Require fit of averaged events prior to fitting individual events",
                self.summary.average.averaged,
            )
            raise (ValueError)
        onsets = self.summary.onsets
        time_past_peak = 0.1  # msec - time after peak to start fitting
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
        self.ev_tau_ratio = nanfill(nevents)
        self.ev_1090 = nanfill(nevents)
        self.ev_2080 = nanfill(nevents)
        self.ev_amp = nanfill(nevents)  # measured peak amplitude from the event itself
        self.ev_Qtotal = nanfill(
            nevents
        )  # measured charge of the event (integral of current * dt)
        self.ev_Q_end = nanfill(
            nevents
        )  # measured charge of last half of event (integrl of current * dt)
        self.fiterr = nanfill(nevents)
        self.bfdelay = nanfill(nevents)
        self.best_fit = (
            np.zeros((nevents, self.summary.average.avgeventtb.shape[0])) * np.nan
        )
        self.best_decay_fit = (
            np.zeros((nevents, self.summary.average.avgeventtb.shape[0])) * np.nan
        )
        self.tsel = 0
        self.tau2_range = 10.0
        self.tau1_minimum_factor = 5.0

        # prescreen events
        minint = np.max(
            self.summary.average.avgeventtb
        )  # msec minimum interval between events.
        self.fitted_events = (
            []
        )  # events that can be used (may not be all events, but these are the events that were fit)
        # f, ax = mpl.subplots(1,1)
        # for j, ev_tr in enumerate(
        #     self.summary.isolated_event_trace_list
        # ):  # trace list of events
        #     ymin = min(self.summary.allevents[ev_tr])
        #     ymax = max(self.summary.allevents[ev_tr])
        #     print(j, ev_tr, "ymin: ", ymin*1e12, " ymax: ", ymax*1e12)
        #     mpl.plot(self.summary.allevents[ev_tr], linewidth=0.35)
        # mpl.show()

        # only use "well-isolated" events in time to make the fit measurements.
        print(
            f"Fitting individual events: {len(self.summary.isolated_event_trace_list):d}"
        )
        for j, ev_tr in enumerate(self.summary.isolated_event_trace_list):
            # trace list of events
            # print('onsetsj: ', len(onsets[j]))
            if not ev_tr:  # event in this trace could be outside data window, so skip
                continue
            i_tr, j_tr = ev_tr
            if onsets[i_tr][j_tr] >= len(self.timebase):
                continue
            te = self.timebase[onsets[i_tr][j_tr]]  # get current event time

            j_nan = np.count_nonzero(np.isnan(self.summary.allevents[ev_tr]))
            if j_nan > 0:
                raise ValueError(
                    f"Event array {j:d} has {j_nan:d} nan values in it, array length = {len(self.summary.allevents[ev_tr]):d} and {len(onsets[i_tr]):d} onset values"
                )

            try:
                max_event = np.max(self.sign * self.summary.allevents[ev_tr])
            except:
                CP.cprint("r", "FITTING FAILED: ")
                print("  minis_methods eventfitter")
                print("  j: ", j)
                print("  allev: ", self.summary.allevents)
                print("  len allev: ", len(self.summary.allevents), len(onsets))
                raise ValueError("  Fit failed)")
            print(f"\r    Event: {j:06d}", end="")
            res = self.event_fitter_lm(
                timebase=self.summary.average.avgeventtb,
                event=self.summary.allevents[ev_tr],
                time_past_peak=time_past_peak,
                tau1=self.tau1,
                tau2=self.tau2,
                fixed_delay=self.template_pre_time,
                label=f"Fitting event in trace: {str(ev_tr):s}  j = {j:d}",
            )
            if res is None:  # skip events that won't fit
                continue

            self.bfdelay[j] = res.values["fixed_delay"]
            self.avg_best_fit = res.best_fit
            self.ev_A_fitamp[j] = res.values["amp"]
            self.ev_tau1[j] = res.values["tau_1"]
            self.ev_tau2[j] = res.values["tau_2"]
            self.ev_tau_ratio[j] = np.nan  # res.values["tau_ratio"]

            # print(self.fitresult.params)
            # print('amp: ', self.fitresult.params['amp'].value)
            # print('tau_1: ', self.fitresult.params['tau_1'].value)
            # print('tau_2: ', self.fitresult.params['tau_2'].value)
            # print('risepower: ', self.fitresult.params['risepower'].value)
            # print('fixed_delay: ', self.fitresult.params['fixed_delay'].value)
            # print('y shape: ', np.shape(self.summary.allevents[j,:]))
            # print('x shape: ', np.shape(self.summary.average.avgeventtb))
            self.fiterr[j] = self.doubleexp_lm(
                y=self.summary.allevents[ev_tr],
                time=self.summary.average.avgeventtb,
                amp=self.fitresult.params["amp"].value,
                amp2=self.fitresult.params["amp2"].value,
                tau_1=self.fitresult.params["tau_1"].value,
                tau_2=self.fitresult.params["tau_2"].value,
                tau_3=self.fitresult.params["tau_3"].value,
                tau_4=self.fitresult.params["tau_4"].value,
                # tau_ratio=self.fitresult.params["tau_ratio"].value,
                # self.sign * self.summary.allevents[j, :],
                risepower=self.fitresult.params["risepower"].value,
                fixed_delay=self.fitresult.params[
                    "fixed_delay"
                ].value,  # self.bfdelay[j],
            )
            self.best_fit[j] = res.best_fit
            # self.best_decay_fit[j] = self.decay_fit  # from event_fitter
            self.ev_fitamp[j] = np.max(self.best_fit[j])
            self.ev_Qtotal[j] = self.dt_seconds * np.sum(
                self.sign * self.summary.allevents[ev_tr]
            )
            last_half = int(self.summary.allevents[ev_tr].shape[0] / 2)
            self.ev_Q_end[j] = self.dt_seconds * np.sum(
                self.summary.allevents[ev_tr][last_half:]
            )
            self.ev_amp[j] = np.max(self.sign * self.summary.allevents[ev_tr])
            self.fitted_events.append(j)
        print()
        self.individual_event_screen(
            fit_err_limit=2000.0, tau2_range=10.0, verbose=False
        )
        self.individual_events = True  # we did this step

    @staticmethod
    def doubleexp_lm(
        time: np.ndarray,
        amp: float,
        amp2: float,
        tau_1: float,
        tau_2: float,
        tau_3: float,
        tau_4: float,
        # tau_ratio: float=1.0,
        risepower: float = 1.0,
        fixed_delay: float = 0.0,
        y: np.ndarray = None,
    ) -> np.ndarray:
        """
        Calculate a double exponential EPSC-like waveform, with 2 double functions added together.
        The rising phase of each
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
        tm[ix:] += (amp2 * (1 - np.exp(-tx / tau_3)) ** risepower) * np.exp(-tx / tau_4)
        # this gets constrained
        if y is not None:  # return error - single value
            tm = np.sqrt(np.sum((tm - y) * (tm - y)))
        return tm

    @staticmethod
    def singleexp_lm(
        time: np.ndarray,
        amp: float,
        amp2: float,
        tau_1: float,
        tau_2: float,
        # tau_ratio: float=1.0,
        risepower: float = 1.0,
        fixed_delay: float = 0.0,
        y: np.ndarray = None,
    ) -> np.ndarray:
        """
        Calculate a double exponential EPSC-like waveform with rise and fall. The rising phase
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
        # this gets constrained
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
        tau3: float = None,
        tau4: float = None,
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
        # dexpmodel = lmfit.Model(self.doubleexp_lm)
        dexpmodel = lmfit.Model(self.singleexp_lm)
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
            if tau2 <= 1.5 * tau1:
                tau2 = 5 * tau1  # move it further out

        amp = event[peak_pos]
        # print("initial tau1: ", tau1, "tau2: ", tau2)

        if self.datatype in ["V", "VC"]:
            tau1min = tau1 / 4.0
            tau2min = tau2 / 2.0
            tau1max = tau1 * 2.0
            tau2max = tau2 * 2.0

            if tau1min < 2e-5:
                tau1min = 2e-5
            params["amp"] = lmfit.Parameter(
                name="amp",
                value=25.0e-12,
                min=0e-9,
                max=50e-9,
                vary=True,
            )
            params["amp2"] = lmfit.Parameter(
                name="amp2",
                value=25.0e-12,
                min=0e-9,
                max=50e-9,
                vary=True,
            )
            tau2min = 0.2e-3
            tau3min = 5.0e-3
        elif self.datatype in ["I", "IC"]:
            tau1min = tau1 / 10.0
            if tau1min < 1e-4:
                tau1min = 1e-4
            # params["amp"] = lmfit.Parameter(
            #     name="amp", value=1.0e-3, min=0.0, max=50e-3, vary=True
            # )
            params["amp"] = lmfit.Parameter(
                name="amp",
                value=amp,
                min=-5 * amp,
                max=5 * amp,
                vary=True,
            )
            params["amp2"] = lmfit.Parameter(
                name="amp2",
                value=amp,
                min=-5 * amp,
                max=5 * amp,
                vary=True,
            )
        else:
            raise ValueError("Data type must be VC or IC: got", self.datatype)
        params["tau_1"] = lmfit.Parameter(
            name="tau_1",
            value=tau1,
            min=tau1min,
            max=tau1max,  # tau1 * tau1_maxfac,
            vary=True,
        )
        # params["tau_ratio"] = lmfit.Parameter(
        #     name="tau_ratio",
        #     value = 1.5,
        #     min=1.2,
        #     max=50.,
        #     vary=True,
        # )
        # params["tau_2"] = lmfit.Parameter(
        #     name="tau_2",
        #     expr = "tau_1*tau_ratio",
        # )
        params["tau_2"] = lmfit.Parameter(
            name="tau_2",
            value=tau2,
            min=tau2min,
            max=tau2max,  # tau1 * tau1_maxfac,
            vary=True,
        )
        params["tau_3"] = lmfit.Parameter(
            name="tau_3",
            value=tau3,
            min=tau1min,
            max=50e-3,  # tau1 * tau1_maxfac,
            vary=True,
        )
        params["tau_4"] = lmfit.Parameter(
            name="tau_4",
            value=tau3,
            min=tau3min,
            max=50e-3,  # tau1 * tau1_maxfac,
            vary=True,
        )
        params["fixed_delay"] = lmfit.Parameter(
            name="fixed_delay",
            value=fixed_delay,
            vary=False,
            min=fixed_delay,
            max=fixed_delay + 1e-3,
        )
        params["risepower"] = lmfit.Parameter(
            name="risepower", value=self.risepower, vary=False
        )
        self.fitresult = dexpmodel.fit(
            evfit,
            params,
            nan_policy="raise",
            time=timebase,
            max_nfev=3000,
            method="nelder",
        )
        # now repeat with 2 exponentials.
        dexpmodel = lmfit.Model(self.doubleexp_lm)
        params["tau_1"].value = self.fitresult.best_values["tau_1"]
        params["tau_2"].value = self.fitresult.best_values["tau_2"]
        params["amp"].value = self.fitresult.best_values["amp"]
        self.fitresult = dexpmodel.fit(
            evfit,
            params,
            nan_policy="raise",
            time=timebase,
            max_nfev=3000,
            method="nelder",
        )

        self.peak_val = maxev
        self.evfit = self.fitresult.best_fit  # handy right out of the result

        debug = False
        if debug:
            import matplotlib.pyplot as mpl

            mpl.figure()
            mpl.plot(timebase, evfit, "k-")
            mpl.plot(timebase, self.fitresult.best_fit, "r--")
            # print(self.fitresult.fit_report())
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
        verbose = False
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
                            print(f"  passed > min amplitude. ", end="")
                        if self.ev_tau1[i] > self.tau1 / self.tau1_minimum_factor:
                            if verbose:
                                print(
                                    f"  tau1 > {self.tau1/self.tau1_minimum_factor:.3f}, ** all passed",
                                    end="",
                                )
                            self.events_ok.append(i)
                        else:
                            if verbose:
                                print(f"  failed tau min factor")
                    else:
                        if verbose:
                            print(
                                f"  failed, < min amplitude. {1e12*self.ev_fitamp[i]:f} < {1e12*self.min_event_amplitude:f} ",
                                end="",
                            )

            if verbose:
                print()
        if failed_charge > 0:
            CP.cprint("y", f"\n    {failed_charge:d} events failed charge screening")

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
            return
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
            ev_bl = np.mean(self.summary.allevents[i, 0:5])
            P.axdict["E"].plot(
                self.summary.average.avgeventtb,
                self.summary.allevents[i] - ev_bl,
                "b-",
                linewidth=0.75,
            )
            # P.axdict['E'].plot()
            P.axdict["F"].plot(
                self.summary.average.avgeventtb,
                self.summary.allevents[i] - ev_bl,
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
            ev_bl = np.mean(self.summary.allevents[i, 0:5])
            P2.axdict["A"].plot(
                self.summary.average.avgeventtb,
                self.summary.allevents[i] + offset - ev_bl,
                "k-",
                linewidth=0.35,
            )
            P2.axdict["A"].plot(
                self.summary.average.avgeventtb,
                self.best_fit[i] + offset,
                "c--",
                linewidth=0.3,
            )
            # P2.axdict["A"].plot(
            #     self.summary.average.avgeventtb,
            #     self.sign * self.best_decay_fit[i] + offset,
            #     "r--",
            #     linewidth=0.3,
            # )
            P3.axdict[idx[k]].plot(
                self.summary.average.avgeventtb,
                self.summary.allevents[i] + offset2,
                "k--",
                linewidth=0.3,
            )
            P3.axdict[idx[k]].plot(
                self.summary.average.avgeventtb,
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
            labelposition=(-1.05, 0.95),
        )
        self.P = P

        ax = P.axarr
        ax = ax.ravel()
        PH.nice_plot(ax)
        for i in range(1):
            ax[i].sharex(ax[i + 1])
        # raw traces, marked with onsets and peaks
        ax[0].set_ylabel("I (pA)")
        ax[0].set_xlabel("T (s)")
        ax[1].set_ylabel("Deconvolution")
        ax[1].set_xlabel("T (s)")
        ax[2].set_ylabel("Averaged I (pA)")
        ax[2].set_xlabel("T (ms)")
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
        label1 = "Data"

        ax[0].plot(tb, scf * data, "k-", linewidth=0.5, label=label1)  # original data
        label2 = "Onsets"
        ax[0].plot(
            tb[self.onsets[i]],
            scf * data[self.onsets[i]],
            onset_marks[index],
            markersize=5,
            markerfacecolor=(1, 1, 0, 0.8),
            label=label2,
        )

        if len(self.onsets[i]) is not None:
            if i == 0:
                label3 = "Smoothed Peaks"
            else:
                label3 = ""
            ax[0].plot(
                tb[self.summary.smpkindex[i]],
                scf * np.array(self.summary.smoothed_peaks[i]),
                peak_marks[index],
                label=label3,
            )
        if markersonly:
            return

        # deconvolution trace, peaks marked (using onsets), plus threshold)
        if i == 0:
            label4 = "Deconvolution"
        else:
            label4 = ""
        ax[1].plot(tb[: self.Criterion[i].shape[0]], self.Criterion[i], label=label4)

        if i == 0:
            label5 = "Threshold ({0:4.2f}) SD".format(self.sdthr)
        else:
            label5 = ""
        ax[1].plot(
            [tb[0], tb[-1]],
            [self.sdthr, self.sdthr],
            "r--",
            linewidth=0.75,
            label=label5,
        )
        if i == 0:
            label6 = "Deconv. Peaks"
        else:
            label6 = ""
        ax[1].plot(
            tb[self.onsets[i]] - self.idelay,
            self.Criterion[i][self.onsets[i]],
            onset_marks[index],
            label=label6,
        )
        if events is not None:  # original events
            ax[1].plot(
                tb[: self.Criterion[i].shape[0]][events],
                self.Criterion[i][events],
                "ro",
                markersize=3.0,
            )

        # averaged events, convolution template, and fit
        if self.summary.average.averaged:
            if i == 0:
                nev = sum([len(x) for x in self.onsets])
                label7 = f"Average Event (N={nev:d})"
            else:
                label7 = ""
            evlen = len(self.summary.average.avgevent)
            ax[2].plot(
                self.summary.average.avgeventtb[:evlen] * 1e3,
                scf * self.summary.average.avgevent,
                "k-",
                label=label7,
            )
            maxa = np.max(self.sign * self.summary.average.avgevent)
            if self.template is not None:
                maxl = int(
                    np.min([len(self.template), len(self.summary.average.avgeventtb)])
                )
                temp_tb = np.arange(0, maxl * self.dt_seconds, self.dt_seconds)
                if i == 0:
                    label8 = "Template"
                else:
                    label8 = ""
                ax[2].plot(
                    self.summary.average.avgeventtb[:maxl] * 1e3 + self.bfdelay,
                    scf * self.sign * self.template[:maxl] * maxa / self.template_amax,
                    "m-",
                    label=label8,
                )

            sf = 1.0
            tau1 = np.power(
                10, (1.0 / self.risepower) * np.log10(self.tau1 * sf)
            )  # correct for rise power
            tau2 = self.tau2 * sf

            if i == 0:
                label9 = "Best Fit:\nRise Power={0:.2f}\nTau1={1:.3f} ms\nTau2={2:.3f} ms\ndelay: {3:.3f} ms".format(
                    self.risepower,
                    self.tau1 * sf,
                    self.tau2 * sf,
                    self.bfdelay * sf,
                )
            else:
                label9 = ""
            ax[2].plot(
                self.summary.average.avgeventtb[: len(self.avg_best_fit)] * 1e3,
                scf * self.avg_best_fit,
                "c--",
                linewidth=2.0,
                label=label9,
            )
