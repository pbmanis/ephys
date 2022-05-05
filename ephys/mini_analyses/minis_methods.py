"""
Classes for methods that do analysis of miniature synaptic potentials

Current implementations are ClementsBekkers, AndradeJonas and zero=crossing

Test run timing:
cb: 0.175 s (with cython version of algorithm); misses overlapping events
aj: 0.028 s, plus gets overlapping events

July 2017

Note: all values are MKS (Seconds, plus Volts, Amps)
per acq4 standards... 

Each method inherits the base class from MiniAnalyses, which provides support
of post-detection analysis.

"""

import timeit
from typing import List, Union, Tuple

import ephys.tools.functions as FN  # Luke's misc. function library
import lmfit
import numba as nb
import numpy as np
import pylibrary.tools.digital_filters as dfilt
import pyximport
import scipy as sp
import scipy.signal
from ephys.mini_analyses import clembek  # cythonized... pyx file
from ephys.mini_analyses.minis_methods_common import MiniAnalyses
from pylibrary.tools.cprint import cprint
from scipy.optimize import curve_fit
import ephys.ephysanalysis.metaarray as metaarray

pyximport.install()


@nb.njit(parallel=False, cache=True)
def nb_clementsbekkers(data, template: Union[List, np.ndarray]):
    """
    cb algorithm for numba jit.
    """
    ## Prepare a bunch of arrays we'll need later
    n_template = len(template)
    # template = np.ndarray(template, dtype=np.float64)  # throws error so not needed?
    # if n_template <= 1:
    #     raise ValueError("nb_clementsbekkers: Length of template must be useful, and > 1")
    n_data = data.shape[0]
    n_dt = int(n_data - n_template)
    # if n_dt < 10:
    #     raise ValueError("nb_clementsbekkers: n_dt, n_template", n_dt, n_template)
    #
    sum_template = template.sum()
    sum_template_2 = (template * template).sum()

    data_2 = data * data
    sum_data = np.sum(data[:n_template])
    sum_data_2 = data_2[:n_template].sum()
    scale = np.zeros(n_dt)
    offset = np.zeros(n_dt)
    detcrit = np.zeros(n_dt)
    for i in range(n_dt):
        if i > 0:
            sum_data = sum_data + data[i + n_template] - data[i - 1]
            sum_data_2 = sum_data_2 + data_2[i + n_template] - data_2[i - 1]
        sum_data_template_prod = np.multiply(data[i : i + n_template], template).sum()
        scale[i] = (sum_data_template_prod - sum_data * sum_template / n_template) / (
            sum_template_2 - sum_template * sum_template / n_template
        )
        offset[i] = (sum_data - scale[i] * sum_template) / n_template
        fitted_template = template * scale[i] + offset[i]
        sse = ((data[i : i + n_template] - fitted_template) ** 2).sum()
        detcrit[i] = scale[i] / np.sqrt(sse / (n_template - 1))
    return (scale, detcrit)


class ClementsBekkers(MiniAnalyses):
    """
    Implements Clements-bekkers algorithm: slides template across data,
    returns array of points indicating goodness of fit.
    Biophysical Journal,  73: 220-229,  1997.d
    We have 3 engines to use:
    numba (using a just-in-time compiler)
    cython (pre-compiled during setups
    python (slow, direct implementation)

    """

    def __init__(self):
        super().__init__()
        self.dt_seconds = None
        self.data = None
        self.template = None
        self.engine = "cython"
        self.method = "cb"
        self.onsets = None
        self.Crit = None

    def set_cb_engine(self, engine: str) -> None:
        """
        Define which detection engine to use
        cython requires compilation in advance in setup.py
        Numba does a JIT compilation (see routine above)
        """
        if engine in ["numba", "cython", "python"]:
            self.engine = engine
        else:
            raise ValueError(
                f"CB detection engine must be one of python, numba or cython. Got{str(engine):s}"
            )

    def clements_bekkers(self, data: np.ndarray) -> None:
        """
        External call point for all engines once parameters are
        set up.

        Parameters
        ----------
        data : np.array (no default)
            1D data array

        """
        starttime = timeit.default_timer()
        if self.template is None:
            self._make_template()

        ## Strip out meta-data for faster computation
        D = self.sign * data.view(np.ndarray)
        if self.template is None:
            self._make_template()
        T = self.template.view(np.ndarray)

        self.timebase = np.arange(0.0, data.shape[0] * self.dt_seconds, self.dt_seconds)
        if self.engine == "numba":
            self.Scale, self.Crit = nb_clementsbekkers(D, T)
            # print('numba')
        elif self.engine == "cython":
            self.Scale, self.Crit = self.clements_bekkers_cython(D, T)
            # print('cython')
        elif self.engine == "python":
            self.Scale, self.Crit = self.clements_bekkers_python(D, T)

        else:
            raise ValueError(
                'Clements_Bekkers: computation engine unknown (%s); must be "python", "numba" or "cython"'
                % self.engine
            )
        endtime = timeit.default_timer() - starttime
        self.runtime = endtime
        self.Crit = self.sign * self.Crit  # assure that crit is positive

    def clements_bekkers_numba(
        self,
        data: np.ndarray,
        T: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Wrapper for numba implementation
        """
        # print('Template len: ', self.template.shape, 'data: ', data.shape, 'max(t): ', np.max(self.timebase))
        if np.std(data) < 5e-12:  # no real data to do - so just return zeros.
            DC = np.zeros(self.template.shape[0])
            Scale = np.zeros(self.template.shape[0])
            Crit = np.zeros(self.template.shape[0])
        else:
            DC, Scale, Crit = nb_clementsbekkers(data, T)
        return DC, Scale, Crit

    def clements_bekkers_cython(
        self,
        data: np.ndarray,
        T: np.ndarray,
    ) -> None:
        # version using cythonized clembek (imported above)
        starttime = timeit.default_timer()
        D = data
        Crit = np.zeros_like(D)
        Scale = np.zeros_like(D)
        DetCrit = np.zeros_like(D)
        pkl = np.zeros(1000000)
        evl = np.zeros(1000000)
        nout = 0
        nt = T.shape[0]
        nd = D.shape[0]
        clembek.clembek(
            D,
            T,
            self.threshold,
            Crit,
            Scale,
            DetCrit,
            pkl,
            evl,
            nout,
            self.sign,
            nt,
            nd,
        )
        endtime = timeit.default_timer() - starttime
        self.runtime = endtime
        return Scale, Crit

    def _rollingSum(self, data, n):
        n = int(n)
        d1 = data.copy()
        d1[1:] = d1[1:] + d1[:-1]
        d2 = np.empty(len(d1) - n + 1, dtype=data.dtype)
        d2[0] = d1[n - 1]  # copy first point
        d2[1:] = d1[n:] - d1[:-n]  # subtract
        return d2

    def clements_bekkers_python(
        self, D: np.ndarray, T: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Implements Clements-bekkers algorithm: slides template across data,
        returns array of points indicating goodness of fit.
        Biophysical Journal, 73: 220-229, 1997.

        Campagnola's version...
        """

        starttime = timeit.default_timer()
        # Strip out meta-data for faster computation
        NDATA = len(D)
        # Prepare a bunch of arrays we'll need later
        N = len(T)
        sumT = T.sum()
        sumT2 = (T**2.0).sum()
        sumD = self._rollingSum(D, N)
        sumD2 = self._rollingSum(D**2.0, N)
        sumTD = scipy.signal.correlate(D, T, mode="valid", method="direct")

        # sumTD2 = np.zeros_like(sumD)
        # for i in range(len(D)-N+1):
        #     sumTD2[i] = np.multiply(D[i : i + N], T).sum()
        # print(np.mean(sumTD-sumTD2))
        # compute scale factor, offset at each location:
        ## compute scale factor, offset at each location:
        scale = (sumTD - sumT * sumD / N) / (sumT2 - sumT * sumT / N)
        offset = (sumD - scale * sumT) / N

        ## compute SSE at every location
        SSE = (
            sumD2
            + scale**2 * sumT2
            + N * offset**2
            - 2 * (scale * sumTD + offset * sumD - scale * offset * sumT)
        )
        ## finally, compute error and detection criterion
        stderror = np.sqrt(SSE / (N - 1))
        DetCrit = scale / stderror
        endtime = timeit.default_timer() - starttime
        self.runtime = endtime
        # import matplotlib.pyplot as mpl
        #        mpl.plot(DetCrit)
        #        mpl.show()
        #        exit()
        return scale, DetCrit

    def cbTemplateMatch(
        self,
        data: np.ndarray,
        itrace: int = 0,
        lpf: Union[float, None] = None,
    ) -> None:
        assert data.ndim == 1
        self.starttime = timeit.default_timer()
        self.prepare_data(data)  # also does timebase
        self.clements_bekkers(self.data)  # flip data sign if necessary
        if self.Crit.ndim > 1:
            self.Crit = self.Crit.squeeze()
        self.Criterion[itrace] = self.Crit.copy()
        # print('criterion trace: min/max/sd: ', itrace, np.min(self.Criterion[itrace]), np.max(self.Criterion[itrace]), np.std(self.Criterion[itrace]))

    def identify_events(
        self,
        data_nostim: Union[list, np.ndarray, None] = None,
        outlier_scale: float = 10.0,
        order: int = 11,
        verbose: bool = False,
    ):
        """
        Identify events. Criterion array should be 2D:
        (trial number, criterion array)
        """
        criterion = np.array(self.Criterion)
        assert criterion.ndim == 2
        if data_nostim is not None:
            # clip to max of crit array, and be sure index array is integer, not float
            for i in range(criterion.shape[0]):
                criterion[i, :] = criterion[
                    i, [int(x) for x in data_nostim if x < criterion.shape[1]]
                ]
        # compute an SD across the entire dataset (all traces)
        # To do this remove "outliers" in a first pass
        valid_data = np.zeros_like(criterion)
        for i in range(criterion.shape[0]):
            valid_data[i, :] = self.remove_outliers(criterion[i], outlier_scale)
        sd = np.nanstd(valid_data)
        # print(f"criterion min = {np.min(criterion):.3e}, max = {np.max(criterion):.3e}, SD: {sd:.3e}")
        self.sdthr = sd * self.threshold  # set the threshold to multiple SD
        self.onsets = [None] * criterion.shape[0]
        for i in range(criterion.shape[0]):
            self.above = np.clip(criterion[i], self.sdthr, None)
            self.onsets[i] = (
                scipy.signal.argrelextrema(self.above, np.greater, order=int(order))[0]
                - 1
                + self.idelay
            )

            endtime = timeit.default_timer() - self.starttime
        self.runtime = endtime
        # self.summarize(self.data)
        endtime = timeit.default_timer() - self.starttime

        # import matplotlib.pyplot as mpl
        # for i in range(criterion.shape[0]):
        #     mpl.plot(self.timebase, criterion[i])
        #     mpl.plot(self.onsets[i]*self.dt, self.sdthr*np.ones_like(self.onsets[i]), 'ro')
        # mpl.plot([self.timebase[0], self.timebase[-1]], [self.sdthr, self.sdthr], 'r--')
        # mpl.show()
        # self.summarize(self.data)
        if verbose:
            print("CB run time: {0:.4f} s".format(endtime))


class AndradeJonas(MiniAnalyses):
    """
    Deconvolution method of Andrade/Jonas,  Biophysical Journal 2012
    Create an instance of the class (aj = AndradeJonas())
    call setup to instantiate the template and data detection sign (1 for positive, -1 for negative)
    call deconvolve to perform the deconvolution
    additional routines provide averaging and some event analysis and plotting

    """

    def __init__(self):
        self.template = None
        self.onsets = None
        self.timebase = None
        self.dt_seconds = None
        self.maxlpf = None
        self.sign = 1
        self.taus = None
        self.template_max = None
        self.idelay = 0
        self.method = "aj"
        self.Crit = None
        super().__init__()

    def deconvolve(
        self,
        data: np.ndarray,
        itrace: int = 0,
        llambda: float = 5.0,
        lpf: Union[float, None] = None,
        verbose: bool = False,
    ) -> None:
        # cprint('r', "STARTING AJ")
        assert data.ndim == 1
        self.starttime = timeit.default_timer()

        self.prepare_data(data)  # also generates a timebase
        if self.template is None:
            self._make_template()

        starttime = timeit.default_timer()

        self.data = self.data - np.mean(self.data)
        # Weiner filtering

        H = np.fft.fft(self.template)
        if H.shape[0] < self.data.shape[0]:
            H = np.hstack((H, np.zeros(self.data.shape[0] - H.shape[0])))
        self.quot = np.fft.ifft(
            np.fft.fft(self.data) * np.conj(H) / (H * np.conj(H) + llambda**2.0)
        )
        self.Crit = np.real(self.quot) * llambda
        self.Crit = self.Crit.squeeze()
        self.Criterion[itrace] = self.Crit.copy()

    def identify_events(
        self,
        data_nostim: Union[list, np.ndarray, None] = None,
        outlier_scale: float = 3.0,
        order: int = 7,
        verbose: bool = False,
    ):
        """
        Identify events. Criterion array should be 2D:
        (trial number, criterion array), thus
        we use the global statistiscs of the set of traces
        to do detection.
        """
        criterion = np.array(self.Criterion)
        assert criterion.ndim == 2
        # criterion = criterion.reshape(1, -1)  # make sure can be treated as a 2-d array
        if data_nostim is not None:
            # clip to max of crit array, and be sure index array is integer, not float
            for i in range(criterion.shape[0]):
                criterion[i, :] = criterion[
                    i, [int(x) for x in data_nostim if x < criterion.shape[1]]
                ]
        # compute an SD across the entire dataset (all traces)
        # To do this remove "outliers" in a first pass
        valid_data = np.zeros_like(criterion)
        for i in range(criterion.shape[0]):
            valid_data[i, :] = self.remove_outliers(criterion[i], outlier_scale)
        sd = np.nanstd(valid_data)

        self.sdthr = sd * self.threshold  # set the threshold to multiple SD
        self.onsets = [None] * criterion.shape[0]
        for i in range(criterion.shape[0]):
            self.above = np.clip(criterion[i], self.sdthr, None)
            self.onsets[i] = (
                scipy.signal.argrelextrema(self.above, np.greater, order=int(order))[0]
                - 1
                + self.idelay
            )
            endtime = timeit.default_timer() - self.starttime
        self.runtime = endtime
        endtime = timeit.default_timer() - self.starttime
        if verbose:
            print("AJ run time: {0:.4f} s".format(endtime))


class RSDeconvolve(MiniAnalyses):
    """Event finder using Richardson Silberberg Method, J. Neurophysiol. 2008"""

    def __init__(self):
        self.template = None
        self.onsets = None
        self.timebase = None
        self.dt_seconds = None
        self.sign = 1
        self.taus = None
        self.template_max = None
        self.idelay = 0
        self.threshold = 2.0
        self.method = "rs"
        super().__init__()

    def deconvolve(
        self,
        data: np.ndarray,
        itrace: int = 0,
        data_nostim: Union[list, np.ndarray, None] = None,
        verbose: bool = False,
    ) -> None:
        self.starttime = timeit.default_timer()
        self.prepare_data(data)  # windowing, filtering and timebase
        # if data_nostim is None:
        #     data_nostim = [range(self.Crit.shape[0])]  # whole trace, otherwise remove stimuli
        # else:  # clip to max of crit array, and be sure index array is integer, not float
        #     data_nostim = [int(x) for x in data_nostim if x < self.Crit.shape[0]]

        # print('RS Tau: ', self.taus[1], self.dt)
        self.Crit = self.expDeconvolve(
            self.sign * self.data,
            tau=self.taus[1],
            dt=self.dt_seconds,  # use decay value for deconvolve tau
        )
        self.Crit = self.Crit.squeeze()
        self.Criterion[itrace] = self.Crit

    def expDeconvolve(self, data, tau, dt=None):
        assert dt is not None
        # if (hasattr(data, 'implements') and data.implements('MetaArray')):
        #     dt = data.xvals(0)[1] - data.xvals(0)[0]
        # if dt is None:
        #     dt = 1
        # d = data[:-1] + (tau / dt) * (data[1:] - data[:-1])
        #  return d

        wlen = int(tau / dt)
        if wlen % 2 == 0:
            wlen += 1

        dVdt = np.gradient(
            data, dt
        )  # sp.signal.savgol_filter(data, window_length=wlen, polyorder=3))
        # d = (tau/dt) * tau *dVdt + data
        d = tau * dVdt + data
        return d

    def expReconvolve(data, tau=None, dt=None):
        if hasattr(data, "implements") and data.implements("MetaArray"):
            if dt is None:
                dt = data.xvals(0)[1] - data.xvals(0)[0]
            if tau is None:
                tau = data._info[-1].get("expDeconvolveTau", None)
        if dt is None:
            dt = 1
        if tau is None:
            raise Exception("Must specify tau.")
        # x(k+1) = x(k) + dt * (f(k) - x(k)) / tau
        # OR: x[k+1] = (1-dt/tau) * x[k] + dt/tau * x[k]
        # print tau, dt
        d = np.zeros(data.shape, data.dtype)
        dtt = dt / tau
        dtti = 1.0 - dtt
        for i in range(1, len(d)):
            d[i] = dtti * d[i - 1] + dtt * data[i - 1]
        if hasattr(data, "implements") and data.implements("MetaArray"):
            info = data.infoCopy()
            # if 'values' in info[0]:
            # info[0]['values'] = info[0]['values'][:-1]
            # info[-1]['expDeconvolveTau'] = tau
            return metaarray.MetaArray(d, info=info)
        else:
            return d

    def identify_events(
        self,
        data_nostim: Union[list, np.ndarray, None] = None,
        outlier_scale: float = 3.0,
        order: int = 7,
        verbose: bool = False,
    ):
        """
        Identify events. Criterion array should be 2D:
        (trial number, criterion array), thus
        we use the global statistiscs of the set of traces
        to do detection.
        """
        criterion = np.array(self.Criterion)
        assert criterion.ndim == 2
        # criterion = criterion.reshape(1, -1)  # make sure can be treated as a 2-d array
        if data_nostim is not None:
            # clip to max of crit array, and be sure index array is integer, not float
            for i in range(criterion.shape[0]):
                criterion[i, :] = criterion[
                    i, [int(x) for x in data_nostim if x < criterion.shape[1]]
                ]
        # compute an SD across the entire dataset (all traces)
        # To do this remove "outliers" in a first pass
        valid_data = np.zeros_like(criterion)
        for i in range(criterion.shape[0]):
            valid_data[i, :] = self.remove_outliers(criterion[i], outlier_scale)
        sd = np.nanstd(valid_data)

        self.sdthr = sd * self.threshold  # set the threshold to multiple SD
        self.onsets = [None] * criterion.shape[0]
        for i in range(criterion.shape[0]):
            self.above = np.clip(criterion[i], self.sdthr, None)
            self.onsets[i] = (
                scipy.signal.argrelextrema(self.above, np.greater, order=int(order))[0]
                - 1
                + self.idelay
            )

            endtime = timeit.default_timer() - self.starttime
        self.runtime = endtime
        endtime = timeit.default_timer() - self.starttime
        if verbose:
            print("RS run time: {0:.4f} s".format(endtime))


class ZCFinder(MiniAnalyses):
    """
    Event finder using Luke's zero-crossing algorithm

    """

    def __init__(self):
        self.template = None
        self.onsets = None
        self.timebase = None
        self.dt_seconds = None
        self.sign = 1
        self.taus = None
        self.template_max = None
        self.idelay = 0
        self.threshold = 2.5  # x SD
        self.method = "zc"
        super().__init__()

    def deconvolve(
        self,
        data: np.ndarray,
        data_nostim: Union[list, np.ndarray, None] = None,
        itrace: int = 0,
        minPeak: float = 0.0,
        minSum: float = 0.0,
        minLength: int = 3,
        verbose: bool = False,
    ) -> None:

        print("Called DECONVOLVE")
        self.prepare_data(data)  # windowing, filtering and timebase
        starttime = timeit.default_timer()

        events = FN.zeroCrossingEvents(
            self.data,
            minLength=minLength,
            minPeak=minPeak,
            minSum=minSum,
            noiseThreshold=self.threshold,
            # sign=self.sign,
        )
        self.Criterion[itrace] = np.zeros_like(self.data)
        self.events = events
        print("events: ", events)
        print('len events: ', len(events))

    def identify_events(
        self,
        data_nostim: Union[list, np.ndarray, None] = None,
        outlier_scale: float = 3.0,
        verbose: bool = False,
    ):
        starttime = timeit.default_timer()
        self.sdthr = self.threshold * np.std(self.data)
        if data_nostim is not None:
            # clip to max of crit array, and be sure index array is integer, not float
            for i in range(self.Criterion.shape[0]):
                self.Criterion[i, :] = self.Criterion[
                    i, [int(x) for x in data_nostim if x < self.Criterion.shape[1]]
                ]

        print(f"ZC thr:  {self.threshold:.3f}  SDxthr: {self.sdthr:.3e}")

        print('events[0]: ', self.events[0])
        self.onsets = np.array(
            [x["index"] for x in self.events if self.sign * x["peak"] > self.sdthr]
        ).astype(int)
        print("zc onsets: ", self.onsets)
        # self.summarize(self.data)
        endtime = timeit.default_timer() - starttime
        if verbose:
            print("ZC run time: {0:.4f} s".format(endtime))
