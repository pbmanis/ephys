from __future__ import print_function

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

import numpy as np
import scipy.signal
from typing import Union, List
import timeit
import pyximport
from scipy.optimize import curve_fit
from numba import jit
import lmfit

import pylibrary.tools.digital_filters as dfilt
from pylibrary.tools.cprint import cprint
import ephys.mini_analyses.functions as FN  # Luke's misc. function library
from ephys.mini_analyses import clembek  # cythonized... pyx file
from ephys.mini_analyses.minis_methods_common import MiniAnalyses


pyximport.install()


@jit(nopython=False, parallel=False, cache=True)
def nb_clementsbekkers(data, template: Union[List, np.ndarray]):
    """
    cb algorithm for numba jit.
    """
    ## Prepare a bunch of arrays we'll need later
    n_template = len(template)
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
    Biophysical Journal,  73: 220-229,  1997.
    We have 3 engines to use:
    numba (using a just-in-time compiler)
    cython (pre-compiled during setup)
    python (slow, direct implementation)
    
    """

    def __init__(self):
        self.dt = None
        self.data = None
        self.template = None
        self.engine = "numba"
        self.method = "cb"

    def set_cb_engine(self, engine: str) -> None:
        """
        Define which detection engine to use
        cython requires compilation in advance in setup.py
        Numba does a JIT compilation (see routine above)
        """
        if engine in ["numba", "cython", "python"]:
            self.engine = engine
        else:
            raise ValueError(f"CB detection engine must be one of python, numba or cython. Got{str(engine):s}")

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
        
        self.timebase = np.arange(0.0, data.shape[0] * self.dt, self.dt)
        if self.engine == "numba":
            self.Scale, self.Crit = nb_clementsbekkers(D, T)
        elif self.engine == "cython":
            self.Scale, self.Crit = self.clements_bekkers_cython(D, T)
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
        self, data: np.ndarray, T: np.ndarray,
    ) -> (np.ndarray, np.ndarray, np.ndarray):
        """
        Wrapper for numba implementation
        """
        # print('Template len: ', self.template.shape, 'data: ', D.shape, 'max(t): ', np.max(self.timebase))
        if np.std(D) < 5e-12:  # no real data to do - so just return zeros.
            DC = np.zeros(self.template.shape[0])
            Scale = np.zeros(self.template.shape[0])
            Crit = np.zeros(self.template.shape[0])
        else:
            DC, Scale, Crit = nb_clementsbekkers(D, T)
        return DC, Scale, Crit

    def clements_bekkers_cython(self, data: np.ndarray, T: np.ndarray,) -> None:
        # version using cythonized clembek (imported above)
        starttime = timeit.default_timer()
        D = data
        Crit = np.zeros_like(D)
        Scale = np.zeros_like(D)
        DetCrit = np.zeros_like(D)
        pkl = np.zeros(100000)
        evl = np.zeros(100000)
        nout = 0
        nt = T.shape[0]
        nd = D.shape[0]
        clembek.clembek(
            D, T, self.threshold, Crit, Scale, DetCrit, pkl, evl, nout, self.sign, nt, nd
        )
        endtime = timeit.default_timer() - starttime
        self.runtime = endtime
        return Scale, Crit

    def _rollingSum(self, data, n):
        n = int(n)
        d1 = data.copy()
        d1[1:] =  d1[1:] + d1[:-1]
        d2 = np.empty(len(d1) - n + 1, dtype=data.dtype)
        d2[0] = d1[n-1]  # copy first point
        d2[1:] = d1[n:] - d1[:-n]  # subtract
        return d2
    
    def clements_bekkers_python(self, D:np.ndarray, T:np.ndarray) ->(np.ndarray, np.ndarray, np.ndarray):
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
        sumT2 =  (T**2.0).sum()
        sumD = self._rollingSum(D, N)
        sumD2 = self._rollingSum(D**2.0, N)
        sumTD = scipy.signal.correlate(D, T, mode='valid', method='direct')
        # sumTD = np.zeros_like(sumD)
#         for i in range(len(D)-N+1):
#        sumTD[i] = np.multiply(D[i : i + N], T).sum()
        # compute scale factor, offset at each location:
    ## compute scale factor, offset at each location:
        scale = (sumTD - sumT * sumD /N) / (sumT2 - sumT*sumT /N)
        offset = (sumD - scale * sumT) /N
    
        ## compute SSE at every location
        SSE = sumD2 + scale**2 * sumT2 + N * offset**2 - 2 * (scale*sumTD + offset*sumD - scale*offset*sumT)
    
        ## finally, compute error and detection criterion
        stderror = np.sqrt(SSE / (N-1))
        DetCrit = scale / stderror
        endtime = timeit.default_timer() - starttime
        self.runtime = endtime
        return scale, DetCrit
        
    def cbTemplateMatch(
        self,
        data: np.ndarray,
        order: int = 7,
        lpf: Union[float, None] = None,
    ) -> None:
        self.data = self.LPFData(data, lpf)

        self.clements_bekkers(self.data)  # flip data sign if necessary
        # svwinlen = self.Crit.shape[0]  # smooth the crit a bit so not so dependent on noise
        # if svwinlen > 11:
        #     svn = 11
        # else:
        #     svn = svwinlen
        # if svn % 2 == 0:  # if even, decrease by 1 point to meet ood requirement for savgol_filter
        #     svn -=1
        #
        # if svn > 3:  # go ahead and filter
        #     self.Crit =  scipy.signal.savgol_filter(self.Crit, svn, 2)
        sd = np.std(self.Crit)  # HERE IS WHERE TO SCREEN OUT STIMULI/EVOKED
        self.sdthr = sd * self.threshold  # set the threshold
        self.above = np.clip(self.Crit, self.sdthr, None)
        self.onsets = (
            scipy.signal.argrelextrema(self.above, np.greater, order=int(order))[0]
            - 1
            + self.idelay
        )
        self.summarize(self.data)


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
        self.dt = None
        self.maxlpf = None
        self.sign = 1
        self.taus = None
        self.template_max = None
        self.idelay = 0
        self.method = "aj"

    def deconvolve(
        self,
        data: np.ndarray,
        data_nostim: Union[list, np.ndarray, None] = None,
        llambda: float = 5.0,
        order: int = 7,
        lpf: Union[float, None] = None,
        verbose: bool = False,
    ) -> None:
        # cprint('r', "STARTING AJ")
        starttime = timeit.default_timer()
        if self.template is None:
            self._make_template()

        starttime = timeit.default_timer()
        self.timebase = np.arange(0.0, data.shape[0] * self.dt, self.dt)
        jmax = np.argmin(np.fabs(self.timebase - 0.6))
        print(jmax)
        self.data = self.LPFData(data[:jmax], lpf)
        self.timebase = self.timebase[:jmax]
        #    print (np.max(self.timebase), self.dt)
        self.data = self.data - np.mean(self.data)
        # Weiner filtering

        H = np.fft.fft(self.template)
        if H.shape[0] < self.data.shape[0]:
            H = np.hstack((H, np.zeros(self.data.shape[0] - H.shape[0])))
        self.quot = np.fft.ifft(
            np.fft.fft(self.data) * np.conj(H) / (H * np.conj(H) + llambda ** 2.0)
        )
        self.Crit = np.real(self.quot)*llambda
        # self.Crit = np.absolute(self.quot)
        if data_nostim is None:
            sd = np.std(self.Crit)
        else:  # clip to max of crit array, and be sure index array is integer, not float
            critmeas = [self.Crit[int(x)] for x in data_nostim if x < self.Crit.shape[0]]
            sd = np.std(critmeas)
        self.sdthr = sd * self.threshold  # set the threshold
        self.above = np.clip(self.Crit, self.sdthr, None)
        self.onsets = (
            scipy.signal.argrelextrema(self.above, np.greater, order=int(order))[0]
            - 1
            + self.idelay
        )
        endtime = timeit.default_timer() - starttime
        self.runtime = endtime
        self.summarize(self.data)
        endtime = timeit.default_timer() - starttime
        if verbose:
            print("AJ run time: {0:.4f} s".format(endtime))


class ZCFinder(MiniAnalyses):
    """
    Event finder using Luke's zero-crossing algorithm
    
    """

    def __init__(self):
        self.template = None
        self.onsets = None
        self.timebase = None
        self.dt = None
        self.sign = 1
        self.taus = None
        self.template_max = None
        self.idelay = 0
        self.threshold=2.5
        self.method = "zc"

    def find_events(
        self,
        data: np.ndarray,
        data_nostim: Union[list, np.ndarray, None] = None,
        lpf: Union[float, None] = None,
        hpf: Union[float, None] = None,
        minPeak: float = 0.0,
        minSum: float = 0.0,
        minLength: int = 3,
        verbose: bool = False,
    ) -> None:
        self.data = self.LPFData(data, lpf)
        if hpf is not None:
            self.data = FN.highPass(self.data, cutoff=hpf, dt=self.dt)
        # self.data = self.HPFData(data, hpf)
        self.timebase = np.arange(0.0, self.data.shape[0] * self.dt, self.dt)

        starttime = timeit.default_timer()
        self.sdthr = self.threshold*np.std(self.data)
        self.Crit = np.zeros_like(self.data)
        # if data_nostim is None:
        #     data_nostim = [range(self.Crit.shape[0])]  # whole trace, otherwise remove stimuli
        # else:  # clip to max of crit array, and be sure index array is integer, not float
        #     data_nostim = [int(x) for x in data_nostim if x < self.Crit.shape[0]]
        # data = FN.lowPass(data,cutoff=3000.,dt = 1/20000.)

        events = FN.zeroCrossingEvents(
            self.data,
            minLength=minLength,
            minPeak=minPeak,
            minSum=minSum,
            noiseThreshold=self.threshold,
            sign=self.sign,
        )
        self.onsets = np.array([x[0] for x in events]).astype(int)

        self.summarize(self.data)
        endtime = timeit.default_timer() - starttime
        if verbose:
            print("ZC run time: {0:.4f} s".format(endtime))


