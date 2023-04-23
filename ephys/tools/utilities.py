"""
Utility.py - general utility routines::
    - power spectrum
    - elliptical filtering
    - handling very long input lines for dictionaries
    - general measurement routines for traces (mean, std, spikes, etc)

"declassed", 7/28/09 p. manis
Use as::
    import Utility as Utils
    then call Utils.xxxxx()

"""
# January, 2009
# Paul B. Manis, Ph.D.
# UNC Chapel Hill
# Department of Otolaryngology/Head and Neck Surgery
# Supported by NIH Grants DC000425-22 and DC004551-07 to PBM.
# See license
#
"""
    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.

"""
import fnmatch
import gc
import itertools
import os
import re
import sys
import types
from typing import Union, List, Tuple
from random import sample

import numpy as np
import scipy.signal
from ephys.ephys_analysis import c_deriv
from numba import jit
from numpy import ma as ma
from scipy import fftpack as spFFT

debugFlag = False

class ScriptError(Exception):
    pass


@jit(nopython=False, parallel=False, cache=True)
def nb_deriv(x, y, order=1):
    """
    Compute a derivative of order n of V

    """
    deriv = np.zeros_like(y)
    d = y.copy()
    for k in range(order):
        deriv[0] = np.diff(d[:2]) / np.diff(x[:2])  # endpoints
        deriv[-1] = np.diff(d[-2:]) / np.diff(x[-2:])
        for i in range(
            1, deriv.shape[0] - 1
        ):  # for all interior points, use 3-point measure.
            # print(np.diff(d[i-1:i+2]))
            deriv[i] = np.mean(np.diff(d[i - 1 : i + 2])) / np.mean(
                np.diff(x[i - 1 : i + 2])
            )
        d = deriv
    return deriv

# @jit(nopython=True, parallel=False, cache=False)
def nb_clean_spiketimes(
    spikeTimes: Union[List, np.ndarray], mindT: float = 0.7
) -> Union[List, np.ndarray]:
    """
    Clean up spike time array, removing all less than mindT

    Parameters
    ----------
    spikeTimes : list or numpy array (1-D)
        array of the spike times
    
    mindT : float (default : 0.7)
        minimum time between spikes, in the same units as spikeTimes
        (normally this will be in milliseconds or seconds)
    
    Return
    ------
    spikeTimes : list or numpy array (1-D)
        A cleaned list of the spike times where the events are at least
        mindT appart.
        Note: If no spikes or just one spike in the input array, just return the array

    """
    st = np.array(spikeTimes)
    print("mindT: ", mindT)
    if st.shape[0] > 1:
        dst = np.diff(st)
        sp_ok = np.argwhere(dst > mindT)
        sp_ok_i = np.zeros(len(sp_ok)+1)
        print(dst)
        print("spok: ", sp_ok)
        sp_ok_i[0] = st[0]
        print([i for i in sp_ok_i])
        sp_ok_i[1:] = [st[int(i)+1] for i in sp_ok_i]
        st[1:] = st[dst] 
        #np.append(st, [spikeTimes[s + 1] for s in sok])
        return st
    else:
        return st

def test_clean():
    x = [0.1, 0.1002, 0.2, 0.2002, 0.3, 0.4, 0.5, 0.5001]
    print(x)
    print(nb_clean_spiketimes(x, mindT=0.001))

@jit(nopython=False, parallel=False, cache=False)
def nb_box_spike_find(x:np.ndarray, y:np.ndarray, dt:float, 
        thr:float=-35.0, C1:float=-12.0, C2:float=11.0, dt2:float=1.75,
        data_time_units:str='s') -> np.ndarray:
        """
        Find spikes using a box method:
        Must be > threshold, and be above the rising/falling values in the window dt2
        Units must be consistent: x, dt, dt2 (s or ms)
        Unist must be consistent: y, thr, C1, C2 (V or mV)
        Note: probably works best with mV and ms, given the constants above.
        to C1, C2 and the width dt2
        From Hight and Kalluri, J Neurophysiol., 2016
        Note: Implementation is in cython (pyx) file in ephys/ephysanalysis
        
        Returns an array of indices in x where spikes occur
        """
        spikes = np.zeros_like(y)
        c_deriv.c_box_spike_find(  # use a cython implementation : much much faster
            x.view(np.ndarray),
            y.view(np.ndarray),
            x.shape[0] - 1,
            thr,  # threshold -35 mV
            C1,  #  # slope value
            C2,  # slope value
            dt2,  # spike window (nominal 1.75 msec)
            spikes,  # calculated spikes (times, set to 1 else 0)
        )
        # print('boxspikefind: ', spikes)
        # spikes = [s[0] for s in spikes] # make into 1-d array
        sf = 1.0
        if data_time_units == 'ms':
            dt *= 1e3
        spikes = np.argwhere(spikes > 0.0) * dt
        # print('thr c1 c2: ', thr, C1, C2, dt2)
        # print('boxspikefind: ', spikes)
        spkt = np.array([s[0] for s in spikes])
        # print('spkt: ', spkt)
        return spkt

def nb_arg_relmax(
        x:np.ndarray, vma:np.ndarray, t0:float, t1:float,
        thresh: float, refract:float, mindip:float, dt: float
        ):
        #  spks = scipy.signal.find_peaks_cwt(vma[spv], np.arange(2, int(peakwidth/dt)), noise_perc=0.1)
        order = int(refract / dt) + 1
        stn = scipy.signal.find_peaks(vma, height=thresh, distance=order)[0]
        # argrelmax seems to miss peaks occasionally
        # spks = scipy.signal.argrelmax(vma, order=order)[0]
        # stn = spks[np.where(vma[spks] >= thresh)[0]]
        if len(stn) > 0:
            stn2 = [stn[0]]
        else:
            stn2 = []
        # filter peaks by checking that valleys between pairs
        # are sufficiently deep. Note that this only checks
        # BETWEEN spikes, so we need to do an additional
        # check of the last "spike" separately
        removed = []
        t_forward = int(0.010 / dt)  # use 10 msec forward for drop
        for i in range(len(stn) - 1):  # for all putative peaks
            if i in removed:  # this can happen if event was removed in j loop
                continue
            test_end = min([stn[i] + t_forward, stn[i + 1], vma.shape[0]])

            if stn[i] == test_end:
                continue
            elif (vma[stn[i]] - np.min(vma[stn[i] : test_end])) < mindip:
                if (
                    i == 0
                ):  # special case: if first event fails, remove it from output list
                    stn2 = []
                removed.append(i)
                continue
            else:
                stn2.append(stn[i])
        # handle "spikes" that do not repolarize and are the *last* spike
        if len(stn2) > 1:
            test_end = stn2[-1] + t_forward
            minv = np.min(vma[stn2[-1] : test_end])
            if (vma[stn2][-1] - minv) < mindip:
                removed.append(stn2[-1])
                stn2 = stn2[:-1]  # remove the last spike
        stn2 = sorted(list(set(stn2)))
        xspk = x[[s + int(t0 / dt) for s in stn2]]
        return xspk


class Utility:
    def __init__(self):
        self.debugFlag = False

    def setDebug(self, debug=False):
        if debug:
            self.debugFlag = True
        else:
            self.debugFlag = False

    def pSpectrum(
        self, data: np.ndarray, samplefreq: float = 44100
    ) -> Tuple[np.ndarray, np.ndarray]:
        npts = len(data)
        # we should window the data here
        if npts == 0:
            print("? no data in pSpectrum")
            return
        # pad to the nearest higher power of 2
        (a, b) = np.frexp(npts)
        if a <= 0.5:
            b = b = 1
        npad = 2 ** b - npts
        if debugFlag:
            print(("npts: %d   npad: %d   npad+npts: %d" % (npts, npad, npad + npts)))
        padw = np.append(data, np.zeros(npad))
        npts = len(padw)
        sigfft = spFFT.fft(padw)
        nUniquePts = np.ceil((npts + 1) / 2.0)
        sigfft = sigfft[0:nUniquePts]
        spectrum = abs(sigfft)
        spectrum = spectrum / float(npts)  # scale by the number of points so that
        # the magnitude does not depend on the length
        # of the signal or on its sampling frequency
        spectrum = spectrum ** 2  # square it to get the power
        spmax = np.amax(spectrum)
        spectrum = spectrum + 1e-12 * spmax
        # multiply by two (see technical document for details)
        # odd nfft excludes Nyquist point
        if npts % 2 > 0:  # we've got odd number of points fft
            spectrum[1 : len(spectrum)] = spectrum[1 : len(spectrum)] * 2
        else:
            spectrum[1 : len(spectrum) - 1] = (
                spectrum[1 : len(spectrum) - 1] * 2
            )  # we've got even number of points fft
        freqAzero = np.arange(0, nUniquePts, 1.0) * (samplefreq / npts)
        return (spectrum, freqAzero)

    def sinefit(self, x: np.ndarray, y: np.ndarray, F: float) -> Tuple[float, float]:
        """ LMS fit of a sine wave with period T to the data in x and y
            aka "cosinor" analysis. 

        """
        npar = 2
        w = 2.0 * np.pi * F
        A = np.zeros((len(x), npar), float)
        A[:, 0] = np.sin(w * x)
        A[:, 1] = np.cos(w * x)
        (p, residulas, rank, s) = np.linalg.lstsq(A, y)
        Amplitude = np.sqrt(p[0] ** 2 + p[1] ** 2)
        Phase = np.arctan2(p[1], p[0])  # better check this...
        return (Amplitude, Phase)

    def sinefit_precalc(self, x: np.ndarray, y: np.ndarray, F: float) -> Tuple[float, float]:
        """ LMS fit of a sine wave with period T to the data in x and y
            aka "cosinor" analysis. 
            assumes that A (in sinefit) is precalculated

        """
        (p, residulas, rank, s) = np.linalg.lstsq(x, y)
        Amplitude = np.sqrt(p[0] ** 2 + p[1] ** 2)
        Phase = np.arctan2(p[1], p[0])  # better check this...
        return (Amplitude, Phase)

    # filter signal with elliptical filter
    def SignalFilter(
        self, signal: np.ndarray, LPF: float, HPF: float, samplefreq: float
    ) -> np.ndarray:
        if debugFlag:
            print(("sfreq: %f LPF: %f HPF: %f" % (samplefreq, LPF, HPF)))
        flpf = float(LPF)
        fhpf = float(HPF)
        sf = float(samplefreq)
        sf2 = sf / 2
        wp = [fhpf / sf2, flpf / sf2]
        ws = [0.5 * fhpf / sf2, 2 * flpf / sf2]
        if debugFlag:
            print(
                "signalfilter: samplef: %f  wp: %f, %f  ws: %f, %f lpf: %f  hpf: %f"
                % (sf, wp[0], wp[1], ws[0], ws[1], flpf, fhpf)
            )
        filter_b, filter_a = scipy.signal.iirdesign(
            wp, ws, gpass=1.0, gstop=60.0, ftype="ellip"
        )
        msig = np.mean(signal)
        signal = signal - msig
        w = scipy.signal.lfilter(
            filter_b, filter_a, signal
        )  # filter the incoming signal
        signal = signal + msig
        if debugFlag:
            print(
                (
                    "sig: %f-%f w: %f-%f"
                    % (np.amin(signal), np.amax(signal), np.amin(w), np.amax(w))
                )
            )
        return w

    # filter with Butterworth low pass, using time-causal lfilter
    def SignalFilter_LPFButter(
        self,
        signal: np.ndarray,
        LPF: float,
        samplefreq: float,
        NPole: int = 8,
        bidir: bool = False,
    ) -> np.ndarray:
        wn = [LPF / (samplefreq / 2.0)]
        b, a = scipy.signal.butter(NPole, wn, btype="low", output="ba")
        zi = scipy.signal.lfilter_zi(b, a)
        if bidir:
            out, zo = scipy.signal.filtfilt(b, a, signal, zi=zi * signal[0])
        else:
            out, zo = scipy.signal.lfilter(b, a, signal, zi=zi * signal[0])
        return np.array(out)

    # filter with Butterworth high pass, using time-causal lfilter
    def SignalFilter_HPFButter(
        self,
        signal: np.ndarray,
        HPF: float,
        samplefreq: float,
        NPole: int = 8,
        bidir: bool = False,
    ) -> np.ndarray:
        flpf = float(HPF)
        sf = float(samplefreq)
        wn = [flpf / (sf / 2.0)]
        b, a = scipy.signal.butter(NPole, wn, btype="high", output="ba")
        zi = scipy.signal.lfilter_zi(b, a)
        if bidir:
            out = scipy.signal.filtfilt(b, a, signal)  # , zi=zi*signal[0])
        else:
            out = scipy.signal.lfilter(b, a, signal)  # , zi=zi*signal[0])
        return np.array(out)

    # filter with Bessel high pass, using time-causal lfilter
    def SignalFilter_HPFBessel(
        self,
        signal: np.ndarray,
        HPF: float,
        samplefreq: float,
        NPole: int = 8,
        bidir: bool = False,
    ) -> np.ndarray:
        flpf = float(HPF)
        sf = float(samplefreq)
        wn = [flpf / (sf / 2.0)]
        b, a = scipy.signal.bessel(NPole, wn, btype="high", output="ba")
        zi = scipy.signal.lfilter_zi(b, a)
        if bidir:
            out = scipy.signal.filtfilt(b, a, signal)  # , zi=zi*signal[0])
        else:
            out = scipy.signal.lfilter(b, a, signal)  # , zi=zi*signal[0])
        return np.array(out)

    # filter signal with low-pass Bessel
    def SignalFilter_LPFBessel(
        self,
        signal: np.ndarray,
        LPF: float,
        samplefreq: float,
        NPole: int = 8,
        bidir: bool = False,
        reduce: bool = False,
    ) -> np.ndarray:
        """ Low pass filter a signal, possibly reducing the number of points in the
            data array.
            Parameters
            ----------
            signal: a numpya array of dim = 1, 2 or 3. The "last" dimension is filtered.
            
            LPF: low pass filter frequency, in Hz
            
            samplefreq: sampline frequency (points/second)
            
            NPole: number of poles in the filter.
            
            reduce: Flag that controls whether the resulting data is subsampled or not

        """
        if self.debugFlag:
            print(f"sfreq: {samplefreq:f}  LPF: {LPF:f}")
        wn = [LPF / (samplefreq / 2.0)]
        filter_b, filter_a = scipy.signal.bessel(NPole, wn, btype="low", output="ba")
        reduction = 1
        if reduce:
            if LPF <= samplefreq / 2.0:
                reduction = int(samplefreq / LPF)
        if self.debugFlag is True:
            print(
                f"signalfilter: samplef: {samplefreq:f}  wn: {wn:f}  lpf: {LPF:f}  NPoles: {NPole:d}"
            )
            sm = np.mean(signal)
            if bidir:
                w = scipy.signal.filtfilt(
                    filter_b, filter_a, signal - sm
                )  # filter the incoming signal
            else:
                w = scipy.signal.lfilter(
                    filter_b, filter_a, signal - sm
                )  # filter the incoming signal

            w = w + sm
            if reduction > 1:
                w = scipy.signal.resample(w, reduction)
            return w
        if signal.ndim == 2:
            sh = np.shape(signal)
            for i in range(0, np.shape(signal)[0]):
                sm = np.mean(signal[i, :])
                if bidir:
                    w1 = scipy.signal.filtfilt(filter_b, filter_a, signal[i, :] - sm)
                else:
                    w1 = scipy.signal.lfilter(filter_b, filter_a, signal[i, :] - sm)

                w1 = w1 + sm
                if reduction == 1:
                    w1 = scipy.signal.resample(w1, reduction)
                if i == 0:
                    w = np.empty((sh[0], np.shape(w1)[0]))
                w[i, :] = w1
            return w
        if signal.ndim == 3:
            sh = np.shape(signal)
            for i in range(0, np.shape(signal)[0]):
                for j in range(0, np.shape(signal)[1]):
                    sm = np.mean(signal[i, j, :])
                    if bidir:
                        w1 = scipy.signal.filtfilt(
                            filter_b, filter_a, signal[i, j, :] - sm
                        )
                    else:
                        w1 = scipy.signal.lfilter(
                            filter_b, filter_a, signal[i, j, :] - sm
                        )
                    w1 = w1 + sm
                    if reduction == 1:
                        w1 = scipy.signal.resample(w1, reduction)
                    if i == 0 and j == 0:
                        w = np.empty((sh[0], sh[1], np.shape(w1)[0]))
                    w[i, j, :] = w1
            return w
        if signal.ndim > 3:
            print(
                "Error: signal dimesions of > 3 are not supported (no filtering applied)"
            )
            return signal

    # do an eval on a long line (longer than 512 characters)
    # assumes input is a dictionary (as a string) that is too long
    # parses by breaking the string down and then reconstructing each element
    #
    def long_Eval(self, line: str) -> object:
        inpunct = False
        sp = ""
        u = {}
        i = 0
        inpunct = 0
        colonFound = False
        inquote = False
        for c in line:
            if c == "{":
                continue
            if (
                (c == "," or c == "}") and colonFound and not inpunct and not inquote
            ):  # separator is ','
                r = eval("{%s}" % sp)
                u[list(r.keys())[0]] = r[list(r.keys())[0]]
                colonFound = False
                sp = ""
                continue
            sp = sp + c
            if c == ":":
                colonFound = True
                continue
            if c == "(" or c == "[":
                inpunct += 1
                continue
            if c == ")" or c == "]":
                inpunct -= 1
                continue
            if c == "'" and inquote:
                inquote = False
                continue
            if c == "'" and not inquote:
                inquote is True
        return u

    # routine to flatten an array/list.
    #
    def flatten(self, data: list, datatypes=(list, tuple)) -> list:
        i = 0
        while i < len(data):
            while isinstance(data[i], datatypes):
                if not data[i]:
                    data.pop(i)
                    if not len(data):
                        break
                else:
                    data[i : i + 1] = list(data[i])
            i += 1
        return data

    def unique(self, seq, keepstr=True):
        t = type(seq)
        if t in (str, str):
            t = (list, "".join)[bool(keepstr)]
        seen = []
        return t(c for c in seq if not (c in seen or seen.append(c)))

    ######################
    # Frequently used analysis routines
    ######################

    def _rollingSum(self, data, n):
        d1 = data.copy()
        d1[1:] += d1[:-1]  # integrate
        d2 = np.empty(len(d1) - n + 1, dtype=data.dtype)
        d2[0] = d1[n - 1]  # copy first point
        d2[1:] = d1[n:] - d1[:-n]  # subtract the rest
        return d2

    # routine to find all the local maxima
    def local_maxima(self, data, span=10, sign=1):
        from scipy.ndimage import maximum_filter, minimum_filter

        data = np.asarray(data)
        print(("data size: ", data.shape))
        if sign <= 0:  # look for minima
            maxfits = minimum_filter(data, size=span, mode="wrap")
        else:
            maxfits = maximum_filter(data, size=span, mode="wrap")
        print(("maxfits shape: ", maxfits.shape))
        maxima_mask = np.where(data == maxfits)
        good_indices = np.arange(len(data))[maxima_mask]
        print(("len good index: ", len(good_indices)))
        good_fits = data[maxima_mask]
        order = good_fits.argsort()
        return good_indices[order], good_fits[order]

    def clementsBekkers(self, data, template, threshold=1.0, minpeakdist=15):
        """
        Luke's quick version
        See mini_analysis package for a more complete version
        """

        D = data.view(np.ndarray)
        T = template.view(np.ndarray)
        N = len(T)
        window = np.ones(N)
        sumT = T.sum()
        sumT2 = (T ** 2).sum()
        sumD = self._rollingSum(D, N)
        sumD2 = self._rollingSum(D ** 2, N)
        sumTD = np.correlate(D, T, mode="valid")
        scale = (sumTD - sumT * sumD / N) / (sumT2 - sumT ** 2 / N)
        offset = (sumD - scale * sumT) / N
        SSE = (
            sumD2
            + scale ** 2 * sumT2
            + N * offset ** 2
            - 2 * (scale * sumTD + offset * sumD - scale * offset * sumT)
        )
        error = np.sqrt(SSE / (N - 1))
        sf = scale / error
        # isolate events from the sf signal
        a = sf * np.where(sf >= threshold, 1, 0)
        (evp, eva) = self.local_maxima(a, span=minpeakdist, sign=1)
        # now clean it up
        u = np.where(eva > 0.0)
        t_start = evp[evp[u]]
        d_start = eva[evp[u]]
        return (t_start, d_start)  # just return the list of the starts

    def RichardsonSilberberg(self, data, tau, time=None):
        D = data.view(np.ndarray)
        rn = tau * np.diff(D) + D[:-2, :]
        rn = self.savitzky_golay(rn, kernel=11, order=4)
        if time is not None:
            vn = rn - tau * self.savitzky_golay(np.diff(D), kernel=11, order=4)
            return (rn, vn)
        else:
            return rn

    def clean_spiketimes(
        self, spikeTimes: Union[List, np.ndarray], mindT: float = 0.7
    ) -> Union[List, np.ndarray]:
        """
        Clean up spike time array, removing all less than mindT

        Parameters
        ----------
        spikeTimes : list or numpy array (1-D)
            array of the spike times
        
        mindT : float (default : 0.7)
            minimum time between spikes, in the same units as spikeTimes
            (normally this will be in milliseconds or seconds)
        
        Return
        ------
        spikeTimes : list or numpy array (1-D)
            A cleaned list of the spike times where the events are at least
            mindT appart.
            Note: If no spikes or just one spike in the input array, just return the array

        """

        if len(spikeTimes) > 1:
            dst = np.diff(spikeTimes)
            st = np.array(spikeTimes[0])  # get first spike
            sok = np.where(dst > mindT)
            st = np.append(st, [spikeTimes[s + 1] for s in sok])
            spikeTimes = st
        return spikeTimes

    def deriv(self, x: np.ndarray, y: np.ndarray, order: int = 1) -> np.ndarray:
        dout = np.zeros_like(y)
        c_deriv.c_deriv(
            x.view(np.ndarray), y.view(np.ndarray), dout.shape[0] - 1, order, dout
        )
        return dout

    def box_spike_find(self, x, y, dt, thr=-35.0, C1=-12.0, C2=11.0, dt2=1.75):
        """
        Find spikes using a box method:
        Must be > threshold, and be above the rising/falling values in the window dt2
        Units must be consistent: x, dt, dt2 (s or ms)
        Unist must be consistent: y, thr, C1, C2 (V or mV)
        Note: probably works best with mV and ms, given the constants above.
        to C1, C2 and the width dt2
        From Hight and Kalluri, J Neurophysiol., 2016
        Note: Implementation is in cython (pyx) file in ephys/ephysanalysis
        
        Returns an array of indices in x where spikes occur
        """
        spikes = np.zeros_like(y)
        c_deriv.c_box_spike_find(  # use a cython implementation : much much faster
            x.view(np.ndarray),
            y.view(np.ndarray),
            x.shape[0] - 1,
            thr,  # threshold -35 mV
            C1,  #  # slope value
            C2,  # slope value
            dt2,  # spike window (nominal 1.75 msec)
            spikes,  # calculated spikes (times, set to 1 else 0)
        )
        # print('boxspikefind: ', spikes)
        # spikes = [s[0] for s in spikes] # make into 1-d array
        sf = 1.0
        spikes = np.argwhere(spikes > 0.0) * dt
        # print('thr c1 c2: ', thr, C1, C2, dt2)
        # print('boxspikefind: ', spikes)
        spkt = [s[0] for s in spikes]
        # print('spkt: ', spkt)
        return spkt

       
    # def findspikes_nb(  # calls numba version... ? 
    #     self,
    #     x: np.ndarray,  # expected in seconds
    #     v: np.ndarray,  # expected in Volts, but can modifiy with scaling below
    #     thresh: float = 0.0,  # V
    #     t0: Union[float, None] = None,  # sec
    #     t1: Union[float, None] = None,  # sec
    #     dt: float = 2e-5,  # sec
    #     mode: str = "schmitt",
    #     detector: str = "threshold",
    #     refract: float = 0.0007,  # sec
    #     interpolate: bool = False,
    #     peakwidth: float = 0.001,  # sec
    #     mindip: float = 0.01,  # V
    #     data_time_units: str='s',
    #     data_volt_units: str='V',
    #     debug: bool = False,
    #     verify: bool = False,
    # ) -> Union[np.ndarray, List]:
    #     """
    #     findspikes identifies the times of action potential in the trace v, with the
    #     times in t. An action potential is simply timed at the first point that exceeds
    #     the threshold... or is the peak. 
    #     4/1/11 - added peak mode
    #     if mode is none or schmitt, we work as in the past.
    #     if mode is peak, we return the time of the peak of the AP instead
    #     7/15/11 - added interpolation flag
    #     if True, the returned time is interpolated, based on a spline fit
    #     if False, the returned time is just taken as the data time. 
                    
    #     Note: TIME UNITS MUST MATCH.
    #     Units are set up for SECONDS in time base (acq4 standard)
    #     Units are set up for VOLTS in voltages.
    #     All time entities should be in SECONDS.
    #     The data can be converted from ms to s and from mV to V
    #     by setting data_time_units and data_volt_units
        
    #     Parameters
    #     ----------
    #         x : numpy array (no default)
    #             time base, in seconds
    #         v : numpy array
    #             voltage array to search for spikes (Volts)
    #         thresh : float (no default)
    #             voltage for threshold detection (Volts)
    #         t0, t1 : float (default: None)
    #             time for start end end of spike search (seconds)
    #             if None, whole trace is used
    #         dt : float (default: 20e-5 seconds)
    #             sample rate, in seconds
    #         mode : string (default: 'schmitt')
    #             trigger mode for most detection algorithms
    #         detector : string (default: 'threshold')
    #             detector mode (choices are 'threshold' and 'argrelmax')
    #             Argrelmax works well to find spikes, and algorithm cleans up
    #             event list
    #         refract : float (default: 0.0007)
    #             minimum refractory period for spike inclusion, in seconds (or units of time base)
    #     """

    #     if mode not in ["schmitt", "threshold", "peak"]:
    #         raise ValueError(
    #             'pylibrary.utility.findspikes: mode must be one of "schmitt", "threshold", "peak" : got %s'
    #             % mode
    #         )
    #     if detector not in ["threshold", "argrelmax", "Kalluri"]:
    #         raise ValueError(
    #             'pylibrary.utility.findspikes: mode must be one of "argrelmax", "threshold" "Kalluri": got %s'
    #             % detector
    #         )
    #     assert data_time_units in ['s', 'ms']
    #     assert data_volt_units in ['V', 'mV']
    #     # parameters for the Hight and Kalluri detecctor
    #     dt2 = 1.75e-3  # s
    #     C1 = -0.012  # V
    #     C2 = 0.011  # V
    #     if data_time_units == 'ms':
    #         x = x*1e-3   # convert to secs
    #         dt = dt*1e-3
            
    #     if data_volt_units == 'mV':
    #         v = v*1e-3
    #         thresh = thresh*1e-3
    #         # C1 = C1 * 1e-3
    #         # C2 = C2 * 1e-3

    #     if detector == "Kalluri":
    #         print("Kalluri")
    #         return nb_box_spike_find(
    #             x=x, y=v, dt=dt, thr=thresh, C1=C1, C2=C2, dt2=dt2,
    #             data_time_units=data_time_units,
    #         )

    #     if t1 is not None and t0 is not None:
    #         xt = ma.masked_outside(x, t0, t1)
    #         vma = ma.array(v, mask=ma.getmask(xt))
    #         xt = ma.compressed(xt)  # convert back to usual numpy arrays then
    #         vma = ma.compressed(vma)
    #     else:
    #         xt = np.array(x)
    #         vma = np.array(vma)
    #     # print('max x: ', np.max(xt))
    #    #  print('dt: ', dt)

    #     dv = np.diff(vma) / dt  # compute slope
    #     dv2 = np.diff(dv) / dt  # and second derivative
    #     st = np.array([])  # store spike times

    #     if detector == "threshold":
    #         print("threshold")
    #         spv = np.where(vma > thresh)[0].tolist()  # find points above threshold
    #         sps = (
    #             np.where(dv > 0.0)[0] + 1
    #         ).tolist()  # find points where slope is positive
    #         sp = list(
    #             set(spv) & set(sps)
    #         )  # intersection defines putative spike start times
    #         # then go on to mode...

    #     elif detector == "argrelmax":
    #         print("argrelmax")
    #         xspk = nb_arg_relmax(x=xt, vma=vma, t0=t0, t1=t1,
    #              thresh=thresh, refract=refract, mindip=mindip, dt=dt)
    #         return nb_clean_spiketimes(xspk, mindT=refract)  # done here.
    #     else:
    #         raise ValueError("Utility:findspikes: invalid detector")

    #     sp.sort()  # make sure all detected events are in order (sets is unordered)

    #     spl = sp
    #     sp = tuple(sp)  # convert to tuple
    #     if sp == ():
    #         return st  # nothing detected

    #     if mode in [
    #         "schmitt",
    #         "Schmitt",
    #     ]:  # normal operating mode is fixed voltage threshold, with hysterisis
    #         for k in sp:
    #             xx = xt[k - 1 : k + 1]
    #             y = vma[k - 1 : k + 1]
    #             if interpolate:
    #                 m = (y[1] - y[0]) / dt  # local slope
    #                 b = y[0] - (xx[0] * m)
    #                 st = np.append(st, xx[1] + (thresh - b) / m)
    #             else:
    #                 if len(x) > 1:
    #                     st = np.append(st, xx[1])

    #     elif mode == "peak":
    #         kpkw = int(peakwidth / dt)
    #         z = (np.array(np.where(np.diff(spv) > 1)[0]) + 1).tolist()
    #         #            print('z: ', z)
    #         z.insert(0, 0)  # first element in spv is needed to get starting AP
    #         for k in z:
    #             zk = spv[k]
    #             spk = np.argmax(vma[zk : zk + kpkw]) + zk  # find the peak position
    #             xx = xt[spk - 1 : spk + 2]
    #             y = vma[spk - 1 : spk + 2]
    #             if interpolate:
    #                 try:
    #                     # mimic Igor FindPeak routine with B = 1
    #                     m1 = (y[1] - y[0]) / dt  # local slope to left of peak
    #                     b1 = y[0] - (xx[0] * m1)
    #                     m2 = (y[2] - y[1]) / dt  # local slope to right of peak
    #                     b2 = y[1] - (xx[1] * m2)
    #                     mprime = (
    #                         m2 - m1
    #                     ) / dt  # find where slope goes to 0 by getting the line
    #                     bprime = m2 - ((dt / 2.0) * mprime)
    #                     st = np.append(st, -bprime / mprime + xx[1])
    #                 except:
    #                     continue
    #             else:
    #                 # print('utility: yere', x)
    #                 if len(xx) > 1:
    #                     st = np.append(st, xx[1])  # always save the first one

    #     # clean spike times
    #     # # st = clean_spiketimes(st, mindT=refract)
    #     # print(("nspikes detected: ", len(st)), 'max spike time:', np.max(st))
    #     # st2 = self.clean_spiketimes(st, mindT=refract)
    #     # print(("nspikes detected after cleaning: ", len(st2)))

    #     if verify:
    #         from matplotlib import pyplot as mpl

    #         print(("nspikes detected: ", len(st)))
    #         mpl.figure()
    #         mpl.plot(x, v, "k-", linewidth=0.5)
    #         mpl.plot(st, thresh * np.ones_like(st), "ro")
    #         mpl.plot(xt[spv], v[spv], "r-")
    #         mpl.plot(xt[sps], v[sps], "m-", linewidth=1)
    #         mpl.show()
    #     # exit(1)

    #     return nb_clean_spiketimes(st, mindT=refract)

    def findspikes(
        self,
        x: np.ndarray,  # expected in seconds
        v: np.ndarray,  # expected in Volts, but can modifiy with scaling below
        thresh: float = 0.0,  # V
        t0: Union[float, None] = None,  # sec
        t1: Union[float, None] = None,  # sec
        dt: float = 2e-5,  # sec
        mode: str = "schmitt",
        detector: str = "threshold",
        refract: float = 0.0007,  # sec
        interpolate: bool = False,
        peakwidth: float = 0.001,  # sec
        mindip: float = 0.01,  # V
        data_time_units: str='s',
        data_volt_units: str='V',
        debug: bool = False,
        verify: bool = False,
        pars: Union[dict, None] = None, # parameters that might be needed (e.g., Hight and Kalluri)
    ) -> Union[np.ndarray, List]:
        """
        findspikes identifies the times of action potential in the trace v, with the
        times in t. An action potential is simply timed at the first point that exceeds
        the threshold... or is the peak. 
        4/1/11 - added peak mode
        if mode is none or schmitt, we work as in the past.
        if mode is peak, we return the time of the peak of the AP instead
        7/15/11 - added interpolation flag
        if True, the returned time is interpolated, based on a spline fit
        if False, the returned time is just taken as the data time. 
                    
        Note: TIME UNITS MUST MATCH.
        Units are set up for SECONDS in time base (acq4 standard)
        Units are set up for VOLTS in voltages.
        All time entities should be in SECONDS.
        The data can be converted from ms to s and from mV to V
        by setting data_time_units and data_volt_units
        
        Parameters
        ----------
            x : numpy array (no default)
                time base, in seconds
            v : numpy array
                voltage array to search for spikes (Volts)
            thresh : float (no default)
                voltage for threshold detection (Volts)
            t0, t1 : float (default: None)
                time for start end end of spike search (seconds)
                if None, whole trace is used
            dt : float (default: 20e-5 seconds)
                sample rate, in seconds
            mode : string (default: 'schmitt')
                trigger mode for most detection algorithms
            detector : string (default: 'threshold')
                detector mode (choices are 'threshold' and 'argrelmax')
                Argrelmax works well to find spikes, and algorithm cleans up
                event list
            refract : float (default: 0.0007)
                minimum refractory period for spike inclusion, in seconds (or units of time base)
            pars: A dictionary of parameters that might be needed for some detection routines
        """

        if mode not in ["schmitt", "threshold", "peak"]:
            raise ValueError(
                'pylibrary.utility.findspikes: mode must be one of "schmitt", "threshold", "peak" : got %s'
                % mode
            )
        if detector not in ["threshold", "argrelmax", "Kalluri", "find_peaks", "find_peaks_cwt"]:
            raise ValueError(
                'pylibrary.utility.findspikes: mode must be one of "argrelmax", "threshold" "Kalluri", "find_peaks", "find_peaks_cwt": got %s'
                % detector
            )
        assert data_time_units in ['s', 'ms']
        assert data_volt_units in ['V', 'mV']

        tfac = 1.0
        if data_time_units == 'ms':
            tfac = 1e-3
            x = x*tfac   # convert to secs
            dt = dt*tfac

        vfac = 1
        if data_volt_units == 'mV':
            vfac = 1e-3
            v = v*vfac
            thresh = thresh*vfac

        if t1 is not None and t0 is not None:
            xt = ma.masked_outside(x, t0, t1)
            vma = ma.array(v, mask=ma.getmask(xt))
            xt = ma.compressed(xt)  # convert back to usual numpy arrays then
            vma = ma.compressed(vma)
            i0 = int(t0/dt)
            i1 = int(t1/dt)
        else:
            xt = np.array(x)
            vma = np.array(vma)
            i0 = 0
            i1 = len(x)
    
        if detector == "Kalluri":
            # parameters for the Hight and Kalluri detecctor
            if pars is None:
                pars = {'dt2': 1.75*1e-3/tfac, # s
                        'C1': -12.0*1e-3/vfac,  # V
                        'C2' : 11.0*1e-3/vfac,  # V
                }
            else:
                k = list(pars.keys())
                assert ('dt2' in k) and ('C1' in k) and ('C2' in k)

            u = self.box_spike_find(
                x=x, y=v, dt=dt, thr=thresh, C1=pars['C1'], C2=pars['C2'], dt2=pars['dt2'],
            )
            st = np.array([x for x in u if (x >= t0)and (x <= t1)])  # limit to those in the window
            # if len(st) > 0:
            #     print('min st, max st, min dtst: ', min(st), max(st), min(np.diff(st)))
            #     print("refract: ", refract)
            #    st = self.clean_spiketimes(st, mindT=refract)
            gc.collect()
            return self.clean_spiketimes(st, mindT=refract)

        # print('max x: ', np.max(xt))
       #  print('dt: ', dt)

        dv = np.diff(vma) / dt  # compute slope
        dv2 = np.diff(dv) / dt  # and second derivative
        st = np.array([])  # store spike times

        if detector == "threshold":
            spv = np.where(vma > thresh)[0].tolist()  # find points above threshold
            sps = (
                np.where(dv > 0.0)[0] + 1
            ).tolist()  # find points where slope is positive
            sp = list(
                set(spv) & set(sps)
            )  # intersection defines putative spike start times
            # then go on to mode...

        elif detector in ["argrelmax", "find_peaks", "find_peaks_cwt"]:
            if detector == "find_peaks_cwt":
                spv = np.where(vma > thresh)[0].tolist()  # find points above threshold
                spks = scipy.signal.find_peaks_cwt(vma, widths=np.arange(2, int(peakwidth/dt)), noise_perc=0.1)
                if len(spks) > 0:
                    stn = spks[np.where(vma[spks] >= thresh)[0]]
                else:
                    stn = []
            elif detector == 'find_peaks':
                order = int(refract / dt) + 1
                stn = scipy.signal.find_peaks(vma, height=thresh, distance=order)[0]
            elif detector == "argrelmax":
                # argrelmax seems to miss peaks occasionally
                order = int(refract / dt) + 1
                spks = scipy.signal.argrelmax(vma, order=order)[0]
                stn = spks[np.where(vma[spks] >= thresh)[0]]
            if len(stn) > 0:
                stn2 = [stn[0]]
            else:
                stn2 = []
            # filter peaks by checking that valleys between pairs
            # are sufficiently deep. Note that this only checks
            # BETWEEN spikes, so we need to do an additional
            # check of the last "spike" separately
            removed = []
            t_forward = int(0.010 / dt)  # use 10 msec forward for drop
            for i in range(len(stn) - 1):  # for all putative peaks
                if i in removed:  # this can happen if event was removed in j loop
                    continue
                test_end = min([stn[i] + t_forward, stn[i + 1], vma.shape[0]])

                if stn[i] == test_end:
                    continue
                elif (vma[stn[i]] - np.min(vma[stn[i] : test_end])) < mindip:
                    if (
                        i == 0
                    ):  # special case: if first event fails, remove it from output list
                        stn2 = []
                    removed.append(i)
                    continue
                else:
                    stn2.append(stn[i])
            # handle "spikes" that do not repolarize and are the *last* spike
            if len(stn2) > 1:
                test_end = stn2[-1] + t_forward
                minv = np.min(vma[stn2[-1] : test_end])
                if (vma[stn2][-1] - minv) < mindip:
                    removed.append(stn2[-1])
                    stn2 = stn2[:-1]  # remove the last spike
            stn2 = sorted(list(set(stn2)))
            if debug:
                print("stn: ", stn)
                print(vma[stn])
            # if len(stn2) > 0:  # good to test algorithm
            #     import matplotlib.pyplot as mpl
            #
            #     f, ax = mpl.subplots(1,1)
            #     ax.plot(xt, vma)
            #     ax.plot(xt[stn2], vma[stn2], 'ro')
            #     ax.plot(xt[stn], vma[stn], 'bx')
            #     mpl.show()
            xspk = x[[s + int(t0 / dt) for s in stn2]]
            return self.clean_spiketimes(xspk, mindT=refract)  # done here.
        else:
            raise ValueError("Utility:findspikes: invalid detector")

        sp.sort()  # make sure all detected events are in order (sets is unordered)

        spl = sp
        sp = tuple(sp)  # convert to tuple
        if sp == ():
            return st  # nothing detected

        if mode in [
            "schmitt",
            "Schmitt",
        ]:  # normal operating mode is fixed voltage threshold, with hysterisis
            for k in sp:
                xx = xt[k - 1 : k + 1]
                y = vma[k - 1 : k + 1]
                if interpolate:
                    m = (y[1] - y[0]) / dt  # local slope
                    b = y[0] - (xx[0] * m)
                    st = np.append(st, xx[1] + (thresh - b) / m)
                else:
                    if len(x) > 1:
                        st = np.append(st, xx[1])

        elif mode == "peak":
            kpkw = int(peakwidth / dt)
            z = (np.array(np.where(np.diff(spv) > 1)[0]) + 1).tolist()
            #            print('z: ', z)
            z.insert(0, 0)  # first element in spv is needed to get starting AP
            for k in z:
                zk = spv[k]
                spk = np.argmax(vma[zk : zk + kpkw]) + zk  # find the peak position
                xx = xt[spk - 1 : spk + 2]
                y = vma[spk - 1 : spk + 2]
                if interpolate:
                    try:
                        # mimic Igor FindPeak routine with B = 1
                        m1 = (y[1] - y[0]) / dt  # local slope to left of peak
                        b1 = y[0] - (xx[0] * m1)
                        m2 = (y[2] - y[1]) / dt  # local slope to right of peak
                        b2 = y[1] - (xx[1] * m2)
                        mprime = (
                            m2 - m1
                        ) / dt  # find where slope goes to 0 by getting the line
                        bprime = m2 - ((dt / 2.0) * mprime)
                        st = np.append(st, -bprime / mprime + xx[1])
                    except:
                        continue
                else:
                    # print('utility: yere', x)
                    if len(xx) > 1:
                        st = np.append(st, xx[1])  # always save the first one

        # clean spike times
        # # st = clean_spiketimes(st, mindT=refract)
        # print(("nspikes detected: ", len(st)), 'max spike time:', np.max(st))
        # st2 = self.clean_spiketimes(st, mindT=refract)
        # print(("nspikes detected after cleaning: ", len(st2)))

        if verify:
            from matplotlib import pyplot as mpl

            print(("nspikes detected: ", len(st)))
            mpl.figure()
            mpl.plot(x, v, "k-", linewidth=0.5)
            mpl.plot(st, thresh * np.ones_like(st), "ro")
            mpl.plot(xt[spv], v[spv], "r-")
            mpl.plot(xt[sps], v[sps], "m-", linewidth=1)
            mpl.show()
        # exit(1)

        return self.clean_spiketimes(st, mindT=refract)

    # def findspikes2(
    #     self,
    #     xin,
    #     vin,
    #     thresh,
    #     t0=None,
    #     t1=None,
    #     dt=1.0,
    #     mode=None,
    #     interpolate=False,
    #     debug=False,
    # ):
    #     """ Findspikes identifies the times of action potential in the trace v, with the
    #     times in t. An action potential is simply timed at the first point that exceeds
    #     the threshold... or is the peak. 
    #     4/1/11 - added peak mode
    #     if mode is none or schmitt, we work as in the past.
    #     if mode is peak, we return the time of the peak of the AP instead
    #     7/15/11 - added interpolation flag
    #     if True, the returned time is interpolated, based on a spline fit
    #     if False, the returned time is just taken as the data time.
    #     2012/10/9: Removed masked arrays and forced into ndarray from start
    #     (metaarrays were really slow...) 
    #     """

    #     st = np.array([])
    #     spk = []
    #     if xin is None:
    #         return (st, spk)
    #     xt = xin.view(np.ndarray)
    #     v = vin.view(np.ndarray)
    #     if t1 is not None and t0 is not None:
    #         it0 = int(t0 / dt)
    #         it1 = int(t1 / dt)
    #         if not isinstance(xin, np.ndarray):
    #             xt = xt[it0:it1]
    #             v = v[it0:it1]
    #         else:
    #             xt = xt[it0:it1]
    #             v = v[it0:it1]
    #     # if debug:
    #     #     f = pylab.figure(1)
    #     #     print "xt: ", xt
    #     #     print "v: ", v
    #     #     pylab.plot(np.array(xt), v, 'k-')
    #     #     pylab.draw()
    #     #     pylab.show()

    #     dv = np.diff(v, axis=0)  # compute slope
    #     try:
    #         dv = np.insert(dv, 0, dv[0])
    #     except:
    #         pass  # print 'dv: ', dv
    #     dv /= dt
    #     st = np.array([])
    #     spk = []
    #     spv = np.where(v > thresh)[0].tolist()  # find points above threshold
    #     sps = np.where(dv > 0.0)[0].tolist()  # find points where slope is positive
    #     sp = list(
    #         set.intersection(set(spv), set(sps))
    #     )  # intersection defines putative spikes
    #     sp.sort()  # make sure all detected events are in order (sets is unordered)
    #     sp = tuple(sp)  # convert to tuple
    #     if sp == ():
    #         return (st, spk)  # nothing detected
    #     dx = 1
    #     mingap = int(
    #         0.0005 / dt
    #     )  # 0.5 msec between spikes (a little unphysiological...)
    #     # normal operating mode is fixed voltage threshold
    #     # for this we need to just get the FIRST positive crossing,
    #     if mode == "schmitt":
    #         sthra = list(np.where(np.diff(sp) > mingap))
    #         sthr = [sp[x] for x in sthra[0]]  # bump indices by 1
    #         # print 'findspikes: sthr: ', len(sthr), sthr
    #         for k in sthr:
    #             if k == 0:
    #                 continue
    #             x = xt[k - 1 : k + 1]
    #             y = v[k - 1 : k + 1]
    #             if interpolate:
    #                 dx = 0
    #                 m = (y[1] - y[0]) / dt  # local slope
    #                 b = y[0] - (x[0] * m)
    #                 s0 = (thresh - b) / m
    #             else:
    #                 s0 = x[1]
    #             st = np.append(st, x[1])

    #     elif mode == "peak":
    #         pkwidth = 1.0e-3  # in same units as dt  - usually msec
    #         kpkw = int(pkwidth / dt)
    #         z = (np.array(np.where(np.diff(spv) > 1)[0]) + 1).tolist()
    #         z.insert(0, 0)  # first element in spv is needed to get starting AP
    #         spk = []
    #         # print 'findspikes peak: ', len(z)
    #         for k in z:
    #             zk = spv[k]
    #             spkp = np.argmax(v[zk : zk + kpkw]) + zk  # find the peak position
    #             x = xt[spkp - 1 : spkp + 2]
    #             y = v[spkp - 1 : spkp + 2]
    #             if interpolate:
    #                 try:
    #                     # mimic Igor FindPeak routine with B = 1
    #                     m1 = (y[1] - y[0]) / dt  # local slope to left of peak
    #                     b1 = y[0] - (x[0] * m1)
    #                     m2 = (y[2] - y[1]) / dt  # local slope to right of peak
    #                     b2 = y[1] - (x[1] * m2)
    #                     mprime = (
    #                         m2 - m1
    #                     ) / dt  # find where slope goes to 0 by getting the line
    #                     bprime = m2 - ((dt / 2.0) * mprime)
    #                     st = np.append(st, -bprime / mprime + x[1])
    #                     spk.append(spkp)
    #                 except:
    #                     continue
    #             else:
    #                 st = np.append(st, x[1])  # always save the first one
    #                 spk.append(spkp)
    #     return (st, spk)

    # getSpikes returns a dictionary with keys that are record numbers, each with values
    # that are the array of spike timesin the spike window.
    # data is studied from the "axis", and only ONE block should be in the selection.
    # thresh sets the spike threshold.

    def getSpikes(
        self,
        x,
        y,
        axis,
        tpts,
        tdel=0,
        thresh=0,
        selection=None,
        refractory=1.0,
        mode="schmitt",
        interpolate=False,
    ):
        if selection is None:  # really means whatever is displayed/selected
            selected = np.arange(0, np.shape(y)[0]).astype(int).tolist()
        else:
            selected = selection
        splist = {}
        if y.ndim == 3:
            for r in selected:
                splist[r] = self.findspikes(
                    x[tpts],
                    y[r, axis, tpts],
                    thresh,
                    dt=refractory,
                    mode=mode,
                    interpolate=interpolate,
                )
        else:
            splist = self.findspikes(
                x[tpts],
                y[tpts],
                thresh,
                dt=refractory,
                mode=mode,
                interpolate=interpolate,
            )
        return splist

    # return a measurement made on a block of traces
    # within the window t0-t1, on the data "axis", and according to the selected mode

    def measureTrace(
        self, x, y, t0=0, t1=10, thisaxis=0, mode="mean", selection=None, threshold=0
    ):
        result = np.array([])
        if selection is None:  # whooops
            return
        else:
            selected = selection
        if np.ndim(y) == 4:  # we have multiple block
            for i in range(0, len(y)):
                d = y[i][selected[i], thisaxis, :]  # get data for this block
                for j in range(0, np.shape(d)[0]):
                    if isinstance(threshold, int):
                        thr = threshold
                    else:
                        thr = threshold[j]
                    (m1, m2) = self.measure(mode, x[i], d[j, :], t0, t1, thresh=thr)
                    result = np.append(result, m1)
        else:
            d = y[selected, thisaxis, :]  # get data for this block
            for j in range(0, np.shape(d)[0]):
                if isinstance(threshold, int):
                    thr = threshold
                else:
                    thr = threshold[j]
                (m1, m2) = self.measure(mode, x, d[j, :], t0, t1, thresh=thr)
                result = np.append(result, m1)
        return result

    def measureTrace2(self, x, y, t0=0, t1=10, thisaxis=0, mode="mean", threshold=0):
        """
        Simplified version that just expects a 2-d array for y, nothing fancy
        """
        result = np.array([])
        d = y.T  # get data for this block
        for j in range(0, np.shape(d)[0]):
            if isinstance(threshold, int):
                thr = threshold
            else:
                thr = threshold[j]
            (m1, m2) = self.measure(mode, x, d[j][:], t0, t1, thresh=thr)
            result = np.append(result, m1)
        return result

    def measure(self, mode, x, y, x0, x1, thresh=0):
        """ return the a measure of y in the window x0 to x1
        """
        xt = x.view(np.ndarray)  # strip Metaarray stuff -much faster!
        v = y.view(np.ndarray)

        xm = ma.masked_outside(xt, x0, x1).T
        ym = ma.array(v, mask=ma.getmask(xm))
        if mode == "mean":
            r1 = ma.mean(ym)
            r2 = ma.std(ym)
        if mode == "max" or mode == "maximum":
            r1 = ma.max(ym)
            r2 = xm[ma.argmax(ym)]
        if mode == "min" or mode == "minimum":
            r1 = ma.min(ym)
            r2 = xm[ma.argmin(ym)]
        if mode == "median":
            r1 = ma.median(ym)
            r2 = 0
        if mode == "p2p":  # peak to peak
            r1 = ma.ptp(ym)
            r2 = 0
        if mode == "std":  # standard deviation
            r1 = ma.std(ym)
            r2 = 0
        if mode == "var":  # variance
            r1 = ma.var(ym)
            r2 = 0
        if mode == "cumsum":  # cumulative sum
            r1 = ma.cumsum(ym)  # Note: returns an array
            r2 = 0
        if mode == "anom":  # anomalies = difference from averge
            r1 = ma.anom(ym)  # returns an array
            r2 = 0
        if mode == "sum":
            r1 = ma.sum(ym)
            r2 = 0
        if mode == "area" or mode == "charge":
            r1 = ma.sum(ym) / (ma.max(xm) - ma.min(xm))
            r2 = 0
        if mode == "latency":  # return first point that is > threshold
            sm = ma.nonzero(ym > thresh)
            r1 = -1  # use this to indicate no event detected
            r2 = 0
            if ma.count(sm) > 0:
                r1 = sm[0][0]
                r2 = len(sm[0])
        if mode == "count":
            r1 = ma.count(ym)
            r2 = 0
        if mode == "maxslope":
            return (0, 0)
            slope = np.array([])
            win = ma.flatnotmasked_contiguous(ym)
            st = int(len(win) / 20)  # look over small ranges
            for k in win:  # move through the slope measurementwindow
                tb = list(range(k - st, k + st))  # get tb array
                newa = np.array(self.dat[i][j, thisaxis, tb])
                ppars = np.polyfit(
                    x[tb], ym[tb], 1
                )  # do a linear fit - smooths the slope measures
                slope = np.append(slope, ppars[0])  # keep track of max slope
            r1 = np.amax(slope)
            r2 = np.argmax(slope)
        return (r1, r2)

    def mask(self, x, xm, x0, x1):
        if np.ndim(xm) != 1:
            print("utility.mask(): array to used to derive mask must be 1D")
            return np.array([])
        xmask = ma.masked_outside(xm, x0, x1)
        tmask = ma.getmask(xmask)
        if np.ndim(x) == 1:
            xnew = ma.array(x, mask=tmask)
            return xnew.compressed()
        if np.ndim(x) == 2:
            for i in range(0, np.shape(x)[0]):
                xnew = ma.array(x[i, :], mask=tmask)
                xcmp = ma.compressed(xnew)
                if i == 0:
                    print((ma.shape(xcmp)[0]))
                    print((np.shape(x)[0]))
                    xout = np.zeros((np.shape(x)[0], ma.shape(xcmp)[0]))
                xout[i, :] = xcmp
            return xout
        else:
            print("Utility.Mask: dimensions of input arrays are not acceptable")
            return np.array([])

    def clipdata(self, y, xm, x0, x1):
        mx = ma.getdata(np.mask(xm, xm, x0, x1))
        my = ma.getdata(np.mask(y, xm, x0, x1))
        return (mx, my)

    def count_spikes(self, spk):
        """ mostly protection for an older error in the findspikes routine, but
            now it should be ok to just get the first element of the shape """
        shspk = np.shape(spk)
        if len(shspk) == 0:
            nspk = 0
        elif shspk[0] == 0:
            nspk = 0
        else:
            nspk = shspk[0]
        return nspk

    def analyzeIV(self, t, V, I, tw, thr):
        """ analyze a set of voltage records (IV), with spike threshold
            tw is a list of [tdelay, tdur, tssw], where tdelay is the delay to
            the start of the step, tdur is the duration of the step, and tssw is
            the duration of the steady-state window prior to the end of the 
            step
            thr is the threshold that will be used for spike detection.
            Returns:
            a dictionary with:
            vmin 
            vss 
            i for vmin and vss 
            spike count 
            ispk 
            eventually should also include time constant measures,and adaptation ratio
        """
        ntraces = np.shape(V)[0]
        vss = []
        vmin = []
        vm = []
        ic = []
        nspikes = []
        ispikes = []
        tmin = []
        fsl = []
        fisi = []
        for j in range(0, ntraces):
            ts = tw[0]
            te = tw[1]
            td = tw[2]
            ssv = self.measure("mean", t, V[j, :], te - td, te)
            ssi = self.measure("mean", t, I[j, :], te - td, te)
            rvm = self.measure("mean", t, V[j, :], 0.0, ts - 1.0)
            minv = self.measure("min", t, V[j, :], ts, te)
            spk = self.findspikes(t, V[j, :], thr, t0=ts, t1=te)
            nspikes.append(self.count_spikes(spk))  # build spike list
            ispikes.append(ssi[0])
            if nspikes[-1] >= 1:
                fsl.append(spk[0])
            else:
                fsl.append(None)
            if nspikes[-1] >= 2:
                fisi.append(spk[1] - spk[0])
            else:
                fisi.append(None)
            vm.append(rvm[0])
            if ssi[0] < 0.0:  # just for hyperpolarizing pulses...
                ic.append(ssi[0])
                vss.append(ssv[0])  # get steady state voltage
                vmin.append(minv[0])  # and min voltage
                tmin.append(minv[1])  # and min time

        return {
            "I": np.array(ic),
            "Vmin": np.array(vmin),
            "Vss": np.array(vss),
            "Vm": np.array(vm),
            "Tmin": np.array(tmin),
            "Ispike": np.array(ispikes),
            "Nspike": np.array(nspikes),
            "FSL": np.array(fsl),
            "FISI": np.array(fisi),
        }

    def trim_psc(self, I_psc, dt:float, sign="+", artifact_duration:float=0.5e-3):
        # trim off positive or negative current spike at beginning of trace (e.g., artifact)
        # dt is sample interval in seconds (e.g, time between points)
        # artifact duration is the minimum duration in seconds
        p_flag = False
        n_flag = False
        for ip, psc in enumerate(I_psc):
            if ip*dt < artifact_duration:  # and a minimum duration
                I_psc[ip] = np.nan
            # elif sign == "+" and psc > 0:  # trim positive artifact at start
            #     I_psc[ip] = np.nan
            # elif sign == '-' and psc < -2e-8:  # trim negative artifact at start
            #     I_psc[ip] = np.nan

            else:
                break
        # print(ip*dt, len(I_psc)*dt)
        if all(np.isnan(I_psc)):
            print("trace was reduced to nothing")
            exit()
        return I_psc

    def ffind(self, path, shellglobs=None, namefs=None, relative=True):
        """
        Finds files in the directory tree starting at 'path' (filtered by
        Unix shell-style wildcards ('shellglobs') and/or the functions in
        the 'namefs' sequence).

        The parameters are as follows:

        - path: starting path of the directory tree to be searched
        - shellglobs: an optional sequence of Unix shell-style wildcards
          that are to be applied to the file *names* found
        - namefs: an optional sequence of functions to be applied to the
          file *paths* found
        - relative: a boolean flag that determines whether absolute or
          relative paths should be returned

        Please note that the shell wildcards work in a cumulative fashion
        i.e. each of them is applied to the full set of file *names* found.

        Conversely, all the functions in 'namefs'
            * only get to see the output of their respective predecessor
              function in the sequence (with the obvious exception of the
              first function)
            * are applied to the full file *path* (whereas the shell-style
              wildcards are only applied to the file *names*)

        Returns a list of paths for files found.
        """
        if not os.access(path, os.R_OK):
            raise ScriptError("cannot access path: '%s'" % path)

        fileList = []  # result list
        try:
            for dir, subdirs, files in os.walk(path):
                if shellglobs:
                    matched = []
                    for pattern in shellglobs:
                        filterf = lambda s: fnmatch.fnmatchcase(s, pattern)
                        matched.extend(list(filter(filterf, files)))
                    fileList.extend(["%s%s%s" % (dir, os.sep, f) for f in matched])
                else:
                    fileList.extend(["%s%s%s" % (dir, os.sep, f) for f in files])
            if not relative:
                fileList = list(map(os.path.abspath, fileList))
            if namefs:
                for ff in namefs:
                    fileList = list(filter(ff, fileList))
        except:
            raise ScriptError(str())
        return fileList

    def seqparse(self, sequence:str, mode:str='nd'):
        """ parse the list of the format:
         12;23/10 etc... like nxtrec in datac
         now also parses matlab functions and array formats, using eval

         first arg is starting number for output array
         second arg is final number
         / indicates the skip arg type
         basic: /n means skip n : e.g., 1;10/2 = 1,3,5,7,9
         special: /##:r means randomize order (/##rn means use seed n for randomization)
         special: /##:l means spacing of elements is logarithmic
         special: /##:s means spacing is logarithmic, and order is randomized. (/##sn means use seed n for randomization)
         special: /:a## means alternate with a number
         multiple sequences are returned in a list... just like single sequences...

         3 ways for list to be structured:
         1. standard datac record parses. List is enclosed inbetween single quotes
         2. matlab : (array) operator expressions. [0:10:100], for example
         3. matlab functions (not enclosed in quotes). Each function generates a new list
         note that matlab functions and matrices are treated identically

         Updated 9/07/2000, 11/13/2000, 4/7/2004 (arbitrary matlab function argument with '=')
         converted to python 3/2/2009
         Paul B. Manis, Ph.D.
         pmanis@med.unc.edu
         """

        seq = []
        target = []
        sequence.replace(
            " ", ""
        )  # remove all spaces - nice to read, not needed to calculate
        (seq2, sep, remain) = sequence.partition(
            "&"
        )  # find  and return nested sequences
        # print("seq: ", seq2, 'sep: ', sep, 'remain: ', remain)
        while len(seq2) != 0:
            try:
                (oneseq, onetarget) = self.recparse(seq2)
                seq.append(oneseq)
                target.append(onetarget)
            except:
                raise ValueError("recparse failed on : ", seq2)
            seq2, sep, remain = remain.partition("&")  # find  and returnnested sequences
        if mode == 'sequential':
            seq = [[x for y in seq for x in y], None]
        return seq, target

    def recparse(self, cmdstr:str):
        """ function to parse basic word unit of the list - a;b/c or the like
        syntax is:
        [target:]a;b[/c][*n]
        where:
        target is a parameter target identification (if present)
        the target can be anything - a step, a duration, a level....
        it just needs to be in a form that will be interepreted by the PyStim
        sequencer.
        a, b and c are numbers
        n, if present *n implies a "mode"
        such as linear, log, randomized, etc.
        """

        recs = []
        target = []
        seed = 0
        skip = 1.0
        (target, sep, rest) = cmdstr.partition(":")  # get the target
        if rest == "":
            rest = target  # no : found, so no target designated.
            target = ""
        rest = rest.replace(' ', ',')
        if ',' in rest:
            recs = eval(f"[{rest:s}]")  # evaluate as a list
            return recs, target
        (sfn, sep, rest1) = rest.partition(";")
        (sln, sep, rest2) = rest1.partition("/")
        (sskip, sep, mo) = rest2.partition("*")  # look for mode
        
        fn = float(sfn)
        ln = float(sln)
        if sskip != '':
            skip = float(sskip)
        else:
            skip = 1.0
        ln = ln + 0.01 * skip
        if mo == "":  # linear spacing; skip is size of step
            recs = np.arange(fn, ln, skip)

        if mo.find("l") >= 0:  # log spacing; skip is length of result
            recs = np.logspace(np.log10(fn), np.log10(ln),skip)

        if mo.find("t") >= 0:  # just repeat the first value
            recs = [fn]

        if mo.find("n") >= 0:  # use the number of steps, not the step size
            if skip == 1.0:
                sk = ln - fn
            else:
                sk = (ln-fn)/(skip-1.0)
            recs = np.arange(fn, ln, sk)

        if mo.find("r") >= 0:  # randomize the result
            if recs == []:
                recs = np.arange(fn, ln, skip)
            recs = sample(recs, len(recs))

        if mo.find("a") >= 0:  # alternation - also test for a value after that
            (arg, sep, value) = mo.partition("a")  # is there anything after the letter?
            if value == "":
                value = 0.0
            else:
                value = float(value)
            c = [value] * len(recs) * 2  # double the length of the sequence
            c[0 : len(c) : 2] = recs  # fill the alternate positions with the sequence
            recs = c  # copy back
        return (recs, target)

    def makeRGB(self, ncol=16, minc=32, maxc=216):
        """
        ncol = 16 # number of color spaces
        minc = 32 # limit color range
        maxc = 216
        """
        subd = int((maxc - minc) / ncol)
        np.random.seed(1)
        RGB = [[]]
        for r in range(minc, maxc, subd):
            for g in range(minc, maxc, subd):
                for b in range(minc, maxc, subd):
                    RGB.append(np.array([r, g, b]))
        # print "# of colors: ", len(self.RGB)
        rgb_order = np.random.permutation(len(RGB))  # randomize the order
        RGB = [RGB[x] for x in rgb_order]
        return RGB


###############################################################################
#
# main entry
#

# If this file is called direclty, then provide tests of some of the routines.
if __name__ == "__main__":
    # test sequence parser
    
    test_clean()
    exit()

    U = Utility()
    seq1 = "1;5"
    r, n = U.seqparse(seq1)
    print("test1: semicolon sequence: \n", 'seq1: ', seq1, 'r: ', r, " n: ", n)
    assert np.allclose(r[0], [1,2,3,4,5])
    
    seq1 = "1,2,3,4,5"
    r, n = U.seqparse(seq1)
    print("test2: comma sequence: \n", 'seq1: ', seq1, 'r: ', r, " n: ", n)
    assert np.allclose(r[0], [1,2,3,4,5])

    seq1 = "1 2 3 4 5"
    r, n = U.seqparse(seq1)
    print("test3: space sequence: \n", 'seq1: ', seq1, 'r: ', r, " n: ", n)
    assert np.allclose(r[0], [1,2,3,4,5])
    
    seq1 = "1,2&3;5"
    r, n = U.seqparse(seq1, mode='sequential')
    print("test4: mixed sequence: \n", 'seq1: ', seq1, 'r: ', r, " n: ", n)
    
    assert np.allclose(r[0], [1,2,3,4,5])
    
    
    #all(r[0]) == all([1,2,3,4,5])

    # from optparse import OptionParser
    # import matplotlib.pylab as MP
    # MP.rcParams['interactive'] = False
    # U = Utility()
    #
    # parser=OptionParser() # command line options
    # parser.add_option("-d", action="store_true", dest="dictionary", default=False)
    # parser.add_option("-s", action="store_true", dest="sinefit", default=False)
    # parser.add_option("-f", action="store_true", dest="findspikes", default=False)
    # parser.add_option("-c", action="store_true", dest="cb", default=False)
    #
    # argsin = sys.argv[1:]
    # if argsin is not None:
    #     (options, args) = parser.parse_args(argsin)
    # else:
    #     (options, args) = parser.parse_args()
    #
    # print (options)
    # if options.dictionary:
    #     d="{'CN_Dur': 100.0, 'PP_LP': 16000.0, 'ST_Dur': 50.0, 'Trials': 24.0, 'PP_HP': 8000.0, 'CN_Mode': 0, 'ITI_Var': 5.0, 'PP_GapFlag': False, 'PS_Dur': 50.0, 'ST_Level': 80.0, 'PP_Mode': 2, 'WavePlot': True, 'PP_Dur': 50.0, 'Analysis_LPF': 500.0, 'CN_Level': 70.0, 'NHabTrials': 2.0, 'PP_Notch_F2': 14000.0, 'PP_Notch_F1': 12000.0, 'StimEnable': True, 'PP_OffLevel': 0.0, 'Analysis_HPF': 75.0, 'CN_Var': 10.0, 'Analysis_Start': -100.0, 'ITI': 20.0, 'PP_Level': 90.0, 'Analysis_End': 100.0, 'PP_Freq': 4000.0, 'PP_MultiFreq': 'linspace(2.0,32.0,4.0)'} "
    #     di = U.long_Eval(d)
    #     print(('The dictionary is: ',))
    #     print (di)
    #
    # if options.cb: # test clements bekkers
    #     # first generate some events
    #     t = np.arange(0, 1000.0, 0.1)
    #     ta = np.arange(0, 50.0, 0.1)
    #     events = np.zeros(t.shape)
    #     events[[50,100,250,350, 475, 525, 900, 1500, 2800, 5000, 5200, 7000, 7500],] = 1
    #     tau1 = 3
    #     alpha = 1.0 * (ta/tau1) * np.exp(1 - ta/tau1)
    #     sig = scipy.signal.fftconvolve(events, alpha, mode='full')
    #     sig = sig[0:len(t)]+np.random.normal(0, 0.25, len(t))
    #     f = MP.figure()
    #     MP.plot(t, sig, 'r-')
    #     MP.plot(t, events, 'k-')
    #     # now call the finding routine, using the exact template (!)
    #     (t_start, d_start) = U.clementsBekkers(sig, alpha, threshold=0.5, minpeakdist=15)
    #     MP.plot(t_start, d_start, 'bs')
    #     MP.show()
    #
    # if options.findspikes: # test the findspikes routine
    #     dt = 0.1
    #     t = np.arange(0, 100, dt)
    #     v = np.zeros_like(t)-60.0
    #     p = range(20, 900, 50)
    #     p1 = range(19,899,50)
    #     p2 = range(21,901,50)
    #     v[p] = 20.0
    #     v[p1] = 15.0
    #     v[p2] = -20.0
    #     sp  self.findspikes(t, v, 0.0, dt = dt, mode = 'schmitt', interpolate = False)
    #     print 'findSpikes'
    #     print 'sp: ', sp
    #     f = MP.figure(1)
    #     MP.plot(t, v, 'ro-')
    #     si = (np.floor(sp/dt))
    #     print 'si: ', si
    #     spk = []
    #     for k in si:
    #         spk.append(np.argmax(v[k-1:k+1])+k)
    #     MP.plot(sp, v[spk], 'bs')
    #     MP.ylim((0, 25))
    #     MP.draw()
    #     MP.show()
    #
    #     exit()
    #     print "getSpikes"
    #     y=[]*5
    #     for j in range(0,1):
    #         d = np.zeros((5,1,len(v)))
    #         for k in range(0, 5):
    #             p = range(20*k, 500, 50 + int(50.0*(k/2.0)))
    #             vn = v.copy()
    #             vn[p] = 20.0
    # d[k, 0, :] = np.array(vn) # load up the "spike" array
    #         y.append(d)
    # tpts = range(0, len(t)) # np.arange(0, len(t)).astype(int).tolist()
    # def findspikes(x, v, thresh, t0=None, t1= None, dt=1.0, mode=None, interpolate=False):
    #     for k in range(0, len(y)):
    #         sp = getSpikes(t, y[k], 0, tpts, tdel=0, thresh=0, selection = None, interpolate = True)
    #         print 'r: %d' % k, 'sp: ', sp
    #
    # test the sine fitting routine
    # if options.sinefit:
    #     from np.random import normal
    #     F = 1.0/8.0
    #     phi = 0.2
    #     A = 2.0
    #     t = np.arange(0.0, 60.0, 1.0/7.5)
    # check over a range of values (is phase correct?)
    #     for phi in np.arange(-2.0*np.pi, 2.0*np.pi, np.pi/8.0):
    #         y = A * np.sin(2.*np.pi*t*F+phi) + normal(0.0, 0.5, len(t))
    #         (a, p) = sinefit(t, y, F)
    #         print "A: %f a: %f  phi: %f p: %f" % (A, a, phi, p)
