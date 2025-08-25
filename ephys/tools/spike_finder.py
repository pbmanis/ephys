import numpy as np
import numba
from numba import jit, njit, types
from numba.typed import List
import scipy.signal
import numpy.ma as ma
import gc
from typing import Union

# Usage example:
# detector = FindSpikesNumba()
# spikes = detector.findspikes(x, v, t0, t1, threshold=threshold_val)
# Recomposed by Claude Sonnet 4.0 from utility.findspikes

# Numba-optimized helper functions
@njit
def find_threshold_crossings_numba(v, threshold, dv):
    """Find points where voltage exceeds threshold and slope is positive."""
    n = len(v)
    crossings = List.empty_list(numba.int64)
    
    for i in range(n):
        if v[i] > threshold and (i < len(dv) and dv[i] > 0.0):
            crossings.append(i)
    
    return crossings

@njit
def interpolate_threshold_crossing(x_vals, y_vals, threshold, dt):
    """Interpolate the exact threshold crossing time."""
    if len(x_vals) < 2 or len(y_vals) < 2:
        return x_vals[0] if len(x_vals) > 0 else 0.0
    
    m = (y_vals[1] - y_vals[0]) / dt  # local slope
    b = y_vals[0] - (x_vals[0] * m)
    return x_vals[1] + (threshold - b) / m if m != 0 else x_vals[1]

@njit
def interpolate_peak(x_vals, y_vals, dt):
    """Interpolate the exact peak time using quadratic fit."""
    if len(x_vals) < 3 or len(y_vals) < 3:
        return x_vals[1] if len(x_vals) > 1 else (x_vals[0] if len(x_vals) > 0 else 0.0)
    
    try:
        # mimic Igor FindPeak routine with B = 1
        m1 = (y_vals[1] - y_vals[0]) / dt  # local slope to left of peak
        m2 = (y_vals[2] - y_vals[1]) / dt  # local slope to right of peak
        mprime = (m2 - m1) / dt  # find where slope goes to 0
        bprime = m2 - ((dt / 2.0) * mprime)
        if mprime != 0:
            return -bprime / mprime + x_vals[1]
        else:
            return x_vals[1]
    except:
        return x_vals[1]

# @njit
def clean_spiketimes_numba(spike_times, mindT):
    """Remove spikes that are too close together (refractory period)."""
    if len(spike_times) <= 1:
        return spike_times
    
    # Sort the spike times
    sorted_spikes = np.sort(spike_times)
    cleaned = np.zeros(len(sorted_spikes))
    # List.empty_list(numba.float64)
    nsp = 0
    if len(sorted_spikes) > 0:
        cleaned[0] = sorted_spikes[0]
        nsp += 1
        for i in range(1, len(sorted_spikes)):
            if sorted_spikes[i] - cleaned[-1] >= mindT:
                cleaned[nsp]=sorted_spikes[i]
                nsp += 1

    return cleaned[:nsp]

@njit
def find_peaks_above_threshold(v, threshold, order):
    """Find local maxima above threshold with minimum distance."""
    n = len(v)
    #peaks = List.empty_list(numba.int64)
    peaks = np.zeros_like(v) # array([], dtype=np.int64)
    npk = 0
    for i in range(order, n - order):
        if v[i] < threshold:
            continue
            
        is_peak = True
        # Check if this point is greater than all points in the window
        for j in range(i - order, i + order + 1):
            if j != i and v[j] >= v[i]:
                is_peak = False
                break
        
        if is_peak:
            # Check minimum distance from previous peaks
            too_close = False
            for prev_peak in peaks:
                if abs(i - prev_peak) < order:
                    too_close = True
                    break
            
            if not too_close:
                peaks[npk] = i #  np.append(peaks, i)
                npk += 1

    return peaks[:npk] # np.array(peaks)
@jit
def filter_peaks_by_depth(v, peaks, mindip, dt, t_forward_samples):
    """Filter peaks by checking valley depth between consecutive peaks."""
    if len(peaks) <= 1:
        return peaks
    
    # filtered_peaks = List.empty_list(numba.int64)
    filtered_peaks = np.zeros_like(peaks) # array([], dtype=np.int64)
    n = len(v)
    npk = 0
    if len(peaks) > 0:
        filtered_peaks[0] = peaks[0]
    
    for i in range(len(peaks) - 1):
        current_peak = int(peaks[i])
        next_peak = int(peaks[i + 1])
        
        # Find the minimum in the test window
        test_end = int(min(current_peak + t_forward_samples, next_peak, n))
        if current_peak >= test_end:
            continue
            
        min_val = np.inf
        for j in range(int(current_peak), test_end):
            if v[j] < min_val:
                min_val = v[j]
        
        # Check if the dip is deep enough
        if (v[current_peak] - min_val) >= mindip:
            if i + 1 < len(peaks):  # Add next peak if depth check passes
                filtered_peaks[npk] = peaks[i + 1]
                npk += 1

    # Handle the last peak separately
    if len(filtered_peaks) > 1:
        last_peak = int(filtered_peaks[-1])
        test_end = int(min(last_peak + t_forward_samples, n))
        
        min_val = np.inf
        for j in range(last_peak, test_end):
            if v[j] < min_val:
                min_val = v[j]
        
        if (v[last_peak] - min_val) < mindip:
            # Remove the last peak
            filtered_peaks = filtered_peaks[:npk-1]
            return filtered_peaks # @ np.array(filtered_peaks[:-1])
    
    return filtered_peaks[:npk] # np.array(filtered_peaks)


# Alternative version that returns indices instead of times for even better performance
@jit
def nb_box_spike_find_indices(
    x: np.ndarray, 
    y: np.ndarray, 
    dt: float,
    thr: float = -35.0, 
    C1: float = -12.0, 
    C2: float = 11.0, 
    dt2: float = 1.75,
    minwidth: float = 0.0001,
) -> np.ndarray:
    """
    Ultra-fast version that returns indices instead of times.
    Convert to times afterwards if needed: spike_times = x[indices]
    """

    npts = len(x)
    spike_indices = np.zeros(npts, dtype=np.int64)
    if npts < 3:
        return None # spike_indices[:0]

    dt_actual = x[1] - x[0]
    iwid = int(dt2 / dt_actual)
    
    if iwid <= 0 or iwid >= npts // 2:
        return None # spike_indices[:0]


    nspk = 0
    # Vectorized-style loop for maximum performance
    for i in range(iwid, npts - iwid):
        if (y[i] > thr and 
            y[i] > y[i-1] and y[i] > y[i+1] and
            (y[i + iwid] - y[i]) < C1 and
            (y[i] - y[i - iwid]) > C2 and
            (x[i + iwid] - x[i - iwid]) > minwidth):
            spike_indices[nspk] = i
            nspk += 1

    return spike_indices[:nspk]

@jit(nopython=True, parallel=False, cache=False)
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
        # from . import c_deriv  # ty: ignore[unresolved-import]
        minwidth = 0.0001
        npts = len(x)
        x_data = x
        y_data = y
        dt = x_data[1]-x_data[0]  # use first interval, makes assumption that rate is constant
        iwid = (int)(dt2/dt)
        spkflag = 0
        spikes = np.zeros_like(y)
        for i in range(iwid, npts-iwid):
            if (y_data[i] > thr): # increasingly restrictive measures: works for clean data
                if (y_data[i]>y_data[i-1]) and (y_data[i] > y_data[i+1]):
                    if ((y_data[i+iwid] - y_data[i]) < C1) and ((y_data[i]-y_data[i-iwid]) > C2):
                        if (x_data[i+iwid] - x_data[i-iwid]) > minwidth:
                            spikes[i] = 1.0

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

# @njit 
def box_spike_find_cython(x, y, dt, thr, C1, C2, dt2, minwidth):
    """Cython implementation of box spike detection (Hight & Kalluri method)."""

    spikes = List.empty_list(numba.float64)
    
    # n = len(y)
    # dt2_samples = int(dt2 / dt)
    # minwidth_samples = int(minwidth / dt)
    from . import c_deriv
    spikes = np.zeros_like(y)
    c_deriv.c_box_spike_find_opt(  # use a cython implementation : much much faster
        x.view(np.ndarray),
        y.view(np.ndarray),
        x.shape[0] - 1,
        thr,  # threshold -35 mV
        C1,   # slope value
        C2,   # slope value
        dt2,  # spike window (nominal 1.75 msec)
        minwidth,  # minimum width (nominal, 0.1 msec)
        spikes,  # calculated spikes (times, set to 1 else 0)
        )
        # print('boxspikefind: ', spikes)
        # spikes = [s[0] for s in spikes] # make into 1-d array

    spikes = np.argwhere(spikes > 0.0)

    return np.array(spikes)

class FindSpikesNumba:
    """Numba-optimized spike detection class."""
    
    def __init__(self):
        pass
    
    def clean_spiketimes(self, spike_times, mindT:float):
        """Clean spike times by removing those within refractory period."""
        return clean_spiketimes_numba(spike_times, mindT)
    
    def box_spike_find(self, x, y, dt, thr, C1, C2, dt2, minwidth):
        """Box spike detection method (Hight & Kalluri)."""
        return box_spike_find_cython(x, y, dt, thr, C1, C2, dt2, minwidth)
        # return nb_box_spike_find_indices(x, y, dt, thr, C1, C2, dt2, minwidth)

    def findspikes(
        self,
        x: np.ndarray,  # expected in seconds
        v: np.ndarray,  # expected in Volts, but can modify with scaling below
        t0: float,  # sec
        t1: float,  # sec
        threshold: float = 0.0,  # V
        dt: float = 2e-5,  # sec
        mode: str = "schmitt",
        detector: str = "threshold",
        refract: float = 0.0007,  # sec
        interpolate: bool = False,
        peakwidth: float = 0.001,  # sec
        minwidth: float = 0.0003,  # sec (300 microseconds)
        mindip: float = 0.01,  # V
        data_time_units: str = 's',
        data_volt_units: str = 'V',
        debug: bool = False,
        verify: bool = False,
        pars: Union[dict, None] = None,
    ) -> Union[np.ndarray, List]:
        """
        Numba-optimized spike detection function.
        
        This function maintains the same interface as the original but uses
        Numba-compiled functions for the computationally intensive parts.
        """
        
        if mode not in ["schmitt", "threshold", "peak"]:
            raise ValueError(
                'pylibrary.utility.findspikes: mode must be one of "schmitt", "threshold", "peak" : got %s'
                % mode
            )
        if detector not in ["threshold", "argrelmax", "Kalluri", "find_peaks", "find_peaks_cwt"]:
            raise ValueError(
                'pylibrary.utility.findspikes: detector must be one of "argrelmax", "threshold" "Kalluri", "find_peaks", "find_peaks_cwt": got %s'
                % detector
            )
        
        assert data_time_units in ['s', 'ms']
        assert data_volt_units in ['V', 'mV']

        # Unit conversions
        tfac = 1.0
        if data_time_units == 'ms':
            tfac = 1e-3
            x = x * tfac   # convert to secs
            dt = dt * tfac

        vfac = 1
        if data_volt_units == 'mV':
            vfac = 1e-3
            v = v * vfac
            threshold = threshold * vfac

        # Time window selection
        if t1 is not None and t0 is not None:
            xt = ma.masked_outside(x, t0, t1)
            vma = ma.array(v, mask=ma.getmask(xt))
            xt = ma.compressed(xt)  # convert back to usual numpy arrays
            vma = ma.compressed(vma)
            i0 = int(t0 / dt)
            i1 = int(t1 / dt)
        else:
            xt = np.array(x)
            vma = np.array(v)
            i0 = 0
            i1 = len(x)

        # Kalluri detector (uses original non-Numba scipy functions for complex logic)
        if detector == "Kalluri":
            if pars is None:
                pars = {
                    'dt2': 1.75 * 1e-3 / tfac,  # s
                    'C1': -12.0 * 1e-3 / vfac,  # V
                    'C2': 11.0 * 1e-3 / vfac,  # V
                    'minwidth': 0.1 * 1e-3 / tfac,  # s
                }
            else:
                k = list(pars.keys())
                assert ('dt2' in k) and ('C1' in k) and ('C2' in k)
            from . import c_deriv  # ty: ignore[unresolved-import]
            u = c_deriv.c_box_spike_find_opt(
                x=x, y=v, dt=dt, thr=threshold, 
                C1=pars['C1'], C2=pars['C2'], dt2=pars['dt2'],
                minwidth=pars['minwidth'],
            )
            # print(u)
            if u is None or len(u) == 0:
                st = np.array([], dtype=np.int64)
            else:
                v = u*dt
                st = np.array([spike for spike in v if (spike >= t0) and (spike <= t1)])

            gc.collect()
            print("Mindt: ", refract)
            return self.clean_spiketimes(st, mindT=refract)

        # Main spike detection logic
        dv = np.diff(vma) / dt  # compute slope
        st = np.array([])  # store spike times

        if detector == "threshold":
            # Use Numba-optimized threshold crossing detection
            sp_indices = find_threshold_crossings_numba(vma, threshold, dv)
            sp = [int(idx) for idx in sp_indices]  # convert to regular list for compatibility

        elif detector in ["argrelmax", "find_peaks", "find_peaks_cwt"]:
            if detector == "find_peaks_cwt":
                # Use scipy for complex CWT - harder to optimize with Numba
                spv = np.where(vma > threshold)[0].tolist()
                spks = scipy.signal.find_peaks_cwt(
                    vma, widths=np.arange(2, int(peakwidth / dt)), noise_perc=0.1
                )
                if len(spks) > 0:
                    stn = spks[np.where(vma[spks] >= threshold)[0]]
                else:
                    stn = []
                    
            elif detector == 'find_peaks':
                # Use scipy for now - could be optimized with Numba in future
                order = int(refract / dt) + 1
                stn = scipy.signal.find_peaks(vma, height=threshold, distance=order)[0]
                
            elif detector == "argrelmax":
                # Use Numba-optimized peak finding
                order = int(refract / dt) + 1
                stn = find_peaks_above_threshold(vma, threshold, order)

            # Use Numba-optimized peak filtering
            if len(stn) > 0:
                t_forward = int(0.010 / dt)  # use 10 msec forward for drop
                stn2 = filter_peaks_by_depth(vma, stn, mindip, dt, t_forward)
            else:
                stn2 = np.array([])

            if len(stn2) > 0:
                xspk = x[[int(s) + int(t0 / dt) for s in stn2]]
                return self.clean_spiketimes(xspk, mindT=refract)
            else:
                return np.array([])
        else:
            raise ValueError("Utility:findspikes: invalid detector")

        if len(sp) == 0:
            return st

        # Process detected spikes based on mode
        if mode in ["schmitt", "Schmitt"]:
            # Schmitt trigger mode with optional interpolation
            for k in sp:
                if k > 0 and k < len(xt) - 1:
                    xx = xt[k - 1:k + 1]
                    y = vma[k - 1:k + 1]
                    
                    if interpolate and len(xx) >= 2 and len(y) >= 2:
                        interpolated_time = interpolate_threshold_crossing(xx, y, threshold, dt)
                        st = np.append(st, interpolated_time)
                    else:
                        if len(xx) > 1:
                            st = np.append(st, xx[1])

        elif mode == "peak":
            # Peak detection mode
            kpkw = int(peakwidth / dt)
            spv = np.where(vma > threshold)[0]
            z = (np.array(np.where(np.diff(spv) > 1)[0]) + 1).tolist()
            z.insert(0, 0)  # first element in spv is needed to get starting AP
            
            for k in z:
                if k < len(spv):
                    zk = spv[k]
                    if zk + kpkw < len(vma):
                        spk = np.argmax(vma[zk:zk + kpkw]) + zk  # find the peak position
                        
                        if spk > 0 and spk < len(xt) - 2:
                            xx = xt[spk - 1:spk + 2]
                            y = vma[spk - 1:spk + 2]
                            
                            if interpolate and len(xx) >= 3 and len(y) >= 3:
                                interpolated_time = interpolate_peak(xx, y, dt)
                                st = np.append(st, interpolated_time)
                            else:
                                if len(xx) > 1:
                                    st = np.append(st, xx[1])

        # Verification plot (if requested)
        if verify:
            from matplotlib import pyplot as mpl
            print(("nspikes detected: ", len(st)))
            mpl.figure()
            mpl.plot(x, v, "k-", linewidth=0.5)
            mpl.plot(st, threshold * np.ones_like(st), "ro")
            mpl.show()

        # Clean and return spike times
        return self.clean_spiketimes(st, mindT=refract)

