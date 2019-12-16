"""
Routines for digital filtering

"""
import numpy as np
import scipy.signal as spSignal

def SignalFilter_LPFButter(signal, LPF, samplefreq, NPole=8):
    """Filter with Butterworth low pass, using time-causal lfilter 
    
        Digitally low-pass filter a signal using a multipole Butterworth
        filter. Does not apply reverse filtering so that result is causal.
    
        Parameters
        ----------
        signal : array
            The signal to be filtered.
        LPF : float
            The low-pass frequency of the filter (Hz)
        samplefreq : float
            The uniform sampling rate for the signal (in seconds)
        NPole : int
            Number of poles for Butterworth filter. Positive integer.

        Returns
        -------
        w : array
        filtered version of the input signal

    """
    flpf = np.float(LPF)
    sf = np.float(samplefreq)
    wn = [flpf/(sf/2.0)]
    b, a = spSignal.butter(NPole, wn, btype='low', output='ba')
    zi = spSignal.lfilter_zi(b,a)
    out, zo = spSignal.lfilter(b, a, signal, zi=zi*signal[0])
    return out

def SignalFilter_HPFButter(signal, HPF, samplefreq, NPole=8):
    """Filter with Butterworth low pass, using time-causal lfilter 
    
        Digitally low-pass filter a signal using a multipole Butterworth
        filter. Does not apply reverse filtering so that result is causal.
    
        Parameters
        ----------
        signal : array
            The signal to be filtered.
        HPF : float
            The high-pass frequency of the filter (Hz)
        samplefreq : float
            The uniform sampling rate for the signal (in seconds)
        NPole : int
            Number of poles for Butterworth filter. Positive integer.

        Returns
        -------
        w : array
        filtered version of the input signal

    """
    fhpf = np.float(HPF)
    sf = np.float(samplefreq)
    wn = [fhpf/(sf/2.0)]
    b, a = spSignal.butter(NPole, wn, btype='high', output='ba')
    zi = spSignal.lfilter_zi(b,a)
    out, zo = spSignal.lfilter(b, a, signal, zi=zi*signal[0])
    return out
        
def SignalFilter_LPFBessel(signal, LPF, samplefreq, NPole=8, reduce=False):
    """Low pass filter a signal with a Bessel filter

        Digitally low-pass filter a signal using a multipole Bessel filter
        filter. Does not apply reverse filtering so that result is causal.
        Possibly reduce the number of points in the result array.

        Parameters
        ----------
        signal : a numpy array of dim = 1, 2 or 3. The "last" dimension is filtered.
            The signal to be filtered.
        LPF : float
            The low-pass frequency of the filter (Hz)
        samplefreq : float
            The uniform sampling rate for the signal (in seconds)
        NPole : int
            Number of poles for Butterworth filter. Positive integer.
        reduce : boolean (default: False)
            If True, subsample the signal to the lowest frequency needed to 
            satisfy the Nyquist critera.
            If False, do not subsample the signal.

        Returns
        -------
        w : array
            Filtered version of the input signal
    """

    flpf = float(LPF)
    sf = float(samplefreq)
    wn = [flpf/(sf/2.0)]
    reduction = 1
    if reduce:
        if LPF <= samplefreq/2.0:
            reduction = int(samplefreq/LPF)
    filter_b,filter_a=spSignal.bessel(
            NPole,
            wn,
            btype = 'low',
            output = 'ba')
    if signal.ndim == 1:
        sm = np.mean(signal)
        w = spSignal.lfilter(filter_b, filter_a, signal-sm) # filter the incoming signal
        w = w + sm
        if reduction > 1:
            w = spSignal.resample(w, reduction)
        return(w)
    if signal.ndim == 2:
        sh = np.shape(signal)
        for i in range(0, np.shape(signal)[0]):
            sm = np.mean(signal[i,:])
            w1 = spSignal.lfilter(filter_b, filter_a, signal[i,:]-sm)
            w1 = w1 + sm
            if reduction == 1:
                w1 = spSignal.resample(w1, reduction)
            if i == 0:
                w = np.empty((sh[0], np.shape(w1)[0]))
            w[i,:] = w1
        return w
    if signal.ndim == 3:
        sh = np.shape(signal)
        for i in range(0, np.shape(signal)[0]):
            for j in range(0, np.shape(signal)[1]):
                sm = np.mean(signal[i,j,:])
                w1 = spSignal.lfilter(filter_b, filter_a, signal[i,j,:]-sm)
                w1 = w1 + sm
                if reduction == 1:
                    w1 = spSignal.resample(w1, reduction)
                if i == 0 and j == 0:
                    w = np.empty((sh[0], sh[1], np.shape(w1)[0]))
                w[i,j,:] = w1
        return(w)
    if signal.ndim > 3:
        print("Error: signal dimesions of > 3 are not supported (no filtering applied)")
        return signal

def SignalFilter_Bandpass(signal, HPF, LPF, samplefreq):
    """Filter signal within a bandpass with elliptical filter

    Digitally filter a signal with an elliptical filter; handles
    bandpass filtering between two frequencies. 

    Parameters
    ----------
    signal : array
        The signal to be filtered.
    HPF : float
        The high-pass frequency of the filter (Hz)
    LPF : float
        The low-pass frequency of the filter (Hz)
    samplefreq : float
        The uniform sampling rate for the signal (in seconds)

    Returns
    -------
    w : array
        filtered version of the input signal
    """
    flpf = float(LPF)
    fhpf = float(HPF)
    sf = float(samplefreq)
    sf2 = sf/2
    wp = [fhpf/sf2, flpf/sf2]
    ws = [0.5*fhpf/sf2, 2*flpf/sf2]
    filter_b,filter_a=spSignal.iirdesign(wp, ws,
            gpass=1.0,
            gstop=60.0,
            ftype="ellip")
    msig = np.mean(signal)
    signal = signal - msig
    w = spSignal.lfilter(filter_b, filter_a, signal) # filter the incoming signal
    signal = signal + msig
    return(w)
    
def NotchFilter(signal, notchf=[60.], Q=90., QScale=True, samplefreq=None):
    assert samplefreq is not None
    w0 = np.array(notchf)/(float(samplefreq)/2.0)  # all W0 for the notch frequency
    if QScale:
        bw = w0[0]/Q
        Qf = (w0/bw)**np.sqrt(2)  # Bandwidth is constant, Qf varies
    else:
        Qf = Q * np.ones(len(notchf))  # all Qf are the same (so bandwidth varies)
    for i, f0 in enumerate(notchf):
        b, a = spSignal.iirnotch(w0[i], Qf[i])
        signal = spSignal.lfilter(b, a, signal, axis=-1, zi=None)
    return signal
    

def downsample(data, n, axis=0, xvals='subsample'):
    """Downsample by averaging points together across axis.
    If multiple axes are specified, runs once per axis.
    If a metaArray is given, then the axis values can be either subsampled
    or downsampled to match.
    """
    ma = None
    if (hasattr(data, 'implements') and data.implements('MetaArray')):
        ma = data
        data = data.view(ndarray)
        
    
    if hasattr(axis, '__len__'):
        if not hasattr(n, '__len__'):
            n = [n]*len(axis)
        for i in range(len(axis)):
            data = downsample(data, n[i], axis[i])
        return data
    
    nPts = int(data.shape[axis] / n)
    s = list(data.shape)
    s[axis] = nPts
    s.insert(axis+1, n)
    sl = [slice(None)] * data.ndim
    sl[axis] = slice(0, nPts*n)
    d1 = data[tuple(sl)]
    #print d1.shape, s
    d1.shape = tuple(s)
    d2 = d1.mean(axis+1)
    
    if ma is None:
        return d2
    else:
        info = ma.infoCopy()
        if 'values' in info[axis]:
            if xvals == 'subsample':
                info[axis]['values'] = info[axis]['values'][::n][:nPts]
            elif xvals == 'downsample':
                info[axis]['values'] = downsample(info[axis]['values'], n)
        return MetaArray(d2, info=info)

if __name__ == '__main__':
    import matplotlib.mlab as mlab
    import matplotlib.pylab as mpl
    detrend = 'linear'
    padding = 16384
    nfft = 512
    
    signal = np.random.randn(5000)
    fs = 2500.
    LPF = 500.
    samplefreq = fs
    fsig = SignalFilter_LPFButter(signal, LPF, samplefreq, NPole=8)
    fnotch = NotchFilter(fsig, notchf=[60.], Q=6., QScale=False, samplefreq=fs)
    
    s1 = mlab.psd(signal, NFFT=nfft, Fs=fs, detrend=detrend, window=mlab.window_hanning, noverlap=64, pad_to=padding)
    s2 = mlab.psd(fsig, NFFT=nfft, Fs=fs, detrend=detrend, window=mlab.window_hanning, noverlap=64, pad_to=padding)
    s3 = mlab.psd(fnotch, NFFT=nfft, Fs=fs, detrend=detrend, window=mlab.window_hanning, noverlap=64, pad_to=padding)
    mpl.plot(s1[1], s1[0], 'g-')
    mpl.plot(s1[1], s2[0], 'b-')
    mpl.plot(s1[1], s3[0], 'r-')
    mpl.show()
    