"""test the notch filtering methods

"""
import numpy as np

import scipy.signal
import matplotlib.pyplot as mpl
import ephys.mini_analyses.minis_methods as MM
import meegkit as MEK
from ephys.mini_analyses.util import UserTester

plot = False
def event(rise:float, decay:float=1.0, amp:float=-1, dur:float=0.025, rate:float=1e-4):
    tb = np.arange(0, dur, rate)
    wv = amp*(1.0-np.exp(-tb/rise))*np.exp(-tb/decay)
    return wv

def test_notchfilter():
    NT = NotchTester()
    NT.run_test()

class NotchTester(UserTester):
    def __init__(self):
        UserTester.__init__(self,  f"NotchTester")

    def run_test(self):
        MA = MM.AndradeJonas()
        samprate = 1e-4 # 10 kHz
        trace_dur = 5.0
        npts = int(trace_dur/samprate)

        notch = []
        notchfreqs = np.arange(60, 4000, 60)

        Q = 6.
        MI = MM.MiniAnalyses()
        MI.dt_seconds = samprate
        tb = np.linspace(0, trace_dur, npts)

        # 1. test flat line
        # data = np.zeros(npts)

        # data, _, _ = MEK.detrend.detrend(data, order=3)
        # datan = MI.NotchData(data=data, notch=notch, notch_Q = Q)
        # print(f"Max difference: {np.max(datan-data):.6e}")
        # f, ax = mpl.subplots(1,1)
        # mpl.plot(tb, data, 'k-')
        # mpl.plot(tb, datan, 'r--')
        # mpl.show()

        # #2. test 60 Hz sine wave
        data = np.cos(tb*(2*np.pi*60))
        # data, _, _ = MEK.detrend.detrend(data, order=5)
        winpts = int(0.5/samprate)
        hwin = int(winpts/2)
        window = scipy.signal.windows.hann(winpts, 1)
        twin = np.concatenate((window[:hwin], np.ones(npts-winpts), window[hwin:]))
        data = data*twin
        datan = MI.NotchFilterData(data=data, notchfreqs=notchfreqs, notchQ=Q)
        datan = datan[hwin:-hwin]
        tbn = tb[hwin:-hwin]
        print(f"Max difference: {np.max(datan-data[hwin:-hwin]):.6e}")

        #2. test multiple sine waves
        data  = np.zeros_like(tb)
        noisefs = np.arange(60, 4000, 100)
        n_noisef = len(noisefs)
        for fs in noisefs:
            data += (1./n_noisef)*(np.sqrt(60)/np.sqrt(fs))*np.cos(tb*(2*np.pi*fs))
        notch = noisefs
        Q = 90
        wv = event(rise=0.0005, decay=0.004, dur=0.020, amp=-1.0, rate=samprate)
        iwv = len(wv)
        rng = np.random.exponential(scale=0.1, size=50)
        evts = np.cumsum(rng)
        for te in evts:
            ist = int(te/samprate)
            if ist+iwv < len(data):
                data[ist:ist+iwv] += wv
        winpts = int(1/samprate)
        hwin = int(winpts/2)
        window = scipy.signal.windows.hann(winpts, 1)# , at=100)
        twin = np.concatenate((window[:hwin], np.ones(npts-winpts), window[hwin:]))
        data = data*twin
        datan = MI.NotchFilterComb(data=data, notchfreqs=notch, notchQ=Q)
        datan = datan[hwin:-hwin]
        tbn = tb[hwin:-hwin]

        if plot:
            print(f"Max difference: {np.max(datan-data[hwin:-hwin]):.6e}")
            f, ax = mpl.subplots(2,1)
            ax[0].plot(tb, data, 'k-')
            ax[0].plot(tbn, datan, 'r--')
            ax[1].magnitude_spectrum(data, Fs=1./samprate, color='k')
            ax[1].magnitude_spectrum(datan, Fs=1./samprate, color='r')
            mpl.show()
        return datan
        
if __name__ == '__main__':
    NT = NotchTester()
    NT.run_test()