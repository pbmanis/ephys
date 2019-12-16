"""
Old implementations of the CB and AJ algorithms.
CB implementation is a slow python implementation
This code has been superseded by the mini_methods.py module,
which includes cython and numba jit methods that are faster.

"""
import numpy as np
import scipy.signal
import numpy.random
import matplotlib.pyplot as mpl

class ClementsBekkers():
    def __init__(self):
        pass
    
    def clements_bekkers(self, data, template):
        """
        Implements Clements-bekkers algorithm: slides template across data,
        returns array of points indicating goodness of fit.
        Biophysical Journal, 73: 220-229, 1997.
        """
    
        ## Strip out meta-data for faster computation
        D = data.view(np.ndarray)
        T = template.view(np.ndarray)
    
        ## Prepare a bunch of arrays we'll need later
        N = len(T)
        sumT = T.sum()
        sumT2 = (T**2.0).sum()
        D2 = D**2.0
        NDATA = len(data)
        crit = np.zeros(NDATA)

        sumD = np.zeros((NDATA-N))
        sumD2 = np.zeros((NDATA-N))
        sumDTprod = np.zeros((NDATA-N))
        sumD[0] = D[:N].sum()
        sumD2[0] = D2[:N].sum()
        # sumD = rollingSum(D[:N], N)
        # sumD2 = rollingSum(D2[:N], N)
        for i in range(NDATA-N):
            if i > 0:
                sumD[i] = sumD[i-1] + D[i+N] - D[i]
                sumD2[i] = sumD2[i-1] + D2[i+N] - D2[i]
            sumDTprod[i] = (D[i:N+i]*T).sum()
        S = (sumDTprod - sumD*sumT/N)/(sumT2 - sumT*sumT/N)
        C = (sumD - S*sumT)/N
        SSE = sumD2 + (S*S*sumT2) + (N*C*C) - 2.0*(S*sumDTprod + C*sumD - (S*C*sumT))
        crit = S/np.sqrt(SSE/(N-1))
        DC = S / crit
        return DC, S, crit

    def rollingSum(self, data, n):
        d1 = data.copy()
        d1[1:] = np.cumsum(d1[1:]) # d1[1:] + d1[:-1]  # integrate
        d2 = np.empty(len(d1) - n + 1, dtype=data.dtype)
        d2[0] = d1[n-1]  # copy first point
        d2[1:] = d1[n:] - d1[:-n]  # subtract
        return d2
    
    def clementsBekkers2(self, data, template):
        """Implements Clements-bekkers algorithm: slides template across data,
        returns array of points indicating goodness of fit.
        Biophysical Journal, 73: 220-229, 1997.
    
        Campagnola's version...
        """
    
        ## Strip out meta-data for faster computation
        D = data.view(np.ndarray)
        T = template.view(np.ndarray)
        NDATA = len(D)
        ## Prepare a bunch of arrays we'll need later
        N = len(T)
        sumT = T.sum()
        sumT2 = (T**2.).sum()
        sumD = self.rollingSum(D, N)
        sumD2 = self.rollingSum(D**2., N)
        sumTD = scipy.signal.correlate(D, T, mode='valid')
    
        ## compute scale factor, offset at each location:
        scale = (sumTD - sumT * sumD /N) / (sumT2 - sumT**2. /N)
        offset = (sumD - scale * sumT) /N
    
        ## compute SSE at every location
        SSE = sumD2 + scale**2.0 * sumT2 + N * offset**2. - 2. * (scale*sumTD + offset*sumD - scale*offset*sumT)
        ## finally, compute error and detection criterion
        error = np.sqrt(SSE / (N-1))
        DC = scale / error
        return DC, scale, offset
    
    def cbTemplateMatch(self, data, template, threshold=3.0):
        dc, scale, crit = self.clementsBekkers(data, template)
        mask = crit > threshold
        diff = mask[1:] - mask[:-1]
        times = np.argwhere(diff==1)[:, 0]  ## every time we start OR stop a spike
    
        ## in the unlikely event that the very first or last point is matched, remove it
        if abs(crit[0]) > threshold:
            times = times[1:]
        if abs(crit[-1]) > threshold:
            times = times[:-1]
    
        nEvents = len(times) / 2
        result = np.empty(nEvents, dtype=[('peak', int), ('dc', float), ('scale', float), ('offset', float)])
        for i in range(nEvents):
            i1 = times[i*2]
            i2 = times[(i*2)+1]
            d = crit[i1:i2]
            p = np.argmax(d)
            result[i][0] = p+i1
            result[i][1] = d[p]
            result[i][2] = scale[p+i1]
            result[i][3] = crit[p+i1]
        return result, crit

    def make_template(self, tau_1, tau_2, tmax, dt):
        t_psc = np.arange(0, tmax, dt)
        Aprime = (tau_2/tau_1)**(tau_1/(tau_1-tau_2))
        g = 1./Aprime * (-np.exp(-t_psc/tau_1) + np.exp((-t_psc/tau_2)))
        return g

    def plot_summary(self):
        pass


class AndradeJonas(object):
    def __init__(self):
        pass

    def make_template(self, tau_1, tau_2, tmax, dt, sign=1):
        t_psc = np.arange(0, tmax+dt, dt)
        Aprime = (tau_2/tau_1)**(tau_1/(tau_1-tau_2))
        self.template = 1./Aprime * (-np.exp(-t_psc/tau_1) + np.exp((-t_psc/tau_2)))
        self.sign = sign
        if sign == -1:
            self.template = -self.template
        
    def deconvolve(self, data, tmax, dt=0.1, thresh=1, llambda=5.0, order=7, events=None):
        # Weiner filter deconvolution
        self.timebase = np.arange(0., data.shape[0]*dt, dt)
        H = np.fft.fft(self.template[:data.shape[0]])
        print(self.timebase.shape)
        print(self.template.shape)
        print(H.shape)
        print(data.shape)
        self.quot = np.real(np.fft.ifft(np.fft.fft(data)*np.conj(H)/(H*np.conj(H) + llambda**2)))
        sd = np.std(self.quot)
        self.sdthr = sd * thresh  # set the threshold
        # threshold (all values below sdthr become 0)
        self.above = scipy.stats.threshold(self.quot, self.sdthr)
        self.onsets = scipy.signal.argrelextrema(self.above, np.greater, order=order)[0] - 1

    def plots(self, tmax, dt, data, events=None):
        fig, ax = mpl.subplots(4, 1)
        for i in range(1,3):
            ax[i].get_shared_x_axes().join(ax[i], ax[0])
        tb = self.timebase
        ax[0].plot(tb, data)  # original data
        ax[0].plot(tb[self.onsets], data[self.onsets], 'y^')
        if events is not None:
            ax[0].plot(tb[events], data[events], 'ro', markersize=5)
        mwin = int(2*(4.)/dt)
        order = int(4/dt)
        ax[1].plot(tb[:self.quot.shape[0]], self.quot)  # deconvolution
        ax[1].plot([tb[0],tb[-1]], [self.sdthr, self.sdthr], 'r--', linewidth=0.75)
        ax[1].plot(tb[self.onsets], self.quot[self.onsets], 'y^')
        if events is not None:  # original events
            peaks = []
            amps = []
            for j in range(len(data[self.onsets])):
                 p =  scipy.signal.argrelextrema(self.sign*data[self.onsets[j]:(self.onsets[j]+mwin)], np.greater, order=order)[0]
                 if len(p) > 0:
                     peaks.extend([int(p[0]+self.onsets[j])])
                     amp = self.sign*data[peaks[-1]] - self.sign*data[self.onsets[j]]
                     amps.extend([amp])
            ax[0].plot(tb[peaks], data[peaks], 'mo', markersize=4)
        ax[2].plot(tb[:self.above.shape[0]], self.above)
        ax[2].plot([tb[0],tb[-1]], [self.sdthr, self.sdthr], 'r--', linewidth=0.75)
        # compute average event with length of template
        npts = int(55./dt)+1
        npre = int(5./dt)
        avg = np.zeros(npts)
        nev = 0  # count events

        for i in self.onsets:
            if (i + npts) < len(data) and (i - npre) >= 0:
                avg = avg + data[i-npre:i+npts-npre]
                nev = nev + 1
        avg = avg/nev
        ttb = np.arange(0, dt*npts, dt)
        ax[3].plot(ttb, avg)
        maxa = np.max(self.sign*avg)
        ax[3].plot(ttb+5., self.template[0:npts]*maxa, 'r-')
        mpl.show()
        
def generate_testdata(meanrate=10, amp=20., ampvar=5., noise=2.5):
    maxt = 1e4
    mean_rate = meanrate # Hz
    tdur = 100.
    dt = 0.1 # ms
    timebase = np.arange(0., maxt, dt) # in ms
    t_psc = np.arange(0., tdur, dt)  # in ms
    tau_1 = 1.0 # ms
    tau_2 = 10.0 # ms
    Apeak = amp # pA
    Aprime = (tau_2/tau_1)**(tau_1/(tau_1-tau_2))
    g = Apeak/Aprime * (-np.exp(-t_psc/tau_1) + np.exp((-t_psc/tau_2)))
    testpsc = np.zeros(timebase.shape)
    eventintervals = np.random.exponential(1e3/mean_rate, 1000)
    events = np.cumsum(eventintervals)
    t_events = events[events < maxt]  # time of events with exp distribution
    i_events = np.array([int(x/dt) for x in t_events])
    testpsc[i_events] = np.random.normal(1., ampvar/amp, len(i_events))
    i_events = i_events-int((tdur/2.)/dt)
    testpsc = scipy.signal.convolve(testpsc, g, mode='same')
    if noise > 0:
        testpscn = testpsc + np.random.normal(0., noise, testpsc.shape)
    else:
        testpscn = testpsc
    return timebase, testpsc, testpscn, i_events


def cb_tests():
    """
    Do some tests of the CB protocol and plot
    """
    timebase, testpsc, testpscn, ievents = generate_testdata()
    cb = ClementsBekkers()
    template = cb.make_template(1, 10., 100., 0.1)
    dc, s, crit = cb.clements_bekkers(tdn, template)
    fig, ax = mpl.subplots(4, 1)
    ax[0].plot(tb, tdn)
    print(tb.shape)
    print(dc.shape)
    ax[1].plot(tb[:dc.shape[0]], dc)
    ax[2].plot(tb[:s.shape[0]], s)
    ax[3].plot(tb[:crit.shape[0]], crit)
    cx = [crit > 2.0]
    ax[0].plot(tb[cx], tdn[cx], 'r')
    mpl.show()

def aj_tests():
    dt = 0.1
    timebase, testpsc, testpscn, i_events = generate_testdata(amp=20., ampvar=5., noise=5.0)
    aj = AndradeJonas()
    template = aj.make_template(1, 10., np.max(timebase), dt)
    aj.deconvolve(testpscn, np.max(timebase), dt=dt, thresh=3.0, events=i_events, llambda=5., order=7)
    aj.plots(np.max(timebase), dt, testpscn, events=i_events)
    

if __name__ == "__main__":
    aj_tests()
    #    cb_tests()