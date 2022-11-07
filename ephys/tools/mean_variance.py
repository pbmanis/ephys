"""
non-stationary noise analysis

"""

from random import random, seed
from scipy.interpolate import UnivariateSpline
import matplotlib.pyplot as mpl
import numpy as np
import pandas as pd
from lmfit.models import QuadraticModel, LinearModel
from statsmodels.nonparametric.smoothers_lowess import lowess

class NSFA():
    """Non stationary fluctuation analysis
    Traynelis et al 1993
    Silver, Cull-Candy and Takahashi, 1996

    Returns
    -------
    _type_
        _description_
    """
    def __init__(self, timebase:np.ndarray, eventtraces:np.ndarray, eventtimes:np.ndarray):
        self.timebase = timebase
        self.dt = np.mean(np.diff(self.timebase))
        self.events = eventtraces # the actual traces of individual "cut out" events, each of length timebase
        self.eventtimes = eventtimes # indicies to the peak times of events

    def align_on_rising(self, prewindow:float=0.007, postwindow=0.025, lowess_frac:float=0.08):
        """Align the traces on the rising slope of an event.

        Parameter
        ---------
        prewindow : float (time, seconds)
            window to look for max rising 
        Returns
        -------
        list of ints
            indices to the maximum rising slope before the peak
        """
        sl_pre = int(prewindow/self.dt)
        ev_post = int(postwindow/self.dt)
        self.max_slope_pts = np.zeros_like(self.events)
        res = []
        d = []
        t = []
        event_indx = []
        maxsl = []
        deriv = []
        for itr in range(self.events.shape[0]):
            # print(self.eventtimes[itr])
            event_indices = list(range(self.eventtimes[itr]-sl_pre,self.eventtimes[itr]+ev_post))
            event_indx.append(event_indices)
            rise  = self.events[itr][event_indices]
            rise_t = self.timebase[event_indices]
            d.append(rise)
            t.append(rise_t) 
            #rise_sm = lowess(rise, rise_t, frac = lowess_frac, return_sorted=False)
            x = []
            n = 5
            ilag = 0
            slmax= 0.0
            lm = LinearModel()
            for i in range(1,len(rise)):
                lm = LinearModel()
                if i < n:
                    s = lm.fit(x=rise_t[ilag:i], y=rise[ilag:i])
                elif i > len(rise) - n:
                    s = lm.fit(x=rise_t[i:n], y=rise[i:n])

                else:
                    ilag = i - int(np.floor(ilag/2))
                    ilead = i + int(np.floor(ilag/2))
                    s = lm.fit(x = rise_t[ilag:ilead], y = rise[ilag:ilead])
                if s['m'] > slmax:
                    slmax = s['m']
                deriv[i] = s['m']


            # rise_sm = UnivariateSpline(rise_t, rise, k=2)
            # res.append(rise_sm(rise_t))

            # dvdt_grad = np.gradient(res[-1], self.dt)
            # gradmax = np.argmax(dvdt_grad)
            deriv.append(dvdt_grad)
            # maxsl.append(gradmax)
            maxsl.append(slmax)


        f, ax = mpl.subplots(3,1, figsize=(8, 9))
        c = ['r', 'g', 'b', 'y', 'm']
        for itr in range(self.events.shape[0]):
            colr = c[itr%(len(c))]
            ax[0].plot(t[itr], d[itr], color=colr) # self.events[itr][event_indices])

            ax[1].plot(t[itr]-self.dt*maxsl[itr], d[itr], color=colr, linewidth=0.25) # self.events[itr][event_indices])
            ax[1].plot(t[itr]-self.dt*maxsl[itr], res[itr], '--', color=colr)
            ax[2].plot(t[itr]-self.dt*maxsl[itr], deriv[itr], '-', color=colr)
            # ax[1].plot(t[itr][int(maxsl[itr]*self.dt)],
            #      res[itr][int(maxsl[itr]*self.dt)], 'kx')
        ax[1].sharex(ax[0])
        ax[2].sharex(ax[0])
        mpl.show()

    
class TestGenerator():
    def __init__(self):
        self.set_transistions()
        self.dt = 0.0001
        self.dur = 0.10
        self.delay = 0.005
        self.delay_tau = 0.001
        self.npts = int(self.dur/self.dt)
        self.time = np.arange(0, self.dt*self.npts, self.dt)
        self.n_trials = 3
        self.n_chans = 10000 # openings in one event
        self.i_chan = 5
        self.noise_var = 0.0 
        self.max_prob = 0.6

    def set_transistions(self):
        self.alpha = 0.0025 # opening rate
        self.beta = 0.005 # closing rate
        self.d0 = 0.0001

    def generate_sweeps(self):
        delay = np.random.normal(scale=self.delay_tau, size=(self.n_chans, self.n_trials))+self.delay
        dels = np.random.exponential(scale = self.alpha, size=(self.n_chans, self.n_trials)) + delay
        durs = np.random.exponential(scale = self.beta,  size=(self.n_chans, self.n_trials)) + delay
        probs = np.random.uniform(size=(self.n_chans, self.n_trials))
        idels = dels/self.dt
        idurs = durs/self.dt
        sweeps = np.zeros((self.n_trials, self.npts))
        for i, s in enumerate(range(sweeps.shape[0])):
            sweeps[i] = np.random.normal(0., scale=self.noise_var, size=self.npts)
            for n in range(self.n_chans):
                if probs[n, i] < self.max_prob:
                    sweeps[i, int(idels[n, i]):(int(idels[n, i] + idurs[n, i]))] += self.i_chan

        return sweeps
    

    def fit_meanvar(self):
        qmod = QuadraticModel()

        pars = qmod.guess(self.var, x=self.mean)
        qfit = qmod.fit(self.var, pars, x=self.mean)
        qfit.params['c'].set( vary=False, value=0.0)
        print(qfit.fit_report(min_correl=0.25))
        """fit parameters:
        y = a*x**2 + b*x + c

        var = -(I**2)/N + i_chan * I
        c should be set to 0
        Therfore, a = -1/N (N = -1/a), and b = i_chan

        """
        NChan = -1.0/qfit.values['a']
        gamma = qfit.values['b']
        Pomax = np.max(self.var)/ (2* gamma * NChan)

        print(f"nchan = {NChan:.3f}  gamma = {gamma:.3e}, Pomax = {Pomax:.2f}")

        return(qfit)

    def run(self):
        sweeps = self.generate_sweeps()
        self.sweeps = sweeps
        f, ax = mpl.subplots(3,1)
        for i in range(np.min((2, sweeps.shape[0]))):
            ax[0].plot(self.time, sweeps[i] + self.i_chan*1.2*i)
        self.mean = np.mean(sweeps, axis=0)
        self.var = np.var(sweeps, axis=0)
        ax[1].plot(self.time, self.mean)
        ax[1].plot(self.time, self.var)
        ax[2].scatter(self.mean, self.var ,s=12)
        qfit = self.fit_meanvar()
        ax[2].plot(self.mean, qfit.best_fit, 'r--')
        mpl.show()


if __name__ == "__main__":

    tg = TestGenerator()
    sweeps = tg.generate_sweeps()
    # tg.run()
    NSA = NSFA(timebase=tg.time, eventtraces=sweeps, eventtimes= [np.argmax(tr) for tr in sweeps])
    NSA.align_on_rising()
    mpl.show()
