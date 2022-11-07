"""
non-stationary noise analysis

"""

from numpy.random import default_rng
import matplotlib.pyplot as mpl
import numpy as np
import pandas as pd
from typing import Tuple
from lmfit.models import QuadraticModel, LinearModel
from statsmodels.nonparametric.smoothers_lowess import lowess
from scipy.interpolate import UnivariateSpline
from scipy.stats import linregress
import digital_filters as DF

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

    def linear_slope(self, x: np.ndarray, y:np.ndarray, N:int=3) -> np.array:
        """Compute the slope of an array by fitting segments of
        points of length N (N must be odd), and returning the 
        value and the index of the maximum slope.
        This function provides local smoothing.

        statsmodel.linregress is rumored to be the fastest way
        to compute the regression.

        Parameters
        ----------
        x: np.ndarray
            x values (time)
        y : np.ndarray
            array of data to test
        N : int, optional
            number of points in the linear segments, by default 3

        Returns
        -------
        the computed slope at each point.

        """
        assert (N >= 3) and ((N % 2) == 1)
        derivs = np.nan*np.zeros_like(y)
        left = int((N-1)/2)
        right = int((N+1)/2)
        for i in range(y.shape[0]):
            il = i - left
            ir = i + right + 1
            if il < 0:
                il = 0
                ir = 3
            if ir > y.shape[0]:
                ir = y.shape[0]
                il = y.shape[0]-3
            res = linregress(x[il:ir], y[il:ir])
            derivs[i] = res.slope
        return derivs

    def align_on_rising(self, prewindow:float=0.030, postwindow=0.03, Nslope:int=7, plot:bool=False):
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
        self.max_slope_pts = np.zeros_like(self.events)
        res = []
        d = []
        t = []
        maxsl = []
        deriv = []
        for itr in range(self.events.shape[0]):
            rise = self.events[itr]
            rise_t = self.timebase
            d.append(rise)
            t.append(rise_t)
            slope = self.linear_slope(rise_t, rise, N=Nslope)
            deriv.append(slope)
            maxsl.append(np.argmax(slope))

        mint = 1000.0
        maxt = -1000.0
        for itr in range(self.events.shape[0]):
            align_i = int(maxsl[itr])
            align_t = self.dt*align_i
            tx = t[itr]-align_t
            if np.min(tx) < mint:
                mint = np.min(tx)
            if np.max(tx) > maxt:
                maxt = np.max(tx)
        tnew = np.arange(mint, maxt, self.dt)
        for itr in range(self.events.shape[0]):
            align_i = int(maxsl[itr])
            align_t = self.dt*align_i
            d[itr] = np.interp(tnew, t[itr]-align_t, d[itr])
        meantr = np.mean(np.array(d), axis=0)
        self.d = np.array(d)
        self.t = tnew
        self.meantr = meantr
        # mpl.plot(tnew, meantr)
        # mpl.show()


        if not plot:
            return
        f, ax = mpl.subplots(3,1, figsize=(8, 9))
        c = ['r', 'g', 'b', 'y', 'm']
        for itr in range(self.events.shape[0]):
            colr = c[itr%(len(c))]
            align_i = int(maxsl[itr])
            align_t = self.dt*align_i
            ax[0].plot(self.timebase, self.events[itr], color=colr, linewidth=0.33)
            ax[0].plot(self.timebase[self.eventtimes[itr]],
                self.events[itr][self.eventtimes[itr]], "x", color=colr, markersize=4,
                )
            ax[0].plot(t[itr][align_i],
                       self.events[itr][align_i], 'x', color=colr, markersize=4)
            ax[1].plot(t[itr]-align_t, d[itr], color=colr, linewidth=0.33) # self.events[itr][event_indices])
            ax[2].plot(t[itr], deriv[itr], '-', color=colr, linewidth=0.33)
            ax[2].plot(t[itr][align_i],
                       deriv[itr][align_i], 'x', color=colr, markersize=4)
        yl1 = ax[1].get_ylim()
        ax[1].plot([0., 0.], yl1, 'k-', linewidth=0.25, alpha = 0.5)
        ax[1].sharex(ax[0])
        yl2 = ax[2].get_ylim()
        ax[2].plot([0., 0.], yl2, 'k-', linewidth=0.25, alpha = 0.5)
        ax[2].sharex(ax[0])
        ax[0].set_title("PSCs")
        ax[1].set_title("max slope aligned PSCs at 0 time")
        ax[2].set_title("Aligned derivative of PSCs")
        mpl.show()

    
class TestGenerator():
    def __init__(self):
        self.set_transistions()
        self.dt = 0.0001
        self.dur = 0.10  # whole event duration
        self.delay = 0.005  # delay to start of event
        self.delay_tau = 0.001
        self.npts = int(self.dur/self.dt)
        self.time = np.arange(0, self.dt*self.npts, self.dt)
        self.n_trials = 100
        self.n_chans = 50 # openings in one event
        self.i_chan = 5e-12
        self.noise_var = 3e-12 
        self.i_chan_var = 0e-13
        self.max_prob = 0.6
        self.rng = default_rng(314159)

    def set_transistions(self):
        """Set forward and backward rates for channel c <-> o
        """        
        self.alpha = 0.002 # opening rate
        self.beta = 0.010 # closing rate
        self.d0 = 0.0001

    def generate_sweeps(self):
        # use gamma as it has minimumum time
        delay = 0.005 # self.rng.gamma(shape=2., scale=self.delay_tau, size=(self.n_chans, self.n_trials))+self.delay
        dels = self.rng.exponential(scale = self.alpha, size=(self.n_chans, self.n_trials)) + delay
        durs = self.rng.exponential(scale = self.beta,  size=(self.n_chans, self.n_trials))
        chans = self.rng.normal(loc=self.i_chan, scale=self.i_chan_var, size=(self.n_chans, self.n_trials))
        probs = self.rng.uniform(size=(self.n_chans, self.n_trials))
        idels = dels/self.dt
        idurs = durs/self.dt
        sweeps = np.zeros((self.n_trials, self.npts))
        for i, s in enumerate(range(sweeps.shape[0])):
            sweeps[i] = self.rng.normal(0., scale=self.noise_var, size=self.npts)
            for n in range(self.n_chans):
                if probs[n, i] < self.max_prob:
                    sweeps[i, int(idels[n, i]):(int(idels[n, i] + idurs[n, i]))] += chans[n, i]
            sweeps[i] = DF.SignalFilter_LPFButter(sweeps[i]-np.mean(sweeps[i][:10]), 3000.0, 1./self.dt, NPole=8)
        return sweeps
    

    def fit_meanvar(self, mean:np.ndarray, var:np.ndarray):
        qmod = QuadraticModel()

        pars = qmod.guess(data=var, x=mean)
        qfit = qmod.fit(var, pars, x=mean)
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
        Pomax = np.max(var)/ (2.0 * gamma * NChan)

        print(f"nchan = {NChan:.3f}  gamma = {gamma:.3e}, Pomax = {Pomax:.2f}")

        return(qfit)

    def run(self):
        sweeps = self.generate_sweeps()
        self.sweeps = sweeps
        self.mean = np.mean(sweeps, axis=0)
        self.var = np.var(sweeps, axis=0)
        qfit = self.fit_meanvar(self.mean, self.var)
        self.plot(qfit, self.sweeps)
        
    
    def plot(self, qfit, time, sweeps):
        f, ax = mpl.subplots(3,1)
        for i in range(np.min((100, np.array(sweeps).shape[0]))):
            ax[0].plot(time, sweeps[i] + self.i_chan, linewidth=0.25)
        
        ax[1].plot(time, self.mean, 'b-', linewidth=0.5)
        ax2 = ax[1].twinx()
        imax_i = np.argmax(self.mean)
        ax2.plot(time, self.var, '-', color='grey', linewidth=0.5)
        ax2.plot(time[imax_i:], self.var[imax_i:], '-', color='c', linewidth=0.75)

        ax2.set_ylim((0, np.max(self.var)))
        imax_i = np.argmax(self.mean)
        qfit = self.fit_meanvar(mean=self.mean[imax_i:], var=self.var[imax_i:])
        ax[2].scatter(self.mean[imax_i:], self.var[imax_i:], c='k', s=9)
        ax[2].plot(self.mean[imax_i:], qfit.best_fit, 'r--')



if __name__ == "__main__":

    tg = TestGenerator()
    sweeps = tg.generate_sweeps()
    # tg.run()
    NSA = NSFA(timebase=tg.time, eventtraces=sweeps, eventtimes= [np.argmax(tr) for tr in sweeps])
    NSA.align_on_rising(Nslope=13)
    meanI = np.mean(NSA.d, axis=0)
    varI = np.var(NSA.d, axis=0)
    tg.mean=meanI
    tg.var=varI
    qfit = tg.fit_meanvar(mean= meanI, var = varI)
    tg.plot(qfit, NSA.t, NSA.d)
    mpl.show()
