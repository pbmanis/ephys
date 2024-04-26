"""
Provide and test non-stationary noise analysis

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


class NSFA:
    """Non stationary fluctuation analysis
    Traynelis et al 1993
    Silver, Cull-Candy and Takahashi, 1996

    Returns
    -------
    _type_
        _description_
    """

    def __init__(self):
        pass
    
    def setup(self, timebase: np.ndarray, eventtraces: np.ndarray, eventpeaktimes: np.ndarray):
        self.timebase = timebase
        self.dt = np.mean(np.diff(self.timebase))
        self.events = (
            eventtraces  # the actual traces of individual "cut out" events, each of length timebase
        )
        self.eventtimes = eventpeaktimes  # indicies to the peak times of events
        self.d: np.ndarray = np.empty(0)
        self.t: float = 0.0
        self.meantr: float = 0.0
        self.max_slope_pts: np.ndarray = np.empty(0)

    def linear_slope(self, x: np.ndarray, y: np.ndarray, N: int = 3) -> np.array:
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
        derivs = np.nan * np.zeros_like(y)
        left = int((N - 1) / 2)
        right = int((N + 1) / 2)
        for i in range(y.shape[0]):
            il = i - left
            ir = i + right + 1
            if il < 0:
                il = 0
                ir = 3
            if ir > y.shape[0]:
                ir = y.shape[0]
                il = y.shape[0] - 3
            res = linregress(x[il:ir], y[il:ir])
            derivs[i] = res.slope
        return derivs

    def align_on_rising(
        self, prewindow: float = 0.003, postwindow=0.03, Nslope: int = 7, plot: bool = False
    ):
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
        d = []
        t = []
        maxsl = []
        deriv = []
        for itr in range(self.events.shape[0]):
            rise = self.events[itr]
            rise_t = self.timebase
            d.append(rise)
            t.append(rise_t)
            if rise.shape != rise_t.shape:
                print(rise.shape, rise_t.shape)
            slope = self.linear_slope(rise_t, rise, N=Nslope)
            deriv.append(slope)
            maxsl.append(np.argmax(slope))

        mint = 1000.0
        maxt = -1000.0
        for itr in range(self.events.shape[0]):
            align_i = int(maxsl[itr])
            align_t = self.dt * align_i
            tx = t[itr] - align_t
            if np.min(tx) < mint:
                mint = np.min(tx)
            if np.max(tx) > maxt:
                maxt = np.max(tx)
        tnew = np.arange(mint, maxt, self.dt)
        for itr in range(self.events.shape[0]):
            align_i = int(maxsl[itr])
            align_t = self.dt * align_i
            d[itr] = np.interp(tnew, t[itr] - align_t, d[itr])
        meantr = np.mean(np.array(d), axis=0)
        self.d = np.array(d)
        self.t = tnew
        self.meantr = meantr
        # mpl.plot(tnew, meantr)
        # mpl.show()

        if not plot:
            return
        f, ax = mpl.subplots(3, 1, figsize=(8, 9))
        c = ["r", "g", "b", "y", "m"]
        for itr in range(self.events.shape[0]):
            colr = c[itr % (len(c))]
            align_i = int(maxsl[itr])
            align_t = self.dt * align_i
            ax[0].plot(self.timebase, self.events[itr]/np.max(self.events[itr]), color=colr, linewidth=0.33)
            ax[0].plot(
                self.timebase[self.eventtimes[itr]],
                self.events[itr][self.eventtimes[itr]]/np.max(self.events[itr][self.eventtimes[itr]]),
                "x",
                color=colr,
                markersize=4,
            )
            ax[0].plot(t[itr][align_i], self.events[itr][align_i], "x", color=colr, markersize=4)
            m = t[itr].shape[0]
            # if t[itr].shape != deriv[itr].shape:
            #     print("unmatched shapes: deriv",  t[itr].shape, deriv[itr].shape)
            # else:
            ax[2].plot(t[itr], deriv[itr][:m], "-", color=colr, linewidth=0.33)
            # if t[itr].shape != d[itr].shape:
            #     print("unmatched shapes: d",  t[itr].shape, d[itr].shape)
            # else:
            ax[1].plot(
                    t[itr] - align_t, d[itr][:m], color=colr, linewidth=0.33
                )  # self.events[itr][event_indices])


            ax[2].plot(t[itr][align_i], deriv[itr][align_i], "x", color=colr, markersize=4)
        yl1 = ax[1].get_ylim()
        ax[0].set_xlim((tnew[0], tnew[-1]))
        ax[1].plot([0.0, 0.0], yl1, "k-", linewidth=0.25, alpha=0.5)
        ax[1].sharex(ax[0])
        yl2 = ax[2].get_ylim()
        ax[2].plot([0.0, 0.0], yl2, "k-", linewidth=0.25, alpha=0.5)
        ax[2].sharex(ax[0])
        ax[0].set_title("PSCs")
        ax[1].set_title("max slope aligned PSCs at 0 time")
        ax[2].set_title("Aligned derivative of PSCs")
        mpl.show()

    def fit_meanvar(self, mean: np.ndarray, var: np.ndarray):
        qmod = QuadraticModel()

        pars = qmod.guess(data=var, x=mean)
        pars['c'].set(value=0.0, vary=False)
        print(pars)
        qfit = qmod.fit(var, pars, x=mean)

        mpl.plot(mean, var, "o", color="k", markersize=2)
        mpl.show()
        print(qfit.fit_report(min_correl=0.25))
        """fit parameters:
        y = a*x**2 + b*x + c

        var = -(I**2)/N + i_chan * I
        c should be set to 0
        Therfore, a = -1/N (N = -1/a), and b = i_chan

        """
        NChan = -1.0 / qfit.values["a"]
        gamma = qfit.values["b"]
        Pomax = np.max(var) / (2.0 * gamma * NChan)

        print(f"nchan = {NChan:.3f}  \u03B3 = {gamma*1e12:.3f} pA, Pomax = {Pomax:.3e}")
        return qfit

    def plot(self, qfit, time, sweeps, i_chan=None, mean: np.ndarray = None, var: np.ndarray = None):
        f, ax = mpl.subplots(3, 1)
        if i_chan is not None:
            for i in range(np.min((100, np.array(sweeps).shape[0]))):
                ax[0].plot(time*1e3, sweeps[i]*1e12 + i_chan*1e12, linewidth=0.25)
        ax[0].set_ylabel("pA")

        if mean is None and var is None:
            print("Mean and var ar none!!!")
            return f, ax
        ax[1].plot(time*1e3, mean*1e12, "b-", linewidth=0.5)
        ax2 = ax[1].twinx()
        imax_i = np.argmax(mean)
        ax2.plot(time*1e3, var, "-", color="grey", linewidth=0.5)
        ax2.plot(time[imax_i:]*1e3, var[imax_i:]*1e24, "-", color="c", linewidth=0.75)
        ax2.set_ylabel("Variance (pA^2)")
        ax[1].set_ylabel("Mean (pA)")

        ax2.set_ylim((0, np.max(var*1e24)))
        imax_i = np.argmax(mean)
        qfit = self.fit_meanvar(mean=mean[imax_i:], var=var[imax_i:])
        ax[2].scatter(mean[imax_i:]*1e12, var[imax_i:]*1e24, c="k", s=6)
        ax[2].plot(mean[imax_i:]*1e12, qfit.best_fit*1e24, "r--", linewidth=0.5)
        ax[2].set_ylabel("Variance (pA^2)")
        ax[2].set_xlabel("Mean (pA)")
        mpl.show()
        exit()
        return f, ax



class TestGenerator:
    def __init__(self, nchans: int = 20, ichan: float = 12e-12, ntrials: int = 400, dt: float = 0.0001):
        self.set_transistions()
        self.dt = dt
        self.dur = 0.10  # whole event duration
        self.delay = 0.005  # delay to start of event
        self.delay_tau = 0.001
        self.npts = int(self.dur / self.dt)
        self.time = np.arange(0, self.dt * self.npts, self.dt)
        self.n_trials = ntrials
        self.n_chans = nchans  # openings in one event
        self.i_chan = ichan  # channel current
        self.noise_var = 3e-12
        self.i_chan_var = 0e-13
        self.max_prob = 0.6
        self.rng = default_rng(314159)

    def set_transistions(self):
        """Set forward and backward rates for channel c <-> o"""
        self.alpha = 0.002  # opening rate
        self.beta = 0.010  # closing rate
        self.d0 = 0.0001

    def generate_sweeps(self):
        # use gamma as it has minimumum time
        delay = 0.005  # self.rng.gamma(shape=2., scale=self.delay_tau, size=(self.n_chans, self.n_trials))+self.delay
        dels = self.rng.exponential(scale=self.alpha, size=(self.n_chans, self.n_trials)) + delay
        durs = self.rng.exponential(scale=self.beta, size=(self.n_chans, self.n_trials))
        chans = self.rng.normal(
            loc=self.i_chan, scale=self.i_chan_var, size=(self.n_chans, self.n_trials)
        )
        probs = self.rng.uniform(size=(self.n_chans, self.n_trials))
        idels = dels / self.dt
        idurs = durs / self.dt
        sweeps = np.zeros((self.n_trials, self.npts))
        for i, s in enumerate(range(sweeps.shape[0])):
            sweeps[i] = self.rng.normal(0.0, scale=self.noise_var, size=self.npts)
            for n in range(self.n_chans):
                if probs[n, i] < self.max_prob:
                    sweeps[i, int(idels[n, i]) : (int(idels[n, i] + idurs[n, i]))] += chans[n, i]
            sweeps[i] = DF.SignalFilter_LPFButter(
                sweeps[i] - np.mean(sweeps[i][:10]), 3000.0, 1.0 / self.dt, NPole=8
            )
        return sweeps

    
    def run(self, NSFA):
        sweeps = self.generate_sweeps()
        self.sweeps = sweeps
        self.mean = np.mean(sweeps, axis=0)
        self.var = np.var(sweeps, axis=0)
        qfit = NSFA.fit_meanvar(self.mean, self.var)
        NSFA.plot(qfit, self.sweeps, NSFA.i_chan)

    


if __name__ == "__main__":
    ichan = 12e-12
    nchans = 20
    tg = TestGenerator(nchans=nchans, ichan=ichan, ntrials=400, dt=1e-4)
    sweep_data = tg.generate_sweeps()
    # tg.run()
    NSA = NSFA(timebase=tg.time, eventtraces=sweep_data, eventtimes=[np.argmax(tr) for tr in sweep_data])
    NSA.align_on_rising(Nslope=7)
    meanI = np.mean(NSA.d, axis=0)
    varI = np.var(NSA.d, axis=0)
    NSA.mean = meanI
    NSA.var = varI
    qfit = NSA.fit_meanvar(mean=meanI, var=varI)
    f, ax = NSA.plot(qfit, NSA.t, NSA.d, mean=meanI, var=varI)
    print(f"NChan original: {nchans:3d}, gamma_original: {ichan*1e12:.3f} pA")
    mpl.show()
