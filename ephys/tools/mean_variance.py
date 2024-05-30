"""
Provide and test non-stationary noise analysis

"""

import sys
from numpy.random import default_rng
import pyqtgraph as pg
import numpy as np
import pandas as pd
from typing import Tuple
from lmfit.models import QuadraticModel, LinearModel
from statsmodels.nonparametric.smoothers_lowess import lowess
from scipy.interpolate import UnivariateSpline
from scipy.stats import linregress
import digital_filters as DF
import meanvartest as MVT


class NSFA:
    """Non stationary fluctuation analysis
    Traynelis et al 1993
    Silver, Cull-Candy and Takahashi, 1996

    Returns
    -------
    _type_
        _description_
    """

    def __init__(self, timebase=None, idata=None):
        self.tracetimebase = timebase
        self.tracedata = idata
        self.setupplots()

    def setup(self, timebase: np.ndarray, eventtraces: np.ndarray, eventpeaktimes: np.ndarray):
        self.timebase = timebase
        self.dt = np.mean(np.diff(self.timebase))
        self.events = (
            eventtraces  # the actual traces of individual "cut out" events, each of length timebase
        )
        self.eventpeaktimes = eventpeaktimes  # indicies to the peak times of events
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

    def rising_midpoint(self, x: np.ndarray, y: np.ndarray) -> int:
        """Find the midpoint of the rising slope of an event.

        Parameters
        ----------
        x : np.ndarray
            timebase
        y : np.ndarray
            array of data to test

        Returns
        -------
        float
            time of the midpoint of the rising phase (half-peak amplitude)
        """
        N = 3
        ymax = np.max(y)
        halfmax = ymax / 2.0
        i_half = int(np.argmin(np.abs(y - halfmax)))
        return i_half

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
        n_events = self.events.shape[0]
        self.max_slope_pts = np.zeros_like(self.events)
        t_rise = np.zeros((self.events.shape[0], self.timebase.shape[0]))
        d_rise = np.zeros((self.events.shape[0], self.timebase.shape[0]))
        maxsl = [0] * n_events  # indices of the maximum rising slope
        midpts = [0] * n_events  # indicies to the midpoint of rising phase
        deriv = np.zeros(
            (self.events.shape[0], self.timebase.shape[0])
        )  # derivative of the traces, on the new time base - list of np arrays

        # first get the indices for the slope, mid point, and compute the derivative.
        for itr in range(self.events.shape[0]):
            rise = self.events[itr] - self.events[itr][0]
            d_rise[itr, :] = rise
            t_rise[itr, :] = self.timebase
            if rise.shape != self.timebase.shape:
                raise ValueError(
                    f"Rising shape and timebase do not match: {rise.shape!s}, {rise_t.shape!s}"
                )
            slope = self.linear_slope(self.timebase, rise, N=Nslope)
            deriv[itr, :] = slope
            maxsl[itr] = int(np.argmax(slope))
            i_peak = np.argmax(rise)
            midpts[itr] = self.rising_midpoint(self.timebase[:i_peak], rise[:i_peak])
        # now compute a time base that can be used with the aligned data.
        # not all events will have the same window after alignment, so we need to provide an
        # extended time base
        mint = 0.0
        maxt = 0.0  # max shift range for events - 10 msec in this case
        for itr in range(self.events.shape[0]):  # each event
            tx = (
                t_rise[itr, :] - midpts[itr] * self.dt
            )  # align the timebase on the selected point (zero time)
            # compute the window min/max shifts
            if np.min(tx) < mint:
                mint = np.min(tx)
            if np.max(tx) > maxt:
                maxt = np.max(tx)
        tnew = np.arange(mint, maxt, self.dt)
        #  Now interpolate the data onto the new time base
        d = np.zeros((self.events.shape[0], tnew.shape[0]))
        for itr in range(self.events.shape[0]):
            d[itr] = np.interp(tnew, t_rise[itr, :] - midpts[itr] * self.dt, d_rise[itr, :])
        self.meantr = np.mean(np.array(d), axis=0)
        self.maxI = np.max(self.meantr)
        self.variance = np.var(np.array(d), axis=0)
        self.scaled = [self.maxI*np.array(d[i]) / np.max(np.array(d[i])) for i in range(self.events.shape[0])]
        self.d = np.array(d)
        self.t = tnew
        self.mean_peak_index = np.argmax(self.meantr)


        if not plot:
            return
        print(self.eventpeaktimes)
        evt_indices = np.array([int(i/self.dt) for i in self.eventpeaktimes])
        # plot the raw data - events.
        self.P0.plot(self.tracetimebase, self.tracedata, linewidth=0.5)
        line = self.P0.plot(
            x=self.tracetimebase[evt_indices],
            y=self.tracedata[evt_indices],
            symbol="o",
            pen=pg.mkPen("r"),
            # markersize=4,
        )
        line.setSymbolSize(4)
        self.P0.setXRange(0, np.max(self.tracetimebase))

        c = ["r", "g", "b", "y", "m"]
        for itr in range(self.events.shape[0]):
            
            colr = c[itr % (len(c))]
            line = self.P1.plot(
                self.timebase,
                self.events[itr] / np.max(self.events[itr]),
                pen=pg.mkPen(pg.intColor(itr)),
                linewidth=0.25,
            )
            line.setSymbolSize(3)
            pk_indices = np.array([int(i/self.dt) for i in self.eventpeaktimes])
            # line = self.P1.plot(
            #     x=self.eventpeaktimes[itr],
            #     y=[
            #         self.events[itr][pk_indices]
            #         / np.max(self.events[itr])
            #     ],
            #     symbol="+",
            #     pen=pg.mkPen("g"),
            #     # markersize=4,
            # )
            # line.setSymbolSize(3)
            line = self.P1.plot(
                x=[t_rise[itr][midpts[itr]]],
                y=[self.events[itr][midpts[itr]]],
                symbol="x",
                symbolPen="m",
            )
            line.setSymbolSize(3)
            m = tnew.shape[0]

            #  plot the aligned events.
            self.P2.plot(tnew, d[itr][:m], pen=pg.mkPen(pg.intColor(itr)), linewidth=0.25)

            # self.events[itr][event_indices])

            # if t[itr].shape != deriv[itr].shape:
            #     print("unmatched shapes: deriv",  t[itr].shape, deriv[itr].shape)
            # else:

            #  plot the derivative of the data
            # self.P3.plot(
            #     tnew, np.gradient(d[itr][:m], tnew), pen=pg.mkPen(pg.intColor(itr)), linewidth=0.33
            # )
            self.P3.plot(tnew, self.scaled[itr][:m], pen=pg.mkPen(pg.intColor(itr)), linewidth=0.25)
            self.P3A.plot(tnew, self.scaled[itr][:m] - self.meantr, pen=pg.mkPen(pg.intColor(itr)), linewidth=0.25)
            # if t[itr].shape != d[itr].shape:
            #     print("unmatched shapes: d",  t[itr].shape, d[itr].shape)
            # else:

            # line = self.P3.plot(
            #     x=[t_rise[itr][midpts[itr]]], y=[np.max(np.gradient(d[itr][:m], tnew))[midpts[itr]]], symbol="x", pen=pg.mkPen("r")
            # )
            # line.setSymbolSize(3)
        #  add mean line to aligned events.

        # plot mean trace
        self.P3.plot(self.t, self.meantr,
                     pen=pg.mkPen("black", width=4, style=pg.QtCore.Qt.PenStyle.SolidLine))
        self.P3.plot(self.t, self.meantr,
                     pen=pg.mkPen("red", width=4, style=pg.QtCore.Qt.PenStyle.DashLine))
        self.P4.plot(
            self.t,
            self.meantr,
            pen=pg.mkPen(0x88FF88, width=1.5, style=pg.QtCore.Qt.PenStyle.SolidLine),
        )
        # plot variance trace
        self.P5.plot(
            self.t,
            self.variance,
            pen=pg.mkPen(0x88FF88, width=1.5, style=pg.QtCore.Qt.PenStyle.SolidLine),
        )
        # yl1 = ax[1].get_ylim()
        # self.P0.setXRange(tnew[0], tnew[-1])
        # ax[1].plot([0.0, 0.0], yl1, "k-", linewidth=0.25, alpha=0.5)
        # ax[1].sharex(ax[0])
        # yl2 = ax[2].get_ylim()
        # ax[2].plot([0.0, 0.0], yl2, "k-", linewidth=0.25, alpha=0.5)
        # ax[2].sharex(ax[0])
        # ax[0].set_title("PSCs")
        # ax[1].set_title("max slope aligned PSCs at 0 time")
        # ax[2].set_title("Aligned derivative of PSCs")
        # mpl.show()
        self.win.show()

    def fit_meanvar(self, mean: np.ndarray, var: np.ndarray):
        qmod = QuadraticModel()

        pars = qmod.guess(data=var, x=mean)
        pars["c"].set(value=0.0, vary=False)
        print(pars)
        qfit = qmod.fit(var, pars, x=mean)

        # self.P4.plot(mean, var, "o", color="k", markersize=2)
        # self.PR.setLabel("left", "Variance (pA^2)")
        # mpl.show()
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

    def setupplots(self):
        self.win = pg.GraphicsLayoutWidget()
        self.win.setWindowTitle("NSFA")
        self.win.resize(800, 1200)
        self.P0 = self.win.addPlot(row=0, col=0, rowspan=1, colspan=2, title="PSCs")
        self.P0.setLabel("left", "Current (pA)")
        self.P0.setLabel("bottom", "Time (s)")
        self.P1 = self.win.addPlot(row=1, col=0, rowspan=2, colspan=2, title="PSCs")
        self.P1.setLabel("left", "Current (Normd)")
        self.P1.setLabel("bottom", "Time (s)")
        self.P2 = self.win.addPlot(
            row=3, col=0, rowspan=1, colspan=2, title="max slope aligned PSCs at 0 time"
        )
        self.P2.setLabel("left", "Current (pA)")
        self.P2.setLabel("bottom", "Time (s)")
        self.P3 = self.win.addPlot(
            row=4, col=0, rowspan=1, colspan=2, title="Scaled PSCs"
        )
        self.P3.setLabel("left", "Scaled I (pA)")
        self.P3.setLabel("bottom", "Time (s)")
        self.P3A = self.win.addPlot(row=5, col=0, rowspan=1, colspan=2, title="Difference")
        self.P3A.setLabel("left", "Scaled I (pA)")
        self.P3A.setLabel("bottom", "Time (s)")
        self.P4 = self.win.addPlot(row=6, col=0, rowspan=1, colspan=2, title="Mean")
        self.P4.setLabel("left", "Current (pA)")
        self.P4.setLabel("bottom", "Time (s)")
        self.P5 = self.win.addPlot(row=7, col=0, rowspan=1, colspan=2, title="Variance")

        self.P5A = self.win.addPlot(row=8, col=0, rowspan=2, colspan=1, title="Mean and variance")
        self.P5B = self.win.addPlot(row=8, col=1, rowspan=2, colspan=1, title="Mean and variance")
        self.win.ci.layout.setRowStretchFactor(8, 4)
    def plot(
        self, qfit, time, sweeps, i_chan=None, mean: np.ndarray = None, var: np.ndarray = None
    ):
        # f, ax = mpl.subplots(3, 1)
        if i_chan is not None:
            for i in range(np.min((100, np.array(sweeps).shape[0]))):
                self.P5.plot(
                    x=time * 1e3, y=sweeps[i] * 1e12 + i_chan * 1e12, pen=pg.mkPen("k", width=0.25)
                )
        self.P5.setLabel("left", "pA")
        self.P5.setLabel("bottom", "Time (ms)")

        if mean is None and var is None:
            print("Mean and var ar none!!!")
            return
        imax_i = np.argmax(mean)
        self.P5A.plot(time[imax_i:] * 1e3, mean[imax_i:] * 1e12, pen=pg.mkPen(color="y", width=1))

        # self.P5A.plot(time * 1e3, var, pen=pg.mkPen(color="lightblue", width=1))
        self.P5A.plot(
            time[imax_i:] * 1e3, var[imax_i:] * 1e24, pen=pg.mkPen(color="lightblue", width=1)
        )
        self.P5A.setLabel("left", "Variance (pA^2)")
        self.P5A.setLabel("bottom", "Time (ms)")

        self.P5B.setYRange(0, np.max(var * 1e24))
        imax_i = np.argmax(mean)
        qfit = self.fit_meanvar(mean=mean[imax_i:], var=var[imax_i:])
        syms = self.P5B.plot(
            mean[imax_i:] * 1e12,
            var[imax_i:] * 1e24,
            pen=None,
            symbol="o",
            symbolPen=pg.mkPen(color=None, width=0.5, style=pg.QtCore.Qt.PenStyle.NoPen),
            symbolBrush=pg.mkBrush(color="w"),
        )
        syms.setSymbolSize(5)
        self.P5B.plot(
            mean[imax_i:] * 1e12,
            qfit.best_fit * 1e24,
            pen=pg.mkPen(color="r", width=1.5, style=pg.QtCore.Qt.PenStyle.SolidLine),
        )
        self.P5B.setLabel("left", "Variance (pA^2)")
        self.P5B.setLabel("bottom", "Mean (pA)")


class TestGenerator:
    def __init__(
        self,
        nchans: int = 20,
        ichan: float = 12e-12,
        sign: float = -1.0,
        ntrials: int = 400,
        dt: float = 0.0001,
    ):
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
        self.sign = sign  # direction of current
        self.noise_var = 3e-12
        self.i_chan_var = 1e-13
        self.max_prob = 1.0
        self.rng = default_rng(314159)

    def set_transistions(self):
        """Set forward and backward rates for channel c <-> o"""
        self.alpha = 0.0002  # opening rate
        self.beta = 0.0015  # closing rate
        self.d0 = 0.0001

    def generate_sweeps(self):
        # use gamma as it has minimumum time
        delay = 0.001  # self.rng.gamma(shape=2., scale=self.delay_tau, size=(self.n_chans, self.n_trials))+self.delay
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
                    sweeps[i, int(idels[n, i]) : (int(idels[n, i] + idurs[n, i]))] += chans[n, i]*self.sign
            sweeps[i] = DF.SignalFilter_LPFButter(
                sweeps[i] - np.mean(sweeps[i][:10]), 3000.0, 1.0 / self.dt, NPole=8
            )
        return sweeps


if __name__ == "__main__":
    NMAX = 10000
    ichan = 12e-12
    nchans = 20
    tg = TestGenerator(nchans=nchans, ichan=ichan, ntrials=1000, dt=1e-4)
    sweep_data = tg.generate_sweeps()
    # tg.run()
    NSA = NSFA()
    print(len(tg.time), np.max(tg.time), tg.dt)
    print(sweep_data.shape)
    print(sweep_data.shape[0] * sweep_data.shape[1])
    sweep_data = sweep_data.flatten(order="C")
    time = np.arange(0, tg.dt * sweep_data.shape[0], tg.dt)
    # win = pg.GraphicsLayoutWidget()
    # P1 = win.addPlot()
    # P1.plot(t, sweep_data)
    # win.show()
    # if sys.flags.interactive != 1:
    #     pg.QtWidgets.QApplication.instance().exec()
    idata = np.tile(sweep_data, (1, 2))
    print("idata shape: ", idata.shape)
    # aj, summary, timebase  = MVT.do_one(
    #     current_data=idata, sampleRate=1./tg.dt, sweepCount=idata.shape[0], timebase=tg.time
    # )
    # clean_event_traces = -1 * np.array(
    #     [summary.allevents[ev] for ev in summary.isolated_event_trace_list]
    # )
    # clean_event_onsets = tuple(
    #     [ev for ev in summary.clean_event_onsets_list if ev in summary.isolated_event_trace_list]
    # )

    # peak_times = np.array([np.argmax(tr) for tr in clean_event_traces])
    # tb = aj.dt_seconds * np.arange(len(clean_event_traces[0]))
    # if len(clean_event_traces) > NMAX:
    #     clean_event_traces = clean_event_traces[0:NMAX]
    #     clean_event_onsets = clean_event_onsets[0:NMAX]
    #     peak_times = peak_times[0:NMAX]
    # print("clean event onsets: ", clean_event_onsets)
    # peak_times = [(summary.dt_seconds*u[1]) for i, u in enumerate(clean_event_onsets)]
    # peak_times = np.array([i[1] for i in clean_event_onsets]) + peak_times
    # mpl.plot(tb, clean_event_traces.T,  alpha=0.5)

    # mpl.show()
    # exit()
    # timebase = aj.dt_seconds * np.arange(len(clean_event_traces[0]))


    (
        aj,
        summary,
        tracetimebase,
    ) = MVT.do_one(current_data=sweep_data, sampleRate=1./tg.dt, sweepCount = 1, timebase=time)
    nsfa = MVT.process_data(aj, summary, tracetimebase)
    nsfa.win.show()
    if sys.flags.interactive != 1:
        pg.QtWidgets.QApplication.instance().exec()


    # NSA.setup(timebase=timebase, eventtraces=clean_event_traces, eventpeaktimes=peak_times)
    # NSA.align_on_rising(Nslope=7)
    # meanI = np.mean(NSA.d, axis=0)
    # varI = np.var(NSA.d, axis=0)
    # NSA.mean = meanI
    # NSA.var = varI
    # qfit = NSA.fit_meanvar(mean=meanI, var=varI)
    # NSA.plot(qfit, NSA.t, NSA.d, mean=meanI, var=varI)
    # print(f"NChan original: {nchans:3d}, gamma_original: {ichan*1e12:.3f} pA")

    # NSA.win.show()
    # nsfa.plot(qfit, nsfa.t, nsfa.d, mean=meanI, var=varI)
    # mpl.show()

    # if sys.flags.interactive != 1:
    #     pg.QtWidgets.QApplication.instance().exec()
