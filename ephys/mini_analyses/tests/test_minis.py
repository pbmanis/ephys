# -*- encoding: utf-8 -*-
"""
Test fixture for minis_methods
Provide synthesis of data, and run each of the tests.
"""

import dataclasses
import sys
from typing import Union
import time
from dataclasses import dataclass, field


import numpy as np
import scipy.signal

import ephys.mini_analyses.minis_methods as MM
from ephys.mini_analyses.util import UserTester

# from cnmodel.protocols import SynapseTest


# import pyqtgraph as pg

testmode = True  # set false to hold graphs up until closed;
# true for 2 sec display


def def_taus():
    return [0.001, 0.010]  # [0.0003, 0.001]  # in seconds (was 0.001, 0.010)


def def_template_taus():
    return [0.001, 0.005]  # [0.001, 0.003] # was 0.001, 0.005


@dataclass
class EventParameters:
    dt: float = 2e-5  # msec
    tdur: float = 1e-1
    maxt: float = 10.0  # sec
    meanrate: float = 10.0  # Hz
    amp: float = 20e-12  # Amperes
    ampvar: float = 5e-12  # Amperes (variance)
    noise: float = 4e-12  # 0.0  # Amperes, gaussian nosie  # was 4e-12
    threshold: float = 2.5  # threshold used in the analyses
    LPF: Union[
        None, float
    ] = 5000.0  # low-pass filtering applied to data, f in Hz or None
    HPF: Union[None, float] = None  # high pass filter (sometimes useful)
    taus: list = field(
        default_factory=def_taus
    )  # rise, fall tau in msec for test events
    template_taus: list = field(
        default_factory=def_template_taus
    )  # initial tau values in search
    sign: int = -1  # sign: 1 for positive, -1 for negative
    mindur: float = 1e-3  # minimum length of an event in seconds
    bigevent: Union[None, dict] = None  # whether to include a big event;
    # if so, the event is {'t', 'I'} np arrays.
    expseed: Union[int, None] = 1  # starting seed for intervals
    noiseseed: Union[int, None] = 1  # starting seed for background noise


def printPars(pars):
    print(dir(pars))
    d = dataclasses.asdict(pars)
    for k in d.keys():
        print("k: ", k, " = ", d[k])


# these are the tests that will be run


def test_ZeroCrossing():
    MinisTester(method="ZC", sign=1)


def test_ClementsBekkers_numba():
    MinisTester(method="CB", sign=1, extra="numba")  # accelerated method


def test_ClementsBekkers_cython():
    MinisTester(method="CB", sign=1, extra="cython")  # accelerated method


def test_AndradeJonas():
    MinisTester(method="AJ", sign=1)


def test_RSDeconvolve():
    MinisTester(method="RS", sign=1)


# def test_ClementsBekkers_python():
#     MinisTester(method='CB', extra='python') # slow interpreted method


def test_ZeroCrossing_neg():
    MinisTester(method="ZC", sign=-1)


def test_ClementsBekkers_numba_neg():
    MinisTester(method="CB", sign=-1, extra="numba")  # accelerated method


def test_ClementsBekkers_cython_neg():
    MinisTester(method="CB", sign=-1, extra="cython")  # accelerated method


# def test_ClementsBekkers_python():
#     MinisTester(method='CB', extra='python') # slow interpreted method


def test_AndradeJonas_neg():
    MinisTester(method="AJ", sign=-1)


def test_RSDeconvolve_neg():
    MinisTester(method="RS", sign=-1)


def generate_testdata(
    pars: dataclass,
    ntrials: int = 1,
    baseclass: Union[object, None] = None,
    func: Union[object, None] = None,
):
    """
        meanrate is in Hz(events/second)
        maxt is in seconds - duration of trace
        bigevent is a dict {'t': delayinsec, 'I': amplitudeinA}
    """
    if baseclass is None and func is not None:
        raise ValueError("Need base class definition")

    timebase = np.arange(0.0, pars.maxt, pars.dt)  # in ms

    t_psc = np.arange(
        0.0, pars.tdur, pars.dt
    )  # time base for single event template in ms
    if func is None:  # make double-exp event
        tau_1 = pars.taus[0]  # ms
        tau_2 = pars.taus[1]  # ms
        Aprime = (tau_2 / tau_1) ** (tau_1 / (tau_1 - tau_2))
        g = Aprime * (-np.exp(-t_psc / tau_1) + np.exp((-t_psc / tau_2)))
        gmax = np.max(g)
        g = pars.sign * g * pars.amp / gmax
    else:  # use template from the class
        baseclass._make_template()
        gmax = np.min(baseclass.template)
        g = pars.sign * pars.amp * baseclass.template / gmax

    testpsc = np.zeros((ntrials, timebase.shape[0]))
    testpscn = np.zeros((ntrials, timebase.shape[0]))
    i_events = [None] * ntrials
    t_events = [None] * ntrials
    for i in range(ntrials):
        pars.expseed = i * 47  # starting seed for intervals
        pars.noiseseed = i  # starting seed for background noise
        if pars.expseed is None:
            eventintervals = np.random.exponential(
                1.0 / pars.meanrate, int(pars.maxt * pars.meanrate)
            )
        else:
            np.random.seed(pars.expseed + i)
            eventintervals = np.random.exponential(
                1.0 / pars.meanrate, int(pars.maxt * pars.meanrate)
            )
        eventintervals = eventintervals[eventintervals < 10.0]
        events = np.cumsum(eventintervals)
        if pars.bigevent is not None:
            events = np.append(events, pars.bigevent["t"])
            events = np.sort(events)
        # time of events with exp distribution:
        t_events[i] = events[events < pars.maxt]
        i_events[i] = np.array([int(x / pars.dt) for x in t_events[i]])

        testpsc[i][i_events[i]] = np.random.normal(
            1.0, pars.ampvar / pars.amp, len(i_events[i])
        )
        if pars.bigevent is not None:
            ipos = int(pars.bigevent["t"] / pars.dt)  # position in array
            testpsc[ipos] = pars.bigevent["I"]
        testpsc[i] = scipy.signal.convolve(testpsc[i], g, mode="full")[
            : timebase.shape[0]
        ]

        if pars.noise > 0:
            if pars.noiseseed is None:
                testpscn[i] = testpsc[i] + np.random.normal(
                    0.0, pars.noise, testpsc.shape[1]
                )
            else:
                np.random.seed(pars.noiseseed)
                testpscn[i] = testpsc[i] + np.random.normal(
                    0.0, pars.noise, testpsc.shape[1]
                )
        else:
            testpscn[i] = testpsc[i]
    return timebase, testpsc, testpscn, i_events, t_events


def run_ZeroCrossing(
    pars=None, bigevent: bool = False, plot: bool = False
) -> object:
    """
    Do some tests of the ZC protocol and plot
    """
    if pars is None:
        pars = EventParameters()
    # minlen = int(pars.mindur / pars.dt)
    if bigevent:
        pars.bigevent = {"t": 1.0, "I": 20.0}
    print(pars)
    pars.threshold = 4.0
    for i in range(1):
        pars.noiseseed = i * 47
        pars.expseed = i
        zc = MM.ZCFinder()
        zc.setup(
            tau1=pars.template_taus[0],
            tau2=pars.template_taus[1],
            dt=pars.dt,
            delay=0.0,
            template_tmax=5.0 * pars.template_taus[1],
            sign=pars.sign,
            threshold=pars.threshold,
            lpf=pars.LPF,
            hpf=pars.HPF,
        )
        timebase, testpsc, testpscn, i_events, t_events = generate_testdata(
            pars
        )
        # events = zc.find_events(testpscn,
        #       data_nostim=None, minLength=minlen,)
        # print("# events in test data set: ", len(t_events))

    if plot:
        fig_handle = zc.plots(title="Zero Crossings", testmode=testmode)
    return zc, fig_handle


def run_ClementsBekkers(
    pars: dataclass = None,
    bigevent: bool = False,
    extra="numba",
    plot: bool = False,
) -> object:
    """
    Do some tests of the CB protocol and plot
    """
    ntraces = 1
    if pars is None:
        pars = EventParameters()
    if bigevent:
        pars.bigevent = {"t": 1.0, "I": 20.0}
    pars.threshold = 2.5
    pars.LPF = 5000.0
    # pars.HPF = None
    cb = MM.ClementsBekkers()
    pars.baseclass = cb
    crit = [None] * ntraces
    timebase, testpsc, testpscn, i_events, t_events = generate_testdata(
        pars, ntrials=ntraces
    )
    tot_seeded = sum([len(x) for x in i_events])
    print("total seeded events: ", tot_seeded)

    cb.setup(
        ntraces=ntraces,
        tau1=pars.template_taus[0],
        tau2=pars.template_taus[1],
        dt=pars.dt,
        delay=0.0,
        template_tmax=5.0 * pars.template_taus[1],
        sign=pars.sign,
        threshold=pars.threshold,
        lpf=pars.LPF,
        hpf=pars.HPF,
    )
    cb._make_template()
    cb.set_cb_engine(extra)

    for i in range(ntraces):
        cb.cbTemplateMatch(testpscn[i], itrace=i, lpf=pars.LPF)
        testpscn[i] = cb.data  # # get filtered data
        cb.reset_filtering()
        print("# events in template: ", len(t_events[i]))
        print("threshold: ", cb.threshold)
        print("len(crit)", len(cb.Criterion[i]))

    cb.identify_events()
    cb.summarize(np.array(testpscn))
    tot_events = sum([len(x) for x in cb.onsets])
    print("total events identified: ", tot_events, "from seeded: ", tot_seeded)

    if plot:
        fig_handle = cb.plots(
            np.array(testpscn), events=None, title="CB", testmode=testmode
        )  # i_events)
        mpl.show()
        time.sleep(5)
    return cb, fig_handle


def run_AndradeJonas(
    pars: dataclass = None, bigevent: bool = False, plot: bool = False
) -> object:

    ntraces = 1
    if pars is None:
        pars = EventParameters()
    pars.threshold = 4.0
    if bigevent:
        pars.bigevent = {"t": 1.0, "I": 20.0}
    aj = MM.AndradeJonas()
    pars.baseclass = aj
    pars.LPF = 5000.0
    # pars.HPF = None
    crit = [None] * ntraces
    timebase, testpsc, testpscn, i_events, t_events = generate_testdata(
        pars, ntrials=ntraces
    )
    tot_seeded = sum([len(x) for x in i_events])
    print("total seeded events: ", tot_seeded)

    aj.setup(
        ntraces=ntraces,
        tau1=pars.template_taus[0],
        tau2=pars.template_taus[1],
        dt=pars.dt,
        delay=0.0,
        template_tmax=pars.maxt,  # taus are for template
        sign=pars.sign,
        risepower=4.0,
        threshold=pars.threshold,
        lpf=pars.LPF,
        hpf=pars.HPF,
    )

    for i in range(ntraces):
        order = int(0.001 / pars.dt)
        # generate test data

        aj.deconvolve(
            testpscn[i], itrace=i, llambda=5.0,
        )  # - np.mean(testpscn),
        testpscn[i] = aj.data  # # get filtered data
        aj.reset_filtering()
        print("# events in template: ", len(t_events[i]))
        print("threshold: ", aj.threshold)
        print("len(crit)", len(aj.Criterion[i]))
    aj.identify_events(order=order)
    aj.summarize(np.array(testpscn))
    tot_events = sum([len(x) for x in aj.onsets])
    print("total events identified: ", tot_events, "from seeded: ", tot_seeded)

    if plot:
        fig_handle = aj.plots(
            np.array(testpscn), events=None, title="AJ", testmode=testmode
        )  # i_events)
        mpl.show()
        # time.sleep(5)
    return aj, fig_handle


def run_RSDeconvolve(
    pars: dataclass = None, bigevent: bool = False, plot: bool = False
) -> object:

    ntraces = 1

    if pars is None:
        pars = EventParameters()
    # print(pars)
    rs = MM.RSDeconvolve()
    pars.threshold = 4.0
    rs.setup(
        ntraces=ntraces,
        tau1=np.power(pars.template_taus[0], 1.0),
        tau2=0.5 * 1e-3,  # pars.template_taus[1],
        dt=pars.dt,
        delay=0.0,
        template_tmax=pars.maxt,  # taus are for template
        sign=pars.sign,
        risepower=4.0,
        threshold=pars.threshold,
        lpf=500.0,  # pars.LPF,
        hpf=pars.HPF,
    )
    pars.baseclass = rs

    # generate test data
    timebase, testpsc, testpscn, i_events, t_events = generate_testdata(
        pars, ntrials=ntraces
    )
    tot_seeded = sum([len(x) for x in i_events])
    print("total seeded events: ", tot_seeded)

    if bigevent:
        pars.bigevent = {"t": 1.0, "I": 20.0}

    for i in range(ntraces):
        rs.deconvolve(testpscn[i], itrace=i)
        testpscn[i] = rs.data  # # get filtered data
        rs.reset_filtering()

    rs.identify_events(order=20)
    rs.summarize(np.array(testpscn))
    tot_events = sum([len(x) for x in rs.onsets])
    print("total events identified: ", tot_events, "from seeded: ", tot_seeded)

    if plot:

        fig_handle = rs.plots(
            np.array(testpscn), events=None, title="RS", testmode=testmode
        )  # i_events)
        mpl.show()
        time.sleep(5)
    return rs, fig_handle


class MiniTestMethods:
    def __init__(
        self, method: str = "cb", sign=1, extra="numba", plot: bool = False
    ):
        self.plot = plot
        self.testmethod = method
        self.extra = extra
        self.sign = sign

    def run_test(self):

        pars = EventParameters()
        pars.LPF = 5000
        pars.sign = self.sign

        if self.testmethod in ["ZC", "zc"]:
            pars.threshold = 0.5
            pars.mindur = 1e-3
            pars.HPF = None
            result = run_ZeroCrossing(pars, plot=True)
            print("Events found: ", len(result.allevents))
            if self.plot:
                zct = np.arange(
                    0, result.allevents.shape[1] * result.dt, result.dt
                )
                for a in range(len(result.allevents)):
                    mpl.plot(zct, result.allevents[a])
                mpl.show()
        if self.testmethod in ["CB", "cb"]:
            result = run_ClementsBekkers(pars, extra=self.extra, plot=True)
            print("Events found: ", len(result.allevents))
            if self.plot:
                for a in range(len(result.allevents)):
                    mpl.plot(result.t_template, result.allevents[a])
                mpl.show()
        if self.testmethod in ["AJ", "aj"]:
            print("pars)")
            result = run_AndradeJonas(pars, plot=True)
            print("Events found: ", len(result.allevents))
            ajt = result.t_template[0 : result.allevents.shape[1]]
            if self.plot:
                for i, a in enumerate(range(len(result.allevents))):
                    mpl.plot(ajt, result.allevents[a] + i + 20.0)
                mpl.show()
        if self.testmethod in ["RS", "rs"]:
            pars.threshold = 2.25
            result = run_RSDeconvolve(pars, plot=True)
            print("Events found: ", len(result.allevents))
            if self.plot:
                rst = np.arange(
                    0, result.allevents.shape[1] * result.dt, result.dt
                )
                for a in range(len(result.allevents)):
                    mpl.plot(rst, result.allevents[a])
                mpl.show()
        # if self.testmethod in ["all", "ALL"]:
        #     run_ZeroCrossing(pars, plot=True)
        #     run_ClementsBekkers(pars, plot=True)
        #     run_AndradeJoans(pars, plot=True)

        testresult = {
            "onsets": result.onsets,
            "peaks": result.peaks,
            "amplitudes": result.amplitudes,
            "fitresult": result.fitresult,
            "fitted_tau1": result.fitted_tau1,
            "fitted_tau2": result.fitted_tau2,
            "risepower": result.risepower,
            "risetenninety": result.risetenninety,
            "decaythirtyseven": result.decaythirtyseven,
        }
        return testresult


class MinisTester(UserTester):
    def __init__(self, method, sign=1, extra="python"):
        self.TM = None
        self.figure = None
        self.extra = extra
        self.sign = sign
        if sign == 1:
            signstr = "positive"
        elif sign == -1:
            signstr = "negative"
        else:
            assert sign in [-1, 1]
        UserTester.__init__(self, "%s_%s" % (method, signstr), method)
        # UserTester.__init__(self, "%s_%s" % (method, extra), method)
        # if you want to store different results by the "extra" parameter

    def run_test(self, method):

        self.TM = MiniTestMethods(
            method=method, sign=self.sign, extra=self.extra
        )
        test_result = self.TM.run_test()

        if "figure" in list(test_result.keys()):
            self.figure = test_result["figure"]
        return test_result

    def assert_test_info(self, *args, **kwds):
        try:
            super(MinisTester, self).assert_test_info(*args, **kwds)
        finally:
            if self.figure is not None:
                del self.figure


def plot_traces_and_markers(method, dy=20e-12):
    tba = method.timebase[:len(method.Summary.allevents[0])]

    for i, a in enumerate(method.Summary.allevents):
        dyi = i * dy
        mpl.plot(tba, a + dyi)
        jtr = method.Summary.event_trace_list[
            i
        ]  # get trace and event number in trace
        if jtr is None:
            continue

        pk = method.Summary.smpkindex[jtr[0]][jtr[1]]
        on = method.Summary.onsets[jtr[0]][jtr[1]]
        onpk = (pk - on) * method.dt
        mpl.plot(
            onpk,
            (method.Summary.smoothed_peaks[jtr[0]][jtr[1]])
            + dyi,
            "ro",
            markersize=4,
        )
        pk = method.Summary.peaks[jtr[0]][jtr[1]]
        on = method.Summary.onsets[jtr[0]][jtr[1]]
        onpk = (pk - on) * method.dt
        mpl.plot(
            onpk,
            (method.Summary.amplitudes[jtr[0]][jtr[1]])
            + dyi,
            "ys",
            markersize=4,
        )

if __name__ == "__main__":
    if len(sys.argv[0]) > 1:
        testmethod = sys.argv[1]
    if testmethod not in [
        "ZC",
        "CB",
        "AJ",
        "zc",
        "cb",
        "aj",
        "all",
        "RS",
        "rs",
        "ALL",
    ]:
        print("Test for %s method is not implemented" % testmethod)
        exit(1)
    else:
        # set up for plotting
        import matplotlib

        rcParams = matplotlib.rcParams
        rcParams[
            "svg.fonttype"
        ] = "none"  # No text as paths. Assume font installed.
        rcParams["pdf.fonttype"] = 42
        rcParams["ps.fonttype"] = 42
        rcParams["text.usetex"] = False
        import warnings  # need to turn off a scipy future warning.

        import matplotlib.pyplot as mpl

        warnings.filterwarnings("ignore", category=FutureWarning)
        warnings.filterwarnings("ignore", category=UserWarning)
        warnings.filterwarnings(
            "ignore",
            message=(
                """UserWarning: findfont: Font family ['sans-serif']
            not found. Falling back to DejaVu Sans"""
            ),
        )

        pars = EventParameters()
        pars.LPF = 5000.0
        if testmethod in ["ZC", "zc"]:
            pars.threshold = 0.9
            pars.mindur = 1e-3
            pars.HPF = 20.0
            zc, fig = run_ZeroCrossing(pars, plot=True)
            print("# detected events: ", len(zc.Summary.allevents))
            zct = np.arange(0, zc.Summary.allevents.shape[1] * zc.dt, zc.dt)
            f, ax = mpl.subplots(1, 1)
            f.set_size_inches(5, 10)
            for i, a in enumerate(range(len(zc.Summary.allevents))):
                mpl.plot(zct, zc.Summary.allevents[a] + i * 10e-12)
            mpl.show()

        if testmethod in ["CB", "cb"]:
            for extras in [
                "cython",
            ]:
                # python version still does not work correctly
                # don't forget Numba, but it is not working well.
                pars.threshold = 3.0
                print("pars: ", pars)
                cb, fig = run_ClementsBekkers(pars, extra=extras, plot=True)
                print("All detected events: ", len(cb.Summary.allevents))
                f, ax = mpl.subplots(1, 1)
                f.set_size_inches(5, 10)
                plot_traces_and_markers(cb)
                mpl.show()

        if testmethod in ["AJ", "aj"]:
            pars.threshold = 5.0
            aj, fig = run_AndradeJonas(pars, plot=True)

            print("# detected events: ", len(aj.Summary.allevents))
            f, ax = mpl.subplots(1, 1)
            f.set_size_inches(5, 10)
            plot_traces_and_markers(aj)
            mpl.show()

        if testmethod in ["RS", "rs"]:
            rs, fig = run_RSDeconvolve(pars, plot=True)
            print("All detected events: ", len(rs.allevents))
            f, ax = mpl.subplots(1, 1)
            f.set_size_inches(5, 10)
            rst = np.arange(0, len(rs.Summary.allevents[0]) * rs.dt, rs.dt)
            for i, a in enumerate(range(len(rs.Summary.allevents))):
                # print(dir(aj))
                mpl.plot(rst, rs.Summary.allevents[a] + i * 10e-12)
            # note: no official template for comparison
            # rst = rs.t_template[0:rs.allevents.shape[1]]
            # for a in range(len(rs.allevents)):
            #     mpl.plot(rst, rs.allevents[a])
            mpl.show()
        if testmethod in ["all", "ALL"]:
            run_ZeroCrossing(pars, plot=True)
            run_ClementsBekkers(pars, plot=True)
            run_AndradeJonas(pars, plot=True)
            run_RSDeconvolve(pars, plot=True)

    #    pg.show()
    # if sys.flags.interactive == 0:
    #     pg.QtGui.QApplication.exec_()