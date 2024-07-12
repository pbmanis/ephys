import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Union

import numpy as np
import pyabf
# import matplotlib.pyplot as mpl
import pyqtgraph as pg

import ephys.mini_analyses.mini_event_dataclasses as MEDC  # get result datastructure
import ephys.mini_analyses.minis_methods as MM
import ephys.tools.mean_variance
from ephys.mini_analyses import minis_methods

NMAX = 100  # maximum number of events to analyze (helps in testing)

def def_taus():
    return [0.001, 0.010]  # [0.0003, 0.001]  # in seconds (was 0.001, 0.010)


def def_template_taus():
    return [0.0005, 0.002]  # [0.001, 0.003] # was 0.001, 0.005


@dataclass
class EventParameters:
    ntraces: int = 1  # number of trials
    dt: float = 2e-5  # msec
    tdur: float = 1e-1
    maxt: float = 10.0  # sec
    meanrate: float = 10.0  # Hz
    amp: float = 20e-12  # Amperes
    ampvar: float = 5e-12  # Amperes (variance)
    noise: float = 4e-12  # 0.0  # Amperes, gaussian nosie  # was 4e-12
    threshold: float = 2.5  # threshold used in the analyses
    LPF: Union[None, float] = 5000.0  # low-pass filtering applied to data, f in Hz or None
    HPF: Union[None, float] = None  # high pass filter (sometimes useful)
    taus: list = field(default_factory=def_taus)  # rise, fall tau in msec for test events
    template_taus: list = field(default_factory=def_template_taus)  # initial tau values in search
    sign: int = -1  # sign: 1 for positive, -1 for negative
    mindur: float = 1e-3  # minimum length of an event in seconds
    bigevent: Union[None, dict] = None  # whether to include a big event;
    # if so, the event is {'t', 'I'} np arrays.
    expseed: Union[int, None] = 1  # starting seed for intervals
    noiseseed: Union[int, None] = 1  # starting seed for background noise
    artifact_suppression: bool = False


# fn = "/Users/pbmanis/Desktop/Python/Downs-McElligott/abf_data/2023_09_20_0001.abf"
# fn = "/Users/pbmanis/Desktop/Python/Downs-McElligott/abf_data/2023_12_08_0016.abf"
fn = "/Users/pbmanis/Desktop/Python/Downs-McElligott/abf_data/2023_12_08_0024.abf"
fn_title = Path(*Path(fn).parts[-2:])

print("File exists: ", Path(fn).exists())
print("title: ", fn_title)
abf = pyabf.ABF(fn)
# print(dir(abf))
# print(abf.sampleRate)
abf.sweepY *= 1e-12  # convert to A (was in pA)

# MA = minis_methods.MiniAnalyses()  # get a minianalysis instance

# all times are in seconds

# MA.setup(
#     datasource="MiniAnalyses",
#     ntraces=1,
#     tau1=0.003,
#     tau2=0.006,
#     dt_seconds=1./abf.sampleRate,
#     template_tmax=0.05,  # sec
#     template_pre_time=0.001,  # sec
#     event_post_time=15,
#     sign=-1,
#     risepower=4,
#     threshold=3.5,
#     filters=None,
# )
# MA.set_timebase(abf.sweepX)

def get_data(abf):
    current_data = abf.sweepY
    sampleRate = abf.sampleRate
    sweepCount = abf.sweepCount
    timebase = abf.sweepX
    return current_data, sampleRate, sweepCount, timebase

def do_one(current_data, sampleRate, sweepCount, timebase):
    print("Shape of current data: ", current_data.shape)
    aj = MM.AndradeJonas()
    # pars.HPF = None
    crit = [None] * 1
    filters = MEDC.Filtering()
    filters.LPF_frequency = 4000.0
    filters.LPF_type = "ba"
    filters.Notch_frequencies = None
    filters.Detrend_method = "meegkit"

    pars = EventParameters()
    pars.template_taus = [0.0005, 0.002]
    pars.dt = 1.0 / sampleRate
    pars.ntraces = sweepCount


    aj.setup(
        ntraces=pars.ntraces,
        tau1=pars.template_taus[0],
        tau2=pars.template_taus[1],
        dt_seconds=pars.dt,
        delay=0.0,
        template_tmax=pars.maxt,  # taus are for template
        sign=pars.sign,
        risepower=4.0,
        threshold=3.0,
        filters=filters,
    )


    aj.set_timebase(timebase)
    print("trace timebase shape: ", timebase.shape)

    idata = np.tile(current_data, (1, 2))
    print("current data shape: ", idata.shape)
    print("Ntraces: ", pars.ntraces)
    for i in range(pars.ntraces):
        aj.prepare_data(idata, pars=pars)
        aj.deconvolve(
            aj.data[i],
            timebase=timebase,
            itrace=i,
            llambda=5.0,
            prepare_data=False,
        )

        # print("threshold: ", aj.threshold)
    aj.identify_events(order=7)
    summary = aj.summarize(np.array(aj.data))
    summary = aj.average_events(traces=[0], data=aj.data, summary=summary)
    tot_events = sum([len(x) for x in aj.onsets])
    print("total events identified: ", tot_events)

    # fig_handle = aj.plots(
    #     idata, events=None, title="AJ", testmode=True, show_indef=True,
    # )  # i_events)
    # mpl.show()
    # exit()

    return aj, summary, timebase

def process_data(aj, summary, tracetimebase):
    sign = -1
    if sign == 1:
        pkfunc = np.nanmax
        pkargfunc = np.argmax
    elif sign == -1:
        pkfunc = np.nanmin
        pkargfunc = np.argmin
    else:
        raise ValueError("sign must be 1 or -1")
    clean_event_traces = sign * np.array(
        [summary.allevents[ev] for ev in summary.isolated_event_trace_list]
    )
    evamps = pkfunc(clean_event_traces, axis=1)
    print("evamps: ", evamps*1e12)
    ev25 = np.nanpercentile(
        evamps,
        q=25,
    )  #  method="median_unbiased")
    ev75 = np.nanpercentile(
        evamps,
        q=75,
    )  #  method="median_unbiased")
    print("ev75: ", ev75)
    clean_event_trace_indices = [i for i, tr in enumerate(clean_event_traces) if pkfunc(tr) > ev75]
    isol_traces = [tr for tr in summary.isolated_event_trace_list if pkfunc(tr) > ev75]

    clean_event_onsets = [summary.clean_event_onsets_list[i] for i in clean_event_trace_indices]
    clean_event_traces = np.array([clean_event_traces[i]for i in clean_event_trace_indices])
    # tuple(
    #     [ev for ev in summary.clean_event_onsets_list if ev in isol_traces]
    # )
    peak_times = np.array([pkargfunc(tr) for tr in clean_event_traces])
    tb = aj.dt_seconds * np.arange(len(clean_event_traces[0]))
    if len(clean_event_traces) > NMAX:
        clean_event_traces = clean_event_traces[0:NMAX]
        clean_event_onsets = clean_event_onsets[0:NMAX]
        peak_times = peak_times[0:NMAX]
    print("clean event onsets: ", clean_event_onsets)
    peak_times = [(summary.dt_seconds*u[1]) for i, u in enumerate(clean_event_onsets)]
    # print("clean event traces: ", clean_event_onsets)
    print("peak times: ", peak_times)
    # mpl.plot(tb, clean_event_traces.T,  alpha=0.5)

    # mpl.show()
    # exit()
    n = aj.data.shape[1]//2
    print(n, aj.data.shape)
    timebase = aj.dt_seconds * np.arange(len(clean_event_traces[0]))
    nsfa = ephys.tools.mean_variance.NSFA(
        tracetimebase, aj.data[0,:n], title=fn_title
    )  # timebase, clean_event_traces, clean_event_onsets)
    nsfa.setup(timebase, clean_event_traces, peak_times)

    nsfa.align_on_rising(Nslope=7, plot=True)
    meanI = np.mean(nsfa.d, axis=0)
    varI = np.var(nsfa.d, axis=0)
    nsfa.mean = meanI
    nsfa.var = varI
    qfit = nsfa.fit_meanvar(mean=meanI, var=varI)
    nsfa.plot(qfit, nsfa.t, nsfa.d, mean=meanI, var=varI)

    return nsfa
    # nsfa.plot(qfit, nsfa.t, nsfa.d, mean=meanI, var=varI)
    # mpl.show()


if __name__ == "__main__":
    current_data, sampleRate, sweepCount, timebase=get_data(abf)
    (
        aj,
        summary,
        tracetimebase,
    ) = do_one(current_data, sampleRate, sweepCount, timebase)
    nsfa = process_data(aj, summary, tracetimebase)
    nsfa.win.show()
    if sys.flags.interactive != 1:
        pg.QtWidgets.QApplication.instance().exec()
