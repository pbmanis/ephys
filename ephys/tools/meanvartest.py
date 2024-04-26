import pyabf
from typing import Union
from dataclasses import dataclass, field
import numpy as np

from pathlib import Path
import matplotlib.pyplot as mpl
from ephys.mini_analyses import minis_methods
import ephys.mini_analyses.mini_event_dataclasses as MEDC  # get result datastructure
import ephys.mini_analyses.minis_methods as MM
import ephys.tools.mean_variance




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


fn = "/Users/pbmanis/Desktop/Python/Downs-McElligott/abf_data/2023_09_20_0001.abf"
fn = "/Users/pbmanis/Desktop/Python/Downs-McElligott/abf_data/2023_12_08_0016.abf"
#fn = "/Users/pbmanis/Desktop/Python/Downs-McElligott/abf_data/2023_12_08_0024.abf"

print(Path(fn).exists())
abf = pyabf.ABF(fn)
print(dir(abf))
print(abf.sampleRate)
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


def do_one(abf):
    aj = MM.AndradeJonas()
    # pars.HPF = None
    crit = [None] * 1
    filters = MEDC.Filtering()
    filters.LPF_frequency = None
    filters.LPF_type = "ba"
    filters.Notch_frequencies = None
    filters.Detrend_method = None

    pars = EventParameters()
    pars.template_taus = [0.0002, 0.0015]
    pars.dt = 1.0 / abf.sampleRate
    pars.ntraces = abf.sweepCount

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

    timebase = abf.sweepX
    aj.set_timebase(timebase)
    print(abf.sweepY.shape)

    idata = np.tile(abf.sweepY, (1,2))
    print(idata.shape)

    for i in range(pars.ntraces):
        aj.prepare_data(idata, pars=pars)
        aj.deconvolve(
            idata[i],
            timebase=timebase,
            itrace=i,
            llambda=5.0,
            prepare_data=False,
        )

        print("threshold: ", aj.threshold)
    aj.identify_events(order=7)
    summary = aj.summarize(np.array(idata))
    summary = aj.average_events(traces=[0], data=idata, summary=summary)
    tot_events = sum([len(x) for x in aj.onsets])
    print("total events identified: ", tot_events)
    print(dir(summary))


    # fig_handle = aj.plots(
    #     idata, events=None, title="AJ", testmode=True, show_indef=True,
    # )  # i_events)
    # mpl.show()
    # exit()

    return aj, summary


if __name__ == "__main__":
    aj, summary= do_one(abf)
    # clean_event_traces = np.array([summary.allevents[ev] for ev in summary.isolated_event_trace_list])
    # print(summary.onsets)
    # clean_event_onsets = np.array([summary.allevents_onsets[ev] for ev in summary.isolated_event_trace_list])
    # print(clean_event_traces.shape)
    # print(summary.clean_event_onsets_list[0:10])
    # sh = clean_event_traces.shape
    clean_event_traces = -1*np.array([summary.allevents[ev] for ev in summary.isolated_event_trace_list])
    clean_event_onsets = tuple([ev for ev in summary.clean_event_onsets_list if ev in summary.isolated_event_trace_list])
    peak_times = np.array([np.argmax(tr) for tr in clean_event_traces])
    tb = aj.dt_seconds * np.arange(len(clean_event_traces[0]))
    # mpl.plot(tb, clean_event_traces.T,  alpha=0.5)

    # mpl.show()
    # exit()
    timebase = aj.dt_seconds * np.arange(len(clean_event_traces[0]))
    nsfa = ephys.tools.mean_variance.NSFA() # timebase, clean_event_traces, clean_event_onsets)
    nsfa.setup(timebase, clean_event_traces, peak_times)

    nsfa.align_on_rising(Nslope=7, plot=True)
    meanI = np.mean(nsfa.d, axis=0)
    varI = np.var(nsfa.d, axis=0)
    nsfa.mean = meanI
    nsfa.var = varI
    qfit = nsfa.fit_meanvar(mean=meanI, var=varI)
    f, ax = nsfa.plot(qfit, nsfa.t, nsfa.d, mean=meanI, var=varI)
  
    mpl.show()
