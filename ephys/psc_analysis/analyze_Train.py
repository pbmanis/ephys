from pathlib import Path
from typing import List
import numpy as np
from collections import OrderedDict

def analyze_Train(
    PSC,
    rmpregion: list = [0.0, 0.045],
    twidth: float = 0.02,
    measure_func: object = np.min,
):
    """
    Analyze Trains of stimuli
    Notes:
        The Train protocol has at least 3-4 pulses at a constant interval.
        Here, we compute the ratio between the Nth pulse and the first pulse response,
        and also save clips of the data waveforms for plotting.

        The stim dict in pulse_train will look like::
            {'start': [0.05, 0.1, 0.5, 0.2, 0.25], 'duration': [0.0001 .. nth],
            'amplitude': [0.00025, ... nth],
            'npulses': [N], 'period': [0.05],
            'type': ['pulseTrain']}

    Parameters
    ----------
    rmpregion: 2 element list (default: [0., 0.05])
        The region of the trace used to measure the resting membrane potential,
        in seconds.

    

    """

    stim_I = [PSC.pulse_train["amplitude"][0]]

    # check the db to see if we have parameters already
    dfiles = PSC.db["date"].tolist()  # protocols matching our prefix
    PSC.sign = 1
    PSC.set_baseline_times(rmpregion)
    PSC.i_mean = []
    PSC.i_mean = []

    PSC.reject_list = []
    for rj in PSC.NGlist:
        trial = int(rj[0:3])
        intvl_trace = int(rj[4:])
        PSC.reject_list.append((trial * len(PSC.reps)) + intvl_trace)

    stim = PSC.AR.getStim("Stim0")
    if stim['type'] == 'pulseTrain':
        stim_dt = stim['start']
    else:
        stim_dt = PSC.stim_dt
    Stim_Intvl = np.tile(stim_dt, len(PSC.reps))  # stimuli in order

    PSC.reject_list = []
    for rj in PSC.NGlist:
        trial = int(rj[0:3])
        intvl_trace = int(rj[4:])
        PSC.reject_list.append((trial * len(stim_dt)) + intvl_trace)

    PSC.analysis_summary[f"Train"] = [[]] * len(stim_dt)
    PSC.analysis_summary["iHold"] = []
    PSC.analysis_summary["train_dt"] = stim_dt
    PSC.i_mean = []

    train_traces_T = {}
    train_traces_R = {}
    nreps = len(PSC.reps)
    for k in range(nreps):
        train_traces_T[k] = [(n, []) for n in range(stim['npulses'])]
        train_traces_R[k] = [(n, []) for n in range(stim['npulses'])]

    train_dat = OrderedDict(
        [(k, []) for k in stim_dt]
    )  # calculated PPF for each trial.

    dead_time = 1.5e-3  # time before start of response measure
    psc_amp = np.zeros((nreps, stim['npulses']))*np.nan
    j = 0
    for nr in range(nreps):  # for all (accepted) traces
        if j in PSC.reject_list:
            print("*" * 80)
            print(f"trace j={j:d} is in rejection list: {str(PSC.reject_list):s}")
            print(f"     from: {str(PSC.NGlist):s}")
            print("*" * 80)
            j += 1
            continue
        # get index into marked/accepted traces then compute the min value minus the baseline
        mi = PSC.AR.trace_index[j] 
        train_windows = []
        for k in range(stim['npulses']):
            t_stim = PSC._compute_interval(
                x0=stim["start"][k],
                artifact_delay=dead_time,
                index=mi,
                stim_intvl=Stim_Intvl,
                max_width=twidth,
                pre_time=1e-3,
                pflag=False,
            )
            train_windows.append(t_stim)
            if k == 0:
                bl = np.mean(PSC.Clamps.traces["Time" : rmpregion[0] : rmpregion[1]][j])
            I_psc = (
                PSC.Clamps.traces["Time" : t_stim[0] : t_stim[1]] - bl
            )
            psc_amp[nr, k] = measure_func(I_psc)*1e12
        j += 1
    print(dir(PSC))
    import matplotlib.pyplot as mpl
    f, ax = mpl.subplots(2,1)
    for nr in range(nreps):
        ax[0].plot(stim['start'], psc_amp[nr,:], 'o', markersize=2)
    ax[0].set_title(str(PSC.datapath) + PSC.protocol, loc="right")
    mpl.show()

    train_tr = da2 / da1  # get facilitation for this trace and interval
    train_dat[Stim_Intvl[mi]].append(train_tr)  # accumulate
    train_traces_T[Stim_Intvl[mi]].append(tb_ref)
    train_traces_R[Stim_Intvl[mi]].append(i_pp1)

    PSC.analysis_summary[f"Train"] = train_dat
    PSC.analysis_summary["Train_traces_T"] = train_traces_T
    PSC.analysis_summary["Train_traces_R"] = train_traces_R
    PSC.analysis_summary["psc_stim_amplitudes"] = np.array(stim['amplitude'])
    PSC.analysis_summary["psc_intervals"] = np.array([stim['period']]*stim['npulses'])
    PSC.analysis_summary["stim_times"] = np.array(stim['start'])
    PSC.analysis_summary["window"] = [PSC.T0, PSC.T1]

    return True