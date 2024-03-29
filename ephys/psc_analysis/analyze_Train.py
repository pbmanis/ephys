from pathlib import Path
from typing import List
import numpy as np
from collections import OrderedDict
import ephys.tools.utilities as UT

UT = UT.Utility()


def plot_one_train(stim, psc_amp, PSC):
    import matplotlib.pyplot as mpl
    f, ax = mpl.subplots(2,1)
    intvl = np.array(stim)*1e3
    nreps = psc_amp.shape[0]

    for nr in range(nreps):
        ax[0].scatter(intvl, psc_amp[nr,:], marker='o', s=12, clip_on=False)
    ax[0].errorbar(intvl, np.mean(psc_amp, axis=0), yerr=np.std(psc_amp, axis=0),
        linestyle='-', linewidth=0.5, clip_on=False)
    cell_id = str(Path(*PSC.datapath.parts[-4:-1]))
    prot = str(Path(PSC.datapath.parts[-1]))
    ax[0].set_title(f"{cell_id:s}  {prot:s}", loc="right", fontdict={"size": 8, "weight":"bold"})
    ax[0].set_xlim(0, 250.)
    mpl.show()

def analyze_Train(
    PSC,
    rmpregion: list = [0.0, 0.045],
    deadtime: float=0.7e-3,
    artifact_sign: str='+',
    twidth: float = 0.04,
    measure_func: object = np.nanmin,
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
    n_reps = len(PSC.reps)
    n_pulses = stim['npulses']
    if stim['type'] == 'pulseTrain':
        stim_dt = stim['start']
    else:
        stim_dt = PSC.stim_dt
    Stim_Intvl = np.tile(stim_dt, n_reps)  # stimuli in order
    twidths = [twidth]*n_pulses
    for i, sdt in enumerate(stim_dt):
        if twidth > sdt:
            twidths[i] = sdt - 0.001 # make sure the window ends before next stimulus


    PSC.reject_list = []
    for rj in PSC.NGlist:
        trial = int(rj[0:3])
        intvl_trace = int(rj[4:])
        PSC.reject_list.append((trial * len(stim_dt)) + intvl_trace)

    PSC.analysis_summary[f"Train"] = [[]] * len(stim_dt)
    PSC.analysis_summary["iHold"] = []
    PSC.i_mean = []

    train_traces_T = {}
    train_traces_R = {}
    train_facilitation_R = {}

    # calculated train response for each stimulus in each trial.
    for k in range(n_reps):
        train_traces_T[k] = [None]*n_pulses
        train_traces_R[k] = [None]*n_pulses
        train_facilitation_R[k] = [(n, []) for n in Stim_Intvl.ravel()]

    psc_amp = np.zeros((n_reps, n_pulses))*np.nan
    j = 0
    for rep_no in PSC.reps:  # for all (accepted) traces
        # get index into marked/accepted traces then compute the min value minus the baseline
        if j >= len(PSC.AR.trace_index):
            j += 1
            continue
        mi = PSC.AR.trace_index[j]
        if j in PSC.reject_list:
            print("*" * 80)
            print(f"trace j={j:d} is in rejection list: {str(PSC.reject_list):s}")
            print(f"     from: {str(PSC.NGlist):s}")
            print("*" * 80)
            j += 1
            continue

        train_windows = []
        for pulse_no in range(n_pulses):
            t_stim = PSC.compute_interval(
                x0=stim["start"][pulse_no],
                artifact_duration=deadtime,
                index=mi,
                stim_intvl=Stim_Intvl,
                max_width=twidths[pulse_no],
                pre_time=1e-3,
                pflag=False,
            )
            train_windows.append(t_stim)

            bl0 = [Stim_Intvl[pulse_no]-0.0025, Stim_Intvl[pulse_no]-0.0005]
            if pulse_no < n_pulses:
                bl1 = [Stim_Intvl[pulse_no+1]-0.0025, Stim_Intvl[pulse_no+1]-0.0005]
            else:
                bl1 = [Stim_Intvl[pulse_no]-0.0025 + twidths[pulse_no], Stim_Intvl[pulse_no]-0.0005+twidths[pulse_no]]
            # print("bl wins: ", bl0, bl1)
            bl_0 = np.mean(PSC.Clamps.traces["Time" : bl0[0] : bl0[1]][j])
            bl_1 = np.mean(PSC.Clamps.traces["Time" : bl1[0] : bl1[1]][j])
            bl = (bl_0 + bl_1)/2.0           
            I_psc = PSC.Clamps.traces["Time" : t_stim[0] : t_stim[1]][j] - bl
            
            # if str(PSC.datapath.parts[-4]).startswith('2022.05.27'):
            #     print('rep: ', rep_no, 'pulse: ', pulse_no, 'baseline: ', bl, "Imin: ", np.min(I_psc))
            train_traces_T[rep_no][pulse_no] = PSC.Clamps.time_base[
                np.where(
                    (PSC.Clamps.time_base >= t_stim[0])
                    & (PSC.Clamps.time_base < t_stim[1])
                    )
                ]
            train_traces_R[rep_no][pulse_no] = I_psc
            sinterval = PSC.Clamps.sample_interval
            psc_amp[rep_no, pulse_no] = measure_func(UT.trim_psc(I_psc, dt=sinterval, artifact_duration=deadtime, sign=artifact_sign))*1e12
        j += 1
        # train_tr = np.divide(np.array(psc_amp[nr,:]),np.array(psc_amp[nr,0]).reshape((-1,1)))  # get facilitation for this trace and interval
        # train_facilitation_R[nr] = train_tr
    #     print(train_traces_T.keys())
    # train_traces_T=Stim_Intvl
    # train_traces_R[Stim_Intvl[mi]].append(psc_amp[nr,:])
    train_facilitation_R = psc_amp/psc_amp[:,0].reshape(-1, 1)
    # print(train_facilitation_R)
    # print(psc_amp)
    PSC.T0 = t_stim[0]
    PSC.T1 = t_stim[1]
    PSC.analysis_summary["npulses"] = n_pulses
    PSC.analysis_summary["nreps"] = n_reps
    PSC.analysis_summary["Stim_I"] = stim_I
    PSC.analysis_summary["Train_Facilitation_R"] = np.array(train_facilitation_R)  # ratio responses
    PSC.analysis_summary["train_dt"] = stim_dt # one rep
    PSC.analysis_summary["Train_traces_T"] = train_traces_T # trace time data
    PSC.analysis_summary["Train_traces_R"] = train_traces_R # trace current data
    PSC.analysis_summary["psc_amp"] = psc_amp
    PSC.analysis_summary["psc_stim_amplitudes"] = np.array(stim['amplitude'])
    PSC.analysis_summary["stim_times"] = np.array(stim['starts'])
    PSC.analysis_summary['sample_interval'] = sinterval
    PSC.analysis_summary["window"] = train_windows
    PSC.analysis_summary["Group"] = PSC.Group


    # plot_one_train(PSC.analysis_summary['train_dt'],
    #             PSC.analysis_summary['Train_traces_R'],
    #             PSC)
    return True