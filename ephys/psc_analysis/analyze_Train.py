from pathlib import Path
from typing import List
import numpy as np
from collections import OrderedDict

def analyze_PPF(
    PSC,
    rmpregion: list = [0.0, 0.045],
    twidth: float = 0.02,
    measure_func: object = np.min,
):
    """
    Analyze paired-pulse facilitiation

    Notes:
        The PPF protocol always involves 2 pulses, the second of which varies in time.
        Here, we compute the ratio between the 2 pulses for each time,
        and also save clips of the data waveforms for plotting.

        The stim dict in pulse_train will look like::
            {'start': [0.05, 0.1], 'duration': [0.0001, 0.0001],
            'amplitude': [0.00025, 0.00025],
            'npulses': [2], 'period': [0.05],
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

    Stim_Intvl = np.tile(PSC.stim_dt, len(PSC.reps))  # stimuli in order

    PSC.reject_list = []
    for rj in PSC.NGlist:
        trial = int(rj[0:3])
        intvl_trace = int(rj[4:])
        PSC.reject_list.append((trial * len(PSC.stim_dt)) + intvl_trace)

    PSC.analysis_summary[f"PPF"] = [[]] * len(PSC.stim_dt)
    PSC.analysis_summary["iHold"] = []
    PSC.analysis_summary["ppf_dt"] = PSC.stim_dt
    PSC.i_mean = []

    ppf_traces_T1 = OrderedDict(
        [(k, []) for k in PSC.stim_dt]
    )  # control response for each dt
    ppf_traces_R1 = OrderedDict(
        [(k, []) for k in PSC.stim_dt]
    )  # Response to the second stimulus at dt
    ppf_traces_T2 = OrderedDict([(k, []) for k in PSC.stim_dt])
    ppf_traces_R2 = OrderedDict([(k, []) for k in PSC.stim_dt])

    ppf_dat = OrderedDict(
        [(k, []) for k in PSC.stim_dt]
    )  # calculated PPF for each trial.
    num_intervals = len(Stim_Intvl)
    dead_time = 1.5e-3  # time before start of response measure
    # f, axx = mpl.subplots(1,1)
    for j in range(len(PSC.AR.traces)):  # for all (accepted) traces
        if j in PSC.reject_list:
            print("*" * 80)
            print(f"trace j={j:d} is in rejection list: {str(PSC.reject_list):s}")
            print(f"     from: {str(PSC.NGlist):s}")
            print("*" * 80)
            continue
        mi = PSC.AR.trace_index[
            j
        ]  # get index into marked/accepted traces then compute the min value minus the baseline
        t_stim1 = PSC._compute_interval(
            x0=PSC.pulse_train["start"][0],
            artifact_delay=dead_time,
            index=mi,
            stim_intvl=Stim_Intvl,
            max_width=twidth,
            pre_time=1e-3,
            pflag=False,
        )
        t_stim2 = PSC._compute_interval(
            x0=Stim_Intvl[mi] + PSC.pulse_train["start"][0],
            artifact_delay=dead_time,
            index=mi,
            stim_intvl=Stim_Intvl,
            max_width=twidth,
            pre_time=1e-3,
            pflag=False,
        )

        PSC.T0 = t_stim2[0]  # kind of bogus
        PSC.T1 = t_stim2[1]

        bl = np.mean(PSC.Clamps.traces["Time" : rmpregion[0] : rmpregion[1]][j])
        i_pp1 = (
            PSC.Clamps.traces["Time" : t_stim1[0] : t_stim1[1]][j] - bl
        )  # first pulse trace
        tb_ref = PSC.Clamps.time_base[
            np.where(
                (PSC.Clamps.time_base >= t_stim1[0])
                & (PSC.Clamps.time_base < t_stim1[1])
            )
        ]
        i_pp2 = (
            PSC.Clamps.traces["Time" : t_stim2[0] : t_stim2[1]][j] - bl
        )  # second pulse trace
        tb_p2 = PSC.Clamps.time_base[
            np.where(
                (PSC.Clamps.time_base >= t_stim2[0])
                & (PSC.Clamps.time_base < t_stim2[1])
            )
        ]

        da1 = measure_func(i_pp1)
        da2 = measure_func(i_pp2)
        ppf_tr = da2 / da1  # get facilitation for this trace and interval
        ppf_dat[Stim_Intvl[mi]].append(ppf_tr)  # accumulate
        ppf_traces_T1[Stim_Intvl[mi]].append(tb_ref)
        ppf_traces_R1[Stim_Intvl[mi]].append(i_pp1)
        ppf_traces_T2[Stim_Intvl[mi]].append(tb_p2)
        ppf_traces_R2[Stim_Intvl[mi]].append(i_pp2)
        # print(np.min(tb_ref), np.max(tb_ref), np.min(tb_p2), np.max(tb_p2))
        # plotWidget = pg.plot(title="traces")
        # si = Stim_Intvl[mi]
        # plotWidget.plot(ppf_traces_T1[si][-1], ppf_traces_R1[si][-1], pen='g')
        # plotWidget.plot(ppf_traces_T2[si][-1], ppf_traces_R2[si][-1], pen='r')
        # plotWidget.plot(tb_ref, np.ones_like(tb_ref)*da1, pen='b')
        # plotWidget.plot(tb_p2, np.ones_like(tb_p2)*da2, pen='m')
        # if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        #     QtGui.QApplication.instance().exec_()

        # axx.plot(tb_ref, i_pp1, 'k-')
        # axx.plot(tb_p2, i_pp2, 'r-')
    PSC.analysis_summary[f"PPF"] = ppf_dat
    PSC.analysis_summary["PPF_traces_T1"] = ppf_traces_T1
    PSC.analysis_summary["PPF_traces_R1"] = ppf_traces_R1
    PSC.analysis_summary["PPF_traces_T2"] = ppf_traces_T2
    PSC.analysis_summary["PPF_traces_R2"] = ppf_traces_R2
    PSC.analysis_summary["psc_stim_amplitudes"] = np.array(stim_I)
    PSC.analysis_summary["psc_intervals"] = np.array(PSC.stim_dt)
    PSC.analysis_summary["stim_times"] = PSC.pulse_train["start"]
    PSC.analysis_summary["window"] = [PSC.T0, PSC.T1]

    # fname = Path(PSC.datapath).parts
    # fname = '/'.join(fname[-4:]).replace('_', '\_')
    # f.suptitle(f"{fname:s}")
    # mpl.show()
    return True