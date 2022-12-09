from pathlib import Path
from typing import List
import numpy as np
from collections import OrderedDict
import pylibrary.tools.cprint as CP
import ephys.tools.Utility as UT
import pint
from pint import UnitRegistry, set_application_registry
UR = UnitRegistry()
set_application_registry(UR)

UT = UT.Utility()

def analyze_PPF(
    PSC,
    rmpregion: list = [0.0, 0.045],
    twidth: float = 0.05,
    measure_func: object = np.nanmin,
    deadtime: float=0.7e-3,
    artifact_sign: str='+',
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
    twidths = [twidth]*2
    for i, sdt in enumerate(PSC.stim_dt):
        if twidth > sdt:
            twidths[i] = sdt - 0.001 # make sure the window ends before next stimulus
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

        t_stim1 = PSC.compute_interval(
            x0=PSC.pulse_train["start"][0],
            artifact_duration=deadtime,
            index=mi,
            stim_intvl=Stim_Intvl,
            max_width=twidths[0],
            pre_time=1e-3,
            pflag=False,
        )
        t_stim2 = PSC.compute_interval(
            x0=Stim_Intvl[mi] + PSC.pulse_train["start"][0],
            artifact_duration=deadtime,
            index=mi,
            stim_intvl=Stim_Intvl,
            max_width=twidths[1],
            pre_time=1e-3,
            pflag=False,
        )
        
        PSC.T0 = t_stim2[0]  # kind of bogus - not used anymore
        PSC.T1 = t_stim2[1]



        bl1_0 = [t_stim1[0]-0.0035-deadtime, t_stim1[0]-deadtime-0.001]
        bl1_1 = [t_stim2[0]-0.0035, t_stim2[0]-0.001]
        bl2_0 = [t_stim2[0]-0.0035, t_stim2[0]-0.001]
        bl2_1 = [t_stim2[1]-0.0035-deadtime, t_stim2[1]-0.001-deadtime]

        bl1_0 = np.mean(PSC.Clamps.traces["Time" : bl1_0[0] : bl1_0[1]][j])
        bl2_1 = np.mean(PSC.Clamps.traces["Time" : bl2_1[0] : bl2_1[1]][j])

        i_pp1 = (
            PSC.Clamps.traces["Time" : t_stim1[0] : t_stim1[1]][j] - bl1_0
        ).view(np.ndarray)*UR.A  # first pulse trace
        tb_ref = PSC.Clamps.time_base[
            np.where(
                (PSC.Clamps.time_base >= t_stim1[0])
                & (PSC.Clamps.time_base < t_stim1[1])
            )
        ]*UR.s
        i_pp2 = (
            PSC.Clamps.traces["Time" : t_stim2[0] : t_stim2[1]][j] - bl2_1
        ).view(np.ndarray)*UR.A  # second pulse trace
        tb_p2 = PSC.Clamps.time_base[
            np.where(
                (PSC.Clamps.time_base >= t_stim2[0])
                & (PSC.Clamps.time_base < t_stim2[1])
            )
        ]*UR.s

        sinterval = PSC.Clamps.sample_interval

        da1 = measure_func(UT.trim_psc(i_pp1, dt=sinterval, artifact_duration=deadtime, sign=artifact_sign))
        da2 = measure_func(UT.trim_psc(i_pp2, dt=sinterval, artifact_duration=deadtime, sign=artifact_sign))
        ppf_tr = da2.magnitude / da1.magnitude  # get facilitation for this trace and interval
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
    PSC.analysis_summary["psc_stim_amplitudes"] = np.array(stim_I)*UR.A
    PSC.analysis_summary["psc_intervals"] = np.array(PSC.stim_dt)
    PSC.analysis_summary['sample_interval'] = sinterval
    PSC.analysis_summary["stim_times"] = PSC.pulse_train["start"]
    PSC.analysis_summary["window"] = [PSC.T0, PSC.T1]
    PSC.analysis_summary["Group"] = PSC.Group

    # fname = Path(PSC.datapath).parts
    # fname = '/'.join(fname[-4:]).replace('_', '\_')
    # f.suptitle(f"{fname:s}")
    # mpl.show()
    return True