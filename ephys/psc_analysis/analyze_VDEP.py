import numpy as np
from typing import List, Tuple, Union
import ephys.psc_analysis.functions as FN

def analyze_VDEP(
    PSC,
    rmpregion: Union[List, Tuple] = [0.0, 0.05],
):
    """
    Analyze the voltage-dependence of EPSCs

    When selecting the analysis window, choose a window that encompases
    the peak of the inward EPSC in the negative voltage range.
    Do not try to balance the trace (the slope should be turned off)

    Parameters
    ----------
    rmpregion: 2 element list (default: [0., 0.05])
        The region of the trace used to measure the resting membrane potential,
        in seconds.
    """
    print("\n" + "******" * 4)

    dt = PSC.Clamps.sample_interval

    stim_I = [PSC.pulse_train["amplitude"][0]]
    if not ("MultiClamp1", "Pulse_amplitude") in PSC.AR.sequence.keys():
        raise ValueError(
            "VDEP requires certain parameters; cannot find (MultiClamp1, Pulse_amplitude) in stimulus command"
        )

    filekey = PSC.make_key(PSC.datapath)
    # check the db to see if we have parameters already
    dfiles = PSC.db["date"].tolist()
    width = 20.0

    if filekey in dfiles:
        # print(PSC.db.loc[PSC.db['date'] == filekey])
        # print(PSC.db.head())
        delays = PSC.db.loc[PSC.db["date"] == filekey]["T0"].values
        t1s = PSC.db.loc[PSC.db["date"] == filekey]["T1"].values
        if isinstance(delays, np.ndarray) and len(delays) > 1:
            delay = delays[0]
        else:
            delay = delays
        if isinstance(t1s, np.ndarray) and len(t1s) > 1:
            t1 = t1s[0]
        else:
            t1 = t1s
        print("delay from file", delay, t1)
    else:
        delay = 1.0 * 1e-3
        t1 = (width - 1.0) * 1e-3
        print("auto delay", delay, t1)
    ndelay = PSC.NMDA_delay
    nwidth = 0.0025
    bl_region = [
        PSC.pulse_train["start"][0] - 0.060,
        PSC.pulse_train["start"][0] - 0.010,
    ]  # time just before stimulus
    baseline = []
    PSC.baseline = bl_region
    stimamp = []
    stimintvl = []
    PSC.sign = 1
    PSC.i_mean = []
    PSC.analysis_summary["iHold"] = []
    PSC.analysis_summary[f"PSP_VDEP_AMPA"] = [[]] * len(PSC.pulse_train["start"])
    PSC.analysis_summary[f"PSP_VDEP_NMDA"] = [[]] * len(PSC.pulse_train["start"])
    
    ###

    bl, results = FN.mean_I_analysis(PSC.Clamps, region=bl_region, mode="baseline", reps=[0])
 
    rgn = [delay, t1]
    # # print('rgn: ', rgn)
    # if PSC.update_regions:
    #     rgn = PSC.set_region(
    #         [
    #             PSC.pulse_train["start"][0],
    #             PSC.pulse_train["start"][0] + PSC.NMDA_delay + 0.010,
    #         ],
    #         baseline=bl,
    #         slope=True,
    #     )
    # PSC.T0 = float(rgn[0])
    # PSC.T1 = float(rgn[1])

    if rgn[0] > 0.012:
        rgn[0] = 0.004
    rgn[1] = 0.20
    slope_region = rgn
    PSC.T0 = float(rgn[0])
    PSC.T1 = float(rgn[1])
    print("t0, t1: ", PSC.T0, PSC.T1)
    # two pass approach:
    # 1 find min, and look at the most negative traces (-100 to -60) to get the time of the peak
    # 2. average those times and make a new window
    # 3. use the new window to do the analysis by taking the mean in a 1msec wide window
    #    around the mean time
    slope_region = np.array(slope_region) + PSC.pulse_train["start"][0]
    print('slope region: ', slope_region)

    baseline_region=[
            PSC.pulse_train["start"][0] + PSC.T0 - 0.0005,
            PSC.pulse_train["start"][0] + PSC.T0,
        ]
    rgn_i = [int(baseline_region[i] / PSC.Clamps.sample_interval) for i in range(len(baseline_region))]

    V_cmd = PSC.Clamps.cmd_wave[:, rgn_i[0] : rgn_i[1]].mean(axis=1).view(np.ndarray)
    
    cmds = np.array(V_cmd) + PSC.AR.holding + PSC.JunctionPotential
    # bl, results = FN.mean_I_analysis(
    #     clamps=PSC.Clamps,
    #     region=baseline_region,
    #     mode="baseline",
    #     reps=[0],
    # )

    data1, tb = FN.get_traces(
        PSC.Clamps,
        region=[0,0.5], # slope_region,
        trlist=None,
        baseline=bl,
        intno=0,
        nint=1,
        reps=PSC.reps,
        slope=False,
    )
    if data1.ndim == 1:
        return False
    PSC.plot_data(tb, data1)

    ind = np.argmin(np.fabs(cmds - PSC.AMPA_voltage)) # find index closest to the reference AMPA voltage for measurement

    PSC.T1 = PSC.T0 + 0.010  # note that this is a narrow time window to use - 10 msec.

    p1delay = PSC.pulse_train["start"][0] + PSC.T0
    p1end = PSC.pulse_train["start"][0] + PSC.T1 
    p1_region = [p1delay, p1end]

    nmdelay = PSC.pulse_train["start"][0] + ndelay
    i_mean, results = FN.mean_I_analysis(
        clamps=PSC.Clamps,
        region=p1_region, mode="min", baseline=bl, reps=PSC.reps, slope=False
    )
    if i_mean is None:
        return False
    # if len(PSC.i_argmin) < 1:
    #     return False

    mintime = results.i_argmin[ind] * dt  # get AMPA peak index in the window
    print(f"AMPA mintime @ {PSC.AMPA_voltage*1e3:.1f} mV: {mintime*1e3:.3f} ms")

    # values for nmda analysis are currently fixed
    i_nmda_mean, results = FN.mean_I_analysis(
        clamps=PSC.Clamps,
        region=[nmdelay - nwidth, nmdelay + nwidth],
        mode="mean",
        baseline=bl,
        reps=PSC.reps,
        slope=False,
    )

    PSC.analysis_summary[f"PSP_VDEP_AMPA"][0] = PSC.sign * i_mean
    PSC.analysis_summary[f"PSP_VDEP_NMDA"][0] = PSC.sign * i_nmda_mean
    stimamp.append(PSC.pulse_train["amplitude"][0])
    stimintvl.append(PSC.pulse_train["period"][0])

    # print('ampa window & mean: ', [p1delay, p1end], i_mean)
    # print('nmda window & mean: ', [nmdelay-nwidth, nmdelay+nwidth], i_nmda_mean)

    # find -80 and +30 voltage indices (so we can save them and save the data)
    iAMPA = np.argmin(np.fabs(-PSC.AMPA_voltage + cmds))
    iNMDA = np.argmin(np.fabs(-PSC.NMDA_voltage + cmds))
    # print(iAMPA, iNMDA)
    # print('-90 mV found closest command: ', cmds[iAMPA])
    # print('+50 mV found closest command: ', cmds[iNMDA])
    if data1 is None or iNMDA >= data1.shape[0]:

        PSC.analysis_summary["Vindices"] = {"vAMPA": np.nan, "vNMDA": np.nan}
        PSC.analysis_summary["NMDAAMPARatio"] = np.nan
        PSC.analysis_summary["AMPA_NMDA_traces"] = {
            "T": None,
            "iAMPA": None,
            "iNMDA": None,
        }
    else:
        # print('data1 shape: ', data1.shape, iAMPA, iNMDA, cmds[iAMPA], cmds[iNMDA])
        # print(PSC.analysis_summary[f'PSP_VDEP_AMPA'])
        PSC.analysis_summary["Vindices"] = {"-90": iAMPA, "50": iNMDA}
        PSC.analysis_summary["NMDAAMPARatio"] = (
            PSC.analysis_summary[f"PSP_VDEP_NMDA"][0][iNMDA]
            / PSC.analysis_summary[f"PSP_VDEP_AMPA"][0][iAMPA]
        )
        PSC.analysis_summary["AMPA_NMDA_traces"] = {
            "T": tb,
            "iAMPA": data1[iAMPA],
            "iNMDA": data1[iNMDA],
        }
    PSC.analysis_summary["meas_times"] = {"tAMPA": mintime, "tNMDA": ndelay}
    PSC.analysis_summary["psc_stim_amplitudes"] = np.array(stim_I)
    PSC.analysis_summary["psc_intervals"] = np.array(stimintvl)
    PSC.analysis_summary["stim_times"] = PSC.pulse_train["start"]
    PSC.analysis_summary["Vcmd"] = cmds
    PSC.analysis_summary["Window"] = [PSC.T0, PSC.T1]
    return True