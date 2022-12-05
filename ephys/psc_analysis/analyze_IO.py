import numpy as np
from pathlib import Path
from typing import List
from collections import OrderedDict

def analyze_IO(
    PSC,
    rmpregion: List = [0.0, 0.05],
    twidth: float = 0.05,
    deadtime: float = 0.0007,
    protocolName: bool = None,
    device: str = "Stim0",
):
    """Analyze in input=output relationship for a specific driving device"""
    
    filekey = Path(PSC.make_key(PSC.datapath))
    # check the db to see if we have parameters already
    dfiles = PSC.db["date"].tolist()
    if filekey in dfiles:
        delay = PSC.db.loc[filekey, "date"]["T0"]
        t1 = PSC.db.loc[filekey, "date"]["T1"]
        width = t1 - delay
    else:
        delay = 1.0 * 1e-3
        width = 15.0 * 1e-3
    PSC.sign = -1

    Stim_IO = np.tile(PSC.stim_io, len(PSC.reps))  # stimuli in order
    PSC.analysis_summary[f"PSP_IO"] = [[]] * len(
        PSC.pulse_train["start"]
    )  # create space for amplitude results, per pulse
    PSC.analysis_summary[f"psc_stim_amplitudes"] = [[]] * len(
        PSC.pulse_train["start"]
    )  # create space for amplitude results, per pulse
    stimintvl = []
    idat = [None] * len(PSC.pulse_train["start"])
    bl = PSC.get_baseline()
    for i in range(len(idat)):  # across each of the pulses in the train
        idat[i] = OrderedDict()  # storage for data for each stimulus level
        # pdelay = PSC.pulse_train["start"][i] + delay
        if (
            i == 0 and PSC.update_regions
        ):  # if PSC.update_region is set, then use cursor plot to get the regions
            rgn = PSC.set_region(
                [PSC.pulse_train["start"][i], PSC.pulse_train["start"][i] + twidth],
                baseline=bl,
            )
        else:  # normal operation, just use stock values
            rgn = [delay, delay + width]
        PSC.T0 = rgn[0]  # kind of bogus
        PSC.T1 = rgn[1]
        region = (
            np.array(rgn) + PSC.pulse_train["start"][i]
        )  # get region relative to start of this pulse
        for j in range(len(PSC.AR.traces)):  # for all traces
            mi = PSC.AR.trace_index[
                j
            ]  # get index into marked traces then compute the min value minus the baseline
            da = np.min(
                PSC.Clamps.traces["Time" : region[0]+deadtime : region[1]][j]
            ) - np.mean(PSC.Clamps.traces["Time" : rmpregion[0] : rmpregion[1]][j])
            if Stim_IO[mi] not in list(idat[i].keys()):
                idat[i][Stim_IO[mi]] = [da]
            else:
                idat[i][Stim_IO[mi]].append(da)
        for j in range(len(PSC.AR.traces)):
            mi = PSC.AR.trace_index[j]
            idat[i][Stim_IO[mi]] = np.mean(
                idat[i][Stim_IO[mi]]
            )  # replace with the mean value for that stimulus level within the protocol

        PSC.analysis_summary[f"PSP_IO"][i] = (
            PSC.sign * 1e12 * np.array([idat[i][k] for k in idat[i].keys()])
        )
        PSC.analysis_summary[f"psc_stim_amplitudes"][i] = 1e6 * np.array(
            [k for k in idat[i].keys()]
        )
        stimintvl.append(PSC.pulse_train["period"][0])

    stim_dt = np.diff(PSC.pulse_train["start"])
    # PSC.analysis_summary['psc_stim_amplitudes'] = 1e6*np.array(stim_io)
    PSC.analysis_summary["psc_intervals"] = np.array(stimintvl)
    PSC.analysis_summary["ppf_dt"] = np.array(stim_dt)
    PSC.analysis_summary["stim_times"] = PSC.pulse_train["start"]
    PSC.analysis_summary["window"] = [PSC.T0, PSC.T1]
    PSC.analysis_summary["Group"] = PSC.Group
    return True