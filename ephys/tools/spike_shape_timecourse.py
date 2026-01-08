"""
ephys.tools.spike_shape_timecourse

This module provides tools to analyze the time course of spike shapes
through current pulses in an IV protocol.
It includes functions to fit spike half-widths over time using a double-exponential model,
and to plot the results.
It also produces a summary dataframe of spike half-widths categorized by age groups.

To use:
Make sure the configuration file is accessible. Set the experiment name and cell type.

8/27/2025 Paul B. Manis, Ph.D.

"""

import argparse
import datetime
from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as mpl
import numpy as np
import pandas as pd
import seaborn as sns
from lmfit import Model
from matplotlib.backends.backend_pdf import PdfPages
from pylibrary.plotting import plothelpers as PH

import ephys.tools.categorize_ages as CA
from ephys.tools import check_inclusions_exclusions as CIE
from ephys.tools import get_configuration


@dataclass
class CellData:
    cell_id: str
    age_category: str
    age: str
    sex: str
    mean_hw: float = np.nan
    mean_ahp: float = np.nan
    mean_dvdt_rise: float = np.nan
    mean_dvdt_fall: float = np.nan
    fits: list[object] = field(default_factory=list)
    amp1: float = np.nan
    tau1: float = np.nan
    amp2: float = np.nan
    tau2: float = np.nan
    dc: float = np.nan


@dataclass
class Figure:
    fig_initiated: bool = False
    ax: mpl.Axes = None
    figure: mpl.figure = None
    do_fig: bool = False


def double_exp(
    x: np.ndarray,
    dc: float = 0.0,
    a1: float = 0.0,
    r1: float = 0.001,
    a2: float = 0.0,
    r2: float = 0.1,
) -> np.ndarray:
    """double_exp _summary_

    Parameters
    ----------
    x : np.array
        Time array
    dc : float
        baseline offset
    a1 : float
        ampllitude of first exponential
    r1 : float
        rate constant for a1
    a2 : float
        amplitude of second exponential
    r2 : float
        rate constant for a2
    Returns
    -------
    np.array
        The value of the double exponential function at each point in x
        given the values of dc, a1, r1, a2, r2.
    """
    return a1 * np.exp(-x / r1) + a2 * np.exp(-x / r2) + dc


def fit_spike_halfwidths(
    d,
    protocol_start_times: dict,
    durations: dict,
    filename: Path,
    junction_potential: float,
    experiment: dict,
    window: list = [0.4, 0.5],
    fig: Figure = Figure(),
) -> list[list, list, dataclass]:
    """fit_spike_halfwidths Fit the time course of the halfwidth changes through
        the stimulus duration.
        Plots the time course of the halfwidths, dv/dt rise and fall, and AP threshold
        for EACH spike detected during the IV protocol, for the rate window specified.
    pr
        Does ONE cell (data is in d)

        Parameters
        ----------
        d : dictionary of analyzed values from pickle file
            includes 'IV', and 'Spikes' entries.
        protocol_start_times : dict
            _description_
        durations : dict
            _description_
        filename : Path
            _description_
        junction_potential : float
            _description_
        experiment : dict
            _description_
        window : list, optional
            _description_, by default [0.4, 0.5]
        fig : Figure, optional
            _description_, by default Figure()

        Returns
        -------
        list[list, list, dataclass]
            _description_
    """

    if d["Spikes"] is None:
        # ax.text(0.5, 0.5, s="No spikes found", ha="center",
        # va="center", fontdict={'color': 'red', 'size': 20})
        return None,  None, fig

    #  fitting with double-exponential
    d2model = Model(double_exp)
    params = d2model.make_params(
        dc={"value": 0.2, "min": 0, "max": 1},
        a1={"value": 0.5, "min": 0, "max": 1},
        r1={"value": 0.1, "min": 0, "max": 20},
        a2={"value": 0.5, "min": 0, "max": 1},
        r2={"value": 0.01, "min": 0, "max": 1},
    )

    prots = list(d["Spikes"].keys())
    ivs = list(d["IV"].keys())
    validivs, additional_ivs, additional_iv_records = CIE.include_exclude(
        d["cell_id"],
        inclusions=experiment["includeIVs"],
        exclusions=experiment["excludeIVs"],
        allivs=ivs,
    )

    colors = sns.color_palette("husl", max(len(prots), 3))
    ss_hws = []
    ss_rise = []
    ss_fall = []

    fits = []
    # first run through the protocols (top level)
    for ip, pn in enumerate(validivs):
        ivs = d["IV"][prots[ip]]
        pname = str(Path(pn).name)
        if pname[:-4] in protocol_start_times:
            start_time = protocol_start_times[pname[:-4]]

        else:
            start_time = 0
        if pname.find("1nA") > 0:
            color = colors[1]
        elif pname.find("4nA") > 0:
            color = colors[2]
        else:
            color = colors[0]
        if pname[:-4] in durations:
            dur = durations[pname[:-4]]
        else:
            dur = None
        Rs = ivs["CCComp"]["CCBridgeResistance"] * 1e-6
        Cp = ivs["CCComp"]["CCNeutralizationCap"] * 1e12
        spks = d["Spikes"][prots[ip]]["spikes"]

        label = f"{pname}: Rs={Rs:.1f}, Cp={Cp:.1f}"
        set_label = True
        # within each protocol, go through the traces with spikes
        for i, ns in enumerate(spks):
            lat = []
            hw = []
            dvdt_rise = []
            dvdt_fall = []
            ap_thr = []
            for j, sn in enumerate(spks[ns]):
                if spks[ns][sn].halfwidth is None:
                    continue
                if 1e3 * spks[ns][sn].halfwidth > 1.0:  # long HW is artifact in analysis
                    continue
                hw.append(1e3 * spks[ns][sn].halfwidth)
                lat.append(spks[ns][sn].AP_latency - start_time)
                dvdt_rise.append(spks[ns][sn].dvdt_rising)
                dvdt_fall.append(spks[ns][sn].dvdt_falling)
                ap_thr.append(spks[ns][sn].AP_begin_V - junction_potential)

            lat = np.array(lat)
            hw = np.array(hw)
            in_window = np.where((lat >= window[0]) & (lat <= window[1]))[0]
            ss_hws.extend(hw[in_window].tolist())
            ss_rise.extend(np.array(dvdt_rise)[in_window].tolist())
            ss_fall.extend(np.array(dvdt_fall)[in_window].tolist())

            if dur is None:
                continue

            # limit to firing rate window of 50 to 200 Hz
            if len(lat) / dur < 50.0 or len(lat) / dur > 200:
                continue

            if (
                fig.do_fig and not fig.fig_initiated
            ):  # only plot if we have data that fits in the range
                f, ax = mpl.subplots(4, 1, figsize=(6, 8))
                fig.ax = ax
                fig.figure = f
                # ax.set_title(f"{str(Path(*filename.parts[-3:])):s}")
                f.text(
                    0.95,
                    0.02,
                    datetime.datetime.now(),
                    fontsize=6,
                    transform=f.transFigure,
                    ha="right",
                )
                fig.fig_initiated = True
            if fig.do_fig:
                ax = fig.ax
                figure = fig.figure

            if set_label:
                lab = label  # only the first plot from each protocol gets tagged
                set_label = False
            else:
                lab = ""
            # fit to rising exponential
            fit = d2model.fit(hw, params, x=lat)
            fits.append(fit)
            # print("fit: \n", fit.fit_report())
            if fig.do_fig and fig.fig_initiated:
                fig.ax[0].plot(lat, hw, "o", color=color, linestyle=None, markersize=0.5, label=lab, linewidth=0.35)

                fig.ax[0].plot(lat, fit.best_fit, "-", color=color, linewidth=0.5)
                figax[1].plot(
                    lat, dvdt_rise, "o-", color=color, markersize=0.5, label=lab, linewidth=0.35
                )
                fig.ax[2].plot(
                    lat,
                    -np.array(dvdt_fall),
                    "o-",
                    color=color,
                    markersize=0.5,
                    label=lab,
                    linewidth=0.35,
                )
                fig.ax[3].plot(
                    lat, ap_thr, "o-", color=color, markersize=0.5, label=lab, linewidth=0.35
                )

    if fig.do_fig and fig.fig_initiated:
        ax = fig.ax
        ax[0].set_ylim(0, 1)
        for i in range(4):
            ax[i].set_xlim(-0.020, 1.0)
        ax[0].set_xlabel("AP Latency (s)")
        ax[0].set_ylabel("AP Halfwidth (ms)")
        ax[1].set_ylabel("AP dV/dt rise (V/s)")
        ax[1].set_ylim(0, 1000)
        ax[2].set_ylabel("AP dV/dt fall (V/s)")
        ax[2].set_ylim(800, 0)
        ax[3].set_ylabel("AP Threshold (V)")
        ax[3].set_ylim(-0.060, 0)
        ax[1].set_xlabel("AP Latency (s)")
        ax[2].set_xlabel("AP Latency (s)")
        ax[3].set_xlabel("AP Latency (s)")
        ax[0].legend(fontsize=5)
    tau1 = []
    amp1 = []
    tau2 = []
    amp2 = []
    dc = []
    for f in fits:
        tau_1 = f.best_values.get("r1", np.nan)
        amp_1 = f.best_values.get("a1", np.nan)
        tau_2 = f.best_values.get("r2", np.nan)
        amp_2 = f.best_values.get("a2", np.nan)
        dc0 = f.best_values.get("dc", np.nan)
        # force order of the values
        # note that the values are inverted.
        if tau_2 < tau_1:
            tau_1, tau_2 = tau_2, tau_1  # swap
            amp_1, amp_2 = amp_2, amp_1
        tau1.append(tau_1)
        amp1.append(amp_1)
        tau2.append(tau_2)
        amp2.append(amp_2)
        dc.append(dc0)
    # print("FITS: ",tau1, tau2)
    if fits is None or len(fits) == 0:
        fits = {"tau1": np.nan, "amp1": np.nan, "tau2": np.nan, "amp2": np.nan, "dc": 0}
    else:
        tau1 = np.nanmean(tau1)
        tau2 = np.nanmean(tau2)
        amp1 = np.nanmean(amp1)
        amp2 = np.nanmean(amp2)
        dc = np.nanmean(dc)
        fits = {"tau1": tau1, "amp1": amp1, "tau2": tau2, "amp2": amp2, "dc": dc}
    print("FITS returned: ", fits)
    return {"hw": ss_hws, "rise": ss_rise, "fall": ss_fall}, fits, fig


def plot_spike_measures(d, protocol_start_times, durations, filename: Path, measure="hw", ax: mpl.Axes=None, junction_potential: float=0.0):
    sc = 1.0
    if ax is None:
        f, ax = mpl.subplots(1, 1)
        f.text(0.95, 0.02, datetime.datetime.now(), fontsize=6, transform=f.transFigure, ha="right")
    ax.set_title(f"{str(Path(*filename.parts[-3:])):s}")
    if d["Spikes"] is None:
        ax.text(
            0.5,
            0.5,
            s="No spikes found",
            ha="center",
            va="center",
            fontdict={"color": "red", "size": 20},
        )
        return
    prots = list(d["Spikes"].keys())
    ivs = list(d["IV"].keys())
    colors = sns.color_palette("husl", max(len(prots), 3))

    labels = []
    for ip, pn in enumerate(prots):
        ivs = d["IV"][prots[ip]]
        if pn[:-4] in protocol_start_times:
            start_time = protocol_start_times[pn[:-4]]
        else:
            start_time = 0
        if pn[:-4] in durations:
            dur = durations[pn[:-4]]
        else:
            dur = None
        print("start time: ", start_time)
        if pn.find("1nA") > 0:
            color = colors[1]
        elif pn.find("4nA") > 0:
            color = colors[2]
        else:
            color = colors[0]
        # print(ivs['CCComp'].keys())
        Rs = ivs["CCComp"]["CCBridgeResistance"] * 1e-6
        Cp = ivs["CCComp"]["CCNeutralizationCap"] * 1e12
        spks = d["Spikes"][prots[ip]]["spikes"]
        label = f"{pn}: Rs={Rs:.1f}, Cp={Cp:.1f}"
        ylabel = "unknown"
        if measure == 'hw':
            ylabel = "AP Halfwidth (us)"
        elif measure == 'rise':
            ylabel = "AP dV/dt rise (V/s)"
        elif measure == 'fall':
            ylabel = "AP dV/dt fall (V/s)"
        elif measure == 'ap_thr':
            ylabel = "AP Threshold (V)"
        for i, ns in enumerate(spks):
            if label not in labels:
                labels.append(label)
            else:
                labels.append("")
                # print(ns, len(spks[ns]))
            lat = []
            meas = []
                
            for j, sn in enumerate(spks[ns]):
                val = None
                sc = 1
                if measure == "hw":
                    val = spks[ns][sn].halfwidth
                    sc = 1e6
                if measure == "rise":
                    val = spks[ns][sn].dvdt_rising
                if measure == "fall":
                    val = spks[ns][sn].dvdt_falling
                    sc = -1
                if val is None:
                    continue
                if sc * val > 1000:  # long HW is artifact in analysis
                    continue
                meas.append(sc * val)
                lat.append(spks[ns][sn].AP_latency - start_time)
                # print("    AP Latency: ", spks[ns][sn].AP_latency-start_time, " halfwidth: ")
                            # limit to firing rate window of 50 to 200 Hz
            if len(lat) / dur < 50.0 or len(lat) / dur > 200:
                continue

            ax.plot(lat, meas, "o", color=color, markersize=1, label=labels[-1], linewidth=0.35)
    ax.set_ylim(0, 1000)
    ax.set_xlim(-0.020, 1.0)
    ax.set_xlabel("AP Latency (s)")
    ax.set_ylabel(ylabel)
    ax.legend(fontsize=5)
    return sc


def post_stimulus_spikes(d):
    f, ax = mpl.subplots(len(d["Spikes"].keys()), 3, figsize=(10, 8))

    for axi, k in enumerate(d["Spikes"].keys()):
        print("protocol, data type: ", k, type(d["Spikes"][k]))
        print("    Spike array keys in protocol: ", d["Spikes"][k].keys())

        print("   pulse duration: ", d["Spikes"][k]["pulseDuration"])
        print("   poststimulus spike window: ", d["Spikes"][k]["poststimulus_spike_window"])
        print("  tstart: ", d["Spikes"][k]["poststimulus_spikes"])
        # print(d['Spikes'][k]['poststimulus_spikes'])
        for i, ivdata in d["IV"].items():
            print("\n   ", i, "\n", ivdata["RMP"], ivdata["RMPs"])

        print("    poststimulus spikes: ", d["Spikes"][k]["poststimulus_spikes"])
        for i in range(len(d["Spikes"][k]["poststimulus_spikes"])):
            nsp = len(d["Spikes"][k]["poststimulus_spikes"][i])
            iinj = d["Spikes"][k]["FI_Curve"][0][i]
            if nsp > 0:
                ax[axi, 0].plot(
                    d["Spikes"][k]["poststimulus_spikes"][i],
                    [iinj] * nsp,
                    marker="o",
                    markersize=2,
                    linestyle="None",
                )
            if nsp > 1:
                dur = (
                    d["Spikes"][k]["poststimulus_spikes"][i][-1]
                    - d["Spikes"][k]["poststimulus_spikes"][i][0]
                )
                ax[axi, 1].plot(iinj * 1e9, dur * 1e3, marker="o", markersize=2, linestyle="None")
                ax[axi, 1].set_xlabel("current (nA)")
                ax[axi, 1].set_ylabel("duration (ms)")
                rate = np.mean(1.0 / np.diff(d["Spikes"][k]["poststimulus_spikes"][i]))
                ax[axi, 2].set_title(f"{k} rate: {rate:.2f} Hz")
                ax[axi, 2].set_ylabel("rate (Hz)")
                ax[axi, 2].set_xlabel("current (nA)")
                ax[axi, 2].plot(iinj * 1e9, rate, marker="o", markersize=2, linestyle="None")
    mpl.tight_layout()
    mpl.show()


def print_LCS_spikes(d):
    # print(d['Spikes'][k]['spikes'].keys())
    # print("   ", k, "LCS: ", d['Spikes'][k]['LowestCurrentSpike'])
    for i, k in enumerate(d["Spikes"].keys()):
        if len(d["Spikes"][k]["LowestCurrentSpike"]) == 0:
            continue
        tr = d["Spikes"][k]["LowestCurrentSpike"]["trace"]
        dt = d["Spikes"][k]["LowestCurrentSpike"]["dt"]
        sn = d["Spikes"][k]["LowestCurrentSpike"]["spike_no"]
        # print("spike values for trace: ", tr, d['Spikes'][k]['spikes'][tr][sn])
        print("LCS spike data: ")

        print("LCS keys: ", d["Spikes"][k]["LowestCurrentSpike"].keys())
        print("   ", k, "LCS HW: ", d["Spikes"][k]["LowestCurrentSpike"]["AP_HW"])
        print("   ", k, "LCS AHP Depth: ", d["Spikes"][k]["LowestCurrentSpike"]["AHP_depth_V"])
        print("   ", k, "LCS AP Peak: ", d["Spikes"][k]["LowestCurrentSpike"]["AP_peak_V"])
        print(
            "   ", k, "LCS AP peak height: ", d["Spikes"][k]["LowestCurrentSpike"]["AP_peak_V"]
        )  #  - d['Spikes'][k]['LowestCurrentSpike']['AP_thr_V'])
        vpk = (
            d["Spikes"][k]["spikes"][tr][sn].peak_V * 1e3
        )  # peak value from the pkl file trace spike number
        vth = d["Spikes"][k]["spikes"][tr][sn].AP_begin_V * 1e3  # threshold value from the pkl file
        icurr = d["Spikes"][k]["spikes"][tr][sn].current * 1e12  # current injection value
        tro = d["Spikes"][k]["LowestCurrentSpike"]["AHP_depth_V"]

        print(
            "   ",
            f"{k:s}, trace: {tr:d} spike #: {sn:d}  peak V: {vpk:5.1f}, thr V: {vth:5.1f}, AP Height: {vpk-vth:5.1f}",
        )
        print(
            f"          AP Trough: {tro:f} current: {icurr:6.1f}"
        )  # confirm that the threshold value is the same

    print("   ", k, "AP1HW: ", d["Spikes"][k]["AP1_HalfWidth"])


def read_pkl_file(filename):

    filename = Path(filename)
    print(filename.is_file())
    d = pd.read_pickle(filename, compression="gzip")

    return d

    """ dataclass to hold info for one cell

    Raises
    ------
    ValueError
        _description_
    FileNotFoundError
        _description_
    """


def build_figure_framework():
    x = -0.1
    y = 1.07

    yht = 0.18
    xp = [0.08, 0.45, 0.72]
    yp = [0.975-(n+1)*0.22 for n in range(4)]
    xw = [0.30, 0.22, 0.22]
    yh = [yht]*4
    sizer = {}
    for ix in range(3):
        for iy in range(4):
            key = chr(65 + iy) + str(ix + 1)
            sizer[key] = {
                "pos": [xp[ix], xw[ix], yp[iy], yh[iy]],
                "labelpos": (x, y),
            }

    gr = [(a, a+1, 0, 1) for a in range(0, 12)] # just generate subplots - shape does not matter
    axmap = dict(zip(sizer.keys(), gr)) 

    P = PH.arbitrary_grid(sizer=sizer, showgrid=False, label=True, figsize=(11, 8.5))
    # PH.show_figure_grid(P)
    # mpl.show()
    # exit()
    return P


def compute_hw_all_cells(adpath, exptname, celltype, experiment):
    datadir = Path(adpath, exptname, celltype)
    files = list(datadir.glob("*_IVs.pkl"))
    # fig.do_fig = True
    fig = Figure()
    # steady-state halfwidths
    age_cats = experiment["age_categories"]
    # build dict to hold results sorted by group
    group_hws_ss = {
        "Preweaning": [],
        "Pubescent": [],
        "Young Adult": [],
        "Mature Adult": [],
        "Old Adult": [],
        "ND": [],
    }
    # prepare for PDF output if needed
    with PdfPages("spk_hwidths.pdf") as pdf:
        for nf, f in enumerate(files):
            print(nf, f)
            d = read_pkl_file(f)
            # if nf > 10:
            #     break
            # print(d.keys())
            print(dir(CA))
            d["age_category"] = CA.get_age_category(d["age"], age_cats)

            ss, taus, fig = fit_spike_halfwidths(
                d,
                experiment["Protocol_start_times"],
                durations=experiment["protocol_durations"],
                filename=f,
                junction_potential=experiment["junction_potential"] * 1e-3,
                fig=fig,
                experiment=experiment,
            )
            if ss is not None:
                hw = np.nanmean(ss["hw"])
                if not isinstance(hw, float):
                    raise ValueError("mean hw is not a float")
                if hw is None or np.isnan(hw).all():
                    hw = np.nan
                dvdt_rise = np.nanmean(ss["rise"])
                dvdt_fall = np.nanmean(ss["fall"])

                group_hws_ss[d["age_category"]].append(
                    CellData(
                        cell_id=d.cell_id,
                        age_category=d.age_category,
                        age=d.age,
                        sex=d.sex,
                        mean_hw=hw,
                        mean_dvdt_rise=dvdt_rise,
                        mean_dvdt_fall=dvdt_fall,
                        dc=taus["dc"],
                        tau1=taus["tau1"],
                        amp1=taus["amp1"],
                        tau2=taus["tau2"],
                        amp2=taus["amp2"],
                    )
                )
            if fig.do_fig and fig.fig_initiated:
                mpl.suptitle(f.stem)
                pdf.savefig()
        # # post_stimulus_spikes(d)
        # print_LCS_spikes(d)
    if fig.do_fig and fig.fig_initiated:
        mpl.close()
    # assemble into a dataframe
    df = pd.DataFrame(
        columns=[
            "cell_id",
            "age_category",
            "sex",
            "age",
            "mean_hw",
            "std_hw",
            "count",
            "dc",
            "amp1",
            "tau1",
            "amp2",
            "tau2",
        ]
    )
    for indx, g in enumerate(group_hws_ss.keys()):
        if len(group_hws_ss[g]) == 0:
            continue
        for j, cell_data in enumerate(group_hws_ss[g]):
            df.loc[len(df.index)] = [
                cell_data.cell_id,
                cell_data.age_category,
                cell_data.sex,
                cell_data.age,
                cell_data.mean_hw,
                0.0,
                1.0,
                cell_data.dc,
                cell_data.amp1,
                cell_data.tau1,
                cell_data.amp2,
                cell_data.tau2,
            ]
    print(df.head(10))

    # plot_ss_hws(experiment)
    # save to a file, using the local data path (not the RAID drive)
    stats_dir = experiment.get("R_statistics_summaries", None)
    local_dir = Path(experiment.get("localdatapath", "."), stats_dir)
    if stats_dir is not None:
        df.to_csv(Path(local_dir, "spike_steady_state_halfwidths_07-Jan-2026.csv"))
        print(
            "Saved spike halfwidth summary to: ",
            Path(local_dir, "spike_steady_state_halfwidths_07-Jan-2026.csv"),
        )


def bar_and_scatter(df: pd.DataFrame, x: str, y: str, hue: str, experiment: dict, ax: mpl.Axes):
    hue_category = "age_category"
    plot_order = experiment["plot_order"]["age_category"]
    plot_colors = experiment.get("plot_colors", {})
    palette = plot_colors["symbol_colors"]
    bar_color = plot_colors.get("bar_background_colors", None)
    # bar_order = plot_colors.get("age_category", None)
    # line_colors =plot_colors.get("line_plot_colors", None)
    edgecolor = (plot_colors["symbol_edge_color"],)
    linewidth = (plot_colors["symbol_edge_width"],)
    sns.boxplot(
        data=df,
        x=x,
        y=y,
        hue=x,
        hue_order=plot_order,
        palette=plot_colors["bar_background_colors"],
        ax=ax,
        order=plot_order,
        saturation=float(plot_colors["bar_saturation"]),
        width=plot_colors["bar_width"],
        orient="v",
        showfliers=False,
        linewidth=plot_colors["bar_edge_width"],
        zorder=50,
        dodge=experiment["dodge"][hue_category],
        # clip_on=False,
    )
    sns.stripplot(
        data=df,
        x=x,
        y=y,
        hue=x,
        order=plot_order,
        palette=plot_colors["symbol_colors"],
        edgecolor=edgecolor,
        linewidth=linewidth,
        size=plot_colors["symbol_size"] * 2,
        alpha=1.0,
        ax=ax,
        zorder=100,
        clip_on=False,
    )


def plot_ss_hws(experiment):
    stats_dir = experiment.get("R_statistics_summaries", None)
    local_dir = Path(experiment.get("localdatapath", "."), stats_dir)

    df = pd.read_csv(Path(local_dir, "spike_steady_state_halfwidths_07-Jan-2026.csv"))
    df = df[df["age_category"] != "ND"]

    hw_fig, hw_ax = mpl.subplots(5, 1, figsize=(6, 8))
    # sns.barplot(data=df, x="age_category", y="mean_hw", hue="age_category",
    #             palette=bar_color, alpha=0.33,
    #             order=bar_order, ax=hw_ax)
    bar_and_scatter(
        df, x="age_category", y="mean_hw", hue="age_category", experiment=experiment, ax=hw_ax[0]
    )
    hw_ax[0].set_ylim(0, 0.5)
    hw_ax[0].set_ylabel("AP Halfwidth (s)")
    
    # =================
    bar_and_scatter(
        df, x="age_category", y="tau1", hue="age_category", experiment=experiment, ax=hw_ax[1]
    )
    hw_ax[1].set_ylim(0, 0.2)
    hw_ax[1].set_ylabel("Fast rate (s)")
    
    # =================
    bar_and_scatter(
        df, x="age_category", y="amp1", hue="age_category", experiment=experiment, ax=hw_ax[2]
    )
    hw_ax[2].set_ylim(-0.5, 0.5)
    hw_ax[2].set_ylabel("Fast amplitude (s)")
    
    # =================
    bar_and_scatter(
        df, x="age_category", y="tau2", hue="age_category", experiment=experiment, ax=hw_ax[3]
    )
    hw_ax[3].set_ylim(0, 20.0)
    hw_ax[3].set_ylabel("Slow rate (s)")
    
    # =================
    bar_and_scatter(
        df, x="age_category", y="amp2", hue="age_category", experiment=experiment, ax=hw_ax[4]
    )
    hw_ax[4].set_ylim(-0.50, 0.5)
    hw_ax[4].set_ylabel("Slow amplitude")
    
    # =================
    lab = ["A", "B", "C", "D", "E"]
    for iax, ax in enumerate(hw_ax):
        PH.nice_plot(ax, direction="out", ticklength=3, position=-0.03)
        ax.set_xlabel("")
        ax.text(-0.1, 1.05, s=lab[iax], transform=ax.transAxes, fontsize=18, fontweight="bold")
    mpl.tight_layout
    mpl.show()

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Analyze spike half-width time course through IV protocols")
    parser.add_argument(
        "-c",
        "--compute",
        action="store_true",
        help="Compute spike half-widths for all cells and save summary",
    )
    parser.add_argument(
        "-s",
        "--summary",
        action="store_true",
        help="Plot spike half-width summary from saved data",
    )
    parser.add_argument(
        "-f3",
        action = "store_true",
        help="Plot supplemental figure 3",
    )

    args = parser.parse_args()


    configpath = Path("/Users/pbmanis/Desktop/Python/RE_CBA/config/experiments.cfg")
    exptname = "CBA_Age"
    celltype = "pyramidal"
    datasets, experiments = get_configuration.get_configuration(configpath)
    experiment = experiments["CBA_Age"]
    adpath = experiment["analyzeddatapath"]
    stats_dir = experiment.get("R_statistics_summaries", None)
    local_dir = Path(experiment.get("localdatapath", "."), stats_dir)
    if not local_dir.is_dir():
        raise FileNotFoundError(f"Local directory does not exist: {local_dir!s}")
  
    if args.compute:
        compute_hw_all_cells(adpath, exptname, celltype, experiment)
        exit()

    if args.summary:
        plot_ss_hws(experiment)
        exit()

    if args.f3:
        P = build_figure_framework()

        # single cell:
        df = pd.read_csv(Path(local_dir, "spike_steady_state_halfwidths_07-Jan-2026.csv"))
        df = df[df["cell_id"] == "2023.09.06_000/slice_000/cell_001"]
        pyrdatapath = Path(adpath, exptname, celltype, "2023_09_06_S0C1_pyramidal_IVs.pkl")
        d = read_pkl_file(filename=pyrdatapath)
        ivs = list(d["Spikes"].keys())
        print(d.keys())

        # print(d['IV'].keys())
        # prots = list(d['IV'].keys())
        # print(d['IV'][prots[0]].keys())
        # print(d['IV'][prots[0]]['tauh_bovera'], d['IV'][prots[0]]['tauh_tau'],d['IV'][prots[0]]['tauh_Gh'])
        meas_dict = {"A1": "hw", "B1": "rise", "C1": "fall"}
        for k, v in meas_dict.items():
            scale = plot_spike_measures(
                d,
                experiment["Protocol_start_times"],
                durations=experiment["protocol_durations"],
                filename=pyrdatapath,
                measure=v,
                junction_potential=experiment["junction_potential"] * 1e-3,
                ax=P.axdict[k],
            )
            fitx = np.linspace(0, 1.0, 10)
            print(df)
            fity = double_exp(
                fitx,
                dc=df["dc"].values[0],
                a1=df["amp1"].values[0],
                r1=df["tau1"].values[0],
                a2=df["amp2"].values[0],
                r2=df["tau2"].values[0],
            )
            
            P.axdict[k].plot(fitx, fity*1e3, "r--", linewidth=1.5)
        mpl.show()
    # post_stimulus_spikes(d)
    # print_LCS_spikes(d)
