import datetime
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple
from dataclasses import field

import matplotlib.pyplot as mpl
import numpy as np
import pandas as pd
import seaborn as sns
from lmfit import Model
from matplotlib.backends.backend_pdf import PdfPages

import ephys.tools.categorize_ages as CA
from ephys.tools import check_inclusions_exclusions as CIE
from ephys.tools import get_configuration
from pylibrary.plotting import plothelpers as PH



@dataclass
class Figure:
    fig_initiated: bool = False
    ax: mpl.Axes = None
    figure: mpl.figure = None
    do_fig: bool = False


def double_exp(x, dc, a1, r1, a2, r2):
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

    if d["Spikes"] is None:
        # ax.text(0.5, 0.5, s="No spikes found", ha="center",
        # va="center", fontdict={'color': 'red', 'size': 20})
        return None, None, fig

    #  fitting with double-exponential
    d2model = Model(double_exp)
    params = d2model.make_params(
        dc={"value": 0.2, "min": 0, "max": 1},
        a1={"value": 0.5, "min": -1, "max": 1},
        r1={"value": 0.1, "min": 0, "max": 20},
        a2={"value": 0.5, "min": -1, "max": 1},
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
                fig.ax[0].plot(lat, hw, "o", color=color, markersize=0.5, label=lab, linewidth=0.35)

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
    tau2 = []
    for f in fits:
        tau_1 = f.best_values.get("r1", np.nan)
        tau_2 = f.best_values.get("r2", np.nan)
        # force order of the values
        # note that the values are inverted. 
        if tau_2 < tau_1:
            tau_1, tau_2 = tau_2, tau_1  # swap
        tau1.append(tau_1)
        tau2.append(tau_2)
    # print("FITS: ",tau1, tau2)
    if fits is None or len(fits) == 0:
        fits = {"tau1": np.nan, "tau2": np.nan}
    else:
        tau1 = np.nanmean(tau1)
        tau2 = np.nanmean(tau2)
        fits =  {"tau1": tau1, "tau2": tau2}
    print("FITS returned: ", fits)
    return ss_hws, fits, fig


def spike_halfwidths(d, protocol_start_times, filename: Path):
    # print(f"Rs = {d['IV'][ivs[0]]['CCComp']['CCBridgeResistance']*1e-6}:.1f")
    # exit()
    f, ax = mpl.subplots(1, 1)
    ax.set_title(f"{str(Path(*filename.parts[-3:])):s}")
    f.text(0.95, 0.02, datetime.datetime.now(), fontsize=6, transform=f.transFigure, ha="right")
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

        for i, ns in enumerate(spks):
            if label not in labels:
                labels.append(label)
            else:
                labels.append("")
                # print(ns, len(spks[ns]))
            lat = []
            hw = []
            for j, sn in enumerate(spks[ns]):
                if spks[ns][sn].halfwidth is None:
                    continue
                if 1e6 * spks[ns][sn].halfwidth > 1000:  # long HW is artifact in analysis
                    continue
                hw.append(1e6 * spks[ns][sn].halfwidth)
                lat.append(spks[ns][sn].AP_latency - start_time)
                # print("    AP Latency: ", spks[ns][sn].AP_latency-start_time, " halfwidth: ")
            ax.plot(lat, hw, "o-", color=color, markersize=1, label=labels[-1], linewidth=0.35)
    ax.set_ylim(0, 1000)
    ax.set_xlim(-0.020, 1.0)
    ax.set_xlabel("AP Latency (s)")
    ax.set_ylabel("AP Halfwidth (us)")
    ax.legend(fontsize=5)


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
@dataclass
class CellData:
    cell_id: str
    age_category: str
    mean_hw: float = np.nan
    fits: list[object] = field(default_factory=list)
    tau1: float = np.nan
    tau2: float = np.nan


def hw_wrapper(adpath, exptname, celltype, experiment):
    datadir = Path(adpath, exptname, celltype)
    files = list(datadir.glob("*_IVs.pkl"))
    fig = Figure()
    # fig.do_fig = True

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
            d["age_category"] = CA.get_age_category(d["age"], age_cats)

            ss_hw, taus, fig = fit_spike_halfwidths(
                d,
                experiment["Protocol_start_times"],
                durations=experiment["protocol_durations"],
                filename=f,
                junction_potential=experiment["junction_potential"] * 1e-3,
                fig=fig,
                experiment=experiment,
            )
            if ss_hw is not None:
                hw = np.nanmean(ss_hw)
                if not isinstance(hw, float):
                    raise ValueError("mean hw is not a float")
                if hw is None or np.isnan(hw).all():
                    hw = np.nan
          
                group_hws_ss[d["age_category"]].append(
                    CellData(
                        cell_id=d.cell_id, age_category=d.age_category, mean_hw=hw, tau1=taus['tau1'], tau2=taus['tau2']
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
    df = pd.DataFrame(columns=["cell_id", "age_category", "mean_hw", "std_hw", "count", "tau1", "tau2"])
    for indx, g in enumerate(group_hws_ss.keys()):
        if len(group_hws_ss[g]) == 0:
            continue
        for j, cell_data in enumerate(group_hws_ss[g]):
            df.loc[len(df.index)] = [cell_data.cell_id, cell_data.age_category, cell_data.mean_hw, 0., 1., cell_data.tau1, cell_data.tau2]
    print(df.head)

    # plot_ss_hws(experiment)
    # save to a file, using the local data path (not the RAID drive)
    stats_dir = experiment.get("R_statistics_summaries", None)
    local_dir = Path(experiment.get("localdatapath", "."), stats_dir)
    if stats_dir is not None:
        df.to_csv(Path(local_dir, "spike_steady_state_halfwidths.csv"))

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

    df = pd.read_csv(Path(local_dir, "spike_steady_state_halfwidths.csv"))
    df = df[df["age_category"] != "ND"]

    hw_fig, hw_ax = mpl.subplots(3, 1, figsize=(6, 8))
    # sns.barplot(data=df, x="age_category", y="mean_hw", hue="age_category",
    #             palette=bar_color, alpha=0.33,
    #             order=bar_order, ax=hw_ax)
    bar_and_scatter(df, x="age_category", y="mean_hw", hue="age_category", experiment=experiment, ax=hw_ax[0])
    hw_ax[0].set_ylim(0, 0.6)
    hw_ax[0].set_ylabel("AP Halfwidth (s)")
    bar_and_scatter(df, x="age_category", y="tau1", hue="age_category", experiment=experiment, ax=hw_ax[1])
    hw_ax[1].set_ylim(0, 0.2)
    hw_ax[1].set_ylabel("Fast rate (s)")
    bar_and_scatter(df, x="age_category", y="tau2", hue="age_category", experiment=experiment, ax=hw_ax[2])
    hw_ax[2].set_ylim(0, 20.0)
    hw_ax[2].set_ylabel("Slow rate (s)")
    lab = ['A', 'B', 'C']
    for iax, ax in enumerate(hw_ax):
        PH.nice_plot(ax, direction="out", ticklength=3, position=-0.03)
        ax.set_xlabel("")
        ax.text(-0.1, 1.05, s=lab[iax], transform=ax.transAxes, fontsize=18, fontweight="bold")
    mpl.tight_layout
    mpl.show()


if __name__ == "__main__":
    configpath = Path("/Users/pbmanis/Desktop/Python/RE_CBA/config/experiments.cfg")
    exptname = "CBA_Age"
    celltype = "pyramidal"
    datasets, experiments = get_configuration.get_configuration(configpath)
    experiment = experiments["CBA_Age"]
    adpath = experiment["analyzeddatapath"]
    stats_dir = experiment.get("R_statistics_summaries", None)
    local_dir = Path(experiment.get("localdatapath", "."), stats_dir)
    print(local_dir.is_dir())
    if not local_dir.is_dir():
        raise FileNotFoundError(f"Local directory does not exist: {local_dir!s}")
    # hw_wrapper(adpath, exptname, celltype, experiment)
    plot_ss_hws(experiment)
    exit()

    # single cell:
    pyrdatapath = Path(adpath, exptname, celltype, "2023_01_25_S0C0_pyramidal_IVs.pkl")
    d = read_pkl_file(filename=pyrdatapath)
    ivs = list(d["Spikes"].keys())
    print(d.keys())

    # print(d['IV'].keys())
    # prots = list(d['IV'].keys())
    # print(d['IV'][prots[0]].keys())
    # print(d['IV'][prots[0]]['tauh_bovera'], d['IV'][prots[0]]['tauh_tau'],d['IV'][prots[0]]['tauh_Gh'])

    spike_halfwidths(
        d,
        experiment["Protocol_start_times"],
        filename=pyrdatapath,
        junction_potential=experiment["junction_potential"] * 1e-3,
    )
    mpl.show()
    # post_stimulus_spikes(d)
    # print_LCS_spikes(d)
