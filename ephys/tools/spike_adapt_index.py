""" spike adaptation index analysis

This is a one-off tool to test the spike analysis index analysis
across a population of cells (Reggie Edwards CBA-Age project). 



"""

import datetime

import pandas as pd
import os
from pathlib import Path
import matplotlib.pyplot as mpl
import ephys.tools.filename_tools as FT
import numpy as np
import seaborn as sns
from ephys.tools.get_configuration import get_configuration
from ephys.tools.check_inclusions_exclusions import include_exclude
import ephys.tools.parse_ages as parse_ages
import ephys.gui.data_table_functions as functions
import pylibrary.plotting.plothelpers as PH

git_hash = (
    functions.get_git_hashes()
)  # get the hash for the current versions of ephys and our project
runpath = "/Users/pbmanis/Desktop/Python/RE_CBA"
os.chdir(runpath)

config_file_path = "./config/experiments.cfg"
datapath = "/Volumes/Pegasus_004/Manislab_Data3/Edwards_Reginald/RE_datasets/CBA_Age/pyramidal"
(
    datasets,
    experiments,
) = get_configuration(
    config_file_path
)  # retrieves the configuration file from the running directory

experiment = experiments["CBA_Age"]
inclusion_dict = experiment["includeIVs"]
exclusion_dict = experiment["excludeIVs"]

age_dict = experiment["age_categories"]
# cellprotocol = "2023.09.12_000/slice_003/cell_000"
# cellid  = "2023_11_13_S1C1"
# cellprotocol = FT.make_cellid_from_slicecell(cellid)

# cellid_pkl = f"{cellid:s}_pyramidal_IVs.pkl"

protocol = "CCIV_long_HK_000"
tr_delay = 0.100
tr_dur = 0.5
protocol = "CCIV_1nA_max_1s_pulse_000"
tr_delay = 0.150
tr_dur = 1.0


def categorize_ages(age: int, experiment: dict):
    age_category = "NA"
    for k in experiment["age_categories"].keys():
        if age >= experiment["age_categories"][k][0] and age <= experiment["age_categories"][k][1]:
            age_category = k
    return age_category


def adaptation_index(spk_lat, tr_dur):
    ai = (-2.0 / len(spk_lat)) * np.sum((spk_lat / tr_dur) - 0.5)
    return ai


def get_adapt_index_to_hist(
    datapath, protocol, experiment, rate_bounds: list = [20, 40], ax_index=None, ax_rate=None
):
    p = Path(datapath)
    pcolors = sns.color_palette("tab20")
    all_files = p.glob("*.pkl")
    wait_to_show = False
    if ax_index is None and ax_rate is None:
        f, ax = mpl.subplots(1, 2, figsize=(8, 5))
        ax = ax.ravel()
    else:
        ax = [ax_index, ax_rate]
        wait_to_show = True
    for i in range(2):
        ax[i].spines.top.set_visible(False)
        ax[i].spines.right.set_visible(False)
    # print(ax)
    # exit()
    maxn = 14 * 14

    na = 0
    df_plot = pd.DataFrame(["age_category", "adapt_index", "rate"])
    for nfile, filename in enumerate(all_files):

        # print(filename)
        df = pd.read_pickle(filename, compression="gzip")
        if df["Spikes"] is None:
            # print(filename)
            # print("    ", "No Spikes")
            continue
        age = df.age
        print(age, parse_ages.age_as_int(parse_ages.ISO8601_age(age)))
        age_cat = categorize_ages(parse_ages.age_as_int(parse_ages.ISO8601_age(age)), experiment)
        if age_cat == "NA" or pd.isnull(age_cat):
            continue
        cellprotocols = list(df["Spikes"].keys())
        # print("    ", cellprotocols)

        adapt_indices = []
        rates = []
        for iprot, cellprotocol in enumerate(cellprotocols):
            if Path(cellprotocol).name == protocol:
                print(filename)
                print("    ", cellprotocol)
                spikes = df["Spikes"][cellprotocol]
                recnums = list(spikes["spikes"].keys())

                for rec in recnums:
                    spikelist = list(spikes["spikes"][rec].keys())
                    if len(spikelist) < 3:
                        continue
                    # if len(spikelist) < 5 or len(spikelist) > 10:
                    #     continue
                    spk_lat = np.array(
                        [
                            spikes["spikes"][rec][spk].AP_latency
                            for spk in spikelist
                            if spk is not None
                        ]
                    )
                    spk_lat = np.array([spk for spk in spk_lat if spk is not None])
                    # print("spk_lat: ", spk_lat)
                    if spk_lat is None:
                        continue
                    else:
                        spk_lat -= tr_delay
                    # print(spk_lat)
                    n_spikes = len(spk_lat)
                    rate = n_spikes / (spk_lat[-1] - spk_lat[0])
                    if rate < rate_bounds[0] or rate > rate_bounds[1]:
                        continue
                    rates.append(rate)
                    adapt_indices.append(adaptation_index(spk_lat, tr_dur))

                    # print("adaptation index: ", adapt_indices[-1])
        if len(adapt_indices) > 0:
            d_dict = {"age": age_cat, "adapt_index": np.mean(adapt_indices), "rate": np.mean(rates)}
            df_d_dict = pd.DataFrame([d_dict])
            df_plot = pd.concat(
                [df_plot, df_d_dict],
                ignore_index=True,
            )
        na += 1
        # if na > 20:
        #     break
    # df_plot = df_plot[df["age"].isin(list(experiment["age_categories"].keys()))]
    print(df_plot.age.unique())
    df_plot.dropna(subset=["age"], inplace=True)
    print(df_plot.age.unique())
    msize = 2.5
    sns.boxplot(
        x="age",
        y="adapt_index",
        data=df_plot,
        hue="age",
        palette=experiment["plot_colors"],
        order=experiment["plot_order"]["age_category"],
        saturation=0.25,
        zorder=50,
        ax=ax[0],
    )
    sns.swarmplot(
        x="age",
        y="adapt_index",
        data=df_plot,
        hue="age",
        palette=experiment["plot_colors"],
        hue_order=experiment["plot_order"]["age_category"],
        edgecolor="black",
        size=msize,
        linewidth=0.5,
        zorder=100,
        ax=ax[0],
        alpha=0.9,
    )
    sns.boxplot(
        x="age",
        y="rate",
        data=df_plot,
        hue="age",
        palette=experiment["plot_colors"],
        order=experiment["plot_order"]["age_category"],
        saturation=0.25,
        zorder=50,
        ax=ax[1],
    )
    sns.swarmplot(
        x="age",
        y="rate",
        data=df_plot,
        hue="age",
        palette=experiment["plot_colors"],
        hue_order=experiment["plot_order"]["age_category"],
        edgecolor="black",
        size=msize,
        linewidth=0.5,
        zorder=100,
        ax=ax[1],
        alpha=0.9,
    )
    # if rate_bounds[1] >= 100:
    # ax[1].set_ylim(0, 150)
    ax[1].set_ylim(0, 160)
    ax[0].set_ylim(-1.0, 1.0)
    ax[0].tick_params(axis="x", labelsize=6, rotation=45),
    ax[0].tick_params(axis="y", labelsize=6),
    ax[1].tick_params(axis="x", labelsize=6, rotation=45),
    ax[1].tick_params(axis="y", labelsize=6),

    if not wait_to_show:
        f.tight_layout()
        mpl.show()


def get_all_protocols(datapath):
    p = Path(datapath)
    pcolors = sns.color_palette("tab20")
    all_files = p.glob("*.pkl")
    f, ax = mpl.subplots(14, 14, figsize=(20, 15))
    maxn = 14 * 14
    ax = ax.ravel()
    na = 0
    for nfile, filename in enumerate(all_files):

        # print(filename)
        df = pd.read_pickle(filename, compression="gzip")
        if df["Spikes"] is None:
            # print(filename)
            # print("    ", "No Spikes")
            continue
        age = df.age
        print(age, parse_ages.age_as_int(parse_ages.ISO8601_age(age)))

        # print(df['Spikes'].keys())
        cellprotocols = list(df["Spikes"].keys())
        # print("    ", cellprotocols)

        for iprot, cellprotocol in enumerate(cellprotocols):
            if Path(cellprotocol).name == protocol:
                print(filename)
                print("    ", cellprotocol)
                spikes = df["Spikes"][cellprotocol]
                recnums = list(spikes["spikes"].keys())
                adapt_indices = []
                rates = []
                tr_dur = 0.5
                for rec in recnums:
                    spikelist = list(spikes["spikes"][rec].keys())
                    if len(spikelist) < 3:
                        continue
                    # if len(spikelist) < 5 or len(spikelist) > 10:
                    #     continue
                    spk_lat = np.array(
                        [
                            spikes["spikes"][rec][spk].AP_latency
                            for spk in spikelist
                            if spk is not None
                        ]
                    )
                    spk_lat = np.array([spk for spk in spk_lat if spk is not None])
                    # print("spk_lat: ", spk_lat)
                    if spk_lat is None:
                        continue
                    else:
                        spk_lat -= 0.10
                    # print(spk_lat)

                    rate = len(spk_lat) / (spk_lat[-1] - spk_lat[0])
                    if rate < 10.0:
                        continue
                    rates.append(rate)
                    adapt_indices.append(adaptation_index(spk_lat, tr_dur))
                    # print("adaptation index: ", adapt_indices[-1])

                ax[na].plot(
                    rates,
                    adapt_indices,
                    color=pcolors[iprot],
                    marker="o",
                    linestyle="-",
                    markersize=2,
                    linewidth=0.5,
                    clip_on=False,
                )
                ax[na].plot([20, 20], [-1, 1], color="gray", linestyle="--", linewidth=0.33)
                ax[na].plot([50, 50], [-1, 1], color="gray", linestyle="--", linewidth=0.33)
                ax[na].set_xlim(0, 500)
                ax[na].set_ylim(-1, 1)
                ax[na].spines.top.set_visible(False)
                ax[na].spines.right.set_visible(False)
                ax[na].tick_params(
                    axis="both",
                    which="both",
                    direction="in",
                    length=2,
                    right=False,
                    top=False,
                    labelsize=6,
                )
                ax[na].set_title(
                    f"{filename.name!s}\n{Path(cellprotocol).name:s}", fontsize=4, ha="center"
                )

                na += 1
                if na >= len(ax):
                    break
        if na >= len(ax):
            break
    f.tight_layout()
    mpl.show()


def superimpose_all_cells(datapath, protocol):

    p = Path(datapath)
    pcolors = sns.color_palette("tab20")
    all_files = p.glob("*.pkl")
    P = PH.regular_grid(
        rows=1,
        cols=1,
        order="rowsfirst",
        verticalspacing=0.05,
        horizontalspacing=0.05,
        margins={"leftmargin": 0.1, "rightmargin": 0.1, "topmargin": 0.1, "bottommargin": 0.1},
        figsize=(5, 5),
    )
    P.figure_handle.facecolor = "lightgrey"

    ax = P.axarr[0, 0]
    ax.set_facecolor("lightgrey")
    f = P.figure_handle
    na = 0
    for nfile, filename in enumerate(all_files):

        # print(filename)
        df = pd.read_pickle(filename, compression="gzip")
        if df["Spikes"] is None:
            # print(filename)
            # print("    ", "No Spikes")
            continue
        age = df.age
        # print(age, parse_ages.age_as_int(parse_ages.ISO8601_age(age)))
        age_cat = categorize_ages(parse_ages.age_as_int(parse_ages.ISO8601_age(age)), experiment)
        if age_cat == "NA" or pd.isnull(age_cat):
            continue
        # print(df['Spikes'].keys())
        cellprotocols = list(df["Spikes"].keys())
        # print("    ", cellprotocols)

        for iprot, cellprotocol in enumerate(cellprotocols):
            if Path(cellprotocol).name == protocol:
                print(filename)
                print("    ", cellprotocol)
                spikes = df["Spikes"][cellprotocol]
                recnums = list(spikes["spikes"].keys())
                adapt_indices = []
                rates = []
                tr_dur = 1.0
                for rec in recnums:
                    spikelist = list(spikes["spikes"][rec].keys())
                    if len(spikelist) < 3:
                        continue
                    # if len(spikelist) < 5 or len(spikelist) > 10:
                    #     continue
                    spk_lat = np.array(
                        [
                            spikes["spikes"][rec][spk].AP_latency
                            for spk in spikelist
                            if spk is not None
                        ]
                    )
                    spk_lat = np.array([spk for spk in spk_lat if spk is not None])
                    # print("spk_lat: ", spk_lat)
                    if spk_lat is None:
                        continue
                    else:
                        spk_lat -= 0.10
                    # print(spk_lat)

                    rate = len(spk_lat) / (spk_lat[-1] - spk_lat[0])
                    if rate < 10.0:
                        continue
                    rates.append(rate)
                    adapt_indices.append(adaptation_index(spk_lat, tr_dur))
                    # print("adaptation index: ", adapt_indices[-1])

                ax.plot(
                    rates,
                    adapt_indices,
                    color=experiment["plot_colors"][age_cat],
                    marker="o",
                    linestyle="-",
                    markersize=2,
                    linewidth=0.5,
                    clip_on=False,
                )
        # if nfile > 20:
        #     break

    ax.set_xlim(0, 500)
    ax.set_ylim(-1, 1)
    ax.plot([0, 500], [0, 0], color="gray", linestyle="--", linewidth=0.33)
    ax.spines.top.set_visible(False)
    ax.spines.right.set_visible(False)
    ax.tick_params(
        axis="both",
        which="both",
        direction="in",
        length=2,
        right=False,
        top=False,
        labelsize=8,
    )
    ax.set_xlabel("Firing Rate (Hz)", fontsize=8)
    ax.set_ylabel("Adaptation Index", fontsize=8)

    # ax.set_title(
    #     f"{filename.name!s}\n{Path(cellprotocol).name:s}", fontsize=4, ha="center"
    # )

    time_stamp_figure(f, git_hash)
    f.tight_layout()
    mpl.show()


# (np.sum(-2.0*(np.array(spk_lat)/tr_dur - 0.5))/len(spk_lat))


def get_spikes(filename):
    df = pd.read_pickle(filename, compression="gzip")
    print(df)
    print(df["Spikes"].keys())

    cellprotocols = list(df["Spikes"].keys())
    return df, cellprotocols


def plot_spikes(df, cellprotocols, filename):
    f, ax = mpl.subplots(len(cellprotocols), 1)
    if not isinstance(ax, list):
        ax = np.array(ax)
    for iprot, cellprotocol in enumerate(cellprotocols):
        spikes = df["Spikes"][cellprotocol]
        print("filename: ", filename)
        print("cell protocol: ", cellprotocol)
        # print(spikes.keys())
        # print(spikes['AP1_HalfWidth'])
        # print(len(spikes['spikes']))
        p1 = False
        ax[iprot].set_title(f"{filename.name!s}  {cellprotocol:s}", fontsize=8)
        for spike in spikes["spikes"]:
            spk = spikes["spikes"][spike]
            for spks in spk:
                this_spike = spikes["spikes"][spike][spks]
                # print(dir(this_spike))
                # exit()
                if this_spike.halfwidth is not None:
                    print(
                        f"{spike:4d} {this_spike.AP_number:4d}: {this_spike.AP_latency*1e3:6.1f} {this_spike.halfwidth*1e3:6.3f}, {this_spike.peak_V*1e3:6.1f}"
                    )
                    if this_spike.AP_number == 0:
                        ax[iprot].plot(
                            this_spike.Vtime - this_spike.Vtime[0], this_spike.V, linewidth=0.33
                        )
                print()
            # if spk['halfwidth'] is not None:
            #     print(f"Halfwidth: {spk.halfwidth}")
    f.tight_layout()
    mpl.show()


def time_stamp_figure(fig, git_hash):
    fig.text(
        0.98,
        0.01,
        f"spike_adapt_index.py {datetime.datetime.now()}",
        ha="right",
        va="top",
        fontsize=6,
    )
    fig.text(
        0.01,
        0.98,
        f"Project git hash: {git_hash['project']!s}\nephys git hash: {git_hash['ephys']!s}\n",
        ha="left",
        va="top",
        fontsize=6,
    )


def summary_hists():
    P = PH.regular_grid(
        rows=5,
        cols=2,
        order="rowsfirst",
        verticalspacing=0.05,
        horizontalspacing=0.05,
        margins={"leftmargin": 0.07, "rightmargin": 0.07, "topmargin": 0.05, "bottommargin": 0.05},
        figsize=(7, 12),
    )
    ax = P.axarr
    for i, rb in enumerate([[20, 40], [40, 60], [60, 80], [80, 100], [100, 150]]):
        get_adapt_index_to_hist(
            datapath, protocol, experiment, rate_bounds=rb, ax_index=ax[i, 0], ax_rate=ax[i, 1]
        )
    fig = P.figure_handle
    time_stamp_figure(fig, git_hash)
    fig.tight_layout()
    mpl.show()


if __name__ == "__main__":
    # fn = Path(datapath, cellid_pkl)
    # get_all_protocols(datapath)
    # fig, ax = mpl.subplots(5, 2, figsize=(7, 12))

    superimpose_all_cells(datapath, protocol)

    # get_adapt_index_to_hist(datapath, protocol, experiment, rate_bounds=[20, 40])
