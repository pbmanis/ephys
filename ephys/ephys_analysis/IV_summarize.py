"""IV_summarize:
A module to summarize data from individual cells generated through from IV_analysis
in the ephys.ephys_analysis library. IV_Analysis in turn computes results from individual
protocols using ephys_analysis RMTauSummary and SpikeSummary

This is part of the ephys IV analysis pipeline:
    dataSummary -> IV_Analysis -> IV_summarize

This program takes as input the output of IV_analysis (IV_analysis takes as input the output
of dataSummary, and an excel sheet with codes and annotations)

The output is a figure with FI curves, plots of general physiology (Rm, Rin, tau, tauh, etc),
and spike parameters. Data is sorted by "group" as set by the "codes" tab of the excel sheet.
The codes sheet should also include a subject id for completeness.

The data can be summarized either by Observational Unit (e.g., cell) or by Experimental Unit (e.g., subject,
with all cells averaged together). Which is appropriate will depend on the experimental design. For example,
EU is the appropriate mode when dealing with KO or transgenic mice where the same manipulation is applied
to all cells. OU is the approopriate mode when dealing with sparse expression (e.g., tamoxifen-induced 
recombination) where the cells in a given subject may have different gene expression patterns, and the
expression pattern can be identified during data collection. 
Currently, there is no "intermediate" mode, where all of the OU data are computed for a subject, and then
averaged with each EU. 

 
This module can be used in two ways:
1. Command line - running from the command line. This may require changing the default "experiments" dictionary
    (see the main() function)

2. As a class (module): Import, setup and run
    g = GetAllIVs(arguments_dict)
    g.set_protocols(["CCIV_long_HK", "CCIV_1nA_max_1s", "CCIV_200pA"])
    g.set_experiments(experiments)
    g.run()

    The arguments dict should hold the base directory for the data, outputfilename, celltype, etc. 
    "experiments" is a list of dictionaries that specify where the input data
    (outputs of will be found and where the
    output files should go.

Returns:
    Nothing

This program writes several files (an output PDF figures) and prints some statistics to the terminal.
"""

import argparse
import datetime
import os
import subprocess
from collections import OrderedDict
from pathlib import Path
from typing import Union

import matplotlib
import matplotlib.cm
import matplotlib.pyplot as mpl
import numpy as np
import pandas as pd
import pylibrary.plotting.plothelpers as PH
import scipy.stats
import seaborn as sns
from pylibrary.tools import cprint as CP

rcParams = matplotlib.rcParams
rcParams["svg.fonttype"] = "none"  # No text as paths. Assume font installed.
rcParams["pdf.fonttype"] = 42
rcParams["ps.fonttype"] = 42
rcParams["text.usetex"] = False

cprint = CP.cprint


# spikes: dict_keys(['FI_Growth', 'AdaptRatio', 'FI_Curve', 'FiringRate',
# 'AP1_Latency', 'AP1_HalfWidth', 'AP2_Latency', 'AP2_HalfWidth',
# 'FiringRate_1p5T', 'AHP_Depth', 'spikes', 'iHold', 'pulseDuration',
# 'baseline_spikes', 'poststimulus_spikes']) IV: dict_keys(['holding', 'WCComp',
# 'CCComp', 'BridgeAdjust', 'RMP', 'RMPs', 'Irmp', 'taum', 'taupars', 'taufunc',
# 'Rin', 'Rin_peak', 'tauh_tau', 'tauh_bovera', 'tauh_Gh', 'tauh_vss'])


# disk = {'Tamalpais2': '/Volumes/Samsung T5/data', 'Lytle':
# '/Volumes/Pegasus/ManisLab_Data3/Kasten_Michael/'}


def get_computer():
    if os.name == "nt":
        computer_name = os.environ["COMPUTERNAME"]
    else:
        computer_name = subprocess.getoutput("scutil --get ComputerName")
    return computer_name


computer_name = get_computer()

# experiments = SEP.get_experiments()


# measures to plot:
measures = [
    "AP1_HalfWidth",
    "AP1_HalfWidth_interpolated",
    "AP2_HalfWidth",  #'AP2_HalfWidth_interpolated',
    "peak_V",
    "AHP_Depth",
    "peaktotrough",
    "spikethreshold",
    "AP1_Latency",
    "AP2_Latency",
    "Ibreak",
    "Irate",
    "FiringRate_1p5T",
    "AdaptRatio",
    "maxrate",
    "age",
    "FR_Slope",
]

# # measures that are obtained specificially from the 'spikes' dict
# measures_fromspikes = ['AP1_Latency', 'AP2_Latency', 'AP1_HalfWidth',
#             'AP1_HalfWidth_interpolated', 'AP2_HalfWidth',
#             'AP2_HalfWidth_interpolated', 'AHP_Depth', 'FiringRate_1p5T',
#             'peak_V', 'peaktotrough', 'spikethreshold', 'maxrate']

palette = {
    "A": [0, 0.5, 0],
    "B": [0, 0, 0.5],
    "Z": [0.8, 0.8, 0.8],
    "C": [0.5, 0.5, 0.0],
    "D": [0, 0.5, 0.5],
    "W": [1.0, 0, 0],
    "AA": [0, 1, 0],
    "AAA": [0, 0, 0],
    "?": [0, 0, 0],
}

paxes = dict.fromkeys(measures)
paxes["AdaptRatio"] = [0.0, 10.0]
paxes["AP1_Latency"] = [0, 100]
paxes["AP2_Latency"] = [0, 150]
paxes["AP1_HalfWidth_interpolated"] = [0, 4]
paxes["AP2_HalfWidth_interpolated"] = [0, 4]
paxes["AP1_HalfWidth"] = [0, 4]
paxes["AP2_HalfWidth"] = [0, 4]
paxes["AHP_Depth"] = [0, 25]
paxes["FiringRate_1p5T"] = [0, 100]
paxes["Ibreak"] = [-0.00, 1.0]
paxes["Irate"] = [0, 10.00]
paxes["peak_V"] = [-0.040, 0.080]
paxes["peaktotroughT"] = [0, 0.040]
paxes["peaktotrough"] = [0, 0.040]
paxes["spikethreshold"] = [0, 400.0]
paxes["maxrate"] = [0, 150]
paxes["age"] = [0, 1000]
paxes["FR_Slope"] = [0, 1000]

iv_measures = ["Rin", "RMP", "taum", "tauh_bovera", "tauh_tau"]

paxesb = dict.fromkeys(iv_measures)
paxesb["Rin"] = [0, 750]
paxesb["RMP"] = [-90, -40]

paxesb["taum"] = [0, 0.050]
paxesb["tauh_bovera"] = [0, 1.5]
paxesb["tauh_tau"] = [0, 0.5]


class GetAllIVs:
    def __init__(self, args):
        """Set up for analysis

        Params: 
        args: dict
            must contain mode; others are filled with defaults if not specified

        Returns:
            _type_: _description_
        """
        self.coding = {}
        self.mode = args['mode']
        if 'celltype' in args.keys():
            self.celltype = args['celltype']
        else:
            celltype = '*'
        self.experiment = args['experiment']
        if 'protocols' in args.keys():
            self.protocols = args['protocols']
        else:
            self.protocols = ["*"]
        if 'basedir' in args.keys():
            self.basedir = args['basedir']
        else:
            self.basedir = None
        if 'outputFilename' in args.keys():
            self.outputFilename = args['outputFilename']
        else:
            self.outputFilename = None
        if 'slicecell' in args.keys():
            self.slicecell = args['slicecell']
        else:
            self.slicecell = None


        self.dfs = None
        self.rainbowcolors = None
        self.code_names = ["A", "B"]
        self.codecolors = OrderedDict(
            [
                ("A", "limegreen"),
                ("B", "violet"),
                ("C", "cornflowerblue"),
                ("D", "orange"),
                ("Z", "red"),
                ("P", "brown"),
                (0.0, "red"),
                ("W", "red"),
                ("AA", "green"),
                ("AAA", "black"),
                ("?", "red"),
            ]
        )
        self.codes = list(self.codecolors.keys())

    def set_experiments(self, experiments: Union[dict, list]):
        """experiments is the dict for experiments - paths, etc. each dict entry has a
            key naming the "experiment" The list has [path_to_data,
            nf107_directory_for_database, database_file_name,
            experiment_coding_dictionary]

        For example, experiments might contain
            "Ank2B": {
                "datadisk": "/Volumes/Pegasus_002/Kasten_Michael/Maness_Ank2_PFC_stim",  # where to find the data
                "directory": "ANK2",  # directory (here, relative) to look for excel and pkl files
                "resultdisk": "ANK2",  # directory to store results
                "db_directory": "ANK2",  # directory to store database
                "datasummary": "Intrinsics",  # name of the file generated by datasummary
                "IVs": "ANK2_NEX_IVs",  # name of the excel sheet generated by (nf107ivs) with the data
                "coding_file": "Intrinsics.xlsx",  # name of the excel workbook that holds the codes
                "coding_sheet": "codes",  # name of the sheet in the workbook that has the code mapping (Date, slice_slice, cell_cell, code)

        Experiments can have multiple entries as well, which are selected by using '+' when specifying the
        "-E" option from the command line (e.g., 'nf107_nihl+nf107_control')
        Args:
            experiments (dict): Dictionary for experiments, data directories, etc.
        """

        self.experiments = experiments

    def build_dataframe(self):
        """Build pandas data frame from selected experiments

        Args:
            args (_type_): _description_
        """

        self.dfs = pd.DataFrame()
        if self.experiment is not None:
            if "+" in self.experiment:
                expts = self.experiment.split("+")
            else:
                expts = [self.experiment]
            for i in range(len(expts)):
                cprint("g", f"Analyzing experiment: {str(expts[i]):s}")
                self.basedir = Path(self.experiments[expts[i]]["datadisk"])
                self.inputFilename = Path(
                    self.experiments[expts[i]]["directory"],
                    self.experiments[expts[i]]["datasummary"],
                ).with_suffix(".pkl")
                self.outputPath = Path(self.experiments[expts[i]]["directory"])
                coding_f = self.experiments[expts[i]]["coding_file"]
                df_c = pd.read_excel(
                    Path(self.experiments[expts[i]]["directory"], coding_f),
                    sheet_name=self.experiments[expts[i]]["coding_sheet"],
                )
                df_i = pd.read_pickle(self.inputFilename)
                select = self.celltype
                if select not in ["any", '*']:
                    df_i = df_i[df_i["cell_type"] == select]
                sLength = len(df_i["date"])
                df_i = self.parse_ages(df_i)

                df_i = df_i.assign(source=expts[i])
                df_i.reset_index(drop=True)
                if coding_f is not None:
                    df_i = pd.merge(
                        df_i,
                        df_c,
                        left_on=["date", "slice_slice", "cell_cell"],
                        right_on=["Date", "slice_slice", "cell_cell"],
                    )
                    df_i.coding = df_i.coding.str.strip()
                self.dfs = pd.concat((self.dfs, df_i))
        self.dfs = self.dfs.reset_index(drop=True)

        self.dfs.to_excel("test.xlsx")
        cprint("g", "All data loaded")
        self.rainbowcolors = iter(
            matplotlib.cm.rainbow(np.linspace(0, 1, len(self.dfs)))
        )
        return

    def run(self):

        if self.dfs is None:
            self.build_dataframe()

        self.PFig = PH.regular_grid(
            1,
            1,
            order="columnsfirst",
            figsize=(11.0, 8.0),
            showgrid=False,
            verticalspacing=0.08,
            horizontalspacing=0.08,
            margins={
                "leftmargin": 0.08,
                "rightmargin": 0.6,
                "topmargin": 0.08,
                "bottommargin": 0.60,
            },
            labelposition=(-0.05, 1.05),
            parent_figure=None,
            panel_labels=["A"],
        )

        self.plot_FI(self.PFig)
        P1 = self.plot_IV_info(self.mode, parentFigure=self.PFig)
        P2 = self.plot_spike_info(self.mode, parentFigure=P1)
        mpl.show()
        exit()

    def parse_ages(self, df):
        """
        Systematize the age representation
        """
        adat = []
        for a in df["age"]:
            a = a.strip()
            if a.endswith("?"):
                a = a[:-1]
            if a.endswith("ish"):
                a = a[:-3]
            if a == "?" or len(a) == 0:
                a = "0"
            if a.startswith("p") or a.startswith("P"):
                try:
                    a = int(a[1:])
                except:
                    a = 0
            elif a.endswith("d") or a.endswith("D"):
                a = int(a[:-1])
            elif a == " ":
                a = 0

            else:
                a = int(a)
            adat.append(a)
        df["age"] = adat
        return df

    def set_protocols(self, protocols: list):
        """Set the list of protocols that we will use for the analysis

        Args:
            protocols (list): protocol names. Use ['*'] to use all protocols in
            spikes and iv_curve
        """
        self.protocols = protocols

    def select_protocols(self, prot: Union[Path, str]):
        if self.protocols[0] == "*":
            return True
        for p in self.protocols:
            if Path(prot).name.startswith(p):
                return True
        return False

    def plot_FI(self, PF):
        # plot all FI Curves in the dataset, superimposed

        codecnt = -1
        ncells = 0
        legcodes = []

        for idx in self.dfs.index:

            dx = self.dfs.at[idx, "Spikes"]  # get the spikes dict
            code = self.dfs.at[idx, "coding"]
            if isinstance(code, list):
                code = code[0]

            if code not in self.code_names:
                continue
            color = self.codecolors[code]

            if isinstance(dx, dict):  # convert to 1-element list
                dx = [dx] 
            for dv in dx:
                for fidata in dv.keys():
                    if not isinstance(fidata, Path):
                        cprint('r', f"fidata is not path:  {str(fidata):s}")
                        continue # fidata = Path(fidata)
                    if not self.select_protocols(fidata):
                        continue
                    if code not in legcodes:
                        legcodes.append(code)
                        leglabel = code
                    else:
                        leglabel = None
                    fi = dv[fidata]["FI_Curve"]
                    lp = PF.axdict["A"].plot(
                        fi[0] * 1e9,
                        fi[1] / dv[fidata]["pulseDuration"],
                        "-",
                        color=color,
                        linewidth=0.75,
                        label=leglabel,
                    )
                ncells += 1

        PF.figure_handle.suptitle(
            "{0:s}   {1:s}".format(self.experiment, self.celltype)
        )
        PF.axdict["A"].set_xlabel("I (nA)")
        PF.axdict["A"].set_ylabel("Firing Rate (Hz)")
        plotcodes = list(set(legcodes).intersection(set(self.codes)))
        cprint("r", f"plotcodes: {str(plotcodes):s}")
        h, l = PF.axdict["A"].get_legend_handles_labels()
        PF.axdict["A"].legend(h, l, fontsize=6, loc='upper left')
        return

    def plot_spike_info(self, mode, coding="code", parentFigure=None):
        self.Pspikes = PH.regular_grid(
            4,
            4,
            order="columnsfirst",
            showgrid=False,
            verticalspacing=0.08,
            horizontalspacing=0.08,
            margins={
                "leftmargin": 0.08,
                "rightmargin": 0.08,
                "topmargin": 0.5,
                "bottommargin": 0.05,
            },
            labelposition=(-0.05, 1.05),
            parent_figure=parentFigure,
            panel_labels=None,
        )
        pax = self.Pspikes.axarr.ravel()
        self.Pspikes.figure_handle.suptitle(
            "{0:s}   {1:s}".format(self.experiment, self.celltype)
        )
        accumulator, spike_hues = self.get_spike_info(mode)

        dcomp = pd.DataFrame(accumulator)
        
        gps = {}
        for param in [
            "Ibreak",
            "Irate",
            "AdaptRatio",
            "AP1_HalfWidth_interpolated",
            "AP2_HalfWidth",
            "peak_V",
            "AHP_Depth",
            "peaktotrough",
            "spikethreshold",
            "AP1_Latency",
            "AP2_Latency",
            "FiringRate_1p5T",
            "maxrate",
            "FR_Slope",
        ]:
            groups = {}
            dcomp[param] = pd.DataFrame(accumulator[param])
            dcomp["coding"] = spike_hues[param]
            dcomp["age"] = pd.DataFrame(accumulator["age"])
            with pd.option_context("mode.use_inf_as_null", True):
                dcomp = dcomp.dropna(subset=[param, "coding"], how="all")
            #    print('dcomp: ', param, '\n', dcomp)
            codes = list(set(dcomp["coding"]))
            for code in codes:
                if code not in list(groups.keys()):
                    new_code = code
                    groups[new_code] = (
                    dcomp[(dcomp["coding"] == new_code)][param]
                    .replace(np.inf, np.nan)
                    .dropna(how="all")
                )
            gps[param] = groups
 
            t, p = scipy.stats.ttest_ind(
                gps[param]['A'], gps[param]['B'], equal_var=False, nan_policy="omit"
            )
            print(f"Param: {param:<20s}:", end="")
            print(f"        t={t:.3f}  p={p:.4f}")
            print(
                f"        A: mean={np.mean(gps[param]['A']):8.3f} std={np.std(gps[param]['A']):8.3f}, N={len(gps[param]['A']):d}"
            )
            print(
                f"        B: mean={np.mean(gps[param]['B']):8.3f} std={np.std(gps[param]['B']):8.3f}, N={len(gps[param]['B']):d}"
            )

            # KW
            if 'AA' in gps[param].keys() and 'AAA' in gps[param].keys():
                if len(gps[param]['AA']) > 0 and len(gps[param]['AAA'])>0:
                    s, p = scipy.stats.kruskal(gps[param]['B'], gps[param]['A'], gps[param]['AA'], gps[param]['AAA'])
                    print(f"Krwukal-Wallis: H:{s:.6f}   p={p:.6f}\n")
                elif len(gps[param]['AA']) > 0 and len(gps[param]['AAA']) == 0:
                    s, p = scipy.stats.kruskal(gps[param]['B'], gps[param]['A'], gps[param]['AA'], )
                    print(f"Krwukal-Wallis: H:{s:.6f}   p={p:.6f}\n")

        y = pd.DataFrame(accumulator)
        y = y.assign(hue=pd.Series(spike_hues))

        for i, m in enumerate(measures):
            if m is None:
                continue
            yd = y[m].replace([np.inf], np.nan)

            x = pd.Series(spike_hues[m])
            iasort = x.argsort()
            print("Plotting measure: ", m)
            ys = yd  # yd[iasort]

            xs = x  # x[iasort]
            if np.all(np.isnan(ys)):
                continue  # skip plot if no data

            dfm = pd.DataFrame({"Group": x, "measure": ys, "group": x })
            sns.violinplot(data=dfm, x='Group', y='measure', ax=pax[i], order=['A', 'B'], palette=palette, inner=None, saturation=1.0)
            mpl.setp(pax[i].collections, alpha=0.3)
            sns.swarmplot(data=dfm, x='Group', y='measure', ax=pax[i], order=['A', 'B'], palette=palette, edgecolor='grey', size=2.5)
            # sns.stripplot(
            #     data=dfm,
            #     x="x",
            #     y="y",
            #     ax=pax[i],
            #     hue="group",
            #     palette="Set2",
            #     edgecolor="grey",
            #     size=4,
            # )
            # sns.boxplot(data = dfm, x='Group', y="measure",  ax=pax[i], palette=palette, orient='v', width=0.5, saturation=1.0)
            pax[i].set_title(m, fontsize=7)  # .replace('_', '\_'))

            pax[i].set_ylim(paxes[m])
            if m == "AdaptRatio":
                if pax[i].get_legend() is not None:
                    mpl.setp(pax[i].get_legend().get_texts(), fontsize="7")
            else:
                if pax[i].get_legend() is not None:
                    pax[i].get_legend().remove()
        for i in range(len(measures)):
            if i > 0:
                continue
            for lh in pax[i].legend().legendHandles:
                lh.set_alpha(1)
                lh._sizes = [4]
                mpl.setp(pax[i].get_legend().get_texts(), fontsize="7")
        return self.Pspikes

    def get_spike_info(self, mode):
        # do spike plots
        assert mode in ["OU", "EU"]

        EU_accumulator = self.make_emptydict(measures)
        EU_spike_hues = self.make_emptydict(measures)
        dates = self.dfs.date.unique()

        for (
            day
        ) in dates:  # # for each mouse (day) :: Biological Unit AND Experimental Unit
            OU_accumulator = self.make_emptydict(measures)  #
            for idx in self.dfs.index[
                self.dfs["date"] == day
            ]:  # for each cell in that day: Observational UNIT
                cellmeas = self.make_emptydict(measures)
                ages = self.make_emptydict(measures)
                d = self.dfs.at[idx, "Spikes"]  # get the Spikes dict
                day = self.dfs.at[idx, "date"][:-4]
                if pd.isnull(d):
                    continue

                for u in d.keys():  # for every protocol in the Spikes dict
                    if not isinstance(u, Path):  # ooops, skip that one
                        continue
                    if not self.select_protocols(u):
                        continue
                    if "FI_Curve" not in list(
                        d[u].keys()
                    ):  # protocol needs an FI_Curve analysis
                        continue
                    for m in measures:
                        if m is None:
                            continue
                        ages[m].append(self.get_age(idx))
                        if m == "spikethreshold":
                            fi = d[u]["FI_Curve"]
                            firstsp = np.where((fi[1] > 0) & (fi[0] > 0))[0]
                            if len(firstsp) > 0:
                                firstsp = firstsp[0]
                                cellmeas[m].append(
                                    fi[0][firstsp] * 1e12
                                )  # convert to pA
                            else:
                                cellmeas[m].append(np.nan)
                        elif m == "maxrate":
                            fi = d[u]["FI_Curve"]
                            firstsp = np.where((fi[1] > 0) & (fi[0] > 0))[
                                0
                            ]  # spikes and positive current together
                            if len(firstsp) > 0 and np.max(
                                fi[1] >= 2e-9
                            ):  # spikes and minimum current injection
                                cellmeas[m].append(
                                    np.max(fi[1][firstsp]) / d[u]["pulseDuration"]
                                )  # convert to spikes/second
                            else:
                                print(
                                    "no max: day: {0:s}  max spks: {1:d}  # traces with spikes {2:d}".format(
                                        self.make_cell(idx),
                                        int(np.max(fi)),
                                        len(firstsp),
                                    )
                                )
                                # print(fi)
                                cellmeas[m].append(np.nan)
                        elif m == "Ibreak":
                            fig = d[u]["FI_Growth"]
                            if len(fig) > 0:
                                par = fig[0]["parameters"]
                                if len(par) > 0 and len(par[0]) > 0:
                                    cellmeas[m].append(par[0][1])
                        elif m == "Irate":
                            fig = d[u]["FI_Growth"]
                            if len(fig) > 0:
                                par = fig[0]["parameters"]
                                if len(par) > 0 and len(par[0]) > 0:
                                    cellmeas[m].append(par[0][4])
                        elif m == "FR_Slope":  # get slope from near-threshold firing
                            rate_spks = []
                            rate_i = []
                            fidata = d[u]["FI_Curve"]
                            for fsp in range(len(fidata[0])):
                                if fsp not in d[u]["spikes"].keys():
                                    continue
                                nspkx = len(d[u]["spikes"][fsp])
                                if nspkx > 0 and fidata[0][fsp] > 0.0:
                                    if len(rate_spks) < 3:
                                        rate_spks.append(nspkx / d[u]["pulseDuration"])
                                        rate_i.append(fidata[0][fsp] * 1e9)
                            if len(rate_i) > 0:
                                #     cellmeas[m].append(np.nan)
                                # else:
                                p = np.polyfit(rate_i, rate_spks, 1)
                                cellmeas[m].append(p[0])  # store slope from fit
                        else:

                            # print('measure: ', m, idx) print(d[u].keys())
                            # print(d[u]['spikes'].keys()) #exit() if
                            # 'AP1_HalfWidth_interpolated' in list(d[u].keys()):
                            # print('found halfwidth at top level!!!!!!', day,
                            # u)
                            if m in list(
                                d[u].keys()
                            ):  #  and m not in measures_fromspikes:
                                cellmeas[m].append(d[u][m])
                            #                               print('got %s from direct' % m)
                            # elif m in list(d[u]['spikes'].keys()): print('got
                            #     %s from d[u][spikes]' % m)
                            #     cellmeas[m].append(d[u]['spikes'][m])
                            else:  # loop through traces
                                xm = []
                                spkdata = d[u]["spikes"]
                                for tr in spkdata.keys():  # each trace with spikes
                                    for spk in spkdata[tr]:
                                        # print('dfs2: ',
                                        # spkdata[tr][spk].keys())
                                        # print(len(d[u]['spikes'][tr]))
                                        # for sp in range(len(d[u]['spikes'][tr])):
                                        #    print(d[u]['spikes'][tr][0].keys())
                                        if m in d[u]["spikes"][tr][spk].keys():
                                            # print('found m in spikes: ',m,
                                            # d[u]['spikes'][tr][spk][m])
                                            if d[u]["spikes"][tr][spk][m] is not None:
                                                xm.append(d[u]["spikes"][tr][spk][m])
                                    #    else:
                                #                                            print('did not find measure in
                                #                                            spikes: ', m,
                                #                                            d[u]['spikes'][tr][spk].keys())
                                # print(idx)
                                if len(xm) > 0:
                                    cellmeas[m].append(np.nanmean(xm))
                            # print(' built from traces, m') print(' got %s
                            # from spike array' % m, cellmeas[m], xm)
                # compute a single value for each measure for each animal for
                # the cells from that animal. print(cellmeas)

                for m in measures:
                    if m == "spikethreshold":
                        if len(cellmeas[m]) == 0:
                            OU_accumulator[m].append(np.nan)
                        else:
                            OU_accumulator[m].append(
                                np.nanmin(cellmeas[m])
                            )  # get lowest threshold for all recordings in the cell
                    elif m == "maxrate":
                        if len(cellmeas[m]) == 0:
                            OU_accumulator[m].append(np.nan)
                        else:
                            # print('m: ', m, cellmeas[m])
                            OU_accumulator[m].append(
                                np.nanmax(cellmeas[m])
                            )  # get maximal measured reat within the cell
                    else:
                        OU_accumulator[m].append(
                            np.nanmean(cellmeas[m])
                        )  # accumulate average from the mesurements made within the cell

            for (
                m
            ) in measures:  # now accumulate the experimental units (mean for each DAY)
                if mode == "EU":
                    EU_accumulator[m].append(np.nanmean(OU_accumulator[m]))
                    code = self.dfs.at[idx, "coding"]
                    EU_spike_hues[m].append(code)
                else:
                    EU_accumulator[m].extend(OU_accumulator[m])
                    code = self.dfs.at[idx, "coding"]
                    EU_spike_hues[m].extend([code] * len(OU_accumulator[m]))

                # if self.coding is not None and  day in self.coding:
                #     EU_spike_hues[m].append(self.coding[day][1]) else:
                # EU_spike_hues[m].append('Z')
        ot = f"Obs,Group"
        for m in measures:
            ot += f",{m:s}"
        ot += "\n"
        for i in range(len(EU_accumulator[measures[0]])):
            ot += f"{i:d},{EU_spike_hues[m][i]:s}"
            for m in measures:
                ot += f",{EU_accumulator[m][i]:f}"
            ot += "\n"
        ofile = Path(f"R_{self.experiment}_Spikes_{self.celltype:s}_{self.mode}.csv")
        ofile.write_text(ot)

        return EU_accumulator, EU_spike_hues

    # def plot_age(self):
    #
    #     self.P_age = PH.regular_grid(3, 4, order='columns', figsize=(8, 5.), showgrid=False,
    #             verticalspacing=0.08, horizontalspacing=0.08,
    #             margins={'leftmargin': 0.12, 'rightmargin': 0.05, 'topmargin': 0.08, 'bottommargin': 0.05},
    #             labelposition=(-0.05, 1.05), parent_figure=None, panel_labels=None)
    #     pax_age = self.P_age.axarr.ravel()
    #     self.P_age.figure_handle.suptitle('{0:s}   {1:s}'.format(args.experiment, args.celltype))
    #
    #     accumulator = self.make_emptydict(measures)
    #     spike_hues = self.make_emptydict(measures)
    #     for idx in self.dfs.index:  # for every entry in the table
    #
    #         cellmeas = self.make_emptydict(measures)
    #         ages =  self.make_emptydict(measures)
    #         d = self.dfs.at[idx, 'Spikes']  # get the Spikes dict
    #         day = self.dfs.at[idx, 'date'][:-4]
    #
    #         for u in d.keys():  # for every protocol in the Spikes dict
    #             if not isinstance(u, Path):  # ooops, skip that one
    #                 continue
    #             for m in measures:
    #                 if 'FI_Curve' not in list(d[u].keys()):  # protocol needs an FI_Curve analysis
    #                     continue
    #                 ages[m].append(self.get_age(idx))
    #                 pax_age[i].plot(ages[m], yd, 'o')
    #                 pax_age[i].set_xlim(0, 150.)
    #                 if m in list(paxes.keys()):
    #                     pax_age[i].set_ylim(paxes[m])
    #                 elif m in list(paxesb.keys()):
    #                     pax_age[i].set_ylim(paxesb[m])
    #
    #                 pax_age[i].set_title(m, fontsize=7)
    #
    #                 if pax_age[i].get_legend() is not None:
    #                     mpl.setp(pax_age[i].get_legend().get_texts(), fontsize='6')
    #         # print('spike_hues: ', spike_hues)
    #
    #
    #

    def plot_IV_info(self, mode, parentFigure=None):
        """
        Mode can be OU for observational unit (CELL) or EU for experimental Unit
        (Animal)
        """
        #         ### IV
        EU_accumulator = self.make_emptydict(iv_measures)
        EU_iv_hues = self.make_emptydict(iv_measures)

        modestring = "Unit = Experiment (animal)"
        if mode == "OU":
            modestring = "Unit = Observational (cell)"
        cprint('c', f"Accumulating by mode: {modestring:s}")
        dates = self.dfs.date.unique()

        # for each mouse (day) :: Biological Unit AND Experimental Unit
        for d in dates:
            OU_accumulator = self.make_emptydict(iv_measures)  #
            for idx in self.dfs.index[
                self.dfs["date"] == d
            ]:  # for each cell in that day: Observational UNIT
                d = self.dfs.at[idx, "IV"]  # get the IV dict
                day = self.dfs.at[idx, "date"][:-4]
                cellmeas = self.make_emptydict(iv_measures)
                if pd.isnull(d):
                    continue
                for u in d.keys():
                    if not isinstance(u, Path):
                        continue
                    for m in iv_measures:
                        if m not in list(d[u].keys()):
                            continue
                        if d[u][m] is not None:
                            cellmeas[m].append(d[u][m])
                        else:
                            continue  # cellmeas[m].append(np.nan)
                        # cellmeas[m] = [x for x in cellmeas[m] if x != None] if
                        # day in self.coding:
                        #     iv_hue[m].append(self.coding[day][1]) else:
                        # iv_hue[m].append('Z')

                for m in OU_accumulator.keys():
                    ms = cellmeas[m]
                    if len(cellmeas[m]) == 0:
                        ms = np.nan
                    else:
                        ms = np.nanmean(cellmeas[m])
                    OU_accumulator[m].append(ms)  # accumulate average within the cell
            # now accumulate the experimental units (mean for each DAY)
            for m in iv_measures:
                if mode == "EU":
                    EU_accumulator[m].append(np.nanmean(OU_accumulator[m]))
                    EU_iv_hues[m].append(self.dfs.at[idx, "coding"])
                else:
                    EU_accumulator[m].extend(OU_accumulator[m])
                    # print(self.dfs.at[idx, 'coding'])
                    EU_iv_hues[m].extend(
                        [self.dfs.at[idx, "coding"]] * len(OU_accumulator[m])
                    )
        # print("EU Accumulator: \n", EU_accumulator)
        ot = f"Obs,Group"
        for m in iv_measures:
            ot += f",{m:s}"
        ot += "\n"
        for i in range(len(EU_accumulator[iv_measures[0]])):
            ot += f"{i:d},{EU_iv_hues[m][i]:s}"
            for m in iv_measures:
                ot += f",{EU_accumulator[m][i]:f}"
            ot += "\n"
        ofile = Path(f"R_{self.experiment}_{self.celltype:s}_{self.mode}.csv")
        ofile.write_text(ot)

        # print(EU_iv_hues) exit()
        dcomp = pd.DataFrame(EU_accumulator)
        gps = {}
        for param in ["Rin", "RMP", "taum", "tauh_tau"]:
            groups = {}
            # if param == 'taum':
            #     print(EU_accumulator)
            #     exit()
            dcomp[param] = pd.DataFrame(EU_accumulator[param])
            dcomp["coding"] = EU_iv_hues[param]
            # dcomp['age'] = pd.DataFrame(EU_accumulator['age'])

            with pd.option_context("mode.use_inf_as_null", True):
                dcomp = dcomp.dropna(subset=[param, "coding"], how="all")
            #            print('dcomp: ', param, '\n', dcomp)
            codes = list(set(dcomp["coding"]))
            for code in codes:
                if code not in list(groups.keys()):
                    new_code = code
                    groups[new_code] = (
                    dcomp[(dcomp["coding"] == new_code)][param]
                    .replace(np.inf, np.nan)
                    .dropna(how="all")
                )
                if code == 'B':
                    print(dcomp[(dcomp["coding"] == new_code)][param])
            gps[param] = groups

            t, p = scipy.stats.ttest_ind(
                gps[param]['A'], gps[param]['B'], equal_var=False, nan_policy="omit"
            )
            print(f"Param: {param:>20s}:", end="")
            print(f"        t={t:.3f}  p={p:.4f}")
            print(
                f"        A: mean={np.mean(gps[param]['A']):8.3f} std={np.std(gps[param]['A']):8.3f}, N={len(gps[param]['A']):d}"
            )
            print(
                f"        B: mean={np.mean(gps[param]['B']):8.3f} std={np.std(gps[param]['B']):8.3f}, N={len(gps[param]['B']):d}"
            )

            if len(gps[param]['A']) > 0 and len(gps[param]['B'] > 0) and 'AA' in gps[param].keys() and 'AAA' in gps[param].keys():
                s, p = scipy.stats.kruskal(gps[param]['A'], gps[param]['B'], gps[param]['AA'], gps[param]['AAA'])
                print(f"Krwukal-Wallis: H:{s:.6f}   p={p:.6f}\n")
            elif len(gps[param]['A']) > 0 and len(gps[param]['B'] > 0) and 'AA' in gps[param].keys() and 'AAA' not in gps[param].keys():
                s, p = scipy.stats.kruskal(gps[param]['A'], gps[param]['B'], gps[param]['AA'],)
                print(f"Krwukal-Wallis: H:{s:.6f}   p={p:.6f}\n")

        y = pd.DataFrame(EU_accumulator)
        y = y.assign(coding=pd.Series(EU_iv_hues['Rin']))
        print(y.head())
        self.Pb = PH.regular_grid(
            2,
            3,
            order="columnsfirst",
            figsize=(4.7, 7.0),
            showgrid=False,
            verticalspacing=0.08,
            horizontalspacing=0.08,
            margins={
                "leftmargin": 0.55,
                "rightmargin": 0.05,
                "topmargin": 0.08,
                "bottommargin": 0.59,
            },
            labelposition=(-0.05, 1.05),
            parent_figure=parentFigure,
            panel_labels=None,
        )
        paxb = self.Pb.axarr.ravel()
        self.Pb.figure_handle.suptitle(
            f"{self.experiment:s}   {self.celltype:s} BY: {modestring:s}"
        )

        for i, m in enumerate(iv_measures):
            yd = y[m].replace([np.inf], np.nan)
            sns.violinplot(data=y, x='coding', y=m, ax=paxb[i], order=['A', 'B'], palette=palette, inner=None)
            mpl.setp(paxb[i].collections, alpha=0.3)
            sns.swarmplot(data=y, x='coding', y=m, ax=paxb[i], order=['A', 'B'], palette=palette, size=2.5)
            # sns.stripplot(
            #     x=x, y=yd, ax=paxb[i], hue=EU_iv_hues[m], palette="Set2", size=4
            # )
            paxb[i].set_title(m, fontsize=7)  # .replace('_', '\_'))
            paxb[i].set_ylim(paxesb[m])
            if m == "Rin":
                if paxb[i].get_legend() is not None:
                    mpl.setp(paxb[i].get_legend().get_texts(), fontsize="7")
                    # ap1l = d[u]['AP1_Latency'] ap1hw = d[u]['AP1_HalfWidth']
                    # apthr = d[u]['FiringRate_1p5T']
            else:
                pass # paxb[i].get_legend().remove()
        for i in range(len(iv_measures)):
            if i > 0:
                continue
            for lh in paxb[i].legend().legendHandles:
                lh.set_alpha(1)
                lh._sizes = [4]
                mpl.setp(paxb[i].get_legend().get_texts(), fontsize="7")
                lh._fontsize = 6
        return self.Pb

    def get_age(self, idx):
        # print(self.dfs.columns)
        age = self.dfs.at[idx, "age"]
        if isinstance(age, str):
            if age[0] in ["p", "P"]:
                age = int(age[1:])
            elif age[-1] in ["d", "D"]:
                age = int(age[:-1])
            elif age != " ":
                age = int(age)
            else:
                age = 0
        return age

    def make_emptydict(self, measures):
        d = dict.fromkeys(measures)
        for k, _ in d.items():
            d[k] = []
        return d

    def make_cell(self, iday):
        datestr = self.dfs.at[iday, "date"]
        slicestr = self.dfs.at[iday, "slice_slice"]
        cellstr = self.dfs.at[iday, "cell_cell"]
        return str(Path(datestr, slicestr, cellstr))


def get_rec_date(filename: Union[Path, str]):
    """get the recording date of record from the filename as listed n the excel sheet

    Args:
        filename (Union[Path, str]): _description_
    """
    fn = Path(filename)
    datename = str(fn.name)
    datename = datename[:-4]
    return datename


def get_cell_name(row):
    dn = str(Path(row.date).name)[:-4]
    sn = f"S{int(row.slice_slice[-3:]):02d}"
    cn = f"C{int(row.cell_cell[-3:]):02d}"
    cellname = f"{dn:s}_{sn:s}_{cn:s}"
    return cellname


def _make_short_name(row):
    return get_rec_date(row["date"])


# def find_protocols(df):
#     # df = pd.read_excel(excelsheet)
#     # code_dataframe = pd.read_excel(codesheet)

#     # generate short names list
#     df['shortdate'] = df.apply(_make_short_name, axis=1)

#     # df_new['date'] = sorted(list(df['shortdate', right_on='Date', how='left')
#     df_new = pd.merge(df, code_dataframe, left_on='shortdate', right_on='Date', how='left')
#     df_new['holding'] = np.nan
#     df_new['dvdt_rising'] =np.nan
#     df_new['dvdt_falling'] =np.nan
#     df_new['AP_thr_V'] = np.nan
#     df_new['AP_HW'] = np.nan
#     df_new['AP_begin_V'] = np.nan
#     df_new['AHP_trough_V'] = np.nan
#     df_new['AHP_depth_V'] = np.nan
#     df_new['FiringRate'] = np.nan
#     df_new['AdaptRatio'] = np.nan
#     df_new['spiketimes'] = np.nan
#     nprots = 0


#     with pd.ExcelWriter(analyzed_data_file) as writer:
#         df_new.to_excel(writer, sheet_name = "Sheet1")
#         for i, column in enumerate(df_new.columns):
#             column_width = max(df_new[column].astype(str).map(len).max(),24) # len(column))
#             writer.sheets['Sheet1'].set_column(first_col=i+1, last_col=i+1, width=column_width) # column_dimensions[str(column.title())].width = column_width
#         # writer.save()
#     print("\nFind_protocols wrote to : ", analyzed_data_file)
#     return df_new


def checkforprotocol(df, index, protocol: str):
    # print(row.data_complete)
    prots = df.data_complete[index].split(",")
    prots2 = df.data_incomplete[index].split(",")
    prots.extend([p.split(".")[0] for p in prots2])
    protlist = []
    for p in prots:
        sp = p.strip()
        if sp.startswith(protocol):
            protlist.append(sp)
    return protlist


def get_protocols_from_datasheet(
    filename,
    filterstring: str,
    outcols: list = [],
    outputfile: str = None,
    code_file: str = None,
    code_sheet: str = None,
):
    """read the protocols from the datasheet, and build a new
    dataframe expanded by individual protocols

    Args:
        filename (_type_): name of data sheet/pickled pandas file to read.
        filterstring (_type_): string to filter protocols on ("CC_Spont", for example)

    Returns:
        pandas dataframe: the pandas data frame with the code dataframe merged in,
        for all of the individual cells/protocols that match.
        adds a column that uniquely identifies cells by date, slice, cell
    """
    df = pd.read_pickle(filename)

    # force some column types
    for coln in ["age", "date"]:
        df[coln] = df[coln].astype(str)
    # generate short names list
    df["shortdate"] = df.apply(_make_short_name, axis=1)
    df["cell_name"] = df.apply(get_cell_name, axis=1)
    # read the code dataframe (excel sheet)
    code_df = pd.read_excel(code_file, sheet_name=code_sheet)
    # be sure types are correct
    for coln in ["age", "ID", "Group"]:
        code_df[coln] = code_df[coln].astype(str)
    code_df = code_df.drop("age", axis="columns")
    dfm = pd.merge(
        df, code_df, left_on="shortdate", right_on="Date", how="left"
    )  # codes are by date only
    dfm["iv_name"] = ""
    # print("dfm columns: ", dfm.columns)
    # make an empty dataframe with defined columns
    df_new = pd.DataFrame(columns=outcols)
    # go through all of the entries in the main df
    for index in dfm.index:
        prots = checkforprotocol(
            df, index, protocol=filterstring
        )  # get all protocols in that entry
        for prot in prots:
            data = {}
            for col in outcols:  # copy over the information from the input dataframe

                if col != "iv_name":
                    data[col] = dfm[col][index]
                else:
                    data["iv_name"] = prot  # add the protocol
            if data["cell_type"] not in ["Pyramidal", "pyramidal"]:
                continue
            if data["age"].strip() in ["", " ", "?"]:
                continue
            a = ""
            age = [a + b for b in str(data["age"]) if b.isnumeric()]
            age = "".join(age)
            try:
                d_age = int(age)
            except:
                print("age: ", age, data["age"])
                exit()
            if d_age < 30 or d_age < 65:
                continue
            df_new.loc[len(df_new.index)] = data  # and store in a new position

    if outputfile is not None:
        df_new.to_excel(outputfile)
    return df_new


def main():
    parser = argparse.ArgumentParser(description="iv analysis")
    parser.add_argument(
        "-E",
        "--experiment",
        type=str,
        dest="experiment",
        default="None",
        help="Select Experiment to analyze",
    )
    parser.add_argument(
        "-m",
        "--mode",
        type=str,
        dest="mode",
        choices=["OU", "EU"],
        help="Specify summary mode for plot : observation (cell) or experiment (animal)",
    )
    parser.add_argument(
        "-D", "--basedir", type=str, dest="basedir", help="Base Directory"
    )
    parser.add_argument(
        "-c",
        "--celltype",
        type=str,
        default="pyramidal",
        dest="celltype",
        choices=[
            "pyramidal",
            "cartwheel",
            "tuberculoventral",
            "giant",
            "bushy",
            "t-stellate",
            "d-stellate",
            "any",
        ],
        help="Specify cell type",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="",
        dest="outputFilename",
        help="Specify output PDF filename (full path)",
    )
    parser.add_argument("-d", "--day", type=str, default="all", help="day for analysis")
    parser.add_argument(
        "-s",
        "--slice",
        type=str,
        default="",
        dest="slicecell",
        help="select slice/cell for analysis: in format: S0C1 for slice_000 cell_001\n"
        + "or S0 for all cells in slice 0",
    )
    args = parser.parse_args()

    experiments = {
        "Ank2B": {
            "datadisk": "/Volumes/Pegasus_002/Kasten_Michael/Maness_Ank2_PFC_stim",
            "directory": "ANK2",
            "resultdisk": "ANK2",
            "db_directory": "ANK2",
            "datasummary": "Intrinsics",
            "IVs": "ANK2_NEX_IVs",
            "coding_file": "Intrinsics.xlsx",
            "coding_sheet": "codes",
        }
    }

    g = GetAllIVs(vars(args))
    g.set_protocols(["CCIV_long_HK", "CCIV_1nA_max_1s", "CCIV_200pA"])
    g.set_experiments(experiments)
    g.run()


if __name__ == "__main__":
    main()
