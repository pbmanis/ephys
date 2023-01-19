"""IV_Summarize:
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
averaged by the "code" with each EU. 

 
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
from ephys.tools.parse_ages import ISO8601_age

rcParams = matplotlib.rcParams
rcParams["svg.fonttype"] = "none"  # No text as paths. Assume font installed.
rcParams["pdf.fonttype"] = 42
rcParams["ps.fonttype"] = 42
rcParams["text.usetex"] = False

cprint = CP.cprint


def get_computer():
    if os.name == "nt":
        computer_name = os.environ["COMPUTERNAME"]
    else:
        computer_name = subprocess.getoutput("scutil --get ComputerName")
    return computer_name


computer_name = get_computer()

# iv measures to track:
iv_measures = [
    "Date",
    "Cell_ID",
    "coding",
    "Animal_ID",
    "Group",
    "protocol",
    "Rin",
    "RMP",
    "taum",
    "tauh_bovera",
    "tauh_tau",
]

# spike measures to track (including indexing (Date, Coding: Cell_ID, Animal_ID, etc.)
spike_measures = [
    "Date",
    "Cell_ID",
    "coding",
    "Animal_ID",
    "Group",
    "age",
    "protocol",
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
    "FR_Slope",
]

# parameters in the tracked items that should not be averaged numerically
no_average = [
    "Date",
    "coding",
    "Cell_ID",
    "Animal_ID",
    "Group",
    "protocol",
    "age",
]

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

# graph limits for spike measures
paxes = dict.fromkeys(spike_measures)
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
paxes["spikethreshold"] = [0, None]
paxes["maxrate"] = [0, 150]
paxes["age"] = [0, 1000]
paxes["FR_Slope"] = [0, 1000]


# graph limits for iv_measures
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
            Nothing
        """
        self.coding = {}
        self.mode = args["mode"]
        if "celltype" in args.keys():
            self.celltype = args["celltype"]
        else:
            celltype = "*"
        self.experiment = args["experiment"]
        if "protocols" in args.keys():
            self.protocols = args["protocols"]
        else:
            self.protocols = ["*"]
        if "basedir" in args.keys():
            self.basedir = args["basedir"]
        else:
            self.basedir = None
        if "outputFilename" in args.keys():
            self.outputFilename = args["outputFilename"]
        else:
            self.outputFilename = None
        if "slicecell" in args.keys():
            self.slicecell = args["slicecell"]
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
        self.iv_dataframe = None
        self.spike_dataframe = None

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
                "IVs": "ANK2_NEX_IVs",  # name of the excel sheet generated by (IV_Analysis) with the data
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
            None
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
                if select not in ["any", "*"]:
                    df_i = df_i[df_i["cell_type"] == select]
                sLength = len(df_i["date"])
                df_i['age'] = ISO8601_age(df_i['age'].values[0])

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
        self.rainbowcolors = iter(
            matplotlib.cm.rainbow(np.linspace(0, 1, len(self.dfs)))
        )
        self.dfs.to_excel("iv_summarize_intermediate_datafile.xlsx")
        cprint(
            "g",
            "All data loaded, intermediate date file iv_summarize_intermediate_datafile.xlsx written",
        )

        return

    def run(self):
        """Run with current parameters and build a summary plot of the measurements."""
        if self.dfs is None:
            self.build_dataframe()

        # create primary figure space
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

        # now do the FI curves and the two subfigures
        # along with their analyses
        self.plot_FI(self.PFig)
        P1 = self.plot_IV_info(self.mode, parentFigure=self.PFig)
        P2 = self.plot_spike_info(self.mode, parentFigure=self.PFig)
        mpl.show()



    def set_protocols(self, protocols: list):
        """Set the list of protocols that we will use for the analysis.
        This should be called at the "user" level according to the names
        of the relevant protocols in the dataset

        Args:
            protocols (list): protocol names. Use ['*'] to use all protocols in
            spikes and iv_curve
        """
        self.protocols = protocols

    def select_protocols(self, prot: Union[Path, str]):
        """run a selector to determine whether a protocol matches
        the ones in self.protocols.

        Args:
            prot (Union[Path, str]): protocol to selct form list of prototcols

        Returns:
            _type_: _description_
        """
        if self.protocols[0] == "*":
            return True
        for p in self.protocols:
            if Path(prot).name.startswith(p):
                return True
        return False

    def plot_FI(self, PF):
        """plot all FI Curves in the dataset, superimposed
        This function does no analysis, but requires that
        analysis be done prior to calling it.

        Args:
            PF (pandas dataframe): the dataframe for this dataset
        """

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
                        cprint("r", f"fidata is not path:  {str(fidata):s}")
                        continue  # fidata = Path(fidata)
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
        PF.axdict["A"].legend(h, l, fontsize=6, loc="upper left")
        return

    def _get_spike_parameter(
        self,
        measure: str,
        measurename: str,
        cellmeas: dict,
        accumulator: dict,
        func: object = np.nanmean,
    ):
        """Get spike values averaged (or smallest) from one protocol/measurement in one cell
        accumulate the measures in the accumulator (dict of measures), using lists.

        Args:
            measure: the name of the measure to be stored
            measurename: the name of the measure as found in the spike array
            accumulator (dict): _description_
            cellmeas (_type_): _description_
            func (function object): name of function to use on values (np.nanmean, np.nanmin)
        Returns:
            updated accumulator
        """
        tracekeys = list(cellmeas["spikes"].keys())
        if len(tracekeys) == 0:
            accumulator[measure].append(np.nan)
        else:
            values = []
            for itr in tracekeys:
                for ispk in cellmeas["spikes"][itr].keys():
                    values.append(cellmeas["spikes"][itr][ispk][measurename])
            if len(values) == 0:
                accumulator[measure].append(np.nan)
            else:
                accumulator[measure].append(
                    func(values)
                )  # get lowest current threshold for this protocol
        return accumulator

    def _get_protocol_spike_data(
        self,
        dspk: dict,
        dfi: dict,
        proto: str,
        Cell_ID: str,
        Animal_ID: str,
        coding: str,
        Group: str,
        day,
        accumulator: dict,
    ):
        """Get the spike data for the specified cell
        and protocol in the dataframe
        computes the mean or min of the parameter as needed if there are multiple measures
        within the protocol.
        The accumulator dict has keys indicating the Cell_ID, etc.
        Args:
            dspk (dict): spike dictionary for one cell
            dfi (dict): FI dictionary (IV) for one cell
            proto (str): Protocol name within the dict to use
        """
        cellmeas = dspk[proto]
        tracekeys = list(cellmeas["spikes"].keys())
        print("get_protocol_spike: ", Cell_ID)
        for m in spike_measures:
            if (
                m == "spikethreshold"
            ):  # find lowest current at which spike was evoked in this series
                accumulator = self._get_spike_parameter(
                    measure=m,
                    measurename="current",
                    cellmeas=cellmeas,
                    accumulator=accumulator,
                    func=np.nanmin,
                )
            elif m == "peak_V":  # find mean peak spike voltage
                accumulator = self._get_spike_parameter(
                    measure=m,
                    measurename=m,
                    cellmeas=cellmeas,
                    accumulator=accumulator,
                    func=np.nanmean,
                )
            elif m == "peaktotrough":  # find mean peak-to-trough
                accumulator = self._get_spike_parameter(
                    measure=m,
                    measurename=m,
                    cellmeas=cellmeas,
                    accumulator=accumulator,
                    func=np.nanmean,
                )
            elif m == "maxrate":
                if cellmeas["FiringRate"] == 0.0:
                    accumulator[m].append(np.nan)
                else:
                    accumulator[m].append(
                        cellmeas["FiringRate"]
                    )  # get maximal measured firing rate from the protocol
            elif m in ["Ibreak", "Irate", "FR_Slope"]:
                if len(cellmeas["FI_Growth"]) == 0:
                    accumulator[m].append(np.nan)
                else:
                    accumulator[m].append(cellmeas["FI_Growth"][m])
            elif m in no_average:
                if m == "Cell_ID":
                    accumulator[m].append(Cell_ID)
                if m == "Date":
                    accumulator[m].append(day)
                if m == "Animal_ID":
                    accumulator[m].append(Animal_ID)
                if m == "Group":
                    accumulator[m].append(Group)
                if m == "coding":
                    accumulator[m].append(coding)
                if m == "protocol":
                    accumulator[m].append(proto)
                if m == "Ibreak":
                    accumulator[m].append(cellmeas[m])
            elif m not in no_average:
                accumulator[m].append(np.nanmean(cellmeas[m]))
            else:
                cprint("r", f"Failed to find data for measure: {m:s}")
                cprint("r", f"     {(tracekeys):s}")
                accumulator[m].append(np.nan)
        return accumulator

    def _get_cell_means(self, accumulator, measures):
        for m in measures:
            if m in no_average:
                if m == "Cell_ID":
                    print(accumulator[m])
                    accumulator[m] = accumulator[m][0]
                if m == "Date":
                    accumulator[m] = accumulator[m][0]
                if m == "Animal_ID":
                    accumulator[m] = accumulator[m][0]
                if m == "Group":
                    accumulator[m] = accumulator[m][0]
                if m == "coding":
                    accumulator[m] = accumulator[m][0]
                if m == "protocol":
                    accumulator[m] = [Path(k).name for k in accumulator[m]]
                    # cprint('y', f"\n {m:s}, {str(accumulator[m]):s}")
                if m == "Ibreak":
                    pass
            else:
                accumulator[m] = np.nanmean(accumulator[m])
        return accumulator

    def _get_spike_info(self, mode: str):
        """analyze spike data

        Args:
            mode (str): Type of accumulation to do:
                EU: "experimental unit", usually animal
                OU: "observational unit", usually cell

        Raises:
            ValueError: mode out of spec.

        Returns:
            _type_: _description_
        """

        assert mode in ["OU", "EU"]

        Accumulator = pd.DataFrame(spike_measures)
        dates = self.dfs.date.unique()

        # for each mouse (day) :: Observational Unit AND Experimental Unit
        # The observational unit is the cell.
        # the Experimetnal unit is the day (perhaps it should be the "ID")

        OU_accumulator = []
        for day in dates:
            Day_accumulator = []
            for idx in self.dfs.index[
                self.dfs["date"] == day
            ]:  # for each cell in that day: Observational UNIT
                Cell_accumulator = self.make_emptydict(spike_measures)
                Cell_ID = self.make_cell(idx)
                coding = self.dfs.at[idx, "coding"]
                Group = self.dfs.at[idx, "Group"]
                Animal_ID = self.dfs.at[idx, "ID"]
                ages = self.make_emptydict(spike_measures)
                dspk = self.dfs.at[idx, "Spikes"]  # get the Spikes dict
                dfi = self.dfs.at[idx, "IV"]
                day_label = self.dfs.at[idx, "date"][:-4]
                if pd.isnull(dspk) or len(dspk) == 0:  # nothing in the Spikes column
                    continue
                for proto in dspk.keys():
                    Cell_accumulator = self._get_protocol_spike_data(
                        dspk,
                        dfi,
                        proto,
                        Cell_ID=Cell_ID,
                        Animal_ID=Animal_ID,
                        Group=Group,
                        coding=coding,
                        day=day,
                        accumulator=Cell_accumulator,
                    )

                Cell_accumulator = self._get_cell_means(
                    Cell_accumulator,
                    spike_measures,
                )  # get mean values for the cell
                OU_accumulator.append(Cell_accumulator)

        return OU_accumulator

        # if mode == "EU":  # mean by day
        #     for m in spike_measures:
        #         if m in no_average:
        #             print(m, OU_accumulator[m])
        #             if len(OU_accumulator[m]) > 0:
        #                 OU_accumulator[m] = OU_accumulator[m][0]
        #             # no change if it is empty...

        #         else:
        #             OU_accumulator[m] = np.nanmean(OU_accumulator[m])  # average across all cells in the day
        #     Accumulator = pd.concat((Accumulator, OU_accumulator), ignore_index=True)
        #     # code = self.dfs.at[idx, "coding"]
        #     # exp_unit = self.dfs.at[idx, "Date"]
        #     # spike_hues[m].append(code)
        #     # Cell_IDs[m].append(exp_unit)
        # elif mode == "OU":  # or mean from each individual cell
        #     Accumulator = pd.concat((Accumulator, pd.DataFrame(OU_accumulator)), ignore_index=True)
        #     # code = self.dfs.at[idx, "coding"]
        #     # obs_unit = self.dfs.at[idx, "Date"]
        #     # sl = self.dfs.at[idx, "slice_slice"]
        #     # cell = self.dfs.at[idx, "cell_cell"]
        #     # spike_hues[m].extend([code] * len(OU_accumulator[m]))
        #     # Cell_IDs[m].extend([Cell_ID])
        # else:
        #     raise ValueError("Mode not in EU, OU")
        #     # if self.coding is not None and  day in self.coding:
        #     #     EU_spike_hues[m].append(self.coding[day][1]) else:
        #     # EU_spike_hues[m].append('Z')
        # ot = f"Obs,Group"
        # for m in spike_measures:
        #     ot += f",{m:s}"
        # ot += "\n"
        # for i in range(len(accumulator[measures[0]])):
        #     ot += f"{i:d},{spike_hues[m][i]:s}"
        #     for m in spike_measures:
        #         ot += f",{accumulator[m][i]:f}"
        #     ot += "\n"
        # ofile = Path(f"R_{self.experiment}_Spikes_{self.celltype:s}_{self.mode}.csv")
        # ofile.write_text(ot)

        return Accumulator

    def plot_spike_info(self, mode, coding="code", parentFigure=None):
        """Break down the spike analysis and make a set of plots

        Args:
            mode (str): _description_
            coding (str, optional): name of the column holding the code. Defaults to "code".
            parentFigure (plotHelpers object): The parent figure into which the spike
            information will be plotted. Defaults to None.

        Returns:
            _type_: _description_
        """
        # create the plot grid for the spike analysis
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
        accumulator = self._get_spike_info(mode)
        dcomp = pd.DataFrame(accumulator)

        gps = {}
        for param in spike_measures:
            groups = {}
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

            if (
                param not in no_average
                and len(gps[param]["A"]) > 1
                and len(gps[param]["B"]) > 1
            ):
                t, p = scipy.stats.ttest_ind(
                    gps[param]["A"], gps[param]["B"], equal_var=False, nan_policy="omit"
                )
                print(f"\nParam: {param:<20s}:", end="")
                print(f"        t={t:.3f}  p={p:.4f}")
                print(
                    f"        A: mean={np.mean(gps[param]['A']):8.3f} std={np.std(gps[param]['A']):8.3f}, N={len(gps[param]['A']):d}"
                )
                print(
                    f"        B: mean={np.mean(gps[param]['B']):8.3f} std={np.std(gps[param]['B']):8.3f}, N={len(gps[param]['B']):d}"
                )

                # KW
                if "AA" in gps[param].keys() and "AAA" in gps[param].keys():
                    if len(gps[param]["AA"]) > 0 and len(gps[param]["AAA"]) > 0:
                        s, p = scipy.stats.kruskal(
                            gps[param]["B"],
                            gps[param]["A"],
                            gps[param]["AA"],
                            gps[param]["AAA"],
                        )
                        print(f"Krwukal-Wallis: H:{s:.6f}   p={p:.6f}\n")
                    elif len(gps[param]["AA"]) > 0 and len(gps[param]["AAA"]) == 0:
                        s, p = scipy.stats.kruskal(
                            gps[param]["B"],
                            gps[param]["A"],
                            gps[param]["AA"],
                        )
                        print(f"Krwukal-Wallis: H:{s:.6f}   p={p:.6f}\n")
            else:
                if param not in no_average:
                    print(
                        "\nEmpty data set for at least one category for param = ", param
                    )
            print("=" * 80)

        iax = 0
        for i, measure in enumerate(spike_measures):
            if measure is None or measure in no_average:
                print("skipping plotting measure: ", measure)
                continue
            yd = dcomp[measure].replace([np.inf], np.nan)

            x = pd.Series(dcomp["coding"])
            iasort = x.argsort()
            if np.all(np.isnan(yd)):
                iax += 1
                continue  # skip plot if no data

            dfm = pd.DataFrame({"Group": x, "measure": yd, "group": x})
            sns.violinplot(
                data=dfm,
                x="Group",
                y="measure",
                ax=pax[iax],
                order=["A", "B"],
                palette=palette,
                inner=None,
                saturation=1.0,
            )
            mpl.setp(pax[iax].collections, alpha=0.3)
            # sns.swarmplot(data=dfm, x='Group', y='measure', ax=pax[i], order=['A', 'B'], hue="Group", palette=palette, edgecolor='grey', size=2.5)
            sns.stripplot(
                data=dfm,
                x="Group",
                y="measure",
                ax=pax[iax],
                hue="Group",
                order=["A", "B"],
                palette=palette,
                edgecolor="grey",
                size=2.5,
            )
            # sns.boxplot(data = dfm, x='Group', y="measure",  ax=pax[i], palette=palette, orient='v', width=0.5, saturation=1.0)
            pax[iax].set_title(measure, fontsize=7)  # .replace('_', '\_'))

            pax[iax].set_ylim(paxes[measure])

            iax += 1

        iax = 0
        for i, measure in enumerate(spike_measures):
            if measure in no_average:
                continue
            if measure == "AdaptRatio":
                if pax[iax].get_legend() is not None:
                    mpl.setp(pax[iax].get_legend().get_texts(), fontsize="5")
                    pax[iax].legend(bbox_to_anchor=(1.02, 1.0))
            else:
                if pax[iax].get_legend() is not None:
                    pax[iax].get_legend().remove()
            iax += 1

        self.spike_dataframe = dcomp
        return self.Pspikes

    # def _get_measures(self, spikedict, fidict, age):
    #     """Get the spike data from all of the protocols
    #     in the spike dict

    #     Args:
    #         spikedict (dictionary): spike analysis from this protocol
    #         fidict (dictionary): fi analysis from this protocol
    #         ages (_type_): _description_

    #     Returns:
    #         _type_: _description_
    #     """
    #     cellmeas = self.make_emptydict(spike_measures)
    #     for protocol in spikedict.keys():  # for every protocol in the Spikes dict
    #         if not isinstance(protocol, Path):  # ooops, skip that one
    #             continue
    #         if not self.select_protocols(u):
    #             continue
    #         for m in spike_measures:
    #             if m is None:
    #                 continue
    #             age[m].append(age)
    #             if m == "spikethreshold":
    #                 fi = fidict[protocol]["FI_Curve"]
    #                 firstsp = np.where((fi[1] > 0) & (fi[0] > 0))[0]
    #                 if len(firstsp) > 0:
    #                     firstsp = firstsp[0]
    #                     cellmeas[m].append(fi[0][firstsp] * 1e12)  # convert to pA
    #                 else:
    #                     cellmeas[m].append(np.nan)
    #             elif m == "maxrate":
    #                 fi = fidict[protocol]["FI_Curve"]
    #                 firstsp = np.where((fi[1] > 0) & (fi[0] > 0))[
    #                     0
    #                 ]  # spikes and positive current together
    #                 if len(firstsp) > 0 and np.max(
    #                     fi[1] >= 2e-9
    #                 ):  # spikes and minimum current injection
    #                     cellmeas[m].append(
    #                         np.max(fi[1][firstsp]) / fidict[protocol]["pulseDuration"]
    #                     )  # convert to spikes/second
    #                 else:

    #                     cellmeas[m].append(np.nan)
    #             elif m == "Ibreak":
    #                 fig = fidict[protocol]["FI_Growth"]
    #                 if len(fig) > 0:
    #                     par = fig[0]["parameters"]
    #                     if len(par) > 0 and len(par[0]) > 0:
    #                         cellmeas[m].append(par[0][1])
    #             elif m == "Irate":
    #                 fig = fidict[protocol]["FI_Growth"]
    #                 if len(fig) > 0:
    #                     par = fig[0]["parameters"]
    #                     if len(par) > 0 and len(par[0]) > 0:
    #                         cellmeas[m].append(par[0][4])
    #             elif m == "FR_Slope":  # get slope from near-threshold firing
    #                 rate_spks = []
    #                 rate_i = []
    #                 fidata = fidict[protocol]["FI_Curve"]
    #                 for fsp in range(len(fidata[0])):
    #                     if fsp not in spikedict.keys():
    #                         continue
    #                     nspkx = len(spikedict[fsp])
    #                     if nspkx > 0 and fidata[0][fsp] > 0.0:
    #                         if len(rate_spks) < 3:
    #                             rate_spks.append(
    #                                 nspkx / fidict[protocol]["pulseDuration"]
    #                             )
    #                             rate_i.append(fidata[0][fsp] * 1e9)
    #                 if len(rate_i) > 0:

    #                     p = np.polyfit(rate_i, rate_spks, 1)
    #                     cellmeas[m].append(p[0])  # store slope from fit
    #             else:

    #                 if m in list(d[u].keys()):  #  and m not in measures_fromspikes:
    #                     cellmeas[m].append(d[u][m])

    #                 else:  # loop through traces
    #                     xm = []
    #                     spkdata = spikedict[protocol]["spikes"]
    #                     for tr in spkdata.keys():  # each trace with spikes
    #                         for spk in spkdata[tr]:
    #                             if m in spikedict[protocol]["spikes"][tr][spk].keys():
    #                                 if (
    #                                     spikedict[protocol]["spikes"][tr][spk][m]
    #                                     is not None
    #                                 ):
    #                                     xm.append(
    #                                         spikedict[protocol]["spikes"][tr][spk][m]
    #                                     )

    #                     if len(xm) > 0:
    #                         cellmeas[m].append(np.nanmean(xm))
    #     return cellmeas

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

    def _get_iv_parameter(
        self,
        measure: str,
        measurename: str,
        cellmeas: dict,
        accumulator: dict,
        func: object = np.nanmean,
    ):
        """Get the IV values averaged (or smallest) from one protocol/measurement in one cell
        accumulate the measures in the accumulator (dict of measures), using lists.

        Args:
            measure: the name of the measure to be stored
            measurename: the name of the measure as found in the iv array
            accumulator (dict): _description_
            cellmeas (_type_): _description_
            func (function object): name of function to use on values (np.nanmean, np.nanmin)
        Returns:
            updated accumulator
        """

        if pd.isnull(cellmeas[measure]):
            accumulator[measure].append(np.nan)
        else:
            accumulator[measure].append(cellmeas[measure])
        return accumulator

    def _get_protocol_iv_data(
        self,
        dspk: dict,
        dfi: dict,
        proto: str,
        Cell_ID: str,
        Animal_ID: str,
        coding: str,
        Group: str,
        day,
        accumulator: dict,
    ):
        """Get the iv data for the specified cell
        and protocol in the dataframe
        computes the mean or min of the parameter as needed if there are multiple measures
        within the protocol.
        The accumulator dict has keys indicating the Cell_ID, etc.
        Args:
            dspk (dict): spike dictionary for one cell
            dfi (dict): FI dictionary (IV) for one cell
            proto (str): Protocol name within the dict to use
        """
        cellmeas = dfi[proto]
        tracekeys = list(cellmeas.keys())
        for m in iv_measures:
            if m not in no_average:
                accumulator = self._get_iv_parameter(
                    measure=m,
                    measurename=m,
                    cellmeas=cellmeas,
                    accumulator=accumulator,
                    func=np.nanmean,
                )

            elif m in no_average:
                if m == "Cell_ID":
                    accumulator[m].append(Cell_ID)
                if m == "Date":
                    accumulator[m].append(day)
                if m == "Animal_ID":
                    accumulator[m].append(Animal_ID)
                if m == "Group":
                    accumulator[m].append(Group)
                if m == "coding":
                    accumulator[m].append(coding)
                if m == "protocol":
                    accumulator[m].append(proto)
                if m == "Ibreak":
                    accumulator[m].append(cellmeas[m])
            # elif m not in no_average:
            #     accumulator[m].append(np.nanmean(cellmeas[m]))
            else:
                cprint("r", f"Failed to find data for measure: {m:s}")
                cprint("r", f"     {(tracekeys):s}")
                accumulator[m].append(np.nan)
        return accumulator

    def _get_iv_info(self, mode: str):
        """analyze iv data

        Args:
            mode (str): Type of accumulation to do:
                EU: "experimental unit", usually animal
                OU: "observational unit", usually cell

        Raises:
            ValueError: mode out of spec.

        Returns:
            _type_: _description_
        """

        assert mode in ["OU", "EU"]

        Accumulator = pd.DataFrame(iv_measures)
        dates = self.dfs.date.unique()

        # for each mouse (day) :: Observational Unit AND Experimental Unit
        # The observational unit is the cell.
        # the Experimetnal unit is the day (perhaps it should be the "ID")

        OU_accumulator = []
        for day in dates:
            Day_accumulator = []
            for idx in self.dfs.index[
                self.dfs["date"] == day
            ]:  # for each cell in that day: Observational UNIT
                Cell_accumulator = self.make_emptydict(iv_measures)
                Cell_ID = self.make_cell(idx)
                coding = self.dfs.at[idx, "coding"]
                Group = self.dfs.at[idx, "Group"]
                Animal_ID = self.dfs.at[idx, "ID"]
                ages = self.make_emptydict(iv_measures)
                dspk = self.dfs.at[idx, "Spikes"]  # get the Spikes dict
                dfi = self.dfs.at[idx, "IV"]

                day_label = self.dfs.at[idx, "date"][:-4]
                if pd.isnull(dspk) or len(dspk) == 0:  # nothing in the Spikes column
                    continue
                for proto in dspk.keys():
                    Cell_accumulator = self._get_protocol_iv_data(
                        dspk,
                        dfi,
                        proto,
                        Cell_ID=Cell_ID,
                        Animal_ID=Animal_ID,
                        Group=Group,
                        coding=coding,
                        day=day,
                        accumulator=Cell_accumulator,
                    )

                Cell_accumulator = self._get_cell_means(
                    Cell_accumulator,
                    iv_measures,
                )  # get mean values for the cell
                OU_accumulator.append(Cell_accumulator)

        return OU_accumulator

    def plot_IV_info(self, mode, parentFigure=None):
        """
        Mode can be OU for observational unit (CELL) or EU for experimental Unit
        (Animal)
        """
        #         ### IV
        EU_accumulator = self.make_emptydict(iv_measures)
        EU_iv_hues = self.make_emptydict(iv_measures)
        EU_Cell_IDs = self.make_emptydict(iv_measures)

        modestring = "Unit = Experiment (animal)"
        if mode == "OU":
            modestring = "Unit = Observational (cell)"
        cprint("c", f"Accumulating by mode: {modestring:s}")
        dates = self.dfs.date.unique()
        accumulator = self._get_iv_info(mode)
        dcomp = pd.DataFrame(accumulator)
        # for each mouse (day) :: Biological Unit AND Experimental Unit
        # for day in dates:
        #     OU_accumulator = self.make_emptydict(iv_measures)  #
        #     for idx in self.dfs.index[
        #         self.dfs["date"] == day
        #     ]:  # for each cell in that day: Observational UNIT
        #         div = self.dfs.at[idx, "IV"]  # get the IV dict
        #         day = self.dfs.at[idx, "date"][:-4]
        #         cellmeas = self.make_emptydict(iv_measures)
        #         if pd.isnull(div):
        #             continue
        #         print("div.keys: ", div.keys())
        #         for u in div.keys():
        #             if not isinstance(u, Path):
        #                 continue
        #             for m in iv_measures:
        #                 if m not in list(div[u].keys()):
        #                     continue
        #                 if div[u][m] is not None:
        #                     cellmeas[m].append(div[u][m])
        #                 else:
        #                     continue  # cellmeas[m].append(np.nan)
        #                 # cellmeas[m] = [x for x in cellmeas[m] if x != None] if
        #                 # day in self.coding:
        #                 #     iv_hue[m].append(self.coding[day][1]) else:
        #                 # iv_hue[m].append('Z')

        #         for m in OU_accumulator.keys():
        #             ms = cellmeas[m]
        #             if len(cellmeas[m]) == 0:
        #                 ms = np.nan
        #             else:
        #                 ms = np.nanmean(cellmeas[m])
        #             OU_accumulator[m].append(ms)  # accumulate average within the cell
        #     # now accumulate the experimental units (mean for each DAY)
        #     for m in iv_measures:
        #         if mode == "EU":  # mean by day
        #             EU_accumulator[m].append(np.nanmean(OU_accumulator[m]))
        #             code = self.dfs.at[idx, "coding"]
        #             exp_unit = self.dfs.at[idx, "Date"]
        #             EU_iv_hues[m].append(code)
        #             EU_Cell_IDs[m].append(exp_unit)
        #         elif mode == "OU":  # or meany from each individual cell
        #             EU_accumulator[m].extend(OU_accumulator[m])
        #             code = self.dfs.at[idx, "coding"]
        #             obs_unit = self.dfs.at[idx, "Date"]
        #             sl = self.dfs.at[idx, "slice_slice"]
        #             cell = self.dfs.at[idx, "cell_cell"]
        #             EU_iv_hues[m].extend([code] * len(OU_accumulator[m]))
        #             EU_Cell_IDs[m].extend(
        #                 [f"{str(obs_unit):s}_{sl:s}_{cell:s}"] * len(OU_accumulator[m])
        #             )

        gps = {}
        for param in iv_measures:
            groups = {}
            with pd.option_context("mode.use_inf_as_null", True):
                dcomp = dcomp.dropna(subset=[param, "coding"], how="all")

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
            if (
                param not in no_average
                and len(gps[param]["A"]) > 1
                and len(gps[param]["B"]) > 1
            ):

                t, p = scipy.stats.ttest_ind(
                    gps[param]["A"], gps[param]["B"], equal_var=False, nan_policy="omit"
                )
                print(f"\nParam: {param:>20s}:", end="")
                print(f"        t={t:.3f}  p={p:.4f}")
                print(
                    f"        A: mean={np.mean(gps[param]['A']):8.3f} std={np.std(gps[param]['A']):8.3f}, N={len(gps[param]['A']):d}"
                )
                print(
                    f"        B: mean={np.mean(gps[param]['B']):8.3f} std={np.std(gps[param]['B']):8.3f}, N={len(gps[param]['B']):d}"
                )

                if (
                    len(gps[param]["A"]) > 0
                    and len(gps[param]["B"] > 0)
                    and "AA" in gps[param].keys()
                    and "AAA" in gps[param].keys()
                ):
                    s, p = scipy.stats.kruskal(
                        gps[param]["A"],
                        gps[param]["B"],
                        gps[param]["AA"],
                        gps[param]["AAA"],
                    )
                    print(f"Krwukal-Wallis: H:{s:.6f}   p={p:.6f}\n")
                elif (
                    len(gps[param]["A"]) > 0
                    and len(gps[param]["B"] > 0)
                    and "AA" in gps[param].keys()
                    and "AAA" not in gps[param].keys()
                ):
                    s, p = scipy.stats.kruskal(
                        gps[param]["A"],
                        gps[param]["B"],
                        gps[param]["AA"],
                    )
                    print(f"Krwukal-Wallis: H:{s:.6f}   p={p:.6f}\n")
                print("=" * 80)

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

        iax = 0
        for i, measure in enumerate(iv_measures):
            if measure in no_average:
                continue

            x = pd.Series(dcomp["coding"])
            yd = dcomp[measure].replace([np.inf], np.nan)
            iasort = x.argsort()
            if np.all(np.isnan(yd)):
                iax += 1
                continue  # skip plot if no data

            dfm = pd.DataFrame({"Group": x, "measure": yd, "group": x})
            # df_iv = df_iv[df_iv[measure].replace([np.inf], np.nan)]
            sns.violinplot(
                data=dfm,
                x="Group",
                y="measure",
                ax=paxb[iax],
                order=["A", "B"],
                palette=palette,
                inner=None,
            )
            mpl.setp(paxb[iax].collections, alpha=0.3)
            # sns.swarmplot(data=y, x='coding', y=m, ax=paxb[i], hue="coding", order=['A', 'B'], palette=palette, size=2.5)
            sns.stripplot(
                data=dfm,
                x="Group",
                y="measure",
                ax=paxb[iax],
                hue="Group",
                order=["A", "B"],
                palette=palette,
                size=2.5,
            )
            paxb[iax].set_title(measure, fontsize=7)  # .replace('_', '\_'))
            paxb[iax].set_ylim(paxesb[measure])
            if measure == "Rin":
                if paxb[iax].get_legend() is not None:
                    mpl.setp(paxb[iax].get_legend().get_texts(), fontsize="6")
                    paxb[iax].legend(bbox_to_anchor=(-0.35, 1.0))
                    # ap1l = d[u]['AP1_Latency'] ap1hw = d[u]['AP1_HalfWidth']
                    # apthr = d[u]['FiringRate_1p5T']
            else:
                paxb[iax].get_legend().remove()
            iax += 1

        self.iv_dataframe = dcomp
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
                print("age: ", age, data["age"], "failed to parse to integer")
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
