"""IV_Summarize:
A module to summarize data from individual cells generated through from iv_analysis
in the ephys.ephys_analysis library. iv_Analysis makes measurements of passive properties
and spike shape from individual cells and protocols using ephys_analysis modules
rm_tau_summary and spike_summary

This is part of the ephys IV analysis pipeline:
    data_summary -> iv_analysis -> summarize_ivs

This program takes as input the output of iv_analysis, but also refers to the data summary
file.

The output is a figure with FI curves, plots of general physiology (Rm, Rin, tau, tauh, etc),
and spike parameters. Data is sorted by "group" as set by the "codes" tab of the excel sheet.
The codes sheet should also include a subject id for completeness.

The data can be summarized either by Observational Unit (e.g., cell) or by Experimental Unit (e.g., subject,
with all cells averaged together). The choice of appropriate "unit" will depend on the experimental design. 
For example, EU is the appropriate mode when dealing with KO or transgenic mice where the same manipulation is applied
to all cells. OU is the appropriate mode when dealing with sparse expression (e.g., tamoxifen-induced 
recombination) where the cells in a given subject may have different gene expression patterns, and the
expression pattern can be identified during data collection. 
Currently, there is no "intermediate" mode, where all of the OU data are computed for a subject, and then
averaged by the "code" with each EU. 

 
This module can be used in two ways:
1. Command line - running from the command line. This may require changing the default "experiments" dictionary
    (see the main() function)

2. As a class (module): 
    This is the recommended mode; build a project-specific script to set the paths to the data
    and metadata/summary tables. That script should be able to run the analysis (iv_analysis) and then
    call this module to summarize the results across cells/experiments.

    Import, setup and run
    g = GetAllIVs(arguments_dict)
    g.set_protocols(["CCIV_long_HK", "CCIV_1nA_max_1s", "CCIV_200pA"])
    g.set_experiments(experiments)
    g.set_codemetaname(name) # 
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
import dataclasses
import pickle
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
from ephys.tools.clean_database_merge import clean_database_merge
from ephys.tools.parse_ages import ISO8601_age
from pylibrary.tools import cprint as CP
from ephys.tools import db_tools as DBTOOLS

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

# measures and metadata to track for the IV analysis:
iv_measures = [
    "Date",
    "Cell_ID",
    "cell_expression",
    "genotype",
    "Animal_ID",
    "code",  # group code assigned to each animal/cell
    "sex",
    "age",
    "Group",
    "protocol",
    "Rin",
    "RMP",
    "taum",
    "tauh_bovera",
    "tauh_tau",
]

ylabels = {
    "Rin": r"M$\Omega$",
    "RMP": "mV",
    "taum": "sec",
    "tauh_bovera": "AU",
    "tauh_tau": "sec",
    "AP1_HalfWidth": "ms",
    "AP1_HalfWidth_interpolated": "ms",
    "AP2_HalfWidth": "ms",
    #'AP2_HalfWidth_interpolated',
    "peak_V": "V",
    "AHP_Depth": "mV",
    "peaktotrough": "mV",
    "spikethreshold": "A",
    "AP1_Latency": "ms",
    "AP2_Latency": "ms",
    "Ibreak": "A",
    "Irate": "spk/s",
    "FiringRate_1p5T": "spk/s",
    "AdaptRatio": "AU",
    "maxrate": "spk/s",
    "FR_Slope": "spk/s/A",
}
# spike measures to track
# includes metadata (Date, Coding: Cell_ID, Animal_ID, etc.)
spike_measures = [
    "Date",
    "Cell_ID",
    "cell_expression",  # from the acq4 metadata
    "genotype",
    "Animal_ID",
    "Group",
    "code",  # group code assigned to each animal/cell
    "age",
    "sex",
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

# categorical data and metadata in the tracked items that should not be averaged numerically
no_average = [
    "Date",
    "cell_expression",
    "genotype",
    "Cell_ID",
    "Animal_ID",
    "Group",
    'grouping',
    "protocol",
    "age",
    "sex",
    "code",  # group code assigned to each animal/cell
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
paxes["AP1_HalfWidth_interpolated"] = [0, 2]
paxes["AP2_HalfWidth_interpolated"] = [0, 2]
paxes["AP1_HalfWidth"] = [0, 2]
paxes["AP2_HalfWidth"] = [0, 2]
paxes["AHP_Depth"] = [0, 20]
paxes["FiringRate_1p5T"] = [0, 40]
paxes["Ibreak"] = [-0.00, 1.0]
paxes["Irate"] = [0, 10.00]
paxes["peak_V"] = [-0.010, 0.060]
paxes["peaktotroughT"] = [0, 0.040]
paxes["peaktotrough"] = [0, 0.040]
paxes["spikethreshold"] = [0, None]
paxes["maxrate"] = [0, 150]
paxes["age"] = [0, 1000]
paxes["FR_Slope"] = [0, 1000]

p_skip_plot = ["Ibreak", "FR_Slope", "Irate"]

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
        self.group_mode = "genotype"  # or code, or some other column
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

        self.set_group_mode("genotype")  # set a default code to break down results
        self.set_exclude_codes([])  # empty list - allow all codes
        codecolors = OrderedDict(
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
        self.set_code_colors(codecolors)
        self.markercode = None
        self.markerfills = None
        self.marker_names = None
        self.iv_dataframe = None
        self.spike_dataframe = None

    def set_code_colors(self, codecolors: dict):
        """assign codes to colors using a dict.
        This map will replace the original code list and codes

        Args:
            codecolors (dict): A list of codes and their colors:
            for example: {'control': 'black', 'experimental': 'red', '?': 'blue', "ND": 'blue'}

        """
        self.codecolors = codecolors
        self.code_names = list(self.codecolors.keys())

    def set_code_markerfill(self, markercode: str = None, markerfills: dict = None):
        """assign marker fills symbols using a dict.
        This map will replace the original code list and codes

        Args:
            marker_code: What parameter in the database is used to set the marker fills
                for example: "cell_expression"
            markerfills (dict): A list of codes and their colors
                for example: {'GFP+': 'filled', 'GFP-': 'open'}

        """
        self.markercode = markercode
        self.markerfills = markerfills
        self.marker_names = list(self.markerfills.keys())

    def set_group_mode(self, group_mode: str = "genotype"):
        """pick out the column from the dataSummary table that is used to pull the "code" from the data
        The name should be the name of one of the columns
        Args:
            group_mode (str): column name to group data by
        """
        assert group_mode in ["genotype", "code", "Group", "cell_expression", "expression_and_genotype"]
        self.group_mode = group_mode

    def set_exclude_codes(self, excludelist: list = []):
        self.exclude_codes = excludelist

    def set_default_group(self, row):
        """Set a default group for a row in the dataframe
        Args:
            row (pandas dataframe row): row of the dataframe to set the group for
        Returns:
            string: group name
        """
        row.Group = "Ctl" 
        return row.Group
    
    def set_experiments(self, experiments: Union[dict, list]):
        """experiments is the dict for experiments - paths, etc. each dict entry has a
            key naming the "experiment" The list has [path_to_data,
            nf107_directory_for_database, database_file_name,
            experiment_coding_dictionary]

        For example, the experiments dictionary might contain:

        '''
            rawdatadisk = "/Volumes/Pegasus_002/ManisLab_Data3/Kasten_Michael/Maness_Ank2_PFC_stim"
            resultdisk = "/Users/pbmanis/Desktop/Python/ank2/ank2_datasets"
            experiments = {
                "Ank2B": {
                    "rawdatapath": rawdatadisk,
                    "databasepath": resultdisk,# location of database files (summary, coding, annotation)
                    "analyzeddatapath": resultdisk, # analyzed data set directory
                    "directory": "", # directory for the raw data, under rawdatadisk
                    "pdfFilename": "Ank2B_IVs.pdf", # PDF figure output file
                    "db_directory": "ANK2",
                    "datasummaryFilename": "Intrinsics",
                    "iv_analysisFilename": "ANK2_NEX_IVs.pkl",
                    "coding_file": None,
                    "coding_sheet": None,
                    "annotationFilename": None,
                    "maps": None,
                    "extra_subdirectories": ["Rig2(PBM)/L23_intrinsic", "Rig4(MRK)/L23_intrinsic"], # directories to include in analysis under rawdatadisk
                    "skip_subdirectories": ["Rig4(MRK)/mEPSCs", "Rig4(MRK)/PSCs"]  # directories to skip in this type of analysis
                }
            }
        '''
        Experiments can have multiple entries as well, which are selected by using '+' when specifying the
        "-E" option from the command line (e.g., 'nf107_nihl+nf107_control')
        Args:
            experiments (dict): Dictionary for experiments, data directories, etc.
        """

        self.experiments = experiments


    def build_dataframe(self):
        """Build pandas data frame from selected experiments
        This function builds a new pandas dataframe ("self.dfs") that
        is the result of merging multiple experiments.
        An intermediate file is written to disk in excel format for viewing.

        Args:
            None

        Modifies:
            Creates self.dfs, the dataframe of merged experiments.
        """
        print("Building merged dataframe")
        self.dfs = pd.DataFrame()
        if self.experiment is not None:
            if "+" in self.experiment:
                expts = self.experiment.split("+")
            else:
                expts = [self.experiment]
            # print(f"Experiments: {str(expts):s}")
            for i in range(len(expts)):
                expt = self.experiments[expts[i]]
                self.analyzed_datapath = Path(expt['databasepath'], expt['directory'])
                cprint("g", f"   Analyzing experiment: {str(expts[i]):s}")
                self.basedir = Path(expt["rawdatapath"])
                self.inputFilename = Path(
                    expt["databasepath"], expt['directory'],
                    expt["iv_analysisFilename"],
                ).with_suffix(".pkl")
                if expt["coding_file"] is not None:
                    if expt["coding_sheet"] is None:
                        raise ValueError(
                            f"Coding sheet in the file {str(expt['coding_file']):s} must be specified; got 'None'"
                        )
                    self.outputPath = Path(expt["databasepath"])

                    coding_f = Path(expt["databasepath"], expt['directory'], expt["coding_file"])
                    sheet_name = expt["coding_sheet"]
                    print("    Input file name: ", self.inputFilename)
                    df_i = clean_database_merge(
                        pkl_file=self.inputFilename,
                        coding_file=coding_f,
                        coding_sheet=sheet_name,
                    )
                    self.dfs = pd.concat((self.dfs, df_i))
                    self.dfs["Group"] = self.dfs["Group"].values.astype(str)
                else:
                    self.dfs = pd.read_pickle(self.inputFilename)
                    self.dfs["Group"] = self.dfs.apply(self.set_default_group, axis=1)
                groups = sorted(list(set(self.dfs.Group.values)))
                dates = sorted(list(set(self.dfs.date.values)))
                print(f"    Expt: {expts[i]:s}, # dates found: {len(dates):d}")
                # for date in dates:
                #     print(f"       {date:s}")
                CP.cprint("c", f"    with these groups: {str(groups):s}")


        self.dfs = self.dfs.reset_index(drop=True)
        self.rainbowcolors = iter(
            matplotlib.cm.rainbow(np.linspace(0, 1, len(self.dfs)))
        )
        self.dfs.to_excel("iv_summarize_intermediate_datafile.xlsx")
        cprint(
            "g",
            "All data loaded, intermediate date file iv_summarize_intermediate_datafile.xlsx written",
        )
        print(self.dfs.cell_id)
        print("merged dataframe built")
        return

    def print_census(self, df_accum):
        """Print out a census of the mice in the accumlated data set
        This is a bit too specific at the moment, but gives the exp date,
        animal information, code, cell ID, and number of protocols run.
        A summary of counts by sex and code is generated, but this is TOO specific.

        Args:
            df_accum (_type_): _description_
        """
        # print("colnames: ", df_accum.columns)

        df_accum = df_accum.rename(columns={"animal identifier": "Animal_ID"})
        # ident = list(set(df_accum["Animal_ID"].values))
        print("Census columns: ", df_accum.columns)
        cell_ids = list(set(df_accum["Cell_ID"].values))
        # counts = {'M_GFP+': 0, 'M_GFP-':0, 'F_GFP+': 0, 'F_GFP-':0}
        print("Census: ")
        # print("Columns in df_accum: ", df_accum.columns)
        groups = list(set(df_accum["code"].values))
        counts = {}
        for g in groups:
            counts[g] = 0
        # counts = {'M_WT': 0, 'M_FF':0, 'F_WT': 0, 'F_FF':0}
        for idn in sorted(cell_ids):
            an = df_accum[df_accum["Cell_ID"] == idn]

            print(
                str(Path(an.Date.values[0]).name),
                " ID: ",
                an.Animal_ID.values[0],
                " Sex: ",
                an.sex.values[0],
                " Code: ",
                an.code.values[0],
                " Cell ID: ",
                an.Cell_ID.values[0],
                " # Prots: ",
                len(an.protocol.values[0]),
                an.protocol.values[0],
            )
            # assemble count key:
            # ckey = f"{an.sex.values[0]:s}_{an.code.values[0]:s}"
            ckey = f"{an.code.values[0]:s}"
            counts[ckey] += 1
        print("Counts: ", counts, "\n <End of Census>")
        print("-" * 80)

    def run(self):
        """Generate a summary with current parameters and build a summary plot of the measurements."""
        if self.dfs is None:
            self.build_dataframe()

        # create primary figure space
        self.PFig = PH.regular_grid(
            1,
            1,
            order="columnsfirst",
            figsize=(11.0, 11.0),
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
        mpl.show()
        
        P1 = self.plot_IV_info(self.mode, code=self.group_mode, parentFigure=self.PFig)
        # P2 = self.plot_spike_info(
        #     self.mode, code=self.group_mode, parentFigure=self.PFig
        # )
        mpl.show()

    def set_protocols(self, protocols: list):
        """Set the list of protocols that are selected for the analysis.
        Only protocols with names in this list will be summarized.

        This should be called at the "user" level according to the names
        of the relevant protocols in the dataset.

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
            bool: True if the protocol name is in the list of protocols we are analyzing,
            False otherwise.
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
        print("FI Plot")
        self.dfs = self.set_grouping(self.dfs)
        for idx in self.dfs.index:
            # print("   ", self.dfs.columns)
            if "Spikes" in self.dfs.iloc[idx]:
                dx = self.dfs.iloc[idx]["Spikes"]  # get the spikes dict
            else:
                dx = None
            code = self.dfs.iloc[idx]['grouping']
            if isinstance(code, list):
                code = code[0]
            cn = code.split("_")
            genotype = "U"
            code = cn[0]
            if len(cn) > 1:
                genotype = cn[1]

            if code not in self.code_names:
                continue
            color = self.codecolors[code]
            if dx is None:
                continue  # no data to analyze
            if isinstance(dx, dict):  # convert to 1-element list
                dx = [dx]
            for dv in dx:
                for fidata in dv.keys():
                    firawdata = Path(self.dfs.iloc[idx].data_directory, fidata)
                    if not self.select_protocols(fidata):
                        continue
                    if code not in legcodes:
                        legcodes.append(code)
                        leglabel = code
                    else:
                        leglabel = None
                    if "FI_Curve" not in list(dv[fidata].keys()):
                        continue
                    fi = dv[fidata]["FI_Curve"]
                    imax = np.argmax(fi[1])   # only plot to max firing rate
                    lp = PF.axdict["A"].plot(
                        fi[0][:imax] * 1e9,
                        fi[1][:imax] / dv[fidata]["pulseDuration"],
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
        plotcodes = list(set(legcodes).intersection(set(self.code_names)))
        cprint("r", f"plotcodes: {str(plotcodes):s}, legcodes")
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
        """Get spike values averaged (or smallest for some parameters)
        from one protocol/measurement in one cell
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
                    onespike = dataclasses.asdict(cellmeas["spikes"][itr][ispk])
                    values.append(onespike[measurename])
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
        Group: str,
        sex: str,
        code: str,
        genotype: str,
        cell_expression: str,
        grouping: str,
        # Group: str,
        day: str = None,
        accumulator: dict = None,
    ):
        """Get the spike data for this protocol
        Computes the mean or min of the parameter as needed if there are multiple measures
        within the protocol.

        The accumulator dict has keys indicating the Cell_ID, etc.
        Args:
            dspk (dict): Spike dictionary for one cell
            dfi (dict): FI dictionary (IV) for one cell
            proto (str):  Protocol name within the dict to use
            Cell_ID (str): the ID for the cell (path, date, slice, cell)
            Animal_ID (str): Animal identifier from acq4 metadata
            Group (str): Assigned group (from acq4 metadata or a table)
            sex (str): Animal sex from acq4 metadata
            code (str): Assigned code (may be from a code table)
            genotype (str): genotype from acq4 metadata
            cell_expression (str): Flag as to whether cell is expressing or not
            day (_type_): date of experiment
            accumulator (dict): The previous accumulator data for this cell; elements are appended

        Returns:
            dict: updated accumulator
        """
        cellmeas = dspk[proto]
        if "spikes" in list(cellmeas.keys()):
            tracekeys = list(cellmeas["spikes"].keys())
        else:
            return self.make_emptydict(spike_measures)

        # print("get_protocol_spike: ", Cell_ID)
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
                if m == "sex":
                    accumulator[m].append(sex)
                if m == "genotype":
                    accumulator[m].append(genotype)
                if m == "cell_expression":
                    accumulator[m].append(cell_expression)
                if m == "Group":
                    accumulator[m].append(Group)
                if m == "grouping":
                    accumulator[m].append(grouping)
                if m == "code":
                    accumulator[m].append(code)
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
        if len(accumulator["Cell_ID"]) == 0:
            return accumulator
        for m in measures:
            if m in no_average:
                if m == "Cell_ID":
                    accumulator[m] = accumulator[m][0]
                if m == "Date":
                    accumulator[m] = accumulator[m][0]
                if m == "Animal_ID":
                    accumulator[m] = accumulator[m][0]
                if m == "sex":
                    accumulator[m] = accumulator[m][0]
                if m == "genotype":
                    accumulator[m] = accumulator[m][0]
                if m == "code":
                    accumulator[m] = accumulator[m][0]
                if m == "grouping":
                    accumulator[m] = accumulator[m][0]
                if m == "Group":
                    # print("Accumulator[m]: ", accumulator[m], m)
                    if len(accumulator[m]) > 0:
                        accumulator[m] = accumulator[m][0]
                if m == "cell_expression":
                    accumulator[m] = accumulator[m][0]
                if m == "protocol":
                    accumulator[m] = [Path(k).name for k in accumulator[m]]
                    # cprint('y', f"\n {m:s}, {str(accumulator[m]):s}")
                if m == "Ibreak":
                    pass
            else:
                if len(accumulator[m]) > 0:
                    accumulator[m] = np.nanmean(accumulator[m])
                else:
                    accumulator[m] = np.nan
        return accumulator

    def _get_spike_info(self, mode: str, existing=None):
        """analyze spike data

        Args:
            mode (str): Type of accumulation to do:
                EU: "experimental unit", usually animal
                OU: "observational unit", usually cell
            existing: existing dataframe to put data into.
                if None, a new dataframe is created
        Raises:
            ValueError: mode out of spec.

        Returns:
            _type_: _description_
        """

        assert mode in ["OU", "EU"]

        # Accumulator = pd.DataFrame(spike_measures)
        dates = self.dfs.date.unique()
       
        # for each mouse (day) :: Observational Unit AND Experimental Unit
        # The observational unit is the cell.
        # the Experimetnal unit is the day (perhaps it should be the "Animal_ID")
        print("get_spike info: dfs head: ", self.dfs.head())
        exit()
        if existing is None:
            OU_accumulator = []
        protocol_list = []
        for day in dates:
            cprint("y", f"Day: {str(day):s}")
            for idx in self.dfs.index[
                self.dfs["date"] == day
            ]:  # for each cell in that day: Observational UNIT
                Cell_accumulator = self.make_emptydict(spike_measures)
                Cell_ID = DBToools.make_cell_ID(self.dfs, idx)
                # cprint("g", f"    gspk: Cell: {str(Cell_ID):s}   idx: {idx:d}, cell_id: {self.dfs.iloc[idx]['cell_id']:s} ")
                if 'code' in self.dfs.columns:
                    code = self.dfs.iloc[idx]['code']
                else:
                    code = ''
                Animal_ID = self.dfs.iloc[idx]["Animal_ID"]
                Group = self.dfs.iloc[idx]["Group"]
                grouping = self.dfs.iloc[idx]["grouping"]
                if "genotype" not in self.dfs.columns:
                    self.dfs["genotype"] = "U"
                else:
                    genotype = self.dfs.iloc[idx]["genotype"]
                if "cell_expression" not in self.dfs.columns:
                    self.dfs["cell_expression"] = "U"
                else:
                    cell_expression = self.dfs.iloc[idx]["cell_expression"]
                sex = self.dfs.iloc[idx]["sex"]
                ages = self.make_emptydict(spike_measures)
                dspk = self.dfs.iloc[idx]["Spikes"]  # get the Spikes dict
                dfi = self.dfs.iloc[idx]["IV"]
                day_label = self.dfs.iloc[idx]["date"][:-4]
                if pd.isnull(dspk) or len(dspk) == 0:  # nothing in the Spikes column
                    cprint("r", "    Empty spikes")
                    continue
                n_protocols = 0
                for proto in dspk.keys():
                    if proto not in protocol_list:
                        protocol_list.append(proto)
                    else:
                        continue
                    cprint("c", f"        gspk: Protocol: {str(proto):s}")
                    Cell_accumulator = self._get_protocol_spike_data(
                        dspk=dspk,
                        dfi=dfi,
                        proto=proto,
                        Cell_ID=Cell_ID,
                        Animal_ID=Animal_ID,
                        Group=Group,
                        code=code,
                        genotype=genotype,
                        cell_expression=cell_expression,
                        grouping=grouping,
                        sex=sex,
                        day=day,
                        accumulator=Cell_accumulator,
                    )
                    n_protocols += 1
                if n_protocols > 0:
                    Cell_means = self._get_cell_means(
                        Cell_accumulator,
                        spike_measures,
                    )  # get mean values for the cell
                    OU_accumulator.append(Cell_means)
        accumulator = pd.DataFrame(OU_accumulator)

        return accumulator

    def nancount(self, a):
        return int(len(~np.isnan(a)))

    def group_by_sex_and_genotype(self, row):
        if 'sex' in row.names and 'genotype' in row.names:
            row.grouping = f"{row.sex:s}_{row.genotype:s}"
        else:
            row.grouping = f"U_U"

        return row

    def group_by_expression(self, row):
        if row.cell_expression in ['+', 'gFP+']:
            row.cell_expression = 'GFP+'
        if row.cell_expression in ['-']:
            row.cell_expression = 'GFP-'
        if row.cell_expression in [' ']:
            row.cell_expression = 'ND'
        row.grouping = f"{row.cell_expression:s}"
        return row

    def group_by_expression_and_genotype(self, row):
        if row.cell_expression in ['+', 'gFP+']:
            row.cell_expression = 'GFP+'
        if row.cell_expression in ['-']:
            row.cell_expression = 'GFP-'
        if row.cell_expression in [' ']:
            row.cell_expression = 'ND'
        row.grouping = f"{row.cell_expression:s}_{row.genotype:s}"
        return row
    
    def group_by_code(self, row):
        row.grouping = f"{row.code:s}"
        return row

    def group_by_group(self, row):
        row.grouping = f"{row.Group:s}"
        return row

    def set_grouping(self, df_accum):
        print("Group mode: ", self.group_mode)

        # if "code" in df_accum.columns:
        #     df_accum = df_accum[df_accum.code != " "]  # remove unidentified groups
        #     df_accum = df_accum[df_accum.cell_expression != "ND"]  # remove unidentified cells
        df_accum["grouping"] = "" # assign some groupings in a new column
        match self.group_mode:
            case "genotype":
                df_accum = df_accum.apply(self.group_by_sex_and_genotype, axis=1)
            case "code":
                df_accum = df_accum.apply(self.group_by_code, axis=1)
            case "Group":
                df_accum = df_accum.apply(self.group_by_group, axis=1)
            case "cell_expression":
                df_accum = df_accum.apply(self.group_by_expression, axis=1)
            case "expression_and_genotype":
                df_accum = df_accum.apply(self.group_by_expression_and_genotype, axis=1)
            case _ :
                raise ValueError(
                    f"Grouping mode: {self.group_mode:s} did not match known values [genotype, code, Group]"
                )
        return df_accum

    def plot_summary_info(
        self, datatype: str, mode, code: str, pax: object, parentFigure=None
    ):
        """Plot some summary data.

        Args:
            datatype (str): type of data to summarize - spike or iv
            mode (_type_):
            code (str): a string description of the data coding variables: genotype, Group or
            pax (object): plot axis to use
            parentFigure (plothelpers object, optional): figure that the axis belongs to. Defaults to None.

        Raises:
            ValueError: datatype argument not in [iv, spike]
            ValueError: _description_
            ValueError: _description_

        Returns:
            _type_: _description_
        """
        cprint("c", f"plot_summary_info: Datatype: {datatype:s}, with code: {code:s} and mode: {mode:s}")
        if datatype not in ["spike", "iv"]:
            raise ValueError(
                f"plot_summary_info: Data type not in [spike, iv], got <{datatype:s}>"
            )
        assert code in ["genotype", "code", "Group", "cell_expression", "expression_and_genotype"]


        df_accum = self._get_iv_info(mode)
        if datatype == "iv":
            use_measures = iv_measures

            paxx = paxesb
        if datatype == "spike":
            # # print("getting spike data")
            # df_accum = self._get_spike_info(mode, existing=df_accum)
            use_measures = spike_measures
            paxx = paxes

        print("Prior to grouping: ")
        self.print_census(df_accum)
        df_accum = self.set_grouping(df_accum)
        print("after grouping: ")
        print("plot summary info: df_accum = \n", df_accum.head())
        self.print_census(df_accum)

        code_by_param = {}  # dict of all the codes available for a parameter we measure
        df_accum_main = df_accum[~df_accum["grouping"].isin(self.exclude_codes)]
        group_codes = list(
            set(df_accum_main.grouping.values)
        )  # list of remaining group codes
        print("group codes: ", list(set(df_accum_main.grouping.values)))
        usecode = "grouping"  # or "code"
        # print("datatype: <", datatype, "> measures: ", use_measures)
        
        def inftona(row):
            for m in ['taum', 'tauh_bovera', 'tauh_tau', 'Rin', 'RMP']:
                if row[m] == np.inf:
                    row[m] = np.nan
            return row
        
        for param in use_measures:
            if param in ["FR_Slope", "Ibreak", "Irate"]:
                continue
            all_codes = {}
            # print("param: ", param, " usecode: ", usecode)
            df_accum_main = df_accum_main.apply(inftona, axis=1)
            # with pd.option_context("mode.use_inf_as_na", True):
            #     df_accum = df_accum_main.dropna(subset=[param, usecode], how="all")
            # # get all of the code values in the dataframe, from the selected column
            # print("ZZZ: ", df_accum)
            no_code = df_accum.iloc[0][usecode] == []

            if no_code:
                codes = []
            else:
                codes = df_accum[usecode]  # get the codes by variable
            code_by_param[param] = codes
            # print("code_by_param: ", code_by_param[param])
            # print("len codes: ", len(codes), len(code_by_param[param][0]), len(code_by_param[param][1]))
            if (
                param
                not in no_average
                # and len(codes) >= 1
                # and len(code_by_param[param][0]) > 1
                # and len(code_by_param[param][1]) > 1
            ):
                print("usecode, param: ", usecode, param)
                print("df_accum columns: ", df_accum.columns)
                group_1 = df_accum[df_accum[usecode] == group_codes[0]][param].values
                group_2 = df_accum[df_accum[usecode] == group_codes[1]][param].values
                print("param: ", param, " group sizes: ", len(group_1), len(group_2))
                group_1 = group_1[~np.isnan(group_1)]
                group_2 = group_2[~np.isnan(group_2)]
                group_1 = group_1[np.isfinite(group_1)]
                group_2 = group_2[np.isfinite(group_2)]
                if (
                    len(group_1) > 1 and len(group_2) > 1
                ):  # need at least 2 samples to calculate SD (really, more...)
                    print(f"\nParam: {param:<20s}:")
                    print(
                        "group1: ",
                        code_by_param[param][0],
                        len(group_1),
                        np.sum(~np.isnan(group_1)),
                        group_1,
                    )
                    print(
                        "group2: ",
                        code_by_param[param][1],
                        len(group_2),
                        np.sum(~np.isnan(group_2)),
                        group_2,
                    )
                    print("group ")
                    t, p = scipy.stats.ttest_ind(
                        group_1,
                        group_2,
                        equal_var=False,
                        nan_policy="omit",
                    )
                    print(f"        t={t:.3f}  p={p:.4f}")
                    print(
                        f"        A: mean={np.nanmean(group_1):8.3f} std={np.nanstd(group_1):8.3f}, N={self.nancount(group_1):d}"
                    )
                    print(
                        f"        B: mean={np.nanmean(group_2):8.3f} std={np.nanstd(group_2):8.3f}, N={self.nancount(group_2):d}"
                    )

                    # KW
                    if len(code_by_param.keys()) == 4:
                        group_3 = df_accum[df_accum[usecode] == code[3]][param].values
                        group_4 = df_accum[df_accum[usecode] == code[4]][param].values
                        if self.nancount(group_3) > 0 and self.nancount(group_4) > 0:
                            if len(group_3) > 0 and len(group_4) > 0:
                                s, p = scipy.stats.kruskal(
                                    group_1,
                                    group_2,
                                    group_3,
                                    group_4,
                                )
                                print(f"Krwukal-Wallis: H:{s:.6f}   p={p:.6f}\n")
                        elif len(code_by_param[param]) == 3:
                            group_3 = df_accum[df_accum[usecode] == code[3]][
                                param
                            ].values
                            if self.nancount(group_3) > 0:
                                s, p = scipy.stats.kruskal(
                                    group_1,
                                    group_2,
                                    group_3,
                                )
                                print(f"Krwukal-Wallis: H:{s:.6f}   p={p:.6f}\n")
                else:
                    if param not in no_average:
                        print(
                            "\nEmpty data set for at least one category for param = ",
                            param,
                        )
                print("=" * 80)

        iax = 0

        for i, measure in enumerate(use_measures):
            if measure is None or measure in no_average or len(codes) == 0:
                # print(
                #     "s    > Skipping plotting measure: ",
                #     measure,
                #     measure in no_average,
                #     len(codes),
                # )
                continue
            if measure in p_skip_plot:
                continue
            df_accum = df_accum_main.assign(grouping="")
            if self.group_mode == "genotype":
                df_accum = df_accum.apply(self.group_by_sex_and_genotype, axis=1)
            elif self.group_mode == "code":
                df_accum = df_accum.apply(self.group_by_code, axis=1)
            elif self.group_mode == "Group":
                df_accum = df_accum.apply(self.group_by_group, axis=1)
            elif self.group_mode == "cell_expression":
                df_accum = df_accum.apply(self.group_by_expression, axis=1)
            elif self.group_mode == "expression_and_genotype":
                df_accum = df_accum.apply(self.group_by_expression_and_genotype, axis=1)
            else:
                raise ValueError(
                    f"Grouping mode: {self.group_mode:s} did not match known values [genotype, code, Group]"
                )
            # print(df_accum.keys())

            yd = df_accum[measure].replace([np.inf], np.nan)
            x = pd.Series(df_accum['grouping'])
            sex = pd.Series(df_accum["sex"])
            # grouping = pd.Series(df_accum["Group"])
            # print("grouping: ", grouping)
            print("measure: ", measure)
            # print("x: ",  x)
            # print("y: ", yd)

            iasort = x.argsort()
            if np.all(np.isnan(yd)):
                iax += 1
                continue  # skip plot if no data
            groups = sorted(list(set(x)))
            # print("Groups: ", groups)
            dfm = pd.DataFrame({"Group": x, "measure": yd, "sex": sex})
            # print("dfm: \n", dfm.head())
            sns.violinplot(
                data=dfm,
                x="Group",
                y="measure",
                ax=pax[iax],
                hue="sex",
                order=groups,  # [code_by_param[param][0], code_by_param[param][1]],
                # palette=self.codecolors,
                inner=None,
                saturation=1.0,
            )
            mpl.setp(pax[iax].collections, alpha=0.3)
            # orders = sorted([code_by_param[param][0], code_by_param[param][1]])
            # sns.swarmplot(data=dfm, x='Group', y='measure', ax=pax[i], order=['A', 'B'], hue="Group", palette=self.codecolors, edgecolor='grey', size=2.5)
            # for sex, marker, edgecolor, color in zip(['M', 'F'], ['s', 'o'], ['k', 'w'], ['k', 'b']):
            #     dfm_persex = dfm[dfm["sex"] == sex]
            sns.stripplot(
                data=dfm,
                x="Group",
                y="measure",
                ax=pax[iax],
                hue="sex",
                # marker=marker,
                # edgecolor=edgecolor,
                linewidth=0.5,
                dodge=True,
                order=groups,  # ['M', 'F'],
                # palette=[color]*2, # self.codecolors,
                # self.codecolors,
                size=2.5,
            )
            # sns.boxplot(data = dfm, x='Group', y="measure",  ax=pax[i], palette=self.codecolors, orient='v', width=0.5, saturation=1.0)
            pax[iax].set_title(measure, fontsize=8)  # .replace('_', '\_'))
            pax[iax].set_ylabel(ylabels[measure], fontsize=7)
            # pax[iax].set_ylim(paxx[measure])
            pax[iax].tick_params(axis="x", labelsize=8, labelrotation=45.0)

            iax += 1
        return df_accum

    def plot_spike_info(self, mode, code="genotype", parentFigure=None):
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
            3,
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

        df_accum = self.plot_summary_info(
            datatype="spike", mode=mode, code=code, pax=pax, parentFigure=parentFigure
        )

        iax = 0
        for i, measure in enumerate(spike_measures):
            if measure in no_average:  # skip data that is not relevant
                continue
            if measure in p_skip_plot:  # skip measures that are not useful
                continue
            pax[iax].tick_params(axis="x", labelsize=8, labelrotation=45.0)
            if measure == "AdaptRatio":
                if pax[iax].get_legend() is not None:
                    mpl.setp(pax[iax].get_legend().get_texts(), fontsize="5")
                    pax[iax].legend(bbox_to_anchor=(1.02, 1.0))
            else:
                if pax[iax].get_legend() is not None:
                    pax[iax].get_legend().remove()
            iax += 1

        self.spike_dataframe = df_accum
        return self.Pspikes

    def _get_iv_parameter(
        self,
        measure: str,
        cellmeas: dict,
        accumulator: dict,
        func: object = np.nanmean,
        dfiv: object = None,
        dspk: object = None,
    ):
        """Get the IV values averaged (or smallest) from one protocol/measurement in one cell
        accumulate the measures in the accumulator (dict of measures), using lists.

        Args:
            measure: the name of the measure to be stored
            accumulator (dict): _description_
            cellmeas (_type_): _description_
            func (function object): name of function to use on values (np.nanmean, np.nanmin)
        Returns:
            updated accumulator
        """
        if measure not in list(cellmeas.keys()):
            accumulator[measure].append(np.nan)
            return accumulator
        try:
            if pd.isnull(cellmeas[measure]):
                accumulator[measure].append(np.nan)
            else:
                accumulator[measure].append(cellmeas[measure])
        except:
            print("failed to get measure in dataframe : ", df)
            raise ValueError()
        return accumulator

    def _get_protocol_spike_data2(self, spike_dict):
        """_get_protocol_spike_data2 get a subset of measures

        Parameters
        ----------
        spike_dict : _type_
            _description_

        Returns
        -------
        _type_
            _description_
        """
        meas = ['FiringRate_1p5T', 'AdaptRatio', 'AHP_Trough', 'AHP_Depth', 'AP1_HalfWidth', 'AP1_Latency']
        sp_dict = {measure: np.nan for measure in meas}
        for m in meas:
            if m in spike_dict.keys():
                sp_dict[m] = spike_dict[m]
            else:
                sp_dict[m] = np.nan
        return sp_dict, meas

    
    def _get_protocol_data(
        self,
        dspk: dict = None,
        dfiv: dict = None,
        proto: str = None,
        Cell_ID: str = None,
        Animal_ID: str = None,
        Group: str = None,
        code: str = None,
        genotype: str = None,
        cell_expression: str = None,
        grouping: str=None,
        age: str = None,
        sex: str = None,
        day: str = None,
        accumulator: dict = None,
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
        cellmeas = dfiv[proto]
        m_dspk, sp_measures = self._get_protocol_spike_data2(spike_dict = dspk[proto])

        cellmeas["age"] = age
        cellmeas["code"] = code
        cellmeas["sex"] = sex
        cellmeas["genotype"] = genotype
        cellmeas["cell_expression"] = cell_expression
        cellmeas["grouping"] = grouping
        cellmeas["Date"] = day
        tracekeys = list(cellmeas.keys())
        for m in iv_measures:
            # print("get iv: m: ", m)
            if m not in no_average:
                accumulator = self._get_iv_parameter(
                    measure=m,
                    cellmeas=cellmeas,
                    accumulator=accumulator,
                    func=np.nanmean,
                    dspk=m_dspk,
                    dfiv=dfiv,
                )

            elif m in no_average:
                if m == "Cell_ID":
                    accumulator[m].append(Cell_ID)
                if m == "Date":
                    accumulator[m].append(day)
                if m == "Animal_ID":
                    accumulator[m].append(Animal_ID)
                if m == "sex":
                    accumulator[m].append(sex)
                if m == "age":
                    accumulator[m].append(age)
                if m == "Group":
                    accumulator[m].append(Group)
                if m == "code":
                    accumulator[m].append(code)
                if m == "genotype":
                    accumulator[m].append(genotype)
                if m == "cell_expression":
                    accumulator[m].append(cell_expression)
                if m == "grouping":
                    accumulator[m].append(grouping)
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
        # print("get_iv_info")
        Accumulator = pd.DataFrame(iv_measures)
        self.dfs = self.dfs.rename(columns={"animal identifier": "Animal_ID"})
        # print("dfs head: \n", self.dfs.head())
        # print("dfs columns: ", self.dfs.columns)
        # print(self.dfs.date)
        dates = self.dfs.date.unique()
        # print(self.dfs.head())
        # print("dates: ", dates)
        # for each mouse (day) :: Observational Unit AND Experimental Unit
        # The observational unit is the cell.
        # the Experimetnal unit is the day (perhaps it should be the "Animal_ID")

        OU_accumulator = []
        protocol_list = []  # protocol includes full path to data
        for day in dates:
            Day_accumulator = []

            for idx in self.dfs.index[
                self.dfs["date"] == day
            ]:  # for each cell in that day: Observational UNIT

                Cell_accumulator = self.make_emptydict(iv_measures)
                n_protocols = 0
                Cell_ID = DBTOOLS.make_cell_ID(self.dfs, idx)
                code = self.dfs.iloc[idx]['grouping']
                # print("code: ", code)
                # print("self.dfs columns (1262): ", self.dfs.columns)

                if code == " " or pd.isnull(code):
                    cprint("m", f"giv: Empty code for cell: {Cell_ID:s}, with group mode: {self.group_mode:s}")
                    continue
                sex = self.dfs.iloc[idx]["sex"]
                if "genotype" in self.dfs.columns:
                    genotype = self.dfs.iloc[idx]["genotype"]
                else:
                    genotype = "U"
                if "cell_expression" in self.dfs.columns:
                    cell_expression = self.dfs.iloc[idx]["cell_expression"]
                else:
                    cell_expression = "ND"
                if "Animal_ID" in self.dfs.columns:
                    Animal_ID = self.dfs.iloc[idx]["Animal_ID"]
                else:
                    Animal_ID = "ND"
                Group = self.dfs.iloc[idx]["Group"]
                grouping = self.dfs.iloc[idx]["grouping"]
                if 'age' in self.dfs.columns:
                    age = self.dfs.iloc[idx]["age"]
                elif "age_x" in self.dfs.columns:
                    age = self.dfs.iloc[idx]["age_x"]
                elif "Age" in self.dfs.columns:
                    age = self.dfs.iloc[idx]["Age"]
                else:
                    age = "ND"

                ages = self.make_emptydict(iv_measures)
                if "Spikes" in self.dfs.columns:
                    dspk = self.dfs.iloc[idx]["Spikes"]  # get the Spikes dict
                else:
                    dspk = []
                if "IV" in self.dfs.columns:
                    dfi = self.dfs.iloc[idx]["IV"]
                else:
                    dfi = []

                day_label = self.dfs.iloc[idx]["date"][:-4]

                if pd.isnull(dspk) or len(dspk) == 0:  # nothing in the Spikes column
                    # try looking for spikes from the analysis .pkl file
                    # make a file name:
                    dx = DBTOOLS.get_pickled_cell_data(self.dfs, idx, analyzed_datapath=self.analyzed_datapath)
                    if dx is None:
                        continue
                
                    dspk = dx["Spikes"]
                    dfi = dx["IV"]

                for proto in dspk.keys():  # protocol name includes day/slice/cell/protocol
                    if proto not in protocol_list:  # protect against double counting
                        protocol_list.append(proto)
                    else:
                        continue
                    cprint("c", f"        Cell: {Cell_ID:s} iv Protocol: {str(Path(proto).name):s}")

                    n_protocols += 1
                    Cell_accumulator = self._get_protocol_data(
                        dspk,
                        dfi,
                        proto,
                        Cell_ID=Cell_ID,
                        Animal_ID=Animal_ID,
                        sex=sex,
                        age=age,
                        code=code,
                        Group=Group,
                        genotype=genotype,
                        cell_expression=cell_expression,
                        grouping=grouping,
                        day=day,
                        accumulator=Cell_accumulator,
                    )

                if n_protocols > 0:
                    Cell_means = self._get_cell_means(
                        Cell_accumulator,
                        iv_measures,
                    )  # get mean values for the cell
                    OU_accumulator.append(Cell_means)
        OU = pd.DataFrame(OU_accumulator)
        print("OU: ", OU.columns)
        return OU

    def plot_IV_info(self, mode, code: str, parentFigure=None):
        """
        Mode can be OU for observational unit (CELL) or EU for experimental Unit
        (Animal)
        """

        modestring = "Unit = Experiment (animal)"
        if mode == "OU":
            modestring = "Unit = Observational (cell)"
        cprint("c", f"Accumulating by mode: {modestring:s}")

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

        df_accum = self.plot_summary_info(
            datatype="iv", mode=mode, code=code, pax=paxb, parentFigure=parentFigure
        )

        iax = 0
        for i, measure in enumerate(iv_measures):
            if measure in no_average:
                continue

            if measure == "Rin":
                if paxb[iax].get_legend() is not None:
                    mpl.setp(paxb[iax].get_legend().get_texts(), fontsize="6")
                    paxb[iax].legend(bbox_to_anchor=(-0.35, 1.0))
                    # handles, labels = paxb[iax].get_legend_handles_labels()
                    # print(handles)
                    # # print("labels: ", labels)
                    # handles = [handles[1], handles[3]]
                    # labels = [labels[2], labels[3]]
                    # paxb[iax].legend(handles, labels, bbox_to_anchor=(-0.35, 1.0), fontsize=7)
                    # ap1l = d[u]['AP1_Latency'] ap1hw = d[u]['AP1_HalfWidth']
                    # apthr = d[u]['FiringRate_1p5T']
            else:
                paxb[iax].get_legend().remove()
            iax += 1

        self.iv_dataframe = df_accum
        return self.Pb

    def get_age(self, idx):
        age = self.dfs.iloc[idx]["age"]
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
    # code_df = pd.read_excel(code_file, sheet_name=code_sheet)
    # # be sure types are correct
    # for coln in ["age", "Animal_ID", "Group", "sex", "cell_expression"]:
    #     code_df[coln] = code_df[coln].astype(str)
    # code_df = code_df.drop("age", axis="columns")
    # dfm = pd.merge(
    #     df, code_df, left_on="shortdate", right_on="Date", how="left"
    # )  # codes are by date only
    dfm = df
    dfm["protocol"] = ""
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
                if col != "protocol":
                    data[col] = dfm[col][index]
                else:
                    data["protocol"] = prot  # add the protocol
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
                raise ValueError("Failed to parse age")
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
