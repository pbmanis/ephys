"""Plot summaries of spike and basic electrophys properties of cells.
Does stats at the end.
"""

import datetime
import pprint
from pathlib import Path
import re
from typing import Optional
import textwrap
import dateutil.parser as DUP
import matplotlib.pyplot as mpl
import concurrent.futures
import numpy as np
import pandas as pd
import pylibrary.plotting.plothelpers as PH
import scikit_posthocs
import seaborn as sns
import statsmodels.api as sa
import statsmodels.formula.api as sfa
import statsmodels.formula.api as smf
import scipy.stats
import pyqtgraph.multiprocess as MP
from pylibrary.tools import cprint
from pyqtgraph.Qt.QtCore import QObject
from statsmodels.stats.multitest import multipletests

from ephys.ephys_analysis import spike_analysis
from ephys.tools.get_computer import get_computer
import ephys.tools.show_combined_datafile as SCD
from ephys.gui import data_table_functions
from ephys.tools import filter_data, fitting, utilities

PP = pprint.PrettyPrinter()

UTIL = utilities.Utility()
FUNCS = data_table_functions.Functions()
CP = cprint.cprint
Fitter = fitting.Fitting()

# Calling these functions from the GUI with "concurrent.futures"
# prevents multiprocessing from failing due to "fork" issues that
# appear once matplotlib is used. (does not happen with plots generated
# by pyqtgraph)

# These functions are wrappers for the PlotSpikeInfo class methods,
# and were orginally in data_tables.py, associated with the
# parametertree parsing.


def concurrent_categorical_data_plotting(
    filename: str,
    mode: str,  # continous or categorical
    plot_title: str = "My Title",
    parameters: dict = None,
    data_class: str = "spike_measures",  # what kind of data to plot
    representation: str = "bestRs",
    picker_active: bool = False,
    infobox: dict = None,
):
    assert mode in ["categorical", "continuous"]
    # unpack parameters:
    header = parameters["header"]
    experiment = parameters["experiment"]
    datasummary = parameters["datasummary"]
    group_by = parameters["group_by"]
    colors = parameters["colors"]
    hue_category = parameters["hue_category"]
    pick_display_function = parameters["pick_display_function"]

    PSI_ = PlotSpikeInfo(
        datasummary,
        experiment,
        pick_display=picker_active,
        pick_display_function=pick_display_function,
        representation=representation,
    )
    df = PSI_.preload(filename)
    if mode == "categorical":
        print("categorical")

        (
            cc_plot,
            picker_funcs1,
        ) = PSI_.summary_plot_spike_parameters_categorical(
            df,
            xname=group_by,
            hue_category=hue_category,
            plot_order=experiment["plot_order"][group_by],
            measures=experiment[data_class],
            colors=colors,
            enable_picking=picker_active,
            representation=representation,
        )
    elif mode == "continuous":
        df = PSI_.preload(filename)
        (
            cc_plot,
            picker_funcs1,
        ) = PSI_.summary_plot_spike_parameters_continuous(
            df,
            xname=group_by,
            measures=experiment[data_class],
            representation=representation,
        )
    cc_plot.figure_handle.suptitle(plot_title, fontweight="bold", fontsize=18)
    picked_cellid = cc_plot.figure_handle.canvas.mpl_connect(  # override the one in plot_spike_info
        "pick_event",
        lambda event: PSI_.pick_handler(event, picker_funcs1),
    )

    cc_plot.figure_handle.text(
        infobox["x"],
        infobox["y"],
        header,
        fontdict={
            "fontsize": infobox["fontsize"],
            "fontstyle": "normal",
            "font": "Courier",
        },
        verticalalignment="top",
        horizontalalignment="left",
    )
    cc_plot.figure_handle.show()
    mpl.show()
    return cc_plot


def concurrent_selected_fidata_data_plotting(
    filename: str,
    parameters: dict = None,
    picker_active: bool = False,
    infobox: dict = None,
):
    print("Unpacking concurrent selected...")

    # unpack parameters:
    header = parameters["header"]
    experiment = parameters["experiment"]
    datasummary = parameters["datasummary"]
    assembleddata = parameters["assembleddata"]
    group_by = parameters["group_by"]
    plot_order = parameters["plot_order"]
    colors = parameters["colors"]
    hue_category = parameters["hue_category"]
    pick_display_function = parameters["pick_display_function"]

    P2 = FUNCS.get_selected_cell_data_FI(
        experiment=experiment,
        assembleddata=assembleddata,
    )

    # P2.figure_handle.suptitle("Firing Rate", fontweight="bold", fontsize=18)
    # picked_cellid = P2.figure_handle.canvas.mpl_connect(  # override the one in plot_spike_info
    #     "pick_event",
    #     lambda event: PSI_.pick_handler(event, picker_funcs2),
    # )
    if P2 is not None:
        P2.figure_handle.text(
            infobox["x"],
            infobox["y"],
            header,
            fontdict={
                "fontsize": infobox["fontsize"],
                "fontstyle": "normal",
                "font": "helvetica",
            },
            verticalalignment="top",
            horizontalalignment="left",
        )

        P2.figure_handle.show()
    return P2


after = "2000.01.01"
after_parsed = datetime.datetime.timestamp(DUP.parse(after))


def set_ylims(experiment):
    """set_ylims adjust the ylimits for a plot

    Parameters
    ----------
    experiment : _type_
        _description_

    Returns
    -------
    _type_
        _description_

    Raises
    ------
    ValueError
        _description_
    """
    if experiment is not None and "ylims" in experiment.keys():
        ylims = {}
        # the key may be a list of cell types all with the same limits
        # CP("r", "setting ylims for cell types")
        for limit_group in experiment["ylims"].keys():

            for ct in experiment["ylims"][limit_group]["celltypes"]:
                if ct not in ylims.keys():
                    ylims[ct] = experiment["ylims"][limit_group]
                else:
                    raise ValueError(
                        f"Cell type {ct!s} already in ylims - check the configuration file 'ylims' entry"
                    )
        return ylims
    else:
        ylims_pyr = {
            "dvdt_rising": [0, 800],
            "dvdt_falling": [0, 800],
            "AP_HW": [0, 1000],
            "AP_thr_V": [-75, 0],
            "AHP_depth_V": [None, None],
            "AdaptRatio": [0, 2],
            "FISlope": [0, 1500],
            "maxHillSlope": [0, 5000],
            "I_maxHillSlope": [0, 1.0],
            "FIMax_1": [0, 800],
            "FIMax_4": [0, 800],
            "taum": [0, None],
            "taum_averaged": [0, None],
            "RMP": [-80, -40],
            "Rin": [0, 250],
        }
        ylims_tv = {
            "dvdt_rising": [0, 800],
            "dvdt_falling": [0, 800],
            "AP_HW": [0, 1000],
            "AHP_depth_V": [None, None],
            "AP_thr_V": [-75, 0],
            "AdaptRatio": [0, 2],
            "FISlope": [0, 1500],
            "maxHillSlope": [0, None],
            "I_maxHillSlope": [0, 0.5],
            "FIMax_1": [0, 800],
            "FIMax_4": [0, 800],
            "taum": [0, None],
            "taum_averaged": [0, None],
            "RMP": [-80, -40],
            "Rin": [0, 400],
        }
        ylims_cw = {
            "dvdt_rising": [0, 800],
            "dvdt_falling": [0, 800],
            "AP_HW": [0, 1000],
            "AHP_depth_V": [None, None],
            "AP_thr_V": [-75, 0],
            "AdaptRatio": [0, 8],
            "FISlope": [0, 250],
            "maxHillSlope": [0, 5000],
            "I_maxHillSlope": [0, 0.5],
            "FIMax_1": [0, 800],
            "FIMax_4": [0, 800],
            "taum": [0, None],
            "taum_averaged": [0, None],
            "RMP": [-80, -40],
            "Rin": [0, 200],
        }

        ylims_giant = {
            "dvdt_rising": [0, 800],
            "dvdt_falling": [0, 800],
            "AP_HW": [0, 1000],
            "AHP_depth_V": [-75, -40],
            "AP_thr_V": [-75, 0],
            "AdaptRatio": [0, 2],
            "FISlope": [0, 1000],
            "maxHillSlope": [0, 1000],
            "I_maxHillSlope": [0, 1.0],
            "FIMax_1": [0, 800],
            "FIMax_4": [0, 800],
            "taum": [0, None],
            "taum_averaged": [0, None],
            "RMP": [-80, -40],
            "Rin": [0, 400],
        }
        ylims_giant_maybe = {
            "dvdt_rising": [0, 800],
            "dvdt_falling": [0, 800],
            "AP_HW": [0, 1000],
            "AHP_depth_V": [-75, -40],
            "AP_thr_V": [-75, 0],
            "AdaptRatio": [0, 2],
            "FISlope": [0, 1000],
            "maxHillSlope": [0, 1000],
            "I_maxHillSlope": [0, 1.0],
            "FIMax_1": [0, 800],
            "FIMax_4": [0, 800],
            "taum": [0, None],
            "taum_averaged": [0, None],
            "RMP": [-80, -40],
            "Rin": [0, 400],
        }

        ylims_bushy = {
            "dvdt_rising": [0, 200],
            "dvdt_falling": [0, 200],
            "AP_HW": [0, 3.0],
            "AHP_depth_V": [-75, -40],
            "AP_thr_V": [-75, 0],
            "AdaptRatio": [0, 2],
            "FISlope": [0, 1000],
            "maxHillSlope": [0, 1000],
            "I_maxHillSlope": [0, 1.0],
            "FIMax_1": [0, None],
            "FIMax_4": [0, None],
            "taum": [0, None],
            "taum_averaged": [0, None],
            "RMP": [-80, -40],
            "Rin": [0, 400],
        }

        ylims_tstellate = {
            "dvdt_rising": [0, 400],
            "dvdt_falling": [0, 400],
            "AP_HW": [0, 3.0],
            "AHP_depth_V": [-75, -40],
            "AP_thr_V": [-75, 0],
            "AdaptRatio": [0, 2],
            "FISlope": [0, 1000],
            "maxHillSlope": [0, 1000],
            "I_maxHillSlope": [0, 1.0],
            "FIMax_1": [0, None],
            "FIMax_4": [0, None],
            "taum": [0, None],
            "taum_averaged": [0, None],
            "RMP": [-80, -40],
            "Rin": [0, 400],
        }

        ylims_dstellate = {
            "dvdt_rising": [0, 400],
            "dvdt_falling": [0, 400],
            "AP_HW": [0, 3.0],
            "AHP_depth_V": [-75, -40],
            "AP_thr_V": [-75, 0],
            "AdaptRatio": [0, 2],
            "FISlope": [0, 1000],
            "maxHillSlope": [0, 1000],
            "I_maxHillSlope": [0, 1.0],
            "FIMax_1": [0, None],
            "FIMax_4": [0, None],
            "taum": [0, None],
            "taum_averaged": [0, None],
            "RMP": [-80, -40],
            "Rin": [0, 400],
        }

        ylims_octopus = {
            "dvdt_rising": [0, 200],
            "dvdt_falling": [0, 200],
            "AP_HW": [0, 3.0],
            "AHP_depth_V": [-75, -40],
            "AP_thr_V": [-75, 0],
            "AdaptRatio": [0, 2],
            "FISlope": [0, 1000],
            "maxHillSlope": [0, 1000],
            "I_maxHillSlope": [0, 1.0],
            "FIMax_1": [0, None],
            "FIMax_4": [0, None],
            "taum": [0, None],
            "taum_averaged": [0, None],
            "RMP": [-80, -40],
            "Rin": [0, 400],
        }
        ylims_default = {
            "dvdt_rising": [0, 1000],
            "dvdt_falling": [0, 1000],
            "AP_HW": [0, 2000],
            "AP_thr_V": [-75, 0],
            "AHP_depth_V": [-85, -40],
            "AdaptRatio": [0, 5],
            "FISlope": [0, 1500],
            "maxHillSlope": [0, None],
            "I_maxHillSlope": [0, 1000],
            "FIMax_1": [0, 800],
            "FIMax_4": [0, 800],
            "taum": [0, None],
            "taum_averaged": [0, None],
            "RMP": [-80, -40],
            "Rin": [0, None],
        }

        ylims = {
            "pyramidal": ylims_pyr,
            "tuberculoventral": ylims_tv,
            "cartwheel": ylims_cw,
            "giant": ylims_giant,
            "giant_maybe": ylims_giant_maybe,
            "bushy": ylims_bushy,
            "t-stellate": ylims_tstellate,
            "d-stellate": ylims_dstellate,
            "octopus": ylims_octopus,
            "default": ylims_default,  # a default set when "default" is specified in the configuration
        }
        return ylims


class Picker(QObject):
    def __init__(self, space=None, data=None, axis=None):
        assert space in [None, 2, 3]
        self.space = space  # dimensions of plot (2 or 3)
        self.setData(data, axis)
        self.annotateLabel = None

    def setData(self, data, axis=None):
        self.data = data
        self.axis = axis

    def setAction(self, action):
        # action is a subroutine that should be called when the
        # action will be called as self.action(closestIndex)
        self.action = action

    def pickEvent(self, event, ax):
        """Event that is triggered when mouse is clicked."""
        # print("event index: ", event.ind)
        # print(dir(event.mouseevent))
        # print(event.mouseevent.inaxes == ax)
        # print(ax == self.axis)
        # print("psi.Picker pickEvent: ", self.data.iloc[event.ind])  # find the matching data.
        return


def get_plot_order(experiment):
    """get_plot_order get the order of the groups to plot

    Parameters
    ----------
    experiment : dict
        experiment dictionary

    Returns
    -------
    list
        list of groups in order to plot
    """
    if "plot_order" in experiment.keys():
        plot_order = experiment["plot_order"]
    else:
        raise ValueError("No Plot Order is defined in the configuration file")
    return plot_order


def get_protodurs(experiment):
    if "protocol_durations" in experiment.keys():
        protodurs = experiment["protocol_durations"]
    else:
        raise ValueError("No protocol durations are defined in the configuration file")
    return protodurs


def get_plot_colors(experiment):
    """get_plot_colors get the colors to use for the groups

    Parameters
    ----------
    experiment : dict
        experiment dictionary

    Returns
    -------
    dict
        dictionary of colors
    """
    if "plot_colors" in experiment.keys():
        plot_colors = experiment["plot_colors"]
    else:
        raise ValueError("No Plot Colors are defined in the configuration file")
    return plot_colors


def rename_groups(row, experiment):
    # print("renaming row group: ", row.Group)
    if row.Group in list(experiment["group_map"].keys()):
        row.groupname = experiment["group_map"][row.Group]
    else:
        row.groupname = np.nan  # deassign the group
    return row.groupname


pd.set_option("display.max_columns", 40)


def get_datasummary(experiment):
    datasummary = Path(
        experiment["analyzeddatapath"],
        experiment["directory"],
        experiment["datasummaryFilename"],
    )
    if not datasummary.exists():
        raise ValueError(f"Data summary file {datasummary!s} does not exist")
    df_summary = pd.read_pickle(datasummary)
    df_summary.rename(
        {"Subject": "animal_identifier", "animal identifier": "animal_identifier"},
        axis=1,
        inplace=True,
    )
    print(df_summary.columns)
    print(df_summary.cell_type.unique())
    return df_summary


def setup(experiment):
    excelsheet = Path(
        experiment["analyzeddatapath"],
        experiment["directory"],
        experiment["datasummaryFilename"],
    )
    analysis_cell_types = experiment["celltypes"]
    adddata = Path(
        experiment["analyzeddatapath"],
        experiment["directory"],
        experiment["adddata"],
    )

    return excelsheet, analysis_cell_types, adddata


cols = [
    "ID",
    "Group",
    "age",
    "weight",
    "sex",
    "Date",
    "cell_id",
    "cell_type",
    "protocol",
    "holding",
    "RMP",
    "RMP_SD",
    "Rin",
    "taum",
    "dvdt_rising",
    "dvdt_falling",
    "AP1_Latency",
    "AP_peak_V",
    "AP_peak_T",
    "AP_thr_V",
    "AP_thr_T",
    "AP_HW",
    "AP15Rate",
    "AdaptRatio",
    "AHP_trough_V",
    "AHP_trough_T",
    "AHP_depth_V",
    "tauh",
    "Gh",
    "FiringRate",
    "FI_Curve",
]

datacols = [
    "holding",
    "RMP",
    "RMP_SD",
    "Rin",
    "taum",
    "dvdt_rising",
    "dvdt_falling",
    "current",
    "AP_thr_V",
    "AP_thr_T",
    "AP_HW",
    "AP15Rate",
    "AdaptRatio",
    "AHP_trough_V",
    "AHP_trough_T",
    "AHP_depth_V",
    "tauh",
    "Gh",
    "FiringRate",
]


"""
Each protocol has:
"Spikes: ", dict_keys(['FI_Growth', 'AdaptRatio', 'FI_Curve', 
    'FiringRate', 'AP1_Latency', 'AP1_HalfWidth', 
'AP1_HalfWidth_interpolated', 'AP2_Latency', 'AP2_HalfWidth', 
    'AP2_HalfWidth_interpolated',
    'FiringRate_1p5T', 'AHP_depth_V', 'AHP_Trough', 'spikes', 'iHold', 
    'pulseDuration', 'baseline_spikes', 'poststimulus_spikes',
    "LowestCurrentSpike])

"IV": dict_keys(['holding', 'WCComp', 'CCComp', 'BridgeAdjust',
     'RMP', 'RMP_SD', 'RMPs',
'Irmp', 'taum', 'taupars', 'taufunc', 'Rin', 'Rin_peak', 
    'tauh_tau', 'tauh_bovera', 'tauh_Gh', 'tauh_vss'])

individual spike data:
spike data:  dict_keys(['trace', 'AP_number', 'dvdt', 'V', 'Vtime',
     'pulseDuration', 'tstart', 'tend', 
'AP_beginIndex', 'AP_peakIndex', 'AP_endIndex', 'peak_T', 'peak_V', 'AP_latency', 'AP_begin_V', 
'halfwidth', 'halfwidth_V', 'halfwidth_up', 'halfwidth_down', 'halfwidth_interpolated', 
'left_halfwidth_T', 'left_halfwidth_V', 'right_halfwidth_T', 'right_halfwidth_V', 
'trough_T', 'trough_V', 'peaktotrough', 'current', 'iHold', 'dvdt_rising', 'dvdt_falling'])

"""


def make_cell_id(row):
    sliceno = int(row["slice_slice"][-3:])
    cellno = int(row["cell_cell"][-3:])
    cell_id = f"{row['date']:s}_S{sliceno:d}C{cellno:d}"
    row["cell_id"] = cell_id
    return row


def get_age(age_value):
    if isinstance(age_value, pd.Series):
        age = age_value.values[0]
    else:
        age = age_value
    if isinstance(age, (float, np.float64)):
        age = int(age)
    elif isinstance(age, str):
        age = int("".join(filter(str.isdigit, age)))
        if isinstance(age, float):
            age = int(age)
    else:
        raise ValueError(f"age is not a pd.Series, float or string: {age!s}")
    return age


def numeric_age(row):
    """numeric_age convert age to numeric for pandas row.apply

    Parameters
    ----------
    row : pd.row_

    Returns
    -------
    value for row entry
    """
    if isinstance(row.age, float):
        return row.age
    row.age = int("".join(filter(str.isdigit, row.age)))
    return float(row.age)


def make_datetime_date(row, colname="date"):
    if colname == "date" and "Date" in row.keys():
        colname = "Date"
    if pd.isnull(row[colname]) or row[colname] == "nan":
        row.shortdate = 0
        return row.shortdate

    date = str(Path(row[colname]).name)
    date = date.split("_", maxsplit=1)[0]
    shortdate = datetime.datetime.strptime(date, "%Y.%m.%d")
    shortdate = datetime.datetime.timestamp(shortdate)
    st = datetime.datetime.timestamp(
        datetime.datetime.strptime("1970-01-01T00:00:00", "%Y-%m-%dT%H:%M:%S")
    )
    row.shortdate = shortdate - st
    if pd.isnull(row.shortdate):
        raise ValueError("row.shortdate is null ... in make_datetime_date")

    return row.shortdate


class PlotSpikeInfo(QObject):
    def __init__(
        self,
        dataset,
        experiment,
        pick_display=False,
        pick_display_function=None,
        representation: str = "all",
    ):
        self.set_experiment(dataset, experiment)
        self.transforms = {
            # "maxHillSlope": np.log10,
        }
        self.pick_display = pick_display
        self.pick_display_function = pick_display_function
        self.representation = representation

    def set_experiment(self, dataset, experiment):
        """set_experiment Update the selected dataset and experiment so we
        access the right files and directories.

        Parameters
        ----------
        dataset : _type_
            _description_
        experiment : _type_
            _description_
        """
        self.dataset = dataset
        self.experiment = experiment
        # print("experiment: ", experiment)
        self.ylims = set_ylims(self.experiment)

    def set_pick_display_function(self, function):
        self.pick_display_function = function

    def read_coding_file(self, df, coding_file, coding_sheet, level="date"):
        df3 = pd.read_excel(coding_file, sheet_name=coding_sheet)
        # print(df3.head())
        for index in df.index:
            row = df.loc[index]
            if pd.isnull(row.date):
                continue
            if "coding_name" in self.experiment.keys():
                coding_name = self.experiment["coding_name"]
            else:
                coding_name = "Group"
            if row.date in df3.date.values:
                if "sex" in df3.columns:  # update sex? Should be in main table.
                    df.loc[index, "sex"] = df3[df3.date == row.date].sex.astype(str).values[0]
                if "cell_expression" in df3.columns:
                    df.loc[index, "cell_expressoin"] = (
                        df3[df3.date == row.date].cell_expression.astype(str).values[0]
                    )
                if level == "date":
                    df.loc[index, "Group"] = (
                        df3[df3.date == row.date][coding_name].astype(str).values[0]
                    )
                elif level == "slice":
                    mask = (df3.date == row.date) & (df3.slice_slice == row.slice_slice)
                    df.loc[index, "Group"] = df3[mask][coding_name].astype(str).values[0]
                elif level == "cell":
                    mask = (
                        (df3.date == row.date)
                        & (df3.slice_slice == row.slice_slice)
                        & (df3.cell_cell == row.cell_cell)
                    )
                    print("mask: ", mask)
                    print("df3.date: ", row.date)
                    print("df3.slice_slice: ", row.slice_slice)
                    print("df3.cell_cell: ", row.cell_cell)
                    print("coding name: ", coding_name)
                    print("Mask: ", df3[mask][coding_name].astype(str))
                    df.loc[index, "Group"] = df3[mask][coding_name].astype(str).values[0]
            else:
                # print("Assigning nan to : ", df.loc[index].cell_id)
                df.loc[index, "Group"] = np.nan
        # print("result: ")
        # print(df["Group"].unique())
        # print(len(df))
        # print(len(df[df.Group.isnull()]))
        # print(len(df[df.Group.isin(["ECIC"])]))
        # print(len(df[df.Group.isin(["CNIC"])]))
        # exit()
        return df

    def combine_summary_and_coding(
        self,
        # excelsheet,
        # adddata=None,
        df_summary: pd.DataFrame,
        coding_file: Optional[str] = None,
        coding_sheet: Optional[str] = "Sheet1",
        coding_level: Optional[str] = "date",
        exclude_unimportant=False,
    ):
        """combine_summary_and_coding: combine the summary data with the coding file

        Parameters
        ----------
        excelsheet : string or Path
            excel filename to read to get the data to plot. This is the sheet generated by process_spike_info
        adddata : _type_, optional
            analysis result data to merge with main excel sheet, by default None
        coding_file: string or Path, optional
            excel file with coding information (full date, plus Group and sex columns as needed),
            by default None.
            Do not specify this if the Group column in the excelsheet already has valid groups,
            and the sex is already specified as well.
        analysis_cell_types : list, optional
            a list of cell type names, to specify which cell types will be analyzed, by default []

        Returns
        -------
        _type_
            _description_
        """
        CP("r", "\nReading intermediate result files")

        # if Path(excelsheet).suffix == ".pkl":  # need to respecify as excel sheet
        #     excelsheet = Path(excelsheet).with_suffix(".xlsx")
        # print(f"    Excelsheet (from process_spike_info): {excelsheet!s}")
        print("     coding_level: ", coding_level)
        # df = pd.read_excel(excelsheet)
        print("    # entries in summary sheet: ", len(df_summary))
        if "cell_id" not in list(df_summary.columns):
            df_summary.apply(make_cell_id, axis=1)

        # print(f"    Adddata in read_intermediate_result_files: {adddata!s}")
        if coding_file is not None:  # add coding from the coding file
            df = self.read_coding_file(df_summary, coding_file, coding_sheet, coding_level)
        else:
            df = df_summary
            df["Group"] = "Control"
        FD = filter_data.FilterDataset(df)
        if "remove_expression" not in self.experiment.keys():
            self.experiment["remove_expression"] = None
        df = FD.filter_data_entries(
            df,
            remove_groups=self.experiment["remove_groups"],
            remove_expression=self.experiment["remove_expression"],
            excludeIVs=self.experiment["excludeIVs"],
            exclude_internals=["cesium", "Cesium"],
            exclude_temperatures=["25C", "room temp"],
            exclude_unimportant=exclude_unimportant,
            verbose=True,
        )

        CP("m", "Finished reading files\n")
        return df

    def combine_by_cell(self, df, valid_protocols=None):
        """
        Rules for combining cells and pulling the data from the original analysis:
        1. Combine data from cells with the same ID
        2. Check the cell name and whether it fits the S00C00 or S1C1 format.
        3. When getting spike parameters, use a logical set of restrictions:
            a. Use only the first spike at the lowest current level that evoke spikes
                for AP HW, AP_thr_V, AP15Rate, AdaptRatio, AHP_trough_V, AHP_depth_V, AHP_trough_T
                This is in ['spikes']["LowestCurrentSpike"]
            b. Do not use traces that are above the spike firing rate turnover point (non-monotonic)

        """
        CP("y", "Combine by cell")

        df = df.apply(make_cell_id, axis=1)
        df.dropna(subset=["cell_id"], inplace=True)
        df.rename(columns={"sex_x": "sex"}, inplace=True)
        if self.experiment["celltypes"] != ["all"]:
            df = df[df.cell_type.isin(self.experiment["celltypes"])]
        df["shortdate"] = df.apply(
            make_datetime_date, colname="date", axis=1
        )  # make a short date as a datetime for sorting
        after_parsedts = after_parsed
        df = df[df["shortdate"] >= after_parsedts]
        cell_list = list(set(df.cell_id))

        cell_list = sorted(cell_list)
        dfdict = {}  # {col: [] for col in cols}
        df_new = pd.DataFrame.from_dict(dfdict)
        computer_name = get_computer()
        nworkers = self.experiment["NWORKERS"][computer_name]
        cells_to_do = [cell for cell in cell_list if cell is not None]

        # here we should check to see if cell has been done in the current file,
        # and remove it from the list.
        combined_file = Path(
            self.experiment["analyzeddatapath"],
            self.experiment["directory"],
            self.experiment["assembled_filename"],
        )
        # first be sure that we even have a combined file!
        if combined_file.is_file():
            already_done = pd.read_pickle(
                Path(
                    self.experiment["analyzeddatapath"],
                    self.experiment["directory"],
                    self.experiment["assembled_filename"],
                )
            )
            already_done = already_done.cell_id.unique()
        else:
            already_done = []
        # cells_to_do = [cell for cell in cells_to_do if cell not in already_done]
        # instrument up to to a limited set of the data for testing
        # The limit numbers refer to the IV data table.
        ilimit = None  # list(range(67, 78))
        tasks = 1
        if ilimit is None:
            limit = len(cells_to_do)
            tasks = range(limit)
        elif isinstance(ilimit, int):
            limit = min(ilimit, len(cells_to_do))
            tasks = range(limit)
        elif isinstance(ilimit, list):
            limit = ilimit
            tasks = limit

        result = [None] * len(tasks)
        results = dfdict
        parallel = True
        if parallel:
            with concurrent.futures.ProcessPoolExecutor(max_workers=nworkers) as executor:
                results = executor.map(
                    FUNCS.compute_FI_Fits,
                    [self.experiment] * len(tasks),
                    [df] * len(tasks),
                    [cell_list[i] for i in tasks],
                    [self.experiment["FI_protocols"]] * len(tasks),
                )
                for i, result in enumerate(results):
                    if result is not None:
                        df_new = pd.concat(
                            [df_new, pd.Series(result).to_frame().T], ignore_index=True
                        )
            # with MP.Parallelize(enumerate(tasks), results=results, workers=nworkers) as tasker:
            #     for i, x in tasker:
            #         result = FUNCS.compute_FI_Fits(
            #             self.experiment, df, cell_list[i], protodurs=self.experiment["FI_protocols"]
            #         )
            #         tasker.results[cell_list[i]] = result

            # for r in results:
            #     df_new = pd.concat([df_new, pd.Series(results[r]).to_frame().T], ignore_index=True)
        else:

            # do each cell in the database
            for icell, cell in enumerate(cell_list):
                print("icell: ", icell)
                if not isinstance(ilimit, list):
                    if ilimit is not None and icell > ilimit:
                        break
                elif isinstance(ilimit, list):
                    if icell not in ilimit:
                        continue
                if cell is None:
                    CP("r", f"    Cell # {icell:d} in the database is None")
                    continue
                CP(
                    "c", f"    Computing FI_Fits for cell: {cell:s}"
                )  # df[df.cell_id==cell].cell_type)
                datadict = FUNCS.compute_FI_Fits(
                    self.experiment, df, cell, protodurs=self.experiment["FI_protocols"]
                )
                if datadict is None:
                    print("    datadict is none for cell: ", cell)
                    continue
                df_new = pd.concat([df_new, pd.Series(datadict).to_frame().T], ignore_index=True)
        return df_new

    def to_1D(self, series):
        return pd.Series([x for _list in series for x in _list])

    def clean_alt_list(self, list_):
        list_ = list_.replace(", ", '","')
        list_ = list_.replace("[", '["')
        list_ = list_.replace("]", '"]')
        list_ = list_.replace("\n", "")
        return list_

    def print_for_prism(self, row, celltype="tuberculoventral"):
        if row.cell_type != celltype:
            return
        print("")
        print(row.cell_type)
        print(f"S:{row.Subject:s}")
        for i, x in enumerate(row.current):
            print(f"{x*1e9:.2f}  {row.spsec[i]:.1f}")
        print("")

    def fill_missing_groups(self, df, groups, celltype):
        """fill_missing_celltypes : add missing cell types in groups,
        to the data frame but with
        NaN values for all parameters

        Parameters
        ----------
        df : pandas dataframe
            _description_
        """
        return df
        present_groups = df.Group.unique()
        print("     Filling missing groups with dummy nan values")
        print("       Present Groups: ", present_groups)
        for group in groups:
            if group not in present_groups:
                df.loc[len(df)] = pd.Series(dtype="float64")
                df.loc[len(df) - 1, "Group"] = group
        return df

    def clear_missing_groups(self, row, data, replacement=None):
        """clear_missing_groups Remove groups that are nan or empty
        by replacing them with "Group = np.nan", unless
        "replacement" is not None, in which case the group
        is replaced with the value of "replacement"

        Parameters
        ----------
        row : _type_
            _description_
        data : _type_
            _description_
        replacement : str, optional
            _description_, by default "Ctl"

        Returns
        -------
        _type_
            _description_
        """
        # print("clear missing groups for : ", data, row[data])
        if "remove_groups" in self.experiment.keys():
            # print("config has groups to remove: ", self.experiment["remove_groups"])
            if self.experiment["remove_groups"] is not None:
                if row.Group in self.experiment["remove_groups"]:
                    print("row group: ", row.Group, " is in removables")
                    row[data] = np.nan
                    return row
        if pd.isnull(row[data]) or len(row[data]) == 0 or row[data] == 'nan':
            # print("row[data] is" , row[data])
            if replacement is not None:
                row[data] = replacement
            else:
                row[data] = np.nan
        return row

    def bar_pts(
        self,
        df,
        xname: str,  # x
        yname: str,  # variable
        celltype: str = "pyramidal",
        hue_category: Optional[str] = None,
        sign: float = 1.0,
        scale: float = 1.0,
        ax: mpl.axes = None,
        plot_order: Optional[list] = None,
        colors: Optional[dict] = None,
        enable_picking=True,
    ):
        """Graph a bar plot and a strip plot on top of each other

        Args:
            df (Pandas data frome): _description_
            groups: Which grouping to use for the categories
            yname (str): dataset name to plot
            celltype (str, optional): Cell type to plot. Defaults to "pyramidal".
            ax (object, optional): Axis to plot into. Defaults to None.
            sf (float, optional): Scale factor to multipy data by.

            Note:
            seaborn.
            Stripplot and boxplot palettes should be a dictionary mapping hue levels to matplotlib colors.

        """
        if celltype != "all":
            df_x = df[df["cell_type"] == celltype]
        else:
            df_x = df
        # print("dfx type 1: ", type(df_x))
        
        df_x = df_x.apply(self.apply_scale, axis=1, measure=yname, scale=sign * scale)
        # print("dfx type 2: ", type(df_x))
        if colors is None:  # set all to blue
            colors = {df_x[g]: "b" for g in plot_order}
        # df_x.dropna(subset=[groups], inplace=True)  # drop anything unassigned
        df_x[yname] = df_x[yname].astype(float)  # make sure values to be plotted are proper floats
        # print("dfx type 3: ", type(df_x))
        if df_x[yname].isnull().values.all(axis=0):
            return None
        df_x = self.fill_missing_groups(df_x, xname, celltype)  # make sure emtpy groups have nan
        # print("dfx type 4: ", type(df_x))
        if "default_group" in self.experiment.keys():
            df_x = df_x.apply(self.clear_missing_groups, axis=1, data=xname, replacement=self.experiment["default_group"])
        else:
            df_x = df_x.apply(self.clear_missing_groups, axis=1, data=xname)
        # print("dfx type 5: ", type(df_x))
        # print("uniques: ", df_x[xname].unique())
        df_x.dropna(subset=[xname], inplace=True)
        CP(
            "y",
            f"      b_rpts: Celltype: {celltype:s}, Groups: {df_x.Group.unique()!s} expression: {df_x.cell_expression.unique()!s}",
        )
 
        dodge = True
        if (
            "hue_palette" in self.experiment.keys()
            and hue_category in self.experiment["hue_palette"].keys()
        ):
            hue_palette = self.experiment["hue_palette"][hue_category]
        else:
            hue_category = xname
            hue_palette = colors
        if hue_category != xname:
            dodge = True
            hue_order = self.experiment["plot_order"][hue_category]
        else:
            dodge = False
            hue_order = plot_order  # print("plotting bar plot for ", celltype, yname, hue_category)
        if (
            "remove_expression" in self.experiment.keys()
            and self.experiment["remove_expression"] is not None
        ):
            for expression in self.experiment["remove_expression"]:
                if expression in hue_palette:
                    hue_palette.pop(expression)
                if expression in hue_order:
                    hue_order.remove(expression)

        # if hue_category == "sex":
        #     print("setting hue_palette via sex")
        #     hue_palette = {
        #         "F": "#FF000088",
        #         "M": "#0000ff88",
        #         " ": "k",
        #         "AIE": "#444488FF",
        #         "CON": "#9999ddFF",
        #     }
        # elif hue_category == "temperature":
        #     hue_palette = {"22": "#0000FF88", "34": "#FF000088", " ": "#888888FF"}

        # dodge = False
        # print("hue Palette: ", hue_palette)
        # print("hue category: ", hue_category)

        # must use scatterplot if you want to use picking.
        if enable_picking:
            # print("xname, uniqe xnames: ", xname, df_x[xname].unique())
            # print("hue category: ", hue_category)
            sns.swarmplot(
                x=xname,
                y=yname,
                hue=hue_category,
                # style=hue_category,
                data=df_x,
                size=3.5,
                # fliersize=None,
                alpha=1.0,
                ax=ax,
                palette=hue_palette,
                edgecolor="k",
                linewidth=0.5,
                order=plot_order,
                hue_order=plot_order,
                picker=enable_picking,
                zorder=100,
                clip_on=False,
            )
        else:

            sns.stripplot(
                x=xname,
                y=yname,
                hue=hue_category,
                data=df_x,
                order=plot_order,
                hue_order=hue_order,
                dodge=dodge,
                size=3.5,
                # fliersize=None,
                jitter=0.25,
                alpha=1.0,
                ax=ax,
                palette=hue_palette,
                edgecolor="k",
                linewidth=0.5,
                picker=enable_picking,
                zorder=100,
                clip_on=False,
            )

        # print("xname: ", xname)
        # print(df_x[xname])
        # print(df_x[yname])
        if not all(np.isnan(df_x[yname])):
            sns.boxplot(
                data=df_x,
                x=xname,
                y=yname,
                hue=hue_category,
                hue_order=hue_order,
                palette=hue_palette,
                ax=ax,
                order=plot_order,
                saturation=0.25,
                orient="v",
                showfliers=False,
                linewidth=0.5,
                zorder=50,
                # clip_on=False,
            )
        # except Exception as e:
        #     print("boxplot failed for ", celltype, yname)
        #     raise e  # re-raise the exception

        ax.set_xticks(ax.get_xticks(), ax.get_xticklabels(), rotation=45, ha="right")
        # remove extra legend handles
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(
            handles[2:],
            labels[2:],
            title="Group",
            bbox_to_anchor=(1, 1.02),
            loc="upper left",
        )
        if enable_picking:
            picker_func = Picker(
                space=2, data=df_x.copy(deep=True), axis=ax
            )  # create a new instance of the picker class
            # print("Set PICKER data")
        else:
            picker_func = None
        return picker_func

    def make_cell_id(self, row):
        sliceno = int(row["slice_slice"][-3:])
        cellno = int(row["cell_cell"][-3:])
        cell_id = f"{row['date']:s}_S{sliceno:d}C{cellno:d}"
        row["cell_id"] = cell_id
        return row

    def getFIMax(self, row):
        """getFIMax get the maximum rate from the FI Hill fit

        Parameters
        ----------
        row : pd.row_

        Returns
        -------
        value for row entry
        """
        data = np.squeeze(row.pars)
        if len(data) == 0:
            return np.nan
        return data[1]

    def getFIRate(self, row):
        data = np.squeeze(row.pars)
        if data.shape == ():  # single value
            return np.nan
        if len(data) == 4:
            return data[2]
        else:
            return np.nan

    def flag_date(self, row):
        if row.shortdate >= after_parsed:
            row.SR = 1
        else:
            row.SR = 0
        return row.SR

    def relabel_yaxes(self, axp, measure):
        if "new_ylabels" in self.experiment.keys():
            if measure in self.experiment["new_ylabels"]:
                axp.set_ylabel(self.experiment["new_ylabels"][measure])

    def relabel_xaxes(self, axp):
        if "new_xlabels" in self.experiment.keys():
            xlabels = axp.get_xticklabels()
            xticks = axp.get_xticks()
            for i, label in enumerate(xlabels):
                labeltext = label.get_text()
                if labeltext in self.experiment["new_xlabels"]:
                    xlabels[i] = self.experiment["new_xlabels"][labeltext]
            axp.set_xticks(xticks)  # we have to reset the ticks to avoid warning in matplotlib
            axp.set_xticklabels(xlabels)  # we just replace them...
        else:
            pass
            #  print("no new x labels available")

    def rescale_values_apply(self, row, measure, scale=1.0):
        if measure in row.keys():
            # print("measure: ", row[measure])
            if isinstance(row[measure], list):
                # meanmeasure = np.nanmean(row[measure])*scale
                # print(type(meanmeasure))
                row[measure] = np.nanmean(row[measure]) * scale
            else:
                row[measure] = row[measure] * scale

        return row[measure]

    def rescale_values(self, df):
        rescaling = {
            "AP_HW": 1e6,  # convert to usec
            "AP_thr_V": 1,
            "AHP_depth_V": 1,
            "AHP_trough_V": 1e3,
            "AHP_trough_T": 1e3,
            "FISlope": 1e-9,
            "maxHillSlope": 1,
            "I_maxHillSlope": 1e-3,
            "dvdt_falling": -1.0,
            "taum": 1e3,
        }
        for measure, scale in rescaling.items():
            if measure not in df.columns:
                continue
            df[measure] = df.apply(
                self.rescale_values_apply,
                axis=1,
                measure=measure,
                scale=scale,
            )
        return df

    def average_Rs(self, row):
        rs = []
        # print("Rs row: ", row['Rs'])
        # for protocol in row['Rs'].keys():
        #     rs.append(row['Rs'][protocol])
        rsa = np.mean(row["Rs"])
        row["Rs"] = rsa
        return row

    def average_CNeut(self, row):
        rs = []
        # for protocol in row['CNeut'].keys():
        #     rs.append(row['CNeut'][protocol])
        rsa = np.mean(row["CNeut"])
        row["CNeut"] = rsa
        return row

    def place_legend(self, P):
        legend_text = {
            "Ctl: Control (sham)  (P56)": "g",
            "103dB: 96-106 dB SPL@P42, 14D": "b",
            "115dB 2W: 115 dB SPL@P42, 14D": "orange",
            "115dB 3D: 115 dB SPL@P53, 3D": "magenta",
        }
        for i, txt in enumerate(legend_text.keys()):
            mpl.text(
                x=0.8,
                y=0.95 - i * 0.02,
                s=txt,
                color=legend_text[txt],  # ns.color_palette()[i],
                fontsize="medium",
                transform=P.figure_handle.transFigure,
            )

    def remove_nans(self, row, measure):
        if isinstance(row[measure], list):
            m = [x for x in row[measure]]  #  if not pd.isnull(x)]
            if len(m) == 1 and (np.isnan(m[0]) or m[0] == "nan"):
                m = "NA"
            return m
        elif isinstance(row[measure], float):
            row[measure] = [row[measure]]
            return row[measure]
        elif np.isnan(row[measure]):
         return "NA"
        else:
            raise ValueError(f"measure is not a list or float: {row[measure]!s}")

    def make_subject_name(self, row):
        sn = Path(row.cell_id).name
        sn = sn.split("_")[0]
        row["Subject"] = sn
        return row

    def export_r(
        self, df: pd.DataFrame, xname: str, measures: str, hue_category: str, filename: str
    ):
        """export_r _summary_
        Export  the relevant data to a csv file to read in R for further analysis.
        """
        # print("Xname: ", xname, "hue_category: ", hue_category, "measures: ", measures)
        if hue_category is None:
            columns = [xname]
        else:
            columns = [xname, hue_category]
        columns.extend(measures)
        parameters = [
            "Rs",
            "Rin",
            "CNeut",
            "RMP",
            "taum",
            "CC_taum",  # CC_taum protocol
            "tauh",
            "Gh",
            "dvdt_falling",
            "dvdt_rising",
            "AP_thr_V",
            "AP_thr_T",
            "AP_HW",
            "AHP_trough_V",
            "AHP_trough_T",
            "AHP_depth_V",
            "AdaptRatio",
            "FIMax_1" "FIMax_4",
            "maxHillSlope",
            "I_maxHillSlope",
        ]
        df = SCD.populate_columns(
            df,
            configuration=self.experiment,
            select_by="Rs",
            select_limits=[0, 1e9],
            parameters=parameters,
        )
        if "animal identifier" in columns:
            df.rename(columns={"animal identifier": "animal_identifier"}, errors="raise")
        ensure_cols = [
            "Group",
            "age",
            "sex",
            "cell_type",
            "cell_expression",
            "cell_id",
            "Rs",
            "CNeut",
            "Subject",
            "protocols",
            "used_protocols",
        ]
        for c in ensure_cols:
            if c not in columns:
                columns.append(c)
        select_by = "Rs"
        df_R = df[[c for c in columns if c != "AP_peak_V"]]
        if "Subject" not in df.columns:
            df_R = df_R.apply(self.make_subject_name, axis=1)
        if "Rs" in df_R.columns:
            df_R = df_R.apply(self.average_Rs, axis=1)
        if "CNeut" in df_R.columns:
            df_R = df_R.apply(self.average_CNeut, axis=1)

        # for meas in measures:
        #     if meas in df_R.columns:
        #         print("meas: ", meas)
        df_R = SCD.perform_selection(
            select_by=select_by,
            select_limits=[0, 1e9],
            data=df_R,
            parameters=measures,
            configuration=self.experiment,
        )

        CP("g", f"Exporting analyzed data to {filename!s}")
        df_R.to_csv(filename, index=False)

    def summary_plot_spike_parameters_categorical(
        self,
        df,
        xname: str,
        hue_category=None,
        plot_order: Optional[list] = None,
        measures: Optional[list] = None,
        representation: str = "all",
        colors=None,
        enable_picking=False,
    ):
        """Make a summary plot of spike parameters for selected cell types.

        Args:
            df (Pandas dataframe): _description_
            xname: str: name to use for x category
            hue_category: str: name to use for hue category
            plot_order: list, optional: order to plot the categories
            measures: list, optional: list of measures to plot
            colors: dict, optional: dictionary of colors to use for the categories
            enable_picking: bool, optional: enable picking of data points
        """
        df = df.copy()  # make sure we don't modifiy the incoming
        print("summary plot incoming x : ", df[xname].unique())

        # Remove cells for which the FI Hill slope is maximal at 0 nA:
        #    These have spont.
        # df = df[df["I_maxHillSlope"] > 1e-11]
        # for p in plot_order:
        #     print(p, df[df['age_category'] == p]['ID'])
        print(
            "summary_plot_spike_parameters_categorical: incoming x categories: ", df[xname].unique()
        )

        ncols = len(measures)
        letters = ["A", "B", "C", "D", "E", "F", "G", "H"]
        plabels = [f"{let:s}{num+1:d}" for let in letters for num in range(ncols)]
        figure_width = ncols * 2.5
        P = PH.regular_grid(
            len(self.experiment["celltypes"]),
            ncols,
            order="rowsfirst",
            figsize=(figure_width, 2.5 * len(self.experiment["celltypes"]) + 1.0),
            panel_labels=plabels,
            labelposition=(0.01, 0.95),
            margins={
                "topmargin": 0.12,
                "bottommargin": 0.12,
                "leftmargin": 0.1,
                "rightmargin": 0.15,
            },
            verticalspacing=0.04,
            horizontalspacing=0.07,
        )
        self.label_celltypes(P, analysis_cell_types=self.experiment["celltypes"])
        for ax in P.axdict:
            PH.nice_plot(P.axdict[ax], direction="outward", ticklength=3, position=-0.03)
        picker_funcs = {}
        # n_celltypes = len(self.experiment["celltypes"])
        # print(df.cell_type.unique())
        df = self.rescale_values(df)
        if representation in ["bestRs", "mean"]:
            df = SCD.get_best_and_mean(
                df,
                experiment=self.experiment,
                parameters=measures,
                select_by="Rs",
                select_limits=[0, 1e9],
            )
            for i, m in enumerate(measures):
                measures[i] = f"{m:s}_{representation:s}"

        # print("plot order: ", plot_order)
        for icol, measure in enumerate(measures):
            # if measure in ["AP_thr_V", "AHP_depth_V"]:
            #     CP("y", f"{measure:s}: {df[measure]!s}")

            if measure in self.transforms.keys():
                tf = self.transforms[measure]
            else:
                tf = None
            # print(":: cell types ::", self.experiment["celltypes"])
            for i, celltype in enumerate(self.experiment["celltypes"]):
                # print("measure y: ", measure, "celltype: ", celltype)
                axp = P.axdict[f"{letters[i]:s}{icol+1:d}"]
                if celltype not in self.ylims.keys():
                    ycell = "default"
                else:
                    ycell = celltype
                x_measure = "_".join((measure.split("_"))[:-1])
                if x_measure not in self.ylims[ycell]:
                    continue
                if measure not in df.columns:
                    continue
                # try:
                picker_func = self.create_one_plot_categorical(
                    data=df,
                    xname=xname,
                    y=measure,
                    ax=axp,
                    celltype=celltype,
                    hue_category=hue_category,
                    plot_order=plot_order,
                    colors=colors,
                    logx=False,
                    ylims=self.ylims[ycell][x_measure],
                    transform=tf,
                    xlims=None,
                    enable_picking=enable_picking,
                )
                # except Exception as e:
                #     print("Categorical plot error in ylims: ", self.ylims.keys(), ycell)
                #     raise KeyError(f"\n{e!s}")
                picker_funcs[axp] = picker_func  # each axis has different data...
                if celltype != self.experiment["celltypes"][-1]:
                    axp.set_xticklabels("")
                    axp.set_xlabel("")
                else:
                    self.relabel_xaxes(axp)
                self.relabel_yaxes(axp, measure=x_measure)

                # if icol > 0:
                #     axp.set_ylabel("")
                #     print("removed ylabel from ", icol, measure, celltype)

        # print("picking: ", enable_picking, picker_funcs)
        if len(picker_funcs) > 0 and enable_picking:
            P.figure_handle.canvas.mpl_connect(
                "pick_event", lambda event: self.pick_handler(event, picker_funcs)
            )
        else:
            picker_funcs = None

        for ax in P.axdict:
            if P.axdict[ax].legend_ is not None:
                P.axdict[
                    ax
                ].legend_.remove()  # , direction="outward", ticklength=3, position=-0.03)
        # place_legend(P)
        i = 0
        icol = 0
        axp = P.axdict[f"{plabels[i]:s}"]
        axp.legend(
            fontsize=7, bbox_to_anchor=(0.95, 0.90), bbox_transform=P.figure_handle.transFigure
        )
        if any(c.startswith("dvdt_rising") for c in measures):
            fn = "spike_shapes.csv"
        elif any(c.startswith("AdaptRatio") for c in measures):
            fn = "firing_parameters.csv"
        elif any(c.startswith("RMP") for c in measures):
            fn = "rmtau.csv"
        self.export_r(df=df, xname=xname, measures=measures, hue_category=hue_category, filename=fn)
        return P, picker_funcs

    def pick_handler(self, event, picker_funcs):
        # print("\n\nPICKER")
        # print("picker funcs: ", len(picker_funcs), "\n", picker_funcs)
        if picker_funcs is None:
            return
        # print("event inaxes: ", event.mouseevent.inaxes)
        if event.mouseevent.inaxes not in picker_funcs.keys():
            CP("r", "The picked value is outside an axis; cannot resolve")
            return None
        picker_func = picker_funcs[event.mouseevent.inaxes]  # get the picker function for this axis
        # print("\nDataframe index: ", event.ind)
        # print("# values in data: ", len(picker_func.data))
        for i in event.ind:
            cell = picker_func.data.iloc[i]

            print(f"   {i:d} {cell['cell_id']:s}")  # find the matching data.
            print(
                "       Age: ",
                int(cell["age"]),
                "group: ",
                cell["Group"],
                "type",
                cell["cell_type"],
                "age category: ",
                cell["age_category"],
            )
            # age = get_age(cell["age"])
            # print(
            #     f"       {cell['cell_type'].values[0]:s}, Age: P{age:d}D Group: {cell['Group'].values[0]!s}"
            # )
            # print(f"       Protocols: {cell['protocols'].values[0]!s}")
            # print(cell["dvdt_rising"], cell["dvdt_falling"])
        # if self.pick_display:
        if len(event.ind) == 1:
            cell_by_id = picker_func.data.iloc[event.ind[0]]["cell_id"]
            self.pick_display_function(cell_by_id)
            return cell_by_id
        return None

    def create_one_plot_continuous(
        self,
        data,
        x,
        y,
        ax,
        celltype: str,
        logx=False,
        ylims=None,
        xlims=None,
        yscale=1,
        transform=None,
    ):
        """create_one_plot create one plot for a cell type

        Parameters
        ----------
        data : Pandas dataframe
            Data to plot
        x : str
            x axis data
        y : str
            y axis data
        ax : object
            Axis to plot into
        celltype : str
            Cell type to plot
        picker_funcs : dict
            Dictionary of picker functions
        picker_func : Picker
            Picker function to use
        logx : bool, optional
            Use log scale on x axis, by default False
        """
        if celltype != "all":
            dfp = data[data["cell_type"] == celltype]
        else:
            dfp = data
        dfp[y] = dfp[y] * yscale
        if transform is not None:
            dfp[y] = dfp[y].apply(transform, axis=1)

        # sns.regplot(
        #     data=dfp,
        #     x=x,
        #     y=y,
        #     ax=ax,
        #     logx=logx,
        # )

        ax.scatter(x=dfp[x], y=dfp[y], c="b", s=4, picker=True)
        if y.endswith("_bestRs") or y.endswith("_mean"):
            y = "_".join([*y.split("_")[:-1]])  # reassemble without the trailing label
        self.relabel_yaxes(ax, measure=y)
        self.relabel_xaxes(ax)
        if ylims is not None:
            # print(ylims[celltype])
            if y in ylims[celltype]:
                ax.set_ylim(ylims[celltype][y])
        if xlims is not None:
            ax.set_xlim(xlims)
        picker_func = Picker()
        picker_func.setData(dfp.copy(deep=True), axis=ax)
        return picker_func
        # picker_funcs[celltype].setAction(handle_pick) # handle_pick is a function that takes the index of the picked point as an argument

    def apply_scale(self, row, measure, scale):
        if measure in row.keys():
            if isinstance(row[measure], list):
                row[measure] = np.nanmean(row[measure]) * scale
            else:
                row[measure] = row[measure] * scale
        return row

    def create_one_plot_categorical(
        self,
        data,
        xname,  # name of the dataframe column holding categorical data
        y,
        ax,
        celltype,
        hue_category: str = None,
        plot_order=None,
        colors=None,
        logx=False,
        ylims=None,
        xlims=None,
        yscale=1,
        transform=None,
        enable_picking=False,
    ):
        """create_one_plot create one plot for a cell type, using categorical data in x.

        Parameters
        ----------
        data : Pandas dataframe
            Data to plot
        x : str
            x axis data (list of categories)
        y : str
            y axis data (column name of measure)
        ax : object
            Axis to plot into
        celltype : str
            Cell type to plot
        colors: : dict or None
            dict of colors by x categories
            if None, all categories will be blue.
        picker_funcs : dict
            Dictionary of picker functions
        picker_func : Picker
            Picker function to use
        logx : bool, optional
            Use log scale on x axis, by default False
        """
        dfp = data.copy()
        # print("database celltypes: ", dfp.cell_type.unique())
        if celltype != "all":
            dfp = dfp[dfp["cell_type"] == celltype]
        dfp = dfp.apply(self.apply_scale, axis=1, measure=y, scale=yscale)
        if transform is not None:
            dfp[y] = dfp[y].apply(transform)
        # if hue_category is None:
        #     raise ValueError(f"Missing Hue category for plot; xname is: {xname:s}")
        # print("calling barpts. Hue_category: ", hue_category, "Plot order: ", plot_order)
        # print("x categories: ", dfp[xname].unique())
        picker_func = self.bar_pts(
            dfp,
            xname=xname,
            yname=y,
            ax=ax,
            hue_category=hue_category,
            celltype=celltype,
            plot_order=plot_order,
            colors=colors,
            enable_picking=enable_picking,
        )
        if ylims is not None:
            ax.set_ylim(ylims)
        if xlims is not None:
            ax.set_xlim(xlims)
        self.relabel_yaxes(ax, measure=y)
        self.relabel_xaxes(ax)
        # print("Plotted measure: ", y, "for celltype: ", celltype)
        # print("dfp: ", dfp)
        return picker_func

    def clean_rin(self, row):
        min_Rin = 6.0
        if "data_inclusion_criteria" in self.experiment.keys():
            if row.cell_type in self.experiment["data_inclusion_criteria"].keys():
                min_Rin = self.experiment["data_inclusion_criteria"][row.cell_type]["Rin_min"]
            else:
                min_Rin = self.experiment["data_inclusion_criteria"]["default"]["Rin_min"]
        # print("rowrin: ", row.Rin)
        if isinstance(row.Rin, float):
            row.Rin = [row.Rin]
        for i, rin in enumerate(row.Rin):
            # print("rin: ", rin)
            if row.Rin[i] < min_Rin:
                row.Rin[i] = np.nan
        return row.Rin

    def clean_rmp(self, row):
        min_RMP = -55.0
        if "data_inclusion_criteria" in self.experiment.keys():
            if row.cell_type in self.experiment["data_inclusion_criteria"].keys():
                min_RMP = self.experiment["data_inclusion_criteria"][row.cell_type]["RMP_min"]
            else:
                min_RMP = self.experiment["data_inclusion_criteria"]["default"]["RMP_min"]
        if isinstance(row.RMP, float):
            row.RMP = [row.RMP]
        for i, rmp in enumerate(row.RMP):
            if rmp > min_RMP:
                row.RMP[i] = np.nan
        return row.RMP

    def clean_rmp_zero(self, row):
        min_RMP = -55.0
        if "data_inclusion_criteria" in self.experiment.keys():
            if row.cell_type in self.experiment["data_inclusion_criteria"].keys():
                min_RMP = self.experiment["data_inclusion_criteria"][row.cell_type]["RMP_min"]
            else:
                min_RMP = self.experiment["data_inclusion_criteria"]["default"]["RMP_min"]
        if isinstance(row.RMP_Zero, float):
            r0 = [row.RMP_Zero]  # handle case where there is only one float value
        else:
            r0 = row.RMP_Zero
        for i, r0 in enumerate(r0):
            if r0 > min_RMP:
                row.RMP_Zero[i] = np.nan
        return row.RMP_Zero

    def summary_plot_spike_parameters_continuous(
        self,
        df,
        measures,
        xname: str = "",
        logx=False,
        xlims=None,
        representation: str = "bestRs",  # bestRs, mean, all
    ):
        """Make a summary plot of spike parameters for selected cell types.

        Args:
            df (Pandas dataframe): _description_
        """
        picker_funcs = {}
        measures = [
            "dvdt_rising",
            "dvdt_falling",
            "AP_HW",
            "AP_thr_V",
            "AdaptRatio",
            "FISlope",
            "maxHillSlope",
            "FIMax_1",
            # "FIMax_4",
        ]

        ncols = len(measures)
        letters = letters = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]
        plabels = [f"{let:s}{num+1:d}" for let in letters for num in range(ncols)]
        df = df.copy()
        # df["FIMax_1"] = df.apply(getFIMax_1, axis=1)
        # df["FIMax_4"] = df.apply(getFIMax_imax, axis=1, imax=4.0)
        df["FIRate"] = df.apply(self.getFIRate, axis=1)
        df.dropna(subset=["Group"], inplace=True)  # remove empty groups
        df["age"] = df.apply(numeric_age, axis=1)
        df["shortdate"] = df.apply(make_datetime_date, axis=1)
        df["SR"] = df.apply(self.flag_date, axis=1)
        # df.dropna(subset=["age"], inplace=True)
        df = self.rescale_values(df)
        if representation in ["bestRs", "mean"]:
            df = SCD.get_best_and_mean(
                df,
                experiment=self.experiment,
                parameters=measures,
                select_by="Rs",
                select_limits=[0, 1e9],
            )
            for i, m in enumerate(measures):
                measures[i] = f"{m:s}_{representation:s}"

        # print("DF columns after get best...", df.columns)
        if "max_age" in self.experiment.keys():
            df = df[(df.Age >= 0) & (df.Age <= self.experiment["max_age"])]
        # df = df[df.SR == 1]
        if xlims is None:
            xlims = (0, 600)
        P = PH.regular_grid(
            len(self.experiment["celltypes"]),
            ncols,
            order="rowsfirst",
            figsize=(16, 9),
            panel_labels=plabels,
            labelposition=(0.01, 0.95),
            margins={
                "topmargin": 0.12,
                "bottommargin": 0.12,
                "leftmargin": 0.1,
                "rightmargin": 0.15,
            },
            verticalspacing=0.12,
            horizontalspacing=0.05,
        )
        for ax in P.axdict:
            PH.nice_plot(P.axdict[ax], direction="outward", ticklength=3, position=-0.03)

        showcells = self.experiment["celltypes"]
        for i, celltype in enumerate(showcells):
            let = letters[i]
            if self.experiment["celltypes"] != ["all"]:
                dfp = df[df["cell_type"] == celltype]
            else:
                dfp = df
            axp = P.axdict[f"{let:s}1"]
            for icol, measure in enumerate(measures):
                if measure in self.transforms.keys():
                    tf = self.transforms[measure]
                    dfp[measure] = dfp[measure].apply(tf, axis=1)
                else:
                    tf = None
                axp = P.axdict[f"{let:s}{icol+1:d}"]
                # print("ylims: ", self.ylims)
                picker_funcs[axp] = self.create_one_plot_continuous(
                    data=dfp,
                    x="age",
                    y=measure,
                    ax=axp,
                    celltype=celltype,
                    logx=logx,
                    ylims=self.ylims,
                    xlims=xlims,
                    transform=tf,
                )  # yscale=yscale[measure]

            # if celltype != "cartwheel":
            #     P.axdict[f"{let:s}7"].set_ylim(0, 800)
            # else:
            #     P.axdict[f"{let:s}7"].set_ylim(0, 100)

        P.figure_handle.canvas.mpl_connect(
            "pick_event", lambda event: self.pick_handler(event, picker_funcs)
        )

        for ax in P.axdict:
            if P.axdict[ax].legend_ is not None:
                P.axdict[
                    ax
                ].legend_.remove()  # , direction="outward", ticklength=3, position=-0.03)

        # place_legend(P)

        for i, cell in enumerate(showcells):
            mpl.text(
                x=0.05,
                y=0.8 - i * 0.3,
                s=cell.title(),
                color="k",
                rotation="vertical",
                ha="center",
                va="center",
                fontsize="large",
                fontweight="bold",
                transform=P.figure_handle.transFigure,
            )
        return P, picker_funcs

    def label_celltypes(self, P, analysis_cell_types):
        """label_celltypes
        put vertical labels on the left side of the figure,
        with each label centered on its row of plots.

        Parameters
        ----------
        P : PH.regular_grid object
            the PlotHandle object for this figure
        """
        n_celltypes = len(self.experiment["celltypes"])

        for i in range(n_celltypes):
            ax = P.axarr[i, 0]
            yp = ax.get_position()
            mpl.text(
                x=0.02,
                y=yp.y0 + 0.5 * (yp.y1 - yp.y0),  # center vertically
                s=self.experiment["celltypes"][i].title(),
                color="k",
                rotation="vertical",
                ha="center",
                va="center",
                fontsize="large",
                fontweight="bold",
                transform=P.figure_handle.transFigure,
            )
        return

    def summary_plot_rm_tau_categorical(
        self,
        df,
        xname: str,
        plot_order: list,
        hue_category: str = None,
        measures: list = None,
        representation: str = "all",
        colors=None,
        enable_picking=True,
    ):
        """Make a summary plot of basic cell measures for selected cell types.

        Args:
            df (Pandas dataframe): _description_
        """
        print("\nSummary Plot rmtau categorical")
        print("# of entries in the table: ", len(df))
        df = df.copy()
        # print("len df: ", len(df))
        letters = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]
        measures = ["RMP", "RMP_Zero", "Rin", "taum"]
        plabels = [f"{let:s}{num+1:d}" for let in letters for num in range(len(measures))]
        self.rescale_values(df)
        if representation in ["bestRs", "mean"]:
            df = SCD.get_best_and_mean(
                df,
                experiment=self.experiment,
                parameters=measures,
                select_by="Rs",
                select_limits=[0, 1e9],
            )
            for i, m in enumerate(measures):
                measures[i] = f"{m:s}_{representation:s}"
        picker_funcs = {}
        figure_width = len(measures) * 2.75
        P = PH.regular_grid(
            len(self.experiment["celltypes"]),
            len(measures),
            order="rowsfirst",
            figsize=(figure_width, 2.5 * len(self.experiment["celltypes"])),
            panel_labels=plabels,
            labelposition=(0.01, 0.95),
            margins={
                "topmargin": 0.12,
                "bottommargin": 0.12,
                "leftmargin": 0.12,
                "rightmargin": 0.15,
            },
            verticalspacing=0.04,
            horizontalspacing=0.07,
        )
        self.label_celltypes(P, analysis_cell_types=self.experiment["celltypes"])

        if representation in ["bestRs", "mean"]:
            df = SCD.get_best_and_mean(
                df,
                experiment=self.experiment,
                parameters=measures,
                select_by="Rs",
                select_limits=[0, 1e9],
            )
            for i, m in enumerate(measures):
                measures[i] = f"{m:s}_{representation:s}"

        for ax in P.axdict:
            PH.nice_plot(P.axdict[ax], direction="outward", ticklength=3, position=-0.03)
        # print("self.experiment celltypes: ", self.experiment["celltypes"])
        for i, celltype in enumerate(self.experiment["celltypes"]):
            let = letters[i]

            for j, measure in enumerate(measures):
                #     if measure in rescaling.keys():
                #         yscale = rescaling[measure]
                #     else:
                #         yscale = 1

                # df = df.apply(apply_scale, axis=1, measure=measure, scale=yscale)
                axp = P.axdict[f"{let:s}{j+1:d}"]
                # print("    enable picking: ", enable_picking)
                # print("xname: ", xname, "hue cat: ", hue_category, "plot_order", plot_order, df['age_category'].unique())
                picker_funcs[axp] = self.create_one_plot_categorical(
                    data=df,
                    xname=xname,
                    y=measure,
                    hue_category=hue_category,
                    plot_order=plot_order,
                    ax=axp,
                    celltype=celltype,
                    colors=colors,
                    logx=False,
                    ylims=self.ylims[celltype][measure],
                    xlims=None,
                    enable_picking=enable_picking,
                )
                # picker_funcs[axp] = pickerfunc
                if celltype != self.experiment["celltypes"][-1]:
                    axp.set_xticklabels("")
                axp.set_xlabel("")
                # if j > 0:
                #     axp.set_ylabel("")
        if any(picker_funcs) is not None and enable_picking:
            P.figure_handle.canvas.mpl_connect(
                "pick_event", lambda event: self.pick_handler(event, picker_funcs)
            )
        else:
            picker_funcs = None
        for ax in P.axdict:
            if P.axdict[ax].legend_ is not None:
                P.axdict[
                    ax
                ].legend_.remove()  # , direction="outward", ticklength=3, position=-0.03)
        # self.place_legend(P)
        i = 0
        icol = 0
        axp = P.axdict[f"{letters[i]:s}{icol+1:d}"]
        axp.legend(
            fontsize=7, bbox_to_anchor=(0.95, 0.90), bbox_transform=P.figure_handle.transFigure
        )
        self.export_r(df, xname, measures, hue_category, "rmtau.csv")
        return P, picker_funcs

    def limit_to_max_rate_and_current(self, fi_data, imax=None, id="", limit=0.9):
        """limit_to_max_rate_and_current:
            return the FI data limited to the maximum rate (no rollover) and current

        Parameters
        ----------
        FI_data : np.ndarray, 2 x # curent levelws
            Dataframe with FI data
        imax : float, optional
            Maximum current to include, by default None

        Returns
        -------
        FI_data
            array with FI data limited to maximum rate and current
        """
        fi_rate = np.array(fi_data[1])
        fi_inj = np.array(fi_data[0])
        iunique = np.unique(fi_inj)
        if len(iunique) < len(fi_inj):
            f_out = np.zeros((2, len(iunique)))
            for i, iu in enumerate(iunique):
                inj = iunique[i]
                f_out[0, i] = inj
                f_out[1, i] = np.mean(fi_rate[np.where(fi_inj == inj)])
            fi_inj = f_out[0]
            fi_rate = f_out[1]

        i_limit = np.argwhere(fi_inj <= imax)[-1][0] + 1  # only consider currents less than imax
        fi_rate = fi_rate[:i_limit]
        fi_inj = fi_inj[:i_limit]
        spike_rate_max_index = np.argmax(fi_rate)  # get where the peak rate is located
        spike_rate_max = fi_rate[spike_rate_max_index]
        spike_rate_slope: np.ndarray = np.gradient(fi_rate, fi_inj)
        xnm = np.where(
            (spike_rate_slope[spike_rate_max_index:] <= 0.0)  # find where slope is negative
            & (fi_rate[spike_rate_max_index:] < (limit * spike_rate_max))
        )[
            0
        ]  # and rate is less than limit of max
        if len(xnm) > 0:
            i_fimax = xnm[0] + spike_rate_max_index
        else:
            i_fimax = len(fi_rate)
        dypos = list(range(0, i_fimax))
        fi_rate = fi_rate[dypos]
        fi_inj = fi_inj[dypos]
        return np.array([fi_inj, fi_rate])

    def summary_plot_fi(
        self,
        df,
        mode: list = ["individual"],
        group_by: str = "Group",
        colors: Optional[dict] = None,
        plot_order: Optional[list] = None,
        enable_picking: bool = False,
    ):
        """summary_plot_fi Plots all of the FI curves for the selected cell types,
        including the mean and SEM for each cell type.

        Parameters
        ----------
        mode : list, optional
            _description_, by default ["individual"]
        colors : Union[dict, None], optional
            _description_, by default None

        Returns
        -------
        _type_
            _description_
        """
        print("summaryplotFI")
        df = df.copy()
        # The "protocol" field is not meaningful here as we combined the FI curves
        # in the combine_fi_curves function.
        # if "protocol" not in df.columns:
        #     df = df.rename({"iv_name": "protocol"}, axis="columns")
        # print("unique protocols: ", df["protocol"].unique())
        # print("protocols: ", df["protocols"])
        plabels = ["A1", "A2", "A3", "B1", "B2", "B3", "G", "H", "I", "J"]
        P = PH.regular_grid(
            rows=len(self.experiment["celltypes"]),
            cols=3,
            order="rowsfirst",
            figsize=(8 + 1.0, 3 * len(self.experiment["celltypes"])),
            panel_labels=plabels,
            labelposition=(0.01, 0.95),
            margins={
                "topmargin": 0.12,
                "bottommargin": 0.12,
                "leftmargin": 0.12,
                "rightmargin": 0.15,
            },
            verticalspacing=0.2,
            horizontalspacing=0.1,
            fontsize={"label": 12, "tick": 8, "panel": 16},
        )
        # print(P.axdict)
        for ax in P.axdict:
            PH.nice_plot(P.axdict[ax], direction="outward", ticklength=3, position=-0.03)
        pos = {
            "pyramidal": [0.0, 1.0],
            "tuberculoventral": [0.0, 1.0],
            "cartwheel": [0.0, 1.0],
            "bushy": [0.05, 0.95],
            "stellate": [0.6, 0.4],
            "t-stellate": [0.6, 0.4],
            "d-stellate": [0.6, 0.4],
            "giant": [0.6, 0.4],
            "giant_maybe": [0.6, 0.4],
            "default": [0.4, 0.15],
        }

        mode = "mean"  # "individual"
        # P.figure_handle.suptitle(f"Protocol: {','.join(protosel):s}", fontweight="bold", fontsize=18)
        if mode == "mean":
            P.figure_handle.suptitle("FI Mean, SEM ", fontweight="normal", fontsize=18)
        elif mode == "individual":
            P.figure_handle.suptitle("FI Individual", fontweight="normal", fontsize=18)
        fi_stats = []
        NCells: Dict[tuple] = {}
        picker_funcs: Dict = {}
        found_groups = []
        fi_group_sum = {}
        fi_dat = {}  # save raw fi
        for ic, ptype in enumerate(["mean", "individual", "sum"]):
            for ir, celltype in enumerate(self.experiment["celltypes"]):
                ax = P.axarr[ir, ic]
                ax.set_title(celltype.title(), y=1.05)
                ax.set_xlabel("I$_{inj}$ (nA)")
                if ic in [0, 1]:
                    ax.set_ylabel("Rate (sp/s)")
                elif ic == 2:
                    ax.set_ylabel("Firing 'Area' (pA*Hz)")
                if celltype != "all":
                    cdd = df[df["cell_type"] == celltype]
                else:
                    cdd = df.copy()

                N = self.experiment["group_map"]

                if ptype == "mean":  # set up arrays to compute mean
                    FIy_all: dict = {k: [] for k in N.keys()}
                    FIx_all: dict = {k: [] for k in N.keys()}
                for index in cdd.index:
                    group = cdd[group_by][index]
                    if (celltype, group) not in NCells.keys():
                        NCells[(celltype, group)] = 0

                    # if cdd["protocol"][index].startswith("CCIV_"):
                    #     this_protocol = "CCIV_"
                    # else:
                    # this_protocol = cdd["protocol"][index]  # # not meaningful if protocols are combined[
                    #                    :-4
                    #               ]  # get protocol, strip number
                    # print("This protocol: ", this_protocol, "sel: ", protosel, cdd["protocol"][index])
                    #
                    # if this_protocol not in protosel:
                    #     continue
                    if pd.isnull(cdd["cell_id"][index]):
                        print("No cell ids")
                        continue
                    try:
                        FI_data = FUNCS.convert_FI_array(cdd["FI_Curve1"][index])
                    except:
                        print("No FI data", cdd.cell_id[index])
                        print(cdd.columns)
                        print("Did you run the assembly on the final IV analysis files?")
                        continue
                    # raise KeyError("No FI data")

                    if len(FI_data[0]) == 0:
                        print("FI data is empty", cdd.cell_id[index])

                        continue
                    # print("FIdata: ", len(FI_data[0]))
                    FI_data[0] = np.round(np.array(FI_data[0]) * 1e9, 2) * 1e-9
                    if FI_data.shape == (2, 0):  # no fi data from the excel table....
                        print("No FI data from excel table?")
                        continue

                    if "max_FI" in self.experiment.keys():
                        max_fi = self.experiment["max_FI"] * 1e-9
                    else:
                        max_fi = 1.05e-9
                    FI_dat_saved = FI_data.copy()
                    ### HERE WE LIMIT FI_data to the range with the max firing
                    FI_data = self.limit_to_max_rate_and_current(
                        FI_data, imax=max_fi, id=cdd["cell_id"][index]
                    )
                    NCells[(celltype, group)] += 1  # to build legend, only use "found" groups
                    if group not in found_groups:
                        found_groups.append(group)
                        fi_group_sum[group] = []
                    maxi = 1000e-12
                    ilim = np.argwhere(FI_data[0] <= maxi)[-1][0]
                    if ptype in ["individual", "mean"]:
                        fix, fiy, fiystd, yn = FUNCS.avg_group(
                            np.array(FI_data[0]), FI_data[1], ndim=1
                        )

                        ax.plot(
                            fix[:ilim] * 1e9,
                            fiy[:ilim],
                            color=colors[group],
                            marker=None,
                            markersize=2.5,
                            linewidth=0.5,
                            clip_on=False,
                            alpha=0.35,
                        )
                    if ptype == "mean":
                        if group in FIy_all.keys():
                            FIy_all[group].append(np.array(FI_data[1][:ilim]))
                            FIx_all[group].append(np.array(FI_data[0][:ilim]) * 1e9)

                    elif ptype == "sum":
                        fi_group_sum[group].append(np.sum(np.array(FI_dat_saved[1])))

                if ptype == "mean":
                    max_FI = 1.0
                    for i, group in enumerate(FIy_all.keys()):
                        fx, fy, fystd, yn = FUNCS.avg_group(FIx_all[group], FIy_all[group])
                        if len(fx) == 0:
                            print("unable to get average", cdd["cell_id"][index])
                            continue
                        else:
                            print("getting average: ", cdd["cell_id"][index])
                        if "max_FI" in self.experiment.keys():
                            max_FI = self.experiment["max_FI"] * 1e-3
                        ax.errorbar(
                            fx[fx <= max_FI],
                            fy[fx <= max_FI],
                            yerr=fystd[fx <= max_FI] / np.sqrt(yn[fx <= max_FI]),
                            color=colors[group],
                            marker="o",
                            markersize=2.5,
                            linewidth=1.5,
                            clip_on=False,
                            label=self.experiment["group_legend_map"][group],
                        )

                        ax.set_xlim(0, max_FI)
                if ptype == "sum":
                    ax = P.axarr[ir, ic]
                    ax.set_title("Summed FI", y=1.05)
                    ax.set_xlabel("Group")
                    fi_list = []
                    # sumdf = pd.DataFrame(fi_group_sum)
                    for i, group in enumerate(fi_group_sum.keys()):
                        fi_list.append(fi_group_sum[group])
                        ax.scatter(
                            0.75 + i + np.random.random(len(fi_group_sum[group])) * 0.5,
                            fi_group_sum[group],
                            color=colors[group],
                            marker="o",
                            # hue="sex",
                            # hue_order=["M", "F"],
                            s=8.0,
                        )

                        ax.boxplot(fi_list, widths=0.8,
                                    )
                    # self.bar_pts(

                    #         df,
                    #     xname="Group",
                    #     yname="FI_sum",
                    #     celltype = "pyramidal",
                    #     hue_category = "sex",
                    #     ax = ax,
                    #     # plot_order: Optional[list] = None,
                    #     # colors: Optional[dict] = None,
                    #     enable_picking=False,
                    # )
                    ax.set_xlim(-0.5, 5.5)
                    ax.set_ylim(0, 1500)
                    p, t = scipy.stats.ttest_ind(fi_list[0], fi_list[1])
                    print(p, t)
                    print(fi_group_sum.keys(), len(fi_list[0]), len(fi_list[1]))
                print("group: ", group, "ptype: ", ptype)

            yoffset = 0.025
            xoffset = 0.0
            xo2 = 0.0
            # if "individual" in mode:
            #     yoffset = 0.7
            #     xoffset = -0.35
            #     if celltype == "pyramidal":
            #         xo2 = 0.0
            #     else:
            #         xo2 = -0.45
            for i, group in enumerate(self.experiment["group_legend_map"].keys()):
                if group not in found_groups:
                    continue
                if celltype == "pyramidal":  # more legend - name of group
                    ax.text(
                        x=pos[celltype][0] + xoffset + xo2,
                        y=pos[celltype][1] + 0.095 * (i - 0.5) + yoffset,
                        s=f"{self.experiment['group_legend_map'][group]:s} (N={NCells[(celltype, group)]:>3d})",
                        ha="left",
                        va="top",
                        fontsize=8,
                        color=colors[group],
                        transform=ax.transAxes,
                    )
                    print(
                        "Pyramidal legend: ",
                        f"{self.experiment['group_legend_map'][group]:s} (N={NCells[(celltype, group)]:>3d}",
                    )
                else:
                    if (celltype, group) in NCells.keys():
                        textline = f"{group:s} N={NCells[(celltype, group)]:>3d}"
                    else:
                        textline = f"N={0:>3d}"
                    fcelltype = celltype
                    if celltype not in pos.keys():
                        fcelltype = "default"
                    ax.text(
                        x=pos[fcelltype][0] + xoffset + xo2,
                        y=pos[fcelltype][1] - 0.095 * (i - 0.5) + yoffset,
                        s=textline,
                        ha="left",
                        va="top",
                        fontsize=8,
                        color=colors[group],
                        transform=ax.transAxes,
                    )

            print("-" * 80)

            if "mean" in mode:
                for i, group in enumerate(self.experiment["group_legend_map"].keys()):
                    if (celltype, group) in NCells.keys():
                        fi_stats.append(
                            {
                                "celltype": celltype,
                                "Group": group,
                                "I_nA": fx,
                                "sp_s": fy,
                                "N": NCells[(celltype, group)],
                            }
                        )
        i = 0
        icol = 0
        axp = P.axdict["A1"]
        axp.legend(
            fontsize=7, bbox_to_anchor=(0.95, 0.90), bbox_transform=P.figure_handle.transFigure
        )
        return P, picker_funcs

    def rename_group(self, row, group_by: str):
        row[group_by] = row[group_by].replace(" ", "")
        return row[group_by]

    def stats(
        self,
        df,
        celltype: str = None,
        measure: str = None,
        group_by: str = None,
        second_group_by: str = None,
        statistical_comparisons: list = None,
        parametric: bool = False,
        nonparametric: bool = True,
    ) -> pd.DataFrame:
        """stats Compute either or both parametric or nonparametric statistics on
        the incoming datasets.

        Parameters
        ----------
        df : pandas dataframe
            dataset to generate statics
        celltype : str, optional
            cell type name to do calculations on
        measure : str, optional
            The column measure to compute form, by default None
        group_by : str, optional
            Groups to compare, by default None
        second_group_by : str, optional
            For 2-way comparisons, second group (e.g., sex), by default None
        statistical_comparisons : list, optional
            _description_, by default None
        parametric : bool, optional
            Set True to do ANOVA, by default False
        nonparametric : bool, optional
            Set True to do Kruskal-Wallis, by default True

        Returns
        -------
        pd.DataFrame
            "cleaned" data used to generate the statistics - after removing nans, etc.
        """
        if celltype != "all":
            df_x = df[df.cell_type == celltype]
        else:
            df_x = df.copy()

        # print(df_x.head())
        # fvalue, pvalue = scipy.stats.f_oneway(df['A'], df['B'], df['AA'], df['AAA'])
        # indent the statistical results
        # wrapper = textwrap.TextWrapper(width=80, initial_indent=" "*8, subsequent_indent=" " * 8)

        headertext = ""
        cellcolors = {
            "pyramidal": "b",
            "cartwheel": "g",
            "tuberculoventral": "m",
            "giant": "k",
            "giant_maybe": "k",
        }
        if celltype not in cellcolors.keys():
            cellcolors[celltype] = "r"
        headertext = "\n\n".join(
            [
                headertext,
                CP(
                    "y",
                    f"{'%'*80:s}\n\nCelltype: {celltype!s}   Measure: {measure!s}",
                    textonly=True,
                ),
            ]
        )
        FUNCS.textappend(headertext)

        def make_numeric(row, measure):
            row[measure] = pd.to_numeric(row[measure])
            return row

        for i, s in enumerate(statistical_comparisons):
            s = s.replace(" ", "")  # replace spaces with nothing for tests
            statistical_comparisons[i] = s
        df_x = df_x.apply(make_numeric, measure=measure, axis=1)
        df_clean = df_x.dropna(subset=[measure], inplace=False)
        # print("group by: ", group_by)
        # print(df_clean[measure])
        # print("Groups found: ", df_clean.Group.unique())
        scale = 1.0
        facs = {"AP_HW": 1e6, "taum": 1e3}
        if measure in facs:
            scale = facs[measure]
        CP("c", "\n==================================================")
        # CP("c", f"Statistics are based on ALL ")
        print(f"Celltype:: {celltype:s}, Measure: {measure:s}")
        print("Subject, Group, sex, Value\n-----------------------------------------------")
        for cell in sorted(list(df_clean.cell_id.values)):
            print(f"{cell:s}, {df_clean[df_clean.cell_id == cell].Group.values[0]:s},", end=" ")
            print(f"{df_clean[df_clean.cell_id == cell].sex.values[0]:s},", end="")
            print(f"{scale*df_clean[df_clean.cell_id == cell][measure].values[0]!s}")

        groups_in_data = df_clean[group_by].unique()
        # print("Groups found in data: ", groups_in_data, len(groups_in_data))

        if len(groups_in_data) == 1 and group_by == "age_category":  # apply new grouping
            # df_clean[group_by] = df_clean.apply(self.rename_group, group_by=group_by, axis=1)
            df_clean = df_clean.apply(self.categorize_ages, axis=1)
        # print("2: ", df_clean[measure])
        if len(groups_in_data) < 2:  # need 2 groups to compare
            nodatatext = "\n".join(
                [
                    "",
                    CP(
                        "r",
                        f"****** Insufficient data for {celltype:s} and {measure:s}",
                        textonly=True,
                    ),
                ]
            )
            FUNCS.textappend(nodatatext)
            return
        # print(df_clean[group_by].unique())
        if parametric:
            # if measure == "maxHillSlope":
            #     df_clean["maxHillSlope"] = np.log(df_clean["maxHillSlope"])
            lm = sfa.ols(f"{measure:s} ~ {group_by:s}", data=df_clean).fit()
            anova = sa.stats.anova_lm(lm)
            in8 = " " * 8
            msg = f"\nANOVA: [{measure:s}]  celltype={celltype:s}\n\n {textwrap.indent(str(anova), in8)!s}"
            stattext = "".join(["", CP("y", msg, textonly=True)])
            print(stattext)
            FUNCS.textappend(stattext)

            lmt = f"\nOLS: \n{textwrap.indent(str(lm.summary()),in8)!s}"
            stattext = "\n".join(["", CP("b", f"{lmt!s}", textonly=True)])
            print(stattext)
            FUNCS.textappend(stattext)
            # ph = sp.posthoc_ttest(df_clean, val_col=measure, group_col="Group", p_adjust="holm")
            # print(ph)

            # res = stat()
            # res.anova_stat(df=df_pyr, res_var='dvdt_rising', anova_model='dvdt_rising ~ C(Group)')
            # print(res.anova_summary)

            # sh = pairwise_tukeyhsd(df[measure], df["Group"])
            # print(sh)

            try:
                pw = lm.t_test_pairwise(group_by, method="sh")
                pwt = f"\nPairwise tests:\n{textwrap.indent(str(pw.result_frame), in8)!s}"
                stattext = "\n".join(["", CP("b", f"{pwt!s}", textonly=True)])
                FUNCS.textappend(stattext)
                print(stattext)

                subdf = pw.result_frame.loc[statistical_comparisons]
                mtests = multipletests(subdf["P>|t|"].values, method="holm-sidak")  # "sh")
                stattext = "\n".join(
                    [
                        "MultiComparison Tests:\n",
                        CP("b", f"{textwrap.indent(str(mtests), in8)!s}", textonly=True),
                    ]
                )
                subdf["adj_p"] = textwrap.indent(str(mtests[1]), in8)
                stattext = "\n".join([stattext, CP("b", f"{subdf!s}", textonly=True)])
                FUNCS.textappend(stattext)
                print(stattext)

            except (KeyError, ValueError) as error:
                msg = f"****** Missing data for comparisons for {celltype:s} and {measure:s}\n{error!s}"
                stattext = "\n".join([msg])
            # # res = stat()
            # # res.tukey_hsd(df=df_pyr, res_var='dvdt_rising', xfac_var='Group', anova_model='dvdt_rising ~ C(Group)')
            # # print(res.tukey_summary)
            # # print("=" * 40)
            stattext = "\n".join([stattext, f"{'='*80:s}"])

        if nonparametric:
            # nonparametric:
            msg = f"KW: [{measure:s}]  celltype={celltype:s}\n\n "
            stattext = "".join(["", CP("y", msg, textonly=True)])
            groups_in_data = df_clean[group_by].unique()

            print("\nMeasure: ", measure)
            print("Groups in this data: ", groups_in_data)

            data = []
            dictdata = {group: [] for group in groups_in_data}
            for group in groups_in_data:
                dg = df_clean[df_clean[group_by] == group]
                # print("group: ", group, "measure: ", measure, "dgmeasure: ", dg[measure].values)
                dv = []
                for d in dg[measure].values:
                    dv.append(np.nanmean(d))
                # I know this is not the most efficient way to do this, but it is fast enough.
                dv = [
                    d for d in dv if ~np.isnan(d)
                ]  # clean out nan's. Should not be necessary, but MW/KW doesn't like them.
                data.append(dv)
                dictdata[group].append(dv)

            if len(data) < 2:
                stattext = "\n".join(
                    [
                        stattext,
                        f"Too few groups to compuare: celltype={celltype:s}  measure={measure:s}",
                    ]
                )
                FUNCS.textappend(stattext)
                return df_clean

            print("  Descriptive Statistics: ")
            # print(dictdata.keys())
            desc_stat = ""
            for gr in dictdata.keys():  # gr is the group (e.g., genotype, treatment, etc)
                # print(dictdata[gr], len(dictdata[gr][0]))
                desc_stat += f"Group: {gr:s}  N: {np.sum(~np.isnan(dictdata[gr])):d}, median: {scale*np.nanmedian(dictdata[gr]):.6f},"
                desc_stat += f"mean: {scale*np.nanmean(dictdata[gr]):.6f}"
                desc_stat += f"  std: {scale*np.nanstd(dictdata[gr]):.6f}, QR: {scale*np.nanquantile(dictdata[gr], 0.25):.6f}"
                desc_stat += f"- {scale*np.nanquantile(dictdata[gr], 0.75):.6f}"
                desc_stat += f" IQR: {scale*(np.nanquantile(dictdata[gr], 0.25)- np.nanquantile(dictdata[gr], 0.75)):.6f}\n"
            FUNCS.textappend(desc_stat)
            print(desc_stat)
            # print("")
            # print(len(groups_in_data), " groups in data", len(data))
            if len(groups_in_data) == 2:
                s, p = scipy.stats.mannwhitneyu(*data)
                stattext = "\n".join(
                    [
                        stattext,
                        CP(
                            "y",
                            f"Mann-Whitney U: U:{s:.6f}   p={p:.6f}\n",
                            textonly=True,
                        ),
                    ]
                )
            else:
                s, p = scipy.stats.kruskal(*data)
                stattext = "\n".join(
                    [
                        "",
                        CP(
                            "y",
                            f"Kruskal-Wallis: H:{s:.6f}   p={p:.6f}\n",
                            textonly=True,
                        ),
                    ]
                )
                print(f"Kruskal-Wallis: H:{s:.6f}   p={p:.6f}\n")
                posthoc = scikit_posthocs.posthoc_dunn(
                    df_clean, val_col=measure, group_col=group_by, p_adjust="holm"
                )
                # print(posthoc)
                stattext = "\n".join(
                    [
                        stattext,
                        CP(
                            "y",
                            f"Dunn posthoc:\n{posthoc!s}\n",
                            textonly=True,
                        ),
                    ]
                )
            FUNCS.textappend(stattext)
            print(stattext)
        return df_clean

    def check_HW(self, row):
        if row.AP_HW < 0 or row.AP_HW > 0.010:
            print(
                f"HW: {row.AP_HW:15.5f}, ID: {Path(row.cell_id).name!s}, Type: {row.cell_type:16s}, Group: {row.Group:5s}"
            )

    def check_APthr(self, row):
        print(
            f"APthr: {row.AP_thr_V:15.5f}, ID: {Path(row.cell_id).name!s}, Type: {row.cell_type:16s}, Group: {row.Group:5s}"
        )
        if "LowestCurrentSpike" in row.keys():
            print("LowestCurrentSpike: ", row.LowestCurrentSpike)

    def check_Group(self, row):
        print(
            f"{row.AP_thr_V:15.5f}, {Path(row.cell_id).name!s}, {row.cell_type:16s}, {row.Group:5s}"
        )

    def assign_default_Group(row):
        age = get_age(row.Age)
        if pd.isnull(row.Group) or row.Group == "nan":
            if age <= 20:
                row.Group = "Preweaning"
            elif age >= 21 and age < 50:
                row.Group = "Pubescent"
            elif age >= 50 and age < 180:
                row.Group = "Young Adult"
            elif age >= 180:
                row.Group = "Mature Adult"
        return row

    def _data_complete_to_series(self, row):
        dc = row.data_complete.split(",")
        dc = [p.strip(" ") for p in dc if p != "nan" and "CCIV".casefold() in p.casefold()]
        # print("\ndc: ", dc)
        row.protocol = pd.Series(dc)
        # print(row.date, row.data_complete.values)
        return row

    def get_assembled_filename(self, experiment):
        """get_assembled_filename Create the filename for the assembled FI data.

        Parameters
        ----------
        experiment : _type_
            _description_

        Returns
        -------
        _type_
            _description_
        """
        assembled_fn = Path(
            Path(
                experiment["analyzeddatapath"],
                experiment["directory"],
                experiment["assembled_filename"],
            )
        )
        return assembled_fn

    def assemble_datasets(
        self,
        df_summary: pd.DataFrame,
        fn: str = "",
        coding_file: Optional[str] = None,
        coding_sheet: Optional[str] = None,
        coding_level: Optional[str] = None,
        exclude_unimportant: bool = False,
    ):
        """assemble_datasets : Assemble the datasets from the summary and coding files,
        then combine FI curves (selected) in IV protocols for each cell.

        Parameters
        ----------
        df_summary : pd.DataFrame
            _description_
        coding_file : Optional[str], optional
            _description_, by default None
        coding_sheet : Optional[str], optional
            _description_, by default None
        coding_level : Optional[str], optional
            _description_, by default None
        exclude_unimportant : bool, optional
            _description_, by default False
        fn : str, optional
            _description_, by default ""
        """
        print(
            f"Assembling:\n  coding file: {coding_file!s}\n    Cells: {self.experiment['celltypes']!s}"
        )
        df = self.combine_summary_and_coding(
            df_summary=df_summary,
            coding_file=coding_file,
            coding_sheet=coding_sheet,
            coding_level=coding_level,
            exclude_unimportant=exclude_unimportant,
        )
        if "protocol" not in df.columns:
            df["protocol"] = ""
        df = df.apply(self._data_complete_to_series, axis=1)
        print(len(df), " rows after data complete to series")

        # now make a new dataframe that has a separate row for each protocol
        df = df.explode(["protocol"], ignore_index=True)
        print("Number of protocols after explode", len(df))
        df = df.dropna(subset=["protocol"])
        print("Number of protocols after dropna", len(df))

        df_null = df[df["cell_id"].isnull()]
        print("Null columns: ", df_null)
        df = df.dropna(subset=["cell_id"])
        print("# of protocols with ID: ", len(df))
        protostrings = "|".join(list(self.experiment["protocols"]["CCIV"].keys()))
        print("protostrings: ", protostrings)
        print("Protocols: ", df["protocol"].unique())

        df = self.combine_by_cell(df)
        print("\nWriting assembled data to : ", fn)
        print("Assembled groups: dataframe Groups: ", df.Group.unique())
        df.to_pickle(fn)

    def categorize_ages(self, row):
        row.age = numeric_age(row)
        for k in self.experiment["age_categories"].keys():
            if (
                row.age >= self.experiment["age_categories"][k][0]
                and row.age <= self.experiment["age_categories"][k][1]
            ):
                row.age_category = k
        return row.age_category

    def clean_sex_column(self, row):
        if row.sex not in ["F", "M"]:
            row.sex = "U"
        return row.sex

    def compute_AHP_depth(self, row):
        # Calculate the AHP depth, as the voltage between the the AP threshold and the AHP trough
        # if the depth is positive, then the trough is above threshold, so set to nan.
        if "LowestCurrentSpike" not in row.keys():
            # This is the first assignment/caluclation of AHP_depth_V, so we need to make sure
            # it is a list of the right length
            if isinstance(row.AP_thr_V, float):
                row.AP_thr_V = [row.AP_thr_V]
            if isinstance(row.AHP_trough_V, float):
                row.AHP_trough_V = [row.AHP_trough_V]
            row["AHP_depth_V"] = [np.nan] * len(row.AP_thr_V)
            for i, apv in enumerate(row.AHP_trough_V):
                row.AHP_depth_V[i] = row.AHP_trough_V[i] - row.AP_thr_V[i]
                if row.AHP_depth_V[i] > 0:
                    row.AHP_depth_V[i] = np.nan
            return row.AHP_depth_V[i]
        else:
            CP("c", "LowestCurrentSpike in row keys")

    def compute_AHP_trough_time(self, row):
        # RE-Calculate the AHP trough time, as the time between the AP threshold and the AHP trough
        # if the depth is positive, then the trough is above threshold, so set to nan.
        # print(len(row.AHP_trough_T), len(row.AP_thr_T))
        if isinstance(row.AP_thr_T, float):
            row.AP_thr_T = [row.AP_thr_T]
        if isinstance(row.AHP_trough_T, float):
            row.AHP_trough_T = [row.AHP_trough_T]
        for i, att in enumerate(row.AP_thr_T):  # base index on threshold measures
            # print(row.AHP_trough_T[i], row.AP_thr_T[i])  # note AP_thr_t is in ms, AHP_trough_T is in s
            row.AHP_trough_T[i] = row.AHP_trough_T[i] - row.AP_thr_T[i] * 1e-3
            if row.AHP_trough_T[i] < 0:
                row.AHP_trough_T[i] = np.nan
        return row.AHP_trough_T

    def get_animal_id(self, row, df_summary):
        """get_animal_id get the animal ID from df_summary
        The result goes in the "Subject" column of the calling dataframe
        The value in df_summary is "animal_identifier"
        The animal ID is the value in the cell_id column of the calling dataframe

        This is meant to be called in a df.apply... statement
        Parameters
        ----------
        row : pandas Series
            current row
        df_summary : pandas DataFrame
            The summary dataframe generated from dataSummary.py

        Returns
        -------
        animal_id / subject
            string

        Raises
        ------
        ValueError
            _description_
        ValueError
            _description_
        """
        cell_id = row.cell_id
        cell_id_match = FUNCS.compare_cell_id(cell_id, df_summary.cell_id.values)
        if cell_id_match is None:
            return ""  # no match, leave empty
        if cell_id_match is not None:  # we have a match, so save as the Subject
            # handle variations in the column name (historical changes)
            idname = "Subject"
            row[idname] = df_summary.loc[df_summary.cell_id == cell_id_match][
                "animal_identifier"
            ].values[0]
            # else:
            #     print("row keys: ", sorted(row.keys()))
            #     if "Subject" in row.keys():
            #         print("Found subject column but not animal[_]identifier: ", row["Subject"])
            #     raise ValueError("could not match animal id/Subject with column")
        return row[idname]

    def get_cell_layer(self, row, df_summary):
        cell_id = row.cell_id
        cell_id_match = FUNCS.compare_cell_id(cell_id, df_summary.cell_id.values)
        if cell_id_match is None:
            return ""
        if cell_id_match is not None:
            row.cell_layer = df_summary.loc[df_summary.cell_id == cell_id_match].cell_layer.values[
                0
            ]
            if row.cell_layer in [" ", "nan"]:
                row.cell_layer = "unknown"
        else:
            print("cell id not found: ", cell_id)
            print("values: ", df_summary.cell_id.values.tolist())
            raise ValueError(f"Cell id {cell_id} not found in summary")
        return row.cell_layer

    def get_cell_expression(self, row, df_summary):
        """get_cell_expression from the main datasummary dataframe

        Parameters
        ----------
        row : pandas dataframe row
            row associated with a cell
        df_summary : pandas dataframe
            main dataframe from the datasummary program

        Returns
        -------
        pandas row
            updated row

        Raises
        ------
        ValueError
            error if cannot match the cell id in the current row with the summary.
        """
        cell_id = row.cell_id
        cell_id_match = FUNCS.compare_cell_id(cell_id, df_summary.cell_id.values)
        if cell_id_match is None:
            CP("y", f"get cell expression: cell id match is None: {cell_id:s}, \n{df_summary.cell_id.values!s}")
            return ""
        # print("cell id match ok: ", cell_id)
        if cell_id_match is not None:
            row.cell_expression = df_summary.loc[
                df_summary.cell_id == cell_id_match
            ].cell_expression.values[0]

            if row.cell_expression in [" ", "nan", "NaN", np.nan] or pd.isnull(row.cell_expression):
                row.cell_expression = "ND"
            if "remove_expression" in self.experiment.keys():
                if row.cell_expression in self.experiment["remove_expression"]:
                    for re in self.experiment["remove_expression"]:
                        if row.cell_expression == re:
                            row.cell_expression = 'ND'
        else:
            print("cell id not found: ", cell_id)
            print("values: ", df_summary.cell_id.values.tolist())
            raise ValueError(f"Cell id {cell_id} not found in summary")
        return row.cell_expression

    def preload(self, fn):
        """preload Load the assembled data from a .pkl file
        Does some clean up of designators and data,
        and adds some columns from the datasummary if they are missing

        Parameters
        ----------
        fn : function
            _description_

        Returns
        -------
        _type_
            _description_
        """
        CP("g", f"    PRELOAD, {fn!s}")
        df = pd.read_pickle(fn)
        df_summary = get_datasummary(self.experiment)
        df = self.preprocess_data(df, self.experiment)
        return df

    def preprocess_data(self, df, experiment):
        pd.options.mode.copy_on_write = True
        """preprocess_data Clean up the data, add columns, etc."""
        df_summary = get_datasummary(experiment)
        # print("   Preprocess_data: df_summary column names: ", sorted(df_summary.columns))
        # print("df summary ids: ")
        # for cid in df_summary.cell_id:
        #     print("    ",cid)
        # print(" ")
        # print("df ids: ")
        # for cid in df.cell_id:
        #     print("    ", cid)
        # print("*"*80)

        # print("   Preprocess_data: df column names: ", sorted(df.columns))
        # print("df: ")
        # for d in df.to_dict(orient="records"):
        #     print(f"<{d['cell_id']!s}, type(d['cell_id'])")
        

        # print("summary: ")
        # for d in df_summary.to_dict(orient="records"):
            # print(f"<{d['cell_id']!s}, ty")
        # note df will have "short" names: Rig2(MRK)/L23_intrinsic/2024.10.22_000_S0C0
        # df_summary will have long names: Rig2(MRK)/L23_intrinsic/2024.10.22_000/slice_000/cell_000
        # print("-2 ", df['cell_id'].eq("Rig2(MRK)/L23_intrinsic/2024.10.22_000_S0C0").any())
        # print("-1 ", df_summary['cell_id'].eq("Rig2(MRK)/L23_intrinsic/2024.10.22_000_S0C0").any())
 
        df["Subject"] = df.apply(self.get_animal_id, df_summary=df_summary, axis=1)

        # print("df idsid df after getting Subject: ")
        # for cid in df.cell_id:
        #     print("    ", cid)
        # print("*"*80)

        if "cell_layer" not in df.columns:
            layers = df_summary.cell_layer.unique()
            if len(layers) == 1 and layers == [" "]:  # no layer designations
                df["cell_layer"] = "unknown"
            else:
                df["cell_layer"] = ""
                df["cell_layer"] = df.apply(self.get_cell_layer, df_summary=df_summary, axis=1)

        # print("1!  prior to checking expression: ", df['cell_id'].eq("Rig2(MRK)/L23_intrinsic/2024.10.22_000_S0C0").any())
        if "cell_expression" not in df.columns:
            expression = df_summary.cell_expression.unique()
            if len(expression) == 1 and expression == [" "]:  # no expressiondesignations
                df["cell_expression"] = "ND"
            else:
                df["cell_expression"] = ""
                df["cell_expression"] = df.apply(
                    self.get_cell_expression, df_summary=df_summary, axis=1
                )
            print("   Preprocess_data: cell expression values: ", df.cell_expression.unique())


        gu = df.Group.unique()
        # print("0!  ", df['cell_id'].eq("Rig2(MRK)/L23_intrinsic/2024.10.22_000_S0C0").any())

#  ******************************************************************************
        CP("c", "\n   Preprocess_data: Groups and cells PRIOR to exclusions: ")
        print("   Preprocess_data: Groups: ", gu)
        # gu_nonan = [g for g in gu if pd.notnull(g)]
        for g in gu:
            print(f"    {g!s}  (N={len(df.loc[df.Group == g]):d})")
            for x in df.loc[df.Group == g].cell_id:
                print("        ", x)
            # if g in ["A", "AA"] and i0 < 20:
            #         print("        ", df.loc[df.cell_id == x].I_maxHillSlope.values)
            #         i0 += 1
        # print("=" * 80)
        df["sex"] = df.apply(self.clean_sex_column, axis=1)
        df["Rin"] = df.apply(self.clean_rin, axis=1)
        df["RMP"] = df.apply(self.clean_rmp, axis=1)
        if "RMP_Zero" in df.columns:
            df["RMP_Zero"] = df.apply(self.clean_rmp_zero, axis=1)
        else:
            df["RMP_Zero"] = np.nan  # not determined...
        if "age_category" not in df.columns:
            df["age_category"] = np.nan
        # print("2 ", df['cell_id'].eq("Rig2(MRK)/L23_intrinsic/2024.10.22_000_S0C0").any())
        df["age_category"] = df.apply(self.categorize_ages, axis=1)
        df["FIRate"] = df.apply(self.getFIRate, axis=1)
        df["Group"] = df["Group"].astype("str")
        if "FIMax_4" not in df.columns:
            df["FIMax_4"] = np.nan
        if "AHP_depth_V" not in df.columns:
            df["AHP_depth_V"] = np.nan
        df["AHP_depth_V"] = df.apply(self.compute_AHP_depth, axis=1)
        # print("3 ", df['cell_id'].eq("Rig2(MRK)/L23_intrinsic/2024.10.22_000_S0C0").any())
  
        if "AHP_trough_V" not in df.columns:
            df["AHP_trough_V"] = np.nan
        df["AHP_trough_T"] = df.apply(self.compute_AHP_trough_time, axis=1)
        if len(df["Group"].unique()) == 1 and df["Group"].unique()[0] == "nan":
            if self.experiment["set_group_control"]:
                df["Group"] = "Control"
        # print("4 ", df['cell_id'].eq("Rig2(MRK)/L23_intrinsic/2024.10.22_000_S0C0").any())
        # print("df ids after getting all cleaning 1: ")
        # for cid in df.cell_id:
        #     print("    ", cid)
        # print("*"*80)

        groups = df.Group.unique()
        # print("   Preprocess_data:  Groups: ", groups)
        expressions = df.cell_expression.unique()
        # print("   Preprocess_data:  Expression: ", expressions)
        # print("   experiment keys: ", self.experiment.keys())
        if (
            "remove_expression" in self.experiment.keys()
            and self.experiment["remove_expression"] is not None
        ):
            print("REMOVING specific cell_expression")
            for expression in self.experiment["remove_expression"]:  # expect a list
                df = df[df.cell_expression != expression]
                print("   Preprocess_data: Removed expression: ", expression)
        # print("5: expression removed", df['cell_id'].eq("Rig2(MRK)/L23_intrinsic/2024.10.22_000_S0C0").any())
        

        # self.data_table_manager.update_table(data=df)
        if "groupname" not in df.columns:
            df["groupname"] = np.nan
        df["groupname"] = df.apply(rename_groups, experiment=self.experiment, axis=1)
        # print("5.0: groups removed ", df['cell_id'].eq("Rig2(MRK)/L23_intrinsic/2024.10.22_000_S0C0").any())
        if len(groups) > 1:
            # df.dropna(subset=["Group"], inplace=True)  # remove empty groups
            df.drop(df.loc[df.Group == "nan"].index, inplace=True)
            # print("5.2: groups removed ", df['cell_id'].eq("Rig2(MRK)/L23_intrinsic/2024.10.22_000_S0C0").any())
        print(
            "           Preprocess_data: # Groups found after dropping nan: ",
            df.Group.unique(),
            len(df.Group.unique()),
        )
        # print("5.10: groups removed ", df['cell_id'].eq("Rig2(MRK)/L23_intrinsic/2024.10.22_000_S0C0").any())
        # raise ValueError()
        df["age"] = df.apply(numeric_age, axis=1)
        df["shortdate"] = df.apply(make_datetime_date, axis=1)
        df["SR"] = df.apply(self.flag_date, axis=1)
        # print("5: age/sr/shortdate ", df['cell_id'].eq("Rig2(MRK)/L23_intrinsic/2024.10.22_000_S0C0").any())
    
        # print("        # entries in dataframe after filtering: ", len(df))
        # if "cell_id2" not in df.columns:
        #     df["cell_id2"] = df.apply(make_cell_id2, axis=1)
        # print("cell ids: \n", df.cell_id)
        if len(self.experiment["excludeIVs"]) > 0:
            CP("c", "Parsing Excluded IV datasets (noise, etc)")
            # print(self.experiment["excludeIVs"])
            for filename, key in self.experiment["excludeIVs"].items():
                fparts = Path(filename).parts
                fn = str(Path(*fparts[-3:]))
                if len(fparts) > 3:
                    fnpath = str(Path(*fparts[:-3]))  # just day/slice/cell
                else:
                    fnpath = None  # no leading path
                reason = key["reason"]
                re_day = re.compile(r"(\d{4})\.(\d{2})\.(\d{2})\_(\d{3})$")
                re_slice = re.compile(r"(\d{4})\.(\d{2})\.(\d{2})\_(\d{3})\/slice_(\d{3})$")
                re_slicecell = re.compile(
                    r"(\d{4})\.(\d{2})\.(\d{2})\_(\d{3})\/slice_(\d{3})\/cell_(\d{3})$"
                )
                # get slice and cell nubmers
                re_slicecell2 = re.compile(
                    r"^(?P<year>\d{4})\.(?P<month>\d{2})\.(?P<day>\d{2})\_(?P<dayno>\d{3})\/slice_(?P<sliceno>\d{3})\/cell_(?P<cellno>\d{3})$"
                )

                # print("   Preprocess_data: Checking exclude for listed exclusion ", filename)
                dropped = False

                if re_day.match(fn) is not None:  # specified a day, not a cell:
                    df.drop(df.loc[df.cell_id.str.startswith(fn)].index, inplace=True)
                    CP(
                        "r",
                        f"   Preprocess_data: dropped DAY {fn:s} from analysis, reason = {reason:s}",
                    )
                    dropped = True
                elif re_slice.match(fn) is not None:  # specified day and slice
                    fns = re_slice.match(fn)
                    df.drop(df.loc[df.cell_id.str.startswith(fns)].index, inplace=True)
                    CP(
                        "r",
                        f"   Preprocess_data: dropped SLICE {fn:s} from analysis, reason = {reason:s}",
                    )
                    dropped = True
                elif re_slicecell.match(fn) is not None:  # specified day, slice and cell
                    fnc = re_slicecell2.match(fn)
                    # generate an id with 1 number for the slice and 1 for the cell,
                    # test variations with _ between S and C as well
                    fn1 = f"{fnc['year']:s}.{fnc['month']:s}.{fnc['day']:s}_{fnc['dayno']:s}_S{int(fnc['sliceno']):1d}C{int(fnc['cellno']):1d}"
                    fn1a = f"{fnc['year']:s}.{fnc['month']:s}.{fnc['day']:s}_{fnc['dayno']:s}_S{int(fnc['sliceno']):1d}_C{int(fnc['cellno']):1d}"
                    fn2 = f"{fnc['year']:s}.{fnc['month']:s}.{fnc['day']:s}_{fnc['dayno']:s}_S{int(fnc['sliceno']):02d}C{int(fnc['cellno']):02d}"
                    fn2a = f"{fnc['year']:s}.{fnc['month']:s}.{fnc['day']:s}_{fnc['dayno']:s}_S{int(fnc['sliceno']):02d}_C{int(fnc['cellno']):02d}"
                    fns = [fn1, fn1a, fn2, fn2a]
                    # print(df.cell_id.unique())
                    dropped = False
                    for i, f in enumerate(fns):
                        if fnpath is not None:
                            fns[i] = str(Path(fnpath, f))  # add back the leading path
                        if not df.loc[df.cell_id == fns[i]].empty:
                            df.drop(df.loc[df.cell_id == fns[i]].index, inplace=True)
                            CP(
                                "m",
                                f"   Preprocess_data: dropped CELL {fns[i]:s} from analysis, reason = {reason:s}",
                            )
                            dropped = True
                        elif not dropped:
                            pass
                            # CP(
                            #     "r",
                            #     f"   Preprocess_data: CELL {fns[i]:s} not found in data set (may already be excluded by prior analysis)",
                            # )
                elif not dropped:
                    CP("y", f"   Preprocess_data: {filename:s} not dropped, but was found in exclusion list")
                else:
                    CP("y", f"   Preprocess_data: No exclusions found for {filename:s}")
        gu = df.Group.unique()
        print("   Preprocess_data: Groups and cells after exclusions: ")

        # for g in sorted(gu):
        #     print(f"    {g:s}  (N={len(df.loc[df.Group == g]):d})")
        #     i0 = 0
        #     for x in df.loc[df.Group == g].cell_id:
        #         print("        ", x)
        #     # if g in ["A", "AA"] and i0 < 20:
        #     #         print("        ", df.loc[df.cell_id == x].I_maxHillSlope.values)
        #     #         i0 += 1
        # print("=" * 80)
        # now apply any external filters that might be specified in the configuration file
        if "filters" in self.experiment.keys():
            print("   Preprocess_data: Filters is set: ")
            for key, values in self.experiment["filters"].items():
                print("      Preprocess_data: Filtering on: ", key, values)
                df = df[df[key].isin(values)]

        # print("age categories: ", df.age_category.unique())
        # print("preload returns with Group list: ", df.Group.unique())
        # raise ValueError('preprocess')
        return df

    def do_stats(
        self, df, experiment, group_by, second_group_by, textbox: object = None, divider="-" * 80
    ):
        if textbox is not None:
            FUNCS.textbox_setup(textbox)
            FUNCS.textclear()
        df = self.preprocess_data(df, experiment)
        # Remove cells for which the FI Hill slope is maximal at 0 nA:
        #    These have spont.
        df = df[df["I_maxHillSlope"] > 1e-11]
        for ctype in experiment["celltypes"]:
            CP("g", f"\n{divider:s}")
            for measure in [
                "dvdt_rising",
                "dvdt_falling",
                "AP_thr_V",
                "AP_HW",
                "AHP_depth_V",
                "AHP_trough_T",
                # "AP15Rate",
                "AdaptRatio",
                # "FISlope",
                "maxHillSlope",
                "I_maxHillSlope",
                "FIMax_1",
                # "FIMax_4",
                "RMP",
                "RMP_Zero",
                "Rin",
                "taum",
            ]:
                df_clean = (
                    self.stats(
                        df,
                        celltype=ctype,
                        measure=measure,
                        group_by=group_by,
                        second_group_by=second_group_by,
                        statistical_comparisons=experiment["statistical_comparisons"],
                    ),
                )
            FUNCS.textappend("=" * 80)
            subjects = df["date"].unique()
            FUNCS.textappend(f"Subjects in this data: (N={len(subjects):d})")
            # print("Subjects in this data: ")
            for s in subjects:
                FUNCS.textappend(f"    {s:s}")
            FUNCS.textappend(f"Cells in this data: (N={len(df['cell_id'].unique()):d})")
            cellsindata = df["cell_id"].unique()
            for c in cellsindata:
                FUNCS.textappend(f"    {c:s}")
            FUNCS.textappend("=" * 80)
        return


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Analyze and plot spike data")
    parser.add_argument(
        "-e",
        "--experiment",
        type=str,
        default="NIHL",
        dest="experiment",
        choices=["NF107Ai32_Het", "MF107Ai32_NIHL", "CBA_Age"],
        help="Experiment to analyze",
    )
    parser.add_argument(
        "-p",
        type=str,
        default="all",
        dest="plot",
        choices=["all", "rmtau", "spikes", "firing", "fi"],
        help="Plot to create",
    )
    parser.add_argument("-s", action="store_true", dest="stats", help="Run stats")

    parser.add_argument(
        "-a",
        "--assemble",
        action="store_true",
        dest="assemble_dataset",
        help="Assemble the dataset",
    )

    parser.add_argument(
        "-P", "--picking", action="store_true", dest="picking", help="Enable picking"
    )

    parser.add_argument(
        "--confirm",
        action="store_true",
        dest="confirm",
        help="plot fits to confirm suitability",
    )
    args = parser.parse_args()
    PSI = PlotSpikeInfo()
    fn = PSI.get_assembled_filename(args.experiment)

    # Do this AFTER running the main analysis

    if args.assemble_dataset:
        PSI.assemble_datasets(
            # excelsheet=excelsheet,
            # adddata=adddata,
            fn=fn,
        )
        exit()

    # df = read_data_files(excelsheet, adddata, analysis_cell_types)

    # Do this AFTER assembling the dataset to a pkl file
    print("Loading fn: ", fn)
    dfa = PSI.preload(fn)

    # dfa = dfa.apply(check_Group, axis=1)
    # dfa = dfa.apply(assign_default_Group, axis=1)

    # Generate stats...
    # print(
    #     "Code:\nB  : Control\nA  : 106 dBSPL 2H, 14D\nAA : 115 dBSPL 2H 14D\nAAA: 115 dBSPL 2H  3D\n"
    # )
    DIVIDER = "=" * 80 + "\n"
    if args.stats:
        PSI.do_stats(dfa, divider=DIVIDER)

    # porder = ["Preweaning", "Pubescent", "Young Adult", "Mature Adult"]
    # now we can do the plots:

    print(f"args.plot: <{args.plot:s}>")

    enable_picking = args.picking
    if args.plot in ["all", "spikes"]:
        P1, picker_funcs1 = PSI.summary_plot_spike_parameters_categorical(
            dfa,
            groups=PSI.experiment["plot_order"]["age"],
            porder=PSI.experiment["plot_order"]["age"],
            colors=PSI.experiment["plot_colors"],
            enable_picking=enable_picking,
            measures=["dvdt_rising", "dvdt_falling", "AP_thr_V", "AP_HW", "AP_Latency"],
        )
        P1.figure_handle.suptitle("Spike Shape", fontweight="bold", fontsize=18)

    if args.plot in ["all", "firing"]:
        print("main: enable-picking: ", enable_picking)
        P2, picker_funcs2 = PSI.summary_plot_spike_parameters_categorical(
            dfa,
            measures=[
                "AdaptRatio",
                "FISlope",
                "maxHillSlope",
                "I_maxHillSlope",
                "FIMax_1",
                "FIMax_4",
            ],
            porder=PSI.experiment["plot_order"],
            colors=PSI.experiment["plot_colors"],
            grouping="named",
            enable_picking=enable_picking,
        )
        P2.figure_handle.suptitle("Firing Rate", fontweight="bold", fontsize=18)

    if args.plot == "rmtau":
        P3, picker_funcs3 = PSI.summary_plot_rm_tau_categorical(
            dfa,
            groups=["Preweaning", "Pubescent", "Young Adult", "Mature Adult"],
            porder=PSI.experiment["plot_order"],
            colors=PSI.experiment["plot_colors"],
            enable_picking=enable_picking,
        )
        P3.figure_handle.suptitle("Membrane Properties", fontweight="bold", fontsize=18)

    # summary_plot_spike_parameters_continuous(df, logx=True)
    # summary_plot_RmTau_continuous(df, porder=experiment["plot_order"])

    # dft = df[df["cell_type"] == "cartwheel"]
    # p = set([Path(p).name[:-4] for protocols in dft["protocols"] for p in protocols])

    if args.plot in ["all", "fi"]:
        P4, picker_funcs4 = PSI.summary_plot_fi(
            dfa,
            mode=["individual", "mean"],
            colors=PSI.experiment["plot_colors"],
            enable_picking=enable_picking,
        )

    mpl.show()
