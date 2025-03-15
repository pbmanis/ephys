"""Plot summaries of spike and basic electrophys properties of cells.
Does stats at the end.
"""

import datetime
import pprint
from pathlib import Path
import re
from string import ascii_letters
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
import statsmodels
import statsmodels.api as sa
import statsmodels.formula.api as sfa
import statsmodels.formula.api as smf
import scipy.stats
from pylibrary.tools import cprint
import pylibrary.plotting.styler as ST
from pyqtgraph.Qt.QtCore import QObject
from statsmodels.stats.multitest import multipletests

from ephys.tools.get_computer import get_computer
import ephys.tools.show_assembled_datafile as SAD
from ephys.gui import data_table_functions
from ephys.tools import fitting, utilities

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


def concurrent_data_plotting(
    filename: str,
    mode: str,  # continous or categorical
    plot_title: str = "My Title",
    parameters: dict = None,
    data_class: str = "spike_measures",  # what kind of data to plot
    representation: str = "bestRs",
    picker_active: bool = False,
    publication_plot_mode: bool = False,
    infobox: dict = None,
    textbox: object = None,
    status_bar_message: object = None,
):
    assert mode in ["categorical", "continuous", "combined"]
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
        publication_plot_mode=publication_plot_mode,
        representation=representation,
        textbox=textbox,
    )
    print("concurrent got: textbox: ", textbox)

    if mode == "categorical":
        df = PSI_.preload(filename)
        print("categorical")
        if group_by == ['nan']:
            if status_bar_message is None:
                raise ValueError("group_by is null: please select group for categorical plot")
            else:
                status_bar_message.showMessage("Please select a group for the categorical plot", color="red")
        (
            cc_plot,
            picker_funcs1,
        ) = PSI_.summary_plot_ephys_parameters_categorical(
            df,
            xname=group_by,
            hue_category=hue_category,
            plot_order=experiment["plot_order"][group_by],
            measures=experiment[data_class],
            colors=colors,
            enable_picking=picker_active,
            representation=representation,
            publication_plot_mode=publication_plot_mode,
        )
    elif mode == "continuous":
        df = PSI_.preload(filename)
        (
            cc_plot,
            picker_funcs1,
        ) = PSI_.summary_plot_ephys_parameters_continuous(
            df,
            xname=group_by,
            hue_category=hue_category,
            plot_order=experiment["plot_order"][group_by],
            measures=experiment[data_class],
            colors=colors,
            representation=representation,
            enable_picking=False,
            publication_plot_mode=publication_plot_mode,
        )
    elif mode == "combined":

        style = ST.styler('JNeurophys', figuresize='full', height_factor=0.75)
    #        AW.style_apply(style)
        print("group by: ", group_by)
        row1_bottom = 0.7
        row2_bottom = 0.05
        vspc = 0.09
        hspc = 0.2
        ncols = len(experiment[data_class])
        up_lets = ascii_letters.upper()
        cat_labels = [up_lets[i] for i in range(ncols)]
        print("cat_labels: ", cat_labels)
        cat_figure = PH.regular_grid(cols=ncols, rows=1, order='rowsfirst', figsize=style.Figure['figsize'],
                                horizontalspacing=hspc, verticalspacing=vspc,
                                margins={'leftmargin': 0.07, 'rightmargin': 0.07, 'topmargin': 0.05, 'bottommargin': row1_bottom},
                                labelposition=(-0.15, 1.05), panel_labels=cat_labels, font="Arial",
                                fontweight=style.get_fontweights(), fontsize=style.get_fontsizes())
        cont_labels = [up_lets[i] for i in range(ncols, 2*ncols)]
        print("cont_labels: ", cont_labels)
        cont_figure = PH.regular_grid(rows=1, cols=ncols, order='rowsfirst',
                                horizontalspacing=hspc, verticalspacing=vspc,
                                margins={'leftmargin': 0.07, 'rightmargin': 0.07, 'topmargin': 0.55, 'bottommargin': row2_bottom},
                                labelposition=(-0.15, 1.05), panel_labels = cont_labels, font="Arial",
                                fontweight=style.get_fontweights(), fontsize=style.get_fontsizes(),
                                parent_figure=cat_figure)
        df = PSI_.preload(filename)
        (
            cc_plot,
            picker_funcs1,
        ) = PSI_.summary_plot_ephys_parameters_categorical(
            df=df,
            xname=group_by,
            hue_category=hue_category,
            plot_order=experiment["plot_order"][group_by],
            measures=experiment[data_class],
            plabels = cat_labels,
            colors=colors,
            enable_picking=picker_active,
            representation=representation,
            publication_plot_mode=publication_plot_mode,
            parent_figure=cat_figure,
        )
        dfc = PSI_.preload(filename)
        (
            cc2_plot,
            picker_funcs1,
        ) = PSI_.summary_plot_ephys_parameters_continuous(
            df=dfc,
            xname=group_by,
            hue_category=hue_category,
            plot_order=experiment["plot_order"][group_by],
            plabels = cont_labels,
            measures=experiment[data_class],
            colors=colors,
            representation=representation,
            enable_picking=False,
            publication_plot_mode=publication_plot_mode,
            parent_figure=cont_figure,
        )



    picked_cellid = cc_plot.figure_handle.canvas.mpl_connect(  # override the one in plot_spike_info
        "pick_event",
        lambda event: PSI_.pick_handler(event, picker_funcs1),
    )

    if not publication_plot_mode:
        plot_title += f"  ({representation:s})"
        cc_plot.figure_handle.suptitle(plot_title, fontweight="bold", fontsize=18)

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
    else:
        datetime_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cc_plot.figure_handle.text(
            0.95,
            0.01,
            datetime_str,
            fontdict={"fontsize": 6, "fontstyle": "normal", "font": "Courier"},
            verticalalignment="bottom",
            horizontalalignment="right",
        )
    cc_plot.figure_handle.show()
    mpl.show()
    return cc_plot


def concurrent_selected_fidata_data_plotting(
    filename: str,
    parameters: dict = None,
    picker_active: bool = False,
    publication_plot_mode: bool = False,
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
    "AdaptIndex",
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
    "AdaptIndex",
    "AHP_trough_V",
    "AHP_trough_T",
    "AHP_depth_V",
    "tauh",
    "Gh",
    "FiringRate",
]


"""
Each protocol has:
"Spikes: ", dict_keys(['FI_Growth', 'AdaptRatio', 'AdaptIndex',
    'FI_Curve',  'FiringRate', 
    'AP1_Latency', 'AP1_HalfWidth', 
    'AP1_HalfWidth_interpolated', 'AP2_Latency', 'AP2_HalfWidth', 'AP2_HalfWidth_interpolated',
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


def set_subject(row):
    """set_subject if subject is empty, set to date name

    Parameters
    ----------
    row : pandas dataframe row
        _description_

    Returns
    -------
    _type_pandas_dataframe_row
        _description_
    """
    # print("row subj: ", row["Subject"])
    if row["Subject"] in ["", " ", None]:
        subj = Path(row.cell_id).name
        # print("   subj: ", subj, subj[:10])
        row["Subject"] = subj[:10]
    if row["Subject"] is None:
        row["Subject"] = "NoID"
    return row["Subject"]


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
    if isinstance(row.age, str):
        if len(row.age) == 0:
            row.age = np.nan
        else:
            row.age = int("".join(filter(str.isdigit, row.age)))
        return float(row.age)
    else:
        raise ValueError(f"age is not a float or string: {row.age!s}")


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
        publication_plot_mode: bool = False,
        representation: str = "all",
        textbox: object = None,
    ):
        self.textbox = textbox
        self.set_experiment(dataset, experiment)
        self.transforms = {
            # "maxHillSlope": np.log10,
        }
        self.pick_display = pick_display
        self.pick_display_function = pick_display_function
        self.representation = representation
        self.publication_plot_mode = publication_plot_mode

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

    def set_textbox(self, textbox: object = None):
        self.textbox = textbox

    def set_pick_display_function(self, function):
        self.pick_display_function = function

    def print_for_prism(self, row, celltype="tuberculoventral"):
        if row.cell_type != celltype:
            return
        print("")
        print(row.cell_type)
        print(f"S:{row.Subject:s}")
        for i, x in enumerate(row.current):
            print(f"{x*1e9:.2f}  {row.spsec[i]:.1f}")
        print("")

    def get_stats_dir(self):
        """get_stats_dir set the directory for the statistics files (CSV etc)
        directory is pulled from the experiment configuration file (dict)
        Returns
        -------
        str with the directory name or '.' of not specified
        """
        if "stats_dir" in self.experiment.keys():
            stats_dir = self.experiment["stats_dir"]
        elif "R_statistics_directory" in self.experiment.keys():
            stats_dir = self.experiment["R_statistics_directory"]
        else:
            stats_dir = "."
        return stats_dir

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

    def fix_cell_expression(self, row):
        if (
            row.cell_expression == ""
            or len(row.cell_expression) == 0
            or row.cell_expression == "ND"
        ):
            row.cell_expression = "-"
        return row

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
        if pd.isnull(row[data]) or len(row[data]) == 0 or row[data] == "nan":
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
        edgecolor: str = "k",
        marker_size: float = 2.75,
        enable_picking: bool = True,
        publication_plot_mode: bool = False,
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
        print("bar_pts: PUB PLOT MODE: ", publication_plot_mode)
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
        # fill missing groups doesn't do anything anymore.
        # df_x = self.fill_missing_groups(df_x, xname, celltype)  # make sure emtpy groups have nan
        # print("dfx type 4: ", type(df_x))
        if "default_group" in self.experiment.keys():
            df_x = df_x.apply(
                self.clear_missing_groups,
                axis=1,
                data=xname,
                replacement=self.experiment["default_group"],
            )
        else:
            df_x = df_x.apply(self.clear_missing_groups, axis=1, data=xname)
        # print("dfx type 5: ", type(df_x))
        # print("uniques: ", df_x[xname].unique())
        df_x.dropna(subset=[xname], inplace=True)
        # CP(
        #     "y",
        #     f"      b_rpts: Celltype: {celltype:s}, Groups: {df_x.Group.unique()!s} expression: {df_x.cell_expression.unique()!s}",
        # )
        # fix cell expression when it is empty (no string). This means "negative".
        if "cell_expression" in df_x.columns:
            df_x = df_x.apply(self.fix_cell_expression, axis=1)

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

        if hue_category == "sex":
            #######
            ### Technically, this should ONLY be set from the configuration file.
            #######
            # print("setting hue_palette via sex")
            hue_palette = {
                "F": "#FF000088",
                "M": "#0000ff88",
                " ": "k",
                "AIE": "#444488FF",
                "CON": "#9999ddFF",
            }
        # elif hue_category == "temperature":
        #     hue_palette = {"22": "#0000FF88", "34": "#FF000088", " ": "#888888FF"}
        out_of_bounds_markers = "^"  #  for h in hue_order]
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
                size=marker_size,
                # fliersize=None,
                alpha=1.0,
                ax=ax,
                palette=hue_palette,
                edgecolor=edgecolor,
                linewidth=0.5,
                order=plot_order,
                hue_order=plot_order,
                picker=enable_picking,
                zorder=100,
                clip_on=False,
            )

        else:
            # main strip plot, but data are clipped to axes
            sns.stripplot(
                x=xname,
                y=yname,
                hue=hue_category,
                data=df_x,
                order=plot_order,
                hue_order=hue_order,
                dodge=dodge,
                size=marker_size,
                # fliersize=None,
                jitter=0.25,
                alpha=1.0,
                ax=ax,
                palette=hue_palette,
                edgecolor=edgecolor,
                linewidth=0.5,
                picker=enable_picking,
                zorder=100,
                clip_on=True,
            )
            # put "out of bounds markers" on the plot at the top and bottom of the axes
            ymax = ax.get_ylim()
            df_outbounds = df_x[df_x[yname] > ymax[1]]
            df_outbounds[yname] = ymax[1] + 0.025 * (ymax[1] - ymax[0])
            # print("out of bounds: ", df_outbounds)
            # print("ymax: ", ymax)

            sns.stripplot(
                x=xname,
                y=yname,
                hue=hue_category,
                data=df_outbounds,
                order=plot_order,
                hue_order=hue_order,
                marker=out_of_bounds_markers,
                dodge=dodge,
                size=marker_size,
                # fliersize=None,
                jitter=0.25,
                alpha=1.0,
                ax=ax,
                palette=hue_palette,
                edgecolor=edgecolor,
                linewidth=0.5,
                picker=enable_picking,
                zorder=100,
                clip_on=False,
            )

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

        angle = 45
        ha = "right"
        if publication_plot_mode:
            angle = 0
            ha = "center"
            xlab = ax.get_xlabel()
            if self.experiment["new_xlabels"] is not None:
                if xlab in self.experiment["new_xlabels"]:
                    xlab = self.experiment["new_xlabels"][xlab]
                    ax.set_xlabel(xlab)
        ax.set_xticks(ax.get_xticks(), ax.get_xticklabels(), rotation=angle, ha=ha)
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
            "AP_thr_V": 1e3,
            "AHP_depth_V": 1e3,
            "AHP_trough_V": 1e3,
            "AHP_trough_T": 1e3,
            "FISlope": 1e-9,
            "maxHillSlope": 1,
            "I_maxHillSlope": 1e-3,
            "dvdt_falling": -1.0,
            "taum": 1e3,
            "Rs": 1e-6,
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
            "AdaptIndex",
            "FIMax_1",
            "FIMax_4",
            "maxHillSlope",
            "I_maxHillSlope",
        ]
        df = SAD.populate_columns(
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
        df_R = SAD.perform_selection(
            select_by=select_by,
            select_limits=[0, 1e9],
            data=df_R,
            parameters=measures,
            configuration=self.experiment,
        )
        fn = self.get_stats_dir()
        filename = Path(fn, filename)
        CP("g", f"Exporting analyzed data to {filename!s}")
        df_R.to_csv(filename, index=False)
        # now remove brackets from the csv data, then rewrite
        with open(filename, "r") as file:
            data = file.read().replace(",[", ",").replace("],", ",")

        with open(filename, "w") as file:
            file.write(data)
    
    def create_plot_figure(self, df, xname, measures: Optional[list] = None,
                           parent_figure: Optional[mpl.figure] = None):
        nrows = len(self.experiment["celltypes"])
        bottom_margin = 0.1
        top_margin = 0.15
        height = 2.5
        edgecolor = "k"
        if self.experiment["celltypes"][0] in df[xname].unique():
            nrows = 1  # doing by cell type, not some other category.
        if nrows == 1:
            bottom_margin = 0.25  # add a little more when just one row
            top_margin = 0.2
            height = 1.75
            edgecolor = "w"
        ncols = len(measures)
        letters = ascii_letters.upper()  # ["A", "B", "C", "D", "E", "F", "G", "H"]
        if nrows > 1:  # use A1, A2, etc for columns, and letters for rows
            plabels = [f"{let.upper():s}{num+1:d}" for let in letters for num in range(ncols)]
        else:  # just use A,
            plabels = [f"{let.upper():s}" for let in letters]
        figure_width = ncols * 2.5
        P = PH.regular_grid(
            nrows,
            ncols,
            order="rowsfirst",
            figsize=(figure_width, height * nrows + 1.0),
            panel_labels=plabels,
            labelposition=(-0.15, 1.05),
            margins={
                "topmargin": top_margin,
                "bottommargin": bottom_margin,
                "leftmargin": 0.1,
                "rightmargin": 0.15,
            },
            verticalspacing=0.1,
            horizontalspacing=0.06,
            parent_figure=parent_figure,
        )
        if nrows > 1:
            self.label_celltypes(P, analysis_cell_types=self.experiment["celltypes"])
        for ax in P.axdict:
            PH.nice_plot(P.axdict[ax], direction="outward", ticklength=3, position=-0.03)
        return P, letters, plabels, cols, nrows

    def summary_plot_ephys_parameters_categorical(
        self,
        df,
        xname: str,
        hue_category=None,
        plot_order: Optional[list] = None,
        plabels: Optional[list] = None,
        measures: Optional[list] = None,
        representation: str = "all",
        publication_plot_mode: bool = False,
        axes:list = None, # if None, we make a new figure; otherwise we use these axes in order
        colors=None,
        enable_picking=False,
        parent_figure=None,
    ):
        """Make a summary plot of spike parameters for selected cell types.

        Args:
            df (Pandas dataframe): _description_
            xname: str: name to use for x category
            hue_category: str: name to use for hue category
            plot_order: list, optional: order to plot the categories
            publication_plot_mode: bool, optional: use publication plot mode
            measures: list, optional: list of measures to plot
            colors: dict, optional: dictionary of colors to use for the categories
            enable_picking: bool, optional: enable picking of data points
        """
        df["Subject"] = df.apply(set_subject, axis=1)
        df = df.copy()  # make sure we don't modifiy the incoming
        print("summary plot incoming x : ", df[xname].unique())

        print(
            "summary_plot_ephys_parameters_categorical: incoming x categories: ", df[xname].unique()
        )


        if parent_figure is None:
            P, letters, plabels, cols, nrows = self.create_plot_figure(df=df, xname=xname, measures=measures,
                                                                   parent_figure=parent_figure)
        else:
            P = parent_figure
            if plabels is None:
                letters = ascii_letters.upper()
                plabels = [f"{let.upper():s}" for let in letters]

            nrows = len(self.experiment["celltypes"])
            cols = len(measures)

        picker_funcs = {}
        # n_celltypes = len(self.experiment["celltypes"])
        # print(df.cell_type.unique())
        df = self.rescale_values(df)
        local_measures = measures.copy()
        if representation in ["bestRs", "mean"]:
            df = SAD.get_best_and_mean(
                df,
                experiment=self.experiment,
                parameters=local_measures,
                select_by="Rs",
                select_limits=[0, 1e9],
            )
            for i, m in enumerate(measures):
                local_measures[i] = f"{m:s}_{representation:s}"

        # print("plot order: ", plot_order)
        for icol, measure in enumerate(local_measures):
            # if measure in ["AP_thr_V", "AHP_depth_V"]:
            #     CP("y", f"{measure:s}: {df[measure]!s}")
            if measure.startswith("dvdt_ratio_best"):
                if measure not in df.columns:
                    df[measure] = {}
                df[measure] = df["dvdt_rising_bestRs"] / df["dvdt_falling_bestRs"]
            if measure in self.transforms.keys():
                tf = self.transforms[measure]
            else:
                tf = None

            # print(":: cell types ::", self.experiment["celltypes"])
            if nrows > 1:
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

                    picker_func = self.create_one_plot_categorical(
                        data=df,
                        xname=xname,
                        y=measure,
                        ax=axp,
                        celltype=celltype,
                        hue_category=hue_category,
                        plot_order=plot_order,
                        colors=colors,
                        # edgecolor=edgecolor,
                        logx=False,
                        ylims=self.ylims[ycell][x_measure],
                        transform=tf,
                        xlims=None,
                        enable_picking=enable_picking,
                        publication_plot_mode=publication_plot_mode,
                    )
                    # except Exception as e:
                    #     print("Categorical plot error in ylims: ", self.ylims.keys(), ycell)
                    #     raise KeyError(f"\n{e!s}")
                    picker_funcs[axp] = picker_func  # each axis has different data...
                    self.relabel_xaxes(axp)
                    if publication_plot_mode:
                        axp.set_xlabel("")

                    elif celltype != self.experiment["celltypes"][-1]:
                        axp.set_xticklabels("")
                        axp.set_xlabel("")
                    
                    self.relabel_yaxes(axp, measure=x_measure)

            else:  # single row
                # here we probably have the cell type or group as the x category,
                # so we will simplify some things
                axp = P.axdict[f"{plabels[icol]:s}"]

                x_measure = "_".join((measure.split("_"))[:-1])
                if x_measure not in self.ylims["default"]:
                    continue
                if measure not in df.columns:
                    continue
                if measure in ["RMP", "RMP_bestRs", "RMP_Zero"]:  # put the assumed JP on the plot.
                    axp.text(
                        x=0.01,
                        y=0.01,
                        s=f"JP: {self.experiment['junction_potential']:.1f}mV",
                        fontsize="x-small",
                        transform=axp.transAxes,
                        ha="left",
                        va="bottom",
                    )
                plot_order = [p for p in plot_order if p in df[xname].unique()]
                picker_func = self.create_one_plot_categorical(
                    data=df,
                    xname=xname,
                    y=measure,
                    ax=axp,
                    celltype="all",
                    hue_category=hue_category,
                    plot_order=plot_order,
                    colors=colors,
                    logx=False,
                    ylims=self.ylims["default"][x_measure],
                    transform=tf,
                    xlims=None,
                    enable_picking=enable_picking,
                    publication_plot_mode=publication_plot_mode,
                )
                picker_funcs[axp] = picker_func
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
        datestring = datetime.datetime.now().strftime("%d-%b-%Y")
        if any(c.startswith("dvdt_rising") for c in measures):
            fn = Path(
                f"spike_shapes_{self.experiment['directory']:s}_{datestring}.csv"
            )  # "spike_shapes.csv"
        elif any(c.startswith("Adapt") for c in measures):
            fn = Path(
                f"firing_parameters_{self.experiment['directory']:s}_{datestring}.csv"
            )  # "firing_parameters.csv"
        elif any(c.startswith("RMP") for c in measures):
            fn = Path(f"rmtau_{self.experiment['directory']:s}_{datestring}.csv")
        self.export_r(df=df, xname=xname, measures=measures, hue_category=hue_category, filename=fn)
        return P, picker_funcs

    def summary_plot_ephys_parameters_continuous(
        self,
        df,
        measures,
        hue_category=None,
        plot_order=None,
        colors=None,
        xname: str = "",
        logx=False,
        xlims=None,
        plabels = None,
        representation: str = "bestRs",  # bestRs, mean, all
        enable_picking: bool = False,
        publication_plot_mode: bool = False,
        parent_figure=None,
    ):
        """Make a summary plot of spike parameters for selected cell types.

        Args:
            df (Pandas dataframe): _description_
        """
        # print("starting continuous with representation = ", representation)
        # print("Incoming parameters: ")
        # print("measures: ", measures)
        # print("hue_category: ", hue_category)
        # print("plot_order: ", plot_order)
        # print("colors: ", colors)
        # print("xname: ", xname)
        # print("logx: ", logx)
        # print("xlims: ", xlims)
        # print("representation: ", representation)
        # print("enable_picking: ", enable_picking)
        # print("publication_plot_mode: ", publication_plot_mode)
        # print("parent_figure: ", parent_figure)

        picker_funcs = {}
        if parent_figure is None:
            P, letters, plabels, cols, nrows = self.create_plot_figure(df=df, xname=xname, measures=measures,
                                                                   parent_figure=parent_figure)
        else:
            P = parent_figure
            if plabels is None:
                letters = ascii_letters.upper()
                plabels = [f"{let.upper():s}" for let in letters]
            else:
                plabels = plabels
            nrows = len(self.experiment["celltypes"])
            cols = len(measures)

        df = df.copy()
        # df["FIMax_1"] = df.apply(getFIMax_1, axis=1)
        # df["FIMax_4"] = df.apply(getFIMax_imax, axis=1, imax=4.0)
        df["FIRate"] = df.apply(self.getFIRate, axis=1)
        df.dropna(subset=["Group"], inplace=True)  # remove empty groups
        df["age"] = df.apply(numeric_age, axis=1)
        if "max_age" in self.experiment.keys():
            df = df[(df.Age >= 0) & (df.Age <= self.experiment["max_age"])]
        df["shortdate"] = df.apply(make_datetime_date, axis=1)
        df["SR"] = df.apply(self.flag_date, axis=1)
        # df.dropna(subset=["age"], inplace=True)
        df = self.rescale_values(df)
        if representation in ["bestRs", "mean"]:
            df = SAD.get_best_and_mean(
                df,
                experiment=self.experiment,
                parameters=measures,
                select_by="Rs",
                select_limits=[0, 1e9],
            )
            local_measures = measures.copy()
            for i, m in enumerate(measures):
                local_measures[i] = f"{m:s}_{representation:s}"

        # print("continuous measures: ", measures)
        for icol, measure in enumerate(local_measures):
            if measure.startswith("dvdt_ratio_bestRs"):
                if measure not in df.columns:
                    df[measure] = {}
                df[measure] = df["dvdt_rising_bestRs"] / df["dvdt_falling_bestRs"]
            if measure in self.transforms.keys():
                tf = self.transforms[measure]
            else:
                tf = None
            # print("continuous: nrows > 1:  "    , nrows)
            if nrows > 1:
                for i, celltype in enumerate(self.experiment["celltypes"]):
                    # print("measure y: ", measure, "celltype: ", celltype)
                    if axes is None:
                        axp = P.axdict[f"{letters[i]:s}{icol+1:d}"]
                    else:
                        axp = axes[i]
                    if celltype not in self.ylims.keys():
                        ycell = "default"
                    else:
                        ycell = celltype
                    x_measure = "_".join((measure.split("_"))[:-1])
                    if x_measure not in self.ylims[ycell]:
                        continue
                    if measure not in df.columns:
                        continue

                    picker_func = self.create_one_plot_continuous(
                        data=df,
                        xname=xname,
                        y=measure,
                        ax=axp,
                        celltype=celltype,
                        # hue_category=hue_category,
                        # plot_order=plot_order,
                        # colors=colors,
                        # edgecolor=edgecolor,
                        logx=False,
                        ylims=self.ylims[ycell][x_measure],
                        transform=tf,
                        xlims=None,
                        enable_picking=enable_picking,
                        publication_plot_mode=publication_plot_mode,
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
                    if publication_plot_mode:
                        xlab = axp.get_xlabel()
                        if self.experimental["new_xlabels"] is not None:
                            if xlab in self.experimental["new_xlabels"]:
                                xlab = self.experimental["new_xlabels"][xlab]
                                axp.set_xlabel(xlab)
            else:
                # print("continous, nrows == 1")
                celltype = self.experiment["celltypes"][0]
                axp = P.axdict[f"{plabels[icol]:s}"]

                if measure.endswith("_bestRs_bestRs"):
                    measure = "_".join(measure.split("_")[:-1])
                
                if measure not in df.columns:
                    continue
                # always indicate JP on RMP data
                if measure in ["RMP", "RMP_bestRs", "RMP_Zero"]:  # put the assumed JP on the plot.
                    axp.text(
                        x=0.01,
                        y=0.01,
                        s=f"JP: {self.experiment['junction_potential']:.1f}mV",
                        fontsize="x-small",
                        transform=axp.transAxes,
                        ha="left",
                        va="bottom",
                    )
         

                plot_order = [p for p in plot_order if p in df[xname].unique()]
        
                picker_func = self.create_one_plot_continuous(
                    data=df,
                    x="age",
                    y=measure,
                    ax=axp,
                    # hue_category=hue_category,
                    # plot_order=plot_order,
                    # colors=colors,
                    # edgecolor=edgecolor,
                    celltype=self.experiment["celltypes"][0],
                    logx=logx,
                    ylims=self.ylims,
                    xlims=xlims,
                    transform=tf,
                    # publication_plot_mode=publication_plot_mode,
                    regplot=True

                ) 

                picker_funcs[axp] = picker_func
                self.relabel_xaxes(axp)
                self.relabel_yaxes(axp, measure=measure)
                if publication_plot_mode:
                    xlab = axp.get_xlabel()
                    if self.experiment["new_xlabels"] is not None:
                        if xlab in self.experiment["new_xlabels"]:
                            xlab = self.experiment["new_xlabels"][xlab]
                            axp.set_xlabel(xlab)

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
     
        if len(self.experiment["celltypes"]) > 1:
               for i, cell in enumerate(self.experiment["celltypes"]):
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
        edgecolor="k",
        logx=False,
        ylims=None,
        xlims=None,
        yscale=1,
        transform=None,
        regplot: bool = False,
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
        # print(dfp.columns)
        # print(dfp[y])
        if regplot:

            sns.regplot(
                data=dfp,
                x=x,
                y=y,
                ax=ax,
                logx=logx,
                scatter_kws={"s": 4, "edgecolor": edgecolor, "facecolor": "k", "alpha": 0.8},
                line_kws={"color": "b"},
            )
            # print(dfp.columns)
            # print(y, x)
            model = statsmodels.api.OLS.from_formula(f"{y:s} ~ {x:s}", data=dfp,
                                                     cov_type='robust')
            results = model.fit()
            print(results.summary())
            # print(dir(results))
            # results.fvalue is the F statistic
            # the f_pvalue is the p of the F statistic
            # These compare the null (slope = 0) to the fit to the data (slope != 0)
            print(results.f_pvalue, results.fvalue)
            r2 = r"$r^2$"
            if results.f_pvalue < 0.001:
                fp = f"{results.f_pvalue:.3e}"
            elif results.f_pvalue < 0.05:
                fp = f"{results.f_pvalue:.3f}"
            else:
                fp = f"{results.f_pvalue:.2f}" 
            stat_text = f"{r2:s}={results.rsquared:.3f} F={results.fvalue:.2f} p={fp:s}"
            stat_text += f"\nd.f.={int(results.df_resid):d},{int(results.df_model):d}"
            ax.text(0.02, 1.00, stat_text, transform=ax.transAxes, ha="left", va="top",
                    fontsize="x-small", color="k")
  
        else:
            ax.scatter(x=dfp[x], y=dfp[y], c="b", s=4, picker=True)

        if y.endswith("_bestRs") or y.endswith("_mean"):
            y = "_".join([*y.split("_")[:-1]])  # reassemble without the trailing label
        self.relabel_yaxes(ax, measure=y)
        self.relabel_xaxes(ax)
        if ylims is not None:  # make sure we have some limits
            for lim in ylims.keys():  # may be "limits1", etc.
                if celltype in ylims[lim]["celltypes"]:  # check the list of cell types in the limit group
                    if y in ylims[lim].keys():  # check the list of measures in the limit group
                        ax.set_ylim(ylims[lim][y])  # finally... 
        print("XLIMS: ", xlims)
        if xlims is not None:
            ax.set_xlim(xlims)
        else:
            ax.set_xlim(0, 600)
        print("XLIMS as set: ", ax.get_xlim())
        # PH.do_talbotTicks(ax,axes='x', density=(0.2, 2))
        picker_func = Picker()
        # picker_func.setData(dfp.copy(deep=True), axis=ax)
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
        edgecolor="k",
        logx: bool = False,
        ylims=None,
        xlims=None,
        yscale=1,
        transform=None,
        enable_picking: bool = False,
        publication_plot_mode: bool = False,
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

        if celltype != "all":
            dfp = dfp[dfp["cell_type"] == celltype]
        dfp = dfp.apply(self.apply_scale, axis=1, measure=y, scale=yscale)
        if transform is not None:
            dfp[y] = dfp[y].apply(transform)
        # if hue_category is None:
        #     raise ValueError(f"Missing Hue category for plot; xname is: {xname:s}")
        if ylims is not None:
            ax.set_ylim(ylims)
        if xlims is not None:
            ax.set_xlim(xlims)
        picker_func = self.bar_pts(
            dfp,
            xname=xname,
            yname=y,
            ax=ax,
            hue_category=hue_category,
            celltype=celltype,
            plot_order=plot_order,
            colors=colors,
            edgecolor=edgecolor,
            enable_picking=enable_picking,
            publication_plot_mode=publication_plot_mode,
        )

        self.relabel_yaxes(ax, measure=y)
        self.relabel_xaxes(ax)
        print("CHECKING X LABEL: ", ax.get_label())
        if publication_plot_mode and ax.get_label() == "age_category":
            ax.set_xlabel("Age Group")
        # print("Plotted measure: ", y, "for celltype: ", celltype)
        # print("dfp: ", dfp)
        return picker_func

    def clean_sex_column(self, row):
        if row.sex not in ["F", "M"]:
            row.sex = "U"
        return row.sex

    def categorize_ages(self, row):
        row.age = numeric_age(row)
        for k in self.experiment["age_categories"].keys():
            if (
                row.age >= self.experiment["age_categories"][k][0]
                and row.age <= self.experiment["age_categories"][k][1]
            ):
                row.age_category = k
            else:
                row.age_category = "ND"
        return row.age_category

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

    def adjust_AHP_depth_V(self, row):

        if isinstance(row.AHP_depth_V, float):
            row.AHP_depth_V = [row.AHP_depth_V + 1e-3 * self.experiment["junction_potential"]]
        else:
            row.AHP_depth_V = [
                ap + 1e-3 * self.experiment["junction_potential"] for ap in row.AHP_depth_V
            ]
        return row.AHP_depth_V

    def adjust_AHP_trough_V(self, row):
        if isinstance(row.AHP_trough_V, float):
            row.AHP_trough_V = [row.AHP_trough_V + 1e-3 * self.experiment["junction_potential"]]
        else:
            row.AHP_trough_V = [
                ap + 1e-3 * self.experiment["junction_potential"] for ap in row.AHP_trough_V
            ]
        return row.AHP_trough_V

    def adjust_AP_thr_V(self, row):
        if isinstance(row.AP_thr_V, float):
            row.AP_thr_V = [row.AP_thr_V + 1e-3 * self.experiment["junction_potential"]]
        else:
            row.AP_thr_V = [
                ap + 1e-3 * self.experiment["junction_potential"] for ap in row.AP_thr_V
            ]
        return row.AP_thr_V

    def clean_rmp(self, row):
        """clean_rmp check that the RMP is in range (from the config file),
        and adjust for the junction potential

        Parameters
        ----------
        row : pandas series
            data row

        Returns
        -------
        RMP values: list
            _description_
        """
        if "data_inclusion_criteria" in self.experiment.keys():
            if row.cell_type in self.experiment["data_inclusion_criteria"].keys():
                min_RMP = self.experiment["data_inclusion_criteria"][row.cell_type]["RMP_min"]
            else:
                min_RMP = self.experiment["data_inclusion_criteria"]["default"]["RMP_min"]
        if isinstance(row.RMP, float):
            row.RMP = [row.RMP + self.experiment["junction_potential"]]
        else:
            row.RMP = [rmp + self.experiment["junction_potential"] for rmp in row.RMP]
        for i, rmp in enumerate(row.RMP):
            if rmp > min_RMP:
                row.RMP[i] = np.nan
        return row.RMP

    def clean_rmp_zero(self, row):
        """clean_rmp_zero adjust RMP measured at zero current, much like clean_rmp.

        Parameters
        ----------
        row : pandas series
            cell data row

        Returns
        -------
        row.RMP_Zero : list
            adjusted RMP_zeros.
        """

        if "data_inclusion_criteria" in self.experiment.keys():
            if row.cell_type in self.experiment["data_inclusion_criteria"].keys():
                min_RMP = self.experiment["data_inclusion_criteria"][row.cell_type]["RMP_min"]
            else:
                min_RMP = self.experiment["data_inclusion_criteria"]["default"]["RMP_min"]
        if isinstance(row.RMP_Zero, float):
            r0 = [
                row.RMP_Zero + self.experiment["junction_potential"]
            ]  # handle case where there is only one float value
        else:
            r0 = [rmp_0 + self.experiment["junction_potential"] for rmp_0 in row.RMP_Zero]
        for i, r0 in enumerate(r0):
            if r0 > min_RMP:
                row.RMP_Zero[i] = np.nan
        return row.RMP_Zero

    

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
        publication_plot_mode: bool = False,
    ):
        """summary_plot_fi plots all of the individual FI curves for the selected cell types,
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
        plabels = []
        for i in range(len(self.experiment["celltypes"])):
            for j in range(3):
                plabels.append(f"{ascii_letters[i].upper():s}{j+1:d}")
        # set up the plot area
        n_celltypes = len(self.experiment["celltypes"])
        if group_by == "cell_type":
            n_celltypes = 1
        P = PH.regular_grid(
            rows=n_celltypes,
            cols=3,
            order="rowsfirst",
            figsize=(8 + 1.0, 3 * n_celltypes),
            panel_labels=plabels,
            labelposition=(-0.05, 1.02),
            margins={
                "topmargin": 0.12,
                "bottommargin": 0.12,
                "leftmargin": 0.12,
                "rightmargin": 0.22,
            },
            verticalspacing=0.2,
            horizontalspacing=0.1,
            fontsize={"label": 12, "tick": 8, "panel": 16},
        )
        # print(P.axdict)
        for ax in P.axdict:
            PH.nice_plot(P.axdict[ax], direction="outward", ticklength=3, position=-0.03)

        # positional information for the plots.
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
        longform = "cell, group, Iinj, rate\n"  # build a csv file the hard way
        prism_form = ""
        mode = "mean"  # "individual"
        # P.figure_handle.suptitle(f"Protocol: {','.join(protosel):s}", fontweight="bold", fontsize=18)
        if mode == "mean":
            P.figure_handle.suptitle("FI Mean, SEM ", fontweight="normal", fontsize=18)
        elif mode == "individual":
            P.figure_handle.suptitle("FI Individual", fontweight="normal", fontsize=18)
        fi_stats = []

        picker_funcs: Dict = {}
        N = self.experiment["group_map"]
        print("N.keys: ", N.keys())
        found_groups = []
        FIy_all: dict = {k: [] for k in N.keys()}
        FIx_all: dict = {k: [] for k in N.keys()}
        fi_dat = {}  # save raw fi
        for ic, ptype in enumerate(["mean", "individual", "sum"]):
            NCells: dict[tuple] = {}
            for ir, celltype in enumerate(self.experiment["celltypes"]):
                if n_celltypes == 1:  # combine all onto one set of plots
                    ixr = 0
                else:
                    ixr = ir
                # create a dataframe for the plot.
                fi_group_sum = pd.DataFrame(
                    columns=["Group", "sum", "sex", "cell_type", "cell_expression"]
                )
                if n_celltypes == 1:
                    ax = P.axarr[0]
                else:
                    ax = P.axarr[ixr]
                ax[ic].set_title(celltype.title(), y=1.05)
                ax[ic].set_xlabel("I$_{inj}$ (nA)")
                if ic in [0, 1]:
                    ax[ic].set_ylabel("Rate (sp/s)")
                elif ic == 2 and ptype == "sum":
                    ax[ic].set_ylabel(self.experiment["new_ylabels"]["summed_FI"])
                if celltype != "all":
                    cdd = df[df["cell_type"] == celltype]
                else:
                    cdd = df.copy()

                # if ptype == "mean":  # set up arrays to compute mean

                for index in cdd.index:  # go through the individual cells
                    group = cdd[group_by][index]
                    sex = cdd["sex"][index]
                    if (celltype, group) not in NCells.keys():
                        NCells[(celltype, group)] = 0
                    if pd.isnull(group):
                        continue
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

                    # convert the current to nA
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

                    maxi = 1000e-12
                    ilim = np.argwhere(FI_data[0] <= maxi)[-1][0]
                    if ptype in ["individual"]:
                        fix, fiy, fiystd, yn = FUNCS.avg_group(
                            np.array(FI_data[0]), FI_data[1], ndim=1
                        )
                        NCells[(celltype, group)] += 1  # to build legend, only use "found" groups
                        if group not in found_groups:
                            if pd.isnull(group) or group == "nan":
                                group = "Unidentified"
                            found_groups.append(group)

                        ax[ic].plot(
                            fix[:ilim] * 1e9,
                            fiy[:ilim],
                            color=colors[group],
                            marker=None,
                            markersize=2.5,
                            linewidth=0.5,
                            clip_on=False,
                            alpha=0.35,
                        )
                    elif ptype == "mean":
                        # while in this loop, build up the arrays for the mean
                        if group in FIy_all.keys():
                            FIy_all[group].append(np.array(FI_data[1][:ilim]))
                            FIx_all[group].append(np.array(FI_data[0][:ilim]) * 1e9)
                            for iv in range(len(FI_data[1][:ilim])):
                                longform += f"{cdd['cell_id'][index]:s}, {group:s}, {1e12*FI_data[0][iv]:f}, {FI_data[1][iv]:f}\n"
                            NCells[
                                (celltype, group)
                            ] += 1  # to build legend, only use "found" groups
                            if pd.isnull(group) or group == "nan":
                                group = "Unidentified"
                            found_groups.append(group)
                    elif ptype == "sum":
                        fi_group_sum.loc[len(fi_group_sum)] = [
                            group,
                            np.sum(np.array(FI_dat_saved[1])),
                            sex,
                            celltype,
                            "None",
                        ]
                        NCells[(celltype, group)] += 1  # to build legend, only use "found" groups
                        if pd.isnull(group) or group == "nan":
                            group = "Unidentified"
                        found_groups.append(group)

                if ptype == "mean":
                    # compute the avearge and plot the data with errorbars
                    max_FI = 1.0
                    for index, group in enumerate(FIy_all.keys()):
                        fx, fy, fystd, yn = FUNCS.avg_group(FIx_all[group], FIy_all[group])
                        if len(fx) == 0:
                            # print("unable to get average", cdd["cell_id"][index])
                            continue
                        # else:
                        #     print("getting average: ", cdd["cell_id"][index])
                        if "max_FI" in self.experiment.keys():
                            max_FI = self.experiment["max_FI"] * 1e-3
                        ax[ic].errorbar(
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

                        ax[ic].set_xlim(0, max_FI)

                if ptype == "sum":
                    # Spike sum plot -
                    # ax[ic] = P.axarr[ir, ic]
                    ax[ic].set_title("Summed FI", y=1.05)
                    ax[ic].set_xlabel("Group")
                    print("SUM: ", fi_group_sum)
                    # fi_group_sum = fi_group_sum[celltype].isin(self.experiment["celltypes"])
                    if not all(np.isnan(fi_group_sum["sum"])):
                        self.bar_pts(
                            fi_group_sum,
                            xname="Group",
                            yname="sum",
                            celltype=celltype,
                            # hue_category = "sex",
                            ax=ax[ic],
                            plot_order=self.experiment["plot_order"][group_by],  # ["age_category"],
                            colors=self.experiment["plot_colors"],
                            enable_picking=False,
                            publication_plot_mode=publication_plot_mode,
                        )
                        all_limits = self.experiment["ylims"]
                        # check if our cell type is in one of the subkeys of the limits:
                        ax[ic].set_ylim(all_limits["default"]["summed_FI_limits"])
                        for limit in all_limits.keys():
                            print(
                                "testing limit: ",
                                limit,
                                "with celltypes: ",
                                all_limits[limit]["celltypes"],
                            )
                            if celltype in all_limits[limit]["celltypes"]:
                                print("found limit: ", limit)
                                ax[ic].set_ylim(all_limits[limit]["summed_FI_limits"])
                                break

                        ax[ic].legend(bbox_to_anchor=(1.05, 1), loc="upper left", borderaxespad=0)
                        PH.talbotTicks(ax[ic], axes="y", density=(1, 1))

                    yoffset = 0
                    xoffset = 1.05
                    xo2 = 0.0

                    i_glp = 0
                    for i, group in enumerate(self.experiment["group_legend_map"].keys()):
                        if group not in found_groups:
                            continue
                        # if celltype == "pyramidal":  # more legend - include the name of group
                        #     ax.text(
                        #         x=pos[celltype][0] + xoffset + xo2,
                        #         y=pos[celltype][1] + 0.095 * (i_glp - 0.5) + yoffset,
                        #         s=f"{self.experiment['group_legend_map'][group]:s} (N={NCells[(celltype, group)]:>3d})",
                        #         ha="left",
                        #         va="top",
                        #         fontsize=8,
                        #         color=colors[group],
                        #         transform=ax.transAxes,
                        #     )
                        #     print(
                        #         "Pyramidal legend: ",
                        #         f"{self.experiment['group_legend_map'][group]:s} (N={NCells[(celltype, group)]:>3d}",
                        #     )
                        #     i_glp += 1
                        # else:
                        if True:
                            if (celltype, group) in NCells.keys():
                                textline = (
                                    f"{group:s}, {celltype:s} N={NCells[(celltype, group)]:>3d}"
                                )
                            else:
                                textline = f"N={0:>3d}"
                            fcelltype = celltype
                            if celltype not in pos.keys():
                                fcelltype = "default"
                            if (group_by != "cell_type") or (n_celltypes == 1 and ir == 0):
                                ax[ic].text(
                                    x=xoffset,  # pos[fcelltype][0] + xoffset + xo2,
                                    y=yoffset
                                    + i_glp
                                    * 0.05,  # pos[fcelltype][1] - 0.095 * (i_glp - 0.5) + yoffset,
                                    s=textline,
                                    ha="left",
                                    va="bottom",
                                    fontsize=8,
                                    color=colors[group],
                                    transform=ax[ic].transAxes,
                                )
                            i_glp += 1

            # print("-" * 80)
            # print("FI keys: ", FIx_all.keys())
            # for k in FIx_all.keys():
            #     print("Cell type: ", k)
            #     if len(FIx_all[k]) > 0:
            #         print("   FIx all: ", FIx_all[k])
            #         print("   FIy all: ", FIy_all[k])
            # df1 = pd.DataFrame({'current': FIx_all['giant'], 'rate': FIy_all['giant'], 'group': 'giant'})
            # df2 = pd.DataFrame({'current': FIx_all['pyramidal'], 'rate': FIy_all['pyramidal'], 'group': 'pyramidal'})
            # df = pd.concat([df1, df2])
            # print(df.head())
            # import statsmodels.api as SM
            # from statsmodels.formula.api import ols
            # model = ols('rate ~ C(group) + C(current) + C(group):C(current)', data=df).fit()
            # SM.stats.anova_lm(model, typ=2)

            # if ptype == "mean":
            #     for i, group in enumerate(self.experiment["group_legend_map"].keys()):
            #         if (celltype, group) in NCells.keys():
            #             fi_stats.append(
            #                 {
            #                     "celltype": celltype,
            #                     "Group": group,
            #                     "I_nA": fx,
            #                     "sp_s": fy,
            #                     "N": NCells[(celltype, group)],
            #                 }
            #             )
            #     print("fi_stats: ", fi_stats)
        i = 0
        icol = 0

        axp = P.axdict["A1"]
        axp.legend(
            fontsize=7, bbox_to_anchor=(0.95, 0.90), bbox_transform=P.figure_handle.transFigure
        )
        with open("FI_Data.csv", "w") as f:
            f.write(longform)

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
                desc_stat += f"Group: {gr!s}  N: {np.sum(~np.isnan(dictdata[gr])):d}, median: {scale*np.nanmedian(dictdata[gr]):.6f},"
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
            return row.AHP_depth_V[i]  # convert to mV here
        else:
            CP("c", "LowestCurrentSpike in row keys")

    def compute_AHP_trough_time(self, row):
        # RE-Calculate the AHP trough time, as the time between the AP threshold and the AHP trough
        # if the depth is positive, then the trough is above threshold, so set to nan.

        if isinstance(row.AP_thr_T, float):
            row.AP_thr_T = [row.AP_thr_T]
        if isinstance(row.AHP_trough_T, float):
            if np.isnan(row.AHP_trough_T):
                return row.AHP_trough_T
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
            row.Subject = df_summary.loc[df_summary.cell_id == cell_id_match][
                "animal_identifier"
            ].values[0]
            # else:
            #     print("row keys: ", sorted(row.keys()))
            #     if "Subject" in row.keys():
            #         print("Found subject column but not animal[_]identifier: ", row["Subject"])
            #     raise ValueError("could not match animal id/Subject with column")
        return row.Subject

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
            CP(
                "y",
                f"get cell expression: cell id match is None: {cell_id:s}, \n{df_summary.cell_id.values!s}",
            )
            return ""
        # print("cell id match ok: ", cell_id)
        if cell_id_match is not None:
            row.cell_expression = df_summary.loc[
                df_summary.cell_id == cell_id_match
            ].cell_expression.values[0]

            if row.cell_expression in [" ", "nan", "NaN", np.nan] or pd.isnull(row.cell_expression):
                row.cell_expression = "ND"
            if (
                "remove_expression" in self.experiment.keys()
                and self.experiment["remove_expression"] is not None
            ):
                if row.cell_expression in self.experiment["remove_expression"]:
                    for re in self.experiment["remove_expression"]:
                        if row.cell_expression == re:
                            row.cell_expression = "ND"
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
        try:
            df = pd.read_pickle(fn, compression="gzip")
        except:
            df = pd.read_pickle(fn)
        df_summary = get_datasummary(self.experiment)
        df = self.preprocess_data(df, self.experiment)
        return df

    def print_preprocessing(self, df_summary):
        print("   Preprocess_data: df_summary column names: ", sorted(df_summary.columns))
        print("df summary ids: ")
        for cid in df_summary.cell_id:
            print("    ", cid)
        print(" ")
        print("df ids: ")
        for cid in df.cell_id:
            print("    ", cid)
        print("*" * 80)

        print("   Preprocess_data: df column names: ", sorted(df.columns))
        print("df: ")
        for d in df.to_dict(orient="records"):
            print(f"<{d['cell_id']!s}, type(d['cell_id'])")

        print("summary: ")
        for d in df_summary.to_dict(orient="records"):
            print(f"<{d['cell_id']!s}, ty")
        # note df will have "short" names: Rig2(MRK)/L23_intrinsic/2024.10.22_000_S0C0
        # df_summary will have long names: Rig2(MRK)/L23_intrinsic/2024.10.22_000/slice_000/cell_000
        # print("-2 ", df['cell_id'].eq("Rig2(MRK)/L23_intrinsic/2024.10.22_000_S0C0").any())
        # print("-1 ", df_summary['cell_id'].eq("Rig2(MRK)/L23_intrinsic/2024.10.22_000_S0C0").any())

    def check_list_contained(self, A: list, B: list):
        # check if any of values in A are in B.
        A_str = " ".join(map(str, A))
        B_str = " ".join(map(str, B))
        # find all instances of A within B, case insensitive
        instances = re.findall(A_str, B_str, re.IGNORECASE)
        # return True if any instances were found, False otherwise
        return len(instances) > 0

    def check_include_exclude(self, df):
        # Note that prior to this step, excluded IVs may have been analyzed
        # so that the pdf and pkl files are present. If the analysis was updated,
        # then the files *may* represent the updated analysis, but there may still be some
        # excluded IVs on the disk.
        # Note: this is tricky, as we may also have 'included' partial datasets
        # Included IVs should *always* take precedence over the exclusion rules,
        # as they will have partial data that is being "rescued" and should NOT be excluded.
        # for the same cell and protocol!

        if len(self.experiment["excludeIVs"]) == 0:
            return df
        re_check_all = re.compile(r"All", re.IGNORECASE)
        re_day = re.compile(r"(\d{4})\.(\d{2})\.(\d{2})\_(\d{3})$")
        re_slice = re.compile(r"(\d{4})\.(\d{2})\.(\d{2})\_(\d{3})\/slice_(\d{3})$")
        re_slicecell = re.compile(
            r"(\d{4})\.(\d{2})\.(\d{2})\_(\d{3})\/slice_(\d{3})\/cell_(\d{3})$"
        )
        # get slice and cell nubmers
        re_slicecell2 = re.compile(
            r"^(?P<year>\d{4})\.(?P<month>\d{2})\.(?P<day>\d{2})\_(?P<dayno>\d{3})\/slice_(?P<sliceno>\d{3})\/cell_(?P<cellno>\d{3})$"
        )
        if self.experiment["includeIVs"] is not None:
            includes = list(self.experiment["includeIVs"].keys())
        else:
            includes = []

        CP("c", "Parsing/checking excluded and included IV datasets (noise, etc)")

        for filename, key in self.experiment["excludeIVs"].items():
            # so we should test for inclusion here first, for a cell and it's protocols.
            # includes are always fully specified day/slice/cell names, with protocols.
            # includes always take precedence. Note that the analysis will have already excluded the
            # protocols in the exclude list, so we can skip those here.
            if filename in includes:
                FUNCS.textappend(
                    # "c",
                    f"   Preprocess_data: {filename:s} is in inclusion list (which takes precedence), so it will not be excluded.",
                )
                continue
            # everything after this relates only to exclusion by day, slice, or cell.
            fparts = Path(filename).parts
            fn = str(Path(*fparts[-3:]))
            # print(fn)
            # raise ValueError("Stop here")

            if len(fparts) > 3:
                fnpath = str(Path(*fparts[:-3]))  # just day/slice/cell
            else:
                fnpath = None  # no leading path
            reason = key["reason"]
            protocols = key["protocols"]
            # includes will ALWAYS be fully specified day/slice/cell names, with protocols.

            # print("   Preprocess_data: Checking exclude for listed exclusion ", filename)
            drAopped = False

            if re_day.match(fn) is not None:  # specified a day, not a cell:
                df.drop(df.loc[df.cell_id.str.startswith(fn)].index, inplace=True)
                FUNCS.textappend(
                    # "r",
                    f"   Preprocess_data: dropped DAY {fn:s} from analysis, reason = {reason:s}",
                )
                dropped = True
            elif re_slice.match(fn) is not None:  # specified day and slice
                fns = re_slice.match(fn)
                df.drop(df.loc[df.cell_id.str.startswith(fns)].index, inplace=True)
                FUNCS.textappend(
                    # "r",
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
                        if self.check_list_contained(["all"], protocols):
                            df.drop(df.loc[df.cell_id == fns[i]].index, inplace=True)
                            FUNCS.textappend(
                                "r",
                                f"   Preprocess_data: dropped CELL {fns[i]:s} from analysis, reason = {reason:s}",
                            )
                            dropped = True
                        # otherwise, the excluded protocols were NOT analyzed in the first place, so we can continue
                        # df.drop(df.loc[df.cell_id == fns[i]].index, inplace=True)
                        # CP(
                        #     "m",
                        #     f"   Preprocess_data: dropped CELL {fns[i]:s} from analysis, reason = {reason:s}",
                        # )
                        # dropped = True
                    elif not dropped:
                        pass
                        # FUNCS.textappend(
                        #     # "y",
                        #     f"   Preprocess_data: CELL {fns[i]:s} not found in data set (may already be excluded by prior analysis)",
                        # )
            elif not dropped:
                FUNCS.textappend(
                    # "y",
                    f"   Preprocess_data: {filename:s} not dropped, but was found in exclusion list",
                )
            else:
                FUNCS.textappend(
                    # "y",
                    f"   Preprocess_data: No exclusions found for {filename:s}"
                )
            # if fn == "2023.09.11_000/slice_001/cell_001":
            #     print("Dropped: ", dropped)
            #     raise ValueError("Dropped: ", dropped)

        return df

    def preprocess_data(self, df, experiment):
        pd.options.mode.copy_on_write = True
        """preprocess_data Clean up the data, add columns, etc., apply junction potential corrections, etc."""
        df_summary = get_datasummary(experiment)
        # self.print_preprocessing(df_summary)

        # generate a subject column
        df["Subject"] = ""
        df["Subject"] = df.apply(self.get_animal_id, df_summary=df_summary, axis=1)

        # generate a usable layer column
        if "cell_layer" not in df.columns:
            layers = df_summary.cell_layer.unique()
            if len(layers) == 1 and layers == [" "]:  # no layer designations
                df["cell_layer"] = "unknown"
            else:
                df["cell_layer"] = ""
                df["cell_layer"] = df.apply(self.get_cell_layer, df_summary=df_summary, axis=1)

        # generate a usable cell expression column (e.g., is cell labeled or not)
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

        #  ******************************************************************************
        CP("c", "\n   Preprocess_data: Groups and cells PRIOR to exclusions: ")
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
        if "AP_thr_V" not in df.columns:
            df["AP_thr_V"] = np.nan
        else:
            df["AP_thr_V"] = df.apply(self.adjust_AP_thr_V, axis=1)
        if "AHP_trough_V" not in df.columns:
            df["AHP_trough_V"] = np.nan
        else:
            df["AHP_trough_V"] = df.apply(self.adjust_AHP_trough_V, axis=1)
        if "AHP_depth_V" not in df.columns:
            df["AHP_depth_V"] = np.nan
        else:
            df["AHP_depth_V"] = df.apply(self.adjust_AHP_depth_V, axis=1)
        df["AHP_depth_V"] = df.apply(self.compute_AHP_depth, axis=1)

        df["AHP_trough_T"] = df.apply(self.compute_AHP_trough_time, axis=1)
        if len(df["Group"].unique()) == 1 and df["Group"].unique()[0] == "nan":
            if self.experiment["set_group_control"]:
                df["Group"] = "Control"

        groups = df.Group.unique()
        FUNCS.textbox_setup(self.textbox)
        #        expressions = df.cell_expression.unique()
        if (  # remove specific expression labels from teh data set?
            "remove_expression" in self.experiment.keys()
            and self.experiment["remove_expression"] is not None
        ):
            FUNCS.textappend("REMOVING specific cell_expression")
            for expression in self.experiment["remove_expression"]:  # expect a list
                df = df[df.cell_expression != expression]
                FUNCS.textappend(f"   Preprocess_data: Removed expression:  {expression:s}")

        if "groupname" not in df.columns:
            df["groupname"] = np.nan
        df["groupname"] = df.apply(rename_groups, experiment=self.experiment, axis=1)
        # print("5.0: groups removed ", df['cell_id'].eq("Rig2(MRK)/L23_intrinsic/2024.10.22_000_S0C0").any())
        if len(groups) > 1:
            # df.dropna(subset=["Group"], inplace=True)  # remove empty groups
            df.drop(df.loc[df.Group == "nan"].index, inplace=True)
            # print("5.2: groups removed ", df['cell_id'].eq("Rig2(MRK)/L23_intrinsic/2024.10.22_000_S0C0").any())
        FUNCS.textappend(
            f"           Preprocess_data: # Groups found after dropping 'nan' groups: "
            + f"{df.Group.unique()!s}"
            + f"{len(df.Group.unique()):d}"
        )

        df["age"] = df.apply(numeric_age, axis=1)
        df["shortdate"] = df.apply(make_datetime_date, axis=1)
        df["SR"] = df.apply(self.flag_date, axis=1)

        # Now make sure the excluded IV data is removed from the dataset and that included
        # datasets are properly included.
        df = self.check_include_exclude(df)
        # raise ValueError("Stop here 2")
        # gu = df.Group.unique()
        # print("   Preprocess_data: Groups and cells after exclusions: ")
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
            FUNCS.textappend("   Preprocess_data: Filters is set: ")
            for key, values in self.experiment["filters"].items():
                FUNCS.textappend(f"      Preprocess_data: Filtering on: (key:s), {values!s}")
                df = df[df[key].isin(values)]

        return df

    def do_stats(
        self, df, experiment, group_by, second_group_by, textbox: object = None, divider="-" * 80
    ):
        if textbox is not None:
            FUNCS.textbox_setup(textbox)
            FUNCS.textclear()
        df = self.preprocess_data(df, experiment, funcs=FUNCS)
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
                "AdaptIndex",
                "AdaptIndex2",
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
        P1, picker_funcs1 = PSI.summary_plot_ephys_parameters_categorical(
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
        P2, picker_funcs2 = PSI.summary_plot_ephys_parameters_categorical(
            dfa,
            measures=[
                "AdaptRatio",
                "AdaptIndex",
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

    # summary_plot_ephys_parameters_continuous(df, logx=True)
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
