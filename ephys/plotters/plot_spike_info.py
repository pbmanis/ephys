"""Plot summaries of spike and basic electrophys properties of cells.
Does stats at the end.
"""
import dataclasses
from pathlib import Path
from typing import Union, List
import datetime
import matplotlib.pyplot as mpl
import numpy as np
import pandas as pd
import scikit_posthocs as sp
import pprint
import scipy.stats
import seaborn as sns
import statsmodels.api as sa
import statsmodels.formula.api as sfa
import statsmodels.formula.api as smf

from statsmodels.stats.anova import AnovaRM
import pylibrary.plotting.plothelpers as PH
from pylibrary.tools import cprint
import ephys.datareaders as DR
import pingouin as ping
from ephys.ephys_analysis import spike_analysis
from ephys.tools import fitting
from ephys.tools import utilities
from ephys.tools import filter_data
from ephys.gui import data_table_functions
from ephys.tools.get_configuration import get_configuration

from pyqtgraph.Qt.QtCore import QObject, pyqtSignal
import dateutil.parser as DUP

PP = pprint.PrettyPrinter()

UTIL = utilities.Utility()
FUNCS = data_table_functions.Functions()
CP = cprint.cprint
Fitter = fitting.Fitting()


after = "2017.01.01"
after_parsed = datetime.datetime.timestamp(DUP.parse(after))
max_age = 200

# # expt = "CBA_Age"
# # experiment = analyze_ivs.experiments[expt]
# datasets, experiment = get_configuration()
# PP.pprint(experiment)
# expt = datasets[0]   # should be a selected one... but for now just pick first
# experiment = experiment[expt]

def set_ylims(experiment):
    if 'ylims' in experiment.keys():
        ylims = experiment['ylims']
    else:
        ylims_pyr = {
            "dvdt_rising": [0, 2000],
            "dvdt_falling": [0, 2000],
            "AP_HW": [0, 1.0],
            "AP_thr_V": [-75, 0],
            "AdaptRatio": [0, 2],
            "FISlope": [0, 1500],
            "maxHillSlope": [0, None],
            "I_maxHillSlope": [0, 1200],
            "FIMax_1": [0, 800],
            "FIMax_4": [0, 800],
            "taum": [0, None],
            "RMP": [-80, -40],
            "Rin": [0, 250],
        }
        ylims_tv =  {
            "dvdt_rising": [0, 2000],
            "dvdt_falling": [0, 1000],
            "AP_HW": [0, 1.0],
            "AP_thr_V": [-75, 0],
            "AdaptRatio": [0,2],
            "FISlope": [0, 1500],
            "maxHillSlope": [0, None],
            "I_maxHillSlope": [0, 250],
            "FIMax_1": [0, 800],
            "FIMax_4": [0, 800],
            "taum": [0, None],
            "RMP": [-80, -40],
            "Rin": [0, 400],
        }
        ylims_cw =  {
            "dvdt_rising": [0, 2000],
            "dvdt_falling": [0, 1000],
            "AP_HW": [0, 1.0],
            "AP_thr_V": [-75, 0],
            "AdaptRatio": [0, 8],
            "FISlope": [0, 250],
            "maxHillSlope": [0, None],
            "I_maxHillSlope": [0,1000],
            "FIMax_1": [0, 800],
            "FIMax_4": [0, 800],
            "taum": [0, None],
            "RMP": [-80, -40],
            "Rin": [0, 200],
        }

        ylims_giant =  {
            "dvdt_rising": [0, 2000],
            "dvdt_falling": [0, 1000],
            "AP_HW": [0, 1.0],
            "AP_thr_V": [-75, 0],
            "AdaptRatio": [0,2],
            "FISlope": [0, 1000],
            "maxHillSlope": [0, 1000],
            "I_maxHillSlope": [0, 1000],
            "FIMax_1": [0, 800],
            "FIMax_4": [0, 800],
            "taum": [0, None],
            "RMP": [-80, -40],
            "Rin": [0, 400],
        }
        ylims_bushy =  {
            "dvdt_rising": [0, 200],
            "dvdt_falling": [0, 200],
            "AP_HW": [0, 3.0],
            "AP_thr_V": [-75, 0],
            "AdaptRatio": [0,2],
            "FISlope": [0, 1000],
            "maxHillSlope": [0, 1000],
            "I_maxHillSlope": [0, 1000],
            "FIMax_1": [0, None],
            "FIMax_4": [0, None],
            "taum": [0, None],
            "RMP": [-80, -40],
            "Rin": [0, 400],
        }

        ylims_tstellate =  {
            "dvdt_rising": [0, 400],
            "dvdt_falling": [0, 400],
            "AP_HW": [0, 3.0],
            "AP_thr_V": [-75, 0],
            "AdaptRatio": [0,2],
            "FISlope": [0, 1000],
            "maxHillSlope": [0, 1000],
            "I_maxHillSlope": [0, 1000],
            "FIMax_1": [0, None],
            "FIMax_4": [0, None],
            "taum": [0, None],
            "RMP": [-80, -40],
            "Rin": [0, 400],
        }

        ylims_dstellate =  {
            "dvdt_rising": [0, 400],
            "dvdt_falling": [0, 400],
            "AP_HW": [0, 3.0],
            "AP_thr_V": [-75, 0],
            "AdaptRatio": [0,2],
            "FISlope": [0, 1000],
            "maxHillSlope": [0, 1000],
            "I_maxHillSlope": [0, 1000],
            "FIMax_1": [0, None],
            "FIMax_4": [0, None],
            "taum": [0, None],
            "RMP": [-80, -40],
            "Rin": [0, 400],
        }

        ylims_octopus=  {
            "dvdt_rising": [0, 200],
            "dvdt_falling": [0, 200],
            "AP_HW": [0, 3.0],
            "AP_thr_V": [-75, 0],
            "AdaptRatio": [0,2],
            "FISlope": [0, 1000],
            "maxHillSlope": [0, 1000],
            "I_maxHillSlope": [0, 1000],
            "FIMax_1": [0, None],
            "FIMax_4": [0, None],
            "taum": [0, None],
            "RMP": [-80, -40],
            "Rin": [0, 400],
        }


        ylims = {"pyramidal": ylims_pyr, "tuberculoventral": ylims_tv, "cartwheel": ylims_cw, "giant": ylims_giant,
                "bushy": ylims_bushy, "t-stellate": ylims_tstellate, "d-stellate": ylims_dstellate, "octopus": ylims_octopus}
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
        print("set action: ", self.action)

    def pickEvent(self, event, ax):
        """Event that is triggered when mouse is clicked."""
        print("event index: ", event.ind)
        print(dir(event.mouseevent))
        print(event.mouseevent.inaxes == ax)
        # print(ax == self.axis)
        print(self.data.iloc[event.ind])  # find the matching data.
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
    if 'protocol_durations' in experiment.keys():
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

GROUP_MAP: dict = {}  # maps group names to other names for consistency

def rename_groups(row):
    # print("renaming row group: ", row.Group)
    if row.Group in list(GROUP_MAP.keys()):
        row.Group = GROUP_MAP[row.Group]
    return row.Group


printflag = False
pd.set_option("display.max_columns", 40)


# def setup(expt, region="DCN"):
#     adddata = None
#     print("expt: ", expt)
#     if region == "DCN":
#         if expt.endswith("NIHL"):
#             excelsheet = "NIHL_IVs_PCT.xlsx"
#             # adddata = "cleaned_NIHL_IVs_PCT.xlsx"
        
#         elif expt.endswith("_Age"):
#             excelsheet = Path(
#                 experiment["analyzeddatapath"],
#                 experiment["directory"],
#                 "CBA_Age_28_Dec_2023.xlsx",
#             )  #  "Het_AP_PCT.xlsx"
#             adddata = Path(
#                 experiment["analyzeddatapath"],
#                 experiment["directory"],
#                 "CBA_IVs_PCT_cleaned.xlsx",
#             )

#             analysis_cell_types = [
#                 "pyramidal",
#                 "tuberculoventral",
#                 "cartwheel",
#                 "giant",
#             ]
#         elif expt.endswith("Het"):
#             excelsheet = "Het_AP_PCT.xlsx"
#             adddata = "cleaned_Het_AP_PCT.xlsx"
#             analysis_cell_types = [
#                 "pyramidal",
#                 "tuberculoventral",
#                 "cartwheel",
#                 # "giant",
#             ]
#     # excelsheet = "test.xlsx"
#     elif region == "VCN":
#         if expt.endswith("NIHL"):
#             excelsheet = "nihl_AP_BuSt.xlsx"
#         else:
#             excelsheet = "Het_AP_BuSt.xlsx"
#         analysis_cell_types = ["bushy", "stellate"]
#     else:
#         raise ValueError(f"Region {region:s} is not known")

#     return excelsheet, analysis_cell_types, adddata


def setup(experiment):
    excelsheet = Path(
                experiment["analyzeddatapath"],
                experiment["directory"],
                experiment['datasummaryFilename']
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
    "AP_thr_V",
    "AP_HW",
    "AP15Rate",
    "AdaptRatio",
    "AHP_trough_V",
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
    "AP_HW",
    "AP15Rate",
    "AdaptRatio",
    "AHP_trough_V",
    "AHP_depth_V",
    "tauh",
    "Gh",
    "FiringRate",
]



"""
Each protocol has:
"Spikes: ", dict_keys(['FI_Growth', 'AdaptRatio', 'FI_Curve', 'FiringRate', 'AP1_Latency', 'AP1_HalfWidth', 
'AP1_HalfWidth_interpolated', 'AP2_Latency', 'AP2_HalfWidth', 'AP2_HalfWidth_interpolated',
    'FiringRate_1p5T', 'AHP_Depth', 'AHP_Trough', 'spikes', 'iHold', 
    'pulseDuration', 'baseline_spikes', 'poststimulus_spikes'])

"IV": dict_keys(['holding', 'WCComp', 'CCComp', 'BridgeAdjust', 'RMP', 'RMP_SD', 'RMPs',
'Irmp', 'taum', 'taupars', 'taufunc', 'Rin', 'Rin_peak', 'tauh_tau', 'tauh_bovera', 'tauh_Gh', 'tauh_vss'])

individual spike data:
spike data:  dict_keys(['trace', 'AP_number', 'dvdt', 'V', 'Vtime', 'pulseDuration', 'tstart', 'tend', 
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


def nantoB_groups(row):
    row.age = numeric_age(row)
    for k in experiment['age_definitions'].keys():
        if row.age >= experiment['age_definitions'][k][0] and row.age <= experiment['age_definitions'][k][1]:
            row.Group = k
    return row.Group

def make_datetime_date(row, colname="Date"):
    if pd.isnull(row[colname]) or row[colname] == "nan":
        row.shortdate = 0
        return row.shortdate

    date = str(Path(row[colname]).name)
    date = date.split("_")[0]
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
    def __init__(self, dataset, experiment):
        self.set_experiment(dataset, experiment)

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


    def read_data_files(self, excelsheet, adddata=None, analysis_cell_types=[]):
        """read_data_files _summary_

        Parameters
        ----------
        excelsheet : string or Path
            excel filename to read to get the data to plot
        adddata : _type_, optional
            analysis result data to merge with main excel sheet, by default None
        analysis_cell_types : list, optional
            a list of cell type names, to specify which cell types will be analyzed, by default []

        Returns
        -------
        _type_
            _description_
        """
        CP("r", "\n Reading data files")
        
        if Path(excelsheet).suffix == ".pkl":  # need to respecify as excel sheet
                excelsheet = Path(excelsheet).with_suffix(".xlsx")
        print(f"    Excelsheet: {excelsheet!s}")
        df = pd.read_excel(excelsheet)
        print("    # entries in excel sheet: ", len(df))
        if "cell_id" not in list(df.columns):
            df.apply(make_cell_id, axis=1)

        print("Adddata in read_data_files: ", adddata)

        if adddata is not None:
            print(f"    Adddata: {adddata!s}")
            ages = [f"P{i:2d}D" for i in range(12, 170)]
            df2 = pd.read_excel(adddata)
            print("    # entries in adddata: ", len(df2))
            df2 = df2[df2["cell_type"].isin(analysis_cell_types)]  # limit to cell types
            df2 = df2[df2["age"].isin(ages)]  # limit ages
            df2["Date"] = df2["date"]
            df2 = df2.apply(make_cell_id, axis=1)
            # print(df2[df2["cell_type"] == "cartwheel"])
            # print(df2[df2["cell_type"] == "pyramidal"])
            print("Add data len:  ", len(df2))
            df = pd.concat([df, df2], ignore_index=True)
            print(f"# entries in combined sheets: {len(df):d}")
        else:
            print("    No data to add")
        # df = df.dropna(subset=["dvdt_rising"])

        FD = filter_data.FilterDataset(df)
        df = FD.clean_data_files(
            df,
            remove_groups=["C", "D", "?", "X", "30D", "nan"],
            excludeIVs=experiment["excludeIVs"],
            exclude_internal=["cesium", "Cesium"],
            exclude_temperature=["25C", "room temp"],
            analysis_cell_types=analysis_cell_types,
            verbose=True,
        )
        df["dvdt_falling"] = -df["dvdt_falling"]
        CP("m", "Finished reading files\n")
        return df




    def combine_by_cell(self,
        df, celltypes=None, plot_fits=False, valid_protocols=None
    ):
        """
        Rules for combining cells and pulling the data from the original analysis:
        1. Combine data from cells with the same ID
        2. Check the cell name and whether it fits the S00C00 or S1C1 format.
        3. When getting spike parameters, use a logical set of restrictions:
            a. Use only the first spike at the 3 lowest current levels that evoke spikes
                for AP HW, AP_thr_V, AP15Rate, AdaptRatio, AHP_trough_V, AHP_depth_V
            b. Do not use traces that are above the spike firing rate turnover point (non-monotonic)

        """
        CP("r", "Combine by cell")

        print("1: starting number of entries", len(df))
        df = df.apply(make_cell_id, axis=1)
        print("1.5, before dropping empty ids: ", len(df))
        df.dropna(subset=["cell_id"], inplace=True)
        print("2: after dropping nan ids", len(df))
        df.rename(columns={"sex_x": "sex"}, inplace=True)
        df = df[df.cell_type.isin(celltypes)]
        print("3: in selected cell types", len(df))
        print("Combine by cell")

        df["shortdate"] = df.apply(
            make_datetime_date, colname="date", axis=1
        )  # make a short date as a datetime for sorting

        after_parsedts = after_parsed
        df = df[df["shortdate"] >= after_parsedts]
        print("dates in range: ", len(df))

        # print(df.head())
        # df = df[df["Spikes"].count() > 0]  # need spikes to count
        cell_list = list(set(df.cell_id))
        cells = sorted(cell_list)
        dfdict = {col: [] for col in cols}
        df_new = pd.DataFrame.from_dict(dfdict)
        # print("cells: ", cells, len(cells))
        
        # do each cell in the database
        for icell, cell in enumerate(cells):
            if cell is None:
                continue
            datadict = FUNCS.compute_FI_Fits(experiment, df, cell, plot_fits=plot_fits)
            df_new = pd.concat(
                [df_new, pd.Series(datadict).to_frame().T], ignore_index=True
            )        
        return df_new


    def oldcode(self, df_new):
        # make sure columns are numeric in the end:
        for m in datacols:
            df_new[m] = pd.to_numeric(df_new[m])
        df_stats = pd.DataFrame(
            {
                "Subject": df_new.cell_id,
                "Group": df_new.Group,
                "celltype": df_new.cell_type,
                "current": df_new.current,
                "spsec": df_new.spsec,
                "pars": df_new.pars,
                "names": df_new.names,
                "fit": df_new.fit,
                "F1amp": df_new.F1amp,
                "F2amp": df_new.F2amp,
                "Irate": df_new.Irate,
                "Slope": df_new.FISlope,
            }
        )
        # print(df_new.current)
        # print(df_new.spsec)

        # df_stats = df_stats.explode(["current", "spsec"], ignore_index=True)
        # df_stats['current'] = pd.to_numeric(df_stats['current'])
        # df_stats['spsec'] = pd.to_numeric(df_stats['spsec'])
        # # print(df_stats['spsec'])
        # # df_id = df_stats[df_stats.celltype == "tuberculoventral"]
        # # mpl.plot(df_id['current'], df_id['spsec'])
        # # mpl.show()
        # # exit()
        # df_stats = df_stats.assign(pars=0.0).astype(object)
        # df_stats = df_stats.assign(names="")
        # df_stats = df_stats.assign(fit=0.0).astype(object)

        # df_statse = df_stats.explode(["current", "spsec"], ignore_index=True)
        # df_statse["current"] = df_statse["current"]*1e9
        # df_colwise= df_statse[[ "Subject", "Group", "current", "spsec"]]
        # df_stats['current'] = pd.to_numeric(df_stats['current'])
        # df_stats['spsec'] = pd.to_numeric(df_stats['spsec'])
        # df_stats.apply(printascol, axis=1)
        # df_stats_tv = df_stats[df_stats.celltype == "tuberculoventral"].transpose()

        # df_colwise.to_csv("TV_FI.csv")

        # groupsel = "A"
        # cellsel = "tuberculoventral"
        # currents = np.arange(0, 1.01, 0.05)
        # print("current shape: ", currents.shape)
        # dfa = df_new[df_new["Group"] == groupsel]
        # dfa = dfa[dfa["cell_type"] == cellsel]
        # sr = np.array(dfa["spsec"])
        # subjs = dfa["Subject"]
        # iinj = np.array(dfa[["current"]]).squeeze()*1e9
        # print("subject: ", subjs)
        # # for i, spks in enumerate(sr):
        # #     mpl.plot(iinj[i], spks)
        # #     mpl.plot(currents, spks, 'o')
        # # mpl.show()
        # subjects = ', '.join(s for s in subjs)
        # print("="*80)
        # print(f"{cellsel.upper():s}  Group: {groupsel:s}\n")
        # print(f"x, {subjects:s}")
        # for i in range(len(currents)):
        #     print(f"{currents[i]:.2f}, ", end="")
        #     for j, spks in enumerate(sr):
        #         print(f"{spks[i]:.1f}, ", end='')
        #     print()

        # return df_new

        anova = (
            False  # data are not normally distributed - Prob ominibus from anova is small
        )
        welch = False
        ancova = True

        def do_FI_fit(row):
            x = row.current
            y = row.spsec
            SA = spike_analysis.SpikeAnalysis()

            SA.fitOne(
                x=x, yd=y, function="fitOneOriginal", fixNonMonotonic=True, max_current=1e-9
            )
            if SA.analysis_summary["FI_Growth"][0]["parameters"] == []:
                print("No FI fit for cell: ", row.Subject)
            else:
                row.pars = [SA.analysis_summary["FI_Growth"][0]["parameters"][0]]
                row.names = SA.analysis_summary["FI_Growth"][0]["names"]
                row.fit = [SA.analysis_summary["FI_Growth"][0]["fit"]]
                # print(SA.analysis_summary['FI_Growth'][0]['parameters'][0][4])
                row.F1amp = SA.analysis_summary["FI_Growth"][0]["parameters"][0][2]
                row.F2amp = SA.analysis_summary["FI_Growth"][0]["parameters"][0][3]
                row.Irate = SA.analysis_summary["FI_Growth"][0]["parameters"][0][4]
            return row

        for celltype in celltypes:
            print("\nCelltype: ", celltype)
            ct_df = df_stats[df_stats["celltype"] == celltype]
            if not ancova:
                ct_df = ct_df.apply(do_FI_fit, axis=1)
            colors = {"B": "green", "A": "lightblue", "AA": "orange", "AAA": "magenta"}
            # f, ax = mpl.subplots(1,1,figsize=(8,8))
            if printflag:
                i = 0
                for index, row in ct_df.iterrows():
                    if len(row.fit) == 0 or row.fit == np.nan:
                        continue
                    fit = np.array(row.fit).squeeze()
                    # ax.plot(row.current*1e9, row.spsec, 'o--', color=colors[row.Group], linewidth=0.4)
                    # ax.plot(fit[0,:]*1e9, fit[1,:], '-', color=colors[row.Group])
                    # print(f"{row.celltype:s}, {row.Subject:s}", end=": ")
                    for ip, names in enumerate(row.names):
                        for iname, name in enumerate(names):
                            print(f" {name:<8s} = {row.pars[0][iname]:9.6g}", end=" ")
                        print()
                        # ax.plot(row.current, y, '-', color=colors[row.Group])
                # mpl.show()

            # for celltype in ['pyramidal', 'tuberculoventral', 'cartwheel']:
            ct_df = ct_df[ct_df["celltype"] == celltype]
            print(ct_df.head())

            for par in ["F1amp", "F2amp", "Irate"]:
                formula = f"{par:s} ~ C(Group)"

                if anova:
                    model = smf.ols(formula=formula, data=ct_df, missing="drop").fit()
                    print(f"Celltype: {celltype:s}, Parameter: {par:s}")
                    print(model.summary())
                    table = sa.stats.anova_lm(model, typ=2)
                    print(table)
                    post = sp.posthoc_ttest(
                        ct_df, val_col=par, group_col="Group", p_adjust="holm"
                    )
                    print("Posthocs: \n", post)
                    # print(table.summary)
                    print("=" * 80)
                if welch:
                    aov = ping.welch_anova(dv=par, between="Group", data=ct_df)
                    print(par, celltype)
                    print("-" * 80, "\n", "Welch Anova: \n", aov)
                    # posthoc = sp.posthoc_mannwhitney(df[df['celltype'] == celltype], val_col=measure, group_col='Group', p_adjust = 'holm-sidak')
                    # print("KW posthoc: ", posthoc)
                    # unequal variances, use Games-Howell test:
                    combinations = [["B", "A"], ["B", "AA"]]  # , ["B", "AAA"]]
                    print(ping.pairwise_gameshowell(dv=par, between="Group", data=ct_df))
                    print("=" * 80)
                    # if ancova:
                    #     # use current and fpsec as covariates
                    #     print(df_new['current'])
                    #     res = ping.ancova(data=df_new, dv='spsec', covar='current', between='Group')
                    #     print(res)
                if ancova:
                    ct_group = df_new[df_new["celltype"] == celltype]
                    print(ct_group["spsec"].values)
                    lmfit = smf.mixedlm(
                        "spsec ~ current + Subject", ct_group, groups=ct_group["Group"]
                    ).fit()
                    print(lmfit.summary())

        return df_new


    def to_1D(self, series):
        return pd.Series([x for _list in series for x in _list])


    def clean_alt_list(lself, ist_):
        list_ = list_.replace(", ", '","')
        list_ = list_.replace("[", '["')
        list_ = list_.replace("]", '"]')
        list_ = list_.replace("\n", "")
        return list_


    def print_for_prism(self, row, celltype="tuberculoventral"):
        if row.celltype != celltype:
            return
        print("")
        print(row.celltype)
        print(f"S:{row.Subject:s}")
        for i, x in enumerate(row.current):
            print(f"{x*1e9:.2f}  {row.spsec[i]:.1f}")
        print("")





    def bar_pts(self,
        df,
        yname,
        celltype: str = "pyramidal",
        sign: float = 1.0,
        scale=1.0,
        ax: mpl.axis = None,
        porder=["Ctl", "103dB 2W", "115dB 2W", "115dB 3D"],
        colors: Union[dict, None] = None,
        enable_picking=True,
    ):
        """Graph a bar pot and a strip plot on top of each other

        Args:
            df (Pandas data frome): _description_
            yname (str): dataset name to plot
            celltype (str, optional): Cell type to plot. Defaults to "pyramidal".
            ax (object, optional): Axis to plot into. Defaults to None.
            sf (float, optional): Scale factor to multipy data by.
        """
        df_x = df[df["cell_type"] == celltype]
        df_x.dropna(subset=["Group"], inplace=True)
        df_x = df_x.apply(apply_scale, axis=1, measure=yname, scale=sign * scale)
            # print('yname: ', yname)
        if colors is None:  # set all to blue
            colors = {df_x[g]: "b" for g in df_x.Group.unique()}

        # print(porder, df_x.Group.unique())
        # print("df yname: ", celltype, yname)
        # print(df_x["Group"], df_x[yname])
        # print(df_x[yname].isnull().values.all(axis=0))
        # if len(df_x) > 0 and celltype != 'cartwheel' and yname != 'FIMax_4':
        if not df_x[yname].isnull().values.all(axis=0):
            try:
                sns.boxplot(
                    data=df_x,
                    x="Group",
                    y=yname,
                    ax=ax,
                    order=porder,
                    color="w",
                    orient="v",
                    showfliers=False,
                )
            except:
                print("boxplot failed for ", celltype, yname)
            # print("porder: ", porder)
            # print("colors: ", colors)
            palette = [colors[g] for g in colors.keys()]
            # print("palette: ", palette )
            sns.stripplot(
                x="Group",
                y=yname,
                data=df_x,
                dodge=False,
                jitter=True,
                size=3.5,
                # fliersize=None,
                alpha=0.6,
                ax=ax,
                hue="Group",
                palette=palette,
                color="k",
                order=porder,
            )
        # print("bar pts picking enabled: ", enable_picking)
        # if not enable_picking:
        #     raise ValueError("Picking not enabled")
            if enable_picking:
                ax.scatter(x=df_x["Group"], y=df_x[yname], c="k", s=30, picker=True)
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
            picker_func = Picker()  # create a new insatnce of the picker class
            picker_func.setData(df_x.copy(deep=True))  # and set the data
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
        print(data.size, data.shape)
        print("row pars: ", row.pars)
        print("data: ", data)
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


    def relabel_axes(self, P):
        # relabel axes by dict:
        relabel_dict = {
            "Group": None,
            "dvdt_rising": "Rising dV/dt (mV/ms)",
            "dvdt_falling": "Falling dV/dt (mV/ms)",
            "AP_HW": "AP Halfwidth (ms)",
            "AP_thr_V": "AP Threshold (mV)",
            "AdaptRatio": "Adaptation Ratio",
            "FISlope": "FI Slope (sp/nA)",
            "maxHillSlope": "Max Hill Slope (sp/nA)",
            "I_maxHillSlope": "I at Max Hill Slope (pA)",
            "FIMax_1": "Spike Rate at 1 nA",
            "FIMax_4": "Spike Rate at 4 nA",
            "taum": "Membrane Time Constant (ms)",
            "Rin": "Input Resistance (MOhm)",
            "RMP": "Resting Membrane Potential (mV)",
        }
        for ax_str in P.axdict:
            ax = P.axdict[ax_str]
            xlab = ax.get_xlabel()
            if xlab in relabel_dict.keys():
                ax.set_xlabel(relabel_dict[xlab])
            ylab = ax.get_ylabel()
            if ylab in relabel_dict.keys():
                ax.set_ylabel(relabel_dict[ylab])
        return



    transforms = {
        # "maxHillSlope": np.log10,
    }


    def rescale_values_apply(self, row, measure, scale=1.0):
        if measure in row.keys():
            row[measure] = row[measure] * scale
        return row[measure]


    rescaling = {
        "AP_HW": 1e3,
        "AP_thr_V": 1e3,
        "FISlope": 1e-9,
        "maxHillSlope": 1,
        "I_maxHillSlope": 1,
        "dvdt_falling": -1.0,
        "taum": 1e3,
    }


    def rescale_values(self, df):
        for measure in rescaling.keys():
            if measure not in df.columns:
                continue
            df[measure] = df.apply(
                rescale_values_apply, axis=1, measure=measure, scale=rescaling[measure]
            )
        return df



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


    def summary_plot_spike_parameters_categorical(self, 
        df,
        porder=["Ctl", "103dB 2W", "115dB 2W", "115dB 3D"],
        measure_cols=["dvdt_rising", "dvdt_falling", "AP_HW", "AP_thr_V"],
        # "AdaptRatio", "FISlope", "FIMax_1", "FIMax_4"],
        colors=None,
        grouping="named",
        enable_picking=False,
    ):
        """Make a summary plot of spike parameters for selected cell types.

        Args:
            df (Pandas dataframe): _description_
        """
        df = df.copy()  # make sure we don't modifiy the incoming
        ncols = len(measure_cols)
        letters = ["A", "B", "C", "D", "E", "F", "G", "H"]
        plabels = [f"{let:s}{num+1:d}" for let in letters for num in range(ncols)]

        # df["FIMax_1"] = df.apply(getFIMax_1, axis=1)
        # df["FIMax_4"] = df.apply(getFIMax_imax, axis=1, imax=4.0)
        df["FIRate"] = df.apply(self.getFIRate, axis=1)
        df["Group"] = df.apply(rename_groups, axis=1)
        df["age"] = df.apply(numeric_age, axis=1)
        df["shortdate"] = df.apply(make_datetime_date, axis=1)
        df["SR"] = df.apply(self.flag_date, axis=1)
        df.dropna(subset=["age"], inplace=True)
        df.dropna(subset=["Group"], inplace=True)
        df = df[df["Group"] != "nan"]
        # df = df[(df.Age > 0) & (df.Age <= max_age)]
        figure_width = ncols * 2.5
        # print("meausre cols: ", measure_cols)
        P = PH.regular_grid(
            len(analysis_cell_types),
            ncols,
            order="rowsfirst",
            figsize=(figure_width, 11),
            panel_labels=plabels,
            labelposition=(-0.05, 1.05),
            margins={
                "topmargin": 0.08,
                "bottommargin": 0.1,
                "leftmargin": 0.1,
                "rightmargin": 0.05,
            },
            verticalspacing=0.04,
            horizontalspacing=0.07,
        )
        label_celltypes(P, analysis_cell_types=analysis_cell_types)
        relabel_axes(P)
        for ax in P.axdict:
            PH.nice_plot(P.axdict[ax], direction="outward", ticklength=3, position=-0.03)
        picker_funcs = {}
        n_celltypes = len(analysis_cell_types)
        df = rescale_values(df)
        for icol, measure in enumerate(measure_cols):
            if measure in transforms.keys():
                tf = transforms[measure]
            else:
                tf = None
            for i, celltype in enumerate(analysis_cell_types):
                axp = P.axdict[f"{letters[i]:s}{icol+1:d}"]

                picker_funcs[axp] = create_one_plot_categorical(
                    data=df,
                    x=porder,
                    y=measure,
                    ax=axp,
                    celltype=celltype,
                    colors=colors,
                    logx=False,
                    ylims=ylims[celltype][measure],
                    transform=tf,
                    xlims=None,
                    enable_picking=enable_picking,
                )

                if celltype != analysis_cell_types[-1]:
                    axp.set_xticklabels("")
                    axp.set_xlabel("")
                    print("removed xlabel from ", icol, measure, celltype)
                # if icol > 0:
                #     axp.set_ylabel("")
                #     print("removed ylabel from ", icol, measure, celltype)

        if any(picker_funcs) is not None and enable_picking:
            P.figure_handle.canvas.mpl_connect(
                "pick_event", lambda event: pick_handler(event, picker_funcs)
            )
        else:
            picker_funcs = None

        for ax in P.axdict:
            if P.axdict[ax].legend_ is not None:
                P.axdict[
                    ax
                ].legend_.remove()  # , direction="outward", ticklength=3, position=-0.03)
        # place_legend(P)

        return P, picker_funcs


    


    def pick_handler(self, event, picker_funcs):
        for pf in picker_funcs.keys():
            if event.mouseevent.inaxes == pf:
                # print("\nDataframe index: ", event.ind)
                cell = picker_funcs[pf].data.iloc[event.ind]
                print(f"   {cell['cell_id'].values[0]:s}")  # find the matching data.
                age = get_age(cell["age"])
                print(
                    f"       {cell['cell_type'].values[0]:s}, Age: P{age:3d}D Group: {cell['Group'].values[0]!s}"
                )
                print(f"       Protocols: {cell['protocols'].values[0]!s}")
                print(cell['dvdt_rising'], cell['dvdt_falling'])
                return cell["cell_id"].values[0]
        return None


    def create_one_plot_continuous(self,
        data,
        x,
        y,
        ax,
        celltype:str,
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
        dfp = data[data["cell_type"] == celltype]
        dfp[y] = dfp[y] * yscale
        if transform is not None:
            dfp[y] = dfp[y].apply(transform, axis=1)
        sns.regplot(
            data=dfp,
            x=x,
            y=y,
            ax=ax,
            logx=logx,
        )
        ax.scatter(x=dfp[x], y=dfp[y], c="b", s=4, picker=True)
        if ylims is not None:
            ax.set_ylim(ylims[celltype][y])
        if xlims is not None:
            ax.set_xlim(xlims)
        picker_func = Picker()
        picker_func.setData(dfp.copy(deep=True))
        return picker_func
        # picker_funcs[celltype].setAction(handle_pick) # handle_pick is a function that takes the index of the picked point as an argument


    def apply_scale(self, row, measure, scale):
        row[measure] = row[measure] * scale
        return row


    def create_one_plot_categorical(self, 
        data,
        x,
        y,
        ax,
        celltype,
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
        dfp = data[data["cell_type"] == celltype]
        dfp = dfp.copy()
        dfp = dfp.apply(self.apply_scale, axis=1, measure=y, scale=yscale)
        if transform is not None:
            dfp[y] = dfp[y].apply(transform)
        # print("1pcat")
        picker_func = self.bar_pts(
            dfp,
            yname=y,
            ax=ax,
            celltype=celltype,
            porder=x,
            colors=colors,
            enable_picking=enable_picking,
        )
        if ylims is not None:
            ax.set_ylim(ylims)
        if xlims is not None:
            ax.set_xlim(xlims)

        if picker_func is not None:
            picker_func.setData(dfp.copy(deep=True))
        else:
            picker_func = None
        return picker_func


    def clean_Rin(self, row):
        if row.Rin < 6.0:
            row.Rin = np.nan
        return row.Rin


    def summary_plot_spike_parameters_continuous(self, df, logx=False, xlims=[0, max_age]):
        """Make a summary plot of spike parameters for selected cell types.

        Args:
            df (Pandas dataframe): _description_
        """
        picker_funcs = {}
        measure_cols = [
            "dvdt_rising",
            "dvdt_falling",
            "AP_HW",
            "AP_thr_V",
            "AdaptRatio",
            "FISlope",
            "maxHillSlope",
            "FIMax_1",
            "FIMax_4",
        ]

        ncols = len(measure_cols)
        letters = ["A", "B", "C"]
        plabels = [f"{let:s}{num+1:d}" for let in ["A", "B", "C"] for num in range(ncols)]
        df = df.copy()
        # df["FIMax_1"] = df.apply(getFIMax_1, axis=1)
        # df["FIMax_4"] = df.apply(getFIMax_imax, axis=1, imax=4.0)
        df["FIRate"] = df.apply(self.getFIRate, axis=1)
        df["Group"] = df.apply(rename_groups, axis=1)
        df["age"] = df.apply(numeric_age, axis=1)
        df["shortdate"] = df.apply(make_datetime_date, axis=1)
        df["SR"] = df.apply(self.flag_date, axis=1)
        df.dropna(subset=["age"], inplace=True)
        df = df[(df.Age > 0) & (df.Age <= max_age)]
        # df = df[df.SR == 1]

        P = PH.regular_grid(
            3,
            ncols,
            order="rowsfirst",
            figsize=(16, 9),
            panel_labels=plabels,
            labelposition=(-0.05, 1.05),
            margins={
                "topmargin": 0.1,
                "bottommargin": 0.1,
                "leftmargin": 0.1,
                "rightmargin": 0.15,
            },
            verticalspacing=0.12,
            horizontalspacing=0.05,
        )
        for ax in P.axdict:
            PH.nice_plot(P.axdict[ax], direction="outward", ticklength=3, position=-0.03)

        showcells = analysis_cell_types
        for i, celltype in enumerate(showcells):
            let = letters[i]
            dfp = df[df["cell_type"] == celltype]
            axp = P.axdict[f"{let:s}1"]
            for icol, measure in enumerate(measure_cols):
                if measure in transforms.keys():
                    tf = transforms[measure]
                    dfp[measure] = dfp[measure].apply(tf, axis=1)
                else:
                    tf = None
                axp = P.axdict[f"{let:s}{icol+1:d}"]

                picker_funcs[axp] = self.create_one_plot_continuous(
                    data=dfp,
                    x="age",
                    y=measure,
                    ax=axp,
                    celltype=celltype,
                    logx=logx,
                    ylims=ylims,
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

        for i in range(len(showcells)):
            mpl.text(
                x=0.05,
                y=0.8 - i * 0.3,
                s=showcells[i].title(),
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
        n_celltypes = len(analysis_cell_types)

        for i in range(n_celltypes):
            ax = P.axarr[i, 0]
            yp = ax.get_position()
            mpl.text(
                x=0.02,
                y=yp.y0 + 0.5*(yp.y1 - yp.y0),  # center vertically
                s=analysis_cell_types[i].title(),
                color="k",
                rotation="vertical",
                ha="center",
                va="center",
                fontsize="large",
                fontweight="bold",
                transform=P.figure_handle.transFigure,
            )
        return


    def summary_plot_RmTau_categorical(self, 
        df, porder, colors=None, grouping="named", enable_picking=True
    ):
        """Make a summary plot of spike parameters for selected cell types.

        Args:
            df (Pandas dataframe): _description_
        """
        df = df.copy()
        letters = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]
        measures = ["RMP", "Rin", "taum"]
        plabels = [
            f"{let:s}{num+1:d}" for let in letters for num in range(len(measures))
        ]
        df["Rin"] = df.apply(self.clean_Rin, axis=1)
        # print("before: ", df["Group"])
        df["Group"] = df.apply(rename_groups, axis=1)
        df = df[df["Group"] != "nan"]
        self.rescale_values(df)
        # print("after: ", df["Group"])
        # print("porder: ", porder)
        # exit()

        picker_funcs = {}
        figure_width = len(measures) * 2.75
        P = PH.regular_grid(
            len(analysis_cell_types),
            len(measures),
            order="rowsfirst",
            figsize=(figure_width, 11),
            panel_labels=plabels,
            labelposition=(-0.05, 1.05),
            margins={
                "topmargin": 0.1,
                "bottommargin": 0.1,
                "leftmargin": 0.12,
                "rightmargin": 0.05,
            },
            verticalspacing=0.04,
            horizontalspacing=0.07,
        )
        self.label_celltypes(P, analysis_cell_types=analysis_cell_types)
        self.relabel_axes(P)

        for ax in P.axdict:
            PH.nice_plot(P.axdict[ax], direction="outward", ticklength=3, position=-0.03)
        for i, celltype in enumerate(analysis_cell_types):
            let = letters[i]

            for j, measure in enumerate(measures):
                #     if measure in rescaling.keys():
                #         yscale = rescaling[measure]
                #     else:
                #         yscale = 1

                # df = df.apply(apply_scale, axis=1, measure=measure, scale=yscale)
                axp = P.axdict[f"{let:s}{j+1:d}"]

                print("summary_plot_categorical  enable picking: ", enable_picking)
                picker_funcs[axp] = self.create_one_plot_categorical(
                    data=df,
                    x=porder,
                    y=measure,
                    ax=axp,
                    celltype=celltype,
                    colors=colors,
                    logx=False,
                    ylims=ylims[celltype][measure],
                    xlims=None,
                    enable_picking=enable_picking,
                )

                # pickerfunc = bar_pts(df, yname=measure, ax=axp, celltype=celltype, porder=porder)
                # picker_funcs[axp] = pickerfunc
                if celltype != analysis_cell_types[-1]:
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
        # place_legend(P)


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

        i_limit = np.argwhere(fi_inj <= imax)[-1][0] + 1 # only consider currents less than imax
        fi_rate = fi_rate[:i_limit]
        fi_inj = fi_inj[:i_limit]
        spike_rate_max_index = np.argmax(fi_rate)  # get where the peak rate is located
        spike_rate_max = fi_rate[spike_rate_max_index]
        spike_rate_slope: np.ndarray = np.gradient(fi_rate, fi_inj)
        xnm = np.where(
            (spike_rate_slope[spike_rate_max_index:] <= 0.0)  # find where slope is negative
            &
            (fi_rate[spike_rate_max_index:] < (limit * spike_rate_max))
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
        # mpl.plot(fi_inj, fi_rate)
        # mpl.title(id)
        # mpl.show()
        return np.array([fi_inj, fi_rate])


    def summary_plot_FI(self,
        df,
        protosel: List = [
            "CCIV_1nA_max",
            "CCIV_1nA_Posonly",
            # "CCIV_4nA_max",
            "CCIV_long",
        ],  # , "CCIV_"],
        mode=["individual"],
        grouping: str = "named",
        colors: Union[dict, None] = None,
        enable_picking: bool = False,
    ):
        print("summaryplotFI")
        df = df.copy()

        df = df.rename({"iv_name": "protocol"}, axis="columns")

        plabels = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]
        P = PH.regular_grid(
            len(analysis_cell_types),
            1,
            order="rowsfirst",
            figsize=(4, 11),
            panel_labels=plabels,
            labelposition=(-0.05, 1.05),
            margins={
                "topmargin": 0.1,
                "bottommargin": 0.1,
                "leftmargin": 0.15,
                "rightmargin": 0.15,
            },
            verticalspacing=0.05,
            horizontalspacing=0.1,
            fontsize={"label": 12, "tick": 8, "panel": 16},
        )
        for ax in P.axdict:
            PH.nice_plot(P.axdict[ax], direction="outward", ticklength=3, position=-0.03)
        # First, limit to just one IV curve type
        allprots = []
        for protocol in protosel:
            if protocol != "CCIV_":
                allprots.extend([f"{protocol:s}_{i:03d}" for i in range(3)])
            else:
                allprots.extend([protocol])
        # print("allprots: ", allprots)
        ymax = {
            "pyramidal": 500,
            "tuberculoventral": 600,
            "cartwheel": 200,
            "bushy": 100,
            "stellate": 800,
            "t-stellate": 800,
            "d-stellate": 800,
            "giant": 800,
            "octopus": 100,
        }
        pos = {
            "pyramidal": [0.4, 0.15],
            "tuberculoventral": [0.4, 0.15],
            "cartwheel": [0.4, 0.15],
            "bushy": [0.05, 0.95],
            "stellate": [0.6, 0.4],
            "t-stellate": [0.6, 0.4],
            "d-stellate": [0.6, 0.4],
            "giant": [0.6, 0.4],
        }
        # if grouping == "letters":
        #     colors = {
        #         "B": "green",
        #         "A": "blue",
        #         "AA": "darkorange",
        #         "AAA": "magenta",
        #         #   "J": "green", "Adol": "blue", "YA": "darkorange", "Adult": "magenta",
        #     }
        # elif grouping == "named":
        #     colors = {
        #         "Ctl": "green",
        #         "103dB 2W": "blue",
        #         "115dB 2W": "darkorange",
        #         "115dB 3D": "magenta",
        #         #   "J": "green", "Adol": "blue", "YA": "darkorange", "Adult": "magenta",
        #     }
        # else:
        #     raise (ValueError("grouping must be 'letters' or 'named'"))

        # P.figure_handle.suptitle(f"Protocol: {','.join(protosel):s}", fontweight="bold", fontsize=18)
        if mode == "mean":
            P.figure_handle.suptitle(f"FI Mean, SEM ", fontweight="normal", fontsize=18)
        elif mode == "individual":
            P.figure_handle.suptitle(f"FI Individual", fontweight="normal", fontsize=18)

        fi_stats = []
        NCells = {}
        picker_funcs = {}

        for i, celltype in enumerate(analysis_cell_types):
            if celltype not in NCells.keys():
                NCells[celltype] = 0
            print("cell types to analyze: ", analysis_cell_types)
            ax = P.axdict[plabels[i]]
            ax.set_title(celltype.title(), y=1.05)
            ax.set_xlabel("I$_{inj}$ (nA)")
            ax.set_ylabel("Rate (sp/s)")

            cdd = df[df["celltype"] == celltype]
            print("Unique CCIVs: ", cdd["protocol"].unique())
            if grouping == "letters":
                N = {"A": 0, "B": 0, "AA": 0, "AAA": 0}
                N = {
                    "Preweaning": 0,
                    "Pubescent": 0,
                    "Young Adult": 0,
                    "Mature Adult": 0,
                }

            elif grouping == "named":
                N = {"Ctl": 0, "103dB 2W": 0, "115dB 2W": 0, "115dB 3D": 0}
                N = {
                    "Preweaning": 0,
                    "Pubescent": 0,
                    "Young Adult": 0,
                    "Mature Adult": 0,
                }
                # cdd["Group"] = cdd.apply(rename_groups, axis=1)

            if "mean" in mode:
                if grouping == "letters":
                    FIy_all: dict = {"A": [], "B": [], "AA": [], "AAA": []}
                    FIx_all: dict = {"A": [], "B": [], "AA": [], "AAA": []}
                elif grouping == "named":
                    FIy_all: dict = {k: [] for k in N.keys()}

                    FIx_all: dict = {k: [] for k in N.keys()}
                    
            for index in cdd.index:
                group = cdd["Group"][index]
                # print("index: ", index, "Group: ", group)
                if (celltype, group) not in NCells.keys():
                    NCells[(celltype, group)] = 0

                if group == "?":
                    continue
                if group not in N.keys():
                    continue
                N[group] += 1
                # if cdd["protocol"][index].startswith("CCIV_"):
                #     this_protocol = "CCIV_"
                # else:
                this_protocol = cdd["protocol"][index]  # [
                #                    :-4
                #               ]  # get protocol, strip number
                # print("This protocol: ", this_protocol, "sel: ", protosel, cdd["protocol"][index])
                # exit()
                # if this_protocol not in protosel:
                #     continue
                # print("CCIV: \n", cdd["FI_Curve"][index], "\n")
                FI_data = FUNCS.convert_FI_array(cdd["FI_Curve"][index])

                if len(FI_data[0]) == 0:
                    continue
                # print("max FI_data current: ", np.max(FI_data[0]))
                FI_data[0] = np.round(np.array(FI_data[0])*1e9, 2)*1e-9
                # print("cell: ", cdd.cell_id[index])
                # print("FI_data: ", FI_data, FI_data.shape)
                if FI_data.shape == (2, 0):  # no fi data from the excel table....
                    continue
                FI_data = self.limit_to_max_rate_and_current(
                    FI_data, imax=1.05e-9, id=cdd["cell_id"][index]
                )
                print("fi_data[0] max: ", FI_data[0].max())
                NCells[(celltype, group)] += 1

                if "individual" in mode:
                    fix, fiy, fiystd, yn = FUNCS.avg_group(
                        np.array(FI_data[0]), FI_data[1], ndim=1
                    )

                    ax.plot(
                        fix*1e9,
                        fiy,
                        color=colors[group],
                        marker="o",
                        markersize=2.5,
                        linewidth=0.5,
                        clip_on=False,
                        alpha=0.35,
                    )
                if "mean" in mode:
                    FIy_all[group].append(np.array(FI_data[1]))
                    FIx_all[group].append(np.array(FI_data[0]) * 1e9)

            if "mean" in mode:
                for i, g in enumerate(FIy_all.keys()):
                    fx, fy, fystd, yn = FUNCS.avg_group(FIx_all[g], FIy_all[g])
                    if len(fx) == 0:
                        continue
                    print("mean in mode: max group average current: ", np.max(fx))
                    ax.errorbar(
                        fx,
                        fy,
                        yerr=fystd / np.sqrt(yn),
                        color=colors[g],
                        marker="o",
                        markersize=2.5,
                        linewidth=1.5,
                        clip_on=False,
                    )

            yoffset = 0.0
            xoffset = 0.0
            xo2 = 0
            if "individual" in mode:
                yoffset = 0.7
                xoffset = -0.35
                if celltype == "pyramidal":
                    xo2 = 0.0
                else:
                    xo2 = -0.45
            for i, g in enumerate(colors):
                if celltype == "pyramidal":
                    ax.text(
                        x=pos[celltype][0] - 0.05 + xoffset + xo2,
                        y=pos[celltype][1] - 0.095 * (i - 0.5) + yoffset,
                        s=f"{self.experiment['group_legend_map'][g]:<24s}",
                        ha="left",
                        va="bottom",
                        fontsize=8,
                        color=colors[g],
                        transform=ax.transAxes,
                    )
                # ax.text(
                #     x=pos[celltype][0] + 0.4 + xoffset + xo2,
                #     y=pos[celltype][1] - 0.0625 * (i - 0.5) + yoffset,
                #     s=f"N={N[g]:>3d} cells",
                #     ha="left",
                #     va="bottom",
                #     fontsize=8,
                #     color=colors[g],
                #     transform=ax.transAxes,
                # )

            ax.set_ylim(0, ymax[celltype])
            if "individual" in mode:
                ax.set_ylim(0, 750)
            ax.set_xlim(0, 1.0)
            print("-" * 80)

            if "mean" in mode:
                fi_stats.append(
                    {"celltype": celltype, "Group": g, "I_nA": fx, "sp_s": fy, "N": N[g]}
                )
        print("Ncells: ", NCells)
        # d_fi = pd.DataFrame(fi_stats)
        # print(d_fi.columns)
        # print(d_fi.sp_s)
        # for celltype in analysis_cell_types:
        #     model = smf.mixedlm("sp_s ~ I_nA",
        #             data = d_fi[d_fi['celltype'] == celltype],
        #             groups=d_fi['Group'],
        #             re_forumla = "~I_nA"
        #             ).fit()
        # print(model.summary())
        return P, picker_funcs


    def stats(self, df, celltype, measure):
        df_x = df[df.cell_type == celltype]
        # print(df_x.head())
        # fvalue, pvalue = scipy.stats.f_oneway(df['A'], df['B'], df['AA'], df['AAA'])

        stattext = ""
        cellcolors = {
            "pyramidal": "b",
            "cartwheel": "g",
            "tuberculoventral": "m",
            "giant": "k",
        }
        if celltype not in cellcolors.keys():
            cellcolors[celltype] = "r"
        stattext = "\n\n".join(
            [
                stattext,
                CP(
                    cellcolors[celltype],
                    f"Celltype: {celltype!s}   Measure: {measure!s}",
                    # textonly=True,
                ),
            ]
        )

        def make_numeric(row, measure):
            # print(row)
            # print(measure)
            row[measure] = pd.to_numeric(row[measure])
            return row

        df_x = df_x.apply(make_numeric, measure=measure, axis=1)
        df_clean = df_x.dropna(subset=[measure], inplace=False)
        if measure == "maxHillSlope":
            df_clean["maxHillSlope"] = np.log(df_clean["maxHillSlope"])
        lm = sfa.ols(f"{measure:s} ~ Group", data=df_clean).fit()
        anova = sa.stats.anova_lm(lm)
        stattext = "\n".join([stattext, CP("w", f"ANOVA: {anova!s}", textonly=True)])

        # ph = sp.posthoc_ttest(df_clean, val_col=measure, group_col="Group", p_adjust="holm")
        # print(ph)

        # res = stat()
        # res.anova_stat(df=df_pyr, res_var='dvdt_rising', anova_model='dvdt_rising ~ C(Group)')
        # print(res.anova_summary)
        from statsmodels.stats.multicomp import pairwise_tukeyhsd
        from statsmodels.stats.multitest import multipletests

        # sh = pairwise_tukeyhsd(df[measure], df["Group"])
        # print(sh)
        pw = lm.t_test_pairwise("Group", method="sh")
        subdf = pw.result_frame.loc[["B-A", "B-AA", "B-AAA"]]
        mtests = multipletests(subdf["P>|t|"].values, method="sh")
        # print("mtests: ", mtests)
        stattext = "\n".join([stattext, CP("w", f"{mtests!s}", textonly=True)])
        subdf["adj_p"] = mtests[1]
        # print("subdf: ", subdf)
        stattext = "\n".join([stattext, CP("w", f"{subdf!s}", textonly=True)])
        # res = stat()
        # res.tukey_hsd(df=df_pyr, res_var='dvdt_rising', xfac_var='Group', anova_model='dvdt_rising ~ C(Group)')
        # print(res.tukey_summary)
        # print("=" * 40)
        stattext = "\n".join([stattext, f"{'='*80:s}"])
        return stattext


    def check_HW(self, row):
        if row.AP_HW < 0 or row.AP_HW > 0.010:
            print(
                f"HW: {row.AP_HW:15.5f}, ID: {Path(row.cell_id).name!s}, Type: {row.cell_type:16s}, Group: {row.Group:5s}"
            )


    def check_APthr(self, row):
        print(
            f"APthr: {row.AP_thr_V:15.5f}, ID: {Path(row.cell_id).name!s}, Type: {row.cell_type:16s}, Group: {row.Group:5s}"
        )


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





    def get_assembled_filename(self, experiment):
        assembled_fn = Path(
                Path(
                    experiment["analyzeddatapath"],
                    experiment["directory"],
                    experiment["assembled_filename"]
                )
            )
        return assembled_fn

    def assemble_datasets(
        self,
        excelsheet: Union[Path, str],
        adddata: Union[str, None] = None,
        analysis_cell_types: list = [],
        fn: str = "",
        plot_fits: bool = False,
    ):
        print(f"Assembling:\n  Excel: {excelsheet!s}\n    Added: {adddata!s}\n    Cells: {analysis_cell_types!s}")


        df = self.read_data_files(excelsheet, adddata, analysis_cell_types)
        originalmax = np.max(df.dvdt_rising)

        df = self.combine_by_cell(df, plot_fits=plot_fits)
        print(originalmax, np.max(df.dvdt_rising))
        print("Writing assembled data to : ", fn)
        df.to_pickle(fn)


    def preload(self, fn):
        CP("g", f"PRELOAD, {fn!s}")
        df = pd.read_pickle(fn)
        df["Rin"] = df.apply(clean_Rin, axis=1)
        df["Group"] = df.apply(nantoB_groups, axis=1)
        print("len df reading pickle: ", len(df))
        # if "cell_id2" not in df.columns:
        #     df["cell_id2"] = df.apply(make_cell_id2, axis=1)
        print("cell ids: \n", df.cell_id)
        if self.experiment["excludeIVs"] is None:
            return df
        for fns in self.experiment["excludeIVs"]:
            fn = fns.split(":")[0]  # remove the comment
            print("checking exclude for: ", fn)
            if fn in list(df.cell_id.values):
                df.drop(df.loc[df.cell_id == fn].index, inplace=True)
                CP("r", f"dropped {fn:s} from analysis, reason = {fns.split(':')[1]:s}")
        return df


    def do_stats(self, df, analysis_cell_types, divider="-" * 80):
        stat_text = " "
        for ctype in analysis_cell_types:
            CP("g", f"\n{divider:s}")
            for measure in [
                "dvdt_rising",
                "dvdt_falling",
                "AP_thr_V",
                "AP_HW",
                # "AP15Rate",
                "AdaptRatio",
                "FISlope",
                "maxHillSlope",
                "I_maxHillSlope",
                "FIMax_1",
                # "FIMax_4",
                "RMP",
                "Rin",
                "taum",
            ]:
                stat_text = "\n".join([stat_text, stats(df, ctype, measure)])
        print(stat_text)
        return stat_text


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

    # porder = ["Juvenile", "Adolescent", "Young Adult", "Adult"]

    # Do this AFTER running process_spike_analysis

    if args.assemble_dataset:
        self.assemble_datasets(
            excelsheet=excelsheet,
            adddata=adddata,
            analysis_cell_types=analysis_cell_types,
            fn=fn,
            plot_fits=args.confirm,
        )
        exit()

    # df = read_data_files(excelsheet, adddata, analysis_cell_types)

    # Do this AFTER assembling the dataset to a pkl file
    print("Loading fn: ", fn)
    df = PSI.preload(fn)

    # df = df.apply(check_Group, axis=1)
    # df = df.apply(assign_default_Group, axis=1)

    # Generate stats...
    # print(
    #     "Code:\nB  : Control\nA  : 106 dBSPL 2H, 14D\nAA : 115 dBSPL 2H 14D\nAAA: 115 dBSPL 2H  3D\n"
    # )
    divider = "=" * 80 + "\n"
    if args.stats:
        PSI.do_stats(df, analysis_cell_types, divider=divider)
    porder = ["Preweaning", "Pubescent", "Young Adult", "Mature Adult"]
    # now we can do the plots:

    print(f"args.plot: <{args.plot:s}>")

    enable_picking = args.picking
    if args.plot in ["all", "spikes"]:
        P1, picker_funcs1 = PSI.summary_plot_spike_parameters_categorical(
            df,
            porder=PSI.experiment["plot_order"],
            colors=PSI.experiment["plot_colors"],
            grouping="named",
            enable_picking=enable_picking,
        )
        P1.figure_handle.suptitle("Spike Shape", fontweight="bold", fontsize=18)

    if args.plot in ["all", "firing"]:
        print("main: enable-picking: ", enable_picking)
        P2, picker_funcs2 = PSI.summary_plot_spike_parameters_categorical(
            df,
            measure_cols=[
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
        P3, picker_funcs3 = PSI.summary_plot_RmTau_categorical(
            df,
            porder=PSI.experiment["plot_order"],
            colors=PSI.experiment["plot_colors"],
            grouping="named",
            enable_picking=enable_picking,
        )
        P3.figure_handle.suptitle("Membrane Properties", fontweight="bold", fontsize=18)

    # summary_plot_spike_parameters_continuous(df, logx=True)
    # summary_plot_RmTau_continuous(df, porder=experiment["plot_order"])

    # dft = df[df["cell_type"] == "cartwheel"]
    # p = set([Path(p).name[:-4] for protocols in dft["protocols"] for p in protocols])

    if args.plot in ["all", "fi"]:
        P4, picker_funcs4 = PSI.summary_plot_FI(
            df,
            mode=["individual", "mean"],
            protosel=[
                "CCIV_1nA_max",
                # "CCIV_4nA_max",
                "CCIV_long",
                "CCIV_short",
                "CCIV_1nA_Posonly",
                # "CCIV_4nA_max_1s_pulse_posonly",
                "CCIV_1nA_max_1s_pulse",
                # "CCIV_4nA_max_1s_pulse",
            ],
            grouping="named",
            colors=PSI.experiment["plot_colors"],
            enable_picking=enable_picking,
        )

    mpl.show()
