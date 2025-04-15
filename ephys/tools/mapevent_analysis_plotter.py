"""
Third level of analysis for mapping data.

    The first level is nihl_maps.py - to do the initial analysis of the maps, event detection, etc. 
    The results from nihl_maps.py are written to the events folder, as individual files for each cell].
    The second level is mapevent_analyzer.py, which takes the individual files and
combines some of the results from the individual maps into a single file, NF107Ai32_NIHL_event_summary.pkl.

This third step provides routines to:
    1. generate the tau (event time constants) database (df = make_taus(basepath))
    2. merge the different databases into a single database (df = merge_db(db))
        This includes getting the taus and the event summary data, and providing
        additional columns for aggregated analysis and plotting.


"""
import importlib
import logging
import os
import re

import sys
from importlib import reload
from pathlib import Path

import ephys
import matplotlib
import matplotlib.pyplot as mpl
import numpy as np
import pandas as pd
import seaborn as sns

import statsmodels.api as sm
import statsmodels.stats
import statsmodels
import pingouin as pg


from statsmodels.formula.api import ols
import scipy.stats as stats
import scikit_posthocs as sp


from pylibrary.plotting import plothelpers as PH
from pypdf import PdfFileMerger, PdfFileReader
from pylibrary.tools import cprint as CP
Logger = logging.getLogger("MapAnalysis2_Log")

# set the experiment type
expt = "NIHL"
#expt = "Het"

order = [
    "bushy",
    "t-stellate",
    "d-stellate",
    "octopus",
    "pyramidal",
    "tuberculoventral",
    "giant",
    "cartwheel",
]
DCN_Cells = [
    'pyramidal',
    'tuberculoventral',
    'cartwheel',
]

if expt == "Het":
    import src.CBA_maps as NM

    # SEP.get_computer()
    expts = NM.experiments["NF107Ai32_Het"]  # SEP.get_experiments()
    # exclusions = SEP.get_exclusions()
    bpath = "../NF107Ai32_Het/"
    basepath = Path(expts["databasepath"], expts["directory"], "events")
    # fn = Path(basepath, '2018.08.01_000~slice_000~cell_001.pkl')
    database = Path(
        expts["databasepath"], expts["directory"], "NF107Ai32_Het_08Aug2023.pkl"
    )
    print(database, database.is_file())
    with open(database, "rb") as fh:
        db = pd.read_pickle(fh, compression=None)
    Groups  = ["B"]
    HueOrder = None
    CellTypes = ['bushy', 't-stellate', 'd-stellate', 'octopus', 'pyramidal', 'tuberculoventral', 'giant', 'cartwheel']

elif expt == "NIHL":
    # SEP.get_computer()
    import nihl_maps as NM

    expts = NM.experiments["NF107Ai32_NIHL"]  # SEP.get_experiments()
    bpath = "../NF107Ai32_NIHL/"
    basepath = Path(expts["databasepath"], expts["directory"], "events")
    database = Path(
        expts["databasepath"], expts["directory"], "NF107Ai32_NIHL_summary.pkl"
    )
    print("Database: ", database, database.is_file())
    with open(database, "rb") as fh:
        db = pd.read_pickle(fh, compression=None)
    Groups = ["B", "A", "AA", "AAA"]
    HueOrder = ["B", "A", "AA", "AAA"]
    CellTypes = ['pyramidal', 'cartwheel', 'tuberculoventral']

def get_coding(expts:str):
    if expts["coding_file"] is None:
        return None
    coding_file = Path(expts['analyzeddatapath'], 
        expts['directory'],
        expts["coding_file"])
    with open(coding_file, 'rb') as fh:
        coding = pd.read_excel(fh, sheet_name=expts['coding_sheet'],
        index_col=0, header=0)
    return coding



def merge_db(expts, db):
    """
    Pass the primary database in as db
    Combine event summary and taus with the main db to get ALL information on a per-cell basis

    Returns the merged pandas database
    """

    print("db columns: ", db.columns)
    
    # sanitize the cell ID in the db - rebuild the cellID from the date, slice, cell
    # removing the leading path information.
    print("Sanitizing cell_id in main database")
    for c in db.index:
        dx = db.iloc[c]
        date = str(Path(dx["date"]).name)
        cell = str(Path(date, dx["slice_slice"], dx["cell_cell"]))
        db.loc[c, "cell_id"] = cell

    taus = Path(
        expts["analyzeddatapath"], expts["directory"], f"NF107Ai32_{expt:s}_taus.pkl"
    )
    if not taus.is_file():
        raise FileNotFoundError(
            f"Could not find taus file: {taus!s}; run make_taus first"
        )
    events = Path(
        expts["analyzeddatapath"], expts["directory"], expts["eventsummaryFilename"]
    )
    
    if not events.is_file():
        raise FileNotFoundError(
            f"Could not find events file: {events!s}; run the mapevent_analyzer first"
        )
    with open(events, "rb") as fh:
        evdict = pd.read_pickle(fh, compression=None)  # not a pd dataframe yet
    evm = {}

    evdict = pd.DataFrame(evdict).transpose()
    evdict.reset_index(inplace=True)
    evdict.columns = evdict.columns.str.replace("index", "cellID")
    print("Read events file ok")

    if not taus.is_file():
        raise FileNotFoundError(
            f"Could not find taus file: {taus!s}; run make_taus first"
        )
    with open(taus, "rb") as fh:
        taudb = pd.read_pickle(fh, compression=None)
    # rename that first column with the full path to get it out of the way
    taudb.columns = taudb.columns.str.replace("cell_id", "cellID")
    taudb['cell_id'] = ''  # and make a new one for merging
    taudb.reset_index(inplace=True)
    taudb.dropna(subset=["cellID"], axis=0, inplace=True)

    for c in taudb.index:
        old_id = taudb.loc[c, "cellID"]
        new_id = old_id.split("~")[0]  # trim the "~map" from the end
        new_id = str(Path(*Path(new_id).parts[-3:]))
        taudb.loc[c, 'cell_id'] = new_id
    taudb.reset_index()
    taudb.to_excel(Path("temp_taus.xlsx"))
    print("Read tau file ok and cleaned up cell_id")

    print("merging on cell id for all 3 databases")
    midb = pd.merge(
        left=db, right=taudb, on=["cell_id", "cell_id"],
        how = "outer"
    )
    midb2 = pd.merge(
        left=midb, right=evdict, left_on="cell_id", right_on="cellID", how="outer"
    )
    midb2 = midb2[midb2.columns[~midb2.columns.str.endswith("_y")]]
    midb2.columns = midb2.columns.str.replace("_x", "")
    midb2.columns = midb2.columns.str.replace(".1", "")
    empty_column = ["" for _ in range(len(midb2))]
    midb2["Group"] = empty_column
    midb2["animalID"] = empty_column
    midb2['SPL'] = empty_column
    midb2['Recording_age'] = empty_column
    midb2['outlier'] = empty_column
    midb2["NE_Date"]   = empty_column
    midb2["ABR_Date"]  = empty_column
    midb2["maxscore"] = empty_column
    midb2["maxscore_thr"] = empty_column
    midb2["agegroup"] = empty_column
    midb2["latency"] = empty_column
    midb2["mean_amp"] = empty_column
    midb2["max_amp"] = empty_column
    midb2["avg_event_qcontent"] = empty_column
    midb2["avg_spont_amps"] = empty_column
    midb2["avg_largest_event_qcontent"] = empty_column

    # further annotation of the data set: Docing, ID, SPL, outlier
    print("Getting coding to assign group ID for each cell")
    
    coding = get_coding(expts)
    for index, c in enumerate(midb2.index):
        dx = midb2.iloc[c]
        #         print(midb2.loc[c, 'cell_type'],)
        if midb2.loc[c, "cell_type"] not in [None, "None"]:
            midb2.loc[c, "celltype"] = midb2.loc[c, "cell_type"]
        #         print('  ', midb2.loc[c, 'celltype'])

        # get assignment group, spl and animal ID for reference
        db_date = midb2.loc[c, 'date']
        if pd.isnull(db_date):
            continue
        dayname = str(Path(midb2.loc[c, 'date']).name).split('_')[0]
        if coding is not None:
            code = coding[coding['date'].str.endswith(dayname)]
            if code.empty:
                raise ValueError(f"Could not find coding for {dayname}")
            midb2.loc[c, 'Group'] = code['Group'].values[0]
            midb2.loc[c, 'animalID'] = code['ID'].values[0]
            midb2.loc[c, 'SPL'] = code['SPL'].values[0]
            midb2.loc[c, 'Recording_age'] = code['age'].values[0]
            midb2.loc[c, 'outlier'] = code['outlier'].values[0]
            midb2.loc[c, 'NE_Date'] = code['NE_Date'].values[0]
            if not pd.isnull(code['ABR_Date'].values[0]):
                midb2.loc[c, 'ABR_Date'] = code['ABR_Date'].values[0]
            else:
                midb2.loc[c, 'ABR_Date'] = "NA"
        else:
            midb2.loc[c, 'Group'] = "B"
            midb2.loc[c, "outlier"] = "no"
        # fill in some other possibly missing vdata
        if dx["temperature"] in ["room temp", "", " ", "room temperature", "25", 25]:
            #         print(f"resetting temp: <{str(dx['temperature']):s}>")
            midb2.loc[c, "temperature"] = "25C"
        #
        # set / parse age and age group
        re_age = re.compile("[~]*[pP]*[~]*(?P<age>[0-9]{1,3})[dD]*[, ]*")
        if dx["age"] is None or dx["age"] == "":
            m = None
        else:
            m = re_age.search(str(dx["age"]))
        if m is not None:
            value = int(m.group("age"))
        else:
            value = 0
        if value < 21:
            agegroup = 21  # 'P10-P21' 3 wk
        elif value < 35:
            agegroup = 35  # 'P21-P35' 5 wk
        elif value < 63:
            agegroup = 63  # 'P28-P56' 9 wk
        else:
            agegroup = 90  #'P65-P200'
        #         print('agegroup: ', agegroup)
        midb2.loc[c, "agegroup"] = agegroup
        
        # 
        # Now process summarize some other data for this cell
        #
        if index == 0:
            print("Adding some computations that are averages for each cell")

        midb2 = average_datasets(midb2, c, dx, "firstevent_latency", "latency", scf=1e3)
        midb2 = average_datasets(
            midb2, c, dx, "spont_amps", "avg_spont_amps", scf=-1e12
        )
        midb2 = average_datasets(midb2, c, dx, "event_qcontent", "avg_event_qcontent")
        midb2 = average_datasets(
            midb2, c, dx, "largest_event_qcontent", "avg_largest_event_qcontent"
        )

        #     lat = []
        #         for l in range(len(dx['firstevent_latency'])):
        #             if dx['firstevent_latency'][l] is not None:
        #                 lat.extend(dx['firstevent_latency'][l])
        #         if len(np.array(lat)) == 0:
        #             fevlat = np.nan
        #         else:
        #             fevlat = np.nanmean(np.array(lat).ravel())*1e3
        #         midb2.loc[c, 'latency'] = fevlat
        # print("Keys: ", dx.keys())
        mscore = np.max(dx["scores"])
        if mscore == 0:
            mscore = 1e-1
        midb2.loc[c, "maxscore"] = np.clip(np.log10(mscore), 0.1, 5.0)
        midb2.loc[c, "maxscore_thr"] = np.clip(np.log10(mscore), 0.1, 5.0) > 1.3
        midb2.loc[c, "mean_amp"] = np.nanmean(dx["Amplitude"])
        midb2.loc[c, "max_amp"] = np.nanmax(dx["Amplitude"])
    #         sa = []
    #         for l in range(len(dx['spont_amps'])):
    #             if dx['spont_amps'][l] is not None:
    #                 sa.extend(dx['spont_amps'][l])
    #         if len(np.array(sa)) == 0:
    #             san = np.nan
    #         else:
    #             san = np.nanmean(np.array(sa).ravel())*1e12
    #         midb2.loc[c, 'spontamps'] = san
    #     print(dx['cellID'], midb2.loc[c, 'max_amp'], dx['celltype'], dx['age'])
    print("Computing some averages")
    midb3 = compute_averages(midb2)
    print("Returning merged database with computations")
    return midb3, evdict


"""
Get the database and annotate it
"""

def annotate_db(db):
    print(db)
    # print('Now annotating: ')
    # annotationfile = Path(bpath, experiments['nf107']['annotation'])
    # annotated = pd.read_excel(annotationfile)

    # annotated.set_index("ann_index", inplace=True)
    # x = annotated[annotated.index.duplicated()]
    # if len(x) > 0:
    #     print("watch it - duplicated index in annotated file")
    #     print(x)
    #     raise ValueError()
    # # self.annotated.set_index("ann_index", inplace=True)
    # db.loc[db.index.isin(annotated.index), 'cell_type'] = annotated.loc[:, 'cell_type']
    # db.loc[db.index.isin(annotated.index), 'annotated'] = True

    # print('Annotation complete')

    # midb2 = {}
    midb2 = merge_db(db)
    # print(set(list(midb2['agegroup'])))
    # print(midb2[midb2['RMP'] < 90.])

    print(set(list(midb2["agegroup"])))
    print(midb2[midb2["RMP"] < -90.0]["cellID"], midb2[midb2["RMP"] < -90.0]["RMP"])
    # midb2 = remove_cells(midb2)
    # print(midb2[midb2['RMP'] < -90.]['cellID'], midb2[midb2['RMP'] < -90.]['RMP'])
    return midb2


### This is not necessary - we sanitize the cell names in mapevent_analyzer before
### writing the database.
# def renamecells(db):
#     for c in db.index:
#         dx = db.iloc[c]
#         cell = str(Path(dx['date'], dx['slice_slice'], dx['cell_cell']))
#         db.loc[c, 'cellID'] = cell
#     #     print('testing cell: ', cell, db.loc[c, 'cell_type'])
#         if db.loc[c, 'cell_type'] in ['T-stellate', 'T-Stellate']:
#             db.loc[c, 'cell_type'] = 't-stellate'
#         if db.loc[c, 'cell_type'] in ['D-stellate', 'D-Stellate']:
#             db.loc[c, 'cell_type'] = 'd-stellate'
#         if db.loc[c, 'cell_type'] in ['fusiform']:
#             db.loc[c, 'cell_type'] = 'pyramidal'
#         if db.loc[c, 'cell_type'] in ['Cartwheel', 'CW']:
#             db.loc[c, 'cell_type'] = 'cartwheel'

#         if db.loc[c, 'cell_type'] in ['Giant', 'GIANT', 'Giant cell']:
#             db.loc[c, 'cell_type'] = 'giant'
#         if db.loc[c, 'cell_type'] in ['GLIAL', 'glial', 'glia', 'glial cell']:
#             db.loc[c, 'cell_type'] = 'glial'
#         if db.loc[c, 'cell_type'] in [' ', 'tuberculoventral?', 'cartwheel?', 'glial cell',
#                                       'graule? tiny']:
#             db.loc[c, 'cell_type'] = 'unknown'
#         if db.loc[c, 'cell_type'] is None:
#             print('cell type is None for: ', cell)
#     return db


def remove_cells(db):
    removelist = ["2017.06.23_000/slice_003/cell_000"]
    for rm in removelist:
        db = db[db.cellID != rm]
    return db


def average_datasets(db, c, dx, measure, outmeasure, scf=1.0):
    sa = []
    if measure not in list(dx.keys()):
        return db
    if isinstance(dx[measure], float):
        sa.extend([dx[measure]])
    elif len(dx[measure]) == 1:
        #         print(dx[measure])
        sa.extend([dx[measure][0]])

    else:
        for l in range(len(dx[measure])):
            if dx[measure][l] is not None:
                if isinstance(dx[measure][l], float):
                    sa.extend([dx[measure][l]])
                else:
                    sa.extend(dx[measure][l])

    if len(sa) == 0 or len(np.array(sa)) == 0:
        san = np.nan
    elif np.array(sa).ndim == 2:
        san = np.nanmean(np.squeeze(np.array(sa))) * scf
    elif len(sa) == 1 and sa[0] == None:
        san = np.nan
    else:
        try:
            san = np.nanmean(np.squeeze(np.array(sa))) * scf
        except:
            print("failed to average: ", np.squeeze(np.array(sa)))
    #     print('san: ', san)
    db.loc[c, outmeasure] = san
    return db


def compute_averages(db):
    """
    Compute some averages from the ivs and add to the db.
    We get average RMP, Rin, taum if available
    """

    # add columns to the db first:
    db["RMP"] = [np.nan for _ in range(len(db))]
    db["Rin"] = [np.nan for _ in range(len(db))]
    db["taum"] = [np.nan for _ in range(len(db))]
    db["tauh"] = [np.nan for _ in range(len(db))]

    for i, c in enumerate(db.index):
        dx = db.iloc[c]
        if "IV" not in list(dx.keys()):
            continue
        rmps = []
        rins = []
        taus = []
        tauhs = []
        if pd.isnull(dx["IV"]):
            continue
        #     if isinstance(dx['IV'], float):
        #         print(dx['IV'])
        for proto in dx["IV"].keys():
            feat = dx["IV"][proto]
            # proto has: ['holding', 'WCComp', 'CCComp', 'BridgeAdjust', 'RMP', 'RMPs',
            #'Irmp', 'taum', 'taupars', 'taufunc', 'Rin', 'Rin_peak', 'tauh_tau', 'tauh_bovera', 'tauh_Gh', 'tauh_vss']
            if "RMP" in feat:
                rmps.append(feat["RMP"] - 12)
                rins.append(feat["Rin"])
                taus.append(feat["taum"])
                if feat["tauh_tau"] is None:
                    taus.append(np.nan)
                else:
                    tauhs.append(feat["tauh_tau"])

        db.loc[c, "RMP"] = np.nanmean(rmps)
        db.loc[c, "Rin"] = np.nanmean(rins)
        db.loc[c, "taum"] = np.nanmean(taus) * 1e3
        #     print(tauhs)
        db.loc[c, "tauh"] = np.nanmean(tauhs) * 1e3
    return db


"""
Utility to plot data with joint plot, but different hues and some flexibility
see: https://stackoverflow.com/questions/35920885/how-to-overlay-a-seaborn-jointplot-with-a-marginal-distribution-histogram-fr

usage:
multivariateGrid('x', 'y', 'kind', df=df)
"""


def multivariateGrid(
    col_x, col_y, col_k, df, k_is_color=False, yscale="linear", scatter_alpha=0.5
):
    def colored_scatter(x, y, c=None):
        def scatter(*args, **kwargs):
            args = (x, y)
            if c is not None:
                kwargs["c"] = c
            kwargs["alpha"] = scatter_alpha
            mpl.scatter(*args, **kwargs)

        return scatter

    g = sns.JointGrid(x=col_x, y=col_y, data=df)

    color = None
    legends = []
    for name, df_group in df.groupby(col_k):
        legends.append(name)
        if k_is_color:
            color = name
        g.plot_joint(
            colored_scatter(df_group[col_x], df_group[col_y], color),
        )
        sns.distplot(
            df_group[col_x].values[~pd.isnull(df_group[col_x].values)],
            # df_group[col_x].values,
            ax=g.ax_marg_x,
            color=color,
        )
        sns.distplot(
            df_group[col_y].values[~pd.isnull(df_group[col_y].values)],
            #             df_group[col_y].values,
            ax=g.ax_marg_y,
            color=color,
            vertical=True,
        )
    # Do also global Hist:
    sns.distplot(
        df_group[col_x].values[~pd.isnull(df_group[col_x].values)],
        ax=g.ax_marg_x,
        color="grey",
    )
    sns.distplot(
        df_group[col_y].values[~pd.isnull(df_group[col_y].values)],
        #         df[col_y].values.ravel(),
        ax=g.ax_marg_y,
        color="grey",
        vertical=True,
    )
    mpl.legend(legends)


"""
Plot two parameters in distribution, using list of celltypes in the order.
Color/separate by "hue"
optional grid
"""


def compare_plot(
    xdat, ydat, db, huefactor=None, figno=1, multivariate=False, title=None
):
    midbs = db.loc[db["celltype"].isin(order)]

    f, ax = mpl.subplots(1, 1)
    PH.nice_plot(ax)

    #     ax = matplotlib.axes.Axes(f, rect=[0.1, 0.1, 0.8, 0.8])
    # sns.boxplot(
    #     x=xdat,
    #     y=ydat,
    #     hue=huefactor,
    #     data=midbs,
    #     ax=ax,
    #     order=order,
    #     hue_order=None,
    #     orient=None,
    #     color=None,
    #     palette=None,
    #     saturation=0.5,
    #     width=0.8,
    #     dodge=True,
    #     fliersize=4,
    #     linewidth=0.5,
    #     whis=1.5,
    #     notch=False,
    # )
    # for patch in ax.artists:
    #     r, g, b, a = patch.get_facecolor()
    #     patch.set_facecolor((r, g, b, 0.3))

    sns.swarmplot(
        x=xdat,
        y=ydat,
        data=midbs,
        order=order,
        hue=huefactor,
        hue_order=HueOrder,

        dodge=True,
        size=3,
        alpha=0.75,
        color=None,
        linewidth=0,
        ax=ax,
    )
    if title is not None:
        f.suptitle(title)
    if multivariate:
        mpl.figure(figno + 10)
        multivariateGrid(
            xdat, ydat, huefactor, midbs, k_is_color=False, scatter_alpha=0.5
        )
    handles, labels = ax.get_legend_handles_labels()

    # When creating the legend, only use the first two elements
    # to effectively remove the last two.
    legs = list(set(midbs[huefactor]))
    nleg = len(legs)  # print(nleg)
    l = mpl.legend(
        handles[0:nleg], labels[0:nleg]
    )  # , bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)


# print(sorted(list(set(midb2['agegroup']))))
# print(list(set(db['celltype'])))
def print_2(midb2):
    for i, c in enumerate(midb2.index):
        dx = midb2.iloc[c]
        #     print(dx['RMP'])
        if dx["RMP"] < -85:
            print(dx["cellID"], dx["RMP"])


def plot_decay(midb2):
    """
    Plot decay tau versus cell type, sort by temperature
    """
    compare_plot(
        "celltype",
        "tau2",
        midb2,
        huefactor="group",
        figno=1,
        multivariate=False,
        title="Decay Tau by celltype, temperature",
    )

    for c in db.index:
        if "_VC_increase_" in db.iloc[c]["data_complete"]:
            print(db.iloc[c]["cellID"], db.iloc[c]["data_complete"])


def plot_decay_vs_celltype(midb2):
    """
    Plot decay tau versus cell type, sort by max score
    """
    compare_plot(
        "celltype",
        "tau2",
        midb2,
        huefactor="maxscore_thr",
        figno=2,
        multivariate=False,
        title="Decay Tau by celltype, maxscore threshold=1.3",
    )


def plot_decay_vs_celltype(midb2):
    """
    Plot decay tau versus cell type, sort by max score
    """
    compare_plot(
        "celltype",
        "tau2",
        midb2,
        huefactor="maxscore_thr",
        figno=2,
        multivariate=False,
        title="Decay Tau by celltype, maxscore threshold=1.3",
    )
    """
    Plot decay tau versus cell type, include only cells with maxscore > threshold
    """
    midbx = midb2.loc[midb2["maxscore"] > 1.3]  # reduce data set by maxscore
    midbx = midbx.loc[midbx["celltype"].isin(order)]
    compare_plot(
        "celltype",
        "tau2",
        midbx,
        huefactor="temperature",
        figno=3,
        multivariate=False,
        title="maxscore > 1.3",
    )


def plot_max_score_vs_celltype(midb2):
    """
    Plot max score versus cell type, sort by age
    """
    print(set(list(midb2["agegroup"])))
    compare_plot(
        "celltype",
        "maxscore",
        midb2,
        huefactor="Group",
        figno=4,
        multivariate=False,
        title="maxscore celltype, age",
    )
    mpl.show()


def plot_RMPRIN(midb2):
    """
    Plot RMP and other measures versus cell type, sort by age
    """
    compare_plot(
        "celltype",
        "RMP",
        midb2,
        huefactor="agegroup",
        figno=45,
        multivariate=False,
        title="RMP by celltype, age",
    )
    compare_plot(
        "celltype",
        "Rin",
        midb2,
        huefactor="agegroup",
        figno=46,
        multivariate=False,
        title="Rin by celltype, age",
    )
    compare_plot(
        "celltype",
        "taum",
        midb2,
        huefactor="agegroup",
        figno=47,
        multivariate=False,
        title="Taum by celltype, age",
    )
    compare_plot(
        "celltype",
        "tauh",
        midb2,
        huefactor="agegroup",
        figno=48,
        multivariate=False,
        title="Tauh by celltype, age",
    )


def plot_FSL_vs_celltype(midb2):
    """
    Plot mean first event latency versus cell type, sort by maxscore true/false
    """
    midby = midb2.loc[midb2["maxscore"] > 1.3]
    compare_plot(
        "celltype",
        "latency",
        midby,
        huefactor="temperature",
        figno=5,
        multivariate=False,
        title="Latency by celltype, temperature for Responding cells",
    )
    midby = midb2.loc[midb2["temperature"].isin(["34C"])]
    compare_plot(
        "celltype",
        "latency",
        midby,
        huefactor="maxscore_thr",
        figno=55,
        multivariate=False,
        title="Latency by celltype, maxscore>1.3, at 34C",
    )


def plot_mean_amp_vs_celltype(midb2):
    """
    Plot mean amplitude versus cell type, sort by sort by maxscore true/false
    """
    compare_plot(
        "celltype",
        "max_amp",
        midb2,
        huefactor="Group",
        figno=6,
        multivariate=False,
        title="Maximum Amplitude",
    )


# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


def compareplots(midb):
  
    # midb3 = midb2[midb2["avg_spont_amps"] > 0]
    # for i, c in enumerate(midb3.index):
    #     print(midb3.loc[c]["cellID"], midb3.loc[c]["avg_spont_amps"])
    factor = "Group"
    compare_plot(
        "celltype",
        "avg_spont_amps",
        midb,
        huefactor=factor,
        figno=101,
        multivariate=False,
        title="Spontaneous Event Amplitudes by Temperature",
    )
    compare_plot(
        "celltype",
        "avg_event_qcontent",
        midb,
        huefactor=factor,
        figno=102,
        multivariate=False,
        title="Event Q content by Temperature",
    )
    compare_plot(
        "celltype",
        "avg_largest_event_qcontent",
        midb,
        huefactor=factor,
        figno=103,
        multivariate=False,
        title="largest event qcontent Amplitudes by Temperature",
    )
    mpl.show()

def score_stats(midby):

    # is maxscore different between celltypes?
    # midbs = midbs.reset_index()
    # for c in midbs.index:
    #     print(c)
    #     print(midbs.iloc[c]['cellID'],)
    #     print(' ', midbs.iloc[c]['maxscore'], midbs.iloc[c]['celltype'])
    # print(midbx.iloc[:]['celltype'])
    # print(midb2.loc[midb2['celltype'] == 'cartwheel'])
    c
    midby = midby.loc[midby["celltype"].isin(order)]
    model = ols("tau2 ~ C(celltype, Sum)", midby).fit()
    # print(midby.loc[midby['celltype'] == 'tuberculoventral'])

    table = sm.stats.anova_lm(model, typ=2)  # Type 2 ANOVA DataFrame

    print(table)
    print(model.nobs)

    print(model.summary())


# %%


# %%
def plot_maxscore(midb2):
    """
    Plot max score versus cell type, sort by temperature
    """
    order = [
        "bushy",
        "t-stellate",
        "d-stellate",
        "pyramidal",
        "tuberculoventral",
        "cartwheel",
    ]
    midbs = midb2.loc[midb2["celltype"].isin(order)]
    compare_plot(
        "celltype",
        "maxscore",
        midbs,
        huefactor="temperature",
        figno=61,
        multivariate=False,
    )


# %%
def print_onecell(evdb, cellname):
    print(evdb.head())
    onecell = evdb.loc[cellname]
    print("celltype: ", onecell["celltype"])
    print("SR: ", onecell["SR"])
    print("prots: ", onecell["protocols"])
    print("shufflescore: ", onecell["shufflescore"])
    print("scores: ", onecell["scores"])
    print("event_qcontent: ", onecell["event_qcontent"])
    print("event_amp: ", len(onecell["event_amp"][0][0]))
    print("event_qcontent: ", len(onecell["event_qcontent"]))
    print("positions: ", len(onecell["event_amp"][0][0]))
    print("largest_event_qcontent", onecell["largest_event_qcontent"])
    print("cell type: ", onecell["celltype"])
    print("firstevent_latency: ", onecell["firstevent_latency"][2])
    print("eventp: ", onecell["eventp"][2])
    print("area_fraction_Z", onecell["area_fraction_Z"])
    # print(onecell['event_amp'])


# db[db['date'].str.contains('2017.07.12_000')]


def get_cell(cellname):
    # get cell entry in main db
    cn = Path(cellname).parts
    day = cn[0]
    sliceno = cn[1]
    cellno = cn[2]
    #     print(day, sliceno, cellno)
    m = db.loc[
        (db["date"] == day)
        & (db["cell_cell"] == cellno)
        & (db["slice_slice"] == sliceno)
    ]  # &
    # (db['cell_type'] == celltype)]
    # print('m: ', cellname, m['cell_type'].values)
    return m["cell_type"].values


# %%
def one_cell(d):
    temp_path = Path(basepath.parent, "temp_pdfs")
    temp_path.mkdir(exist_ok=True)  # be sure of dir...
    tau_path = Path(basepath.parent, "tau_fits")
    tau_path.mkdir(exist_ok=True)  # be sure of dir...
    props = dict(boxstyle="round", facecolor="wheat", alpha=0.5)
    props2 = dict(boxstyle="round", facecolor="red", alpha=0.5)

    df = pd.DataFrame(
        columns=[
            "cell_id",
            "map",
            "tau1",
            "tau2",
            "Amplitude",
            "bfdelay",
            "Nevents",
            "FitError",
        ],
        index=[0],
    )
    maps = list(d.keys())
    if len(maps) == 0:
        return df

    hgt, wid = PH.getLayoutDimensions(len(maps), pref="width")
    P = PH.regular_grid(
        hgt,
        wid,
        order="columnsfirst",
        figsize=(8.0, 10),
        showgrid=False,
        verticalspacing=0.08,
        horizontalspacing=0.08,
        margins={
            "leftmargin": 0.07,
            "rightmargin": 0.05,
            "topmargin": 0.08,
            "bottommargin": 0.1,
        },
        labelposition=(-0.05, 0.95),
        parent_figure=None,
        panel_labels=None,
    )
    ax = P.axarr.ravel()
    cn = get_cell(str(Path(maps[0]).parent))

    if isinstance(cn, list) or isinstance(cn, np.ndarray):
        if len(cn) > 0:
            cn = cn[0]

    if cn is None or cn == "" or cn == [] or len(cn) == 0:
        cn = "Unknown"

    cn = str(cn)

    P.figure_handle.suptitle(f"{str(Path(maps[0]).parent):s}  Type: {cn:s}")
    scf = 1e12
    all_averages = []
    all_fits = []
    all_tau1 = []
    all_tau2 = []
    all_amp = []
    avg_fiterr = []
    neventsum = 0
    results = []  # hold results to build dataframe at the end
    for mapno in range(len(maps)):
        p = Path(maps[mapno])
        ax[mapno].set_title(str(p.name), fontsize=10)
        if (
            p is None
            or p.match("*_IC_*")
            or p.match("*_CA_*")
            or p.match("*_VC_increase_*")
            or p.match("*_VC_3mMCa_*")
            or p.match("*_VC_Range_*")
            # or p.match('*_VC_WCChR2_*')  # same as "weird" and "single"
            or p.match("*objUPplus60*")
            # or p.match('*range test*')
            or p.match("*5mspulses_5Hz*")
        ):
            ax[mapno].text(
                0.5,
                0.5,
                "Excluded on Protocol",
                transform=ax[mapno].transAxes,
                fontsize=12,
                verticalalignment="bottom",
                horizontalalignment="center",
                bbox=props2,
            )
            continue

        """
        dict_keys(['criteria', 'onsets', 'peaktimes', 'smpks', 'smpksindex', 
        'avgevent', 'avgtb', 'avgnpts', 'avgevoked', 'avgspont', 'aveventtb', 
        'fit_tau1', 'fit_tau2', 'fit_amp', 'spont_dur', 'ntraces', 'evoked_ev', 'spont_ev', 'measures', 'nevents'])
        """

        nev = 0
        avedat = None

        if d[maps[mapno]] is None or d[maps[mapno]]["events"] is None:
            continue
        if "FittingResults" not in d[maps[mapno]]:
            print("no fit results for map: ", maps[mapno])
            continue
        fit_res = d[maps[mapno]]["FittingResults"]
        # print(minisum)
        # print("evoked: ", fit_res['Evoked'])
        # print("spont: ", fit_res['Spont'])
        if fit_res['Evoked']['Amplitude'] is None:
            continue
        results.append(
            {
                "cell_id": str(Path(maps[mapno]).parent) + f"~map_{mapno:03d}",
                "celltype": cn,
                "map": str(Path(maps[mapno]).name),
                "tau1": fit_res['Evoked']['tau1'],
                "tau2": fit_res['Evoked']['tau2'],
                "Amplitude": fit_res['Evoked']['Amplitude']* 1e12,  # actual mean amplitude
                "bfdelay": fit_res['Evoked']['bfdelay'],
                # "Nevents": fit_res['Evoked']['Nevents'],
                # "FitError": fit_res['Evoked']['avg_fiterr'],
            }
        )

        # exit()
        # for i in range(len(minisum)):
        #     ev = minisum[i]
        #     if ev is None:
        #         continue
        #     print(ev)
        #     exit()
        #     if (
        #         not ev.average.averaged
        #         or ev.average.avgeventtb is None
        #         or ev.average.avgevent is None
        #     ):
        #         continue

        #     ax[mapno].plot(
        #         ev.average.avgeventtb,
        #         ev.average.avgevent,
        #         color=[0.25, 0.25, 0.25],
        #         linewidth=0.5,
        #     )
        #     if ev.average.best_fit is not None:
        #         ax[mapno].plot(
        #             ev.average.avgeventtb, ev.average.best_fit, color="c", linewidth=0.5
        #         )
        #     all_averages.append(ev.average.avgevent)
        #     all_fits.append(ev.average.best_fit)
        #     all_tau1.append(ev.average.fitted_tau1)
        #     all_tau2.append(ev.average.fitted_tau2)
        #     all_amp.append(ev.average.amplitude)
        #     nev += 1
    #     if nev == 0:
    #         ax[mapno].text(
    #             0.5,
    #             0.5,
    #             "No events detected",
    #             transform=ax[mapno].transAxes,
    #             fontsize=12,
    #             verticalalignment="bottom",
    #             horizontalalignment="center",
    #             bbox=props2,
    #         )
    #         continue
    #     # if nev < 10:
    #     #     ax[mapno].text(0.5, 0.5, f"Too Few Events", transform=ax[mapno].transAxes,
    #     #         fontsize=10,
    #     #         verticalalignment='bottom', horizontalalignment='center', bbox=props2)
    #     #     continue
    #     # if ev.average.avgeventb is None or ev.average.best_fit is None:
    #     #     print("No average event data", maps[mapno])
    #     #     continue
    #     neventsum += nev
        # results.append(
        #     {
        #         "cell_id": str(Path(maps[mapno]).parent) + f"~map_{mapno:03d}",
        #         "celltype": cn,
        #         "map": str(Path(maps[mapno]).name),
        #         "tau1": ev.average.fitted_tau1,
        #         "tau2": ev.average.fitted_tau2,
        #         "Amplitude": ev.average.amplitude * 1e12,
        #         "bfdelay": 1.0,
        #         "Nevents": ev.average.Nevents,
        #         "FitError": ev.average.avg_fiterr,
        #     }
        # )
    #     # avedat = np.array(avedat)/nev
        ax[mapno].plot(fit_res['Evoked']['tb'], fit_res['Evoked']['avedat'], 'r-', linewidth=2)
        ax[mapno].plot(fit_res['Evoked']['tb'], fit_res['Evoked']['bestfit'], 'c--')
        textstr = f"Tau1: {fit_res['Evoked']['tau1']*1e3:.3f} ms\n"
        textstr += f"Tau2: {fit_res['Evoked']['tau2']*1e3:.3f} ms \n"
        textstr += f"Amp: {fit_res['Evoked']['Amplitude']*1e12:.3e}\nNMaps: {nev:4d}"
        # place a text box in upper left in axes coords
        ax[mapno].text(
            0.95,
            0.05,
            textstr,
            transform=ax[mapno].transAxes,
            fontsize=10,
            verticalalignment="bottom",
            horizontalalignment="right",
            bbox=props,
        )
    #     # print(textstr)

    df = pd.DataFrame(results)
    # mpl.show()
    ofile = Path(tau_path, str(Path(maps[mapno]).parent).replace("/", "~") + ".pdf")
    #     print('output file ', ofile )
    mpl.savefig(ofile)
    mpl.close("all")

    #     fns = list(Path(temp_path).glob('*.pdf'))
    #     # merge the maps into one file and store elsewhere
    #     pdf_merger = PdfFileMerger()
    #     for i, fn in enumerate(fns):
    #         print('Merging fn: ', fn) # source file
    #         with open(fn, 'rb') as fh:
    #             pdf_merger.append(PdfFileReader(fh))  # now append the rest
    #     print('mapse: ', str(maps[0].parent))
    #     with open(Path(tau_path, str(maps[0].parent).replace('/', '~')+'.pdf'), 'wb') as fo:
    #         pdf_merger.write(fo)
    #     for fn in fns:  # delete the files in the tempdir
    #         os.remove(fn)
    return df


def make_taus(basepath):
    print("basepath: ", basepath)  # goes to events diretory

    files = list(basepath.glob("*.pkl"))

    files = sorted(basepath.glob("*.pkl"))
    # print(files)

    df_taus = pd.DataFrame(
        columns=[
            "cell_id",
            "map",
            "tau1",
            "tau2",
            "Amplitude",
            "bfdelay",
            "Nevents",
            "FitError",
        ]
    )
    for n, f in enumerate(list(files)):
        if (
            str(f).find("_signflip") > 0 or str(f).find("_alt") > 0
        ):  # non-cannonical measures
            print("excluding on sign/alt: ", f)
            continue
        print(f"reading: {str(f.name):s}   n={n:d}")

        with open(f, "rb") as fh:
            try:
                d = pd.read_pickle(fh, compression=None)
            except:
                print("Error reading file: ", f)
                continue
        df_tau = one_cell(d)
        df_taus = pd.concat([df_taus, df_tau], sort=False)
    df_taus.reindex()
    taufile = Path(basepath.parent, f"NF107Ai32_{expt:s}_taus.pkl")
    taufile_xlsx = Path(basepath.parent, f"NF107Ai32_{expt:s}_taus.xlsx")
    with open(taufile, "wb") as fo:
        df_taus.to_pickle(fo)
    with open(taufile_xlsx, "wb") as fo:
        df_taus.to_excel(fo)
    print(df_taus.head())

def show_tau(basepath, cellname):
    f = Path(basepath, cellname.replace("/", "~")+ ".pkl")
    d = pd.read_pickle(open(f, "rb"), compression=None)
    k = list(d.keys())
    f, ax = mpl.subplots(2,1)
    print("# protocols")
    for k in list(d.keys()): # protocols
        events = d[k]['events']
        FR = d[k]['FittingResults']
        print(k)
        print(FR)
        for i, evs in enumerate(['Evoked', 'Spont']):
            if FR[evs]['avedat'] is None:
                continue
            print("tau1: ", FR[evs]['tau1'])
            print("tau2: ", FR[evs]['tau2'])

            print("A: ", FR[evs]['Amplitude'])
            ax[i].plot(FR[evs]['tb'], FR[evs]['avedat'])
            ax[i].plot(FR[evs]['tb'], FR[evs]['bestfit'])
    mpl.show()

def plot_one(dp, axi):
    Qlist = []
    Alist = []
    HWlist = []
    HWDlist = []
    HWUlist = []

    # print(dp.keys())
    # print(dp['Qr'])
    # exit()
    print(dp["events"][0])
    print(len(dp["events"]))
    for x in dp["events"]:
        m = dp["events"][x]
        if len(m.events.event) > 0:
            #         print(len(m['Q']), m['Q'])
            Qlist.extend(m.events.Qtotal)
            Alist.extend(m.events.smoothed_peaks)
            # HWlist.extend(m['HW'])
            # HWDlist.extend(m['HWdown'])
            # HWUlist.extend(m['HWup'])

    Qlist = np.array(Qlist)
    Alist = np.array(Alist)
    # HWlist = np.array(HWlist)
    # HWDlist = np.array(HWDlist)
    # HWUlist = np.array(HWUlist)

    axi.plot(Alist * 1e12, Qlist * 1e9, "ro", markersize=3)
    axi.set_xlim(0, 200)
    axi.set_ylim(0, 1.0)

def remove_outliers(df):
    """
    Based on the coding sheet, remove the "outlisers" from the data set. 
    The definition of an outlier is by animal. 
    Animals for which the lab notebook did not cleary indicate the noise exposure,
    where the post-exposure time was different, there was no data, or where the age
    at recording was too old > P70
    """
    df.drop(df.loc[df["outlier"] == "yes"].index, inplace=True)

    return df


def Games_Howell(df, combs):
    group_comps = []
    mean_differences = []
    degrees_freedom = []
    t_values = []
    p_values = []
    std_err = []
    up_conf = []
    low_conf = []

    for comb in combs:
        # Mean differences of each group combination
        diff = group_means[comb[1]] - group_means[comb[0]]
        
        # t-value of each group combination
        t_val = np.abs(diff) / np.sqrt((group_variance[comb[0]] / group_obs[comb[0]]) + 
                                    (group_variance[comb[1]] / group_obs[comb[1]]))
        
        # Numerator of the Welch-Satterthwaite equation
        df_num = (group_variance[comb[0]] / group_obs[comb[0]] + group_variance[comb[1]] / group_obs[comb[1]]) ** 2
        
        # Denominator of the Welch-Satterthwaite equation
        df_denom = ((group_variance[comb[0]] / group_obs[comb[0]]) ** 2 / (group_obs[comb[0]] - 1) +
                    (group_variance[comb[1]] / group_obs[comb[1]]) ** 2 / (group_obs[comb[1]] - 1))
        
        # Degrees of freedom
        df = df_num / df_denom
        
        # p-value of the group comparison
        p_val = psturng(t_val * np.sqrt(2), k, df)

        # Standard error of each group combination
        se = np.sqrt(0.5 * (group_variance[comb[0]] / group_obs[comb[0]] + 
                            group_variance[comb[1]] / group_obs[comb[1]]))
        
        # Upper and lower confidence intervals
        upper_conf = diff + qsturng(1 - alpha, k, df)
        lower_conf = diff - qsturng(1 - alpha, k, df)
        
        # Append the computed values to their respective lists.
        mean_differences.append(diff)
        degrees_freedom.append(df)
        t_values.append(t_val)
        p_values.append(p_val)
        std_err.append(se)
        up_conf.append(upper_conf)
        low_conf.append(lower_conf)
        group_comps.append(str(comb[0]) + ' : ' + str(comb[1]))

    result_df = pd.DataFrame({'groups': group_comps,
                            'mean_difference': mean_differences,
                            'std_error': std_err,
                            't_value': t_values,
                            'p_value': p_values,
                            'upper_limit': up_conf,
                            'lower limit': low_conf})

    result_df

def plot_taus(df, pl,   celltypes = ["pyramidal", "cartwheel", "tuberculoventral"]):
    df = remove_outliers(df)
    groups = Groups # defined when selectng data
    df = df[df['Group'].isin(groups)]
    # print summary stats on data
    desc_stats =["mean", "std", "count"]
    for g in groups:
        for ct in celltypes:
            print(f"\nCelltype: {ct:s}  Group: {g:s}")
            print(df[(df["celltype"] == ct) & (df["Group"] == g)].agg(
                {"tau1": desc_stats,
                "tau2": desc_stats,
                "Amplitude": desc_stats,}
                )
            )
    f, ax = mpl.subplots(2, 2, figsize=(8, 8))
    ax = np.array(ax).ravel()
    pax = {}
    for ia, i in enumerate(pl):
        dp = df[i]
        # print(df[pl[ia]])
        dfp = df.dropna(subset=["celltype", pl[ia]], axis=0)
        sns.set_theme(style="ticks", palette="pastel")
        try:
            print("celltype: ", i)
            sns.boxplot(
                x="celltype", y=pl[ia], data=dfp,  ax=ax[ia],
                hue="Group",
                hue_order=HueOrder, 
                palette = ["g", "b", "orange", "magenta"],
                # facecolor='w',
                # edgecolor='k',
                order = celltypes,
                saturation=1.0,
            )
        except:
            print(dfp['celltype'], dfp[pl[ia]])

        sns.swarmplot(
            x="celltype", y=pl[ia], hue="Group", data=dfp, ax=ax[ia],
            dodge=True, hue_order=HueOrder, order = celltypes,
            palette = ["k", "k", "k", "k"],
            size=3, edgecolor="k", linewidths=1,
        )
        PH.nice_plot(ax[ia], position=-0.03, direction="outward", ticklength=4.0)
        pax[i] = ax[ia]
    # sns.despine(offset=10, trim=True)
    print(pax)
    if 'tau1' in pax.keys():
        pax['tau1'].set_ylabel(r"$\tau_{rise}$ (ms)")
        pax['tau1'].set_ylim(0, 1.5)
    if 'tau2' in pax.keys():
        pax['tau2'].set_ylabel(r"$\tau_{fall}$ (ms)")
        pax['tau2'].set_ylim(0, 25)
    if "Amplitude" in pax.keys():
        pax["Amplitude"].set_ylabel(r"EPSC (pA)")
        pax["Amplitude"].set_ylim(0, 200)
    if "SR" in pax.keys():
        pax["SR"].set_ylabel(r"SR (Hz)")
        pax["SR"].set_ylim(0, 30)



    # Fit the ANOVA model
    # print(df['SR'])

    # for measure in pl:
    #     print("Measure: ", measure)
    #     dfok = df
    #     if measure == "SR":
    #         for index in df.index:
    #             print(df.loc[index, measure])
    #         dfok.dropna(subset=[measure])


    #     for celltype in celltypes:

    #         model = ols(f'{measure} ~ Group', data=dfok[dfok['celltype']==celltype]).fit()

    #         # Perform ANOVA
    #         anova_table = sm.stats.anova_lm(model, typ=2)
    #         print("\nCelltype: ", celltype, "Measure: ", measure)
    #         print(anova_table)
    #         print()


    for measure in pl:
        for celltype in celltypes:
            data = [df.loc[ids, measure].values for ids in df[df['celltype'] == celltype].groupby('Group').groups.values()]
            dframe = df[df['celltype'] == celltype]
            dframe[measure] = pd.to_numeric(dframe[measure], errors='coerce') 
            
            for i in range(len(data)):
                data[i] = [d for d in data[i] if not np.isnan(d)]
    #         g = [['A']*len(data[0]), ['B']*len(data[1]), ['AA']*len(data[2]), ['AAA']*len(data[3])]
    #         x = np.concatenate(data)
    #         pth = sp.posthoc_tukey_hsd(x, np.concatenate(g))
    #         print(pth)
            # H, p = stats.kruskal(*data)
            # print("\nCelltype: ", celltype, "Measure: ", measure)
            # if p < 0.05:
            #     color = "m"
            # else:
            #     color = "w"
            # # CP.cprint(color, f"Kruskal-Wallis H-test test: H={H:.5f}  p: {p:.3f}")
            # welches anova

            aov = pg.welch_anova(dv=measure, between='Group', data=dframe)
            print(measure, celltype)
            print("-"*80, "\n", "Welch Anova: \n",aov)
            # posthoc = sp.posthoc_mannwhitney(df[df['celltype'] == celltype], val_col=measure, group_col='Group', p_adjust = 'holm-sidak')
            # print("KW posthoc: ", posthoc)
            # unequal variances, use Games-Howell test:
            combinations = [["B", "A"], ["B", "AA"], ["B", "AAA"]]
            print(pg.pairwise_gameshowell(dv=measure, between='Group', data=dframe))
            print("="*80)


    mpl.show()

def show_taus(df):
    pl = ["Amplitude"]
    for c in df.index:
        dx = mdb.iloc[c]
        print(dx.cellID)
        if c > 3:
            exit()
        print("cellID: ", dx["cellID"], "tau1: ", dx["tau1"], "tau2: ", dx["tau2"])

def get_averages(df):
    """
    Replace the measurements in the FIRST map of each cell with the average of all
    the maps for tau1, tau2 and amplitude; replace the remainder with NaN.
    This way, we only have one measure for each cell, but from all maps.
    """
    scales = {'tau1': 1e3, 'tau2': 1e3, 'Amplitude': 1.0, 'SR': 1.0, 'spont_amps': 1.0e12}
    cells = df['cell_id'].unique()
    for c in cells:
        d = df.loc[df['cell_id'] == c]
        for measure in list(scales.keys()):
            ma = []
            for i, m in enumerate(d['map']):
                if pd.isnull(m):
                    continue
                # print("m: ", m)
                # values can come in 2 ways:
                # 1. as a scalar, one value per map
                # 2. as a list, with each element being a list of values for the map
                # It can also be empty
                value = d.loc[d['map']==m, measure].values[0]
                print("cell, measure, type: ", c, m, measure, type(value))
                if isinstance(value, list):
                    print("      is a list")
                    # print(value*1e12)
                    if  len(value) == 0:
                        value = np.nan
                        op = 'nan'
                    else:
                        print(value)
                        xv = []
                        print("value: ", value)
                        for v in value:
                            if isinstance(v, float):
                                v = [v]
                            xv.extend(v)
                        value = xv
                        print("now: ", value)
                        op = 'reduced'
                        try:
                            value = np.mean(value)
                            op = 'mean'
                        except:
                            try:
                                value = np.mean(value[0])
                            except:
                                try:
                                    value = float(value)
                                    op = 'float'
                                except:
                                    value = np.nan
                                    op = "nan"
                elif isinstance(value, np.ndarray):
                    value = np.mean(value)
                    op = np.ndarray
                elif isinstance(value, float):
                    op = "float"
                    pass
                else:
                    raise ValueError("Value is not a list or scalar", repr(value), type(value))
                print("    And now it is:     ", np.array(value)*scales[measure], "op = ", op)
                if np.isscalar(value):
                    ma.append(value*scales[measure])
                    # if measure == 'tau1':
                    #     ma[-1] = np.power(ma[-1], 0.25)
            if len(ma) == 0:
                continue
            for i, index in enumerate(d.index):
                if i == 0:
                    df.at[index, measure] = np.nanmean(ma)
                else:
                    df.at[index, measure] = np.nan
    for c in cells:
        
        srdata = df.loc[df['cell_id'] == c, 'SR']
        # print("type of sr: ", type(df.loc[df['cell_id'] == c, 'SR']))
        # print("sr values: ", type(df.loc[df['cell_id'] == c, 'SR'].values))
        # print("len sr: ", len(df.loc[df['cell_id'] == c, 'SR'].values))
        # print(srdata.values[0], type(df.loc[df['cell_id'] == c, 'SR'].values[0]))
        if len(srdata) > 0 and isinstance(srdata.values[0], list):
            # print("SR data: ", df.loc[df['cell_id'] == c, 'SR'].values[0])
            sr_data = []
            for u in srdata.values[0]:
                if isinstance(u, list):
                    sr_data.extend(u)
                else:
                    sr_data.append(u)
            print("new sr data: ", sr_data)
            v = np.array(sr_data)
            # print(v.ndim)
            if v.ndim > 1:
                v = np.mean(v) # axisl = (0,1)
            else:
                pass
            df.loc[df['cell_id'] == c, 'SR'] = np.mean(v)
        # print(">>>> ", df.loc[df['cell_id'] == c, 'SR'].values[0], type(df.loc[df['cell_id'] == c, 'SR'].values[0]))
        elif len(srdata) == 0:
            df.loc[df['cell_id'] == c, 'SR'] = np.nan
    # exit()
    return df


def plot_averages(df):
    # df.filter(like='avgs')
    davg = df[df["cell"].str.contains("avgs", regex=False)]
    for celltype in ["pyramidal", "tuberculoventral"]:
        for c in davg["cell"]:
            cn = c.split("~")
            day = cn[0]
            sliceno = cn[1]
            cellno = cn[2]
            print(day, sliceno, cellno)
            m = db.loc[
                (db["date"] == day)
                & (db["cell_cell"] == cellno)
                & (db["slice_slice"] == sliceno)
                & (db["cell_type"] == celltype)
            ]
            print(
                "m: ",
                m["cell_type"].values,
                davg.loc[davg["cell"] == c]["tau1"].values[0],
            )


# dbase = Path(expts['databasepath'], expts['directory'])
# df.to_pickle(Path(dbase, 'NF107Ai32_Het_taus.pkl'))
# df.to_excel( Path(dbase, 'NF107Ai32_Het_taus.xlsx'))

# This code collects some data regarding the analysis into a file for pickling


"""
Compute the depression ratio of evoked events for maps in cells that have 
shuffle score > 1.3 and points in map with Z score > 2.1 for the first stimulus
"""


def depression_ratio(p=None, stim_N=4):
    verbose = False
    #     proto = pathlib.PurePosixPath(p)  # make sure of type
    #     protocol, self.protodata = self._get_cell_protocol_data(p)
    for i, c in enumerate(midb2.index):
        dx = midb2.iloc[c]
        for j, p in enumerate(dx["protocols"]):  # select protocol
            if p is None:
                continue
            prot = p.name
            if prot.startswith("Map_NewBlueLaser_VC_10Hz_0") and prot.find("plus") < 0:
                print(dx["cellID"], prot)
                print(
                    dx["cellID"],
                    "\n   scores: ",
                    dx["scores"],
                    "\n   shuffle: ",
                    dx["shufflescore"],
                    "\n   Area Fract>Z: ",
                    dx["area_fraction_Z"],
                    "\n  #Event Amp: ",
                    dx["event_amp"][0],
                    "\n   #Alleventlat: ",
                    dx["allevent_latency"][0],
                )


# def xyz():
#     envx = self.protodata[p]
#     if len(envx['stimtimes']['start']) < stim_N+1:
#         # print('not enough stim times', self.events[proto]['stimtimes']['start'])
#         # print('in ', str(p))
#         return np.nan
#     # print('Enough stim times', self.events[proto]['stimtimes']['start'])
#     # print('in ', str(p))
#     stimstarts = envx['stimtimes']['start']
#     stim1 = [stimstarts[0]+0.0001, stimstarts[0]+0.011]
#     stim5 = [stimstarts[stim_N]+0.0001, stimstarts[stim_N]+0.011]
#     # print('eventtimes: ', self.events[u]['eventtimes'])  # trials
#     # print('# positions: ', len(self.events[u]['positions']))  # 60 positions
#     protoevents = envx['events']
#     amp1 = []
#     amp5 = []
#     tev1 = []
#     tev5 = []
#     if verbose:
#         print('p: ', str(p))
#         print(' # protoevents: ', len(protoevents))
#     for trial in range(len(protoevents)):
#         if verbose:
#             print('zs trial: ', len(envx['ZScore'][trial]))
#             print(envx['ZScore'][trial])
#             print(envx['events'][trial][0].keys()) #['event_amp'][trial])
#             print(len(envx['events'][trial]))
#             for i in range(len(envx['events'][trial])):
#                 print('i peaktimes T: ', i, np.array(envx['events'][trial][i]['smpksindex'])*5e-2)
#                 print('i peaktimes pA: ', i, np.array(envx['events'][trial][i]['smpks'])*1e12)
#         # print(trial, len(protoevents), len(envx['ZScore']))
#         spots = list(np.where(envx['ZScore'][trial] > 0.2)[0])
#         # print('spots: ', len(spots), spots)
#         if len(spots) == 0:  # no significant events
#             continue
#         for ispot in spots:  # now for all spots in this trial, get the events
#             if 'smpksindex' in protoevents[trial].keys():  # handle both orders (old and new...)
#                 event_t = np.array(protoevents[trial]['smpksindex'][ispot])*5e-5  # time
#                 event_a = np.array(protoevents[trial]['smpks'][ispot])  # amplitude
#             else:
#                 event_t = (np.array(protoevents[trial][ispot]['smpksindex'])*5e-5)[0]
#                 event_a = np.array(protoevents[trial][ispot]['smpks'])[0]
#             iev1 = np.argwhere((event_t > stim1[0]) & (event_t <= stim1[1]))  # select events in the window

#             # print(f"\n{str(event_fn):s}  {str(p):s}")
#             # print(event_t)
#             # print(event_a)
#             # print(iev1)
#             # for i in range(len(event_t)):
#             #     print(f"{i:d}  {event_t[i]:.3f}  {event_a[i]*1e12:.2f}")
#             if len(iev1) > 0:
#                 amp1.append(np.max(event_a[iev1[0]]))
#                 tev1.append(event_t[iev1[0]][0])
#             else:
#                 amp1.append(np.nan)
#                 tev1.append(np.nan)
#             iev5 = np.argwhere((event_t > stim5[0]) & (event_t <= stim5[1]))
#             if len(iev5) > 0:
#                 amp5.append(np.max(event_a[iev5[0]]))
#                 tev5.append(event_t[iev5[0]][0])
#             else:
#                 amp5.append(np.nan)
#                 tev5.append(np.nan)
#             # print('event indices: ', iev1, iev5)
#             # print(Zs, protoevents[trial]['smpks'][zk], np.array(protoevents[trial]['smpksindex'][zk])*5e-5)
#     if verbose:
#         # print(amp1, amp5)
#    #      print('at:', np.array(amp5))
#    #      print(f"amp1: {np.array(amp1)*1e12:.1f}  amp5: {np.array(amp5)*1e12:.1f}")
#         print(tev1)
#         print(tev5)
#         print(amp1)
#         print(amp5)
#     if len(amp5) == 0:
#         amp5 = [0]
#     if len(amp1) == 0:
#         ratio = np.nan
#     else:
#         # print(amp1, amp5)
#         ratio = np.nanmean(amp5)/np.nanmean(amp1)
#     if verbose:
#         for i in range(len(amp1)):
#             print(f"amp1: {amp1[i]*1e12:.1f}  amp5: {amp5[i]*1e12:.1f} tev1: {tev1[i]*1e3:.1f}  tev5: {tev5[i]*1e3:.1f}")
#     # print(f"Ratio: {ratio:.3f}")
#     return ratio


# depression_ratio()


if __name__ == "__main__":
    # do this once:
    # show_tau(basepath, '2018.07.17_000/slice_000/cell_001')
    # exit()
    # df = make_taus(basepath)
    # exit()
    mdb_file = Path(expts["databasepath"], expts["directory"], f"merged_db_{expt:s}.pkl")
    # if mdb_file.exists():
    #     mdb = pd.read_pickle(mdb_file)

    # else:
    mdb, evdb = merge_db(db=db, expts=expts)
    with open(mdb_file, 'wb') as fo:
        mdb.to_pickle(fo)
    with open(f"merged_db_{expt:s}.xlsx", "wb") as fo:  # keep local
        mdb.to_excel(fo)

    df = get_averages(mdb)
    # mdb = annotate_db(mdb)
    # # print_onecell(evdb, '2017.07.12_000~slice_001~cell_000')

    plot_taus(df, ['tau1', 'tau2', 'Amplitude', 'SR'], celltypes = CellTypes)
    
    #
    # compareplots(mdb)
    # plot_max_score_vs_celltype(mdb)
    # plot_mean_amp_vs_celltype(mdb)
    # mpl.show()
