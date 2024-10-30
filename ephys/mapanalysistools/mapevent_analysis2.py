# %%
# matplotlib.use("Qt5Agg")
import importlib
import logging
import os

import re

# importlib.reload(minis)
# MA = minis.minis_methods.MiniAnalyses()
import sys
from pathlib import Path

import matplotlib.pyplot as mpl
import numpy as np
import pandas as pd
import seaborn as sns

# import minis
from pylibrary.plotting import plothelpers as PH
from PyPDF2 import PdfFileMerger, PdfFileReader

from ephys.tools.get_computer import get_computer
from ephys.tools.get_configuration import get_configuration
import ephys.tools.map_cell_types as MCT
import ephys.tools.filename_tools as FT

# go up one directory level from this file's directory:
current_dir = Path.cwd()
print(current_dir)
parent_path = current_dir.parent
# prepend parent directory to the system path:
sys.path.insert(0, parent_path)
# setting path
# sys.path.append("../nf107")
config_file_path = Path("../mrk-nf107", "config", "experiments.cfg")
datasets, experiments = get_configuration(config_file_path)

Logger = logging.getLogger("MapAnalysis2_Log")

expt = "nf107"
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


class MapEventAnalyzer:
    def __init__(self, experiment: str, force_update: bool = False):
        self.expts = experiments[experiment]
        # print(self.expts)
        self.force_map_update = force_update
        database = Path(
            self.expts["databasepath"], self.expts["directory"], self.expts["datasummaryFilename"]
        )
        self.tau_db_name = Path(
            self.expts["analyzeddatapath"], self.expts["directory"], self.expts["map_tausFilename"]
        )
        if not database.is_file():
            print(f"Database file not found: {database}")
            return
        with open(database, "rb") as fh:
            db = pd.read_pickle(fh, compression=None)

        # take care of some cleanup before returning the database
        db = self.renamecells(db)  # consistent cell names
        db = self.remove_cells(db)  # remove bad datasets after the merge
        # now do the merge
        self.db = db
        self.tau_db = None
        if not self.tau_db_name.is_file() or self.force_map_update:
            self.tau_db = self.get_events()  # create the file and get the data
        self.merged_db = self.merge_db(tau_db=self.tau_db)

    def renamecells(self, db):
        for c in db.index:
            dx = db.iloc[c]
            cell = str(Path(dx["date"], dx["slice_slice"], dx["cell_cell"]))
            db.loc[c, "cell_id"] = cell
            # print(db.columns)
            db.loc[c, "cell_type"] = MCT.map_cell_type(db.loc[c, "cell_type"])
            # if db.loc[c, "cell_type"] is None:
            #     print("cell type is None for: ", cell)
        return db

    def remove_cells(self, db):
        removelist = ["2017.06.23_000/slice_003/cell_000"]
        for rm in removelist:
            db = db[db.cell_id != rm]
        return db

    def merge_db(self, tau_db=None):
        """
        Combine event summary and taus with the main db to get ALL information on a per-cell basis

        Returns the merged pandas database
        """

        """
        Insert a new column, cellID, for matching with other db's
        and population it with a path version for searches
        """
        self.db["cellID"] = ["" for _ in range(len(self.db))]

        if tau_db is None:  # try to read an existing file instead

            if not self.tau_db_name.is_file():
                print(f"File not found: {self.tau_db_name}")
                return
            try:
                with open(self.tau_db_name, "rb") as fh:
                    tau_db = pd.read_pickle(fh, compression=None)
            except Exception as exc:
                print(f"Error reading file: {self.tau_db_name}")
                print(exc)
                exit()

        events = Path(
            self.expts["analyzeddatapath"],
            self.expts["directory"],
            self.expts["eventsummaryFilename"],
        )
        for f in [events]:
            if not f.is_file():
                print(f"File not found: {f}")
                return
        with open(events, "rb") as fh:
            evdict = pd.read_pickle(fh, compression=None)
        evm = {}

        for i, c in enumerate(evdict.keys()):  # reformat dict so cells are in dict
            evm[i] = {"cell": c}
            for d in evdict[c].keys():
                evm[i][d] = evdict[c][d]

        evdb = pd.DataFrame(evm).transpose()
        evdb.rename(columns={"cell": "cell_id"}, inplace=True)

        midb2 = pd.merge(left=evdb, right=tau_db, left_on="cell_id", right_on="cell_id")
        midb2 = pd.merge(left=midb2, right=self.db, left_on="cell_id", right_on="cell_id")
        midb2.rename(columns={"celltype_x": "cell_type", "cell_id_x": "cell_id",
                              "celltype_x": "celltype"}, inplace=True)
        midb2 = midb2.drop("celltype_y", axis=1)


        midb2["maxscore"] = ["" for _ in range(len(midb2))]
        midb2["maxscore_thr"] = ["" for _ in range(len(midb2))]
        midb2["agegroup"] = ["" for _ in range(len(midb2))]
        midb2["latency"] = ["" for _ in range(len(midb2))]
        midb2["mean_amp"] = ["" for _ in range(len(midb2))]
        midb2["max_amp"] = ["" for _ in range(len(midb2))]
        midb2["avg_event_qcontent"] = ["" for _ in range(len(midb2))]
        midb2["avg_spont_amps"] = ["" for _ in range(len(midb2))]
        midb2["avg_largest_event_qcontent"] = ["" for _ in range(len(midb2))]

        df_iv_name = Path(
            self.expts["analyzeddatapath"],
            self.expts["directory"],
            self.expts["assembled_filename"],
        )
        with open(df_iv_name, "rb") as fh:
            df_iv = pd.read_pickle(fh, compression=None)
        def _make_cellid(row):
            return FT.make_cellid_from_slicecell(row["cell_id"])
        df_iv['cell_id'] = df_iv.apply(_make_cellid, axis=1)

        midb2 = pd.merge(left=midb2, right=df_iv, left_on="cell_id", right_on="cell_id")
        args = ['celltype', 'date', 'sex', 'weight', 'important', 'protocols', 'age', 'cell_type', 'animal_identifier']
        for arg in args:
            midb2.drop(arg+"_y", axis=1, inplace=True)
            midb2.rename(columns={arg+"_x": arg}, inplace=True)


        # print("RMP", midb2["RMP"].values)
        # print("df_iv cellids: ", df_iv["cell_id"].values)
        # exit()

        # print("1:::evdb ", (evdb.head()))
    #    print("2:::midb ", (midb.head()))
        # print("3:::midb2 ", (midb2.columns))
        # print("midb2 head: ", midb2.head(), "\n", len(midb2))
        # print(midb2.cell_id.values)

        for c in midb2.index:
            dx = midb2.iloc[c]
            #         print(midb2.loc[c, 'cell_type'],)
            if midb2.loc[c, "cell_type"] not in [None, "None"]:
                midb2.loc[c, "cell_type"] = midb2.loc[c, "cell_type"]
            #         print('  ', midb2.loc[c, 'celltype'])
            # print("temperature" in dx.keys() )
            if dx["temperature"] in ["room temp", "", " ", "room temperature", "25", 25]:
                #         print(f"resetting temp: <{str(dx['temperature']):s}>")
                midb2.loc[c, "temperature"] = "25C"
            re_age = re.compile(r"[~]*[pP]*[~]*(?P<age>[0-9]{1,3})[dD]*[, ]*")
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
            midb2 = self.average_datasets(midb2, c, dx, "firstevent_latency", "latency", scf=1e3)
            midb2 = self.average_datasets(midb2, c, dx, "spont_amps", "avg_spont_amps", scf=-1e12)
            midb2 = self.average_datasets(midb2, c, dx, "event_qcontent", "avg_event_qcontent")
            midb2 = self.average_datasets(
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


        midb3 = self.compute_averages(midb2)
        # merged_inner.loc[merged_inner['celltype'] == 'bushy']

        return midb3

    """
    Get the database and annotate it
    """

    def annotate(self, db):
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
        midb2 = self.merge_db(self.expts, db)
        # print(set(list(midb2['agegroup'])))
        # print(midb2[midb2['RMP'] < 90.])

        midb2.rename(columns={"celltype_x": "cell_type", "cell_id_x": "cell_id"}, inplace=True)

        print("age group: ", set(list(midb2["agegroup"])))
        print("RMP 0: ", midb2[midb2["RMP"] < -90.0]["cellID"], midb2[midb2["RMP"] < -90.0]["RMP"])


        def _apply_show_RMP(row):
            # print(row["RMP"])
            if row["RMP"] < -85:
                print("cell RMP out of range: ", row["cell_id"], row["RMP"])
            else:
                print("cell RMP out in range: ", row["cell_id"], row["RMP"])

        # print(midb2.head())
        midb2.apply(_apply_show_RMP)

        for i, c in enumerate(midb2.index):
            dx = midb2.iloc[c]
            print("RMP: ", dx["RMP"])
            if dx["RMP"] < -90:
                print("cell RMP out of range: ", dx["cell_id"], dx["RMP"])
            else:
                print("cell RMP out in range: ", dx["cell_id"], dx["RMP"])

    def average_datasets(self, db, c, dx, measure, outmeasure, scf=1.0):
        sa = []
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
        elif len(sa) == 1 and sa[0] is None:
            san = np.nan
        else:
            try:
                san = np.nanmean(np.squeeze(np.array(sa))) * scf
            except:
                print("failed to average: ", np.squeeze(np.array(sa)))
        #     print('san: ', san)
        db.loc[c, outmeasure] = san
        return db

    def compute_averages(self, db):
        """
        Compute some averages from the ivs and add to the db.
        We get average RMP, Rin, taum if available
        """

        for i, c in enumerate(db.index):
            # dx = db.iloc[c]
            # if "IV" not in dx or dx["IV"] == {}:
            #     continue
            # rmps = []
            # rins = []
            # taus = []
            # tauhs = []

            #     if isinstance(dx['IV'], float):
            #         print(dx['IV'])
            # for proto in dx["IV"].keys():
            #     feat = dx["IV"][proto]
            #     print(proto.keys())
                # proto has: ['holding', 'WCComp', 'CCComp', 'BridgeAdjust', 'RMP', 'RMPs',
                #'Irmp', 'taum', 'taupars', 'taufunc', 'Rin', 'Rin_peak', 'tauh_tau', 'tauh_bovera', 'tauh_Gh', 'tauh_vss']
                # if "RMP" in feat:
                #     rmps.append(feat["RMP"] - 12)
                #     rins.append(feat["Rin"])
                #     taus.append(feat["taum"])
                #     if feat["tauh_tau"] is None:
                #         taus.append(np.nan)
                #     else:
                #         tauhs.append(feat["tauh_tau"])

            db.loc[c, "Rin"] = np.nanmean(db.loc[c, "Rin"])
            db.loc[c, "Rin"] = np.nanmean(db.loc[c, "Rin"])
            db.loc[c, "taum"] = np.nanmean(db.loc[c, "taum"])
            #     print(tauhs)
            db.loc[c, "tauh"] = np.nanmean(db.loc[c, "tauh"])
        return db

    """
    Utility to plot data with joint plot, but different hues and some flexibility
    see: https://stackoverflow.com/questions/35920885/how-to-overlay-a-seaborn-jointplot-with-a-marginal-distribution-histogram-fr

    usage:
    multivariateGrid('x', 'y', 'kind', df=df)
    """

    def multivariateGrid(
        self, col_x, col_y, col_k, df, k_is_color=False, yscale="linear", scatter_alpha=0.5
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
            df_group[col_x].values[~pd.isnull(df_group[col_x].values)], ax=g.ax_marg_x, color="grey"
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

    def compare_plot(self, xdat, ydat, db, huefactor=None, figno=1, multivariate=False, title=None):
        print(db["cell_type"].values)
        print(db["cell_type"].isin(order))
        midbs = db.loc[db["cell_type"].isin(order)]

        f, ax = mpl.subplots(1, 1)
        PH.nice_plot(ax)

        #     ax = matplotlib.axes.Axes(f, rect=[0.1, 0.1, 0.8, 0.8])
        sns.boxplot(
            x=xdat,
            y=ydat,
            hue=huefactor,
            data=midbs,
            ax=ax,
            order=order,
            hue_order=None,
            orient=None,
            color=None,
            palette=None,
            saturation=0.5,
            width=0.8,
            dodge=True,
            fliersize=4,
            linewidth=0.5,
            whis=1.5,
            notch=False,
        )
        for patch in ax.artists:
            r, g, b, a = patch.get_facecolor()
            patch.set_facecolor((r, g, b, 0.3))

        sns.swarmplot(
            x=xdat,
            y=ydat,
            data=midbs,
            order=order,
            hue=huefactor,
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
            self.multivariateGrid(xdat, ydat, huefactor, midbs, k_is_color=False, scatter_alpha=0.5)
        handles, labels = ax.get_legend_handles_labels()

        # When creating the legend, only use the first two elements
        # to effectively remove the last two.
        legs = list(set(midbs[huefactor]))
        nleg = len(legs)  # print(nleg)
        l = mpl.legend(
            handles[0:nleg], labels[0:nleg]
        )  # , bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

    def plot_data(self, midb2, plotwhat):
        """
        Plot decay tau versus cell type, sort by temperature
        """
        for c in midb2.index:
            if "_VC_increase_" in midb2.iloc[c]["data_complete"]:
                print(midb2.iloc[c]["cell_id"], midb2.iloc[c]["data_complete"])

        match plotwhat:
            case "temperature":
                self.compare_plot(
                    "cell_type",
                    "tau2",
                    midb2,
                    huefactor="temperature",
                    figno=1,
                    multivariate=False,
                    title="Decay Tau by celltype, temperature",
                )

            case "tau2":
                """
                Plot decay tau versus cell type, sort by max score
                """
                self.compare_plot(
                    "cell_type",
                    "tau2",
                    midb2,
                    huefactor="maxscore_thr",
                    figno=2,
                    multivariate=False,
                    title="Decay Tau by celltype, maxscore threshold=1.3",
                )

            case "decay_tau":
                """
                Plot decay tau versus cell type, include only cells with maxscore > threshold
                """
                midbx = midb2.loc[midb2["maxscore"] > 1.3]  # reduce data set by maxscore
                midbx = midbx.loc[midbx["cell_type"].isin(order)]
                self.compare_plot(
                    "cell_type",
                    "tau2",
                    midbx,
                    huefactor="temperature",
                    figno=3,
                    multivariate=False,
                    title="Decay tau by celltype, maxscore>1.3",
                )

                """
                Plot max score versus cell type, sort by age
                """
            case "max_score":
                print(set(list(midb2["agegroup"])))
                self.compare_plot(
                    "cell_type",
                    "maxscore",
                    midb2,
                    huefactor="agegroup",
                    figno=4,
                    multivariate=False,
                    title="maxscore celltype, age",
                )

                """
                Plot RMP and other measures versus cell type, sort by age
              """
            case "RMP":
                self.compare_plot(
                    "cell_type",
                    "RMP",
                    midb2,
                    huefactor="agegroup",
                    figno=45,
                    multivariate=False,
                    title="RMP by celltype, age",
                )
            case "Rin":
                self.compare_plot(
                    "cell_type",
                    "Rin",
                    midb2,
                    huefactor="agegroup",
                    figno=46,
                    multivariate=False,
                    title="Rin by celltype, age",
                )
            case "taum":
                self.compare_plot(
                    "cell_type",
                    "taum",
                    midb2,
                    huefactor="agegroup",
                    figno=47,
                    multivariate=False,
                    title="Taum by celltype, age",
                )
            case "tauh":
                self.compare_plot(
                    "cell_type",
                    "tauh",
                    midb2,
                    huefactor="agegroup",
                    figno=48,
                    multivariate=False,
                    title="Tauh by celltype, age",
                )

                """
                Plot mean first event latency versus cell type, sort by maxscore true/false        """

            case "maxscore":
                midby = midb2.loc[midb2["maxscore"] > 1.3]
                self.compare_plot(
                    "cell_type",
                    "latency",
                    midby,
                    huefactor="temperature",
                    figno=5,
                    multivariate=False,
                    title="Latency by celltype, temperature for Responding cells",
                )

            case "latency":
                midby = midb2.loc[midb2["temperature"].isin(["34C"])]
                self.compare_plot(
                    "cell_type",
                    "latency",
                    midby,
                    huefactor="maxscore_thr",
                    figno=55,
                    multivariate=False,
                    title="Latency by celltype, maxscore>1.3, at 34C",
                )

            case "max_amp":
                """
                Plot mean amplitude versus cell type, sort by sort by maxscore true/false
                """
                self.compare_plot(
                    "cell_type",
                    "max_amp",
                    midb2,
                    huefactor="maxscore_thr",
                    figno=6,
                    multivariate=False,
                    title="Maximum Amplitude, maxscore thresh 1.3",
                )
            case "mean_amp":
                self.compare_plot(
                    "cell_type",
                    "mean_amp",
                    midb2,
                    huefactor="maxscore_thr",
                    figno=6,
                    multivariate=False,
                    title="Mean Amplitude, maxscore thresh 1.3",
                )

            case "avg_spont_amp":
                midb3 = midb2[midb2["avg_spont_amps"] > 0]
                for i, c in enumerate(midb3.index):
                    print(midb3.loc[c]["cellID"], midb3.loc[c]["avg_spont_amps"])

                self.compare_plot(
                    "cell_type",
                    "avg_spont_amps",
                    midb3,
                    huefactor="temperature",
                    figno=101,
                    multivariate=False,
                    title="Spontaneous Event Amplitudes by Temperature",
                )
            case "avg_event_qcontent":
                self.compare_plot(
                    "cell_type",
                    "avg_event_qcontent",
                    midb2,
                    huefactor="temperature",
                    figno=102,
                    multivariate=False,
                    title="Event Q content by Temperature",
                )
            case "avg_largest_event_qcontent":
                self.compare_plot(
                    "cell_type",
                    "avg_largest_event_qcontent",
                    midb2,
                    huefactor="temperature",
                    figno=103,
                    multivariate=False,
                    title="largest event qcontent Amplitudes by Temperature",
                )
            case _:
                pass

    # import statsmodels.api as sm
    # from statsmodels.formula.api import ols

    # # is maxscore different between celltypes?
    # # midbs = midbs.reset_index()
    # # for c in midbs.index:
    # #     print(c)
    # #     print(midbs.iloc[c]['cellID'],)
    # #     print(' ', midbs.iloc[c]['maxscore'], midbs.iloc[c]['celltype'])
    # # print(midbx.iloc[:]['celltype'])
    # # print(midb2.loc[midb2['celltype'] == 'cartwheel'])
    # c
    # midby = midby.loc[midby['celltype'].isin(order)]
    # model = ols("tau2 ~ C(celltype, Sum)", midby).fit()
    # # print(midby.loc[midby['celltype'] == 'tuberculoventral'])

    # table = sm.stats.anova_lm(model, typ=2) # Type 2 ANOVA DataFrame

    # print(table)
    # print(model.nobs)

    # print(model.summary())

    """
    Plot max score versus cell type, sort by temperature
    """
    # order = ['bushy', 't-stellate', 'd-stellate', 'pyramidal', 'tuberculoventral', 'cartwheel']
    # midbs = midb2.loc[midb2['celltype'].isin(order)]
    # compare_plot('celltype', 'maxscore', midbs, huefactor='temperature',  figno=61, multivariate=False)

    # # %%
    # print(evdb.head())
    # onecell = evdb.loc['2017.03.01_000/slice_000/cell_001']
    # print('celltype: ', onecell['celltype'])
    # print('SR: ', onecell['SR'])
    # print('prots: ', onecell['protocols'])
    # print('shufflescore: ', onecell['shufflescore'])
    # print('scores: ', onecell['scores'])
    # print('event_qcontent: ', onecell['event_qcontent'])
    # print('event_amp: ', len(onecell['event_amp'][0][0]))
    # print('event_qcontent: ', len(onecell['event_qcontent']))
    # print('positions: ', len(onecell['event_amp'][0][0]))
    # print('largest_event_qcontent', onecell['largest_event_qcontent'])
    # print('cell type: ', onecell['celltype'])
    # print('firstevent_latency: ', onecell['firstevent_latency'][2])
    # print('eventp: ', onecell['eventp'][2])
    # print('area_fraction_Z', onecell['area_fraction_Z'])
    # # print(onecell['event_amp'])

    # # %%
    # db[db['date'].str.contains('2017.07.12_000')]

    def get_cell(self, cellname, db):
        # get cell entry in main db
        cn = cellname.split("~")
        day = cn[0]
        sliceno = cn[1]
        cellno = cn[2]
        #     print(day, sliceno, cellno)
        m = db.loc[
            (db["date"] == day) & (db["cell_cell"] == cellno) & (db["slice_slice"] == sliceno)
        ]  # &
        # (db['cell_type'] == celltype)]
        # print('m: ', cellname, m['cell_type'].values)
        return m["cell_type"].values

    def one_cell(self, d):
        bp = self.expts["databasepath"]
        temp_path = Path(bp, "temp_pdfs")
        temp_path.mkdir(exist_ok=True)  # be sure of dir...
        tau_path = Path(bp, self.expts["directory"], "tau_fits")
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
            return None

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
        cn = self.get_cell(str(Path(maps[0]).parent).replace("/", "~"), db=self.db)

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
            names of data in the map dictionary
            dict_keys(['criteria', 'onsets', 'peaktimes', 'smpks', 'smpksindex', 
            'avgevent', 'avgtb', 'avgnpts', 'avgevoked', 'avgspont', 'aveventtb', 
            'fit_tau1', 'fit_tau2', 'fit_amp', 'spont_dur', 'ntraces', 'evoked_ev', 'spont_ev', 'measures', 'nevents'])
            """

            nev = 0
            avedat = None

            if d[maps[mapno]] is None or d[maps[mapno]]["events"] is None:
                continue
            minisum = d[maps[mapno]]["events"]
            # print(minisum)

            for i, evx in enumerate(minisum):
                ev = minisum[i]
                if ev is None:
                    continue
                if (
                    not ev.average.averaged
                    or ev.average.avgeventtb is None
                    or ev.average.avgevent is None
                ):
                    continue

                ax[mapno].plot(
                    ev.average.avgeventtb,
                    ev.average.avgevent,
                    color=[0.25, 0.25, 0.25],
                    linewidth=0.5,
                )
                if ev.average.best_fit is not None:
                    ax[mapno].plot(
                        ev.average.avgeventtb, ev.average.best_fit, color="c", linewidth=0.5
                    )
                all_averages.append(ev.average.avgevent)
                all_fits.append(ev.average.best_fit)
                all_tau1.append(ev.average.fitted_tau1)
                all_tau2.append(ev.average.fitted_tau2)
                all_amp.append(ev.average.amplitude)
                nev += 1
            if nev == 0:
                ax[mapno].text(
                    0.5,
                    0.5,
                    "No events detected",
                    transform=ax[mapno].transAxes,
                    fontsize=12,
                    verticalalignment="bottom",
                    horizontalalignment="center",
                    bbox=props2,
                )
                continue
            # if nev < 10:
            #     ax[mapno].text(0.5, 0.5, f"Too Few Events", transform=ax[mapno].transAxes,
            #         fontsize=10,
            #         verticalalignment='bottom', horizontalalignment='center', bbox=props2)
            #     continue
            # if ev.average.avgeventb is None or ev.average.best_fit is None:
            #     print("No average event data", maps[mapno])
            #     continue
            neventsum += nev
            print("MAPS MAPNO: maps[mapno]: ", maps[mapno])
            results.append(
                {
                    "cell_id": str(Path(maps[mapno]).parent),  # .replace("/", "~"),
                    # + f"~map_{mapno:03d}",
                    "cell_type": cn,
                    "map": str(Path(maps[mapno]).name),
                    "tau1": ev.average.fitted_tau1,
                    "tau2": ev.average.fitted_tau2,
                    "Amplitude": ev.average.amplitude * 1e12,
                    "bfdelay": 1.0,
                    "Nevents": nev,
                    "FitError": ev.average.avg_fiterr,
                }
            )
            # avedat = np.array(avedat)/nev
            # ax[mapno].plot(ev.average.avgeventtb, np.mean(all_averages), 'r-', linewidth=2)
            # ax[mapno].plot(ev.average.avgeventtb, np.mean(all_fits), 'c--')
            textstr = f"Tau1: {all_tau1[-1]*1e3:.3f} ms\n"
            textstr += f"Tau2: {all_tau2[-1]*1e3:.3f} ms \n"
            textstr += f"Amp: {all_amp[-1]*1e12:.3e}\nNMaps: {nev:4d}"
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
            # print(textstr)

        df = pd.DataFrame(results)

        ofile = Path(tau_path, str(Path(maps[mapno]).parent).replace("/", "~") + ".pdf")
        print("******* tau output file ", ofile)
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

    # AGGREGATE taus (_taus.pkl):
    # The next section reads the files in the base "event" directory and assembles the measurements
    # into a new dataframe, which is written to disk.
    # This might be named "NF107Ai32_Het_taus.pkl" for example. It is read by the code above.....
    def get_events(self):
        print("Getting events")
        bp = Path(self.expts["databasepath"], self.expts["directory"], "events")
        files = sorted(bp.glob("*.pkl"))
        print("# event files: ", len(files))
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
            ]
        )
        for n, f in enumerate(list(files)):
            if str(f).find("_signflip") > 0 or str(f).find("_alt") > 0:  # non-cannonical measures
                print("excluding : ", f)
                continue
            print(f"{str(f.name):s}   n={n:d}")

            eventdata = pd.read_pickle(open(f, "rb"), compression=None)
            df2 = self.one_cell(eventdata)
            if df2 is None or len(df2) == 0:
                continue
            # print("df2 columns: ", df2.columns)
            # print("cell ids: ", df2["cell_id"].values)f
            df = pd.concat([df, df2], sort=False)
        with open(self.tau_db_name, "wb") as f:
            pd.to_pickle(df, f)
        return df

    # %%
    def plot_one(self, dp, axi):

        Qlist = []
        Alist = []
        HWlist = []
        HWDlist = []
        HWUlist = []
        for x in dp["events"][0].keys():
            m = dp["events"][0][x]["measures"][0]
            if len(m["Q"]) > 0:
                #         print(len(m['Q']), m['Q'])
                Qlist.extend(m["Q"])
                Alist.extend(m["A"])
                HWlist.extend(m["HW"])
                HWDlist.extend(m["HWdown"])
                HWUlist.extend(m["HWup"])

        Qlist = np.array(Qlist)
        Alist = np.array(Alist)
        HWlist = np.array(HWlist)
        HWDlist = np.array(HWDlist)
        HWUlist = np.array(HWUlist)

        axi.plot(Alist * 1e12, Qlist * 1e9, "ro", markersize=3)
        axi.set_xlim(0, 200)
        axi.set_ylim(0, 1.0)

    def make_tau_file(self, df):

        pl = ["tau1", "tau2", "amplitude"]
        f, ax = mpl.subplots(1, len(list(df.keys())))
        ax = np.array(ax)
        ax = ax.ravel()
        for ia, i in enumerate(d.keys()):
            dp = d[i]
            plot_one(dp, ax[ia])
        mpl.show()

        # %%
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
                print("m: ", m["cell_type"].values, davg.loc[davg["cell"] == c]["tau1"].values[0])

        dbase = Path(expts["databasepath"], expts["directory"])
        df.to_pickle(Path(dbase, "NF107Ai32_Het_taus_x.pkl"))
        df.to_excel(Path(dbase, "NF107Ai32_Het_taus_x.xlsx"))

    """
    Compute the depression ratio of evoked events for maps in cells that have 
    shuffle score > 1.3 and points in map with Z score > 2.1 for the first stimulus
    """

    def depression_ratio(self, p=None, stim_N=4):
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


if __name__ == "__main__":
    print("running main")
    MEA = MapEventAnalyzer("NF107Ai32_Het")
    MEA.plot_data(MEA.merged_db, "tau2")
    mpl.show()
