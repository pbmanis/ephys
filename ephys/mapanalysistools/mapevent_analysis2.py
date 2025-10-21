"""
Third level of analysis for mapping data.

    The first level is to do the initial analysis of the maps, event detection, etc. this
    is done in datatables, with analyze selected maps or analyze all maps.
    The results from the first level of analysis are written to the events folder,
    as *individual* files for each cell.

    The second level is mapevent_analyzer.py, which takes the individual files and
    combines some of the results from the individual maps into a single file, <experimentname>_event_summary.pkl.

    This third step provides routines to:
    1. Generate the tau (event time constants) database (df = make_taus(basepath))

    2. Merge the different databases into a single database (df = merge_db(db))
        This includes getting the taus and the event summary data, and providing
        additional columns for aggregated analysis and plotting.

"""

import dataclasses
import datetime
import itertools
import logging
import os
import pathlib
import re

import sys
from pathlib import Path
from typing import Union

import matplotlib.pyplot as mpl
import numpy as np
import pandas as pd
import seaborn as sns

from pylibrary.plotting import plothelpers as PH
from pylibrary.tools import cprint as CP

import ephys.gui.data_table_functions as DTF
import ephys.mini_analyses.mini_event_dataclass_reader_V1 as MEDC
import ephys.tools.filename_tools as FT
import ephys.tools.map_cell_types as MCT
from ephys.tools.get_configuration import get_configuration

DTFuncs = DTF.Functions()

# go up one directory level from this file's directory:
current_dir = Path.cwd()


Logger = logging.getLogger("MapAnalysis2_Log")

expt = "NF107Ai32_Het"


class MapEventAnalyzer:
    def __init__(self, experiment: str, force_update: bool = False):
        self.expts = experiment
        self.force_map_update = force_update
        self.EventColumns = [
            "cell_id",
            "cell_type",
            "map",
            "tau1",
            "tau2",
            "eventAmplitude",
            "fitAmplitude",
            "best_fit",
            "bfdelay",
            "Nevents",
            "FitError",
        ]

        self.cell_order = [
            "bushy",
            "t-stellate",
            "d-stellate",
            "octopus",
            "pyramidal",
            "tuberculoventral",
            "giant",
            "cartwheel",
        ]
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
        print("tau db file exists: ", self.tau_db_name, self.tau_db_name.is_file())
        if not self.tau_db_name.is_file() or self.force_map_update:
            print("... getting tau events")
            self.tau_db = self.get_events()  # create             the file and get the data

        self.tau_db = pd.read_pickle(self.tau_db_name, compression=None)
        self.tau_db.reset_index(inplace=True)
        print("\nTau database head: \n")
        print(self.tau_db.head(50))
        self.merged_db = self.merge_db(tau_db=self.tau_db, verbose=False)
        # print("Merged database: \n", self.merged_db.cell_type)

    def renamecells(self, db):
        for c in db.index:
            dx = db.iloc[c]
            cell = str(Path(dx["date"], dx["slice_slice"], dx["cell_cell"]))
            db.loc[c, "cell_id"] = cell
            db.loc[c, "cell_type"] = MCT.map_cell_type(db.loc[c, "cell_type"])
        return db

    def remove_cells(self, db):
        removelist = ["2017.06.23_000/slice_003/cell_000"]
        for rm in removelist:
            db = db[db.cell_id != rm]
        return db

    def merge_db(self, tau_db=None, verbose: bool = False):
        """
        Combine event summary and taus with the main db to get ALL information on a per-cell basis

        Returns the merged pandas database
        """

        """
        Insert a new column, cellID, for matching with other db's
        and population it with a path version for searches

        The following database names are present here:
        tau_db: tau analysis database (has full cell id)
        self.db: the main (datasummary-derived) database (has full cell_id)
        evdb: the event summary database (does not have full cell_id)
        midb2: the merged database of evdb and tau_db
        midb3: the merged database of midb2 and self.db
        midbavg: the final merged database with averages computed
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

        event_summary_filename = Path(
            self.expts["analyzeddatapath"],
            self.expts["directory"],
            self.expts["eventsummaryFilename"],
        )
        if not event_summary_filename.is_file():
            print(f"File not found: {event_summary_filename}")
            raise FileNotFoundError()
        with open(event_summary_filename, "rb") as fh:
            evdb = pd.read_pickle(f"", compression=None)
        # seems evdb can be a dictionary or a pd dataframe, from different versions of
        # preprocessing. If it is a dict, convert to dataframe
        if isinstance(evdb, dict):
            evm = {}
            for i, c in enumerate(evdb.keys()):  # reformat dict so cells are in dict
                evm[i] = {"cell": c}
                for d in evdb[c].keys():
                    evm[i][d] = evdb[c][d]

            evdb = pd.DataFrame(evm).transpose()
            evdb.rename(columns={"cell": "cell_id"}, inplace=True)
        tau_db.drop("cell_type", axis=1, inplace=True)
        evdb.drop("celltype", axis=1, inplace=True)
        midb2a = pd.merge(left=evdb, right=tau_db, left_on="cell_id", right_on="cell_id")
        midb2 = pd.merge(left=midb2a, right=self.db, left_on="cell_id", right_on="cell_id")
        midb2.rename(
            # columns={"celltype_x": "cell_type", "cell_id_x": "cell_id"},
            columns={"cell_id_x": "cell_id"},
            inplace=True,
        )
        midb2_len = len(midb2)
        # add columns that need to be calculated (summarized)
        midb2["age_category"] = ["" for _ in range(midb2_len)]
        midb2["maxscore"] = ["" for _ in range(midb2_len)]
        midb2["maxscore_thr"] = ["" for _ in range(midb2_len)]
        midb2["agegroup"] = ["" for _ in range(midb2_len)]
        midb2["latency"] = ["" for _ in range(midb2_len)]
        midb2["mean_amp"] = ["" for _ in range(midb2_len)]
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
            try:
                df_iv = pd.read_pickle(fh, compression=None)
            except:
                try:
                    df_iv = pd.read_pickle(fh, compression="gzip")
                except Exception as exc:
                    print(f"Error reading file: {df_iv_name}, likely format (copression) issue")
        df_iv.drop("cell_type", axis=1, inplace=True)

        def _make_cellid(row):
            return FT.make_cellid_from_slicecell(row["cell_id"])

        df_iv["cell_id"] = df_iv.apply(_make_cellid, axis=1)
        midb3 = pd.merge(left=midb2, right=df_iv, left_on="cell_id", right_on="cell_id")
        args = [
            "cell_type",
            "date",
            "sex",
            "weight",
            "important",
            "age",
            "animal_identifier",
            "celltype",
            "protocols",
        ]
        for arg in args:
            if arg + "_y" in midb3.columns:
                midb3.drop(arg + "_y", axis=1, inplace=True)
            if arg + "_x" in midb3.columns:
                midb3.rename(columns={arg + "_x": arg}, inplace=True)

        for c in midb3.index:
            dx = midb3.iloc[c]
            if dx["temperature"] in ["room temp", "", " ", "room temperature", "25", 25]:
                #         print(f"resetting temp: <{str(dx['temperature']):s}>")
                midb3.loc[c, "temperature"] = "25C"

            midb3.loc[c, "age_category"] = DTFuncs.categorize_ages(midb3.loc[c], self.expts)
            # print(sorted([c for c in midb3.columns]))
            # print([x[0] for x in dx.firstevent_latency if x is not None])
            # print([x[0] for x in dx.allevent_latency if x is not None])
            # print(dx.eventp)
            # return
            midb3 = self.average_datasets(midb3, c, dx, "firstevent_latency", "latency", scf=1e3)
            midb3 = self.average_datasets(midb3, c, dx, "spont_amps", "avg_spont_amps", scf=-1e12)
            midb3 = self.average_datasets(midb3, c, dx, "event_qcontent", "avg_event_qcontent")
            midb3 = self.average_datasets(
                midb3, c, dx, "largest_event_qcontent", "avg_largest_event_qcontent"
            )

            midb3 = self.average_datasets(
                midb3, c, dx, "largest_event_qcontent", "avg_largest_event_qcontent"
            )
            # print("\nLatencies")
            # print("\nFirstEvent Latency")
            # for i, lat in enumerate(midb3["firstevent_latency"].values):
            #     print(i, lat)
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
            # for k in dx.keys():
            #     print("dx keys: ", k)
            midb3.loc[c, "maxscore"] = np.clip(np.log10(mscore), 0.1, 5.0)
            midb3.loc[c, "maxscore_thr"] = np.clip(np.log10(mscore), 0.1, 5.0) > 1.3
            midb3.loc[c, "mean_amp"] = self.compute_npfunc(dx["fitAmplitude"], 1.0)
            midb3.loc[c, "max_amp"] = self.compute_npfunc(dx["fitAmplitude"], 1.0, func=np.nanmax)

        midbavg = self.compute_averages(midb3)
        if verbose:
            self.print_cell_ids(self.db, "DataSummary database", verbose=verbose)
            self.print_cell_ids(tau_db, "TAU (Tau database)", verbose=verbose)
            self.print_cell_ids(evdb, "EVDB (event databse)", verbose=verbose)
            self.print_cell_ids(midb2a, "MIDD2a (Tau and EVDB merge)", verbose=verbose)
            self.print_cell_ids(midb2, "MIDD2 (MIDD2 and DataSummary merge)", verbose=verbose)
            self.print_cell_ids(midb3, "MIDD3 (MIDD2 and df_iv merge)", verbose=verbose)
            self.print_cell_ids(midbavg, "MIDBAVG (averaged measurements)", verbose=verbose)
        return midbavg

    def print_cell_ids(self, db, dataname, verbose: bool = False):
        if not verbose:
            return
        print(f"\nData set: {dataname}")
        if "cell_type" in db.columns:
            print("cell types: ", set(list(db["cell_type"])))
        else:
            print("Dataset does not have cell types")
        ids = list(set(list(db["cell_id"])))
        print("# of entries: ", len(ids))
        print("IDs, sorted: ", sorted(ids))
        print("db.head(): ", db.head())
        print("*" * 80)
        # id2 = [id1 for id1 in ids if id1.startswith("2020")]
        # print("ids: ", id2)
        return

    def matchup_summarydb_events(self):
        """matchup_summarydb_events match the cell_ids for in the summary database with
        the keys in the events database. This is necessary because the events database
        does not have the full cell_id, only the date/slice/cell version. This function
        updates the events database with the full cell_id (e.g., leading path)

        If there is more than one cell ID in the events database, then we need to figure out the match...

        **** This ONLY needs to be run once, to update the events database with the full cell_id ****
        **** it should not be run repeatedly ****
        (11/1/2024)
        """
        event_summary_filename = Path(
            self.expts["analyzeddatapath"],
            self.expts["directory"],
            self.expts["eventsummaryFilename"],
        )
        event_file_path = Path(self.expts["analyzeddatapath"], self.expts["directory"], "events")
        fns = sorted(list(event_file_path.glob("*.pkl")))
        print("Number of event files: ", len(fns))
        # print("day before: ", day_before, "day after: ", day_after)
        with open(event_summary_filename, "rb") as fh:
            evdict = pd.read_pickle(fh, compression=None)
        evdb = pd.DataFrame(evdict).transpose()
        evdb.reset_index(names=["cell_id"], inplace=True)

        for cellid in self.db.cell_id:
            dx = evdb.loc[evdb.cell_id == cellid]
            if len(dx) > 1:
                print("More than one cell ID for: ", cellid)
                # print(dx)
                exit()
            elif len(dx) == 0:
                # print("No match in the event databse for: ", cellid)
                # try reduced cellid:
                short_cell_id = str(Path(*Path(cellid).parts[-3:]))
                # print("short cell id: ", short_cell_id)
                dx = evdb.loc[evdb.cell_id.values == short_cell_id]
                print(
                    "checking short match: ",
                    "dx: ",
                    dx.cell_id.values,
                    "short cell: ",
                    short_cell_id,
                    "full cell: ",
                    cellid,
                )
                evdb.loc[evdb.cell_id.values == short_cell_id, "cell_id"] = cellid
                print("changed to: ", evdb.loc[evdb.cell_id.values == cellid, "cell_id"].values)
            else:
                print("Matched: ", cellid)
        # print(evdb.tail(20))
        event_summary_filename = Path(
            self.expts["analyzeddatapath"],
            self.expts["directory"],
            "test_eventsummary.pkl",
            # self.expts["eventsummaryFilename"],
        )
        print("Writing to: ", event_summary_filename)
        with event_summary_filename as fh:
            evdb.to_pickle(fh)

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
            print(row["RMP"])
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

    def compute_npfunc(self, data, scf: float = 1.0, func: object = np.nanmean):
        """compute_npfunc:  Handle cases where there is no data
        to averge by setting result to np.nan

        Parameters
        ----------
        data : list or np array
            The data to be averaged (may have nans)

        scf : float
            Scale factor for the data
        """
        data_out = np.nan
        sa = np.array(data)
        if sa.ndim == 0 or len(sa) == 0:
            # print("no data in sa: ", sa.ndim)
            return np.nan
        elif np.array(sa).ndim == 2:
            return func(sa) * scf
        elif len(sa) == 1 and sa[0] is None:
            print("sa len 1 and sa[0] is None")
            return np.nan
        else:
            try:
                sa = func(sa) * scf
            except ValueError as exc:
                print(f"failed to apply {func}: {data} due to {exc}")
        return sa

    # routine to flatten an array/list.
    #
    def flatten(self, l, ltypes=(list, tuple)):
        i = 0
        while i < len(l):
            while isinstance(l[i], ltypes):
                if not l[i]:
                    l.pop(i)
                    if not len(l):
                        break
                else:
                    l[i : i + 1] = list(l[i])
            i += 1
        return l

    def average_datasets(
        self,
        db: pd.DataFrame,
        c: pd.Index,
        dx: pd.DataFrame,
        measure: str,
        outmeasure: str,
        scf: float = 1.0,
    ):
        stopper = None  # which cell to stop the run on when testing or debugging
        # stopper = "2017.12.04_000/slice_001/cell_001"
        sa = []  # array to hold the accepted data for the average
        if isinstance(dx[measure], float):
            dx[measure] = [dx[measure]]  # convert to element of the list
        sa = [x for x in dx[measure] if x is not None]
        if len(sa) > 1:
            sa = list(itertools.chain.from_iterable(sa))
        sa = np.array(sa).ravel()
        if dx["cell_id"] == stopper:
            print(f"cell_id: {dx['cell_id']}, {dx['cell_type']:s} {measure}: {dx[measure]}")

        if measure == "firstevent_latency":
            # first event latency is measured from the stimulus time, not trace onset
            # see nrk-nf107/src/map_eventanalyzer.py/EventAnalyzer/score_events
            sa = [x for x in sa if x > 0.0]  # limit to positive latencies

        san = self.compute_npfunc(sa, func=np.mean, scf=scf)
        # if measure == "spont_amps":
        #     print("spont_amps: ", san)
        # exit()
        db.loc[c, outmeasure] = san
        if dx["cell_id"] == stopper:
            print(f"   san: {san}")
            # import pyautogui
            # pyautogui.hotkey('command', 'end') # print("\033\u0003")
            raise ValueError()
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

            db.loc[c, "Rin"] = self.compute_npfunc(db.loc[c, "Rin"], 1.0)
            db.loc[c, "RMP"] = self.compute_npfunc(db.loc[c, "RMP"], 1.0)
            db.loc[c, "taum"] = self.compute_npfunc(db.loc[c, "taum"], 1.0)
            db.loc[c, "tauh"] = self.compute_npfunc(db.loc[c, "tauh"], 1.0)
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
    Plot two parameters in distribution, using list of cell_types in the order.
    Color/separate by "hue"
    optional grid
    """

    def compare_plot(self, xdat, ydat, db, huefactor=None, figno=1, multivariate=False, title=None):
        # print(db["cell_type"].values)
        # print(db["cell_type"].isin(order))
        midbs = db.loc[db["cell_type"].isin(self.cell_order)]
        midbs = midbs.loc[db["cell_type"].isin(["pyramidal", "tuberculoventral", "cartwheel"])]

        fig, ax = mpl.subplots(1, 1)
        PH.nice_plot(ax)
        # print(db[xdat])
        # print(db[ydat])
        # if title is not None:
        #     f.suptitle(f"{title:s}\n{self.experiment_name:s}", fontsize=10)
        timestamp = pd.Timestamp.now()
        fig.text(
            x=0.97,
            y=0.02,
            s=f"Generated: {timestamp}",
            ha="right",
            va="bottom",
            transform=fig.transFigure,
            fontdict={"size": 7, "color": "black"},
        )
        if all(pd.isnull(db[ydat].values)):
            ax.text(
                x=0.5,
                y=0.5,
                s="No data",
                fontdict={"size": 24, "color": "red"},
                ha="center",
                va="center",
            )
            mpl.show()
            return

        #     ax = matplotlib.axes.Axes(f, rect=[0.1, 0.1, 0.8, 0.8])
        sns.boxplot(
            x=xdat,
            y=ydat,
            hue=huefactor,
            data=midbs,
            ax=ax,
            order=self.cell_order,
            hue_order=None,
            orient=None,
            color=None,
            palette=None,
            saturation=0.1,
            width=0.8,
            dodge=True,
            fliersize=4,
            linewidth=0.5,
            whis=1.5,
            notch=False,
        )
        # for patch in ax.artists:
        #     r, g, b, a = patch.get_facecolor()
        #     patch.set_facecolor((r, g, b, 0.1))

        sns.swarmplot(
            x=xdat,
            y=ydat,
            data=midbs,
            order=self.cell_order,
            hue=huefactor,
            dodge=True,
            size=3,
            alpha=1,
            color=None,
            linewidth=0,
            ax=ax,
        )

        if multivariate:
            mpl.figure(figno + 10)
            self.multivariateGrid(xdat, ydat, huefactor, midbs, k_is_color=False, scatter_alpha=0.5)
        handles, labels = ax.get_legend_handles_labels()

        # When creating the legend, only use the first two elements
        # to effectively remove the last two.
        legs = list(set(midbs[huefactor]))
        nleg = len(legs)  # print(nleg)
        legend = mpl.legend(
            handles[0:nleg], labels[0:nleg]
        )  # , bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

    def plot_data(self, df: pd.DataFrame, plotwhat: str, maxscore=0):
        """
        Plot decay tau versus cell type, sort by temperature
        """
        # for c in midb2.index:
        #     if "_VC_increase_" in midb2.iloc[c]["data_complete"]:
        #         print(midb2.iloc[c]["cell_id"], midb2.iloc[c]["data_complete"])
        df = df.drop_duplicates(subset=["cell_id"])
        df = df.sort_values(by=["cell_type", "cell_id"])
        for i, dx in df.iterrows():
            print(
                f"cell_id: {dx['cell_id']}, {dx['cell_type']}: {dx['latency']:.2f}, {dx['temperature']}"
            )
        plot_ok = True
        scored_df = df.loc[df["maxscore"] > maxscore]  # reduce data set by maxscore
        scored_df = scored_df.loc[
            scored_df["cell_type"].isin(["pyramidal", "tuberculoventral", "cartwheel"])
        ]
        match plotwhat:
            case "temperature":
                self.compare_plot(
                    "cell_type",
                    "tau2",
                    scored_df,
                    huefactor=df["temperature",],
                    figno=1,
                    multivariate=False,
                    title=f"Decay Tau by cell type, temperature, {maxscore:.2f}",
                )

            case "tau_2":
                """
                Plot decay tau versus cell type, sort by max score
                """
                self.compare_plot(
                    "cell_type",
                    "tau2",
                    scored_df,
                    huefactor="maxscore_thr",
                    figno=2,
                    multivariate=False,
                    title=f"Decay Tau by cell type, maxscore threshold={maxscore:.2f}",
                )

            case "decay_tau":
                """
                Plot decay tau versus cell type, include only cells with maxscore > threshold
                """
                self.compare_plot(
                    "cell_type",
                    "tau2",
                    scored_df,
                    huefactor="temperature",
                    figno=3,
                    multivariate=False,
                    title=f"Decay tau by cell type, maxscore>1.3",
                )

                """
                Plot max score versus cell type, sort by age
                """
            case "max_score":
                # print(set(list(df["agegroup"])))
                self.compare_plot(
                    "cell_type",
                    "maxscore",
                    df,
                    huefactor="agegroup",
                    figno=4,
                    multivariate=False,
                    title="Maxscore by cell type, age (all scores)",
                )

                """
                Plot RMP and other measures versus cell type, sort by age
              """
            case "RMP":
                self.compare_plot(
                    "cell_type",
                    "RMP",
                    df,
                    huefactor="agegroup",
                    figno=45,
                    multivariate=False,
                    title="RMP by cell type, age (all cells)",
                )
            case "Rin":
                self.compare_plot(
                    "cell_type",
                    "Rin",
                    df,
                    huefactor="agegroup",
                    figno=46,
                    multivariate=False,
                    title="Rin by cell type, age (all cells)",
                )
            case "taum":
                self.compare_plot(
                    "cell_type",
                    "taum",
                    df,
                    huefactor="agegroup",
                    figno=47,
                    multivariate=False,
                    title="Taum by cell type, age (all cells)",
                )
            case "tauh":
                self.compare_plot(
                    "cell_type",
                    "tauh",
                    df,
                    huefactor="agegroup",
                    figno=48,
                    multivariate=False,
                    title="Tauh by cell type, age",
                )

                """
                Plot mean first event latency versus cell type, sort by maxscore true/false        """

            case "maxscore":
                self.compare_plot(
                    "cell_type",
                    "latency",
                    scored_df,
                    huefactor="temperature",
                    figno=5,
                    multivariate=False,
                    title=f"Latency by cell type, temperature for Responding cells (maxscore>{maxscore:.2f})",
                )

            case "latency":
                scored_df = scored_df.loc[df["temperature"].isin(["34C"])]
                self.compare_plot(
                    "cell_type",
                    "latency",
                    scored_df,
                    huefactor="maxscore_thr",
                    figno=55,
                    multivariate=False,
                    title=f"Latency by cell type, maxscore>{maxscore:.2f}, at 34C",
                )

            case "max_amp":
                """
                Plot mean amplitude versus cell type, sort by sort by maxscore true/false
                """
                self.compare_plot(
                    "cell_type",
                    "max_amp",
                    scored_df,
                    huefactor="maxscore_thr",
                    figno=6,
                    multivariate=False,
                    title=f"Maximum Amplitude, maxscore thresh {maxscore:.2f}",
                )
            case "mean_amp":
                self.compare_plot(
                    "cell_type",
                    "mean_amp",
                    df,
                    huefactor="maxscore_thr",
                    figno=6,
                    multivariate=False,
                    title=f"Mean Amplitude, maxscore thresh 1.3",
                )

            case "avg_spont_amp":
                df = df[df["avg_spont_amps"] >= 0]

                # for i, c in enumerate(df.index):
                #     print(df.loc[c]["cell_id"], df.loc[c]["avg_spont_amps"], df.loc[c]["protocol"])
                self.compare_plot(
                    "cell_type",
                    "avg_spont_amps",
                    df,
                    huefactor="temperature",
                    figno=101,
                    multivariate=False,
                    title=f"Spontaneous Event Amplitudes by Temperature (all scores)",
                )
            case "avg_event_qcontent":
                self.compare_plot(
                    "cell_type",
                    "avg_event_qcontent",
                    scored_df,
                    huefactor="temperature",
                    figno=102,
                    multivariate=False,
                    title=f"Event Q content by Temperature, {maxscore:.2f}",
                )
            case "avg_largest_event_qcontent":
                self.compare_plot(
                    "cell_type",
                    "avg_largest_event_qcontent",
                    scored_df,
                    huefactor="temperature",
                    figno=103,
                    multivariate=False,
                    title=f"largest event qcontent Amplitudes by Temperature {maxscore:.2f}",
                )
            case _:
                plot_ok = False
                print("Unrecognized plot selection: {plotwhat:s}")
        if plot_ok:
            mpl.show()

    def print_summary_stats(self, df, groups, cell_types):
        # print summary stats on data
        desc_stats = ["nanmean", "nanstd", "count"]
        for g in groups:
            for ct in cell_types:
                print(f"\nCelltype: {ct:s}  Group: {g:s}")
                print(
                    df[(df["cell_type"] == ct) & (df["Group"] == g)].agg(
                        {
                            "tau1": desc_stats,
                            "tau2": desc_stats,
                            "Amplitude": desc_stats,
                        }
                    )
                )

    def plot_taus(
        self,
        df: pd.DataFrame,
        plot_list: list,
        cell_types: list = [
            "bushy",
            "t-stellate",
            "d-stellate",
            "octopus",
            "pyramidal",
            "cartwheel",
            "tuberculoventral",
        ],
        groups: list = ["Control", "Noise", "Salicylate", "Saline"],
    ):

        for ct in df["cell_type"].unique():
            print("ct: ", ct)

        if "Group" in df.columns:
            df = df[df["Group"].isin(groups)]

        print("Dupes? : ", df[df.index.duplicated()])
        df = df[~df.index.duplicated()]
        print("Dupes2? : ", df[df.index.duplicated()])

        print("Cell types in database: ")
        for ct in df["cell_type"].unique():
            print("cell_type: ", ct)
        # exit()

        # self.print_summary_stats(df, groups, cell_types)

        fig, ax = mpl.subplots(1, 3, figsize=(8, 4))
        ax = np.array(ax).ravel()
        pax = {}
        print("plot_taus: dataframe columns: ", df.columns)
        for ia, plot_what in enumerate(plot_list):
            sns.set_theme(style="ticks", palette="pastel")
            print(
                "cell_types: ",
                df["cell_type"].unique(),
                df["Group"].unique(),
                "plotwhat: ",
                plot_what,
            )
            sns.boxplot(
                x="cell_type",
                y=plot_what,
                data=df,
                ax=ax[ia],
                hue="temperature",  # "cell_type",
                # hue_order="cell_type",
                palette="Pastel2",  # ["g", "b", "orange", "magenta"],
                # facecolor='w',
                # edgecolor='k',
                order=cell_types,
                saturation=1.0,
                width=0.8,
            )
            # except:
            #     print("Failed to boxplot: ", dfp["cell_type"], dfp[plot_what])
            sns.swarmplot(
                x="cell_type",
                y=plot_what,
                data=df,
                ax=ax[ia],
                dodge=True,
                hue="temperature",  # "cell_type",
                # hue_order=HueOrder,
                order=cell_types,
                # palette = ["k", "k", "k", "k"],
                size=3,
                alpha=0.5,
                edgecolor="k",
                linewidth=1,
            )
            PH.nice_plot(ax[ia], position=-0.03, direction="outward", ticklength=4.0)
            pax[plot_what] = ax[ia]
        # sns.despine(offset=10, trim=True)

        if "tau1" in pax.keys():
            pax["tau1"].set_ylabel(r"$\tau_{rise}$ (ms)")
            # pax["tau1"].set_ylim(0, 1.5)
        if "tau2" in pax.keys():
            pax["tau2"].set_ylabel(r"$\tau_{fall}$ (ms)")
            # pax["tau2"].set_ylim(0, 15)
        if "Amplitude" in pax.keys():
            pax["Amplitude"].set_ylabel(r"EPSC (pA)")
            # pax["Amplitude"].semilogy()
            # pax["Amplitude"].set_ylim(0, 4000)
        if "SR" in pax.keys():
            pax["SR"].set_ylabel(r"SR (Hz)")
            # pax["SR"].set_ylim(0, 30)
        for aname in pax.keys():
            pax[aname].set_xticklabels(cell_types, rotation=45, fontsize=8, ha="right")
            # pax[aname].set_xlabel("Cell Type")
        # mpl.tight_layout()
        # Fit the ANOVA model
        # print(df['SR'])

        # for measure in pl:
        #     print("Measure: ", measure)
        #     dfok = df
        #     if measure == "SR":
        #         for index in df.index:
        #             print(df.loc[index, measure])
        #         dfok.dropna(subset=[measure])

        #     for cell_type in cell_types:

        #         model = ols(f'{measure} ~ Group', data=dfok[dfok['cell_type']==cell_type]).fit()

        #         # Perform ANOVA
        #         anova_table = sm.stats.anova_lm(model, typ=2)
        #         print("\nCelltype: ", cell_type, "Measure: ", measure)
        #         print(anova_table)
        #         print()

        # for measure in plot_what:
        #     for cell_type in cell_types:
        #         data = [
        #             df.loc[ids, measure].values
        #             for ids in df[df["cell_type"] == cell_type].groupby("Group").groups.values()
        #         ]
        #         dframe = df[df["cell_type"] == cell_type]
        #         try:
        #             dframe[measure] = dframe.apply(numeric, measure)
        #         except:
        #             print("Dframe columns, measure: ", dframe.columns, measure)
        #             continue

        #         for i in range(len(data)):
        #             data[i] = [d for d in data[i] if not np.isnan(d)]
        #         g = [['A']*len(data[0]), ['B']*len(data[1]), ['AA']*len(data[2]), ['AAA']*len(data[3])]
        #         x = np.concatenate(data)
        #         pth = sp.posthoc_tukey_hsd(x, np.concatenate(g))
        #         print(pth)
        # H, p = stats.kruskal(*data)
        # print("\nCelltype: ", cell_type, "Measure: ", measure)
        # if p < 0.05:
        #     color = "m"
        # else:
        #     color = "w"
        # # CP.cprint(color, f"Kruskal-Wallis H-test test: H={H:.5f}  p: {p:.3f}")
        # welches anova

        # aov = pg.welch_anova(dv=measure, between='Group', data=dframe)
        # print(measure, cell_type)
        # print("-"*80, "\n", "Welch Anova: \n",aov)
        # # posthoc = sp.posthoc_mannwhitney(df[df['cell_type'] == cell_type], val_col=measure, group_col='Group', p_adjust = 'holm-sidak')
        # # print("KW posthoc: ", posthoc)
        # # unequal variances, use Games-Howell test:
        # combinations = [["B", "A"], ["B", "AA"], ["B", "AAA"]]
        # print(pg.pairwise_gameshowell(dv=measure, between='Group', data=dframe))
        # print("="*80)

        mpl.show()

    def show_taus(df):
        pl = ["Amplitude"]
        for c in df.index:
            dx = mdb.iloc[c]
            print("show_taus: cellID: ", dx.cellID)
            # if c > 3:
            #     exit()
            print("cellID: ", dx["cellID"], "tau1: ", dx["tau1"], "tau2: ", dx["tau2"])

    """
    Plot max score versus cell type, sort by temperature
    """
    # order = ['bushy', 't-stellate', 'd-stellate', 'pyramidal', 'tuberculoventral', 'cartwheel']
    # midbs = midb2.loc[midb2['cell_type'].isin(order)]
    # compare_plot('celltype', 'maxscore', midbs, huefactor='temperature',  figno=61, multivariate=False)

    # # %%
    # print(evdb.head())
    # onecell = evdb.loc['2017.03.01_000/slice_000/cell_001']
    # print('cell_type: ', onecell['cell_type'])
    # print('SR: ', onecell['SR'])
    # print('prots: ', onecell['protocols'])
    # print('shufflescore: ', onecell['shufflescore'])
    # print('scores: ', onecell['scores'])
    # print('event_qcontent: ', onecell['event_qcontent'])
    # print('event_amp: ', len(onecell['event_amp'][0][0]))
    # print('event_qcontent: ', len(onecell['event_qcontent']))
    # print('positions: ', len(onecell['event_amp'][0][0]))
    # print('largest_event_qcontent', onecell['largest_event_qcontent'])
    # print('cell type: ', onecell['cell_type'])
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
        ]
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
            columns=self.EventColumns,
            index=[0],
        )

        maps = list(d.keys())
        # CP.cprint("c", f"maps: {maps}")

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
        cell_types = self.get_cell(str(Path(maps[0]).parent).replace("/", "~"), db=self.db)

        if isinstance(cell_types, list) or isinstance(cell_types, np.ndarray):
            if len(cell_types) > 0:
                cell_types = cell_types[0]
        if not isinstance(cell_types, str):
            cell_types = "unknown"
        print("cell type: ", cell_types)
        cn = MCT.map_cell_type(cell_types)
        # if cn is None or cn == "" or cn == [] or len(cn) == 0:
        #     cn = "Unknown"
        # cn = str(cn)

        P.figure_handle.suptitle(f"{str(Path(maps[0]).parent):s}  Type: {cn:s}")
        # scf = 1e12
        all_averages = []
        all_fits = []
        all_tau1 = []
        all_tau2 = []
        all_amp = []
        # avg_fiterr = []
        neventsum = 0
        results = []  # hold results to build dataframe at the end
        for mapno, map in enumerate(maps):
            p = Path(maps[mapno])
            # print("MAPS MAPNO: maps[mapno]: ", map, p)
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

            # names of data in the map dictionary
            # dict_keys(['criteria', 'onsets', 'peaktimes', 'smpks', 'smpksindex',
            # 'avgevent', 'avgtb', 'avgnpts', 'avgevoked', 'avgspont', 'aveventtb',
            # 'fit_tau1', 'fit_tau2', 'fit_amp', 'spont_dur', 'ntraces', 'evoked_ev', 'spont_ev', 'measures', 'nevents'])

            nev = 0
            # avedat = None

            if d[map] is None or d[map]["events"] is None:
                CP.cprint("m", f"Map {map} is empty")
                continue
            minisum = d[map]["events"]
            # print("minisum keys: ", minisum.keys())

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
                    ev.average.avgevent - np.mean(ev.average.avgevent[0:3]),
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
            # print("MAPS MAPNO: maps[mapno]: ", map)

            # print(ev.average)

            # print(
            #     "Max event: ", np.max(ev.average.avgevent)*1e12,
            #       "Min: ", np.min(ev.average.avgevent*1e12),
            #       "baseline: ", np.mean(ev.average.avgevent[0:10])*1e12)
            # mpl.plot(
            #     ev.average.avgeventtb,
            #     ev.average.avgevent * 1e12,
            #     color="k",
            #     markersize=3,

            #     label="Average Event",
            # )
            # mpl.plot(
            #     ev.average.avgeventtb,
            #     ev.average.best_fit * 1e12,
            #     "r--",
            #     label="Double Exponential Fit",
            # )
            # mpl.fill_between(ev.average.avgeventtb,
            #                  ev.average.avgevent * 1e12 - ev.average.stdevent * 1e12,
            #                     ev.average.avgevent * 1e12 + ev.average.stdevent * 1e12,
            #                     color="blue",
            #                     alpha=0.2,
            #                     label="Error Band")

            # mpl.show()
            if ev.average.best_fit is None:
                continue

            results.append(
                {
                    "cell_id": str(Path(map).parent),  # .replace("/", "~"),
                    "cell_type": cn,
                    "map": str(Path(map).name),
                    "tau1": ev.average.fitted_tau1,
                    "tau2": ev.average.fitted_tau2,
                    "eventAmplitude": np.min(ev.average.avgevent * 1e12),
                    "fitAmplitude": ev.average.amplitude * 1e12,
                    "best_fit": ev.average.best_fit * 1e12,
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
            textstr += f"\nFitErr: {ev.average.avg_fiterr:.3e}"
            # place a text box in upper left in axes coords
            ax[mapno].text(
                0.95,
                0.05,
                textstr,
                transform=ax[mapno].transAxes,
                fontsize=7,
                verticalalignment="bottom",
                horizontalalignment="right",
                bbox=props,
            )
            # print(textstr)

        df = pd.DataFrame(results)
        analysis_time = f"{datetime.datetime.now().strftime('%m/%d/%Y, %H:%M:%S'):s}"
        mpl.text(
            0.95,
            0.02,
            s=analysis_time,
            transform=P.figure_handle.transFigure,
            fontsize=6,
            ha="right",
        )
        ofile = Path(tau_path, str(Path(map).parent).replace("/", "~") + ".pdf")
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

    def get_events(self):
        """AGGREGATE taus (_taus.pkl):
        Reads the files in the base "event" directory and assembles the measurements
        into a new dataframe, which is written to disk.
        The output file might be named "NF107Ai32_Het_taus.pkl" for example.
        It is read by the code above.....
        """
        CP.cprint("y", "\n\n{'*'*60:s}\nGetting events")
        events_path = Path(self.expts["databasepath"], self.expts["directory"], "events")
        event_files = sorted(events_path.glob("*.pkl"))
        CP.cprint("g", f"# event files: {len(event_files)}")
        # basic dataframe that will hold the collated event data
        df = pd.DataFrame(
            columns=self.EventColumns,
        )

        for n, f in enumerate(list(event_files)):
            # if n > 5:
            #     break
            # skip over alterantive analyses in event files
            if str(f).find("_signflip") > 0 or str(f).find("_alt") > 0:  # non-cannonical measures
                print("Excluding : ", f)
                continue
            CP.cprint("g", f"Reading (get_events) from : {str(f.name):s}   n={n:d}")
            CP.cprint("g", f"        file: {f}, Exists: {f.exists()}")
            with open(f, "rb") as fh:
                eventdata = pd.read_pickle(fh, compression="infer")
            df2 = self.one_cell(eventdata)
            if df2 is None or len(df2) == 0:
                continue
            # accumulate data
            df = pd.concat([df, df2], sort=False)
        with open(self.tau_db_name, "wb") as f:
            pd.to_pickle(df, f)
        CP.cprint("g", f"Saved aggregated tau data to : {self.tau_db_name!s}")
        return df

    # %%
    def plot_one(self, dp, axi):
        """plot_one Plot one set of data into one axis

        Parameters
        ----------
        dp : dataframe
            data frame containing analyzed data to be plotted
        axi : matplotlib.axes.Axes
            axis into which the plot will be drawn
        """

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
            self.plot_one(dp, ax[ia])
        mpl.show()

        davg = df[df["cell"].str.contains("avgs", regex=False)]
        for cell_type in self.cell_order:
            for c in davg["cell"]:
                cn = c.split("~")
                day = cn[0]
                sliceno = cn[1]
                cellno = cn[2]
                # print(day, sliceno, cellno)
                m = self.midb3.loc[
                    (db["date"] == day)
                    & (db["cell_cell"] == cellno)
                    & (db["slice_slice"] == sliceno)
                    & (db["cell_type"] == cell_type)
                ]
                print("m: ", m["cell_type"].values, davg.loc[davg["cell"] == c]["tau1"].values[0])

        dbase = Path(self.expts["databasepath"], self.expts["directory"])
        df.to_pickle(Path(dbase, "NF107Ai32_Het_taus_x.pkl"))
        df.to_excel(Path(dbase, "NF107Ai32_Het_taus_x.xlsx"))

    def _get_cell_protocol_data(self, proto: Union[str, pathlib.PurePosixPath]):
        """
        Get the protocol data for the cell in the path p
        """
        proto = pathlib.PurePosixPath(proto)  # make sure of type
        with open(proto, "rb") as f:
            protodata = pd.read_pickle(f)
        return proto.name, protodata

    """
    Compute the depression ratio of evoked events for maps in cells that have 
    shuffle score > 1.3 and points in map with Z score > 2.1 for the first stimulus
    """

    # def _get_cell_protocol_data(self, fn):
    #     # the pickled files have no subdirectory information in the filename, so strip that out
    #     fnx = Path(fn).parts
    #     fnx = fnx[-3:]
    #     fn = "~".join(fnx)
    #     eventspath = Path(self.expts['databasepath'], self.expts['directory'], "events")
    #     # cprint("magenta", f"Fn: {str(fn):s}")
    #     fn = Path(eventspath, fn + ".pkl") # Path(self.NM[self.database]["analyzeddatapath"], self.NM[self.database]['directory'], "events", fn + ".pkl")
    #     with open(fn, "rb") as fh:
    #         d = pd.read_pickle(fh)

    #     protocols = sorted(
    #         list(d.keys())
    #     )  # keys include date/slice/cell/protocol as pathlib Path
    #     return (protocols, d)

    def depression_ratios(self, db: pd.DataFrame, stim_N: int = 4, verbose: bool = False):
        """
        Compute the depression ratio of evoked events for maps in cells that have
        shuffle score > 1.3 and points in map with Z score > 2.1 for the first stimulus
        """
        for i, c in enumerate(db.index):
            dx = db.iloc[c]
            # print(dx['cell_id'])
            cell_id = dx["cell_id"]
            fn = FT.make_event_filename_from_cellid(dx["cell_id"])
            eventfilename = fn
            fn = Path(self.expts["databasepath"], self.expts["directory"], "events", fn)
            with open(fn, "rb") as fh:
                eventdata = pd.read_pickle(fh)
            # print(eventdata.keys())
            self.depression_ratio(
                eventdata,
                eventfilename=eventfilename,
                db=db,
                cell_id=cell_id,
                stim_N=stim_N,
                verbose=verbose,
            )

    def depression_ratio(
        self,
        eventdata: dict,
        eventfilename: Union[str, Path],
        cell_id: str,
        db: pd.DataFrame,
        stim_N: int = 4,
        verbose: bool = False,
    ):
        ratio = np.nan
        for j, protocol in enumerate(eventdata):  # for all protocols in this cell
            prot = Path(protocol).name
            evdata = eventdata[protocol]
            # print(evdata.keys())
            if not prot.startswith("Map_NewBlueLaser_VC_10Hz_0") or prot.find("plus") >= 0:
                continue
            print(cell_id, prot)
            if "scores" not in evdata.keys():
                print("No scores in eventdata")
                print(evdata.keys())
                print(evdata["ZScore"])

                print(dir(evdata["events"][0]))
                print(dir(MEDC))
                evnx = MEDC.Reader(evdata["events"])
                # evnx = dataclasses.asdict(evdata['events'][0])
                flds = dataclasses.fields(evnx.data[0])
                for f in flds:
                    print(dir(f))
                print(evnx.data[0].average)
                exit()
            print(
                "Zscore: ",
                evdata["ZScore"][0],
                "\n   scores: ",
                evdata["scores"],
                "\n   shuffle: ",
                evdata["shufflescore"],
                "\n   Area Fract>Z: ",
                evdata["area_fraction_Z"],
                "\n  #Event Amp: ",
                "\n   #Alleventlat: ",
                evdata["allevent_latency"][0],
            )

            envx = evdata
            if len(envx["stimtimes"]["start"]) < stim_N + 1:
                # print('not enough stim times', self.events[proto]['stimtimes']['start'])
                # print('in ', str(p))
                continue
            # print('Enough stim times', self.events[proto]['stimtimes']['start'])
            # print('in ', str(p))
            stimstarts = envx["stimtimes"]["start"]
            stim1 = [stimstarts[0] + 0.0001, stimstarts[0] + 0.011]
            stim5 = [stimstarts[stim_N] + 0.0001, stimstarts[stim_N] + 0.011]
            # print('eventtimes: ', self.events[u]['eventtimes'])  # trials
            # print('# positions: ', len(self.events[u]['positions']))  # 60 positions
            protoevents = envx["events"]
            amp1 = []
            amp5 = []
            tev1 = []
            tev5 = []
            if verbose:
                print("p: ", str(p))
                print(" # protoevents: ", len(protoevents))
            for trial, pevent in enumerate(protoevents):
                if verbose:
                    print("zs trial: ", len(envx["ZScore"][trial]))
                    print(envx["ZScore"][trial])
                    print(envx["events"][trial][0].keys())  # ['event_amp'][trial])
                    print(len(envx["events"][trial]))
                    for i in range(len(envx["events"][trial])):
                        print(
                            "i peaktimes T: ",
                            i,
                            np.array(envx["events"][trial][i]["smpksindex"]) * 5e-2,
                        )
                        print(
                            "i peaktimes pA: ",
                            i,
                            np.array(envx["events"][trial][i]["smpks"]) * 1e12,
                        )
                # print(trial, len(protoevents), len(envx['ZScore']))
                spots = list(np.where(envx["ZScore"][trial] > 0.2)[0])
                # print('spots: ', len(spots), spots)
                if len(spots) == 0:  # no significant events
                    continue
                for ispot in spots:  # now for all spots in this trial, get the events
                    if (
                        "smpksindex" in protoevents[trial].keys()
                    ):  # handle both orders (old and new...)
                        event_t = np.array(protoevents[trial]["smpksindex"][ispot]) * 5e-5  # time
                        event_a = np.array(protoevents[trial]["smpks"][ispot])  # amplitude
                    else:
                        event_t = (np.array(protoevents[trial][ispot]["smpksindex"]) * 5e-5)[0]
                        event_a = np.array(protoevents[trial][ispot]["smpks"])[0]
                    iev1 = np.argwhere(
                        (event_t > stim1[0]) & (event_t <= stim1[1])
                    )  # select events in the window

                    # print(f"\n{str(event_fn):s}  {str(p):s}")
                    # print(event_t)
                    # print(event_a)
                    # print(iev1)
                    # for i in range(len(event_t)):
                    #     print(f"{i:d}  {event_t[i]:.3f}  {event_a[i]*1e12:.2f}")
                    if len(iev1) > 0:
                        amp1.append(np.max(event_a[iev1[0]]))
                        tev1.append(event_t[iev1[0]][0])
                    else:
                        amp1.append(np.nan)
                        tev1.append(np.nan)
                    iev5 = np.argwhere((event_t > stim5[0]) & (event_t <= stim5[1]))
                    if len(iev5) > 0:
                        amp5.append(np.max(event_a[iev5[0]]))
                        tev5.append(event_t[iev5[0]][0])
                    else:
                        amp5.append(np.nan)
                        tev5.append(np.nan)
                    # print('event indices: ', iev1, iev5)
                    # print(Zs, protoevents[trial]['smpks'][zk], np.array(protoevents[trial]['smpksindex'][zk])*5e-5)
            if verbose:
                # print(amp1, amp5)
                #      print('at:', np.array(amp5))
                #      print(f"amp1: {np.array(amp1)*1e12:.1f}  amp5: {np.array(amp5)*1e12:.1f}")
                print(tev1)
                print(tev5)
                print(amp1)
                print(amp5)
            if len(amp5) == 0:
                amp5 = [0]
            if len(amp1) == 0:
                ratio = np.nan
            else:
                # print(amp1, amp5)
                ratio = np.nanmean(amp5) / np.nanmean(amp1)
            if verbose:
                for i in range(len(amp1)):
                    print(
                        f"amp1: {amp1[i]*1e12:.1f}  amp5: {amp5[i]*1e12:.1f} tev1: {tev1[i]*1e3:.1f}  tev5: {tev5[i]*1e3:.1f}"
                    )
        print(f"Ratio: {ratio:.3f}")
        return ratio


if __name__ == "__main__":
    expt = "NF107Ai32_Het"
    import os

    print(os.getcwd())
    # config = get_configuration.get_configuration('../config/experiments.cfg')
    datasets, experiments = get_configuration("config/experiments.cfg")
    experiment = experiments[expt]

    MEA = MapEventAnalyzer(experiment, force_update=True)
    print([c for c in sorted(MEA.merged_db.columns)])
    print("Merged as reported: \n", MEA.merged_db.cell_type)
    # print("and celltypex: \n", MEA.merged_db["celltype_x"])
    # print("Merged as reported: ", [c for c in sorted(MEA.merged_db.columns)])
    exit()
    
    MEA.plot_taus(
        df=MEA.merged_db,
        plot_list=["tau1", "tau2", "fitAmplitude"],
        groups=["Control", "Noise"],
    )
    mpl.show()
    exit()
    allp = False
    if allp:
        all_plots = [
            "temperature",
            "RMP",
            "Rin",
            "taum",
            "tauh",
            "tau_2",
            "decay_tau",
            "max_score",
            "latency",
            "max_amp",
            "mean_amp",
            "avg_spont_amp",
            "avg_event_qcontent",
            "avg_largest_event_qcontent",
        ]
        for p in all_plots:
            MEA.plot_data(MEA.merged_db, p)
    else:
        MEA.plot_data(MEA.merged_db, "avg_event_qcontent")
    # # MEA.depression_ratios(MEA.merged_db)

    mpl.show()
