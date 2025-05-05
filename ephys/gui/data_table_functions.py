"""General functions for data_tables and data_table_manager
We are using a class here just to make it easier to pass around
"""

import logging
import pprint
import re
import subprocess
import warnings
from pathlib import Path
from typing import Union

import colormaps
import numpy as np
import pandas as pd
import pyqtgraph as pg
import scipy
import seaborn as sns
from pylibrary.plotting import plothelpers as PH
from pylibrary.tools import cprint
from pyqtgraph.Qt import QtCore, QtGui
from scipy.interpolate import spalde, splev, splrep

import ephys
import ephys.datareaders as DR
from ephys.ephys_analysis import spike_analysis
from ephys.tools import annotated_cursor, annotated_cursor_pg, filename_tools, utilities

UTIL = utilities.Utility()
AnnotatedCursor = annotated_cursor.AnnotatedCursor
AnnotatedCursorPG = annotated_cursor_pg.AnnotatedCursorPG

CP = cprint.cprint


class CustomFormatter(logging.Formatter):
    grey = "\x1b[38;21m"
    yellow = "\x1b[33;21m"
    red = "\x1b[31;21m"
    bold_red = "\x1b[31;1m"
    white = "\x1b[37m"
    reset = "\x1b[0m"
    lineformat = "%(asctime)s - %(levelname)s - (%(filename)s:%(lineno)d) %(message)s "

    FORMATS = {
        logging.DEBUG: grey + lineformat + reset,
        logging.INFO: white + lineformat + reset,
        logging.WARNING: yellow + lineformat + reset,
        logging.ERROR: red + lineformat + reset,
        logging.CRITICAL: bold_red + lineformat + reset,
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


def get_git_hashes():
    process = subprocess.Popen(["git", "rev-parse", "HEAD"], shell=False, stdout=subprocess.PIPE)
    git_head_hash = process.communicate()[0].strip()

    ephyspath = Path(ephys.__file__).parent
    process = subprocess.Popen(
        ["git", "-C", str(ephyspath), "rev-parse", "HEAD"],
        shell=False,
        stdout=subprocess.PIPE,
    )
    ephys_git_hash = process.communicate()[0].strip()
    text = f"Git hashes: Proj={git_head_hash[-9:]!s}\n       ephys={ephys_git_hash[-9:]!s}\n"
    return {"project": git_head_hash, "ephys": ephys_git_hash, "text": text}


def create_logger(
    log_name: str = "Log Name",
    log_file: str = "log_file.log",
    log_message: str = "Starting Logging",
):
    logging.getLogger("fontTools.subset").disabled = True
    Logger = logging.getLogger(log_name)
    level = logging.DEBUG
    Logger.setLevel(level)
    # create file handler which logs even debug messages
    filename = Path("logs", log_file)
    if not filename.is_file():
        filename.parent.mkdir(parents=True, exist_ok=True)
        with open(filename, "w") as f:
            pass
    logging_fh = logging.FileHandler(filename)
    logging_fh.setLevel(level)
    logging_sh = logging.StreamHandler()
    logging_sh.setLevel(level)
    log_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s  (%(filename)s:%(lineno)d) - %(message)s "
    )
    logging_fh.setFormatter(log_formatter)
    logging_sh.setFormatter(CustomFormatter())  # log_formatter)
    Logger.addHandler(logging_fh)
    Logger.addHandler(logging_sh)
    Logger.info(log_message)
    return Logger


Logger = create_logger(
    log_name="Spike Analysis",
    log_file="spike_analysis.log",
    log_message="Starting Process Spike Analysis",
)

PrettyPrinter = pprint.PrettyPrinter

datacols = [
    "holding",
    "RMP",
    "RMP_SD",
    "Rin",
    "taum",
    "taum_averaged",
    "dvdt_rising",
    "dvdt_falling",
    "dvdt_ratio",
    "current",
    "AP_thr_V",
    "AP_thr_T",
    "AP_HW",
    "AP_peak_V",
    "AP15Rate",
    "AdaptRatio",
    "AdaptRatio2",
    "AdaptIndex",
    "AdaptIndex2",
    "AHP_trough_V",
    "AHP_depth_V",
    "AHP_relative_depth_V",
    "AHP_trough_T",
    "tauh",
    "Gh",
    # "FiringRate",
]

iv_keys: list = [
    "holding",
    "WCComp",
    "CCComp",
    "BridgeAdjust",
    "RMP",
    "RMP_SD",
    "RMPs",
    "Irmp",
    "taum",
    "taupars",
    "taufunc",
    "taum_averaged",
    "Rin",
    "Rin_peak",
    "tauh_tau",
    "tauh_bovera",
    "tauh_Gh",
    "tauh_vss",
]

spike_keys: list = [
    "FI_Growth",
    "AdaptRatio",
    "AdaptIndex",
    "AdaptIndex2",
    "FI_Curve",
    "FiringRate",
    "AP1_Latency",
    "AP1_HalfWidth",
    "AP1_HalfWidth_interpolated",
    "AP2_Latency",
    "AP2_HalfWidth",
    "AP2_HalfWidth_interpolated",
    "FiringRate_1p5T",
    "AP_peak",
    "AP_peak_V",
    "AP_thr_V",
    "AP_thr_T",
    "AP_HW",
    "dvdt_rising",
    "dvdt_falling",
    "dvdt_ratio",
    "AHP_depth_V",
    "AHP_trough_V",
    "AHP_relative_depth_V",
    "AHP_trough_T" "spikes",
    "iHold",
    "pulseDuration",
    "baseline_spikes",
    "poststimulus_spikes",
]

# map spike measurements to top level keys
mapper: dict = {
    "AP1_HalfWidth": "halfwidth",
    "AP1_HalfWidth_interpolated": "halfwidth_interpolated",
    "AHP_trough_V": "trough_V",
    "AHP_trough_T": "trough_T",
    "AHP_depth_V": "trough_V",
    "AP1_Latency": "AP_latency",
    "AP_peak_V": "peak_V",
    "AP_thr_V": "AP_begin_V",
    "AP_thr_T": "AP_begin_T",
    "AP_HW": "halfwidth",
    "dvdt_rising": "dvdt_rising",
    "dvdt_falling": "dvdt_falling",
    "dvdt_ratio": "dvdt_ratio",
    "AdaptRatio": "AdaptRatio",
    "AdaptIndex": "AdaptIndex",
    "AdaptIndex2": "AdaptIndex2",
}
# map summary/not individual spike data to top level keys
mapper1: dict = {
    "AP15Rate": "FiringRate_1p5T",
    "AdaptRatio": "AdaptRatio",
    "AdaptIndex": "AdaptIndex",
    # "holding": "iHold",
}

iv_mapper: dict = {
    "tauh": "tauh_tau",
    "Gh": "tauh_Gh",
    "taum": "taum",
    "taum_averaged": "taum_averaged",
    "Rin": "Rin",
    "RMP": "RMP",
}


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
    age = float(int("".join(filter(str.isdigit, row.age))))
    return age


def print_spike_keys(row):
    if pd.isnull(row.IV):
        return row
    # print(row.IV)
    return row


class Functions:
    def __init__(self):
        # self.textbox = None
        self.cursor = []  # a list to hold cursors (need to keep a live reference)
        self.textbox = None
        self.status_bar = None

    def set_status_bar(self, status_bar):
        self.status_bar = status_bar

    def get_row_selection(self, table_manager):
        """
        Find the selected rows in the currently managed table, and if there is a valid selection,
        return the index to the first row and the data from that row
        """
        self.selected_index_rows = table_manager.table.selectionModel().selectedRows()
        if self.selected_index_rows is None or len(self.selected_index_rows) == 0:
            return None, None
        else:
            print("selected: ", self.selected_index_rows)
            index_row = self.selected_index_rows[0]
            selected = table_manager.get_table_data(index_row)  # table_data[index_row]
            if selected is None:
                return None, None
            else:
                return index_row, selected

    def get_multiple_row_selection(self, table_manager):
        """
        Find the selected rows in the currently managed table, and if there is a valid selection,
        return a list of indexs from the selected rows.

        """
        self.selected_index_rows = table_manager.table.selectionModel().selectedRows()
        if self.selected_index_rows is None:
            return None
        else:
            rows = []
            print(self.selected_index_rows)
            for row in self.selected_index_rows:
                # index_row = self.selected_index_rows[0]
                # print(dir(row))
                selected = table_manager.get_table_data(row)  # table_data[index_row]
                if selected is None:
                    return None
                else:
                    rows.append(selected)
            return rows

    def get_current_table_selection(self, table_manager):
        index_rows = table_manager.table.selectionModel().selectedRows()
        selected_cell_ids = []
        for selected_row_number in index_rows:
            if selected_row_number is None:
                print("No selected rows")
                break
            cell_id = table_manager.get_selected_cellid_from_table(selected_row_number)
            if cell_id is None:
                # print("cell id is none?")
                continue
            else:
                selected_cell_ids.append(cell_id)
        return selected_cell_ids

    def set_current_table_selection(self, table_manager, cell_ids):
        """set_current_table_selection: Set the current table selection to the cell_ids in the list
        Used to restore the selection after a table update.
        """
        if len(cell_ids) == 0:
            return  # nothing to select
        selection = pg.Qt.QtCore.QItemSelection()  # create a selection object
        for row in range(table_manager.table.model().rowCount()):
            index_rows = table_manager.table.model().index(row, 0)
            if table_manager.get_selected_cellid_from_table(index_rows) in cell_ids:
                selection.select(index_rows, index_rows)
        mode = (
            pg.Qt.QtCore.QItemSelectionModel.SelectionFlag.Select
            | pg.Qt.QtCore.QItemSelectionModel.SelectionFlag.Rows
        )
        results = table_manager.table.selectionModel().select(selection, mode)
        table_manager.table.scrollTo(selection.indexes()[0])
        return results

    def get_datasummary(self, experiment):
        datasummary = Path(
            experiment["analyzeddatapath"],
            experiment["directory"],
            experiment["datasummaryFilename"],
        )
        if not datasummary.exists():
            raise ValueError(f"Data summary file {datasummary!s} does not exist")
        df_summary = pd.read_pickle(datasummary)
        return df_summary

    def rewrite_datasummary(self, experiment, df_summary):
        datasummary = Path(
            experiment["analyzeddatapath"],
            experiment["directory"],
            experiment["datasummaryFilename"],
        )
        df_summary.to_pickle(datasummary)

    def get_datasummary_protocols(self, datasummary):
        """
        Print a configuration file-like text of all the datasummary protocols, as categorized here.

        """
        data_complete = datasummary["data_complete"].values
        print("# of datasummary entries: ", len(data_complete))
        protocols = []
        for i, prots in enumerate(data_complete):
            prots = prots.split(",")
            for prot in prots:
                protocols.append(prot[:-4].strip(" "))  # remove trailing "_000" etc

        allprots = sorted(list(set(protocols)))
        print("# of unique protocols: ", len(allprots))
        # print(allprots)

        # make a little table for config dict:
        txt = "protocols:\n"
        txt += "    CCIV:"
        ncciv = 0
        prots_used = []
        for i, prot in enumerate(allprots):
            if (
                "CCIV".casefold() in prot.casefold() or "CC_".casefold() in prot.casefold()
            ):  # includes CC_taum
                computes = "['RmTau', 'IV', 'Spikes', 'FI']"
                if "posonly".casefold() in prot.casefold():  # cannot compute rmtau for posonly
                    computes = "['IV', 'Spikes', 'FI']"
                txt += f"\n        {prot:s}: {computes:s}"
                prots_used.append(i)
                ncciv += 1
        if ncciv == 0:
            txt += " None"
        txt += "\n    VCIV:"
        nvciv = 0
        for i, prot in enumerate(allprots):
            if "VCIV".casefold() in prot.casefold():
                computes = "['VC']"
                txt += f"\n        {prot:s}: {computes:s}"
                nvciv += 1
                prots_used.append(i)
        if nvciv == 0:
            txt += " None"
        txt += "\n    Maps:"
        nmaps = 0
        for i, prot in enumerate(allprots):
            if "Map".casefold() in prot.casefold():
                computes = "['Maps']"
                txt += f"\n        {prot:s}: {computes:s}"
                nmaps += 1
                prots_used.append(i)
        if nmaps == 0:
            txt += " None"
        txt += "\n    Minis:"
        nminis = 0
        for i, prot in enumerate(allprots):
            cprot = prot.casefold()
            if "Mini".casefold() in cprot or "VC_Spont".casefold() in cprot:
                computes = "['Mini']"
                txt += f"\n        {prot:s}: {computes:s}"
                nminis += 1
                prots_used.append(i)
        if nminis == 0:
            txt += " None"
        txt += "\n    PSCs:"
        npsc = 0
        for i, prot in enumerate(allprots):
            if "PSC".casefold() in prot.casefold():
                computes = "['PSC']"
                txt += f"\n        {prot:s}: {computes:s}"
                npsc += 1
                prots_used.append(i)
        if npsc == 0:
            txt += " None"
        txt += "\n    Uncategorized:"
        allprots = [prot for i, prot in enumerate(allprots) if i not in prots_used]
        nother = 0
        for i, prot in enumerate(allprots):
            if len(prot) == 0 or prot == " ":
                prot = "No Name"
            computes = "None"
            txt += f"\n        {prot:s}: {computes:s}"
            nother += 1
        if nother == 0:
            txt += "\n        None"
        print(f"\n{txt:s}\n")
        # this print should be pasted into the configuration file (watch indentation)

    def moving_average(self, data, window_size):
        """moving_average Compute a triangular moving average on the data over a window

        Parameters
        ----------
        data : _type_
            _description_
        window_size : _type_
            _description_

        Returns
        -------
        _type_
            _description_
        """
        window = np.bartlett(window_size)
        # Normalize the window
        window /= window.sum()

        return np.convolve(data, window, "valid") / window_size

    def get_slope(self, y, x, index, window_size):
        """get_slope get slope of a smoothed curve at a given index

        Parameters
        ----------
        y : _type_
            _description_
        x : _type_
            _description_
        index : _type_
            _description_
        window_size : _type_
            _description_

        Returns
        -------
        _type_
            _description_
        """
        # Smooth the data
        y_smooth = self.moving_average(y, window_size)
        x_smooth = self.moving_average(x, window_size)

        # Adjust the index for the reduced size of the smoothed data
        index -= window_size // 2

        if index < 1 or index >= len(y_smooth) - 1:
            # Can't calculate slope at the start or end
            return None
        else:
            dy = y_smooth[index + 1] - y_smooth[index - 1]
            dx = x_smooth[index + 1] - x_smooth[index - 1]
            return dy / dx

    def spline_fits(
        self,
        x: np.ndarray,
        y: np.ndarray,
        region: list,
        k: int = 5,  # order of bspline( 1, 3, 5)
        s: int = None,  # s parameter
        npts: int = 256,
        label: str = "",
    ):
        # fit the spline
        if s is None:
            s = 2  # int(len(x[region]) / 8)
        # if s > len(x[region]) / 32:
        #     s = int(len(x[region]) / 32)+1
        print(f"Len region {label:s}: {len(x[region]):d}, s: {s:.1f}")
        tck = splrep(
            x[region],
            y[region],
            k=k,
            s=s,
        )  # t=[x[region][2], x[region][-(k+1)]])
        xfit = np.linspace(x[region[0]], x[region[-1]], npts)
        yfit = splev(xfit, tck)  # , der=0)

        # calculate curvature on the spline
        derivs = spalde(xfit, tck)
        derivs = np.array(derivs).T
        num = derivs[2]
        den = 1.0 + (derivs[1] ** 2)
        den = np.power(den, 1.5)
        kappa = num / den
        return tck, xfit, yfit, kappa

    def curvature(self, x, y):

        dx = np.gradient(x)
        dy = np.gradient(y)
        ddx = np.gradient(dx)
        ddy = np.gradient(dy)
        curvature = (ddx * ddy - dx * ddy) / (dx**2 + dy**2) ** 1.5
        return curvature

    def draw_orthogonal_line(self, x, y, index, slope, length, color, ax):
        # Calculate the slope of the orthogonal line
        orthogonal_slope = -1.0 / slope

        # Calculate the start and end points of the orthogonal line
        x_start = x[index] - length / 2
        x_end = x[index] + length / 2
        y_start = y[index] + orthogonal_slope * (x_start - x[index])
        y_end = y[index] + orthogonal_slope * (x_end - x[index])

        # Plot the orthogonal line
        ax.plot([x_start, x_end], [y_start, y_end], color=color)

    def remove_excluded_protocols(self, experiment, cell_id, protocols):
        """remove_excluded_protocols For all protocols in the list of protocols,
        remove any that are in the excludeIVs list for the cell_id.
        In general, protocols will include all protocols run on this cell.

        Parameters
        ----------
        experiment : experiment configuration dictionary
            _description_
        cell_id : str
            The specific cell identifier ('2024.01.01_000/slice_000/cell_001')
        protocols : list
            List of all protocols for this cell

        Returns
        -------
        list
            The list of protocols to keep. List is [] if all are excluded
        """
        if isinstance(protocols, str):
            protocols = protocols.replace(" ", "")  # remove all white space
            protocols = [p for p in protocols.split(",")]  # turn into a list
        if experiment["excludeIVs"] is None:  # no protocols to exclude
            return protocols
        # cell_id = cell_id.item()
        if cell_id in experiment["excludeIVs"].keys():  # consider excluding some or all protocols
            if experiment["excludeIVs"][cell_id]["protocols"] == ["all"]:
                return []
            protocols = [
                protocol
                for protocol in protocols
                if protocol not in experiment["excludeIVs"][cell_id]["protocols"]
            ]
        return protocols

    def check_excluded_dataset(self, day_slice_cell, experiment, protocol):
        if experiment["excludeIVs"] is None:
            return False
        print("     and in exclude list: ", day_slice_cell in experiment["excludeIVs"])
        if experiment["includeIVs"] is not None:
            print(
                "check_excluded_dataset in include list: ",
                day_slice_cell,
                day_slice_cell in experiment["includeIVs"],
            )
            include_flag = day_slice_cell in experiment["includeIVs"]
            if (
                include_flag and protocol in experiment["includeIVs"][day_slice_cell]["protocols"]
            ):  # inclusions
                CP("g", f"    Cell/protocol has entries in Inclusion table: {include_flag!s}")
                return False
        exclude_flag = day_slice_cell in experiment["excludeIVs"]
        if exclude_flag:
            CP("r", f"    Cell has entries in exclusion table: {exclude_flag!s}")
            exclude_table = experiment["excludeIVs"][day_slice_cell]
            CP("r", f"        excluded table data: {exclude_table!s}")
            CP("r", f"        testing protocol: {protocol!s}")
            proto = Path(protocol).name  # passed protocol has day/slice/cell/protocol
            if (
                proto in exclude_table["protocols"]
                or exclude_table["protocols"] == ["all"]
                or exclude_table["protocols"] == ["All"]
            ):
                CP(
                    "y",
                    f"        Excluded cell/protocol: {day_slice_cell:s}, {proto:s} because: {exclude_table['reason']:s}",
                )
                # cannot log when multiprocessing.
                # Logger.info(
                #     f"    Excluded cell: {day_slice_cell:s}, {proto:s} because: {exclude_table['reason']:s}"
                # )
                return True
        else:
            CP("c", f"    Cell has no entries in exclusion table: {exclude_flag!s}")
            return False
        print("    Protocol passed: ", protocol)
        return False

    def get_selected_cell_data_spikes_mpl(
        self, experiment, table_manager, assembleddata, bspline_s
    ):
        self.spike_cursors = []
        self.get_row_selection(table_manager)
        if self.selected_index_rows is None:
            return None
        N = len(self.selected_index_rows)
        colors = colormaps.sinebow_dark.discrete(N)
        for nplots, index_row in enumerate(self.selected_index_rows):
            selected = table_manager.get_table_data(index_row)
            day = selected.date[:-4]
            slicecell = selected.cell_id[-4:]
            pcolor = colors[nplots].colors
            cell_df, cell_df_tmp = filename_tools.get_cell(
                experiment, assembleddata, cell_id=selected.cell_id
            )
            protocols = list(cell_df["Spikes"].keys())
            protocols = self.remove_excluded_protocols(experiment, cell_df["cell_id"], protocols)
            min_index = None
            min_current = 1
            V = None
            min_protocol = None
            spike = None
            for ip, protocol in enumerate(protocols):
                # print('lowest current spike: ', cell_df["Spikes"][protocol]["LowestCurrentSpike"])

                min_current_index, current, trace = self.find_lowest_current_trace(
                    cell_df["Spikes"][protocol]
                )
                if current < min_current:
                    I = current
                    V = trace
                    min_index = min_current_index
                    min_protocol = ip
                    min_current = current
                    spike = cell_df["Spikes"][protocol]
            pp = PrettyPrinter(indent=4)

            if spike is None:
                continue
            try:
                low_spike = spike["spikes"][V][min_index]
            except KeyError:
                print("Min index: ", min_index)
                print("len spikes: ", len(spike["spikes"][V]))
                return None
            print("min index: ", min_index, "trace: ", V)
            print(low_spike)
            if nplots == 0:
                import matplotlib.pyplot as mpl

                # print(low_spike)
                P = PH.regular_grid(
                    1,
                    4,
                    order="rowsfirst",
                    figsize=(12, 4),
                    margins={
                        "bottommargin": 0.1,
                        "leftmargin": 0.07,
                        "rightmargin": 0.07,
                        "topmargin": 0.1,
                    },
                )
                # f, ax = mpl.subplots(1, 4, figsize=(12, 4), gridspec_kw={'hspace': 0.2})
                f = P.figure_handle
                f.suptitle(
                    f"{day:s} {slicecell:s} {Path(protocol).name:s} trace: {low_spike.trace:d} I (nA): {1e9*low_spike.current:.3f}"
                )
                ax = P.axarr[0]
                self.spike_plots_ax = ax
                # Text location in data coords
                props = dict(boxstyle="round", facecolor="wheat", alpha=0.4)
                self.txt1 = ax[1].text(0, 0, "", fontsize=8, bbox=props)
                self.txt0 = ax[0].text(0, 0, "", fontsize=8, bbox=props)
            vtime = (low_spike.Vtime - low_spike.peak_T) * 1e3
            voltage_plot_line = ax[0].plot(vtime, low_spike.V * 1e3, color=pcolor, linewidth=1.25)
            phase_plot_line = ax[1].plot(
                low_spike.V[: low_spike.dvdt.shape[0]] * 1e3, low_spike.dvdt, color=pcolor
            )

            dvdt_ticks = np.arange(-4, 2.01, 0.1)
            t_indices = np.array([np.abs(vtime - point).argmin() for point in dvdt_ticks])
            thr_index = np.abs(vtime - (low_spike.AP_latency - low_spike.peak_T) * 1e3).argmin()
            thr_t = vtime[thr_index]
            peak_index = np.abs(vtime - 0).argmin()
            peak_t = vtime[peak_index]

            # for i in range(len(t_indices)):
            #     local_slope = self.get_slope(
            #         low_spike.V * 1e3, low_spike.dvdt, t_indices[i], 7,
            #     )
            #     if local_slope is not None:
            #         self.draw_orthogonal_line(
            #             low_spike.V * 1e3,
            #             low_spike.dvdt,
            #             index=t_indices[i],
            #             slope=local_slope,
            #             length=5.0,
            #             color=colors[i],
            #             ax=ax[1],
            #         )

            #     ax[1].scatter(
            #     low_spike.V[t_indices[i]] * 1e3,
            #     low_spike.dvdt[t_indices[i]],
            #     s=12,
            #     marker='|',
            #     color=colors[i],
            #     zorder = 10
            # )
            # Plot each point with a different color
            # ax[1].scatter(
            #     low_spike.V[t_indices] * 1e3,
            #     low_spike.dvdt[t_indices],
            #     s=12,
            #     marker='|',
            #     color=colors,
            #     zorder = 10
            # )
            # indicate the threshold on the phase plot
            ax[1].scatter(
                low_spike.V[thr_index] * 1e3,
                low_spike.dvdt[thr_index],
                s=12,
                marker="o",
                color="r",
                zorder=12,
            )
            maxdvdt = np.argmax(low_spike.dvdt) + 1
            rising_phase_indices = np.nonzero(
                (low_spike.Vtime >= low_spike.Vtime[thr_index])
                & (low_spike.Vtime <= low_spike.Vtime[maxdvdt])  # [peak_index])
            )[0]
            falling_phase_indices = np.nonzero(
                (low_spike.Vtime >= low_spike.Vtime[peak_index])
                & (low_spike.V >= low_spike.V[thr_index])
            )[0]
            ahp_phase_indices = np.nonzero(
                (low_spike.Vtime >= low_spike.Vtime[peak_index])
                & (low_spike.V < low_spike.V[thr_index])
            )[0]

            # indicate the AHP nadir on the phase plot
            ahp_index = np.abs(vtime - (low_spike.trough_T - low_spike.peak_T) * 1e3).argmin()
            ax[1].scatter(
                low_spike.V[ahp_index] * 1e3,
                low_spike.dvdt[ahp_index],
                s=12,
                marker="o",
                color="orange",
                zorder=12,
            )
            # indicate the AHP on the voltage trace
            spike_pk_to_trough_T = low_spike.trough_T - low_spike.peak_T
            AHP_t = spike_pk_to_trough_T * 1e3  # in msec, time from peak
            latency = (low_spike.AP_latency - low_spike.peak_T) * 1e3  # in msec
            ax[0].scatter(
                AHP_t,
                low_spike.trough_V * 1e3,
                s=3.0,
                color="orange",
                marker="o",
                zorder=10,
            )
            x_line = [latency, AHP_t]
            # draw dashed line at threshold
            ax[0].plot(
                x_line,
                low_spike.V[thr_index] * np.ones(2) * 1e3,
                "k--",
                linewidth=0.3,
            )
            # draw dashed line at nadir of AHP
            ax[0].plot(
                x_line,
                low_spike.V[ahp_index] * np.ones(2) * 1e3,
                "m--",
                linewidth=0.3,
            )

            # indicate the threshold on the voltage plot
            ax[0].plot(
                latency,
                low_spike.AP_begin_V * 1e3,
                "ro",
                markersize=2.5,
                zorder=10,
            )
            ax[0].plot(
                [
                    (low_spike.left_halfwidth_T - low_spike.peak_T - 0.0001) * 1e3,
                    (low_spike.right_halfwidth_T - low_spike.peak_T + 0.0001) * 1e3,
                ],
                [  # in msec
                    low_spike.halfwidth_V * 1e3,
                    low_spike.halfwidth_V * 1e3,
                ],
                "g-",
                zorder=10,
            )
            # ax[0].plot(
            #     (low_spike.right_halfwidth_T - low_spike.peak_T)
            #     * 1e3,  # in msec
            #     low_spike.halfwidth_V * 1e3,
            #     "co",
            # )
            savgol = True
            bspline = False
            if bspline:  # fit with a bspline
                # rising phase of AP from threshold to peak dvdt
                tck_r, xfit_r, yfit_r, kappa_r = self.spline_fits(
                    vtime, low_spike.V, rising_phase_indices[:-2], k=5, s=bspline_s, label="rising"
                )
                # ax[0].plot(vtime[rising_phase_indices]-peak_t, low_spike.V[rising_phase_indices]*1e3,
                #            color="b", linestyle="--", linewidth=2)

                # falling phase of AP from peak to spike threshold
                tck_f, xfit_f, yfit_f, kappa_f = self.spline_fits(
                    vtime, low_spike.V, falling_phase_indices, label="falling"
                )
                # ahp phase of AP from threshold time on out
                tck_a, xfit_a, yfit_a, kappa_a = self.spline_fits(
                    vtime, low_spike.V, ahp_phase_indices, k=5, s=bspline_s, label="ahp"
                )
            print("len rising phase indices: ", len(rising_phase_indices))
            fits_ok = True
            if savgol:
                from scipy.signal import savgol_filter

                window_length = int(bspline_s)
                if window_length > 5:
                    polyorder = 5
                else:
                    polyorder = 3
                    window_length = 5
                try:
                    yfit_r = savgol_filter(
                        low_spike.V[rising_phase_indices],
                        window_length=window_length,
                        polyorder=polyorder,
                    )
                    yfit_f = savgol_filter(
                        low_spike.V[falling_phase_indices],
                        window_length=window_length,
                        polyorder=polyorder,
                    )
                    yfit_a = savgol_filter(
                        low_spike.V[ahp_phase_indices],
                        window_length=window_length,
                        polyorder=polyorder,
                    )
                except:
                    yfit_r = np.zeros(len(rising_phase_indices))
                    yfit_f = np.zeros(len(falling_phase_indices))
                    yfit_a = np.zeros(len(ahp_phase_indices))
                    fits_ok = False
                xfit_r = vtime[rising_phase_indices]
                xfit_f = vtime[falling_phase_indices]
                xfit_a = vtime[ahp_phase_indices]
                try:
                    kappa_r = self.curvature(xfit_r, yfit_r)
                    kappa_f = self.curvature(xfit_f, yfit_f)
                    kappa_a = self.curvature(xfit_a, yfit_a)
                except:
                    kappa_r = np.zeros(len(rising_phase_indices))
                    kappa_f = np.zeros(len(falling_phase_indices))
                    kappa_a = np.zeros(len(ahp_phase_indices))
                    fits_ok = False

            if fits_ok:
                ax[3].plot(
                    xfit_r,
                    1e3 * np.gradient(yfit_r, xfit_r),
                    color=pcolor,
                    linestyle="-",
                    linewidth=1,
                )
            if (bspline or savgol) and fits_ok:
                ax[0].plot(xfit_f, yfit_f * 1e3, color="cyan", linestyle="--", linewidth=0.5)
                ax[1].plot(
                    yfit_f[:-1] * 1e3,
                    np.diff(yfit_f * 1e3) / np.diff(xfit_f),
                    color="cyan",
                    linestyle="--",
                    linewidth=0.5,
                )

                ax[0].plot(xfit_r, yfit_r * 1e3, color="orange", linestyle="--", linewidth=0.5)
                ax[1].plot(
                    yfit_r[:-1] * 1e3,
                    np.diff(yfit_r * 1e3) / (np.diff(xfit_r)),
                    color="orange",
                    linestyle="--",
                    linewidth=0.5,
                )
                ax[2].plot(
                    yfit_r[1:] * 1e3, kappa_r[1:], color=pcolor, linestyle="--", linewidth=0.5
                )
                ax[0].plot(xfit_a, yfit_a * 1e3, color="m", linestyle="--", linewidth=0.5)
                ax[1].plot(
                    yfit_a[:-1] * 1e3,
                    np.diff(yfit_a * 1e3) / np.diff(xfit_a),
                    color="m",
                    linestyle="--",
                    linewidth=0.5,
                )
                ax[2].plot(yfit_a[1:] * 1e3, kappa_a[1:], color="m", linestyle="--", linewidth=0.25)

                ax[2].plot(
                    yfit_f[1:] * 1e3, kappa_f[1:], color="cyan", linestyle="--", linewidth=0.25
                )
                # horizontal line at 0 curvature
                ax[2].plot([-60, 20], [0, 0], color="black", linestyle="--", linewidth=0.33)

                # m = int(len(low_spike.V[rising_phase_indices]) / 4)
                # if m > len(low_spike.V[rising_phase_indices]) / 4:
                #     m = int(len(low_spike.V[rising_phase_indices]) / 2)
                # print(f"Len ahp: {len(low_spike.V[rising_phase_indices]):d}, m: {m:d}")
                # if m == 0:
                #     break
                # # interpolation in V and dvdt to get a uniform distribution of points in V space
                # # yv = np.interp(xv, low_spike.V[rising_phase_indices], low_spike.dvdt[rising_phase_indices])
                # tck_p = splrep(
                #     low_spike.V[rising_phase_indices],
                #     low_spike.dvdt[rising_phase_indices],
                #     k=5,
                #     s=64,
                # )
                # print("xv: ", xv)
                # print("tck: ", tck_p)
                # yfit_p = splev(xv, tck_p)  # , der=0)
            # ax[0].plot(xfit_r, yfit_r, color="orange", linestyle="--", linewidth=0.5)

            if nplots == 0:  # annotate
                ax[0].set_xlabel("Time (msec), re Peak")
                ax[0].set_ylabel("V (mV)")
                ax[1].set_xlabel("V (mV)")
                ax[1].set_ylabel("dV/dt (mV/ms)")
                # ax[1].set_clip_on(False)
                ax[2].set_xlabel("V (mV)")
                ax[2].set_ylabel("Curvature (1/mV)")
                ax[3].set_xlabel("Time (msec)")
                ax[3].set_ylabel("Rising dV/dt (mV/ms)")
                for i in range(2):
                    PH.nice_plot(ax[i])
                    PH.talbotTicks(ax[i])
                self.spike_cursors.append(
                    AnnotatedCursor(
                        line=voltage_plot_line[0],
                        ax=ax[0],
                        numberformat="{0:.3f} {1:5.3f}",
                        mode="stick",
                        useblit=True,
                        color="skyblue",
                        linewidth=0.5,
                    )
                )

                ax[0].plot([-2, 5], [-60, -60], "m--", linewidth=0.5)
                # print("Made cursor")
                self.spike_cursors.append(
                    AnnotatedCursor(
                        line=phase_plot_line[0],
                        ax=ax[1],
                        numberformat="{0:.3f} {1:5.3f}",
                        useblit=True,
                        mode="stick",
                        color="skyblue",
                        linewidth=0.5,
                    )
                )
                ax[1].plot([-60, 20], [12, 12], "m--", linewidth=0.5)

            nplots += 1

        if nplots > 0:
            mpl.show()
        return cell_df

    def report_data(self, pos):
        thr = self.cell_df["Spikes"][self.match_protocol]["LowestCurrentSpike"]["AP_begin_V"]
        msg = f"{self.cell_df.cell_id:s},{self.cell_df.cell_type:s},{self.cell_df.age!s},{thr:.3f},{pos[0]:.2f},{pos[1]:.3f}"
        print(msg)
        self.textappend(msg, color="w")

    def get_selected_cell_data_spikes(
        self, experiment, table_manager, assembleddata, bspline_s, dock, window
    ):
        """get_selected_cell_data_spikes  plot cell data spikes into the pg window; avoid MPL.

        Parameters
        ----------
        experiment : _type_
            _description_
        table_manager : _type_
            _description_
        assembleddata : _type_
            _description_
        bspline_s : _type_
            _description_

        Returns
        -------
        _type_
            _description_
        """
        self.get_row_selection(table_manager)
        if self.selected_index_rows is None:
            return None
        N = len(self.selected_index_rows)
        print("N: ", N)
        colors = colormaps.sinebow_dark.discrete(N)
        for nplots, index_row in enumerate(self.selected_index_rows):
            selected = table_manager.get_table_data(index_row)
            day = selected.date[:-4]
            slicecell = selected.cell_id[-4:]
            pcolor = colors[nplots].colors[0]
            cell_df, cell_df_tmp = filename_tools.get_cell(
                experiment, assembleddata, cell_id=selected.cell_id
            )
            self.cell_df = cell_df
            print("cell id: ", selected.cell_id)
            self.current_selection = selected
            print("cell_df: ", cell_df["date"], cell_df["slice_slice"], cell_df["cell_id"])
            protocols = list(cell_df["Spikes"].keys())
            protocols = self.remove_excluded_protocols(
                experiment, cell_id=cell_df["cell_id"], protocols=protocols
            )

            min_index = None
            min_current = 1
            trace = None
            min_protocol = None
            for ip, protocol in enumerate(protocols):
                # print('\nprotocol: ',protocol, ' lowest current spike: ', cell_df["Spikes"][protocol]["LowestCurrentSpike"])
                if cell_df["Spikes"][protocol]["LowestCurrentSpike"] is None or len(cell_df["Spikes"][protocol]["LowestCurrentSpike"]) == 0:
                    continue
                print(cell_df["Spikes"][protocol]["LowestCurrentSpike"])
                lcs_trace = cell_df["Spikes"][protocol]["LowestCurrentSpike"]["trace"]
                min_current_index, current, trace = self.find_lowest_current_trace(
                    cell_df["Spikes"][protocol]
                )
                if trace != lcs_trace:
                    print("Mismatch: ", protocol, trace, lcs_trace)
                elif current < min_current:
                    min_trace = trace
                    min_index = min_current_index
                    min_protocol = protocol
                    min_current = current
                    print("match, protocol", protocol)
            pp = PrettyPrinter(indent=4)

            if min_protocol is None:
                CP("r", f"No protocol found with a small current spike!: {protocols!s} ")
                continue
            try:
                # print("min protocol: ", min_protocol)
                # print("min protocol keys: ", cell_df["Spikes"][min_protocol].keys())
                # print("min trace #: ", min_trace)
                low_spike = cell_df["Spikes"][min_protocol]["spikes"][min_trace][0]
            except KeyError:
                CP("Failed to get min spike")
                print("min protocol spikes: ", cell_df["Spikes"][min_protocol].keys())
                print("V: ", min_trace)
                print("Min index: ", min_index)
                print(
                    "len spikes in min current trace: ", cell_df["Spikes"][min_protocol][min_index]
                )
                return None
            self.match_protocol = protocol
            # print("min index: ", min_index, "trace: ", V)
            # print(low_spike)
            if nplots == 0:
                dock.raiseDock()
                title = f"{day:s} {slicecell:s} {Path(protocol).name:s} trace: {low_spike.trace:d} {1e9*low_spike.current:.3f} nA"
                # print(dir(dock))
                window.setWindowTitle(title)
                dock.setTitle("Spike Analysis")
                dock.layout.setContentsMargins(50, 50, 50, 50)
                dock.layout.setSpacing(50)
                # dock.setTitle(title)
                self.P1 = pg.PlotWidget(title=f"{title:s}\nAP Voltage")
                self.P2 = pg.PlotWidget(title="AP Phase Plot")
                self.P3 = pg.PlotWidget(title="AP Curvature")
                self.P4 = pg.PlotWidget(title="AP Rising dV/dt")
                ax: list = []
                self.spike_cursors: list = []

                dock.addWidget(self.P1, 1, 0)  # row=0, column=0)
                dock.addWidget(self.P2, 1, 1)  # row=0, column=1)
                dock.addWidget(self.P3, 2, 1)  # row=1, column=0)
                dock.addWidget(self.P4, 2, 0)  # row=1, column=1)
                dock.setAutoFillBackground(True)
                ax = [self.P1, self.P2, self.P3, self.P4]

                self.spike_plots_ax = ax
                # Text location in data coords
                # props = dict(boxstyle="round", facecolor="wheat", alpha=0.4)
                # self.txt1 = ax[1].text(0, 0, "", fontsize=8, bbox=props)
                # self.txt0 = ax[0].text(0, 0, "", fontsize=8, bbox=props)

            vtime = (low_spike.Vtime - low_spike.peak_T) * 1e3
            voltage_plot_line = ax[0].plot(
                vtime, low_spike.V * 1e3, pen=pg.mkColor(pcolor), linewidth=2
            )
            cursor1 = AnnotatedCursorPG(line=voltage_plot_line, ax=ax[0], mode="stick")
            self.spike_cursors.append(cursor1)
            phase_plot_line = ax[1].plot(
                low_spike.V[: low_spike.dvdt.shape[0]] * 1e3, low_spike.dvdt, pen=pg.mkColor(pcolor)
            )
            cursor2 = AnnotatedCursorPG(line=phase_plot_line, ax=ax[1], mode="stick")
            self.spike_cursors.append(cursor2)
            dvdt_ticks = np.arange(-4, 2.01, 0.1)
            t_indices = np.array([np.abs(vtime - point).argmin() for point in dvdt_ticks])
            thr_index = np.abs(vtime - (low_spike.AP_latency - low_spike.peak_T) * 1e3).argmin()
            thr_t = vtime[thr_index]
            peak_index = np.abs(vtime - 0).argmin()
            peak_t = vtime[peak_index]

            # indicate the threshold on the phase plot
            scatter0 = pg.PlotDataItem(
                [low_spike.V[thr_index] * 1e3],
                [low_spike.dvdt[thr_index]],
                symbolBrush=pg.mkBrush("r"),
                symbolSize=6,
                symbol="o",
            )

            ax[1].addItem(scatter0)
            maxdvdt = np.argmax(low_spike.dvdt) + 1
            rising_phase_indices = np.nonzero(
                (low_spike.Vtime >= low_spike.Vtime[thr_index])
                & (low_spike.Vtime <= low_spike.Vtime[maxdvdt])  # [peak_index])
            )[0]
            falling_phase_indices = np.nonzero(
                (low_spike.Vtime >= low_spike.Vtime[peak_index])
                & (low_spike.V >= low_spike.V[thr_index])
            )[0]
            ahp_phase_indices = np.nonzero(
                (low_spike.Vtime >= low_spike.Vtime[peak_index])
                & (low_spike.V < low_spike.V[thr_index])
            )[0]

            # indicate the AHP nadir on the phase plot
            ahp_index = np.abs(vtime - (low_spike.trough_T - low_spike.peak_T) * 1e3).argmin()
            scatter1 = pg.PlotDataItem(
                [low_spike.V[ahp_index] * 1e3],
                [low_spike.dvdt[ahp_index]],
                symbolBrush=pg.mkBrush("orange"),
                symbolSize=9,
                symbol="o",
            )
            ax[1].addItem(scatter1)
            # indicate the AHP on the voltage trace
            spike_pk_to_trough_T = low_spike.trough_T - low_spike.peak_T
            AHP_t = spike_pk_to_trough_T * 1e3  # in msec, time from peak
            latency = (low_spike.AP_latency - low_spike.peak_T) * 1e3  # in msec
            scatter2 = pg.PlotDataItem(
                [AHP_t],
                [low_spike.trough_V * 1e3],
                symbolBrush=pg.mkBrush("orange"),
                symbolSize=9,
                symbol="o",
            )
            ax[0].addItem(scatter2)
            x_line = [latency, AHP_t]
            # draw dashed line at threshold
            thrline = pg.PlotDataItem(
                x_line,
                low_spike.V[thr_index] * np.ones(2) * 1e3,
                pen=pg.mkPen("k", width=1, style=QtCore.Qt.PenStyle.DashLine),
            )
            # draw dashed line at nadir of AHP
            nadirahpline = pg.PlotDataItem(
                x_line,
                low_spike.V[ahp_index] * np.ones(2) * 1e3,
                pen=pg.mkPen("m", width=1, style=QtCore.Qt.PenStyle.DashLine),
            )
            ax[0].addItem(thrline)
            ax[0].addItem(nadirahpline)
            # indicate the threshold on the voltage plot
            thrpoint = pg.PlotDataItem(
                [latency],
                [low_spike.AP_begin_V * 1e3],
                symbolBrush=pg.mkBrush("r"),
                symbol="o",
                symbolSize=9,
            )
            ax[0].addItem(thrpoint)
            hwline = pg.PlotDataItem(
                [
                    (low_spike.left_halfwidth_T - low_spike.peak_T - 0.0001) * 1e3,
                    (low_spike.right_halfwidth_T - low_spike.peak_T + 0.0001) * 1e3,
                ],
                [  # in msec
                    low_spike.halfwidth_V * 1e3,
                    low_spike.halfwidth_V * 1e3,
                ],
                pen=pg.mkPen("g", width=0.5),
            )
            ax[0].addItem(hwline)
            # ax[0].plot(
            #     (low_spike.right_halfwidth_T - low_spike.peak_T)
            #     * 1e3,  # in msec
            #     low_spike.halfwidth_V * 1e3,
            #     "co",
            # )
            savgol = True
            bspline = False
            if bspline:  # fit with a bspline
                # rising phase of AP from threshold to peak dvdt
                tck_r, xfit_r, yfit_r, kappa_r = self.spline_fits(
                    vtime, low_spike.V, rising_phase_indices[:-2], k=5, s=bspline_s, label="rising"
                )
                # ax[0].plot(vtime[rising_phase_indices]-peak_t, low_spike.V[rising_phase_indices]*1e3,
                #            color="b", linestyle="--", linewidth=2)

                # falling phase of AP from peak to spike threshold
                tck_f, xfit_f, yfit_f, kappa_f = self.spline_fits(
                    vtime, low_spike.V, falling_phase_indices, label="falling"
                )
                # ahp phase of AP from threshold time on out
                tck_a, xfit_a, yfit_a, kappa_a = self.spline_fits(
                    vtime, low_spike.V, ahp_phase_indices, k=5, s=bspline_s, label="ahp"
                )
            # print("len rising phase indices: ", len(rising_phase_indices))
            fits_ok = True
            if savgol:
                from scipy.signal import savgol_filter

                window_length = int(bspline_s)
                if window_length > 5:
                    polyorder = 5
                else:
                    polyorder = 3
                    window_length = 5
                try:
                    yfit_r = savgol_filter(
                        low_spike.V[rising_phase_indices],
                        window_length=window_length,
                        polyorder=polyorder,
                    )
                    yfit_f = savgol_filter(
                        low_spike.V[falling_phase_indices],
                        window_length=window_length,
                        polyorder=polyorder,
                    )
                    yfit_a = savgol_filter(
                        low_spike.V[ahp_phase_indices],
                        window_length=window_length,
                        polyorder=polyorder,
                    )
                except:
                    yfit_r = np.zeros(len(rising_phase_indices))
                    yfit_f = np.zeros(len(falling_phase_indices))
                    yfit_a = np.zeros(len(ahp_phase_indices))
                    fits_ok = False
                xfit_r = vtime[rising_phase_indices]
                xfit_f = vtime[falling_phase_indices]
                xfit_a = vtime[ahp_phase_indices]
                try:
                    kappa_r = self.curvature(xfit_r, yfit_r)
                    kappa_f = self.curvature(xfit_f, yfit_f)
                    kappa_a = self.curvature(xfit_a, yfit_a)
                except:
                    kappa_r = np.zeros(len(rising_phase_indices))
                    kappa_f = np.zeros(len(falling_phase_indices))
                    kappa_a = np.zeros(len(ahp_phase_indices))
                    fits_ok = False

            if fits_ok:
                fit_plot = pg.PlotDataItem(
                    xfit_r,
                    1e3 * np.gradient(yfit_r, xfit_r),
                    pen=pg.mkPen("w", width=1, style=QtCore.Qt.PenStyle.SolidLine),
                )
                ax[3].addItem(fit_plot)
                cursor4 = AnnotatedCursorPG(line=fit_plot, ax=ax[3], mode="stick")
                self.spike_cursors.append(cursor4)
            if (bspline or savgol) and fits_ok:
                vfitplot = pg.PlotDataItem(
                    xfit_f,
                    yfit_f * 1e3,
                    pen=pg.mkPen("c", width=1, style=QtCore.Qt.PenStyle.DashLine),
                )

                dvdtfitplot = pg.PlotDataItem(
                    yfit_f[:-1] * 1e3,
                    np.diff(yfit_f * 1e3) / np.diff(xfit_f),
                    pen=pg.mkPen("c", width=1, style=QtCore.Qt.PenStyle.DashLine),
                )
                dvdt_thr_line = pg.PlotDataItem(
                    [-60.0, 20.0],
                    [12.0, 12.0],
                    pen=pg.mkPen(color="r", width=1.0),  #  style=QtCore.Qt.PenStyle.DashLine)
                )
                ax[0].addItem(vfitplot)
                ax[1].addItem(dvdtfitplot)
                ax[1].addItem(dvdt_thr_line)

                repol_fitplot = pg.PlotDataItem(
                    xfit_r,
                    yfit_r * 1e3,
                    pen=pg.mkPen("y", width=1.5, style=QtCore.Qt.PenStyle.DashLine),
                )

                dvdt_repol_fitploat = pg.PlotDataItem(
                    yfit_r[:-1] * 1e3,
                    np.diff(yfit_r * 1e3) / (np.diff(xfit_r)),
                    pen=pg.mkPen("orange", width=0.3, style=QtCore.Qt.PenStyle.DashLine),
                )
                ax[1].addItem(repol_fitplot)
                ax[1].addItem(dvdt_repol_fitploat)

                ahp_plot = pg.PlotDataItem(
                    xfit_a,
                    yfit_a * 1e3,
                    pen=pg.mkPen("m", width=1.0, style=QtCore.Qt.PenStyle.DashLine),
                )
                dvdt_ahp_plot = pg.PlotDataItem(
                    yfit_a[:-1] * 1e3,
                    np.diff(yfit_a * 1e3) / np.diff(xfit_a),
                    pen=pg.mkPen("m", width=1.0, style=QtCore.Qt.PenStyle.DashLine),
                )
                ax[0].addItem(ahp_plot)
                ax[1].addItem(dvdt_ahp_plot)

                kappa_r_plot = pg.PlotDataItem(
                    yfit_r[1:] * 1e3,
                    kappa_r[1:],
                    pen=pg.mkPen(pcolor, width=1.0, style=QtCore.Qt.PenStyle.DashLine),
                )

                kappa_a_plot = pg.PlotDataItem(
                    yfit_a[1:] * 1e3,
                    kappa_a[1:],
                    pen=pg.mkPen("m", width=0.3, style=QtCore.Qt.PenStyle.DashLine),
                )

                kappa_f_plot = pg.PlotDataItem(
                    yfit_f[1:] * 1e3,
                    kappa_f[1:],
                    pen=pg.mkPen("c", width=0.3, style=QtCore.Qt.PenStyle.DashLine),
                )

                # horizontal line at 0 curvature
                zcurve_plot = pg.PlotDataItem(
                    [-60, 20],
                    [0, 0],
                    pen=pg.mkPen("k", width=0.3, style=QtCore.Qt.PenStyle.DashLine),
                )
                ax[2].addItem(kappa_r_plot)
                ax[2].addItem(kappa_a_plot)
                ax[2].addItem(kappa_f_plot)
                ax[2].addItem(zcurve_plot)

                cursor3 = AnnotatedCursorPG(
                    line=kappa_r_plot, ax=ax[2], mode="stick", report_func=self.report_data
                )
                cursor3.set_tracker_from(cursor2)

                self.spike_cursors.append(cursor3)
                # m = int(len(low_spike.V[rising_phase_indices]) / 4)
                # if m > len(low_spike.V[rising_phase_indices]) / 4:
                #     m = int(len(low_spike.V[rising_phase_indices]) / 2)
                # print(f"Len ahp: {len(low_spike.V[rising_phase_indices]):d}, m: {m:d}")
                # if m == 0:
                #     break
                # # interpolation in V and dvdt to get a uniform distribution of points in V space
                # # yv = np.interp(xv, low_spike.V[rising_phase_indices], low_spike.dvdt[rising_phase_indices])
                # tck_p = splrep(
                #     low_spike.V[rising_phase_indices],
                #     low_spike.dvdt[rising_phase_indices],
                #     k=5,
                #     s=64,
                # )
                # print("xv: ", xv)
                # print("tck: ", tck_p)
                # yfit_p = splev(xv, tck_p)  # , der=0)
            # ax[0].plot(xfit_r, yfit_r, color="orange", linestyle="--", linewidth=0.5)

            if nplots == 0:  # annotate
                ax[0].setLabel("bottom", "Time (msec), re Peak")
                ax[0].setLabel("left", "V (mV)")
                ax[1].setLabel("bottom", "V (mV)")
                ax[1].setLabel("left", "dV/dt (mV/ms)")
                # ax[1].set_clip_on(False)
                ax[2].setLabel("bottom", "V (mV)")
                ax[2].setLabel("left", "Curvature (1/mV)")
                ax[3].setLabel("bottom", "Time (msec)")
                ax[3].setLabel("left", "Rising dV/dt (mV/ms)")

            nplots += 1

        return cell_df

    def mouse_move(self, event):
        """
        Display text in box in response to mouse movement
        """
        print("mouse: event")
        if not event.inaxes:
            return
        x, y = event.xdata, event.ydata
        self.txt1.set_text("Test\n x=%1.2f, y=%1.2f" % (x, y))

    def categorize_ages(self, row, experiment):
        age = numeric_age(row)
        for k in experiment["age_categories"].keys():
            if (
                age >= experiment["age_categories"][k][0]
                and age <= experiment["age_categories"][k][1]
            ):
                age_category = k
        return age_category

    def get_selected_cell_data_FI(
        self,
        experiment: pd.DataFrame,
        assembleddata: pd.DataFrame,
    ):
        """get_selected_cell_data_FI Analyze and Plot FI data for selected cells.
        If the configuration file includes dropouts, then the dropouts are plotted.

        Parameters
        ----------
        experiment : pd.DataFrame
            experiment configuration file
        assembleddata : pd.DataFrame
            assembled IV data file for this experiment

        Returns
        -------
        figure information
            plothelpers Plotter object with figure information.
        """
        print("got row selection")
        print("# selected: ", len(assembleddata))
        if len(assembleddata) == 0:
            return None

        import matplotlib.pyplot as mpl

        P = PH.regular_grid(
            3,
            2,
            margins={
                "bottommargin": 0.08,
                "leftmargin": 0.1,
                "rightmargin": 0.05,
                "topmargin": 0.1,
            },
            figsize=(10, 12),
        )
        fig = P.figure_handle
        ax = P.axarr
        dropout_plot = False
        iplot = 0
        firing_failure_calculation = False
        N = len(assembleddata)
        if "firing_failure_analysis" in experiment.keys():
            if experiment["firing_failure_analysis"]["compute"]:
                firing_failure_calculation = True
                dropout_plot = False
                dropout_test_current = experiment["firing_failure_analysis"]["test_current"]
                max_failure_test_current = experiment["firing_failure_analysis"]["max_current"]
                dropout_test_time = experiment["firing_failure_analysis"]["test_time"]

            if "Max_FI_I" in experiment.keys():
                max_I = experiment["Max_FI_I"]
            fcol = {"+/+": "green", "-/-": "orange"}
            max_fr = {"+/+": [], "-/-": []}
            dropout = np.zeros((2, N))
            cellids = ["None"] * N
            expression = ["None"] * N
            genotype = ["None"] * N
            # cmap = mpl.cm.get_cmap("tab20", N)
            # colors = [matplotlib.colors.to_hex(cmap(i)) for i in range(N)]
            colors = colormaps.sinebow_dark.discrete(N)
            xpalette = sns.xkcd_palette(
                ["teal", "crimson", "light violet", "faded green", "dusty purple"]
            )
            nplots = 0
            expression_list = []
            age_list = []
            # arrays for summary histograms
            #
            drop_out_currents = pd.DataFrame(
                columns=["cell", "genotype", "sex", "expression", "current", "time", "group"]
            )  # current at which cell drops out by the selected time, dict by genotype, expression

            pos_expression = ["+", "GFP+", "EYFP", "EYFP+"]
            neg_expression = ["-", "GFP-", "EYFP-", "EGFP-"]

            # One analysis is this:
            # for each cell class,
            #   for each cell, for each level, plot the latency of the LAST spike in the train
            LS = dict.fromkeys(list(experiment["group_map"].keys()), {})
            n1 = dict.fromkeys(list(LS.keys()), [])
            # {
            #     "+/+": {},
            #     "-/-": {},   # dict keys are current level, lastspikes are arrays of spikes
            # }  # last spike dict: keys are current level, lastspikes are arrays of spikes .
            # LS[I]{genotype1: dict of {I: , lastspikesarray of latencies, genotype2: array of latencies}
            Ns: dict = {}

            for ells in LS.keys():
                Ns[ells] = []  # {"+/+": [], "-/-": []}  # track cells used.
            for selected in assembleddata.itertuples():
                pcolor = colors[nplots].colors
                grp = selected.Group
                day = selected.date[:-4]
                slicecell = selected.cell_id[-4:]
                # we need to reach into the main data set to get the expression,
                # as it is not carried into the assembled data.
                cell_df, _ = filename_tools.get_cell(
                    experiment, assembleddata, cell_id=selected.cell_id
                )
                if "age_category" not in cell_df.keys():
                    cell_df["age_category"] = np.nan
                cell_df["age_category"] = self.categorize_ages(cell_df, experiment=experiment)
                if cell_df.cell_expression is None:
                    continue
                if cell_df.cell_expression not in expression_list:
                    expression_list.append(cell_df.cell_expression)

                prots = list(cell_df["IV"].keys())
                # IVdata = cell_df["IV"][prots[0]].keys()

                # for this cell, build spike arrays for all spike
                # latencies at each current level.
                splat = []
                spinj = []
                pused = []

                # print("Experiment FI protocols: ", experiment["raster_protocols"])
                rpnames = [r for r in experiment["raster_protocols"]]
                pulse_start = {}
                for prot in prots:
                    if cell_df["Spikes"][prot] is None:
                        continue
                    protname = str(Path(prot).name)
                    if protname[:-4] not in rpnames:
                        continue
                    if "4nA" in protname:
                        print("skipping: ", protname)
                        continue
                    pused.append(prot)
                    pulse_start = experiment["Protocol_start_times"][protname[:-4]]

                    for tr in cell_df["Spikes"][prot]["spikes"].keys():
                        for nsp in cell_df["Spikes"][prot]["spikes"][tr].keys():
                            lat = cell_df["Spikes"][prot]["spikes"][tr][nsp].AP_latency
                            if lat is not None and lat > pulse_start + 0.0005:
                                splat.append(lat - pulse_start)
                                spinj.append(cell_df["Spikes"][prot]["spikes"][tr][0].current)
                # print("prots: used ", pused)
                # print(splat)
                # print(spinj)
                spinj = np.array(spinj) * 1e9  # currents in nA
                splat = np.array(splat)  # latencies in seconds
                currents = np.unique(spinj)
                if len(currents) == 0:  # be sure there is something to be had..
                    continue

                print("currents in/max: ", np.min(currents), np.max(currents))
                firing_rate_mean = np.zeros(len(currents))
                firing_rate_fl = np.zeros(len(currents))  # first to last spike
                last_spike = np.zeros(len(currents)) * np.nan
                last_spike_current = np.zeros(len(currents)) * np.nan
                for i, current in enumerate(currents):
                    if current > 1.0:
                        continue
                    # get latencies at this current level
                    latency_at_I = [splat[j] for j in range(len(spinj)) if spinj[j] == current]
                    current_at_I = [spinj[j] for j in range(len(spinj)) if spinj[j] == current]
                    if len(latency_at_I) <= 1:  # always more than one spike
                        continue
                    i_arg = np.argmax(latency_at_I)
                    last_spike[i] = latency_at_I[
                        i_arg
                    ]  # latency of the LAST spike in the train at this current
                    last_spike_current[i] = current_at_I[i_arg]  # current for this last spike
                    firing_rate_mean[i] = len(latency_at_I)  # 1 is pulse duration in sec
                    firing_rate_fl[i] = len(latency_at_I) / ((last_spike[i] - np.min(latency_at_I)))
                    ucurr = np.around(current, 2)

                    print(grp)
                    if isinstance(grp, float) and np.isnan(grp):
                        continue
                    if grp is None or (isinstance(grp, float) and np.isnan(grp)):
                        grp = cell_df["age_category"]
                    if ucurr not in LS[grp].keys():
                        LS[grp][ucurr] = [last_spike[i]]
                    else:
                        LS[grp][ucurr].append(last_spike[i])

                # designate symbols:
                # s for +/+ genotype
                # o for -/- genotype
                # filled for GFP+ or EYFP+
                # open for GFP- or EYFP-

                if isinstance(grp, float) and np.isnan(grp):
                    continue
                Ns[grp].append(selected.cell_id)
                gcolor = "k"
                symbol = "o"
                fillstyle = "full"
                facecolor = "lightgrey"

                if grp in ["+/+", "-/-", "+/-"]:
                    match grp:
                        case "+/+":
                            symbol = "s"
                            gcolor = "blue"
                        case "-/-":
                            symbol = "o"
                            gcolor = "salmon"
                        case "+/-" | "-/+":
                            symbol = "D"
                            gcolor = "grey"
                        case _:
                            symbol = "X"
                    match cell_df.cell_expression:
                        case "+" | "GFP+" | "EYFP" | "EYFP+":
                            fillstyle = "full"
                            facecolor = gcolor
                        case "-" | "GFP-" | "EYFP-" | "EGFP-":
                            fillstyle = "full"
                            facecolor = "white"
                        case _:
                            fillstyle = "full"
                            facecolor = "lightgrey"
                elif grp in ["Preweaning", "Pubescent", "Young Adult", "Old Adult"]:
                    facecolor = experiment["plot_colors"][grp]
                    symbol = "o"
                    fillstyle = "full"
                    # facecolor = "lightgrey"
                # print("Fill style, symbol, facecolor: ", fillstyle, symbol, facecolor)
                try:
                    fit = selected.fit[0][0]
                except:
                    print("No fit? : ")
                    print("     fit ", selected.fit)
                ax[0, 0].plot(
                    currents,
                    firing_rate_mean,
                    marker=symbol,
                    linestyle="-",
                    markersize=4,
                    fillstyle=fillstyle,
                    markerfacecolor=facecolor,
                    color=pcolor,
                )
                if len(selected.fit[0]) > 0:
                    fit = selected.fit[0][0]
                    ax[0, 0].plot(np.array(fit[0][0]) * 1e9, fit[1][0], "b--")
                # try:
                #     max_fr[selected.Group].append(np.nanmax(datadict["FI_Curve1"][1]))
                # except:
                #     print("FI curve: ", datadict["FI_Curve1"])
                #     print("cell, group: ", selected.cell_id, selected.Group)
                #     print(selected)
                ax[0, 0].set_title("0,0")
                if max_failure_test_current > 1000:  # add the 4 nA data
                    ax[0, 0].plot(
                        np.array(selected.FI_Curve4[0]) * 1e9,
                        np.array(selected.FI_Curve4[1]),
                        marker=symbol,
                        linestyle="-",
                        markersize=4,
                        fillstyle=fillstyle,
                        markerfacecolor=facecolor,
                        color=pcolor,
                    )
                # print(selected.firing_currents)
                if len(splat) == 0:
                    # print(selected)
                    continue  # raise ValueError("No firing currents")
                # firing rate (time to last spike basis)
                ax[0, 1].set_title("x=current, y=[f-l]rates, 0, 1")
                ax[0, 1].plot(
                    currents,
                    firing_rate_fl,
                    marker=symbol,
                    linestyle="-",
                    markersize=4,
                    fillstyle=fillstyle,
                    markerfacecolor=facecolor,
                    color=pcolor,
                )
                ax[1, 0].set_title("x=last spike time, y = current, 1 ,0")
                ax[1, 0].plot(
                    currents,
                    last_spike,
                    marker=symbol,
                    linestyle="-",
                    markersize=4,
                    fillstyle=fillstyle,
                    markerfacecolor=facecolor,
                    color=pcolor,
                )
                # plot maximal firing rate

                # compute "dropout" time to the last spike in a train.
                # if the FI curve is non-monotonic, this will be the time of the
                # last spike that occurs in a train.

                if len(last_spike) > 0 and firing_failure_calculation:
                    # find the largest current where cell fires that is above the dropuout threshold
                    # print("datadict last spikes: ", selected.last_spikes)
                    print("dropuout threshold, max current: ", dropout_test_current, np.max(currents))
                    # get current levels >= test current
                    do_pts_I = np.nonzero(np.array(currents >= dropout_test_current))
                    do_pts_T = np.array(last_spike > dropout_test_time)
                    print("do_pts: ", do_pts_I)
                    print("currents[do_pts: ", currents[do_pts_I])
                    if len(do_pts_I) > 0 and not np.all(np.isnan(currents[do_pts_I])):
                        idrop = np.nanargmax(currents[do_pts_I])# + do_pts[0]
                        dropout[1, nplots] = currents[idrop]
                        dropout[0, nplots] = last_spike[idrop]
                    else: # not points in current range, or no spikes above the threshold current
                        dropout[1, nplots] = currents[0]
                        dropout[0, nplots] = np.nan
                    print(
                        "cellexpression: ",
                        cell_df.cell_expression,
                        pos_expression,
                        neg_expression,
                    )
                    if (
                        cell_df.cell_expression in pos_expression
                        or cell_df.cell_expression in neg_expression
                    ):
                        print("Dropout, lastspikes: adding to table", cell_df.cell_expression)
                        do_curr = pd.DataFrame(
                            {
                                "cell": [Path(selected.cell_id).name],
                                "genotype": [selected.Group],
                                "sex": [cell_df.sex],
                                "expression": [cell_df.cell_expression],
                                "current": [dropout[1, nplots]],
                                "time": [dropout[0, nplots]],
                            }
                        )
                        drop_out_currents = pd.concat(
                            [drop_out_currents, do_curr], ignore_index=True
                        )

                    cellids[nplots] = selected.cell_id
                    expression[nplots] = cell_df.cell_expression
                    genotype[nplots] = selected.Group
                    print(
                        "Dropout info: ",
                        selected.cell_id,
                        dropout[0, nplots],
                        dropout[1, nplots],
                    )
                    ax[1, 1].set_title("1,1")
                    # ax[1, 1].plot(
                    #     last_spike, # [dropout_test_current, dropout_test_current],
                    #     last_spike_current, # [0, max_failure_test_current*1e-3],
                    #     color="gray",
                    #     linestyle="--",
                    #     linewidth=0.33,
                    # )
                    ax[1, 1].plot(
                        dropout[1, nplots],  # currents are 1, on x axis
                        dropout[0, nplots],  # time of last spike is 0, on y axis
                        marker=symbol,
                        linestyle="-",
                        markersize=4,
                        fillstyle=fillstyle,
                        markerfacecolor=facecolor,
                        color=gcolor, # pcolor,
                        clip_on=False,
                    )
                    # ax[1, 1].text(
                    #     x=dropout[1, nplots],
                    #     y=dropout[0, nplots],
                    #     s=str(Path(selected.cell_id).name),
                    #     fontsize=6,
                    #     horizontalalignment="right",
                    #     color=fcol[cell_df.genotype],
                    #     )
                # for g in LS.keys(): # genotypes:

                iplot += 1
                nplots += 1

            for grp in list(LS.keys()):
                for i, current in enumerate(LS[grp].keys()):
                    # print("current: ", current)
                    icurr = current
                    if icurr < 0.2:
                        continue
                    h1, b1 = np.histogram(
                        np.array(LS[grp][current]),
                        bins=np.linspace(0, 1, 20, endpoint=True),
                    )
                    # h2, b2 = np.histogram(
                    #     np.array(LS[grp][current]),
                    #     bins=np.linspace(0, 1, 20, endpoint=True),
                    # )
                    n1[grp].append([icurr, np.sum(h1), h1[-1], h1[0]])
                    # n2.append([icurr, np.sum(h2), h2[-1], h1[0]])
                    # print(np.array(LS["+/+"][current]))
                    # print(h1)
                    yd = 0.035
                    pmode = 1
                    color = experiment["plot_colors"][grp]
                    if np.sum(h1) > 0:
                        if pmode == 0:
                            ax[1, 1].bar(
                                b1[:-1] - 0.05,
                                yd * h1 / np.sum(h1),
                                width=0.05,
                                color=color,
                                alpha=0.7,
                                bottom=current,
                            )
                        if pmode == 1:
                            ax[1, 1].stairs(
                                edges=b1,
                                values=yd * h1 / np.sum(h1) + icurr,
                                color=color,
                                alpha=0.7,
                                baseline=icurr,
                                fill=None,
                            )
                    # if np.sum(h2) > 0:
                    #     if pmode == 0:
                    #         ax[1, 1].bar(b2[:-1] + 0.05, yd*h2/np.sum(h2), width=0.05, color="salmon", alpha=0.7, bottom=current)
                    #     if pmode == 1:
                    #         ax[1,1].stairs(edges=b2+0.05, values=yd*h2/np.sum(h2)+icurr, color="salmon", alpha=0.7, baseline=icurr, fill="salmon")

                    #         ax[1,1].plot([0, 1], [icurr, icurr], '--', linewidth=0.33, color="gray")
                    # n1[grp] = np.array(n1[grp])#.reshape(-1, 4)
                    # n2 = np.array(n2)#.reshape(-1, 4)
                    # print(n1[:,0])
                    ax[2, 0].plot(
                        np.array(n1[grp])[:, 0],
                        np.array(n1[grp])[:, 2] / np.array(n1[grp])[:, 1],
                        color=experiment["plot_colors"][grp],
                        marker="o",
                        linestyle="-",
                        markersize=4,
                        fillstyle=fillstyle,
                        markerfacecolor=experiment["plot_colors"][grp],
                        clip_on=False,
                    )
                # ax[2, 0].plot(n2[:,0], n2[:, 2]/n2[:,1], "ro-")
                ax[2, 0].set_ylim(0, 1)
                ax[2, 0].set_xlim(0, 1)
                ax[1, 1].set_ylim(0.4, 1.05)

        # plot some summary stuff
        # sns.jointplot(
        #     x="time",
        #     y="current",
        #     hue="expression",
        #     data=drop_out_currents,
        #     palette="tab10",
        #     ax=ax[2, 0],
        # )
        # sns.jointplot(
        #     x="time",
        #     y="current",
        #     hue="genotype",
        #     data=drop_out_currents,
        #     palette=xpalette,
        #     ax=ax[2, 1],
        # )
        print("Dropout currents: ", drop_out_currents.columns)
        print("Dropout current table: ", drop_out_currents.head(100))
        if not drop_out_currents.empty:
            sns.ecdfplot(
                data=drop_out_currents, x="current", hue="genotype", palette=xpalette, ax=ax[2, 0]
            )
            sns.ecdfplot(
                data=drop_out_currents, x="current", hue="expression", palette="tab10", ax=ax[2, 1]
            )

        print("Dropout current levels: ")
        print("-" * 20)
        # print(cell_df.keys())
        for g in Ns.keys():
            print(f"Genotype: {g:s}, N={len(Ns[g]):d}")
        print("N, cell, expression, genotype, idrop")
        drop_out_currents.to_csv("dropout.csv")
        for i in np.argsort(dropout[1]):
            print(i, ", ", cellids[i], ", ", expression[i], ", ", genotype[i], ", ", dropout[1, i])
        print("-" * 20)
        if dropout_plot:
            ax[0, 0].set_xlabel("Current (nA)")
            ax[0, 0].set_ylabel("Firing Rate (mean, Hz) (1 sec)")

            ax[0, 1].set_ylabel("Firing rate (1st to last) (Hz)")
            ax[0, 1].set_xlabel("Current(nA)")

            ax[1, 0].set_xlabel("Current (pA)")
            ax[1, 0].set_ylabel("Time to last spike)")

            ax[1, 1].set_ylabel("Current (nA)")
            test_i = experiment["firing_failure_analysis"]["test_current"]
            ax[1, 1].set_xlabel(f"Time of last spike (sec)")
            # ax[1, 1].set_xlim(0, 1)
            # ax[1, 1].set_ylim(0, max_failure_test_current * 1e-3)

            exprs = (", ").join(list(set(expression)))
            genotypes = (", ").join(list(set(genotype)))
            fig.suptitle(f"Firing analysis for {exprs:s} and {genotypes:s}")
            mpl.show()
        return P

    def average_FI(self, FI_Data_I_, FI_Data_FR_, max_current: float = 1.0e-9):
        if len(FI_Data_I_) > 0:
            try:
                FI_Data_I, FI_Data_FR = zip(*sorted(zip(FI_Data_I_, FI_Data_FR_)))
            except:
                raise ValueError("couldn't zip the data sets: ")
            if len(FI_Data_I) > 0:  # has data...
                # print("averaging FI data")
                # print("   FI_Data_I : ", FI_Data_I*1e9)
                # print("   FI_Data_FR: ", FI_Data_FR)

                # raise ValueError()
                FI_Data_I_ = np.array(FI_Data_I)
                FI_Data_FR_ = np.array(FI_Data_FR)
                f1_index = np.where((FI_Data_I_ >= 0.0) & (FI_Data_I_ <= max_current))[
                    0
                ]  # limit to 1 nA, regardless
                FI_Data_I, FI_Data_FR, FI_Data_FR_Std, FI_Data_N = self.avg_group(
                    FI_Data_I_[f1_index], FI_Data_FR_[f1_index], ndim=FI_Data_I_.shape
                )
        return FI_Data_I, FI_Data_FR, FI_Data_FR_Std, FI_Data_N

    def avg_group(self, x, y, errtype: str = "std", Q: float = 0.95, ndim=2):
        if ndim == 2:
            x = np.array([a for b in x for a in b])
            y = np.array([a for b in y for a in b])
        else:
            x = np.array(x)
            y = np.array(y)
        # x = np.ravel(x) # np.array(x)
        # y = np.array(y)
        # print(x.shape, y.shape)
        xa, ind, counts = np.unique(
            x, return_index=True, return_counts=True
        )  # find unique values in x
        ya = y[ind]

        ystd = np.zeros_like(ya)
        yn = np.ones_like(ya)
        for dupe in xa[counts > 1]:  # for each duplicate value, replace with mean
            # print("dupe: ", dupe)
            # print(np.where(x==dupe), np.where(xa==dupe))
            ya[np.where(xa == dupe)] = np.nanmean(y[np.where(x == dupe)])
            if errtype == "std":
                ystd[np.where(xa == dupe)] = np.nanstd(y[np.where(x == dupe)])
            elif errtype == "sem":
                ystd[np.where(xa == dupe)] = np.nanstd(y[np.where(x == dupe)]) / np.sqrt(
                    np.count_nonzero(~np.isnan(y[np.where(x == dupe)]))
                )
            elif errtype == "CI":
                # bootstrapped maybe?
                rng = np.random.default_rng(seed=19)
                res = scipy.stats.bootstrap(
                    y[np.where(x == dupe)],
                    axis=1,
                    statistic=np.std,
                    n_resamples=10000,
                    confidence_level=Q,
                    rng=rng,
                )
                ystd[np.where(xa == dupe)] = res.confidence_interval
                # ystd[np.where(xa == dupe)] = np.nanpercentile(y[np.where(x == dupe)], q=Q/100, axis=0, method='weibull')
        return xa, ya, ystd, yn

    # get maximum slope from fit.
    def hill_deriv(self, x: float, y0: float, ymax: float, m: float, n: float):
        """hill_deriv
        analyztical solution computed from SageMath

        Parameters
        ----------
        x : float
            current
        y0 : float
            baseline
        ymax : float
            maximum y value
        m : float
            growth rate
        n : float
            growth power
        """
        hd = m * n * ymax
        hd *= np.power(m / x, n - 1)
        hd /= (x * x) * np.power((np.power(m / x, n) + 1.0), 2.0)
        return hd

    def fit_FI_Hill(
        self,
        FI_Data_I,
        FI_Data_FR,
        FI_Data_FR_Std,
        FI_Data_I_,
        FI_Data_FR_,
        FI_Data_N,
        hill_max_derivs,
        hill_i_max_derivs,
        FI_fits,
        linfits,
        cell: str,
        celltype: str,
        # plot_fits=False,
        # ax: Union[mpl.Axes, None] = None,
    ):
        plot_raw = False  # only to plot the unaveraged points.
        spanalyzer = spike_analysis.SpikeAnalysis()
        spanalyzer.fitOne(
            i_inj=FI_Data_I,
            spike_count=FI_Data_FR,
            pulse_duration=None,  # protodurs[ivname],
            info="",
            function="Hill",
            fixNonMonotonic=True,
            excludeNonMonotonic=False,
            max_current=None,
        )

        try:
            fitpars = spanalyzer.analysis_summary["FI_Growth"][0]["parameters"][0]
        except:
            CP(
                "r",
                f"fitpars has no solution? : {cell!s}, {celltype:s}, {spanalyzer.analysis_summary['FI_Growth']!s}",
            )
            return (
                hill_max_derivs,
                hill_i_max_derivs,
                FI_fits,
                linfits,
            )  # no fit, return without appending a new fit
        # raise ValueError("couldn't get fitpars: no solution?")
        y0 = fitpars[0]
        ymax = fitpars[1]
        m = fitpars[2]
        n = fitpars[3]
        xyfit = spanalyzer.analysis_summary["FI_Growth"][0]["fit"]
        i_range = np.linspace(1e-12, np.max(xyfit[0]), 1000)
        # print(f"fitpars: y0={y0:.3f}, ymax={ymax:.3f}, m={m*1e9:.3f}, n={n:.3f}")
        deriv_hill = [self.hill_deriv(x=x, y0=y0, ymax=ymax, m=m, n=n) for x in i_range]
        deriv_hill = np.array(deriv_hill) * 1e-9  # convert to sp/nA
        max_deriv = np.max(deriv_hill)
        arg_max_deriv = np.argmax(deriv_hill)
        i_max_deriv = i_range[arg_max_deriv] * (1e12)
        hill_max_derivs.append(max_deriv)
        hill_i_max_derivs.append(i_max_deriv)
        # print(f"max deriv: {max_deriv:.3f} sp/nA at {i_max_deriv:.1f} pA")
        # print(xyfit[1])
        if len(spanalyzer.analysis_summary["FI_Growth"]) > 0:
            FI_fits["fits"].append(spanalyzer.analysis_summary["FI_Growth"][0]["fit"])
            FI_fits["pars"].append(spanalyzer.analysis_summary["FI_Growth"][0]["parameters"])
        linfit = spanalyzer.getFISlope(
            i_inj=FI_Data_I,
            spike_count=FI_Data_FR,
            pulse_duration=None,  # FR is already duration
            min_current=0e-12,
            max_current=300e-12,
        )
        linfits.append(linfit)
        linx = np.arange(0, 300e-12, 10e-12)
        liny = linfit.slope * linx + linfit.intercept

        # if plot_fits:
        #     if ax is None:
        #         fig, ax = mpl.subplots(1, 1)
        #         fig.suptitle(f"{celltype:s} {cell:s}")

        #     line_FI = ax.errorbar(
        #         np.array(FI_Data_I) * 1e9,
        #         FI_Data_FR,
        #         yerr=FI_Data_FR_Std,
        #         marker="o",
        #         color="k",
        #         linestyle=None,
        #     )
        #     # ax[1].plot(FI_Data_I * 1e12, FI_Data_N, marker="s")
        #     if plot_raw:
        #         for i, d in enumerate(FI_Data_I_):  # plot the raw points before combining
        #             ax.plot(np.array(FI_Data_I_[i]) * 1e9, FI_Data_FR_[i], "x", color="k")
        #     # print("fit x * 1e9: ", spanalyzer.analysis_summary['FI_Growth'][0]['fit'][0]*1e9)
        #     # print("fit y * 1: ", spanalyzer.analysis_summary['FI_Growth'][0]['fit'][1])

        #     # ax[0].plot(linx * 1e12, liny, color="c", linestyle="dashdot")
        #     celln = Path(cell).name

        #     if len(spanalyzer.analysis_summary["FI_Growth"]) >= 0:
        #         line_fit = ax.plot(
        #             spanalyzer.analysis_summary["FI_Growth"][0]["fit"][0][0] * 1e9,
        #             spanalyzer.analysis_summary["FI_Growth"][0]["fit"][1][0],
        #             color="r",
        #             linestyle="-",
        #             zorder=100,
        #         )
        #         # derivative (in blue)
        #         line_deriv = ax.plot(
        #             i_range * 1e9, deriv_hill, color="b", linestyle="--", zorder=100
        #         )
        #         d_max = np.argmax(deriv_hill)
        #         ax2 = ax.twinx()
        #         ax2.set_ylim(0, 500)
        #         ax2.set_ylabel("Firing Rate Slope (sp/s/nA)")
        #         line_drop = ax2.plot(
        #             [i_range[d_max] * 1e9, i_range[d_max] * 1e9],
        #             [0, 1.1 * deriv_hill[d_max]],
        #             color="b",
        #             zorder=100,
        #         )
        #         ax.set_xlabel("Current (nA)")
        #         ax.set_ylabel("Firing Rate (sp/s)")
        #         # turn off top box
        #         for loc, spine in ax.spines.items():
        #             if loc in ["left", "bottom"]:
        #                 spine.set_visible(True)
        #             elif loc in ["right", "top"]:
        #                 spine.set_visible(False)
        #         for loc, spine in ax2.spines.items():
        #             if loc in ["right", "bottom"]:
        #                 spine.set_visible(True)
        #             elif loc in ["left", "top"]:
        #                 spine.set_visible(False)
        #         # spine.set_color('none')
        #         # do not draw the spine
        #         # spine.set_color('none')
        #         # do not draw the spine
        #         PH.talbotTicks(ax, density=[2.0, 2.0])
        #         PH.talbotTicks(ax2, density=[2.0, 2.0])
        #         ax.legend(
        #             [line_FI, line_fit[0], line_deriv[0], line_drop[0]],
        #             ["Firing Rate", "Hill Fit", "Derivative", "Max Derivative"],
        #             loc="best",
        #             frameon=False,
        #         )

        #     mpl.show()

        return hill_max_derivs, hill_i_max_derivs, FI_fits, linfits

    def combine_measures_and_FI_Fits(
        self,
        experiment,
        df: pd.DataFrame,
        cell: str,
        protodurs: dict = None,
    ):
        """combine_measures_and_FI_Fits
        Assemble data from multiple protocols to compute the FI curve for a cell.
        This allows us to splice together data from multiple protocols to get a complete FI curve.
        We also assemble other measures into the assembled file (input resistances, adaptation indices, etc).

        Parameters
        ----------
        experiment : _type_
            _description_
        df : pd.DataFrame
            _description_
        cell : str
            _description_
        protodurs : dict, optional
            _description_, by default None

        Returns
        -------
        _type_
            _description_

        Raises
        ------
        ValueError
            _description_
        ValueError
            _description_
        """
        if self.status_bar is not None:
            self.status_bar.showMessage(
                f"\n{'='*80:s}\nCell: {cell!s}, {df[df.cell_id == cell].cell_type.values[0]:s}  Group {df[df.cell_id == cell].Group.values[0]!s}"
            )
        CP("g", f"\n{'='*80:s}\nCell: {cell!s}, {df[df.cell_id == cell].cell_type.values[0]:s}")
        CP("g", f"     Group {df[df.cell_id == cell].Group.values[0]!s}")
        cell_group = df[df.cell_id == cell].Group.values[0]
        if (
            pd.isnull(cell_group) and len(df[df.cell_id == cell].Group.values) > 1
        ):  # incase first value is nan
            cell_group = df[df.cell_id == cell].Group.values[1]
        try:
            df_cell, df_tmp = filename_tools.get_cell(experiment, df, cell_id=cell)
        except ValueError:
            print("datatable functions:ComputeFIFits:get_cell failed to find:\n    ", df.cell_id)
            print("Here are the cell_ids in the dataframe: ", [x for x in df.cell_id.values])
            if self.status_bar is not None:
                self.status_bar.showMessage(f"Couldn't get cell: {cell:s} from dataframe")
            raise ValueError(f"Couldn't get cell: {cell:s} from dataframe")
        if df_cell is None:
            return None
        # print("    df_tmp group>>: ", df_tmp.Group.values)
        # print("    df_cell group>>: ", df_cell.keys())
        # print("combine_measures_and_FI_Fits: ", df_tmp.keys())
        protocol_list = list(df_cell.Spikes.keys())
        # build protocols excluding removed protocols
        protocols = []
        day_slice_cell = str(Path(df_cell.date, df_cell.slice_slice, df_cell.cell_cell))
        for protocol in protocol_list:  # check for spike analysis exclusions later
            if not self.check_excluded_dataset(day_slice_cell, experiment, protocol):
                protocols.append(protocol)
            # else:
            #     raise ValueError(f"Code stop #3 cell: {day_slice_cell:s}  protocol: {protocol:s}")
        srs = []
        durations = []
        delays = []
        Rs = []
        CNeut = []
        important = []
        # for each CCIV type of protocol that was run:
        valid_prots = []
        for nprot, protocol in enumerate(protocols):
            if protocol.endswith("0000"):  # bad protocol name
                continue
            if self.status_bar is not None:
                self.status_bar.showMessage(
                    f"Reading data for cell: {cell:s}, protocol: {protocol:s}"
                )
            day_slice_cell = str(Path(df_cell.date, df_cell.slice_slice, df_cell.cell_cell))
            CP("m", f"day_slice_cell: {day_slice_cell:s}, protocol: {protocol:s}")
            # if self.check_excluded_dataset(day_slice_cell, experiment, protocol):
            # continue
            # print(experiment["rawdatapath"], "\n  D: ", experiment["directory"], "\n  DSC: ", day_slice_cell, "\n  P: ", protocol)
            if str(experiment["rawdatapath"]).find(experiment["directory"]) == -1:
                fullpath = Path(
                    experiment["rawdatapath"],
                    experiment["directory"],
                    day_slice_cell,
                    Path(protocol).name,
                )
            else:
                fullpath = Path(experiment["rawdatapath"], day_slice_cell, protocol)
            with DR.acq4_reader.acq4_reader(fullpath, "MultiClamp1.ma") as AR:
                try:
                    if not AR.getData(
                        fullpath, allow_partial=True, record_list=[0]
                    ):  # just get the first record.
                        continue
                    sample_rate = AR.sample_rate[0]
                    duration = AR.tend - AR.tstart
                    delay = AR.tstart
                    srs.append(sample_rate)
                    Rs.append(AR.CCComp["CCBridgeResistance"])
                    CNeut.append(AR.CCComp["CCNeutralizationCap"])
                    durations.append(duration)
                    delays.append(delay)
                    important.append(AR.checkProtocolImportant(fullpath))
                    # CP("g", f"    Protocol {protocol:s} has sample rate of {sample_rate:e}")
                    valid_prots.append(protocol)
                except ValueError:
                    CP("r", f"Acq4Read failed to read data file: {str(fullpath):s}")
                    if self.status_bar is not None:
                        self.status_bar.showMessage(
                            f"Acq4Read failed to read data file: {str(fullpath):s}"
                        )
                    raise ValueError(f"Acq4Read failed to read data file: {str(fullpath):s}")

        protocols = valid_prots  # only count valid protocols
        CP("c", f"Valid Protocols: {protocols!s}")
        if len(protocols) > 1:
            protname = protocols
        elif len(protocols) == 1:
            protname = [protocols[0]]
        else:
            return None
        # CP("r", f"ccif cellid: {df_tmp.cell_id.values!s} ccif group: {group!s}")
        if "Date" in df_tmp.keys():  # solve poor programming practice from earlier versions.
            df_tmp.rename(columns={"Date": "date"}, inplace=True)
        datadict = {
            "ID": str(df_tmp.cell_id.values[0]),
            "Subject": str(df_tmp.cell_id.values[0]),
            "cell_id": cell,
            "Group": cell_group,
            "date": str(df_tmp.date.values[0]),
            "age": str(df_tmp.age.values[0]),
            "weight": str(df_tmp.weight.values[0]),
            "sex": str(df_tmp.sex.values[0]),
            "cell_type": df_tmp.cell_type.values[0],
            "protocol": protname,
            "important": important,
            "protocols": list(df_cell.IV),
            "sample_rate": srs,
            "delay": delay,
            "duration": duration,
            "Rs": Rs,
            "CNeut": CNeut,
            "holding": None,
            "currents": None,
            "firing_rates": None,
            "last_spikes": None,
        }
        # add any missing columns for computations.
        for colname in datacols:
            if colname not in datadict.keys():
                datadict[colname] = None
        self.experiment = experiment
        # get the measures for the fixed values from the measure list
        for measure in datacols:
            datadict = self.get_measure(
                df_cell,
                cell,
                measure,
                datadict,
                protocols,
                threshold_slope=experiment["AP_threshold_dvdt"],
            )
            # print("datadict: ", datadict)

        # now combine the FI data across protocols for this cell
        FI_Data_I1_: list = []
        FI_Data_FR1_: list = []  # firing rate
        FI_Data_I4_: list = []
        FI_Data_FR4_: list = []  # firing rate
        FI_fits: dict = {"fits": [], "pars": [], "names": []}

        linfits: list = []
        hill_max_derivs: list = []
        hill_i_max_derivs: list = []

        firing_currents: list = []
        firing_rates: list = []
        firing_last_spikes: list = []
        latencies: list = []
        adaptation_indices: list = []
        adaptation_rates: list = []
        adaptation_indices2: list = []
        adaptation_rates2: list = []
        protofails = 0
        # check protocols for AT least the minimum required

        if "FI_protocols_required" in experiment.keys():
            any_required_protocols = False
            CP("y", "\nChecking for required protocols")
            print("   Protocols in dataset: ", protocols)
            print("   Protocols required: ", experiment["FI_protocols_required"].keys())
            for ip, protocol in enumerate(protocols):
                short_proto_name = Path(protocol).name[:-4]
                print("    short_proto_name: ", short_proto_name)
                if short_proto_name in experiment["FI_protocols_required"].keys():
                    any_required_protocols = True
            CP("g", f"  Have required protocols: {any_required_protocols}")
            if not any_required_protocols:
                CP("y", f"    >>>> No required FI protocols found for cell: {cell:s}")
                if self.status_bar is not None:
                    self.status_bar.showMessage(
                        f"No required FI protocols found for cell: {cell:s}"
                    )
                return None
            print("\n")

        # check the protocols
        for ip, protocol in enumerate(protocols):
            if protocol.endswith("0000"):  # bad protocol name
                continue
        
            full_protocol = Path(protocol).name
            short_proto_name = Path(protocol).name[:-4]
            # limit FI protocols to ones with specific ranges and current levels
            if "FI_protocols_selected"  in experiment.keys():
                if short_proto_name not in experiment["FI_protocols_selected"].keys():
                    CP("y", f"    >>>> Protocol {protocol:s} not selected for FI analysis")
                    continue
            # check if duration is acceptable: protodurs is a dictionary from the configuration file.
            # Keys are acceptable protocols
            # values are a list of their acceptable durations
            if protodurs is not None:  # see if we can match what is in the protocol durations dict
                durflag = False  # with the actual protocol duration in dur[protocol]
                if short_proto_name not in protodurs.keys():
                    continue  # prototol will not be considered for this analysis
                if isinstance(protodurs[short_proto_name], float):
                    protodurs[short_proto_name] = [protodurs[short_proto_name]]  # make it a list
                for duration in protodurs[
                    short_proto_name
                ]:  # check if the duration is within the acceptable limits
                    # print("    >>>> Protocol: ", protocol, "duration of proto: ", dur[protocol],  "dur to test: ", duration)
                    if not np.isclose(durations[ip], duration):
                        durflag = True
                if durflag:
                    CP("y", f"    >>>> Protocol {protocol:s} has duration of {durations[ip]:e}")
                    CP("y", f"               This is not in accepted limits of: {protodurs!s}")
                    continue
                else:
                    CP(
                        "g",
                        f"    >>>> Protocol {protocol:s} has acceptable duration of {durations[ip]:e}",
                    )
            # print("protocol: ", protocol, "spikes: ", df_cell.Spikes[protocol]['spikes'])
            if len(df_cell.Spikes[protocol]["spikes"]) == 0:
                CP("y", f"    >>>> Skipping protocol with no spikes:  {protocol:s}")
                continue
            else:
                CP("g", f"    >>>> Analyzing FI for protocol: {protocol:s}")
            try:
                fidata = df_cell.Spikes[protocol]["FI_Curve"]
            except KeyError:
                print("FI curve not found for protocol: ", protocol, "for cell: ", cell)
                if self.status_bar is not None:
                    self.status_bar.showMessage(
                        f"FI curve not found for protocol: {protocol:s} for cell: {cell:s}"
                    )
                # print(df_cell.Spikes[protocol])
                protofails += 1
                if protofails > 4:
                    raise ValueError(
                        "FI curve data not found for protocol: ",
                        protocol,
                        "for cell: ",
                        cell,
                    )
                else:
                    continue
            # get mean rate spike rate in the train
            current = []
            rate = []

            last_spike = []
            # only do adaptation calculation for the protocols in the list
            if short_proto_name in experiment["Adaptation_index_protocols"].keys():
                adaptation_index, adaptation_rate = self.compute_adaptation_index(
                    df_cell.Spikes[protocol]["spikes"],
                    trace_duration=experiment["Adaptation_index_protocols"][short_proto_name],
                    trace_delay=experiment["Protocol_start_times"][short_proto_name],
                    rate_bounds=[
                        experiment["Adaptation_measurement_parameters"]["min_rate"],
                        experiment["Adaptation_measurement_parameters"]["max_rate"],
                    ],
                )
                adaptation_index2, adaptation_rate2 = self.compute_adaptation_index2(
                    df_cell.Spikes[protocol]["spikes"],
                    trace_duration=experiment["Adaptation_index_protocols"][short_proto_name],
                    trace_delay=experiment["Protocol_start_times"][short_proto_name],
                    rate_bounds=[
                        experiment["Adaptation_measurement_parameters"]["min_rate"],
                        experiment["Adaptation_measurement_parameters"]["max_rate"],
                    ],
                )

            else:
                adaptation_index = []
                adaptation_index2 = []
                adaptation_rate = []
                adaptation_rate2 = []
            # print("adaptation_indices: ", adaptation_indices)
            # print("adaptation_rates: ", adaptation_rates)
            # exit()

            for k, spikes in df_cell.Spikes[protocol]["spikes"].items():
                if len(spikes) == 0:
                    continue
                latencies = np.sort(
                    [
                        spikes[spike].AP_latency - spikes[spike].tstart
                        for spike in spikes
                        if spikes[spike].AP_latency is not None
                    ]
                )
                # print("latencies: ", k, latencies)
                # for spike in spikes:
                #     latencies.append(spikes[spike].AP_latency-spikes[spike].tstart)
                current.append(
                    spikes[0].current
                )  # current is the same for all spikes in this trace
                if len(latencies) >= 3:
                    rate.append(1.0 / np.mean(np.diff(latencies)))

                else:  # keep arrays the same length
                    rate.append(np.nan)
                if len(latencies) > 0:
                    last_spike.append(latencies[-1])
                else:
                    last_spike.append(np.nan)
            if np.max(fidata[0]) > 1.01e-9:  # accumulate high-current protocols
                FI_Data_I4_.extend(fidata[0])
                FI_Data_FR4_.extend(fidata[1] / durations[ip])
            else:  # accumulate other protocols <= 1 nA
                FI_Data_I1_.extend(fidata[0])
                FI_Data_FR1_.extend(fidata[1] / durations[ip])
                # accumulate this other information as well.
                firing_currents.extend(current)
                firing_rates.extend(rate)
                firing_last_spikes.extend(last_spike)
                adaptation_indices.extend(adaptation_index)
                adaptation_indices2.extend(adaptation_index2)
                adaptation_rates.extend(adaptation_rate)
                adaptation_rates2.extend(adaptation_rate2)
        adaptation_indices = np.array(adaptation_indices)
        adaptation_indices2 = np.array(adaptation_indices2)
        adaptation_rates = np.array(adaptation_rates)
        adaptation_rates2 = np.array(adaptation_rates2)

        FI_Data_I1 = []
        FI_Data_FR1 = []
        FI_Data_I4 = []
        FI_Data_FR4 = []
        if len(FI_Data_I1_) > 0:
            FI_Data_I1, FI_Data_FR1, FI_Data_FR1_Std, FI_Data_N1 = self.average_FI(
                FI_Data_I1_, FI_Data_FR1_, 1e-9
            )
            # do a curve fit on the first 1 nA of the protocol
            hill_max_derivs, hill_i_max_derivs, FI_fits, linfits = self.fit_FI_Hill(
                FI_Data_I=FI_Data_I1,
                FI_Data_FR=FI_Data_FR1,
                FI_Data_I_=FI_Data_I1_,
                FI_Data_FR_=FI_Data_FR1_,
                FI_Data_FR_Std=FI_Data_FR1_Std,
                FI_Data_N=FI_Data_N1,
                hill_max_derivs=hill_max_derivs,
                hill_i_max_derivs=hill_i_max_derivs,
                FI_fits=FI_fits,
                linfits=linfits,
                cell=cell,
                celltype=df_tmp.cell_type.values[0],
            )
        if len(FI_Data_I4_) > 0:
            FI_Data_I4, FI_Data_FR4, FI_Data_FR4_Std, FI_Data_N1 = self.average_FI(
                FI_Data_I4_, FI_Data_FR4_, 4e-9
            )

        # save the results
        datadict["FI_Curve1"] = [FI_Data_I1, FI_Data_FR1]
        datadict["FI_Curve4"] = [FI_Data_I4, FI_Data_FR4]
        datadict["current"] = FI_Data_I1
        datadict["spsec"] = FI_Data_FR1
        # datadict["Subject"] = df_tmp.cell_id.values[0]
        datadict["animal_identifier"] = df_tmp.cell_id.values[0]
        # datadict["Group"] = df_tmp.Group.values[0]
        datadict["sex"] = df_tmp.sex.values[0]
        datadict["celltype"] = df_tmp.cell_type.values[0]
        datadict["pars"] = [FI_fits["pars"]]
        datadict["names"] = []
        datadict["fit"] = [FI_fits["fits"]]
        datadict["F1amp"] = np.nan
        datadict["F2amp"] = np.nan
        datadict["Irate"] = np.nan
        datadict["maxHillSlope"] = np.nan
        datadict["maxHillSlope_SD"] = np.nan
        datadict["I_maxHillSlope"] = np.nan
        datadict["I_maxHillSlope_SD"] = np.nan
        datadict["firing_currents"] = None
        datadict["firing_rates"] = None
        datadict["AdaptIndex"] = adaptation_indices[~np.isnan(adaptation_indices)]  # remove nans...
        datadict["AdaptIndex2"] = adaptation_indices2[
            ~np.isnan(adaptation_indices2)
        ]  # remove nans...
        datadict["AdaptRates"] = adaptation_rates[~np.isnan(adaptation_rates)]  # remove nans...
        datadict["AdaptRates2"] = adaptation_rates2[~np.isnan(adaptation_rates2)]  # remove nans...
        datadict["last_spikes"] = None
        if len(linfits) > 0:
            datadict["FISlope"] = np.mean([s.slope for s in linfits])
        else:
            datadict["FISlope"] = np.nan
        if len(hill_max_derivs) > 0:
            datadict["maxHillSlope"] = np.mean(hill_max_derivs)
            datadict["maxHillSlope_SD"] = np.std(hill_max_derivs)
            datadict["I_maxHillSlope"] = np.mean(hill_i_max_derivs)
            datadict["I_maxHillSlope_SD"] = np.std(hill_i_max_derivs)
        if len(FI_Data_I1) > 0:
            i_one = np.where(FI_Data_I1 <= 1.01e-9)[0]
            datadict["FIMax_1"] = np.nanmax(FI_Data_FR1[i_one])
        if len(FI_Data_I4) > 0:
            i_four = np.where(FI_Data_I4 <= 4.01e-9)[0]
            datadict["FIMax_4"] = np.nanmax(FI_Data_FR4[i_four])
        i_curr = np.argsort(firing_currents)
        fc = np.array(firing_currents)[i_curr]
        fr = np.array(firing_rates)[i_curr]
        fs = np.array(firing_last_spikes)[i_curr]

        # prepare for Last Spikes in the train
        bins = np.linspace(
            -25e-12, 1025e-12, 22, endpoint=True
        )  # 50 pA bins centered on 0, 50, 100, etc.
        # print("fc len: ", len(fc), "fr: ", len(fr), "fs: ", len(fs))
        # for i in range(len(fc)):
        #     print(fc[i], fr[i], fs[i])
        if len(fc) > 0:
            datadict["firing_currents"] = scipy.stats.binned_statistic(
                fc, fc, bins=bins, statistic=np.nanmean
            ).statistic
        if len(fr) > 0 and len(fc) > 0:
            datadict["firing_rates"] = scipy.stats.binned_statistic(
                fc, fr, bins=bins, statistic=np.nanmean
            ).statistic
        if len(fc) > 0 and len(fs) > 0:
            datadict["last_spikes"] = scipy.stats.binned_statistic(
                fc, fs, bins=bins, statistic=np.nanmean
            ).statistic
        if self.status_bar is not None:
            self.status_bar.showMessage(f"Finished computing FI fits for cell: {cell:s}")
        # print(datadict)
        # exit()

        return datadict

    def compare_cell_id(self, cell_id: str, cell_ids: list):
        """compare_cell_id
        compare the cell_id to the list of cell_ids
        to see if it is in the list.
        The cellid is expected to be in the format:
            [path/]2019.02.22_000_S0C0 (or S00C00, S000C000)
        if there is not match, we check for an expanded name
        by converting 2019.02.22_000_S0C0 to 2019_02_22_000/slice_000/cell_000

        The first comparison is direct (exact).
        The second matches the path (as in the datasummary file)
        Parameters
        ----------
        cell_id : str
            cell_id to compare
        cell_ids : list
            list of cell_ids to compare to

        Returns
        -------
        bool
            True if cell_id is in the list, False otherwise
        """

        if pd.isnull(cell_id):
            return (
                None  # raise ValueError(f"Cell_id: {cell_id!s} is Null, which is a strange error")
            )
        if cell_id in cell_ids:
            return cell_id  # return the exact match
        else:

            cell_parts = cell_parts = str(cell_id).split("_")
            # print("cell_parts: ", cell_id, cell_parts)
            re_parse = re.compile(r"([Ss]{1})(\d{1,3})([Cc]{1})(\d{1,3})")
            if re_parse is None:
                raise ValueError(
                    f"Cell_id: {cell_id:s} not found in list of cell_ids in the datasummary file"
                )

            # print("cell id: ", cell_id)
            # print("cell_parts: ", cell_parts[-1])
            # print("re   match: ", re_parse.match(cell_parts[-1]))
            cnp = re_parse.match(cell_parts[-1]).group(4)
            cn = int(cnp)
            snp = re_parse.match(cell_parts[-1]).group(2)
            sn = int(snp)
            cp = "_".join(cell_parts[:-1])
            cell_id2 = f"{cp:s}/slice_{sn:03d}/cell_{cn:03d}"
            # print("Expanded cell_id: ", cell_id2)
            if cell_id2 in cell_ids:
                return cell_id2
            else:
                raise ValueError(
                    f"Cell_id: {cell_id:s} or {cell_id2:s} not found in list of cell_ids in the datasummary file"
                )

    def find_lowest_current_spike(self, row, SP):
        """find_lowest_current_spike Find the first spike that is evoked at the lowest current
        level from all the spike data for this protocol.
        ***** NOT CALLED IN THIS CLASS ****
        Parameters
        ----------
        row : Pandas series from the main dataframe
            row of the dataframe to fill in with the results
        SP : spike shape data structure


        Returns
        -------
        row : Pandas series
            from the main dataframe, updated with specific measures for
            the lowest current spike

        """

        dvdts = []
        for tr in SP.spikeShapes:  # for each trace
            if len(SP.spikeShapes[tr]) > 1:  # if there is a spike
                spk = SP.spikeShapes[tr][0]  # get the first spike in the trace
                dvdts.append(spk)  # accumulate first spike info

        if len(dvdts) > 0:
            currents = []
            for d in dvdts:  # for each first spike, make a list of the currents
                currents.append(d.current)
            min_current = np.argmin(currents)  # find spike elicited by the minimum current
            row.dvdt_rising = dvdts[min_current].dvdt_rising
            row.dvdt_falling = dvdts[min_current].dvdt_falling
            if not np.isnan(row.dvdt_falling):
                row.dvdt_ratio = dvdts[min_current].dvdt_rising / dvdts[min_current].dvdt_falling
            else:
                row.dvdt_ratio = np.nan
            row.dvdt_current = currents[min_current] * 1e12  # put in pA
            row.AP_thr_V = 1e3 * dvdts[min_current].AP_begin_V
            if dvdts[min_current].halfwidth_interpolated is not None:
                row.AP_HW = dvdts[min_current].halfwidth_interpolated * 1e3
            row.AP_begin_V = 1e3 * dvdts[min_current].AP_begin_V
            row.AP_peak_V = 1e3 * dvdts[min_current].peak_V
            CP(
                "y",
                f"I={currents[min_current]*1e12:6.1f} pA, dvdtRise={row.dvdt_rising:6.1f},"
                + f" dvdtFall={row.dvdt_falling:6.1f}, dvdtRatio={row.dvdt_ratio:7.3f},"
                + f" APthr={row.AP_thr_V:6.1f} mV, HW={row.AP_HW*1e3:6.1f} usec",
            )
        return row

    def find_lowest_current_trace(self, spikes):
        """find_lowest_current_trace : find the trace with the lowest current
        that evokes at least one spike.

        Parameters
        ----------
        spikes : dict spike data
            spike data dict by trace and spike number
            Each element is of the dataclass OneSpike (see spike_analysis.py for
            definition)

        Returns
        -------
        tuple
            Index into the current array, the current avlue, and the trace number
            corresponding to the trace with the lowest current that evoked a spike.
        """
        current = []
        trace = []
        for sweep in spikes["spikes"]:
            for spike in spikes["spikes"][sweep]:
                this_spike = spikes["spikes"][sweep][spike]
                current.append(this_spike.current)
                trace.append(this_spike.trace)
                break  # only get the first one
            # now find the index of the lowest current
        if len(current) == 0:
            return np.nan, np.nan, np.nan
        min_current_index = np.argmin(current)
        # print("current: ", current, "traces: ", trace)
        # print(current[min_current_index], trace[min_current_index])
        return min_current_index, current[min_current_index], trace[min_current_index]

    def adaptation_index(self, spk_lat, trace_duration: float = 1.0):
        """adaptation_index Compute an adaptation index based on Manis et al., 2019
        The adaptation index goes from:
        1 (only a spike in the frst half of the stimulus;
        no spikes in second half)
        to 0 (rate in the first and second half are identical)
        to -1 (all the spikes are in the second half)

        Parameters
        ----------
        spk_lat : _type_
            _description_
        trace_duration : float, optional
            _description_, by default 1.0

        Returns
        -------
        float
            adaptation index as described above
        """
        ai = (-2.0 / len(spk_lat)) * np.sum((spk_lat / trace_duration) - 0.5)
        return ai

    def adaptation_index2(self, spk_lat: Union[list, np.ndarray]):
        """adaptation_index Compute an adaptation index from eFEL (2025)
        Modified to use Allen Institute ephys SDK version (norm_diff)
        Parameters
        ----------
        spk_lat : array
            spike latencies, already trimmed to current step window.
        trace_duration : float, optional
            _description_, by default 1.0

        Returns
        -------
        float
            adaptation index as described above
        """

        if len(spk_lat) < 4:
            return np.nan
        # eFEL version:

        # clean up the spike latencies to be sure there are no duplicates ?
        # spk_lat = np.unique(spk_lat)  # prevent isi_sub from being zero. how that might happen is a mystery
        # isi_values = spk_lat[1:] - spk_lat[:-1]
        # isi_sum = isi_values[1:] + isi_values[:-1]
        # isi_sub = isi_values[1:] - isi_values[:-1]
        # print("adaptation_index2: ISI SUB: ", isi_sub)
        # nonzeros = np.argwhere(isi_sub != 0.0)
        # adaptation_index = np.mean(isi_sum[nonzeros] / isi_sub[nonzeros])

        # Allen Institute version:
        if np.allclose((spk_lat[1:] + spk_lat[:-1]), 0.0):
            return np.nan
        norm_diffs = (spk_lat[1:] - spk_lat[:-1]) / (spk_lat[1:] + spk_lat[:-1])
        norm_diffs[(spk_lat[1:] == 0) & (spk_lat[:-1] == 0)] = 0.0
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning, module="numpy")
            adaptation_index = np.nanmean(norm_diffs)
        return adaptation_index

    def compute_adaptation_index(
        self,
        spikes,
        trace_delay: float = 0.15,
        trace_duration: float = 1.0,
        rate_bounds: list = [20.0, 40.0],
    ):
        """compute_adapt_index
        Compute the adaptation index for a set of spikes
        The adaptation index is the ratio of the last interspike interval
        to the first interspike interval.
        Assumes the spikes are times corrected for the delay to the start of the stimulus.

        Parameters
        ----------
        spikes : list
            list of spikes, with 0 time at the start of the stimulus
        trace_duration: float
            duration of the current step, in seconds

        Returns
        -------
        lists
            adaptation indices, rates at which index was computed
        """

        recnums = list(spikes.keys())
        adapt_rates = []
        adapt_indices = []
        print("Rate bounds: ", rate_bounds)

        for rec in recnums:
            spikelist = list(spikes[rec].keys())
            if len(spikelist) < 3:
                continue
            # if len(spikelist) < 5 or len(spikelist) > 10:
            #     continue
            spk_lat = np.array(
                [spikes[rec][spk].AP_latency for spk in spikelist if spk is not None]
            )
            spk_lat = np.array([spk for spk in spk_lat if spk is not None])
            # print("spk_lat: ", spk_lat)
            if spk_lat is None or len(spk_lat) < 3:
                continue
            else:
                spk_lat -= trace_delay
            # print(spk_lat)
            n_spikes = len(spk_lat)
            rate = n_spikes / (spk_lat[-1] - spk_lat[0])
            if rate < rate_bounds[0] or rate > rate_bounds[1]:
                # CP("y", f"Adaption calculation: rec: {rec:d} failed rate limit: {rate:.1f}, {rate_bounds!s}, spk_lat: {spk_lat!s}")
                continue
            # else:
            #     CP("c", f"Adaption calculation: rec: {rec:d} rate: {rate:.1f}, spk_lat: {spk_lat!s} PASSED")
            adapt_rates.append(rate)
            adapt_indices.append(self.adaptation_index(spk_lat, trace_duration))
        #    CP("y", f"Adaptation index: {adapt_indices[-1]:6.3f}, rate: {rate:6.1f}")

        return adapt_indices, adapt_rates

    def compute_adaptation_index_one_trace(
        self, latencies, trace_duration: float = 1.0, rate_bounds: list = [0.1, 100.0]
    ):
        n_spikes = len(latencies)
        rate = n_spikes / (latencies[-1] - latencies[0])
        if rate < rate_bounds[0] or rate > rate_bounds[1]:
            return np.nan, rate
        adapt_index = self.adaptation_index(latencies, trace_duration)
        CP("y", f"Adaptation index: {adapt_index:6.3f}, rate: {rate:6.1f}")
        return adapt_index, rate

    def compute_adaptation_index2(
        self,
        spikes,
        trace_delay: float = 0.15,
        trace_duration: float = 1.0,
        rate_bounds: list = [20.0, 40.0],
    ):
        """compute_adapt_index
        Compute the adaptation index for a set of spikes
        This uses the algorithm from eFEL to compute the adaptation index
        Assumes the spikes are times corrected for the delay to the start of the stimulus.

        Parameters
        ----------
        spikes : list
            list of spikes, with 0 time at the start of the stimulus
        trace_duration: float
            duration of the current step, in seconds

        Returns
        -------
        lists
            adaptation indices, rates at which index was computed
        """

        recnums = list(spikes.keys())
        adapt_rates2 = []
        adapt_indices2 = []
        print("Rate bounds: ", rate_bounds)

        for rec in recnums:
            spikelist = list(spikes[rec].keys())
            if len(spikelist) < 4:
                continue
            spk_lat = np.array(
                [spikes[rec][spk].AP_latency for spk in spikelist if spk is not None]
            )
            spk_lat = np.array([spk for spk in spk_lat if spk is not None])
            # print("spk_lat: ", spk_lat)
            if spk_lat is None:
                continue
            else:
                spk_lat -= trace_delay
                spk_lat = spk_lat[spk_lat < trace_duration]
            # print(spk_lat)
            n_spikes = len(spk_lat)
            rate = n_spikes / (spk_lat[-1] - spk_lat[0])
            if rate < rate_bounds[0] or rate > rate_bounds[1]:
                # CP("y", f"Adaption calculation: rec: {rec:d} failed rate limit: {rate:.1f}, {rate_bounds!s}, spk_lat: {spk_lat!s}")
                continue
            # else:
            #     CP("c", f"Adaption calculation: rec: {rec:d} rate: {rate:.1f}, spk_lat: {spk_lat!s} PASSED")
            adapt_rates2.append(rate)
            adapt_indices2.append(self.adaptation_index2(spk_lat))
        #    CP("y", f"Adaptation index: {adapt_indices[-1]:6.3f}, rate: {rate:6.1f}")

        return adapt_indices2, adapt_rates2

    def convert_FI_array(self, FI_values):
        """convert_FI_array Take a potential string representing the FI_data,
        and convert it to a numpy array

        Parameters
        ----------
        FI_values : str or numpy array
            data to be converted

        Returns
        -------
        numpy array
            converted data from FI_values
        """
        if isinstance(FI_values, str):
            fistring = FI_values.replace("[", "").replace("]", "").replace("\n", "")
            fistring = fistring.split(" ")
            FI_data = np.array([float(s) for s in fistring if len(s) > 0])
            FI_data = FI_data.reshape(2, int(FI_data.shape[0] / 2))
        else:
            FI_data = FI_values
        FI_data = np.array(FI_data)
        return FI_data

    def add_measure(self, proto, measure, value, trace=None):
        self.measures["protocol"].append(proto)
        self.measures["measure"].append(measure)
        self.measures["value"].append(value)
        self.measures["trace"].append(trace)

    def get_measure(self, df_cell:pd.DataFrame, cell_id: str, measure:str, datadict, protocols, threshold_slope: float = 20.0):
        """get_measure : for the given cell, get the measure from the protocols

        Parameters
        ----------
        df_cell : _type_
            _description_
        cell_id: str
            The Cell id in the form of yyyy.mm.dd_000/slice_nnn/cell_nnn
        measure : _type_
            _description_
        datadict : _type_
            _description_
        protocols : _type_
            _description_

        Returns
        -------
        _type_
            _description_
        """
        spkkeys = df_cell.Spikes[protocols[0]]["spikes"].keys()
        # print(spkkeys)
        # print("Get Measure, protocol keys: ", df_cell.Spikes[protocols[0]]['spikes'][1])
        # df_cell.Spikes[protocols[0]]['spikes'][1] is a dictionary of the spikes in the trace
        # each dict entry is a OneSpike, and has AP_begin_V (threshold V), peak_V, peak_T, current, etc.
        # raise ValueError("Check the keys")
        # print("="*120)
        # print("get measure starting")
        self.measures = {"protocol": [], "measure": [], "value": [], "trace": []}
        # First we do the parameters in the iv (rmp, tau, rin, etc)
        # print("  >>>> Measure: ", measure)
        # known_measure = (measure in iv_keys) or (measure in iv_mapper.keys()) or (
        #     measure in spike_keys
        # ) or (measure in ["current"])
        # if not known_measure:
        #     print("        >>>> Not a known measure: ", measure)
        #     raise ValueError()
        #     datadict[measure] = np.nan
        #     return datadict
        match measure:
            case measure if measure in iv_keys:
                # print("        >>>> in iv_keys")
                for protocol in protocols:
                    if measure in list(df_cell.IV[protocol].keys()):
                        value = np.nan
                        if measure != "taum":
                            value = df_cell.IV[protocol][measure]
                        # self.add_measure(protocol, measure, value=df_cell.IV[protocol][measure])
                        else:  # handle taum by building array from the tau in taupars array
                            if len(df_cell.IV[protocol]["taupars"]) == 0:
                                value = (
                                    np.nan
                                )  # self.add_measure(protocol, measure, value=np.nan)  # no fit data
                            elif len(df_cell.IV[protocol]["taupars"]) == 3 and isinstance(
                                df_cell.IV[protocol]["taupars"][2], float
                            ):  # get the value from the fit, single nested
                                value = df_cell.IV[protocol]["taupars"][2]
                                # self.add_measure(
                                #     protocol, measure, value=df_cell.IV[protocol]["taupars"][2]
                                # )
                            elif (  # get value from fit, nested list
                                len(df_cell.IV[protocol]["taupars"]) == 1
                                and isinstance(df_cell.IV[protocol]["taupars"], list)
                                and len(df_cell.IV[protocol]["taupars"][0]) == 3
                            ):
                                value = df_cell.IV[protocol]["taupars"][0][2]
                                # self.add_measure(
                                #     protocol, measure, value=df_cell.IV[protocol]["taupars"][0][2]
                                # )
                            else:  # def a coding error
                                print(
                                    "what is wrong with taupars: ", df_cell.IV[protocol]["taupars"]
                                )
                                exit()
                        # print("TAU: ", protocol, measure, value)
                        self.add_measure(protocol, measure, value=value)
            #
            # Next we check what is in iv_mapper - this remaps the "measure" to
            # a value in the data table.
            # iv_mapper: dict = {
            #     "tauh": "tauh_tau",
            #     "Gh": "tauh_Gh",
            #     "taum": "taum",
            #     "taum_averaged": "taum_averaged",
            #     "Rin": "Rin",
            #     "RMP": "RMP",
            # }

            case measure if (measure in list(iv_mapper.keys())) and (iv_mapper[measure] in iv_keys):
                # print("        >>>> in iv_mapper")
                for protocol in protocols:
                    if iv_mapper[measure] in df_cell.IV[protocol].keys():
                        self.add_measure(
                            protocol, measure, value=df_cell.IV[protocol][iv_mapper[measure]]
                        )
            # get the current...
            case ["current"]:
                # print("        >>>> in current")
                for (
                    protocol
                ) in protocols:  # for all protocols with spike analysis data for this cell
                    if "spikes" not in df_cell.Spikes[protocol].keys():
                        self.add_measure(protocol, measure, value=np.nan)
                        CP(
                            "y",
                            f"No spikes in protocol (measure=current):  {protocol!s}, {df_cell.Spikes[protocol].keys():s}",
                        )
                        self.add_measure(protocol, measure, value=np.nan)
                        continue
                    # we need to get the first spike evoked by the lowest current level ...
                    min_current_index, current, trace = self.find_lowest_current_trace(
                        df_cell.Spikes[protocol]
                    )
                    if not np.isnan(min_current_index):
                        self.add_measure(protocol, measure, value=current)
                    else:
                        self.add_measure(protocol, measure, value=np.nan)

            #
            # Next we check the spike keys.
            # this actually wont ever pass:
            # df_cell.Spikes[protocols[0]]['spikes'][1] is a dictionary of the spikes in the trace
            # because the keys in spikes are the trace numbers, not the measures.
            # elif measure in spike_keys:
            #     for protocol in protocols:
            #         if measure in df_cell.Spikes[protocol].keys():
            #             print("adding measure from spike keys: ", measure)
            #             self.add_measure(protocol, measure, value=df_cell.Spikes[protocol][measure])
            # or if they are mapped....
            # elif measure in mapper1.keys() and mapper1[measure] in spike_keys:
            #     for protocol in protocols:
            #         if mapper1[measure] in df_cell.Spikes[protocol].keys():
            #             self.add_measure(
            #                 protocol, measure, value=df_cell.Spikes[protocol][mapper1[measure]]
            #             )
            case measure if measure in spike_keys:  # here we actually check the spike data
                print("       >>>> in spike_keys")
                for (
                    protocol
                ) in protocols:  # for all protocols with spike analysis data for this cell
                    # we need to get the first spike evoked by the lowest current level ...
                    prot_spike_count = 0
                    sp_exc_list = self.experiment["exclude_Spikes"]

                    # check the spike exclusion list for this protocol.
                    cell_id_full = filename_tools.make_cellid_from_slicecell(cell_id)
                    # print(cell_id, cell_id_full, "\n   ", sp_exc_list.keys())
                    if sp_exc_list is not None and (cell_id_full in sp_exc_list.keys()):
                        skip = False
                        if (protocol in sp_exc_list[cell_id_full]['protocols']):
                            skip = True
                        if 'all' in sp_exc_list[cell_id_full]['protocols']:
                            skip = True
                        if skip:
                            CP(
                            "y",
                            f"Skipping protocol {cell_id_full:s} {protocol:s} because it is in the spike exclusion list, reason: {sp_exc_list[cell_id_full]['reason']}",
                            )
                            continue
                    if "spikes" not in df_cell.Spikes[protocol].keys():
                        CP(
                            "y",
                            f"No spikes in protocol (measure={measure:s}: {protocol!s}, {df_cell.Spikes[protocol].keys():s}",
                        )
                        self.add_measure(protocol, measure, value=np.nan)

                    have_LCS_data = (
                        "LowestCurrentSpike" in df_cell.Spikes[protocol].keys()
                        and df_cell.Spikes[protocol]["LowestCurrentSpike"] is not None
                        and len(df_cell.Spikes[protocol]["LowestCurrentSpike"]) > 0
                    )
                    if have_LCS_data:
                        CP("y", f"    have LowestCurrentSpike: {have_LCS_data!s}  from {protocol:s}")
                    else:
                        CP("r", f"    DO NOT have LowestCurrentSpike: {have_LCS_data!s}  from {protocol:s}")
                    if have_LCS_data:
                        print("LCS Data: ", df_cell.Spikes[protocol]["LowestCurrentSpike"])
                    # exit()
                    spike_data = df_cell.Spikes[protocol]["spikes"]
                    # print("    checking measure in protocol: ", measure)
                    if (
                        measure
                        in [
                            "dvdt_rising",
                            "dvdt_falling",
                            # "dvdt_ratio",
                            "AP_HW",
                            "AP_begin_V",
                            "AP_thr_V",
                            "peak_V",
                            "AP_peak_T",
                            "AP_max_V"
                            "AHP_trough_V",
                            "AHP_trough_T",
                            "AHP_depth_V",
                            "AHP_depth_T",

                        ]
                        and have_LCS_data
                    ):  # use lowest current spike
                        min_current_index, current, trace = self.find_lowest_current_trace(
                            df_cell.Spikes[protocol]
                        )
                        # print(
                        #     "    Got spike for min current: ",
                        #     min_current_index,
                        #     current,
                        #     trace,
                        #     measure,
                        # )
                        if not np.isnan(min_current_index):
                            spike_data = df_cell.Spikes[protocol]["spikes"][trace][0].__dict__
                            self.add_measure(protocol, measure, value=spike_data[mapper[measure]])
                        else:
                            self.add_measure(protocol, measure, value=np.nan)
                        # print("spike data: ", spike_data.keys())

                    elif not have_LCS_data:
                        CP("r", "Lowest Current spike data is empty")

                    elif measure in ["AP_thr_V", "AP_begin_V"]:
                        CP("m", "APTHR or BEGIN")
                        value = np.nan
                        if have_LCS_data:
                            CP("c", f"Adding {measure:s} from LCS to data [{protocol:s}]")
                            if "AP_thr_V" in df_cell.Spikes[protocol]["LowestCurrentSpike"].keys():
                                Vthr = df_cell.Spikes[protocol]["LowestCurrentSpike"]["AP_thr_V"]
                                value = Vthr
                            elif (
                                "AP_begin_V"
                                in df_cell.Spikes[protocol]["LowestCurrentSpike"].keys()
                            ):
                                Vthr = df_cell.Spikes[protocol]["LowestCurrentSpike"]["AP_begin_V"]
                                value = Vthr

                            else:
                                print(
                                    "Failed to find AP_begin_V/AP_thr_V in LowestCurrentSpike",
                                    protocol,
                                    measure,
                                )
                                print(
                                    "LCS: ", df_cell.Spikes[protocol]["LowestCurrentSpike"].keys()
                                )
                                value = np.nan
                        else:
                            df_cell.Spikes[protocol][
                                "LowestCurrentSpike"
                            ] = None  # install value, but set to None

                            CP(
                                "r",
                                f"Missing lowest current spike data in spikes dictionary: {cell_id_full:s} {protocol:s}, {df_cell.Spikes[protocol].keys()!s}",
                            )
                            value = np.nan
                        self.add_measure(protocol, measure, value=value)

                    elif measure in ["AP_thr_T"]:
                        if have_LCS_data:
                            Vthr_time = df_cell.Spikes[protocol]["LowestCurrentSpike"]["AP_thr_T"]
                            self.add_measure(protocol, measure, value=Vthr_time)
                        else:
                            self.add_measure(protocol, measure, value=np.nan)
                    
                    elif measure in ["AHP_trough_V"]:
                        if have_LCS_data:
                            print("through look, : ", df_cell.Spikes[protocol]["LowestCurrentSpike"])
                            try:
                                AHP_trough_V = (
                                df_cell.Spikes[protocol]["LowestCurrentSpike"]["AHP_trough_V"] * 1e-3
                            )
                            except:
                                print("AHP_trough_V not found in LCS data")
                                AHP_trough_V = np.nan
                                raise ValueError(
                                    f"AHP_trough_V not found in LCS data: {cell_id_full:s}  {df_cell.Spikes[protocol]['LowestCurrentSpike'].keys()!s}"
                                )
                        else:
                            AHP_trough_V = np.nan
                        self.add_measure(protocol, measure, value=AHP_trough_V)

                    elif measure in ["AP_peak_V"]:
                        if have_LCS_data:
                            AP_peak_V = (
                                df_cell.Spikes[protocol]["LowestCurrentSpike"]["AP_peak_V"] * 1e-3
                            )  # convert back to V *like AP_thr_V*
                            self.add_measure(protocol, measure, value=AP_peak_V)
                        else:
                            self.add_measure(protocol, measure, value=np.nan)
                   
                    elif measure in ["AHP_relative_depth_V"]:
                        if have_LCS_data:
                            AHP_relative_depth = (
                                df_cell.Spikes[protocol]["LowestCurrentSpike"]["AHP_relative_depth_V"]
                            )  # convert back to V *like AP_thr_V*
                        else:
                            AHP_relative_depth = np.nan
                        self.add_measure(protocol, measure, value=AHP_relative_depth)

                        CP("m", f"AHP_relative depth depth_V set in get_measure: {AHP_relative_depth:6.3f}")

                    
                    elif measure in ["AP_HW"]:
                        if have_LCS_data:
                            AP_HW = df_cell.Spikes[protocol]["LowestCurrentSpike"]["AP_HW"]
                            self.add_measure(protocol, measure, value=AP_HW)
                    elif (
                        measure in mapper.keys() and mapper[measure] in spike_data.keys()
                    ):  # if the measure exists for this sweep
                        CP("r", f"Adding measure: {measure:s} from {mapper[measure]:s}")
                        self.add_measure(protocol, measure, value=spike_data[mapper[measure]][0])
                    elif measure in [
                        "dvdt_ratio",
                        "AP_height_V",
                    ]:  # calculated values based on data (ratio, difference)
                        pass
                    else:
                        CP(
                            "r",
                            f"Measure <{measure:s}> not found in spike_data keys:, {mapper.keys()!s}",
                        )
                        # CP(
                        #     "r",
                        #     f"\n   or mapped in {mapper[measure]!s} to {spike_data.keys()!s}",
                        # )
                        # raise ValueError()

                    prot_spike_count += 1

            case _:
                # CP("r", f"Measure {measure:s} did not match any actions")
                pass
        for i, xm in enumerate(self.measures["value"]):
            if self.measures["value"][i] is None:
                self.measures["value"][i] = np.nan
        N = np.count_nonzero(~np.isnan(self.measures["value"]))

        if datadict[measure] is None:
            datadict[measure] = []
        if N > 0:
            datadict[measure].extend(self.measures["value"])  # = np.nanmean(m)
        else:
            datadict[measure].append(np.nan)
        return datadict

    def textbox_setup(self, textbox):
        if textbox is not None:
            self.textbox = textbox
            self.textclear()

    def textclear(self):
        if self.textbox is None:
            raise ValueError("datatables - functions - textbox has not been set up")

        if self is None:  #  or self.in_Parallel:
            return
        else:
            self.textbox.clear()

    def text_get(self):
        if self.textbox is None:
            raise ValueError("datatables - functions - textbox has not been set up")
        return self.textbox.toPlainText()

    def textappend(self, text, color="white"):
        if self.textbox is None:
            print(text)
            return

        colormap = {
            "[31m": "red",
            "[48:5:208:0m": "orange",
            "[33m": "yellow",
            "[32m": "limegreen",
            "[34m": "pink",
            "[35m": "magenta",
            "[36m": "cyan",
            "[30m": "black",
            "[37m": "white",
            "[0m": "white",
            "[100m": "backgray",
        }
        if self is None:
            CP(color, text)  # just go straight to the terminal
        else:
            text = "".join(text)
            text = text.split("\n")

            for textl in text:
                # print(f"text: <{textl:s}>")

                if len(textl) > 0 and textl[0] == "\x1b":
                    textl = textl[1:]  # clip the escape sequence
                    for k in colormap.keys():
                        if textl.startswith(k):  # skip the escape sequence
                            textl = textl[len(k) :]
                            textl = textl.replace("[0m", "")
                            color = colormap[k]
                            self.textbox.setTextColor(QtGui.QColor(color))
                            break
                textl = textl.replace("[0m", "")
                self.textbox.append(textl)
                self.textbox.setTextColor(QtGui.QColor("white"))
