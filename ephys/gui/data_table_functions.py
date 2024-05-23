""" General functions for data_tables and data_table_manager
    We are using a class here just to make it easier to pass around
"""

import logging
import pprint
import re
import subprocess
from pathlib import Path
from typing import Union

import colormaps

import numpy as np
import pandas as pd
import pyqtgraph as pg
import scipy
from pylibrary.plotting import plothelpers as PH
from pylibrary.tools import cprint
from pyqtgraph.Qt import QtGui, QtCore
from scipy.interpolate import spalde, splev, splrep

import ephys
import ephys.datareaders as DR
from ephys.ephys_analysis import spike_analysis
from ephys.tools import annotated_cursor, filename_tools, utilities
from ephys.tools import annotated_cursor_pg

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
    logging_fh = logging.FileHandler(filename=log_file)
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
    "dvdt_rising",
    "dvdt_falling",
    "current",
    "AP_thr_V",
    "AP_thr_T",
    "AP_HW",
    "AP15Rate",
    "AdaptRatio",
    "AHP_trough_V",
    "AHP_depth_V",
    "AHP_trough_T",
    "tauh",
    "Gh",
    "FiringRate",
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
    "FI_Curve",
    "FiringRate",
    "AP1_Latency",
    "AP1_HalfWidth",
    "AP1_HalfWidth_interpolated",
    "AP2_Latency",
    "AP2_HalfWidth",
    "AP2_HalfWidth_interpolated",
    "FiringRate_1p5T",
    "AHP_Depth",
    "AHP_Trough",
    "spikes",
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
    "AP_thr_V": "AP_begin_V",
    "AP_thr_T": "AP_begin_T",
    "AP_HW": "halfwidth",
    "dvdt_rising": "dvdt_rising",
    "dvdt_falling": "dvdt_falling",
}
# map summary/not individual spike data to top level keys
mapper1: dict = {
    "AP15Rate": "FiringRate_1p5T",
    "AdaptRatio": "AdaptRatio",
}

iv_mapper: dict = {
    "tauh": "tauh_tau",
    "Gh": "tauh_Gh",
    "taum": "taum",
    "Rin": "Rin",
    "RMP": "RMP",
}


def print_spike_keys(row):
    if pd.isnull(row.IV):
        return row
    # print(row.IV)
    return row


class Functions:
    def __init__(self):
        self.textbox = None
        self.cursor = []  # a list to hold cursors (need to keep a live reference)
        pass

    def get_row_selection(self, table_manager):
        """
        Find the selected rows in the currently managed table, and if there is a valid selection,
        return the index to the first row and the data from that row
        """
        self.selected_index_rows = table_manager.table.selectionModel().selectedRows()
        if self.selected_index_rows is None:
            return None, None
        else:
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
        print("get multiple row selection: ", len(self.selected_index_rows))
        if self.selected_index_rows is None:
            return None
        else:
            rows = []
            for row in self.selected_index_rows:
                # index_row = self.selected_index_rows[0]
                selected = table_manager.get_table_data(row)  # table_data[index_row]
                if selected is None:
                    return None
                else:
                    rows.append(selected)
            return rows

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
            if "CCIV".casefold() in prot.casefold():
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
        exclude_flag = day_slice_cell in experiment["excludeIVs"]
        print("    IV is in exclusion table: ", exclude_flag)
        if exclude_flag:
            exclude_table = experiment["excludeIVs"][day_slice_cell]
            print("    excluded table data: ", exclude_table)
            print("    testing protocol: ", protocol)
            proto = Path(protocol).name  # passed protocol has day/slice/cell/protocol
            if proto in exclude_table["protocols"] or exclude_table["protocols"] == ["all"]:
                CP(
                    "y",
                    f"Excluded cell/protocol: {day_slice_cell:s}, {proto:s} because: {exclude_table['reason']:s}",
                )
                Logger.info(
                    f"Excluded cell: {day_slice_cell:s}, {proto:s} because: {exclude_table['reason']:s}"
                )
                return True
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
                if cell_df["Spikes"][protocol]["LowestCurrentSpike"] is None:
                    continue
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
                CP.cprint("r", f"No protocol found with a small current spike!: {protocols!s} ")
                continue
            try:
                # print("min protocol: ", min_protocol)
                # print("min protocol keys: ", cell_df["Spikes"][min_protocol].keys())
                # print("min trace #: ", min_trace)
                low_spike = cell_df["Spikes"][min_protocol]["spikes"][min_trace][0]
            except KeyError:
                CP.cprint("Failed to get min spike")
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

    def get_selected_cell_data_FI(
        self,
        experiment: pd.DataFrame,
        assembleddata: pd.DataFrame,
    ):
        print("got row selection")
        print("# selected: ", len(assembleddata))
        if len(assembleddata) > 0:
            import matplotlib.pyplot as mpl

            P = PH.regular_grid(
                2,
                3,
                margins={
                    "bottommargin": 0.08,
                    "leftmargin": 0.1,
                    "rightmargin": 0.05,
                    "topmargin": 0.1,
                },
                figsize=(11, 8),
            )
            fig = P.figure_handle
            ax = P.axarr
            iplot = 0
            dropout_threshold = 0.8  # seconds
            N = len(assembleddata)
            max_I = 1000.0
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
            nplots = 0
            for selected in assembleddata.itertuples():
                pcolor = colors[nplots].colors

                day = selected.date[:-4]
                slicecell = selected.cell_id[-4:]
                # we need to reach into the main data set to get the expression,
                # as it is not carried into the assembled data.
                cell_df, _ = filename_tools.get_cell(
                    experiment, assembleddata, cell_id=selected.cell_id
                )
                # print(cell_df.keys())
                # print(cell_df.cell_expression)
                # print(cell_df['IV'].keys())
                datadict = self.compute_FI_Fits(
                    experiment,
                    assembleddata,
                    selected.cell_id,
                    protodurs=experiment["FI_protocols"],
                )
                # designate symbols:
                # s for +/+ genotype
                # o for -/- genotype
                # filled for GFP+ or EYFP+
                # open for GFP- or EYFP-
                print("expression: ", cell_df.cell_expression)
                match selected.Group:
                    case "+/+":
                        symbol = "s"
                    case "-/-":
                        symbol = "o"
                    case "+/-" | "-/+":
                        symbol = "D"
                    case _:
                        symbol = "X"
                match cell_df.cell_expression:
                    case "+" | "GFP+" | "EYFP" | "EYFP+":
                        fillstyle = "full"
                        facecolor = pcolor
                    case "-" | "GFP-" | "EYFP-" | "EGFP-":
                        fillstyle = "full"
                        facecolor = "white"
                    case _:
                        fillstyle = "full"
                        facecolor = "lightgrey"
                # print(cell_df.cell_expression, selected.Group)
                # print(symbol, fillstyle, facecolor, pcolor)
                if datadict is not None:
                    try:
                        fit = datadict["fit"][0][0]
                    except:
                        print("No fit? : ")
                        print("     datadict keys: ", datadict.keys())
                        print("     fit keys: ", datadict["fit"])
                    ax[0, 0].plot(
                        np.array(datadict["FI_Curve1"][0]) * 1e12,
                        datadict["FI_Curve1"][1],
                        marker=symbol,
                        linestyle="-",
                        markersize=4,
                        fillstyle=fillstyle,
                        markerfacecolor=facecolor,
                        color=pcolor,
                    )
                    if "fit" in datadict.keys() and len(datadict["fit"][0]) > 0:
                        fit = datadict["fit"][0][0]
                        ax[0, 0].plot(np.array(fit[0][0]) * 1e12, fit[1][0], "b--")
                    max_fr[selected.Group].append(np.max(datadict["FI_Curve1"][1]))
                    
                    if max_I > 1000:
                        ax[0, 0].plot(
                            np.array(datadict["FI_Curve4"][0]) * 1e12,
                            np.array(datadict["FI_Curve4"][1]),
                            marker=symbol,
                            linestyle="-",
                            markersize=4,
                            fillstyle=fillstyle,
                            markerfacecolor=facecolor,
                            color=pcolor,
                        )

                    if datadict["firing_currents"] is not None:
                        ax[1, 0].plot(
                            np.array(datadict["firing_currents"]) * 1e12,
                            datadict["firing_rates"],
                            marker=symbol,
                            linestyle="-",
                            markersize=4,
                            fillstyle=fillstyle,
                            markerfacecolor=facecolor,
                            color=pcolor,
                        )
                        ax[0, 1].plot(
                            np.array(datadict["firing_currents"]) * 1e12,
                            datadict["last_spikes"],
                            marker=symbol,
                            linestyle="-",
                            markersize=4,
                            fillstyle=fillstyle,
                            markerfacecolor=facecolor,
                            color=pcolor,
                        )
                        # plot maximal firing rate

                    # compute "dropout" time
                    if datadict["last_spikes"] is not None:
                        # find the largest current that is above the dropuout threshold
                        # for ils in range(len(datadict["last_spikes"])):
                        #     print("Last spikes: ", ils, datadict["last_spikes"][ils] * 1e3)
                        do_pts = np.nonzero(np.array(datadict["last_spikes"]) >= dropout_threshold)[
                            0
                        ]
                        # print("do_pts: ", do_pts)
                        # print("Firing currents: ", datadict["firing_currents"])

                        # print("idrop: ", idrop)
                        # print(datadict["last_spikes"][idrop] * 1e3)
                        # print(datadict["firing_currents"][idrop] * 1e12)
                        if len(do_pts) > 0:
                            idrop = np.argmax(datadict["firing_currents"][do_pts]) + do_pts[0]
                            dropout[1, nplots] = datadict["firing_currents"][idrop] * 1e12
                            dropout[0, nplots] = datadict["last_spikes"][idrop] * 1e3
                        else:
                            dropout[1, nplots] = datadict["firing_currents"][0] * 1e12
                            dropout[0, nplots] = 0
                        cellids[nplots] = selected.cell_id
                        expression[nplots] = cell_df.cell_expression
                        genotype[nplots] = selected.Group
                        print(selected.cell_id, dropout[0, nplots], dropout[1, nplots])
                        ax[1, 1].plot(
                            [0, 1000],
                            [dropout_threshold, dropout_threshold],
                            color="gray",
                            linestyle="--",
                            linewidth=0.33,
                        )
                        ax[1, 1].plot(
                            dropout[1, nplots],
                            dropout[0, nplots],
                            marker=symbol,
                            linestyle="-",
                            markersize=4,
                            fillstyle=fillstyle,
                            markerfacecolor=facecolor,
                            color=pcolor,
                            clip_on=False,
                        )
                        ax[1, 1].text(
                            x=dropout[1, nplots],
                            y=dropout[0, nplots],
                            s=selected.cell_id,
                            fontsize=6,
                            horizontalalignment="right",
                            color=fcol[cell_df.genotype],
                        )
                    iplot += 1
                    nplots += 1
            print("Dropout current levels: ")
            print("-" * 20)
            # print(cell_df.keys())
            print("N, cell, expression, genotype, idrop")
            for i in np.argsort(dropout[1]):
                print(i,", ", cellids[i], ", ", expression[i], ", ", genotype[i], ", ", dropout[1, i])
            print("-" * 20)
            if iplot > 0:
                ax[1, 0].set_xlabel("Current (pA)")
                ax[1, 1].set_xlabel("Current (pA)")
                ax[0, 0].set_ylabel("Firing Rate (Hz) (1 sec)")
                ax[1, 0].set_ylabel("Firing Rate (Hz) (1st to last spike)")
                ax[0, 1].set_ylabel("Time of last spike (sec)")
                ax[1, 1].set_ylabel("Time of last spike < 0.8 sec (sec)")
                ax[1, 1].set_xlim(0, 1000.0)
                ax[1, 1].set_ylim(0, 1000.0)

                fig.suptitle(f"Firing analysis for {cell_df.cell_expression:s}")
                mpl.show()
            return P
        else:
            return None

    def average_FI(self, FI_Data_I_, FI_Data_FR_, max_current: float = 1.0e-9):
        if len(FI_Data_I_) > 0:
            try:
                FI_Data_I, FI_Data_FR = zip(*sorted(zip(FI_Data_I_, FI_Data_FR_)))
            except:
                raise ValueError("couldn't zip the data sets: ")
            if len(FI_Data_I) > 0:  # has data...
                print("averaging FI data")
                FI_Data_I_ = np.array(FI_Data_I)
                FI_Data_FR_ = np.array(FI_Data_FR)
                f1_index = np.where((FI_Data_I_ >= 0.0) & (FI_Data_I_ <= max_current))[
                    0
                ]  # limit to 1 nA, regardless
                FI_Data_I, FI_Data_FR, FI_Data_FR_Std, FI_Data_N = self.avg_group(
                    FI_Data_I_[f1_index], FI_Data_FR_[f1_index], ndim=FI_Data_I_.shape
                )
        return FI_Data_I, FI_Data_FR, FI_Data_FR_Std, FI_Data_N

    def avg_group(self, x, y, ndim=2):
        if ndim == 2:
            x = np.array([a for b in x for a in b])
            y = np.array([a for b in y for a in b])
        else:
            x = np.array(x)
            y = np.array(y)
        # x = np.ravel(x) # np.array(x)
        # y = np.array(y)
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
            ystd[np.where(xa == dupe)] = np.nanstd(y[np.where(x == dupe)])
            yn[np.where(xa == dupe)] = np.count_nonzero(~np.isnan(y[np.where(x == dupe)]))
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

    def compute_FI_Fits(
        self,
        experiment,
        df: pd.DataFrame,
        cell: str,
        protodurs: dict = None,
    ):
        CP("g", f"\n{'='*80:s}\nCell: {cell!s}, {df[df.cell_id==cell].cell_type.values[0]:s}")
        CP("g", f"     Group {df[df.cell_id==cell].Group.values[0]!s}")
        cell_group = df[df.cell_id == cell].Group.values[0]
        if (
            pd.isnull(cell_group) and len(df[df.cell_id == cell].Group.values) > 1
        ):  # incase first value is nan
            cell_group = df[df.cell_id == cell].Group.values[1]
        df_cell, df_tmp = filename_tools.get_cell(experiment, df, cell_id=cell)
        if df_cell is None:
            return None
        # print("    df_tmp group>>: ", df_tmp.Group.values)
        # print("    df_cell group>>: ", df_cell.keys())
        # print("compute_FI_Fits: ", df_tmp.keys())
        protocol_list = list(df_cell.Spikes.keys())
        # build protocols excluding removed protocols
        protocols = []
        day_slice_cell = str(Path(df_cell.date, df_cell.slice_slice, df_cell.cell_cell))
        for protocol in protocol_list:
            if not self.check_excluded_dataset(day_slice_cell, experiment, protocol):
                protocols.append(protocol)
        srs = {}
        dur = {}
        important = {}
        # for each CCIV type of protocol that was run:
        for nprot, protocol in enumerate(protocols):
            if protocol.endswith("0000"):  # bad protocol name
                continue
            day_slice_cell = str(Path(df_cell.date, df_cell.slice_slice, df_cell.cell_cell))
            CP("m", f"day_slice_cell: {day_slice_cell:s}, protocol: {protocol:s}")
            if self.check_excluded_dataset(day_slice_cell, experiment, protocol):
                continue
            # print(experiment["rawdatapath"], "\n  D: ", experiment["directory"], "\n  DSC: ", day_slice_cell, "\n  P: ", protocol)
            if str(experiment["rawdatapath"]).find(experiment["directory"]) == -1:
                fullpath = Path(experiment["rawdatapath"], experiment["directory"], protocol)
            else:
                fullpath = Path(experiment["rawdatapath"], protocol)
            with DR.acq4_reader.acq4_reader(fullpath, "MultiClamp1.ma") as AR:
                try:
                    if not AR.getData(fullpath):
                        continue
                    sample_rate = AR.sample_rate[0]
                    duration = AR.tend - AR.tstart
                    srs[protocol] = sample_rate
                    dur[protocol] = duration
                    important[protocol] = AR.checkProtocolImportant(fullpath)
                    CP("g", f"    Protocol {protocol:s} has sample rate of {sample_rate:e}")
                except ValueError:
                    CP("r", f"Acq4Read failed to read data file: {str(fullpath):s}")
                    raise ValueError(f"Acq4Read failed to read data file: {str(fullpath):s}")

        protocols = list(srs.keys())  # only count valid protocols
        CP("c", f"Valid Protocols: {protocols!s}")
        if len(protocols) > 1:
            protname = "combined"
        elif len(protocols) == 1:
            protname = protocols[0]
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
            "duration": dur,
            "currents": None,
            "firing_rates": None,
            "last_spikes": None,
        }

        # get the measures for the fixed values from the measure list
        for measure in datacols:
            datadict = self.get_measure(
                df_cell,
                measure,
                datadict,
                protocols,
                threshold_slope=experiment["AP_threshold_dvdt"],
            )
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
        protofails = 0
        # check the protocols
        for protocol in protocols:
            if protocol.endswith("0000"):  # bad protocol name
                continue
            short_proto_name = Path(protocol).name[:-4]
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
                    if not np.isclose(dur[protocol], duration):
                        durflag = True
                if durflag:
                    CP("y", f"    >>>> Protocol {protocol:s} has duration of {dur[protocol]:e}")
                    CP("y", f"               This is not in accepted limits of: {protodurs!s}")
                    continue
                else:
                    CP(
                        "g",
                        f"    >>>> Protocol {protocol:s} has acceptable duration of {dur[protocol]:e}",
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
            # get mean rate from second to last spike
            current = []
            rate = []
            last_spike = []
            # print("protocol: ", protocol)
            # print("# traces with spikes: ", len(df_cell.Spikes[protocol]['spikes']))
            # for k in df_cell.Spikes[protocol]['spikes'].keys():
            #     print(df_cell.Spikes[protocol]['spikes'][k].keys())
            # raise

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
                FI_Data_FR4_.extend(fidata[1] / dur[protocol])
            else:  # accumulate other protocols <= 1 nA
                FI_Data_I1_.extend(fidata[0])
                FI_Data_FR1_.extend(fidata[1] / dur[protocol])
                # accumulate this other information as well.
                firing_currents.extend(current)
                firing_rates.extend(rate)
                firing_last_spikes.extend(last_spike)

        FI_Data_I1 = []
        FI_Data_FR1 = []
        FI_Data_I4 = []
        FI_Data_FR4 = []
        if len(FI_Data_I1_) > 0:
            FI_Data_I1, FI_Data_FR1, FI_Data_FR1_Std, FI_Data_N1 = self.average_FI(
                FI_Data_I1_, FI_Data_FR1_, 1e-9
            )
        if len(FI_Data_I4_) > 0:
            FI_Data_I4, FI_Data_FR4, FI_Data_FR4_Std, FI_Data_N1 = self.average_FI(
                FI_Data_I4_, FI_Data_FR4_, 4e-9
            )
        if len(FI_Data_I1) > 0:
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

        # save the results
        datadict["FI_Curve1"] = [FI_Data_I1, FI_Data_FR1]
        datadict["FI_Curve4"] = [FI_Data_I4, FI_Data_FR4]
        datadict["current"] = FI_Data_I1
        datadict["spsec"] = FI_Data_FR1
        datadict["Subject"] = df_tmp.cell_id.values[0]
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
        bins = b = np.linspace(
            -25e-12, 1025e-12, 22, endpoint=True
        )  # 50 pA bins centered on 0, 50, 100, etc.
        # print("fc len: ", len(fc), "fr: ", len(fr), "fs: ", len(fs))
        # for i in range(len(fc)):
        #     print(fc[i], fr[i], fs[i])
        if len(fc) > 0:
            datadict["firing_currents"] = scipy.stats.binned_statistic(
                fc, fc, bins=bins, statistic=np.nanmean
            ).statistic
            datadict["firing_rates"] = scipy.stats.binned_statistic(
                fc, fr, bins=bins, statistic=np.nanmean
            ).statistic
            datadict["last_spikes"] = scipy.stats.binned_statistic(
                fc, fs, bins=bins, statistic=np.nanmean
            ).statistic
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
        measured_first_spike = False
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
            row.dvdt_current = currents[min_current] * 1e12  # put in pA
            row.AP_thr_V = 1e3 * dvdts[min_current].AP_begin_V
            if dvdts[min_current].halfwidth_interpolated is not None:
                row.AP_HW = dvdts[min_current].halfwidth_interpolated * 1e3
            row.AP_begin_V = 1e3 * dvdts[min_current].AP_begin_V
            CP(
                "y",
                f"I={currents[min_current]*1e12:6.1f} pA, dvdtRise={row.dvdt_rising:6.1f}, dvdtFall={row.dvdt_falling:6.1f}, APthr={row.AP_thr_V:6.1f} mV, HW={row.AP_HW*1e3:6.1f} usec",
            )
        return row

    def find_lowest_current_trace(self, spikes):
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

    def get_measure(self, df_cell, measure, datadict, protocols, threshold_slope: float = 20.0):
        """get_measure : for the given cell, get the measure from the protocols

        Parameters
        ----------
        df_cell : _type_
            _description_
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
        m = []
        if measure in iv_keys:
            for protocol in protocols:
                if measure in df_cell.IV[protocol].keys():
                    m.append(df_cell.IV[protocol][measure])

        elif measure in iv_mapper.keys() and iv_mapper[measure] in iv_keys:
            for protocol in protocols:
                if iv_mapper[measure] in df_cell.IV[protocol].keys():
                    m.append(df_cell.IV[protocol][iv_mapper[measure]])
        elif measure in spike_keys:
            maxadapt = 0
            for protocol in protocols:
                # print("p: ", p)
                if measure == "AdaptRatio":
                    if df_cell.Spikes[protocol][mapper1[measure]] > 8.0:
                        continue
                        # print("\nprot, measure: ", protocol, measure, df_cell.Spikes[protocol][mapper1[measure]])
                        # print(df_cell.Spikes[protocol].keys())
                        # maxadapt = np.max([maxadapt, df_cell.Spikes[protocol][mapper1['AdaptRatio']]])

                if measure in df_cell.Spikes[protocol].keys():
                    m.append(df_cell.Spikes[protocol][measure])
            # if maxadapt > 8:
            #     exit()

        elif measure in mapper1.keys() and mapper1[measure] in spike_keys:
            for protocol in protocols:
                if mapper1[measure] in df_cell.Spikes[protocol].keys():
                    m.append(df_cell.Spikes[protocol][mapper1[measure]])
        elif measure == "current":
            for protocol in protocols:  # for all protocols with spike analysis data for this cell
                if "spikes" not in df_cell.Spikes[protocol].keys():
                    continue
                # we need to get the first spike evoked by the lowest current level ...
                min_current_index, current, trace = self.find_lowest_current_trace(
                    df_cell.Spikes[protocol]
                )
                if not np.isnan(min_current_index):
                    m.append(current)
                else:
                    m.append(np.nan)

        else:
            for protocol in protocols:  # for all protocols with spike analysis data for this cell
                # we need to get the first spike evoked by the lowest current level ...
                prot_spike_count = 0
                if "spikes" not in df_cell.Spikes[protocol].keys():
                    continue
                spike_data = df_cell.Spikes[protocol]["spikes"]
                if measure in [
                    "dvdt_rising",
                    "dvdt_falling",
                    "AP_HW",
                    "AP_begin_V",
                    "AP_peak_V",
                    "AP_peak_T",
                    "AHP_trough_V",
                    "AHP_trough_T",
                    "AHP_depth_V",
                    "AHP_depth_T",
                ]:  # use lowest current spike
                    min_current_index, current, trace = self.find_lowest_current_trace(
                        df_cell.Spikes[protocol]
                    )
                    if not np.isnan(min_current_index):
                        spike_data = df_cell.Spikes[protocol]["spikes"][trace][0].__dict__
                        # print("spike data ", spike_data['dvdt_rising'])
                        m.append(spike_data[mapper[measure]])
                    else:
                        m.append(np.nan)
                    # print("spike data: ", spike_data.keys())

                elif "LowestCurrentSpike" in df_cell.Spikes[protocol].keys() and (
                    df_cell.Spikes[protocol]["LowestCurrentSpike"] is None
                    or len(df_cell.Spikes[protocol]["LowestCurrentSpike"]) == 0
                ):
                    CP("r", "Lowest Current spike is None")
                elif measure in ["AP_thr_V", "AP_begin_V"]:
                    if "LowestCurrentSpike" in df_cell.Spikes[protocol].keys():
                        if "AP_thr_V" in df_cell.Spikes[protocol]["LowestCurrentSpike"].keys():
                            Vthr = df_cell.Spikes[protocol]["LowestCurrentSpike"]["AP_thr_V"]
                            m.append(Vthr)
                        elif "AP_begin_V" in df_cell.Spikes[protocol]["LowestCurrentSpike"].keys():
                            Vthr = df_cell.Spikes[protocol]["LowestCurrentSpike"]["AP_begin_V"]
                            m.append(Vthr)

                        else:
                            print(
                                "Failed to find AP_begin_V/AP_thr_V in LowestCurrentSpike",
                                protocol,
                                measure,
                            )
                            print("LCS: ", df_cell.Spikes[protocol]["LowestCurrentSpike"].keys())
                            continue
                    else:
                        df_cell.Spikes[protocol][
                            "LowestCurrentSpike"
                        ] = None  # install value, but set to None
                        CP(
                            "r",
                            f"Missing lowest current spike data in spikes dictionary: {protocol:s}, {df_cell.Spikes[protocol].keys()!s}",
                        )

                elif measure in ["AP_thr_T"]:
                    if "LowestCurrentSpike" in df_cell.Spikes[protocol].keys():
                        Vthr_time = df_cell.Spikes[protocol]["LowestCurrentSpike"]["AP_thr_T"]
                        m.append(Vthr_time)
                    else:
                        print(df_cell.Spikes[protocol].keys())
                    # min_current_index, current, trace = self.find_lowest_current_trace(
                    #     df_cell.Spikes[protocol]
                    # )

                    # if not np.isnan(min_current_index):
                    #     spike_data = df_cell.Spikes[protocol]["spikes"][trace][0].__dict__
                    #     # CP("c", "Check AP_thr_V")

                    #     Vthr, Vthr_time = UTIL.find_threshold(
                    #         spike_data["V"],
                    #         np.mean(np.diff(spike_data["Vtime"])),
                    #         threshold_slope=threshold_slope,
                    #     )
                    #     m.append(Vthr)
                    # else:
                    #     m.append(np.nan)

                elif (
                    measure in mapper.keys() and mapper[measure] in spike_data.keys()
                ):  # if the measure exists for this sweep
                    m.append(spike_data[mapper[measure]])
                else:
                    # print(measure in mapper.keys())
                    # print(spike_data.keys())
                    CP(
                        "r",
                        f"measure <{measure:s}> not found in spike_data keys:, {mapper.keys()!s}",
                    )
                    CP(
                        "r",
                        f"\n   or mapped in {mapper[measure]!s} to {spike_data.keys()!s}",
                    )
                    raise ValueError()

                prot_spike_count += 1

        # CP("c", f"measure: {measure!s}  : {m!s}")
        # else:
        #     print(
        #         f"measure {measure:s} not found in either IV or Spikes keys. Skipping"
        #     )
        #     raise ValueError(f"measure {measure:s} not found in either IV or Spikes keys. Skipping")
        for i, xm in enumerate(m):
            if xm is None:
                m[i] = np.nan
            # m = [u for u in m if u is not None else np.nan] # sanitize data
        N = np.count_nonzero(~np.isnan(m))
        # print("N: ", N)
        if N > 0:
            datadict[measure] = np.nanmean(m)
        else:
            datadict[measure] = np.nan
        return datadict

    def textbox_setup(self, textbox):
        self.textbox = textbox

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
            return  # raise ValueError("datatables - functions - textbox has not been set up")

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
