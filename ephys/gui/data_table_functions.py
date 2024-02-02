""" General functions for data_tables and data_table_manager
    We are using a class here just to make it easier to pass around
"""

import logging
import pprint
import subprocess
from pathlib import Path
import re
from typing import Union

import matplotlib.pyplot as mpl
import numpy as np
import pandas as pd
from pylibrary.plotting import plothelpers as PH
from pylibrary.tools import cprint
from pyqtgraph.Qt import QtGui

import ephys.datareaders as DR
from ephys.ephys_analysis import spike_analysis
from ephys.tools import utilities
import ephys

UTIL = utilities.Utility()
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
    return {"project": git_head_hash, "ephys": ephys_git_hash}


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
    "AP_HW",
    "AP15Rate",
    "AdaptRatio",
    "AHP_trough_V",
    "AHP_depth_V",
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
    "AHP_Trough": "trough_T",
    "AHP_depth_V": "trough_V",
    "AP1_Latency": "AP_latency",
    "AP_thr_V": "AP_begin_V",
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
            if self.selected_index_rows is None:
                return None, None
            else:
                return self.selected_index_rows

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

    def get_selected_cell_data_spikes(self, experiment, table_manager, assembleddata):
        self.get_row_selection(table_manager)
        if self.selected_index_rows is not None:
            for nplots, index_row in enumerate(self.selected_index_rows):
                selected = table_manager.get_table_data(index_row)
                day = selected.date[:-4]
                slicecell = selected.cell_id[-4:]
                cell_df, cell_df_tmp = self.get_cell(experiment, assembleddata, cell=selected.cell_id)
                protocols = list(cell_df["Spikes"].keys())
                min_index = None
                min_current = 1
                V = None
                min_protocol = None
                spike = None
                for ip, protocol in enumerate(protocols):
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
                print("spike keys: ", spike["spikes"].keys())
                print(
                    "min I : ",
                    I,
                    "min V: ",
                    V,
                    "min index: ",
                    min_index,
                    "min_current: ",
                    min_current,
                )
                pp.pprint(spike["spikes"][V][min_index])
                low_spike = spike["spikes"][V][min_index]
                if nplots == 0:
                    import matplotlib.pyplot as mpl

                    f, ax = mpl.subplots(1, 2, figsize=(10, 5))
                vtime = (low_spike.Vtime - low_spike.peak_T) * 1e3
                ax[0].plot(vtime, low_spike.V * 1e3)
                ax[1].plot(low_spike.V * 1e3, low_spike.dvdt)
                dvdt_ticks = np.arange(-4, 2.01, 0.1)
                t_indices = np.array([np.abs(vtime - point).argmin() for point in dvdt_ticks])
                thr_index = np.abs(vtime - (low_spike.AP_latency - low_spike.peak_T) * 1e3).argmin()
                # Create a colormap
                cmap = mpl.get_cmap("tab10")
                # Create an array of colors based on the index of each point
                colors = cmap(np.linspace(0, 1, len(t_indices)))
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
                ax[1].scatter(
                    low_spike.V[thr_index] * 1e3,
                    low_spike.dvdt[thr_index],
                    s=12,
                    marker="o",
                    color="r",
                    zorder=12,
                )

                latency = (low_spike.AP_latency - low_spike.peak_T) * 1e3  # in msec
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

                if nplots == 0:  # annotate
                    ax[0].set_xlabel("Time (msec), re Peak")
                    ax[0].set_ylabel("V (mV)")
                    ax[1].set_xlabel("V (mV)")
                    ax[1].set_ylabel("dV/dt (mV/ms)")
                    PH.nice_plot(ax[0])
                    PH.nice_plot(ax[1])
                    PH.talbotTicks(ax[0])
                    PH.talbotTicks(ax[1])

                nplots += 1

            if nplots > 0:
                mpl.show()

            return cell_df
        else:
            return None

    def get_selected_cell_data_FI(self, experiment, table_manager, assembleddata):
        self.get_row_selection(table_manager)
        pp = PrettyPrinter(indent=4, width=120)
        if self.selected_index_rows is not None:
            for nplots, index_row in enumerate(self.selected_index_rows):
                selected = table_manager.get_table_data(index_row)
                day = selected.date[:-4]
                slicecell = selected.cell_id[-4:]
                # cell_df, _ = self.get_cell(
                #     experiment, assembleddata, cell=selected.cell_id
                # )
                fig, ax = mpl.subplots(1, 1)
                self.compute_FI_Fits(
                    experiment, assembleddata, selected.cell_id, plot_fits=True, ax=ax
                )

            if nplots > 0:
                mpl.show()
            return self.selected_index_rows
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
        plot_fits=False,
        ax: Union[mpl.Axes, None] = None,
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

        if plot_fits:
            if ax is None:
                fig, ax = mpl.subplots(1, 1)
                fig.suptitle(f"{celltype:s} {cell:s}")

            line_FI = ax.errorbar(
                np.array(FI_Data_I) * 1e9,
                FI_Data_FR,
                yerr=FI_Data_FR_Std,
                marker="o",
                color="k",
                linestyle=None,
            )
            # ax[1].plot(FI_Data_I * 1e12, FI_Data_N, marker="s")
            if plot_raw:
                for i, d in enumerate(FI_Data_I_):  # plot the raw points before combining
                    ax.plot(np.array(FI_Data_I_[i]) * 1e9, FI_Data_FR_[i], "x", color="k")
            # print("fit x * 1e9: ", spanalyzer.analysis_summary['FI_Growth'][0]['fit'][0]*1e9)
            # print("fit y * 1: ", spanalyzer.analysis_summary['FI_Growth'][0]['fit'][1])

            # ax[0].plot(linx * 1e12, liny, color="c", linestyle="dashdot")
            celln = Path(cell).name

            if len(spanalyzer.analysis_summary["FI_Growth"]) >= 0:
                line_fit = ax.plot(
                    spanalyzer.analysis_summary["FI_Growth"][0]["fit"][0][0] * 1e9,
                    spanalyzer.analysis_summary["FI_Growth"][0]["fit"][1][0],
                    color="r",
                    linestyle="-",
                    zorder=100,
                )
                # derivative (in blue)
                line_deriv = ax.plot(
                    i_range * 1e9, deriv_hill, color="b", linestyle="--", zorder=100
                )
                d_max = np.argmax(deriv_hill)
                ax2 = ax.twinx()
                ax2.set_ylim(0, 500)
                ax2.set_ylabel("Firing Rate Slope (sp/s/nA)")
                line_drop = ax2.plot(
                    [i_range[d_max] * 1e9, i_range[d_max] * 1e9],
                    [0, 1.1 * deriv_hill[d_max]],
                    color="b",
                    zorder=100,
                )
                ax.set_xlabel("Current (nA)")
                ax.set_ylabel("Firing Rate (sp/s)")
                # turn off top box
                for loc, spine in ax.spines.items():
                    if loc in ["left", "bottom"]:
                        spine.set_visible(True)
                    elif loc in ["right", "top"]:
                        spine.set_visible(False)
                for loc, spine in ax2.spines.items():
                    if loc in ["right", "bottom"]:
                        spine.set_visible(True)
                    elif loc in ["left", "top"]:
                        spine.set_visible(False)
                # spine.set_color('none')
                # do not draw the spine
                # spine.set_color('none')
                # do not draw the spine
                PH.talbotTicks(ax, density=[2.0, 2.0])
                PH.talbotTicks(ax2, density=[2.0, 2.0])
                ax.legend(
                    [line_FI, line_fit[0], line_deriv[0], line_drop[0]],
                    ["Firing Rate", "Hill Fit", "Derivative", "Max Derivative"],
                    loc="best",
                    frameon=False,
                )

            mpl.show()

        return hill_max_derivs, hill_i_max_derivs, FI_fits, linfits

    def check_excluded_dataset(self, day_slice_cell, experiment, protocol):
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

    def compute_FI_Fits(
        self,
        experiment,
        df: pd.DataFrame,
        cell: str,
        protodurs: list = [1.0],
        plot_fits: bool = False,
        ax: Union[mpl.Axes, None] = None,
    ):
        CP("g", f"\n{'='*80:s}\nCell: {cell!s}, {df[df.cell_id==cell].cell_type.values[0]:s}")

        df_cell, df_tmp = self.get_cell(experiment, df, cell)
        if df_cell is None:
            return None
        print("    df_tmp group>>: ", df_tmp.Group.values)
        print("    df_cell group>>: ", df_cell.keys())
        protocols = list(df_cell.Spikes.keys())
        spike_keys = list(df_cell.Spikes[protocols[0]].keys())
        iv_keys = list(df_cell.IV[protocols[0]].keys())

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
            fullpath = Path(experiment["rawdatapath"], experiment["directory"], protocol)
            with DR.acq4_reader.acq4_reader(fullpath, "MultiClamp1.ma") as AR:
                try:
                    AR.getData(fullpath)
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
        # parse group correctly.
        # the first point in the Group column is likely a nan.
        # if it is, then use the next point.
        print("Group: ", df_tmp.Group, "protoname: ", protname)
        group = df_tmp.Group.values[0]


        datadict = {
            "ID": str(df_tmp.cell_id.values[0]),
            "Subject": str(df_tmp.cell_id.values[0]),
            "cell_id": cell,
            "Group": group,
            "Date": str(df_tmp.Date.values[0]),
            "age": str(df_tmp.age.values[0]),
            "weight": str(df_tmp.weight.values[0]),
            "sex": str(df_tmp.sex.values[0]),
            "cell_type": df_tmp.cell_type.values[0],
            "protocol": protname,
            "important": important,
            "protocols": list(df_cell.IV),
            "sample_rate": srs,
            "duration": dur,
        }

        # get the measures for the fixed values from the measure list
        for measure in datacols:
            datadict = self.get_measure(df_cell, measure, datadict, protocols, threshold_slope=experiment["AP_threshold_dvdt"])
        # now combine the FI data across protocols for this cell
        FI_Data_I1_:list_ = []
        FI_Data_FR1_:list_ = []  # firing rate
        FI_Data_I4_:list_ = []
        FI_Data_FR4_:list_ = []  # firing rate
        FI_fits:dict = {"fits": [], "pars": [], "names": []}
        linfits:list = []
        hill_max_derivs:list = []
        hill_i_max_derivs:list = []
        protofails = 0
        for protocol in protocols:
            if protocol.endswith("0000"):  # bad protocol name
                continue
            # check if duration is acceptable:
            if protodurs is not None:
                durflag = False
                for d in protodurs:
                    if not np.isclose(dur[protocol], d):
                        durflag = True
                if durflag:
                    CP("y", f"    >>>> Protocol {protocol:s} has duration of {dur[protocol]:e}")
                    CP("y", f"               This is not in accepted limits of: {protodurs!s}")
                    continue
                else:
                    CP("g", f"    >>>> Protocol {protocol:s} has acceptable duration of {dur[protocol]:e}")
            # print("protocol: ", protocol, "spikes: ", df_cell.Spikes[protocol]['spikes'])
            if len(df_cell.Spikes[protocol]["spikes"]) == 0:
                CP("y", f"    >>>> Skipping protocol with no spikes:  {protocol:s}")
                continue
            else:
                CP("g", f"   >>>> Analyzing FI for protocol: {protocol:s}")
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
            if np.max(fidata[0]) > 1.01e-9:  # accumulate high-current protocols
                FI_Data_I4_.extend(fidata[0])
                FI_Data_FR4_.extend(fidata[1] / dur[protocol])
            else:  # accumulate other protocols <= 1 nA
                FI_Data_I1_.extend(fidata[0])
                FI_Data_FR1_.extend(fidata[1] / dur[protocol])

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
                plot_fits=plot_fits,
                ax=ax,
            )

        # save the results
        datadict["FI_Curve"] = [FI_Data_I1, FI_Data_FR1]
        datadict["FI_Curve4"] = [FI_Data_I4, FI_Data_FR4]
        datadict["current"] = FI_Data_I1
        datadict["spsec"] = FI_Data_FR1
        # datadict["Subject"] = df_tmp.cell_id.values[0]
        # datadict["Group"] = df_tmp.Group.values[0]
        # datadict["sex"] = df_tmp.sex.values[0]
        # datadict["celltype"] = df_tmp.cell_type.values[0]
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
        return datadict

    def get_cell(self, experiment, df: pd.DataFrame, cell: str):
        df_tmp = df[df.cell_id == cell] # df.copy() # .dropna(subset=["Date"])
        print("\nGet_cell:: df_tmp head: \n", "Groups: ", df_tmp["Group"].unique(), "\n len df_tmp: ", len(df_tmp))

        if len(df_tmp) == 0:
            return None, None
        try:
            celltype = df_tmp.cell_type.values[0]
        except ValueError:
            celltype = df_tmp.cell_type
        celltype = str(celltype).replace("\n", "")
        if celltype == " ":  # no cell type
            celltype = "unknown"
        CP("m", f"get cell: df_tmp cell type: {celltype:s}")
        # look for original PKL file for cell in the dataset
        # if it exists, use it to get the FI curve
        # base_cellname = str(Path(cell)).split("_")
        # print("base_cellname: ", base_cellname)
        # sn = int(base_cellname[-1][1])
        # cn = int(base_cellname[-1][3])
        # different way from cell_id:
        # The cell name may be a path, or just the cell name.
        # we have to handle both cases.

        parent = Path(cell).parent
        if parent == ".":  # just cell, not path
            cell_parts = str(cell).split("_")
            re_parse = re.compile("([Ss]{1})(\d{1,3})([Cc]{1})(\d{1,3})")
            cnp = re_parse.match(cell_parts[-1]).group(2)
            cn = int(cnp)
            snp = re_parse.match(cell_parts[-1]).group(4)
            sn = int(snp)
            cell_day_name = cell_parts[-3].split("_")[0]
            dir_path = None
        else:
            cell = Path(cell).name  # just get the name here
            cell_parts = cell.split("_")
            re_parse = re.compile("([Ss]{1})(\d{1,3})([Cc]{1})(\d{1,3})")
            # print("cell_parts: ", cell_parts[-1])
            snp = re_parse.match(cell_parts[-1]).group(2)
            sn = int(snp)
            cnp = re_parse.match(cell_parts[-1]).group(4)
            cn = int(cnp)
            cell_day_name = cell_parts[0]
            dir_path = parent

        # print("Cell name, slice, cell: ", cell_parts, sn, cn)
        # if cell_parts != ['2019.02.22', '000', 'S0C0']:
        #     return None, None
        cname2 = f"{cell_day_name.replace('.', '_'):s}_S{snp:s}C{cnp:s}_{celltype:s}_IVs.pkl"
        datapath2 = Path(experiment["analyzeddatapath"], experiment["directory"], celltype, cname2)
 
        # cname2 = f"{cell_day_name.replace('.', '_'):s}_S{sn:02d}C{cn:02d}_{celltype:s}_IVs.pkl"
        # datapath2 = Path(experiment["analyzeddatapath"], experiment["directory"], celltype, cname2)
        # cname1 = f"{cell_day_name.replace('.', '_'):s}_S{sn:01d}C{cn:01d}_{celltype:s}_IVs.pkl"
        # datapath1 = Path(experiment["analyzeddatapath"], experiment["directory"], celltype, cname1)
        # print(datapath)
        # if datapath1.is_file():
        #     CP("c", f"... {datapath1!s} is OK")
        #     datapath = datapath1
        if datapath2.is_file():
            CP("c", f"...  {datapath2!s} is OK")
            datapath = datapath2
        else:
            CP("r", f"no file: matching: {datapath2!s}, \n") #    or: {datapath2!s}\n")
            print("cell type: ", celltype)
            raise ValueError
            return None, None
        try:
            df_cell = pd.read_pickle(datapath, compression="gzip")
        except ValueError:
            try:
                df_cell = pd.read_pickle(datapath)  # try with no compression
            except ValueError:
                CP("r", f"Could not read {datapath!s}")
                raise ValueError("Failed to read compressed pickle file")
        if "Spikes" not in df_cell.keys() or df_cell.Spikes is None:
            CP(
                "r",
                f"df_cell: {df_cell.age!s}, {df_cell.cell_type!s}, No spike protos:",
            )
            return None, None
        # print(
        #     "df_cell: ",
        #     df_cell.age,
        #     df_cell.cell_type,
        #     "N spike protos: ",
        #     len(df_cell.Spikes),
        #     "\n",
        #     df_tmp['Group'],
        # )
        return df_cell, df_tmp

    def get_lowest_current_spike(self, row, SP):
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

    def get_measure(self, df_cell, measure, datadict, protocols, threshold_slope:float=20.0):
        """get_measure : for the giveen cell, get the measure from the protocols

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
                    "AHP_trough_V",
                    "AHP_depth_V",
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

                elif measure == "AP_thr_V":  # have to try two variants. Note that threshold slope is defined in config file
                    min_current_index, current, trace = self.find_lowest_current_trace(
                        df_cell.Spikes[protocol]
                    )
                    if not np.isnan(min_current_index):
                        spike_data = df_cell.Spikes[protocol]["spikes"][trace][0].__dict__
                        # CP("c", "Check AP_thr_V")

                        Vthr, Vthr_time = UTIL.find_threshold(
                            spike_data["V"],
                            np.mean(np.diff(spike_data["Vtime"])),
                            threshold_slope=threshold_slope,
                        )
                        m.append(Vthr)
                    else:
                        m.append(np.nan)

                elif (
                    measure in mapper.keys() and mapper[measure] in spike_data.keys()
                ):  # if the measure exists for this sweep
                    m.append(spike_data[mapper[measure]])
                else:
                    # print(measure in mapper.keys())
                    # print(spike_data.keys())
                    CP(
                        "r",
                        f"measure not found in spike_data, either: <{measure:s}>, {mapper.keys()!s}",
                    )
                    CP(
                        "r",
                        f"\n   or mapped in {mapper[measure]!s} to {spike_data.keys()!s}",
                    )
                    raise ValueError()
                    exit()
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
            raise ValueError("datatables - functions - textbox has not been set up")

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
