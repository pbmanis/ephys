"""
Compute IV from voltage clamp data.
Version 0.1, does only min negative peak IV, max pos IV and ss IV

"""

import gc
import logging
import re
from pathlib import Path
from typing import Union

import matplotlib
import matplotlib.colors
import matplotlib.pyplot as mpl
import MetaArray as EM
import numpy as np
import pylibrary.plotting.plothelpers as PH
import pyqtgraph as pg
from pylibrary.tools import cprint as CP

import ephys.tools.build_info_string as BIS
import ephys.tools.exp_estimator_lmfit as exp_estimator_lmfit
import ephys.tools.filename_tools as filename_tools
import ephys.tools.functions as functions
from ephys.datareaders.acq4_reader import acq4_reader
from ephys.ephys_analysis.analysis_common import Analysis
from ephys.tools import check_inclusions_exclusions as CIE

from ..datareaders import acq4_reader

color_sequence = ["k", "r", "b"]
colormap = "snshelix"


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


logging.getLogger("fontTools.subset").disabled = True
Logger = logging.getLogger("AnalysisLogger")
level = logging.DEBUG
Logger.setLevel(level)
# create file handler which logs even debug messages
logging_fh = logging.FileHandler(filename="logs/vc_analysis.log")
logging_fh.setLevel(level)
logging_sh = logging.StreamHandler()
logging_sh.setLevel(level)
log_formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s  (%(filename)s:%(lineno)d) - %(message)s "
)
logging_fh.setFormatter(log_formatter)
logging_sh.setFormatter(CustomFormatter())  # log_formatter)
Logger.addHandler(logging_fh)


class VCAnalysis(Analysis):

    Logger = logging.getLogger("AnalysisLogger")

    def __init__(self, args: Union[None, object] = None):
        if args is not None:
            super().__init__(args)
        self.reset_analysis()
        Logger.info("Instantiating VCAnalysis class")

    def __enter__(self):
        self.reset_analysis()
        return self

    def __exit__(self, type, value, traceback):
        del self.VCA
        gc.collect()

    def reset_analysis(self):
        self.mode = "acq4"
        self.AR: object = acq4_reader.acq4_reader()
        self.VCA: Union[object, None] = None
        self.allow_partial = False
        self.record_list = []

    def configure(
        self,
        datapath=None,
        altstruct=None,
        file: Union[str, Path, None] = None,
        experiment: Union[dict, None] = None,
        reader: Union[object, None] = None,
        plot: bool = True,
        pdf_pages: Union[object, None] = None,
    ):
        self.pdf_pages = pdf_pages
        if datapath is not None:
            self.AR = reader
            self.datapath = datapath
        else:
            self.AR = altstruct
            self.datapath = file
            self.mode = "nwb2.5"
        self.datapath = str(datapath)
        self.experiment = experiment
        self.plot = plot
        self.plotting_mode = "normal"
        self.decorate = True
        self.vcsummary = VCSummary(
            datapath=self.datapath,
            altstruct=self.AR,
            file=file,
            experiment=self.experiment,
            plot=self.plot,
        )
        self.vcsummary.setup(clamps=self.AR)
        self.analyze()
        # self..analysis_summary = {}  # dictionary holding analysis results: steady state IV, Ihold, rate fits, etc.

    def analyze(self):

        self.vcsummary.compute_vc()


class VCSummary:
    def __init__(
        self,
        datapath,
        altstruct=None,
        file: Union[str, Path, None] = None,
        experiment: Union[dict, None] = None,
        plot: bool = True,
    ):
        assert experiment is not None
        self.experiment = experiment
        self.datapath = datapath
        self.mode = "acq4"
        self.plot = plot

        if datapath is not None:
            self.AR = (
                acq4_reader.acq4_reader()
            )  # make our own private version of the analysis and reader
            self.datapath = datapath
        else:
            self.AR = altstruct
            self.datapath = file
            self.mode = "nwb2.5"

    def setup(self, clamps=None):
        """
        Set up for the fitting

        Parameters
        ----------
        clamps : A datamodel structure (required)
            Brings the data to the module. This usually will be a PatchEphys object.

        spikes : A spikeAnalysis structure (required)
            Has information about which traces have spikes

        dataplot : pyqtgraph plot object
            pyqtgraph plot to use for plotting the data. No data is plotted if not specified

        baseline : list (2 elements)
            times over which baseline is measured (in seconds)

        taumbounds : list (2 elements)
            Lower and upper bounds of the allowable taum fitting range (in seconds).
        """

        if clamps is None:
            raise ValueError("VC analysis requires defined clamps ")
        self.Clamps = clamps
        self.analysis_summary = {}

    def compute_vc(self):
        """
        Simple plot voltage clamp traces
        """
        # print('path: ', self.datapath)
        self.AR.setProtocol(self.datapath)  # define the protocol path where the data is
        self.setup(clamps=self.AR)
        if self.AR.getData():  # get that data.
            self.analyze_vc()
            # self.plot_vciv(self.analysis_summary)
            return True
        return False

    def analyze_vc(self):
        protocol = Path(self.AR.protocol).name
        protocol = protocol[:-4]
        proto_parameters = self.experiment["voltage_clamp_protocols"][protocol]
        if proto_parameters is None:
            raise ValueError(
                "VC analysis requires defined protocol parameters in the experiment configuration file"
            )
        self.ihold_region = proto_parameters.get("ihold_region", None)
        self.tau_activation_region = proto_parameters.get("tau_activation_region", None)
        self.tau_bounds = proto_parameters.get("tau_bounds", [0, 1e3])
        self.tau_exponential = proto_parameters.get("tau_exponential", 1.0)
        self.tau_voltage_range = proto_parameters.get("tau_voltage_range", [-np.inf, np.inf])
        self.tau_gap = proto_parameters.get(
            "tau_gap", 0.00
        )  # in seconds, time after step to start fitting (allow cap transient to settle)
        self.tail_region = proto_parameters.get("tail_region", [0.0, np.inf])
        self.tail_delay = proto_parameters.get("tail_delay", np.nan)  # gap for tail current analysis
        self.steady_state_activation_region = proto_parameters.get(
            "steady_state_activation_region", np.nan
        )

        assert (
            self.ihold_region is not None
        ), "rmp_region must be defined in the experiment configuration file"
        assert (
            self.tau_activation_region is not None
        ), "tau_region must be defined in the experiment configuration file"
        self.ihold_analysis(region=self.ihold_region)
        self.vcss_analysis(steady_state_activation_region=self.steady_state_activation_region)

        self.tau_analysis(measuretype="activation", result={})
        self.tau_analysis(measuretype="tail", result={})
        self.plot_vciv(self.analysis_summary)

    def ihold_analysis(self, region=None):
        """
        Get the holding current

        Parameters
        ----------
        region : tuple, list or numpy array with 2 values (default: None)
            start and end time of a trace used to measure the RMP across
            traces.

        Return
        ------
        Nothing

        Stores computed RMP in mV in the class variable rmp.
        """
        if region is None:
            raise ValueError(
                "VCSummary, ihold_analysis requires a region beginning and end to measure the RMP"
            )
        data1 = self.Clamps.traces["Time" : region[0] : region[1]]
        data1 = data1.view(np.ndarray)

        self.vcbaseline = data1.mean(axis=1)  # all traces
        self.vcbaseline_cmd = self.Clamps.commandLevels
        self.iHold = np.mean(self.vcbaseline) * 1e9  # convert to nA

    def vcss_analysis(self, steady_state_activation_region: list):
        """
        compute steady-state IV curve - from the mean current
        across the stimulus set over the defined time region
        (this usually will be the last half or third of the trace)

        Parameters
        ----------
        region : list or tuple
            Start and end times for the analysis
        """
        region = steady_state_activation_region
        data0 = self.Clamps.traces["Time" : self.Clamps.tstart : self.Clamps.tend]
        data0 = data0.view(np.ndarray)
        data0 = data0.reshape(
            self.Clamps.repetitions, int(data0.shape[0] / self.Clamps.repetitions), data0.shape[1]
        )
        data0 = data0.mean(axis=0)  # average across repetitions

        data1 = self.Clamps.traces["Time" : region[0] : region[1]]
        data1 = data1.view(np.ndarray).reshape(
            self.Clamps.repetitions, int(data1.shape[0] / self.Clamps.repetitions), data1.shape[1]
        )
        data1 = data1.mean(axis=0)

        cmds = self.Clamps.cmd_wave.view(np.ndarray)
        cmds = cmds.reshape(
            self.Clamps.repetitions, int(cmds.shape[0] / self.Clamps.repetitions), cmds.shape[1]
        )
        cmds = cmds.mean(axis=0)  # average across repetitions

        cmdlevels = self.Clamps.commandLevels.view(np.ndarray)
        cmdlevels = cmdlevels.reshape(
            (self.Clamps.repetitions, cmdlevels.shape[0] // self.Clamps.repetitions)
        )
        cmdlevels = cmdlevels.mean(axis=0)  # average across repetitions

        icmds = EM.MetaArray(
            cmds,  # easiest = turn in to a matching metaarray...
            info=[
                {"name": "Command", "units": "A", "values": cmdlevels},
                self.Clamps.traces.infoCopy("Time"),
                self.Clamps.traces.infoCopy(-1),
            ],
        )
        self.vcss_vcmd = icmds["Time" : region[0] : region[1]].mean(axis=1)
        self.r_in = np.nan
        self.analysis_summary["Rin"] = np.nan
        self.vcss_v = []
        if data1.shape[1] == 0 or data1.shape[0] == 1:
            return  # skip it

        ntr = len(self.Clamps.traces)

        self.vcss_Im = data1.mean(axis=1)  # steady-state, all traces
        self.vcpk_Im = data0.max(axis=1)
        self.vcmin_Im = data0.min(axis=1)
        self.analysis_summary["Rin"] = np.nan
        #        self.Clamps.plotClampData()

        isort = np.argsort(self.vcss_vcmd)
        self.vcss_Im = self.vcss_Im[isort]
        self.vcss_vcmd = self.vcss_vcmd[isort]
        bl = self.vcbaseline[isort]
        self.vcss_bl = bl
        # compute Rin from the SS IV:
        # this makes the assumption that:
        # successive trials are in order so we sort above
        # commands are not repeated...
        if len(self.vcss_vcmd) > 1 and len(self.vcss_v) > 1:
            pf = np.polyfit(
                self.vcss_vcmd,
                self.vcss_v,
                3,
                rcond=None,
                full=False,
                w=None,
                cov=False,
            )
            pval = np.polyval(pf, self.vcss_vcmd)
            # print('pval: ', pval)
            slope = np.diff(pval) / np.diff(self.vcss_vcmd)  # local slopes
            imids = np.array((self.vcss_vcmd[1:] + self.vcss_vcmd[:-1]) / 2.0)
            self.rss_fit = {"I": imids, "V": np.polyval(pf, imids)}
            # print('fit V: ', self.rss_fit['V'])
            # slope = slope[[slope > 0 ] and [self.vcss_vcmd[:-1] > -0.8] ] # only consider positive slope points
            l = int(len(slope) / 2)
            maxloc = np.argmax(slope[l:]) + l
            self.r_in = slope[maxloc]
            self.r_in_loc = [
                self.vcss_vcmd[maxloc],
                self.vcss_v[maxloc],
                maxloc,
            ]  # where it was found
            minloc = np.argmin(slope[:l])
            self.r_in_min = slope[minloc]
            self.r_in_minloc = [
                self.vcss_vcmd[minloc],
                self.vcss_v[minloc],
                minloc,
            ]  # where it was found
            self.Rin = self.r_in * 1.0e-6

    def tau_analysis(self, measuretype: str = "activation", result: dict = None):

        result["tauh_activation_region"] = self.tau_activation_region
        result["tauh_bounds"] = self.tau_bounds
        result["tau_exponential"] = self.tau_exponential
        result["tau_voltage_range"] = self.tau_voltage_range
        result["tau"] = []
        result["tau_gap"] = self.tau_gap
        result["tau_measuretype"] = measuretype
        result["tau_tail_region"] = self.tail_region
        result["tau_tail_delay"] = self.tail_delay
        debug = False
        # use an estimator from the average of the traces to be fit
        # also exclude traces with APs before the current step
        # limit whichdata to traces where icmd is betwwen 0 and -200 pA.
        # implemented 8/24/2024.
        c_range = np.sort(self.tau_voltage_range)  # make sure [0] is less than [1]

        fpar = []
        names = []
        okdata = []
        self.tau_fitted = {}
        debug = False

        traces = self.Clamps.traces["Time" : self.Clamps.tstart : self.Clamps.tend].view(np.ndarray)
        commands = self.Clamps.commandLevels.view(np.ndarray)
        if self.Clamps.repetitions > 1:
            traces = traces.reshape(
                self.Clamps.repetitions,
                int(traces.shape[0] / self.Clamps.repetitions),
                traces.shape[1],
            )

            traces = traces.mean(axis=0)  # average across repetitions
            commands = commands.reshape(
                (self.Clamps.repetitions, commands.shape[0] // self.Clamps.repetitions)
            )
            commands = commands.mean(axis=0)  # average across repetitions
            whichdata = np.argwhere(
                (commands + self.Clamps.holding >= c_range[0])
                & (commands + self.Clamps.holding <= c_range[1])
            )
            whichdata = whichdata.flatten()
        if debug:
            print("\n\nNot averaging: TAU estimates with traces: ", whichdata)
            print("c_range: ", c_range * 1e12)
            print(commands * 1e12)
            print((commands >= c_range[0]))
            print((commands <= c_range[1]))
            print((commands >= c_range[0]) & (commands <= c_range[1]))

            print("tau_voltage_range: ", self.tau_voltage_range)
            print([l for l in commands * 1e12])
            print("   shape of all trace data: ", traces.shape)

        if len(whichdata) == 0:
            if debug:
                print("No data available for taum measurement")
            self.analysis_summary = {}
            return  # no available taum data from this protocol.

        if measuretype == "activation":
            fit_result = self.do_fit(
                traces=traces,
                commands=commands,
                whichdata=whichdata,
                t_window=self.tau_activation_region,
                tau_gap=self.tau_gap,
                result=result,
                debug=False,
            )
        elif measuretype == "tail":
            fit_result = self.do_fit(
                traces=traces,
                commands=commands,
                whichdata=whichdata,
                t_window=self.tail_region,
                tau_gap=self.tail_delay,
                result=result,
                debug=False,
            )
        else:
            raise ValueError(f"Unknown measuretype: {measuretype}")
        for k in fit_result:
            self.analysis_summary[f"{k}_{measuretype}"] = fit_result[k]

    def do_fit(
        self,
        traces: np.ndarray,
        commands: np.ndarray,
        whichdata: list,
        t_window: list,
        tau_gap: float,
        result: dict,
        debug: bool = False,
    ):
        dt = self.Clamps.sample_interval
        time_base = self.Clamps.time_base.view(np.ndarray)
        # traces = self.Clamps.traces.view(np.ndarray)
        # print("Traces shape: ", traces.shape)
        # print(time_base.shape)
        # for i in range(traces.shape[0]):
        #     mpl.plot(time_base[:int(traces.shape[1])], traces[i, :])
        # mpl.show()
        # exit()
        mean_trace = np.mean(traces[whichdata], axis=0)
        debug = False
        igap = int(tau_gap / dt)
        it0 = int(t_window[0] / dt) + igap
        it1 = int(t_window[1] / dt)
        data_to_fit = mean_trace[0 : it1 - it0]
        t_fit = time_base[it0:it1]

        tau_func = "LME"  # use the LME estimator
        LME = exp_estimator_lmfit.LMexpFit()
        LME.initial_estimator(t_fit, data_to_fit, verbose=False)
        pw = []
        fpar = []
        tau_fitted = {}
        names = []
        okdata = []
        fparx = [np.nan, np.nan, np.nan]
        xf = None
        yf = None
        epsilon = self.Clamps.sample_interval * 3
        bounds = sorted(self.tau_bounds)
        self.data_fitted = {}
        for i, k in enumerate(whichdata):
            data_to_fit = traces[k][0 : it1 - it0]
            t_fit = time_base[it0:it1] - t_window[0]  # shift to zero
            if debug:  # plot raw traces
                if i == 0:
                    pwa = pg.plot(
                        t_fit + t_window[0],
                        data_to_fit,
                        pen=pg.mkPen(pg.intColor(i + 1), width=1),
                        title=f"Fitted Traces: {str(Path(*Path(self.Clamps.protocol).parts[-4:]))!s}",
                    )
                    pw.append(pwa)
            # update the estimates for the tau
            LME.initial_estimator(t_fit[igap:], data_to_fit[igap:], verbose=False)
            fit = LME.fit1(
                t_fit, data_to_fit, taum_bounds=self.tau_bounds, plot=False, verbose=False
            )
            self.data_fitted[i] = (
                t_fit + t_window[0] - tau_gap,
                data_to_fit,
            )  # save for plotting tails.
            fit_curve = LME.exp_decay1(
                t_fit,
                DC=fit.params["DC"].value,
                A1=fit.params["A1"].value,
                R1=fit.params["R1"].value,
            )
            if debug:  # now plot the fitted traces
                if i == 0:
                    pw[-1].plot(
                        x=t_fit + t_window[0],
                        y=fit_curve,
                        pen=pg.mkPen("r", width=2, linestyle="--"),
                    )

                else:
                    pw[-1].plot(
                        x=t_fit + t_window[0],
                        y=data_to_fit,
                        pen=pg.mkPen(pg.intColor(i + 1), width=1),
                        title=f"Fitted Traces: {self.Clamps.protocol!s}",
                    )
                    pw[-1].plot(
                        x=t_fit + t_window[0],
                        y=fit_curve,
                        pen=pg.mkPen("r", width=2, linestyle="--"),
                    )

            xf, yf = t_fit + t_window[0], fit_curve
            if (
                1.0 / fit.params["R1"].value < 0.0
            ):  # bounds on LME.fit1 should prevent this, but you never know
                print("Negative tau: ", 1.0 / fit.params["R1"].value)
                continue
            tau_k = 1.0 / fit.params["R1"].value
            # print("tau_k: ", tau_k, fit.params["R1"].value)
            # only accept the fit value if it is not at the boundaries.
            if tau_k < (bounds[0] + epsilon) or tau_k > (bounds[1] - epsilon):
                fparx = None
            else:
                fparx = [
                    fit.params["DC"].value,
                    fit.params["A1"].value,
                    1.0 / fit.params["R1"].value,
                ]
                namesx = ["DC", "A", "taum"]
                if fparx is None:
                    fpar.append([np.nan, np.nan, np.nan])
                else:
                    fpar.append(fparx)
                names.append(namesx)
                okdata.append(k)
                tau_fitted[k] = [xf, yf]

        taus = []
        for j in range(len(fpar)):
            taus.append(fpar[j][2])
        if len(taus) > 0:
            result["tau"] = taus
        else:
            result["tau"] = np.nan
        # if fit is against boundary, declare it invalid
        # if taus < (bounds[0] + epsilon) or taus > (bounds[1] - epsilon):
        #     result["tau"] = np.nan
        result["tauV"] = commands[whichdata] + self.Clamps.holding
        result["taupars"] = fpar
        result["taufunc"] = tau_func
        result["tau_fitted"] = tau_fitted
        result["tau_fitmode"] = "multiple"
        result["tau_traces"] = whichdata
        return result

    def plot_vciv(self, result: dict):
        if len(result) == 0:
            CP.cprint("No data to plot", color="red")
            return
        x = -0.05
        y = 1.05
        sizer = {
            "A": {"pos": [0.10, 0.51, 0.32, 0.60], "labelpos": (x, y)},
            "B": {"pos": [0.10, 0.51, 0.08, 0.15], "labelpos": (x, y)},
            "C": {"pos": [0.65, 0.30, 0.65, 0.20], "labelpos": (x, y)},
            "D": {"pos": [0.69, 0.30, 0.40, 0.20], "labelpos": (x, y)},
            "E": {"pos": [0.69, 0.30, 0.08, 0.20], "labelpos": (x, y)},
        }
        P = PH.arbitrary_grid(
            sizer,
            figsize=(8, 6),
        )
        (date, sliceid, cell, proto, p3) = self.file_cell_protocol(self.datapath)

        P.figure_handle.suptitle(str(Path(date, sliceid, cell, proto)), fontsize=12)
        for i in range(self.AR.traces.shape[0]):
            P.axdict["A"].plot(
                self.AR.time_base * 1e3,
                self.AR.traces[i, :].view(np.ndarray) * 1e12,
                c = "grey",
                linewidth=0.25,
                alpha=0.5,
            )
        # average traces and plot the average as a thicker line
        trace = self.AR.traces.view(np.ndarray)
        trace = trace.reshape(
            self.AR.repetitions, int(trace.shape[0] / self.AR.repetitions), trace.shape[1]
        )
        trace = trace.mean(axis=0)  # average across repetitions
        tb = self.AR.time_base.view(np.ndarray)[:trace.shape[1]]
        for i in range(trace.shape[0]):
            P.axdict["A"].plot(
                tb * 1e3,
                trace[i, :] * 1e12,
                "k-",
                linewidth=1.2,
            )

        # print("Tau fitted:\n", self.tau_fitted)
        tau_fitted = result["tau_fitted_activation"]
        for i, fitxy in enumerate(tau_fitted):
            if i in tau_fitted:
                xf, yf = tau_fitted[fitxy]
                P.axdict["A"].plot(
                    xf * 1e3,
                    yf * 1e12,
                    "r-",
                    linewidth=1.0,
                )
        # traces in A
        PH.nice_plot(
            P.axdict["A"], position={"left": -0.03, "y": -0.03}, direction="outward", ticklength=3
        )
        PH.talbotTicks(P.axdict["A"], tickPlacesAdd={"x": 0, "y": 0}, floatAdd={"x": 0, "y": 0})
        P.axdict["A"].set_xlabel("T (ms)")
        P.axdict["A"].set_ylabel("I (pA)")
        PH.crossAxes(P.axdict["C"], xyzero=(-60.0, 0.0))

        # crossed IV in C
        cmdv = (self.vcss_vcmd.view(np.ndarray)) * 1e3
        # print(self.vcss_vcmd.view(np.ndarray))
        # print(self.AR.holding)

        P.axdict["C"].plot(
            cmdv,
            self.vcss_Im.view(np.ndarray) * 1e12,
            "ks-",
            linewidth=1,
            markersize=2.5,
            label="Steady State IV",
        )
        P.axdict["C"].plot(
            cmdv,
            self.vcpk_Im.view(np.ndarray) * 1e12,
            "ro-",
            linewidth=1,
            markersize=4,
            label="Peak IV",
        )
        P.axdict["C"].plot(
            cmdv, self.vcmin_Im.view(np.ndarray) * 1e12, "b^-", linewidth=1, markersize=4
        )
        P.axdict["C"].set_xlabel("V (mV)")
        P.axdict["C"].set_ylabel("I (pA)")

        tau_fitted = result["tau_fitted_tail"]
        for i, fitxy in enumerate(self.data_fitted):
            if i in tau_fitted:
                xf, yf = self.data_fitted[fitxy]
                xf += result["tau_tail_delay_tail"]
                P.axdict["D"].plot(
                    xf * 1e3,
                    yf * 1e12,
                    "k-",
                    linewidth=1.0,
                )
        for i, fitxy in enumerate(tau_fitted):
            if i in tau_fitted:
                xf, yf = tau_fitted[fitxy]
                xf += result["tau_tail_delay_tail"]
                P.axdict["D"].plot(
                    xf * 1e3,
                    yf * 1e12,
                    "r-",
                    linewidth=0.75,
                )
        P.axdict["D"].legend(
            loc="upper left",
            fontsize=8,
        )
        # print(result.keys())
        # print(result['tau_activation'])
        # print(result['tauV_activation'])
        tm = len(result["tau_activation"])
        P.axdict["E"].plot(
            np.array(result["tauV_activation"])[:tm] * 1e3,
            np.array(result["tau_activation"]) * 1e3,
            "ks-",
            markersize=2.5,
        )
        P.axdict["E"].set_xlabel("V (mV)")
        P.axdict["E"].set_ylabel("tau (ms)")
        P.axdict["E"].set_ylim(0)
        PH.talbotTicks(P.axdict["C"], tickPlacesAdd={"x": 0, "y": 0}, floatAdd={"x": 0, "y": 0})

        # Voltage command B
        PH.nice_plot(
            P.axdict["B"], position={"left": -0.03, "y": -0.03}, direction="outward", ticklength=3
        )
        P.axdict["B"].set_xlabel("I (nA)")
        P.axdict["B"].set_ylabel("V (mV)")
        # PH.talbotTicks(
        #     P.axdict["B"], tickPlacesAdd={"x": 1, "y": 1}, floatAdd={"x": 2, "y": 1}
        # )
        P.axdict["B"].set_ylim(-120.0, 60.0)
        for i in range(self.AR.traces.shape[0]):
            P.axdict["B"].plot(
                self.AR.time_base * 1e3,
                self.AR.cmd_wave[i, :].view(np.ndarray) * 1e3,
                "k-",
                linewidth=0.5,
            )

        # something in D
        PH.nice_plot(
            P.axdict["D"], position={"left": -0.03, "y": -0.03}, direction="outward", ticklength=3
        )
        P.axdict["D"].set_xlabel("V (mV)")
        P.axdict["D"].set_ylabel("I (pA)")

        self.IVFigure = P.figure_handle

        if self.plot:
            mpl.show()

    # this should use filename_tools
    def file_cell_protocol(self, filename):
        """
        file_cell_protocol breaks the current filename down and returns a
        tuple: (date, cell, protocol)

        Parameters
        ----------
        filename : str
            Name of the protocol to break down

        Returns
        -------
        tuple : (date, sliceid, cell, protocol, any other...)
            last argument returned is the rest of the path...
        """
        fileparts = Path(filename).parts
        if len(fileparts) < 3:
            # probably a short version, nwb file.
            fileparts = filename.split("~")
            date = fileparts[0] + "_000"
            sliceid = f"slice_{int(fileparts[1][1]):03d}"
            cell = f"cell_{int(fileparts[1][3]):03d}"
            proto = fileparts[-1]
            p3 = ""
        else:
            proto = fileparts[-1]
            sliceid = fileparts[-2]
            cell = fileparts[-3]
            date = fileparts[-4]
            p3 = fileparts[:-4]
        return (date, sliceid, cell, proto, p3)


def concurrent_vc_analysis(
    vcanalysis: VCSummary,
    icell: int,
    i: int,
    x: int,
    cell_directory: Union[Path, str],
    validvcs: list,
    additional_vc_records: Union[dict, None] = None,
    nfiles: int = 0,
):
    result = vcanalysis.analyze_vc()
    return result
