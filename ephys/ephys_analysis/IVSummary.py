"""
Compute IV Information


"""
import gc
from collections import OrderedDict
from pathlib import Path
from typing import Union, Literal, List

import numpy as np
import pylibrary.plotting.plothelpers as PH
from pylibrary.tools import cprint

from ..datareaders import acq4_reader

from . import RmTauAnalysis, SpikeAnalysis

color_sequence = ["k", "r", "b"]
colormap = "snshelix"

CP = cprint.cprint

class IVSummary:
    def __init__(
        self, datapath, altstruct=None, file: Union[str, Path, None] = None, 
        plot:bool=True,
        pdf_pages:Union[object, None]=None,
    ):

        self.IVFigure = None
        self.mode = "acq4"
        self.pdf_pages = pdf_pages

        if datapath is not None:
            self.AR = (
                acq4_reader.acq4_reader()
            )  # make our own private version of the analysis and reader
            self.datapath = datapath
        else:
            self.AR = altstruct
            self.datapath = file
            self.mode = "nwb2.5"
        self.datapath = str(datapath)
        self.SP = SpikeAnalysis.SpikeAnalysis()
        self.RM = RmTauAnalysis.RmTauAnalysis()
        self.plot = plot
        self.plotting_mode = "normal"
        self.decorate = True
        self.plotting_alternation = 1

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        gc.collect()

    def iv_check(self, duration=0.0):
        """
        Check the IV for a particular duration, but does no analysis
        """
        if duration == 0:
            return True
        if self.mode == "acq4":
            self.AR.setProtocol(self.datapath)
        if self.AR.getData():
            dur = self.AR.tend - self.AR.tstart
            if np.fabs(dur - duration) < 1e-4:
                return True
        return False

    def plot_mode(
        self,
        mode: Literal["pubmode", "traces_only", "normal", None] = None,
        alternate: int = 1,
        decorate: bool = True,
    ):
        assert mode in ["pubmode", "traces_only", "normal"]
        self.plotting_mode = mode
        self.plotting_alternation = alternate
        self.decorate = decorate

    def stability_check(self, rmpregion:List=[0.0, 0.005], threshold = 2.0):
        """_summary_

        Args:
            rmpregion (List, optional): timw window over which to measure RMP. Defaults to [0.0, 0.005]. in seconds
            threshold (float, optional): Maximum SD of trace RMP. Defaults to 2.0. In mV

        Returns:
            bool: True if RMP is stable, False otherwise.
        """
        if self.mode == "acq4":
            self.AR.setProtocol(
                self.datapath
            )  # define the protocol path where the data is
        if self.AR.getData():  # get that data.
            self.RM.setup(self.AR, self.SP, bridge_offset=0.0)
        self.RM.rmp_analysis(time_window=rmpregion)
        # print(self.RM.rmp, self.RM.rmp_sd, threshold)
        if self.RM.rmp_sd > threshold:
            return False
        else:
            return True

    def compute_iv(
        self,
        threshold=-0.010,
        bridge_offset=0.0,
        tgap=0.0005,
        plotiv=False,
        full_spike_analysis=True,
        to_peak=True,
    ) -> Union[None, object]:
        """
        Simple plot of spikes, FI and subthreshold IV

        """
        if self.mode == "acq4":
            self.AR.setProtocol(
                self.datapath
            )  # define the protocol path where the data is
        if self.AR.getData():  # get that data.
            self.RM.setup(self.AR, self.SP, bridge_offset=bridge_offset)
            self.SP.setup(
                clamps=self.AR,
                threshold=threshold,
                refractory=0.0001,
                peakwidth=0.001,
                interpolate=True,
                verify=False,
                mode="schmitt",
            )
            self.SP.analyzeSpikes()
            if full_spike_analysis:
                self.SP.analyzeSpikeShape()
                # self.SP.analyzeSpikes_brief(mode="evoked")
                self.SP.analyzeSpikes_brief(mode="baseline")
                self.SP.analyzeSpikes_brief(mode="poststimulus")
            # self.SP.fitOne(function='fitOneOriginal')
            self.RM.analyze(
                rmpregion=[0.0, self.AR.tstart - 0.001],
                tauregion=[
                    self.AR.tstart,
                    self.AR.tstart + (self.AR.tend - self.AR.tstart) / 2.0,
                ],
                to_peak=to_peak,
                tgap=tgap,
            )
            if plotiv:
                fh = None
                if self.plotting_mode == "normal":
                    fh = self.plot_iv()
                elif self.plotting_mode == "pubmode":
                    fh = self.plot_iv(pubmode=True)
                elif self.plotting_mode == "traces_only":
                    fh = self.plot_fig()
                else:
                    raise ValueError(
                        "Plotting mode not recognized: ", self.plotting_mode
                    )
                return fh
        else:
            print(
                "IVSummary::compute_iv: acq4_reader.getData found no data to return from: \n  >  ",
                self.datapath,
            )
            return None

    def plot_iv(self, pubmode=False) -> Union[None, object]:
        if not self.plot:
            return None
        x = -0.08
        y = 1.02
        sizer = {
            "A": {"pos": [0.05, 0.50, 0.2, 0.71], "labelpos": (x, y), "noaxes": False},
            "A1": {
                "pos": [0.05, 0.50, 0.08, 0.05],
                "labelpos": (x, y),
                "noaxes": False,
            },
            "B": {"pos": [0.62, 0.30, 0.74, 0.17], "labelpos": (x, y), "noaxes": False},
            "C": {"pos": [0.62, 0.30, 0.52, 0.17], "labelpos": (x, y)},
            "D": {"pos": [0.62, 0.30, 0.30, 0.17], "labelpos": (x, y)},
            "E": {"pos": [0.62, 0.30, 0.08, 0.17], "labelpos": (x, y)},
        }
        # dict pos elements are [left, width, bottom, height] for the axes in the plot.
        gr = [
            (a, a + 1, 0, 1) for a in range(0, len(sizer))
        ]  # just generate subplots - shape does not matter
        axmap = OrderedDict(zip(sizer.keys(), gr))
        P = PH.Plotter((len(sizer), 1), axmap=axmap, label=True, figsize=(8.0, 10.0))
        # PH.show_figure_grid(P.figure_handle)
        P.resize(sizer)  # perform positioning magic

        if self.mode == "acq4":
            P.figure_handle.suptitle(self.datapath, fontsize=8)
        elif self.mode == "nwb":
            P.figure_handle.suptitle(str(self.datapath), fontsize=8)
        dv = 50.0
        jsp = 0
        # create a color map ans use it for all data in the plot
        # the map represents the index into the data array,
        # and is then mapped to the current levels in the trace
        # in case the sequence was randomized or run in a different
        # sequence
        import matplotlib.pyplot as mpl
        import colorcet
        
        # matplotlib versions:
        # cmap = mpl.colormaps['tab20'].resampled(self.AR.traces.shape[0])
        cmap = colorcet.cm.glasbey_bw_minc_20_maxl_70
        trace_colors = [cmap(float(i)/self.AR.traces.shape[0]) for i in range(self.AR.traces.shape[0])]

        # colorcet versions:

        for i in range(self.AR.traces.shape[0]):
            if self.plotting_alternation > 1:
                if i % self.plotting_alternation != 0:
                    continue
            if i in list(self.SP.spikeShape.keys()):
                idv = float(jsp) * dv
                jsp += 1
            else:
                idv = 0.0
            P.axdict["A"].plot(
                self.AR.time_base * 1e3,
                idv + self.AR.traces[i, :].view(np.ndarray) * 1e3,
                "-",
                color=trace_colors[i],
                linewidth=0.35,
            )
            P.axdict["A1"].plot(
                self.AR.time_base * 1e3,
                self.AR.cmd_wave[i, :].view(np.ndarray) * 1e9,
                "-",
                color=trace_colors[i],
                linewidth=0.35,
            )
            ptps = np.array([])
            paps = np.array([])
            if i in list(self.SP.spikeShape.keys()) and self.decorate:
                for j in list(self.SP.spikeShape[i].keys()):
                    paps = np.append(paps, self.SP.spikeShape[i][j]["peak_V"] * 1e3)
                    ptps = np.append(ptps, self.SP.spikeShape[i][j]["peak_T"] * 1e3)
                P.axdict["A"].plot(ptps, idv + paps, "ro", markersize=0.5)

            # mark spikes outside the stimlulus window
            if self.decorate:
                ptps = np.array([])
                paps = np.array([])
                for window in ["baseline", "poststimulus"]:
                    ptps = np.array(self.SP.analysis_summary[window + "_spikes"][i])
                    uindx = [int(u / self.AR.sample_interval) + 1 for u in ptps]
                    paps = np.array(self.AR.traces[i, uindx])
                    P.axdict["A"].plot(
                        ptps * 1e3, idv + paps * 1e3, "bo", markersize=0.5
                    )
        if not pubmode:
            for k in self.RM.taum_fitted.keys():
                # CP('g', f"tau fitted keys: {str(k):s}")
                if self.RM.tau_membrane == np.nan:
                    continue
                P.axdict["A"].plot(
                    self.RM.taum_fitted[k][0] * 1e3,  # ms
                    self.RM.taum_fitted[k][1] * 1e3,  # mV
                    "--g",
                    linewidth=1.0,
                )
            for k in self.RM.tauh_fitted.keys():
            # CP('r', f"tau fitted keys: {str(k):s}")
                if self.RM.tauh_meantau == np.nan:
                    continue
                P.axdict["A"].plot(
                    self.RM.tauh_fitted[k][0] * 1e3,
                    self.RM.tauh_fitted[k][1] * 1e3,
                    "--r",
                    linewidth=0.75,
                )
        if pubmode:
            PH.calbar(
                P.axdict["A"],
                calbar=[0.0, -90.0, 25.0, 25.0],
                axesoff=True,
                orient="left",
                unitNames={"x": "ms", "y": "mV"},
                fontsize=10,
                weight="normal",
                font="Arial",
            )
        P.axdict["B"].plot(
            self.SP.analysis_summary["FI_Curve"][0] * 1e9,
            self.SP.analysis_summary["FI_Curve"][1] / (self.AR.tend - self.AR.tstart),
            "grey",
            linestyle='-',
            # markersize=4,
            linewidth=0.5,
        )

        P.axdict["B"].scatter(
            self.SP.analysis_summary["FI_Curve"][0] * 1e9,
            self.SP.analysis_summary["FI_Curve"][1] / (self.AR.tend - self.AR.tstart),
            c=trace_colors,
            s=16,
            linewidth=0.5,
        )

        clist = ["r", "b", "g", "c", "m"]  # only 5 possiblities
        linestyle = ["-", "--", "-.", "-", "--"]
        n = 0
        for i, figrowth in enumerate(self.SP.analysis_summary["FI_Growth"]):
            legstr = "{0:s}\n".format(figrowth["FunctionName"])
            if len(figrowth["parameters"]) == 0:  # no valid fit
                P.axdict["B"].plot(
                    [np.nan, np.nan], [np.nan, np.nan], label="No valid fit"
                )
            else:
                for j, fna in enumerate(figrowth["names"][0]):
                    legstr += "{0:s}: {1:.3f} ".format(
                        fna, figrowth["parameters"][0][j]
                    )
                    if j in [2, 5, 8]:
                        legstr += "\n"
                P.axdict["B"].plot(
                    figrowth["fit"][0][0] * 1e9,
                    figrowth["fit"][1][0],
                    linestyle=linestyle[i % len(linestyle)],
                    color=clist[i % len(clist)],
                    linewidth=0.5,
                    label=legstr,
                )
            n += 1
        if n > 0:
            P.axdict["B"].legend(fontsize=6)

        P.axdict["C"].plot(
            np.array(self.RM.ivss_cmd) * 1e9,
            np.array(self.RM.ivss_v) * 1e3,
            "grey",
            # markersize=4,
            linewidth=1.0,
        )
        P.axdict["C"].scatter(
            np.array(self.RM.ivss_cmd) * 1e9,
            np.array(self.RM.ivss_v) * 1e3,
            c=trace_colors[0:len(self.RM.ivss_cmd)],
            s=16,
        )
        if not pubmode:

            if isinstance(self.RM.analysis_summary["CCComp"], float):
                enable = "Off"
                cccomp = 0.0
                ccbridge = 0.0
            elif self.RM.analysis_summary["CCComp"]["CCBridgeEnable"] == 1:
                enable = "On"
                cccomp = np.mean(
                    self.RM.analysis_summary["CCComp"]["CCPipetteOffset"] * 1e3
                )
                ccbridge = (
                    np.mean(self.RM.analysis_summary["CCComp"]["CCBridgeResistance"])
                    / 1e6
                )
            else:
                enable = "Off"
                cccomp = 0.0
                ccbridge = 0.0
            taum = r"$\tau_m$"
            tauh = r"$\tau_h$"
            tstr = f"RMP: {self.RM.analysis_summary['RMP']:.1f} mV\n"
            tstr += f"${{R_{{in}}}}$: {self.RM.analysis_summary['Rin']:.1f} ${{M\Omega}}$\n"
            tstr += f"{taum:s}: {self.RM.analysis_summary['taum']*1e3:.2f} ms\n"
            tstr += f"{tauh:s}: {self.RM.analysis_summary['tauh_tau']*1e3:.3f} ms\n"
            tstr += f"${{G_h}}$: {self.RM.analysis_summary['tauh_Gh'] *1e9:.3f} nS\n"
            tstr += f"Holding: {np.mean(self.RM.analysis_summary['Irmp']) * 1e12:.1f} pA\n"
            tstr += f"Bridge [{enable:3s}]: {ccbridge:.1f} ${{M\Omega}}$\n"
            tstr += f"Bridge Adjust: {self.RM.analysis_summary['BridgeAdjust']:.1f} ${{M\Omega}}$\n"
            tstr += f"Pipette: {cccomp:.1f} mV\n"

            P.axdict["C"].text(
                -0.05,
                0.80,
                tstr,
                transform=P.axdict["C"].transAxes,
                horizontalalignment="left",
                verticalalignment="top",
                fontsize=7,
            )
        #   P.axdict['C'].xyzero=([0., -0.060])
        PH.talbotTicks(
            P.axdict["A"], tickPlacesAdd={"x": 0, "y": 0}, floatAdd={"x": 0, "y": 0}
        )
        P.axdict["A"].set_xlabel("T (ms)")
        P.axdict["A"].set_ylabel("V (mV)")
        P.axdict["A1"].set_xlabel("T (ms)")
        P.axdict["A1"].set_ylabel("I (nV)")
        P.axdict["B"].set_xlabel("I (nA)")
        P.axdict["B"].set_ylabel("Spikes/s")
        PH.talbotTicks(
            P.axdict["B"], tickPlacesAdd={"x": 1, "y": 0}, floatAdd={"x": 2, "y": 0}
        )
        try:
            maxv = np.max(self.RM.ivss_v * 1e3)
        except:
            maxv = 0.0  # sometimes IVs do not have negative voltages for an IVss to be available...
        ycross = np.around(maxv / 5.0, decimals=0) * 5.0
        if ycross > maxv:
            ycross = maxv
        PH.crossAxes(P.axdict["C"], xyzero=(0.0, ycross))
        PH.talbotTicks(
            P.axdict["C"], tickPlacesAdd={"x": 1, "y": 0}, floatAdd={"x": 2, "y": 0}
        )
        P.axdict["C"].set_xlabel("I (nA)")
        P.axdict["C"].set_ylabel("V (mV)")

        """
        Plot the spike intervals as a function of time
        into the stimulus

        """
        for i in range(len(self.SP.spikes)):
            if len(self.SP.spikes[i]) == 0:
                continue
            spx = np.argwhere(
                (self.SP.spikes[i] > self.SP.Clamps.tstart)
                & (self.SP.spikes[i] <= self.SP.Clamps.tend)
            ).ravel()
            spkl = (
                np.array(self.SP.spikes[i][spx]) - self.SP.Clamps.tstart
            ) * 1e3  # just shorten...
            if len(spkl) == 1:
                P.axdict["D"].plot(spkl[0], spkl[0], "o", color=trace_colors[i], markersize=4)
            else:
                P.axdict["D"].plot(
                    spkl[:-1], np.diff(spkl), "o-",color=trace_colors[i], markersize=3, linewidth=0.5
                )

        PH.talbotTicks(
            P.axdict["C"], tickPlacesAdd={"x": 1, "y": 0}, floatAdd={"x": 1, "y": 0}
        )
        P.axdict["D"].set_yscale("log")
        P.axdict["D"].set_ylim((1.0, P.axdict["D"].get_ylim()[1]))
        P.axdict["D"].set_xlabel("Latency (ms)")
        P.axdict["D"].set_ylabel("ISI (ms)")
        P.axdict["D"].text(
            1.00,
            0.05,
            "Adapt Ratio: {0:.3f}".format(self.SP.analysis_summary["AdaptRatio"]),
            fontsize=9,
            transform=P.axdict["D"].transAxes,
            horizontalalignment="right",
            verticalalignment="bottom",
        )

        # phase plot
        # print(self.SP.spikeShape.keys())
        import matplotlib.pyplot as mpl

        # P.axdict["E"].set_prop_cycle('color',[mpl.cm.jet(i) for i in np.linspace(0, 1, len(self.SP.spikeShape.keys()))])
        for k, i in enumerate(self.SP.spikeShape.keys()):
            # print("ss i: ", i, self.SP.spikeShape[i][0])
            P.axdict["E"].plot(self.SP.spikeShape[i][0]["V"], self.SP.spikeShape[i][0]["dvdt"], linewidth=0.35,
            color = trace_colors[i])
        P.axdict["E"].set_xlabel("V (mV)")
        P.axdict["E"].set_ylabel("dV/dt (mv/ms)")

        self.IVFigure = P.figure_handle

        if self.plot:
            if self.pdf_pages is None:
                import matplotlib.pyplot as mpl
                # mpl.show()
            else:
                self.pdf_pages.savefig(dpi=300)
                import matplotlib.pyplot as mpl
                mpl.close()
        return P.figure_handle

    def plot_fig(self, pubmode=True):
        if not self.plot:
            return
        x = -0.08
        y = 1.02
        sizer = {
            "A": {"pos": [0.08, 0.82, 0.23, 0.7], "labelpos": (x, y), "noaxes": False},
            "A1": {"pos": [0.08, 0.82, 0.08, 0.1], "labelpos": (x, y), "noaxes": False},
        }
        # dict pos elements are [left, width, bottom, height] for the axes in the plot.
        gr = [
            (a, a + 1, 0, 1) for a in range(0, len(sizer))
        ]  # just generate subplots - shape does not matter
        axmap = OrderedDict(zip(sizer.keys(), gr))
        P = PH.Plotter((len(sizer), 1), axmap=axmap, label=True, figsize=(7.0, 5.0))
        # PH.show_figure_grid(P.figure_handle)
        P.resize(sizer)  # perform positioning magic
        P.figure_handle.suptitle(self.datapath, fontsize=8)
        dv = 0.0
        jsp = 0
        for i in range(self.AR.traces.shape[0]):
            if self.plotting_alternation > 1:
                if i % self.plotting_alternation != 0:
                    continue
            if i in list(self.SP.spikeShape.keys()):
                idv = float(jsp) * dv
                jsp += 1
            else:
                idv = 0.0
            P.axdict["A"].plot(
                self.AR.time_base * 1e3,
                idv + self.AR.traces[i, :].view(np.ndarray) * 1e3,
                "-",
                linewidth=0.35,
            )
            P.axdict["A1"].plot(
                self.AR.time_base * 1e3,
                self.AR.cmd_wave[i, :].view(np.ndarray) * 1e9,
                "-",
                linewidth=0.35,
            )
        for k in self.RM.taum_fitted.keys():
            P.axdict["A"].plot(
                self.RM.taum_fitted[k][0] * 1e3,
                self.RM.taum_fitted[k][1] * 1e3,
                "-g",
                linewidth=1.0,
            )
        for k in self.RM.tauh_fitted.keys():
            P.axdict["A"].plot(
                self.RM.tauh_fitted[k][0] * 1e3,
                self.RM.tauh_fitted[k][1] * 1e3,
                "--r",
                linewidth=0.750,
            )
        if pubmode:
            PH.calbar(
                P.axdict["A"],
                calbar=[0.0, -50.0, 50.0, 50.0],
                axesoff=True,
                orient="left",
                unitNames={"x": "ms", "y": "mV"},
                fontsize=10,
                weight="normal",
                font="Arial",
            )
            PH.calbar(
                P.axdict["A1"],
                calbar=[0.0, 0.05, 50.0, 0.1],
                axesoff=True,
                orient="left",
                unitNames={"x": "ms", "y": "nA"},
                fontsize=10,
                weight="normal",
                font="Arial",
            )
        self.IVFigure = P.figure_handle

        if self.plot:
            import matplotlib.pyplot as mpl

            mpl.show()
        return P.figure_handle
