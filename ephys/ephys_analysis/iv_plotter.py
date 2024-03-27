import datetime
from pathlib import Path
import pandas as pd
from typing import Union
import matplotlib.pyplot as mpl
import numpy as np
import pylibrary.plotting.plothelpers as PH
from pylibrary.tools.cprint import cprint as CP
import ephys.datareaders.acq4_reader as acq4_reader
from ephys.ephys_analysis.analysis_common import Analysis
import ephys.tools.build_info_string as BIS
import ephys.tools.filename_tools as filename_tools
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.figure
import colorcet


class IVPlotter(object):
    def __init__(
        self,
        experiment:dict,
        df_summary:pd.DataFrame,
        file_out_path: Union[Path, str],
        decorate=True,
    ):
        self.df_summary = df_summary
        self.experiment = experiment
        self.file_out_path = file_out_path
        self.decorate = decorate
        self.IVFigure = None
        self.plotting_alternation = 1
        self.downsample = 1
        self.nfiles = 0

    def finalize_plot(self, plot_handle, protocol_directory, pdf, acq4reader):
        """finalize_plot Generate each iv plot in the temporary directory

        Parameters
        ----------
        plot_handle : _type_
            _description_
        """
        if plot_handle is not None:
            shortpath = Path(protocol_directory).parts
            shortpath2 = str(Path(*shortpath[4:]))
            plot_handle.suptitle(
                f"{str(shortpath2):s}\n{BIS.build_info_string(acq4reader, protocol_directory):s}",
                fontsize=8,
            )

            pdf.savefig()  # t_path, dpi=300)  # use the map filename, as we will sort by this later
            mpl.close()  # (plot_handle)
            msg = f"Plotted {protocol_directory!s}"
            CP("g", msg)
            self.nfiles += 1

    def plot_ivs(self, df_selected=None):
        if df_selected is None:
            return

        celltype = df_selected["cell_type"]
        celltype = filename_tools.check_celltype(celltype)

        index = self.df_summary.index[self.df_summary["cell_id"] == df_selected.cell_id].values[
            0
        ]  # find the cell in the main index
        # print("index: ", index)
        datestr, slicestr, cellstr = filename_tools.make_cell(icell=index, df=self.df_summary)
        slicecell = filename_tools.make_slicecell(slicestr, cellstr)
        matchcell, slicecell3, slicecell2, slicecell1 = filename_tools.compare_slice_cell(
            slicecell=slicecell, datestr=datestr, slicestr=slicestr, cellstr=cellstr
        )
        if not matchcell:
            return False
        # print("datestr, clicecel, slice, cell, celltype: ", datestr, slicecell, slicestr, cellstr, df_selected['cell_type'])
        thisday = datestr.replace(".", "_").split("_")
        thisday = "_".join(thisday[:-1])

        self.plot_df, _tmp = filename_tools.get_cell(
            experiment=self.experiment, df=self.df_summary, cell_id=df_selected.cell_id
        )
        if self.plot_df is None: # likely no spike or IV protocols for this cell
            return
        
        self.nfiles = 0
        print("self.file_out_path: ", self.file_out_path)
        print("now calling make_pdf_filename")
        pdffile = filename_tools.make_pdf_filename(
            self.file_out_path,
            thisday=thisday,
            celltype=df_selected["cell_type"],
            analysistype="IVs",
            slicecell=slicecell,
        )

        with PdfPages(Path(pdffile)) as pdf:
            for iv in self.plot_df["IV"].keys():
                protodir = Path(self.file_out_path, iv)
                plot_handle, acq4 = self.plot_one_iv(iv)
                self.finalize_plot(
                    plot_handle, protocol_directory=protodir, pdf=pdf, acq4reader=acq4
                )

    def plot_one_iv(self, protocol, pubmode=False) -> Union[None, object]:
        spikes = self.plot_df["Spikes"][protocol]["spikes"]
        spike_dict = self.plot_df["Spikes"][protocol]
        ivs = self.plot_df["IV"][protocol]
        # print("spike keys in protocol: ", spike_dict.keys()) # get keys for protocol
        # print("iv keys in protocol: ", ivs.keys())

        # build plot layout
        x = -0.08  # label position offsets
        y = 1.02
        sizer = {
            "A": {"pos": [0.05, 0.50, 0.2, 0.71], "labelpos": (x, y), "noaxes": False},
            "A1": {
                "pos": [0.05, 0.50, 0.08, 0.05],
                "labelpos": (x, y),
                "noaxes": False,
            },
            "B": {"pos": [0.62, 0.30, 0.74, 0.17], "labelpos": (x, y), "noaxes": False},
            "C": {"pos": [0.70, 0.22, 0.52, 0.17], "labelpos": (x, y)},
            "D": {"pos": [0.62, 0.30, 0.30, 0.17], "labelpos": (x, y)},
            "E": {"pos": [0.62, 0.30, 0.08, 0.17], "labelpos": (x, y)},
        }
        # dict pos elements are [left, width, bottom, height] for the axes in the plot.
        gr = [
            (a, a + 1, 0, 1) for a in range(0, len(sizer))
        ]  # just generate subplots - shape does not matter
        axmap = dict(zip(sizer.keys(), gr))
        P = PH.Plotter((len(sizer), 1), axmap=axmap, label=True, figsize=(8.0, 10.0))
        # PH.show_figure_grid(P.figure_handle)
        P.resize(sizer)  # perform positioning magic
        now = datetime.datetime.now().strftime("%Y-%m-%d  %H:%M:%S %z")
        P.axdict["A"].text(
            0.96,
            0.01,
            s=now,
            fontsize=6,
            ha="right",
            transform=P.figure_handle.transFigure,
        )

        self.AR = acq4_reader.acq4_reader(Path(self.plot_df["data_directory"], protocol))
        infostr = BIS.build_info_string(self.AR, self.AR.protocol)

        P.figure_handle.suptitle(f"{str(Path(self.plot_df["data_directory"], protocol)):s}\n{infostr:s}", fontsize=8)
        dv = 50.0
        jsp = 0

        if not self.AR.getData():
            return None
        # create a color map ans use it for all data in the plot
        # the map represents the index into the data array,
        # and is then mapped to the current levels in the trace
        # in case the sequence was randomized or run in a different
        # sequence
        # matplotlib versions:
        # cmap = mpl.colormaps['tab20'].resampled(self.AR.traces.shape[0])
        cmap = colorcet.cm.glasbey_bw_minc_20_maxl_70
        trace_colors = [
            cmap(float(i) / self.AR.traces.shape[0]) for i in range(self.AR.traces.shape[0])
        ]

        for i in range(self.AR.traces.shape[0]):
            if self.plotting_alternation > 1:
                if i % self.plotting_alternation != 0:
                    continue

            if i in list(spikes.keys()):
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

            # mark spikes inside the stimulus window
            ptps = np.array([])
            paps = np.array([])
            if i in list(spikes.keys()) and self.decorate:
                for j in list(spikes[i].keys()):
                    paps = np.append(paps, spikes[i][j].peak_V * 1e3)
                    ptps = np.append(ptps, spikes[i][j].peak_T * 1e3)
                    # print("spikes ij ", i, j, spikes[i][j])
                P.axdict["A"].plot(ptps, idv + paps, "ro", markersize=0.5)

            # mark spikes outside the stimlulus window if we ask for them
            if self.decorate:
                ptps = np.array([])
                paps = np.array([])
                clist = ["g", "b"]
                windows = ["baseline_spikes", "poststimulus_spikes"]  #
                #    : [0.0, self.AR.tstart],
                #            'poststimulus': [self.AR.tend, np.max(self.AR.time_base)]}
                for k, window in enumerate(windows):
                    ptps = spike_dict[window]
                    # print("ptps: ", ptps)
                    for ispt, spt in enumerate(ptps):
                        if len(spt) == 0:
                            continue
                        uindx = [int(u / self.AR.sample_interval) + 1 for u in spt]
                        paps = np.array(self.AR.traces[ispt, uindx])
                        P.axdict["A"].plot(
                            np.array(spt) * 1e3,
                            idv + paps * 1e3,
                            "o",
                            color=clist[k],
                            markersize=0.5,
                        )
        if not pubmode:
            if "taum_fitted" not in ivs.keys():
                print(ivs.keys())
            if ivs["taum"] != np.nan and "taum_fitted" in ivs.keys():
                for k in ivs["taum_fitted"].keys():
                    # CP('g', f"tau fitted keys: {str(k):s}")
                    P.axdict["A"].plot(
                        ivs["taum_fitted"][k][0] * 1e3,  # ms
                        ivs["taum_fitted"][k][1] * 1e3,  # mV
                        "--g",
                        linewidth=1.0,
                    )
            if ivs["tauh_tau"] != np.nan and "tauh_fitted" in ivs.keys():
                for k in ivs["tauh_fitted"].keys():
                    # CP('r', f"tau fitted keys: {str(k):s}")
                    P.axdict["A"].plot(
                        ivs["tauh_fitted"][k][0] * 1e3,
                        ivs["tauh_fitted"][k][1] * 1e3,
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
            spike_dict["FI_Curve"][0] * 1e9,
            spike_dict["FI_Curve"][1] / (self.AR.tend - self.AR.tstart),
            "grey",
            linestyle="-",
            # markersize=4,
            linewidth=0.5,
        )

        P.axdict["B"].scatter(
            spike_dict["FI_Curve"][0] * 1e9,
            spike_dict["FI_Curve"][1] / (self.AR.tend - self.AR.tstart),
            c=trace_colors,
            s=16,
            linewidth=0.5,
        )

        clist = ["r", "b", "g", "c", "m"]  # only 5 possiblities
        linestyle = ["-", "--", "-.", "-", "--"]
        n = 0
        for i, figrowth in enumerate(spike_dict["FI_Growth"]):
            legstr = "{0:s}\n".format(figrowth["FunctionName"])
            if len(figrowth["parameters"]) == 0:  # no valid fit
                P.axdict["B"].plot([np.nan, np.nan], [np.nan, np.nan], label="No valid fit")
            else:
                for j, fna in enumerate(figrowth["names"][0]):
                    legstr += "{0:s}: {1:.3f} ".format(fna, figrowth["parameters"][0][j])
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
        if "ivss_cmd" not in ivs.keys():
            print("\nNo ivss_cmd found in the iv keys")
            print(self.plot_df["cell_id"], protocol)
            print("\n", ivs, "\n")
            print(ivs.keys())

        else:
            P.axdict["C"].plot(
                np.array(ivs["ivss_cmd"]) * 1e9,
                np.array(ivs["ivss_v"]) * 1e3,
                "grey",
                # markersize=4,
                linewidth=1.0,
            )
            P.axdict["C"].scatter(
                np.array(ivs["ivss_cmd"]) * 1e9,
                np.array(ivs["ivss_v"]) * 1e3,
                c=trace_colors[0 : len(ivs["ivss_cmd"])],
                s=16,
            )
        if not pubmode:
            if isinstance(ivs["CCComp"], float):
                enable = "Off"
                cccomp = 0.0
                ccbridge = 0.0
            elif ivs["CCComp"]["CCBridgeEnable"] == 1:
                enable = "On"
                cccomp = np.mean(ivs["CCComp"]["CCPipetteOffset"] * 1e3)
                ccbridge = np.mean(ivs["CCComp"]["CCBridgeResistance"]) / 1e6
            else:
                enable = "Off"
                cccomp = 0.0
                ccbridge = 0.0
            taum = r"$\tau_m$"
            tauh = r"$\tau_h$"
            tstr = f"RMP: {ivs['RMP']:.1f} mV\n"
            omega = r"$\Omega$"
            tstr += f"${{R_{{in}}}}$: {ivs['Rin']:.1f} M{omega:s}\n"
            tstr += f"SR: {self.AR.sample_rate[0] / 1e3:.1f} kHz\n"
            tstr += f"Downsample: {self.downsample:d}\n"
            tstr += f"${{Pip Cap}}$: {ivs['CCComp']['CCNeutralizationCap']*1e12:.2f} pF\n"
            tstr += f"{taum:s}: {ivs['taum']*1e3:.2f} ms\n"
            tstr += f"{tauh:s}: {ivs['tauh_tau']*1e3:.3f} ms\n"
            tstr += f"${{G_h}}$: {ivs['tauh_Gh'] *1e9:.3f} nS\n"
            tstr += f"Holding: {np.mean(ivs['Irmp']) * 1e12:.1f} pA\n"
            tstr += f"Bridge [{enable:3s}]: {ccbridge:.1f} M{omega:s}\n"
            tstr += f"Bridge Adjust: {ivs['BridgeAdjust']:.1f} m{omega:s}\n"
            tstr += f"Pipette: {cccomp:.1f} mV\n"

            P.axdict["C"].text(
                -0.60,
                0.85,
                tstr,
                transform=P.axdict["C"].transAxes,
                horizontalalignment="left",
                verticalalignment="top",
                fontsize=7,
            )
        #   P.axdict['C'].xyzero=([0., -0.060])
        PH.talbotTicks(P.axdict["A"], tickPlacesAdd={"x": 0, "y": 0}, floatAdd={"x": 0, "y": 0})
        P.axdict["A"].set_xlabel("T (ms)")
        P.axdict["A"].set_ylabel("V (mV)")
        P.axdict["A1"].set_xlabel("T (ms)")
        P.axdict["A1"].set_ylabel("I (nV)")
        P.axdict["B"].set_xlabel("I (nA)")
        P.axdict["B"].set_ylabel("Spikes/s")
        PH.talbotTicks(P.axdict["B"], tickPlacesAdd={"x": 1, "y": 0}, floatAdd={"x": 2, "y": 0})
        maxv = 0.0
        if "ivss_v" in ivs.keys():
            if len(ivs["ivss_v"]) > 0:
                maxv = np.max(ivs["ivss_v"] * 1e3)

            ycross = np.around(maxv / 5.0, decimals=0) * 5.0
            if ycross > maxv:
                ycross = maxv
            PH.crossAxes(P.axdict["C"], xyzero=(0.0, ycross))
        PH.talbotTicks(P.axdict["C"], tickPlacesAdd={"x": 1, "y": 0}, floatAdd={"x": 2, "y": 0})
        P.axdict["C"].set_xlabel("I (nA)")
        P.axdict["C"].set_ylabel("V (mV)")

        """
        Plot the spike intervals as a function of time
        into the stimulus

        """

        for i, spike_tr in enumerate(spikes):  # this is the trace number
            spike_train = spikes[spike_tr]  # get teh spike train, then get just the latency
            spk_tr = np.array([spike_train[sp].AP_latency for sp in spike_train.keys()])

            if len(spk_tr) == 0:
                continue
            spx = np.argwhere(  # get the spikes that are in the stimulus window
                (spk_tr > self.AR.tstart) & (spk_tr <= self.AR.tend)
            ).ravel()
            spkl = (np.array(spk_tr[spx]) - self.AR.tstart) * 1e3  # just shorten...
            if len(spkl) == 1:
                P.axdict["D"].plot(spkl[0], spkl[0], "o", color=trace_colors[i], markersize=4)
            else:
                P.axdict["D"].plot(
                    spkl[:-1],
                    np.diff(spkl),
                    "o-",
                    color=trace_colors[i],
                    markersize=3,
                    linewidth=0.5,
                )

        PH.talbotTicks(P.axdict["C"], tickPlacesAdd={"x": 1, "y": 0}, floatAdd={"x": 1, "y": 0})
        P.axdict["D"].set_yscale("log")
        P.axdict["D"].set_ylim((1.0, P.axdict["D"].get_ylim()[1]))
        P.axdict["D"].set_xlabel("Latency (ms)")
        P.axdict["D"].set_ylabel("ISI (ms)")
        P.axdict["D"].text(
            1.00,
            0.05,
            "Adapt Ratio: {0:.3f}".format(self.plot_df["Spikes"][protocol]["AdaptRatio"]),
            fontsize=9,
            transform=P.axdict["D"].transAxes,
            horizontalalignment="right",
            verticalalignment="bottom",
        )

        # phase plot
        # P.axdict["E"].set_prop_cycle('color',[mpl.cm.jet(i) for i in np.linspace(0, 1, len(self.SP.spikeShapes.keys()))])
        for k, i in enumerate(spikes.keys()):
            n_dvdt = len(spikes[i][0].dvdt)
            P.axdict["E"].plot(
                spikes[i][0].V[:n_dvdt],
                spikes[i][0].dvdt,
                linewidth=0.35,
                color=trace_colors[i],
            )
            if i == 0:
                break  # only plot the first one
        P.axdict["E"].set_xlabel("V (mV)")
        P.axdict["E"].set_ylabel("dV/dt (mv/ms)")

        self.IVFigure = P.figure_handle
        return P.figure_handle, self.AR

    def plot_fig(self, pubmode=True):
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
        axmap = dict(zip(sizer.keys(), gr))
        P = PH.Plotter((len(sizer), 1), axmap=axmap, label=True, figsize=(7.0, 5.0))
        # PH.show_figure_grid(P.figure_handle)
        P.resize(sizer)  # perform positioning magic
        infostr = BIS.build_info_string(self.AR, self.AR.protocol)
        P.figure_handle.suptitle(
            f"{str(Path(self.plot_df["data_directory"], protocol)):s}\n{infostr:s}",
            fontsize=8,
        )
        dv = 0.0
        jsp = 0
        for i in range(self.AR.traces.shape[0]):
            if self.plotting_alternation > 1:
                if i % self.plotting_alternation != 0:
                    continue
            if i in list(self.plot_df["Spikes"][protocol].keys()):
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
        mpl.show()
        return P.figure_handle
