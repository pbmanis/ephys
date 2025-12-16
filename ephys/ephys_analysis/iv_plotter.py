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
import ephys.tools.map_cell_types as map_cell_types
from ephys.tools import check_inclusions_exclusions as CIE
import ephys.gui.data_table_functions as data_table_functions
from matplotlib.backends.backend_pdf import PdfPages
import colorcet

FUNCS = data_table_functions.Functions()


def concurrent_iv_plotting(pkl_file, experiment, df_summary, file_out_path, decorate):
    print("concurrent iv plotting: pkl_file: ", pkl_file)
    with open(pkl_file, "rb") as fh:
        df_selected = pd.read_pickle(fh, compression="gzip")
        plotter = IVPlotter(
            experiment=experiment,
            df_summary=df_summary,
            file_out_path=file_out_path,
            decorate=decorate,
        )
        # print("Plotting for: ", df_selected)
        plotter.plot_IVs(df_selected)


class IVPlotter(object):
    def __init__(
        self,
        experiment: dict,
        df_summary: pd.DataFrame,
        file_out_path: Union[Path, str],
        decorate=True,
        allow_partial: bool = False,
        record_list: list = [],
    ):
        self.df_summary = df_summary
        self.experiment = experiment
        self.file_out_path = file_out_path
        self.decorate = decorate
        self.IVFigure = None
        self.IV_pdf_file = None
        self.plotting_alternation = 1
        self.downsample = 1
        self.nfiles = 0
        self.allow_partial = allow_partial
        self.record_list = record_list

    def finalize_plot(self, plot_handle, protocol_directory, pdf, acq4reader):
        """finalize_plot Generate each iv plot in the temporary directory

        Parameters
        ----------
        plot_handle : _type_
            _description_
        """
        if plot_handle is not None:
            datadir = Path(self.plot_df["cell_id"], Path(protocol_directory).name)
            shortpath = Path(protocol_directory).parts
            shortpath2 = str(Path(*shortpath[5:]))
            infostr = BIS.build_info_string(experiment=self.experiment, 
                                            cell_id=self.plot_df['cell_id'], 
                                            AR=acq4reader, 
                                           )
            plot_handle.suptitle(
                f"{str(datadir):s}\n{infostr:s}",
                fontsize=8,
            )

            pdf.savefig()  # t_path, dpi=300)  # use the map filename, as we will sort by this later
            mpl.close()  # (plot_handle)
            msg = f"Plotted protocol from {protocol_directory!s}"
            CP("g", msg)
            self.nfiles += 1

    def plot_IVs(self, df_selected=None, types: str = "IV", allprots: list = None):
        self.IV_pdf_file = None
        if df_selected is None or df_selected[types] is None:
            return
        print("IV Plotter: plot_IVs for cell id: ", df_selected.cell_id)
        print("    cell type: ", df_selected["cell_type"])
        assert types in ["IV", "MAP"], f"types must be IV or MAP, not {types!s}"
        celltype = df_selected["cell_type"].values[0]
        celltype = filename_tools.check_celltype(celltype)
        # check to see if this one is in the exclusion list:
        # print(df_selected.cell_id, self.experiment["excludeIVs"])
        protocols = df_selected[types].keys()
        # print("Plot IVs...", df_selected.cell_id)
        # print(type(df_selected.cell_id))
        if isinstance(df_selected.cell_id, pd.Series):
            cell_id = df_selected.cell_id.values[0]  # convert series to str
        elif not isinstance(df_selected.cell_id, str):
            cell_id = df_selected.cell_id.item()  # convert series to str
        else:
            cell_id = df_selected.cell_id
        protocols = FUNCS.remove_excluded_protocols(
            self.experiment, cell_id=cell_id, protocols=protocols
        )
        if len(protocols) == 0:
            CP("y", f"Excluding {cell_id} from the plotting; no valid protocols")
            return
        try:
            # find the cell in the main index
            # print("summary id: ", self.df_summary["cell_id"])
            # print("selected id: ", df_selected.cell_id)
            index = self.df_summary[self.df_summary["cell_id"] == cell_id].index[0]
        except IndexError:
            CP("r", f"Could not find cell: {cell_id} in the summary table")
            raise IndexError(f"Could not find cell: {cell_id} in the summary table")
        # print("INDEX: ", index)
        datestr, slicestr, cellstr = filename_tools.make_cell(icell=index, df=self.df_summary)
        if datestr is None:
            CP("r", f"Could not make filename partition for cell: {self.df_summary.cell_id!s}")
            return False
        slicecell = filename_tools.make_slicecell(slicestr, cellstr)
        matchcell, slicecell3, slicecell2, slicecell1 = filename_tools.compare_slice_cell(
            slicecell=slicecell, datestr=datestr, slicestr=slicestr, cellstr=cellstr
        )
        if not matchcell:
            return False
        thisday = datestr.replace(".", "_").split("_")
        thisday = "_".join(thisday[:-1])
        # print("IV Plotter calling get_cell: ")
        # print("    cell type: ", celltype)
        # print("    cell id: ", cell_id)
        # match to standard cell names
        cell_matched_type = map_cell_types.map_cell_type(self.df_summary.at[index, "cell_type"])
        # print("plot_IVs: Adjusted cell type: ", cell_matched_type)
        self.df_summary.at[index, "cell_type"] = cell_matched_type
        # print("\n******** df_summary row: ********\n", self.df_summary[self.df_summary["cell_id"] == cell_id])
        self.plot_df, _tmp = filename_tools.get_cell(
            experiment=self.experiment, df=self.df_summary, cell_id=cell_id,
            map_cell_names=True
        )
        if self.plot_df is None:  # likely no spike or IV protocols for this cell
            CP("r", f"Cell had no spike or IV protocol cell: {cell_id!s}")
            return

        if isinstance(df_selected["cell_type"], str):
            celltype = df_selected["cell_type"]
        else:
            celltype = df_selected["cell_type"].values[0]
        celltype = map_cell_types.map_cell_type(celltype)
        # print("    Mapped cell type: ", celltype)
        if celltype in ["no data", None, "None", " ", ""]:
            celltype = "unknown"
        print(" === adusted cell type: ", celltype)
        self.nfiles = 0
        self.IV_pdf_file = filename_tools.make_pdf_filename(
            self.file_out_path,
            thisday=thisday,
            celltype=celltype,
            analysistype="IVs",
            slicecell=slicecell,
        )
        # print("IV plotter: ", self.plot_df.keys())

        with PdfPages(Path(self.IV_pdf_file)) as pdf:
            for iv in self.plot_df["IV"].keys():
                protodir = Path(self.file_out_path, self.plot_df["cell_id"], iv)
                plot_handle, acq4 = self.plot_one_iv(iv, allprots=allprots)
                self.finalize_plot(
                    plot_handle, protocol_directory=protodir, pdf=pdf, acq4reader=acq4
                )
        
    def plot_one_iv(self, protocol, pubmode=False, allprots: list = None) -> Union[None, object]:

        git_hash = (
            data_table_functions.get_git_hashes()
        )  # get the hash for the current versions of ephys and our project

        # print("Plotting IV: ", protocol)
        if isinstance(self.plot_df["Spikes"][protocol], str):
            CP("r", f"Spikes for {protocol} is a string: {self.plot_df['Spikes'][protocol]!s}")
            return None, None
        if self.plot_df["Spikes"][protocol] is None:
            CP("r", f"Spikes for {protocol} is None")
            return None, None
        if "spikes" in self.plot_df["Spikes"][protocol].keys():
            spikes = self.plot_df["Spikes"][protocol]["spikes"]
        else:
            spikes = None
        spike_dict = self.plot_df["Spikes"][protocol]
        ivs = self.plot_df["IV"][protocol]
        # print("spike keys in protocol: ", spike_dict.keys()) # get keys for protocol
        # print("iv keys in protocol: ", ivs.keys())

        # build plot layout
        x = -0.08  # label position offsets
        y = 1.02
        sizer = {
            "A": {"pos": [0.05, 0.50, 0.4, 0.51], "labelpos": (x, y), "noaxes": False},
            "A1": {
                "pos": [0.05, 0.50, 0.32, 0.04],
                "labelpos": (x, y),
                "noaxes": False,
            },
            "A2": {"pos": [0.05, 0.50, 0.19, 0.09], "labelpos": (x, y), "noaxes": False},
            "A3": {"pos": [0.05, 0.50, 0.05, 0.09], "labelpos": (x, y), "noaxes": False},
            "B": {"pos": [0.62, 0.30, 0.78, 0.13], "labelpos": (x, y), "noaxes": False},
            "C": {"pos": [0.75, 0.17, 0.60, 0.13], "labelpos": (x, y)},
            "D1": {"pos": [0.62, 0.30, 0.40, 0.12], "labelpos": (x, y)},
            "D2": {"pos": [0.62, 0.30, 0.24, 0.12], "labelpos": (x, y)},
            "E": {"pos": [0.62, 0.30, 0.08, 0.12], "labelpos": (x, y)},
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
        P.figure_handle.text(
            0.01,
            0.98,
            f"Project git hash: {git_hash['project'][-9:]!s}\nephys git hash: {git_hash['ephys'][-9:]!s}\n",
            ha="left",
            va="top",
            fontsize=6,
        )

        P.axdict["A"].text(
            0.96,
            0.01,
            s=now,
            fontsize=6,
            ha="right",
            transform=P.figure_handle.transFigure,
        )
        P.axdict["A"].text(
            0.01,
            0.01,
            s=f"{self.plot_df['cell_id']:s}",
            fontsize=6,
            ha="left",
            transform=P.figure_handle.transFigure,
        )
        datadir = Path(self.experiment["rawdatapath"], self.experiment["directory"])
        self.AR = acq4_reader.acq4_reader(
            Path(datadir, self.plot_df["cell_id"], Path(protocol).name)
        )
        infostr = BIS.build_info_string(self.experiment, self.plot_df["cell_id"], self.AR)

        P.figure_handle.suptitle(
            f"{str(Path(datadir, self.plot_df["cell_id"], protocol)):s}\n{infostr:s}",
            fontsize=8,
        )
        dv = 50.0
        jsp = 0
        # guard against empty allprots
        if allprots is None:
            allprots = {'CCIV': None}
        res = (
            CIE.include_exclude(
                cell_id=self.plot_df["cell_id"],
                inclusions=self.experiment["includeIVs"],
                exclusions=self.experiment["excludeIVs"],
                allivs=allprots["CCIV"],
            ),
        )
        validivs, additional_ivs, additional_iv_records = res[0][0], res[0][1], res[0][2]
        self.allow_partial = False
        self.record_list = []
        # print("getting data for protocol: ", protocol)
        if (
            additional_iv_records is not None
            and len(additional_iv_records) > 0
            and protocol in additional_iv_records.keys()
        ):
            self.allow_partial = True
            self.record_list = additional_iv_records[protocol][1]

        # print(" allow partial,  record_list: ", self.allow_partial, self.record_list)
        if not self.AR.getData(allow_partial=self.allow_partial, record_list=self.record_list):
            return None, None
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
        cc_taum_protocol = str(protocol).find("_taum") > 1
        if not cc_taum_protocol:  # plot all traces (if doing taum, then only plot the mean)
            # print("\n\nSpike current array: ", spike_dict["FI_Curve"][0]*1e9, "\n\n")
            traces = np.argsort(spike_dict["FI_Curve"][0])  # sort by current level
            # print("spike array: \n", np.array(spike_dict["FI_Curve"]).shape)
            if self.allow_partial and len(self.record_list) > 0:
                traces = np.argsort(spike_dict["FI_Curve"][0][self.record_list])
            valid_traces = traces
            # print("protocol: ", protocol, "traces: ", traces, 'valid_traces: ', valid_traces)
            # print("AR protocol: ", self.AR.protocol)
            # list(range(self.AR.traces.shape[0]))
            # if spike_dict["FI_Curve"][0][0] > spike_dict["FI_Curve"][0][-1]:
            #     traces = traces.reverse()  # check for reverse acquisition order
            dv = 100.0
            for trn, trace_number in enumerate(traces):
                # print("trace number: ", trace_number, "trn: ", trn, self.AR.traces.shape[0])
                if self.plotting_alternation > 1:
                    if i % self.plotting_alternation != 0:
                        continue

                if spikes is not None and trace_number in list(spikes.keys()):
                    idv = float(jsp) * dv
                    jsp += 1
                else:
                    idv = 0.0
                P.axdict["A"].plot(
                    self.AR.time_base * 1e3,
                    idv + self.AR.traces[trace_number, :].view(np.ndarray) * 1e3,
                    "-",
                    color=trace_colors[trace_number],
                    linewidth=0.35,
                )

                P.axdict["A1"].plot(
                    self.AR.time_base * 1e3,
                    self.AR.cmd_wave[trace_number, :].view(np.ndarray) * 1e9,
                    "-",
                    color=trace_colors[trace_number],
                    linewidth=0.35,
                )

                # mark spikes inside the stimulus window
                ptps = np.array([])
                paps = np.array([])
                if (spikes is not None) and (trace_number in list(spikes.keys())) and self.decorate:
                    for j in list(spikes[trace_number].keys()):
                        paps = np.append(paps, spikes[trace_number][j].peak_V * 1e3)
                        ptps = np.append(ptps, spikes[trace_number][j].peak_T * 1e3)
                        # print("spikes ij ", i, j, spikes[i][j])
                    P.axdict["A"].plot(ptps, idv + paps, "ro", markersize=0.5)

                # mark spikes outside the stimlulus window if we ask for them
                if self.decorate:

                    clist = ["g", "b"]
                    windows = ["baseline_spikes", "poststimulus_spikes"]
                    for window_number, window in enumerate(windows):
                        ptps = spike_dict[window]
                        if len(ptps[trace_number]) == 0:
                            continue
                        uindx = [
                            int(u / (self.AR.sample_interval)) + 1
                            for u in ptps[trace_number]
                            if (int(u / (self.AR.sample_interval)) + 1) < self.AR.traces.shape[1]
                        ]
                        spike_times = np.array(
                            self.AR.time_base[uindx]
                        )  #  ptps[trace_number] # np.array(self.AR.time_base[uindx])
                        peak_aps = np.array(self.AR.traces[trace_number, uindx])
                        if len(peak_aps) < len(spike_times):
                            spike_times = spike_times[: len(peak_aps)]
                        P.axdict["A"].plot(
                            spike_times * 1e3,
                            idv + peak_aps * 1e3,
                            "o",
                            color=clist[window_number],
                            markersize=0.5,
                        )
        else:  # taum measure: plot the mean trace from CC_taum protocol
            # print("taum traces for taum measure: ", ivs["taum_traces"])
            # print("protocol: ", protocol)
            # print("TAUM TRACES: ", ivs["taum_traces"])

            # print(self.AR.traces.view(np.ndarray).shape)
            # print(self.AR.traces.view(np.ndarray).shape)
            P.axdict["A"].plot(
                self.AR.time_base * 1e3,
                np.mean(self.AR.traces.view(np.ndarray), axis=0) * 1e3,
                "-",
                color=trace_colors[0],
                linewidth=0.35,
            )
            if "ivss_cmd" in ivs.keys():
                # plot the fit
                fitn = list(ivs["taum_fitted"].keys())
                if len(fitn) > 0:
                    first_fn = fitn[0]
                    t0 = np.min(ivs["taum_fitted"][first_fn][0]) - 0.003
                    t1 = np.max(ivs["taum_fitted"][first_fn][0]) + 0.003
                    for fit_number in ivs["taum_fitted"].keys():
                        # CP('g', f"tau fitted keys: {str(k):s}")
                        P.axdict["A"].plot(
                            ivs["taum_fitted"][fit_number][0] * 1e3,  # ms
                            ivs["taum_fitted"][fit_number][1] * 1e3,  # mV
                            "--k",
                            linewidth=1.0,
                        )
            P.axdict["A1"].plot(
                self.AR.time_base * 1e3,
                np.mean(self.AR.cmd_wave.view(np.ndarray), axis=0) * 1e9,
                "-",
                color=trace_colors[0],
                linewidth=0.35,
            )
        if not pubmode:
            if "taum_fitted" not in ivs.keys():
                CP("y", f"iv_plotter: taum fitted is not in the ivs: {ivs.keys()!s}")
            if "taum" in ivs.keys() and ivs["taum"] != np.nan and "taum_fitted" in ivs.keys():
                # plot the taum trace fit magnified and on the relevant traces
                # print(ivs.keys())
                # print(ivs["taum_fitted"].keys())
                if "ivss_cmd" in ivs.keys() and not cc_taum_protocol:
                    # plot the fit
                    fitn = list(ivs["taum_fitted"].keys())
                    if len(fitn) > 0:
                        first_fn = fitn[0]
                        t0 = np.min(ivs["taum_fitted"][first_fn][0]) - 0.003
                        t1 = np.max(ivs["taum_fitted"][first_fn][0]) + 0.003
                        for fit_number in ivs["taum_fitted"].keys():
                            # CP('g', f"tau fitted keys: {str(k):s}")
                            P.axdict["A2"].plot(
                                ivs["taum_fitted"][fit_number][0] * 1e3,  # ms
                                ivs["taum_fitted"][fit_number][1] * 1e3,  # mV
                                "--k",
                                linewidth=1.0,
                            )
                        # plot the traces that were in the fit
                        for itr in ivs["taum_fitted"].keys():
                            it0 = np.argmin(np.abs(self.AR.time_base - t0))
                            it1 = np.argmin(np.abs(self.AR.time_base - t1))
                            # print(itr, it0, it1, t0, t1)
                            P.axdict["A2"].plot(
                                self.AR.time_base[it0:it1] * 1e3,
                                self.AR.traces[itr, it0:it1].view(np.ndarray) * 1e3,
                                "-",
                                color=trace_colors[itr],
                                linewidth=0.35,
                            )

            if (
                ("tauh_tau" in ivs.keys())
                and (ivs["tauh_tau"] != np.nan)
                and ("tauh_fitted" in ivs.keys())
            ):
                for fit_number in ivs["tauh_fitted"].keys():
                    first_fn = fit_number
                    CP('r', f"tau fitted keys: {str(first_fn):s}")
                    t0 = np.min(ivs["tauh_fitted"][first_fn][0]) - 0.003
                    t1 = np.max(ivs["tauh_fitted"][first_fn][0])
                    for itr in ivs["tauh_fitted"].keys():
                        it0 = np.argmin(np.abs(self.AR.time_base - t0))
                        it1 = np.argmin(np.abs(self.AR.time_base - t1))
                        P.axdict["A3"].plot(
                            self.AR.time_base[it0:it1] * 1e3,
                            self.AR.traces[itr, it0:it1].view(np.ndarray) * 1e3,
                            "-",
                            color=trace_colors[itr],
                            linewidth=0.35,
                        )
                    P.axdict["A3"].plot(
                        ivs["tauh_fitted"][fit_number][0] * 1e3,
                        ivs["tauh_fitted"][fit_number][1] * 1e3,
                        "--k",
                        linewidth=0.5,
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
        if not cc_taum_protocol:

            P.axdict["B"].plot(
                spike_dict["FI_Curve"][0][valid_traces] * 1e9,
                spike_dict["FI_Curve"][1][valid_traces] / (self.AR.tend - self.AR.tstart),
                "grey",
                linestyle="-",
                # markersize=4,
                linewidth=0.5,
            )

            P.axdict["B"].scatter(
                spike_dict["FI_Curve"][0][valid_traces] * 1e9,
                spike_dict["FI_Curve"][1][valid_traces] / (self.AR.tend - self.AR.tstart),
                color=trace_colors[: len(valid_traces)],
                s=16,
                linewidth=0.5,
            )
            xlims = list(P.axdict["B"].get_xlim())  # returns tuple but we need to modify...
            if xlims[0] > 0:
                xlims[0] = 0.0
                P.axdict["B"].set_xlim(xlims)

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
        if "ivss_cmd" not in ivs.keys() or "ivss_v" not in ivs.keys():
            print("\niv_plotter: No ivss_cmd found in the iv keys")
            print("    cell, protocol: ", self.plot_df["cell_id"], protocol)
            # print("\n    Ivs: ", ivs), "\n")
            # print("    Keys in the ivs: ", ivs.keys())

        elif str(protocol).find("_taum") < 0:  # not a taum protocol
            # P.axdict["C"].plot(
            #     np.array(ivs["ivss_cmd"]) * 1e9,
            #     np.array(ivs["ivss_v"]) * 1e3,
            #     "grey",
            #     # markersize=4,
            #     linewidth=1.0,
            # )
            print("ivss, iss_cmd: ", len(ivs["ivss_cmd"]), len(ivs["ivss_v"]))
            P.axdict["C"].scatter(
                np.array(ivs["ivss_cmd"]) * 1e9,
                np.array(ivs["ivss_v"]) * 1e3,
                color=trace_colors[0 : len(ivs["ivss_cmd"])],
                s=12,
            )
            if "ivss_fit" in ivs.keys() and len(ivs["ivss_cmd"]) > 0:
                ifit = np.linspace(np.min(ivs["ivss_cmd"]), np.max(ivs["ivss_cmd"]), 50)
                # print("ivs: ", ivs)
                # print(ifit)
                fit = np.polyval(ivs["ivss_fit"]["pars"], ifit)
                P.axdict["C"].plot(
                    ifit * 1e9,
                    fit * 1e3,
                    "--k",
                    linewidth=0.75,
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
            # Build a text table of parameters.
            taum = r"$\tau_m$"
            tauh = r"$\tau_h$"
            omega = r"$\Omega$"
            tstr = "Recording:"
            tstr += f"  SR: {self.AR.sample_rate[0] / 1e3:.1f} kHz\n"
            tstr += f"  Downsample: {self.downsample:d}\n"
            tstr += f"  ${{Pip Cap}}$: {ivs['CCComp']['CCNeutralizationCap']*1e12:.2f} pF\n"
            tstr += f"  Bridge [{enable:3s}]: {ccbridge:.1f} M{omega:s}\n"
            tstr += f"  Bridge Adjust: {ivs['BridgeAdjust']:.1f} m{omega:s}\n"
            tstr += f"  Pipette Offset: {cccomp:.1f} mV\n"
            tstr += "Measures:"
            tstr += f"  RMP: {ivs['RMP']:.1f} mV\n"
            if "Rin" in ivs.keys():
                tstr += f"  ${{R_{{in}}}}$: {ivs['Rin']:.1f} M{omega:s}\n"
            # print("IV plotter: ivs.keys: ", ivs.keys())
            # print("ivs['taum']: ", ivs['taum'])
            # print("ivs['taupars': ", ivs['taupars'])
            # determine the structure of taupars:
            # len = 3 means taupars is a list of 3 values, assuming taupars[0] is a float and not a list
            # if taupars[0] is a list, then use the 0th element from the list.

            if "taupars" in ivs.keys() and len(ivs["taupars"]) > 0:
                if isinstance(ivs["taupars"][0], list) and len(ivs["taupars"][0]) == 3:
                    tau_value = ivs["taupars"][0][2]
                elif len(ivs["taupars"]) == 3:
                    tau_value = ivs["taupars"][2]
                tstr += f"  {taum:s}: {tau_value*1e3:.2f} ms\n"
            else:
                tstr += f"  {taum:s}: <no measure>\n"
            if "tauh_tau" in ivs.keys():
                tstr += f"  {tauh:s}: {ivs['tauh_tau']*1e3:.3f} ms\n"
                tstr += f"  ${{G_h}}$: {ivs['tauh_Gh'] *1e9:.3f} nS\n"
            else:
                tstr += f"  {tauh:s}: <no measure>\n"
            tstr += f"  Holding: {np.mean(ivs['Irmp']) * 1e12:.1f} pA\n"
            P.axdict["C"].text(
                0.54,
                0.72,
                tstr,
                transform=P.figure_handle.transFigure,
                horizontalalignment="left",
                verticalalignment="top",
                fontsize=7,
                bbox=dict(facecolor="blue", alpha=0.15),
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

        # Plot the spike intervals as a function of time into the stimulus
        spk_isi = None
        if spikes is not None:
            for i, spike_tr in enumerate(spikes):  # this is the trace number
                if self.allow_partial and spike_tr not in valid_traces:
                    continue
                # print("Spike tr: ", i, spike_tr)
                spike_train = spikes[
                    spike_tr
                ]  # get the spike train for this trace, then get just the latency
                spk_tr = np.array([spike_train[sp].AP_latency for sp in spike_train.keys()])
                if (len(spk_tr) == 0) or (spk_tr[0] is None):
                    continue
                spk_tr = np.array([spk for spk in spk_tr if spk is not None])
                # print("spk_tr: ", i, spike_tr, spk_tr)
                # print("    tstart, end: ", self.AR.tstart, self.AR.tend)
                spx = np.nonzero(  # get the spikes that are in the stimulus window
                    (spk_tr > self.AR.tstart) & (spk_tr <= self.AR.tend)
                )
                spkl = (np.array(spk_tr[spx]) - self.AR.tstart) * 1e3  # relative to stimulus start
                # print("    spkl: ", spkl)
                if len(spkl) == 1:
                    P.axdict["D1"].plot(
                        spkl[0], spkl[0], "o", color=trace_colors[spike_tr], markersize=4
                    )
                    spk_isi = None
                else:
                    # print("spkl shape: " , spkl.shape)
                    spk_isi = np.diff(spkl)
                    spk_isit = spkl[: len(spk_isi)]
                    P.axdict["D1"].plot(
                        spk_isit,
                        spk_isi,
                        "o-",
                        color=trace_colors[spike_tr],
                        markersize=3,
                        linewidth=0.5,
                    )

        PH.talbotTicks(P.axdict["C"], tickPlacesAdd={"x": 1, "y": 0}, floatAdd={"x": 1, "y": 0})
        P.axdict["D1"].set_yscale("log")
        P.axdict["D1"].set_ylim((1.0, P.axdict["D1"].get_ylim()[1]))
        P.axdict["D1"].set_xlabel("Latency (ms)")
        P.axdict["D1"].set_ylabel("ISI (ms)")
        P.axdict["D1"].text(
            1.00,
            0.05,
            "Adapt Ratio: {0:.3f}".format(self.plot_df["Spikes"][protocol]["AdaptRatio"]),
            fontsize=9,
            transform=P.axdict["D1"].transAxes,
            horizontalalignment="right",
            verticalalignment="bottom",
        )

        if (spk_isi is not None) and len(spk_isi > 7) and spikes is not None:
            mode_thr = 1.5
            fi_currents = spike_dict["FI_Curve"][0] * 1e9
            # print("fi_currents: ", fi_currents)
            # isi_mode = np.zeros(len(fi_currents))*np.nan
            # isi_skips = np.zeros(len(fi_currents))*np.nan
            # isi_curr = np.zeros(len(fi_currents))*np.nan
            # c = []
            # for i, spike_tr in enumerate(spikes):  # this is the trace number
            # # print("Spike tr: ", i, spike_tr)
            #     spike_train = spikes[spike_tr]  # get the spike train for this trace, then get just the latency
            #     spk_tr = np.array([spike_train[sp].AP_latency for sp in spike_train.keys()])
            #     if (len(spk_tr) <= 6) or (spk_tr[0] is None ):
            #         continue
            #     spk_isi = np.diff(spk_tr)
            #     isi_mode[spike_tr] = np.median(spk_isi[4:])
            #     isi_skips[spike_tr] = np.count_nonzero(spk_isi[4:] > mode_thr*isi_mode[spike_tr])
            #     isi_curr[spike_tr] = fi_currents[i]
            for i, spike_tr in enumerate(spikes):  # this is the trace number
                if spike_tr not in valid_traces:
                    continue
                spike_train = spikes[
                    spike_tr
                ]  # get the spike train for this trace, then get just the latency
                spk_tr = np.array([spike_train[sp].AP_latency for sp in spike_train.keys()])
                if (None in spk_tr) or (len(spk_tr) <= 7):
                    continue
                spk_isi = np.diff(spk_tr)
                # plot joint isi
                P.axdict["D2"].scatter(
                    spk_isi[2:-1],
                    spk_isi[3:],
                    s=6,
                    color=trace_colors[spike_tr],
                    marker="o",
                    alpha=0.5,
                )
                P.axdict["D2"].scatter(
                    spk_isi[0], spk_isi[1], s=16, color=trace_colors[spike_tr], marker="+", alpha=1
                )  # mark first spike
                P.axdict["D2"].scatter(
                    spk_isi[1], spk_isi[2], s=6, color=trace_colors[spike_tr], marker="^", alpha=1
                )  # mark first spike
            # plot lines with slope of 1, 2 and 3
            axmax = np.max([P.axdict["D2"].get_xlim()[1], P.axdict["D2"].get_ylim()[1]])
            xb = np.linspace(0, axmax, 100)
            y1 = xb
            y2 = 2 * xb
            y3 = 3 * xb
            y4 = 4 * xb
            P.axdict["D2"].plot(xb, y1, "k--", linewidth=0.5)
            P.axdict["D2"].plot(xb, y2, "b--", linewidth=0.5)
            P.axdict["D2"].plot(xb, y3, "r--", linewidth=0.5)
            P.axdict["D2"].plot(xb, y4, "c--", linewidth=0.5)

            # P.axdict["D2"].plot(isi_curr, isi_mode, "o", markersize=4)
            P.axdict["D2"].set_xlim((0, axmax))
            P.axdict["D2"].set_ylim((0, axmax))
            P.axdict["D2"].set_xlabel("ISI(n)")
            P.axdict["D2"].set_ylabel("ISI(n+1)")

        # phase plot
        # P.axdict["E"].set_prop_cycle('color',[mpl.cm.jet(i) for i in np.linspace(0, 1, len(self.SP.spikeShapes.keys()))])
        if not cc_taum_protocol and spikes is not None:
            for k, i in enumerate(spikes.keys()):
                if i not in valid_traces:
                    continue
                if len(spikes[i]) == 0 or spikes[i][0] is None or spikes[i][0].dvdt is None:
                    continue
                n_dvdt = len(spikes[i][0].dvdt)
                P.axdict["E"].plot(
                    spikes[i][0].V[:n_dvdt],
                    spikes[i][0].dvdt,
                    linewidth=0.35,
                    color=trace_colors[i],
                )
                if i == 0:
                    break  # only plot the first one
        if "exclude_Spikes" in self.experiment.keys() and self.experiment["exclude_Spikes"] is not None:
            ex_spks = self.experiment["exclude_Spikes"]
            if self.plot_df["cell_id"] in ex_spks.keys():
                cell = ex_spks[self.plot_df["cell_id"]]
                mark = False
                if cell["protocols"] in [["all"], ["All"], ["ALL"]]:
                    mark = True
                elif protocol in cell["protocols"]:
                    mark = True
                else:
                    print("Did not match: ", protocol, cell["protocols"])
                if mark:
                    P.axdict["E"].text(
                        0.5,
                        0.5,
                        s=f"Excluded Spikes\nReason: {cell['reason']:s}",
                        color="r",
                        fontsize=11,
                        ha="center",
                        va="center",
                        transform=P.axdict["E"].transAxes,
                    )

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
        infostr = BIS.build_info_string(self.experiment, self.plot_df["cell_id"], self.AR)
        protocol = self.AR.protocol.name
        P.figure_handle.suptitle(
            f"{str(Path(self.plot_df['data_directory'], self.plot_df['cell_id'], protocol)):s}\n{infostr:s}",
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
                "--k",
                linewidth=0.75,
            )
        # for k in self.RM.tauh_fitted.keys():
        #     P.axdict["A"].plot(
        #         self.RM.tauh_fitted[k][0] * 1e3,
        #         self.RM.tauh_fitted[k][1] * 1e3,
        #         "--r",
        #         linewidth=0.750,
        #     )
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
