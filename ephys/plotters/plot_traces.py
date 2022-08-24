
from typing import Union, List
import matplotlib.pyplot as mpl
from pathlib import Path
import numpy as np
from pylibrary.util import cprint
from pylibrary.plotting import plothelpers as PH
from pylibrary.plotting import styler as PLS
from ephysanalysis import acq4read

class PlotTraces():

    def __init__(self):
        pass

    def set_file(self, filename:Union[Path, str]):
        pass

    def simple_plot_traces(
            self, rows=1, cols=1, width=5.0, height=4.0, stack=True, ymin=-120.0, ymax=0.0
        ):
            self.P = PH.regular_grid(
                rows,
                cols,
                order="rowsfirst",
                figsize=(width, height),
                showgrid=False,
                verticalspacing=0.01,
                horizontalspacing=0.01,
                margins={
                    "bottommargin": 0.1,
                    "leftmargin": 0.07,
                    "rightmargin": 0.05,
                    "topmargin": 0.03,
                },
                labelposition=(0.0, 0.0),
                parent_figure=None,
                panel_labels=None,
            )

            PD = self.newPData()
            for iax, index_row in enumerate(self.parent.selected_index_rows):
                selected = self.parent.table_manager.get_table_data(index_row)
                if selected is None:
                    return
                sfi = Path(selected.simulation_path, selected.files[0])
                if stack:
                    self.plot_traces(
                        self.P.axarr[iax, 0],
                        sfi,
                        PD,
                        protocol=selected.runProtocol,
                        ymin=ymin,
                        ymax=ymax,
                        iax=iax,
                        figure=self.P.figure_handle,
                    )
                else:
                    self.plot_traces(
                        self.P.axarr[0, iax],
                        sfi,
                        PD,
                        protocol=selected.runProtocol,
                        ymin=ymin,
                        ymax=ymax,
                        iax=iax,
                        figure=self.P.figure_handle,
                    )
            self.P.figure_handle.show()

    def print_file_info(self, selected, mode="list"):
        if mode not in ["list", "dict"]:
            raise ValueError()
        self.textappend("For copy into figure_data.py: ")
        if mode == "dict":
            br = "{}"
            self.textappend(f"{int(self.parent.cellID):d}: {br[0]:s}")
        if mode == "list":
            br = "[]"
            self.textappend(f"    {int(self.parent.cellID):d}: {br[0]:s}")
        for sel in selected:
            data = self.parent.table_manager.get_table_data(sel)
            fn = Path(data.files[0])
            fnr = str(fn.parts[-2])
            fkey = data.dendriteExpt
            if mode == "dict":
                self.textappend(f'    "{fkey:s}": "{fnr:s}",')
            if mode == "list":
                self.textappend(f'        "{fnr:s}",')
        if mode == "dict":
            self.textappend(f"{br[1]:s},")
        if mode == "list":
            self.textappend(f"    {br[1]:s},")

    def plot_traces(
        self,
        ax: object,
        fn: Union[Path, str],
        PD: dataclass,
        protocol: str,
        ymin: float = -80.0,
        ymax: float = 20.0,
        xmin: float = 0.0,
        xmax: Union[float, None] = None,
        yoffset: float = 0.0,
        iax: Union[int, None] = None,
        rep: Union[int, list, None] = None,  # which rep : none is for all.
        figure: object = None,
        show_title: bool = True,
        longtitle: bool = True,
        trace_color: str = "k",
        ivaxis: object = None,
        ivcolor: str = "k",
        iv_spike_color: str = "r",
        spike_marker_size: float = 2.5,
        spike_marker_color: str = "r",
        spike_marker_shape: str = "o",
        calx: Union[float, None] = 0.0,
        caly: Union[float, None] = 0.0,
        calt: Union[float, None] = 10.0,
        calv: Union[float, None] = 20.0,
        calx2: Union[float, None] = 20.0,
        caly2: Union[float, None] = 20.0,
        calt2: Union[float, None] = 10.0,
        calv2: Union[float, None] = 10.0,
        clipping: bool = False,
        axis_index: int = 0,  # index for axes, to prevent replotting text
    ) -> tuple:
        """Plot traces in a general way
        Yes, this should be broken up with fewer parameters,
        probably with dataclasses for the cals, etc... 

        Parameters
        ----------
        ax : object
            target matplotlib axis
        fn : Union[Path, str]
            filename of data to plot
        PD : dataclass
            PData object with info about the dataset
        protocol : str
            String name of the protocol
        ymin : float, optional
            Minimum for y axis, normally mV, by default -80.0
        ymax : float, optional
            Maximum for y axis, normally mV, by default 20.0
        xmin : float, optional
            Minimum for x axis, normally msec, but may be sec, by default 0.0
        xmax : Union[float, None], optional
            Maximum for x axis, normally msec, but may be sec, by default None
        yoffset : float, optional
            Value to offset Y by with an axis (for stacked traces), by default 0.0
        iax : Union[int, None], optional
            axis number, by default None
        rep : Union[int, list, None], optional
            repetition, by default None
        show_title : bool, optional
            display descriptive title, by default True
        longtitle : bool, optional
            display a long descriptive title, by default True
        trace_color : str, optional
            color of the traces, by default "k"
        ivaxis : object, optional
            optional axis for an IV plot, by default None
        ivcolor : str, optional
            color for an ivplot trace, by default "k"
        iv_spike_color : str, optional
            color to mark spikes in IV plot, by default "r"
        spike_marker_size : float, optional
            size of spike marker in traces, by default 2.5
        spike_marker_color : str, optional
            color of spike marker in traces, by default "r"
        spike_marker_shape : str, optional
            shape of spike marker in traces, by default "o"
        calx : Union[float, None], optional
            calibration bar x length, by default 0.0
        caly : Union[float, None], optional
        calibration bar y length, by default 0.0
        calt : Union[float, None], optional
            calibration bar position along x axis (time), by default 10.0
        calv : Union[float, None], optional
            calibration bar position along y axis (voltage), by default 20.0
        calx2 : Union[float, None], optional
            secondary cal bar, by default 20.0
        caly2 : Union[float, None], optional
            secondary cal bar, by default 20.0
        calt2 : Union[float, None], optional
            secondary cal bar, by default 10.0
        calv2 : Union[float, None], optional
            secondary cal bar, by default 10.0
        axis_index : int, optional
            index for axes, to prevent replotting text_, by default 0

        Returns
        -------
        tuple
            synno : number of synapses on this cell
            noutspikes : number of spikes from the cell
            ninspikes : number of input spikes to the cell
        """

        inx = str(fn).find("_Syn")
        synno = None
        if inx > 0:
            synno = int(str(fn)[inx + 4 : inx + 7])
        if protocol in ["IV", "runIV"]:
            protocol = "IV"
        elif protocol in ["VC", "runVC"]:
            protocol = "VC"

        model_data = self.ReadModel.get_data(fn, PD=PD, protocol=protocol)
        data = model_data.data
        si = model_data.SI
        ri = model_data.RI
        if figure is None:  # no figure... just analysis...
            return model_data.AR, model_data.SP, model_data.RM
        AR = model_data.AR
        SP = model_data.SP
        RM = model_data.RM
        cprint("c", "plot_traces: preparing for plot")
        ntr = len(AR.MC.traces)  # number of trials
        v0 = -160.0
        if isinstance(ri, dict):
            deadtime = ri["stimDelay"]
        else:
            deadtime = ri.stimDelay
        trstep = 25.0 / ntr
        inpstep = 2.0 / ntr
        sz = 50.0 / ntr
        noutspikes = 0
        ninspikes = 0
        ispikethr = None
        spike_rheobase = None
        if xmax is None and protocol not in ["IV", "VC"]:
            xmax = 1e3 * (ri.pip_start + ri.pip_duration)
            xmin = 1e3 * ri.pip_start
        elif xmax is not None:
            pass
        elif xmax is None and protocol in ["IV"]:
            xmax = ri.stimDelay + ri.stimDur + ri.stimPost
        elif xmax is None and protocol in ["VC"]:
            xmax = ri.vstimDelay + ri.vstimDur + ri.vstimPost
        else:
            print('xmax: ', xmax)
            print('protocol: ', protocol)
            raise ValueError("Need to specificy xmax for plot") 
        if isinstance(ax, list):
            ax1 = ax[0]
            ax2 = ax[1]
        elif hasattr("ax", "len") and len(ax) == 2:
            ax1 = ax[0]
            ax2 = ax[1]
        elif hasattr("ax", "len") and len(ax) == 1:
            ax1 = ax
            ax2 = None
        elif not hasattr("ax", "len"):
            ax1 = ax
            ax2 = None
        for trial, icurr in enumerate(data["Results"]):
            if rep is not None and trial != rep:
                continue
            AR.MC.traces[trial][0] = AR.MC.traces[trial][1]
            if protocol in ["VC", "vc", "vclamp"]:
                AR.MC.traces[trial] = AR.MC.traces[trial].asarray() * 1e9  # nA
                cmd = AR.MC.cmd_wave[trial] * 1e3  # from V to mV
            else:
                AR.MC.traces[trial] = AR.MC.traces[trial].asarray() * 1e3  # mV
                cmd = AR.MC.cmd_wave[trial] * 1e9  # from A to nA
            xclip = np.argwhere((AR.MC.time_base >= xmin) & (AR.MC.time_base <= xmax))
            # plot trace
            ax1.plot(
                AR.MC.time_base[xclip],
                AR.MC.traces[trial][xclip] + yoffset,
                linestyle="-",
                color=trace_color,
                linewidth=0.5,
                clip_on=clipping,
            )
            print(cmd)
            if ax2 is not None:
                ax2.plot(AR.MC.time_base[xclip], cmd[xclip], linewidth=0.5)
            if "spikeTimes" in list(data["Results"][icurr].keys()):
                # cprint('r', 'spiketimes from results')
                # print(data["Results"][icurr]["spikeTimes"])
                #  print(si.dtIC)
                spikeindex = [
                    int(t * 1e3 / (si.dtIC))
                    for t in data["Results"][icurr]["spikeTimes"]
                ]
            else:
                cprint('r', 'spikes from SP.spikeIndices')
                spikeindex = SP.spikeIndices[trial]
            # print(f"Trial: {trial:3d} Nspikes: {len(spikeindex):d}")
            # plot spike peaks
            ax1.plot(
                AR.MC.time_base[spikeindex],
                AR.MC.traces[trial][spikeindex] + yoffset,
                marker = spike_marker_shape, # "o",
                color=spike_marker_color,
                markerfacecolor=spike_marker_color,
                markersize=spike_marker_size,
                linestyle="none",
            )
            sinds = np.array(spikeindex) * AR.MC.sample_rate[trial]
            # print('sinds: ', sinds, deadtime, ri.stimDelay, ri.stimDur)
            nspk_in_trial = len(np.argwhere(sinds > deadtime))
            if (
                nspk_in_trial > 0
                and ispikethr is None
                and sinds[0] < (ri.stimDelay + ri.stimDur)
            ):
                # cprint('c', f"Found threshold spike:  {icurr:.2f}, {trial:d}")
                spike_rheobase = icurr
                ispikethr = trial
            noutspikes += nspk_in_trial
            if protocol in ["AN", "runANSingles"]:
                if trial in list(data["Results"].keys()) and "inputSpikeTimes" in list(
                    data["Results"][icurr].keys()
                ):
                    spkt = data["Results"][icurr]["inputSpikeTimes"]
                elif "inputSpikeTimes" in list(data["Results"].keys()):
                    spkt = data["Results"]["inputSpikeTimes"][trial]
                tr_y = trial * (trstep + len(spkt) * inpstep)
                if synno is None:
                    for ian in range(len(spkt)):
                        vy = v0 + tr_y * np.ones(len(spkt[ian])) + inpstep * ian
                        ax1.scatter(spkt[ian], vy, s=sz, marker="|", linewidths=0.35)
                else:
                    ian = synno
                    vy = v0 + tr_y * np.ones(len(spkt[ian])) + inpstep * ian
                    # ax.scatter(spkt[ian], vy, s=sz, marker="|", linewidths=0.35)
                    ninspikes += len(spkt[ian] > deadtime)

                ax1.set_ylim(ymin, ymax)
                if xmin is None:
                    xmin = 0.050
                if xmax is None:
                    xmax = np.max(AR.MC.time_base)
                ax1.set_xlim(xmin, xmax)
            elif protocol in ["VC", "vc", "vclamp"]:
                pass  #
                # ax.set_ylim((-100.0, 100.0))
            else:
                ax1.set_ylim(ymin, ymax)
                if xmin is None:
                    cprint("r", "2. xmin is None")
                    xmin = 0.050
                if xmax is None:
                    xmax = np.max(AR.MC.time_base)
                ax1.set_xlim(xmin, xmax)
        ftname = str(Path(fn).name)
        ip = ftname.find("_II_") + 4
        ftname = ftname[:ip] + "...\n" + ftname[ip:]
        toptitle = ""
        if longtitle and show_title:
            toptitle = f"{ftname:s}"
        else:
            if show_title:
                toptitle = si.dendriteExpt
        if protocol in ["IV"]:
            cprint("r", f"RM analysis taum: {RM.analysis_summary['taum']:.2f}")
            if show_title:
                omega = r"$\Omega$"
                tau_m = r"$\tau_m$"
                toptitle += f"\nRin={RM.analysis_summary['Rin']:.1f} M{omega:s}  {tau_m}{RM.analysis_summary['taum']:.2f} ms"

            if iax is not None and calx is not None:
                PH.calbar(
                    ax1,
                    calbar=[calx, caly, calt, calv],
                    scale=[1.0, 1.0],
                    axesoff=True,
                    orient="right",
                    unitNames={"x": "ms", "y": "mV"},
                    fontsize=11,
                    weight="normal",
                    color="k",
                    font="Arial",
                )
            else:
                PH.noaxes(ax)
            # insert IV curve

            if ivaxis is None:
                secax = PLS.create_inset_axes(
                    [0.45, -0.05, 0.3, 0.3], ax, label=str(ax)
                )
                color = "k"
                ticklabelsize = 6
            else:
                secax = ivaxis
                color = ivcolor
                ticklabelsize = 8

            secax.plot(
                RM.ivss_cmd_all * 1e9,
                RM.ivss_v_all.asarray() * 1e3,
                f"{color:s}s-",
                markersize=3,
                markerfacecolor="k",
                zorder=10,
                clip_on=False,
            )

            ltz = np.where(RM.ivss_cmd_all <= 0.0)[0]

            secax.plot(
                RM.ivpk_cmd_all[ltz] * 1e9,
                RM.ivpk_v_all[ltz] * 1e3,
                f"{color:s}o-",
                markersize=3,
                markerfacecolor="w",
                zorder=10,
                clip_on=False,
            )
            if ispikethr is not None:  # decorate spike threshold point
                secax.plot(
                    RM.ivss_cmd_all[ispikethr] * 1e9,
                    RM.ivss_v_all[ispikethr] * 1e3,
                    marker=spike_marker_shape,
                    markersize=spike_marker_size,
                    color=iv_spike_color,
                    markerfacecolor=iv_spike_color,
                    zorder=100,
                    clip_on=False,
                )
            PH.crossAxes(
                secax,
                xyzero=[0.0, -60.0],
                limits=[
                    np.min(RM.ivss_cmd_all) * 1e9,
                    ymin,
                    np.max(RM.ivss_cmd_all) * 1e9,
                    -25.0,
                ],  #
            )
            PH.talbotTicks(
                secax,
                axes="xy",
                density=(1.0, 1.0),
                insideMargin=0.02,
                pointSize=ticklabelsize,
                tickPlacesAdd={"x": 1, "y": 0},
                floatAdd={"x": 1, "y": 0},
            )
            if axis_index == 0:
                secax.text(2.0, -70.0, "nA", ha="center", va="top", fontweight="normal")
                secax.text(
                    0.0, -40.0, "mV ", ha="right", va="center", fontweight="normal"
                )
            self.traces_ax = ax1
            self.crossed_iv_ax = secax

        elif protocol in ["VC", "vc", "vclamp"]:
            maxt = np.max(AR.MC.time_base)
            if calx is not None:
                PH.calbar(
                    ax1,
                    calbar=[calx, caly, calt, calv],
                    orient="left",
                    unitNames={"x": "ms", "y": "nA"},
                    fontsize=9,
                )
            else:
                PH.noaxes(ax1)
            if ax2 is not None and calx2 is not None:
                PH.calbar(
                    ax2,
                    calbar=[calx2, caly2, calt2, calv2],
                    scale=[1.0, 1.0],
                    axesoff=True,
                    orient="left",
                    unitNames={"x": "ms", "y": "mV"},
                    fontsize=9,
                    weight="normal",
                    color="k",
                    font="Arial",
                )
            else:
                PH.noaxes(ax2)
        else:
            if calx is not None and iax is not None:
                cprint("r", "**** making cal bar")
                print("calx, y, calt, v: ", iax, calx, caly, calt, calv)
                PH.calbar(
                    ax,
                    calbar=[calx, caly, calt, calv],
                    unitNames={"x": "ms", "y": "mV"},
                    fontsize=8,
                )
            else:
                PH.noaxes(ax)
            if protocol in ["IV"]:  # only valid for an IV
                if RM.analysis_summary is not None:
                    PH.referenceline(ax, RM.analysis_summary["RMP"])
                ax.text(
                    -1.0,
                    RM.analysis_summary["RMP"],
                    f"{RM.analysis_summary['RMP']:.1f}",
                    verticalalignment="center",
                    horizontalalignment="right",
                    fontsize=9,
                )
        if show_title:
            cprint("r", "ShowTitle in plot_traces")
            toptitle += f"\n{model_data.timestamp:s}"
            figure.suptitle(toptitle, fontsize=9)

        return (synno, noutspikes, ninspikes)

    def plot_VC(
        self,
        selected_index_rows=None,
        sfi=None,
        parent_figure=None,
        loc=None,
        show=True,
    ):
        if selected_index_rows is not None:
            n_columns = len(selected_index_rows)
            if n_columns == 0:
                return
            sfi = [None for i in range(n_columns)]
            protocol = [None for i in range(n_columns)]
            dendriteMode = [None for i in range(n_columns)]
            for i in range(n_columns):
                index_row = selected_index_rows[i]
                selected = self.parent.table_manager.get_table_data(index_row)
                sfi[i] = Path(selected.simulation_path, selected.files[0])
                protocol[i] = selected.runProtocol
                dendriteMode[i] = selected.dendriteMode

        else:
            n_columns = len(sfi)
            protocol = [None for i in range(n_columns)]
            dendriteMode = [None for i in range(n_columns)]
            for i in range(n_columns):
                with open(sfi[i], "rb") as fh:
                    sfidata = FPM.pickle_load(fh)  # , encoding='latin-1')
                    # print(sfidata['Params'])
                    dendriteMode[i] = sfidata["Params"].dendriteMode
                    protocol[i] = sfidata["runInfo"].runProtocol
        P = self.setup_VC_plots(n_columns, parent_figure=parent_figure, loc=loc)
        titlemap = {"normal": "Half-active", "passive": "Passive", "active": "Active"}

        for i in range(n_columns):
            PD = self.newPData()
            trace_ax = P.axarr[i * 3, 0]
            cmd_ax = P.axarr[i * 3 + 1, 0]
            if i == 0:
                calx=120.
            else:
                calx = None
            self.plot_traces(
                ax=[trace_ax, cmd_ax],
                fn=sfi[i],
                PD=PD,
                protocol=protocol[i],
                calx=calx,
                caly=5.0,
                calt=10.0,
                calv=2.0,
                calx2=calx,
                caly2=-40.0,
                calt2=10.0,
                calv2=20.0,
                xmax=150.0,
                figure=P.figure_handle,
                clipping=True,
                show_title=False
            )
            trace_ax.set_ylim((-1, 15))
            trace_ax.set_clip_on(True)
            self.analyzeVC(P.axarr[i * 3 + 2, 0], sfi[i], PD, protocol=protocol[i])
            PH.nice_plot(P.axarr[i * 3 + 2, 0], position=-0.03, direction="outward", ticklength=3)
            trace_ax.text(
                0.5,
                1.1,
                titlemap[dendriteMode[i]],
                transform=trace_ax.transAxes,
                fontsize=9,
                verticalalignment="bottom",
                horizontalalignment="center",
            )
            vref = -80.0
            PH.referenceline(cmd_ax, vref)
            cmd_ax.text(
                -5.25,
                vref,
                f"{int(vref):d} mV",
                fontsize=9,
                color="k",
                # transform=secax.transAxes,
                horizontalalignment="right",
                verticalalignment="center",
            )
        if show:
            P.figure_handle.show()
        return P