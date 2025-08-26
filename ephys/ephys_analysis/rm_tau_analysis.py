"""
RMTauAnalysis - Analyze Rm, tau - pulled out of IVCurve 2/29/2016 pbm.
Allows routine to be used to responses to hyperpolarizing pulses independent of acq4's data models.
Create instance, then  call setup to define the "Clamps" structure and analysis parameters. 

Clamps must have the following variables defined:

    commandLevels (current injection levels, list)
    time_base (np.array of times corresponding to traces)
    data_mode ()
    tstart (time for start of looking at spikes; ms)
    tend
    trace
    sample_interval (time between samples, sec)
    values (command waveforms; why it is called this in acq4 is a mystery)

The "Clamps" object can be provided by acq4's PatchEphys module, or by
an instance of acq4_reader.

RmTauAnalysis requires that the SpikeAnalysis be run first.

Paul B. Manis, 2015-2019
for acq4

"""

import datetime
import numpy as np
from typing import Union
from scipy.signal import savgol_filter  # for smoothing
import ephys.tools as TOOLS
from pathlib import Path
import pyqtgraph as pg
import ephys.tools.exp_estimator_lmfit as exp_estimator_lmfit
import warnings  # use to catch poor polynomial fits for Rin


class RmTauAnalysis:
    """
    RmTau analysis measures Rm (input resistance) and the membrane time constant for
    the traces in the Clamps structure.
    It also makes some measures related to Ih for hyperpolarizing pulses, including
    the peak and steady-state voltages, and the time constant of Ih activation (e.g.
    the sag itself after the peak).

    To use: create an instance of the class, then call RmTauAnalysis.setup(...)
    to initialized the parameters. You can then call any of the methods to get
    the tau, rmp, etc. from the data set
    """

    def __init__(self):

        self.reset()

    def reset(self):
        self.Clamps = None
        self.Spikes = None
        self.dataPlot = None
        self.baseline = None
        self.rmp = None
        self.taum_fitted = {}
        self.taum_bounds = []
        self.taum_current_range = [0, -200e-12]  # in A
        self.analysis_summary = {}
        self.rin_current_limit:float = np.nan  # no limit, should be in A

    def setup(
        self,
        clamps=None,
        spikes=None,
        dataplot=None,
        baseline: list = [0, 0.001],
        bridge_offset: float = 0,
        taum_bounds: list = [0.001, 0.050],
        taum_current_range: list = [0, -200e-12],  # in A
        tauh_voltage: float = -0.08, # in V
        rin_current_limit: float = np.nan   # no limit, should be in A
    ):
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

        taum_bounds : list (2 elements)
            Lower and upper bounds of the allowable taum fitting range (in seconds).

        bridge_offset : float (default:  0.0 Ohms)
            Bridge offset resistance (Ohms)
        """

        if clamps is None or spikes is None:
            raise ValueError("RmTau analysis requires defined clamps and spike analysis")
        self.Clamps = clamps
        self.Spikes = spikes
        self.dataPlot = dataplot
        self.baseline = baseline
        self.bridge_offset = bridge_offset
        self.taum_fitted = {}
        self.tauh_fitted = {}
        self.taum_bounds = taum_bounds
        self.taum_current_range = taum_current_range
        self.tauh_voltage = tauh_voltage
        self.rin_current_limit = rin_current_limit
        self.analysis_summary["holding"] = self.Clamps.holding
        self.analysis_summary["WCComp"] = self.Clamps.WCComp
        self.analysis_summary["CCComp"] = self.Clamps.CCComp
        if self.bridge_offset != 0.0:
            self.bridge_adjust()
        self.analysis_summary["BridgeAdjust"] = self.bridge_offset  # save the bridge offset value

    def bridge_adjust(self):
        """
        Adjust the voltage waveform according to the bridge offset value
        """
        print("RmTau adjusting bridge...")
        self.Clamps.traces = (
            self.Clamps.traces - self.Clamps.cmd_wave.view(np.ndarray) * self.bridge_offset
        )

    def analyze(
        self,
        rmp_region=[0.0, 0.05],
        tau_region=[0.1, 0.125],
        rin_region: Union[list, None] = None,
        rin_protocols: Union[list, None] = None,
        to_peak=False,
        tgap=0.0005,
        average_flag: bool = False,
    ):
        # print("self.clamps: ", self.Clamps)
        print("starting rm_tau analysis, average_flag: ", average_flag)
        print("    rmpregion: ", rmp_region)
        print("    tauregion: ", tau_region)
        print("    rin_region: ", rin_region)
        print("    rin_current_limit: ", self.rin_current_limit)
        self.analysis_summary["analysistimestamp"] = datetime.datetime.now().strftime(
            "%m/%d/%Y, %H:%M:%S"
        )
        if tau_region[1] > self.Clamps.tend:
            tau_region[1] = self.Clamps.tend
        stepdur = self.Clamps.tend - self.Clamps.tstart
        self.rmp_analysis(time_window=rmp_region)
        self.tau_membrane(
            time_window=tau_region, peak_time=to_peak, tgap=tgap, average_flag=average_flag
        )

        if not average_flag:
            # make sure protocol in in the range of those we want to analyze
            r_pk = [self.Clamps.tstart, self.Clamps.tstart + 0.4 * stepdur]
            if rin_region is None:  # default values
                r_ss = [self.Clamps.tstart + 0.9 * stepdur, self.Clamps.tend]  # steady-state region
            else:  # use a defined input
                r_ss = rin_region
            this_protocol = Path(self.Clamps.protocol).name[:-4]
            # Note that if Rin protocls is NOT present, then we analyze anyway.
            if rin_protocols is not None and this_protocol not in rin_protocols:
                print("    Protocol is not in list of protocols supporting Rin analysis")
                print("    This protocol: ", this_protocol, "\n    Supporting protocols: ", rin_protocols)
                # raise ValueError("Protocol does not match rin_protocols")
                return
            print("    RM.analyze: rss: ", r_ss)
            self.ivss_analysis(time_window=r_ss)
            self.ivpk_analysis(time_window=r_pk)  # peak region
            try:
                self.tau_h(
                    v_steadystate=self.tauh_voltage,
                    peak_timewindow=[r_pk[0], r_ss[0]],  # self.Clamps.tstart, r_pk],
                    steadystate_timewindow=[r_ss[0], self.Clamps.tend],
                    printWindow=False,
                )
            except ValueError:
                
                print("Error in tau_h analysis")
                print("r_pk: ", r_pk, "r_ss: ", r_ss)
                print("rin region: ", rin_region)
                pass# raise ValueError("Error in tau_h analysis")

    def tau_membrane(
        self,
        time_window: list = [],
        peak_time: bool = False,
        printWindow: bool = False,
        whichTau: int = 1,
        vrange: list = [-0.002, -0.050],
        tgap: float = 0.0,
        average_flag: bool = False,
    ):
        r"""
        Compute time constant (single exponential) from the onset of the response to a current step

        Parameters
        ----------
        time_window: list (s) (default: [])
            Define the time region for the fitting

        peak_time : bool (default: False)
            Whether to fit only to the (negative) peak of the data. If it is True, the
            fit will only go until the peak time. Otherwise it is set by the end of the time_window.

        printWindow : Boolean (default: False)
            Flag to allow printing of the fit results in detail during a run

        whichTau : int (default: 1)
            Not used

        vrange : list (V) (default: [-0.005, -0.020])
            Define the voltage range _below_ RMP for the traces that will be fit to obtain tau_m.

        tgap: float (sec)
            gap for the fitting to start (e.g., duration of initial points to ignore)

        average_flag: bool (default: False)
            If True, the routine will average the traces before fitting. This is useful for
            noisy data with a single current injection,
            but may not be appropriate for data with multiple current levels.
        Return
        ------
            Nothing

        Class variables with a leading taum\_ are set by this routine, to return results.

        """
        assert len(time_window) == 2
        USE_OLD_FIT = False  # fitting prior to 8/25/2024 - works, but new method might be better
        USE_NEW_FIT_8_24 = True
        debug = False

        # print("rm_tau_analysis: tau_membrane Time Window: ", time_window)
        Func = "exp1"  # single exponential fit with DC offset.
        self.taum_func = Func
        self.analysis_summary["taum"] = np.nan
        self.analysis_summary["taupars"] = []
        self.analysis_summary["taufunc"] = self.taum_func
        self.analysis_summary["taum_fitted"] = []
        self.analysis_summary["taum_fitmode"] = "average"
        self.analysis_summary["taum_traces"] = []
        self.analysis_summary["ivss_cmd"] = []

        if debug:
            print("taum")
            print("time_window: ", time_window)
            print("PEAK TIME: ", peak_time)

        if self.rmp is None:
            self.rmp_analysis(time_window=self.baseline)

        Fits = TOOLS.fitting.Fitting()  # get a fitting instance
        initpars = [self.rmp * 1e-3, -0.010, 0.020]  # rmp is in units of mV

        # determine which current injection levels to use for the measurements.
        # We need to exclude traces with spikes on them either befor or during
        # the current pulse (it happens). Probably don't care about rebound spikes?
        baselinespikes = [False] * len(self.Spikes.analysis_summary["baseline_spikes"])

        for i, bspk in enumerate(self.Spikes.analysis_summary["baseline_spikes"]):
            baselinespikes[i] = len(bspk) > 0
        poststimulusspikes = [False] * len(self.Spikes.analysis_summary["poststimulus_spikes"])
        for i, pspk in enumerate(self.Spikes.analysis_summary["poststimulus_spikes"]):
            poststimulusspikes[i] = len(pspk) > 0
        if not average_flag and USE_OLD_FIT:  # unless we are averaging for a single command level!
            ineg_valid = self.Clamps.commandLevels < -10e-12
            if len(ineg_valid) != len(self.Spikes.spikecount):
                print("    taum - rm_tau_analysis: len(ineg_valid) != len(self.Spikes.spikecount)")
                print(len(ineg_valid), len(self.Spikes.spikecount))
                assert len(ineg_valid) == len(self.Spikes.spikecount)
            ineg_valid = ineg_valid & (self.Spikes.spikecount == 0)

            if not any(ineg_valid):  # no valid traces to use
                print("    taum - rm_tau_analysis: No valid traces for taum measure in this protocol")
                return
        if not average_flag and USE_NEW_FIT_8_24:
            # include traces with no baseline spikes or stimulus-time spikes;
            # post stimulus spikes are ok.
            ineg_valid = self.Clamps.commandLevels < 0.0

            # print("baseline spikes: ", len(baselinespikes))
            # print("poststimulus spikes: ", len(poststimulusspikes))
            # print("spikecount: ", self.Spikes.spikecount)
            if len(ineg_valid) != len(self.Spikes.spikecount):
                print(
                    "ERROR: # traces in ineg_valid bool array is not the same as the number of traces in spikecount"
                )
                print("Clamp traces: ", self.Clamps.traces.shape)
                print("Command levels: ", self.Clamps.commandLevels.shape)
                print("rm_tau_analysis: len(ineg_valid) != len(self.Spikes.spikecount)")
                print(len(ineg_valid), len(self.Spikes.spikecount))
                print("len baseline spikes: ", len(baselinespikes))
                print("len poststimulus spikes: ", len(poststimulusspikes))
                # assert len(ineg_valid) == len(self.Spikes.spikecount)
            ineg_valid = [
                ineg_valid[ibls]
                and not baselinespikes[ibls]
                and (self.Spikes.spikecount[ibls] == 0)
                # and not poststimulusspikes[i]
                for ibls in range(len(ineg_valid))
            ]
        if average_flag:  # averaging for a single command level
            # establish valid traces for averaging
            # Note there may be only ONE command level listed above, so we need to
            # include ALL the traces in the average, but only those with no
            # detected spikes.
            ineg_valid = [
                not baselinespikes[i]
                and (self.Spikes.spikecount[i] == 0)
                and not poststimulusspikes[i]
                for i in range(len(self.Spikes.spikecount))
            ]

        # print("ineg valid: ", ineg_valid)

        if len(list(self.taum_fitted.keys())) > 0 and self.dataPlot is not None:
            [self.taum_fitted[k].clear() for k in list(self.taum_fitted.keys())]
        self.taum_fitted = {}

        if debug:
            print("command levels: ", self.Clamps.commandLevels * 1e12)
            print("valid traces with negative commands: ", ineg_valid)
        # if peak_time is not None and ineg != np.array([]):
        #     rgnpk[1] = np.max(peak_time[ineg[0]])
        dt = self.Clamps.sample_interval
        time_base = self.Clamps.time_base.view(np.ndarray)
        i_fitwin = [int(time_window[0] / dt), int(time_window[1] / dt)]
        time_base_fit = time_base[int(time_window[0] / dt) : int(time_window[1] / dt)]
        traces = self.Clamps.traces.view(np.ndarray)
        vrange = [-0.01, -0.001]
        vrange = np.sort(vrange)  # vrange is in mV
        vmeans = (  # mean of trace at the last 1 msec the measurment time window, as difference from baseline
            np.mean(
                traces[:, int((time_window[1] - 0.001) / dt) : int(time_window[1] / dt)], axis=1
            )
            - self.ivbaseline
        )
        vbl = (  # mean of trace at the last 2 msec prior to the start of the measurement step
            np.mean(
                traces[:, int((time_window[0] - 0.002) / dt) : int(time_window[0] / dt)], axis=1
            )
        )
        # print("VBL: ", vbl, traces.shape[0])
        # print("vmeans: ", vmeans.shape)
        # print("VMEANS: ", vmeans)
        # print("vranges: ", vrange)
        # print("ineg_valud: ", ineg_valid)
        if not average_flag:
            indxs = []
            # for indxs = [i for i, x in enumerate(vmeans) if (x >= vrange[0]) and (x <= vrange[1]) and (i in ineg_valid)]
            for i, x in enumerate(vmeans):
                # print(x, vrange, ineg_valid[i])
                if (x >= vrange[0]) and (x <= vrange[1]) and ineg_valid[i]:
                    indxs.append(i)
                    # print("appended: ", i)
            # print(indxs)

        else:
            indxs = [[0]]

        # debug = True
        if debug:
            print("baseline: ", self.ivbaseline)
            print("vrange: ", vrange)
            print("vmeans: ", vmeans.view(np.ndarray))
            print("Vbl: ", vbl.view(np.ndarray))
            print("indxs: ", indxs)
            print("ineg: ", ineg_valid)
            print("self.Clamps.commandLevels", self.Clamps.commandLevels.view(np.ndarray))
            print("twindow: ", time_window)
            print("IV baseline: ", self.ivbaseline)
            print("vrange: ", vrange)
            print("vmeans[ineg_valid]: ", vmeans[ineg_valid].view(np.ndarray))
            print("indxs: ", indxs)
        # indxs = list(indxs[0])
        # exit()
        whichdata = indxs  # restricts to valid values

        if not average_flag:
            itaucmd = self.Clamps.commandLevels[ineg_valid]
        else:
            itaucmd = self.Clamps.commandLevels[0]  # just use the first one for the average
        whichaxis = 0
        fpar = []
        names = []
        okdata = []

        # ------------------------------------------------------------------
        # Fit Averaged Traces  (CC_taum: all traces averaged)
        # ------------------------------------------------------------------
        if average_flag:  # just fit the average of all traces
            mean_trace = np.mean(traces[ineg_valid], axis=0)
            data_to_fit = mean_trace[i_fitwin[0] : i_fitwin[1]]
            LME = exp_estimator_lmfit.LMexpFit()
            t_fit = time_base_fit - time_base_fit[0]
            LME.initial_estimator(t_fit, data_to_fit, verbose=False)
            # print(f"estm: DC: {LME.DC:8.3f}, A1: {LME.A1:8.3f}, R1: {LME.R1:8.3f}")
            fit = LME.fit1(t_fit, data_to_fit, plot=False, verbose=False)
            # print(f"Fit:  DC: {fit.params['DC'].value:8.3f}, A1: {fit.params['A1'].value:8.3f}, R1: {fit.params['R1'].value:8.3f}")
            fit_curve = LME.exp_decay1(
                t_fit,
                DC=fit.params["DC"].value,
                A1=fit.params["A1"].value,
                R1=fit.params["R1"].value,
            )
            debug = False
            if debug:
                pw = pg.plot(
                    time_base, mean_trace, pen="g", title=f"Averaged Fit: {self.Clamps.protocol!s}"
                )
                pw.plot(x=time_base_fit, y=fit_curve, pen=pg.mkPen("r", width=4, linestyle="--"))

            xf, yf = time_base_fit, fit_curve
            fparx = [fit.params["DC"].value, fit.params["A1"].value, 1.0 / fit.params["R1"].value]
            namesx = ["DC", "A", "taum"]
            fpar.append(fparx)
            names.append(namesx)
            okdata.append(0)
            self.taum_fitted = {0: [xf, yf]}

            if debug:
                print("Fpar: ", fpar)
            self.taum_pars = fpar
            self.taum_win = time_window
            self.taum_func = Func
            self.taum_whichdata = okdata
            # raise ValueError("checking taus in average fit")
            self.taum_taum = fparx[2]

            self.analysis_summary["taum"] = self.taum_taum
            self.analysis_summary["taupars"] = fpar
            self.analysis_summary["taufunc"] = self.taum_func
            self.analysis_summary["taum_fitted"] = self.taum_fitted
            self.analysis_summary["taum_fitmode"] = "average"
            self.analysis_summary["taum_traces"] = self.taum_whichdata
            self.analysis_summary["ivss_cmd"] = self.Clamps.commandLevels[self.taum_whichdata]
            return

        # Otherwise we fit only those traces in a certain current range/voltage range,
        # and pay attention to the peak negativity if needed.

        # whichdata = whichdata[-1:]
        if debug:
            print("whichdata: ", whichdata)

        # ------------------------------------------------------------------
        # Fit single traces, new way
        # ------------------------------------------------------------------

        if USE_NEW_FIT_8_24:
            debug = False
            # use an estimator from the average of the traces to be fit
            # also exclude traces with APs before the current step
            # limit whichdata to traces where icmd is betwwen 0 and -200 pA.
            # implemented 8/24/2024.
            c_range = np.sort(self.taum_current_range)  # make sure [0] is less than [1]
            whichdata = np.argwhere(
                (self.Clamps.commandLevels >= c_range[0])
                & (self.Clamps.commandLevels <= c_range[1])
            )
            whichdata = whichdata.flatten()

            # for i in whichdata:
            #     print("command levels to test:", i, self.Clamps.commandLevels[i] * 1e12)
            if debug:
                print("\n\nNot averaging: TAU estimates with traces: ", whichdata)
                print("c_range: ", c_range * 1e12)
                print(self.Clamps.commandLevels * 1e12)
                print((self.Clamps.commandLevels >= c_range[0]))
                print((self.Clamps.commandLevels <= c_range[1]))
                print(
                    (self.Clamps.commandLevels >= c_range[0])
                    & (self.Clamps.commandLevels <= c_range[1])
                )

                print("taum_current_range: ", self.taum_current_range)
                print([l for l in self.Clamps.commandLevels * 1e12])
                print("   shape of all trace data: ", traces.shape)
                # print(len(ineg_valid), ineg_valid)
                # print("Spikes: ", len(self.Spikes.spikecount), self.Spikes.spikecount)

            if len(whichdata) == 0:
                if debug:
                    print("No data available for taum measurement")
                self.taum_pars = fpar
                self.taum_win = time_window
                self.taum_func = Func
                self.taum_whichdata = okdata
                self.taum_taum = np.nan
                self.analysis_summary["taum"] = self.taum_taum
                self.analysis_summary["taupars"] = fpar
                self.analysis_summary["taufunc"] = self.taum_func
                self.analysis_summary["taum_fitted"] = self.taum_fitted
                self.analysis_summary["taum_fitmode"] = "multiple"
                self.analysis_summary["taum_traces"] = self.taum_whichdata
                return  # no available taum data from this protocol.

            mean_trace = np.mean(traces[whichdata], axis=0)
            it0 = int(time_window[0] / dt)
            it1 = int(time_window[1] / dt)
            data_to_fit = mean_trace[it0:it1]
            t_fit = time_base[it0:it1] - time_base[i_fitwin[0]]
            igap = int(tgap / dt)
            ipeak = it1
            if peak_time:
                # find the peak of the hyperpolarization of the trace.
                # This is set as the END of the fit
                # We also account for a short time after the pulse (tgap) before actually
                # finding the minimum
                ipeak = (
                    int(np.argmin(savgol_filter(data_to_fit[igap:], window_length=5, polyorder=3)))
                    + igap
                )  # relative to start of the fit window
                if debug:
                    print("   ipeak: ", ipeak * dt)
                if (
                    ipeak * dt <= 0.020
                ):  # at least the defined duration of 20 ms, but could be longer.
                    ipeak = int(0.020 / dt)
                data_to_fit = data_to_fit[:ipeak]
                t_fit = t_fit[:ipeak]

            LME = exp_estimator_lmfit.LMexpFit()
            LME.initial_estimator(t_fit[igap:], data_to_fit[igap:], verbose=False)
            # print(f"   Init: DC: {LME.DC:8.3f}, A1: {LME.A1:8.3f}, R1: {LME.R1:8.3f}")
            if debug:
                pass
                # Generate 2 plots: one for the fit to the mean accepted trace (ineg_valid)
                # and a set for the individual traces
                # print(dir(self.Clamps))
                # pw = pg.plot(
                #     time_base, mean_trace, pen="g", title=f"New Fit all traces: {self.Clamps.protocol!s}"
                # )
                # pw = pg.plot(
                #     x=t_fit[igap:] + time_window[0],
                #     y=data_to_fit[igap:],
                #     pen=pg.mkPen("r", width=4, linestyle="--"),
                #     title=f"New Fit, mean data from: {self.Clamps.protocol!s}",
                # )

                # for i in range(traces.shape[0]):
                #     if ineg_valid[i]:
                #         width = 3.0
                #         lcolor = 'g'
                #         style = pg.QtCore.Qt.PenStyle.SolidLine
                #         ppen = pg.mkPen(color=lcolor, width=width, style=style)
                #     else:
                #         width = 0.5
                #         lcolor = 'w'
                #         style = pg.QtCore.Qt.PenStyle.DashLine
                #         ppen = pg.mkPen(color=lcolor, width=width, style=style)
                #     if i == 0:
                #         pw2 = pg.plot(
                #             time_base,
                #             traces[i],
                #             pen=ppen,
                #             title=f"Which: {self.Clamps.protocol!s}",
                #         )
                #     else:
                #         pw2.plot(time_base, traces[i], pen=ppen, width=width)

            # now for each trace, fit multiple traces

            pw = []
            fparx = [np.nan, np.nan, np.nan]
            xf = None
            yf = None
            epsilon = self.Clamps.sample_interval * 3
            bounds = sorted(self.taum_bounds)
            for i, k in enumerate(whichdata):
                data_to_fit = traces[k][it0 + igap : it0 + ipeak]
                t_fit = time_base[it0 + igap : it0 + ipeak] - time_base[it0]
                if debug:
                    print("      trace number: ", k)
                    print("      tfit shape, data_to_fit shape: ", t_fit.shape, data_to_fit.shape)
                if debug:
                    if i == 0:
                        pwa = pg.plot(
                            t_fit + time_window[0],
                            data_to_fit,
                            pen=pg.mkPen(pg.intColor(i + 1), width=1),
                            title=f"Fitted Traces: {str(Path(*Path(self.Clamps.protocol).parts[-4:]))!s}",
                        )
                        pw.append(pwa)
                # update the estimates for the tau
                LME.initial_estimator(t_fit, data_to_fit, verbose=False)
                fit = LME.fit1(
                    t_fit, data_to_fit, taum_bounds=self.taum_bounds, plot=False, verbose=False
                )
                fit_curve = LME.exp_decay1(
                    t_fit,
                    DC=fit.params["DC"].value,
                    A1=fit.params["A1"].value,
                    R1=fit.params["R1"].value,
                )
                if debug:
                    if i == 0:
                        pw[-1].plot(
                            x=t_fit + time_window[0],
                            y=fit_curve,
                            pen=pg.mkPen("r", width=2, linestyle="--"),
                        )

                    else:
                        pw[-1].plot(
                            t_fit + time_window[0],
                            data_to_fit,
                            pen=pg.mkPen(pg.intColor(i + 1), width=1),
                            title=f"Fitted Traces: {self.Clamps.protocol!s}",
                        )
                        pw[-1].plot(
                            x=t_fit + time_window[0],
                            y=fit_curve,
                            pen=pg.mkPen("r", width=2, linestyle="--"),
                        )

                xf, yf = t_fit + time_window[0], fit_curve
                if (
                    1.0 / fit.params["R1"].value < 0.0
                ):  # bounds on LME.fit1 should prevent this, but you never know
                    print("Negative tau: ", 1.0 / fit.params["R1"].value)
                    continue
                tau_k = 1.0/fit.params["R1"].value
                # print("tau_k: ", tau_k, fit.params["R1"].value)
                # only accept the fit value if it is not at the boundaries.
                if tau_k < (bounds[0]+epsilon) or tau_k > (bounds[1]-epsilon):
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
                    self.taum_fitted[k] = [xf, yf]
            taus = []
            # epsilon = self.Clamps.sample_interval * 3
            # bounds = sorted(self.taum_bounds)
            for j in range(len(fpar)):
                # tau_j = fpar[j][2]
                # print("tauj etc: ", tau_j, self.taum_bounds, epsilon)
                # if tau_j > (bounds[0]+epsilon) and tau_j < (bounds[1]-epsilon):
                taus.append(fpar[j][2])
            # print("taus: ", taus)
            if len(taus) > 0:
                self.taum_taum = np.nanmean(taus)
                self.analysis_summary["taum"] = self.taum_taum
            else:
                self.taum_taum = np.nan
                self.analysis_summary["taum"] = np.nan
            # if fit is against boundary, declare it invalid
            if self.taum_taum < (bounds[0]+epsilon) or self.taum_taum > (bounds[1]-epsilon):
                self.taum_taum = np.nan
                self.analysis_summary["taum"] = np.nan
            self.taum_pars = fpar
            self.taum_win = time_window
            self.taum_func = Func
            self.taum_whichdata = okdata
            self.analysis_summary["taupars"] = fpar
            self.analysis_summary["taufunc"] = self.taum_func
            self.analysis_summary["taum_fitted"] = self.taum_fitted
            self.analysis_summary["taum_fitmode"] = "multiple"
            self.analysis_summary["taum_traces"] = self.taum_whichdata
            # print("WHICH: ", self.taum_whichdata)
        #
        # ------------------------------------------------------------------
        # Fit single traces, old way
        # ------------------------------------------------------------------
        #
        elif USE_OLD_FIT:
            for j, k in enumerate(whichdata):
                taubounds = self.taum_bounds.copy()
                initpars[2] = np.mean(taubounds)

                if peak_time:
                    # find the peak of the hyperpolarization of the trace to do the fit.
                    # We account for a short time after the pulse (tgap) before actually
                    # finding the minimum, then reset the end time of the fit to the following
                    # peak negativity.
                    vtr1 = traces[k][int((time_window[0] + tgap) / dt) : int(time_window[1] / dt)]
                    ipeak = np.argmin(vtr1) + int(tgap / dt)
                    if ipeak * dt < 0.002:  # force a minimum fit window of 2 msec
                        ipeak = 0.002 / dt
                        time_window[1] = 0.002 + time_window[0]
                    else:
                        time_window[1] = (ipeak + 1) * dt + time_window[0]
                    vtr2 = traces[k][int(time_window[0] / dt) : int(time_window[1] / dt)]
                    v0 = vtr2[0]
                    v1 = vtr2[-1] - v0
                    for m in range(len(vtr2)):
                        if vtr2[m] - v0 <= 0.63 * v1:
                            break
                    if debug:
                        print("peak time true, fit window = ", time_window)
                        print("initial estimate for tau: (pts, time)", m, m * dt)
                    taubounds[0] = 0.0002
                    taubounds[1] = np.min((time_window[1] - time_window[0], 100.0))
                    # ensure that the bounds are ordered and have some range
                    if taubounds[1] < 10.0 * taubounds[0]:
                        taubounds[1] = taubounds[0] * 10.0
                    if debug:
                        print("timewindow: ", time_window)
                        print("taubounds: ", taubounds)

                    tau_init = m * dt
                    if tau_init >= taubounds[0] and tau_init <= taubounds[1]:
                        initpars[2] = tau_init
                    else:
                        initpars[2] = 0.5 * ipeak * dt
                    # print('inits: ', initpars)

                if debug:
                    print(
                        "times: ", time_window, len(time_base), np.min(time_base), np.max(time_base)
                    )
                    print("taubounds: ", taubounds)

                (fparx, xf, yf, namesx) = Fits.FitRegion(
                    [k],
                    thisaxis=whichaxis,
                    tdat=time_base,
                    ydat=traces,
                    dataType="2d",
                    t0=time_window[0],
                    t1=time_window[1],
                    fitFunc=Func,
                    fitPars=initpars,
                    fixedPars=[tgap],
                    tgap=tgap,
                    method="SLSQP",
                    # 11/8/2023: tighten up the baseline bounds and use the trace's own baseline for fit
                    bounds=[(vbl[k] - 0.005, vbl[k] + 0.005), (0, 0.05), (taubounds)],
                    capture_error=True,
                )
                debug = True
                if debug:
                    pw = pg.plot(
                        self.Clamps.time_base,
                        np.array(self.Clamps.traces[k]),
                        "k-",
                        title=f"Old method Tau Membrane Fit: {self.Clamps.protocol!s}",
                    )
                    pw.plot(xf[0], yf[0], "r--", linewidth=1)
                    pw.plot([np.min(xf[0]), np.max(xf[0])], [vbl[k], vbl[k]], "b--", linewidth=0.5)

                # exit(1)

                if not fparx:
                    raise Exception(
                        "IVCurve::update_Tau_membrane: Charging tau fitting failed - see log"
                    )
                # print 'j: ', j, len(fpar)
                # if fparx[0][1] < 2.5e-3:  # amplitude must be > 2.5 mV to be useful
                #     continue
                # if fit time constant is against boundary, declare it invalid
                if fparx[0][2] in taubounds:
                    fparx[0][2] = np.nan
                fpar.append(fparx[0])
                names.append(namesx[0])
                okdata.append(k)
                self.taum_fitted[k] = [xf[0], yf[0]]

        if debug:
            print("Fpar: ", fpar)
        self.taum_pars = fpar
        self.taum_win = time_window
        self.taum_func = Func
        self.taum_whichdata = okdata
        taus = []
        for j in range(len(fpar)):
            if np.isnan(fpar[j][2]):
                continue
            outstr = ""
            taus.append(fpar[j][2])
            for i in range(0, len(names[j])):
                outstr += "%s = %f, " % (names[j][i], fpar[j][i])
            if printWindow:
                print(("FIT(%d, %.1f pA): %s " % (whichdata[j], itaucmd[j] * 1e12, outstr)))
        if len(taus) > 0:
            self.taum_taum = np.nanmean(taus)
            self.analysis_summary["taum"] = self.taum_taum
        else:
            self.taum_taum = np.nan
            self.analysis_summary["taum"] = np.nan

        if len(self.taum_pars) > 0:
            if isinstance(self.taum_pars, list):
                self.analysis_summary["taupars"] = self.taum_pars[0]
            else:
                self.analysis_summary["taupars"] = self.taum_pars[0].tolist()

        else:
            self.analysis_summary["taupars"] = self.taum_pars
        self.analysis_summary["taufunc"] = self.taum_func
        self.analysis_summary["taum_fitted"] = self.taum_fitted
        self.analysis_summary["taum_fitmode"] = "multiple"
        self.analysis_summary["taum_traces"] = self.taum_whichdata

    def rmp_analysis(self, time_window: list = []):
        """
        Get the resting membrane potential

        Parameters
        ----------
        time_window : tuple, list or numpy array with 2 values (default: None)
            start and end time of a trace used to measure the RMP across
            traces.

        Return
        ------
        Nothing

        Stores computed RMP in mV in the class variable rmp.
        """
        assert len(time_window) == 2

        data1 = self.Clamps.traces["Time" : time_window[0] : time_window[1]]
        data1 = data1.view(np.ndarray)
        self.ivbaseline = data1.mean(axis=1)  # all traces
        self.ivbaseline_cmd = self.Clamps.commandLevels
        self.rmp = np.mean(self.ivbaseline) * 1e3  # convert to mV
        self.rmp_sd = np.std(self.ivbaseline) * 1e3
        data2 = self.Clamps.cmd_wave["Time" : time_window[0] : time_window[1]]
        self.irmp = np.mean(data2.view(np.ndarray).mean(axis=1))
        # get the RMP_Zero from any runs where the injected current is < 10 pA from 0
        self.analysis_summary["RMP"] = self.rmp
        # get the RMP_Zero from any runs where the injected current is < 10 pA from 0
        if self.irmp >= -10e-12 and self.irmp >= -10e-12:
            self.analysis_summary["RMP_Zero"] = self.rmp
        else:
            self.analysis_summary["RMP_Zero"] = np.nan
        self.analysis_summary["RMP_SD"] = self.rmp_sd
        self.analysis_summary["RMPs"] = self.ivbaseline.tolist()  # save raw baselines as well
        self.analysis_summary["Irmp"] = self.irmp.tolist()

    def ivss_analysis(self, time_window: list = []):
        """
        compute steady-state IV curve - from the mean voltage
        across the stimulus set over the defined time region
        (this usually will be the last half or third of the trace,
        but may be specifically defined)

        Parameters
        ----------
        region : list or tuple
            Start and end times for the analysis
        """
        assert len(time_window) == 2
        data1 = self.Clamps.traces["Time" : time_window[0] : time_window[1]]
        self.r_in = np.nan
        self.analysis_summary["Rin"] = np.nan
        self.ivss_v = []
        self.ivss_v_all = []
        self.ivss_cmd = []
        self.ivss_cmd_all = []
        if data1.shape[1] == 0 or data1.shape[0] == 1:
            return  # skip it

        # check out whether there are spikes in the window that is selected
        threshold = self.Spikes
        ntr = len(self.Clamps.traces)
        if not self.Spikes.spikes_counted:
            print("ivss_analysis: spikes not counted yet? - let's go analyze them...")
            self.analyzeSpikes()

        self.ivss_v_all = data1.mean(axis=1)  # all traces
        self.analysis_summary["Rin"] = np.nan
        # print("*************** len self.Spikes.nospk: ", len(self.Spikes.nospk))
        if len(self.Spikes.nospk) >= 1:
            # Steady-state IV where there are no spikes
            # however, handle case where there are spikes at currents LESS
            # than some of those with no spikes.
            ivss_valid = self.Clamps.commandLevels < 0.5e-9
            # handle an upper current limit on the measure of ivss
            # This is so that we don't try to measure Rin for positive, but subthreshold steps,
            # where a persistent Na might interfere.
            # print("************** ivss valid before current limit: ", ivss_valid)
            if not np.isnan(self.rin_current_limit):
                ivss_valid = ivss_valid & (self.Clamps.commandLevels <= self.rin_current_limit)
            # print("*************** ivss valid after current limit: ", ivss_valid)
            if len(ivss_valid) != len(self.Spikes.spikecount) and np.isnan(self.rin_current_limit):
                print("Valid traces for IVSS is not the same as the number of traces in spikecount")
                return  # somehow is invalid...
            ivss_valid = ivss_valid & (self.Spikes.spikecount == 0)
            if not any(ivss_valid):
                print(" *********** No valid traces for IVSS analysis *********") 
                return
            self.ivss_v = self.ivss_v_all[ivss_valid]
            self.ivss_cmd_all = self.Clamps.commandLevels[ivss_valid]
            self.ivss_cmd = self.ivss_cmd_all
            isort = np.argsort(self.ivss_cmd)
            self.ivss_cmd = self.ivss_cmd[isort]
            self.ivss_v = self.ivss_v[isort]
            bl = self.ivbaseline[isort]
            self.ivss_bl = bl
            self.analysis_summary["ivss_cmd"] = self.ivss_cmd
            self.analysis_summary["ivss_v"] = self.ivss_v
            # compute Rin from the SS IV:
            # this makes the assumption that
            # successive trials are in order, so we sort above
            # commands are not repeated...
            if len(self.ivss_cmd) > 2 and len(self.ivss_v) > 2:
                with warnings.catch_warnings(record=True) as w:
                    pf = np.polyfit(
                        self.ivss_cmd,
                        self.ivss_v,
                        3,
                        rcond=None,
                        full=False,
                        w=None,
                        cov=False,
                    )
                if w:
                    print(f"*********** Polyfit in ivss_analysis: Warning: {w[0].message}")
                    print("     We do not use the fits if they are poorly conditioned, returning")
                    return
                
                def pderiv(pf, x):
                    y = 3 * pf[0] * x**2 + 2 * pf[1] * x + pf[2]
                    return y

                # pval = np.polyval(pf, self.ivss_cmd)

                slope = pderiv(
                    pf, np.array(self.ivss_cmd)
                )  # np.diff(pval[iasort]) / np.diff(self.ivss_cmd[iasort])  # local slopes
                imids = np.array((self.ivss_cmd[1:] + self.ivss_cmd[:-1]) / 2.0)
                self.rss_fit = {"I": imids, "V": np.polyval(pf, imids), "pars": pf}
                # print('fit V: ', self.rss_fit['V'])
                # slope = slope[[slope > 0 ] and [self.ivss_cmd[:-1] > -0.8] ] # only consider positive slope points
                l = int(len(slope) / 2)
                if len(slope) > 1:
                    maxloc = np.argmax(slope[l:]) + l
                    minloc = np.argmin(slope[:l])
                else:
                    maxloc = 0
                    minloc = 0
                # print("ivss cmd max loc: ", self.ivss_cmd[maxloc])
                # if self.ivss_cmd[maxloc] >= 0.0 :
                #     return  # must be in response to hyperpolarization
                self.r_in = slope[maxloc]
                self.r_in_loc = [
                    self.ivss_cmd[maxloc],
                    self.ivss_v[maxloc],
                    maxloc,
                ]  # where it was found
                self.r_in_min = slope[minloc]
                self.r_in_minloc = [
                    self.ivss_cmd[minloc],
                    self.ivss_v[minloc],
                    minloc,
                ]  # where it was founds
                self.analysis_summary["Rin"] = self.r_in * 1.0e-6
                self.analysis_summary["ivss_cmd"] = self.ivss_cmd
                self.analysis_summary["ivss_v"] = self.ivss_v
                self.analysis_summary["ivss_bl"] = self.ivss_bl
                self.analysis_summary["ivss_fit"] = self.rss_fit
            else:
                self.analysis_summary["Rin"] = np.nan
                self.analysis_summary["ivss_cmd"] = []
                self.analysis_summary["ivss_v"] = []
                self.analysis_summary["ivss_bl"] = np.nan
                self.analysis_summary["ivss_fit"] = []

    def ivpk_analysis(self, time_window: list = []):
        """
        compute peak IV curve - from the minimum voltage
        across the stimulus set

        Parameters
        ----------
        region : list or tuple
            Start and end times for the analysis
        """
        assert len(time_window) == 2

        self.r_in_peak = np.nan
        self.analysis_summary["Rin_peak"] = np.nan
        self.ivpk_cmd = []
        self.ivpk_cmd_all = []
        self.ivpk_v = []
        self.ivpk_v_all = []
        data1 = self.Clamps.traces["Time" : time_window[0] : time_window[1]]
        if data1.shape[1] == 0 or data1.shape[0] == 1:
            return  # skip it

        # check out whether there are spikes in the window that is selected
        threshold = self.Spikes
        ntr = len(self.Clamps.traces)
        if not self.Spikes.spikes_counted:
            print("ivss_analysis: spikes not counted yet? - let's go analyze them...")
            self.analyzeSpikes()

        self.ivpk_v_all = data1.min(axis=1)  # all traces, minimum voltage found
        if len(self.Spikes.nospk) >= 1:
            # print("ivpk_analysis: nospk: ", self.Spikes.nospk)
            # print("ivpk_analysis: ivpk_v_all: ", self.ivpk_v_all)
            # Steady-state IV where there are no spikes
            self.ivpk_v = self.ivpk_v_all[self.Spikes.nospk]
            self.ivpk_cmd_all = self.Clamps.commandLevels
            # if np.max(self.Spikes.nospk) >= len(self.ivss_cmd_all):
            #     return
            self.ivpk_cmd = self.ivpk_cmd_all[self.Spikes.nospk]


            bl = self.ivbaseline[self.Spikes.nospk]
            isort = np.argsort(self.ivpk_cmd)
            self.ivpk_cmd = self.ivpk_cmd[isort]
            self.ivpk_v = self.ivpk_v[isort]
            bl = bl[isort]
            self.ivpk_bl = bl
            if len(self.ivpk_cmd) > 2 and len(self.ivpk_v) > 2:
                with warnings.catch_warnings(record=True) as w:
                    warnings.simplefilter("always")
                    pf = np.polyfit(
                        self.ivpk_cmd,
                        self.ivpk_v,
                        3,
                        rcond=None,
                        full=False,
                        w=None,
                        cov=False,
                    )
                if w:
                    print(f"*********** Polyfit in ivpk_analysis: Warning: {w[0].message}")
                    print("     We do not use the fits if they are poorly conditioned, returning")
                    return
                
                def pderiv(pf, x):
                    y = 3 * pf[0] * x**2 + 2 * pf[1] * x + pf[2]
                    return y

                # pval = np.polyval(pf, self.ivss_cmd)

                slope = pderiv(pf, np.array(self.ivpk_cmd))
                # np.diff(pval[iasort]) / np.diff(self.ivss_cmd[iasort])  # local slopes
                # pval = np.polyval(pf, self.ivpk_cmd)
                # slope = np.diff(pval) / np.diff(self.ivpk_cmd)
                imids = np.array((self.ivpk_cmd[1:] + self.ivpk_cmd[:-1]) / 2.0)
                self.rpk_fit = {"I": imids, "V": np.polyval(pf, imids)}
                l = int(len(slope) / 2)
                if len(slope) > 1:
                    maxloc = np.argmax(slope[l:]) + l
                    minloc = np.argmin(slope[:l])
                else:
                    maxloc = 0
                    minloc = 0
                self.r_in_peak = slope[maxloc]
                self.r_in_peak_loc = [
                    self.ivpk_cmd[maxloc],
                    self.ivpk_v[maxloc],
                    maxloc,
                ]  # where it was found
                self.r_in_minpeak = slope[minloc]
                self.r_in_minpeak_loc = [
                    self.ivpk_cmd[minloc],
                    self.ivpk_v[minloc],
                    minloc,
                ]  # where it was found
                self.analysis_summary["Rin_peak"] = self.r_in_peak * 1.0e-6
                self.analysis_summary["ivpk_cmd"] = self.ivpk_cmd
                self.analysis_summary["ivpk_v"] = self.ivpk_v
                self.analysis_summary["ivpk_bl"] = self.ivpk_bl
                self.analysis_summary["ivpk_fit"] = self.rpk_fit

    def leak_subtract(self):
        self.yleak = np.zeros(len(self.ivss_v))
        # basically, should not do this blind...so it is commented out.

        # if self.ctrl.IVCurve_subLeak.isChecked():
        #     if self.Clamps.data_mode in self.dataModel.ic_modes:
        #         sf = 1e-12
        #     elif self.Clamps.data_mode in self.dataModel.vc_modes:
        #         sf = 1e-3
        #     else:
        #         sf = 1.0
        #     (x, y) = Utility.clipdata(self.ivss, self.ivss_cmd,
        #                               self.ctrl.IVCurve_LeakMin.value() * sf,
        #                               self.ctrl.IVCurve_LeakMax.value() * sf)
        #     try:
        #         p = np.polyfit(x, y, 1)  # linear fit
        #         self.yleak = np.polyval(p, self.ivss_cmd)
        #         self.ivss = self.ivss - self.yleak
        #     except:
        #         raise ValueError('IVCurve Leak subtraction: no valid points to correct')

    def tau_h(
        self,
        v_steadystate: float=-0.080,  # target steady-state voltage, V
        peak_timewindow: list = [],  # [start, end] time window for peak measurement
        steadystate_timewindow: list = [],  # [start, end] time window for steady-state measurement
        printWindow=False,
    ):
        """
        Measure the time constant associated with activation of the hyperpolarization-
        activated current, Ih. The tau is measured from the peak of the response to the
        end of the steady-state part, at a single current level.

        Parameters
        ----------
        v_steadystate : float (voltage, V; no default).
             The steady-state voltage that will be used as the target for measuring Ih. A single
             voltage level is given; the closest one in the test set that is negative to the
             resting potential will be used.

        peak_timewindow : list of floats ([time, time], ms; no default)
            The time window over which the peak voltage will be identified.

        steadystate_timewindow : list of floats ([time, time], ms; no default)
            The time window over which the steady-state voltage will be identified.

        Return
        ------
        Nothing

        Class variables with a leading tauh_ are set by this routine, to return the
        results of the measurements.

        """
        assert len(peak_timewindow) == 2
        assert len(steadystate_timewindow) == 2

        # initialize result variables
        self.tauh_vpk = np.nan  # peak voltage for the tau h meausure
        self.tauh_neg_pk = np.nan
        self.tauh_vss = np.nan  # ss voltage for trace used for tauh
        self.tauh_neg_ss = np.nan
        self.tauh_vrmp = np.nan
        self.tauh_xf = []
        self.tauh_yf = []
        self.tauh_fitted = {}
        self.tauh_meantau = np.nan
        self.tauh_bovera = np.nan
        self.tauh_Gh = np.nan
        self.analysis_summary["tauh_tau"] = self.tauh_meantau
        self.analysis_summary["tauh_bovera"] = self.tauh_bovera
        self.analysis_summary["tauh_Gh"] = self.tauh_Gh
        self.analysis_summary["tauh_vss"] = self.tauh_vss  # actual steady-state voltage
        self.analysis_summary["tauh_voltage"] = v_steadystate  # target measurement value
        print(self.rmp/1000., v_steadystate)
        if self.rmp/1000. < v_steadystate:  # don't measure if rmp is below the target steady-state value
            return

        Func = "exp1"  # single exponential fit to the seleccted region
        Fits = TOOLS.fitting.Fitting()

        # for our time windows, get the ss voltage to use
        ss_voltages = self.Clamps.traces[
            "Time" : steadystate_timewindow[0] : steadystate_timewindow[1]
        ].view(np.ndarray)
        ss_voltages = ss_voltages.mean(axis=1)
        # find trace closest to test voltage at steady-state
        try:
            itrace = np.argmin((ss_voltages - v_steadystate) ** 2)  # ignore "no spikes?"
        except:
            return
        if np.fabs(ss_voltages[itrace] - v_steadystate) > 0.005:
            print("no trace close enough to target vss for tau_h measurement")
            return
        pk_voltages = self.Clamps.traces["Time" : peak_timewindow[0] : peak_timewindow[1]].view(
            np.ndarray
        )
        pk_voltages_tr = pk_voltages.min(axis=1)
        ipk_start = pk_voltages[itrace].argmin()
        ipk_start += int(
            peak_timewindow[0] / self.Clamps.sample_rate[itrace]
        )  # get starting index as well
        pk_time = self.Clamps.time_base[ipk_start] + self.Clamps.tstart
        if pk_time > self.Clamps.tend:
            pk_time = self.Clamps.tend - 0.050
        if not self.Spikes.spikes_counted:
            self.analyzeSpikes()

        # now find trace with voltage closest to target steady-state voltage
        # from traces without spikes in the standard window
        whichdata = [int(itrace)]
        # prepare to fit
        initpars = [-80.0 * 1e-3, -10.0 * 1e-3, 50.0 * 1e-3]
        bounds = [(-0.15, 0.0), (-0.1, 0.1), (0, 5.0)]

        v_rmp = self.ivbaseline[itrace]
        itaucmd = self.Clamps.commandLevels[itrace]
        if itaucmd is None or np.fabs(itaucmd) < 1e-11:
            return  # don't attempt to fit a tiny current
        whichaxis = 0
        (fpar, xf, yf, names) = Fits.FitRegion(
            whichdata,
            whichaxis,
            self.Clamps.time_base,
            self.Clamps.traces.view(np.ndarray),
            dataType="2d",
            t0=pk_time,
            t1=steadystate_timewindow[1],
            fitFunc=Func,
            fitPars=initpars,
            method="Nelder-Mead",  # "SLSQP",
            bounds=[
                (-0.120, 0.05),
                (-0.1, 0.1),
                (0.001, (steadystate_timewindow[1] - pk_time) * 2.0),
            ],
        )
        if not fpar:
            raise Exception("IVCurve::update_Tauh: tau_h fitting failed")
        s = np.shape(fpar)
        taus = []
        for j in range(0, s[0]):
            outstr = ""
            taus.append(fpar[j][2])
            for i in range(0, len(names[j])):
                outstr += "%s = %f, " % (names[j][i], fpar[j][i])
            if printWindow:
                print(("Ih FIT(%d, %.1f pA): %s " % (whichdata[j], itaucmd[j] * 1e12, outstr)))
        self.tauh_fitted[itrace] = [xf[0], yf[0]]
        self.tauh_vrmp = self.ivbaseline[itrace]
        self.tauh_vss = ss_voltages[itrace]
        self.tauh_vpk = pk_voltages_tr[itrace]
        self.tauh_neg_ss = self.tauh_vss - self.tauh_vrmp
        self.tauh_neg_pk = self.tauh_vpk - self.tauh_vrmp
        self.tauh_xf = xf
        self.tauh_yf = yf
        self.tauh_meantau = np.mean(taus)
        self.tauh_bovera = (self.tauh_vss - self.tauh_vrmp) / (self.tauh_vpk - self.tauh_vrmp)
        if self.tauh_bovera > 1.0:
            self.tauh_bovera = 1.0
        # print("Gh: ", itaucmd, self.tauh_neg_ss)
        Gpk = itaucmd / self.tauh_neg_pk  # units are A / V = S
        Gss = itaucmd / self.tauh_neg_ss
        self.tauh_Gh = Gss - Gpk
        if self.tauh_Gh < 0:
            self.tauh_Gh = 0.0

        self.analysis_summary["tauh_tau"] = self.tauh_meantau
        self.analysis_summary["tauh_bovera"] = self.tauh_bovera
        self.analysis_summary["tauh_Gh"] = self.tauh_Gh
        self.analysis_summary["tauh_vss"] = self.tauh_vss
        self.analysis_summary["tauh_fitted"] = self.tauh_fitted

        # print(self.analysis_summary)
