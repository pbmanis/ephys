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


from collections import OrderedDict
import numpy as np
import pyqtgraph as pg
from typing import List, Union
from ..tools import Fitting


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
        self.Clamps = None
        self.Spikes = None
        self.dataPlot = None
        self.baseline = None
        self.rmp = None
        self.taum_fitted = {}
        self.taum_bounds = []
        self.analysis_summary = {}

    def setup(
        self,
        clamps=None,
        spikes=None,
        dataplot=None,
        baseline=[0, 0.001],
        bridge_offset=0,
        taumbounds=[0.001, 0.050],
        tauhvoltage=-0.08,
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

        taumbounds : list (2 elements)
            Lower and upper bounds of the allowable taum fitting range (in seconds).

        bridge_offset : float (default:  0.0 Ohms)
            Bridge offset resistance (Ohms)
        """

        if clamps is None or spikes is None:
            raise ValueError(
                "RmTau analysis requires defined clamps and spike analysis"
            )
        self.Clamps = clamps
        self.Spikes = spikes
        self.dataPlot = dataplot
        self.baseline = baseline
        self.bridge_offset = bridge_offset
        self.taum_fitted = {}
        self.tauh_fitted = {}
        self.taum_bounds = taumbounds
        self.tauh_voltage = tauhvoltage
        self.analysis_summary["holding"] = self.Clamps.holding
        self.analysis_summary["WCComp"] = self.Clamps.WCComp
        self.analysis_summary["CCComp"] = self.Clamps.CCComp
        if self.bridge_offset != 0.0:
            self.bridge_adjust()
        self.analysis_summary[
            "BridgeAdjust"
        ] = self.bridge_offset  # save the bridge offset value

    def bridge_adjust(self):
        """
        Adjust the voltage waveform according to the bridge offset value
        """
        print("RmTau adjusting bridge...")
        self.Clamps.traces = (
            self.Clamps.traces
            - self.Clamps.cmd_wave.view(np.ndarray) * self.bridge_offset
        )

    def analyze(
        self, rmpregion=[0.0, 0.05], tauregion=[0.1, 0.125], to_peak=False, tgap=0.0005
    ):

        # print("rmpregion: ", rmpregion)
        # print("tauregion: ", tauregion)
        if tauregion[1] > self.Clamps.tend:
            tauregion[1] = self.Clamps.tend
        stepdur = (self.Clamps.tend - self.Clamps.tstart)
        self.rmp_analysis(time_window=rmpregion)
        self.tau_membrane(time_window=tauregion, peak_time=to_peak, tgap=tgap)
        r_pk = self.Clamps.tstart + 0.4 * stepdur
        r_ss = self.Clamps.tstart + 0.9 * stepdur  # steady-state region
        self.ivss_analysis(time_window=[r_ss, self.Clamps.tend])
        self.ivpk_analysis(time_window=[self.Clamps.tstart, r_pk])  # peak region
        # print("r_ss: ", r_ss, self.Clamps.tend)
        self.tau_h(
            self.tauh_voltage,
            peak_timewindow=[r_pk, r_ss], # self.Clamps.tstart, r_pk],
            steadystate_timewindow=[r_ss, self.Clamps.tend],
            printWindow=False,
        )

    def tau_membrane(
        self,
        time_window: list = [],
        peak_time: bool = False,
        printWindow: bool = False,
        whichTau: int = 1,
        vrange: list = [-0.002, -0.050],
        tgap: float = 0.0,
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

        Return
        ------
            Nothing

        Class variables with a leading taum\_ are set by this routine, to return results.

        """

        assert len(time_window) == 2
        debug = False

        if debug:
            print("taum")
            print("time_window: ", time_window)
            print("PEAK TIME: ", peak_time)
        Func = "exp1"  # single exponential fit with DC offset.
        if self.rmp is None:
            self.rmp_analysis(time_window=self.baseline)

        Fits = Fitting.Fitting()  # get a fitting instance
        initpars = [self.rmp * 1e-3, -0.010, 0.010]  # rmp is in units of mV
        icmdneg = np.where(self.Clamps.commandLevels < -10e-12)
        maxcmd = np.min(self.Clamps.commandLevels)
        ineg = np.where(self.Clamps.commandLevels[icmdneg] < 0.0)
        if debug:
            print('command levels: ', self.Clamps.commandLevels)
            print("ineg: ", ineg)
        # if peak_time is not None and ineg != np.array([]):
        #     rgnpk[1] = np.max(peak_time[ineg[0]])
        dt = self.Clamps.sample_interval
        time_base = self.Clamps.time_base.view(np.ndarray)
        time_base_fit = time_base[int(time_window[0]/dt) : int(time_window[1]/dt)]
        traces = self.Clamps.traces.view(np.ndarray)
        vrange = np.sort(vrange)         # vrange is in mV
        vmeans = (  # mean of trace at the last 1 msec the time window, as difference from baseline
            np.mean(traces[:, int((time_window[1]-0.001)/dt) : int(time_window[1]/dt)], axis=1)
                - self.ivbaseline
        )

        indxs = np.where(
            np.logical_and((vmeans[ineg] >= vrange[0]), (vmeans[ineg] <= vrange[1]))
        )
        if debug:
            print('baseline: ', self.ivbaseline)
            print('vrange: ', vrange)
            print('vmeans: ', vmeans.view(np.ndarray))
            print('indxs: ', indxs)
            print('ineg: ', ineg)
            print('self.Clamps.commandLevels', self.Clamps.commandLevels.view(np.ndarray))
            print("IV baseline: ", self.ivbaseline)
            print( 'ineg: ', ineg)
            print( 'icmdneg: ', icmdneg)
            print( 'vrange: ', vrange)
            print( 'vmeans[ineg]: ', vmeans[ineg].view(np.ndarray))
            print( 'indxs: ', indxs)
        indxs = list(indxs[0])
        if len(ineg) == 0:  # if there are no traces that match, just return
            return
        whichdata = ineg[0][indxs]  # restricts to valid values
        # print('whichdata: ', whichdata)
        itaucmd = self.Clamps.commandLevels[ineg]
        whichaxis = 0
        fpar = []
        names = []
        okdata = []
        if len(list(self.taum_fitted.keys())) > 0 and self.dataPlot is not None:
            [self.taum_fitted[k].clear() for k in list(self.taum_fitted.keys())]
        self.taum_fitted = {}
        whichdata = whichdata[-1:]
        if debug:
            print("whichdata: ", whichdata)
        for j, k in enumerate(whichdata):
            if debug:
                import matplotlib.pyplot as mpl
                n = "w_%03d" % j
                fig, ax = mpl.subplots(3,1)
                ax[0].plot(
                    time_base, traces[k], "k-"
                )
                xspan = ax[0].get_xlim()
            taubounds = self.taum_bounds.copy()
            initpars[2] = np.mean(taubounds)
            if debug:
                print("Fitting for k: ", k, self.Clamps.commandLevels[k])
                import matplotlib.pyplot as mpl
                print(peak_time)
                print(time_window, dt, time_window[0]/dt, time_window[1]/dt)
                ax[1].plot(time_base_fit,
                    traces[k][int(time_window[0] / dt) : int(time_window[1] / dt)], 'b-')
                ax[1].set_xlim(xspan)
            

            if peak_time:
                # find the peak of the hyperpolarization of the trace to do the fit. 
                # We account for a short time after the pulse (tgap) before actually
                # finding the minimum, then reset the end time of the fit to the following
                # peak negativity.
                vtr1 = traces[k][int((time_window[0]+tgap) / dt) : int(time_window[1] / dt)]
                ipeak = np.argmin(vtr1)+int(tgap/dt)
                time_window[1] = (ipeak * dt) + time_window[0]
                vtr2 = traces[k][int(time_window[0] / dt) : int(time_window[1] / dt)]
                v0 = vtr2[0]
                v1 = vtr2[-1] - v0
                for m in range(len(vtr2)):
                    if vtr2[m] - v0 <= 0.63 * v1:
                        break
                if debug:
                    print("peak time true, fit window = ", time_window)
                    print("initial estimate for tau: (pts, time)", m, m*dt)
                taubounds[0] = 0.0002
                taubounds[1] = np.min((time_window[1]-time_window[0], 100.))
                # ensure that the bounds are ordered and have some range
                if taubounds[1] < 10.0*taubounds[0]:
                    taubounds[1] = taubounds[0]*10.0
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
                print("times: ", time_window, len(time_base), np.min(time_base), np.max(time_base))
                print('taubounds: ', taubounds)

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
                tgap = tgap,
                method="SLSQP",
                bounds=[(-0.1, 0.0), (-0.05, 0.05), (taubounds)],
                capture_error = True,
            )

            if not fparx:
                raise Exception(
                    "IVCurve::update_Tau_membrane: Charging tau fitting failed - see log"
                )
            # print 'j: ', j, len(fpar)
            # if fparx[0][1] < 2.5e-3:  # amplitude must be > 2.5 mV to be useful
            #     continue
            fpar.append(fparx[0])
            names.append(namesx[0])
            okdata.append(k)
            self.taum_fitted[k] = [xf[0], yf[0]]
            if debug:
                ax[1].plot(xf[0], yf[0], 'r--')
                ax[2].plot(time_base, traces[k], 'm-')

                mpl.show()
            #    ax[1].set_xlim(xspan)
            # import matplotlib.pyplot as mpl
            # mpl.plot(self.Clamps.time_base, np.array(self.Clamps.traces[k]), 'k-')
            # mpl.plot(xf[0], yf[0], 'r--', linewidth=1)
            # mpl.show()
            # exit(1)
        if debug:
            print("Fpar: ", fpar)
        self.taum_pars = fpar
        self.taum_win = time_window
        self.taum_func = Func
        self.taum_whichdata = okdata
        taus = []
        for j in range(len(fpar)):
            outstr = ""
            taus.append(fpar[j][2])
            for i in range(0, len(names[j])):
                outstr += "%s = %f, " % (names[j][i], fpar[j][i])
            if printWindow:
                print(
                    (
                        "FIT(%d, %.1f pA): %s "
                        % (whichdata[j], itaucmd[j] * 1e12, outstr)
                    )
                )
        if len(taus) > 0:
            self.taum_taum = np.nanmean(taus)
            self.analysis_summary["taum"] = self.taum_taum
        else:
            self.taum_taum = np.NaN
            self.analysis_summary["taum"] = np.NaN
        if len(self.taum_pars) > 0:
            self.analysis_summary["taupars"] = self.taum_pars[0].tolist()
        else:
            self.analysis_summary["taupars"] = self.taum_pars
        self.analysis_summary["taufunc"] = self.taum_func

    def rmp_analysis(self, time_window:list=[]):
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
        self.irmp = data2.view(np.ndarray).mean(axis=1)
        self.analysis_summary["RMP"] = self.rmp
        self.analysis_summary["RMP_SD"] = self.rmp_sd
        self.analysis_summary["RMPs"] = self.ivbaseline.tolist()  # save raw baselines as well
        self.analysis_summary["Irmp"] = self.irmp.tolist()

    def ivss_analysis(self, time_window:list=[]):
        """
        compute steady-state IV curve - from the mean voltage
        across the stimulus set over the defined time region
        (this usually will be the last half or third of the trace)

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
        self.analysis_summary["Rin"] = np.NaN
        if len(self.Spikes.nospk) >= 1:
            # Steady-state IV where there are no spikes
            self.ivss_v = self.ivss_v_all[self.Spikes.nospk]
            self.ivss_cmd_all = self.Clamps.commandLevels
            # print("** ", self.Spikes.nospk)
            # print("    ", np.max(self.Spikes.nospk))
            # print("   ", len(self.ivss_cmd_all), "**")
            if len(self.Spikes.nospk[0]) == 0:
                return
            if np.max(self.Spikes.nospk) >= len(self.ivss_cmd_all):
                return
            self.ivss_cmd = self.ivss_cmd_all[self.Spikes.nospk]
            isort = np.argsort(self.ivss_cmd)
            self.ivss_cmd = self.ivss_cmd[isort]
            self.ivss_v = self.ivss_v[isort]
            bl = self.ivbaseline[isort]
            self.ivss_bl = bl
            # compute Rin from the SS IV:
            # this makes the assumption that:
            # successive trials are in order so we wort above
            # commands are not repeated...
            if len(self.ivss_cmd) > 2 and len(self.ivss_v) > 2:
                pf = np.polyfit(
                    self.ivss_cmd,
                    self.ivss_v,
                    3,
                    rcond=None,
                    full=False,
                    w=None,
                    cov=False,
                )
                def pderiv(pf, x):
                    y = 3*pf[0]*x**2 + 2*pf[1]*x + pf[2]
                    return y
                
                # pval = np.polyval(pf, self.ivss_cmd)

                slope = pderiv(pf, np.array(self.ivss_cmd)) # np.diff(pval[iasort]) / np.diff(self.ivss_cmd[iasort])  # local slopes
                imids = np.array((self.ivss_cmd[1:] + self.ivss_cmd[:-1]) / 2.0)
                self.rss_fit = {"I": imids, "V": np.polyval(pf, imids)}
                # print('fit V: ', self.rss_fit['V'])
                # slope = slope[[slope > 0 ] and [self.ivss_cmd[:-1] > -0.8] ] # only consider positive slope points
                l = int(len(slope) / 2)
                if len(slope) > 1:
                    maxloc = np.argmax(slope[l:]) + l
                    minloc = np.argmin(slope[:l])
                else:
                    maxloc = 0
                    minloc = 0
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
                ]  # where it was found
                self.analysis_summary["Rin"] = self.r_in * 1.0e-6

    def ivpk_analysis(self, time_window:list = []):
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
                pf = np.polyfit(
                    self.ivpk_cmd,
                    self.ivpk_v,
                    3,
                    rcond=None,
                    full=False,
                    w=None,
                    cov=False,
                )

                def pderiv(pf, x):
                    y = 3*pf[0]*x**2 + 2*pf[1]*x + pf[2]
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

    def tau_h(self, v_steadystate, peak_timewindow:list=[], steadystate_timewindow:list=[], printWindow=False):
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

        # initialize result varibles
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
        self.analysis_summary["tauh_vss"] = self.tauh_vss

        if self.rmp / 1000.0 < v_steadystate:  # rmp is in mV...
            return

        Func = "exp1"  # single exponential fit to the seleccted region
        Fits = Fitting.Fitting()

        # for our time windows, get the ss voltage to use
        ss_voltages = self.Clamps.traces[
            "Time" : steadystate_timewindow[0] : steadystate_timewindow[1]
        ].view(np.ndarray)
        ss_voltages = ss_voltages.mean(axis=1)
        # find trace closest to test voltage at steady-state
        try:
            itrace = np.argmin((ss_voltages[self.Spikes.nospk] - v_steadystate) ** 2)
        except:
            return
        pk_voltages = self.Clamps.traces["Time" : peak_timewindow[0] : peak_timewindow[1]].view(
            np.ndarray
        )
        pk_voltages_tr = pk_voltages.min(axis=1)
        ipk_start = pk_voltages[itrace].argmin()
        ipk_start += int(peak_timewindow[0] / self.Clamps.sample_rate[itrace])  # get starting index as well
        pk_time = self.Clamps.time_base[ipk_start]+self.Clamps.tstart
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
            method="Nelder-Mead", # "SLSQP",
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
                print(
                    (
                        "Ih FIT(%d, %.1f pA): %s "
                        % (whichdata[j], itaucmd[j] * 1e12, outstr)
                    )
                )
        self.tauh_fitted[itrace] = [xf[0], yf[0]]
        self.tauh_vrmp = self.ivbaseline[itrace]
        self.tauh_vss = ss_voltages[itrace]
        self.tauh_vpk = pk_voltages_tr[itrace]
        self.tauh_neg_ss = (self.tauh_vss - self.tauh_vrmp)
        self.tauh_neg_pk = (self.tauh_vpk - self.tauh_vrmp)
        self.tauh_xf = xf
        self.tauh_yf = yf
        self.tauh_meantau = np.mean(taus)
        self.tauh_bovera = (self.tauh_vss - self.tauh_vrmp) / (
            self.tauh_vpk - self.tauh_vrmp
        )
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

        # print(self.analysis_summary)
