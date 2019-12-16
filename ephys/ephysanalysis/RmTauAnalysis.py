"""
Analyze Rm, tau - pulled out of IVCurve 2/29/2016 pbm.
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
an instance of acq4read.

RmTauAnalysis requires that the SpikeAnalysis be run first.

Paul B. Manis, 2015-2019
for acq4

"""


from collections import OrderedDict
import os
import os.path
import itertools
import functools
import numpy as np
import scipy
import pyqtgraph as pg
import ephysanalysis.Fitting as Fitting
import pprint
import time

class RmTauAnalysis():
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
        self.rmp = []
        self.taum_fitted = {}
        self.taum_bounds = []
        self.analysis_summary = {}

        

    def setup(self, clamps=None, spikes=None, dataplot=None,
                baseline=[0, 0.001], bridge_offset=0,
                taumbounds = [0.001, 0.050], tauhvoltage=-0.08):
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
            raise ValueError("RmTau analysis requires defined clamps and spike analysis")
        self.Clamps = clamps
        self.Spikes = spikes
        self.dataPlot = dataplot
        self.baseline = baseline
        self.bridge_offset = bridge_offset
        self.taum_fitted = {}
        self.tauh_fitted = {}
        self.taum_bounds = taumbounds
        self.tauh_voltage = tauhvoltage
        self.analysis_summary['holding'] = self.Clamps.holding
        self.analysis_summary['WCComp'] = self.Clamps.WCComp
        self.analysis_summary['CCComp'] = self.Clamps.CCComp
        if self.bridge_offset != 0.0:
            self.bridge_adjust()
        self.analysis_summary['BridgeAdjust'] = self.bridge_offset  # save the bridge offset value
    
    def bridge_adjust(self):
        """
        Adjust the voltage waveform according to the bridge offset value
        """
        print('RmTau adjusting bridge...')
        self.Clamps.traces = self.Clamps.traces - self.Clamps.cmd_wave.view(np.ndarray)*self.bridge_offset
        
    
    def analyze(self, rmpregion=[0., 0.05], tauregion=[0.1, 0.125], to_peak=False, tgap=0.):
        self.rmp_analysis(region=rmpregion)
        self.tau_membrane(region=tauregion, peak_time=to_peak, tgap=tgap)
        r_ss = self.Clamps.tstart + 0.9*(self.Clamps.tend-self.Clamps.tstart) # steady-state region
        r_pk = self.Clamps.tstart + 0.4*(self.Clamps.tend-self.Clamps.tstart)
        self.ivss_analysis(region=[r_ss, self.Clamps.tend])
        self.ivpk_analysis(region=[self.Clamps.tstart, r_pk])  # peak region
        self.tau_h(self.tauh_voltage, peakRegion=[self.Clamps.tstart, r_pk], steadystateRegion=[r_ss, self.Clamps.tend], printWindow=False)

    def tau_membrane(self, peak_time=False, printWindow=False, whichTau=1, vrange=[-0.002, -0.050], region=[], tgap=0.):
        """
        Compute time constant (single exponential) from the onset of the response to a current step
        
        Parameters
        ----------
        
        peak_time : bool (default: False)
            Whether to fit only to the (negative) peak of the data. If it is True, the 
            fit will only go until the peak time. Otherwise it is set by the end of the region.
        
        printWindow : Boolean (default: False)
            Flag to allow printing of the fit results in detail during a run
        
        whichTau : int (default: 1)
            Not used
        
        vrange : list (V) (default: [-0.005, -0.020])
            Define the voltage range below RMP for the traces that will be fit to obtain tau_m.
            
        
        region: list (s) (default: [])
            Define the time region for the fitting
        
        tgap: float (sec)
            gap for the fitting to start (e.g., initial points to ignore)
        
        Return
        ------
            Nothing
            
        Class variables with a leading taum_ are set by this routine, to return results.
        
        """
        
        assert len(region) > 0.
        
        rgnpk = list(region)
        Func = 'exp1'  # single exponential fit with DC offset.
        if self.rmp == []:
            self.rmp_analysis(region=self.baseline)

        Fits = Fitting.Fitting() # get a fitting instance
        initpars = [self.rmp*1e-3, -0.010, 0.010]  # rmp is in units of mV
        icmdneg = np.where(self.Clamps.commandLevels < -10e-12)
        maxcmd = np.min(self.Clamps.commandLevels)
        ineg = np.where(self.Clamps.commandLevels[icmdneg] < 0.0)
        # if peak_time is not None and ineg != np.array([]):
        #     rgnpk[1] = np.max(peak_time[ineg[0]])
        dt = self.Clamps.sample_interval
        rgnpk = sorted(rgnpk)
        vrange = np.sort(vrange) # /1000.
        rgnindx = [int((rgnpk[1]-0.005)/dt), int((rgnpk[1])/dt)]
        rmps = self.ivbaseline
        vmeans = np.mean(self.Clamps.traces[:, rgnindx[0]:rgnindx[1]], axis=1) - self.ivbaseline
        # vrange is in mV
        indxs = np.where(np.logical_and((vmeans[ineg] >= vrange[0]), 
                         (vmeans[ineg] <= vrange[1])))
        # print('baseline: ', self.ivbaseline)
        # print('vrange: ', vrange)
        # print('vmeans: ', vmeans.view(np.ndarray))
        # print('indxs: ', indxs)
        # print('ineg: ', ineg)
        # print('self.Clamps.commandLevels', self.Clamps.commandLevels)

        # print 'ineg: ', ineg
        # print self.Clamps.commandLevels
        # print 'icmdneg: ', icmdneg
        # print 'vrange: ', vrange
        # print 'vmeans[ineg]: ', vmeans[ineg]
        # print 'indxs: ', indxs
        indxs = list(indxs[0])
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

        for j, k in enumerate(whichdata):
            if self.dataPlot is not None:
                n = 'w_%03d' % j
                self.taum_fitted[n] = self.dataPlot.plot(self.Clamps.time_base,
                                     self.Clamps.traces[k], pen=pg.mkPen('w'))
            taubounds = self.taum_bounds.copy()
            initpars[2] = np.mean(taubounds)
            if peak_time:
                vtr1 = self.Clamps.traces[k][int(rgnpk[0]/dt):int(rgnpk[1]/dt)]
                ipeak = np.argmin(vtr1)
                rgnpk[1] = ipeak*dt+rgnpk[0]
                vtr2 = self.Clamps.traces[k][int(rgnpk[0]/dt):int(rgnpk[1]/dt)]
                v0 = vtr2[0]
                v1 = vtr2[-1]-v0
                for m in range(len(vtr2)):
                    if vtr2[m]-v0 <= 0.63*v1:
                        break
                taubounds[0] = 0.0002
                taubounds[1] = 2.0*(rgnpk[1] - rgnpk[0])
                tau_init = m*dt
                if tau_init >= taubounds[0] and tau_init <= taubounds[1]:
                    initpars[2] = tau_init
                else:
                    initpars[2] = 0.5*ipeak*dt
                # print('inits: ', initpars)
            (fparx, xf, yf, namesx) = Fits.FitRegion([k], whichaxis,
                                               self.Clamps.time_base,
                                               np.array(self.Clamps.traces),
                                               dataType='2d',
                                               t0=rgnpk[0], t1=rgnpk[1],
                                               fitFunc=Func,
                                               fitPars=initpars,
                                               fixedPars=[tgap],
                                               method='SLSQP',
                                               bounds=[(-0.1, 0.0), (-0.05, 0.05), 
                                               (taubounds)],
                                               )
        
            if not fparx:
              raise Exception('IVCurve::update_Tau_membrane: Charging tau fitting failed - see log')
            #print 'j: ', j, len(fpar)
            # if fparx[0][1] < 2.5e-3:  # amplitude must be > 2.5 mV to be useful
            #     continue
            fpar.append(fparx[0])
            names.append(namesx[0])
            okdata.append(k)
            self.taum_fitted[k] = [xf[0], yf[0]]
            # import matplotlib.pyplot as mpl
            # mpl.plot(self.Clamps.time_base, np.array(self.Clamps.traces[k]), 'k-')
            # mpl.plot(xf[0], yf[0], 'r--', linewidth=1)
            # mpl.show()
            # exit(1)
        self.taum_pars = fpar
        self.taum_win = rgnpk
        self.taum_func = Func
        self.taum_whichdata = okdata
        taus = []
        for j in range(len(fpar)):
            outstr = ""
            taus.append(fpar[j][2])
            for i in range(0, len(names[j])):
                outstr += '%s = %f, ' % (names[j][i], fpar[j][i])
            if printWindow:
                print(("FIT(%d, %.1f pA): %s " %
                      (whichdata[j], itaucmd[j] * 1e12, outstr)))
        if len(taus) > 0:
            self.taum_taum = np.nanmean(taus)
            self.analysis_summary['taum'] = self.taum_taum
        else:
            self.taum_taum = np.NaN
            self.analysis_summary['taum'] = np.NaN
        self.analysis_summary['taupars'] = self.taum_pars
        self.analysis_summary['taufunc'] = self.taum_func
    
    def rmp_analysis(self, region=None):
        """
        Get the resting membrane potential
        
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
            raise ValueError("IVCurve, rmp_analysis requires a region beginning and end to measure the RMP")
        data1 = self.Clamps.traces['Time': region[0]:region[1]]
        data1 = data1.view(np.ndarray)
        self.ivbaseline = data1.mean(axis=1)  # all traces
        self.ivbaseline_cmd = self.Clamps.commandLevels
        self.rmp = np.mean(self.ivbaseline) * 1e3  # convert to mV
        data2 = self.Clamps.cmd_wave['Time': region[0]:region[1]]
        self.irmp = data2.view(np.ndarray).mean(axis=1)
        self.analysis_summary['RMP'] = self.rmp
        self.analysis_summary['RMPs'] = self.ivbaseline  # save raw baselines as well
        self.analysis_summary['Irmp'] = self.irmp
#        print('irmp: ', self.irmp, ' rmp: ', self.rmp)

    def ivss_analysis(self, region=None):
        """
        compute steady-state IV curve - from the mean voltage 
        across the stimulus set over the defined time region 
        (this usually will be the last half or third of the trace)
        
        Parameters
        ----------
        region : list or tuple
            Start and end times for the analysis
        """
        
        data1 = self.Clamps.traces['Time': region[0]:region[1]]
        self.r_in = np.nan
        self.analysis_summary['Rin'] = np.nan
        self.ivss_v = []
        if data1.shape[1] == 0 or data1.shape[0] == 1:
            return  # skip it

        # check out whether there are spikes in the window that is selected
        threshold = self.Spikes
        ntr = len(self.Clamps.traces)
        if not self.Spikes.spikes_counted:
            print("ivss_analysis: spikes not counted yet? - let's go analyze them...")
            self.analyzeSpikes()

        self.ivss_v = data1.mean(axis=1)  # all traces
        self.analysis_summary['Rin'] = np.NaN
        if len(self.Spikes.nospk) >= 1:
            # Steady-state IV where there are no spikes
            self.ivss_v = self.ivss_v[self.Spikes.nospk]
            self.ivss_cmd = self.Clamps.commandLevels[self.Spikes.nospk]
            isort = np.argsort(self.ivss_cmd)
            self.ivss_cmd = self.ivss_cmd[isort]
            self.ivss_v = self.ivss_v[isort]
            bl = self.ivbaseline[isort]
            self.ivss_bl = bl
            # compute Rin from the SS IV:
            # this makes the assumption that:
            # successive trials are in order so we wort above
            # commands are not repeated...
            if len(self.ivss_cmd) > 1 and len(self.ivss_v) > 1:
                pf = np.polyfit(self.ivss_cmd, self.ivss_v, 3, rcond=None, full=False, w=None, cov=False)
                pval = np.polyval(pf, self.ivss_cmd)
                #print('pval: ', pval)
                slope = np.diff(pval) / np.diff(self.ivss_cmd)  # local slopes
                imids = np.array((self.ivss_cmd[1:] + self.ivss_cmd[:-1]) / 2.)
                self.rss_fit ={'I': imids, 'V': np.polyval(pf, imids)}
                #print('fit V: ', self.rss_fit['V'])
                #slope = slope[[slope > 0 ] and [self.ivss_cmd[:-1] > -0.8] ] # only consider positive slope points
                l = int(len(slope)/2)
                if len(slope) > 1:
                    maxloc = np.argmax(slope[l:]) + l
                    minloc = np.argmin(slope[:l])
                else:
                    maxloc = 0
                    minloc = 0
                self.r_in = slope[maxloc]
                self.r_in_loc = [self.ivss_cmd[maxloc], self.ivss_v[maxloc], maxloc]  # where it was found
                self.r_in_min = slope[minloc]
                self.r_in_minloc = [self.ivss_cmd[minloc], self.ivss_v[minloc], minloc]  # where it was found
                self.analysis_summary['Rin'] = self.r_in*1.0e-6

    def ivpk_analysis(self, region=None):
        """
        compute peak IV curve - from the minimum voltage 
        across the stimulus set
        
        Parameters
        ----------
        region : list or tuple
            Start and end times for the analysis
        """
        
        self.r_in_peak = np.nan
        self.analysis_summary['Rin_peak'] = np.nan
        self.ivpk_v = []
        data1 = self.Clamps.traces['Time': region[0]:region[1]]
        if data1.shape[1] == 0 or data1.shape[0] == 1:
            return  # skip it

        # check out whether there are spikes in the window that is selected
        threshold = self.Spikes
        ntr = len(self.Clamps.traces)
        if not self.Spikes.spikes_counted:
            print("ivss_analysis: spikes not counted yet? - let's go analyze them...")
            self.analyzeSpikes()

        self.ivpk_v = data1.min(axis=1)  # all traces, minimum voltage found
        if len(self.Spikes.nospk) >= 1:
            # Steady-state IV where there are no spikes
            self.ivpk_v = self.ivpk_v[self.Spikes.nospk]
            self.ivpk_cmd = self.Clamps.commandLevels[self.Spikes.nospk]
            bl = self.ivbaseline[self.Spikes.nospk]
            isort = np.argsort(self.ivpk_cmd)
            self.ivpk_cmd = self.ivpk_cmd[isort]
            self.ivpk_v = self.ivpk_v[isort]
            bl = bl[isort]
            self.ivpk_bl = bl
            if len(self.ivpk_cmd) > 1 and len(self.ivpk_v) > 1:
                pf = np.polyfit(self.ivpk_cmd, self.ivpk_v, 3, rcond=None, full=False, w=None, cov=False)
                pval = np.polyval(pf, self.ivpk_cmd)
                slope = np.diff(pval) / np.diff(self.ivpk_cmd)
                imids = np.array((self.ivpk_cmd[1:] + self.ivpk_cmd[:-1]) / 2.)
                self.rpk_fit ={'I': imids, 'V': np.polyval(pf, imids)}
                l = int(len(slope)/2)
                if len(slope) > 1:
                    maxloc = np.argmax(slope[l:]) + l
                    minloc = np.argmin(slope[:l])
                else:
                    maxloc = 0
                    minloc = 0
                self.r_in_peak = slope[maxloc]
                self.r_in_peak_loc = [self.ivpk_cmd[maxloc], self.ivpk_v[maxloc], maxloc]  # where it was found
                self.r_in_minpeak = slope[minloc]
                self.r_in_minpeak_loc = [self.ivpk_cmd[minloc], self.ivpk_v[minloc], minloc]  # where it was found
                self.analysis_summary['Rin_peak'] = self.r_in_peak*1.0e-6

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
        
    
    def tau_h(self, v_steadystate, peakRegion, steadystateRegion, printWindow=False):
        """
        Measure the time constant associated with activation of the hyperpolarization-
        activated current, Ih. The tau is measured from the peak of the response to the
        steady-state part, at a single current level.
        
        Parameters
        ----------
        v_steadystate : float (voltage, V; no default).
             The steady-state voltage that will be used as the target for measuring Ih. A single
             voltage level is given; the closest one in the test set that is negative to the
             resting potential will be used.

        peakRegion : list of floats ([time, time], ms; no default)
            The time window over which the peak voltage will be identified.

        steadystateRegion : list of floats ([time, time], ms; no default)
            The time window over which the steady-state voltage will be identified.
        
        Return
        ------
        Nothing
            
        Class variables with a leading tauh_ are set by this routine, to return the
        results of the measurements.
        
        """
        # initialize result varibles
        self.tauh_vpk = None  # peak voltage for the tau h meausure
        self.tauh_neg_pk = None 
        self.tauh_vss = None  # ss voltage for trace used for tauh 
        self.tauh_neg_ss = None 
        self.tauh_vrmp = None
        self.tauh_xf = []
        self.tauh_yf = []
        self.tauh_fitted = {}
        self.tauh_meantau = None
        self.tauh_bovera = None
        self.tauh_Gh = None
        self.analysis_summary['tauh_tau'] = self.tauh_meantau
        self.analysis_summary['tauh_bovera'] = self.tauh_bovera
        self.analysis_summary['tauh_Gh'] = self.tauh_Gh
        self.analysis_summary['tauh_vss'] = self.tauh_vss                
        
        if self.rmp/1000. < v_steadystate:  # rmp is in mV... 
            return

        Func = 'exp1'  # single exponential fit to the whole region
        Fits = Fitting.Fitting()

        # for our time windows, get the ss voltage to use
        ss_voltages = self.Clamps.traces['Time': steadystateRegion[0]:steadystateRegion[1]].view(np.ndarray)
        ss_voltages = ss_voltages.mean(axis=1)
        try:
            itrace = np.argmin((ss_voltages[self.Spikes.nospk] - v_steadystate)**2)
        except:
            return
        pk_voltages = self.Clamps.traces['Time': peakRegion[0]:peakRegion[1]].view(np.ndarray)
        pk_voltages_tr = pk_voltages.min(axis=1)
        ipk_start = pk_voltages[itrace].argmin() + int(peakRegion[0]*self.Clamps.sample_rate[itrace]) # get starting index as well
        pk_time = self.Clamps.time_base[ipk_start] 

        if not self.Spikes.spikes_counted:
            self.analyzeSpikes()

        # now find trace with voltage closest to target steady-state voltage
        # from traces without spikes in the standard window
        whichdata = [int(itrace)]
        # prepare to fit
        initpars = [-80.0 * 1e-3, -10.0 * 1e-3, 50.0 * 1e-3]
        bounds = [(-0.1, 0.), (0., 0.1), ()]
        v_rmp = self.ivbaseline[itrace]
        itaucmd = self.Clamps.commandLevels[itrace]
        whichaxis = 0
        (fpar, xf, yf, names) = Fits.FitRegion(whichdata, whichaxis,
                                               self.Clamps.time_base,
                                               self.Clamps.traces.view(np.ndarray),
                                               dataType='2d',
                                               t0=pk_time, t1=steadystateRegion[1],
                                               fitFunc=Func,
                                               fitPars=initpars,
                                               method='SLSQP',
                                               bounds=[(-0.120, 0.05), (-0.1, 0.1), 
                                               (0.005, (steadystateRegion[1]-pk_time)*2.0)],
                                               )
        if not fpar:
            raise Exception('IVCurve::update_Tauh: tau_h fitting failed')
        s = np.shape(fpar)
        taus = []
        for j in range(0, s[0]):
            outstr = ""
            taus.append(fpar[j][2])
            for i in range(0, len(names[j])):
                outstr += '%s = %f, ' % (names[j][i], fpar[j][i])
            if printWindow:
                print(("Ih FIT(%d, %.1f pA): %s " %
                      (whichdata[j], itaucmd[j] * 1e12, outstr)))
        self.taum_fitted[itrace] = [xf[0], yf[0]]
        self.tauh_vrmp = self.ivbaseline[itrace]
        self.tauh_vss = ss_voltages[itrace]
        self.tauh_vpk = pk_voltages_tr[itrace]
        self.tauh_neg_ss = (self.tauh_vss - self.tauh_vrmp) / 1e3
        self.tauh_neg_pk = (self.tauh_vpk - self.tauh_vrmp) / 1e3
        self.tauh_xf = xf
        self.tauh_yf = yf
        self.tauh_meantau = np.mean(taus)
        self.tauh_bovera = (self.tauh_vss - self.tauh_vrmp) / (self.tauh_vpk - self.tauh_vrmp)
        if self.tauh_bovera > 1.0:
            self.tauh_bovera = 1.0
        Gpk = itaucmd / self.tauh_neg_pk
        Gss = itaucmd / self.tauh_neg_ss
        self.tauh_Gh = Gss - Gpk
        if self.tauh_Gh < 0:
            self.tauh_Gh = 0.
        
        self.analysis_summary['tauh_tau'] = self.tauh_meantau
        self.analysis_summary['tauh_bovera'] = self.tauh_bovera
        self.analysis_summary['tauh_Gh'] = self.tauh_Gh
        self.analysis_summary['tauh_vss'] = self.tauh_vss

        
