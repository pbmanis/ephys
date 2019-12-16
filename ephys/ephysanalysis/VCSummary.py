from __future__ import print_function

"""
Compute IV from voltage clamp data.
Version 0.1, does only min negative peak IV, max pos IV and ss IV

"""

import sys

import matplotlib
import numpy as np

import os.path
import ephysanalysis as EP
import ephysanalysis.metaarray as EM
import matplotlib.pyplot as mpl
import matplotlib.colors
import matplotlib
#import colormaps.parula
import pylibrary.plotting.plothelpers as PH
# from matplotlib import rc
# #rc('font',**{'family':'sans-serif','sans-serif':['Arial']})
# #rcParams['font.sans-serif'] = ['Arial']
# #rcParams['font.family'] = 'sans-serif'
# rc('text', usetex=True)
# rcParams = matplotlib.rcParams
# rcParams['svg.fonttype'] = 'none' # No text as paths. Assume font installed.
# rcParams['pdf.fonttype'] = 42
# rcParams['ps.fonttype'] = 42
# rcParams['text.latex.unicode'] = True
#import seaborn
#cm_sns = mpl.cm.get_cmap('terrain')  # terrain is not bad
#cm_sns = mpl.cm.get_cmap('parula')  # terrain is not bad
#cm_sns = mpl.cm.get_cmap('jet')  # jet is terrible
color_sequence = ['k', 'r', 'b']
colormap = 'snshelix'


class VCSummary():
    def __init__(self, datapath, plot=True):
        self.datapath = datapath
        self.AR = EP.acq4read.Acq4Read()  # make our own private cersion of the analysis and reader
        self.plot = plot

    def setup(self, clamps=None, baseline=[0, 0.001], taumbounds = [0.002, 0.050]):
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

    def compute_iv(self):
        """
        Simple plot voltage clamp traces
        """
        #print('path: ', self.datapath)
        self.AR.setProtocol(self.datapath)  # define the protocol path where the data is
        self.setup(clamps=self.AR)
        if self.AR.getData():  # get that data.
            self.analyze()
            self.plot_vciv()
            return True
        return False

    def analyze(self, rmpregion=[0., 0.05], tauregion=[0.1, 0.125]):
        #self.rmp_analysis(region=rmpregion)
#        self.tau_membrane(region=tauregion)
        r0 = self.Clamps.tstart + 0.9*(self.Clamps.tend-self.Clamps.tstart) # 
        self.ihold_analysis(region=[0., self.Clamps.tstart])
        self.vcss_analysis(region=[r0, self.Clamps.tend])
        #self.ivpk_analysis(region=[self.Clamps.tstart, self.Clamps.tstart+0.4*(self.Clamps.tend-self.Clamps.tstart)])

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
            raise ValueError("VCSummary, ihold_analysis requires a region beginning and end to measure the RMP")
        data1 = self.Clamps.traces['Time': region[0]:region[1]]
        data1 = data1.view(np.ndarray)
        self.vcbaseline = data1.mean(axis=1)  # all traces
        self.vcbaseline_cmd = self.Clamps.commandLevels
        self.iHold = np.mean(self.vcbaseline) * 1e9  # convert to nA
        self.analysis_summary['iHold'] = self.iHold

    def vcss_analysis(self, region=None):
        """
        compute steady-state IV curve - from the mean current 
        across the stimulus set over the defined time region 
        (this usually will be the last half or third of the trace)
        
        Parameters
        ----------
        region : list or tuple
            Start and end times for the analysis
        """
        
        data1 = self.Clamps.traces['Time': region[0]:region[1]]
        icmds = EM.MetaArray(self.Clamps.cmd_wave,  # easiest = turn in to a matching metaarray... 
            info=[{'name': 'Command', 'units': 'A',
             'values': np.array(self.Clamps.clampValues)},
             self.Clamps.traces.infoCopy('Time'), self.Clamps.traces.infoCopy(-1)])
        self.vcss_vcmd = icmds['Time': region[0]:region[1]].mean(axis=1)
        self.r_in = np.nan
        self.analysis_summary['Rin'] = np.nan
        self.vcss_v = []
        if data1.shape[1] == 0 or data1.shape[0] == 1:
            return  # skip it

        ntr = len(self.Clamps.traces)

        self.vcss_Im = data1.mean(axis=1)  # steady-state, all traces
        self.analysis_summary['Rin'] = np.NaN
#        self.Clamps.plotClampData()
        
        isort = np.argsort(self.vcss_vcmd)
        self.vcss_Im= self.vcss_Im[isort]
        self.vcss_vcmd = self.vcss_vcmd[isort]
        bl = self.vcbaseline[isort]
        self.vcss_bl = bl
        # compute Rin from the SS IV:
        # this makes the assumption that:
        # successive trials are in order so we sort above
        # commands are not repeated...
        if len(self.vcss_vcmd) > 1 and len(self.vcss_v) > 1:
            pf = np.polyfit(self.vcss_vcmd, self.vcss_v, 3, rcond=None, full=False, w=None, cov=False)
            pval = np.polyval(pf, self.vcss_vcmd)
            #print('pval: ', pval)
            slope = np.diff(pval) / np.diff(self.vcss_vcmd)  # local slopes
            imids = np.array((self.vcss_vcmd[1:] + self.vcss_vcmd[:-1]) / 2.)
            self.rss_fit ={'I': imids, 'V': np.polyval(pf, imids)}
            #print('fit V: ', self.rss_fit['V'])
            #slope = slope[[slope > 0 ] and [self.vcss_vcmd[:-1] > -0.8] ] # only consider positive slope points
            l = int(len(slope)/2)
            maxloc = np.argmax(slope[l:]) + l
            self.r_in = slope[maxloc]
            self.r_in_loc = [self.vcss_vcmd[maxloc], self.vcss_v[maxloc], maxloc]  # where it was found
            minloc = np.argmin(slope[:l])
            self.r_in_min = slope[minloc]
            self.r_in_minloc = [self.vcss_vcmd[minloc], self.vcss_v[minloc], minloc]  # where it was found
            self.analysis_summary['Rin'] = self.r_in*1.0e-6

    # def ivpk_analysis(self, region=None):
    #     """
    #     compute peak IV curve - from the minimum voltage
    #     across the stimulus set
    #
    #     Parameters
    #     ----------
    #     region : list or tuple
    #         Start and end times for the analysis
    #     """
    #
    #     self.r_in_peak = np.nan
    #     self.analysis_summary['Rin_peak'] = np.nan
    #     self.ivpk_v = []
    #     data1 = self.Clamps.traces['Time': region[0]:region[1]]
    #     if data1.shape[1] == 0 or data1.shape[0] == 1:
    #         return  # skip it
    #
    #     # check out whether there are spikes in the window that is selected
    #     threshold = self.Spikes
    #     ntr = len(self.Clamps.traces)
    #     if not self.Spikes.spikes_counted:
    #         print("ivss_analysis: spikes not counted yet? - let's go analyze them...")
    #         self.analyzeSpikes()
    #
    #     self.ivpk_v = data1.min(axis=1)  # all traces, minimum voltage found
    #     if len(self.Spikes.nospk) >= 1:
    #         # Steady-state IV where there are no spikes
    #         self.ivpk_v = self.ivpk_v[self.Spikes.nospk]
    #         self.ivpk_cmd = self.Clamps.commandLevels[self.Spikes.nospk]
    #         bl = self.vcbaseline[self.Spikes.nospk]
    #         isort = np.argsort(self.ivpk_cmd)
    #         self.ivpk_cmd = self.ivpk_cmd[isort]
    #         self.ivpk_v = self.ivpk_v[isort]
    #         bl = bl[isort]
    #         self.ivpk_bl = bl
    #         if len(self.ivpk_cmd) > 1 and len(self.ivpk_v) > 1:
    #             pf = np.polyfit(self.ivpk_cmd, self.ivpk_v, 3, rcond=None, full=False, w=None, cov=False)
    #             pval = np.polyval(pf, self.ivpk_cmd)
    #             slope = np.diff(pval) / np.diff(self.ivpk_cmd)
    #             imids = np.array((self.ivpk_cmd[1:] + self.ivpk_cmd[:-1]) / 2.)
    #             self.rpk_fit ={'I': imids, 'V': np.polyval(pf, imids)}
    #             l = int(len(slope)/2)
    #             maxloc = np.argmax(slope[l:]) + l
    #             self.r_in_peak = slope[maxloc]
    #             self.r_in_peak_loc = [self.ivpk_cmd[maxloc], self.ivpk_v[maxloc], maxloc]  # where it was found
    #             minloc = np.argmin(slope[:l])
    #             self.r_in_minpeak = slope[minloc]
    #             self.r_in_minpeak_loc = [self.ivpk_cmd[minloc], self.ivpk_v[minloc], minloc]  # where it was found
    #
    #             self.analysis_summary['Rin_peak'] = self.r_in_peak*1.0e-6

    def plot_vciv(self):
        P = PH.regular_grid(2 , 2, order='columns', figsize=(8., 6.), showgrid=False,
                        verticalspacing=0.1, horizontalspacing=0.12,
                        margins={'leftmargin': 0.12, 'rightmargin': 0.12, 'topmargin': 0.08, 'bottommargin': 0.1},
                        labelposition=(-0.12, 0.95))
        (date, sliceid, cell, proto, p3) = self.file_cell_protocol(self.datapath)
        
        P.figure_handle.suptitle(os.path.join(date, sliceid, cell, proto).replace('_', '\_'), fontsize=12)
        for i in range(self.AR.traces.shape[0]):
            P.axdict['A'].plot(self.AR.time_base*1e3, self.AR.traces[i,:]*1e12, 'k-', linewidth=0.5)

        PH.talbotTicks(P.axdict['A'], tickPlacesAdd={'x': 0, 'y': 0}, floatAdd={'x': 0, 'y': 0})
        P.axdict['A'].set_xlabel('T (ms)')
        P.axdict['A'].set_ylabel('I (pA)')
        PH.crossAxes(P.axdict['C'], xyzero=(-60., 0.))
        P.axdict['C'].plot(self.vcss_vcmd*1e3, self.vcss_Im*1e12, 'ks-', linewidth=1, markersize=4)
        P.axdict['C'].set_xlabel('V (mV)')
        P.axdict['C'].set_ylabel('I (pA)')
        PH.talbotTicks(P.axdict['C'], tickPlacesAdd={'x': 0, 'y': 0}, floatAdd={'x': 0, 'y': 0})

        P.axdict['B'].set_xlabel('I (nA)')
        P.axdict['B'].set_ylabel('V (mV)')
        PH.talbotTicks(P.axdict['B'], tickPlacesAdd={'x': 1, 'y': 0}, floatAdd={'x': 2, 'y': 0})

        P.axdict['D'].set_xlabel('I (pA)')
        P.axdict['D'].set_ylabel('Latency (ms)')

        self.IVFigure = P.figure_handle
    
        if self.plot:
             mpl.show()

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
        (p0, proto) = os.path.split(filename)
        (p1, cell) = os.path.split(p0)
        (p2, sliceid) = os.path.split(p1)
        (p3, date) = os.path.split(p2)
        return (date, sliceid, cell, proto, p3)
        
     
