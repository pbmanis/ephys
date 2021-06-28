from __future__ import print_function

"""
Just plot the stack of voltage clamp traces; no analysis
Version 0.1

"""

import sys
from typing import Union
import matplotlib
import numpy as np

import os.path
from . import acq4read
from . import metaarray as EM
from ..tools import digital_filters as DF
import matplotlib.pyplot as mpl
import matplotlib.colors
import matplotlib
import matplotlib.patches as mpatches
from matplotlib.collections import PatchCollection
import pylibrary.plotting.plothelpers as PH
import pylibrary.tools.utility as U

color_sequence = ['k', 'r', 'b']
colormap = 'snshelix'


class VCTraceplot():
    def __init__(self, datapath, plot=True):
        self.datapath = datapath
        self.AR = acq4read.Acq4Read()  # make our own private cersion of the analysis and reader
        self.plot = plot
        self.reset()

    def reset(self):
        self.pre_process_filters = {"LPF": None, "Notch": []}
        self.pre_process_flag = False
        self.stim_times = None
        self.stim_dur = None

    def set_datapath(self, datapath):
        self.datapath = datapath

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

    def set_pre_process(self, LPF:Union[None, float]=None, Notch:Union[None, list]=None):
        self.pre_process_filters['LPF'] = LPF
        self.pre_process_filters['Notch'] = Notch

    def pre_process(self):

        if self.pre_process_flag:
            raise ValueError("VCTraceplot: Already applited pre-processing (filtering) to this data set")
        if self.pre_process_filters['LPF'] is not None:
            self.AR.data_array = DF.SignalFilter_LPFBessel(
                    self.AR.data_array,
                    LPF=self.pre_process_filters['LPF'],
                    samplefreq=self.AR.sample_rate[0],
                    NPole = 8,
                )
            self.pre_process_flag = True
        if self.pre_process_filters['Notch'] is not None:

            self.AR.data_array = DF.NotchFilter(
                 self.AR.data_array,
                 notchf=self.pre_process_filters['Notch'],
                 Q=15, QScale=True, 
                 samplefreq = self.AR.sample_rate[0],)
            self.pre_process_flag = True

    def set_stim_mark(self, stim:Union[None, list]=None, dur:Union[None, list, float]= None):
        self.stim_times = stim
        self.stim_dur = dur
        
        
    def plot_vc(self):
        """
        Simple plot voltage clamp traces
        """
        #print('path: ', self.datapath)
        self.AR.setProtocol(self.datapath)  # define the protocol path where the data is
        self.AR.set_pre_process(self.pre_process_filters)
        self.setup(clamps=self.AR)
        if self.AR.getData():  # get that data.
            self.pre_process()
            self.analyze()
            fh = self.plot_vc_stack()
            return fh
        return False

    def analyze(self, rmpregion=[0., 0.05], tauregion=[0.1, 0.125]):
        #self.rmp_analysis(region=rmpregion)
#        self.tau_membrane(region=tauregion)
        r0 = self.Clamps.tstart + 0.9*(self.Clamps.tend-self.Clamps.tstart) # 
        self.ihold_analysis(region=[0., self.Clamps.tstart])
        self.LaserBlue = self.AR.getLaserBlueCommand()

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
            raise ValueError("VCTraceplot, ihold_analysis requires a region beginning and end to measure the RMP")
        data1 = self.Clamps.traces['Time': region[0]:region[1]]
        data1 = data1.view(np.ndarray)
        self.vcbaseline = data1.mean(axis=1)  # all traces
        self.vcbaseline_cmd = self.Clamps.commandLevels
        self.iHold = np.mean(self.vcbaseline) * 1e9  # convert to nA
        self.analysis_summary['iHold'] = self.iHold


    def plot_vc_stack(self):
        lx = -0.05
        ly = 1.05
        sizer = {'A': {'pos': [0.08, 0.80, 0.28, 0.77], 'labelpos': (lx,ly), 'noaxes': False},
                 'B': {'pos': [0.08, 0.80, 0.18, 0.05], 'labelpos': (lx,ly), 'noaxes': False},
                 'C': {'pos': [0.08, 0.80, 0.05, 0.05], 'labelpos': (lx,ly)},
                 # 'D': {'pos': [0.62, 0.30, 0.08, 0.22], 'labelpos': (x,y)},
                }

        # dict pos elements are [left, width, bottom, height] for the axes in the plot.
        gr = [(a, a+1, 0, 1) for a in range(0, 4)]   # just generate subplots - shape does not matter
        axmap = dict(zip(sizer.keys(), gr))
        P = PH.Plotter((len(sizer), 1), axmap=axmap, label=True, figsize=(8., 10.))
        # PH.show_figure_grid(P.figure_handle)
        P.resize(sizer)  # perform positioning magic

        # P = PH.arbitrary_grid(sizer = {"A": {"pos": [0.08, 0.8, 0.20, 0.7],
        #                                     "labelpos": (lx,ly), "noaxes": False},
        #                                "B": {"pos": [0.08, 0.8, 0.50, 0.1],
        #                                      "labelpos": (lx,ly)},},
        #          order='columnsfirst', figsize=(8., 6.), showgrid=False,
        #                 # verticalspacing=0.1, horizontalspacing=0.12,
        #                 margins={'leftmargin': 0.12, 'rightmargin': 0.12, 'topmargin': 0.08, 'bottommargin': 0.1},
        #                 labelposition=(-0.12, 0.95))
        (date, sliceid, cell, proto, p3) = self.file_cell_protocol(self.datapath)
        sf1 = 1e12  # for currents, top plot, voltage clamp
        sf2 = 1e3 # for voltages, bottom plot, voltage clamp
        trstep = 50. # pA
        ylabel1 = "I (pA)"
        ylabel2 = "V (mV)"
        if self.AR.mode in ["IC", "I=0"]:
            sf1 = 1e3  # for voltage, top plot, current clamp
            sf2 = 1e9 # nA bottom plot currents injected
            trstep = 10. # mV
            ylabel1 = "V (mV)"
            ylabel2 = "I (nA)"
        P.figure_handle.suptitle(os.path.join(date, sliceid, cell, proto).replace('_', '\_'), fontsize=12)
        for i in range(self.AR.traces.shape[0]):
            P.axdict['A'].plot(self.AR.time_base*1e3, self.AR.data_array[i,:]*sf1 + i*trstep, 'k-', linewidth=0.5)
            if self.LaserBlue:
                P.axdict['B'].plot(self.AR.LBR_time_base[0,:]*1e3, self.AR.LaserBlue_pCell[i,:], 'b-', linewidth=0.5)
            P.axdict['C'].plot(self.AR.time_base*1e3, self.AR.cmd_wave[i,:]*sf2, 'k-', linewidth=0.5)
        if self.AR.traces.shape[0] > 1:
            iavg = np.mean(self.AR.data_array, axis=0)
            P.axdict['A'].plot(self.AR.time_base*1e3, iavg*sf1 + -2*trstep, 'm-', linewidth=0.5)
        if self.stim_times is not None and self.stim_dur is not None:
            stim_patch_collection = []
            ylims = P.axdict['A'].get_ylim()
            for stime in self.stim_times:
                p = mpatches.Rectangle(
                    [stime, ylims[0]], # anchor
                    width=self.stim_dur,
                    height=ylims[1]-ylims[0],
                    ec=None, color='skyblue')
                stim_patch_collection.append(p)
            collection = PatchCollection(stim_patch_collection, alpha=0.3)
            #collection.set_array(colors)
            P.axdict['A'].add_collection(collection)
 
        PH.talbotTicks(P.axdict['A'], tickPlacesAdd={'x': 0, 'y': 0}, floatAdd={'x': 0, 'y': 0})
        P.axdict['A'].set_xlabel('T (ms)')
        P.axdict['A'].set_ylabel(ylabel1)
        P.axdict['B'].set_xlabel('T (ms)')
        P.axdict['B'].set_ylabel('V (V)')  # command to laser or LED is in V

        P.axdict['C'].set_xlabel('T (ms)')
        P.axdict['C'].set_ylabel(ylabel2)
        PH.talbotTicks(P.axdict['B'], tickPlacesAdd={'x': 0, 'y': 0}, floatAdd={'x': 0, 'y': 0})

        # P.axdict['B'].set_xlabel('I (nA)')
        # P.axdict['B'].set_ylabel('V (mV)')
        # PH.talbotTicks(P.axdict['B'], tickPlacesAdd={'x': 1, 'y': 0}, floatAdd={'x': 2, 'y': 0})
        #
        # P.axdict['D'].set_xlabel('I (pA)')
        # P.axdict['D'].set_ylabel('Latency (ms)')

        self.VCTracesFigure = P.figure_handle
    
        if self.plot:
             mpl.show()
        return P.figure_handle

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
        
     
