#!/usr/bin/env python3

"""



Graphical interface
Part of Ephysanalysis package

Usage:
notch
"""
import importlib
import os
import sys
import argparse
from pathlib import Path
import pathlib
import pickle
import numpy as np
from ..ephysanalysis import acq4read
from ..ephysanalysis import SpikeAnalysis
from ..ephysanalysis import RmTauAnalysis
import pylibrary.tools.fileselector as FS
import pylibrary.tools.cprint as CP
import pyqtgraph as pg
from pyqtgraph.parametertree import Parameter, ParameterTree
import ephys.mapanalysistools.digital_filters as FILT
from ..mini_analyses import minis_methods
from ..mini_analyses import minis_methods_common
os.environ['QT_MAC_WANTS_LAYER'] = '1'

all_modules = [
    SpikeAnalysis,
    acq4read,
    RmTauAnalysis,
    FILT,
    minis_methods,
    minis_methods_common
]


# or MultiClamp1.ma... etc
# datadir = '/Volumes/Pegasus/ManisLab_Data3/Kasten_Michael/NF107Ai32Het'
# dbfile = 'NF107Ai32Het_bcorr2.pkl'
# all_modules = [
#     table_manager,
#     plot_sims,
#     vcnmodel.correlation_calcs,
#     vcnmodel.spikestatistics,
#     vcnmodel.analysis,
# ]

class TraceAnalyzer(pg.QtGui.QWidget):
    keyPressed = pg.QtCore.pyqtSignal(pg.QtCore.QEvent)
    def __init__(self, app=None):
        super(TraceAnalyzer, self).__init__()
        self.app = app
        self.datadir = '/Volumes/Pegasus/ManisLab_Data3/Kasten_Michael/NF107Ai32Het'
        self.AR = acq4read.Acq4Read()  # make our own private cersion of the analysis and reader
        self.SP = SpikeAnalysis.SpikeAnalysis()
        self.RM = RmTauAnalysis.RmTauAnalysis()
        self.ampdataname = 'MultiClamp1.ma'
        self.LPF = 5000.
        self.HPF = 0.
        self.tb = None
        self.notch_60HzHarmonics = [60., 120., 180., 240.]
        self.notch_frequency = "None"
        self.notch_Q = 30.
        self.curves = []
        self.crits = []
        self.scatter = []
        self.threshold_line = None
        self.lines = []
        self.tstart = 0.
        self.tend = 0.
        self.maxT = 0.6
        self.tau1 = 0.1
        self.tau2 = 0.4
        self.method = None
        self.Order = 7
        self.minis_risetau = self.tau1
        self.minis_falltau= self.tau2
        self.thresh = 3.0
        self.minis_threshold = self.thresh
        self.ZC_mindur = 1e-3
        self.sign = -1
        self.minis_sign = '-'
        self.scalar = 1
        self.n_adjusted = 0
        self.curve_set = False
        self.last_method = 'CB'
        self.compare_flag = False
        self.compare_data = None
        self.data_set = None

        self.MA = minis_methods.MiniAnalyses()  # get a minianalysis instance
        
    
    def getProtocolDir(self):
        sel = FS.FileSelector(dialogtype='dir', startingdir=self.datadir)
        print(sel.fileName)
        self.clampfiles = []
        self.AR.setDataName(self.ampdataname)
        if sel.fileName is not None:
            self.pdirs = Path(sel.fileName).glob(f'**/{self.ampdataname:s}')
            for p in self.pdirs:
                self.clampfiles.append(p)
                # print(p)
        wtparts = Path(sel.fileName).parts
        wt = '/'.join(wtparts[-4:])
        self.fileName = sel.fileName
        self.win.setWindowTitle(wt)
        self.w1.slider.setValue(0)
        print('# clamp files: ', len(self.clampfiles))
        self.w1.slider.setRange(0, len(self.clampfiles))
        self.w1.slider.setTickInterval(10)
        # self.w1.slider.setMaximum(len(self.clampfiles))
        # setMinimum(0)
        # self.w1.slider.setMaximum(len(self.clampfiles))
        self.protocolPath = sel.fileName
        self.compare_data = False # new prototocol; trigger new comparision if needed
        # print('protocolpath: ', sel.fileName)
        self.updateTraces()

    def setProtocol(self, date, sliceno, cellno, protocolName):
        # create an IV protocol path:
        self.newbr = 0.
        self.protocolBridge = 0.
        self.date = date
        self.slice = sliceno
        self.cell = cellno
        if not '_' in date:
            self.date = date+'_000'
        if isinstance(sliceno, int):
            self.slice = 'slice_{0:03d}'.format(sliceno)
        if isinstance(cellno, int):
            self.cell = 'cell_{0:03d}'.format(cellno)
        self.protocolName = protocolName
        self.protocolPath = Path(self.datadir, self.date, self.slice, self.cell, self.protocolName)
        self.protocolKey = Path(self.date, self.slice, self.cell, self.protocolName)
        if not self.protocolPath.is_dir():
            print('dir not found: ', str(self.protocolPath))
            return

    def updateTraces(self):
        self.AR.setProtocol(self.protocolPath)  # define the protocol path where the data is
        if self.AR.getData():  # get that data.
            # trim time window if needed
            dt = 1./self.AR.sample_rate[0]
            # trx = self.AR.data_array
            print(self.AR.data_array.shape)
            if self.tend == 0:
                tend = self.AR.data_array.shape[1]*dt
            else:
                tend = self.tend
            print(dt, self.tstart, tend)
            istart = int(self.tstart/dt)
            iend = int(tend/dt)
            print(istart, iend)
            self.AR.data_array = self.AR.data_array[:, istart:iend]
            print(self.AR.data_array.shape)
            self.update_traces()

    def _getpars(self):
        signdict = {'-': -1, '+':1}
        self.tau1 = 1e-3*self.minis_risetau  # .value()*1e-3
        self.tau2 = 1e-3*self.minis_falltau  #.value()*1e-3
        self.thresh = self.minis_threshold  #.value()
        sign = self.minis_sign
        self.sign = signdict[sign]
        # print(self.tau1, self.tau2, self.thresh, self.sign)
        
    def CB(self):
        self._getpars()
        cb = minis_methods.ClementsBekkers()
        rate = np.mean(np.diff(self.tb))
        jmax = int((2*self.tau1 + 3*self.tau2)/rate)
        cb.setup(
            ntraces = self.AR.data_array.shape[0],
            tau1=self.tau1, tau2=self.tau2, dt_seconds=rate, delay=0.0, template_tmax=rate*(jmax-1),
            threshold=self.thresh, sign=self.sign, eventstartthr=None,
            lpf=self.LPF,
            hpf=self.HPF,
            )
        self.imax = int(self.maxT*self.AR.sample_rate[0])

        meandata = np.mean(self.AR.data_array[:,:self.imax])
        template = cb._make_template()
        for i in range(self.AR.data_array.shape[0]):
            cb.cbTemplateMatch(self.AR.data_array[i,:self.imax], itrace=i, lpf=self.LPF)
            self.AR.data_array[i,:self.imax] = cb.data  # # get filtered data
            cb.reset_filtering()
        self.method = cb
        self.last_method = 'CB'
        self.CB_update()
    
    def CB_update(self):
        print('cbupdate')
        self.method.threshold = self.minis_threshold
        self.method.identify_events(order=self.Order)
        self.method.summarize(self.AR.data_array[:,:self.imax])
        self.decorate(self.method)

        
    def AJ(self):
        self._getpars()
        aj = minis_methods.AndradeJonas()
        rate = np.mean(np.diff(self.tb))
        jmax = int((2*self.tau1 + 3*self.tau2)/rate)
        print('template len: ', jmax, 'template max t: ', rate*(jmax-1), rate)
        aj.setup(
            ntraces = self.AR.data_array.shape[0],
            tau1=self.tau1, tau2=self.tau2, dt_seconds=rate,
            delay=0.0, template_tmax=np.max(self.tb),
            threshold=self.thresh, sign=self.sign, eventstartthr=None,
            lpf=self.LPF,
            hpf=self.HPF,
        )
        self.imax = int(self.maxT*self.AR.sample_rate[0])
        meandata = np.mean(self.AR.data_array[:,:self.imax])
        # self.AJorder = int(1e-3/rate)
        print('AJ.: Order, rate, taus: ', self.Order, rate, self.tau1, self.tau2)
        for i in range(self.AR.data_array.shape[0]):
            aj.deconvolve(self.AR.data_array[i,:self.imax]-meandata, 
                itrace = i,
                # data_nostim=None,
                llambda=5.)   # assumes times are all in same units of msec
            self.AR.data_array[i,:self.imax] = aj.data  # # get filtered data
            aj.reset_filtering()

        self.method = aj
        self.last_method = 'AJ'
        self.AJ_update()
    
    def AJ_update(self):
        print('ajupdate')
        self.method.threshold = self.minis_threshold
        self.method.identify_events(order=self.Order)
        self.method.summarize(self.AR.data_array[:,:self.imax])
        tot_events = sum([len(x) for x in self.method.onsets])
        self.decorate(self.method)

    def RS(self):
        self._getpars()
        rs = minis_methods.RSDeconvolve()

        rs.setup(
            ntraces = self.AR.data_array.shape[0],
            tau1=self.tau1,
            tau2=self.tau2,
            dt_seconds=rate,
            delay=0.0,
            template_tmax = np.max(self.tb),  # taus are for template
            sign=self.sign,
            risepower=4.0,
            threshold=self.thresh,
            lpf = self.LPF,
            hpf = self.HPF,
        )
        # generate test data
        self.imax = int(self.maxT*self.AR.sample_rate[0])
        meandata = np.mean(self.AR.data_array[:,:self.imax])
        for i in range(self.AR.data_array.shape[0]):
            rs.deconvolve(
                self.AR.data_array[i,:self.imax],
                itrace = i,
        )
        rs.identify_events(order=self.Order)
        rs.summarize(self.AR.data_array[:,:self.imax])
        self.decorate(rs)
        self.method = rs
        self.last_method = 'RS'
        self.RS_update()
    
    def RS_update(self):
        print('rsupdate')
        self.method.threshold = self.minis_threshold
        self.method.identify_events(order=self.Order)
        self.method.summarize(self.AR.data_array[:,:self.imax])
        self.decorate(self.method)
    

    def ZC(self):
        self._getpars()
        zc = minis_methods.ZCFinder()
        rate = np.mean(np.diff(self.tb))
        jmax = int((2*self.tau1 + 3*self.tau2)/rate)
        mindur = self.ZC_mindur  # msec
        minlen = int(mindur / rate)
        zc.setup(
            tau1=self.tau1,
            tau2=self.tau2,
            dt_seconds=rate,
            delay=0.0,
            template_tmax= np.max(self.tb),
            sign=self.sign,
            threshold=self.thresh,
            lpf = self.LPF,
            hpf = self.HPF,
        )
        events = zc.find_events(
            self.current_data, data_nostim=None, minLength=minlen,
        )

        self.method = zc
        self.last_method = 'ZC'
    
    def ZC_update(self):
        print('zsupdate')
        self.method.threshold = self.minis_threshold
        self.decorate(self.method)

    def decorate(self, minimethod):
        if not self.curve_set:
            return
        # print('decorating', )
        for s in self.scatter:
            s.clear()
        for c in self.crits:
            c.clear()
        # for line in self.threshold_line:
        #     line.clear()
        self.scatter = []
        self.crits = []

        if minimethod.Summary.onsets is not None and len(minimethod.Summary.onsets) > 0:
            self.scatter.append(self.dataplot.plot(self.tb[minimethod.Summary.peaks[self.current_trace]]*1e3, 
                         self.current_data[minimethod.Summary.peaks[self.current_trace]],
                      pen = None, symbol='o', symbolPen=None, symbolSize=10, symbolBrush=(255, 0, 0, 255)))
            # self.scatter.append(self.dataplot.plot(self.tb[minimethod.peaks]*1e3,  np.array(minimethod.amplitudes),
            #           pen = None, symbol='o', symbolPen=None, symbolSize=5, symbolBrush=(255, 0, 0, 255)))

        self.crits.append(self.dataplot2.plot(self.tb[:len(minimethod.Criterion[self.current_trace])]*1e3, 
                minimethod.Criterion[self.current_trace], pen='r'))
        self.threshold_line.setValue(minimethod.sdthr)
        # print('sdthr: ', minimethod.sdthr, self.minis_threshold)
            # print(' ... decorated')
    
    def update_threshold(self):
        trmap1 = {'CB': self.CB_update, 'AJ': self.AJ_update, 'ZC': self.ZC_update, "RS": self.RS_update}
        trmap1[self.last_method]() # threshold/scroll, just update
        
      
    def update_traces(self, value=None, update_analysis=False):
        if isinstance(value, int):
            self.current_trace = value
        print('update_traces, analysis update = ', update_analysis)
        trmap1 = {'CB': self.CB_update, 'AJ': self.AJ_update, 'ZC': self.ZC_update, "RS": self.RS_update}
        trmap2 = {'CB': self.CB, 'AJ': self.AJ, 'ZC': self.ZC, "RS": self.RS}
        if len(self.AR.traces) == 0:
            return
        self.current_trace = int(self.w1.x)
        self.dataplot.setTitle(f'Trace: {self.current_trace:d}')
        for c in self.curves:
            c.clear()
        for s in self.scatter:
            s.clear()
        for l in self.lines:
            self.dataplot.removeItem(l)
        
        self.scatter = []
        self.curves = []
        self.lines = []
        self.curve_set = False
        if (self.current_trace >= self.AR.data_array.shape[0]):
            self.dataplot.setTitle(f'Trace > Max traces: {self.AR.data_array.shape[0]:d}')
            return
        # imax = int(self.maxT*self.AR.sample_rate[0])
        imax = len(self.AR.data_array[self.current_trace])
        self.imax = imax
        self.maxT = self.AR.sample_rate[0]*imax
        self.mod_data = self.AR.data_array[self.current_trace,:imax]
        if self.notch_frequency is not "None":
            if self.notch_frequency == "60HzHarm":
                notchfreqs = self.notch_60HzHarmonics
            else:
                notchfreqs = [self.notch_frequency]
            CP.cprint('y', f"Notch Filtering at: {str(notchfreqs):s}")
            self.mod_data =  FILT.NotchFilterZP(self.mod_data, notchf=notchfreqs, Q=self.notch_Q,
                            QScale=False, samplefreq=self.AR.sample_rate[0])

        if self.LPF is not "None":
            CP.cprint('y', f"LPF Filtering at: {self.LPF:.2f}")
            self.mod_data = FILT.SignalFilter_LPFBessel(self.mod_data, self.LPF,
                        samplefreq=self.AR.sample_rate[0], NPole=8)

        self.curves.append(self.dataplot.plot(self.AR.time_base[:imax]*1e3,
                            # self.AR.traces[i,:],
                           self.mod_data, 
                           pen=pg.intColor(1)))
        self.current_data = self.mod_data
        self.tb = self.AR.time_base[:imax]
        # print(self.tb.shape, imax)
        self.curve_set = True
        if update_analysis:
            trmap2[self.last_method]()  # recompute from scratch
        elif self.method is not None:
            self.decorate(self.method)
        # if self.method is not None:
        #     self.decorate(self.method)
        # self.compareEvents()

    def quit(self):
        exit(0)

    def keyPressEvent(self, event):
        super(TraceAnalyzer, self).keyPressEvent(event)
        print('key pressed, event=', event)
        self.keyPressed.emit(event)
    
    def on_key(self, event):
        print('Got event key: ', event.key())
        if event.key() == Qt.Key_Right:
            self.w1.slider.setValue(self.slider.value() + 1)
        elif event.key() == Qt.Key_Left:
            self.w1.slider.setValue(self.slider.value() - 1)
        else:
            QWidget.keyPressEvent(self, event) # just pass it on

    def getProtocols(self):
        thisdata = self.df.index[(self.df['date'] == self.date) &
                                (self.df['slice_slice'] == self.slice) &
                                (self.df['cell_cell'] == self.cell)].tolist()
        if len(thisdata) > 1:
            raise ValueError('Search for data resulted in more than one entry!')
        ivprots = self.df.iloc[thisdata]['IV'].values[0]  # all the protocols in the dict
        return thisdata, ivprots
        
    def getProtocol(self, protocolName):
        thisdata, ivprots = self.getIVProtocols()
        if protocolName not in ivprots.keys():
            return None
        else:
            return ivprots[protocolName]

    def compareEvents(self):
        """
        Try to compare traces from what is shown with the events file on the disk
        """
        if not self.compare_flag:
            if len(self.lines) > 0:
                for l in self.lines:
                    self.dataplot.removeItem(l)
            self.compare_flag = False
            self.compare_data = None
            self.data_set = None
            return

        if self.compare_data is None:
            pathparts = Path(self.fileName).parts
            evfolder = Path("datasets", "NF107Ai32_Het", "events")
            evfile = pathparts[-4] + '~' +pathparts[-3] + '~' + pathparts[-2] + '.pkl'
            evfile = Path(evfolder, evfile)
            wtparts = Path(self.fileName).parts
            self.data_set = Path('/'.join(wtparts[-4:]))
            proto = wtparts[-1]
            if not evfile.is_file():
                print('Evfile: ', evfile, ' is not a file?')
                self.compare_data = False  # no comparison file; just return
                self.compare_d = None
                return
            with open(evfile, 'rb') as fh:
                self.compare_data = pickle.load(fh)
        if self.compare_data is not None:
            rate = self.compare_data[self.data_set]['rate']
            ev = self.compare_data[self.data_set]['events']  # list of trials, spots
            tr = self.current_trace  # which spot/trace?
            trd = ev[0][tr]  # get the trace data
            for line in np.array(trd['peaktimes'][0])*1e3*rate:
                self.lines.append(pg.InfiniteLine(line, pen='m'))
                self.dataplot.addItem(self.lines[-1])
            self.scatter.append(self.dataplot.plot(np.array(trd['peaktimes'][0])*1e3*rate,  trd['smpks'][0],
                      pen = None, symbol='t', symbolPen=None, symbolSize=12, symbolBrush=(0, 255, 0, 255)))
                  
    def show_data_pars(self):
        if self.compare_flag and self.compare_data is not None:
            CP.cprint('g', "Parameters for current dataset: ")
            if "analysis_parameters" in list(self.compare_data[self.data_set].keys()):
                print(self.compare_data[self.data_set]['analysis_parameters'])
            else:
                print('analysis pars not in dateset: ', self.compare_data[self.data_set].keys())

    def build_ptree(self):
        self.params = [
            # {"name": "Pick Cell", "type": "list", "values": cellvalues, "value": cellvalues[0]},
            {"name": "Get Protocol", "type": "action"},
            {"name": "Set Start", "type": "float", 'value': 0., 'limits': (0, 30.), 'default': 0.},
            {"name": "Set End", "type": "float", 'value': 0., 'limits': (0, 30.), 'default': 0.},
            
            {"name": "Notch Frequency", "type": "list",
                "values": ["None", "60HzHarm", 30., 60., 120., 180., 240.], "value": None},
            {"name": "LPF", "type": "list",
                "values": ["None", 500., 1000., 1200., 1500., 1800., 2000., 2500., 3000., 4000., 5000.],
                "value": 5000., "renamable": True},
            {"name": "Apply Filters", "type": "action"},
            {"name": "Method", "type": "list",
                "values": ["CB", "AJ", "RS", "ZC"], "value": "CB"},
            {"name": "Sign", "type": "list",
                "values": ['+', '-'], "value": "-"},
            {"name": "Rise Tau", "type": "float",
                'value': 0.15, 'step': 0.05, 'limits': (0.05, 10.0), 'default': 0.15},
            {"name": "Fall Tau", "type": "float",
                'value': 1.0, 'step': 0.1, 'limits': (0.15, 10.0), 'default': 1.0},
            {"name": "Threshold", "type": "float",
                'value': 3.0, 'step': 0.1, 'limits': (-1e-6, 50.0), 'default': 2.5},
            {"name": "Order", "type": "float",
                'value': 7, 'step': 1, 'limits': (1, 100), 'default': 7},
            {"name": "Apply Analysis", "type": "action"},

            {'name': "Channel Name", "type": "list",
                "values": ['Clamp1.ma', 'MultiClamp1.ma', 'Clamp2.ma', 'MultiClamp2.ma'],
                "value": 'MultiClamp1.ma'},

            {"name": "Compare Events", 'type': 'bool', 'value': False, 'tip': "Try to compare with events previously analyzed"},
            {"name": "Show Data Pars", 'type': "action"},
            {"name": "Reload", "type": "action"},
            {"name": "Quit", "type": "action"},
        ]
        self.ptree = ParameterTree()
        self.ptreedata = Parameter.create(
            name="Commands", type="group", children=self.params
        )
        self.ptree.setParameters(self.ptreedata)
        self.ptree.setMaximumWidth(300)
        self.ptree.setMinimumWidth(250)

    def command_dispatcher(self, param, changes):
        """
        Dispatcher for the commands from parametertree
        path[0] will be the command name
        path[1] will be the parameter (if there is one)
        path[2] will have the subcommand, if there is one
        data will be the field data (if there is any)
        """
        for param, change, data in changes:
            path = self.ptreedata.childPath(param)

            if path[0] == "Quit":
                self.quit()
            elif path[0] == "Get Protocol":
                self.getProtocolDir()
            elif path[0] == "Show Data Pars":
                self.show_data_pars()
            elif path[0] == "Reload":
                self.reload()
            elif path[0] == "Compare Events":
                self.compare_flag = data
                print('compare flg: ', self.compare_flag)
                self.compareEvents()
            elif path[0] == "Set Start":
                self.tstart = data
            elif path[0] == "Set End":
                if data > self.tstart:
                    self.tend = data
                else:
                    pass

            elif path[0] == "LPF":
               self.LPF = data
               # self.update_traces()
            
            elif path[0] == "Notch Frequency":
                self.notch_frequency = data
                # self.update_traces()

            elif path[0] == "Apply Filters":
                self.update_traces(update_analysis=False)

            elif path[0] == "Rise Tau":
                self.minis_risetau = data
                # self.update_traces(update_analysis=True)

            elif path[0] == "Fall Tau":
                self.minis_falltau = data
                # self.update_traces(update_analysis=True)
            elif path[0] == "Order":
                self.Order = data
                
            elif path[0] == "sign":
                self.minis_sign = data
                # self.update_traces(update_analysis=True)

            elif path[0] == "Threshold":
                self.minis_threshold = data
                self.update_threshold()

            elif path[0] == "Method":
                if data == "CB":
                    self.CB()
                elif data == "AJ":
                    self.AJ()
                elif data == "ZC":
                    self.ZC()
                elif data == "RS":
                    self.RS()
                else:
                    print('Not implemented: ', data)
                        
            elif path[0] == "Apply Analysis":
                self.update_traces(update_analysis=True)

            elif path[0] == "Channel Name":
                self.ampdataname = data
            elif path[0] == "Reload":
                self.reload()
            else:
                print(f"**** Command {path[0]:s} was not parsed in command dispatcher")


    def reload(self):
        print("reloading...")
        for module in all_modules:
            print("reloading: ", module)
            importlib.reload(module)
        
    def set_window(self, parent=None):
        super(TraceAnalyzer, self).__init__(parent=parent)
        self.win = pg.GraphicsWindow(title="TraceAnalyzer")
        layout = pg.QtGui.QGridLayout()
        layout.setSpacing(8)
        self.win.setLayout(layout)
        self.win.resize(1280, 800)
        self.win.setWindowTitle('No File')
        self.buttons = pg.QtGui.QGridLayout()
        self.build_ptree()
        self.buttons.addWidget(self.ptree)
        self.ptreedata.sigTreeStateChanged.connect(self.command_dispatcher)


        self.w1 = Slider(0, 500, scalar=1., parent=parent)
        self.w1.setGeometry(0, 0, 500, 30)
        self.w1.slider.setSingleStep(1)
        self.w1.slider.setPageStep(1)
        self.w1.slider.valueChanged.connect(self.update_traces)

        # spacerItem = pg.QtGui.QSpacerItem(0, 10, pg.QtGui.QSizePolicy.Expanding, pg.QtGui.QSizePolicy.Minimum)
        # self.buttons.addItem(spacerItem)

        self.dataplot = pg.PlotWidget()
        self.dataplot2 = pg.PlotWidget()
        self.dataplot2.setXLink(self.dataplot)
        self.threshold_line = pg.InfiniteLine(self.minis_threshold,
                angle=0., pen='c', hoverPen='y',
                bounds = [0., 20.], movable=True,
                label=f"{self.minis_threshold:.3e}")
        self.dataplot2.addItem(self.threshold_line)
        self.threshold_line.sigDragged.connect(self.update_threshold)
        
        layout.addLayout(self.buttons,   0, 0, 10, 1)
        layout.addWidget(self.dataplot,  0, 1, 1, 6)
        layout.addWidget(self.dataplot2, 6, 1, 4, 6)
        layout.addWidget(self.w1,        11, 1, 1, 6)
        layout.setColumnStretch(0, 1)  # reduce width of LHS column of buttons
        layout.setColumnStretch(1, 7)  # and stretch out the data dispaly
        
        self.keyPressed.connect(self.on_key)


class FloatSlider(pg.QtGui.QSlider):
    def __init__(self, parent, decimals=3, *args, **kargs):
        super(FloatSlider, self).__init__(parent, *args, **kargs)
        self._multi = 10 ** decimals
        print('multi: ', self._multi)
        self.setMinimum(self.minimum())
        self.setMaximum(self.maximum())
 
    def value(self):
        return float(super(FloatSlider, self).value()) / self._multi
 
    def setMinimum(self, value):
        self.min_val = value
        return super(FloatSlider, self).setMinimum(value * self._multi)
 
    def setMaximum(self, value):
        self.max_val = value
        return super(FloatSlider, self).setMaximum(value * self._multi)
 
    def setValue(self, value):
        super(FloatSlider, self).setValue(int((value-self.min_val) * self._multi))



class Slider(pg.QtGui.QWidget):
    def __init__(self, minimum, maximum, scalar=1.0, parent=None):
        super(Slider, self).__init__(parent=parent)
        self.verticalLayout = pg.QtGui.QVBoxLayout(self)
        self.label = pg.QtGui.QLabel(self)
        self.verticalLayout.addWidget(self.label, alignment=pg.QtCore.Qt.AlignHCenter)
        self.horizontalLayout = pg.QtGui.QHBoxLayout()
        spacerItem = pg.QtGui.QSpacerItem(0, 20, pg.QtGui.QSizePolicy.Expanding, pg.QtGui.QSizePolicy.Minimum)
        self.horizontalLayout.addItem(spacerItem)
        self.slider = FloatSlider(self, decimals=0)
        self.slider.setOrientation(pg.QtCore.Qt.Horizontal)
        self.horizontalLayout.addWidget(self.slider)
        spacerItem1 = pg.QtGui.QSpacerItem(0, 20, pg.QtGui.QSizePolicy.Expanding, pg.QtGui.QSizePolicy.Minimum)
        self.horizontalLayout.addItem(spacerItem1)
        self.verticalLayout.addLayout(self.horizontalLayout)
        self.resize(self.sizeHint())

        self.minimum = minimum*scalar
        self.maximum = maximum*scalar
        self.scalar = scalar
        self.slider.setMinimum(self.minimum)
        self.slider.setMaximum(self.maximum-1)
        #self.slider.setRange(self.minimum, self.maximum)
        self.slider.valueChanged.connect(self.setLabelValue)
        self.setLabelValue(self.slider.value())

    def setLabelValue(self, value):
        self.x = int(value) # int((self.minimum + (float(value) / (self.slider.maximum() - self.slider.minimum())) * (
                      # self.maximum - self.minimum)) /self.scalar)
        # print(self.minimum, self.slider.minimum(), self.maximum, self.slider.maximum(), self.scalar, value, self.x)
        self.label.setText(f"{self.x:4d}")
    
    def getPosValue(self, x):
        return int((x-self.minimum)*(self.slider.maximum() - self.slider.minimum())/(self.maximum - self.minimum))


def main():


    app = pg.QtGui.QApplication([])
    TA = TraceAnalyzer(app)
    app.aboutToQuit.connect(TA.quit)  # prevent python exception when closing window with system control
    TA.set_window()
    import sys
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        pg.QtGui.QApplication.instance().exec_()


if __name__ == '__main__':
    main()

