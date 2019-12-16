#!/usr/bin/env python3

"""
Brige balance tool
Version 0.1

Graphical interface
Part of Ephysanalysis package

Usage:
notch
"""

import os
import sys
import argparse
from pathlib import Path
import pathlib
import numpy as np
import ephysanalysis as EP
import pylibrary.fileselector as FS
import pyqtgraph as pg
from pyqtgraph.parametertree import Parameter, ParameterTree
import mapanalysistools.digital_filters as FILT
from minis import minis_methods

ampdataname = 'Clamp1.ma'
# or MultiClamp1.ma... etc
# datadir = '/Volumes/Pegasus/ManisLab_Data3/Kasten_Michael/NF107Ai32Het'
# dbfile = 'NF107Ai32Het_bcorr2.pkl'

class TraceAnalyzer(pg.QtGui.QWidget):
    def __init__(self, app=None):
        super(TraceAnalyzer, self).__init__()
        self.app = app
        self.datadir = '/Volumes/Pegasus/ManisLab_Data3/Kasten_Michael/NF107Ai32Het'
        self.AR = EP.acq4read.Acq4Read()  # make our own private cersion of the analysis and reader
        self.SP = EP.SpikeAnalysis.SpikeAnalysis()
        self.RM = EP.RmTauAnalysis.RmTauAnalysis()
        self.LPF = 5000.
        self.HPF = 0.
        self.tb = None
        self.notch_freqs = [60., 120., 180., 240.]
        self.notch_Q = 30.
        self.curves = []
        self.crits = []
        self.scatter = []
        self.maxT = 0.6
        self.tau1 = 0.1
        self.tau2 = 0.4
        self.thresh = 3.0
        self.sign = -1
        self.scalar = 1
        self.n_adjusted = 0
        self.curve_set = False
        self.last_method = 'cb'
        self.MA = minis_methods.MiniAnalyses()  # get a minianalysis instance
        
    
    def getProtocolDir(self):
        sel = FS.FileSelector(dialogtype='dir', startingdir=self.datadir)
        print(sel.fileName)
        self.clampfiles = []
        self.AR.setDataName(ampdataname)
        if sel.fileName is not None:
            self.pdirs = Path(sel.fileName).glob(f'**/{ampdataname:s}')
            for p in self.pdirs:
                self.clampfiles.append(p)
                # print(p)
        self.w1.slider.setValue(0)
        print('# clamp files: ', len(self.clampfiles))
        self.w1.slider.setRange(0, len(self.clampfiles))
        self.w1.slider.setTickInterval(10)
        # self.w1.slider.setMaximum(len(self.clampfiles))
        # setMinimum(0)
        # self.w1.slider.setMaximum(len(self.clampfiles))
        self.protocolPath = sel.fileName
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
            self.update_traces()

    def _getpars(self):
        signdict = {'-': -1, '+':1}
        self.tau1 = self.minis_risetau.value()*1e-3
        self.tau2 = self.minis_falltau.value()*1e-3
        self.thresh = self.minis_thresh.value()
        sign = self.minis_sign.currentText()
        self.sign = signdict[sign]
        # print(self.tau1, self.tau2, self.thresh, self.sign)
        
    def CB(self):

        self._getpars()
        cb = minis_methods.ClementsBekkers()
        rate = np.mean(np.diff(self.tb))
        jmax = int((2*self.tau1 + 3*self.tau2)/rate)
        cb.setup(tau1=self.tau1, tau2=self.tau2, dt=rate, delay=0.0, template_tmax=rate*(jmax-1),
                sign=self.sign, eventstartthr=None)
        meandata = np.mean(self.current_data)
        cb._make_template()
        cb.cbTemplateMatch(self.current_data,  threshold=self.thresh)
        self.decorate(cb)
        self.method = cb
        self.last_method = 'cb'
        
    def AJ(self):
        self._getpars()
        aj = minis_methods.AndradeJonas()
        rate = np.mean(np.diff(self.tb))
        jmax = int((2*self.tau1 + 3*self.tau2)/rate)
        aj.setup(tau1=self.tau1, tau2=self.tau2, dt=rate, delay=0.0, template_tmax=rate*(jmax-1),
                sign=self.sign, eventstartthr=None)
        meandata = np.mean(self.current_data)
        aj.deconvolve(self.current_data, data_nostim=None,
                thresh=self.thresh, llambda=1., order=7)  # note threshold scaling...
        self.decorate(aj)
        self.method = aj
        self.last_method = 'aj'
    
    def decorate(self, minimethod):
        if not self.curve_set:
            return
        # print('decorating', )
        for s in self.scatter:
            s.clear()
        for c in self.crits:
            c.clear()
        self.scatter = []
        self.crits = []
        if len(minimethod.onsets) is not None:
            self.scatter.append(self.dataplot.plot(self.tb[minimethod.peaks]*1e3,  self.current_data[minimethod.peaks],
                      pen = None, symbol='o', symbolPen=None, symbolSize=5, symbolBrush=(255, 0, 0, 255)))
            # self.scatter.append(self.dataplot.plot(self.tb[minimethod.peaks]*1e3,  np.array(minimethod.amplitudes),
            #           pen = None, symbol='o', symbolPen=None, symbolSize=5, symbolBrush=(255, 0, 0, 255)))

            self.crits.append(self.dataplot2.plot(self.tb[:len(minimethod.Crit)]*1e3, minimethod.Crit, pen='r'))
            # print(' ... decorated')
    
    def update_traces(self):
        trmap = {'cb': self.CB, 'aj': self.AJ}
        if len(self.AR.traces) == 0:
            return
        self.current_trace = int(self.w1.x)
        self.dataplot.setTitle(f'Trace: {self.current_trace:d}')
        for c in self.curves:
            c.clear()
        for s in self.scatter:
            s.clear()
        self.scatter = []
        self.curves = []
        self.curve_set = False
        notchfr = self.notch_button.value()
        i = self.current_trace
        if (i >= self.AR.data_array.shape[0]):
            self.dataplot.setTitle(f'Trace > Max traces: {self.AR.data_array.shape[0]:d}')
            return
        imax = int(self.maxT*self.AR.sample_rate[0])
        mod_data = self.AR.data_array[i,:imax]
        if self.notch_check.checkState():
            mod_data =  FILT.NotchFilterZP(mod_data, notchf=[notchfr], Q=self.notch_Q,
                            QScale=False, samplefreq=self.AR.sample_rate[0])

        if self.lpf_check.checkState():
            mod_data = FILT.SignalFilter_LPFBessel(mod_data, self.lpf_button.value(),
                        samplefreq=self.AR.sample_rate[0], NPole=8)

        self.curves.append(self.dataplot.plot(self.AR.time_base[:imax]*1e3,
                            # self.AR.traces[i,:],
                            mod_data, 
                           pen=pg.intColor(1)))
        self.current_data = mod_data
        self.tb = self.AR.time_base[:imax]
        # print(self.tb.shape, imax)
        self.curve_set = True
        trmap[self.last_method]()

    def quit(self):
        exit(0)

    def keyPressEvent(self, event):
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

    def set_window(self, parent=None):
        super(TraceAnalyzer, self).__init__(parent=parent)
        self.win = pg.GraphicsWindow(title="TraceAnalyzer")
        layout = pg.QtGui.QGridLayout()
        layout.setSpacing(10)
        self.win.setLayout(layout)
        self.win.resize(1280, 800)
        
        self.buttons = pg.QtGui.QGridLayout()
        self.b3 = pg.QtGui.QPushButton("Get Protocol Directory")
        self.buttons.addWidget(self.b3)
        self.b3.clicked.connect(self.getProtocolDir)
        
        self.notch_button = pg.QtGui.QDoubleSpinBox()
        self.notch_button.setValue(60.)
        self.notch_button.setMinimum(10.)
        self.notch_button.setMaximum(5000.)
        self.buttons.addWidget(self.notch_button)
        self.notch_button.valueChanged.connect(self.update_traces)
        self.notch_check = pg.QtGui.QCheckBox("Notch")
        self.notch_check.stateChanged.connect(self.update_traces)
        self.buttons.addWidget(self.notch_check)
        
        
        self.lpf_button = pg.QtGui.QDoubleSpinBox()
        self.lpf_button.setMinimum(200.)
        self.lpf_button.setMaximum(10000.)
        self.lpf_button.setValue(3000.)
        self.buttons.addWidget(self.lpf_button)
        self.lpf_button.valueChanged.connect(self.update_traces)
        self.buttons.addWidget(self.lpf_button)
        self.lpf_check = pg.QtGui.QCheckBox("LPF")
        self.lpf_check.setChecked(True)
        self.lpf_check.stateChanged.connect(self.update_traces)
        self.buttons.addWidget(self.lpf_check)
        
        self.cb_button = pg.QtGui.QPushButton("Clements-Bekkers")
        self.buttons.addWidget(self.cb_button)
        self.cb_button.clicked.connect(self.CB)

        self.aj_button = pg.QtGui.QPushButton("AJ")
        self.buttons.addWidget(self.aj_button)
        self.aj_button.clicked.connect(self.AJ)
        
        self.minis_risetau = pg.QtGui.QDoubleSpinBox()
        self.minis_risetau.setRange(0.1, 10.)
        self.minis_risetau.setValue(0.2)
        self.minis_risetau.setDecimals(2)
        self.minis_risetau.setSingleStep(0.1)
        self.minis_risetau.setSuffix(' ms')
        self.buttons.addWidget(self.minis_risetau)
        self.minis_risetau.valueChanged.connect(self.update_traces)
        
        self.minis_falltau = pg.QtGui.QDoubleSpinBox()
        self.minis_falltau.setRange(0.2, 20.)
        self.minis_falltau.setSingleStep(0.1)
        self.minis_falltau.setSuffix(' ms')
        self.minis_falltau.setValue(1.0)
        self.minis_falltau.setDecimals(2)
        self.buttons.addWidget(self.minis_falltau)
        self.minis_falltau.valueChanged.connect(self.update_traces)
        
        self.minis_thresh = pg.QtGui.QDoubleSpinBox()
        self.minis_thresh.setRange(1.0, 12.)
        self.minis_thresh.setValue(3.5)
        self.minis_thresh.setDecimals(2)
        self.minis_thresh.setSingleStep(0.1)
        self.minis_thresh.setSuffix(' n SD')
        self.buttons.addWidget(self.minis_thresh)
        self.minis_thresh.valueChanged.connect(self.update_traces)
        
        self.minis_sign = pg.QtGui.QComboBox()
        self.minis_sign.addItems(['-', '+'])
        self.buttons.addWidget(self.minis_sign)
        self.minis_sign.currentIndexChanged.connect(self.update_traces)
        
        
        spacerItem1 = pg.QtGui.QSpacerItem(0, 100, pg.QtGui.QSizePolicy.Expanding, pg.QtGui.QSizePolicy.Minimum)
        self.buttons.addItem(spacerItem1)
        
        self.b2 = pg.QtGui.QPushButton("Quit")
        self.buttons.addWidget(self.b2)
        self.b2.clicked.connect(self.quit)

        self.w1 = Slider(0, 500, scalar=1., parent=parent)
        self.w1.setGeometry(0, 0, 500, 30)
        self.w1.slider.setSingleStep(1)
        self.w1.slider.setPageStep(1)
        self.w1.slider.valueChanged.connect(self.update_traces)

        # spacerItem = pg.QtGui.QSpacerItem(0, 10, pg.QtGui.QSizePolicy.Expanding, pg.QtGui.QSizePolicy.Minimum)
        # self.buttons.addItem(spacerItem)

        self.dataplot = pg.PlotWidget()
        self.dataplot2 = pg.PlotWidget()
        layout.addLayout(self.buttons,   0, 0, 7, 1)
        layout.addWidget(self.dataplot,  0, 1, 1, 6)
        layout.addWidget(self.dataplot2, 6, 1, 4, 6)
        layout.addWidget(self.w1,        11, 1, 1, 6)
        layout.setColumnStretch(0, 1)  # reduce width of LHS column of buttons
        layout.setColumnStretch(1, 7)  # and stretch out the data dispaly
        

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
        self.slider.setMaximum(self.maximum)
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
    # TA.show()
    import sys
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        pg.QtGui.QApplication.instance().exec_()
        
    # if sys.flags.interactive == 0:
    #  app.exec_()


if __name__ == '__main__':
    main()

