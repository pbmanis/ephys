#!/usr/bin/env python3
from __future__ import print_function

"""
Brige balance tool
Version 0.1

Graphical interface
Part of Ephysanalysis package

Usage:
bridge datadir dbfile
"""

import os
import sys
import argparse
from pathlib import Path
import pathlib
import numpy as np
import ephysanalysis as EP
import pyqtgraph as pg
from pyqtgraph.parametertree import Parameter, ParameterTree
import pandas as pd

# datadir = '/Volumes/Pegasus/ManisLab_Data3/Kasten_Michael/NF107Ai32Het'
# dbfile = 'NF107Ai32Het_bcorr2.pkl'

class Bridge(pg.QtGui.QWidget):
    def __init__(self, args):
        self.datadir = Path(args.datadir) 
        self.dbFilename = Path(args.dbfile)
        self.day = args.day
        self.df = pd.read_pickle(str(self.dbFilename))
        if 'IV' not in self.df.columns.values:
            print('Brige Tool requires that IVs have been created and run against the database')
            exit(1) 
        self.AR = EP.acq4read.Acq4Read()  # make our own private cersion of the analysis and reader
        self.SP = EP.SpikeAnalysis.SpikeAnalysis()
        self.RM = EP.RmTauAnalysis.RmTauAnalysis()
        self.curves = []
        self.n_adjusted = 0

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
            exit(1)

    def zoom(self):
        """
        Zoom to ~10 mV +/- and 5 msec before, 10 msec after first step
        """
        if self.b6.isChecked():
            t0 = self.AR.tstart*1e3
            ts = t0 - 5
            te = t0 + 10
            vm = self.RM.analysis_summary['RMP']
            vm0 = vm-10
            vm1 = vm + 10
        else:
            ts = 0
            te = np.max(self.AR.time_base)*1e3
            vm0 = -100
            vm1 = 20
        self.dataplot.setXRange(ts, te)
        self.dataplot.setYRange(vm0, vm1)

    def zero_bridge(self):
        self.newbr = 0.
        self.w1.slider.setValue(self.w1.getPosValue(self.newbr))
        self.update_data()

    def skip(self):
        """
        Advance to the next entry
        """
        self.currentiv += 1
        self.next()

    def save(self):
        # first, save new value into the database
        self.update_database()
        self.n_adjusted += 1
        self.currentiv += 1
        self.next()

    def next(self):
        # now get the next protocol
        if self.currentiv not in self.validivs:
            if self.currentiv < len(self.validivs):
                self.currentiv += 1
            if self.currentiv >= len(self.validivs):
                print('Exhausted valid IV protocols')
                return
        k = self.validivs[self.currentiv]
        p = k.parts
        self.setProtocol(p[0], p[1], p[2], p[3])
        thisdata, ivprots = self.getIVProtocols()
        try:
            self.protocolBridge = ivprots[self.protocolKey]['BridgeAdjust']
            print('Next Bridge is %f' % self.protocolBridge)
        except:
            self.protocolBridge = 0.
        self.analyzeIVProtocol()

    def runAllValidIVs(self):
        self.validivs = []
        self.currentiv = 0
        for i in self.df.index:  # run through all entries in the db
            if self.day != 'all':  # if all, just do it; otherwise, select
                date = self.df.at[i, 'date']
                day = str(self.day)
                if '_' not in day:
                    day = day + '_000'
                if day != date:
                    continue
            ivdata = self.df.at[i, 'IV'] # get the IV
            if len(ivdata) == 0:
                print('no ivdata for: ', i, self.df.at[i, 'date'])
                continue

            for k in list(ivdata.keys()):  # multiple IV's by keys in the dict
                if 'BridgeAdjust' not in ivdata[k] or ('BridgeAdjust' in ivdata[k] and ivdata[k]['BridgeAdjust'] == 0.):
                    print('         Do Bridge Adjustment for: ', k)
                    self.validivs.append(k)  # add iv key to the valid iv dict
                else:
                    print('BridgeAdjust exists for: {0:s}  {1:f}'.format(str(k), ivdata[k]['BridgeAdjust']))
                 #   self.validivs.append(k)
        self.next()  # go get the first valid protocol and analyze it
        
    def analyzeIVProtocol(self):
#        print('opening: ', self.protocolPath)
        self.AR.setProtocol(self.protocolPath)  # define the protocol path where the data is
        threshold = -0.020
        if self.AR.getData():  # get that data.
            self.RM.setup(clamps=self.AR, spikes=self.SP, bridge_offset=0.)   # doing setup here also does bridge correction
            self.SP.setup(clamps=self.AR, threshold=threshold, 
                    refractory=0.0001, peakwidth=0.001, interpolate=False, verify=False, mode='peak')
            self.SP.analyzeSpikes()
#            self.SP.analyzeSpikeShape()
#            self.SP.analyzeSpikes_brief(mode='baseline')
#            self.SP.analyzeSpikes_brief(mode='poststimulus')
            self.RM.analyze(rmpregion=[0., self.AR.tstart-0.001],
                            tauregion=[self.AR.tstart,
                                       self.AR.tstart + (self.AR.tend-self.AR.tstart)/5.])
            self.cmd = 1e9*self.AR.cmd_wave.view(np.ndarray)
            self.update_traces()

    def quit(self):
        exit(0)

    def getIVProtocols(self):
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

    def check_for_bridge(self, protocolName):
        prots = self.getProtocol(protcolName)
        if 'BridgeAdjust' not in ivprots[protocolName].keys():
            return False
        else:
            return True

    def update_database(self):
        thisdata, ivprots = self.getIVProtocols()
        print('update database with br: ', self.newbr)
        ivprots[self.protocolKey]['BridgeAdjust'] = self.newbr*1e6  # convert to ohms here
        self.df.at[thisdata[0], 'IV'] = ivprots  # update with the new bridge value
        print('Updated IV: ', ivprots[self.protocolKey])
        self.skipflag = False

    def save_database(self):
        # before updating, save previous version
        # self.dbFilename.rename(self.dbFilename.with_suffix('.bak'))
        print('Saving database so far ({0:d} entries with Bridge Adjustment)'.format(self.n_adjusted-1))
        self.df.to_pickle(str(self.dbFilename))  # now update the database
        self.n_adjusted = 0
        
    def set_window(self, parent=None):
        super(Bridge, self).__init__(parent=parent)
        self.horizontalLayout = pg.QtGui.QHBoxLayout(self)
        self.w1 = Slider(-20., 40., scalar=1.)
        
        self.buttons = pg.QtGui.QVBoxLayout(self)

        self.b3 = pg.QtGui.QPushButton("Do All Valid IVs")
        self.buttons.addWidget(self.b3, stretch=2)
        self.b3.clicked.connect(self.runAllValidIVs)

        self.b1 = pg.QtGui.QPushButton("Save Database")
        self.buttons.addWidget(self.b1, stretch=2)
        self.b1.clicked.connect(self.save_database)


        self.b5 = pg.QtGui.QPushButton("Save and load Next")
        self.buttons.addWidget(self.b5, stretch=2)
        self.b5.clicked.connect(self.save)

        self.b4 = pg.QtGui.QPushButton("Skip")
        self.buttons.addWidget(self.b4, stretch=2)
        self.b4.clicked.connect(self.skip)

        self.bzero = pg.QtGui.QPushButton("Zero Bridge")
        self.buttons.addWidget(self.bzero, stretch=2)
        self.bzero.clicked.connect(self.zero_bridge)

        self.b6 = pg.QtGui.QPushButton("Zoom/unzoom")
        self.b6.setCheckable(True)
        self.b6.setChecked(False)
        self.buttons.addWidget(self.b6, stretch=10)
        self.b6.clicked.connect(self.zoom)

        spacerItem1 = pg.QtGui.QSpacerItem(0, 400, pg.QtGui.QSizePolicy.Expanding, pg.QtGui.QSizePolicy.Minimum)
        self.buttons.addItem(spacerItem1)
        
        self.b2 = pg.QtGui.QPushButton("Quit")
        self.buttons.addWidget(self.b2, stretch=10)
        self.b2.clicked.connect(self.quit)

        spacerItem = pg.QtGui.QSpacerItem(0, 10, pg.QtGui.QSizePolicy.Expanding, pg.QtGui.QSizePolicy.Minimum)
        self.buttons.addItem(spacerItem)
        
        self.sliderpane = pg.QtGui.QVBoxLayout(self)
        self.sliderpane.addWidget(self.w1)

        self.win = pg.GraphicsWindow(title="Bridge Correction Tool")
       # self.buttons.addWidget(self.w1, stretch=-100)
        self.horizontalLayout.addLayout(self.buttons)
        self.horizontalLayout.addLayout(self.sliderpane)
        self.horizontalLayout.addWidget(self.win)
        self.dataplot = self.win.addPlot(title="Data")
        self.w1.slider.valueChanged.connect(self.update_data)
        
    def update_traces(self):
        print('Update traces, br: {0:f}'.format(self.protocolBridge))
        self.dataplot.setTitle('{0:s} {1:.2f}'.format(str(self.protocolKey), self.protocolBridge))
        self.newbr = self.protocolBridge/1e6  # convert to megaohms
        self.w1.slider.setValue(self.newbr)
        cmdindxs = np.unique(self.AR.commandLevels)  # find the unique voltages
        colindxs = [int(np.where(cmdindxs == self.AR.commandLevels[i])[0]) for i in range(len(self.AR.commandLevels))]  #
        for c in self.curves:
            c.clear()
        self.curves = []
        for i, d in enumerate(self.AR.traces):
            self.curves.append(self.dataplot.plot(self.AR.time_base*1e3,
                               self.AR.traces[i,:]*1e3-(self.cmd[i]*self.newbr),
                               pen=pg.intColor(colindxs[i], len(cmdindxs), maxValue=255)))

    def update_data(self):
        a = self.w1.x
        self.newbr = self.w1.x
        for i, c in enumerate(self.curves):
            c.setData(self.AR.time_base*1e3,
                      self.AR.traces[i,:]*1e3-(self.cmd[i]*self.newbr))


class FloatSlider(pg.QtGui.QSlider):
    def __init__(self, parent, decimals=3, *args, **kargs):
        super(FloatSlider, self).__init__(parent, *args, **kargs)
        self._multi = 10 ** decimals
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
        self.slider = FloatSlider(self, decimals=2)
        self.slider.setOrientation(pg.QtCore.Qt.Vertical)
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
        self.x = (self.minimum + (float(value) / (self.slider.maximum() - self.slider.minimum())) * (
        self.maximum - self.minimum)) /self.scalar
        self.label.setText("{0:6.2f}".format(self.x))
    
    def getPosValue(self, x):
        return (x-self.minimum)*(self.slider.maximum() - self.slider.minimum())/(self.maximum - self.minimum)

def main():
    parser = argparse.ArgumentParser(description="""Bridge balance correction tool.
            Allows the user to adjust the bridge balance on IV curves from cells post-hoc.
            Bridge values are read from an existing database (generated by dataSummary),
            and the modified (delta) values of bridge resistance are written back to 
            the dataSummary database for future use.
            --10/2018 pbm
            """)
    parser.add_argument('datadir', type=str, default='',
                        help='Full path to data directory')
    parser.add_argument('dbfile', type=str, default='',
                        help='Name of database file (including path)')
    parser.add_argument('-d', '--day', type=str, default='all', 
                     help='Day for analysis if only analyzing data from one day')
                 
    args = parser.parse_args()

    BR = Bridge(args)

    app = pg.mkQApp()
    app.aboutToQuit.connect(BR.quit)  # prevent python exception when closing window with system control
    BR.set_window()
    BR.show()

    # BR.setProtocol(args.date, args.slice, args.cell, args.IV)
    # BR.analyzeIVProtocol()

    # BR.set_window()
    # BR.show()
    # ptreedata.sigTreeStateChanged.connect(BR.process_changes)  # connect parameters to their updates

    if sys.flags.interactive == 0:
     app.exec_()


if __name__ == '__main__':
    main()

