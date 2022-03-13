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

import argparse
import os
import pathlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyqtgraph as pg
from pylibrary.tools import cprint as CP
from pyqtgraph.parametertree import Parameter, ParameterTree

from .. import ephysanalysis as EP

cprint = CP.cprint
# datadir = '/Volumes/Pegasus/ManisLab_Data3/Kasten_Michael/NF107Ai32Het'
# dbfile = 'NF107Ai32Het_bcorr2.pkl'


class Bridge(pg.QtGui.QMainWindow):
    """
    Visual adjustment of bridge balance on acq4 data sets. 
    Writes out to the database
    """
    def __init__(self, args):
        super().__init__()
        self.setWindowTitle("Bridge Balance")
        self.datadir = Path(args.datadir)
        self.dbFilename = Path(args.dbfile)
        self.day = args.day
        self.df = pd.read_pickle(str(self.dbFilename))
        if "IV" not in self.df.columns.values:
            print(
                "The Brige Balance Tool requires that IVs have been created and run against the database"
            )
            exit(1)
        self.AR = (
            EP.acq4read.Acq4Read()
        )  # make our own private cersion of the analysis and reader
        self.SP = EP.SpikeAnalysis.SpikeAnalysis()
        self.RM = EP.RmTauAnalysis.RmTauAnalysis()
        self.curves = []
        self.n_adjusted = 0
        self.this_cell = 0

    def setProtocol(self, date, sliceno, cellno, protocolName):
        # create an IV protocol path:
        self.newbr = 0.0
        self.protocolBridge = 0.0
        self.date = date
        self.slice = sliceno
        self.cell = cellno
        if not "_" in date:
            self.date = date + "_000"
        if isinstance(sliceno, int):
            self.slice = "slice_{0:03d}".format(sliceno)
        if isinstance(cellno, int):
            self.cell = "cell_{0:03d}".format(cellno)
        self.protocolName = protocolName
        self.protocolPath = Path(
            self.datadir, self.date, self.slice, self.cell, self.protocolName
        )
        self.protocolKey = Path(self.date, self.slice, self.cell, self.protocolName)
        if not self.protocolPath.is_dir():
            cprint("r", f"Protocol directory not found: {str(self.protocolPath):s}")
            exit(1)

    def zoom(self):
        """
        Zoom to ~10 mV +/- and 5 msec before, 10 msec after first step
        """
        if self.button_zoom_toggle.isChecked():
            t0 = self.AR.tstart * 1e3
            ts = t0 - 5
            te = t0 + 10
            vm = self.RM.analysis_summary["RMP"]
            vm0 = vm - 10
            vm1 = vm + 10
        else:
            ts = 0
            te = np.max(self.AR.time_base) * 1e3
            vm0 = -100
            vm1 = 20
        self.dataplot.setXRange(ts, te)
        self.dataplot.setYRange(vm0, vm1)

    def zero_bridge(self):
        self.newbr = 0.0
        self.w1.slider.setValue(self.w1.getPosValue(self.newbr))
        self.update_data()

    def skip(self):
        """
        Advance to the next entry
        """
        self.next()

    def save(self):
        # first, save new value into the database
        self.update_database()
        self.n_adjusted += 1
        self.next()

    def next(self):
        # get the next protocol in the
        # validivs list, and analyze it.
        if len(self.validivs) == 0:
            cprint("r", "No valid ivs in the list yet")
            return
        if self.currentiv >= len(self.validivs):
            self.exhausted_label.setText(f"Remaining: 0")
            return
        k = self.validivs[self.currentiv]
        p = k.parts
        print(f"Protocols: {str(p):s}")
        self.setProtocol(p[0], p[1], p[2], p[3])
        thisdata, ivprots = self.getIVProtocols()
        try:
            self.protocolBridge = ivprots[self.protocolKey]["BridgeAdjust"]
            print("Next Bridge is %f" % self.protocolBridge)
        except:
            self.protocolBridge = 0.0
        self.analyzeIVProtocol()
        self.currentiv += 1  # cpimt i[]
        self.exhausted_label.setText(f"Remaining: {len(self.validivs)-self.currentiv:d}")
        

    def _date_compare(self, day, date):
        if "_" not in day:
            day = day + "_000"
        if day[-1] in ["\\", "/"]:
            day = day[:-1]
        if day != date:
            # print(f"Day {day:s} did not match date: {date:s}")
            return False
        return True

    def printAllValidIVs(self):
        validivs = []
        print(f"Listing valid IVs, for date = {self.day:s}")
        for i in self.df.index:  # run through all entries in the db
            date = str(
                Path(
                    self.df.at[i, "date"],
                    self.df.at[i, "slice_slice"],
                    self.df.at[i, "cell_cell"],
                )
            )
            if self.day != "all":  # if all, just do it; otherwise, build a selection
                if not self._date_compare(self.day, date):
                    continue  # keep looking
            ivdata = self.df.at[i, "IV"]  # get the IV
            # print(ivdata)
            if isinstance(ivdata, float) or len(ivdata) == 0:
                print(f"No ivdata for: {i:d} at {self.df.at[i, 'date']:s}")
                continue

            for k in list(ivdata.keys()):  # multiple IV's by keys in the dict
                if "BridgeAdjust" not in ivdata[k] or (
                    "BridgeAdjust" in ivdata[k] and ivdata[k]["BridgeAdjust"] == 0.0
                ):
                    print(f"         Do Bridge Adjustment for:  {str(k):s}")
                else:
                    print(
                        f"           BridgeAdjust exists for: {str(k):s}  {ivdata[k]['BridgeAdjust']:f}"
                    )
                validivs.append(k)
        print(
            f"Checked {i:d} indexed protocols, found {len(validivs):d} previously computed IVs"
        )

    def setupValidIVs(self):
        """
        Set up the list of validivs
        then move on and do the first one
        """
        self.this_cell = -1
        for i in self.df.index:  # run through all entries in the db for this cell
            date = str(
                Path(
                    self.df.at[i, "date"],
                    self.df.at[i, "slice_slice"],
                    self.df.at[i, "cell_cell"],
                )
            )
            print("Date: ", date)
            if self.day != "all":  # if all, just do it; otherwise, select
                if not self._date_compare(self.day, date):
                    continue  # keep looking
            self.findIVs(i)

    def findIVs(self, index):
        self.validivs = []  # list of valid IVs for this cell at the dataframe index
        self.currentiv = 0  # number for the current IV we are doing in the list (bumped up in next())
        i = index
        print('index: ', index)
        ivdata = self.df.at[i, "IV"]  # get the IV
        print("ivdata: ", ivdata)
        if len(ivdata) == 0:
            print("no ivdata for: ", i, self.df.at[i, "date"])
            self.this_cell = i+1

        for k in list(ivdata.keys()):  # multiple IV's by keys in the dict
            if "BridgeAdjust" not in ivdata[k] or (
                "BridgeAdjust" in ivdata[k] and ivdata[k]["BridgeAdjust"] == 0.0
            ):
                print("         Do Bridge Adjustment for: ", k)
            else:
                print(
                      "         BridgeAdjust exists for: {0:s}  {1:f}".format(
                        str(k), ivdata[k]["BridgeAdjust"]
                    )
                )
            self.validivs.append(k)  # add iv key to the valid iv dict
        self.exhausted_label.setText(f"Remaining: {len(self.validivs):d}")
        self.this_cell = i
        self.next()  # go get the first valid protocol and analyze it

    def next_cell(self):
        print('Getting next cell')
        if self.this_cell < 0:
            return
        inext = self.this_cell + 1
        if inext < len(self.df.index):
            self.day = str(
                Path(
                    self.df.at[inext, "date"],
                    self.df.at[inext, "slice_slice"],
                    self.df.at[inext, "cell_cell"],
                )
            )
            self.findIVs(inext)
            
        
    def analyzeIVProtocol(self):
        #        print('opening: ', self.protocolPath)
        self.AR.setProtocol(
            self.protocolPath
        )  # define the protocol path where the data is
        threshold = -0.020
        if self.AR.getData():  # get that data.
            self.RM.setup(
                clamps=self.AR, spikes=self.SP, bridge_offset=0.0
            )  # doing setup here also does bridge correction
            self.SP.setup(
                clamps=self.AR,
                threshold=threshold,
                refractory=0.0001,
                peakwidth=0.001,
                interpolate=False,
                verify=False,
                mode="peak",
            )
            self.SP.analyzeSpikes()
            #            self.SP.analyzeSpikeShape()
            #            self.SP.analyzeSpikes_brief(mode='baseline')
            #            self.SP.analyzeSpikes_brief(mode='poststimulus')
            self.RM.analyze(
                rmpregion=[0.0, self.AR.tstart - 0.001],
                tauregion=[
                    self.AR.tstart,
                    self.AR.tstart + (self.AR.tend - self.AR.tstart) / 5.0,
                ],
            )
            self.cmd = 1e9 * self.AR.cmd_wave.view(np.ndarray)
            self.update_traces()

    def quit(self):
        exit(0)

    def getIVProtocols(self):
        thisdata = self.df.index[
            (self.df["date"] == self.date)
            & (self.df["slice_slice"] == self.slice)
            & (self.df["cell_cell"] == self.cell)
        ].tolist()
        if len(thisdata) > 1:
            raise ValueError("Search for data resulted in more than one entry!")
        ivprots = self.df.iloc[thisdata]["IV"].values[
            0
        ]  # all the protocols in the dict
        return thisdata, ivprots

    def getProtocol(self, protocolName):
        thisdata, ivprots = self.getIVProtocols()
        if protocolName not in ivprots.keys():
            return None
        else:
            return ivprots[protocolName]

    def check_for_bridge(self, protocolName):
        prots = self.getProtocol(protcolName)
        if "BridgeAdjust" not in ivprots[protocolName].keys():
            return False
        else:
            return True

    def printCurrent(self):
        """
        Print information about the currently selected protocol
        """
        thisdata, ivprots = self.getIVProtocols()
        thisprot = ivprots[self.protocolKey]
        print("\n", "-"*80)
        print(f"Protocol: {str(self.protocolKey):s}")
        for k in thisprot.keys():
            if isinstance(thisprot[k], dict):
                print(f"{k:s}:")
                for dk in thisprot[k].keys():
                    print(f"    {dk:>24s} = {str(thisprot[k][dk]):<24s}")
            else:
                print(f"{k:s}:   {str(thisprot[k]):s}")
        print("-"*80, "\n")

    def update_database(self):
        thisdata, ivprots = self.getIVProtocols()
        cprint("g", f"    Updating database IV:  {str(ivprots[self.protocolKey]):s}")
        cprint("g", f"         with bridge {self.newbr:.3f}")
        ivprots[self.protocolKey]["BridgeAdjust"] = (
            self.newbr * 1e6
        )  # convert to ohms here
        self.df.at[thisdata[0], "IV"] = ivprots  # update with the new bridge value
        self.skipflag = False

    def save_database(self):
        # before updating, save previous version
        # self.dbFilename.rename(self.dbFilename.with_suffix('.bak'))
        cprint("y",
            f"Saving database so far ({(self.n_adjusted - 1):d} entries with Bridge Adjustment)"
        )
        self.update_database()
        self.df.to_pickle(str(self.dbFilename))  # now update the database
        self.n_adjusted = 0

    def set_window(self, parent=None):
        super(Bridge, self).__init__(parent=parent)
        self.win = pg.GraphicsWindow(title="Bridge Balance Tool")
        self.main_layout = pg.QtGui.QGridLayout()
        self.main_layout.setSpacing(8)
        self.win.setLayout(self.main_layout)
        self.win.resize(1280, 800)
        self.win.setWindowTitle("No File")
        self.buttons = pg.QtGui.QGridLayout()

        self.buttons = pg.QtGui.QVBoxLayout(self)

        self.button_do_valid_ivs = pg.QtGui.QPushButton("Setup Valid IVs")
        self.buttons.addWidget(self.button_do_valid_ivs, stretch=2)
        self.button_do_valid_ivs.clicked.connect(self.setupValidIVs)

        self.button_save_and_load = pg.QtGui.QPushButton("Save and load Next")
        self.buttons.addWidget(self.button_save_and_load, stretch=2)
        self.button_save_and_load.clicked.connect(self.save)

        self.button_skip = pg.QtGui.QPushButton("Skip this IV")
        self.buttons.addWidget(self.button_skip, stretch=2)
        self.button_skip.clicked.connect(self.skip)

        self.button_save_db = pg.QtGui.QPushButton("Save Database")
        self.buttons.addWidget(self.button_save_db, stretch=2)
        self.button_save_db.clicked.connect(self.save_database)

        self.exhausted_label = pg.QtGui.QLabel(self)
        self.exhausted_label.setAutoFillBackground(True)
        palette = pg.QtGui.QPalette()
        palette.setColor(pg.QtGui.QPalette.Window, pg.QtCore.Qt.gray)
        self.exhausted_label.setPalette(palette)
        self.exhausted_label.setAlignment(pg.QtCore.Qt.AlignRight)
        self.exhausted_label.setText(f"Remaining: 0")
        self.buttons.addWidget(self.exhausted_label, stretch=2)
        
        spacerItem0 = pg.QtGui.QSpacerItem(
            0, 20, pg.QtGui.QSizePolicy.Expanding, pg.QtGui.QSizePolicy.Minimum
        )
        self.buttons.addItem(spacerItem0)

        self.button_next_cell = pg.QtGui.QPushButton("Next Cell")
        self.buttons.addWidget(self.button_next_cell, stretch=2)
        self.button_next_cell.clicked.connect(self.next_cell)
        
        self.bzero = pg.QtGui.QPushButton("Zero Bridge")
        self.buttons.addWidget(self.bzero, stretch=2)
        self.bzero.clicked.connect(self.zero_bridge)

        self.button_zoom_toggle = pg.QtGui.QPushButton("Zoom/unzoom")
        self.button_zoom_toggle.setCheckable(True)
        self.button_zoom_toggle.setChecked(False)
        self.buttons.addWidget(self.button_zoom_toggle, stretch=10)
        self.button_zoom_toggle.clicked.connect(self.zoom)

        self.button_list_IVs = pg.QtGui.QPushButton("List Valid IVs")
        self.buttons.addWidget(self.button_list_IVs, stretch=2)
        self.button_list_IVs.clicked.connect(self.printAllValidIVs)
        self.br_label = pg.QtGui.QLabel(self)

        self.button_print_current = pg.QtGui.QPushButton("Print Current Info")
        self.buttons.addWidget(self.button_print_current, stretch=2)
        self.button_print_current.clicked.connect(self.printCurrent)

        spacerItem1 = pg.QtGui.QSpacerItem(
            0, 200, pg.QtGui.QSizePolicy.Expanding, pg.QtGui.QSizePolicy.Minimum
        )
        self.buttons.addItem(spacerItem1)

        self.button_quit = pg.QtGui.QPushButton("Quit")
        self.buttons.addWidget(self.button_quit, stretch=10)
        self.button_quit.clicked.connect(self.quit)

        spacerItem = pg.QtGui.QSpacerItem(
            0, 10, pg.QtGui.QSizePolicy.Expanding, pg.QtGui.QSizePolicy.Minimum
        )
        self.buttons.addItem(spacerItem)

        self.br_label = pg.QtGui.QLabel(self)
        self.br_label.setAutoFillBackground(True)
        palette = pg.QtGui.QPalette()
        palette.setColor(pg.QtGui.QPalette.Window, pg.QtCore.Qt.gray)
        self.br_label.setPalette(palette)
        self.br_label.setAlignment(pg.QtCore.Qt.AlignRight)
        self.br_label.setText(f"Bridge: 0.0 Mohm")
        self.buttons.addWidget(self.br_label, stretch=2)

        self.dataplot = pg.PlotWidget()
        self.dlayout = pg.QtGui.QGridLayout()
        self.dlayout.addWidget(self.dataplot, 0, 0, 10, 8)

        self.sliderpane = pg.QtGui.QVBoxLayout(self)
        self.w1 = Slider(-20.0, 40.0, scalar=1.0)
        self.sliderpane.addWidget(self.w1)

        self.main_layout.addLayout(self.buttons, 0, 0, 10, 1)
        # self.main_layout.addLayout(self.reports, 1, 0, 10, 1)
        self.main_layout.addLayout(self.sliderpane, 10, 1, 1, 8)
        self.main_layout.addLayout(self.dlayout, 0, 1, 10, 8)
        self.w1.slider.valueChanged.connect(self.update_data)
        self.main_layout.setColumnStretch(0, 0)  # reduce width of LHS column of buttons
        self.main_layout.setColumnStretch(1, 20)  # and stretch out the data dispaly
        self.main_layout.setColumnStretch(2, 0)  # and stretch out the data dispaly

    def update_traces(self):
        print(f"Update traces, br: {self.protocolBridge:f}")
        self.dataplot.setTitle(
            f"{str(self.protocolKey):s} {self.protocolBridge:.2f}")
        self.newbr = self.protocolBridge / 1e6  # convert to megaohms
        self.w1.slider.setValue(self.newbr)
        cmdindxs = np.unique(self.AR.commandLevels)  # find the unique voltages
        colindxs = [
            int(np.where(cmdindxs == self.AR.commandLevels[i])[0])
            for i in range(len(self.AR.commandLevels))
        ]  #
        for c in self.curves:
            c.clear()
        self.curves = []
        for i, d in enumerate(self.AR.traces):
            self.curves.append(
                self.dataplot.plot(
                    self.AR.time_base * 1e3,
                    self.AR.traces[i, :] * 1e3 - (self.cmd[i] * self.newbr),
                    pen=pg.intColor(colindxs[i], len(cmdindxs), maxValue=255),
                )
            )

    def update_data(self):
        a = self.w1.x
        self.newbr = self.w1.x
        self.br_label.setText(f"{self.newbr:8.3f}")
        # print(len(self.curves))
        # print(len(self.AR.traces))
        # print(len(self.cmd))
        for i, c in enumerate(self.curves):
            c.setData(
                self.AR.time_base * 1e3,
                self.AR.traces[i, :] * 1e3 - (self.cmd[i] * self.newbr),
            )


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
        super(FloatSlider, self).setValue(int((value - self.min_val) * self._multi))


class Slider(pg.QtGui.QWidget):
    def __init__(self, minimum, maximum, scalar=1.0, parent=None):
        super(Slider, self).__init__(parent=parent)
        self.verticalLayout = pg.QtGui.QVBoxLayout(self)
        self.label = pg.QtGui.QLabel(self)
        self.verticalLayout.addWidget(self.label, alignment=pg.QtCore.Qt.AlignHCenter)
        self.horizontalLayout = pg.QtGui.QHBoxLayout()
        spacerItem = pg.QtGui.QSpacerItem(
            0, 20, pg.QtGui.QSizePolicy.Expanding, pg.QtGui.QSizePolicy.Minimum
        )
        self.horizontalLayout.addItem(spacerItem)
        self.slider = FloatSlider(self, decimals=2)
        self.slider.setOrientation(pg.QtCore.Qt.Vertical)
        self.horizontalLayout.addWidget(self.slider)
        spacerItem1 = pg.QtGui.QSpacerItem(
            0, 20, pg.QtGui.QSizePolicy.Expanding, pg.QtGui.QSizePolicy.Minimum
        )
        self.horizontalLayout.addItem(spacerItem1)
        self.verticalLayout.addLayout(self.horizontalLayout)
        self.resize(self.sizeHint())

        self.minimum = minimum * scalar
        self.maximum = maximum * scalar
        self.scalar = scalar
        self.slider.setMinimum(self.minimum)
        self.slider.setMaximum(self.maximum)
        # self.slider.setRange(self.minimum, self.maximum)
        self.slider.valueChanged.connect(self.setLabelValue)
        self.setLabelValue(self.slider.value())

    def setLabelValue(self, value):
        self.x = (
            self.minimum
            + (float(value) / (self.slider.maximum() - self.slider.minimum()))
            * (self.maximum - self.minimum)
        ) / self.scalar
        self.label.setText("{0:6.2f}".format(self.x))

    def getPosValue(self, x):
        return (
            (x - self.minimum)
            * (self.slider.maximum() - self.slider.minimum())
            / (self.maximum - self.minimum)
        )


def main():
    parser = argparse.ArgumentParser(
        description="""Bridge balance correction tool.
            Allows the user to adjust the bridge balance on IV curves from cells post-hoc.
            Bridge values are read from an existing database (generated by dataSummary),
            and the modified (delta) values of bridge resistance are written back to 
            the dataSummary database for future use.
            Steps:
            1. call with the path to the cell and the database
            2. setupValidIVs will find the valid IVs, and store them in a list
                Then it brings in the first one.
            3. Make adjustments to bridge balance
            4. Either do: "save and next" (updates database, brings up next IV in the list)
                      or: "skip" (move on to next IV)
            During the adjustments, you can zoom in/out to look carefully at how the balance is set
            
            --10/2018, 11/2021 pbm
            """
    )
    parser.add_argument(
        "datadir", type=str, default="", help="Full path to data directory"
    )
    parser.add_argument(
        "dbfile", type=str, default="", help="Name of database file (including path)"
    )
    parser.add_argument(
        "-d",
        "--day",
        type=str,
        default="all",
        help="Day for analysis if only analyzing data from one day",
    )

    args = parser.parse_args()

    app = pg.mkQApp()
    BR = Bridge(args)
    # app.aboutToQuit.connect(BR.quit)  # prevent python exception when closing window with system control
    BR.set_window()
    BR.show()

    # BR.setProtocol(args.date, args.slice, args.cell, args.IV)
    # BR.analyzeIVProtocol()

    # BR.set_window()
    # BR.show()
    # ptreedata.sigTreeStateChanged.connect(BR.process_changes)  # connect parameters to their updates

    if sys.flags.interactive == 0:
        app.exec_()


if __name__ == "__main__":
    main()
