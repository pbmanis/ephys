"""
Brige balance tool
Version 0.2

Graphical interface
Part of Ephysanalysis package

Usage:
bridge datadir dbfile

Requires:
Bridge Correction File. this is a Pandas Pickled database with 3 columns:
Cell_ID, Protocol, BridgeValue

If the Bridge Correction File does not exist, then you need to create it with "bridge_tool.py -db name_of_main_database"
This will NOT overwrite an existing Bridge Correction file, but will append new rows with the bridge value set to 0
for all cell_id/protocol files that are present in the main database but not in the bridge file.


"""

import argparse
import functools
import os
import pathlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyqtgraph as pg
from pylibrary.tools import cprint as CP
from pyqtgraph.parametertree import Parameter, ParameterTree
import pyqtgraph.dockarea as PGD
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets


from .. import ephys_analysis as EP
from .. import datareaders as DR

cprint = CP.cprint

# known data dirs by experiment:

experiments = {"nf107": {
                'datadir':'/Volumes/Pegasus_002/ManisLab_Data3/Kasten_Michael/NF107Ai32Het',
                'dbfile': '/Users/pbmanis/Desktop/Python/mrk-nf107-data/datasets/NF107Ai32_Het/NF107Ai32_Het_BridgeCorrections.xlsx'
                },
            }

pg.setConfigOption('antialias', True)



class Bridge():
    """
    Visual adjustment of bridge balance on acq4 data sets. 
    Writes out to the database
    """
    def __init__(self, args, *kwargs):
       
        self.datadir = experiments[args.datadir]['datadir']
        self.dbFilename = experiments[args.datadir]['dbfile']
        self.day = args.day
        self.df = pd.read_excel(str(self.dbFilename))
        
        self.AR = (
            DR.acq4_reader.acq4_reader()
        )  # make our own private cersion of the analysis and reader
        self.SP = EP.SpikeAnalysis.SpikeAnalysis()
        self.RM = EP.RmTauAnalysis.RmTauAnalysis()
        self.curves = []
        self.n_adjusted = 0
        self.this_cell = 0
        self.zoom_mode = False
        
        self.app = pg.mkQApp()
        self.app.setStyle("fusion")
        self.app.setStyleSheet("QLabel{font-size: 11pt;} QText{font-size: 11pt;} {QWidget{font-size: 8pt;}")
        self.app.setStyleSheet("QTreeWidgetItem{font-size: 9pt;}") #  QText{font-size: 11pt;} {QWidget{font-size: 8pt;}")
        # Apply dark theme to Qt application

        pg.setConfigOption('background', 'k')
        pg.setConfigOption('foreground', 'w')
        # QtWidgets.QWidget.__init__(self)
        self.win = pg.QtWidgets.QMainWindow()
        self.DockArea = PGD.DockArea()
        self.win.setCentralWidget(self.DockArea)
        self.win.setWindowTitle("Bridge Balance Corrector V0.3")
        self.win.resize(1600, 1024)
        self.fullscreen_widget = None

        self.Dock_Controls = PGD.Dock("Controls", size=(200, 200))
        
        self.Dock_Table = PGD.Dock("Table", size=(400, 400))
        self.Dock_Viewer = PGD.Dock("Viewer", size=(1280, 400))
        self.DockArea.addDock(self.Dock_Controls, "left")
        self.DockArea.addDock(self.Dock_Viewer, "right", self.Dock_Controls)
        self.DockArea.addDock(self.Dock_Table, "bottom", self.Dock_Viewer)

        self.table = pg.TableWidget(sortable=False)
        self.table.setFont(QtGui.QFont('Arial', 10))

        self.Dock_Table.addWidget(self.table)
        # self.Dock_Table.raiseDock()
        self.setup_controls_and_window()
        # populate the table
        table_data = self.df.T.to_dict()
        self.table.setData(table_data)
        # self.table.resizeRowsToContents()
        self.table.itemDoubleClicked.connect(
            functools.partial(self.on_double_Click, self.table)
        )
        self.win.show()

    def setup_controls_and_window(self, parent=None):

        self.controls = [
            {"name": "Update Database", "type": "action"},
            {"name": "Next Cell", "type": "action"},
            {"name": "Zero Bridge", "type": "action"},
            {"name": "Zoom", "type": "action"},
            {"name": "Unzoom", "type": "action"},
            {"name": "Quit", "type": "action"},
        ]
        self.ptree = ParameterTree()
        self.ptreedata = Parameter.create(
            name="Controls", type="group", children=self.controls
        )
        self.ptree.setParameters(self.ptreedata)
        self.ptree.setMaximumWidth(3200)
        self.ptree.setMinimumWidth(150)
        self.Dock_Controls.addWidget(self.ptree)  # 
        
        self.dataplot = pg.PlotWidget(title="Trace Plots")
        self.Dock_Viewer.addWidget(self.dataplot, rowspan=5, colspan=1)
        self.dataplot.setXRange(-5.0, 2.5, padding=0.2)
        self.dataplot.setContentsMargins(10, 10, 10, 10)

        self.w1 = Slider(-20.0, 40.0, scalar=1.0)

        self.Dock_Controls.addWidget(self.w1)
        self.w1.slider.valueChanged.connect(self.update_data)

    def command_dispatcher(self, param, changes):
        """
        Dispatcher for the commands from parametertree path[0] will be the
        command name path[1] will be the parameter (if there is one) path[2]
        will have the subcommand, if there is one data will be the field data
        (if there is any)
        """
        for param, change, data in changes:
            path = self.ptreedata.childPath(param)

            if path[0] == "Quit":
                print("Quitting")
                exit()
            if path[0] == "Update Database":
                self.save_database()
            if path[0] == "Next Cell":
                self.next_cell()
            if path[0] == "Zero Bridge":
                self.zero_bridge()
            if path[0] == "Zoom":
                self.zoom_mode = True
                self.zoom()
            if path[0] == "UnZoom":
                self.zoom_mode = False
                self.zoom()
 

    def on_double_Click(self, w):
        """
        Double click gets the selected row and then does an analysis
        """
        index = w.selectionModel().currentIndex()
        self.selected_index_row = index.row()
        self.analyze_from_table(index.row())
        print("index: ", index.row)

    def get_table_data(self, index_row):
        value = index_row

        return self.df.iloc[value]

    def analyze_from_table(self, index):
        selected = self.get_table_data(index)
        if selected is None:
            return
        print(selected)
        self.protocolPath = Path(selected.data_directory, selected.Cell_ID, selected.Protocol)
        self.newbr = 0.0
        self.protocolBridge = 0.0
        self.analyzeIVProtocol()

  
    # def setProtocol(self, date, sliceno, cellno, protocolName):
    #     # create an IV protocol path:
    #     self.newbr = 0.0
    #     self.protocolBridge = 0.0
    #     self.date = date
    #     self.slice = sliceno
    #     self.cell = cellno
    #     if not "_" in date:
    #         self.date = date + "_000"
    #     if isinstance(sliceno, int):
    #         self.slice = "slice_{0:03d}".format(sliceno)
    #     if isinstance(cellno, int):
    #         self.cell = "cell_{0:03d}".format(cellno)
    #     self.protocolName = protocolName
    #     self.protocolPath = Path(
    #         self.datadir, self.date, self.slice, self.cell, self.protocolName
    #     )
    #     self.protocolKey = Path(self.date, self.slice, self.cell, self.protocolName)
    #     if not self.protocolPath.is_dir():
    #         cprint("r", f"Protocol directory not found: {str(self.protocolPath):s}")
    #         exit(1)

    def zoom(self):
        """
        Zoom to ~10 mV +/- and 5 msec before, 10 msec after first step
        """
        if self.zoom_mode:
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
        for inext in self.df.index:  # run through all entries in the db for this cell
            date = self.build_date(inext)
            if self.day != "all":  # if all, just do it; otherwise, select
                if not self._date_compare(self.day, date):
                    continue  # keep looking
            self.findIVs(inext)

    def build_date(self, i):
        date = str(
                Path(
                    self.df.at[i, "date"],
                    self.df.at[i, "slice_slice"],
                    self.df.at[i, "cell_cell"],
                )
            )
        print("Date: ", date)
        return date

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
            self.day = self.build_date(inext)
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
        # prots = self.getProtocol(protocolName)
        thisdata, ivprots = self.getIVProtocols()
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
        # self.df.to_excel(str(self.dbFilename))  # now update the database
        self.n_adjusted = 0

    

    def update_traces(self):
        print(f"Update traces, br: {self.protocolBridge:f}")
        self.dataplot.setTitle(
            f"{str(self.protocolPath):s} {self.protocolBridge:.2f}")
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
        self.dataplot.clear()
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
class DoubleSlider(pg.QtWidgets.QSlider):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.decimals = 5
        self._max_int = 10 ** self.decimals

        super().setMinimum(0)
        super().setMaximum(self._max_int)

        self._min_value = 0.0
        self._max_value = 1.0

    @property
    def _value_range(self):
        return self._max_value - self._min_value

    def value(self):
        return float(super().value()) / self._max_int * self._value_range + self._min_value

    def setValue(self, value):
        super().setValue(int((value - self._min_value) / self._value_range * self._max_int))

    def setMinimum(self, value):
        if value > self._max_value:
            raise ValueError("Minimum limit cannot be higher than maximum")

        self._min_value = value
        self.setValue(self.value())

    def setMaximum(self, value):
        if value < self._min_value:
            raise ValueError("Minimum limit cannot be higher than maximum")

        self._max_value = value
        self.setValue(self.value())

    def minimum(self):
        return self._min_value

    def maximum(self):
        return self._max_value


class FloatSlider(pg.Qt.QtWidgets.QSlider):
    def __init__(self, parent, decimals=3, *args, **kargs):
        super(FloatSlider, self).__init__(parent, *args, **kargs)
        self._multi = 10.0 ** float(decimals)
        self.setMinimum(self.minimum())
        self.setMaximum(self.maximum())

    def value(self):
        return float(super(FloatSlider, self).value()) / self._multi

    def setMinimum(self, value):
        self.min_val = value
        return super(FloatSlider, self).setMinimum(int(value * self._multi))

    def setMaximum(self, value):
        self.max_val = value
        return super(FloatSlider, self).setMaximum(int(value * self._multi))

    def setValue(self, value):
        super(FloatSlider, self).setValue(int((value - self.min_val) * self._multi))


class Slider(pg.Qt.QtWidgets.QWidget):
    def __init__(self, minimum, maximum, scalar=1.0, parent=None):
        super(Slider, self).__init__(parent=parent)
        self.verticalLayout = pg.QtWidgets.QVBoxLayout(self)
        self.label = pg.QtWidgets.QLabel(self)
        self.verticalLayout.addWidget(self.label, alignment=pg.QtCore.Qt.AlignmentFlag.AlignHCenter)
        self.horizontalLayout = pg.QtWidgets.QHBoxLayout()
        spacerItem = pg.QtWidgets.QSpacerItem(
            0, 20, pg.QtWidgets.QSizePolicy.Policy.Expanding, pg.QtWidgets.QSizePolicy.Policy.Minimum
        )
        self.horizontalLayout.addItem(spacerItem)
        self.slider = FloatSlider(self, decimals = 2)
        self.slider.setOrientation(pg.QtCore.Qt.Orientation.Horizontal)
        self.horizontalLayout.addWidget(self.slider)
        spacerItem1 = pg.QtWidgets.QSpacerItem(
            0, 20, pg.QtWidgets.QSizePolicy.Policy.Expanding, pg.QtWidgets.QSizePolicy.Policy.Minimum
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
        "--dbfile", type=str, default="", help="Name of database file (including path)"
    )
    parser.add_argument(
        "-d",
        "--day",
        type=str,
        default="all",
        help="Day for analysis if only analyzing data from one day",
    )

    args = parser.parse_args()

    BR = Bridge(args)
    if (sys.flags.interactive != 1) or not hasattr(QtCore, "PYQT_VERSION"):
        QtWidgets.QApplication.instance().exec()


if __name__ == "__main__":
    main()
