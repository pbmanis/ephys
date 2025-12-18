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
import datetime
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
import ephys.tools.bridge_tool as BT
import ephys.tools.get_configuration as get_config


from .. import ephys_analysis as EP
from .. import datareaders as DR

cprint = CP.cprint


# To make slider handle bigger:
sliderStyle = """ 
    QSlider{
    min-height: 300px;
    max-height: 300px;
    min-width: 100;
    max-width: 100;
    background: #353535;
    }
    QSlider::groove:vertical {
        background: darkgray;
        border: 1px solid #999999;
        height: 300px;
        width: 5px;
        margin: 0px;
        alignment: center;
        }

    QSlider::handle:vertical {
        background: red;
        border: 1px solid #5c5c5c;
        height: 3px;
        width: 150px;
        margin: -30 30;
        }
    
    QSlider::add-page:vertical {
        background: blue;
        }
    QSlider::sub-page:vertical {
        background: white;
        }
    """


class Bridge:
    """
    Visual adjustment of bridge balance on acq4 data sets.
    Writes out to the database
    """

    def __init__(self, experiments,  args, *kwargs):
        if args.experiment in experiments.keys():
            self.experiment = experiments[args.experiment]
        else:
            raise ValueError(f"Experiment {args.experiment} not found in configuration file; found [{self.experiment.keys()}]")
        self.datadir = Path(self.experiment["rawdatapath"], self.experiment["directory"])
        self.dbFilename = Path(
            self.experiment["databasepath"],
            self.experiment["directory"],
            self.experiment["datasummaryFilename"],
        )
        self.day = args.day
        self.df = pd.read_pickle(str(self.dbFilename))
        cols = self.df.columns.values
        dropcols = []
        for c in cols:
            if c.startswith("Unnamed"):
                dropcols.append(c)
        self.df = self.df.drop(dropcols, axis=1)
        cols = self.df.columns
        if "update_date" not in cols:
            self.df.insert(6, "update_date", "")
        self.df.reset_index()

        self.AR = (
            DR.acq4_reader.acq4_reader()
        )  # make our own private cersion of the analysis and reader
        self.SP = EP.spike_analysis.SpikeAnalysis()
        self.RM = EP.rm_tau_analysis.RmTauAnalysis()
        self.curves = []
        self.n_adjusted = 0
        self.this_cell = 0
        self.zoom_mode = False
        self.updated = []  # a list of rows that have been updated since last save
        self.header_map = {}
        dark_palette = QtGui.QPalette()
        white = QtGui.QColor(255, 255, 255)
        black = QtGui.QColor(0, 0, 0)
        red = QtGui.QColor(255, 0, 0)
        dark_palette.setColor(QtGui.QPalette.ColorRole.Window, QtGui.QColor(53, 53, 53))
        dark_palette.setColor(QtGui.QPalette.ColorRole.WindowText, white)
        dark_palette.setColor(QtGui.QPalette.ColorRole.Base, QtGui.QColor(53, 53, 53))
        dark_palette.setColor(QtGui.QPalette.ColorRole.AlternateBase, QtGui.QColor(53, 53, 53))
        dark_palette.setColor(QtGui.QPalette.ColorRole.ToolTipBase, white)
        dark_palette.setColor(QtGui.QPalette.ColorRole.ToolTipText, white)
        dark_palette.setColor(QtGui.QPalette.ColorRole.Text, white)
        dark_palette.setColor(QtGui.QPalette.ColorRole.Button, QtGui.QColor(53, 53, 53))
        dark_palette.setColor(QtGui.QPalette.ColorRole.ButtonText, white)
        dark_palette.setColor(QtGui.QPalette.ColorRole.BrightText, red)
        dark_palette.setColor(QtGui.QPalette.ColorRole.Link, QtGui.QColor(42, 130, 218))
        dark_palette.setColor(QtGui.QPalette.ColorRole.Highlight, QtGui.QColor(42, 130, 218))
        dark_palette.setColor(QtGui.QPalette.ColorRole.HighlightedText, black)

        self.app = pg.mkQApp()
        self.app.setStyle("fusion")
        self.app.setStyleSheet(
            "QLabel{font-size: 11pt;} QText{font-size: 11pt;} {QWidget{font-size: 9pt;}"
        )
        self.app.setStyleSheet(
            "QTreeWidgetItem{font-size: 9pt;}"
        )  #  QText{font-size: 11pt;} {QWidget{font-size: 8pt;}")
        self.app.setPalette(dark_palette)
        # Apply dark theme to Qt application

        pg.setConfigOption("background", "k")
        pg.setConfigOption("foreground", "w")
        # QtWidgets.QWidget.__init__(self)
        self.win = pg.QtWidgets.QMainWindow()
        self.DockArea = PGD.DockArea()
        self.win.setCentralWidget(self.DockArea)
        self.win.setWindowTitle("Bridge Balance Corrector V0.4")
        self.win.resize(1600, 1024)
        self.fullscreen_widget = None

        self.Dock_Controls = PGD.Dock("Controls", size=(200, 200))

        self.Dock_Table = PGD.Dock("Table", size=(400, 400))
        self.Dock_Viewer = PGD.Dock("Viewer", size=(1280, 400))
        self.DockArea.addDock(self.Dock_Controls, "left")
        self.DockArea.addDock(self.Dock_Viewer, "right", self.Dock_Controls)
        self.DockArea.addDock(self.Dock_Table, "bottom", self.Dock_Viewer)

        self.table = pg.TableWidget(sortable=False)
        self.table.setFont(QtGui.QFont("Arial", 11))

        self.Dock_Table.addWidget(self.table)
        self.setup_controls_and_window()
        # populate the table
        def _data_complete_to_series(row):
            dc = row.data_complete.split(",")
            dc = [p.strip() for p in dc if p != "nan" and "CCIV".casefold() in p.casefold()]
            row.protocols = pd.Series(dc)
            return row
            
        self.df['protocols'] = {}
        self.df  = self.df.apply(_data_complete_to_series, axis=1)
        self.df = self.df.explode('protocols', ignore_index=True) 

        table_data_df = self.df[['cell_id', 'cell_type', 'protocols', 'date']]   # reduce the table structure to the protocols
        # table_data_df = table_data_df.explode('protocols', ignore_index=True)  # expand so each row is a protocol
        table_data_df = table_data_df.T.to_dict()
        self.table.setData(table_data_df)
        self.table.resizeRowsToContents()
        self.table.itemDoubleClicked.connect(functools.partial(self.on_double_Click, self.table))
        # seems QtTableWidget should provide this... or a lookup method
        # Create a map from column names to column number
        self.n_cols = self.table.columnCount()
        for n in range(self.n_cols):
            hdr = self.table.horizontalHeaderItem(n).text()
            self.header_map[hdr] = n
        self.win.show()

    def setup_controls_and_window(self, parent=None):

        self.controls = [
            {"name": "Update Database", "type": "action"},
            {"name": "Accept", "type": "action"},
            {"name": "Next protocol", "type": "action"},
            {"name": "Previous protocol", "type": "action"},
            {"name": "Zero Bridge", "type": "action"},
            {"name": "Reset Bridge", "type": "action"},
            {"name": "Zoom", "type": "action"},
            {"name": "UnZoom", "type": "action"},
            # {"name": "Br Slider", "type": "action", },
            {"name": "Quit", "type": "action"},
        ]
        self.ptree = ParameterTree()
        self.ptreedata = Parameter.create(name="Controls", type="group", children=self.controls)
        self.ptree.setParameters(self.ptreedata)
        self.ptree.setMaximumWidth(320)
        self.ptree.setMinimumWidth(150)
        self.Dock_Controls.addWidget(self.ptree)  #

        self.dataplot = pg.PlotWidget(title="Trace Plots")
        pg.setConfigOption("leftButtonPan", True)
        self.Dock_Viewer.addWidget(self.dataplot, rowspan=5, colspan=1)
        self.dataplot.setXRange(-5.0, 2.5, padding=0.2)
        self.dataplot.setContentsMargins(10, 10, 10, 10)

        self.w1 = Slider(-20.0, 20.0, scalar=10.)

        self.Dock_Controls.addWidget(self.w1)
        self.w1.slider.valueChanged.connect(self.update_data_from_slider)
        self.ptreedata.sigTreeStateChanged.connect(self.command_dispatcher)

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
            if path[0] == "Accept":
                self.accept_new_value()
            if path[0] == "Update Database":
                self.update_database()
            if path[0] == "Next protocol":
                self.next_protocol()
            if path[0] == "Previous protocol":
                self.previous_protocol()
            if path[0] == "Zero Bridge":
                self.zero_bridge()
            if path[0] == "Reset Bridge":
                self.reset_bridge()
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

    def get_table_data(self, index_row):
        value = index_row
        return self.df.iloc[value]

    def previous_protocol(self):
        if self.this_cell < 0:
            return
        self.last_row = self.selected_index_row
        if self.last_row == 0:
            return
        self.selected_index_row -= 1
        self.get_protocol()

    def next_protocol(self):
        if self.this_cell < 0:
            return
        self.last_row = self.selected_index_row
        if self.last_row >= self.table.rowCount():
            return
        self.selected_index_row += 1
        self.get_protocol()

    def get_protocol(self):
        # deselect previous row, then select new row
        self.table.setRangeSelected(
            QtWidgets.QTableWidgetSelectionRange(
                self.last_row,
                0,
                self.last_row - 1,
                7,
            ),
            False,
        )
        self.table.setRangeSelected(
            QtWidgets.QTableWidgetSelectionRange(
                self.selected_index_row,
                0,
                self.selected_index_row,
                7,
            ),
            True,
        )
        index = self.table.selectionModel().selectedRows()
        self.table.scrollTo(index[0], hint=QtWidgets.QAbstractItemView.ScrollHint.PositionAtCenter)
        # This is for the bar to scroll automatically and then the current item added is always visible
        self.analyze_from_table(self.selected_index_row)

    def protocols_to_list(self, selected):
        """
        Get the complete protocols from the selected row
        """
        protocols = [p.strip() for p in selected.data_complete.split(",") if p.strip().startswith("CCIV")]
        return protocols
    
    def analyze_from_table(self, index):
        selected = self.get_table_data(index)
        if selected is None:
            return
        print(selected.data_complete)
        protocols = self.protocols_to_list(selected)

        self.protocolPath = Path(selected.data_directory, selected.cell_id, protocols[0])
        self.newbr = 0.0
        self.protocolBridge = 0.0
        self.analyzeIVProtocol()

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
        # print("zero bridge")
        self.newbr = 0.0
        self.w1.slider.setValue(self.w1.getPosValue(self.newbr))
        self.update_data(self.newbr)

    def reset_bridge(self):
        # print("reset bridge")
        row = self.df.iloc[self.selected_index_row]
        self.protocolBridge = row["BridgeAdjust(ohm)"]
        self.newbr = self.protocolBridge / 1e6  # convert to megaohms
        # print("brige new: ", self.newbr, " prot: ", self.protocolBridge/1e6)
        # self.w1.slider.setValue(self.w1.getPosValue(self.newbr))
        self.updated.append(row)
        self.update_data(self.newbr)

    # def skip(self):
    #     """
    #     Advance to the next entry
    #     """
    #     self.next()

    # def save(self):
    #     # first, save new value into the database
    #     self.update_database()
    #     self.n_adjusted += 1
    #     self.next()

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
        #  print(f"Protocols: {str(p):s}")
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
        print(f"Checked {i:d} indexed protocols, found {len(validivs):d} previously computed IVs")

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
        self.currentiv = (
            0  # number for the current IV we are doing in the list (bumped up in next())
        )
        i = index
        print("index: ", index)
        ivdata = self.df.at[i, "IV"]  # get the IV
        print("ivdata: ", ivdata)
        if len(ivdata) == 0:
            print("no ivdata for: ", i, self.df.at[i, "date"])
            self.this_cell = i + 1

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

    def analyzeIVProtocol(self):
        #        print('opening: ', self.protocolPath)
        self.AR.setProtocol(self.protocolPath)  # define the protocol path where the data is
        if self.AR.getData():  # get that data.
            print(self.AR.CCComp)
            print("tstart, end: ", self.AR.tstart, self.AR.tend)
            self.RM.setup(
                clamps=self.AR, spikes=self.SP, bridge_offset=0.0
            )  # doing setup here also does bridge correction
            self.SP.setup(
                clamps=self.AR,
                threshold=self.experiment["AP_threshold_V"],
                refractory=0.0005,
                peakwidth=0.001,
                interpolate=False,
                verify=False,

            )
            self.SP.set_detector(detector = self.experiment['spike_detector'],
                                 pars=self.experiment['detector_pars'])
            for analysismode in ["baseline", "evoked", "poststimulus"]:
                self.SP.analyzeSpikes_brief(analysismode)
            # self.SP.analyzeSpikes()
            # print(self.AR.tstart, self.AR.tend)
            self.RM.analyze(
                rmp_region=[0.0, self.AR.tstart - 0.001],
                tau_region=[
                    self.AR.tstart,
                    self.AR.tstart + (self.AR.tend - self.AR.tstart) / 5.0,
                ],
                average_flag=False,

            )
            self.cmd = 1e9 * self.AR.cmd_wave.view(np.ndarray)
            self.draw_traces()

    def quit(self):
        exit(0)

    # def getIVProtocols(self):
    #     thisdata = self.df.index]
    #         (self.df["date"] == self.date)
    #         & (self.df["slice_slice"] == self.slice)
    #         & (self.df["cell_cell"] == self.cell)
    #     ].tolist()
    #     if len(thisdata) > 1:
    #         raise ValueError("Search for data resulted in more than one entry!")
    #     ivprots = self.df.iloc[thisdata]["IV"].values[
    #         0
    #     ]  # all the protocols in the dict
    #     return thisdata, ivprots

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
        print("\n", "-" * 80)
        print(f"Protocol: {str(self.protocolKey):s}")
        for k in thisprot.keys():
            if isinstance(thisprot[k], dict):
                print(f"{k:s}:")
                for dk in thisprot[k].keys():
                    print(f"    {dk:>24s} = {str(thisprot[k][dk]):<24s}")
            else:
                print(f"{k:s}:   {str(thisprot[k]):s}")
        print("-" * 80, "\n")

    def accept_new_value(self):
        self.df.loc[self.selected_index_row, "update_date"] = datetime.datetime.now().strftime(
            "%Y:%m:%d %H:%M:%S"
        )
        self.df.loc[self.selected_index_row, "BridgeAdjust(ohm)"] = (
            self.newbr * 1e6
        )  # convert to ohms here
        if not self.df.loc[self.selected_index_row, "BridgeEnabled"]:
            self.df.loc[self.selected_index_row, "BridgeResistance"] = 0.0
        # update "TrueRsistance":
        self.df.loc[self.selected_index_row, "TrueResistance"] = (
            self.df.loc[self.selected_index_row, "BridgeResistance"]
            + self.df.loc[self.selected_index_row, "BridgeAdjust(ohm)"]
        )
        # update in table with red text
        brc = self.header_map["BridgeAdjust(ohm)"]
        self.table.item(self.selected_index_row, brc).setText(
            f"{self.df.loc[self.selected_index_row, 'BridgeAdjust(ohm)']:f}",
        )
        self.table.item(self.selected_index_row, brc).setForeground(pg.mkBrush("r"))
        btr = self.header_map["TrueResistance"]
        self.table.item(self.selected_index_row, btr).setText(
            f"{self.df.loc[self.selected_index_row, 'TrueResistance']:f}",
        )
        self.table.item(self.selected_index_row, btr).setForeground(pg.mkBrush("r"))

    def update_database(self):
        self.df.reset_index()
        BT.save_excel(self.df, self.dbFilename)  # this also formats the excel file
        self.updated = []
        self.n_adjusted = 0

    def rescale(self):
        vb = self.dataplot.getViewBox()
        vb.enableAutoRange(enable=True)
        

    def draw_traces(self):
        print(f"Update traces, br: {self.protocolBridge:f}")
        self.dataplot.setTitle(f"{str(self.protocolPath):s} {self.protocolBridge:.2f}")
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
        bc = self.newbr
        if np.isnan(bc):
            bc = 0.0
        self.newbr = bc
        for i, d in enumerate(self.AR.traces):
            self.curves.append(
                self.dataplot.plot(
                    self.AR.time_base * 1e3,
                    self.AR.traces[i, :] * 1e3 - (self.cmd[i] * bc),
                    pen=pg.intColor(colindxs[i], len(cmdindxs), maxValue=255),
                )
            )
        self.rescale()
        self.zoom()

    def update_data_from_slider(self):
        br = self.w1.x
        self.update_data(bridge_value=br)

    def update_data(self, bridge_value=np.nan):

        if np.isnan(bridge_value):
            bc = 0.0
        self.newbr = bridge_value
        for i, c in enumerate(self.curves):
            c.setData(
                self.AR.time_base * 1e3,
                self.AR.traces[i, :] * 1e3 - (self.cmd[i] * bridge_value),
            )
        # self.updated.append(self.selected_index_row)
        # self.rescale()
        # self.zoom()


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
    def __init__(self, minimum:float=-20., maximum:float=20., scalar:float=1.0, parent=None):
        super(Slider, self).__init__(parent=parent)
        self.label = pg.QtWidgets.QLabel(self)

        spacerItem = pg.QtWidgets.QSpacerItem(
            0,
            40,
            pg.QtWidgets.QSizePolicy.Policy.Expanding,
            pg.QtWidgets.QSizePolicy.Policy.Minimum,
        )

        self.verticalLayout = pg.QtWidgets.QVBoxLayout(self)
        self.verticalLayout.addWidget(self.label, alignment=pg.QtCore.Qt.AlignmentFlag.AlignHCenter)

        self.horizontalLayout = pg.QtWidgets.QHBoxLayout()
        self.horizontalLayout.addItem(spacerItem)

        self.slider = FloatSlider(self, decimals=2)
        self.slider.setOrientation(pg.QtCore.Qt.Orientation.Vertical)  # Horizontal)
        self.slider.setStyleSheet(sliderStyle)
        self.verticalLayout.addWidget(self.slider)
        self.slider.setTickInterval(5)
        self.slider.setSingleStep(1)
        self.slider.setTickPosition(pg.QtWidgets.QSlider.TickPosition.TicksRight)
        # spacerItem1 = pg.QtWidgets.QSpacerItem(
        #     0, 20, pg.QtWidgets.QSizePolicy.Policy.Expanding, pg.QtWidgets.QSizePolicy.Policy.Minimum
        # )
        # self.horizontalLayout.addItem(spacerItem1)
        # self.verticalLayout.addLayout(self.horizontalLayout)
        self.resize(self.sizeHint())

        self.minimum = minimum * scalar
        self.maximum = maximum * scalar
        self.scalar = scalar
        self.slider.setMinimum(self.minimum)
        self.slider.setMaximum(self.maximum)
        self.slider.setRange(int(self.minimum), int(self.maximum))
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



def main_gui():
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
    parser.add_argument("--datadir", type=str, default="", help="Full path to data directory")
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
    parser.add_argument(
        "-e",
        "--experiment",
        type=str,
        default=None,
        help="specify experiment in the configuration file for this data set",
    )

    args = parser.parse_args()
    experiments = None
    # known data dirs by experiment:
    try:
        print("Current Directory: ", os.getcwd())
        cpath = Path("./config", "experiments.cfg")
        datasets, experiments = get_config.get_configuration(cpath)
        print("Datasets in current experiment configuration: ", experiments.keys())


    except FileNotFoundError:
        # no valid configuration file found
        raise FileNotFoundError(
            "No valid configuration file found. Please check the path to the configuration file, which should be config/experiments.cfg"
        )


    pg.setConfigOption("antialias", True)

    BR = Bridge(experiments=experiments, args=args)
    if (sys.flags.interactive != 1) or not hasattr(QtCore, "PYQT_VERSION"):
        QtWidgets.QApplication.instance().exec()


if __name__ == "__main__":
    main_gui()
