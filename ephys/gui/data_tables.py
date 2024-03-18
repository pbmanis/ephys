"""
This program provides a graphical interface to help access ephys data
from datasummary and analysis results from ephys.
 The display appears as 3 panels: One on the
left with controls, one on the top right that is tabbed, showing either the
current table, or a pdf file of the plots, and one on the bottom for text output
and error messages.

The left panel provides a set of organized controls:

    Create New Experiment DataSet: Creates a new experiment dataset. This looks in the
    selected dataset directory,
    and lets you select the experiment name, set the paths to the data and the
    analysis directories, and then creates the directory structure, and writes a python file
    with the experiment information. You may need to edit this file further by hand to customize
    it for your experiments. 

    Choose Experiment: Select the experiment to work on. This sets the experiment
    name used for subsequent analyses. 

    Update DataSummary: This updates the datasummary file for a particular experiment.
        It is not necessary to do this unless you have added new data to the experiment. 
        DataSummary files cannot be edited.
        If there is an existing DataSummary file for this
    experiment, it is renamed to have the date and time appended to the name, and then 
    a new clean DataSummary file is created.
    

    Analysis:

        This provides different fixed kinds of analysis for the model data.
        Traces: just plot the traces, stacked, for reference. IV : plot
        current-voltage relationships and calculate Rin, Taum, find spikes. VC :
        plot current voltage relationships in voltage clamp. Singles: For the
        "single" AN protocol, where only one input at a time is active, creates
        stacked plot Trace Viewer : dynamic plot of APs and preceding times for
        AN inputs in the "Traces" tab RevcorrSPKS : reverse correlation against
        postsynaptic spikes for each input. Using brian package RevcorrEleph :
        reverse correlation using the elephant pacakge. RevcorrSimple : Simple
        reverse correlation calculation. RevcorrSTTC : not implemented. PSTH :
        Plot PSTH, raster, for bu cell and AN input; also compute phase locking
        to AM if needed.

    Filters:
        This provides data selection in the table. Most entries provide a
        drop-down list. The values that are not  None are applied with "and" 
        logic. The filters are not applied until the Apply button is pressed.
        The filters are cleared by the Clear button.


    Figures:
        Interface to figure generation. Figures are generated from the model
        data directly as much as possible. Some figures are generated from
        analysis data that is either compiled manually, or using a script.
        Also includes 3 analysis hooks (for the SAM tone data)

    Tools:
        Reload: for all modules under data_tables, reload the code. Mostly used
        during development. View IndexFile: Print the index file in the text
        window. Print File Info: Prints the file info for the selected entries
        into the text window. Delete Selected Sim : for deleting simulations
        that are broken (or stopped early). 

    Quit:
        Exit the program.

Uses pyqtgraph tablewidget to build a table showing simulation files/runs and
enabling analysis via a GUI


This module is derived from vcnmodel.

Support::

    NIH grants:
    DC R01 DC015901 (Spirou, Manis, Ellisman),
    DC R01 DC004551 (Manis, 2013-2019, Early development)
    DC R01 DC019053 (Manis, 2020-2025, Later development)

Copyright 2019-2023 Paul B. Manis
Distributed under MIT/X11 license. See license.txt for more infomation. 
"""

import datetime
import functools
import importlib
import pprint
import sys
from pathlib import Path

import pandas as pd
import pyqtgraph as pg
import pyqtgraph.dockarea as PGD
from pylibrary.plotting import plothelpers as PH
from pylibrary.tools import cprint as CP
import PyQt6.QtWebEngineWidgets
from PyQt6.QtWebEngineWidgets import QWebEngineView
from pyqtgraph.parametertree import Parameter, ParameterTree
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets

import ephys
from ephys.gui import data_summary_table
from ephys.gui import data_table_functions as functions
from ephys.gui import data_table_manager as table_manager
from ephys.gui import table_tools
from ephys.plotters import plot_spike_info
from ephys.tools import process_spike_analysis
from ephys.tools import data_summary
from ephys.tools.get_computer import get_computer
from ephys.tools.get_configuration import get_configuration
import ephys.tools.configuration_manager as configuration_manager

from ephys.ephys_analysis import (
    analysis_common,
    iv_analysis,
    map_analysis,
    summarize_ivs,
)

PSI_2 = plot_spike_info  # reference to non-class routines in the module.
PSI = plot_spike_info.PlotSpikeInfo(dataset=None, experiment=None)
PSA = process_spike_analysis.ProcessSpikeAnalysis(dataset=None, experiment=None)

FUNCS = functions.Functions()  # get the functions class
cprint = CP.cprint

# List reloadable modules
all_modules = [
    table_manager,
    data_summary_table,
    plot_spike_info,
    process_spike_analysis,
    functions,
    ephys.ephys_analysis.spike_analysis,
    ephys.ephys_analysis.rm_tau_analysis,
    ephys.tools.utilities,
    ephys.ephys_analysis.make_clamps,
    PH,
]


runtypes = [
    "IV",
    "Maps",
    "VC",
]


def make_rundates():
    """
    Make a list of run dates
    """
    rundates = ["None"]
    for y in ["2017", "2018", "2019", "2020", "2021", "2022", "2023", "2024", "2025"]:
        for m in [
            "01",
            "02",
            "03",
            "04",
            "05",
            "06",
            "07",
            "08",
            "09",
            "10",
            "11",
            "12",
        ]:
            rundates.append(f"{y}-{m}-01")
    return rundates


Age_Values = [  # this is just for selecting age ranges in the GUI
    "None",
    [10, 14],
    [15, 19],
    [0, 21],
    [21, 28],
    [28, 60],
    [60, 90],
    [90, 182],
    [28, 365],
    [365, 900],
]

RMP_Values = [
    [-80, -50],
    [-70, -50],
]

taum_Values = [0.0005, 0.05]

experimenttypes = ["CCIV", "VC", "Map", "Minis", "PSC"]

run_dates = make_rundates()


class TableModel(QtGui.QStandardItemModel):
    _sort_order = QtCore.Qt.SortOrder.AscendingOrder

    def sortOrder(self):
        return self._sort_order

    def sort(self, column, order):
        if column == 0:
            self._sort_order = order
            QtGui.QStandardItemModel.sort(self, column, order)


tdir = Path.cwd()
sys.path.append(str(Path(tdir, "src")))  # add to sys path in order to fix imports.
sys.path.append(str(Path(tdir, "src/util")))  # add to sys path in order to fix imports.


class DataTables:
    """
    Main entry point for building the table and operating on the data
    """

    def __init__(self, datasets=None, experiments=None):
        self.QColor = QtGui.QColor  # for access in plotsims (pass so we can reload)
        self.git_hash = (
            functions.get_git_hashes()
        )  # get the hash for the current versions of ephys and our project
        self.datasummary = None
        self.experimentname = None
        self.datasets = datasets
        self.experiments = experiments
        self.assembleddata = None
        self.doing_reload = False
        self.picker_active = False
        self.show_pdf_on_pick = False
        self.dry_run = False
        self.exclude_unimportant = False

        self.PSI = plot_spike_info.PlotSpikeInfo(dataset=None, experiment=None, pick_display=self.show_pdf_on_pick,
                                                 pick_display_function=self.display_from_table_by_cell_id)
        # self.FIGS = figures.Figures(parent=self)
        self.ptreedata = None  # flag this as not set up initially
        self.table_manager = None  # get this later
        ptreewidth = 350
        self.app = pg.mkQApp()
        self.app.setStyle("fusion")
        self.infobox_x = 0.02
        self.infobox_y = 0.94
        self.infobox_fontsize = 5.5

        # Define the table style for various parts dark scheme
        dark_palette = QtGui.QPalette()
        white = self.QColor(255, 255, 255)
        black = self.QColor(0, 0, 0)
        red = self.QColor(255, 0, 0)
        dark_palette.setColor(QtGui.QPalette.ColorRole.Window, self.QColor(53, 53, 53))
        dark_palette.setColor(QtGui.QPalette.ColorRole.WindowText, white)
        dark_palette.setColor(QtGui.QPalette.ColorRole.Base, self.QColor(25, 25, 25))
        dark_palette.setColor(QtGui.QPalette.ColorRole.AlternateBase, self.QColor(53, 53, 53))
        dark_palette.setColor(QtGui.QPalette.ColorRole.ToolTipBase, white)
        dark_palette.setColor(QtGui.QPalette.ColorRole.ToolTipText, white)
        dark_palette.setColor(QtGui.QPalette.ColorRole.Text, white)
        dark_palette.setColor(QtGui.QPalette.ColorRole.Button, self.QColor(53, 53, 53))
        dark_palette.setColor(QtGui.QPalette.ColorRole.ButtonText, white)
        dark_palette.setColor(QtGui.QPalette.ColorRole.BrightText, red)
        dark_palette.setColor(QtGui.QPalette.ColorRole.Link, self.QColor(42, 130, 218))
        dark_palette.setColor(QtGui.QPalette.ColorRole.Highlight, self.QColor(42, 130, 218))
        dark_palette.setColor(QtGui.QPalette.ColorRole.HighlightedText, self.QColor(0, 255, 0))

        self.app.setPalette(dark_palette)
        self.app.setStyleSheet(
            "QToolTip { color: #ffffff; background-color: #2a82da; border: 1px solid white; }"
        )

        self.win = pg.QtWidgets.QMainWindow()
        # use dock system instead of layout.
        self.dockArea = PGD.DockArea()
        self.win.setCentralWidget(self.dockArea)
        self.win.setWindowTitle("DataTables")
        win_width = 1600
        right_docks_width = 1600 - ptreewidth - 20
        right_docks_height = 800
        self.win.resize(1600, 1024)
        # Initial Dock Arrangment
        self.Dock_Params = PGD.Dock("Params", size=(ptreewidth, 1024))
        self.Dock_Table = PGD.Dock("Dataset Table", size=(right_docks_width, right_docks_height))
        self.Dock_Traces = PGD.Dock("Traces", size=(right_docks_width, right_docks_height))
        self.Dock_DataSummary = PGD.Dock(
            "DataSummary", size=(right_docks_width, right_docks_height)
        )
        self.Dock_PDFView = PGD.Dock("PDFs", size=(right_docks_width, right_docks_height))
        self.Dock_Report = PGD.Dock("Reporting", size=(right_docks_width, right_docks_height))

        self.dockArea.addDock(self.Dock_Params, "left")
        self.dockArea.addDock(self.Dock_Table, "right", self.Dock_Params)
        self.dockArea.addDock(self.Dock_PDFView, "below", self.Dock_Table)
        self.dockArea.addDock(self.Dock_DataSummary, "below", self.Dock_PDFView)
        self.dockArea.addDock(self.Dock_Traces, "below", self.Dock_DataSummary)
        self.dockArea.addDock(self.Dock_Report, "below", self.Dock_Table)
        # self.dockArea.addDock(self.Dock_Traces_Slider, 'below',
        # self.Dock_Traces)

        # self.Dock_Traces.addContainer(type=pg.QtGui.QGridLayout,
        # obj=self.trace_layout)
        self.table = pg.TableWidget(sortable=True)
        self.Dock_Table.addWidget(self.table)
        self.Dock_Table.raiseDock()

        self.DS_table = pg.TableWidget(sortable=True)
        self.Dock_DataSummary.addWidget(self.DS_table)  # don't raise yet

        self.PDFView = QWebEngineView()
        self.PDFView.settings().setAttribute(
            self.PDFView.settings().WebAttribute.PluginsEnabled, True
        )
        self.PDFView.settings().setAttribute(
            self.PDFView.settings().WebAttribute.PdfViewerEnabled, True
        )
        self.Dock_PDFView.addWidget(self.PDFView)

        self.textbox = QtWidgets.QTextEdit()
        FUNCS.textbox_setup(self.textbox)  # make sure the functions know about the textbox
        self.textbox.setReadOnly(True)
        self.textbox.setTextColor(QtGui.QColor("white"))
        self.textbox.setStyleSheet("background-color: black")
        self.textbox.setFontFamily(
            "Monaco"
        )  # Lucida Console; Menlo, Monaco, San Francisco (some are OSX only)
        self.textbox.setFontPointSize(11)
        self.textbox.clear()
        self.textbox.setText("")
        self.Dock_Report.addWidget(self.textbox)

        style = "::section {background-color: darkblue; }"
        self.selected_index_row = None  # for single selection mode
        self.selected_index_rows = None  # for multirow selection mode
        self.table.horizontalHeader().setStyleSheet(style)
        self.model = None
        # self.table.sortingEnabled(True)
        self.voltage = False
        self.runtype = runtypes[0]
        self.cellID = None
        self.start_date = "None"
        self.end_date = "None"
        self.dataset = self.datasets[0]
        self.set_experiment(self.dataset)
        self.selvals = {
            "DataSets": [self.datasets, self.dataset],
            "Run Type": [runtypes, self.runtype],
            # "Cells": [cellvalues, self.cellID],
            "Start Date": [run_dates, self.start_date],
            "End Date": [run_dates, self.end_date],
            # "Mode": [modetypes, self.modetype], "Experiment":
            # [experimenttypes, self.experimenttype], "Analysis":
            # [analysistypes, self.analysistype], "Dendrites": [dendriteChoices,
            # self.dendriteChoices],
        }
        self.filters = {
            "Use Filter": False,
            "cell_type": None,
            "flag": None,
            "age": None,
            "sex": None,
            "group": None,
            "DataTable": None,
        }
        self.trvalues = [1, 2, 4, 8, 16, 32]
        self.n_trace_sel = self.trvalues[2]
        self.V_disp = ["Vm", "dV/dt"]
        self.V_disp_sel = self.V_disp[0]
        self.movie_state = False
        self.frame_intervals = [0.033, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0, 2.0]
        self.frame_interval = self.frame_intervals[3]
        self.target_figure = (None,)
        self.deselect_flag = False
        self.deselect_threshold = 180.0  # um2
        self.revcorr_window = [-2.7, -0.5]

        # We use pyqtgraph's ParameterTree to set up the menus/buttons. This
        # defines the layout.

        self.params = [
            # {"name": "Pick Cell", "type": "list", "values": cellvalues,
            # "value": cellvalues[0]},
            {"name": "Create New DataSet", "type": "action"},
            {
                "name": "Choose Experiment",
                "type": "list",
                "limits": [ds for ds in self.datasets],
                "value": self.datasets[0],
            },
            {"name": "Reload Configuration", "type": "action"}, # probably not needed...
            {"name": "Update DataSummary", "type": "action"},
            {"name": "Load DataSummary", "type": "action"},
            {"name": "Load Assembled Data", "type": "action"},
            {
                "name": "IV Analysis",
                "type": "group",
                "expanded": False,
                "children": [
                    {"name": "Analyze Selected IVs", "type": "action"},
                    {"name": "Analyze Selected IVs m/Important", "type": "action"},
                    {"name": "Dry run (test)", "type": "bool", "value": False},
                    {"name": "Analyze ALL IVs", "type": "action"},
                    {"name": "Analyze ALL IVs m/Important", "type": "action"},
                    {"name": "Process Spike Data", "type": "action"},
                    {"name": "Assemble IV datasets", "type": "action"},
                    {"name": "Exclude unimportant in assembly", "type": "bool", "value": False},
                ],
            },
            {
                "name": "Plotting",
                "type": "group",
                "children": [
                    {
                        "name": "Group By",
                        "type": "list",
                        "limits": [gr for gr in self.experiment["group_by"]],
                        "value": self.experiment["group_by"][0],
                    },
                    {
                        "name": "2nd Group By",
                        "type": "list",
                        "limits": [gr for gr in self.experiment["secondary_group_by"]],
                        "value": self.experiment["secondary_group_by"][0],
                    },
                    {"name": "View Cell Data", "type": "action"},
                    {"name": "Use Picker", "type": "bool", "value": False},
                    {"name": "Show PDF on Pick", "type": "bool", "value": False},
                    {"name": "Plot Spike Data", "type": "action"},
                    {"name": "Plot Rmtau Data", "type": "action"},
                    {"name": "Plot FIData Data", "type": "action"},
                    {"name": "Plot FICurves", "type": "action"},
                    {"name": "Plot Selected Spike", "type": "action"},
                    {"name": "Plot Selected FI Fitting", "type": "action"},
                    {"name": "Print Stats on IVs and Spikes", "type": "action"},
                ],
            },
            {
                "name": "Filters",
                "type": "group",
                "expanded": False,
                "children": [
                    # {"name": "Use Filter", "type": "bool", "value": False},
                    {
                        "name": "cell_type",
                        "type": "list",
                        "limits": [
                            "None",
                            "bushy",
                            "tstellate",
                            "dstellate",
                            "octopus",
                            "pyramidal",
                            "cartwheel",
                            "golgi",
                            "granule",
                            "stellate",
                            "tuberculoventral",
                            "unclassified",
                        ],
                        "value": "None",
                    },
                    {
                        "name": "age",
                        "type": "list",
                        "limits": Age_Values,
                        "value": "None",
                    },
                    {
                        "name": "sex",
                        "type": "list",
                        "limits": ["None", "M", "F"],
                        "value": "None",
                    },
                    {
                        "name": "group",
                        "type": "list",
                        "limits": ["None", "Control", "NIHL", "A", "AA", "AAA", "B"],
                        "value": "None",
                    },
                    {
                        "name": "RMP",
                        "type": "list",
                        "limits": RMP_Values,
                        "value": 0,
                    },
                    {
                        "name": "taum",
                        "type": "list",
                        "limits": taum_Values,
                        "value": "None",
                    },
                    {
                        "name": "PulseDur",
                        "type": "list",
                        "limits": ["None", 50.0, 100.0, 200.0, 500.0, 1000.0, 2000.0],
                        "value": "None",
                    },
                    {
                        "name": "Protocol",
                        "type": "list",
                        "limits": [
                            "None",
                            "CCIV_+",
                            "CCIV_1nA",
                            "CCIV_200pA",
                            "CCIV_long",
                            "CCIV_long_HK",
                        ],
                        "value": "None",
                    },
                    {
                        "name": "Filter Actions",
                        "type": "group",
                        "children": [
                            {"name": "Apply", "type": "action"},
                            {"name": "Clear", "type": "action"},
                        ],
                    },
                ],
            },
            #  for plotting figures
            #
            {
                "name": "Figures",
                "type": "group",
                "children": [
                    {
                        "name": "Figures",
                        "type": "list",
                        "limits": [
                            "-------NF107_WT_Ctl-------",
                            "Figure1",
                            "Figure2",
                            "Figure3",
                            "EPSC_taurise_Age",
                            "EPSC_taufall_age",
                            "-------NF107_NIHL--------",
                            "Figure-rmtau",
                            "Figure-spikes",
                            "Figure-firing",
                        ],
                        "value": "-------NF107_WT_Ctl-------",
                    },
                    {"name": "Create Figure/Analyze Data", "type": "action"},
                ],
            },
            {
                "name": "Tools",
                "type": "group",
                "expanded": False,
                "children": [
                    {"name": "Reload", "type": "action"},
                    {"name": "View IndexFile", "type": "action"},
                    {"name": "Print File Info", "type": "action"},
                    {"name": "Export Brief Table", "type": "action"},
                ],
            },
            {"name": "Quit", "type": "action"},
        ]
        self.ptree = ParameterTree()
        self.ptreedata = Parameter.create(name="Models", type="group", children=self.params)
        self.ptree.setParameters(self.ptreedata)

        self.ptree.setMaximumWidth(ptreewidth + 50)
        self.ptree.setMinimumWidth(ptreewidth)

        self.Dock_Params.addWidget(self.ptree)  # put the parameter three here

        self.trace_plots = pg.PlotWidget(title="Trace Plots")
        self.Dock_Traces.addWidget(self.trace_plots, rowspan=5, colspan=1)
        self.trace_plots.setXRange(-5.0, 2.5, padding=0.2)
        self.trace_plots.setContentsMargins(10, 10, 10, 10)
        # Build the trace selector
        self.trace_selector = pg.graphicsItems.InfiniteLine.InfiniteLine(
            0, movable=True, markers=[("^", 0), ("v", 1)]
        )
        self.trace_selector.setPen((255, 255, 0, 200))  # should be yellow
        self.trace_selector.setZValue(1)
        self.trace_selector_plot = pg.PlotWidget(title="Trace selector")
        self.trace_selector_plot.hideAxis("left")
        self.frameTicks = pg.graphicsItems.VTickGroup.VTickGroup(yrange=[0.8, 1], pen=0.4)
        self.trace_selector_plot.setXRange(0.0, 10.0, padding=0.2)
        self.trace_selector.setBounds((0, 10))
        self.trace_selector_plot.addItem(self.frameTicks, ignoreBounds=True)
        self.trace_selector_plot.addItem(self.trace_selector)
        self.trace_selector_plot.setMaximumHeight(100)
        self.trace_plots.setContentsMargins(10, 10, 10, 10)

        self.Dock_Traces.addWidget(self.trace_selector_plot, row=5, col=0, rowspan=1, colspan=1)

        self.set_experiment(self.dataset)

        self.win.show()
        self.table.setSelectionMode(pg.QtWidgets.QAbstractItemView.SelectionMode.ExtendedSelection)
        self.table.setSelectionBehavior(QtWidgets.QTableView.SelectionBehavior.SelectRows)
        self.table_manager = table_manager.TableManager(
            parent=self,
            table=self.table,
            experiment=self.experiment,
            selvals=self.selvals,
            altcolormethod=self.altColors,
        )
        self.table.setSelectionMode(pg.QtWidgets.QAbstractItemView.SelectionMode.ExtendedSelection)
        self.table.setSelectionBehavior(QtWidgets.QTableView.SelectionBehavior.SelectRows)
        self.table.itemDoubleClicked.connect(functools.partial(self.on_double_click, self.table))
        self.table.clicked.connect(functools.partial(self.on_single_click, self.table))
        self.ptreedata.sigTreeStateChanged.connect(self.command_dispatcher)
        self.update_assembled_data()

        self.DS_table_manager = data_summary_table.TableManager(
            table=self.DS_table,
            experiment=self.experiment,
            selvals=self.selvals,
            altcolormethod=self.altColors,
        )
        self.DS_table.setSelectionMode(
            pg.QtWidgets.QAbstractItemView.SelectionMode.ExtendedSelection
        )
        self.DS_table.setSelectionBehavior(QtWidgets.QTableView.SelectionBehavior.SelectRows)
        if self.datasummary is not None:
            self.DS_table_manager.build_table(self.datasummary, mode="scan")

        # test PDF viewer:
        # self.PDFView.setUrl(pg.QtCore.QUrl(f"file://{self.datapaths['databasepath']}/NF107Ai32_NIHL/pyramidal/2018_06_20_S04C00_pyramidal_IVs.pdf#zoom=80"))
        # self.PDFView.setZoomFactor(80.0) # broken?
        # Ok, we are in the loop - anything after this is menu-driven and
        # handled either as part of the TableWidget, the Traces widget, or
        # through the CommandDispatcher.

    def on_double_click(self, w):
        """
        Double click gets the selected row and then does an analysis
        """
        index = w.selectionModel().currentIndex()

        modifiers = QtWidgets.QApplication.keyboardModifiers()
        mode="IV"
        if modifiers == QtCore.Qt.KeyboardModifier.ControlModifier:
            mode='map'
        # elif modifiers == QtCore.Qt.ControlModifier:
        #     print('Control+Click')
        # elif modifiers == (QtCore.Qt.ControlModifier |
        #                    QtCore.Qt.ShiftModifier):
        #     print('Control+Shift+Click')
        # handle sorted table: use cell_id to get row key
        i_row = index.row()  # clicked index row

        self.selected_index_row = i_row
        self.display_from_table_by_row(i_row, mode=mode)

    def on_single_click(self, w):
        """
        Single click simply sets the selected rows
        """
        selrows = w.selectionModel().selectedRows()
        self.selected_index_rows = selrows
        if len(selrows) == 0:
            self.selected_index_rows = None
        # for index in selrows: self.selected_index_row = index.row()
        #     self.analyze_from_table(index.row())

    def display_from_table_by_row(self, i_row, mode="IV"):
        self.table_manager.select_row_by_row(i_row)
        self.display_from_table(mode=mode)

    def display_from_table_by_cell_id(self, cell_id):
        """
        Display the selected cell_id from the table
        """
        if not self.show_pdf_on_pick:
            print("show pdf on pick is turned off")
            return
        FUNCS.textappend(f"Displaying cell: {cell_id:s}")

        i_row = self.table_manager.select_row_by_cell_id(cell_id)
        if i_row is not None:
            self.display_from_table(mode="IV")
        else:
            FUNCS.textappend(f"Cell {cell_id:s} not found in table")

    def handleSortIndicatorChanged(self, index, order):
        """
        If the sorting changes, and we are not reloading the modules,
        then go ahead and update the table
        """
        if self.doing_reload:
            return  # don't do anything if we are reloading to avoid big looping
        self.table_manager.update_table(self.table_manager.data, QtCore=QtCore, QtGui=QtGui)
        if index != 0:
            self.table.horizontalHeader().setSortIndicator(0, self.table.model().sortOrder())

    def set_experiment(self, data):
        FUNCS.textclear()
        self.experimentname = data
        self.dataset = data
        self.experiment = self.experiments[data]
        self.win.setWindowTitle(f"DataTables: {self.experimentname!s}")
        if self.ptreedata is not None:
            group = self.ptreedata.child("Plotting").child("Group By")
            group.setLimits(self.experiment["group_by"])  # update the group_by list
            group2 = self.ptreedata.child("Plotting").child("2nd Group By")
            group2.setLimits(self.experiment["secondary_group_by"])  # update the 2nd group_by list
        self.PSI.set_experiment(self.dataset, self.experiment)  # pass around to other functions
        PSA.set_experiment(self.dataset, self.experiment)  # pass around to other functions
        FUNCS.textappend(f"Contents of Experiment Named: {self.experimentname:s}")
        pp = pprint.PrettyPrinter(indent=4)
        FUNCS.textappend(pp.pformat(self.experiment))
        self.load_assembled_data()  # try to load the assembled data

    # def pick_handler(self, event, picker_funcs):
    #     for pf in picker_funcs.keys():  # axis keys
    #         if event.mouseevent.inaxes == pf: # this was our axis
    #             # print("\nDataframe index: ", event.ind)
    #             cell = picker_funcs[pf].data.iloc[event.ind]
    #             cell_id = cell["cell_id"].values[0]
    #             print(f"\nSelected:   {cell_id!s}")  # find the matching data.
    #             age = PSI_2.get_age(cell["age"])
    #             print(
    #                 f"     Cell: {cell['cell_type'].values[0]:s}, Age: P{age:d}D Group: {cell['Group'].values[0]!s}"
    #             )
    #             print(f"     Protocols: {cell['protocols'].values[0]!s}")
    #             if self.show_pdf_on_pick:
    #                 i_row = self.table_manager.select_row_by_cell_id(cell_id)
    #                 if i_row is not None:
    #                     self.display_from_table(i_row)
    #             return cell_id
    #     return None

    def command_dispatcher(self, param, changes):
        """
        Dispatcher for the commands from parametertree path[0] will be the
        command name path[1] will be the parameter (if there is one) path[2]
        will have the subcommand, if there is one data will be the field data
        (if there is any)
        """
        porder = plot_spike_info.get_plot_order(self.experiments[self.dataset])
        colors = plot_spike_info.get_plot_colors(self.experiments[self.dataset])

        for param, change, data in changes:
            path = self.ptreedata.childPath(param)

            match path[0]:
                case "Quit":
                    exit()
                case "Create New DataSet":

                    data = table_tools.TableTools().create_new_dataset()
                    if data is not None and data.experimentname is not None:
                        self.experimentname = data.experimentname
                        self.set_experiment(self.experimentname)

                case "Choose Experiment":
                    self.set_experiment(data)

                case "Reload Configuration":
                    self.datasets, self.experiments = get_configuration()
                    if self.experimentname not in self.datasets:
                        self.experimentname = self.datasets[0]
                    self.experiment = self.experiments[self.experimentname]
                    self.PSI.set_experiment(self.dataset, self.experiment)
                    # print("New configuration: ", self.experimentname)
                    # print(self.experiment)

                case "Update DataSummary":
                    # FUNCS.textappend(f"Updating DataSummary NOT IMPLEMENTED", color="r")
                    self.create_data_summary()

                case "Load DataSummary":
                    self.load_data_summary()
                    self.Dock_DataSummary.raiseDock()

                case "Load Assembled Data":
                    self.load_assembled_data()

                case "IV Analysis":
                    match path[1]:
                        case "Dry run (test)":
                            self.dry_run = data
                        
                        case "Exclude unimportant in assembly":
                            self.exclude_unimportant = data

                        case "Analyze ALL IVs":
                            self.analyze_ivs("all")

                        case "Analyze ALL IVs m/Important":
                            self.analyze_ivs("important")

                        case "Analyze Selected IVs":  # work from the *datasummary* table.
                            index_rows = FUNCS.get_multiple_row_selection(self.DS_table_manager)
                            if index_rows is None:
                                return

                            FUNCS.textappend(
                                f"Analyze IVs from selected cell(s) at rows: {len(index_rows)!s}"
                            )
                            self.Dock_Report.raiseDock()  # so we can monitor progress
                            for index_row in index_rows:
                                selected = self.DS_table_manager.get_table_data(index_row)
                                FUNCS.textappend(f"    Selected: {selected!s}")
                                pathparts = Path(selected.cell_id).parts
                                day = pathparts[0]
                                slicecell = f"S{pathparts[1][-1]:s}C{pathparts[2][-1:]:s}"
                                FUNCS.textappend(f"    Day: {day:s}  slice_cell: {slicecell:s}")
                                self.analyze_ivs(mode="selected", day=day, slicecell=slicecell)
                                self.iv_finished_message()
                            self.Dock_DataSummary.raiseDock()  # back to the original one

                        case "Analyze Selected IVs m/Important":  # work from the *datasummary* table.
                            index_rows = FUNCS.get_multiple_row_selection(self.DS_table_manager)
                            if index_rows is None:
                                return

                            FUNCS.textappend(
                                f"Analyze IVs from selected cell(s) at rows: {len(index_rows)!s}"
                            )
                            self.Dock_Report.raiseDock()  # so we can monitor progress
                            for index_row in index_rows:
                                selected = self.DS_table_manager.get_table_data(index_row)
                                FUNCS.textappend(f"    Selected: {selected!s}")
                                pathparts = Path(selected.cell_id).parts
                                day = pathparts[0]
                                slicecell = f"S{pathparts[1][-1]:s}C{pathparts[2][-1:]:s}"
                                FUNCS.textappend(f"    Day: {day:s}  slice_cell: {slicecell:s}")
                                self.analyze_ivs(mode="selected", important=True, day=day, slicecell=slicecell)
                                self.iv_finished_message()
                            self.Dock_DataSummary.raiseDock()  # back to the original one

                        case "Assemble IV datasets":
                            (
                                excelsheet,
                                analysis_cell_types,
                                adddata,
                            ) = plot_spike_info.setup(self.experiment)
                            print("adddata: ", adddata)
                            print(self.experiment["coding_file"])
                            if self.experiment["coding_file"] is not None:
                                coding_file = Path(self.experiment["analyzeddatapath"], 
                                            self.experiment["directory"], 
                                            self.experiment["coding_file"])
                            else:
                                coding_file = None
                            fn = self.PSI.get_assembled_filename(self.experiment)
                            print("adddata: ", adddata)
                            self.PSI.assemble_datasets(
                                excelsheet=excelsheet,
                                adddata=adddata,
                                coding_file=coding_file,
                                coding_sheet=self.experiment["coding_sheet"],
                                coding_level=self.experiment["coding_level"],
                                exclude_unimportant=self.exclude_unimportant,
                                fn=fn,
                            )

                        case "Process Spike Data":
                            PSA.process_spikes()

                case "Plotting":
                    match path[1]:
                        case "View Cell Data":
                            FUNCS.get_row_selection(self.table_manager)
                            if self.selected_index_rows is not None:
                                index_row = self.selected_index_rows[0]
                                selected = self.table_manager.get_table_data(index_row)
                                print("selected: ", selected)
                                day = selected.date[:-4]
                                slicecell = selected.cell_id[-4:]
                                cell_df, _ = FUNCS.get_cell(
                                    self.experiment, self.assembleddata, cell=selected.cell_id
                                )
                                pp = pprint.PrettyPrinter(indent=4)
                                pp.pprint(cell_df.__dict__)
                                pp.pprint(cell_df["IV"].keys())
                                pp.pprint(cell_df["Spikes"].keys())
                                for prot in cell_df["IV"].keys():
                                    pp.pprint(f"Protocol: {prot:s}")
                                    pp.pprint(
                                        f"   prot IV data keys:  {cell_df['IV'][prot].keys()!s}"
                                    )
                                    pp.pprint(
                                        f"   prot Spike data keys: {cell_df['Spikes'][prot].keys()!s}"
                                    )
                                    print(
                                        f"   CC Comp: , {1e-6*cell_df['IV'][prot]['CCComp']['CCBridgeResistance']:7.3f}  MOhm"
                                    )
                                    print(
                                        f"   CC Comp: , {1e12*cell_df['IV'][prot]['CCComp']['CCNeutralizationCap']:7.3f}  pF"
                                    )

                        case "Use Picker":
                            self.picker_active = data
                            print("Setting enable_picking to: ", self.picker_active)

                        case "Show PDF on Pick":
                            self.show_pdf_on_pick = data

                        case "Plot Spike Data":
                            fn = self.PSI.get_assembled_filename(self.experiment)
                            group_by = self.ptreedata.child("Plotting").child("Group By").value()
                            hue_category = (
                                self.ptreedata.child("Plotting").child("2nd Group By").value()
                            )
                            if hue_category == "None":
                                hue_category = None
                            df = self.PSI.preload(fn)
                            (
                                P1,
                                picker_funcs1,
                            ) = self.PSI.summary_plot_spike_parameters_categorical(
                                df,
                                xname=group_by,
                                hue_category=hue_category,
                                plot_order=self.experiment["plot_order"][group_by],
                                measures=self.experiment["spike_measures"],
                                colors=colors,
                                enable_picking=self.picker_active,
                            )
                            P1.figure_handle.suptitle("Spike Shape", fontweight="bold", fontsize=18)
                            picked_cellid = P1.figure_handle.canvas.mpl_connect(  # override the one in plot_spike_info
                                "pick_event",
                                lambda event: PSI.pick_handler(event, picker_funcs1),
                            )
                            header = self.get_analysis_info(fn)
                            P1.figure_handle.text(
                                self.infobox_x,
                                self.infobox_y,
                                header,
                                fontdict={
                                    "fontsize": self.infobox_fontsize,
                                    "fontstyle": "normal",
                                    "font": "Courier",
                                },
                            )
                            P1.figure_handle.show()

                        case "Plot Rmtau Data":
                            fn = self.PSI.get_assembled_filename(self.experiment)
                            print("RmTau Plotting: \n    Loading fn: ", fn)
                            group_by = self.ptreedata.child("Plotting").child("Group By").value()
                            print("    group_by: ", group_by)
                            hue_category = (
                                self.ptreedata.child("Plotting").child("2nd Group By").value()
                            )
                            if hue_category == "None":
                                hue_category = None
                                # hue_category = self.experiment["plot_order"][group_by]
                            plot_order = self.experiment["plot_order"][group_by]

                            print("    plot order: ", plot_order)
                            # if not isinstance(group_by, list):
                            #     group_by = [group_by]
                            df = self.PSI.preload(fn)
                            (
                                P3,
                                picker_funcs3,
                            ) = self.PSI.summary_plot_rm_tau_categorical(
                                df,
                                xname=group_by,
                                hue_category=hue_category,
                                plot_order=plot_order,
                                colors=colors,
                                enable_picking=self.picker_active,
                                measures=self.experiment["rmtau_measures"],
                            )
                            P3.figure_handle.suptitle(
                                "Membrane Properties", fontweight="bold", fontsize=18
                            )
                            picked_cellid = P3.figure_handle.canvas.mpl_connect(  # override the one in plot_spike_info
                                "pick_event",
                                lambda event: PSI.pick_handler(event, picker_funcs3),
                            )
                            header = self.get_analysis_info(fn)
                            P3.figure_handle.text(
                                self.infobox_x,
                                self.infobox_y,
                                header,
                                fontdict={
                                    "fontsize": self.infobox_fontsize,
                                    "fontstyle": "normal",
                                    "font": "helvetica",
                                },
                            )
                            P3.figure_handle.show()

                        case "Plot FIData Data":
                            fn = self.PSI.get_assembled_filename(self.experiment)
                            print("Loading fn: ", fn)
                            group_by = self.ptreedata.child("Plotting").child("Group By").value()
                            hue_category = (
                                self.ptreedata.child("Plotting").child("2nd Group By").value()
                            )
                            if hue_category == "None":
                                hue_category = None
                            plot_order = self.experiment["plot_order"][group_by]
                            df = self.PSI.preload(fn)
                            (
                                P2,
                                picker_funcs2,
                            ) = self.PSI.summary_plot_spike_parameters_categorical(
                                df,
                                xname=group_by,
                                hue_category=hue_category,
                                measures=self.experiment["FI_measures"],
                                plot_order=plot_order,
                                colors=self.experiment["plot_colors"],
                                enable_picking=self.picker_active,
                            )
                            P2.figure_handle.suptitle("Firing Rate", fontweight="bold", fontsize=18)
                            picked_cellid = P2.figure_handle.canvas.mpl_connect(  # override the one in plot_spike_info
                                "pick_event",
                                lambda event: PSI.pick_handler(event, picker_funcs2),
                            )
                            header = self.get_analysis_info(fn)
                            P2.figure_handle.text(
                                self.infobox_x,
                                self.infobox_y,
                                header,
                                fontdict={
                                    "fontsize": self.infobox_fontsize,
                                    "fontstyle": "normal",
                                    "font": "helvetica",
                                },
                            )
                            P2.figure_handle.show()

                        case "Plot FICurves":
                            fn = self.PSI.get_assembled_filename(self.experiment)
                            print("Loading fn: ", fn)
                            df = self.PSI.preload(fn)
                            P4, picker_funcs4 = self.PSI.summary_plot_fi(
                                df,
                                mode=["mean"],  # could be ["individual", "mean"]
                                group_by=self.ptreedata.child("Plotting").child("Group By").value(),
                                # protosel=[
                                #     "CCIV_1nA_max",
                                #     # "CCIV_4nA_max",
                                #     "CCIV_long",
                                #     "CCIV_short",
                                #     "CCIV_1nA_Posonly",
                                #     # "CCIV_4nA_max_1s_pulse_posonly",
                                #     "CCIV_1nA_max_1s_pulse",
                                #     # "CCIV_4nA_max_1s_pulse",
                                # ],
                                colors=self.experiment["plot_colors"],
                                enable_picking=self.picker_active,
                            )
                            picked_cellid = P4.figure_handle.canvas.mpl_connect(  # override the one in plot_spike_info
                                "pick_event",
                                lambda event: PSI.pick_handler(event, picker_funcs4),
                            )
                            header = self.get_analysis_info(fn)
                            P4.figure_handle.text(
                                self.infobox_x,
                                self.infobox_y,
                                header,
                                fontdict={
                                    "fontsize": self.infobox_fontsize,
                                    "fontstyle": "normal",
                                    "font": "helvetica",
                                },
                            )
                            P4.figure_handle.show()

                        case "Plot Selected Spike":
                            if self.assembleddata is None:
                                raise ValueError("Must load assembled data file first")
                            FUNCS.get_selected_cell_data_spikes(
                                self.experiment, self.table_manager, self.assembleddata
                            )

                        case "Plot Selected FI Fitting":
                            if self.assembleddata is None:
                                raise ValueError("Must load assembled data file first")
                            FUNCS.get_selected_cell_data_FI(
                                self.experiment, self.table_manager, self.assembleddata
                            )

                        case "Print Stats on IVs and Spikes":
                            (
                                excelsheet,
                                analysis_cell_types,
                                adddata,
                            ) = PSI_2.setup(self.experiment)
                            fn = self.PSI.get_assembled_filename(self.experiment)
                            print("Loading fn: ", fn)
                            df = self.PSI.preload(fn)
                            divider = "=" * 80
                            group_by = self.ptreedata.child("Plotting").child("Group By").value()
                            second_group_by = (
                                self.ptreedata.child("Plotting").child("2nd Group By").value()
                            )
                            FUNCS.textclear()
                            FUNCS.textappend(f"Stats on IVs and Spikes for {fn!s}")
                            self.PSI.do_stats(
                                df,
                                experiment=self.experiment,
                                group_by=group_by,
                                second_group_by=second_group_by,
                                divider=divider,
                                textbox=self.textbox,
                            )
                            stats_text = self.textbox.toPlainText()
                            datetime_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            header = f"Stats on IVs and Spikes for file:\n    {fn!s}\n"
                            header += f"File date: {datetime.datetime.fromtimestamp(fn.stat().st_mtime)!s}\n"
                            header += f"Statistics Run date: {datetime_str:s}\n"
                            header += f"Primary Group By: {group_by:s}      Secondary Group by: {second_group_by!s}\n"
                            header += f"Experiment Name: {self.experimentname:s}\n"
                            header += f"Project git hash: {self.git_hash['project']!s}\nephys git hash: {self.git_hash['ephys']!s}\n"
                            header += f"{'='*80:s}\n\n"
                            stats_text = header + stats_text
                            stats_filename = f"{self.experiment['stats_filename']:s}_G1_{group_by:s}_G2_{second_group_by:s}.txt"
                            stats_path = Path(
                                self.experiment["analyzeddatapath"],
                                self.experiment["directory"],
                                stats_filename,
                            )
                            with open(stats_path, "w") as f:
                                f.write(stats_text)
                            print(f"Stats written to: {stats_path!s}")

                case "Selections":
                    self.selvals[path[1]][1] = str(data)
                    self.cellID = self.selvals["Cells"][1]
                    # self.setPaths("AN", cell=data)
                    self.start_date = self.selvals["Start Date"][1]
                    self.end_date = self.selvals["End Date"][1]
                    if self.assembleddata is not None:
                        self.table_manager.build_table(
                            self.assembleddata, mode="scan", QtCore=QtCore, QtGui=QtGui
                        )

                case "Analysis":
                    analyze_map = {
                        "Maps": self.analyze_Maps,
                        "IV": self.analyze_IV,
                        "Trace Viewer": self.trace_viewer,
                    }
                    analyze_map[path[1]](path[1])  # call the analysis function
                case "Parameters":
                    pass

                case "Filters":
                    self.filters[path[1]] = "None"
                    match path[1]:
                        case "Use Filter":  # currently not an option
                            # print(data)
                            self.filters["Use Filter"] = data
                        case "cell_type" | "age" | "sex" | "group":
                            if data != None:
                                self.filters[path[1]] = data

                        case "Filter Actions":
                            if path[2] in ["Apply"]:
                                self.filters["Use Filter"] = True
                                self.table_manager.apply_filter(QtCore=QtCore, QtGui=QtGui)
                            elif path[2] in ["Clear"]:
                                self.filters["Use Filter"] = False
                                self.table_manager.apply_filter(QtCore=QtCore, QtGui=QtGui)
                        case _:
                            print("Fell thorugh with : ", path[1])
                            if data != "None":
                                self.filters[path[1]] = float(data)

                case "Options":
                    pass

                case "Figures":
                    if path[1] == "Figures":
                        self.target_figure = data
                    elif path[1] == "Create Figure/Analyze Data":
                        self.FIGS.make_figure(self.target_figure)

                case "Tools":
                    match path[1]:
                        case "Reload":
                            print("reloading...")
                            # get the current list selection - first put tabke in the
                            # same order we will see later
                            self.doing_reload = True
                            self.table.sortByColumn(
                                0, QtCore.Qt.SortOrder.AscendingOrder
                            )  # by date
                            selected_rows = self.table.selectionModel().selectedRows()
                            selection_model = self.table.selectionModel()
                            for module in all_modules:
                                print("reloading: ", module)
                                importlib.reload(module)
                            # self.PLT = plot_sims.PlotSims(parent=self)
                            self.table_manager = table_manager.TableManager(
                                parent=self,
                                table=self.table,
                                experiment=self.experiment,
                                selvals=self.selvals,
                                altcolormethod=self.altColors,
                            )

                            print("   reload ok")
                            print("-" * 80)

                            if self.assembleddata is not None:
                                self.table_manager.build_table(
                                    self.assembleddata,
                                    mode="scan",
                                    QtCore=QtCore,
                                    QtGui=QtGui,
                                )
                                self.table.setSortingEnabled(True)
                                self.table.horizontalHeader().sortIndicatorChanged.connect(
                                    self.handleSortIndicatorChanged
                                )
                                self.table_manager.apply_filter(QtCore=QtCore, QtGui=QtGui)
                                self.table.sortByColumn(
                                    0, QtCore.Qt.SortOrder.AscendingOrder
                                )  # by date
                                self.altColors(self.table)  # reset the color list.
                                # now reapply the original selection
                                mode = (
                                    QtCore.QItemSelectionModel.SelectionFlag.Select
                                    | QtCore.QItemSelectionModel.SelectionFlag.Rows
                                )
                                for row in selected_rows:
                                    selection_model.select(row, mode)  # for row in selected_rows:
                            try:
                                self.table_manager.update_table(
                                    self.table_manager.data, QtCore=QtCore, QtGui=QtGui
                                )
                            except AttributeError:
                                pass
                            self.Dock_Table.raiseDock()

                            FUNCS.textappend("Reload OK", color="g")
                            self.doing_reload = False

                        case "View IndexFile":
                            selected = self.table.selectionModel().selectedRows()
                            if selected is None:
                                return
                            index_row = selected[0]
                            self.table_manager.print_indexfile(index_row)
                        case "Print File Info":
                            selected = self.table.selectionModel().selectedRows()
                            if selected is None:
                                return
                            self.print_file_info(selected)
                        case "Export Brief Table":
                            self.table_manager.export_brief_table(self.textbox)

    def analyze_ivs(self, mode="all", important:bool=False, day: str = None, slicecell: str = None):
        """
        Analyze the IVs for the selected cell
        Parameters
        mode can be: "all", "selected", "important"
        If all, all protocols are analyzed.
        if selected, only the selected protocols are analyzed for the day/slicecell
        if important, only the "important" protocols are analyzed for the day/slicecell

        important: bool 
            if True, we pay attention to the "Important" flag in the data.
            Otherwise, we just analyze everything.

        """
        args = analysis_common.cmdargs  # get from default class
        args.dry_run = self.dry_run
        args.merge_flag = True
        args.experiment = self.experiment
        args.iv_flag = True
        args.map_flag = False
        args.autoout = True
        computer_name = get_computer()
        nworkers = self.experiment["NWORKERS"][computer_name]
        args.nworkers = 1
        args.noparallel = True
        # if nworkers > 1:
        #     args.noparallel = False
        #     args.nworkers = nworkers
        # print("data_table: nworkers, noparallel ", args.nworkers, args.noparallel)
        if mode == "important" or important:
            args.important_flag_check = True # only analyze the "important" ones

        args.verbose = False
        args.spike_threshold = self.experiment['AP_threshold_V']  # always in Volts
        if mode == "selected":
            args.day = day
            args.slicecell = slicecell

        # args.after= "2021.07.01"

        if args.configfile is not None:
            config = None
            if args.configfile is not None:
                if ".json" in args.configfile:
                    # The escaping of "\t" in the config file is necesarry as
                    # otherwise Python will try to treat is as the string escape
                    # sequence for ASCII Horizontal Tab when it encounters it
                    # during json.load
                    config = json.load(open(args.configfile))
                elif ".toml" in args.configfile:
                    config = toml.load(open(args.configfile))

            vargs = vars(args)  # reach into the dict to change values in namespace
            for c in config:
                if c in args:
                    # print("c: ", c)
                    vargs[c] = config[c]
        CP.cprint(
            "g",
            f"Starting IV analysis at: {datetime.datetime.now().strftime('%m/%d/%Y, %H:%M:%S'):s}",
        )
        print("\n" * 3)
        CP.cprint("r", "=" * 80)
        IV = iv_analysis.IVAnalysis(args)
        IV.set_experiment(self.experiment)
        # IV.set_exclusions(exclusions)
        IV.setup()
        IV.run()

        if self.dry_run:
            CP.cprint(
                "cyan",
                f"Finished DRY RUN of IV analysis at: {datetime.datetime.now().strftime('%m/%d/%Y, %H:%M:%S'):s}",
            )
        else:
            CP.cprint(
                "cyan",
                f"Finished IV analysis at: {datetime.datetime.now().strftime('%m/%d/%Y, %H:%M:%S'):s}",
                )
            if mode == "all":
                self.iv_finished_message()

    def iv_finished_message(self):
        CP.cprint("r", "=" * 80)
        CP.cprint("r", f"Now run 'process_spike_analysis' to generate the summary data file")
        CP.cprint(
            "r", f"Then run 'assemble datasets' to plot summaries and get statistical results"
        )
        CP.cprint(
            "r", f"Then try the plotting functions to plot summaries and get statistical results"
        )
        CP.cprint("r", "=" * 80)

    def get_analysis_info(self, filename):
        group_by = self.ptreedata.child("Plotting").child("Group By").value()
        second_group_by = self.ptreedata.child("Plotting").child("2nd Group By").value()
        datetime_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        header = f"Experiment Name: {self.experimentname:s}\n"
        header += f"File: {filename.name!s} [{datetime.datetime.fromtimestamp(filename.stat().st_mtime)!s}]\n"
        header += f"Plot date: {datetime_str:s}\n"
        header += f"1 Group By: {group_by:s} 2 Group by: {second_group_by!s}\n"
        header += f"Git hashes: Proj={self.git_hash['project'][-9:]!s}\n       ephys={self.git_hash['ephys'][-9:]!s}\n"
        return header

    def error_message(self, text):
        """
        Provide an error message to the lower text box
        """
        self.textbox.clear()
        color = "red"
        self.textbox.setTextColor(self.QColor(color))
        self.textbox.append(text)
        self.textbox.setTextColor(self.QColor("white"))

    def setColortoRow(self, rowIndex, color):
        """
        Set the color of a row
        """
        for j in range(self.table.columnCount()):
            if self.table.item(rowIndex, j) is not None:
                self.table.item(rowIndex, j).setBackground(color)

    def altColors(
        self, table, colors=[QtGui.QColor(0x00, 0x00, 0x00), QtGui.QColor(0x22, 0x22, 0x22)]
    ):
        """
        Paint alternating table rows with different colors

        Parameters
        ----------
        colors : list of 2 elements
            colors[0] is for odd rows (RGB, Hex) colors[1] is for even rows
        """
        for j in range(table.rowCount()):
            if j % 2:
                self.setColortoRow(j, colors[0])
            else:
                self.setColortoRow(j, colors[1])

    def force_suffix(self, filename, suffix=".pkl"):
        """
        Set the file suffix to the selected value (default: .pkl)
        """
        fn = Path(filename)
        if fn.suffix != suffix:
            fn = str(fn)
            fn = fn + suffix
            fn = Path(fn)
        return fn

    # @trace_calls.winprint
    def print_file_info(self, selected, mode="list"):
        if mode not in ["list", "dict"]:
            raise ValueError()
        FUNCS.textappend("For copy into figure_data.py: ")
        if mode == "dict":
            br = "{}"
            FUNCS.textappend(f"{int(self.cellID):d}: {br[0]:s}")
        if mode == "list":
            br = "[]"
            FUNCS.textappend(f"    {int(self.parent.cellID):d}: {br[0]:s}")
        for sel in selected:
            data = self.table_manager.get_table_data(sel)
            fn = Path(data.files[0])
            fnr = str(fn.parts[-2])
            fkey = data.dendriteExpt
            if mode == "dict":
                FUNCS.textappend(f'    "{fkey:s}": "{fnr:s}",')
            if mode == "list":
                FUNCS.textappend(f'        "{fnr:s}",')
        if mode == "dict":
            FUNCS.textappend(f"{br[1]:s},")
        if mode == "list":
            FUNCS.textappend(f"    {br[1]:s},")

    def create_data_summary(self):
        # -o pandas -f NF107_after_2018.04.16 -w --depth all
        configuration_manager.verify_paths(self.experiment)

        ds = data_summary.DataSummary(
            basedir=Path(self.experiment["rawdatapath"], self.experiment['directory']),
            outputFile=Path(
                self.experiment["analyzeddatapath"],
                self.experiment["directory"],
                self.experiment["datasummaryFilename"],
            ),
            subdirs=self.experiment["extra_subdirectories"],
            dryrun=False,
            depth="all",
            verbose=True,
        )
        print("Writing to output, recurively through directories ")
        data_summary.dir_recurse(
            ds, Path(self.experiment["rawdatapath"], self.experiment['directory']), self.experiment["excludeIVs"]
        )
        # after creation, load it
        self.load_data_summary()

    def find_unique_protocols(self):
        """
        find all of the unique protocols in this database

        """
        if self.datasummary is None:
            raise ValueError("Please load the datasummary file first")
        FUNCS.get_datasummary_protocols(self.datasummary )
        


    def load_data_summary(self):
        self.datasummaryfile = Path(
            self.experiment["databasepath"],
            self.experiment["directory"],
            self.experiment["datasummaryFilename"],
        )
        if not self.datasummaryfile.is_file():
            FUNCS.textappend(
                f"DataSummary file: {self.datasummaryfile!s} does not yet exist - please generate it first"
            )
            return
        FUNCS.textappend(
            f"DataSummary file: {self.datasummaryfile!s}  exists. Last updated: {self.datasummaryfile.stat().st_mtime:f}"
        )
        FUNCS.textappend("Loading ...")
        self.datasummary = pd.read_pickle(self.datasummaryfile)
        if self.datasummary is not None:
            self.DS_table_manager.build_table(self.datasummary, mode="scan")
        FUNCS.textappend(f"DataSummary file loaded with {len(self.datasummary.index):d} entries.")
        # FUNCS.textappend(f"DataSummary columns: \n, {self.datasummary.columns:s}")
        self.find_unique_protocols()

    def load_assembled_data(self):
        """get the current assembled data file, if it exists
        if not, just pass on it.
        """
        self.assembledfile = self.PSI.get_assembled_filename(self.experiment)
        if not self.assembledfile.is_file():
            FUNCS.textappend(
                f"Assembled data file: {self.assembledfile!s} does not yet exist - please generate it first"
            )
            return

        self.assembleddata = pd.read_pickle(self.assembledfile)
        FUNCS.textappend(f"Assembled data loaded, entries: {len(self.assembleddata.index):d}")
        FUNCS.textappend(f"Assembled data columns: {self.assembleddata.columns!s}")

        self.update_assembled_data()

    def update_assembled_data(self):
        if self.table_manager is not None and self.assembleddata is not None:
            self.table_manager.build_table(
                self.assembleddata, mode="scan", QtCore=QtCore, QtGui=QtGui
            )

    # Next we provide dispatches for a few specific actions. These are mostly
    # to routines in plot_sims.py

    def analyze_singles(self, ana_name=None):
        """
        Analyze data this is formatted from the 'singles' runs in model_run2
        These are runs in which each input is evaluated independently
        """
        index_row, selected = FUNCS.get_row_selection(self.table_manager)
        if selected is None:
            return
        self.PLT.analyze_singles(index_row, selected)

    def trace_viewer(self, ana_name=None):
        """
        Invoke the trace viewer (experimental)
        """
        index_row, selected = FUNCS.get_row_selection(self.table_manager)
        if selected is None:
            return
        nfiles = len(selected.files)
        print(" nfiles: ", nfiles)
        print("s elected files: ", selected.files)
        # if nfiles > 1: self.PLT.textappend('Please select only one file to
        #     view') else:
        PD = plot_sims.PData(gradeA=GRPDEF.gradeACells)
        self.PLT.trace_viewer(selected.files[0], PD, selected.runProtocol)

    def analyze_traces(self, ana_name=None):
        """
        Plot traces from the selected run
        """
        index_row, selected = FUNCS.get_row_selection(self.table_manager)
        if selected is None:
            return
        self.plot_traces(
            rows=len(self.selected_index_rows),
            cols=1,
            height=len(self.selected_index_rows),
            width=6.0,
            stack=True,
            ymin=-80.0,
            ymax=20.0,
        )

    def analyze_IV(self, ana_name=None):
        """
        Plot traces from an IV run
        """
        index_row, selected = FUNCS.get_row_selection(self.table_manager)
        if selected is None:
            return
        self.plot_traces(
            rows=1,
            cols=len(self.selected_index_rows),
            height=3.0,
            width=3 * len(self.selected_index_rows),
            stack=False,
            ymin=-120.0,
            ymax=20.0,
        )
        return

    def plot_traces(self, rows=1, cols=1, width=5.0, height=4.0, stack=True, ymin=-120.0, ymax=0.0):
        """
        Plot traces, but do so by redirecting to simple plotting in plotsims
        """
        self.PLT.simple_plot_traces(
            rows=rows,
            cols=cols,
            width=width,
            height=height,
            stack=stack,
            ymin=ymin,
            ymax=ymax,
        )

    def analyze_VC(self, ana_name=None):
        """
        Analyze and plot voltage-clamp runs
        """
        index_row, selected = FUNCS.get_row_selection(self.table_manager)
        if selected is None:
            return
        self.PLT.plot_VC(self.selected_index_rows)

    def display_from_table(self, mode="IV"):
        """
        display the data in the PDF dock
        """
        self.selected_index_rows = self.table.selectionModel().selectedRows()
        if self.selected_index_rows is None:
            return
        # print(len(self.selected_index_rows))
        # print(self.selected_index_rows)
        index_row = self.selected_index_rows[0]
        selected = self.table_manager.get_table_data(index_row)  # table_data[index_row]
        # build the filename for the IVs, format = "2018_06_20_S4C0_pyramidal_IVs.pdf"
        # the first part is based on the selected cell_id from the table, and the rest
        # just makes life easier when looking at the directories.
        cell_type = selected.cell_type
        if cell_type == " " or len(cell_type) == 0:
            cell_type = "unknown"
        sdate = selected.date[:-4]
        cell_id = str(Path(selected.cell_id).name)
        cellname_parts = cell_id.split("_")
        print("mode: ", mode)
        if mode == "IV":
            modename = "IVs"
        elif mode == "map":
            modename = "maps"
        pdfname = (
            f"{cellname_parts[0].replace('.', '_'):s}_{cellname_parts[2]:s}_{cell_type:s}_{modename:s}.pdf"
        )
        datapath = self.experiment["databasepath"]
        direct = self.experiment["directory"]
        filename = f"{Path(datapath, direct, cell_type, pdfname)!s}"
        url = "file://" + filename
        FUNCS.textappend(f"File exists:  {filename!s}, {Path(filename).is_file()!s}")
        print(f"File exists:  {filename!s}, {Path(filename).is_file()!s}")
        self.PDFView.setUrl(pg.QtCore.QUrl(url))  # +"#zoom=80"))
        self.Dock_PDFView.raiseDock()
        if selected is None:
            return


def main():
    # Entry point. Why do I do this ? It keeps sphinxdoc from running the
    # code...
    (
        datasets,
        experiments,
    ) = get_configuration()  # retrieves the configuration file from the running directory
    D = DataTables(datasets, experiments)  # must retain a pointer to the class, else we die!
    if (sys.flags.interactive != 1) or not hasattr(QtCore, "PYQT_VERSION"):
        QtWidgets.QApplication.instance().exec()


if __name__ == "__main__":
    main()
