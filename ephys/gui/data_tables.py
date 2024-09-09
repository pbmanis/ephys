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
import pprint
import sys
from pathlib import Path
import multiprocessing
import textwrap
import concurrent.futures

import pandas as pd
import pyqtgraph as pg
import pyqtgraph.reload as reload
import pyqtgraph.dockarea as PGD
from pylibrary.plotting import plothelpers as PH
from pylibrary.tools import cprint as CP
import PyQt6.QtWebEngineWidgets
from PyQt6.QtWebEngineWidgets import QWebEngineView
from pyqtgraph.parametertree import Parameter, ParameterTree
from pyqtgraph import multiprocess as MP
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets

from ephys.gui import data_summary_table
from ephys.gui import data_table_functions as functions
from ephys.gui import data_table_manager as table_manager
from ephys.gui import table_tools
from ephys.plotters import plot_spike_info as plot_spike_info
from ephys.tools.get_computer import get_computer
from ephys.tools.get_configuration import get_configuration

from ephys.ephys_analysis import (
    analysis_common,
    iv_analysis,
    iv_plotter,
    map_analysis,
)


from ephys.tools import (
    process_spike_analysis,
    data_summary,
    configuration_manager as configuration_manager,
    filename_tools,
)


import sys

# if sys.version_info.major >= 3 and sys.version_info.minor >= 4:
#     multiprocessing.set_start_method('spawn')

config_file_path = "./config/experiments.cfg"

PSI_2 = plot_spike_info  # reference to non-class routines in the module.
PSI = plot_spike_info.PlotSpikeInfo(dataset=None, experiment=None)
PSA = process_spike_analysis.ProcessSpikeAnalysis(dataset=None, experiment=None)

FUNCS = functions.Functions()  # get the functions class
cprint = CP.cprint

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
    [7, 20],
    [21, 49],
    [50, 179],
    [180, 1200],
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
        self.parallel_mode = "cell"
        self.exclude_unimportant = False
        self.computer_name = get_computer()

        self.PSI = plot_spike_info.PlotSpikeInfo(
            dataset=None,
            experiment=None,
            pick_display=self.show_pdf_on_pick,
            pick_display_function=self.display_from_table_by_cell_id,
        )
        self.spike_plot = None
        self.rmtau_plot = None
        self.fidata_plot = None
        self.ficurve_plot = None
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
        self.Dock_DataSummary = PGD.Dock(
            "DataSummary", size=(right_docks_width, right_docks_height)
        )
        self.Dock_IV_Table = PGD.Dock("IV Data Table", size=(right_docks_width, right_docks_height))
        self.Dock_Map_Table = PGD.Dock(
            "Map Data Table", size=(right_docks_width, right_docks_height)
        )
        self.Dock_Minis_Table = PGD.Dock(
            "Mini Data Table", size=(right_docks_width, right_docks_height)
        )
        self.Dock_Traces = PGD.Dock("Traces", size=(right_docks_width, right_docks_height))

        self.Dock_PDFView = PGD.Dock("PDFs", size=(right_docks_width, right_docks_height))
        self.Dock_Report = PGD.Dock("Reporting", size=(right_docks_width, right_docks_height))

        self.dockArea.addDock(self.Dock_Params, "left")
        self.dockArea.addDock(self.Dock_DataSummary, "right", self.Dock_Params)
        self.dockArea.addDock(self.Dock_IV_Table, "below", self.Dock_DataSummary)
        self.dockArea.addDock(self.Dock_Map_Table, "below", self.Dock_IV_Table)
        self.dockArea.addDock(self.Dock_Minis_Table, "below", self.Dock_Map_Table)
        self.dockArea.addDock(self.Dock_Report, "below", self.Dock_Minis_Table)
        self.dockArea.addDock(self.Dock_PDFView, "below", self.Dock_Report)
        self.dockArea.addDock(self.Dock_Traces, "below", self.Dock_PDFView)

        # self.dockArea.addDock(self.Dock_Traces_Slider, 'below',
        # self.Dock_Traces)

        # self.Dock_Traces.addContainer(type=pg.QtGui.QGridLayout,
        # obj=self.trace_layout)
        self.table = pg.TableWidget(sortable=True)
        self.Dock_IV_Table.addWidget(self.table)  # don't raise yet

        self.DS_table = pg.TableWidget(sortable=True)
        self.Dock_DataSummary.addWidget(self.DS_table)
        self.Dock_DataSummary.raiseDock()

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
        self.bspline_s = 1.0
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
            {"name": "Reload Configuration", "type": "action"},  # probably not needed...
            {"name": "Update DataSummary", "type": "action"},
            {"name": "Load DataSummary", "type": "action"},
            {"name": "Load Assembled Data", "type": "action"},
            {"name": "Save Assembled Data", "type": "action"},
            {
                "name": "Parallel Mode",
                "type": "list",
                "limits": ["cell", "day", "trace", "map", "off"],
                "value": "cell",
            },
            {"name": "Dry run (test)", "type": "bool", "value": False},
            {"name": "Only Analyze Important Flagged Data", "type": "bool", "value": False},
            {
                "name": "IV Analysis",
                "type": "group",
                "expanded": False,
                "children": [
                    {"name": "Analyze Selected IVs", "type": "action"},
                    {"name": "Plot from Selected IVs", "type": "action"},
                    {"name": "Analyze ALL IVs", "type": "action"},
                    {"name": "Analyze ALL IVs m/Important", "type": "action"},
                    # {"name": "Process Spike Data", "type": "action"},
                    {"name": "Assemble IV datasets", "type": "action"},
                    {"name": "Exclude unimportant in assembly", "type": "bool", "value": False},
                ],
            },
            {
                "name": "Map Analysis",
                "type": "group",
                "expanded": False,
                "children": [
                    {"name": "Analyze Selected Maps", "type": "action"},
                    {"name": "Analyze ALL Maps", "type": "action"},
                    # {"name": "Assemble Map datasets", "type": "action"},
                    # {"name": "Plot from Selected Maps", "type": "action"},
                ],
            },
            {
                "name": "Mini Analysis",
                "type": "group",
                "expanded": False,
                "children": [
                    {"name": "Analyze Selected Minis", "type": "action"},
                    {"name": "Analyze ALL Minis", "type": "action"},
                    # {"name": "Assemble Mini datasets", "type": "action"},
                    # {"name": "Plot from Selected Minis", "type": "action"},
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
                    {"name": "Plot Spike Data categorical", "type": "action"},
                    {"name": "Plot Spike Data continuous", "type": "action"},
                    {"name": "Plot Rmtau Data categorical", "type": "action"},
                    {"name": "Plot Rmtau Data continuous", "type": "action"},
                    {"name": "Plot FIData Data categorical", "type": "action"},
                    {"name": "Plot FIData Data continuous", "type": "action"},
                    {"name": "Plot FICurves", "type": "action"},
                    {
                        "name": "Set BSpline S",
                        "type": "float",
                        "value": 1.0,
                        "limits": [0.0, 100.0],
                    },
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
                            "t-stellate",
                            "d-stellate",
                            "octopus",
                            "pyramidal",
                            "cartwheel",
                            "giant",
                            "giant_maybe",
                            "golgi",
                            "glial",
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
                        "name": "Group",
                        "type": "list",
                        "limits": ["None", "-/-", "+/+", "+/-"],
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
        self.ptree.setStyleSheet(
            """
            QTreeView {
                background-color: '#282828';
                alternate-background-color: '#646464';   
                color: rgb(238, 238, 238);
            }
            QLabel {
                color: rgb(238, 238, 238);
            }
            QTreeView::item:has-children {
                background-color: '#212627';
                color: '#00d4d4';
            }
            QTreeView::item:selected {
                background-color: '##c1c3ff';
            }
                """
        )
        self.ptree.setParameters(self.ptreedata)

        self.ptree.setMaximumWidth(ptreewidth + 50)
        self.ptree.setMinimumWidth(ptreewidth)

        self.Dock_Params.addWidget(self.ptree)  # put the parameter three here

        # self.trace_plots = pg.PlotWidget(title="Trace Plots")
        # self.Dock_Traces.addWidget(self.trace_plots, rowspan=5, colspan=1)
        # self.trace_plots.setXRange(-5.0, 2.5, padding=0.2)
        # self.trace_plots.setContentsMargins(10, 10, 10, 10)
        # # Build the trace selector
        # self.trace_selector = pg.graphicsItems.InfiniteLine.InfiniteLine(
        #     0, movable=True, markers=[("^", 0), ("v", 1)]
        # )
        # self.trace_selector.setPen((255, 255, 0, 200))  # should be yellow
        # self.trace_selector.setZValue(1)
        # self.trace_selector_plot = pg.PlotWidget(title="Trace selector")
        # self.trace_selector_plot.hideAxis("left")
        # self.frameTicks = pg.graphicsItems.VTickGroup.VTickGroup(yrange=[0.8, 1], pen=0.4)
        # self.trace_selector_plot.setXRange(0.0, 10.0, padding=0.2)
        # self.trace_selector.setBounds((0, 10))
        # self.trace_selector_plot.addItem(self.frameTicks, ignoreBounds=True)
        # self.trace_selector_plot.addItem(self.trace_selector)
        # self.trace_selector_plot.setMaximumHeight(100)
        # self.trace_plots.setContentsMargins(10, 10, 10, 10)

        # self.Dock_Traces.addWidget(self.trace_selector_plot, row=5, col=0, rowspan=1, colspan=1)

        self.set_experiment(self.dataset)

        self.win.show()
        self.table.setSelectionMode(pg.QtWidgets.QAbstractItemView.SelectionMode.ExtendedSelection)
        self.table.setSelectionBehavior(QtWidgets.QTableView.SelectionBehavior.SelectRows)
        self.table_manager = table_manager.TableManager(
            parent=self,
            table=self.table,
            experiment=self.experiment,
            selvals=self.selvals,
            altcolormethod=self.alt_colors,
        )
        self.table.setSelectionMode(pg.QtWidgets.QAbstractItemView.SelectionMode.ExtendedSelection)
        self.table.setSelectionBehavior(QtWidgets.QTableView.SelectionBehavior.SelectRows)
        self.table.itemDoubleClicked.connect(functools.partial(self.on_double_click, self.table))
        self.table.clicked.connect(functools.partial(self.on_single_click, self.table))
        self.ptreedata.sigTreeStateChanged.connect(self.command_dispatcher)
        self.update_assembled_data()

        self.DS_table_manager = data_summary_table.TableManager(
            parent=self,
            table=self.DS_table,
            experiment=self.experiment,
            selvals=self.selvals,
            altcolormethod=self.alt_colors,
        )
        self.DS_table.setSelectionMode(
            pg.QtWidgets.QAbstractItemView.SelectionMode.ExtendedSelection
        )
        self.DS_table.setSelectionBehavior(QtWidgets.QTableView.SelectionBehavior.SelectRows)
        if self.datasummary is None:
            self.load_data_summary()
            # self.Dock_DataSummary.raiseDock()
        self.DS_table.cellDoubleClicked.connect(
            functools.partial(self.DSTable_show_cell_on_double_click, self.DS_table)
        )

        if self.datasummary is not None:
            self.DS_table_manager.build_table(self.datasummary, mode="scan")
        self.table_manager.update_table(self.table_manager.data, QtCore=QtCore, QtGui=QtGui)

        # Ok, we are in the loop - anything after this is menu-driven and
        # handled either as part of the TableWidget, the Traces widget, or
        # through the CommandDispatcher.

    def on_double_click(self, w):
        """
        Double click gets the selected row and then displays the associated PDF (IV or Map)
        """
        index = w.selectionModel().currentIndex()

        modifiers = QtWidgets.QApplication.keyboardModifiers()
        print("Modifiers: ", modifiers)
        mode = "IV"
        if modifiers == QtCore.Qt.KeyboardModifier.MetaModifier:
            mode = "maps"
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
        match mode:
            case "IV" | "IVs" | "iv" | "ivs":
                self.DS_table_manager.select_row_by_row(i_row)
                self.display_from_table(mode=mode)

            case "maps" | "map" | "Map" | "Maps":
                self.DS_table_manager.select_row_by_row(i_row)
                self.display_from_table(mode=mode)
            case _:
                print("Invalid mode")
                raise ValueError(f"Invalid mode for display from table: got {mode}")

    def display_from_table_by_cell_id(self, cell_id, mode: str = "IV", pickable: bool = True):
        """
        Display the selected cell_id from the table
        """
        msg = f"Displaying cell: {cell_id:s}, mode={mode:s}, pickable={pickable!s}"
        print(msg)
        FUNCS.textappend(msg)

        i_row = self.table_manager.select_row_by_cell_id(cell_id)
        if i_row is not None:
            self.display_from_table_by_row(row=i_row, mode=mode)
        else:
            FUNCS.textappend(f"Cell {cell_id:s} not found in table")

    def DSTable_show_pdf_on_control_click(self, w):
        """
        Control click gets the selected row and then displays the associated PDF (IV or Map)
        """
        index = w.selectionModel().currentIndex()
        i_row = index.row()
        self.selected_index_row = i_row
        self.display_from_table_by_row(i_row, mode="maps", pickable=True)

    def DSTable_show_cell_on_double_click(self, w):
        """
        Double click gets the selected row and cell and then displays the full contents
        of the cell in a QMessageBox
        """
        modifier = QtWidgets.QApplication.keyboardModifiers()
        mode = "IV"
        row = w.selectionModel().currentIndex().row()
        col = w.selectionModel().currentIndex().column()
        if modifier == QtCore.Qt.KeyboardModifier.AltModifier:
            mode = "maps"
        elif modifier == QtCore.Qt.KeyboardModifier.ControlModifier:
            mode = "IVs"
        elif modifier == QtCore.Qt.KeyboardModifier.NoModifier:
            msg = pg.QtWidgets.QMessageBox()

            colname = self.DS_table_manager.table.horizontalHeaderItem(col).text()
            infotext = f"{colname:s}:<br><br>{self.DS_table_manager.table.item(row, col).text():s}"
            title = f"<center>{self.DS_table_manager.table.item(row, 0).text()!s}</center>"
            text = "{}<br><br>{}".format(title, "\n".join(textwrap.wrap(infotext, width=120)))
            msg.setText(text)
            msg.setFont(QtGui.QFont("Arial", 11, QtGui.QFont.Weight.Normal))
            # msg.setInformativeText(f"{colname:s} = {self.DS_table_manager.table.item(row, col).text():s}")
            msg.exec()
            return
        else:
            return
        cell_id = self.DS_table_manager.get_table_data(w.selectionModel().currentIndex()).cell_id
        print("cell_id: ", cell_id)
        self.display_from_table_by_row(row, mode=mode)
        return

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
                    self.datasets, self.experiments = get_configuration(config_file_path)
                    if self.datasets is None:
                        print("Unable to get configuration file from: ", config_file_path)
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

                # some general flags that can be used in analysis.
                case "Parallel Mode":
                    self.parallel_mode = data
                    CP.cprint("b", f"Parallel mode set to: {self.parallel_mode!s}")
                case "Dry run (test)":
                    self.dry_run = data

                case "Exclude unimportant in assembly":
                    self.exclude_unimportant = data

                case "IV Analysis":

                    def _do_row(row, mode: str = "selected"):
                        pathparts = Path(row.cell_id).parts
                        slicecell = f"S{pathparts[-2][-1]:s}C{pathparts[-1][-1:]:s}"
                        day = str(Path(*pathparts[0:-2]))

                        FUNCS.textappend(f"    Day: {day:s}  slice_cell: {slicecell:s}")
                        print((f"    Day: {day!s}  slice_cell: {slicecell!s}"))
                        self.analyze_ivs(
                            mode=mode, day=day, slicecell=slicecell, cell_id=row.cell_id
                        )

                    match path[1]:
                        case "Analyze ALL IVs":
                            self.DS_table_manager.dataframe.apply(_do_row, mode="selected", axis=1)

                        case "Analyze ALL IVs m/Important":
                            self.DS_table_manager.dataframe.apply(_do_row, mode="important", axis=1)

                        # analyze the selected cells, working from the *datasummary* table, not the Assembled table.
                        case "Analyze Selected IVs":
                            index_rows = self.DS_table_manager.table.selectionModel().selectedRows() 
                            # FUNCS.get_multiple_row_selection(self.DS_table_manager)
                            FUNCS.textappend(
                                f"Analyze all IVs from selected cell(s) at rows: {len(index_rows)!s}"
                            )
                            # print("Index rows: ", index_rows)
                            for selected_row_number in index_rows:
                                if selected_row_number is None:
                                    print("No selected rows")
                                    break
                                # print("selected row: ", selected_row_number)
                                cell_id = self.DS_table_manager.get_selected_cellid_from_table(selected_row_number)
                                if cell_id is None:
                                    # print("cell id is none?")
                                    continue
                                # print("selected cell: ", cell_id)
                                FUNCS.textappend(cell_id)
                                pathparts = Path(cell_id).parts
                                slicecell = f"S{pathparts[-2][-1]:s}C{pathparts[-1][-1:]:s}"
                                day = str(Path(*pathparts[0:-2]))

                                FUNCS.textappend(f"    Day: {day:s}  slice_cell: {slicecell:s}")
                                print(
                                    (
                                        f"datatables:analyzeselectedIVS:     Day: {day!s}  slice_cell: {slicecell!s}, cellid: {cell_id!s}"
                                    )
                                )
                                if self.dry_run:
                                    print(f"(DRY RUN) Analyzing {cell_id!s} from row: {selected_row_number!s}")
                                else:
                                    self.analyze_ivs(
                                        mode="selected",
                                        day=day,
                                        slicecell=slicecell,
                                        cell_id=cell_id,
                                    )
                            self.iv_finished_message()
                            self.Dock_DataSummary.raiseDock()  # back to the original one

                        case "Plot from Selected IVs":
                            index_rows = self.DS_table_manager.table.selectionModel().selectedRows() 
                            # FUNCS.get_multiple_row_selection(self.DS_table_manager)
                            FUNCS.textappend(
                                f"Analyze all IVs from selected cell(s) at rows: {len(index_rows)!s}"
                            )
                            # print("Index rows: ", index_rows)
                            selections = {"selected": []}
                            for selected_row_number in index_rows:
                                if selected_row_number is None:
                                    print("No selected rows")
                                    break
                                # print("selected row: ", selected_row_number)
                                cell_id = self.DS_table_manager.get_selected_cellid_from_table(selected_row_number)
                                if cell_id is None:
                                    # print("cell id is none?")
                                    continue
                                # print("selected cell: ", cell_id)
                                FUNCS.textappend(cell_id)
                                pathparts = Path(cell_id).parts
                                slicecell = f"S{pathparts[-2][-1]:s}C{pathparts[-1][-1:]:s}"
                                day = str(Path(*pathparts[0:-2]))
                                selections[cell_id] = {"day": day, "slicecell": slicecell}
                                FUNCS.textappend(f"    Day: {day:s}  slice_cell: {slicecell:s}")
                                print(
                                    (
                                        f"datatables:plotselectedIVS:     Day: {day!s}  slice_cell: {slicecell!s}, cellid: {cell_id!s}"
                                    )
                                )
                            if self.parallel_mode == "off":
                                for selection in selections.keys():
                                    if selection is None:
                                        print("Selection was None!")
                                        break
                                    print("Plotting selected cell: ", cell_id)
                                    self.plot_selected(cell_id)

                            else:
                                nworkers = self.experiment["NWORKERS"][self.computer_name]
                                tasks = range(len(index_rows))
                                task_cell = [self.DS_table_manager.get_selected_cellid_from_table(i) for i in index_rows]
                                results = {}
                                result = [None] * len(tasks)
                                with MP.Parallelize(
                                    enumerate(tasks), results=results, workers=nworkers
                                ) as tasker:
                                    for i, x in tasker:
                                        # result= self.plot_selected(index_rows[i+run*chunksize], dspath)
                                        result = self.plot_selected(task_cell[i])
                                        tasker.results[i] = result
                                # reform the results for our database

                            CP.cprint("g", f"\nFinished plotting {len(index_rows):d} IVs")
                            self.Dock_DataSummary.raiseDock()  # back to the original one

                        case (
                            "Analyze Selected IVs m/Important"
                        ):  # work from the *datasummary* table.
                            index_rows = FUNCS.get_multiple_row_selection(self.DS_table_manager)
                            if index_rows is None:
                                return

                            FUNCS.textappend(
                                f"Analyze IVs with Important flag set from selected cell(s) at rows: {len(index_rows)!s}"
                            )
                            self.Dock_Report.raiseDock()  # so we can monitor progress
                            for index_row in index_rows:
                                selected = self.DS_table_manager.get_table_data(index_row)
                                FUNCS.textappend(f"    Selected: {selected!s}")
                                pathparts = Path(selected.cell_id).parts
                                day = pathparts[0]
                                slicecell = f"S{pathparts[1][-1]:s}C{pathparts[2][-1:]:s}"
                                FUNCS.textappend(f"    Day: {day:s}  slice_cell: {slicecell:s}")
                                self.analyze_ivs(
                                    mode="selected",
                                    important=True,
                                    day=day,
                                    slicecell=slicecell,
                                    cell_id=selected.cell_id,
                                )
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
                                coding_file = Path(
                                    self.experiment["analyzeddatapath"],
                                    self.experiment["directory"],
                                    self.experiment["coding_file"],
                                )
                            else:
                                coding_file = None
                            fn = self.PSI.get_assembled_filename(self.experiment)
                            self.PSI.assemble_datasets(
                                df_summary=self.datasummary,
                                coding_file=coding_file,
                                coding_sheet=self.experiment["coding_sheet"],
                                coding_level=self.experiment["coding_level"],
                                exclude_unimportant=self.exclude_unimportant,
                                fn=fn,
                            )

                        # case "Process Spike Data":
                        #     PSA.process_spikes()
                case "Map Analysis":
                    match path[1]:
                        case "Analyze ALL Maps":
                            pass
                        case "Analyze Selected Maps":

                            # work from the *datasummary* table, not the Assembled table.
                            index_rows = FUNCS.get_multiple_row_selection(self.DS_table_manager)
                            FUNCS.textappend(
                                f"Analyze all LSPS maps from selected cell(s) at rows: {len(index_rows)!s}"
                            )

                            for selected in index_rows:
                                if selected is None:
                                    print("selected was None")
                                    break
                                FUNCS.textappend(selected.cell_id)
                                pathparts = Path(selected.cell_id).parts
                                slicecell = f"S{pathparts[-2][-1]:s}C{pathparts[-1][-1:]:s}"
                                day = str(Path(*pathparts[0:-2]))
                                FUNCS.textappend(f"    Day: {day:s}  slice_cell: {slicecell:s}")
                                print((f"    Day: {day!s}  slice_cell: {slicecell!s}"))
                                self.analyze_maps(mode="selected", day=day, slicecell=slicecell)
                            self.maps_finished_message()
                            self.Dock_DataSummary.raiseDock()  # back to the original one

                        case "Analyze Selected Maps m/Important":
                            pass

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
                                cell_df, _ = filename_tools.get_cell(
                                    self.experiment, self.assembleddata, cell_id=selected.cell_id
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
                                    if "LowestCurrentSpike" in cell_df["Spikes"][prot].keys():
                                        pp.pprint(
                                            f"   prot LowestCurrentSpike: {cell_df['Spikes'][prot]['LowestCurrentSpike']!s}"
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

                        case "Plot Spike Data categorical":
                            self.spike_plot = self.generate_summary_plot(
                                plotting_function=plot_spike_info.concurrent_categorical_data_plotting,
                                mode="categorical",
                                title="Spike Data",
                                data_class="spike_measures",
                                colors=colors,
                            )

                        case "Plot Spike Data continuous":
                            self.spike_plot = self.generate_summary_plot(
                                plotting_function=plot_spike_info.concurrent_categorical_data_plotting,
                                mode="continuous",
                                title="Spike Data",
                                data_class="spike_measures",
                                colors=colors,
                            )

                        case "Plot Rmtau Data categorical":
                            self.rmtau_plot = self.generate_summary_plot(
                                plotting_function=plot_spike_info.concurrent_categorical_data_plotting,
                                mode="categorical",
                                title="RmTau Data",
                                data_class="rmtau_measures",
                                colors=colors,
                            )

                        case "Plot Rmtau Data continuous":
                            self.rmtau_plot = self.generate_summary_plot(
                                plotting_function=plot_spike_info.concurrent_categorical_data_plotting,
                                mode="continuous",
                                title="RmTau Data",
                                data_class="rmtau_measures",
                                colors=colors,
                            )

                        case "Plot FIData Data categorical":
                            self.fidata_plot = self.generate_summary_plot(
                                plotting_function=plot_spike_info.concurrent_categorical_data_plotting,
                                mode="categorical",
                                title="FI Data",
                                data_class="FI_measures",
                                colors=colors,
                            )

                        case "Plot FIData Data continuous":
                            self.fidata_plot = self.generate_summary_plot(
                                plotting_function=plot_spike_info.concurrent_categorical_data_plotting,
                                mode="continuous",
                                title="FI Data",
                                data_class="FI_measures",
                                colors=colors,
                            )

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
                        case "Set BSpline S":
                            self.bspline_s = data
                        case "Plot Selected Spike":
                            if self.assembleddata is None:
                                raise ValueError("Must load assembled data file first")
                            FUNCS.get_selected_cell_data_spikes(
                                self.experiment,
                                self.table_manager,
                                self.assembleddata,
                                self.bspline_s,
                                self.Dock_Traces,
                                self.win,  # target dock and window for plot
                            )

                        case "Plot Selected FI Fitting":
                            if self.assembleddata is None:
                                raise ValueError("Must load assembled data file first")

                            fn = self.PSI.get_assembled_filename(self.experiment)
                            print("Loading fn: ", fn)
                            group_by = self.ptreedata.child("Plotting").child("Group By").value()
                            hue_category = (
                                self.ptreedata.child("Plotting").child("2nd Group By").value()
                            )
                            if hue_category == "None":
                                hue_category = None
                            plot_order = self.experiment["plot_order"][group_by]
                            header = self.get_analysis_info(fn)
                            self.selected_index_rows = self.table.selectionModel().selectedRows()
                            table_data = pd.DataFrame()
                            for irow in self.selected_index_rows:
                                cellid = self.table_manager.get_table_data(irow).cell_id
                                dfi = self.assembleddata[self.assembleddata.ID == cellid]
                                table_data = pd.concat([table_data, dfi])
                            parameters = {
                                "header": header,
                                "experiment": self.experiment,
                                "datasummary": self.datasummary,
                                "assembleddata": table_data,  # only the
                                "group_by": group_by,
                                "plot_order": plot_order,
                                "colors": colors,
                                "hue_category": hue_category,
                                "pick_display_function": None,  # self.display_from_table_by_cell_id
                            }
                           # plot_spike_info.concurrent_selected_fidata_data_plotting(
                            #         fn,
                            #         parameters,
                            #         self.picker_active,
                            #         infobox={
                            #             "x": self.infobox_x,
                            #             "y": self.infobox_y,
                            #             "fontsize": self.infobox_fontsize,
                            #         },
                            #     )

                            with concurrent.futures.ProcessPoolExecutor() as executor:
                                print("executing")
                                f = executor.submit(
                                    plot_spike_info.concurrent_selected_fidata_data_plotting,
                                    filename=fn,
                                    parameters=parameters,
                                    picker_active=self.picker_active,
                                    infobox={
                                        "x": self.infobox_x,
                                        "y": self.infobox_y,
                                        "fontsize": self.infobox_fontsize,
                                    },
                                )
                                print(f.result())
                                self.fidata_plot = f.result()
                            print("Plotting selected FI's Done")

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
                        case "cell_type" | "age" | "sex" | "Group":
                            if data != None:
                                self.filters[path[1]] = data

                        case "Filter Actions":
                            if path[2] in ["Apply"]:
                                self.filters["Use Filter"] = True
                                self.table_manager.apply_filter(QtCore=QtCore, QtGui=QtGui)
                                self.DS_table_manager.apply_filter(QtCore=QtCore, QtGui=QtGui)
                            elif path[2] in ["Clear"]:
                                self.filters["Use Filter"] = False
                                self.table_manager.apply_filter(QtCore=QtCore, QtGui=QtGui)
                                self.DS_table_manager.apply_filter(QtCore=QtCore, QtGui=QtGui)
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
                            reload.reloadAll(debug=True)
                            print("\033[1000B")
                            # self.table_manager = table_manager.TableManager(
                            #     parent=self,
                            #     table=self.table,
                            #     experiment=self.experiment,
                            #     selvals=self.selvals,
                            #     altcolormethod=self.alt_colors,
                            # )

                            print("   reload ok")
                            print("-" * 80)

                            # if self.assembleddata is not None:
                            #     self.table_manager.build_table(
                            #         self.assembleddata,
                            #         mode="scan",
                            #         QtCore=QtCore,
                            #         QtGui=QtGui,
                            #     )
                            #     self.table.setSortingEnabled(True)
                            #     self.table.horizontalHeader().sortIndicatorChanged.connect(
                            #         self.handleSortIndicatorChanged
                            #     )
                            #     self.table_manager.apply_filter(QtCore=QtCore, QtGui=QtGui)
                            #     self.table.sortByColumn(
                            #         0, QtCore.Qt.SortOrder.AscendingOrder
                            #     )  # by date
                            #     self.alt_colors(self.table)  # reset the color list.
                            #     # now reapply the original selection
                            mode = (
                                QtCore.QItemSelectionModel.SelectionFlag.Select
                                | QtCore.QItemSelectionModel.SelectionFlag.Rows
                            )
                            for row in selected_rows:
                                selection_model.select(row, mode)  # for row in selected_rows:
                            # try:
                            #     self.table_manager.update_table(
                            #         self.table_manager.data, QtCore=QtCore, QtGui=QtGui
                            #     )
                            # except AttributeError:
                            #     pass
                            # leave current dock up
                            # self.Dock_DataSummary.raiseDock()

                            FUNCS.textappend("Reload OK", color="g")
                            self.doing_reload = False

                        case "View IndexFile":
                            selected = self.DS_table.selectionModel().selectedRows()
                            if selected is None:
                                return
                            index_row = selected[0]
                            self.DS_table_manager.print_indexfile(index_row)
                        case "Print File Info":
                            selected = self.table.selectionModel().selectedRows()
                            if selected is None:
                                return
                            self.print_file_info(selected)
                        case "Export Brief Table":
                            self.DS_table_manager.export_brief_table(
                                self.textbox, dataframe=self.datasummary
                            )

    def plot_selected(self, cell_id):
        dspath = Path(self.experiment["analyzeddatapath"], self.experiment["directory"])
        FUNCS.textappend(f"    Selected: {cell_id!s}")
        print("Selected for plot: ", cell_id)
        pkl_file = filename_tools.get_cell_pkl_filename(self.experiment, self.datasummary, cell_id=cell_id)
        # pkl_file = filename_tools.get_pickle_filename_from_row(selected, dspath)

        # print("Reading from pkl_file: ", pkl_file)
        FUNCS.textappend(f"    Reading and plotting from: {pkl_file!s}")

        tempdir = Path(dspath, "temppdfs")
        decorate = True
        self.pdfFilename = Path(dspath, self.experiment["pdfFilename"]).with_suffix(".pdf")
        print(f"Plotting would write to: {self.pdfFilename!s}")
        # use concurrent futures to prevent issue with multiprocessing
        # when using matplotlib.
        with concurrent.futures.ProcessPoolExecutor() as executor:
            print("   submitting execution to concurrent futures")
            f = executor.submit(
                iv_plotter.concurrent_iv_plotting,
                pkl_file,
                self.experiment,
                self.datasummary,
                dspath,
                decorate,
            )
            ret = f.result()

    def analyze_ivs(
        self,
        mode="all",
        important: bool = False,
        day: str = None,
        slicecell: str = None,
        cell_id: str = None,
    ):
        """
        Analyze the IVs for the selected cell
        Parameters
        mode can be: "all", "selected", "important"
        If all, all protocols are analyzed.
        if selected, only the selected cells are analyzed for the day/slicecell.
        if important, only the "important" protocols are analyzed for the day/slicecell.

        important: bool
            if True, we pay attention to the "Important" flag in the data.
            Otherwise, we just analyze whatever comes our way.

        """
        args = analysis_common.cmdargs  # get from default class
        args.dry_run = self.dry_run
        args.merge_flag = True
        args.experiment = self.experiment
        args.iv_flag = True
        args.map_flag = False
        args.autoout = True
        nworkers = self.experiment["NWORKERS"][self.computer_name]
        args.parallel_mode = self.parallel_mode  # get value from the parameter tree

        if nworkers == 1:
            args.parallel_mode = "off"
            args.nworkers = 1
        if args.parallel_mode != "off":
            args.nworkers = nworkers
        print("data_table: nworkers, parallel_mode", args.nworkers, args.parallel_mode)
        if mode == "important" or important:
            args.important_flag_check = True  # only analyze the "important" ones
        else:
            args.important_flag_check = False

        args.verbose = False
        args.spike_threshold = self.experiment["AP_threshold_V"]  # always in Volts
        args.spike_detector = self.experiment["spike_detector"]
        args.fit_gap = self.experiment["fit_gap"]

        if mode == "selected":
            args.day = day
            args.slicecell = slicecell
            args.cell_id = cell_id
        else:
            args.cell_id = None

        CP.cprint(
            "g",
            f"Starting IV analysis at: {datetime.datetime.now().strftime('%m/%d/%Y, %H:%M:%S'):s}",
        )
        print("\n" * 2)
        CP.cprint("g", "=" * 80)
        IV = iv_analysis.IVAnalysis(args)  # all args are passed
        IV.set_experiment(self.experiment)
        CP.cprint("c", " datatables: experiment set")
        if "excludeIVs" in self.experiment.keys():
            IV.set_exclusions(self.experiment["excludeIVs"])
            # CP.cprint("y", self.experiment['excludeIVs'])
        else:
            CP.cprint("y", "No IV exclusions set")
            return
        IV.setup()
        CP.cprint("c", "analyze_ivs datatables: setup completed")
        IV.run()
        CP.cprint("c", "analyze_ivs datatables: run completed")
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
        if self.dry_run:
            return
        CP.cprint("r", "=" * 80)
        # CP.cprint("r", f"Now run 'process_spike_analysis' to generate the summary data file")
        CP.cprint("r", f"Now run 'assemble datasets' to combine the FI curves")
        CP.cprint(
            "r", f"Then try the plotting functions to plot summaries and get statistical results"
        )
        CP.cprint("r", "=" * 80)

    def generate_summary_plot(
        self, plotting_function: object, mode: str = "categorical", data_class:str="spike_measures",
         title:str="My Title", colors: dict = {}
    ):
        # data class must be in the experimeng configuration file, top level keys.
        # e.g.:
        # spike_measures: ["dvdt_rising", "dvdt_falling", "AP_HW", "AP_thr_V", "AHP_depth_V", "AHP_trough_T"]
        # rmtau_measures: ["RMP", "Rin", "taum"]
        # FI_measures: ["AdaptRatio",  "maxHillSlope", "I_maxHillSlope", "FIMax_1", "FIMax_4"]

        assert data_class in self.experiment.keys()
        fn = self.PSI.get_assembled_filename(self.experiment)
        group_by = self.ptreedata.child("Plotting").child("Group By").value()
        plot_order = self.experiment["plot_order"][group_by]
        hue_category = self.ptreedata.child("Plotting").child("2nd Group By").value()
        if hue_category == "None":
            hue_category = None
        header = self.get_analysis_info(fn)
        parameters = {
            "header": header,
            "experiment": self.experiment,
            "datasummary": self.datasummary,
            "group_by": group_by,
            "colors": colors,
            "hue_category": hue_category,
            "plot_order": plot_order,
            "pick_display_function": None,  # self.display_from_table_by_cell_id
        }
        with concurrent.futures.ProcessPoolExecutor() as executor:
            f = executor.submit(
                plotting_function,
                filename=fn,
                parameters=parameters,
                mode=mode,
                plot_title = title,
                data_class=data_class,
                picker_active=self.picker_active,
                infobox={
                    "x": self.infobox_x,
                    "y": self.infobox_y,
                    "fontsize": self.infobox_fontsize,
                },
            )
        return f.result()

    def analyze_maps(
        self, mode="all", important: bool = False, day: str = None, slicecell: str = None
    ):
        import numpy as np

        args = analysis_common.cmdargs  # get from default class
        args.dry_run = False
        args.autoout = True
        args.merge_flag = True
        args.experiment = self.experiment
        args.iv_flag = False
        args.map_flag = True
        args.mapsZQA_plot = False
        args.zscore_threshold = 1.96  # p = 0.05 for charge relative to baseline

        args.plotmode = "document"
        args.recalculate_events = True
        args.artifact_filename = self.experiment["artifactFilename"]
        args.artifact_path = self.experiment["artifactPath"]

        args.artifact_suppression = True
        args.artifact_derivative = False
        args.post_analysis_artifact_rejection = False
        args.autoout = True
        args.verbose = False
        # these are all now in the excel table
        # args.LPF = 3000.0
        # args.HPF = 0.

        args.detector = "aj"
        args.spike_threshold = -0.020  # always in Volts
        # args.threshold = 7

        if mode == "selected":
            args.day = day
            args.slicecell = slicecell

        args.notchfilter = True
        odds = np.arange(1, 43, 2) * 60.0  # odd harmonics
        nf = np.hstack(
            (odds, [30, 15, 120.0, 240.0, 360.0])
        )  # default values - replaced by what is in table
        str_nf = "[]"  # "[" + ", ".join(str(f) for f in nf) + "]"
        args.notchfreqs = str_nf  # "[60., 120., 180., 240., 300., 360., 600., 4000]"
        args.notchQ = 90.0

        # if args.configfile is not None:
        #     config = None
        #     if args.configfile is not None:
        #         if ".json" in args.configfile:
        #             config = json.load(open(args.configfile))
        #         elif ".toml" in args.configfile:
        #             config = toml.load(open(args.configfile))

        # vargs = vars(args)  # reach into the dict to change values in namespace
        # for c in config:
        #     if c in args:
        #         # print("c: ", c)
        #         vargs[c] = config[c]
        CP.cprint(
            "cyan",
            f"Starting MAP analysis at: {datetime.datetime.now().strftime('%m/%d/%Y, %H:%M:%S'):s}",
        )
        print("\n" * 3)
        CP.cprint("r", "=" * 80)

        MAP = map_analysis.MAP_Analysis(args)
        MAP.set_experiment(self.experiment)
        # MAP.set_exclusions(self.experiment.exclusions)
        MAP.AM.set_artifact_suppression(args.artifact_suppression)
        MAP.AM.set_artifact_path(self.experiment["artifactPath"])
        MAP.AM.set_artifact_filename(self.experiment["artifactFilename"])
        MAP.AM.set_post_analysis_artifact_rejection(args.post_analysis_artifact_rejection)
        MAP.AM.set_template_parameters(tmax=0.009, pre_time=0.001)
        MAP.AM.set_shutter_artifact_time(0.050)

        CP.cprint("b", "=" * 80)
        MAP.setup()

        CP.cprint("c", "=" * 80)
        MAP.run()

        # allp = sorted(list(set(NF.allprots)))
        # print('All protocols in this dataset:')
        # for p in allp:
        #     print('   ', path)
        # print('---')
        #
        CP.cprint(
            "cyan",
            f"Finished analysis at: {datetime.datetime.now().strftime('%m/%d/%Y, %H:%M:%S'):s}",
        )

    def maps_finished_message(self):
        if self.dry_run:
            return
        CP.cprint("r", "=" * 80)
        # CP.cprint("r", f"Now run 'process_spike_analysis' to generate the summary data file")
        # CP.cprint("r", f"Now run 'assemble datasets' to combine the FI curves")
        CP.cprint(
            "r", f"Now try the plotting functions to plot summaries and get statistical results"
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

    def set_color_to_row(self, rowIndex, color):
        """
        Set the color of a row
        """
        for j in range(self.table.columnCount()):
            if self.table.item(rowIndex, j) is not None:
                self.table.item(rowIndex, j).setBackground(color)

    def alt_colors(
        self, table, colors=[QtGui.QColor(0x22, 0x22, 0x22), QtGui.QColor(0x44, 0x44, 0x44)]
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
                self.set_color_to_row(j, colors[0])
            else:
                self.set_color_to_row(j, colors[1])

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
            basedir=Path(self.experiment["rawdatapath"], self.experiment["directory"]),
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
            ds,
            Path(self.experiment["rawdatapath"], self.experiment["directory"]),
            self.experiment["excludeIVs"],
        )
        # after creation, load it
        self.load_data_summary()

    def find_unique_protocols(self):
        """
        find all of the unique protocols in this database

        """
        if self.datasummary is None:
            raise ValueError("Please load the datasummary file first")
        FUNCS.get_datasummary_protocols(self.datasummary)

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

    # def analyze_singles(self, ana_name=None):
    #     """
    #     Analyze data this is formatted from the 'singles' runs in model_run2
    #     These are runs in which each input is evaluated independently
    #     """
    #     index_row, selected = FUNCS.get_row_selection(self.table_manager)
    #     if selected is None:
    #         return
    #     self.PLT.analyze_singles(index_row, selected)

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

    def display_from_table(self, mode="IVs"):
        """
        display the data in the PDF dock
        """
        match mode:
            case "IV" | "IVs" | "iv" | "ivs":
                self.selected_index_rows = self.DS_table.selectionModel().selectedRows()
                if self.selected_index_rows is None:
                    return
                index_row = self.selected_index_rows[0]
                selected = self.DS_table_manager.get_table_data(index_row)  # table_data[index_row]
                modename = "IVs"
                # build the filename for the IVs, format = "2018_06_20_S4C0_pyramidal_IVs.pdf"
                # the first part is based on the selected cell_id from the table, and the rest
                # just makes life easier when looking at the directories.
            case "Map" | "map" | "Maps" | "maps":
                self.selected_index_rows = self.DS_table.selectionModel().selectedRows()
                if self.selected_index_rows is None:
                    return
                index_row = self.selected_index_rows[0]
                selected = self.DS_table_manager.get_table_data(index_row)
                modename = "maps"
            case _:
                raise ValueError(f"Invalid mode: {mode!s}")

        cell_type = selected.cell_type
        if cell_type == " " or len(cell_type) == 0:
            cell_type = "unknown"

        datapath = self.experiment["databasepath"]
        directory = self.experiment["directory"]

        filename = filename_tools.get_pickle_filename_from_row(
            selected, Path(datapath, directory), mode=modename
        )
        filename = Path(filename).with_suffix(".pdf")
        url = "file://" + str(filename)
        FUNCS.textappend(f"File exists:  {filename!s}, {Path(filename).is_file()!s}")
        print(f"File exists:  {filename!s}, {Path(filename).is_file()!s}")
        if not Path(filename).is_file():
            FUNCS.textappend(f"File does not exist: {filename!s}")
            msg = pg.QtWidgets.QMessageBox()
            infotext = f"PDF file for {modename:s} does not exist for cell: {selected.cell_id!s}\nSearched file: {str(filename)!s}"
            title = f"<center>{selected.cell_id:s}</center>"
            text = "{}<br><br>{}".format(title, "\n".join(textwrap.wrap(infotext, width=120)))
            msg.setText(text)
            msg.setFont(QtGui.QFont("Arial", 11, QtGui.QFont.Weight.Normal))
            # msg.setInformativeText(f"{colname:s} = {self.DS_table_manager.table.item(row, col).text():s}")
            msg.exec()
            return

        self.PDFView.setUrl(pg.QtCore.QUrl(url))  # +"#zoom=80"))
        self.Dock_PDFView.raiseDock()


def main():
    # Entry point. Why do I do this ? It keeps sphinxdoc from running the
    # code...
    (
        datasets,
        experiments,
    ) = get_configuration(config_file_path)  # retrieves the configuration file from the running directory
    D = DataTables(datasets, experiments)  # must retain a pointer to the class, else we die!
    if (sys.flags.interactive != 1) or not hasattr(QtCore, "PYQT_VERSION"):
        QtWidgets.QApplication.instance().exec()


if __name__ == "__main__":
    if plotform.system() == "Darwin":
        multiprocessing.set_start_method("spawn")
    main()
