""" tools for the data_tables

A configuration fiel has the follwing variables and dictionary structure:

rawdatadisk = "/Volumes/Pegasus_002/ManisLab_Data3/Kasten_Michael/"
analyzeddatapath = "/Users/pbmanis/Desktop/Python/mrk-nf107-data/datasets"
experiments = { 
    "NF107Ai32_Het": {   # name of experiment
        "rawdatapath": rawdatadisk,
        "databasepath": analyzeddatapath,  # location of database files (summary, coding, annotation)
        "analyzeddatapath": analyzeddatapath,  # analyzed data set directory
        "directory": "NF107Ai32_Het",  # directory for the raw data, under rawdatadisk
        "pdfFilename": "NF107Ai32_IVs_08-2023.pdf",  # PDF figure output file
        "datasummaryFilename": "NF107Ai32_Het_05_Oct_2023.pkl",  # name of the dataSummary output file, in analyzeddatapath
        "IVs": "NF107Ai32_Spikes",  # name of the excel sheet generated to hold the IV results
        "iv_analysisFilename": "IV_Analysis-05_Oct_2023.pkl",  #  "IV_Analysis.h5", # name of the pkl or HDF5 file that holds all of the IV analysis results
        "coding_file":  None, # "Intrinsics.xlsx",
        "coding_sheet": None,  # "codes",
        "cell_annotationFilename": None, # "NF107Ai32_Het_cell_annotations.xlsx",  # annotation file, in resultdisk
        "bridgeCorrectionFilename": None,  # location of the excel table with the bridge corrections
        "extra_subdirectories": ["Parasagittal", "OLD",  "NF107Ai32-TTX-4AP"],
        "maps": None,
        "result_sheet": "Het_IVs_PCT.xlsx",  # excel sheet with results
        "pdf_filename": "Het_IVs_PCT.pdf",
        "excludeIVs": ["2022.03.02_000_S0C1: in ttx", 
                       "2022.03.28_000_S0C1: in ttx",
                       "2017.08.28_000_S0C0: poor recording",
                       "2017.08.11_000_S0C0: poor recording",
                       "2017.06.20_000_S0C1: no depol data",
                       "2022.02.28_000_S0C2: in TTX",
                       ]
    },
    "NF107Ai32_NIHL": {  # name of experiment
        "rawdatapath": rawdatadisk,
        "databasepath": analyzeddatapath,  # location of database files (summary, coding, annotation)
        "analyzeddatapath": analyzeddatapath,  # analyzed data set directory
        "directory": "NF107Ai32_NIHL",  # directory for the raw data, under rawdatadisk
        "pdfFilename": "NF107Ai32_NIHL_IVs_09-2023.pdf",  # PDF figure output file
        "datasummaryFilename": "NF107Ai32_NIHL_Summary.pkl",  # name of the dataSummary output file, in analyzeddatapath
        "IVs": "NF107Ai32_NIHL_Spikes",  # name of the excel sheet generated to hold the IV results
        "iv_analysisFilename": "NIHL_combined_by_cell.pkl", # "IV_Analysis_NIHL-9-2023.pkl",  #  "IV_Analysis.h5", # name of the pkl or HDF5 file that holds all of the IV analysis results
        "coding_file":  "NF107Ai32_NoiseExposure_Code.xlsx", # "Intrinsics.xlsx",
        "coding_sheet": "SubjectData", # "codes",
        "cell_annotationFilename": None, # "NF107Ai32_Het_cell_annotations.xlsx",  # annotation file, in resultdisk
        "bridgeCorrectionFilename": None,  # location of the excel table with the bridge corrections
        "extra_subdirectories": ["2W", "BNE_3d", "Parasagittal", "OLD", "NF107Ai32-TTX-4AP"],
        "maps": None,
        "result_sheet": "NIHL_IVs_PCT.xlsx", # "nihl_AP_PCT.xlsx",  # excel sheet with results
        "pdf_filename": "NIHL_IVs_PCT.pdf",
        "excludeIVs": [
            "BNE_3D/2021.03.05_000_S1C0: lost cell in middle of protocol; no map data",
                        ]
    }
}
    
"""
import sys
import pyqtgraph as pg
from pyqtgraph.parametertree import Parameter, ParameterTree
import pyqtgraph.dockarea as PGD
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets

table_keys = [
    "datatype", # either IVs or Maps or Events?
    "rawdatapath",
    "databasepath",
    "analyzeddatapath",
    "directory",
    "pdfFilename",
    "datasummaryFilename",
    "IVs",
    "excludeIVs",
    "iv_analysisFilename",
    "eventsummaryFilename",
    "coding_file",
    "coding_sheet",
    "cell_annotationFilename",
    "bridgeCorrectionFilename",
    "extra_subdirectories",
    "maps",
    "map_annotationFilename",
    "result_sheet",
    "pdf_filename",
    "artifactPath",
    "artifactFilename",
    "selectedMapsTable",
]

class TableTools():
    def __init__(self):
        self.app = pg.mkQApp()
        self.app.setStyle("fusion")
        self.param = {}
        self.ptree = ParameterTree()
        self.ptreedata = None
        self.table_data = None
        self.win = pg.QtWidgets.QMainWindow()
        # use dock system instead of layout.
        self.dockArea = PGD.DockArea()
        self.win.setCentralWidget(self.dockArea)
        self.win.setWindowTitle("NF107 DataTables")
        self.win.resize(1600, 1024)
        # Initial Dock Arrangment
        self.Dock_Params = PGD.Dock("Params", size=(500, 1000))

        # self.layout = QtGui.QGridLayout()
        # self.window.setLayout(self.layout)
        l = QtWidgets.QLabel("Set up Dataset")
        l.setAlignment(QtCore.Qt.AlignmentFlag.AlignHCenter | QtCore.Qt.AlignmentFlag.AlignVCenter)
        # self.layout.addWidget(l)
        # self.layout.addWidget(self.ptree)
        self.win.setWindowFlags(QtCore.Qt.WindowType.WindowStaysOnTopHint)
        self.win.setGeometry(500, 1000, 800, 1000) 
        self.dockArea.addDock(self.Dock_Params, "left")
        self.win.show()


    def create_new_dataset(self):
        """create_new_dataset creates a new dataset from the template"""
        self.table_data = {key: "" for i, key in enumerate(table_keys)}
        # just make a bunch of string entry fields. 
        self.param = []
        self.param.append({"name": "New Dataset", "type": "group", "children": []})
    
        for i, key in enumerate(self.table_data.keys()):
            self.param[0]["children"].append({"name": key, "type": "str", "value": self.table_data[key]})
        self.param.append({"name": "Validate", "type": "action"})
        self.param.append({"name": "Quit", "type": "action"})

        self.ptree = ParameterTree()
        self.ptreedata = Parameter.create(
            name="Dataset", type="group", children=self.param
        )
        self.ptree.setParameters(self.ptreedata)
        self.Dock_Params.addWidget(self.ptree)  # put the parameter three here
        self.ptreedata.sigTreeStateChanged.connect(self.command_dispatcher)
        return None

    def command_dispatcher(self, param, changes):
        for param, change, data in changes:
            path = self.ptreedata.childPath(param)

            match path[0]:
                case "Quit":
                    exit()
                case "Validate":
                    self.validate_dataset()
                case "Save":
                    self.save_dataset()


    def validate_dataset(self):
        """validate_dataset checks the dataset for completeness and consistency, and confirms
        the existence of the expected paths and files."""

        return None

    def save_dataset(self):
        return None
        
if __name__ == "__main__":
    tt = TableTools()
    tt.create_new_dataset()
    tt.validate_dataset()
    if (sys.flags.interactive != 1) or not hasattr(QtCore, "PYQT_VERSION"):
        QtWidgets.QApplication.instance().exec()