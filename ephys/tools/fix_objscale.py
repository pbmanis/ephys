"""
Fix the objective scale factor
Give "stated objective", "actual objective"

('objective', '4x 0.1na ACHROPLAN')

"""
import os
import sys

from dataclasses import dataclass
from typing import Union
import numpy as np
import argparse
from pathlib import Path
from pyqtgraph import configfile
import pprint
from collections import OrderedDict
import datetime
import pyqtgraph as pg
import pyqtgraph.dockarea as PGD
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets
from pylibrary.tools import fileselector as FS
from pyqtgraph.parametertree import Parameter, ParameterTree

import pylibrary.tools.cprint as CP

os.environ["QT_MAC_WANTS_LAYER"] = "1"

pp = pprint.PrettyPrinter(indent=4)


CineScale = 1.0
refscale = [(1.0 / CineScale) * 6.54e-6, -(1.0 / CineScale) * 6.54e-6]

objectiveDict = {
    "4x 0.1na ACHROPLAN": 4.0,
    "10x 0.3na W N-ACHROPLAN": 10.0,
    "20x 0.5na W N-ACHROPLAN": 20.0,
    "40x 0.8na ACHROPLAN": 40.0,
    "63x 0.9na ACHROPLAN": 63.0,
}


@dataclass
class Changer:
    change_type: str = "new"
    filename: str = ""
    from_objective: str = "4x 0.1na ACHROPLAN"
    to_objective: str = "10x 0.3na W N-ACHROPLAN"
    videos: Union[str, None] = None
    images: Union[str, None] = None



class FixObjective(pg.QtGui.QWidget):
    def __init__(self, app=None):
        super(FixObjective, self).__init__()
        self.app = app
        self.write = False
        self.datadir = "/Volumes/Pegasus_002/ManisLab_Data3/Kasten_Michael/HK_collab_ICinj/DCN_IC_inj"
        self.objdata = Changer()
        self.filelistpath = "../fixobjective.toml"

    def getProtocolDir(self, reload_last=False):
        current_filename = None
        if not reload_last:
            sel = FS.FileSelector(dialogtype="dir", startingdir=self.datadir)
            current_filename = sel.fileName
        else:
            if self.filelistpath.is_file():
                file_dict = toml.load(self.filelistpath)
                current_filename = file_dict['MostRecent']
            else:
                print('No Previous Files Found')
                return
        self.objdata.filename = current_filename
        self.read_index(objective=self.objdata, write=False)
        self.show_index()
        
    def rewrite_index(self, index: dict, index_file: Union[str, Path]):
        configfile.writeConfigFile(index, index_file)


    def read_indexes(self,changelist: list = [], write: bool = False):
        """
        To change a list of images/videos
        """
        print("write: ", write)
        for objective in changeList:
            index = read_index(index_path, objective, write=write)

    def read_index(self, objective: object = None, write: bool = False):
        print("\nfix_objscale: We will be using the following reference scale: ", refscale)
        print("   This scale may be specific to your camera!!!!!")
        print("read_index write flag is: ", write)
        nl = "\n"
        print(f"Objective: ")
        print(f"    Change Type: {objective.change_type:s}")
        print(f"    File: {objective.filename:s}")
        print(f"    From: {objective.from_objective:3s}X to {objective.to_objective:3s}X")
        print(f"    For images: {str(objective.images):s}")
        print(f"    For videos: {str(objective.videos):s}")
        print("="*40)
        self.index_file = Path(objective.filename, ".index")
        self.index = configfile.readConfigFile(self.index_file)
        return self.index

    def show_index(self):
        print("self.index: ")
        # print(self.index)
        self.textbox.setCurrentFont(QtGui.QFont("Courier New"))
        text = [] # ["<font color='red' size='3' font-family='monospace'>Hello PyQt5!\nHello"]
        # ["<font-family:'Courier New' color='blue' size='10'>"]
        for k in self.index.keys():
            print("index is: ", k)
            if k.startswith("image"):
                text.append(f" {k:16s}: {self.index[k]['objective']:24s}, {str(self.index[k]['binning']):8s} {str(self.index[k]['transform']['scale']):s}<br>")
            if k.startswith("video"):
                text.append(f" {k:16s}: {self.index[k]['objective']:24s}, {str(self.index[k]['binning']):8s} {str(self.index[k]['transform']['scale']):s}<br>")

        self.textbox.setHtml("\n".join([t for t in text]))

    def set_new_objective(self, data):
        self.objdata.to_objective = data

    def update_from_objective(self):
        fn = self.get_imagefilename(self.objdata)  # get for the current
        # print("fn: ", fn)
        # print('self.index: ', self.index)
        objname = self.index[fn]['objective']
        if objname not in list(objectiveDict.keys()):
            print(f"Objective {objname:s} not found in list of known objectives.")
            return
        print("objname: ", objname)
        self.ptreedata.param('Original Objective').setValue(objname)

        
    def get_original_objective(self):
        pass
    def view_proposed_changes(self):
        print(self.objdata)
        self.change_scale(self.objdata, write=False)
    
    def get_imagefilename(self, objective: object) -> str:
        if objective.images is not None:
            imagefile = f"image_{int(objective.images):03d}.tif"
        elif objective.videos is not None:
            imagefile = f"video_{int(objective.videos):03d}.ma"
        else:
            imagefile = None
        return imagefile
        
    def change_scale(self, objective:object=None, write:bool=False):
        imagefile = self.get_imagefilename(objective)
        if imagefile is None:
            return

        print("\n----------------------------")
        print(imagefile)
        print(self.index.keys())
        if imagefile not in self.index.keys():
            CP.cprint("m", f"File {imagefile:s} not found in {str(list(self.index.keys())):s}")
            return
        k = imagefile
        print("Index: ", k)
        print("Old objective: ", self.index[k]["objective"])  # pp.pprint(index[k] )
        old_objective = self.index[k]["objective"]
        pp.pprint("   Old transform: ")
        pp.pprint(self.index[k]["transform"])
        pp.pprint("   Old device transform,: ")
        pp.pprint(self.index[k]["deviceTransform"])
        binning = self.index[k]["binning"]
        new_objective = objective.to_objective # string name
        f_new_objective = float(objectiveDict[new_objective])  # new magnification
        self.index[k]["transform"]["scale"] = (
            binning[0] * refscale[0] / f_new_objective,
            binning[1] * refscale[1] / f_new_objective,
            1.0,
        )
        self.index[k]["deviceTransform"]["scale"] = (
            binning[0] * refscale[0] / f_new_objective,
            binning[1] * refscale[1] / f_new_objective,
            1.0,
        )
        d = datetime.datetime.now()
        dstr = d.strftime("%Y-%m-%d %H:%M:%S")
        self.index[k]["objective"] = objectiveDict[objective.to_objective]
        self.index[k][
            "note"
        ] = f"Objective scale corrected from {str(objectiveDict[old_objective]):s} to {str(objectiveDict[new_objective]):s} on {dstr:s} by PBM"
        print("New objective: ", self.index[k]["objective"])  # pp.pprint(index[k] )
        print("   New transform: ")
        pp.pprint(self.index[k]["transform"])
        print("   New device transform: ")
        pp.pprint(self.index[k]["deviceTransform"])
        print("   Added Note: ", self.index[k]["note"])
        print("----------------------------")

        print("read_index write: ", write)
        if write:
            rewrite_index(self.index, index_file)
            CP.cprint("g", ".index file has been updated")

        else:
            print("Dry Run: .index file was NOT modified")

    def build_ptree(self):
        self.params = [
            # {"name": "Pick Cell", "type": "list", "values": cellvalues, "value": cellvalues[0]},
            {"name": "Set Directory/Protocol", "type": "action"},
            {"name": "Reload Last Protocol", "type": "action"},
            {"name": "View .index", "type": "action"},
            
            {
                "name": "Original Objective",
                "type": "list",
                "values": list(objectiveDict.keys()),
                "value": self.objdata.from_objective,
                "renamable": False,
            },
            {
                "name": "New Objective",
                "type": "list",
                "values": list(objectiveDict.keys()),
                "value": self.objdata.to_objective,
                "renamable": False,
            },
            {
                "name": "Images",
                "type": "str",
                "value": "",
            },
            {
                "name": "Videos",
                "type": "str",
                "value": "",
            },
            {
                "name": "View proposed changes",
                "type": "action"
            },
            {
                "name": "Quit",
                "type": "action",
            },
            
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
            elif path[0] == "View .index":
                self.show_index()
            elif path[0] == "Set Directory/Protocol":
                self.getProtocolDir()

            elif path[0] == "Original Objective":
                self.objdata.from_objective = data
            elif path[0] == "New Objective":
                self.objdata.to_objective = data
            elif path[0] == "View proposed changes":
                self.view_proposed_changes()
            elif path[0] == "Videos":
                self.objdata.videos = int(data)
            elif path[0] == "Images":
                self.objdata.images = int(data)
                print(self.objdata.images)
                self.update_from_objective()
                
            elif path[0] == "Quit":
                self.quit()

    def set_window(self, parent=None):
        super(FixObjective, self).__init__(parent=parent)
        self.win = pg.QtGui.QMainWindow()
        self.win.setWindowTitle("FixObjective")
        self.dockArea = PGD.DockArea()

        layout = pg.QtGui.QGridLayout()
        layout.setSpacing(8)
        self.win.setCentralWidget(self.dockArea)
        self.win.setWindowTitle("Model DataTables/FileSelector")
        self.win.resize(850, 400)
        self.win.setWindowTitle("No File")
        
        self.buttons = pg.QtGui.QGridLayout()
        self.build_ptree()
        self.buttons.addWidget(self.ptree)
        self.ptreedata.sigTreeStateChanged.connect(self.command_dispatcher)

        # Initial Dock Arrangment
        self.Dock_Params = PGD.Dock("Params", size=(350, 400))
        self.Dock_Params.addWidget(self.ptree)
        self.Dock_Report = PGD.Dock("Reporting", size=(500, 400))

        self.textbox = QtWidgets.QTextEdit()
        self.textbox.setReadOnly(True)
        self.textbox.setText("Text Edit box (RO)")
        self.Dock_Report.addWidget(self.textbox) 

        self.dockArea.addDock(self.Dock_Params, "left")
        self.dockArea.addDock(self.Dock_Report, "right", self.Dock_Params)
        
        self.win.show()
    
    def quit(self):
        exit(0)


def main():
    app = pg.QtGui.QApplication([])
    FO = FixObjective(app)
    app.aboutToQuit.connect(
        FO.quit
    )  # prevent python exception when closing window with system control
    FO.set_window()

    if (sys.flags.interactive != 1) or not hasattr(pg.QtCore, "PYQT_VERSION"):
        pg.QtGui.QApplication.instance().exec_()




# old command line version
# def main():
#     parser = argparse.ArgumentParser(description="Fix the objective")
#     parser.add_argument(
#         "-w",
#         "--write",
#         action="store_true",
#         dest="write",
#         help="Rewrite the .index file (otherwise, we just do a dry run)",
#     )
#
#     args = parser.parse_args()
#
#     read_indexes(bp, changeList, args.write)


if __name__ == "__main__":
    main()
