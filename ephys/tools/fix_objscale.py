"""
Fix the objective scale factor
Give "stated objective", "actual objective"

('objective', '4x 0.1na ACHROPLAN')

"""

import datetime
import os
import pprint
import sys
import tomllib as toml
from dataclasses import dataclass
from pathlib import Path
from typing import Union

import numpy as np
import pylibrary.tools.cprint as CP
import pyqtgraph as pg
import pyqtgraph.dockarea as PGD
from pylibrary.tools import fileselector as FS
from pylibrary.tools.utility import seqparse as SQP
from pyqtgraph import configfile
from pyqtgraph.parametertree import Parameter, ParameterTree
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets

pp = pprint.PrettyPrinter(indent=4)


def get_configuration(configfile):
    cpath = Path(Path().absolute(), configfile)
    print("Getting Configuration file from: ", str(cpath))
    config = pg.configfile.readConfigFile(cpath)
    return config


# default dict for objective data:
objective_data = {
    "Objectives": {
        # format name: "magnification na type"
        "4x 0.1na ACHROPLAN": 4.0,
        "5x 0.25na ACHROPLAN": 5.0,
        "10x 0.3na W N-ACHROPLAN": 10.0,
        "20x 0.5na W N-ACHROPLAN": 20.0,
        "40x 0.75na W ACHROPLAN": 40.0,
        "40x 0.8na ACHROPLAN": 40.0,
        "63x 0.9na ACHROPLAN": 63.0,
    },
    "CineScales": {
        1.0: 1.0,  # scale factor for cine data
        0.5: 0.5,
        0.3: 0.3,
    },
    # This is the coupler between the camera and microscope.
    # our couplers are 1x or 0.5x.
    "Cameras": {
        # information about the cameras - specifically, the pixel size
        "RetigaR1": {"pixelsize": 6.54e-6},
        "Electro": {"pixelsize": 6.45e-6},
        "Prism95B": {"pixelsize": 11.0e-6},
        "Retiga2000DC": {"pixelsize": 7.40e-6},
    },
    "LastFile": "/Volumes/Pegasus_002/ManisLab_Data3/Kasten_Michael/NF107Ai32_Het/Parasagittal/2020.07.07_000/slice_000/cell_000",
}


@dataclass
class Changer:
    change_type: str = "new"
    filename: str = ""
    from_objective: str = list(objective_data["Objectives"].keys())[0]
    to_objective: str = list(objective_data["Objectives"].keys())[4]
    videos: Union[str, None] = None
    images: Union[str, None] = None


class FixObjective(pg.QtWidgets.QWidget):
    def __init__(self, objective_data=None, app=None):
        super(FixObjective, self).__init__()
        self.objective_data = objective_data
        self.app = app
        self.write = False
        self.datadir = "/Volumes/Pegasus_002/ManisLab_Data3/Kasten_Michael/NF107Ai32_Het"
        self.objdata = Changer()
        self.filelistpath = "../fixobjective.toml"
        self.last_file = None
        self.selected_camera = "RetigaR1"
        self.selected_cinescale = "1.0"
        self.ptreewid = 250
        self.update_camera_settings()

    def update_camera_settings(self):
        self.camera = self.objective_data["Cameras"][self.selected_camera]
        self.cinescale = self.objective_data["CineScales"][self.selected_cinescale]
        self.pixelsize = self.objective_data["Cameras"][self.selected_camera]["pixelsize"]
        self.refscale = [
            (1.0 / self.cinescale) * self.pixelsize,
            -(1.0 / self.cinescale) * self.pixelsize,
        ]

    def getProtocolDir(self, reload_last=False):
        current_filename = None
        if not reload_last:
            sel = FS.FileSelector(dialogtype="dir", startingdir=self.datadir)
            current_filename = sel.fileName
            if Path(current_filename).is_file():
                self.datadir = Path(current_filename)
        else:
            current_filename = self.objective_data["LastFile"]
            # if self.filelistpath.is_file():
            #     file_dict = toml.load(self.filelistpath)
            #     current_filename = file_dict["MostRecent"]
            # else:
            #     print("No Previous Files Found")
            #     return
        print("Current Filename: ", current_filename)

        self.objdata.filename = current_filename
        self.read_index(objective=self.objdata, write=False)
        self.show_index()

    def rewrite_index(self, index: dict, index_file: Union[str, Path]):
        configfile.writeConfigFile(index, index_file)

    def read_indexes(self, changeList: list = [], write: bool = False):
        """
        To change a list of images/videos
        """
        print("write flag: ", write)
        for objective in changeList:
            self.read_index(objective, write=write)

    def read_index(self, objective: object = None, write: bool = False):
        print("\nfix_objscale: We will be using the following reference scale: ", self.refscale)
        print("   This scale may be specific to your camera!!!!!")
        print("read_index write flag is: ", write)
        print("Objective: ")
        print(f"    Change Type: {objective.change_type!s}")
        print(f"    File: {objective.filename!s}")
        print(f"    From: {objective.from_objective!s}X to {objective.to_objective!s}X")
        print(f"    For images: {str(objective.images)!s}")
        print(f"    For videos: {str(objective.videos)!s}")
        print("=" * 40)
        self.index_file = Path(objective.filename, ".index")
        self.index = configfile.readConfigFile(self.index_file)

    # def show_index(self):
    #     """show_index display the relevant data from the index on the right side.

    #     """

    #     self.textbox.setCurrentFont(QtGui.QFont("Courier New"))
    #     text = []  # ["<font color='red' size='3' font-family='monospace'>Hello PyQt5!\nHello"]
    #     # ["<font-family:'Courier New' color='blue' size='10'>"]
    #     # imagelist = [""]
    #     # videolist = [""]

    #     # I'm sure there is a better way to do this, but parametrees are poorly documented
    #     # and few practical examples
    #     # u = self.ptreedata.children()
    #     # ulist = {}  # hold the parameter objects for the images and videos
    #     # for uu in u:
    #     #     if uu.name() in ["Images", "Videos"]:
    #     #         uu.setValue("")
    #     #         uu.setLimits([])
    #     #         ulist[uu.name()] = uu

    #     for k in self.index.keys():
    #         if k.startswith("image"):
    #             text.append(
    #                 f" {k:16s}: {self.index[k]['objective']:24s}, {str(self.index[k]['binning']):8s}"
    #                 + f" {str(self.index[k]['transform']['scale']):s}<br>"
    #             )
    #             imagelist.append(k)
    #         if k.startswith("video"):
    #             text.append(
    #                 f" {k:16s}: {self.index[k]['objective']:24s}, {str(self.index[k]['binning']):8s}"
    #                 + f" {str(self.index[k]['transform']['scale']):s}<br>"
    #             )
    #             videolist.append(k)

    #     self.textbox.setHtml("\n".join([t for t in text]))
    #     # populate the drop-down lists in the parametertree
    #     ulist["Images"].setValue(imagelist[0])
    #     ulist["Images"].setLimits(imagelist)
    #     ulist["Videos"].setValue(videolist[0])
    #     ulist["Videos"].setLimits(videolist)

    def show_index(self):
        print("self.index: ")
        # print(self.index)
        self.textbox.setCurrentFont(QtGui.QFont("Courier New"))
        text = []  # ["<font color='red' size='3' font-family='monospace'>Hello PyQt5!\nHello"]
        # ["<font-family:'Courier New' color='blue' size='10'>"]
        for k in self.index.keys():
            print("index is: ", k)
            if k.startswith("image"):
                text.append(
                    f" {k:16s}: {self.index[k]['objective']:24s}, {str(self.index[k]['binning']):8s}"
                    + f" {str(self.index[k]['transform']['scale']):s}<br>"
                )
            if k.startswith("video"):
                text.append(
                    f" {k:16s}: {self.index[k]['objective']:24s}, {str(self.index[k]['binning']):8s}"
                    + f" {str(self.index[k]['transform']['scale']):s}<br>"
                )

        self.textbox.setHtml("\n".join([t for t in text]))

    def set_new_objective(self, data):
        self.objdata.to_objective = data

    def update_from_objective(self):
        fns = self.get_imagefilenames(self.objdata)  # get for the current
        print("image file names: ", fns)
        # print('self.index: ', self.index)
        for fn in fns:
            objname = self.index[fn]["objective"]
            if objname not in list(self.objective_data["Objectives"].keys()):
                print(f"Objective {objname:s} not found in list of known objectives.")
                continue
            print("objective(s) found: ", objname)
            self.ptreedata.param("Original Objective").setValue(objname)

    def get_original_objective(self):
        pass

    def view_proposed_changes(self, write=False):
        print("Proposed changes - new objective data: ", self.objdata)
        self.change_scale(self.objdata, write=write)

    def get_imagefilenames(self, objective: object) -> str:
        imagefiles = []
        print("objective img: ", objective.images)
        if objective.images is not None and len(objective.images) > 0:
            imagefiles.append(objective.images)
            images, target = SQP(objective.images)
            print("images: ", images)
            for img in images[0]:
                print("img: ", img)
                imagefiles.append(f"image_{int(img[0]):03d}.tif")
        print("objective.videos: ", objective.videos)
        if objective.videos is not None:
            videos, target = SQP(objective.videos)
            print('Videls: ', videos)
            for vid in videos[0]:
                imagefiles.append(f"video_{int(vid):03d}.ma")
        else:
            pass
        return imagefiles

    def change_scale(self, objective: object = None, write: bool = False):
        print("change scale objective: ", objective)
        imagefiles = self.get_imagefilenames(objective)
        if len(imagefiles) == 0:
            return
        text = []
        text.append(f"Changescale: imagefiles: {imagefiles}")

        text.append(f"\nindex keys:  {self.index.keys()}")
        for imagefile in imagefiles:
            if imagefile not in self.index.keys():
                text.append(f"File {imagefile:s} not found in {str(list(self.index.keys())):s}")
                continue
            text.append(f"Index imagefile:  {imagefile}")
            text.append(f"Old objective: {self.objdata.from_objective}")  # self.index[imagefile]["objective"]
            # pp.pprint(index[imagefile] )
            old_objective = self.index[imagefile]["objective"]
            pp.pprint("   Old transform: ")
            pp.pprint(self.index[imagefile]["transform"])
            pp.pprint("   Old device transform,: ")
            pp.pprint(self.index[imagefile]["deviceTransform"])
            binning = self.index[imagefile]["binning"]
            new_objective = objective.to_objective  # string name
            magnification_new_objective = float(
                self.objective_data["Objectives"][new_objective]
            )  # new magnification
            self.index[imagefile]["transform"]["scale"] = (
                binning[0] * self.refscale[0] / magnification_new_objective,
                binning[1] * self.refscale[1] / magnification_new_objective,
                1.0,
            )
            self.index[imagefile]["deviceTransform"]["scale"] = (
                binning[0] * self.refscale[0] / magnification_new_objective,
                binning[1] * self.refscale[1] / magnification_new_objective,
                1.0,
            )
            d = datetime.datetime.now()
            dstr = d.strftime("%Y-%m-%d %H:%M:%S")
            self.index[imagefile]["objective"] = objective.to_objective
            self.index[imagefile]["note"] = f"Objective scale corrected from {old_objective:s}"
            self.index[imagefile]["note"] += f" to {new_objective:s} on {dstr:s} by PBM"
            print(
                "New objective: ", self.index[imagefile]["objective"]
            )  # pp.pprint(index[imagfile] )
            print("   New transform: ")
            pp.pprint(self.index[imagefile]["transform"])
            print("   New device transform: ")
            pp.pprint(self.index[imagefile]["deviceTransform"])
            print("   Added Note: ", self.index[imagefile]["note"])
            print("----------------------------")

            print("read_index write: ", write)
            index_filename = Path(objective.filename, ".index")
            if write:
                self.rewrite_index(self.index, index_filename)
                CP.cprint("g", ".index file has been updated")

            else:
                print("Dry Run: .index file was NOT modified")
                print(f" file to modify is: {str(index_filename):s}")
        # then update index file display
        text.append(f"\n\nChanges made to {str(imagefiles):s}")
        self.change_text.setHtml("\n".join([t for t in text]))

        self.show_index()

    def build_ptree(self):
        cameras = list(self.objective_data["Cameras"].keys())
        objectives = list(self.objective_data["Objectives"].keys())
        cinescales = list(self.objective_data["CineScales"].keys())
        self.params = [
            # {"name": "Pick Cell", "type": "list", "values": cellvalues, "value": cellvalues[0]},
            {"name": "Set Directory/Protocol", "type": "action"},
            {"name": "Reload Last Protocol", "type": "action"},
            {"name": "Images", "type": "str", "value": ""},
            {"name": "Videos", "type": "str", "value": ""},
            {
                "name": "Camera",
                "type": "list",
                "limits": cameras,
                "value": cameras[0],
                "renamable": False,
            },
            {
                "name": "Cine Scale",
                "type": "list",
                "limits": cinescales,
                "value": cinescales[0],
                "readonly": True,
            },
            {
                "name": "Original Objective",
                "type": "list",
                "limits": objectives,
                "value": objectives[0],
                "renamable": False,
            },
            {
                "name": "New Objective",
                "type": "list",
                "limits": objectives,
                "value": objectives[1],
                "renamable": False,
            },
            {"name": "View .index", "type": "action"},
            {"name": "View proposed changes", "type": "action"},
            {"name": "Apply changes", "type": "action"},
            {'name': "Reload", 'type': 'action'},
            {"name": "Quit", "type": "action"},
        ]
        self.ptree = ParameterTree()
        self.ptreedata = Parameter.create(name="Commands", type="group", children=self.params)

        self.ptree.setParameters(self.ptreedata)
        self.ptree.setMaximumWidth(self.ptreewid)
        self.ptree.setMinimumWidth(self.ptreewid)

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
            elif path[0] == "Reload Last Protocol":
                self.getProtocolDir(reload_last=True)
            elif path[0] == "Camera":
                self.selected_camera = data
                self.update_camera_settings()  # when changing cameras, make sure pixel size is updated
            elif path[0] == "Cine Scale":
                self.selected_cinescale = data
                self.update_camera_settings()  # when changing the cinescale, also make sure pixel size is updated
            elif path[0] == "Videos":
                self.objdata.videos = data
            elif path[0] == "Images":
                self.objdata.images = data
            elif path[0] == "Original Objective":
                self.objdata.from_objective = data
            elif path[0] == "New Objective":
                self.objdata.to_objective = data
                self.update_from_objective()
            elif path[0] == "View proposed changes":
                self.view_proposed_changes(write=False)
            elif path[0] == "Apply changes":
                self.view_proposed_changes(write=True)

            elif path[0] == "Quit":
                self.quit()

    def set_window(self, parent=None):
        super(FixObjective, self).__init__(parent=parent)
        self.win = pg.QtWidgets.QMainWindow()
        self.win.setWindowTitle("FixObjective")
        self.DockArea = PGD.DockArea()
        self.win.setCentralWidget(self.DockArea)
        self.win.resize(1600, 1024)
        self.fullscreen_widget = None
        self.win.setWindowTitle("Model DataTables/FileSelector")
        win_wid = 1024
        win_ht = 512

        self.win.setWindowTitle("No File")

        # self.buttons = pg.QtGui.QGridLayout()
        self.build_ptree()
        # self.buttons.addWidget(self.ptree)

        # Initial Dock Arrangment

        self.Dock_Params = PGD.Dock("Params", size=(self.ptreewid, win_ht))
        self.Dock_Params.addWidget(self.ptree)
        self.Dock_Report = PGD.Dock("Reporting", size=(win_wid - self.ptreewid, win_ht))
        self.Dock_Help = PGD.Dock("Help", size=(win_wid - self.ptreewid, win_ht))
        self.Dock_Changes = PGD.Dock("Changes", size=(win_wid - self.ptreewid, win_ht))

        self.textbox = QtWidgets.QTextEdit()
        self.textbox.setReadOnly(True)
        self.textbox.setText("(.index file)")
        self.Dock_Report.addWidget(self.textbox)
        
        self.help_text = QtWidgets.QTextEdit()
        self.help_text.setReadOnly(True)
        self.help_text.setText("Help")
        
        self.change_text = QtWidgets.QTextEdit()
        self.change_text.setReadOnly(True)
        self.change_text.setText("Changes")
        self.Dock_Changes.addWidget(self.change_text)

        self.Dock_Help.addWidget(self.help_text)
        self.DockArea.addDock(self.Dock_Params, "left")
        self.DockArea.addDock(self.Dock_Report, "right", self.Dock_Params)
        self.DockArea.addDock(self.Dock_Help, "bottom", self.Dock_Report)
        self.DockArea.addDock(self.Dock_Changes, "bottom", self.Dock_Help)
        self.ptreedata.sigTreeStateChanged.connect(self.command_dispatcher)
        self.put_help()
        self.win.show()

    def quit(self):
        exit(0)

    def put_help(self):
        help_text = """
        <h1>Fix Objective</h1>
        <p>This tool is used to correct the objective scale factor in the .index file for a set of images or videos.</p>
        <p>First select the directory that holds the image data you wish to correct by selecting the 'Set Directory/Protocol' button.</p>
        <p>When you have selected this directory, the 'Reporting' window should show the .index file data entries for the images and videos in the directory.</p>
        <p>Identify the images and videos you want to change by using the 'seqparse' format of the corresponding numbers in the 'Images' and 'Videos' fields.</p>
        <p> For example, if you want to change images 1, 2, and 3, you would enter '1;3' in the 'Images' field.
        if you wnat to change images 1, 3, and 5, enter '1,3,5' in the image field. Note that the image and video numbers start at 0.
        </p>
        <p> YOu can then view the proposed changes by selecting the 'View proposed changes' button.</p>
        <p>It is also important to select the correct objective for the data you are working with.</p>
        <p>Once you have selected the camera, cine scale, and the images or videos you wish to correct, you can select the new objective and apply the changes.</p>
        <p> If you are satisfied with the changes, you can apply them by selecting the 'Apply changes' button.</p>
        <p>It is important to select the correct camera and cine scale for the data you are working with.</p>
        <p>It is important to note that the changes are not reversible, so be sure you have selected the correct data and the correct objective before applying the changes.</p>
        <p>Once you have applied the changes, you can view the changes in the .index file by selecting the 'View .index' button.</p>
        <p>Once you have applied the changes, you can quit the program by selecting the 'Quit' button.</p>
        """
        self.help_text.setHtml(help_text)

def main(objective_data):
    app = pg.mkQApp()
    app.setStyle("fusion")
    FO = FixObjective(app=app, objective_data=objective_data)
    app.aboutToQuit.connect(
        FO.quit
    )  # prevent python exception when closing window with system control
    FO.set_window()

    if (sys.flags.interactive != 1) or not hasattr(pg.QtCore, "PYQT_VERSION"):
        pg.QtWidgets.QApplication.instance().exec()


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
    objective_data = get_configuration("ephys/config/fix_objective_data.cfg")
    main(objective_data)
    # FixObjective(objective_data=objective_data)
