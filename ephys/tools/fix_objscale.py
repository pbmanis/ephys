"""
Fix the objective scale factor
Give "stated objective", "actual objective"

('objective', '4x 0.1na ACHROPLAN')

"""
from dataclasses import dataclass
from typing import Union
import numpy as np
import argparse
from pathlib import Path
from pyqtgraph import configfile
import pprint
from collections import OrderedDict
import datetime
from pylibrary.tools.fileselector import FileSelector
import pylibrary.tools.cprint as CP

pp = pprint.PrettyPrinter(indent=4)


CineScale = 1.0
refscale = [(1.0 / CineScale) * 6.54e-6, -(1.0 / CineScale) * 6.54e-6]
sfactors = [4.0, 10.0, 20.0, 40.0, 63.0]

objectiveList = {
    4: "4x 0.1na ACHROPLAN",
    10: "10x 0.3na W N-ACHROPLAN",
    20: "20x 0.5na W N-ACHROPLAN",
    40: "40x 0.8na ACHROPLAN",
    63: "63x 0.9na ACHROPLAN",
}


bp = "/Volumes/Pegasus/ManisLab_Data3/Kasten_Michael/NF107Ai32Het"
bp = (
    "/Volumes/Pegasus_002/ManisLab_Data3/Kasten_Michael/HK_collab_ICinj/Thalamocortical"
)
bp = "/Volumes/Pegasus_002/ManisLab_Data3/Kasten_Michael/HK_collab_ICinj/DCN_IC_inj"


@dataclass
class Changer:
    change_type: str = "new"
    filename: str = ""
    from_objective: int = 40
    to_objective: int = 10
    videos: Union[str, None] = None
    images: Union[str, None] = None


objdata = Changer(filename="2021.04.27_000/slice_003/cell_001", videos="0")

changeList = [objdata]


def rewrite_index(index: dict, index_file: Union[str, Path]):
    configfile.writeConfigFile(index, index_file)


def read_indexes(index_path, changelist: list = [], write: bool = False):
    print("write: ", write)
    for objective in changeList:
        index = read_index(index_path, objective, write=write)
        

def read_index(index_path: Union[str, Path], objective: object = None, write: bool = False):
    print("\nfix_objscale: We will be using the following reference scale: ", refscale)
    print("   This scale may be specific to your camera!!!!!")
    print("read_index write flag is: ", write)
    nl = "\n"
    print(f"Objective: ")
    print(f"    Change Type: {objective.change_type:s}")
    print(f"    File: {objective.filename:s}")
    print(f"    From: {objective.from_objective:3d}X to {objective.to_objective:3d}X")
    print(f"    For images: {str(objective.images):s}")
    print(f"    For videos: {str(objective.videos):s}")
    print("="*40)
    d = datetime.datetime.now()
    dstr = d.strftime("%Y-%m-%d %H:%M:%S")
    index_file = Path(index_path, objective.filename, ".index")
    index = configfile.readConfigFile(index_file)
    return index

def change_scale(objective:object=None, write:bool=False):
    if objective.images is not None:
        imagefile = f"image_{int(obj.images):03d}.tif"
    elif objective.videos is not None:
        imagefile = f"video_{int(obj.videos):03d}.ma"
    print("\n----------------------------")
    if imagefile not in index.keys():
        CP.cprint("m", f"File {imagefile:s} not found in {str(list(index.keys())):s}")
        return
    k = imagefile
    print("Index: ", k)
    print("Old objective: ", index[k]["objective"])  # pp.pprint(index[k] )
    oldobjective = index[k]["objective"]
    pp.pprint("   Old transform: ")
    pp.pprint(index[k]["transform"])
    pp.pprint("   Old device transform,: ")
    pp.pprint(index[k]["deviceTransform"])
    binning = index[k]["binning"]
    new_objective = objective.to_objective
    f_new_objective = float(new_objective)
    index[k]["transform"]["scale"] = (
        binning[0] * refscale[0] / fnewobj,
        binning[1] * refscale[1] / fnewobj,
        1.0,
    )
    index[k]["deviceTransform"]["scale"] = (
        binning[0] * refscale[0] / f_new_objective,
        binning[1] * refscale[1] / f_new_objective,
        1.0,
    )
    index[k]["objective"] = objectiveList[obj.to_obj]
    index[k][
        "note"
    ] = f"Objective scale corrected from {oldobj:s} to {objlist[newobj]:s} on {dstr:s} by PBM"
    print("New objective: ", index[k]["objective"])  # pp.pprint(index[k] )
    print("   New transform: ")
    pp.pprint(index[k]["transform"])
    print("   New device transform: ")
    pp.pprint(index[k]["deviceTransform"])
    print("   Added Note: ", index[k]["note"])
    print("----------------------------")

    print("read_index write: ", write)
    if write:
        rewrite_index(index, index_file)
        CP.cprint("g", ".index file has been updated")

    else:
        print("Dry Run: .index file was NOT modified")


def main():
    parser = argparse.ArgumentParser(description="Fix the objective")
    parser.add_argument(
        "-w",
        "--write",
        action="store_true",
        dest="write",
        help="Rewrite the .index file (otherwise, we just do a dry run)",
    )

    args = parser.parse_args()

    read_indexes(bp, changeList, args.write)


if __name__ == "__main__":
    main()
