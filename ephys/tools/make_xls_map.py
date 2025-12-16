"""
This script makes an excel file from the .pkl file.
The excel file is a "temporary" file with some default values
for analysis parameters. Typically this file will be used to seed
the main excel file for that is used to gather parameters for
the map analysis (and event deetection). The main excel file
will also have annotations that are not present in this file. 
This file can be used to update the main file, by copying the new
lines into the main file. 

Basically, this script reads through a pkl pandas file.
For each cell in the original file that corresponds to a map protocol,
it find the images in that file.
Amongst the images, it finds the best image in the cell
directory (brightest, in this case). 
It then writes the database with the map file and the image file(s)
 indicated in new columns,
with a separate row for each map.

Use the results of nf107_maps (the pdf output file) to identify
issues and manually correct the images xlsx file

pbm 8/6/2018-11/2021
last revision 8/28/2023

"""

import argparse
import dataclasses
from typing import Union, Tuple, List
from dataclasses import dataclass
from pathlib import Path
import re
from typing import Union

import ephys.datareaders as DR
import ephys.tools.write_excel_sheet as write_excel_sheet
import ephys.tools.get_configuration as get_configuration
import numpy as np
import pandas as pd
from pylibrary.tools import cprint as CP

re_protocol = re.compile(r"^(?P<protocol>[a-z_0-9]+)_[0-9]{0,3}[\_0-9]{0,4}[\_a-z]*[a-z0-9\_]*$", re.IGNORECASE)


datasets, experiments = get_configuration.get_configuration("config/experiments.cfg")
print("known datasets:  ", datasets)

def strip_protocol(smap:str)->str:
    m1= re_protocol.match(smap)
    if m1 is None:
        raise ValueError(f"Could not parse protocol from map name: {smap:s}")    
    m2 = re_protocol.match(m1.group("protocol"))
    if m2 is not None:
        return m2.group("protocol") 
    elif m1 is not None:
        return m1.group("protocol")
    else:
        return None

# testp = "VGAT_5mspulses_5Hz_pt045_to_pt1_000_001"
# print(testp, strip_protocol(testp))
# testp = "Map_VGAT_5mspulses_5Hz_pt045_to_pt1_000"
# print(testp, strip_protocol(testp))
# exit()

def test_maps(smap, maps):
    for m in maps:
        if smap.startswith(m):
            return True
    return False

cprint = CP.cprint


# This class holds the column values for one row.
@dataclass
class NewRow:
    date: str = ""#  = d.iloc[index]["date"]
    slice_slice: str = "" # "] = Path(d.iloc[index]["slice_slice"]).parts[-1]
    cell_cell: str = ""# "] = Path(d.iloc[index]["cell_cell"]).parts[-1]
    mapname: str = ""# "] = smap
    data_directory: str = "" # the baseDataDirectory item from set_expt_paths.
    reference_image: str = ""# "] = bestimage
    tau1: float = 0.5 # fitting/convolution initial parameters (filled in by cell type)
    tau2: float = 2.0 # fitting/convolution initial parameter (filled in by cell type)
    threshold: float = 5.0 # fitting/convolution initial parameter (filled in by cell type)
    cc_tau1: float = 1.5
    cc_tau2: float = 3.5
    cc_threshold: float = 2.0
    spike_thr: float=-0.020
    LPF: float=3000.0
    HPF: float=0.0
    Notch: str = "np.arange(60,4000,60)"
    NotchQ: str = "90"
    Detrend: str = ""
    cell_type: str ="" #  = d.iloc[index]["cell_type"].lower()
    Notes: str = ""
    artifact_epoch: str = ""
    use_artifact_file: str = ""
    artScale: str= ""
    Usable: str = "y"
    notes2: str = ""
    subdirectory:str = ""
    stimdur: str = ""
    fl_tau1: str = ""  # flipped sign tau
    fl_tau2: str = ""  # flipped sign tau
    fl_threshold: str = ""  # flipped sign tau
    alt1_tau1: str = ""  # flipped sign tau
    alt1_tau2: str = ""  # flipped sign tau
    alt1_threshold: str = ""  # flipped sign tau
    alt2_tau1: str = ""  # flipped sign tau
    alt2_tau2: str = ""  # flipped sign tau
    alt2_threshold: str = ""  # flipped sign tau

def get_initial_values(celltype:str)->dict:
    values = {"bushy": {"tau1": 0.2, "tau2": 0.5, "threshold": 5.0},
              "octopus": {"tau1": 0.2, "tau2": 0.5, "threshold": 5.0},
              "d-stellate": {"tau1": 0.2, "tau2": 1, "threshold": 5.0},
              "t-stellate": {"tau1": 0.2, "tau2": 1.5, "threshold": 5.0},
              "pyramidal": {"tau1": 0.5, "tau2": 2.5, "threshold": 5.0},
              "giant": {"tau1": 0.5, "tau2": 2.5, "threshold": 5.0},
              "tuberculoventral": {"tau1": 0.2, "tau2": 0.5, "threshold": 5.0},
              "cartwheel": {"tau1": 1.0, "tau2": 3.0, "threshold": 5.0},
              "default": {"tau1": 0.25, "tau2": 1.5, "threshold": 5.0},
              }
    if celltype.lower() in values.keys():
        return values[celltype.lower()]
    else:
        return values["default"]

def get_best_image(AR, supindex, dpath):
    maptimes = []
    mapnames = []
    imagetimes = []
    imagenames = []
    bestimage = ""
    # look for the "best" image
    for k in supindex:
        if k.startswith("image_"):
            # print('Found Image: ', k)
            imagetimes.append(supindex[k]["__timestamp__"])
            imagenames.append(k)
        if k.startswith("Map_"):
            maptimes.append(supindex[k]["__timestamp__"])
            mapnames.append(k)
            brightest = 0
            bestimage = ""
            for imno, im in enumerate(imagenames):
                imgd = AR.getImage(Path(dpath, im))
                avgbright = np.mean(np.mean(imgd, axis=1), axis=0)
                if avgbright > brightest:
                    brightest = avgbright
                    bestimage = im
    return bestimage


def do_one_disk(d, disk:Union[Path, str, None] = None, directory:Union[Path, str, None] = None, 
                experiment:dict=None, db:object=None, args:object=None)->object:
    # now read through the data itself
    AR = DR.acq4_reader.acq4_reader()
    for index in range(d.shape[0]):
        if args.day != "all":  # specified day
            day = str(args.day)
            if "_" not in day:
                day = day + "_000"
            day_x = d.iloc[index]["date"]
            if day_x != day:
                continue
            print(" dayx: ", day_x)
            cprint("yellow", f"found at day: {day:s}")

        maps = d.iloc[index]["data_complete"].split(", ")
        thisrow = d.iloc[index]

        dpath = Path(
            disk,
            directory,
            thisrow["date"],
            thisrow["slice_slice"],
            thisrow["cell_cell"],
        )
        if not dpath.is_dir():
            dpath = Path(
                disk,
                directory,
                thisrow["date"],
                thisrow["slice_slice"],
                thisrow["cell_cell"],
            )
            if not dpath.is_dir():
                cprint("r", f"!!! Directory was not found on disk: {str(dpath):s}")
                continue
        print("\nData Directory:: ", dpath)
        supindex = AR.readDirIndex(currdir=dpath)
        if supindex is None:
            cprint("red", "Continuing.. no .index found")
            continue
        bestimage = get_best_image(AR, supindex, dpath)

        # build new rows in the dataframe for each map protocol
        # of a known type 
        for smapP in maps:
            smap = Path(smapP).parts[-1]  # get just the protocol
            if len(smap) <= 1 or smap.startswith("CCIV_") or smap.startswith("VCIV_") or "test" in smap or "IC_single" in smap:
                continue

            cell_type =d.iloc[index]["cell_type"].lower()
            newrow = NewRow()
            newrow.date = Path(d.iloc[index]["date"]).name
            newrow.slice_slice = Path(d.iloc[index]["slice_slice"]).parts[-1]
            newrow.cell_cell = Path(d.iloc[index]["cell_cell"]).parts[-1]
            newrow.mapname = smap
            newrow.data_directory = Path(d.iloc[index]["date"]).parent
            newrow.reference_image = bestimage
            values = get_initial_values(celltype=cell_type)
            newrow.tau1 = values['tau1']
            newrow.tau2 =  values['tau2']
            newrow.threshold = values['threshold']
            newrow.cc_tau1 = 1.5
            newrow.cc_tau2 = 3.5
            newrow.cc_threshold = 2.0
            newrow.cell_type = cell_type
            newrow.Notch= "np.arange(60,4000,60)"
            newrow.NotchQ= "90"
            newrow.Detrend= ""

            newrow.Notes = ""
            newrow.artifact_epoch = ""
            newrow.use_artifact_file = ""
            newrow.artScale = ""
            newrow.notes2 = ""
            newrow.stimdur = ""
            newrow.fl_tau1 = ""  # flipped sign tau
            newrow.fl_tau2 = ""  # flipped sign tau
            newrow.fl_threshold = ""  # flipped sign tau
            newrow.alt1_tau1 = ""  # flipped sign tau
            newrow.alt1_tau2 = ""  # flipped sign tau
            newrow.alt1_threshold = ""  # flipped sign tau
            newrow.alt2_tau1 = ""  # flipped sign tau
            newrow.alt2_tau2 = ""  # flipped sign tau
            newrow.alt2_threshold = ""  # flipped sign tau

            newrow.Usable = 'n'
            if test_maps(smap, experiment['protocols']["Maps"]):
                newrow.Usable = "y"
                # print("   Marking map as usable\n")
            else:
                cprint("yellow", f"   Map <{smap:s}> not in known map protocols for this experiment")
                cprint("yellow", "   Add protocol to the experiment configuration file if appropriate")
                cprint("yellow", "   Otherwise, edit the exclusion criteria above in the code")

                raise ValueError(f"Map {smap:s} not in known map protocols for this experiment")

            # db = pd.concat([db, pd.from_dict(dataclasses.asdict(newrow))], ignore_index=True)
            db = pd.concat([db, pd.DataFrame([newrow])], ignore_index=True)
    return db

def parse_database(args):
    experimentname = args.experiment
    print("datasets: ", datasets)
    print("experimentname: ", experimentname)
    # make filenames
    if experimentname not in datasets:
        cprint('r', f"Experiment name {experimentname:s} not found in configuration")
        return
    experiment = experiments[experimentname]
    input_fn = Path(
        experiment['analyzeddatapath'],
        experiment['directory'],
        experiment["datasummaryFilename"],
    ).with_suffix(".pkl")
    print(experiment)
    output_fn = Path(
        experiment['analyzeddatapath'],
        experiment['directory'],
        experiment["map_annotationFilename"],
    ).with_suffix(".xlsx")
    # output_fn2 = Path(
    #     resultdisk,
    #     experiments[experimentname]["maps"] + "_tmp2",
    # ).with_suffix(".xlsx")
    # read the .pkl file
    cprint('c', f"Reading main data frame from: {str(input_fn):s}")
    d = pd.read_pickle(input_fn)  # read the pandas data
    print(d.columns)
    for i in d.index:
        if isinstance(d.iloc[i]['data_directory'], str):
            print(f"{d.iloc[i]['data_directory']:120s}")

    # create an empty dataframe and create columns from the dataclass NewRow
    d2 = (
        pd.DataFrame()
    )  # d.loc[:, ['date', 'slice_slice', 'cell_cell', 'data_complete']]
    newd = NewRow()
    d2keys = list(dataclasses.asdict(newd).keys())
    for k in d2.keys():
        d2[k] = newd[k]
    d2 = do_one_disk(d, disk=experiment["rawdatapath"], directory=experiment["directory"], 
                     experiment = experiment, db=d2, args=args)

    print("Writing: ", output_fn)
    EXC = write_excel_sheet.ColorExcel()
    EXC.make_excel(df=d2, outfile=output_fn, sheetname="Sheet1")
    # d2.to_excel(output_fn, columns=d2keys)  # column parameter forces order of fields in excel file


def main():
    """
    Handle command line paremeters: 
        - the name of the experiment, 
        - possibly just a day and/or a slice/cell from that day
    """
    parser = argparse.ArgumentParser(
        description="Mapping data analysis: make excel table with maps"
    )
    parser.add_argument(
        "-E",
        "--experiment",
        type=str,
        dest="experiment",
        # choices=list(set_expt_paths.experiments.keys()),
        default="None",
        nargs="?",
        const="None",
        help="Select Experiment to analyze",
    )
    parser.add_argument("-d", "--day", type=str, default="all", help="day for analysis")
    parser.add_argument(
        "-s",
        "--slice",
        type=str,
        default="",
        dest="slicecell",
        help="select slice/cell for analysis: in format: S0C1 for slice_000 cell_001\n"
        + "or S0 for all cells in slice 0",
    )

    args = parser.parse_args()

    parse_database(args)


if __name__ == "__main__":
    main()
