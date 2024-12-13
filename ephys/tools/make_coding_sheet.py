""" List the days, animal ids, slices and mosaics

This table is used to keep track of the slice and the mosaices
created, and the coding of the cells in the slice.

The table can be used to drive specified analyses later.

"""

import json
from pathlib import Path
import pandas as pd
import re
import textwrap
from ephys.datareaders import acq4_reader
from pylibrary.tools import cprint
from ephys.tools import get_configuration
from ephys.tools import write_excel_sheet
from pprint import PrettyPrinter
import deepdiff
from collections import OrderedDict
import shutil

from ephys.gui import data_table_functions as functions

FUNCS = functions.Functions()
PP = PrettyPrinter(indent=4)

CP = cprint.cprint
AR = acq4_reader.acq4_reader()

day_re = re.compile(r"(\d{4}).(\d{2}).(\d{2})_(\d{3})")
slice_re = re.compile(r"slice_(\d{3})")
cell_re = re.compile(r"cell_(\d{3})")

exptname = "DCN_IC_inj"

config = get_configuration.get_configuration()[1]
# print(config.keys())
experiment = config[exptname]


def make_coding_sheet(experiment):
    # print(experiment)
    excelfilename = Path(experiment["analyzeddatapath"], exptname, experiment["coding_file"])
    # if excelfilename.exists():
    #     print(f"File {excelfilename} already exists. Not overwriting.")
    #     exit()
        
    datasummary = FUNCS.get_datasummary(experiment)

    coding_columns = [
        "animal_identifier",
        "strain",
        "reporters",
        "date",
        "slice_slice",
        "cell_cell",
        "mosaic",
        "coding",

    ]
    # print(datasummary.columns)
    sliceonly = False # set True to only code by slices, not cells.
    coding = pd.DataFrame(columns=coding_columns)
    for index in datasummary.index:
        ai = datasummary.loc[index, "animal_identifier"]
        date = datasummary.loc[index, "date"]
        strain = datasummary.loc[index, "strain"]
        reporters = datasummary.loc[index, "reporters"]
        # look for all slices in the date, regardless of whether they are in the datasummary or not
        datepath = Path(experiment["rawdatapath"], exptname, date)
        slices = list(datepath.glob("slice_*"))
        # sliceno = datasummary.loc[index, 'slice_slice']
        for sliceno in slices:
            slicepath = Path(experiment["rawdatapath"], exptname, date, sliceno)
            if slicepath.is_dir():
                allfiledir = list(slicepath.glob("*"))
                if len(allfiledir) == 1 and allfiledir[0].name == ".index":
                    mosaic = "No Images"
                else:
                    mosaicfiles = list(slicepath.glob("*.mosaic"))
                    if len(mosaicfiles) > 0:
                        mosaic = Path(mosaicfiles[0].name)
                    else:
                        mosaic = ""
                if sliceonly and (date in coding["date"].values):
                    thisday = coding[coding["date"] == date]
                    if slicepath.name in thisday["slice_slice"].values:
                        print(f"Date: {date} and {sliceno} are already in coding sheet")
                        continue
                    coding = coding._append(
                        {
                            "animal_identifier": ai,
                            "strain": strain,
                            "reporters": reporters,
                            "date": date,
                            "slice_slice": slicepath.name,
                            "cell_cell": "",
                            "mosaic": mosaic,
                            "coding": "",
                        },
                        ignore_index=True,
                    )
                if not sliceonly:
                    cells = list(slicepath.glob("cell_*"))
                    for cellno in cells:
                        cellpath = Path(slicepath, cellno)
                        coding = coding._append(
                            {
                                "animal identifier": ai,
                                "strain": strain,
                                "reporters": reporters,
                                "date": date,
                                "slice_slice": slicepath.name,
                                "cell_cell": cellpath.name,
                                "coding": "",
                                "mosaic": mosaic,

                            },
                            ignore_index=True,
                        )

    print(coding.head(20))
    sheet_name = experiment["coding_sheet"]
    excelfilename = Path(experiment["analyzeddatapath"], exptname, experiment["coding_file"])
    CE = write_excel_sheet.ColorExcel()
    CE.make_excel(
        coding,
        outfile=excelfilename,
        sheetname=sheet_name,
        columns=coding_columns,
    )

    print("Coding Sheet written to: ", excelfilename)


def collect_mosaic_files(experiment):
    datasummary = FUNCS.get_datasummary(experiment)
    target = Path(Path.cwd(), "mosaics")
    if not target.exists():
        target.mkdir()

    for index in datasummary.index:  # this is every cell in the database    
        ai = datasummary.loc[index, "animal_identifier"]
        date = datasummary.loc[index, "date"]
        strain = datasummary.loc[index, "strain"]
        reporters = datasummary.loc[index, "reporters"]
        layer = datasummary.loc[index, "cell_layer"]
        cell_cell = datasummary.loc[index, "cell_cell"]
        # look for all slices in the date, regardless of whether they are in the datasummary or not
        datepath = Path(experiment["rawdatapath"], exptname, date)
        slices = list(datepath.glob("slice_*"))
        # sliceno = datasummary.loc[index, 'slice_slice']
        for sliceno in slices:
            slicepath = Path(experiment["rawdatapath"], exptname, date, sliceno)
            if slicepath.is_dir():
                allfiledir = list(slicepath.glob("*"))
                if len(allfiledir) == 1 and allfiledir[0].name == ".index":
                    mosaic = "No Images"
                else:
                    mosaicfiles = list(slicepath.glob("*.mosaic"))
                    if len(mosaicfiles) > 0:
                        mosaic = Path(mosaicfiles[0].name)
                    else:
                        mosaic = ""
                if mosaic and mosaic != "No Images":
                    src = Path(slicepath, mosaic)
                    dst = Path(target, mosaic)
                    if not dst.exists():
                        print("would copy: ", src, " to ", dst)
                        shutil.copy(src, dst)
                    else:
                        print(f"File {dst} already exists")
    # tar the target directory
    tarfile = Path(target, "datasets", experiment["directory"], "mosaics.tar")
    shutil.make_archive(target, "tar", target)
    print(f"Tar file: {tarfile} created.")

def compare_dir_structure(experiment, compdir):

    """ compare the directory structure of the experiment with the compdir
    Walks two directory trees and compares them. Returns a dictionary of differences.
    The experiment (reference directory) is driven from the datasummary file.
    comp_diff will hold the directores that are either absent in the compdir,
    or which are in the compdir but not the datasummary.
    All directory entries are compared at the day level, and the slice level, 
    so as to capture subdirs for images (e.g. "TRITC", "FITC", etc),
    """
    assert Path(compdir).is_dir()
    datasummary = FUNCS.get_datasummary(experiment)
    toppath = Path(experiment["rawdatapath"], exptname)
    comppath = Path(compdir)
    print("Reference:  ", toppath)
    print("Comparison: ", comppath)
    comp_diff = {"missing": [], "extra": []}
    dates = []
    for index in datasummary.index:
        date =datasummary.loc[index, "date"]
        rig = Path(date).parent.name
        if date in dates:
            continue # don't look repeatedly at a date
        # look for all slices in the date, regardless of whether they are in the datasummary or not
        datepath = Path(experiment["rawdatapath"], exptname, date)
        compdate = Path(compdir, date)
        if datepath.is_dir():
            CP("w", f"primary date {date!s} on {rig:s}  exists")
            dates.append(date)
        else:
            CP("r", f"primary date {date!s} on {rig:s}  does not exist")
        if compdate.is_dir():
            CP("w", f"compare date {date!s} on {rig:s}  exists in comparison directory")
        else:
            CP("r", f"compare date {date!s} on {rig:s}  does not exist in comparison directory")
            comp_diff['missing'].append(str(Path(date)))
            continue
        # Look at all the slice dirs for the date
        slices = list(datepath.glob("*"))
        # sliceno = datasummary.loc[index, 'slice_slice']
        for sliceno in slices:
            slicename = sliceno.name
            if not slicename.startswith("slice_") or sliceno.suffix == ".png":
                continue
            slicepath = Path(experiment["rawdatapath"], exptname, date, slicename)
            compslicepath = Path(compdir, date, slicename)
            if slicepath.is_dir():
                CP("w", f"    primary slice {slicename!s} exists")
            else:
                CP("r", f"    primary slice {slicename!s} does not exist")
                comp_diff['missing'].append(str(sliceno))
                continue
            if compslicepath.is_dir():
                CP("w", f"    compare slice {slicename!s} exists in comparison directory")
            else:    
                CP("r", f"    compare slice {sliceno!s} does not exist in comparison directory")
                comp_diff['missing'].append(str(sliceno))
                continue
            # Now look at the contents of the slice directory: cells, image directories, etc.
            allfiles = list(slicepath.glob("*"))
            compallfiles = list(compslicepath.glob("*"))
            for file in allfiles:
                if not file.is_dir():
                    continue
                    # look in the compdir for the same directory
                compfile = Path(compslicepath, file.name)
                if compfile.is_dir():
                    CP("w", f"            primary dir {file!s} exists")
                else:
                    CP("r", f"            primary dir {compfile!s} does not exist")
                    comp_diff['missing'].append(str(compfile))
            # if slicepath.is_dir():
            #     allfiledir = list(slicepath.glob("*"))
            #     if len(allfiledir) == 1:  # don't consider .index files
            #         continue
            #     comp_slicepath = Path(comppath, date, sliceno)
            #     if not comp_slicepath.exists():
            #         comp_diff['missing'] = str(Path(date), sliceno)


                
    print("\nMissing in comparison data: ")
    for m in comp_diff['missing']:
        CP("r", m)



if __name__ == "__main__":
    make_coding_sheet(experiment)
    # collect_mosaic_files(experiment)
    # fp2 = "/Volumes/Pegasus_002/ManisLab_Data3/Kasten_Michael/HK_Collab/Thalamocortical"
    # compare_dir_structure(experiment, fp2)
    # print("Done.")
