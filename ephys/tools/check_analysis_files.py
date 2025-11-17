""" check analysis files
The map and IV analysis data files are stored in subdirectories by cell type.
When cell types are reassigned/changed, these may no longer be in the correct locations.
With this tool, we compare the files in each of the cell-specific directories with
the type of the cell that is currently assigned. 
We then print out either the correct assignments, or the mis-typed assignments.

No intermediate analysis result files are changed by this tool; it only reports discrepancies.

"""
from unittest import case
import numpy as np
import pandas as pd
from pathlib import Path
import ephys.datareaders.acq4_reader as AR # maybe not needed?
from ephys.tools import get_configuration, map_cell_types, filename_tools
from pylibrary.tools import cprint as CP
import datetime


datasets, experiments = get_configuration.get_configuration("config/experiments.cfg")
expt = experiments["NF107Ai32_Het"]

data_summary_file = Path(expt["analyzeddatapath"], expt["directory"], expt["datasummaryFilename"])

cell_dirs = [v[1] for v in map_cell_types.all_types]
print("Cell directories to check: ", cell_dirs)

datasets, experiments = get_configuration.get_configuration("config/experiments.cfg")
expt = experiments["NF107Ai32_Het"]

data_summary_file = Path(expt["analyzeddatapath"], expt["directory"], expt["datasummaryFilename"])
with open(data_summary_file, "rb") as f:
    df = pd.read_pickle(f)
analyzed_dir = Path(expt["analyzeddatapath"], expt["directory"])
event_dir = Path(analyzed_dir, 'events')
event_files = event_dir.glob(f"*.pkl")

excluded_maps = expt.get("excludeMaps", {})

datatype = "IVs"  # could be "map" or "IVs"

print(len(df))
counts = {ct: 0 for ct in cell_dirs}
for icell, row in df.iterrows():
    cell_type = row['cell_type']
    if cell_type in counts:
        counts[cell_type] += 1
    else:
        counts[cell_type] = 1   
print("Cell type counts in datasummary:")
total = 0
for ct, count in counts.items():
    print(f"   {ct:20s}: {count:d}")
    total += count
print(f"   {'Total':20s}: {total:d}")

# for each major cell type directory, check for event files
major_types = ["bushy", "t-stellate", "d-stellate", "octopus", "pyramidal", "cartweel", "tuberculoventral"]

CP.cprint("y", "\nNow checking event analysis files:\n")
for icell, row in df.iterrows():
    cell_type = row['cell_type']
    if cell_type not in major_types:
        continue
    cell_id = row['cell_id']
    if cell_id in excluded_maps.keys():
        continue
    event_file = filename_tools.make_event_filename_from_cellid(cell_id)
    event_path = Path(event_dir, event_file)
    if not event_path.exists():
        CP.cprint("r", f"Cell id {cell_id} of type {cell_type} missing event file {event_file} in directory {event_dir.name!s}  *****")
    else:
        mod_time = event_path.stat().st_mtime
        ts = datetime.datetime.fromtimestamp(mod_time).strftime('%Y-%m-%d %H:%M:%S')
        CP.cprint("g", f"Cell id {cell_id} of type {cell_type} has event file {event_file} analyzed on {ts:s} {u'\u2713':s}")

CP.cprint("y", "\nNow checking MAP analysis files:\n")
# for each major cell type directory, check for map analysis files
for cell_type in major_types:
    for icell, row in df.iterrows():
        if row['cell_type'] != cell_type:
            continue
        cell_id = row['cell_id']
        if cell_id in excluded_maps.keys():
            continue
        map_file, mapped_cell_type = filename_tools.get_pickle_filename_from_row(row, mode="maps", map_cell_name=True)
        map_file = str(map_file).replace(".pkl", ".pdf")
        map_dir = Path(analyzed_dir, mapped_cell_type)
        map_path = Path(map_dir, map_file)
        if not map_path.exists():
            CP.cprint("r", f"Cell id {cell_id:>56s} of type {mapped_cell_type:<16s} (from cell_type:{cell_type:<16s}) missing map file {map_file} in directory {map_dir.name!s}  *****")
        else:
            mod_time = map_path.stat().st_mtime
            ts = datetime.datetime.fromtimestamp(mod_time).strftime('%Y-%m-%d %H:%M:%S')
            CP.cprint("c", f"Cell id {cell_id:>56s} of type {mapped_cell_type:<16s} (from cell_type:{cell_type:<16s}) has map file {map_file} analyzed on {ts:s} {u'\u2713':s}")
exit()

    # print out the cells that have no assigned cell type - may have no data.

