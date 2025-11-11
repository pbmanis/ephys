"""
This script is a tool that compares cell type annotations in a DataFrame with the original data stored in Acq4 index files.
It reads a DataFrame from a pickle file, retrieves cell type information from Acq4 index
files, maps both cell types to a standard format, and reports any discrepancies.

Cell types are compared against the known cell types in the map_cell_types module.
If the cell types are discrepant, a message is printed in red to highlight the mismatch.
If run with the --update flag, the original pandas dataframe is updated with the cell type from Acq4
for the FIRST discrepancy found. The updated dataframe is saved back to the same pickle file.
The datasummary excell file is also updated to reflect the changes.
The reason for stopping at the first discrpenency is to allow the user to verify that the updates are correct
before proceeding with further updates.

Using this tool sidesteps the need to completely regenerate the datasummary dataframe from 
scratch, which can be time-consuming. It is meant only to update the metadata in the file,
which sometimes is incorrect or has missing values (this is the top level .index file
for a cell entry in acq4).

Requirements: A configuration file that sets the paths to the raw and analyzed data.
This should be run from the specific analysis directory for the expeirment of interest, 
as ../ephys/ephys/tools/data_summary_celltype_updater.py [--update]

11/6/2025 pbm.

"""
from pathlib import Path

import numpy as np
import pandas as pd
import pylibrary.tools.cprint as CP

from ephys.datareaders import acq4_reader as AR
from ephys.tools import data_summary, get_configuration, map_cell_types

AR = AR.acq4_reader()

datasets, experiments = get_configuration.get_configuration("config/experiments.cfg")
expt = experiments["NF107Ai32_Het"]

data_summary_file = Path(expt["analyzeddatapath"], expt["directory"], expt["datasummaryFilename"])
with open(data_summary_file, "rb") as f:
    df = pd.read_pickle(f)

def replace_empties(row)-> pd.Series:
    """replace any entries with empty cell types from the dataframe
    with cell_type = 'unknown'
    """
    match row['cell_type']:
        case ' '| '' | np.nan | None:
            CP.cprint("m", f"Replacing Cell id {row['cell_id']:>56s} cell type <{row['cell_type']:^s}> with 'unknown' in datasummary.")
            row['cell_type'] = 'unknown'
        case _:
            row['cell_type'] = row['cell_type'].lstrip().rstrip()
    return row

df = df.apply(replace_empties, axis=1)
# # save to same file
# with open(data_summary_file, "wb") as f:
#     df = pd.to_pickle(f)

# print("Cell types: ", df.cell_type.unique().tolist())
# exit()

#####################################
# check every cell in the dataframe against the acq4 index file

def check_celltypes(df: pd.DataFrame):
    """check every cell in the dataframe against the acq4 index file
    and report any discrepancies.
    If --update flag is set, update the dataframe with the acq4 cell type
    for the FIRST discrepancy found.
    """
    ndiffs = 0
    all_acq4_types_found = []
    for icell, row in df.iterrows():
        cell_data = df.loc[icell]
        df_cell_type = str(cell_data['cell_type'])
        # print(f"Cell {icell}: {cell_data['cell_type']}")
        cell_path = Path(expt["rawdatapath"], expt["directory"], cell_data["date"], cell_data["slice_slice"], cell_data["cell_cell"])
        if not cell_path.exists():
            color = "r"
        else:
            color = "w"  
        AR.setProtocol(cell_path)
        index = AR.readDirIndex(cell_path)
        acq4_celltype = index['.'].get('type', " ")
        if acq4_celltype in [' ', '', None]:
            acq4_celltype = 'unknown'
        # convert cell types in acq4 and datasummary to mapped (consistent) versions, and compare
        mapped_acq4_celltype = map_cell_types.map_cell_type(acq4_celltype)
        all_acq4_types_found.append(mapped_acq4_celltype)
        mapped_df_celltype = map_cell_types.map_cell_type(cell_data['cell_type'])
        # if they don't match, then let's inform the user.
        if df_cell_type != mapped_acq4_celltype:
            CP.cprint("r", f"Cell {icell}: MISMATCH: DataFrame cell type: {df_cell_type}/ {mapped_df_celltype}, Acq4 cell type: <{acq4_celltype!s}>/ {mapped_acq4_celltype}, {df.loc[icell].cell_id!s}")
            print(f"<{df_cell_type:s}> != <{acq4_celltype!s}>")
            print(f"<{mapped_df_celltype}> != <{mapped_acq4_celltype:s}>")
            # if cell_data['cell_type'].lower().lstrip() == acq4_celltype.lower():  # types match, so update the dataframe to use the matched type
            #     df.loc[icell, 'cell_type'] = mapped_acq4_celltype
            #     ndiffs += 1
            #     CP.cprint("g", f"    UPDATED DataFrame cell type <{df_cell_type:s}> to: <{mapped_acq4_celltype:s}> for cell id: {df.loc[icell].cell_id!s}")
            # if mapped_df_celltype != mapped_acq4_celltype:
            df.loc[icell, 'cell_type'] = mapped_acq4_celltype
            ndiffs += 1
            CP.cprint("y", f"    UPDATED DataFrame cell type <{mapped_df_celltype:s}> to: <{mapped_acq4_celltype:s}> for cell id: {df.loc[icell].cell_id!s}")

    return df, ndiffs, all_acq4_types_found

# len original data frame:
CP.cprint("b", f"Original DataFrame has {len(df)} entries.")
# first pass - may produce notifications!

df, ndiffs, all_acq4_types_found = check_celltypes(df)
CP.cprint("b", f"First pass found {ndiffs} discrepancies between DataFrame and Acq4 index files.")
print(f"\n{'-'*60}\nSecond pass to verify no discrepancies remain:\n{'-'*60}\n")
# do it again to be sure that all discrepancies are resolved
df, ndiffs, all_acq4_types_found = check_celltypes(df)
if ndiffs == 0:
    CP.cprint("g", f"No discrepancies found between DataFrame and Acq4 index files. # entries: {len(df)}")
else:
    CP.cprint("r", f"{ndiffs} discrepancies remain between DataFrame and Acq4 index files.")    

print("Unique cell types found in MAPPED Acq4 index files: ", set(all_acq4_types_found))
# if len(df) > 0:
#     # save to same file
#     with open(data_summary_file, "wb") as f:
#         df.to_pickle(f)
#     CP.cprint("g", f"Updated DataFrame saved to {data_summary_file!s}")

#     # if we get past all that, then update the datasummary excel file
#     ds = data_summary.DataSummary()
#     ds.make_excel(df, outfile=data_summary_file.with_suffix('.xlsx'))
#     print(df.cell_type.unique().tolist())
# else:
#     CP.cprint("r", "DataFrame is empty, not overwriting updated file.")