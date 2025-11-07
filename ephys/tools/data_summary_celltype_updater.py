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
import pandas as pd
from pathlib import Path
from ephys.tools import data_summary
from ephys.datareaders import acq4_reader as AR
from ephys.tools import get_configuration, map_cell_types
import pylibrary.tools.cprint as CP

AR = AR.acq4_reader()

datasets, experiments = get_configuration.get_configuration("config/experiments.cfg")
expt = experiments["NF107Ai32_Het"]

data_summary_file = Path(expt["analyzeddatapath"], expt["directory"], expt["datasummaryFilename"])

with open(data_summary_file, "rb") as f:
    df = pd.read_pickle(f)


for icell in df.index:
    cell_data = df.loc[icell]
    df_cell_type = str(cell_data['cell_type'])
    # print(f"Cell {icell}: {cell_data['cell_type']}")
    cell_path = Path(expt["rawdatapath"], expt["directory"], cell_data["date"], cell_data["slice_slice"], cell_data["cell_cell"])
    if not cell_path.exists():
        color = "r"
    else:
        color = "w"
    # CP.cprint(color, f"  Cell path:  {cell_path!s}, {cell_path.exists()!s}")
    AR.setProtocol(cell_path)
    index = AR.readDirIndex(cell_path)
    acq4_celltype = index['.'].get('type', " ")
    # if acq4_celltype == "Missing":
    #     CP.cprint("r", f"    MISSING cell type in Acq4 index for cell path: {cell_path!s}")
    #     exit()
    mapped_acq4_celltype = map_cell_types.map_cell_type(acq4_celltype)
    mapped_df_celltype = map_cell_types.map_cell_type(cell_data['cell_type'])
    print(f"Cell {icell}: DataFrame cell type: {cell_data['cell_type']}/{mapped_df_celltype}, Acq4 cell type: {acq4_celltype}/{mapped_acq4_celltype}")
    if cell_data['cell_type'].lower() != acq4_celltype.lower():
        CP.cprint("r", f"    MISMATCH: DataFrame cell type: {cell_data['cell_type']}/ {mapped_df_celltype}, Acq4 cell type: {acq4_celltype}/ {mapped_acq4_celltype}, {df.loc[icell].cell_id!s}")
        print(type(mapped_df_celltype), type(mapped_acq4_celltype))
        print(type(cell_data['cell_type']), type(acq4_celltype))
        print(f"<{cell_data['cell_type']:s}> != <{acq4_celltype:s}>")
        print(f"<{mapped_df_celltype}> != <{mapped_acq4_celltype:s}>")
        if cell_data['cell_type'].lower().lstrip() == acq4_celltype.lower():
            df.loc[icell, 'cell_type'] = acq4_celltype
            CP.cprint("g", f"    UPDATED DataFrame cell type <{df_cell_type:s}> to: <{acq4_celltype:s}> for cell id: {df.loc[icell].cell_id!s}")
            with open(data_summary_file, "wb") as f:
                pd.to_pickle(df, f)
            if df_cell_type == ' ':
                # seems to have been a default value, 
                continue
            else:
                exit()
        elif mapped_df_celltype != mapped_acq4_celltype:
            df.loc[icell, 'cell_type'] = mapped_acq4_celltype
            CP.cprint("g", f"    UPDATED DataFrame cell type <{mapped_df_celltype:s}> to: <{mapped_acq4_celltype:s}> for cell id: {df.loc[icell].cell_id!s}")
            with open(data_summary_file, "wb") as f:
                pd.to_pickle(df, f)
            exit()
    # if we get past all that, then update the datasummary excel file
ds = data_summary.DataSummary()
ds.make_excel(df, outfile=data_summary_file.with_suffix('.xlsx'))

    # if we get past all that, then update the datasummary excel file

