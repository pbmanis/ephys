"""
db_tools:

some subroutines for dealing with the database and generated files.
Make standardized cell IDS from the database
Get pickled cell data from the database


"""
from typing import Union
from pathlib import Path
from pylibrary.tools.cprint import cprint
import pandas as pd


def make_cell_ID(dfs, iday):
    """make_cell make the cell name from the cell ID and date
    This is meant to be called within a loop over the dataframe

    Parameters
    ----------
    dfs : pandas dataframe
        _description_
    iday : index into the dataframe
        _description_

    Returns
    -------
    str
        a cell name as a string, including the date, slice and cell.
    """
    # print("make cell: ", iday, self.dfs.iloc[iday])
    datestr = dfs.at[iday, "date"]
    if "slice_slice" in dfs.columns:
        slicestr = dfs.at[iday, "slice_slice"]
        cellstr = dfs.at[iday, "cell_cell"]
    else:
        id = dfs.at[iday, "cell_id"].split("_")[1]
        slicestr = f"slice_{int(id[:2]):03d}"
        cellstr = f"cell_{int(id[-1:]):03d}"
    return str(Path(datestr, slicestr, cellstr))


def get_pickled_cell_data(df, idx, analyzed_datapath: Union[Path, str] = None):
    # try looking for spikes from the analysis .pkl file
    # make a file name:
    cell_id = df.iloc[idx].cell_id
    cname = cell_id.replace(".", "_")
    # print('cname: ', cname)
    cname = cname.replace("_000", "")
    cname = f"{cname:s}_{df.iloc[idx]['cell_type']:s}_IVs.pkl"
    # print("cname 1: ", cname)
    fpath = Path(analyzed_datapath, df.iloc[idx]["cell_type"], cname)
    if not fpath.is_file():
        # try with double numbers for s/c:
        cname = cname[:12] + "0" + cname[12:]
        cname = cname[:15] + "0" + cname[15:]
        # print("cname2: ", cname)
        fpath = Path(analyzed_datapath, df.iloc[idx]["cell_type"], cname)

    if not fpath.is_file():
        cprint(
            "m",
            f"giv: No spikes for cell: {cell_id:s}, type: {df.iloc[idx]['cell_type']:s}",
        )
        return None
    with open(fpath, "rb") as fh:
        dx = pd.read_pickle(fh, compression="gzip")
    return dx

