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

def gather_protocols(
    experiment: dict,
    df: pd.DataFrame,
    cell_id: str,
    protocols: list,
    prots: dict,
    allprots: dict = None,
    day: str = None,
):
    """
    Gather all the protocols and sort by functions/types
    First call will likely have allprots = None, to initialize
    after that will update allprots with new lists of protocols.

    The variable "allprots" is a dictionary that accumulates
    the specific protocols from this cell according to type.
    The type is then used to determine what analysis to perform.

    Parameters
    ----------
    protocols: list
        a list of all protocols that are found for this day/slice/cell
    prots: dict
        data, slice, cell information
    allprots: dict
        dict of all protocols by type in this day/slice/cell
    day : str
        str indicating the top level day for this slice/cell

    Returns
    -------
    allprots : dict
        updated copy of allprots.
    """

    cell_entry = df.loc[(df.cell_id == cell_id)]
    if cell_entry.empty:
        print(f"Date not found: {day!s} {cell_id!s}")
        # for dx in self.df.date.values:
        #     CP.cprint("r", f"    day: {dx:s}")
        raise FileNotFoundError(f"Cell: {self.cell_id!s} was not found in database")

    # CP.cprint("c", f"  ... [Analysis:run] Retrieved cell:\n       {cell_id:s}")
    icell = cell_entry.index
    if allprots is None:  # Start with the protocol groups in the configuration file
        protogroups = list(experiment["protocols"].keys())
        allprots = {k: [] for k in protogroups}
        # {"maps": [], "stdIVs": [], "CCIV_long": [], "CCIV_posonly": [], "VCIVs": []}
    else:
        protogroups = list(experiment["protocols"].keys())
    prox = sorted(list(set(protocols)))  # remove duplicates and sort alphabetically

    for i, protocol in enumerate(prox):  # construct filenames and sort by analysis types
        if len(protocol) == 0:
            continue
        # if a single protocol name has been selected, then this is the filter
        # if (
        #     (protocol is not None)
        #     and (len(protocol) > 1)
        #     and (protocol != protocol)
        # ):
        #     continue
        # clean up protocols that have a path ahead of the protocol (can happen when combining datasets in datasummary)
        protocol = Path(protocol).name

        # construct a path to the protocol, starting with the day
        # print(prots["date"], prots["slice_slice"], prots["cell_cell"], protocol)
        if day is None:
            c = Path(prots["date"].values[0], prots["slice_slice"].values[0], prots["cell_cell"].values[0], protocol)
        else:
            c = Path(day, prots.iloc[i]["slice_slice"].values[0], prots.iloc[i]["cell_cell"].values[0], protocol)
        c_str = str(c)  # make sure it is serializable for later on with JSON.
        # maps
        this_protocol = protocol[:-4]
        for pg in protogroups:
            pg_prots = experiment["protocols"][pg]
            if pg_prots is None:
                continue
            if this_protocol in pg_prots:
                allprots[pg].append(c_str)

        # if x.startswith("Map"):
        #     allprots["maps"].append(c_str)
        # if x.startswith("VGAT_"):
        #     allprots["maps"].append(c_str)
        # if x.startswith(
        #     "Vc_LED"
        # ):  # these are treated as maps, even though they are just repeated...
        #     allprots["maps"].append(c_str)
        # # Standard IVs (100 msec, devined as above)
        # for piv in self.stdIVs:
        #     if x.startswith(piv):
        #         allprots["stdIVs"].append(c_str)
        # # Long IVs (0.5 or 1 second)
        # if x.startswith("CCIV_long"):
        #     allprots["CCIV_long"].append(c_str)
        # # positive only ivs:
        # if x.startswith("CCIV_1nA_Pos"):
        #     allprots["CCIV_posonly"].append(c_str)
        # if x.startswith("CCIV_4nA_Pos"):
        #     allprots["CCIV_posonly"].append(c_str)
        # # VCIVs
        # if x.startswith("VCIV"):
        #     allprots["VCIVs"].append(c_str)
    print("Gather_protocols: Found these protocols: ", allprots)
    return allprots
