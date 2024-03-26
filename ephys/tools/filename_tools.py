from typing import Union, Tuple
import datetime
import pylibrary.tools.cprint as CP
from pathlib import Path
import pandas as pd
import re


def check_celltype(celltype: Union[str, None] = None):
    if isinstance(celltype, str):
        celltype = celltype.strip()
    if celltype in [None, "", "?", " ", "  ", "\t"]:
        print(f"Changing Cell type to unknown from <{celltype:s}>")
        celltype = "unknown"
    return celltype


def make_cellstr(sdf: pd.DataFrame, icell: int, shortpath: bool = False):
    """
    Make a day string including slice and cell from the icell index in the pandas dataframe df
    Example result:
        s = self.make_cellstr (df, 1)
        s: '2017.01.01_000/slice_000/cell_001'  # Mac/linux path string

    Parameters
    ----------
    df : Pandas dataframe instance

    icell : int (no default)
        index into pandas dataframe instance

    returns
    -------
    Path
    """

    if shortpath:
        day = Path(df.iloc[icell]["date"]).parts[-1]
        cellstr = Path(
            day,
            Path(df.iloc[icell]["slice_slice"]).name,
            Path(df.iloc[icell]["cell_cell"]).name,
        )
    else:
        cellstr = Path(
            df.iloc[icell]["date"],
            Path(df.iloc[icell]["slice_slice"]).name,
            Path(df.iloc[icell]["cell_cell"]).name,
        )
    # print("make_cellstr: ", daystr)
    return cellstr


def make_pdf_filename(
    dpath: Union[Path, str], thisday: str, celltype: str, analysistype: str, slicecell: str
):
    pdfname = make_cell_filename(
        thisday=thisday, celltype=celltype, slicecell=slicecell, analysistype=analysistype
    )
    pdfname = Path(pdfname)
    # print("make pdf filename pdfname: ", pdfname)
    # check to see if we have a sorted directory with this cell type
    pdfdir = Path(dpath, celltype)
    
    # print("make_pdf filename dpath: ", dpath)
    # print("make pdf filename pdfdir: ", pdfdir)

    if not pdfdir.is_dir():
        pdfdir.mkdir()
    return Path(pdfdir, pdfname.stem).with_suffix(".pdf")


def get_pickle_filename_from_row(selected: pd.Series, dspath: Union[str, Path]):
    print("selected cell: ", selected.cell_id)
    celln = selected.cell_cell
    if selected.cell_type is None or selected.cell_type == " ":
        cell_type = "unknown"
    else:
        cell_type = selected.cell_type
    slicen = selected.slice_slice
    pathparts = Path(selected.cell_id).parts
    re_day = re.compile(r"(\d{4}).(\d{2}).(\d{2})_(\d{3})")
    found = False
    for i, part in enumerate(pathparts):
        m = re_day.match(part)
        if m is not None:
            found = True
            break
    if not found:
        raise ValueError(f"Failed to find formatted date in cell_id: {selected.cell_id:s}")    

    day = f"{m.group(1):s}_{m.group(2):s}_{m.group(3):s}"

    re_sliceno = re.compile(r"slice_(\d{3})")
    re_cellno = re.compile(r"cell_(\d{3})")
    m = re_sliceno.match(slicen)
    slicecell = ""
    if m is not None:
        slicecell = f"S{int(m.group(1)):d}"
        m = re_cellno.match(celln)
        if m is not None:
            slicecell += f"C{int(m.group(1)):d}"
    else:
        re_sc = re.compile(r"S(\d{1,3})C(\d{1,3})")
        sc = selected.cell_id.split("_")[-1]
        m = re_sc.match(sc)
        if m is not None:
            slicecell = f"S{int(m.group(1)):d}C{int(m.group(2)):d}"
        else:
            raise ValueError(f"Failed to find slice and cell in cell_id: {selected.cell_id:s}")

    day = f"{day:s}_{slicecell:s}_{cell_type:s}_IVs.pkl"
    pkl_file = Path(dspath, cell_type, day)
    return pkl_file

def change_pickle_filename(original_name, slicecell):
    """change_pickle_filename slicecell designator with the new slicecell value
    Only pass in the original name, not the full path.
    Parameters
    ----------
    original_name : _type_
        _description_
    slicecell : _type_
        _description_
    """
    pathparts = Path(original_name).parts
    filename = Path(pathparts[-1]).name
    re_fn = re.compile(r"(\d{4})_(\d{2})_(\d{2})_S(\d{1,3})C(\d{1,3})_(\w+)_IVs.pkl")
    m = re_fn.match(filename)
    if m is not None:
        newname = f"{m.group(1):s}_{m.group(2):s}_{m.group(3):s}_{slicecell:s}_{m.group(6):s}_IVs.pkl"
        return newname
    else:
        return None



def make_pickle_filename(dpath: Union[str, Path], thisday:str, celltype: str, slicecell: str):

    pklname = make_cell_filename(thisday, celltype=celltype, slicecell=slicecell)
    pklname = Path(pklname)
    # check to see if we have a sorted directory with this cell type
    pkldir = Path(dpath, celltype)
    if not pkldir.is_dir():
        pkldir.mkdir()
    return Path(pkldir, pklname.stem).with_suffix(".pkl")


def make_cell(icell: int, df: pd.DataFrame):
    assert df is not None
    datestr = Path(df.iloc[icell]["date"]).name
    slicestr = str(Path(df.iloc[icell]["slice_slice"]).parts[-1])
    cellstr = str(Path(df.iloc[icell]["cell_cell"]).parts[-1])
    return (datestr, slicestr, cellstr)


def make_slicecell(slicestr: str, cellstr: str):
    """make_slice_cell Convert slice_000 and cell_000 into a slicecell string
    such as S0C0 or S00C00.

    Parameters
    ----------
    slicestr : _type_
        _description_
    cellstr : _type_
        _description_
    """

    re_slice = re.compile(r"(slice)_(\d{1,3})")
    re_cell = re.compile(r"(cell)_(\d{1,3})")
    slice_parts = re_slice.match(slicestr)
    cell_parts = re_cell.match(cellstr)
    if slice_parts is not None and cell_parts is not None:
        slicecell = f"S{int(slice_parts[2]):1d}C{int(cell_parts[2]):1d}"
        return slicecell
    else:
        raise ValueError("Failed to parse slice and cell strings")


def make_cell_filename(
    thisday: str, celltype: str, slicecell: str, analysistype: str=None, extras: dict = None, flags: dict = None
):
    celltype = check_celltype(celltype)
    print("make cell filename thisday: ", thisday)
    file_name = thisday.replace(".", "_")
    file_name += f"_{slicecell:s}"
    if extras is not None:
        for extra in extras:
            if extras[extra]:
                file_name += f"_{extra:s}"

    file_name += f"_{celltype:s}"
    if analysistype is not None:
        file_name += f"_{analysistype:s}"
    if flags is not None:
        for flag in flags:
            if flags[flag]:
                file_name += f"_{flag:s}"

    return Path(file_name)


def compare_slice_cell(
    slicecell: str,
    datestr: str,
    slicestr: str,
    cellstr: str,
    after_dt: datetime.datetime = None,
    before_dt: datetime.datetime = None,
) -> Tuple[bool, str, str, str]:
    """compare_slice_cell - compare the slice and cell strings in the dataframe
    with the specified slicecell value.

    Parameters
    ----------
    slicecell: str
        e.g., like "S00C01" or "S0C1"

    Returns
    -------
    bool
        True if the slice and cell match, False otherwise

    """

    dsday, nx = Path(datestr).name.split("_")
    # check dates
    thisday = datetime.datetime.strptime(dsday, "%Y.%m.%d")
    if (after_dt is not None) and (before_dt is not None):
        if thisday < after_dt or thisday > before_dt:
            CP.cprint(
                "y",
                f"Day {datestr:s} is not in range {after_dt!s} to {before_dt!s}",
            )
            return (False, "", "", "")
    # check slice/cell:
    slicecell3 = f"S{int(slicestr[-3:]):02d}C{int(cellstr[-3:]):02d}"  # recognize that slices and cells may be more than 10 (# 9)
    slicecell2 = f"S{int(slicestr[-3:]):01d}C{int(cellstr[-3:]):01d}"  # recognize that slices and cells may be more than 10 (# 9)
    slicecell1 = f"{int(slicestr[-3:]):1d}{int(cellstr[-3:]):1d}"  # only for 0-9
    compareslice = ""
    if slicecell is not None:  # limiting to a particular cell?
        match = False
        if len(slicecell) == 2:  # 01
            compareslice = f"{int(slicecell[0]):1d}{int(slicecell[1]):1d}"
            if compareslice == slicecell1:
                match = True
        elif len(slicecell) == 4:  # S0C1
            compareslice = f"S{int(slicecell[1]):1d}C{int(slicecell[3]):1d}"
            if compareslice == slicecell2:
                match = True
        elif len(slicecell) == 6:  # S00C01
            compareslice = f"S{int(slicecell[1:3]):02d}C{int(slicecell[4:]):02d}"
            if compareslice == slicecell3:
                match = True

        if match:
            return (True, slicecell3, slicecell2, slicecell1)
        else:
            # raise ValueError()
            # Logger.error(f"Failed to find cell: {slicecell:s} in {slicestr:s} and {cellstr:s}")
            # Logger.error(f"Options were: {slicecell3:s}, {slicecell2:s}, {slicecell1:s}")
            return (False, "", "", "")  # fail silently... but tell caller.
    else:
        return (True, slicecell3, slicecell2, slicecell1)


def get_cell(experiment: dict, df: pd.DataFrame, cell_id: str):
    """get_cell get the pickled data file for this cell - this is an analyzed file,
    usually in the "dataset/experimentname" directory, likely in a celltype subdirectory

    Parameters
    ----------
    experiment : dictionary
        configuration dictionary
    df : pd.DataFrame
        main data summary dataframe
    cell : str
        the cell_id for this cell (typically, a partial path to the cell file)
        For example: Rig2/2022.01.01_000/slice_000/cell_000

    Returns
    -------
    a pandas df (series) with the pickled data
        _description_
        returns None, None if the cell is not found, or if the file
        does not have a Spikes entry.

    Raises
    ------
    ValueError
        no matching file
    ValueError
        failed to read the compressed pickle file
    """
    df_tmp = df[df.cell_id == cell_id]  # df.copy() # .dropna(subset=["Date"])
    # print("\nGet_cell:: df_tmp head: \n", "Groups: ", df_tmp["Group"].unique(), "\n len df_tmp: ", len(df_tmp))
    # print("filename tools: get_cell: cell_id: ", cell_id)
    if len(df_tmp) == 0:
        CP.cprint("r", f"filename_tools:get_cell:: Cell ID not found in summary dataframe: {cell_id:s}")
        return None, None
    try:
        celltype = df_tmp.cell_type.values[0]
    except ValueError:
        celltype = df_tmp.cell_type
    celltype = str(celltype).replace("\n", "")
    if celltype == " ":  # no cell type
        celltype = "unknown"
    CP.cprint("m", f"get cell: df_tmp cell type: {celltype:s}")
    # look for original PKL file for cell in the dataset
    # if it exists, use it to get the FI curve
    # base_cellname = str(Path(cell)).split("_")
    # print("base_cellname: ", base_cellname)
    # sn = int(base_cellname[-1][1])
    # cn = int(base_cellname[-1][3])
    # different way from cell_id:
    # The cell name may be a path, or just the cell name.
    # we have to handle both cases.

    parent = Path(cell_id).parent
    if parent == ".":  # just cell, not path
        cell_parts = str(cell_id).split("_")
        re_parse = re.compile(r"([Ss]{1})(\d{1,3})([Cc]{1})(\d{1,3})")
        cnp = re_parse.match(cell_parts[-1]).group(2)
        cn = int(cnp)
        snp = re_parse.match(cell_parts[-1]).group(4)
        sn = int(snp)
        cell_day_name = cell_parts[-3].split("_")[0]
        dir_path = None
    else:
        cell = Path(cell_id).name  # just get the name here
        cell_parts = cell.split("_")
        cell_day_name = cell_parts[0]
        re_parse = re.compile(r"([Ss]{1})(\d{1,3})([Cc]{1})(\d{1,3})")
        m = re_parse.match(cell_parts[-1])
        if m is not None:
            # print("cell_parts: ", cell_parts[-1])
            snp = re_parse.match(cell_parts[-1]).group(2)
            sn = int(snp)
            cnp = re_parse.match(cell_parts[-1]).group(4)
            cn = int(cnp)
            cname2 = f"{cell_day_name.replace('.', '_'):s}_S{snp:s}C{cnp:s}_{celltype:s}_IVs.pkl"
        elif cell_id.find("cell"):  # try to use /slice_000 /cell_000 style
            cell_parts = Path(cell_id).parts
            sc = make_slicecell(cell_parts[-2], cell_parts[-1])
            cell_day_name = cell_parts[-3].replace('.', '_')
            cname2 = f"{cell_day_name[:-4]:s}_{sc:s}_{celltype:s}_IVs.pkl"
        else:

            raise ValueError(f"Failed to parse cell name: {cell:s}")
        dir_path = parent

    datapath2 = Path(experiment["analyzeddatapath"], experiment["directory"], celltype, cname2)

    if datapath2.is_file():
        CP.cprint("c", f"...  {datapath2!s} is OK")
        datapath = datapath2
    else:
        CP.cprint("r", f"no file: matching: {datapath2!s}, \n") #    or: {datapath2!s}\n")
        print("cell type: ", celltype)
        raise ValueError
        return None, None
    try:
        df_cell = pd.read_pickle(datapath, compression="gzip")
    except ValueError:
        try:
            df_cell = pd.read_pickle(datapath)  # try with no compression
        except ValueError:
            CP.cprint("r", f"Could not read {datapath!s}")
            raise ValueError("Failed to read compressed pickle file")
    if "Spikes" not in df_cell.keys() or df_cell.Spikes is None:
        CP.cprint(
            "y",
            f"df_cell: {df_cell.age!s}, {df_cell.cell_type!s}, No spike protos:",
        )
        return None, None
   
    return df_cell, df_tmp

if __name__ == "__main__":
    slicestr = "slice_001"
    cellstr = "cell_003"
    # slicecell = make_slicecell(slicestr, cellstr)
    fn = "2022_01_01_S00C02_pyramidal_IVs.pkl"
    res = change_pickle_filename(fn, "S0C2")
    print(res)
