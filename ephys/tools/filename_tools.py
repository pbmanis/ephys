from typing import Union, Tuple
import datetime
import pylibrary.tools.cprint as cprint
from pathlib import Path
import pandas as pd
import re

CP = cprint.cprint

def check_celltype(celltype: Union[str, None] = None):
    """check_celltype: convert cell type to "unknown" if it is None, empty, or whitespace or '?

    Parameters
    ----------
    celltype : Union[str, None], original cell type
        string from table celltype, by default None

    Returns
    -------
    str
        updated cell type. Specifically
    """

    if isinstance(celltype, str):
        celltype = celltype.strip()
    # print("celltype: ", celltype, type(celltype))
    celltype = str(celltype)
    if len(celltype) == 0:
        celltype = 'unknown'
    if celltype in [None, "", "?", " ", "  ", "\t"]:
        # CP("y", f"check_celltype:: Changing Cell type to unknown from <{celltype:s}>")
        celltype = "unknown"
    return celltype


def make_cellstr(df: pd.DataFrame, icell: int, shortpath: bool = False):
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
    """make_pdf_filename Given the path, day, celltype, and analysis type, make a pdf filename

    Parameters
    ----------
    dpath : Union[Path, str]
        Path to the data file
    thisday : str
        date string (2022.01.01, for example)
    celltype : str
        cell name ("pyramidal", for example)
    analysistype : str
        type of analysis plotted in this pdf ("IV", for example)
    slicecell : str
        slice and cell number (S00C01 or S0C0, for example)

    Returns
    -------
    Path
        Full path for a pdf file name.
    """
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


def get_pickle_filename_from_row(selected: pd.Series, dspath: Union[str, Path], mode="IVs"):
    """get_pickle_filename_from_row given the selected row in a table / dataframe,
    return the full path to the pickle file.

    Parameters
    ----------
    selected : pd.Series
        Panda Series of the selected row
    dspath : Union[str, Path]
        Path to the dataset
    mode : str, optional (default "IVs")
        Type of analysis held in the file (IVs, maps, etc)

    Returns
    -------
    Path
        Full file path to the pickle file

    Raises
    ------
    ValueError
        Correctly formatted data not found in the selected Series
    ValueError
        Slice/cell not found in the selected Series
    """
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

    day = f"{day:s}_{slicecell:s}_{cell_type:s}_{mode:s}.pkl"
    pkl_file = Path(dspath, cell_type, day)
    return pkl_file


def change_pickle_filename(original_name, slicecell):
    """change_pickle_filename slicecell designator with the new slicecell value
    Only pass in the original name, not the full path.
    Parameters
    ----------
    original_name : _type_
        The original full filename (with path, perhaps)
    slicecell : _type_
        new slicecell to use.
    """
    pathparts = Path(original_name).parts
    filename = Path(pathparts[-1]).name
    re_fn = re.compile(r"(\d{4})_(\d{2})_(\d{2})_S(\d{1,3})C(\d{1,3})_(\w+)_IVs.pkl")
    m = re_fn.match(filename)
    if m is not None:
        newname = (
            f"{m.group(1):s}_{m.group(2):s}_{m.group(3):s}_{slicecell:s}_{m.group(6):s}_IVs.pkl"
        )
        return newname
    else:
        return None


def make_pickle_filename(dpath: Union[str, Path], thisday: str, celltype: str, slicecell: str, analysistype:str="IVs", makedir:bool=False):
    """make_pickle_filename make a fully qualified path to a pickle file for a cell

    Parameters
    ----------
    dpath : Union[str, Path]
        base path to data
    thisday : str
        date string (2022.01.01, for example)
    celltype : str
        cell name ("pyramidal", for example)
    slicecell : str
        slice and cell number (S00C01 or S0C0, for example, or slice_000/cell_001)
    analysistype : str, optional
        type of data in the file (IVs, maps, etc), by default "IVs"

    Returns
    -------
    Path
        Path to the pickle file
    """
    if slicecell.find('/') > 0:
        sc = Path(slicecell).parts
        slicecell = make_slicecell(sc[0], sc[1])
    if thisday.find("_000") > 0:
        thisday = thisday.split("_000")[0]
    pklname = make_cell_filename(thisday, celltype=celltype, slicecell=slicecell, analysistype=analysistype)
    pklname = Path(pklname)
    # check to see if we have a sorted directory with this cell type
    pkldir = Path(dpath, celltype)
    if not pkldir.is_dir() and makedir:
        pkldir.mkdir()
    return Path(pkldir, pklname.stem).with_suffix(".pkl")


def make_cell(icell: int, df: pd.DataFrame = None):
    assert df is not None
    try:
        datestr = Path(df.iloc[icell]["date"]).name
    except ValueError:
        CP("r", f"Failed to get date string from dataframe with icell={icell:d}")
        return None, None, None
    print("make_cell: datestr = ", datestr, "icell: ", icell)
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

def make_cellid_from_slicecell(slicecell):
    """make_cellid_from_slicecell Convert slicecell string
     of the form 2022.01.01_000_S0C0 into a cell_id string
    such as 2022.01.01_000/slice_000/cell_000

    Parameters
    ----------
    slicecell : _type_
        _description_
    returns
    cell_id : str
    """
    re_sc = re.compile(r"S(\d{1,3})C(\d{1,3})")
    sc = slicecell.split("_")[-1]
    m = re_sc.match(sc)

    if m is not None:
        slicestr = f"slice_{int(m.group(1)):03d}"
        cellstr = f"cell_{int(m.group(2)):03d}"
        scdate = slicecell.split("_")
        cell_id = f"{scdate[0]:s}_{scdate[1]:s}/{slicestr:s}/{cellstr:s}"

        return cell_id
    else:
        print(m)
        raise ValueError("Failed to parse slicecell string")

def make_cell_filename(
    thisday: str,
    celltype: str,
    slicecell: str,
    analysistype: str = None,
    extras: dict = None,
    flags: dict = None,
):
    """make_cell_filename Make a full cell/filename string for a cell file

    Parameters
    ----------
    thisday : str
        date string (2022.01.01, for example)
    celltype : str
        cell name ("pyramidal", for example)
    slicecell : str
        slice and cell number (S00C01 or S0C0, for example)
    analysistype : str, optional
        Type if analysis ("IVs", "maps"), by default None
    extras : dict, optional
        other analyses (flipped, etc), by default None
    flags : dict, optional
        flags to add to the filename, by default None

    Returns
    -------
    Path
        filename path
    """
    celltype = check_celltype(celltype)
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

def make_event_filename_from_cellid(cell_id: str):
    """make_event_filename_from_cellid 
    Make a full event filename string for a cell file
    Parameters
    cell_id: str or Path - cell_id in form dpath/2022.01.01_000/slice_000/cell_000

    Returns
    eventfile name in format:
        2022_01_01_000~slice_000~cell_000.pkl
    """

    eventname = Path(cell_id)
    eventname = Path(*eventname.parts[-3:])
    eventfile= str(eventname).replace("/", "~")+".pkl"
    return eventfile

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
    print('compare_slice_cell datestr: ', datestr)
    print('compare_slice_cell slicestr: ', str)
    dsday, nx = Path(datestr).name.split("_")
    # check dates
    thisday = datetime.datetime.strptime(dsday, "%Y.%m.%d")
    if (after_dt is not None) and (before_dt is not None):
        if thisday < after_dt or thisday > before_dt:
            CP(
                "y",
                f"Day {datestr:s} is not in range {after_dt!s} to {before_dt!s}",
            )
            return (False, "", "", "")

    # check slice/cell:
    slicecell3 = f"S{int(slicestr[-3:]):02d}C{int(cellstr[-3:]):02d}"  # recognize that slices and cells may be more than 10 (# 9)
    slicecell2 = f"S{int(slicestr[-3:]):01d}C{int(cellstr[-3:]):01d}"  # recognize that slices and cells may be more than 10 (# 9)
    slicecell1 = f"{int(slicestr[-3:]):1d}{int(cellstr[-3:]):1d}"  # only for 0-9
    compareslice = ""
    print(slicecell, slicecell3, slicecell2, slicecell1)
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


def get_cell_pkl_filename(experiment: dict, df: pd.DataFrame, cell_id: str):
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
    df_tmp = df[df.cell_id == cell_id]  # df.copy() # .dropna(subset=["date"])
    # print("get_cell: df_tmp: ", df_tmp.keys())
    # print("\nGet_cell:: df_tmp head: \n", "Groups: ", df_tmp["Group"].unique(), "\n len df_tmp: ", len(df_tmp))
    # print("filename tools: get_cell: cell_id: ", cell_id)
    if len(df_tmp) == 0:
        CP(
            "r", f"filename_tools:get_cell:: Cell ID not found in summary dataframe: {cell_id:s}"
        )
        return None, None
    try:
        celltype = df_tmp.cell_type.values[0]
    except ValueError:
        celltype = df_tmp.cell_type
    celltype = str(celltype).replace("\n", "")
    if celltype == " ":  # no cell type
        celltype = "unknown"
    CP("m", f"get cell: df_tmp cell type: {celltype:s}")
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
        snp = re_parse.match(cell_parts[-1]).group(4)
        cell_day_name = cell_parts[-3].split("_")[0]
    else:
        cell = Path(cell_id).name  # just get the name here
        cell_parts = cell.split("_")
        cell_day_name = cell_parts[0]
        re_parse = re.compile(r"([Ss]{1})(\d{1,3})([Cc]{1})(\d{1,3})")
        m = re_parse.match(cell_parts[-1])
        if m is not None:
            # print("cell_parts: ", cell_parts[-1])
            snp = re_parse.match(cell_parts[-1]).group(2)
            cnp = re_parse.match(cell_parts[-1]).group(4)
            cname2 = f"{cell_day_name.replace('.', '_'):s}_S{snp:s}C{cnp:s}_{celltype:s}_IVs.pkl"
        elif cell_id.find("cell"):  # try to use /slice_000 /cell_000 style
            cell_parts = Path(cell_id).parts
            sc = make_slicecell(cell_parts[-2], cell_parts[-1])
            cell_day_name = cell_parts[-3].replace(".", "_")
            cname2 = f"{cell_day_name[:-4]:s}_{sc:s}_{celltype:s}_IVs.pkl"
        else:

            raise ValueError(f"Failed to parse cell name: {cell:s}")

    datapath2 = Path(experiment["analyzeddatapath"], experiment["directory"], celltype, cname2)

    if datapath2.is_file():
        CP("c", f"...  datapath: {datapath2!s} is OK")
        datapath = datapath2
    else:
        # print("tried datapath: ", datapath2, celltype)
        print(f"no file: matching: {datapath2!s} with celltype: {celltype:s}")
        CP("r", f"no file: matching: {datapath2!s} with celltype: {celltype:s}")
        raise ValueError
        return None, None
    return datapath

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
    # print("get cell: ", experiment, df, cell_id)
    datapath = get_cell_pkl_filename(experiment, df, cell_id)
    df_tmp = df[df.cell_id == cell_id]
    try:
        df_cell = pd.read_pickle(datapath, compression="gzip")
    except ValueError:
        try:
            df_cell = pd.read_pickle(datapath)  # try with no compression
        except ValueError:
            CP("r", f"Could not read {datapath!s}")
            raise ValueError("Failed to read compressed pickle file")
        
    if "Spikes" not in df_cell.keys() or df_cell.Spikes is None:
        CP(
            "y",
            f"df_cell: {df_cell.age!s}, {df_cell.cell_type!s}, No spike protos:",
        )
        return None, None

    return df_cell, df_tmp


if __name__ == "__main__":
    test_slicestr = "slice_001"
    test_cellstr = "cell_003"
    # slicecell = make_slicecell(slicestr, cellstr)
    test_fn = "2022_01_01_S00C02_pyramidal_IVs.pkl"
    test_res = change_pickle_filename(test_fn, "S0C2")
    print(test_res)

    cell_id = "2017.03.28_000/slice_000/cell_001"
    print(make_cell_filename( "stellate", "S00C01", "IVs"))

