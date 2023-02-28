
from pathlib import Path

import numpy as np
import pandas as pd
from pylibrary.tools import cprint as CP

"""write to an excel sheet with a standard set of colors for the cell types
in the sheet. Also formats the column widths.
"""

colors = {"pyramidal": "#c5d7a5", #"darkseagreen",
        "fusiform": "#c5d7a5",
        "pyr": "#c5d7a5",
        "cartwheel": "skyblue",
        "tuberculoventral": "lightpink",
        "granule": "linen",
        "golgi": "yellow",
        "unipolar brush cell": "sienna",
        "chestnut": "saddlebrown",
        "giant": "sandybrown",
        "giant?": "sandybrown",
        "giant cell": "sandybrown",
        "Unknown": "white",
        "unknown": "white",

        "bushy": "lightslategray",
        "t-stellate": "thistle",
        "l-stellate": "darkcyan",
        "d-stellate": "thistle",
        "stellate": "thistle",
        "octopus": "darkgoldenrod",
        
        # cortical (uses some of the same colors)
        "basket": "lightpink",
        "chandelier": "sienna",
        "fast spiking": "darksalmon",
        "fs": "darksalmon",
        "RS": "lightgreen",
        "LTS": "paleturquoise",
        "rs": "lightgreen",
        "lts": "paleturquoise",

        # thalamic neurons
        "thalamic": "yellow",

        # cerebellar
        "Purkinje": "mediumorchid",
        "purkinje": "mediumorchid",
        "purk": "mediumorchid",
        
        # not neurons
        'glia': 'lightslategray',
        'Glia': 'lightslategray',
}

def _find_celltype_in_field(celltype:str, cellkeys:list):
    for k in cellkeys:
        if celltype.find(k) >= 0:
            return k
    return None

def _highlight_by_cell_type(row):

    cellkeys = list(colors.keys())
    celltype = row.cell_type.lower()
    ctype = _find_celltype_in_field(celltype, cellkeys) 
    CP.cprint("y", f"celltype: {celltype:s},  ctype: {str(ctype):s}")
    if ctype is not None:
        return [f"background-color: {colors[ctype]:s}" for s in range(len(row))]
    else:
        return [f"background-color: red" for s in range(len(row))]


def _organize_columns(df, columns = None):
    if columns is None:
        return df
    # cols = ['ID', 'Group', 'Date', 'slice_slice','cell_cell', 'cell_type', 
    #     'iv_name', 'holding', 'RMP', 'RMP_SD', 'Rin', 'taum',
    #     'dvdt_rising', 'dvdt_falling', 'AP_thr_V', 'AP_HW', "AP15Rate", "AdaptRatio", 
    #     "AP_begin_V", "AHP_trough_V", "AHP_depth_V", "tauh", "Gh", "FiringRate",
    #     "FI_Curve",
    #     'date']
    df = df[columns + [c for c in df.columns if c not in columns]]
    return df

def make_excel(df:object, outfile:Path, sheetname:str="Sheet1", columns:list=None):
    """cleanup: reorganize columns in spreadsheet, set column widths
    set row colors by cell type

    Args:
        df: object
            Pandas dataframe object
        excelsheet (_type_): _description_
    """
    if outfile.suffix != '.xlsx':
        outfile = outfile.with_suffix('.xlsx')

    writer = pd.ExcelWriter(outfile, engine='xlsxwriter')

    df.to_excel(writer, sheet_name = sheetname)
    # df = _organize_columns(df, columns=columns)
    workbook = writer.book
    worksheet = writer.sheets[sheetname]
    fdot3 = workbook.add_format({'num_format': '####0.000'})
    df.to_excel(writer, sheet_name = sheetname)

    resultno = []
    # resultno = ['holding', 'RMP', 'Rin', 'taum', 'dvdt_rising', 'dvdt_falling', 
    #     'AP_thr_V', 'AP_HW', "AP15Rate", "AdaptRatio", "AP_begin_V", "AHP_trough_V", "AHP_depth_V"]
    # df[resultno] = df[resultno].apply(pd.to_numeric)    
    for i, column in enumerate(df):
        # print('column: ', column)
        if column in resultno:
            writer.sheets[sheetname].set_column(first_col=i+1, last_col=i+1,  cell_format=fdot3)
        if column not in ['notes', 'description', 'OriginalTable', 'FI_Curve']:
            coltxt = df[column].astype(str)
            coltxt = coltxt.map(str.rstrip)
            maxcol = coltxt.map(len).max()
            column_width = np.max([maxcol, len(column)]) # make sure the title fits
            if column_width > 50:
                column_width = 50 # but also no super long ones
            #column_width = max(df_new[column].astype(str).map(len).max(), len(column))
        else:
            column_width = 25
        if column_width < 8:
            column_width = 8
        if column in resultno:
            writer.sheets[sheetname].set_column(first_col=i+1, last_col=i+1, cell_format=fdot3, width=column_width) # column_dimensions[str(column.title())].width = column_width
            # print(f"formatted {column:s} with {str(fdot3):s}")
        else:
            writer.sheets[sheetname].set_column(first_col=i+1, last_col=i+1, width=column_width) # column_dimensions[str(column.title())].width = column_width

    df = df.style.apply(_highlight_by_cell_type, axis=1)
    try:
        df.to_excel(writer, sheet_name = sheetname, columns=columns)  # organize columns at the end
        writer.close()
    except:
        print(df.columns)
        print(columns)
        raise ValueError


if __name__ == "__main__":
    celltype =  "layer vi lts"
    ct = _find_celltype_in_field(celltype, list(colors.keys()))
    print(ct)