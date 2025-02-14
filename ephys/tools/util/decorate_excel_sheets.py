import numpy as np
import pandas as pd

colors = {"pyramidal": "#c5d7a5", #"darkseagreen",
            "cartwheel": "skyblue",
            "tuberculoventral": "lightpink",
            "Unknown": "white",
            "unknown": "white",
            "bushy": "lightslategray",
            "unipolar brush cell": "sienna",
            "giant": "sandybrown",
            # "giant?": "sandybrown",
            # "Giant": "sandybrown",
            "t-stellate": "thistle",
            "stellate": "thistle",
            # "glia": "yellow",
            "glial": "yellow",
}

def highlight_by_cell_type(row):

    if row.cell_type in colors:
        color = colors[row.cell_type]
    else:
        color = "white"
    return [f"background-color: {color:s}" for s in range(len(row))]


def organize_columns(df):
    """organize_columns _summary_

    Parameters
    ----------
    df : pandas dataframe
        contains the data to be organized

    Returns
    -------
    pandas dataframe
        the original data, just reorganized.
    """
    cols = ['ID', 'Group', 'date', 'slice_slice','cell_cell', 'cell_type', 'age', 'temperature', 'internal', 
        'protocol', 'holding', 'sample_rate', 'RMP', 'RMP_SD', 'Rin', 'taum',
        'dvdt_rising', 'dvdt_falling', 'AP_thr_V', 'AP_HW', "AP15Rate", "AdaptRatio", 
        "AP_begin_V", "AHP_trough_V", "AHP_depth_V", "tauh", "Gh", "FiringRate",
        "FI_Curve",
        'date']
    # print(df.columns)
    # print(df.head())
    df = df[cols + [c for c in df.columns if c not in cols]]
    return df

def sanitize_celltype(row):
    if row.cell_type not in colors.keys():
        row.cell_type = "Mark for review"
    if row.cell_type in ['Unknown', 'unknown']:
        row.cell_type = "Mark for review"
    return row

def empty_to_nan(row, *args):
    variable = args[0]
    if row[variable] in [" ", ""]:
        row[variable] = pd.NA
    return row[variable]

def not_numbers(var):
    if isinstance(var, pd.Series):
        return var
    if not var.str.is_numeric:
        var = pd.NA
    return var

def cleanup(excelsheet, outfile:str="test.xlsx", dropmarked:bool=True):
    """cleanup: reorganize columns in spreadsheet, set column widths
    set row colors by cell type

    Args:
        excelsheet (_type_): _description_
    """
    df_new = pd.read_excel(excelsheet)
    df_new = df_new.apply(sanitize_celltype, axis=1)
    df_new = df_new[df_new.cell_type != "Mark for review"] # drop rows marked for review

    writer = pd.ExcelWriter(outfile, engine='xlsxwriter')

    df_new.to_excel(writer, sheet_name = "Sheet1")
    # skipping reorganizing columns - does not matter much
    # df_new = organize_columns(df_new)
    workbook = writer.book
    worksheet = writer.sheets["Sheet1"]
    fdot_3 = workbook.add_format({'num_format': '##,##0.000'})
    fdot_0 = workbook.add_format({'num_format': '#,###,##0.'})
    df_new.to_excel(writer, sheet_name = "Sheet1")

    resultno_3 = ['holding', 'RMP', 'Rin', 'taum', 'dvdt_rising', 'dvdt_falling', 
        'AP_thr_V', 'AP_HW', "AP15Rate", "AdaptRatio", "AP_begin_V", "AHP_trough_V", "AHP_depth_V"]
    resultno_0 = ['sample_rate']
    # for v in resultno:
    #     df_new[v] = df_new.apply(empty_to_nan, args=(v))  
    df_new[resultno_3] = df_new[resultno_3].apply(pd.to_numeric, axis=1, errors='coerce')    
    df_new[resultno_0] = df_new[resultno_0].apply(pd.to_numeric, axis=1, errors='coerce')    
    for i, column in enumerate(df_new):
        # print('column: ', column)
        if column in resultno_3:
            writer.sheets['Sheet1'].set_column(first_col=i+1, last_col=i+1,  cell_format=fdot_3)
        if column in resultno_0:
            writer.sheets['Sheet1'].set_column(first_col=i+1, last_col=i+1,  cell_format=fdot_0)

        if column not in ['notes', 'description', 'OriginalTable', 'FI_Curve', 'IV', 'Spikes', 'date', 'age', 'internal']:
            coltxt = df_new[column].astype(str)
            coltxt = coltxt.map(str.rstrip)
            maxcol = coltxt.map(len).max()
            column_width = maxcol
            #column_width = max(df_new[column].astype(str).map(len).max(), len(column))
        else:
            column_width = 25
        # print("column width: ", column_width)
        if column_width < 9:
            column_width = 9
        if column in resultno_3:
            writer.sheets['Sheet1'].set_column(first_col=i+1, last_col=i+1, cell_format=fdot_3, width=column_width) # column_dimensions[str(column.title())].width = column_width
            # print(f"formatted {column:s} with {str(fdot3):s}")
        elif column in resultno_0:
            writer.sheets['Sheet1'].set_column(first_col=i+1, last_col=i+1, cell_format=fdot_0, width=column_width) # column_dimensions[str(column.title())].width = column_width
        
        else:
            writer.sheets['Sheet1'].set_column(first_col=i+1, last_col=i+1, width=column_width) # column_dimensions[str(column.title())].width = column_width
        

    # df_new = df_new.style.apply(highlight_by_cell_type, axis=1)
    # print("new dataframe columns: ", df_new.columns)
    df_new.to_excel(writer, sheet_name = "Sheet1")
    writer.close()