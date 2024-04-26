import pandas as pd

def highlight_by_cell_type(row):
    colors = {"pyramidal": "#c5d7a5", #"darkseagreen",
              "cartwheel": "skyblue",
              "tuberculoventral": "lightpink",
              "Unknown": "white",
              "unknown": "white",
              " ": "white",
              "bushy": "lightslategray",
              "unipolar brush cell": "sienna",
              "giant": "sandybrown",
              "giant?": "sandybrown",
              "t-stellate": "thistle",
              "stellate": "thistle",
    }
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
        "AP_begin_V", "AHP_trough_V", "AHP_trough_T", "AHP_depth_V", "tauh", "Gh", "FiringRate",
        "FI_Curve",
        'date']
    # print(df.columns)
    # print(df.head())
    df = df[cols + [c for c in df.columns if c not in cols]]
    return df

def cleanup(excelsheet, outfile:str="test.xlsx"):
    """cleanup: reorganize columns in spreadsheet, set column widths
    set row colors by cell type

    Args:
        excelsheet (_type_): _description_
    """
    df_new = pd.read_excel(excelsheet)
    writer = pd.ExcelWriter(outfile, engine='xlsxwriter')

    df_new.to_excel(writer, sheet_name = "Sheet1")
    df_new = organize_columns(df_new)
    workbook = writer.book
    worksheet = writer.sheets["Sheet1"]
    fdot3 = workbook.add_format({'num_format': '####0.000'})
    df_new.to_excel(writer, sheet_name = "Sheet1")

    resultno = ['holding', 'sample_rate', 'RMP', 'Rin', 'taum', 'dvdt_rising', 'dvdt_falling', 
        'AP_thr_V', 'AP_thr_T', 'AP_HW', "AP15Rate", "AdaptRatio", "AP_begin_V", "AHP_trough_V", "AHP_trough_T", "AHP_depth_V"]
    df_new[resultno] = df_new[resultno].apply(pd.to_numeric)    
    for i, column in enumerate(df_new):
        # print('column: ', column)
        if column in resultno:
            writer.sheets['Sheet1'].set_column(first_col=i+1, last_col=i+1,  cell_format=fdot3)
        if column not in ['notes', 'description', 'OriginalTable', 'FI_Curve', 'IV', 'Spikes', 'date', 'age', 'internal']:
            coltxt = df_new[column].astype(str)
            coltxt = coltxt.map(str.rstrip)
            maxcol = coltxt.map(len).max()
            column_width = maxcol
            #column_width = max(df_new[column].astype(str).map(len).max(), len(column))
        else:
            column_width = 25
        # print("column width: ", column_width)
        if column_width < 8:
            column_width = 8
        if column in resultno:
            writer.sheets['Sheet1'].set_column(first_col=i+1, last_col=i+1, cell_format=fdot3, width=column_width) # column_dimensions[str(column.title())].width = column_width
            # print(f"formatted {column:s} with {str(fdot3):s}")
        else:
            writer.sheets['Sheet1'].set_column(first_col=i+1, last_col=i+1, width=column_width) # column_dimensions[str(column.title())].width = column_width
        

    df_new = df_new.style.apply(highlight_by_cell_type, axis=1)
    # print(df_new.columns)
    df_new.to_excel(writer, sheet_name = "Sheet1")
    writer.close()