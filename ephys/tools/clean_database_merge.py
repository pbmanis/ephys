import numpy as np
import pandas as pd
from pathlib import Path
from typing import Union
from ephys.tools import parse_ages

"""Perform a merge of the main pickled database and a coding file,
using the coding sheet. 
This also performs some cleaning of the database, such as
groups and ages

"""
def clean_database_merge(pkl_file: Union[str, Path], coding_file: Union[str, Path], coding_sheet:str):

    def mapdate(row):
        if not pd.isnull(row["Date"]):
            row["Date"] = row["Date"] + "_000"
        return row
    
    def sanitize_age(row, agename = "Age"):
        row[agename] = parse_ages.ISO8601_age(row[agename])
        return row
    
    print("Reading pkl file: ", pkl_file)
    print(f"    File exists: {str(Path(pkl_file).is_file()):s}")
    df = pd.read_pickle(pkl_file) # , compression={'method': 'gzip', 'compresslevel': 5, 'mtime': 1})
    if "ID" in df.columns:
        df = df.rename(columns={"ID": "Animal_ID"})

    df = df.apply(sanitize_age, axis=1)
    print(f"    With {len(df.index):d} entries")
    df["cell_type"] = df["cell_type"].values.astype(str)
    
    def _cell_type_lower(row):
        row.cell_type = row.cell_type.lower()
        return row
        
    df = df.apply(_cell_type_lower, axis=1)
    df.reset_index(drop=True)
    if coding_sheet is None:
        return df
    
    df_c = pd.read_excel(
        Path(coding_file),
        sheet_name=coding_sheet,
        )
    print(f"    Successfully Read Coding sheet {str(coding_file)}.pkl")
    print(f"    With these columns: {str(df_c.columns):s}")
    print(f"    and {int(np.max(df_c.index.values)):d} entries")
    gr = list(set(df_c.Group.values))
    print(f"    With these Groups: {str(gr):s}")

    df_c["Group"] = df_c['Group'].values.astype(str)
    # df_c["age"] = df_c["age"].values.astype(str)
    # df_c = df_c.apply(mapdate, axis=1)
    df_c = df_c.apply(sanitize_age, axis=1, agename='Coded_Age')
    df_c["Group"] = df_c['Group'].values.astype(str)
    # print(df.columns)
    # print(df_c.columns)
    # print("df dates: ", df.Date.values)
    # print("df_c dates: ", df_c.date.values)
    df_i = pd.merge(
                    df,
                    df_c,
                    left_on=["Date"],  # , "slice_slice", "cell_cell"],
                    right_on=["date"],  # , "slice_slice", "cell_cell"],
                    suffixes=('', '_coding'),
                )
    print(f"    Merge has {len(df_i.index):d} entries")
    return df_i
