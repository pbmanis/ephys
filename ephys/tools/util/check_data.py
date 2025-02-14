from pathlib import Path
import pandas as pd
import numpy as np

""" Check the data in the directory - specifically, the dvdt_rising data as a function of the cell ID
"""
fn ="/Volumes/Pegasus_004/ManisLab_Data3/Edwards_Reginald/RE_datasets/CBA_Age/CBA_Age_combined_by_cell.pkl"

df = pd.read_pickle(fn)
for index in df.index:
    print(df.loc[index].cell_id, df.loc[index].dvdt_rising)

print("----------------------")
fne ="/Volumes/Pegasus_004/ManisLab_Data3/Edwards_Reginald/RE_datasets/CBA_Age/CBA_IVs_PCT.xlsx"
df = pd.read_excel(fne)
for index in df.index:
    print(df.loc[index].cell_id, df.loc[index].dvdt_rising)
