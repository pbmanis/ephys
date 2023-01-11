"""make_IV_table
Create an excel sheet with the data from the input .pkl file.
The resulting sheet will have the following columns:
date, slice, cell, IVs, Spikes and data_directory

For a given cell, if the IVs and Spikes are already populated in the output file,

It is expected that this will be run ONCE, as we transistion the database
so that the dataSummary file does not contain result data.

"""
import pandas as pd
from pathlib import Path

resultdisk = "/Users/pbmanis/Desktop/Python/mrk-nf107-data/datasets/NF107Ai32_Het"
datasummaryFilename = "NF107Ai32_Het.pkl"
outputfile = "NF107Ai32_IVs.xlsx"

infile = Path(resultdisk, datasummaryFilename)

df_in = pd.read_pickle(infile)
print(df_in.columns)


