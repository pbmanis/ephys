import pandas as pd
from pathlib import Path
import datetime
import ephys.tools.filename_tools as FT

""" Find cell pairs with the following criteria:
1. 2 different cell types, same slice
2. 2 different cell types, different slices, same day
3. 2 different cell types, different slices, different days, but with similar ages
4. same cell type in the same slice.
"""


def find_pairs(celltype1_name, celltype2_name, csv_file:str, diff_slice:bool=False, diff_date:bool=False):
    files = Path(csv_file)  # Path("R_statistics_summaries/rmtau_05-Feb-2025.csv")
    print("File exists: ", files.exists())
    df = pd.read_csv(files)
    print(df.columns)

    celltype1 = df[df["cell_type"] == celltype1_name]
    celltype2 = df[df["cell_type"] == celltype2_name]

    for row in celltype1.itertuples():
        cell_id = row.cell_id
        date = cell_id.split("_")[0]
        celltype2_on_date = celltype2[celltype2["cell_id"].str.startswith(date)]
        celltype1_id = FT.make_cellid_from_slicecell(cell_id)
        celltype1_slice = Path(celltype1_id).parts[1]
        if len(celltype2_on_date) :

            for celltype2_cell in celltype2_on_date.itertuples():
                celltype2_id = FT.make_cellid_from_slicecell(celltype2_cell.cell_id)
                sliceno = Path(celltype2_id).parts[1]
                if sliceno == celltype1_slice and celltype2_id != celltype1_id:
                    print(
                    f"Celltype 1 {celltype1_name:s}: {celltype1_id}, P{int(row.age):d} has these pyramidal cells to match: "
                        )
                    print(f"          Same Slice: {celltype2_id:s}, P{int(celltype2_cell.age):d}\n")
                else:
                    if diff_slice:
                        print(f"         Diff Slice: {celltype2_id:s}, P{int(celltype2_cell.age):d}")
        else:
            if not diff_date:
                continue
            # Search for pyr cells with same age, sorted by nearby dates
            celltype1_datetime = datetime.datetime.strptime(date, "%Y.%m.%d")
            print(
                f"{celltype1_name:s} cell: {celltype1_id} P{int(row.age):d} has these pyramidal cells to match: "
            )
            for celltype2 in celltype2_on_date.itertuples():
                celltype2_date = datetime.datetime.strptime(
                    celltype2_on_date.cell_id.split("_")[0], "%Y.%m.%d"
                )
                daydiff = abs((celltype1_datetime - celltype2_date).days)
                if daydiff < 10 and daydiff > 0 and abs(row.age - celltype2_on_date.age) < 5:
                    print(
                        f"    Close date: {celltype2_on_date.cell_id} {daydiff:d} days apart and age difference: {int(abs(row.age - celltype2_on_date.age)):d}"
                    )

            # print(f"Giant cell: {giant_id} P{int(row.age):d} has no pyramidal cells to match")
            # print("Found a pair, same date: ", cell_id, pyrs_on_date.iloc[0].cell_id)

if __name__ == "__main__":
    celltype1_name = "pyramidal"
    celltype2_name = "pyramidal"
    csv_file = "/Users/pbmanis/Desktop/Python/mrk-nf107/rmtau_12-Feb-2025.csv"

    find_pairs(celltype1_name, celltype2_name, csv_file, diff_slice=False, diff_date=False)