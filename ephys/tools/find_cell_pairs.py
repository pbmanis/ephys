import pandas as pd
from pathlib import Path

    """ Find cell pairs with the following criteria:
    1. 2 different cell types, same slice
    2. 2 different cell types, different slices, same day
    3. 2 different cell types, different slices, different days, but with similar ages

    """

files  = Path("R_statisc_sumaries/rmtau_05-feb02025.csv")
print("File exists: ", files.exists())
df = pd.read_csv(files)

giants = df[df["cell_type"] == "giant"]
pyramids = df[df["cell_type"] == "pyramidal"]
print("Giants: ", giants)