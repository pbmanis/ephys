import ephys.datareaders.datac_reader as DR
import pylibrary.plotting.plothelpers as PH
import matplotlib.pyplot as MPL
import numpy as np
import re

re_d = re.compile("^(?P<day>[\d]{1,2})(?P<month>\w{3})(?P<year>[\d]{2})(?P<cell>[\w]{1,2})(?P<ext>.[\w]{3})")
from pathlib import Path
monthmap = {"JAN": "01", "FEB": "02", "MAR": "03", "APR": "04", "MAY": "05", "JUN": "06",
            "JUL": "07", "AUG": "08", "SEP": "09", "OCT": "10", "NOV": "11", "DEC": "12"}

somdirs1 = list(Path("/Volumes/Pegasus/ManisLab_Data3/VCN1/SOM").glob("*.SOM"))
somdirs2 = list(Path("/Volumes/Pegasus/ManisLab_Data3/VCN2/SOM").glob("*.SOM"))
pbmdirs = list(Path("/Volumes/Pegasus/ManisLab_Data3/VCN2/PBM").glob("*.PBM"))
somdirs = [*somdirs1, *somdirs2, *pbmdirs]

def datacsort(dirs):
    sortable = []
    for f in dirs:
        try:
            fn = Path(f).name
        except:
            print(f)
            exit()

        # print(fn)
        fm = re_d.match(fn)
        yr = fm.group("year")
        mo = monthmap[fm.group("month")]
        day = fm.group("day")
        letter = fm.group("cell")
        date = yr + mo + day + letter
        sortable.append(date)
    sort_index = [i for i, x in sorted(enumerate(sortable), key=lambda x: x[1])]
    # print("\nSorted: \n")
    # [print(f"   {i:03d}: {str(dirs[i]):s}") for i in sort_index]
    return [dirs[i] for i, d in enumerate(sort_index)]

fns = datacsort(somdirs)

# for i in range(10):
#     fn = fns[i]
#     D = DR.ReadDatac()
#     x = D.read_datac_header(fn)

# for f in sorted(somdirs1):
#     print(f)

# DR.example_dir_scan("/Volumes/Pegasus/ManisLab_Data3/VCN2/PBM")

DR.show_file_recs("/Volumes/Pegasus/ManisLab_Data3/VCN2/SOM/10MAY89A.SOM", 1, 16, datamode='VC')

