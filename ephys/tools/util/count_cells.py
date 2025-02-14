# %%
import pandas as pd
import openpyxl as OX
from pathlib import Path
import datetime
import src.set_expt_paths as SEP

baseDataDirectory, codeDirectory, datasetDirectory = SEP.get_paths()

print(datasetDirectory)

fn = Path(datasetDirectory, "NF107Ai32_Het/NF107Ai32_Het_maps.xlsx")
#fn = "../datasets/VGAT_NIHL/VGAT_NIHL_maps.xlsx"

d = pd.read_excel(fn)


# %%
print(d.head(2))

# %%
"""
Generate dates for date clipping
"""
d['cname'] = d['date'] + d['slice_slice'] + d['cell_cell'] 
d['datestr'] = d['date'].str.slice(0, 10)
print(d['datestr'])
d['datenum'] = pd.to_datetime(d['datestr'], format='%Y.%m.%d')

# %%
"""
Fix cell names as they were not always consistent
"""
d['cell_type'] = d['cell_type'].replace('glial cell', 'glial')
d['cell_type'] = d['cell_type'].replace('giant cell', 'giant')
d['cell_type'] = d['cell_type'].replace('nan', 'unknown')
d['cell_type'] = d['cell_type'].replace(' ', 'unknown')


print(set(d['cell_type']))

# %%
sn_all = set(d['cname'])
sn = set(d['cname'][d['Usable']!=0])
print("sn: ", len(sn), "sn_all: ", len(sn_all))
usable = d['cname'][d['Usable']!=0]

# %%
celltypes = ['bushy', 't-stellate', 'd-stellate', 
             'octopus', 'pyramidal',
             'cartwheel', 'tuberculoventral', 
             'giant', 'unknown']
dcncelltypes = ['pyramidal',
             'cartwheel', 'tuberculoventral', 
             'giant', ]
def count_cells(startdate):
    print(f"Since: {str(startdate):s}")
    nct = 0
    nc = 0
    ndcn = 0
    for c in celltypes:
        cok = set(usable[(d['cell_type'] == c) & (d['datenum'] >= startdate)])
        print(f"{c:>20s} : {len(cok):4d} cells")
        if c != 'unknown':
            nc += len(cok)
        nct += len(cok)
        if c in dcncelltypes:
            ndcn += len(cok)
    print(f'    Total known cells: {nc:4d}; total cells: {nct:4d}  dcn cells: {ndcn:4d}\n')

startdate = datetime.datetime(2010, 1, 1)
count_cells(startdate)
startdate = datetime.datetime(2020, 7, 1)
count_cells(startdate)




# %%



