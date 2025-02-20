from pathlib import Path
import pandas as pd

code = Path("/Users/pbmanis/Desktop/Python/mrk-nf107-data/datasets/NF107Ai32_NIHL/NF107Ai32_NoiseExposure_Code.xlsx")
datapath = Path("/Users/pbmanis/Desktop/Python/mrk-nf107-data/datasets/NF107Ai32_NIHL/")
print(Path(code))
print(code.is_file())
codedb = pd.read_excel(code, sheet_name='SubjectData').reindex()
print(codedb.head())

celltypes = ['pyramidal', 'cartwheel', 'tuberculoventral']

ds = []
for ct in celltypes:
    cnames = sorted(list(Path(datapath, ct).glob(f"*_{ct:s}_maps.pdf")))
    for cell in cnames:
        cn = str(cell.name)
        cn = cn.replace("_", '.')
        day = cn[:10]
        slicecell = cn[11:15]

        d = codedb.loc[codedb[d].str.endswith(day)]
        ds.append({'Cell_ID': day, 'slicecell': slicecell, 'Group': d.Group.values[0], 'CellType': ct})
df = pd.DataFrame(ds)
print(df.head(20))
df.to_excel(Path(datapath, 'NF107Ai32_NIHL_coded_map_summary.xlsx'))
    
