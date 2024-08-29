"""Inspect pkl analysis files and plot the results."""
from pathlib import Path
import pandas as pd
import numpy as np

fn_base = "/Volumes/Pegasus_004/ManisLab_Data3/Edwards_Reginald/RE_datasets/CBA_Age"
fn_file = "pyramidal/2024_05_15_S6C1_pyramidal_IVs.pkl"

fnp = Path(fn_base, fn_file)
if not fnp.exists():
    print(f"File {fnp} not found.")
    exit()

d = pd.read_pickle(fnp, compression='gzip')

taum = []
rin = []
rmp = []

for k, v in d["IV"].items():
    # print(k, v)
    print("k: ", k)
    if isinstance(v, dict):
        print(v.keys())
        print("taum: ", v["taum"])
        print("taupars: ", v["taupars"])
        taum.append(v["taum"])
        rin.append(v["Rin"])
        rmp.append(v["RMP"])
        if 'analysistimestamp' in v.keys():
            print("analysistime: ", v["analysistimestamp"])
print(taum, np.nanmean(taum))
print(rin, np.nanmean(rin))
print(rmp, np.nanmean(rmp))


