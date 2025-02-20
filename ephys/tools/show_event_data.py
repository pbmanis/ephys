"""
show_event_data.py

Read an events file and show some of the data in it.
"""

import argparse
from pathlib import Path
import pickle

fn = "/Users/pbmanis/Desktop/Python/mrk-nf107-data/datasets/NF107Ai32_NIHL/events/2018.07.17_000~slice_000~cell_001.pkl"
fnp = Path(fn)
if not fnp.is_file():
    raise ValueError(f"File {fn} does not exist")

with (open(fn, "rb")) as f:
    data = pickle.load(f)

protocols = list(data.keys())
for p in protocols:
    protocol = str(Path(p).name)
    protdata = data[p]
    print(p, "\n    ", protocol)
    # print("    ", protdata['FittingResults'].keys())
    if protdata['FittingResults']['Evoked']['tau1']:
        print("     ", f"{protdata['FittingResults']['Evoked']['tau1']*1e3:.3f}, {protdata['FittingResults']['Evoked']['tau2']*1e3:.3f}")
    else:
        print("     no data were fit")
