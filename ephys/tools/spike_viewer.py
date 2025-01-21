""" spike viewer

"""
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as mpl
import ephys.tools.filename_tools as FT
import numpy as np

datapath = "/Volumes/Pegasus_004/Manislab_Data3/Edwards_Reginald/RE_datasets/CBA_Age/pyramidal"

# cellprotocol = "2023.09.12_000/slice_003/cell_000"
cellid  = "2023_11_13_S1C1"
cellprotocol = FT.make_cellid_from_slicecell(cellid)

cellid_pkl = f"{cellid:s}_pyramidal_IVs.pkl"


protocol = "CCIV_long_HK_000"

def main(filename):
    df = pd.read_pickle(filename, compression='gzip')
    print(df)
    print(df['Spikes'].keys())
    
    cellprotocols = list(df['Spikes'].keys())
    f, ax = mpl.subplots(len(cellprotocols), 1)
    if not isinstance(ax, list):
        ax = np.array(ax)
    for iprot, cellprotocol in enumerate(cellprotocols):
        spikes = df['Spikes'][cellprotocol]
        print("filename: ", filename)
        print("cell protocol: ", cellprotocol)
        # print(spikes.keys())
        # print(spikes['AP1_HalfWidth'])
        # print(len(spikes['spikes']))
        p1 = False
        ax[iprot].set_title(f"{filename.name!s}  {cellprotocol:s}", fontsize=8)
        for spike in spikes['spikes']:
            spk = spikes['spikes'][spike]
            for spks in spk:
                this_spike = spikes['spikes'][spike][spks]
                # print(dir(this_spike))
                # exit()
                if this_spike.halfwidth is not None:
                    print(f"{spike:4d} {this_spike.AP_number:4d}: {this_spike.AP_latency*1e3:6.1f} {this_spike.halfwidth*1e3:6.3f}, {this_spike.peak_V*1e3:6.1f}")
                    if  this_spike.AP_number == 0:
                        ax[iprot].plot(this_spike.Vtime-this_spike.Vtime[0], this_spike.V, linewidth=0.33)
                print()
            # if spk['halfwidth'] is not None:
            #     print(f"Halfwidth: {spk.halfwidth}")
    f.tight_layout()
    mpl.show()

if __name__ == "__main__":
    fn = Path(datapath, cellid_pkl)
    main(fn)