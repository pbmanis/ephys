import pandas as pd
import pickle
import numpy as np
from pathlib import Path
from ephys.tools import get_configuration














if __name__ == '__main__':
    configpath = Path("/Users/pbmanis/Desktop/Python/RE_CBA/config/experiments.cfg")
    print(configpath.is_file())
    datasets, experiments = get_configuration.get_configuration(configpath)
    experiment = experiments['CBA_Age']
    adpath = experiment['analyzeddatapath']
    pyrdatapath = Path(adpath, 'CBA_Age', 'pyramidal', '2023_11_13_S1C1_pyramidal_IVs.pkl')
    print(pyrdatapath.is_file())
    d = pd.read_pickle(pyrdatapath, compression='gzip')
    print(d['Spikes'].keys())
    for k in d['Spikes'].keys():
        print(k, type(d['Spikes'][k]))
        print(d['Spikes'][k].keys())
        print("   ", k, "LCS: ", d['Spikes'][k]['LowestCurrentSpike'])
        if (len(d['Spikes'][k]['LowestCurrentSpike']) > 0):
            print("   ", k, "LCS HW: ", d['Spikes'][k]['LowestCurrentSpike']['AP_HW'])
            print("   ", k, "LCS AP Depth: ", d['Spikes'][k]['LowestCurrentSpike']['AHP_depth'])
        print("   ", k, "AP1HW: ", d['Spikes'][k]['AP1_HalfWidth'])

