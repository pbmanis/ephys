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
    # pyrdatapath = Path(adpath, 'CBA_Age', 'pyramidal', '2023_11_13_S2C0_pyramidal_IVs.pkl')
    pyrdatapath = Path(adpath, 'CBA_Age', 'pyramidal', '2024_12_11_S5C0_pyramidal_IVs.pkl')
    print(pyrdatapath.is_file())
    d = pd.read_pickle(pyrdatapath, compression='gzip')
    # print(d['Spikes'].keys())
    for k in d['Spikes'].keys():
        # print("protocol, data type: ", k, type(d['Spikes'][k]))
        # print("    Spike array keys in protocol: ", d['Spikes'][k].keys())
        # print(d['Spikes'][k]['spikes'].keys())
        # print("   ", k, "LCS: ", d['Spikes'][k]['LowestCurrentSpike'])
        if (len(d['Spikes'][k]['LowestCurrentSpike']) > 0):
            tr = d['Spikes'][k]['LowestCurrentSpike']['trace']
            dt = d['Spikes'][k]['LowestCurrentSpike']['dt']
            sn = d['Spikes'][k]['LowestCurrentSpike']['spike_no']
            # print("spike values for trace: ", tr, d['Spikes'][k]['spikes'][tr][sn])
            print("LCS spike data: ")

            print("LCS keys: ", d['Spikes'][k]['LowestCurrentSpike'].keys())
            print("   ", k, "LCS HW: ", d['Spikes'][k]['LowestCurrentSpike']['AP_HW'])
            print("   ", k, "LCS AHP Depth: ", d['Spikes'][k]['LowestCurrentSpike']['AHP_depth'])
            print("   ", k, "LCS AP Peak: ", d['Spikes'][k]['LowestCurrentSpike']['AP_peak_V'])
            print("   ", k, "LCS AP peak height: ", d['Spikes'][k]['LowestCurrentSpike']['AP_peak_V']) #  - d['Spikes'][k]['LowestCurrentSpike']['AP_thr_V'])
            vpk = d['Spikes'][k]['spikes'][tr][sn].peak_V*1e3  # peak value from the pkl file trace spike number
            vth = d['Spikes'][k]['spikes'][tr][sn].AP_begin_V*1e3  # threshold value from the pkl file
            icurr = d['Spikes'][k]['spikes'][tr][sn].current*1e12  # current injection value
            tro = d['Spikes'][k]['LowestCurrentSpike']['AHP_depth']

            print("   ", f"{k:s}, trace: {tr:d} spike #: {sn:d}  peak V: {vpk:5.1f}, thr V: {vth:5.1f}, AP Height: {vpk-vth:5.1f}")
            print(f"          AP Trough: {tro:f} current: {icurr:6.1f}")  # confirm that the threshold value is the same

        # print("   ", k, "AP1HW: ", d['Spikes'][k]['AP1_HalfWidth'])
        
