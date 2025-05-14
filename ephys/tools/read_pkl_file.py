import pandas as pd
import pickle
import numpy as np
from pathlib import Path
from ephys.tools import get_configuration
import matplotlib.pyplot as mpl

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
    # print(d.keys())

    # prots = list(d['Spikes'].keys())
    # print(prots)
    # print(d["Spikes"][prots[0]]['spikes'][12][0].fields())
    # exit()

    f, ax = mpl.subplots(len(d['Spikes'].keys()), 3, figsize=(10, 8))

    for axi, k in enumerate(d['Spikes'].keys()):
        print("protocol, data type: ", k, type(d['Spikes'][k]))
        print("    Spike array keys in protocol: ", d['Spikes'][k].keys())
        exit()
        print("   pulse duration: ", d['Spikes'][k]['pulseDuration'])
        print("   poststimulus spike window: ", d['Spikes'][k]['poststimulus_spike_window'])
        print("  tstart: ", d['Spikes'][k]['poststimulus_spikes'])
        # print(d['Spikes'][k]['poststimulus_spikes'])
        for i, ivdata in d['IV'].items():
            print("\n   ", i, "\n", ivdata['RMP'], ivdata['RMPs'])
        exit()
        print("    poststimulus spikes: ", d['Spikes'][k]['poststimulus_spikes'])
        for i in range(len(d['Spikes'][k]['poststimulus_spikes'])):
            nsp = len(d['Spikes'][k]['poststimulus_spikes'][i])
            iinj = d['Spikes'][k]['FI_Curve'][0][i]
            if nsp > 0:
                ax[axi, 0].plot(d['Spikes'][k]['poststimulus_spikes'][i], [iinj]*nsp, marker='o', markersize=2, linestyle='None')
            if nsp > 1:
                dur = d['Spikes'][k]['poststimulus_spikes'][i][-1] - d['Spikes'][k]['poststimulus_spikes'][i][0]
                ax[axi, 1].plot(iinj*1e9, dur*1e3, marker='o', markersize=2, linestyle='None')
                ax[axi, 1].set_xlabel("current (nA)")
                ax[axi, 1].set_ylabel("duration (ms)")
                rate = np.mean(1./np.diff(d['Spikes'][k]['poststimulus_spikes'][i]))
                ax[axi, 2].set_title(f"{k} rate: {rate:.2f} Hz")
                ax[axi, 2].set_ylabel("rate (Hz)")
                ax[axi, 2].set_xlabel("current (nA)")
                ax[axi, 2].plot(iinj*1e9, rate, marker='o', markersize=2, linestyle='None')
    mpl.tight_layout()
    mpl.show()
    #     # print(d['Spikes'][k]['spikes'].keys())
    #     # print("   ", k, "LCS: ", d['Spikes'][k]['LowestCurrentSpike'])
    #     if (len(d['Spikes'][k]['LowestCurrentSpike']) > 0):
    #         tr = d['Spikes'][k]['LowestCurrentSpike']['trace']
    #         dt = d['Spikes'][k]['LowestCurrentSpike']['dt']
    #         sn = d['Spikes'][k]['LowestCurrentSpike']['spike_no']
    #         # print("spike values for trace: ", tr, d['Spikes'][k]['spikes'][tr][sn])
    #         print("LCS spike data: ")

    #         print("LCS keys: ", d['Spikes'][k]['LowestCurrentSpike'].keys())
    #         print("   ", k, "LCS HW: ", d['Spikes'][k]['LowestCurrentSpike']['AP_HW'])
    #         print("   ", k, "LCS AHP Depth: ", d['Spikes'][k]['LowestCurrentSpike']['AHP_depth'])
    #         print("   ", k, "LCS AP Peak: ", d['Spikes'][k]['LowestCurrentSpike']['AP_peak_V'])
    #         print("   ", k, "LCS AP peak height: ", d['Spikes'][k]['LowestCurrentSpike']['AP_peak_V']) #  - d['Spikes'][k]['LowestCurrentSpike']['AP_thr_V'])
    #         vpk = d['Spikes'][k]['spikes'][tr][sn].peak_V*1e3  # peak value from the pkl file trace spike number
    #         vth = d['Spikes'][k]['spikes'][tr][sn].AP_begin_V*1e3  # threshold value from the pkl file
    #         icurr = d['Spikes'][k]['spikes'][tr][sn].current*1e12  # current injection value
    #         tro = d['Spikes'][k]['LowestCurrentSpike']['AHP_depth']

    #         print("   ", f"{k:s}, trace: {tr:d} spike #: {sn:d}  peak V: {vpk:5.1f}, thr V: {vth:5.1f}, AP Height: {vpk-vth:5.1f}")
    #         print(f"          AP Trough: {tro:f} current: {icurr:6.1f}")  # confirm that the threshold value is the same

        # print("   ", k, "AP1HW: ", d['Spikes'][k]['AP1_HalfWidth'])
        
