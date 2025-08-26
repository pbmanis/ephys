import pandas as pd
import seaborn as sns
import datetime
import numpy as np
from pathlib import Path
from ephys.tools import get_configuration
import matplotlib.pyplot as mpl
from matplotlib.backends.backend_pdf import PdfPages

def spike_halfwidths(d, protocol_start_times, filename: Path):


    # print(f"Rs = {d['IV'][ivs[0]]['CCComp']['CCBridgeResistance']*1e-6}:.1f")
    # exit()
    f, ax = mpl.subplots(1, 1)
    ax.set_title(f"{str(Path(*filename.parts[-3:])):s}")
    f.text(0.95, 0.02, datetime.datetime.now(), fontsize=6, transform=f.transFigure, ha='right')
    if d['Spikes'] is None:
        ax.text(0.5, 0.5, s="No spikes found", ha="center", va="center", fontdict={'color': 'red', 'size': 20})
        return
    prots = list(d['Spikes'].keys())
    ivs = list(d['IV'].keys())
    colors = sns.color_palette("husl", max(len(prots), 3))

    labels = []
    for ip, pn in enumerate(prots):
        ivs = d['IV'][prots[ip]]
        if pn[:-4] in protocol_start_times:
            start_time = protocol_start_times[pn[:-4]]
        else:
            start_time = 0
        print("start time: ", start_time)
        if pn.find("1nA") > 0:
            color = colors[1]
        elif pn.find("4nA") > 0:
            color = colors[2]
        else:
            color = colors[0]
        # print(ivs['CCComp'].keys())
        Rs = ivs['CCComp']['CCBridgeResistance']*1e-6
        Cp = ivs['CCComp']['CCNeutralizationCap']*1e12
        spks = d["Spikes"][prots[ip]]['spikes']
        label=f"{pn}: Rs={Rs:.1f}, Cp={Cp:.1f}"

        for i, ns in enumerate(spks):
            if label not in labels:
                labels.append(label)
            else:
                labels.append("")
                        # print(ns, len(spks[ns]))
            lat = []
            hw = []
            for j, sn in enumerate(spks[ns]):
                if spks[ns][sn].halfwidth is None:
                    continue
                if 1e6*spks[ns][sn].halfwidth > 1000:  # long HW is artifact in analysis
                    continue
                hw.append(1e6*spks[ns][sn].halfwidth)
                lat.append(spks[ns][sn].AP_latency - start_time)
                # print("    AP Latency: ", spks[ns][sn].AP_latency-start_time, " halfwidth: ")
            ax.plot(lat, hw, 'o-', color=color, markersize=1, label=labels[-1], linewidth=0.35)
    ax.set_ylim(0, 1000)
    ax.set_xlim(-0.020, 1.0)
    ax.set_xlabel("AP Latency (s)")
    ax.set_ylabel("AP Halfwidth (us)")
    ax.legend(fontsize=5)

    # exit()

def post_stimulus_spikes(d):
    f, ax = mpl.subplots(len(d['Spikes'].keys()), 3, figsize=(10, 8))

    for axi, k in enumerate(d['Spikes'].keys()):
        print("protocol, data type: ", k, type(d['Spikes'][k]))
        print("    Spike array keys in protocol: ", d['Spikes'][k].keys())

        print("   pulse duration: ", d['Spikes'][k]['pulseDuration'])
        print("   poststimulus spike window: ", d['Spikes'][k]['poststimulus_spike_window'])
        print("  tstart: ", d['Spikes'][k]['poststimulus_spikes'])
        # print(d['Spikes'][k]['poststimulus_spikes'])
        for i, ivdata in d['IV'].items():
            print("\n   ", i, "\n", ivdata['RMP'], ivdata['RMPs'])

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

def print_LCS_spikes(d):
    # print(d['Spikes'][k]['spikes'].keys())
    # print("   ", k, "LCS: ", d['Spikes'][k]['LowestCurrentSpike'])
    for i, k in enumerate(d['Spikes'].keys()):
        if (len(d['Spikes'][k]['LowestCurrentSpike']) == 0):
            continue
        tr = d['Spikes'][k]['LowestCurrentSpike']['trace']
        dt = d['Spikes'][k]['LowestCurrentSpike']['dt']
        sn = d['Spikes'][k]['LowestCurrentSpike']['spike_no']
        # print("spike values for trace: ", tr, d['Spikes'][k]['spikes'][tr][sn])
        print("LCS spike data: ")

        print("LCS keys: ", d['Spikes'][k]['LowestCurrentSpike'].keys())
        print("   ", k, "LCS HW: ", d['Spikes'][k]['LowestCurrentSpike']['AP_HW'])
        print("   ", k, "LCS AHP Depth: ", d['Spikes'][k]['LowestCurrentSpike']['AHP_depth_V'])
        print("   ", k, "LCS AP Peak: ", d['Spikes'][k]['LowestCurrentSpike']['AP_peak_V'])
        print("   ", k, "LCS AP peak height: ", d['Spikes'][k]['LowestCurrentSpike']['AP_peak_V']) #  - d['Spikes'][k]['LowestCurrentSpike']['AP_thr_V'])
        vpk = d['Spikes'][k]['spikes'][tr][sn].peak_V*1e3  # peak value from the pkl file trace spike number
        vth = d['Spikes'][k]['spikes'][tr][sn].AP_begin_V*1e3  # threshold value from the pkl file
        icurr = d['Spikes'][k]['spikes'][tr][sn].current*1e12  # current injection value
        tro = d['Spikes'][k]['LowestCurrentSpike']['AHP_depth_V']

        print("   ", f"{k:s}, trace: {tr:d} spike #: {sn:d}  peak V: {vpk:5.1f}, thr V: {vth:5.1f}, AP Height: {vpk-vth:5.1f}")
        print(f"          AP Trough: {tro:f} current: {icurr:6.1f}")  # confirm that the threshold value is the same

    print("   ", k, "AP1HW: ", d['Spikes'][k]['AP1_HalfWidth'])

def read_pkl_file(filename):

    filename = Path(filename)
    print(filename.is_file())
    d = pd.read_pickle(filename, compression='gzip')

    return d

def hw_wrapper(adpath, exptname, celltype, experiment):
    datadir = Path(adpath, exptname, celltype)
    files = list(datadir.glob('*_IVs.pkl'))
    with PdfPages("spk_hwidths.pdf") as pdf:
        for nf, f in enumerate(files):
            print(nf, f)
            d = read_pkl_file(f)
            # if nf > 10:
            #     break
            # print(d.keys())
            spike_halfwidths(d, experiment['Protocol_start_times'], filename=f)
            mpl.suptitle(f.stem)
            pdf.savefig()
        # # post_stimulus_spikes(d)
        # print_LCS_spikes(d)



if __name__ == '__main__':
    configpath = Path("/Users/pbmanis/Desktop/Python/RE_CBA/config/experiments.cfg")
    exptname = "CBA_Age"
    celltype = 'pyramidal'
    datasets, experiments = get_configuration.get_configuration(configpath)
    experiment = experiments['CBA_Age']
    adpath = experiment['analyzeddatapath']
    # hw_wrapper(adpath, exptname, celltype, experiment)
    # exit()
    # single cell:
    pyrdatapath = Path(adpath, exptname, celltype, '2023_10_09_S1C0_pyramidal_IVs.pkl')
    d = read_pkl_file(filename = pyrdatapath)
    ivs = list(d['Spikes'].keys())
    print(d.keys())

      # print(d['IV'].keys())
    # prots = list(d['IV'].keys())
    # print(d['IV'][prots[0]].keys())
    # print(d['IV'][prots[0]]['tauh_bovera'], d['IV'][prots[0]]['tauh_tau'],d['IV'][prots[0]]['tauh_Gh'])
  
    spike_halfwidths(d, experiment['Protocol_start_times'], filename = pyrdatapath)
    mpl.show()
    # post_stimulus_spikes(d)
    # print_LCS_spikes(d)

  
    
        
