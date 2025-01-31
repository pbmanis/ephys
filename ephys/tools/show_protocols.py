import pandas as pd
import datetime
from pathlib import Path
import matplotlib.pyplot as mpl
from ephys.tools import get_configuration
import ephys.datareaders.acq4_reader as acq4_reader
from pylibrary.tools import cprint

CP = cprint.cprint

AR = acq4_reader.acq4_reader()

configdir = "/Users/pbmanis/Desktop/Python/RE_CBA/config/experiments.cfg"
expt = "CBA_Age"
# configdir = "/Users/pbmanis/Desktop/Python/mrk-nf107/config/experiments.cfg"
# expt="NF107Ai32_Het"
# expt="NF107Ai32_NIHL"
# configdir = "/Users/pbmanis/Desktop/Python/Macht_Data/config/experiments.cfg"
# expt = "VM_Dentate"

config = get_configuration.get_configuration(configdir)

if expt not in config[0]:
    print(f"Experiment {expt} not found in configuration file: has {config[0]} in the entries")
    raise ValueError(f"Experiment {expt} not found in configuration file")

# print(config[1][expt])
experiment = config[1][expt]


data_summary = pd.read_pickle(Path(experiment['databasepath'], experiment['directory'], experiment['datasummaryFilename']))


def convert_protocol_string(protocol_string):
    """convert_protocol_string Change the string in the cell to a list of protocol names.


    Parameters
    ----------
    protocol_string : string
        comma separated string of protocol names

    Returns
    -------
    list
        cleaned up list of full protocol
    """
    prots = []
    prot = str(protocol_string).split(',')
    for p in prot:
        p = p.rstrip()[:-4].lstrip()
        if len(p) == 0:
            continue
        prots.append(p)
    return prots


def get_all_protocols_string(protocol_string):
    """convert_protocol_string Change the string in the cell to a list of protocol names.


    Parameters
    ----------
    protocol_string : string
        comma separated string of protocol names

    Returns
    -------
    list
        cleaned up list of full protocol
    """
    prots = []
    prot = str(protocol_string).split(',')
    for p in prot:
        p = p.rstrip().lstrip()
        if len(p) == 0:
            continue
        prots.append(p)
    return prots


def get_protocols(data_summary):
    """get_protocols Get ALL of the protocols in the data_summary file, without
    the trailing numbers. The result returned is a sorted set of unique
    protocol names that were used.

    Parameters
    ----------
    data_summary : pandas DataFrame
        The data summary file, with the 'data_complete' column containing the
        protocol names

    Returns
    -------
    list
        alphabetically sorted list of the protocols found in the dataframe
    """
    dc = data_summary['data_complete']
    prots = []
    for i, d in dc.items():
        if not d:
            continue
        protocols = convert_protocol_string(d)
        prots.extend(protocols)

    protocols = sorted(list(set(prots)))
    return protocols

def get_protocols_days(data_summary):
    protocols = get_protocols(data_summary)
    by_date = {}
   
def plot_protocols_by_date(data_summary):
    protocols = get_protocols(data_summary)
    print(f"Found {len(protocols)} unique protocols in the data summary file")
    print(f"Protocols: {protocols}")
    protodict = {}
    for d in data_summary.index:
        # print(f"Date: {data_summary.loc[d, 'date'][:-4]}")
        # print(f"Data: {data_summary.loc[d, 'data_complete']}")
        protos = convert_protocol_string(data_summary.loc[d, 'data_complete'])
        # print(f"Protocols: {protos}")
        for p in protos:
            if p not in protodict:
                protodict[p] = []
            date = data_summary.loc[d, 'date'][:-4]
            if date not in protodict[p]:
                protodict[p].append(date)
    # for prot in protodict:
    #     print(f"\nProtocol: {prot}  Used: {len(protodict[prot])} times")
    #     print("    ", protodict[prot])
    
    f, ax = mpl.subplots(1, 1, figsize=(10.5, 8.0))
    f.suptitle(f"Protocols used in {expt}")
    
    skeys = sorted(protodict.keys())
    nkey = 0
    y_labels = []
    for skey in skeys:
        if skey.startswith("CCIV_") or skey.find("CCIV_")>= 0:
            print(f"Protocol: {skey}  Used: {len(protodict[skey])} times")
            print("    ", protodict[skey])
            days = []
            for day in protodict[skey]:
                date = datetime.datetime.strptime(Path(day).name, "%Y.%m.%d")
                days.append(date)
            # print(f"    {date}")
            yd = [nkey]*len(days)
            ax.plot(days, yd, 'o', markersize=2.5, label=skey)
            # ann = ax.text(days[0], nkey, skey, fontsize=6, verticalalignment='center', ha='right')
            y_labels.append(skey)
            nkey += 1
    ax.set_yticks(range(nkey))
    ax.set_yticklabels(y_labels)
    ax.grid(True)
    mpl.tight_layout()
    now = datetime.datetime.now()
    f.text(0.02, 0.01, f"ephys/tools/show_protocols.py. Ploted: {now}", ha='left', fontsize=6)
    mpl.show()

def compare_times(t1, t2):
    missing_key = False
    for k, t in t1.items():
        if k not in t2.keys():
            CP("y", f"Key {k} not found in t2")
            missing_key = True
            continue
        if t != t2[k]:
            if k in ["start", "duration"]:
                CP("r", f"Key value {k} does not match: {t} vs {t2[k]}")
                return False
    if missing_key:
        return False
    return True

def get_protocol_timing(data_summary):
    protocols = get_protocols(data_summary)
    protodict = {}
    for ndata, d in enumerate(data_summary.index):
        # print(f"Date: {data_summary.loc[d, 'date'][:-4]}")
        # print(f"Data: {data_summary.loc[d, 'data_complete']}")
        protos = get_all_protocols_string(data_summary.loc[d, 'data_complete'])
        # print(data_summary.columns)
        datapath = Path(data_summary.loc[d, 'data_directory'], data_summary.loc[d, 'cell_id'])
        # print("\nDatapath: ", datapath)
        for p in protos:
            ppath = Path(datapath, p)
            ps = p[:-4]
            if ps not in protodict.keys():
                protodict[ps] = []
            # print(f"    Path: {ppath}, {ppath.is_dir()}")
            AR.setProtocol(ppath)
            index = AR.getIndex()
            for k in index.keys():
                if k == 'devices':
                    for dev in index[k].keys():
                        if dev=='MultiClamp1':
                            if p.find("Rheobase") >= 0:
                                continue
                            if p.find('VCIV') >= 0:
                                continue
                            if p.find('VC_Spont') >= 0:
                                continue
                            if 'stimuli' not in index[k][dev]['waveGeneratorWidget']:
                                print(f"    Path: {ppath}, {ppath.is_dir()}")
                                CP('y', "Missing stimuli?")
                                continue
                            stimuli =  index[k][dev]['waveGeneratorWidget']['stimuli']
                            # print("    Stimuli: ", stimuli)
                            if len(stimuli) == 0:
                                # print("##############\nNo stimuli found\n##############")
                                continue
                            times = AR._getPulses(stimuli)
                            # print("    times: ", times)
                            if times not in protodict[ps]:
                                protodict[ps].append(times)
                            elif times != protodict[ps]:
                                ok = compare_times(times, protodict[ps][0])
                                if ok:
                                    # print(times)
                                    CP('g', f"    {p:>42s} : times match.  start: {times['start'][0]:.3f} duration: {times['duration'][0]:.3f} (prior: {ps:s})")
                                else:
                                    CP('m', f"    Path: {ppath}, {ppath.is_dir()}")
                                    CP('r', f"    {p} times do not match for {ps}")
                                    CP('r', f"        found: {times}")
                                    CP('r', f"        prior: {protodict[ps][0]}")

                        
            # print("\n")

        # if ndata > 20:    
        #     break
    # print(protodict)

    return protodict

if __name__ == '__main__':
    # print(df_prots)

    protodict = get_protocol_timing(data_summary)
    for p in sorted(protodict):
        if len(protodict[p]) == 0:
            print(f"protocol: {p:>42s} : No information")
        else:
            print(f"protocol: {p:>42s} : {protodict[p][0]['start'][0]:.3f}  {protodict[p][0]['duration'][0]:.3f}")