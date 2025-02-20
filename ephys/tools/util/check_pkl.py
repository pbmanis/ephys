import pandas as pd
import pickle
from pathlib import Path
from pylibrary.tools import cprint

CP = cprint.cprint
"""
Check the pickle file - list out header columns, and also all of the paths in the datadirectory column
"""
fn = "datasets/NF107Ai32_Het/NF107Ai32_Het_bcorr2.pkl"
df = pd.read_pickle(fn)

print(df.columns)

dates = sorted(set(df['date']))

# for date in dates:
#     print("Date: ", date)
#     dd = df[df.date == date]
#     print("   Data dir: ", dd.data_directory)  # all the directories listed
#     try:
#         dpara = dd.loc[dd['data_directory'].str.contains("NF107", case=False)]
#         if len(dpara['data_directory']) > 0:
#             CP("r", f"     :: Para: {str(dpara['data_directory']):s}")
#     except:
#         pass
    
    
for date in dates:
    print("Date: ", date)
    dd = df[df.date == date]
    print("\n", dd.date.values)
    for iv in dd.IV.keys():
        prot = list(dd.loc[iv].keys())[0]
        # print("prot: ", prot)
        prot_cc = dd.loc[iv][prot]
        # print("prot_cc: ", prot_cc)
        if len(prot_cc) > 0:
            pcck = list(prot_cc.keys())
            for p in pcck:
                ccinfo = list(prot_cc[p].keys())
                print("ccinfo: ", ccinfo)
                if "CCComp" in ccinfo:
                    print("CComp: ", prot_cc[p]["CCComp"])
                else:
                    print("no cccomp in list, found:")
                    print(prot_cc[p])
                if "BridgeAdjust" in ccinfo:
                    print("Bridge Adjust: ", prot_cc[p]["BridgeAdjust"])
        # print("   dd.IV: ", prot_cc['Bridge Resostamce'])
        # print("   dd.Spikes adapt: " , dd.loc[iv].Spikes[prot]['Adap Ratio'])