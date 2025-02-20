"""
Get voltage clamp parameters (series resistance, compensation, etc)
from files in a particular directory of specified protocols.

Duplicates a jupyter lab notebook of the same name (at least, at one time)

"""

import matplotlib
matplotlib.use('Qt5Agg')
import re
from IPython.display import Markdown, display
def printmd(string):
    display(Markdown(string))
from pathlib import Path
import scipy.signal
import numpy as np
from matplotlib import rc
import ephys.ephys_analysis as EP
import ephys.ephys_analysis.metaarray as EM  # need to use this version for Python 3
import ephys.tools.cursor_plot as CP
AR = EP.acq4read.Acq4Read()  # make our own private cersion of the analysis and reader

# datapath = Path('/Volumes/Pegasus/ManisLab_Data3/Kasten_Michael/Maness_PFC_stim/')
# datapath = Path('/Volumes/PBM_005/data/NF107Ai32Het')
datapath = Path('/Volumes/Pegasus/ManisLab_Data3/Kasten_Michael/NF107Ai32Het/')
plen = len(str(datapath))

###############
baseprot = {}
baseprot['VCIVs'] = ['VCIV_(\d{3})/000',
                      'VCIV_simple_(\d{3})/000',
                      'VC*(\d{3})/000',
                     ]
baseprot['Maps'] = ['MapNewBlueLaser_VC_10Hz_(\d{3})/000_000',
                    'MapNewBlueLaser_VC_Single_(\d{3})/000_000',
                    'MapNewBlueLaser_VC_single_(\d{3})/000_000',
                    'MapNewBlueLaser_VC_weird_(\d{3})/000_000',
                    'MapNewBlueLaser_VC_2mW_(\d{3})/000_000',
                    'MapNewBlueLaser_VC_1mW_(\d{3})/000_000',
                    'MapNewBlueLaser_VC_(\d{3})/000_000',
                    'MapNewBlueLaser_VC_range test_(\d{3})/000_000',
            ]
# baseprot['VC'] = ['VC-EPSC_3_(\d{3})/000_000',
#                   'VC-EPSC_3_ver2_(\d{3})/000_000',
#                   'VC-EPSC_3_ver2_gain2_(\d{3})/000_000',]
re_prot = {}

for b in list(baseprot.keys()):
    re_prot[b] = []
    for r in baseprot[b]:
        re_prot[b].append(re.compile(r))
print(re_prot)
prots = []
allprots = datapath.glob('**')
selprot = 'Maps'
# print(list(prots))
for p in allprots:
    sp = str(p)
    for i, r in enumerate(re_prot[selprot]):
        a = r.search(sp)
        if a is not None:
            prots.append(sp)
            print(f"'{str(sp)[35:]:<s}','")
print('done!')

###########################################################################
compvalues = []
rsvalues = []
nuncomp = 0
ncomp = 0
j = 0
lastp = ''
for p in prots:
    pp = Path(p).parts
    rpath = str(Path(pp[-5], pp[-4], pp[-3], pp[-2])).replace('/', '~')

#     if rpath not in datasets[selprot]:
#         continue

    AR.setProtocol(str(p))
    info = AR.readDirIndex(currdir=Path(p).parent.parent.parent.parent)  # get from the day
#     print(info['.']['age'])
#     print(info['.'].keys())
#     continue
    print(':: ', str(p))
    epos = -4
    proto_ok = AR.checkProtocol(str(p[:epos]))
#     print('protocol: ', p)
    if not proto_ok: # <span style='color:red'>Red text</span>
        printmd(f" <span style='color:red'>Protocol incomplete: {str(p)[:epos]:>s}</span>")
        continue
    try:
        info2 = AR.getIndex(str(p))
    except:
        continue #  print('failed on :', str(p))
    try:
        tr = EM.MetaArray(file=str(Path(p, AR.dataname)))
    except:
        continue
    thisp = str(p)[plen:]
    if thisp != lastp:
        print('----------\n')
        
        lastp = thisp

    info3 = tr[0].infoCopy()
    c = info3[1]['ClampState']
#     c = AR.parseClampWCCompSettings(info)
#     print(c)
    if c['mode'] == 'VC':
        cp = c['ClampParams']
#         print(cp.keys())
    
        print(f"{str(p)[plen:epos]:<s}")
        print(f"   Age: {info['.']['age']:s}   Temp: {info['.']['temperature']:s}  Internal: {info['.']['internal']:s}")
        compon = False
        cenable = 'Off'
        if cp['WholeCellCompEnable'] == 1:
            compon = True
            cenable = 'On'
            ncomp += 1
        if compon:
            compvalues.append(cp['RsCompCorrection'])
            rsvalues.append(cp['WholeCellCompResist']*1e-6)

        else:
#             compvalues.append(np.nan)
#             rsvalues.append(np.nan)
            nuncomp += 1


            
        print(f"   Comp Enabled: {cenable:>3s}, Rs= {cp['WholeCellCompResist']*1e-6:4.1f} Mohm, %Corr= {cp['RsCompCorrection']:4.2f} Hold= {c['holding']*1e3:.1f} mV")

compvalues = np.array(compvalues)
rsvalues = np.array(rsvalues)
residuals = rsvalues - rsvalues*compvalues/100.
print('-'*80)
print(f"Avg Corr: {np.nanmean(compvalues):.2f} (SD={np.nanstd(compvalues):.2f} N={compvalues.shape[0]:d})")
print(f"Avg Rs: {np.nanmean(rsvalues):.2f} (SD={np.nanstd(rsvalues):.2f})")
print(f"Avg residual Rs: {np.nanmean(residuals):.2f} (SD={np.nanstd(residuals):.2f})")
print(f"# uncompensated: {nuncomp:d}   # compensated: {ncomp:d}")


