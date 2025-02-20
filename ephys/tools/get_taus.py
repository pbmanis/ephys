import sys
if sys.version_info[0] < 3:
    exit()
# print('sys.version: ', sys.version_info)
import os
import subprocess
import argparse
from pathlib import Path
import pathlib
import datetime
import textwrap
import dill
import matplotlib.pyplot as mpl
import matplotlib
import numpy as np
import pandas as pd
from collections import OrderedDict
# matplotlib.use('Qt4Agg')
# rcParams = matplotlib.rcParams
# rcParams['svg.fonttype'] = 'none' # No text as paths. Assume font installed.
# rcParams['pdf.fonttype'] = 42
# rcParams['ps.fonttype'] = 42
#rcParams['text.latex.unicode'] = True
#rcParams['font.family'] = 'sans-serif'
#rcParams['font.sans-serif'] = 'DejaVu Sans'
# rcParams['font.weight'] = 'regular'                  # you can omit this, it's the default
# rcParams['font.sans-serif'] = ['Arial']
#rcParams['text.usetex'] = False
import seaborn as sns
import matplotlib.collections as collections
#from matplotlib.backends.backend_pdf import PdfPages

import dateutil.parser as DUP
import ephys.ephys_analysis.poisson_score as EPPS
import pylibrary.plotting.plothelpers as PH


# rcParams = matplotlib.rcParams
# #matplotlib.use("MacOSX")
# rcParams['svg.fonttype'] = 'none' # No text as paths. Assume font installed.
# rcParams['pdf.fonttype'] = 42
# rcParams['ps.fonttype'] = 42
# #rcParams['text.latex.unicode'] = True
# rcParams['font.family'] = 'sans-serif'
# rcParams['font.weight'] = 'regular'                  # you can omit this, it's the default
# rcParams['font.sans-serif'] = ['Arial']
# rcParams['text.usetex'] = True
# # Include packages `amssymb` and `amsmath` in LaTeX preamble
# # as they include extended math support (symbols, envisonments etc.)
# mpl.rcParams['text.latex.preamble']=[r"\usepackage{amssymb}",
#                                      r"\usepackage{amsmath}"]
"""
Event monger - python 3 only

Read events from the nf107_ivs map output format (pickled files, one for each cell)
Try to do some statistical evaluation of responses based on time of occurance and prob

"""

# x['events'][0][1]['fit_tau1']

import set_expt_paths as set_expt_paths
set_expt_paths.get_computer()
experiments = set_expt_paths.get_experiments()
exclusions = set_expt_paths.get_exclusions()

class GetTaus():
    def __init__(self, dataset:str="nf107", use_coding:bool=False):
        self.dataset = dataset
        self.use_coding = use_coding
        if self.dataset in list(experiments.keys()):
            self.database = Path(experiments[self.dataset]['directory'], 
                experiments[self.dataset]['datasummary']).with_suffix('.pkl')
        else:
            self.database = Path(self.dataset)
        self.databasedict = experiments[self.dataset]
        parentpath = self.database.parent
        print('Database, parentpath: ', self.database, parentpath)
        self.eventsummaryfile = Path(parentpath, str(parentpath)+'_event_summary.pkl')
        self.eventpath = Path(parentpath, 'events')
        self.map_annotationFile = Path(parentpath, experiments[self.dataset]['maps'])

        if self.map_annotationFile is not None:
            self.map_annotations = pd.read_excel(Path(self.map_annotationFile).with_suffix('.xlsx'), sheet_name='Sheet1')
            print('Reading map annotation file: ', self.map_annotationFile)
        else:
            self.map_annotations = None
        self.dbfile = Path(parentpath, experiments[dataset]['datasummary']+ '.pkl')
        self.main_db = pd.read_pickle(str(self.dbfile))
        # print(self.main_db.keys())
        # make data_complete only contain the PROTOCOLS, not the full path to dath protocol
        self.main_db['data_complete'] = self.main_db['data_complete'].map(lambda a: ', '.join(str(Path(b).name) for b in a.split(',')))
        self.colors = {'A': 'g', 'AA': 'b', 'AAA': 'r', 'B': 'k', 'C': 'c'}
        self.dtaus = {'A': [], 'B': [], 'AA': [], 'AAA': [], 'C': []}
        self.hax = []

    def find_cell(self, df, datestr, slicestr, cellstr, protocolstr=None):
        """
        Find the dataframe element for the specified date, slice, cell and protocol in the input dataframe
        """
        dstr = str(datestr)
        if protocolstr is None:
            cf = df.query(f'date == "{dstr:s}" & slice_slice == "{slicestr:s}" & cell_cell == "{cellstr:s}"')
        else:
            dprot = str(protocolstr.name)
            cf = df.query(f'date == "{dstr:s}" & slice_slice == "{slicestr:s}" & cell_cell == "{cellstr:s}" & map == "{dprot:s}"')
        return(cf)

    def with_coding(self, celltype:str=None):
        for p in self.databasedict['coding']:
            code = self.databasedict['coding'][p][1]
            if code in ['A', 'AA', 'AAA', 'B', 'C']:
                efile = f"{str(p):s}_000*.pkl"
                self.analyze(p, celltype=celltype)
    
    def multiple(self):
        f, axl = mpl.subplots(3,3, layout='tight', figsize=[10,10])
        self.ax = axl.ravel()
        celllist = ['bushy', 't-stellate', 'd-stellate', 'octopus', 'pyramidal', 'cartwheel', 'tuberculoventral']
        celllist = ['bushy']
        self.leghandles = []
        for i, name in enumerate(celllist):
            self._analyze(efile = "*.pkl", celltype=name, ax=self.ax[i])
            self.ax[i].legend(handles=self.leghandles, fontsize=6)

    def without_coding(self, celltype:str=None):
        self.figure, self.ax = mpl.subplots(1,1)
        self._analyze(efile="*.pkl", celltype=celltype)


    def _analyze(self, efile, celltype:str = None, ax=None, protocol_name="Map_NewBlueLaser_VC_10Hz"):
        dfiles = list(self.eventpath.glob(efile))
        code = "A"
        
        # print(dfiles)
        # celltype = None
        # if map_annotationFile is not None: # get from annotation file
        #     cell_df = find_cell(map_annotations, str(p), slicestr, cellstr, )
        #     celltype = cell_df['cell_type'].values[0]
        for dfile in dfiles:  # screen all pickled files
            # print("cell: ", dfile)
            dn = Path(dfile).name
            x = dn.split('~')
            date = x[0]
            slicestr = x[1]
            cellstr = x[2][:-4]
            # print(date, slicestr, cellstr)
            if self.map_annotationFile is not None: # get from annotation file
                cell_df = self.find_cell(self.map_annotations, date, slicestr, cellstr)
                try:
                    thiscelltype = cell_df['cell_type'].values[0]
                except:
                    continue
            
            if celltype is not None and thiscelltype != celltype:
                continue
            cell_main_df = self.find_cell(self.main_db, date, slicestr, cellstr)
            d_analyzed = pd.read_pickle(open(dfile, 'rb'))
            protos = list(d_analyzed.keys())
            protos = [str(p) for p in protos]
            # print('   ', protos)
            for p in protos:  # now check the protocols in the event files
                if p.find(protocol_name) >= 0:  # found the protocol
                    # print('   protocol: ', p)
                    dx = d_analyzed[Path(p)]
                    events = dx['events'][0]
                    if events is None:
                        continue
                    # print(dx['events'][0][1]['avgevent'])
                    # mpl.plot(np.array(dx['events'][0][1]['avgtb'][0])*1e3, np.array(dx['events'][0][1]['avgevent'][0])*1e12, 'k', linewidth=1)
                    try:
                        tb = np.array(events[1]['avgtb'][0])*1e3
                    except:
                        try:
                            tb = np.array(events['avgtb'][0])*1e3
                        except:
                            # print(dx['events'][0].keys())
  #                           print("whoops!@!!")
                            raise ValueError("This is not good... ")
                            continue
                    temp = cell_main_df['temperature'].values[0]
                    internal = cell_main_df['internal'].values[0]
                    tau1 = []
                    tau2 = []
                    amp = []
                    avev = []
                    if temp in ['25C', '25', 'room temp']:
                        code = "A"
                    elif temp in ['34C', '34']:
                        code = "B"
                    else:
                        print(f"Temp for cell {str(dfile):s} [{celltype:s}]  {str(p):s} is {str(temp):s}  Internal: {str(internal):s}")
                        code = "C"
                        # continue
                   # print('average event: ', dx['avgerage_event'])
                    # print('event keys: ', events.keys())
                    # print('********** fit_tau1: ', dx['events'][0]['fit_tau1'])
                    # for it in list(dx['events'][0].keys()):
                    # print(dx['events'][0].keys(), code)
                    # exit()
                    # print(p)
                    # print('Keys of dx: ', events.keys())
                    if 'fit_tau1' in list(events.keys()) and len(events['fit_tau1'])>0:
                        # print("keys: ", dx['events'][0].keys())
                        avev.append(np.array(events['avgevent']))
                        tb = np.array(events['avgtb']).squeeze()
                        tau1.append(events['fit_tau1'])
                        tau2.append(events['fit_tau2'])
                        amp.append(events['fit_amp'])
                        # print(tau1, tau2, amp)
                       # avgevent = np.mean(np.array(avev[0]), axis=0)
                        self.dtaus[code].append(np.mean(tau2)*1e3)
                        # print(f"{str(dfile):s} temp: {temp:s}", end="")
                        # print(f"    tau1: {np.nanmean(tau1)*1e3:6.2f}  ")
                        # print(f"    tau2: {np.nanmean(tau2)*1e3:6.2f}  ")
                        # print(f"EPSC: {np.nanmean(amp)*1e12:.2f}  ")
                        # print(f"tb shape: , {str(tb.shape):s}")
                        if tb.shape[0] == 0:
                            continue
                        flab = f"{str(Path(dfile).name):s}"
                        # print(flab)
                        itmax = len(tb)
                        # print(tb)
                        # print(avev)
                        # print(avev[-1].shape, itmax)
                        if avev[-1].shape[0] > 0 and len(avev[-1][0]) > 0:
                            hax, = ax.plot(tb, np.array(avev[-1]).squeeze(),  linewidth=1, label=flab)
                        # c=self.colors[code],
                            self.leghandles.append(hax)
        ax.set_title(celltype)
        # ax.set_ylim((-2e-10, 2e-10))

    def finalize(self):
        # # print(self.dtaus)
        # for x in self.dtaus.keys():
        #     print('X: ', x, self.dtaus[x], '\n')
        mpl.show()

                    # print(dx['events'][0][1]['fit_tau1'], dx['events'][0][1]['fit_tau1'])

if __name__ == '__main__':
    # f, ax = mpl.subplots(1,1)
    G = GetTaus("nf107")
    #G.without_coding(celltype="tuberculoventral")
    G.multiple()
    G.finalize()
    