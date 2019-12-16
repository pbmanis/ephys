from __future__ import print_function

"""
mini_summary_plots.py
Simple plots of data points for mini analysis from mini_analysis.py

Paul B. Manis, 3/2018
"""

import os
import sys
import re
import pickle
import numpy as np
from collections import OrderedDict
from matplotlib import rc
import matplotlib.pyplot as mpl
import pandas as pd
import seaborn as sns
import scipy.stats

rc('text', usetex=False)
#rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})

import pylibrary.plotting.plothelpers as PH

class MiniSummarize():
    
    # each mouse m entry in d has the following keys:
    # ['amplitudes', 'genotype', 'intervals', 'mouse', 'protocols', 'eventcounts']
    def __init__(self, id):
        self.experiment_id = id

    def load_file(self, fn):
        fh = open(fn, 'rb')
        self.d = pickle.load(fh) # , encoding='latin1')  # encoding needed for python 2 file written, to read in python 3
        fh.close()
        self.filename = fn
        print ('Mice/cells: ', self.d.keys())

    def set_groups(self, groups):
        self.group_names = groups 
        # ['WT', 'CHL1']
        #self.groups = ['F/+', 'F/F']
        
    def average_taus(self, d, t):
        tau = np.zeros(len(d))
        for i in range(len(d)):
            tau[i] = d[i]['fit'][t]
        return np.mean(tau)

    def compute_means(self):
        self.gtypes = []
        self.holding = []
        self.amps = []
        self.meanamps = []
        self.intvls = []
        self.mouse = []
        self.nevents = []
        self.tau1 = []
        self.tau2 = []
        for i, m in enumerate(self.d.keys()):
            #print ('m: ', m)
            gt = self.d[m]['genotype']
            #print('gt: ', gt)
            # if gt not in self.gtypes:

            self.gtypes.append(gt)
            # if i == 0:
            #     print( self.d[m].keys())
            self.holding.append(self.d[m]['holding'])
            self.amps.append(np.nanmean(self.d[m]['amplitude_midpoint']))
            self.meanamps.append(np.nanmean(self.d[m]['amplitudes']))
            self.intvls.append(np.nanmean(self.d[m]['intervals']))
            self.nevents.append(len(self.d[m]['intervals']))
            self.tau1.append(self.average_taus(d[m]['averaged'], 'tau1'))
            self.tau2.append(self.average_taus(d[m]['averaged'], 'tau2'))
            self.mouse.append(m)
        self.pddata = pd.DataFrame({'Genotype': pd.Categorical(self.gtypes),
                                    'Holding': np.array(self.holding),
                                    'Amp': np.array(self.amps),
                                    'MeanAmp': np.array(self.meanamps),
                                    'Intvls': 1000./np.array(self.intvls),
                                    'Nevents': np.array(self.nevents),
                                    'tau1' : np.array(self.tau1),
                                    'tau2' : np.array(self.tau2),
                                    'Mouse': pd.Categorical(self.mouse)
                                    })
        ps = str(self.pddata)
        ps = re.sub(' +', '\t', ps)
       # print(ps)
        df = self.pddata
        g1 = []
        g2 = []
        for i in range(len(self.gtypes)):
            if self.gtypes[i] == "F/+":
                g1.append(self.amps[i])
            else:
                g2.append(self.amps[i])

        gm1 = []
        gm2 = []
        for i in range(len(self.gtypes)):
            if self.gtypes[i] == "F/+":
                gm1.append(self.meanamps[i])
            else:
                gm2.append(self.meanamps[i])
      #  print( g1, g2)
        t, p = scipy.stats.ttest_ind(g1, g2, axis=0, equal_var=False, nan_policy='propagate')
        print('F/+: u = {0:.3f} (SD={1:.3f})'.format(np.mean(g1), np.std(g1)))
        print('F/F: u = {0:.3f} (SD={1:.3f})'.format(np.mean(g2), np.std(g2)))
        print('t= {0:.3f}, p={1:.3f}'.format(t, p))

        tm, pm = scipy.stats.ttest_ind(gm1, gm2, axis=0, equal_var=False, nan_policy='propagate')
        print('F/+: u = {0:.3f} (SD={1:.3f})'.format(np.mean(gm1), np.std(gm1)))
        print('F/F: u = {0:.3f} (SD={1:.3f})'.format(np.mean(gm2), np.std(gm2)))
        print('t= {0:.3f}, p={1:.3f}'.format(tm, pm))

    # print (self.amps)
    # print (self.meanamps)
    # print (self.intvls)
        self.plot()
        
    # compute stats on the ampitudes and intervals
    def plot(self):
        sizer = OrderedDict([ ('A', {'pos': [0.12, 0.2, 0.15, 0.5]}),
                              ('B', {'pos': [0.45, 0.2, 0.15, 0.5]}),
                              ('C', {'pos': [0.72, 0.2, 0.15, 0.5]}),
                             ])  # dict elements are [left, width, bottom, height] for the axes in the plot.
        n_panels = len(sizer.keys())
        gr = [(a, a+1, 0, 1) for a in range(0, n_panels)]   # just generate subplots - shape does not matter
        axmap = OrderedDict(zip(sizer.keys(), gr))
        P = PH.Plotter((n_panels, 1), axmap=axmap, label=True, labeloffset=[-0.15, 0.08], 
            fontsize={'tick': 8, 'label': 10, 'panel': 14}, figsize=(5., 3.))
        P.resize(sizer)  # perform positioning magic

        # P.axdict['A'].plot(np.ones(len(self.intvls[self.group_names[0]])), 1000./np.array(self.intvls[self.group_names[0]]),
        #     'ko', markersize=4.0, label=self.group_names[0])
        # P.axdict['A'].plot(1+np.ones(len(self.intvls[self.group_names[1]])), 1000./np.array(self.intvls[self.group_names[1]]),
        #     'bs', markersize=4.0, label=self.group_names[1])
        # P.axdict['B'].plot(np.ones(len(self.amps[self.group_names[0]])), self.amps[self.group_names[0]],
        #      'ko', markerfacecolor='k', markeredgecolor='k', markersize=4.0, markeredgewidth=1)
        # P.axdict['B'].plot(1.0 + np.ones(len(self.amps[self.group_names[1]])), self.amps[self.group_names[1]],
        #     'bs', markerfacecolor='b', markeredgecolor='b', markersize=4.0, markeredgewidth=1)
        # P.axdict['B'].plot(0.2 + np.ones(len(self.meanamps[self.group_names[0]])), self.meanamps[self.group_names[0]],
        #     'ko', markerfacecolor='w', markeredgecolor='k', markersize=4.0, markeredgewidth=1)
        # P.axdict['B'].plot(1.2 + np.ones(len(self.meanamps[self.group_names[1]])), self.meanamps[self.group_names[1]],
        #     'bs', markerfacecolor='w', markeredgecolor='b', markersize=4.0, markeredgewidth=1)        
        for a in ['A', 'B', 'C']:
            PH.formatTicks(P.axdict[a], font='Helvetica')        
        sns.swarmplot(x='Genotype', y='Intvls', data=self.pddata, ax=P.axdict['A'])
        sns.boxplot(x='Genotype', y='Intvls', data=self.pddata, ax=P.axdict['A'], color="0.8")

        sns.swarmplot(x='Genotype', y='Amp', data=self.pddata, ax=P.axdict['B'])
        sns.boxplot(x='Genotype', y='Amp', data=self.pddata, ax=P.axdict['B'], color="0.8")

        sns.swarmplot(x='Genotype', y='MeanAmp', data=self.pddata, ax=P.axdict['C'])
        sns.boxplot(x='Genotype', y='MeanAmp', data=self.pddata, ax=P.axdict['C'], color="0.8")

        P.axdict['A'].set_ylim(0.0, 25.)
        P.axdict['A'].set_ylabel('Event Frequency (Hz)')
        P.axdict['B'].set_ylim(0.0, 30.)
        P.axdict['B'].set_ylabel('Median Amplitude (pA)')
        P.axdict['C'].set_ylabel('Mean Amplitude (pA)')
        P.axdict['C'].set_ylim(0.0, 30.)
        
        # P.axdict['A'].set_xlabel('Group')
        # P.axdict['B'].set_xlabel('Group')
        # P.axdict['A'].set_xlim(0.5, 2.5)
        # P.axdict['B'].set_xlim(0.5, 2.5)
        # P.axdict['B'].set_xlim(0.5, 2.5)

       # P.axdict['A'].legend()
        P.figure_handle.suptitle(self.filename.replace('_', '\_'))
        
        
        
        mpl.savefig('msummary_%s.pdf' % self.experiment_id)
        mpl.show() 


if __name__ == '__main__':
    fn = 'summarydata_%s.p' % sys.argv[1]  # data file to plot from - should hold a dict of mouse entries
    MS = MiniSummarize(sys.argv[1])
    MS.load_file(fn)
    MS.set_groups(['F/+', 'F/F'])
    #MS.set_groups(['WT', 'CHL1'])
    MS.compute_means()

    
