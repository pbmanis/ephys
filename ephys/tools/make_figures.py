"""
Make figures - create various simple plots for mapping experiments
"""

import argparse
from pathlib import Path
import pandas as pd
import datetime
from collections import OrderedDict
import numpy as np
from statsmodels.stats.weightstats import ttest_ind
import matplotlib
matplotlib.use('Qt5Agg')

import matplotlib.pyplot as mpl
import pylibrary.plotting.plothelpers as PH
import pylibrary.tools.utility as PU
import seaborn as sns
import src.set_expt_paths
import ephys.ephys_analysis as EP
import ephys.datareaders as DR
import nf107_maps as MP

AR = DR.acq4_reader.acq4_reader()
SP = EP.spike_analysis.SpikeAnalysis()

experiments = src.set_expt_paths.get_experiments()
if "Lowest_current_spike_paramters" in self.experiment.keys():
    minimum_current = self.experiment["Lowest_current_spike_paramters"]["minimum_current"]
    minimum_postspike_interval = self.experiment["Lowest_current_spike_paramters"]["minimum_postspike_interval"]
else:
    minimum_current = 20e-12
    minimum_postspike_interval = 0.025

class Mfig(object):
    def __init__(self):
        pass
    def check_spont_protocol(self, thiscell, dbn):
        dpath = Path(experiments[dbn]['disk'], thiscell)
        # check for CC_Spont data
        ccd = list(dpath.glob('CC_Spont*'))
        if len(ccd) > 0:
            # print('ccsponts: ', ccd)
            spks = []
            for c in ccd:
                AR.setProtocol(c)
                AR.getData()
                SP.setup(clamps=AR, threshold=0., refractory=0.0007, peakwidth=0.001,
                                    verify=False, interpolate=True, verbose=False, mode='peak', min_halfwidth=0.010,
                                    lcs_minimum_current=minimum_current,
                                    lcs_minimum_postspike_interval=minimum_postspike_interval,
                                    )
                SP.analyzeSpikes()
                spks.append(SP.spikes)
            return spks
        else:
            return None    

class EPSCFig(Mfig):
    def __init__(self):
        pass

class MapFigure(Mfig):
    def __init__(self, expt=None, cell=None, protocol=None, figname='Mapfigure1.pdf', traces=None, 
                cellpos=None, bkimage=None, bestimage=None, offset=None, alpha=0.5):
        CM = MP.CellMaps(expt)  # get instance of the maps
        r = CM.getMapsandImages(cell, protocol)
        CM.map_names = [protocol]
        if bkimage is not None:
            CM.setBackground(Path(CM.datapath, cell, bkimage), offset=offset)  # can use '../image_006.tif' except it is offset even more
        CM.setReferenceImageAlpha(alpha=alpha)
        if bestimage is not None:
            CM.setBestImage(bestimage)
        CM.plotOne(figname, Path(CM.datapath, cell), protocol=protocol, cellpos=cellpos)

        mpl.show()

class SpontFig(Mfig):
    """
    Makes figure showing spontaneous activity traces
    """
    def __init__(self):
        self.dbn = ['nf107nihl', 'nf107nihl', 'nf107nihl', 'nf107']
        self.thiscell = ['2019.03.18_000/slice_002/cell_000/VC_Spont_ZeroV_000',
                         '2019.03.15_000/slice_000/cell_000/CC_Spont_ZeroI_000', 
                         '2019.03.15_000/slice_000/cell_000/CC_Spont_ZeroI_000',
                         '2017.07.12_000/slice_002/cell_000/CCIV_short_000']
        self.JP = 0
        self.recs = [0, 0, 0, 10]
        self.xr = [[0,5000], [0,5000], [2000., 2800.], [0, 300]]
        self.calx = [(0., 250.),(0., 250.), (2000., 50.), (0., 50.)]
        self.ys = [1e12, 1e3, 1e3, 1e3]
        self.caly = [[0., 50.], [-80., 25.],[-80., 25.], [-80., 25.]]
        x = -0.05
        y = 1.05
        sizer = {'A': {'pos': [0.08, 0.8, 0.82, 0.15], 'labelpos': (x,y), 'noaxes': True},
                 'B': {'pos': [0.08, 0.8, 0.45, 0.32], 'labelpos': (x,y), 'noaxes': True},
                 'C': {'pos': [0.08, 0.4, 0.05, 0.32], 'labelpos': (x,y)},
                 'D': {'pos': [0.55, 0.4, 0.05, 0.32], 'labelpos': (x,y)},
                }
            # dict pos elements are [left, width, bottom, height] for the axes in the plot.
        gr = [(a, a+1, 0, 1) for a in range(0, 4)]   # just generate subplots - shape does not matter
        axmap = OrderedDict(zip(sizer.keys(), gr))
        P = PH.Plotter((4, 1), axmap=axmap, label=True, figsize=(4,3))
        P.resize(sizer)  # perform positioning magic
        
        # self.check_spont_protocol(thiscell, dbn)
        # P = PH.regular_grid(3, 1, figsize=(4, 3), panel_labels=['A', 'B', 'C'], labelposition=(-0.05, 0.95))
        ax = P.axarr.ravel()
        for i, a in enumerate(P.axdict.keys()):
            PH.cleanAxes(P.axdict[a])
            PH.calbar(P.axdict[a], calbar=[self.calx[i][0], self.caly[i][0],
                self.calx[i][1], self.caly[i][1]], axesoff=True,
                orient='left', unitNames=None, fontsize=11, weight='normal', font='Arial')
        
        for i in range(len(self.dbn)):
            self.spont_fig(i, P, ax)
            PH.referenceline(ax[i], 0.)
        
        mpl.show()

    def spont_fig(self, i, P, ax):
        # print(self.dbn[i])
        dpath = Path(experiments[self.dbn[i]]['disk'], self.thiscell[i])
        print(dpath)
        print(dpath.is_dir())
        AR.setProtocol(dpath)
        AR.getData()
        ax[i].plot(AR.time_base*1e3, AR.traces.view(np.ndarray)[self.recs[i]]*self.ys[i]+self.JP, 'k', linewidth=0.5)
        ax[i].set_xlim(self.xr[i])
        print('ok')

def makepars(dc):
    parnames = ['offset', 'cellpos', 'invert', 'vmin', 'vmax', 'xscale', 'yscale', 'calbar', 'twin', 'ioff', 'ticks', 'notch_freqs', 'notch_q']
    pars = dict()
    for n in parnames:
        if n in ['calbar', 'twin', 'offset', 'cellpos']:
            if len(dc[n].values) == 0:
                continue
            else:
                pars[n] = eval('['+dc[n].values[0]+']')
        elif n in ['ticks']:
            if len(dc[n].values) == 0:
                continue
            else:
                pars[n] = [dc[n].values[0]]
        else:
            print(dc[n].values)
            if len(dc[n].values) == 0:
                continue
            else:
                pars[n] = dc[n].values[0]
    return pars

def useTable():
    cellchoices = [     'bushy',
                        'tstellate', 
                        'dstellate',
                        'tuberculoventral', 
                        'pyramidal', 
                        'giant', 
                        'cartwheel',
                        'octopus', 
                        'unknown', 'all']
    parser = argparse.ArgumentParser(description='Plot maps with traces on top')
    parser.add_argument('-E', '--experiment', type=str, dest='experiment',
                        choices = list(src.set_expt_paths.experiments.keys()), default='None', nargs='?', const='None',
                        help='Select Experiment to analyze')
    parser.add_argument('-c', '--celltype', type=str, default=None, dest='celltype',
                        choices=cellchoices,
                        help='Set celltype for figure')
    parser.add_argument('-s', '--sequence', type=str, default='1', dest='sequence',
                        help='sequence of ID numbers of the cells to plot')

    # parser.add_argument('-S', '--scanner', action='store_true', dest='scannerimages',
    #                     help='Plot the scanner spots on the map')
                        
    args = parser.parse_args()
    experimentname = args.experiment 
    basepath = Path(experiments[experimentname]['disk'])
    """
    Cells for paper
 
    """
    if args.celltype == 'all':
        docell = args.celltype
    else:
        docell = [args.celltype]
    
    if docell in ['unknown', 'all']:
        return
    docell = docell[0]
    sequence, target = PU.recparse(args.sequence)
    if sequence is None or len(sequence) > 1:  # invoked help
        return
    
    table = pd.read_excel('datasets/NF107Ai32_Het/Example Maps/SelectedMapsTable.xlsx')
    cellno = sequence[0]
    print('docell: ', docell)
    dc = table.loc[table['cellname'] == docell]  # get the cell type
    dc = dc.loc[table['cellno'].isin([cellno])] # then the specific cell
    
    pars = makepars(dc)
    if len(pars) == 0:
        return
    MapFigure(expt='nf107', cell=dc['cellID'].values[0], protocol=dc['map'].values[0], 
                    figname=f"{docell:s}_{int(dc['cellno']):03d}.pdf",
                    bkimage=dc['bkimage'].values[0], offset=pars['offset'], 
                    cellpos=pars['cellpos'])
    return

if __name__ == '__main__':
    # useTable()
    # exit()
    
    # SpontFig() 
    
    ct = 'bu'
    # Bushy example:
    if ct == 'bu':
        MapFigure(expt='nf107', cell='2017.10.04_000/slice_002/cell_000',
                            protocol='Map_NewBlueLaser_VC_10Hz_002',
                            figname='Bushy_lowpower.pdf',
                            bkimage='image_007.tif', offset=(-95*1e-6, -20*1e-6), 
                            cellpos=[0.0416768, 0.00640874])  # requires manual offset
    
    #Dstellate example:
    if ct == 'ds':
        MapFigure(expt='nf107', cell='2018.01.26_000/slice_000/cell_001',
                            protocol='Map_NewBlueLaser_VC_10Hz_000', bestimage='image_001.tif',
                            figname='rMP_lowpower.pdf',
                            bkimage='image_000.tif', offset=None, alpha=0.5,
                            cellpos = [0.0014559, 0.0030914])

    #Dstellate example:
    if ct == 'ts':
        MapFigure(expt='nf107', cell='2017.12.04_000/slice_001/cell_001',
                            protocol='Map_NewBlueLaser_VC_10Hz_000', bestimage='image_001.tif',
                            figname='pMP_lowpower.pdf',
                            bkimage='image_000.tif', offset=None, alpha=0.5,
                            cellpos=[0.0419920, 0.00604804])
    
    if ct == 'tv0':
        MapFigure(expt='nf107', cell='2018.01.12_000/slice_000/cell_000',
                            protocol='Map_NewBlueLaser_VC_10Hz_002', bestimage='image_002.tif',
                            figname='TV0_lowpower.pdf',
                            bkimage='image_001.tif', offset=None, alpha=0.75,
                            cellpos=[0.0405298, 0.00327486])

    if ct == 'tv':
        MapFigure(expt='nf107', cell='2018.02.27_000/slice_001/cell_000',
                            protocol='Map_NewBlueLaser_VC_10Hz_002', bestimage='image_010.tif',
                            figname='TV1_lowpower.pdf',
                            bkimage='image_009.tif', offset=None, alpha=0.75,)
                            # cellpos=[0.0405298, 0.00327486])

    if ct == 'pyr1':
        MapFigure(expt='nf107', cell='2018.02.21_000/slice_001/cell_000',
                            protocol='Map_NewBlueLaser_VC_10Hz_001', bestimage='image_003.tif',
                            figname='Pyr0_lowpower.pdf',
                            bkimage='image_000.tif', offset=None, alpha=0.75,)
                            # cellpos=[0.0405298, 0.00327486])
    if ct == 'pyr0':
        MapFigure(expt='nf107', cell='2018.02.12_000/slice_001/cell_000',
                            protocol='Map_NewBlueLaser_VC_10Hz_001', bestimage='image_003.tif',
                            figname='Pyr1_lowpower.pdf',
                            bkimage='image_000.tif', offset=None, alpha=0.75,)
                            # cellpos=[0.0405298, 0.00327486])
    if ct == 'cartwheel':
        MapFigure(expt='nf107', cell='2017.12.13_000/slice_003/cell_001',
                     protocol='Map_NewBlueLaser_VC_10Hz_004', bestimage='image_003.tif',
                     figname='CW_lowpower.pdf',
                     bkimage='image_000.tif', offset=None, alpha=0.75,)
                     # cellpos=[0.0405298, 0.00327486])
