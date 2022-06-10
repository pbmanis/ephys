from __future__ import absolute_import, print_function

import os
import pprint
import re
import sys
import timeit
from collections import OrderedDict

import numpy as np
import pandas as pd
import pylibrary.tools.fileselector as FS
import pyqtgraph as pg
from ephys.ephysanalysis import acq4read
from ephys.mapanalysistools import analyzeMapData as AMD
#from PyQt5 import pg.Qt.QtGui, pg.Qt.QtCore
from pyqtgraph.parametertree import Parameter, ParameterTree

AR = acq4read.Acq4Read()  # instance of the acq4 file reader

def readProtocol(protocolFilename, caller, records=None, sparsity=None, getPhotodiode=False):
    starttime = timeit.default_timer()
    AR.setProtocol(protocolFilename)
#    a.setProtocol('/Volumes/Pegasus/ManisLab_Data3/Kasten, Michael/2017.11.20_000/slice_000/cell_000/CCIV_4nA_max_000')
    AR.getData()
    AR.getScannerPositions()
    #mpl.plot(AR.scannerpositions[:,0], AR.scannerpositions[:,1], 'ro')
    #mpl.show()
    try:
        data = np.reshape(AR.traces, (AR.repetitions, len(AR.targets), AR.traces.shape[1]))
    except:
        caller.showErrMsg('Unable to reshape the data, probably missing records')
 
    endtime = timeit.default_timer()
    print("Time to read data: %f s" % (endtime-starttime))
    return data, AR.time_base, AR.sequenceparams, AR.scannerinfo

#def make_sortableindex(a):
  
class BuildGui():
    def __init__(self, tree):
        
        self.basename = '/Users/pbmanis/Documents/data/MRK_Pyramidal'        
        self.filename = None
        self.deltaI = 50. # pA
        self.LPF = 4000. # Hz low pass filter
        self.event_taus = OrderedDict([('EPSC_fast', [0.1, 0.2]), ('IPSC_fast', [0.33, 1.5]),
            ('IPSC_slow', [1.5, 12.])])
        self.threshold = 2.0
        self.sign = 1.0
        self.tree = tree
        self.data_plotted = False  # track plot of data
        self.measures = ['EPSC_fast', 'IPSC_fast', 'IPSC_slow']
        self.mp = dict(zip(self.measures, [[]]*len(self.measures)))  # need to keep track of all the plots 
        
        self.app = pg.mkQApp()
        self.mainwin = pg.Qt.QtGui.QMainWindow()
        self.win = pg.Qt.QtGui.QWidget()
        self.main_layout = pg.Qt.QtGui.QGridLayout()  # top level layout for the window
        self.win.setLayout(self.main_layout)
        self.mainwin.setCentralWidget(self.win)
        self.mainwin.show()
        self.mainwin.setWindowTitle('Data Selection')
        self.mainwin.setGeometry( 100 , 100 , 1400 , 900)

        # build buttons at top of controls
        self.current_DSC = list(self.tree.keys())[0]
        self.btn_read = pg.Qt.QtGui.QPushButton("Read")
        self.btn_analyze = pg.Qt.QtGui.QPushButton("Analyze")
        self.btn_test = pg.Qt.QtGui.QPushButton('Test')
        self.btn_find = pg.Qt.QtGui.QPushButton('Find and Read')
        # use a nested grid layout for the buttons
        button_layout = pg.Qt.QtGui.QGridLayout()
        button_layout.addWidget(self.btn_read,    1, 0, 1, 1)  
        button_layout.addWidget(self.btn_analyze, 0, 1, 1, 1)
        button_layout.addWidget(self.btn_test,    1, 1, 1, 1)
        button_layout.addWidget(self.btn_find,    0, 0, 1, 1)

        # build parametertree in left column
        #
        
        ptreewidth = 320
        self.main_layout.setColumnMinimumWidth(0, ptreewidth)
        
        # analysis
        params = [
            {'name': 'Analysis', 'type': 'group', 'children': [
                {'name': 'EventType', 'type': 'list', 'values': list(self.event_taus.keys()),
                        'value': list(self.event_taus.keys())},
                {'name': 'EPSC_fast_enable', 'type': 'bool', 'value': True, 'default': True},
                {'name': 'EPSC_fast_taur', 'type': 'float', 'value': 0.25, #'step': 0.05, 
                    'limits': [0.1, 10.0],
                    'suffix': 'ms'},
                {'name': 'EPSC_fast_tauf', 'type': 'float', 'value': 0.5, #'step': 0.05, 
                    'limits': [0.15, 10.0],
                    'suffix': 'ms'},
                {'name': 'EPSC_fast_threshold', 'type': 'float', 'value': 2.0, 'step': 0.1, 
                    'limits': [1., 10.0],
                    },
                
                {'name': 'IPSC_fast_enable', 'type': 'bool', 'value': True, 'default': True},
                {'name': 'IPSC_fast_taur', 'type': 'float', 'value': 0.3, 'step': 0.05, 
                    'limits': [0.1, 10.0],
                    'suffix': 'ms', 'default': 0.3},
                {'name': 'IPSC_fast_tauf', 'type': 'float', 'value': 1.0, 'step': 0.05, 
                    'limits': [0.1, 10.0],
                    'suffix': 'ms', 'default': 1.0},
                {'name': 'IPSC_fast_threshold', 'type': 'float', 'value': 2.0, 'step': .2, 
                    'limits': [1., 10.0],
                    'default': 2.0},
                
                {'name': 'IPSC_slow_enable', 'type': 'bool', 'value': True, 'default': True},
                {'name': 'IPSC_slow_taur', 'type': 'float', 'value': 2, 'step': 0.01, 
                    'limits': [1, 50.0],
                    'suffix': 'ms', 'default': 2},
                {'name': 'IPSC_slow_tauf', 'type': 'float', 'value': 12., 'step': 0.5, 
                    'limits': [1, 50.0],
                    'suffix': 'ms', 'default': 12},

                {'name': 'IPSC_slow_threshold', 'type': 'float', 'value': 2.0, 'step': .2, 
                    'limits': [1., 10.0],
                    'default': 2.0},
                ],}]
                
        self.analysis_ptree = ParameterTree()
        self.analysis_ptreedata = Parameter.create(name='params', type='group', children=params)
        self.analysis_ptree.setParameters(self.analysis_ptreedata)

        self.ptree = ParameterTree()
        self.ptreedata = Parameter.create(name='dataset', type='group', children=self.setParams(0))
        self.ptree.setParameters(self.ptreedata) # add the table with granularity of "cells"
        self.prottree = ParameterTree()
        self.setProtocols()  # add the protocols
        
        # use a grid layout to hold the trees
        self.ptree_widget = pg.Qt.QtGui.QWidget()
        self.ptree_layout = pg.Qt.QtGui.QGridLayout()
        self.ptree_widget.setLayout(self.ptree_layout)
        self.ptree_layout.setSpacing(2)
        # ptree in row 1 col 0, 4 rows, 2 cols
        self.ptree_layout.addWidget(self.analysis_ptree)
        self.ptree_layout.addWidget(self.ptree) # Parameter Tree on left
        self.ptree_layout.addWidget(self.prottree)  # protocol tree just below
#        self.ptree_layout.setColumnStretch(0, 5)
        self.ptree_layout.setRowStretch(0, 5)
        self.ptree_layout.setRowStretch(1, 1)
        self.ptree_layout.setRowStretch(2, 1)

        # build plot window 
        self.plots_widget = pg.Qt.QtGui.QWidget()
        self.plots_layout = pg.Qt.QtGui.QGridLayout()
        self.plots_widget.setLayout(self.plots_layout)
        self.plots_layout.setContentsMargins(4, 4, 4, 4)
        self.plots_layout.setSpacing(2)
        


        self.plots = {}
        for panel in zip(['Wave', 'Average', 'PSTH'], [0, 14, 18], [1, 5, 5],):
            self.plots[panel[0]] = pg.PlotWidget()
            self.plots_layout.addWidget(self.plots[panel[0]], 
                    panel[1], 0, panel[2], 1)
            self.plots[panel[0]].getAxis('left').setLabel('V', color="#ff0000")
            self.plots[panel[0]].setTitle(panel[0], color="#ff0000")
            self.plots[panel[0]].getAxis('bottom').setLabel('t (sec)', color="#ff0000")
        
        self.main_layout.addWidget(self.plots_widget, 0, 2, 22, 1)
        self.main_layout.addLayout(button_layout, 0, 0, 1, 2)       
        self.main_layout.addWidget(self.ptree_widget, 1, 0, -1, 2)        
        self.retrieveAllParameters()
        
        # connect buttons and ptrees to actions
        self.ptreedata.sigTreeStateChanged.connect(self.update_DSC)
        self.prottreedata.sigTreeStateChanged.connect(self.get_current)
        self.btn_read.clicked.connect(self.read_run)
        self.btn_analyze.clicked.connect(self.analyze)
        self.btn_test.clicked.connect(self.test)
        self.btn_find.clicked.connect(self.find_run)
        print( self.MParams)

    def showErrMsg(self, message, message2=''):
       msg = pg.Qt.QtGui.QMessageBox()
       msg.setIcon(pg.Qt.QtGui.QMessageBox.Critical)
       msg.setText(message)
       if len(message2) > 0:
           msg.setInformativeText(message2)
       msg.setWindowTitle("mapAnalysisTools Error")
       #msg.setDetailedText("The details are as follows:")
       msg.setStandardButtons(pg.Qt.QtGui.QMessageBox.Ok)
       msg.buttonClicked.connect(self.msgbtn)
       retval = msg.exec_()

    def msgbtn(self, i):
       pass  # do nothing, but could have action here


    ## If anything changes in the tree, print a message
    def change(self, param, changes):
        for param, change, data in changes:
            # path = self.ptreedata.childPath(param)
            # if path is not None:
            #     childName = '.'.join(path)
            # else:
            #     childName = param.name()
            self.current_DSC = change
            self.setProtocols()
            # print('  parameter: %s'% childName)
            # print('  change:    %s'% change)
            # print('  data:      %s'% str(data))
            # print('  ----------')

    def retrieveAllParameters(self):
        """
        get all of the local parameters from the parameter tree

        Parameters
        ----------
        ptree : ParameterTree object

        Returns
        -------
        Nothing
        """
        # fill the Parameter dictionary from the parametertree
        self.MParams = OrderedDict()

        for ch in self.analysis_ptreedata.childs:
            self.MParams[ch.name()] = {}
            for par in ch.childs:
                #print(' name: %s ' % par.name()),
                if par.type() == 'int':
                    self.MParams[ch.name()][par.name()] = int(par.value())
                elif par.type() == 'float':
                    self.MParams[ch.name()][par.name()] = float(par.value())
                elif par.type() == 'list':
                    self.MParams[ch.name()][par.name()] = str(par.value())
                elif par.type() == 'str':
                    self.MParams[ch.name()][par.name()] = str(par.value())
                elif par.type() == 'bool':
                    self.MParams[ch.name()][par.name()] = bool(par.value())

    def update_DSC(self, param, changes):
        for param, change, data in changes:
            # path = self.ptreedata.childPath(param)
            # if path is not None:
            #     childName = '.'.join(path)
            # else:
            #     childName = param.name()
            self.current_DSC = data
            self.setProtocols()

    def setParams(self, isel):
        self.params = [
            {'name': 'Day', 'type': 'group', 'children': 
                [{'name': 'Slices/Cells', 'type': 'list', 'values': list(self.tree.keys()), 'value': list(self.tree.keys())[isel]}]
            }
        ]
        return self.params
        
    def setProtocols(self): 
        """
        Update the prototocls to correspond to the current parameters, top protocol selected
        """
        self.protocols = [
            {'name': 'Protos', 'type': 'group', 'children': 
                [{'name': 'Protocols', 'type': 'list', 'values': self.tree[self.current_DSC][:], 'value': self.tree[self.current_DSC][0]}]
            }
        ]
        self.prottreedata = Parameter.create(name='protocol', type='group', children=self.protocols)
        self.prottree.setParameters(self.prottreedata)
        self.current_protocol = self.tree[self.current_DSC][0]
        self.prottreedata.sigTreeStateChanged.connect(self.get_current)
        return self.protocols

    def get_current(self, param, changes):
        for param, change, data in changes:
            # path = self.prottreedata.childPath(param)
            # if path is not None:
            #     childName = '.'.join(path)
            # else:
            #     childName = param.name()
            self.current_protocol = data

    def get_filename(self, test=False):
        if test == True:
            return self.filename
        fn = os.path.join(self.current_DSC.strip(), self.current_protocol)
        print( "filename: ", fn)
        return fn

    def test(self):
        # self.filename = os.path.join('2017.02.14_000/slice_000/cell_001/',
        #              'Map_NewBlueLaser_VC_single_MAX_002')
        self.filename = os.path.join('2017.07.17_000/slice_000/cell_001/Map_NewBlueLaser_VC_10Hz_001')
        self.filename = self.get_filename(test=True)
        self.finish_read()
        self.analyze()
    
    def read_run(self):
        self.filename = self.get_filename()
        self.finish_read()

    def find_run(self):
        fileselect = FS.FileSelector(dialogtype='dir', startingdir=self.basename)
        self.filename = fileselect.fileName
        #self.filename = self.get_filename()
        self.finish_read()
        
    def finish_read(self):
        (self.basename, self.filename)
        fullfn = os.path.join(self.basename, self.filename)
        self.data_plotted = False  # reset
        self.data, self.time_base, self.sequenceparams, self.scannerinfo = readProtocol(fullfn, self)
        for p in ['Wave', 'Average', 'PSTH']:
            self.plots[p].clear()
        self.average = self.data[0,:,:].squeeze().mean(axis=0)
        self.plots['Average'].plot(AR.time_base, 1e12*self.average)
        self.plot_traces()

    def plot_traces(self, pen='w'):
        for i in range(self.data.shape[1]):
            self.data[0,i,:] = self.data[0,i,:] - self.average  # remove average signal
            self.plots['Wave'].plot(AR.time_base, 1e12*self.data[0,i,:] + self.deltaI*i,
                pen=pg.mkPen(pen, width=1.0))
        self.data_plotted = True
        
    def analyze(self):
        if self.filename is None:
            return
        verbose = False
        self.retrieveAllParameters()
        brushes = [(0, 255, 0, 200), (255, 0, 0, 200), (0, 0, 255, 200)]
        signs = [-1, 1, 1]
        if not self.data_plotted:
            self.plot_traces(pen='w')  # make sure traces are plotted
       # print (self.MParams['Analysis'].keys())
        mpa = self.MParams['Analysis']
        self.plots['Average'].clear()
        self.plots['PSTH'].clear()
        measures = self.measures
        self.avgplots = dict(zip(measures, [[]]*len(measures)))
        self.psthplots = dict(zip(measures, [[]]*len(measures)))
        self.avg_taur = dict(zip(measures, [[]]*len(measures)))
        self.avg_tauf = dict(zip(measures, [[]]*len(measures)))

        for evi, eventtype in enumerate(measures):
            for eventplot in self.mp[eventtype]:
                self.plots['Wave'].removeItem(eventplot)
            self.mp[eventtype] = []
            for ap in self.avgplots[eventtype]:
                pg.PlotItem.removeItem(ap)
            for i in range(len(self.psthplots[eventtype])):
                pg.PlotItem.removeItem(self.psthplots[eventtype])
                
            self.avgplots[eventtype] = []
            if not mpa[eventtype+'_enable']:
                continue
            if verbose:
                print('evi: ', evi, eventtype)
            taus = [mpa[eventtype+'_taur'], mpa[eventtype+'_tauf']]
            threshold = mpa[eventtype+'_threshold']
            results = AMD.analyze_protocol(self.data, self.time_base, self.scannerinfo, 
                                            taus=taus, LPF=self.LPF, sign=signs[evi],
                                            threshold=threshold, eventhist=True)

            y = []
            for x in range(len(results['eventtimes'])):
                for xn in results['eventtimes'][x]:
                    y.append(xn)
            # et_indices = np.array(results['events'][0]['result'])  # get event time indices
            # events contains: ['avgnpts', 'peaktimes', 'avgtb', 'result', 'criteria', 'avgevent', 'smpks']

            pk_indices = np.array(results['events'][0]['smpksindex'])  # get event time indices
            pk_values  = np.array(results['events'][0]['smpks'])  # get event time indices

            responses = np.where(results['events'][0]['avgnpts'] > 0)[0] # only if there is data... 
            if len(responses) == 0:
                continue
            for i in range(self.data.shape[1]):
                p = self.plots['Wave'].plot(self.time_base[pk_indices[i]], 
                                np.array(pk_values[i])+self.deltaI*i,
                                symbol='o',
                                pen = pg.mkPen(None), symbolBrush=pg.mkBrush(brushes[evi]),
                                symbolSize=5.0,
                    )
                self.mp[eventtype].append(p)  # keep list for later removal
            # plot average event and template fit

            # plot grand mean of average events - requires removing traces with no detected events
            reva = results['events'][0]['avgevent']
            lreva = np.max([len(r) for r in reva])
            nreva = np.zeros((len(reva), lreva))
            k = 0
            for i in range(len(reva)):
                if len(reva[i]) == lreva:
                    nreva[k,:] = reva[i]
                    k += 1
            nreva = nreva[:k,:]
            allavg = np.mean(nreva, axis=0)
            self.plots['Average'].plot(results['events'][0]['avgtb'][0], allavg, pen=pg.mkPen(brushes[evi]))
            maxa = np.max(signs[evi]*allavg)
            templ = results['aj'].template*maxa/results['aj'].template_amax
            ntempl = results['events'][0]['avgtb'][0].shape[0]

            p = self.plots['Average'].plot(results['events'][0]['avgtb'][0][:ntempl] + results['aj'].tpre,
                    templ[:ntempl], pen=pg.mkPen(brushes[evi], style=pg.Qt.QtCore.Qt.DashLine))
            self.avgplots[eventtype] = [p]
            #results['aj'].fit_average_event()
            self.avg_taur[eventtype] = results['aj'].tau1
            self.avg_tauf[eventtype] = results['aj'].tau2
            print('Measure: {:s}: Best tau1: {:.3f}  tau2: {:.3f}'.format(eventtype,
                    self.avg_taur[eventtype], self.avg_tauf[eventtype]))
            p = self.plots['Average'].plot(results['aj'].avgeventtb[results['aj'].tsel:], results['aj'].best_fit,
                            pen=pg.mkPen('y', style=pg.Qt.QtCore.Qt.DashLine, linewidth=2.0))

            # plot histogram of event times
            hist, bins = np.histogram(y, 250, range=[0., 0.6], normed=1)
            p = self.plots['PSTH'].plot(bins, hist, stepMode=True, fillLevel=0., brush=brushes[evi])
            self.psthplots[eventtype] = [p]


def readtable(fn = 'MRK_VCNData.txt', listprotocols=False):
    df = pd.read_table(fn, header=1)
    #df.columns = df.columns.str.strip()
    #print df.columns.values
    df = df.set_index('Date')
    df.index = df.index.str.strip()
    df.sort_index(inplace=True)
    #
    # we now have easily accessible data frame
    #print df.loc['2016.11.18_000']  # all from this date


    allprotocols = []
    tree = OrderedDict()
    
    #print df.index[df.loc['2016.11.18_000']['Cell']].tolist()
    alldays =  sorted(set(df.index.values.tolist()))
    for day in alldays:
        subdf = df.loc[day]
        dayn = subdf.index.values.tolist()
        if isinstance(subdf['Slice'], str):
            continue
        if isinstance(subdf['Slice'], float):
            continue

        slices = subdf['Slice'].tolist()
        cells = subdf['Cell'].tolist()
        protocols = subdf['Protocols'].tolist()
#        print( '\033[1;31;40m' +dayn[0] +'\033[0;37;40m')
        for i in range(len(dayn)):
            if listprotocols:
                print ("\033[0;33;40m    "+ os.path.join(dayn[i], slices[i], cells[i])+ '\033[0;37;40m')
            prs = protocols[i][11:][1:-1].split(',')
            dsc = dayn[i] + os.sep + slices[i] + os.sep + cells[i]
            tree[dsc] = []
            for pr in prs:
                if pr == '':
                    continue
                prnopar = re.sub(r'\([^(]*\)', '', pr).strip()
                allprotocols.append(prnopar)
                nres = re.search(r'\((.*?)\)',pr)
                #print (nres)
                if nres is not None:
                    n = nres.group(1)
                else:
                    n = 0
                for i in range(int(n)):
                    if listprotocols:
                        print( '         %s_%03d' % (prnopar, i))
                    tree[dsc].append('%s_%03d' % (prnopar, i))
    
#    pprint.pprint(tree)
    print('Read {0:d} days and {1:d} protocols'.format(len(alldays), len(tree)))
    allprotocols = sorted(set(allprotocols))
    # print '\n All protocols: '
    # for p in allprotocols:
    #     print ('   %s' % p)
    return list(alldays), tree, df

if __name__== '__main__':
    alldays, tree, df = readtable()

    G = BuildGui(tree) 
    if (sys.flags.interactive != 1): #  or not hasattr(pg.Qt.QtCore, 'PYQT_VERSION'):
        pg.Qt.QtGui.QApplication.instance().exec_()
