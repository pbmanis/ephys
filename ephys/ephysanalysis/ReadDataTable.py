import pandas as pd
from pathlib import Path
import sys
import re
from collections import OrderedDict
import pyqtgraph as pg
from pyqtgraph.parametertree import Parameter, ParameterTree

from ephysanalysis import acq4read
from mapanalysistools import analyzeMapData as AMD
import pylibrary.fileselector as FS


class ReadDataTable():

    def __init__(self):
        """
        """
        self.tree = None
        self.ptree = ParameterTree()

        # self.ptreedata = Parameter.create(name='dataset', type='group', children=self.setParams(0))
        # self.ptree.setParameters(self.ptreedata) # add the table with granularity of "cells"
        
        self.prottree = ParameterTree()
        # self.setProtocols()  # add the protocols
        
        # self.ptree_layout.addWidget(self.analysis_ptree)
        # self.ptree_layout.addWidget(self.ptree) # Parameter Tree on left
        # self.ptree_layout.addWidget(self.prottree)  # protocol tree just below
        #

    def get_filename(self, test=False):
        if test == True:
            return self.filename
        fn = Path(self.current_DSC.strip(), self.current_protocol)
        print( "filename: ", fn)
        return fn
    
    def readtable(self, fn=None, datainfo='protocols', listprotocols=False):
        if datainfo == 'protocols':
            dataname = 'data_complete'
        if datainfo == 'images':
            dataname = 'data_images'
        df = pd.read_pickle(open(fn, 'rb'))

        allprotocols = []
        self.tree = OrderedDict()
        alldays =  sorted(set(df.index.values.tolist()))
        for day in alldays:
            subdf = df.loc[day]
            print(subdf)
            dayn = subdf.index.values.tolist()

            slices = subdf['slice_slice']
            cells = subdf['cell_cell']
            protocols = subdf[dataname]
            date = subdf['date']
            dsc = str(Path(date, slices, cells))
            self.tree[dsc] = []
            prs = protocols.split(',')
            for i in range(len(prs)):
                if listprotocols:
                    print ("\033[0;33;40m    "+ str(Path(dayn, slices, cells))+ '%s'%prs[i] + '\033[0;37;40m')
                self.tree[dsc].append(prs[i])

        print(f'Read {len(alldays):d} records with {len(self.tree):d} {dataname:s}')
        allprotocols = sorted(set(allprotocols))

        return list(alldays), self.tree, df


class BuildGui():
    def __init__(self, tree):
        """
        Test fixture
        """
        
        self.basename = '/Users/pbmanis/Documents/data/MRK_Pyramidal'        
        self.filename = None
        self.tree = tree
        print('tree: ', tree)
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
        self.btn_find = pg.Qt.QtGui.QPushButton('Find and Read')
        # use a nested grid layout for the buttons
        button_layout = pg.Qt.QtGui.QGridLayout()
        button_layout.addWidget(self.btn_read,    1, 0, 1, 1)  
        # button_layout.addWidget(self.btn_analyze, 0, 1, 1, 1)
        button_layout.addWidget(self.btn_find,    0, 0, 1, 1)

        # build parametertree in left column
        #
        ptreewidth = 320
        self.main_layout.setColumnMinimumWidth(0, ptreewidth)
        
        # analysis   # empty in test rig
        params = [
            {'name': 'Analysis', 'type': 'group', 'children': [],
            }]
                
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
        # self.btn_analyze.clicked.connect(self.analyze)
        self.btn_find.clicked.connect(self.find_run)
        # print( self.MParams)
    
    def retrieveAllParameters(self):
        pass
    
    def read_run(self):
        pass
    
    def find_run(self):
        pass
    
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
        if self.tree == None:
            raise ValueError('setProtocols: Must set up read data before setting up protocols')
        self.protocols = [
            {'name': 'Protos', 'type': 'group', 'children': 
                [{'name': 'Protocols', 'type': 'list',
                'values': self.tree[self.current_DSC][:], 'value': self.tree[self.current_DSC][0]}]
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
            
    def update_DSC(self, param, changes):
        for param, change, data in changes:
            # path = self.ptreedata.childPath(param)
            # if path is not None:
            #     childName = '.'.join(path)
            # else:
            #     childName = param.name()
            self.current_DSC = data
            self.setProtocols()       

        
def test():
    app = pg.mkQApp()
    RDT =  ReadDataTable() 
    alldays, tree, df = RDT.readtable(fn='/Users/pbmanis/Desktop/Python/mrk-nf107/NF107Ai32_Het/NF107Ai32_Het.pkl',
        datainfo='images')

    G = BuildGui(tree)
    if (sys.flags.interactive != 1): #  or not hasattr(pg.Qt.QtCore, 'PYQT_VERSION'):
        pg.Qt.QtGui.QApplication.instance().exec_()



if __name__ == '__main__':
    test()
