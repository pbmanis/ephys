"""
CursorPlot just plots some data with cursors, and allows you to pick off values (times, amplitudes) from the
traces. This is meant to be used as a module that is called from within another module (see
PSCAnalyzer for an example of usage).
"""
import os
os.environ['no_proxy']="*"

import numpy as np
import time
from pathlib import Path
import pyqtgraph as pg
from pyqtgraph.Qt import QtGui, QtCore
from pyqtgraph.Point import Point
from pyqtgraph.parametertree import Parameter, ParameterTree
import ephysanalysis as EP
import pylibrary.fileselector as FS
import lmfit
font = QtGui.QFont()
font.setFamily('Arial')
font.setFixedPitch(False)
font.setPointSize(11)




class CursorPlot(object):
    def __init__(self, title='crosshair'):
        
        self.title = title
        self.app = QtGui.QApplication([])
        self.win = pg.GraphicsWindow()
        self.layout = QtGui.QGridLayout()
        self.win.setLayout(self.layout)
        self.win.resize(1024,800)
        self.AR = EP.acq4read.Acq4Read()  # make our own private version of the analysis and reader
        self.initParameters()
        self.buildParameters()

    def initParameters(self):
        self.nextStore = 'T0'
        self.slope = False
        self.selectedRegion = [None, None]
        self.fitx = None
        self.fity = None
        self.fitplot = None
        self.win.setWindowTitle(self.title)
        self.label = pg.LabelItem(justify='right')
        self.win.addItem(self.label)
        self.datadir = Path('/')
        self.vb = None
        self.lastx = 0.
        self.lasty = 0.
        self.minX = None
        self.maxX = None
        self.time_zero = 0.
        self.pdata = None
        self.lastmove = time.time()
        self.vLine = None
        self.hLine = None
        self.tLine = None
        self.time_region = None
        self.T0_m = 0.
        self.T1_m = 0.
        
        
        

        # self.plots['Windowed'] = self.win.addPlot(row=1, col=0)
        # self.plots['Full'] = self.win.addPlot(row=2, col=0)
    def buildParameters(self):
        params = [
                {'name': 'Cursor Plot', 'type': 'group', 'children': [
                    {'name': 'Open File', 'type': 'action'},
                    {'name': '>', 'type': 'action'},
                    {'name': '<', 'type': 'action'},
                    
                    {'name': 'Fit Exp', 'type': 'action'},
                    {'name': 'Fit EPSC', 'type': 'action'},
                    {'name': 'Save T0', 'type': 'action'},
                    {'name': 'Save T1', 'type': 'action'},
                    {'name': 'Get Region', 'type': 'action'},
                    {'name': 'Get T0,T1', 'type': 'action'},
                ],
                },
                {'name': 'Initial Fit Parameters', 'type': 'group', 'children': [
                    {'name': 'DC', 'type': 'float', 'value': 0.},
                    {'name': 'amp', 'type': 'float', 'value': 1.},
                    {'name': 'taurise', 'type': 'float', 'value': 1e-3},
                    {'name': 'taufall', 'type': 'float', 'value': 3e-3},
                ],
                },
                {'name': 'Data', 'type': 'group', 'children': [
                    {'name': 'NextStore', 'type': 'str', 'value': self.nextStore},
                    {'name': 'T0', 'type': 'float', 'value': 0.},
                    {'name': 'T1', 'type': 'float', 'value': 0.},
                    {'name': 'amp', 'type': 'float', 'value': 0.},
                    {'name': 'DC', 'type': 'float', 'value': 0.},
                    {'name': 'tau_fall', 'type': 'float', 'value': 0.},
                    {'name': 'tau_rise', 'type': 'float', 'value': 0.},
                    # {'name': 'ProtocolDir', 'type': 'list', 'values': []},
                ],
                }
            
            ]
        ptree = ParameterTree()
        ptree.setMaximumWidth(200)
        ptree.setFont(font)
        ptree.sizeHintForRow(30)
#        ptree.setUniformRowHeights(True)
        self.ptreedata = Parameter.create(name='params', type='group', children=params)
        ptree.setParameters(self.ptreedata, showTop=False)
        self.T0 = self.ptreedata.param('Data').param('T0')
        self.T1 = self.ptreedata.param('Data').param('T1')
        self.T0.sigValueChanged.connect(self.T0_Changed)
        self.T1.sigValueChanged.connect(self.T1_Changed)
        self.par_amp = self.ptreedata.param('Data').param('amp')
        self.par_DC = self.ptreedata.param('Data').param('DC')
        self.par_tau_fall = self.ptreedata.param('Data').param('tau_fall')
        self.par_tau_rise = self.ptreedata.param('Data').param('tau_rise')
        
        # self.protodir = self.ptreedata.param('Data').param('ProtocolDir')
        # self.protodir.sigValueChanged.connect(self.updateTraces)
        
        self.layout.addWidget(ptree, 0, 0, 9, 1) # Parameter Tree on left

        # add space for the graphs
        view = pg.GraphicsView()
        r_layout = pg.GraphicsLayout(border=(50,100,50))
        

        self.plots = {}
        self.pwin = None
        self.pfull = None

        self.plots['Windowed'] = r_layout.addPlot(row=0, col=1, rowspan=5, colspan=5, title="Windowed")
        self.plots['Windowed'].addItem(pg.PlotDataItem([], []))
        self.pwin = self.plots['Windowed'].items[0]

        # self.plots['Full'] = r_layout.addPlot(row=5, col=1, rowspan=1, colspan=5, title="Full")
        # self.plots['Full'].addItem(pg.PlotDataItem([], []))
        # self.pfull = self.plots['Full'].items[0]
        view.setCentralItem(r_layout)
        self.layout.addWidget(view, 1, 1, 6, 6)  # data plots on right

        self.ptreedata.sigTreeStateChanged.connect(self.command_dispatcher)  # connect parameters to their updates

        # self.region = pg.LinearRegionItem()
        # self.region.setZValue(10)
        self.time_region = pg.LinearRegionItem()
        self.time_region.setZValue(10)
        
        # Add the LinearRegionItem to the ViewBox, but tell the ViewBox to exclude this 
        # item when doing auto-range calculations.
        # self.plots['Full'].addItem(self.region, ignoreBounds=True)
        self.plots['Windowed'].addItem(self.time_region, ignoreBounds=True)

        #pg.dbg()
        self.plots['Windowed'].setAutoVisible(y=True)
        self.proxy = pg.SignalProxy(self.plots['Windowed'].scene().sigMouseMoved, rateLimit=60, slot=self.mouseMoved)
        self.proxyclick = pg.SignalProxy(self.plots['Windowed'].scene().sigMouseClicked, rateLimit=60, slot=self.mouseClicked)
    
    def command_dispatcher(self, param, changes):
        for param, change, data in changes:
            path = self.ptreedata.childPath(param)
            if path[1] == 'Save T0':
                self.nextstore = 'T0'
            elif path[1] == 'Save T1':
                self.nextstore = 'T1'
            elif path[1] == 'Get T0,T1':
                print('done')
                self.selectedRegion = [float(self.T0.value()), float(self.T1.value())]
                self.win.close()
            elif path[1] == 'Get Region':
                self.selectedRegion = self.region.getRegion()
                self.win.close()
            elif path[1] == 'Fit Exp':
                self.selectedRegion = [float(self.T0.value()), float(self.T1.value())]
                self.fitTau()
            elif path[1] == 'Fit EPSC':
                self.selectedRegion = [float(self.T0.value()), float(self.T1.value())]
                self.fitEPSC()
            elif path[1] == 'Open File':
                self.getProtocolDir()
            elif path[1] == '>':
                self.updateTraces('>')
            elif path[1] == '<':
                self.updateTraces('<')

    def getProtocolDir(self):
        
        sel = FS.FileSelector(dialogtype='dir', startingdir=self.datadir)
        print('sel filename: ', sel.fileName)
        self.clampfiles = []
        if sel.fileName is not None:
            self.pdirs = Path(sel.fileName).glob('**/MultiClamp1.ma')
            for p in self.pdirs:
                self.clampfiles.append(p)
        # print('clamp files: ', self.clampfiles)
        self.cfileptr = 0
        # print(p)
        # self.w1.slider.setValue(0)
        # # print('# clamp files: ', len(self.clampfiles))
        # self.w1.slider.setRange(0, len(self.clampfiles))
        # self.w1.slider.setMaximum(len(self.clampfiles)*self.scalar)
        # setMinimum(0)
        # self.w1.slider.setMaximum(len(self.clampfiles))
        self.protocolPath = sel.fileName
        # print('protocolpath: ', sel.fileName)
        self.AR.setProtocol(self.protocolPath)  # define the protocol path where the data is
        if self.AR.getData():  # get that data.
            self.data = self.AR.traces.view(np.ndarray)
            self.dt = 1.0/self.AR.sample_rate[0]
        if self.AR.mode in ['IC', 'CC']:
            self.scale = 1e3
            self.dmax = 200.
        else:
            self.scale = 1e12
            self.dmax = 20000.
            
        self.updateTraces(None)

    def updateTraces(self, direction):
        if direction == '<':
            self.cfileptr -= 1
            if self.cfileptr < 0:
                self.cfileptr = 0
        if direction == '>':
            self.cfileptr += 1
            if self.cfileptr >= self.data.shape[0]:
                self.cfileptr = self.data.shape[0]-1
        if direction == None:
            self.cfileptr = 0
            self.dataY = self.data[self.cfileptr]
            self.dataX = self.AR.time_base
            print('plotted')
        else:
            self.dataY = self.data[self.cfileptr]
        self.plotData(self.dataX, self.dataY)
        self.update()
        # print('dir: ', direction, '   fileptr: ', self.cfileptr, '   len data: ', self.data.shape)

        
    def T0_Changed(self):
        pass
    
    def T1_Changed(self):
        pass

    def _fcn_tau(self, params, x, data):
        """Model single exponential"""
        v = params.valuesdict()

        model = v['amp'] * np.exp(-x/v['tau_fall']) + v['DC']
        return model - data

    def fitTau(self):

        # create a set of Parameters
        params = lmfit.Parameters()
        params.add('amp', value=self.ptreedata.param('Initial Fit Parameters').param('amp').value(), min=-self.dmax, max=self.dmax)
        params.add('tau_fall', value=self.ptreedata.param('Initial Fit Parameters').param('taufall').value(), min=1e-4, max=1e-1)
        params.add('DC', value=self.ptreedata.param('Initial Fit Parameters').param('DC').value(), min=-1e3, max=1e3)
        
        t0 = self.T0.value()
        t1 = self.T1.value()
        it0 = int(t0/self.dt)
        it1 = int(t1/self.dt)
        if it0 > it1:
            t = it0
            it0 = it1
            it1 = t

        time_zero = int(self.time_zero/self.dt)
        print('timezero: ', time_zero, self.dataX[time_zero])
        # do fit, here with the default leastsq algorithm
        minner = lmfit.Minimizer(self._fcn_tau, params, fcn_args=(self.dataX[it0:it1]-self.dataX[time_zero], self.dataY[it0:it1]))
        self.fitresult = minner.minimize('leastsq')

        # calculate final result
        final = self.dataY[it0:it1] + self.fitresult.residual

        # write error report
        lmfit.report_fit(self.fitresult)

        self.updateFit(self.fitresult)
        self.fitx = self.dataX[it0:it1]
        self.fity = final
        self.showFit()

    def _fcn_EPSC(self, params, x, data):
        """Model EPSC"""
        v = params.valuesdict()

        model = v['amp'] * (((1. - np.exp(-x/v['tau_rise']))**4.0)*np.exp(-x/v['tau_fall'])) + v['DC']
        return model - data
    
    def fitEPSC(self):

        # create a set of Parameters
        params = lmfit.Parameters()
        params.add('amp', value=self.ptreedata.param('Initial Fit Parameters').param('amp').value(), min=-self.dmax, max=self.dmax)
        params.add('tau_rise', value=self.ptreedata.param('Initial Fit Parameters').param('taurise').value(), min=1e-4, max=1e-1)
        params.add('tau_fall', value=self.ptreedata.param('Initial Fit Parameters').param('taufall').value(), min=1e-4, max=1e-1)
        params.add('DC', value=self.ptreedata.param('Initial Fit Parameters').param('DC').value(), min=-1e3, max=1e3)
        dc = np.mean(self.dataY[0:10])
        params.add('DC', value=dc, min=dc-dc*1, max=dc+dc*1)
        t0 = self.T0.value()
        t1 = self.T1.value()
        it0 = int(t0/self.dt)
        it1 = int(t1/self.dt)
        if it0 > it1:
            t = it0
            it0 = it1
            it1 = t
        
        # do fit, here with the default leastsq algorithm
        time_zero = int(self.time_zero/self.dt)
        print('timezero: ', time_zero, self.dataX[time_zero])
        print(self.dataX[it0:it1]-self.time_zero)
        print(self.dataY[it0:it1])
        
        minner = lmfit.Minimizer(self._fcn_EPSC, params, fcn_args=(self.dataX[it0:it1]-self.dataX[time_zero], self.dataY[it0:it1]))
        self.fitresult = minner.minimize(method='least_squares', )

        # calculate final result
        final = self.dataY[it0:it1] + self.fitresult.residual

        # write error report
        lmfit.report_fit(self.fitresult)
        self.updateFit(self.fitresult)
        self.fitx = self.dataX[it0:it1]
        self.fity = final
        self.showFit()
    
    def updateFit(self, result):
        v = result.params.valuesdict()
        self.par_amp.setValue(v['amp']*self.scale)
        self.par_DC.setValue(v['DC']*self.scale)
        self.par_tau_fall.setValue(v['tau_fall']*1e3)
        if 'tau_rise' in list(v.keys()):
            self.par_tau_rise.setValue(v['tau_rise']*1e3)
        
    def make_testdata(self): #create numpy arrays
        #make the numbers large to show that the xrange shows data from 10000 to all the way 0
        ntr = 1
        npts = 10000
        dt = 1e-2
        dataX = np.linspace(0, dt, npts)
        self.dt = np.mean(np.diff(dataX))
        shortx = dataX[1500:5000]
        t0 = shortx[0]

        d = -200. + 10 * pg.gaussianFilter(np.random.random(size=npts), 10) #  + 10 * np.random.random(size=npts)
        d[1500:5000] += -20*(np.exp(-(shortx-t0)/1e-3)-np.exp(-(shortx-t0)/0.25e-3))
        dataY = d

        # dataX = 10000 + 15000 * pg.gaussianFilter(np.random.random(size=10000), 10) + 3000 * np.random.random(size=10000)
        return(dataX, dataY)
        
    def plotData(self, x, y, setline=True, slope=False):
        self.dataX = np.array(x)
        self.dataY = np.array(y)
        
        self.slope = slope
        if self.dataY.ndim == 1:
            self.pwin.setData(x=self.dataX, y=self.dataY, pen='c')
            #self.pdata.append(self.plots['Windowed'].plot(x=self.dataX, y=self.dataY, pen="c"))
            # self.plots['Full'].plot(x=self.dataX, y=self.dataY, pen="w")
            # self.pfull.setData(x=self.dataX, y=self.dataY, pen="w")
        else:
            # for i in range(self.dataY.shape[0]):
            #     self.pdata.append(self.plots['Windowed'].plot(x=self.dataX, y=self.dataY[i], pen="c"))
            #     self.plots['Full'].plot(x=self.dataX, y=self.dataY[i], pen="w")
            self.pwin.setData(x=self.dataX, y=self.dataY, pen='c')
            # self.pfull.setData(x=self.dataX, y=self.dataY, pen="w")

        # self.region.sigRegionChanged.connect(self.update_f)
        xmin = np.min(self.dataX)
        xmax = np.max(self.dataX)
        xmin = 0.1*(xmax-xmin) + xmin
        xmax = 0.9*(xmax-xmin) + xmin
        self.time_region.setRegion([xmin, xmax])
        self.time_region.sigRegionChanged.connect(self.update_w)
        self.plots['Windowed'].sigRangeChanged.connect(self.updateRegion)
        
        #cross hair
        if self.vLine is None:
            self.vLine = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen('y'))
            self.hLine = pg.InfiniteLine(angle=0, movable=False, pen=pg.mkPen('y'))
            self.plots['Windowed'].addItem(self.vLine, ignoreBounds=True)
            self.plots['Windowed'].addItem(self.hLine, ignoreBounds=True)

        # zero time line for fit t=0
        if self.tLine is None:
            self.tLine = pg.InfiniteLine(angle=90, movable=True, label='T0', pen=pg.mkPen('g'), hoverPen=pg.mkPen('r'))
            self.tLine.setValue(xmin+(0.5*(xmax-xmin)))
            self.tLine.sigDragged.connect(self.tlineDragged)
            self.plots['Windowed'].addItem(self.tLine, ignoreBounds=False)
            self.tLine.setZValue(100) # should be in front of region
        
        if self.vb is None:
            self.vb = self.plots['Windowed'].vb  # get the viewbox

    def showFit(self):
        if self.fitx is None:
            return
        if self.fitplot is None:
            self.fitplot = self.plots['Windowed'].plot(x=self.fitx, y=self.fity, pen=pg.mkPen('r', width=2) )
        else:
            self.fitplot.setData(x=self.fitx, y=self.fity)

    def tlineDragged(self):
        self.time_zero = self.tLine.value()

    def update_w(self):
        self.update(region=self.time_region)

    # def update_f(self):
    #     self.update(region=self.region)
        
    def update(self, region):
        region.setZValue(10)
        minX, maxX = region.getRegion()
        minX, maxX = sorted([minX, maxX])
        self.minX = minX
        self.maxX = maxX
        self.T0.setValue(self.minX)
        self.T1.setValue(self.maxX)
        # self.plots['Windowed'].setXRange(minX, maxX, padding=0)

        if not self.slope:
            return
        # operate on the data that is displayed

        minX = int(minX/self.dt)
        maxX = int(maxX/self.dt)
        if maxX > len(self.dataX):
            maxX = len(self.dataX)
            
        x0 = list(range(minX,minX+3))
        ml = maxX
        x0.extend(list(range(ml-10, ml)))

        if self.dataY.ndim == 1:
            fdx = np.array(self.dataX[minX:maxX+3])
            fdy = np.array(self.dataY[minX:maxX+3])
            pf = np.polyfit(fdx, fdy, 1)
            bline = np.polyval(pf, self.dataX)
            self.pdata[0].setData(self.dataX, self.dataY - bline)
        else:
            fdx = np.array(self.dataX[minX:maxX+3])
            for i in range(self.dataY.shape[0]):
                fdy = np.array(self.dataY[minX:maxX+3])
                pf = np.polyfit(fdx, fdy, 1)
                bline = np.polyval(pf, self.dataX)
                self.pdata[i].setData(self.dataX, self.dataY - bline)
        self.showFit()  


    def updateRegion(self, window, viewRange, region=None):
        if region is None:
            return
        rgn = viewRange[0]
        region.setRegion(rgn)
        self.rgn = rgn

    def mouseMoved(self, evt):
        # evt is a PyQt5.QtCore.QPointF
        if self.vb is None:  # wait until we have a plot
            return
        # if time.time() - self.lastmove < 0.033:  # slow down activity
        #     return
        self.lastmove= time.time()
        if self.plots['Windowed'].sceneBoundingRect().contains(evt[0]):
            mousePoint = self.vb.mapSceneToView(evt[0])
            self.lastx = mousePoint.x()
            self.lasty = mousePoint.y()
            self.vLine.setPos(self.lastx)
            self.hLine.setPos(self.lasty)

            xpt = self.lastx
            # index = int(self.lastx)
            # print('index, len: ', index, len(self.dataX))
            # if index > 0 and index < len(self.dataX):
            if self.minX is not None and (xpt >= self.minX and xpt <= self.maxX):
                self.label.setText("<span style='font-size: 18pt'><span style='font-family:courier'>x=%0.5f,   <span style='color: red'>y1=%0.5f</span>" % (mousePoint.x(), mousePoint.y()))

    def mouseClicked(self, evt):

        if self.nextStore == 'T0':
            self.T0_m.setValue(self.lastx)
            self.nextStore = 'T1'
        elif self.nextStore == 'T1':
            self.T1_m.setValue(self.lastx)
            self.nextStore = 'T0'
        else:
            pass

def main():
    import sys
    app = pg.mkQApp()
    CP = CursorPlot()
    # x, y = CP.make_testdata()
    # CP.plotData(x, y)
    #if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
    app.exec_() # QtGui.QApplication.instance().exec_()    


## Start Qt event loop unless running in interactive mode or using pyside.
if __name__ == '__main__':
    main()
    