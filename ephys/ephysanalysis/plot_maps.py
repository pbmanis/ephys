from __future__ import print_function
#!/usr/bin/python

"""
Simple adjunct routine to plot LSPS/CRACM maps with traces, over cell image if possible.
Reuqires acq4read. 

Takes advantage of acq4read code to access all data and attributes.


"""
import os
import re
import itertools
import argparse
import json
from collections import OrderedDict
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as mpl
from matplotlib.widgets import RectangleSelector
import matplotlib.backend_bases as MBB
import scipy.ndimage as SND
import shapely as SH
import shapely.affinity as SHA
import shapely.geometry as SG
import descartes

from pylibrary.plotting import plothelpers as PH
import pylibrary.utility as PU
import seaborn as sns
import ephysanalysis.acq4read as ARC
import ephysanalysis.metaarray as EM
from pyqtgraph import configfile
from pylibrary.plotting import picker
import scipy.ndimage
import scipy.signal
import numpy as np
import datetime
import pprint
import textwrap as WR
import collections
import tifffile as tf
import ephysanalysis.boundrect as BR
import mapanalysistools.digital_filters as FILT
import montage
import mahotas as MH


class ScannerInfo(object):
    """
    Get scanner information, compute the scanner box and some additional parameters
    Do this as a class to encapsulate the data and make reusable.
    """
    def __init__(self, AR, offset):
        # print('ScannerInfo called')
        BRI = BR.BoundRect()
        self.offset = offset
        self.AR = AR  # save the acq4read instance for access to the data
        self.AR.getScannerPositions()
        self.scannerpositions = np.array(AR.scannerpositions)
        for i, s in enumerate(self.scannerpositions):
            self.scannerpositions[i,0] += float(self.offset[0])
            self.scannerpositions[i,1] += float(self.offset[1])
        print(self.scannerpositions[:10])
        
        pos = self.AR.scannerCamera['frames.ma']['transform']['pos']
        scale = self.AR.scannerCamera['frames.ma']['transform']['scale']
        region = self.AR.scannerCamera['frames.ma']['region']
        self.binning = self.AR.scannerCamera['frames.ma']['binning']
        # print('Scanner pos, scale, region: ', pos, scale, region)
        # print('Scanner binning: ', self.binning)
        binning = self.binning
        scale = list(scale)
        self.scale = scale
        if self.AR.spotsize is None:
            self.AR.spotsize=50.
            print ('Spot Size reset to: {0:0.3f} microns'.format(self.AR.spotsize*1e6))
        x0 = pos[0] + scale[0]*region[0]/binning[0]
        x1 = pos[0] + scale[0]*(region[0]+region[2])/binning[0]
        y0 = pos[1] + scale[1]*region[1]/binning[1]
        y1 = pos[1] + scale[1]*(region[1]+region[3])/binning[1]
        self.camerabox = [[x0, y0], [x0, y1], [x1, y1], [x1, y0], [x0, y0]]
        # self.camerabox = [[pos[0] + scale[0]*region[0]/binning[0], pos[1] + scale[1]*region[1]/binning[1]],
        #        [pos[0] + scale[0]*region[0]/binning[0], pos[1] + scale[1]*region[3]/binning[1]],
        #        [pos[0] + scale[0]*region[2]/binning[0], pos[1] + scale[1]*region[3]/binning[1]],
        #        [pos[0] + scale[0]*region[2]/binning[0], pos[1] + scale[1]*region[1]/binning[1]],
        #        [pos[0] + scale[0]*region[0]/binning[0], pos[1] + scale[1]*region[1]/binning[1]]
        #    ]
        scannerbox = BRI.getRectangle(self.AR.scannerpositions)
        self.scanner_sh = SH.geometry.MultiPoint(self.AR.scannerpositions)
        self.envelope_sh = self.scanner_sh.envelope
        self.centroid_sh = self.scanner_sh.centroid
        if scannerbox is None:  # likely just one point
            pt = self.AR.scannerpositions
            fp = np.array([[pt[0][0]], [pt[0][1]]])
            scannerbox = fp
        else:
            fp = np.array([scannerbox[0][0], scannerbox[1][1]]).reshape(2,1)
        # print('fp: ', fp)
        scannerbox = np.append(scannerbox, fp, axis=1)
        self.scboxw = np.array(scannerbox)
        # print('scanner camerabox: ', self.camerabox)
        self.boxw = np.swapaxes(np.array(self.camerabox), 0, 1)
        # print('scanner box: ', self.boxw)


class ImageInfo(object):
    """
    Get Image information, compute the scanner box and some additional parameters
    Do this as a class to encapsulate the data and make reusable.
    """
    def __init__(self, AR):
        BRI = BR.BoundRect()
        self.AR = AR  # save the acq4read instance for access to the data
        # self.AR.getImage()
        pos = self.AR.Image_pos
        scale = self.AR.Image_scale
        region = self.AR.Image_region
        self.binning = self.AR.Image_binning
        binning = self.binning
        # print('Image pos, scale, region: ', pos, scale, region)
        # print('Image binning: ', self.binning)
        scale = list(scale)
        self.scale = scale
        self.filename = self.AR.Image_filename

        x0 = pos[0] # + scale[0]*region[0]/binning[0]
        x1 = pos[0] + scale[0]*(region[2])/binning[0]
        y0 = pos[1] #+ scale[1]*region[1]/binning[1]
        y1 = pos[1] + scale[1]*(region[3])/binning[1]
        self.camerabox = [[x0, y0], [x0, y1], [x1, y1], [x1, y0], [x0, y0]]
        self.boxw = np.swapaxes(np.array(self.camerabox), 0, 1)
        # self.boxw = np.array(self.camerabox)

class EventReader(object):
    def __init__(self, basepath, filename, mapname):

        fne = Path(filename.replace('/', '~')+'.pkl') # '2019.11.08_000~slice_002~cell_000.pkl'
        fe = Path(basepath, 'events', fne)
        with open(fe, 'rb') as fh:
            d = pd.read_pickle(fh, compression=None)
        # print(d.keys())
        dx = d[Path(str(fe.stem).replace('~', '/').replace('../', ''), mapname)]
        self.data = dx
        

class MosaicReader(object):
    """
    Read a mosaic editor mosaic file
    """
    def __init__(self, filename, basepath):
        self.basepath = basepath
        self._saveVersion = (2, 0)  # default.
        state = json.load(open(filename, 'r'))
        if state.get('contents', None) != 'MosaicEditor_save':
            raise TypeError("This does not appear to be MosaicEditor save data.")
        if state['version'][0] > self._saveVersion[0]:
            raise TypeError("Save data has version %d.%d, but this MosaicEditor only supports up to version %d.x." % (state['version'][0], state['version'][1], self._saveVersion[0]))

        root = state['rootPath']

        loadfail = []
        for itemState in state['items']:
            fname = itemState.get('filename')
            fname = Path(root, Path(fname).name)
            # print('fname: ', fname.is_file())
            # print(root)
            # print('itemstate: ', itemState)
            if fname is None:
                # create item from scratch and restore state
                itemtype = itemState.get('type')
                if itemtype not in items.itemTypes():
                    # warn the user later on that we could not load this item
                    loadfail.append((itemState.get('name'), 'Unknown item type "%s"' % itemtype))
                    continue
                item = self.addItem(type=itemtype, name=itemState['name'])
            else:
                # create item by loading file and restore state
                # if root is None:
                #     fh = DataManager.getHandle(fh)
                # else:
                if str(fname.name).startswith('image_'):
                    image_data = self.AR.getImage(fname)
                elif str(fname.name).startswith('video_'):
                    image_data = EM.MetaArray(file=fname)
                    
                fh = root[fname]
                item = self.addFile(fh, name=itemState['name'], inheritTransform=False)
            item.restoreState(itemState)

        self.canvas.view.setState(state['view'])
        if len(loadfail) > 0:
            msg = "\n".join(["%s: %s" % m for m in loadfail])
            raise Exception("Failed to load some items:\n%s" % msg)

    def addItem(self, item=None, type=None, **kwds):
        """Add an item to the MosaicEditor canvas.

        May provide either *item* which is a CanvasItem or QGraphicsItem instance, or
        *type* which is a string specifying the type of item to create and add.
        """
        if isinstance(item, Qt.QGraphicsItem):
            print('was qt')
            return self.canvas.addGraphicsItem(item, **kwds)
        else:
            print('not qt')
            return self.canvas.addItem(item, type, **kwds)

    
class MapTraces(object):
    def __init__(self):
        self.cell = None
        self.datasets = OrderedDict()
        self.image = None
        self.AR = ARC.Acq4Read()
        self.AR.setImportant(False) # turn off the default flag
        self.outputfn = None
        self.invert = True
        self.vmax = 20000.
        self.voff = 0.
        self.ioff = 0.050
        self.basezero = True
        self.ax = None
        self.ax2 = None
        self.xscale = 1.0
        self.yscale = 1.0
        self.nspots = 0
        self.ticks = None
        self.overlay = True
        self.indicesplotted = []
        self.twin = [0, 0.6]
        self.averageScannerImages = False # normally, would not do
        self.SI_minmax = [0., 1.]
        self.SI_original_minmax = [0., 1.]
        self.cellpos = None
        self.experiment = None
        self.image = None
        self.mosaic = None
        self.mosaics = []
        self.calbar = [20, 500]  # 20 ms, 500 pA
        self.offset = [0., 0.]  # offset between pos and image
        self.tbarpos = None # gets set to intersection once created
        self.tbar_coords = None  # line for the tbar
        self.tbar = None # matplotlib line object for tbar
        self.tbar_visible = False
        self.tbar_angle = 0.
        self.scholl_plot = False
        self.ref_angles = None
        self.picker = picker.Picker()
        sns.set()
        sns.color_palette("colorblind", 10)
        self.palette = itertools.cycle(sns.color_palette("colorblind", 10))
        sns.set_style("white")
        sns.set_style("ticks")
        self.window = False
        self.XY = [[None, None]]
        self.XYdepth = 0
        self.calbarobj = None
        self.calbartext = None
        self.mx = 0
        self.my = 0
        self.notch_freqs = None
        self.notch_flag = False
        self.notch_Q = 12.
        
    
    def setScannerImages(self, flag):
        self.averageScannerImages = flag
    
    def setEventData(self, eventdata):
        self.eventdata = eventdata

    def setProtocol(self, cell, image=None, videos=None, mosaic=None):
        self.cell = Path(cell)
        if not self.cell.is_dir():
            print(f"Did not find directory: {str(cell):s}")
            raise ValueError
        if image is not None:
            self.image = Path(self.cell, image)
            print('image path: ', self.image)
            if str(Path(self.image).name).startswith('image_'):
                imagefile = Path(self.image).with_suffix('.tif')  # make sure right suffix is there
                self.image_data = self.AR.getImage(Path(imagefile))
            elif str(Path(self.image).name).startswith('image_'):
                imagefile = Path(self.image).with_suffix('.tif')  # make sure right suffix is there
                self.image_data = self.AR.getImage(Path(imagefile))
            elif str(Path(self.image).name).startswith('video_'):
                imagefile = Path(self.image).with_suffix('.ma')
                print('imagefile: ', imagefile)
                self.image_data = self.AR.getImage(Path(imagefile))
                self.image_data = np.max(self.image_data, axis=0)  # max projection along stack
                self.image_data = np.rot90(np.fliplr(self.image_data))
            else:
                raise ValueError('Do not know how to handle image: ', self.image)
            self.AR.getIndex(currdir=imagefile.parent)
            if 'userTransform' not in list(self.AR._index[imagefile.name].keys()):
                self.refpos = self.AR._index[imagefile.name]['deviceTransform']['pos']
            else:
                self.refpos = self.AR._index[imagefile.name]['userTransform']['pos']  # use repositioned image location

            self.ImgInfo = ImageInfo(self.AR)

        else:
            self.image = None
        
        if mosaic is not None:
            # print('mosaic: ', mosaic)
            self._saveVersion = (2, 0)  # default.
            state = json.load(open(Path(self.cell, mosaic), 'r'))
            if state.get('contents', None) != 'MosaicEditor_save':
                raise TypeError("This does not appear to be MosaicEditor save data.")
            if state['version'][0] > self._saveVersion[0]:
                raise TypeError("Save data has version %d.%d, but this MosaicEditor only supports up to version %d.x." % (state['version'][0], state['version'][1], self._saveVersion[0]))
            self.mosaics = []
            root = state['rootPath']
            # for i in state['items']:
            #     print(i['name'], i['alpha'], i['userTransform'])
            for v in state['items']:  # just copy the items that are relevant
                if v['name'].startswith('video_') or v['name'].startswith('image_'):
                    self.mosaics.append(v)

        self.videos = []
        if videos is not None:
            for v in videos:
                self.videos.append(Path(self.cell, f"video_0{v:02d}"))
        self.AR.setProtocol(self.cell)
        # print('mosaics: ', self.mosaics)
    
    def setWindow(self, x0, x1, y0, y1):
         self.xlim = (x0, x1)
         self.ylim = (y0, y1)
         if not pd.isnull(x0):
             self.window = True
             print('window set!!!!!')
         else:
            self.window = False
    
    def setOutputFile(self, filename):
        self.outputfn = filename
        
    def setPars(self, pdict):
        for k in list(pdict.keys()):
            if k == 'invert':
                self.invert = pdict[k]
            if k == 'vmax':
                self.vmax = pdict[k]
            if k == 'voff':
                self.voff = pdict[k]
            if k == 'ioff':
                self.ioff = pdict[k]
            if k == 'xscale':
                self.xscale = pdict[k]
            if k == 'yscale':
                self.yscale = pdict[k]
            if k == 'calbar':
                self.calbar[0] = pdict[k][0]*1e-3
                self.calbar[1] = pdict[k][1]*1e-12
            if k == 'twin':
                self.twin[0] = pdict[k][0]
                self.twin[1] = pdict[k][1]
            if k == 'ticks':
                self.ticks = pdict[k]
            if k == 'notch_freqs':
                if not pd.isnull(pdict[k]):
                    p = pdict[k].replace('[', '').replace(']', '').replace('"', '')
                    fp = p.split(',')
                    lp = [float(f) for f in fp]
                    self.notch_freqs = np.array(lp)
                    self.notch_flag = True
            if k == 'notch_q':
                self.notch_Q = float(pdict[k])
            if k == 'cellpos':
                self.cellpos = pdict[k]
            if k == 'experiment':
                self.experiment = pdict[k]
            if k == 'angle':
                self.tbar_angle = pdict[k]
            if k == 'cellID':
                self.cellID = pdict[k]
            if k == 'map':
                self.mapname = pdict[k]
            if k == 'offset':
                self.offset[0] = pdict[k][0]
                self.offset[1] = pdict[k][1]

    def filter_data(self, tb, data, LPF=3000.):
        self.HPF_flag = False
        self.LPF = LPF
        self.maxtime = 0.599 # sec
        filtfunc = scipy.signal.filtfilt
        samplefreq = 1./np.mean(np.diff(tb))
        rate = 1.0/samplefreq
        nyquistfreq = samplefreq*0.95
        wn = self.LPF/nyquistfreq
        b, a = scipy.signal.bessel(2, wn)
        if self.HPF_flag:
            wnh = self.HPF/nyquistfreq
            bh, ah = scipy.signal.bessel(2, wnh, btype='highpass')
        imax = int(max(np.where(tb < self.maxtime)[0]))
        print(np.max(tb), imax)
        # imax = len(tb)
        data2 = np.zeros_like(data)

        data2 = filtfunc(b, a, data[:,:imax] - np.mean(data[0:250]))
        # if self.HPF_flag:
        #     data2[r,t,:imax] = filtfunc(bh, ah, data2[r, t, :imax]) #  - np.mean(data[r, t, 0:250]))
        data3 = np.zeros_like(data2)
        if self.notch_flag:
            print('Notch Filtering Enabled', self.notch_freqs)
            for r in range(data2.shape[0]):
                data3[r] = FILT.NotchFilterZP(data2[r], notchf=self.notch_freqs, Q=self.notch_Q,
                    QScale=False, samplefreq=samplefreq)
        else:
            data3 = data2
        # mpl.figure()
        # mpl.plot(tb[:imax], data2[0,:imax])
        # mpl.plot(tb[:imax], data3[0,:imax])
        # mpl.show()
        # exit()
        #
        return data3

    def plot_maps(self, protocols, ax=None, traces=None, linethickness=1.0, tbar=False, axb=None):
        """
        Plot map or superimposed maps...
        """
        print('plot_maps: plot_maps with protocols: ', protocols)
        if ax is None:
            self.figure = mpl.figure()
            # print(dir(self.figure))
            self.figure.set_size_inches(14., 8.)
            if traces is None:
                self.ax = self.figure.add_subplot('111')
                # print('set ax')
            else:
                self.ax = self.figure.add_subplot('121')
                self.ax2 = self.figure.add_subplot('122')
                sns.despine(ax=self.ax2, left=True, bottom=True, right=True, top=True)
                # print('set ax and ax2')
        else:
            self.ax = ax
            print('ax: ', ax)
        # self.ax3 = self.figure.add_subplot('133')
        self.data = dict.fromkeys(list(protocols.keys()))
        cols = ['r', 'b', 'c', 'g']
        self.traces = traces

        for i, p in enumerate(protocols):
            prot = protocols[p]
            self.datasets[p] = []
            if i == 0:
                self.setProtocol(prot, self.image)
                self.show_traces(self.ax, pcolor=cols[i], name=p, linethickness=linethickness)
            else:
                self.setProtocol(prot)
                self.show_traces(self.ax, pcolor=cols[i], name=p, linethickness=linethickness)
        if self.traces is not None:
            for tr in self.traces:
                self.handle_event(index=tr)
            PH.calbar(self.ax2, calbar=[0, -6., 50, 5.], scale=[1.0, 1.0],
                axesoff=True, orient='left', unitNames=None, fontsize=11, weight='normal', color='k', font='Arial')

        else:
            self.picker = picker.Picker()
            self.picker.setData(2, self.scp)
            self.picker.setAction(self.handle_event)

        # self.figure.canvas.mpl_connect('button_press_event', self.picker.pickEvent)
        # self.figure.canvas.mpl_connect('pick_event', self.picker.pickEvent)
        # self.figure.canvas.mpl_connect('motion_notify_event', self.picker.onMouseMotion)
        x1, x2 = self.ax.get_xlim()
        y1, y2 = self.ax.get_ylim()
        self.XY = [self.get_XYlims()]
        cp = self.cell.parts
        cellname = '/'.join(cp[-4:])
        self.update_tbar(None)
        self.compute_sector_distance_map()
        # self.plot_scholl()
        if axb is not None:
            self.plot_sector_distance_map(axb)
        if self.ax is None:
            self.figure.suptitle(cellname, fontsize=11)
            self.fig2 = None
            if self.outputfn is not None:
                mpl.savefig(self.outputfn)
        # mpl.show()

    def get_XYlims(self):
        x1, x2 = self.ax.get_xlim()
        y1, y2 = self.ax.get_ylim()
        return([x1, y1, x2, y2])

    def adjust(self, extent, a):
        # adjust the extent so that the minimum boundary to the edge of the plot is a, 
        # and so that the area plotted is "common" across datasets
        minx = self.SI.centroid_sh.x - a/2.
        maxx = self.SI.centroid_sh.x + a/2.
        miny = self.SI.centroid_sh.y - a/2.
        maxy = self.SI.centroid_sh.y + a/2.
        return([minx, maxx, miny, maxy])
 
    def set_minmax(self, imagedata):
        """
        return the min, max for the data
        """
        mind = SND.minimum(imagedata)
        maxd = SND.maximum(imagedata)
        if maxd == mind:
            maxd = mind + 1
        return [mind, maxd]

    def show_traces(self, ax, pcolor='r', linethickness=0.5, name=None):

        self.cell.glob('*')
        # imageplotted = False
        # imagetimes = []
        # imagename = []
        # maptimes = []
        # mapname = []
        supindex = self.AR.readDirIndex(currdir=self.cell)
        
        self.SI = ScannerInfo(self.AR, self.offset)
        
        self.extent = np.array([-1, 1, -1, 1])*7e-2
        if self.invert:
            cmap = 'gist_gray_r'
        else:
            cmap = 'gist_gray'
        mapwidth = 1e-3
        self.imageax = None
        self.SI_ax = None   # scanner image for rescale
        self.max_camera = None
        sc_alpha = 1.0
        vid_alpha = 0.75
        image_alpha = 0.75
        mosaic_alpha = 0.75
        box = None
        if self.averageScannerImages:
            sc_alpha = 1.0
            vid_alpha = 0.5
            image_alpha = 0.5
            mosaic_alpha = 0.5
            
        if self.averageScannerImages:
            self.max_camera = self.AR.getAverageScannerImages(dataname='Camera/frames.ma', mode='max', 
                    subtractFlag = True, firstonly=False, limit=None)
            sm = self.max_camera
            sm = sm/np.max(sm)
            sm = sm*sm
            sm = np.asarray(sm, dtype=float)
            # sm = np.clip(sm, a_min=0.5, a_max=None)
            self.extent0 = [np.min(self.SI.boxw[0]), np.max(self.SI.boxw[0]), np.min(self.SI.boxw[1]), np.max(self.SI.boxw[1])]
            self.extent = self.adjust(self.extent0, mapwidth)
            self.SI_ax = ax.imshow(sm, aspect='equal', cmap='Blues', alpha=sc_alpha, vmin = 0, vmax=np.max(sm),
                extent=self.extent0)
        # max_camera = scipy.ndimage.gaussian_filter(max_camera, sigma=256/(4.*10))
            self.set_minmax(self.max_camera)
            self.SI_minmax = list(self.SI_ax.get_clim())# use the clim for the min/max
            self.SI_original_minmax = self.SI_ax.get_clim()  # keep original
            box = self.SI
        
        if len(self.videos) > 0:
            self.Montager = montage.Montager(celldir=self.cell)
            self.Montager.run()
            # M.list_images_and_videos()
            self.Montager.process_videos(window='mpl', show=True, gamma=1.5, merge_gamma=-1., sigma=2.5)
            # bounds are in  self.Montager.bounds: (minx, miny, maxx, maxy)
            bounds = self.Montager.bounds
            self.extent0 = [bounds[0], bounds[2], bounds[1], bounds[3]]
            self.extent = self.adjust(self.extent0, mapwidth)
            self.imageax = ax.imshow(np.asarray(self.merged_image, dtype=float), aspect='equal', cmap=cmap, alpha=vid_alpha, vmin = 0, vmax=self.vmax,
                extent=self.extent0)
            self.cmin = SND.minimum(self.merged_image)
            self.cmax = SND.maximum(self.merged_image)
            box = self.extent
            
        if self.image is not None:
            # mpl.imshow(self.image_data)
            # mpl.show()
            self.extent0 = [np.min(self.ImgInfo.boxw[0]), np.max(self.ImgInfo.boxw[0]), np.min(self.ImgInfo.boxw[1]), np.max(self.ImgInfo.boxw[1])]
            self.extent = self.adjust(self.extent0, mapwidth)
            self.imageax = ax.imshow(np.asarray(self.image_data, dtype=float), aspect='equal', cmap=cmap, alpha=image_alpha, vmin = 0, vmax=self.vmax,
                extent=self.extent0)
            self.cmin = SND.minimum(self.image_data)
            self.cmax = SND.maximum(self.image_data)
            box = self.extent
            
            
        if self.mosaics:
            self.Montager = montage.montager.Montager(celldir=self.cell)
            self.Montager.setup(self.mosaics)
            # M.list_images_and_videos()
            # should pass some info to process_videos to balance alpha etc from the mosaic.
            self.Montager.process_videos(window=None, show=True, gamma=1.5, merge_gamma=-1., sigma=2.5, register=False, mosaic_data=self.mosaics)
            # bounds are in  self.Montager.bounds: (minx, miny, maxx, maxy)
            bounds = self.Montager.image_boundary
            self.extent0 = [bounds[0], bounds[2], bounds[1], bounds[3]]
            self.extent = self.adjust(self.extent0, mapwidth)
            mpl.imshow(self.Montager.merged_image)
            self.imageax = ax.imshow(np.array(self.Montager.merged_image, dtype=float), aspect='equal', cmap=cmap, alpha=mosaic_alpha, vmin = 0, vmax=self.vmax,
                extent=self.extent0)
            self.cmin = SND.minimum(self.Montager.merged_image)
            self.cmax = SND.maximum(self.Montager.merged_image)
            box = self.extent

        if self.window:
            if self.xlim == (0., 0.) and self.ylim == (0., 0.):
                xylims = self.extent
                ax.set_xlim(xylims[0:2])
                ax.set_ylim(xylims[2:4])
                # print('autoset: ', xylims)
            else:    
                ax.set_xlim(self.xlim)
                ax.set_ylim(self.ylim)
                # print('self set: ', self.xlim, self.ylim)
            
        self.scp = self.SI.scannerpositions
        scp = self.scp
        xmin = np.min(scp[:,0])
        xmax = np.max(scp[:,0])
        ymin = np.min(scp[:,1])
        ymax = np.max(scp[:,1])
        # print(xmin, ymin, xmax, ymax)
        ax.scatter(scp[:,0], scp[:,1], s=4, c='c', marker='o', alpha=0.3, picker=5)
        # print('getdata: ', name, self.datasets)
        self.AR.getData()
        d = self.AR.data_array
        d = self.filter_data(self.AR.time_base, d)
        if name is not None:
            self.datasets[name] = d

        if self.AR.mode in ['v', 'V', 'VC']:
            self.vscale = 1e5
            self.off = self.voff
        else:
            self.vscale = 1e-3
            self.off = self.ioff
        # print(len(self.AR.traces))
        tb = self.AR.time_base
        im0 = np.argmin(np.fabs(tb - self.twin[0]))
        im1 = np.argmin(np.fabs(tb - self.twin[1]))
        self.im0 = im0
        self.im1 = im1
        self.tb = tb[im0:im1]-tb[im0]
        # just plot as many as we have!
        dshape = self.datasets[name].shape
        for p in range(dshape[0]): # scp.shape[0]):
            self._plot_one(ax, p, pcolor, name=name, ythick=linethickness)
        self.plot_calbar(ax, xmin, ymin)
        
        # plot length (physical dimension) bar
        if box is not None:
            PH.calbar(ax, calbar=[box[0]+0.00012,
                                     box[3]-0.00005,
                                     0.0001 , 0], 
                scale = [1e6, 1e6], axesoff=True, orient='left',
                unitNames={'x': r'$\mu$m', 'y': ''}, fontsize=11, weight='normal', color='k', font='Arial')
        ax.set_xlim(self.extent[0:2])
        ax.set_ylim(self.extent[2:4])
        if self.cellpos is not None:
            mpl.gca().plot(self.cellpos[0], self.cellpos[1], color='blue', marker='P', markersize=6)
        if self.experiment is not None:
            mpl.gca().set_title(f"{self.experiment:s}", fontsize=9)
            
        # print(dir(self.imageax))
        
    def plot_calbar(self, ax, x0, y0):
        x0 += 0.5e-4
        y0 += 0.5e-4
        xcal = self.xscale*3.5e-5*self.calbar[0]*1.25
        ycal = self.yscale*self.vscale*self.calbar[1]*0.5
        zero = 0

        self.calbarobj = ax.plot(self.xscale*3.5e-5*np.array([0., 0., self.calbar[0]])+x0 - xcal,
               (self.yscale*self.vscale*(np.array([self.calbar[1],  0., 0. ])-zero))+self.off*self.vscale+y0 - ycal, 'k-', linewidth=1)
        self.calbartext = ax.text(x0-xcal, y0-ycal, f"{int(self.calbar[0]*1e3):d} ms\n{int(self.calbar[1]*1e12):d} pA",
        verticalalignment='top', horizontalalignment='center', fontsize=8)
        self.calx_zero = self.calbarobj[0].get_xdata()
        self.caly_zero = self.calbarobj[0].get_ydata()
        # self.reposition_cal()

    def reposition_cal(self, movex=0, movey=0, home=False):
        if not home:
            calxdata = self.calbarobj[0].get_xdata()
            calydata = self.calbarobj[0].get_ydata()
        else:
            calxdata = self.calx_zero
            calydata = self.caly_zero
            self.mx = 0
            self.my = 0
        # print('xdata: ', calxdata)
        # print('ydata: ', calydata)
        xd = calxdata[2] - calxdata[1]
        yd = calydata[0] - calydata[1]
        xl = sorted(self.ax.get_xlim())
        yl = sorted(self.ax.get_ylim())
        # print(xl, yl)
        x0 = xl[0] + (movex+self.mx)*0.001*xl[1]
        y0 = yl[0] + (movey+self.my)*0.001*yl[1]
        self.calbarobj[0].set_xdata([x0, x0, x0+xd])
        self.calbarobj[0].set_ydata([y0+yd, y0, y0])
        # print([x0, x0, x0+xd])
       #  print([y0+yd, y0, y0])
        self.mx += movex
        self.my += movey
        
        # print(dir(MT.calbartext))
        # calxy = MT.calbartext.get_position()
        calxy = [0, 0]
        calxy[0] = x0 + xl[1]*(movex+self.mx)*0.001
        calxy[1] = y0 + yl[1]*(movey+self.my)*0.001 - yl[1]*0.015
        self.calbartext.set_position(calxy)
        # print('reposition : ', movex, movey)
        # print(calxy, self.calx_zero, self.caly_zero)
        
    def _plot_one(self, ax, p, pcolor, name=None, yscaleflag=True, tscale=True, offflag=True, ystep = 0., ythick=0.3):
        zero = 0.
        vdat = FILT.SignalFilter_LPFBessel(self.datasets[name][p, :], 2000.,
                        samplefreq=self.AR.sample_rate[0], NPole=8)
        if self.basezero:
            zero = np.mean(vdat[self.im0:self.im0+20])
        if offflag:
            xoff = self.scp[p,0]
            yoff = self.off*self.vscale+self.scp[p, 1]
        else:
            xoff = 0.
            yoff = 0.
        if yscaleflag:
            y_scale = self.yscale*self.vscale
        else:
            y_scale = 1e3
        yoff += ystep
        if tscale:
            ts = self.xscale*3.5e-5
        else:
            ts = 1e3
        ax.plot(ts*self.tb+xoff, (y_scale*(vdat[self.im0:self.im1]-zero))+yoff, color=pcolor, linewidth=ythick)
        if self.ticks is not None:
            for t in self.ticks:
                ax.plot(ts*np.array([t, t])+xoff,  y_scale*np.array([-20e-12, 20e-12])+yoff, color='k',  linewidth=0.8)

    def handle_event(self, index):
        # print('handle event index: ', index)
        # print(self.SI.scannerpositions[index,:])
        if self.ax2 is None:
            return
        if index in self.indicesplotted:
            return
        if self.overlay:
            ystep = -self.nspots*10.
        self.palette = itertools.cycle(sns.color_palette("colorblind", 10))
        for i, name in enumerate(list(self.datasets.keys())):
            c = next(self.palette)
            self._plot_one(self.ax2, index, pcolor=c, name=name, yscaleflag=False, offflag=False,
                ystep=ystep, ythick=1.3-(i+1)*0.3, tscale=False)
            # if i == 1:
            #     self._plot_one(self.ax3, index, pcolor=c, name=name, yscaleflag=False, offflag=False,
            #         ystep=ystep, ythick=0.5, tscale=False)
        # self.xscale*3.5e-5*self.tb+xoff, (y_scale*(vdat[self.im0:self.im1]-zero))+yoff, color=pcolor, linewidth=0.3)
        trn = self.traces.index(index)+1
        self.ax.text(self.SI.scannerpositions[index][0], self.SI.scannerpositions[index][1],
                f"{trn:d}", fontsize=9, horizontalalignment='center')
        self.ax2.text(0., ystep,
                f"{trn:d}", fontsize=9, horizontalalignment='right')
        self.nspots += 1
        self.indicesplotted.append(index)
        mpl.draw()
    
    def toggle_tbar(self, event):
        if self.tbar_visible and self.tbar_coords is not None:
            self.tbar[0].set_alpha(0)
            self.tbar_visible = False
            return
        else:
            self.update_tbar(event)
            
    def _getAngle(self, pt1, pt2):
        """
        Pt1 and 2 must be shapely Point objects"""
        x_diff = pt2.x - pt1.x
        y_diff = pt2.y - pt1.y

        return np.arctan2(y_diff, x_diff)
    
    def update_tbar(self, event):
        center = SH.geometry.Point(self.cellpos[0], self.cellpos[1])  # center (flip y axis)
        # first time through, just draw the bar
        if self.tbar_coords is None:
            cx = self.cellpos[0]  # center (cell pos)
            cy = self.cellpos[1]

            tlinex = [cx,      cx, cx,      cx-1e-4, cx+1e-4]  # draw the T bar
            tliney = [cy-1e-4, cy, cy+1e-4, cy+1e-4, cy+1e-4]
            self.tbar_coords = SH.geometry.LineString([(tlinex[i], tliney[i]) for i in range(len(tlinex))])
            self.tbar = self.ax.plot(self.tbar_coords.xy[0], self.tbar_coords.xy[1], 'ko-', linewidth=1, markersize=2.5)
        elif event is not None and event.xdata is not None:
            e = SH.geometry.Point([event.xdata, event.ydata])  # point in direction for top of T bar
            t = SH.geometry.Point([self.tbar_coords.xy[0][2], self.tbar_coords.xy[1][2]])
            angle1 = self._getAngle(center, e)
            angle2 = self._getAngle(center, t)
            self.tbar_angle = -(angle2-angle1)
            print('angle: ', self.tbar_angle)
        else:
            pass
        newT = SH.affinity.rotate(self.tbar_coords, self.tbar_angle, origin=center, use_radians=True)
        self.tbar[0].set_xdata(newT.xy[0])
        self.tbar[0].set_ydata(newT.xy[1])
        self.tbar[0].set_alpha(1)
        self.tbar_visible = True
        self.compute_sector_distance_map()
        self.plot_scholl()

    def _plot_coords(self, ax, ob, c='#999999', **kwds):
        """
        Plot the data in a shapely object
        
        Parameters
        ----------
        ax : matplotlib axis object
            target axis to plot data into
        
        ob : shapely object with a .xy list of values
        
        c : matplotlib color value
            RGBA, string, etc
        
        Returns
        -------
        Nothing
        """
        
        x, y = ob.xy
        ax.plot(x, y, '-', color=c, **kwds)
        
    def compute_sector_distance_map(self):
        """
        Compute the responses divided by distance (scholl rings) and sector angle
        """
        measure = 'ZScore'
        thresh = 1.96
        r = np.array([0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45])*1e-3 # convert to display units of meters from mm
        nrings = r.shape[0]
        t = self.cellpos
        # first generate the rings and plot them
        if not self.scholl_plot:
            self.C_rings = [[] for _ in range(len(r))] # concentric rings
            for i, rad in enumerate(r):
                self.C_rings[i] = SG.Polygon([(rad*np.sin(theta) + t[0], rad*np.cos(theta) + t[1]) for theta in np.linspace(0., 2*np.pi, 100)])

        # get the angles for each point in the map
        x = self.SI.scannerpositions.T # self.eventdata['positions'].T  # just for the way we handle the values here.
        nsectors = 4
        rot_angle = np.pi/nsectors
        point_angles = np.arctan2(x[1,:]-t[1], x[0,:]-t[0])-self.tbar_angle  # get angles relative to cell position, then rotate by tbar angle
        try:
            point_angles = np.where(point_angles < -rot_angle, point_angles+2*np.pi, point_angles)
        except:
            print('point_angles: ', point_angles)
            print('rot_angle: ', rot_angle)
            raise ValueError()
        
        # compute the angles that divide the sectors
        # and assign a sector index to each point that falls into
        # the sector (list per sector are held in angle_group)
        sector_angles = []  # sector angles for every sector
        self.angle_group = [[] for _ in range(nsectors)]
        for quad in range(nsectors):
            quad0 = quad*np.pi/(nsectors/2.)
            qa = quad0 - rot_angle
            qb = quad0 + rot_angle
            (qa, qb) = sorted([qa, qb])
            sector_angles.append(qa)

            # find all the points in this sector
            pgr = np.where((point_angles >= qa) & (point_angles < qb))[0]
            self.angle_group[quad] = pgr
        # for q in range(nsectors):
        #     print(f'angle group[{q:d}]: ', angle_group[q])

        # assign points by rings as well
        Px2 = SG.MultiPoint([SG.Point(x[0,i], x[1,i]) for i in range(x.shape[1])])
        ptlocs = np.zeros((x.shape[1], len(r)+1))  # for each point, all rings that it is in
        for i, p in enumerate(Px2):
            for j in range(len(self.C_rings)):
                ptlocs[i,j] = self.C_rings[j].contains(p)

        ring_index = [None]*ptlocs.shape[0]  # hold the ring index for each point
        edgecolor = ['r', 'm', 'b', 'c', 'y', 'g', 'turquoise', 'brown', 'lightblue']
        sectorsymbol = ['o', 's', '^', 'D']
        for i in range(ptlocs.shape[0]):
            u = np.where(ptlocs[i,:] == 1)[0]  # get the index of the first ring where the point is found
            if len(u) > 0:  # some may be outside the outermost ring
                ring_index[i] = u[0]  # get the index for this point (which ring)

        zs = [[] for _ in range(nsectors)]  # ring index by sectors,but value stored is total zscore....
        for i in range(len(ring_index)):
            if ring_index[i] is not None: # outside outermost ring; ignore
                for j in range(nsectors):  # find sector for this point
                    if i in self.angle_group[j]:
                        zs[j].append(i)
            # angles.append(qb)
        # for j in range(nsectors):
        #     try:
        #         print('?: ', self.eventdata['I_max'][0][zs[j]])
        #     except:
        #         print(j, zs[j])
        #         self.eventdata['I_max']
        for j in range(nsectors):
            print('sector: ', j, '  total score: ', np.sum(self.eventdata['ZScore'][0][zs[j]]))
            print('sector: ', j, '  total charge: ', np.sum(self.eventdata['Qr'][0][zs[j]]))
            if len(self.eventdata['I_max'][0][zs[j]]) > 0:
                print('sector: ', j, '  mean I_max: ', np.mean(self.eventdata['I_max'][0][zs[j]]))
            else:
                print('sector: ', j, '  no event data')
        secdistmap = np.zeros((nrings, nsectors))
        for i in range(nrings):
            ri = np.where(np.array(ring_index) == i)[0]  # get the points in this ring
            for j in range(nsectors):
                siri = list(set(self.angle_group[j]).intersection(ri))  # get those points in this sector
                if len(siri) > 0:
                    # secdistmap[i, j] = np.mean(self.eventdata['ZScore'][0][siri])
                    secdistmap[i, j] = np.sum(self.eventdata[measure][0][siri] > thresh)

        center_of_mass = (0.,0.)  # relative to cell body.
        for i in range(0, 2):
            center_of_mass[i] = np.sum((self.eventdata[measure][0][:] > thresh)*(x[i,:]))

        sector_center_of_mass = np.zeros((nsectors, 2))
        self.mass = np.zeros(nsectors)
        for i in range(nsectors):
            self.mass[i] = np.sum(self.eventdata[measure][0][self.angle_group[i]] > thresh)
            for j in range(2):
                sector_center_of_mass[i][j] = np.sum((self.eventdata[measure][0][self.angle_group[i]] > thresh)*(x[j,self.angle_group[i]]))
                try:
                    sector_center_of_mass[i][j] /= np.sum(self.eventdata[measure][0][self.angle_group[i]] > thresh)
                except:
                    sector_center_of_mass[i][j] = np.nan # possible for no values > threshold
        print('COM: ', sector_center_of_mass)
        self.sector_center_of_mass = sector_center_of_mass
        print('Sector distance map: ')
        for j in range(nsectors):
            print('   ', secdistmap[:,j])
        self.secdistmap = secdistmap
        self.ring_index = ring_index
        self.nsectors = nsectors
        self.edgecolor = edgecolor
        self.sectorsymbol = sectorsymbol
        self.radii = r
        
        rad = np.max(r)
        # this plots lines and/or rotates them 
        self.radlines = [((t[0], rad*np.sin(theta-self.tbar_angle)+t[0]), 
                          (t[1], rad*np.cos(theta-self.tbar_angle)+t[1])) for theta in sector_angles]
        cl = ['r', 'r', 'b', 'b', 'y', 'y']
        # print('L: ', self.radlines)
        L = self.radlines
        if self.ref_angles is None:
            self.ref_angles = [None]*len(L)
            self.polydata = SG.Polygon([L[0][0], 
                                   L[0][1], L[1][1],
                                   L[0][0]])
            self.polydata_ob = None
        else:
            if self.polydata is not None:
                try:
                    self.polypatch_ob.remove()
                except:
                    pass
                self.polydata = SG.Polygon([L[0][0], L[0][1], L[1][1], L[0][0]])  # update data
        # print('Polypatch generation')
        # print('exterior: ', self.polydata.exterior)
        # polypatch = descartes.PolygonPatch(self.polydata, facecolor='r', edgecolor='k', alpha=0.95, zorder=100000) # on top
        # self.polypatch_ob = self.ax.add_patch(polypatch)
        # self.polypatch_ob.set_alpha(0.5)
        # self.polypatch_ob.set_color(cl[0])

    def get_sector_center_of_mass(self):
        """
        Compute the center of mass for the input locations
        Requires compute_sector_distance_map already run
        """
        

    def plot_scholl(self):
        for i in range(len(self.C_rings)):
                self._plot_coords(self.ax, self.C_rings[i].exterior, c="#888888")
        for i in range(len(self.radlines)):
            if self.ref_angles[i] is None:
                self.ref_angles[i] = self.ax.plot(self.radlines[i][0], self.radlines[i][1], c='gray')
            else:
                self.ref_angles[i][0].set_xdata(self.radlines[i][0])
                self.ref_angles[i][0].set_ydata(self.radlines[i][1])
        self._plot_coords(ax=self.ax, ob=self.polydata.exterior, c='#000000', linewidth=2, alpha=1.0)

        x = self.SI.scannerpositions.T # just for the way we handle the values here.
        for i in range(len(self.ring_index)):
            if self.ring_index[i] is None: # outside outermost ring; ignore
                self.ax.plot(x[0,i], x[1,i], 'ro', markersize=4)
            else:  # in side rings
                for j in range(self.nsectors):  # find sector for this point
                    if i in self.angle_group[j]:
                        # self.ax.plot(x[0,i], x[1,i], c=edgecolor[ring_index[i]], marker=sectorsymbol[j], markersize=4)
                        self.ax.plot(x[0,i], x[1,i], c=self.edgecolor[j], marker=self.sectorsymbol[j], markersize=1)  # color by sector
        maxmass = np.max(self.mass)
        for i in range(self.nsectors):
            self.ax.plot(self.sector_center_of_mass[i,0], self.sector_center_of_mass[i,1], c=self.edgecolor[i], 
                marker= self.sectorsymbol[i], markersize=8*self.mass[i]/maxmass, alpha=0.75)
        self.scholl_plot = True
        
    def plot_sector_distance_map(self, axin=None):
        """
        Plot a map of the measure, by distances and sorted by sector
        """
        if axin is None:
            fig, ax = mpl.subplots(self.nsectors)
            # ax = mpl.gca()
        else:
            ax = [axin]*self.nsectors
        nmax = int(np.max(self.secdistmap))
        PH.nice_plot(ax, position={'left': -0.02, 'bottom': -0.1})
        for s in range(self.nsectors):
            ax[s].plot(self.radii*1e6, self.secdistmap[:,s], self.edgecolor[s])
            ax[s].set_ylim(0, int(nmax))
            ax[s].set_xlim(0, 1e6*np.max(self.radii))
            if s != self.nsectors-1:
                PH.noaxes(ax[s], whichaxes='x')
            PH.talbotTicks(ax[s], axes='xy', density=(1.0, 1.0), pointSize=10,
                tickPlacesAdd={'x': 0, 'y': 0}, floatAdd={'x': 0, 'y': 0})
            ax[s].set_ylabel(r'Responsive sites', fontsize=8)
        ax[self.nsectors-1].set_xlabel(r'Distance ($\mu$m)')
        if axin is None:
            mpl.show()
    

        
        
def main():

    import nf107.set_expt_paths as set_expt_paths
    set_expt_paths.get_computer()
    experiments = set_expt_paths.get_experiments()
    exclusions = set_expt_paths.get_exclusions()

    cellchoices = [     'bushy',
                        'tstellate', 
                        'dstellate',
                        'tuberculoventral', 
                        'pyramidal', 
                        'giant', 
                        'cartwheel',
                        'unknown', 'all']
    parser = argparse.ArgumentParser(description='Plot maps with traces on top')
    parser.add_argument('-E', '--experiment', type=str, dest='experiment',
                        choices = list(set_expt_paths.experiments.keys()), default='None', nargs='?', const='None',
                        help='Select Experiment to analyze')
    parser.add_argument('-c', '--celltype', type=str, default=None, dest='celltype',
                        choices=cellchoices,
                        help='Set celltype for figure')
    parser.add_argument('-s', '--sequence', type=str, default='1', dest='sequence',
                        help='sequence of ID numbers of the cells to plot')

    parser.add_argument('-S', '--scanner', action='store_true', dest='scannerimages',
                        help='Plot the scanner spots on the map')

    parser.add_argument('-t', '--tbar', action='store_true', dest='plotwithtbar',
                        help='Plot the traces and scanner spots with the tbar')
                        
    args = parser.parse_args()
    experimentname = args.experiment 
    basepath = Path(experiments[experimentname]['disk'])
    # basepath = '/Volumes/Pegasus/ManisLab_Data3/Kasten_Michael/NF107ai32Het/'

    MT = MapTraces()
    MT.setScannerImages(args.scannerimages)
    
    if args.celltype == 'lsps':
        cell = Path('/Users/pbmanis/Desktop/Data/Glutamate_LSPS_DCN/2019.08.05_000/slice_002/cell_000/LSPS_dendrite_VC_testmap_MAX_001')  # pyr
        image = '../image_001.tif'
        MT.setPars({'invert': True, 'vmax': 30000, 'xscale': 1.5, 'yscale': 1.5, 'calbar': [0.5, 200.e-12]})  # calbar in ms, pA
        prots = [cell]

        MT.setPars({'invert': False, 'vmax': 30000, 'xscale': 1.5, 'yscale': 0.05, 'calbar': [0.5, 5000.e-12]})  # calbar in ms, pA
        MT.setPars({'invert': True, 'vmin': 1000, 'vmax': 18000, 'xscale': 6, 'yscale': 1.5, 'calbar': [0.5, 20.e-3], 'twin': [0.25, 0.5],
                 'ioff': -0.0})  # calbar in ms, pA
        cell1 = Path('/Users/pbmanis/Desktop/Data/Glutamate_LSPS_DCN/2019.08.08_000/slice_001/cell_000/LSPS_dendrite_CC_testmap_MAX_000')
        cell2 = Path('/Users/pbmanis/Desktop/Data/Glutamate_LSPS_DCN/2019.08.08_000/slice_001/cell_000/LSPS_dendrite_CC_testmap_MAX_001')
        cell3 = Path('/Users/pbmanis/Desktop/Data/Glutamate_LSPS_DCN/2019.08.08_000/slice_001/cell_000/LSPS_dendrite_CC_testmap_MAX_002')
        cell_vc = Path('/Users/pbmanis/Desktop/Data/Glutamate_LSPS_DCN/2019.08.08_000/slice_001/cell_000/LSPS_dendrite_VC_testmap_MAX_000')
        MT.setPars({'invert': True, 'vmin': 1000, 'vmax': 18000, 'xscale': 6, 'yscale': 1.5, 'calbar': [0.5, 200e-12], 'twin': [0.25, 0.5],
                 })
        image = '../image_002.tif'
        prots = [cell1, cell2, cell3]
        prots = [cell_vc]

    # cell = Path('/Users/pbmanis/Desktop/Data/Glutamate_LSPS_DCN/2019.08.05_000/slice_002/cell_000/LSPS_dendrite_VC_testmap_MAX_001')  # pyr
 #    image = '../image_001.tif'
 #    MT.setPars({'invert': True, 'vmax': 30000, 'xscale': 1.5, 'yscale': 1.5, 'calbar': [0.5, 200.e-12]})  # calbar in ms, pA
 #    prots = [cell]

    """
    Cells for paper
 
    """
    if args.celltype == 'all':
        docell = args.celltype
    else:
        docell = [args.celltype]
    
    if docell in ['unknown', 'all']:
        return
    sequence, target = PU.recparse(args.sequence)
    if sequence is None:  # invoked help
        return
    shiftkey = False
    
    table = pd.read_excel('NF107Ai32_Het/Example Maps/SelectedMapsTable.xlsx')

    def makepars(dc):
        parnames = ['offset', 'cellpos', 'angle', 'experiment', 'invert', 'vmin', 'vmax', 'xscale', 'yscale', 'calbar', 'twin', 'ioff', 'ticks', 'notch_freqs', 'notch_q', 'cellID', 'map']
        pars = dict()
        for n in parnames:
            if n in ['calbar', 'twin', 'offset', 'cellpos']:
                try:
                    pars[n] = eval('['+dc[n].values[0]+']')
                except:
                    print('n: ', n, dc[n].values[0])
                    raise ValueError()
            elif n in ['ticks']:
                pars[n] = [dc[n].values[0]]
            elif n in ['angle']: # single float value
                pars[n] = float(dc[n].values[0])
            else:
                pars[n] = dc[n].values[0]
        return pars


    def line_select_callback(eclick, erelease):
        'eclick and erelease are the press and release events'
        
        
        if eclick.button == MBB.MouseButton.LEFT and erelease.button== MBB.MouseButton.LEFT:
            x1, y1 = eclick.xdata, eclick.ydata
            x2, y2 = erelease.xdata, erelease.ydata
            print(f"Corners: {x1:.6f}, {x2:.6f}) --> {y1:.6f}, {y2:.6f})")
            print(f"Copy:    {x1:.6f}\t{y1:.6f}\t{x2:.6f}\t{y2:.6f}")
            # print(" The buttons you used were: %s %s" % (eclick.button, erelease.button))
            MT.XY.append([x1, y1, x2, y2])
        elif eclick.button == MBB.MouseButton.RIGHT:
            if len(MT.XY) == 0:
                return
            print('MT.xy: ', MT.XY)
            x1, y1, x2, y2 = MT.XY.pop()
        xl = sorted([x1, x2])
        yl = sorted([y1, y2])
        MT.ax.set_xlim(xl)
        MT.ax.set_ylim(yl)
        MT.reposition_cal()
        mpl.draw()

    def release_selector(event):
        if event.key in ['shift']:
            pass

    def toggle_selector(event):
        # print(event.key, event.key in ['\x1b[A', '\x1b[B','\x1b[C','\x1b[C',])
        if event.key in ['shift']:
            pass
        
        elif event.key in ['q'] and toggle_selector.RS.active:
            print(' RectangleSelector deactivated.')
            toggle_selector.RS.set_active(False)
       
        elif event.key in ['a'] and not toggle_selector.RS.active:
            print(' RectangleSelector activated.')
            toggle_selector.RS.set_active(True)
       
        elif event.key in ['p']:
            xylims = MT.get_XYlims()
            print(f"{xylims[0]:.5f}\t{xylims[2]:.5f}\t{xylims[1]:.5f}\t{xylims[3]:.5f}")
            if MT.calbarobj is not None:
                print(MT.calbarobj[0].get_xydata())
       
        elif event.key in ['s']:
            xylims = MT.get_XYlims()
            print(f"Position: {xylims[0]:.5f}\t{xylims[2]:.5f}\t{xylims[1]:.5f}\t{xylims[3]:.5f}")
            mpl.savefig(MT.outputfn)
            exit()
        
        elif event.key in ['z']:
            MT.cmin = SND.minimum(MT.image_data)
            MT.cmax = SND.maximum(MT.image_data)
            MT.imageax.set_clim(MT.cmin, MT.cmax)
            mpl.draw()

        # scanner image control - min and max
        # max is controlled by pageup and down
        # min is controlled by home and end
        # reset is 'r'
        elif event.key in ['pagedown', '\x1b[6~']: # pgdown - decrease max int
            MT.SI_minmax[1] = np.max((0., MT.SI_minmax[1] - 0.05))
            # print('clim pgdn: ', MT.SI_ax.get_clim())
            if MT.SI_minmax[1] > MT.SI_minmax[0]:
                MT.SI_ax.set_clim(MT.SI_minmax)
                mpl.draw()

        elif event.key in ['pageup', '\x1b[6~']: # pgup - increase max int
            # print('climpgup: ', MT.SI_ax.get_clim())
            MT.SI_minmax[1] = np.min((1.0, MT.SI_minmax[1] + 0.05))
            if MT.SI_minmax[1] > MT.SI_minmax[0]:
                MT.SI_ax.set_clim(MT.SI_minmax)
                mpl.draw()

        elif event.key in ['end', '\x1b[F']: # end -decrease min int
            MT.SI_minmax[0] = np.max((0., MT.SI_minmax[0] - 0.05))
            if MT.SI_minmax[1] > MT.SI_minmax[0]:
                # print('minmax end: ', MT.SI_minmax)
                MT.SI_ax.set_clim(MT.SI_minmax)
                mpl.draw()

        elif event.key in ['home', '\x1b[H']: # home - increase max int
            MT.SI_minmax[0] = np.min((MT.SI_minmax[1], MT.SI_minmax[0] + 0.05))
            if MT.SI_minmax[1] > MT.SI_minmax[0]:
                # print('minmax home: ', MT.SI_minmax)
                MT.SI_ax.set_clim(MT.SI_minmax)
                mpl.draw()
        
        elif event.key in ['r']: # reset
            MT.SI_minmax = MT.SI_original_minmax
            MT.SI_ax.set_clim(MT.SI_minmax)
            mpl.draw()

        # image control group
        # grayscale image
        elif event.key in ['+']:  # adjust max up or down
            MT.cmax = np.min((65535, MT.cmax + 500.))
            MT.imageax.set_clim(MT.cmin, MT.cmax)
            mpl.draw()
        elif event.key in ['-']:
            MT.cmax = np.max((MT.cmin, MT.cmax - 500.))
            MT.imageax.set_clim(MT.cmin, MT.cmax)
            mpl.draw()

        elif event.key in ['u']: # adjust min up or down
            MT.cmin = np.min((MT.cmax-10., MT.cmin + 100.))
            MT.imageax.set_clim(MT.cmin, MT.cmax)
            mpl.draw()

        elif event.key in ['d']:
            MT.cmin = np.max((0, MT.cmin - 100.))
            MT.imageax.set_clim(MT.cmin, MT.cmax)
            mpl.draw()

        # map and T bar
        elif event.key in ['t']  and shiftkey:
            MT.toggle_tbar(event)
            
        elif event.key in ['t'] and not shiftkey:
            # set the top t-bar position
            MT.update_tbar(event)
        
        elif event.key in ['s']: # draw scholl
            MT.draw_scholl(5e-5, 5)  # 50 um spacing, 5 circles
            
        elif event.key in ['v']:
            print(f'Cmin, max: {MT.cmin:.1f}\t{MT.cmax:.1f}')

        elif event.key in ['c']: # report cell position
            print(f'Cell Position: {event.xdata:.9f},{event.ydata:.9f}')
            MT.cellPos = [event.xdata, event.ydata]
        
        elif event.key in ['m']:
            MT.compute_sector_distance_map()
            MT.plot_sector_distance_map()  # plot the section distance map
        
        # these move the calibration bar
        elif event.key in ['right', '\x1b[C']:
            MT.reposition_cal(movex=-1)  # move right
        elif event.key in ['left', '\x1b[D']:
            MT.reposition_cal(movex=1)  # move left
        elif event.key in ['up', '\x1b[A']:
            MT.reposition_cal(movey=1)
        elif event.key in ['down', '\x1b[B']:
            MT.reposition_cal(movey=-1)
        elif event.key in ['h']  and not shiftkey: 
            MT.reposition_cal(home=True)  # home
        else:
            print('event key not mapped: ', str(event.key))
        mpl.draw()
    

    def plot_a_cell(cellname, cellno, ax=None, axb=None, tbar=False):
        dc = table.loc[table['cellname'] == cellname]
        dc = dc.loc[table['cellno'].isin([cellno])]
        if len(dc) == 0:
            print(f"Did not find cellname: {cellname:s}  with no: {cellno:s}")
            return False
        # print('cellname: ', cellname)
        cell = Path(basepath, str(dc['cellID'].values[0]), str(dc['map'].values[0]))
        imagename = dc['image'].values[0]
        print('image: ', imagename)
        if not pd.isnull(imagename) and imagename.startswith('image_') or imagename.startswith('../image_'):
            image = Path('..', imagename).with_suffix('.tif')
        elif not pd.isnull(imagename) and imagename.startswith('video_'):
            image = Path('..', imagename).with_suffix('.ma')
        else:
            image = None
        if not pd.isnull(dc['mosaic'].values[0]):
            mosaic = '../' + str(dc['mosaic'].values[0]) + '.mosaic'
        else:
            mosaic = None
        
        pars = makepars(dc)
        MT.setPars(pars)
        ER = EventReader(experiments[experimentname]['directory'], dc['cellID'].values[0], dc['map'].values[0])
        MT.setEventData(ER.data)
        MT.setWindow(dc['x0'].values[0], dc['x1'].values[0], dc['y0'].values[0], dc['y1'].values[0])
        MT.setOutputFile(Path(experiments[experimentname]['directory'], f"{cellname:s}{int(cellno):d}_map.pdf"))
        prots = {'ctl': cell}
        MT.setProtocol(cell, image=image, mosaic=mosaic)
        MT.plot_maps(prots, ax=ax, linethickness=0.5, tbar=tbar, axb=axb)
        return True

    def count_pages(cs, nperpage=8):
        nmaps = len(cs)
        npages = np.floor(nmaps/nperpage)
        if npages*nperpage < nmaps:
            npages += 1
        return int(npages)

    def plot_cells(cellname, sequence, tbar=False):
        
        if len(sequence) > 1:  # all of a type
            sequence = [int(x) for x in sequence]
            cs = table.loc[table['cellname'] == cellname[0]]
            cs = cs.loc[table['cellno'].isin(sequence)]
            print('len cs: ', len(cs))
            nperpage = len(sequence)
            npages = count_pages(cs, nperpage=nperpage)
            lcs = list(cs.index)
            print('npages: ', npages)
            icell = 0
            for npage in range(npages):
                c, r = PH.getLayoutDimensions(nperpage, pref='height')
                # c = 2
                # r = int(nperpage/c)
                P1 = PH.regular_grid(r, c, order='rowsfirst', figsize=(10., 12.), showgrid=False,
                    verticalspacing=0.08, horizontalspacing=0.08,
                    margins={'leftmargin': 0.07, 'rightmargin': 0.05, 'topmargin': 0.12, 'bottommargin': 0.1},
                    labelposition=(0., 0.), parent_figure=None, panel_labels=None)
                P2 = PH.regular_grid(r, c, order='rowsfirst', figsize=(10., 12.), showgrid=False,
                    verticalspacing=0.08, horizontalspacing=0.08,
                    margins={'leftmargin': 0.07, 'rightmargin': 0.05, 'topmargin': 0.12, 'bottommargin': 0.1},
                    labelposition=(0., 0.), parent_figure=None, panel_labels=None)

                axarr = P1.axarr.ravel()
                axarr2 = P2.axarr.ravel()
                for axn in range(nperpage):
                    if icell >= len(cs):  # done
                        break
                    print('page, axn, i: ', npage, axn, icell)
                    print('cell, cell number: ', cs.iloc[icell]['cellname'], cs.iloc[icell]['cellno'])
                    plot_a_cell(cs.iloc[icell]['cellname'], cs.iloc[icell]['cellno'], ax=axarr[axn],
                        tbar=tbar, axb=axarr2[axn])
                    
                    # reduce cell name:
                    cname = cs.iloc[icell]['cellID'].replace('slice_00', 'S').replace('cell_00', 'C')
                    axarr[axn].set_title(f"{cname:s} #={cs.iloc[icell]['cellno']:d}\n{cs.iloc[icell]['map']:s}",
                        fontsize=9, horizontalalignment='center')
                    axarr2[axn].set_title(f"{cname:s} #={cs.iloc[icell]['cellno']:d}\n{cs.iloc[icell]['map']:s}",
                        fontsize=9, horizontalalignment='center')
                    sn = sequence[icell]
                    icell += 1
                P1.figure_handle.suptitle(f"Celltype: {cellname[0]:s} Page: {npage:d}", fontsize=14)
                P2.figure_handle.suptitle(f"Celltype: {cellname[0]:s} Page: {npage:d} distancemaps", fontsize=14)
                ymax = 0
                for ax2 in axarr2:
                    yl = ax2.get_ylim()
                    ymax = np.max((ymax, yl[1]))
                for ax2 in axarr2:
                    ax2.set_ylim((0, ymax))  # make all y axes have the same scale
                if sn > 100 and sn < 300:
                    t = '_pairs'
                    seq = args.sequence.replace(';', '-')
                elif sn > 300:
                    t = '_TTX'
                    seq = args.sequence.replace(';', '-')
                else:
                    t = ''
                    seq = args.sequence.replace(';' '-')
                P1.figure_handle.savefig(Path('NF107Ai32_Het/Example Maps/', f"{cellname[0]:s}{t:s}_{seq:s}_Page{npage:d}.pdf"))
                P2.figure_handle.savefig(Path('NF107Ai32_Het/Example Maps/', f"{cellname[0]:s}{t:s}_{seq:s}_Page{npage:d}_distmap.pdf"))
                # mpl.close()

        else: # specific cell
            c, r = PH.getLayoutDimensions(1, pref='height')
            P = PH.regular_grid(r, c, order='columnsfirst', figsize=(8., 10), showgrid=False,
                verticalspacing=0.08, horizontalspacing=0.08,
                margins={'leftmargin': 0.07, 'rightmargin': 0.05, 'topmargin': 0.03, 'bottommargin': 0.1},
                labelposition=(0., 0.), parent_figure=None, panel_labels=None)
            success = plot_a_cell(cellname[0], cellno=str(int(sequence[0])), ax=P.axarr[0,0])
            if not success:
                return
            # drawtype is 'box' or 'line' or 'none'
            # print(dir(MT))
            rectprops = dict(facecolor='yellow', edgecolor = 'black',
                             alpha=0.2, fill=True)
            toggle_selector.RS = RectangleSelector(MT.ax, line_select_callback,
                                                   drawtype='box', useblit=True,
                                                   button=[1, 3],  # don't use middle button
                                                   minspanx=1e-5, minspany=1e-5,
                                                   spancoords='data',
                                                   interactive=True, 
                                                   rectprops=rectprops)

            mpl.connect('key_press_event', toggle_selector)
            mpl.connect('key_release_event', release_selector)
            mpl.show()
    
    
    print(docell, sequence)
    plot_cells(docell, sequence, tbar=args.plotwithtbar)
    
if __name__ == '__main__':
    main()

    