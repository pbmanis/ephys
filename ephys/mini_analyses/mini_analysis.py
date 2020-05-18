from __future__ import print_function

"""
Analysis of miniature synaptic potentials
Provides measure of event amplitude distribution and event frequency distribution
The analysis is driven by an imported dictionary.

Example for the data table:
self.basedatadir = '/Volumes/Pegasus/ManisLab_Data3/Sullivan_Chelsea/miniIPSCs'

datasets = {'m1a': {'dir': '2017.04.25_000/slice_000/cell_001', 'prots': [0,1,3], 
				'thr': 1.75, 'rt': 0.35, 'decay': 6., 'G': 'F/+', 'exclist': []},
            'm1b': {'dir': '2017.04.25_000/slice_000/cell_002', 'prots': [7], 
				'thr': 1.75, 'rt': 0.35, 'decay': 6., 'G': 'F/+', 'exclist': []},
            'm2a': {'dir': '2017.05.02_000/slice_000/cell_000/', 'prots': [0,1,2], 
				'thr': 1.75, 'rt': 0.32, 'decay': 5., 'G': 'F/+', 'exclist': []},
            'm2b': {'dir': '2017.05.02_000/slice_000/cell_001', 'prots': [0,1,2], 
				'thr': 1.75, 'rt': 0.35, 'decay': 4., 'G': 'F/+', 'exclist': {1: [4, 5, 6], 2: [8]}},
            'm2c': {'dir': '2017.05.02_000/slice_000/cell_002', 'prots': [0,1,2], 
        }

Where:
each dict key indicates a cell from a mouse (mice are numbered, cells are lettered)
'dir': The main cell directory, relative to the base directory,
'prots': a list of the protocols to be analyzed,
'exclist': a dict of the protocols that have traces to be excluded
    The excluded traces are in a tuple or list for each protocol.
    For example, exclist: {0 : [1,2], 1: [3,4,5]} results in the exclusion of
        traces 1 and 2 from protocol run 0, and traces 3, 4 and 5 from protocol run 1
'thr' : SD threshold for event detection (algorithm dependent)
'rt' : rise time for template (in msec)
'decay': decay time constant for the template (in msec)
'G' : group identifier (e.g, genotype, treatment, etc.)

Requires asome of Manis' support libraries/modules, including:
    ephysanalysis module  (git clone https://github/pbmanis/ephysanalysis)
    pylibrary utilities, (git clone https://github/pbmanis/ephysanalysis)
     
Output summary is a Python pickle file (.p extension) that isread by mini_summary_plots.py

Paul B. Manis, Ph.D. Jan-March 2018.

"""

import sys
import os
import argparse
from collections import OrderedDict
from pathlib import Path
import numpy as np
import pickle
import pyqtgraph.multiprocess as mp
from matplotlib import rc
import matplotlib.pyplot as mpl
import matplotlib.gridspec as gridspec
import scipy.signal
import scipy.stats

import pylibrary.tools.utility as PU
import pylibrary.plotting.plothelpers as PH
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.collections as collections
import matplotlib.path as mplpath
from matplotlib.path import Path


from . import minis_methods as minis
from . import digital_filters as DF
from .. import ephysanalysis as EP


rc('text', usetex=False)
#rc('font',**{'family':'sans-serif','sans-serif':['Verdana']})



def splitall(path):
    """
    split a filename into a list of individual directories, in order
    
    Parameters
    ----------
    path : str (no default)
        The path to be split
    
    Returns
    -------
    list : all elements of the path in order
    """
    
    allparts = []
    while 1:
        parts = os.path.split(path)
        if parts[0] == path:  # sentinel for absolute paths
            allparts.insert(0, parts[0])
            break
        elif parts[1] == path: # sentinel for relative paths
            allparts.insert(0, parts[1])
            break
        else:
            path = parts[0]
            allparts.insert(0, parts[1])
    return allparts


class MiniAnalysis():
    def __init__(self, dataplan):
        """
        Perform detection of miniature synaptic events, and some analysis
        
        Parameters
        ----------
        dataplan : object
            a dataplan object, with 
                datasource: the name of the file holding the dict
                datadir : the path to the data itself
                datasets : the dict with information driving the analysis
        """

        self.datasource = dataplan.datasource
        self.basedatadir = dataplan.datadir
        self.datasets = dataplan.datasets
        self.dataplan_data = dataplan.data
        try:
            self.global_threshold = dataplan.data['global_threshold']
            self.override_threshold = True
        except:
            self.override_threshold = False
        try:
            self.global_decay = dataplan.data['global_decay']
            self.override_decay = True
        except:
            self.override_decay = False
        
        self.filter = False
        self.filterstring = 'notfiltered'
        try:
            self.filter = dataplan.data['notch_and_hpf_filter']
            if self.filter:
                self.filterstring = 'filtered'
        except:
            self.filter = False
            self.filterstring = 'notfiltered'
        try:
            self.min_event_amplitude = dataplan.data['min_event_amplitude']
        except:
            self.min_event_amplitude = 2.0
        
    # from acq4 functions:
    def measure_baseline(self, data, threshold=2.0, iterations=2):
        """Find the baseline value of a signal by iteratively measuring the median value, then excluding outliers."""
        data = data.view(np.ndarray)
        med = np.median(data)
        if iterations > 1:
            std = data.std()
            thresh = std * threshold
            arr = np.ma.masked_outside(data, med - thresh, med + thresh)
            try:
                len(arr.mask)  # could be an array or just a boolean
            except:
                if arr.mask == False:  # nothing to mask... 
                    return med
            if len(arr) == 0:
                raise Exception("Masked out all data. min: %f, max: %f, std: %f" % (med - thresh, med + thresh, std))
            return self.measure_baseline(arr[~arr.mask], threshold, iterations-1)
        else:
            return med
            
    def analyze_all(self, fofilename, check=False, mode='aj', engine='cython'):
        """
        Wraps analysis of individual data sets, writes plots to 
        a file named "summarydata%s.p % self.datasource" in pickled format.
    
        Parameters
        ----------
        fofilename : str (no default)
            name of the PDF plot output file
        check : bool (default: False)
            If true, run just checks for the existence of the data files, 
            but does no analysis.
    
        Returns
        -------
        Nothing
        """
        acqr = EP.acq4read.Acq4Read(dataname='Clamp1.ma')  # creates a new instance every time - probably should just create one.
        summarydata = {}
        with PdfPages(fofilename) as pdf:
            for id, mouse in enumerate(sorted(self.datasets.keys())):
                self.analyze_one(mouse, pdf, maxprot=10, arreader=acqr, check=check, mode=mode, engine=engine,)
                summarydata[mouse] = self.cell_summary
        if not check:
            fout = 'summarydata_{0:s}_{1:s}_{2:s}.p'.format(self.datasource, self.filterstring, mode)
            fh = open(fout, 'wb')
            pickle.dump(summarydata, fh)
            fh.close()
        else:
            print('All files found (an exception would be raised if one was not found)')

    def analyze_one(self, mouse, pdf, maxprot=10, arreader=None, check=False, mode='aj', engine='cython'):
        """
        Provide analysis of one entry in the data table using the Andrade_Joans algorithm
        and fitting distributions. 
        Generates a page with plots for each protocol/trace stacked, and histograms with
        fits for the interval and amplitude distributions
    
        Parameters
        ----------
        mouse : str (no default)
            key into the dictionary for the data to be analyzed
        pdf : pdf file object (no default):
            the pdf file object to write the plots to.
        maxprot : int (default: 10)
            Maximum numober of protocols to do in the analysis
        check : bool (default: False)
            If true, run just checks for the existence of the data files, 
            but does no analysis.    

        Returns
        -------
            cell summary dictionary for the 'mouse' entry. 
        """
        if arreader is None:
            acqr = EP.acq4read.Acq4Read(dataname='Clamp1.ma')  # only if we don't already have one
        else:
            acqr = arreader
        self.rasterize = True    # set True to rasterize traces to reduce size of final document
                            # set false for pub-quality output (but large size)
        self.acqr = acqr
        dt = 0.1
        mousedata = self.datasets[mouse]
        if self.override_threshold:
            mousedata['thr'] = self.global_threshold  # override the threshold setting
        if self.override_decay:
            mousedata['decay'] = self.global_decay  # override decay settings
            
        self.sign = 1
        if 'sign' in self.dataplan_data:
            self.sign = int(self.dataplan_data['sign'])
            
        print ('\nMouse: ', mouse)
        genotype = mousedata['G']
        excl = mousedata['exclist']  # get the exclusion list
        self.cell_summary={'intervals': [], 'amplitudes': [], 'protocols': [], 'eventcounts': [],
            'genotype': genotype, 'mouse': mouse, 
            'amplitude_midpoint': 0., 'holding': [], 'averaged': [], 'sign': [], 'threshold': [],
            'indiv_evok': [], 'indiv_notok': [], 'indiv_amp': [], 'indiv_fitamp': [], 
            'indiv_tau1': [], 'indiv_tau2': [], 'indiv_fiterr': [], 'indiv_Qtotal': [],
            'indiv_tb': [], 'allevents': [], 'fitted_events': [],
            'best_fit': [], 'best_decay_fit': [], 
            }
    
        if not check:
            self.plot_setup()
        datanameposted = False
        self.yspan = 40.
        self.ypqspan = 2000.
        ntr = 0
        # for all of the protocols that are included for this cell (identified by mouse and a letter)
        print ('mousedata prots: ', mousedata['prots'])

        for nprot, dprot in enumerate(mousedata['prots']):
            if nprot > maxprot:
                return
            self.nprot = nprot
            self.dprot = dprot
            exclude_traces = []
            if len(mousedata['exclist']) > 0:  # any traces to exclude?
                if dprot in mousedata['exclist'].keys():
                    #print (mousedata['exclist'], dprot, nprot)
                    exclude_traces = mousedata['exclist'][dprot]

            try:
                sign = self.dataplan_data['sign']
            except:
                raise ValueError('Event Sign is not defined in the cell data')

            fn = os.path.join(self.basedatadir, mousedata['dir'])
            fx, ext = os.path.splitext(fn)
            #print('ext: ', ext, fn)
            if not check:
                print('Protocol file: ', fn)
                print('   sign: ', sign)
            if ext != '.mat':
                try:
                    fn = os.path.join(fn, ('minis_{0:03d}'.format(dprot)))
                except:
                    print('path failed to join')
                    print ('    ', dprot, nprot)
                    print ('    ', mousedata['prots'])
                    print('     ', dprot, mousedata['prots'][nprot])
                    exit(1)
                split = splitall(fn)[-4:-1]
                dataname = ''
                for i in range(len(split)):
                    dataname = os.path.join(dataname, split[i])
                acqr.setProtocol(fn)
            else:
                df = EP.MatdatacRead.DatacFile(fn)
                dataname = fn
                # print ('df: ', df.summary())
            if not check:
                print('  Protocol dataname: ', dataname)
                print('  exclude traces: ',  exclude_traces)

            dainfo = fn

            if check:
                try:
                    result = acqr.getData(check=True)
                    if result is False:
                        sys.stdout.write(r'\n\033[0;31m******* Get data failed to find a file ', fn, dataname, ' *******\033[0;97m\n')
                    else:
                        # print('getdata finished')
                        continue
                except:
                    print('  ******* Get data failed... in acqr.getData ', fn, dataname)
                    continue
                continue
            print('ext: ', ext)
            if ext != '.mat':
                print('acq4')
                acqr.getData()
                data = np.array(acqr.data_array)
                time_base = acqr.time_base
                dt = 1000.*acqr.sample_interval  # in sec... so convert to msec
            else:
                print('.mat')
                DF = EP.MatdatacRead.DatacFile(fn)
                self.acqr = EP.MatdatacRead.GetClamps(DF, self.basedatadir)
                self.acqr.getClampData()
                print('self.acq4: ', self.acqr)
                time_base = self.acqr.time_base
                data = self.acqr.traces
                dt = self.acqr.sample_interval
            if not datanameposted and not check:
                self.P.figure_handle.suptitle(mouse + '  ' + dataname + ' : ' + genotype, fontsize=9, weight='bold')
                datanameposted = True

            data = data*1e12  # convert to pA
            time_base = time_base*1000.
            maxt = np.max(time_base)
            tracelist, nused = self.analyze_block_traces(mode, data, time_base, maxt, dt, mousedata, exclude_traces, ntr)
            ntr = ntr + len(tracelist)# - len(exclude_traces)
            if nused == 0:
                continue

        if check:
            return None # no processing
        # summarize the event and amplitude distributions
        # For the amplitude, the data are fit against normal, skewed normal and gamma distributions.
        self.plot_hists()
        # show to sceen or save plots to a file
        if pdf is None:
            mpl.show()
        else:
            pdf.savefig(dpi=300)  # rasterized to 300 dpi is ok for documentation.
            mpl.close()
        self.plot_individual_events(fit_err_limit=50., 
                    title=('{0:s}  {1:s}  {2:s}'.format(dataname, self.cell_summary['mouse'], self.cell_summary['genotype'])),
                    pdf=pdf)

    def analyze_block_traces(self, mode, data, time_base, maxt, dt, mousedata, exclude_traces, ntr):
        if mode == 'aj':
            aj = minis.AndradeJonas()
            aj.setup(tau1=mousedata['rt'], tau2=mousedata['decay'],
                    template_tmax=maxt, dt=dt, delay=0.0, sign=self.sign,
                    risepower=1.0, min_event_amplitude=self.min_event_amplitude)
        elif mode == 'cb':
            cb = minis.ClementsBekkers()
            cb.setup(tau1=mousedata['rt'], tau2=mousedata['decay'],
                        template_tmax=3.0*mousedata['decay'], dt=dt, delay=0.0, sign=self.sign,
                        risepower=1.0, min_event_amplitude=self.min_event_amplitude)
            #aj.set_cb_engine('numba')
            cb.set_cb_engine('cython')
        else:
            raise ValueError('Mode must be aj or cb for event detection')
#        print('mousedata rt: ', mousedata['rt'], '   mousedata decay: ', mousedata['decay'])

        mwin = int(2*(4.)/dt)
        order = int(1.0/dt)
        ntraces = data.shape[0]
        tracelist = range(ntraces)
        print ('  tracelist: ', tracelist, ' exclude: ', exclude_traces)
        self.ax0.set_xlim(-1500., np.max(self.acqr.time_base)*1000.)
        nused = 0
        # for all of the traces collected in this protocol run

        parallel = True
        tasks = []
        for i in tracelist:
            tasks.append(i)
        results = {}
        workers = 1 if not parallel else 4
        tot_runs = len(tracelist)


#         with mp.Parallelize(enumerate(tasks), results=results, progressDialog='Running parallel simulation..', workers=workers) as tasker:
#             for i, task in tasker:
#                 print("=== Start analysis %d (Workers=%d)===" % (i+1, workers))
# #                tasker.results[(f, db, iteration)] = (stim, result)
        
        for i in tracelist:
            yp = (ntr+i)*self.yspan
            ypq = (ntr*i)*self.ypqspan
            linefit= np.polyfit(time_base, data[i], 1)
            refline = np.polyval(linefit, time_base)
            holding = self.measure_baseline(data[i])
            odata = data[i].copy()
            data[i] = data[i] - refline  # linear correction
            self.ax0.text(-1200, yp, '%03d %d' % (self.dprot, i), fontsize=8)  # label the trace
            if i in exclude_traces:  # plot the excluded traces, but do not analyze them
                print('    **** Trace {0:d} excluded in list'.format(i))
               # self.ax0.plot(self.acqr.time_base*1000., odata + yp ,'y-', linewidth=0.25, alpha=0.25, rasterized=self.rasterize)
                continue
            if holding < self.dataplan_data['holding']:
              #  self.ax0.plot(self.acqr.time_base*1000., odata + yp ,'m-', linewidth=0.25, alpha=0.25, rasterized=self.rasterize)
                print('    >>>> Trace {0:d} excluded for holding {1:.3f}'.format(i, holding))
            nused = nused + 1
    
           # frs, freqs = PU.pSpectrum(data[i], samplefreq=1000./dt)
            if self.filter: # notch and HPF traxes
                dfilt = DF.NotchFilter(data[i], [60., 120., 180., 240., 300., 360, 420.,
                                480., 660., 780., 1020., 1140., 1380., 1500., 4000.], Q=20., samplefreq=1000./dt)
                #dfilt = DF.SignalFilter_HPFButter(np.pad(dfilt, (len(dfilt), len(dfilt)), mode='median', ), 2.5, 1000./dt, NPole=4)
                #data[i] = dfilt[len(data[i]):2*len(data[i])]  # remove padded segments
                data[i] = dfilt
        
            # data[i] = DF.SignalFilter_LPFButter(data[i], 2800., 1000./dt, NPole=4) # LPF THE DATA
#            frsfilt, freqsf = PU.pSpectrum(dfilt, samplefreq=1000./dt)

            if mode == 'aj':
                aj.deconvolve(data[i], thresh=float(mousedata['thr']), llambda=10., order=order)
            else:
                cb.cbTemplateMatch(data[i],  threshold=float(mousedata['thr']), order=order)

            intervals = np.diff(aj.timebase[aj.onsets])
            self.cell_summary['intervals'].extend(intervals)
            self.cell_summary['averaged'].extend([{'tb': aj.avgeventtb, 'avg': aj.avgevent, 'fit': {'amplitude': aj.Amplitude,
                'tau1': aj.tau1, 'tau2': aj.tau2, 'risepower': aj.risepower}, 'best_fit': aj.avg_best_fit,
                'risetenninety': aj.risetenninety, 'decaythirtyseven': aj.decaythirtyseven,
                'Qtotal': aj.Qtotal}])
            #print('smooth peak index: ', aj.smpkindex)
            self.cell_summary['amplitudes'].extend(aj.sign*data[i][aj.smpkindex])  # smoothed peak amplitudes
            self.cell_summary['eventcounts'].append(len(intervals))
            self.cell_summary['protocols'].append((self.nprot, i))
            self.cell_summary['holding'].append(holding)
            self.cell_summary['sign'].append(aj.sign)
            self.cell_summary['threshold'].append(mousedata['thr'])
        
            aj.fit_individual_events() # fit_err_limit=2000., tau2_range=2.5)  # on the data just analyzed
            self.cell_summary['indiv_amp'].append(aj.ev_amp)
            self.cell_summary['indiv_fitamp'].append(aj.ev_fitamp)
            self.cell_summary['indiv_tau1'].append(aj.ev_tau1)
            self.cell_summary['indiv_tau2'].append(aj.ev_tau2)
            self.cell_summary['indiv_fiterr'].append(aj.fiterr)
            self.cell_summary['fitted_events'].append(aj.fitted_events)
            self.cell_summary['indiv_Qtotal'].append(aj.ev_Qtotal)
            self.cell_summary['indiv_evok'].append(aj.events_ok)
            self.cell_summary['indiv_notok'].append(aj.events_notok)
            self.cell_summary['allevents'].append(np.array(aj.allevents))
            self.cell_summary['best_fit'].append(np.array(aj.best_fit))
            self.cell_summary['best_decay_fit'].append(np.array(aj.best_decay_fit))
#            for jev in range(len(aj.allevents)):
#                self.cell_summary['allevents'].append(aj.allevents[jev])
#                self.cell_summary['best_fit'].append(aj.best_fit[jev])
            self.cell_summary['indiv_tb'].append(aj.avgeventtb)        
            self.ax0.plot(aj.timebase, odata + yp ,'c-', linewidth=0.25, alpha=0.25, rasterized=self.rasterize)
            self.ax0.plot(aj.timebase, data[i] + yp, 'k-', linewidth=0.25, rasterized=self.rasterize)
            self.ax0.plot(aj.timebase[aj.smpkindex], data[i][aj.smpkindex] + yp, 'ro', markersize=1.75, rasterized=self.rasterize)
        
            if 'A1' in self.P.axdict.keys():
                self.axdec.plot(aj.timebase[:aj.Crit.shape[0]],  aj.Crit, label='Deconvolution') 
                self.axdec.plot([aj.timebase[0],aj.timebase[-1]],  [aj.sdthr,  aj.sdthr],  'r--',  linewidth=0.75, 
                        label='Threshold ({0:4.2f}) SD'.format(aj.sdthr))
                self.axdec.plot(aj.timebase[aj.onsets]-aj.idelay,  ypq + aj.Crit[aj.onsets],  'y^', label='Deconv. Peaks')
    #            axdec.plot(aj.timebase, aj.Crit+ypq, 'k', linewidth=0.5, rasterized=self.rasterize)
            print('--- finished run %d/%d ---' % (i+1, tot_runs))
                
        return tracelist, nused
        

    def plot_individual_events(self, fit_err_limit=1000., tau2_range=2.5, title='', pdf=None):
        P = PH.regular_grid(3 , 3, order='columns', figsize=(8., 8.), showgrid=False,
                        verticalspacing=0.1, horizontalspacing=0.12,
                        margins={'leftmargin': 0.12, 'rightmargin': 0.12, 'topmargin': 0.03, 'bottommargin': 0.1},
                        labelposition=(-0.12, 0.95))
        P.figure_handle.suptitle(title)
        all_evok = self.cell_summary['indiv_evok']  # this is the list of ok events - a 2d list by 
        all_notok = self.cell_summary['indiv_notok']
        # print('all evok: ', all_evok)
        # print('len allevok: ', len(all_evok))
        #
        # # print('all_notok: ', all_notok)
        # # print('indiv tau1: ', self.cell_summary['indiv_tau1'])
        # exit(1)
        trdat = []
        trfit = []
        trdecfit = []
        for itr in range(len(all_evok)):  # for each trace
            for evok in all_evok[itr]:  # for each ok event in that trace
                P.axdict['A'].plot(self.cell_summary['indiv_tau1'][itr][evok], self.cell_summary['indiv_amp'][itr][evok], 'ko', markersize=3)
                P.axdict['B'].plot(self.cell_summary['indiv_tau2'][itr][evok], self.cell_summary['indiv_amp'][itr][evok], 'ko', markersize=3)
                P.axdict['C'].plot(self.cell_summary['indiv_tau1'][itr][evok], self.cell_summary['indiv_tau2'][itr][evok], 'ko', markersize=3)
                P.axdict['D'].plot(self.cell_summary['indiv_amp'][itr][evok], self.cell_summary['indiv_fiterr'][itr][evok], 'ko', markersize=3)
                P.axdict['H'].plot(self.cell_summary['indiv_tau1'][itr][evok], self.cell_summary['indiv_Qtotal'][itr][evok], 'ko', markersize=3)
                trdat.append(np.column_stack([self.cell_summary['indiv_tb'][itr], self.cell_summary['allevents'][itr][evok]]))
                #idl = len(self.cell_summary['best_decay_fit'][itr][evok])
                trfit.append(np.column_stack([self.cell_summary['indiv_tb'][itr], 
                    -self.cell_summary['best_fit'][itr][evok]]))
                trdecfit.append(np.column_stack([self.cell_summary['indiv_tb'][itr], 
                    -self.cell_summary['best_decay_fit'][itr][evok]]))
        dat_coll = collections.LineCollection(trdat, colors='k', linewidths=0.5)
        fit_coll = collections.LineCollection(trfit, colors='r', linewidths=0.25)
#        decay_fit_coll = collections.LineCollection(trdecfit, colors='c', linewidths=0.3)
        P.axdict['G'].add_collection(dat_coll)
        P.axdict['G'].add_collection(fit_coll)
#        P.axdict['G'].add_collection(decay_fit_coll) 
        n_trdat = []
        n_trfit = []
        for itr in range(len(all_notok)):
            for notok in all_notok[itr]:
                n_trdat.append(np.column_stack([self.cell_summary['indiv_tb'][itr], self.cell_summary['allevents'][itr][notok]]))
                n_trfit.append(np.column_stack([self.cell_summary['indiv_tb'][itr], -self.cell_summary['best_fit'][itr][notok]]))
                P.axdict['D'].plot(self.cell_summary['indiv_amp'][itr][notok], self.cell_summary['indiv_fiterr'][itr][notok], 'ro', markersize=3)
        n_dat_coll = collections.LineCollection(n_trdat, colors='b', linewidths=0.35)
        n_fit_coll = collections.LineCollection(n_trfit, colors='y', linewidths=0.25)
        P.axdict['E'].add_collection(n_dat_coll) 
        P.axdict['E'].add_collection(n_fit_coll) 
        
        P.axdict['A'].set_xlabel(r'$tau_1$ (ms)')
        P.axdict['A'].set_ylabel(r'Amp (pA)')
        P.axdict['B'].set_xlabel(r'$tau_2$ (ms)')
        P.axdict['B'].set_ylabel(r'Amp (pA)')
        P.axdict['C'].set_xlabel(r'$\tau_1$ (ms)')
        P.axdict['C'].set_ylabel(r'$\tau_2$ (ms)')
        P.axdict['D'].set_xlabel(r'Amp (pA)')
        P.axdict['D'].set_ylabel(r'Fit Error (cost)')
        P.axdict['H'].set_xlabel(r'$\tau_1$ (ms)')
        P.axdict['H'].set_ylabel(r'Qtotal')
        P.axdict['G'].set_ylim((-100., 20.))
        P.axdict['G'].set_xlim((-2., 25.))
        P.axdict['E'].set_ylim((-100., 20.))
        P.axdict['E'].set_xlim((-2., 25.))

# put in averaged event too
        # self.cell_summary['averaged'].extend([{'tb': aj.avgeventtb, 'avg': aj.avgevent, 'fit': {'amplitude': aj.Amplitude,
        #     'tau1': aj.tau1, 'tau2': aj.tau2, 'risepower': aj.risepower}, 'best_fit': aj.avg_best_fit,
        #     'risetenninety': aj.risetenninety, 'decaythirtyseven': aj.decaythirtyseven}])
        aev = self.cell_summary['averaged']
        for i in range(len(aev)):
            P.axdict['F'].plot(aev[i]['tb'], aev[i]['avg'], 'k-', linewidth=0.8)
            P.axdict['F'].plot(aev[i]['tb'], aev[i]['best_fit'], 'r--', linewidth=0.4)

            
        if pdf is None:
            mpl.show()
        else:
            pdf.savefig(dpi=300)
            mpl.close()
    
    def plot_all_events_and_fits(self):
        P3 = PH.regular_grid(1 , 5, order='columns', figsize=(12, 8.), showgrid=False,
                        verticalspacing=0.1, horizontalspacing=0.02,
                        margins={'leftmargin': 0.07, 'rightmargin': 0.05, 'topmargin': 0.03, 'bottommargin': 0.05},
                        labelposition=(-0.12, 0.95))        
        idx = [a for a in P3.axdict.keys()]
        ncol = 5
        offset2 = 0.
        k = 0
        for itr in range(len(all_evok)):  # for each trace
            for evok in all_evok[itr]:  # for each ok event in that trace
                P3.axdict[idx[k]].plot([self.cell_summary['indiv_tb'][itr][0],self.cell_summary['indiv_tb'][itr][-1]], 
                            np.zeros(2)+offset2, 'b--', linewidth=0.3)
                P3.axdict[idx[k]].plot(self.cell_summary['indiv_tb'][itr], 
                                self.cell_summary['allevents'][itr][evok]+offset2, 'k--', linewidth=0.5)
                P3.axdict[idx[k]].plot(self.cell_summary['indiv_tb'][itr], 
                                -self.cell_summary['best_fit'][itr][evok]+offset2, 'r--', linewidth=0.5)
                if k == 4:
                    k = 0
                    offset2 += 20.
                else:
                    k += 1
        mpl.show()
        
    def plot_hists(self):   # generate histogram of amplitudes for plots
        histBins = 50
        nsamp = 1
        nevents = len(self.cell_summary['amplitudes'])

        amp, ampbins, amppa= self.axAmps.hist(self.cell_summary['amplitudes'], histBins, alpha=0.5, density=True)

        # fit to normal distribution
        ampnorm = scipy.stats.norm.fit(self.cell_summary['amplitudes'])  #
        print('    Amplitude (N={0:d} events) Normfit: mean {1:.3f}   sigma: {2:.3f}'.format(nevents, ampnorm[0], ampnorm[1]))
        x = np.linspace(scipy.stats.norm.ppf(0.01, loc=ampnorm[0], scale=ampnorm[1]),
                         scipy.stats.norm.ppf(0.99, loc=ampnorm[0], scale=ampnorm[1]), 100)
        self.axAmps.plot(x, scipy.stats.norm.pdf(x, loc=ampnorm[0], scale=ampnorm[1]),
                'r-', lw=2, alpha=0.6, label='Norm: u={0:.3f} s={1:.3f}'
                    .format(ampnorm[0], ampnorm[1]))
        k2, p = scipy.stats.normaltest(self.cell_summary['amplitudes'])
        print("    p (amplitude is Gaussian) = {:g}".format(1-p))
        print('    Z-score for skew and kurtosis = {:g} '.format(k2))
    
        # fit to skewed normal distriubution
        ampskew = scipy.stats.skewnorm.fit(self.cell_summary['amplitudes'])
        print('    ampskew: mean: {0:.4f} skew:{1:4f}  scale/sigma: {2:4f} '.format(ampskew[1], ampskew[0], 2*ampskew[2]))
        x = np.linspace(scipy.stats.skewnorm.ppf(0.002, a=ampskew[0], loc=ampskew[1], scale=ampskew[2]),
                         scipy.stats.skewnorm.ppf(0.995, a=ampskew[0], loc=ampskew[1], scale=ampskew[2]), 100)
        self.axAmps.plot(x, scipy.stats.skewnorm.pdf(x, a=ampskew[0], loc=ampskew[1], scale=ampskew[2]),
                'm-', lw=2, alpha=0.6, label='skewnorm a={0:.3f} u={1:.3f} s={2:.3f}'
                    .format(ampskew[0], ampskew[1], ampskew[2]))

        # fit to gamma distriubution
        ampgamma = scipy.stats.gamma.fit(self.cell_summary['amplitudes'], loc=0.)
        gamma_midpoint = scipy.stats.gamma.ppf(0.5, a=ampgamma[0], loc=ampgamma[1], scale=ampgamma[2])  # midpoint of distribution
        print('    ampgamma: mean: {0:.4f} gamma:{1:.4f}  loc: {1:.4f}  scale: {3:.4f} midpoint: {4:.4f}'.
                        format(ampgamma[0]*ampgamma[2], ampgamma[0], ampgamma[2],  ampgamma[2], gamma_midpoint))
        x = np.linspace(scipy.stats.gamma.ppf(0.002, a=ampgamma[0], loc=ampgamma[1], scale=ampgamma[2]),
                         scipy.stats.gamma.ppf(0.995, a=ampgamma[0], loc=ampgamma[1], scale=ampgamma[2]), 100)
        self.axAmps.plot(x, scipy.stats.gamma.pdf(x, a=ampgamma[0], loc=ampgamma[1], scale=ampgamma[2]),
                'g-', lw=2, alpha=0.6, label='gamma: a={0:.3f} loc={1:.3f}\nscale={2:.3f}, mid={3:.3f}'
                    .format(ampgamma[0], ampgamma[1], ampgamma[2], gamma_midpoint) ) #ampgamma[0]*ampgamma[2]))
        self.axAmps.plot([gamma_midpoint, gamma_midpoint], [0., scipy.stats.gamma.pdf([gamma_midpoint],
                        a=ampgamma[0], loc=ampgamma[1], scale=ampgamma[2])], 'k--',
                        lw=2, alpha=0.5)
        self.axAmps.legend(fontsize=6)
        self.cell_summary['amplitude_midpoint'] = gamma_midpoint

        #
        # Interval distribution
        #
        an, bins, patches = self.axIntvls.hist(self.cell_summary['intervals'], histBins, density=True)
        nintvls = len(self.cell_summary['intervals'])
        expDis = scipy.stats.expon.rvs(scale=np.std(self.cell_summary['intervals']), loc=0, size=nintvls)
        # axIntvls.hist(expDis, bins=bins, histtype='step', color='r')
        ampexp = scipy.stats.expon.fit(self.cell_summary['intervals'])
        x = np.linspace(scipy.stats.expon.ppf(0.01, loc=ampexp[0], scale=ampexp[1]),
                         scipy.stats.expon.ppf(0.99, loc=ampexp[0], scale=ampexp[1]), 100)
        self.axIntvls.plot(x, scipy.stats.expon.pdf(x, loc=ampexp[0], scale=ampexp[1]),
                'r-', lw=3, alpha=0.6, label='Exp: u={0:.3f} s={1:.3f}\nMean Interval: {2:.3f}\n#Events: {3:d}'
                    .format(ampexp[0], ampexp[1], np.mean(self.cell_summary['intervals']), len(self.cell_summary['intervals'])))
        self.axIntvls.legend(fontsize=6)

        # report results
        print('   N events: {0:7d}'.format(nintvls))
        print('   Intervals: {0:7.1f} ms SD = {1:.1f} Frequency: {2:7.1f} Hz'.format(np.mean(self.cell_summary['intervals']),
                np.std(self.cell_summary['intervals']), 1e3/np.mean(self.cell_summary['intervals'])))
        print('    Amplitude: {0:7.1f} pA SD = {1:.1f}'.format(np.mean(self.cell_summary['amplitudes']), np.std(self.cell_summary['amplitudes'])))
    
        # test if interval distribtuion is poisson:
        stats = scipy.stats.kstest(expDis,'expon', args=((np.std(self.cell_summary['intervals'])),), alternative = 'two-sided')
        print('    KS test for intervals Exponential: statistic: {0:.5f}  p={1:g}'.format(stats.statistic, stats.pvalue))
        stats_amp = scipy.stats.kstest(expDis,'norm', 
                args=(np.mean(self.cell_summary['amplitudes']),
                      np.std(self.cell_summary['amplitudes'])), alternative = 'two-sided')
        print('    KS test for Normal amplitude: statistic: {0:.5f}  p={1:g}'.format(stats_amp.statistic, stats_amp.pvalue))


    def plot_setup(self):
        sizer = OrderedDict([('A', {'pos': [0.12, 0.8, 0.35, 0.60]}),
                       #  ('A1', {'pos': [0.52, 0.35, 0.35, 0.60]}),
                         ('B', {'pos': [0.12, 0.35, 0.08, 0.20]}),
                         ('C', {'pos': [0.60, 0.35, 0.08, 0.20]}),
                         ])  # dict elements are [left, width, bottom, height] for the axes in the plot.
        n_panels = len(sizer.keys())
        gr = [(a, a+1, 0, 1) for a in range(0, n_panels)]   # just generate subplots - shape does not matter
        axmap = OrderedDict(zip(sizer.keys(), gr))
        self.P = PH.Plotter((n_panels, 1), axmap=axmap, label=True, figsize=(7., 9.))
        self.P.resize(sizer)  # perform positioning magic
        hht = 3
        ax0 = self.P.axdict['A']
        ax0.set_ylabel('pA', fontsize=9)
        ax0.set_xlabel('T (ms)', fontsize=9)
        #self.axdec = P.axdict['A1']
        axIntvls = self.P.axdict['B']
        axIntvls.set_ylabel('Fraction of Events', fontsize=9)
        axIntvls.set_xlabel('Interevent Interval (ms)', fontsize=9)
        axIntvls.set_title('mEPSC Interval Distributon', fontsize=10)
        axAmps = self.P.axdict['C']
        axAmps.set_ylabel('Fraction of Events', fontsize=9)
        axAmps.set_xlabel('Event Amplitude (pA)', fontsize=9)
        axAmps.set_title('mEPSC Amplitude Distribution', fontsize=10)
        self.ax0 = ax0
        self.axIntvls = axIntvls
        self.axAmps = axAmps
        
        

if __name__ == '__main__':
    
    # example of how to use the analysis in conjunction with a data plan
    # usually this kind of code will be in a separate directory where the specific
    # runner and results for a given experiment are located.
    
    parser = argparse.ArgumentParser(description='mini synaptic event analysis')
    parser.add_argument('datadict', type=str,
                        help='data dictionary')
    parser.add_argument('-o', '--one', type=str, default='', dest='do_one',
                        help='just do one')
    parser.add_argument('-c', '--check', action='store_true',
                        help='Check for files; no analysis')
    parser.add_argument('-m', '--mode', type=str, default='aj', dest='mode',
                        choices=['aj', 'cb'],
                        help='just do one')
    parser.add_argument('-v', '--view', action='store_false',
                        help='Turn off pdf for single run')

    args = parser.parse_args()

    dataplan = EP.DataPlan.DataPlan(args.datadict)
    
    MI = MiniAnalysis(dataplan)
    
    if args.do_one == '': # no second argument, run all data sets
        print('doing all...', args.do_one)
        MI.analyze_all(fofilename='all_{0:s}_{1:s}_{2:s}.pdf'.format(args.datadict, 
            filterstring, args.mode), check=args.check, mode=args.mode)
    else:
        summarydata = {}
        try:
            filtered = dataplan.data['notch_and_hpf_filter']
            filterstring = 'filtered'
        except:
            filtered = False
            filterstring = 'nofilter'
        fout = 'summarydata_{0:s}_{1:s}_{2:s}.p'.format(args.do_one, filterstring, args.mode)
        
        if not args.view:
            fofilename = 'summarydata_{0:s}_{1:s}_{2:s}.pdf'.format(args.do_one, filterstring, args.mode)
            print('fofile: ', fofliename)
            with PdfPages(fofilename) as pdf:
                MI.analyze_one(args.do_one, pdf=fofilename, maxprot=10, check=args.check, mode=args.mode)
        else:
            MI.analyze_one(args.do_one, pdf=None, maxprot=10, check=args.check, mode=args.mode)
        
        #print('MI summary: ', MI.cell_summary)
        summarydata = {args.do_one: MI.cell_summary}
        fh = open(fout, 'wb')
        pickle.dump(summarydata, fh)
        fh.close()