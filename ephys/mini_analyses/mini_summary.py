"""
Analysis

"""
from __future__ import print_function
import sys
import os
#import read_protocol as rp
import ephysanalysis as EP
import minis_methods as minis
import numpy as np
import matplotlib.pyplot as mpl
from scipy.optimize import curve_fit
from collections import OrderedDict

import commands
computer_name = commands.getoutput('scutil --get ComputerName')
print('computer name: {0}'.format(computer_name))
if computer_name in ['Tamalpais']:
    basedir = ('/Volumes/PBM_004/data/NCAM-miniIPSCs')
elif computer_name in ['Lytle']:
    basedir = ('/Volumes/Pegasus/ManisLab_Data3/Sullivan_Chelsea/miniIPSCs')
elif computer_name in ['Tule']:
    basedir = ('/Users/experimenters/Data/Chelsea/CHL1/')
elif computer_name in ['Tamalpais2']:
    basedir = ('/Users/pbmanis/Documents/data/CHL1/')
else:
    raise ValueError('Computer name not in known list of names to set base path')



class Summary():
    def __init__(self):
        pass
        
    def doubleexp(self, x, A, tau_1, tau_2, dc):
        tm = A * ((1-(np.exp(-x/tau_1)))**4 * np.exp((-x/tau_2))) + dc
        return tm
    
    def fit_average_event(self, x, y):

        init_vals = [-20., 0.5, 5., 0.]
        best_vals, covar = curve_fit(self.doubleexp, x, y, p0=init_vals)
#        print ('best vals: ', best_vals)
        self.fitresult = best_vals
        self.best_fit = self.doubleexp(self.tbx, best_vals[0], best_vals[1], best_vals[2], best_vals[3])
        # lmfit version - fails for odd reason
        # dexpmodel = Model(self.doubleexp)
        # params = dexpmodel.make_params(A=-10., tau_1=0.5, tau_2=4.0, dc=0.)
        # self.fitresult = dexpmodel.fit(self.avgevent[tsel:], params, x=self.avgeventtb[tsel:])
        # print(self.fitresult.fit_report())
        self.best_vals = best_vals
        self.tau1 = best_vals[1]
        self.tau2 = best_vals[2]

    def do_one_protocol(self, ds, dprot, sign=-1, plots=False):
        fn = os.path.join(basedir, datasets[ds]['dir'], ('minis_{0:03d}'.format(datasets[ds]['prots'][dprot])))
        print('fn: ', fn)
        self.acq = EP.acq4read.Acq4Read(dataname='Clamp1.ma')
        self.acq.setProtocol(fn)
        self.acq.getData()
        data = self.acq.data_array*1e12
        time_base = self.acq.time_base
        dt = self.acq.sample_interval
        aj = minis.AndradeJonas()
        # try:
        #     data, time_base, dt = rp.readPhysProtocol(fn, records=None)
        # except:
        #     print("Incomplete protocol: {:s}".format(fn))
        #     return
        #     #raise ValueError('bad data')
        title = ('data: {0:s}   protocol #: {1:d}'.format(ds, dprot))
        print (title)

        dt = dt * 1000. # convert to msec
        time_base = time_base*1000.
        maxt = np.max(time_base)
        aj.setup(tau1=datasets[ds]['rt'], tau2=datasets[ds]['decay'], template_tmax=maxt, dt=dt, sign=sign)
        intv = []
        ampd = []
        events = []
        mwin = int(2*(4.)/dt)
        order = int(4/dt)
        
        for i in range(data.shape[0]):  # typically 10
            data[i] = data[i] - data[i].mean()
            # low pass and high pass the data
            # clip the data time window
            aj.deconvolve(data[i], thresh=datasets[ds]['thr'], llambda=20., order=7)
            ampd.extend(aj.amplitudes)
            intv.extend(aj.intervals)
            if i == 0: # get event shape
                print(aj.allevents.shape)
            events.extend(aj.allevents)
            title2 = ('data: {0:s}   protocol #: {1:d} trace: {2:d}'.format(ds, dprot, i))
            if i >= 0 and plots:
                aj.plots(title=title2)
    #    print len(ampd)
        if plots:
            f, ax = mpl.subplots(2, 1)
            f.suptitle(title)
            try:
                ax[0].hist(ampd, 50)
            except:
                pass
                # print('ampd: ', ampd)
            ax[1].hist(intv, 50)

        print("")
        print('Dataset: {:s}'.format(fn))
        print('    N events: {0:7d}'.format(len(intv)))
        print('    Intervals: {0:7.1f} ms SD = {1:.1f} Frequency: {2:7.1f} Hz'.format(np.nanmean(intv), np.nanstd(intv), 1e3/np.mean(intv)))
        print('    Amplitude: {0:7.1f} pA SD = {1:.1f}'.format(np.nanmean(ampd), np.nanstd(ampd)))
        # print('         tau1: {0:7.1f} pA SD = {1:.1f}'.format(np.nanmean(tau1), np.nanstd(tau1)))
       #  print('         tau2: {0:7.1f} pA SD = {1:.1f}'.format(np.nanmean(tau2), np.nanstd(tau2)))
       #   
        #return np.nanmean(intv), np.nanmean(ampd), np.nanmean(tau1), np.nanmean(tau2)
        self.delay = aj.delay
        self.intv = intv
        self.ampd = ampd
        self.events = np.array(events)
        self.aj = aj
        self.dt = dt
        #print('events: ', self.events.shape)
        
 
    def runall(self, sign=-1):
        gt_rate =  OrderedDict()
        gt_ampd = OrderedDict()
        gt_tau1 = OrderedDict()
        gt_tau2 = OrderedDict()
        gt_mouse = []
        for d in datasets.keys():  # by cell (not mouse)
            cintv = []
            campd = []
            ctau1 = 0.
            ctau2 = 0.
            n = 0
            g = datasets[d]['G']
#            cintv = 0.
            for i, dprot in enumerate(range(len(datasets[d]['prots']))):
                self.do_one_protocol(d, dprot, sign=sign, plots=False)
                
                cintv.extend(self.intv)
                campd.extend(self.ampd)
                n += 1
                if i == 0:
                    allev = self.events
                else:
                    allev = np.concatenate((allev, self.events))
#                print('allev shape: ', allev.shape)
#            print(cintv)
            # now get average event shape, and fit it
            avg = np.mean(allev, axis=0)
            t = np.arange(0, self.dt*avg.shape[0], self.dt)
            tsel = np.argwhere(t >= 6.0)[0][0]
            self.tbx = t[:-tsel]
            self.fit_average_event(t[:-tsel], avg[tsel:])            
            
            if g not in gt_rate.keys():
                gt_rate[g] = [1000./np.nanmean(cintv)]
            else:
                gt_rate[g].extend([1000./np.nanmean(cintv)])
            
            if g not in gt_ampd.keys():
                gt_ampd[g] = [np.nanmean(campd)]
            else:
                gt_ampd[g].extend([np.nanmean(campd)])
            
            if g not in gt_tau1.keys():
                gt_tau1[g] = [self.tau1]
            else:
                gt_tau1[g].extend([self.tau1])
            
            if g not in gt_tau2.keys():
                gt_tau2[g] = [self.tau2]
            else:
                gt_tau2[g].extend([self.tau2])
        

            if d not in gt_mouse:
                gt_mouse.append([d])

        print( gt_mouse)
        for k in gt_rate.keys():
            print('Genotype: {0:s}'.format(k))
            print('Rate: ', gt_rate[k])
        for k in gt_rate.keys():  # keeps same order.
            print('Genotype: {0:s}'.format(k))
            print('amps: ', gt_ampd[k])
        for k in gt_rate.keys():
            print('Genotype: {0:s}'.format(k))
            print('tau1: ', gt_tau1[k])
        for k in gt_rate.keys():
            print('Genotype: {0:s}'.format(k))
            print('tau2: ', gt_tau2[k])
        
    #mpl.show()


if __name__ == '__main__':
    import CS_CHL1_minis as cs
    basedir = cs.basepath
    datasets = cs.datasets
    #print( datasets)
    s = Summary()
    sign = -1 # pick negative going events
    if len(sys.argv) == 1:
        s.runall(sign=sign)
    else:
        ds = sys.argv[1]
        for i, dprot in enumerate(range(len(datasets[ds]['prots']))):
            s.do_one_protocol(ds, dprot, sign=sign, plots=True)
            if i == 0:
                allev = s.events
            else:
                allev = np.concatenate((allev, s.events))
        avg = np.mean(allev, axis=0)
        t = np.arange(0, s.dt*avg.shape[0], s.dt)
        tsel = np.argwhere(t >= 5.0 + s.delay)[0][0]
        s.tbx = t[:-tsel]
        s.fit_average_event(t[:-tsel], avg[tsel:])
        print('    Tau1: %7.3f  Tau2: %7.3f' % (s.best_vals[1], s.best_vals[2]))
#        s.best_fit = [-30., 0.5, 4., -3]
        mpl.plot(t,  avg, 'k')
        mpl.plot(t[tsel:], s.best_fit, 'r--')
#        mpl.plot(t[tsel:], s.doubleexp(t[:-tsel], s.best_fit[0] , s.best_fit[1], s.best_fit[2], s.best_fit[3]), 'r--')
        mpl.show()

          