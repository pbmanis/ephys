from __future__ import print_function

"""
Classes for methods that do analysis of miniature synaptic potentials

Current implementations are ClementsBekkers and AndradeJonas

Test run timing:
cb: 0.175 s (with cython version of algorithm); misses overlapping events
aj: 0.028 s, plus gets overlapping events

July 2017

Note: all values are MKS (Seconds, plus Volts, Amps)
per acq4 standards... 
"""
import numpy as np
import scipy.signal

import timeit
from scipy.optimize import curve_fit
from numba import jit
import lmfit
from . import digital_filters as dfilt
from . import functions as FN # Luke's misc. function library
import pyximport
pyximport.install()
from ephys.mini_analyses import clembek

# ANSI terminal colors  - just put in as part of the string to get color terminal output
colors = {'red': '\x1b[31m', 'yellow': '\x1b[33m', 'green': '\x1b[32m', 'magenta': '\x1b[35m',
              'blue': '\x1b[34m', 'cyan': '\x1b[36m' , 'white': '\x1b[0m', 'backgray': '\x1b[100m'}

class MiniAnalyses():
    def __init__(self):
        """
        Base class for Clements-Bekkers and Andrade-Jonas methods
        Provides template generation, and summary analyses
        Allows use of common methods between different algorithms
        """
        self.risepower = 4.
        self.min_event_amplitude = 5.0e-12 # pA default
        self.template = None
        pass

    def set_sign(self, sign):
        self.sign = sign
    
    def set_risepower(self, risepower):
        if risepower > 0 and risepower < 8:
            self.risepower = risepower

    def _make_template(self, taus=None):
        """
        Private function: make template when it is needed
        """
        if taus is None:
            tau_1,  tau_2 = self.taus  # use the predefined taus
        else:
            tau_1,  tau_2 = taus
            
        t_psc = np.arange(0,  self.template_tmax,  self.dt)
        Aprime = (tau_2/tau_1)**(tau_1/(tau_1-tau_2))
        self.template = np.zeros_like(t_psc)
        tm = 1./Aprime * ((1-(np.exp(-t_psc/tau_1)))**self.risepower * np.exp((-t_psc/tau_2)))
       # tm = 1./2. * (np.exp(-t_psc/tau_1) - np.exp(-t_psc/tau_2))
        if self.idelay > 0:
            self.template[self.idelay:] = tm[:-self.idelay]  # shift the template
        else:
            self.template = tm
        if self.sign > 0:
            self.template_amax = np.max(self.template)
        else:
            self.template = -self.template
            self.template_amax = np.min(self.template)    

    def summarize(self, data, order=11, verbose=False):
        """
        compute intervals,  peaks and ampitudes for all found events in a trace
        """
        self.intervals = np.diff(self.timebase[self.onsets])  # event intervals
        i_decay_pts = int(2*self.taus[1]/self.dt)  # decay window time (points)
        self.peaks = []
        self.smpkindex = []
        self.smoothed_peaks = []
        self.amplitudes = []
        self.Qtotal = []
        self.averaged = False  # set flags in case of no events found
        self.individual_events = False
        self.fitted = False
        self.fitted_tau1 = np.nan
        self.fitted_tau2 = np.nan
        self.Amplitude = np.nan
        self.avg_fiterr = np.nan
        ndata = len(data)
        avgwin = 5 # int(1.0/self.dt)  # 5 point moving average window for peak detection
#        print('dt: ', self.dt)
        mwin = int((0.050)/self.dt)
#        print('mwin: ', mwin)
        #order = int(0.0004/self.dt)
   #     print('onsets: ', self.onsets)
        if self.sign > 0:
            nparg = np.greater
        else:
            nparg = np.less
        if len(self.onsets) > 0:  # original events
#            print('no: ', len(self.onsets))
            acceptlist = []
            for j in range(len(data[self.onsets])):
                if self.sign > 0 and self.eventstartthr is not None:
                    if self.data[self.onsets[j]] < self.eventstartthr:
                        continue
                if self.sign < 0 and self.eventstartthr is not None:
                    if self.data[self.onsets[j]] > -self.eventstartthr:
                        continue
                svwinlen = data[self.onsets[j]:(self.onsets[j]+mwin)].shape[0]
                if svwinlen > 11:
                    svn = 11
                else:
                    svn = svwinlen
                if svn % 2 == 0:  # if even, decrease by 1 point to meet ood requirement for savgol_filter
                    svn -=1
                    
                if svn > 3:  # go ahead and filter
                    p =  scipy.signal.argrelextrema(scipy.signal.savgol_filter(data[self.onsets[j]:(self.onsets[j]+mwin)], svn, 2), nparg, order=order)[0]
                else: # skip filtering
                    p =  scipy.signal.argrelextrema(data[self.onsets[j]:(self.onsets[j]+mwin)], nparg, order=order)[0]
                if len(p) > 0:
                    self.peaks.extend([int(p[0]+self.onsets[j])])
                    amp = self.sign*(self.data[self.peaks[-1]] - data[self.onsets[j]])

                    self.amplitudes.extend([amp])
                    i_end = i_decay_pts + self.onsets[j] # distance from peak to end
                    i_end = min(ndata, i_end)  # keep within the array limits
                    if j < len(self.onsets)-1:
                        if i_end > self.onsets[j+1]:
                            i_end = self.onsets[j+1]-1 # only go to next event start
                    move_avg, n = moving_average(data[self.onsets[j]:i_end], n=min(avgwin, len(data[self.onsets[j]:i_end])))
                    if self.sign > 0:
                        pk = np.argmax(move_avg) # find peak of smoothed data
                    else:
                        pk = np.argmin(move_avg)
                    self.smoothed_peaks.extend([move_avg[pk]])  # smoothed peak
                    self.smpkindex.extend([self.onsets[j]+pk])
                    acceptlist.append(j)
            if len(acceptlist) < len(self.onsets):
                if verbose:
                    print('Trimmed %d events' % (len(self.onsets)-len(acceptlist)))
                self.onsets = self.onsets[acceptlist] # trim to only the accepted values
           # print(self.onsets)
            self.avgevent, self.avgeventtb, self.allevents = self.average_events(self.onsets) 
            if self.averaged:
                self.fit_average_event(self.avgeventtb, self.avgevent, debug=False)
            
        else:
            if verbose:
                print('No events found')
            return
    
    def measure_events(self, eventlist):
        # compute simple measurements of events (area, amplitude, half-width)
        #
        self.measured = False
        # treat like averaging
        tdur = np.max((np.max(self.taus)*5.0, 0.010))  # go 5 taus or 10 ms past event
        tpre = 0. # self.taus[0]*10.
        self.avgeventdur = tdur
        self.tpre = tpre
        self.avgnpts = int((tpre+tdur)/self.dt)  # points for the average
        npre = int(tpre/self.dt) # points for the pre time
        npost = int(tdur/self.dt)
        avg = np.zeros(self.avgnpts)
        avgeventtb = np.arange(self.avgnpts)*self.dt

        allevents = np.zeros((len(eventlist),  self.avgnpts))
        k = 0
        pkt = 0 # np.argmax(self.template)  # accumulate
        meas = {'Q': [], 'A': [], 'HWup': [], 'HWdown': [], 'HW': []}
        for j, i in enumerate(eventlist):
            ix = i + pkt # self.idelay
            if (ix + npost) < len(self.data) and (ix - npre) >= 0:
                allevents[k,:] = self.data[ix-npre:ix+npost]
                k = k + 1
        if k > 0:
            allevents = allevents[0:k, :]  # trim unused
            for j in range(k):
                ev_j = scipy.signal.savgol_filter(self.sign*allevents[j, :], 7, 2, mode='nearest')  # flip sign if negative
                ai = np.argmax(ev_j)
                if ai == 0:
                    continue  # skip events where max is first point
                q = np.sum(ev_j)*tdur
                meas['Q'].append(q)
                meas['A'].append(ev_j[ai])
                hw_up = self.dt*np.argmin(np.fabs((ev_j[ai]/2.0) - ev_j[:ai]))
                hw_down = self.dt*np.argmin(np.fabs(ev_j[ai:] - (ev_j[ai]/2.0)))
                meas['HWup'].append(hw_up)
                meas['HWdown'].append(hw_down)
                meas['HW'].append(hw_up+hw_down)
            self.measured = True
        else:
            self.measured = False
        return(meas)

    def average_events(self, eventlist):
        # compute average event with length of template
        self.averaged = False
        tdur = np.max((np.max(self.taus)*5.0, 0.010))  # go 5 taus or 10 ms past event
        tpre = 0. # self.taus[0]*10.
        self.avgeventdur = tdur
        self.tpre = tpre
        self.avgnpts = int((tpre+tdur)/self.dt)  # points for the average
        npre = int(tpre/self.dt) # points for the pre time
        npost = int(tdur/self.dt)
        avg = np.zeros(self.avgnpts)
        avgeventtb = np.arange(self.avgnpts)*self.dt

        allevents = np.zeros((len(eventlist),  self.avgnpts))
        k = 0
        pkt = 0 # np.argmax(self.template)
        for j, i in enumerate(eventlist):
            ix = i + pkt # self.idelay
            if (ix + npost) < len(self.data) and (ix - npre) >= 0:
                allevents[k,:] = self.data[(ix-npre):(ix+npost)]
                k = k + 1
        if k > 0:
            allevents = allevents[0:k, :]  # trim unused
            avgevent = allevents.mean(axis=0)
            avgevent = avgevent-np.mean(avgevent[:3])
            self.averaged = True
        else:
            avgevent = []
            allevents = []
            self.averaged = False
        return(avgevent, avgeventtb, allevents)

    def doubleexp(self, p, x, y, risepower, fixed_delay=0., mode=0):
        """
        Calculate a double expoential EPSC-like waveform with the rise to a power
        to make it sigmoidal
        """
        #fixed_delay = p[3]  # allow to adjust; ignore input value
        ix = np.argmin(np.fabs(x-fixed_delay))
        tm = np.zeros_like(x)
        tm[ix:] = p[0] * (1.0 - np.exp(-(x[ix:]-fixed_delay)/p[1]))**risepower
        tm[ix:] *= np.exp(-(x[ix:]-fixed_delay)/p[2])

        if mode == 0:
            return tm - y
        elif mode == 1:
            return np.linalg.norm(tm-y)
        elif mode == -1:
            return tm
        else:
            raise ValueError('doubleexp: Mode must be 0 (diff), 1 (linalg.norm) or -1 (just value)')

    def risefit(self, p, x, y, risepower, mode=0):
        """
        Calculate a delayed EPSC-like waveform rise shape with the rise to a power
        to make it sigmoidal, and an adjustable delay
        input data should only be the rising phase.
        p is in order: [amplitude, tau, delay]
        """
        assert mode in [-1, 0, 1]
        ix = np.argmin(np.fabs(x-p[2]))
        tm = np.zeros_like(x)
        expf = (x[ix:]-p[2])/p[1]
        pclip = 1.e3
        nclip = 0.
        expf[expf>pclip]= pclip
        expf[expf<-nclip] = -nclip
        tm[ix:] = p[0] * (1.0 - np.exp(-expf))**risepower
        if mode == 0:
            return tm - y
        elif mode == 1:
            return np.linalg.norm(tm-y)
        elif mode == -1:
            return tm
        else:
            raise ValueError('doubleexp: Mode must be 0 (diff), 1 (linalg.norm) or -1 (just value)')

    def decayexp(self, p, x, y, fixed_delay=0., mode=0):
        """
        Calculate an exponential decay (falling phase fit)
        """
        tm = p[0] * np.exp(-(x-fixed_delay)/p[1])
        if mode == 0:
            return tm - y
        elif mode == 1:
            return np.linalg.norm(tm-y)
        elif mode == -1:
            return tm
        else:
            raise ValueError('doubleexp: Mode must be 0 (diff), 1 (linalg.norm) or -1 (just value)')
            
    def fit_average_event(self, tb, average_event, debug=False, label='', 
                          inittaus=[0.001, 0.005], initdelay=None):
        """
        Fit the averaged event to a double exponential epsc-like function
        """
        #tsel = np.argwhere(self.avgeventtb > self.tpre)[0]  # only fit data in event,  not baseline
        tsel = 0  # use whole averaged trace
        self.tsel = tsel
        self.tau1 = inittaus[0]
        self.tau2 = inittaus[1]
        self.tau2_range = 10.
        self.tau1_minimum_factor = 2.
        time_past_peak = 2.5e-4
        self.fitted_tau1 = np.nan
        self.fitted_tau2 = np.nan
        self.Amplitude = np.nan
        # peak_pos = np.argmax(self.sign*self.avgevent[self.tsel:])
        # decay_fit_start = peak_pos + int(time_past_peak/self.dt)
        # init_vals = [self.sign*10.,  1.0,  4., 0.]
        # init_vals_exp = [20.,  5.0]
        # bounds_exp  = [(0., 0.5), (10000., 50.)]
        
        res, rdelay = self.event_fitter(tb, average_event, time_past_peak=time_past_peak, initdelay=initdelay,  debug=debug, label=label)
        # print('rdelay: ', rdelay)
        self.fitresult = res.x
        self.Amplitude = self.fitresult[0]
        self.fitted_tau1 = self.fitresult[1]
        self.fitted_tau2 = self.fitresult[2]
        self.bfdelay = rdelay
        self.DC = 0. # best_vals[3]
        self.avg_best_fit = self.doubleexp(self.fitresult, tb[self.tsel:],
            np.zeros_like(tb[self.tsel:]), risepower=self.risepower, mode=0, fixed_delay=self.bfdelay)
        self.avg_best_fit = self.sign*self.avg_best_fit
        fiterr = np.linalg.norm(self.avg_best_fit-average_event[self.tsel:])
        self.avg_fiterr = fiterr
        # print("averaged data fit, Error = ", self.avg_fiterr)
        ave = self.sign*average_event
        ipk = np.argmax(ave)
        pk = ave[ipk]
        p10 = 0.1*pk
        p90 = 0.9*pk
        p37 = 0.37*pk
        try:
            i10 = np.argmin(np.fabs(ave[:ipk]-p10))
        except:
            self.fitted = False
            return
        i90 = np.argmin(np.fabs(ave[:ipk]-p90))
        i37 = np.argmin(np.fabs(ave[ipk:]-p37))
        self.risetenninety = self.dt*(i90-i10)
        self.decaythirtyseven = self.dt*(i37-ipk)
        self.Qtotal = self.dt*np.sum(average_event[self.tsel:])
        self.fitted = True

    def fit_individual_events(self, onsets):
        """
        Fitting individual events
        Events to be fit are selected from the entire event pool as:
        1. events that are completely within the trace, AND
        2. events that do not overlap other events
        
        Fit events are further classified according to the fit error
        
        """
        if not self.averaged or not self.fitted:  # averaging should be done first: stores events for convenience and gives some tau estimates
            print('Require fit of averaged events prior to fitting individual events')
            raise(ValueError)
        time_past_peak = 0.75 # msec - time after peak to start fitting
        
        # allocate arrays for results. Arrays have space for ALL events
        # okevents, notok, and evok are indices
        nevents = len(self.allevents)  # onsets.shape[0]
        self.ev_fitamp = np.zeros(nevents)  # measured peak amplitude from the fit
        self.ev_A_fitamp = np.zeros(nevents)  # fit amplitude - raw value can be quite different than true amplitude.....
        self.ev_tau1 = np.zeros(nevents)
        self.ev_tau2 = np.zeros(nevents)
        self.ev_1090 = np.zeros(nevents)
        self.ev_2080 = np.zeros(nevents)
        self.ev_amp = np.zeros(nevents)  # measured peak amplitude from the event itself
        self.ev_Qtotal = np.zeros(nevents)  # measured charge of the event (integral of current * dt)
        self.fiterr = np.zeros(nevents)
        self.bfdelay = np.zeros(nevents)
        self.best_fit = np.zeros((nevents, self.avgeventtb.shape[0]))
        self.best_decay_fit = np.zeros((nevents, self.avgeventtb.shape[0]))
        self.tsel = 0
        self.tau2_range = 10.
        self.tau1_minimum_factor = 5.

        # prescreen events
        minint = self.avgeventdur # msec minimum interval between events.
        self.fitted_events = []  # events that can be used (may not be all events, but these are the events that were fit)
        for i in range(nevents):
            te = self.timebase[onsets[i] ] # get current event
            try:
                tn = self.timebase[onsets[i+1]]  # check time to next event
                if tn-te < minint:  # event is followed by too soon by another event
                    continue
            except:
                    pass  # just handle trace end condition
            try:
                tp = self.timebase[onsets[i-1] ] # check previous event
                if te-tp < minint:  # if current event too close to a previous event, skip
                    continue
                self.fitted_events.append(i)  # passes test, include in ok events
            except:
                pass

        for n, i in enumerate(self.fitted_events):
            try:
                max_event = np.max(self.sign*self.allevents[i,:])
            except:
                print("minis_methods eventfitter")
                print('fitted: ', self.fitted_events)
                print('i: ', i)
                print('allev: ', self.allevents)
                print('len allev: ', len(self.allevents), onsets.shape[0])
                raise
            res, rdelay = self.event_fitter(self.avgeventtb, self.allevents[i,:], time_past_peak=time_past_peak)
            self.fitresult = res.x

            # lmfit version - fails for odd reason
            # dexpmodel = Model(self.doubleexp)
            # params = dexpmodel.make_params(A=-10.,  tau_1=0.5,  tau_2=4.0,  dc=0.)
            # self.fitresult = dexpmodel.fit(self.avgevent[tsel:],  params,  x=self.avgeventtb[tsel:])
            self.ev_A_fitamp[i] = self.fitresult[0]
            self.ev_tau1[i] = self.fitresult[1]
            self.ev_tau2[i] = self.fitresult[2]
            self.bfdelay[i] = rdelay
            self.fiterr[i] = self.doubleexp(self.fitresult, self.avgeventtb, self.sign*self.allevents[i,:], 
                                                risepower=self.risepower, fixed_delay=self.bfdelay[i], mode=1)
            self.best_fit[i] = self.doubleexp(self.fitresult, self.avgeventtb,
                                                np.zeros_like(self.avgeventtb), 
                                                risepower=self.risepower, fixed_delay=self.bfdelay[i], mode=0)
            self.best_decay_fit[i] = self.decay_fit  # from event_fitter
            self.ev_fitamp[i] = np.max(self.best_fit[i])
            self.ev_Qtotal[i] = self.dt*np.sum(self.sign*self.allevents[i,:])
            self.ev_amp[i] = np.max(self.sign*self.allevents[i,:])
        self.individual_event_screen(fit_err_limit=2000., tau2_range=10.)
        self.individual_events = True  # we did this step

    def event_fitter(self, timebase, event, time_past_peak=0.0001, initdelay=None, debug=False, label=''):
        """
        Fit the event
        Procedure:
        First we fit the rising phase (to the peak) with (1-exp(t)^n), allowing
        the onset of the function to slide in time. This onset time is locked after this step
        to minimize trading in the error surface between the onset and the tau values.
        Second, we fit the decay phase, starting just past the peak (and accouting for the fixed delay)
        Finally, we combine the parameters and do a final optimization with somewhat narrow
        bounds.
        Fits are good on noiseless test data. 
        Fits are affected by noise on the events (of course), but there is no "systematic"
        variation that is present in terms of rise-fall tau tradeoffs.
        
        """
        debug=False
        try:
            dt = self.dt
        except:
            dt = np.mean(np.diff(timebase))
            self.dt = dt
        
        ev_bl = np.mean(event[:int(dt/dt)])   # just first point... 
        evfit = self.sign*(event - ev_bl)
        maxev = np.max(evfit)
        if maxev == 0:
            maxev = 1
        # if peak_pos == 0:
        #     peak_pos = int(0.001/self.dt) # move to 1 msec later
        evfit = evfit/maxev # scale to max of 1
        peak_pos = np.argmax(evfit)+1
        amp_bounds = [0., 1.]
        # set reasonable, but wide bounds, and make sure init values are within bounds
        # (and off center, but not at extremes)
        
        bounds_rise = [amp_bounds, (dt, 4.*dt*peak_pos), (0., 0.005)]
        if initdelay is None or initdelay<dt:
            fdelay = 0.2*np.mean(bounds_rise[2])
        else:
            fdelay = initdelay
        if fdelay > dt*peak_pos:
            fdelay = 0.2*dt*peak_pos
        init_vals_rise = [0.9, dt*peak_pos, fdelay]
        
        res_rise = scipy.optimize.minimize(self.risefit, 
                        init_vals_rise, bounds=bounds_rise,
                        method=  'SLSQP',  # x_scale=[1e-12, 1e-3, 1e-3],
                        args=(timebase[:peak_pos], # x
                              evfit[:peak_pos], # 'y
                              self.risepower, 1)  # risepower, mode
                        )
        if debug:
            import matplotlib.pyplot as mpl
            f, ax = mpl.subplots(2, 1)
            ax[0].plot(timebase,
                            evfit, '-k')
            ax[1].plot(timebase[:peak_pos],
                            evfit[:peak_pos], '-k')
            print('\nrise fit:')
            print('dt: ', dt, ' maxev: ', maxev, ' peak_pos: ', peak_pos)
            print('bounds: ', bounds_rise)
            print('init values: ', init_vals_rise)
            print('result: ', res_rise.x)
            rise_tb = timebase[:peak_pos]
            rise_yfit = self.risefit(res_rise.x, rise_tb, np.zeros_like(rise_tb), self.risepower, -1)
            ax[0].plot(rise_tb, rise_yfit, 'r-')
            ax[1].plot(rise_tb, rise_yfit, 'r-')
            # mpl.show()

        
        self.res_rise = res_rise
        # fit decay exponential next:
        bounds_decay  = [amp_bounds, (dt, self.tau2*20.)] # be sure init values are inside bounds
        init_vals_decay = [0.9*np.mean(amp_bounds),  self.tau2]
        # print('peak, tpast, tdel',  peak_pos , int(time_past_peak/self.dt) , int(res_rise.x[2]/self.dt))
        decay_fit_start = peak_pos + int(time_past_peak/self.dt) #+ int(res_rise.x[2]/self.dt)
        # print('decay start: ', decay_fit_start, decay_fit_start*self.dt, len(event[decay_fit_start:]))

        
        res_decay = scipy.optimize.minimize(self.decayexp, init_vals_decay,
                         bounds=bounds_decay, 
                         method=  'L-BFGS-B', 
                      #  bounds=bounds_decay, method='L-BFGS-B', 
                        args=(timebase[decay_fit_start:]-decay_fit_start*dt, 
                        evfit[decay_fit_start:], res_rise.x[2], 1)) # res_rise.x[2], 1))
        self.res_decay = res_decay

        if debug:
            decay_tb = timebase[decay_fit_start:]
            decay_ev = evfit[decay_fit_start:]
            # f, ax = mpl.subplots(2, 1)
            # ax[0].plot(timebase, evfit)
            ax[1].plot(decay_tb, decay_ev, 'g-')
            print('\ndecay fit:')
            print('dt: ', dt, ' maxev: ', maxev, ' peak_pos: ', peak_pos)
            print('bounds: ', bounds_decay)
            print('init values: ', init_vals_decay)
            print('result: ', res_decay.x)
            y = self.decayexp(res_decay.x, decay_tb, np.zeros_like(decay_tb), fixed_delay=decay_fit_start*dt, mode=-1)
            # print(y)
            # ax[1].plot(decay_tb, y, 'bo', markersize=3)
            ax[1].plot(decay_tb, y, 'g-')
        
        # now tune by fitting the whole trace, allowing some (but not too much) flexibility
        bounds_full  = [ [a*10. for a in amp_bounds], # overall amplitude
                        (0.2*res_rise.x[1],  5.*res_rise.x[1]),  # rise tau
                        (0.2*res_decay.x[1], 50.*res_decay.x[1]),  # decay tau
                        (0.3*res_rise.x[2], 20.*res_rise.x[2]),  # delay
                        #(0, 1), # amplitude of decay component
                    ]
        init_vals = [amp_bounds[1],  res_rise.x[1], res_decay.x[1], res_rise.x[2]]  # be sure init values are inside bounds
        # if len(label) > 0:
        #     print('Label: ', label)
        #     print('bounds full: ', bounds_full)
        #     print('init_vals: ', init_vals)
        res = scipy.optimize.minimize(self.doubleexp, init_vals, 
                         method=  'L-BFGS-B', 
                        args=(timebase, evfit,
                        self.risepower, res_rise.x[2], 1),
                        bounds=bounds_full, options={'maxiter': 100000},
                        )
        if debug:
            print('\nFull fit:')
            print('dt: ', dt, ' maxev: ', maxev, ' peak_pos: ', peak_pos)
            print('bounds: ', bounds_full)
            print('init values: ', init_vals)
            print('result: ', res.x, res_rise.x[2])
            f, ax = mpl.subplots(2, 1)
            ax[0].plot(timebase, evfit, 'k-')
            ax[1].plot(timebase, evfit, 'k-')
            y = self.doubleexp(res.x, timebase, event,
                    risepower=self.risepower, fixed_delay=res_rise.x[2], mode=-1)
            ax[1].plot(timebase, y, 'bo', markersize=3)
            mpl.show()


        self.rise_fit = self.risefit(res_rise.x, timebase, np.zeros_like(timebase), self.risepower, mode=0)
        self.rise_fit[peak_pos:] = 0
        self.rise_fit = self.rise_fit*maxev
        
        self.decay_fit = self.decayexp(self.res_decay.x, timebase,
                                np.zeros_like(timebase), fixed_delay=self.res_rise.x[2], mode=0)
        self.decay_fit[:decay_fit_start] = 0  # clip the initial part
        self.decay_fit = self.decay_fit*maxev
        
        self.bferr = self.doubleexp(res.x, timebase, event,
                    risepower=self.risepower, fixed_delay=decay_fit_start*dt, mode=1)
       # print('fit result: ', res.x, res_rise.x[2])
        res.x[0] = res.x[0]*maxev  # correct for factor
        self.peak_val = maxev
        return res, res_rise.x[2]

    def individual_event_screen(self, fit_err_limit=2000., tau2_range=2.5):
        """
        Screen events:
        error of he fit must be less than a limit,
        and
        tau2 must fall within a range of the default tau2
        and
        tau1 must be breater than a minimum tau1
        sets:
        self.events_ok : the list of fitted events that pass
        self.events_notok : the list of fitted events that did not pass
        """
        self.events_ok = []
        for i in self.fitted_events:  # these are the events that were fit
            if self.fiterr[i] <= fit_err_limit:
                if  self.ev_tau2[i] <= self.tau2_range*self.tau2:
                    if self.ev_fitamp[i] > self.min_event_amplitude:
                        if self.ev_tau1[i] > self.tau1/self.tau1_minimum_factor:
                            self.events_ok.append(i)
        self.events_notok = list(set(self.fitted_events).difference(self.events_ok))

    def plot_individual_events(self, fit_err_limit=1000., tau2_range=2.5, show=True):
        if not self.individual_events:
            raise
        P = PH.regular_grid(3 , 3, order='columns', figsize=(8., 8.), showgrid=False,
                        verticalspacing=0.1, horizontalspacing=0.12,
                        margins={'leftmargin': 0.12, 'rightmargin': 0.12, 'topmargin': 0.03, 'bottommargin': 0.1},
                        labelposition=(-0.12, 0.95))
#        evok, notok = self.individual_event_screen(fit_err_limit=fit_err_limit, tau2_range=tau2_range)
        evok = self.events_ok
        notok = self.events_notok
        
        P.axdict['A'].plot(self.ev_tau1[evok], self.ev_amp[evok], 'ko', markersize=4)
        P.axdict['A'].set_xlabel(r'$tau_1$ (ms)')
        P.axdict['A'].set_ylabel(r'Amp (pA)')
        P.axdict['B'].plot(self.ev_tau2[evok], self.ev_amp[evok], 'ko', markersize=4)
        P.axdict['B'].set_xlabel(r'$tau_2$ (ms)')
        P.axdict['B'].set_ylabel(r'Amp (pA)')
        P.axdict['C'].plot(self.ev_tau1[evok], self.ev_tau2[evok], 'ko', markersize=4)
        P.axdict['C'].set_xlabel(r'$\tau_1$ (ms)')
        P.axdict['C'].set_ylabel(r'$\tau_2$ (ms)')
        P.axdict['D'].plot(self.ev_amp[evok], self.fiterr[evok], 'ko', markersize=3)
        P.axdict['D'].plot(self.ev_amp[notok], self.fiterr[notok], 'ro', markersize=3)
        P.axdict['D'].set_xlabel(r'Amp (pA)')
        P.axdict['D'].set_ylabel(r'Fit Error (cost)')
        for i in notok:
            ev_bl = np.mean(self.allevents[i, 0:5])
            P.axdict['E'].plot(self.avgeventtb, self.allevents[i]-ev_bl, 'b-', linewidth=0.75)
                # P.axdict['E'].plot()
            P.axdict['F'].plot(self.avgeventtb, self.allevents[i]-ev_bl, 'r-', linewidth=0.75)
        P2= PH.regular_grid(1, 1, order='columns', figsize=(8., 8.), showgrid=False,
                        verticalspacing=0.1, horizontalspacing=0.12,
                        margins={'leftmargin': 0.12, 'rightmargin': 0.12, 'topmargin': 0.03, 'bottommargin': 0.1},
                        labelposition=(-0.12, 0.95))
        P3 = PH.regular_grid(1 , 5, order='columns', figsize=(12, 8.), showgrid=False,
                        verticalspacing=0.1, horizontalspacing=0.12,
                        margins={'leftmargin': 0.12, 'rightmargin': 0.12, 'topmargin': 0.03, 'bottommargin': 0.1},
                        labelposition=(-0.12, 0.95))        
        idx = [a for a in P3.axdict.keys()]
        ncol = 5
        offset2 = 0.
        k = 0
        for i in evok:
          #  print(self.ev_tau1, self.ev_tau2)
            offset = i*3.0
            ev_bl = np.mean(self.allevents[i, 0:5])
            P2.axdict['A'].plot(self.avgeventtb, self.allevents[i]+offset-ev_bl, 'k-', linewidth=0.35)
            # p = [self.ev_amp[i], self.ev_tau1[i],self.ev_tau2[i]]
            # x = self.avgeventtb
            #y = self.doubleexp(p, x, np.zeros_like(x), self.risepower, mode=-1)
            #y = p[0] * (((np.exp(-x/p[1]))) - np.exp(-x/p[2]))
            P2.axdict['A'].plot(self.avgeventtb, self.sign*self.best_fit[i]+offset, 'c--', linewidth=0.3)
            P2.axdict['A'].plot(self.avgeventtb, self.sign*self.best_decay_fit[i]+offset, 'r--', linewidth=0.3)
            P3.axdict[idx[k]].plot(self.avgeventtb, self.allevents[i]+offset2, 'k--', linewidth=0.3)
            P3.axdict[idx[k]].plot(self.avgeventtb, self.sign*self.best_fit[i]+offset2, 'r--', linewidth=0.3)
            if k == 4:
                k = 0
                offset2 += 10.
            else:
                k += 1
        
         
        if show:
            mpl.show()

    def plots(self,  events=None,  title=None):
        """
        Plot the results from the analysis and the fitting
        """
        data = self.data
        P = PH.regular_grid(3 , 1, order='columnsfirst', figsize=(8., 6), showgrid=False,
                        verticalspacing=0.08, horizontalspacing=0.08,
                        margins={'leftmargin': 0.07, 'rightmargin': 0.20, 'topmargin': 0.03, 'bottommargin': 0.1},
                        labelposition=(-0.12, 0.95))
        scf = 1e12
        ax = P.axarr
        ax = ax.ravel()
        PH.nice_plot(ax)
        for i in range(1,2):
            ax[i].get_shared_x_axes().join(ax[i],  ax[0])
        # raw traces, marked with onsets and peaks
        tb = self.timebase[:len(data)]
        ax[0].plot(tb,  scf*data,  'k-',  linewidth=0.75, label='Data')  # original data
        ax[0].plot(tb[self.onsets],  scf*data[self.onsets],  'k^',  
                        markersize=6,  markerfacecolor=(1,  1,  0,  0.8),  label='Onsets')
        if len(self.onsets) is not None:
#            ax[0].plot(tb[events],  data[events],  'go',  markersize=5, label='Events')
#        ax[0].plot(tb[self.peaks],  self.data[self.peaks],  'r^', label=)
            ax[0].plot(tb[self.smpkindex],  scf*np.array(self.smoothed_peaks),  'r^', label='Smoothed Peaks')
        ax[0].set_ylabel('I (pA)')
        ax[0].set_xlabel('T (s)')
        ax[0].legend(fontsize=8, loc=2, bbox_to_anchor=(1.0, 1.0))
        
        # deconvolution trace, peaks marked (using onsets), plus threshold)
        ax[1].plot(tb[:self.Crit.shape[0]],  self.Crit, label='Deconvolution') 
        ax[1].plot([tb[0],tb[-1]],  [self.sdthr,  self.sdthr],  'r--',  linewidth=0.75, 
                label='Threshold ({0:4.2f}) SD'.format(self.sdthr))
        ax[1].plot(tb[self.onsets]-self.idelay,  self.Crit[self.onsets],  'y^', label='Deconv. Peaks')
        if events is not None:  # original events
            ax[1].plot(tb[:self.Crit.shape[0]][events],  self.Crit[events],
                    'ro',  markersize=5.)
        ax[1].set_ylabel('Deconvolution')
        ax[1].set_xlabel('T (s)')
        ax[1].legend(fontsize=8, loc=2, bbox_to_anchor=(1.0, 1.0))
#        print (self.dt, self.template_tmax, len(self.template))
        # averaged events, convolution template, and fit
        if self.averaged:
            ax[2].plot(self.avgeventtb[:len(self.avgevent)],  scf*self.avgevent, 'k', label='Average Event')
            maxa = np.max(self.sign*self.avgevent)
            #tpkmax = np.argmax(self.sign*self.template)
            if self.template is not None:
                maxl = int(np.min([len(self.template), len(self.avgeventtb)]))
                temp_tb = np.arange(0, maxl*self.dt, self.dt)
                #print(len(self.avgeventtb[:len(self.template)]), len(self.template))
                ax[2].plot(self.avgeventtb[:maxl],  scf*self.sign*self.template[:maxl]*maxa/self.template_amax,  
                    'r-', label='Template')
            # compute double exp based on rise and decay alone
            # print('res rise: ', self.res_rise)
            # p = [self.res_rise.x[0], self.res_rise.x[1], self.res_decay.x[1], self.res_rise.x[2]]
            # x = self.avgeventtb[:len(self.avg_best_fit)]
            # y = self.doubleexp(p, x, np.zeros_like(x), risepower=4, fixed_delay=0, mode=0)
            # ax[2].plot(x, y, 'b--', linewidth=1.5)
            tau1 = np.power(10, (1./self.risepower)*np.log10(self.tau1*1e3))  # correct for rise power
            tau2 = self.tau2*1e3
            ax[2].plot(self.avgeventtb[:len(self.avg_best_fit)],  scf*self.avg_best_fit,  'c--', linewidth=2.0,
                label='Best Fit:\nRise Power={0:.2f}\nTau1={1:.3f} ms\nTau2={2:.3f} ms\ndelay: {3:.3f} ms'.
                        format(self.risepower, self.res_rise.x[1]*1e3, self.res_decay.x[1]*1e3, self.bfdelay*1e3))
           # ax[2].plot(self.avgeventtb[:len(self.decay_fit)],  self.sign*scf*self.rise_fit,  'g--', linewidth=1.0,
           #     label='Rise tau  {0:.2f} ms'.format(self.res_rise.x[1]*1e3))
           # ax[2].plot(self.avgeventtb[:len(self.decay_fit)],  self.sign*scf*self.decay_fit,  'm--', linewidth=1.0,
           #     label='Decay tau {0:.2f} ms'.format(self.res_decay.x[1]*1e3))
            if title is not None:
                P.figure_handle.suptitle(title)
            ax[2].set_ylabel('Averaged I (pA)')
            ax[2].set_xlabel('T (s)')
            ax[2].legend(fontsize=8, loc=2, bbox_to_anchor=(1.0, 1.0))
        if self.fitted:
            print('measures: ', self.risetenninety, self.decaythirtyseven)
        mpl.show()
        

@jit(nopython=True,  cache=True)
def nb_clementsbekkers(data,  template):
    """
    cb algorithm for numba jit.
    """
    ## Prepare a bunch of arrays we'll need later
    n_template = len(template)
    # if n_template <= 1:
    #     raise ValueError("nb_clementsbekkers: Length of template must be useful, and > 1")
    n_data = data.shape[0]
    n_dt = n_data - n_template
    # if n_dt < 10:
    #     raise ValueError("nb_clementsbekkers: n_dt, n_template", n_dt, n_template)
    #
    sum_template = template.sum()
    sum_template_2 = (template*template).sum()

    data_2 = data*data
    sum_data = np.sum(data[:n_template])
    sum_data_2 = data_2[:n_template].sum()
    scale = np.zeros(n_dt)
    offset = np.zeros(n_dt)
    crit = np.zeros(n_dt)
    for i in range(n_dt):
        if i > 0:
            sum_data = sum_data + data[i+n_template] - data[i-1]
            sum_data_2 = sum_data_2 + data_2[i+n_template] - data_2[i-1]
        sum_data_template_prod = np.multiply(data[i:i+n_template],  template).sum()
        scale[i] = (
                    (sum_data_template_prod - sum_data * sum_template/n_template)/
                    (sum_template_2 - sum_template*sum_template/n_template)
                     )
        offset[i] = (sum_data - scale[i]*sum_template)/n_template
        fitted_template = template * scale[i] + offset[i]
        sse = ((data[i:i+n_template] - fitted_template)**2).sum()
        crit[i] = scale[i]/np.sqrt(sse/(n_template-1))
    DC = scale/crit
    return(DC,  scale,  crit)


class ClementsBekkers(MiniAnalyses):
    """
    Python implementation of Clements and Bekkers 1997 algorithm
    """
    def __init__(self):
        self.dt = None
        self.data = None
        self.template = None
        self.engine = 'numba'
        self.method = 'cb'

    def setup(self, tau1=None,  tau2=None,  template_tmax=0.05,  dt=None,  
            delay=0.0,  sign=1, eventstartthr=None, risepower=4.0, min_event_amplitude=2.0):
        """
        Just store the parameters - will compute when needed
        """
        assert sign in [-1, 1]
        self.sign = sign
        self.taus = [tau1,  tau2]
        self.dt = dt
        self.template_tmax = template_tmax
        self.idelay = int(delay/dt)  # points delay in template with zeros
        self.template = None  # reset the template if needed.
        self.eventstartthr = eventstartthr
        self.risepower = risepower
        self.min_event_amplitude = min_event_amplitude

    def _make_template(self):
        """
        Private function: make template when it is needed
        """
        tau_1,  tau_2 = self.taus
        t_psc = np.arange(0,  self.template_tmax,  self.dt)
        Aprime = (tau_2/tau_1)**(tau_1/(tau_1-tau_2))
        self.template = np.zeros_like(t_psc)
        tm = 1./Aprime * ((1-(np.exp(-t_psc/tau_1)))**self.risepower * np.exp((-t_psc/tau_2)))
        
        if self.idelay > 0:
            self.template[self.idelay:] = tm[:-self.idelay]  # shift the template
        else:
            self.template = tm
        if self.sign > 0:
            self.template_amax = np.max(self.template)
        else:
            self.template = -self.template
            self.template_amax = np.min(self.template)
    
    def set_cb_engine(self, engine):
        """
        Define which detection engine to use
        cython requires compilation outised
        Numba does a JIT compilation (see routine above)
        """
        if engine in ['numba','cython']:
            self.engine = engine
        else:
            raise ValueError('CB detection engine must be either numba or cython')

    def clements_bekkers_numba(self, data):
        self.timebase = np.arange(0.,  self.data.shape[0]*self.dt,  self.dt)
        D = data.view(np.ndarray)
        if np.std(D) < 5e-12:
            DC = np.zeros(self.template.shape[0])
            Scale = np.zeros(self.template.shape[0])
            Crit = np.zeros(self.template.shape[0])
        else:
            DC, Scale, Crit = nb_clementsbekkers(D,  self.template)
        return DC, Scale, Crit
        
    def clements_bekkers_cython(self,  data):
        # pass
        ### broken for py3 at the moment
        if self.template is None:
            self._make_template()
        self.timebase = np.arange(0.,  self.data.shape[0]*self.dt,  self.dt)
        D = data.view(np.ndarray)
        T = self.template.view(np.ndarray)
        crit = np.zeros_like(D)
        scale = np.zeros_like(D)
        offset = np.zeros_like(D)
        pkl = np.zeros(100000)
        evl = np.zeros(100000)
        nout = 0
        nt = T.shape[0]
        nd = D.shape[0]
        clembek.clembek(D,  T,  self.threshold,  crit,  scale,  offset,  pkl,  evl,  nout,  self.sign,  nt,  nd)
        self.Scale = scale
        self.Crit = crit
        self.DC = offset

    def clements_bekkers(self,  data):
        """
        Implements Clements-bekkers algorithm: slides template across data,
        returns array of points indicating goodness of fit.
        Biophysical Journal,  73: 220-229,  1997.
        
        Parameters
        ----------
        data : np.array (no default)
            1D data array
        
        """
        starttime = timeit.default_timer()
        if self.template is None:
            self._make_template()    

        ## Strip out meta-data for faster computation
        D = self.sign*data.view(np.ndarray)
        T = self.template.view(np.ndarray)
        self.timebase = np.arange(0.,  data.shape[0]*self.dt,  self.dt)
        if self.engine == 'numba':
            self.DC,  self.Scale,  self.Crit = nb_clementsbekkers(D,  T)
        elif self.engine == 'cython':
            self.clements_bekkers_cython(D)
        else:
            raise ValueError('clements_bekkers: computation engine unknown (%s); must be "numba" or "cython"' % self.engine)
        endtime = timeit.default_timer() - starttime
        self.Crit = self.sign*self.Crit  # assure that crit is positive
    
    def cbTemplateMatch(self,  data, threshold=3.0, order=7):
        self.data = data
        self.threshold = threshold

        self.clements_bekkers(self.data)  # flip data sign if necessary
        # svwinlen = self.Crit.shape[0]  # smooth the crit a bit so not so dependent on noise
        # if svwinlen > 11:
        #     svn = 11
        # else:
        #     svn = svwinlen
        # if svn % 2 == 0:  # if even, decrease by 1 point to meet ood requirement for savgol_filter
        #     svn -=1
        #
        # if svn > 3:  # go ahead and filter
        #     self.Crit =  scipy.signal.savgol_filter(self.Crit, svn, 2)
        sd = np.std(self.Crit)  # HERE IS WHERE TO SCREEN OUT STIMULI/EVOKED 
        self.sdthr = sd * self.threshold  # set the threshold
        self.above = np.clip(self.Crit,  self.sdthr,  None)
        self.onsets = scipy.signal.argrelextrema(self.above,  np.greater,  order=int(order))[0] - 1 + self.idelay
        # import matplotlib.pyplot as mpl
        # f, ax = mpl.subplots(3,1)
        # ax[0].plot(self.data)
        # ax[1].plot(self.Crit)
        # ax[2].plot(self.above)
        # mpl.show()
        # exit()
        self.summarize(self.data)


class AndradeJonas(MiniAnalyses):
    """
    Deconvolution method of Andrade/Jonas,  Biophysical Journal 2012
    Create an instance of the class (aj = AndradeJonas())
    call setup to instantiate the template and data detection sign (1 for positive, -1 for negative)
    call deconvolve to perform the deconvolution
    additional routines provide averaging and some event analysis and plotting
    
    """
    def __init__(self):
        self.template = None
        self.onsets = None
        self.timebase = None
        self.dt = None
        self.sign = 1
        self.taus = None
        self.template_max = None
        self.idelay = 0
        self.method = 'aj'

    def setup(self,  tau1=None,  tau2=None,  template_tmax=None,  dt=None,  
            delay=0.0,  sign=1, eventstartthr=None, risepower=4., min_event_amplitude=2.0):
        """
        Just store the parameters - will compute when needed
        """
        assert sign in [-1, 1]
        self.sign = sign
        self.taus = [tau1,  tau2]
        self.dt = dt
        self.template_tmax = template_tmax
        self.idelay = int(delay/dt)  # points delay in template with zeros
        self.template = None  # reset the template if needed.
        self.eventstartthr = eventstartthr
        self.risepower = risepower
        self.min_event_amplitude = min_event_amplitude
        
 
    def _make_template(self):
        """
        Private function: make template when it is needed
        """
        tau_1,  tau_2 = self.taus
        t_psc = np.arange(0,  self.template_tmax,  self.dt)
        Aprime = (tau_2/tau_1)**(tau_1/(tau_1-tau_2))
        self.template = np.zeros_like(t_psc)
        tm = 1./Aprime * ((1-(np.exp(-t_psc/tau_1)))**self.risepower * np.exp((-t_psc/tau_2)))
        
        if self.idelay > 0:
            self.template[self.idelay:] = tm[:-self.idelay]  # shift the template
        else:
            self.template = tm
        if self.sign > 0:
            self.template_amax = np.max(self.template)
        else:
            self.template = -self.template
            self.template_amax = np.min(self.template)

    def deconvolve(self,  data,  data_nostim=None, thresh=1.0,  llambda=5.0,  order=7, lpf=6000., verbose=False):
        if self.template is None:
            self._make_template()
        self.data = dfilt.SignalFilter_LPFButter(data,  lpf,  1./self.dt,  NPole=8)
        self.timebase = np.arange(0.,  self.data.shape[0]*self.dt,  self.dt)
    #    print (np.max(self.timebase), self.dt)
        
        # Weiner filtering
        starttime = timeit.default_timer()
        H = np.fft.fft(self.template)
        if H.shape[0] < self.data.shape[0]:
            H = np.hstack((H,  np.zeros(self.data.shape[0]-H.shape[0])))
        self.quot = np.fft.ifft(np.fft.fft(self.data)*np.conj(H)/(H*np.conj(H) + llambda**2.0))
        self.Crit = np.real(self.quot)
        if data_nostim is None:
            data_nostim = [range(self.Crit.shape[0])]  # whole trace, otherwise remove stimuli
        else:  # clip to max of crit array, and be sure index array is integer, not float
            data_nostim = [int(x) for x in data_nostim if x < self.Crit.shape[0]]
        sd = np.nanstd(self.Crit[data_nostim])
        self.sdthr = sd * thresh  # set the threshold
        self.above = np.clip(self.Crit,  self.sdthr,  None)
        self.onsets = scipy.signal.argrelextrema(self.above,  np.greater,  order=int(order))[0] - 1 + self.idelay
        self.summarize(data)
        endtime = timeit.default_timer() - starttime
        if verbose:
            print('AJ run time: {0:.4f} s'.format(endtime))

class ZCFinder(MiniAnalyses):
    """
    Event finder using Luke's zero-crossing algorithm
    
    """
    def __init__(self):
        self.template = None
        self.onsets = None
        self.timebase = None
        self.dt = None
        self.sign = 1
        self.taus = None
        self.template_max = None
        self.idelay = 0
        self.method = 'aj'

    def setup(self,  tau1=None,  tau2=None,  template_tmax=None,  dt=None,  
            delay=0.0,  sign=1, eventstartthr=None, risepower=4., min_event_amplitude=2.0):
        """
        Just store the parameters - will compute when needed
        """
        assert sign in [-1, 1]  # must be selective, positive or negative events only
        self.sign = sign
        self.taus = [tau1,  tau2]
        self.dt = dt
        self.template_tmax = template_tmax
        self.idelay = int(delay/dt)  # points delay in template with zeros
        self.template = None  # reset the template if needed.
        self.eventstartthr = eventstartthr
        self.risepower = risepower
        self.min_event_amplitude = min_event_amplitude
        
 
    def _make_template(self):
        """
        Private function: make template when it is needed
        """
        pass  # no template used...
        
    def find_events(self, data, data_nostim=None, minPeak=0., thresh=0., minSum=0., minLength=3, verbose=False):
        self.data = data # filtering should be done before this ... # dfilt.SignalFilter_LPFButter(data,  lpf,  1./self.dt,  NPole=8)
        self.timebase = np.arange(0.,  self.data.shape[0]*self.dt,  self.dt)

        starttime = timeit.default_timer()
        self.sdthr = thresh
        self.Crit = np.zeros_like(self.data)
        # if data_nostim is None:
        #     data_nostim = [range(self.Crit.shape[0])]  # whole trace, otherwise remove stimuli
        # else:  # clip to max of crit array, and be sure index array is integer, not float
        #     data_nostim = [int(x) for x in data_nostim if x < self.Crit.shape[0]]
        events = FN.zeroCrossingEvents(data, minLength=minLength, minPeak=minPeak, minSum=minSum, noiseThreshold=thresh, sign=self.sign)
        self.onsets = np.array([x[0] for x in events]).astype(int)

        self.summarize(data)
        endtime = timeit.default_timer() - starttime
        if verbose:
            print('ZC run time: {0:.4f} s'.format(endtime))


#  Some general functions

def moving_average(a,  n=3) :
    ret = np.cumsum(a,  dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n, n


def generate_testdata(dt,  maxt=10., meanrate=10.,  amp=20.e-12,  ampvar=5.e-12,  
        noise=2.5e-12, taus=[0.001, 0.010], baseclass=None, func=None, sign=1, 
        expseed=None, noiseseed=None,
        bigevent=None):
    """
        meanrate is in Hz(events/second)
        maxt is in seconds
        bigevent is a dict {'t': delayinsec, 'I': amplitudeinA}
    """
    if baseclass is None and func is not None:
        raise ValueError('Need base class definition')
    tdur = 0.020
    timebase = np.arange(0.,  maxt,  dt) # in ms
    t_psc = np.arange(0.,  tdur,  dt)  # time base for single event template in ms
    if func is None:
        tau_1 = taus[0] # ms
        tau_2 = taus[1] # ms
        Apeak = amp # pA
        Aprime = (tau_2/tau_1)**(tau_1/(tau_1-tau_2))
        g = Aprime * (-np.exp(-t_psc/tau_1) + np.exp((-t_psc/tau_2)))
        gmax = np.max(g)
        print('gmax: ', gmax)
        g = sign*g*amp/gmax
        print(f'max g: {np.min(g):.6e}')
    else:
        baseclass._make_template()
        gmax = np.min(baseclass.template)
        g = sign*amp*baseclass.template/gmax
        print('gmaxb: ', np.max(gmax))
   
    testpsc = np.zeros(timebase.shape)
    if expseed is None:
        eventintervals = np.random.exponential(1./meanrate, int(maxt*meanrate))
    else:
        np.random.seed(expseed)
        eventintervals = np.random.exponential(1./meanrate, int(maxt*meanrate))
    eventintervals = eventintervals[eventintervals < 10.]
    events = np.cumsum(eventintervals)
    if bigevent is not None:
        events = np.append(events, bigevent['t'])
        events = np.sort(events)
    t_events = events[events < maxt]  # time of events with exp distribution
    i_events = np.array([int(x/dt) for x in t_events])
    testpsc[i_events] = np.random.normal(1.,  ampvar/amp,  len(i_events))
    if bigevent is not None:
        ipos = int(bigevent['t']/dt) # position in array
        testpsc[ipos] = bigevent['I']
    testpsc = scipy.signal.convolve(testpsc,  g,  mode='full')[:timebase.shape[0]]
    # f, ax = mpl.subplots(1,1)
    # ax.plot(dt*np.arange(len(testpsc)), testpsc)
    # mpl.show()
    if noise > 0:
        if noiseseed is None:
            testpscn = testpsc + np.random.normal(0.,  noise,  testpsc.shape)
        else:
            np.random.seed(noiseseed)
            testpscn = testpsc + np.random.normal(0.,  noise,  testpsc.shape)
    else:
        testpscn = testpsc
    return timebase,  testpsc,  testpscn,  i_events

def zc_test():
    """
    Do some tests of the CB protocol and plot
    """
    sign = -1
    trace_dur = 10.
    dt = 5e-5
    taus = [0.001, 0.005]
    minlen = int(0.003/dt)
    for i in range(1):
        timebase,  testpsc,  testpscn,  i_events = generate_testdata(dt, maxt=trace_dur,
            amp=100e-12,  ampvar=20e-12,  meanrate=10., noise=15.0e-12, taus=taus, func=None, sign=sign,
            expseed=i, noiseseed=i*47, bigevent={'t': 1.0, 'I': 20.})
        zc = ZCFinder()
        zc.setup(dt=dt, tau1=0.001, tau2=0.005, sign=-1)
        events = zc.find_events(testpscn,  data_nostim=None, thresh=1.5,  minLength=minlen)
    # print(len(events))
    zc.plots(title = 'Zero Crossings')
    # f, ax = mpl.subplots(3, 1)
    # ax[0].plot(timebase, testpscn)
    # evindex = [x[0] for x in events]
    # ax[0].plot(timebase[evindex], testpscn[evindex], 'ro', markersize=2)
    # mpl.show()


def cb_tests():
    """
    Do some tests of the CB protocol and plot
    """
    sign = 1
    trace_dur = 10.
    dt = 5e-5
    taus = [0.001, 0.005]
    for i in range(1):
        timebase,  testpsc,  testpscn,  i_events = generate_testdata(dt, maxt=trace_dur,
            amp=100e-12,  ampvar=20e-12,  meanrate=10., noise=15.0e-12, taus=taus, func=None, sign=sign,
            expseed=i, noiseseed=i*47, bigevent={'t': 1.0, 'I': 20.})
        cb = ClementsBekkers()
        cb.setup(tau1=0.001,  tau2=0.003,  dt=dt,  delay=0.0, template_tmax=3*taus[1],  sign=sign)
        cb._make_template()
        cb.cbTemplateMatch(testpscn,  threshold=3.0)
    cb.plots(title='Clements Bekkers')
    return cb


def aj_tests():
    sign = -1
    trace_dur = 10.  # seconds
    dt = 5e-5
    amp = 100e-12
    for i in range(1):
        aj = AndradeJonas()
        aj.setup(tau1=0.001,  tau2=0.007,  dt=dt,  delay=0.0, template_tmax=trace_dur, # taus are for template
            sign=sign, risepower=4.0)
        # generate test data
        timebase,  testpsc,  testpscn,  i_events = generate_testdata(aj.dt, maxt=trace_dur,
                meanrate=10.,
                amp=amp,  ampvar=20e-12,  noise=15e-12, taus=[0.001, 0.005], baseclass=aj, func=None, sign=sign,
                expseed=i, noiseseed=i*47, bigevent={'t': 1.0, 'I': 20})

        aj.deconvolve(testpscn-np.mean(testpscn),  thresh=5, llambda=1,  order=int(0.001/aj.dt))
    aj.summarize(aj.data)
    aj.plots(events=None, title='AJ') # i_events)
    return aj
    

if __name__ == "__main__":
    import matplotlib

    rcParams = matplotlib.rcParams
    rcParams['svg.fonttype'] = 'none' # No text as paths. Assume font installed.
    rcParams['pdf.fonttype'] = 42
    rcParams['ps.fonttype'] = 42
    #rcParams['text.latex.unicode'] = True
    #rcParams['font.family'] = 'sans-serif'
    # rcParams['font.sans-serif'] = 'DejaVu Sans'
    # rcParams['font.weight'] = 'regular'                  # you can omit this, it's the default
    # rcParams['font.sans-serif'] = ['Arial']
    rcParams['text.usetex'] = False
    import matplotlib.pyplot as mpl
    import matplotlib.collections as collections
    import warnings  # need to turn off a scipy future warning.
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", message="UserWarning: findfont: Font family ['sans-serif'] not found. Falling back to DejaVu Sans")
    import pylibrary.plotting.plothelpers as PH

    # aj = aj_tests()
    #aj.fit_individual_events(aj.onsets)
    #aj.plot_individual_events()
    cb_tests()
    # zc_test()