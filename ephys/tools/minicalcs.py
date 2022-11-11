import importlib
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import scipy.signal
import pyqtgraph as pg
import toml
from pylibrary.tools import cprint as CP
from pylibrary.tools import fileselector as FS
from pyqtgraph.parametertree import Parameter, ParameterTree

from . import digital_filters as FILT

from ..ephys_analysis import RmTauAnalysis, SpikeAnalysis
from ..datareaders import acq4_reader
from ..mini_analyses import minis_methods, minis_methods_common

class MiniCalcs():
    def __init__(self, parent=None):
        self.parent = parent
        self.hp = []
        self.xp = []
        
    def CB(self):
        self.parent._getpars()
        self.parent.method = minis_methods.ClementsBekkers()
        self.parent.method.set_cb_engine('numba')
        rate = np.mean(np.diff(self.parent.tb))
        jmax = int((2 * self.parent.tau1 + 3 * self.parent.tau2) / rate)
        CP.cprint("r", f"showdata CB threshold: {self.parent.thresh_reSD:8.2f}")
        self.parent.method.setup(
            ntraces=self.parent.mod_data.shape[0],
            tau1=self.parent.tau1,
            tau2=self.parent.tau2,
            dt_seconds=rate,
            delay=0.0,
            template_tmax=rate * (jmax - 1),
            threshold=self.parent.thresh_reSD,
            sign=self.parent.sign,
            eventstartthr=None,
            lpf=self.parent.LPF,
            hpf=self.parent.HPF,
        )
        self.parent.imax = int(self.parent.maxT * self.parent.AR.sample_rate[0])

        # meandata = np.mean(self.parent.mod_data[:, : self.parent.imax])
        self.parent.method._make_template()
        for i in range(self.parent.mod_data.shape[0]):
            self.parent.method.cbTemplateMatch(
                self.parent.mod_data[i, : self.parent.imax], itrace=i, lpf=self.parent.LPF
            )
            self.parent.mod_data[i, : self.parent.imax] = self.parent.method.data  # # get filtered data
            self.parent.method.reset_filtering()
        self.parent.last_method = "CB"
        self.CB_update()

    def CB_update(self):
        if self.parent.method is None:
            return
        self.parent.method.threshold = self.parent.thresh_reSD
        self.parent.method.identify_events(order=self.parent.Order)
        self.parent.method.summarize(self.parent.mod_data[:, : self.parent.imax])
        self.decorate(self.parent.method)

    def AJ(self):
        self.parent._getpars()
        self.parent.method = minis_methods.AndradeJonas()
        rate = np.mean(np.diff(self.parent.tb))
        jmax = int((2 * self.parent.tau1 + 3 * self.parent.tau2) / rate)
        CP.cprint("g", f"showdata AJ threshold: {self.parent.thresh_reSD:8.2f}")
        print("template len: ", jmax, "template max t: ", rate * (jmax - 1), rate)
        self.parent.method.setup(
            ntraces=self.parent.mod_data.shape[0],
            tau1=self.parent.tau1,
            tau2=self.parent.tau2,
            dt_seconds=rate,
            delay=0.0,
            template_tmax=np.max(self.parent.tb),
            threshold=self.parent.thresh_reSD,
            sign=self.parent.sign,
            eventstartthr=None,
            lpf=self.parent.LPF,
            hpf=self.parent.HPF,
        )
        self.parent.imax = int(self.parent.maxT * self.parent.AR.sample_rate[0])
        meandata = np.mean(self.parent.mod_data[:, : self.parent.imax])
        # self.parent.AJorder = int(1e-3/rate)
        print(f"AJ: Order={int(self.parent.Order):d}, rate: {rate*1e3:.3f} ms, tau1: {self.parent.tau1*1e3:.5f} ms  tau2: {self.parent.tau2*1e3:.5f}")
        for i in range(self.parent.mod_data.shape[0]):
            self.parent.method.deconvolve(
                self.parent.mod_data[i, : self.parent.imax] - meandata,
                itrace=i,
                # data_nostim=None,
                llambda=5.0,
            )  # assumes times are all in same units of msec
            self.parent.mod_data[i, : self.parent.imax] = self.parent.method.data  # # get filtered data
            self.parent.method.reset_filtering()

        self.parent.last_method = "AJ"
        self.AJ_update()

    def AJ_update(self):
        if self.parent.method is None:
            return
        self.parent.method.threshold = self.parent.thresh_reSD
        self.parent.method.identify_events(order=self.parent.Order)
        self.parent.method.summarize(self.parent.mod_data[:, : self.parent.imax])
        # tot_events = sum([len(x) for x in self.parent.method.onsets])
        self.decorate(self.parent.method)
        # self.parent.method.average_event()
        # print(dir(self.parent.method.Summary))
        # print(dir(self.parent.method.Summary.average))
        # print(dir(self.parent.method.Summary.average.fitted))
        # print('tau1: ', self.parent.method.Summary.average.fitted_tau1)
        # print('tau2: ', self.parent.method.Summary.average.fitted_tau2)
        # print('Amp:  ', self.parent.method.Summary.average.Amplitude)
        # print('Nev:  ', self.parent.method.Summary.average.Nevents)
        # print('1090RT:  ', self.parent.method.Summary.average.risetenninety)
        # print('37Decay: ', self.parent.method.Summary.average.decaythirtyseven)
        
        

    def RS(self):
        self.parent._getpars()
        self.parent.method = minis_methods.RSDeconvolve()
        rate = np.mean(np.diff(self.parent.tb))

        self.parent.method.setup(
            ntraces=self.parent.mod_data.shape[0],
            tau1=self.parent.tau1,
            tau2=self.parent.tau2,
            dt_seconds=rate,
            delay=0.0,
            template_tmax=np.max(self.parent.tb),  # taus are for template
            sign=self.parent.sign,
            risepower=4.0,
            threshold=self.parent.thresh_reSD,
            lpf=self.parent.LPF,
            hpf=self.parent.HPF,
        )
        CP.cprint("c", f"showdata RS threshold: {self.parent.thresh_reSD:8.2f}")
        # generate test data
        self.parent.imax = int(self.parent.maxT * self.parent.AR.sample_rate[0])
        # meandata = np.mean(self.parent.mod_data[:, : self.parent.imax])
        with pg.ProgressDialog("RS Processing", 0, self.parent.mod_data.shape[0]) as dlg:
            for i in range(self.parent.mod_data.shape[0]):
                self.parent.method.deconvolve(
                    self.parent.mod_data[i, : self.parent.imax], itrace=i,
                )
                self.parent.mod_data[i, : self.parent.imax] = self.parent.method.data  # # get filtered data
                self.parent.method.reset_filtering()
            self.parent.last_method = "RS"
            self.RS_update()
            dlg.setValue(i)
            if dlg.wasCanceled():
                raise Exception("Processing canceled by user")

    def RS_update(self):
        if self.parent.method is None:
            return
        self.parent.method.threshold = self.parent.thresh_reSD
        self.parent.method.identify_events(order=self.parent.Order)
        self.parent.method.summarize(self.parent.mod_data[:, : self.parent.imax])
        self.decorate(self.parent.method)

    def ZC(self):
        self.parent._getpars()
        self.parent.method = minis_methods.ZCFinder()
        rate = np.mean(np.diff(self.parent.tb))
        minlen = int(self.parent.ZC_mindur / rate)
        self.parent.method.setup(
            ntraces=self.parent.mod_data.shape[0],
            tau1=self.parent.tau1,
            tau2=self.parent.tau2,
            dt_seconds=rate,
            delay=0.0,
            template_tmax=np.max(self.parent.tb),
            sign=self.parent.sign,
            threshold=self.parent.thresh_reSD,
            lpf=self.parent.LPF,
            hpf=self.parent.HPF,
        )
        CP.cprint("y", f"showdata ZC threshold: {self.parent.thresh_reSD:8.2f}")

        for i in range(self.parent.mod_data.shape[0]):
            self.parent.method.deconvolve(
                self.parent.mod_data[i, : self.parent.imax], itrace=i,
            )
            self.parent.mod_data[i, : self.parent.imax] = self.parent.method.data  # # get filtered data
            self.parent.method.reset_filtering()
        self.parent.last_method = "ZC"
        self.ZC_update()

    def ZC_update(self):
        if self.parent.method is None:
            return
        self.parent.method.threshold = self.parent.thresh_reSD
        self.parent.method.identify_events(data_nostim=None)
        self.parent.method.summarize(self.parent.mod_data[:, : self.parent.imax])
        self.decorate(self.parent.method)

    def decorate(self, minimethod):
        if not self.parent.curve_set:
            return
        # print('decorating', )
        for s in self.parent.scatter:
            s.clear()
        for c in self.parent.crits:
            c.clear()
        # for line in self.parent.threshold_line:
        #     line.clear()
        self.parent.scatter = []
        self.parent.crits = []

        if minimethod.Summary.onsets is not None and len(minimethod.Summary.onsets) > 0:
            # self.parent.scatter.append(
            #     self.parent.dataplot.plot(
            #         self.parent.tb[minimethod.Summary.peaks[self.parent.current_trace]] * 1e3,
            #         self.parent.current_data[minimethod.Summary.peaks[self.parent.current_trace]],
            #         pen=None,
            #         symbol="o",
            #         symbolPen=None,
            #         symbolSize=10,
            #         symbolBrush=(255, 0, 0, 255),
            #     )
            # )
            pass
            # self.parent.scatter.append(self.parent.dataplot.plot(self.parent.tb[minimethod.peaks]*1e3,
            # np.array(minimethod.amplitudes),
            #           pen = None, symbol='o', symbolPen=None, symbolSize=5,
            # symbolBrush=(255, 0, 0, 255)))

        self.parent.crits.append(
            self.parent.dataplot2.plot(
                self.parent.tb[: len(minimethod.Criterion[self.parent.current_trace])] * 1e3,
                minimethod.Criterion[self.parent.current_trace],
                pen="r",
            )
        )
        self.parent.threshold_line.setValue(minimethod.sdthr)
        # self.parent.threshold_line.setLabel(f"SD thr: {self.parent.thresh_reSD:.2f}  Abs: {self.parent.minimethod.sdthr:.3e}")
        # print(' ... decorated')
    def fold_data(self, t, d, period):
        """
        Fold a data set within a time period
        """
    def show_fitting_pars(self):
        if self.parent.w1.slider.value() != 0:
            return
        try:
            if not self.parent.method.Summary.average.averaged:
                CP.cprint("r", "Fit not yet run")
                return
        except:
            return
        avg = self.parent.method.Summary.average
        print('-'*60)
        print(f"File: {self.parent.fileName:s}")
        print(f"tau1:    {avg.fitted_tau1*1e3:.3f} msec")
        print(f"tau2:    {avg.fitted_tau2*1e3:.3f} msec")
        print(f"Amp:     {avg.Amplitude*1e12:.1f} pA")
        print(f"Nevents: {avg.Nevents:d}")
        print(f"1090RT:  {avg.risetenninety*1e3:.3f} msec")
        print(f"37Decay: {avg.decaythirtyseven*1e3:.3f} msec")
        print('-'*60)
        # print line for datasets files that look like:
        # '2021.05.19_000~slice_000~cell_000~Vc_spont_5min_Vrest_000':MINID(ID=921, GT='WT', EYFP='-', NG=[25,29], rt=0.0005, decay=0.003, thr=3.2),
        fnparts = Path(self.parent.fileName).parts
        fn = "~".join(fnparts[-4:])
        print(f"'{fn:s}': MINID(ID=100, GT='FF', EYFP='-', NG=[], ", end="")
        print(f"rt={avg.fitted_tau1:5f}, decay={avg.fitted_tau2:.5f}, thr={self.parent.thresh_reSD:.2f}),")
        print('-'*60)
        self.parent.clear_fit_lines()
        # make all events in one long array broken by nans
        allevents = self.parent.method.Summary.allevents
        allev = np.concatenate((allevents*1e12, np.array([np.nan]*allevents.shape[0])[:, None]), axis=1)
        x = np.concatenate((avg.avgeventtb*1e3, [np.nan]))
        x = np.reshape(np.tile(x, allevents.shape[0]), (allevents.shape[0], x.shape[0]))
        allev = np.reshape(allev, (1, np.prod(allev.shape)))[0]
        x = np.reshape(x, (1, np.prod(x.shape)))[0]
        self.parent.fitplot.setXRange(0, np.nanmax(x))  # preset to avoid rescaling during plotting
        self.parent.fitplot.setYRange(np.nanmin(allev), np.nanmax(allev))
        
        l = self.parent.fitplot.plot(x, allev, pen=pg.mkPen({'color': "#66F", 'width': 0.25}),
            connect="finite")
        self.parent.fitlines.append(l)
        l = self.parent.fitplot.plot(avg.avgeventtb*1e3, avg.avgevent*1e12, pen="w")
        self.parent.fitlines.append(l)
        l = self.parent.fitplot.plot(avg.avgeventtb*1e3, self.parent.method.avg_best_fit*1e12, pen="r")
        self.parent.fitlines.append(l)
        self.direct_60Hz()

    def direct_60Hz(self):
        d = self.parent.mod_data # 
        freq = self.parent.AR.sample_rate[0]
        tb = self.parent.tb
        maxt_original = np.max(tb)
        fundamental = 60.
        n60periods = int(np.floor(maxt_original*fundamental))
        f60 = int((np.floor(freq/fundamental)+1)*fundamental)   # new sampling freq that is divisible by 60 Hz
        tb60max = n60periods/fundamental
        tb60max_2 = (n60periods+1)/fundamental  # need for extra cycle
        assert tb60max <= maxt_original
        t60 = np.arange(0, tb60max, 1./f60)  # time base to go with
        ifold = int(t60.shape[0]/n60periods)
        t60_2 = np.arange(0, tb60max_2, 1./f60)
        # interpolate the data onto a sample frequency that is an exact multiple of 60 Hz
        xpl = self.parent.xplot.listDataItems()
        for x in xpl:
            self.parent.xplot.removeItem(x)
        hpl = self.parent.histplot.listDataItems()
        for x in hpl:
            self.parent.histplot.removeItem(x)
        
        dnew_filt = np.zeros_like(d)
        dlen = d.shape[1]
        for k in range(d.shape[0]):
            dn = np.interp(t60, tb, d[k,:])
            dna = np.reshape(dn, (n60periods, ifold))
            dna = np.mean(dna, axis=0)
            dnas = scipy.signal.savgol_filter(dna,31, 9, mode='wrap')
            self.parent.xplot.plot(t60[:ifold], dna, pen=pg.mkColor((k, d.shape[0])))
            self.parent.xplot.plot(t60[:ifold], dnas, pen=pg.mkColor((k, d.shape[0])))
            dnn = np.tile(dna, n60periods+1)[:dlen]
            dnn2 = np.interp(tb, t60_2[:dlen], dnn)
            # dnn = dnn[:tb.shape[0]]
            dnew_filt[k] = d[k,:]-dnn2
            if k == 0:
                self.parent.histplot.plot(tb, d[k,:], pen=pg.mkPen({'color': "#FFF", 'width': 1.5}))
                # self.parent.histplot.plot(tb, dnew_filt[k], pen="y")
            # dna = np.tile(dna, n60periods)
            # self.parent.xplot.plot(t60, dna, pen=pg.mkColor((k, d.shape[0])))
            # self.parent.histplot.plot(t60, dn, pen='m')  # interpolated trace
        return dnew_filt