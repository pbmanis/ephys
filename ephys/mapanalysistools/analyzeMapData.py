from __future__ import absolute_import

"""
analyze ChR2 or uncaging map data

"""
import re
import sys
import sqlite3
from pathlib import Path
from dataclasses import dataclass
from dataclasses import dataclass, field
from typing import Union, Dict, List

import numpy as np
import scipy.signal
import scipy.ndimage

import os.path
from collections import OrderedDict

import math
import dill as pickle
import datetime
import timeit

import pyqtgraph.multiprocess as mp

import ephys.ephysanalysis as EP
import montage as MONT

import ephys.mini_analyses as minis
from ephys.mapanalysistools import functions
from ephys.mapanalysistools import digital_filters as FILT
from ephys.mapanalysistools import plotMapData as PMD

from ephys.mini_analyses import minis_methods

import pylibrary.tools.cprint as CP



basedir = "/Users/pbmanis/Desktop/Python/mapAnalysisTools"

re_degree = re.compile(r"\s*(\d{1,3}d)\s*")
re_duration = re.compile(r"(\d{1,3}ms)")
np.seterr(divide="raise")
# print ('maps: ', colormaps)
#


def def_notch():
    """ defaults for notch frequencies"""
    return [60.0, 120.0, 180.0, 240.0]

def def_stimtimes():
    return {'start': [0.3], 'duration': [1e-3]}

def def_twin_base():
    return [0.0, 0.295]


def def_twin_response():
    return [0.301, 0.325]


def def_taus():
    return [0.0002, 0.005]


def def_shutter_artifacts():
    return [0.055]


def def_analysis_window():
    return [0.0, 0.999]


@dataclass
class AnalysisPars:
    """
    Data class that holds most of the analysis parameters
    This class is also passed to the plotMapdata routines
    (display_one_map).
    """
    spotsize: float = 42e-6  # spot size in meters
    LPF_flag: bool = False  # flag enabling low-pass filter
    HPF_flag: bool = False  # flag enabling high-pass filter
    LPF: float = 5000.0  # low-pass filter frequency Hz
    HPF: float = 0.0  # high-pass filter frequrency, Hz
    notch_flag: bool = False  # flag enabling notch filters
    notch_freqs: list = field(default_factory=def_notch)  # list of notch frequencies
    notch_Q: float = 90.0  # Q value for notch filters (sharpness)
    fix_artifact_flag: bool = True  # flag enabling removeal of artifacts
    ar_start: float = 0.10 # starting time for stimuli
    stimtimes: dict=field(
        default_factory=def_stimtimes)
    spont_deadtime: float = 0.010  # time after trace onset before counting spont envents
    direct_window: float = 0.001  # window after stimulus for direct response
    response_window: float = 0.015  # window end for response (response is between direct and respons), seconds
    twin_base: list = field(
        default_factory=def_twin_base
    )  # time windows baseline measures
    twin_resp: list = field(
        default_factory=def_twin_response
    )  # time windows for repeated responses
    analysis_window: list = field(
        default_factory=def_analysis_window
    )  # window for data analysis
    # taus = [0.5, 2.0]
    taus: list = field(
        default_factory=def_taus
    )  # initial taus for fitting detected events
    threshold: float = 3.0  # default threshold for CB, AJ or ZC
    sign: int = -1  # negative for EPSC, positive for IPSC
    datatype: str='V'  # data type - VC or IC
    stepi : float = 25.0  # step size for stacked traces, in pA (25 default for cc; 2 for cc)
    scale_factor: float = 1.0  # scale factore for data (convert to pA or mV,,, )
    overlay_scale: float = 0.0
    shutter_artifact: list = field(
        default_factory=def_shutter_artifacts
    )  # time of shutter electrical artifat
    artifact_suppress: bool = True  # flag enabling suppression of artifacts
    artifact_duration: float = 2.0  # length of artifat, in msec
    stimdur: Union[float, None] = None  # duration of stimuli
    noderivative_artifact: bool = True  # flag enableing artifact suppression based on derivative of trace
    sd_thr: float = 3.0  # threshold in sd for diff based artifact suppression.


@dataclass
class AnalysisData:
    """
    Data class that holds the analysis data separate from the parameters
    and metadata
    This class is also made available to the plotMapdata routines
    """
    tb: Union[None, np.ndarray] = None
    data_clean: Union[None, np.ndarray] = None
    photodiode: Union[None, np.ndarray] = None
    photodiode_timebase: Union[None, np.ndarray] = None
    MA : Union[object, None] = None  # point to minanalysis instance used for analysis
    


class AnalyzeMap(object):
    def __init__(self, rasterize=True):
        self.Pars = AnalysisPars()
        self.Data = AnalysisData()
        self.AR = EP.acq4read.Acq4Read()
        self.SP = EP.SpikeAnalysis.SpikeAnalysis()
        self.RM = EP.RmTauAnalysis.RmTauAnalysis()
        self.verbose = True
        self.last_dataset = None
        self.last_results = None
        self.lbr_command = False  # laser blue raw waveform (command)

        # set some defaults - these will be overwrittein with readProtocol
        self.template_file = None

        self.methodname = "aj"  # default event detector
        
        self.MA = minis.minis_methods.MiniAnalyses()  # get a minianalysis instance
        self.Pars.MA = self.MA  # instance may be needed for plotting

    def set_analysis_window(self, t0: float = 0.0, t1: Union[float, None] = None):
        assert t1 is not None  # force usage of t1
        self.Pars.analysis_window = [t0, t1]

    def set_notch(self, notch, freqs=[60], Q=90.0):
        self.Pars.notch_flag = notch
        self.Pars.notch_freqs = freqs
        self.Pars.notch_Q = Q

    def set_LPF(self, LPF):
        self.Pars.LPF = LPF
        self.Pars.LPF_flag = True

    def set_HPF(self, HPF):
        self.Pars.HPF = HPF
        if HPF > 0:
            self.Pars.HPF_flag = True

    def set_methodname(self, methodname):
        if methodname.lower() in ["aj"]:
            self.methodname = "aj"
            self.engine = "pythonn"
        elif methodname.lower() in ["cb"]:
            self.methodname = "cb"
            self.engine = "numba"  # set this as the default
        elif methodname.lower() in ["cb_numba"]:
            self.methodname = "cb"
            self.engine = "numba"
        elif methodname.lower() in ["cb_cython"]:
            self.methodname = "cb"
            self.engine = "cython"
        elif methodname.lower() in ["cb_python"]:
            self.methodname = "cb"
            self.engine = "python"
        elif methodname.lower() in ["zc"]:
            self.methodname = "zc"
            self.engine = "python"
        else:
            raise ValueError("Selected event detector %s is not valid" % methodname)

    def set_taus(self, taus):
        if len(taus) != 2:
            raise ValueError(
                "Analyze Map Data: need two tau values in list!, got: ", taus
            )
        self.Pars.taus = sorted(taus)

    def set_shutter_artifact_time(self, t):
        self.Pars.shutter_artifact = t

    def set_artifact_suppression(self, suppr=True):
        if not isinstance(suppr, bool):
            raise ValueError(
                "analyzeMapData: artifact suppresion must be True or False"
            )
        self.Pars.artifact_suppress = suppr
        self.Pars.fix_artifact_flag = suppr

    def set_artifact_duration(self, duration=2.0):
        self.Pars.artifact_duration = duration

    def set_stimdur(self, duration=None):
        self.Pars.stimdur = duration

    def set_noderivative_artifact(self, suppr=True):
        if not isinstance(suppr, bool):
            raise ValueError(
                "analyzeMapData: derivative artifact suppresion must be True or False"
            )
        self.Pars.noderivative_artifact = suppr

    def set_artifact_file(self, filename):
        self.template_file = Path("template_data_" + filename + ".pkl")

    def readProtocol(
        self, protocolFilename, records=None, sparsity=None, getPhotodiode=False
    ):
        starttime = timeit.default_timer()
        self.protocol = protocolFilename
        print("Reading Protocol: ", protocolFilename)
        self.AR.setProtocol(protocolFilename)
        if not protocolFilename.is_dir() or not self.AR.getData():
            print("  >>No data found in protocol: %s" % protocolFilename)
            return None, None, None, None
        # print('Protocol: ', protocolFilename)
        self.Pars.datatype = self.AR.mode[0].upper()  # get mode and simplify to I or V
        if self.Pars.datatype == "I":
            self.Pars.stepi = 2.0
        # otherwise use the default, which is set in the init routine

        self.Pars.stimtimes = self.AR.getBlueLaserTimes()
        if self.Pars.stimtimes is not None:
            self.Pars.twin_base = [
                0.0,
                self.Pars.stimtimes["start"][0] - 0.001,
            ]  # remember times are in seconds
            self.Pars.twin_resp = []
            for j in range(len(self.Pars.stimtimes["start"])):
                self.Pars.twin_resp.append(
                    [
                        self.Pars.stimtimes["start"][j] + self.Pars.direct_window,
                        self.Pars.stimtimes["start"][j] + self.Pars.response_window,
                    ]
                )
        self.lbr_command = (
            self.AR.getLaserBlueCommand()
        )  # just get flag; data in self.AR

        if self.AR.getPhotodiode():
            self.Data.photodiode = self.AR.Photodiode
            self.Data.photodiode_timebase = self.AR.Photodiode_time_base
        else:
            CP.cprint('r', 'Could not get photodiode traces')

        self.shutter = self.AR.getDeviceData("Laser-Blue-raw", "Shutter")
        self.AR.getScannerPositions()
        self.Pars.ar_tstart = self.AR.tstart
        self.Pars.spotsize = self.AR.spotsize
        self.Data.tb = self.AR.time_base

        data = np.reshape(
            self.AR.traces,
            (
                self.AR.repetitions,
                int(self.AR.traces.shape[0] / self.AR.repetitions),
                self.AR.traces.shape[1],
            ),
        )
        endtime = timeit.default_timer()

        print(
            "    Reading protocol {0:s} took {1:6.1f} s".format(
                protocolFilename.name, endtime - starttime
            )
        )
        return data, self.AR.time_base, self.AR.sequenceparams, self.AR.scannerinfo

    def set_analysis_windows(self):
        pass

    def calculate_charge(
        self,
        tb: np.ndarray,
        data: np.ndarray,
        twin_base: list = [0, 0.1],
        twin_resp: list = [[0.101, 0.130]],
    ) -> (float, float):
        """
        Integrate durrent over a time window to get charges
        
        Returns two charge measures: the baseline, and the value
        in the response window
        """
        # get indices for the integration windows
        tbindx = np.where((tb >= twin_base[0]) & (tb < twin_base[1]))
        trindx = np.where((tb >= twin_resp[0]) & (tb < twin_resp[1]))
        Qr = 1e6 * np.sum(data[trindx]) / (twin_resp[1] - twin_resp[0])  # response
        Qb = 1e6 * np.sum(data[tbindx]) / (twin_base[1] - twin_base[0])  # baseline
        return Qr, Qb

    def ZScore(
        self,
        tb: np.ndarray,
        data: np.ndarray,
        twin_base: list = [0, 0.1],
        twin_resp: list = [[0.101, 0.130]],
    ) -> float:
        """
        Compute a Z-Score on the currents, comparing
        the mean and standard deviation during the baseline
        with the mean in a response window
        # abs(post.mean() - pre.mean()) / pre.std()
        """
        # get indices for the integration windows
        tbindx = np.where((tb >= twin_base[0]) & (tb < twin_base[1]))
        trindx = np.where((tb >= twin_resp[0]) & (tb < twin_resp[1]))
        mpost = np.mean(data[trindx])  # response
        mpre = np.mean(data[tbindx])  # baseline
        try:
            zs = np.fabs((mpost - mpre) / np.std(data[tbindx]))
        except:
            zs = 0
        return zs

    def Imax(
        self,
        tb: np.ndarray,
        data: np.ndarray,
        twin_base: list = [0, 0.1],
        twin_resp: list = [[0.101, 0.130]],
        sign: int = 1,
    ) -> float:

        tbindx = np.where((tb >= twin_base[0]) & (tb < twin_base[1]))
        trindx = np.where((tb >= twin_resp[0]) & (tb < twin_resp[1]))
        mpost = np.max(sign * data[trindx])  # response goes negative...
        return mpost

    def select_events(
        self,
        pkt: Union[list, np.ndarray],
        tstarts: list,  # window starts
        tdurs: list,  # window durations
        rate: float,
        mode: str = "reject",
        thr: float = 5e-12,
        data: Union[np.ndarray, None] = None,
        first_only: bool = False,
        debug: bool = False,
    ) -> list:
        """
        return indices where the input index is outside (or inside) a set of time windows.
        tstarts is a list of window starts
        twin is the duration of each window
        rate is the data sample rate (in msec...)
        pkt is the list of times to compare against.
        """

        # print('rate: ', rate)
        if mode in ["reject", "threshold_reject"]:
            npk = list(range(len(pkt)))  # assume all
        else:
            npk = []
        for itw, tw in enumerate(tstarts):  # and for each stimulus
            first = False
            if isinstance(tdurs, list) or isinstance(
                tdurs, np.ndarray
            ):  # either use array parallel to tstarts, or
                ttwin = tdurs[itw]
            else:
                ttwin = tdurs  # or use just a single value
            ts = int(tw / rate)
            te = ts + int(ttwin / rate)
            for k, pk in enumerate(pkt):  # over each index
                if (
                    mode == "reject" and npk[k] is None
                ):  # means we have already rejected the n'th one
                    continue
                if mode == "reject":
                    if pk >= ts and pk < te:
                        npk[k] = None
                elif (mode == "threshold_reject") and (data is not None):
                    if (pk >= ts) and (pk < te) and (np.fabs(data[k]) < thr):
                        print("np.fabs: ", np.fabs(data[k]), thr)
                        npk[k] = None
                elif mode == "accept":
                    if debug:
                        print("accepting ?: ", ts, k, pk, te, rate)
                    if pk >= ts and pk < te and not first:
                        if debug:
                            print("    ok")
                        if k not in npk:
                            npk.append(k)
                        if first_only and not first:
                            first = True
                            break

                else:
                    raise ValueError(
                        "analyzeMapData:select_times: mode must be accept, threshold_reject, or reject; got: %s"
                        % mode
                    )
        if debug:
            print("npk: ", npk)
        npks = [
            n for n in npk if n is not None
        ]  # return the truncated list of indices into pkt
        return npks

    def select_by_sign(
        self, method: object, npks: int, data: np.ndarray, min_event: float = 5e-12
    ) -> Union[list, np.ndarray]:
        """
        Screen events for correct sign and minimum amplitude.
        Here we use the onsets and smoothed peak to select
        for events satisfying criteria.

        Parameters
        ----------
        method : object (mini_analysis object)
            result of the mini analysis. The object must contain
            at least two lists, one of onsets and one of the smoothed peaks.
            The lists must be of the same length.

        data : array
            a 1-D array of the data to be screened. This is the entire
            trace.

        event_min : float (default 5e-12)
            The smallest size event that will be considered acceptable.
        """

        pkt = []
        if len(method.onsets) == 0:
            return pkt
        tb = method.timebase  # full time base
        smpks = np.array(method.smpkindex)
        # events[trial]['aveventtb']
        rate = np.mean(np.diff(tb))
        tb_event = method.avgeventtb  # event time base
        tpre = 0.002  # 0.1*np.max(tb0)
        tpost = np.max(method.avgeventtb) - tpre
        ipre = int(tpre / rate)
        ipost = int(tpost / rate)
        # tb = np.arange(-tpre, tpost+rate, rate) + tpre
        pt_fivems = int(0.0005 / rate)
        pk_width = int(0.0005 / rate / 2)

        # from pyqtgraph.Qt import QtGui, QtCore
        # import pyqtgraph as pg
        # pg.setConfigOption('leftButtonPan', False)
        # app = QtGui.QApplication([])
        # win = pg.GraphicsLayoutWidget(show=True, title="Basic plotting examples")
        # win.resize(1000,600)
        # win.setWindowTitle('pyqtgraph example: Plotting')
        # # Enable antialiasing for prettier plots
        # pg.setConfigOptions(antialias=True)
        # p0 = win.addPlot(0, 0)
        # p1 = win.addPlot(1, 0)
        # p1.plot(tb, data[:len(tb)]) # whole trace
        # p1s = pg.ScatterPlotItem(size=10, pen=pg.mkPen(None), brush=pg.mkBrush(255, 255, 255, 255))
        # p0s = pg.ScatterPlotItem(size=10, pen=pg.mkPen(None), brush=pg.mkBrush(255, 0, 255, 255))

        for npk, jevent in enumerate(np.array(method.onsets[npks])):
            jstart = jevent - ipre
            jpeak = method.smpkindex[npk]
            jend = jevent + ipost + 1
            evdata = data[jstart:jend].copy()
            l_expect = jend - jstart
            # print('data shape: ', evdata.shape[0], 'expected: ', l_expect)

            if evdata.shape[0] == 0 or evdata.shape[0] < l_expect:
                # print('nodata', evdata.shape[0], l_expect)
                continue
            bl = np.mean(evdata[:pt_fivems])
            evdata -= bl
            # p0.plot(tb_event, evdata)  # plot every event we consider
            # p1s.addPoints(x=[tb[jpeak]], y=[data[jpeak]])
            # next we make a window over which the data will be averaged to test the ampltiude
            left = jpeak - pk_width
            right = jpeak + pk_width
            left = max(0, left)
            right = min(right, len(data))
            if right - left == 0:  # peak and onset cannot be the same
                # p0s.addPoints(x=[tb_event[jpeak-jstart]], y=[evdata[jpeak-jstart]], pen=pg.mkPen('y'), symbolBrush=pg.mkBrush('y'), symbol='o', size=6)
                # print('r - l = 0')
                continue
            if (self.Pars.sign < 0) and (
                np.mean(data[left:right]) > self.Pars.sign * min_event
            ):  # filter events by amplitude near peak
                # p0s.addPoints(x=[tb_event[jpeak-jstart]], y=[evdata[jpeak-jstart]], pen=pg.mkPen('y'), symbolBrush=pg.mkBrush('y'), symbol='o', size=6)
                #  print('data pos, sign neg', np.mean(data[left:right]))
                continue
            if (self.Pars.sign >= 0) and (
                np.mean(data[left:right]) < self.Pars.sign * min_event
            ):
                # p0s.addPoints([tb_event[jpeak-jstart]], [evdata[jpeak-jstart]], pen=pg.mkPen('y'), symbolBrush=pg.mkBrush('y'), symbol='o', size=6)
                # print('data neg, sign pos', np.mean(data[left:right]))
                continue
            # print('dataok: ', jpeak)
            pkt.append(npk)  # build array through acceptance.
        #     p0s.addPoints([tb_event[jpeak-jstart]], [evdata[jpeak-jstart]], pen=pg.mkPen('b'), symbolBrush=pg.mkBrush('b'), symbol='o', size=4)
        # p1s.addPoints(tb[smpks[pkt]], data[smpks[pkt]], pen=pg.mkPen('r'), symbolBrush=pg.mkBrush('r'), symbol='o', size=4)
        # p1.addItem(p1s)
        # p0.addItem(p0s)
        # if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        #     QtGui.QApplication.instance().exec_()

        return pkt

    def filter_data(self, tb: np.ndarray, data: np.ndarray) -> np.ndarray:
        filtfunc = scipy.signal.filtfilt
        samplefreq = 1.0 / self.rate
        nyquistfreq = samplefreq / 1.95
        wn = self.Pars.LPF / nyquistfreq
        b, a = scipy.signal.bessel(2, wn)
        if self.Pars.HPF_flag:
            wnh = self.Pars.HPF / nyquistfreq
            bh, ah = scipy.signal.bessel(2, wnh, btype="highpass")

        imax = int(max(np.where(tb < self.Pars.analysis_window[1])[0]))
        # imax = len(tb)
        data2 = np.zeros_like(data)
        if data.ndim == 3 and self.Pars.LPF_flag:
            if self.Pars.notch_flag:
                CP.cprint("y", f"Notch Filtering Enabled: {str(self.Pars.notch_freqs):s}")
            for r in range(data.shape[0]):
                for t in range(data.shape[1]):
                    data2[r, t, :imax] = filtfunc(
                        b, a, data[r, t, :imax]
                    )  #  - np.mean(data[r, t, 0:250]))
                    if self.Pars.HPF_flag:
                        data2[r, t, :imax] = filtfunc(
                            bh, ah, data2[r, t, :imax]
                        )  #  - np.mean(data[r, t, 0:250]))

                    if self.Pars.notch_flag:
                        data2[r, t, :imax] = FILT.NotchFilterZP(
                            data2[r, t, :imax],
                            notchf=self.Pars.notch_freqs,
                            Q=self.Pars.notch_Q,
                            QScale=False,
                            samplefreq=samplefreq,
                        )
        elif data.ndim == 2 and self.LPF_flag:
            data2 = filtfunc(b, a, data - np.mean(data[0:250]))
            if self.Pars.HPF_flag:
                data2[r, t, :imax] = filtfunc(
                    bh, ah, data2[r, t, :imax]
                )  #  - np.mean(data[r, t, 0:250]))
            if self.Pars.notch_flag:
                if self.Pars.notch_flag:
                    CP.cprint(
                        "y", "Notch Filtering Enabled {str(self.Pars.notch_freqs):s}"
                    )
                data2 = FILT.NotchFilterZP(
                    data2,
                    notchf=self.notch_freqs,
                    Q=self.Pars.notch_Q,
                    QScale=False,
                    samplefreq=samplefreq,
                )
        else:
            return data
        # if self.notch_flag:  ### DO NOT USE THIS WHEN RUNNING PARALLEL MODE
        # f, ax = mpl.subplots(1,1)
        # f.set_figheight(14.)
        # f.set_figwidth(8.)
        # # ax = ax.ravel()
        # for i in range(data.shape[-2]):
        #     ax.plot(tb[:imax], data[0, i,:imax]+i*50e-12, color='grey', linewidth=0.5)
        #     ax.plot(tb[:imax], data2[0, i,:imax]+i*50e-12, 'r-', linewidth=0.3)
        # f2, ax2 = mpl.subplots(1,1)
        # f2.set_figheight(8.)
        # f2.set_figwidth(8.)
        # ax2.magnitude_spectrum(data[0, 0, :imax], Fs=samplefreq, scale='dB', color='k')
        # ax2.magnitude_spectrum(data2[0, 0, :imax], Fs=samplefreq, scale='dB', color='r')
        # # ax2.set_xlim(0., 500.)
        # mpl.show()
        # exit()

        return data2

    """
    Analyze one map calls:
        Analyze protocol; calls
            Analyze one map; calls
                Analyze one trace
    """

    def analyze_one_map(
        self, dataset, plotevents=False, raster=False, noparallel=False, verbose=False
    ) ->Union[None, dict]:
        self.verbose = verbose
        if self.verbose:
            CP.cprint("c", "  ANALYZE ONE MAP")
        self.noparallel = noparallel
        self.data, self.Data.tb, pars, info = self.readProtocol(dataset, sparsity=None)
        if self.data is None or self.Data.tb is None:
            return None
        self.rate = np.mean(np.diff(self.Data.tb))  # t is in seconds, so freq is in Hz
        if self.data is None:  # check that we were able to retrieve data
            self.P = None
            return None
        self.last_dataset = dataset
        if self.Pars.fix_artifact_flag:
            self.Data.data_clean, self.avgdata = self.fix_artifacts(self.data)
            if self.verbose:
                CP.cprint("c", "      Fixing Artifacts")
        else:
            self.Data.data_clean = self.data
        if self.Pars.LPF_flag or self.Pars.notch_flag or self.Pars.HPF_flag:
            if self.verbose:
                CP.cprint("c", f"      LPF Filtering at {self.Pars.LPF:.2f} Hz")
            self.Data.data_clean = self.filter_data(self.Data.tb, self.Data.data_clean)

        stimtimes = []
        data_nostim = []
        # get a list of data points OUTSIDE the stimulus-response window
        lastd = 0  # keeps track of the last valid point
        # print(self.Pars.twin_resp)
        for i, tr in enumerate(self.Pars.twin_resp):  # get window for response
            notokd = np.where((self.Data.tb >= tr[0]) & (self.Data.tb < tr[1]))[0]
            data_nostim.append(list(range(lastd, notokd[0])))
            lastd = notokd[-1]
        # fill end space...
        endindx = np.where(self.Data.tb >= self.Pars.ar_tstart)[0][0]
        data_nostim.append(list(range(lastd, endindx)))
        data_nostim = list(np.hstack(np.array(data_nostim)))
        if self.verbose:
            CP.cprint('c', f"      Data shape going into analyze_protocol: str(elf.data_clean.shape:s)")
        results = self.analyze_protocol(
            self.Data.data_clean,
            self.Data.tb,
            info,
            eventhist=True,
            dataset=dataset,
            data_nostim=data_nostim,
        )
        self.last_results = results
        if self.verbose:
            print("MAP Analyzed")
        return results

    def analyze_protocol(
        self,
        data: np.ndarray,
        tb: np.ndarray,
        info: dict,
        eventstartthr: Union[None, float] = None,
        eventhist: bool = True,
        testplots: bool = False,
        dataset: object = None,  # seems like just passed through
        data_nostim: Union[list, np.ndarray, None] = None,
    ):
        """
        data_nostim is a list of points where the stimulus/response DOES NOT occur, so we can compute the SD
        for the threshold in a consistent manner if there are evoked responses in the trace.
        """
        if self.verbose:
            print("analyze protocol")
        rate = self.rate
        mdata = np.mean(data, axis=0)  # mean across ALL reps
        #        rate = rate*1e3  # convert rate to msec

        # make visual maps with simple scores
        nstim = len(self.Pars.twin_resp)
        self.nstim = nstim
        # find max position stored in the info dict
        pmax = len(list(info.keys()))
        Qr = np.zeros((nstim, data.shape[1]))  # data shape[1] is # of targets
        Qb = np.zeros((nstim, data.shape[1]))
        Zscore = np.zeros((nstim, data.shape[1]))
        I_max = np.zeros((nstim, data.shape[1]))
        pos = np.zeros((data.shape[1], 2))
        infokeys = list(info.keys())
        for ix, t in enumerate(range(data.shape[1])):  # compute for each target
            for s in range(len(self.Pars.twin_resp)):  # and for each stimulus
                Qr[s, t], Qb[s, t] = self.calculate_charge(
                    tb,
                    mdata[t, :],
                    twin_base=self.Pars.twin_base,
                    twin_resp=self.Pars.twin_resp[s],
                )
                Zscore[s, t] = self.ZScore(
                    tb,
                    mdata[t, :],
                    twin_base=self.Pars.twin_base,
                    twin_resp=self.Pars.twin_resp[s],
                )
                I_max[s, t] = (
                    self.Imax(
                        tb,
                        data[0, t, :],
                        twin_base=self.Pars.twin_base,
                        twin_resp=self.Pars.twin_resp[s],
                        sign=self.Pars.sign,
                    )
                    * self.Pars.scale_factor
                )  # just the FIRST pass
            try:
                pos[t, :] = [info[infokeys[ix]]["pos"][0], info[infokeys[ix]]["pos"][1]]
            except:
                CP.cprint(
                    "r",
                    "Failed to establish position for t=%d, ix=%d of max values %d,  protocol: %s"
                    % (t, ix, pmax, self.protocol),
                )
                raise ValueError()
        # print('Position in analyze protocol: ', pos)
        nr = 0

        key1 = []
        key2 = []
        for ix in infokeys:
            k1, k2 = ix
            key1.append(k1)
            key2.append(k2)
        self.nreps = len(set(list(key1)))
        self.nspots = len(set(list(key2)))
        #  print('Repetitions: {0:d}   Spots in map: {1:d}'.format(self.nreps, self.nspots))
        events = {}
        eventlist = []  # event histogram across ALL events/trials
        nevents = 0
        avgevents = []
        if not eventhist:
            return None

        tmaxev = np.max(tb)  # msec
        for jtrial in range(data.shape[0]):  # all trials
            res = self.analyze_one_trial(
                data[jtrial],
                pars={
                    "rate": rate,
                    "jtrial": jtrial,
                    "tmaxev": tmaxev,
                    "eventstartthr": eventstartthr,
                    "data_nostim": data_nostim,
                    "eventlist": eventlist,
                    "nevents": nevents,
                    "tb": tb,
                    "testplots": testplots,
                },
            )
            events[jtrial] = res
        if self.verbose:
            print("  ALL trials in protocol analyzed")
        return {
            "Qr": Qr,
            "Qb": Qb,
            "ZScore": Zscore,
            "I_max": I_max,
            "positions": pos,
            "stimtimes": self.Pars.stimtimes,
            "events": events,
            "eventtimes": eventlist,
            "dataset": dataset,
            "sign": self.Pars.sign,
            "avgevents": avgevents,
            "rate": rate,
            "ntrials": data.shape[0],
        }

    def analyze_one_trial(self, data: np.ndarray, pars: dict = None) -> dict:
        """
        data: numpy array (2D): no default
             data, should be [target, tracelen]; e.g. already points to the trial
        pars: dict
            Dictionary with the following entries:
                rate, jtrial, tmaxev, evenstartthr, data-nostim, eventlist, nevents, tb, testplots
        """
        if self.verbose:
            print("   analyzeone trial")
        nworkers = 7
        tasks = range(
            data.shape[0]
        )  # number of tasks that will be needed is number of targets
        result = [None] * len(tasks)  # likewise
        results = {}
        # print('noparallel: ', self.noparallel)
        if not self.noparallel:
            print("Parallel on all traces in a map")
            with mp.Parallelize(
                enumerate(tasks), results=results, workers=nworkers
            ) as tasker:
                for itarget, x in tasker:
                    result = self.analyze_one_trace(data[itarget], itarget, pars=pars)
                    tasker.results[itarget] = result
            # print('Result keys parallel: ', results.keys())
        else:
            print(" Not parallel...each trace in map in sequence")
            for itarget in range(data.shape[0]):
                results[itarget] = self.analyze_one_trace(
                    data[itarget], itarget, pars=pars
                )
            # print('Result keys no parallel: ', results.keys())
        if self.verbose:
            print("trial analyzed")
        return results

    def analyze_one_trace(
        self, data: np.ndarray, itarget: int, pars: dict = None
    ) -> dict:
        """
        Analyze just one trace

        Parameters
        ----------
        data : 1D array length of trace
            The trace for just one target

        """
        if self.verbose:
            print("      analyze one trace")
        jtrial = pars["jtrial"]
        rate = pars["rate"]
        jtrial = pars["jtrial"]
        tmaxev = pars["tmaxev"]
        eventstartthr = pars["eventstartthr"]
        data_nostim = pars["data_nostim"]
        eventlist = pars["eventlist"]
        nevents = pars["nevents"]
        tb = pars["tb"]
        testplots = pars["testplots"]

        onsets = []  # list of onsete times in this trace
        crit = []  # criteria from CB
        scale = []  # scale from CB
        tpks = []  # time for peaks
        smpks = []  # boxcar smoothed peak values
        smpksindex = []  # matching array indices into smoothed peaks
        avgev = []  # average of events
        avgtb = []  # time base for average events
        avgnpts = []  # numbeer of points in averaged events
        avg_spont = []  # average
        avg_evoked = []  # average evoked events
        measures = []  # simple measures, q, amp, half-width
        fit_tau1 = []  # fit ties to
        fit_tau2 = []
        fit_amp = []
        spont_dur = []
        evoked_ev = []  # subsets that pass criteria, onset values stored
        spont_ev = []
        order = []
        nevents = 0
        if tmaxev > self.Pars.analysis_window[1]:  # block step information
            tmaxev = self.Pars.analysis_window[1]
        idmax = int(self.Pars.analysis_window[1] / rate)

        if self.methodname == "aj":
            aj = minis_methods.AndradeJonas()
            jmax = int((2 * self.Pars.taus[0] + 3 * self.Pars.taus[1]) / rate)
            if self.Pars.LPF_flag is None:
                lpf = None
            else:
                lpf = self.Pars.LPF
            lpf = None
            print("sign: ", self.Pars.sign)
            aj.setup(
                tau1=self.Pars.taus[0],
                tau2=self.Pars.taus[1],
                dt=rate,
                delay=0.0,
                template_tmax=rate * (idmax - 1),  # taus are for template
                sign=self.Pars.sign,
                risepower=4.0,
                threshold=self.Pars.threshold,
            )
            # aj.setup(tau1=self.taus[0], tau2=self.taus[1], dt=rate, delay=0.0, template_tmax=rate*(jmax-1),
            #         sign=self.sign, eventstartthr=eventstartthr, threshold=self.threshold)
            idata = data.view(np.ndarray)  # [jtrial, itarget, :]
            # meandata = np.mean(idata[:idmax])
            aj.deconvolve(
                idata[:idmax], lpf=lpf, llambda=5.0, order=int(0.001 / rate),
            )
            # aj.deconvolve(idata[:idmax]-meandata, data_nostim=data_nostim,
            #                   llambda=1., order=7)  # note threshold scaling...
            method = aj
        elif self.methodname == "cb":
            cb = minis_methods.ClementsBekkers()
            jmax = int((2 * self.Pars.taus[0] + 3 * self.Pars.taus[1]) / rate)
            cb.setup(
                tau1=self.Pars.taus[0],
                tau2=self.Pars.taus[1],
                dt=rate,
                delay=0.0,
                template_tmax=rate * (jmax - 1),
                sign=self.Pars.sign,
                eventstartthr=eventstartthr,
                threshold=self.Pars.threshold,
            )
            cb.set_cb_engine(engine=self.engine)
            idata = data.view(np.ndarray)  # [jtrial, itarget, :]
            meandata = np.mean(idata[:jmax])
            cb.cbTemplateMatch(idata[:idmax] - meandata)
            # result.append(res)
            # crit.append(cb.Crit)
            # scale.append(cb.Scale)
            method = cb
        elif self.methodname == "zc":
            zc = minis_methods.ZCFinder()
            print("sign: ", self.Pars.sign)
            zc.setup(
                dt=rate,
                tau1=self.Pars.taus[0],
                tau2=self.Pars.taus[1],
                sign=self.Pars.sign,
                threshold=self.Pars.threshold,
            )
            idata = data.view(np.ndarray)  # [jtrial, itarget, :]
            jmax = int((2 * self.Pars.taus[0] + 3 * self.Pars.taus[1]) / rate)
            meandata = np.mean(idata[:jmax])
            tminlen = self.Pars.taus[0] + self.Pars.taus[1]
            iminlen = int(tminlen / rate)

            zc.find_events(
                idata[:idmax] - meandata,
                data_nostim=None,
                minPeak=5.0 * 1e-12,
                minSum=5.0 * 1e-12 * iminlen,
                minLength=iminlen,
            )
            method = zc

        else:
            raise ValueError(
                f'analyzeMapData:analyzeOneTrace: Method <{self.methodname:s}> is not valid (use "aj" or "cb" or "zc")'
            )

        # filter out events at times of stimulus artifacts
        # build array of artifact times first
        art_starts = []
        art_durs = []
        art_starts = [
            self.Pars.analysis_window[1],
            self.Pars.shutter_artifact,
        ]  # generic artifacts
        art_durs = [2, 2 * rate]
        if self.Pars.artifact_suppress:
            for si, s in enumerate(self.Pars.stimtimes["start"]):
                if s in art_starts:
                    continue
                art_starts.append(s)
                if isinstance(self.Pars.stimtimes["duration"], float):
                    if self.Pars.stimdur is None:  # allow override
                        art_starts.append(s + self.Pars.stimtimes["duration"])
                    else:
                        art_starts.append(s + self.Pars.stimdur)

                else:
                    if self.Pars.stimdur is None:
                        art_starts.append(s + self.Pars.stimtimes["duration"][si])
                    else:
                        art_starts.append(s + self.Pars.stimdur)
                art_durs.append(2.0 * rate)
                art_durs.append(2.0 * rate)

        # if self.stimdur is not None:
        #     print('used custom stimdur: ', self.stimdur)
        npk0 = self.select_events(
            method.smpkindex, art_starts, art_durs, rate, mode="reject"
        )
        npk4 = self.select_by_sign(
            method, npk0, idata, min_event=5e-12
        )  # events must also be of correct sign and min magnitude
        npk = list(
            set(npk0).intersection(set(npk4))
        )  # only all peaks that pass all tests
        # if not self.artifact_suppress:
        #     npk = npk4  # only suppress shutter artifacts  .. exception

        nevents += len(np.array(method.onsets)[npk])
        # # collate results

        onsets.append(np.array(method.onsets)[npk])
        eventlist.append(tb[np.array(method.onsets)[npk]])
        tpks.append(np.array(method.peaks)[npk])
        smpks.append(np.array(method.smoothed_peaks)[npk])
        smpksindex.append(np.array(method.smpkindex)[npk])
        spont_dur.append(self.Pars.stimtimes["start"][0])  # window until the FIRST stimulus

        if method.averaged:  # grand average, calculated after deconvolution
            avgev.append(method.avgevent)
            avgtb.append(method.avgeventtb)
            avgnpts.append(method.avgnpts)
        else:
            avgev.append([])
            avgtb.append([])
            avgnpts.append(0)

        # define:
        # spont is not in evoked window, and no sooner than 10 msec before a stimulus,
        # at least 4*tau[0] after start of trace, and 5*tau[1] before end of trace
        # evoked is after the stimulus, in a window (usually ~ 5 msec)
        # data for events are aligned on the peak of the event, and go 4*tau[0] to 5*tau[1]
        # stimtimes: dict_keys(['start', 'duration', 'amplitude', 'npulses', 'period', 'type'])
        st_times = np.array(self.Pars.stimtimes["start"])
        ok_events = np.array(method.smpkindex)[npk]
        # print(ok_events*rate)

        npk_ev = self.select_events(
            ok_events,
            st_times,
            self.Pars.response_window,
            rate,
            mode="accept",
            first_only=True,
        )
        ev_onsets = np.array(method.onsets)[npk_ev]
        evoked_ev.append(
            [np.array(method.onsets)[npk_ev], np.array(method.smpkindex)[npk_ev]]
        )

        avg_evoked_one, avg_evokedtb, allev_evoked = method.average_events(ev_onsets)
        fit_tau1.append(
            method.fitted_tau1
        )  # these are the average fitted values for the i'th trace
        fit_tau2.append(method.fitted_tau2)
        fit_amp.append(method.Amplitude)
        avg_evoked.append(avg_evoked_one)
        measures.append(method.measure_events(ev_onsets))
        txb = avg_evokedtb  # only need one of these.
        if not np.isnan(method.fitted_tau1):
            npk_sp = self.select_events(
                ok_events,
                [0.0],
                st_times[0] - (method.fitted_tau1 * 5.0),
                rate,
                mode="accept",
            )
            sp_onsets = np.array(method.onsets)[npk_sp]
            avg_spont_one, avg_sponttb, allev_spont = method.average_events(sp_onsets)
            avg_spont.append(avg_spont_one)
            spont_ev.append(
                [np.array(method.onsets)[npk_sp], np.array(method.smpkindex)[npk_sp]]
            )
        else:
            spont_ev.append([])

        # if testplots:
        #     method.plots(title='%d' % i, events=None)
        res = {
            "criteria": crit,
            "onsets": onsets,
            "peaktimes": tpks,
            "smpks": smpks,
            "smpksindex": smpksindex,
            "avgevent": avgev,
            "avgtb": avgtb,
            "avgnpts": avgnpts,
            "avgevoked": avg_evoked,
            "avgspont": avg_spont,
            "aveventtb": txb,
            "fit_tau1": fit_tau1,
            "fit_tau2": fit_tau2,
            "fit_amp": fit_amp,
            "spont_dur": spont_dur,
            "ntraces": 1,
            "evoked_ev": evoked_ev,
            "spont_ev": spont_ev,
            "measures": measures,
            "nevents": nevents,
        }
        if self.verbose:
            print("      --trace analyzed")
        return res



    def fix_artifacts(self, data: np.ndarray) -> (np.ndarray, np.ndarray):
        """
        Use a template to subtract the various transients in the signal...

        """
        testplot = False
        CP.cprint('c', "Fixing artifacts")
        avgd = data.copy()
        while avgd.ndim > 1:
            avgd = np.mean(avgd, axis=0)
        meanpddata = self.AR.Photodiode.mean(
            axis=0
        )  # get the average PD signal that was recorded
        shutter = self.AR.getBlueLaserShutter()
        dt = np.mean(np.diff(self.Data.tb))
        # if meanpddata is not None:
        #     Util = EP.Utility.Utility()
        #     # filter the PD data - low pass to match data; high pass for apparent oupling
        #     crosstalk = Util.SignalFilter_HPFBessel(meanpddata, 2100., self.AR.Photodiode_sample_rate[0], NPole=1, bidir=False)
        #     crosstalk = Util.SignalFilter_LPFBessel(crosstalk, self.LPF, self.AR.Photodiode_sample_rate[0], NPole=1, bidir=False)
        #     crosstalk -= np.mean(meanpddata[0:int(0.010*self.AR.Photodiode_sample_rate[0])])
        #     crosstalk = np.hstack((np.zeros(1), crosstalk[:-1]))
        # else:
        #     return data, avgd
        protocol = self.protocol.name
        ptype = None

        if self.template_file is None:  # use generic templates for subtraction
            if protocol.find("_VC_10Hz") > 0:
                template_file = "template_data_map_10Hz.pkl"
                ptype = "10Hz"
            elif (
                protocol.find("_single") > 0
                or protocol.find("_Single") > 0
                or (protocol.find("_weird") > 0)
                or (protocol.find("_WCChR2")) > 0
            ):
                template_file = "template_data_map_Singles.pkl"
                ptype = "single"
        else:
            template_file = self.template_file
            if protocol.find("_VC_10Hz") > 0:
                ptype = "10Hz"
            elif (
                protocol.find("_Single") > 0
                or protocol.find("_weird") > 0
                or (protocol.find("_WCChR2")) > 0
            ):
                ptype = "single"
        if ptype is None:
            lbr = np.zeros_like(avgd)
        else:
            CP.cprint("w", f"   Artifact template: {str(template_file):s}")
            CP.cprint("w", f"   Current Working Dir: {os.getcwd():s}")
            with open(template_file, "rb") as fh:
                d = pickle.load(fh)
            ct_SR = np.mean(np.diff(d["t"]))

            # or if from photodiode:
            # ct_SR = 1./self.AR.Photodiode_sample_rate[0]
            crosstalk = d["I"] - np.mean(
                d["I"][0 : int(0.020 / ct_SR)]
            )  # remove baseline
            # crosstalk = self.filter_data(d['t'], crosstalk)
            avgdf = avgd - np.mean(avgd[0 : int(0.020 / ct_SR)])
            # meanpddata = crosstalk
            # if self.shutter is not None:
            #     crossshutter = 0* 0.365e-21*Util.SignalFilter_HPFBessel(self.shutter['data'][0], 1900., self.AR.Photodiode_sample_rate[0], NPole=2, bidir=False)
            #     crosstalk += crossshutter

            maxi = np.argmin(np.fabs(self.Data.tb - self.Pars.analysis_window[1]))
            ifitx = []
            art_times = np.array(self.Pars.stimtimes["start"])
            # known artifacts are:
            # 0.030 - 0.050: Camera
            # 0.050: Shutter
            # 0.055 : Probably shutter actual opening
            # 0.0390, 0.0410: Camera
            # 0.600 : shutter closing
            if ptype == "10Hz":
                other_arts = np.array(
                    [
                        0.030,
                        shutter["start"],
                        0.055,
                        0.390,
                        0.410,
                        shutter["start"] + shutter["duration"],
                    ]
                )
            else:
                other_arts = np.array(
                    [
                        0.010,
                        shutter["start"],
                        0.055,
                        0.305,
                        0.320,
                        shutter["start"] + shutter["duration"],
                    ]
                )

            art_times = np.append(
                art_times, other_arts
            )  # unknown (shutter is at 50 msec)
            art_durs = np.array(self.Pars.stimtimes["duration"])
            other_artdurs = self.Pars.artifact_duration * np.ones_like(other_arts)
            art_durs = np.append(art_durs, other_artdurs)  # shutter - do 2 msec

            for i in range(len(art_times)):
                strt_time_indx = int(art_times[i] / ct_SR)
                idur = int(art_durs[i] / ct_SR)
                send_time_indx = (
                    strt_time_indx + idur + int(0.001 / ct_SR)
                )  # end pulse plus 1 msec
                # avglaser = np.mean(self.AR.LaserBlue_pCell, axis=0) # FILT.SignalFilter_LPFButter(np.mean(self.AR.LaserBlue_pCell, axis=0), 10000., self.AR.sample_rate[0], NPole=8)
                fitx = crosstalk[strt_time_indx:send_time_indx]  # -np.mean(crosstalk)
                ifitx.extend(
                    [
                        f[0] + strt_time_indx
                        for f in np.argwhere((fitx > 0.5e-12) | (fitx < -0.5e-12))
                    ]
                )
            wmax = np.max(np.fabs(crosstalk[ifitx]))
            weights = np.sqrt(np.fabs(crosstalk[ifitx]) / wmax)
            scf, intcept = np.polyfit(crosstalk[ifitx], avgdf[ifitx], 1, w=weights)
            avglaserd = meanpddata  # np.mean(self.AR.LaserBlue_pCell, axis=0)

            lbr = np.zeros_like(crosstalk)
            lbr[ifitx] = scf * crosstalk[ifitx]

        datar = np.zeros_like(data)
        for i in range(data.shape[0]):
            datar[i, :] = data[i, :] - lbr

        if not self.Pars.noderivative_artifact:
            # derivative=based artifact suppression - for what might be left
            # just for fast artifacts
            CP.cprint("w", f"Derivative-based artifact suppression is ON")
            itmax = int(self.Pars.analysis_window[1] / dt)
            avgdr = datar.copy()
            olddatar = datar.copy()
            while olddatar.ndim > 1:
                olddatar = np.mean(olddatar, axis=0)
            olddatar = olddatar - np.mean(olddatar[0:20])
            while avgdr.ndim > 1:
                avgdr = np.mean(avgdr, axis=0)
            diff_avgd = np.diff(avgdr) / np.diff(self.Data.tb)
            sd_diff = np.std(diff_avgd[:itmax])  # ignore the test pulse

            tpts = np.where(np.fabs(diff_avgd) > sd_diff*self.Pars.sd_thr)[0]
            tpts = [t-1 for t in tpts]

            for i in range(datar.shape[0]):
                for j in range(datar.shape[1]):
                    idt = 0
                    #    print(len(tpts))
                    for k, t in enumerate(tpts[:-1]):
                        if (
                            idt == 0
                        ):  # first point in block, set value to previous point
                            datar[i, j, tpts[k]] = datar[i, j, tpts[k] - 1]
                            datar[i, j, tpts[k] + 1] = datar[i, j, tpts[k] - 1]
                            # print('idt = 0, tpts=', t)
                            idt = 1  # indicate "in block"
                        else:  # in a block
                            datar[i, j, tpts[k]] = datar[
                                i, j, tpts[k] - 1
                            ]  # blank to previous point
                            datar[i, j, tpts[k] + 1] = datar[
                                i, j, tpts[k] - 1
                            ]  # blank to previous point
                            if (
                                tpts[k + 1] - tpts[k]
                            ) > 1:  # next point would be in next block?
                                idt = 0  # reset, no longer in a block
                                datar[i, j, tpts[k] + 1] = datar[
                                    i, j, tpts[k]
                                ]  # but next point is set
                                datar[i, j, tpts[k] + 2] = datar[
                                    i, j, tpts[k]
                                ]  # but next point is set

        if testplot:
            PMD.testplot()
        
        return(datar, avgd)

    def reorder(self, a: float, b: float):
        """
        make sure that b > a
        if not, swap and return
        """
        if a > b:
            t = b
            b = a
            a = t
        return (a, b)

    def shortName(self, name: Union[Path, str]) -> Union[Path, str]:
        (h, pr) = os.path.split(name)
        (h, cell) = os.path.split(h)
        (h, sliceno) = os.path.split(h)
        (h, day) = os.path.split(h)
        return os.path.join(day, sliceno, cell, pr)

    def save_pickled(self, dfile: str, data: np.ndarray) -> None:
        now = datetime.datetime.now().isoformat()
        dstruct = {
            "date": now,
            "data": data,
        }
        print("\nWriting to {:s}".format(dfile))
        fn = open(dfile, "wb")
        pickle.dump(dstruct, fn)
        fn.close()

    def read_pickled(self, dfile: str) -> object:
        fn = open(dfile + ".p", "rb")
        data = pickle.load(fn)
        fn.close()
        return data




if __name__ == "__main__":
    # these must be done here to avoid conflict when we import the class, versus
    # calling directly for testing etc.
    matplotlib.use("Agg")
    rcParams = matplotlib.rcParams
    rcParams["svg.fonttype"] = "none"  # No text as paths. Assume font installed.
    rcParams["pdf.fonttype"] = 42
    rcParams["ps.fonttype"] = 42
    rcParams["text.latex.unicode"] = True
    # rcParams['font.family'] = 'sans-serif'
    rcParams["font.weight"] = "regular"  # you can omit this, it's the default
    # rcParams['font.sans-serif'] = ['Arial']

    datadir = "/Volumes/PBM_004/data/MRK/Pyramidal"
    parser = argparse.ArgumentParser(description="mini synaptic event analysis")
    parser.add_argument("datadict", type=str, help="data dictionary")
    parser.add_argument(
        "-s",
        "--scale",
        type=float,
        default=0.0,
        dest="scale",
        help="set maximum scale for overlay plot (default=0 -> auto)",
    )
    parser.add_argument(
        "-i", "--IV", action="store_true", dest="do_iv", help="just do iv"
    )
    parser.add_argument(
        "-o", "--one", type=str, default="", dest="do_one", help="just do one"
    )
    parser.add_argument(
        "-m", "--map", type=str, default="", dest="do_map", help="just do one map"
    )
    parser.add_argument(
        "-c", "--check", action="store_true", help="Check for files; no analysis"
    )
    # parser.add_argument('-m', '--mode', type=str, default='aj', dest='mode',
    #                     choices=['aj', 'cb'],
    #                     help='just do one')
    parser.add_argument(
        "-v", "--view", action="store_false", help="Turn off pdf for single run"
    )

    args = parser.parse_args()

    filename = os.path.join(datadir, args.datadict)
    if not os.path.isfile(filename):
        print("File not found: %s" % filename)
        exit(1)

    DP = EP.DataPlan.DataPlan(os.path.join(datadir, args.datadict))  # create a dataplan
    plan = DP.datasets
    print("plan dict: ", plan.keys())
    # print('plan: ', plan)
    if args.do_one != "":
        cellid = int(args.do_one)
    else:
        raise ValueError("no cell id found for %s" % args.do_one)
    cell = DP.excel_as_df[DP.excel_as_df["CellID"] == cellid].index[0]

    print("cellid: ", cellid)
    print("cell: ", cell)

    print("cell: ", plan[cell]["Cell"])
    datapath = os.path.join(
        datadir,
        str(plan[cell]["Date"]).strip(),
        str(plan[cell]["Slice"]).strip(),
        str(plan[cell]["Cell"]).strip(),
    )
    # print( args)

    if args.do_iv:

        EPIV = EP.IVSummary.IVSummary(
            os.path.join(datapath, str(plan[cell]["IV"]).strip())
        )
        EPIV.compute_iv()
        print("cell: ", cell, plan[cell]["Cell"])
        DP.post_result("CellID", cellid, "RMP", EPIV.RM.analysis_summary["RMP"])
        DP.post_result("CellID", cellid, "Rin", EPIV.RM.analysis_summary["Rin"])
        DP.post_result("CellID", cellid, "taum", EPIV.RM.analysis_summary["taum"])
        now = datetime.datetime.now()
        DP.post_result("CellID", cellid, "IV Date", str(now.strftime("%Y-%m-%d %H:%M")))
        DP.update_xlsx(os.path.join(datadir, args.datadict), "Dataplan")
        exit(1)

    if args.do_map:
        getimage = True
        plotevents = True
        dhImage = None
        rotation = 0
        AM = AnalyzeMap()
        AM.sign = plan[cell]["Sign"]
        AM.overlay_scale = args.scale
        AM.display_one_map(
            os.path.join(datapath, str(plan[cell]["Map"]).strip()),
            justplot=False,
            imagefile=os.path.join(datapath, plan[cell]["Image"]) + ".tif",
            rotation=rotation,
            measuretype="ZScore",
        )
        mpl.show()
