from __future__ import absolute_import

"""
analyze ChR2 or uncaging map data

"""
import argparse
import datetime
import math
import os.path
import re
import sqlite3
import sys
import timeit
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple, Union

import dill as pickle
import matplotlib
import matplotlib.pyplot as mpl
import numpy as np
import pandas as pd
import pylibrary.tools.cprint as CP
import pyqtgraph.multiprocess as mp
import scipy.ndimage
import scipy.signal

import ephys.datareaders as DR
import ephys.ephys_analysis as EP
import ephys.mini_analyses.mini_analysis as mini_analysis
import ephys.mini_analyses.mini_event_dataclasses as MEDC  # get result datastructure
import ephys.tools.digital_filters as FILT
import ephys.tools.functions as functions
from ephys.mapanalysistools import compute_scores
from ephys.mapanalysistools import plot_map_data as PMD
from ephys.mini_analyses import minis_methods

re_degree = re.compile(r"\s*(\d{1,3}d)\s*")
re_duration = re.compile(r"(\d{1,3}ms)")
np.seterr(divide="raise")


def def_notch():
    """defaults for notch frequencies"""
    return [60.0, 120.0, 180.0, 240.0]


def def_stimtimes():
    return {"start": [0.3], "duration": [1e-3]}

def def_steptimes():
    return {"start": [0.6], "duration": [1e-2]}

def def_twin_base():
    return [0.0, 0.295]


def def_twin_response():
    return [0.301, 0.325]


def def_taus():
    return [0.0002, 0.005, 0.005, 0.030]


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

    dt_seconds: float = 2e-5  # sample rate in seconds
    spotsize: float = 42e-6  # spot size in meters
    baseline_flag: bool = False
    baseline_subtracted: bool = False
    filters: object = field(default_factory=MEDC.Filtering)
    fix_artifact_flag: bool = False  # flag enabling removeal of artifacts
    artifact_file: Union[Path, None] = None
    artifact_file_path: Union[Path, None] = None
    ar_tstart: float = 0.10  # starting time for VC or IC stimulus pulse
    ar_tend: float = 0.015  # end time for VC or IC stimulus pulse
    time_zero: float = 0.0  # in seconds, where we start the trace
    time_zero_index: int = 0
    time_end: float = 1.0  # in seconds, end time of the trace
    time_end_index: int = 0  # needs to be set from the data
    stimtimes: dict = field(default_factory=def_stimtimes)
    currentsteptimes: dict = field(default_factory=def_steptimes)
    spont_deadtime: float = (
        0.010  # time after trace onset before counting spont envents
    )
    direct_window: float = 0.0005  # window after stimulus for direct response
    response_window: float = 0.015  # window end for response (response is between direct and response), seconds
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
    risepower: float = 4.0
    taus: list = field(
        default_factory=def_taus
    )  # initial taus for fitting detected events
    threshold: float = 3.0  # default threshold for CB, AJ or ZC
    sign: int = -1  # negative for EPSC, positive for IPSC
    datatype: str = "V"  # data type - VC or IC
    stepi: float = (
        25.0  # step size for stacked traces, in pA (25 default for cc; 2 for cc)
    )
    scale_factor: float = 1.0  # scale factore for data (convert to pA or mV,,, )
    overlay_scale: float = 0.0
    shutter_artifact: list = field(
        default_factory=def_shutter_artifacts
    )  # time of shutter electrical artifat
    artifact_suppress: bool = True  # flag enabling suppression of artifacts
    artifact_duration: float = 2e-3  # length of artifat, in seconds
    stimdur: Union[float, None] = None  # duration of stimuli
    noderivative_artifact: bool = (
        True  # flag enableing artifact suppression based on derivative of trace
    )
    sd_thr: float = 3.0  # threshold in sd for diff based artifact suppression.
    global_SD: float = 0.0  # raw global SD
    global_mean: float = 0.0  # raw global mean
    global_trim_scale: float = 3.0
    global_trimmed_SD: float = 0.0  # raw global trimeed SD with outliers > 3.0
    global_trimmed_median: float = 0.0


@dataclass
class AnalysisData:
    """
    Data class that holds the analysis data separate from the parameters
    and metadata
    This class is also made available to the plotMapdata routines
    """

    timebase: Union[None, np.ndarray] = None  # the time base corresponding to the clean data
    data_clean: Union[None, np.ndarray] = None  # data with filtering, detrending, artifact removal
    photodiode: Union[None, np.ndarray] = None  # photodiode current trace (monitor)
    photodiode_timebase: Union[None, np.ndarray] = None # photodiode time base matching current trace
    MA: Union[object, None] = None  # point to minanalysis instance used for analysis


class AnalyzeMap(object):
    def __init__(self, rasterize=True):
        self.filters = MEDC.Filtering()
        self.reset()
        self.tstart = 0
        self.tend = 0
        self.timebase = None

    def reset(self):
        self.Pars = AnalysisPars()
        self.Data = AnalysisData()

        self.verbose = True
        self.last_dataset = None
        self.last_results = None
        self.lbr_command = False  # laser blue raw waveform (command)

        # set some defaults - these will be overwrittein with readProtocol
        self.template_file = None

        self.methodname = "aj"  # default event detector
        self.set_methodname(self.methodname)
        self.reset_filters()

    def configure(
        self,
        reader: Union[object, None] = None,
        spikeanalyzer: Union[object, None] = None,
        rmtauanalyzer: Union[object, None] = None,
        minianalyzer: Union[object, None] = None,
    ):
        self.AR = reader
        self.SP = spikeanalyzer  # spike_analysis.SpikeAnalysis()
        self.RM = rmtauanalyzer  # rm_tau_analysis.RmTauAnalysis()
        self.MA = minianalyzer

    def set_analysis_window(self, t0: float = 0.0, t1: Union[float, None] = None):
        assert t1 is not None  # force usage of t1
        self.Pars.analysis_window = [t0, t1]
        CP.cprint(
            "r",
            f"analyze_map_data: set_analysis_window:  {str(self.Pars.analysis_window):s}",
        )

    def set_filters(self, filters):
        """Set the filtering arguments

        Args:
            filters (dataclass): Class structure with filtering information
        """
        if filters is None:
            self.filters = MEDC.Filtering()
        else:
            self.filters = filters
        self.reset_filters()
        print("Filter set in AMD: \n", self.filters)

    def reset_filters(self):
        """
        Reset the filtering flags so we know which have been done.
        The purpose of this is to keep from applying filters repeatedly
        """
        self.filters.enabled = True
        self.filters.Detrend_applied = False
        self.filters.LPF_applied = False
        self.filters.HPF_applied = False
        self.filters.Notch_applied = False

    def set_notch(self, notch, freqs=[60], Q=90.0):
        self.filters.Notch_applied = False
        self.filters.Notch_frequencies = freqs
        self.filters.Notch_Q = Q

    def set_LPF(self, LPF):
        self.filters.LPF_frequency = LPF
        self.filters.LPF_applied = False
        self.filters.LPF_type = "ba"

    def set_HPF(self, HPF):
        self.filters.HPF_frequency = HPF
        self.filters.HPF_applied = False
        self.filters.HPF_type = "ba"

    def reset_filtering(self):
        self.filters.LPF_applied = False
        self.filters.HPF_applied = False
        self.filters.Notch_applied = False

    def set_artifactfile_path(self, artpath):
        self.Pars.artifact_file_path = artpath

    def set_baseline(self, bl):
        self.Pars.baseline_flag = bl

    def set_methodname(self, methodname):
        if methodname.lower() in ["aj"]:
            self.methodname = "aj"
            self.engine = "python"
        elif methodname.lower() in ["aj_cython"]:
            self.methodname = "aj"
            self.engine = "cython"
        elif methodname.lower() in ["cb"]:
            self.methodname = "cb"
            self.engine = "cython"  # set this as the default
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
        if len(taus) != 4:
            raise ValueError(
                "Analyze Map Data: need two tau values in list to sort!, got: ", taus
            )
        self.Pars.taus[0:2] = sorted(taus[0:2])
        self.Pars.taus[2:4] = sorted(taus[2:4])

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
        self.template_file = filename

    def readProtocol(
        self, protocolFilename, records=None, sparsity=None, getPhotodiode=False
    ):
        self.AR = DR.acq4_reader.acq4_reader()
        starttime = timeit.default_timer()
        self.protocol = protocolFilename
        CP.cprint("g", f"Reading Protocol:: {str(protocolFilename):s}")
        self.AR.setProtocol(protocolFilename)
        if not protocolFilename.is_dir() or not self.AR.getData(allow_partial=True):
            CP.cprint("r", f"**** No data found in protocol: {str(protocolFilename):s}")
            return None
        # print('Protocol: ', protocolFilename)
        self.Pars.datatype = self.AR.mode[0].upper()  # get mode and simplify to I or V
        if self.Pars.datatype == "I":
            self.Pars.stepi = 2.0
        # otherwise use the default, which is set in the init routine

        # find if there is a stimulus step to consider
        if self.AR.tstart is not None and self.AR.tstart > 0.1:  # pulse at end of trace
            self.Pars.analysis_window[1] = self.AR.tstart - 0.010
        elif self.AR.tstart is not None and self.AR.tstart < 0.1: # pulse at start of trace
            self.Pars.analysis_window[0] = self.AR.tend + 0.010

        # get the laser pulse times
        self.AR.getLaserBlueTimes()
        self.Pars.stimtimes = self.AR.LaserBlueTimes
        self.Pars.ar_tstart = self.AR.tstart
        self.Pars.ar_tend = self.AR.tend
        self.Pars.dt_seconds = 1.0 / self.AR.sample_interval

        # Adjust the time limits for the data analysis to exclude any
        # vc or ic monitoring pulises that might be before or after the data.
        # This is done to clean up the filtering that takes place later
        self.Pars.time_end = np.max(self.AR.time_base)
        self.Pars.time_end_index = int(self.Pars.time_end * self.Pars.dt_seconds)
        self.Pars.time_zero = 0.0
        self.Pars.time_zero_index = 0
        if (
            self.Pars.ar_tend < self.Pars.stimtimes["start"][0]
        ):  # VC/IC pulse precedes stimuli
            self.Pars.time_zero = (
                self.Pars.ar_tend + 0.010
            )  # give it 10 msec to settle down
            self.Pars.time_zero_index = int(self.Pars.time_zero * self.Pars.dt_seconds)
            if self.Pars.twin_base[0] < self.Pars.time_zero:
                self.Pars.twin_base[0] = self.Pars.time_zero
        elif (
            self.Pars.ar_tstart > self.Pars.stimtimes["start"][-1]
        ):  # after the last stimulus:
            self.Pars.time_end = (
                self.Pars.ar_tstart - 0.001
            )  # end 1 msec before end of trace
            self.Pars.time_end_index = int(self.Pars.time_end * self.Pars.dt_seconds)
        self.Data.timebase = self.AR.time_base[
            self.Pars.time_zero_index : self.Pars.time_end_index
        ]

        if self.Pars.stimtimes is not None:
            self.Pars.twin_base = [
                0.0,
                self.Pars.stimtimes["start"][0] - 0.001 - self.Pars.time_zero,
            ]  # remember times are in seconds
            self.Pars.twin_resp = []
            for j in range(len(self.Pars.stimtimes["start"])):
                self.Pars.twin_resp.append(
                    [
                        self.Pars.stimtimes["start"][j]
                        + self.Pars.direct_window
                        - self.Pars.time_zero,
                        self.Pars.stimtimes["start"][j]
                        + self.Pars.response_window
                        - self.Pars.time_zero,
                    ]
                )
        if self.AR.getLaserBlueCommand():
            self.Data.laser_blue_pCell = self.AR.LaserBlue_pCell[
                self.Pars.time_zero_index : self.Pars.time_end_index
            ]
            self.Data.laser_blue_timebase = self.AR.LaserBlue_time_base[
                self.Pars.time_zero_index : self.Pars.time_end_index
            ]
            self.Data.laser_blue_sample_rate = self.AR.LaserBlue_sample_rate
        else:
            CP.cprint("r", "**** Could not get blue laser command traces")

        if self.AR.getPhotodiode():
            self.Data.photodiode = self.AR.Photodiode[
                self.Pars.time_zero_index : self.Pars.time_end_index
            ]
            self.Data.photodiode_timebase = self.AR.Photodiode_time_base[
                self.Pars.time_zero_index : self.Pars.time_end_index
            ]
        else:
            CP.cprint("r", "**** Could not get photodiode traces")

        self.shutter = self.AR.getDeviceData("Laser-Blue-raw", "Shutter")
        self.AR.getScannerPositions()
        print("-" * 46)
        print("Scanning array shape;  min and max x,y limits")
        print("  Shape: ", self.AR.scanner_positions.shape)
        print(f"  Min:  {np.min(self.AR.scanner_positions[:, 0]):.6f}, {np.min(self.AR.scanner_positions[:, 1]):.6f}")
        print(f"  Max:  {np.max(self.AR.scanner_positions[:, 0]):.6f}, {np.max(self.AR.scanner_positions[:, 1]):.6f}")
        print("-" * 46)
 
        self.Pars.spotsize = self.AR.scanner_spotsize
        if "LED" in str(protocolFilename):
            self.Pars.spotsize = 1e-4
        data = self.AR.data_array.copy()
        if self.AR.repetitions > 1:
            print("reshape array to account for repetitions")
            data = np.reshape(
                data,
                (
                    self.AR.repetitions,
                    int(data.shape[0] / self.AR.repetitions),
                    self.AR.traces.shape[1],
                ),
            )
        else:
            data = data[np.newaxis, ...]
        data = data[
            :, :, self.Pars.time_zero_index : self.Pars.time_end_index
        ]  # clip the data to the analysis window
        endtime = timeit.default_timer()
        CP.cprint(
            "g",
            "    Reading protocol {0:s} took {1:6.1f} s".format(
                protocolFilename.name, endtime - starttime
            ),
        )
        return data

    def calculate_charge(
        self,
        timebase: np.ndarray,
        data: np.ndarray,
        twin_base: list = [0, 0.1],
        twin_resp: list = [[0.101, 0.130]],
    ) -> Tuple[float, float]:
        """
        Integrate current over a time window to get charges

        Returns two charge measures: the baseline, and the value
        in the response window
        """
        # get indices for the integration windows
     
        tbindx = np.where((timebase >= twin_base[0]) & (timebase < twin_base[1]))
        trindx = np.where((timebase >= twin_resp[0]) & (timebase < twin_resp[1]))
        # print("amd:calculate charge: ", twin_resp)
        Qr = 1e6 * np.sum(data[trindx]) / (twin_resp[1] - twin_resp[0])  # response
        Qb = 1e6 * np.sum(data[tbindx]) / (twin_base[1] - twin_base[0])  # baseline
        return Qr, Qb

    """
    Analyze one map calls:
        Analyze protocol; calls
    """

    def analyze_one_map(
        self,
        mapdir: Union[str, Path] = None,
        plotevents=False,
        raster=False,
        noparallel=False,
        verbose=False,
    ) -> Union[None, dict]:
        """_summary_

        Args:
            mapdir (Union[str, Path]): Directory of map protocol
            plotevents (bool, optional): Flag to allow plotting of events. Defaults to False.
            raster (bool, optional): Flag to cause plot output to be rasterized rather than vectorized. Defaults to False.
            noparallel (bool, optional): Cause to run in parallel mode for each trace. Defaults to False.
            verbose (bool, optional): If True, print out a lot of debugging stuff. Defaults to False.

        Returns:
            Union[None, dict]: _description_
        """
        self.verbose = verbose

        # self.MA = minis_methods.MiniAnalyses()  # get a minianalysis instance
        self.AR = (
            DR.acq4_reader.acq4_reader()
        )  # make our own private cersion of the analysis and reader

        self.mod_data = self.readProtocol(mapdir)
        # self.info = self.AR.getDataInfo(Path(mapdir))  # copy out the info
        if self.mod_data is None:
            CP.cprint("r", f"Unable to read the file: {str(mapdir):s}")
            return None
            datasource: str = "",

        self.MA.setup(
            datasource="analyze_map_data",
            ntraces=self.mod_data.shape[0],
            tau1=self.Pars.taus[0],
            tau2=self.Pars.taus[1],
            tau3=self.Pars.taus[2],
            tau4=self.Pars.taus[3],
            template_tmax= 0.05, # sec
            template_pre_time= 0.001, # sec
            dt_seconds=self.AR.sample_interval,
            sign=self.Pars.sign,
            risepower=self.Pars.risepower,
            threshold=self.Pars.threshold,
            analysis_window = self.Pars.analysis_window,
            filters=self.filters,
        )
        if self.verbose:
            CP.cprint("c", "  ANALYZE ONE MAP")
        self.noparallel = noparallel
        self.rate = self.AR.sample_rate[0]  # sample frequency in Hz
        self.last_dataset = mapdir
        if self.Pars.fix_artifact_flag:
            self.mod_data, self.avgdata = self.fix_artifacts(self.mod_data)
            CP.cprint("c", "        Fixing Artifacts")

        self.Data.data_clean = []
        for i in range(self.mod_data.shape[0]):
            self.MA.prepare_data(self.mod_data[i])
            self.Data.data_clean.append(self.MA.data)
        self.Data.data_clean = np.array(self.Data.data_clean)
        self.Data.timebase = self.MA.timebase

        data_nostim = []
    
        if self.verbose:
            CP.cprint(
                "c",
                f"        Data shape going into analyze_protocol: str(self.data_clean.shape:s)",
            )
        print("before analyze protocol, timebase max is: ", np.max(self.Data.timebase))
        results = self.analyze_protocol(
            data=self.Data.data_clean,
            timebase=self.Data.timebase,
            #   info = self.info,
            eventhist=True,
            dataset=mapdir,
            data_nostim=data_nostim,
        )
        self.last_results = results
        if self.verbose:
            print("MAP Analyzed")
        return results

    def analyze_protocol(
        self,
        data: np.ndarray,
        timebase: np.ndarray,
        #    info: dict,
        eventstartthr: Union[None, float] = None,
        eventhist: bool = True,
        testplots: bool = False,
        dataset: object = None,  # seems like just passed through
        data_nostim: Union[list, np.ndarray, None] = None,
    ):
        """
        analyze_protocol calls:
            analyze_one_trial (repetition)

        data_nostim is a list of points where the stimulus/response DOES NOT occur, so we can compute the SD
        for the threshold in a consistent manner if there are evoked responses in the trace.

        """
        CP.cprint("g", "    Analyzing protocol")
        print("-" * 40)
        rate = self.rate
        mdata = np.mean(data, axis=0)  # mean across ALL reps
        #        rate = rate*1e3  # convert rate to msec

        # make visual maps with simple scores
        nstim = len(self.Pars.twin_resp)
        self.nstim = nstim
        # find max position stored in the info dict
        Qr = np.zeros((nstim, data.shape[1]))  # data shape[1] is # of targets
        Qb = np.zeros((nstim, data.shape[1]))
        zscore = np.zeros((nstim, data.shape[1]))
        I_max = np.zeros((nstim, data.shape[1]))
        pos = np.zeros((data.shape[1], 2))
        # infokeys = list(info.keys())
        # timebase = timebase - self.Pars.time_zero
        # print(np.min(tb), np.max(tb), self.Pars.twin_base, self.Pars.twin_resp)

        for ix, t in enumerate(range(data.shape[1])):  # compute for each target
            for s in range(len(self.Pars.twin_resp)):  # and for each stimulus
                print(t, s)
                print(self.Pars.twin_base, self.Pars.twin_resp[s])
                Qr[s, t], Qb[s, t] = self.calculate_charge(
                    timebase=timebase,
                    data=mdata[t, :],
                    twin_base=self.Pars.twin_base,
                    twin_resp=self.Pars.twin_resp[s],
                )
                zscore[s, t] = compute_scores.ZScore(
                    timebase=timebase,
                    data=mdata[t, :],
                    twin_base=self.Pars.twin_base,
                    twin_resp=self.Pars.twin_resp[s],
                )
                I_max[s, t] = (
                    compute_scores.Imax(
                        timebase=timebase,
                        data=data[t, :],
                        twin_base=self.Pars.twin_base,
                        twin_resp=self.Pars.twin_resp[s],
                        sign=self.Pars.sign,
                    )
                    * self.Pars.scale_factor
                )  # just the FIRST pass
            # print("info: ", info[infokeys[ix]])
            try:
                pos[t, :] = self.AR.scanner_positions[
                    t
                ]  # [info[infokeys[ix]]["pos"][0], info[infokeys[ix]]["pos"][1]]
            except:
                CP.cprint(
                    "r",
                    "**** Failed to establish position for t=%d, ix=%d of  protocol: %s"
                    % (t, ix, self.protocol),
                )
                raise ValueError()
        # print('Position in analyze protocol: ', pos)
        nr = 0
        # key1 = []
        # key2 = []
        # for ix in infokeys:
        #     k1, k2 = ix
        #     key1.append(k1)
        #     key2.append(k2)
        self.nreps = data.shape[0]  # len(set(list(key1)))
        self.nspots = len(self.AR.scanner_positions)
        events = {}
        eventlist = []  # event histogram across ALL events/trials
        nevents = 0
        avgevents = []
        if not eventhist:
            return None

        tmaxev = np.max(timebase)  # msec
        for jtrial in range(data.shape[0]):  # all trials
            CP.cprint("g", f"Analyzing Trial # {jtrial:4d}")
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
                    "timebase": timebase,
                    "testplots": testplots,
                },
                datatype=self.Pars.datatype,
            )
            events[jtrial] = res
            # print("Trial results: ", res)

        if self.verbose:
            print("  ALL trials in protocol analyzed")
        return {
            "analysis_parameters": self.Pars,  # save the analysis parameters (added 9/2020)
            "engine": self.engine,
            "method": self.methodname,
            "Qr": Qr,
            "Qb": Qb,
            "ZScore": zscore,
            "I_max": I_max,
            "positions": pos,
            "stimtimes": self.Pars.stimtimes,
            "events": events,
            # "eventtimes": res.onsets,
            "dataset": dataset,
            "sign": self.Pars.sign,
            # "avgevents": res.average,
            "rate": rate,
            "ntrials": data.shape[0],
        }

    def analyze_one_trial(
        self, data: np.ndarray, pars: dict = None, datatype: str = None
    ) -> dict:
        """Analyze one trial in a protocol (one map; maps may have been repeated)

        analyze_one_trial calls
        analyze_traces_in_trial
        and returns "method" (the class that analyzed the data in a trial)

        Parameters
        ----------
        data: numpy array (2D): no default
             data, should be [target, tracelen]; e.g. already points to the trial

        pars: dict
            Dictionary with the following entries:
            rate, jtrial, tmaxev, evenstartthr, data-nostim, eventlist, nevents, timebase, testplots

        datatype: str
            Data type (IC, VC)
        """
        if self.verbose:
            print("   analyze one trial")
        nworkers = 7
        tasks = range(
            data.shape[0]
        )  # number of tasks that will be needed is number of targets
        # result = [None] * len(tasks)  # likewise
        # results = {}
        # print('noparallel: ', self.noparallel)
        # if not self.noparallel:
        #     print("Parallel on all traces in a map")
        #     with mp.Parallelize(
        #         enumerate(tasks), results=results, workers=nworkers
        #     ) as tasker:
        #         for itarget, x in tasker:
        #             result = self.analyze_one_trace(data[itarget], itarget, pars=pars)
        #             tasker.results[itarget] = result
        #     # print('Result keys parallel: ', results.keys())
        # else:
        #     print("Not parallel...each trace in map in sequence")
        #     for itarget in range(data.shape[0]):
        #         results[itarget] = self.analyze_traces(
        #             data[itarget], itarget, pars=pars
        #         )
        # print('Result keys no parallel: ', results.keys())

        method = self.analyze_traces_in_trial(data, pars=pars, datatype=datatype)
        method.identify_events(verbose=True)  # order=order)
        summary = method.summarize(data, verbose=True)
        ok_onsets = method.get_data_cleaned_of_stimulus_artifacts(
            data, summary=summary, pars=self.Pars
        )
        summary.ok_onsets = ok_onsets
        summary.spont_dur = [self.Pars.stimtimes["start"][0]] * data.shape[0]
        summary = method.average_events(
            traces=range(data.shape[0]), data=data, summary=summary
        )
        summary = self.average_trial_events(
            method, data=data, minisummary=summary, pars=self.Pars
        )

        if len(summary.average.avgevent) == 0:
            return None

        method.fit_average_event(
            summary.average.avgeventtb,
            summary.average.avgevent,
            inittaus=self.Pars.taus,
        )

        if self.verbose:
            print("    Trial analyzed")

        return summary

    def analyze_traces_in_trial(
        self,
        data: np.ndarray,
        pars: dict = None,
        datatype: str = None,
    ) -> dict:
        """
        Analyze the block of traces
        Calls the methods in minis_methods that are avalable for
        detecting events.

        Parameters
        ----------
        data : 1D array length of trace
            The trace is for just one target, one trial

        pars : dict

        datatype: str
            Data type (IC or VC)
        Returns
        -------
        The method class that was used (and which holds the
        various results)

        """
        if self.verbose:
            print("      analyze traces")
        CP.cprint("g", f"     Analyzing {data.shape[0]:4d} traces")
        idmax = int(self.Pars.analysis_window[1] * self.rate)

        # get the lpf and HPF settings - if they were used

        if self.methodname == "aj":
            aj = minis_methods.AndradeJonas()
            jmax = int((2 * self.Pars.taus[0] + 3 * self.Pars.taus[1]) * self.rate)
            aj.setup(
                datasource=self.protocol,
                ntraces=data.shape[0],
                tau1=self.Pars.taus[0],
                tau2=self.Pars.taus[1],
                tau3=self.Pars.taus[2],
                tau4=self.Pars.taus[3],  # taus are for template
                dt_seconds=1.0 / self.rate,
                delay=0.0,
                template_tmax=self.Pars.analysis_window[1],
                sign=self.Pars.sign,
                risepower=4.0,
                threshold=self.Pars.threshold,
                global_SD=self.Pars.global_trimmed_SD,
                filters=self.filters,
            )
            aj.set_datatype(datatype)
            idata = data.view(np.ndarray)
            # This loop can be parallelized

            for i in range(data.shape[0]):
                aj.deconvolve(
                    idata[i][:idmax],
                    timebase = self.Data.timebase,
                    itrace=i,
                    llambda=5.0,
                    prepare_data=False,
                    # order=int(0.001 / self.rate),
                )
            return aj

        elif self.methodname == "cb":
            cb = minis_methods.ClementsBekkers()
            jmax = int((2 * self.Pars.taus[0] + 3 * self.Pars.taus[1]) / self.rate)
            cb.setup(
                datasource=self.protocol,
                ntraces=data.shape[0],
                tau1=self.Pars.taus[0],
                tau2=self.Pars.taus[1],
                tau3=self.Pars.taus[2],
                tau4=self.Pars.taus[3],
                dt_seconds=1.0 / self.rate,
                delay=0.0,
                template_tmax=5.0 * self.Pars.taus[1],  # rate * (jmax - 1),
                sign=self.Pars.sign,
                # eventstartthr=eventstartthr,
                threshold=self.Pars.threshold,
                filters=self.filters,
            )
            cb.set_datatype(datatype)
            cb.set_cb_engine(engine=self.engine)
            cb._make_template()
            idata = data.view(np.ndarray)  # [jtrial, itarget, :]

            for i in range(data.shape[0]):
                cb.cbTemplateMatch(idata[i][:idmax], itrace=i, prepare_data=False)
            return cb

        else:
            raise ValueError(
                f'analyzeMapData:analyzetracesintrial: Method <{self.methodname:s}> is not valid (use "aj" or "cb")'
            )

    def get_artifact_times(self):
        """Compute the artifact times and store them in a structure

        Returns:
            artifacts structure: dataclass holding artifact times (starts, durations)
        """
        # set up parameters for artifact exclusion
        artifacts = MEDC.Artifacts()
        artifacts.starts = [
            self.Pars.analysis_window[1],
            self.Pars.shutter_artifact,
        ]  # generic artifacts
        artifacts.durations = [2, 2 * self.Pars.dt_seconds]
        if self.Pars.artifact_suppress:
            for si, s in enumerate(self.Pars.stimtimes["start"]):
                if s in artifacts.starts:
                    continue
                artifacts.starts.append(s)
                if isinstance(self.Pars.stimtimes["duration"], float):
                    if pd.isnull(self.Pars.stimdur):  # allow override
                        artifacts.starts.append(s + self.Pars.stimtimes["duration"])
                    else:
                        artifacts.starts.append(s + self.Pars.stimdur)

                else:
                    if pd.isnull(self.Pars.stimdur):
                        artifacts.starts.append(s + self.Pars.stimtimes["duration"][si])
                    else:
                        artifacts.starts.append(s + self.Pars.stimdur)
                artifacts.durations.append(2.0 * self.Pars.dt_seconds)
                artifacts.durations.append(2.0 * self.Pars.dt_seconds)

        return artifacts

    def average_trial_events(
        self,
        method: object,
        data: object,
        minisummary: object = None,
        pars: dict = None,
    ):
        """
        After the traces have been analyzed, we next
        filter out events at times of stimulus artifacts and
        then make an average of all of the traces, separately for spontaneous
        and evoked events. We also save the indices of those traces into the data array
        """
        # build array of artifact times first
        assert data.ndim == 2
        assert minisummary is not None
        assert pars is not None
        artifacts = self.get_artifact_times()

        CP.cprint("c", "\nanalyze_map_data: Average Trial Events")
        ev_done = []
        for itrace in range(data.shape[0]):
            # get events in the trace:
            # CP.cprint("c", f"    AMD: analyzing trace: {itrace:d}")
            evtr = list(
                [x[1] for x in minisummary.isolated_event_trace_list if x[0] == itrace]
            )

            # Define:
            # spontaneous events are those that are:
            #   a: not in evoked window, and,
            #   b: no sooner than 10 msec before a stimulus,
            # Also, events must be at least 4*tau[0] after start of trace, and 5*tau[1] before end of trace
            # Evoked events are those that occur after the stimulus with in a window (usually ~ 5 msec)
            # All data for events are aligned on the peak of the event, and go 4*tau[0] to 5*tau[1]
            #
            # process spontaneous events first
            # also, only use isolated events for this calculation

            npk_sp = method.select_events(
                pkt=[minisummary.onsets[itrace][x] for x in evtr],
                tstarts=[0.0],
                tdurs=self.Pars.stimtimes["start"][0] - (0.010),
                rate=minisummary.dt_seconds,
                mode="accept",
            )
            if len(npk_sp) > 0:
                sp_onsets = [minisummary.onsets[itrace][x] for x in npk_sp]
                minisummary.spontaneous_event_trace_list.append(
                    [sp_onsets, [minisummary.smpkindex[itrace][x] for x in npk_sp]]
                )
                avg_spont_one, avg_sponttb, allev_spont = method.average_events_subset(
                    data[itrace], eventlist=sp_onsets, minisummary=minisummary
                )
                minisummary.average_spont.avgevent.append(avg_spont_one)
            else:
                minisummary.spontaneous_event_trace_list.append([[], []])
            # print(ok_events*rate)

            # Now get the events in the evoked event window across all traces
            npk_ev = method.select_events(
                pkt=[minisummary.onsets[itrace][x] for x in evtr],
                tstarts=self.Pars.stimtimes["start"],
                tdurs=self.Pars.response_window,
                rate=minisummary.dt_seconds,
                mode="accept",
                first_only=True,
            )
            if len(npk_ev) > 0:
                ev_onsets = [minisummary.onsets[itrace][x] for x in npk_ev]
                minisummary.evoked_event_trace_list.append(
                    [ev_onsets, [minisummary.smpkindex[itrace][x] for x in npk_ev]]
                )
                (
                    avg_evoked_one,
                    avg_sponttb,
                    allev_evoked,
                ) = method.average_events_subset(
                    data[itrace], eventlist=ev_onsets, minisummary=minisummary
                )
                minisummary.average_evoked.avgevent.append(avg_evoked_one)
            else:
                minisummary.evoked_event_trace_list.append([[], []])
            ok_events = np.array(minisummary.smpkindex[itrace])[npk_ev]
            # if len(npk) > 0:  # only do this
            #     summary = method.average_events(traces=[i], eventlist = summary.onsets, data=data)
            # these are the average fitted values for the i'th trace
            # fit_tau1.append(method.fitted_tau1)
            # fit_tau2.append(method.fitted_tau2)
            # fit_amp.append(method.amplitude)
            # avg_evoked.append(summary.allevents)
            # measures.append(method.measure_events(data[i], summary.onsets[i]))
            # if len(method.Summary.average.avgeventtb) > 0:
            #     txb = method.Summary.average.avgeventtb  # only need one of these.

        sa = np.array(minisummary.average_spont.avgevent)
        ea = np.array(minisummary.average_evoked.avgevent)

        if sa.shape[0] > 0:
            minisummary.average_spont.avgevent = np.mean(sa, axis=1)
        else:
            minisummary.average_spont.avgevent == None
        if ea.shape[0] > 0:
            minisummary.average_evoked.avgevent = np.mean(ea, axis=1)
        else:
            minisummary.average_evoked.avgevent = None

        if self.verbose:
            print("      --trace analyzed")
        return minisummary

    def fix_artifacts(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Use a template to subtract the various transients in the signal...
        """
        assert self.AR is not None
        testplot = False
        CP.cprint("c", "Fixing artifacts")
        avgd = data.copy()
        while avgd.ndim > 1:
            avgd = np.mean(avgd, axis=0)
        meanpddata = self.AR.Photodiode.mean(
            axis=0
        )  # get the average PD signal that was recorded
        shutter = self.AR.getLaserBlueShutter()
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
            datar = data.copy()
        else:
            crosstalk = None
            if self.Pars.artifact_file_path is not None:
                template_file = Path(self.Pars.artifact_file_path, template_file)
                CP.cprint("w", f"   Artifact template: {str(template_file):s}")
                with open(template_file, "rb") as fh:
                    d = pickle.load(fh)
                ct_SR = np.mean(np.diff(d["t"]))
                # or if from photodiode:
                # ct_SR = 1./self.AR.Photodiode_sample_s[0]
                crosstalk = d["I"] - np.mean(
                    d["I"][0 : int(0.020 / ct_SR)]
                )  # remove baseline
                # crosstalk = self.preprocess_data(d['t'], crosstalk)
                avgdf = avgd - np.mean(avgd[0 : int(0.020 / ct_SR)])
                # meanpddata = crosstalk
                # if self.shutter is not None:
                #     crossshutter = 0* 0.365e-21*Util.SignalFilter_HPFBessel(self.shutter['data'][0], 1900., self.AR.Photodiode_sample_rate[0], NPole=2, bidir=False)
                #     crosstalk += crossshutter

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
            datar = data.copy()
            for i in range(len(art_times)):
                strt_time_indx = int(art_times[i] * self.rate)
                idur = int(art_durs[i] * self.rate)
                send_time_indx = (
                    strt_time_indx + idur + int(0.001 * self.rate)
                )  # end pulse plus 1 msec
                if crosstalk is not None:
                    fitx = crosstalk[
                        strt_time_indx:send_time_indx
                    ]  # -np.mean(crosstalk)
                    ifitx.extend(
                        [
                            f[0] + strt_time_indx
                            for f in np.argwhere((fitx > 0.5e-12) | (fitx < -0.5e-12))
                        ]
                    )
                    wmax = np.max(np.fabs(crosstalk[ifitx]))
                    weights = np.sqrt(np.fabs(crosstalk[ifitx]) / wmax)
                    scf, intcept = np.polyfit(
                        crosstalk[ifitx], avgdf[ifitx], 1, w=weights
                    )
                    avglaserd = meanpddata  # np.mean(self.AR.LaserBlue_pCell, axis=0)

                    lbr = np.zeros_like(crosstalk)
                    lbr[ifitx] = scf * crosstalk[ifitx]
                    datar[i, :] = data[i, :] - lbr

        if not self.Pars.noderivative_artifact:
            # derivative=based artifact suppression - for what might be left
            # just for fast artifacts
            CP.cprint("w", f"   Derivative-based artifact suppression is ON")
            itmax = int(self.Pars.analysis_window[1] * self.rate)
            avgdr = datar.copy()
            olddatar = datar.copy()
            while olddatar.ndim > 1:
                olddatar = np.mean(olddatar, axis=0)
            olddatar = olddatar - np.mean(olddatar[0:20])
            while avgdr.ndim > 1:
                avgdr = np.mean(avgdr, axis=0)
            diff_avgd = np.diff(avgdr) / np.diff(self.Data.timebase)
            sd_diff = np.std(diff_avgd[:itmax])  # ignore the test pulse

            tpts = np.where(np.fabs(diff_avgd) > sd_diff * self.Pars.sd_thr)[0]
            tpts = [t - 1 for t in tpts]

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

        return (datar, avgd)

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


def main():
    # these must be done here to avoid conflict when we import the class, versus
    # calling directly for testing etc.
    # matplotlib.use("Agg")
    rcParams = matplotlib.rcParams
    rcParams["svg.fonttype"] = "none"  # No text as paths. Assume font installed.
    rcParams["pdf.fonttype"] = 42
    rcParams["ps.fonttype"] = 42
    rcParams["text.usetex"] = True
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

    DP = EP.data_plan.DataPlan(
        os.path.join(datadir, args.datadict)
    )  # create a dataplan
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
        EPIV = EP.iv_analysis.IVSummary(
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
        # mpl.show()


if __name__ == "__main__":
    main()
