import sys

if sys.version_info[0] < 3:
    exit()
import argparse
import collections
import datetime
import logging
import pathlib
import pickle
import re
import textwrap
from collections import OrderedDict
from pathlib import Path
from typing import List, Union

import dateutil.parser as DUP
import matplotlib
import numpy as np
import pandas as pd
import scipy.special as scsp

rcParams = matplotlib.rcParams
rcParams["svg.fonttype"] = "none"  # No text as paths. Assume font installed.
rcParams["pdf.fonttype"] = 42
rcParams["ps.fonttype"] = 42
rcParams["text.usetex"] = False
import ephys.datareaders as DR
from ephys.tools.get_configuration import get_configuration
import ephys.ephys_analysis as EP
import ephys.tools.filename_tools as FT
import ephys.ephys_analysis.poisson_score as EPPS
import matplotlib.pyplot as mpl
import seaborn as sns

mode = 1 # set for old files
if mode == 0:
    import ephys.mini_analyses.mini_event_dataclasses as MEDC
    import ephys.mini_analyses.mini_event_dataclass_reader as MEDR
elif mode == 1:
    import ephys.mini_analyses.mini_event_dataclasses_V1 as MEDC
    import ephys.mini_analyses.mini_event_dataclass_reader_V1 as MEDR

AR = DR.acq4_reader.acq4_reader()
import pylibrary.plotting.plothelpers as PH
import pylibrary.tools.cprint as CP

import shuffler


class CustomFormatter(logging.Formatter):
    grey = "\x1b[38;21m"
    yellow = "\x1b[33;21m"
    red = "\x1b[31;21m"
    bold_red = "\x1b[31;1m"
    white = "\x1b[37m"
    reset = "\x1b[0m"
    lineformat = "%(asctime)s - %(levelname)s - (%(filename)s:%(lineno)d) %(message)s "

    FORMATS = {
        logging.DEBUG: grey + lineformat + reset,
        logging.INFO: white + lineformat + reset,
        logging.WARNING: yellow + lineformat + reset,
        logging.ERROR: red + lineformat + reset,
        logging.CRITICAL: bold_red + lineformat + reset,
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


logging.getLogger("fontTools.subset").disabled = True
Logger = logging.getLogger("MapEventScoring")
level = logging.DEBUG
Logger.setLevel(level)
# create file handler which logs even debug messages
logging_fh = logging.FileHandler(filename="map_scoring.log")
logging_fh.setLevel(level)
logging_sh = logging.StreamHandler()
logging_sh.setLevel(level)
log_formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s  (%(filename)s:%(lineno)d) - %(message)s "
)
logging_fh.setFormatter(log_formatter)
logging_sh.setFormatter(CustomFormatter())  # log_formatter)
Logger.addHandler(logging_fh)
# Logger.addHandler(logging_sh)
Logger.info("Starting map_analysis")


config_file_path = "./config/experiments.cfg"


cprint = CP.cprint


class MapEventScoring:
    def __init__(self, dataSummary, experiment):
        self.dataSummary = dataSummary  # pandas dataframe with the datasummary
        self.experiment = experiment  # experiment name (configuration file)
        df_file = Path(self.experiment['databasepath'], self.experiment['directory'], self.experiment['datasummaryFilename'])
        with open(df_file, "rb") as fh:
            self.dataSummary = pickle.load(fh)

    def check_event_data(self, protocols):
        """
        Determine if the data requested is analyzed already, and if so, compare
        the times. If the last recorded analysis run is more recent than the dbase analysis
        time, then we need to update the database.

        Parameters:

        protocols : dict:
            List of protocols to check against.

        Returns
        -------

        self.events : the event dict
                 {} if the data needs updating (empty dict means have not updated dict)
                 None if the file is not in the dict
        """

        basefile = Path(protocols[0])

        fp = basefile.parts
        eventfilename = Path("~".join(fp[0:3]) + ".pkl")
        eventfilekey = str(basefile.parent)
        if self.events is None:
            return False
        if eventfilekey not in list(self.events.keys()):
            cprint(
                "red",
                f"    Data for {eventfilekey:s}  was NOT found in Event Summary File",
            )
            return False  # not found...
        cprint("green", f"\nFile Data for {eventfilekey:s} was found in Event Summary File")
        cprint(
            "magenta",
            "    Last Analysis Time in event summary file: "
            + str(self.events[eventfilekey]["lastanalysistime"]),
        )

        if "lastanalysistime" not in list(self.events[eventfilekey].keys()):
            self.events[eventfilekey][
                "lastanalysistime"
            ] = datetime.datetime.now()  # provide a time
        lastanalysis = self.events[eventfilekey]["lastanalysistime"]
        thisevfile = Path(self.eventspath, eventfilename)
        if thisevfile.is_file():  # get latest analysis time from inside the file
            cprint("yellow", f"   Checking {str(eventfilename)} for updates")
            with open(thisevfile, "rb") as fh:
                data = pickle.load(fh)
            dps = list(data.keys())  # llist of protocols in the data
            anynew = False
            for p in dps:
                if data[p] is None:
                    continue
                try:
                    prot_analysistime = data[p]["analysisdatetime"]
                except:
                    print("p: ", p)
                    print(data[p])
                    raise ValueError
                if prot_analysistime > lastanalysis:  # more recent than in events?
                    anynew = True
                if anynew:
                    cprint(
                        "yellow",
                        f"   Data analysis needs updating Old: {eventfilekey:s}",
                    )
                    print(
                        f"         Old: {lastanalysis.strftime('%m.%d.%Y %H:%M:%S'):s},",
                        end="",
                    )
                    print(f" New: {prot_analysistime.strftime('%m.%d.%Y %H:%M:%S')}")
                    return False
        cprint("green", "   Data analysis is current")
        return self.events[eventfilekey]  # no changes, just return the data.

    def getZ_fromprotocol(
        self,
        protocol,
        filename=None,
        param="area_fraction_Z",
        area_z_threshold=2.1,
        stimno=0,
    ):
        cprint("magenta", f"Filename: {str(filename):s}")
        protocols, d = self._get_cell_protocol_data(filename)
        protocol = str(protocol)
        # print("protocols: ", protocols)
        if protocol not in protocols:
            # print("protocols: ", protocols)
            # return np.nan
            raise ValueError("Protocol not found: ", protocol)
        if param == "area_fraction_Z":
            # area_fraction = float(len([d[protocol]['ZScore'][stimno] > area_z_threshold]))/float(len(d[protocol]['positions']))
            dz = np.where(d[protocol]["ZScore"][stimno] > area_z_threshold)
            print(
                "# above thr: ",
                len(dz[0]),
                "  pts: ",
                len(d[protocol]["ZScore"][stimno]),
            )
            posxy = d[protocol]["positions"]
            nspots = len(posxy)
            # dz = np.where(d[protocol]['ZScore'][stimno] > area_z_threshold)

            ngtthr = np.array(d[protocol]["ZScore"][stimno][dz]).sum()
            print("Fraction: ", ngtthr / nspots)
            return ngtthr / nspots

        if param == "median_Z_abovethreshold":
            dz = np.where(d[protocol]["ZScore"][stimno] > area_z_threshold)
            return np.nanmedian(d[protocol]["ZScore"][stimno][dz])

        if param == "maximum_Z":
            return np.nanmax(d[protocol]["ZScore"][stimno])

        if param == "mean_Z":
            return np.nanmean(d[protocol]["ZScore"][stimno])

    def z2p(self, z):
        """From z-score return p-value."""
        return 0.5 * (1 + scsp.erf(z / np.sqrt(2)))

    #
    # repair missing stim information in "increase" files
    #
    def fix_increase_file(self, protocol_name, this_eventlist):
        fixstim = False

        if protocol_name.find("_increase_") >= 0:
            fixstim = True
            this_eventlist["stimtimes"] = {
                "starts": [0.1, 0.2, 0.3, 0.4, 0.5],
                "npulses": 5,
                "period": 0.1,
            }
        else:  # use the original data
            this_eventlist["stimtimes"]["npulses"] = len(this_eventlist["stimtimes"]["starts"])
            if this_eventlist["stimtimes"]["npulses"] >= 2:
                this_eventlist["stimtimes"]["period"] = (
                    this_eventlist["stimtimes"]["starts"][1]
                    - this_eventlist["stimtimes"]["starts"][0]
                )
            else:
                this_eventlist["period"] = 0.0
        return this_eventlist

    def get_protocol_file(self, filename):
        """get_protocol_file get the protocol file, the list of
        protocols contained in the file, the number of maps, and the basefilename

        Parameters
        ----------
        filename : str or Path
            protocol file to read

        Returns
        -------
        protocols : list,
        nmaps: int,
        basefile: Path

        """
        file_exists = Path(filename).is_file()
        print("File exists: ", file_exists)
        if not file_exists:
            cprint("red", f"    Protocol File {str(filename):s} does not exist")
            raise FileNotFoundError
        # return None, False, None
        with open(filename, "rb") as fh:  # get the data from the individual (not summary) file
            try:
                dfn = pickle.load(fh)
            except pickle.UnpicklingError:
                print(f"Problem reading on file: {str(filename):s}")
                return None, False, None

        protocols = sorted(list(dfn.keys()))
        # keys include date/slice/cell/protocol as pathlib Path
        # d[protocol] will have keys:
        # ['Qr', 'Qb', 'ZScore', 'I_max', 'positions', 'aj', 'stimtimes', 'events', 'eventtimes',
        #        'dataset', 'analysisdatetime', 'onsets']
        # These come from the "results" in analyzemap...
        if not protocols:
            cprint("red", "    No protocols found")
            return None, False, None
        nmaps = len(protocols)
        if nmaps == 0:
            cprint("red", "    No maps found")
            return None, False, None
        cprint("g", f"    {nmaps:d} maps found")
        basefile = Path(protocols[0])
        return protocols, nmaps, basefile

    def flatten(self, l):
        for el in l:
            if isinstance(el, collections.abc.Iterable) and not isinstance(el, (str, bytes)):
                yield from self.flatten(el)
            else:
                yield el


    def score_events(
        self,
        filename,
        eventwindow=[0.000, 0.005],
        area_z_threshold=2.1,
        plotflag=False,
        force=False,
        sel_celltype=None,
        mode:int=1,
    ):
        """
        Calculate various scores and information about events and maps

        Parameters
        ----------
        filename: str or Path
            filename of the cell event file to analyze (e.g. '2019-12-03_slice_1_cell_1.pkl')

        eventwindow : list or tuple (2 values; default [0.002, 0.005])
            starting time and duration for detecting events after a stimulus

        plotflag : bool (default: False)
            enable plotting for each call to score a dataset

        force : boolean : (default: False)
            force computation of events even if previous seems up to date.

        celltype : str (default: None)
            if set, only analyze for a particular cell type

        Returns
        -------
        dict of analysis with the following keys and information:
            {'scores': scores, 'SR': sr, 'celltype': celltype,
                    'shufflescore': shufflescore, 'eventp': eventp, "event_amps": event_amp,
                    'spont_amp': spontevent_amp,
                    'protocols': allprotos, 'validdata': validdata,
                    'lastanalysistime': datetime.datetime.now()}
        """

        cprint("g",    f"    Starting ScoreEvents on Cell {str(filename):s}")
        Logger.info(f"Starting ScoreEvents on Cell {str(filename):s}")
        assert (
            len(eventwindow) == 2
        )  # need to be sure eventwindow is properly formatted on the call
        SH = shuffler.Shuffler()  # get am instance of the shuffling code.

        if str(filename).find("_alt") > 0 or str(filename).find("_signflip") > 0:
            Logger.warning(f"Protocol {fn!s} is an alternate or signflip protocol, skipping")
            return None, False
        ds_sel = self.dataSummary[self.dataSummary["cell_id"] == filename]
        print("ds_sel: ", ds_sel)
        fn = FT.make_event_filename_from_cellid(filename)

        eventfilename = Path(self.experiment["databasepath"], self.experiment["directory"], "events", fn)
        cprint("g", f"    Reading event file: {eventfilename!s}")
        with open(eventfilename, "rb") as fh:
            eventdata = pd.read_pickle(fh)
       
        protocols = list(eventdata.keys())
        nmaps = len(protocols)

        if protocols is None:
            return None, False


        # cprint("yellow", f"forceflag: {force:b}")
        # if not force:  # check for, and just return existing data
        #     evndata = self.check_event_data(protocols)
        #     if bool(evndata):
        #         cprint("magenta", f"File: {str(eventfilekey):s} is up to date")
        #         return evndata, False  # returns filled data in dict
        # otherwise we need to do the analysis on the .pkl file

        # analyze
        # if force is set, then we recalculate from the cell's own .pkl file
        # analyze
        cprint("g", "    Starting analysis")
        evndata = {
            "scores": None,
            "SR": None,
            "celltype": None,
            "shufflescore": None,
            "minprobability": None,
            "meanprobability": None,
            "eventp": None,
            "event_amps": None,
            "protocols": None,
            "validdata": None,
            "spont_amps": None,
            "event_qcontent": None,
            "largest_event_qcontent": None,
            "depression_ratio": None,
            "paired_pulse_ratio": None,
            "firstevent_latency": None,
            "allevent_latency": None,
            "lastanalysistime": None,
            "area_fraction_Z": None,
            "positions": None,
        }

        scores = np.zeros(len(protocols))
        # every protocol will have the same top name, so just get the first one
        allprotos = [None] * len(protocols)
        shufflescore = [[0.0]] * len(protocols)
        minprobability = [[1.0]] * len(protocols)
        meanprobability = [[]] * len(protocols)
        eventp = [[0.0]] * len(protocols)
        event_amp = [[0.0]] * len(protocols)
        firstevent_latency = [None] * len(protocols)
        allevent_latency = [None] * len(protocols)
        spont_rate = [[0.0]] * len(protocols)
        spontevent_amp = [[0.0]] * len(protocols)
        event_Q_Content = [[0.0]] * len(protocols)
        event_Largest_Q_Content = [[0.0]] * len(protocols)
        validdata = [[False]] * len(protocols)
        datamode = [None] * len(protocols)
        area_fraction_Z = [None] * len(protocols)
        positions = [None] * len(protocols)
        depression_ratio = [None] * len(protocols)
        paired_pulse_ratio = [None] * len(protocols)
        cprint("g", "    initialized result arrays")

        celltype = ds_sel["cell_type"].values[0]
        if plotflag:
            self.plot_setup(nmaps)

        for i, protocol in enumerate(protocols):
            if sel_celltype is not None and sel_celltype != celltype:
                cprint(
                    "r",
                    f"celltype: {sel_celltype:s} does not match input argument celltype: {celltype:s}",
                )
                Logger.error(
                    f"Protocol {dxf:s}  celltype: {sel_celltype:s} does not match input argument celltype: {celltype:s}"
                )
                # raise()
                continue
            cprint("y", f"    Celltype: {celltype:s}")
            # skip some cell types that are not well represented in the database
            if (
                celltype
                in [
                    "None",
                    "glial",
                    "unknown",
                    " ",
                    "horizontal bipolar",
                    "chestnut",
                    "ml-stellate",
                    "0",
                ]
                or len(celltype) == 0
            ):
                continue
            cprint("g", f"    Proceeding with celltype:  <{celltype:s}>")
            temperature = ds_sel['temperature']
            age = ds_sel['age']
            this_eventlist = eventdata[protocol]
            if this_eventlist is None:
                cprint("red", f"    No data in protocol {str(protocol):s}")
                raise ValueError()
                continue

            # this_eventlist = evl[str(dxp)]

            protocol_name = str(Path(protocol).parts[-1])
            sign = -1
            pmode = "0"
            if protocol_name.find("_IC_") >= 0:
                sign = 1
                scale = 1e3
                pmode = "I"
            elif protocol_name.find("_VC_") >= 0:  # includes "increase" protocol
                scale = 1e12
                pmode = "V"
            elif protocol_name.find("VGAT_5mspulses") >= 0:
                sign = 1
                pmode = "V"
                scale = 1e12
            elif protocol_name.find("CC_VGAT_5mspulses") >= 0:
                sign = -1
                scale = 1e3
                pmode = "I"
            else:
                scale = 1.0

            if "stimtimes" not in list(this_eventlist.keys()):
                cprint(
                    "red",
                    f"    Missing 'stimtimes' in event list: " + str(list(this_eventlist.keys())),
                )
                continue
            # area_fraction is the fractional area of the map where the scores exceed 1SD above
            # the baseline (Z Scored; charge based)
            ngtthr = (np.array(this_eventlist["ZScore"][0]) > area_z_threshold).sum()

            # get the positions
            posxy = this_eventlist["positions"] # dfn[dxf]["positions"]
            print("stimtimes 0:", this_eventlist["stimtimes"])
            print("posxy: ", posxy)
            this_eventlist = self.fix_increase_file(protocol_name, this_eventlist)

            te = 0
            ts = 0.0  # evl['stimtimes']['start'][0]
            stimtimes = []
            for n in range(this_eventlist["stimtimes"]["npulses"]):
                st0 = this_eventlist["stimtimes"]["starts"][n]
                stimtimes.append((st0, eventwindow[0], eventwindow[1]))
                te = st0 + np.sum(eventwindow)

            # accumulate the event amplitudes and matching times for this protocol, across all trials
            reader = MEDR.Reader(eventdata[protocol])
            events = this_eventlist["events"]
            evamps = []
            evtimes = []
            ntrials = reader.get_ntrials()
            nspots = len(posxy) / ntrials
            # area_fraction = float(len([d[dx]['ZScore'][-1] > area_z_threshold]))/float(len(d[dx]['positions']))
            area_fraction = (
                ngtthr / nspots
            )  # now compute the "area fraction" of the map that has a zscore above threshold
            cprint("g", f"    Area fraction: {area_fraction:0.3f}")
            cprint("c", f"    Ntrials: {reader.get_ntrials():d}")
            for trial in range(reader.get_ntrials()):  # trials
                trial_events = reader.get_events()[
                    trial
                ]  # trial events is a Mini_Event_Summary dataclass
                if trial_events is None:
                    cprint("r", "No trial events found")
                    continue
                dt = reader.get_sample_rate(trial)
                evt = reader.get_trial_event_onset_times(trial, trial_events.onsets)
                eva = reader.get_trial_event_amplitudes(trial, trial_events.smpkindex)
                evtimes.append(evt)
                evamps.append(eva)
            if len(evtimes) == 0:
                cprint("r", "Event times list is empty")
                return None, False
            evtimes = evtimes[0]
            evamps = evamps[0]
            evtimes_flat = [t for j in range(len(evtimes)) for t in evtimes[j]]
            evamps_flat = [a for j in range(len(evamps)) for a in evamps[j]]

            cprint("c", f"    # of traces in all trials:            {len(evtimes):>6d}")
            cprint("c", f"    Length of all event times all trials: {len(evtimes_flat):>6d}")

            #   debugging: plot event times and amplitudes to confirm we have the data
            # f, ax = mpl.subplots(1,2)
            # for tr in range(len(evtimes)):
            #     if len(evtimes[tr]) == 0:
            #         continue
            #     # exit()
            #     ax[0].plot(evtimes[tr], evamps[tr], 'o', markersize=2)
            # ax[0].set_xlabel('Time (s)')
            # mpl.show()
            # exit()

            # do poisson or shuffle scoring on evtimes
            # prepare the event array for PoissonScore
            # PoissonScore.score expects the events to be a list of data in a record array format
            evp = []
            for trial in range(len(evtimes)):  # across all *trials* in the map
                evtx = []
                evax = []
                ev_tr = evtimes[trial]
                if len(ev_tr) == 0:
                    continue
                for j, ev_lat in enumerate(ev_tr):  # handle individual events
                    if ev_lat >= ts and ev_lat <= te:
                        evtx.extend([ev_lat])
                        evax.extend([evamps[trial][j]])
                ev = np.empty(
                    len(evtimes[trial]), dtype=[("time", float), ("amp", float)]
                )  # rec array
                ev["time"] = evtimes[trial]
                ev["amp"] = np.array(evamps[trial])
                evp.append(ev)
                # capture latencies here
                if (  # limit the protocols that we will use for
                    # latency measurements
                    str(protocol).find("_VC_10Hz") > 0
                    or str(protocol).find("Single") > 0
                    or str(protocol).find("single") > 0
                    or str(protocol).find("_VC_weird") > 0
                    or str(protocol).find("_VC_2mW") > 0
                    or str(protocol).find("_VC_1mW") > 0
                    or str(protocol).find("_VC_00") > 0
                    or str(protocol).find("_range test")
                    or str(protocol).find("_VC_increase") > 0
                ):
                    for ifsl, st in enumerate(stimtimes):
                        t0 = st[0] + st[1]
                        t1 = st[0] + st[2]
                        # print('t0, t1: ', t0, t1)
                        evi = np.where((evtimes[trial][j] > t0) & (evtimes[trial][j] <= t1))
                        evw = evtimes[trial][evi[0]]
                        # print('evi, evw: ', evi, evw)
                        if len(evw) == 0:  # no events in the window
                            # print('no data in window')
                            continue
                        if ifsl == 0 and len(evw) > 0:
                            if firstevent_latency[i] == None:
                                firstevent_latency[i] = [evw[0] - st[0]]
                            else:
                                firstevent_latency[i].extend([evw[0] - st[0]])
                        if allevent_latency[i] == None:
                            allevent_latency[i] = [t - st[0] for t in evw]
                        else:
                            allevent_latency[i].extend([t - st[0] for t in evw if not pd.isnull(t)])
                else:
                    cprint("red", f"    Protocol Excluded on type: {str(protocol):s}")

            ev = {}
            ev["time"] = np.array(evtimes_flat)
            ev["amp"] = np.array(evamps_flat)
            if len(evtimes) == 0:  # evp:
                cprint("red", "    No data in protocol?")
                sr = 0.0
                continue  # no data in this protocol?
            # print('evp: ', evp)
            # compute spont detected event rate, and get average event amplitude for spont events
            spont_evt_index = [i for i, t in enumerate(ev["time"]) if t < stimtimes[0][0]]
            spont_evt_amps = ev["amp"][spont_evt_index]
            if len(spont_evt_amps) > 0:
                cprint(
                    "g",
                    f"    mean spont amp: {np.mean(spont_evt_amps)*scale:.2f} (SD: {np.std(spont_evt_amps)*scale:.2f}, N={len(spont_evt_amps):d}",
                )
            else:
                cprint("y", f"    No spont events")
            ev_evt_index = [
                i
                for i, t in enumerate(ev["time"])
                if t > stimtimes[0][0] and t < stimtimes[0][0] + 0.015
            ]
            ev_evt_amps = ev["amp"][ev_evt_index]
            nspont = len(spont_evt_index)
            if nspont > 0:
                sr = len(spont_evt_index) / (
                    stimtimes[0][0] * nspots
                )  # count up events and divide by total time examined
            else:
                sr = 0.0
            spontaneous_amplitudes = ev["amp"][spont_evt_index]
            print("            SpontRate: {0:.3f} [N={1:d}]".format(sr, nspont))

            print(
                "            Mean spont amp (all trials): {0:.2f} pA  SD {1:.2f} N={2:4d}".format(
                    np.nanmean(spont_evt_amps) * scale,
                    np.nanstd(spont_evt_amps) * scale,
                    np.shape(spont_evt_amps)[0],
                )
            )

            firststimt = this_eventlist["stimtimes"]["starts"][0]
            # finally, calculate the Poisson Score
            if evp[0].shape[0] > 0:
                # mscore_n, mscore, mean_ev_amp = SH.shuffle_score(evp, stimtimes, nshuffle=5000, maxt=0.6)  # spontsonly...
                mscore_n, prob = EPPS.PoissonScore.score(evp, rate=sr, tMax=0.6, normalize=True)
            else:
                mscore_n = np.ones(len(stimtimes))
                prob = 1.0
            cprint("g", f"    Poisson score: {mscore_n:6.4f}")
            mscore = mscore_n
            mean_ev_amp = 1.0
            if not isinstance(mscore, list):
                mscore = [mscore]
            m_shc = np.mean(mscore)
            s_shc = np.std(mscore)
            # print('mscore, : ', mscore, mean_ev_amp)

            # find the probability in the response windows
            probs = np.ones(len(mscore))
            winlen = 0.010
            ev_prob_response_events = dict(
                zip(range(len(stimtimes)), [] * len(stimtimes))
            )  # indices into "response" window events, by stim window
            ev_prob = []  # actual probabilities
            for ist, st in enumerate(stimtimes):
                t0 = st[0] + st[1]
                t1 = t0 + winlen
                # print('t0, t1: ', t0, t1)
                response_events = list(
                    np.where((np.array(evtimes_flat) > t0) & (np.array(evtimes_flat) <= t1))[0]
                )
                # print("response events: ", response_events)
                if len(response_events) > 0:
                    ev_prob_response_events[ist] = response_events
                    ev_prob.extend(prob[response_events])
                else:
                    ev_prob.extend([1.0])
                    ev_prob_response_events[ist] = []

            for ist in range(len(ev_prob_response_events)):
                if len(ev_prob_response_events[ist]) > 0:
                    meanp = np.mean(prob[ev_prob_response_events[ist]])
                else:
                    meanp = 1.0

            ev_prob_response_events_flat = [x for y in ev_prob_response_events.values() for x in y]

            if nspont > 10:
                mean_spont_amp = np.nanmean(spontaneous_amplitudes)
            else:
                cprint(
                    "red",
                    f"    Using Mean spont from table for : {str(sel_celltype):s}, temp= {str(temperature):s}C",
                )
                if (sel_celltype == " ") or (
                    (sel_celltype, temperature)
                    not in list(set_expt_paths.mean_spont_by_cell.keys())
                ):
                    mean_spont_amp = 20e-12  # pA

                elif (sel_celltype, temperature) in list(set_expt_paths.mean_spont_by_cell.keys()):
                    mean_spont_amp = (
                        set_expt_paths.mean_spont_by_cell[(sel_celltype, temperature)] * 1e-12
                    )  # uset the mean value for the cell type

            detevt_n, detevt, detamp = SH.detect_events(
                event_times=evtimes,
                event_amplitudes=evamps,
                stim_times=stimtimes,
                mean_spont_amp=mean_spont_amp,
            )
            if detevt_n == 0:
                CP.cprint("y", f"            No events detected in protocol?")
                Z = 0.0
                continue
            det_amp = [a[0] for v in detamp for a in v if len(v) > 0]
            # print("detamp: ", det_amp)
            cprint("g", f"    Found {len(det_amp):d} events")
            if s_shc != 0.0:
                Z = (detevt_n[0] - m_shc) / s_shc  # mscore[0])
            else:
                Z = 100.0  # arbitrary high value

            detamp = np.array(det_amp)

            # exit()
            if not any(detamp):
                detamp = [np.array([0])]
            print(
                "            Mean evoked amp (trial 0): {0:.2f} pA  SD {1:.2f} N={2:3d}".format(
                    np.nanmean(detamp) * scale,
                    np.std(detamp) * scale,
                    np.shape(detamp)[0],
                )
            )
            # print('mscore: ', mscore)
            event_qcontent = np.zeros(len(mscore))
            largest_event_qcontent = np.zeros(len(mscore))

            for j, m in enumerate(mscore):
                if mscore[j] > 100:
                    mscore[j] = 100.0  # clip mscore
                # try:
                if mean_ev_amp == 0.0:
                    continue
                sa = np.nanmean(spontaneous_amplitudes)
                if j >= len(detamp):
                    continue

                if sign == -1:
                    det = detamp[j] < sa
                    if det.any():
                        event_qcontent[j] = np.nanmean(detamp[j][detamp[j] < sa]) / mean_ev_amp
                        largest_event_qcontent[j] = np.min(detamp[j]) / mean_ev_amp
                else:
                    det = detamp[j] > sa
                    if det.any():
                        event_qcontent[j] = np.nanmean(detamp[j][detamp[j] > sa]) / mean_ev_amp
                        largest_event_qcontent[j] = np.max(detamp[j]) / mean_ev_amp

                if mscore[j] > 0:
                    print("prob: ", prob)
                    print("probs: ", probs)
                    print(
                        f"        Stim/trial {j:2d} ShuffleScore= {mscore[j]:6.4f}", end="")
                    print(f"Lowest Shuffle Prob = {np.min(prob[ev_prob_response_events_flat]):6.3g}", end="")
                    print(f"EventP: {detevt[j]:6.3e} Z: {Z:7.4f}, P(Z): {(1.0-self.z2p(Z)):.3e}",
                        end=""
                    )
                    print(
                        f" Event Amp re Spont: {event_qcontent[j]:g}", end="")
                    print(f"Largest re spont: {largest_event_qcontent[j]:7.4f}"
                    )

                else:
                    print(
                        f"        Stim/Trial {j:2d} ShuffleScore= {mscore[j]:6.4f}", end="")
                    print(f"Lowest Shuffle Prob = {np.min(prob[ev_prob_response_events_flat]):6.3g}", end="")
                    print(f"EventP: {detevt[j]:6.3e} P(Z): {(1.0-self.z2p(Z)):.3e}")

            mscore[mscore == 0.0] = 100
            # build result arrays
            allprotos[i] = protocol_name
            shufflescore[i] = Z
            minprobability[i] = np.min(prob[ev_prob_response_events_flat])
            for ist in range(len(stimtimes)):
                if len(prob[ev_prob_response_events[ist]]) > 0:
                    meanprobability[i] = np.mean(prob[ev_prob_response_events[ist]])
                else:
                    meanprobability[i] = 1.0
            eventp[i] = detevt
            event_amp[i] = detamp

            scores[i] = 1.0 - self.z2p(Z)  # take max score for the stimuli here
            spont_rate[i] = sr
            spontevent_amp[i] = spontaneous_amplitudes

            validdata[i] = True
            event_Q_Content[i] = event_qcontent
            event_Largest_Q_Content[i] = largest_event_qcontent
            datamode[i] = pmode
            area_fraction_Z[i] = area_fraction

            # drdata = self.depression_ratio(
            #     evfile=dxp,
            #     stim_N=4,
            # )  # compute depression ratio
            # depression_ratio[i] = drdata["ratio"]

            positions[i] = posxy
            if plotflag:
                self.plot_one_map(evtimes, this_eventlist, i, protocol=protocol)

        ########
        # END OF i loop over protocols
        ########

        if plotflag:
            fxl = []
            firstevent_latency = self.flatten(firstevent_latency)
            for ifl in firstevent_latency:
                if ifl is None:
                    continue
                fxl.extend([ifl])
            afl = []
            for ifl in allevent_latency:
                if ifl is None:
                    continue
                afl.extend(ifl)
            f, ax = mpl.subplots(2, 1)
            fbins = np.linspace(0, eventwindow[1], int(eventwindow[1] / 0.00010))
            abins = np.linspace(0, eventwindow[1], int(eventwindow[1] / 0.00010))
            if len(fxl) < 20:
                fbins = "auto"
            ax[0].hist(np.array(fxl), fbins, histtype="bar", density=False)
            if len(afl) < 20:
                abins = "auto"
            ax[1].hist(afl, abins, histtype="bar", density=False)
            mpl.show()

        if plotflag:
            self.P.figure_handle.suptitle(str(filename.parent))  # .replace(r"_", r"\_"))
            mpl.show()

        for i, d in enumerate(depression_ratio):
            if depression_ratio[i] is None:
                depression_ratio[i] = np.nan
        for i, d in enumerate(paired_pulse_ratio):
            if paired_pulse_ratio[i] is None:
                paired_pulse_ratio[i] = np.nan
        print("depression ratio: ", depression_ratio)
        depression_ratio = np.array(depression_ratio)
        paired_pulse_ratio = np.array(paired_pulse_ratio)
        print(paired_pulse_ratio)
        result = {
            "scores": scores,
            "SR": spont_rate,
            "celltype": sel_celltype,
            "shufflescore": shufflescore,
            "minprobability": minprobability,
            "meanprobability": meanprobability,
            "eventp": eventp,
            "event_amps": event_amp,
            "protocols": allprotos,
            "validdata": validdata,
            "spont_amps": spontevent_amp,
            "event_qcontent": event_Q_Content,
            "largest_event_qcontent": event_Largest_Q_Content[i],
            "depression_ratio": np.nanmean(depression_ratio),
            "paired_pulse_ratio": np.nanmean(paired_pulse_ratio),
            "firstevent_latency": firstevent_latency,
            "allevent_latency": allevent_latency,
            "lastanalysistime": datetime.datetime.now(),
            "mode": datamode,
            "area_fraction_Z": area_fraction_Z,
            "positions": positions,
        }  # highest score for this cell's maps
        # except:
        #     print('i: ', i)
        #     print(event_Largest_Q_Content)
        #     print(depression_ratio)
        #     exit()
        # print('RES: ', result)
        return result, True

    def plot_setup(self, nmaps):
        rc = PH.getLayoutDimensions(nmaps, pref="width")
        self.P = PH.regular_grid(
            rc[0],
            rc[1],
            figsize=(10.0, 8),
            position=-10,
            margins={
                "leftmargin": 0.07,
                "rightmargin": 0.05,
                "topmargin": 0.1,
                "bottommargin": 0.1,
            },
        )
        self.binstuff = list(np.arange(0, 0.6, 0.005))
        self.axl = self.P.axarr.ravel()
    
    def plot_one_map(self, evtimes, this_eventlist, i:int=0, protocol:Union[str, Path]=""):
        evtimes = list(self.flatten(evtimes))
        evt = np.hstack(np.array(evtimes).ravel())
        print(evt)
        n, bins, patches = self.axl[i].hist(
            evt, bins=self.binstuff, histtype="stepfilled", color="k", align="right"
        )
        # print(this_eventlist["stimtimes"].keys())
        for s in this_eventlist["stimtimes"]["starts"]:
            self.axl[i].plot([s, s], [0, 40], "r-", linewidth=0.5)
        self.axl[i].set_ylim(0, 40)
        self.axl[i].set_title(
            str(Path(protocol).parts[-1]), fontsize=10
        )  # .replace(r"_", r"\_"), fontsize=10)

if __name__ == "__main__":

    experiment = "NF107Ai32_Het"
    config = "/Users/pbmanis/Desktop/Python/mrk-nf107/config/experiments.cfg"
    expt = get_configuration(config)
    expt = expt[1][experiment]
    fn = "2017.02.14_000/slice_000/cell_001"

    MES = MapEventScoring(experiment, expt)
    MES.score_events(fn, plotflag=True)
