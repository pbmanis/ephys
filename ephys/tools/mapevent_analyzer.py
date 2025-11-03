"""
mapevent_analyzer.py
Python 3 only
This class reads event data from the map analyzer output (in datatables),
which is stored in a pickled format (pickled files, one for each cell), usually
in the "events" directory.
The output is stored in a summary file (also pickled) which has one entry.

The directories and output files are specified in the configuration file for
the project being analyzed. This means that this package should be called
from the command line when in the specific project directory (not ephys directory).

It does some statistical evaluation of responses based on time of occurrence and probability.

To update from the events database:

mapevent_analyzer -E nf107 --eventsummary
If this indicates that some files need updating:
Do (for example):
mapevent_analyzer -E nf107 --eventsummary -d 2018.02.09 -s S3C0 --force

To update the entire events database from the files in the events/ directory:
mapevent_analyzer -E nf107 --eventsummary --force

"""

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
import prettyprinter as pprint
from typing import List, Union

import awkward as AK
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
import ephys.ephys_analysis as EP
import ephys.ephys_analysis.poisson_score as EPPS
import ephys.mapanalysistools.shuffler as shuffler
import ephys.mini_analyses.mini_event_dataclass_reader as MEDR

import ephys.mini_analyses.mini_event_dataclasses as MEDC
import ephys.tools
import matplotlib.collections as collections
import matplotlib.patches as patches
import matplotlib.pyplot as mpl
import seaborn as sns
from ephys.tools.get_configuration import get_configuration

AR = DR.acq4_reader.acq4_reader()
import pylibrary.plotting.plothelpers as PH
import pylibrary.tools.cprint as CP

cprint = CP.cprint


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
Logger = logging.getLogger("MapEvent_Analyzer")
level = logging.DEBUG
Logger.setLevel(level)
# create file handler which logs even debug messages
logging_fh = logging.FileHandler(filename="map_analysis.log")
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


np.seterr(divide="raise", invalid="raise")


# Cell names vary, so group by base name
# all_neurons = {
#     "t-stellate": [
#         "t-stellate",
#         "Tstellate",
#         "T-stellate",
#         "tstellate",
#     ],
#     "d-stellate": ["d-stellate", "Dstellate", "D-stellate", "dstellate"],
#     "bushy": ["bushy", "Bushy"],
#     "octopus": ["octopus", "squid"],
#     "glial": ["glial", "glial cell", "glia", "GLIAL"],
#     "giant": ["giant", "Giant", "Giant cell", "GIANT"],
#     "cartwheel": ["cartwheel", "cw"],
#     "pyramidal": ["pyramidal", "fusiform", "stellate", "pyr"],
#     "tuberculoventral": ["tuberculoventral", "tv"],
#     "unknown": ["unknown", "no morphology", " ", None, "None", ""],
#     "ubc": ["unipolar brush cell", "UBC"],
#     "mlstellate": ["ml-stellate"],
#     "chestnut": ["chestnut"],
#     "horizbipolar": ["horizontal bipolar"],
#     "granule": ["granule"],
#     "typeB": ["Type-B"],
# }

DCN_celltypes = ["cartwheel", "tuberculoventral", "pyramidal", "giant"]
VCN_celltypes = ["bushy", "t-stellate", "d-stellate", "octopus"]
all_celltypes = (
    VCN_celltypes
    + DCN_celltypes
    + ["unknown", "glial", "ubc", "mlstellate", "chestnut", "horizbipolar", "granule", "typeB"]
)
valid_celltypes = DCN_celltypes + VCN_celltypes


# def class_cell(cellname):
#     """
#     For a cell with a particular name, get the cannonical name from the list of neuron names
#     """
#     for c in list(all_neurons.keys()):
#         if cellname.lower() in all_neurons[c]:
#             cellname = c
#             if cellname not in all_celltypes:
#                 return None
#             return cellname
#     cprint(
#         "r", f"Cell name <{cellname:s}> (type: {str(type(cellname)):s}) not found in known neurons"
#     )
#     return None


def eng_string(x, format="%s", si=False, suffix="", default_suffix="p"):
    """
    Returns float/int value <x> formatted in a simplified engineering format -
    using an exponent that is a multiple of 3.

    format: printf-style string used to format the value before the exponent.

    si: if true, use SI suffix for exponent, e.g. k instead of e3, n instead of
    e-9 etc.

    E.g. with format='%.2f':
        1.23e-08 => 12.30e-9
             123 => 123.00
          1230.0 => 1.23e3
      -1230000.0 => -1.23e6

    and with si=True:
          1230.0 => 1.23kd
      -1230000.0 => -1.23M
    """
    import math

    sign = ""
    if np.isnan(x):
        return "np.nan" + format + "0" + suffix
    if x < 0:
        x = -x
        sign = "-"
    if x == 0.0:
        exp = 3
    else:
        exp = int(math.floor(math.log10(x)))
    exp3 = exp - (exp % 3)
    x3 = x / (10**exp3)

    # print('exp3: ', exp, exp3)
    if x == 0.0:
        exp3_text = default_suffix
    elif si and exp3 >= -24 and exp3 <= 24 and exp3 != 0:
        exp3_text = "yzafpnum kMGTPEZY"[int((exp3 - (-24)) / 3)]
    elif exp3 == 0:
        exp3_text = ""
    else:
        exp3_text = "e%s" % exp3

    return ("%s" + format + " %s") % (sign, x3, exp3_text + suffix)


#########################################################################################
# The next 2 functions get events times within a specified time window.
# because the evnt times for each trace are stored as a list of numpy arrays, they
# don't all have the smae length.
# So we need to process each trace individually.
##########################################################################################
def get_trace_events_in_window(
    event_times: np.array, window_start: float, window_end: float
) -> np.array:
    """
    Given an array of event times for a trace, return only those events
    that occur within the specified time window.

    Parameters
    ----------
    event_times : np.array
        Array of event times (in seconds).
    window_start : float
        Start time of the window (in seconds).
    window_end : float
        End time of the window (in seconds).

    Returns
    -------
    np.array
        Array of event times within the specified window.
    """
    event_times = np.array(event_times)
    in_window_mask = (event_times >= window_start) & (event_times <= window_end)
    return event_times[in_window_mask]


def get_map_event_times_in_window(map_data: list, window_start: float, window_end: float):
    """
    Given map data containing event times for multiple traces,
    return a list of arrays with event times within the specified window for each trace.

    Parameters
    ----------
    map_data : list
        list of numpy arrays of map data traces, each containing event times.
    window_start : float
        Start time of the window (in seconds).
    window_end : float
        End time of the window (in seconds).

    Returns
    -------
    List[np.array]
        List of arrays, each containing event times within the specified window for each trace.
        Traces withn o events in a window are empty.

    """
    all_event_times = map_data.get("event_times", [])
    filtered_event_times = []
    for trace_events in all_event_times:
        filtered_events = get_trace_events_in_window(trace_events, window_start, window_end)
        filtered_event_times.append(filtered_events)
    return filtered_event_times


#########################################################################################


class EventAnalyzer(object):
    def __init__(self, experiment_dict):
        self.db = None  # the main database, shared amongst all routines
        self.verbose = False
        self.taudata = None  # from the tau fit data base
        self.expt = experiment_dict
        self.coding = None
        self.coding_file = None
        self.events = None  # event dict, shared
        self.eventsummary_file = None

        # db and events are set by getDatabase

    def set_events_Path(self, eventspath):
        if Path(eventspath).is_dir():
            self.eventspath = eventspath
        else:
            raise ValueError(f"The events path: {str(eventspath):s} was not found")

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
        # for c in self.events.keys():
        #     print("Event key: ", c, eventfilekey)
        # exit()
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
            cprint("cyan", f"   Checking {str(eventfilename)} for updates")
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
                    print("data key has not field 'analysisdatetime', p: ", p)
                    print("Here is the data dict: ", data[p])
                    raise ValueError
                if prot_analysistime > lastanalysis:  # more recent than in events?
                    anynew = True
                if anynew:
                    cprint(
                        "cyan",
                        f"   Data analysis needs updating Old: {eventfilekey:s}",
                    )
                    print(
                        f"         Old: {lastanalysis.strftime('%m.%d.%Y %H:%M:%S'):s},",
                        end="",
                    )
                    print(f" New: {prot_analysistime.strftime('%m.%d.%Y %H:%M:%S')}")
                    return False
        cprint("green", "   Data analysis is up-to-date")
        return self.events[eventfilekey]  # no changes, just return the data.

    def _get_cell_protocol_data(self, fn):
        """_get_cell_protocol_data
        get the protocol for the specified cell (by filename)

        Parameters
        ----------
        fn : function
            _description_

        Returns
        -------
        tuple
            List of protocols, and the data in the .pkl file.
        """
        # the pickled files have no subdirectory information in the filename, so strip that out
        fnx = Path(fn).parts
        fnx = fnx[-3:]
        fn = "~".join(fnx)  # assemble the filename with tilde separator.
        fn = Path(self.eventspath, fn + ".pkl")
        with open(fn, "rb") as fh:
            d = pickle.load(fh)
        protocols = sorted(list(d.keys()))  # keys include date/slice/cell/protocol as pathlib Path
        return (protocols, d)

    def _get_cell_information(self, cell_ID: str, parameter: str):
        """
        Get one parameter measure from this cell using the cell_ID
        For many parameters, this reads from the main database (datasummary).
        The exceptions are if the parmeter are the "Group", "ID", or "SPL", in which case
        we read from the coding file using just the date information
        """
        cell_ID = str(cell_ID).lstrip()
        if len(cell_ID) == 0:
            return None
        # do match of cell_ID against the main database cell_id (day/slice/cell), ignoring the leading path information
        # first strip the protocol information off the cell_ID
        cell_ID_parts = Path(cell_ID).parts
        if not cell_ID_parts[-1].startswith("cell_"):
            cell_ID = str(Path(*cell_ID_parts[:-1]))
        day_x = self.db.loc[self.db["cell_id"].str.endswith(cell_ID)]
        if day_x.empty:
            return None
        # print("GCI, day_x: ", cell_ID, "\n", day_x)
        if parameter in ["Group", "ID", "SPL"]:
            cid = day_x["cell_id"].values[0]
            cid = str(Path(*Path(cid).parts[-3:]))

            cell_date = cid.split("_")[0]
            day_coding = self.coding.loc[self.coding["date"].str.endswith(cell_date)]
            # print("coding dates:]n", self.coding['date'])
            if day_coding is None:
                msg = f"Bad coding database date query: Day '{cell_date:s}' is not in the coding database"
                cprint("r", msg)
                Logger.error(msg)
                return None
            if parameter not in day_coding.columns:
                msg = f"Bad coding database inforrmation query: Parameter '{parameter:s}' is not in coding database columns"
                cprint("r", msg)
                Logger.error(msg)
                return None
            value = day_coding[parameter].values[0]
            if parameter == "SPL" and pd.isnull(value):
                value = "ND"
            if parameter == "SPL" and isinstance(value, (int, float)):
                value = str(value)
            return value
        elif parameter not in day_x.columns:
            msg = f"Bad database inforrmation query: Parameter '{parameter:s}' is not in database columns"
            cprint("r", msg)
            Logger.error(msg)
            return None

        value = day_x[parameter].values
        if isinstance(value, (list, pd.Series, np.ndarray)):
            if len(value) > 0:
                value = value[0]

        if parameter == "celltype":
            return value

        elif parameter in ["temperature", "internal", "weight", "age"]:
            if parameter == "temperature":
                value = value.replace("C", "")
                value = value.replace("room temp", "25")
                if len(value) == 0 or value == " ":
                    ret_value = 25
                ret_value = int(value)
            elif parameter == "internal":
                if value in ["Cesium", "cesium"]:
                    ret_value = "Cs"
                elif value == "Standard K-Gluc":
                    ret_value = "K-Gluc"
                else:
                    ret_value = value
                    cprint("y", f"returning Unknown internal solution: {value:s}")
            elif parameter == "weight":
                for c in ["g", "G"]:
                    value = value.replace(c, "")
                    ret_value = int(value)
                else:
                    ret_value = int(value)
                    cprint("y", f"returning weight without unit type: {value:s}")
            elif parameter == "age":
                agestr = ephys.tools.parse_ages.ISO8601_age(value)
                ret_value = ephys.tools.parse_ages.age_as_int(agestr)
            return ret_value
        else:
            return value

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
        if protocol not in protocols:
            raise ValueError("Protocol not found: ", protocol)
        if param == "area_fraction_Z":
            # area_fraction = float(len([d[protocol]['ZScore'][stimno] > area_z_threshold]))/float(len(d[protocol]['positions']))
            dz = np.where(d[protocol]["ZScore"][stimno] > area_z_threshold)
            # print(
            #     "# above thr: ",
            #     len(dz[0]),
            #     "  pts: ",
            #     len(d[protocol]["ZScore"][stimno]),
            # )
            posxy = d[protocol]["positions"]
            nspots = len(posxy)

            ngtthr = np.array(d[protocol]["ZScore"][stimno][dz]).sum()
            # print("Fraction: ", ngtthr / nspots)
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

    def score_events(
        self,
        fn,
        eventwindow=[0.000, 0.005],
        area_z_threshold=2.1,
        plotflag=False,
        force=False,
        celltype=None,
    ) -> (dict, bool):
        """
        Calculate various scores and information about events and maps

        Parameters
        ----------

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
        msg = f"Starting ScoreEvents on Protocol {str(fn):s}"
        cprint("green", f"    {msg:s}")
        Logger.info(msg)
        assert (
            len(eventwindow) == 2
        )  # need to be sure eventwindow is properly formatted on the call
        SH = shuffler.Shuffler()  # instance of the shuffling code.

        # currently, do not do the alternate or inverted data protocols.
        if str(fn).find("_alt") > 0 or str(fn).find("_signflip") > 0:
            Logger.warning(f"Protocol {fn!s} is an alternate or signflip protocol, skipping")
            return None, False
        if not Path(fn).is_file():
            msg = f"File not found: {str(fn):s}"
            cprint("red", msg)
            Logger.error(msg)
            return None, False
        with open(fn, "rb") as fh:  # get the data from the individual (not summary) file
            try:
                dfn = pickle.load(fh)
            except FileNotFoundError:
                msg = f"Problem reading on file: {str(fn):s}"
                cprint("red", msg)
                Logger.error(msg)
                return None, False

        protocols = sorted(list(dfn.keys()))

        # These come from the "results" in analyzemap...
        if not protocols:
            cprint("red", "    No protocols found")
            Logger.error("No protocols found")
            return None, False
        nmaps = len(protocols)
        if nmaps == 0:
            cprint("red", "    No maps found")
            Logger.error("No maps found")
            return None, False
        cprint("g", f"    {nmaps:d} maps found")
        basefile = Path(protocols[0])
        fp = basefile.parts
        outfn = Path("~".join(fp[0:3]) + ".pkl")
        eventfilekey = str(basefile.parent)
        cprint("yellow", f"Forceflag is {force:b}")
        if not force:  # check for, and just return existing data
            evndata = self.check_event_data(protocols)
            if bool(evndata):  # we have data, and no update required, so just return it
                cprint("magenta", f"File: {str(eventfilekey):s} is up to date")
                return evndata, False  # returns filled data in dict
        # otherwise we need to do the analysis on the .pkl file
        # analyze
        # if force is set, then we recalculate from the cell's own .pkl file
        # analyze
        cprint("g", f"{'*'*40}\n    Starting analysis")
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
        minprobability = [[]] * len(protocols)
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
        depression_ratio = [np.nan] * len(protocols)
        paired_pulse_ratio = [np.nan] * len(protocols)
        cprint("g", "    Initialized result arrays")
        if plotflag:
            rc = PH.getLayoutDimensions(nmaps, pref="width")
            P = PH.regular_grid(
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
            binstuff = np.arange(0, 0.6, 0.005)
            axl = P.axarr.ravel()

        for i_protocol, dxf in enumerate(protocols):
            evres, flag = self.score_protocol_events(
                dxf,
                evndata,
                dfn,
                eventwindow=eventwindow,
                area_z_threshold=area_z_threshold,
                celltype=celltype,
                i_protocol=i_protocol,
            )
            if evres is not None:
                evndata = evres
        return evres, True

    def get_protocol_scaling(self, protocol_name: str):
        sign = -1  # EPSC is negative
        pmode = "0"  # i == 0 clamp mode
        scale = 1.0
        if protocol_name.find("_IC_") >= 0:
            sign = 1
            scale = 1e3
            pmode = "I"
        elif protocol_name.find("_VC_") >= 0:  # includes "increase" protocol
            sign = -1.0
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
        return sign, scale, pmode

    def filter_celltypes(self, dxp, celltype):
        """Filter cell types based on the provided cell type.
        confirm that this is a cell type that we want to analyze

        Parameters
        ----------
        dxp : str or Path
            path to protocol
        celltype : str
            expected cell tupe

        Returns
        -------
        None or str
            The filtered cell type or None if it doesn't match.

        Raises
        ------
        ValuError
            Unknown cell type
        ValueError
            Unknown cell type is 0
        """
        sel_celltype = self._get_cell_information(dxp, "cell_type")
        sel_celltype = ephys.tools.map_cell_types.map_cell_type(sel_celltype)
        # sel_celltype = class_cell(sel_celltype)
        if sel_celltype is None:
            CP.cprint("r", "Celltype identification failed to match with canonical types")
            Logger.error(
                f"Protocol {dxp!s}  celltype: {sel_celltype!s} does not match input argument celltype: {celltype!s}"
            )
            return None
        CP.cprint("g", f"Successfully Identified cell class: {sel_celltype}")

        if sel_celltype not in valid_celltypes or len(sel_celltype) == 0:
            return None
        return sel_celltype

    def fix_event_list(self, this_eventlist, protocol_name):
        # repair missing stim information in "increase" files
        # Otherwise, get information from the eventlist.
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
        return this_eventlist, fixstim

    def combine_events(self, event_times, event_amplitudes):
        """
        Combine event times and amplitudes from a single trial of a given
        map into a list of tuples.

        Parameters
        ----------
        event_times : list of event times for ONE trial
            List of arrays containing event times for each trace.
        event_amplitudes : list of corresponding event amplitudes
            List of arrays containing event amplitudes for each trace.

        Returns
        -------
        list of list of tuples
            A list where each element corresponds to a trace and contains a list of
            tuples (time, amplitude, sweep#).
        """

        event_trace = [[i] * len(event_times[i]) for i in range(len(event_times))]
        event_trace = AK.flatten(event_trace, axis=None)
        evt = AK.flatten(event_times, axis=None)
        evt = [float(t) for t in evt]
        eva = AK.flatten(event_amplitudes, axis=None)
        eva = [float(a) for a in eva]
        combined_events = np.array(list(zip(evt, eva, event_trace)))
        return combined_events

    def compute_spont_windows(self, stim_times, tmax: float = 1.0):
        nstim = len(stim_times)
        wins = np.zeros((nstim + 1, 2))
        t0 = 0
        for ist, st in enumerate(stim_times):
            t1 = st["start"]
            wins[ist] = [t0, t1]
            # compute the start of the next spont window from the end of the current response window
            t0 = st["start"] + st["window_start"] + st["window_duration"]
        # include the final window from end of last stim to tmax
        wins[nstim] = [t0, tmax]
        return wins

    def compute_spontaneous_rate(self, combined_events: list, stim_times: list, tmax: float = 1.0):
        """
        Compute the spontaneous event rate from combined trial events.
        SR is measured from events occurring outside of stimulus / response windows.
        These are defined in the stim_times dictionary.
        Amplitudes are measured from all spontaneous events.

        ==================================================================
        We compute the SR for each trace in the current map,
        Then, accumulate the events across trials for summary statistics
        NOTE: The SR value here may differ from what was computed on the plots because
        we are using intervals rather tnan fitted event counts across the
        spont window duration. In addition, the whole of each trace is taken into
        consideration, with events in the response window removed (so as to not
        force the rate to be too high or based on too few intervals).
        The values are similar, but not exact.
        A better approach would be to take blank (no stim) traces alternating
        with the stim traces, and compute SR from those. Here, we work with the
        data that we have.

        Parameters
        ----------
        combined_events : list of tuples, event times, amplitudes and sweep numbers for ONE trial
            List of tuples (time, amplitude, sweep#) for all events in a trial.
        stim_times : list of float
            List of stimulus start times.
        tmax: float (default 1.0)
            Maximum time of the recording trace.

        Returns
        -------
        this_spont_rate (float), in Hz
        spontevent_times: a dict of event times. Each key is a tuple (trace, stim_index)
        spontevent_amps: a list of spontaneous event amplitudes (not keyed by trace)
        """

        spont_isis = []
        n_spont_amps = 0
        n_spont_isis = 0
        spontevent_amps = []
        spontevent_times = {}

        wins = self.compute_spont_windows(stim_times)
        # remove events in the evoked windows...
        spont_events = []
        spwin = []
        for iw, win in enumerate(wins):
            t0, t1 = win
            spwin_iw = np.argwhere(
                (np.array(combined_events)[:, 0] >= t0) & (np.array(combined_events)[:, 0] < t1)
            )
            spwin_iw = [int(iw[0]) for iw in spwin_iw]
            spont_events.extend(combined_events[spwin_iw])
        spont_events = np.array(spont_events)

        # verify spont_event detection procedure
        # f, ax = mpl.subplots(1,1)
        # ax.scatter(spont_events[:,0], np.ones_like(spont_events[:,0]),  color='blue',
        #            s=6)
        # ax.scatter(combined_events[:,0], 1.1*np.ones_like(combined_events[:,0]), color='red',
        #            s=6)
        # ax.set_ylim(0, 10)
        # mpl.show()
        # exit()
        # print("spwin: ", spwin)

        # reorganize spont_events by trace
        ntraces = int(np.max(combined_events[:, 2])) + 1
        spont_list = [None] * ntraces
        for tr in range(ntraces):
            tr_events = spont_events[spont_events[:, 2] == tr]
            if len(tr_events) > 0:
                spont_list[tr] = tr_events[:, 0]
            else:
                spont_list[tr] = []
        # assemble inter-event intervals per trace
        tr_isis = np.zeros(ntraces) * np.nan
        for tr in range(ntraces):
            se_in_trace = spont_events[spont_events[:, 2] == tr][:, 0]
            spontevent_amps.extend(spont_events[spont_events[:, 2] == tr][:, 1])
            if len(se_in_trace) > 1:
                tr_isis[tr] = np.mean(np.diff(se_in_trace))
        # compute summary statistics
        mean_spont_isi = np.nanmean(tr_isis)
        std_spont_isi = np.nanstd(tr_isis)
        n_spont_isis = len(tr_isis)
        n_spont_amps = len(spontevent_amps)
        mean_spont_amp = np.nanmean(spontevent_amps)
        std_spont_amp = np.nanstd(spontevent_amps)

        if n_spont_amps > 0:
            cprint(
                "g",
                f"    mean spont isi: {mean_spont_isi:.3f} (SD:{std_spont_isi:.3f} s N={n_spont_isis:d}) mean spont amp: {mean_spont_amp:.3e} pA (SD: {std_spont_amp:.3e}, N={n_spont_amps:d})",
            )
            # estimate of what is computed for the plots: print((1/len(spontevent_times[trial].keys()))*n_spont_amps/0.3)
        else:
            cprint("y", f"    No spont events")
        this_spont_rate = 1.0 / mean_spont_isi if mean_spont_isi > 0 else 0.0
        cprint("g", f"    Spontaneous rate: {this_spont_rate:.3f} Hz")
        return this_spont_rate, spont_events, spont_list

    def analyze_events_one_trial(self, combined: np.ndarray, stimtimes: list):
        """analyze_events_one_trial
        Extract spontaneous rates and event data
        for one map (one "trial" or repeat)

        Parameters
        ----------
        combined : np.ndarray
            an nx3 array of event times, amplitudes, and sweep numbers  for ONE trial
        stimtimes : list
            list of stimulus times and windows, each elemnt of list is for one stimulus

        Returns
        -------
        tuple
            (firstevent_latency, allevent_latency, sr, spontevent_amps, spontevent_times), True
            or None, False if no data
        """
        # arrays for the current trial/map
        print("analyze_events_one_trial: combined shape: ", combined.shape)

        if len(combined[:, 0]) == 0:
            cprint("red", "    No data in protocol?")
            sr = 0.0
            return None, False  # no data in this protocol?
        firstevent_latency = []
        allevent_latency = []
        spontevent_amps = []
        spontevent_times = []

        # now find the first event latenxy for each stimulus, for each sweep/spot
        # and harvest all event latencies in the window too
        for ifsl, st in enumerate(stimtimes):
            t0 = st["start"] + st["window_start"]
            t1 = st["start"] + st["window_start"] + st["window_duration"]
            inwin = np.nonzero((t0 < np.array(combined[:, 0])) & (np.array(combined[:, 0]) <= t1))[
                0
            ]
            if len(inwin) == 0:  # no events detected
                continue
            firstevent_latency.append(combined[inwin, 0] - st["start"])
            allevent_latency.extend([t - st["start"] for t in combined[inwin, 0]])
        sr, spontevents, spont_list = self.compute_spontaneous_rate(combined, stimtimes)
        return (firstevent_latency, allevent_latency, sr, spontevents, spont_list), True

    def plot_probs(
        self,
        evtimes: list,
        probabilities: list,
        spont_list_all_trials: list,
        pvalues: dict,
        posxy: np.ndarray,
        dxp: Path,
        sel_celltype: str,
        ntrials: int,
    ):
        """plot_probs Graph some of the results for verification

        Parameters
        ----------
        evtimes : list
            list of event times pert trial
        spont_list_all_trials : list
            list of spontaneous event times per trial
        pvalues : dict
            dictionary of p-values for each event
        posxy : np.ndarray
            array of x, y positions for each event
        dxp : Path
            data path
        sel_celltype : str
            the cell type
        ntrials : int
            number of trials
        """

        f, ax = mpl.subplots(1, 5, figsize=(12, 4))
        f.suptitle(f"Protocol: {str(dxp):s}  Celltype: {sel_celltype:s}")
        ev_flat = AK.flatten(evtimes, axis=None).to_list()

        # plot histogram of event times
        # two overlapping plots: spont events in black, evoked events on top in red
        spx = AK.flatten(spont_list_all_trials, axis=None).to_list()
        evx = [et for et in ev_flat if et not in spx]
        bins = np.arange(0, 0.6, 0.005)  # 5 ms bins up to 600 ms
        ax[0].hist(
            spx,
            bins=bins,
            color="grey",
            alpha=0.5,
            histtype="stepfilled",
            align="right",
        )
        ax[0].hist(evx, bins=bins, color="red", alpha=1, histtype="stepfilled", align="right")
        ax[0].set_title("Event Times Histogram")
        ax[0].set_xlabel("Time (s)")
        ax[0].set_ylabel("Count")
        # plot hisotgram of pvalues
        pval = [1 - pvalues[k] for k in pvalues.keys()]
        ax[3].hist(pval, bins=50, color="blue", alpha=0.7)
        ax[3].set_title("Event Probabilities Histogram")
        ax[3].set_xlabel("Probability")
        ax[3].set_ylabel("Count")

        # plot spot raster
        for trial in range(ntrials):
            for trnum in range(len(evtimes[trial])):
                ax[2].scatter(
                    evtimes[trial][trnum],
                    [trnum] * len(evtimes[trial][trnum]),
                    s=5,
                    c="red",
                    alpha=1,
                    marker="o",
                )
            for trnum in range(len(spont_list_all_trials[trial])):
                ax[2].scatter(
                    spont_list_all_trials[trial][trnum],
                    [trnum] * len(spont_list_all_trials[trial][trnum]),
                    s=3,
                    c="black",
                    alpha=0.75,
                    marker="o",
                )
        # Combine the X and Y coordinates into a list of (x, y) pairs
        points = np.vstack([posxy[:, 0].ravel(), posxy[:, 1].ravel()]).T
        pvals = []
        spot_patches = []
        prob_colors = []
        # now assign colors based on pvalues to spots

        for i in range(len(evtimes[0])):
            if (i, 0) in pvalues.keys():
                pvals.append(pvalues[(i, 0)])
            else:
                pvals.append(1)
            # if pvals[-1] == 1.0:
            #     col = "grey"
            # elif pvals[-1] > 0.1 and pvals[-1] <= 1.0:
            #     col = "blue"
            # elif pvals[-1] > 0.01 and pvals[-1] <= 0.1:
            #     col = "cyan"
            # elif pvals[-1] > 0.001 and pvals[-1] <= 0.01:
            #     col = "green"
            # elif pvals[-1] > 0.0001 and pvals[-1] <= 0.001:
            #     col = "yellow"
            # elif pvals[-1] < 0.0001:
            #     col = "orange"
            # else:
            #     col = "red"
            spot_patch = matplotlib.patches.Circle(
                (posxy[i, 0], posxy[i, 1]),
                radius=21e-6,
                fill=True,
                edgecolor="k",
                lw=0.5,
                # facecolor=col,
                alpha=0.8,
            )
            spot_patches.append(spot_patch)
        print(f"Pvals: {np.min(pvals):.3e}, {np.max(pvals):.3e}")
        log_p = [-np.log10(p) if p > 0 else 1.0 for p in pvals]
        print(f"-log(p): {np.min(log_p):.3e}, {np.max(log_p):.3e}")
        cmap = mpl.cm.CMRmap
        norm = mpl.Normalize(vmin=np.min(log_p), vmax=1.0 ) # np.max(log_p))
        pc = matplotlib.collections.PatchCollection(spot_patches, cmap=cmap, 
                                                    norm=norm, match_original=True)
        ax[1].add_collection(pc)
        pc.set_array(log_p)
        ax[1].autoscale_view()
        cbar = mpl.colorbar(pc)

        # lspsmap = ax[1].scatter(points[:, 0], points[:, 1], c=pvals, s=50, cmap="viridis", #vmin=0.0, vmax=1.0,
        # mpl.colorbar(lspsmap, label="Log10(Probability)")

        ax[1].set_aspect("equal")
        ax[1].set_title("Event Probabilities Map")

        # ax[1].scatter(points[:,0], points[:,1], c=probabilities, s=50)
        mpl.show()

    def score_protocol_events(
        self,
        dxf: Union[str, Path],
        evndata: dict,
        dfn: dict,
        eventwindow: tuple,
        area_z_threshold: float,
        celltype: Union[str, None] = None,
        i_protocol: int = 0,
        plotflag: bool = True,
    ):
        """
        Score the given protocol according to whether there are "significant" evoked events.
        1. Check that the protocol is valid for the analysis we need to do
        2. Get the protocol data, and adjust the event list. Find
        events in a respons window after each stimulus.
        3. Accumulate event amplitudes and times across all trials
        4. Calculate spontaneous rate from events outside of stimulus windows
        5. Compute scores (calculate p expectancy based on spont rate)
        6. opionally plot the event histogram, map with p values. raster, and histogram of p values

        Parameters
        ----------
        dxf : str or Path
            path to protocol file
        evndata : dict of event-related data
            accumulated event data
        dfn : dict of file information, used to extract positions.
        eventwindow : tuple
            (start time, duration) of the event detection window after each stimulus
        area_z_threshold : float
            threshold for area fraction calculation
        celltype : str or None
            if set, only analyze for a particular cell type
        i_protocol : int
            index of the protocol in the list

        """
        assert celltype is not None
        lat_protocols = [
            "_VC_10Hz",
            "Single",
            "single",
            "_VC_weird",
            "_VC_2mW",
            "_VC_1mW",
            "_VC_00",
            "_range test",
            "_VC_increase",
        ]
        if all(
            [str(dxf).find(p) < 0 for p in lat_protocols]
        ):  # limit the protocols that we will use for
            cprint("red", f"    Protocol Excluded on type: {str(dxf):s}")
            return None, False

        # get cell type from protocol path
        dxp = Path(dxf)
        dx = str(dxp.parent)
        dxpl = dxp.parts
        dxp = Path(*dxpl)  # dxpl[-4], dxpl[-3], dxpl[-2], dxpl[-1])
        protocol_name = str(dxp.parts[-1])
        skip = [
            "2017.02.14_000/slice_000/cell_000/Map_NewBlueLaser_VC_single_test_000",
            "2017.02.14_000/slice_000/cell_001/Map_NewBlueLaser_VC_single_MAX_000",
            "2017.02.14_000/slice_000/cell_001/Map_NewBlueLaser_VC_single_MAX_001",
            "2017.02.14_000/slice_000/cell_001/Map_NewBlueLaser_VC_single_MAX_002",
            "2017.02.14_000/slice_000/cell_001/Map_NewBlueLaser_VC_single_MAX_003",
        ]
        # if str(dxp) in skip:
        #     cprint("y", f"    Skipping known bad protocol: {str(dxp):s}")
        #     return None, False
        if str(dxp) not in [
            # "2017.02.14_000/slice_000/cell_001/Map_NewBlueLaser_VC_single_MAX_003"
            # "2018.09.26_000/slice_001/cell_002/Map_NewBlueLaser_VC_10Hz_003"
            "2017.05.01_000/slice_000/cell_000/Map_NewBlueLaser_VC_10Hz_000"
        ]:
            return None, False
        sel_celltype = self.filter_celltypes(dxp, celltype)
        if sel_celltype is None:
            return None, False

        cprint("g", f"    Proceeding with celltype:  <{sel_celltype:s}> and protocol: {dxp!s}")
        temperature = self._get_cell_information(dxp, "temperature")
        protocol, evl = self._get_cell_protocol_data(dxp.parent)
        if evl is None or evl[str(dxp)] is None:
            cprint("red", f"    No data in protocol {str(dxp):s}")
            raise ValueError()
        this_eventlist = evl[str(dxp)]
        if "stimtimes" not in list(this_eventlist.keys()):
            cprint(
                "red", "    Missing 'stimtimes' in event list: " + str(list(this_eventlist.keys()))
            )
        this_eventlist, fixstim = self.fix_event_list(this_eventlist, protocol_name)
        sign, scale, pmode = self.get_protocol_scaling(protocol_name)

        # get the positions
        try:
            posxy = dfn[str(dxp)]["positions"]
        except KeyError:
            cprint("r", f"    No positions found in protocol {str(dxp):s}")
            return None, False

        # build list of dicts holding the stimulus times and windows
        # for this protocol
        stimtimes = []
        for n in range(this_eventlist["stimtimes"]["npulses"]):
            st0 = this_eventlist["stimtimes"]["starts"][n]
            stimtimes.append(
                {"start": st0, "window_start": eventwindow[0], "window_duration": eventwindow[1]}
            )
            tend = st0 + np.sum(eventwindow)

        # accumulate the event amplitudes and matching times for this protocol, across all trials
        reader = MEDR.Reader(evl[dxf])
        evamps = []

        ntrials = reader.get_ntrials()
        nspots = len(posxy) / ntrials
        # area_fraction is the fractional area of the map where the scores exceed 1SD above
        # the baseline (Z Scored; charge based)
        ngtthr = (np.array(this_eventlist["ZScore"][0]) > area_z_threshold).sum()
        # area_fraction = float(len([d[dx]['ZScore'][-1] > area_z_threshold]))/float(len(d[dx]['positions']))
        area_fraction = ngtthr / nspots
        cprint("g", f"    Area fraction: {area_fraction:0.3f}")
        cprint("c", f"    Ntrials: {reader.get_ntrials():d}")

        # ==================
        evtimes = []
        evamps = []
        combined_trial = []
        # accumulated values across all trials
        firstevent_latency_all_trials = []
        allevent_latency_all_trials = []
        spontevent_amps_all_trials = []
        spontevent_times_all_trials = []
        spont_list_all_trials = []
        tpr = []
        tevt = []
        for trial in range(ntrials):  # trials
            trial_events = reader.get_events()[
                trial
            ]  # trial events is a Mini_Event_Summary dataclass
            if trial_events is None:
                cprint("r", "No trial events found")
                evtimes.append(np.nan)
                evamps.append(np.nan)
                continue
            cprint("g", f"Analyzing events in trial: {trial:d}")
            dt = reader.get_sample_rate(trial)
            evt = reader.get_trial_event_onset_times(trial, trial_events.onsets)
            eva = reader.get_trial_event_amplitudes(trial, trial_events.smpkindex)
            # assert isinstance(evt, (list, np.ndarray))
            # assert isinstance(eva, (list, np.ndarray))
            evtimes.extend([evt])
            evamps.extend([eva])
            combined = self.combine_events(evt, eva)
            combined_trial.append(combined)

            # now analyze one trial
            res, resflag = self.analyze_events_one_trial(combined=combined, stimtimes=stimtimes)
            if not resflag:
                cprint("r", f"    No valid events found in trial {trial:d}")
                continue
            else:  # unpack results
                (
                    firstevent_latency,
                    allevent_latency,
                    this_spont_rate,
                    spontevents,
                    spont_list,
                ) = res
                firstevent_latency_all_trials.extend(firstevent_latency)
                allevent_latency_all_trials.extend(allevent_latency)
                spontevent_amps_all_trials.extend(spontevents[:, 1])
                spontevent_times_all_trials.extend(spontevents[:, 0])
                spont_list_all_trials.append(spont_list)  # keep list of spont events by trace

            if len(evtimes) > 0:
                # mscore_n, mscore, mean_ev_amp = SH.shuffle_score(evp, stimtimes, nshuffle=5000, maxt=0.6)  # spontsonly...
                CP.cprint("g", "Computing PoissonScore")
                mscore_n, probabilities = EPPS.PoissonScore.score(
                    evt, amplitudes=eva, rate=this_spont_rate, tMax=0.6, normalize=True
                )
            else:
                mscore_n = np.ones(len(stimtimes))
                probabilities = 1.0
            evt_flat = []
            for j, p in enumerate(evt):
                evt_flat.extend(p)
            evt_flat = np.array(evt_flat)
            ievt = np.argsort(evt_flat)
            tpr.append(probabilities[ievt])
            tevt.append(evt_flat[ievt])

            print(f"Probabilities len: {len(probabilities)}, len(evt): {len(evt_flat)}")
            print(evt_flat)
            print("Probabilities: ", tpr)
            # mpl.plot(tevt[0], tpr[0], '-')
            # mpl.show()
            # exit()
            cprint("g", f"    m score: {mscore_n:6.4f}, spontrate: {this_spont_rate:.3f} Hz")
            # print(
            #     "Probs: ",
            #     probabilities,
            #     len(probabilities),
            #     np.min(probabilities),
            #     np.max(probabilities),
            # )

            # get the probabilities for the responses after the first stimulus:
            pvalues = {}  # dict, keys are trace numbers, stim numbers
            for trial in range(ntrials):
                for ifsl, st in enumerate(stimtimes):
                    t0 = st["start"] + st["window_start"]
                    t1 = st["start"] + st["window_start"] + st["window_duration"]
                    inwin = np.nonzero(
                        (t0 < np.array(combined_trial[trial][:, 0]))
                        & (np.array(combined_trial[trial][:, 0]) <= t1)
                    )[0]
                    if len(combined_trial[trial][inwin]) > 0:
                        spots = combined_trial[trial][inwin][:, 2]
                        for spot in spots:
                            pvalues[(int(spot), ifsl)] = float(probabilities[int(spot)])
                    else:
                        print("?? spot: ", combined_trial[trial][inwin], len(combined_trial[trial][inwin]))
                        continue  # missing spot information
                        # spot = combined_trial[trial][inwin][2]
                        # # print("spot2: ", spot)
                        # pvalues[(int(spot[2]), ifsl)] = 1.0
        if plotflag:
            self.plot_probs(
                evtimes=evtimes,
                spont_list_all_trials=spont_list_all_trials,
                probabilities=probabilities,
                pvalues=pvalues,
                posxy=posxy,
                dxp=dxp,
                sel_celltype=sel_celltype,
                ntrials=ntrials,
            )
        print(f"Mscore: {mscore_n:.3f}")
        exit()

        mscore = mscore_n
        mean_ev_amp = 1.0
        if not isinstance(mscore, list):
            mscore = [mscore]
        m_shc = np.mean(mscore)
        s_shc = np.std(mscore)
        # print('mscore, : ', mscore, mean_ev_amp)

        # find the probability in the response windows for each stimulus
        winlen = 0.010

        # build a dictinary
        ev_prob_response_events = {k["start"]: [] for k in stimtimes}
        ev_probabilities = {}  # actual probabilities
        for ist, stime in enumerate(stimtimes):
            t0 = stime["start"] + stime["window_start"]
            t1 = t0 + stime["window_duration"]
            response_events = np.where(
                (np.array(evtimes_flat) > t0) & (np.array(evtimes_flat) <= t1)
            )[0]
            # print("\nStime: ", stime['start'], "response events: ", response_events)
            res = [float(evtimes_flat[e]) for e in response_events]
            pro = [float(probabilities[e]) for e in response_events]
            if len(response_events) > 0:
                ev_prob_response_events[stime["start"]] = res
                ev_probabilities[stime["start"]] = pro
            else:
                ev_probabilities[stime["start"]] = [np.nan]
                ev_prob_response_events[stime["start"]] = []
        # at this point, we have the event times in a dictionary, by stimulus start time,
        # and the corresponding probabilities in a second dictionary of matching structure
        # print("\nev_response_events: ", ev_prob_response_events)
        # print("\nev_probs: ", ev_probabilities)
        # for ist, evre in enumerate(ev_prob_response_events):
        #     if len(evre) > 0:
        #         meanp = np.mean(probabilities[evre])
        #     else:
        #         meanp = 1.0
        # now we will flatten the list of events across all stimuli
        ev_prob_response_events_flat = []
        for istim, stime in enumerate(ev_prob_response_events):
            ev_prob_response_events_flat.extend(ev_prob_response_events[stime])

        if nspont > 10:
            mean_spont_amp = np.nanmean(spontaneous_amplitudes)
        else:
            cprint(
                "red",
                f"    Using Mean spont from table for : {str(sel_celltype):s}, temp= {str(temperature):s}C",
            )
            key = (sel_celltype, temperature)
            if key not in self.expt["mean_spont_amplitude_by_cell"]:
                mean_spont_amp = 20e-12  # A: use a default value

            else:
                print(
                    "mean spont amp by cell retrieved: ",
                    self.expt["mean_spont_amplitude_by_cell"][(sel_celltype, temperature)],
                )
                mean_spont_amp = (
                    self.expt["mean_spont_amplitude_by_cell"][(sel_celltype, temperature)] * 1e-12
                )  # use the mean value for the cell type and temp, scale to A
        # print("stim times: ", stimtimes)
        detevt_n, detevt, detamp = SH.detect_events(
            event_times=evtimes,
            event_amplitudes=evamps,
            stim_times=stimtimes,
            mean_spont_amp=mean_spont_amp,
        )
        if detevt_n == 0:
            CP.cprint("y", f"            No events detected in protocol?")
            Z = 0.0
            return None, False
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
        print(f"            Mean evoked amp (trial 0): {np.nanmean(detamp) * scale:.2f} pA", end="")
        print(f" SD {np.std(detamp) * scale:.2f} N={np.shape(detamp)[0]:3d}")
        # print('mscore: ', mscore)
        event_qcontent = np.zeros(len(mscore))
        largest_event_qcontent = np.zeros(len(mscore))

        for j, m in enumerate(mscore):
            if mscore[j] > 100:
                mscore[j] = 100.0  # clip mscore
            # try:
            if mean_ev_amp == 0.0:
                continue
            spont_evt_amps = np.nanmean(spontaneous_amplitudes)
            if j >= len(detamp):
                continue

            if sign == -1:
                det = detamp[j] < spont_evt_amps
                if det.any():
                    event_qcontent[j] = (
                        np.nanmean(detamp[j][detamp[j] < spont_evt_amps]) / mean_ev_amp
                    )
                    largest_event_qcontent[j] = np.min(detamp[j]) / mean_ev_amp
            else:
                det = detamp[j] > spont_evt_amps
                if det.any():
                    event_qcontent[j] = (
                        np.nanmean(detamp[j][detamp[j] > spont_evt_amps]) / mean_ev_amp
                    )
                    largest_event_qcontent[j] = np.max(detamp[j]) / mean_ev_amp

            if mscore[j] > 0:

                # print("probs: ", probs)
                print(
                    f"        Stim/trial {j:2d} ShuffleScore= {mscore[j]:6.4f} Lowest Shuffle Prob = {np.min(ev_prob_response_events_flat):6.3g} EventP: {detevt[j]:6.3e} Z: {Z:7.4f}, P(Z): {(1.0-self.z2p(Z)):.3e}",
                    end="",
                )
                print(
                    f" Event Amp re Spont: {event_qcontent[j]:g}  Largest re spont: {largest_event_qcontent[j]:7.4f}"
                )
            else:
                print(
                    f"        Stim/Trial {j:2d} ShuffleScore= {mscore[j]:6.4f} Lowest Shuffle Prob = {np.min(ev_prob_response_events_flat):6.3g} EventP: {detevt[j]:6.3e} P(Z): {(1.0-self.z2p(Z)):.3e}"
                )

        mscore[mscore == 0.0] = 100
        # build result arrays
        allprotos[i_protocol] = dxp
        shufflescore[i_protocol] = Z
        for ist, stime in enumerate(stimtimes):
            sts = stime["start"]
            if len(ev_probabilities[sts]) > 0:
                meanprobability[i_protocol].append(np.mean(ev_probabilities[sts]))
                minprobability[i_protocol].append(np.min(ev_probabilities[sts]))

            else:
                meanprobability[i_protocol] += 1.0
                minprobability[i_protocol] += 1.0

        eventp[i_protocol] = detevt
        event_amp[i_protocol] = detamp

        scores[i_protocol] = 1.0 - self.z2p(Z)  # take max score for the stimuli here
        spont_rate[i_protocol] = this_spont_rate
        spontevent_amp[i_protocol] = spontaneous_amplitudes

        validdata[i_protocol] = True
        event_Q_Content[i_protocol] = event_qcontent
        event_Largest_Q_Content[i_protocol] = largest_event_qcontent
        datamode[i_protocol] = pmode
        area_fraction_Z[i_protocol] = area_fraction

        positions[i_protocol] = posxy

        depression_ratio, paired_pulse_ratio = self.compute_depression_ppr(stimtimes, dxp)

        if plotflag:
            evt = np.hstack(np.array(evtimes).ravel())
            n, bins, patches = axl[i_protocol].hist(
                evt, bins=binstuff, histtype="stepfilled", color="k", align="right"
            )
            for s in this_eventlist["stimtimes"]["start"]:
                axl[i_protocol].plot([s, s], [0, 40], "r-", linewidth=0.5)
            axl[i_protocol].set_ylim(0, 40)
            axl[i_protocol].set_title(
                str(dxp.parts[-1]), fontsize=10
            )  # .replace(r"_", r"\_"), fontsize=10)

        if plotflag:
            fxl = []
            firstevent_latency = np.array(firstevent_latency)
            for ifl in firstevent_latency:
                if ifl is None:
                    continue
                fxl.extend(ifl)
            afl = []
            for ifl in allevent_latency:
                if ifl is None:
                    continue
                afl.extend(ifl)
            fig, ax = mpl.subplots(2, 1)
            fbins = np.linspace(0, eventwindow[1], int(eventwindow[1] / 0.00010))
            abins = np.linspace(0, eventwindow[1], int(eventwindow[1] / 0.00010))
            if len(fxl) < 20:
                fbins = "auto"
            ax[0].hist(np.array(fxl), fbins, histtype="bar", density=False)
            if len(afl) < 20:
                abins = "auto"
            ax[1].hist(afl, abins, histtype="bar", density=False)
            mpl.show()

            P.figure_handle.suptitle(str(Path(dx).parent))  # .replace(r"_", r"\_"))
            mpl.show()

        result = {
            "scores": scores,
            "SR": this_spont_rate,
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

    def compute_depression_ppr(self, stim_times, dxp):

        if len(stim_times) > 1:  # need at least 2 stimuli to get depression/pp ratios
            drdata = self.depression_ratio(
                evfile=dxp,
                stim_N=4,
            )  # compute depression ratio
            depression_ratio[i_protocol] = drdata["ratio"]
            paired_pulse_ratio[i_protocol] = drdata["pp_ratio"]
        else:
            depression_ratio[i_protocol] = np.nan
            paired_pulse_ratio[i_protocol] = np.nan

        for i, dr in enumerate(depression_ratio):
            if dr is None:
                dr = np.nan
        paired_pulse_ratio[paired_pulse_ratio == None] = np.nan
        # for i, ppr in enumerate(paired_pulse_ratio):
        #     print("i, ppr: ", i, ppr)
        #     if ppr is None:
        #         ppr = np.nan
        depression_ratio = np.array(depression_ratio)
        paired_pulse_ratio = np.array(paired_pulse_ratio)
        print("depression_ratio: ", len(depression_ratio), [dr for dr in depression_ratio])
        print("paired_pulse_ratio: ", len(paired_pulse_ratio), [ppr for ppr in paired_pulse_ratio])
        return depression_ratio, paired_pulse_ratio

    def listcells(
        self,
        celltype: Union[str, None] = None,
        cmds=None,
        day=None,
        day_after=None,
        day_before=None,
        slice_cell=None,
    ):
        """
        List data for selected cells by cell/protocol
        Note: we grab the database to get the cell notes for this cell, and
        print out the shufflescore for each stimulus

        Parameters
        ----------
        celltype: str (default: []])
            Cell type to

        Returns
        -------
        Nothing

        """
        assert self.db is not None
        if not isinstance(celltype, list):
            celltype = [celltype]
        tw = textwrap.TextWrapper(initial_indent=" " * 3, subsequent_indent=" " * 8, width=80)

        sumtab = OrderedDict()
        # print("isting cells")
        #########
        # FOR EACH Cell in the database
        # print('exclusions: ', exclusions)
        if self.events is None:
            cprint("r", "No events found in this cell's data")
            return
        else:
            cprint("c", f"Found {len(self.events):d} events in this cell's data")

        for cell_id in sorted(list(self.events.keys())):  # list of cells
            if (
                cell_id in self.expt["Excluded_maps"]
            ):  # excluded based on day/slice/cell, not just protocol
                cprint("yellow", f"Cell excluded from table: {str(efd):s}")
                continue
            match = self.check_day_slice_cell(
                fn=cell_id,
                day=day,
                day_after=day_after,
                day_before=day_before,
                slice_cell=slice_cell,
            )
            if not match:
                raise ValueError()
            else:
                cprint("cyan", f"Match : {str(cell_id):s}")
            cellclass = class_cell(str(self.events[cell_id]["celltype"]).lower())
            if celltype != [None]:
                if cellclass not in celltype:
                    cprint(
                        "yellow",
                        f"cellclass {str(cellclass):s} not in celltype {str(celltype):s}, {str(cell_id):s}",
                    )
                    continue
            gr = "Z"
            basename = cell_id.split("_")[0]
            if self.coding is not None:
                # print("coding: ", coding)
                # print("basename, coding: ", basename, coding.keys(), coding.ID)
                # print(coding.index)
                # check to see if date is in the index
                group = None
                for name in self.coding.date:
                    if not pd.isnull(name):
                        testname = Path(name).name  # only get the actual date name, not the path
                        if testname == basename:
                            group = self.coding[self.coding.date == testname].Group
                            if pd.isna(group.values) or len(group.values) == 0:
                                group = "X"
                            else:
                                group = group.values[0]
                            break
                if group is None:
                    cprint("y", f"Base file (date) {basename:s} not in coding database")
                    continue

            cprint("cyan", f"\n Cell: {cell_id:42s} {cellclass:18s} Group: {group:s}")
            sign = -1
            if "signflip" in cell_id or "alt1" in cell_id or "alt2" in cell_id:
                continue
                # sign = 1
            # dblink = Path(efd).parts
            # day_x = self.db.loc[(self.db['date'] == dblink[0]) & (self.db['slice_slice'] == dblink[1]) & (self.db['cell_cell'] == dblink[2])]
            notestr = "".join([s for s in self._get_cell_information(cell_id, "cell_notes")])
            notes = tw.wrap(f"Notes: {notestr:s}")
            for n in notes:
                print(n)

            ##################################
            # For each protocol for this cell:
            cell_maxev = 0.0
            cell_maxscore = 0.0
            cell_maxspont = 0.0
            # clean the protocol list first:

            cell_protocols = sorted(
                [str(x) for x in self.events[cell_id]["protocols"] if x not in [None, "None"]]
            )
            for i_protoindex, p in enumerate(sorted(cell_protocols)):
                if self.events[cell_id]["protocols"][i_protoindex] == None or str(
                    self.events[cell_id]["protocols"][i_protoindex]
                ) in [None, ""]:
                    cprint("yellow", f"nothing in protocol {p:s}")
                    continue
                if (
                    str(self.events[cell_id]["protocols"][i_protoindex]) in exclusions
                ):  # protocol exclusion
                    cprint(
                        "yellow",
                        f"Protocol Excluded: {str(self.events[cell_id]['protocols'][i_protoindex]):s}",
                    )
                    continue  # check the protocol for inclusion

                cprint("magenta", f"    {self.events[cell_id]['protocols'][i_protoindex].stem:40s}")

                if self.events[cell_id]["shufflescore"][i_protoindex] > 0.0:
                    score = np.array(
                        self.events[cell_id]["shufflescore"][i_protoindex]
                    )  # calculate the scores
                else:
                    score = np.zeros_like(np.array(self.events[cell_id]["eventp"]))
                # print(len(self.events[cell_id]["eventp"]))
                # print("score: ", score)
                # print(f"       Max shuffle score in map {np.max(score):10.3g}")
                # print(type(score), score.shape)
                if not isinstance(score, (np.ndarray, list)) or score.shape == ():
                    print(f"   {0:d}: {score:.3g}")
                else:
                    for k, scr in enumerate(score):
                        print(f"  {k:d}: {scr:.3g}", end="")
                    print()
                # print(f"  {:s}'.format(self.events[cell_id]['protocols'][i].stem, str(score))")
                if "_VC_" in str(
                    p
                ):  # accumulate the largest score across all the protocols for this cell
                    cell_mscore = np.max(np.max(score))
                    if cell_mscore > cell_maxscore:
                        cell_maxscore = cell_mscore

                print("       Event Amplitudes for each stimulus: ")

                if self.events[cell_id]["protocols"][i_protoindex] == None or str(
                    self.events[cell_id]["protocols"][i_protoindex]
                ) in [None, ""]:
                    continue
                # print(len(events[cell_id]["event_amps"][i_protoindex))
                # print(events[cell_id]["event_amps"][i_protoindex])
                # print(np.mean(events[cell_id]['spont_amps'][i_protoindex]))
                # print(f"     {self.events[cell_id]['protocols'][i_protoindex].stem:45s}  ")
                print(f"           {'Stim N':6s}  {'Ev Amp':^10s}   (N)    {'Sp Amp':^10s}   (N)")
                if self.events[cell_id]["mode"][i_protoindex] == "V":
                    suffix = "A"
                else:
                    suffix = "V"

                ev = self.events[cell_id]["event_amps"][i_protoindex]
                nev = len(ev)
                if len(ev) > 0:
                    an_event = np.max(sign * ev)

                    if suffix == "A":  # only voltage clamp data
                        if an_event > cell_maxev:
                            cell_maxev = an_event
                if nev > 0:
                    print(
                        f"                {eng_string(an_event, format='%7.3f', si=True, suffix=suffix):s}  ({nev:4d})  ",
                        end="",
                    )
                else:
                    print(f"              0.000nA  ({nev:4d})  ", end="")
                nsp = len(self.events[cell_id]["spont_amps"][i_protoindex])
                if nsp == 1 and self.events[cell_id]["spont_amps"][i_protoindex] == [0.0]:
                    nsp = 0
                    # cprint('red', self.events[cell_id]['spont_amps'][ii_protoindex])
                spamp = np.nanmean(sign * self.events[cell_id]["spont_amps"][i_protoindex])
                if suffix == "A":
                    if spamp > cell_maxspont:
                        cell_maxspont = spamp
                    print(
                        f"{eng_string(spamp, format='%7.3f', si=True, suffix=suffix):s} ({nsp:d})"
                    )

                print(f"")
                # print('    EV: ', ev)
                sumtab[cell_id] = {
                    "maxscore": cell_maxscore,
                    "maxev": cell_maxev * 1e12,
                    "spontAmp": cell_maxspont * 1e12,
                }

        print(
            f"\n{'Cell':^48s}  {'cellclass':^14s} {'max score':^11s} {'log score':^11s} {'maxev':^10s} {'Spont':^10s}"
        )
        for f in sumtab:
            d = sumtab[f]
            print(
                f"{f:48s} {cellclass:^14s} {d['maxscore']:9.1f} {np.log10(np.clip(d['maxscore'], 0.1, 10000.)):9.3f} {d['maxev']:10.1f} {d['spontAmp']:10.1f}"
            )

    def flatten(self, l):
        for el in l:
            if isinstance(el, collections.Iterable) and not isinstance(el, (str, bytes)):
                yield from self.flatten(el)
            else:
                yield el

    def depression_ratio(self, evfile, measuretype: str = "depression_ratio", stim_N=5):
        verbose = False
        # assert measuretype in ["depression_ratio", "paired_pulse_ratio"]
        # if measuretype == "paired_pulse_ratio":
        #     stim_N = 2
        proto = pathlib.PurePosixPath(evfile).name  # make sure of type
        protocol, protodata = self._get_cell_protocol_data(evfile.parent)
        for p in protodata.keys():
            if str(p).endswith(str(proto)):
                proto = p
                break
        if proto not in list(protodata.keys()):
            return None, None, None
        envx = protodata[str(proto)]

        # print('stimtimes: ', envx['stimtimes'])
        if len(envx["stimtimes"]["starts"]) < stim_N:
            # print('not enough stim times', envx['stimtimes']['start'])
            # print('in ', str(proto))
            return {"ratio": np.nan, "amplitudes": np.nan, "eventtimes": np.nan}
        # print('Enough stim times', self.events[proto]['stimtimes']['start'])
        # print('in ', str(p))
        self.protodata = protodata
        stimstarts = envx["stimtimes"]["starts"]
        stimwins = [[s + 0.001, s + 0.011] for s in stimstarts]

        # protoevents = envx["events"]
        k = range(stim_N)
        amps = dict((ik, []) for ik in k)
        tevs = dict((ik, []) for ik in k)

        cprint("cyan", f"{str(evfile):s},    proto: {str(proto):s}")
        reader = MEDR.Reader(envx)
        if verbose:
            print("proto: ", str(proto))
            print(" # protocol has # events: ", len(events))
        evamps = []
        evtimes = []
        nspots = reader.get_ntrials()
        for trial in range(reader.get_ntrials()):  # trials
            if trial >= len(envx["ZScore"]):
                continue
            spots = list(np.where(envx["ZScore"][trial] > 2.1)[0])
            if len(spots) == 0:  # no significant events
                print("no spots in trial: ", trial)
                continue
            trial_events = reader.get_events()[trial]
            dt = reader.get_sample_rate(trial)
            if trial_events is None:
                continue
            event_t = reader.get_trial_event_smpks_times(trial, trial_events.smpkindex)
            event_a = reader.get_trial_event_amplitudes(trial, trial_events.smpkindex)
            event_t = np.array([t for j in range(len(event_t)) for t in event_t[j]])
            event_a = np.array([t for j in range(len(event_a)) for t in event_a[j]])

            for jwin in range(stim_N):
                for iev in range(event_t.shape[0]):
                    if (event_t[iev] >= stimwins[jwin][0]) and (event_t[iev] < stimwins[jwin][1]):
                        # print('jwins, iev, event_t, event_a, stimwins: ', jwin, iev, event_t[iev], event_a[iev], stimwins[jwin])
                        amps[jwin].append(event_a[iev])
                        tevs[jwin].append(event_t[iev])
            # evtimes.extend(evt)
            # evamps.extend(eva)

        # for trial in range(len(protoevents)):
        #     if trial >= len(envx["ZScore"]):
        #         continue
        #     spots = list(np.where(envx["ZScore"][trial] > 2.1)[0])
        #     if len(spots) == 0:  # no significant events
        #         continue
        #     for ispot in spots:
        #      # now for all spots in this trial, get the events for each stimulus so we can average them
        #          # handle both orders (old and new...)
        #         if "smpksindex" in protoevents[trial].keys():
        #             event_t = (
        #                 np.array(protoevents[trial]["smpksindex"][ispot]) * 5e-5
        #             )  # time
        #             event_a = np.array(protoevents[trial]["smpks"][ispot])  # amplitude
        #         else:
        #             event_t = (
        #                 np.array(protoevents[trial][ispot]["smpksindex"]) * 5e-5
        #             )[0]
        #             event_a = np.array(protoevents[trial][ispot]["smpks"])[0]
        #         for jwin in range(stim_N):
        #             for iev in range(event_t.shape[0]):
        #                 if (event_t[iev] >= stimwins[jwin][0]) and (
        #                     event_t[iev] < stimwins[jwin][1]
        #                 ):
        #                     # print('jwins, iev, event_t, event_a, stimwins: ', jwin, iev, event_t[iev], event_a[iev], stimwins[jwin])
        #                     amps[jwin].append(event_a[iev])
        #                     tevs[jwin].append(event_t[iev])

        if verbose:
            print("Tevs: ", tevs)
            print("amps: ", amps)
        # check for all edge cases
        if len(amps[0]) == 0:  # no events from the first stimulus, can't calculate a ratio
            all_ratios = [np.nan] * stim_N
        else:  # normal case - ratio of last to first
            all_ratios = [np.nanmean(amps[i]) / np.nanmean(amps[0]) for i in range(stim_N)]
        print(f"Ratio:", all_ratios)
        # also, if possible, compute paired pulse ratio between first two stimuli
        paired_pulse_ratio = all_ratios[1] if stim_N >= 2 else np.nan
        return {
            "ratio": all_ratios,
            "pp_ratio": paired_pulse_ratio,
            "amplitudes": amps,
            "eventtimes": tevs,
        }

    def IO(self, evfile, proto, stim_N=5):
        """
        Compute IO function for a single cell.
        """
        verbose = False
        proto = pathlib.PurePosixPath(evfile.parent, proto)  # make sure of type
        protocol, protodata = self._get_cell_protocol_data(evfile.parent)
        dpath = Path(
            self.expt["rawdatapath"],
            proto,
        )
        AR.setProtocol(dpath)
        pd = AR.getPhotodiode()
        blt = AR.getLaserBlueTimes()
        pd = np.mean(AR.Photodiode, axis=0)
        SR = 1.0 / AR.Photodiode_sample_rate[0]

        pdt = np.mean(AR.Photodiode_time_base, axis=0)
        t0 = [0] * AR.LaserBlueTimes["npulses"][0]
        t1 = [0] * AR.LaserBlueTimes["npulses"][0]
        pd_amps = np.zeros(AR.LaserBlueTimes["npulses"][0])
        for i, st in enumerate(AR.LaserBlueTimes["start"]):
            t0[i] = int(st / SR) + int(
                0.5 * AR.LaserBlueTimes["duration"][i] / SR
            )  # measure partway into the pulse
            t1[i] = int(st / SR) + int(AR.LaserBlueTimes["duration"][i] / SR)
            pd_amps[i] = np.mean(pd[t0[i] : t1[i]])
        print("pd_amps in Watts: ", pd_amps)
        AR.getScannerPositions()
        # recalibrate st_amps in terms of laser intensity in milliwatts/mm2
        # power is in watts
        # scanner spotsize is in meters and is diameter
        # compute spot area (in mm2)
        spotradius_mm = 0.5 * AR.scanner_spotsize * 1e3
        spotarea = np.pi * (spotradius_mm**2)
        # print(AR.spotsize, spotarea)
        attenuation_factor = 0.0428  # from the .cfg file, but NEED TO REMEASURE
        st_amps = attenuation_factor * pd_amps * 1e3 / spotarea
        print("spot amps in mW/mm2: ", st_amps)
        # exit()

        celltype = ephys.tools.map_cell_types.map_cell_type(
            self._get_cell_information(evfile.parent, "cell_type")[0]
        )
        # celltype = self._get_cell_information(evfile.parent, "cell_type")[0]
        if celltype is None:
            CP.cprint("r", f"Check cell type: {celltype:s}")
            raise ValueError()

        cellcolor = {
            "pyramidal": "red",
            "cartwheel": "orange",
            "tuberculoventral": "green",
            "bushy": "blue",
            "t-stellate": "cyan",
            "T-stellate": "cyan",
            "d-stellate": "yellow",
        }
        if celltype not in list(cellcolor.keys()):
            print("check celltype: ", celltype)
            # exit()
        envx = protodata[str(proto)]
        # print('stimtimes: ', envx['stimtimes'])
        if len(envx["stimtimes"]["starts"]) < stim_N:
            # print('not enough stim times', envx['stimtimes']['start'])
            # print('in ', str(proto))
            return (np.nan, np.nan, np.nan)
        # print('Enough stim times', self.events[proto]['stimtimes']['start'])
        # print('in ', str(p))
        self.protodata = protodata
        stimstarts = envx["stimtimes"]["starts"]
        stimwins = [[s + 0.001, s + 0.011] for s in stimstarts]
        protoevents = envx["events"]
        k = range(stim_N)
        amps = dict((ik, []) for ik in k)
        tevs = dict((ik, []) for ik in k)
        # st_amps = envx['stimtimes']['amplitude']  # raw voltage amplitude controlling level to laser
        if verbose:
            print("proto: ", str(proto))
            print(" # protoevents: ", len(protoevents))
        cprint("cyan", f"{str(evfile):s},    proto: {str(proto):s}")
        for trial in range(len(protoevents)):
            spots = list(np.where(envx["ZScore"][trial] > 0)[-1])  # all trials!
            if len(spots) == 0:  # no significant events
                continue
            for (
                ispot
            ) in (
                spots
            ):  # now for all spots in this trial, get the events for each stimulus so we can average them
                # if (
                #     "smpksindex" in protoevents[trial].keys()
                # ):  # handle both orders (old and new...)
                event_t = np.array(protoevents[trial].smpkindex[ispot]) * 5e-5  # time
                event_a = np.array(protoevents[trial].smoothed_peaks[ispot])  # amplitude
                # else:
                #     event_t = (
                #         np.array(protoevents[trial][ispot]["smpksindex"]) * 5e-5
                #     )[0]
                #     event_a = np.array(protoevents[trial][ispot]["smpks"])[0]
                for jwin in range(stim_N):
                    for iev in range(event_t.shape[0]):
                        if (event_t[iev] >= stimwins[jwin][0]) and (
                            event_t[iev] < stimwins[jwin][1]
                        ):
                            # print('jwins, iev, event_t, event_a, stimwins: ', jwin, iev, event_t[iev], event_a[iev], stimwins[jwin])
                            amps[jwin].append(event_a[iev])
                            tevs[jwin].append(event_t[iev])

        if verbose:
            print("tevs: ", tevs)
            print("amps: ", amps)
        ampl = [None] * len(amps)
        for j in amps.keys():
            ampl[j] = np.nanmean(amps[j])
        if np.nanmax(ampl) < 0.0:
            if celltype in list(cellcolor.keys()):
                vdmax = np.array(ampl) / np.min(ampl)
            #     mpl.plot(st_amps, vdmax, 'o-', color=cellcolor[celltype], markersize=4, label=celltype)
        else:
            print(f"IO: no data to plot? {celltype:s}, {str(proto):s}")
        return (st_amps, -1 * np.array(ampl) * 1e12, tevs, celltype)

    def filter_protocols(self, protocol: str, exclusions, doIO: bool = False) -> bool:
        if protocol is None:
            return False
        if (
            protocol.match("*_IC_*")
            or protocol.match("*_CA_*")
            or protocol.match("*_VC_3mMCa_*")
            or protocol.match("*_VC_Range_*")
            # or protocol.match('*_VC_WCChR2_*')  # same as "weird" and "single"
            or protocol.match("*objUPplus60*")
            or protocol.match("*5mspulses_5Hz*")
            or (protocol.match("*_VC_increase_*") and not doIO)  # IO protocols
            or (protocol.match("*range test*") and not doIO)
            or (protocol.match("*single_MAX_*"))
            or (protocol.match("*single_test_*"))
            # or (protocol.match("*_VC_1mW_*"))
            # or (protocol.match("*_VC_4mW_*"))
            # or (protocol.match("*_VC_pt1mW_*"))
            # or (protocol.match("*_VC_pt2mW_*"))
            # or (protocol.match("*_VC_0*"))
        ):
            if protocol is not None:
                cprint("yellow", "    * excluded on protocol: " + str(protocol))
                return False
        if str(protocol) in exclusions:
            cprint("yellow", f"    * File {str(protocol):s} in exclusions list")
            return False

        cprint("c", "    * included protocol: " + str(protocol))
        return True

    def summarize_one_protocol(
        self,
        protocol: str,
        cell_events,
        k: int,
        thiscell: str,
        whatplot: str,
        protocol_SR: float,
        nspots: int,
        cmds: object,
        doIO: bool,
    ):

        # cprint("m", "cellname, , protocol, whatplot: ")
        # print("     ", cellname, protocol, whatplot)

        if whatplot == "None" or whatplot is None:
            raise ValueError("whatplot is None")
        v = []

        match [whatplot]:
            case ["scores"]:
                v.append(cell_events["scores"])

            case ["SR"]:
                v.append(protocol_SR)

            case ["event_amps"]:
                # for m in range(len(cell_events["event_amps"][k])):
                v.append(cell_events["event_amps"][k])

            case ["event_qcontent"]:
                # for m in range(len(cell_events["event_qcontent"][k])):
                v.append(cell_events["event_qcontent"][k])

            case ["spont_amps"]:
                v.append(cell_events["spont_amps"][k])

            case ["amp_ratio"]:  # exception for calculation
                mean_spamp = np.mean(cell_events["spont_amps"][k])
                for m in range(len(cell_events["event_amps"][k])):
                    if mean_spamp != 0.0:
                        v.append(cell_events["event_amps"][k][m][0] / mean_spamp)
                    else:
                        v.append(np.nan)  # no spont; no ratio

            case ["depression_ratio"]:
                # need to access the data a bit deeper than from the event_summary file
                """
                For each trial:
                    For each trace that has 5 stimuli and for which the zscore is > 2.1 (e.g., has a response):
                        Get the detected event amplitudes for the 1st and last stimulus
                Then average the 1st and last amplitudes, and compute their ratio.
                Note that the "increase" protocol is already excluded above.
                """
                # We have to reach into the individual event file to do this:
                dr = self.depression_ratio(protocol, measuretype="depression_ratio")
                # print("dr: ", dr)
                if dr["ratio"] is not None:
                    # cprint('yellow', 'Depression ratio: ' + str(dr))
                    # v.append({"ratio": dr, "amps": amps, "tevs": tevs})
                    v.append([dr["ratio"]])

            case ["paired_pulse_ratio"]:
                # need to access the data a bit deeper than from the event_summary file
                """
                For each trial:
                    For each trace that has 5 stimuli and for which the zscore is > 2.1 (e.g., has a response):
                        Get the detected event amplitudes for the 1st and second stimulus
                Then average the 1st and second amplitudes, and compute their ratio.
                Note that the "increase" protocol is already excluded above.
                """
                # We have to reach into the individual event file to do this:
                dr = self.depression_ratio(protocol, measuretype="paired_pulse_ratio")
                # print("dr: ", dr)
                if dr["ratio"] is not None:
                    # cprint('yellow', 'PPR: ' + str(dr))
                    # v.append({"ratio": dr, "amps": amps, "tevs": tevs})
                    v.append([dr["ratio"]])

            case ["IO"]:
                if protocol.match("*_VC_increase_*") or protocol.match("*range test*"):
                    """
                    For each trial:
                        For each trace that has 5 stimuli:
                            Get the detected event amplitudes for each of the stimuli
                    Then average the amplitudes per stimulus
                    """
                    # We have to reach into the individual event file to do this:
                    iodata, amps, tevs, celltype = self.IO(p, protocol.name)
                    v.append(
                        {
                            "IOdata": iodata,
                            "amps": amps,
                            "tevs": tevs,
                            "celltype": celltype,
                        }
                    )

            case ["firstevent_latency"]:
                if cell_events["firstevent_latency"][k] is not None:
                    v.append(cell_events["firstevent_latency"][k])

            case ["tau1"]:
                # reader = MEDR.Reader(evl[dxf])
                # print(d[protocols[0]].  keys())
                tau1, tau2 = self.get_tau(thiscell)
                v.append([tau1])

            case ["tau2"]:
                tau1, tau2 = self.get_tau(thiscell)
                v.append([tau2])

            case ["area_fraction_Z"]:
                areaZ = self.getZ_fromprotocol(
                    filename=thiscell,
                    protocol=protocol,
                    param="area_fraction_Z",
                    area_z_threshold=cmds.area_z_threshold,
                    stimno=0,
                )
                v.append(
                    [areaZ]
                )  #  = float(len([cell_events['ZScore'] > area_z_threshold]))/float(len(cell_events['positions']))

            case ["median_Z_abovethreshold"]:
                # areaZ = self.getZ_fromprotocol(fn=x, db=cmds.database, protocol=p, param='median_Z_abovethreshold',area_z_threshold=cmds.area_z_threshold, stimno=0)
                areaZ = self.getZ_fromprotocol(
                    protocol=protocol,
                    filename=thiscell,
                    param="median_Z_abovethreshold",
                    area_z_threshold=cmds.area_z_threshold,
                    stimno=0,
                )
                v.append(
                    [areaZ]
                )  #  = float(len([cell_events['ZScore'] > area_z_threshold]))/float(len(cell_events['positions']))

            case ["maximum_Z"]:
                areaZ = self.getZ_fromprotocol(
                    filename=thiscell,
                    protocol=protocol,
                    param="maximum_Z",
                    area_z_threshold=cmds.area_z_threshold,
                    stimno=0,
                )
                v.append(
                    [areaZ]
                )  #  = float(len([cell_events['ZScore'] > area_z_threshold]))/float(len(cell_events['positions']))

            case ["mean_Z"]:
                areaZ = self.getZ_fromprotocol(
                    filename=thiscell,
                    protocol=protocol,
                    param=whatplot,
                    area_z_threshold=cmds.area_z_threshold,
                    stimno=0,
                )
                v.append(
                    [areaZ]
                )  #  = float(len([cell_events['ZScore'] > area_z_threshold]))/float(len(cell_events['positions']))

            case ["scores"]:
                if protocol_SR < 0.1 and len(cell_events["event_amps"][k]) < 0.05 * nspots:
                    v.append(
                        [0.0]
                    )  # too few events or too low spont to get an accurate measurement

                elif isinstance(cell_events[whatplot], float):
                    v.append([cell_events[whatplot]])
                else:

                    if len(cell_events[whatplot]) > k:
                        # cprint('cyan', f'  scores, Appended, {str(cell_events[whatplot]):s}, k: {k:d}')
                        v.append(cell_events[whatplot][k])
                    else:
                        pass
                        cprint("red", "  scores, NOT Appended")
            case _:
                cprint("red", f"    * Unknown whatplot: {whatplot:s}")
                pass
        return v

    def summarize_protocols(
        self, cell_events, thiscell=None, whatplot=None, min_spots=None, cmds=None, doIO=False
    ):
        """
        Summarize across the protocols in cell_events (event list of protocols)

        """

        values = []  # hold the values from each protocol for the selected measure
        protocols = []  # likewise, the protocol names that the values came from
        temperatures = []  # likewise, the temperature...
        cellIDs = []
        # print('protocols: ', cell_events['protocols'])
        # print(cell_events['positions'][0])
        temperature = self._get_cell_information(thiscell, "temperature")
        age = self._get_cell_information(thiscell, "age")
        for k, protocol in enumerate(cell_events["protocols"]):
            # exclude current clamp and cell-attached and other special
            print("checking protocol: ", protocol)
            if not self.filter_protocols(protocol, exclusions, doIO=doIO):
                continue
            print("ok on protocol: ", protocol)
            if (
                protocol.match("*_VC_10Hz*")
                or protocol.match("*_VC_increase_*")
                or protocol.match("*range test*")
            ):
                dur = 0.1
            elif protocol.match("*_VC_Single*") or protocol.match("*_VC_WCChR2*"):
                dur = 0.3
            elif protocol.match("*_VC_*"):  # old protocol name
                dur = 0.3
            else:
                raise ValueError("? protocol: ", p)
            # print('pos shape: ', np.array(cell_events['positions']).shape)
            nspots = len(cell_events["positions"][k])
            if nspots < min_spots and whatplot not in ["depression_ratio", "paried_pulse_ratio"]:
                cprint(
                    "magenta",
                    f"          Protocol {str(protocol):s} has too few spots ({nspots:d} < {min_spots:d} for {whatplot:s} analysis)",
                )
                continue

            protocol_SR = len(cell_events["spont_amps"][k]) / (dur * nspots)
            if whatplot == "SR":  # retrieve SR from event count
                sr_evs = len(cell_events["spont_amps"])

            cellname = protocol.parent
            v = self.summarize_one_protocol(
                protocol,
                cell_events,
                k=k,
                thiscell=cellname,
                whatplot=whatplot,
                protocol_SR=protocol_SR,
                nspots=nspots,
                cmds=cmds,
                doIO=doIO,
            )

            cellIDs.append(cellname)
            values.append(v)
            protocols.append(protocol)
            temperatures.append(temperature)

        res = {
            "measure": whatplot,
            "values": values,
            "cellID": cellIDs,
            "protocols": protocols,
            "temperature": temperatures,
        }
        # print("res: ", res)
        # print(whatplot, len(res["values"]), [np.mean(x) for x in res["values"] if len(x)  > 0])

        return res

    def summarize_cells(
        self,
        cmds=None,
        celltype=None,
        group=None,
        whatplot=None,
        min_spots=20,
    ):
        """
        Summarize results for all the cells of a given type in a given group
        Parameters
        ----------

        cmds : argparse output from command line call

        celltype : str
            Cell type to select for this run

        group : Experimental group (for example, 'A')

        min_spots : int (default: 50)
            minimum number of spots in a map to consider it valid for calculating scores

        """

        allcells = sorted(
            list(self.events.keys())
        )  # keys include date/slice/cell/protocol as pathlib Path
        # if whatplot is None or whatplot == "None":
        #     print("summarize cells, whatplot is None")
        #     raise ValueError("whatplot is None")
        if whatplot is None:
            whatplot = cmds.summarize
            print("whatplot was None, now is: ", whatplot)

        sum_data = []
        j = 0
        # if whatplot is None:
        #     whatplot = cmds.summarize
        ioflag = False
        if whatplot == "IO":
            ioflag = True
        # print('eventsummaryfile: ', eventsummaryfile)
        print("# of cells: ", len(allcells))
        for icell, thiscell in enumerate(allcells):  # for each cell
            if thiscell in exclusions:  # test based on day/slice/cell
                cprint("y", f"Cell is excluded: {str(thiscell):s}")
                continue
            if (
                thiscell.find("signflip") >= 0
                or thiscell.find("alt1") >= 0  # skip special, non-cannonical analysis files as well
                or thiscell.find("alt2") >= 0
            ):
                cprint("y", f"Cells is signflip alt1 or alt2 skipping")
                continue
            evn = self.events[thiscell]
            cellt = class_cell(evn["celltype"])
            if celltype.lower() != cellt.lower():  # select only celltype
                # cprint('y', f"    Cell type does not match({str(celltype.lower()):s} :: {str(cellt.lower()):s}), continuing")
                continue

            if self.coding is None or len(self.coding) == 0:
                # print("coding is None or len(coding) is 0")
                if group != "B":  # if input group is B, we will take all uncoded cells
                    # cprint("yellow", f"File {str(x):s} wrong code")
                    continue
            elif self.coding is not None and group is not None:  # select
                basename = thiscell.split("_")[0]
                coding = self.coding.dropna(subset="date")
                matchday = self.coding[self.coding["date"].str.endswith(basename)]
                # cprint("g", f"File {str(thiscell):s} matches cell {str(matchday['date']):s} in group {group:s}, ")
                # print(" with code: matchday.Group.values[0]: ", matchday["Group"].values[0], group)
                # exit()
                if matchday["Group"].values[0] != group:  # exists, and does not match group
                    # cprint("yellow", f"File {str(thiscell):s} in group {matchday['Group'].values[0]:s} does not match group {group:s}")
                    continue
            # permit restriction to just one cell when spot checking:

            # if cmds.day != "all":

            #     match = self.check_day_slice_cell(
            #         thiscell,
            #         day=DUP.parse(cmds.day),
            #         day_after=DUP.parse(cmds.after),
            #         day_before=DUP.parse(cmds.before),
            #         slice_cell=cmds.slicecell,
            #     )
            #     if not match:  # only do requrested daya/slices/cells
            #         CP.cprint("m", f"Cell did not match day/slice/cell criteria: {str(x):s}")
            #         continue
            #     sd = self.summarize_one_cell(thiscell=thiscell, cell_events=evn, whatplot=whatplot,
            #         min_spots=min_spots, doIO=ioflag, cmds=cmds)
            # else:
            print("Doing thiscell: ", thiscell)
            sd = self.summarize_one_cell(
                thiscell=thiscell,
                cell_events=evn,
                whatplot=whatplot,
                min_spots=min_spots,
                doIO=ioflag,
                cmds=cmds,
            )
            print(f"\nsum_data for: {thiscell:s}\n{str(sd):s}")
            sum_data.append(sd)
        return {"celltypes": celltype.lower(), "code": group, "data": sum_data}

    def summarize_one_cell(
        self,
        thiscell: str,
        cell_events: dict,
        whatplot: str,
        min_spots: int,
        doIO: bool = False,
        cmds: object = None,
    ):

        print("Summariing one cell: ", thiscell)

        sum_data = self.summarize_protocols(
            thiscell=thiscell,
            whatplot=whatplot,
            cell_events=cell_events,
            min_spots=min_spots,
            doIO=doIO,
            cmds=cmds,
        )
        return sum_data

        if cmds.listonly:
            # accumulate means by cell type
            all_values = []
            if whatplot in ["spont_amps", "event_amps"]:
                suffix = "A"
            else:
                suffix = ""
            ntotalevents = 0
            print(
                f"\n      {x:<44s}\t{whatplot:>12s}\t{celltype:>18s}\t{group:>4s}\t(cell# ={j+1:>3d})"
            )
            for z, p in enumerate(sum_data[-1]["prots"]):
                nprotevents = 0
                cdat = sum_data[-1]["values"][z]
                print(f"    {str(protocol.name):>40s}", end="")
                if isinstance(cdat, list):
                    if isinstance(cdat[0], dict):
                        v = np.mean(cdat[0]["ratio"])  # only depression ratio has this
                    elif isinstance(cdat[0], float):
                        v = cdat[0]
                        nprotevents += 1
                    elif isinstance(cdat[0], np.ndarray):
                        try:
                            v = np.mean(cdat[0])
                            print("..cdat[0]: ", cdat[0])
                            nprotevents += cdat[0].shape[0]
                        except:
                            print("cdat[0]: ", cdat[0])
                            v = np.mean(cdat[0][0])
                            nprotevents += cdat[0][0].shape[0]
                else:
                    v = cdat
                    nprotevents += len(cdat)
                if isinstance(v, list):
                    v = v[0]
                if isinstance(v, np.ndarray):
                    if v.shape[0] == 0:
                        v = 0.0
                    else:
                        v = np.mean(v)  # one more time...
                if whatplot not in ["depression_ratio", "paried_pulse_ratio"]:
                    print(
                        f"\t{eng_string(v, format='%9.3f', si=True, suffix=suffix):s}\t(N={nprotevents:d})"
                    )
                else:
                    print(f"\t{v:7.3f}", v)
                ntotalevents += nprotevents
            print(f"      Total events: {ntotalevents:d}")
            if len(sum_data["prots"]) == 0:
                print(f"       <no analyzable data>")
            j += 1
            # for k in list(csum.keys()):
            #     if k in 'giant':
            #         print(f'{k:<32s} ({len(csum[k]):3d})  {str(csum[k]):s}')
            return sum_data

    def extract_data(self, celltype, sum_data, whatplot):
        """
        extract data for one group (treatment) and celltype"""

        evthr = np.log10(20.0)
        firststim = True
        increase_last = True
        # x_celltype_max = 0.
        histdata = {"cellID": None, "data": None, "bins": {}}
        ncells = len(sum_data["data"])
        ctitle = ""
        dataok = True
        print(f"Extracting {whatplot:s} for {celltype:s} cells (N = {ncells:d})")
        if whatplot in ["event_amps", "spont_amps"]:
            scale = 1e12
            sign = -1.0
        elif whatplot in ["firstevent_latency", "allevent_latency"]:
            scale = 1e3
            sign = 1.0
        else:
            scale = 1.0
            sign = 1.0

        # print('what: ', whatplot)
        avgbycell = np.zeros(ncells)
        cellid = [" "] * ncells
        # print('ncells: ', ncells, len(sum_data['data']))
        for nc in range(ncells):  # for all cells in the group
            cell_maps = sum_data["data"][nc]  # get data for all of the maps for this cell
            if cell_maps is None:
                continue
            print("cell_maps: ", cell_maps)
            nmaps = len(cell_maps["values"])
            prots = cell_maps["protocols"]  #  # all the protocols
            cprint("yellow", f"prots: {nc:d}, {str(prots):s}, {nmaps:d} maps")
            # continue
            avgbymap = np.zeros(nmaps)  # to hold results
            depr_ratio = np.zeros(nmaps)
            for nm in range(nmaps):
                if len(cell_maps["values"][nm]) == 0:
                    continue
                d = np.array(cell_maps["values"][nm])[0]
                # print(f'map : {nm:d} of {nmaps:d}')
                try:
                    sh = d.shape
                except:
                    # print(' no data in # ', nc)
                    continue

                # print(cell_maps['values'])
                if whatplot in ["firstevent_latency", "allevent_latency"]:
                    d = d[0]
                # print('d shape: ', d.shape)
                if isinstance(d, float):  # convert single float value to an array with the value
                    # cprint('red', f'FLOAT: {d:f}')
                    avgbymap[nm] = d
                    continue
                sh = d.shape
                if d.ndim == 2 and (
                    sh[0] == 1 and sh[1] > 1
                ):  # need to look deeper. Does this array contain an array?
                    avgbymap[nm] = np.nanmean(d[0])
                    # cprint('red', 'Single value '+str(map_meanvalue))
                    continue
                else:  # build array
                    d = cell_maps["values"][nm]
                    # print('d: ', d)
                    #                   print(np.array(d[0]).shape)
                    #                   print(np.array(d[0]))
                    if np.array(d[0]).shape == () or len(d[0]) == 0:  # np.array(d[0]) == [None]:
                        # cprint('red', '  Empty array')
                        continue
                    try:
                        avgbymap[nm] = np.nanmean([np.nanmean(x) for x in np.array(d[0])])
                    except:
                        print("EXTRACT DATA FAILED: \n", np.array(d[0]))
                        exit()

            # print('nprots: ', len(prots))
            # for nm, p in enumerate(prots):
            #     cprint('cyan', avgbymap)
            #     cprint('yellow', np.nanmean(avgbymap))
            #     if np.isnan(avgbymap[nm]):
            #         cprint('red', f"    {str(p):s}   {avgbymap[nm]:f}")

            if len(prots) > 0:
                avgbycell[nc] = np.nanmean(avgbymap)
            else:
                avgbycell[nc] = np.nan
                cprint("red", f"No data for cell/maps: {str(cell_maps['cellID']):s}")

            if pd.isnull(avgbycell[nc]):
                avgbycell[nc] = 0.0
            if isinstance(avgbycell[nc], float):
                avgbycell[nc] = avgbycell[nc]
            cellid[nc] = sum_data["data"][nc]["cellID"]
            # avgbycell[nc]  = avgbycell[nc][~np.isnan(avgbycell[nc])]
        # ctitle = f'{group:s} {celltype:s} (N={ncells:3d})'
        if len(avgbycell) == 0:  # no cells!
            return None

        if whatplot == "largest_event_qcontent":
            if celltype in ["bushy", "Bushy"]:
                bw = 5.0
                xmax = 200.0
                binstuff = np.arange(0, xmax, bw)
            else:
                bw = 1.0
                xmax = 20.0
                binstuff = np.arange(0, xmax, bw)
            histdata["data"] = avgbycell
            if len(histdata["bins"]) == 0:
                histdata["bins"] = {"xmax": xmax, "bins": binstuff, "binw": bw}
            histdata["cellID"] = cellid

        elif whatplot in ["scores", "shufflescore"]:
            bw = 0.1
            xmax = 4.0
            binstuff = np.arange(0.05, xmax, bw)  #  + ng*bw/3.
            # ns = len(np.where(np.log10(avgbycell) > evthr)[0])
            cellid = [cid for i, cid in enumerate(cellid) if avgbycell[i] > 0]
            histdata["cellID"] = cellid
            avgbycell = avgbycell[avgbycell > 0]
            log_avgbycell = np.log10(np.clip(avgbycell, 0.1, 10000.0))
            ns = len(np.where(np.log10(avgbycell) > evthr)[0])
            histdata["data"] = log_avgbycell  # np.log10(np.clip(avgbycell, 0.01, 1e5))
            # print('hist?;')
            if len(histdata["bins"]) == 0:
                histdata["bins"] = {"xmax": xmax, "bins": binstuff, "binw": bw}
            nt = len(avgbycell)
            if nt == 0:
                nt = 1

            # tx = f"{group:s} {celltype:s} (N={ncells:3d}) {100*float(ns)/nt:.1f} thr={evthr:.1f}"
            # cttitle = tx
            # print('histdata: ', histdata)

        else:
            mx = np.max(scale * sign * avgbycell)
            x_max = PH.nextup(mx, steps=[1, 2, 4, 5, 8])
            if x_max > 0:
                nb = 20
                bw = x_max / nb
                binstuff = np.arange(0.0, x_max, x_max / nb)  # +ng*(x_max/nb)/3.
            else:
                bw = 0.1
                binstuff = np.arange(0.0, 1, bw)  # +ng*bw/3.
            histdata["data"] = sign * scale * avgbycell

            if len(histdata["bins"]) == 0:  # update to max
                histdata["bins"] = {"xmax": x_max, "bins": binstuff, "binw": bw}
            histdata["cellID"] = cellid
        # print(len(histdata['data']), len(histdata['cellID']))
        return histdata

    def get_data(
        self,
        cmds=None,
        filename=None,
        celltypes=["bushy"],
        parameter="RMP",
        coding=None,
        groups_to_do=None,
    ):
        cttitle = {}
        histdata_sum = {}
        whatplot = None
        self.pddata = pd.DataFrame(
            columns=["celltype", "parameter", "coding", "group", "summary_data"]
        )
        if parameter != cmds.summarize:
            whatplot = cmds.summarize
        # print("get_data: celltypes: ", celltypes)
        # print("whatplot: ", whatplot)
        if celltypes == "all":
            celltypes = all_celltypes

        for n, celltype in enumerate(celltypes):  # for each cell type, which is on a separate plot
            groups_found = []  # for this cell type
            histdata = {}
            cttitle[celltype] = {}
            x_celltype_max = 0
            for ng, group in enumerate(groups_to_do):  # for all groups in this cell type
                print(
                    f"\nget_data:  Group: {group:s}, Cell type: {celltype:<15s} parameter: {parameter:s}"
                )
                # get the data for the cells in this group
                sum_data = self.summarize_cells(
                    cmds=cmds,
                    # filename=filename,
                    celltype=celltype,
                    group=group,
                    whatplot=whatplot,
                )
                # if len(sum_data["data"]) > 0:
                #     print(f" summary_data: ", sum_data)
                #     exit()

                eqs = "=" * 80
                cprint("c", f"\n{eqs:s}")
                print("    celltype: ", celltype)
                print("    parameter: ", parameter)
                print("    whatplot:  ", whatplot)
                print("    coding: ", sum_data["code"])
                print("    group: ", group)
                print("    summary_data: ", sum_data["data"])

                # print(self.pddata)

                # print("summary_data: ", sum_data)
                # exit()

                self.pddata = pd.concat(
                    [
                        self.pddata,
                        pd.DataFrame(
                            {
                                "celltype": celltype,
                                "parameter": parameter,
                                "coding": sum_data["code"],
                                "group": group,
                                "summary_data": sum_data,
                            }
                        ),
                    ],
                    ignore_index=True,
                )
                cprint("c", f"<{eqs:s}>\n")
                if sum_data is None or len(sum_data["data"]) == 0:
                    cprint(
                        "r",
                        f"summarize_cells: No data from dict for: code={sum_data['code']:s}, celltype={celltype:s}, group={group:s}",
                    )
                    continue
                groups_found.append(group)
                # unpack
                histdata[group] = self.extract_data(celltype, sum_data, parameter)
                if histdata[group] is None:
                    cprint("r", f"get_data: Hist data for group is None")
                    continue
                cttitle[celltype][group] = ""
            histdata_sum[celltype] = histdata

        fout = Path(
            self.expt["analyzeddatapath"],
            self.expt["directory"],
            f"{parameter:s}_summary.pkl",
        )
        print("get_data:, writing to Output file: ", str(fout))
        self.pddata.to_pickle(fout, compression=None)
        return histdata_sum, cttitle

    def get_all_data(self, cmds=None, celltypes=["bushy"], groups_to_do="B"):
        """
        A better way to do this is with pandas...
        """
        # sanitize celltypes (should be list)
        if isinstance(celltypes, str):
            celltypes = [celltypes]
        print("get all data")
        parameters = [
            "SR",
            "scores",
            "shufflescore",
            "minprobability",
            "meanprobability",
            "event_amps",
            "spont_amps",
            "amp_ratio",
            "depression_ratio",
            "paired_pulse_ratio",
            "event_qcontent",
            "largest_event_qcontent",
            "area_fraction_Z",
            "median_Z_abovethreshold",
            "maximum_Z",
            "mean_Z",
            "None",
            "firstevent_latency",
            "allevent_latency",
            "tau1",
            "tau2",
        ]
        # print(self.events.keys())
        pdevo = pd.DataFrame.from_dict(self.events, orient="index")

        # print(pdev.head())
        # print(pdev.columns.values)
        # print(pdev.index)
        pdev = pdevo.copy()
        pdev.insert(2, "temp", 3.0)
        pdev.insert(3, "age", 0)
        pdev.insert(4, "internal", "")
        drops = []
        for j, c in enumerate(pdev.index):
            if c.find("alt1") > 0 or c.find("signflip") > 0 or c.find("alt2") > 0:
                drops.append(c)
                continue
            pdev.loc[c, "temp"] = int(self._get_cell_information(c, "temperature"))

        # print(pdev.head(10))
        # print(pdev.columns)
        # print(pdev.celltype)
        all_groups = OrderedDict()
        if celltypes is None:
            celltypes = ["pyramidal", "cartwheel", "tuberculoventral"]
        for n, celltype in enumerate(celltypes):  # for each cell type, which is on a separate plot
            groups_found = []  # for this cell type
            x_celltype_max = 0
            # print("pdev.celltype: ", [p for p in pdev.celltype.values])
            # print("celltype: ", celltype)
            ct_data = pdev[
                pdev["celltype"] == celltype
            ].reset_index()  # data for this cell type only

            for ng, group in enumerate(groups_to_do):  # for all specified groups
                # print(f"\n{celltype:<15s} Group: {group:s}")
                # get the data for the cells in this group
                ndata = len(ct_data.index)
                bigdata = np.recarray(
                    (ndata),
                    dtype=[
                        ("cell", "S50"),
                        ("age", int),
                        ("temperature", int),
                        ("internal", "S10"),
                        ("scores", float),
                        ("minprobability", float),
                        ("meanprobability", float),
                        ("shufflescore", float),
                        ("event_amps", float),
                        ("spont_amps", float),
                        ("depression_ratio", float),
                        ("paired_pulse_ratio", float),
                        ("event_qcontent", float),
                        ("firstevent_latency", float),
                        ("SR", float),
                        ("tau1", float),
                        ("tau2", float),
                    ],
                )
                bd = {}
                for j, cell in enumerate(ct_data.index):
                    # print("cell data: ", cell, ct_data.iloc[cell])
                    cell_id = str(ct_data.iloc[cell]["index"])
                    # print(ct_data.iloc[cell]["event_amps"])

                    for whatplot in bigdata.dtype.names:
                        ioflag = False
                        if whatplot == "IO":
                            ioflag = True
                        if whatplot in ["cell", "age", "temperature", "internal"]:
                            temp = self._get_cell_information(cell, "temperature")
                            elec_sol = self._get_cell_information(cell, "internal")
                            age = self._get_cell_information(cell, "age")
                            if temp is None or temp == "None":
                                temp = 34
                            if elec_sol is None or elec_sol == "None":
                                elec_sol = "KGluc"
                            bigdata[j].cell = cell_id
                            print(f"age: <{age!s}>")
                            print("elec_sol: ", elec_sol)
                            print("temp: ", temp)
                            if age is None or age == "None":
                                age = 56  # placeholder
                            bigdata[j].age = age
                            bigdata[j].temperature = int(temp)
                            bigdata[j].internal = elec_sol
                        else:
                            sum_data = self.summarize_one_cell(
                                thiscell=cell_id,
                                cell_events=self.events[cell_id],
                                cmds=cmds,
                                whatplot=whatplot,
                                doIO=ioflag,
                                min_spots=5,
                            )
                            # print("sumdata: ", sum_data)
                            if sum_data is None or len(sum_data) == 0:
                                continue
                            # print(sum_data.keys())
                            # print(group, celltype, sum_data['data'])
                            # exit()
                            groups_found.append(group)
                            # print("whatplot: ", whatplot)
                            # print("bigdata: ", bigdata)
                            bd[whatplot] = sum_data["values"]
                            if len(sum_data["values"]) == 0:
                                print("No data from dict for ", celltype, group)
                                continue
                all_groups[(celltype, group)] = bd

        return all_groups

    def summarize(
        self,
        cmds=None,
        celltypes="all",
        filename=None,
        groups_to_do=["A", "AA", "B", "AAA"],
    ):
        assert self.coding is not None
        if cmds.summarize in ["tau1", "tau2"]:
            taudb = Path(
                self.expt["analyzeddatapath"],
                self.expt["directory"],
                "merged_db",
            ).with_suffix(".pkl")

            dbtaus = pd.read_pickle(taudb)
        print("starting to Summarize")
        rain = False
        histflag = True
        whatplot = cmds.summarize

        area_z_threshold = cmds.area_z_threshold

        cprint("c", f"Celltype selection: {str(celltypes):s}")

        if celltypes == "all":
            axis_no = dict(zip(all_celltypes, range(len(all_celltypes))))
            ncelltypes = len(all_celltypes)
        elif cmds.celltypes == "VCN":
            axis_no = dict(zip(VCN_celltypes, range(len(VCN_celltypes))))
            ncelltypes = len(VCN_celltypes)
        elif cmds.celltypes == "DCN":
            axis_no = dict(zip(DCN_celltypes, range(len(DCN_celltypes))))
            ncelltypes = len(DCN_celltypes)
        else:
            axis_no = {celltypes: 0}
            ncelltypes = 1

        histdata_sum, cttitle = self.get_data(
            cmds=cmds,
            filename=filename,
            celltypes=celltypes,
            parameter=whatplot,
            coding=self.coding,
            groups_to_do=groups_to_do,
        )

        refdata_sum, ref_cttitle = self.get_data(
            cmds=cmds,
            filename=filename,
            celltypes=celltypes,
            parameter="scores",
            coding=self.coding,
            groups_to_do=groups_to_do,
        )

        if not cmds.listonly and histflag:
            print("buildplot")
            rc = PH.getLayoutDimensions(ncelltypes, pref="width")
            # P = PH.regular_grid(rc[0], rc[1], figsize=(10., 8), position=-0.05,
            #             margins={'leftmargin': 0.07, 'rightmargin': 0.05, 'topmargin': 0.1, 'bottommargin': 0.1})
            P = PH.regular_grid(
                1,
                1,
                figsize=(8.0, 8),
                position=0.00,
                margins={
                    "leftmargin": 0.1,
                    "rightmargin": 0.1,
                    "topmargin": 0.1,
                    "bottommargin": 0.1,
                },
            )
            axl = P.axarr.ravel().flatten()
            esc_underscore = "\_"
            toptitle = f"{whatplot:s}"

            if whatplot == "area_fraction_Z":
                toptitle += f":  Z threshold={area_z_threshold:.2f}"
                print("set toptitle")
            P.figure_handle.suptitle(toptitle, fontsize=12)

            if len(groups_to_do) in [3, 4]:
                gcolors = OrderedDict([("A", "b"), ("AA", "r"), ("B", "g"), ("AAA", "m")])
                gxcolors = [[0, 0, 1, 1], [1, 0, 0, 1], [0, 1, 0, 1], [1, 0, 1, 1]]
            else:
                gcolors = OrderedDict([("B", "g")])
                gxcolors = [[0, 1, 0, 1]]
                ngroups = len(groups_to_do)

            groups = []
            values = []
            cellid = []
            print("axisno keys: ", axis_no.keys())
            for nax, ct in enumerate(axis_no.keys()):  # cell types in each plot
                print("CT: ", ct)
                nax = 0
                resp = 0
                noresp = 0
                hddf = pd.DataFrame()
                # print("histdata keys: ", histdata_sum.keys())
                print("groups to do: ", groups_to_do)
                for group in groups_to_do:
                    print("histdata: ", histdata_sum[ct])
                    if group in list(histdata_sum[ct].keys()):
                        hd = histdata_sum[ct][group]
                        refd = refdata_sum[ct][group]
                    else:
                        hd = []
                        refd = []
                    print("ct: ", ct, "group: ", group, "hd: ", hd)
                    # exit()
                    if len(hd) == 0:
                        print("empty hd")
                        continue
                    gc = gxcolors[: len(hd)]
                    # nn, bins, patches = axl[nax].hist(hd['data'], bins=hd['bins']['bins'], # rwidth= 1./ngroups),
                    #     color=gc, align='mid', alpha=0.8)
                    # print(hd)
                    if "xmax" in hd["bins"].keys():
                        axl[nax].set_xlim(0, hd["bins"]["xmax"])
                    # print(hd.keys())
                    #                  exit()
                    groups.extend([ct] * len(hd["data"]))
                    cellid.extend(hd["cellID"])
                    # print("\ncell type: ", ct, "cellID: ", hd["cellID"])
                    what_values = []
                    # build numpy rec array from data we are getting, rather than just printing it out
                    for j in range(len(hd["cellID"])):
                        print("cell id: ", hd["cellID"][j])
                        temp = self._get_cell_information(hd["cellID"][j], "temperature")
                        elec_sol = self._get_cell_information(hd["cellID"][j], "internal")
                        age = self._get_cell_information(hd["cellID"][j], "age")
                        if age is None:  # means missing information for cell ID in database....
                            continue

                        print(f"CELL ID:  {hd['cellID'][j]!r}\t", end="")
                        # print(f"{hd['data'][j]:.3f}\t", end="")
                        print(f"AGE:  {age:3d}\t{temp:2d}\t{elec_sol:s}")
                        if whatplot == "scores" and hd["data"][j] >= 1.2:
                            resp += 1
                        elif (
                            whatplot in ["maximum_Z", "mean_Z", "median_Z_abovethreshold"]
                            and hd["data"][j] > 2.1
                        ):
                            resp += 1
                        elif whatplot == "shufflescores":
                            print("Shufflescore data: ", hd["data"][j])
                            resp += 1

                        else:
                            noresp += 1
                        # print(len(hd['data']), len(refd['data']))
                        if whatplot in ["firstevent_latency", "allevent_latency"]:
                            # cid = hd['cellID'][j]
                            try:
                                cid = refd["cellID"].index(hd["cellID"][j])
                                # print('cid: ', cid, refd['data'][cid])
                                if (
                                    hd["data"][j] > 1e-5
                                    and (refd["cellID"][cid] == hd["cellID"][j])
                                    and refd["data"][cid] > 1.5
                                ):  # exclude zeros and low-p responders
                                    what_values.append(hd["data"][j])
                            except ValueError:
                                pass

                    if len(hd["data"]) > 0:
                        values.extend(hd["data"])
                    else:
                        pass
                        # values.extend([np.nan])
                    # print(what_values)
                    if whatplot == "scores":
                        print(
                            f"{' '*40:s}  Resp: {resp:3d}  Noresp: {noresp:3d} (N={resp+noresp:3d})"
                        )
                    if whatplot in ["firstevent_latency", "allevent_latency"]:
                        print(
                            f"{whatplot:s} Lat: {np.mean(what_values):.3f} (SD={np.std(what_values):.3f} N={len(what_values):d})"
                        )
            hddf = pd.DataFrame(
                {
                    "Group": pd.Categorical(groups),
                    whatplot: np.array(values),
                    "cellID": cellid,
                }
            )
            nax = 0
            if len(hddf) == 0:
                return
            if rain:
                #     pt.RainCloud(
                #         x="Group",
                #         y=whatplot,
                #         data=hddf,
                #         ax=axl[nax],
                #         width_viol=1.5,
                #         width_box=0.35,
                #         orient="h",
                #     )
                #     axlim = axl[nax].set_xlim
                # pt.half_violinplot(x='Group', y=whatplot, data=hddf, ax=axl[nax], bw='scott',  linewidth=1, cut=0.,
                #     width=0.8, inner=None, orient='h')
                sns.boxplot(
                    x="Group",
                    y=whatplot,
                    data=hddf,
                    ax=axl[nax],
                    color="0.8",
                    showcaps=True,
                    boxprops={"facecolor": "none", "zorder": 10},
                    showfliers=True,
                    whiskerprops={"linewidth": 2, "zorder": 10},
                )
                sns.stripplot(
                    x="Group",
                    y=whatplot,
                    data=hddf,
                    ax=axl[nax],
                    size=2.5,
                    jitter=1,
                    color="0.8",
                    orient="h",
                )

            else:
                print(hddf)
                if whatplot in ["depression_ratio", "paired_pulse_ratio"]:
                    hddf = hddf.replace(0.0, value=np.nan)
                sns.swarmplot(x="Group", y=whatplot, data=hddf, ax=axl[nax])
                # sns.violinplot(x='Group', y=whatplot, data=hddf, ax=axl[nax], color='0.8')
                sns.boxplot(x="Group", y=whatplot, data=hddf, ax=axl[nax], color="0.8")
                axlim = axl[nax].set_ylim

            xlim = axl[nax].get_xlim()
            if whatplot == "spont_amps":
                axlim(0, 150.0)  # axl[nax].plot(xlim, [1.3, 1.3], '--', color='gray')
            if whatplot == "evoked_amp":
                axlim(0, 3000.0)  # axl[nax].plot(xlim, [1.3, 1.3], '--', color='gray')
            if whatplot == "scores":
                axlim(0, 4.1)  # axl[nax].plot(xlim, [1.3, 1.3], '--', color='gray')
            if whatplot == "median_Z_abovethreshold":
                axlim(0, 33.0)  # axl[nax].plot(xlim, [1.3, 1.3], '--', color='gray')
            if whatplot == "maximum_Z":
                axlim(0, 20.0)  # axl[nax].plot(xlim, [1.3, 1.3], '--', color='gray')
            if whatplot in ["depression_ratio", "paired_pulse_ratio"]:
                axlim(0, 2.0)  # axl[nax].plot(xlim, [1.3, 1.3], '--', color='gray')
            if whatplot in ["firstevent_latency", "allevent_latency"]:
                axlim(0, 10)
            # print(axl[nax].get_ylim())# axl[nax].set_ylim(0, 5)

            mpl.show()

            ot = f"Obs,Group"
            ot += f",{whatplot:s}"
            ot += "\n"
            for i in range(len(hddf[whatplot])):
                ot += f"{i:d},{hddf['Group'][i]:s}"
                ot += f",{hddf[whatplot][i]:f}"
                ot += "\n"
            # print(ot)
            # ofile = Path(f"R_{str(eventsummaryfile.name):s}_{whatplot:s}_{celltype:s}_{'OU':s}.csv")
            # ofile.write_text(ot)

        if not cmds.listonly:
            for n, celltype in enumerate(axis_no.keys()):
                nax = axis_no[celltype]
                tx = ""
                for ng, group in enumerate(["A", "AA", "B", "AAA"]):
                    if group in cttitle[celltype].keys():
                        tx += cttitle[celltype][group] + "\n"
                axl[0].set_title(tx, fontsize=9, verticalalignment="top")

            mpl.show()

    def check_day_slice_cell(self, fn, day=None, day_after=None, day_before=None, slice_cell=None):
        """
        Check the filename to see if it falls in the day range or specific day,
        and optionally filter by the slice and cell

        Parameters
        ----------
        fn : Path or str
            Filename to check. Should be in format of year.mo.dd_00n/slice_nnn/cell_nnn
            Could also have '~' instead of '/'

        day : str
            Day pattern to match (year.mo.dd)

        day_after : str
            Return true only if day is on or after this date

        day_before : str
            Return true only if day is on or before this date

        slice_cell : str
            String of format sncm for slice#n, cell#m

        Returns
        -------
        boolean : True if fn matches criteria, False otherwise

        Note:

        """

        if day == None and day_after == None and day_before == None and slice_cell == None:
            return True  # no selection, fn is always a match
        fnp = Path(fn)  # convert if not already
        if str(fnp).find("~"):
            fnk = Path(str(fnp).replace("~", "/"))  # make original key that matches.
        else:
            fnk = fnp
        fnparts = fnk.parts
        dsday = fnparts[0].split("_")[0]
        # print(fnparts)
        # print('dsday', dsday)
        thisday = datetime.datetime.strptime(dsday, "%Y.%m.%d")
        # print('thisday: ', thisday, dsday, day)
        slicestr = fnparts[1]
        cellstr = fnparts[2][:8]
        if day_after is not None:
            if thisday < day_after:
                return False
        if day_before is not None:
            # print("thisday, daybefore: ", thisday, day_before)
            if thisday > day_before:
                return False
        # print(thisday == day)
        if (day is not None) and (day != "all") and (thisday != day):
            return False
        if slice_cell is not None and len(slice_cell) > 0:
            slicen = "slice_%03d" % int(slice_cell[1])  # should do with regex
            if slicestr != slicen:
                return False
            if len(slice_cell) == 4:
                celln = "cell_%03d" % int(slice_cell[3])
                # if thisday == day:
                #     print('sc, cellstr, slicen, celln: ', slice_cell, cellstr, slicen, celln)
                if cellstr != celln:
                    return False
        return True

    def get_tau(self, cellname):
        if self.taudata is None:
            self.read_tau_file()
        if self.taudata is None:  # still - no tau data
            return np.nan, np.nan
        checktau = self.taudata.dropna(subset=["cell"])
        cellname = str(cellname).replace("/", "~")
        davg = checktau.loc[checktau["cell"].str.contains(cellname)]
        tau1_avg = np.nanmean(davg["tau1"]) * 1e3
        tau2_avg = np.nanmean(davg["tau2"]) * 1e3
        return tau1_avg, tau2_avg

    def read_tau_file(self):

        taudb = Path(
            self.expt["analyzeddatapath"],
            self.expt["directory"],
            "NF107Ai32_Het_taus",
        ).with_suffix(".pkl")
        try:
            with open(taudb, "rb") as fh:
                self.taudata = pd.read_pickle(fh, compression=None)
        except:
            self.taudata = None

    def _do_annotation(self, eventsummary_file, annotation_file: Union[Path, None] = None):
        """
        Given the event summary file, and the annotation file, we update
        the cell type  in the event summary file from the annotation file.
        This may or may not be a good idea...

        Parameters
        eventsummary_file : str or path of filename
        annotation_file : str or path of filename
        """
        with open(eventsummary_file, "rb") as fh:  # write at each pass
            try:
                self.events = pickle.load(fh)  # get the current file
                self.eventsummary_file = eventsummary_file
            except:
                cprint("r", f"eventsummaryfile is corrupted: {str(fh):s} ")
                Logger.error(f"Event Summary File is corrupted: {str(fh):s} ")
                self.events = None
                return self.events
        cprint("g", f"do_annotation: retrieved event summary file: {str(eventsummary_file):s}")
        cprint("magenta", f"Reading annotation file: {str(annotation_file):s}")
        if annotation_file is None:
            return self.events

        with open(annotation_file, "rb") as fh:
            self.annotated = pd.read_excel(fh)
        # print(self.annotated.head())
        # self.annotated.set_index("ann_index", inplace=True)
        x = self.annotated[self.annotated.index.duplicated()]
        if len(x) > 0:
            print("watch it - duplicated index in annotated file")
            print(x)
            exit()
        # self.annotated.set_index("ann_index", inplace=True)
        self.db.loc[self.db.index.isin(self.annotated.index), "cell_type"] = self.annotated.loc[
            :, "cell_type"
        ]
        self.db.loc[self.db.index.isin(self.annotated.index), "annotated"] = True
        if self.verbose:  # check whether it actually took
            for icell in range(len(self.df.index)):
                print(
                    "{0:<32s}  type: {1:>20s}, annotated (T,F): {2!s:>5} Index: {3:d}".format(
                        str(self.make_cellstr(self.db, icell)),
                        self.db.iloc[icell]["cell_type"],
                        str(self.db.iloc[icell]["annotated"]),
                        icell,
                    )
                )
        for efd in list(self.events.keys()):  # for all of the data in the event files
            dblink = Path(efd).parts
            # print('dblink: ', dblink)
            day_x = self.db.loc[
                (self.db["date"] == dblink[0])
                & (self.db["slice_slice"] == dblink[1])
                & (self.db["cell_cell"] == dblink[2])
            ]
            if len(day_x["cell_type"].values) == 0:
                continue
            if str(self.events[efd]["celltype"]).strip() != str(day_x["cell_type"]).strip():
                # print('efd: ', efd, 'efd celltype: ', self.events[efd]['celltype'], end='')
                # print( ' dayx: ', day_x['cell_type'])
                self.events[efd]["celltype"] = day_x["cell_type"].values[0]

        return self.events

    def get_eventsummary_filename(self, database):
        eventsummary_file = Path(
            self.expt["analyzeddatapath"],
            self.expt["directory"],
            self.expt["directory"] + "_event_summary.pkl",
        )
        return eventsummary_file

    def get_databases(self, database):
        """
        given the experiment designation,
        Open the main databases (e.g., NF107Ai32_Het, with all the data for the file)
        Optionally open the annotation file, and update the annotations,
        and get the coding information

        This expects that the event summary file has been generated.
         The first pass of analysis generates individual
            event summary files for each cell, and these are stored in directories by cell type (name).
            The eventsummary file is generated from those files using this program, with the
            --eventsummary  flag and --force to update the entries in the eventsummary file

        Parameters
        ----------
        database: str : name of the database (e.g., NF107Ai32_Het, NF107Ai32_NIHL, etc)

        Returns
        -------
        Nothing

        The pandas database is saved in self.db
        The coding information is saved in self.coding
        """
        self.database = database
        database_file = Path(
            self.expt["analyzeddatapath"],
            self.expt["directory"],
            self.expt["datasummaryFilename"],
        )
        with open(database_file, "rb") as fh:
            self.db = pd.read_pickle(fh)
        self.get_coding(database)

        eventsummary_file = self.get_eventsummary_filename(database)

        if eventsummary_file.is_file():
            annotation_file = Path(
                self.expt["analyzeddatapath"],
                self.expt["directory"],
                self.expt["map_annotationFilename"],
            )
            if annotation_file.is_file():
                self.events = self._do_annotation(eventsummary_file, annotation_file)
            else:
                try:  # if file is corrrupted, this will fail
                    with open(eventsummary_file, "rb") as fh:
                        self.events = pickle.load(fh)  # get the current file
                    self.eventsummary_file = eventsummary_file
                except:
                    cprint(
                        "r",
                        f"get_database: eventsummaryfile is corrupted: {str(eventsummary_file):s}",
                    )
                    Logger.error(
                        f"get_database: Event Summary File is corrupted: {str(eventsummary_file):s} "
                    )
                    self.events = None
                    exit()
            cprint("g", f"get_database: retrieved event summary file: {str(eventsummary_file):s}")

    def get_coding(self, database: str):
        self.coding = None
        if self.expt["coding_file"] is None:
            return None
        coding_file = Path(
            self.expt["analyzeddatapath"],
            self.expt["directory"],
            self.expt["coding_file"],
        )
        with open(coding_file, "rb") as fh:
            coding = pd.read_excel(fh, index_col=0, header=0)
            self.coding = coding
            self.coding_file = coding_file
        return coding

    def listdatabase(self):
        if self.events is None or self.coding is None:
            raise ValueError("get the databases first before trying to list them")
        all_entries = []
        for celltype in all_celltypes:
            of_this_celltype = []
            for cell_ID in list(self.events.keys()):  # for all of the data in the event files
                cc = class_cell(self.events[cell_ID]["celltype"])
                if cc == celltype:
                    if str(cell_ID).find("_alt") > 0 or str(cell_ID).find("_signflip") > 0:
                        continue
                    of_this_celltype.append(str(cell_ID))
                    all_entries.append(str(cell_ID))
            print(f"\nCelltype: {celltype:s}  (N={len(of_this_celltype):d})")
            for cell_id in sorted(of_this_celltype):
                temp = self._get_cell_information(cell_id, "temperature")
                age = self._get_cell_information(cell_id, "age")
                elec_sol = self._get_cell_information(cell_id, "internal")
                group = self._get_cell_information(cell_id, "Group")
                SPL = self._get_cell_information(cell_id, "SPL")
                print(
                    f"{str(cell_id):42s} {self.events[cell_id]['celltype']:>14s} {age:3d}D {temp:2d}C {elec_sol:>8s} {group:>5s} {SPL:>4s}dB"
                )
        print("=" * 80)
        for efd in list(self.events.keys()):  # for all of the data in the event files
            if str(efd) not in all_entries:
                print(f"** {str(efd):42s} {self.events[efd]['celltype']:s}")


# =========================================================
# The following code is outside of the EventAnalysis class
# =========================================================


def do_big_summary_plot(EA, args):
    cprint("green", "BigSummaryPlot")
    pdevo = pd.DataFrame.from_dict(EA.events, orient="index")
    groups_to_do = EA.expt["Groups"]

    def shape(lst):
        def ishape(lst):
            shapes = [ishape(x) if isinstance(x, list) else [] for x in lst]
            shape = shapes[0]
            if shapes.count(shape) != len(shapes):
                raise ValueError("Ragged list")
            shape.append(len(lst))
            return shape

        return tuple(reversed(ishape(lst)))

    # print(pdev.head())
    # print(pdev.columns.values)
    # print(pdev.index)
    pdev = pdevo.copy()
    pdev.insert(2, "temp", 3.0)
    pdev.insert(3, "age", 0)
    pdev.insert(4, "internal", "")
    drops = []
    for j, c in enumerate(pdev.index):
        if c.find("alt1") > 0 or c.find("signflip") > 0 or c.find("alt2") > 0:
            drops.append(c)
            continue
        pdev.loc[c, "temp"] = int(EA._get_cell_information(c, "temperature"))
        pdev.loc[c, "age"] = int(EA._get_cell_information(c, "age"))
        pdev.loc[c, "internal"] = EA._get_cell_information(c, "internal")
        scores = np.log10(np.clip(np.max(pdev.loc[c, "scores"]), a_min=1e-1, a_max=1e4))
        scores = np.clip(scores, 0, 4)
        # print(scores)
        pdev.loc[c, "scores"] = np.max(scores)
        # SR - can be list of one dim or list of 2 dim
        srx = np.nanmean([np.nanmean(x) for x in pdev.loc[c, "SR"]])
        pdev.loc[c, "SR"] = srx

        eva = pdev.loc[c, "event_amps"]
        aveev = []
        if isinstance(eva, float):
            eva = [eva]
        for e in eva:  # number of maps for the cell
            # print('  2: ', len(e))
            for ev in e:
                # print('    3: ', len(ev))
                if isinstance(ev, float):
                    if ev != 0.0:
                        aveev.append(ev)
                    continue
                for evn in ev:
                    # print('      4: ', evn)
                    if evn != 0.0:
                        aveev.append(evn)
        evamp = np.nanmean(aveev)
        pdev.loc[c, "event_amps"] = -evamp * 1e12

        fevl = pdev.loc[c, "firstevent_latency"]
        avefevl = []
        if isinstance(fevl, float):
            fevl = [fevl]
        for e in fevl:  # number of maps for the cell
            # print('  2: ', len(e))
            if e is None:
                continue
            if isinstance(e, float):
                avefevl.append(e)
                continue
            for ev in e:
                # print('    3: ', len(ev))
                if isinstance(ev, float):
                    if ev != 0.0:
                        avefevl.append(ev)
                    continue
                for evn in ev:
                    # print('      4: ', evn)
                    if evn != 0.0:
                        avefevl.append(evn)
        fevl = np.nanmean(avefevl)
        # print(fevl)
        pdev.loc[c, "firstevent_latency"] = fevl * 1e3

        # eva = np.nanmean([np.nanmean(x) for x in pdev.loc[c, "event_amps"]])
        # print('   ', eva)
        # for par in ["event_amps", 'SR', 'firstevent_latency']:
        #     # print('par: ', par)
        #     # print(pdev.loc[c, par][0])
        #     if isinstance(pdev.loc[c, par], float):
        #         ash = np.array([pdev.loc[c, par]])
        #     else:
        #         ash = np.array(pdev.loc[c, par][0]).ravel()
        #     # print(par, ash, [ash == None], len(ash))
        #
        #     if isinstance(ash, list):
        #         print('ash is list: ', ash)
        #         exit()
        #     try:
        #         x = np.nanmean(ash.ravel())
        #         # print(x.shape, x.ndim)
        #         if x.ndim >= 1:
        #             x=x.squeeze()
        #             x = np.namean(x)
        #         pdev.loc[c, par] = x
        #         # print(pdev.loc[c, par].ndim)
        #     except:
        #         drops.append(c) #print('ash: ', ash)

    #  for j, c in enumerate(pdev.index):
    #      # print(type(pdev.loc[c, "event_amps"]))
    #      if isinstance(pdev.loc[c, "event_amps"], list):
    #          # return just the average of the first event amp in the first trial
    #          x = pdev.loc[c, "event_amps"][0]  # first map repeat
    #          if isinstance(x, list):
    #              x = np.nanmean(x[0])
    #          elif x.shape[0] > 1:
    #              x = np.nanmean(x[0])
    #          else:
    #              x = np.nanmean(x)
    #          pdev.loc[c]["event_amps"] = x
    #
    #  u = pdev.loc['2017.03.24_000/slice_001/cell_000', "event_amps"]
    # print(type(u)) # u[0])
    # exit()
    pdev = pdev.drop(drops)
    pars = ["scores", "SR", "event_amps", "firstevent_latency"]

    P = PH.regular_grid(
        len(pars),
        len(all_celltypes),
        order="rowsfirst",
        figsize=(10.0, 6),
        margins={
            "leftmargin": 0.08,
            "rightmargin": 0.08,
            "topmargin": 0.1,
            "bottommargin": 0.1,
        },
    )
    ysc = {
        "scores": [0, 4.1],
        "SR": [0, 125],
        "event_amps": [0, 4e3],
        "firstevent_latency": [0, 10.0],
    }
    for i, ct in enumerate(DCN_celltypes):
        for j, par in enumerate(pars):
            a = pdev.query(f'temp < 30.0 and celltype == "{ct:s}"')
            b = pdev.query(f'temp > 30.0 and celltype == "{ct:s}"')
            # print('\n', par, '\n', a.age, a[par])
            # print(a[par].shape)
            if par not in ["event_amps"]:
                P.axarr[j, i].plot(a.age, a[par], "ro", markersize=3)
                P.axarr[j, i].plot(b.age, b[par], "bo", markersize=3)
            else:
                P.axarr[j, i].semilogy(a.age, a[par], "ro", markersize=3)
                P.axarr[j, i].semilogy(b.age, b[par], "bo", markersize=3)

            P.axarr[j, i].set_title(f"{ct:s}  {par:s}", fontsize=10)

            P.axarr[j, i].set_xlim(0, 180)
            P.axarr[j, i].set_ylim(ysc[par])

    mpl.show()
    EA.get_all_data(cmds=args, celltypes=all_celltypes, groups_to_do=groups_to_do)
    exit()


def main():
    # test_events_shuffle()
    parser = argparse.ArgumentParser(description="Map Event Analyzer")
    parser.add_argument(
        "-E",
        "--database",
        type=str,
        default="",
        # choices=["NF107Ai32_Het", "NF107Ai32_NIHL"],
        help="Experimental database for analysis",
    )
    parser.add_argument(
        "-d",
        "--day",
        type=str,
        default="all",
        help="day for analysis",
    )
    parser.add_argument(
        "-a",
        "--after",
        type=str,
        default="1970.1.1",
        dest="after",
        help="only analyze dates on or after a date",
    )
    parser.add_argument(
        "-b",
        "--before",
        type=str,
        default="2266.1.1",
        dest="before",
        help="only analyze dates on or before a date",
    )
    parser.add_argument(
        "-S",
        "--Slice",
        type=str,
        default="",
        dest="slicecell",
        help="select slice/cell for analysis: in format: S0C1 for slice_000 cell_001\n"
        + "or S0 for all cells in slice 0",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        dest="dryrun",
        help="Do not overwrite Event Summary File with --eventsummary",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        dest="verbose",
        help="Print verbose output",
    )
    parser.add_argument(
        "-r",
        "--read-annotations",
        type=str,
        default="",
        dest="annotationFile",
        help="Read an annotation file of selected cells, to replace cell type with post-hoc definitions",
    )

    parser.add_argument(
        "-s",
        "--summarize",
        type=str,
        dest="summarize",
        nargs="?",
        const="None",
        choices=[
            "SR",
            "scores",
            "shufflescore",
            "event_amps",
            "spont_amps",
            "amp_ratio",
            "depression_ratio",
            "paired_pulse_ratio",
            "event_qcontent",
            "largest_event_qcontent",
            "area_fraction_Z",
            "median_Z_abovethreshold",
            "maximum_Z",
            "mean_Z",
            "None",
            "firstevent_latency",
            "allevent_latency",
            "tau1",
            "tau2",
            "IO",
        ],
        default="None",
        help="summarize the data in the eventsummaryfile",
    )
    parser.add_argument(
        "-Z",
        "--zthreshold",
        type=float,
        dest="area_z_threshold",
        default=2.1,
        help="Set threshold for Z score measure of 'responsive' map area",
    )

    parser.add_argument(
        "-P", "--plot", action="store_true", dest="plotflag", help="enable plotting"
    )
    parser.add_argument(
        "-l", "--listsimple", action="store_true", dest="listsimple", help="just list"
    )
    parser.add_argument("-L", "--listonly", action="store_true", dest="listonly", help="just list")
    # parser.add_argument('-g', '--getZ', action='store_true', dest='getz',
    #                                         help='get raw z scores')
    parser.add_argument(
        "-e",
        "--events",
        action="store_true",
        dest="showevents",
        help="show events structure",
    )
    parser.add_argument(
        "--eventsummary",
        action="store_true",
        dest="eventsummary",
        help="Create an updated event summary database",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        dest="force",
        help="force an update of the analysis",
    )
    parser.add_argument(
        "--bigsummaryplot",
        action="store_true",
        dest="bigsummaryplot",
        help="Plot parameters against age...",
    )

    parser.add_argument(
        "--listdatabase",
        action="store_true",
        dest="listdatabase",
        help="list entries in the database after annotations are applied",
    )

    parser.add_argument(
        "--testshuffler", action="store_true", dest="shuffler", help="test the shuffler"
    )

    parser.add_argument(
        "-c",
        "--celltype",
        type=str,
        default="all",
        dest="celltypes",
        choices=all_celltypes + ["DCN", "VCN", "all"],
        help="Set celltype for analysis",
    )

    args = parser.parse_args()

    print("Command args: ", args)
    ############################################

    # get the configuraiton file
    expt = args.database
    datasets, experiments = get_configuration("config/experiments.cfg")
    if args.database not in datasets:
        print(f"Database {args.database:s} not found in configuration file")
        print("Available databases are: ", datasets)
        exit()
    experiment = experiments[args.database]
    # set up list of cell types to process
    if args.celltypes == "DCN":
        celltypes = ["cartwheel", "tuberculoventral", "pyramidal", "giant"]
    elif args.celltypes == "VCN":
        celltypes = ["bushy", "t-stellate", "d-stellate", "octopus"]
    elif args.celltypes == "all":
        celltypes = all_celltypes
    else:
        celltypes = [args.celltypes]  # must be a single specification

    print("Database in Use: ", args.database)

    EA = EventAnalyzer(experiment_dict=experiment)
    EA.set_events_Path(
        Path(
            EA.expt["analyzeddatapath"],
            EA.expt["directory"],
            "events",
        )
    )
    EA.get_databases(args.database)
    print("** event path: ", EA.eventspath)
    print("** event summary file: ", EA.eventsummary_file)
    if args.listonly or args.listsimple:
        print("listonly and listsimple are modifiers to other commands")

    day_before = DUP.parse(args.before)
    day_after = DUP.parse(args.after)
    if args.day == "all":
        day_day = None
    else:
        day_day = DUP.parse(args.day)

    if args.database.find("nihl") > -1 or args.database.find("NIHL") > -1:
        groups = ["A", "AA", "B", "AAA"]
    else:
        groups = ["B"]
        coding = None

    if args.shuffler:
        SH = shuffler.Shuffler()
        SH.test_events_shuffle()
        return

    # Expt = EA.expt[args.database]
    # database = Path(
    #     Expt["analyzeddatapath"],
    #     Expt['directory'],
    #     Expt["datasummaryFilename"],
    # ).with_suffix(".pkl")

    # # print("database: ", database)
    # if not database.is_file():
    #     print("database not found")
    #     exit()
    # if Expt["map_annotationFilename"] is not None:
    #     annotationFile = Path(
    #         Expt["analyzeddatapath"],
    #         Expt["map_annotationFilename"],
    #     )
    # else:
    #     annotationFile = None
    # print(Expt)

    if args.listsimple:
        print("Listing of data in the main database: ")
        EA.listcells(
            celltype=celltypes,
            cmds=args,
            day=day_day,
            day_after=day_after,
            day_before=day_before,
            slice_cell=args.slicecell,
        )
        return

    if args.listdatabase:  # just list the cells in the database after annotation
        EA.listdatabase()
        return

    if args.bigsummaryplot:
        do_big_summary_plot(EA, args)

    if args.summarize != "None":
        # print(eventsummaryfile)
        cprint("green", "args summarize is not None, setting summarization")
        if args.database.find("nihl") > -1 or args.database.find("NIHL") > -1:
            groups = ["A", "AA", "B", "AAA"]
        else:
            groups = ["B"]

        EA.get_all_data(cmds=args, celltypes=celltypes, groups_to_do=groups)
        # print("EA.events: ", dir(EA))
        allf = list(EA.events.keys())
        cts = [EA.events[c]["celltype"] for c in list(EA.events.keys())]
        print("Groups: ", groups)
        # print(cts)

        EA.summarize(cmds=args, groups_to_do=groups, filename=EA.eventsummary_file)
        return

    # if args.getz:
    #     fns = sorted(list(eventspath.glob('*.pkl')))
    #     print('day, slicecell: ', day_day, args.slicecell)
    #     for i, fn in enumerate(fns):
    #         match = EA.check_day_slice_cell(fn.name, day=day_day, day_after=day_after, day_before=day_before,
    #                                      slice_cell=args.slicecell)
    #         # print('match: ', match, fn)
    #         if not match:
    #              continue
    #         evn, updatedata = EA.getZ(filename=fn, eventwindow=[0.001, 0.010], area_z_threshold=args.area_z_threshold,
    #                             plotflag = args.plotflag, force=args.force)
    #         print(evn)

    # if args.get_mepscs:
    #     self.events, db = get_database(database, eventsummaryfile, annotationFile)
    #     fns = sorted(list(eventspath.glob('*.pkl')))
    #     print('day, slicecell: ', day_day, args.slicecell)
    #     for i, fn in enumerate(fns):
    #         match = check_day_slice_cell(fn.name, day=day_day, day_after=day_after, day_before=day_before,
    #                                      slice_cell=args.slicecell)
    #         if not match:
    #              continue
    #         evn, updatedata = getZ(fn, db=db, eventwindow=[0.001, 0.010], area_z_threshold=args.area_z_threshold,
    #                             self.events=self.events,
    #                             plotflag = args.plotflag, force=args.force)
    #         print(evn.keys())
    #

    if args.showevents:
        print("show events: ")
        for c in EA.events:
            print("events for cell: ", c, "\n", EA.events[c]["eventp"])

    if args.eventsummary:
        """
        Update the main eventsummary pkl file from the individual cell events.
        The Cell events are these are held in pickled files in the 'events' directory
        for each cell. This program reads those files, and updates the main eventsummary
        file.
        """
        fns = sorted(list(EA.eventspath.glob("*.pkl")))
        print("Number of event files: ", len(fns))
        # print("day before: ", day_before, "day after: ", day_after)

        updatestatus = {"Nothing": [], "Updated": [], "NoChange": []}
        for i, fn in enumerate(fns):  # for all the cells for which events have been analyzed
            # if there is a selection by day, slice, check it
            match = EA.check_day_slice_cell(
                fn.name,
                day=day_day,
                day_after=day_after,
                day_before=day_before,
                slice_cell=args.slicecell,
            )
            if not match:
                if args.verbose:
                    cprint("r", f"   Cell does not fit selection criteria: {str(fn):s}")
                continue
            else:
                cprint("green", f"\nProceeding to summarize: {str(fn):s}")

            fnk = str(fn.stem).replace("~", "/")  # make original key that matches.
            if fnk.endswith("_alt1") or fnk.endswith("_alt2") or fnk.endswith("_signflip"):
                continue
            addflag = False
            if EA.events is None:
                EA.events = {}
            if fnk not in EA.events.keys():
                cprint("r", f"Cell id: {fnk:s} is not in events database")
                addflag = True
                cellclass = EA._get_cell_information(fnk, "cell_type")
                cellclass = ephys.tools.map_cell_types.map_cell_type(cellclass)
                if cellclass is None:
                    continue
                cprint(
                    "green",
                    f"    Adding scored events on : {fn!s}\n      cell class: {cellclass!s}",
                )
            else:
                cellclass = EA._get_cell_information(fnk, "cell_type")
                cellclass = ephys.tools.map_cell_types.map_cell_type(cellclass)
                if cellclass is None:
                    continue
                if str(cellclass).strip() == "" or str(cellclass).strip() == "nan":
                    cprint(
                        "yellow",
                        f"   Special case cell ({fnk:s}), setting cell class to 'unknown'",
                    )
                    cellclass = "unknown"
                cprint("green", f"    Scoring events on : {fn!s}\n      cell class: {cellclass!s}")

            evn, updatedata = EA.score_events(
                fn,
                eventwindow=[0.0001, 0.010],
                area_z_threshold=args.area_z_threshold,
                plotflag=args.plotflag,
                force=args.force,
                celltype=cellclass,
            )
            if evn is None or evn["scores"] is None:  # nothing to analyze
                cprint("yellow", f"Nothing in scores for day: {fn.name!s} evn={type(evn)},")
                updatestatus["Nothing"].append(fn)
                continue
            # continue
            EA.eventsummary_file = EA.get_eventsummary_filename(args.database)
            cprint("y", f"Updatedata flag is {updatedata!r}")
            if updatedata or addflag:
                if EA.events is None:  # no existing file, so start it up
                    EA.events = {}

                cprint("m", f"Write to event summary file: {EA.eventsummary_file!s}")
                fnk = str(fn.stem).replace("~", "/")  # make original key that matches.
                EA.events[fnk] = evn
                cprint("red", "New data to update is: " + str(fnk))
                updatestatus["Updated"].append(fnk)
                if not args.dryrun:
                    with open(EA.eventsummary_file, "wb") as fh:  # write at each pass
                        pickle.dump(EA.events, fh)  # get the current file
                    cprint("cyan", f"Updated EventSummaryFile: {EA.eventsummary_file!s}")
                else:
                    cprint(
                        "cyan",
                        f"Dry Run; would update EventSummaryFile: {EA.eventsummary_file!s}",
                    )
            else:
                updatestatus["NoChange"].append(fn)

        # report
        print(f"No Change ({len(updatestatus['NoChange']):d} files)")
        for f in updatestatus["NoChange"]:
            print(f"    {str(f):>40s}")
        print()
        cprint("yellow", f"No Data ({len(updatestatus['Nothing']):d} files)")
        for f in updatestatus["Nothing"]:
            cprint("yellow", f"    {str(f):>40s}")
        print()
        if args.dryrun:
            cprint("m", f"To be Updated ({len(updatestatus['Updated']):d} files)")
        else:
            cprint("m", f"Updated ({len(updatestatus['Updated']):d} files)")
        for f in updatestatus["Updated"]:
            cprint("m", f"    {str(f):>40s}")
        print()


if __name__ == "__main__":
    main()
