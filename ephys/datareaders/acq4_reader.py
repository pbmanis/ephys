from __future__ import print_function

#!/usr/bin/python

"""
Class to read acq4 data blocks in simple manner, as a standalone program.
Does not require acq4 link; bypasses DataManager and PatchEPhys

Requires pyqtgraph to read the .ma files and the .index file

"""
import collections
import datetime
import gc
import os
import pprint
import re
import textwrap as WR
from pathlib import Path
from typing import List, Type, Union

import numpy as np
import pylibrary.tools.cprint as CP
import pylibrary.tools.tifffile as tf
import scipy.ndimage as SND

# from ephys.ephys_analysis import MetaArray as EM
from pyqtgraph import configfile

import MetaArray as EM

pp = pprint.PrettyPrinter(indent=4)


class acq4_reader:
    """
    Provides methods to read an acq4 protocol directory
    including data and .index files
    """

    def __init__(
        self, pathtoprotocol: Union[Path, str, None] = None, dataname: Union[str, None] = None
    ) -> None:
        """
        Parameters
        ----------
        pathtoprotocol str or Path (default: None)
            Path to the protocol directory to set for this instance of the reader

        dataname: str (default: None)
            Name of the data file to read (for example, 'MultiClamp1.ma')

        Returns
        -------
        Nothing
        """
        self.protocol = None
        self.last_protocol = None  # to keep from having to look for subdirs many times.
        self.last_dirs = []
        self.bad_clamp_mode = False
        if pathtoprotocol is not None:
            self.setProtocol(pathtoprotocol)
        if dataname is None:
            dataname = "MultiClamp1.ma"  # the default, but sometimes need to use Clamp1.ma
        self.setDataName(dataname)
        self.clampInfo = {}
        self.lb = "\n"
        # establish known clamp devices:
        maxclamps = 4
        clamps = []
        self.clampdevices = []
        for nc in range(maxclamps):  # create a list of valid clamps and multiclamps
            cno = nc + 1  # clamps numbered from 1 typically, not 0
            cname = "Clamp%d" % cno
            mcname = "MultiClamp%d" % cno
            clamps.extend([(cname, "Pulse_amplitude"), (mcname, "Pulse_amplitude")])
            self.clampdevices.extend([cname, mcname])
        aon = "AxoPatch200"
        apn = "AxoProbe"
        clamps.extend([(aon, "Pulse_amplitude"), (apn, "Pulse_amplitude")])
        self.clampdevices.extend([aon, apn])
        self.clamps = clamps

        self.tstamp = re.compile(r"\s*(__timestamp__: )([\d.\d]*)")
        self.clampInfo["dirs"] = []
        self.clampInfo["missingData"] = []
        self.traces = []
        self.data_array = []
        self.commandLevels = []
        self.cmd_wave = []
        self.time_base = []
        self.values = []
        self.trace_StartTimes = np.zeros(0)
        self.sample_rate = []
        self.pre_process_filters = {"LPF": None, "Notch": []}

        self.importantFlag = False  # set to false to IGNORE the important flag for traces
        # CP.cprint('r', f"Important flag at entry is: {self.importantFlag:b}")
        self.error_info = None  # additional information about errors.

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        del self.time_base
        del self.traces
        del self.data_array
        del self.cmd_wave
        del self.values
        del self.clamps
        del self.sample_rate
        del self.clampInfo
        del self.clampdevices
        gc.collect()

    def setImportant(self, flag: bool = False) -> None:
        """
        Parameters
        ----------
        flag : bool (default: False)
            Set the important flag: if True, we pay attention to the flag
            for each trace when returning data; if False, we ignore the flag
        """
        self.importantFlag = flag
        CP.cprint("r", f"Important' flag was set: {flag:b}")

    def setProtocol(self, pathtoprotocol: Union[str, Path, None] = None) -> None:
        """
        Parameters
        ----------
        pathtoprotocol str or Path (default: None)
            Path to the protocol directory to set for this instance of the reader

        Returns
        -------
        Nothing
        """
        self.protocol = pathtoprotocol

    def set_pre_process(self, LPF: Union[None, float] = None, Notch: Union[None, list] = None):
        self.pre_process_filters["LPF"] = LPF
        self.pre_process_filters["Notch"] = Notch

    def setDataName(self, dataname: Union[str, Path]) -> None:
        """
        Set the type (name) of the data metaarray name that will be read
        Parameters
        ----------
        """
        self.dataname = dataname
        self.shortdname = str(Path(self.dataname).stem)

    def subDirs(self, p: Union[str, Path]) -> List:
        """
        return a list of the subdirectories just below this path

        Parameters
        ----------
        p : str  or path (no default)
            path to investigate

        Returns
        -------
        Sorted list of the directories in the path
        """
        p = Path(p)  # convert if not already
        if self.last_protocol == p:
            return self.last_dirs
        dirs = [d for d in list(p.glob("*")) if d.is_dir()]
        CP.cprint("c", f"Found {len(dirs):d} directories under {str(p):s}")
        if len(dirs) == 0:
            return []
            # raise ValueError(f"No directories found under {str(p):s}")
        # CP.cprint("c", f"   all dirs: {str(dirs)!s}")
        # dirs = filter(Path.is_dir, list(Path(p).glob("*")))
        dirs = sorted(list(dirs))  # make sure these are in proper order...
        # for i, dir in enumerate(dirs):
        #     CP.cprint("c", f"   {i:04d} {str(dir.name)!s}")
        self.last_dirs = dirs  # save for the next time
        self.last_protocol = Path(p)
        return dirs

    def checkProtocol(
        self, protocolpath: Union[str, Path, None] = None, allow_partial=False
    ) -> bool:
        """
        Check the protocol to see if the data is complete

        Parameters
        ----------
        protocolpath: str or path (no default)

        allow_partial: bool (default: False)
            If True, allow incomplete protocols (use all available data)
            Empty protocols and protocols with no .index file will still return False
            If False, incomplete protocols will still return false

        Returns
        -------
        Boolean for protocol data found (True) or not found/incomplete (False)

        """
        if protocolpath is None:
            protocolpath = self.protocol
        dirs = self.subDirs(
            protocolpath
        )  # get all sequence entries (directories) under the protocol
        modes = []
        info = self.readDirIndex(protocolpath)  # top level info dict
        if info is None:
            print(
                f"acq4_reader.checkProtocol: Protocol is not managed (no .index file found): {str(protocolpath):s}"
            )

            return False
        info = info["."]
        if "devices" not in info.keys():  # just safety...
            CP.cprint("r", "acq4_reader.checkProtocol: No devices in the protocol")
            CP.cprint("r", "      info keys: {str(list(info.keys())):s")
            return False
        devices = info["devices"].keys()
        clampDevices = []
        for d in devices:
            if d in self.clampdevices:
                clampDevices.append(d)
        if len(clampDevices) == 0:
            CP.cprint("r", "acq4_reader.checkProtocol: No clamp devices found?")
            return False
        mainDevice = clampDevices[0]

        nexpected = len(dirs)  # acq4 writes dirs before, so this is the expected fill
        ncomplete = 0  # count number actually done
        for i, directory_name in enumerate(
            dirs
        ):  # dirs has the names of the runs within the protocol
            datafile = Path(directory_name, mainDevice + ".ma")  # clamp device file name
            clampInfo = self.getDataInfo(datafile)
            if clampInfo is None:
                break
            ncomplete += 1  # count up
        if ncomplete != nexpected:
            print(
                f"acq4_reader.checkProtocol: Completed dirs and expected dirs are different: Completed {ncomplete: d}, expected: {nexpected:d}"
            )
            if not allow_partial:  # block incomplete protocols, unless explicitely allowed
                return False
        return True

    def checkProtocolImportantFlags(self, protocolpath: Union[str, Path, None] = None) -> bool:
        """
        Check the protocol directory to see what "important" flags might be set or not
        for individual traces
        Return a dict of the traces that are "important"
        """

        important = {}
        if protocolpath is None:
            protocolpath = self.protocol
        dirs = self.subDirs(
            protocolpath
        )  # get all sequence entries (directories) under the protocol
        modes = []
        info = self.readDirIndex(protocolpath)  # top level info dict
        if info is None:
            CP.cprint(
                "r",
                "acq4_reader.checkProtocol: Protocol is not managed (no .index file found): {0:s}".format(
                    protocolpath
                ),
            )
            return False
        info = info["."]
        if "devices" not in info.keys():  # just safety...
            CP.cprint("acq4_reader.checkProtocol: No devices in the protocol")
            CP.cprint("  Here are the keys: \n", info.keys())
            return False
        if "important" in info.keys():
            CP.cprint("r", "Important flag set for protocol: {0:s}".format(protocolpath))

        devices = info["devices"].keys()
        clampDevices = []
        for d in devices:
            if d in self.clampdevices:
                clampDevices.append(d)
        if len(clampDevices) == 0:
            print("acq4_reader.checkProtocol: No clamp devices found?")
            return False
        mainDevice = clampDevices[0]

        ncomplete = 0
        nexpected = len(dirs)  # acq4 writes dirs before, so this is the expected fill
        for i, directory_name in enumerate(
            dirs
        ):  # dirs has the names of the runs within the protocol
            datafile = Path(directory_name, mainDevice + ".ma")  # clamp device file name
            tr_info = self.readDirIndex(directory_name)["."]  # get info
            # print('tr_info: ', directory_name.name,  tr_info['.'])
            clampInfo = self.getDataInfo(datafile)
            if clampInfo is None:
                continue
            else:
                if "important" in list(tr_info.keys()):
                    important[directory_name.name] = True
                ncomplete += 1  # count up
        if (
            len(important) == 0 or not self.importantFlag
        ):  # if none were marked, treat as if ALL were marked (reject at top protocol level)
            for i, directory_name in enumerate(dirs):
                important[directory_name.name] = True
        self.important = important  # save, but also return
        return True

    def listSequenceParams(self, dh):
        """Given a directory handle for a protocol sequence, return the dict of sequence parameters"""
        try:
            return dh.info()["sequenceParams"]
        except KeyError:
            if len(dh.info()) == 0:
                CP.cprint(
                    "r",
                    "****************** Error: Missing .index file? (fails to detect protocol sequence)",
                )
                raise Exception(
                    "Directory '%s' does not appear to be a protocol sequence." % dh.name()
                )

    def getIndex(self, currdir: Union[str, Path, None] = None, lineend: str = "\n"):
        self.lb = lineend  # set line break character
        self._readIndex(currdir=currdir)
        if self._index is not None:
            return self._index["."]
        else:
            return None

    def _readIndex(self, currdir: Union[str, Path, None] = None):
        self._index = None
        # first try with currdir value, read current protocolSequence directory
        if currdir == None:
            indexFile = Path(self.protocol, ".index")  # use current
        else:
            indexFile = Path(currdir, ".index")
        if not indexFile.is_file():
            CP.cprint(
                "r", "Directory '%s' is not managed or '.index' file not found" % (str(indexFile))
            )
            print("_readIndex 343:  ", indexFile, " is file: ", indexFile.is_file())
            raise FileNotFoundError
            return self._index
        self._index = configfile.readConfigFile(indexFile)

        return self._index

    def readDirIndex(self, currdir: Union[str, Path, None] = None):
        self._dirindex = None
        indexFile = Path(currdir, ".index")
        if not indexFile.is_file():
            CP.cprint(
                "r", f"Directory '{str(currdir):s}' is not managed or '.index' file not found"
            )
            print("readDirIndex 347:  ", indexFile, " is file: ", indexFile.is_file())
            raise FileNotFoundError
            return self._dirindex
        # print('\nindex file found for currdir: ', currdir)
        # self._dirindex = configfile.readConfigFile(str(indexFile))
        # print(self._dirindex)
        try:
            self._dirindex = configfile.readConfigFile(str(indexFile))
        except:
            CP.cprint("r", f"Failed to read index file for {str(currdir):s}")
            CP.cprint("r", "Probably bad formatting or broken .index file")
            return self._dirindex
        return self._dirindex

    def _parse_timestamp(self, lstr: str):
        tstamp = None
        ts = self.tstamp.match(lstr)
        if ts is not None:
            fts = float(ts.group(2))
            tstamp = datetime.datetime.fromtimestamp(fts).strftime("%Y-%m-%d  %H:%M:%S %z")
        return tstamp

    def convert_timestamp(self, fts: datetime.datetime.timestamp) -> str:
        tstamp = datetime.datetime.fromtimestamp(fts).strftime("%Y-%m-%d  %H:%M:%S %z")
        return tstamp

    def _parse_index(self, index: Union[list, tuple, dict, bytes]):
        """
        Recursive version
        """
        self.indent += 1
        if isinstance(index, list):
            for i in range(len(index)):
                index[i] = self._parse_index(index[i])
                if isinstance(index[i], list):
                    self.textline += "{0:s}  list, len={1:d}{2:s}".format(
                        " " * self.indent * 4, len(index[i]), self.lb
                    )
                else:
                    if not isinstance(index[i], tuple):
                        self.textline += (
                            "{0:s}  {1:d}{2:s}",
                            format(" " * self.indent * 4, index[i], self.lb),
                        )

        elif isinstance(index, tuple):
            self.textline += "{0:s} Device, Sequence : {1:s}, {2:s}{3:s}".format(
                " " * self.indent * 4, str(index[0]), str(index[1]), self.lb
            )

        elif isinstance(index, dict):
            for k in index.keys():
                if k.endswith(".ma") or k.endswith(".tif"):
                    continue
                if k in ["splitter"]:
                    continue

                index[k] = self._parse_index(index[k])
                if isinstance(index[k], list) or isinstance(index[k], np.ndarray):
                    self.textline += f"{' ' * self.indent * 4:s} {str(k):3s} : list/array, len= {len(index[k]):4d}{str(self.lb):s}"
                elif k not in ["__timestamp__", "."]:
                    indents = " " * (self.indent * 4)
                    indents2 = " " * (self.indent * 4)
                    # do a textwrap on ths string
                    if k in ["description", "notes"]:
                        hdr = "{0:s} {1:>20s} : ".format(indents, k)
                        #  self.textline += hdr
                        wrapper = WR.TextWrapper(
                            initial_indent="",
                            subsequent_indent=len(hdr) * " ",
                            width=100,
                        )
                        for t in wrapper.wrap(hdr + str(index[k])):
                            self.textline += t + self.lb
                    else:
                        if not isinstance(index[k], collections.OrderedDict):
                            self.textline += "{0:s} {1:>20s} : {2:<s}{3:s}".format(
                                indents, k, str(index[k]), self.lb
                            )
                        else:
                            break
                elif k in ["__timestamp__"]:
                    tstamp = self.convert_timestamp(index[k])
                    if tstamp is not None:
                        self.textline += "{0:s} {1:>20s} : {2:s}{3:s}".format(
                            " " * self.indent * 4, "timestamp", tstamp, self.lb
                        )

        elif isinstance(
            index, bytes
        ):  # change all bytestrings to string and remove internal quotes
            index = index.decode("utf-8").replace("'", "")
            self.textline += "{0:s}  b: {1:d}{2:s}".format(" " * self.indent * 4, index, self.lb)
        self.indent -= 1
        return index

    def printIndex(self, index: Union[list, tuple, dict, bytes]):
        """
        Generate a nice printout of the index, about as far down as we can go
        """
        self.indent = 0
        self.textline = ""
        t = self._parse_index(index)
        return

    def getIndex_text(self, index: Union[list, tuple, dict, bytes]):
        """
        Generate a nice printout of the index, about as far down as we can go
        """
        self.indent = 0
        self.textline = ""
        t = self._parse_index(index)
        return self.textline

        # for k in index['.'].keys():
        #     print( '  ', k, ':  ', index['.'][k])
        #     if isinstance(index['.'][k], dict):
        #         for k2 in index['.'][k].keys():
        #             print ('    ', k, ' ', k2, '::  ', index['.'][k][k2])
        #             if isinstance(index['.'][k][k2], dict):
        #                 for k3 in index['.'][k][k2]:
        #                     print ('    ', k, ' ', k2, ' ', k3, ':::  ', index['.'][k][k2][k3])
        #                     if isinstance(index['.'][k][k2][k3], dict):
        #                         for k4 in index['.'][k][k2][k3]:
        #                             print( '    [', k, '][', k2, '][', k3, '][', k4, '] ::::  ', index['.'][k][k2][k3][k4])

    def file_cell_protocol(self, filename: Union[str, Path, None] = None) -> tuple:
        """
        file_cell_protocol breaks the current filename down and returns a
        tuple: (date, cell, protocol)
        last argument returned is the rest of the path...
        """
        assert filename is not None
        filename = Path(filename)
        proto = filename.stem
        cell = filename.parent
        sliceid = cell.parent
        date = sliceid.parent.name
        return (date, sliceid.name, cell.name, proto, sliceid.parent)

    def getClampDevices(
        self, currdir: Union[str, Path, None] = None, verbose: bool = False
    ) -> dict:
        """
        Search for a known clamp device in the list of devices
        used in the current protocol directory...

        Return
        ------
        list of valid clamp devices found (there may be more than one)
            List will be empty if no recognized device is found.
        """
        assert currdir is not None
        info = self.getIndex(currdir=currdir)
        if verbose:
            print("\ngetClampDevices info: ", info["devices"])
        devs = []
        if info is not None and "devices" in info.keys():
            devices = info["devices"]
            for d in devices:
                if d in self.clampdevices:
                    devs.append(d)
        return devs

    def getDataInfo(self, filename: Union[str, Path, None] = None, silent: bool = False):
        """
        Get the index info for a record, without reading the trace data
        """
        assert filename is not None
        info = None
        fn = Path(filename)
        if fn.is_file():
            try:
                tr = EM.MetaArray(file=fn, readAllData=False)
                info = tr[0].infoCopy()
                self.parseClampInfo(info)
            except:
                CP.cprint("r", f"The file: {str(fn):s} could not be parsed... ")
                return None
        return info

    def parseClampInfo(self, info: list):
        """
        Get important information from the info[1] directory that we can use
        to determine the acquisition type and channel order
        """
        self.trClampInfo = info
        # print("switchchan: ", switchchan)
        # indices correspond to info[0][cols]
        self.primary_trace_index = 0
        self.secondary_trace_index = 1
        self.command_trace_index = 1
        # print("\nparse clamp info, Clamp state: \n", info[1]) # ["ClampState"])

        # try:  # old acq4 files may not have this
        self.mode = info[1]["ClampState"]["mode"]
        primary_signal = info[1]["ClampState"]["primarySignal"]
        secondary_signal = info[1]["ClampState"]["secondarySignal"]
        # print("Mode: ", self.mode)
        if self.mode in ["IC", "I=0"]:
            match primary_signal:
                case "Membrane Potential":
                    self.primary_trace_index = 0
                    self.secondary_trace_index = 1
                case "Pipette Potential":
                    self.primary_trace_index = 1
                    self.secondary_trace_index = 0
                case _:
                    pass
        elif self.mode in ["VC"]:
            match primary_signal:
                case "Membrane Current":
                    self.primary_trace_index = 0
                    self.secondary_trace_index = 1
                case "Pipette Potential":
                    self.primary_trace_index = 1
                    self.secondary_trace_index = 0
                case _:
                    pass
        #         chorder[0] = 1
        #     if secondary_signal == "Pipette Potential":
        #         chorder[1] = 2
        #     self.command_trace_index = chorder[1]
        # elif self.mode in ["VC"]:
        #     self.primary_trace_index = chorder[0]
        #     if secondary_signal == "Pipette Potential":
        #         chorder[1] = 2
        #     self.command_trace_index = chorder[1]
        else:
            raise ValueError(f"Unable to determine how to map channels for mode = {self.mode:s}")
        # print(self.primary_trace_index, self.secondary_trace_index, self.command_trace_index)

        # print("Mode found: ", self.mode)
        # except:  # this is for "old" acq4 data files.  very old.
        #     print("info 1: ")
        #     for k in list(info[1].keys()):
        #         print(k, '=', info[1][k])
        #     print("info 0: ")
        #     for k in list(info[0].keys()):
        #         print(k, '=', info[0][k])
        #     self.units = [info[1]['units'], 'V']
        #     self.samp_rate = info[1]['rate']
        #     if info[1]['units'] == 'V':
        #         self.mode = 'IC'
        #     if self.mode in ["IC", "I=0"]:
        #         if not switchchan:
        #             self.primary_trace_index = 1
        #             self.command_trace_index = 0
        #         else:
        #             self.primary_trace_index = 0
        #             self.command_trace_index = 1
        #     elif self.mode in ["VC"]:
        #         if not switchchan:
        #             self.primary_trace_index = 1
        #             self.command_trace_index = 0
        #         else:
        #             self.primary_trace_index = 0
        #             self.command_trace_index = 1

        #     return

        self.units = [
            info[1]["ClampState"]["primaryUnits"],
            info[1]["ClampState"]["secondaryUnits"],
        ]
        self.samp_rate = info[1]["DAQ"]["primary"]["rate"]
        # CP.cprint("r", f"parseclampinfo, mode = {self.mode:s}")

    def parseClampWCCompSettings(self, info: list) -> dict:
        """
        Given the .index file for this protocol dir, try to parse the
        clamp state and compensation
        """
        d = {}
        if "ClampState" in info[1].keys() and "ClampParams" in info[1]["ClampState"].keys():
            par = info[1]["ClampState"]["ClampParams"]
            d["WCCompValid"] = True
            d["WCEnabled"] = par["WholeCellCompEnable"]
            d["WCResistance"] = par["WholeCellCompResist"]
            d["WCCellCap"] = par["WholeCellCompCap"]
            d["RsCompCorrection"] = par["RsCompCorrection"]
            d["CompEnabled"] = par["RsCompEnable"]
            d["CompCorrection"] = par["RsCompCorrection"]
            d["CompBW"] = par["RsCompBandwidth"]
            return d
        else:
            return {
                "WCCompValid": False,
                "WCEnable": 0,
                "WCResistance": 0.0,
                "WholeCellCap": 0.0,
                "CompEnable": 0,
                "CompCorrection": 0.0,
                "CompBW": 50000.0,
                "RsCompCorrection": 0.0,
            }

    def parseClampCCCompSettings(self, info: list) -> dict:
        d = {}
        if "ClampState" in info[1].keys() and "ClampParams" in info[1]["ClampState"].keys():
            par = info[1]["ClampState"]["ClampParams"]
            d["CCCompValid"] = True
            d["CCBridgeEnable"] = par["BridgeBalEnable"]
            d["CCBridgeResistance"] = par["BridgeBalResist"]
            d["CCNeutralizationEnable"] = par["NeutralizationEnable"]
            d["CCNeutralizationCap"] = par["NeutralizationCap"]
            d["CCLPF"] = par["PrimarySignalLPF"]
            d["CCPipetteOffset"] = par["PipetteOffset"]
            return d
        else:
            return {
                "CCCompValid": False,
                "CCBridgeEnable": 0,
                "CCBridgeResistance": 0.0,
                "CCNeutralizationEnable": 0.0,
                "CCNeutralizationCap": 0,
                "CCPipetteOffset": 0.0,
                "CCLPF": 10000.0,
            }

    def parseClampHoldingLevel(self, info: list):  # -> Union[float, list]:
        """
        Given the .index file for a protocol dir, try to get
        the holding level from the clamp state
        """
        try:
            return info[1]["ClampState"]["holding"]
        except KeyError:
            return 0.0

    def _getImportant(self, info: Union[int, dict]):
        if info is None:
            important = False
            return important
        if "important" in list(info.keys()):
            important = info["important"]
        else:
            important = False
        # CP.cprint('r', f"_getImportant: Important flag was identified: {important:b}")
        return important

    def checkProtocolImportant(self, protocol):
        """
        Check the *protocol dir* to see if the protocol is marked as "important"
        """
        self.setProtocol(protocol)
        self.info = self.getIndex(self.protocol)  # self.protocol)
        important = self._getImportant(self.info)
        return important

    def getData(
        self,
        pos: int = 1,
        check: bool = False,
        allow_partial: bool = False,
        record_list: list = [],  # if allow partial, if this list is not None, we only return the records in this list, otherwise we return all lthe partial records
        downsample: int = 1,
        silent: bool = True,
    ):
        """
        Get the data for the current protocol
        if check is True, we just check that the requested file exists and return
        True if it does and false if it does not
        if allow_partial is true, we get as much data as we can even if the protocol
        did not complete.
        if record_list is an empty list, and allow_partial is FALSE, we try to get ALL the records.
        if record_list is an empty list and allow_partial is TRUE, get as much as we can.
        if record_list is a list of integers and allow_partial is TRUE, we only get the records in that list (e.g., predetermined)
        if record_list is a list of integers and allow_partial is FALSE, we raise an error, because we did not intend to allow partial dataset reads.

        Returns False if it fails.
        Returns True if everything succeeds.

        Raises value errors :
        1. if record_list is not empty and allow_partial is False
        2. The protocol selected is not a directory
        3. The protocol directory is empty
        4. The selected device (e.g., Multiclamp1) is not found in the protocol
        5. The protocol is not managed (no .index file found)
        6. 

        """
        # non threaded
        # CP.cprint('c', 'GETDATA ****')
        if len(record_list) > 0 and not allow_partial:
            raise ValueError("acq4_reader.getData: The record_list is not empty and allow_partial is False. This is NOT an accepted combination")
        
        self.error_info = None  # clear any previous error info
        dirs = self.subDirs(self.protocol)
        if len(dirs) == 0:
            if Path(self.protocol).is_dir():
                self.error_info = f"Protocol dir exists, but is empty or has malformed structure: {self.protocol!s}"
                CP.cprint("r", self.error_info)
            else:
                self.error_info = f"Protocol directory was not found: {self.protocol!s}"
                CP.cprint("r", self.error_info)
            return False
        index = self._readIndex()
        self.info = self.getIndex(self.protocol)  # self.protocol)
        # print("info: ", self.info)
        # CP.cprint("c", f"acq4_read: Found {len(dirs):d} directories in {self.protocol:s}")
        self.clampInfo["dirs"] = dirs
        self.clampInfo["missingData"] = []
        self.traces = []
        self.trace_index = []
        self.trace_important = []
        self.data_array = []
        self.commandLevels = []
        self.cmd_wave = []
        self.time_base = []
        self.values = []
        self.trace_StartTimes = np.zeros(0)
        self.sample_rate = []
        self.info = self.getIndex(self.protocol)  # self.protocol)
        self.protocol_important = self._getImportant(self.info)  # sa
        holdcheck = False
        holdvalue = 0.0
        switchchan = False
        self.getDataInfo(Path(dirs[0], self.dataname))
        if self.info is not None and "devices" in list(self.info.keys()):
            devices = list(self.info["devices"].keys())
            if devices[0] == "DAQ":
                device = devices[1]
            else:
                device = devices[0]

            if device in self.clampdevices:
                if device not in list(self.info["devices"].keys()):
                    print(f"**Unable to match device: {device:s} in ")
                    print("    Device keys: ", self.info["devices"].keys())
                    raise ValueError

                holdcheck = self.info["devices"][device]["holdingCheck"]
                holdvalue = self.info["devices"][device]["holdingSpin"]
                # priSignal = self.info["devices"][device]["primarySignalCombo"]
                # secSignal = self.info["devices"][device]["secondarySignalCombo"]
                # ic_ampmode = self.info["devices"][device]["icModeRadio"]
                # CP.cprint('r', f"priSignal: {priSignal:s}   secSignal: {secSignal:s}  ic_ampmode: {ic_ampmode!s}")
                ic_amp_mode = self.trClampInfo[1]["ClampState"][
                    "mode"
                ]  # self.checkProtocolinfo[1]["ClampState"]["mode"]
                primary_signal = self.trClampInfo[1]["ClampState"]["primarySignal"]
                # secondary_signal = self.trClampInfo[1]["ClampState"]["secondarySignal"]
                # CP.cprint('r', f"priSignal: {primary_signal:s}   secSignal: {secondary_signal:s}  ic_ampmode: {ic_amp_mode:s}")
                self.v_scalefactor = 1.0
                self.bad_clamp_mode = False
                if ic_amp_mode in ["IC", "I=0"] and primary_signal != "Membrane Potential":
                    # Erroneous report from mulitclamp - Change scaling
                    CP.cprint(
                        "r",
                        f"Inconsistent amplifier mode: {ic_amp_mode:s} and primary signal: {primary_signal:s}",
                    )
                    CP.cprint("r", f"    Will attempt to correct...")
                    self.v_scalefactor = (
                        52e-3 / 1.332e-9
                    )  # picked from a trace where this happened: mV/nA match up.
                    self.bad_clamp_mode = True

        else:
            if check:
                print("Just checking - no data to load")
                return False
            else:
                raise ValueError
        self.holding = holdvalue
        trx = []
        cmd: list = []

        # CP.cprint('r', f"_getImportant: Protocol Important flag was identified: {self.protocol_important:b}")
        sequence_values = None
        self.sequence = []
        if index is not None and "sequenceParams" in index["."].keys():
            self.sequence = index["."]["sequenceParams"]

        # building command voltages or currents - get amplitudes to clamp
        reps = ("protocol", "repetitions")
        foundclamp = False
        for clamp in self.clamps:
            if clamp in self.sequence:
                foundclamp = True
                self.clampValues = self.sequence[clamp]
                self.nclamp = len(self.clampValues)
                if sequence_values is not None:
                    sequence_values = [x for x in self.clampValues for y in sequence_values]
                else:
                    sequence_values = [x for x in self.clampValues]
        self.mode = None
        self.protoDirs = []
        # get traces marked "important"
        # if no such traces exist, then accept ALL traces
        important = []
        self.trace_StartTimes = np.zeros(len(dirs))
        for i, d in enumerate(dirs):
            if self.importantFlag:
                important.append(self._getImportant(self.getIndex(d)))
            else:
                important.append(True)
        if sum(important) % 2 == 0:  # even number of "True", fill in between.
            state = False
            for i in range(len(important)):
                if important[i] is True and state is False:
                    state = True
                    continue
                if important[i] is False and state is True:  # transistion to True
                    important[i] = state
                    continue
                if important[i] is True and state is True:  # next one goes back to false
                    state = False
                    continue
                if important[i] is False and state is False:  # no change...
                    continue

        if not any(important):
            important = [True for i in range(len(important))]  # set all true
        self.trace_important = important

        j = 0
        # get traces.
        # if traces are not marked (or computed above) to be "important", then they
        # are skipped
        self.nprotodirs = len(dirs)  # save this...
        tr = None
        for i, d in enumerate(dirs):
            fn = Path(d, self.dataname)
            if check:
                if not fn.is_file():
                    CP.cprint(
                        "r",
                        f"acq4_reader.getData: File not found:  {str(fn):s}, {str(self.dataname):s}",
                    )
                    return False
                return True  # just note we found the first file

            if (
                self.importantFlag and not important[i] and not silent
            ):  # only return traces marked "important"
                CP.cprint("m", "acq4_reader: Skipping non-important data")
                continue
            self.rec_info = self.getDataInfo(fn)
            self.protoDirs.append(Path(d).name)  # keep track of valid protocol directories here
            if allow_partial and isinstance(record_list, list) and len(record_list) > 0:  # only get the records in the list - allows us to capture incomplete protocols
                # CP.cprint("y", "Partial with record list > 0")
                if i not in record_list:  # that was easy...
                    continue
                else:
                    try:
                        tr = EM.MetaArray(file=fn)
                        if tr is None:
                            CP.cprint("y", f"acq4_reader: Failed to read traces in file: {str(fn):s}, when allowing partial read")
                            CP.cprint("y", f"    Record {i:d} will be skipped; record list is: {record_list}")
                            raise ValueError(f"acq4_reader:getData: File read with MetaArray failed: {str(fn):s}")
                        if not silent:
                            CP.cprint("y", f"acq4_reader: Successfully read traces in file: {str(fn):s}, when allowing partial read")
                            CP.cprint("y", f"    Record {i:d} will be included; record list is: {record_list}")
                    except FileExistsError:
                        CP.cprint("y", f"acq4_reader: Failed to read traces in file: {str(fn):s}, when allowing partial read")
                        CP.cprint("y", f"    Record {i:d} will be skipped; record list is: {record_list}")
                        raise ValueError(f"acq4_reader:getData: File read with MetaArray failed: {str(fn):s}")
                    except not FileExistsError:
                        CP.cprint("y", f"acq4_reader: Failed to read traces in MISSING file: {str(fn):s}, when allowing partial read")
                        raise ValueError(f"acq4_reader:getData: File read with MetaArray failed: {str(fn):s}")


            else:  # this is the canonical case: get all records.
                # CP.cprint("g", "NO Partial")
                try:
                    tr = EM.MetaArray(file=fn)
                    if tr is None:
                        CP.cprint("r", f"acq4_reader: Failed to read traces in file: {str(fn):s}")
                        raise ValueError(f"acq4_reader:getData: File read with MetaArray failed: {str(fn):s}")
                except ValueError:
                    if not silent:
                        msg = f"acq4_reader: Failed to read traces in file, could not read metaarray: \n    {str(fn):s}"
                        CP.cprint("r", msg)
                        CP.cprint("r", f"{str(fn):s} \n    may not be a valid clamp file or may be corrupted")
                    raise ValueError(f"acq4_reader:getData: File read with MetaArray failed: {str(fn):s}")


            # now we should have data and record info to decode
            self.trace_StartTimes[i] = self.rec_info[1]['startTime']
            tr_info = tr[0].infoCopy()
            self.parseClampInfo(tr_info)
            self.WCComp = self.parseClampWCCompSettings(tr_info)
            self.CCComp = self.parseClampCCCompSettings(tr_info)
            if self.bad_clamp_mode:
                if tr.hasColumn("Channel", "primary"):
                    tr["Channel":"primary"] *= self.v_scalefactor
            cmd = self.getClampCommand(tr)
            self.trace_StartTimes[i] = self.rec_info[1]['startTime']
            self.traces.append(tr)
            self.trace_index.append(i)
            trx.append(tr.view(np.ndarray))
            if tr.hasColumn("Channel", "primary"):  # hascolumn is a metaarray method
                self.data_array.append(tr["Channel":"primary"])
            if tr.hasColumn("Channel", "command"):
                self.cmd_wave.append(tr["Channel":"command"])
            else:
                print("cmd wave len: ", len(self.cmd_wave))
                print("tr view shape: ", tr.view(np.ndarray).shape)
                print("cmd trace index: ", self.command_trace_index)
                self.cmd_wave.append(tr.view(np.ndarray)[self.command_trace_index])

            if sequence_values is not None:
                if j >= len(sequence_values):
                    j = 0
                self.values.append(sequence_values[j])
                j += 1
            self.time_base.append(tr.xvals("Time"))
            sr = tr_info[1]["DAQ"]["primary"]["rate"]
            self.sample_rate.append(self.samp_rate)

        if tr is None and not allow_partial and not silent:
            CP.cprint("r", "acq4_reader.getData - Failed to read trace data: No traces found?")
            return False
        if self.mode is None:
            units = "A"  # just fake it
            self.mode = "VC"
        if "v" in self.mode.lower():
            units = "V"
        else:
            units = "A"
        try:
            self.traces = np.array(trx)
        except:
            CP.cprint("y", "acq4_reader ?data does not have consistent shape in the dataset")
            dim1_len = []
            for i in range(len(trx)):
                dim1_len.append(trx[i].shape[1])
            CP.cprint("y", f"          Dim 1 has lengths of {str(sorted(list(set(dim1_len)))):s}")
            dim1_new = np.min(dim1_len)
            # print("data array: ", len(self.data_array[0]))

            CP.cprint("y", "          Reshaping to shortest length in time dimension")
            for i in range(len(trx)):
                trx[i] = trx[i][:, :dim1_new]  # make all lise elements the same shorter size
                self.data_array[i] = self.data_array[i][:dim1_new]
                self.cmd_wave[i] = self.cmd_wave[i][:dim1_new]
                self.time_base[i] = self.time_base[i][:dim1_new]
            try:
                self.traces = np.array(trx)
            except:
                self.error_info = f"Failed to reshape array as required"
                CP.cprint("r", self.error_info)
                return False

        if len(self.values) == 0:
            ntr = len(self.traces)
            self.traces = self.traces[:ntr]
            self.values = np.zeros(ntr)  # fake
        else:
            ntr = len(self.values)
        if isinstance(cmd, list):
            uni = "None"
        else:
            uni = cmd.axisUnits(-1)
        # try:
        self.traces = EM.MetaArray(
            self.data_array,
            info=[
                {
                    "name": "Command",
                    "units": uni,
                    "values": np.array(self.values),
                },
                tr.infoCopy("Time"),
                tr.infoCopy(-1),
            ],
        )
    # except:
    #     CP.cprint("r", "No valid traces found")
    #     return False
        self.cmd_wave = EM.MetaArray(
            self.cmd_wave,
            info=[
                {
                    "name": "Command",
                    "units": cmd.axisUnits(-1),
                    "values": np.array(self.values),
                },
                tr.infoCopy("Time"),
                tr.infoCopy(-1),
            ],
        )
        self.sample_interval = 1.0 / self.sample_rate[0]
        self.data_array = np.array(self.data_array)
        self.time_base = np.array(self.time_base[0])
        protoreps = ("protocol", "repetitions")
        scannertargets = ("Scanner", "targets")
        mclamppulses = (self.shortdname, "Pulse_amplitude")
        mclamppulses2 = (self.shortdname, "Pulse2_amplitude")
        valid_seq_keys = set([ protoreps, mclamppulses, mclamppulses2, scannertargets])
        # set some defaults in case there is no .index file
        self.repetitions = 1
        self.tstart = 0.0
        self.tend = np.max(self.time_base)
        self.commandLevels = np.array([0.0])
        if index is not None:
            seqparams = index["."]["sequenceParams"]
            stimuli = index["."]["devices"][self.shortdname]["waveGeneratorWidget"]["stimuli"]
            if "Pulse" in list(stimuli.keys()):
                self.tstart = stimuli["Pulse"]["start"]["value"]
                self.tend = self.tstart + stimuli["Pulse"]["length"]["value"]
            else:
                self.tstart = 0.0
                self.tend = np.max(self.time_base)
            seqkeys = list(seqparams.keys())
            if (mclamppulses not in seqkeys) and (mclamppulses2 not in seqkeys) and (protoreps not in seqkeys) and (scannertargets not in seqkeys):
                print("seqkeys cannot handle the standard mclamppulse names and missing protoreps ")
                print("Seqkeys is: ", seqkeys)
                print("This is what I encountered: ")
                print(f"    mclamppulses: {mclamppulses}")
                print(f"    mclamppulses2: {mclamppulses2}")
                print(f"    protoreps: {protoreps}")
                print("yYu probably need to add this to datareaders/acq4_reader.py near line 1008")
                raise ValueError("Cannot parse the protocol sequence information")
            if mclamppulses in seqkeys:
                if protoreps in list(seqparams.keys()):
                    self.repetitions = len(seqparams[protoreps])
                self.commandLevels = np.repeat(
                    np.array(seqparams[mclamppulses]), self.repetitions
                ).ravel()
                function = index["."]["devices"][self.shortdname]["waveGeneratorWidget"]["function"]
            elif mclamppulses2 in seqkeys:
                if protoreps in list(seqparams.keys()):
                    self.repetitions = len(seqparams[protoreps])
                self.commandLevels = np.repeat(
                    np.array(seqparams[mclamppulses2]), self.repetitions
                ).ravel()
                function = index["."]["devices"][self.shortdname]["waveGeneratorWidget"]["function"]
            
            elif protoreps in seqkeys:
                self.repetitions = len(seqparams[protoreps])
                # WE probably should reshape the data arrays here (traces, cmd_wave, data_array)
                # data = np.reshape(self.AR.traces, (self.AR.repetitions, int(self.AR.traces.shape[0]/self.AR.repetitions), self.AR.traces.shape[1]))
            elif scannertargets in seqkeys and protoreps not in seqkeys:  # no depth, just one flat rep
                self.repetitions = 1
            else:
                print("sequence parameter keys: ", seqkeys)
                raise ValueError(" cannot determine the protocol repetitions with available scanner keys")
            if allow_partial and len(record_list) > 0:
                self.commandLevels = self.commandLevels[record_list]
        return True

    def getClampCommand(
        self, data: Type[EM.MetaArray], generateEmpty: bool = True
    ) -> Union[Type[EM.MetaArray], None]:
        """Returns the command data from a clamp MetaArray.
        If there was no command specified, the function will
        return all zeros if generateEmpty=True (default).
        """

        if data.hasColumn("Channel", "Command"):  # hascolumn is a metaarray method
            return data["Channel":"Command"]
        elif data.hasColumn("Channel", "command"):
            return data["Channel":"command"]
        else:
            if generateEmpty:
                tVals = data.xvals("Time")
                #   mode = getClampMode(data)
                print("Mode: ", self.mode)
                if "v" in self.mode.lower():
                    units = "V"
                else:
                    units = "A"
                return EM.MetaArray(
                    np.zeros(tVals.shape),
                    info=[
                        {"name": "Time", "values": tVals, "units": "s"},
                        {"units": units},
                    ],
                )
        return None

    def getStim(self, stimname: str = "Stim") -> dict:
        supindex = self._readIndex(currdir=self.protocol)
        # print(supindex["."]["devices"].keys())
        if supindex is None:
            supindex = self._readIndex()
            if supindex is None:
                raise ValueError("Cannot read index....")
        stimuli = supindex["."]["devices"][stimname]["channels"]["command"]
        stimuli = stimuli["waveGeneratorWidget"]["stimuli"]
        return self._getPulses(stimuli)

    def getLaserBlueTimes(self) -> dict:
        """
        Get laser pulse times  - handling multiple possible configurations (ugly)
        """
        supindex = self._readIndex(currdir=self.protocol)
        if supindex is None:
            supindex = self._readIndex()
            if supindex is None:
                raise ValueError("Cannot read index....")
        # print(supindex['.']['devices']['PockelCell']['channels']['Switch'].keys())
        if "Laser-Blue-raw" not in list(supindex["."]["devices"].keys()):
            return False
        try:
            stimuli = supindex["."]["devices"]["Laser-Blue-raw"]["channels"]["pCell"]
        except:
            try:
                stimuli = supindex["."]["devices"]["PockelCell"]["channels"]["Switch"]
            except:
                print(supindex["."].keys())
                print(supindex["."]["devices"].keys())
                print(supindex["."]["devices"]["PockelCell"])
                print(supindex["."]["devices"]["PockelCell"]["channels"].keys())
                raise ValueError("Unable to parse devices PockeCell")
        stimuli = stimuli["waveGeneratorWidget"]["stimuli"]
        self.LaserBlueTimes = self._getPulses(stimuli)
        return True

    def getLEDCommand(self) -> bool:
        """
        Get LED pulse times
        Note: In some early recordings, this was used with the laser driver, so the name
        might be "Laser-Blue-raw" instead of "LED-Blue"
        """
        dirs = self.subDirs(self.protocol)

        supindex = self._readIndex(currdir=self.protocol)
        if supindex is None:
            supindex = self._readIndex()
            if supindex is None:
                raise ValueError("Cannot read index....")
        # print(supindex['.']['devices']['PockelCell']['channels']['Switch'].keys())
        self.LED_Raw = []
        self.LED_time_base = []
        self.LED_sample_rate = []
        light_device_name = "LED-Blue"
        if "LED-Blue" not in list(supindex["."]["devices"].keys()):
            # # try substitute with Laser-Blue-raw (was sometimes used when LED was being used instead)
            # if "Laser-Blue-raw" in list(supindex["."]["devices"].keys()):
            #     print("LED-Blue not in device list, but Laser-Blue-raw is")
            #     light_device_name = "Laser-Blue-raw"
            # else:
            #     # print(f"LED-BLue not in device list: {list(supindex['.']['devices'].keys())!s}")
               return False
        # if light_device_name == "Laser-Blue-raw":
        #     # fake it by getting the data from the stated device
        #     self.getLaserBlueTimes()
        #     self.LEDTimes = self.LaserBlueTimes
        #     for i, d in enumerate(dirs):
        #         fn = Path(d, f"{light_device_name:s}.ma")
        #         if not fn.is_file():
        #             print(" acq4_reader.getLEDCommand, File not found: ", fn)
        #             return False
        #         lbr = EM.MetaArray(file=fn)
        #         info = lbr[0].infoCopy()
        #         # print("LBT daq info: ", info[1]["DAQ"])
        #         try:
        #             sr = info[1]["DAQ"]["Shutter"]["rate"] / info[1]["DAQ"]["Shutter"]["downsampling"]
        #         except:
        #             raise ValueError(
        #                 f"Info keys for LED/laser is missing requested DAQ.samplerate: {info[1]['DAQ'].keys()!s}"
        #             )
        #         self.LED_sample_rate.append(sr)
        #         self.LED_Raw.append(lbr.view(np.ndarray)[0])  # shutter
        #         self.LED_time_base.append(lbr.xvals("Time"))
        #     return True
    

        stimuli = supindex["."]["devices"][light_device_name]["channels"]["Command"]
        self.LEDTimes = None
        if "stimuli" in stimuli["waveGeneratorWidget"]:
            real_stimuli = stimuli["waveGeneratorWidget"]["stimuli"]
            self.LEDTimes = self._getPulses(real_stimuli)
        elif 'function' in stimuli["waveGeneratorWidget"]:
            cmd = stimuli["waveGeneratorWidget"]['function']
            # print 'function'
            # clean up the command text to evaluate
            pulse = cmd.split('\n ')
            pulse[0] = pulse[0].strip('\'')
            # print("pulse: ", pulse[0])
            # here we get the paramaters, and clean up the text so we can
            # eval them. Not very clean... 
            pulsestart = eval(pulse[1][7:-1].strip())
            pulsedur = eval(pulse[2][8:-1].strip().replace("*ms", ""))
            amplitudes = eval(pulse[3][8:-1].strip())
            # print("pulsestart: ", pulsestart, type(pulsestart))
            # now build the dictionary that we expect (similar to _getPulses below)
            real_stimuli = {}
            real_stimuli["start"] = pulsestart
            real_stimuli["duration"] = [pulsedur]*len(pulsestart)
            real_stimuli["amplitude"] = amplitudes
            real_stimuli["period"] = []
            real_stimuli["type"] = "pulse"
            real_stimuli["npulses"] = len(pulsestart)
            self.LEDTimes = real_stimuli
        else:
            raise ValueError(f"Unable to parse LED command from {stimuli!s}")

        if ("Scanner", "targets") in supindex["."]['sequenceParams']:
            target = supindex["."]['sequenceParams'][("Scanner", "targets")][0]  # only one target with LED
        else:
            target = [0,0]
        self.LED_target = target  # nominal LED position - center of the FOV

        for i, d in enumerate(dirs):
            fn = Path(d, f"{light_device_name:s}.ma")
            if not fn.is_file():
                print(" acq4_reader.getLEDCommand, File not found: ", fn)
                return False
            lbr = EM.MetaArray(file=fn)
            info = lbr[0].infoCopy()
            try:
                sr = info[1]["DAQ"]["Command"]["rate"] / info[1]["DAQ"]["Command"]["downsampling"]
            except:
                raise ValueError(
                    f"Info keys for LED/laser is missing requested DAQ.samplerate: {info[1]['DAQ'].keys()!s}"
                )
            self.LED_sample_rate.append(sr)
            self.LED_Raw.append(lbr.view(np.ndarray)[0])  # shutter
            self.LED_time_base.append(lbr.xvals("Time"))
        return True

    def _getPulses(self, stimuli: dict) -> dict:
        if "PulseTrain" in stimuli.keys():
            times = {}
            times["start"] = []
            tstart = [stimuli["PulseTrain"]["start"]["value"]]
            times["duration"] = []
            times["amplitude"] = []
            times["npulses"] = stimuli["PulseTrain"]["pulse_number"]["value"]
            times["period"] = stimuli["PulseTrain"]["period"]["value"]
            times["type"] = stimuli["PulseTrain"]["type"]
            for n in range(times["npulses"]):
                times["start"].append(tstart[0] + n * times["period"])
                times["duration"].append(stimuli["PulseTrain"]["length"]["value"])
                times["amplitude"].append(stimuli["PulseTrain"]["amplitude"]["value"])

        elif "Pulse" in stimuli.keys():
            times = {}
            times["start"] = []
            times["duration"] = []
            times["amplitude"] = []
            times["period"] = []
            times["type"] = stimuli["Pulse"]["type"]
            times["npulses"] = [len(list(stimuli.keys()))]
            laststarttime = 0.0
            for n, key in enumerate(stimuli.keys()):  # extract each "pulse" - keys will vary...
                starttime = stimuli[key]["start"]["value"]
                times["start"].append(stimuli[key]["start"]["value"])
                times["duration"].append(stimuli[key]["length"]["value"])
                times["amplitude"].append(stimuli[key]["amplitude"]["value"])
                times["period"].append(starttime - laststarttime)
                laststarttime = starttime

        elif "Pulse3" in stimuli.keys():
            times = {}
            times["start"] = [stimuli["Pulse3"]["start"]["value"]]
            times["duration"] = [stimuli["Pulse3"]["length"]["value"]]
            times["amplitude"] = [stimuli["Pulse3"]["amplitude"]["value"]]
            times["type"] = stimuli["Pulse3"]["type"]

        elif "Pulse5" in stimuli.keys():
            # this is what it looks like:
            # Stimuli:  OrderedDict([('Pulse5',
            #   OrderedDict([('start', OrderedDict([('sequence', 'off'), ('value', 0.3)])),
            #   ('length', OrderedDict([('sequence', 'off'),  ('value', 0.005)])),
            #   ('amplitude', OrderedDict([('sequence', 'off'), ('value', 5.0)])),
            #   ('sum', OrderedDict([('affect', 'length'),
            #   ('sequence', 'off'), ('value', 0.025)])),
            #  ('type', 'pulse')]))])
            times = {}
            times["start"] = [stimuli["Pulse5"]["start"]["value"]]
            times["duration"] = [stimuli["Pulse5"]["length"]["value"]]
            times["amplitude"] = [stimuli["Pulse5"]["amplitude"]["value"]]
            times["type"] = stimuli["Pulse5"]["type"]
        else:
            print("Stimuli: ", stimuli)
            raise ValueError("need to find keys for stimulus (might be empty): " % stimuli)

        return times

    def getDeviceData(
        self,
        device="Photodiode",
        devicename="Photodiode",
        allow_partial=False,
    ) -> Union[dict, None]:
        """
        Get the data from a device

        Parameters
        ----------
        device : str (default: 'Photodiode')
            The base name of the file holding the data. '.ma' will be appended
            to the name
        devicename : str (default: 'Photodiode')
            The name of the device as set in the config (might be 'pCell', etc)
            This might or might not be the same as the device
        allow_partial: bool (default: False)
            return true even with partial information returned.

        Returns
        -------
        Success : dict
        failure: None

        The results are stored data for the current protocol
        """
        # non threaded
        dirs = self.subDirs(self.protocol)
        index = self._readIndex()
        trx = []
        cmd = []
        sequence_values = None
        if "sequenceParams" in index["."].keys():
            self.sequence = index["."]["sequenceParams"]
        else:
            self.sequence = []
        # building command voltages or currents - get amplitudes to clamp

        reps = ("protocol", "repetitions")
        foundLaser = False
        self.Device_data = []
        self.Device_sample_rate = []
        self.Device_time_base = []
        for i, d in enumerate(dirs):
            fn = Path(d, device + ".ma")
            if not fn.is_file():
                if not allow_partial:
                    print(" acq4_reader.getDeviceData: File not found: ", fn)
                    return None
                else:
                    continue
            try:
                lbr = EM.MetaArray(file=fn)
            except:
                print(" acq4_reader.getDeviceData: Corrupt Metaarray: ", fn)
                return None
            info = lbr[0].infoCopy()
            self.Device_data.append(lbr.view(np.ndarray)[0])
            self.Device_time_base.append(lbr.xvals("Time"))
            sr = info[1]["DAQ"][devicename]["rate"]
            self.Device_sample_rate.append(sr)
        if len(self.Device_data) == 0:
            return None
        self.Device_data = np.array(self.Device_data)
        self.Device_sample_rate = np.array(self.Device_sample_rate)
        self.Device_time_base = np.array(self.Device_time_base)
        return {
            "data": self.Device_data,
            "time_base": self.Device_time_base,
            "sample_rate": self.Device_sample_rate,
        }

    def getLaserBlueCommand(self) -> bool:
        """
        Get the command waveform for the blue laser
        data for the current protocol
        """
        # non threaded
        dirs = self.subDirs(self.protocol)
        index = self._readIndex()
        trx = []
        cmd = []
        sequence_values = None
        if "sequenceParams" in index["."].keys():
            self.sequence = index["."]["sequenceParams"]
        else:
            self.sequence = []
        # building command voltages or currents - get amplitudes to clamp

        reps = ("protocol", "repetitions")
        foundLaser = False
        self.LaserBlue_Raw = []
        self.LaserBlue_pCell = []
        self.LaserBlue_sample_rate = []
        self.LaserBlue_time_base = []
        for i, d in enumerate(dirs):
            fn = Path(d, "Laser-Blue-raw.ma")
            if not fn.is_file():
                # print(" acq4_reader.getLaserBlueCommand: File not found: ", fn)
                return False
            lbr = EM.MetaArray(file=fn)
            info = lbr[0].infoCopy()
            self.LaserBlue_Raw.append(lbr.view(np.ndarray)[0])  # shutter
            self.LaserBlue_time_base.append(lbr.xvals("Time"))
            try:
                self.LaserBlue_pCell.append(lbr.view(np.ndarray)[1])  # pCell
            except:
                # see if have a PockelCell as a seprate thing
                fn = Path(d, "PockelCell.ma")
                if not fn.is_file():
                    print(" acq4_reader.getLaserBlueCommand: File not found: ", fn)
                    self.LaserBlue_pCell.append(None)
                else:
                    pcell = EM.MetaArray(file=fn)
                    self.LaserBlue_pCell.append(pcell.view(np.ndarray)[0])
            try:
                sr = info[1]["DAQ"]["Shutter"]["rate"]
            except:
                raise ValueError(
                    f"Info keys is missing requested DAQ.Shutter.rate:  {info[1]['DAQ'].keys()!s}"
                )
                exit(1)
            self.LaserBlue_sample_rate.append(sr)
        self.LaserBlue_Info = info
        self.LaserBlue_Raw = np.array(self.LaserBlue_Raw[0])
        self.LaserBlue_pCell = np.array(self.LaserBlue_pCell[0])
        self.LaserBlue_sample_rate = np.array(self.LaserBlue_sample_rate)
        self.LaserBlue_time_base = np.array(self.LaserBlue_time_base[0])
        self.LaserBlue_times = self.getLaserBlueTimes()
        return True

    def getPhotodiode(self) -> bool:
        """
        Get the command waveform for the blue laser
        data for the current protocol
        """
        # non threaded
        dirs = self.subDirs(self.protocol)
        index = self._readIndex()
        trx = []
        cmd = []
        sequence_values = None
        if "sequenceParams" in index["."].keys():
            self.sequence = index["."]["sequenceParams"]
        else:
            self.sequence = []
        # building command voltages or currents - get amplitudes to clamp
        reps = ("protocol", "repetitions")
        foundPhotodiode = False
        self.Photodiode = []
        self.Photodiode_time_base = []
        self.Photodiode_sample_rate = []
        self.Photodiode_command = []
        for i, d in enumerate(dirs):
            fn = Path(d, "Photodiode.ma")
            if not fn.is_file():
                # print(" acq4_reader.getPhotodiode: File not found: ", fn)
                return False  # continue
            pdr = EM.MetaArray(file=fn)
            info = pdr[0].infoCopy()
            self.Photodiode.append(pdr.view(np.ndarray)[0])
            self.Photodiode_time_base.append(pdr.xvals("Time"))
            sr = info[1]["DAQ"]["Photodiode"]["rate"]
            self.Photodiode_sample_rate.append(sr)
        self.Photodiode = np.array(self.Photodiode)
        self.Photodiode_sample_rate = np.array(self.Photodiode_sample_rate)
        self.Photodiode_time_base = np.array(self.Photodiode_time_base)
        return True

    def _getWaveGeneratorWidget(
        self, parent_device: str = "", channels: str = None, device: str = None
    ):
        supindex = self._readIndex()
        if channels is not None and device is not None:
            stimuli = supindex["."]["devices"][parent_device][channels][device][
                "waveGeneratorWidget"
            ]["stimuli"]
        else:
            print("parent: ", parent_device)
            print(supindex["."]["devices"][parent_device])
            stimuli = supindex["."]["devices"][parent_device]["waveGeneratorWidget"]["stimuli"]
        times = []
        waveinfo = {}
        waveinfo["start"] = stimuli["Pulse"]["start"]["value"]
        waveinfo["duration"] = stimuli["Pulse"]["length"]["value"]
        waveinfo["type"] = stimuli["Pulse"]["type"]
        return waveinfo

    def getLaserBlueShutter(self) -> dict:
        shutter = self._getWaveGeneratorWidget(
            parent_device="Laser-Blue-raw", channels="channels", device="Shutter"
        )
        return shutter
        # supindex = self._readIndex()
        # stimuli = supindex["."]["devices"]["Laser-Blue-raw"]["channels"]["Shutter"][
        #     "waveGeneratorWidget"
        # ]["stimuli"]
        # times = []
        # shutter = {}
        # shutter["start"] = stimuli["Pulse"]["start"]["value"]
        # shutter["duration"] = stimuli["Pulse"]["length"]["value"]
        # shutter["type"] = stimuli["Pulse"]["type"]
        # return shutter
    
    def getScannerPositions(self, dataname: str = "Laser-Blue-raw.ma") -> bool:
        dirs = self.subDirs(self.protocol)
        self.scanner_positions = np.zeros((len(dirs), 2))
        self.scanner_camera = {}
        self.scanner_info = {}
        self.scanner_sequenceparams = {}
        self.scanner_targets = [[]] * len(dirs)
        self.scanner_spotsize = 0.0
        rep = 0
        target = 0
        n_valid_targets = 0
        supindex = self._readIndex()  # get protocol index (top level, dirType=ProtocolSequence)
        # print('supindex in getScannerPositions: ', supindex, self.protocol)

        if supindex is None or "sequenceParams" not in list(
            supindex["."].keys()
        ):  # should have this key, along with (scanner, targets)
            print("no sequenceParams key in top level protocol directory; in getScannerPosition")
            return False
        try:
            ntargets = len(supindex["."]["sequenceParams"][("Scanner", "targets")])
        except:
            ntargets = 1
            # print('Cannot access (Scanner, targets) in getScannerPosition')
            # return(False)

        pars = {}
        pars["sequence1"] = {}
        pars["sequence2"] = {}
        try:
            reps = supindex["."]["sequenceParams"][
                ("protocol", "repetitions")
            ]  # should have this key also
        except:
            reps = [
                0
            ]  # just fill in one rep. SOme files may be missing the protocol/repetitions entry for some reason
        pars["sequence1"]["index"] = reps
        pars["sequence2"]["index"] = ntargets
        self.scanner_sequenceparams = pars
        for i, d in enumerate(
            dirs
        ):  # now run through the subdirectories : all of dirType 'Protocol'
            index = self._readIndex(
                currdir=Path(self.protocol, Path(d).name)
            )  # subdirectories _nnn or _nnn_mmm or ...
            if index is not None and "Scanner" in index["."].keys():
                self.scanner_positions[i] = index["."]["Scanner"]["position"]
                if ntargets > 1:
                    self.scanner_targets[i] = index["."][("Scanner", "targets")]
                self.scanner_spotsize = index["."]["Scanner"]["spotSize"]
                self.scanner_info[(rep, target)] = {
                    "directory": d,
                    "rep": rep,
                    "pos": self.scanner_positions[i],
                }
                n_valid_targets += 1
            # elif ('Scanner', 'targets') in index['.']:
            #     print('found "(Scanner, targets)" in index')
            #     #print ('scanner targets: ', index['.'][('Scanner', 'targets')])
            #     self.scannerpositions[i] = index['.'][('Scanner', 'targets')]['position']
            #     self.targets[i] = index['.'][('Scanner', 'targets')]
            #     self.spotsize = index['.']['Scanner']['spotSize']
            #     self.scannerinfo[(rep, tar)] = {'directory': d, 'rep': rep, 'pos': self.scannerpositions[i]}
            else:
                return False
            #     print(
            #         f"Scanner information for point {i:d} not found in index: ",
            #         d,
            #         "\n",
            #         index["."].keys(),
            #     )
            #     continue  # protocol is short...
            # #                self.scannerinfo[(rep, tar)] = {'directory': d, 'rep': rep, 'pos': self.scannerpositions[i]}
            if (
                "Camera" in supindex["."]["devices"].keys() and len(self.scanner_camera) == 0
            ):  # read the camera outline
                cindex = self._readIndex(currdir=Path(self.protocol, Path(d).name, "Camera"))
                self.scanner_camera = cindex
            else:
                pass

            target = target + 1
            if target > ntargets:  # wrap for repetitions
                target = 0
                rep = rep + 1
        # print(n_valid_targets)
        # adjust so only valid targets are included
        self.scanner_positions = self.scanner_positions[:n_valid_targets, :]
        # print(self.scanner_positions.shape)
        return True  # indicate protocol is all ok

    def getImage(self, filename: Union[str, Path, None] = None) -> dict:
        """
        getImage
        Returns the image file in the dataname
        Requires full path to the data
        Can also read a video (.ma) file, returning the stack
        """
        assert filename is not None
        filename = Path(filename)
        if filename.suffix in [".tif", ".tiff"]:
            self.imageData = tf.imread(str(filename))
        elif filename.suffix in [".ma"]:
            self.imageData = EM.MetaArray(file=filename)
        d = str(filename.name)
        self.Image_filename = d
        cindex = self._readIndex(Path(filename.parent))

        if "userTransform" in list(cindex[d].keys()) and cindex[d]["userTransform"]["pos"] != (
            0.0,
            0.0,
        ):
            z = np.vstack(cindex[d]["userTransform"]["pos"] + cindex[d]["transform"]["pos"]).ravel()
            self.Image_pos = ((z[0] + z[2]), (z[1] + z[3]), z[4])
        else:
            self.Image_pos = cindex[d]["transform"]["pos"]

        self.Image_scale = cindex[d]["transform"]["scale"]
        self.Image_region = cindex[d]["region"]
        self.Image_binning = cindex[d]["binning"]
        return self.imageData

    def getAverageScannerImages(
        self,
        dataname: str = "Camera/frames.ma",
        mode: str = "average",
        firstonly: bool = False,
        subtractFlag: bool = False,
        limit: Union[int, None] = None,
        filter: bool = True,
    ):
        """
        Average (or max or std) the images across the scanner camera files
        the images are collected into a stack prior to any operation

        Parameters
        ----------
        dataname : str (default: 'Camera/frames.ma')
            Name of the camera data file (metaarray format)

        mode : str (default: 'average')
            Operation to do on the collected images
            average : compute the average image
            max : compute the max projection across the stack
            std : compute the standard deviation across the stack

        limit : maximum # of images in stack to combine (starting with first)

        subtractFlag : boolean (default: False)
                subtract first frame from second when there are pairs of frames

        firstonly : boolean (default: False)
                return the first image only

        filter : boolean (default: True)
                Not implemented


        Returns
        -------
            a single image frame that is the result of the specified operation

        """
        assert mode in ["average", "max", "std"]
        print("average scanner images")
        dirs = self.subDirs(self.protocol)

        rep = 0
        tar = 0
        supindex = self._readIndex()
        ntargets = len(supindex["."]["sequenceParams"][("Scanner", "targets")])
        pars = {}
        pars["sequence1"] = {}
        pars["sequence2"] = {}
        try:
            reps = supindex["."]["sequenceParams"][("protocol", "repetitions")]
        except:
            reps = [0]
        pars["sequence1"]["index"] = reps
        pars["sequence2"]["index"] = ntargets
        scannerImages = []
        self.sequenceparams = pars
        self.scannerinfo = {}
        if limit is None:
            nmax = len(dirs)
        else:
            nmax = min(limit, len(dirs))
        refimage = None
        for i, d in enumerate(dirs):
            if i == nmax:  # check limit here first
                break
            index = self._readIndex(d)
            imageframe = EM.MetaArray(file=Path(d, dataname))
            cindex = self._readIndex(Path(d, "Camera"))
            frsize = cindex["frames.ma"]["region"]
            binning = cindex["frames.ma"]["binning"]
            # print ('image shape: ', imageframe.shape)
            if imageframe.ndim == 3 and imageframe.shape[0] > 1 and not subtractFlag:
                imageframed = imageframe[1]
            if imageframe.ndim == 3 and imageframe.shape[0] > 1 and subtractFlag:
                if refimage is None:
                    refiamge = imageframe[0]
                else:
                    refimage += imageframe[0]
                imageframed = imageframe[1]  # take difference in images

            elif imageframe.ndim == 3 and imageframe.shape[0] == 1:
                imageframed = imageframe[0]
            imageframed = imageframed.view(np.ndarray)
            if filter:
                imageframed = SND.gaussian_filter(imageframed, 3)
            if firstonly:
                return imageframed

            if i == 0:
                scannerImages = np.zeros(
                    (nmax, int(frsize[2] / binning[0]), int(frsize[3] / binning[1]))
                )
            # import matplotlib.pyplot as mpl
            # mpl.imshow(imageframed)
            # mpl.show()
            # if i > 3:
            #     exit()
            scannerImages[i] = imageframed
        if refimage is None:
            refimage = np.zeros_like(imageframed)
        resultframe = np.zeros((scannerImages.shape[1], scannerImages.shape[2]))
        # simple maximum projection
        print("mode: %s" % mode)
        print("scanner images: ", scannerImages.shape)
        nimages = scannerImages.shape[0]
        refimage = refimage / nimages  # get average
        print("binning: ", binning)
        for i in range(nimages):
            scannerImages[i] -= refimage
        if mode == "max":
            for i in range(scannerImages.shape[0]):
                resultframe = np.maximum(resultframe, scannerImages[i])
        elif mode == "average":
            resultframe = np.mean(scannerImages, axis=0)
        elif mode == "std":
            resultframe = np.std(scannerImages, axis=0)
        return resultframe.T  # must transpose to match other data...

    def plotClampData(self, all=True):
        import matplotlib.pyplot as mpl

        f, ax = mpl.subplots(2)
        if all:
            for i in range(len(self.data_array)):
                ax[0].plot(self.time_base, self.data_array[i])
                ax[1].plot(self.time_base, self.cmd_wave[i])
        else:
            ax[0].plot(self.time_base, np.array(self.data_array).mean(axis=0))
        mpl.show()


def one_test():
    import tools.boundrect as BR

    BRI = BR.BoundRect()
    #    a.setProtocol('/Users/pbmanis/Documents/data/MRK_Pyramidal/2018.01.26_000/slice_000/cell_000/CCIV_1nA_max_000/')
    # this won't work in the wild, need appropriate data for testing.
    import matplotlib

    # matplotlib.use('')
    import matplotlib.pyplot as mpl

    # test on a big file
    a = acq4_reader()
    # cell = '/Users/pbmanis/Documents/data/mrk/2017.09.12_000/slice_000/cell_001'
    cell = "/Users/pbmanis/Desktop/Data/Glutamate_LSPS_DCN/2019.08.06_000/slice_002/cell_000"
    if not Path(cell).is_dir():
        raise ValueError
    datasets = Path(cell).glob("*")
    imageplotted = False
    imagetimes = []
    imagename = []
    maptimes = []
    mapname = []
    supindex = a.readDirIndex(currdir=cell)
    print("supindex: ", list(supindex.keys()))
    for k in list(supindex.keys()):
        if k.startswith("image_"):
            # print("Found Image: ", k)
            imagetimes.append(supindex[k]["__timestamp__"])
            imagename.append(k)
        if k.startswith("Map_") or k.startswith("LSPS_" or k.startswith("LED_")):
            maptimes.append(supindex[k]["__timestamp__"])
            mapname.append(k)
    print("maptimes: ", maptimes)
    print("imagetimes: ", imagetimes)
    maptoimage = {}
    for im, m in enumerate(maptimes):
        u = np.argmin(maptimes[im] - np.array(imagetimes))
        maptoimage[mapname[im]] = imagename[u]

    print("map to image: ", maptoimage)
    print("datasets: ", datasets)

    for i, d in enumerate(datasets):
        d = str(d)
        print("d: ", d)
        pa, da = os.path.split(d)
        print("da: ", da)
        if "Map" not in da and "LSPS" not in da:
            continue
        print("d: ", d)
        a.setProtocol(os.path.join(cell, d))
        #    a.setProtocol('/Volumes/Pegasus/ManisLab_Data3/Kasten, Michael/2017.11.20_000/slice_000/cell_000/CCIV_4nA_max_000')
        if not a.getScannerPositions():
            continue

        print("scanner transform: ", a.scannerCamera["frames.ma"]["transform"])
        pos = a.scannerCamera["frames.ma"]["transform"]["pos"]
        scale = a.scannerCamera["frames.ma"]["transform"]["scale"]
        region = a.scannerCamera["frames.ma"]["region"]
        binning = a.scannerCamera["frames.ma"]["binning"]
        print("bining: ", binning)
        if a.spotsize is not None:
            print("Spot Size: {0:0.3f} microns".format(a.spotsize * 1e6))
        else:
            a.spotsize = 50.0

        camerabox = [
            [pos[0] + scale[0] * region[0], pos[1] + scale[1] * region[1]],
            [pos[0] + scale[0] * region[0], pos[1] + scale[1] * region[3]],
            [pos[0] + scale[0] * region[2], pos[1] + scale[1] * region[3]],
            [pos[0] + scale[0] * region[2], pos[1] + scale[1] * region[1]],
            [pos[0] + scale[0] * region[0], pos[1] + scale[1] * region[1]],
        ]
        scannerbox = BRI.getRectangle(a.scannerpositions)
        print("scannerbox: ", scannerbox)
        print("scannerbox shape: ", scannerbox.shape)
        fp = np.array([scannerbox[0][0], scannerbox[1][1]]).reshape(2, 1)
        print("rehaped scanner box: ", fp.shape)
        scannerbox = np.append(scannerbox, fp, axis=1)
        print("new scannerbox: ", scannerbox)

        boxw = np.swapaxes(np.array(camerabox), 0, 1)
        print("camera box: ", boxw)
        scboxw = np.array(scannerbox)
        print("scanner box: ", scboxw)
        mpl.plot(boxw[0, :], boxw[1, :], linewidth=1.5)
        avgfr = a.getAverageScannerImages(firstonly=True, mode="average")
        if not imageplotted:
            imgd = a.getImage(os.path.join(cell, "image_001.tif"))
            # mpl.imshow(np.flipud(np.rot90(avgfr), aspect='equal', extent=[np.min(boxw[0]), np.max(boxw[0]), np.min(boxw[1]), np.max(boxw[1])])
            mpl.imshow(
                imgd,
                aspect="equal",
                cmap="gist_gray",
                extent=[
                    np.min(boxw[0]),
                    np.max(boxw[0]),
                    np.min(boxw[1]),
                    np.max(boxw[1]),
                ],
            )
            imageplotted = True
        mpl.plot(
            a.scannerpositions[:, 0],
            a.scannerpositions[:, 1],
            "ro",
            alpha=0.2,
            markeredgecolor="w",
        )
        mpl.plot(boxw[0, :], boxw[1, :], "g-", linewidth=5)
        mpl.plot(scboxw[0, :], scboxw[1, :], linewidth=1.5, label=d.replace(r"_", r"\_"))

    # a.getData()
    # a.plotClampData(all=True)
    # print a.clampInfo
    # print a.traces[0]
    pos = mpl.ginput(-1, show_clicks=True)
    print("pos: ", pos)

    mpl.legend()
    mpl.show()


if __name__ == "__main__":
    pass

