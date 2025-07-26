#!/usr/bin/env python3
__author__ = "pbmanis"
"""
dataSummary: This script reads all of the data files in a given directory, and prints out top level information
including notes, protocols run (and whether or not they are complete), and image files associated with a cell.
Currently, this routine makes assumptions about the layout as a hierarchical structure [days, slices, cells, protocols]
and does not print out information if there are no successful protocols run.
June, 2014, Paul B. Manis.

Mar 2015:
added argparse to expand command line options rather than editing file. 
The following options are recognized:
begin (b) (define start date; end is set to current date) default: 1/1/1970
end (e)(define end date: start date set to 1/1/1970; end is set to the end date) default: "today"
mode =  full (f) : do a full investigation of the data files. Makes processing very slow. (reports incomplete protocols)
        partial (p) : do a partial investiagion of protocols: is there anything in every protocol directory? (reports incomplete protocols) - slow
        quick (q) : do a quick scan : does not run through protocols to find incomplete protocols. Default (over full and partial)
debug (d) : debug monitoring of progress
output (o) : define output file (tab delimited file for import to other programs)


Mar 2018: Version 2
Uses acq4_reader and is independent of acq4 itself.

July 2018: 
Major surgery - to output Pandas (pickled) files as well. UGH.

Nov 2022:
Change writing to generate a  pandas database and a corresponding excel file. 

----------------
usage: dataSummary [-h]
                   [-f OUTPUTFILENAME] [-r] [-u] [-D] [--daylist DAYLIST]
                   [-d DAY] [-a AFTER] [-b BEFORE] [--dry-run] [-v]
                   [--no-inspect] [--depth {days,slices,cells,protocols,all}]
                   [-A] [-p]
                   basedir

    Generate Data Summaries from acq4 datasets

    positional arguments:
      basedir               Base Directory

    optional arguments:
      -h, --help            show this help message and exit
      -f OUTPUTFILENAME, --filename OUTPUTFILENAME
                            Specify output file name (including full path)
      -r, --read            just read the summary table
      -u, --update          If writing, force update for days already in list
      -D, --deep            perform deep inspection (very slow)
      --daylist DAYLIST     Specify daylistfile
      -d DAY, --day DAY     day for analysis
      -a AFTER, --after AFTER
                            only analyze dates on or after a date
      -b BEFORE, --before BEFORE
                            only analyze dates on or before a date
      --dry-run             Do a dry run, reporting only directories
      -v, --verbose         Verbose print out during run
      --no-inspect          Do not inspect protocols, only report directories
      --depth {days,slices,cells,protocols,all}
                            Specify depth for --dry-run
      -A, --append          update new/missing entries to specified output file
      -p, --pairs           handle pairs

Example:
    python ephysanalysis/dataSummary.py   /Volumes/Pegasus/ManisLab_Data3/Kasten_Michael/NF107ai32Het/ -a 2017.04.16 -o pandas -f NF107_after_2018.04.16 -w --depth all --dry-run

Note: the -w is essential for the process to actually occur...

"""
import argparse
import datetime
import gc
import os
import os.path
import re
import sys
import textwrap
from collections import OrderedDict

# import pandas_compat # for StringIO - separate package - but only for pandas < 0.24 or so
from io import StringIO
from pathlib import Path
from typing import Union
from ephys.tools.parse_ages import ISO8601_age
import dateutil.parser as DUP
import MetaArray
import numpy as np
import pandas as pd
from pylibrary.tools import cprint
from ..datareaders import acq4_reader
from . import parse_layers


CP = cprint.cprint


def ansi_colors(color):
    colors = {
        "black": "\u001b[30m",
        "red": "\u001b[31m",
        "green": "\u001b[32m",
        "yellow": "\u001b[33m",
        "blue": "\u001b[34m",
        "magenta": "\u001b[35m",
        "cyan": "\u001b[36m",
        "white": "\u001b[37m",
        "reset": "\u001b[0m",
    }
    return colors[color]


class Printer:
    """Print things to stdout on one line dynamically"""

    def __init__(self, data, color="white"):

        sys.stdout.write("\r\u001b[2K%s" % ansi_colors(color) + data.__str__())
        sys.stdout.flush()


slsp = "   "  # slice leading indent
clsp = "        "  # cell within slice leading indent
prsp = "             "  # protocol within cell leading indent


class DataSummary:
    def __init__(
        self,
        basedir=None,
        outputMode="pandas",
        outputFile=None,
        daylistfile=None,
        after=None,
        before=None,
        day=None,
        dryrun=False,
        depth="all",
        inspect=True,
        subdirs=False,
        deep=False,
        append=False,
        verbose=False,
        update=False,
        pairflag=False,
        device="MultiClamp1.ma",
        excludedirs: list = [],
    ):
        """
        Note that the init is just setup - you have to call getDay with the object to do anything

        Parameters
        ----------
        basedir : str (required)
            base directory to be summarized

        Obselete:
        #outputMode : str (default: 'pandas')
        #    How to write the output. Current options are 'terminal', which writes to the terminal, and
        #    'pandas', which will create a pandas dataframe and pickle it to a file

        outputFile : str (default: None)
            File name and path for an output file (pandas and excel files are written)

        daylistfile : str (default: None)
            A filename that stores a list of days that sould be processed

        after : str (default: Jan 1, 1970)
            A starting date for the summary - files before this date will not be processed
            A string in a format that can be parsed by dateutil.parser.parse
            The following will work:
            without string quotes:
            2018.01.01
            2018.1.1
            2018/7/6
            2018-7-6
            With quotes:
            'Jan 1 2017'
            'Jan 1, 2017'
            etc..

        before : str (default 2266)
            An ending date for the summary. Files after this date will not be processed.
            The format is the same as for the starting date

        day : str(no default)
            The day to do a summary on.

        dryrun : bool (default False)
            Causes the output to be limited and protocols are not fully investigated

        depth : str (default: 'all')
            Causes output with dry-run to go to the depth specified
            options are day, slice, cell, prototol

        inspect : bool (default: True)
            perform inspection to find complete versus incomplete protocols

        subdirs: bool (default: False)
            look also for files in subdirectories that are not acq4 data files.

        deep: bool (default: False)
            do a "deep" inspection of the data files, actually reading the .ma files
                 to confirm the data existence. This is slow...
        append: bool (default: False)
            Add new entries to the output file if they are not in the
                 database by appending them at the end
        verbose: bool (default: False)
            Provide extra print out during analysis for debugging.
        excludedir: list(default:[])
            A list of the names of the directories to exclude from the summary

        Note that if neither before or after are specified, the entire directory is read.
        """

        self.setups()  # build some regex and wrappers
        # gather input parameters
        if basedir is not None:
            self.basedir = Path(basedir)
        else:
            self.basedir = None
        self.outputMode = "pandas"  # outputMode  # terminal, tabfile, pandas
        self.outFilename = outputFile
        self.daylistfile = daylistfile
        self.dryrun = dryrun
        self.after = after
        self.before = before
        self.day = day
        self.depth = depth
        self.subdirs = subdirs
        self.verbose = verbose
        self.update = update
        self.deep_check = deep
        self.pairflag = pairflag
        self.device = device
        self.append = append
        self.all_dataset_protocols = []  # a list of ALL protocols found in the dataset
        self.excludedirs = excludedirs
        self.daylist = None
        self.index = 0
        # flags - for debugging and verbosity
        self.reportIncompleteProtocols = True  # do include incomplete protocol runs in print
        self.InvestigateProtocols = inspect  # set True to check out the protocols in detail
        if self.dryrun:
            self.reportIncompleteProtocols = False  # do include incomplete protocol runs in print
            self.InvestigateProtocols = False  # set True to check out the protocols in detail
        self.panda_string = ""
        self.cell_id = ""
        # column definitions - may need to adjust if change data that is pasted into the output
        self.day_defs = [
            "date",
            "description",
            "notes",
            "species",
            "strain",
            "genotype",
            "reporters",
            "age",
            "animal identifier",
            "sex",
            "weight",
            "reporters",
            "solution",
            "internal",
            "temperature",
            "important",
            "expUnit",
        ]
        self.slice_defs = [
            "slice_slice",
            "slice_notes",
            "slice_location",
            "slice_orientation",
            "slice_mosaic",
            "slice_important",
        ]
        self.cell_defs = [
            "cell_cell",
            "cell_notes",
            "cell_type",
            "cell_location",
            "cell_layer",
            "cell_expression",
            "cell_mosaic",
            "cell_important",
            "cell_id",
        ]
        self.data_defs = [
            "data_incomplete",
            "data_complete",
            "data_images",
            "annotated",
            "data_directory",
        ]

        # expected keys in various structural levels: days, slices, cells
        self.day_keys = [
            "description",
            "notes",
            "species",
            "strain",
            "genotype",
            "age",
            "animal identifier",
            "sex",
            "weight",
            "solution",
            "__timestamp__",
            "internal",
            "temperature",
            "expUnit",
            "dirType",
            "important",
            "time",
        ]
        self.slice_keys = [
            "notes",
            "location",
            "orientation",
            "mosaic",
            "important",
            "__timestamp__",
        ]
        self.cell_keys = [
            "notes",
            "type",
            "location",
            "mosaic",
            "important",
            "__timestamp__"
        ]
        self.data_dkeys = [
            "incomplete",
            "complete",
            "data_images",
            "annotated",
            "directory",
        ]

        self.day_template = OrderedDict(
            [
                ("species", "{:>s}"),
                ("strain", "{:>s}"),
                ("genotype", "{:>12s}"),
                ("reporters", "{:>12s}"),
                ("age", "{:>5s}"),
                ("animal_identifier", "{:>8s}"),
                ("sex", "{:>2s}"),
                ("weight", "{:>5s}"),
                ("solution", "{:>s}"),
                ("internal", "{:>s}"),
                ("temperature", "{:>5s}"),
                ("important", "{:>s}"),
                ("elapsedtime", "{:>8.2f}"),
                ("expUnit", "{:>s}"),
            ]
        )
        self.slice_template = OrderedDict(
            [
                ("type", "{:>s}"),
                ("location", "{:>12s}"),
                ("orientation", "{:>5s}"),
                ("mosaic", "{:>s}"),
                ("important", "{:>s}"),
            ]
        )
        self.cell_template = OrderedDict(
            [
                ("type", "{:>s}"),
                ("location", "{:>12s}"),
                ("layer", "{:>10s}"),
                ("expression", "{:2s}"),
                ("mosaic", "{:>s}"),
                ("important", "{:>s}"),
            ]
        )
        self.data_template = OrderedDict(
            [
                ("incomplete", "{0:s}"),
                ("complete", "{1:s}"),
                ("images", "{2:s}"),
                ("annotated", "{3:s}"),
                ("directory", "{4:s}"),
            ]
        )
        self.coldefs = "Date \tDescription \tNotes \tGenotype \tAge \tAnimal_Identifier\tSex \tWeight \tTemp \tElapsed T \tSlice \tSlice Notes \t"
        self.coldefs += "Cell \t Cell Notes \t \tProtocols \tImages \t"

        self.AR = acq4_reader.acq4_reader()  # instance of the reader
        self.AR.setDataName(device)
        # if self.outputMode == "tabfile":
        #     print("Tabfile output: Writing to {:<s}".format(self.outFilename))
        #     with open(self.outFilename, "w") as fh:  # write new file
        #         fh.write(self.basedir + "\n")
        #         fh.write(self.coldefs + "\n")

        # elif (
        #     self.outputMode == "pandas"
        # ):  # save output as a pandas data structure, pickled
        #     print("Pandas output: will write to {:<s}".format(self.outFilename))
        # else:
        #     pass  # just print if terminal

        # figure out the before/after limits
        # read the before and after day strings, parse them and set the min and maxdays
        if self.after is None:
            mindayx = (1970, 1, 1)  # all dates after this - unix start date convention
        else:
            try:
                dt = DUP.parse(self.after)
                mindayx = (dt.year, dt.month, dt.day)
            except:
                raise ValueError("Date for AFTER cannot be parsed : {0:s}".format(self.after))
        if self.before is None:
            maxdayx = (
                2266,
                1,
                1,
            )  # far enough into the future for you? Maybe when the Enterprise started it's journey?
        else:
            try:
                dt = DUP.parse(self.before)
                maxdayx = (dt.year, dt.month, dt.day)
            except:
                raise ValueError("Date for BEFORE cannot be parsed : {0:s}".format(self.before))
        if self.day is not None:
            dt = DUP.parse(self.day)
            mindayx = (dt.year, dt.month, dt.day)
            maxdayx = (dt.year, dt.month, dt.day)

        print(
            "after, before, mindayx, maxdayx: ",
            self.after,
            self.before,
            mindayx,
            maxdayx,
        )
        print("daylistfile: ", self.daylistfile)
        if self.daylistfile is None:  # get from command line
            self.minday = mindayx[0] * 1e4 + mindayx[1] * 1e2 + mindayx[2]
            self.maxday = maxdayx[0] * 1e4 + maxdayx[1] * 1e2 + maxdayx[2]
            print("Min, max day: ", self.minday, self.maxday)
        else:
            self.daylist = []
            with open(self.daylistfile, "r") as f:
                for line in f:
                    if line[0] != "#":
                        self.daylist.append(line[0:10])

    def setups(self):
        self.tw = {}  # for notes
        self.tw["day"] = textwrap.TextWrapper(
            initial_indent="", subsequent_indent=" " * 2
        )  # used to say "initial_indent ="Description: ""
        self.tw["slice"] = textwrap.TextWrapper(initial_indent="", subsequent_indent=" " * 2)
        self.tw["cell"] = textwrap.TextWrapper(initial_indent="", subsequent_indent=" " * 2)
        self.tw["pair"] = textwrap.TextWrapper(initial_indent="", subsequent_indent=" " * 2)

        self.twd = {}  # for description
        self.twd["day"] = textwrap.TextWrapper(
            initial_indent="", subsequent_indent=" " * 2
        )  # used to ays initial_indent ="Notes: ""
        self.twd["slice"] = textwrap.TextWrapper(initial_indent="", subsequent_indent=" " * 2)
        self.twd["cell"] = textwrap.TextWrapper(initial_indent="", subsequent_indent=" " * 2)
        self.twd["pair"] = textwrap.TextWrapper(initial_indent="", subsequent_indent=" " * 2)

        self.img_re = re.compile(
            r"^[Ii]mage_(\d{3,3}).tif"
        )  # make case insensitive - for some reason in Xuying's data
        self.s2p_re = re.compile(r"^2pStack_(\d{3,3}).ma")
        self.i2p_re = re.compile(r"^2pImage_(\d{3,3}).ma")
        self.video_re = re.compile(r"^[Vv]ideo_(\d{3,3}).ma")

        self.daytype = re.compile(r"(\d{4,4}).(\d{2,2}).(\d{2,2})_(\d{3,3})")

    #        daytype = re.compile(r"(2011).(06).(08)_(\d{3,3})")  # specify a day

    def getDay(self, allfiles):
        """
        getDay is the entry point for scanning through all the data files in a given directory,
        returning information about those within the date range, with details as specified by the options

        Parameters
        ----------
        allfiles : list of all of the files in the directory

        Returns
        -------
        Nothing

        The result is stored in the class variable self.day_index

        """
        if self.append:
            print("\nreading for append: ", self.outFilename)
            self.pddata = pd.read_pickle(self.outFilename)  # get the current file
        # print('alldays: ', allfiles)
        self.pstring = ""
        days = []
        # print('allfiles: ', allfiles)
        for file_n, thisfile in enumerate(allfiles):
            if not str(thisfile.name).startswith(
                "20"
            ):  # skip directories that are not data directories at this level
                continue
            str_file = str(thisfile.name)
            if str_file.endswith(".sql") or str_file.startswith("corrupted"):
                continue
            if str_file in [".DS_Store", ".index"]:
                continue
            m = self.daytype.search(str_file)
            if m is None:
                print("no match in daytype : ", thisfile)
                continue  # no match
            if len(m.groups()) >= 3:  # perfect match
                idl = [int(d) for d in m.groups()]
                id = idl[0] * 1e4 + idl[1] * 1e2 + idl[2]

                if self.daylist is None:
                    # print('id, self.minday, self.maxday: ', id, self.minday, self.maxday)
                    if id >= self.minday and id <= self.maxday:
                        days.append(thisfile)  # was [0:10]
                else:
                    if str_file[0:10] in self.daylist:
                        days.append(str_file)
        # print('days: ', days)
        if self.verbose:
            print("Days reported: ", days)
            if self.append:
                print("Days in pandas frame: ", self.pddata["date"].tolist())

        for nd, day in enumerate(days):
            # if nd > 10:
            #     exit()
            if self.append and (day in self.pddata["date"].tolist()):
                if not self.update:
                    print("\nAppend mode: day already in list: {0:s}".format(day))
                    continue  # skip
                else:
                    print(self.pddata["date"])
                    print(day)
                    k = self.pddata.index[self.pddata["date"] == day]
                    print(k)
                    self.pddata = self.pddata.drop(index=k)  # remove the day and update it
            else:
                pass
                # print('Day to do: ', day)
            if self.verbose:
                self.pstring = "Processing day[%3d/%3d]: %s " % (nd, len(days), day)
            self.AR.setProtocol(Path(self.basedir, day))
            dir_list = Path(self.basedir).parts
            day_list = Path(day).parts
            n = len(dir_list)
            day_list = day_list[n:]
            # print("\nday_list: ", day_list)
            # print("\nbasedir: ", self.basedir, day)

            self.day_index = self.AR.readDirIndex(Path(self.basedir, day))
            if self.day_index is None:
                print("\nDay {0:s} is not managed (no .index file found)".format(day))
                self.day_index = {}
                continue
            self.day_index = self.day_index["."]
            # print('ind: ', ind)
            #  self.day_index = self.AR.readDirIndex(ind)
            # print('day index: ', self.day_index, "day: ", day)
            self.day_index["date"] = str(Path(*day_list)).strip()
            # print("Day index: ",self.day_index)
            # CP("y", f"day index animal id:  {self.day_index['animal identifier']:s}")
            # now add the rest of the index information to the daystring
            # print(self.day_index.keys())
            # print(self.day_defs)
       
            for k in self.day_defs:  # for all the keys
                if k not in self.day_index.keys():
                    # print('\nadded: ', k)
                    self.day_index[k] = ""  # 'missing'
                # else:
                #     print(' ? k in day index: ', k)
                if isinstance(self.day_index[k], bool):
                    self.day_index[k] = str(self.day_index[k])
                if k in [
                    "sex"
                ]:  # make uppercase or maybe "U" or empty. Anything else is not valid for mice.
                    self.day_index[k] = self.day_index[k].upper()
                    if self.day_index[k] not in ["M", "F", "m", "f", None, "", " ", "U"]:
                        print("? sex: <" + self.day_index[k] + ">")
                        exit()
                if k in ["age"]:  # match ISO8601 date standards
                    self.day_index[k] = ISO8601_age(agestr=self.day_index[k])
                if k in ["weight"]:
                    wt = self.day_index[k]
                    if not isinstance(wt, str):
                        self.day_index[k] = f"{wt:d}g"
                    else:
                        if len(wt) > 0 and wt[-1] not in ["g", "G"]:
                            self.day_index[k] = wt + "g"

                self.day_index[k].replace("\n", " ")
                if len(self.day_index[k]) == 0:
                    self.day_index[k] = " "
            for k in self.day_defs:
                print("{:>32s} : {:<40s}".format(k, self.day_index[k]))

            self._doSlices(day)  # next level
            # os.closerange(8, 65535)  # close all files in each iteration
            gc.collect()

    def _doSlices(self, day):
        """
        process all of the slices for a given day

        Parameters
        ----------

        day : str (no default)
            Path to the directory holding the data for the day

        Returns
        -------
        Nothing

        The result is stored in teh class variable slice_index

        """
        allfiles = Path(self.basedir, day).glob("*")
        slicetype = re.compile(r"(slice\_)(\d{3,3})")
        slices = []
        for thisfile in list(allfiles):
            # print(slsp + 'slicefile: ', thisfile)
            thisfile = str(thisfile)
            m = slicetype.search(str(thisfile))
            if m is None:
                # print(slsp + 'm is none')
                continue
            if len(m.groups()) == 2:
                slices.append("".join(m.groups(2)))  # Path(thisfile).parts[-1])
        for slicen in slices:
            slicen = str(slicen)
            self.sstring = str(day) + " " + self.pstring + " %s" % slicen
            Printer(self.sstring)
            self.slicestring = "%s\t" % (slicen)
            self.slice_index = self.AR.readDirIndex(Path(self.basedir, day, slicen))
            # print(slsp + 'slice index: ', self.slice_index)
            if self.slice_index is None:  # directory is not managed and probably empty
                # print(slsp + 'Slice {0:s} is not managed (no .index file found)'.format(slicen))
                self.slice_index = {}
                continue
            self.slice_index = self.slice_index["."]
            self.slice_index["slice"] = slicen
            if "mosaic" not in self.slice_index.keys():
                self.slice_index["mosaic"] = " "
            for k in self.slice_defs:
                if k == "slice_mosaic":
                    slicepath = Path(self.basedir, day, slicen)
                    mosaics = list(slicepath.glob("*.mosaic"))  # look for mosaic files
                    print ("\n****  SLICE :", slicepath, " MOSAICS: ", mosaics)
                    if len(mosaics) > 0:
                        self.slice_index["mosaic"] = ", ".join(m.name for m in mosaics)
                    else:
                        self.slice_index["mosaic"] = " "
                else:
                    ks = k.replace("slice_", "")
                    if ks not in self.slice_index.keys():
                        self.slice_index[ks] = " "
                    if isinstance(self.slice_index[ks], bool):
                        self.slice_index[ks] = str(self.slice_index[ks])
                    if len(self.slice_index[ks]) == 0:
                        self.slice_index[ks] = " "
                    self.slice_index[ks].replace("\n", " ")
            self._doCells(Path(self.basedir, day, slicen))
            gc.collect()

    def _doCells(self, thisslice, pair=False):
        """
        process all of the cells from a slice
        This will usually be called from dataSummary.day()

        Parameters
        ----------
        thisslice : str
            Path to the slice directory

        Returns
        -------
        Nothing

        The result is stored in the class variable cell_index

        """
        # print(clsp + 'docells')
        allfiles = Path(thisslice).glob("*")
        # print('in doCells, allfiles: ', list(allfiles))
        if not self.pairflag:
            cell_re = re.compile(r"(cell_)(\d{3,3})")
        else:
            cell_re = re.compile(r"(pair_)(\d{3,3})")
        cells = []
        for thisfile in allfiles:
            thisfile = str(thisfile)
            m = cell_re.search(thisfile)
            # print('docells: thisfile: ', thisfile)
            if m is None:
                # print('docells: m is None')
                continue
            if len(m.groups()) == 2:
                # print('thisfile: ', thisfile)
                cells.append("".join(m.groups(2)))
        for cell in cells:
            # print("\n", clsp + "cell: ", cell)
            self.cstring = self.sstring + " %s" % cell
            sparts = Path(self.sstring.split(" ")[0]).parts
            Printer(self.cstring)
            bparts = Path(self.basedir).parts
            nbase = len(bparts)  # length of path components up to the actual date
            self.cell_id = str(
                Path(
                    str(Path(*sparts[nbase:])),
                    self.cstring.split(" ")[-2],
                    self.cstring.split(" ")[-1],
                )
            )

            try:
                self.cell_index = self.AR.readDirIndex(Path(thisslice, cell))[
                    "."
                ]  # possible that .index file is missing, so we cannot read
            except:
                self.cell_index = {}  # unreadable...
                continue
            if self.cell_index is None:
                self.cell_index = {}  # directory is not managed, so skip
                continue
            self.cell_index["cell"] = cell
            self.cell_index["id"] = self.cell_id
            if "mosaic" not in self.cell_index.keys():
                self.cell_index["mosaic"] = " "
            if "notes" not in self.cell_index.keys():
                self.cell_index["notes"] = ""
            # probably better to set this manually in acq4. Too many permutations to consider.
            # if "cell_layer" not in self.cell_index.keys() or len(self.cell_index["cell_layer"]) == 0:
            #     self.cell_index["cell_layer"] = self.regularize_layer(self.cell_index["notes"])

            for k in self.cell_defs:
                if k == "slice_mosaic":
                    cellpath = Path(thisslice, cell)
                    mosaics = list(
                        cellpath.glob("*.mosaic")
                    )  # look for mosaic files at the cell level
                    if len(mosaics) > 0:
                        self.cell_index["mosaic"] = ", ".join(m.name for m in mosaics)
                    else:
                        self.cell_index["mosaic"] = " "
                    continue
                ks = k.replace("cell_", "")
                if ks not in list(self.cell_index.keys()):
                    self.cell_index[ks] = " "
                if isinstance(self.cell_index[ks], bool):
                    self.cell_index[ks] = str(self.cell_index[ks])
                self.cell_index[ks].replace("\n", " ")
                if len(self.cell_index[ks]) == 0:
                    self.cell_index[ks] = " "
            #            print('\n cell index: ', self.cell_index)
            self._doProtocols(Path(thisslice, cell))
            gc.collect()

    def _doProtocols(self, thiscell):
        """
        process all of the protocols for a given cell
        Parameters
        ----------
        thiscell : str
            Path to the cell directory, where the data from the protocols (protocol directories) are stored.

        Returns
        -------
        Nothing

        The results are stored in a class variable "ostring", which is a dict of protocols and summary of images and videos
        """
        allfiles = thiscell.glob("*")
        protocols = []
        nonprotocols = []
        anyprotocols = False
        images = []  # tiff
        stacks2p = []
        images2p = []
        videos = []

        endmatch = re.compile(r"[\_(\d{3,3})]$")  # look for _lmn at end of directory name
        for thisfile in allfiles:
            if Path(thiscell, thisfile).is_dir():
                protocols.append(thisfile)
            else:
                nonprotocols.append(thisfile)

        self.incompleteprotocolstring = ""
        self.allprotocols = []
        self.incompleteprotocols = []
        self.completeprotocols = []
        self.compprotstring = ""
        #        if self.InvestigateProtocols is True:
        # self.thiscell_summarystring = 'NaN\t'*6
        if self.verbose:
            print("\n" + prsp + "Investigating Protocols")
        for np, protocol in enumerate(protocols):  # all protocols on the cell
            mp = Path(protocol).parts
            protocol = mp[-1]
            protocol = str(protocol)
            if protocol not in self.all_dataset_protocols:
                self.all_dataset_protocols.append(protocol)
            if protocol.startswith("Patch"):
                continue

            Printer(self.cstring + " Prot[%2d/%2d]: %s" % (np, len(protocols), protocol))
            self.allprotocols += protocol + ", "
            protocolpath = Path(thiscell, protocol)
            dirs = self.AR.subDirs(
                protocolpath
            )  # get all sequence entries (directories) under the protocol
            modes = []
            info = self.AR.readDirIndex(protocolpath)  # top level info dict
            if info is None:
                print(f"Protocol is not managed (no .index file found): {str(protocolpath):s}")
                continue
            info = info["."]
            if "devices" not in info.keys():  # just safety...
                continue
            devices = info["devices"].keys()
            clampDevices = []
            for d in devices:
                if d in self.AR.clampdevices:
                    clampDevices.append(d)
            if len(clampDevices) == 0:
                continue  # ignore protocol
            mainDevice = clampDevices[0]
            modes = self.getClampDeviceMode(info, mainDevice, modes)
            #            print('modes: ', modes)
            nexpected = len(dirs)  # acq4 writes dirs before, so this is the expected fill
            ncomplete = 0  # count number actually done
            for i, directory_name in enumerate(
                dirs
            ):  # dirs has the names of the runs within the protocol
                # if self.verbose:
                #    print('**DATA INFO: ', info)
                datafile = Path(directory_name, mainDevice + ".ma")  # clamp device file name
                if self.deep_check and i == 0:  # .index file is found, so proceed
                    clampInfo = self.AR.getDataInfo(datafile)
                    if self.verbose:
                        print(f"\n{prsp:s}**Datafile: {str(datafile):s}")
                        print(f"{prsp:s}**CLAMPINFO: {str(clampInfo):s}")
                        print(prsp + "**DATAFILE: ", datafile)
                        print(prsp + "**DEVICE: ", mainDevice)
                    if clampInfo is None:
                        break
                    self.holding = self.AR.parseClampHoldingLevel(clampInfo)
                    self.amp_settings = self.AR.parseClampWCCompSettings(clampInfo)
                    if (
                        self.amp_settings["WCEnabled"] == 1
                        and self.amp_settings["CompEnabled"] == 1
                    ):
                        print(
                            f"\n{prsp:s}WC R (MOhm) : {1e-6*self.amp_settings['WCResistance']:>6.2f}",
                            end="",
                        )
                        print(
                            f"{prsp:s}WC C (pF)   : {1e12*self.amp_settings['WCCellCap']:>6.1f}",
                            end="",
                        )
                        print(
                            f"{prsp:s}WC % Comp   : {self.amp_settings['CompCorrection']:>6.1f}",
                            end="",
                        )
                        print(f"{prsp:s}WC BW (kHz) : {1e-3*self.amp_settings['CompBW']:>6.2f}")
                    else:
                        print(prsp + "****NO WC Compensation****")

                    ncomplete += 1  # count up
                else:  # superficial check for existence of the file
                    datafile = Path(directory_name, mainDevice + ".ma")  # clamp device file name
                    if datafile.is_file():  # only check for existence of the fle
                        ncomplete += 1  # count anyway without going "deep"
            if ncomplete == nexpected:
                self.completeprotocols.append(protocol)
                # self.protocolstring += '[{:<s}: {:s} {:d}], '.format(protocol, modes[0][0], ncomplete)
            else:
                self.incompleteprotocols.append(protocol)
                self.incompleteprotocolstring += "{0:<s}.{1:s}.{2:d}/{3:d}, ".format(
                    protocol, modes[0][0], ncomplete, nexpected
                )
            if modes == []:
                modes = ["Unknown mode"]
            gc.collect()
            if self.verbose:
                print(prsp + "completeprotocols", self.completeprotocols)
                print(prsp + "incompleteprotocols", self.incompleteprotocols)

        if len(self.completeprotocols) == 0:
            self.completeprotocols = " "
        else:
            self.compprotstring = ", ".join([str(cp) for cp in self.completeprotocols])
        self.allprotocols = ", ".join(self.allprotocols)
        # Printer(f"Non prots: {str(nonprotocols):s}", color="cyan")
        for thisfile in nonprotocols:
            thisfile = str(thisfile.name)
            x = self.img_re.search(thisfile)  # look for image files
            if x is not None:
                images.append(thisfile)
            x = self.s2p_re.search(thisfile)  # two photon stacks
            if x is not None:
                stacks2p.append(thisfile)
            x = self.i2p_re.search(thisfile)  # simple two photon images
            if x is not None:
                images2p.append(thisfile)
            x = self.video_re.search(thisfile)  # video images
            if x is not None:
                videos.append(thisfile)
        self.imagestring = ""
        if len(images) > 0:
            self.imagestring += "Images: %3d" % len(images)
        if len(stacks2p) > 0:
            self.imagestring += ", 2pStacks: %3d" % len(stacks2p)
        if len(images2p) > 0:
            self.imagestring += ", 2pImages: %3d" % len(images2p)
        if len(videos) > 0:
            self.imagestring += ", Videos: %3d" % len(videos)
        # Printer(f"imagestring:  {self.imagestring:s}", color="red")
        if len(images) + len(stacks2p) + len(images2p) + len(videos) == 0:
            self.imagestring = "No Images or Videos"

        ostring = OrderedDict(
            [
                ("incomplete", self.incompleteprotocolstring.rstrip(", ")),
                ("complete", self.compprotstring.rstrip(", ")),
                ("images", self.imagestring),
                ("annotated", False),
                ("directory", self.basedir),
                ("cell_id", self.cell_id),
            ]
        )
        self.outputString(ostring)

    def getClampDeviceMode(self, info, clampDevice, modes):
        # print('info: ', info)
        if info is not None:  # no index, so we have a problem.
            if "devices" in info.keys():
                data_mode = info["devices"][clampDevice][
                    "mode"
                ]  # get mode from top of protocol information
            else:
                print(prsp + "? no clamp devices... ")
            if data_mode not in modes:
                modes.append(data_mode)
        return modes

    def colprint(self, phdr, ostring):
        ps = phdr.split("\t")
        os = ostring.split("\t")
        for i in range(len(ps)):
            if i > len(os):
                break
            if os[i] == " ":
                os[i] = "--"
            print("{0:3d}: {1:>20s} : {2:<s}".format(i, ps[i], os[i]))

    def outputString(self, ostring):
        day_string = ""
        phdr = ""
        for k in self.day_defs:
            day_string += str(self.day_index[k]) + "\t"
            phdr += k + "\t"

        slice_string = ""
        for k in self.slice_defs:
            if k == "mosaic":
                slice_string += self.slice_index[k] + "\t"

            else:
                ks = k.replace("slice_", "")
                slice_string += str(self.slice_index[ks]) + "\t"
            phdr += k + "\t"

        cell_string = ""
        for k in self.cell_defs:
            if k == "mosaic":
                cell_string += self.cell_index[k] + "\t"
            else:
                kc = k.replace("cell_", "")
                cell_string += str(self.cell_index[kc]) + "\t"
            phdr += k + "\t"

        prot_string = ""
        ohdr = ""
        for k in self.data_defs:
            kc = k.replace("data_", "")
            pstx = str(ostring[kc])
            if len(pstx) == 0:
                pstx = " "
            prot_string += pstx + "\t"
            phdr += k + "\t"
            ohdr += k + "\t"

        ostring = day_string + slice_string + cell_string + prot_string
        ostring = ostring.replace("\n", " ")
        ostring = ostring.rstrip("\t ")
        ostring += "\n"

        phdr = phdr.rstrip("\t\n")
        if len(self.panda_string) == 0:  # catch the header
            self.panda_string = phdr.replace("\n", "") + "\n"  # clip the last \t and add a newline

        # if self.outputMode in [None, "terminal"]:
        #     print("{0:s}".format(ostring))

        # elif self.outputMode == "text":
        #     h = open(self.outFilename, "a")  # append mode
        #     h.write(ostring)
        #     h.close()

        # elif
        if self.outputMode == "pandas":
            if self.verbose:
                self.colprint(phdr, ostring)
                print("\n******* building Pandas string", "ostring: \n", ostring)
            self.panda_string += ("{0:d}\t{1:s}").format(
                self.index, ostring
            )  # -1 needed to remove last tab...
            self.index += 1
        else:
            pass

    def write_string_pandas(self):
        """
        Write an output string using pandas dataframe
        """
        if self.dryrun:
            return
        if len(self.panda_string) == 0:
            return
        outfile = Path(self.outFilename)
        excelfile = Path(self.outFilename).with_suffix(".xlsx")
        if self.outputMode == "pandas" and not self.append:
            print("\nOUTPUTTING DIRECTLY VIA PANDAS, extension is .pkl")
            df = pd.read_csv(StringIO(self.panda_string), delimiter="\t")
            if outfile.suffix != ".pkl":
                outfile = outfile.with_suffix(".pkl")
            print("outfile: ", outfile)
            print("is file: ", Path(outfile).is_file())
            print("is parent path: ", Path(outfile).parent.is_dir())
            df.to_pickle(outfile)
            print(f"Wrote NEW pandas dataframe to pickled file: {str(outfile):s}")
            df.to_excel(excelfile, index=False)
            print(f"Wrote excel verion of dataframe to: {str(excelfile):s}")
            maindf = df

        elif self.outputMode == "pandas" and self.append:
            print("\nAPPENDING to EXISTING PANDAS DATAFRAME")
            # first save the original with a date-time string appended
            if outfile.suffix != ".pkl":
                outfile = outfile.with_suffix(".pkl")
            ofile = Path(outfile)
            if ofile.is_file:
                n = datetime.datetime.now()  # get current time
                dateandtime = n.strftime(
                    "_%Y%m%d-%H%M%S"
                )  # make a value indicating date and time for backup file
                bkfile = Path(ofile.parent, str(ofile.stem) + dateandtime).with_suffix(".bak")
                print("Copied original to backup file: ", bkfile)
                maindf = pd.read_pickle(ofile)  # read in the current file
                ofile.rename(bkfile)
            else:
                raise ValueError(f"Cannot append to non-existent file: {str(outfile):s}")
            df = pd.read_csv(StringIO(self.panda_string), delimiter="\t")
            maindf = pd.concat([maindf, df])
            maindf = maindf.reset_index(level=0, drop=True)
            # maindf = maindf.reset_index()  # redo the indices so all in sequence
            maindf.to_pickle(outfile)
            print(f"APPENDED pandas dataframe to pickled file: {str(outfile):s}")

        self.make_excel(maindf, outfile=excelfile)
        print(f"Wrote excel verion of dataframe to: {str(excelfile):s}")

    def get_file_information(self, dh=None):
        """
        get_file_information reads the sequence information from the
        currently selected data file

        Two-dimensional sequences are supported.
        :return nothing:
        """
        self.sequence = self.dataModel.listSequenceParams(dh)
        keys = self.sequence.keys()
        leftseq = [str(x) for x in self.sequence[keys[0]]]
        if len(keys) > 1:
            rightseq = [str(x) for x in self.sequence[keys[1]]]
        else:
            rightseq = []
        leftseq.insert(0, "All")
        rightseq.insert(0, "All")

    def file_cell_protocol(self, filename):
        """
        file_cell_protocol breaks the current filename down and returns a
        tuple: (date, cell, protocol)

        Parameters
        ----------
        filename : str
            Name of the protocol to break down

        Returns
        -------
        tuple : (date, sliceid, cell, protocol, any other...)
            last argument returned is the rest of the path...
        """
        filename = str(filename)
        proto = filename.name
        cell = filename.parent.name
        sliceid = filename.parent.parent.name
        date = filename.parent.parent.parent.name
        p3 = str(filename.parent.parent.parent.parent)
        return (date, sliceid, cell, proto, p3)

        # (p0, proto) = os.path.split(filename)
        # (p1, cell) = os.path.split(p0)
        # (p2, sliceid) = os.path.split(p1)
        # (p3, date) = os.path.split(p2)

        # print(date, sliceid, cell, proto)
        # print(protof, cellf, sliceidf, datef)
        # exit()
        # return (date, sliceid, cell, proto, p3)

    def highlight_by_cell_type(self, row):
        colors = {
            "pyramidal": "#c5d7a5",  # "darkseagreen",
            "cartwheel": "skyblue",
            "tuberculoventral": "lightpink",
            "granule": "linen",
            "golgi": "yellow",
            "unipolar brush cell": "sienna",
            "chestnut": "saddlebrown",
            "giant": "sandybrown",
            "giant_maybe": "sandybrown",
            "giant cell": "sandybrown",
            "Unknown": "white",
            "unknown": "white",
            " ": "white",
            "bushy": "lightslategray",
            "t-stellate": "thistle",
            "l-stellate": "darkcyan",
            "d-stellate": "thistle",
            "stellate": "thistle",
            "octopus": "darkgoldenrod",
            # cortical (uses some of the same colors)
            "basket": "lightpink",
            "chandelier": "sienna",
            "fast spiking": "darksalmon",
            "RS": "lightgreen",
            "LTS": "paleturquoise",
            # cerebellar
            "Purkinje": "mediumorchid",
            "purkinje": "mediumorchid",
            "purk": "mediumorchid",
            # not neurons
            "glia": "lightslategray",
            "Glia": "lightslategray",
        }
        if row.cell_type.lower() in colors.keys():
            return [f"background-color: {colors[row.cell_type.lower()]:s}" for s in range(len(row))]
        else:
            return [f"background-color: red" for s in range(len(row))]

    def regularize_layer(self, noteinfo: str):
        """Try to get layer information from the cell notes

        Args:
            noteinfo (str): the cell_notes field
        """
        layer_text = parse_layers.parse_layer(noteinfo)
        if layer_text is None:
            return ""
        else:
            return layer_text

    def organize_columns(self, df):
        return df
        cols = [
            "ID",
            "Group",
            "Date",
            "slice_slice",
            "cell_cell",
            "cell_type",
            "iv_name",
            "holding",
            "RMP",
            "RMP_SD",
            "Rin",
            "taum",
            "dvdt_rising",
            "dvdt_falling",
            "AP_thr_V",
            "AP_thr_T",
            "AP_HW",
            "AP15Rate",
            "AdaptRatio",
            "AP_begin_V",
            "AHP_trough_V",
            "AHP_trough_T",
            "AHP_depth_V",
            "tauh",
            "Gh",
            "FiringRate",
            "FI_Curve",
            "date",
        ]
        df = df[cols + [c for c in df.columns if c not in cols]]
        return df

    def make_excel(self, df: pd, outfile: Path):
        """cleanup: reorganize columns in spreadsheet, set column widths
        set row colors by cell type

        Args:
            df: object
                Pandas dataframe object
            excelsheet (_type_): _description_
        """
        if outfile.suffix != ".xlsx":
            outfile = outfile.with_suffix(".xlsx")

        writer = pd.ExcelWriter(outfile, engine="xlsxwriter")

        df.to_excel(writer, sheet_name="Sheet1")
        df = self.organize_columns(df)
        workbook = writer.book
        worksheet = writer.sheets["Sheet1"]
        fdot3 = workbook.add_format({"num_format": "####0.000"})
        df.to_excel(writer, sheet_name="Sheet1")

        resultno: list = []

        for i, column in enumerate(df):
            # print('column: ', column)
            if column in resultno:
                writer.sheets["Sheet1"].set_column(
                    first_col=i + 1, last_col=i + 1, cell_format=fdot3
                )
            if column not in ["notes", "description", "OriginalTable", "FI_Curve"]:
                coltxt = df[column].astype(str)
                coltxt = coltxt.map(str.rstrip)
                maxcol = coltxt.map(len).max()
                column_width = np.max([maxcol, len(column)])  # make sure the title fits
                if column_width > 50:
                    column_width = 50  # but also no super long ones
                # column_width = max(df_new[column].astype(str).map(len).max(), len(column))
            else:
                column_width = 25
            if column_width < 8:
                column_width = 8
            if column in resultno:
                writer.sheets["Sheet1"].set_column(
                    first_col=i + 1, last_col=i + 1, cell_format=fdot3, width=column_width
                )  # column_dimensions[str(column.title())].width = column_width
                print(f"formatted {column:s} with {str(fdot3):s}")
            else:
                writer.sheets["Sheet1"].set_column(
                    first_col=i + 1, last_col=i + 1, width=column_width
                )  # column_dimensions[str(column.title())].width = column_width

        df = df.style.apply(self.highlight_by_cell_type, axis=1)
        df.to_excel(writer, sheet_name="Sheet1")
        writer.close()


def dir_recurse(ds, current_dir, exclude_list: list = [], indent=0):
    if exclude_list is None:
        exclude_list = []
    print("Found current dir?: ", Path(current_dir).is_dir())
    files = sorted(list(current_dir.glob("*")))
    # print("files: ", files)
    alldatadirs = [
        f
        for f in files
        if f.is_dir() and str(f.name).startswith("20") and str(f.name) not in exclude_list
    ]
    # print("# dirs: ", alldatadirs)
    sp = " " * indent
    for d in alldatadirs:
        Printer(f"{sp:s}Data: {str(d.name):s}", "green")
    if len(alldatadirs) > 0:
        ds.getDay(alldatadirs)
        ds.write_string_pandas()
    # print("files 2: ", files)
    allsubdirs = [
        f
        for f in files
        if f.is_dir() and not str(f.name).startswith("20") and str(f.name) not in exclude_list
    ]
    indent += 2
    sp = " " * indent
    for d in allsubdirs:
        Printer(f"\n{sp:s}Subdir: {str(d.name):s}\n", "yellow")
        indent = dir_recurse(ds, d, exclude_list=exclude_list, indent=indent)
    indent -= 2
    if indent < 0:
        indent = 0
    Printer(f"\n{' '*indent:s}All Protocols in dataset: \n", "white")
    for prot in sorted(ds.all_dataset_protocols):
        Printer(f"{' '*indent:s}    {prot:s}", "blue")
    print("\nall done?")
    return indent


def main():
    parser = argparse.ArgumentParser(description="Generate Data Summaries from acq4 datasets")
    parser.add_argument("basedir", type=str, help="Base Directory")
    # This has been changed - the -f flag enables output to files. If specified, the -w flag is
    # also automatically set.
    # Output is written to both a pandas pickled database, and then to an excel file.
    # parser.add_argument('-o', '--output', type=str, default='pandas', dest='output',
    #                     choices=['terminal', 'pandas', 'excel', 'tabfile', 'text'],
    #                     help='Specify output dataplan key for one entry to process')
    parser.add_argument(
        "-f",
        "--filename",
        type=str,
        default="",
        dest="outputFilename",
        help="Specify output file name (including full path)",
    )
    parser.add_argument(
        "-r",
        "--read",
        action="store_true",
        dest="read",
        help="just read the summary table",
    )
    # parser.add_argument('-w', '--write', action='store_true', dest='write',
    #                     help='Analyze and write the data summary')
    parser.add_argument(
        "-u",
        "--update",
        action="store_true",
        dest="update",
        help="If writing, force update for days already in list",
    )
    parser.add_argument(
        "-D",
        "--deep",
        action="store_true",
        dest="deep",
        help="perform deep inspection (very slow)",
    )
    parser.add_argument(
        "--daylist", type=str, default=None, dest="daylist", help="Specify daylistfile"
    )
    parser.add_argument("-d", "--day", type=str, default=None, help="day for analysis")
    parser.add_argument(
        "-a",
        "--after",
        type=str,
        default=None,
        dest="after",
        help="only analyze dates on or after a date",
    )
    parser.add_argument(
        "-b",
        "--before",
        type=str,
        default=None,
        dest="before",
        help="only analyze dates on or before a date",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        dest="dryrun",
        help="Do a dry run, reporting only directories",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        dest="verbose",
        help="Verbose print out during run",
    )
    parser.add_argument(
        "--no-inspect",
        action="store_false",
        dest="noinspect",
        help="Do not inspect protocols, only report directories",
    )
    parser.add_argument(
        "--depth",
        type=str,
        default="all",
        dest="depth",
        choices=["days", "slices", "cells", "protocols", "all"],
        help="Specify depth for --dry-run",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="MultiClamp1.ma",
        dest="device",
        choices=["MultiClamp1.ma", "MultiClamp2.ma", "Clamp1.ma", "Clamp2.ma"],
        help="Specify device to examine parameters from",
    )
    parser.add_argument(
        "-A",
        "--append",
        action="store_true",
        dest="append",
        help="update new/missing entries to specified output file",
    )
    parser.add_argument("-p", "--pairs", action="store_true", dest="pairflag", help="handle pairs")
    parser.add_argument(
        "--subdirs",
        action="store_true",
        dest="subdirs",
        help="Also get data from subdirs that are not acq4 data dirs.",
    )
    parser.add_argument(
        "--exclude",
        type=str,
        default="",
        help="Exclude subdirectory by name",
    )

    args = parser.parse_args()
    ds = DataSummary(
        basedir=args.basedir,
        daylistfile=args.daylist,  # outputMode=args.output,
        outputFile=args.outputFilename,
        after=args.after,
        before=args.before,
        day=args.day,
        dryrun=args.dryrun,
        depth=args.depth,
        inspect=args.noinspect,
        deep=args.deep,
        append=args.append,
        subdirs=args.subdirs,
        verbose=args.verbose,
        update=args.update,
        pairflag=args.pairflag,
        device=args.device,
        excludedirs=args.exclude,
    )

    if args.outputFilename is not None:
        print("Writing to output, recurively through directories ")
        if args.exclude == None:
            args.exclude = []
        dir_recurse(ds, ds.basedir, args.exclude)

        exit()
        files = list(ds.basedir.glob("*"))
        alldirs = [f for f in files if f.is_dir() and str(f.name).startswith("20")]
        ds.getDay(alldirs)
        if args.output in ["pandas"]:
            ds.write_string()
        # check for subdirectories: 2 levels deep
        alldirs = [f for f in files if f.is_dir() and not str(f.name).startswith("20")]
        if len(alldirs) > 0:
            for directory in alldirs:
                newdir = Path(ds.basedir, directory).glob("*")
                files = list(newdir.glob("*"))
                allfiles = [f for f in files if f.is_file()]
                ds.getDay(allfiles)
                if args.output in ["pandas"]:
                    ds.write_string()

    if args.read:
        print("args.read")
        print("Valid file: ", Path(args.basedir).is_dir())
        print("reading: ", args.basedir)
        df = pd.read_pickle(args.basedir)

        print(df.head(10))

        df2 = df.set_index("date", drop=False)
        print(df2.head(5))

        for day in range(len(df2.index)):
            maps = []
            CCIVs = []
            VCIVs = []
            stdIVs = []
            map_types = []
            CCIV_types = []
            VCIV_types = []
            stdIV_types = []
            CC_Sponts = []
            VC_Sponts = []
            VC_Spont_types = []
            CC_Spont_types = []

            #       Only returns a dataframe if there is more than one entry
            #       Otherwise, it is like a series or dict
            date = df2.iloc[day]["date"]
            u = df2.iloc[day]["data_complete"].split(", ")
            prox = sorted(list(set(u)))  # adjust for duplicates (must be a better way in pandas)
            for p in prox:
                c = (
                    date
                    + "/"
                    + df2.iloc[day]["slice_slice"]
                    + "/"
                    + df2.iloc[day]["cell_cell"]
                    + "/"
                    + p
                )
                ps = c.rstrip("_0123456789")  # remove sequence numbers
                # print("protocols: ", c)
                if "Map".casefold() in c.casefold():
                    maps.append(c)
                    if ps not in map_types:
                        map_types.append(p)
                if "CCIV".casefold() in c.casefold():
                    CCIVs.append(c)
                    if ps not in CCIV_types:
                        CCIV_types.append(p)
                elif "VCIV".casefold() in c.casefold():
                    VCIVs.append(c)
                    if ps not in VCIV_types:
                        VCIV_types.append(p)
                elif "CC_Spont".casefold() in c.casefold():
                    CC_Sponts.append(c)
                    if ps not in CC_Spont_types:
                        CC_Spont_types.append(p)
                elif "VC_Spont".casefold() in c.casefold():
                    VC_Sponts.append(c)
                    if ps not in VC_Spont_types:
                        VC_Spont_types.append(p)
            print("=" * 80)
            print("COMPLETE PROTOCOLS")
            print("=" * 80)
            print("{0:<32s}".format(date))
            print("    Maps: ")
            for m in maps:
                print("        {0:<32s}".format(m))
            print("    CCIVs:")
            for iv in CCIVs:
                print("        {0:<32s}".format(iv))
            print("    VCIVs:")
            for iv in VCIVs:
                print("        {0:<32s}".format(iv))
            print("    STANDARD IVs: ")
            for iv in stdIVs:
                print("        {0:<32s}".format(iv))
            print("    CC_Sponts: ")
            for tr in CC_Sponts:
                print("        {0:<32s}".format(tr))
            print("    CV_Sponts: ")
            for tr in VC_Sponts:
                print("        {0:<32s}".format(tr))
            print("\n------------")
            print("    Map types: ")
            for m in map_types:
                print("        {0:<32s}".format(m))
            print("    CCIV types:")
            for iv in CCIV_types:
                print("        {0:<32s}".format(iv))
            print("    VCIV types:")
            for iv in VCIV_types:
                print("        {0:<32s}".format(iv))
            print("    STANDARD IVs types: ")
            for iv in stdIV_types:
                print("       {0:<32s}".format(iv))

            print("=" * 80)
            print("INCOMPLETE PROTOCOLS")
            print("=" * 80)
            u = df2.iloc[day]["data_incomplete"].split(", ")
            prox = sorted(list(set(u)))  # adjust for duplicates (must be a better way in pandas)
            for p in prox:
                # print('    protocol: ', p)

                c = (
                    date
                    + "/"
                    + df2.iloc[day]["slice_slice"]
                    + "/"
                    + df2.iloc[day]["cell_cell"]
                    + "/"
                    + p
                )
                print(c)
            print("=" * 80)

    print("\n\n")


if __name__ == "__main__":
    pass
    # main()
