#!/usr/bin/env python3
__author__ = 'pbmanis'
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
Uses acq4read and is independent of acq4 itself.

July 2018: 
Major surgery - to output Pandas (pickled) files as well. UGH.

----------------
usage: dataSummary [-h] [-o {terminal,pandas,excel,tabfile,text}]
                   [-f OUTPUTFILENAME] [-r] [-w] [-u] [-D] [--daylist DAYLIST]
                   [-d DAY] [-a AFTER] [-b BEFORE] [--dry-run] [-v]
                   [--no-inspect] [--depth {days,slices,cells,protocols,all}]
                   [-A] [-p]
                   basedir

    Generate Data Summaries from acq4 datasets

    positional arguments:
      basedir               Base Directory

    optional arguments:
      -h, --help            show this help message and exit
      -o {terminal,pandas,excel,tabfile,text}, --output {terminal,pandas,excel,tabfile,text}
                            Specify output dataplan key for one entry to process
      -f OUTPUTFILENAME, --filename OUTPUTFILENAME
                            Specify output file name (including full path)
      -r, --read            just read the summary table
      -w, --write           Analyze and write the data summary
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
import sys
import os
import re
from pathlib import Path
import math  # use to check nan value...
import argparse
import os.path
import gc
import argparse
import datetime
import dateutil.parser as DUP
import numpy as np
import textwrap
from collections import OrderedDict
import pandas as pd
import pandas.compat # for StringIO

from ephysanalysis import acq4read
from ephysanalysis.metaarray import MetaArray

def ansi_colors(color):
    colors = {
    'black': '\u001b[30m',
    'red': '\u001b[31m',
    'green': '\u001b[32m',
    'yellow': '\u001b[33m',
    'blue': '\u001b[34m',
    'magenta': '\u001b[35m',
    'cyan': '\u001b[36m',
    'white': '\u001b[37m',
    'reset': '\u001b[0m',
    }
    return colors[color]
    
class Printer():
    """Print things to stdout on one line dynamically"""
    def __init__(self, data, color='white'):
        
        sys.stdout.write(u"\r\u001b[2K%s"%ansi_colors(color)+data.__str__())
        sys.stdout.flush()

slsp = '   '
clsp = '        '
prsp = '             '

class DataSummary():

    def __init__(self, basedir, outputMode='terminal', outputFile=None, daylistfile=None,
                 after=None, before=None, day=None, dryrun=False, depth='all', inspect=True,
                 deep=False, append=False, verbose=False, update=False, pairflag=False):
        """
        Note that the init is just setup - you have to call getDay the object to do anything
    
        Parameters
        ----------
        basedir : str (required)
            base directory to be summarized
    
        outputMode : str (default: 'terminal')
            How to write the output. Current options are 'terminal', which writes to the terminal, and
            'pandas', which will create a pandas dataframe and pickle it to a file
        
        outputFile : str (default: None)
            File name and path for an output file (pandas/text)

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
        
        deep: bool (default: False)
            do a "deep" inspection of the data files, actually reading the .ma files
                 to confirm the data existence. This is slow...
        append: bool (default: False)
            Add new entries to the output file if they are not in the
                 database by appending them at the end
        verbose: bool (default: False)
            Provide extra print out during analysis for debugging.
    
        Note that if neither before or after are specified, the entire directory is read.
        """

        self.setups()  # build some regex and wrappers
        # gather input parameters
        self.basedir = basedir
        self.outputMode = outputMode  # terminal, tabfile, pandas
        self.outFilename = outputFile
        self.daylistfile = daylistfile
        self.dryrun = dryrun
        self.after = after
        self.before = before
        self.day = day
        self.depth = depth
        self.verbose = verbose
        self.update = update
        self.deep_check = deep
        self.pairflag = pairflag
        self.append = append
 
        self.daylist = None
        self.index = 0
        # flags - for debugging and verbosity
        self.reportIncompleteProtocols = True  # do include incomplete protocol runs in print
        self.InvestigateProtocols = inspect  # set True to check out the protocols in detail
        if self.dryrun:
            self.reportIncompleteProtocols = False # do include incomplete protocol runs in print
            self.InvestigateProtocols = False  # set True to check out the protocols in detail
        self.panda_string = ''

        # column definitions - may need to adjust if change data that is pasted into the output
        self.day_defs = ['date', 'description', 'notes', 'species', 'strain', 'genotype', 'age', 
                         'sex', 'weight', 'solution', 'internal', 'temperature', 'important', 'expUnit']
        self.slice_defs = ['slice_slice', 'slice_notes', 'slice_location', 'slice_orientation', 'important']
        self.cell_defs =  ['cell_cell',   'cell_notes',  'cell_type', 'cell_location', 'cell_important']
        self.data_defs = ['data_incomplete', 'data_complete', 'data_images', 'annotated', 'data_directory']
        
        # expected keys in various structural levels: days, slices, cells
        self.day_keys = ['description', 'notes', 'species', 'strain', 'genotype', 'age', 'sex', 'weight', 'solution', 'animal identification', '__timestamp__', 
                        'internal', 'temperature', 'expUnit', 'dirType', 'important', 'time']
        self.slice_keys = ['notes', 'location', 'orientation', 'important', '__timestamp__']
        self.cell_keys = [ 'notes', 'type', 'location', 'important', '__timestamp__']
        self.data_dkeys = ['incomplete', 'complete', 'data_images', 'annotated', 'directory']
        
        self.day_template = (
            OrderedDict([('species', '{:>s}'), ('strain', '{:>s}'),('genotype', '{:>12s}'), ('age', '{:>5s}'), ('sex', '{:>2s}'), ('weight', '{:>5s}'),
                         ('solution', '{:>s}'), ('internal', '{:>s}'), ('temperature', '{:>5s}'), ('important', '{:>s}'), ('elapsedtime', '{:>8.2f}'), 
                         ('expUnit', '{:>s}')])
                        )        
        self.slice_template = (
            OrderedDict([('type', '{:>s}'), ('location', '{:>12s}'), ('orientation', '{:>5s}')])
                        )        
        self.cell_template = (
            OrderedDict([('type', '{:>s}'), ('location', '{:>12s}'), ('important', '{:>s}')])
                        )        
        self.data_template = (
            OrderedDict([('incomplete', '{0:s}'), ('complete', '{1:s}'), ('images', '{2:s}'), ('annotated', '{3:s}'), ('directory', '{4:s}')])
                        )        
        self.AR = acq4read.Acq4Read()  # instance of the reader

        if self.outputMode == 'tabfile':
            print('Tabfile output: Writing to {:<s}'.format(self.outFilename))
            h = open(self.outFilename, 'w')  # write new file
            h.write(self.basedir+'\n')
            h.write(self.coldefs + '\n')
            h.close()
        elif self.outputMode == 'pandas':  # save output as a pandas data structure, pickled
            print('Pandas output: will write to {:<s}'.format(self.outFilename))
        else:
            pass  # just print if terminal
            
        # figure out the before/after limits
        # read the before and after day strings, parse them and set the min and maxdays
        if self.after is None:
            mindayx = (1970, 1, 1)  # all dates after this - unix start date convention
        else:
            try:
                dt = DUP.parse(self.after)
                mindayx = (dt.year, dt.month, dt.day)
            except:
                raise ValueError('Date for AFTER cannot be parsed : {0:s}'.format(self.after))
        if self.before is None:
            maxdayx = (2266, 1, 1)  # far enough into the future for you? Maybe when the Enterprise started it's journey?
        else:
            try:
                dt = DUP.parse(self.before)
                maxdayx = (dt.year, dt.month, dt.day)
            except:
                raise ValueError('Date for BEFORE cannot be parsed : {0:s}'.format(self.before))
        if self.day is not None:
                dt = DUP.parse(self.day)
                mindayx = (dt.year, dt.month, dt.day)
                maxdayx = (dt.year, dt.month, dt.day)
            
        print('after, before, mindayx, maxdayx: ', self.after, self.before, mindayx, maxdayx)
        print('daylistfile: ', self.daylistfile)
        if self.daylistfile is None:  # get from command line
            self.minday = mindayx[0]*1e4+mindayx[1]*1e2+mindayx[2]
            self.maxday = maxdayx[0]*1e4+maxdayx[1]*1e2+maxdayx[2]
            print(self.minday, self.maxday)
        else:
            self.daylist = []
            with open(self.daylistfile, 'r') as f:
                for line in f:
                    if line[0] != '#':
                        self.daylist.append(line[0:10])
            f.close()  # silly - it dhould be closed.
        
    def setups(self):
        self.tw = {}  # for notes
        self.tw['day'] = textwrap.TextWrapper(initial_indent="", subsequent_indent=" "*2)  # used to say "initial_indent ="Description: ""
        self.tw['slice'] = textwrap.TextWrapper(initial_indent="", subsequent_indent=" "*2)
        self.tw['cell'] = textwrap.TextWrapper(initial_indent="", subsequent_indent=" "*2)
        self.tw['pair'] = textwrap.TextWrapper(initial_indent="", subsequent_indent=" "*2)

        self.twd = {}  # for description
        self.twd['day'] = textwrap.TextWrapper(initial_indent="", subsequent_indent=" "*2)  # used to ays initial_indent ="Notes: ""
        self.twd['slice'] = textwrap.TextWrapper(initial_indent="", subsequent_indent=" "*2)
        self.twd['cell'] = textwrap.TextWrapper(initial_indent="", subsequent_indent=" "*2)
        self.twd['pair'] = textwrap.TextWrapper(initial_indent="", subsequent_indent=" "*2)
        
        self.img_re = re.compile('^[Ii]mage_(\d{3,3}).tif')  # make case insensitive - for some reason in Xuying's data
        self.s2p_re = re.compile('^2pStack_(\d{3,3}).ma')
        self.i2p_re = re.compile('^2pImage_(\d{3,3}).ma')
        self.video_re = re.compile('^[Vv]ideo_(\d{3,3}).ma')

        self.daytype = re.compile("(\d{4,4}).(\d{2,2}).(\d{2,2})_(\d{3,3})")
#        daytype = re.compile("(2011).(06).(08)_(\d{3,3})")  # specify a day

    def getDay(self):
        """
        getDay is the entry point for scanning through all the data files in a given directory,
        returning information about those within the date range, with details as specified by the options
        
        Parameters
        ----------
        None
        
        Returns 
        -------
        Nothing
        
        The result is stored in the class variable self.day_index
        
        """
        allfiles = os.listdir(self.basedir)
        if self.append:
            print('reading for append: ', self.outFilename)
            self.pddata = pd.read_pickle(self.outFilename)  # get the current file
        print('alldays: ', allfiles)
        self.pstring = ''
        days = []
        #print('allfiles: ', allfiles)
        for thisfile in allfiles:
            if thisfile.endswith('.sql'):
                continue
            if str(thisfile) in ['.DS_Store', '.index']:
                continue
            m = self.daytype.search(thisfile)
            if str(thisfile) in ['.DS_Store', '.index']:
                continue
            if m is None:
                print('no match in daytype : ', thisfile)
                continue  # no match
            if len(m.groups()) >= 3:  # perfect match
                idl = [int(d) for d in m.groups()]
                id = idl[0]*1e4+idl[1]*1e2+idl[2]

                if self.daylist is None:
                    # print('id, self.minday, self.maxday: ', id, self.minday, self.maxday)
                    if id >= self.minday and id <= self.maxday:
                        days.append(thisfile)  # was [0:10]
                else:
                    if thisfile[0:10] in self.daylist:
                        days.append(thisfile)
        #print('days: ', days)
        if self.verbose:
            print ('Days reported: ', days)
            if self.append:
                print('Days in pandas frame: ', self.pddata['date'].tolist())

        for nd, day in enumerate(days):
            if self.append and (day in self.pddata['date'].tolist()):
                if not self.update:
                    print('\nAppend mode: day already in list: {0:s}'.format(day))
                    continue  # skip
                else:
                    print(self.pddata['date'])
                    print(day)
                    k = self.pddata.index[self.pddata['date'] == day]
                    print(k)
                    self.pddata = self.pddata.drop(index=k)  # remove the day and update it
            else:
                print('Day to do: ', day)
            if self.verbose:
                self.pstring = 'Processing day[%3d/%3d]: %s ' % (nd, len(days), day)
            self.AR.setProtocol(Path(self.basedir, day))
            self.day_index = self.AR.readDirIndex(Path(self.basedir, day))
            if self.day_index is None:
                print('Day {0:s} is not managed (no .index file found)'.format(day))
                self.day_index={}
                continue
            self.day_index = self.day_index['.']
            # print('ind: ', ind)
           #  self.day_index = self.AR.readDirIndex(ind)
            # print('day index: ', self.day_index)
            self.day_index['date'] = day.strip()
            # print('\nday index: ', self.day_index.keys())
            # print('daydefs: ', self.day_defs)
            # now add the rest of the index information to the daystring
            for k in self.day_defs:  # for all the keys
                if k not in self.day_index.keys():
                    # print('\nadded: ', k)
                    self.day_index[k] = '' # 'missing'
                # else:
                #     print(' ? k in day index: ', k)
                if isinstance(self.day_index[k], bool):
                    self.day_index[k] = str(self.day_index[k])
                if k in 'sex':
                    if self.day_index[k] not in ['M', 'F', 'm', 'f', None, '', ' ']:
                        print('? sex: <'+self.day_index[k]+'>')
                        exit()
                if k in 'age':
                    chrs = str.maketrans('pPdDmMyY', '        ')
                    self.day_index[k] = self.day_index[k].translate(chrs) # strip characters
                    # self.day_index[k] = int(self.day_index[k])
                self.day_index[k].replace('\n', ' ')
                if len(self.day_index[k]) == 0:
                    self.day_index[k] = ' '
            for k in self.day_defs:
                print('{:>32s} : {:<40s}'.format(k, self.day_index[k]))
            self._doSlices(day)  # next level
            os.closerange(8, 65535)  # close all files in each iteration
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
        allfiles = Path(self.basedir, day).glob('*')
        slicetype = re.compile("(slice\_)(\d{3,3})")
        slices = []
        for thisfile in list(allfiles):
            print(slsp + 'slicefile: ', thisfile)
            thisfile = str(thisfile)
            m = slicetype.search(str(thisfile))
            if m is None:
                print(slsp + 'm is none')
                continue
            if len(m.groups()) == 2:
                slices.append(''.join(m.groups(2))) # Path(thisfile).parts[-1])
        for slicen in slices:
            slicen = str(slicen)
            self.sstring = day + ' ' + self.pstring + ' %s' % slicen
            Printer(self.sstring)
            self.slicestring = '%s\t' % (slicen)
            self.slice_index = self.AR.readDirIndex(Path(self.basedir, day, slicen))
            # print(slsp + 'slice index: ', self.slice_index)
            if self.slice_index is None:  # directory is not managed and probably empty
                # print(slsp + 'Slice {0:s} is not managed (no .index file found)'.format(slicen))
                self.slice_index = {}
                continue
            self.slice_index = self.slice_index['.']
            self.slice_index['slice'] = slicen
            for k in self.slice_defs:
                ks = k.replace('slice_', '')
                if ks not in self.slice_index.keys():
                    self.slice_index[ks] = ' '
                if isinstance(self.slice_index[ks], bool):
                    self.slice_index[ks] = str(self.slice_index[ks])
                if len(self.slice_index[ks]) == 0 :
                    self.slice_index[ks] = ' '
                self.slice_index[ks].replace('\n', ' ')
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
        
        The result is stored in teh calss variable cell_index
        
        """
        print(clsp + 'docells')
        allfiles = Path(thisslice).glob('*')
        # print('in doCells, allfiles: ', list(allfiles))
        if not self.pairflag:
            cell_re = re.compile("(cell_)(\d{3,3})")
        else:
            cell_re = re.compile("(pair_)(\d{3,3})")
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
                cells.append(''.join(m.groups(2)))
        for cell in cells:
            print(clsp + 'cell: ', cell)
            self.cstring = self.sstring + " %s" % cell
            Printer(self.cstring)
            try:
                self.cell_index = self.AR.readDirIndex(Path(thisslice, cell))['.']  # possible that .index file is missing, so we cannot read
            except:
                self.cell_index={}  # unreadable...
                continue
            if self.cell_index is None:
                self.cell_index = {}  # directory is not managed, so skip
                continue
            # print('\nCell Index: ', self.cell_index)
            self.cell_index['cell'] = cell
            for k in self.cell_defs:
                ks = k.replace('cell_', '')
                if ks not in self.cell_index.keys():
                    self.cell_index[ks] = ' '
                if isinstance(self.cell_index[ks], bool):
                    self.cell_index[ks] = str(self.cell_index[ks])
                self.cell_index[ks].replace('\n', ' ')
                if len(self.cell_index[ks]) == 0:
                    self.cell_index[ks] = ' '
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
        allfiles = thiscell.glob('*')
        protocols = []
        nonprotocols = []
        anyprotocols = False
        images = []  # tiff
        stacks2p = []
        images2p = []
        videos = []

        endmatch = re.compile("[\_(\d{3,3})]$")  # look for _lmn at end of directory name
        for thisfile in allfiles:
            if Path(thiscell, thisfile).is_dir():
                protocols.append(thisfile)
            else:
                nonprotocols.append(thisfile)

        self.incompleteprotocolstring = ''
        self.allprotocols = []
        self.incompleteprotocols = []
        self.completeprotocols = []
        self.compprotstring = ''
#        if self.InvestigateProtocols is True:
           # self.thiscell_summarystring = 'NaN\t'*6
        if self.verbose:
            print('\n'+prsp+'Investigating Protocols')
        for np, protocol in enumerate(protocols):  # all protocols on the cell
            mp = Path(protocol).parts
            protocol = mp[-1]
            protocol = str(protocol)
            if protocol.startswith('Patch'):
                continue

            Printer(self.cstring + ' Prot[%2d/%2d]: %s' % (np, len(protocols), protocol))
            self.allprotocols += protocol + ', '
            protocolpath = Path(thiscell, protocol)
            dirs = self.AR.subDirs(protocolpath)  # get all sequence entries (directories) under the protocol
            modes = []
            info = self.AR.readDirIndex(protocolpath) # top level info dict
            if info is None:
                print('prsp+Protocol is not managed (no .index file found): {0:s}'.format(protocolpath))
                continue
            info = info['.']
            if 'devices' not in info.keys():  # just safety... 
                continue
            devices = info['devices'].keys()
            clampDevices = []
            for d in devices:
                if d in self.AR.clampdevices:
                    clampDevices.append(d)
            if len(clampDevices) == 0:
                continue # ignore protocol
            mainDevice = clampDevices[0]
            modes = self.getClampDeviceMode(info, mainDevice, modes)
#            print('modes: ', modes)
            nexpected = len(dirs)  # acq4 writes dirs before, so this is the expected fill
            ncomplete = 0  # count number actually done
            for i, directory_name in enumerate(dirs):  # dirs has the names of the runs within the protocol
                # if self.verbose:
#    print('**DATA INFO: ', info)
                datafile = Path(directory_name, mainDevice+'.ma')  # clamp device file name
                if self.deep_check:  # .index file is found, so proceed
                    clampInfo = self.AR.getDataInfo(datafile)
                    if self.verbose:
                        print(prsp+'**CLAMPINFO: ', clampInfo)
                        print(prsp+'**DATAFILE: ', datafile)
                        print(prsp+'**DEVICE: ', mainDevice)
                    if clampInfo is None:
                        break
                    self.holding = self.AR.parseClampHoldingLevel(clampInfo)
                    self.amp_settings = self.AR.parseClampWCCompSettings(clampInfo)
                    ncomplete += 1  # count up
                else:  # superficial check for existence of the file
                    datafile = Path(directory_name, mainDevice+'.ma')  # clamp device file name
                    if datafile.is_file():  # only check for existence of the fle
                        ncomplete += 1  # count anyway without going "deep"
            if ncomplete == nexpected:
                self.completeprotocols.append(protocol)
                #self.protocolstring += '[{:<s}: {:s} {:d}], '.format(protocol, modes[0][0], ncomplete)
            else:
                self.incompleteprotocols.append(protocol)
                self.incompleteprotocolstring += '{0:<s}.{1:s}.{2:d}/{3:d}, '.format(protocol, modes[0][0], ncomplete, nexpected)
            if modes == []:
                modes = ['Unknown mode']
            gc.collect()
            if self.verbose:
                print(prsp+'self.completeprotocols', self.completeprotocols)
                print(prsp+'self.incompleteprotocols', self.incompleteprotocols)

        if len(self.completeprotocols) == 0:
            self.completeprotocols = ' '
        else:
            self.compprotstring = ', '.join([str(cp) for cp in self.completeprotocols])
        self.allprotocols = ', '.join(self.allprotocols)
        for thisfile in nonprotocols:
            thisfile = str(thisfile)
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
        self.imagestring = ''
        if len(images) > 0:
            self.imagestring += 'Images: %3d' % len(images)
        if len(stacks2p) > 0:
            self.imagestring += ', 2pStacks: %3d' % len(stacks2p)
        if len(images2p) > 0:
            self.imagestring += ', 2pImages: %3d' % len(images2p)
        if len(videos) > 0:
            self.imagestring += ', Videos: %3d' % len(videos)
        if len(images) + len(stacks2p) + len(images2p) + len(videos) == 0:
            self.imagestring = 'No Images or Videos'

        ostring = OrderedDict([('incomplete', self.incompleteprotocolstring.rstrip(', ')),
                               ('complete', self.compprotstring.rstrip(', ')),
                               ('images', self.imagestring),
                               ('annotated', False),
                               ('directory', self.basedir),])
        self.outputString(ostring)

    def getClampDeviceMode(self, info, clampDevice, modes):
        # print('info: ', info)
        if info is not None:  # no index, so we have a problem.
            if 'devices' in info.keys():
                data_mode = info['devices'][clampDevice]['mode']  # get mode from top of protocol information
            else:
                print(prsp+'? no clamp devices... ')
            if data_mode not in modes:
                modes.append(data_mode)
        return(modes)

    def colprint(self, phdr, ostring):
        ps = phdr.split('\t')
        os = ostring.split('\t')
        for i in range(len(ps)):
            if i > len(os):
                break
            if os[i] == ' ':
                os[i] = '--'
            print('{0:3d}: {1:>20s} : {2:<s}'.format(i, ps[i], os[i]))

    def outputString(self, ostring):
        day_string = ''
        phdr = ''
        for k in self.day_defs:
            day_string += str(self.day_index[k])+'\t'
            phdr += k +'\t'
        
        slice_string = ''
        for k in self.slice_defs:
            ks = k.replace('slice_', '')
            slice_string += str(self.slice_index[ks])+'\t'
            phdr += k +'\t'
        
        cell_string = ''
        for k in self.cell_defs:
            kc = k.replace('cell_', '')
            cell_string += str(self.cell_index[kc])+'\t'
            phdr += k +'\t'
        
        prot_string = ''
        ohdr = ''
        for k in self.data_defs:
            kc = k.replace('data_', '')
            pstx = str(ostring[kc])
            if len(pstx) == 0:
                pstx = ' '
            prot_string += pstx + '\t'
            phdr += k + '\t'
            ohdr += k + '\t'
            
        ostring = day_string + slice_string + cell_string + prot_string
        ostring = ostring.replace('\n', ' ')
        ostring = ostring.rstrip('\t ')
        ostring += '\n'

        phdr = phdr.rstrip('\t\n')
        if len(self.panda_string) == 0:  # catch the header
            self.panda_string = phdr.replace('\n', '') + '\n'  # clip the last \t and add a newline
        
        if self.outputMode in [None, 'terminal']:
            print('{0:s}'.format(ostring))
        
        elif self.outputMode == 'text':
            h = open(self.outFilename, 'a')  # append mode
            h.write(ostring )
            h.close()
        
        elif self.outputMode == 'pandas':
            if self.verbose:
                self.colprint(phdr, ostring)            
                print('\n******* building Pandas string', 'ostring: \n', ostring)
            self.panda_string += ('{0:d}\t{1:s}').format(self.index, ostring)  # -1 needed to remove last tab...
            self.index += 1
        else:
            pass
    
    def write_string(self):
        """
        Write an output string using pandas dataframe
        """
        if self.dryrun:
            return
        if self.outputMode == 'pandas' and not self.append:
            print('\nOUTPUTTING DIRECTLY VIA PANDAS')
          #  self.colprint()
            print('pandstring: ', self.panda_string)
            df = pd.read_csv(pandas.compat.StringIO(self.panda_string), delimiter='\t')
           # print('Head write: \n', df.head(5), '\n')
            df.to_pickle(self.outFilename)
        if self.outputMode == 'pandas' and self.append:
            print('\nAPPENDING to EXISTING PANDAS DATAFRAME')
            # first save the original with a date-time string appended
            ofile = Path(self.outFilename)
            if ofile.is_file:
                n = datetime.datetime.now()  # get current time
                dateandtime = n.strftime("_%Y%m%d-%H%M%S") # make a value indicating date and time for backup file
                bkfile = Path(ofile.parent, str(ofile.stem)+dateandtime).with_suffix('.bak')
                print('Copied original to backup file: ', bkfile)
                maindf = pd.read_pickle(ofile) # read in the current file
                ofile.rename(bkfile)
            else:
                raise ValueError(f'Cannot append to non-existent file: {self.outFilename:s}')
            df = pd.read_csv(pandas.compat.StringIO(self.panda_string), delimiter='\t')
            maindf = maindf.append(df)
            maindf = maindf.reset_index(level=0, drop=True)
           # maindf = maindf.reset_index()  # redo the indices so all in sequence
            maindf.to_pickle(self.outFilename)
        print('Wrote pandas dataframe to pickled file: {0:s}'.format(self.outFilename))

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
        leftseq.insert(0, 'All')
        rightseq.insert(0, 'All')

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
        (p0, proto) = os.path.split(filename)
        (p1, cell) = os.path.split(p0)
        (p2, sliceid) = os.path.split(p1)
        (p3, date) = os.path.split(p2)
        return (date, sliceid, cell, proto, p3)

def main():
    parser = argparse.ArgumentParser(description='Generate Data Summaries from acq4 datasets')
    parser.add_argument('basedir', type=str,
                        help='Base Directory')
    parser.add_argument('-o', '--output', type=str, default='pandas', dest='output',
                        choices=['terminal', 'pandas', 'excel', 'tabfile', 'text'],
                        help='Specify output dataplan key for one entry to process')
    parser.add_argument('-f', '--filename', type=str, default='terminal', dest='outputFilename',
                        help='Specify output file name (including full path)')
    parser.add_argument('-r', '--read', action='store_true', dest='read',
                        help='just read the summary table')
    parser.add_argument('-w', '--write', action='store_true', dest='write',
                        help='Analyze and write the data summary')
    parser.add_argument('-u', '--update', action='store_true', dest='update',
                        help='If writing, force update for days already in list')
    parser.add_argument('-D', '--deep', action='store_true', dest='deep',
                        help='perform deep inspection (very slow)')
    parser.add_argument('--daylist', type=str, default=None, dest='daylist',
                        help='Specify daylistfile')
    parser.add_argument('-d', '--day', type=str, default=None, 
                        help='day for analysis')
    parser.add_argument('-a', '--after', type=str, default=None, dest='after',
                        help='only analyze dates on or after a date')
    parser.add_argument('-b', '--before', type=str, default=None, dest='before',
                        help='only analyze dates on or before a date')
    parser.add_argument('--dry-run', action='store_true', dest='dryrun',
                        help='Do a dry run, reporting only directories')
    parser.add_argument('-v', '--verbose', action='store_true', dest='verbose',
                        help='Verbose print out during run')
    parser.add_argument('--no-inspect', action='store_false', dest='noinspect',
                        help='Do not inspect protocols, only report directories')
    parser.add_argument('--depth', type=str, default='all', dest='depth',
                        choices = ['days', 'slices', 'cells', 'protocols', 'all'],
                        help='Specify depth for --dry-run')
    parser.add_argument('-A', '--append', action='store_true', dest='append',
                        help='update new/missing entries to specified output file')
    parser.add_argument('-p', '--pairs', action='store_true', dest='pairflag',
                        help='handle pairs')
                    
    args = parser.parse_args()
    ds = DataSummary(basedir=args.basedir, daylistfile=args.daylist, outputMode=args.output, outputFile=args.outputFilename,
            after=args.after, before=args.before, day=args.day, dryrun=args.dryrun, depth=args.depth, inspect=args.noinspect,
            deep=args.deep, append=args.append,
            verbose=args.verbose, update=args.update, pairflag=args.pairflag)

    if args.write:
        print('Writing: ')
        ds.getDay()
        if args.output in ['pandas']:
            ds.write_string()
    
    if args.read:
        print('args.read')
        print('reading: ', args.basedir)
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
        
#       Only returns a dataframe if there is more than one entry
#       Otherwise, it is like a series or dict
            date = df2.iloc[day]['date']
            u = df2.iloc[day]['data_complete'].split(', ')
            prox = sorted(list(set(u)))  # adjust for duplicates (must be a better way in pandas)
            for p in prox:
               # print('    protocol: ', p)

                c = date + '/' + df2.iloc[day]['slice_slice'] + '/' + df2.iloc[day]['cell_cell'] + '/' + p
                ps = c.rstrip('_0123456789')  # remove sequence numbers
                if 'Map' in c:
                    maps.append(c)
                    if ps not in map_types:
                        map_types.append(p)
                if 'CCIV' in c:
                    CCIVs.append(c)
                    if ps not in CCIV_types:
                        CCIV_types.append(p)
                if 'VCIV' in c:
                    VCIVs.append(c)
                    if ps not in VCIV_types:
                        VCIV_types.append(p)
            print('='*80)
            print('COMPLETE PROTOCOLS')
            print('='*80)
            print('{0:<32s}'.format(date))
            print('    Maps: ')
            for m in maps:
                print('        {0:<32s}'.format(m))
            print('    CCIVs:')
            for iv in CCIVs:
                print('        {0:<32s}'.format(iv))
            print('    VCIVs:')
            for iv in VCIVs:
                print('        {0:<32s}'.format(iv))
            print('    STANDARD IVs: ')
            for iv in stdIVs:
                print('        {0:<32s}'.format(iv))
        
            print('\n------------')
            print('    Map types: ')
            for m in map_types:
                print('        {0:<32s}'.format(m))
            print('    CCIV types:')
            for iv in CCIV_types:
                print('        {0:<32s}'.format(iv))
            print('    VCIV types:')
            for iv in VCIV_types:
                print('        {0:<32s}'.format(iv))
            print('    STANDARD IVs types: ')
            for iv in stdIV_types:
                print('       {0:<32s}'.format(iv))
        
            print('='*80)
            print('INCOMPLETE PROTOCOLS')
            print('='*80)
            u = df2.iloc[day]['data_incomplete'].split(', ')
            prox = sorted(list(set(u)))  # adjust for duplicates (must be a better way in pandas)
            for p in prox:
               # print('    protocol: ', p)

                c = date + '/' + df2.iloc[day]['slice_slice'] + '/' + df2.iloc[day]['cell_cell'] + '/' + p
                print(c)
            print('='*80)

    print('\n\n')
    
if __name__ == "__main__":
    main()

