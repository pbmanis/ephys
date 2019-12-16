from __future__ import print_function

__author__ = 'pbmanis'
"""
dir_check.py
Check an acq4 directory struture to be sure that the hierarchy is correct.
day
    slice
        cell
            protocols
        pair
            cells
                protocols

The slice may have images and videos, but no protocols
The cells and pairs may have both
Pairs should have cells and protocols under cells

Any other organization is flagged.


"""
import sys
import os
import re
import math  # use to check nan value...
import argparse
import contextlib
import subprocess

import os.path
import stat
import gc
import argparse
import datetime
import dateutil.parser as DUP
from termcolor import colored
import numpy as np
import textwrap
from collections import OrderedDict
import pandas as pd
import pandas.compat # for StringIO

from ephysanalysis import acq4read
from pyqtgraph.metaarray import MetaArray

latex_header = """\\documentclass[8pt, letterpaper, oneside]{article}
\\usepackage[utf8]{inputenc}
\\usepackage{fancyvrb}
\\usepackage{xcolor}
\\usepackage{geometry}
\\geometry{
 landscape,
 left=0.5in,
 top=0.5in,
 }

\\title{dircheck}
\\author{dir_check.py}
\\date{1:s}
 
\\begin{document}

"""

latex_footer = """

\\end{Verbatim}
\\end{document}
"""
class Printer():
    """Print things to stdout on one line dynamically"""
    def __init__(self, data):
        sys.stdout.write("\033[1;36m\r\x1b[K\033[0;0m"+data.__str__())
        sys.stdout.flush()


class DirCheck():
    def __init__(self, topdir, protocol=False, output=None, args=None):
        """
        Check directory structure
        """
        if topdir.endswith(os.path.sep):
            topdir = topdir[:-1]
        self.topdir = topdir
        self.outfile = None
        self.coloredOutput = True
        self.outputMode = 'text'
        self.after = DUP.parse(args.after)
        self.before = DUP.parse(args.before)
        
        self.lb = '\n'
        if output is not None:
            self.coloredOutput = False
            self.outfile = output
            with open(self.outfile, 'w') as f:
                pass

            p, e = os.path.splitext(self.outfile)
            if e == '.tex':  # write latex header and set up 
                self.outputMode = 'latex'
                with open(self.outfile, 'a') as f:
                    f.write(latex_header)
                self.lb = '\n'
            
            
        self.show_protocol = protocol
        self.img_re = re.compile('^[Ii]mage_(\d{3,3}).tif')  # make case insensitive - for some reason in Xuying's data
        self.s2p_re = re.compile('^2pStack_(\d{3,3}).ma')
        self.i2p_re = re.compile('^2pImage_(\d{3,3}).ma')
        self.video_re = re.compile('^[Vv]ideo_(\d{3,3}).ma')
        self.daytype = re.compile("(\d{4,4}).(\d{2,2}).(\d{2,2})_(\d{3,3})")
        self.tstamp = re.compile('\s*(__timestamp__: )([\d.\d]*)')
        AR = acq4read.Acq4Read()
        
        if not os.path.isdir(self.topdir):
            raise ValueError('top dir does not exist: %s' % self.topdir)

        metaarraytypes = []
        protocols = []
        level = 0
        parent = ''
        lastcb = ''
        # walk the directory from top down - but in a highly controlled manner:
        # check to see if top dir is a day or just a generic directory
        # if it is a generic directory, we get a list of dirs
        # if it is a day, we make a list out of the day
        path, lastdir = os.path.split(self.topdir)
        td = self.daytype.match(lastdir)
        #print(path, td)
        if td is not None:
            topdirs = [lastdir]
            self.topdir = path
        else:
            topdirs = os.listdir(self.topdir)
        
        fmtstring = '{0:>15s} {1:<10s} {2:<10s} {3:<40} {4:>20}'
        if self.outfile is not None:
            fmtstring += self.lb  # insert newlines when writing output to file
        fmtstring2 = '{0:>15s} {1:<10s} {2:<40s} {3:<10} {4:>20}'
        if self.outputMode == 'latex':
        #    self.printLine('\\end{Verbatim}' + self.lb)
            self.printLine('\\vspace{2cm}\\center{\\textbf{\\large{Directory Check/Protocol Listing}}}'+self.lb)
            self.printLine('\\vspace{2cm}\\center{\\textbf{\huge{'+self.outfile.replace('_', '$\_$')+'}}}' + self.lb)
            self.printLine('\\vspace{1cm}\\center{\\textbf{\\large{'+self.topdir.replace('_', '$\_$')+'}}}' + self.lb)
            now = datetime.datetime.now().strftime('%Y-%m-%d  %H:%M:%S %z')
            self.printLine('\\vspace{1cm}\\center{\\textbf{\\huge{'+now+'}}}' + self.lb)
            self.printLine('\\newpage'+self.lb)
            self.printLine('\\begin{Verbatim} '+self.lb)
            
        for ndir, d in enumerate(sorted(topdirs)):

            if d in ['.DS_Store', 'log.txt'] or self.check_extensions(d):
                continue
            if any([d.endswith(e) for e in ['.tif', '.ma']]):
                continue
            if d.startswith('Accidental'):
                continue
            if d in ['.index']:
                indir = os.path.join(self.topdir, d)
                ind = AR.readDirIndex(self.topdir)
                if ndir > 0:
                    self.printLine(AR.getIndex_fromstring(ind))
                continue
            
            dp, ds = os.path.split(d)
            dsday, nx = ds.split('_')
         #   print('&&&& ', dsday)
            thisday = datetime.datetime.strptime(dsday, '%Y.%m.%d')
            if thisday < self.after or thisday > self.before:
                continue

            self.printLine(' ')

            m = self.daytype.match(d)
            tstamp = self.gettimestamp(os.path.join(self.topdir, d))
            if ndir > 1:
                self.printLine(self.lb + '\\end{Verbatim}'+self.lb)
                self.printLine('+'*100 + self.lb + '\\newpage')
                self.printLine('\\begin{Verbatim} ' + self.lb)
            if m is not None:
                self.printLine(fmtstring.format(d, '', '', '', tstamp))
            else:
                self.printLine((fmtstring+'is not a DAY directory').format(d, '', '', '', tstamp), 'red')
            daysdone = []

            for s in sorted(os.listdir(os.path.join(self.topdir, d))):
                if ds not in daysdone:
                    indir = os.path.join(self.topdir, ds)
                    ind = AR.readDirIndex(indir)
                    self.printLine(AR.getIndex_fromstring(ind), color='blue')
                    self.printLine('            {0:16s} : {1:20s}{2:s}'.format('-'*16, '-'*20, self.lb))             
                    daysdone.append(d)
                if s in ['.index', '.DS_Store', 'log.txt'] or self.check_extensions(s):
                    continue


                tstamp = self.gettimestamp(os.path.join(self.topdir, d, s))
                if any([s.endswith(e) for e in ['.tif', '.ma']]):
                    st = os.stat(os.path.join(self.topdir, d, s))  # unmanaged (though may be in top index file)
                    tstamp = datetime.datetime.fromtimestamp(st[stat.ST_MTIME]).strftime('%Y-%m-%d  %H:%M:%S %z')
                    self.printLine((fmtstring + 'data file not associated with slice or cell').format('', s, '', '', tstamp), color='cyan')
                    continue
                if s.startswith('slice_'):
                    self.printLine(fmtstring.format('', s, '', '', tstamp))
                    indir = os.path.join(self.topdir, d, s)
                    ind = AR.readDirIndex(indir)
                    self.printLine(AR.getIndex_fromstring(ind))
                else:
                    self.printLine((fmtstring + '   is not a SLICE directory').format('', s, '', '', tstamp), color='red')
                
                for c in sorted(os.listdir(os.path.join(self.topdir, d, s))):
                    if c in ['.index', '.DS_Store', 'log.txt'] or self.check_extensions(c):
                        continue
                    tstamp = self.gettimestamp(os.path.join(self.topdir, d, s, c))
                    if c.startswith('cell_'):
                        self.printLine(self.lb+fmtstring.format('', '', c, '', tstamp))
                        indir = os.path.join(self.topdir, d, s, c)
                        try:
                            ind = AR.readDirIndex(indir)
                        except:
                            self.printLine((fmtstring2 + 'Broken Index file'+self.lb).format('', '', c, '', tstamp), color='red')
                            self.printLine((fmtstring2 + 'File: '+self.lb).format('', '', indir, '', ''))
                            continue
                        self.printLine(AR.getIndex_fromstring(ind))
                    else:
                        self.printLine((fmtstring2 + 'is not a CELL directory'+self.lb).format('', '', c, '', tstamp), color='red')
                        continue
                    protodir = os.path.join(self.topdir, d, s, c)
                    for pr in sorted(os.listdir(protodir)):  # all protocols
                        if pr in ['.DS_Store', 'log.txt'] or self.check_extensions(pr):
                            continue
                        if pr in ['.index']:
                            stx = os.stat(os.path.join(protodir, pr))
                            if stx.st_size == 0:
                                self.printLine('   {0:s} is empty'.format(pr), color='red')
                            continue
                        if any([pr.endswith(e) for e in ['.tif', '.ma']]):
                            continue
                        tstamp = self.gettimestamp(os.path.join(self.topdir, d, s, c, pr))
                        self.printLine(fmtstring.format('', '', '', pr, tstamp))
                        if self.show_protocol:
                            indir = os.path.join(self.topdir, d, s, c, pr)
                            ind = AR.readDirIndex(indir)
                            self.printLine(AR.getIndex_fromstring(ind))
                            self.printLine('              -----------------------'+self.lb)
                        for f in os.listdir(os.path.join(protodir, pr)):  # for all runs in the protocol directory
                            if f in ['.DS_Store', 'log.txt']:
                                continue
                            protodatafile = os.path.join(protodir, pr, f)
                            if os.path.isdir(protodatafile):  # directory  - walk it
                                for f2 in os.listdir(protodatafile):  # for all runs in the protocol directory
                                    if f2 in ['.DS_Store', 'log.txt']:
                                        continue
                                    stx = os.stat(os.path.join(protodatafile, f2))
                                    # print('***** F2: ', f2, stx.st_size)
                                    if stx.st_size == 0:
                                        self.printLine('   {0:s} is empty'.format(os.path.join(protodatafile, f2)), color='cyan')
                                        raise ValueError('File with 0 length is wrong - check data transfers')
                            elif os.path.isfile(protodatafile):  # is a file
                                stx = os.stat(protodatafile)
                                # print('***** F: ', protodatafile, stx.st_size)
                                if stx.st_size == 0:
                                    self.printLine('   {0:s} is empty'.format(protodatafile), color='red')
                                    raise ValueError('File with 0 length is wrong - check data transfers')

                            
            #self.printLine('\f')  # would be nice to put a page break here, but ... doesn't survive. 
        self.printLine('-'*100+self.lb, color='blue')
        if self.outputMode == 'latex':
            with open(self.outfile, 'a') as f:
                f.write(latex_footer)
        
    def check_extensions(self, d):
        return(any([d.endswith(e) for e in ['.p', '.pkl', '.sql',
            '.txt', '.doc', '.docx', '.xlsx', '.py', '.tex', '.cfg', '.ini',
            '.tif', '.tiff', '.jpg', '.jpeg', '.gif', '.png', '.pdf',
            '.ma', '.mosaic']]))
    
 #   def show_index(self, )
    def gettimestamp(self, path):
        """
        Get the timestamp of an .index file
        """
        tstamp = 'None'
        indexfile = os.path.join(path, '.index')
        if not os.path.isfile(indexfile):
            return tstamp
        with open(indexfile, 'r') as f:
            for l in f:
                ts = self.tstamp.match(l)
                if ts is not None:
                    fts = float(ts.group(2))
                    tstamp = datetime.datetime.fromtimestamp(fts).strftime('%Y-%m-%d  %H:%M:%S %z')
#                            print('time: ', tstamp)
        return tstamp

    def printLine(self, text, color=None):
        # if self.outputMode == 'latex':
        #     text = text.replace('_', '$\_$')
        if self.outfile is None:
            if self.coloredOutput:
                if color==None:
                    color='white'
                print(colored(text, color))
            else:
                print(text)
        else:
            with open(self.outfile, 'a') as f:
                if self.outputMode == 'latex' and color != None:
                    f.write(self.lb + '\\end{Verbatim} ' + self.lb+ '\\begin{Verbatim}[formatcom=\\color{%s}]'%color+self.lb)
                f.write(text)
                if self.outputMode == 'latex' and color != None:
                    f.write(self.lb + '\\end{Verbatim} ' + self.lb + '\\begin{Verbatim} ' + self.lb)
 


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate Data Summaries from acq4 datasets')
    parser.add_argument('basedir', type=str,
                        help='Base Directory')
    parser.add_argument('-r', '--read', action='store_true', dest='read',
                        help='just read the protocol')
    parser.add_argument('-p', '--show-protocol', action='store_true', dest='protocol',
                        help='Print protocol information (normally suppressed, very verbose)')
    parser.add_argument('-o', '--output', type=str, dest='output', default=None,
                        help='Designate an output file name (if .tex, generates a LaTeX file)')
    parser.add_argument('-a', '--after', type=str, default='1970.1.1', dest='after',
                        help='only analyze dates on or after a date')
    parser.add_argument('-b', '--before', type=str, default='2266.1.1', dest='before',
                        help='only analyze dates on or before a date')
    args = parser.parse_args()

    DC = DirCheck(args.basedir, args.protocol, args.output, args=args)
    
    # remove latex intermediate files that are not needed after pdf generation
    p, e = os.path.splitext(DC.outfile)
    if e != '.tex':
        exit(0)
    subprocess.call(["pdflatex", DC.outfile])
    exts = ['.aux', '.log', '.dvi', '.ps']
    for filename in [p+e for e in exts]:
        with contextlib.suppress(FileNotFoundError):
            os.remove(filename)
    subprocess.call(['open', p+'.pdf'])
    