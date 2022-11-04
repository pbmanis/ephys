__author__ = "pbmanis"
"""
dir_check.py
Check an acq4 directory struture to be sure that the hierarchy is correct.
Prints out a TeX file that summarizes all of the files that are available.
Use TeXShop or similar to turn this into a PDF for reference.
This is also useful as documentation of what data is actually under
a given file structure

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
import argparse
import contextlib
import datetime
import os
import os.path
import re
import stat
import subprocess
import sys

# import textwrap

from pathlib import Path
from typing import List, Union

import dateutil.parser as DUP

from ephys.datareaders import acq4read
from termcolor import colored

"""
Make a LaTex Header and Footer for the output
"""
latex_header = """\\documentclass[8pt, letterpaper, oneside]{article}
\\usepackage[utf8]{inputenc}
\\usepackage{fancyvrb}
\\usepackage{geometry}
\\usepackage[dvipsnames]{xcolor}
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

% \\end{Verbatim}
\\end{document}
"""


class Printer:
    """Print things to stdout on one line dynamically"""

    def __init__(self, data: object):
        sys.stdout.write("\033[1;36m\r\x1b[K\033[0;0m" + data.__str__())
        sys.stdout.flush()


class DirCheck:
    def __init__(
        self,
        topdir: Path,
        protocol: bool = False,
        output: str = None,
        args: object = None,
    ):
        """
        Check directory structure
        """
        if str(topdir).endswith(os.path.sep):
            topdir = topdir[:-1]
        self.topdir = topdir
        if not self.topdir.is_dir():
            raise ValueError("Top directory not found: %s" % str(self.topdir))

        self.outfile = None
        self.coloredOutput = True
        self.outputMode = "text"
        self.after = DUP.parse(args.after)
        self.before = DUP.parse(args.before)
        self.just_list_dirs = False
        self.lb = "\n"  # define linebreak

        self.show_protocol = protocol
        self.img_re = re.compile(
            "^[Ii]mage_(\d{3,3}).tif"
        )  # make case insensitive - for some reason in Xuying's data
        self.s2p_re = re.compile("^2pStack_(\d{3,3}).ma")
        self.i2p_re = re.compile("^2pImage_(\d{3,3}).ma")
        self.video_re = re.compile("^[Vv]ideo_(\d{3,3}).ma")
        self.daytype = re.compile("(\d{4,4}).(\d{2,2}).(\d{2,2})_(\d{3,3})")
        self.tstamp = re.compile("\s*(__timestamp__: )([\d.\d]*)")
        self.AR = acq4read.Acq4Read()
        self.recurselevel = 0
        self.fmtstring = "{0:>15s} {1:<10s} {2:<10s} {3:<40} {4:>20}"
        self.fmtstring2 = "{0:>15s} {1:<10s} {2:<40s} {3:<10} {4:>20}"
        # if self.outfile is not None:
        #     self.fmtstring += self.lb  # insert newlines when writing output to file


        if output is not None:
            self.coloredOutput = False
            self.outfile = output
            Path(self.outfile).unlink(missing_ok=True)

            e = Path(self.outfile).suffix
            if e == ".tex":  # write latex header and set up
                self.outputMode = "latex"
                with open(self.outfile, "w") as f:
                    f.write(latex_header)

            if self.outputMode == "latex":
                #    self.printLine('\\end{Verbatim}' + self.lb)
                print("Writing title page")
                self.printLine(
                    "\\vspace{2cm}\\center{\\textbf{\\large{Directory Check/Protocol Listing}}}"
                    + self.lb
                )
                self.printLine(
                    "\\vspace{2cm}\\center{\\textbf{\huge{"
                    + str(self.outfile) # .replace("_", "\_")
                    + "}}}"
                    + self.lb
                )
                self.printLine(
                    "\\vspace{1cm}\\center{\\textbf{\\large{"
                    + str(self.topdir) # .replace("_", "\_")
                    + "}}}"
                    + self.lb
                )
                self.now = datetime.datetime.now().strftime("%Y-%m-%d  %H:%M:%S %z")
                self.rundate = datetime.datetime.now().strftime("%Y.%m.%d")
                self.printLine(
                    "\\vspace{1cm}\\center{\\textbf{\\LARGE{Run Date:}}}" + self.lb
                )
                self.printLine(
                    "\\vspace{0.5cm}\\center{\\textbf{\\huge{" + self.now + "}}}" + self.lb
                )
                self.printLine("\\newpage", color=None)
                # self.printLine("\\begin{Verbatim} " + self.lb)

        path = self.topdir.parent
        lastdir = self.topdir.stem
        # path, lastdir = os.path.split(self.topdir)
        td = self.daytype.match(lastdir)
        # print(path, td)
        if td is not None:
            topdirs = [lastdir]
            self.topdir = path
        else:
            topdirs = list(self.topdir.glob("*"))




        self.do_dirs(topdirs, current_top_dir=self.topdir)

        # self.printLine('\f')  # would be nice to put a page break here, but ... doesn't survive.
        # finally, write the footer
        self.printLine("-" * 100 + self.lb, color="blue")
        # if self.outputMode == "latex" and output is not None:
        #     print("writing footer")
        #     with open(self.outfile, "a") as f:
        #         f.write(latex_footer)
        if self.outputMode == "latex" and output is not None:
            with open(self.outfile, "a") as f:
                f.write(latex_footer)
        self.tex_out()  # print and then format

    def _check_name(self, dname, ndir, parent_dir):
        dpath = Path(dname)
        if str(dpath.name) in [".DS_Store", "log.txt" "."] or self.check_extensions(
            dpath
        ):
            return None
        if any([dpath.name.endswith(e) for e in [".tif", ".ma"]]):
            return None
        if dpath.name.startswith("Accidental"):
            return None
        if dpath.name in [".index"]:
            indir = os.path.join(parent_dir, dpath)
            ind = self.AR.readDirIndex(parent_dir)
            if ndir > 0:
                self.printLine(self.AR.getIndex_text(ind))
            return None
        else:
            return dpath

    def do_dirs(self, topdirs, current_top_dir):
        self.recurselevel += 1
        # print("recurse: ", self.recurselevel, len(topdirs), current_top_dir)
        for ndir, d in enumerate(
            sorted(list(topdirs))
        ):  # go throgu the list of top directories
            dpath = self._check_name(
                d, ndir, current_top_dir
            )  # skip certain kinds of names
            if dpath is None:
                continue

            daystring = self.daytype.match(str(dpath.name))  # check if name is a
            # print("daystring: ", daystring, "is dir: ", dpath.is_dir())
            if daystring is None and d.is_dir():
                # consider recursively going into this directory to look for more data files
                topdirs = [fdir for fdir in d.iterdir() if fdir.is_dir()]
                self.do_dirs(topdirs, current_top_dir=d)

            else:  # check filtering by date
                thisday = datetime.datetime.strptime(dpath.name[:10], "%Y.%m.%d")
                if thisday < self.after or thisday > self.before:
                    continue
            # if self.just_list_dirs:
            #     print("doing dir: ", dpath.parent, dpath.name)
            #     continue
            
            daydir = dpath.name
            dsday = dpath.name[:10]

            self.printLine(" ")

            tstamp = self.gettimestamp(Path(current_top_dir, daydir))
            if ndir > 1:
                # self.printLine(self.lb + "\\end{Verbatim}" + self.lb)
                self.printLine("+" * 80 + self.lb + "\\newpage" + self.lb)
               # self.printLine("\\begin{Verbatim} " + self.lb)
            if daystring is not None:
                self.printLine(self.fmtstring.format(str(d), "", "", "", tstamp))
            else:
                self.printLine(
                    (self.fmtstring + "is not a DAY directory").format(
                        str(daydir), "", "", "", tstamp
                    ),
                    color="red",
                )
                self.printLine(
                    (self.fmtstring + "Daystring: ").format(
                        str(daystring), "", "", "", tstamp
                    ),
                    color="red",
                )
            daysdone = []

            for slicedir in sorted(
                Path(current_top_dir, daydir).iterdir()
            ):  # go through the directory
                if daydir not in daysdone:
                    indir = Path(current_top_dir, daydir)
                    ind = self.AR.readDirIndex(indir)
                    self.printLine(self.AR.getIndex_text(ind), color="blue", verbatim=True)
                    self.printLine(
                        "            {0:16s} : {1:20s}{2:s}".format(
                            "-" * 16, "-" * 20, self.lb
                        )
                    )
                    daysdone.append(daydir)
                if slicedir.name in [
                    ".index",
                    ".DS_Store",
                    "log.txt",
                ] or self.check_extensions(slicedir):
                    continue

                if any([slicedir.name.endswith(e) for e in [".tif", ".ma"]]):
                    st = os.stat(
                        Path(current_top_dir, daydir, slicedir)
                    )  # unmanaged (though may be in top index file)
                    tstamp = datetime.datetime.fromtimestamp(
                        st[stat.ST_MTIME]
                    ).strftime("%Y-%m-%d  %H:%M:%S %z")
                    self.printLine(
                        (
                            self.fmtstring
                            + "data file not associated with slice or cell"
                        ).format("", str(slicedir), "", "", tstamp),
                        color="red",
                    )
                    continue
                if any([slicedir.name.endswith(e) for e in [".db"]]):
                    continue
                # if self.just_list_dirs:
                #     continue
                if slicedir.name.startswith("slice_"):
                    slicepath = Path(current_top_dir, daydir, slicedir)
                    tstamp = self.gettimestamp(slicepath)
                    self.printLine(self.fmtstring.format("", str(slicedir), "", "", tstamp), "teal")
                    a4index = self.AR.readDirIndex(slicepath)
                    self.printLine(self.AR.getIndex_text(a4index), verbatim=True)
                else:
                    self.printLine(
                        (self.fmtstring + "   is not a SLICE directory").format(
                            "", str(slicedir), "", "", tstamp
                        ),
                        color="red",
                    )

                for celldir in sorted(slicepath.iterdir()):
                    if celldir.name in [
                        ".index",
                        ".DS_Store",
                        "log.txt",
                    ] or self.check_extensions(celldir):
                        continue
                    if celldir.name.startswith("cell_"):
                        cellpath = Path(current_top_dir, daydir, slicedir, celldir)
                        tstamp = self.gettimestamp(cellpath)
                        self.printLine(
                            self.lb + self.fmtstring.format("", str(celldir), "", "", tstamp), "violet",
                        )
                        try:
                            a4index = self.AR.readDirIndex(cellpath)
                        except:
                            self.printLine(
                                (
                                    self.fmtstring2
                                    + "  *** Broken Index file"
                                    + self.lb
                                ).format("", "", str(celldir), "", tstamp),
                                color="red",
                            )
                            self.printLine(
                                (self.fmtstring2 + "File: " + self.lb).format(
                                    "", "", str(indir), "", ""
                                )
                            )
                            continue
                        self.printLine(self.AR.getIndex_text(a4index), verbatim=True)
                    else:
                        self.printLine(
                            (
                                self.fmtstring2 + "is not a CELL directory" + self.lb
                            ).format("", "", str(celldir), "", tstamp),
                            color="red",
                        )
                        continue
                    protodir = Path(current_top_dir, daydir, slicedir, celldir)
                    for protocol in sorted(protodir.iterdir()):  # all protocols
                        if protocol.name in [".DS_Store", "log.txt"] or self.check_extensions(
                            protocol
                        ):
                            continue
                        if protocol.name in [".index"]:
                            stx = os.stat(Path(protodir, protocol))
                            if stx.st_size == 0:
                                self.printLine(
                                    "   {0:s} is empty".format(str(protocol)), color="red"
                                )
                            continue
                        if any([protocol.name.endswith(e) for e in [".tif", ".ma"]]):
                            continue
                        protopath = Path(current_top_dir, daydir, slicedir, celldir, protocol)
                        tstamp = self.gettimestamp(protopath)
                        self.printLine(
                            self.fmtstring.format("", "", "", protocol.name, tstamp), verbatim=True, suppress_trailing_lb=True
                        )
                        if self.show_protocol:
                            a4index = self.AR.readDirIndex(protopath)
                            self.printLine(self.AR.getIndex_text(a4index), verbatim=True)
                            self.printLine(
                                "              -----------------------" + self.lb
                            )
                        for f in sorted(
                            Path(protodir, protocol).iterdir()
                        ):  # for all runs in the protocol directory
                            if f.name in [".DS_Store", "log.txt"]:
                                continue
                            protodatafile = Path(protodir, protocol, f)
                            if protodatafile.is_dir():  # directory  - walk it
                                for f2 in sorted(
                                    protodatafile.iterdir()
                                ):  # for all runs in the protocol directory
                                    if f2.name in [".DS_Store", "log.txt"]:
                                        continue
                                    stx = os.stat(Path(protodatafile, f2))
                                    # print('***** F2: ', f2, stx.st_size)
                                    if stx.st_size == 0:
                                        self.printLine(
                                            "   {0:s} is empty".format(
                                                str(Path(protodatafile, f2))
                                            ),
                                            color="cyan",
                                        )
                                        raise ValueError(
                                            "File with 0 length is wrong - check data transfers"
                                        )
                            elif protodatafile.is_file():  # is a file
                                stx = os.stat(protodatafile)
                                # print('***** F: ', protodatafile, stx.st_size)
                                if stx.st_size == 0:
                                    self.printLine(
                                        "   {0:s} is empty".format(protodatafile),
                                        color="red",
                                    )
                                    raise ValueError(
                                        "File with 0 length is wrong - check data transfers"
                                    )

    def check_extensions(self, d: str):
        """Filter out files with extraneous extensions

        Args:
            d (str): file name to look for extensions...
        """
        if isinstance(d, Path):
            d = str(d.name)
        return any(
            [
                d.endswith(e)
                for e in [
                    ".p",
                    ".pkl",
                    ".sql",
                    ".txt",
                    ".doc",
                    ".docx",
                    ".xlsx",
                    ".py",
                    ".tex",
                    ".cfg",
                    ".ini",
                    ".tif",
                    ".tiff",
                    ".jpg",
                    ".jpeg",
                    ".gif",
                    ".png",
                    ".pdf",
                    ".ma",
                    ".mosaic",
                    ".db",
                ]
            ]
        )

    #   def show_index(self, )
    def gettimestamp(self, path):
        """
        Get the timestamp of an .index file
        """
        tstamp = "None"
        indexfile = os.path.join(path, ".index")
        if not os.path.isfile(indexfile):
            return tstamp
        with open(indexfile, "r") as f:
            for l in f:
                ts = self.tstamp.match(l)
                if ts is not None:
                    fts = float(ts.group(2))
                    tstamp = datetime.datetime.fromtimestamp(fts).strftime(
                        "%Y-%m-%d  %H:%M:%S %z"
                    )
        #                            print('time: ', tstamp)
        return tstamp

    def printLine(self, text, verbatim=False, suppress_trailing_lb=False, color=None):
        # if self.outputMode == 'latex':
        if not verbatim:
            text = text.replace('_', '\_')
        if self.outfile is None:
            if self.coloredOutput:
                if color == None:
                    color = "white"
                if color in ["teal", "violet", "orange"]:
                    color = "cyan"
                print(colored(text, color))
            else:
                print(text)
            return
        if suppress_trailing_lb:
            lb2 = ""
        else:
            lb2 = self.lb
            
        with open(self.outfile, "a") as f:
            if self.outputMode == "latex":
                if color != None and color != "red":
                    if verbatim:
                        f.write(
                            self.lb + "{\\color{" + f"{color:s}" + "}"
                            + "\\begin{Verbatim}" + self.lb                            
                            + self.lb + text  +  self.lb
                            + "\\end{Verbatim}" + self.lb + "}"
                        )
                    else:
                        f.write(
                            # + "{\\color{%s}" % color
                            self.lb + text +  self.lb
                        )      
                elif color == "red":
                        f.write(self.lb+"{\\colorbox{"+f"{color:s}"+"}"+self.lb+"{\\framebox{"+self.lb+ "\\begin{minipage}{1.0\\linewidth} " +
                                self.lb + text+ self.lb + "\\end{minipage}}" +self.lb + "}")
                elif color == None:
                    if verbatim:
                        f.write(
                            self.lb 
                            + "\\begin{Verbatim} " + self.lb
                            + text  
                            + self.lb + "\\end{Verbatim}"
                        )
                    else:
                        f.write(
                            self.lb 
                             + text  
                            + self.lb
                        )

                else:
                    if verbatim:
                        f.write(
                            self.lb
                            + "\\begin{Verbatim} "
                            + self.lb
                            + text
                            + self.lb + "\\end{Verbatim}"
                        )
                    else:
                        f.write(
                            self.lb
                            + text
                            + self.lb
                        )


    # remove latex intermediate files that are not needed after pdf generation
    def tex_out(self):
        if self.outfile is None:
            return
        e = Path(self.outfile).suffix
        if e != ".tex":  # Require a .tex file!
            return
        p = Path(self.outfile).stem
        # insert the rundate into the filename
        p = f"{p:s}_{self.rundate:s}_dircheck"
        subprocess.call(["pdflatex", p + e])
        exts = [".aux", ".log", ".dvi", ".ps"]
        for filename in [Path(p + e) for e in exts]:
            with contextlib.suppress(FileNotFoundError):
                # os.remove(filename)
                filename.unlink(missing_ok=True)
        subprocess.call(["open", p + ".pdf"])


def main():
    parser = argparse.ArgumentParser(
        description="Generate Data Summaries from acq4 datasets"
    )
    parser.add_argument("basedir", type=str, help="Base Directory")
    parser.add_argument(
        "-r", "--read", action="store_true", dest="read", help="just read the protocol"
    )
    parser.add_argument(
        "-p",
        "--show-protocol",
        action="store_true",
        dest="protocol",
        help="Print protocol information (normally suppressed, very verbose)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        dest="output",
        default=None,
        help="Designate an output file name (if .tex, generates a LaTeX file)",
    )
    parser.add_argument(
        "-a",
        "--after",
        type=str,
        default="1970.1.1",
        dest="after",
        help="only analyze data from on or after a date",
    )
    parser.add_argument(
        "-b",
        "--before",
        type=str,
        default="2266.1.1",
        dest="before",
        help="only analyze data from on or before a date",
    )
    args = parser.parse_args()

    DC = DirCheck(Path(args.basedir), args.protocol, args.output, args=args)


if __name__ == "__main__":
    main()
