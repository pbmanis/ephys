"""
Module for analyzing datasets with optogenetic or uncaging laser mapping
current-voltage relationships, and so on... 
Used as a wrapper for multiple experiments. 
"""
from multiprocessing import set_start_method

import os
import sys

if sys.version_info[0] < 3:
    raise Exception("Must be using Python 3")
import datetime
import json
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Type, Union

import dateutil.parser as DUP
import dill
import matplotlib
import numpy as np
import pandas as pd
import toml

matplotlib.use("QtAgg")
import pylibrary.plotting.plothelpers as PH
import pylibrary.tools.cprint as CP
import pyqtgraph as pg
import pyqtgraph.console as console
import pyqtgraph.multiprocess as mp
from matplotlib.backends.backend_pdf import PdfPages

import ephys.ephys_analysis as EP
import ephys.ephys_analysis.IV_Analysis_Params as AnalysisParams
import ephys.mapanalysistools as mapanalysistools

PMD = mapanalysistools.plotMapData.PlotMapData()

np.seterr(divide="raise", invalid="raise")

@dataclass
class cmdargs:
    experiment: Union[str, list, None] = None
    basedir: Union[str, Path, None] = None
    exptdir : Union[str, Path, None] = None
    inputFilename: Union[str, Path, None] = None
    outputFilename: Union[str, Path, None] = None
    outputPath: Union[str, Path, None] = None
    annotationFile: Union[str, Path, None] = None
    artifactFilename: Union[str, Path, None] = None
    map_annotationFile: Union[str, Path, None] = None
    
    day: str = "all"
    after: str = "1970.1.1"
    before: str = "2266.1.1"
    slicecell: str = ""
    protocol: str = ""
    configfile: Union[str, None] = None
    iv_flag: bool=False
    vc_flag: bool=False
    map_flag: bool=False
    merge_flag: bool=False
    dry_run: bool=False
    verbose: bool=False
    autoout: bool=False
    excel: bool=False
    # graphics controls
    plotmode: str = "document"
    IV_pubmode:str = "normal"
    rasterize:bool = True
    update:bool = False
    noparallel:bool=True
    replot:bool=False

    # analysis parameters
    ivduration:float = 0.0
    threshold:float=2.5  # cb event detection threshold (x baseline std)
    signflip:bool = False
    alternate_fit1: bool=False 
    alternate_fit2: bool = False  # second alternate
    measuretype: str = "ZScore"  # display measure for spot plot in maps
    spike_threshold: float=-0.035
    artifact_suppression:bool=False
    noderivative:bool=False
    whichstim:int = -1
    trsel: Union[int, None] = None
    notchfilter:bool=False
    notchfreqs:str = "60, 120, 180, 240"
    LPF:float = 0.0
    HPF: float = 0.0
    notchQ: float=90.0
    detector:str = 'cb'    


class IV_Analysis(object):
    def __init__(self, args:object):

        #args = vars(in_args)  # convert to dict

        set_start_method("spawn")
        self.basedir = None
        self.exptdir = None
        self.inputFilename = None
        self.outputFilename = None
        self.outputPath = None
        self.annotationFile = None
        self.map_annotationFile = None
        self.artifactFilename = None
        self.exclusions = None

        self.experiment = args.experiment
        
        # selection of data for analysis
        self.day = args.day
        self.before = DUP.parse(args.before)
        self.after = DUP.parse(args.after)
        self.slicecell = args.slicecell
        self.protocol = args.protocol
        
        # modes for analysis
        self.iv_flag = args.iv_flag
        self.vc_flag = args.vc_flag
        self.map_flag = args.map_flag
        
        # output controls
        self.merge_flag = args.merge_flag

        self.dry_run = args.dry_run
        self.verbose = args.verbose
        self.autoout = args.autoout
        self.excel = args.excel
        # graphics controls
        self.plotmode = args.plotmode
        self.IV_pubmode = args.IV_pubmode
        self.rasterize = True
        self.update = args.update
        self.noparallel = args.noparallel
        self.replot = args.replot

        # analysis parameters
        self.ivduration= args.ivduration
        self.threshold = args.threshold
        self.signflip = args.signflip
        self.alternate_fit1 = (
            args.alternate_fit1
        )  # use alterate time constants in template for event
        self.alternate_fit2 = args.alternate_fit2  # second alternate
        self.measuretype = args.measuretype  # display measure for spot plot in maps
        self.spike_threshold = args.spike_threshold
        self.artifact_suppress = args.artifact_suppression
        self.noderivative_artifact = args.noderivative
        self.whichstim = args.whichstim
        self.trsel = args.trsel
        self.notchfilter = args.notchfilter
        self.notchfreqs = eval(args.notchfreqs)
        self.LPF = args.LPF
        self.HPF = args.HPF
        self.notch_Q = args.notchQ
        self.detector = args.detector

        self.tempdir = None
        self.annotated_dataframe: Union[pd.DataFrame, None] = None
        self.allprots = []

        self.AM = mapanalysistools.analyzeMapData.AnalyzeMap(rasterize=self.rasterize)
        if self.detector == 'cb':
            self.AM.set_methodname("cb_cython")
        elif self.detector == 'aj':
            self.AM.set_methodname("aj_cython")
        elif self.detector == 'zc':
            raise ValueError("Detector ZC is not recommended at this time")
        else:
            raise ValueError(f"Detector {self.detector:s} is not one of [cb, aj, zc")

        self.setup()
    
    def set_exclusions(self, exclusions: Union[dict, None] = None):
        self.exclusions = exclusions

    def set_experiment(self, expt:dict):
        self.experiment = expt

    def setup(self):
        if self.experiment not in ["None", None]:
            self.basedir = Path(self.experiment["datadisk"])
            self.exptdir = Path(self.basedir)
            self.inputFilename = Path(self.experiment['db_directory'], self.experiment['datasummary']
                ).with_suffix(".pkl")
            self.outputPath = self.exptdir
            if self.experiment['outputFilename'] is not None:
                self.outputFilename = Path(self.outputPath, self.experiment['outputFilename']).with_suffix(
                ".pdf")
                self.outputFilename.unlink(missing_ok=True)

            if self.experiment["annotation"] is not None:
                self.annotationFile = Path(
                    self.exptdir, self.experiment["annotation"]
                )
            else:
                self.annotationFile = None
            
            if 'maps' in list(self.experiment.keys()) and self.experiment["maps"] is not None:
                self.map_annotationFile = Path(
                    self.exptdir, self.experiment["maps"]
                )
            else:
                self.map_annotationFile = None
        else:
            raise ValueError(
                'Experiment was not specified"'
            )

        if self.artifactFilename is not None and len(self.artifactFilename) > 0:
            self.artifactFilename = self.artifactFilename
        else:
            self.artifactFilename = None


        # get the input file (from dataSummary)
        self.df = pd.read_pickle(str(self.inputFilename))
        if self.excel:  # just re-write as an excel and we are done
            excel_file = Path(self.outputPath, self.inputFilename.stem + ".xlsx")
            print(f"Writing to {str(excel_file):s}" )
            self.df.to_excel(excel_file)
            exit(0)

        self.iv_select = {
            "duration": self.ivduration
        }  # dictionary of selection flags for IV data

        if self.verbose:
            dates = self.df["date"]
            alld = [date for date in dates]
            print("Dates found in input file (excel): \n", alld)

        # pull in the annotated data (if present) and update the cell type in the main dataframe
        self.df["annotated" ] = False   # clear annotations
        if self.annotationFile is not None:
            print(f"Reading annotation file: {str(self.annotationFile):s}")
            ext = self.annotationFile.suffix
            if ext in [".p", ".pkl", ".pkl3"]:
                self.annotated_dataframe = pd.read_pickle(Path(self.basedir, self.annotationFile))
            elif ext in [".xls", ".xlsx"]:
                self.annotated_dataframe = pd.read_excel(self.annotationFile)
            else:
                raise ValueError(
                    f"Do not know how to read annotation file: {str(self.annotationFile):s}, Valid extensions are for pickle and excel"
                )
            self.update_annotations()

        if self.outputFilename is not None:
            self.outputFilename.unlink(missing_ok=True)
    
    def run(self):
    # select a date:
        if self.day != "all":  # specified day
            day = str(self.day)
            print(f"Looking for day: {day:s} in {str():s}")
            if "_" not in day:
                day = day + "_000"
            day_x = self.df.loc[self.df["date"] == day]
            print("  ... Retrieved day: ", day_x)
            for iday in day_x.index:
                self.do_day(iday, 0)

        # get the complete protocols:
        # Only returns a dataframe if there is more than one entry
        # Otherwise, it is like a series or dict
        else:
            if self.outputFilename is None:
                for n, iday in enumerate(range(len(self.df.index))):
                    self.do_day(iday, n)
            else:
                with PdfPages(self.outputFilename) as pdf:
                    for n, iday in enumerate(range(len(self.df.index))):
                        self.do_day(iday, n, pdf)
                

        if self.update:
            n = datetime.datetime.now()  # get current time
            dateandtime = n.strftime(
                "_%Y%m%d-%H%M%S"
            )  # make a value indicating date and time for backup file
            h = random.getrandbits(32)
            dateandtime = (
                dateandtime + f"_{h:08x}"
            )  # add a random hash to end of string as well
            if self.inputFilename.is_file():
                self.inputFilename.rename(
                    Path(
                        self.inputFilename.parent,
                        str(self.inputFilename.name) + dateandtime,
                    ).with_suffix(".bak")
                )
            self.df.to_pickle(str(self.inputFilename))

    def update_annotations(self):
        if self.annotated_datframe is not None:
            self.annotated_dataframe.set_index("ann_index", inplace=True)
            x = self.annotated_dataframe[self.annotated_dataframe.index.duplicated()]
            if len(x) > 0:
                print("watch it - duplicated index in annotated file")
                print(x)
                exit()
            # self.annotated_dataframe.set_index("ann_index", inplace=True)
            self.df.loc[
                self.df.index.isin(self.annotated_dataframe.index), "cell_type"
            ] = self.annotated_dataframe.loc[:, "cell_type"]
            self.df.loc[self.df.index.isin(self.annotated_dataframe.index), "annotated"] = True
            if self.verbose:  # check whether it actually took
                for icell in range(len(self.df.index)):
                    print(
                        "{0:<32s}  type: {1:>20s}, annotated (T,F): {2!s:>5} Index: {3:d}".format(
                            str(self.make_cellstr(self.df, icell)),
                            self.df.iloc[icell]["cell_type"],
                            str(self.df.iloc[icell]["annotated"]),
                            icell,
                        )
                    )
        # pull in map annotations. Thesea are ALWAYS in an excel file, which should be created initially by make_xlsmap.py
        if self.map_annotationFile is not None:
            self.map_annotations = pd.read_excel(
                Path(self.map_annotationFile).with_suffix(".xlsx"), sheet_name="Sheet1"
            )
            print("Reading map annotation file: ", self.map_annotationFile)

    def make_cellstr(self, df: object, iday: int, sep: str = os.sep):
        """
        Make a day string including slice and cell from the iday index in the pandas datafram df
        Example result:
            r = self.make_cellstr (df, 1, sep=';')
            r: '2017.01.01_000;slice_000;cell_001'

            s = self.make_cellstr (df, 1)
            s: '2017.01.01_000/slice_000/cell_001'  # Mac/linux path string

        Parameters
        ----------
        df : Pandas dataframe instance

        iday : int (no default)
            index into pandas dataframe instance

        sep : str (default: os.sep)
            separator to use between date, slice and cell

        returns
        -------
        Path
            """
        daystr = Path(
            df.iloc[iday]["date"],
            df.iloc[iday]["slice_slice"],
            df.iloc[iday]["cell_cell"],
        )
        return daystr

    """
    Handle the temporary directory pdf accumulation and merging.
    """

    def make_tempdir(self):
        """
        Make a temporary directory; if the directory exists, just clean it out
        """
        self.tempdir = Path("./temppdfs")
        if not self.tempdir.is_dir():
            self.tempdir.mkdir(mode=0o755, exist_ok=True)
        else:
            self.clean_tempdir()  # clean up

    def clean_tempdir(self):
        """
        Delete the files in the current temporary directory
        """
        fns = sorted(
            list(self.tempdir.glob("*.pdf"))
        )  # list filenames in the tempdir and sort by name
        for fn in fns:  # delete the files in the tempdir
            os.remove(fn)

    def merge_pdfs(self, celltype: Union[str, None] = None):
        """
        Merge the PDFs in tempdir with the outputfile (self.outputFilename)
        The tempdir PDFs are deleted once the merge is complete.
        """
        if (
            self.outputFilename is None and not self.autoout
        ):  # no output file, do nothing
            return
        if self.dry_run:
            return
        # check autooutput and reset outputfilename if true:
        CP.cprint("c",
            f"Merging pdfs at {datetime.datetime.now().strftime('%m/%d/%Y, %H:%M:%S'):s}"
        )
        if self.merge_flag:
            self.tempdir = Path("./temppdfs")  # use standard and do not clear it
        if self.autoout:
            print("  in auto mode")
            expt = (
                experiments[self.experiment]["directory"]
                + "_"
                + self.day.replace(".", "_")
            )
            expt += "_" + self.slicecell
            if self.signflip:
                expt += "_signflip"
            if self.alternate_fit1:
                expt += "_alt1"
            if self.alternate_fit2:
                expt += "_alt2"

            if celltype is None:
                celltype = "unknown"
            expt += "_" + celltype
            if self.iv_flag:
                expt += f"_IVs"
            elif self.replot:
                expt += f"_maps"  # tag if using other than zscore in the map plot
            
            expt = Path(expt)
            print("   expt: ", expt)
            # check to see if we have a sorted directory with this cell type
            outputdir = Path(self.outputPath, celltype)
            if not outputdir.is_dir():
                outputdir.mkdir()
            self.outputFilename = Path(outputdir, expt.stem).with_suffix(".pdf")
            print('output path: ', self.outputPath)
            print('output dir: ', outputdir)
            print('output file: ', self.outputFilename)
        fns = sorted(
            list(self.tempdir.glob("*.pdf"))
        )  # list filenames in the tempdir and sort by name
        print("merging: ", fns)
        if len(fns) == 0:
            return  # nothing to do 
        CP.cprint("c", f"Merge Output file name: {str(self.outputFilename):s}")

        pdf_merger = PdfMerger()
        if self.outputFilename.is_file(): # add to existing file
            pdf_merger.append(self.outputFilename)
        for i, fn in enumerate(fns):
            print("Merging fn: ", fn)  # source file
            with open(fn, "rb") as fh:
                pdf_merger.append(fh)  # now append the rest
        #with open(self.outputFilename, "wb") as fo:
        print(self.outputFilename)
        pdf_merger.write(str(self.outputFilename))
        # print(dir(pdf_merger))
        pdf_merger.close()
        # self.clean_tempdir()  # clean up

    def gather_protocols(
        self,
        protocols: list,
        prots: dict,
        allprots: Union[dict, None] = None,
        day: Union[str, None] = None,
    ):
        """
        Gather all the protocols and sort by functions/types
        First call will likely have allprots = None, to initialize
        after that will update allprots with new lists of protocols.

        Parameters
        ----------
        protocols: list
            a list of all protocols that are found for this day/slice/cell
        prots: dict
            data, slice, cell information
        allprots: dict
            dict of all protocols by type in this day/slice/cell
        day : str
            str indicating the top level day for this slice/cell

        Returns
        -------
        allprots : dict
            uupdated copy of allprots.
        """
        if allprots is None:
            allprots = {"maps": [], "stdIVs": [], "CCIV_long": [], "VCIVs": []}
        self.stdIVs = ["CCIV_short", "CCIV_1nA_max", "CCIV_4nA_max"]
        prox = sorted(
            list(set(protocols))
        )  # adjust for duplicates (must be a better way in pandas)
        # print('prox: ', prox)
        for i, x in enumerate(prox):  # construct filenames and sort by analysis types
            # print(f'x: <{x:s}>')
            # print(f'self.protocol: <{self.protocol:s}>')
            if len(x) == 0:
                continue
            # print(self.protocol != x)
            # clean up protocols that have a path ahead of the protocol (can happen when combining datasets in datasummary)
            xp = Path(x).parts
            if xp[0] == "/" and len(xp) > 1:
                x = xp[-1]  # just get the protocol directory
            if len(self.protocol) > 1 and self.protocol != x:
                continue
            # c = os.path.join(day, prots.iloc[i]['slice_slice'], prots.iloc[i]['cell_cell'], x)
            if day is None:
                c = Path(prots["date"], prots["slice_slice"], prots["cell_cell"], x)
            else:
                c = Path(
                    day, prots.iloc[i]["slice_slice"], prots.iloc[i]["cell_cell"], x
                )
            # maps
            # print('map?: ', x.startswith('Map'))
            if x.startswith("Map"):
                allprots["maps"].append(c)
            if x.startswith("VGAT_"):
                allprots["maps"].append(c)
            # Standard IVs (100 msec, devined as above)
            for piv in self.stdIVs:
                if x.startswith(piv):
                    allprots["stdIVs"].append(c)
            # Long IVs (0.5 or 1 second)
            if x.startswith("CCIV_long"):
                allprots["CCIV_long"].append(c)
            # VCIVs
            if x.startswith("VCIV"):
                allprots["VCIVs"].append(c)
        return allprots

    def make_cell(self, iday: int):
        datestr = self.df.iloc[iday]["date"]
        slicestr = str(Path(self.df.iloc[iday]["slice_slice"]).parts[-1])
        cellstr = str(Path(self.df.iloc[iday]["cell_cell"]).parts[-1])
        return (datestr, slicestr, cellstr)

    def find_cell(
        self,
        df: pd.DataFrame,
        datestr: Union[str, datetime.datetime],
        slicestr: str,
        cellstr: str,
        protocolstr: Union[Path, str] = None,
    ):
        """Find the dataframe element for the specified date, slice, cell and
        protocol in the input dataframe

        Parameters
        ----------
        df : Pandas dataframe
            The dataframe for the experiments, as a Pandas frame
        datestr : Union[str, datetime.datetime]
            The date string we are looking for ("2017.04.19_000")
        slicestr : str
            The slice string ("slice_000")
        cellstr : str
            The cell identification string ("cell_000")
        protocolstr : Union[Path, str], optional
            The protocol string ("VCIV_000"), by default None

        Returns
        -------
        dataframe
            result of the dataframe query.
        """        
        dstr = str(datestr)
        qstring = f'date == "{dstr:s}"'
        cf = df.query(qstring)
        if protocolstr is None:
            cf = df.query(
                f'date == "{dstr:s}" & slice_slice == "{slicestr:s}" & cell_cell == "{cellstr:s}"'
            )
        else:
            dprot = str(protocolstr.name)
            cf = df.query(
                f'date == "{dstr:s}" & slice_slice == "{slicestr:s}" & cell_cell == "{cellstr:s}" & map == "{dprot:s}"'
            )

        return cf

    def get_celltype(self, iday):
        """Find the type of the cell associated with this entry

        Parameters
        ----------
        iday : int
            The index into the table

        Returns
        -------
        str or None
            The celltype, or None if it is missing or cannot be parsed.
        """
        datestr, slicestr, cellstr = self.make_cell(iday)
        fullfile = Path(self.basedir, self.make_cellstr(self.df, iday))
        #        prots = [x for x in fullfile.glob('*') if x.is_dir()]
        prots = self.df.iloc[iday]
        # print('prots: ', prots)
        #     print('data complete: ', prots['data_complete'])
        u = prots["data_complete"].split(", ")

        allprots = self.gather_protocols(u, prots)

        if self.map_annotationFile is not None and len(allprots["maps"]) > 0:  # get from annotation file
            cell_df = self.find_cell(
                self.map_annotations, datestr, slicestr, cellstr, allprots["maps"][0]
            )
            celltype = cell_df["cell_type"].values[0]
            if isinstance(celltype, float):
                return None
            CP.cprint("red", f"Map annotated celltype: {celltype:s}")
        elif self.annotated_dataframe is not None:
            cell_df = self.find_cell(self.annotated_dataframe, datestr, slicestr, cellstr)
            if len(cell_df["cell_type"]) == 0:
                return None
            celltype = cell_df["cell_type"]
            CP.cprint("yellow", f"Cell re-annotated celltype: {celltype:s})")
        else:
            celltype = self.df.at[iday, "cell_type"]
            CP.cprint("magenta", f"Original Database designated celltype: {celltype:s}")

        print("       Celltype: {0:s}".format(celltype))
        print("   with {0:4d} protocols".format(len(allprots["maps"])))
        for i, p in enumerate(allprots["maps"]):
            print("      {0:d}. {1:s}".format(i + 1, str(p.name)))
        return celltype

    def do_day(self, iday: int, nout: int = 0, pdf=None):
        """
        Do analysis on a day's data
        Permits subsetting if slicecell is specified (to do just one specific cell)

        Parameters
        ----------
        iday : int
            index into pandas database for the day to be analyzed


        """
        datestr, slicestr, cellstr = self.make_cell(iday)

        dsday, nx = Path(datestr).name.split("_")
        thisday = datetime.datetime.strptime(dsday, "%Y.%m.%d")
        if thisday < self.after or thisday > self.before:
            return
        celltype = None
        if self.slicecell:
            slicen = "slice_%03d" % int(self.slicecell[1])
            if slicestr != slicen:
                return
            if len(self.slicecell) == 4:
                celln = "cell_%03d" % int(self.slicecell[3])
                if cellstr != celln:
                    return

        #       Only returns a dataframe if there is more than one entry
        #       Otherwise, it is like a series or dict
        fullfile = Path(self.basedir, self.make_cellstr(self.df, iday))
        #        prots = [x for x in fullfile.glob('*') if x.is_dir()]
        prots = self.df.iloc[iday]
        # print("Protocols: ", prots)
        print("data complete: ", prots["data_complete"])
        u = prots["data_complete"].split(", ")
        allprots = self.gather_protocols(u, prots)
        # if self.dry_run:
        #     print('Would process day: {0:s} slice: {1:s} cell: {2:s}'.format(
        #             datestr, slicestr, cellstr))
        #     if not fullfile.is_dir() or (fullfile.is_dir() and len(prots) == 0):
        #         print('   but that file/directory was not found or has no protocols.')
        #     else:
        #         print('   with {0:4d} protocols'.format(len(prots)))
        #         if self.map_flag:
        #             for i, p in enumerate(sorted(prots)):
        #                 if p in allprots['maps']:
        #                     print('      {0:d}. {1:s}'.format(i+1, str(p.name)))
        #     return

        # prots = self.df.iloc[iday]

        if self.verbose:
            for k in list(allprots.keys()):
                print("protocol type: {:s}:".format(k))
                for m in allprots[k]:
                    print("    {0:s}".format(str(m)))
                if len(allprots[k]) == 0:
                    print("    No protocols of this type")
            print("All protocols: ")
            print([allprots[p] for p in allprots.keys()])

        celltype = self.get_celltype(iday)
        # if self.merge_flag:
        #     self.merge_pdfs(celltype)
            # return
        # CP.cprint("r", f"iv flag: {str(self.iv_flag):s}")
        if self.iv_flag:
            # self.make_tempdir()  # clean up temporary directory
            self.analyze_ivs(iday, allprots, pdf)
            # self.merge_pdfs(celltype)  # do not do this until all the parallel processing for the day is done.

        if self.vc_flag:
            self.analyze_vcs(iday, allprots)

        if self.map_flag:
            if len(allprots["maps"]) == 0:
                return
            if self.dry_run:
                print(
                    "Would process day: {0:s} slice: {1:s} cell: {2:s}".format(
                        datestr, slicestr, cellstr
                    )
                )
                if self.map_annotationFile is not None:  # get from annotation file
                    cell_df = self.find_cell(
                        self.map_annotations,
                        datestr,
                        slicestr,
                        cellstr,
                        allprots["maps"][0],
                    )
                    celltype = cell_df["cell_type"].values[0]
                    CP.cprint("red", f"map annotated celltype: {celltype:s}")
                elif self.annotated_dataframe is not None:
                    cell_df = self.find_cell(self.annotated_dataframe, datestr, slicestr, cellstr)
                    celltype = cell_df["cell_type"].values[0]
                    print("cell annotated celltype: ", celltype)
                else:
                    celltype = self.df.at[iday, "cell_type"].values[0]
                    print("database celltype: ", celltype)

                print("       Celltype: {0:s}".format(celltype))
                print("   with {0:4d} protocols".format(len(allprots["maps"])))
                for i, p in enumerate(allprots["maps"]):
                    print("      {0:d}. {1:s}".format(i + 1, str(p.name)))
                return

            validmaps = []
            for p in allprots["maps"]:  # first remove excluded protocols
                if str(p) not in self.exclusions:
                    validmaps.append(
                        p
                    )  # note we do not just remove as this messes up the iterator of the maps
            allprots["maps"] = validmaps
            nworkers = 16  # number of cores/threads to use
            tasks = range(len(allprots["maps"]))  # number of tasks that will be needed
            results = dict()  # storage for results
            result = [None] * len(tasks)  # likewise
            self.make_tempdir()  # clean up temporary directory
            plotmap = True
            foname = "%s~%s~%s" % (datestr, slicestr, cellstr)
            if self.signflip:
                foname += "_signflip"
            if self.alternate_fit1:
                foname += "_alt1"
            if self.alternate_fit2:
                foname += "_alt2"

            foname += ".pkl"
            exptdir = self.inputFilename.parent
            picklefilename = Path(exptdir, "events", foname)
            ###
            ### Parallel is done at lowest level of analyzing a trace, not at this top level
            ### can only have ONE parallel loop going (no nested ones allowed!)
            ###
            # if self.noparallel:  # just serial...
            for i, x in enumerate(tasks):
                result = self.analyze_map(
                    iday,
                    i,
                    x,
                    allprots,
                    plotmap,
                    measuretype=self.measuretype,
                    verbose=self.verbose,
                    picklefilename=picklefilename,
                )
                if result is None:
                    continue
                results[allprots["maps"][x]] = result
        # terminate early when testing
                # if i == 0:
                #     break
        #             else:
        #                 with mp.Parallelize(enumerate(tasks), results=results, workers=nworkers) as tasker:
        #                     for i, x in tasker:
        #                         result = self.analyze_map(iday, i, x, allprots, plotmap)
        # #                        print(x)
        #                         tasker.results[allprots['maps'][x]] = result
            if not self.replot:  # only save if we are NOT just replotting
                print("events to : ", exptdir, foname)
                print('result keys: ', results.keys())
                with open(picklefilename, "wb") as fh:
                    dill.dump(results, fh)

                    
            if plotmap:
                #                if self.slicecell:
                if self.map_annotationFile is not None:  # get from annotation file
                    cell_df = self.find_cell(
                        self.map_annotations,
                        datestr,
                        slicestr,
                        cellstr,
                        allprots["maps"][0],
                    )
                    if cell_df["cell_type"].shape != (0,):
                        celltype = cell_df["cell_type"].values[0]
                        CP.cprint("red", f"map annotated celltype: {celltype:s}")
                    else:
                        celltype = 'Unknown'
                        CP.cprint("yellow", f"map annotated celltype: {celltype:s}")
                        
                elif self.annotated_dataframe is not None:
                    cell_df = self.find_cell(self.annotated_dataframe, datestr, slicestr, cellstr)
                    celltype = cell_df["cell_type"].values[0]
                    CP.cprint("yellow", f"cell annotated celltype: {celltype:s})")
                else:
                    celltype = self.df.at[iday, "cell_type"].values[0]
                    CP.cprint("magenta", f"database celltype: {celltype:s}")

                print("       Celltype: {0:s}".format(celltype))
                # print("merging pdfs")
                # self.merge_pdfs(celltype=celltype)

    def set_vc_taus(self, iday: int, path: Union[Path, str]):
        """
        """
        datestr, slicestr, cellstr = self.make_cell(iday)
        # print(self.map_annotationFile)
        # print(self.map_annotations)
        if self.map_annotationFile is not None:
            cell_df = self.find_cell(
                self.map_annotations, datestr, slicestr, cellstr, path
            )
            if cell_df is None:
                return
            # print(datestr, slicestr, cellstr)
            # print(cell_df)
            # print(cell_df['tau1'])
            # print(cell_df['map'])
            sh = cell_df["alt1_tau1"].shape
            sh1 = cell_df['tau1'].shape
            sh2 = cell_df['fl_tau1'].shape
            # print(sh, sh1, sh2)
            if sh != (0,) and sh1 != (0,) and sh2 != (0,):
                if not self.signflip:
                    if self.alternate_fit1:
                        self.AM.Pars.taus = [
                            cell_df["alt1_tau1"].values[0] * 1e-3,
                            cell_df["alt1_tau2"].values[0] * 1e-3,
                        ]
                    else:
                        self.AM.Pars.taus = [
                            cell_df["tau1"].values[0] * 1e-3,
                            cell_df["tau2"].values[0] * 1e-3,
                        ]
                    print("    Setting VC taus: ", end="")
                else:
                    self.AM.Pars.taus = [
                        cell_df["fl_tau1"].values[0] * 1e-3,
                        cell_df["fl_tau2"].values[0] * 1e-3,
                    ]
                    print("   SIGN flip, set VC taus: ", end="")

            else:
                CP.cprint('r', "Using default VC taus")
                exit()
        CP.cprint('w', f"    [{self.AM.Pars.taus[0]:8.4f}, {self.AM.Pars.taus[1]:8.4f}]")


    def set_cc_taus(self, iday: int, path: Union[Path, str]):
        datestr, slicestr, cellstr = self.make_cell(iday)
        if self.map_annotationFile is not None:
            cell_df = self.find_cell(
                self.map_annotations, datestr, slicestr, cellstr, path,
            )
            sh = cell_df["cc_tau1"].shape
            if cell_df is not None and sh!= (0,):
                self.AM.Pars.taus = [
                    cell_df["cc_tau1"].values[0] * 1e-3,
                    cell_df["cc_tau2"].values[0] * 1e-3,
                ]
                print("    Setting CC taus from map annotation: ", end="")
            else:
                print("    Using default taus")
        CP.cprint('w', f"    [{self.AM.Pars.taus[0]:8.4f}, {self.AM.Pars.taus[1]:8.4f}]")

    def set_vc_threshold(self, iday: int, path: Union[Path, str]):
        datestr, slicestr, cellstr = self.make_cell(iday)
        if self.map_annotationFile is not None:
            cell_df = self.find_cell(
                self.map_annotations, datestr, slicestr, cellstr, path
            )
            sh = cell_df["alt1_threshold"].shape
            if cell_df is not None and sh!= (0,):
                if not self.signflip:
                    if self.alternate_fit1:
                        self.AM.Pars.threshold = cell_df["alt1_threshold"].values[0]
                    else:
                        self.AM.Pars.threshold = cell_df["threshold"].values[0]
                    print(f"    Setting VC threshold from map table", end=" ")
                else:  # sign flip
                    self.AM.Pars.threshold = cell_df["fl_threshold"].values[0]
                    print("    Setting VC threshold from flipped values", end=" ")
            else:
                print("    Using default threshold", end=" ")
        print(f"      Threshold: {self.AM.Pars.threshold:6.1f}")

    def set_cc_threshold(self, iday: int, path: Union[Path, str]):
        datestr, slicestr, cellstr = self.make_cell(iday)
        if self.map_annotationFile is not None:
            cell_df = self.find_cell(
                self.map_annotations, datestr, slicestr, cellstr, path
            )
            sh = cell_df["cc_threshold"].shape
            if cell_df is not None and sh!= (0,):
                self.AM.Pars.threshold = cell_df["cc_threshold"].values[0]
                print("set cc_threshold from map annotation")
            else:
                print("using default cc threshold")

    def set_stimdur(self, iday: int, path: Union[Path, str]):
        datestr, slicestr, cellstr = self.make_cell(iday)
        if self.map_annotationFile is not None:
            cell_df = self.find_cell(
                self.map_annotations, datestr, slicestr, cellstr, path
            )
            sh = cell_df["stimdur"].shape
            if sh != (0,):
                self.AM.set_stimdur(cell_df["stimdur"].values[0])
            else:
                print("using default stimdur") 
            # print(dir(self.AM))
            # print(self.AM.Pars)
            # print(self.AM.Pars.stimdur)
            # if cell_df is not None and not np.isnan(cell_df["stimdur"].astype(float)):
            #     self.AM.stimdur = cell_df["stimdur"].astype(float)
            #     print("Set stimdur from map annotation")
            # else:
            #     print("using default stimdur")
            

    def set_map_factors(self, iday: int, path: Union[Path, str]):
        """
        Configure signs, scale factors and E/IPSP/C template shape
        Rules:
        If VC but not "VGAT", use EPSC scaling and negative sign
        If VC and VGAT, use IPSC scaling. Check electrode as to what sign to use
        If CA (cell attached), use CA scaling (200 pA, high threshold, and fast event)
        if IC (and KGluc) but not VGAT use EPSP scaling (traces zero offset and 10 mV steps)
        If IC (and KGluc) and VGAT, use IPSP scaling (traces zero offset and 10 mV steps)
        Parameters
        ----------
        p : string
            name of acq4 protocol (used to select scales etc)

        Returns
        -------
        Nothing - modifies the analyzeMap object.
        """
        notes = self.df.at[iday, "notes"]
        internal_sol = self.df.at[iday, "internal"]
        self.internal_Cs = False
        self.high_Cl = False
        csstr = re.compile("(Cs|Cesium|Cs)", re.IGNORECASE)
        if re.search(csstr, internal_sol) is not None:
            self.internal_Cs = True
        if (
            re.search(csstr, notes) is not None
        ):  # could also be in notes; override interal setting
            self.internal_Cs = True
        clstr = re.compile("(Hi|High)\s+(Cl|Chloride)", re.IGNORECASE)
        if re.search(clstr, notes) is not None:
            self.high_Cl = True  # flips sign for detection
            # print(' HIGH Chloride cell ***************')
        # read the mapdir protocol
        protodir = Path(self.basedir, path)
        try:
            assert protodir.is_dir()
            protocol = self.AM.AR.readDirIndex(str(protodir))
            record_mode = protocol["."]["devices"]["MultiClamp1"]["mode"]
        except:
            if  path.match("*_VC_*"):
                record_mode = 'VC'
            elif path.match("*_IC_*"):
                record_mode = 'IC'
            else:
                raise ValueError ("Cant figure record mode")
        
        self.set_stimdur(iday, path)

        if (path.match("*_VC_*") or record_mode == "VC") and not self.basedir.match(
            "*VGAT_*"
        ):  # excitatory PSC
            self.AM.datatype = "V"
            self.AM.Pars.sign = -1
            self.AM.Pars.scale_factor = 1e12
            self.AM.Pars.taus = [1e-3, 3e-3]  # fast events
            self.set_vc_taus(iday, path)
            self.AM.Pars.threshold = self.threshold  # threshold...
            self.set_vc_threshold(iday, path)

        elif (path.match("*_VC_*") or record_mode == "VC") and self.basedir.match(
            "*VGAT_*"
        ):  # inhibitory PSC
            self.AM.datatype = "V"
            if self.high_Cl:
                self.AM.Pars.sign = -1
            else:
                self.AM.Pars.sign = 1
            self.AM.Pars.scale_factor = 1e12
            self.AM.Pars.taus = [2e-3, 10e-3]  # slow events
            self.AM.Pars.analysis_window = [0, 0.999]
            self.AM.Pars.threshold = self.threshold  # low threshold
            self.set_vc_taus(iday, path)
            self.AM.Pars.threshold = self.threshold  # threshold...
            self.set_vc_threshold(iday, path)
            print("sign: ", self.AM.Pars.sign)

        elif path.match("*_CA_*") and record_mode == "VC":  # cell attached (spikes)
            self.AM.datatype = "V"
            self.AM.Pars.sign = -1  # trigger on negative current
            self.AM.Pars.scale_factor = 1e12
            self.AM.Pars.taus = [0.5e-3, 0.75e-3]  # fast events
            self.AM.Pars.threshold = self.threshold  # somewhat high threshold...
            self.set_vc_taus(iday, path)
            self.set_vc_threshold(iday, path)

        elif (
            path.match("*_IC_*") and record_mode in ["IC", "I=0"]
        ) and not self.basedir.match(
            "*VGAT_*"
        ):  # excitatory PSP
            self.AM.Pars.sign = 1  # positive going
            self.AM.Pars.scale_factor = 1e3
            self.AM.Pars.taus = [1e-3, 4e-3]  # fast events
            self.AM.datatype = "I"
            self.AM.Pars.threshold = self.threshold  # somewhat high threshold...
            self.set_cc_taus(iday, path)
            self.set_cc_threshold(iday, path)

        elif path.match("*_IC_*") and self.basedir.match("*VGAT_*"):  # inhibitory PSP
            print("IPSP detector!!!")
            self.AM.Pars.sign = -1  # inhibitory so negative for current clamp
            self.AM.Pars.scale_factor = 1e3
            self.AM.Pars.taus = [3e-3, 10e-3]  # slow events
            self.AM.datatype = "I"
            self.AM.Pars.threshold = self.threshold  #
            self.AM.Pars.analysis_window = [0, 0.999]
            self.set_cc_taus(iday, path)
            self.set_cc_threshold(iday, path)

        elif path.match("VGAT_*") and not (
            path.match("*_IC_*") or path.match("*_VC_*") or path.match("*_CA_*")
        ):  # VGAT but no mode information
            if record_mode in ["IC", "I=0"]:
                self.AM.datatype = "I"
                if self.high_Cl:
                    self.AM.Pars.sign = 1
                else:
                    self.AM.Pars.sign = -1
            elif record_mode in ["VC"]:
                self.AM.datatype = "V"
                if self.high_Cl:
                    self.AM.Pars.sign = -1
                else:
                    self.AM.Pars.sign = 1

            else:
                raise ValueError(
                    "Record mode not recognized: {0:s}".format(record_mode)
                )
            self.AM.Pars.analysis_window = [0, 0.999]
            self.AM.Pars.scale_factor = 1e12
            self.AM.Pars.taus = [2e-3, 10e-3]  # slow events
            self.AM.Pars.threshold = self.threshold  # setthreshold...

            if self.AM.datatype == "V":
                self.set_vc_taus(iday, path)
                self.set_vc_threshold(iday, path)
            else:
                self.set_cc_taus(iday, path)
                self.set_cc_threshold(iday, path)

        else:
            print("Undetermined map factors - add to the function!")
        if self.verbose:
            print(
                "Data Type: {0:s}/{1:s}  Sign: {2:d}  taus: {3:s}  thr: {4:5.2f}  Scale: {4:.3e}".format(
                    self.AM.datatype,
                    record_mode,
                    self.AM.Pars.sign,
                    str(np.array(self.AM.Pars.taus) * 1e3),
                    self.AM.Pars.threshold,
                    self.AM.Pars.scale_factor,
                )
            )

    def analyze_map(
        self,
        iday: int,
        i: int,
        x: str,
        allprots: dict,
        plotmap: bool,
        measuretype: str = "ZScore",
        verbose: bool = False,
        picklefilename:Union[Path, str, None] = None,
    ) ->Union[None, dict]:
        """
        Analyze the ith map in the allprots dict of maps
        This routine is designed so that it can be called for parallel processing.

        Parameters
        ----------
        iday : int
            index to the day in the pandas database
        i : int
            index into the list of map protocols for this cell/day
        x : str
            name of protocol
        allprots : dict
            dictionary containing parsed protocols for this day/slice/cell
        plotmap : boolean
            Boolean flag indicating whether plots will be generated
        results : dict
            An empty or existing results dictionary. New results are appended
            to this dict with keys based on the day/slice/cell/protocol

        Returns
        -------
        results : dict
            Updated copy of the results dict that was passed in
        success : boolean
            true if there data was processed; otherwise False
        """
        # import matplotlib
        import matplotlib.pyplot as mpl
        CP.cprint('g', 'Entering IV_Analysis:analyze_map')
        # matplotlib.use('Agg')
        rcParams = matplotlib.rcParams
        rcParams["svg.fonttype"] = "none"  # No text as paths. Assume font installed.
        rcParams["pdf.fonttype"] = 42
        rcParams["ps.fonttype"] = 42
        # rcParams['text.latex.unicode'] = True
        # rcParams['font.family'] = 'sans-serif'
        # rcParams['font.sans-serif'] = 'DejaVu Sans'
        # rcParams['font.weight'] = 'regular'                  # you can omit this, it's the default
        # rcParams['font.sans-serif'] = ['Arial']
        rcParams["text.usetex"] = True
        mapname = allprots["maps"][i]
        if len(str(mapname)) == 0:
            return None
        
        mapdir = Path(self.basedir, mapname)
        # if self.dry_run:
        #     datestr, slicestr, cellstr = self.make_cell(iday)
        #     print('   Would analyze %s' % mapdir, '\n    base: ', str(self.basedir))
        #     foname = '%s~%s~%s.pkl'%(datestr, slicestr, cellstr)
        #     print('   Output file would be: ', foname)
        #     return results, False
        if "_IC__" in str(mapdir.name) or "CC" in str(mapdir.name):
            scf = 1e3  # mV
        else:
            scf = 1e12  # pA, vc
        if self.replot:
            CP.cprint('g', f"IV_Analysis:analyze_map  Replotting from .pkl file: {str(picklefilename):s}")
            with open(picklefilename, "rb") as fh:  # read the previously analyzed data set
                results = dill.load(fh)
            mapkey = Path('/'.join(mapname.parts[-4:]))
            # print(results.keys())
#             print('mapkey: ', mapkey)
            result = results[mapkey] # get individual map result

        self.set_map_factors(iday, mapdir)
        if self.LPF > 0:
            self.AM.set_LPF(self.LPF)
        if self.HPF > 0:
            self.AM.set_HPF(self.HPF)
        if self.notchfilter:
            self.AM.set_notch(True, self.notchfreqs, self.notch_Q)
        else:
            self.AM.set_notch(False)
        self.AM.set_methodname(self.detector)
        if self.signflip:
            self.AM.Pars.sign = -1 * self.AM.Pars.sign  # flip analysis sign
        
        self.AM.set_analysis_window(0., 0.599)
        self.AM.set_artifact_suppression(self.artifact_suppress)
        self.AM.set_noderivative_artifact(self.noderivative_artifact)
        if self.artifactFilename is not None:
            self.AM.set_artifact_file(Path('/Users/pbmanis/Desktop/Python/mrk_nf107/datasets/NF107Ai32_Het', self.artifactFilename))
        self.AM.set_taus(self.AM.Pars.taus)  # [1, 3.5]

        if not self.replot:
            CP.cprint('g', f"IV_Analysis:analyze_map  Running map analysis: {str(mapname):s}")
            result = self.AM.analyze_one_map(
                mapdir, noparallel=self.noparallel, verbose=verbose
            )

            if result is not None:
                # result['onsets'] = [result['events'][i]['onsets'] for i in result['events']]
                result["analysisdatetime"] = datetime.datetime.now()  # add timestamp
            if result is None:
                return
        else:
            pass # already got the file
            
        #        results[p] = result
        # print('results keys: ', results.keys())
        if plotmap:
            if self.map_annotationFile is not None:
                datestr, slicestr, cellstr = self.make_cell(iday)
                cell_df = self.find_cell(
                    self.map_annotations, datestr, slicestr, cellstr, mapdir
                )
                if cell_df is not None and cell_df['cell_type'].shape != (0,):
                    ann_celltype = cell_df["cell_type"].values[0].lstrip().rstrip()
                else:
                    ann_celltype = None
            print("Annotated cell type: ", ann_celltype)
            ctype = self.df.at[iday, "cell_type"]
            celltype_text = f"{ctype:s} [orig]"
            if ann_celltype is not None and ctype != ann_celltype:
                celltype_text = f"{ctype:s} [orig] -> {ann_celltype:s} [revised]"
            getimage = False
            plotevents = True
            self.AM.Pars.overlay_scale = 0.0
            PMD.set_Pars_and_Data(self.AM.Pars, self.AM.Data)
            if self.replot:
                mapok = PMD.display_position_maps(dataset_name=mapdir, result=result, pars=self.AM.Pars)
            else:
                # print("results: ", self.AM.last_results)
                if mapdir != self.AM.last_dataset:
                    results = self.analyze_one_map(self.AM.last_dataset)
                else:
                    results = self.AM.last_results
                # print(results)
                # exit()
                mapok = PMD.display_one_map(
                    mapdir,
                    results=results, # self.AM.last_results,
                    # justplot=self.replot,
                    imagefile=None,
                    rotation=0.0,
                    measuretype=measuretype,
                    plotevents=plotevents,
                    whichstim=self.whichstim,
                    trsel=self.trsel,
                    plotmode=self.plotmode,
                    average=False,
                    rasterized=False,
                )  # self.AM.rasterized, firstonly=True, average=False)
            print(f"{str(mapname):s} done.")
            print("="*80)
            print()
            if mapok:
                infostr = ""
                # notes = self.df.at[iday,'notes']
                if self.internal_Cs:
                    if self.high_Cl:
                        infostr += "Hi-Cl Cs, "
                    elif self.internal_Cs:
                        infostr += "Norm Cs, "
                else:
                    infostr += self.df.at[iday, "internal"] + ", "

                temp = self.df.at[iday, "temperature"]
                if temp == "room temperature":
                    temp = "RT"
                infostr += "{0:s}, ".format(temp)
                infostr += "{0:s}, ".format(self.df.at[iday, "sex"].upper())
                infostr += "{0:s}".format(str(self.df.at[iday, "age"]).upper())
                # ftau1 = np.nanmean(np.array(result['events'][0]['fit_tau1']))
                # ftau2 = np.nanmean(np.array(result['events'][0]['fit_tau2']))
                # famp = np.nanmean(np.array(result['events'][0]['fit_amp']))
                params = "Mode: {0:s}  Sign: {1:d}  taus: {2:.2f}, {3:.2f}  thr: {4:5.2f}  Scale: {5:.1e} Det: {6:2s}".format(
                    self.AM.datatype,
                    self.AM.Pars.sign,
                    self.AM.Pars.taus[0] * 1e3,
                    self.AM.Pars.taus[1] * 1e3,
                    self.AM.Pars.threshold,
                    self.AM.Pars.scale_factor,
                    self.AM.methodname,
                )
                fix_mapdir = str(mapdir).replace("_", "\_")
                PMD.P.figure_handle.suptitle(
                    f"{fix_mapdir:s}\n{celltype_text:s} {infostr:s} {params:s}",
                    fontsize=8,
                )
                t_path = Path(self.tempdir, "temppdf_{0:s}.pdf".format(str(mapdir.name)))
                if not self.tempdir.is_dir():
                    print("temp dir not found: ", self.tempdir)
                # mpl.savefig(t_path) # use the map filename, as we will sort by this later
                # mpl.show()
                if t_path.is_file():
                    t_path.unlink()
                pp = PdfPages(t_path)
                # try:
                mpl.savefig(
                    pp, format="pdf"
                )  # use the map filename, as we will sort by this later
                pp.close()
                # except ValueError:
                #       print('Error in saving map %s, file %s' % (t_path, str(mapdir)))
                mpl.close(PMD.P.figure_handle)
        #        print('returning results', results.keys())
        return result

    def analyze_ivs(self, iday: int, allprots: dict, pdf: None):
        """
        Overall analysis of IV protocols for one day

        Parameters
        ----------
        iday : int
            index into Pandas database

        allprots : dict
            dictionary of protocols for the day/slice/cell

        Returns
        -------
        Nothing - generates pdfs and updates the pickled database file.
        """
        CP.cprint("c", f"analyze ivs for index: {iday: d}")
        if "IV" not in self.df.columns.values:
            self.df = self.df.assign(IV=None)
        if "Spikes" not in self.df.columns.values:
            self.df = self.df.assign(Spikes=None)
        # self.make_tempdir()
        datestr, slicestr, cellstr = self.make_cell(iday)

        nfiles = 0
        allivs = []
        if self.dry_run:
            print(allprots.keys())
        for ptype in allprots.keys():  # check all the protocols
            if ptype not in ["stdIVs", "CCIV_long"]:  # just CCIV types
                continue
            allivs.extend(allprots[ptype])  # combine into a new list
        if self.dry_run:
            print("  allivs: ", iday, datestr, slicestr, cellstr, allivs)
        nworkers = 16  # number of cores/threads to use
        tasks = range(len(allivs))  # number of tasks that will be needed
        results = dict(
            [("IV", {}), ("Spikes", {})]
        )  # storage for results; predefine the dicts.
        result = [None] * len(tasks)  # likewise
        if self.noparallel:  # just serial...
            for i, x in enumerate(tasks):
                r, nfiles = self.analyze_iv(iday, i, x, allivs, nfiles, pdf)
                if self.dry_run:
                    continue
                if r is None:
                    continue
                results["IV"][allivs[i]] = r["IV"]
                results["Spikes"][allivs[i]] = r["Spikes"]
            if not self.dry_run:
                self.df.at[iday, "IV"] = results[
                    "IV"
                ]  # everything in the RM analysis_summary structure
                self.df.at[iday, "Spikes"] = results[
                    "Spikes"
                ]  # everything in the SP analysus_summary structure
        #            print(self.df.at[iday, 'IV'])
        else:
            with mp.Parallelize(
                enumerate(tasks), results=results, workers=nworkers
            ) as tasker:
                for i, x in tasker:
                    result, nfiles = self.analyze_iv(iday, i, x, allivs, nfiles, pdf=pdf)
                    tasker.results[allivs[i]] = result
            # reform the results for our database
            if self.dry_run:
                return
            riv = {}
            rsp = {}
            for f in list(results.keys()):
                if "IV" in results[f]:
                    riv[f] = results[f]["IV"]
                if "Spikes" in results[f]:
                    rsp[f] = results[f]["Spikes"]
            #            print('parallel results: \n', [(f, results[f]['IV']) for f in results.keys() if 'IV' in results[f].keys()])
            #            print('riv: ', riv)
            self.df.at[
                iday, "IV"
            ] = riv  # everything in the RM analysis_summary structure
            self.df.at[
                iday, "Spikes"
            ] = rsp  # everything in the SP analysus_summary structure
            # foname = '%s~%s~%s.pkl'%(datestr, slicestr, cellstr)
        # exptdir = self.inputFilename.parent
        # if self.dry_run:
        #     return  # NEVER update the database...
        # fns = sorted(list(results.keys()))  # get all filenames
        # print('results: ', results['IV'])

        # with open(Path(exptdir, 'events', foname), 'wb') as fh:
        #      dill.dump(results, fh)

    def analyze_iv(self, iday: int, i: int, x: int, allivs: list, nfiles: int, pdf:None):
        """
        Compute various measures (input resistance, spike shape, etc) for one IV
        protocol in the day. Designed to be used in parallel or serial mode

        Parameters
        ----------
        iday : int
            index into Pandas database for the day
        i : int
            index into the list of protocols
        x : task
            task number
        allivs :
            list of protocols
        nfiles : int
            number for file...

        pdf: matplotlib pdfpages instance or None
            if None, no plots are accumulated
            if a pdfpages instance, all of the pages plotted by IVSummary are concatenated
            into the output file (self.outputFilename)
        """
        import matplotlib.pyplot as mpl  # import locally to avoid parallel problems

        f = allivs[i]
        result = {}
        iv_result = {}
        sp_result = {}
        if self.iv_select["duration"] > 0.0:
            EPIV = EP.IVSummary.IV(str(Path(self.basedir, f)), plot=True,)
            check = EPIV.iv_check(duration=self.iv_select["duration"])
            if check is False:
                return (None, 0)  # skip analysis
        if not self.dry_run:
            print("IV analysis for  %s" % Path(self.basedir, f))
            EPIV = EP.IVSummary.IVSummary(str(Path(self.basedir, f)), plot=True)
            br_offset = 0.0
            if (
                not pd.isnull(self.df.at[iday, "IV"])
                and f in self.df.at[iday, "IV"]
                and "--Adjust" in self.df.at[iday, "IV"][f].keys()
            ):
                br_offset = self.df.at[iday, "IV"][f]["BridgeAdjust"]
                print(
                    "Bridge: {0:.2f} Mohm".format(
                        self.df.at[iday, "IV"][f]["BridgeAdjust"] / 1e6
                    )
                )
            ctype = self.df.at[iday, "cell_type"]
            tgap = 0.0015
            tinit = True
            if ctype in ["bushy", "Bushy", "d-stellate", "D-stellate", "Dstellate", "octopus", "Octopus"]:
                tgap = 0.0005
                tinit = False
            EPIV.plot_mode(mode=self.IV_pubmode)
            plot_handle = EPIV.compute_iv(
                threshold=self.spike_threshold,
                bridge_offset=br_offset,
                tgap=tgap,
                plotiv=True,
                full_spike_analysis=True,
            )
            iv_result = EPIV.RM.analysis_summary
            sp_result = EPIV.SP.analysis_summary
            result["IV"] = iv_result
            result["Spikes"] = sp_result
            ctype = self.df.at[iday, "cell_type"]
            annot = self.df.at[iday, "annotated"]
            if annot:
                ctwhen = "[revisited]"
            else:
                ctwhen = "[original]"
            print("Checking for figure, plothandle is: ", plot_handle)
            if plot_handle is not None:
                shortpath = Path(self.basedir, f).parts
                shortpath = str(Path(*shortpath[4:]))
                plot_handle.suptitle(
                    "{0:s}\nType: {1:s} {2:s}".format(
                        shortpath, # .replace("_", "\_"),
                        self.df.at[iday, "cell_type"],
                        ctwhen,
                    ),
                    fontsize=8,
                )
                # t_path = Path(self.tempdir, "temppdf_{0:04d}.pdf".format(nfiles))
                if pdf is not None:
                    pdf.savefig(plot_handle)
                # else:
                #     mpl.savefig(
                #         t_path, dpi=300
                #     )  # use the map filename, as we will sort by this later
                #     mpl.close(plot_handle)
                #     CP.cprint("g", f"saved to: {str(t_path):s}")
                nfiles += 1
       
            return result, nfiles

        else:
            print("Dry Run: would analyze %s" % Path(self.basedir, f))
            br_offset = 0

            # print('IV keys: ', self.df.at[iday, 'IV'])
            if self.df.at[iday, "IV"] == {} or pd.isnull(self.df.at[iday, "IV"]):
                print("   current database has no IV data set for this file")
            elif f not in list(self.df.at[iday, "IV"].keys()):
                print(
                    "Protocol {0:s} not found in day: {1:s}".format(
                        str(f), self.df.at[iday, "date"]
                    )
                )
            elif "BridgeAdjust" in self.df.at[iday, "IV"][f].keys():
                br_offset = self.df.at[iday, "IV"][f]["BridgeAdjust"]
                print("   with Bridge: {0:.2f}".format(br_offset / 1e6))
            else:
                print("... has no bridge, will use 0")
            return None, 0

    def analyze_vcs(self, iday: int, allprots: dict):
        """
        Overall analysis of VC protocols
        Incomplete - mostly just plots the data

        Parameters
        ----------
        iday : int
            index into Pandas database for the day
        
        allprots : dict
            dictionary of protocols for the day/slice/cell

        Returns
        -------
        Nothing - generates pdfs and updates the pickled database file.
        """
        import matplotlib.pyplot as mpl

        self.make_tempdir()
        for pname in ["VCIVs"]:  # the different VCIV protocols
            if len(allprots[pname]) == 0:
                continue
            for f in allprots[pname]:
                if not self.dry_run:
                    print("Analyzing %s" % Path(self.basedir, f))
                    EPVC = EP.VCSummary.VCSummary(Path(self.basedir, f), plot=False,)
                    plotted = EPVC.compute_iv()
                    if plotted:
                        t_path = Path(
                            self.tempdir, "temppdf_{0:04d}.pdf".format(nfiles)
                        )
                        mpl.savefig(
                            t_path, dpi=300
                        )  # use the map filename, as we will sort by this later
                        nfiles += 1
                        # if pdf is not None and mapok:
                        #     pdf.savefig(dpi=300)
                        mpl.close(EPVC.IVFigure)
                    mpl.close(EPVC.IVFigure)
        self.merge_pdfs()


def main():

    # import warnings  # need to turn off a scipy future warning.
    # warnings.filterwarnings("ignore", category=FutureWarning)
    # warnings.filterwarnings("ignore", category=UserWarning)
    # warnings.filterwarnings("ignore", message="UserWarning: findfont: Font family ['sans-serif'] not found. Falling back to DejaVu Sans")
    
    experiments = nf107.set_expt_paths.get_experiments()
    exclusions = nf107.set_expt_paths.get_exclusions()
    args = AnalysisParams.getCommands(experiments)  # get from command line

    if args.basedir is None:
        X, args.basedir, code_dir = set_expt_paths.get_paths()
        if args.basedir is None:
            raise ValueError("No path set for computer")

    if args.configfile is not None:
        config = None
        if args.configfile is not None:
            if ".json" in args.configfile:
                # The escaping of "\t" in the config file is necesarry as
                # otherwise Python will try to treat is as the string escape
                # sequence for ASCII Horizontal Tab when it encounters it
                # during json.load
                config = json.load(open(args.configfile))
            elif ".toml" in args.configfile:
                config = toml.load(open(args.configfile))

        vargs = vars(args)  # reach into the dict to change values in namespace
        for c in config:
            if c in args:
                # print("c: ", c)
                vargs[c] = config[c]
    CP.cprint(
        "cyan",
        f"Starting analysis at: {datetime.datetime.now().strftime('%m/%d/%Y, %H:%M:%S'):s}",
    )
    IV = IV_Analysis(args)
    IV.setup(args, exclusions)
    IV.run()

    # allp = sorted(list(set(NF.allprots)))
    # print('All protocols in this dataset:')
    # for p in allp:
    #     print('   ', path)
    # print('---')
    #
    CP.cprint(
        "cyan",
        f"Finished analysis at: {datetime.datetime.now().strftime('%m/%d/%Y, %H:%M:%S'):s}",
    )


if __name__ == "__main__":
    import matplotlib
    import matplotlib.collections as collections

    rcParams = matplotlib.rcParams
    rcParams["svg.fonttype"] = "none"  # No text as paths. Assume font installed.
    rcParams["pdf.fonttype"] = 42
    rcParams["ps.fonttype"] = 42
    # rcParams['text.latex.unicode'] = True
    # rcParams['font.family'] = 'sans-serif'
    # rcParams['font.sans-serif'] = 'DejaVu Sans'
    # rcParams['font.weight'] = 'regular'                  # you can omit this, it's the default
    # rcParams['font.sans-serif'] = ['Arial']
    rcParams["text.usetex"] = False
    import matplotlib.colors
    import matplotlib.pyplot as mpl

    main()
