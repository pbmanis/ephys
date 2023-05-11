"""
Module for analyzing electrophysiology data.

This entry point is used as a wrapper for multiple experimental protocols,
including IV curves (current and voltage clamp), and optogenetic experiments, including
laser scanning photstimulation and glutamate uncaging maps.
"""
import gc
import sys
from collections.abc import Iterable
from multiprocessing import set_start_method

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
from pandas import HDFStore

matplotlib.use("QtAgg")
import matplotlib.pyplot as mpl  # import locally to avoid parallel problems
import pylibrary.plotting.plothelpers as PH
import pylibrary.tools.cprint as CP
import pyqtgraph as pg
import pyqtgraph.console as console
import pyqtgraph.multiprocess as mp
from matplotlib.backends.backend_pdf import PdfPages
from PyPDF2 import PdfMerger, PdfReader, PdfWriter

import ephys.ephys_analysis as EP
import ephys.datareaders as DR
import ephys.mapanalysistools as mapanalysistools
import ephys.mini_analyses as MINIS
import ephys.tools.build_info_string as BIS

from . import analysis_parameters as AnalysisParams

PMD = mapanalysistools.plot_map_data.PlotMapData()

# do not silently pass some common numpy errors
np.seterr(divide="raise", invalid="raise")


@dataclass
class cmdargs:
    """ This data class holds the "command arguments" that control
    data anlaysis. These arguments are defined in 3 blocks:
        1. Experiment specification and file names
        2. Analysis flags and experiment selection
        3. Graphics controls
        4. Specific analysis parameters
    
    These may be set either by IV_Analysis_params from the commandline,
    or by specifying them programatically.

    """
    # Files and filenames:
    experiment: Union[str, list, None] = None
    rawdatapath: Union[str, Path, None] = None
    directory:  Union[str, Path, None] = None
    analyzeddatapath: Union[str, Path, None] = None
    databasepath: Union[str, Path, None] = None
    inputFilename: Union[str, Path, None] = None
    pdfFilename: Union[str, Path, None] = None
    annotationFilename: Union[str, Path, None] = None
    artifactFilename: Union[str, Path, None] = None
    map_annotationFilename: Union[str, Path, None] = None
    map_pdfs: bool=False
    iv_analysisFilename: Union[str, Path, None] = None
    map_pdfs: bool=False
    extra_subdirectories: object = None

    # analysis flags and experiment selection
    update_results:bool = False
    day: str = "all"
    after: str = "1970.1.1"
    before: str = "2266.1.1"
    slicecell: str = ""
    protocol: str = ""
    configfile: Union[str, None] = None
    iv_flag: bool = False
    vc_flag: bool = False
    map_flag: bool = False
    merge_flag: bool = False
    dry_run: bool = False
    verbose: bool = False
    autoout: bool = False
    excel: bool = False

    # graphics controls
    plotmode: str = "document"
    IV_pubmode: str = "normal"
    plotsoff: bool = False
    rasterize: bool = True
    update: bool = False
    noparallel: bool = True
    mapsZQA_plot: bool = False
    recalculate_events: bool = True

    # analysis parameters
    ivduration: float = 0.0
    threshold: float = 2.5  # cb event detection threshold (x baseline std)
    refractory: float = 0.0007 # absolute refractory period
    signflip: bool = False
    alternate_fit1: bool = False
    alternate_fit2: bool = False  # second alternate
    measuretype: str = "ZScore"  # display measure for spot plot in maps
    spike_threshold: float = -0.035
    artifact_suppression: bool = False
    noderivative: bool = False
    whichstim: int = -1
    trsel: Union[int, None] = None
    notchfilter: bool = False
    notchfreqs: str = "[60, 120, 180, 240]"
    LPF: float = 0.0
    HPF: float = 0.0
    notchQ: float = 90.0
    detector: str = "cb"
    
class Analysis():
    """Provides handling of data analysis for IVs and MAPs
    This involves:
        1. setting various paths to the data files, output directories and
        some excel sheets that may be needed to set fitting parameters or manual annotations
        2. Selecting from the data_summary file (index of all datasets in an experiment with 
        some metadata) by date and/or slice/cell (or you can just run analysis on every entry... )
        3. Generate PDF files for output, including merging pdfs into a single large book. Seemed
        like a good idea, but the PDFs get large (approaching 1 GB). 
    """
    def __init__(self, args: object):

        # args = vars(in_args)  # convert to dict

        self._testing_counter = 0  # useful to run small tests
        self._testing_count = 400  # should be 0 in production

        self.rawdatapath = None
        self.analyzeddatapath = None
        self.directory = None
        self.inputFilename = None
        self.annotationFilename = None
        self.map_annotationFilename = None
        self.extra_subdirectories = None
        self.skip_subdirectories = None
        self.artifactFilename = None
        self.pdfFilename = None
        self.exclusions = None
        self.map_annotations = None  # this will be the map_annotationsFilename DATA (pandas from excel)
        self.experiment = args.experiment

        # selection of data for analysis
        self.update_results = args.update_results
        self.day = args.day
        self.before = DUP.parse(args.before)
        self.before_str = args.before
        self.after = DUP.parse(args.after)
        self.after_str = args.after
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
        self.plotsoff = args.plotsoff
        self.rasterize = True
        self.update = args.update
        self.noparallel = args.noparallel
        self.mapsZQA_plot = args.mapsZQA_plot
        self.recalculate_events = args.recalculate_events

        # analysis parameters

        self.ivduration = args.ivduration
        self.threshold = args.threshold
        self.refractory = args.refractory
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

        self.cell_tempdir = None
        self.annotated_dataframe: Union[pd.DataFrame, None] = None
        self.allprots = []

    
        # load up the analysis modules (don't allow multiple instances to exist)
        self.SP = EP.spike_analysis.SpikeAnalysis()
        self.RM = EP.rm_tau_analysis.RmTauAnalysis()
        self.EPIV = EP.iv_analysis.IVAnalysis()
        self.AR = DR.acq4_reader.acq4_reader()
        self.MM = MINIS.minis_methods.MiniAnalyses()
        self.AM = mapanalysistools.analyze_map_data.AnalyzeMap(rasterize=self.rasterize)
        self.AM.configure(reader=self.AR, spikeanalyzer=self.SP, rmtauanalyzer=self.RM,
                            minianalyzer=self.MM)
        if self.detector == "cb":
            self.AM.set_methodname("cb_cython")
        elif self.detector == "aj":
            self.AM.set_methodname("aj_cython")
        elif self.detector == "zc":
            raise ValueError("Detector ZC is not recommended at this time")
        else:
            raise ValueError(f"Detector {self.detector:s} is not one of [cb, aj, zc]")

        super().__init__()

    def set_exclusions(self, exclusions: Union[dict, None] = None):
        """Set the datasets that will be excluded

        Args:
            exclusions (Union[dict, None], optional): a dict of the files. Defaults to None.
        """
        self.exclusions = exclusions

    def set_experiment(self, expt: dict):
        self.experiment = expt

    def setup(self):
        if self.experiment not in ["None", None]:
            self.rawdatapath = Path(self.experiment["rawdatapath"])
            if self.experiment["directory"] is None:
                self.analyzeddatapath = Path(self.experiment["analyzeddatapath"])
                self.databasepath = Path(self.experiment["databasepath"])
            else:
                self.rawdatapath = Path(self.rawdatapath, self.experiment["directory"])
                self.analyzeddatapath = Path(self.experiment["analyzeddatapath"], self.experiment["directory"])
                self.databasepath = Path(self.experiment["databasepath"], self.experiment["directory"])
    
            self.inputFilename = Path(
                self.databasepath, self.experiment["datasummaryFilename"]
            ).with_suffix(".pkl")
            if self.experiment["pdfFilename"] is not None:
                self.pdfFilename = Path(
                    self.analyzeddatapath, self.experiment["pdfFilename"]
                ).with_suffix(".pdf")
                self.pdfFilename.unlink(missing_ok=True)
            else:
                self.pdfFilename = None

            if self.experiment["annotationFilename"] is not None:
                self.annotationFilename = Path(
                    self.analyzeddatapath, self.experiment["annotationFilename"]
                )
                if not self.annotationFilename.is_file():
                    raise FileNotFoundError(f"{str(self.annotationFilename):s}")
            else:
                self.annotationFilename = None

            if (
                "map_annotationFilename" in list(self.experiment.keys())
                and self.experiment["map_annotationFilename"] is not None
            ):
                self.map_annotationFilename = Path(
                    self.analyzeddatapath, self.experiment["map_annotationFilename"]
                )
                if not self.map_annotationFilename.is_file():
                    raise FileNotFoundError(f" missing file: {str(self.map_annotationFilename):s}")
            else:
                self.map_annotationFilename = None


             # always specify the temporary directory for intermediate plot results
            self.cell_tempdir = Path(self.analyzeddatapath, "temppdfs")

            # handle directories to include or skip
            if (
                "extra_subdirectories" in list(self.experiment.keys())
                and self.experiment["extra_subdirectories"] is not None
            ):
                self.extra_subdirectories = self.experiment["extra_subdirectories"]
            else:
                self.extra_subdirectories = None
            if (
                "skip_subdirectories" in list(self.experiment.keys())
                and self.experiment["skip_subdirectories"] is not None
            ):
                self.skip_subdirectories = self.experiment["skip_subdirectories"]
            else:
                self.skip_subdirectories = None
            if (
                "iv_analysisFilename" in list(self.experiment.keys())
                and self.experiment["iv_analysisFilename"] is not None
            ):
                self.iv_analysisFilename = Path(
                    self.analyzeddatapath, self.experiment["iv_analysisFilename"]
                )
                if len(Path(self.iv_analysisFilename).suffix) == 0:
                    raise ValueError(f"Analysis output file specified, but required extension (.h5, .pkl, or .feather) is missing: {str(Path(self.iv_analysisFilename).suffix):s}")
            else:
                self.iv_analysisFilename = None

            if ("artifactFilename" in list(self.experiment.keys()) 
                and self.experiment["artifactFilename"] is not None):
                self.artifactFilename = self.experiment["artifactFilename"]
            else:
                self.artifactFilename = None
        
        else:
            raise ValueError('Experiment was not specified"')

        if self.artifactFilename is not None and len(self.artifactFilename) > 0 :
            self.artifactFilename = Path(self.analyzeddatapath, self.artifactFilename)
            if not self.annotationFilename.is_file():
                raise FileNotFoundError
        else:
            self.artifactFilename = None

        # get the input file (from dataSummary)
        self.df = pd.read_pickle(str(self.inputFilename))
        CP.cprint("g", f"Read input file: {str(self.inputFilename):s}")
        # self.df[
        #     "day"
        # ] = None  # self.df = self.df.assign(day=None)  # make sure we have short day available
        # self.df = self.df.apply(self._add_date, axis=1)
        # self.df.drop_duplicates(['date', 'slice_slice', 'cell_cell', 'data_complete'],
        #      keep='first', inplace=True, ignore_index=True)

        if self.excel:  # just re-write as an excel and we are done
            excel_file = Path(self.analyzeddatapath, self.inputFilename.stem + ".xlsx")
            print(f"Writing to {str(excel_file):s}")
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
        self.df["annotated"] = False  # clear annotations
        if self.annotationFilename is not None:
            CP.cprint(
                "yellow", f"Reading annotation file: {str(self.annotationFilename):s}"
            )
            ext = self.annotationFilename.suffix
            if ext in [".p", ".pkl", ".pkl3"]:
                self.annotated_dataframe = pd.read_pickle(self.annotationFilename)
            elif ext in [".xls", ".xlsx"]:
                self.annotated_dataframe = pd.read_excel(self.annotationFilename)
            else:
                raise ValueError(
                    f"Do not know how to read annotation file: {str(self.annotationFilename):s}, Valid extensions are for pickle and excel"
                )
        self.update_annotations()

        if self.pdfFilename is not None:
            self.pdfFilename.unlink(missing_ok=True)

    def _show_paths(self):
        allpaths = {
            "experiment name": self.experiment,
            "raw data": self.rawdatapath,
            "analyzed data": self.analyzeddatapath,
            "input": self.inputFilename,
            "pdf": self.pdfFilename,
            "annotation (excel)": self.annotationFilename,
            "map_annotation (excel)": self.map_annotationFilename,
            "extra_subdirectories": self.extra_subdirectories,
            "artifact": self.artifactFilename,
            "exclusions": self.exclusions,
            "IVs (excel)": self.IVs,
            "IV analysis (hdft)": self.iv_analysisFilename,
        }
        print(f"\nPaths and files:")
        for p in allpaths:
            print(f"   {p:>20s}   {str(allpaths[p]):<s}")



    def run(self):
        """Perform analysis on one day, a range of days, one cell, or everything.

        Returns:
            nothing
        """
        self.n_analyzed = 0
        def _add_day(row):
            row.day = str(Path(row.date).name)
            daystr = Path(
                row.day,
                row.slice_slice,
                row.cell_cell,
            )
            return row
        
        self.df = self.df.assign(day="")
        self.df = self.df.apply(_add_day, axis=1) # add a day (short name)
        
        if self.day != "all":  # specified day
            day = str(self.day)
            print(
                f"Looking for day: {day:s} in database from {str(self.inputFilename):s}"
            )
            if "_" not in day:
                day = day + "_000"
            cells_in_day = self.df.loc[self.df.day == day]

            if len(cells_in_day) == 0:
                print("date not found: here are the valid dates:")
                for dx in self.df.date.values:
                    print(f"    day: {dx:s}")
            # print("  ... Retrieved day:\n", day_x)
            for icell in cells_in_day.index:  # for all the cells in the day
                self.do_cell(icell, pdf=self.pdfFilename)

        # get the complete protocols:
        # Only returns a dataframe if there is more than one entry
        # Otherwise, it is like a series or dict
        else:
            if self.pdfFilename is None:
                for n, icell in enumerate(range(len(self.df.index))):
                    self.do_cell(icell, pdf=None)
            else:
                with PdfPages(self.pdfFilename) as pdf:
                    for n, icell in enumerate(range(len(self.df.index))):
                        self.do_cell(icell, pdf=pdf)
            CP.cprint("c", f"Writing ALL IV analysis results to PKL file: {str(self.iv_analysisFilename):s}")
            with open(self.iv_analysisFilename, 'wb') as fh:
               self.df.to_pickle(fh, compression={'method': 'gzip', 'compresslevel': 5, 'mtime': 1})
        
        
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
        if self.annotated_dataframe is not None:
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
            self.df.loc[
                self.df.index.isin(self.annotated_dataframe.index), "annotated"
            ] = True
            if self.verbose:  # check whether it actually took
                for icell in range(len(self.df.index)):
                    print(
                        f"   {str(self.make_cellstr(self.df, icell)):>56s}  type: {self.df.iloc[icell]['cell_type']:<20s}, annotated (T,F): {str(self.df.iloc[icell]['annotated'])!s:>5} Index: {icell:d}"
                    )

        # pull in map annotations. Thesea are ALWAYS in an excel file, which should be created initially by make_xlsmap.py
        if self.map_annotationFilename is not None:
            CP.cprint("c", f"Reading map annotation file:  {str(self.map_annotationFilename):s}")
            self.map_annotations = pd.read_excel(
                Path(self.map_annotationFilename).with_suffix(".xlsx"),
                sheet_name="Sheet1",
            )

    def make_cellstr(self, df: object, icell: int, shortpath: bool = False):
        """
        Make a day string including slice and cell from the icell index in the pandas datafram df
        Example result:
            s = self.make_cellstr (df, 1)
            s: '2017.01.01_000/slice_000/cell_001'  # Mac/linux path string

        Parameters
        ----------
        df : Pandas dataframe instance

        icell : int (no default)
            index into pandas dataframe instance

        returns
        -------
        Path
        """

        if shortpath:
            day = Path(df.iloc[icell]["date"]).parts[-1]
            cellstr = Path(
                day,
                Path(df.iloc[icell]["slice_slice"]).name,
                Path(df.iloc[icell]["cell_cell"]).name,
            )
        else:
            cellstr = Path(
                df.iloc[icell]["date"],
                Path(df.iloc[icell]["slice_slice"]).name,
                Path(df.iloc[icell]["cell_cell"]).name,
            )
        # print("make_cellstr: ", daystr)
        return cellstr

    """
    Handle the temporary directory pdf accumulation and merging.
    """

    def make_tempdir(self):
        """
        Make a temporary directory; if the directory exists, just clean it out
        """
        if not self.cell_tempdir.is_dir():
            self.cell_tempdir.mkdir(mode=0o755, exist_ok=True)
        else:
            self.clean_tempdir()  # clean up

    def clean_tempdir(self):
        """
        Delete the files in the current temporary directory
        """
        fns = sorted(
            list(self.cell_tempdir.glob("*.pdf"))
        )  # list filenames in the tempdir and sort by name
        for fn in fns:  # delete the files in the tempdir
            Path(fn).unlink(missing_ok=True)
    
    def check_celltype(self, celltype: Union[str, None]=None):
        if isinstance(celltype, str):
            celltype = celltype.strip()
        if celltype in [None, "", "?", " ", "  ", "\t"]:
            celltype = "unknown"
        return celltype

    def make_cell_filename(self, celltype:str, slicecell:str):
        celltype = self.check_celltype(celltype)
       
        file_dir = str(Path(self.analyzeddatapath, celltype))
        file_name = self.thisday.replace(".", "_")
        file_name += f"_{slicecell:s}"
        if self.signflip:
            file_name += "_signflip"
        if self.alternate_fit1:
            file_name += "_alt1"
        if self.alternate_fit2:
            file_name += "_alt2"

        file_name += f"_{celltype:s}"
        if self.iv_flag:
            file_name += f"_IVs"
        elif self.mapsZQA_plot:
            file_name += f"_ZQAmaps"  # tag if using other than zscore in the map plot
        elif self.map_flag:
            file_name += "_maps"
        elif self.vc_flag:
            file_name += "_VC"

        file_name = Path(file_name)
        return file_name
    
    def merge_pdfs(self, celltype: Union[str, None] = None, slicecell:Union[str, None]=None, pdf=None, overwrite:bool=True):
        """
        Merge the PDFs in tempdir with the pdffile (self.pdfFilename)
        The tempdir PDFs are deleted once the merge is complete.
        Merging should be done on a per-cell basis, and on a per-protocol class basis.

        """
        celltype = self.check_celltype(celltype)
        if self.dry_run or not self.autoout:
            print("Dry run or not automatic output")
            return
        if not self.merge_flag:
            print("Merge flag is False")
            return
        CP.cprint('c', "********* MERGEING PDFS ************\n")
        # if self.pdfFilename is None and not self.autoout:  # no output file, do nothing
        #     return

        # check autooutput and reset pdfFilename if true:
        # CP.cprint("c",
        #     f"Merging pdfs at {datetime.datetime.now().strftime('%m/%d/%Y, %H:%M:%S'):s}"
        # )
        if slicecell is None:
            raise ValueError("iv_analysis:merge_pdfs:: Slicecell is None: should always have a value set")
        if self.autoout:
            # print("  in auto mode")
            pdfname = self.make_cell_filename(celltype, slicecell)
            pdfname = Path(pdfname)
            # check to see if we have a sorted directory with this cell type
            pdfdir = Path(self.analyzeddatapath, celltype)
            if not pdfdir.is_dir():
                pdfdir.mkdir()
            self.cell_pdfFilename = Path(pdfdir, pdfname.stem).with_suffix(".pdf")
        fns = sorted(
            list(self.cell_tempdir.glob("*.pdf"))
        )  # list filenames in the tempdir and sort by name
        if len(fns) == 0:
            CP.cprint("m", f"No pdfs to merge for {str(self.cell_pdfFilename):s}")
            return  # nothing to do
        CP.cprint("c", f"Merging pdf files: {str(fns):s}")
        CP.cprint("c", f"    into: {str(self.cell_pdfFilename):s}")
      
        # cell file merged
        mergeFile = PdfMerger()
        if overwrite:  # remove existing file
            self.cell_pdfFilename.unlink(missing_ok=True)
        fns.insert(0, str(self.cell_pdfFilename))
        for i, fn in enumerate(fns):
            if Path(fn).is_file():
                mergeFile.append(PdfReader(open(fn, "rb")))
        with open(self.cell_pdfFilename, "wb") as fout:
            mergeFile.write(fout)
        CP.cprint("g", f"Wrote output pdf to : {str(self.cell_pdfFilename):s}")
        fns.pop(0)
        # remove temporary files
        for fn in fns:
            fn.unlink(missing_ok=True)
        

        # big file merge
        # removing this as it really slows down the analysis
        # and generates a big file that we can't read = better to 
        # do this with a separate routine AFTER the analysis is complete
        # if self.pdfFilename is not None:
        #     mergeFile = PdfMerger()
        #     fns = [str(self.pdfFilename), str(self.cell_pdfFilename)]
        #     for i, fn in enumerate(fns):
        #         fn = Path(fn)
        #         if fn.is_file() and fn.stat().st_size > 0:
        #             mergeFile.append(PdfReader(open(fn, "rb")))
        #     with open(self.pdfFilename, "wb") as fout:
        #         mergeFile.write(fout)
        print("="*80)
        print()

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
        for i, x in enumerate(prox):  # construct filenames and sort by analysis types
            if len(x) == 0:
                continue
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
            c = str(c)  # make sure it is serializable for later on with JSON.
            # maps
            if x.startswith("Map"):
                allprots["maps"].append(c)
            if x.startswith("VGAT_"):
                allprots["maps"].append(c)
            if x.startswith("Vc_LED"):  # these are treated as maps, even though they are just repeated... 
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

    def make_cell(self, icell: int):
        datestr = Path(self.df.iloc[icell]["date"]).name
        slicestr = str(Path(self.df.iloc[icell]["slice_slice"]).parts[-1])
        cellstr = str(Path(self.df.iloc[icell]["cell_cell"]).parts[-1])
        return (datestr, slicestr, cellstr)

    def find_cell(
        self,
        df: pd.DataFrame,
        datestr: str,
        slicestr: str,
        cellstr: str,
        protocolstr: Union[Path, str, None] = None,
    ):
        """Find the dataframe element for the specified date, slice, cell and
        protocol in the input dataframe

        Parameters
        ----------
        df : Pandas dataframe
            The dataframe for the experiments, as a Pandas frame
        datestr : Union[str, datetime.datetime]
            The date string we are looking for ("2017.04.19_000")
            This should NOT have the full path to the data
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
        qstring = f"date == '{dstr:s}'"
        cf = pd.DataFrame()
        colnames = df.columns
        
        mapcolumn = "map"  # old version
        if "mapname" in colnames:
            mapcolumn = "mapname"  # newer versions
        if protocolstr is None:
            cf = df.query(
                f'date == "{dstr:s}" & slice_slice == "{slicestr:s}" & cell_cell == "{cellstr:s}"'
            )
        else:
            dprot = str(protocolstr.name)
            cf = df.query(
                f'date == "{dstr:s}" & slice_slice == "{slicestr:s}" & cell_cell == "{cellstr:s}" & {mapcolumn:s} == "{dprot:s}"'
            )
        return cf

    def get_celltype(self, icell):
        """Find the type of the cell associated with this entry

        Parameters
        ----------
        icell : int
            The index into the table

        Returns
        -------
        str, bool
            The celltype
            If the celltype changed in annotation, the bool value will be True,
            otherwise False
        """
        original_celltype = self.df.at[icell, "cell_type"]
        datestr, slicestr, cellstr = self.make_cell(icell)

        if self.annotated_dataframe is None:
            return original_celltype, None
        # print("have annotated df")
        cell_df = self.find_cell(self.annotated_dataframe, datestr, slicestr, cellstr)
        if cell_df.empty:  # cell was not in annotated dataframe
            if self.verbose:
                print(datestr, cellstr, slicestr, " Not annotated")
            return original_celltype, False

        celltype = cell_df["cell_type"].values[0].strip()
        if not pd.isnull(celltype) and not isinstance(celltype, str):
            if self.verbose:
                print(
                    "   Annotated dataFrame: celltype = ",
                    cell_df["cell_type"],
                    "vs. ",
                    original_celltype,
                )
            CP.cprint("yellow", f"   Annotation did not change celltype: {celltype:s}")
            return original_celltype, False

        if not pd.isnull(celltype) and isinstance(celltype, str):
            if celltype != original_celltype:
                CP.cprint(
                    "red",
                    f"   Cell re-annotated celltype: {celltype:s} (original: {original_celltype:s})",
                )
                return celltype, True
            else:
                if self.verbose:
                    print(
                        "celltype and original cell type are the same: ",
                        original_celltype,
                    )
                return original_celltype, False

    def do_cell(self, icell: int, pdf=None):
        """
        Do analysis on one cell's
        Runs all protocols for the cell 

        Includes a dispatcher for different kinds of protocols: IVs, Maps, VCs
        
        Parameters
        ----------
        icell : int
            index into pandas database for the day to be analyzed

        pdf : bool, default=False

        """
        celltype, celltypechanged = self.get_celltype(icell)
        datestr, slicestr, cellstr = self.make_cell(icell)
        # celltype = self.df.iloc[icell].cell_type
        celltype = self.check_celltype(celltype)
        self.df.iloc[icell].cell_type = celltype  # force relabel in database

        slicecell = f"S{int(slicestr[-3:]):02d}C{int(cellstr[-3:]):02d}"  # recognize that slices and cells may be more than 10 (# 9)
        dsday, nx = Path(datestr).name.split("_")
        self.thisday = dsday

        thisday = datetime.datetime.strptime(dsday, "%Y.%m.%d")
        if thisday < self.after or thisday > self.before:
            CP.cprint(
                "y",
                f"Day {datestr:s} is not in range {self.after_str:s} to {self.before_str:s}",
            )
            return

        fullfile = Path(
            self.rawdatapath, self.df.iloc[icell].cell_id)
        #)
        # print("fullfile: ", fullfile, fullfile.is_dir())
        if self.skip_subdirectories is not None:
            # skip matching subdirectories
            for skip in self.skip_subdirectories:
                # print(f"Checking skip = {skip:s} with {str(fullfile):s}")
                if str(fullfile).find(skip) >= 0:
                    ffparts = fullfile.parts
                    fftail = str(Path(*ffparts[len(self.rawdatapath.parts)-1:]))
                    CP.cprint(
                        "r",
                        f"SKIPPING data/date: {fftail:s}  containing: {skip:s}",
                    )
                    return  # skip this data set

        if not fullfile.is_dir():
            fullfile = Path(self.df.iloc[icell]["data_directory"],  self.make_cellstr(self.df, icell, shortpath=True))

        if self.extra_subdirectories is not None and not fullfile.is_dir():
            # try extra sub directories
            pathparts = fullfile.parts
            day = None
            for i, p in enumerate(pathparts):
                if p.startswith("20"):
                    day = Path(*pathparts[i:])
                    break
            if day is None:
                CP.cprint("r", f"do_cell: Day <None> found in fileparts: {str(pathparts):s}")
                exit()
            for subdir in self.extra_subdirectories:
                fullfile = Path(self.rawdatapath, subdir, day)
                if fullfile.is_dir():
                    break

            if not fullfile.is_dir():
                CP.cprint("r", f"Unable to get the file: {str(fullfile):s}")
                return


        prots = self.df.iloc[icell]["data_complete"]
        allprots = self.gather_protocols(prots.split(", "), self.df.iloc[icell])

        if self.dry_run:
            print(
                f"\nIV_Analysis:do_cell:: Would process day: {datestr:s} slice: {slicestr:s} cell: {cellstr:s}"
            )
            print(f"        with fullpath {str(fullfile):s}")

        if not fullfile.is_dir():
            CP.cprint("r", "   But that cell was not found.")
            print("*" * 40)
            print(self.df.iloc[icell])
            print("*" * 40)
            print()
            self.merge_pdfs(celltype, pdf=pdf)
            return

        elif fullfile.is_dir() and len(allprots) == 0:
            CP.cprint("m", "   Cell found, but no protocols were found")
            return
        elif fullfile.is_dir() and len(allprots) > 0:
            for prottype in allprots.keys():
                for prot in allprots[prottype]:
                    ffile = Path(self.df.iloc[icell].data_directory, prot)
                    if not ffile.is_dir():
                        CP.cprint("r", f"file/protocol day={icell:d} not found: {str(ffile):s}")
                        print(prottype,  prot)
                        exit()
        else:
            msg = f"   Cell OK, with {len(allprots['stdIVs'])+len(allprots['CCIV_long']):4d} IV protocols"
            msg += f" and {len(allprots['maps']):4d} map protocols"
            msg += f"  Electrode: {self.df.iloc[icell]['internal']:s}"
            CP.cprint("g", msg)
            if self.map_flag:
                for i, p in enumerate(sorted(prots)):
                    if p in allprots["maps"]:
                        print("      {0:d}. {1:s}".format(i + 1, str(p.name)))
        # if self.dry_run:
        #     return

        if self.verbose:
            for k in list(allprots.keys()):
                print("protocol type: {:s}:".format(k))
                for m in allprots[k]:
                    print("    {0:s}".format(str(m)))
                if len(allprots[k]) == 0:
                    print("    No protocols of this type")
            print("All protocols: ")
            print([allprots[p] for p in allprots.keys()])

        # self.merge_pdfs(celltype, pdf=pdf)
        # CP.cprint("r", f"iv flag: {str(self.iv_flag):s}")
        
        # DISPATCH according to requested analysis:
        if self.iv_flag:
            if self.iv_analysisFilename is not None and Path(self.iv_analysisFilename).suffix == ".h5":
                self.df["IV"] = None
                self.df["Spikes"] = None
            if self.cell_tempdir is not None:
                self.make_tempdir()  # clean up temporary directory
            self.analyze_ivs(
                icell=icell, allprots=allprots, celltype=celltype, pdf=pdf
            )
            self.merge_pdfs(celltype, slicecell=slicecell, pdf=pdf)  

            # store pandas db analysis of this cell in a pickled file:
            pklname = self.make_cell_filename(celltype, slicecell)
            pklname = Path(pklname)
            # check to see if we have a sorted directory with this cell type
            pkldir = Path(self.analyzeddatapath, celltype)
            if not pkldir.is_dir():
                pkldir.mkdir()
            self.cell_pklFilename = Path(pkldir, pklname.stem).with_suffix(".pkl")
            CP.cprint("c", f"Writing cell IV analysis results to PKL file: {str(self.cell_pklFilename):s}")
            if self.df.iloc[icell]['Spikes'] is None:
                CP.cprint("r", f"   @@@ Spikes array is empty @@@")
            else:
                CP.cprint('g', f"*** Have Spikes keys: {str(self.df.iloc[icell]['Spikes'].keys()):s}")
            with open(self.cell_pklFilename, 'wb') as fh:
               self.df.iloc[icell].to_pickle(fh, compression={'method': 'gzip', 'compresslevel': 5, 'mtime': 1})

            # print(self.df.iloc[icell].keys())
            # exit()
            # self.df["IV"] = None
            # self.df["Spikes"] = None
            gc.collect()
            

        if self.vc_flag:
            self.analyze_vcs(icell, allprots)
            gc.collect()

        if self.map_flag:
            if self.cell_tempdir is not None:
                self.make_tempdir()
            self.analyze_maps(icell=icell, allprots=allprots, celltype=celltype, pdf=pdf)
            # analyze_maps stores events in an events subdirectory by cell
            # It also merges the PDFs for that cell in the celltype directory
            # CP.cprint("r", f"Merging pdfs, with: {str(pdf):s}")
            # self.merge_pdfs(celltype, slicecell=slicecell, pdf=pdf)

            gc.collect()

    

    def analyze_ivs(
        self, icell, allprots: dict, celltype: str, pdf=None,
    ):
        """
        Overall analysis of IV protocols for one cell in the day

        Parameters
        ----------
        icell : int
            index into Pandas database for the selected cell

        allprots : dict
            dictionary of protocols for the day/slice/cell

        pdf : None
            if not none, then is the pdffile to write the data to

        Returns
        -------
        Nothing - generates pdfs and updates the pickled database file.
        """
        CP.cprint(
            "c",
            f"analyze ivs for index: {icell: d} dir: {str(self.df.iloc[icell].data_directory):s}  cell: ({str(self.df.iloc[icell].cell_id):s} )",
        )
        cell_directory = Path(self.df.iloc[icell].data_directory, self.df.iloc[icell].cell_id )
        CP.cprint("m", f"file: {str(cell_directory):s}")
        CP.cprint("m", f"Cell id: {str(self.df.iloc[icell].cell_id):s} ")
        if "IV" not in self.df.columns.values:
            self.df = self.df.assign(IV=None)
        if "Spikes" not in self.df.columns.values:
            self.df = self.df.assign(Spikes=None)

        CP.cprint(
            "c",
            f"      Cell: {str(cell_directory):s}\n           at: {datetime.datetime.now().strftime('%m/%d/%Y, %H:%M:%S'):s}",
        )

        # clean up data in IV and SPikes : remove Posix
        def _cleanup_ivdata(results: dict):
            import numpy
            if isinstance(results, dict) and len(results) == 0:
                results = None
            if results is not None:
                for r in list(results.keys()):
                    u = results[r]
                    for k in u.keys():
                        if isinstance(u[k], dict):
                            for uk in u[k].keys():
                                if isinstance(u[k][uk], bool) or isinstance(
                                    u[k][uk], np.int32
                                ):
                                    u[k][uk] = int(u[k][uk])
                        if k in ["taupars", "RMPs", "Irmp"]:
                            # if isinstance(u[k], Iterable) and not isinstance(u[k], (dict, list, float, str, np.float)):
                            # print("type for ", k, type(u[k]))
                            if isinstance(u[k], numpy.ndarray):
                                u[k] = u[k].tolist()
                            elif isinstance(u[k], list) and len(u[k]) > 0:
                                if isinstance(u[k][0], numpy.ndarray):
                                    u[k] = u[k][0].tolist()
                            elif isinstance(u[k], np.float64):
                                u[k] = float(u[k])
                            elif isinstance(u[k], list):
                                pass
                            else:
                                print(type(u[k]))
                                raise ValueError
                        # if isinstance(u[k], Iterable) and not isinstance(u[k], (dict, list, float, str)):
                        #     u[k] = u[k].tolist()
                        elif isinstance(u[k], np.float64):
                            u[k] = float(u[k])
                    results[str(r)] = results.pop(r)
            return results

        def _cleanup_spikedata(results: dict):
            if isinstance(results, dict) and len(results) == 0:
                results = None

            if results is not None:
                for r in list(results.keys()):
                    results[str(r)] = results.pop(r)
            return results

        def cleanIVs(row):
            if not isinstance(row.IV, dict):
                return row.IV
            newrow = {}
            for k, v in row.IV.items():
                newrow[str(k)] = row.IV[k]
            _cleanup_ivdata(newrow)
            return newrow

        def cleanSpikes(row):
            if not isinstance(row.Spikes, dict):
                return row.Spikes
            newrow = {}
            for k, v in row.Spikes.items():
                newrow[str(k)] = row.Spikes[k]
            _cleanup_ivdata(newrow)
            return newrow

        self.df["IV"] = self.df.apply(cleanIVs, axis=1)
        self.df["Spikes"] = self.df.apply(cleanSpikes, axis=1)

        nfiles = 0
        allivs = []
        for ptype in allprots.keys():  # check all the protocols
            if ptype in ["stdIVs", "CCIV_long"]:  # just CCIV types
                for prot in allprots[ptype]:
                    allivs.append(prot)  # combine into a new list
        validivs = []
        if self.exclusions is not None:
            for p in allivs:  # first remove excluded protocols
                if p not in self.exclusions:
                    validivs.append(
                        p
                    )  # note we do not just remove as this messes up the iterator of the maps
        else:
            validivs = allivs
        nworkers = 16  # number of cores/threads to use
        tasks = range(len(validivs))  # number of tasks that will be needed
        results = dict(
            [("IV", {}), ("Spikes", {})]
        )  # storage for results; predefine the dicts.
        if self.noparallel:  # just serial...
            for i, x in enumerate(tasks):
                r, nfiles = self.analyze_iv(icell=icell, i=i, x=x, cell_directory=cell_directory, allivs=validivs, nfiles=nfiles, pdf=pdf)
                if self.dry_run:
                    continue
                if r is None:
                    continue
                results["IV"][validivs[i]] = r["IV"]
                results["Spikes"][validivs[i]] = r["Spikes"]
            results["IV"] = _cleanup_ivdata(results["IV"])
            results["Spikes"] = _cleanup_ivdata(results["Spikes"])
            if not self.dry_run:
                self.df.at[icell, "IV"] = results[
                    "IV"
                ]  # everything in the RM analysis_summary structure
                self.df.at[icell, "Spikes"] = results[
                    "Spikes"
                ]  # everything in the SP analysus_summary structure
        #            print(self.df.at[icell, 'IV'])
        else:
            result = [None] * len(tasks)  # likewise
            with mp.Parallelize(
                enumerate(tasks), results=results, workers=nworkers
            ) as tasker:
                for i, x in tasker:
                    result, nfiles = self.analyze_iv(
                        icell, i, x, cell_directory, validivs, nfiles, pdf=pdf
                    )
                    tasker.results[validivs[i]] = result
            # reform the results for our database
            if self.dry_run:
                return
            riv = {}
            rsp = {}
            for f in list(results.keys()):
                if "IV" in results[f]:
                    riv[f] = _cleanup_ivdata(results[f]["IV"])
                if "Spikes" in results[f]:
                    rsp[f] = _cleanup_ivdata(results[f]["Spikes"])
            #            print('parallel results: \n', [(f, results[f]['IV']) for f in results.keys() if 'IV' in results[f].keys()])
            #            print('riv: ', riv)

            self.df.at[
                icell, "IV"
            ] = riv  # everything in the RM analysis_summary structure
            self.df.at[
                icell, "Spikes"
            ] = rsp  # everything in the SP analysus_summary structure

        self.df["annotated"] = self.df["annotated"].astype(int)
        self.df["expUnit"] = self.df["expUnit"].astype(int)

        if len(allivs) > 0 and self.iv_analysisFilename is not None and Path(self.iv_analysisFilename).suffix == ".h5":
            CP.cprint("m", f"Writing IV analysis results to HDF5 file: {str(self.iv_analysisFilename):s}")
            # with hdf5:
            # Note, reading this will be slow - it seems to be rather a large file.
            day, slice, cell = self.make_cell(icell=icell)
            keystring = str(
                Path(Path(day).name, slice, cell)
            )  # the keystring is the cell.
            # pytables does not like the keystring starting with a number, or '.' in the string
            # so put "d_" at start, and then replace '.' with '_'
            # what a pain.
            keystring = 'd_'+keystring.replace('.', '_')
            if self.n_analyzed == 0:
                self.df.iloc[icell].to_hdf(self.iv_analysisFilename, key=keystring, mode="w")
            else:
                self.df.iloc[icell].to_hdf(self.iv_analysisFilename, key=keystring, mode="a")
            # self.df.at[icell, "IV"] = None
            # self.df.at[icell, "Spikes"] = None
            self.n_analyzed += 1
            gc.collect()

        elif len(allivs) > 0 and Path(self.iv_analysisFilename).suffix == ".pkl":
            # with pickle and compression (must open with gzip, then read_pickle)
            # CP.cprint("c", f"Writing IV analysis results to PKL file: {str(self.iv_analysisFilename):s}")
            # with open(self.iv_analysisFilename, 'wb') as fh:
            #    self.df.to_pickle(fh, compression={'method': 'gzip', 'compresslevel': 5, 'mtime': 1})
            pass

        elif len(allivs) > 0 and Path(self.iv_analysisFilename).suffix == ".feather":
            # with feather (experimental)
            CP.cprint("b", f"Writing IV analysis results to FEATHER file: {str(self.iv_analysisFilename):s}")
            with open(self.iv_analysisFilename, 'wb') as fh:
               self.df.to_feather(fh)

    def analyze_iv(
        self,
        icell: int,
        i: int,
        x: int,
        cell_directory: Union[Path, str],
        allivs: list,
        nfiles: int,
        pdf: None,
    ):
        """
        Compute various measures (input resistance, spike shape, etc) for one IV
        protocol in the day. Designed to be used in parallel or serial mode

        Parameters
        ----------
        icell : int
            index into Pandas database for the day
        i : int
            index into the list of protocols
        x : task
            task number
        file: full filename and path to the IV protocol data
        allivs :
            list of protocols
        nfiles : int
            number for file...

        pdf: matplotlib pdfpages instance or None
            if None, no plots are accumulated
            if a pdfpages instance, all of the pages plotted by IVSummary are concatenated
            into the output file (self.pdfFilename)
        """
        # import matplotlib.pyplot as mpl  # import locally to avoid parallel problems

        protocol = Path(allivs[i]).name
        result = {}
        iv_result = {}
        sp_result = {}

        protocol_directory = Path(cell_directory, protocol)
        
        if not protocol_directory.is_dir():
            print("cell directory: ", cell_directory)
            print("protocol: ", protocol)
            CP.cprint("r", f"protocol directory not found (analyze_iv)!! {str(protocol_directory):s}")
            exit()
        self.EPIV.configure(protocol_directory, plot=not self.plotsoff,
                          reader = self.AR,
                          spikeanalyzer=self.SP,
                          rmtauanalyzer=self.RM)
        if self.iv_select["duration"] > 0.0:
            check = self.EPIV.iv_check(duration=self.iv_select["duration"])
            if check is False:
                gc.collect()
                return (None, 0)  # skip analysis
        if not self.dry_run:
            print(f"      IV analysis for {str(protocol_directory):s}")
            br_offset = 0.0
            if (
                not pd.isnull(self.df.at[icell, "IV"])
                and protocol in self.df.at[icell, "IV"]
                and "--Adjust" in self.df.at[icell, protocol]["BridgeAdjust"]
            ):
                print(
                    "Bridge: {0:.2f} Mohm".format(
                        self.df.at[icell, "IV"][protocol]["BridgeAdjust"] / 1e6
                    )
                )
            ctype = self.df.at[icell, "cell_type"].lower()
            tgap = 0.0015
            tinit = True
            if ctype in [
                "bushy",
                "d-stellate",
                "octopus",
            ]:
                tgap = 0.0005  # shorten gap for measures for fast cell types
                tinit = False
            self.EPIV.plot_mode(mode=self.IV_pubmode)
            plot_handle = self.EPIV.compute_iv(
                threshold=self.spike_threshold,
                refractory=self.refractory,
                bridge_offset=br_offset,
                tgap=tgap,
                plotiv=True,
                full_spike_analysis=True,
            )
            iv_result = self.EPIV.RM.analysis_summary
            sp_result = self.EPIV.SP.analysis_summary
            result["IV"] = iv_result
            result["Spikes"] = sp_result
            ctype = self.df.at[icell, "cell_type"]
            annot = self.df.at[icell, "annotated"]
            if annot:
                ctwhen = "[revisited]"
            else:
                ctwhen = "[original]"
            print("Checking for figure, plothandle is: ", plot_handle)
            if plot_handle is not None:
                shortpath = protocol_directory.parts
                shortpath = str(Path(*shortpath[4:]))
                plot_handle.suptitle(
                    f"{str(shortpath):s}\n{BIS.build_info_string(self.EPIV.AR, protocol_directory):s}",
                    fontsize=8,
                )
                t_path = Path(self.cell_tempdir, "temppdf_{0:04d}.pdf".format(nfiles))
                # print("PDF: ", pdf)
                # if pdf is not None:
                #     pdf.savefig(plot_handle)
                # else:
                mpl.savefig(
                    t_path, dpi=300
                )  # use the map filename, as we will sort by this later
                mpl.close(plot_handle)
                CP.cprint("g", f"saved to: {str(t_path):s}")
                nfiles += 1
            del iv_result
            del sp_result
            gc.collect()
            return result, nfiles

        else:
            print("Dry Run: would analyze %s" % Path(self.rawdatapath, protocol))
            br_offset = 0

            if self.df.at[icell, "IV"] == {} or pd.isnull(self.df.at[icell, "IV"]):
                print("   current database has no IV data set for this file")
            elif protocol not in list(self.df.at[icell, "IV"].keys()):
                print(
                    "Protocol {0:s} not found in day: {1:s}".format(
                        str(protocol), self.df.at[icell, "date"]
                    )
                )
            elif "BridgeAdjust" in self.df.at[icell, "IV"][protocol].keys():
                br_offset = self.df.at[icell, "IV"][protocol]["BridgeAdjust"]
                print("   with Bridge: {0:.2f}".format(br_offset / 1e6))
            else:
                print("... has no bridge, will use 0")
            gc.collect()
            return None, 0

    def analyze_vcs(self, icell: int, allprots: dict, pdf=None):
        """
        Overall analysis of VC protocols
        Incomplete - mostly just plots the data

        Parameters
        ----------
        icell : int
            index into Pandas database for the day

        allprots : dict
            dictionary of protocols for the day/slice/cell

        Returns
        -------
        Nothing - generates pdfs and updates the pickled database file.
        """
        import matplotlib.pyplot as mpl
        celltype = self.check_celltype(self.df.iloc[icell].cell_type)
        self.df.iloc[icell].cell_type = celltype

        self.make_tempdir()
        for pname in ["VCIVs"]:  # the different VCIV protocols
            if len(allprots[pname]) == 0:
                continue
            for f in allprots[pname]:
                if not self.dry_run:
                    print("Analyzing %s" % Path(self.rawdatapath, f))
                    EPVC = EP.vc_summary.VCSummary(
                        Path(self.rawdatapath, f),
                        plot=False,
                    )
                    plotted = EPVC.compute_iv()
                    if plotted:
                        t_path = Path(
                            self.cell_tempdir, "temppdf_{0:04d}.pdf".format(nfiles)
                        )
                        mpl.savefig(
                            t_path, dpi=300
                        )  # use the map filename, as we will sort by this later
                        nfiles += 1
                        # if pdf is not None and mapok:
                        #     pdf.savefig(dpi=300)
                        mpl.close(EPVC.IVFigure)
                    mpl.close(EPVC.IVFigure)
        self.merge_pdfs(celltype=celltype, pdf=pdf)


# This is old code. Use project-specific files instead.
# def main():

#     # import warnings  # need to turn off a scipy future warning.
#     # warnings.filterwarnings("ignore", category=FutureWarning)
#     # warnings.filterwarnings("ignore", category=UserWarning)
#     # warnings.filterwarnings("ignore", message="UserWarning: findfont: Font family ['sans-serif'] not found. Falling back to DejaVu Sans")

#     experiments = nf107.set_expt_paths.get_experiments()
#     exclusions = nf107.set_expt_paths.get_exclusions()
#     args = AnalysisParams.getCommands(experiments)  # get from command line

#     if args.rawdatapath is None:
#         X, args.rawdatapath, code_dir = set_expt_paths.get_paths()
#         if args.rawdatapath is None:
#             raise ValueError("No path set for computer")

#     if args.configfile is not None:
#         config = None
#         if args.configfile is not None:
#             if ".json" in args.configfile:
#                 # The escaping of "\t" in the config file is necesarry as
#                 # otherwise Python will try to treat is as the string escape
#                 # sequence for ASCII Horizontal Tab when it encounters it
#                 # during json.load
#                 config = json.load(open(args.configfile))
#             elif ".toml" in args.configfile:
#                 config = toml.load(open(args.configfile))

#         vargs = vars(args)  # reach into the dict to change values in namespace
#         for c in config:
#             if c in args:
#                 # print("c: ", c)
#                 vargs[c] = config[c]
#     CP.cprint(
#         "cyan",
#         f"Starting analysis at: {datetime.datetime.now().strftime('%m/%d/%Y, %H:%M:%S'):s}",
#     )
#     IV = IV_Analysis(args)
#     IV.setup(args, exclusions)
#     IV.run()



#     # allp = sorted(list(set(NF.allprots)))
#     # print('All protocols in this dataset:')
#     # for p in allp:
#     #     print('   ', path)
#     # print('---')
#     #
#     CP.cprint(
#         "cyan",
#         f"Finished analysis at: {datetime.datetime.now().strftime('%m/%d/%Y, %H:%M:%S'):s}",
#     )


if __name__ == "__main__":
    pass
