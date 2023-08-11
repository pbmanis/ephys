"""
Module for analyzing electrophysiology data.

This entry point is used as a wrapper for multiple experimental protocols,
including IV curves (current and voltage clamp), and optogenetic experiments, including
laser scanning photstimulation and glutamate uncaging maps.
"""
import argparse
import gc
import json
import logging
import sys
from collections.abc import Iterable
from multiprocessing import set_start_method

if sys.version_info[0] < 3:
    raise Exception("Must be using Python 3")
import datetime
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Union, List, Dict, Tuple

import dateutil
import dateutil.parser as DUP
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("QtAgg")
import matplotlib.pyplot as mpl  # import locally to avoid parallel problems
import pylibrary.plotting.plothelpers as PH
import pylibrary.tools.cprint as CP
import pyqtgraph as pg
import pyqtgraph.multiprocess as mp
from matplotlib.backends.backend_pdf import PdfPages
from PyPDF2 import PdfMerger, PdfReader, PdfWriter

import ephys.datareaders as DR
import ephys.ephys_analysis as EP
import ephys.mapanalysistools as mapanalysistools
import ephys.mini_analyses as MINIS
import ephys.tools.build_info_string as BIS

from . import analysis_parameters as AnalysisParams

PMD = mapanalysistools.plot_map_data.PlotMapData()

# do not silently pass some common numpy errors
np.seterr(divide="raise", invalid="raise")

Logger = logging.getLogger("AnalysisLogger")


def def_empty_list():
    return []


@dataclass
class cmdargs:
    """This data class holds the "command arguments" that control
    data anlaysis. These arguments are defined in 3 blocks:
        1. Experiment specification and file names
        2. Analysis flags and experiment selection
        3. Graphics controls
        4. Specific analysis parameters

    These may be set either by IV_Analysis_params from the commandline,
    or by specifying them programatically.

    """

    # Files and filenames:
    experiment: Union[str, List, None] = None
    rawdatapath: Union[str, Path, None] = None
    directory: Union[str, Path, None] = None
    analyzeddatapath: Union[str, Path, None] = None
    databasepath: Union[str, Path, None] = None
    inputFilename: Union[str, Path, None] = None
    pdfFilename: Union[str, Path, None] = None
    cell_annotationFilename: Union[str, Path, None] = None
    artifactFilename: Union[str, Path, None] = None
    map_annotationFilename: Union[str, Path, None] = None
    map_pdfs: bool = False
    iv_analysisFilename: Union[str, Path, None] = None
    extra_subdirectories: List = field(default_factory=def_empty_list)

    # analysis flags and experiment selection
    update_results: bool = False
    day: str = "all"
    after: str = "1970.1.1"
    before: str = "2266.1.1"
    slicecell: Union[str, None] = None
    protocol: Union[str, None] = None
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
    refractory: float = 0.0007  # absolute refractory period
    signflip: bool = False
    alternate_fit1: bool = False
    alternate_fit2: bool = False  # second alternate
    measuretype: str = "ZScore"  # display measure for spot plot in maps
    spike_threshold: float = -0.035
    artifact_suppression: bool = False
    artifact_derivative: bool = False
    post_analysis_artifact_rejection: bool = False
    whichstim: int = -1
    trsel: Union[int, None] = None
    notchfilter: bool = False
    notchfreqs: str = "[60, 120, 180, 240]"
    LPF: float = 0.0
    HPF: float = 0.0
    notchQ: float = 90.0
    detrend_method: Union[str, None] = None
    detrend_order: int = 5
    detector: str = "cb"


class Analysis:
    """Provides handling of data analysis for IVs and MAPs
    This involves:
        1. setting various paths to the data files, output directories and
        some excel sheets that may be needed to set fitting parameters or manual annotations
        2. Selecting from the data_summary file (index of all datasets in an experiment with
        some metadata) by date and/or slice/cell (or you can just run analysis on every entry... )
        3. Generate PDF files for output, including merging pdfs into a single large book. Seemed
        like a good idea, but the PDFs get large (approaching 1 GB).
    """

    def __init__(self, args: argparse.Namespace):
        # args = vars(in_args)  # convert to dict

        self._testing_counter = 0  # useful to run small tests
        self._testing_count = 400  # should be 0 in production

        self.rawdatapath: Path = Path()
        self.analyzeddatapath = None
        self.directory = None
        self.inputFilename = None
        self.cell_annotationFilename = None
        self.map_annotationFilename = None
        self.extra_subdirectories = None
        self.skip_subdirectories = None
        self.artifactFilename = None
        self.pdfFilename = None
        self.exclusions: Dict = {}
        self.map_annotations = (
            None  # this will be the map_annotationsFilename DATA (pandas from excel)
        )
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
        self.artifact_suppression = args.artifact_suppression
        self.artifact_derivative = args.artifact_derivative
        self.post_analysis_artifact_rejection = args.post_analysis_artifact_rejection
        self.whichstim = args.whichstim
        self.trsel = args.trsel
        self.notchfilter = args.notchfilter
        if args.notchfreqs is not None:
            self.notchfreqs = eval(args.notchfreqs)
        else:
            self.notchfreqs = None
        self.LPF = args.LPF
        self.HPF = args.HPF
        self.notch_Q = args.notchQ
        self.detrend_method = args.detrend_method
        self.detrend_order = args.detrend_order
        self.detector = args.detector

        self.cell_tempdir = None
        self.annotated_dataframe: Union[pd.DataFrame, None] = None
        self.allprots: List = []

        # load up the analysis modules (don't allow multiple instances to exist)
        self.SP = EP.spike_analysis.SpikeAnalysis()
        self.RM = EP.rm_tau_analysis.RmTauAnalysis()
        self.AR = DR.acq4_reader.acq4_reader()
        self.MA = MINIS.minis_methods.MiniAnalyses()
        self.AM = mapanalysistools.analyze_map_data.AnalyzeMap(rasterize=self.rasterize)
        self.AM.configure(
            reader=self.AR,
            spikeanalyzer=self.SP,
            rmtauanalyzer=self.RM,
            minianalyzer=self.MA,
        )
        if self.detector == "cb":
            self.AM.set_methodname("cb_cython")
        elif self.detector == "aj":
            self.AM.set_methodname("aj_cython")
        elif self.detector == "zc":
            raise ValueError("Detector ZC is not recommended at this time")
        else:
            raise ValueError(f"Detector {self.detector:s} is not one of [cb, aj, zc]")

        super().__init__()

    def set_exclusions(self, exclusions: dict):
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
                self.analyzeddatapath = Path(
                    self.experiment["analyzeddatapath"], self.experiment["directory"]
                )
                self.databasepath = Path(
                    self.experiment["databasepath"], self.experiment["directory"]
                )

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

            if self.experiment["cell_annotationFilename"] is not None:
                self.cell_annotationFilename = Path(
                    self.analyzeddatapath, self.experiment["annotationFilename"]
                )
                if not self.cell_annotationFilename.is_file():
                    raise FileNotFoundError(f"{str(self.cell_annotationFilename):s}")
            else:
                self.cell_annotationFilename = None

            if (
                "map_annotationFilename" in list(self.experiment.keys())
                and self.experiment["map_annotationFilename"] is not None
            ):
                self.map_annotationFilename = Path(
                    self.analyzeddatapath, self.experiment["map_annotationFilename"]
                )
                if not self.map_annotationFilename.is_file():
                    raise FileNotFoundError(
                        f" missing file: {str(self.map_annotationFilename):s}"
                    )
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
                    raise ValueError(
                        f"Analysis output file specified, but required extension (.h5, .pkl, or .feather) is missing: {str(Path(self.iv_analysisFilename).suffix):s}"
                    )
            else:
                self.iv_analysisFilename = None

            if (
                "artifactFilename" in list(self.experiment.keys())
                and self.experiment["artifactFilename"] is not None
            ):
                self.artifactFilename = self.experiment["artifactFilename"]
            else:
                self.artifactFilename = None

        else:
            raise ValueError('Experiment was not specified"')

        if self.artifactFilename is not None and len(self.artifactFilename) > 0:
            self.artifactFilename = Path(self.analyzeddatapath, self.artifactFilename)
            if not self.cell_annotationFilename.is_file():
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
        if self.cell_annotationFilename is not None:
            CP.cprint(
                "yellow",
                f"Reading annotation file: {str(self.cell_annotationFilename):s}",
            )
            ext = self.cell_annotationFilename.suffix
            if ext in [".p", ".pkl", ".pkl3"]:
                self.annotated_dataframe = pd.read_pickle(self.cell_annotationFilename)
            elif ext in [".xls", ".xlsx"]:
                self.annotated_dataframe = pd.read_excel(self.cell_annotationFilename)
            else:
                raise ValueError(
                    f"Do not know how to read annotation file: {str(self.cell_annotationFilename):s}, Valid extensions are for pickle and excel"
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
            "annotation (excel)": self.cell_annotationFilename,
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
        self.df = self.df.apply(_add_day, axis=1)  # add a day (short name)

        if self.day != "all":  # specified day
            day = str(self.day)
            print(
                f"Looking for day: {day:s} in database from {str(self.inputFilename):s}"
            )
            if "_" not in day:
                day = day + "_000"
            cells_in_day = self.df.loc[self.df.day == day]
            # print("cells in day: ", cells_in_day)
            if len(cells_in_day) == 0:
                CP.cprint("r", f"Date not found: {day:s}")
                # for dx in self.df.date.values:
                #     CP.cprint("r", f"    day: {dx:s}")
                return None
            CP.cprint("c", f"  ... Retrieved day:\n    {day:s}")
            for (
                icell
            ) in (
                cells_in_day.index
            ):  # for all the cells in the day (but will check for slice_cell too)
                cellok = self.do_cell(icell, pdf=self.pdfFilename)

        # get the complete protocols:
        # Only returns a dataframe if there is more than one entry
        # Otherwise, it is like a series or dict
        else:
            if self.pdfFilename is None:
                for n, icell in enumerate(range(len(self.df.index))):
                    cell_ok = self.do_cell(icell, pdf=None)
            else:
                with PdfPages(self.pdfFilename) as pdf:
                    for n, icell in enumerate(range(len(self.df.index))):
                        cell_ok = self.do_cell(icell, pdf=pdf)
            CP.cprint(
                "c",
                f"Writing ALL IV analysis results to PKL file: {str(self.iv_analysisFilename):s}",
            )
            with open(self.iv_analysisFilename, "wb") as fh:
                self.df.to_pickle(
                    fh, compression={"method": "gzip", "compresslevel": 5, "mtime": 1}
                )

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

        # pull in map annotations. These are ALWAYS in an excel file, which should be created initially by make_xlsmap.py
        if self.map_annotationFilename is not None:
            CP.cprint(
                "c",
                f"Reading map annotation file:  {str(self.map_annotationFilename):s}",
            )
            self.map_annotations = pd.read_excel(
                Path(self.map_annotationFilename).with_suffix(".xlsx"),
                sheet_name="Sheet1",
            )
            CP.cprint(
                "c",
                f"   ... Loaded map annotation file: {str(self.map_annotationFilename):s}",
            )

    def make_cellstr(self, df: pd.DataFrame, icell: int, shortpath: bool = False):
        """
        Make a day string including slice and cell from the icell index in the pandas dataframe df
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

    def check_celltype(self, celltype: Union[str, None] = None):
        if isinstance(celltype, str):
            celltype = celltype.strip()
        if celltype in [None, "", "?", " ", "  ", "\t"]:
            celltype = "unknown"
        return celltype

    def make_cell_filename(self, celltype: str, slicecell: str):
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

        return Path(file_name)

    def make_pdf_filename(self, dpath:Union[Path, str], celltype: str, slicecell: str):
        pdfname = self.make_cell_filename(celltype, slicecell)
        pdfname = Path(pdfname)
        # check to see if we have a sorted directory with this cell type
        pdfdir = Path(dpath, celltype)
        if not pdfdir.is_dir():
            pdfdir.mkdir()
        return Path(pdfdir, pdfname.stem).with_suffix(".pdf")

    def merge_pdfs(
        self, celltype: str, slicecell: str = None, pdf=None, overwrite: bool = True
    ):
        """
        Merge the PDFs in tempdir with the pdffile (self.pdfFilename)
        The tempdir PDFs are deleted once the merge is complete.
        Merging should be done on a per-cell basis, and on a per-protocol class basis.

        """
        celltype = self.check_celltype(celltype)
        if slicecell is None:
            raise ValueError(
                "iv_analysis:merge_pdfs:: Slicecell is None: should always have a value set"
            )

        if self.dry_run or not self.autoout:
            print("Dry run or not automatic output")
            return
        if not self.merge_flag:
            print("Merge flag is False")
            return
        CP.cprint("c", "********* MERGE PDFS ************\n")
        # if self.pdfFilename is None and not self.autoout:  # no output file, do nothing
        #     return

        # check autooutput and reset pdfFilename if true:
        # CP.cprint("c",
        #     f"Merging pdfs at {datetime.datetime.now().strftime('%m/%d/%Y, %H:%M:%S'):s}"
        # )
        if self.autoout:
            # print("  in auto mode")
            self.cell_pdfFilename = self.make_pdf_filename(self.analyzeddatapath, celltype, slicecell)

        fns = sorted(
            list(self.cell_tempdir.glob("*.pdf"))
        )  # list filenames in the tempdir and sort by name
        if len(fns) == 0:
            # CP.cprint("m", f"No pdfs to merge for {str(self.cell_pdfFilename):s}")
            return  # nothing to do
        CP.cprint("c", f"Merging pdf files: {str(fns):s}")
        CP.cprint("c", f"    into: {str(self.cell_pdfFilename):s}")

        # cell file merged
        mergeFile = PdfMerger()
        if overwrite:  # remove existing file
            self.cell_pdfFilename.unlink(missing_ok=True)
        fns.insert(0, str(self.cell_pdfFilename))
        for i, fn in enumerate(fns):
            if Path(fn).is_file() and fn.stat().st_size > 0:
                try:
                    mergeFile.append(PdfReader(open(fn, "rb")))
                except:
                    Logger.critical(f"Unable to merge PDF: {str(fn):s}")
                    continue
        with open(self.cell_pdfFilename, "wb") as fout:
            mergeFile.write(fout)
        msg = f"Wrote output pdf to : {str(self.cell_pdfFilename):s}"
        CP.cprint("g", msg)
        Logger.info(msg)
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
        print("=" * 80)
        print()

    def gather_protocols(
        self,
        protocols: list,
        prots: dict,
        allprots: dict = None,
        day: str = None,
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
            if (
                (self.protocol is not None)
                and (len(self.protocol) > 1)
                and (self.protocol != x)
            ):
                continue
            # c = os.path.join(day, prots.iloc[i]['slice_slice'], prots.iloc[i]['cell_cell'], x)
            if day is None:
                c = Path(prots["date"], prots["slice_slice"], prots["cell_cell"], x)
            else:
                c = Path(
                    day, prots.iloc[i]["slice_slice"], prots.iloc[i]["cell_cell"], x
                )
            c_str = str(c)  # make sure it is serializable for later on with JSON.
            # maps
            if x.startswith("Map"):
                allprots["maps"].append(c_str)
            if x.startswith("VGAT_"):
                allprots["maps"].append(c_str)
            if x.startswith(
                "Vc_LED"
            ):  # these are treated as maps, even though they are just repeated...
                allprots["maps"].append(c_str)
            # Standard IVs (100 msec, devined as above)
            for piv in self.stdIVs:
                if x.startswith(piv):
                    allprots["stdIVs"].append(c_str)
            # Long IVs (0.5 or 1 second)
            if x.startswith("CCIV_long"):
                allprots["CCIV_long"].append(c_str)
            # VCIVs
            if x.startswith("VCIV"):
                allprots["VCIVs"].append(c_str)
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
        protocolstr: Path,
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
        CP.cprint("c", f"Original cell type: {original_celltype:s}")
        annotated_celltype = None
        map_annotated_celltype = None
        if self.annotated_dataframe is not None:  # get annotation file cell type
            cell_df = self.find_cell(
                self.annotated_dataframe, datestr, slicestr, cellstr, protocolstr=None
            )
            if cell_df.empty:  # cell was not in annotated dataframe
                msg = f"    {datestr:s} {cellstr:s}{slicestr:s} does not have an annotation cell type specification (that is ok)"
                CP.cprint(
                    "c",
                    msg,
                )
                Logger.warning(msg)
            annotated_celltype = cell_df["cell_type"].values[0].strip()

        if self.map_annotations is not None:  # get map annotation cell type
            cell_df = self.find_cell(
                self.map_annotations, datestr, slicestr, cellstr, protocolstr=None
            )
            if cell_df.empty:
                msg = f"    {datestr:s} {cellstr:s}{slicestr:s} does not have an map_annotation celltype specification (field was empty)"
                CP.cprint(
                    "c",
                    msg,
                )
                Logger.warning(msg)
            map_annotated_celltype = cell_df["cell_type"].values[0].strip()

        # now we have several possibilities: only original (prefered), annotated_celltype from
        # the old annotation table, or map_annotated_celltype from the map annotation table.
        # compare these to the original cell type and see if we need to change it.
        #
        if pd.isnull(map_annotated_celltype) and pd.isnull(annotated_celltype):
            return original_celltype, False
        elif pd.isnull(
            annotated_celltype
        ):  # check map annotated type and use instead if different from original
            if not pd.isnull(map_annotated_celltype) and isinstance(
                map_annotated_celltype, str
            ):
                if map_annotated_celltype != original_celltype:
                    msg = f"   Cell type was re-annotated in map_annotation file from: {original_celltype:s} to: {map_annotated_celltype:s})"
                    CP.cprint(
                        "red",
                        msg,
                    )
                    Logger.info(msg)
                    return map_annotated_celltype, True
                else:
                    msg = f"    Map annotation celltype and original cell types were identical, not changed from: {original_celltype:s}"
                    CP.cprint(
                        "c",
                        msg,
                    )
                    Logger.info(msg)
                    return original_celltype, False
            else:
                msg = f"    Map annotation celltype was empty, so using original: {original_celltype:s}"
                CP.cprint(
                    "c",
                    msg,
                )
                Logger.info(msg)
                return original_celltype, False

        # if not pd.isnull(celltype) and not isinstance(celltype, str):
        #     CP.print("c",
        #             f"    Annotated dataFrame: new celltype = {cell_df['cell_type']:s} vs. {original_celltype:s}"
        #         )
        #     CP.cprint("c", f"    Annotation did not change celltype: {celltype:s}")
        #     return original_celltype, False

        elif pd.isnull(
            map_annotated_celltype
        ):  # no map annotation, use the annotated celltype if it is different from original
            if not pd.isnull(annotated_celltype) and isinstance(
                annotated_celltype, str
            ):
                if annotated_celltype != original_celltype:
                    msg = f"   Cell type was re-annotated from: {original_celltype:s} to: {annotated_celltype:s})"
                    CP.cprint(
                        "red", msg)
                    Logger.info(msg)
                    return annotated_celltype, True
                else:
                    msg = f"    Annotation and original cell types were identical, not changed from: {original_celltype:s}"
                    CP.cprint(
                        "c",
                        msg,
                    )
                    Logger.info(msg)
                    return original_celltype, False
            else:
                msg = f"    Annotated celltype was not specified, so using original: {original_celltype:s}"
                CP.cprint(
                    "c",
                    msg,
                )
                Logger.info(msg)
                return original_celltype, False
        else:
            if (
                map_annotated_celltype != original_celltype
                or map_annotated_celltype != annotated_celltype
                or annotated_celltype != original_celltype
            ):
                Logger.critical(
                    f"Cell type mismatch between original, annotated, and map_annotated cell types, Please resolve in the tables"
                )
                raise ValueError(
                    "Cell type mismatch between original, annotated, and map_annotated cell types, Please resolve in the tables"
                )

    def compare_slice_cell(
        self, icell: int, datestr: str, slicestr: str, cellstr: str
    ) -> Tuple[bool, str, str, str]:
        """compare_slice_cell - compare the slice and cell strings in the dataframe
        with the specified self.slicecell value.

        Parameters
        ----------
        icell : int
            index into the dataframe

        Returns
        -------
        bool
            True if the slice and cell match, False otherwise

        """

 
        dsday, nx = Path(datestr).name.split("_")
        self.thisday = dsday
        # check dates
        thisday = datetime.datetime.strptime(dsday, "%Y.%m.%d")
        if thisday < self.after or thisday > self.before:
            CP.cprint(
                "y",
                f"Day {datestr:s} is not in range {self.after_str:s} to {self.before_str:s}",
            )
            return (False, "", "", "")
        # check slice/cell:
        slicecell3 = f"S{int(slicestr[-3:]):02d}C{int(cellstr[-3:]):02d}"  # recognize that slices and cells may be more than 10 (# 9)
        slicecell2 = f"S{int(slicestr[-3:]):01d}C{int(cellstr[-3:]):01d}"  # recognize that slices and cells may be more than 10 (# 9)
        slicecell1 = f"{int(slicestr[-3:]):1d}{int(cellstr[-3:]):1d}"  # only for 0-9
        compareslice = ""
        if self.slicecell is not None:  # limiting to a particular cell?
            match = False
            if len(self.slicecell) == 2:  # 01
                compareslice = f"{int(self.slicecell[0]):1d}{int(self.slicecell[1]):1d}"
                if compareslice == slicecell1:
                    match = True
            elif len(self.slicecell) == 4:  # S0C1
                compareslice = f"S{int(self.slicecell[1]):1d}C{int(self.slicecell[3]):1d}"
                if compareslice == slicecell2:
                    match = True
            elif len(self.slicecell) == 6:  # S00C01
                compareslice = (
                    f"S{int(self.slicecell[1:3]):02d}C{int(self.slicecell[4:]):02d}"
                )
                if compareslice == slicecell3:
                    match = True

            if match:
                return (True, slicecell3, slicecell2, slicecell1)
            else:
                # raise ValueError()
                # Logger.error(f"Failed to find cell: {self.slicecell:s} in {slicestr:s} and {cellstr:s}")
                # Logger.error(f"Options were: {slicecell3:s}, {slicecell2:s}, {slicecell1:s}")
                return (False, "", "", "")  # fail silently... but tell caller.
        else:
            return(True, slicecell3, slicecell2, slicecell1)

    def get_markers(self, fullfile: Path, verbose: bool = True) -> dict:
        # dict of known markers and one calculation of distance from soma to surface
        dist = np.nan
        marker_dict:dict = {
            "soma": [],
            "surface": [],
            "medialborder": [],
            "lateralborder": [],
            "rostralborder": [],
            "caudalborder": [],
            "ventralborder": [],
            "dorsalborder": [],
            "AN": [],
            "dist": dist,
        }

        mosaic_file = list(fullfile.glob("*.mosaic"))
        if len(mosaic_file) > 0:
            if verbose:
                CP.cprint("c", f"    Have mosaic_file: {mosaic_file[0].name:s}")
            state = json.load(open(mosaic_file[0], "r"))
            cellmark = None
            for item in state["items"]:
                if item["type"] == "MarkersCanvasItem":
                    markers = item["markers"]
                    for markitem in markers:
                        if verbose:
                            CP.cprint(
                                "c",
                                f"    {markitem[0]:>12s} x={markitem[1][0]*1e3:7.2f} y={markitem[1][1]*1e3:7.2f} z={markitem[1][2]*1e3:7.2f} mm ",
                            )

                        for j in range(len(markers)):
                            markname = markers[j][0]
                            if markname in marker_dict:
                                marker_dict[markname] = [
                                    markers[j][1][0],
                                    markers[j][1][1],
                                ]
                if item["type"] == "CellCanvasItem":  # get Cell marker position also
                    cellmark = item['userTransform']
            soma_xy:list = []
            if "soma" in marker_dict.keys() and cellmark is None:
                somapos = marker_dict["soma"]
            if cellmark is not None:  # override soma position with cell marker position
                somapos = cellmark['pos']
                marker_dict['soma'] = somapos

            surface_xy:list = []
            if len(somapos) == 2 and len(marker_dict["surface"]) == 2:
                soma_xy = somapos
                surface_xy = marker_dict["surface"]
                dist = np.sqrt(
                    (soma_xy[0] - surface_xy[0]) ** 2
                    + (soma_xy[1] - surface_xy[1]) ** 2
                )
                if verbose:
                    CP.cprint("c", f"    soma-surface distance: {dist*1e6:7.1f} um")
            else:
                if verbose:
                    CP.cprint(
                        "r", "    Not enough markers to calculate soma-surface distance"
                    )
            if soma_xy == [] or surface_xy == []:
                if verbose:
                    CP.cprint("r", "    No soma or surface markers found")
        else:
            if verbose:
                CP.cprint("r", "No mosaic file found")
        return marker_dict

    def make_pickle_filename(self, dpath: Union[str, Path], celltype: str, slicecell: str):
        pklname = self.make_cell_filename(celltype, slicecell)
        pklname = Path(pklname)
        # check to see if we have a sorted directory with this cell type
        pkldir = Path(dpath, celltype)
        if not pkldir.is_dir():
            pkldir.mkdir()
        return Path(pkldir, pklname.stem).with_suffix(".pkl")
    
    def do_cell(self, icell: int, pdf=None) -> bool:
        """
        Do analysis on one cell
        Runs all protocols for the cell

        Includes a dispatcher for different kinds of protocols: IVs, Maps, VCs

        Parameters
        ----------
        icell : int
            index into pandas database for the day to be analyzed

        pdf : bool, default=False

        Returns
        -------
        success: bool

        """
        datestr, slicestr, cellstr = self.make_cell(icell)
        matchcell, slicecell3, slicecell2, slicecell1 = self.compare_slice_cell(
            icell, datestr=datestr, slicestr=slicestr, cellstr=cellstr
        )
        if not matchcell:
            return False
        # reassign cell type if the annotation table changes it.
        celltype, celltypechanged = self.get_celltype(icell)
        celltype = self.check_celltype(celltype)

        fullfile = Path(self.rawdatapath, self.df.iloc[icell].cell_id)

        self.get_markers(fullfile, verbose=True)

        if self.skip_subdirectories is not None:
            # skip matching subdirectories
            for skip in self.skip_subdirectories:
                # print(f"Checking skip = {skip:s} with {str(fullfile):s}")
                if str(fullfile).find(skip) >= 0:
                    ffparts = fullfile.parts
                    fftail = str(Path(*ffparts[len(self.rawdatapath.parts) - 1 :]))
                    CP.cprint(
                        "r",
                        f"SKIPPING data/date: {fftail:s}  containing: {skip:s}",
                    )
                    return  False# skip this data set

        if not fullfile.is_dir():
            fullfile = Path(
                self.df.iloc[icell]["data_directory"],
                self.make_cellstr(self.df, icell, shortpath=True),
            )

        if self.extra_subdirectories is not None and not fullfile.is_dir():
            # try extra sub directories
            pathparts = fullfile.parts
            day = None
            for i, p in enumerate(pathparts):
                if p.startswith("20"):
                    day = Path(*pathparts[i:])
                    break
            if day is None:
                CP.cprint(
                    "r", f"do_cell: Day <None> found in fileparts: {str(pathparts):s}"
                )
                exit()
            for subdir in self.extra_subdirectories:
                fullfile = Path(self.rawdatapath, subdir, day)
                if fullfile.is_dir():
                    break

            if not fullfile.is_dir():
                CP.cprint("r", f"Unable to get the file: {str(fullfile):s}")
                return False

        prots = self.df.iloc[icell]["data_complete"]
        allprots = self.gather_protocols(prots.split(", "), self.df.iloc[icell])

        if self.dry_run:
            msg = f"\nIV_Analysis:do_cell:: Would process day: {datestr:s} slice: {slicestr:s} cell: {cellstr:s}"
            msg += f"\n        with fullpath {str(fullfile):s}"
            print(msg)
            Logger.info(msg)

        if not fullfile.is_dir():
            msg = "   But that cell was not found.\n"
            msg += f"{str(self.df.iloc[icell]):s}\n\n"
            CP.cprint("r", msg)
            print("*" * 40)
            Logger.warning(msg)
            self.merge_pdfs(celltype, pdf=pdf)
            return False

        elif fullfile.is_dir() and len(allprots) == 0:
            msg = "Cell found, but no protocols were found"
            CP.cprint("m", "Cell found, but no protocols were found")
            Logger.warning(msg)
            return False
        elif fullfile.is_dir() and len(allprots) > 0:
            for prottype in allprots.keys():
                for prot in allprots[prottype]:
                    ffile = Path(self.df.iloc[icell].data_directory, prot)
                    if not ffile.is_dir():
                        msg = f"file/protocol day={icell:d} not found: {str(ffile):s}\n"
                        msg += f"    {str(prottype):s}  {str(prot):s}\n"
                        CP.cprint("r",msg)
                        Logger.error(msg)
                        exit()
        else:
            msg = f"   Cell OK, with {len(allprots['stdIVs'])+len(allprots['CCIV_long']):4d} IV protocols"
            msg += f" and {len(allprots['maps']):4d} map protocols"
            msg += f"  Electrode: {self.df.iloc[icell]['internal']:s}"
            CP.cprint("g", msg)
            Logger.info(msg)
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
            if (
                self.iv_analysisFilename is not None
                and Path(self.iv_analysisFilename).suffix == ".h5"
            ):
                self.df["IV"] = None
                self.df["Spikes"] = None
            if self.cell_tempdir is not None:
                self.make_tempdir()  # clean up temporary directory
            self.analyze_ivs(icell=icell, allprots=allprots, celltype=celltype, pdf=pdf)

            self.merge_pdfs(celltype, slicecell=slicecell2, pdf=pdf)

            # store pandas db analysis of this cell in a pickled file:
            self.cell_pklFilename = self.make_pickle_filename(self.analyzeddatapath, celltype, slicecell2)


            msg = f"Writing cell IV analysis results to PKL file: {str(self.cell_pklFilename):s}"
            CP.cprint("c", msg)
            Logger.info(msg)
            if self.df.iloc[icell]["Spikes"] is None:
                msg = f"   @@@ Spikes array is empty @@@"
                CP.cprint("r", msg)
                Logger.warning(msg)
            else:
                msg = f"*** Have Spikes keys: {str(self.df.iloc[icell]['Spikes'].keys()):s}"
                CP.cprint("g", msg)
                Logger.info(msg)
            with open(self.cell_pklFilename, "wb") as fh:
                self.df.iloc[icell].to_pickle(
                    fh, compression={"method": "gzip", "compresslevel": 5, "mtime": 1}
                )
            # if they exist, remove the slicecell1 and slicecell3 files to avoid confusion
            pk1 = self.make_pickle_filename(self.analyzeddatapath, celltype, slicecell1)
            pk1.unlink(missing_ok=True)
            pk3 = self.make_pickle_filename(self.analyzeddatapath, celltype, slicecell3)
            pk3.unlink(missing_ok=True)

            self.df["IV"] = None
            self.df["Spikes"] = None
            gc.collect()

        elif self.vc_flag:
            self.analyze_vcs(icell, allprots)
            gc.collect()

        elif self.map_flag:
            if self.cell_tempdir is not None:
                self.make_tempdir()
            self.analyze_maps(
                icell=icell, allprots=allprots, celltype=celltype, pdf=pdf
            )
            # analyze_maps stores events in an events subdirectory by cell
            # It also merges the PDFs for that cell in the celltype directory
            msg = f"Merging pdfs, with: {str(pdf):s}"
            CP.cprint("r", msg)
            Logger.info(msg)
            self.merge_pdfs(celltype, slicecell=slicecell2, pdf=pdf)
            # also remove slicecell3 and slicecell1 filenames if they exist
            pdf1 = self.make_pdf_filename(self.analyzeddatapath, celltype, slicecell1)
            pdf1.unlink(missing_ok=True)
            pdf3 = self.make_pdf_filename(self.analyzeddatapath, celltype, slicecell3)
            pdf3.unlink(missing_ok=True)

            gc.collect()
        return True

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
        nfiles = 0
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
