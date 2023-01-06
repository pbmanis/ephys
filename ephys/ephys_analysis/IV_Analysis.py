"""
Module for analyzing datasets with optogenetic or uncaging laser mapping
current-voltage relationships, and so on... 
Used as a wrapper for multiple experiments. 
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
from PyPDF2 import PdfFileMerger, PdfFileReader, PdfFileWriter

import ephys.ephys_analysis as EP
import ephys.ephys_analysis.IV_Analysis_Params as AnalysisParams
import ephys.mapanalysistools as mapanalysistools

PMD = mapanalysistools.plotMapData.PlotMapData()

np.seterr(divide="raise", invalid="raise")


@dataclass
class cmdargs:
    experiment: Union[str, list, None] = None
    rawdatapath: Union[str, Path, None] = None
    analyzeddatapath: Union[str, Path, None] = None
    databasepath: Union[str, Path, None] = None
    inputFilename: Union[str, Path, None] = None
    pdfFilename: Union[str, Path, None] = None
    annotationFilename: Union[str, Path, None] = None
    artifactFilename: Union[str, Path, None] = None
    map_annotationFilename: Union[str, Path, None] = None
    map_pdfs: bool=False
    extra_subdirectories: object = None

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
    rasterize: bool = True
    update: bool = False
    noparallel: bool = True
    replot: bool = False

    # analysis parameters
    ivduration: float = 0.0
    threshold: float = 2.5  # cb event detection threshold (x baseline std)
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
    notchfreqs: str = "60, 120, 180, 240"
    LPF: float = 0.0
    HPF: float = 0.0
    notchQ: float = 90.0
    detector: str = "cb"
    
class IV_Analysis():
    def __init__(self, args: object):

        # args = vars(in_args)  # convert to dict

        self._testing_counter = 0  # useful to run small tests
        self._testing_count = 400  # should be 0 in production

        self.rawdatapath = None
        self.analyzeddatapath = None

        self.inputFilename = None
        self.analyzeddatapath = None
        self.annotationFilename = None
        self.map_annotationFilename = None
        self.extra_subdirectories = None
        self.artifactFilename = None
        self.pdfFilename = None
        self.exclusions = None

        self.experiment = args.experiment

        # selection of data for analysis
        self.day = args.day
        print(type(args.before))
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
        self.rasterize = True
        self.update = args.update
        self.noparallel = args.noparallel
        self.replot = args.replot

        # analysis parameters
        self.ivduration = args.ivduration
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
        if self.detector == "cb":
            self.AM.set_methodname("cb_cython")
        elif self.detector == "aj":
            self.AM.set_methodname("aj_cython")
        elif self.detector == "zc":
            raise ValueError("Detector ZC is not recommended at this time")
        else:
            raise ValueError(f"Detector {self.detector:s} is not one of [cb, aj, zc")

        super().__init__()
        # self.setup()

    def set_exclusions(self, exclusions: Union[dict, None] = None):
        self.exclusions = exclusions

    def set_experiment(self, expt: dict):
        self.experiment = expt

    def setup(self):
        if self.experiment not in ["None", None]:
            self.rawdatapath = Path(self.experiment["rawdatapath"])
            self.analyzeddatapath = Path(self.experiment["analyzeddatapath"])
            self.databasepath = Path(self.experiment["databasepath"])
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
                    raise FileNotFoundError
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
                    raise FileNotFoundError
            else:
                self.map_annotationFilename = None

            if (
                "extra_subdirectories" in list(self.experiment.keys())
                and self.experiment["extra_subdirectories"] is not None
            ):
                self.extra_subdirectories = self.experiment["extra_subdirectories"]
            else:
                self.extra_subdirectories = None
            if ("artifactFilename" in list(self.experiment.keys())
                and self.experiment["artifactFilename"] is not None):
                self.artifactFilename = self.experiment["artifactFilename"]
            else:
                self.artifactFilename = None
        
        else:
            raise ValueError('Experiment was not specified"')

        if self.artifactFilename is not None and len(self.artifactFilename) > 0:
            self.artifactFilename = Path(self.analyzeddatapath, self.artifactFilename)
            if not self.annotationFilename.is_file():
                raise FileNotFoundError
        else:
            self.artifactFilename = None

        # get the input file (from dataSummary)
        self.df = pd.read_pickle(str(self.inputFilename))
        CP.cprint("g", f"Read in put file: {str(self.inputFilename):s}")
        self.df[
            "day"
        ] = None  # self.df = self.df.assign(day=None)  # make sure we have short day available
        self.df = self.df.apply(self._add_date, axis=1)
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

        allpaths = {
            "experiment": self.experiment,
            "raw": self.rawdatapath,
            "analyzed": self.analyzeddatapath,
            "input": self.inputFilename,
            "pdf": self.pdfFilename,
            "annotation": self.annotationFilename,
            "map_annotation": self.map_annotationFilename,
            "extra_subdirectories": self.extra_subdirectories,
            "artifact": self.artifactFilename,
            "exclusions": self.exclusions,
        }
        print(f"\nPaths and files:")
        for p in allpaths:
            print(f"   {p:>20s}   {str(allpaths[p]):<s}")

    def _add_date(self, row, axis=1):
        row.day = str(Path(row.date).name)
        return row

    def run(self):
        # select a date:
        self.n_analyzed = 0
        if self.day != "all":  # specified day
            day = str(self.day)
            print(
                f"Looking for day: {day:s} in database from {str(self.inputFilename):s}"
            )
            if "_" not in day:
                day = day + "_000"  # lambda x: (x['temp_f'] +  459.67) * 5 / 9
            # print(self.df.columns)
            day_x = self.df.loc[self.df["day"] == day]
            if len(day_x) == 0:
                print("day not found")
                for dx in self.df.day.values:
                    print(f"    day: {dx:s}")
            print("  ... Retrieved day: ", day_x)
            for iday in day_x.index:
                self.do_day(iday, 0, pdf=self.pdfFilename)

        # get the complete protocols:
        # Only returns a dataframe if there is more than one entry
        # Otherwise, it is like a series or dict
        else:
            if self.pdfFilename is None:
                for n, iday in enumerate(range(len(self.df.index))):
                    self.do_day(iday, n)
            else:
                with PdfPages(self.pdfFilename) as pdf:
                    for n, iday in enumerate(range(len(self.df.index))):
                        self.do_day(iday, n, pdf=pdf)

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
            print("Reading map annotation file: ", self.map_annotationFilename)
            self.map_annotations = pd.read_excel(
                Path(self.map_annotationFilename).with_suffix(".xlsx"),
                sheet_name="Sheet1",
            )

    def make_cellstr(self, df: object, iday: int, shortpath: bool = False):
        """
        Make a day string including slice and cell from the iday index in the pandas datafram df
        Example result:
            s = self.make_cellstr (df, 1)
            s: '2017.01.01_000/slice_000/cell_001'  # Mac/linux path string

        Parameters
        ----------
        df : Pandas dataframe instance

        iday : int (no default)
            index into pandas dataframe instance

        returns
        -------
        Path
        """

        if shortpath:
            day = Path(df.iloc[iday]["date"]).parts[-1]
            daystr = Path(
                day,
                Path(df.iloc[iday]["slice_slice"]).name,
                Path(df.iloc[iday]["cell_cell"]).name,
            )
        else:
            daystr = Path(
                df.iloc[iday]["date"],
                Path(df.iloc[iday]["slice_slice"]).name,
                Path(df.iloc[iday]["cell_cell"]).name,
            )
        # print("make_cellstr: ", daystr)
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
            Path(fn).unlink(missing_ok=True)

    def merge_pdfs(self, celltype: Union[str, None] = None, slicecell:Union[str, None]=None):
        """
        Merge the PDFs in tempdir with the pdffile (self.pdfFilename)
        The tempdir PDFs are deleted once the merge is complete.
        """
        if self.pdfFilename is None and not self.autoout:  # no output file, do nothing
            return
        if self.dry_run:
            return
        # check autooutput and reset pdfFilename if true:
        # CP.cprint("c",
        #     f"Merging pdfs at {datetime.datetime.now().strftime('%m/%d/%Y, %H:%M:%S'):s}"
        # )
        if self.merge_flag:
            self.tempdir = Path("./temppdfs")  # use one folder and do not clear it
        if self.autoout:
            self.tempdir = Path("./temppdfs")  # use one folder and do not clear it
            # print("  in auto mode")
            pdfname = str(self.analyzeddatapath) + "_" + self.thisday.replace(".", "_")
            if slicecell is not None:
                pdfname += f"_{slicecell:s}"
            else:
                pdfname += "_"
            if self.signflip:
                pdfname += "_signflip"
            if self.alternate_fit1:
                pdfname += "_alt1"
            if self.alternate_fit2:
                pdfname += "_alt2"

            if isinstance(celltype, str):
                celltype = celltype.strip()
            if celltype in [None, "", "?"]:
                celltype = "unknown"
            pdfname += f"_{celltype:s}"
            if self.iv_flag:
                pdfname += f"_IVs"
            elif self.replot:
                pdfname += f"_maps"  # tag if using other than zscore in the map plot

            pdfname = Path(pdfname)
            # check to see if we have a sorted directory with this cell type
            pdfdir = Path(self.analyzeddatapath, celltype)
            if not pdfdir.is_dir():
                pdfdir.mkdir()
            self.cell_pdfFilename = Path(pdfdir, pdfname.stem).with_suffix(".pdf")
        fns = sorted(
            list(self.tempdir.glob("*.pdf"))
        )  # list filenames in the tempdir and sort by name
        if len(fns) == 0:
            return  # nothing to do
        CP.cprint("c", f"Merging pdf files: {str(fns):s}")
        CP.cprint("c", f"    into: {str(self.cell_pdfFilename):s}")

        # cell file merged
        mergeFile = PdfFileMerger()
        fns.insert(0, str(self.cell_pdfFilename))
        for i, fn in enumerate(fns):
            if Path(fn).is_file():
                mergeFile.append(PdfFileReader(open(fn, "rb")))
        with open(self.cell_pdfFilename, "wb") as fout:
            mergeFile.write(fout)
        print(f"Wrote map pdf to : {str(self.cell_pdfFilename):s}")
        fns.pop(0)
        # remove temporary files
        for fn in fns:
            fn.unlink(missing_ok=True)

        # main file merge
        if self.pdfFilename is not None:
            mergeFile = PdfFileMerger()
            fns = [str(self.pdfFilename), str(self.cell_pdfFilename)]
            for i, fn in enumerate(fns):
                fn = Path(fn)
                if fn.is_file() and fn.stat().st_size > 0:
                    mergeFile.append(PdfFileReader(open(fn, "rb")))
            with open(self.pdfFilename, "wb") as fout:
                mergeFile.write(fout)

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
        if protocolstr is None:
            cf = df.query(
                f'date == "{dstr:s}" & slice_slice == "{slicestr:s}" & cell_cell == "{cellstr:s}"'
            )
        else:
            dprot = str(protocolstr.name)
            cf = df.query(
                f'date == "{dstr:s}" & slice_slice == "{slicestr:s}" & cell_cell == "{cellstr:s}" & map == "{dprot:s}"'
            )
        # self._testing_counter += 1
        # if self._testing_counter > self._testing_count:
        #     exit()
        return cf

    def get_celltype(self, iday):
        """Find the type of the cell associated with this entry

        Parameters
        ----------
        iday : int
            The index into the table

        Returns
        -------
        str, bool
            The celltype
            If the celltype changed in annotation, the bool value will be True,
            otherwise False
        """
        original_celltype = self.df.at[iday, "cell_type"]
        datestr, slicestr, cellstr = self.make_cell(iday)
        # fullfile = Path(self.rawdatapath, self.make_cellstr(self.df, iday))
        #        prots = [x for x in fullfile.glob('*') if x.is_dir()]
        # protocols = self.df.iloc[iday]
        # allprots = self.gather_protocols(protocols["data_complete"].split(", "), protocols)
        # print(f"   map Annotation filename:  {str(self.map_annotationFilename):s}")

        # if self.map_annotationFilename is not None and len(allprots["maps"]) > 0:  # get from annotation file
        #     cell_df = self.find_cell(
        #         self.map_annotations, datestr, slicestr, cellstr, allprots["maps"][0]
        #     )
        #     if not cell_df.empty:
        #         celltype = cell_df["cell_type"].values[0]
        #         if not isinstance(celltype, str):
        #             CP.cprint("cyan", f"   Map annotated celltype: {celltype:s} (original: {original_celltype:s}")
        #             celltype = original_celltype

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

    def do_day(self, iday: int, nout: int = 0, pdf=None):
        """
        Do analysis on a day's data
        Runs all cells in the day, unless slicecell has been specified - then
        permits subsetting to do just one specific cell (or slice) on the day

        Parameters
        ----------
        iday : int
            index into pandas database for the day to be analyzed


        """
        datestr, slicestr, cellstr = self.make_cell(iday)
        if len(self.slicecell) >= 2:
            slicen = "slice_%03d" % int(self.slicecell[1])
            if slicestr != slicen:
                return
            if len(self.slicecell) == 4:
                celln = "cell_%03d" % int(self.slicecell[3])
                if cellstr != celln:
                    return

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
            self.rawdatapath, self.make_cellstr(self.df, iday, shortpath=True)
        )
        if not fullfile.is_dir() and self.extra_subdirectories is not None:
            # try extra sub directories
            pathparts = fullfile.parts
            day = None
            for i, p in enumerate(pathparts):
                if p.startswith("20"):
                    day = Path(*pathparts[i:])
                    break
            if day is None:
                CP.cprint("r", f"do_day: Day found in fileparts: {str(pathparts):s}")
                exit()

            for subdir in self.extra_subdirectories:
                fullfile = Path(self.rawdatapath, subdir, day)
                if fullfile.is_dir():
                    break
        if not fullfile.is_dir():
            CP.cprint("r", f"Unable to get the file: {str(fullfile):s}")
            exit()
        celltype, celltypechanged = self.get_celltype(iday)
        prots = self.df.iloc[iday]["data_complete"]
        allprots = self.gather_protocols(prots.split(", "), self.df.iloc[iday])

        if self.dry_run:
            print(
                f"\nWould process day: {datestr:s} slice: {slicestr:s} cell: {cellstr:s}"
            )
            print(f"        fullpath {str(fullfile):s}")

        if not fullfile.is_dir():
            CP.cprint("r", "   But that cell was not found.")
            print("*" * 40)
            print(self.df.iloc[iday])
            print("*" * 40)
            print()
            return

        elif fullfile.is_dir() and len(allprots) == 0:
            CP.cprint("m", "   Cell found, but no protocols were found")
            return

        else:
            msg = f"   Cell OK, with {len(allprots['stdIVs'])+len(allprots['CCIV_long']):4d} IV protocols"
            msg += f" and {len(allprots['maps']):4d} map protocols"
            msg += f"  Electrode: {self.df.iloc[iday]['internal']:s}"
            CP.cprint("g", msg)
            if self.map_flag:
                for i, p in enumerate(sorted(prots)):
                    if p in allprots["maps"]:
                        print("      {0:d}. {1:s}".format(i + 1, str(p.name)))
        if self.dry_run:
            return

        if self.verbose:
            for k in list(allprots.keys()):
                print("protocol type: {:s}:".format(k))
                for m in allprots[k]:
                    print("    {0:s}".format(str(m)))
                if len(allprots[k]) == 0:
                    print("    No protocols of this type")
            print("All protocols: ")
            print([allprots[p] for p in allprots.keys()])

        if self.merge_flag and pdf is not None:
            self.merge_pdfs(celltype)
            return
        # CP.cprint("r", f"iv flag: {str(self.iv_flag):s}")
        if self.iv_flag:
            self.df["IV"] = None
            self.df["Spikes"] = None
            if pdf is not None:
                self.make_tempdir()  # clean up temporary directory
            self.analyze_ivs(
                iday=iday, file=fullfile, allprots=allprots, celltype=celltype, pdf=pdf
            )
            if pdf is not None:
                self.merge_pdfs(
                    celltype
                )  # do not do this until all the parallel processing for the day is done.
            gc.collect()

        if self.vc_flag:
            self.analyze_vcs(iday, allprots)

        if self.map_flag:
            if pdf is not None:
                self.make_tempdir()
            self.analyze_maps(iday=iday, file=fullfile, allprots=allprots, celltype=celltype, pdf=pdf)
            if pdf is not None:
                self.merge_pdfs(celltype)

    

    def analyze_ivs(
        self, iday, file: Union[Path, str], allprots: dict, celltype: str, pdf=None
    ):
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
        CP.cprint(
            "c",
            f"analyze ivs for index: {iday: d} ({str(self.df.at[iday, 'date']):s} )",
        )

        if "IV" not in self.df.columns.values:
            self.df = self.df.assign(IV=None)
        if "Spikes" not in self.df.columns.values:
            self.df = self.df.assign(Spikes=None)

        CP.cprint(
            "c",
            f"      {str(file):s}\n           at: {datetime.datetime.now().strftime('%m/%d/%Y, %H:%M:%S'):s}",
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
        nworkers = 16  # number of cores/threads to use
        tasks = range(len(allivs))  # number of tasks that will be needed
        results = dict(
            [("IV", {}), ("Spikes", {})]
        )  # storage for results; predefine the dicts.
        if self.noparallel:  # just serial...
            for i, x in enumerate(tasks):
                r, nfiles = self.analyze_iv(iday=iday, i=i, x=x, file=file, allivs=allivs, nfile=nfiles, pdf=pdf)
                if self.dry_run:
                    continue
                if r is None:
                    continue
                results["IV"][allivs[i]] = r["IV"]
                results["Spikes"][allivs[i]] = r["Spikes"]
            results["IV"] = _cleanup_ivdata(results["IV"])
            results["Spikes"] = _cleanup_ivdata(results["Spikes"])
            if not self.dry_run:
                self.df.at[iday, "IV"] = results[
                    "IV"
                ]  # everything in the RM analysis_summary structure
                self.df.at[iday, "Spikes"] = results[
                    "Spikes"
                ]  # everything in the SP analysus_summary structure
        #            print(self.df.at[iday, 'IV'])
        else:
            result = [None] * len(tasks)  # likewise
            with mp.Parallelize(
                enumerate(tasks), results=results, workers=nworkers
            ) as tasker:
                for i, x in tasker:
                    result, nfiles = self.analyze_iv(
                        iday, i, x, file, allivs, nfiles, pdf=pdf
                    )
                    tasker.results[allivs[i]] = result
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
                iday, "IV"
            ] = riv  # everything in the RM analysis_summary structure
            self.df.at[
                iday, "Spikes"
            ] = rsp  # everything in the SP analysus_summary structure

        # foname = '%s~%s~%s.pkl'%(datestr, slicestr, cellstr)
        self.df["annotated"] = self.df["annotated"].astype(int)
        self.df["expUnit"] = self.df["expUnit"].astype(int)

        if len(allivs) > 0:
            # with pickle:
            # with open(Path(self.analyzeddatapath, 'IV_Analysis.pkl'), 'wb') as fh:
            #    self.df.to_pickle(fh, compression={'method': 'gzip', 'compresslevel': 1, 'mtime': 1})
            #
            # with hdf5:
            day, slice, cell = self.make_cell(iday=iday)
            keystring = str(
                Path(Path(day).name, slice, cell)
            )  # the keystring is the cell.
            if self.n_analyzed == 0:
                self.df.iloc[iday].to_hdf("IV_Analysis.h5", key=keystring, mode="w")
            else:
                self.df.iloc[iday].to_hdf("IV_Analysis.h5", key=keystring, mode="a")
            self.df.iloc[iday]["IV"] = None
            self.df.iloc[iday]["Spikes"] = None
            self.n_analyzed += 1
            gc.collect()

        # with open(Path(analyzeddatapath, 'events', foname), 'wb') as fh:
        #      dill.dump(results, fh)

    def analyze_iv(
        self,
        iday: int,
        i: int,
        x: int,
        file: Union[Path, str],
        allivs: list,
        nfiles: int,
        pdf: None,
    ):
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

        fpath = Path(file, protocol)
        if not fpath.is_dir():
            CP.cprint("r", f"File not found (analyze_iv)!! {str(fpath):s}")
            exit()

        if self.iv_select["duration"] > 0.0:
            EPIV = EP.IVSummary.IV(
                str(fpath),
                plot=True,
            )
            check = EPIV.iv_check(duration=self.iv_select["duration"])
            if check is False:
                gc.collect()
                return (None, 0)  # skip analysis
        if not self.dry_run:
            print(f"      IV analysis for {str(fpath):s}")
            EPIV = EP.IVSummary.IVSummary(fpath, plot=True)
            br_offset = 0.0
            if (
                not pd.isnull(self.df.at[iday, "IV"])
                and protocol in self.df.at[iday, "IV"]
                and "--Adjust" in self.df.at[iday, protocol]["BridgeAdjust"]
            ):
                print(
                    "Bridge: {0:.2f} Mohm".format(
                        self.df.at[iday, "IV"][protocol]["BridgeAdjust"] / 1e6
                    )
                )
            ctype = self.df.at[iday, "cell_type"]
            tgap = 0.0015
            tinit = True
            if ctype in [
                "bushy",
                "Bushy",
                "d-stellate",
                "D-stellate",
                "Dstellate",
                "octopus",
                "Octopus",
            ]:
                tgap = 0.0005
                tinit = False
            EPIV.plot_mode(mode=self.IV_pubmode)
            plot_handle = EPIV.compute_iv(
                threshold=self.spike_threshold,
                bridge_offset=br_offset,
                tgap=tgap,
                plotiv=False,
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
            # print("Checking for figure, plothandle is: ", plot_handle)
            if plot_handle is not None:
                shortpath = fpath.parts
                shortpath = str(Path(*shortpath[4:]))
                plot_handle.suptitle(
                    "{0:s}\nType: {1:s} {2:s}".format(
                        shortpath,  # .replace("_", "\_"),
                        self.df.at[iday, "cell_type"],
                        ctwhen,
                    ),
                    fontsize=8,
                )
                t_path = Path(self.tempdir, "temppdf_{0:04d}.pdf".format(nfiles))
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
            del EPIV
            del iv_result
            del sp_result
            gc.collect()
            return result, nfiles

        else:
            print("Dry Run: would analyze %s" % Path(self.rawdatapath, protocol))
            br_offset = 0

            if self.df.at[iday, "IV"] == {} or pd.isnull(self.df.at[iday, "IV"]):
                print("   current database has no IV data set for this file")
            elif protocol not in list(self.df.at[iday, "IV"].keys()):
                print(
                    "Protocol {0:s} not found in day: {1:s}".format(
                        str(protocol), self.df.at[iday, "date"]
                    )
                )
            elif "BridgeAdjust" in self.df.at[iday, "IV"][protocol].keys():
                br_offset = self.df.at[iday, "IV"][protocol]["BridgeAdjust"]
                print("   with Bridge: {0:.2f}".format(br_offset / 1e6))
            else:
                print("... has no bridge, will use 0")
            gc.collect()
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
                    print("Analyzing %s" % Path(self.rawdatapath, f))
                    EPVC = EP.VCSummary.VCSummary(
                        Path(self.rawdatapath, f),
                        plot=False,
                    )
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

    if args.rawdatapath is None:
        X, args.rawdatapath, code_dir = set_expt_paths.get_paths()
        if args.rawdatapath is None:
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
    set_start_method("spawn")

    main()
