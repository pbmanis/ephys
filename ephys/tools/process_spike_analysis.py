"""
Process spike analysis. 
After running analyze_ivs.py, this script will read the datasummary file and access the individual
spike/IV analyses, or run them anew.
Generates an excel sheet with the result of analysis, including mean spike parameters,
the FI_Curve, RMP, Rin, taum, etc. 
This data is computed from raw data sets, rather than from the .pkl intermediate files. 

The output spreadsheet can be passed to plot_spike_info to generate plots for these parameters


**** DEPRECATED ****
Use the tools in datatables instead.

"""

import logging
from pathlib import Path
from typing import Union
import pprint
import psutil
import gc

import ephys.ephys_analysis as EP
import ephys.datareaders as DR
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from pandas_parallel_apply import DataFrameParallel, SeriesParallel
from pylibrary.tools import cprint
import pyqtgraph as pg

from ephys.tools import decorate_excel_sheets as DE
from ephys.tools.get_computer import get_computer


from ephys.ephys_analysis import (
    analysis_common,
    iv_analysis,
    map_analysis,
    summarize_ivs,
)


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
Logger = logging.getLogger("SpikeAnalysis")
level = logging.DEBUG
Logger.setLevel(level)
# create file handler which logs even debug messages
logging_fh = logging.FileHandler(filename="logs/spike_analysis.log")
logging_fh.setLevel(level)
logging_sh = logging.StreamHandler()
logging_sh.setLevel(level)
log_formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s  (%(filename)s:%(lineno)d) - %(message)s "
)
logging_fh.setFormatter(log_formatter)
logging_sh.setFormatter(CustomFormatter())  # log_formatter)
Logger.addHandler(logging_fh)
Logger.addHandler(logging_sh)
# Logger.info("Starting Process Spike Analysis")

pd.set_option("display.max_columns", 40)

CP = cprint.cprint
nr = 0


cumulative_memory = 0

#######################
# Set some criteria
# RMP SD across protocol
rmp_sd_limit = 3.0  # mV
#
max_rows = -1


class ProcessSpikeAnalysis:
    def __init__(self, dataset=None, experiment=None):
        self.set_experiment(dataset, experiment)

    def set_experiment(self, dataset=None, experiment=None):
        self.dataset = dataset
        self.experiment = experiment
        if self.experiment is not None:
            computer_name = get_computer()
            nworkers = self.experiment["NWORKERS"][computer_name]
            self.set_workers(nworkers, computer=computer_name)

    def set_workers(self, nworkers: int = 10, computer: str = "Unknown"):
        self.nworkers = nworkers
        CP("c", f"Using {nworkers:d} workers for parallel processing on {computer:s}")
        Logger.info(f"Using {nworkers:d} workers for parallel processing on {computer:s}")

    def get_rec_date(self, filename: Union[Path, str]):
        """get the recording date of record from the filename as listed n the excel sheet

        Args:
            filename (Union[Path, str]): _description_
        """
        fn = Path(filename)
        datename = str(fn.name)
        datename = datename[:-4]
        return datename

    def datenum(self, datestr: str):
        s = datestr.split(".")
        d = int(1e4 * int(s[0])) + int(1e2 * int(s[1])) + int(s[2])
        return d

    def find_lowest_current_spike(self, row, SP):
        dvdts = []
        for tr in SP.spikeShapes:  # for each trace
            if len(SP.spikeShapes[tr]) > 1:  # if there is a spike
                spk = SP.spikeShapes[tr][0]  # get the first spike in the trace
                dvdts.append(spk)  # accumulate first spike info

        if len(dvdts) > 0:
            currents = []
            for d in dvdts:  # for each first spike, make a list of the currents
                currents.append(d.current)
            min_current = np.argmin(currents)  # find spike elicited by the minimum current
            row.dvdt_rising = dvdts[min_current].dvdt_rising
            row.dvdt_falling = dvdts[min_current].dvdt_falling
            row.dvdt_current = currents[min_current] * 1e12  # put in pA
            row.AP_thr_V = 1e3 * dvdts[min_current].AP_begin_V
            if dvdts[min_current].halfwidth_interpolated is not None:
                row.AP_HW = dvdts[min_current].halfwidth_interpolated * 1e3
            row.AP_begin_V = 1e3 * dvdts[min_current].AP_begin_V
            CP(
                "y",
                f"I={currents[min_current]*1e12:6.1f} pA, dvdtRise={row.dvdt_rising:6.1f}, dvdtFall={row.dvdt_falling:6.1f}, APthr={row.AP_thr_V:6.1f} mV, HW={row.AP_HW*1e3:6.1f} usec",
            )
        return row

    def get_iv_protocol_from_pkl(
        self,
        row,
        pdf_pages: Union[object, None] = None,
        ds: Union[pd.DataFrame, None] = None,
        celltypes: Union[list, None] = None,
    ):
        """Get the IV protocol from this dataframe row.
        This is meant to be used in an "df.apply" function, so the row is passed in.
        Also checks the protocol "important" flag and adds it to the row information.
        Args:
            row (_type_): _description_
            pdf_pages:

            ds: datasummary dataframe to pull age information from
        """
        dataok = False
        if pd.isnull(row.date):
            return row
        if (
            self.experiment["celltypes"] != ["all"]
            and row.cell_type not in self.experiment["celltypes"]
        ):
            return row

        if int(row.name) > max_rows and max_rows != -1:
            return row
        row.date = row.date.replace("NF107ai32", "NF107Ai32")
        # if datenum(row.shortdate) != datenum("2019.04.15"):
        #     return row
        date = Path(row.date).name
        dsa = ds[ds["date"].str.endswith(date)]

        if len(dsa) == 0:
            raise ValueError(f"Date {row.date!s} / {date:s} not found in datasummary")
        row.age = dsa.iloc[0]["age"]
        print("    >>>> got row age: ", row.date, row.age)

        fullpatha = Path(
            self.experiment["rawdatapath"],
            self.experiment["directory"],
            date,
            row.slice_slice,
            row.cell_cell,
            row.iv_name,
        )
        fullpath = None
        for protocol in list(self.experiment["protocols"]["CCIV"].keys()):
            # print("testing protocol and fullpath name: ", protocol, fullpatha.name)
            if fullpatha.name.startswith(protocol):
                fullpath = fullpatha

        if fullpath is None:
            CP("r", "    No IVs of the right kind were found")
            return row  # Nothing to do here, there are no IVs of the right kind to analyze.

        # CP("c", f"RAM Used (GB): {psutil.virtual_memory()[3]/1000000000:.1f} ({psutil.virtual_memory()[2]:.1f}%)")

        with DR.acq4_reader.acq4_reader(fullpath, "MultiClamp1.ma") as AR:
            try:
                if AR.getData():
                    supindex = AR.readDirIndex(currdir=Path(fullpath.parent))
                    CP("g", f"    protocol read OK: {str(fullpath):s}")
                    dataok = True
                else:
                    if AR.error_info is not None:
                        Logger.error(AR.error_info)
                    AR.error_info = None
                    return row  # failed to get data, error will be indicated by acq4read
            except:
                CP("r", f"Acq4Read failed to read data file: {str(fullpath.name):s}")
                raise ValueError

            if not dataok:
                return row

            args = analysis_common.cmdargs  # get from default class
            args.dry_run = False
            args.merge_flag = True
            args.experiment = self.dataset
            args.iv_flag = True
            args.map_flag = False
            args.autoout = True
            args.parallel_mode = "cell"  # try to use parallel processing
            args.nworkers = self.nworkers
            args.max_spikeshape = 1
            args.downsample = 1
            print("process spike analysis: nworkers, parallel_mode ", args.nworkers, args.parallel_mode)
            IVA = EP.iv_analysis.IVAnalysis(args)
            IVA.configure(datapath=fullpatha, reader=AR, plot=True, pdf_pages=pdf_pages)
            # print('   Data sample freq (Hz): ', IVA.AR.sample_rate[0])
            # if IVA.AR.sample_rate[0] < 10000.0:
            #     print("    SR too low")
            #     return row
            # args.slicecell = "S1C0"
            IVA.iv_check(duration=0.1)
            try:
                IVA.AR.tstart
            except:
                CP("r", f"Failed to correctly read file: \n        {str(fullpath):s}")
                raise ValueError("Failed to read file")
                # return row
            row.sample_rate = IVA.AR.sample_rate[0]
            stable = IVA.stability_check(
                rmpregion=[0, IVA.AR.tstart - 0.002], threshold=rmp_sd_limit
            )
            # print("stablity: ", stable, " SD: ", IVA.RM.rmp_sd)
            row.rmp_sd = IVA.RM.rmp_sd
            if not stable:
                CP(
                    "m",
                    f"Cell is not stable (RMP SD = {row.rmp_sd:.4f}): {fullpath!s}\n",
                )
                return row

            IVA.compute_iv(to_peak=False, threshold=self.experiment["AP_threshold_V"], plotiv=True)
            if len(IVA.SP.spikeShapes) == 0:  # no spikes
                return row

            row = self.find_lowest_current_spike(row, IVA.SP)
            print(IVA.SP.analysis_summary.keys())
            
            row.AP15Rate = IVA.SP.analysis_summary["FiringRate_1p5T"]
            row.AdaptRatio = IVA.SP.analysis_summary["AdaptRatio"]
            row.RMP = IVA.RM.analysis_summary["RMP"]
            row.RMP_SD = IVA.RM.analysis_summary["RMP_SD"]
            row.Rin = IVA.RM.analysis_summary["Rin"]
            row.taum = IVA.RM.analysis_summary["taum"]
            row.AHP_trough_V = 1e3 * IVA.SP.analysis_summary["AHP_Trough_V"]
            row.AHP_trough_T = 1e3 * IVA.SP.analysis_summary["AHP_Trough_T"]
            row.AHP_depth_V = IVA.SP.analysis_summary["AHP_Depth"]
            row.holding = IVA.RM.analysis_summary["holding"]
            row.tauh = IVA.RM.analysis_summary["tauh_tau"]
            row.Gh = IVA.RM.analysis_summary["tauh_Gh"]
            row.FI_Curve = IVA.SP.analysis_summary["FI_Curve"]
            # CP("g", f"RAM Used (GB): {psutil.virtual_memory()[3]/1000000000:.1f} ({psutil.virtual_memory()[2]:.1f}%)")
            # print(resource.getrusage(resource.RUSAGE_SELF))
            gc.collect()
            return row

    def _get_iv_protocol(
        self,
        row,
        AR: DR.acq4_reader.acq4_reader,
        IVA: EP.iv_analysis.IVAnalysis,
        pdf_pages: Union[object, None] = None,
        axis=1,
    ):
        """Get the IV protocol from this dataframe row,
        compute the IV and spike information from the raw data using the ephys routines,
        and return the results in the row.
        This is meant to be called with .apply from a dataframe
        This does NOT update the data in the individual cell .pkl files.

        Args:
            row (Pandas row): Row with all the protocol information for analysis
            AR: an instance of the acq4_reader class
            IV: an instance of the IV_analysis class
            pdfpages: PDF_write object for plotting
            ds: datasummary dataframe to retrieve metadata information from
        """

        # if row.date != "2023.08.23_000":
        #     return row
        # if row.slice_slice != "slice_003" and row.cell_cell != "cell_000":
        #     return row

        # print("\nRow date: ", row.date, row.slice_slice, row.cell_cell)
        dataok = False
        if (
            self.experiment["celltypes"] != ["all"]
            and row.cell_type not in self.experiment["celltypes"]
        ):
            return row
        if pd.isnull(row.date):
            return row
        if int(row.name) > max_rows and max_rows != -1:
            return row

        row.date = row.date.replace("NF107ai32", "NF107Ai32")

        # Logger.info(f"\n**** Running analysis for :  {row.date!s}, {row.ID!s}, {row.age!s}, {row.Group!s}")

        # print(row.keys())
        msg = f"\nRetrieved metadata for: {row.date!s}\n    Type: {row.cell_type:s}"
        msg += f" Age={row.age!s}, T={row.temperature!s}, Internal Sol={row.internal:s}"
        CP("m", msg)
        Logger.info(msg)
        
        day_slice_cell = str(Path(row.date, row.slice_slice, row.cell_cell))
        CP("m", f"day_slice_cell: {day_slice_cell:s}")
        if (
            self.experiment["excludeIVs"] is not None
            and (day_slice_cell in self.experiment["excludeIVs"])
            and (
                (row.protocol in self.experiment["excludeIVs"][day_slice_cell]["protocols"])
                or self.experiment["excludeIVs"][day_slice_cell]["protocols"] in ["all", ["all"]]
            )
        ):
            CP("m", f"Excluded cell/protocol: {day_slice_cell:s}, {row.protocol:s}")
            Logger.info(f"Excluded cell: {day_slice_cell:s}, {row.protocol:s}")
            gc.collect()
            return row

        fullpatha = Path(
            self.experiment["rawdatapath"],
            # self.experiment["directory"],
            row.date,
            row.slice_slice,
            row.cell_cell,
            row.protocol,
        )
        fullpath = None
        for protocol in list(self.experiment["protocols"]["CCIV"].keys()):
            # print("testing protocol and fullpath name: ", protocol, fullpatha.name)
            if fullpatha.name.startswith(protocol):
                fullpath = fullpatha
        if fullpath is None:
            msg = f"protocol did not match usable CCIV-type list: {day_slice_cell:s}, {row.protocol:s}"
            CP("m", msg)
            gc.collect()
            raise ValueError(msg)
            return row

        CP(
            "m",
            f"RAM Used (GB): {psutil.virtual_memory()[3]/1000000000:.1f} ({psutil.virtual_memory()[2]:.1f}%)",
        )
        if fullpath.exists():
            CP("m", f"OK: {fullpath!s}")
            Logger.info(f"OK: {fullpath!s}")
        else:
            CP("m", f"File not found: {fullpath!s}")
            Logger.error(f"File not found: {fullpath!s}")
        AR.setProtocol(fullpath)
        AR.setDataName("MultiClamp1.ma")  # set the data file name
        # with DR.acq4_reader.acq4_reader(fullpath, "MultiClamp1.ma") as AR:
        try:
            if AR.getData():
                supindex = AR.readDirIndex(currdir=Path(fullpath.parent))
                CP("m", f"    Protocol read OK: {fullpath.name!s}")
                Logger.info(f"    Protocol read OK: {fullpath.name!s}")
                dataok = True
            else:
                if AR.error_info is not None:
                    Logger.error(AR.error_info)
                gc.collect()
                return row  # failed to get data, error will be indicated by acq4read
        except ValueError:
            CP("m", f"Acq4Read failed to read data file: {str(fullpath):s}")
            Logger.critical(f"Acq4Read failed to read data file: {str(fullpath):s}")
            gc.collect()
            return row

        if not dataok:
            gc.collect()
            return row  # no update
        CP("m", f"\nFullpath to protocol: {fullpatha!s}")
        row.important = AR.checkProtocolImportant(fullpatha)
        # print("protocol_index: ", protocol_index)
        # if important:
        #     # print(f"   Protocol index important: <{protocol_index["important]"!s}>")
        #     row.important = important
        #     # CP("m", f"   Important: {row.important!s}")
        # else:
        #     CP("r", f"   Important flag: not found, set False")
        #     row.important = False
        # return row

        IVA.configure(datapath=fullpatha, reader=AR, plot=True, pdf_pages=pdf_pages)
        if AR.sample_rate[0] < 10000.0:
            CP("m", "    Sample Rate too low")
            Logger.info("    Sample Rate for this protocol is too low")
            return row
        IVA.iv_check(duration=0.1)
        row.sample_rate = AR.sample_rate[0]

        try:
            IVA.AR.tstart
        except:
            CP("m", f"Failed to correctly read file: \n        {str(fullpath):s}")
            Logger.critical(f"Failed to correctly read file: \n        {str(fullpath):s}")
            raise ValueError("Failed to read file")
            # return row
        stable = IVA.stability_check(rmpregion=[0, IVA.AR.tstart - 0.002], threshold=rmp_sd_limit)
        # print("stablity: ", stable, " SD: ", IVA.RM.rmp_sd)
        row.rmp_sd = IVA.RM.rmp_sd
        if not stable:
            return row

        CP("m", f"Processing Spikes: Computing IV: {day_slice_cell:s}, {row.protocol:s}")
        IVA.compute_iv(to_peak=False, threshold=self.experiment["AP_threshold_V"], plotiv=True)
        if len(IVA.SP.spikeShapes) == 0:  # no spikes
            CP("y", f"No spike shape data: {day_slice_cell:s}, {row.protocol:s}")
            return row

        row = self.find_lowest_current_spike(row, IVA.SP)
        # print("DVDT CURRENT: ", row.dvdt_current)
        row.AP15Rate = IVA.SP.analysis_summary["FiringRate_1p5T"]
        row.AdaptRatio = IVA.SP.analysis_summary["AdaptRatio"]
        row.RMP = IVA.RM.analysis_summary["RMP"]
        row.RMP_SD = IVA.RM.analysis_summary["RMP_SD"]
        row.Rin = IVA.RM.analysis_summary["Rin"]
        row.taum = IVA.RM.analysis_summary["taum"]
        if "AP_begin_V" in IVA.SP.analysis_summary.keys():
            row.AP_begin_V = 1e3 * IVA.SP.analysis_summary["AP_begin_V"]
        row.AHP_trough_V = 1e3 * IVA.SP.analysis_summary["AHP_Trough_V"]  # depth, absolute V
        row.AHP_trough_T = IVA.SP.analysis_summary["AHP_Trough_T"]  # absolute time
        row.AHP_depth_V = IVA.SP.analysis_summary["AHP_Depth"]  # depth from spike threshold
        row.holding = IVA.RM.analysis_summary["holding"]
        row.tauh = IVA.RM.analysis_summary["tauh_tau"]
        row.Gh = IVA.RM.analysis_summary["tauh_Gh"]

        row.FI_Curve = IVA.SP.analysis_summary["FI_Curve"]
        CP(
            "g",
            f"RAM Used (GB): {psutil.virtual_memory()[3]/1000000000:.1f} ({psutil.virtual_memory()[2]:.1f}%)",
        )
        # print(resource.getrusage(resource.RUSAGE_SELF))
        gc.collect()
        return row

    def _make_short_name(self, row):
        return self.get_rec_date(row["date"])

    def _data_complete_to_series(self, row):
        dc = row.data_complete.split(",")
        dc = [p.strip(" ") for p in dc if p != "nan" and "CCIV".casefold() in p.casefold()]
        # print("\ndc: ", dc)
        row.protocol = pd.Series(dc)
        # print(row.date, row.data_complete.values)
        return row

    def clean_up_after_merge(self, df, colname):
        col_x = colname + "_x"
        col_y = colname + "_y"
        if col_x in df.columns and col_y in df.columns:
            # two different sets of data in the merge with same name.
            df.fillna({col_x: col_y}, inplace=True)
            df.loc[df[col_x] == " ", col_x] = df[col_y]
            df.rename(columns={col_x: colname}, inplace=True)
            df.drop([col_y], axis=1)
        return df

    def find_and_process_protocols(
        self,
        exp: dict,
        datasummary: Union[Path, str],
        codesheet: Union[Path, None, str] = None,
        result_sheet: Union[Path, None, str] = None,
        pdf_pages: Union[object, None] = None,
    ):
        """find_and_process_protocols - find the complete IV protocols from the datasummary file,
        generate a new row for each one, and merge with the code sheet (to
        set the coding).
        Return the results as a pandas dataframe.

        Parameters
        ----------
        datasummary: The original pd datasummary output filename.
        codesheet : Union[Path, None, str], optional
            The excel sheet holding the "coding" information - how
            to code individual days (NOT cells) according to subject treatment.
            by default None
        result_sheet : Union[Path, None, str], optional
            the name of the result sheet file, by default None
            This file is used by plot_spike_info.py to generate plots
        pdf_pages : Union[object, None], optional
            Output file for plots, by default None

        Returns
        -------
        _type_
            _description_
        """

        if codesheet is not None:
            df_codes = pd.read_excel(codesheet)

        assert result_sheet is not None

        df_summary = pd.read_pickle(datasummary)

        # print("code_df: ", df_codes.date)
        # NOTE: This merge requires that the date fields in both the codes and summary files
        # match up. If they don't the problem is likely in the coding file date names.

        if codesheet is not None:
            df = pd.merge(df_summary, df_codes, left_on="date", right_on="date", how="left")
        else:
            df = df_summary
            df["Group"] = "Control"  # all are control if there is no codesheet
            df["ID"] = ""  # no animal ID here
            df["Date"] = ""  # no Date from here

        cols_to_clean = [
            "sex",
            "age",
            "cell_expression",
            "cell_cell",
            "slice_slice",
            "animal_identifier",
            "important",
            "reporters",
            "strain",
        ]
        for colname in cols_to_clean:
            df = self.clean_up_after_merge(df, colname)

        print("Number of potential cells: ", len(df))
        df["protocol"] = np.nan
        # for index in df.index:
        #     row = df.loc[index]
        #     print("before data_complete_to_series:\n    ", index, row.date, row.slice_slice, row.cell_cell, row.data_complete)
        #     if index > 20:
        #         break
        # convert the data complete column to a list of protocols
        df = df.apply(self._data_complete_to_series, axis=1)
        # for index in df.index:
        #     row = df.loc[index]
        #     print("after data_complete_to_series:\n    ", 
        #           index, row.date, row.slice_slice, row.cell_cell, "<", row.data_complete, ">", "prot: ", row.protocol)
        #     if index > 20:
        #         exit()

        print(len(df), " rows after data complete to series")
        # now make a new dataframe that has a separate row for each protocol
        df = df.explode(["protocol"], ignore_index=True)
        print("Number of protocols after explode", len(df))
        df = df.dropna(subset=["protocol"])
        print("Number of protocols after dropna", len(df))
        # for index in df.index:
        #     row = df.loc[index]
        #     print("after data_complete_to_series:\n    ", 
        #           index, row.date, row.slice_slice, row.cell_cell, "<", row.data_complete, ">", "prot: ", row.protocol)
        #     if index > 19:
        #         exit()

        df_null = df[df["cell_id"].isnull()]
        print("Null columns: ", df_null)
        df = df.dropna(subset=["cell_id"])
        print("# of protocols with ID: ", len(df))
        protostrings = "|".join(list(self.experiment["protocols"]["CCIV"].keys()))
        print("protostrings: ", protostrings)
        print(df["protocol"].unique())

        df = df[df["protocol"].str.contains(protostrings)]

        print("# of protocols of right type: ", len(df))
        print(df["protocol"].unique())
        Logger.info(f"Number of protocols of right type for analysis: {len(df):d}")
        add_cols = [
            "important",
            "holding",
            "sample_rate",
            "RMP",
            "RMP_SD",
            "Rin",
            "taum",
            "dvdt_rising",
            "dvdt_falling",
            "dvdt_current",
            "AP_thr_V",
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
        ]

        for col in add_cols:
            df[col] = np.nan
        nprots = 0
        # now do an analysis
        Logger.info("Starting analysis on protocols in database")
        print("workers: ", self.nworkers)
        args = analysis_common.cmdargs  # get from default class
        args.dry_run = False
        args.merge_flag = True
        args.experiment = self.experiment
        args.iv_flag = True
        args.map_flag = False
        args.autoout = True
        args.nworkers = self.nworkers
        AR = DR.acq4_reader.acq4_reader()
        IVA = EP.iv_analysis.IVAnalysis(args)
        if len(df) < self.nworkers:
            nworkers = len(df)
        else:
            nworkers = self.nworkers
        df = df.apply(self._get_iv_protocol, AR=AR, IVA=IVA, pdf_pages=None, axis=1)

        # df = DataFrameParallel(df, n_cores=nworkers - 2, pbar=True).apply(
        #     self._get_iv_protocol, AR=AR, IVA=IVA, pdf_pages=None, axis=1
        # )

        writer = pd.ExcelWriter(result_sheet)
        df.to_excel(writer, sheet_name="Sheet1")
        for i, column in enumerate(df.columns):
            column_width = max(df[column].astype(str).map(len).max(), 24)  # len(column))
            writer.sheets["Sheet1"].set_column(
                first_col=i + 1, last_col=i + 1, width=column_width
            )  # column_dimensions[str(column.title())].width = column_width
        writer.close()
        CP("g", f"\nSpike analysis complete. Results in {str(result_sheet):s}\n{'='*80:s}\n")
        return df

    def process_spikes(self):
        exp = self.experiment
        result_sheet = Path(exp["databasepath"], exp["directory"], exp["result_sheet"])
        cleaned_sheet = Path(
            exp["databasepath"],
            exp["directory"],
            result_sheet.stem + "_cleaned",
        ).with_suffix(".xlsx")

        pp = pprint.PrettyPrinter(width=88, compact=False)
        pp.pprint(exp)
        print("output file: \n .   ", cleaned_sheet)

        if exp["coding_file"] is None:
            codesheet = None
        else:
            codesheet = Path(exp["databasepath"], exp["directory"], exp["coding_file"])

        with PdfPages(exp["pdf_filename"]) as pdfs:
            datasummaryfile = Path(
                exp["databasepath"], exp["directory"], exp["datasummaryFilename"]
            )
            protos = self.find_and_process_protocols(
                exp,
                datasummary=datasummaryfile,
                codesheet=codesheet,
                result_sheet=result_sheet,
                pdf_pages=pdfs,
            )
            gc.collect()

        # DE.cleanup(result_sheet, outfile=cleaned_sheet)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Further Processing for Spike Analysis")
    parser.add_argument(
        "-d",
        type=str,
        default="",
        dest="dataset",
        help="Plot to create",
    )
    parser.add_argument(
        "-e",
        "--experiment",
        type=str,
        default="",
        dest="experiment",
        help="Experiment to analyze",
    )

    args = parser.parse_args()
    PSA = ProcessSpikeAnalysis(args.dataset, args.experiment)
    # exp = analyze_ivs.experiments[dataset]
    # result_sheet = Path(exp["databasepath"], exp["directory"], exp["result_sheet"])
    # cleaned_sheet = Path(
    #     exp["databasepath"],
    #     exp["directory"],
    #     result_sheet.stem + "_cleaned",
    # ).with_suffix(".xlsx")
    # # print(result_sheet)
    # result_df = pd.read_excel(result_sheet)
    # result_df.dropna(subset=["cell_type"], inplace=True)
    # DE.cleanup(result_sheet, outfile=str(cleaned_sheet))
