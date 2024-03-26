"""
Compute IV Information


"""
import datetime
import gc
import logging
from collections import OrderedDict
from pathlib import Path
from typing import List, Literal, Union

import numpy as np
import pandas as pd
import pyqtgraph.multiprocess as MP
import pylibrary.tools.cprint as CP

import ephys.tools.build_info_string as BIS
import ephys.tools.functions as functions
from ephys.ephys_analysis.analysis_common import Analysis
import ephys.datareaders.acq4_reader as acq4_reader
import ephys.ephys_analysis.spike_analysis as spike_analysis
import ephys.ephys_analysis.rm_tau_analysis as rm_tau_analysis

color_sequence = ["k", "r", "b"]
colormap = "snshelix"

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
        logging.CRITICAL: bold_red + lineformat + reset
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)

logging.getLogger("fontTools.subset").disabled = True
Logger = logging.getLogger("AnalysisLogger")
level = logging.DEBUG
Logger.setLevel(level)
# create file handler which logs even debug messages
logging_fh = logging.FileHandler(filename="iv_analysis.log")
logging_fh.setLevel(level)
logging_sh = logging.StreamHandler()
logging_sh.setLevel(level)
log_formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s  (%(filename)s:%(lineno)d) - %(message)s ")
logging_fh.setFormatter(log_formatter)
logging_sh.setFormatter(CustomFormatter()) # log_formatter)
Logger.addHandler(logging_fh)
# Logger.addHandler(logging_sh)
# setFileConfig(filename="iv_analysis.log", encoding='utf=8')


class IVAnalysis(Analysis):

    Logger = logging.getLogger("AnalysisLogger")
    def __init__(self, args):
        super().__init__(args)
        self.IVFigure = None
        self.mode = "acq4"
        self.AR: acq4_reader = acq4_reader.acq4_reader()
        self.RM: rm_tau_analysis = rm_tau_analysis.RmTauAnalysis()
        self.SP: spike_analysis = spike_analysis.SpikeAnalysis()
    
        Logger.info("Instantiating IVAnalysis class")

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        del self.AR
        del self.SP
        del self.RM
        gc.collect()

    def reset_analysis(self):
        pass

    def configure(
        self,
        datapath,
        altstruct=None,
        file: Union[str, Path, None] = None,
        spikeanalyzer: Union[object, None] = None,
        reader: Union[object, None] = None,
        rmtauanalyzer: Union[object, None] = None,
        plot: bool = True,
        pdf_pages: Union[object, None] = None,
    ):
        self.pdf_pages = pdf_pages
        if datapath is not None:
            self.AR = reader
            self.datapath = datapath
        else:
            self.AR = altstruct
            self.datapath = file
            self.mode = "nwb2.5"
        self.datapath = str(datapath)
        if spikeanalyzer is not None:
            self.SP = spikeanalyzer  # spike_analysis.SpikeAnalysis()
        if rmtauanalyzer is not None:
            self.RM = rmtauanalyzer  # rm_tau_analysis.RmTauAnalysis()
        self.plot = plot
        self.plotting_mode = "normal"
        self.decorate = True
        self.plotting_alternation = 1
        self.SP.analysis_summary = {}
        self.RM.analysis_summary = {}

    def iv_check(self, duration=0.0):
        """
        Check the IV for a particular duration, but does no analysis
        """
        if duration == 0:
            return True
        if self.mode == "acq4":
            self.AR.setProtocol(self.datapath)
        if self.AR.getData():
            dur = self.AR.tend - self.AR.tstart
            if np.fabs(dur - duration) < 1e-4:
                return True
        return False

    def plot_mode(
        self,
        mode: Literal["pubmode", "traces_only", "normal", None] = None,
        alternate: int = 1,
        decorate: bool = True,
    ):
        assert mode in ["pubmode", "traces_only", "normal"]
        self.plotting_mode = mode
        self.plotting_alternation = alternate
        self.decorate = decorate

    def stability_check(self, rmpregion: List = [0.0, 0.005], threshold=2.0):
        """_summary_

        Args:
            rmpregion (List, optional): timw window over which to measure RMP. Defaults to [0.0, 0.005]. in seconds
            threshold (float, optional): Maximum SD of trace RMP. Defaults to 2.0. In mV

        Returns:
            bool: True if RMP is stable, False otherwise.
        """
        if self.mode == "acq4":
            self.AR.setProtocol(
                self.datapath
            )  # define the protocol path where the data is
        if self.AR.getData():  # get that data.
            self.RM.setup(self.AR, self.SP, bridge_offset=0.0)
        self.RM.rmp_analysis(time_window=rmpregion)
        # print(self.RM.rmp, self.RM.rmp_sd, threshold)
        if self.RM.rmp_sd > threshold:
            return False
        else:
            return True

    def analyze_ivs(
        self,
        icell,
        allprots: dict,
        celltype: str,
        pdf=None,
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
        Logger.info("Starting iv_analysis")
        msg = f"    Analyzing IVs for index: {icell: d} dir: {str(self.df.iloc[icell].data_directory):s}"
        msg += f"cell: ({str(self.df.iloc[icell].cell_id):s} )"
        CP.cprint(
            "c", msg
        )
        Logger.info(msg)
        cell_directory = Path(
            # self.df.iloc[icell].data_directory, self.experiment['directory'], self.df.iloc[icell].cell_id
            self.df.iloc[icell].data_directory, self.df.iloc[icell].cell_id
        )
        CP.cprint("m", f"File: {str(cell_directory):s}")
        CP.cprint("m", f"   Cell id: {str(self.df.iloc[icell].cell_id):s},  cell_type: {str(self.df.iloc[icell].cell_type):s}")

        cell_index = self.AR.getIndex(cell_directory)
        if "important" in cell_index.keys():
            important = cell_index["important"]
            CP.cprint("m", f"   Important: {important!s}")
        else:
            CP.cprint("r", f"   Important flag: flag was not found")   
            important = False 
        if self.important_flag_check and not important:  # cell not marked important, so skip
            print("Not important? ")
            return
        # CP.cprint("m", f"Notes: {self.df.iloc[icell].notes!s}")
        # print(self.df.columns)
        # CP.cprint("m", f"Location: {self.df.iloc[icell].cell_location!s}")
        # CP.cprint("m", f"   type: {self.df.iloc[icell].cell_type!s}  layer: {self.df.iloc[icell].cell_layer!s}")
        if "IV" not in self.df.columns.values:
            self.df = self.df.assign(IV=None)
        if "Spikes" not in self.df.columns.values:
            self.df = self.df.assign(Spikes=None)
        msg = f"      Cell: {str(cell_directory):s}\n           at: {datetime.datetime.now().strftime('%m/%d/%Y, %H:%M:%S'):s}"
        # CP.cprint(
        #     "c", msg,
        # )
        Logger.info(msg)

        # clean up data in IV and Spikes : remove Posix
        def _cleanup_ivdata(results: dict):
            if isinstance(results, dict) and len(results) == 0:
                results = None
            if results is not None:
                for r in list(results.keys()):
                    u = results[r]
                    if not isinstance(u, dict):
                        results[str(r)] = results.pop(r)
                        continue
                    for k in u.keys():
                        if isinstance(u[k], dict):
                            for uk in u[k].keys():
                                if isinstance(u[k][uk], bool) or isinstance(
                                    u[k][uk], int
                                ):
                                    u[k][uk] = int(u[k][uk])
                        if k in ["taupars", "RMPs", "Irmp"]:
                            # if isinstance(u[k], Iterable) and not isinstance(u[k], (dict, list, float, str, nfloat)):
                            # print("type for ", k, type(u[k]))
                            if isinstance(u[k], np.ndarray):
                                u[k] = u[k].tolist()
                            elif isinstance(u[k], list) and len(u[k]) > 0:
                                if isinstance(u[k][0], np.ndarray):
                                    u[k] = u[k][0].tolist()
                            elif isinstance(u[k], float):
                                u[k] = float(u[k])
                            elif isinstance(u[k], list):
                                pass
                            else:
                                print(type(u[k]))
                                raise ValueError
                        # if isinstance(u[k], Iterable) and not isinstance(u[k], (dict, list, float, str)):
                        #     u[k] = u[k].tolist()
                        elif isinstance(u[k], float):
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
        print("allprots: ", allprots["CCIV"])
        cciv_protocols = list(self.experiment['protocols']["CCIV"].keys())
        for protoname in allprots["CCIV"]:  # check all the protocols
            protocol_type = str(Path(protoname).name)[:-4]
            print(" protocol type: ", protocol_type, end="")
            if protocol_type in cciv_protocols: # ["stdIVs", "CCIV_long", "CCIV_posonly", "CCIV_GC"]:  # just CCIV types
                allivs.append(protoname)  # combine into a new list
                print("     appended")
            else:
                print("     not appended")

        validivs = allivs 
        # build a list of all exclusions
        exclude_ivs = []
        for ex_cell in self.exclusions.keys():
            for ex_proto in self.exclusions[ex_cell]['protocols']:
                exclude_ivs.append(str(Path(ex_cell, ex_proto)))
        print("Excluded IVs: ", exclude_ivs)
        # the exclusion list is shorter (we hope), so let's iterate over it for removal
        for iv_proto in exclude_ivs:
            if iv_proto in validivs:
                validivs.remove(iv_proto)
        self.noparallel = False
        print("Valid IDs: ", validivs)
        print("NoParallel: ", self.noparallel)
        nworkers = self.nworkers  # number of cores/threads to use
        print("nworkers: ", nworkers)
        tasks = range(len(validivs))  # number of tasks that will be needed
        results: dict = dict(
            [("IV", {}), ("Spikes", {})]
        )  # storage for results; predefine the dicts.

        if self.noparallel:  # just serial...
            for i, x in enumerate(tasks):
                r, nfiles = self.analyze_iv(
                    icell=icell,
                    i=i,
                    x=x,
                    cell_directory=cell_directory,
                    allivs=validivs,
                    nfiles=nfiles,
                    pdf=pdf,
                )
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
            with MP.Parallelize(
                enumerate(tasks), results=results, workers=nworkers
            ) as tasker:
                for i, x in tasker:
                    result, nfiles = self.analyze_iv(
                        icell, i, x, cell_directory, validivs, nfiles
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
        def clean_exp_unit(row):
            if pd.isnull(row.expUnit):
                return 0
            else:
                if row.expUnit:
                    return 1
                else:
                    return 0
        self.df["expUnit"] = self.df.apply(clean_exp_unit, axis=1)
        # self.df["expUnit"] = self.df["expUnit"].astype(int)

        if (
            len(allivs) > 0
            and self.iv_analysisFilename is not None
            and Path(self.iv_analysisFilename).suffix == ".h5"
        ):
            msg = f"Writing IV analysis results to HDF5 file: {str(self.iv_analysisFilename):s}"
            CP.cprint(
                "m",
                msg,
            )
            Logger.info(msg)
            # with hdf5:
            # Note, reading this will be slow - it seems to be rather a large file.
            day, slice, cell = self.make_cell(icell=icell)
            keystring = str(
                Path(Path(day).name, slice, cell)
            )  # the keystring is the cell.
            # pytables does not like the keystring starting with a number, or '.' in the string
            # so put "d_" at start, and then replace '.' with '_'
            # what a pain.
            keystring = "d_" + keystring.replace(".", "_")
            if self.n_analyzed == 0:
                self.df.iloc[icell].to_hdf(
                    self.iv_analysisFilename, key=keystring, mode="w"
                )
            else:
                self.df.iloc[icell].to_hdf(
                    self.iv_analysisFilename, key=keystring, mode="a"
                )
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
            msg = f"Writing IV analysis results to FEATHER file: {str(self.iv_analysisFilename):s}"
            CP.cprint(
                "b",
                msg,
            )
            Logger.info(msg)
            with open(self.iv_analysisFilename, "wb") as fh:
                self.df.to_feather(fh)


    def analyze_iv(
        self,
        icell: int,
        i: int,
        x: int,
        cell_directory: Union[Path, str],
        allivs: list,
        nfiles: int,
    ):
        """
        Compute various measures (input resistance, spike shape, etc) for ONE IV
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

        """

        protocol = Path(allivs[i]).name
        result = {}
        iv_result = {}
        sp_result = {}
        print("cell directory: ", cell_directory)
        print("protocol: ", protocol)

        protocol_directory = Path(cell_directory, protocol)

        if not protocol_directory.is_dir():
            msg = f"Protocol directory not found (A): {str(protocol_directory):s}"
            CP.cprint(
                "r",
                msg,   )
            Logger.error(msg)
            exit()
        if self.important_flag_check:
            if not self.AR.checkProtocolImportant(protocol_directory):
                msg = f"Skipping protocol marked as not important: {str(protocol_directory):s}"
                CP.cprint(
                    "r",
                    msg,
                )
                Logger.info(msg)
                return (None, 0)

        self.configure(
            protocol_directory,
            plot=not self.plotsoff,
            reader=self.AR,
            spikeanalyzer=self.SP,
            rmtauanalyzer=self.RM,
        )
        if self.iv_select["duration"] > 0.0:
            check = self.iv_check(duration=self.iv_select["duration"])
            if check is False:
                gc.collect()
                return (None, 0)  # skip analysis
        if not self.dry_run:
            msg = f"      IV analysis for: {str(protocol_directory):s}"
            print(msg)
            # Logger.info(msg)
            br_offset = 0.0
            if (
                not pd.isnull(self.df.at[icell, "IV"])
                and protocol in self.df.at[icell, "IV"]
                and "--Adjust" in self.df.at[icell, protocol]["BridgeAdjust"]
            ):
                msg = "Bridge: {0:.2f} Mohm".format(
                        self.df.at[icell, "IV"][protocol]["BridgeAdjust"] / 1e6
                    )
                print(msg)
                Logger.info(msg)

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
            self.plot_mode(mode=self.IV_pubmode)
            plot_handle = self.compute_iv(
                threshold=self.spike_threshold,
                refractory=self.refractory,
                bridge_offset=br_offset,
                tgap=tgap,
                max_spikeshape=self.max_spikeshape,
                max_spike_look = self.max_spike_look,
                plotiv=True,
                full_spike_analysis=True,
            )
            iv_result = self.RM.analysis_summary
            sp_result = self.SP.analysis_summary
            result["IV"] = iv_result
            result["Spikes"] = sp_result
            ctype = self.df.at[icell, "cell_type"]
            annot = self.df.at[icell, "annotated"]
            if annot:
                ctwhen = "[revisited]"
            else:
                ctwhen = "[original]"
            nfiles += 1
            # print("Checking for figure, plothandle is: ", plot_handle)
            
            del iv_result
            del sp_result
            gc.collect()
            return result, nfiles

        else:
            print(f"Dry Run: would analyze {str(protocol_directory):s}")
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


    def compute_iv(
        self,
        threshold: float = -0.010,
        refractory=0.0007,
        bridge_offset: float = 0.0,
        tgap: float = 0.0005,
        plotiv: bool = False,
        full_spike_analysis: bool = True,
        max_spikeshape: int = 2,
        max_spike_look: float = 0.010,  # time in seconds to look for AHPs
        to_peak: bool = True,
    ) -> bool:
        """
        Simple computation of spikes, FI and subthreshold IV for one protocol

        """
        if self.mode == "acq4":
            self.AR.setProtocol(
                self.datapath
            )  # define the protocol path where the data is
        if self.AR.getData():  # get that data.
            if self.important_flag_check:
                if not self.AR.protocol_important:
                    return None  # skip this protocol
            # downsample, but also change the data in the acq4Reader
            if self.downsample > 1 and self.AR.sample_rate > 1e4:
                print("Decimating with: ", self.decimate)
                self.AR.traces = functions.downsample(self.AR.traces, n=self.downsample, axis=1)
                self.AR.cmd_wave = functions.downsample(self.AR.cmd_wave, n=self.downsample, axis=1)
                self.AR.time_base = functions.downsample(self.AR.time_base, n=self.downsample, axis=0)
            self.RM.setup(self.AR, self.SP, bridge_offset=bridge_offset)
            self.SP.setup(
                clamps=self.AR,
                threshold=threshold,
                refractory=refractory,
                peakwidth=0.001,
                interpolate=True,
                verify=False,
                mode="schmitt",
                max_spike_look = max_spike_look,
            )
            self.SP.analyzeSpikes()
            if full_spike_analysis:
                self.SP.analyzeSpikeShape(max_spikeshape=max_spikeshape)
                # self.SP.analyzeSpikes_brief(mode="evoked")
                self.SP.analyzeSpikes_brief(mode="baseline")
                self.SP.analyzeSpikes_brief(mode="poststimulus")
            # self.SP.fitOne(function='fitOneOriginal')
            self.RM.analyze(
                rmpregion=[0.0, self.AR.tstart - 0.001],
                tauregion=[
                    self.AR.tstart,
                    self.AR.tstart + (self.AR.tend - self.AR.tstart) / 2.0,
                ],
                to_peak=to_peak,
                tgap=tgap,
            )
            return True

        else:
            msg = f"IVAnalysis::compute_iv: acq4_reader.getData found no data to return from: \n  > {str(self.datapath):s} ",
            print(msg)
            Logger.error(msg)
        return None

    
