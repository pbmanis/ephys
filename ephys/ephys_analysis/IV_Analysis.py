"""
Compute IV Information


"""
import datetime
import gc
import logging
from collections import OrderedDict
from pathlib import Path
from typing import List, Literal, Union

import matplotlib
matplotlib.use("QtAgg")
import matplotlib.pyplot as mpl  # import locally to avoid parallel problems
import numpy as np
import pandas as pd
import pyqtgraph.multiprocess as MP
import pylibrary.plotting.plothelpers as PH
import pylibrary.tools.cprint as CP

import ephys.tools.build_info_string as BIS
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
    def __init__(self, args):
        super().__init__(args)
        self.IVFigure = None
        self.mode = "acq4"
        self.AR: acq4_reader = acq4_reader.acq4_reader()
        self.RM: rm_tau_analysis = rm_tau_analysis.RmTauAnalysis()
        self.SP: spike_analysis = spike_analysis.SpikeAnalysis()

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
            self.df.iloc[icell].data_directory, self.experiment['directory'], self.df.iloc[icell].cell_id
        )
        CP.cprint("m", f"File: {str(cell_directory):s}")
        CP.cprint("m", f"   Cell id: {str(self.df.iloc[icell].cell_id):s},  cell_type: {str(self.df.iloc[icell].cell_type):s}")

        cell_index = self.AR.getIndex(cell_directory)
        if "important" in cell_index.keys():
            important = cell_index["important"]
            CP.cprint("m", f"   Important: {important!s}")
        else:
            CP.cprint("r", f"   Important flag: not found")   
            important = False 
        if not important:  # cell not marked important, so skip
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
                                    u[k][uk], int
                                ):
                                    u[k][uk] = int(u[k][uk])
                        if k in ["taupars", "RMPs", "Irmp"]:
                            # if isinstance(u[k], Iterable) and not isinstance(u[k], (dict, list, float, str, nfloat)):
                            # print("type for ", k, type(u[k]))
                            if isinstance(u[k], numpy.ndarray):
                                u[k] = u[k].tolist()
                            elif isinstance(u[k], list) and len(u[k]) > 0:
                                if isinstance(u[k][0], numpy.ndarray):
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
        print(allprots)
        print("allprots: ", allprots["CCIV"])
        cciv_protocols = list(self.experiment['protocols']["CCIV"].keys())
        for protoname in allprots["CCIV"]:  # check all the protocols
            protocol_type = str(Path(protoname).name)[:-4]
            print(" protocol type: ", protocol_type)
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
        print("Valid IDs: ", validivs)
        nworkers = 8  # number of cores/threads to use
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
            Logger.info(msg)
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
            # print("Checking for figure, plothandle is: ", plot_handle)
            if plot_handle is not None:
                shortpath = protocol_directory.parts
                shortpath2 = str(Path(*shortpath[4:]))
                plot_handle.suptitle(
                    f"{str(shortpath2):s}\n{BIS.build_info_string(self.AR, protocol_directory):s}",
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
                msg = f"saved to: {str(t_path):s}"
                CP.cprint("g", msg)
                Logger.info(msg)
                nfiles += 1
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

    import matplotlib.figure

    def compute_iv(
        self,
        threshold: float = -0.010,
        refractory=0.0007,
        bridge_offset: float = 0.0,
        tgap: float = 0.0005,
        plotiv: bool = False,
        full_spike_analysis: bool = True,
        max_spikeshape: int = 2,
        to_peak: bool = True,
    ) -> Union[matplotlib.figure.Figure, None]:
        """
        Simple plot of spikes, FI and subthreshold IV

        """
        if self.mode == "acq4":
            self.AR.setProtocol(
                self.datapath
            )  # define the protocol path where the data is
        if self.AR.getData():  # get that data.
            if self.important_flag_check:
                if not self.AR.protocol_important:
                    return None  # skip this protocol
            self.RM.setup(self.AR, self.SP, bridge_offset=bridge_offset)
            self.SP.setup(
                clamps=self.AR,
                threshold=threshold,
                refractory=refractory,
                peakwidth=0.001,
                interpolate=True,
                verify=False,
                mode="schmitt",
            )
            self.SP.analyzeSpikes()
            if full_spike_analysis:
                self.SP.analyzeSpikeShape()
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
            if plotiv:
                fh = None
                if self.plotting_mode == "normal":
                    fh = self.plot_iv()
                elif self.plotting_mode == "pubmode":
                    fh = self.plot_iv(pubmode=True)
                elif self.plotting_mode == "traces_only":
                    fh = self.plot_fig()
                else:
                    raise ValueError(
                        "Plotting mode not recognized: ", self.plotting_mode
                    )
                return fh
        else:
            msg = f"IVAnalysis::compute_iv: acq4_reader.getData found no data to return from: \n  > {str(self.datapath):s} ",
            print(msg)
            Logger.error(msg)
        return None

    def plot_iv(self, pubmode=False) -> Union[None, object]:
        if not self.plot:
            return None
        x = -0.08
        y = 1.02
        sizer = {
            "A": {"pos": [0.05, 0.50, 0.2, 0.71], "labelpos": (x, y), "noaxes": False},
            "A1": {
                "pos": [0.05, 0.50, 0.08, 0.05],
                "labelpos": (x, y),
                "noaxes": False,
            },
            "B": {"pos": [0.62, 0.30, 0.74, 0.17], "labelpos": (x, y), "noaxes": False},
            "C": {"pos": [0.62, 0.30, 0.52, 0.17], "labelpos": (x, y)},
            "D": {"pos": [0.62, 0.30, 0.30, 0.17], "labelpos": (x, y)},
            "E": {"pos": [0.62, 0.30, 0.08, 0.17], "labelpos": (x, y)},
        }
        # dict pos elements are [left, width, bottom, height] for the axes in the plot.
        gr = [
            (a, a + 1, 0, 1) for a in range(0, len(sizer))
        ]  # just generate subplots - shape does not matter
        axmap = OrderedDict(zip(sizer.keys(), gr))
        P = PH.Plotter((len(sizer), 1), axmap=axmap, label=True, figsize=(8.0, 10.0))
        # PH.show_figure_grid(P.figure_handle)
        P.resize(sizer)  # perform positioning magic
        now = datetime.datetime.now().strftime("%Y-%m-%d  %H:%M:%S %z")
        P.axdict["A"].text(
            0.96,
            0.01,
            s=now,
            fontsize=6,
            ha="right",
            transform=P.figure_handle.transFigure,
        )
        infostr = BIS.build_info_string(self.AR, self.AR.protocol)

        P.figure_handle.suptitle(f"{str(self.datapath):s}\n{infostr:s}", fontsize=8)
        dv = 50.0
        jsp = 0
        # create a color map ans use it for all data in the plot
        # the map represents the index into the data array,
        # and is then mapped to the current levels in the trace
        # in case the sequence was randomized or run in a different
        # sequence
        import colorcet
        import matplotlib.pyplot as mpl

        # matplotlib versions:
        # cmap = mpl.colormaps['tab20'].resampled(self.AR.traces.shape[0])
        cmap = colorcet.cm.glasbey_bw_minc_20_maxl_70
        trace_colors = [
            cmap(float(i) / self.AR.traces.shape[0])
            for i in range(self.AR.traces.shape[0])
        ]

        # colorcet versions:

        for i in range(self.AR.traces.shape[0]):
            if self.plotting_alternation > 1:
                if i % self.plotting_alternation != 0:
                    continue
            if i in list(self.SP.spikeShapes.keys()):
                idv = float(jsp) * dv
                jsp += 1
            else:
                idv = 0.0
            P.axdict["A"].plot(
                self.AR.time_base * 1e3,
                idv + self.AR.traces[i, :].view(np.ndarray) * 1e3,
                "-",
                color=trace_colors[i],
                linewidth=0.35,
            )
            P.axdict["A1"].plot(
                self.AR.time_base * 1e3,
                self.AR.cmd_wave[i, :].view(np.ndarray) * 1e9,
                "-",
                color=trace_colors[i],
                linewidth=0.35,
            )
            ptps = np.array([])
            paps = np.array([])
            if i in list(self.SP.spikeShapes.keys()) and self.decorate:
                for j in list(self.SP.spikeShapes[i].keys()):
                    paps = np.append(paps, self.SP.spikeShapes[i][j].peak_V * 1e3)
                    ptps = np.append(ptps, self.SP.spikeShapes[i][j].peak_T * 1e3)
                P.axdict["A"].plot(ptps, idv + paps, "ro", markersize=0.5)

            # mark spikes outside the stimlulus window
            if self.decorate:
                ptps = np.array([])
                paps = np.array([])
                clist = ["g", "b"]
                for k, window in enumerate(["baseline", "poststimulus"]):
                    ptps = np.array(self.SP.analysis_summary[window + "_spikes"][i])
                    uindx = [int(u / self.AR.sample_interval) + 1 for u in ptps]
                    paps = np.array(self.AR.traces[i, uindx])
                    P.axdict["A"].plot(
                        ptps * 1e3,
                        idv + paps * 1e3,
                        "o",
                        color=clist[k],
                        markersize=0.5,
                    )
        if not pubmode:
            for k in self.RM.taum_fitted.keys():
                # CP('g', f"tau fitted keys: {str(k):s}")
                if self.RM.tau_membrane == np.nan:
                    continue
                P.axdict["A"].plot(
                    self.RM.taum_fitted[k][0] * 1e3,  # ms
                    self.RM.taum_fitted[k][1] * 1e3,  # mV
                    "--g",
                    linewidth=1.0,
                )
            for k in self.RM.tauh_fitted.keys():
                # CP('r', f"tau fitted keys: {str(k):s}")
                if self.RM.tauh_meantau == np.nan:
                    continue
                P.axdict["A"].plot(
                    self.RM.tauh_fitted[k][0] * 1e3,
                    self.RM.tauh_fitted[k][1] * 1e3,
                    "--r",
                    linewidth=0.75,
                )
        if pubmode:
            PH.calbar(
                P.axdict["A"],
                calbar=[0.0, -90.0, 25.0, 25.0],
                axesoff=True,
                orient="left",
                unitNames={"x": "ms", "y": "mV"},
                fontsize=10,
                weight="normal",
                font="Arial",
            )
        P.axdict["B"].plot(
            self.SP.analysis_summary["FI_Curve"][0] * 1e9,
            self.SP.analysis_summary["FI_Curve"][1] / (self.AR.tend - self.AR.tstart),
            "grey",
            linestyle="-",
            # markersize=4,
            linewidth=0.5,
        )

        P.axdict["B"].scatter(
            self.SP.analysis_summary["FI_Curve"][0] * 1e9,
            self.SP.analysis_summary["FI_Curve"][1] / (self.AR.tend - self.AR.tstart),
            c=trace_colors,
            s=16,
            linewidth=0.5,
        )

        clist = ["r", "b", "g", "c", "m"]  # only 5 possiblities
        linestyle = ["-", "--", "-.", "-", "--"]
        n = 0
        for i, figrowth in enumerate(self.SP.analysis_summary["FI_Growth"]):
            legstr = "{0:s}\n".format(figrowth["FunctionName"])
            if len(figrowth["parameters"]) == 0:  # no valid fit
                P.axdict["B"].plot(
                    [np.nan, np.nan], [np.nan, np.nan], label="No valid fit"
                )
            else:
                for j, fna in enumerate(figrowth["names"][0]):
                    legstr += "{0:s}: {1:.3f} ".format(
                        fna, figrowth["parameters"][0][j]
                    )
                    if j in [2, 5, 8]:
                        legstr += "\n"
                P.axdict["B"].plot(
                    figrowth["fit"][0][0] * 1e9,
                    figrowth["fit"][1][0],
                    linestyle=linestyle[i % len(linestyle)],
                    color=clist[i % len(clist)],
                    linewidth=0.5,
                    label=legstr,
                )
            n += 1
        if n > 0:
            P.axdict["B"].legend(fontsize=6)

        P.axdict["C"].plot(
            np.array(self.RM.ivss_cmd) * 1e9,
            np.array(self.RM.ivss_v) * 1e3,
            "grey",
            # markersize=4,
            linewidth=1.0,
        )
        P.axdict["C"].scatter(
            np.array(self.RM.ivss_cmd) * 1e9,
            np.array(self.RM.ivss_v) * 1e3,
            c=trace_colors[0 : len(self.RM.ivss_cmd)],
            s=16,
        )
        if not pubmode:
            if isinstance(self.RM.analysis_summary["CCComp"], float):
                enable = "Off"
                cccomp = 0.0
                ccbridge = 0.0
            elif self.RM.analysis_summary["CCComp"]["CCBridgeEnable"] == 1:
                enable = "On"
                cccomp = np.mean(
                    self.RM.analysis_summary["CCComp"]["CCPipetteOffset"] * 1e3
                )
                ccbridge = (
                    np.mean(self.RM.analysis_summary["CCComp"]["CCBridgeResistance"])
                    / 1e6
                )
            else:
                enable = "Off"
                cccomp = 0.0
                ccbridge = 0.0
            taum = r"$\tau_m$"
            tauh = r"$\tau_h$"
            tstr = f"RMP: {self.RM.analysis_summary['RMP']:.1f} mV\n"
            tstr += (
                f"${{R_{{in}}}}$: {self.RM.analysis_summary['Rin']:.1f} ${{M\Omega}}$\n"
            )

            tstr += (f"${{Pip Cap}}$: {self.RM.analysis_summary['CCComp']['CCNeutralizationCap']*1e12:.2f} pF\n")
            tstr += f"{taum:s}: {self.RM.analysis_summary['taum']*1e3:.2f} ms\n"
            tstr += f"{tauh:s}: {self.RM.analysis_summary['tauh_tau']*1e3:.3f} ms\n"
            tstr += f"${{G_h}}$: {self.RM.analysis_summary['tauh_Gh'] *1e9:.3f} nS\n"
            tstr += (
                f"Holding: {np.mean(self.RM.analysis_summary['Irmp']) * 1e12:.1f} pA\n"
            )
            tstr += f"Bridge [{enable:3s}]: {ccbridge:.1f} ${{M\Omega}}$\n"
            tstr += f"Bridge Adjust: {self.RM.analysis_summary['BridgeAdjust']:.1f} ${{M\Omega}}$\n"
            tstr += f"Pipette: {cccomp:.1f} mV\n"

            P.axdict["C"].text(
                -0.05,
                0.80,
                tstr,
                transform=P.axdict["C"].transAxes,
                horizontalalignment="left",
                verticalalignment="top",
                fontsize=7,
            )
        #   P.axdict['C'].xyzero=([0., -0.060])
        PH.talbotTicks(
            P.axdict["A"], tickPlacesAdd={"x": 0, "y": 0}, floatAdd={"x": 0, "y": 0}
        )
        P.axdict["A"].set_xlabel("T (ms)")
        P.axdict["A"].set_ylabel("V (mV)")
        P.axdict["A1"].set_xlabel("T (ms)")
        P.axdict["A1"].set_ylabel("I (nV)")
        P.axdict["B"].set_xlabel("I (nA)")
        P.axdict["B"].set_ylabel("Spikes/s")
        PH.talbotTicks(
            P.axdict["B"], tickPlacesAdd={"x": 1, "y": 0}, floatAdd={"x": 2, "y": 0}
        )
        try:
            maxv = np.max(self.RM.ivss_v * 1e3)
        except:
            maxv = 0.0  # sometimes IVs do not have negative voltages for an IVss to be available...
        ycross = np.around(maxv / 5.0, decimals=0) * 5.0
        if ycross > maxv:
            ycross = maxv
        PH.crossAxes(P.axdict["C"], xyzero=(0.0, ycross))
        PH.talbotTicks(
            P.axdict["C"], tickPlacesAdd={"x": 1, "y": 0}, floatAdd={"x": 2, "y": 0}
        )
        P.axdict["C"].set_xlabel("I (nA)")
        P.axdict["C"].set_ylabel("V (mV)")

        """
        Plot the spike intervals as a function of time
        into the stimulus

        """
        for i in range(len(self.SP.spikes)):
            if len(self.SP.spikes[i]) == 0:
                continue
            spx = np.argwhere(
                (self.SP.spikes[i] > self.SP.Clamps.tstart)
                & (self.SP.spikes[i] <= self.SP.Clamps.tend)
            ).ravel()
            spkl = (
                np.array(self.SP.spikes[i][spx]) - self.SP.Clamps.tstart
            ) * 1e3  # just shorten...
            if len(spkl) == 1:
                P.axdict["D"].plot(
                    spkl[0], spkl[0], "o", color=trace_colors[i], markersize=4
                )
            else:
                P.axdict["D"].plot(
                    spkl[:-1],
                    np.diff(spkl),
                    "o-",
                    color=trace_colors[i],
                    markersize=3,
                    linewidth=0.5,
                )

        PH.talbotTicks(
            P.axdict["C"], tickPlacesAdd={"x": 1, "y": 0}, floatAdd={"x": 1, "y": 0}
        )
        P.axdict["D"].set_yscale("log")
        P.axdict["D"].set_ylim((1.0, P.axdict["D"].get_ylim()[1]))
        P.axdict["D"].set_xlabel("Latency (ms)")
        P.axdict["D"].set_ylabel("ISI (ms)")
        P.axdict["D"].text(
            1.00,
            0.05,
            "Adapt Ratio: {0:.3f}".format(self.SP.analysis_summary["AdaptRatio"]),
            fontsize=9,
            transform=P.axdict["D"].transAxes,
            horizontalalignment="right",
            verticalalignment="bottom",
        )

        # phase plot
        # print(self.SP.spikeShapes.keys())
        import matplotlib.pyplot as mpl

        # P.axdict["E"].set_prop_cycle('color',[mpl.cm.jet(i) for i in np.linspace(0, 1, len(self.SP.spikeShapes.keys()))])
        for k, i in enumerate(self.SP.spikeShapes.keys()):
            # print("ss i: ", i, self.SP.spikeShapes[i][0])
            P.axdict["E"].plot(
                self.SP.spikeShapes[i][0].V,
                self.SP.spikeShapes[i][0].dvdt,
                linewidth=0.35,
                color=trace_colors[i],
            )
            if i == 0:
                break # only plot the first one
        P.axdict["E"].set_xlabel("V (mV)")
        P.axdict["E"].set_ylabel("dV/dt (mv/ms)")

        self.IVFigure = P.figure_handle

        if self.plot:
            if self.pdf_pages is None:
                import matplotlib.pyplot as mpl

                # mpl.show()
            else:
                self.pdf_pages.savefig(dpi=300)
                import matplotlib.pyplot as mpl

                mpl.close()
        return P.figure_handle

    def plot_fig(self, pubmode=True):
        if not self.plot:
            return
        x = -0.08
        y = 1.02
        sizer = {
            "A": {"pos": [0.08, 0.82, 0.23, 0.7], "labelpos": (x, y), "noaxes": False},
            "A1": {"pos": [0.08, 0.82, 0.08, 0.1], "labelpos": (x, y), "noaxes": False},
        }
        # dict pos elements are [left, width, bottom, height] for the axes in the plot.
        gr = [
            (a, a + 1, 0, 1) for a in range(0, len(sizer))
        ]  # just generate subplots - shape does not matter
        axmap = OrderedDict(zip(sizer.keys(), gr))
        P = PH.Plotter((len(sizer), 1), axmap=axmap, label=True, figsize=(7.0, 5.0))
        # PH.show_figure_grid(P.figure_handle)
        P.resize(sizer)  # perform positioning magic
        infostr = BIS.build_info_string(self.AR, self.AR.protocol)
        P.figure_handle.suptitle(
            f"{str(self.datapath):s}\n{infostr:s}",
            fontsize=8,
        )
        dv = 0.0
        jsp = 0
        for i in range(self.AR.traces.shape[0]):
            if self.plotting_alternation > 1:
                if i % self.plotting_alternation != 0:
                    continue
            if i in list(self.SP.spikeShapes.keys()):
                idv = float(jsp) * dv
                jsp += 1
            else:
                idv = 0.0
            P.axdict["A"].plot(
                self.AR.time_base * 1e3,
                idv + self.AR.traces[i, :].view(np.ndarray) * 1e3,
                "-",
                linewidth=0.35,
            )
            P.axdict["A1"].plot(
                self.AR.time_base * 1e3,
                self.AR.cmd_wave[i, :].view(np.ndarray) * 1e9,
                "-",
                linewidth=0.35,
            )
        for k in self.RM.taum_fitted.keys():
            P.axdict["A"].plot(
                self.RM.taum_fitted[k][0] * 1e3,
                self.RM.taum_fitted[k][1] * 1e3,
                "-g",
                linewidth=1.0,
            )
        for k in self.RM.tauh_fitted.keys():
            P.axdict["A"].plot(
                self.RM.tauh_fitted[k][0] * 1e3,
                self.RM.tauh_fitted[k][1] * 1e3,
                "--r",
                linewidth=0.750,
            )
        if pubmode:
            PH.calbar(
                P.axdict["A"],
                calbar=[0.0, -50.0, 50.0, 50.0],
                axesoff=True,
                orient="left",
                unitNames={"x": "ms", "y": "mV"},
                fontsize=10,
                weight="normal",
                font="Arial",
            )
            PH.calbar(
                P.axdict["A1"],
                calbar=[0.0, 0.05, 50.0, 0.1],
                axesoff=True,
                orient="left",
                unitNames={"x": "ms", "y": "nA"},
                fontsize=10,
                weight="normal",
                font="Arial",
            )
        self.IVFigure = P.figure_handle

        if self.plot:
            import matplotlib.pyplot as mpl

            mpl.show()
        return P.figure_handle
