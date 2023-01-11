import datetime
import re
from pathlib import Path
from typing import Union

import dill
import matplotlib
import matplotlib.pyplot as mpl  # import locally to avoid parallel problems
import numpy as np
import pylibrary.plotting.plothelpers as PH
import pylibrary.tools.cprint as CP
import pyqtgraph as pg
import pyqtgraph.console as console
import pyqtgraph.multiprocess as mp
from matplotlib.backends.backend_pdf import PdfPages
from pylibrary.tools import cprint as CP
from PyPDF2 import PdfFileMerger, PdfFileReader, PdfFileWriter

from ephys.ephys_analysis.IV_Analysis import IV_Analysis
import ephys.mapanalysistools as mapanalysistools

PMD = mapanalysistools.plotMapData.PlotMapData()

class MAP_Analysis(IV_Analysis):
    def __init__(self, args):
        super().__init__(args)
        print(self._testing_counter)

    def analyze_maps(self, iday:int, file:Union[Path, str], celltype:str, allprots:dict, pdf=None):
        if len(allprots["maps"]) == 0:
            print(f"No maps to analyze for {iday:d}")
            return
        print(f"Analzying map: {str(file):s}")
        datestr, slicestr, cellstr = self.make_cell(iday)
        slicecellstr = f"S{slicestr[-1]:s}C{cellstr[-1]:s}"
        self.celltype, self.celltype_changed = self.get_celltype(iday)
        if self.dry_run:
            print(
                "Would process day: {0:s} slice: {1:s} cell: {2:s}".format(
                    datestr, slicestr, cellstr
                )
            )

            print("       Celltype: {0:s}".format(celltype))
            print("   with {0:4d} protocols".format(len(allprots["maps"])))
            for i, p in enumerate(allprots["maps"]):
                print("      {0:d}. {1:s}".format(i + 1, str(p.name)))
            return

        validmaps = []
        for p in allprots["maps"]:  # first remove excluded protocols
            if self.exclusions is None or str(p) not in self.exclusions:
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
        picklefilename = Path(self.analyzeddatapath, "events", foname)
        # CP.cprint("m", f"Pickle filename: {str(picklefilename):s}")


        ###
        ### Parallel is done at lowest level of analyzing a trace, not at this top level
        ### can only have ONE parallel loop going (no nested ones allowed!)
        ###
        # if self.noparallel:  # just serial...
        for i, x in enumerate(tasks):
            result = self.analyze_map(
                iday,
                file=file,
                i=i,
                x=x,
                allprots=allprots,
                plotmap=plotmap,
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
        if self.recalculate_events:  # save the recalculated events to the events file
            CP.cprint("g", f"Events written to :  {str(picklefilename):s}")
            with open(picklefilename, "wb") as fh:
                dill.dump(results, fh)

        if self.celltype_changed:
            CP.cprint("yellow", f"cell annotated celltype: {self.celltype:s})")
        else:
            txt = self.celltype.strip()
            print("celltype: ", self.celltype)
            if len(txt) == 0 or txt == " " or txt is None:
                CP.cprint("magenta", f"Database celltype: Not specified")
            else:
                CP.cprint("g", f"Database celltype: {txt:s}")


        self.merge_pdfs(celltype=celltype, slicecell=slicecellstr)

    def set_vc_taus(self, iday: int, path: Union[Path, str]):
        """ """
        datestr, slicestr, cellstr = self.make_cell(iday)
        # print(self.map_annotationFilename)
        # print(self.map_annotations)
        if self.map_annotationFilename is not None:
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
            sh1 = cell_df["tau1"].shape
            sh2 = cell_df["fl_tau1"].shape
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
                CP.cprint("r", "Using default VC taus for detection - likely no entry in excel file")
                # exit()
        CP.cprint(
            "w", f"    [{self.AM.Pars.taus[0]:8.4f}, {self.AM.Pars.taus[1]:8.4f}]"
        )

    def set_cc_taus(self, iday: int, path: Union[Path, str]):
        datestr, slicestr, cellstr = self.make_cell(iday)
        if self.map_annotationFilename is not None:
            cell_df = self.find_cell(
                self.map_annotations,
                datestr,
                slicestr,
                cellstr,
                path,
            )
            sh = cell_df["cc_tau1"].shape
            if cell_df is not None and sh != (0,):
                self.AM.Pars.taus = [
                    cell_df["cc_tau1"].values[0] * 1e-3,
                    cell_df["cc_tau2"].values[0] * 1e-3,
                ]
                print("    Setting CC taus from map annotation: ", end="")
            else:
                print("    Using default taus")
        CP.cprint(
            "w", f"    [{self.AM.Pars.taus[0]:8.4f}, {self.AM.Pars.taus[1]:8.4f}]"
        )

    def set_vc_threshold(self, iday: int, path: Union[Path, str]):
        datestr, slicestr, cellstr = self.make_cell(iday)
        if self.map_annotationFilename is not None:
            cell_df = self.find_cell(
                self.map_annotations, datestr, slicestr, cellstr, path
            )
            sh = cell_df["alt1_threshold"].shape
            if cell_df is not None and sh != (0,):
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
        if self.map_annotationFilename is not None:
            cell_df = self.find_cell(
                self.map_annotations, datestr, slicestr, cellstr, path
            )
            sh = cell_df["cc_threshold"].shape
            if cell_df is not None and sh != (0,):
                self.AM.Pars.threshold = cell_df["cc_threshold"].values[0]
                print("set cc_threshold from map annotation")
            else:
                print("using default cc threshold")

    def set_stimdur(self, iday: int, path: Union[Path, str]):
        datestr, slicestr, cellstr = self.make_cell(iday)
        if self.map_annotationFilename is not None:
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
        protodir = Path(self.rawdatapath, path)
        try:
            assert protodir.is_dir()
            protocol = self.AM.AR.readDirIndex(str(protodir))
            record_mode = protocol["."]["devices"]["MultiClamp1"]["mode"]
        except:
            if path.match("*_VC_*"):
                record_mode = "VC"
            elif path.match("*_IC_*"):
                record_mode = "IC"
            else:
                raise ValueError("Cant figure record mode")

        self.set_stimdur(iday, path)

        if (path.match("*_VC_*") or record_mode == "VC") and not self.rawdatapath.match(
            "*VGAT_*"
        ):  # excitatory PSC
            self.AM.datatype = "V"
            self.AM.Pars.sign = -1
            self.AM.Pars.scale_factor = 1e12
            self.AM.Pars.taus = [1e-3, 3e-3]  # fast events
            self.set_vc_taus(iday, path)
            self.AM.Pars.threshold = self.threshold  # threshold...
            self.set_vc_threshold(iday, path)

        elif (path.match("*_VC_*") or record_mode == "VC") and self.rawdatapath.match(
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
            ) and not self.rawdatapath.match(
                "*VGAT_*"
            ):  # excitatory PSP
            self.AM.Pars.sign = 1  # positive going
            self.AM.Pars.scale_factor = 1e3
            self.AM.Pars.taus = [1e-3, 4e-3]  # fast events
            self.AM.datatype = "I"
            self.AM.Pars.threshold = self.threshold  # somewhat high threshold...
            self.set_cc_taus(iday, path)
            self.set_cc_threshold(iday, path)

        elif path.match("*_IC_*") and self.rawdatapath.match(
                "*VGAT_*"
            ):  # inhibitory PSP
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
        x: object,
        file: Union[Path, str],
        allprots: dict,
        plotmap: bool=False,
        measuretype: str = "ZScore",
        verbose: bool = False,
        picklefilename: Union[Path, str, None] = None,
    ) -> Union[None, dict]:
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
        CP.cprint("g", "\nEntering MAP_Analysis:analyze_map")

        mapname = allprots["maps"][i]
        if len(mapname) == 0:
            return None

        mapdir = Path(file, Path(mapname).name)
        self.mapname = mapname
        if "_IC__" in str(mapdir.name) or "CC" in str(mapdir.name):
            scf = 1e3  # mV
        else:
            scf = 1e12  # pA, vc
        # plot the Z score, Charge and amplitude maps:
        if self.mapsZQA_plot:
            CP.cprint(
                "g",
                f"MAP_Analysis:analyze_map  Replotting from .pkl file: {str(picklefilename):s}",
            )
            CP.cprint(
                "g",
                f"    Protocol: {mapname:s}",
            )
            with open(
                picklefilename, "rb"
            ) as fh:  # read the previously analyzed data set
                results = dill.load(fh)
            mapkey = Path("/".join(Path(mapname).parts[-4:]))
            if str(mapkey) not in results.keys():
                # try prepending path to data
                mapkey = Path(self.rawdatapath, mapkey)
                if str(mapkey) not in results.keys():
                    CP.cprint("r", "**** Map key missing from result dictionary: ")
                    CP.cprint("r", f"     {str(mapkey):s}")
                    CP.cprint("r", f"     Known keys:")
                    for k in results.keys():
                        CP.cprint("r", f"     {str(k):s}")
                    return

            result = results[str(mapkey)]  # get individual map result

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

        self.AM.set_analysis_window(0.0, 0.599)
        self.AM.set_artifact_suppression(self.artifact_suppress)
        self.AM.set_noderivative_artifact(self.noderivative_artifact)
        if self.artifactFilename is not None:
            self.AM.set_artifact_file(
                Path(
                    self.analyzeddatapath,
                    self.artifactFilename,
                )
            )
        self.AM.set_taus(self.AM.Pars.taus)  # [1, 3.5]

        if self.recalculate_events:
            CP.cprint(
                "g", f"IV_Analysis:analyze_map  Running map analysis: {str(mapname):s}"
            )
            result = self.AM.analyze_one_map(
                mapdir, noparallel=self.noparallel, verbose=verbose
            )

            if result is not None:
                # result['onsets'] = [result['events'][i]['onsets'] for i in result['events']]
                result["analysisdatetime"] = datetime.datetime.now()  # add timestamp
            if result is None:
                return
        else:
            pass  # already got the file

        # CP.cprint("c", f"Plotmap: {str(plotmap):s}")
        if plotmap:
            if self.celltype_changed:
                celltype_text = f"{self.celltype:s} [revised]"
            else:
                celltype_text = f"{self.celltype:s} [orig]"
            getimage = False
            plotevents = True
            self.AM.Pars.overlay_scale = 0.0
            PMD.set_Pars_and_Data(self.AM.Pars, self.AM.Data)
            if self.mapsZQA_plot:
                mapok = PMD.display_position_maps(
                    dataset_name=mapdir, result=result, pars=self.AM.Pars
                )
            else:
                if mapdir != self.AM.last_dataset:
                    results = self.analyze_one_map(self.AM.last_dataset)
                else:
                    results = self.AM.last_results
                mapok = PMD.display_one_map(
                    mapdir,
                    results=results,  # self.AM.last_results,
                    imagefile=None,
                    rotation=0.0,
                    measuretype=measuretype,
                    plotevents=plotevents,
                    whichstim=self.whichstim,
                    trsel=self.trsel,
                    plotmode=self.plotmode,
                    average=False,
                    rasterized=False,
                    datatype=self.AM.datatype,
                )  # self.AM.rasterized, firstonly=True, average=False)
            print(f"Map analysis done: {str(mapname):s}")

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
                t_path = Path(
                    self.tempdir, "temppdf_{0:s}.pdf".format(str(mapdir.name))
                )
                if not self.tempdir.is_dir():
                    print("temp dir not found: ", self.tempdir)
                    exit()

                if t_path.is_file():
                    t_path.unlink()
                pp = PdfPages(t_path)
                # try:
                print("        ***** Temp file to : ", t_path)
                mpl.savefig(
                    pp, format="pdf"
                )  # use the map filename, as we will sort by this later
                pp.close()
                # except ValueError:
                #       print('Error in saving map %s, file %s' % (t_path, str(mapdir)))
                mpl.close(PMD.P.figure_handle)
        return result