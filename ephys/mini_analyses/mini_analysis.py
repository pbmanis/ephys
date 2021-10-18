import argparse
import os
import pickle
import sys
from pathlib import Path
from collections import OrderedDict
from typing import Union

import numpy as np
import scipy.signal
import scipy.stats
from matplotlib import collections as collections
from matplotlib import pyplot as mpl
from matplotlib import rc
from matplotlib.backends.backend_pdf import PdfPages
from pylibrary.plotting import plothelpers as PH
from pylibrary.tools import cprint
CP = cprint.cprint

from .. import ephysanalysis as EP
from ..tools import digital_filters as DF
from . import minis_methods as minis

"""
Analysis of miniature synaptic potentials
Provides measure of event amplitude distribution and event frequency distribution
The analysis is driven by an imported dictionary.

Example for the data table:
self.basedatadir = '/Volumes/Pegasus/ManisLab_Data3/Sullivan_Chelsea/miniIPSCs'

dataplan_params = {'m1a': {'dir': '2017.04.25_000/slice_000/cell_001', 'prots': [0,1,3],
                'thr': 1.75, 'rt': 0.35, 'decay': 6., 'G': 'F/+', 'exclist': []},
            'm1b': {'dir': '2017.04.25_000/slice_000/cell_002', 'prots': [7],
                'thr': 1.75, 'rt': 0.35, 'decay': 6., 'G': 'F/+', 'exclist': []},
            'm2a': {'dir': '2017.05.02_000/slice_000/cell_000/', 'prots': [0,1,2],
                'thr': 1.75, 'rt': 0.32, 'decay': 5., 'G': 'F/+', 'exclist': []},
            'm2b': {'dir': '2017.05.02_000/slice_000/cell_001', 'prots': [0,1,2],
                'thr': 1.75, 'rt': 0.35, 'decay': 4., 'G': 'F/+', 'exclist': {1: [4, 5, 6], 2: [8]}},
            'm2c': {'dir': '2017.05.02_000/slice_000/cell_002', 'prots': [0,1,2],
        }

Where:
each dict key indicates a cell from a mouse (mice are numbered, cells are lettered)
'dir': The main cell directory, relative to the base directory,
'prots': a list of the protocols to be analyzed,
'exclist': a dict of the protocols that have traces to be excluded
    The excluded traces are in a tuple or list for each protocol.
    For example, exclist: {0 : [1,2], 1: [3,4,5]} results in the exclusion of
        traces 1 and 2 from protocol run 0, and traces 3, 4 and 5 from protocol run 1
'thr' : SD threshold for event detection (algorithm dependent)
'rt' : rise time for template (in msec)
'decay': decay time constant for the template (in msec)
'G' : group identifier (e.g, genotype, treatment, etc.)


The data table can be generated on the fly from another table structure, such as an excel
sheet read by panas, etc.

Requires asome of Manis' support libraries/modules, including:
    ephysanalysis module  (git clone https://github/pbmanis/ephysanalysis)
    pylibrary utilities, (git clone https://github/pbmanis/ephysanalysis)
    Output summary is a Python pickle file (.p extension) that isread by mini_summary_plots.py

Paul B. Manis, Ph.D. Jan-March 2018.
Revised 2/2021 to work with new mini_methods output formats.

"""


rc("text", usetex=False)
# rc('font',**{'family':'sans-serif','sans-serif':['Verdana']})


class MiniAnalysis:
    def __init__(self, dataplan):
        """
        Perform detection of miniature synaptic events, and some analysis
                   Parameters
        ----------
        dataplan : object
            a dataplan object, with
                datasource: the name of the file holding the dict
                datadir : the path to the data itself
                dataplan_params : the dict with information driving the analysis
        """

        self.datasource = dataplan.datasource
        self.basedatadir = dataplan.datadir
        self.shortdir = dataplan.shortdir
        self.dataset = dataplan.dataset
        self.outputpath = dataplan.outputpath
        self.dataplan_params = dataplan.dataplan_params
        self.dataplan_data = dataplan.data
        self.min_time = dataplan.min_time
        self.max_time = dataplan.max_time
        self.clamp_name = "Clamp1.ma"
        self.protocol_name = "minis"
        print('mini_analysis dataplan parameters: ', self.dataplan_params)
        try:
            self.global_threshold = dataplan.data["global_threshold"]
            self.override_threshold = True
        except KeyError:
            self.override_threshold = False
        self.global_decay = None
        if "global_decay" in dataplan.data.keys():
            self.global_decay = dataplan.data["global_decay"]
            self.override_decay = True
        else:
            self.override_decay = False
        print('override decay: ', self.override_decay, self.override_threshold)

        self.filter = False
        self.filterstring = "no_notch_filter"
        try:
            self.filter = dataplan.data["notch_filter"]
            if self.filter:
                self.filterstring = "notch_filtered"
                CP("r", "*** NOTCH FILTER ENABLED ***")
        except KeyError:
            raise ValueError("No Notch filter specified")
            # self.filter = False
            # self.filterstring = "no_notch_filter"
        # moved this to mousedata (e.g., per mouse).
        # defined in ank2_datasets.py, in MINID dataclass instead of in main 
        # dataplan
        # try:
        #     CP('r', dataplan.dataplan_params)
        #     print(dataplan.dataplan_params.keys())
        #     mouse = dataplan
        #     self.min_event_amplitude = dataplan[mouse].dataplan_params["min_event_amplitude"]
        # except KeyError:
        #     raise ValueError("Data plan has no minimum event amplitude")
        #     # self.min_event_amplitude = 2.0e-12

    def set_clamp_name(self, name):
        self.clamp_name = name

    def set_protocol_name(self, name):
        self.protocol_name = name

    # from acq4 functions:
    def measure_baseline(self, data, threshold=2.0, iterations=2):
        """Find the baseline value of a signal by iteratively measuring the median value, then excluding outliers."""
        data = data.view(np.ndarray)
        med = np.median(data)
        if iterations > 1:
            std = data.std()
            thresh = std * threshold
            arr = np.ma.masked_outside(data, med - thresh, med + thresh)
            try:
                len(arr.mask)  # could be an array or just a boolean
            except TypeError:
                if arr.mask is False:  # nothing to mask...
                    return med
            if len(arr) == 0:
                raise Exception(
                    "Masked out all data. min: %f, max: %f, std: %f"
                    % (med - thresh, med + thresh, std)
                )
            return self.measure_baseline(arr[~arr.mask], threshold, iterations - 1)
        else:
            return med

    def analyze_all(self, fofilename, check=False, mode="aj", engine="cython"):
        """
        Wraps analysis of individual data sets, writes plots to
        a file named "summarydata%s.p % self.datasource" in pickled format.
               Parameters
        ----------
        fofilename : str (no default)
            name of the PDF plot output file
        check : bool (default: False)
            If true, run just checks for the existence of the data files,
        but does no analysis.
               Returns
        -------
        Nothing
        """
        print('analyze_all')
        acqr = EP.acq4read.Acq4Read(
            dataname=self.clamp_name
        )  # creates a new instance every time - probably should just create one.
        summarydata = {}
        with PdfPages(fofilename) as pdf:
            for index, mouse in enumerate(sorted(self.dataplan_params.keys())):
                self.analyze_one_cell(
                    mouse,
                    pdf,
                    maxprot=10,
                    arreader=acqr,
                    check=check,
                    mode=mode,
                    engine=engine,
                )
                summarydata[mouse] = self.cell_summary
                # if index >= 0:
                #     break
        if not check:
            print("output file: ", self.shortdir, self.dataset, self.filterstring, mode)
            ofile = Path(self.outputpath, f"{self.dataset:s}_{str(self.filterstring):s}_{mode:s}.p")
            fout = str(ofile)
            print("outfile: ", ofile)
            fh = open(fout, "wb")
            pickle.dump(summarydata, fh)
            fh.close()
        else:
            print("All files found (an exception would be raised if one was not found)")

    def build_summary_dict(self, genotype:Union[str, None] = None, eyfp:str = "ND", mouse:Union[str, int, None]=None):
        self.cell_summary = {
            "intervals": [],
            "amplitudes": [],
            "protocols": [],
            "eventcounts": [],
            "genotype": genotype,
            "EYFP": eyfp,
            "mouse": mouse,
            "amplitude_midpoint": 0.0,
            "holding": [],
            "averaged": [],
            "sign": [],
            "threshold": [],
            "indiv_evok": [],
            "indiv_notok": [],
            "indiv_amp": [],
            "indiv_fitamp": [],
            "indiv_tau1": [],
            "indiv_tau2": [],
            "indiv_fiterr": [],
            "indiv_Qtotal": [],
            "indiv_tb": [],
            "allevents": [],
            "fitted_events": [],
            "best_fit": [],
            "best_decay_fit": [],
        }


    def analyze_one_cell(
        self,
        mouse,
        pdf,
        maxprot=10,
        arreader=None,
        check=False,
        mode="aj",
        engine="cython",
    ):
        """
        Provide analysis of one entry in the data table using the Andrade_Jonas algorithm
        and fitting distributions.
        Generates a page with plots for all protocols and traces stacked, and 
        a second page of histograms with
        fits for the interval and amplitude distributions
        
        Parameters
        ----------
        mouse : str (no default)
            key into the dictionary for the data to be analyzed
        pdf : pdf file object (no default):
            the pdf file object to write the plots to.
        maxprot : int (default: 10)
            Maximum numober of protocols to do in the analysis
        check : bool (default: False)
            If true, run just checks for the existence of the data files,
        but does no analysis.
        
        Returns
        -------
            cell summary dictionary for the 'mouse' entry.
        """
        print('analyze one cell')
        if arreader is None:
            acqr = EP.acq4read.Acq4Read(
                dataname=self.clamp_name
            )  # only if we don't already have one
        else:
            acqr = arreader
        self.rasterize = (
            True  # set True to rasterize traces to reduce size of final document
        ) # set false for pub-quality output (but large size)
        self.acqr = acqr
        dt = 0.1
        mousedata = self.dataplan_params[mouse]

        if self.override_threshold:
            mousedata["thr"] = self.global_threshold  # override the threshold setting
        if self.override_decay:
            mousedata["decay"] = self.global_decay  # override decay settings

        self.sign = 1
        if "sign" in self.dataplan_data:
            self.sign = int(self.dataplan_data["sign"])

        print("\nMouse: ", mouse)

        self.build_summary_dict(genotype=mousedata["G"], eyfp=mousedata['EYFP'], mouse=mouse)

        if not check:
            self.plot_setup()
        datanameposted = False
        self.yspan = 40.0
        self.ypqspan = 2000.0
        ntr = 0
        # for all of the protocols that are included for this cell (identified by number and maybe letters)
        CP("g", f"    mousedata: {str(mousedata):s}")
        print("    mousedata prots: ", mousedata["prots"])
        if len(mousedata["prots"]) == 0:
            print("  No protocols, moving on")
            return None
        
        for nprot, dprot in enumerate(mousedata["prots"]):
            if nprot > maxprot:
                return
            self.nprot = nprot
            self.dprot = dprot
            exclude_traces = []
            print('    exclusion list: ', mousedata["exclist"])
            if dprot in mousedata["exclist"].keys():
                # print (mousedata['exclist'], dprot, nprot)
                exclude_traces = mousedata["exclist"][dprot]
            # print('dprot: ', dprot)
            # print('exc: ', exclude_traces)
            sign = self.dataplan_data["sign"]

            fn = Path(self.basedatadir, mousedata["dir"])
            fx = fn.name
            ext = fn.suffix
            print("    sign: ", sign)
            fn = Path(fn, f"{mousedata['protocol_name']:s}_{dprot:03d}")
            print("    Protocol file: ", fn)
            split = fn.parts[-4:-1]
            # dataname = ""
            # for i in range(len(split)):
            #     dataname = Path(dataname, split[i])
            # print('dataname: ', fn)
            acqr.setProtocol(fn)
            # print(check)
            if not check:
                print("  Protocol dataname: ", fn)
                if len(exclude_traces) > 0:
                    CP("c", f"  Excluding traces: {str(exclude_traces):s}")
                else:
                    print("  No traces excluded")
            else:
                result = acqr.getData(check=True)
                if result is False:
                    CP('r', f"******* Get data failed to find a file : {str(fn):s}")
                    CP('r', f"        dataname: {fn:s}")
                continue
            acqr.getData()
            oktraces = [x for x in range(acqr.data_array.shape[0]) if x not in exclude_traces]
            data = np.array(acqr.data_array[oktraces])
            dt_seconds = acqr.sample_interval
            min_index = int(self.min_time / dt_seconds)
            if self.max_time > 0.0:
                max_index = int(self.max_time / dt_seconds)
            else:
                max_index = data.shape[1]
            data = data[:, min_index:max_index]
            time_base = acqr.time_base[min_index:max_index]
            time_base = time_base - self.min_time
            if not datanameposted and not check:
                self.P.figure_handle.suptitle(f"{mouse:s}\n{str(mousedata):s}\n{self.cell_summary['genotype']:s}",
                    fontsize=8,
                    weight="normal",
                )
                datanameposted = True
            # data = data * 1e12  # convert to pA

            maxt = np.max(time_base)

            tracelist, nused = self.analyze_protocol_traces(
                mode=mode, data=data, time_base=time_base,
                maxt=maxt, dt_seconds=dt_seconds, 
                mousedata=mousedata, ntr=ntr
            )
            ntr = ntr + len(tracelist)  # - len(exclude_traces)
            if nused == 0:
                continue

        if check:
            return None  # no processing

        # summarize the event and amplitude distributions
        # For the amplitude, the data are fit against normal, 
        # skewed normal and gamma distributions.
        self.plot_hists()
        # show to sceen or save plots to a file
        if pdf is None:
            mpl.show()
        else:
            pdf.savefig(dpi=300)  # rasterized to 300 dpi is ok for documentation.
            mpl.close()
        self.plot_individual_events(
            fit_err_limit=50.0,
            title=f"{str(mousedata):s} {self.cell_summary['mouse']:s} {self.cell_summary['genotype']:s}",
            pdf=pdf,
        )

    def analyze_protocol_traces(
        self, mode:str='cb', data:Union[object, None]=None, 
            time_base:Union[np.ndarray, None]=None, 
            maxt:float=0., dt_seconds:Union[float, None] = None,
            mousedata:Union[dict, None] = None, 
            ntr:int=0
        ):
        """
        perform mini analyzis on all the traces in one protocol
        
        mode: str
            which event detector to use ()
        """
        assert data is not None
        assert time_base is not None
        assert dt_seconds is not None
        order = int(1e-3 / dt_seconds)
        ntraces = data.shape[0]
        tracelist = list(range(ntraces))
        self.ax0.set_xlim(-1.0, np.max(time_base))
        nused = 0
        # for all of the traces collected in this protocol that 
        # are accepted (not excluded) 

        tasks = []
        for i in tracelist:
            tasks.append(i)
        print("*** Mode: ", mode)
        if mode == "aj":
            aj = minis.AndradeJonas()
            print("calling aj setup: ")
            aj.setup(
                ntraces=ntraces,
                tau1=mousedata["rt"],
                tau2=mousedata["decay"],
                template_tmax=maxt,
                dt_seconds=dt_seconds,
                delay=0.0,
                sign=self.sign,
                risepower=1.0,
                min_event_amplitude=float(mousedata["min_event_amplitude"]), # self.min_event_amplitude,
                threshold=float(mousedata["thr"]),
                lpf=mousedata["lpf"],
                hpf=mousedata["hpf"],
                notch=mousedata["notch"],
                notch_Q=mousedata["notch_Q"]
            )
        elif mode == "cb":
            cb = minis.ClementsBekkers()
            cb.setup(
                ntraces=ntraces,
                tau1=mousedata["rt"],
                tau2=mousedata["decay"],
                template_tmax=3.0 * mousedata["decay"],
                dt_seconds=dt_seconds,
                delay=0.0,
                sign=self.sign,
                risepower=1.0,
                min_event_amplitude=self.min_event_amplitude,
                threshold=float(mousedata["thr"]),
                lpf=mousedata["lpf"],
                hpf=mousedata["hpf"],
                notch=mousedata["notch"],
                notch_Q=mousedata["notch_Q"]
            )
            cb.set_cb_engine("cython")
        else:
            raise ValueError("Mode must be aj or cb for event detection")


        # now detect events...
        if mode == "aj":
            for i in tracelist:
                aj.reset_filtering()
                aj.deconvolve(
                    data[i], itrace=i, llambda=10.0
                )  # , order=order) # threshold=float(mousedata['thr']),
                data[i] = aj.data.copy()
            aj.identify_events(order=order)
            aj.summarize(np.array(data))

            # print(aj.Summary)
            method = aj
        elif mode == "cb":
            for i in tracelist:
                cb.reset_filtering()
                cb.cbTemplateMatch(
                    data[i], itrace=i, # order=order, #  threshold=float(mousedata["thr"]),
                )
                data[i] = cb.data.copy()

            cb.identify_events(outlier_scale=3.0, order=101)
            cb.summarize(np.array(data))
            method = cb

        # print('aj onsets2: ', method.onsets)
        self.cell_summary['averaged'].extend([{'tb': method.Summary.average.avgeventtb,
        'avg': method.Summary.average.avgevent,
            'fit': {'amplitude': method.Amplitude,
            'tau1': method.fitted_tau1,
            'tau2': method.fitted_tau2, 
            'risepower': method.risepower},
            'best_fit': method.avg_best_fit,
            'risetenninety': method.Summary.average.risetenninety, 
            'decaythirtyseven': method.Summary.average.decaythirtyseven,
            'Qtotal': method.Summary.Qtotal}])

        for i in tracelist:
            intervals = np.diff(method.timebase[method.onsets[i]])
            self.cell_summary['intervals'].extend(intervals)
            self.cell_summary['amplitudes'].extend(method.sign*data[i][method.Summary.smpkindex[i]])  # smoothed peak amplitudes
            self.cell_summary['protocols'].append((self.nprot, i))
            holding = self.measure_baseline(data[i])
            self.cell_summary['holding'].append(holding)
        self.cell_summary['eventcounts'].append(len(intervals))
        self.cell_summary['sign'].append(method.sign)
        self.cell_summary['threshold'].append(mousedata['thr'])

        # method.fit_individual_events() # fit_err_limit=2000., tau2_range=2.5)  # on the data just analyzed
        # self.cell_summary['indiv_amp'].append(method.ev_amp)
        # self.cell_summary['indiv_fitamp'].append(method.ev_fitamp)
        # self.cell_summary['indiv_tau1'].append(method.ev_tau1)
        # self.cell_summary['indiv_tau2'].append(method.ev_tau2)
        # self.cell_summary['indiv_fiterr'].append(method.fiterr)
        # self.cell_summary['fitted_events'].append(method.fitted_events)
        # self.cell_summary['indiv_Qtotal'].append(method.ev_Qtotal)
        # self.cell_summary['indiv_evok'].append(method.events_ok)
        # self.cell_summary['indiv_notok'].append(method.events_notok)
        # self.cell_summary['allevents'].append(np.array(method.allevents))
        # self.cell_summary['best_fit'].append(np.array(method.best_fit))
        # self.cell_summary['best_decay_fit'].append(np.array(method.best_decay_fit))
        #
        #
        # for jev in range(len(method.allevents)):
        #    self.cell_summary['allevents'].append(method.Summary.allevents[jev])
        #    self.cell_summary['best_fit'].append(method.best_fit[jev])
        # self.cell_summary['indiv_tb'].append(aj.avgeventtb)
        scf = 1e12
        for i, dat in enumerate(data):  # this copy of data is lpf/hpf/notch filtered as requested
            yp = (ntr + i) * self.yspan
            ypq = (ntr * i) * self.ypqspan
            linefit = np.polyfit(time_base, dat, 1)
            refline = np.polyval(linefit, time_base)
            jtr = method.Summary.event_trace_list[
                i
            ]  # get trace and event number in trace
            if len(jtr) == 0:
                continue
            peaks = method.Summary.smpkindex[i]
            onsets = method.Summary.onsets[i]
            onset_times = np.array(onsets) * method.dt_seconds
            peak_times = np.array(peaks) * method.dt_seconds
            self.ax0.plot(
                method.timebase, scf*(dat-refline) + yp, "k-", linewidth=0.25, rasterized=self.rasterize
            )
            # self.ax0.plot(method.timebase[pkindex], data[i][pkindex] + yp,
            # 'ro', markersize=1.75, rasterized=self.rasterize)
            # self.ax0.plot(aj.timebase[aj.smpkindex], data[i][aj.smpkindex] + yp,
            # 'ro', markersize=1.75, rasterized=self.rasterize)
            self.ax0.plot(
                peak_times,
                scf*(dat[peaks]- refline[peaks]) + yp , # method.Summary.smoothed_peaks[jtr[0]][jtr[1]] + yp,
                "ro",
                markersize=1.,
                rasterized=self.rasterize,
                alpha=0.5,
            )
            # self.ax0.plot(
            #     onset_times,
            #     scf*data[i][onsets] + yp,
            #     "y^",
            #     markersize=1.5,
            #     rasterized=self.rasterize,
            # )

        if "A1" in self.P.axdict.keys():
            self.axdec.plot(
                aj.timebase[: aj.Crit.shape[0]], aj.Crit, label="Deconvolution"
            )
            self.axdec.plot(
                [aj.timebase[0], aj.timebase[-1]],
                [aj.sdthr, aj.sdthr],
                "r--",
                linewidth=0.75,
                label="Threshold ({0:4.2f}) SD".format(aj.sdthr),
            )
            self.axdec.plot(
                aj.timebase[aj.onsets] - aj.idelay,
                ypq + aj.Crit[aj.onsets],
                "y^",
                label="Deconv. Peaks",
            )
        #            axdec.plot(aj.timebase, aj.Crit+ypq, 'k', linewidth=0.5, rasterized=self.rasterize)
        # print("--- finished run %d/%d ---" % (i + 1, tot_runs))

        return tracelist, nused

    def plot_individual_events(
        self, fit_err_limit=1000.0, tau2_range=2.5, title="", pdf=None
    ):
        P = PH.regular_grid(
            3,
            3,
            order="columnsfirst",
            figsize=(8.0, 8.0),
            showgrid=False,
            verticalspacing=0.1,
            horizontalspacing=0.12,
            margins={
                "leftmargin": 0.12,
                "rightmargin": 0.12,
                "topmargin": 0.03,
                "bottommargin": 0.1,
            },
            labelposition=(-0.12, 0.95),
        )
        P.figure_handle.suptitle(title)
        all_evok = self.cell_summary[
            "indiv_evok"
        ]  # this is the list of ok events - a 2d list by
        all_notok = self.cell_summary["indiv_notok"]
        # print('all evok: ', all_evok)
        # print('len allevok: ', len(all_evok))
        #
        # # print('all_notok: ', all_notok)
        # # print('indiv tau1: ', self.cell_summary['indiv_tau1'])
        # exit(1)
        trdat = []
        trfit = []
        trdecfit = []
        for itr in range(len(all_evok)):  # for each trace
            for evok in all_evok[itr]:  # for each ok event in that trace
                P.axdict["A"].plot(
                    self.cell_summary["indiv_tau1"][itr][evok],
                    self.cell_summary["indiv_amp"][itr][evok],
                    "ko",
                    markersize=3,
                )
                P.axdict["B"].plot(
                    self.cell_summary["indiv_tau2"][itr][evok],
                    self.cell_summary["indiv_amp"][itr][evok],
                    "ko",
                    markersize=3,
                )
                P.axdict["C"].plot(
                    self.cell_summary["indiv_tau1"][itr][evok],
                    self.cell_summary["indiv_tau2"][itr][evok],
                    "ko",
                    markersize=3,
                )
                P.axdict["D"].plot(
                    self.cell_summary["indiv_amp"][itr][evok],
                    self.cell_summary["indiv_fiterr"][itr][evok],
                    "ko",
                    markersize=3,
                )
                P.axdict["H"].plot(
                    self.cell_summary["indiv_tau1"][itr][evok],
                    self.cell_summary["indiv_Qtotal"][itr][evok],
                    "ko",
                    markersize=3,
                )
                trdat.append(
                    np.column_stack(
                        [
                            self.cell_summary["indiv_tb"][itr],
                            self.cell_summary["allevents"][itr][evok],
                        ]
                    )
                )
                # idl = len(self.cell_summary['best_decay_fit'][itr][evok])
                trfit.append(
                    np.column_stack(
                        [
                            self.cell_summary["indiv_tb"][itr],
                            -self.cell_summary["best_fit"][itr][evok],
                        ]
                    )
                )
                trdecfit.append(
                    np.column_stack(
                        [
                            self.cell_summary["indiv_tb"][itr],
                            -self.cell_summary["best_decay_fit"][itr][evok],
                        ]
                    )
                )
        dat_coll = collections.LineCollection(trdat, colors="k", linewidths=0.5)
        fit_coll = collections.LineCollection(trfit, colors="r", linewidths=0.25)
        #        decay_fit_coll = collections.LineCollection(trdecfit, colors='c', linewidths=0.3)
        P.axdict["G"].add_collection(dat_coll)
        P.axdict["G"].add_collection(fit_coll)
        #        P.axdict['G'].add_collection(decay_fit_coll)
        n_trdat = []
        n_trfit = []
        for itr in range(len(all_notok)):
            for notok in all_notok[itr]:
                n_trdat.append(
                    np.column_stack(
                        [
                            self.cell_summary["indiv_tb"][itr],
                            self.cell_summary["allevents"][itr][notok],
                        ]
                    )
                )
                n_trfit.append(
                    np.column_stack(
                        [
                            self.cell_summary["indiv_tb"][itr],
                            -self.cell_summary["best_fit"][itr][notok],
                        ]
                    )
                )
                P.axdict["D"].plot(
                    self.cell_summary["indiv_amp"][itr][notok],
                    self.cell_summary["indiv_fiterr"][itr][notok],
                    "ro",
                    markersize=3,
                )
        n_dat_coll = collections.LineCollection(n_trdat, colors="b", linewidths=0.35)
        n_fit_coll = collections.LineCollection(n_trfit, colors="y", linewidths=0.25)
        P.axdict["E"].add_collection(n_dat_coll)
        P.axdict["E"].add_collection(n_fit_coll)

        P.axdict["A"].set_xlabel(r"$tau_1$ (ms)")
        P.axdict["A"].set_ylabel(r"Amp (pA)")
        P.axdict["B"].set_xlabel(r"$tau_2$ (ms)")
        P.axdict["B"].set_ylabel(r"Amp (pA)")
        P.axdict["C"].set_xlabel(r"$\tau_1$ (ms)")
        P.axdict["C"].set_ylabel(r"$\tau_2$ (ms)")
        P.axdict["D"].set_xlabel(r"Amp (pA)")
        P.axdict["D"].set_ylabel(r"Fit Error (cost)")
        P.axdict["H"].set_xlabel(r"$\tau_1$ (ms)")
        P.axdict["H"].set_ylabel(r"Qtotal")
        P.axdict["G"].set_ylim((-100.0, 20.0))
        P.axdict["G"].set_xlim((-2.0, 25.0))
        P.axdict["E"].set_ylim((-100.0, 20.0))
        P.axdict["E"].set_xlim((-2.0, 25.0))

        # put in averaged event too
        # self.cell_summary['averaged'].extend([{'tb': aj.avgeventtb,
        # 'avg': aj.avgevent, 'fit': {'amplitude': aj.Amplitude,
        #     'tau1': aj.tau1, 'tau2': aj.tau2, 'risepower': aj.risepower}, 'best_fit': aj.avg_best_fit,
        #     'risetenninety': aj.risetenninety, 'decaythirtyseven': aj.decaythirtyseven}])
        aev = self.cell_summary["averaged"]
        for i in range(len(aev)):
            P.axdict["F"].plot(aev[i]["tb"], aev[i]["avg"], "k-", linewidth=0.8)
            P.axdict["F"].plot(aev[i]["tb"], aev[i]["best_fit"], "r--", linewidth=0.4)

        if pdf is None:
            mpl.show()
        else:
            pdf.savefig(dpi=300)
            mpl.close()

    def plot_all_events_and_fits(self):
        P3 = PH.regular_grid(
            1,
            5,
            order="columns",
            figsize=(12, 8.0),
            showgrid=False,
            verticalspacing=0.1,
            horizontalspacing=0.02,
            margins={
                "leftmargin": 0.07,
                "rightmargin": 0.05,
                "topmargin": 0.03,
                "bottommargin": 0.05,
            },
            labelposition=(-0.12, 0.95),
        )
        idx = [a for a in P3.axdict.keys()]
        offset2 = 0.0
        k = 0
        all_evok = self.cell_summary[
            "indiv_evok"
        ]  # this is the list of ok events - a 2d list by
        for itr in range(len(all_evok)):  # for each trace
            for evok in all_evok[itr]:  # for each ok event in that trace
                P3.axdict[idx[k]].plot(
                    [
                        self.cell_summary["indiv_tb"][itr][0],
                        self.cell_summary["indiv_tb"][itr][-1],
                    ],
                    np.zeros(2) + offset2,
                    "b--",
                    linewidth=0.3,
                )
                P3.axdict[idx[k]].plot(
                    self.cell_summary["indiv_tb"][itr],
                    self.cell_summary["allevents"][itr][evok] + offset2,
                    "k--",
                    linewidth=0.5,
                )
                P3.axdict[idx[k]].plot(
                    self.cell_summary["indiv_tb"][itr],
                    -self.cell_summary["best_fit"][itr][evok] + offset2,
                    "r--",
                    linewidth=0.5,
                )
                if k == 4:
                    k = 0
                    offset2 += 20.0
                else:
                    k += 1
        mpl.show()

    def plot_hists(self):  # generate histogram of amplitudes for plots
        return
        histBins = 50
        nevents = len(self.cell_summary["amplitudes"])
        amp, ampbins, amppa = self.axAmps.hist(
            self.cell_summary["amplitudes"], histBins, alpha=0.5, density=True
        )
        # fit to normal distribution
        ampnorm = scipy.stats.norm.fit(self.cell_summary["amplitudes"])  #
        print(
            "    Amplitude (N={0:d} events) Normfit: mean {1:.3f}   sigma: {2:.3f}".format(
                nevents, ampnorm[0], ampnorm[1]
            )
        )
        x = np.linspace(
            scipy.stats.norm.ppf(0.01, loc=ampnorm[0], scale=ampnorm[1]),
            scipy.stats.norm.ppf(0.99, loc=ampnorm[0], scale=ampnorm[1]),
            100,
        )
        self.axAmps.plot(
            x,
            scipy.stats.norm.pdf(x, loc=ampnorm[0], scale=ampnorm[1]),
            "r-",
            lw=2,
            alpha=0.6,
            label="Norm: u={0:.3f} s={1:.3f}".format(ampnorm[0], ampnorm[1]),
        )
        k2, p = scipy.stats.normaltest(self.cell_summary["amplitudes"])
        print("    p (amplitude is Gaussian) = {:g}".format(1 - p))
        print("    Z-score for skew and kurtosis = {:g} ".format(k2))

        # fit to skewed normal distriubution
        ampskew = scipy.stats.skewnorm.fit(self.cell_summary["amplitudes"])
        print(
            "    ampskew: mean: {0:.4f} skew:{1:4f}  scale/sigma: {2:4f} ".format(
                ampskew[1], ampskew[0], 2 * ampskew[2]
            )
        )
        x = np.linspace(
            scipy.stats.skewnorm.ppf(
                0.002, a=ampskew[0], loc=ampskew[1], scale=ampskew[2]
            ),
            scipy.stats.skewnorm.ppf(
                0.995, a=ampskew[0], loc=ampskew[1], scale=ampskew[2]
            ),
            100,
        )
        self.axAmps.plot(
            x,
            scipy.stats.skewnorm.pdf(x, a=ampskew[0], loc=ampskew[1], scale=ampskew[2]),
            "m-",
            lw=2,
            alpha=0.6,
            label="skewnorm a={0:.3f} u={1:.3f} s={2:.3f}".format(
                ampskew[0], ampskew[1], ampskew[2]
            ),
        )

        # fit to gamma distriubution
        ampgamma = scipy.stats.gamma.fit(self.cell_summary["amplitudes"], loc=0.0)
        gamma_midpoint = scipy.stats.gamma.ppf(
            0.5, a=ampgamma[0], loc=ampgamma[1], scale=ampgamma[2]
        )  # midpoint of distribution
        print(
            "    ampgamma: mean: {0:.4f} gamma:{1:.4f}  loc: {2:.4f}  scale: {3:.4f} midpoint: {4:.4f}".format(
                ampgamma[0] * ampgamma[2],
                ampgamma[0],
                ampgamma[2],
                ampgamma[2],
                gamma_midpoint,
            )
        )
        x = np.linspace(
            scipy.stats.gamma.ppf(
                0.002, a=ampgamma[0], loc=ampgamma[1], scale=ampgamma[2]
            ),
            scipy.stats.gamma.ppf(
                0.995, a=ampgamma[0], loc=ampgamma[1], scale=ampgamma[2]
            ),
            100,
        )
        self.axAmps.plot(
            x,
            scipy.stats.gamma.pdf(x, a=ampgamma[0], loc=ampgamma[1], scale=ampgamma[2]),
            "g-",
            lw=2,
            alpha=0.6,
            label="gamma: a={0:.3f} loc={1:.3f}\nscale={2:.3f}, mid={3:.3f}".format(
                ampgamma[0], ampgamma[1], ampgamma[2], gamma_midpoint
            ),
        )  # ampgamma[0]*ampgamma[2]))
        self.axAmps.plot(
            [gamma_midpoint, gamma_midpoint],
            [
                0.0,
                scipy.stats.gamma.pdf(
                    [gamma_midpoint], a=ampgamma[0], loc=ampgamma[1], scale=ampgamma[2]
                ),
            ],
            "k--",
            lw=2,
            alpha=0.5,
        )
        self.axAmps.legend(fontsize=6)
        self.cell_summary["amplitude_midpoint"] = gamma_midpoint

        #
        # Interval distribution
        #
        an, bins, patches = self.axIntvls.hist(
            self.cell_summary["intervals"], histBins, density=True
        )
        nintvls = len(self.cell_summary["intervals"])
        expDis = scipy.stats.expon.rvs(
            scale=np.std(self.cell_summary["intervals"]), loc=0, size=nintvls
        )
        # axIntvls.hist(expDis, bins=bins, histtype='step', color='r')
        ampexp = scipy.stats.expon.fit(self.cell_summary["intervals"])
        x = np.linspace(
            scipy.stats.expon.ppf(0.01, loc=ampexp[0], scale=ampexp[1]),
            scipy.stats.expon.ppf(0.99, loc=ampexp[0], scale=ampexp[1]),
            100,
        )
        self.axIntvls.plot(
            x,
            scipy.stats.expon.pdf(x, loc=ampexp[0], scale=ampexp[1]),
            "r-",
            lw=3,
            alpha=0.6,
            label="Exp: u={0:.3f} s={1:.3f}\nMean Interval: {2:.3f}\n#Events: {3:d}".format(
                ampexp[0],
                ampexp[1],
                np.mean(self.cell_summary["intervals"]),
                len(self.cell_summary["intervals"]),
            ),
        )
        self.axIntvls.legend(fontsize=6)

        # report results
        print("   N events: {0:7d}".format(nintvls))
        print(
            "   Intervals: {0:7.1f} ms SD = {1:.1f} Frequency: {2:7.1f} Hz".format(
                np.mean(self.cell_summary["intervals"]),
                np.std(self.cell_summary["intervals"]),
                1e3 / np.mean(self.cell_summary["intervals"]),
            )
        )
        print(
            "    Amplitude: {0:7.1f} pA SD = {1:.1f}".format(
                np.mean(self.cell_summary["amplitudes"]),
                np.std(self.cell_summary["amplitudes"]),
            )
        )

        # test if interval distribtuion is poisson:
        stats = scipy.stats.kstest(
            expDis,
            "expon",
            args=((np.std(self.cell_summary["intervals"])),),
            alternative="two-sided",
        )
        print(
            "    KS test for intervals Exponential: statistic: {0:.5f}  p={1:g}".format(
                stats.statistic, stats.pvalue
            )
        )
        stats_amp = scipy.stats.kstest(
            expDis,
            "norm",
            args=(
                np.mean(self.cell_summary["amplitudes"]),
                np.std(self.cell_summary["amplitudes"]),
            ),
            alternative="two-sided",
        )
        print(
            "    KS test for Normal amplitude: statistic: {0:.5f}  p={1:g}".format(
                stats_amp.statistic, stats_amp.pvalue
            )
        )

    def plot_setup(self):
        sizer = OrderedDict(
            [
                ("A", {"pos": [0.12, 0.8, 0.35, 0.60]}),
                #  ('A1', {'pos': [0.52, 0.35, 0.35, 0.60]}),
                ("B", {"pos": [0.12, 0.35, 0.08, 0.20]}),
                ("C", {"pos": [0.60, 0.35, 0.08, 0.20]}),
            ]
        )  # dict elements are [left, width, bottom, height] for the axes in the plot.
        n_panels = len(sizer.keys())
        gr = [
            (a, a + 1, 0, 1) for a in range(0, n_panels)
        ]  # just generate subplots - shape does not matter
        axmap = OrderedDict(zip(sizer.keys(), gr))
        self.P = PH.Plotter((n_panels, 1), axmap=axmap, label=True, figsize=(7.0, 9.0))
        self.P.resize(sizer)  # perform positioning magic
        ax0 = self.P.axdict["A"]
        ax0.set_ylabel("pA", fontsize=9)
        ax0.set_xlabel("T (s)", fontsize=9)
        # self.axdec = P.axdict['A1']
        axIntvls = self.P.axdict["B"]
        axIntvls.set_ylabel("Fraction of Events", fontsize=9)
        axIntvls.set_xlabel("Interevent Interval (ms)", fontsize=9)
        axIntvls.set_title("mEPSC Interval Distributon", fontsize=10)
        axAmps = self.P.axdict["C"]
        axAmps.set_ylabel("Fraction of Events", fontsize=9)
        axAmps.set_xlabel("Event Amplitude (pA)", fontsize=9)
        axAmps.set_title("mEPSC Amplitude Distribution", fontsize=10)
        self.ax0 = ax0
        self.axIntvls = axIntvls
        self.axAmps = axAmps


if __name__ == "__main__":

    # example of how to use the analysis in conjunction with a data plan
    # usually this kind of code will be in a separate directory where the specific
    # runner and results for a given experiment are located.

    parser = argparse.ArgumentParser(description="mini synaptic event analysis")
    parser.add_argument("datadict", type=str, help="data dictionary")
    parser.add_argument(
        "-o", "--one", type=str, default="", dest="do_one", help="just do one"
    )
    parser.add_argument(
        "-c", "--check", action="store_true", help="Check for files; no analysis"
    )
    parser.add_argument(
        "-m",
        "--mode",
        type=str,
        default="aj",
        dest="mode",
        choices=["aj", "cb"],
        help="just do one",
    )
    parser.add_argument(
        "-v", "--view", action="store_false", help="Turn off pdf for single run"
    )

    args = parser.parse_args()

    dataplan = EP.DataPlan.DataPlan(args.datadict)

    MI = MiniAnalysis(dataplan)
    filterstring = "test"
    if args.do_one == "":  # no second argument, run all data sets
        print("doing all...", args.do_one)
        MI.analyze_all(
            fofilename="all_{0:s}_{1:s}_{2:s}.pdf".format(
                args.datadict, filterstring, args.mode
            ),
            check=args.check,
            mode=args.mode,
        )
    else:
        summarydata = {}
        try:
            filtered = dataplan.data["notch_and_hpf_filter"]
            filterstring = "filtered"
        except KeyError:
            filtered = False
            filterstring = "nofilter"
        fout = "summarydata_{0:s}_{1:s}_{2:s}.p".format(
            args.do_one, filterstring, args.mode
        )

        if not args.view:
            fofilename = "summarydata_{0:s}_{1:s}_{2:s}.pdf".format(
                args.do_one, filterstring, args.mode
            )
            print("fofile: ", fofilename)
            with PdfPages(fofilename) as pdf:
                MI.analyze_one_cell(
                    args.do_one,
                    pdf=fofilename,
                    maxprot=10,
                    check=args.check,
                    mode=args.mode,
                )
        else:
            MI.analyze_one_cell(
                args.do_one, pdf=None, maxprot=10, check=args.check, mode=args.mode
            )

        # print('MI summary: ', MI.cell_summary)
        summarydata = {args.do_one: MI.cell_summary}
        fh = open(fout, "wb")
        pickle.dump(summarydata, fh)
        fh.close()
