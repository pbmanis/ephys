"""
Analyze EPSCs or IPSCs
Or EPSPs and IPSPs...

This module provides the following analyses:

    1. Amplitudes from a train
    2. Paired pulse facilitation for pulse pairs, and the first pair in a train.
    3. Current-voltage relationship in voltage clamp measured over a time window

The results of the analysis are stored in the class variable analysis_summary

Note: if the analyzer is called with update_regions set True, then traces will be
sent to cursor_plot to get start and end times. (this might be broken now - need to test)

"""
import os  # legacy
import sys
from collections import OrderedDict
from pathlib import Path
from typing import List, Tuple, Union

import lmfit
import MetaArray as EM  # need to use this version for Python 3
import numpy as np
import pandas as pd
from matplotlib import pyplot as mpl
from pylibrary.plotting import plothelpers as PH
from pylibrary.tools import cprint
from pyqtgraph.Qt import QtCore, QtGui

import ephys.psc_analysis.analyze_IO as A_IO
import ephys.psc_analysis.analyze_PPF as A_PPF
import ephys.psc_analysis.analyze_VDEP as A_VDEP
import ephys.psc_analysis.functions as FN
from ephys.datareaders import acq4_reader
from ephys.tools import cursor_plot as CP


class PSCAnalyzer:
    def __init__(
        self,
        datapath: Union[str, Path],
        plot: bool = True,
        update_regions: bool = False,
    ):
        """
        Analyze PSCs in a few different formats:
            1. IO - a stimulus sequence with increasing stimulation current,
            all collected at a single holding voltage
            2. VDEP - a Meausrement of EPSCs across voltage, targeted at obtaining
            an NMDA/AMPA current ratio from currents at +50 and -90 mV. Data may include
            averaging of repetead trials.
            3. PPF - Paired pulse facilitiation over several intervals; may include repeated
            trials

        Parameters
        ----------
        datapath: path to the data protocol (Path or string)

        plot: boolean (default: True)
            Flag to control plotting of the data

        update_regions: Boolean (default: False)
            A flag that forces the routines to plot data so that a time window for the
            analysis can be defined and saved.

        """
        self.datapath = datapath
        self.AR = (
            acq4_reader.acq4_reader()
        )  # make our own private version of the analysis and reader
        self.plot = plot
        self.db = None
        self.db_filename = None
        self.protocol_map = (
            {}
        )  # hold the mapping between protocol names and the analysis routine
        self.assign_default_protocol_map()  # set up some defaults for the protocol map
        self.update_regions = update_regions
        self.JunctionPotential = -8.0 * 1e-3  # junction potential for correction
        self.NMDA_voltage = 0.050  # in V  positive
        self.AMPA_voltage = (
            -0.0741
        )  # in V  - this is the Cl eq potential to minize GABA interference
        self.NMDA_delay = 0.050  # delay in s to make measurement

    def setup(
        self,
        clamps: object = None,
        spikes: dict = None,
        baseline: Union[List, Tuple] = [0, 0.001],
        ignore_important_flag: bool = True,
        device: str = "Stim0",
    ):
        """
        Set up for the fitting

        Parameters
        ----------
        clamps: A datamodel structure (required)
            Brings the data to the module. This usually will be a PatchEphys object.

        spikes: A spikeAnalysis structure (optional)
            Has information about which traces have spikes
            Use this when analyzing events that may be contaminated by spikes

        baseline: list (2 elements)
            times over which baseline is measured (in seconds)

        """

        if clamps is None:
            raise ValueError("VC analysis requires defined clamps ")
        self.Clamps = clamps
        self.spikes = spikes
        self.device = device
        self.ignore_important_flag = ignore_important_flag
        self.set_baseline_times(baseline)
        self.analysis_summary = {}  # init the result structure
        self.set_NGlist()

    def make_key(self, pathname):
        """
        Make a key string using the date, slice, cell and protocol from the path name

        """
        p = pathname.parts
        return str("~".join([p[i] for i in range(-4, 0)]))

    def get_meta_data(self, protocol: str):
        self.protocol = protocol
        self.pulse_train = self.AR.getStim(self.device)  # get the stimulus information
        # stim dict in pulse_train will look like:
        # {'start': [0.05, 0.1], 'duration': [0.0001, 0.0001],
        # 'amplitude': [0.00025, 0.00025], 'npulses': [2], 'period': [0.05], 'type': ['pulseTrain']}
        # try:
        self.devicedata = self.AR.getDeviceData(
            device=self.device, devicename="command"
        )
        if self.devicedata is None:
            print("No device data? name command, ", self.device)
            return False
        self.reps = self.AR.sequence[("protocol", "repetitions")]

        try:  # get the stimulus amplitude data for an IO functoin
            self.stim_io = self.AR.sequence[
                (self.device, "command.PulseTrain_amplitude")
            ]
        except:
            self.stim_io = None
        try:  # get the stimulus rate
            self.stim_dt = self.AR.sequence[(self.device, "command.PulseTrain_period")]
        except:
            self.stim_dt = None
        try:  # get the voltage from the multiclamp in vclamp mode
            self.stim_V = self.AR.sequence[("MultiClamp1", "Pulse_amplitude")]
        except:
            self.stim_V = None

    def _getData(self, protocolName: str, device: str = "Stim0"):
        self.AR.setProtocol(self.datapath)  # define the protocol path where the data is
        self.setup(clamps=self.AR, device=device)
        if not self.AR.getData():  # get that data.
            return False
        # print("Protocol important: ", self.AR.protocol_important, "ignore: ", ignore_important_flag)
        if not self.AR.protocol_important and not self.ignore_important_flag:
            return False
        self.get_meta_data(protocol=protocolName)
        self.read_database(f"{protocolName:s}.p")
        return True

    def set_NGlist(self, NGlist=[]):
        self.NGlist = NGlist  # list of "not good" traces within a protocol

    def check_protocol(self, protocol):
        """
        Verify that the protocol we are examining is complete.
        Returns True or False
        """

        return self.AR.checkProtocol(protocol)

    def assign_default_protocol_map(self):
        self.protocol_map["IO_protocols"] = ["Stim_IO"]
        self.protocol_map["VDEP"] = ["VC_EPSC_3"]
        self.protocol_map["PPF"] = ["PPF"]

    def assign_protocols(self, protocol_type: str, protocol_name: str):
        """Assign analysis routines to protocol names.
        protocol_type must be a known type (see assign_default_protocol_map for
        currently implemented types)

        """
        if protocol_type not in self.protocol_map.keys():
            print("Protocol type not in implemented protocol types")
            return
        else:
            self.protocol_map[self.protocoltype].append(protocol_name)

    def read_database(self, filename: Union[str, Path]):
        """
        Read the database that will be used for analysis
        The database is a pandas pickled file with columns
        date, protocol, T0 and T1

        Parameters
        ----------
        filename: str or Path
            The name of the database file (full path or file if in the current
            working directory)
        """

        self.db_filename = Path(filename)
        if self.db_filename.is_file():
            with (open(self.db_filename, "rb")) as fh:
                self.db = pd.read_pickle(fh, compression=None)
        else:
            self.db = pd.DataFrame(columns=["date", "protocol", "T0", "T1"])

    def update_database(self):
        """
        Write the database
        """

        if self.db is not None:
            self.db.to_pickle(self.db_filename)

    def measure_PSC(
        self,
        protocolName: str,
        plot: bool = True,
        savetimes: bool = False,
        ignore_important_flag: bool = True,
        device: str = "Stim0",
    ):
        """
        Direct the analysis
        Uses the beginning of the protocol name to select which analysis to use

        Parameters
        ----------
        protocolName: str
            Name of the protocol to analyze, underneath the datapath

        plot: boolean (default: True)
            Flag to plot data

        """
        if not self._getData(protocolName=protocolName, device=device):
            return False

        if protocolName.startswith("Stim_IO"):
            ok = A_IO.analyze_IO(self)
        elif protocolName.startswith("VC-EPSC_3"):
            ok = A_VDEP.analyze_VDEP(self)
        elif protocolName.startswith("PPF"):
            ok = A_PPF.analyze_PPF(self)
        if not ok:
            print("Failed on protocol in IV: ", self.datapath, protocolName)
            return False
        if plot:
            self.plot_vciv()
        if savetimes:
            date = self.make_key(self.datapath)

            if date not in self.db["date"].tolist():
                self.db.loc[len(self.db)] = [date, protocolName, self.T0, self.T1]
                print("new date added")
            else:
                self.db.loc[date, "date"] = date
                self.db.loc[date, "protocol"] = protocolName
                self.db.loc[date, "T0"] = self.T0
                self.db.loc[date, "T1"] = self.T1
                print("old date data updated")
            self.update_database()
            # print('db head: ', self.db.head())
        return True

    def get_stimtimes(self):
        """
        This should get the stimulus times from the Acq4 protocol.
        Right now, it does nothing
        """
        pass

    def set_baseline_times(self, baseline: Union[List, Tuple]):
        """
        baseline: 2-element list or numpy array
        """
        if len(baseline) != 2:
            raise ValueError("Baseline must be a 2-element array")
        if isinstance(baseline, list):
            baseline = np.array(baseline)
        self.baseline = np.sort(baseline)

    def get_baseline(self):
        """Return the mean values in the data over the baseline region."""
        bl, result = FN.mean_I_analysis(
            clamps=self.Clamps, region=self.baseline, reps=[0]
        )
        return bl

    def _clean_array(self, rgn: Union[List, Tuple]):
        """
        Just make sure that the rgn array is a list of two values
        """
        if isinstance(rgn[0], list) or isinstance(rgn[0], np.ndarray):
            rgn = [x[0] for x in rgn]
        return rgn

    def _compute_interval(
        self,
        x0: float = 0.0,
        artifact_delay: float = 0.0,
        index: int = 0,
        stim_intvl: list = [],
        max_width: float = 25.0,
        pre_time: float = 1.0e-3,
        pflag: bool = False,
    ):
        """
        Comptue the interval over which to measure an event.
        The interval cannot be longer than the interval to the next event.

        x0: float
            starting time for the interval
        artifact_delay: float
            duration to remove from the start of the trace as the stimulus
            artifact
        index: int
            index into the stim_intvl list
        stim_intvl:
            list of stimulus intervals, in order
        max_width: float
            width of the ttrace to retrun, starting at x0
        pre_time: float
            Time to clip at the end of the trace (in case the next stim is
            not exactly where it is expected to be)
        Returns
            2 element list of times adjusted for the delays and widths.
        --------
        window as an np array of 2 elements, min and max time
        """
        num_intervals = len(stim_intvl)
        if index < num_intervals - 1:
            nxt_intvl = (
                stim_intvl[index + 1] - stim_intvl[index]
            )  # check interval sequence
            max_w = np.min((nxt_intvl, max_width - pre_time))
            if nxt_intvl > 0:  # still ascending
                t_stim = [
                    x0 + artifact_delay,
                    x0 + max_w - pre_time,
                ]  # limit width if interval is
                if pflag:
                    print("nxt>0: ", t_stim)
            else:
                t_stim = [x0 + artifact_delay, x0 + max_width - pre_time]
                if pflag:
                    print("nxt < 0: ", t_stim)
        else:
            t_stim = [x0 + artifact_delay, x0 + max_width - pre_time]
            if pflag:
                print("last index: ", t_stim)
        t_stim = self._clean_array(t_stim)
        return t_stim

    def plot_data(self, tb, data1, title=""):
        """
        Quick plot of data for testing purposes

        Parameters
        ----------
        tb: np.array (no default)
            the time base (one dimension)

        data1: np.array (no default)
            The data, can be [m traces x npoints]

        title: str (default: '')
            A title to put on the plot

        Return
        ------
        Nothing
        """

        f, ax = mpl.subplots(1)
        ax = np.array(ax).ravel()
        ie = data1.shape[1]
        it = tb.shape[0]
        if ie > it:
            ie = it
        if it > ie:
            it = ie
        print(it, ie)
        for i in range(data1.shape[0]):
            ax[0].plot(tb[:it], data1[i, :ie])
        ax[0].set_title(
            str(self.datapath).replace("_", r"\_") + " " + title, fontsize=8
        )
        mpl.show()

    def set_region(self, region=None, baseline=None, slope=True):
        print("set region")
        if region is None:
            raise ValueError(
                "PSCAnalyzer, set_region requires a region beginning and end to measure the current"
            )

        data1 = self.Clamps.traces["Time" : region[0] : region[1]]
        if baseline is None:
            baseline = [0.0]

        tb = np.arange(
            0, data1.shape[1] * self.Clamps.sample_interval, self.Clamps.sample_interval
        )
        data1 = data1.view(np.ndarray)
        newCP = CP.CursorPlot(str(self.datapath))
        setline = True

        # for i in range(data1.shape[0]):
        newCP.plotData(
            x=tb,
            y=np.array([data1[i] - baseline[i] for i in range(data1.shape[0])]) * 1e12,
            setline=setline,
            slope=slope,
        )
        # setline = False # only do on first plot

        if (sys.flags.interactive != 1) or not hasattr(QtCore, "PYQT_VERSION"):
            QtGui.QApplication.instance().exec_()
        print("done with cp")
        self.T0, self.T1 = newCP.selectedRegion
        if self.T0 is None:
            return None
        return newCP.selectedRegion

    def plot_vciv(self):
        """
        Plot the current voltage-clamp IV function
        """
        P = PH.regular_grid(
            rows=2,
            cols=2,
            order="columnsfirst",
            figsize=(8.0, 6.0),
            showgrid=False,
            verticalspacing=0.1,
            horizontalspacing=0.1,
            margins={
                "leftmargin": 0.12,
                "rightmargin": 0.12,
                "topmargin": 0.08,
                "bottommargin": 0.1,
            },
            labelposition=(-0.12, 0.95),
            panel_labels=["A", "B", "C", "D"],
        )
        (date, sliceid, cell, proto, p3) = self.file_cell_protocol(self.datapath)
        P.figure_handle.suptitle(str(Path(date, sliceid, cell, proto)), fontsize=12)
        bl = self.get_baseline()
        if "PPF" in self.analysis_summary.keys():
            maxt = 250.0
        else:
            maxt = 150.0
        for i in range(self.AR.traces.shape[0]):
            P.axdict["A"].plot(
                self.AR.time_base * 1e3,
                (self.AR.traces[i].view(np.ndarray) - bl[i]) * 1e12,
                # "k-",
                linewidth=0.5,
            )
        if (
            "PSP_VDEP_AMPA" in self.analysis_summary.keys()
            or "PSP_VDEP_NMDA" in self.analysis_summary.keys()
        ):
            P.axdict["A"].set_xlim(
                self.analysis_summary["stim_times"][0] * 1e3 - 10,
                self.analysis_summary["stim_times"][0] * 1e3 + 50,
            )
            # P.axdict['A'].set_ylim(-2500, 2500)
        else:
            P.axdict["A"].set_xlim(40, maxt + 200)
            P.axdict["A"].set_ylim(-3000, 2000)
        # PH.talbotTicks(P.axdict['A'], tickPlacesAdd={'x': 0, 'y': 0}, floatAdd={'x': 0, 'y': 0})
        P.axdict["A"].set_xlabel("T (ms)")
        P.axdict["A"].set_ylabel("I (pA)")
        if "PSP_IO" in self.analysis_summary.keys():  # io function
            for i in range(len(self.analysis_summary["stim_times"])):
                P.axdict["C"].plot(
                    self.analysis_summary["psc_stim_amplitudes"][0],
                    np.array(self.analysis_summary[f"PSP_IO"][i]),
                    linewidth=1,
                    markersize=4,
                    marker="s",
                )
            # except:
            #     print("Plot Failed on protocol: ", self.datapath, proto)
            P.axdict["C"].set_xlabel("Istim (microAmps)")
            P.axdict["C"].set_ylabel("EPSC I (pA)")
            PH.talbotTicks(
                P.axdict["C"], tickPlacesAdd={"x": 0, "y": 0}, floatAdd={"x": 0, "y": 0}
            )
        elif (
            "PSP_VDEP_AMPA" in self.analysis_summary.keys()
            or "PSP_VDEP_NMDA" in self.analysis_summary.keys()
        ):

            n_voltages = len(self.analysis_summary[f"PSP_VDEP_AMPA"][0])

            for i in range(len(self.analysis_summary["stim_times"])):
                P.axdict["C"].plot(
                    self.analysis_summary["Vcmd"][:n_voltages] * 1e3,
                    self.sign
                    * np.array(self.analysis_summary[f"PSP_VDEP_AMPA"][i])
                    * 1e12,
                    marker="o",
                    linewidth=1,
                    markersize=4,
                )
                P.axdict["C"].plot(
                    self.analysis_summary["Vcmd"][:n_voltages] * 1e3,
                    self.sign
                    * np.array(self.analysis_summary[f"PSP_VDEP_NMDA"][i])
                    * 1e12,
                    marker="s",
                    linewidth=1,
                    markersize=4,
                )
            P.axdict["C"].set_xlabel("V (mV)")
            P.axdict["C"].set_ylabel("EPSC I (pA)")
            P.axdict["C"].set_ylim(-2000, 5000)
            PH.crossAxes(P.axdict["C"], xyzero=(-60.0, 0.0))
            # PH.talbotTicks(
            #     P.axdict["C"], tickPlacesAdd={"x": 0, "y": 0}, floatAdd={"x": 0, "y": 0}
            # )
        elif "PPF" in self.analysis_summary.keys():
            k = list(self.analysis_summary["PPF"].keys())
            xm = []
            ym = []
            for i, sdt in enumerate(self.stim_dt):
                x = (
                    np.repeat(np.array(sdt), len(self.analysis_summary["PPF"][sdt]))
                    * 1e3
                )
                y = self.sign * np.array(self.analysis_summary[f"PPF"][sdt])
                P.axdict["C"].scatter(
                    x,
                    y,
                    s=12,
                )
                xm.append(x[0])
                ym.append(y)
                P.axdict["C"].set_xlim(0, 200.0)
                P.axdict["C"].set_ylim(0, 3.0)
                PH.referenceline(P.axdict["C"], 1.0)
                P.axdict["C"].set_xlabel("Interval (ms)")
                P.axdict["C"].set_ylabel("PPF (R2/R1)")
                PH.talbotTicks(
                    P.axdict["C"],
                    tickPlacesAdd={"x": 0, "y": 1},
                    floatAdd={"x": 0, "y": 1},
                )
            P.axdict["C"].errorbar(x=xm, y=np.mean(ym, axis=1), yerr=np.std(ym, axis=1))

        P.axdict["B"].set_xlabel("I (nA)")
        P.axdict["B"].set_ylabel("V (mV)")
        PH.talbotTicks(
            P.axdict["B"], tickPlacesAdd={"x": 1, "y": 0}, floatAdd={"x": 2, "y": 0}
        )

        P.axdict["D"].set_xlabel("I (pA)")
        P.axdict["D"].set_ylabel("Latency (ms)")

        self.IVFigure = P.figure_handle

        if self.plot:
            mpl.show()

    def file_cell_protocol(self, filename):
        """
        file_cell_protocol breaks the current filename down and returns a
        tuple: (date, cell, protocol)

        Parameters
        ----------
        filename: str
            Name of the protocol to break down

        Returns
        -------
        tuple: (date, sliceid, cell, protocol, any other...)
            last argument returned is the rest of the path...
        """
        (p0, proto) = os.path.split(filename)
        (p1, cell) = os.path.split(p0)
        (p2, sliceid) = os.path.split(p1)
        (p3, date) = os.path.split(p2)
        return (date, sliceid, cell, proto, p3)


def test():
    """
    This is for testing - normally an instance of EPSC_analyzer would be
    created and these values would be filled in.
    """
    import matplotlib
    from matplotlib import rc

    # rc('font',**{'family':'sans-serif','sans-serif':['Arial']})
    # rcParams['font.sans-serif'] = ['Arial']
    # rcParams['font.family'] = 'sans-serif'
    rc("text", usetex=False)
    rcParams = matplotlib.rcParams
    rcParams["svg.fonttype"] = "none"  # No text as paths. Assume font installed.
    rcParams["pdf.fonttype"] = 42
    rcParams["ps.fonttype"] = 42
    # disk = '/Volumes/Pegasus/ManisLab_Data3'
    # disk = '/Volumes/PBM_005/data'
    disk = "/Volumes/Pegasus/ManisLab_Data3"
    middir = "Kasten_Michael"
    directory = "Maness_PFC_stim"
    cell = "2019.03.19_000/slice_000/cell_001"
    cell = "2019.03.19_000/slice_001/cell_000"

    fn = Path(disk, middir, directory, cell)
    protocol = "Stim_IO_1_001"
    protocol = "PPF_2_001"
    protocol = "VC-EPSC_3_ver2_003"
    fn = Path(fn, protocol)
    if not fn.is_dir():
        print("protocol directory not found: ", fn)
        exit()

    PSC = PSCAnalyzer(fn, protocol[:-4])
    PSC.measure_PSC(protocolName=protocol, savetimes=True)


if __name__ == "__main__":
    test()
