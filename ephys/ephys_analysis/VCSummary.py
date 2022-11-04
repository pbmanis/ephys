from __future__ import print_function

"""
Compute IV from voltage clamp data.
Version 0.1, does only min negative peak IV, max pos IV and ss IV

"""

from pathlib import Path
from typing import Union

import matplotlib
import matplotlib.colors
import matplotlib.pyplot as mpl
import numpy as np
import pylibrary.plotting.plothelpers as PH

import MetaArray as EM

from ..datareaders import acq4read

color_sequence = ["k", "r", "b"]
colormap = "snshelix"


class VCSummary:
    def __init__(
        self, datapath, altstruct=None, file: Union[str, Path, None] = None, plot=True
    ):
        self.datapath = datapath
        self.mode = "acq4"
        self.plot = plot

        if datapath is not None:
            self.AR = (
                acq4read.Acq4Read()
            )  # make our own private version of the analysis and reader
            self.datapath = datapath
        else:
            self.AR = altstruct
            self.datapath = file
            self.mode = "nwb2.5"

    def setup(self, clamps=None, baseline=[0, 0.001], taumbounds=[0.002, 0.050]):
        """
        Set up for the fitting

        Parameters
        ----------
        clamps : A datamodel structure (required)
            Brings the data to the module. This usually will be a PatchEphys object.

        spikes : A spikeAnalysis structure (required)
            Has information about which traces have spikes

        dataplot : pyqtgraph plot object
            pyqtgraph plot to use for plotting the data. No data is plotted if not specified

        baseline : list (2 elements)
            times over which baseline is measured (in seconds)

        taumbounds : list (2 elements)
            Lower and upper bounds of the allowable taum fitting range (in seconds).
        """

        if clamps is None:
            raise ValueError("VC analysis requires defined clamps ")
        self.Clamps = clamps
        self.analysis_summary = {}

    def compute_iv(self):
        """
        Simple plot voltage clamp traces
        """
        # print('path: ', self.datapath)
        self.AR.setProtocol(self.datapath)  # define the protocol path where the data is
        self.setup(clamps=self.AR)
        if self.AR.getData():  # get that data.
            self.analyze()
            self.plot_vciv()
            return True
        return False

    def analyze(self, rmpregion=[0.0, 0.05], tauregion=[0.1, 0.125]):
        # self.rmp_analysis(region=rmpregion)
        #        self.tau_membrane(region=tauregion)
        r0 = self.Clamps.tstart + 0.9 * (self.Clamps.tend - self.Clamps.tstart)  #
        self.ihold_analysis(region=[0.0, self.Clamps.tstart])
        self.vcss_analysis(region=[r0, self.Clamps.tend])
        # self.ivpk_analysis(region=[self.Clamps.tstart, self.Clamps.tstart+0.4*(self.Clamps.tend-self.Clamps.tstart)])

    def ihold_analysis(self, region=None):
        """
        Get the holding current

        Parameters
        ----------
        region : tuple, list or numpy array with 2 values (default: None)
            start and end time of a trace used to measure the RMP across
            traces.

        Return
        ------
        Nothing

        Stores computed RMP in mV in the class variable rmp.
        """
        if region is None:
            raise ValueError(
                "VCSummary, ihold_analysis requires a region beginning and end to measure the RMP"
            )
        data1 = self.Clamps.traces["Time" : region[0] : region[1]]
        data1 = data1.view(np.ndarray)
        self.vcbaseline = data1.mean(axis=1)  # all traces
        self.vcbaseline_cmd = self.Clamps.commandLevels
        self.iHold = np.mean(self.vcbaseline) * 1e9  # convert to nA
        self.analysis_summary["iHold"] = self.iHold

    def vcss_analysis(self, region=None):
        """
        compute steady-state IV curve - from the mean current
        across the stimulus set over the defined time region
        (this usually will be the last half or third of the trace)

        Parameters
        ----------
        region : list or tuple
            Start and end times for the analysis
        """
        data0 = self.Clamps.traces["Time": self.Clamps.tstart: self.Clamps.tend]
        data1 = self.Clamps.traces["Time" : region[0] : region[1]]
        icmds = EM.MetaArray(
            self.Clamps.cmd_wave,  # easiest = turn in to a matching metaarray...

            info=[
                {
                    "name": "Command",
                    "units": "A",
                    "values": np.array(self.Clamps.commandLevels),
                },
                self.Clamps.traces.infoCopy("Time"),
                self.Clamps.traces.infoCopy(-1),
            ],
        )
        self.vcss_vcmd = icmds["Time" : region[0] : region[1]].mean(axis=1)
        self.r_in = np.nan
        self.analysis_summary["Rin"] = np.nan
        self.vcss_v = []
        if data1.shape[1] == 0 or data1.shape[0] == 1:
            return  # skip it

        ntr = len(self.Clamps.traces)

        self.vcss_Im = data1.mean(axis=1)  # steady-state, all traces
        self.vcpk_Im = data0.max(axis=1)
        self.vcmin_Im = data0.min(axis=1)
        self.analysis_summary["Rin"] = np.NaN
        #        self.Clamps.plotClampData()

        isort = np.argsort(self.vcss_vcmd)
        self.vcss_Im = self.vcss_Im[isort]
        self.vcss_vcmd = self.vcss_vcmd[isort]
        bl = self.vcbaseline[isort]
        self.vcss_bl = bl
        # compute Rin from the SS IV:
        # this makes the assumption that:
        # successive trials are in order so we sort above
        # commands are not repeated...
        if len(self.vcss_vcmd) > 1 and len(self.vcss_v) > 1:
            pf = np.polyfit(
                self.vcss_vcmd,
                self.vcss_v,
                3,
                rcond=None,
                full=False,
                w=None,
                cov=False,
            )
            pval = np.polyval(pf, self.vcss_vcmd)
            # print('pval: ', pval)
            slope = np.diff(pval) / np.diff(self.vcss_vcmd)  # local slopes
            imids = np.array((self.vcss_vcmd[1:] + self.vcss_vcmd[:-1]) / 2.0)
            self.rss_fit = {"I": imids, "V": np.polyval(pf, imids)}
            # print('fit V: ', self.rss_fit['V'])
            # slope = slope[[slope > 0 ] and [self.vcss_vcmd[:-1] > -0.8] ] # only consider positive slope points
            l = int(len(slope) / 2)
            maxloc = np.argmax(slope[l:]) + l
            self.r_in = slope[maxloc]
            self.r_in_loc = [
                self.vcss_vcmd[maxloc],
                self.vcss_v[maxloc],
                maxloc,
            ]  # where it was found
            minloc = np.argmin(slope[:l])
            self.r_in_min = slope[minloc]
            self.r_in_minloc = [
                self.vcss_vcmd[minloc],
                self.vcss_v[minloc],
                minloc,
            ]  # where it was found
            self.analysis_summary["Rin"] = self.r_in * 1.0e-6

    def plot_vciv(self):

        x = -0.05
        y = 1.05
        sizer =  {"A": {"pos": [0.10, 0.51, 0.32, 0.60], "labelpos": (x,y)},
                  "B": {"pos": [0.10, 0.51, 0.08, 0.15], "labelpos": (x,y)}, 
                  "C": {"pos": [0.65, 0.30, 0.55, 0.30], "labelpos": (x,y)}, 
                  "D": {"pos": [0.69, 0.30, 0.08, 0.30], "labelpos": (x,y)},
                  }
        P = PH.arbitrary_grid(sizer, figsize=(8, 6), 

            )
        (date, sliceid, cell, proto, p3) = self.file_cell_protocol(self.datapath)

        P.figure_handle.suptitle(
            str(Path(date, sliceid, cell, proto)), fontsize=12
        )
        for i in range(self.AR.traces.shape[0]):
            P.axdict["A"].plot(
                self.AR.time_base * 1e3,
                self.AR.traces[i, :].view(np.ndarray) * 1e12,
                "k-",
                linewidth=0.5,
            )
        # traces in A
        PH.nice_plot(P.axdict["A"], position=-0.03, direction="outward", ticklength=3)
        PH.talbotTicks(
            P.axdict["A"], tickPlacesAdd={"x": 0, "y": 0}, floatAdd={"x": 0, "y": 0}
        )
        P.axdict["A"].set_xlabel("T (ms)")
        P.axdict["A"].set_ylabel("I (pA)")
        PH.crossAxes(P.axdict["C"], xyzero=(-60.0, 0.0))
        
        # crossed IV in C
        cmdv = (self.vcss_vcmd.view(np.ndarray)) * 1e3
        print(self.vcss_vcmd.view(np.ndarray))
        print(self.AR.holding)

        P.axdict["C"].plot(
            cmdv, self.vcss_Im.view(np.ndarray) * 1e12, "ks-", linewidth=1, markersize=2.5
        )
        P.axdict["C"].plot(
            cmdv, self.vcpk_Im.view(np.ndarray) * 1e12, "ro-", linewidth=1, markersize=4
        )
        P.axdict["C"].plot(
            cmdv, self.vcmin_Im.view(np.ndarray) * 1e12, "b^-", linewidth=1, markersize=4
        )
        P.axdict["C"].set_xlabel("V (mV)")
        P.axdict["C"].set_ylabel("I (pA)")
        PH.talbotTicks(
            P.axdict["C"], tickPlacesAdd={"x": 0, "y": 0}, floatAdd={"x": 0, "y": 0}
        )

        # Voltage command B
        PH.nice_plot(P.axdict["B"], position=-0.03, direction="outward", ticklength=3)
        P.axdict["B"].set_xlabel("I (nA)")
        P.axdict["B"].set_ylabel("V (mV)")
        # PH.talbotTicks(
        #     P.axdict["B"], tickPlacesAdd={"x": 1, "y": 1}, floatAdd={"x": 2, "y": 1}
        # )
        P.axdict["B"].set_ylim(-120., 60.)
        for i in range(self.AR.traces.shape[0]):
            P.axdict["B"].plot(
                self.AR.time_base * 1e3,
                self.AR.cmd_wave[i, :].view(np.ndarray)*1e3,
                "k-",
                linewidth=0.5,
            )

        
        # something in D
        PH.nice_plot(P.axdict["D"], position=-0.03, direction="outward", ticklength=3)
        P.axdict["D"].set_xlabel("V (mV)")
        P.axdict["D"].set_ylabel("I (pA)")

        self.IVFigure = P.figure_handle

        if self.plot:
            mpl.show()

    def file_cell_protocol(self, filename):
        """
        file_cell_protocol breaks the current filename down and returns a
        tuple: (date, cell, protocol)

        Parameters
        ----------
        filename : str
            Name of the protocol to break down

        Returns
        -------
        tuple : (date, sliceid, cell, protocol, any other...)
            last argument returned is the rest of the path...
        """
        fileparts = Path(filename).parts
        if len(fileparts) < 3:
            # probably a short version, nwb file.
            fileparts = filename.split('~')
            date = fileparts[0]+"_000"
            sliceid = f"slice_{int(fileparts[1][1]):03d}"
            cell =    f"cell_{int(fileparts[1][3]):03d}"
            proto = fileparts[-1]
            p3 = ""
        else:
            proto = fileparts[-1]
            sliceid = fileparts[-2]
            cell = fileparts[-3]
            date = fileparts[-4]
            p3 = fileparts[:-4]
        return (date, sliceid, cell, proto, p3)
