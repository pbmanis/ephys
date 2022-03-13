import argparse
from pathlib import Path
import numpy as np
import scipy.signal
import scipy.ndimage
from collections import OrderedDict
from dataclasses import dataclass
from dataclasses import dataclass, field
from typing import Union, Dict, List

import matplotlib
import seaborn

import matplotlib.pyplot as mpl
import matplotlib.colors
import matplotlib
import matplotlib.collections as collections
from matplotlib.patches import Wedge
from matplotlib.collections import PatchCollection
from matplotlib import colors as mcolors
import matplotlib.cm


import pylibrary.plotting.plothelpers as PH
import pylibrary.tools.cprint as CP

color_sequence = ["k", "r", "b"]
colormapname = "parula"

def setMapColors(colormapname: str, reverse: bool = False) -> object:
    cmnames = dir(matplotlib.cm)
    cmnames = [c for c in cmnames if not c.startswith("__")]
    if colormapname == "parula":
        colormapname = "snshelix"
    elif colormapname == "cubehelix":
        cm_sns = seaborn.cubehelix_palette(
            n_colors=6,
            start=0,
            rot=0.4,
            gamma=1.0,
            hue=0.8,
            light=0.85,
            dark=0.15,
            reverse=reverse,
            as_cmap=False,
        )
    elif colormapname == "snshelix":
        cm_sns = seaborn.cubehelix_palette(
            n_colors=64,
            start=3,
            rot=0.5,
            gamma=1.0,
            dark=0,
            light=1.0,
            reverse=reverse,
            as_cmap=True,
        )
    elif colormapname in cmnames:
        cm_sns = mpl.cm.get_cmap(colormapname)
    # elif colormapname == 'a':
    #     cm_sns = matplotlib.colors.LinearSegmentedColormap.from_list('option_a', colormaps.option_a.cm_data)
    # elif colormapname == 'b':
    #     cm_sns = matplotlib.colors.LinearSegmentedColormap.from_list('option_b', colormaps.option_b.cm_data)
    # elif colormapname == 'c':
    #     cm_sns = matplotlib.colors.LinearSegmentedColormap.from_list('option_c', colormaps.option_c.cm_data)
    # elif colormapname == 'd':
    #     cm_sns = matplotlib.colors.LinearSegmentedColormap.from_list('option_d', colormaps.option_d.cm_data)
    # elif colormapname == 'parula':
    #     cm_sns = matplotlib.colors.LinearSegmentedColormap.from_list('parula', colormaps.parula.cm_data)
    else:
        print(
            '(analyzemapdata) Unrecongnized color map {0:s}; setting to "snshelix"'.format(
                colormapname
            )
        )
        cm_sns = seaborn.cubehelix_palette(
            n_colors=64,
            start=3,
            rot=0.5,
            gamma=1.0,
            dark=0,
            light=1.0,
            reverse=reverse,
            as_cmap=True,
        )
    # elif colormapname == '
    return cm_sns


cm_sns = setMapColors("CMRmap")


# arc collection form https://gist.github.com/syrte/592a062c562cd2a98a83
# retrieved 10/5/2018


def wedges(
    x: Union[float, np.ndarray],
    y: Union[float, np.ndarray],
    w: Union[float, np.ndarray],
    h: Union[float, np.ndarray, None] = None,
    theta1: float = 0.0,
    theta2: float = 360.0,
    c: str = "b",
    **kwargs,
) -> object:
    """
    Make a scatter plot of Wedges.
    Parameters
    ----------
    x, y : scalar or array_like, shape (n, )
        Center of ellipses.
    w, h : scalar or array_like, shape (n, )
        Total length (diameter) of horizontal/vertical axis.
        `h` is set to be equal to `w` by default, ie. circle.

    theta1 : float
        start angle in degrees
    theta2 : float
        end angle in degrees
    c : color or sequence of color, optional, default : 'b'
        `c` can be a single color format string, or a sequence of color
        specifications of length `N`, or a sequence of `N` numbers to be
        mapped to colors using the `cmap` and `norm` specified via kwargs.
        Note that `c` should not be a single numeric RGB or RGBA sequence
        because that is indistinguishable from an array of values
        to be colormapped. (If you insist, use `color` instead.)
        `c` can be a 2-D array in which the rows are RGB or RGBA, however.
    kwargs : `~matplotlib.collections.Collection` properties
        Eg. alpha, edgecolor(ec), facecolor(fc), linewidth(lw), linestyle(ls),
        norm, cmap, transform, etc.
    Returns
    -------
    paths : `~matplotlib.collections.PathCollection`
    Examples
    --------
    plt.figure()
    x = np.arange(20)
    y = np.arange(20)
    arcs(x, y, 3, h=x, c = x, rot=0., theta1=0., theta2=35.)
    plt.show()

    License
    --------
    This code is under [The BSD 3-Clause License]
    (http://opensource.org/licenses/BSD-3-Clause)
    """
    if np.isscalar(c):
        kwargs.setdefault("color", c)
        c = None

    if "fc" in kwargs:
        kwargs.setdefault("facecolor", kwargs.pop("fc"))
    if "ec" in kwargs:
        kwargs.setdefault("edgecolor", kwargs.pop("ec"))
    if "ls" in kwargs:
        kwargs.setdefault("linestyle", kwargs.pop("ls"))
    if "lw" in kwargs:
        kwargs.setdefault("linewidth", kwargs.pop("lw"))
    # You can set `facecolor` with an array for each patch,
    # while you can only set `facecolors` with a value for all.

    if h is None:
        h = w

    zipped = np.broadcast(x, y, w, h, theta1, theta2)
    patches = [Wedge((x_, y_), w_, t1_, t2_) for x_, y_, w_, h_, t1_, t2_ in zipped]

    collection = PatchCollection(patches, **kwargs)

    if c is not None:
        c = np.broadcast_to(c, zipped.shape).ravel()
        collection.set_array(c)
        collection.set_clim(vmin, vmax)
    return collection


def testplot():
    """
    Note: This cannot be used if we are running in multiprocessing mode - will throw an error
    """
    from pyqtgraph.Qt import QtGui, QtCore
    import pyqtgraph as pg

    pg.setConfigOption("leftButtonPan", False)
    app = QtGui.QApplication([])
    win = pg.GraphicsLayoutWidget(show=True, title="Basic plotting examples")
    win.resize(1000, 600)
    win.setWindowTitle("pyqtgraph example: Plotting")
    # Enable antialiasing for prettier plots
    pg.setConfigOptions(antialias=True)
    p1 = win.addPlot(0, 0)
    p2 = win.addPlot(1, 0)
    p3 = win.addPlot(2, 0)
    p4 = win.addPlot(3, 0)
    p1r = win.addPlot(0, 1)
    lx = np.linspace(np.min(crosstalk[ifitx]), np.max(crosstalk[ifitx]), 50)
    sp = pg.ScatterPlotItem(
        crosstalk[ifitx], avgdf[ifitx]
    )  # plot regression over points
    ly = scf * lx + intcept
    p1r.addItem(sp)
    p1r.plot(lx, ly, pen=pg.mkPen("r", width=0.75))
    for i in range(10):
        p1.plot(self.Data.tb, data[0, i, :] + 2e-11 * i, pen=pg.mkPen("r"))
        p1.plot(self.Data.tb, datar[0, i, :] + 2e-11 * i, pen=pg.mkPen("g"))

    p1.plot(self.Data.tb, lbr, pen=pg.mkPen("c"))
    p2.plot(self.Data.tb, crosstalk, pen=pg.mkPen("m"))
    p2.plot(self.Data.tb, lbr, pen=pg.mkPen("c"))
    p2.setXLink(p1)
    p3.setXLink(p1)
    p3.plot(self.Data.tb, avgdf, pen=pg.mkPen("w", width=1.0))  # original
    p3.plot(self.Data.tb, olddatar, pen=pg.mkPen("b", width=1.0))  # original
    meandata = np.mean(datar[0], axis=0)
    meandata -= np.mean(meandata[0 : int(0.020 / ct_SR)])
    p3.plot(self.Data.tb, meandata, pen=pg.mkPen("y"))  # fixed
    p3sp = pg.ScatterPlotItem(
        self.Data.tb[tpts],
        meandata[tpts],
        pen=None,
        symbol="o",
        pxMode=True,
        size=3,
        brush=pg.mkBrush("r"),
    )  # points corrected?
    p3.addItem(p3sp)
    p4.plot(self.Data.tb[:-1], diff_avgd, pen=pg.mkPen("c"))
    p2.setXLink(p1)
    p3.setXLink(p1)
    p4.setXLink(p1)
    if (sys.flags.interactive != 1) or not hasattr(QtCore, "PYQT_VERSION"):
        QtGui.QApplication.instance().exec_()
    exit()
    #return datar, avgd

class PlotMapData():
    def __init__(self, verbose=False):
        self.rasterized = False
        self.verbose = verbose
        self.reset_flags()

    def reset_flags(self):
        self.plotted_em = {'histogram': False, 'stack': False, 'avgevents': False, 'avgax': [0, None, None]}

    def set_Pars_and_Data(self, pars, data):
        """
        save parameters passed from analyze Map Data
        Analysis parameters and data are in separae data classes.
        """
        self.Pars = pars
        self.Data = data

    def show_images(self, show: bool = True) -> None:
        r, c = PH.getLayoutDimensions(len(self.images), pref="height")
        f, ax = mpl.subplots(r, c)
        self.figure_handle = f
        f.suptitle(self.celldir.replace(r"_", r"\_"), fontsize=9)
        ax = ax.ravel()
        PH.noaxes(ax)
        for i, img in enumerate(self.imagedata):
            fna, fne = os.path.split(self.images[i])
            imfig = ax[i].imshow(self.gamma_correction(img, 2.2))
            PH.noaxes(ax[i])
            ax[i].set_title(fne.replace(r"_", r"\_"), fontsize=8)
            imfig.set_cmap(self.cmap)
        if show:
            mpl.show()

    def scale_and_rotate(
        self,
        poslist: list,
        sign: list = [1.0, 1.0],
        scaleCorr: float = 1.0,
        scale: float = 1e6,
        autorotate: bool = False,
        angle: float = 0.0,
    ) -> np.ndarray:
        """
        Angle is in radians
        """
        poslist = [
            tuple(
                [sign[0] * p[0] * scale * scaleCorr, sign[1] * p[1] * scale * scaleCorr]
            )
            for p in poslist
        ]
        posl = np.array(poslist, dtype=[("x", float), ("y", float)]).view(np.recarray)

        newpos = np.array(poslist)

        # get information to do a rotation to horizontal for positions.
        if autorotate:
            iy = np.argsort(posl.y)
            y = posl.y[iy[-3:]]
            x = posl.x[iy[-3:]]
            theta = np.arctan2((y[1] - y[0]), (x[1] - x[0]))
            # perform rotation around 0 using -theta to flatten to top of the array
            c, s = np.cos(-theta), np.sin(-theta)
            rmat = np.matrix([[c, -s], [s, c]])  # rotation matrix
            newpos = np.dot(rmat, newpos.T).T
        if not autorotate and angle != 0.0:
            theta = angle
            c, s = np.cos(-theta), np.sin(-theta)
            rmat = np.matrix([[c, -s], [s, c]])  # rotation matrix
            newpos = np.dot(rmat, newpos.T).T
        return newpos
    
    def plot_timemarker(self, ax: object) -> None:
        """
        Plot a vertical time line marker for the stimuli
        """
        yl = ax.get_ylim()
        for j in range(len(self.Pars.stimtimes["start"])):
            t = self.Pars.stimtimes["start"][j]
            if isinstance(t, float) and np.diff(yl) > 0:  # check that plot is ok to try
                ax.plot(
                    [t, t],
                    yl,
                    "b-",
                    linewidth=0.5,
                    alpha=0.6,
                    rasterized=self.rasterized,
                )

    def plot_hist(self, axh: object, results: dict, colorid: int = 0) -> None:
        """
        Plot  the histogram of event times
        hist goes into axh

        """
        CP.cprint("c", "    Plot_hist")

        # assert not self.plotted_em['histogram']
        self.plotted_em['histogram'] = True
        plotevents = True
        rotation = 0.0
        plotFlag = True
        idn = 0
        self.newvmax = None
        eventtimes = []
        events = results["events"]
        if events[0] == None:
            CP.cprint("r", "**** plot_hist: no events were found")
            return
        rate = results["rate"]
        tb0 = events[0]["aveventtb"]  # get from first trace in first trial
        # rate = np.mean(np.diff(tb0))
        nev = 0  # first count up events
        for itrial in events.keys():
            for j, jtrace in enumerate(events[itrial]['onsets']):
                nev += len(jtrace)
        eventtimes = np.zeros(nev)
        iev = 0
        for itrial in events.keys():
            for j, onsets in enumerate(events[itrial]['onsets']):
                ntrialev = len(onsets)
                eventtimes[iev : iev + ntrialev] = onsets
                iev += ntrialev
        CP.cprint("c", 
            f"    plot_hist:: total events: {iev:5d}  # event times: {len(eventtimes):5d}  Sample Rate: {1e6*rate:6.1f} usec")

        if plotevents and len(eventtimes) > 0:
            nevents = 0
            y = np.array(eventtimes) * rate
            # print('AR Tstart: ', self.AR.tstart, y.shape)
            bins = np.linspace(
                0.0, self.Pars.ar_tstart, int(self.Pars.ar_tstart * 1000.0 / 2.0) + 1
            )
            axh.hist(
                y,
                bins=bins,
                facecolor="k",
                edgecolor="k",
                linewidth=0.5,
                histtype="stepfilled",
                align="right",
            )
            self.plot_timemarker(axh)
            PH.nice_plot(
                axh,
                spines=["left", "bottom"],
                position=-0.025,
                direction="outward",
                axesoff=False,
            )
            axh.set_xlim(0.0, self.Pars.ar_tstart - 0.005)

    def plot_stacked_traces(
        self,
        tb: np.ndarray,
        mdata: np.ndarray,
        title: str,
        results: dict,
        ax: Union[object, None] = None,
        trsel: Union[int, None] = None,
    ) -> None:

        # assert not self.plotted_em['stack']
        CP.cprint("c", "    Starting stack plot")
        linewidth=0.35  # base linewidth
        self.plotted_em['stack'] = True
        if ax is None:
            f, ax = mpl.subplots(1, 1)
            self.figure_handle = f
        events = results["events"]
        # print('event keys: ', events.keys())
        nevtimes = 0
        spont_ev_count = 0
        dt = np.mean(np.diff(self.Data.tb))
        itmax = int(self.Pars.analysis_window[1] / dt)

        if trsel is not None:
            for j in range(mdata.shape[0]):
                if tb.shape[0] > 0 and mdata[jtrial, trsel, :].shape[0] > 0:
                    ax.plot(
                        tb[:itmax],
                        mdata[0, trsel, :itmax] * self.Pars.scale_factor,
                        linewidth=0.2,
                        rasterized=False,
                        zorder=10,
                    )
            PH.clean_axes(ax)
            PH.calbar(
                ax,
                calbar=[
                    0.6,
                    -200e-12 * self.Pars.scale_factor,
                    0.05,
                    100e-12 * self.Pars.scale_factor,
                ],
                axesoff=True,
                orient="left",
                unitNames={"x": "s", "y": "pA"},
                fontsize=11,
                weight="normal",
                font="Arial",
            )
            mpl.suptitle(str(title).replace(r"_", r"\_"), fontsize=8)
            self.plot_timemarker(ax)
            ax.set_xlim(0, self.Pars.ar_tstart - 0.001)
            return
        
        crflag = [False for i in range(mdata.shape[0])]

        for itrial in range(mdata.shape[0]):
            if events[itrial] is None:
                continue
            evtr = events[itrial]['event_trace_list']  # of course it is the same for every entry.
            for itrace in range(mdata.shape[1]):
                smpki = events[itrial]["smpksindex"][itrace]
                pktimes = events[itrial]["peaktimes"][itrace]
                # print(itrace, smpki)
                if len(pktimes) > 0:
                    nevtimes += len(smpki)
                    if (
                        len(smpki) > 0
                        and len(tb[smpki]) > 0
                        and len(mdata[itrial, itrace, smpki]) > 0
                    ):
                        sd = events[itrial]["spont_dur"][itrace]
                        tsi = smpki[
                            np.where(tb[smpki] < sd)[0].astype(int)
                        ]  # find indices of spontanteous events (before first stimulus)
                        tri = np.ndarray(0)
                        for (
                            iev
                        ) in (
                            self.Pars.twin_resp
                        ):  # find events in all response windows
                            tri = np.concatenate(
                                (
                                    tri.copy(),
                                    smpki[
                                        np.where(
                                            (tb[smpki] >= iev[0])
                                            & (tb[smpki] < iev[1])
                                        )[0]
                                    ],
                                ),
                                axis=0,
                            ).astype(int)
                        ts2i = list(
                            set(smpki)
                            - set(tri.astype(int)).union(set(tsi.astype(int)))
                        )  # remainder of events (not spont, not possibly evoked)
                        ms = np.array(
                            mdata[itrial, itrace, tsi]
                        ).ravel()  # spontaneous events
                        mr = np.array(
                            mdata[itrial, itrace, tri]
                        ).ravel()  # response in window
                        if len(mr) > 0:
                            crflag[itrial] = True  # flag traces with detected responses
                        ms2 = np.array(
                            mdata[itrial, itrace, ts2i]
                        ).ravel()  # events not in spont and outside window
                        spont_ev_count += ms.shape[0]
                        cr = matplotlib.colors.to_rgba(
                            "r", alpha=0.6
                        )  # just set up color for markers
                        ck = matplotlib.colors.to_rgba("k", alpha=1.0)
                        cg = matplotlib.colors.to_rgba("gray", alpha=1.0)

                        ax.plot(
                            tb[tsi],
                            ms * self.Pars.scale_factor + self.Pars.stepi * itrace,
                            "o",
                            color=ck,
                            markersize=2,
                            markeredgecolor="None",
                            zorder=0,
                            rasterized=self.rasterized,
                        )
                        ax.plot(
                            tb[tri],
                            mr * self.Pars.scale_factor + self.Pars.stepi * itrace,
                            "o",
                            color=cr,
                            markersize=2,
                            markeredgecolor="None",
                            zorder=0,
                            rasterized=self.rasterized,
                        )
                        ax.plot(
                            tb[ts2i],
                            ms2 * self.Pars.scale_factor + self.Pars.stepi * itrace,
                            "o",
                            color=cg,
                            markersize=2,
                            markeredgecolor="None",
                            zorder=0,
                            rasterized=self.rasterized,
                        )
        for itrial in range(mdata.shape[0]):
            for itrace in range(mdata.shape[1]):            
                if tb.shape[0] > 0 and mdata[itrial, itrace, :].shape[0] > 0:
                    if crflag[itrial]:
                        alpha = 1.0
                        lw = linewidth
                    else:
                        alpha = 0.3
                        lw = linewidth*0.25
                    ax.plot(
                        tb[:itmax],
                        mdata[itrial, itrace, :itmax] * self.Pars.scale_factor
                        + self.Pars.stepi * itrace,
                        linewidth=lw,
                        rasterized=False,
                        zorder=10,
                        alpha=alpha,
                    )
        CP.cprint("c", f"        Spontaneous Event Count: {spont_ev_count:d}")

        mpl.suptitle(str(title).replace(r"_", r"\_"), fontsize=8)
        self.plot_timemarker(ax)
        ax.set_xlim(0, self.Pars.ar_tstart - 0.001)


    def get_calbar_Yscale(self, amp: float):
        """
        Pick a scale for the calibration bar based on the amplitude to be represented
        """
        sc = [
            10.0,
            20.0,
            50.0,
            100.0,
            200.0,
            400.0,
            500.0,
            1000.0,
            1500.0,
            2000.0,
            2500.0,
            3000.0,
            5000.0,
            10000.0,
            15000.0,
            20000.0,
        ]
        a = amp
        if a < sc[0]:
            return sc[0]
        for i in range(len(sc) - 1):
            if a >= sc[i] and a < sc[i + 1]:
                return sc[i + 1]
        return sc[-1]

    def plot_avgevent_traces(
        self,
        evtype: str,
        mdata: Union[np.ndarray, None] = None,
        trace_tb: Union[np.ndarray, None] = None,
        events: Union[dict, None] = None,
        ax: Union[object, None] = None,
        scale: float = 1.0,
        label: str = "pA",
        rasterized: bool = False,
    ) -> None:
        # ensure we don't plot more than once...
        # CP.cprint('y', f"start avgevent plot for  {evtype:s}, ax={str(ax):s}")
        # assert not self.plotted_em['avgevents']
        if self.plotted_em['avgax'][0] == 0:
            self.plotted_em['avgax'][1] = ax
        elif self.plotted_em['avgax'][0] == 1:
            if self.plotted_em['avgax'][1] == ax:
                raise ValueError('plot_avgevent_traces : repeated into same axis')
            else:
                self.plotted_em['avgax'][2] = ax
                self.plotted_em['avgevents'] = True
        # CP.cprint('c', f"plotting avgevent plot for  {evtype:s}, ax={str(ax):s}")

        # self.plotted_em['avgevents'] = True
        if events is None or ax is None or trace_tb is None:
            CP.cprint("r", f"[plot_avgevent_traces]:: evtype: {evtype:s}. No events, no axis, or no time base")
            return
        nevtimes = 0
        line = {"avgevoked": "k-", "avgspont": "k-"}
        ltitle = {"avgevoked": "Evoked (%s)" % label, "avgspont": "Spont (%s)" % label}
        result_names = {"avgevoked": "evoked_ev", "avgspont": "spont_ev"}

        ax.set_ylabel(ltitle[evtype])
        ax.spines["left"].set_color(line[evtype][0])
        ax.yaxis.label.set_color(line[evtype][0])
        ax.tick_params(axis="y", colors=line[evtype][0], labelsize=7)
        ax.tick_params(axis="x", colors=line[evtype][0], labelsize=7)
        ev_min = 5e-12
        sp_min = 5e-12
        if evtype == "avgevoked":
            eventmin = ev_min
        else:
            eventmin = sp_min
        ave = []
        # compute average events and time bases
        minev = 0.0
        maxev = 0.0
        # for trial in events.keys():
        spont_ev_count = 0
        evoked_ev_count = 0
        npev = 0
        # tau1 = self.Pars.MA.fitresult[1]  # get rising tau so we can make a logical tpre
        # print("plotting # trials = ", mdata.shape[0])
        for trial in range(mdata.shape[0]):
            if self.verbose:
                print("plotting events for trial: ", trial)
            if events[trial] is None or len(events[trial]['aveventtb']) == 0:
                # print(trial, evtype, result_names[evtype])
                CP.cprint("r", "**** plot_avgevent_traces: no events....")
                continue
            tb0 = events[trial]["aveventtb"]  # get from first trace
            
            rate = np.mean(np.diff(tb0))
            tpre = 0.1 * np.max(tb0)
            tpost = np.max(tb0)
            ipre = int(tpre / rate)
            ipost = int(tpost / rate)
            tb = np.arange(-tpre, tpost + rate, rate) + tpre
            ptfivems = int(0.0005 / rate)
            pwidth = int(0.0005 / rate / 2.0)
            # for itrace in events[trial].keys():  # traces in the evtype list
            for itrace in range(mdata.shape[1]):  # traces in the evtype list
                if events is None or trial not in list(events.keys()):
                    if self.verbose:
                        print(f"     NO EVENTS in trace: {itrace:4d}")
                    continue
                evs = events[trial][result_names[evtype]][itrace]
                if len(evs) == 0:  # skip trace if there are NO events
                    if self.verbose:
                        print(
                            f"     NO EVENTS of type {evtype:10s} in trace: {itrace:4d}"
                        )
                    continue
                spont_dur = events[trial]["spont_dur"][itrace]
                for jevent in evs[0]:  # evs is 2 element array: [0] are onsets and [1] is peak; this aligns to onsets
                    if evtype == "avgspont":
                        spont_ev_count += 1
                        if (
                            trace_tb[jevent] + self.Pars.spont_deadtime > spont_dur
                        ):  # remove events that cross into stimuli
                            if self.verbose:
                                print(
                                    f"     Event {jevent:6d} in trace {itrace:4d} crosses into stimulus"
                                )
                            continue
                    if evtype == "avgevoked":
                        evoked_ev_count += 1
                        if trace_tb[jevent] <= spont_dur:  # only post events
                            if self.verbose:
                                print(
                                    f"     Event in spont window, not plotting as evoked: {jevent:6d} [t={float(jevent*rate):8.3f}] trace: {itrace:4d}"
                                )
                            continue
                    # print('ipre: ', ipre, itrace, evtype)
                    if (
                        jevent - ipre < 0
                    ):  # event to be plotted would go before start of trace
                        if self.verbose:
                            print(
                                f"     Event starts too close to trace start {jevent:6d} trace: {itrace:4d}"
                            )
                        continue
                    evdata = mdata[
                        trial, itrace, jevent - ipre : jevent + ipost
                    ].copy()  # 0 is onsets
                    bl = np.mean(evdata[0 : ipre - ptfivems])
                    evdata -= bl
                    if self.verbose:
                        print(
                            "Len EVDATA: ",
                            len(evdata),
                            " evdata 0:10",
                            np.mean(evdata[:10]),
                            " evtype: ",
                            evtype,
                        )
                    if len(evdata) > 0:
                        ave.append(evdata)
                        npev += 1
                        # and only plot when there is data, otherwise matplotlib complains with "negative dimension are not allowed" error
                        if self.verbose:
                            CP.cprint("green", "   Plotting")
                        ax.plot(
                            tb[: len(evdata)] * 1e3,
                            scale * evdata,
                            line[evtype],
                            linewidth=0.15,
                            alpha=0.25,
                            rasterized=False,
                        )
                        minev = np.min([minev, np.min(scale * evdata)])
                        maxev = np.max([maxev, np.max(scale * evdata)])
            # print(f"      {evtype:s} Event Count in AVERAGE: {spont_ev_count:d}, len ave: {len(ave):d}")

        # print('evtype: ', evtype, '  nev plotted: ', npev, ' nevoked: ', evoked_ev_count)
        # print('maxev, minev: ', maxev, minev)
        nev = len(ave)
        aved = np.asarray(ave)
        if (len(aved) == 0) or (aved.shape[0] == 0) or (nev == 0):
            return
        if self.verbose:
            CP.cprint("red", f"aved shape is {str(aved.shape):s}")
            return
        tx = np.broadcast_to(tb, (aved.shape[0], tb.shape[0])).T
        if self.Pars.sign < 0:
            maxev = -minev
        self.Pars.MA.set_sign(self.Pars.sign)
        self.Pars.MA.set_dt_seconds(rate)
        avedat = np.mean(aved, axis=0)
        tb = tb[: len(avedat)]
        avebl = np.mean(avedat[:ptfivems])
        avedat = avedat - avebl
        
        CP.cprint('c', '    plotmapdata: Fitting average event')
        self.Pars.MA.fit_average_event(
            tb,
            avedat,
            debug=False,
            label="Map average",
            inittaus=self.Pars.taus,
            initdelay=tpre,
        )
        CP.cprint('c', '        Event fitting completed')
        Amplitude = self.Pars.MA.fitresult.values['amp']
        tau1 = self.Pars.MA.fitresult.values['tau_1']
        tau2 = self.Pars.MA.fitresult.values['tau_2']
        bfdelay = self.Pars.MA.fitresult.values['fixed_delay']
        bfit = self.Pars.MA.avg_best_fit
        if self.Pars.sign == -1:
            amp = np.min(bfit)
        else:
            amp = np.max(bfit)
        txt = f"Amp: {scale*amp:.1f} tau1:{1e3*tau1:.2f} tau2: {1e3*tau2:.2f} (N={aved.shape[0]:d})"
        if evtype == "avgspont":
            srate = float(aved.shape[0]) / (
                events[0]["spont_dur"][0] * mdata.shape[1]
            )  # dur should be same for all trials
            txt += f" SR: {srate:.2f} Hz"
        ax.text(0.05, 0.95, txt, fontsize=7, transform=ax.transAxes)
        # ax.plot(tx, scale*ave.T, line[evtype], linewidth=0.1, alpha=0.25, rasterized=False)
        ax.plot(
            tb * 1e3,
            scale * bfit,
            "c",
            linestyle="-",
            linewidth=0.35,
            rasterized=self.rasterized,
        )
        ax.plot(
            tb * 1e3,
            scale * avedat,
            line[evtype],
            linewidth=0.625,
            rasterized=self.rasterized,
        )

        # ax.set_label(evtype)
        ylims = ax.get_ylim()
        if evtype == "avgspont":
            PH.calbar(
                ax,
                calbar=[
                    np.max(tb) - 2.0,
                    ylims[0],
                    2.0,
                    self.get_calbar_Yscale(np.fabs(ylims[1] - ylims[0]) / 4.0),
                ],
                axesoff=True,
                orient="left",
                unitNames={"x": "ms", "y": label},
                fontsize=11,
                weight="normal",
                font="Arial",
            )
        elif evtype == "avgevoked":
            PH.calbar(
                ax,
                calbar=[
                    np.max(tb) - 2.0,
                    ylims[0],
                    2.0,
                    self.get_calbar_Yscale(maxev / 4.0),
                ],
                axesoff=True,
                orient="left",
                unitNames={"x": "ms", "y": label},
                fontsize=11,
                weight="normal",
                font="Arial",
            )

    def plot_average_traces(
        self,
        ax: object,
        tb: Union[np.ndarray, None],
        mdata: Union[np.ndarray, None],
        color: str = "k",
    ) -> None:
        """
        Average the traces

        Parameters
        ----------
        ax : matplotlib axis object
            axis to plot into
        tb : float array (list or numpy)
            time base for the plot
        mdata : float 2d numpy array
            data to be averaged. Must be trace number x trace
        color : str
            color for trace to be plotter

        """
        if mdata is None:
            return
        while mdata.ndim > 1:
            mdata = mdata.mean(axis=0)
        if len(tb) > 0 and len(mdata) > 0:
            ax.plot(
                tb * 1e3,
                mdata * self.Pars.scale_factor,
                color,
                rasterized=self.rasterized,
                linewidth=0.6,
            )
        ax.set_xlim(0.0, self.Pars.ar_tstart * 1e3 - 1.0)
        return

    def clip_colors(self, cmap, clipcolor):
        cmax = len(cmap)
        colmax = cmap[cmax - 1]
        for i in range(cmax):
            if (cmap[i] == colmax).all():
                cmap[i] = clipcolor
        return cmap

    def plot_photodiode(self, ax, tb, pddata, color="k"):
        if len(tb) > 0 and len(np.mean(pddata, axis=0)) > 0:
            ax.plot(
                tb,
                np.mean(pddata, axis=0),
                color,
                rasterized=self.rasterized,
                linewidth=0.6,
            )
        ax.set_xlim(0.0, self.Pars.ar_tstart - 0.001)

    def plot_map(
        self,
        axp,
        axcbar,
        pos,
        measure,
        measuretype="I_max",
        vmaxin=None,
        imageHandle=None,  # could be a montager instance
        imagefile=None,
        angle=0,
        spotsize=42e-6,
        cellmarker=False,
        whichstim=-1,
        average=False,
        pars=None,
    ):

        sf = 1.0  # could be 1e-6 ? data in Meters? scale to mm.
        cmrk = 50e-6 * sf  # size, microns
        linewidth = 0.2
        npos = pos.shape[0]
        npos += 1  # need to count up one more to get all of the points in the data
        pos = pos[:npos, :] * 1e3  # clip unused positions

        pz = [np.mean(pos[:, 0]), np.mean(pos[:, 1])]
        if imageHandle is not None and imagefile is not None:
            imageInfo = imageHandle.imagemetadata[0]
            # compute the extent for the image, offsetting it to the map center position
            ext_left = imageInfo["deviceTransform"]["pos"][0] * 1e3  # - pz[0]
            ext_right = (
                ext_left
                + imageInfo["region"][2]
                * imageInfo["deviceTransform"]["scale"][0]
                * 1e3
            )
            ext_bottom = imageInfo["deviceTransform"]["pos"][1] * 1e3  # - pz[1]
            ext_top = (
                ext_bottom
                + imageInfo["region"][3]
                * imageInfo["deviceTransform"]["scale"][1]
                * 1e3
            )
            ext_left, ext_right = self.reorder(ext_left, ext_right)
            ext_bottom, ext_top = self.reorder(ext_bottom, ext_top)
            # extents are manually adjusted - something about the rotation should be computed in them first...
            # but fix it manually... worry about details later.
            # yellow cross is aligned on the sample cell for this data now
            extents = [ext_bottom, ext_top, ext_left, ext_right]  #
            extents = [ext_left, ext_right, ext_bottom, ext_top]
            img = imageHandle.imagedata[0]
            if (
                angle != 0.0
            ):  # note that angle arrives in radians - must convert to degrees for this one.
                img = scipy.ndimage.interpolation.rotate(
                    img,
                    angle * 180.0 / np.pi + 90.0,
                    axes=(1, 0),
                    reshape=True,
                    output=None,
                    order=3,
                    mode="constant",
                    cval=0.0,
                    prefilter=True,
                )
            axp.imshow(
                img,
                aspect="equal",
                extent=extents,
                origin="lower",
                cmap=setMapColors("gray"),
            )

        spotsize = 1e3 * spotsize
        # print(measure.keys())
        # print(measuretype)
        if measuretype in ["A", "Q"]:
            mtype = "I_max"
        else:
            mtype = measuretype
        if whichstim < 0:
            spotsizes = spotsize * np.linspace(1.0, 0.2, len(measure[mtype]))
        else:
            spotsizes = spotsize * np.ones(len(measure[measuretype]))
        pos = self.scale_and_rotate(pos, scale=1.0, angle=angle)
        xlim = [np.min(pos[:, 0]) - spotsize, np.max(pos[:, 0]) + spotsize]
        ylim = [np.min(pos[:, 1]) - spotsize, np.max(pos[:, 1]) + spotsize]
        # if vmaxin is not None:
        #     vmax = vmaxin  # fixed
        # else:
        sign = measure["sign"]

        upscale = 1.0
        if measuretype == "ZScore":
            data = measure[measuretype]

            vmax = np.max(np.max(measure[measuretype]))
            vmin = np.min(np.min(measure[measuretype]))
            # print('vmax: ', vmin, vmax)
            if vmax < 6.0:
                vmax = 6.0  # force a fixed
            scaler = PH.NiceScale(0.0, vmax)
            vmax = scaler.niceMax

        elif measuretype == "Qr":
            data = sign * (measure["Qr"] - measure["Qb"])
            vmin = 0.0
            vmax = np.max(data)

        elif measuretype in ["A", "Q"]:
            events = measure["events"]
            nspots = len(measure["events"][0])  # on trial 0
            if 'npulses' in list(measure['stimtimes'].keys()):
                npulses = measure["stimtimes"]["npulses"][0]
            else:
                npulses =  len(measure["stimtimes"]["start"])

            # data = np.zeros((measure['ntrials'], npulses, nspots))
            data = np.zeros((npulses, nspots))
            if measure["stimtimes"] is not None:
                twin_base = [
                    0.0,
                    measure["stimtimes"]["start"][0] - 0.001,
                ]  # remember times are in seconds
                twin_resp = []
                stims = measure["stimtimes"]["start"]
                for j in range(len(stims)):
                    twin_resp.append(
                        [stims[j] + self.Pars.direct_window, stims[j] + self.Pars.response_window]
                    )

            rate = measure["rate"]
            nev = 0
            nev_spots = 0
            # print("event measures: ", events[0][0]["measures"])
            for trial in range(measure["ntrials"]):
                for spot in range(nspots):  # for each spot
                    for ipulse in range(npulses):
                        if measuretype == "Q":
                            if len(events[trial][spot]["measures"][0]["Q"]) > ipulse:  # there is only one, the first trial
                                data[ipulse, spot] = events[trial][spot]["measures"][
                                    0
                                ]["Q"][ipulse]
                            continue
                        smpki = events[trial][spot]["smpksindex"][0]
                        # print('smpki: ', smpki)
                        tri = np.ndarray(0)
                        tev = twin_resp[ipulse]  # go through stimuli
                        iev0 = int(tev[0] / rate)
                        iev1 = int(tev[1] / rate)
                        tri = np.concatenate(
                            (
                                tri.copy(),
                                smpki[np.where((smpki >= iev0) & (smpki <= iev1))[0]],
                            ),
                            axis=0,
                        ).astype(int)

                        smpki = list(smpki)
                        for t in range(len(tri)):
                            if tri[t] in smpki:
                                r = smpki.index(tri[t])  # find index
                                data[ipulse, spot] += (
                                    sign * events[trial][spot]["smpks"][0][r]
                                )
                                nev_spots += 1

            vmin = 0.0
            vmax = np.max(data)
            nev = len(np.where(data > 0.0)[0])
            # print(nev, nev_spots)

            if vmax == 0:
                vmax = 1e-10
        # print("vmax: ", vmax, "vmin : ", vmin)

        scaler = PH.NiceScale(vmin, vmax)

        if whichstim >= 0:  # which stim of -1 is ALL stimuli
            whichmeasures = [whichstim]
        elif average:
            whichmeasures = [0]
            data[0] = np.mean(data)  # just
        else:
            # print("else... ", len(data))
            whichmeasures = range(len(data))
        # print(whichstim, whichmeasures, average)
        norm = matplotlib.colors.Normalize(vmin=0, vmax=vmax)
        # red_rgba = matplotlib.colors.to_rgba('r')
        # black_rgba = matplotlib.colors.to_rgba('k')
        cmx = matplotlib.cm.ScalarMappable(norm=norm, cmap=cm_sns)
        axp.set_facecolor([0.75, 0.75, 0.75])
        for (
            im
        ) in (
            whichmeasures
        ):  # there may be multiple measure[measuretype]s (repeated stimuli of different kinds) in a map
            # note circle size is radius, and is set by the laser spotsize (which is diameter)
            # print("spotsizes size: ", spotsizes.shape)
            # print("im: ", im)
            # print("spotsizes: ", spotsizes)
            # print("whichmeasures: ", whichmeasures)
            # print( ", measuretype)
            # print("reps: ", self.nreps)
            radw = np.ones(pos.shape[0]) * spotsizes[im]
            radh = np.ones(pos.shape[0]) * spotsizes[im]
            if measuretype in ["Qr", "Imax"]:
                spotcolors = cmx.to_rgba(np.clip(data[im], 0.0, vmax))
            else:  # zscore, no sign.
                spotcolors = cmx.to_rgba(np.clip(data[im], 0.0, vmax))
            edgecolors = spotcolors.copy()
            for i in range(len(data[im])):
                em = data[im][i]
                if measuretype == "ZScore" and em < 1.96:
                    spotcolors[i][3] = em / 1.96  # scale down
                edgecolors[i] = matplotlib.colors.to_rgba([0.6, 0.6, 0.6, 0.5])
                # edgecolors[i] = matplotlib.colors.to_rgba([0.2, 0.2, 0.2, 0.5])
                #     print(' .. ', spotcolors[i])
            # print('spotcolors, edgecolors: ', spotcolors, edgecolors)
            order = np.argsort(
                data[im]
            )  # plot from smallest to largest (so largest on top)
            # for i, p in enumerate(pos[order]):  # just checking - it appears in one map.
            #     if p[0] > 55.2 and p[1] < 3.45:
            #         print('wayward: ', i, p)
            #   colors = self.clip_colors(colors, [1., 1., 1., 1.])
            if self.nreps == 1:
                # concentric circles (overlaid ellipses) showing the response measure for
                # each stimulus. Organization is outside-in first to last stimulus
                ec = collections.EllipseCollection(
                    radw,
                    radh,
                    np.zeros_like(radw),
                    offsets=pos[order],
                    units="xy",
                    transOffset=axp.transData,
                    facecolor=spotcolors[order],
                    edgecolor=edgecolors[order],
                    linewidth=0.02,
                )
                axp.add_collection(ec)
                # for o in order:
                #     print('m: ', measure[measuretype][im][o]/vmax, spotcolors[o])
            else:
                # make arcs within the circle, each arc is for different trial
                # these were averaged across repetitions (see Zscore, Q, etc above), so nreps is 1
                # maybe later don't average and store ZScore per map trial.
                # print(self.nreps)
                nrep = 1
                ic = 0
                npos = pos.shape[0]
                dtheta = 360.0 / nrep
                ri = 0
                rs = int(npos / nrep)
                for nr in range(nrep):
                    ec = wedges(
                        pos[ri : (ri + rs), 0],
                        pos[ri : (ri + rs), 1],
                        radw[ri : (ri + rs)] / 2.0,
                        theta1=nr * dtheta,
                        theta2=(nr + 1) * dtheta,
                        color=spotcolors[ri : ri + rs],
                    )
                    axp.add_collection(ec)
                    ri += rs
        if cellmarker:
            CP.cprint("yellow", "Cell marker is plotted")
            axp.plot(
                [-cmrk, cmrk], [0.0, 0.0], "-", color="r"
            )  # cell centered coorinates
            axp.plot(
                [0.0, 0.0], [-cmrk, cmrk], "-", color="r"
            )  # cell centered coorinates

        tickspace = scaler.tickSpacing
        try:
            ntick = 1 + int(vmax / tickspace)
        except:
            ntick = 3
        ticks = np.linspace(0, vmax, num=ntick, endpoint=True)

        # PH.talbotTicks(
        #     axp, tickPlacesAdd={"x": 1, "y": 1}, floatAdd={"x": 2, "y": 2})

        if axcbar is not None:
            c2 = matplotlib.colorbar.ColorbarBase(
                axcbar, cmap=cm_sns, ticks=ticks, norm=norm
            )
            if measuretype == "ZScore":
                c2.ax.plot([0, 10], [1.96, 1.96], "w-")
            c2.ax.tick_params(axis="y", direction="out")
            # PH.talbotTicks(
            #     c2.ax, tickPlacesAdd={"x": 0, "y": 0}, floatAdd={"x": 0, "y": 0}
            # )
        # axp.scatter(pos[:,0], pos[:,1], s=2, marker='.', color='k', zorder=4)
        axr = 250.0
        axp.set_facecolor([0.0, 0.0, 0.0])
        axp.set_xlim(xlim)
        axp.set_ylim(ylim)
        if imageHandle is not None and imagefile is not None:
            axp.set_aspect("equal")
        axp.set_aspect("equal")
        title = measuretype.replace(r"_", r"\_")
        if whichstim >= 0:
            title += f", Stim \# {whichstim:d} Only"
        if average:
            title += ", Average"
        # axp.set_title(title)
        if vmaxin is None:
            return vmax
        else:
            return vmaxin

    def display_position_maps(
        self, dataset_name: Union[Path, str], result: dict, pars:object=None,
    ) -> bool:

        measures = ["ZScore", "Qr-Qb", "A", "Q"]
        measuredict = {"ZScore": "ZScore", "Qr-Qb": "Qr", "A": "A", "Q": "Q"}
        nmaptypes = len(measures)
        rows, cols = PH.getLayoutDimensions(4)

        plabels = measures
        for i in range(rows * cols - nmaptypes):
            plabels.append(f"empty{i:d}")
            plabels[-1].replace("_", "")

        # self.P = PH.regular_grid(
        #     rows,
        #     cols,
        #     order='rowsfirst',
        #     figsize=(10, 10),
        #     panel_labels=plabels,
        #     labelposition=(0.05, 0.95),
        #     margins={
        #         "leftmargin": 0.07,
        #         "rightmargin": 0.05,
        #         "topmargin": 0.12,
        #         "bottommargin": 0.1,
        #     },
        # )

        self.plotspecs = OrderedDict(
            [
                ("ZScore", {"pos": [0.07, 0.3, 0.55, 0.3], "labelpos": (0.5, 1.05)}),
                (
                    "ZScore-Cbar",
                    {"pos": [0.4, 0.012, 0.55, 0.3], "labelpos": (0, 1.05)},
                ),  # scale bar
                ("Qr-Qb", {"pos": [0.52, 0.3, 0.55, 0.3], "labelpos": (0.5, 1.05)}),
                (
                    "Qr-Qb-Cbar",
                    {"pos": [0.85, 0.012, 0.55, 0.3], "labelpos": (0, 1.05)},
                ),
                ("A", {"pos": [0.07, 0.3, 0.065, 0.3], "labelpos": (0.5, 1.05)}),
                ("A-Cbar", {"pos": [0.4, 0.012, 0.05, 0.3], "labelpos": (0, 1.05)}),
                ("Q", {"pos": [0.52, 0.3, 0.065, 0.3], "labelpos": (0.5, 1.05)}),
                ("Q-Cbar", {"pos": [0.85, 0.012, 0.05, 0.3], "labelpos": (0, 1.05)}),
                # ('E', {'pos': [0.47, 0.45, 0.05, 0.85]}),
                #     ('F', {'pos': [0.47, 0.45, 0.10, 0.30]}),
            ]
        )  # a1 is cal bar
        spotsize = 42e-6
        self.P = PH.Plotter(self.plotspecs, label=True, figsize=(10.0, 8.0))
        self.nreps = result["ntrials"]

        for measure in measures:
            if measure.startswith("empty"):
                continue
            cbar = self.P.axdict[f"{measure:s}-Cbar"]
            # print("\nmeasure: ", measure)
            self.plot_map(
                self.P.axdict[measure],
                cbar,
                result["positions"],
                measure=result,
                measuretype=measuredict[measure],
                pars=pars
            )
        return True

    def display_one_map(
        self,
        dataset,
        results,
        imagefile=None,
        rotation=0.0,
        measuretype=None,
        plotevents=True,
        rasterized=False,
        whichstim=-1,
        average=False,
        trsel=None,
        plotmode="document",
    ) -> bool:
        if results is None or self.Pars.datatype is None:
            CPcprint("r", f"NO Results in the call, from {str(dataset.name):s}")
            return

        if (
            "_IC" in str(dataset.name)
            or "_CC" in str(dataset.name)
            or self.Pars.datatype == "I"
        ):
            scf = 1e3
            label = "mV"  # mV
        elif (
            ("_VC" in str(dataset.name))
            or ("VGAT_5ms" in str(dataset.name))
            or ("_WCChR2" in str(dataset.name))
            or (self.Pars.datatype == "V")
        ):
            scf = 1e12  # pA, vc
            label = "pA"
        else:
            scf = 1.0
            label = "AU"
        # f, ax = mpl.subplots(2,2)
        #         ax = ax.ravel()
        #         for k, s in enumerate(['I_max', 'ZScore', 'Qr', 'Qb']):
        #             for i in results[s]:
        #                 ax[k].scatter(np.arange(len(i)), i, s=5)
        #                 ax[k].set_title(s)
        #             if s in ['Qb', 'Qr']:
        #                 ax[k].set_ylim(-0.20, np.max(i))
        #         mpl.show()

        # build a figure
        self.nreps = results['ntrials']

        l_c1 = 0.1  # column 1 position
        l_c2 = 0.50  # column 2 position
        trw = 0.32  # trace x width
        trh = 0.10  # trace height
        imgw = 0.25  # image box width
        imgh = 0.25
        trs = imgh - trh  # 2nd trace position (offset from top of image box)
        y = 0.08 + np.arange(0.0, 0.7, imgw + 0.05)  # y positions
        self.mapfromid = {0: ["A", "B", "C"], 1: ["D", "E", "F"], 2: ["G", "H", "I"]}
        if plotmode == "document":
            self.plotspecs = OrderedDict(
                [
                    ("A", {"pos": [0.07, 0.3, 0.62, 0.3]}),
                    ("A1", {"pos": [0.37, 0.012, 0.62, 0.3]}),  # scale bar
                    ("B", {"pos": [0.07, 0.3, 0.475, 0.125]}),
                    ("C1", {"pos": [0.07, 0.3, 0.31, 0.125]}),
                    ("C2", {"pos": [0.07, 0.3, 0.16, 0.125]}),
                    ("D", {"pos": [0.07, 0.3, 0.05, 0.075]}),
                    ("E", {"pos": [0.47, 0.45, 0.05, 0.85]}),
                    #     ('F', {'pos': [0.47, 0.45, 0.10, 0.30]}),
                ]
            )  # a1 is cal bar
        if plotmode == "publication":
            self.plotspecs = OrderedDict(
                [
                    ("A", {"pos": [0.45, 0.35, 0.58, 0.4]}),
                    ("A1", {"pos": [0.82, 0.012, 0.58, 0.4]}),  # scale bar
                    ("B", {"pos": [0.1, 0.78, 0.40, 0.1]}),
                    ("C1", {"pos": [0.1, 0.36, 0.05, 0.25]}),
                    ("C2", {"pos": [0.52, 0.36, 0.05, 0.25]}),
                    ("D", {"pos": [0.1, 0.78, 0.32, 0.05]}),
                    ("E", {"pos": [0.1, 0.78, 0.45, 0.125]}),
                    #     ('F', {'pos': [0.47, 0.45, 0.10, 0.30]}),
                ]
            )  # a1 is cal bar

        self.P = PH.Plotter(self.plotspecs, label=False, figsize=(10.0, 8.0))

        self.plot_hist(self.P.axdict["B"], results)  # PSTH
        # if imagefile is not None:
        #     self.MT.get_image(imagefile)
        #     self.MT.load_images()
        # print (self.MT.imagemetadata)
        # self.MT.show_images()
        # exit(1)
        # self.plot_average_traces(self.P.axdict['C'], self.Data.tb, self.Data.data_clean)

        ident = 0
        if ident == 0:
            cbar = self.P.axdict["A1"]
        else:
            cbar = None
        idm = self.mapfromid[ident]

        spotsize = self.Pars.spotsize
        self.newvmax = np.max(results[measuretype])
        if self.Pars.overlay_scale > 0.0:
            self.newvmax = self.Pars.overlay_scale
        self.newvmax = self.plot_map(
            self.P.axdict["A"],
            cbar,
            results["positions"],
            measure=results,
            measuretype=measuretype,
            vmaxin=self.newvmax,
            imageHandle=None, # self.MT,
            imagefile=imagefile,
            angle=rotation,
            spotsize=spotsize,
            whichstim=whichstim,
            average=average,
        )
        self.plot_stacked_traces(
            self.Data.tb,
            self.Data.data_clean,
            dataset,
            results,
            ax=self.P.axdict["E"],
            trsel=trsel,
        )  # stacked on right

        self.plot_avgevent_traces(
            evtype="avgevoked",
            mdata=self.Data.data_clean,
            trace_tb=self.Data.tb,
            ax=self.P.axdict["C1"],
            events=results["events"],
            scale=scf,
            label=label,
            rasterized=rasterized,
        )
        self.plot_avgevent_traces(
            evtype="avgspont",
            mdata=self.Data.data_clean,
            trace_tb=self.Data.tb,
            ax=self.P.axdict["C2"],
            events=results["events"],
            scale=scf,
            label=label,
            rasterized=rasterized,
        )

        if self.Data.photodiode is not None:
            self.plot_photodiode(
                self.P.axdict["D"], self.Data.photodiode_timebase[0], self.Data.photodiode
            )
        # mpl.show()
        return True  # indicated that we indeed plotted traces.
