import argparse
import datetime
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Union

import matplotlib
import matplotlib.cm
import matplotlib.collections as collections
import matplotlib.colors
import matplotlib.pyplot as mpl
import numpy as np
import pylibrary.plotting.plothelpers as PH
import pylibrary.tools.cprint as CP
import scipy.ndimage
import scipy.signal
import seaborn
import MetaArray
# import montager as MT
from matplotlib import colors as mcolors
from matplotlib.collections import PatchCollection
from matplotlib.patches import Wedge

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
            '(plot_map_data) Unrecongnized color map {0:s}; setting to "snshelix"'.format(
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
        # collection.set_clim(vmin, vmax)
    return collection


def testplot(crosstalk: np.ndarray, ifitx: list, avgdf: np.ndarray, 
             intcept: float, scf:float=1e12):
    """
    Note: This cannot be used if we are running in multiprocessing mode - will throw an error
    """
    import pyqtgraph as pg
    from pyqtgraph.Qt import QtCore, QtGui

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
        p1.plot(self.Data.timebase, data[0, i, :] + 2e-11 * i, pen=pg.mkPen("r"))
        p1.plot(self.Data.timebase, datar[0, i, :] + 2e-11 * i, pen=pg.mkPen("g"))

    p1.plot(self.Data.timebase, lbr, pen=pg.mkPen("c"))
    p2.plot(self.Data.timebase, crosstalk, pen=pg.mkPen("m"))
    p2.plot(self.Data.timebase, lbr, pen=pg.mkPen("c"))
    p2.setXLink(p1)
    p3.setXLink(p1)
    p3.plot(self.Data.timebase, avgdf, pen=pg.mkPen("w", width=1.0))  # original
    p3.plot(self.Data.timebase, olddatar, pen=pg.mkPen("b", width=1.0))  # original
    meandata = np.mean(datar[0], axis=0)
    meandata -= np.mean(meandata[0 : int(0.020 / ct_SR)])
    p3.plot(self.Data.timebase, meandata, pen=pg.mkPen("y"))  # fixed
    p3sp = pg.ScatterPlotItem(
        self.Data.timebase[tpts],
        meandata[tpts],
        pen=None,
        symbol="o",
        pxMode=True,
        size=3,
        brush=pg.mkBrush("r"),
    )  # points corrected?
    p3.addItem(p3sp)
    p4.plot(self.Data.timebase[:-1], diff_avgd, pen=pg.mkPen("c"))
    p2.setXLink(p1)
    p3.setXLink(p1)
    p4.setXLink(p1)
    if (sys.flags.interactive != 1) or not hasattr(QtCore, "PYQT_VERSION"):
        QtGui.QApplication.instance().exec_()
    exit()
    # return datar, avgd


class PlotMapData:
    def __init__(self, verbose=False):
        self.rasterized = False
        self.verbose = verbose
        self.reset_flags()

    def reset_flags(self):
        self.plotted_em = {
            "histogram": False,
            "stack": False,
            "avgevents": False,
            "avgax": [0, None, None],
        }

    def set_Pars_and_Data(self, pars, data, minianalyzer):
        """
        save parameters passed from analyze Map Data
        Analysis parameters and data are in separae data classes.
        """
        self.Pars = pars
        self.Data = data
        self.MA = minianalyzer
        self.Data.timebase = self.MA.timebase

    def gamma_correction(self, image, gamma=2.2, imagescale=np.power(2, 16)):
        if gamma == 0.0:
            return image
        imagescale= float(imagescale)
        # print('iscale, gamma: ', imagescale, gamma)
        try:
            imr = image/imagescale
            corrected = (np.power(imr, (1. / gamma)))*imagescale
        except:
            print('image minm: ', np.min(image), '  max: ', np.max(image))
            print('any neg or zero? ', np.any(image <= 0.))
            raise ValueError
        return corrected

    def show_images(self, show: bool = True) -> None:
        r, c = PH.getLayoutDimensions(len(self.images), pref="height")
        f, ax = mpl.subplots(r, c)
        self.figure_handle = f
        f.suptitle(self.celldir, fontsize=9) # .replace(r"_", r"\_"), fontsize=9)
        ax = ax.ravel()
        PH.noaxes(ax)
        for i, img in enumerate(self.imagedata):
            fne = Path(self.images[i]).name
            imfig = ax[i].imshow(self.gamma_correction(img, 2.2))
            PH.noaxes(ax[i])
            ax[i].set_title(fne, fontsize=8)
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
        units: str='radians'
    ) -> np.ndarray:
        """
        Angle is in radians unless otherwise specified
        """
        if units == 'degrees': # convert to radians
            angle = 360.*angle/(2.0*np.pi)
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
        for j in range(len(self.Pars.stimtimes["starts"])):
            t = self.Pars.stimtimes["starts"][j]-self.Pars.time_zero
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
        self.plotted_em["histogram"] = True
        plotevents = True
        rotation = 0.0
        plotFlag = True
        idn = 0
        self.newvmax = None
        events = results["events"]
        if events[0] == None:
            CP.cprint("r", "**** plot_hist: no events were found")
            return
        rate = 1.0/results["rate"]
        tb0 = events[0].average.avgeventtb  # get from first trace in first trial
        # rate = np.mean(np.diff(tb0))
        nev = 0  # first count up events
        for itrial in events.keys():
            if events[itrial] is None:
                continue
            for j, jtrace in enumerate(events[itrial].onsets):
                nev += len(jtrace)
        eventtimes:np.ndarray = np.zeros(nev)
        iev = 0
        for itrial in events.keys():
            if events[itrial] is None:
                continue
            for j, onsets in enumerate(events[itrial].onsets):
                ntrialev = len(onsets)
                eventtimes[iev : iev + ntrialev] = onsets
                iev += ntrialev
        CP.cprint(
            "c",
            f"    plot_hist:: total events: {iev:5d}  # event times: {len(eventtimes):5d}  Sample Rate: {1e6*rate:6.1f} usec",
        )

        if plotevents and len(eventtimes):
            nevents = 0
            y = np.array(eventtimes) * rate
            # print('AR Tstart: ', self.AR.tstart, y.shape)
            bins = np.arange(
                self.Pars.time_zero, self.Pars.time_end+1e-3, 1e-3
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
            axh.set_xlim(self.Pars.time_zero, self.Pars.time_end)

    def plot_stacked_traces(
        self,
        tb: np.ndarray,
        mdata: np.ndarray,
        title: str,
        results: dict,
        zscore_threshold: Union[list, None] = None,
        ax: matplotlib.pyplot.axes = None,
        trsel: Union[int, None] = None,
    ) -> None:
        """Plot the data traces in a stacked array

        Args:
            tb (np.ndarray): timebase
            mdata (np.ndarray): map data to plot (traces, 2d array)
            title (str): text tile for plot
            results (dict): The results dictionary to use
            zscore_threshold (float, None): whether to clip by zscores. None plots all traces, 
                a float value only plots those traces where the zscore to ANY stimulus
                exceeds the float value
            ax (Union[object, None], optional): _description_. Defaults to None.
            trsel (Union[int, None], optional): _description_. Defaults to None.
        """
        # assert not self.plotted_em['stack']
        CP.cprint("c", f"    Starting stack plot: {str(title):s}")
        now = datetime.datetime.now().strftime("%Y-%m-%d  %H:%M:%S %z")
        if ax is None:
            f, ax = mpl.subplots(1, 1)
            self.figure_handle = f
        mpl.text(
            0.96,
            0.01,
            s=now,
            fontsize=6,
            ha="right",
            transform=ax.get_figure().transFigure,
        )
        linewidth = 0.35  # base linewidth
        self.plotted_em["stack"] = True
    
        events = results["events"]

        if zscore_threshold is not None:
            zs = np.max(np.array(results['ZScore']), axis=0)

        nevtimes = 0
        spont_ev_count = 0
        dt = np.mean(np.diff(self.Data.timebase))
        step_I = self.Pars.stepi
        if trsel is not None:
            # only plot the selected traces
            for jtrial in range(mdata.shape[0]):
                if tb.shape[0] > 0 and mdata[jtrial, trsel, :].shape[0] > 0:
                    ax.plot(
                        tb, #  - self.Pars.time_zero,
                        mdata[0, trsel, :], #  * self.Pars.scale_factor,
                        linewidth=0.2,
                        rasterized=False,
                        zorder=10,
                    )
            PH.clean_axes(ax)
            PH.calbar(
                ax,
                calbar=[
                    self.Pars.time_end,
                    -200e-12, # 3  * self.Pars.scale_factor,
                    0.05,
                    100e-12, #  * self.Pars.scale_factor,
                ],
                axesoff=True,
                orient="left",
                unitNames={"x": "s", "y": "pA"},
                fontsize=11,
                weight="normal",
                font="Arial",
            )
            mpl.suptitle(str(Path(*Path(title).parts[-5:])), fontsize=8) # .replace(r"_", r"\_"), fontsize=8)
            self.plot_timemarker(ax)

            ax.set_xlim(0.0, (self.Pars.time_end-self.Pars.time_start-0.001))
            return

        crflag = [False for i in range(mdata.shape[0])]

        iplot_tr = 0
        # decorate traces with dots for spont, evoked and "other"
        for itrial in range(mdata.shape[0]):  # across trials (repeats)
            if events[itrial] is None:
                continue
            for itrace in range(mdata.shape[1]):  # across all traces
                if zscore_threshold is not None and zs[itrace] < zscore_threshold:
                    continue
                smpki_original = events[itrial].smpkindex[itrace]  # indices to peaks in ths trial/trace
                smpki = smpki_original
                # for k, smpk in enumerate(smpki_original):
                #     if smpk > len(tb):
                #         smpki.pop(k)

                pktimes = np.array(smpki)*dt # *events[itrial].dt_seconds  # times
                # print(smpki)
                if len(pktimes) > 0:
                    nevtimes += len(smpki)
                    if (
                        nevtimes > 0
                        and len(tb[smpki]) > 0
                        and len(mdata[itrial, itrace, smpki]) > 0
                    ):
                        # identify those events in the spont window - before the first stimulus
                        sd = events[itrial].spont_dur[itrace]
                        pre_stim_indices = np.where(tb[smpki] < sd)[0].astype(int)
                        tsi = [int(smpki[x]) for x in pre_stim_indices]
                     # find indices of "response" events (events occuring in a short window after each stimulus)
                        tri = np.ndarray(0)
                        for (
                            t_ev
                        ) in self.Pars.twin_resp:  # find events in all response windows
                            in_stim_indices = np.where(
                                            (tb[smpki] >= t_ev[0]) & (tb[smpki] < t_ev[1])
                                        )[0].astype(int)
                            if len(in_stim_indices) > 0:
                                tri = np.concatenate(
                                    (
                                        tri.copy(),
                                        [smpki[x] for x in 
                                            in_stim_indices
                                        ],
                                    ),
                                    axis=0,
                                ).astype(int)
                        ts2i = list(
                            set(smpki)
                            - set(tri.astype(int)).union(set(tsi))
                        )  # remainder of events (not spont, not possibly evoked)
                        # now get the amplitude of the events in each group
                        ms = np.array(
                            mdata[itrial, itrace, tsi]
                        )  # spontaneous events
                        mr = np.array(
                            [mdata[itrial, itrace, x] for x in tri]
                        ) # response in window
                        if len(mr) > 0:
                            crflag[itrial] = True  # flag traces with detected responses
                        ms2 = np.array(
                            mdata[itrial, itrace, ts2i]
                        )  # events not in spont and outside window
                        spont_ev_count += ms.shape[0]
                        cr = matplotlib.colors.to_rgba(
                            "r", alpha=0.6
                        )  # red is for evoked responses
                        ck = matplotlib.colors.to_rgba("k", alpha=1.0) # black is for spont
                        cg = matplotlib.colors.to_rgba("gray", alpha=1.0) # gray is for spont between stimuli

                        ax.plot(
                            tb[tsi], # -self.Pars.time_zero,
                            ms * self.Pars.scale_factor + step_I * iplot_tr,
                            "o",
                            color=ck,
                            markersize=2,
                            markeredgecolor="None",
                            zorder=0,
                            rasterized=self.rasterized,
                        )
                        ax.plot(
                            [tb[x] for x in tri], # -self.Pars.time_zero,
                            mr * self.Pars.scale_factor + step_I * iplot_tr,
                            "o",
                            color=cr,
                            markersize=2,
                            markeredgecolor="None",
                            zorder=0,
                            rasterized=self.rasterized,
                        )
                        ax.plot(
                            [tb[x] for x in ts2i], # -self.Pars.time_zero,
                            ms2 * self.Pars.scale_factor + step_I * iplot_tr,
                            "o",
                            color=cg,
                            markersize=2,
                            markeredgecolor="None",
                            zorder=0,
                            rasterized=self.rasterized,
                        )
                iplot_tr += 1
        # now plot the traces in a stacked format
        iplot_tr = 0  # index to compute stack position
        for itrial in range(mdata.shape[0]):
            for itrace in range(mdata.shape[1]):
                if zscore_threshold is not None and zs[itrace] < zscore_threshold:
                    continue
                if tb.shape[0] > 0 and mdata[itrial, itrace, :].shape[0] > 0:
                    if crflag[itrial]:
                        alpha = 1.0
                        lw = linewidth
                    else:
                        alpha = 0.3
                        lw = linewidth * 0.25
                    ax.plot(
                        tb, # -self.Pars.time_zero,
                        mdata[itrial, itrace, :] * self.Pars.scale_factor
                        + step_I * iplot_tr,
                        linewidth=lw,
                        rasterized=False,
                        zorder=10,
                        alpha=alpha,
                    )
                iplot_tr += 1
        CP.cprint("c", f"        Spontaneous Event Count: {spont_ev_count:d}")

        mpl.suptitle(str(title), fontsize=8) # .replace(r"_", r"\_"), fontsize=8)
        self.plot_timemarker(ax)
        # ax.set_xlim(0.0, (self.Pars.time_end - self.Pars.time_zero-0.001))

    def get_calbar_Yscale(self, amp: float) -> float:
        """
        Pick a scale for the calibration bar based on the amplitude to be represented
        """
        sc = [
            1.0,
            2.0,
            5.0,
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

    def plot_event_traces(
        self,
        evtype: str,
        mdata: np.ndarray,
        trace_tb: np.ndarray,
        datatype: str,
        results: dict,
        tpre: float=0.0,
        zscore_threshold: Union[float, None] = None,
        plot_minmax:Union[list, None] = None,  # put bounds on amplitude of events that are plotted
        ax: Union[object, None] = None,
        scale: float = 1.0,
        label: str = "pA",
        rasterized: bool = False,
    ) -> None:
        # ensure we don't plot more than once...
        # modified so that events that are "positive" (meaning the detection picked uip
        # an event that was on a + baseline...) are removed from the plot. 
        # the criteria is > 0 pA at the minimum of the average
        # those traces are then removed from the single event plots and
        # the average is recomputed.
        # CP.cprint('y', f"start avgevent plot for  {evtype:s}, ax={str(ax):s}")
        # assert not self.plotted_em['avgevents']
        # print("results: ", results)

        events = results['events']
        if self.plotted_em["avgax"][0] == 0:
            self.plotted_em["avgax"][1] = ax
        elif self.plotted_em["avgax"][0] == 1:
            if self.plotted_em["avgax"][1] == ax:
                raise ValueError("plot_avgevent_traces : repeated into same axis")
            else:
                self.plotted_em["avgax"][2] = ax
                self.plotted_em["avgevents"] = True
        CP.cprint('c', f"plotting avgevent plot for  {evtype:s}, ax={str(ax):s}")

        # self.plotted_em['avgevents'] = True
        if events is None or ax is None or trace_tb is None:
            CP.cprint(
                "r",
                f"[plot_avgevent_traces]:: evtype: {evtype:s}. No events, no axis, or no time base",
            )
            return
        nevtimes = 0
        line = {"avgevoked": "k-", "avgspont": "k-", "evoked_events": "k-", "spont_events": "k-"}
        ltitle = {"avgevoked": "Evoked (%s)" % label, "evoked_events": "Evoked (%s)" % label,
                   "avgspont": "Spont (%s)" % label, "spont_events": "Spont (%s)" % label}
        result_names = {"avgevoked": "evoked_events", "avgspont": "spont_events"}

        ax.set_ylabel(ltitle[evtype])
        ax.spines["left"].set_color(line[evtype][0])
        ax.yaxis.label.set_color(line[evtype][0])
        ax.tick_params(axis="y", colors=line[evtype][0], labelsize=7)
        ax.tick_params(axis="x", colors=line[evtype][0], labelsize=7)
        ev_min = 5e-12
        sp_min = 5e-12

        ave = []
        minev = 0.0
        maxev = 0.0
        npev = 0
        print("plot minmax:", plot_minmax, " zscore_threshold: ", zscore_threshold)
        # plot events from each trial
        for trial in range(mdata.shape[0]):
            CP.cprint("b", f"plotting events for trial: {trial:d}")
            if events[trial] is None or len(events[trial].average.avgeventtb)  == 0:
                CP.cprint("r", "**** plot_event_traces: no events....")
                continue

            tb0 = events[trial].average.avgeventtb  # get from the averaged trace
            rate = events[trial].dt_seconds
            tpost = np.max(tb0)
            tb = np.arange(0, tpost + rate, rate)
            ptfivems = int(0.0005 / rate)
            allevents = events[trial].allevents

            for itrace in range(mdata.shape[1]):  # traces in the evtype list
                if events is None or trial not in list(events.keys()):
                    if self.verbose:
                        print(f"     NO EVENTS in trace: {itrace:4d}")
                    continue
                if evtype == "avgevoked":
                    evs = events[trial].evoked_event_trace_list[itrace]
                elif evtype == "avgspont":
                    evs = events[trial].spontaneous_event_trace_list[itrace]
                else:
                    continue
                if evs is None or evs == [[],[]]:  # skip if there are NO event to plot from this trace
                    if self.verbose:
                        print(
                            f"     NO EVENTS of type {evtype:10s} in trace: {itrace:4d}"
                        )
                    continue
                for j, jevent in enumerate(evs): 
                    # evs is 2 element array: [0] are onsets and [1] is peak; here we align the traces to onsets
                    event_id = (itrace, j)
                    if event_id not in allevents.keys():
                        continue
                    evdata = allevents[event_id]
                    # print("ipre: ", ipre, "ptfivems: ", ptfivems)
                    bl = np.mean(evdata[0 : ptfivems]) # ipre - ptfivems])
                    evdata -= bl
                    if len(evdata) > 0:
                        append = False
                        if plot_minmax is not None:  # only plot events that fall in an ampltidue window
                            if (np.min(evdata) < plot_minmax[0]) or (np.max(evdata) > plot_minmax[1]):
                                continue # 
                        if zscore_threshold is not None and np.max(results['ZScore'], axis=0)[itrace] > zscore_threshold and evtype == "avgspont":
                            append = True
                        # elif np.max(evdata[:int(len(evdata)/2)]) > 50e-12:
                        #     append=False
                        else: # zscore_threshold == None:  # accept all comers.
                            append = True
                        if append:
                            ave.append(evdata)
                            npev += 1
                        # and only plot when there is data, otherwise matplotlib complains with "negative dimension are not allowed" error

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

        nev = len(ave)
        aved = np.asarray(ave)
        if 0 in (len(aved), aved.shape[0],  nev):
            if evtype == 'avgevoked':
                evname = "evoked"
            elif evtype == 'avgspont':
                evname = "spontaneous"
            else:
                raise ValueError("plotMapData:plotAvgEventTraces: evtype not recognized: ", evtype)
            PH.noaxes(ax)
            ax.text(0.5, 0.5, f"No {evname:s} events", fontsize=12, ha="center", va="center")
            return
        if self.verbose:
            CP.cprint("red", f"aved shape is {str(aved.shape):s}")
            return
        tx = np.broadcast_to(tb, (aved.shape[0], tb.shape[0])).T
        if self.Pars.sign < 0:
            maxev = -minev
        self.MA.set_sign(self.Pars.sign)
        self.MA.set_dt_seconds(rate)
        self.MA.set_datatype(datatype)
        avedat = np.mean(aved, axis=0)
        tb = tb[: len(avedat)]
        avebl = 0 # np.mean(avedat[:ptfivems])
        avedat = avedat - avebl
        self.MA.fit_average_event(
            tb,
            avedat,
            debug=False,
            label="Map average",
            inittaus=self.Pars.taus,
            initdelay= tpre,
        )
        CP.cprint("c", "        Event fitting completed")


        Amplitude = np.max(self.MA.sign*avedat)
        Amplitude1 = self.MA.fitresult.values["amp"]
        Amplitude2 = self.MA.fitresult.values["amp2"]
        tau1 = self.MA.fitresult.values["tau_1"]
        tau2 = self.MA.fitresult.values["tau_2"]
        tau3 = self.MA.fitresult.values["tau_3"]
        tau4 = self.MA.fitresult.values["tau_4"]
        bfdelay = self.MA.fitresult.values["fixed_delay"]
        bfit = self.MA.avg_best_fit

        # if self.Pars.sign == -1:
        #     amp = np.min(bfit)
        # else:
        #     amp = np.max(bfit)
        txt = f"Amp: {scale*Amplitude:.1f}pA tau1:{1e3*tau1:.2f}ms tau2: {1e3*tau2:.2f}ms (N={aved.shape[0]:d} del={bfdelay:.4f})"
        txt2 = f"Amp2: {scale*Amplitude2:.1f}pA tau3:{1e3*tau3:.2f}ms tau4: {1e3*tau4:.2f}ms (N={aved.shape[0]:d})"

        if evtype == "avgspont" and events[0] is not None:
            srate = float(aved.shape[0]) / (
                events[0].spont_dur[0] * mdata.shape[1]
            )  # dur should be same for all trials
            txt += f" SR: {srate:.2f} Hz"
        if events[0] is None:
            txt = txt + "SR: No events"
        ax.text(0.05, 0.97, txt, fontsize=6, transform=ax.transAxes)
        ax.text(0.05, 0.89, txt2, fontsize=6, transform=ax.transAxes)
        if bfit is not None:
            ax.plot(
                tb * 1e3,
                scale * bfit,
                "c",
                linestyle="-",
                linewidth=0.35,
                rasterized=self.rasterized,
            )
        else:
            ax.text(0.05, 0.0, "fit to average failed", fontsize=6, transform=ax.transAxes)
        ax.plot(
            tb * 1e3,
            scale * avedat,
            line[evtype],
            linewidth=0.625,
            rasterized=self.rasterized,
        )

        ylims = ax.get_ylim()
        ylimsn = 1.4*ylims[0]
        ylimsp = 1.4*ylims[1]
        ax.set_ylim([ylimsn, ylimsp])
        if evtype == "avgspont":
            PH.calbar(
                ax,
                calbar=[
                    np.max(tb*1e3) - 2.0,
                    ylims[0],
                    2,
                    self.get_calbar_Yscale(np.fabs(ylims[1] - ylims[0]) / 4.0),
                ],
                axesoff=True,
                orient="right",
                unitNames={"x": "ms", "y": label},
                fontsize=11,
                weight="normal",
                font="Arial",
            )
        elif evtype == "avgevoked":
            PH.calbar(
                ax,
                calbar=[
                    np.max(tb*1e3) - 2.0,
                    ylims[0],
                    2,
                    self.get_calbar_Yscale(maxev / 4.0),
                ],
                axesoff=True,
                orient="right",
                unitNames={"x": "ms", "y": label},
                fontsize=11,
                weight="normal",
                font="Arial",
            )

    def plot_average_traces(
        self,
        ax: object,
        results: dict,
        tb: np.ndarray,
        mdata: np.ndarray,
        color: str = "k",
    ) -> None:
        """
        Average the traces and plot into an axis

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
        print("plot_average_traces, pars: ", self.Pars)
        if len(tb) > 0 and len(mdata) > 0:
            ax.plot(
                (tb-self.Pars.time_zero) * 1e3,
                mdata * self.Pars.scale_factor,
                color,
                rasterized=self.rasterized,
                linewidth=0.6,
            )

        ax.set_xlim(0.0, self.Pars.time_end * 1e3 - 1.0)

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
                tb-self.Pars.time_zero,
                np.mean(pddata, axis=0)*1e3,
                color,
                rasterized=self.rasterized,
                linewidth=0.6,
            )
            maxpower = np.max(pddata)*1e3 # in mW
            ax.text(x=0.05, y=0.95, s=f"Max = {maxpower:.2f} mW", fontsize=6, transform=ax.transAxes)
            PH.talbotTicks(ax, axes="xy", density=(1.0, 1.0), insideMargin=0.05, pointSize=6, tickPlacesAdd={"x": 1, "y": 1})

        ax.set_xlim(0.0, self.Pars.ar_tstart - 0.001)
        ax.set_ylabel("P (mW)", fontsize=10)
        ax.set_xlabel("Time (s)", fontsize=10)

    def get_laser_spots(self, pos, mapdir:Union[Path, str]):
        """
        This function is not needed if the spots have been loaded in the mosiac editor...
        et_laser_spots from the selected map directory camera images,
        and compare to the spot locations in the scanner file
        Generates a maximal image projection of the camera images
        taken during the mapping experiment, and retuns that image
        """
        imagecount = 0
        mappoints = list(Path(mapdir).glob("*"))
        mappoints = [mp for mp in mappoints if mp.is_dir()]
        useframe = 1
        for imagecount, mp in enumerate(mappoints):
            cameraframe = Path(mp, 'Camera', 'frames.ma')
            frame = MetaArray.MetaArray(file=str(cameraframe),  # read the camera frame
                                        readAll=True,  # read all data into memory
                                        verbose=False)
            frame_data = frame.view(np.ndarray)
            if imagecount == 0:
                frame_data_max = frame_data[useframe,:,:]
                frame_bkgd = np.zeros_like(frame_data[useframe,:,:])
            else:
                if useframe == 0:
                    frame_data_max += frame_data[useframe,:,:]
                    frame_bkgd = np.zeros_like(frame_data[useframe,:,:])
                else:
                    frame_data_max = np.maximum(frame_data_max, frame_data[useframe,:,:])
                    frame_bkgd += frame_data[0,:,:]

        frame_bkgd = frame_bkgd/int(imagecount)
        if useframe == 0:
            frames = frame_data_max/int(imagecount)
        else:
            frames = frame_data_max - frame_bkgd
        # print(np.max(frames), np.min(frames))
        # frames = frames  > np.min(frames)*1.5
        info = frame.infoCopy()
        spotimage = MetaArray.MetaArray(frames, info=info[1:])  # remove the time axis.
        # fout = Path(str(mapdir)+'_spotimage.ma')
        # print('info: ', info)
        # spotimage.write(str(fout))
        # exit()
        return spotimage

    def reorder(self, a: float, b: float):
        """
        make sure that b > a
        if not, swap and return
        """
        if a > b:
            t = b
            b = a
            a = t
        return (a, b)

    def add_image(self, axp, imageInfo, imageData, angle:float=0., alpha:float=1.0, cmap:str="gray"):
        """
        Add an image to the plot (laser spots or camera image)"""

        # compute the extent for the image, offsetting it to the map center position
        ext_left = imageInfo["deviceTransform"]["pos"][0]  # - pz[0]
        ext_right = (
            ext_left
            + imageInfo["region"][2]
            * imageInfo["deviceTransform"]["scale"][0]
        )
        ext_bottom = imageInfo["deviceTransform"]["pos"][1]  # - pz[1]
        ext_top = (
            ext_bottom
            + imageInfo["region"][3]
            * imageInfo["deviceTransform"]["scale"][1]
        )
        ext_left, ext_right = self.reorder(ext_left, ext_right)
        ext_bottom, ext_top = self.reorder(ext_bottom, ext_top)
        # extents are manually adjusted - something about the rotation should be computed in them first...
        # but fix it manually... worry about details later.
        # extents = [ext_bottom, ext_top, ext_left, ext_right]  #
        extents = [ext_left, ext_right, ext_bottom, ext_top]
        if (
            angle != 0.0
        ):  # note that angle arrives in radians - must convert to degrees for this one.
            print("rotation angle: ", angle)
            img = scipy.ndimage.interpolation.rotate(
                imageData,
                angle,  #* 180.0 / np.pi + 90.0,
                axes=(1, 0),
                reshape=True,
                output=None,
                order=3,
                mode="constant",
                cval=0.0,
                prefilter=True,
            )
        else:
            img = imageData

        axp.imshow(
            img,
            aspect="equal",
            extent=extents,
            origin="lower",
            cmap=setMapColors(cmap),
            alpha=alpha,
        )
        return extents
    

    def plot_map(
        self,
        axp,
        axcbar,
        pos,
        measure,
        measuretype: Union[None, str]=None,
        vmaxin=None,
        imageHandle=None,  # could be a montager instance
        imagefile=None,
        mapdir: Union[str, Path, None] = None,
        angle=0,
        spotsize=42e-6,
        cellmarker=False,
        markers:dict=None,
        whichstim=-1,
        average=False,
        pars=None,
    ):
        sf = 1.0  # everything is scaled in meters
        cmrk = 50e-6 * sf  # size, microns
        linewidth = 0.2
        npos = pos.shape[0]
  
        npos += 1  # need to count up one more to get all of the points in the data
        pos = pos[:npos, :]  # clip unused positions
        if imageHandle is not None and imagefile is not None:
            imagedata = imageHandle.imagemetadata[0]
            img = imageHandle.imagedata[0]
            # extents = self.add_image(axp,imageInfo=imageHandle.imagemetadata[0], 
            #                imageData=imageHandle.imagedata,  angle=angle, alpha=0.5)

        if mapdir is not None:
            spotsdata = self.get_laser_spots(pos, mapdir) # get the laser spots from the map directory
            # if spotsdata is not None:
            #     extents = self.add_image(axp, imageInfo=spotsdata._info[2], imageData=spotsdata.view(np.ndarray), angle=90)# =angle-np.pi )


        if measuretype in ("A", "Q"):
            mtype = "I_max"
        else:
            mtype = measuretype

        if measuretype is None:
            spotsizes = spotsize * np.ones(len(pos))
        else: 
            if whichstim < 0:
                spotsizes = spotsize * np.linspace(1.0, 0.2, len(measure[mtype]))
            else:
                spotsizes = spotsize * np.ones(len(measure[measuretype]))
        pos = self.scale_and_rotate(pos, scale=1.0, angle=angle)
        xlim = [np.min(pos[:, 0]) - spotsize, np.max(pos[:, 0]) + spotsize]
        ylim = [np.min(pos[:, 1]) - spotsize, np.max(pos[:, 1]) + spotsize]
        # make sure the key markers are on the plot
        for marker in markers:
            if marker not in ['soma', 'surface', 'medialborder', 'lateralborder', 'AN', 'rostralborder', 'caudalborder',
                              'ventralborder', 'dorsalborder']:
                continue
            if markers[marker] is not None and len(markers[marker]) >= 2:
                position = markers[marker]
                msize = 20e-6
                xlim = [np.min([xlim[0], position[0]-msize]), np.max([xlim[1], position[0]+msize])]
                ylim = [np.min([ylim[0], position[1]-msize]), np.max([ylim[1], position[1]+msize])]
        sign = measure["sign"]

        upscale = 1.0
        vmin = 0
        vmax = 1
        data = []
        if measuretype == "ZScore":
            data = measure[measuretype]

            vmax = np.max(np.max(measure[measuretype]))
            vmin = np.min(np.min(measure[measuretype]))
            if vmax < 6.0:
                vmax = 6.0  # force a fixed minimum scale
            scaler = PH.NiceScale(0.0, vmax)
            vmax = scaler.niceMax

        elif measuretype == "Qr":
            data = sign * (measure["Qr"] - measure["Qb"])
            data  = np.clip(data, 0., np.inf)
            vmin = 0.0
            vmax = np.max(data)
            if vmax == vmin:
                vmax = vmin + 1

        elif measuretype in ("A", "Q") and measure["events"][0] is not None:
            events = measure["events"]
            nspots = len(measure["events"][0])  # on trial 0
            if "npulses" in list(measure["stimtimes"].keys()):
                npulses = measure["stimtimes"]["npulses"]
            else:
                npulses = len(measure["stimtimes"]["start"])
            if isinstance(npulses, list):
                npulses = npulses[0]
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
                        [
                            stims[j] + self.Pars.direct_window,
                            stims[j] + self.Pars.response_window,
                        ]
                    )

            rate = measure["rate"]
            nev = 0
            nev_spots = 0
            # print("event measures: ", events[0][0]["measures"])
            for trial in range(measure["ntrials"]):
                skips = False
                for spot in range(nspots):  # for each spot
                    for ipulse in range(npulses):
                        if measuretype == "Q":
                            # try:  # repeated trials not implemented here.
                            #     u = isinstance(events[trial]["measures"][spot])
                            # except:
                            #     if not skips:
                            #         CP.cprint("r", f"Skipped Plotting Q for trial {trial:d} of {measure['ntrials']:d}, spots {nspots:d}, stimpulses: {npulses:d} and subsequent")
                            #         skips = True
                            #     continue
                            u = events[trial]
                            try:
                                u = events[trial]["measures"][spot]
                            except:
                                CP.cprint("r", f"Skipped Plotting Q for trial {trial:d} of {measure['ntrials']:d}, spots {nspots:d}, stimpulses: {npulses:d}")
                                continue
                            if not isinstance(events[trial]["measures"][spot], dict):
                                continue
                            if (
                                len(events[trial]["measures"][spot]["Q"]) > ipulse
                            ):  # there is only one, the first trial
                                data[ipulse, spot] = events[trial]["measures"][spot][
                                    "Q"
                                ][ipulse]
                            continue
                        try:
                            x = events[trial]
                        except:
                            raise ValueError("Unable to store events On trial: ", trial)
                        try:
                            smpki =  events[trial]["smpksindex"][0][spot]
                        except:
                            try:
                                smpki = events[trial]["smpksindex"][spot]
                            except:
                                # continue
                                print("on Spot: ", spot)
                                print(smpki)
                                smpki =  events[trial]["smpksindex"]
                                raise ValueError(smpki)
                        tri = np.ndarray(0)
                        tev = twin_resp[ipulse]  # go through stimuli
                        iev0 = int(tev[0] / rate)
                        iev1 = int(tev[1] / rate)
                        if not isinstance(smpki, (list, np.ndarray)):
                            smpki = np.ndarray([smpki])
                        idx = np.where((smpki >= iev0) & (smpki <= iev1))[0]
                        if len(smpki) > 0:
                            tri = np.concatenate(
                                (
                                    tri.copy(),
                                    smpki[idx],
                                ),
                                axis=0,
                            ).astype(int)

                        smpki = list(smpki)
                        for t in range(len(tri)):
                            if tri[t] in smpki:
                                r = smpki.index(tri[t])  # find index
                                data[ipulse, spot] += (
                                    sign * events[trial]["smpks"][spot][r]
                                )
                                nev_spots += 1

            vmin = 0.0
            vmax = np.max(data)
            # nev = len(np.where(data > 0.0)[0])
            # print(nev, nev_spots)

        if vmax == 0:
            vmax = 1e-10
        if vmax == np.nan:
            vmax = 1
        if vmin == np.nan:
            vmin = 0
        # print("vmax: ", vmax, "vmin : ", vmin)

        scaler = PH.NiceScale(vmin, vmax)

        if whichstim >= 0:  # which stim of -1 is ALL stimuli
            whichmeasures = [whichstim]
        elif average:
            whichmeasures = [0]
            data[0] = np.mean(data)  # just
        else:
            whichmeasures = range(len(data))
        norm = matplotlib.colors.Normalize(vmin=0, vmax=vmax)
        cmx = matplotlib.cm.ScalarMappable(norm=norm, cmap=cm_sns)
        # axp.set_facecolor([0.75, 0.75, 0.75])
        for (
            im
        ) in (
            whichmeasures
        ):  # there may be multiple measure[measuretype]s (repeated stimuli of different kinds) in a map
            # note circle size is radius, and is set by the laser spotsize (which is diameter)
            radw = np.ones(pos.shape[0]) * spotsizes[im]
            radh = np.ones(pos.shape[0]) * spotsizes[im]
            if measuretype in ("Qr", "Imax"):
                spotcolors = cmx.to_rgba(np.clip(data[im], 0.0, vmax))
            else:  # zscore, no sign.
                spotcolors = cmx.to_rgba(np.clip(data[im], 0.0, vmax))
            edgecolors = spotcolors.copy()
            for i in range(len(data[im])):
                em = data[im][i]
                if measuretype == "ZScore" and em < 1.96:
                    spotcolors[i][3] = em / 1.96  # scale down
                edgecolors[i] = matplotlib.colors.to_rgba([0.6, 0.6, 0.6, 0.5])
            order = np.argsort(
                data[im]
            )  
            # plot from smallest to largest (so largest on top)
            if self.nreps == 1:
                # concentric circles (overlaid ellipses) showing the response measure for
                # each stimulus. Organization is outside-in first to last stimulus
                if len(order) > len(pos): # guess that this is one spot, multiple trials
                    pos = np.tile(pos, (len(order), 1))
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
                nreps = self.nreps # 1
                ic = 0
                npos = pos.shape[0]
                dtheta = 360.0 / nreps
                ri = 0
                rs = int(npos / nreps)
                if rs == 0:
                    rs = 1
                for nr in range(nreps):
                    ec = wedges(
                        pos[ri : (ri + rs), 0],
                        pos[ri : (ri + rs), 1],
                        radw[ri : (ri + rs)] / 2.0,
                        theta1=nr * dtheta,
                        theta2=(nr + 1) * dtheta,
                        color=spotcolors[ri : ri + rs],
                    )
                    axp.add_collection(ec)
                    if npos > 1:
                        ri += rs

        if cellmarker:
            CP.cprint("yellow", "Cell marker is plotted")
            axp.plot(
                [-cmrk, cmrk], [0.0, 0.0], "-", color="r"
            )  # cell centered coorinates
            axp.plot(
                [0.0, 0.0], [-cmrk, cmrk], "-", color="r"
            )  # cell centered coorinates
        mark_colors = {'soma': 'y', 'surface': 'c', 'medialborder': 'r', 'lateralborder': 'm', 'AN': 'g', 
                       'rostralborder': 'r', 'caudalborder': 'm', 'ventralborder': 'b', 'dorsalborder': 'g'}
        mark_symbols = {'soma': '*', 'surface': 'v', 'medialborder': '^', 'lateralborder': 's', 'AN': 'D',
                        'rostralborder': '>', 'caudalborder': '<', 'ventralborder': '2', 'dorsalborder': '1'}
        if markers is not None:
            for marktype in markers.keys():
                if marktype not in mark_colors.keys():
                    continue
                position = markers[marktype]
                if position is not None and len(position) >= 2:
                    axp.plot([position[0], position[0]],
                            [position[1], position[1]],
                            marker = mark_symbols[marktype], color=mark_colors[marktype], 
                            markersize=8, alpha=0.8)
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
        axp.set_facecolor([0.0, 0.0, 0.0])
        PH.noaxes(axp)
        axp.plot([xlim[0]+10e-6, xlim[0]+110e-6], [ylim[0]+20e-6, ylim[0]+20e-6], 'w-', lw=2)
        axp.text(x=xlim[0]+60e-6, y=ylim[0]+30e-6, s="100 um", color='w', fontsize=8, ha='center', va='bottom')
        # axp.set_xlim(xlim)
        # axp.set_ylim(ylim)
        if imageHandle is not None and imagefile is not None:
            axp.set_aspect("equal")
        axp.set_aspect("equal")
        title = measuretype# .replace(r"_", r"\_")
        if whichstim >= 0:
            stima = r"Stim \# "
            title += f", {stima:s} {whichstim:d} Only"
        if average:
            title += ", Average"
        if vmaxin is None:
            return vmax
        return vmaxin

    def display_position_maps(
        self,
        dataset_name: Union[Path, str],
        result: dict,
        pars: object = None,
        markers:dict=None,
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
                    {"pos": [0.4, 0.012, 0.55, 0.3], "labelpos": (-0.05, 1.05)},
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
        now = datetime.datetime.now().strftime("%Y-%m-%d  %H:%M:%S %z")
        mpl.text(
            0.96,
            0.01,
            s=now,
            fontsize=6,
            ha="right",
            transform=self.P.figure_handle.transFigure,
        )
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
                measuretype=None, # measuredict[measure],
                pars=pars,
                markers=markers,
                mapdir=dataset_name,
            )
        return True

    def display_one_map(
        self,
        dataset:str,  # typically is the map directory
        results:dict,
        imagefile=None,
        rotation:float=0.0,
        measuretype:str=None,
        zscore_threshold:Union[float, None]=None,
        plotevents=True,
        rasterized:bool=False,
        whichstim:int=-1,
        average:bool=False,
        trsel=None,
        plotmode:str="document",
        datatype=None,
        imagedata = None,
        markers: Union[dict, None] = None,
        plot_minmax: Union[list, None] = None,
        cal_height: Union[float, None] = None,

    ) -> bool:
        if results is None or self.Pars.datatype is None:
            CP.cprint("r", f"NO Results in the call, from {dataset.name:s}")
            return

        if (
            ("_IC" in str(dataset.name))
            or ("_CC" in str(dataset.name))
            or ("_Ic" in str(dataset.name))
            or (self.Pars.datatype in ("I", "IC"))
        ):
            scf = 1e3
            label = "mV"  # mV
        elif (
            ("_VC" in str(dataset.name))
            or("_Vc" in str(dataset.name))
            or ("VGAT_5ms" in str(dataset.name))
            or ("_WCChR2" in str(dataset.name))
            or (self.Pars.datatype in ("V", "VC"))
        ):
            scf = 1e12  # pA, vc
            label = "pA"
        else:
            scf = 1.0
            label = "AU"

        # build a figure
        self.nreps = results["ntrials"]

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
                    ("E", {"pos": [0.47, 0.45, 0.2, 0.75]}),
                    ("F", {'pos': [0.47, 0.45, 0.05, 0.18]}),
                ]
            )
            scale_bar = "A1"
            evoked_panel = "C1"
            spont_panel = "C2"
            trace_panel = "E"
            hist_panel = "B"
            map_panel = "A"
            photodiode_panel = "D"
            slice_image_panel = None
            cell_image_panel = None
            average_panel = "F"

        elif plotmode == "publication":
            label_fsize = 16
            self.plotspecs = OrderedDict(
                [
                    ("A1", {"pos": [0.05, 0.25, 0.6, 0.38], "labelpos": [-0.12, 0.95], "fontsize": label_fsize}),
                    ("A2", {"pos": [0.35, 0.25, 0.6, 0.38], "labelpos": [-0.12, 0.95], "fontsize": label_fsize}),
                    ("B", {"pos": [0.65, 0.25, 0.6, 0.38], "labelpos": [-0.12, 0.95], "fontsize": label_fsize}),
                    ("B1", {"pos": [0.94, 0.012, 0.7, 0.2], "labelpos": [-1, 1.1], "fontsize": label_fsize}),  # scale bar
                    ("C", {"pos": [0.1, 0.78, 0.41, 0.17], "labelpos": [-0.03, 1.05], "fontsize": label_fsize}),
                    ("D", {"pos": [0.1, 0.78, 0.36, 0.03], "labelpos": [-0.03, 1.2], "fontsize": label_fsize}),
                    ("E1", {"pos": [0.1, 0.36, 0.05, 0.22], "fontsize": label_fsize}),
                    ("E2", {"pos": [0.52, 0.36, 0.05, 0.22], "fontsize": label_fsize}),

                ]
            )  # b1 is cal bar
            scale_bar = "B1"
            trace_panel = "C"
            evoked_panel = "E1"
            spont_panel = "E2"
            hist_panel = None
            map_panel = "B"
            photodiode_panel = "D"
            slice_image_panel = 'A1'
            cell_image_panel = 'A2'
            average_panel = None
        # self.panelmap = panelmap
        self.panels = {'scale_bar': scale_bar,
                        'trace_panel': trace_panel,
                        'evoked_panel': evoked_panel,
                        'spont_panel': spont_panel,
                        'hist_panel': hist_panel,
                        'map_panel': map_panel,
                        'photodiode_panel': photodiode_panel,
                        'slice_image_panel': slice_image_panel,
                        'cell_image_panel': cell_image_panel,
                        'average_panel': average_panel,
                    }
        self.P = PH.Plotter(self.plotspecs, label=False, fontsize=10, figsize=(10.0, 8.0))

        if hist_panel is not None:
            self.plot_hist(self.P.axdict[hist_panel], results)  # PSTH
        # if imagefile is not None:
        #     self.MT.get_image(imagefile)
        #     self.MT.load_images()
        # print (self.MT.imagemetadata)
        # self.MT.show_images()
        # exit(1)
        # self.plot_average_traces(self.P.axdict['C'], self.Data.timebase, self.Data.data_clean)

        ident = 0
        if ident == 0:
            cbar = self.P.axdict[scale_bar]
        else:
            cbar = None
        idm = self.mapfromid[ident]

        spotsize = self.Pars.spotsize
        dt = np.mean(np.diff(self.Data.timebase))
        itmax = int(self.Pars.ar_tstart / dt) - 1
        self.newvmax = np.max(results[measuretype])
        if self.Pars.overlay_scale > 0.0:
            self.newvmax = self.Pars.overlay_scale

        self.newvmax = self.plot_map(
            self.P.axdict[map_panel],
            cbar,
            results["positions"],
            measure=results,
            measuretype=measuretype,
            vmaxin=self.newvmax,
            mapdir = dataset,
            imageHandle=None,  # self.MT,
            imagefile=imagefile,
            angle=rotation,
            spotsize=spotsize,
            whichstim=whichstim,
            average=average,
            markers=markers,
        )
        # print("plot map done)")
        # mpl.show()
        # exit()

        if plotmode == "document":  # always show all responses/events, regardless of amplitude
            self.plot_stacked_traces(
                tb = self.Data.timebase-self.Pars.time_zero,
                mdata = self.Data.data_clean,
                title = dataset,
                results=results,
                ax=self.P.axdict[trace_panel],
                zscore_threshold=zscore_threshold,
                trsel=trsel,
            )  # stacked on right
            trpanel = "E"
            if self.panels["average_panel"] is not None:
                avedata = np.squeeze(np.mean(self.Data.data_clean, axis=0))
                if avedata.ndim > 1:
                    avedata = np.mean(avedata, axis=0)
                dt = np.mean(np.diff(self.Data.timebase))
                self.P.axdict[self.panels["average_panel"]].plot((self.Data.timebase-self.Pars.time_zero), avedata, 'k-', alpha=0.5, linewidth=0.4)
                self.P.axdict[self.panels["average_panel"]].plot(self.Data.raw_timebase, self.Data.raw_data_averaged,
                                                                'r-', linewidth=0.3, alpha=0.5)
                self.P.axdict[self.panels["average_panel"]].set_xlim(0.0, (self.Pars.time_end - self.Pars.time_zero-0.001))

                # self.P.axdict[self.panels["average_panel"]].set_ylabel("Ave I (pA)")
                # self.P.axdict[self.panels["average_panel"]].set_xlabel("T (msec)")
                PH.noaxes(self.P.axdict[self.panels["average_panel"]])
                ntraces = self.Data.data_clean.shape[0]
                cal_height = None# pA
                ylims = self.P.axdict[self.panels["average_panel"]].get_ylim()
    #            print(ylims, np.fabs(ylims[1] - ylims[0]))
                if cal_height == None:
                    cal_height = self.get_calbar_Yscale((np.fabs(ylims[1] - ylims[0]) / 4.0)*1e12)*1e-12
                else:
                    cal_height = 1e-12*cal_height

                PH.calbar(
                    self.P.axdict[self.panels["average_panel"]],
                    calbar=[
                    self.Pars.time_end - 0.1,
                    ylims[0]*0.9,
                    0.05,
                    cal_height
                    ],
                    scale=[1e3, 1e12],
                    axesoff=True,
                    orient="left",
                    unitNames={"x": "ms", "y": "pA"},
                    fontsize=11,
                    weight="normal",
                    font="Arial",
                )
                PH.referenceline(self.P.axdict[self.panels["average_panel"]], np.mean(avedata[:20]))
                self.plot_timemarker(self.P.axdict[self.panels["average_panel"]])
        elif plotmode is "publication":
            # plot average of all the traces for which score is above threshold
            # and amplitude is in a specified range (to eliminate spikes)
            if zscore_threshold is not None:
                zs = np.max(np.array(results['ZScore']), axis=0)
                i_zscore = np.where(zs > zscore_threshold)[0]
                plotable = self.Data.data_clean.squeeze()[i_zscore, :]
                plotable = plotable[:, :itmax]
                if plot_minmax is not None:
                    iplot = np.where((np.min(plotable, axis=1) > plot_minmax[0]) & (np.max(plotable, axis=1) < plot_minmax[1]))[0]
                    plotable = plotable[iplot, :]
                    print(f"plotMapData:DisplayOneMap:publication mode: Averaging {len(iplot):d} traces, min/max = ", plot_minmax)
                if len(plotable) > 0:
                    d = np.mean(plotable, axis=0)
                    self.P.axdict[trace_panel].plot(self.Data.timebase-self.Pars.time_zero, d-np.mean(d[0:50]))

        self.P.axdict[trace_panel].set_xlim(0, self.Pars.time_end)
        PH.nice_plot(self.P.axdict[trace_panel], direction="outward",
                ticklength=3, position=-0.03)
        ylims = self.P.axdict[trace_panel].get_ylim()
        # PH.referenceline(self.P.axdict[trace_panel], 0.)
        if cal_height == None:
            cal_height = self.get_calbar_Yscale(np.fabs(ylims[1] - ylims[0]) / 4.0)*1e-12
        else:
            cal_height = 1e-12*cal_height
        # PH.noaxes(self.P.axdict[trace_panel])
        self.P.axdict[trace_panel].yaxis.tick_right()
        self.P.axdict[trace_panel].spines.right.set_visible(True)
        self.P.axdict[trace_panel].spines.right.set_position(("outward", 5))
        
        self.P.axdict[trace_panel].yaxis.set_label_position("right")
        self.P.axdict[trace_panel].spines.right.set_color('black')
        self.P.axdict[trace_panel].spines[["bottom", "left"]].set_visible(False)
        self.P.axdict[trace_panel].set_xticklabels([])
        self.P.axdict[trace_panel].set_xticks([])

        # print(cal_height)
        # PH.calbar(
        #         self.P.axdict[trace_panel],
        #         calbar=[
        #             self.Pars.time_end - 0.1,
        #             ylims[0]*0.9,
        #             0.05,
        #             cal_height
        #         ],
        #         scale=[1e3, 1e12],
        #         axesoff=True,
        #         orient="left",
        #         unitNames={"x": "ms", "y": "pA"},
        #         fontsize=11,
        #         weight="normal",
        #         font="Arial",
        #     )

        # PH.calbar(self.P.axdict[panelmap["E"]], calbar=[20e-3, ylim[0]*0.8, 0.05e-3, 100e-12],
        #     scale=[1e-3, 1e-12], 
        #     unitNames={"x": 'pA', "y": 'ms'})

        self.plot_event_traces(
            evtype="avgevoked",
            mdata=self.Data.data_clean,
            trace_tb=self.Data.timebase,
            datatype = datatype,
            ax=self.P.axdict[evoked_panel],
            results = results,
            tpre=self.Pars.template_pre_time,
            zscore_threshold=zscore_threshold,
            plot_minmax=plot_minmax,
            scale=scf,
            label=label,
            rasterized=rasterized,
        )
        self.plot_event_traces(
            evtype="avgspont",
            mdata=self.Data.data_clean,
            trace_tb=self.Data.timebase,
            datatype = datatype,
            ax=self.P.axdict[spont_panel],
            results = results,
            tpre=self.Pars.template_pre_time,
            zscore_threshold=zscore_threshold,
            plot_minmax=plot_minmax,
            scale=scf,
            label=label,
            rasterized=rasterized,
        )

        if self.Data.photodiode is not None and not "LED" in str(dataset.name):
            if len(self.Data.photodiode_timebase) > 0:
                    self.plot_photodiode(
                    self.P.axdict[photodiode_panel],
                    self.Data.photodiode_timebase[0],
                    self.Data.photodiode,
                )
            self.P.axdict[photodiode_panel].set_xlim(0., self.Pars.time_end - self.Pars.time_zero)
            PH.nice_plot(self.P.axdict[photodiode_panel], direction="outward",
                ticklength=3, position=-0.03)
        elif "LED" in str(dataset.name):
            lbt = self.Data.laser_blue_timebase.squeeze()
            if lbt.shape[0] == 0:
                lbt = np.squeeze(lbt)
            else:
                lbt = self.Data.laser_blue_timebase
                self.P.axdict[photodiode_panel].plot(
                lbt,
                self.Data.laser_blue_pCell,
                'b-',
            )
            PH.nice_plot(self.P.axdict[photodiode_panel], direction="outward",
                ticklength=3, position=-0.03)  
            self.P.axdict[photodiode_panel].set_ylabel("Laser Command (V)")
            self.P.axdict[photodiode_panel].set_xlim(0., self.Pars.time_end - self.Pars.time_zero)

        
        # mpl.show()
        return True  # indicated that we indeed plotted traces.
