import sys
from dataclasses import dataclass, field

if sys.version_info[0] < 3:
    print("Shuffler Requires Python 3")
    exit()
from typing import Tuple, Union
import awkward
import matplotlib.pyplot as mpl
import matplotlib.patches as mpatches
from matplotlib.backends.backend_pdf import PdfPages
from mpl_toolkits.axes_grid1 import make_axes_locatable
import ephys.ephys_analysis.poisson_score as EPPS
import numpy as np
import pandas as pd
import pylibrary.plotting.plothelpers as PH
import pylibrary.tools.cprint as CP
import scipy.special as scsp
import scipy.stats as stats
import ephys.tools.functions as EFuncs
import ephys.mapanalysistools.compute_scores as ECS

cprint = CP.cprint


# The Event datalass defines the times and amplitudes of events.
# it is meant to be used as an element in an array of SpotData.
# It defines ONE event
@dataclass
class Event:
    time: float = np.nan
    amplitude: float = np.nan


# The SpotData dataclass holds a list of event objects (Event)
# from ONE spot (trace).
@dataclass
class SpotData:
    _events: list[Event] = field(default_factory=list)
    pos = (-1, -1)  # optional x,y position
    spot_id: int = -1  # optional spot id

    def set_events(self, events: list[Event], event_id: int = -1):
        """set_events Store a list of events into a SpotData

        Parameters
        ----------
        events : list[Event]
            a list Events to be stored
        event_id : int, optional
            an ID to tag the events, by default -1
        """
        self.spot_id = event_id
        # assert isinstance(events, list)
        # assert len(self._events)==0 # set_events can only be called once per instantiation
        # assert all(isinstance(e, Event) for e in events)
        # print("spotdata setting events: ", self.spot_id, len(events), events[0])
        # assert not isinstance(events[0], list)
        self._events = events

    def get_events(self):
        # get all events in the spot
        return self._events

    def get_event_by_index(self, index: int):
        # get one event, by index in the array
        return self._events[index]

    def get_times(self):
        # get an array of the event times only
        return np.array([t.time for t in self._events])

    def get_amplitudes(self):
        # get an array of the event amplitudes only
        return np.array([e.amplitude for e in self._events])

    def len(self):
        # get number of events in the spot
        return len(self._events)

    def event_count(self):
        # alternate name for len()
        return len(self._events)


""" 
The MapData class holds a list of SpotData traces for multiple spots/trials
from one "map" as done in Acq4.
Represents ONE trial (repeat) of a map.
"""


@dataclass
class MapData:
    _map_data: list[SpotData] = field(default_factory=list)

    def append_data(self, spot_data: SpotData):
        # add a spot's data to the map
        assert isinstance(spot_data, SpotData)
        self._map_data.append(spot_data)

    def set_data(self, spot_datas: list[SpotData]):
        # set a spot's data in the map, at index (or append if index<0)
        self._map_data = spot_datas

    def get_map_data(self):
        # get all event data in the map
        return self._map_data

    def get_spot(self, index: int):
        # get one spot's data, based on the spot index
        return self._map_data[index]

    def get_times(self, index: int):
        # get the event times for one spot
        return np.array(self.get_spot(index).get_times())

    def get_array_of_times(self):
        # get all event times in the map as a 2D array (spots x events)
        # the result is a list - it is inhomogeneous since spots have different numbers of events
        all_times = []
        for i_spot, spot in enumerate(self._map_data):
            all_times.append(spot.get_times())
        return all_times

    def get_array_of_amplitudes(
        self,
    ):
        # get the event amplitudes for one spot
        all_amplitudes = []
        for i_spot, spot in enumerate(self._map_data):
            all_amplitudes.append(spot.get_amplitudes())
        return all_amplitudes

    def get_n_spots(self):
        # get number of spots in the map
        return len(self._map_data)

    def event_count(self):
        # get total number of events in the map
        total = 0
        for i_spot, spot in enumerate(self._map_data):
            total += len(spot.get_events())
        return total

    def plot_map(
        self,
        stim_times: Union[dict, None] = None,
        ax: Union[mpl.axes, None] = None,
        plot_map: bool = True,
    ):
        """plot_map: Show the current map spots"""

        if not plot_map:
            return
        if ax is None:
            f, ax = mpl.subplots(1, 1)
        nspots = self.get_n_spots()
        mean_rate = self.event_count() / (nspots * 1.0)
        # print("Mean event rate: ", mean_rate, " Hz")
        all_times = []
        all_spots = []
        for i_spot, spot in enumerate(self._map_data):
            times = spot.get_times()
            all_times.extend(times)
            all_spots.extend([i_spot] * len(times))
            amps = spot.get_amplitudes()
            ax.plot([0, 1.0], [i_spot, i_spot], "k-", linewidth=0.3, alpha=0.3)  # baseline
            ax.scatter(times, np.ones_like(times) * i_spot, s=8, marker="|")
        if stim_times is not None:
            for i_stim, stim in enumerate(stim_times):
                twin = [
                    stim["start"] + stim["window_start"],
                    stim["start"] + stim["window_start"] + stim["window_duration"],
                ]
                patch = mpatches.Rectangle(
                    [twin[0], 0],
                    width=stim["window_duration"],
                    height=nspots + 1,
                    color="b",
                    linewidth=0.5,
                    alpha=0.3,
                )
                ax.add_artist(
                    patch,
                )
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Spot Index")
        ax.set_title(f"Map Test Data: {nspots} spots, Mean Rate: {mean_rate:.2f} Hz")
        # 3. Create locatable axes for the marginal histograms
        divider = make_axes_locatable(ax)
        ax_histx = divider.append_axes("top", 1.05, pad=0.025, sharex=ax)
        ax_histy = divider.append_axes("right", 1.05, pad=0.025, sharey=ax)
        # 5. Plot the marginal histograms
        ax_histx.hist(all_times, bins=100, color="skyblue", edgecolor="black")
        ax_histy.hist(
            all_spots, bins=nspots, color="lightcoral", edgecolor="black", orientation="horizontal"
        )

        # 6. Clean up the marginal histogram axes (remove labels and ticks)
        ax_histx.tick_params(axis="x", labelbottom=False)
        ax_histx.tick_params(axis="y", labelleft=False)
        ax_histy.tick_params(axis="x", labelbottom=False)
        ax_histy.tick_params(axis="y", labelleft=False)


def generate_2d_grid_points(x_min, x_max, y_min, y_max, num_x_points, num_y_points):
    """
    Generates a set of evenly spaced 2D points within a rectangular region.
    AI generated.
    Args:
        x_min (float): The minimum x-coordinate of the region.
        x_max (float): The maximum x-coordinate of the region.
        y_min (float): The minimum y-coordinate of the region.
        y_max (float): The maximum y-coordinate of the region.
        num_x_points (int): The number of evenly spaced points along the x-axis.
        num_y_points (int): The number of evenly spaced points along the y-axis.

    Returns:
        numpy.ndarray: A 2D array where each row represents a point (x, y).
    """
    x_coords = np.linspace(x_min, x_max, num_x_points)
    y_coords = np.linspace(y_min, y_max, num_y_points)

    # Create a meshgrid from the x and y coordinate arrays
    X, Y = np.meshgrid(x_coords, y_coords)

    # Combine the X and Y coordinates into a list of (x, y) pairs
    points = np.vstack([X.ravel(), Y.ravel()]).T
    return points


class Shuffler(object):
    """
    This class provides a set of methods to help with performing shuffling or permutation of a
    set of events that includes spontaneous and evoked responses, by separating out
    the evoked responses and then estimating how frequencty they occur in a data set relative
    to time-shuffled versions of the spontaneous events

    """

    def __init__(self):
        pass

    def _make_test_events(
        self,
        nspots: int,
        mean_amp=20.0,
        spontrate=1.0,
        prob: Union[list, np.ndarray] = [0.1],
        stim_times=dict(start=0.3, window_start=0.001, window_duration=0.01),
        prob_resp: list = [0.5],
        plot_map: bool = False,
        maxt=1.0,
        seed=42,
    ):
        """
        Make a test set of events in the format needed for event detection and p calculatoins

        Parameters
        ----------
        nspots : int (no default)
            number of spots or trials in the data

        spontrate :float (default 1.0)
            Spontaneous rate for Poisson process that generates events, in Hz

        prob : list[float] (default [0.1])
            Probability that spots will have a response event added at the stim times.
            If the list has more than one element, then the llist is assumed
            to be aligned with stim_times, and each stimulus time will have the
            designated probability.

        stim_times : dict (default dict(start=0.3, window_start=0.001, window_duration=0.01))
            Dictionary of stimulus times, with keys 'start', 'window_start', and 'window_duration'

        maxt : float (default 1.0)
            Maximum time of events, in seconds (determines the length of each spot/trial)

        seed : int (default 42)
            Random number seed for np.random.seed for this generation instance.
            The default is "of course".

        Returns
        --------
        test_map : MapData class : event times and amplitudes as Events, in a list of
        n_activesites : number of sites that were selected to have an event inserted
        """

        rng = np.random.default_rng()
        test_map = MapData()
        n_activesites = 0
        np.random.seed(seed=seed)
        totalt = nspots * maxt  # total time in seconds
        if spontrate > 0:
            eventintervals = np.cumsum(
                rng.exponential(1.0 / spontrate, int(spontrate * totalt * 2))
            )  # generate event intervals for all spots
        else:
            eventintervals = np.array([])  # no spontaneous events
        # eventintervals = np.cumsum(eventintervals) # turn into event times
        eventintervals = eventintervals[eventintervals < totalt]  # trim to total time
        for i_spot in range(nspots):
            # chop out event intervals per spot from the main array
            spoteventintervals = eventintervals[
                (eventintervals > i_spot * maxt) & (eventintervals <= (i_spot + 1) * maxt)
            ]
            spoteventintervals -= i_spot * maxt  # make relative to spot start time
            # eventintervals = eventintervals[eventintervals > stimtime]
            spos = []

            for ip in range(len(prob)):  # probs and stim times are in order with each other
                if rng.random() > (1.0 - prob[ip]):
                    spoteventintervals = np.append(  # add a response event
                        spoteventintervals,
                        stim_times[ip]["start"] + stim_times[ip]["window_start"] + 0.001,
                    )  # insert one at the stim time
                    # add more of quanta > 1:
                    if self.quanta > 1:
                        for q in range(1, self.quanta):
                            spoteventintervals = np.append(  # add a response event
                                spoteventintervals,
                                stim_times[ip]["start"]
                                + stim_times[ip]["window_start"]
                                + 0.001
                                + 1e-4 * q,  # separate in time slightly
                            )  # insert one at the stim time
                    spos.append(len(spoteventintervals) - 1)
                    # eventintervals = np.append(eventintervals, stimtime+0.005)  # insert one at the stim time
                    n_activesites += 1

                # print("spot ", i+1, " n events: ", len(spoteventintervals), " n inserted: ", len(spos), prob[ip])
            sortintervals = spoteventintervals  # np.argsort(spoteventintervals)
            evamps = np.random.normal(loc=mean_amp, scale=3.0, size=len(spoteventintervals))
            evamps[spos] = mean_amp * self.quanta  # make inserted events larger amplitude
            # for si in spos:  # quantal size 2x...
            #     spoteventintervals = np.append(spoteventintervals, spoteventintervals[si]) # boost inserted amps
            #     evamps = np.append(evamps, mean_amp)
            events = [
                Event(spoteventintervals[i], evamps[i]) for i in range(len(spoteventintervals))
            ]
            spot_events = SpotData()
            spot_events.set_events(events, event_id=i_spot)
            # print("spot event id: ", spot_events.spot_id, " n events: ", len(spot_events.get_events()))
            test_map.append_data(spot_events)
        test_map.plot_map(stim_times=stim_times, plot_map=plot_map)
        # mpl.show()
        # exit()

        return test_map, n_activesites

    def _get_events_in_win(self, spot: SpotData, stim_time: dict) -> list:
        """
        Get the indices of events in the event dictionary in the defined window,
        from one trace in a map
        """
        si = np.argwhere(
            (spot.get_times() >= stim_time["start"] + stim_time["window_start"])
            & (
                spot.get_times()
                <= (stim_time["start"] + stim_time["window_start"] + stim_time["window_duration"])
            )
        )
        return si

    def _count_map_events_in_allwindows(
        self, map_data: MapData, stim_times: dict, tzero=0.0
    ) -> Tuple[int, list]:
        """
        Count the number of events in all of the windows for one map
        Also return the amplitudes of those events.
        """
        trace_sum = 0
        evoked_amps = []
        evoked_times = []
        for i_spot, spot in enumerate(map_data.get_map_data()):
            event_times = spot.get_times()
            event_amplitudes = spot.get_amplitudes()
            if event_times.ndim > 1:
                raise ValueError("Event times array has too many dimensions")
            # for i_stim, stim in enumerate(stim_times):
            #     twin = [
            #         stim["start"] + stim["window_start"],
            #         stim["start"] + stim["window_start"] + stim["window_duration"],
            #     ]
            #     in_window = np.argwhere((event_times > twin[0]) & (event_times <= twin[1]))
            #     # print("\n", i_spot, i_stim, "min/max twin: ", twin, " ts: ", ts)
            #     # print("in window: ", in_window)
            #     # print("spot.get_amplitudes(i_spot): ", map_data.get_amplitudes(i_spot))
            #     if len(in_window) > 0:
            #         trace_sum += len(in_window[0])
            #         evoked_amps.extend([float(event_amplitudes[e]) for e in in_window[0]])
            #         evoked_times.extend([float(event_times[e]) for e in in_window[0]])
        # print("evamps: ", evamps)
        # print("evtimes: ", evtimes)
        # exit()
        return trace_sum, evoked_amps

    def _get_spont_isis(self, map_data: MapData, stim_times: dict):
        """
        Get spontaneous events and compute isi's from a map
        To get spontaneous events, the possible evoked responses
        following a stimulus are removed from the event list.
        The ISIs are then computed from the sections of events outside
        the response windows (but, not including the empty response window).
        It returns
        arrays of the event times as ISIs, and the amplitudes of those events

        Parameters
        ----------
        map_data: MapData: a list of SpotData objects, each containing a list of
            Event objects with time and amplitude attributes

        stim_times : dict
            times for response analysis window, in format {stimtime, start, duration}
            where stim is the stimulus delivery time (in seconds), start is the
            minimum latency after the stimulus to measure events in,
            and duration is the length of time after the minimum latency
            to consider a response event included.

        Returns
        -------
        evtd : list of interevent intervals
            list is by trace. nan events break traces and
        evad : list of amplitudes for each interval, in soame order

        """
        n_spots = map_data.get_n_spots()
        n_events_total = 0
        n_isi_events = 0
        evtd = [
            []
        ] * n_spots  # list of the events, and at end of processing each trace, the intervals
        evisis = []
        evad = [[]] * n_spots  # amplitudes, matched order to evtd events
        for i_spot in range(n_spots):  # one trace for each spot
            for j, twin in enumerate(stim_times):  # for all stimulus-response windows
                n_events_total += map_data.event_count()
                si = self._get_events_in_win(map_data.get_spot(i_spot), stim_time=twin)
                si = [int(s) for s in si.flatten()]
                evtd[i_spot] = np.append(evtd[i_spot], map_data.get_times(i_spot))
                evad[i_spot] = np.append(evad[i_spot], map_data.get_amplitudes(i_spot))
                if len(si) > 0:
                    for s in si:
                        evtd[i_spot][s] = np.nan
                        evad[i_spot][s] = np.nan
                    # for s in si[0]:
                    #     np.put(
                    #         evtd[i], s, np.nan
                    #     )  # mark events in the response window with nans.
                    #     np.put(evad[i], s, np.nan)
                n_isi_events += len([x for x in si if not np.isnan(x)])
                if j < len(stim_times) - 1:
                    evtd[i_spot] = np.concatenate(
                        (evtd[i_spot], [np.nan])
                    )  # also force a break between twins.
                    evad[i_spot] = np.concatenate((evad[i_spot], [np.nan]))  # keep synchronized...

            evtd[i_spot] = np.diff(evtd[i_spot])  # now take the differences to get the disbribution
            evisis.extend(evtd[i_spot])
        # print("n_events: ", n_events_total, " n_isi_events: ", n_isi_events)
        # print("mean isi: ", np.nanmean(np.array(evisis).ravel()))
        # exit()
        return evtd, evad

    def shuffle_score(self, evp: SpotData, stim_times: dict, nshuffle=10000):
        """
        Shuffle data and compute prob of a spont event occuring in the response window

        Parameters
        ----------
        evp : SpotData
            SpotData object containing Events for each spot/trial.
            Each Event has 'time' and 'amplitude' attributes.
        stim_times : dict
            dictionary defining response windows to test, with keys 'start', 'window_start', and 'window_duration'
        nshuffle : int (default 10000)
            number of shuffles to generate to compute a probability value

        Returns
        -------
        shc : list
            list of probabilities, one for each stimulus window, after shuffling the data
            this represents the base probabiilty of "no events actually occured in the window
            with a probability > that of spontaneous activity"
        """
        ntraces = len(evp)

        evd = np.concatenate(evp)  # combine all traces in map
        evt = evd["time"]
        eva = evd["amp"]
        am = np.mean(eva)
        asd = np.std(eva)
        nbig = np.where(eva > am + 2 * asd)
        shc = np.ones(len(stim_times))

        # remove potential responses in data intervals before shuffle to avoid biases
        for i, tx in enumerate(stim_times):
            si = np.argwhere(
                (evt > tx["start"] + tx["window_start"])
                & (evt <= (tx["start"] + tx["window_start"] + tx["window_duration"]))
            )
            if any(si):
                # print('si', si)
                evt = np.delete(evt, si)
        if evt.shape[0] == 0:  # removed all of them - must have been no spont.
            return shc

        allsh = np.zeros((nshuffle, len(evt) - 1))
        # shuffle data intervals
        for n in range(nshuffle):
            evtd = np.diff(evt)  # just maintain interval distribution
            np.random.shuffle(evtd)  # shuffle order
            allsh[n, :] = np.cumsum(evtd)  # regenerate a sample with same isis
        spl = allsh.ravel()  # np.sort(allsh.ravel())
        # look in every window for spikes
        for i, tx in enumerate(stim_times):
            si = np.argwhere(
                (spl > tx["start"] + tx["window_start"])
                & (spl <= (tx["start"] + tx["window_start"] + tx["window_duration"]))
            )
            #        nspk = len([s for s in spl if (s > np.sum(tx[0:2]) and s < np.sum(tx[0:3]))])  # argwhere is much faster
            if any(si):
                nspk = len(si)
                shc[i] += float(nspk)
        shc = shc / (ntraces * len(stim_times) * nshuffle)
        return shc  # base probability of detecting event in window given input distribution, over all traces

    def shuffle_data1(self, map_data: MapData, stim_times: dict, maxt=1.0, nshuffle=10000):
        """

        Shuffle event time data in a map, returning counts of events
        across all stimulus windows for each trial
        This version combines all data in all trials across the map, while
        excluding responses within a response window (replacing with nan).
        It then computes the mean interevent interval.
        The mean interval is then used to generate Poisson (exponential)
        event trains which are permuted to estimate the chance of an event
        falling in the response window(s). Summary counting is used to
        estimate the probability that observed events are more frequent than
        expected.

        Parameters
        ----------
        map_data : MapData
            MapData.map_data is a list containing SpotData Event objects with 'times' and 'amplitudes'
            for each event in each spot/trial
        stim_times : dict
            dictionary defining response windows to test, with keys 'stim', 'start', and 'duration'
        maxt : float (default 1.0)
            Maximim time in seconds to examine from the dataset.
        nshuffle : int (default 10000)
            number of shuffles to generate to compute a probability value

        Returns
        -------
        shc : list
            list of spike counts, one for each stimulus window, after shuffling the data
            this represents the base probabiilty of "no events actually occured in the window
            with a probability > that of spontaneous activity"
        """
        rng = np.random.default_rng()

        evtd, evad = self._get_spont_isis(map_data, stim_times)

        mean_sp_amp = np.nanmean(
            np.hstack(evad).ravel()
        )  # get the mean amplitude of spontaneous events
        evtd_x = np.hstack(evtd).ravel()  # flatten the isis across ALL points in the map

        data_count, evamps = self._count_map_events_in_allwindows(
            map_data=map_data, stim_times=stim_times, tzero=0.0
        )  # count events in data window in real data
        if all(np.isnan(evtd_x)):
            cprint("yellow", f"No spontaneous intervals events detected")
            if (
                data_count > 0
            ):  # no spontaneous data, so this tells us that there were no events to use.
                return (evtd, 0.0, 0.0)  # high prob
            else:
                return (evtd, 1.0, 0.0)  # no prob, no events

        meanisi = np.nanmean(evtd_x)
        mean_rate = 1.0 / meanisi

        nisi = len(evtd_x)
        ntraces = map_data.get_n_spots()  # number of traces (or spots in a map)
        shuffle_evts = [[]] * nshuffle  # track events in windows for each trace

        trwins = [{} for nt in range(ntraces)]  # build trace windows for each trace
        for tr in range(ntraces):  # compute times for windows across concatenated trains
            # We store the times for each trace, since it is possible
            # that the user specified different stimulus times for each trace
            t0 = tr * maxt
            trtwin = stim_times.copy()
            for nwin, tx in enumerate(stim_times):
                trtwin[nwin] = [
                    tx["start"] + tx["window_start"],
                    tx["start"] + tx["window_start"] + tx["window_duration"],
                ]
            trwins[tr] = trtwin

        for t in range(nshuffle):
            CP.cprint("r", f"Shuffle count: {t}")
            one_shuffled_map = MapData()  # create a new empty map for each shuffle
            print("id: ", id(one_shuffled_map))
            for i_spot in range(ntraces):  # pull data from individual traces
                times = []
                amps = []
                poiss = rng.exponential(  # generate new events for each shuffle trial
                    meanisi, size=nisi * 2  # make extra long, will trim later
                )  # make an artifical distribution that matches
                poiss = np.cumsum(poiss)
                poiss = np.where(poiss < maxt)
                amps.append(np.ones_like(poiss))  # all the same...
                times.append(poiss)  # generate new trains from all intervals...
                spot_data = SpotData()
                # print("init'd event data: ", sh_event_data, sh_event_data.spot_id, id(sh_event_data))
                spot_data.set_events(
                    [Event(times[i], amps[i]) for i in range(len(times))], event_id=i_spot
                )
                # print(f"event data, tr={i_spot:d}, {len(sh_event_data.get_events())}")
                one_shuffled_map.append_data(spot_data)
            one_shuffled_map.plot_map()
            se, sa = self._count_map_events_in_allwindows(
                one_shuffled_map,
                stim_times=stim_times,
            )  # count events in shuffled windows across all traces/trials
            shuffle_evts[t].append(se)
        n_exceed = len(np.where(np.array(shuffle_evts) >= data_count)[0])
        shuffle_probs = n_exceed / float(nshuffle)
        # print('**** Data count: ', data_count, ' mean events: ', np.mean(shuffle_evts), '   shuf probs: ', shuffle_probs)
        return (
            data_count,
            shuffle_evts,
            shuffle_probs,
            mean_rate,
            mean_sp_amp,
        )  # base probability of detecting event in window given input distribution, over all traces
        # and mean amplitude of spontaneous events not in the response window

    # def shuffle_data2(self, evp: SpotData, stim_times: dict, maxt=1.0, nshuffle=10000):
    #     """
    #     Shuffle event time data in a map, returning counts of events
    #     across all stimulus windows for each trial

    #     Unclear that this version works well.

    #     Parameters
    #     ----------
    #     evp : list of numpy record arrays
    #         list containing numpy arrays with 'time' and 'amp' for each event in each spot/trial
    #     stim_times : dict
    #         dictionary defining stimulus windows to test, with keys 'stim', 'start', and 'duration'
    #     maxt : float (default 1.0)
    #         Maximim time in seconds to examine from the dataset.
    #     nshuffle : int (default 10000)
    #         number of shuffles to generate to compute a probability value

    #     Returns
    #     -------
    #     shc : list
    #         list of spike counts, one for each stimulus window, after shuffling the data
    #         this represents the base probabiilty of "no events actually occured in the window
    #         with a probability > that of spontaneous activity"
    #     """

    #     rng = np.random.default_rng()
    #     evtd, evad = self._get_spont_isis(evp, stim_times)
    #     evtd_x = np.hstack(evtd).ravel()  # flatten the isis across ALL points in the map
    #     if all(
    #         np.isnan(evtd_x)
    #     ):  # no spontaneous data, so this tells us that there were no events to use.
    #         cprint("yellow", f"no spontaneous intervals events detected")
    #         return (evtd, 0, 0.0)

    #     mean_sp_amp = np.nanmean(
    #         np.hstack(evad).ravel()
    #     )  # get the mean amplitude of spontaneous events
    #     # print('mean isi: ', evtd_x)
    #     # import matplotlib.pyplot as mpl
    #     # f, ax = mpl.subplots(1,1)
    #     # ax.hist(evtd_x, bins=np.arange(0, 0.5, 0.01))
    #     # mpl.show()
    #     # exit()
    #     ntraces = len(evp)  # number of traces (or spots in a map)
    #     shuffle_evts = np.zeros(ntraces)  # track events in windows for each trace
    #     for n in range(nshuffle):
    #         for t in range(ntraces):
    #             ts = evtd[t].copy()  #  ts = evtd[tr_rand[t]].copy()  # selecct data
    #             if len(ts) == 0:
    #                 shuffle_evts[t] += np.nan
    #             else:
    #                 rng.random.shuffle(ts)  # shuffle interval order
    #                 ts = np.cumsum(ts)  # generate new train
    #                 shuffle_evts[t] += self._get_events_in_allwindows(ts, twin)
    #         # normalize on per-trace (spot) basis
    #     shuffle_probs = np.array(shuffle_evts) / (nshuffle)
    #     print("shuf probs: ", shuffle_probs)
    #     return (
    #         evtd,
    #         shuffle_probs,
    #         mean_sp_amp,
    #     )  # base probability of detecting event in window given input distribution, over all traces
    #     # and mean amplitude of spontaneous events not in the response window

    def detect_events(
        self,
        map_data: MapData,
        stim_times: dict,  # stimulus times as dict with start, window_start, window_end
        mean_spont_amp: Union[float, None] = None,
    ):
        """
        find events in the stimulus window (just detect and count)
        twin is a list of 2-tuples, each defining a response window to test
        returns fraction of events relative to all traces

        Parameters
        ----------
        event_data : list
            list of SpotData objects for each trace

        stim_times : list
            stimulus time window

        Returns
        -------
        shc_n, detected_counts/ntraces, detected_amplitudes
            shc_n : fraction of sites during which event was detected
            counts : estimated frequency of events normalized by traces
            amplitudes : amplitudes of detected events.

        """
        ntraces = len(map_data.map_data)
        if ntraces == 0:
            cprint("red", "ntraces is 0, no events were included")
            return 0, 0, None
        CP.cprint("g", f"Detecting events in {ntraces} traces")
        # events = np.concatenate(evp)  # combine all traces in map
        # print('event times len: ', event_times)
        # event_times_c = np.concatenate(event_times)
        # event_amplitudes_c = np.concatenate(event_amplitudes)
        # print(event_times)
        # event_amplitudes_c = np.hstack(event_amplitudes).ravel()
        # event_times_c = np.hstack(event_times).ravel()
        event_times_c = np.concatenate([x.get_times() for x in map_data.map_data])
        print(event_times_c)
        event_amplitudes_c = np.concatenate([x.get_amplitudes() for x in map_data.map_data])
        mean_amplitude = np.nanmean(event_amplitudes_c)

        sd_amplitude = np.nanstd(event_amplitudes_c)
        nbig = np.where(event_amplitudes_c > mean_amplitude + 2.0 * sd_amplitude)[0]

        detected_counts = np.zeros(ntraces)  # keep tab of events in each window
        detected_amplitudes = [[]] * ntraces  # np.zeros(len(twin))

        for i in range(ntraces):
            (
                detected_counts[i],
                detected_amplitudes[i],
            ) = self._count_events_in_allwindows(
                event_data=map_data.map_data[i], stim_times=stim_times
            )

        shc_n = np.sum([detected_counts > 0]) / (
            float(ntraces) * len(stim_times)
        )  # fraction of sites overall stimuli during which an event was detected

        return (
            shc_n,
            detected_counts / ntraces,
            detected_amplitudes,
        )  # base probability of detecting event in window given input distribution, over all traces

    def z2p(self, z: float):
        """From z-score return p-value."""
        return 0.5 * (1 + scsp.erf(z / np.sqrt(2)))

    def plot_map(self, test_map: MapData, i_map: int = 0, i_repeat: int = 0):

        evh = np.array([])
        # print("evp: ", evp)
        # for i_spot in range(self.nspots):
        #     evh = np.concatenate((evh,
        #         np.array(test_map.get_spot(i_spot).get_times())))
        # evh = evh.ravel()
        fig = mpl.figure()
        ax = mpl.subplot2grid((2, 1), (0, 0), rowspan=3)
        ax2 = mpl.subplot2grid((2, 1), (1, 0), rowspan=1)
        n_spots = test_map.get_n_spots()
        for i_spot in range(n_spots):
            times = test_map.get_times(i_spot)
            ax.scatter(
                times,
                np.ones_like(times) * i_spot + 5,
                s=test_map.get_amplitudes(i_spot),
                marker="|",
            )
            ax.plot([0, 1.0], [i_spot + 5, i_spot + 5], "k-", linewidth=0.5)
            for i_s, st in enumerate(self.stimtimes):
                ax.axvline(st["start"], color="b", linestyle="--", linewidth=0.5)
        ax2.hist(evh, bins=60, alpha=1)
        ax2.set_ylim(0, 100)
        ax.set_title(f"Test events map {i_map+1} repeat {i_repeat+1}")
        ax.set_xlim(0, 0.6)
        fig.tight_layout()
        # mpl.show()
        # exit()

    def test_events_shuffle(self):
        """
        Use shuffle methods (resampling) to directly test event probability

        Parameters
        ----------
        None

        """
        import pylibrary.plotting.plothelpers as PH

        CP.cprint("g", f"\nTesting shuffler with resampling method\n{'='*80:s}\n")

        rc = PH.getLayoutDimensions(np.min((10, self.nmaps)) + 4, pref="width")
        P = PH.regular_grid(
            rc[0],
            rc[1],
            figsize=(10.0, 8),
            margins={
                "leftmargin": 0.07,
                "rightmargin": 0.05,
                "topmargin": 0.1,
                "bottommargin": 0.1,
            },
        )
        binstuff = np.arange(0, 1.0, 0.01)
        binstuff2 = np.arange(0, 20.0, 1)
        axl = P.axarr.ravel()
        x = np.zeros((self.nmaps, self.nrepeats))
        y = []
        PShuffle = np.zeros((self.nmaps, self.nrepeats))
        test_maps = {}
        # print("namps: ", self.nspots, self.nmaps, self.nrepeats)
        for i_repeat in range(self.nrepeats):
            for i_map in range(self.nmaps):
                test_maps[i_map], nactivesites = self._make_test_events(
                    self.nspots,
                    spontrate=self.srate,
                    prob=self.probs,
                    stim_times=self.stimtimes,
                    maxt=self.maxt,
                    seed=i_map,
                    plot_flag=False,
                )
                if i_map > 2:
                    mpl.show()
                    exit()
                #         else:
                #             return
                #         # self.plot_map(test_maps[i_map])

                ms = np.zeros(self.nspots)
                for i_spot in range(self.nspots):
                    spot_times = test_maps[i_map].get_times(i_spot)
                    if len(spot_times) > 0:
                        ms[i_spot] = len(spot_times)
                rate = np.sum(ms) / (self.maxt * self.nspots)
                print(
                    f"Repeat: {i_repeat+1:d}/{self.nrepeats:d}  map: {i_map+1:d}/{self.nmaps}   prot: {self.probs[i_repeat]:.4f}"
                )
                # print("    stimulus times: ", self.stimtimes)
                (
                    data_count,
                    shuffle_evts,
                    shuffle_probs,
                    mean_rate,
                    mean_ev_amp,
                ) = self.shuffle_data1(
                    test_maps[i_map], self.stimtimes, nshuffle=self.nshuffle, maxt=self.maxt
                )
                axl[-2].hist(shuffle_evts, bins=binstuff2, histtype="stepfilled", align="right")
                print("data count, max# sh_evts: ", data_count, np.max(shuffle_evts))
                PShuffle[i_repeat, i_map] = shuffle_probs

                frac_detected, detevt, detamp = self.detect_events(
                    # evdata,
                    test_map,
                    stim_times=self.stimtimes,
                    mean_spont_amp=1.0,
                )
                shuffle_mean = np.mean(shuffle_evts)
                shuffle_sd = np.std(shuffle_evts)
                Z = (data_count - shuffle_mean) / shuffle_sd
                Zprob = self.z2p(Z)
                print(f"Z: {Z:f}    Prob: {1.0-Zprob:e}")
                print(self.stimtimes, mean_rate)
                pprob = poissonProb(
                    float(data_count) / self.nspots, self.stimtimes[0][2], mean_rate
                )
                print("Poisson prob: ", pprob)
                x[i, k] = nactivesites
                print("frac det: ", frac_detected)
                if frac_detected > 0:
                    print(f"              Raw Shuffle Prob: {shuffle_probs:.6e}")
                    print(f"  original sites: {nactivesites:d}", end="")
                    print(f", fracsites: {frac_detected:5.3f}", end="")
                    print(f", Amplitudes: {np.mean(detamp):6.2f}")
                else:  # no events detected, so y is 0
                    y.append(0.0)
                c = f"{(k/self.nrepeats):f}"
                n, bins, patches = axl[i].hist(
                    evt, bins=binstuff, histtype="stepfilled", color=c, align="right"
                )
            #     for s in evl['stimtimes']['start']:
            # #        print('s: ', s)
            #         axl[i].plot([s, s], [0, 40], 'r-', linewidth=0.5)
            # axl[i].set_ylim(0, 100)
            # axl[i].set_title(f"Mean expected={m_shuffle_event:.3f}, active sites={nactivesites:.0f}\n Z={y[-1]:7.4f} PofZ: {(1.0-self.z2p(y[-1])):6.4f}", fontsize=6)
            P.figure_handle.suptitle(
                f"Testing shuffler: srate={self.srate:.2f},  Q={self.quanta:d} probs={self.probs},nshuffle={self.nshuffle:d}"
            )
            # PofZ = 1.0-self.z2p(y)

        for i in range(np.min((10, self.nmaps))):
            axl[i].set_title(
                f"Active sites={np.mean(x[i,:]):.0f}\n Mean ShuffleProb: {np.mean(PShuffle[i,:]):8.6f}",
                fontsize=6,
            )
            axl[i].set_xlim(0, self.maxt)

        axl[-1].plot(x, PShuffle, "ok", markersize=4)
        # axl[-1].plot(x, PofZ, 'ok', markersize=4)
        axl[-1].set_ylim(0, 1.1 * np.max(PShuffle))
        axl[-1].set_ylabel("P value from z")
        axl[-1].set_xlim(0, 1.1 * np.max(x))
        axl[-1].set_xlabel("# active nsites")
        PH.talbotTicks(axl[-1], tickPlacesAdd={"x": 0, "y": 0}, floatAdd={"x": 2, "y": 2})

        mpl.show()

    def compute_ZScore(self, map_events, stim_times, plot: bool = False, ax=None):
        """
        Compute Z-score from data count and shuffle event counts
        Event_times is a list of arrays, one per trace/spot in the map

        Parameters
        ----------
        data_count : int
            number of events detected in the response window in the data
        shuffle_evts : list
            list of event counts in the response window across all shuffles

        Returns
        -------
        Z : float
            Z-score value
        PofZ : float
            probability of observing this Z-score by chance

        """
        close_fig = False
        trace_spacing = 1.5
        dt = 1e-4
        alpha = EFuncs.alpha(t=np.arange(0, 0.010, dt), tau=0.0015)
        alpha = np.concatenate(
            (np.zeros_like(alpha), alpha)
        )  # The convolved event is centered at the dirac pulse time
        # print(len(event_times))
        waves = []
        rnd = np.random.RandomState(0)
        timebase = np.arange(0, self.maxt, dt)
        pre_std = None
        pre_mean = None
        twin_base = [0, stim_times[0]["start"]]
        twin_resp = [
            stim_times[0]["start"] + stim_times[0]["window_start"],
            stim_times[0]["start"]
            + stim_times[0]["window_start"]
            + stim_times[0]["window_duration"],
        ]
        zscores = np.zeros(len(map_events))
        if plot and ax is None:
            f, ax = mpl.subplots(1, 1)
            close_fig = True
        for imap, evt_times in enumerate(map_events):
            if len(evt_times) == 0:
                continue
            times = np.zeros_like(timebase)
            iti = (evt_times / dt).astype(int)
            times[iti] = 1.0 + 0.15 * rnd.randn(len(iti))
            wave = np.convolve(a=times, v=alpha, mode="same")
            waves.append(wave)

        # compute the pre-stim mean and std across all waves for Qs
        pre_mean = np.mean(
            np.hstack([waves[i][timebase <= twin_base[1]] for i in range(len(waves))])
        )
        pre_std = np.std(np.hstack([waves[i][timebase <= twin_base[1]] for i in range(len(waves))]))

        for imap, wave in enumerate(waves):
            zscores[imap] = ECS.ZScore(
                timebase,
                waves[imap],
                pre_std=pre_std,
                pre_mean=pre_mean,
                twin_base=twin_base,
                twin_resp=twin_resp,
            )

            if plot:
                ax.plot(timebase, waves[imap] + imap / trace_spacing, linewidth=0.5)
                if zscores[imap] > 1:
                    ax.text(
                        twin_resp[1], imap / trace_spacing, f"Z={zscores[imap]:.2f}", fontsize=10
                    )
        if plot:
            rect = mpatches.Rectangle(
                [twin_resp[0], -1],
                twin_resp[1] - twin_resp[0],
                len(waves) / trace_spacing + 1,
                color="blue",
                alpha=0.1,
            )
            ax.add_artist(rect)
        if plot and close_fig:
            mpl.show()
            exit()
        return zscores

        shuffle_mean = np.mean(shuffle_evts)
        shuffle_sd = np.std(shuffle_evts)
        Z = (data_count - shuffle_mean) / shuffle_sd
        PofZ = self.z2p(Z)
        return Z, PofZ

    def generate_example_traces(self, rate: float = 0.5, prob: float = 0.2, ax=None):
        """
        Generate example traces for a given spontaneous rate and event probability

        Parameters
        ----------
        rate : float
            spontaneous event rate in Hz
        prob : float
            probability of evoked response event

        Returns
        -------
        test_map : MapData
            generated MapData object with events

        """
        print(f"Rate: {rate:.1f}  prob: {prob:.2f}, Q: self.quanta: {self.quanta:d}")
        test_map, n_active_sites = self._make_test_events(
            self.nspots,
            spontrate=rate,
            prob=[prob],
            stim_times=self.stimtimes,
            maxt=self.maxt,
            seed=0,
            plot_map=False,
        )
        if ax is None:
            plot = False
        else:
            plot = True
        z_scores_map = self.compute_ZScore(
            test_map.get_array_of_times(), self.stimtimes, plot=plot, ax=ax
        )
        return test_map

    def test_events_poisson(self, rate: Union[float, None] = None):
        """
        Like test events, except computes Poisson Score from Chase and Young
        Based on code from Luke Campagnola

        Each map is generated with a different probability of response events,
        using _make_test_events.
        The reponse events are single "quanta" added to a spontaneous Poisson process.

        Parameters
        ----------
        None
        """
        import matplotlib.pyplot as mpl

        if rate is not None:
            self.srate = rate

        plot_maps = True
        plot_hists = not plot_maps
        rc = PH.getLayoutDimensions(np.min((self.nmaps, 10)) + 4, pref="width")
        P = PH.regular_grid(
            rc[0],
            rc[1],
            figsize=(10.0, 8),
            margins={
                "leftmargin": 0.07,
                "rightmargin": 0.05,
                "topmargin": 0.1,
                "bottommargin": 0.1,
            },
        )
        binstuff = np.arange(0, 0.6, 0.005)
        axl = P.axarr.ravel()

        x = []
        mscores = np.zeros(self.nmaps)
        n_sig_mscores = np.zeros(self.nmaps) * np.nan  # number of significant zscores per map
        map_probs = {}
        min_prob = np.zeros(self.nmaps)
        tpr = []
        tevt = []
        min_p = np.zeros(self.nmaps)  # min detection p values from poisson_score
        zscores = np.zeros(self.nmaps)  # zscores from compute_ZScore for each map
        n_sig_zscores = np.zeros(self.nmaps)  # number of significant zscores per map
        print("spont rate: ", self.srate)
        print("Probs: ", self.probs)
        range_probs = np.linspace(0.0, 1.0, self.nmaps)
        n_active_sites = [0] * self.nmaps
        evp = [None] * self.nmaps
        maps_plotted = 0
        for i_map in range(self.nmaps):
            evp[i_map], n_active_sites[i_map] = self._make_test_events(
                self.nspots,
                spontrate=self.srate,
                prob=[range_probs[i_map]],  # single prob for each map
                stim_times=self.stimtimes,
                maxt=self.maxt,
                seed=i_map,
            )
            # get scores for this map
            plot_flag = False
            # if i_map == 99:
            #     plot_flag = True

            z_scores_map = self.compute_ZScore(
                evp[i_map].get_array_of_times(), self.stimtimes, plot=plot_flag
            )
            # print("Z scores map: ", z_scores_map)
            sig_zscores = z_scores_map[z_scores_map > 1.96]
            n_sig_zscores[i_map] = len(sig_zscores)
            if len(sig_zscores) > 0:
                zscores[i_map] = np.mean(sig_zscores)  # only average significant Z
            tMax = self.maxt  # self.stimtimes[0][0] + self.stimtimes[0][2]
            mscores_map, map_probs[i_map] = EPPS.PoissonScore.score(
                evp[i_map].get_array_of_times(),
                evp[i_map].get_array_of_amplitudes(),
                rate=self.srate,
                tMax=tMax,
                normalize=True,
            )
            sig_mscores = mscores_map[mscores_map > 1.3]
            n_sig_mscores[i_map] = len(sig_mscores)
            if len(sig_mscores) > 0:
                mscores[i_map] = np.mean(sig_mscores)  # only average significant Z

            sprobs = ", ".join([f"{p:.3e}" for p in self.probs])
            min_prob[i_map] = np.min(map_probs[i_map])
            # print(f"Testing {i_map:d} Event Prob: {self.probs!s}")
            # print(
            #     f"    Log mscore= {np.log10(mscores[i_map]):.4f} mscore: {mscores[i_map]:.2f} prob: {min_prob[i_map]:.3e} : {n_active_sites[i_map]:d} of {self.nspots:d} sites active"
            # )
            x.append(n_active_sites)

            evt = []
            for j, p in enumerate(evp[i_map].get_array_of_times()):
                evt.extend(p)
            evt = np.array(evt)
            ievt = np.argsort(evt)
            tpr.append(map_probs[i_map][ievt])
            tevt.append(evt[ievt])

            # plot a few examples
            if i_map % 10 == 0 and maps_plotted < 10:  #  np.min((self.nmaps, 10)):
                # for j in range(len(evt)):
                #     if prob[j] < 0.5 and (evt[j] > 0.1 and evt[j] < 0.11):
                #         print('j, evt, p: ', j, evt[j], prob[j])
                if plot_hists:
                    n, bins, patches = axl[maps_plotted].hist(
                        evt, bins=binstuff, histtype="stepfilled", align="mid"
                    )
                    axl[maps_plotted].set_ylim(0, np.max(n) * 1.1)

                if plot_maps:
                    pos = generate_2d_grid_points(
                        x_min=0,
                        x_max=10,
                        y_min=0,
                        y_max=10,
                        num_x_points=int(np.sqrt(self.nspots)),
                        num_y_points=int(np.sqrt(self.nspots)),
                    )
                    axl[maps_plotted].set_facecolor("k")
                    axl[maps_plotted].scatter(
                        pos[:, 0],
                        pos[:, 1],
                        c=zscores,
                        cmap="gnuplot2",
                        s=20.0,
                        marker="o",
                        linewidths=0.1,
                        edgecolors="w",
                        alpha=0.9,
                    )
                    axl[maps_plotted].set_aspect("equal", "box")
                if mscores[i_map] == 0:
                    ms = 0.0
                else:
                    ms = np.log10(mscores[i_map])
                axl[maps_plotted].set_title(
                    f"Map {i_map:d} mscore={ms:.3f}, sites={n_active_sites[i_map]:d}",
                    fontsize=9,
                )
                maps_plotted += 1

            axl[-4].plot(tevt[-1], tpr[-1], "-", linewidth=0.15)
            # get min p vaue in the first stim window
            win0 = (self.stimtimes[0]["start"] + self.stimtimes[0]["window_start"],)
            win1 = (
                self.stimtimes[0]["start"]
                + self.stimtimes[0]["window_start"]
                + 2 * self.stimtimes[0]["window_duration"]
            )
            swi = np.argwhere((tevt[-1] >= win0) & (tevt[-1] <= win1))
            if len(swi) > 0:  # find the lowest probability eventin the first stim window
                # print(np.min(tpr[-1][swi]))
                min_p[i_map] = np.min(tpr[-1][swi])
            axl[-3].plot(tevt[-1][swi], tpr[-1][swi], "o-", markersize=0.5, linewidth=0.2)

        axl[-5].plot(n_active_sites, n_sig_zscores, "ok", markersize=3)
        axl[-5].set_ylabel("ZScore", fontsize=8)
        axl[-5].set_xlabel("# active sites", fontsize=8)
        axl[-3].set_ylim(0, 1)
        axl[-3].set_ylabel("Poiss expect p-value", fontsize=8)
        axl[-3].set_xlabel("Event time (s)", fontsize=8)

        P.figure_handle.suptitle(
            f"testing Shuffler with Poisson Prob: srate={self.srate:.2f} Q={self.quanta:d} Event Prob: {self.probs!s}"
        )

        # make a histogram of the tpr values across all maps, all events
        nbins = 51
        pbins = np.linspace(0.0, 1.0, nbins)
        phist = np.zeros(len(pbins)) * np.nan
        p_data = [[]] * nbins
        # for i in range(pbins.shape[0]-1):
        for ispot, tspot in enumerate(tevt):
            for ibin in range(len(pbins) - 1):
                values = np.argwhere((tevt[ispot] >= pbins[ibin]) & (tevt[ispot] < pbins[ibin + 1]))
                values = list(np.ravel(values))
                if len(values) > 0:
                    if len(p_data[ibin]) == 0:
                        p_data[ibin] = tpr[ispot][values]
                    else:
                        p_data[ibin] = np.concatenate((p_data[ibin], tpr[ispot][values]))

                    # print("ispot: ", ispot, "bin: ", pbins[ibin], "values: ", tpr[ispot][values][0], len(values), tevt[ispot][values])
                else:
                    p_data[ibin] = [np.nan]  # np.concatenate(([np.nan], tpr[ispot]))

        adata = awkward.Array(p_data)
        pmean = awkward.ravel(awkward.mean(adata, axis=1))
        pstd = awkward.ravel(awkward.std(adata, axis=1))
        # print(len(pbins), len(pmean), len(pstd))
        axl[-4].errorbar(pbins, pmean, yerr=pstd, fmt="k-", linewidth=0.5)
        axl[-4].set_ylim(0, 1)
        axl[-4].set_ylabel("Poiss expect p-value", fontsize=8)
        ms = [np.log10(m) if not np.isnan(m) and m > 0 else np.nan for m in mscores]
        axl[-2].plot(n_active_sites, ms, "ok", markersize=3)
        axl[-2].set_ylim(0, 1.1 * np.nanmax(ms))
        axl[-2].set_ylabel("mscore")
        axl[-2].set_xlabel("Time (sec)")
        axl[-2].set_xlim(0, 1.1 * np.max(x))
        axl[-2].set_xlabel("# active sites")
        PH.talbotTicks(axl[-3], tickPlacesAdd={"x": 0, "y": 0}, floatAdd={"x": 0, "y": 2})
        # show minimum p value for first stim window
        amin = np.argsort(min_p)
        min_p[min_p <= 1e-6] = 1e-6
        # axl[-1].plot(np.arange(self.nmaps), np.log10(min_p[amin]), "ok", markersize=3)
        # axl[-1].plot([0, self.nmaps], [np.log10(1e-3), np.log10(1e-3)], "r--", linewidth=0.3)
        axl[-1].plot(range_probs, np.log10(min_p), "ok", markersize=3)
        axl[-1].plot(
            [range_probs[0], range_probs[-1]],
            [np.log10(1e-3), np.log10(1e-3)],
            "r--",
            linewidth=0.3,
        )
        axl[-1].plot(
            [range_probs[0], range_probs[-1]],
            [np.log10(5e-2), np.log10(5e-2)],
            "b--",
            linewidth=0.3,
        )
        axl[-1].text(
            1.0,
            np.log10(1e-3) + 0.1,
            "p=0.001",
            color="r",
            fontsize=8,
            ha="left",
            transform=axl[-1].get_xaxis_transform(),
        )
        axl[-1].text(
            1.0,
            np.log10(5e-2) + 0.1,
            "p=0.05",
            color="b",
            fontsize=8,
            ha="left",
            transform=axl[-1].get_xaxis_transform(),
        )
        axl[-1].set_ylabel("Log10 Min p-value", fontsize=8)
        # mpl.tight_layout()
        return P.figure_handle

    def set_pars(self, probtype="single"):
        self.nrepeats = 1
        self.nmaps = 100
        self.nspots = 100
        self.srate = 5.0
        self.stimtimes = [
            dict(start=0.1 + i * 0.1, window_start=0.001, window_duration=0.01) for i in range(5)
        ]
        self.maxt = 1.0
        self.quanta = 2
        self.nshuffle = 100
        if probtype == "single":
            self.probs = [0.1]
        else:
            self.probs = [0.2 * (i + 1) for i in range(len(self.stimtimes))]


def main(quanta=1):
    import matplotlib.pyplot as mpl

    S = Shuffler()
    S.set_pars(probtype="single")
    srates = [1, 5, 10, 20, 50]
    pn = 0
    k = 0
    S.quanta = quanta
    with PdfPages(f"test_detect_SR_Q.pdf") as pdf:
        for k in range(len(srates)):
            for q in [1, 2, 3, 5, 10]:
                S.srate = srates[k]
                S.quanta = q
                fig = S.test_events_poisson(rate=srates[k])
                pn += 1
                mpl.text(0.95, 0.02, s=f"Page {pn:d}", fontsize=7, transform=fig.transFigure)
                pdf.savefig(fig)  # or you can pass a Figure object to pdf.savefig
                mpl.close()

    # S.set_pars(probtype="multi")
    # S.test_events_shuffle()


def main_gen_traces(quanta=2):
    import matplotlib.pyplot as mpl

    S = Shuffler()
    S.set_pars(probtype="single")
    srates = [1, 5, 10, 20, 50]
    pn = 0
    k = 0
    prob = 0.1
    S.quanta = quanta
    with PdfPages(f"test_traces_SR_Q.pdf") as pdf:
        for k in range(len(srates)):  # different rates
            for q in [1, 2, 3, 5, 10]:  # different quanta
                S.srate = srates[k]
                S.quanta = q
                fig, ax = mpl.subplots(1, 1, figsize=(8, 10))
                test_map = S.generate_example_traces(rate=srates[k], prob=prob, ax=ax)
                fig.suptitle(
                    f"Example traces: Spont Rate={srates[k]:.1f} Hz, Q={q:d}, prob={prob:.2f}"
                )
                pn += 1
                fig.text(
                    0.95,
                    0.02,
                    s=f"Page {pn:d}",
                    fontsize=7,
                )
                pdf.savefig(fig)
                mpl.close(fig)


if __name__ == "__main__":
    # main(quanta=2)
    main_gen_traces(quanta=2)
