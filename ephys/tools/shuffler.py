import sys

if sys.version_info[0] < 3:
    print("Shuffler Requires Python 3")
    exit()
from typing import Union, Tuple
import pandas as pd
import numpy as np
import ephys.ephys_analysis.poisson_score as EPPS
import pylibrary.tools.cprint as CP
import pylibrary.plotting.plothelpers as PH
import scipy.special as scsp
import scipy.stats as stats

cprint = CP.cprint

def poissonProb(n, t, l, clip=False):
    """
    For a poisson process, return the probability of seeing at least *n* events in *t* seconds given
    that the process has a mean rate *l*.
    """
    if l == 0:
        if np.isscalar(n):
            if n == 0:
                return 1.0
            else:
                return 1e-25
        else:
            return np.where(n==0, 1.0, 1e-25)

    p = stats.poisson(l*t).sf(n)
    if clip:
        p = np.clip(p, 0, 1.0-1e-25)
    return p


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
        prob=0.1,
        stimtime=Union[None, float],
        maxt=1.0,
        seed=1,
    ):
        """
        Make a test set of events in the format needed for event detection and p calculatoins

        Parameters
        ----------
        nspots : int (no default)
            number of spots or trials in the data

        spontrate :float (default 1.0)
            Spontaneous rate for Poisson process, in Hz

        prob : float (default 0.1)
            probability that spots will have a response event added add the stimtime

        stimtime : list (default None)
            List of stim times at which events will be inserted

        maxt : float (default 1.0)
            Maximum time of events, in seconds

        seed : int (default 1)
            Randpm number seed for np.random.seed for this generation instance

        Returns
        --------
        evp : event times as a list of record arrays
        nsites : number of sites that were selected to have an event inserted
        """

        evp = []
        nsites = 0
        np.random.seed(seed=seed)
        totalt = nspots * maxt  # total time in seconds
        if spontrate > 0:
            eventintervals = np.cumsum(
                np.random.exponential(1.0 / spontrate, int(spontrate * totalt * 2))
            )  # generate enough intervals
        else:
            eventintervals = np.array([])
        for i in range(nspots):
            # chop out event intervals per spot from the main array
            spoteventintervals = eventintervals[
                (eventintervals > i * maxt) & (eventintervals <= (i + 1) * maxt)
            ]
            spoteventintervals -= i * maxt
            # eventintervals = eventintervals[eventintervals > stimtime]
            spos = []
            if stimtime is not None and np.random.random() > (1 - prob):
                spoteventintervals = np.append(
                    spoteventintervals, stimtime + 0.001
                )  # insert one at the stim time
                spos.append(len(spoteventintervals) - 1)
                # eventintervals = np.append(eventintervals, stimtime+0.005)  # insert one at the stim time
                nsites += 1
            sortintervals = np.argsort(spoteventintervals)
            # print(eventintervals)
            ev = np.empty(
                len(spoteventintervals), dtype=[("time", float), ("amp", float)]
            )
            ev["time"] = spoteventintervals[sortintervals]
            evamps = np.random.normal(
                loc=mean_amp, scale=3.0, size=len(spoteventintervals)
            )
            evamps[spos] *= 2.0
            ev["amp"] = evamps[sortintervals]
            evp.append(ev)

        return evp, nsites

    def _get_events_in_win(self, evt: dict, tx: Union[list, np.ndarray]) -> list:
        """
        Get tne number of events in the current window
        """
        si = np.argwhere((evt >= np.sum(tx[0:2])) & (evt <= np.sum(tx[0:3])))
        return si

    def _count_events_in_allwindows(
        self, event_times: list, event_amps: list, twin: list, tzero=0.0
    ) -> Tuple[int, list]:
        """
        Count the number of events in all of the windows for one trace
        """
        trace_sum = 0
        evamps = []
        ts = event_times[pd.notnull(event_times)]
        for j, tx in enumerate(twin):
            tx = list(tx)
            tx[0] += tzero  # offset the window without widening it.
            ev_indices = np.argwhere(
                (ts >= np.nansum(tx[0:2])) & (ts <= np.nansum(tx[0:3]))
            )
            if len(ev_indices) > 0:
                trace_sum += len(ev_indices[0])
                evamps.append([event_amps[e] for e in ev_indices[0]])
        return trace_sum, evamps

    def _get_spont_isis(self, evp: dict, twin: Union[list, np.ndarray]):
        """
        Get spontaneous events and compute isi's
        We remove the possible evoked responses from the traces, and return
        arrays of the event times as ISIs, and the amplitudes of those events

        Parameters
        ----------
        evp: list of dicts with time and amplitide for events
            each list element is a trace

        twin : list or tuple
            times for analysis window, in format [start, deadtime, duration]

        Returns
        -------
        evtd : list of interevent intervals
            list is by trace. nan events break traces and
        evad : list of amplitudes for each interval, in soame order

        """
        ntraces = len(evp)
        evtd = [
            []
        ] * ntraces  # list of the events, and at end of processing each trace, the intervals
        evad = [[]] * ntraces  # amplitudes, matched order to evtd events
        for i in range(ntraces):  # one trace for each spot
            for j, tx in enumerate(twin):  # for all stimulus-response windows
                si = self._get_events_in_win(evp[i]["time"], tx)
                evtd[i] = np.append(evtd[i], evp[i]["time"])
                evad[i] = np.append(evad[i], evp[i]["amp"])
                if len(si) > 0:
                    for s in si[0]:
                        np.put(
                            evtd[i], s, np.nan
                        )  # mark events in the response windo with nans.
                        np.put(evad[i], s, np.nan)
                if j < len(twin) - 1:
                    evtd[i] = np.concatenate(
                        (evtd[i], [np.nan])
                    )  # also force a break between twins.
                    evad[i] = np.concatenate(
                        (evad[i], [np.nan])
                    )  # keep synchronized...
            evtd[i] = np.diff(
                evtd[i]
            )  # now take the differences to get the disbribution
        return evtd, evad

    def shuffle_score(self, evp, twin, nshuffle=10000):
        """
        Shuffle data and compute prob of a spont event occuring in the response window

        Parameters
        ----------
        evp : list of numpy record arrays
            list containing numpy arrays with 'time' and 'amp' for each event in each spot/trial
        twin : list of tuples
            list of 2-tuples, each defining a response window to test
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
        shc = np.ones(len(twin))

        # remove potential responses in data intervals before shuffle to avoid biases
        for i, tx in enumerate(twin):
            si = np.argwhere((evt > np.sum(tx[0:2])) & (evt <= np.sum(tx[0:3])))
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
        for i, tx in enumerate(twin):
            si = np.argwhere((spl > np.sum(tx[0:2])) & (spl <= np.sum(tx[0:3])))
            #        nspk = len([s for s in spl if (s > np.sum(tx[0:2]) and s < np.sum(tx[0:3]))])  # argwhere is much faster
            if any(si):
                nspk = len(si)
                shc[i] += float(nspk)
        shc = shc / (ntraces * len(twin) * nshuffle)
        return shc  # base probability of detecting event in window given input distribution, over all traces

    def shuffle_data1(
        self, evp: dict, twin: Union[list, np.ndarray], maxt=1.0, nshuffle=10000
    ):
        """
        Shuffle event time data in a map, returning counts of events
        across all stimulus windows for each trial
        This version combines all data in all trials across the map
        excludes responses within a response window (replacing with nan),
        then computes the mean interevent interval.
        The mean interval is then used to generate Poisson (exponential)
        event trains which are permuted to estimate the chance of an event
        falling in the response window(s). Summary counting is used to
        estimate the probability that observed events are more frequent than
        expected.

        Parameters
        ----------
        evp : list of numpy record arrays
            list containing numpy arrays with 'time' and 'amp' for each event in each spot/trial
        twin : list of tuples
            list of 2-tuples, each defining a response window to test
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
        evtd, evad = self._get_spont_isis(evp, twin)

        mean_sp_amp = np.nanmean(
            np.hstack(evad).ravel()
        )  # get the mean amplitude of spontaneous events
        evtd_x = np.hstack(
            evtd
        ).ravel()  # flatten the isis across ALL points in the map

        data_count = 0
        for i in range(len(evp)):
            data_count += self._count_events_in_allwindows(
                evp[i]["time"], twin
            )  # count events in data window in real data

        if all(np.isnan(evtd_x)):
            cprint("yellow", f"No spontaneous intervals events detected")
            if (
                data_count > 0
            ):  # no spontaneous data, so this tells us that there were no events to use.
                return (evtd, 0.0, 0.0)  # high prob
            else:
                return (evtd, 1.0, 0.0)  # no prob, no events

        print(evtd_x)
        print(maxt)
        meanisi = np.nanmean(evtd_x)
        mean_rate = 1.0 / meanisi
        print("mean isi: ", meanisi, "   mean rate: ", mean_rate)
        print(
            np.count_nonzero(~np.isnan(evtd_x)) / (maxt * len(evp))
        )  # events over entire time
        exit()
        nisi = len(evtd_x)
        ntraces = len(evp)  # number of traces (or spots in a map)
        shuffle_evts = np.zeros(nshuffle)
        evtd = np.array(evtd).ravel()
        trwins = [[None] for nt in range(ntraces)]  # build trace windows for each trace
        for tr in range(
            ntraces
        ):  # compute times for windows across concatenated trains
            t0 = tr * maxt
            trtwin = twin.copy()
            for nwin in range(len(twin)):
                trtwin[nwin] = t0 + np.array(twin[nwin])
            trwins[tr] = trtwin

        for t in range(nshuffle):
            poiss = np.random.exponential(
                meanisi, size=nisi * 2
            )  # make an artifical distribution that matches
            tss = np.cumsum(poiss)  # generate new trains from all intervals
            for nt in range(ntraces):  # pull data from individual traces
                t0 = (
                    nt * maxt
                )  # start time for each trace (remember, they are concatenated now)
                shuffle_evts[t] += self._count_events_in_allwindows(
                    tss, np.array(twin), tzero=t0
                )  # count events in shuffled windows across all traces/trials
        nexceed = len(np.where(shuffle_evts >= data_count)[0])
        # print(np.where(shuffle_evts > data_count))
        # print('n shuffle > data: ', nexceed)
        shuffle_probs = nexceed / float(nshuffle)
        # print('**** Data count: ', data_count, ' mean events: ', np.mean(shuffle_evts), '   shuf probs: ', shuffle_probs)
        return (
            data_count,
            shuffle_evts,
            shuffle_probs,
            mean_rate,
            mean_sp_amp,
        )  # base probability of detecting event in window given input distribution, over all traces
        # and mean amplitude of spontaneous events not in the response window

    def shuffle_data2(
        self, evp: dict, twin: Union[list, np.ndarray], maxt=1.0, nshuffle=10000
    ):
        """
        Shuffle event time data in a map, returning counts of events
        across all stimulus windows for each trial

        Unclear that this version works well.

        Parameters
        ----------
        evp : list of numpy record arrays
            list containing numpy arrays with 'time' and 'amp' for each event in each spot/trial
        twin : list of tuples
            list of 2-tuples, each defining a response window to test
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
        evtd, evad = self._get_spont_isis(evp, twin)
        evtd_x = np.hstack(
            evtd
        ).ravel()  # flatten the isis across ALL points in the map
        if all(
            np.isnan(evtd_x)
        ):  # no spontaneous data, so this tells us that there were no events to use.
            cprint("yellow", f"no spontaneous intervals events detected")
            return (evtd, 0, 0.0)

        mean_sp_amp = np.nanmean(
            np.hstack(evad).ravel()
        )  # get the mean amplitude of spontaneous events
        # print('mean isi: ', evtd_x)
        # import matplotlib.pyplot as mpl
        # f, ax = mpl.subplots(1,1)
        # ax.hist(evtd_x, bins=np.arange(0, 0.5, 0.01))
        # mpl.show()
        # exit()
        ntraces = len(evp)  # number of traces (or spots in a map)
        shuffle_evts = np.zeros(ntraces)  # track events in windows for each trace
        for n in range(nshuffle):
            for t in range(ntraces):
                ts = evtd[t].copy()  #  ts = evtd[tr_rand[t]].copy()  # selecct data
                if len(ts) == 0:
                    shuffle_evts[t] += np.nan
                else:
                    np.random.shuffle(ts)  # shuffle interval order
                    ts = np.cumsum(ts)  # generate new train
                    shuffle_evts[t] += self._get_events_in_allwindows(ts, twin)
            # normalize on per-trace (spot) basis
        shuffle_probs = np.array(shuffle_evts) / (nshuffle)
        print("shuf probs: ", shuffle_probs)
        return (
            evtd,
            shuffle_probs,
            mean_sp_amp,
        )  # base probability of detecting event in window given input distribution, over all traces
        # and mean amplitude of spontaneous events not in the response window

    def detect_events(
        self,
        event_times: np.ndarray,
        event_amplitudes: np.ndarray,
        stim_times: Union[list, np.ndarray],
        mean_spont_amp: Union[float, None] = None,
    ):
        """
        find events in the stimulus window (just detect and count)
        twin is a list of 2-tuples, each defining a response window to test
        returns fraction of events relative to all traces

        Parameters
        ----------
        event_times : list
            array of events times in traces
        event_amplitudes: np.ndarry:
            array of corresponding event amplitudes

        stim_times : list
            stimulus time window

        Returns
        -------
        shc_n, detected_counts/ntraces, detected_amplitudes
            shc_n : fraction of sites during which event was detected
            counts : estimated frequency of events normalized by traces
            amplitudes : amplitudes of detected events.

        """
        ntraces = len(event_times)
        if ntraces == 0:
            cprint("red", "ntraces is 0, no events were included")
            return 0, 0, None

        # events = np.concatenate(evp)  # combine all traces in map

        event_times_c = np.concatenate(event_times)
        event_amplitudes_c = np.concatenate(event_amplitudes)
        # print(event_times)

        mean_amplitude = np.nanmean(event_amplitudes_c)
        sd_amplitude = np.nanstd(event_amplitudes_c)
        nbig = np.where(event_amplitudes_c > mean_amplitude + 2.0 * sd_amplitude)[0]

        detected_counts = np.zeros(ntraces)  # keep tab of events in each window
        detected_amplitudes = [[]] * ntraces  # np.zeros(len(twin))
        spont_amp = mean_spont_amp

        for i in range(ntraces):
            (
                detected_counts[i],
                detected_amplitudes[i],
            ) = self._count_events_in_allwindows(
                event_times[i], event_amplitudes[i], stim_times
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

    def test_events_shuffle(self):
        """
        Use shuffle methods (resampling) to directly test event probability

        Parameters
        ----------
        None

        """
        import pylibrary.plotting.plothelpers as PH
        import matplotlib.pyplot as mpl

        rc = PH.getLayoutDimensions(self.nmaps + 2, pref="width")
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
        for k in range(self.nrepeats):
            for i in range(self.nmaps):
                evp, nactivesites = self._make_test_events(
                    self.nspots,
                    spontrate=self.srate,
                    prob=self.probs[i],
                    stimtime=self.stimtimes[0][0],
                    maxt=self.maxt,
                    seed=(i + k) * k,
                )
                ms = np.zeros(self.nspots)
                for j in range(self.nspots):
                    if len(evp[j]) > 0:
                        ms[j] = len(evp[j])
                rate = np.sum(ms) / (self.maxt * self.nspots)
                print(
                    f"Repeat: {k+1:d}/{self.nrepeats:d}  map: {i+1:d}/{self.nmaps}   prot: {self.probs[i]:.4f}"
                )
                # print("    stimulus times: ", self.stimtimes)
                (
                    data_count,
                    shuffle_evts,
                    shuffle_probs,
                    mean_rate,
                    mean_ev_amp,
                ) = self.shuffle_data1(
                    evp.copy(), self.stimtimes, nshuffle=self.nshuffle, maxt=self.maxt
                )
                axl[-2].hist(
                    shuffle_evts, bins=binstuff2, histtype="stepfilled", align="right"
                )
                # print(shuffle_evts)
                print("data count, max# sh_evts: ", data_count, np.max(shuffle_evts))
                PShuffle[i, k] = shuffle_probs
                evt = []
                for j, ep in enumerate(evp):
                    evt.extend(np.array(ep["time"]))

                frac_detected, detevt, detamp = self.detect_events(
                    evp, self.stimtimes, mean_spont_amp=1.0
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
            P.figure_handle.suptitle(f"testing shuffler: srate={self.srate:.2f}")
            # PofZ = 1.0-self.z2p(y)

        for i in range(self.nmaps):
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
        PH.talbotTicks(
            axl[-1], tickPlacesAdd={"x": 0, "y": 0}, floatAdd={"x": 2, "y": 2}
        )
        mpl.show()

    def test_events_poisson(self):
        """
        Like test events, except computes Poisson Score from Chase and Young
        Based on code from Luke Campagnola

        Parameters
        ----------
        None
        """
        import matplotlib.pyplot as mpl

        rc = PH.getLayoutDimensions(self.nmaps + 2, pref="width")
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
        y = []
        probs = []
        tpr = []
        tevt = []

        for i in range(self.nmaps):
            evp, nactivesites = self._make_test_events(
                self.nspots,
                spontrate=self.srate,
                prob=self.probs[i],
                stimtime=self.stimtimes[0][0],
                maxt=self.maxt,
                seed=i,
            )
            tMax = self.maxt  # self.stimtimes[0][0] + self.stimtimes[0][2]
            mscore, prob = EPPS.PoissonScore.score(
                evp, rate=self.srate, tMax=tMax, normalize=True
            )

            print(
                f"Testing {i:d} Score= {np.log10(mscore):.4f} mscore: {mscore:.2f} sites: {nactivesites:d}"
            )
            x.append(nactivesites)
            y.append(mscore)
            evt = []
            for j, p in enumerate(evp):
                evt.extend(np.array(p["time"]))
            evt = np.array(evt)
            ievt = np.argsort(evt)
            tpr.append(prob[ievt])
            tevt.append(evt[ievt])

            # for j in range(len(evt)):
            #     if prob[j] < 0.5 and (evt[j] > 0.1 and evt[j] < 0.11):
            #         print('j, evt, p: ', j, evt[j], prob[j])
            n, bins, patches = axl[i].hist(
                evt, bins=binstuff, histtype="stepfilled", align="mid"
            )
            #     for s in evl['stimtimes']['start']:
            # #        print('s: ', s)
            #         axl[i].plot([s, s], [0, 40], 'r-', linewidth=0.5)
            axl[i].set_ylim(0, 100)
            axl[i].set_title(
                f"score={np.log10(mscore):.3f}, sites={nactivesites:d}", fontsize=9
            )

            axl[-2].plot(tevt[-1], tpr[-1], "o-", markersize=1.0)
        P.figure_handle.suptitle(str("testing"))

        axl[-1].plot(x, np.log10(y), "ok", markersize=3)
        axl[-1].set_ylim(0, 1.1 * np.max(np.log10(y)))
        axl[-1].set_ylabel("mscore")
        axl[-1].set_xlim(0, 1.1 * np.max(x))
        axl[-1].set_xlabel("# active sites")
        PH.talbotTicks(
            axl[-1], tickPlacesAdd={"x": 0, "y": 0}, floatAdd={"x": 0, "y": 2}
        )
        mpl.show()

    def set_pars(self):
        self.nrepeats = 1
        self.nmaps = 10
        self.nspots = 100
        self.srate = 20.0
        self.stimtimes = [(0.1, 0.0, 0.01)]
        self.maxt = 0.6
        self.nshuffle = 100
        self.probs = np.linspace(0.0, 0.2, self.nmaps, endpoint=True)


def main():
    import matplotlib.pyplot as mpl

    S = Shuffler()
    S.set_pars()

    S.test_events_poisson()
    # S.test_events_shuffle()


if __name__ == "__main__":
    main()
