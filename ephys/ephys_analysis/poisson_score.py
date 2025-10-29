"""
Description:

    Experiments that characterize the functional synaptic connectivity between
    two neurons often rely on being able to evoke a spike in the presynaptic
    cell and detect an evoked synaptic response in the postsynaptic cell.
    These synaptic responses can be difficult to distinguish from the constant
    background of spontaneous synaptic activity.

    The method implemented here assumes that spontaneous activity can be
    described by a poisson process. For any given series of synaptic events,
    we calculate the probability that the event times could have been generated
    by a poisson process.

    Here are some scenarios we need to consider:

    (1) Obvious, immediate rate change

    |___||____|_|_______|____|___|___|_|||||||_|_||__|_||___|____|__|_|___|____|___
                                      ^
    (2) Obvious, delayed rate change

    |___||____|_|_______|____|___|___|_|____|__|_|___|___|_||_|_||_|_|__|_|_||_____
                                      ^
    (3) Non-obvious rate change, but responses have good precision

    |______|______|_________|_______|____|______|________|________|_________|______
    _____|___________|_______|___|_______|__________|____|______|______|___________
    ___|________|_________|___|______|___|__|_________|____|_______|___________|___
                                      ^
    (4) Very low spont rate (cannot measure intervals between events)
        with good response precision

    ______________________________________|________________________________________
    ________|___________________________________|__________________________________
    _________________________________________|________________________|____________
                                      ^
    (5) Non-obvious rate change, but response amplitudes are very different

    __,______.___,_______.___,_______,_____|___,_____._________,_,______.______,___
                                      ^
"""

import os
from pathlib import Path
import sys
import copy
from typing import Union
import numpy as np
import scipy
import scipy.stats as stats
import scipy.interpolate

import pyqtgraph as pg
import pyqtgraph.console
import pyqtgraph.multiprocess as mp


def poissonProcess(
    rate: float, tmax: Union[float, None] = None, n: Union[int, None] = None
) -> np.ndarray:
    """Simulate a poisson process; return a list of event times

    Parameters
    ---------_
    rate : float, Hz
        process rate
    tmax : float (default None), seconds
        maximum time for generation, in seconds
    n : int (default: None)
        limiting number of events to terminate generation
    """
    events = []
    t = 0
    while True:
        t += np.random.exponential(1.0 / rate)
        if tmax is not None and t > tmax:
            break
        events.append(t)
        if n is not None and len(events) >= n:
            break
    return np.array(events)


def poissonProb(n: int, t: float, l: float, clip: bool = False) -> float:
    """
    For a poisson process, return the probability of seeing at least *n* events in *t* seconds given
    that the process has a mean rate *l*.
    If the mean rate l == 0, then the probability is 1.0 if n==0 and 1e-25 otherwise.

    Parameters
    ----------
    n : int
        Number of events
    t : float
        Time to test
    l : float
        mean process rate

    Returns
    -------
    p : float

    """
    if l == 0:
        if np.isscalar(n):
            if n == 0:
                print("l=0, scalar n = 0")
                return 1.0
            else:
                print("l=0, scalar n > 0")
                return 1e-25
        else:
            print("not scalar, return array")
            return np.where(n == 0, 1.0, 1e-25)
    p = stats.poisson(l * t).sf(n)
    if clip:
        p = np.clip(p, 0, 1.0 - 1e-25)
    # print("SCORE: ", p)
    # ts = np.where(p < 0.01)[0]
    # print(ts)
    # print(t[ts])  # could use to mark where unexpected events occur
    return p


def gaussProb(amps: Union[list, np.ndarray], mean: float, stdev: float) -> float:
    ## Return the survival function for gaussian distribution
    if len(amps) == 0:
        return 1.0
    return stats.norm(mean, stdev).sf(amps)


class PoissonScore:
    """
    Class for computing a statistic that asks "what is the probability that a poisson process
    would generate a set of events like this"

    General procedure:
      1. For each event n in a list of events, compute the probability of a poisson
         process generating at least n-1 events in the time up to event n (this is
         poissonProb() applied individually to each event)
      2. The maximum value over all events is the score. For multiple trials, simply
         mix together all events and assume an accordingly faster poisson process.
      3. Normalize the score to a probability using a precomputed table generated
         by a poisson process simulations.
    """

    normalizationTable = None

    @classmethod
    def score(
        cls,
        events: Union[list, np.ndarray],
        amplitudes: Union[list, np.ndarray],
        rate: Union[float, np.ndarray] = 2.0,
        tMax: float = None,
        normalize: bool = True,
        **kwds,
    ):
        """
        Compute poisson score for a set of events. This is the main routine to call to make
        the calculation.
        evvents is a numpy array of event times.
        Each array describes a set of events; multiple sets can be provided as a list of arrays.
        *rate* may be either a single value or a list (in which case the mean will be used)
        """
        nSets = len(events)
        events = np.concatenate(events)
        pi0 = 1.0
        if isinstance(rate, np.ndarray):
            rate = np.mean(rate)
        if len(events) == 0:
            score = 1.0
        else:
            # ev = [x['time'] for x in ev]  ## select times from event set
            # ev = np.concatenate(ev)   ## mix events together
            ev = events # events["time"]

            nVals = np.array(
                [(ev <= t).sum() - 1.0 for t in ev]

            )  ## looks like arange, but consider what happens if two events occur at the same time.
            

            pi0 = poissonProb(
                nVals, ev, rate * nSets
            )  ## note that by using n=0 to len(ev)-1, we correct for the fact that the time window always ends at the last event

            
            try:
                pi = 1.0 / pi0
            except:
                print("poisson_score:score: pi0: ", pi0*1e6)
                pi = 1
            ## apply extra score for uncommonly large amplitudes
            ## (note: by default this has no effect; see amplitudeScore)
            ampScore = cls.amplitudeScore(events, **kwds)
            pi *= ampScore

            # print(['%9.2e'%p for p in pi])
            # print(np.max(pi))
            mp = pi.max()
            # mpp = min(cls.maxPoissonProb(ev, rate*nSets), 1.0-1e-12)  ## don't allow returning inf
            # mpp = min(mp, 1.0-1e-12)
            score = mp
            # score =  1.0 / (1.0 - mpp)

        # n = len(ev)
        if normalize and rate > 0:
            print("rate, tmax, nsets: ", rate, tMax, nSets)
            ret = cls.mapScore(x=score, n=rate * tMax * nSets)
        else:
            ret = score
        if np.isscalar(ret):
            assert not np.isnan(ret)
        else:
            assert not any(np.isnan(ret))

        return ret, pi0

    @classmethod
    def amplitudeScore(cls, events, **kwds):
        """Computes extra probability information about events based on their amplitude.
        Inputs to this method are:
            events: record array of events; fields include 'time' and 'amp'

        By default, no extra score is applied for amplitude (but see also PoissonRepeatAmpScore)
        """
        return np.ones(len(events))

    @classmethod
    def mapScore(cls, x: float, n: int, nEvents=10000):
        """
        Map score x to probability given we expect n events per set
        """
        # print('checking normalization table')
        if cls.normalizationTable is None:
            print("generating table")
            cls.normalizationTable = cls.generateNormalizationTable(nEvents=nEvents)
            cls.extrapolateNormTable()
        nind = max(0, np.log(n) / np.log(2))
        n1 = np.clip(int(np.floor(nind)), 0, cls.normalizationTable.shape[1] - 2)
        n2 = n1 + 1

        mapped1 = []
        for i in [n1, n2]:
            norm = cls.normalizationTable[:, i]
            ind = np.argwhere(norm[0] > x)
            if len(ind) == 0:
                ind = len(norm[0]) - 1
            else:
                ind = ind[0, 0]
            if ind == 0:
                ind = 1
            x1, x2 = norm[0, ind - 1 : ind + 1]
            y1, y2 = norm[1, ind - 1 : ind + 1]
            if x1 == x2:
                s = 0.0
            else:
                s = (x - x1) / float(x2 - x1)
            if s == np.inf:
                s = 1e12
            mapped1.append(y1 + s * (y2 - y1))
        mapped1 = sorted(mapped1)
        try:
            mapped = mapped1[0] + (mapped1[1] - mapped1[0]) * (nind - n1) / float(n2 - n1)
        except FloatingPointError:
            print("nind, n1, n2: ", nind, n1, n2)
            print("x1 x2, y1 y2: ", x1, x2, y1, y2)
            print("s: ", s)
            print("mapped1: ", mapped1)
            return []

        ## doesn't handle points outside of the original data.
        # mapped = scipy.interpolate.griddata(poissonScoreNorm[0], poissonScoreNorm[1], [x], method='cubic')[0]
        # normTable, tVals, xVals = poissonScoreNorm
        # spline = scipy.interpolate.RectBivariateSpline(tVals, xVals, normTable)
        # mapped = spline.ev(n, x)[0]
        # raise Exception()
        assert not (np.isinf(mapped) or np.isnan(mapped))
        # print("mapped: ", mapped, mapped1)
        assert mapped > 0
        return mapped

    @classmethod
    def generateRandom(cls, rate, tMax, reps=3):
        if np.isscalar(rate):
            rate = [rate] * reps
        ret = []
        for i in range(reps):
            times = poissonProcess(rate[i], tMax)
            ev = np.empty(len(times), dtype=[("time", float), ("amp", float)])
            ev["time"] = times
            ev["amp"] = np.random.normal(size=len(times))
            ret.append(ev)
        return ret

    @classmethod
    def generateNormalizationTable(cls, nEvents=1000000):
        ## table looks like this:
        ##   (2 x M x N)
        ##   Axis 0:  (score, mapped)
        ##   Axis 1:  expected number of events  [1, 2, 4, 8, ...]
        ##   Axis 2:  score axis

        ## To map:
        ##    determine axis-1 index by expected number of events
        ##    look up axis-2 index from table[0, ind1]
        ##    look up mapped score at table[1, ind1, ind2]

        ## parameters determining sample space for normalization table
        rate = 1.0
        tVals = 2 ** np.arange(9)  ## set of tMax values
        nev = (nEvents / (rate * tVals) ** 0.5).astype(
            int
        )  # number of events to generate for each tMax value

        xSteps = 1000
        r = 10 ** (30.0 / xSteps)
        xVals = r ** np.arange(xSteps)  ## log spacing from 1 to 10**20 in 500 steps
        tableShape = (2, len(tVals), len(xVals))

        path = os.path.dirname(__file__)
        cacheFile = os.path.join(
            path,
            "test_data/%s_normTable_%s_float64.dat"
            % (cls.__name__, "x".join(map(str, tableShape))),
        )

        if os.path.exists(cacheFile):
            # norm = np.fromstring(
            norm = copy.copy(
                np.frombuffer(open(cacheFile, "rb").read(), dtype=float).reshape(tableShape)
            )
        else:
            print(
                "Generating poisson score normalization table (will be cached here: %s)" % cacheFile
            )
            cf = Path(cacheFile)
            if not cf.parent.is_dir():
                print(f"cannot find: {str(cf.parent):s}")
                cf.parent.mkdir()
                print("Created directory:", str(cf.parent))
            else:
                print("path found: ", str(cf.parent))

            norm = np.empty(tableShape)
            counts = []
            with mp.Parallelize(counts=counts) as tasker:
                for task in tasker:
                    count = np.zeros(tableShape[1:], dtype=float)
                    for i, t in enumerate(tVals):
                        n = nev[i] / tasker.numWorkers()
                        for j in range(int(n)):
                            if j % 10000 == 0:
                                print("%d/%d  %d/%d" % (i, len(tVals), j, int(n)))
                                tasker.process()
                            ev = cls.generateRandom(rate=rate, tMax=t, reps=1)

                            score = cls.score(ev, rate, normalize=False)
                            ind = np.log(score[0]) / np.log(r)
                            count[i, : int(ind) + 1] += 1
                    tasker.counts.append(count)

            count = sum(counts)
            count[count == 0] = 1
            norm[0] = xVals.reshape(1, len(xVals))
            norm[1] = nev.reshape(len(nev), 1) / count

            open(cacheFile, "wb").write(norm.tostring())

        return norm

    @classmethod
    def testMapping(cls, rate=1.0, tMax=1.0, n=10000, reps=3):
        scores = np.empty(n)
        mapped = np.empty(n)
        ev = []
        for i in range(len(scores)):
            ev.append(cls.generateRandom(rate, tMax, reps))
            scores[i] = cls.score(ev[-1], rate, tMax=tMax, normalize=False)
            mapped[i] = cls.mapScore(scores[i], np.mean(rate) * tMax * reps, nEvents=10000)

        for j in [1, 2, 3, 4]:
            print("  %d: %f" % (10**j, (mapped > 10**j).sum() / float(n)))
        return ev, scores, mapped

    @classmethod
    def showMap(cls):
        plt = pg.plot()
        for i in range(cls.normalizationTable.shape[1]):
            plt.plot(
                cls.normalizationTable[0, i],
                cls.normalizationTable[1, i],
                pen=(i, 14),
                symbolPen=(i, 14),
                symbol="o",
            )

    @classmethod
    def poissonScoreBlame(cls, ev, rate):
        events = np.concatenate(ev)
        ev = events["time"]
        nVals = np.array([(ev <= t).sum() - 1 for t in ev])
        x = poissonProb(nVals, ev, rate, clip=True)
        # print(x)
        pp1 = 1.0 / (1.0 - poissonProb(nVals, ev, rate, clip=True))
        pp2 = 1.0 / (1.0 - poissonProb(nVals - 1, ev, rate, clip=True))
        diff = pp1 / pp2
        blame = np.array([diff[np.argwhere(ev >= ev[i])].max() for i in range(len(ev))])
        return blame

    @classmethod
    def extrapolateNormTable(cls):
        ## It appears that, on a log-log scale, the normalization curves appear to become linear after reaching
        ## about 50 on the y-axis.
        ## we can use this to overwrite all the junk at the end caused by running too few test iterations.
        d = cls.normalizationTable
        for n in range(d.shape[1]):
            trace = d[:, n]
            logtrace = np.log(trace)
            ind1 = np.argwhere(trace[1] > 60)[0, 0]
            ind2 = np.argwhere(trace[1] > 100)[0, 0]
            dd = logtrace[:, ind2] - logtrace[:, ind1]
            slope = dd[1] / dd[0]
            npts = trace.shape[1] - ind2
            yoff = logtrace[1, ind2] - logtrace[0, ind2] * slope
            trace[1, ind2:] = np.exp(logtrace[0, ind2:] * slope + yoff)


class PoissonAmpScore(PoissonScore):

    normalizationTable = None

    @classmethod
    def amplitudeScore(cls, events, ampMean=1.0, ampStdev=1.0, **kwds):
        """Computes extra probability information about events based on their amplitude.
        Inputs to this method are:
            events: record array of events; fields include 'time' and 'amp'
            times:  the time points at which to compute probability values
                    (the output must have the same length)
            ampMean, ampStdev: population statistics of spontaneous events
        """
        if ampStdev == 0.0:  ## no stdev information; cannot determine probability.
            return np.ones(len(events))
        scores = 1.0 / np.clip(gaussProb(events["amp"], ampMean, ampStdev), 1e-100, np.inf)
        assert not np.any(np.isnan(scores) | np.isinf(scores))
        return scores


class PoissonRepeatScore:
    """
    Class for analyzing poisson-process spike trains with evoked events mixed in.
    This computes a statistic that asks "assuming spikes have poisson timing and
    normally-distributed amplitudes, what is the probability of seeing this set
    of times/amplitudes?".

    A single set of events is merely a list of time values; we can also ask a
    similar question for multiple trials: "what is the probability that a poisson
    process would produce all of these spike trains"
    The statistic should be able to pick out:
      - Spikes that are very close to the stimulus (assumed to be at t=0)
      - Abnormally high spike rates, particularly soon after the stimulus
      - Spikes that occur with similar post-stimulus latency over multiple trials
      - Spikes that are larger than average, particularly soon after the stimulus

    """

    normalizationTable = None

    @classmethod
    def score(cls, ev, rate, tMax=None, normalize=True, **kwds):
        """
        Given a set of event lists and a background (spontaneous) rate,
        return probability that a poisson process would generate all sets of events.
        ev = [
        [t1, t2, t3, ...],    ## trial 1
        [t1, t2, t3, ...],    ## trial 2
        ...
        ]

        *rate* must have the same length as *ev*.
        Extra keyword arguments are passed to amplitudeScore
        """
        events = ev
        nSets = len(ev)
        ev = [x["time"] for x in ev]  ## select times from event set

        if np.isscalar(rate):
            rate = [rate] * nSets
        if len(rate) != len(events):
            raise ValueError("poisson_score:score:: rate must have same length as ev")
        ev2 = []
        for i in range(len(ev)):
            arr = np.zeros(len(ev[i]), dtype=[("trial", int), ("time", float)])
            arr["time"] = ev[i]
            arr["trial"] = i
            ev2.append(arr)
        ev2 = np.sort(np.concatenate(ev2), order=["time", "trial"])
        if len(ev2) == 0:
            return 1.0

        ev = list(map(np.sort, ev))
        pp = np.empty((len(ev), len(ev2)))
        for i, trial in enumerate(ev):
            nVals = []
            for j in range(len(ev2)):
                n = (trial < ev2[j]["time"]).sum()
                if (
                    any(trial == ev2[j]["time"]) and ev2[j]["trial"] > i
                ):  ## need to correct for the case where two events in separate trials happen to have exactly the same time.
                    n += 1
                nVals.append(n)

            pp[i] = 1.0 / (1.0 - poissonProb(np.array(nVals), ev2["time"], rate[i]))

            ## apply extra score for uncommonly large amplitudes
            ## (note: by default this has no effect; see amplitudeScore)
            pp[i] *= cls.amplitudeScore(events[i], ev2["time"], **kwds)

        score = pp.prod(
            axis=0
        ).max()  ##** (1.0 / len(ev))  ## normalize by number of trials [disabled--we WANT to see the significance that comes from multiple trials.]
        if normalize:
            ret = cls.mapScore(score, np.mean(rate) * tMax, nSets)
        else:
            ret = score
        if np.isscalar(ret):
            assert not np.isnan(ret)
        else:
            assert not any(np.isnan(ret))
        print("ret: ", ret)
        return ret

    @classmethod
    def amplitudeScore(cls, events, times, **kwds):
        """Computes extra probability information about events based on their amplitude.
        Inputs to this method are:
            events: record array of events; fields include 'time' and 'amp'
            times:  the time points at which to compute probability values
                    (the output must have the same length)

        By default, no extra score is applied for amplitude (but see also PoissonRepeatAmpScore)
        """
        return np.ones(len(times))

    @classmethod
    def mapScore(cls, x, n, m):
        """
        Map score x to probability given we expect n events per set and m repeat sets
        """
        if cls.normalizationTable is None:
            cls.normalizationTable = cls.generateNormalizationTable()
            cls.extrapolateNormTable()

        table = cls.normalizationTable[
            :, min(m - 1, cls.normalizationTable.shape[1] - 1)
        ]  # select the table for this repeat number

        nind = np.log(n) / np.log(2)
        n1 = np.clip(int(np.floor(nind)), 0, table.shape[2] - 2)
        n2 = n1 + 1

        mapped1 = []
        for i in [n1, n2]:
            norm = table[:, i]
            ind = np.argwhere(norm[0] > x)
            if len(ind) == 0:
                ind = len(norm[0]) - 1
            else:
                ind = ind[0, 0]
            if ind == 0:
                ind = 1
            x1, x2 = norm[0, ind - 1 : ind + 1]
            y1, y2 = norm[1, ind - 1 : ind + 1]
            if x1 == x2:
                s = 0.0
            else:
                s = (x - x1) / float(x2 - x1)
            mapped1.append(y1 + s * (y2 - y1))

        mapped = mapped1[0] + (mapped1[1] - mapped1[0]) * (nind - n1) / float(n2 - n1)

        ## doesn't handle points outside of the original data.
        # mapped = scipy.interpolate.griddata(poissonScoreNorm[0], poissonScoreNorm[1], [x], method='cubic')[0]
        # normTable, tVals, xVals = poissonScoreNorm
        # spline = scipy.interpolate.RectBivariateSpline(tVals, xVals, normTable)
        # mapped = spline.ev(n, x)[0]
        # raise Exception()
        assert not (np.isinf(mapped) or np.isnan(mapped))
        return mapped

    @classmethod
    def generateRandom(cls, rate, tMax, reps):
        ret = []
        for i in range(reps):
            times = poissonProcess(rate, tMax)
            ev = np.empty(len(times), dtype=[("time", float), ("amp", float)])
            ev["time"] = times
            ev["amp"] = np.random.normal(size=len(times))
            ret.append(ev)
        return ret

    @classmethod
    def generateNormalizationTable(cls, nEvents=10000):

        ## parameters determining sample space for normalization table
        reps = np.arange(1, 5)  ## number of repeats
        rate = 1.0
        tVals = 2 ** np.arange(4)  ## set of tMax values
        nev = (nEvents / (rate * tVals) ** 0.5).astype(int)

        xSteps = 1000
        r = 10 ** (30.0 / xSteps)
        xVals = r ** np.arange(xSteps)  ## log spacing from 1 to 10**20 in 500 steps
        tableShape = (2, len(reps), len(tVals), len(xVals))

        path = os.path.dirname(__file__)
        cacheFile = os.path.join(
            path,
            "%s_normTable_%s_float64.dat" % (cls.__name__, "x".join(map(str, tableShape))),
        )

        if os.path.exists(cacheFile):
            norm = np.fromstring(open(cacheFile, "rb").read(), dtype=float).reshape(tableShape)
        else:
            print("Generating %s ..." % cacheFile)
            norm = np.empty(tableShape)
            counts = []
            with mp.Parallelize(tasks=[0, 1], counts=counts) as tasker:
                for task in tasker:
                    count = np.zeros(tableShape[1:], dtype=float)
                    for i, t in enumerate(tVals):
                        n = nev[i]
                        for j in range(int(n)):
                            if j % 1000 == 0:
                                print("%d/%d  %d/%d" % (i, len(tVals), j, int(n)))
                            ev = cls.generateRandom(rate=rate, tMax=t, reps=reps[-1])
                            for m in reps:
                                score = cls.score(ev[:m], rate, normalize=False)
                                ind = int(np.log(score) / np.log(r))
                                count[m - 1, i, : ind + 1] += 1
                    tasker.counts.append(count)

            count = sum(counts)
            count[count == 0] = 1
            norm[0] = xVals.reshape(1, 1, len(xVals))
            norm[1] = nev.reshape(1, len(nev), 1) / count

            open(cacheFile, "wb").write(norm.tostring())

        return norm

    @classmethod
    def extrapolateNormTable(cls):
        ## It appears that, on a log-log scale, the normalization curves appear to become linear after reaching
        ## about 50 on the y-axis.
        ## we can use this to overwrite all the junk at the end caused by running too few test iterations.
        d = cls.normalizationTable
        for rep in range(d.shape[1]):
            for n in range(d.shape[2]):
                trace = d[:, rep, n]
                logtrace = np.log(trace)
                ind1 = np.argwhere(trace[1] > 60)[0, 0]
                ind2 = np.argwhere(trace[1] > 100)[0, 0]
                dd = logtrace[:, ind2] - logtrace[:, ind1]
                slope = dd[1] / dd[0]
                npts = trace.shape[1] - ind2
                yoff = logtrace[1, ind2] - logtrace[0, ind2] * slope
                trace[1, ind2:] = np.exp(logtrace[0, ind2:] * slope + yoff)

    # @classmethod
    # def testMapping(cls, rate=1.0, tmax=1.0, n=10000):
    # scores = np.empty(n)
    # mapped = np.empty(n)
    # ev = []
    # for i in range(len(scores)):
    # ev.append([{'time': poissonProcess(rate, tmax)}])
    # scores[i] = cls.score(ev[-1], rate, tMax=tmax)

    # for j in [1,2,3,4]:
    # print "  %d: %f" % (10**j, (scores>10**j).sum() / float(len(scores)))
    # return ev, scores

    @classmethod
    def testMapping(cls, rate=1.0, tMax=1.0, n=10000, reps=3):
        scores = np.empty(n)
        mapped = np.empty(n)
        ev = []
        for i in range(len(scores)):
            ev.append(cls.generateRandom(rate, tMax, reps))
            scores[i] = cls.score(ev[-1], rate, tMax=tMax, normalize=False)
            mapped[i] = cls.mapScore(scores[i], rate * tMax * reps)

        for j in [1, 2, 3, 4]:
            print("  %d: %f" % (10**j, (mapped > 10**j).sum() / float(n)))
        return ev, scores, mapped

    @classmethod
    def showMap(cls):
        plt = pg.plot()
        for n in range(cls.normalizationTable.shape[1]):
            for i in range(cls.normalizationTable.shape[2]):
                plt.plot(
                    cls.normalizationTable[0, n, i],
                    cls.normalizationTable[1, n, i],
                    pen=(n, 14),
                    symbolPen=(i, 14),
                    symbol="o",
                )


class PoissonRepeatAmpScore(PoissonRepeatScore):

    normalizationTable = None

    @classmethod
    def amplitudeScore(cls, events, times, ampMean=1.0, ampStdev=1.0, **kwds):
        """Computes extra probability information about events based on their amplitude.
        Inputs to this method are:
            events: record array of events; fields include 'time' and 'amp'
            times:  the time points at which to compute probability values
                    (the output must have the same length)
            ampMean, ampStdev: population statistics of spontaneous events
        """
        return [gaussProb(events["amp"][events["time"] <= t], ampMean, ampStdev) for t in times]


if __name__ == "__main__":
    import pyqtgraph as pg
    import pyqtgraph.console

    app = pg.mkQApp()
    con = pg.console.ConsoleWidget()
    con.show()
    con.catchAllExceptions()

    ## Create a set of test cases:

    reps = 3
    trials = 30
    spontRate = [2.0, 3.0, 5.0]
    miniAmp = 1.0
    tMax = 0.5

    def randAmp(n=1, quanta=1):
        return np.random.gamma(4.0, size=n) * miniAmp * quanta / 4.0

    ## create a standard set of spontaneous events
    spont = []  ## trial, rep
    allAmps = []
    for i in range(trials):
        spont.append([])
        for j in range(reps):
            times = poissonProcess(spontRate[j], tMax)
            amps = randAmp(
                len(times)
            )  ## using scale=4 gives a nice not-quite-gaussian distribution
            source = ["spont"] * len(times)
            spont[i].append((times, amps, source))
            allAmps.append(amps)

    miniStdev = np.concatenate(allAmps).std()

    def spontCopy(i, j, extra):
        times, amps, source = spont[i][j]
        ev = np.zeros(
            len(times) + extra,
            dtype=[("time", float), ("amp", float), ("source", object)],
        )
        ev["time"][: len(times)] = times
        ev["amp"][: len(times)] = amps
        ev["source"][: len(times)] = source
        return ev

    ## copy spont. events and add on evoked events
    testNames = []
    tests = [[[] for i in range(trials)] for k in range(7)]  # test, trial, rep
    for i in range(trials):
        for j in range(reps):
            ## Test 0: no evoked events
            testNames.append("No evoked")
            tests[0][i].append(spontCopy(i, j, 0))

            ## Test 1: 1 extra event, single quantum, short latency
            testNames.append("1ev, fast")
            ev = spontCopy(i, j, 1)
            ev[-1] = (np.random.gamma(1.0) * 0.01, 1, "evoked")
            tests[1][i].append(ev)

            ## Test 2: 2 extra events, single quantum, short latency
            testNames.append("2ev, fast")
            ev = spontCopy(i, j, 2)
            for k, t in enumerate(np.random.gamma(1.0, size=2) * 0.01):
                ev[-(k + 1)] = (t, 1, "evoked")
            tests[2][i].append(ev)

            ## Test 3: 3 extra events, single quantum, long latency
            testNames.append("3ev, slow")
            ev = spontCopy(i, j, 3)
            for k, t in enumerate(np.random.gamma(1.0, size=3) * 0.07):
                ev[-(k + 1)] = (t, 1, "evoked")
            tests[3][i].append(ev)

            ## Test 4: 1 extra event, 2 quanta, short latency
            testNames.append("1ev, 2x, fast")
            ev = spontCopy(i, j, 1)
            ev[-1] = (np.random.gamma(1.0) * 0.01, 2, "evoked")
            tests[4][i].append(ev)

            ## Test 5: 1 extra event, 3 quanta, long latency
            testNames.append("1ev, 3x, slow")
            ev = spontCopy(i, j, 1)
            ev[-1] = (np.random.gamma(1.0) * 0.05, 3, "evoked")
            tests[5][i].append(ev)

            ## Test 6: 1 extra events specific time (tests handling of simultaneous events)
            # testNames.append('3ev simultaneous')
            # ev = spontCopy(i, j, 1)
            # ev[-1] = (0.01, 1, 'evoked')
            # tests[6][i].append(ev)

            ## 2 events, 1 failure
            testNames.append("0ev; 1ev; 2ev")
            ev = spontCopy(i, j, j)
            if j > 0:
                for k, t in enumerate(np.random.gamma(1.0, size=j) * 0.01):
                    ev[-(k + 1)] = (t, 1, "evoked")
            tests[6][i].append(ev)

    # raise Exception()

    ## Analyze and plot all:

    def checkScores(scores):
        """
        I try to understand how this works.
        best is the 'threshold' when this is called,
        bestn is the 'error' when this is called.

        So...
        """
        best = None
        bestn = None
        bestval = None
        for i in [0, 1]:  # there are 2 sets of scores that are compared
            # 0 has the spont + events; 1 has just spont
            for j in range(scores.shape[1]):  # for each trial for this score type
                x = scores[i, j]
                fn = (scores[0] < x).sum()  # how many sponts are less than spont
                fp = (scores[1] >= x).sum()  # how many evokeds are greater than
                diff = abs(fp - fn)  # find the largest difference
                if bestval is None or diff < bestval:  # find the smallest difference over trials
                    bestval = diff  # save the smallest difference
                    best = x  # save the score for this difference
                    bestn = (fp + fn) / 2.0  # ?
        return best, bestn

    algorithms = [
        ("Poisson Score", PoissonScore.score),
        ("Poisson Score + Amp", PoissonAmpScore.score),
        # ('Poisson Multi', PoissonRepeatScore.score),
        # ('Poisson Multi + Amp', PoissonRepeatAmpScore.score),
    ]
    app = pg.mkQApp()

    view = pg.GraphicsView() #border=0.3)
    l = pg.GraphicsLayout(border=0.3)
    view.setCentralItem(l)
    view.show()
    view.setWindowTitle("Poisson Score Tests")
    win = l
    tMax = 0.5
    with pg.ProgressDialog("processing..", maximum=len(tests)) as dlg:
        for i, _ in enumerate(tests):
            first = i == 0
            last = i == len(tests) - 1

            if first:
                evLabel = win.addLabel("Event amplitude", angle=-90, rowspan=len(tests))
            evPlt = win.addPlot()

            plots = []
            scorePlots = []
            repScorePlots = []
            for title, fn in algorithms:
                if first:
                    label = win.addLabel(title, angle=-90, rowspan=len(tests))
                plt = win.addPlot()
                plots.append(plt)
                if first:
                    plt.register(title)
                else:
                    plt.setXLink(title)
                plt.setLogMode(False, True)
                plt.hideAxis("bottom")
                if last:
                    plt.showAxis("bottom")
                    plt.setLabel("bottom", "Trial")
            plt = win.addPlot()
            scorePlots.append(plt)
            # plt = win.addPlot()
            # repScorePlots.append(plt)

            if first:
                evPlt.register("EventPlot1")
            else:
                evPlt.setXLink("EventPlot1")

            evPlt.hideAxis("bottom")
            evPlt.setLabel("left", testNames[i])
            if last:
                evPlt.showAxis("bottom")
                evPlt.setLabel("bottom", "Event time", "s")

            trials = tests[i]
            scores = np.empty((len(algorithms), 2, len(trials)))
            repScores = np.empty((2, len(trials)))
            for j in range(len(trials)):

                ## combine all trials together for poissonScore tests
                ev = tests[i][j]
                spont = tests[0][j]
                evTimes = [x["time"] for x in ev]
                spontTimes = [x["time"] for x in spont]

                allEv = np.concatenate(ev)
                allSpont = np.concatenate(spont)

                colors = [
                    (
                        pg.mkBrush(0, 255, 0, 50)
                        if source == "spont"
                        else pg.mkBrush(255, 255, 255, 150)
                    )
                    for source in allEv["source"]
                ]
                evPlt.plot(
                    x=allEv["time"],
                    y=allEv["amp"],
                    pen=None,
                    symbolBrush=colors,
                    symbol="d",
                    symbolSize=6,
                    symbolPen=None,
                )

                for k, opts in enumerate(algorithms):
                    title, fn = opts

                    score1 = fn(ev["time"], amplitudes=None, rate=np.mean(spontRate), tMax=tMax, ampMean=miniAmp, ampStdev=miniStdev)
                    score2 = fn(spont["time"], amplitudes=None, rate=np.mean(spontRate), tMax=tMax, ampMean=miniAmp, ampStdev=miniStdev)

                    scores[k, :, j] = score1[0], score2[0]
                    plots[k].plot(
                        x=[j],
                        y=[score1[0]],
                        pen=None,
                        symbolPen=None,
                        symbol="o",
                        symbolBrush=(255, 255, 255, 50),
                    )
                    plots[k].plot(
                        x=[j],
                        y=[score2[0]],
                        pen=None,
                        symbolPen=None,
                        symbol="o",
                        symbolBrush=(0, 255, 0, 50),
                    )

            ## Report on ability of each algorithm to separate spontaneous from evoked
            for k, opts in enumerate(algorithms):
                thresh, errors = checkScores(scores[k])
                plots[k].setTitle("%0.2g, %d" % (thresh, errors))

            # Plot score histograms
            bins = np.linspace(-1, 6, 50)
            h1 = np.histogram(np.log10(scores[0, :]), bins=bins)
            h2 = np.histogram(np.log10(scores[1, :]), bins=bins)
            # scorePlt.plot(x=0.5*(h1[1][1:]+h1[1][:-1]), y=h1[0], pen='w')
            # scorePlt.plot(x=0.5*(h2[1][1:]+h2[1][:-1]), y=h2[0], pen='g')

            # bins = np.linspace(-1, 14, 50)
            # h1 = np.histogram(np.log10(repScores[0, :]), bins=bins)
            # h2 = np.histogram(np.log10(repScores[1, :]), bins=bins)
            # repScorePlt.plot(x=0.5*(h1[1][1:]+h1[1][:-1]), y=h1[0], pen='w')
            # repScorePlt.plot(x=0.5*(h2[1][1:]+h2[1][:-1]), y=h2[0], pen='g')

            dlg += 1
            if dlg.wasCanceled():
                break

            win.nextRow()
    if sys.flags.interactive == 0:
        app.exec()
