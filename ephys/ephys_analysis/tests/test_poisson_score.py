# -*- encoding: utf-8 -*-
"""
Pytest coverage for ephys.ephys_analysis.poisson_score.

Covers the pure statistical helpers (poissonProb, poissonProcess, gaussProb)
and the core behaviour of PoissonScore / PoissonRepeatScore.score with
``normalize=False``.  The normalized path is intentionally *not* exercised
because it generates (and caches to disk) a large Monte-Carlo normalization
table via multiprocessing, which is unsuitable for a unit test.

Note on poissonProb semantics: it returns the Poisson survival function,
``stats.poisson(l*t).sf(n)`` = P(X > n), with special-cased behaviour for a
zero mean rate.
"""
import numpy as np
import scipy.stats as stats

import ephys.ephys_analysis.poisson_score as PS


# ---------------------------------------------------------------------------
# poissonProb
# ---------------------------------------------------------------------------
def test_poissonprob_zero_rate_zero_events_is_certain():
    assert PS.poissonProb(n=0, t=1.0, l=0.0) == 1.0


def test_poissonprob_zero_rate_nonzero_events_is_tiny():
    assert PS.poissonProb(n=1, t=1.0, l=0.0) == 1e-25


def test_poissonprob_analytic_value():
    # sf(0) for mean 1 == 1 - P(X=0) == 1 - e^-1
    assert np.isclose(PS.poissonProb(n=0, t=1.0, l=1.0), 1.0 - np.exp(-1.0))


def test_poissonprob_matches_scipy_survival_function():
    for n, t, l in [(2, 1.0, 3.0), (5, 2.0, 1.5), (0, 0.5, 4.0)]:
        assert np.isclose(PS.poissonProb(n=n, t=t, l=l), stats.poisson(l * t).sf(n))


def test_poissonprob_is_a_probability():
    p = PS.poissonProb(n=3, t=1.0, l=2.0)
    assert 0.0 <= p <= 1.0


# ---------------------------------------------------------------------------
# poissonProcess
# ---------------------------------------------------------------------------
def test_poissonprocess_length_equals_n():
    ev = PS.poissonProcess(rate=50.0, n=7)
    assert isinstance(ev, np.ndarray)
    assert len(ev) == 7


def test_poissonprocess_times_are_increasing():
    ev = PS.poissonProcess(rate=50.0, n=20)
    assert np.all(np.diff(ev) > 0)


def test_poissonprocess_respects_tmax():
    ev = PS.poissonProcess(rate=100.0, tmax=1.0)
    assert ev.size > 0
    assert ev.max() <= 1.0


# ---------------------------------------------------------------------------
# gaussProb
# ---------------------------------------------------------------------------
def test_gaussprob_at_mean_is_half():
    p = PS.gaussProb(np.array([5.0]), mean=5.0, stdev=2.0)
    assert np.isclose(float(np.asarray(p)[0]), 0.5)


def test_gaussprob_empty_returns_one():
    assert PS.gaussProb(np.array([]), mean=5.0, stdev=2.0) == 1.0


def test_gaussprob_monotonic_in_amplitude():
    p = np.asarray(PS.gaussProb(np.array([3.0, 5.0, 7.0]), mean=5.0, stdev=2.0))
    # survival function decreases as amplitude increases
    assert p[0] > p[1] > p[2]


# ---------------------------------------------------------------------------
# PoissonScore.score (normalize=False)
# ---------------------------------------------------------------------------
_EVOKED = [np.array([0.01, 0.012, 0.014, 0.016])]
_AMPS = [np.array([1.0, 1.0, 1.0, 1.0])]


def test_score_empty_events_is_unity():
    score, pi0 = PS.PoissonScore.score([np.array([])], [np.array([])],
                                       rate=2.0, tMax=0.5, normalize=False)
    assert score == 1.0
    assert pi0 == 1.0


def test_score_higher_background_rate_lowers_score():
    """The same tightly clustered events are less surprising against a faster
    background process, so the raw score must be lower."""
    s_lo, _ = PS.PoissonScore.score(_EVOKED, _AMPS, rate=2.0, tMax=0.5, normalize=False)
    s_hi, _ = PS.PoissonScore.score(_EVOKED, _AMPS, rate=50.0, tMax=0.5, normalize=False)
    assert s_hi < s_lo


def test_score_is_deterministic():
    a = PS.PoissonScore.score(_EVOKED, _AMPS, rate=2.0, tMax=0.5, normalize=False)[0]
    b = PS.PoissonScore.score(_EVOKED, _AMPS, rate=2.0, tMax=0.5, normalize=False)[0]
    assert a == b


def test_score_evoked_exceeds_unity():
    # a tight cluster of events is more surprising than pure background -> >1
    s, _ = PS.PoissonScore.score(_EVOKED, _AMPS, rate=2.0, tMax=0.5, normalize=False)
    assert s > 1.0


# ---------------------------------------------------------------------------
# PoissonRepeatScore.score (normalize=False)
# ---------------------------------------------------------------------------
def _repeat_trial():
    return np.array([(0.01,), (0.02,), (0.03,)], dtype=[("time", float)])


def test_repeat_score_finite_and_positive():
    tr = _repeat_trial()
    score = PS.PoissonRepeatScore.score([tr, tr], rate=5.0, tMax=0.5, normalize=False)
    assert np.isfinite(score)
    assert score > 0.0


def test_repeat_score_deterministic_for_identical_trials():
    tr = _repeat_trial()
    a = PS.PoissonRepeatScore.score([tr, tr], rate=5.0, tMax=0.5, normalize=False)
    b = PS.PoissonRepeatScore.score([tr, tr], rate=5.0, tMax=0.5, normalize=False)
    assert a == b


def test_repeat_score_empty_is_unity():
    empty = np.array([], dtype=[("time", float)])
    score = PS.PoissonRepeatScore.score([empty, empty], rate=5.0, tMax=0.5, normalize=False)
    assert score == 1.0
