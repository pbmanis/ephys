# -*- encoding: utf-8 -*-
"""
Pytest coverage for ephys.mapanalysistools.shuffler.Shuffler.

Shuffler builds synthetic maps of spontaneous + evoked events and scores them
(z-score / Poisson).  These tests target the pieces that run without plotting or
the heavy Poisson-normalization tables:

  - z2p:            z-score -> probability (pure function)
  - compute_ZScore: per-trace response-window z-scores
  - _make_test_events / generate_example_traces: synthetic map generation

The event-generation helpers use an un-seeded ``np.random.default_rng()`` for the
spontaneous process, so only structural invariants (map size, active-site bounds)
are asserted for them.  compute_ZScore uses a fixed internal RandomState, so its
input-driven behaviour is deterministic and is asserted directly.
"""
import numpy as np
import pytest

import ephys.mapanalysistools.shuffler as SH


STIM = [dict(start=0.3, window_start=0.001, window_duration=0.02)]  # window [0.301, 0.321]


@pytest.fixture
def shuffler():
    S = SH.Shuffler()
    S.set_pars(probtype="single")
    S.maxt = 1.0
    S.quanta = 2
    return S


# ---------------------------------------------------------------------------
# z2p
# ---------------------------------------------------------------------------
def test_z2p_zero_is_half():
    assert np.isclose(SH.Shuffler().z2p(0.0), 0.5)


def test_z2p_known_values():
    S = SH.Shuffler()
    assert np.isclose(S.z2p(1.96), 0.975, atol=1e-3)
    assert np.isclose(S.z2p(-1.96), 0.025, atol=1e-3)


def test_z2p_is_symmetric():
    S = SH.Shuffler()
    for z in (0.5, 1.0, 2.5):
        assert np.isclose(S.z2p(z) + S.z2p(-z), 1.0)


# ---------------------------------------------------------------------------
# compute_ZScore
# ---------------------------------------------------------------------------
def _spont_traces(ntr=40, n_per=15, seed=0):
    rng = np.random.default_rng(seed)
    return [np.sort(rng.uniform(0.0, 1.0, size=n_per)) for _ in range(ntr)]


def test_compute_zscore_shape_and_finite(shuffler):
    traces = _spont_traces()
    z = shuffler.compute_ZScore(traces, STIM, plot=False)
    assert z.shape == (len(traces),)
    assert np.all(np.isfinite(z))


def test_compute_zscore_evoked_window_raises_score(shuffler):
    """Traces carrying a dense cluster inside the response window get much
    higher z-scores than spontaneous-only traces measured in the same map."""
    spont = _spont_traces(ntr=40, n_per=15, seed=0)
    evoked = [s.copy() for s in spont]
    odd = np.arange(len(spont)) % 2 == 1
    for i in np.where(odd)[0]:
        evoked[i] = np.sort(np.concatenate([spont[i], np.linspace(0.302, 0.318, 6)]))
    z = shuffler.compute_ZScore(evoked, STIM, plot=False)
    assert np.mean(z[odd]) > np.mean(z[~odd])
    assert np.any(z[odd] > 1.96)          # at least some reach significance


# ---------------------------------------------------------------------------
# _make_test_events / generate_example_traces
# ---------------------------------------------------------------------------
def test_make_test_events_map_size_and_active_bounds(shuffler):
    nspots = 25
    test_map, n_active = shuffler._make_test_events(
        nspots, spontrate=5.0, prob=[0.5], stim_times=STIM, maxt=1.0, seed=3,
    )
    assert test_map.get_n_spots() == nspots
    assert 0 <= n_active <= nspots        # one stim window, at most one insert per spot


def test_make_test_events_zero_spontrate_still_builds_map(shuffler):
    nspots = 10
    test_map, n_active = shuffler._make_test_events(
        nspots, spontrate=0.0, prob=[1.0], stim_times=STIM, maxt=1.0, seed=1,
    )
    assert test_map.get_n_spots() == nspots


def test_generate_example_traces_returns_full_map(shuffler):
    test_map = shuffler.generate_example_traces(rate=5.0, prob=0.3)
    assert test_map.get_n_spots() == shuffler.nspots
