# -*- encoding: utf-8 -*-
"""
Pytest coverage for ephys.ephys_analysis.adaptation_indices.

These are pure functions operating on spike-time / ISI arrays, so no live data
is required.  Three canonical synthetic trains are used throughout:

  regular  - evenly spaced spike times          -> ~constant rate, index ~ 0
  decel    - growing ISIs (spikes early)        -> adapting / decelerating
  accel    - shrinking ISIs (spikes late)       -> accelerating

Sign convention (verified against the implementation, and NOT the guess in
TODO_missing_tests.md, which had it backwards): for both MKX2019 and AllenInst,
an *adapting / decelerating* train yields a POSITIVE index and an accelerating
train a NEGATIVE index.  See adaptation_index_MKX2019's own docstring: spikes
concentrated in the first half -> +1.
"""
import numpy as np
import pytest

import ephys.ephys_analysis.adaptation_indices as AI


# ---------------------------------------------------------------------------
# Canonical synthetic spike-time trains
# ---------------------------------------------------------------------------
def _regular():
    # evenly spaced spike times -> constant instantaneous rate
    return np.arange(1, 21) * 0.05


def _decel():
    # ISIs grow (0.01, 0.02, 0.03, ...): spikes bunched early -> adapting
    return np.cumsum(np.arange(1, 21) * 0.01)


def _accel():
    # ISIs shrink (1, 1/2, 1/3, ...): spikes bunched late -> accelerating
    return np.cumsum(1.0 / np.arange(1, 21))


def _dur(train):
    return float(train[-1]) + 0.05


# ---------------------------------------------------------------------------
# adaptation_index_MKX2019
# ---------------------------------------------------------------------------
def test_mkx2019_regular_near_zero():
    r = _regular()
    ai = AI.adaptation_index_MKX2019(r, trace_duration=_dur(r))
    assert np.isclose(ai, 0.0, atol=0.05)


def test_mkx2019_decelerating_positive():
    d = _decel()
    ai = AI.adaptation_index_MKX2019(d, trace_duration=_dur(d))
    assert ai > 0.0


def test_mkx2019_accelerating_negative():
    a = _accel()
    ai = AI.adaptation_index_MKX2019(a, trace_duration=_dur(a))
    assert ai < 0.0


def test_mkx2019_too_few_spikes_returns_nan():
    assert np.isnan(AI.adaptation_index_MKX2019(np.array([0.1, 0.2]),
                                                trace_duration=1.0, minimum_spikes=4))


# ---------------------------------------------------------------------------
# adaptation_index_AllenInst
# ---------------------------------------------------------------------------
def test_alleninst_regular_near_zero():
    r = _regular()
    ai = AI.adaptation_index_AllenInst(r, trace_duration=_dur(r))
    assert np.isclose(ai, 0.0, atol=1e-6)


def test_alleninst_decelerating_positive():
    d = _decel()
    ai = AI.adaptation_index_AllenInst(d, trace_duration=_dur(d))
    assert ai > 0.0


def test_alleninst_accelerating_negative():
    a = _accel()
    ai = AI.adaptation_index_AllenInst(a, trace_duration=_dur(a))
    assert ai < 0.0


def test_alleninst_isis_input_matches_latency_input():
    """Passing precomputed ISIs (input='isis') for the whole train should give
    the same value as passing spike latencies (input='spike_latencies'), because
    the latency path just diffs the (already in-window) latencies to get ISIs."""
    d = _decel()
    from_lat = AI.adaptation_index_AllenInst(d, trace_duration=_dur(d),
                                             input="spike_latencies")
    isis = np.diff(d)
    from_isis = AI.adaptation_index_AllenInst(isis, trace_duration=_dur(d),
                                              input="isis")
    assert np.isclose(from_lat, from_isis)


def test_alleninst_invalid_input_raises():
    with pytest.raises(AssertionError):
        AI.adaptation_index_AllenInst(_decel(), input="not_a_mode")


def test_alleninst_too_few_spikes_returns_nan():
    assert np.isnan(AI.adaptation_index_AllenInst(np.array([0.1, 0.2, 0.3]),
                                                  trace_duration=1.0, minimum_spikes=4))


# ---------------------------------------------------------------------------
# adaptation_index_AllenInst_spktimes
# ---------------------------------------------------------------------------
def test_alleninst_spktimes_finite_for_valid_train():
    d = _decel()
    ai = AI.adaptation_index_AllenInst_spktimes(d, trace_duration=_dur(d))
    assert np.isfinite(ai)


def test_alleninst_spktimes_trims_window_and_honours_minimum():
    # only 3 spikes fall inside [0, trace_duration); minimum_spikes=4 -> nan
    spk = np.array([0.1, 0.2, 0.3, 5.0, 6.0])
    assert np.isnan(AI.adaptation_index_AllenInst_spktimes(spk, trace_duration=1.0,
                                                           minimum_spikes=4))


# ---------------------------------------------------------------------------
# adaptation_first_last_isi
# ---------------------------------------------------------------------------
def test_first_last_isi_adapting_ratio_gt_one():
    d = _decel()
    ratio = AI.adaptation_first_last_isi(d, trace_duration=_dur(d))
    assert ratio > 1.0


def test_first_last_isi_accelerating_ratio_lt_one():
    a = _accel()
    ratio = AI.adaptation_first_last_isi(a, trace_duration=_dur(a))
    assert 0.0 < ratio < 1.0


def test_first_last_isi_too_few_spikes_returns_nan():
    assert np.isnan(AI.adaptation_first_last_isi(np.array([0.1, 0.2]),
                                                 trace_duration=1.0, minimum_spikes=4))


# ---------------------------------------------------------------------------
# adaptation_index_exponential
# ---------------------------------------------------------------------------
def test_exponential_returns_a_number_without_raising():
    d = _decel()
    val = AI.adaptation_index_exponential(d, trace_duration=_dur(d))
    # exact value depends on the lmfit result; just require a real scalar
    assert np.isnan(val) or np.isfinite(val)


def test_exponential_too_few_spikes_returns_nan():
    assert np.isnan(AI.adaptation_index_exponential(np.array([0.1, 0.2]),
                                                    trace_duration=1.0, minimum_spikes=4))


# ---------------------------------------------------------------------------
# compute_adaptation_index (dispatcher)
# ---------------------------------------------------------------------------
# Spike-time keys (the dispatcher reads spike times from the inner dict keys).
_DISPATCH_TIMES = (0.16, 0.18, 0.21, 0.25, 0.30, 0.36)
_DISPATCH_DELAY = 0.15


def _windowed_latencies():
    """Reproduce the dispatcher's own windowing: subtract trace_delay, keep
    latencies within [0, trace_duration)."""
    lat = np.array(_DISPATCH_TIMES) - _DISPATCH_DELAY
    return lat[(lat < 1.0) & (lat >= 0.0)]


def test_compute_adaptation_index_dispatcher_runs():
    spikes = {0: {t: None for t in _DISPATCH_TIMES}}
    idx, rates = AI.compute_adaptation_index(
        "MKX2019", spikes, trace_delay=_DISPATCH_DELAY, trace_duration=1.0,
        rate_bounds=[0.0, 1e6], minimum_spikes=4,
    )
    assert isinstance(idx, list) and isinstance(rates, list)
    assert len(idx) == 1 and len(rates) == 1


@pytest.mark.parametrize("method, fn", [
    ("MKX2019", AI.adaptation_index_MKX2019),
    ("AllenInst", lambda lat, dur, minimum_spikes: AI.adaptation_index_AllenInst(
        lat, dur, minimum_spikes=minimum_spikes, input="spike_latencies")),
    ("FirstLastISI", AI.adaptation_first_last_isi),
])
def test_dispatcher_matches_named_subfunction(method, fn):
    spikes = {0: {t: None for t in _DISPATCH_TIMES}}
    idx, _ = AI.compute_adaptation_index(
        method, spikes, trace_delay=_DISPATCH_DELAY, trace_duration=1.0,
        rate_bounds=[0.0, 1e6], minimum_spikes=4,
    )
    expected = fn(_windowed_latencies(), 1.0, minimum_spikes=4)
    assert np.isclose(idx[0], expected, equal_nan=True)


def test_dispatcher_excludes_out_of_rate_bounds():
    # windowed rate is ~30 Hz; a [100, 200] Hz band excludes the sweep entirely
    spikes = {0: {t: None for t in _DISPATCH_TIMES}}
    idx, rates = AI.compute_adaptation_index(
        "MKX2019", spikes, trace_delay=_DISPATCH_DELAY, trace_duration=1.0,
        rate_bounds=[100.0, 200.0], minimum_spikes=4,
    )
    assert idx == [] and rates == []


def test_dispatcher_skips_sweeps_below_minimum_spikes():
    spikes = {0: {0.2: None, 0.3: None}}  # only 2 spikes, minimum 4
    idx, rates = AI.compute_adaptation_index(
        "MKX2019", spikes, trace_delay=_DISPATCH_DELAY, trace_duration=1.0,
        rate_bounds=[0.0, 1e6], minimum_spikes=4,
    )
    assert idx == [] and rates == []
