# -*- encoding: utf-8 -*-
"""
Pytest coverage for ephys.ephys_analysis.burst_analysis.BurstAnalyzer.

BurstAnalyzer detects bursts in per-trial spike-time arrays using an absolute
inter-spike-interval threshold (max_ISI) and a minimum spike count (min_spikes).
All inputs here are deterministic synthetic spike trains, so no live data is
needed.  burst_maker() (which uses np.random) is tested only for structural
shape, not exact contents.
"""
import numpy as np
import pytest

import ephys.ephys_analysis.burst_analysis as BA


# ---------------------------------------------------------------------------
# Burst detection
# ---------------------------------------------------------------------------
def test_isolated_spikes_give_no_bursts():
    # ISI of 1.0 s is far above max_ISI=0.1 -> nothing clusters
    trials = [np.arange(0, 5) * 1.0]
    a = BA.BurstAnalyzer(trials, [0.0], max_ISI=0.1, min_spikes=3)
    assert len(a.bursts) == 0


def test_tight_cluster_gives_one_burst():
    trials = [np.array([0.0, 0.01, 0.02, 0.03, 0.04])]  # ISI = 0.01 < max_ISI
    a = BA.BurstAnalyzer(trials, [0.0], max_ISI=0.1, min_spikes=3)
    assert len(a.bursts) == 1
    assert len(a.bursts[0]["spikes"]) == 5
    assert a.bursts[0]["trial"] == 0


def test_two_separated_clusters_give_two_bursts():
    trials = [np.array([0.0, 0.01, 0.02, 0.03, 5.0, 5.01, 5.02, 5.03])]
    a = BA.BurstAnalyzer(trials, [0.0], max_ISI=0.1, min_spikes=3)
    assert len(a.bursts) == 2


def test_cluster_below_min_spikes_not_a_burst():
    trials = [np.array([0.0, 0.01])]  # only 2 spikes, min_spikes=3
    a = BA.BurstAnalyzer(trials, [0.0], max_ISI=0.1, min_spikes=3)
    assert len(a.bursts) == 0


def test_min_spikes_threshold_boundary():
    # exactly min_spikes tightly clustered -> one burst
    trials = [np.array([0.0, 0.01, 0.02])]
    a = BA.BurstAnalyzer(trials, [0.0], max_ISI=0.1, min_spikes=3)
    assert len(a.bursts) == 1


def test_length_mismatch_raises():
    with pytest.raises(ValueError):
        BA.BurstAnalyzer([np.array([0.0, 0.01])], [0.0, 1.0])


# ---------------------------------------------------------------------------
# Evolution / metrics
# ---------------------------------------------------------------------------
def test_calculate_evolution_returns_expected_keys():
    trials = [np.array([0.0, 0.01, 0.02, 0.03, 5.0, 5.01, 5.02, 5.03])]
    a = BA.BurstAnalyzer(trials, [0.0], max_ISI=0.1, min_spikes=3)
    ev = a.calculate_evolution(bin_type="mixed", bin_size=1)
    assert set(ev.keys()) == {"by_trial", "by_time"}
    assert isinstance(ev["by_trial"], list)
    assert isinstance(ev["by_time"], list)


def test_calculate_evolution_empty_when_no_bursts():
    trials = [np.arange(0, 5) * 1.0]
    a = BA.BurstAnalyzer(trials, [0.0], max_ISI=0.1, min_spikes=3)
    ev = a.calculate_evolution()
    assert ev["by_trial"] == [] and ev["by_time"] == []


def test_trial_metrics_counts_match_detected_bursts():
    # trial 0: two clusters; trial 1: one cluster
    trials = [
        np.array([0.0, 0.01, 0.02, 0.03, 5.0, 5.01, 5.02, 5.03]),
        np.array([0.0, 0.01, 0.02, 0.03]),
    ]
    starts = [0.0, 10.0]
    a = BA.BurstAnalyzer(trials, starts, max_ISI=0.1, min_spikes=3)
    metrics = a.trial_metrics()
    by_trial = {m["trial"]: m for m in metrics}
    assert by_trial[0]["n_bursts"] == 2
    assert by_trial[1]["n_bursts"] == 1
    # metric bookkeeping is self-consistent with self.bursts
    total = sum(m["n_bursts"] for m in metrics)
    assert total == len(a.bursts)


# ---------------------------------------------------------------------------
# burst_maker helper
# ---------------------------------------------------------------------------
def test_burst_maker_returns_one_array_per_trial():
    np.random.seed(0)
    ntrials = 6
    out = BA.burst_maker(
        ntrials=ntrials,
        mean_spikes_per_burst=20,
        mean_interburst_interval=1.0,
        mean_interburst_std=0.05,
        mean_burst_duration=0.4,
        trial_duration=10,
    )
    assert isinstance(out, list)
    assert len(out) == ntrials
    assert all(isinstance(t, np.ndarray) for t in out)


def test_burst_maker_output_feeds_analyzer():
    np.random.seed(1)
    ntrials = 5
    trials = BA.burst_maker(
        ntrials=ntrials,
        mean_spikes_per_burst=25,
        mean_interburst_interval=1.0,
        mean_interburst_std=0.05,
        mean_burst_duration=0.4,
        trial_duration=10,
    )
    starts = np.linspace(0, 10.0 * ntrials, ntrials)
    a = BA.BurstAnalyzer(trials, starts, max_ISI=0.1, min_spikes=3)
    # densely-packed synthetic bursts should yield at least one detected burst
    assert len(a.bursts) >= 1
    for b in a.bursts:
        assert len(b["spikes"]) >= 3
        assert b["duration"] >= 0.0
