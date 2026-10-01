# -*- encoding: utf-8 -*-
"""
Pytest coverage for ephys.ephys_analysis.rm_tau_analysis.RmTauAnalysis.

RmTauAnalysis measures Rm (input resistance), the membrane time constant tau_m,
the resting membrane potential, and the peak / steady-state IV curves from
current-clamp data.  It requires an acq4-style Clamps object whose ``traces``
support MetaArray ``["Time":t0:t1]`` slicing, and it must be run *after*
SpikeAnalysis.

The synthetic HH-simulation fixture (HHData.pkl, also used by test_spikes.py) is
wrapped into a proper Clamps structure with MakeClamps so the MetaArray "Time"
slicing works.  Expected ranges below were calibrated against the actual output
of the pipeline on this fixture (RMP ~ -59.6 mV, Rin ~ 17 MOhm,
tau_m ~ 12.8 ms).
"""
import pickle
from pathlib import Path

import numpy as np
import pytest

import ephys.ephys_analysis.make_clamps as MK
import ephys.ephys_analysis.spike_analysis as SA
import ephys.ephys_analysis.rm_tau_analysis as RMT

MINIMAL_EXPERIMENT = {"FI_maximum_current_by_celltype": None}


def _load_hhdata():
    with open(Path(__file__).parent / "HHData.pkl", "rb") as fh:
        return pickle.load(fh)


def _build_clamps():
    """Wrap the HH IV fixture into an acq4-style (MetaArray-backed) Clamps."""
    IV = _load_hhdata()
    tstart = float(IV["tstart"])
    tdur = float(IV["tend"]) - float(IV["tstart"])
    MC = MK.MakeClamps()
    MC.set_clamps(
        np.asarray(IV["time_base"]),
        np.asarray(IV["traces"]),
        cmddata=np.asarray(IV["cmd_wave"]),
        dmode="CC",
        tstart_tdur=[tstart, tdur],
        protocol="test/CCIV_test_000",
    )
    MC.getData()
    return MC, tstart, tdur


def _build_spikes(clamps):
    sa = SA.SpikeAnalysis()
    sa.setup(experiment=MINIMAL_EXPERIMENT, clamps=clamps, threshold=-0.020, verbose=False)
    sa.set_detector("Kalluri", pars=None)
    sa.analyzeSpikes()
    sa.analyzeSpikes_brief(mode="baseline")
    sa.analyzeSpikes_brief(mode="poststimulus")
    return sa


def _analyze(clamps, spikes, tstart, tdur, bridge_offset=0.0):
    rmt = RMT.RmTauAnalysis()
    rmt.setup(
        clamps=clamps,
        spikes=spikes,
        taum_bounds=[0.0002, 0.1],
        baseline=[0.0, tstart - 0.002],
        bridge_offset=bridge_offset,
    )
    rmt.analyze(
        rmp_region=[0.0, tstart - 0.002],
        tau_region=[tstart, tstart + 0.05],
        rin_region=[tstart + 0.9 * tdur, tstart + tdur],
    )
    return rmt


# ---------------------------------------------------------------------------
# Module-scoped fixtures: build the (expensive) pipeline once.
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def clamps_and_spikes():
    clamps, tstart, tdur = _build_clamps()
    spikes = _build_spikes(clamps)
    return clamps, spikes, tstart, tdur


@pytest.fixture(scope="module")
def analyzed(clamps_and_spikes):
    clamps, spikes, tstart, tdur = clamps_and_spikes
    return _analyze(clamps, spikes, tstart, tdur)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
def test_analyze_runs_without_exception(analyzed):
    assert analyzed.analysis_summary is not None
    assert "RMP" in analyzed.analysis_summary


def test_rmp_in_physiological_range(analyzed):
    rmp = analyzed.analysis_summary["RMP"]  # mV
    assert -80.0 < rmp < -50.0


def test_rin_positive_and_plausible(analyzed):
    rin = analyzed.analysis_summary["Rin"]  # MOhm
    assert np.isfinite(rin)
    assert 1.0 < rin < 1000.0


def test_taum_positive(analyzed):
    taum = analyzed.analysis_summary["taum"]  # seconds
    assert np.isfinite(taum)
    assert taum > 0.0
    # within the fitting bounds passed to setup
    assert 0.0002 < taum < 0.1


def test_ivss_curve_shapes_consistent(analyzed):
    cmd = np.asarray(analyzed.analysis_summary["ivss_cmd"])
    v = np.asarray(analyzed.analysis_summary["ivss_v"])
    assert cmd.shape == v.shape
    assert cmd.size > 2
    # commands should be sorted ascending (rm_tau sorts them internally)
    assert np.all(np.diff(cmd) >= 0)


def test_ivpk_curve_shapes_consistent(analyzed):
    cmd = np.asarray(analyzed.analysis_summary["ivpk_cmd"])
    v = np.asarray(analyzed.analysis_summary["ivpk_v"])
    assert cmd.shape == v.shape
    assert cmd.size > 2


def test_rmp_analysis_matches_direct_baseline_mean(clamps_and_spikes):
    """rmp_analysis should equal the mean of the raw baseline window (in mV)."""
    clamps, spikes, tstart, tdur = clamps_and_spikes
    rmt = _analyze(clamps, spikes, tstart, tdur)
    win = [0.0, tstart - 0.002]
    raw = np.asarray(clamps.traces["Time":win[0]:win[1]])
    expected_mV = np.mean(raw.mean(axis=1)) * 1e3
    assert np.isclose(rmt.analysis_summary["RMP"], expected_mV, rtol=0, atol=1e-6)


def test_bridge_adjust_is_noop_when_offset_is_zero(clamps_and_spikes):
    """bridge_adjust() with a zero bridge resistance must not alter the traces."""
    clamps, spikes, tstart, tdur = clamps_and_spikes
    before = np.array(clamps.traces.view(np.ndarray), copy=True)
    rmt = RMT.RmTauAnalysis()
    rmt.setup(clamps=clamps, spikes=spikes, bridge_offset=0.0,
              baseline=[0.0, tstart - 0.002])
    rmt.bridge_adjust()  # offset is 0 -> subtracts zero
    after = np.asarray(clamps.traces.view(np.ndarray))
    assert np.allclose(before, after, rtol=0, atol=0.0)


def test_results_stable_across_repeated_analyses(clamps_and_spikes):
    """Repeated independent analyses of the same data give identical results
    (no accumulated state leaking between runs)."""
    clamps, spikes, tstart, tdur = clamps_and_spikes
    r1 = _analyze(clamps, spikes, tstart, tdur)
    r2 = _analyze(clamps, spikes, tstart, tdur)
    for key in ("RMP", "Rin", "taum"):
        a, b = r1.analysis_summary[key], r2.analysis_summary[key]
        assert np.isclose(a, b, rtol=0, atol=1e-9, equal_nan=True), f"{key}: {a} != {b}"
