# -*- encoding: utf-8 -*-
"""
Pytest coverage for ephys.ephys_analysis.make_clamps.MakeClamps.

MakeClamps builds an acq4-style Clamps object (with a MetaArray ``traces`` that
supports ``["Time":t0:t1]`` slicing) from plain time / data / command arrays.
It is the adapter that lets SpikeAnalysis and RmTauAnalysis consume non-acq4
data.  These tests use small synthetic current-clamp arrays only.
"""
import numpy as np
import pytest

import ephys.ephys_analysis.make_clamps as MK


# ---------------------------------------------------------------------------
# Synthetic CC dataset: flat membrane trace + boxcar current command steps
# ---------------------------------------------------------------------------
NPTS = 1000
DT = 1e-4                      # 100 us -> 10 kHz
TSTART = 0.01
TDUR = 0.05
LEVELS = np.array([-100e-12, 0.0, 100e-12])  # 3 command steps, in A


def _build(dmode="CC", protocol="test/CCIV_000"):
    time = np.arange(NPTS) * DT
    data = np.zeros((len(LEVELS), NPTS))        # flat membrane traces
    cmd = np.zeros((len(LEVELS), NPTS))
    i0 = int(TSTART / DT)
    i1 = int((TSTART + TDUR) / DT)
    for i, lev in enumerate(LEVELS):
        cmd[i, i0:i1] = lev                     # boxcar step
    MC = MK.MakeClamps()
    MC.set_clamps(time, data, cmddata=cmd, dmode=dmode,
                  tstart_tdur=[TSTART, TDUR], protocol=protocol)
    ok = MC.getData()
    return MC, ok


def test_getdata_returns_true():
    _, ok = _build()
    assert ok is True


def test_time_base_length_matches_traces():
    MC, _ = _build()
    assert MC.traces.shape == (len(LEVELS), NPTS)
    assert len(MC.time_base) == MC.traces.shape[1]


def test_tstart_tend_tdur_from_constructor_args():
    MC, _ = _build()
    assert np.isclose(MC.tstart, TSTART)
    assert np.isclose(MC.tdur, TDUR)
    assert np.isclose(MC.tend, TSTART + TDUR)


def test_sample_interval_from_time_array():
    MC, _ = _build()
    assert np.isclose(MC.sample_interval, DT)


def test_values_and_command_levels_match_steps():
    MC, _ = _build()
    # the mean of each command over the step window is the step level
    assert np.allclose(np.asarray(MC.values), LEVELS)
    assert np.allclose(np.asarray(MC.commandLevels), LEVELS)


def test_holding_is_prestep_mean_zero():
    MC, _ = _build()
    # command is zero before the step -> holding ~ 0
    assert np.allclose(np.asarray(MC.holding).ravel(), 0.0)


def test_traces_support_metaarray_time_slicing():
    MC, _ = _build()
    sl = MC.traces["Time":0.0:0.01]          # first 10 ms
    arr = np.asarray(sl)
    assert arr.shape == (len(LEVELS), int(0.01 / DT))


def test_no_data_raises_valueerror():
    MC = MK.MakeClamps()
    MC.set_clamps(np.arange(NPTS) * DT, None, cmddata=None,
                  tstart_tdur=[TSTART, TDUR])
    with pytest.raises(ValueError):
        MC.getClampData()


def test_set_protocol_updates_protocol():
    MC, _ = _build()
    MC.setProtocol("some/other/Protocol_001")
    assert MC.protocol == "some/other/Protocol_001"
