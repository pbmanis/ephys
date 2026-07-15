# -*- coding: utf-8 -*-
"""
spike_analysis_fast.py — Optimised wrappers for SpikeAnalysis.analyzeSpikeShape.

Optimisations implemented (each cumulative on the one before):

  Phase 1 — dvdt pre-computation
      np.diff(trace)/dt is computed ONCE per trace, not once per spike.

  Phase 2 — np.std pre-computation
      np.std(trace) is computed ONCE per trace and passed as a scalar to
      interpolate_halfwidth_fast, eliminating two redundant full-trace std
      calls per spike (one per branch of the original interpolate_halfwidth).

  Phase 2 — Numba backward scan
      The pure-Python backward-scan loop that locates the pre-spike minimum
      is replaced by a @jit(nopython=True) function (_backward_scan).

  Phase 3 (planned) — thread parallelism across traces.

All numeric results are identical to the original SpikeAnalysis methods;
test_spikes.py verifies this with atol=1e-10.
"""

from __future__ import annotations

import os
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Union

import numpy as np
from numba import jit  # type: ignore[import]

_USE_NUMBA_CACHE = not bool(os.environ.get("EPHYS_NO_NUMBA_CACHE", ""))
# Claude fixed 2026-07-15: concurrent first-run compilation races on the shared .nbc file
# Set EPHYS_NO_NUMBA_CACHE=1 in CI or when running tests in parallel to disable

# from ephys.ephys_analysis.spike_analysis import (
#     OneSpike,
#     SpikeAnalysis,
# )
import ephys.ephys_analysis.spike_analysis as SA  # 2026-07-15: change import so that module reference stays live after hot-reload


# ---------------------------------------------------------------------------
# Numba-compiled helpers
# ---------------------------------------------------------------------------

@jit(nopython=True, cache=_USE_NUMBA_CACHE)
def _backward_scan(
    trace: np.ndarray,
    kpeak: int,
    t_step_start: int,
    band: float,
) -> int:
    """Find the index of the pre-spike voltage minimum.

    Scans backwards from kpeak toward the stimulus onset (t_step_start),
    tracking the lowest voltage encountered.  Stops as soon as the voltage
    rises more than *band* above that minimum.

    Replaces the equivalent pure-Python loop in SpikeAnalysis.analyze_one_spike.
    Logic is identical; None-sentinel replaced by a boolean flag for nopython.
    """
    min_point = kpeak
    min_v_found = False
    min_v = 0.0
    for km in range(kpeak - 1, t_step_start, -1):
        delta = trace[km] - trace[km + 1]
        if delta < 0:
            min_point = km
            min_v = trace[km]
            min_v_found = True
        elif delta > 0:
            if not min_v_found:
                continue
            if trace[km] > (min_v + band):
                break
    return min_point


@jit(nopython=True, cache=_USE_NUMBA_CACHE)
def interpolate_halfwidth_fast(
    tr: np.ndarray,
    xr: np.ndarray,
    kup: int,
    halfv: float,
    kdown: int,
    std_tr: float,        # pre-computed std — NOT recomputed inside here
) -> tuple:
    """Like interpolate_halfwidth but std_tr is passed in rather than computed.

    The original calls np.std(tr) twice (once per guard branch), each scanning
    the full trace.  Here it is computed once per trace by the caller and
    passed as a scalar, eliminating two O(N) scans per spike.
    """
    if tr[kup] <= halfv:
        vi = tr[kup - 1 : kup + 1]
        xi = xr[kup - 1 : kup + 1]
    else:
        vi = tr[kup : kup + 2]
        xi = xr[kup : kup + 2]
    m1 = (vi[1] - vi[0]) / (xi[1] - xi[0])
    b1 = vi[1] - m1 * xi[1]
    if m1 == 0.0 or std_tr == 0.0:
        return None, None

    t_hwup = (halfv - b1) / m1

    if tr[kdown] <= halfv:
        vi = tr[kdown : kdown + 2]
        xi = xr[kdown : kdown + 2]
    else:
        vi = tr[kdown - 1 : kdown + 1]
        xi = xr[kdown - 1 : kdown + 1]
    m2 = (vi[1] - vi[0]) / (xi[1] - xi[0])
    b2 = vi[1] - m2 * xi[1]
    if m2 == 0.0 or std_tr == 0.0:
        return None, None

    t_hwdown = (halfv - b2) / m2
    return t_hwdown, t_hwup


# ---------------------------------------------------------------------------
# Per-spike analysis
# ---------------------------------------------------------------------------

def analyze_one_spike_fast(
    sa: SA.SpikeAnalysis,
    trace_number: int,
    spike_number: int,
    spike_begin_dV: float,
    max_spike_shape: Union[int, None],
    dvdt: np.ndarray,    # Phase 1: pre-computed for this trace
    tr_arr: np.ndarray,  # Phase 2: typed float64 view of the trace
    std_tr: float,       # Phase 2: pre-computed std of the trace
) -> SA.OneSpike:
    """Like SpikeAnalysis.analyze_one_spike with three pre-computed inputs.

    dvdt    — computed once per trace (Phase 1)
    tr_arr  — float64 array of the trace, computed once per trace
    std_tr  — np.std of the trace, computed once per trace (Phase 2)

    The backward scan is delegated to _backward_scan (JIT, Phase 2).
    The halfwidth interpolation uses interpolate_halfwidth_fast (JIT, Phase 2).
    """
    # clamps typed as Any: setup() takes clamps=None (untyped), so pyright
    # cannot infer Clamps attributes — the Any cast bypasses those false errors.
    clamps: Any = sa.Clamps

    thisspike = SA.OneSpike(trace=trace_number, AP_number=spike_number)
    thisspike.current = clamps.values[trace_number] - sa.iHold_i[trace_number]
    thisspike.iHold = sa.iHold_i[trace_number]
    thisspike.pulseDuration = clamps.tend - clamps.tstart
    thisspike.AP_peakIndex = sa.spikeIndices[trace_number][spike_number]
    thisspike.peak_T = clamps.time_base[thisspike.AP_peakIndex]
    thisspike.peak_V = tr_arr[thisspike.AP_peakIndex]  # type: ignore[assignment]
    thisspike.tstart = clamps.tstart
    thisspike.tend = clamps.tend

    dt = float(clamps.time_base[1] - clamps.time_base[0])
    thisspike.dt = dt
    t_step_start = int(clamps.tstart / dt)
    kpeak: int = int(sa.spikeIndices[trace_number][spike_number])

    if spike_number > 0:
        kprevious = sa.spikeIndices[trace_number][spike_number - 1]
    else:
        # Phase 2: JIT-compiled backward scan replaces Python loop
        kprevious = _backward_scan(tr_arr, kpeak, t_step_start, 1e-3)

    if kpeak - kprevious <= 2:
        print("peak too close to 'previous' spike: ", trace_number, kprevious, kpeak)
        return thisspike

    kbegin = int(np.argmin(tr_arr[kprevious:kpeak])) + kprevious
    km: int = int(np.argmax(dvdt[kbegin:kpeak])) + kbegin
    kthresholds = np.argwhere(dvdt[kbegin:km] < spike_begin_dV)
    if len(kthresholds) == 0:
        return thisspike

    kthresh: int = int(kthresholds[-1][0]) + kbegin
    thisspike.AP_latency = clamps.time_base[kthresh]
    thisspike.AP_beginIndex = kthresh
    thisspike.AP_begin_V = tr_arr[thisspike.AP_beginIndex]  # type: ignore[assignment]
    if max_spike_shape is not None and spike_number > max_spike_shape and max_spike_shape > 0:
        return thisspike

    k = sa.spikeIndices[trace_number][spike_number] + 1
    if spike_number < int(sa.spikecount[trace_number]) - 1:
        kend = sa.spikeIndices[trace_number][spike_number + 1]
    else:
        kend = int(sa.spikeIndices[trace_number][spike_number] + sa.max_spike_look / dt)
    if kend >= dvdt.shape[0]:
        return thisspike
    else:
        if kend < k:
            kend = k + 1
        km = int(np.argmin(dvdt[k:kend])) + k

    kmin = int(np.argmin(tr_arr[km:kend])) + km
    thisspike.AP_endIndex = kmin
    thisspike.trough_T = clamps.time_base[thisspike.AP_endIndex]
    thisspike.trough_V = tr_arr[kmin]  # type: ignore[assignment]

    if thisspike.AP_endIndex is not None:
        thisspike.peaktotrough = thisspike.trough_T - thisspike.peak_T

    five_ms = int(5e-3 / dt)
    four_ms = int(4e-3 / dt)
    one_ms = int(1e-3 / dt)
    thisspike.dvdt_rising = np.max(dvdt[thisspike.AP_beginIndex : thisspike.AP_peakIndex])
    thisspike.dvdt_falling = np.min(dvdt[thisspike.AP_peakIndex : thisspike.AP_endIndex])
    thisspike.dvdt = dvdt[thisspike.AP_beginIndex - four_ms : thisspike.AP_endIndex + one_ms].copy()
    thisspike.V = tr_arr[
        thisspike.AP_beginIndex - four_ms : thisspike.AP_endIndex + five_ms
    ].copy()
    thisspike.Vtime = clamps.time_base[
        thisspike.AP_beginIndex - four_ms : thisspike.AP_endIndex + five_ms
    ].copy()

    if (
        thisspike.AP_beginIndex is not None
        and thisspike.AP_beginIndex > 0
        and thisspike.AP_endIndex is not None
        and thisspike.AP_beginIndex < thisspike.AP_peakIndex
        and thisspike.AP_peakIndex < thisspike.AP_endIndex
    ):
        halfv = 0.5 * (thisspike.peak_V + thisspike.AP_begin_V)
        xr = clamps.time_base
        kup = (
            int(np.argmin(np.fabs(tr_arr[thisspike.AP_beginIndex : thisspike.AP_peakIndex] - halfv)))
            + thisspike.AP_beginIndex
        )
        kdown = (
            int(np.argmin(np.fabs(tr_arr[thisspike.AP_peakIndex : thisspike.AP_endIndex] - halfv)))
            + thisspike.AP_peakIndex
        )
        thisspike.halfwidth = xr[kdown] - xr[kup]
        thisspike.halfwidth_up = xr[kup] - xr[thisspike.AP_peakIndex]
        thisspike.halfwidth_down = xr[thisspike.AP_peakIndex] - xr[kdown]
        thisspike.halfwidth_V = halfv
        thisspike.left_halfwidth_T = xr[kup]
        thisspike.left_halfwidth_V = tr_arr[kup]    # type: ignore[assignment]
        thisspike.right_halfwidth_T = xr[kdown]
        thisspike.right_halfwidth_V = tr_arr[kdown]  # type: ignore[assignment]

        # Phase 2: use JIT version with pre-computed std_tr
        t_hwdown, t_hwup = interpolate_halfwidth_fast(
            tr_arr, xr, kup, halfv, kdown, std_tr
        )
        if t_hwdown is None:
            return thisspike

        thisspike.halfwidth = t_hwdown - t_hwup
        if thisspike.halfwidth > sa.min_halfwidth:
            if sa.verbose:
                print("   halfv: ", halfv, thisspike.peak_V, thisspike.AP_begin_V)
            thisspike.halfwidth = None
            thisspike.halfwidth_interpolated = None
        else:
            thisspike.halfwidth_interpolated = t_hwdown - t_hwup
        pkvI = tr_arr[thisspike.AP_peakIndex]
        pkvM = np.max(tr_arr[thisspike.AP_beginIndex : thisspike.AP_endIndex])
        pkvMa = np.argmax(tr_arr[thisspike.AP_beginIndex : thisspike.AP_endIndex])  # noqa: F841
        if pkvI != pkvM:
            pktrap = True  # noqa: F841

    return thisspike


# ---------------------------------------------------------------------------
# Per-trace analysis
# ---------------------------------------------------------------------------

def analyze_one_trace_fast(
    sa: SA.SpikeAnalysis,
    trace_number: int,
    begin_dV: float = 12.0,
    max_spike_shape: Union[int, None] = 5,
    printSpikeInfo: bool = False,
) -> Union[dict, None]:
    """Like SpikeAnalysis.analyze_one_trace with per-trace pre-computation.

    Computes tr_arr (float64 array), dvdt, and std_tr once per trace and
    passes them to analyze_one_spike_fast for every spike in the trace.
    """
    clamps: Any = sa.Clamps  # see note in analyze_one_spike_fast

    if len(sa.spikes[trace_number]) == 0:
        return None
    if printSpikeInfo:
        print("spikes: ", sa.spikes[trace_number])
        print(np.array(clamps.values))
        print(len(clamps.traces))

    (sa.rmps[trace_number], _) = sa.U.measure(
        "mean", clamps.time_base, clamps.traces[trace_number],
        0.0, clamps.tstart,
    )
    (sa.iHold_i[trace_number], _) = sa.U.measure(
        "mean", clamps.time_base, clamps.cmd_wave[trace_number],
        0.0, clamps.tstart,
    )

    # Phase 1 + 2: compute per-trace quantities ONCE
    tr_arr = np.asarray(clamps.traces[trace_number], dtype=np.float64)
    dt     = float(clamps.time_base[1] - clamps.time_base[0])
    dvdt   = np.diff(tr_arr) / dt
    std_tr = float(np.std(tr_arr))

    trspikes: dict = OrderedDict()
    for spike_number in range(len(sa.spikes[trace_number])):
        thisspike = analyze_one_spike_fast(
            sa, trace_number, spike_number, begin_dV, max_spike_shape,
            dvdt, tr_arr, std_tr,
        )
        if thisspike is not None:
            trspikes[spike_number] = thisspike
 #  just return this trace, let caller write dictionary.
 #  old: sa.spikeShapes[trace_number] = trspikes
    return trspikes


# ---------------------------------------------------------------------------
# Top-level entry point
# ---------------------------------------------------------------------------

def analyzeSpikeShape_fast(
    sa: SA.SpikeAnalysis,
    spike_begin_dV: float = 12.0,
    max_spike_shape: Union[int, None] = 5,
    printSpikeInfo: bool = False,
    n_workers: int = 1,
) -> None:
    """Drop-in replacement for SpikeAnalysis.analyzeSpikeShape.

    Uses analyze_one_trace_fast (Phase 1+2 optimisations).
    n_workers > 1 enables thread-parallel trace processing (Phase 3).
    All post-loop logic is identical to the original.
    """
    clamps: Any = sa.Clamps  # see note in analyze_one_spike_fast

    sa._initialize_summarymeasures()
    sa.madeplot = False
    ntr = len(clamps.traces)
    sa.spikeShapes = OrderedDict()
    sa.rmps    = np.zeros(ntr)
    sa.iHold_i = np.zeros(ntr)

    # results are collected into a list, merged serially to make thread-safe
    # 7/15/2026 pbm
    traces_with_spikes = [tr for tr in range(ntr) if len(sa.spikes[tr]) > 0]
    if n_workers <= 1:
        for tr in traces_with_spikes:
            result = analyze_one_trace_fast(sa, tr, spike_begin_dV, max_spike_shape, printSpikeInfo)
            if result is not None:  # 7/15/2026: fix for analyze_one_trace_fast new return
                sa.spikeShapes[tr] = result
    else:
        _results: list = [None] * ntr   # pre-allocated; each thread writes a unique index
        def _worker(tr):
            _results[tr] = analyze_one_trace_fast(
                sa, tr, spike_begin_dV, max_spike_shape, printSpikeInfo
            )
        _nw = min(n_workers, len(traces_with_spikes), os.cpu_count() or 1)
        with ThreadPoolExecutor(max_workers=_nw) as pool:
            pool.map(_worker, traces_with_spikes)
        # Merge serially — no concurrent access
        for tr in traces_with_spikes:
            if _results[tr] is not None:
                sa.spikeShapes[tr] = _results[tr]

    sa.iHold = np.mean(sa.iHold_i)
    sa.analysis_summary["spikes"] = sa.spikeShapes
    sa.analysis_summary["iHold"]  = sa.iHold
    try:
        sa.analysis_summary["pulseDuration"] = clamps.tend - clamps.tstart
    except Exception:
        sa.analysis_summary["pulseDuration"] = np.max(clamps.time_base)

    if len(sa.spikeShapes) > 0:
        lcs = sa.get_lowest_current_spike(
            minimum_current=sa.lcs_minimum_current,
            minimum_postspike_interval=sa.lcs_minimum_postspike_interval,
        )
        sa.analysis_summary["LowestCurrentSpike"] = lcs
        sa.getClassifyingInfo()
    else:
        sa.analysis_summary["LowestCurrentSpike"] = None
    print("\n\nLowest current spike info (fast): ", sa.analysis_summary["LowestCurrentSpike"])
