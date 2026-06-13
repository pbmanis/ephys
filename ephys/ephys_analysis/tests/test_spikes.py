# -*- encoding: utf-8 -*-
"""
Test fixture for ephys spike analysis methods.

Provides:
  - Existing pytest regression tests (unchanged below)
  - CLI comparison test: runs analyzeSpikeShape (original) vs the fast version,
    reports timing and any discrepancies in OneSpike attributes.

Usage (CLI):
  python test_spikes.py [OPTIONS]

  --data PATH     Path to HHData.pkl (default), or to an acq4 protocol directory
                  (e.g. /data/2024.01.15_000/slice_000/cell_000/CCIV_short_000)
  --method METHOD Spike detection method: Kalluri (default), argrelmax,
                  find_peaks, threshold
  --max-spikes N  Max spikes per trace analysed in detail (default: 5)
  --plot          Show trace + spike-marker plot after analysis
  --verbose       Enable verbose output from SpikeAnalysis

Usage (pytest):
  pytest test_spikes.py
"""

import argparse
import dataclasses
import pickle
import sys
import time
import traceback
from dataclasses import dataclass, fields
from pathlib import Path
from types import ModuleType
from typing import Any, Optional, Tuple

import numpy as np

import matplotlib.pyplot as mpl
import ephys.ephys_analysis.rm_tau_analysis as RMT
import ephys.ephys_analysis.spike_analysis as SA
from ephys.ephys_analysis.spike_analysis import OneSpike
from ephys.mini_analyses.util import UserTester

# ---------------------------------------------------------------------------
# Try importing the fast module; gracefully skip comparison if not yet built.
# ---------------------------------------------------------------------------
SAF: Optional[ModuleType] = None
HAS_FAST: bool = False
try:
    import ephys.ephys_analysis.spike_analysis_fast as _saf_mod  # type: ignore[import]
    SAF = _saf_mod
    HAS_FAST = True
except ImportError:
    pass

# ---------------------------------------------------------------------------
# Minimum experiment dict required by SpikeAnalysis.setup().
# Only FI_maximum_current_by_celltype is accessed (in fitOne / getFISlope),
# neither of which is called during analyzeSpikeShape.
# ---------------------------------------------------------------------------
MINIMAL_EXPERIMENT: dict = {
    "FI_maximum_current_by_celltype": None,
}

# ---------------------------------------------------------------------------
# OneSpike field classification for comparison
# ---------------------------------------------------------------------------
_ONESPIKE_INT_FIELDS = {"AP_beginIndex", "AP_peakIndex", "AP_endIndex", "AP_number", "trace"}
_ONESPIKE_ARRAY_FIELDS = {"dvdt", "V", "Vtime"}
# All remaining numeric fields are treated as floats.
_ONESPIKE_FLOAT_FIELDS = {
    f.name for f in fields(OneSpike)
    if f.name not in _ONESPIKE_INT_FIELDS | _ONESPIKE_ARRAY_FIELDS
    and f.name not in {"trace", "AP_number"}  # ints stored as int
}


# ---------------------------------------------------------------------------
# Clamps-compatible dataclass (mirrors acq4 Clamps interface)
# ---------------------------------------------------------------------------
@dataclass
class HHIV:
    """Mimics the Clamps object expected by SpikeAnalysis."""
    mode: str = "ic"
    traces: Any = None
    cmd_wave: Any = None
    values: Any = None
    commandLevels: Any = None
    time_base: Any = None
    tstart: float = 0.02   # seconds
    tend: float = 0.07     # seconds
    tdur: float = 0.2      # seconds
    sample_interval: float = 1e-5  # seconds


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def get_testdata() -> HHIV:
    """Load the built-in HH simulation test data."""
    testpath = Path(__file__).parent
    with open(Path(testpath, "HHData.pkl"), "rb") as fh:
        IV = pickle.load(fh)
    return HHIV(**IV)


def build_clamps_from_acq4(protocol_path: Path) -> HHIV:
    """Load an acq4 current-clamp protocol and return an HHIV-compatible object.

    Parameters
    ----------
    protocol_path : Path
        Full path to an acq4 protocol directory
        (e.g. /data/2024.01.15_000/slice_000/cell_000/CCIV_short_000).

    Returns
    -------
    HHIV
        Populated with traces, time_base, cmd_wave, values, tstart, tend,
        sample_interval, commandLevels read from the protocol.
    """
    from ephys.datareaders.acq4_reader import acq4_reader

    AR = acq4_reader(pathtoprotocol=protocol_path)
    AR.getData()

    # acq4_reader stores time_base as a list of arrays (one per trace);
    # SpikeAnalysis expects a single 1-D time array.
    time_base = np.array(AR.time_base[0]) if isinstance(AR.time_base, list) else np.array(AR.time_base)
    traces = [np.array(t) for t in AR.traces]
    cmd_wave = [np.array(c) for c in AR.cmd_wave]

    return HHIV(
        mode=AR.mode.lower() if AR.mode else "ic",
        traces=traces,
        cmd_wave=cmd_wave,
        values=list(AR.values) if AR.values is not None else list(AR.commandLevels),
        commandLevels=np.array(AR.commandLevels),
        time_base=time_base,
        tstart=AR.tstart,
        tend=AR.tend,
        sample_interval=AR.sample_interval,
    )


def load_data(data_arg: Optional[str]) -> Tuple[HHIV, str]:
    """Return (clamps, source_label) from a --data argument string."""
    if data_arg is None or data_arg in ("HHData", "HHData.pkl"):
        return get_testdata(), "HHData.pkl (built-in HH simulation)"
    p = Path(data_arg)
    if not p.exists():
        raise FileNotFoundError(f"Data path not found: {p}")
    if p.suffix == ".pkl":
        with open(p, "rb") as fh:
            IV = pickle.load(fh)
        return HHIV(**IV), str(p)
    # Assume acq4 protocol directory
    return build_clamps_from_acq4(p), str(p)


# ---------------------------------------------------------------------------
# SpikeAnalysis runner
# ---------------------------------------------------------------------------

def build_analyzer(clamps: HHIV, method: str = "Kalluri",
                   verbose: bool = False) -> SA.SpikeAnalysis:
    """Create and configure a SpikeAnalysis instance, detect spikes."""
    sa = SA.SpikeAnalysis()
    sa.setup(
        experiment=MINIMAL_EXPERIMENT,
        clamps=clamps,
        threshold=-0.020,
        verbose=verbose,
    )
    sa.set_detector(method, pars=None)
    sa.analyzeSpikes()
    return sa


def run_original(clamps: HHIV, method: str = "Kalluri",
                 max_spike_shape: int = 5,
                 verbose: bool = False) -> Tuple[dict, float]:
    """Run original analyzeSpikeShape; return (spikeShapes, elapsed_seconds)."""
    sa = build_analyzer(clamps, method=method, verbose=verbose)
    print("  build_analyzer done — starting analyzeSpikeShape …", flush=True)
    t0 = time.perf_counter()
    try:
        sa.analyzeSpikeShape(max_spike_shape=max_spike_shape, printSpikeInfo=False)
    except Exception:
        elapsed = time.perf_counter() - t0
        print(f"\n*** analyzeSpikeShape raised an exception after {elapsed:.4f} s ***", flush=True)
        traceback.print_exc()
        raise
    elapsed = time.perf_counter() - t0
    return dict(sa.spikeShapes), elapsed


def run_fast(clamps: HHIV, method: str = "Kalluri",
             max_spike_shape: int = 5,
             verbose: bool = False,
             n_workers: int = 1) -> Tuple[dict, float]:
    """Run fast analyzeSpikeShape_fast; return (spikeShapes, elapsed_seconds).

    Raises ImportError if spike_analysis_fast module is not available.
    """
    if not HAS_FAST or SAF is None:
        raise ImportError(
            "spike_analysis_fast module not found — implement it first.\n"
            "Expected location: ephys/ephys_analysis/spike_analysis_fast.py"
        )
    sa = build_analyzer(clamps, method=method, verbose=verbose)
    t0 = time.perf_counter()
    SAF.analyzeSpikeShape_fast(sa, spike_begin_dV=12.0,  # type: ignore[attr-defined]
                               max_spike_shape=max_spike_shape,
                               n_workers=n_workers)
    elapsed = time.perf_counter() - t0
    return dict(sa.spikeShapes), elapsed


# ---------------------------------------------------------------------------
# OneSpike comparison
# ---------------------------------------------------------------------------

def _fmt(v) -> str:
    if v is None:
        return "None"
    if isinstance(v, float):
        return f"{v:.8g}"
    return str(v)


def compare_one_spike(orig: OneSpike, fast: OneSpike,
                      trace: int, spk: int,
                      atol: float = 1e-10) -> list:
    """Return a list of human-readable discrepancy strings (empty = identical)."""
    diffs = []

    for fname in _ONESPIKE_INT_FIELDS:
        v1 = getattr(orig, fname, None)
        v2 = getattr(fast, fname, None)
        if v1 != v2:
            diffs.append(
                f"  tr={trace} spk={spk}  {fname:30s}  orig={v1}  fast={v2}"
            )

    for fname in _ONESPIKE_FLOAT_FIELDS:
        v1 = getattr(orig, fname, None)
        v2 = getattr(fast, fname, None)
        if v1 is None and v2 is None:
            continue
        if v1 is None or v2 is None:
            diffs.append(
                f"  tr={trace} spk={spk}  {fname:30s}  "
                f"orig={_fmt(v1)}  fast={_fmt(v2)}  (one is None)"
            )
            continue
        if not np.isclose(float(v1), float(v2), rtol=0, atol=atol):
            diffs.append(
                f"  tr={trace} spk={spk}  {fname:30s}  "
                f"orig={_fmt(v1)}  fast={_fmt(v2)}  "
                f"delta={abs(float(v1)-float(v2)):.2e}"
            )

    for fname in _ONESPIKE_ARRAY_FIELDS:
        v1 = getattr(orig, fname, None)
        v2 = getattr(fast, fname, None)
        if v1 is None and v2 is None:
            continue
        if v1 is None or v2 is None:
            diffs.append(
                f"  tr={trace} spk={spk}  {fname:30s}  "
                f"orig={'None' if v1 is None else 'array'}  "
                f"fast={'None' if v2 is None else 'array'}  (one is None)"
            )
            continue
        a1, a2 = np.asarray(v1), np.asarray(v2)
        if a1.shape != a2.shape:
            diffs.append(
                f"  tr={trace} spk={spk}  {fname:30s}  "
                f"shape mismatch: orig={a1.shape}  fast={a2.shape}"
            )
        elif not np.allclose(a1, a2, rtol=0, atol=atol, equal_nan=True):
            max_dev = np.nanmax(np.abs(a1 - a2))
            diffs.append(
                f"  tr={trace} spk={spk}  {fname:30s}  "
                f"max delta={max_dev:.2e}"
            )
    return diffs


def compare_spike_shapes(shapes_orig: dict, shapes_fast: dict,
                         atol: float = 1e-10) -> list:
    """Compare two spikeShapes dicts; return list of discrepancy strings."""
    all_diffs = []

    traces_orig = set(shapes_orig)
    traces_fast = set(shapes_fast)
    for tr in sorted(traces_orig - traces_fast):
        all_diffs.append(f"  trace {tr}: present in original, missing in fast")
    for tr in sorted(traces_fast - traces_orig):
        all_diffs.append(f"  trace {tr}: present in fast, missing in original")

    for tr in sorted(traces_orig & traces_fast):
        spks_orig = shapes_orig[tr]
        spks_fast = shapes_fast[tr]
        s_orig = set(spks_orig)
        s_fast = set(spks_fast)
        for spk in sorted(s_orig - s_fast):
            all_diffs.append(f"  tr={tr}  spike {spk}: in original, missing in fast")
        for spk in sorted(s_fast - s_orig):
            all_diffs.append(f"  tr={tr}  spike {spk}: in fast, missing in original")
        for spk in sorted(s_orig & s_fast):
            all_diffs.extend(
                compare_one_spike(spks_orig[spk], spks_fast[spk], tr, spk, atol=atol)
            )
    return all_diffs


# ---------------------------------------------------------------------------
# Main comparison routine (CLI entry point)
# ---------------------------------------------------------------------------

def run_comparison_test(data_arg: Optional[str] = None,
                        method: str = "Kalluri",
                        max_spike_shape: int = 5,
                        plot: bool = False,
                        verbose: bool = False,
                        atol: float = 1e-10,
                        n_workers: int = 1) -> int:
    """Run original and fast analyzeSpikeShape, compare and report results.

    Returns exit code: 0 = no discrepancies (or fast not available), 1 = diffs found.
    """
    clamps, source_label = load_data(data_arg)
    print(f"\nData source : {source_label}")
    print(f"Method      : {method}")
    print(f"Max spikes  : {max_spike_shape}")
    print(f"tstart/tend : {clamps.tstart:.4f} s / {clamps.tend:.4f} s")
    print(f"Traces      : {len(clamps.traces)}")
    print(f"Workers     : {n_workers}")
    print(f"Plot flag     : {plot}")
    print("-" * 60)

    # --- original -----------------------------------------------------------
    print("Running original analyzeSpikeShape …", flush=True)
    try:
        shapes_orig, t_orig = run_original(clamps, method=method,
                                           max_spike_shape=max_spike_shape,
                                           verbose=verbose)
    except Exception:
        print("\n*** run_original failed — see traceback above ***", flush=True)
        return 2
    n_spikes_orig = sum(len(v) for v in shapes_orig.values())
    print(f"  Done in {t_orig:.4f} s  |  {len(shapes_orig)} traces with spikes"
          f"  |  {n_spikes_orig} spikes analysed", flush=True)

    # --- fast ---------------------------------------------------------------
    t_fast: Optional[float] = None
    shapes_fast: Optional[dict] = None

    if not HAS_FAST:
        print(
            "  Fast module (spike_analysis_fast) not found — "
            "implement ephys/ephys_analysis/spike_analysis_fast.py to enable comparison."
        )
    else:
        print("Running fast analyzeSpikeShape_fast …", flush=True)
        shapes_fast, t_fast = run_fast(clamps, method=method,
                                       max_spike_shape=max_spike_shape,
                                       verbose=verbose,
                                       n_workers=n_workers)
        n_spikes_fast = sum(len(v) for v in shapes_fast.values())
        print(f"  Done in {t_fast:.4f} s  |  {len(shapes_fast)} traces with spikes"
              f"  |  {n_spikes_fast} spikes analysed")

    # --- timing summary -----------------------------------------------------
    print("\n" + "=" * 60, flush=True)
    print("TIMING SUMMARY", flush=True)
    print("=" * 60, flush=True)
    if t_fast is not None and t_fast > 0:
        speedup = t_orig / t_fast
        print(f"  original : {t_orig:.4f} s", flush=True)
        print(f"  fast     : {t_fast:.4f} s", flush=True)
        print(f"  speedup  : {speedup:.2f}x", flush=True)
    else:
        fast_str = f"{t_fast:.4f} s" if t_fast is not None else "N/A (module not built)"
        print(f"  original : {t_orig:.4f} s", flush=True)
        print(f"  fast     : {fast_str}", flush=True)
    print("=" * 60, flush=True)

    # --- correctness comparison ---------------------------------------------
    print("-" * 60)

    def _do_plot():
        try:
            _maybe_plot(clamps, shapes_orig, plot)
        except Exception:
            print("\n*** _maybe_plot raised an exception ***", flush=True)
            traceback.print_exc()

    if shapes_fast is None:
        print("Correctness comparison skipped (fast module not available).")
        _do_plot()
        return 0

    diffs = compare_spike_shapes(shapes_orig, shapes_fast, atol=atol)
    if diffs:
        print(f"DISCREPANCIES FOUND ({len(diffs)}):")
        for d in diffs:
            print(d)
        _do_plot()
        return 1
    else:
        print("All spike attributes match within tolerance "
              f"(atol={atol:.1e}).")
        _do_plot()
        return 0


def _maybe_plot(clamps: HHIV, spike_shapes: dict, do_plot: bool):
    print("_maybe_plot called with do_plot: ", do_plot)
    if not do_plot:
        return

    print("\nPlotting traces with detected spikes marked …", flush=True)
    traces = clamps.traces
    time_base = np.asarray(clamps.time_base)
    n_traces = len(traces)
    print(f"\n[plot] {n_traces} traces, time_base shape={time_base.shape}, "
          f"traces type={type(traces).__name__}", flush=True)

    fig, ax = mpl.subplots(1, 1)
    for i in range(n_traces):
        tr = np.asarray(traces[i]).ravel()  # ensure 1-D
        try:
            (line,) = ax.plot(time_base, tr, linewidth=0.33)
        except Exception as exc:
            print(f"  [plot] trace {i}: ax.plot failed — {exc}", flush=True)
            continue
        if i in spike_shapes:
            sh = spike_shapes[i]
            for j in sh:
                ax.plot(sh[j].peak_T, sh[j].peak_V, "o", color="red", markersize=3)
                if sh[j].trough_T is not None:
                    ax.plot(sh[j].trough_T, sh[j].trough_V, "^",
                            color=line.get_color(), markersize=3)
                if sh[j].left_halfwidth_T is not None:
                    ax.plot(
                        [sh[j].left_halfwidth_T, sh[j].right_halfwidth_T],
                        [sh[j].left_halfwidth_V, sh[j].right_halfwidth_V],
                        color=line.get_color(), marker="o", markersize=1,
                        linestyle="-", linewidth=1,
                    )
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Voltage (V)")
    ax.set_title("Spike analysis — original")
    print("[plot] calling mpl.show() …", flush=True)
    mpl.show(block=True)


# ===========================================================================
# Existing pytest regression tests (unchanged in behaviour)
# ===========================================================================

def printPars(pars):
    print(dir(pars))
    d = dataclasses.asdict(pars)
    for k in d.keys():
        print("k: ", k, " = ", d[k])


def run_spike_tester(method: str = "Kalluri", plot: bool = False) -> dict:
    """Run the original spike tester and optionally plot. Returns spikeShapes."""
    testdata = get_testdata()
    spike_analyzer = SA.SpikeAnalysis()
    spike_analyzer.setup(
        experiment=MINIMAL_EXPERIMENT,
        clamps=testdata,
        threshold=-0.020,
        verbose=True,
    )
    spike_analyzer.set_detector(method, pars=None)

    t_spikes = time.perf_counter()
    spike_analyzer.analyzeSpikes()
    t_spikes = time.perf_counter() - t_spikes

    t_shape = time.perf_counter()
    spike_analyzer.analyzeSpikeShape(max_spike_shape=3, printSpikeInfo=False)
    t_shape = time.perf_counter() - t_shape

    spksh = spike_analyzer.spikeShapes
    n_spikes = sum(len(v) for v in spksh.values())
    print(
        f"\n>>> Timing [{method}]: "
        f"analyzeSpikes={t_spikes:.4f} s  "
        f"analyzeSpikeShape={t_shape:.4f} s  "
        f"total={t_spikes + t_shape:.4f} s  "
        f"| {len(spksh)} traces with spikes, {n_spikes} spikes",
        flush=True,
    )

    if plot:
        import matplotlib.pyplot as mpl
        tr_line = []
        for i in range(len(testdata.traces)):
            tr_line.append(mpl.plot(testdata.time_base, testdata.traces[i], linewidth=0.33))
            if i in spksh.keys():
                sh = spksh[i]
                for j in sh.keys():
                    l_color = tr_line[i][0].get_color()
                    mpl.plot(sh[j].peak_T, sh[j].peak_V, "o", color="red", markersize=3)
                    mpl.plot(sh[j].trough_T, sh[j].trough_V, "^",
                             color=l_color, markersize=3)
                    mpl.plot(
                        [sh[j].left_halfwidth_T, sh[j].right_halfwidth_T],
                        [sh[j].left_halfwidth_V, sh[j].right_halfwidth_V],
                        color=l_color, marker="o", markersize=1,
                        linestyle="-", linewidth=1,
                    )
        mpl.show()
    return spksh


class SpikeTester(UserTester):
    def __init__(self, method: str = "Kalluri", extra=None, plot: bool = False):  # noqa: ARG002
        self.TM = None
        self.figure = None
        UserTester.__init__(self, "%s_%s" % (method, "spikeshape"), method)

    def run_test(self, method, plot: bool = False):
        test_result = run_spike_tester(method=method, plot=plot)
        if isinstance(test_result, dict) and "figure" in test_result:
            self.figure = test_result["figure"]
        return test_result

    def assert_test_info(self, *args, **kwds):
        try:
            super(SpikeTester, self).assert_test_info(*args, **kwds)
        finally:
            if self.figure is not None:
                del self.figure


def test_spikes_Kalluri():
    SpikeTester(method="Kalluri", plot=False)


def test_spikes_argrelmax():
    SpikeTester(method="argrelmax")


def test_spikes_find_peaks():
    SpikeTester(method="find_peaks")


def test_spikes_threshold():
    SpikeTester(method="threshold")


# ===========================================================================
# CLI entry point
# ===========================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare original and fast spike shape analysis.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--data",
        default=None,
        metavar="PATH",
        help=(
            "Path to data: HHData.pkl (default), a .pkl file, or an acq4 "
            "protocol directory (e.g. /data/2024.01.15_000/slice_000/cell_000/CCIV_short_000)"
        ),
    )
    parser.add_argument(
        "--method",
        default="Kalluri",
        choices=["Kalluri", "argrelmax", "find_peaks", "threshold", "all"],
        help="Spike detection method, or 'all' to run every method (default: Kalluri)",
    )
    parser.add_argument(
        "--max-spikes",
        type=int,
        default=5,
        dest="max_spikes",
        metavar="N",
        help="Max spikes per trace for detailed shape analysis (default: 5)",
    )
    parser.add_argument(
        "--atol",
        type=float,
        default=1e-10,
        help="Absolute tolerance for float comparisons (default: 1e-10)",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        default=False,
        help="Show trace + spike-marker plot after analysis",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="Enable verbose output from SpikeAnalysis",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        metavar="N",
        help="Number of worker threads for fast analysis (default: 1 = sequential)",
    )

    args = parser.parse_args()

    ALL_METHODS = ["Kalluri", "argrelmax", "find_peaks", "threshold"]
    methods_to_run = ALL_METHODS if args.method == "all" else [args.method]

    overall_exit = 0
    for meth in methods_to_run:
        if len(methods_to_run) > 1:
            print(f"\n{'='*60}")
            print(f"METHOD: {meth}")
            print(f"{'='*60}")
        rc = run_comparison_test(
            data_arg=args.data,
            method=meth,
            max_spike_shape=args.max_spikes,
            plot=args.plot and len(methods_to_run) == 1,  # only plot for single-method runs
            verbose=args.verbose,
            atol=args.atol,
            n_workers=args.workers,
        )
        overall_exit = max(overall_exit, rc)
    if args.plot:
        mpl.show(block=True)

    sys.exit(overall_exit)
