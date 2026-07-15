# -*- coding: utf-8 -*-
"""
pytest tests for spike_analysis_fast.

Verifies that analyzeSpikeShape_fast produces results identical to the
original analyzeSpikeShape (within atol=1e-10) for every supported spike
detector, and that thread-parallel execution matches sequential execution
for the Kalluri detector.

Run as part of the full suite:
    python test.py

Or directly:
    pytest ephys/ephys_analysis/tests/test_spike_analysis_fast.py -v

Do NOT run with --audit; audit mode is for human use only.
"""

import sys
from pathlib import Path

import pytest

# Reuse helpers from test_spikes rather than duplicating them.
from ephys.ephys_analysis.tests.test_spikes import (
    HAS_FAST,
    compare_spike_shapes,
    get_testdata,
    run_fast,
    run_original,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
ATOL = 1e-10
MAX_SPIKE_SHAPE = 5
METHODS = ["Kalluri", "argrelmax", "find_peaks", "threshold"]
WORKER_COUNTS = [1, 2, 4]

pytestmark = pytest.mark.skipif(
    not HAS_FAST, reason="spike_analysis_fast module not available"
)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def clamps():
    """Load HHData.pkl once for the whole module."""
    return get_testdata()


# ---------------------------------------------------------------------------
# Tests: fast vs original
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("method", METHODS)
def test_fast_matches_original(clamps, method):
    """analyzeSpikeShape_fast must match original for every spike detector."""
    shapes_orig, _ = run_original(clamps, method=method, max_spike_shape=MAX_SPIKE_SHAPE)
    shapes_fast, _ = run_fast(clamps, method=method, max_spike_shape=MAX_SPIKE_SHAPE, n_workers=1)
    diffs = compare_spike_shapes(shapes_orig, shapes_fast, atol=ATOL)
    assert len(diffs) == 0, (
        f"[{method}] {len(diffs)} discrepancy(ies) between original and fast:\n"
        + "\n".join(diffs)
    )


# ---------------------------------------------------------------------------
# Tests: parallel vs sequential
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n_workers", WORKER_COUNTS)
def test_parallel_matches_sequential(clamps, n_workers):
    """Thread-parallel fast analysis must match sequential fast analysis."""
    shapes_seq, _ = run_fast(clamps, method="Kalluri", max_spike_shape=MAX_SPIKE_SHAPE, n_workers=1)
    shapes_par, _ = run_fast(clamps, method="Kalluri", max_spike_shape=MAX_SPIKE_SHAPE, n_workers=n_workers)
    diffs = compare_spike_shapes(shapes_seq, shapes_par, atol=ATOL)
    assert len(diffs) == 0, (
        f"[workers={n_workers}] {len(diffs)} discrepancy(ies) between sequential and parallel:\n"
        + "\n".join(diffs)
    )


# ---------------------------------------------------------------------------
# CLI entry point (human use only — never called by automated hooks)
# ---------------------------------------------------------------------------

def _run_all(verbose: bool = False):
    """Run all comparisons and print a summary. Not for automated use."""
    clamps = get_testdata()
    all_pass = True
    for method in METHODS:
        shapes_orig, t_orig = run_original(clamps, method=method, max_spike_shape=MAX_SPIKE_SHAPE)
        shapes_fast, t_fast = run_fast(clamps, method=method, max_spike_shape=MAX_SPIKE_SHAPE, n_workers=1)
        diffs = compare_spike_shapes(shapes_orig, shapes_fast, atol=ATOL)
        status = "PASS" if not diffs else f"FAIL ({len(diffs)} diffs)"
        print(f"  {method:15s}  orig={t_orig:.4f}s  fast={t_fast:.4f}s  "
              f"speedup={t_orig/t_fast:.2f}x  {status}")
        if diffs and verbose:
            for d in diffs:
                print(d)
        all_pass = all_pass and not diffs

    print()
    for n_workers in WORKER_COUNTS:
        shapes_seq, t_seq = run_fast(clamps, method="Kalluri", max_spike_shape=MAX_SPIKE_SHAPE, n_workers=1)
        shapes_par, t_par = run_fast(clamps, method="Kalluri", max_spike_shape=MAX_SPIKE_SHAPE, n_workers=n_workers)
        diffs = compare_spike_shapes(shapes_seq, shapes_par, atol=ATOL)
        status = "PASS" if not diffs else f"FAIL ({len(diffs)} diffs)"
        print(f"  workers={n_workers}  seq={t_seq:.4f}s  par={t_par:.4f}s  {status}")
        if diffs and verbose:
            for d in diffs:
                print(d)
        all_pass = all_pass and not diffs

    return 0 if all_pass else 1


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Compare spike_analysis_fast against original (human use only)."
    )
    parser.add_argument("--verbose", action="store_true", help="Print all discrepancy details")
    args = parser.parse_args()

    print("spike_analysis_fast comparison")
    print("=" * 60)
    rc = _run_all(verbose=args.verbose)
    print("=" * 60)
    print("Result:", "ALL PASS" if rc == 0 else "FAILURES FOUND")
    sys.exit(rc)
