# Missing test coverage — data analysis modules
# Generated 2026-06-13
# Updated 2026-09-30: six suites added; file reorganized (to-do on top, done at bottom).

Scope: analysis code only.
Excluded: gui/, plotters/, tools/, archive/archived directories.
Existing test directories: ephys/ephys_analysis/tests/, ephys/mini_analyses/tests/,
                           ephys/mapanalysistools/tests/ (added 2026-09-30)

Priority scale: HIGH = no test at all, core computation.
               MEDIUM = partial/internal test exists but not pytest-discoverable.
               LOW = hard to automate (needs live data) or low analytical risk.

Status legend: [DONE] covered by a pytest suite · [TODO] not yet covered.

---

## Summary — priority order

To do (highest analytical value first):
  1. compute_scores         — [TODO] pure math, ~1-hour effort
  2. psc_analysis/functions — [TODO] pure math, no data needed  [HIGH]
  3. psc_analyzer           — [TODO] needs synthetic PSC fixture
  4. vc_analysis            — [TODO] requires synthetic VC fixture first
  5. analyze_ppf / analyze_train / analyze_io — [TODO] LOW, small synthetic data
  6. map_event_scoring      — [TODO] LOW, deferred until compute_scores + shuffler
  7. datareaders            — [TODO] informational only (binary-format readers)

Completed 2026-09-30:
  - rm_tau_analysis     — test_rm_tau.py
  - adaptation_indices  — test_adaptation_indices.py
  - burst_analysis      — test_burst_analysis.py
  - poisson_score       — test_poisson_score.py
  - shuffler            — test_shuffler.py
  - make_clamps         — test_make_clamps.py

---
---

# ========================= TO DO — not yet covered =========================

## ephys/ephys_analysis/

### vc_analysis.py — VCAnalysis / VCSummary  [LOW -> MEDIUM with fixture]  [TODO]
test_vc_analysis.py exists but is marked @pytest.mark.skip because it requires
a live acq4 data directory.

To make this testable without live data:
  - Add a synthetic VC fixture to hh_sim.py (simple RC circuit, known
    time constant and current amplitude).
  - Upgrade test_vc_analysis.py to use it for the core analysis path.
  - Keep the live-data test as a separate skipif, gated on the data path
    existing.

---

## ephys/psc_analysis/   [NO TESTS — entire package]  [TODO]

### functions.py — mean_I_analysis, slope_subtraction, get_traces  [HIGH]  [TODO]
Pure analysis functions; no test at all; no live data required.

Suggested additions to test_psc_analyzer.py or test_psc_functions.py:
  - mean_I_analysis: known waveform → mean and SD match numpy ground truth
  - slope_subtraction: linear ramp trace → residual is near-zero after
    subtraction; mode="mean" and mode="linear" both tested
  - get_traces: returns correct shape given a synthetic multi-trace array

### psc_analyzer.py — PSCAnalyzer  [MEDIUM]  [TODO]
A `test()` function exists at module level (not pytest).
PSCAnalyzer wraps peak detection and fitting on synaptic current traces.

Suggested test file: ephys/psc_analysis/tests/test_psc_analyzer.py
What to test:
  - Synthetic double-exponential PSC (known tau_rise, tau_decay) → recovered
    parameters within 10% of ground truth
  - Empty trace (all zeros) → no events detected, no exception
  - Very noisy trace (high-amplitude noise) → detector does not crash

### analyze_ppf.py — analyze_PPF  [LOW]  [TODO]
Paired-pulse facilitation ratio computation.
Needs two-pulse synthetic data; straightforward to construct.
  - Two identical pulses → PPF ratio ≈ 1.0
  - Second pulse larger → PPF ratio > 1.0

### analyze_train.py — analyze_Train  [LOW]  [TODO]
Synaptic depression/facilitation across a stimulus train.
  - Perfectly facilitating synthetic train → monotonically increasing amplitudes
  - Perfectly depressing train → monotonically decreasing amplitudes

### analyze_io.py — analyze_IO  [LOW]  [TODO]
Input-output curve extraction.
  - Monotonically increasing amplitude vs. stimulus intensity → positive slope

---

## ephys/mapanalysistools/

### compute_scores.py — ZScore, ZScore2D, grand_mean_std, Imax  [HIGH]  [TODO]
Pure mathematical functions; no data dependency. Easiest tests in the project.
No direct test exists.  (ZScore is exercised indirectly via test_shuffler.py, but
a direct suite is still wanted.)

Suggested test file: ephys/mapanalysistools/tests/test_compute_scores.py
What to test:
  - ZScore: input equal to mean → z = 0; input 1 SD above mean → z = 1
  - ZScore2D: same invariants along both axes
  - grand_mean_std: all-zeros array → mean=0, std=0; known array → matches
    numpy ground truth restricted to the specified window
  - Imax: returns max amplitude in the correct time window; empty window → nan

### map_event_scoring.py — MapEventScoring  [LOW]  [TODO]
Complex integration class that wraps shuffler + mini_analysis.
Requires coordinated synthetic data (event times + map geometry).
Best deferred until compute_scores and shuffler are covered.

---

## ephys/datareaders/   [informational — hard to unit-test]  [TODO]

### acq4_reader.py, abf_reader.py, datac_reader.py  [LOW]
These read proprietary binary formats; unit testing requires either
real data files or synthetic byte-level fixtures.  Too fragile and
data-dependent for automated CI; skip for now.

---
---

# ======================= COMPLETED — covered by pytest =======================

## ephys/ephys_analysis/

### rm_tau_analysis.py — RmTauAnalysis  [HIGH]  [DONE 2026-09-30]
The most important untested analysis class. Computes Rm, tau_m, RMP, and
peak/steady-state IV curves from current-clamp data.

Test file: ephys/ephys_analysis/tests/test_rm_tau.py  (9 tests)
The HHData.pkl fixture is wrapped into a real acq4-style Clamps via MakeClamps
(so the MetaArray "Time" slicing works), then run through SpikeAnalysis and
RmTauAnalysis. Expected ranges were calibrated to actual output on this fixture
(RMP ~ -59.6 mV, Rin ~ 17 MOhm, tau_m ~ 12.8 ms).
What is tested:
  - [x] RmTauAnalysis.analyze() runs without exception on HHData
  - [x] Computed RMP is in a physiological range (-80 to -50 mV)
  - [x] Computed Rin is positive and plausible
        (NOTE: the HH cell is ~17 MOhm, not the 50-500 MOhm guessed originally;
         the test uses a broad 1-1000 MOhm bound)
  - [x] tau_membrane() returns a positive time constant
  - [x] rmp_analysis() / ivss_analysis() / ivpk_analysis() return consistent shapes
        (plus rmp_analysis matches a direct baseline-window mean)
  - [x] bridge_adjust() does not alter results when bridge resistance is zero
  - [x] Results are stable across repeated calls (no accumulated state)
Source touched: spike_analysis.py np.array(v_raw) -> v_raw.view(np.ndarray).copy()
  (2026-09-30) to clear a numpy-2.0 MetaArray.__array__ DeprecationWarning that
  surfaced while running this suite.

---

### adaptation_indices.py  [HIGH]  [DONE 2026-09-30]
Adaptation-index functions operating purely on spike-time arrays.
The module `tests()` needs allensdk (unavailable), so synthetic trains are used.

Test file: ephys/ephys_analysis/tests/test_adaptation_indices.py  (23 tests)
What is tested:
  - [x] adaptation_index_MKX2019: regular train -> ~0; decelerating (adapting,
        spikes early) -> index > 0; accelerating -> index < 0
        (NOTE: this is the OPPOSITE of the sign convention originally written
         here; corrected to match the implementation / docstring)
  - [x] adaptation_index_AllenInst: same sign convention as MKX2019; input="isis"
        matches input="spike_latencies"; invalid input raises AssertionError
  - [x] adaptation_index_AllenInst_spktimes: finite for a valid train; honours
        window trimming and minimum_spikes
        (NOTE: it is NOT equivalent to AllenInst — it takes norm-diffs of spike
         *times*, not ISIs — so equivalence is not asserted)
  - [x] adaptation_first_last_isi: ratio > 1 for adapting, < 1 for accelerating;
        too few spikes -> NaN
  - [x] adaptation_index_exponential: returns a real scalar without raising;
        too few spikes -> NaN
        (NOTE: it returns an adaptation index, not a fit R^2, so the original
         "R^2 near 1.0" check does not apply)
  - [x] compute_adaptation_index: dispatcher matches the named sub-function for
        MKX2019 / AllenInst / FirstLastISI; respects rate_bounds and minimum_spikes
  - [x] All functions handle fewer spikes than minimum_spikes without exception
Source touched: adaptation_indices.py:304 fixed a numpy-array Python-`and` bug
  (ValueError) so the dispatcher runs (2026-09-30).

---

### burst_analysis.py — BurstAnalyzer  [MEDIUM]  [DONE 2026-09-30]
Detects bursts in per-trial spike-time arrays; no live data needed.

Test file: ephys/ephys_analysis/tests/test_burst_analysis.py  (11 tests)
What is tested:
  - [x] Perfectly isolated spikes (large ISI) -> zero bursts
  - [x] Tight clusters (ISI < max_ISI) -> one burst per cluster; two clusters -> 2
  - [x] burst_maker() returns one ndarray per trial and feeds BurstAnalyzer
  - [x] calculate_evolution() returns a dict
        (NOTE: keys are "by_trial"/"by_time", not "trial"/"time"); empty when no bursts
  - [x] trial_metrics() returns per-trial counts consistent with self.bursts
  - [x] min_spikes threshold: clusters below threshold not classified as bursts
  - [x] mismatched trials / trial_start_times -> ValueError

---

### poisson_score.py — PoissonScore family  [MEDIUM]  [DONE 2026-09-30]
Pure statistical scoring; no live data required.

Test file: ephys/ephys_analysis/tests/test_poisson_score.py  (18 tests)
What is tested:
  - [x] poissonProb(n=0, t=1.0, l=0.0) -> 1.0; (n>0, l=0) -> 1e-25
  - [x] poissonProb analytic value: sf(0) for mean 1 == 1 - e^-1; matches
        scipy.stats.poisson(l*t).sf(n)
        (NOTE: poissonProb is the SURVIVAL function P(X>n); the original
         "P(n=1,t=1,lambda=1) ~ 0.368" was incorrect — sf(1)=0.264)
  - [x] poissonProcess: length == n; times increasing; respects tmax
  - [x] gaussProb: 0.5 at the mean, 1.0 for empty input, monotone in amplitude
  - [x] PoissonScore.score(): higher background rate -> lower raw score; empty
        events -> 1.0; deterministic; evoked cluster -> score > 1
  - [x] PoissonRepeatScore.score(): finite/positive and deterministic for
        identical trials; empty -> 1.0
  (Both score() paths use normalize=False to avoid generating/caching the heavy
   Monte-Carlo normalization table.)

---

### make_clamps.py — MakeClamps  [MEDIUM]  [DONE 2026-09-30]
Helper that constructs synthetic Clamps objects from arrays.

Test file: ephys/ephys_analysis/tests/test_make_clamps.py  (9 tests)
What is tested:
  - [x] flat trace -> time_base length matches traces length; getData() -> True
  - [x] tstart/tend/tdur and sample_interval set from constructor args
  - [x] values / commandLevels match the command step levels; holding ~ 0
  - [x] traces support MetaArray ["Time":t0:t1] slicing
  - [x] no data set -> ValueError; setProtocol() updates the protocol

---

## ephys/mapanalysistools/

### shuffler.py — Shuffler  [MEDIUM]  [DONE 2026-09-30]
Internal `test_events_shuffle()` / `test_events_poisson()` methods exist but are
not pytest-discoverable.

Test file: ephys/mapanalysistools/tests/test_shuffler.py  (8 tests)
What is tested:
  - [x] z2p: 0 -> 0.5; +/-1.96 -> 0.975 / 0.025; symmetric
  - [x] compute_ZScore: returns one finite z per trace; traces with a dense
        in-window cluster score much higher than spontaneous-only traces (some
        reach z > 1.96)
        (NOTE: the "zero events -> graceful" case is NOT graceful — with no
         baseline events pre_std=0 and z collapses to 0 — so it is not asserted)
  - [x] _make_test_events / generate_example_traces: map has the requested
        number of spots; active-site count within [0, nspots]; zero spontaneous
        rate still builds a map
  (shuffle_score was left uncovered: its input format and internal randomness
   make a stable p-value assertion fragile.)
