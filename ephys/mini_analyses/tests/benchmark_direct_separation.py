"""
benchmark_direct_separation.py

Benchmark harness for optional direct/synaptic response separation
(ephys.mini_analyses.direct_response_separation). This is NOT a pass/fail
pytest -- it builds synthetic ground-truth data and reports recovery-rate
metrics, both as a standalone diagnostic (run as a script) and as the
scoring function for the threshold/parameter tuning pass (see the plan's
Verification step 2). Not collected by pytest (file name does not start
with test_).

Benchmark design:
1. Baseline -- detect on synaptic PSCs only (no direct response). This sets
   the achievable recall reference: even on uncontaminated data, threshold
   detection does not recover 100% of seeded events, so later comparisons
   are made against this reference, not against the raw seeded count.
2. Sweep -- take the exact same synaptic event set from step 1 and add a
   slow "direct" response starting 1 ms after the stimulus, with amplitude
   swept from 0 to 5x the nominal mean PSC amplitude, in both the same sign
   and opposite sign relative to the synaptic events. For each condition,
   detect twice (without and with direct-response separation enabled) and
   report PSC recall relative to the step-1 baseline, plus how well the
   direct response peak itself is recovered when separation is used.
"""
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

import ephys.mini_analyses.mini_event_dataclasses as MEDC
import ephys.mini_analyses.minis_methods as MM
import ephys.mini_analyses.direct_response_separation as DRS


@dataclass
class BenchPars:
    dt_seconds: float = 2e-5
    maxt: float = 0.5  # seconds, trace duration
    stim_time: float = 0.05  # seconds
    # --- spontaneous synaptic PSCs (Poisson over the whole trace) ---
    mean_rate: float = 40.0  # Hz, background Poisson spontaneous rate
    mean_amp: float = 20e-12  # A, mean PSC amplitude
    amp_sigma: float = 4e-12  # A
    noise_amp: float = 0.3e-12  # A, gaussian background noise
    sign: int = -1  # EPSC-like (negative-going)
    tau1: float = 5e-4  # PSC rise tau, s
    tau2: float = 2e-3  # PSC decay tau, s
    # --- stimulus-evoked "response" PSCs (inhomogeneous Poisson post-stim) ---
    p_evoked: float = 0.3  # probability a trace has evoked responses
    evoked_start: float = 0.005  # s after stimulus: earliest evoked response
    evoked_rate_tau: float = 0.020  # s: time constant of the decaying evoked rate
    evoked_mean_count: float = 3.0  # expected # evoked events in the count window
    evoked_count_window: float = 0.015  # s: window (from evoked_start) the count refers to (5-20 ms)
    # --- slow direct (non-synaptic) response ---
    p_direct: float = 0.3  # probability a trace has a direct response
    direct_amp_range: tuple = (0.0, 5.0)  # x mean PSC amplitude, uniform
    direct_sign_choices: tuple = (1, -1)  # sampled per direct trace; +1 = same sign as PSCs
    # (negative, EPSC-like), -1 = opposite. Set to (1,) for all same-sign directs.
    direct_delay: float = 0.001  # seconds after stimulus
    direct_tau_rise: float = 0.005  # s
    direct_tau_decay: float = 0.050  # s
    # --- detection ---
    threshold: float = 5.0  # detector threshold, SD units
    # --- population size / RNG ---
    n_trials: int = 50
    seed: int = 12345

    multitrace: bool = True


def _psc_template(t_local: np.ndarray, tau1: float, tau2: float, sign: int) -> np.ndarray:
    """Normalized double-exponential PSC shape, peak amplitude = sign * 1."""
    aprime = (tau2 / tau1) ** (tau1 / (tau1 - tau2))
    g = aprime * (-np.exp(-t_local / tau1) + np.exp(-t_local / tau2))
    return sign * g / np.max(np.abs(g))


def spontaneous_onsets(pars: BenchPars, rng) -> np.ndarray:
    """Poisson spontaneous PSC onset times over the whole trace."""
    intervals = rng.exponential(1.0 / pars.mean_rate, int(pars.maxt * pars.mean_rate * 3) + 10)
    onset_times = np.cumsum(intervals)
    return onset_times[onset_times < pars.maxt - 5.0 * pars.tau2]


def evoked_onsets(pars: BenchPars, rng) -> np.ndarray:
    """
    Stimulus-evoked "response" PSC onset times: an inhomogeneous Poisson
    process with rate lambda(t) = lambda0 * exp(-(t - t0) / tau), starting at
    t0 = stim_time + evoked_start, with lambda0 chosen so the expected count
    in [t0, t0 + evoked_count_window] equals evoked_mean_count. Event times
    are t0 + Exponential(tau) (exponentially-decreasing frequency, Poisson in
    time; events may partially overlap). Returns onset times (absolute s).
    """
    t0 = pars.stim_time + pars.evoked_start
    tau = pars.evoked_rate_tau
    win = pars.evoked_count_window
    # lambda0 from the expected count in the count window
    lam0 = pars.evoked_mean_count / (tau * (1.0 - np.exp(-win / tau)))
    t_end = pars.maxt - 5.0 * pars.tau2
    total_span = max(0.0, t_end - t0)
    expected_total = lam0 * tau * (1.0 - np.exp(-total_span / tau))
    n = rng.poisson(expected_total)
    if n == 0:
        return np.array([])
    times = t0 + rng.exponential(tau, n)
    return np.sort(times[times < t_end])


def _add_pscs(trace, onset_times, pars, rng):
    """Add PSC waveforms at onset_times to trace in place; return (onsets, amps) actually placed."""
    t_psc = np.arange(0.0, 5.0 * pars.tau2, pars.dt_seconds)
    template = _psc_template(t_psc, pars.tau1, pars.tau2, pars.sign)
    n = len(trace)
    placed_onsets, placed_amps = [], []
    for onset_t in onset_times:
        i0 = int(onset_t / pars.dt_seconds)
        if i0 < 0 or i0 + len(template) >= n:
            continue
        amp = pars.mean_amp * rng.normal(1.0, pars.amp_sigma / pars.mean_amp)
        trace[i0 : i0 + len(template)] += amp * template
        placed_onsets.append(onset_t)
        placed_amps.append(pars.sign * amp)
    return placed_onsets, placed_amps


def make_synaptic_only_trial(pars: BenchPars, rng=None, with_evoked: bool = False) -> dict:
    """
    Build one synthetic trace of synaptic PSCs (no direct response): always
    Poisson-distributed spontaneous PSCs, plus (if with_evoked) a burst of
    stimulus-evoked response PSCs. Gaussian noise added last. Returns
    dict(t, trace, true_onsets, true_amps, evoked_onsets, has_evoked) where
    true_onsets/true_amps cover ALL synaptic events (spontaneous + evoked) --
    the events we want to recover after direct-response subtraction.
    """
    if rng is None:
        rng = np.random.default_rng(pars.seed)
    t = np.arange(0.0, pars.maxt, pars.dt_seconds)
    trace = np.zeros(len(t))

    spont = spontaneous_onsets(pars, rng)
    on_s, amp_s = _add_pscs(trace, spont, pars, rng)

    evk = evoked_onsets(pars, rng) if with_evoked else np.array([])
    on_e, amp_e = _add_pscs(trace, evk, pars, rng)

    trace += rng.normal(0.0, pars.noise_amp, len(t))

    all_onsets = np.array(sorted(on_s + on_e))
    combined = sorted(zip(on_s + on_e, amp_s + amp_e))
    all_amps = np.array([a for _, a in combined]) if combined else np.array([])
    return dict(
        t=t,
        trace=trace,
        true_onsets=all_onsets,
        true_amps=all_amps,
        evoked_onsets=np.array(sorted(on_e)),
        has_evoked=with_evoked and len(on_e) > 0,
    )


def add_direct_response(
    pars: BenchPars, trial: dict, amplitude_factor: float, direct_sign: int
) -> dict:
    """
    Return a copy of `trial` with a slow "direct" response added, starting
    pars.direct_delay after the stimulus. amplitude_factor scales the
    nominal mean PSC amplitude (0 = no direct response, 5 = 5x); direct_sign
    is +1 for same sign as the synaptic events, -1 for opposite sign.
    """
    t = trial["t"]
    trace = trial["trace"].copy()
    onset_t = pars.stim_time + pars.direct_delay
    i0 = int(onset_t / pars.dt_seconds)
    i1 = min(i0 + int(5.0 * pars.direct_tau_decay / pars.dt_seconds), len(trace))

    true_peak = direct_sign * pars.sign * amplitude_factor * pars.mean_amp
    if i1 > i0 and amplitude_factor > 0:
        tt = t[i0:i1] - t[i0]
        shape = (1.0 - np.exp(-tt / pars.direct_tau_rise)) * np.exp(-tt / pars.direct_tau_decay)
        shape_peak = np.max(shape)
        if shape_peak > 0:
            trace[i0:i1] += true_peak * (shape / shape_peak)

    out = dict(trial)
    out["trace"] = trace
    out["direct_window"] = (i0, i1)
    out["direct_true_peak"] = true_peak
    return out


def match_events(
    detected_onsets: Sequence[float], true_onsets: Sequence[float], tolerance_s: float = 0.001
) -> list:
    """
    Greedy nearest-neighbor matching of detected onsets to true onsets within
    tolerance_s. Returns a list of (true_index, detected_index) pairs.
    """
    true = list(true_onsets)
    detected = list(detected_onsets)
    matched_true = set()
    pairs = []
    for j, d in enumerate(detected):
        best_i, best_dt = None, tolerance_s
        for i, tr in enumerate(true):
            if i in matched_true:
                continue
            dt = abs(d - tr)
            if dt <= best_dt:
                best_dt = dt
                best_i = i
        if best_i is not None:
            matched_true.add(best_i)
            pairs.append((best_i, j))
    return pairs


def score_recovery(
    detected_onsets: Sequence[float], true_onsets: Sequence[float], tolerance_s: float = 0.001
) -> dict:
    """
    Recall (fraction of true onsets matched) and false-positive count
    (detected onsets left unmatched), via match_events.
    """
    pairs = match_events(detected_onsets, true_onsets, tolerance_s)
    n_true = len(list(true_onsets))
    n_detected = len(list(detected_onsets))
    n_matched = len(pairs)
    return dict(
        recall=(n_matched / n_true if n_true > 0 else float("nan")),
        n_true=n_true,
        n_matched=n_matched,
        n_detected=n_detected,
        n_false_positive=n_detected - n_matched,
    )


def _detector_kwargs(pars: BenchPars) -> dict:
    filters = MEDC.Filtering()
    filters.LPF_frequency = None
    filters.Notch_frequencies = None
    filters.Detrend_method = None
    return dict(
        dt_seconds=pars.dt_seconds,
        risepower=4.0,
        tau1=pars.tau1,
        tau2=pars.tau2,
        tau3=pars.tau1,
        tau4=pars.tau2,
        template_tmax=5.0 * pars.tau2,
        template_pre_time=0.0,
        delay=0.0,
        sign=pars.sign,
        threshold=pars.threshold,
        filters=filters,
        engine=None,
    )


def detect_once(
    t: np.ndarray,
    trace: np.ndarray,
    pars: BenchPars,
    use_separation: bool,
    direct_sep_pars: Optional[MEDC.DirectSeparationPars] = None,
    stim_time: Optional[float] = None,
) -> dict:
    """
    Run AJ detection once on a single trace. If use_separation is True,
    subtract detected events, fit+subtract the stimulus-locked residual, and
    re-detect via DRS.separate_direct_response (the same orchestrator
    analyze_map_data.py calls in production).

    Returns dict(onsets [s], amplitudes, corrected_trace, records) --
    corrected_trace/records are just the input trace / [] when
    use_separation is False.
    """
    kwargs = _detector_kwargs(pars)
    aj = MM.AndradeJonas()
    aj.setup(ntraces=1, **{k: v for k, v in kwargs.items() if k != "engine"})
    aj.deconvolve(trace.copy(), timebase=t, itrace=0, prepare_data=False)
    aj.identify_events(verbose=False)
    summary = aj.summarize(trace.copy().reshape(1, -1), verbose=False)

    if not use_separation:
        return dict(
            onsets=np.asarray(summary.onsets[0]) * pars.dt_seconds,
            amplitudes=np.asarray(summary.amplitudes[0]),
            corrected_trace=trace,
            records=[],
        )

    corrected, onsets, _peaks, _smpk, _smoothed, amps, records = DRS.separate_direct_response(
        raw_trace=trace,
        t=t,
        dt_seconds=pars.dt_seconds,
        summary=summary,
        itrace=0,
        stim_starts=[stim_time],
        template=aj.template,
        template_amax=aj.template_amax,
        detector_cls=MM.AndradeJonas,
        detector_kwargs=kwargs,
        pars=direct_sep_pars,
    )
    return dict(
        onsets=np.asarray(onsets) * pars.dt_seconds,
        amplitudes=np.asarray(amps),
        corrected_trace=corrected,
        records=records,
    )


def run_benchmark(
    amplitude_factors: Sequence[float] = (0.0, 0.5, 1.0, 2.0, 5.0),
    direct_signs: Sequence[int] = (1, -1),
    tolerance_s: float = 0.001,
    pars: Optional[BenchPars] = None,
) -> dict:
    if pars is None:
        pars = BenchPars()

    # 1. Baseline: synaptic PSCs only, no direct response.
    trial = make_synaptic_only_trial(pars)
    res_baseline = detect_once(trial["t"], trial["trace"], pars, use_separation=False)
    baseline_score = score_recovery(res_baseline["onsets"], trial["true_onsets"], tolerance_s)
    print(
        f"[baseline, no direct response] recall={baseline_score['recall']:.2f} "
        f"({baseline_score['n_matched']}/{baseline_score['n_true']}), "
        f"false_positives={baseline_score['n_false_positive']}"
    )

    # 2. Sweep: add a direct response, compare recovery with/without separation.
    dsp = MEDC.DirectSeparationPars()
    dsp.enabled = True

    results = []
    seen = set()
    for amp_factor in amplitude_factors:
        for direct_sign in direct_signs:
            key = (amp_factor, direct_sign if amp_factor > 0 else 0)
            if key in seen:
                continue  # amplitude 0 is the same regardless of sign
            seen.add(key)

            contaminated = add_direct_response(pars, trial, amp_factor, direct_sign)

            res_nosep = detect_once(
                contaminated["t"], contaminated["trace"], pars, use_separation=False
            )
            score_nosep = score_recovery(res_nosep["onsets"], trial["true_onsets"], tolerance_s)

            res_sep = detect_once(
                contaminated["t"],
                contaminated["trace"],
                pars,
                use_separation=True,
                direct_sep_pars=dsp,
                stim_time=pars.stim_time,
            )
            score_sep = score_recovery(res_sep["onsets"], trial["true_onsets"], tolerance_s)
            n_records = len(res_sep["records"])
            recorded_charge = res_sep["records"][0].charge if n_records else 0.0

            print(
                f"amp={amp_factor:4.1f}x sign={direct_sign:+d}  "
                f"no-sep recall={score_nosep['recall']:.2f}  "
                f"sep recall={score_sep['recall']:.2f}  "
                f"(baseline={baseline_score['recall']:.2f})  "
                f"n_direct_recorded={n_records}  "
                f"true_direct_peak={contaminated['direct_true_peak'] * 1e12:7.1f} pA  "
                f"recorded_charge={recorded_charge * 1e12:7.2f} pA*ms"
            )
            results.append(
                dict(
                    amplitude_factor=amp_factor,
                    direct_sign=direct_sign,
                    baseline_recall=baseline_score["recall"],
                    no_sep_recall=score_nosep["recall"],
                    sep_recall=score_sep["recall"],
                    no_sep_false_positives=score_nosep["n_false_positive"],
                    sep_false_positives=score_sep["n_false_positive"],
                    n_direct_recorded=n_records,
                    true_direct_peak=contaminated["direct_true_peak"],
                )
            )

    return dict(baseline=baseline_score, sweep=results)


def run_population_benchmark(
    tolerance_s: float = 0.001,
    pars: Optional[BenchPars] = None,
    seed: int = 2024,
    n_examples: int = 12,
) -> dict:
    """
    Population benchmark over pars.n_trials independent traces. Each trace
    independently gets stimulus-evoked response PSCs (probability
    pars.p_evoked) and/or a slow direct response (probability pars.p_direct,
    amplitude uniform in pars.direct_amp_range x mean PSC amplitude, random
    sign). Each trace's own synaptic content (spontaneous + evoked, before
    any direct response) is its ground truth. Reports PSC recall split by
    whether the trace has a direct response, with vs without separation, and
    collects example traces (preferring direct+evoked) for plot_examples()
    and the pyqtgraph viewer.
    """
    if pars is None:
        pars = BenchPars()
    rng = np.random.default_rng(seed)
    dsp = MEDC.DirectSeparationPars()
    dsp.enabled = True

    results = []
    example_traces = []

    for i in range(pars.n_trials):
        trial_rng = np.random.default_rng(pars.seed + i)
        with_evoked = bool(trial_rng.random() < pars.p_evoked)
        with_direct = bool(trial_rng.random() < pars.p_direct)

        clean = make_synaptic_only_trial(pars, rng=trial_rng, with_evoked=with_evoked)
        if with_direct:
            amp_factor = float(trial_rng.uniform(*pars.direct_amp_range))
            direct_sign = int(trial_rng.choice([1, -1]))
            contaminated = add_direct_response(pars, clean, amp_factor, direct_sign)
            trace_for_detection = contaminated["trace"]
            true_direct_peak = contaminated["direct_true_peak"]
        else:
            amp_factor, direct_sign, true_direct_peak = 0.0, 0, 0.0
            trace_for_detection = clean["trace"]

        res_nosep = detect_once(clean["t"], trace_for_detection, pars, use_separation=False)
        score_nosep = score_recovery(res_nosep["onsets"], clean["true_onsets"], tolerance_s)

        res_sep = detect_once(
            clean["t"],
            trace_for_detection,
            pars,
            use_separation=True,
            direct_sep_pars=dsp,
            stim_time=pars.stim_time,
        )
        score_sep = score_recovery(res_sep["onsets"], clean["true_onsets"], tolerance_s)

        results.append(
            dict(
                has_direct=with_direct,
                has_evoked=with_evoked,
                amplitude_factor=amp_factor,
                direct_sign=direct_sign,
                true_direct_peak=true_direct_peak,
                recall_nosep=score_nosep["recall"],
                recall_sep=score_sep["recall"],
                fp_nosep=score_nosep["n_false_positive"],
                fp_sep=score_sep["n_false_positive"],
                n_true=score_nosep["n_true"],
                n_direct_recorded=len(res_sep["records"]),
                recorded_charge=res_sep["records"][0].charge if res_sep["records"] else 0.0,
            )
        )

        if with_direct and len(example_traces) < n_examples:
            example_traces.append(
                dict(
                    t=clean["t"],
                    clean_trace=clean["trace"],
                    contaminated_trace=trace_for_detection,
                    corrected_trace=res_sep["corrected_trace"],
                    true_onsets_ms=(np.asarray(clean["true_onsets"]) - pars.stim_time) * 1e3,
                    detected_onsets_nosep_ms=(np.asarray(res_nosep["onsets"]) - pars.stim_time) * 1e3,
                    detected_onsets_sep_ms=(np.asarray(res_sep["onsets"]) - pars.stim_time) * 1e3,
                    amp_factor=amp_factor,
                    direct_sign=direct_sign,
                    true_direct_peak=true_direct_peak,
                    has_evoked=with_evoked,
                    stim_time=pars.stim_time,
                )
            )

    def _mean(xs):
        xs = [x for x in xs if not np.isnan(x)]
        return float(np.mean(xs)) if xs else float("nan")

    direct_results = [r for r in results if r["has_direct"]]
    clean_results = [r for r in results if not r["has_direct"]]
    print(f"n_trials={pars.n_trials}  n_with_direct={len(direct_results)}  n_without={len(clean_results)}")
    print(
        f"[no direct response]   mean recall no-sep={_mean([r['recall_nosep'] for r in clean_results]):.3f}  "
        f"mean recall sep={_mean([r['recall_sep'] for r in clean_results]):.3f}"
    )
    print(
        f"[with direct response] mean recall no-sep={_mean([r['recall_nosep'] for r in direct_results]):.3f}  "
        f"mean recall sep={_mean([r['recall_sep'] for r in direct_results]):.3f}"
    )
    print(
        f"[with direct response] mean false-positives no-sep={_mean([r['fp_nosep'] for r in direct_results]):.2f}  "
        f"sep={_mean([r['fp_sep'] for r in direct_results]):.2f}"
    )
    print(
        f"[with direct response] fraction with a recorded direct response: "
        f"{_mean([1.0 if r['n_direct_recorded'] > 0 else 0.0 for r in direct_results]):.2f}"
    )

    return dict(results=results, example_traces=example_traces)


def plot_examples(
    example_traces: List[dict],
    filename: str = "direct_separation_examples.png",
    window: Tuple[float, float] = (-0.005, 0.15),
) -> Optional[str]:
    """
    Plot a handful of example "with direct response" traces, one panel per
    trace, each zoomed to [stim_time + window[0], stim_time + window[1]] and
    showing three colors: the raw contaminated trace, the corrected (direct
    subtracted) trace, and the ground-truth trace before the direct response
    was added. Saved to filename (Agg backend, no display needed) for visual
    review. One panel per trace (rather than a single stacked-offset axis)
    so each trace keeps its own y-scale and the comparison stays legible.
    """
    if not example_traces:
        print("No example traces (with direct response) to plot.")
        return None

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(example_traces)
    fig, axes = plt.subplots(n, 1, figsize=(9, 2.2 * n), sharex=False)
    if n == 1:
        axes = [axes]

    dt = example_traces[0]["t"][1] - example_traces[0]["t"][0]
    for i, (ax, ex) in enumerate(zip(axes, example_traces)):
        t0 = ex["stim_time"] + window[0]
        t1 = ex["stim_time"] + window[1]
        i0 = max(0, int(t0 / dt))
        i1 = min(len(ex["t"]), int(t1 / dt))
        tw = ex["t"][i0:i1]
        ax.plot(tw, ex["contaminated_trace"][i0:i1] * 1e12, color="tab:red", lw=0.8, label="raw (contaminated)")
        ax.plot(tw, ex["corrected_trace"][i0:i1] * 1e12, color="tab:blue", lw=0.8, label="corrected (direct removed)")
        ax.plot(
            tw, ex["clean_trace"][i0:i1] * 1e12, color="tab:green", lw=0.8, alpha=0.9,
            label="ground truth (before direct added)",
        )
        ax.axvline(ex["stim_time"], color="k", lw=0.5, ls="--")
        ax.set_ylabel("pA")
        ax.set_title(
            f"direct amp={ex['amp_factor']:.1f}x sign={ex['direct_sign']:+d} "
            f"(true peak {ex['true_direct_peak'] * 1e12:.0f} pA)",
            fontsize=9,
        )
        if i == 0:
            ax.legend(loc="upper right", fontsize=7)
    axes[-1].set_xlabel("Time (s)")
    fig.suptitle("Direct-response separation: raw vs. corrected vs. ground truth", y=1.0)
    fig.tight_layout()
    fig.savefig(filename, dpi=150)
    plt.close(fig)
    print(f"Saved example plot to {filename}")
    return filename


def _baseline_relative_amp(trace, onset_s, pars, n_baseline=5):
    """Baseline-relative amplitude (signed) of an event at onset_s, measured
    on `trace`: peak (by sign) in a post-onset window minus a short pre-onset
    baseline. Used to score how accurately recovered PSC amplitudes are
    measured, independent of the detector's own (baseline-inclusive) value."""
    dt = pars.dt_seconds
    i0 = int(round(onset_s / dt))
    npost = int(max(pars.tau2 * 5.0, 0.005) / dt)
    b0 = max(0, i0 - n_baseline)
    if i0 <= b0 or i0 + npost >= len(trace):
        return np.nan
    baseline = np.mean(trace[b0:i0])
    seg = trace[i0 : i0 + npost]
    peak = np.min(seg) if pars.sign < 0 else np.max(seg)
    return peak - baseline


def _build_map(pars: BenchPars):
    """Generate one synthetic map: per-trace clean ground truth + observed
    trace (maybe direct) + true direct charge over the fit window. Returns
    (traces_2d, cleans, true_dcharges, t)."""
    dsp = MEDC.DirectSeparationPars()
    traces, cleans, dcharges, has_direct, direct_peaks = [], [], [], [], []
    for i in range(pars.n_trials):
        trng = np.random.default_rng(pars.seed + i)
        with_evoked = bool(trng.random() < pars.p_evoked)
        with_direct = bool(trng.random() < pars.p_direct)
        clean = make_synaptic_only_trial(pars, rng=trng, with_evoked=with_evoked)
        if with_direct:
            amp_factor = float(trng.uniform(*pars.direct_amp_range))
            dsign = int(trng.choice(pars.direct_sign_choices))
            contam = add_direct_response(pars, clean, amp_factor, dsign)
            trace = contam["trace"]
            pure = add_direct_response(
                pars, {"t": clean["t"], "trace": np.zeros_like(clean["trace"])}, amp_factor, dsign
            )["trace"]
            wi0 = int((pars.stim_time + dsp.direct_window_start) / pars.dt_seconds)
            wi1 = min(wi0 + int(dsp.direct_window_duration / pars.dt_seconds), len(pure))
            dcharges.append(float(np.sum(pure[wi0:wi1]) * pars.dt_seconds))
            direct_peaks.append(contam["direct_true_peak"])  # signed peak current (A)
        else:
            trace = clean["trace"]
            dcharges.append(0.0)
            direct_peaks.append(0.0)
        traces.append(trace)
        cleans.append(clean)
        has_direct.append(with_direct)
    return np.array(traces), cleans, dcharges, cleans[0]["t"], has_direct, direct_peaks


def run_condition(pars: BenchPars, seed: int = 2024, tolerance_s: float = 0.001) -> dict:
    """
    Evaluate separation over one synthetic map (pars.n_trials traces) and
    return aggregate recovery metrics: synaptic recall & false-positives (with
    vs without separation), median relative synaptic-amplitude error (with
    separation), and direct-charge recovery error. Uses the multi-trace
    algorithm when pars.multitrace is True, else the single-trace path -- so
    the same harness scores either by toggling pars.multitrace.
    """
    dsp = MEDC.DirectSeparationPars()
    dsp.enabled = True
    dsp.multitrace = pars.multitrace

    traces_2d, cleans, dcharges, t, has_direct, _direct_peaks = _build_map(pars)

    # no-separation baseline (per trace)
    rec_ns, fp_ns = [], []
    for k in range(len(cleans)):
        r = detect_once(t, traces_2d[k], pars, use_separation=False)
        s = score_recovery(r["onsets"], cleans[k]["true_onsets"], tolerance_s)
        rec_ns.append(s["recall"]); fp_ns.append(s["n_false_positive"])

    # with separation
    sep_results = []  # (onsets_s, corrected, records) per trace
    if pars.multitrace:
        kwargs = _detector_kwargs(pars)
        aj = MM.AndradeJonas(); aj.setup(ntraces=1, **{k: v for k, v in kwargs.items() if k != "engine"})
        aj._make_template(timebase=t)
        res = DRS.separate_direct_response_map(
            traces_2d, t, pars.dt_seconds, [pars.stim_time], aj.template, aj.template_amax,
            MM.AndradeJonas, kwargs, dsp,
        )
        for pt in res["per_trace"]:
            sep_results.append((np.asarray(pt[1]) * pars.dt_seconds, pt[0], pt[6]))
    else:
        for k in range(len(cleans)):
            r = detect_once(t, traces_2d[k], pars, use_separation=True,
                            direct_sep_pars=dsp, stim_time=pars.stim_time)
            sep_results.append((r["onsets"], r["corrected_trace"], r["records"]))

    rec_s, fp_s, amp_relerr, direct_charge_relerr = [], [], [], []
    # direct-subset (traces with a direct response) recall, where the hard case lives
    rec_ns_dir, rec_s_dir, amp_relerr_dir = [], [], []
    for k in range(len(cleans)):
        onsets_s, corrected, records = sep_results[k]
        s = score_recovery(onsets_s, cleans[k]["true_onsets"], tolerance_s)
        rec_s.append(s["recall"]); fp_s.append(s["n_false_positive"])
        if has_direct[k]:
            rec_ns_dir.append(rec_ns[k]); rec_s_dir.append(s["recall"])
        for true_i, det_j in match_events(onsets_s, cleans[k]["true_onsets"], tolerance_s):
            true_amp = cleans[k]["true_amps"][true_i]
            meas = _baseline_relative_amp(corrected, onsets_s[det_j], pars)
            if np.isfinite(meas) and abs(true_amp) > 0:
                amp_relerr.append(abs(meas - true_amp) / abs(true_amp))
                if has_direct[k]:
                    amp_relerr_dir.append(abs(meas - true_amp) / abs(true_amp))
        if dcharges[k] != 0.0 and records:
            direct_charge_relerr.append(abs(records[0].charge - dcharges[k]) / abs(dcharges[k]))

    def _m(x):
        x = [v for v in x if v is not None and not (isinstance(v, float) and np.isnan(v))]
        return float(np.mean(x)) if x else float("nan")

    def _med(x):
        x = [v for v in x if v is not None and not (isinstance(v, float) and np.isnan(v))]
        return float(np.median(x)) if x else float("nan")

    return dict(
        recall_nosep=_m(rec_ns), recall_sep=_m(rec_s),
        fp_nosep=_m(fp_ns), fp_sep=_m(fp_s),
        amp_relerr_median=_med(amp_relerr),
        direct_charge_relerr_median=_med(direct_charge_relerr),
        n_direct=int(sum(has_direct)),
        recall_nosep_direct=_m(rec_ns_dir), recall_sep_direct=_m(rec_s_dir),
        amp_relerr_direct_median=_med(amp_relerr_dir),
    )


def sweep_oat(base_pars: Optional[BenchPars] = None, sweeps: Optional[dict] = None,
              seed: int = 2024) -> dict:
    """
    One-at-a-time robustness sweep: for each parameter in `sweeps`
    (name -> list of values), vary it over its values while holding all
    others at base_pars, run_condition, and print/collect the metrics. This
    is the scaffold for the parametric robustness suite (synaptic shape/amp,
    direct shape/amp, SNR). Returns {param: [ (value, metrics), ... ]}.
    """
    if base_pars is None:
        base_pars = BenchPars()
    if sweeps is None:
        # SNR via noise_amp; synaptic amp; synaptic decay; direct decay; direct amp range
        sweeps = {
            "noise_amp": [0.3e-12, 1e-12, 3e-12, 6e-12],
            "mean_amp": [10e-12, 20e-12, 40e-12],
            "tau2": [1e-3, 2e-3, 4e-3],
            "direct_tau_decay": [0.010, 0.020, 0.040],
        }
    results = {}
    for name, values in sweeps.items():
        print(f"\n=== sweep: {name} ===")
        print(f"{'value':>12}  recall(ns/sep)  FP(ns/sep)   ampErr%  directQerr%")
        rows = []
        for v in values:
            pars = BenchPars(**{**base_pars.__dict__, name: v})
            m = run_condition(pars, seed=seed)
            rows.append((v, m))
            snr = pars.mean_amp / pars.noise_amp if name == "noise_amp" else float("nan")
            vstr = f"{v:.3e}" if isinstance(v, float) and abs(v) < 1e-2 else f"{v}"
            print(f"{vstr:>12}  {m['recall_nosep']:.2f}/{m['recall_sep']:.2f}     "
                  f"{m['fp_nosep']:.1f}/{m['fp_sep']:.1f}    "
                  f"{100*m['amp_relerr_median']:5.1f}    {100*m['direct_charge_relerr_median']:6.1f}"
                  + (f"   (SNR={snr:.0f})" if name == 'noise_amp' else ""))
        results[name] = rows
    return results


if __name__ == "__main__":
    run_benchmark()
    print()
    population = run_population_benchmark()
    plot_examples(population["example_traces"])
    print("\n\n########## Robustness sweep (single-trace separation) ##########")
    sweep_oat()
