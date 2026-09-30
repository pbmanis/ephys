"""
direct_response_separation.py

Optional separation of "direct" (non-synaptic, slow) responses from synaptic
events in laser-scanning photostimulation (LSPS) map data. Inert unless
explicitly wired in and enabled via MEDC.DirectSeparationPars.enabled (see
the `direct_response_separation` experiment configuration flag) -- nothing
in this module is imported or called by the existing detection pipeline by
default.

Design (revised after benchmarking against synthetic ground truth): the
original ephys-jl Julia approach (MiniAnalysis.jl:label_events,
LSPSFitting.jl:fit_direct) classifies each individually detected event as
spontaneous/direct/evoked by latency/rise-time/duration, then fits and
subtracts only the "direct"-labeled ones. Benchmarking here showed AJ/CB
correctly detect fast synaptic events riding on a slow baseline without
confusing them for it -- the direct response is a trace-level slow
background, not itself a detected event -- so per-event classification is
unnecessary. Instead:

  1. Detect fast events normally (AJ/CB, unchanged).
  2. Subtract every detected event's own contribution from the trace using
     the detector's own template, scaled to each event's measured amplitude
     (subtract_detected_events).
  3. Fit the stimulus-locked window of what remains -- now dominated by any
     slow direct component plus noise, fast content already removed
     (fit_direct_component, reused from the derivative-masked spline/
     template approach).
  4. Subtract that fit from the ORIGINAL trace (not the event-subtracted
     residual) in the stimulus window, and record onset time, duration, and
     charge of the fitted direct response (DirectResponseRecord) for later
     use -- e.g. as guidance/training data for a future ML-based
     direct/synaptic separation classifier.
  5. Re-run detection on the corrected trace for final, more accurate
     measurements (separate_direct_response ties 2-5 together per trace).

With no real direct response present, step 3's fit should be small (just
noise), so the whole pipeline should reproduce the no-separation result
almost exactly -- this is the key regression check in the test suite.
"""
from typing import List, Sequence, Tuple, Type

import numpy as np
import scipy.interpolate
import scipy.optimize
import scipy.signal

from ephys.mini_analyses import mini_event_dataclasses as MEDC


def subtract_detected_events(
    trace: np.ndarray,
    onsets: Sequence[int],
    peaks: Sequence[int],
    template: np.ndarray,
    template_amax: float,
    n_baseline_points: int = 5,
) -> np.ndarray:
    """
    Subtract each detected event's own contribution from `trace`, using the
    same template the detector used to find it, scaled to each event's own
    LOCAL amplitude: peak value minus a short pre-onset baseline average,
    both read directly from `trace`.

    This deliberately does NOT use MiniAnalyses.summarize()'s
    Mini_Event_Summary.amplitudes -- that field stores the raw absolute
    trace value at the peak (summarize() does `windowed_data[rawpk]` with no
    baseline subtraction), which is fine when the local baseline is flat but
    wrong here: for an event riding on a sloped direct-response baseline,
    the absolute peak value conflates the event's own size with the
    direct-response's level at that instant, and scaling the template by it
    would subtract (or fail to subtract) part of the direct response itself
    -- confirmed by benchmarking (it produced 2-3x overshoot past the true
    direct-response amplitude). Using a local, pre-onset baseline reference
    instead isolates just the fast event's own contribution.

    Template index 0 corresponds to the event onset (see
    MiniAnalyses._make_template), matching `onsets` directly. Leaves a
    residual dominated by any slow direct component (plus noise), and by
    fast-subtraction leftovers from events whose true shape did not exactly
    match the template (e.g. closely overlapping events) -- an expected,
    small source of error.
    """
    residual = trace.copy()
    n = len(residual)
    ntempl = len(template)
    if template_amax == 0 or ntempl == 0:
        return residual
    norm_template = template / template_amax  # peak = +-1, matching sign convention
    for onset_i, peak_i in zip(onsets, peaks):
        onset_i = int(onset_i)
        peak_i = int(peak_i)
        i1 = min(onset_i + ntempl, n)
        if i1 <= onset_i:
            continue
        b0 = max(0, onset_i - n_baseline_points)
        baseline = np.mean(trace[b0:onset_i]) if onset_i > b0 else trace[onset_i]
        local_amp = trace[peak_i] - baseline
        residual[onset_i:i1] -= local_amp * norm_template[: i1 - onset_i]
    return residual


def estimate_robust_derivative(
    y: np.ndarray, dt_seconds: float, window_points: int = 9
) -> np.ndarray:
    """
    Smoothed derivative of y, used to mask out any remaining fast content
    before fitting the slow direct component. window_points must be a
    small, fixed timescale (a fraction of the fastest synaptic tau), NOT
    scaled to the length of `y` -- a length-scaled window over-smooths away
    fast transients entirely (confirmed by smoke test during development).
    """
    n = len(y)
    window = window_points | 1  # force odd
    if window > n:
        window = n if n % 2 == 1 else n - 1
    if window < 5:
        return np.gradient(y, dt_seconds)
    return scipy.signal.savgol_filter(y, window_length=window, polyorder=2, deriv=1, delta=dt_seconds)


def _make_direct_template(risepower: int, ndecay: int):
    """
    Build a direct-response template function for curve_fit, generalizing
    ephys-jl's LSPSFitting.jl:direct_template
    (A*(1-exp(-(t-d)/tr))^n * exp(-(t-d)/td)) with adjustable rise exponent
    `risepower` (n) and either one or two decay time constants.

    Returns (func, param_names) where func(t, *params) evaluates the
    template. For ndecay==1 the params are (A, tau_rise, tau_decay, delay);
    for ndecay==2 they are (A, tau_rise, tau_decay1, tau_decay2, frac, delay)
    where the decay is (1-frac)*exp(-t/td1) + frac*exp(-t/td2).
    """
    n = risepower
    if ndecay == 1:
        def func(t, A, tau_rise, tau_decay, delay):
            tt = np.clip(t - delay, a_min=0.0, a_max=None)
            return A * (1.0 - np.exp(-tt / tau_rise)) ** n * np.exp(-tt / tau_decay)
        return func, ("A", "tau_rise", "tau_decay", "delay")
    elif ndecay == 2:
        def func(t, A, tau_rise, tau_decay1, tau_decay2, frac, delay):
            tt = np.clip(t - delay, a_min=0.0, a_max=None)
            decay = (1.0 - frac) * np.exp(-tt / tau_decay1) + frac * np.exp(-tt / tau_decay2)
            return A * (1.0 - np.exp(-tt / tau_rise)) ** n * decay
        return func, ("A", "tau_rise", "tau_decay1", "tau_decay2", "frac", "delay")
    else:
        raise ValueError(f"_make_direct_template: ndecay must be 1 or 2, got {ndecay}")


def fit_direct_component(
    t: np.ndarray,
    y: np.ndarray,
    mindy: float,
    mode: str = "spline",
    derivative_window_points: int = 9,
    spline_smooth_factor: float = 1.0,
    risepower: int = 2,
    template_ndecay: int = 1,
) -> np.ndarray:
    """
    Estimate the slow "direct" component in a windowed, event-subtracted
    trace segment by masking out any remaining fast points and fitting the
    remainder.

    mode="spline" (default): mask points where the smoothed derivative
    exceeds mindy in magnitude, then fit a smoothing spline through the
    remaining points and evaluate it at every t.

    mode="template": as above, but fit a bounded rise/decay function
    (_direct_template) instead of a spline. Less tested than "spline".

    The fit is performed internally in pA (y*1e12) and ms (t*1e3), matching
    ephys-jl's LSPSFitting.jl:fit_direct. This scaling matters: scipy's
    UnivariateSpline smoothing factor `s` is in the data's own (squared)
    units, so with raw Amps (~1e-11) the default `s=N` is trivially
    satisfied and the spline maximally-smooths into a cubic that ignores the
    data (confirmed by diagnostic: it read -15 pA where the true residual
    was -0.01 pA, and swung tens of pA above/below baseline). Instead `s` is
    set from an estimated noise sigma so RMS residual ~= sigma: smooth
    through noise while following the direct response, and start/end at ~0
    when no direct response is present in the window.

    Returns the fitted direct-component waveform, same shape as y, in Amps.
    """
    dt_seconds = t[1] - t[0]
    dy = estimate_robust_derivative(y, dt_seconds, window_points=derivative_window_points)
    ok = np.abs(dy) < mindy
    n_ok = int(np.count_nonzero(ok))
    if n_ok < 4:
        # not enough slow points to constrain a fit -- nothing to subtract
        return np.zeros_like(y)

    x_ms = (t - t[0]) * 1e3
    y_pA = y * 1e12

    if mode == "spline":
        # estimate noise sigma from successive differences of the slow
        # (masked) points: for near-white noise, std(diff)/sqrt(2) ~= sigma.
        yk = y_pA[ok]
        if len(yk) > 2:
            sigma = np.std(np.diff(yk)) / np.sqrt(2.0)
        else:
            sigma = np.std(yk)
        sigma = max(sigma, 1e-6)  # floor to avoid s=0 (interpolating) on flat input
        s = spline_smooth_factor * n_ok * (sigma ** 2)
        spline = scipy.interpolate.UnivariateSpline(x_ms[ok], y_pA[ok], k=3, s=s)
        return spline(x_ms) * 1e-12
    elif mode == "template":
        # Fit against a peak-normalized trace (amplitude ~= +-1) so curve_fit
        # is well-conditioned regardless of the absolute current scale, then
        # rescale the result. taus/delay in ms.
        scale = np.max(np.abs(y_pA[ok]))
        if scale <= 0:
            return np.zeros_like(y)
        yn = y_pA[ok] / scale
        sign = 1.0 if np.max(yn) >= -np.min(yn) else -1.0
        func, _names = _make_direct_template(risepower, template_ndecay)
        if template_ndecay == 1:
            #        A,    tau_rise, tau_decay, delay
            p0 = [sign, 3.0, 15.0, 0.0]
            lo = [-np.inf if sign < 0 else 0.0, 0.2, 1.0, 0.0]
            hi = [0.0 if sign < 0 else np.inf, 25.0, 100.0, x_ms[-1]]
        else:
            #        A,    tau_rise, tau_decay1, tau_decay2, frac, delay
            p0 = [sign, 3.0, 10.0, 40.0, 0.3, 0.0]
            lo = [-np.inf if sign < 0 else 0.0, 0.2, 1.0, 5.0, 0.0, 0.0]
            hi = [0.0 if sign < 0 else np.inf, 25.0, 40.0, 200.0, 1.0, x_ms[-1]]
        try:
            popt, _ = scipy.optimize.curve_fit(
                func, x_ms[ok], yn, p0=p0, bounds=(lo, hi), maxfev=5000
            )
        except RuntimeError:
            return np.zeros_like(y)
        return func(x_ms, *popt) * scale * 1e-12
    else:
        raise ValueError(f"fit_direct_component: unknown mode '{mode}'")


def _highpass(trace: np.ndarray, dt_seconds: float, fc_hz: float) -> np.ndarray:
    """Zero-phase 2nd-order Butterworth high-pass (filtfilt). fc_hz<=0 is a
    no-op. Used ONLY on the copy fed to event detection, so the detector does
    not fire on the slow direct response; amplitudes are always measured on
    the unfiltered trace."""
    if fc_hz is None or fc_hz <= 0:
        return trace
    sos = scipy.signal.butter(2, fc_hz, btype="high", fs=1.0 / dt_seconds, output="sos")
    return scipy.signal.sosfiltfilt(sos, trace)


def _detect(
    detect_trace: np.ndarray,
    measure_trace: np.ndarray,
    t: np.ndarray,
    detector_cls: Type,
    detector_kwargs: dict,
) -> MEDC.Mini_Event_Summary:
    """
    Detect events on `detect_trace` (e.g. a high-passed copy) but measure
    them (amplitudes) on `measure_trace` (the real, unfiltered trace):
    identify_events runs off the criterion computed from detect_trace, while
    summarize() reads amplitudes from measure_trace. Returns the summary.
    """
    m = detector_cls()
    kw = dict(detector_kwargs)
    engine = kw.pop("engine", None)
    kw["ntraces"] = 1
    m.setup(**kw)
    if m.method == "cb":
        m.set_cb_engine(engine=engine or "cython")
        m._make_template(timebase=t)
        m.cbTemplateMatch(detect_trace.copy(), itrace=0, prepare_data=False)
    else:
        m.deconvolve(detect_trace.copy(), timebase=t, itrace=0, prepare_data=False)
    m.identify_events(verbose=False)
    return m.summarize(measure_trace.copy().reshape(1, -1), verbose=False)


def separate_direct_response(
    raw_trace: np.ndarray,
    t: np.ndarray,
    dt_seconds: float,
    summary: MEDC.Mini_Event_Summary,
    itrace: int,
    stim_starts: Sequence[float],
    template: np.ndarray,
    template_amax: float,
    detector_cls: Type,
    detector_kwargs: dict,
    pars: MEDC.DirectSeparationPars,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[MEDC.DirectResponseRecord]]:
    """
    Single-trace orchestration of the full pipeline described in this
    module's docstring: subtract detected events -> fit each stimulus
    window of the residual -> subtract the fit from the original trace ->
    re-detect for final measurements. This is the one place production code
    (analyze_map_data.py) and test/benchmark code both call, so they always
    exercise the same logic.

    Returns
    -------
    (corrected_trace, onsets, peakindices, smpkindex, smoothed_peaks,
     amplitudes, direct_records)
        corrected_trace: raw_trace with any fitted direct response(s)
        subtracted (identical to raw_trace if no window's fit exceeded
        pars.min_direct_charge).
        onsets..amplitudes: the re-detected event arrays for this trace on
        corrected_trace, to replace summary's original arrays for this
        trace. If no direct response was recorded, these are simply copies
        of summary's original arrays (nothing changes).
        direct_records: one DirectResponseRecord per stimulus window where a
        direct response was recorded (empty if none).
    """
    # Detect the events to subtract on a HIGH-PASSED copy so the detector
    # does not fire on the slow direct response (see DirectSeparationPars.
    # detect_hpf); amplitudes are measured on the unfiltered raw_trace.
    detect_trace = _highpass(raw_trace, dt_seconds, pars.detect_hpf)
    det_summary = _detect(detect_trace, raw_trace, t, detector_cls, detector_kwargs)
    onsets = det_summary.onsets[0]
    peaks = det_summary.peakindices[0]

    event_free = subtract_detected_events(raw_trace, onsets, peaks, template, template_amax)

    corrected = raw_trace.copy()
    records: List[MEDC.DirectResponseRecord] = []
    for stim_i, stim_t in enumerate(stim_starts):
        window_t0 = stim_t + pars.direct_window_start
        i0 = int(window_t0 / dt_seconds)
        i1 = min(i0 + int(pars.direct_window_duration / dt_seconds), len(corrected))
        if i0 < 0 or i1 - i0 < 8:
            continue
        t_win = t[i0:i1] - t[i0]
        y_win = event_free[i0:i1]
        fit = fit_direct_component(
            t_win,
            y_win,
            mindy=pars.mindy,
            mode=pars.fit_mode,
            derivative_window_points=pars.derivative_window_points,
            spline_smooth_factor=pars.spline_smooth_factor,
            risepower=pars.risepower,
            template_ndecay=pars.template_ndecay,
        )
        charge = float(np.sum(fit) * dt_seconds)
        if abs(charge) < pars.min_direct_charge:
            continue
        corrected[i0:i1] = raw_trace[i0:i1] - fit
        peak_amplitude = float(fit[np.argmax(np.abs(fit))]) if len(fit) else 0.0
        records.append(
            MEDC.DirectResponseRecord(
                trace=itrace,
                stim_index=stim_i,
                onset_time=window_t0,
                duration=(i1 - i0) * dt_seconds,
                charge=charge,
                peak_amplitude=peak_amplitude,
                fit_mode=pars.fit_mode,
            )
        )

    if not records:
        # no direct response found for this trace -> leave the upstream
        # (raw-detected) events unchanged.
        return (
            corrected,
            summary.onsets[itrace],
            summary.peakindices[itrace],
            summary.smpkindex[itrace],
            summary.smoothed_peaks[itrace],
            summary.amplitudes[itrace],
            records,
        )

    # Final re-detection on the direct-subtracted trace: detect on a
    # high-passed copy (consistent with the subtraction-step detection),
    # measure amplitudes on the corrected (unfiltered) trace.
    corrected_detect = _highpass(corrected, dt_seconds, pars.detect_hpf)
    new_summary = _detect(corrected_detect, corrected, t, detector_cls, detector_kwargs)

    return (
        corrected,
        np.asarray(new_summary.onsets[0]),
        np.asarray(new_summary.peakindices[0]),
        np.asarray(new_summary.smpkindex[0]),
        np.asarray(new_summary.smoothed_peaks[0]),
        np.asarray(new_summary.amplitudes[0]),
        records,
    )


# ---------------------------------------------------------------------------
# Phase 2: multi-trace (cross-trace) direct-template estimation
# ---------------------------------------------------------------------------
# Rationale: the direct response is stimulus-locked and stereotyped across the
# traces/spots that have it, while spontaneous PSCs are random. So estimate the
# direct SHAPE once by averaging (event-subtracted, sign/amplitude-normalized)
# windows across the direct-containing traces, then fit only its amplitude
# (+small latency shift) per trace. A single coincident same-sign PSC barely
# perturbs a 1-2 parameter fit, which is the single-trace ill-posed case.


def _event_free_window(
    raw_trace, t, dt_seconds, i0, i1, template, template_amax, detector_cls, detector_kwargs, pars
):
    """Return the event-subtracted window [i0:i1] of raw_trace: detect PSCs on
    a high-passed copy, subtract them from raw, and slice the window. This is
    the per-trace direct estimate source used both for selection (its charge)
    and for building the cross-trace template."""
    detect_trace = _highpass(raw_trace, dt_seconds, pars.detect_hpf)
    det = _detect(detect_trace, raw_trace, t, detector_cls, detector_kwargs)
    ef = subtract_detected_events(raw_trace, det.onsets[0], det.peakindices[0], template, template_amax)
    return ef[i0:i1]


def estimate_direct_templates(
    traces_2d: np.ndarray,
    t: np.ndarray,
    dt_seconds: float,
    stim_starts: Sequence[float],
    template: np.ndarray,
    template_amax: float,
    detector_cls: Type,
    detector_kwargs: dict,
    pars: MEDC.DirectSeparationPars,
) -> dict:
    """
    Build direct-response shape template(s) from a map's traces.

    For each trace: event-subtract, take the first stimulus window, and use its
    charge as the direct-presence metric. Traces whose |charge| exceeds
    median + k*MAD (k = pars.direct_select_charge_factor) are "direct-containing".
    Each selected window is normalized by its own signed peak (canonicalizing
    scale AND sign to a +1-peak shape) and averaged (optionally into
    pars.n_shape_groups clusters). template_form="parametric" instead fits the
    rise/decay function to the averaged shape.

    Returns dict(templates=[unit-peak waveform,...], members=[indices,...],
    selected=[trace indices], charges=[per-trace window charge], window=(i0,i1)).
    """
    n_traces = traces_2d.shape[0]
    stim_t = stim_starts[0]
    i0 = int((stim_t + pars.direct_window_start) / dt_seconds)
    i1 = min(i0 + int(pars.direct_window_duration / dt_seconds), traces_2d.shape[1])

    windows = []
    charges = np.zeros(n_traces)
    for k in range(n_traces):
        ef_win = _event_free_window(
            traces_2d[k], t, dt_seconds, i0, i1, template, template_amax,
            detector_cls, detector_kwargs, pars,
        )
        windows.append(ef_win)
        charges[k] = np.sum(ef_win) * dt_seconds

    absq = np.abs(charges)
    med = np.median(absq)
    mad = np.median(np.abs(absq - med)) + 1e-30
    thresh = med + pars.direct_select_charge_factor * 1.4826 * mad
    selected = [k for k in range(n_traces) if absq[k] > thresh]

    result = dict(templates=[], members=[], selected=selected, charges=charges,
                  window=(i0, i1), threshold=thresh)
    if not selected:
        return result

    # normalize each selected window by its signed smoothed peak -> +1-peak shape
    norm_windows = []
    for k in selected:
        w = np.asarray(windows[k], dtype=float)
        sw = scipy.signal.savgol_filter(w, min(11, len(w) - (len(w) + 1) % 2), 2) if len(w) > 12 else w
        pk = sw[int(np.argmax(np.abs(sw)))]
        if pk == 0:
            continue
        norm_windows.append(w / pk)
    if not norm_windows:
        return result
    norm_windows = np.array(norm_windows)

    # group into shape clusters (default 1 = single average)
    if pars.n_shape_groups <= 1 or len(norm_windows) < pars.n_shape_groups:
        groups = [list(range(len(norm_windows)))]
    else:
        try:
            from sklearn.cluster import KMeans
            labels = KMeans(n_clusters=pars.n_shape_groups, n_init=5,
                            random_state=0).fit_predict(norm_windows)
            groups = [np.where(labels == g)[0].tolist() for g in range(pars.n_shape_groups)]
        except Exception:
            groups = [list(range(len(norm_windows)))]

    sel_arr = np.array(selected)
    for g in groups:
        if not g:
            continue
        avg = np.mean(norm_windows[g], axis=0)
        if pars.template_form == "parametric":
            x_ms = (t[i0:i1] - t[i0]) * 1e3
            func, _ = _make_direct_template(pars.risepower, pars.template_ndecay)
            sign = 1.0 if np.max(avg) >= -np.min(avg) else -1.0
            try:
                if pars.template_ndecay == 1:
                    p0 = [sign, 3.0, 15.0, 0.0]
                    lo = [-np.inf if sign < 0 else 0.0, 0.2, 1.0, 0.0]
                    hi = [0.0 if sign < 0 else np.inf, 25.0, 100.0, x_ms[-1]]
                else:
                    p0 = [sign, 3.0, 10.0, 40.0, 0.3, 0.0]
                    lo = [-np.inf if sign < 0 else 0.0, 0.2, 1.0, 5.0, 0.0, 0.0]
                    hi = [0.0 if sign < 0 else np.inf, 25.0, 40.0, 200.0, 1.0, x_ms[-1]]
                popt, _ = scipy.optimize.curve_fit(func, x_ms, avg, p0=p0, bounds=(lo, hi), maxfev=5000)
                avg = func(x_ms, *popt)
            except (RuntimeError, ValueError):
                pass  # fall back to the data-driven average
        # re-normalize to unit signed peak
        pk = avg[int(np.argmax(np.abs(avg)))]
        if pk != 0:
            avg = avg / pk
        result["templates"].append(avg)
        result["members"].append(sel_arr[g].tolist())
    return result


def _robust_amplitude(basis: np.ndarray, y: np.ndarray, niter: int = 3, k: float = 1.345) -> float:
    """
    Huber iteratively-reweighted least-squares estimate of the scalar `a` in
    y ~= a * basis. Points with large residuals (synaptic PSCs not fully
    removed, or direct shape mismatch) are downweighted, so the template
    amplitude is not biased by them -> cleaner subtraction, fewer residual
    in-window false positives. Falls back to plain projection if degenerate.
    """
    denom = float(np.dot(basis, basis))
    if denom <= 0:
        return 0.0
    a = float(np.dot(basis, y)) / denom
    for _ in range(niter):
        r = y - a * basis
        scale = 1.4826 * np.median(np.abs(r - np.median(r))) + 1e-30
        z = np.abs(r) / (k * scale)
        w = np.where(z <= 1.0, 1.0, 1.0 / np.maximum(z, 1e-30))  # Huber weights
        wdenom = float(np.dot(w * basis, basis))
        if wdenom <= 0:
            break
        a = float(np.dot(w * basis, y)) / wdenom
    return a


def apply_direct_template(
    raw_trace: np.ndarray,
    t: np.ndarray,
    dt_seconds: float,
    stim_starts: Sequence[float],
    templates: Sequence[np.ndarray],
    window: Tuple[int, int],
    template: np.ndarray,
    template_amax: float,
    detector_cls: Type,
    detector_kwargs: dict,
    pars: MEDC.DirectSeparationPars,
    max_shift_pts: int = 10,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, list]:
    """
    Fit the best cross-trace template's amplitude (+ small latency shift) to a
    trace's event-subtracted window (projection fit, robust to a single
    coincident PSC because it is a 1-parameter fit), subtract a*template from
    the ORIGINAL trace, and re-detect. Returns the same tuple shape as
    separate_direct_response plus a list of DirectResponseRecord.
    """
    i0, i1 = window
    corrected = raw_trace.copy()
    records: list = []

    if templates:
        ef_win = _event_free_window(
            raw_trace, t, dt_seconds, i0, i1, template, template_amax,
            detector_cls, detector_kwargs, pars,
        )
        # pick the template + integer shift maximizing correlation; amplitude
        # by projection onto the (shifted) template.
        best = None  # (abs_corr, amp, shift, tmpl_index)
        for ti, tmpl in enumerate(templates):
            tmpl = np.asarray(tmpl)
            for shift in range(-max_shift_pts, max_shift_pts + 1):
                shifted = np.roll(tmpl, shift)
                if shift > 0:
                    shifted[:shift] = 0.0
                elif shift < 0:
                    shifted[shift:] = 0.0
                denom = float(np.dot(shifted, shifted))
                if denom <= 0:
                    continue
                amp = float(np.dot(shifted, ef_win)) / denom
                resid = ef_win - amp * shifted
                corr = abs(amp) * denom / (np.linalg.norm(shifted) * np.linalg.norm(ef_win) + 1e-30)
                if best is None or corr > best[0]:
                    best = (corr, amp, shift, ti)
        if best is not None:
            _, amp, shift, ti = best
            tmpl = np.asarray(templates[ti])
            shifted = np.roll(tmpl, shift)
            if shift > 0:
                shifted[:shift] = 0.0
            elif shift < 0:
                shifted[shift:] = 0.0
            # robust (Huber IRLS) amplitude so residual PSCs / shape mismatch
            # do not bias the template scaling (reduces in-window residual FP)
            amp = _robust_amplitude(shifted, ef_win)
            fit = amp * shifted
            charge = float(np.sum(fit) * dt_seconds)
            if abs(charge) >= pars.min_direct_charge:
                corrected[i0:i1] = raw_trace[i0:i1] - fit
                records.append(
                    MEDC.DirectResponseRecord(
                        trace=0,
                        stim_index=0,
                        onset_time=stim_starts[0] + pars.direct_window_start,
                        duration=(i1 - i0) * dt_seconds,
                        charge=charge,
                        peak_amplitude=float(fit[int(np.argmax(np.abs(fit)))]),
                        fit_mode="multitrace",
                    )
                )

    detect_trace = _highpass(corrected, dt_seconds, pars.detect_hpf)
    new_summary = _detect(detect_trace, corrected, t, detector_cls, detector_kwargs)
    return (
        corrected,
        np.asarray(new_summary.onsets[0]),
        np.asarray(new_summary.peakindices[0]),
        np.asarray(new_summary.smpkindex[0]),
        np.asarray(new_summary.smoothed_peaks[0]),
        np.asarray(new_summary.amplitudes[0]),
        records,
    )


def separate_direct_response_map(
    traces_2d: np.ndarray,
    t: np.ndarray,
    dt_seconds: float,
    stim_starts: Sequence[float],
    template: np.ndarray,
    template_amax: float,
    detector_cls: Type,
    detector_kwargs: dict,
    pars: MEDC.DirectSeparationPars,
) -> dict:
    """
    Multi-trace orchestration over one map's traces (traces_2d = [n_traces,
    npts]): estimate the direct template(s) across traces, then apply the
    amplitude-only fit + re-detect per trace. Returns dict(per_trace=[...],
    templates_info=...). Each per_trace entry mirrors separate_direct_response's
    tuple (corrected, onsets, peakindices, smpkindex, smoothed_peaks,
    amplitudes, records).
    """
    tinfo = estimate_direct_templates(
        traces_2d, t, dt_seconds, stim_starts, template, template_amax,
        detector_cls, detector_kwargs, pars,
    )
    charges = tinfo["charges"]
    thresh = tinfo["threshold"]
    per_trace = []
    for k in range(traces_2d.shape[0]):
        # only apply the template to traces flagged as direct-containing (same
        # charge criterion used to build the template); non-direct traces get
        # an empty template list -> no subtraction, just re-detection. This
        # avoids injecting subtraction artifacts (false positives) into
        # PSP-only traces.
        tmpls = tinfo["templates"] if abs(charges[k]) > thresh else []
        per_trace.append(
            apply_direct_template(
                traces_2d[k], t, dt_seconds, stim_starts, tmpls,
                tinfo["window"], template, template_amax, detector_cls, detector_kwargs, pars,
            )
        )
    return dict(per_trace=per_trace, templates_info=tinfo)
