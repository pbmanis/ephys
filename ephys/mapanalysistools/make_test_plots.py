"""
make_test_plots.py

Generate diagnostic figures for the direct-response separation algorithm
(ephys.mini_analyses.direct_response_separation) on synthetic map data, and
write them as captioned PDFs under ephys/mapanalysistools/test_plots/.

Figures:
  1. map_grid.pdf   -- the synthetic LSPS map as a 2D (approximately square)
                       grid of spots. Boxless; all cells share one amplitude
                       scale (vertical pA scale bar at left) and time base
                       (ms axis along the bottom row). Each spot's RAW trace
                       is blue if its direct response is positive-going, red
                       if negative-going, grey if it has no direct response;
                       the corrected (direct-removed) trace is overlaid in
                       black and the ground-truth synaptic-only trace in
                       green. A corner number gives that spot's detection
                       recall (fraction of true synaptic events recovered);
                       spots with false positives or poor recall are
                       highlighted.
  2. latency_hist.pdf -- detected-event latency histograms (true vs false),
                       single- vs multi-trace, with a statistics panel
                       (correct/incorrect counts, recall, precision, F1,
                       direct-subset recall, false positives in the direct
                       window).

Run:  python -m ephys.mapanalysistools.make_test_plots
"""
import os
import datetime

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from ephys.mini_analyses.tests.benchmark_direct_separation import (
    BenchPars, _build_map, _detector_kwargs, detect_once, match_events,
)
import ephys.mini_analyses.minis_methods as MM
import ephys.mini_analyses.direct_response_separation as DRS
from ephys.mini_analyses.mini_event_dataclasses import DirectSeparationPars

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUT = os.path.join(os.path.dirname(__file__), "test_plots")
MODEL = "Claude (Opus 4.8, claude-opus-4-8) generated"
TOL = 0.001  # s, onset-matching tolerance
WIN_PRE, WIN_POST = 0.010, 0.460  # display window around stimulus (s)

BLUE, RED, GREEN, BLACK, GREY = "#2f6fd0", "#d23b3b", "#3aa35a", "#222222", "#b8b8b8"


def _git_hash(repo=REPO):
    """Read the current commit hash from .git metadata directly (no git exec)."""
    try:
        head = open(os.path.join(repo, ".git", "HEAD")).read().strip()
        if head.startswith("ref:"):
            ref = head.split(" ", 1)[1]
            p = os.path.join(repo, ".git", ref)
            if os.path.exists(p):
                return open(p).read().strip()[:12]
            packed = os.path.join(repo, ".git", "packed-refs")
            if os.path.exists(packed):
                for line in open(packed):
                    if line.strip().endswith(ref):
                        return line.split()[0][:12]
        return head[:12]
    except Exception:
        return "unknown"


def _caption(fig, description, y=0.006):
    stamp = f"{description}\n{datetime.date.today().isoformat()}  |  git {_git_hash()}  |  {MODEL}"
    fig.text(0.5, y, stamp, ha="center", va="bottom", fontsize=7.5, color="#444", wrap=True)


def _nice(x):
    """Round x up to a friendly 1/2/5 x 10^n value (for scale bars)."""
    if x <= 0:
        return 1.0
    mag = 10 ** np.floor(np.log10(x))
    for m in (1, 2, 5, 10):
        if m * mag >= x:
            return m * mag
    return 10 * mag


def build(pars):
    """Generate the map and run both single- and multi-trace separation."""
    traces, cleans, dcharges, t, has_direct, direct_peaks = _build_map(pars)
    kwargs = _detector_kwargs(pars)
    aj = MM.AndradeJonas()
    aj.setup(ntraces=1, **{k: v for k, v in kwargs.items() if k != "engine"})
    aj._make_template(timebase=t)
    dsp = DirectSeparationPars()
    dsp.enabled = True
    dsp.multitrace = True
    res = DRS.separate_direct_response_map(
        traces, t, pars.dt_seconds, [pars.stim_time], aj.template, aj.template_amax,
        MM.AndradeJonas, kwargs, dsp,
    )
    single = [detect_once(t, traces[k], pars, use_separation=True, direct_sep_pars=dsp,
                          stim_time=pars.stim_time) for k in range(len(cleans))]
    return dict(pars=pars, traces=traces, cleans=cleans, t=t, has_direct=has_direct,
                direct_peaks=direct_peaks, res=res, single=single)


def _per_trace_stats(onsets_s, true_onsets):
    pairs = match_events(onsets_s, true_onsets, TOL)
    n_true = len(list(true_onsets))
    n_match = len(pairs)
    recall = (n_match / n_true) if n_true else 1.0
    fp = len(list(onsets_s)) - n_match
    return recall, fp, n_true, n_match


def _mode_results(data, mode):
    """Return (corrected_traces, onsets_s_per_trace) for 'single' or 'multi'
    trace separation, so both figures can be produced for either mode."""
    n = len(data["cleans"])
    dt = data["pars"].dt_seconds
    if mode == "multi":
        pt = data["res"]["per_trace"]
        corrected = [pt[k][0] for k in range(n)]
        onsets_s = [np.asarray(pt[k][1]) * dt for k in range(n)]
    else:
        single = data["single"]
        corrected = [single[k]["corrected_trace"] for k in range(n)]
        onsets_s = [np.asarray(single[k]["onsets"]) for k in range(n)]  # already seconds
    return corrected, onsets_s


def plot_map_grid(data, filename, mode="multi"):
    pars = data["pars"]
    traces, cleans, t = data["traces"], data["cleans"], data["t"]
    direct_peaks = data["direct_peaks"]
    corrected_all, onsets_all = _mode_results(data, mode)
    dt = pars.dt_seconds
    stim = pars.stim_time
    i0 = int((stim - WIN_PRE) / dt)
    i1 = int((stim + WIN_POST) / dt)
    tw = (t[i0:i1] - stim) * 1e3  # ms

    n = len(cleans)
    ncol = int(np.ceil(np.sqrt(n)))
    nrow = int(np.ceil(n / ncol))

    # global amplitude scale across every trace shown (pA)
    lo, hi = np.inf, -np.inf
    for k in range(n):
        for arr in (traces[k][i0:i1], corrected_all[k][i0:i1], cleans[k]["trace"][i0:i1]):
            lo = min(lo, float(np.min(arr))); hi = max(hi, float(np.max(arr)))
    lo *= 1e12; hi *= 1e12
    pad = 0.05 * (hi - lo)
    ylim = (lo - pad, hi + pad)
    scale_pA = _nice((hi - lo) / 4.0)

    fig, axes = plt.subplots(nrow, ncol, figsize=(1.55 * ncol, 1.15 * nrow), squeeze=False)
    fig.subplots_adjust(left=0.055, right=0.995, top=0.95, bottom=0.075, wspace=0.08, hspace=0.12)

    for k in range(nrow * ncol):
        ax = axes[k // ncol][k % ncol]
        ax.set_xlim(tw[0], tw[-1]); ax.set_ylim(*ylim)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_xticks([]); ax.set_yticks([])
        if k >= n:
            ax.axis("off"); continue

        raw_color = GREY if direct_peaks[k] == 0 else (BLUE if direct_peaks[k] > 0 else RED)
        ax.plot(tw, cleans[k]["trace"][i0:i1] * 1e12, color=GREEN, lw=0.35, alpha=0.8)
        ax.plot(tw, corrected_all[k][i0:i1] * 1e12, color=BLACK, lw=0.35)
        ax.plot(tw, traces[k][i0:i1] * 1e12, color=raw_color, lw=0.5, alpha=0.9)

        onsets_s = onsets_all[k]
        recall, fp, n_true, _ = _per_trace_stats(onsets_s, cleans[k]["true_onsets"])
        # highlight the separation-specific failure mode: false positives
        # (spurious detections, mostly direct-window residual). Low recall
        # alone is not flagged -- baseline detection misses a few spontaneous
        # events by chance on non-direct traces too.
        failure = fp > 0
        if failure:
            ax.set_facecolor("#fceaea")
        txt = f"{recall:.2f}" + (f" +{fp}" if fp > 0 else "")
        ax.text(0.03, 0.97, txt, transform=ax.transAxes, fontsize=6.0, va="top", ha="left",
                color=(RED if failure else "#666"),
                fontweight=("bold" if failure else "normal"))

    # time base along the bottom row (ms)
    xticks = [x for x in (0, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500) if tw[0] <= x <= tw[-1]]
    for c in range(ncol):
        ax = axes[nrow - 1][c]
        if c < n - (nrow - 1) * ncol or (nrow - 1) * ncol + c < n:
            ax.set_xticks(xticks)
            ax.set_xticklabels([str(x) for x in xticks], fontsize=6)
            ax.tick_params(axis="x", length=2, pad=1)
    axes[nrow - 1][ncol // 2].set_xlabel("time re: stimulus (ms)", fontsize=8)

    # amplitude scale bar (pA) at the left, on a middle-row leftmost cell
    sb_ax = axes[nrow // 2][0]
    x_sb = tw[0] - 0.02 * (tw[-1] - tw[0])
    y0 = ylim[0] + 0.10 * (ylim[1] - ylim[0])
    sb_ax.add_line(Line2D([x_sb, x_sb], [y0, y0 + scale_pA], color=BLACK, lw=1.5, clip_on=False))
    sb_ax.text(x_sb - 0.01 * (tw[-1] - tw[0]), y0 + scale_pA / 2,
               f"{scale_pA:g} pA", rotation=90, va="center", ha="right", fontsize=7)

    # legend
    handles = [
        Line2D([0], [0], color=BLUE, lw=1.2, label="raw: +direct"),
        Line2D([0], [0], color=RED, lw=1.2, label="raw: -direct"),
        Line2D([0], [0], color=GREY, lw=1.2, label="raw: no direct"),
        Line2D([0], [0], color=BLACK, lw=1.2, label="corrected"),
        Line2D([0], [0], color=GREEN, lw=1.2, label="ground truth"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=5, fontsize=7.5,
               frameon=False, bbox_to_anchor=(0.5, 1.0))

    label = "multi-trace" if mode == "multi" else "single-trace"
    _caption(fig,
             f"Synthetic LSPS map ({n} spots) as a 2D grid, each zoomed to "
             f"[-{WIN_PRE*1e3:.0f}, +{WIN_POST*1e3:.0f}] ms about the stimulus; all spots share "
             f"the amplitude scale (left, {scale_pA:g} pA bar) and time base (bottom row). "
             f"Raw trace colored by direct-response polarity (blue=+, red=-, grey=none); black=corrected "
             f"(direct removed, {label}); green=ground-truth synaptic-only. Corner number = detection "
             "recall (fraction of true synaptic events recovered), '+n' = false positives; red/shaded = "
             f"failure (any false positive). p_direct={pars.p_direct}, p_evoked={pars.p_evoked}, "
             f"direct amp {pars.direct_amp_range[0]:g}-{pars.direct_amp_range[1]:g}x mean PSC.")
    fig.savefig(filename)
    plt.close(fig)
    print("saved", filename)


def plot_latency(data, filename, mode="multi"):
    pars = data["pars"]
    cleans, has_direct = data["cleans"], data["has_direct"]
    stim = pars.stim_time
    dwin = (0.0, DirectSeparationPars().direct_window_duration * 1e3)  # ms, direct window
    _, onsets_all = _mode_results(data, mode)

    true_lat, false_lat = [], []
    n_true = n_match = n_false = fp_in_dwin = 0
    n_true_dir = n_match_dir = 0
    for k in range(len(cleans)):
        onsets_s = onsets_all[k]
        pairs = match_events(onsets_s, cleans[k]["true_onsets"], TOL)
        matched = set(j for _, j in pairs)
        nt = len(list(cleans[k]["true_onsets"]))
        n_true += nt
        n_match += len(pairs)
        if has_direct[k]:
            n_true_dir += nt
            n_match_dir += len(pairs)
        for j, o in enumerate(onsets_s):
            lat = (o - stim) * 1e3
            if j in matched:
                true_lat.append(lat)
            else:
                false_lat.append(lat)
                n_false += 1
                if dwin[0] <= lat <= dwin[1]:
                    fp_in_dwin += 1
    recall = n_match / n_true if n_true else float("nan")
    precision = n_match / (n_match + n_false) if (n_match + n_false) else float("nan")
    f1 = (2 * recall * precision / (recall + precision)
          if recall and precision else float("nan"))
    recall_dir = n_match_dir / n_true_dir if n_true_dir else float("nan")

    label = "multi-trace" if mode == "multi" else "single-trace"
    bins = np.arange(-10, 500, 5)
    fig, ax = plt.subplots(1, 1, figsize=(8, 4.6))
    ax.hist(true_lat, bins=bins, color=GREEN, alpha=0.75, label=f"correct (matched)  n={n_match}")
    ax.hist(false_lat, bins=bins, color=RED, alpha=0.75, label=f"incorrect (false)  n={n_false}")
    ax.axvline(0, color="k", ls="--", lw=0.8)
    ax.set_title(f"{label}: detected-event latency", fontsize=12)
    ax.set_xlabel("latency after stimulus (ms)")
    ax.set_ylabel("count")
    ax.legend(fontsize=9, loc="upper right")
    stats = (f"true events: {n_true}\n"
             f"correct: {n_match}   incorrect: {n_false}\n"
             f"recall: {recall:.3f}\n"
             f"precision: {precision:.3f}\n"
             f"F1: {f1:.3f}\n"
             f"recall (direct spots): {recall_dir:.3f}\n"
             f"FP in direct window: {fp_in_dwin}/{n_false}")
    ax.text(0.98, 0.62, stats, transform=ax.transAxes, fontsize=9, va="top", ha="right",
            family="monospace", bbox=dict(boxstyle="round", fc="#f4f4f0", ec="#ccc"))
    fig.tight_layout(rect=[0, 0.12, 1, 1])
    _caption(fig,
             f"Detected-event latency (time after stimulus), {label} direct-response separation: "
             "correct (matched to a seeded event) vs incorrect (false-positive) detections. "
             f"Synthetic map: {pars.n_trials} traces, p_direct={pars.p_direct}, p_evoked={pars.p_evoked}, "
             f"direct amp {pars.direct_amp_range[0]:g}-{pars.direct_amp_range[1]:g}x mean PSC. "
             "Recall=correct/true; precision=correct/(correct+false); F1=harmonic mean; "
             "recall(direct spots)=recall restricted to spots with a direct response.")
    fig.savefig(filename)
    plt.close(fig)
    print("saved", filename)


def main():
    os.makedirs(OUT, exist_ok=True)
    pars = BenchPars()
    pars.n_trials = 400
    pars.p_direct = 0.1
    pars.p_evoked = 0.5
    pars.direct_amp_range = (3.0, 6.0)
    pars.direct_sign_choices = (1,)  # all direct responses same sign as PSCs (negative)
    pars.multitrace = True
    data = build(pars)
    # single-trace analysis (plain names) and multi-trace analysis (_MT)
    plot_map_grid(data, os.path.join(OUT, "map_grid.pdf"), mode="single")
    plot_latency(data, os.path.join(OUT, "latency_hist.pdf"), mode="single")
    plot_map_grid(data, os.path.join(OUT, "map_grid_MT.pdf"), mode="multi")
    plot_latency(data, os.path.join(OUT, "latency_hist_MT.pdf"), mode="multi")


if __name__ == "__main__":
    main()
