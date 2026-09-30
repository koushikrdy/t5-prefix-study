"""Tables and poster-scale figures from results/runs.jsonl.

    python -m src.report [--runs results/runs.jsonl] [--out figures]

Figures are rendered at 300 PPI and sized for a DIN A1 portrait poster with a
two-column grid, so they are legible at reading distance without rescaling.
"""
from __future__ import annotations

import argparse
import os
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .analysis import load_runs, load_swap, prefix_cost, write_all
from .config import TASK_ORDER, CONDITIONS_MULTI

DPI = 300
COL_IN = 10.2          # one poster column, inches (~260 mm)
FULL_IN = 21.0         # full text width, inches (~534 mm)

# Palette: categorical slots 1-3 of the validated default (blue / orange /
# aqua). Three slots is the documented all-pairs-safe cap, and three is what we
# need. Aqua sits under 3:1 on a light surface, so every bar carries a direct
# value label - the relief rule, and legible from two metres besides.
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
MUTED = "#52514e"
GRID = "#e4e3df"
CAT = {"cola": "#2a78d6", "sst2": "#eb6834", "mrpc": "#1baf7a"}
SLOT = ["#2a78d6", "#eb6834", "#1baf7a"]
# sequential blue ramp, light -> dark, one hue
BLUE_RAMP = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
             "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281",
             "#0d366b"]

COND_LABEL = {"correct": "correct", "wrong": "wrong task", "empty": "none",
              "nonsense": "nonsense", "shared": "shared"}
TASK_LABEL = {"cola": "CoLA", "sst2": "SST-2", "mrpc": "MRPC"}


def _style(base: int = 26) -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": base,
        "axes.titlesize": base + 4,
        "axes.labelsize": base,
        "xtick.labelsize": base - 2,
        "ytick.labelsize": base - 2,
        "legend.fontsize": base - 2,
        "axes.edgecolor": MUTED,
        "axes.linewidth": 1.4,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 1.2,
        "axes.axisbelow": True,
        "text.color": INK,
        "axes.labelcolor": INK,
        "xtick.color": INK,
        "ytick.color": INK,
        "grid.linestyle": "-",
        "figure.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
    })


def _despine(ax) -> None:
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def fig_accuracy_panels(df: pd.DataFrame, out: str,
                        width_in: float = FULL_IN,
                        height_in: float = 6.1,
                        base: int = 25,
                        compact: bool = False) -> str:
    """One panel per task, each on its own GLUE metric and its own scale.

    Putting CoLA on the same axis as SST-2 would hide the only thing that
    happens to it: its accuracy is pinned near the 0.69 majority baseline in
    every condition while its MCC moves from 0.24 to 0. Small multiples keep
    each task readable and keep the metrics honest.
    """
    _style(base=base)
    if compact:
        # Side by side on a poster there is no room for long legend entries or
        # a number over every bar; the numbers that matter are in the caption.
        regimes = [("single", "distinct", "A  single-task"),
                   ("multi", "distinct", "B  multi-task"),
                   ("multi", "collided", "C  collided labels")]
    else:
        regimes = [("single", "distinct", "A  single-task"),
                   ("multi", "distinct", "B  multi-task, disjoint labels"),
                   ("multi", "collided", "C  multi-task, collided labels")]
    conds = ["correct", "wrong", "empty", "nonsense"]
    n_seeds = int(df["seed"].nunique())

    fig, axes = plt.subplots(1, 3, figsize=(width_in, height_in))
    del n_seeds
    for ax, task in zip(axes, TASK_ORDER):
        metric = "mcc" if task == "cola" else "accuracy"
        col = "primary"
        x = np.arange(len(conds))
        slot = 0.82 / len(regimes)
        vals = []
        for i, (regime, scheme, lbl) in enumerate(regimes):
            d = df[(df.regime == regime) & (df.label_scheme == scheme)
                   & (df.task == task)]
            means = []
            for c in conds:
                sel = d[d.condition == c][col]
                means.append(sel.mean() if len(sel) else np.nan)
            pos = x + (i - (len(regimes) - 1) / 2) * slot
            ax.bar(pos, means, slot * 0.86, color=SLOT[i], label=lbl)
            for xi, m in zip(pos, means):
                if m == m:
                    vals.append(m)
                    if not compact:
                        ax.text(xi, m, f" {m:.2f}", ha="center", va="bottom",
                                fontsize=base - 7, color=INK, rotation=90)
                    elif abs(m) < 0.005:
                        # a zero-height bar is invisible; say it is a measured
                        # zero rather than a missing condition
                        ax.text(xi, 0, "0.00 ", ha="center", va="bottom",
                                fontsize=base - 5, color=INK, rotation=90)
        hi = max(vals) if vals else 1.0
        if metric == "mcc":
            ax.set_ylim(0, hi * (1.12 if compact else 1.45))
            ax.axhline(0, color=INK, lw=2.0)
            ax.set_ylabel("Matthews correlation")
            floor_note = "MCC 0 = chance"
        else:
            maj = df[df.task == task]["majority"].mean()
            ax.set_ylim(0, 1.05 if compact else 1.18)
            ax.axhline(maj, color=INK, lw=2.4, ls=(0, (2, 1.6)), zorder=5)
            ax.set_ylabel("accuracy")
            floor_note = f"majority class {maj:.2f}"
        ax.set_xticks(x)
        ax.set_xticklabels([COND_LABEL[c] for c in conds], rotation=20,
                           ha="right")
        ax.set_title(f"{TASK_LABEL[task]}", loc="left", pad=28,
                     fontsize=base + 3)
        ax.text(0, 1.015, floor_note, transform=ax.transAxes, fontsize=base - 6,
                color=MUTED)
        _despine(ax)

    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, ncol=3, loc="upper left",
               bbox_to_anchor=(0.008, 1.0))
    fig.tight_layout(rect=(0, 0.01, 1, 0.87))
    p = os.path.join(out, "accuracy_panels.png")
    fig.savefig(p, dpi=DPI); plt.close(fig)
    return p


def fig_prefix_cost(df: pd.DataFrame, out: str) -> str:
    """The headline: accuracy lost relative to the correct prefix, paired by seed."""
    _style()
    blocks = [("single", "distinct", "single-task"),
              ("multi", "distinct", "multi-task, disjoint labels"),
              ("multi", "collided", "multi-task, collided labels")]
    conds = ["wrong", "empty", "nonsense"]
    series = []
    for regime, scheme, lbl in blocks:
        pc = prefix_cost(df, regime, scheme)
        means, sds = [], []
        for c in conds:
            sel = pc[pc.condition == c]
            means.append(sel["delta_mean"].mean() if len(sel) else np.nan)
            sds.append(sel["delta_mean"].std(ddof=1) if len(sel) > 1 else 0.0)
        series.append((lbl, means, sds))

    fig, ax = plt.subplots(figsize=(FULL_IN, 8.4))
    x = np.arange(len(conds))
    group_w, n = 0.82, len(series)
    slot = group_w / n
    finite = [m for _, ms, _ in series for m in ms if m == m]
    span = max(0.02, max(abs(v) for v in finite) if finite else 0.02)
    pad = span * 0.26
    for i, (lbl, means, sds) in enumerate(series):
        pos = x + (i - (n - 1) / 2) * slot
        ax.bar(pos, means, slot * 0.88, yerr=sds, capsize=6, color=SLOT[i],
               label=lbl, error_kw={"elinewidth": 1.8, "ecolor": MUTED})
        for xi, m, sd in zip(pos, means, sds):
            if m != m:
                continue
            below = m < 0
            y = m - sd - pad * 0.18 if below else m + sd + pad * 0.18
            ax.text(xi, y, f"{m:+.3f}", ha="center",
                    va="top" if below else "bottom", fontsize=19, color=INK)
    lo = min(0.0, min(m - s for _, ms, ss in series
                      for m, s in zip(ms, ss) if m == m))
    hi = max(0.0, max(m + s for _, ms, ss in series
                      for m, s in zip(ms, ss) if m == m))
    ax.set_ylim(lo - pad, hi + pad)
    ax.axhline(0, color=INK, lw=2.2)
    ax.set_xticks(x)
    ax.set_xticklabels([COND_LABEL[c] for c in conds])
    ax.set_ylabel("accuracy relative to\nthe correct prefix")
    ax.set_xlabel("prefix used at training and test time")
    _despine(ax)
    h, l = ax.get_legend_handles_labels()
    fig.legend(h, l, frameon=False, ncol=3, loc="upper left",
               bbox_to_anchor=(0.01, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    p = os.path.join(out, "prefix_cost.png")
    fig.savefig(p, dpi=DPI); plt.close(fig)
    return p


def fig_swap_grid(sw: pd.DataFrame, out: str,
                  width_in: float = FULL_IN,
                  height_in: float = 5.3,
                  base: int = 24,
                  compact: bool = False) -> str:
    """The contribution figure: the same prefix swap under two label schemes.

    With disjoint verbalizers a wrong prefix makes the model emit another
    task's label strings, which score zero by construction. Collapsing every
    task onto yes/no removes that floor effect, so the right-hand panel shows
    what the swap actually costs rather than what the verbalizer design makes
    it look like.
    """
    _style(base=base)
    if compact:
        panels = [("distinct", "disjoint labels"),
                  ("collided", "collided labels (yes / no)")]
    else:
        panels = [("distinct", "disjoint labels\n(acceptable / positive / equivalent)"),
                  ("collided", "collided labels\n(every task answers yes / no)")]
    have = [p for p in panels if not sw[(sw.regime == "multi") &
                                        (sw.label_scheme == p[0])].empty]
    if not have:
        return ""
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("seq_blue", BLUE_RAMP)
    fig, axes = plt.subplots(1, len(have), figsize=(width_in, height_in))
    axes = np.atleast_1d(axes)
    im = None
    for ax, (scheme, title) in zip(axes, have):
        d = sw[(sw.regime == "multi") & (sw.label_scheme == scheme)]
        tokens = [t for t in ["cola", "sst2", "mrpc", "(none)", "zorbex"]
                  if t in set(d.inference_token)]
        M = np.full((len(TASK_ORDER), len(tokens)), np.nan)
        for i, t in enumerate(TASK_ORDER):
            for j, tok in enumerate(tokens):
                sel = d[(d.task == t) & (d.inference_token == tok)]
                if len(sel):
                    M[i, j] = sel["accuracy"].mean()
        im = ax.imshow(M, cmap=cmap, vmin=0, vmax=1, aspect="auto")
        ax.set_xticks(range(len(tokens)))
        ax.set_xticklabels([("none" if t == "(none)" else t) for t in tokens],
                           rotation=20, ha="right")
        ax.set_yticks(range(len(TASK_ORDER)))
        first = ax is axes[0]
        ax.set_yticklabels([TASK_LABEL[t] for t in TASK_ORDER]
                           if (first or not compact) else [])
        ax.set_title(title, loc="left", pad=16, fontsize=base + 1)
        ax.set_xlabel("token at test time" if compact
                      else "task token supplied at test time")
        ax.grid(False)
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                if np.isnan(M[i, j]):
                    continue
                on_diag = j < len(TASK_ORDER) and TASK_ORDER[j] == TASK_ORDER[i]
                ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center",
                        fontsize=base + 3 if on_diag else base + 1,
                        fontweight="bold" if on_diag else "normal",
                        color="white" if M[i, j] > 0.60 else INK)
        for j in range(1, M.shape[1]):
            ax.axvline(j - 0.5, color=SURFACE, lw=3.0)
        for i in range(1, M.shape[0]):
            ax.axhline(i - 0.5, color=SURFACE, lw=3.0)
        for spine in ax.spines.values():
            spine.set_visible(False)
    axes[0].set_ylabel("evaluated on")
    fig.colorbar(im, ax=axes.tolist(), label="accuracy", fraction=0.03, pad=0.02)
    p = os.path.join(out, "swap_grid.png")
    fig.savefig(p, dpi=DPI, bbox_inches="tight"); plt.close(fig)
    return p


def fig_misroute(df: pd.DataFrame, out: str) -> str:
    """How often the model answers a different question than the one asked."""
    _style(base=24)
    d = df[df.label_scheme == "distinct"]
    conds = [c for c in CONDITIONS_MULTI if c in set(d.condition)]
    fig, ax = plt.subplots(figsize=(COL_IN, 7.4))
    x = np.arange(len(conds))
    slot = 0.42
    top = 0.0
    labels: List[tuple] = []
    for i, regime in enumerate(["single", "multi"]):
        dr = d[d.regime == regime]
        means = [dr[dr.condition == c]["misroute_rate"].mean() for c in conds]
        pos = x + (i - 0.5) * slot
        ax.bar(pos, means, slot * 0.88, color=SLOT[i], label=f"{regime}-task")
        for xi, m in zip(pos, means):
            if m == m:
                top = max(top, m)
                labels.append((xi, m))
    ax.set_ylim(0, max(0.01, top * 1.45))
    dec = 0 if top >= 0.03 else 1
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(
        lambda v, _: f"{v * 100:.{dec}f}%"))
    for xi, m in labels:
        ax.text(xi, m + top * 0.05, f"{m * 100:.{dec}f}%", ha="center",
                va="bottom", fontsize=18, color=INK)
    ax.set_xticks(x)
    ax.set_xticklabels([COND_LABEL[c] for c in conds], rotation=20, ha="right")
    ax.set_ylabel("answered a different task")
    ax.set_xlabel("prefix condition")
    _despine(ax)
    h, l = ax.get_legend_handles_labels()
    fig.legend(h, l, frameon=False, ncol=2, loc="upper left",
               bbox_to_anchor=(0.02, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    p = os.path.join(out, "misroute.png")
    fig.savefig(p, dpi=DPI); plt.close(fig)
    return p


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="results/runs.jsonl")
    ap.add_argument("--out", default="figures")
    ap.add_argument("--results-dir", default="results")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    paths = write_all(a.runs, a.results_dir)
    df, sw = load_runs(a.runs), load_swap(a.runs)
    made = [fig_accuracy_panels(df, a.out), fig_prefix_cost(df, a.out),
            fig_swap_grid(sw, a.out), fig_misroute(df, a.out)]
    # Column-width variants for the poster's two-up figure row.
    pcol = os.path.join(a.out, "poster")
    os.makedirs(pcol, exist_ok=True)
    # Matched aspect ratios so the two sit side by side with aligned captions,
    # and type sized to be read at poster distance rather than on screen.
    made += [fig_accuracy_panels(df, pcol, width_in=11.0, height_in=6.9,
                                 base=21, compact=True),
             fig_swap_grid(sw, pcol, width_in=11.0, height_in=6.9,
                           base=19, compact=True)]
    for p in made:
        if p:
            print("wrote", p)
    for k, v in paths.items():
        print("wrote", v)


if __name__ == "__main__":
    main()
