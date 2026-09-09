"""The two charts, and the reasoning behind how they are drawn.

THE DISTRIBUTION PLOT answers one question: where did the bounds land, and
which of the two candidate terms put them there. So the histogram is a
recessive grey -- it is context -- and the four candidate bounds carry the
colour. The winning candidate on each side is a solid line, the loser dashed,
and every line is labelled with its own value: which term bound must not be
carried by colour alone, and must be readable without the legend.

OVERFLOW BINS RATHER THAN A CLIPPED AXIS. A handful of extreme orders would
otherwise squash the body of the distribution into one bar. The x-range is the
band plus ten percent, and everything past it is clamped into the edge bins
with the count printed there. Clamping and hiding are different things.

THE CALIBRATION PLOT is the leave-one-month-out table drawn: k on the x-axis,
out-of-sample flags per month on the y, the median as a line and the month-to-
month range as a band around it. The range is the point -- a median of 5 with
a worst month of 12 is a different proposition from a median of 5 with a worst
month of 6, and a single line would hide that.

One y-axis on both. A second scale would make two quantities look comparable
that are not.
"""

from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")   # rendering to files, never to a display

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

# Validated light-mode tokens. A PNG has one theme, so one is committed to.
INK = {
    "surface": "#fcfcfb",
    "text": "#0b0b0b",
    "text_secondary": "#52514e",
    "grid": "#e1e0d9",
    "axis": "#c3c2b7",
    "hist": "#c3c2b7",      # recessive: the distribution is context
    "sigma": "#2a78d6",     # categorical slot 1
    "pct": "#eb6834",       # categorical slot 2
    "month": "#4a3aa7",     # categorical slot 7, for the scored-month overlay
    "target": "#0ca30c",    # status: good
}


def _fig(width=9.0, height=5.0):
    fig, ax = plt.subplots(figsize=(width, height), dpi=150)
    fig.patch.set_facecolor(INK["surface"])
    ax.set_facecolor(INK["surface"])
    ax.grid(True, color=INK["grid"], linewidth=0.8, alpha=0.9)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK["axis"])
    ax.tick_params(colors=INK["text_secondary"], labelsize=8)
    return fig, ax


def _save(fig, out_path: str) -> str:
    directory = os.path.dirname(os.path.abspath(out_path))
    os.makedirs(directory, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, facecolor=fig.get_facecolor())
    plt.close(fig)
    return out_path


def clip_with_overflow(values, x_lo: float, x_hi: float):
    """Clamp into [x_lo, x_hi]; return the clamped array and both counts."""
    a = np.asarray(values, dtype=float).ravel()
    a = a[np.isfinite(a)]
    if a.size == 0:
        return a, 0, 0
    below = int(np.count_nonzero(a < x_lo))
    above = int(np.count_nonzero(a > x_hi))
    return np.clip(a, x_lo, x_hi), below, above


def _bound_line(ax, x, color, solid: bool, label: str, y_frac: float,
                x_lo: float, x_hi: float):
    """One candidate bound, labelled on whichever side has room.

    A bound sitting in the right half of the axis gets its label to the LEFT
    of the line -- otherwise the upper bounds, which are always near the right
    edge, run off the figure.
    """
    if not np.isfinite(x):
        return
    ax.axvline(x, color=color, linewidth=2.0 if solid else 1.5,
               linestyle="-" if solid else "--", zorder=3)
    right_half = x > (x_lo + x_hi) / 2.0
    ax.annotate(f"{label}\n{x:.2f}", xy=(x, y_frac),
                xycoords=("data", "axes fraction"),
                xytext=(-4 if right_half else 4, 0),
                textcoords="offset points",
                color=INK["text"], fontsize=8, va="top",
                ha="right" if right_half else "left")


def distribution(values, band_row, out_path: str, title: str,
                 month_values=None, bins: int = 80) -> str:
    """Histogram of the metric with all four candidate bounds drawn."""
    x = np.asarray(values, dtype=float).ravel()
    x = x[np.isfinite(x)]

    lo = float(band_row.get("lo", np.nan))
    hi = float(band_row.get("hi", np.nan))
    fig, ax = _fig()

    if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
        pad = 0.10 * (hi - lo)
        x_lo, x_hi = lo - pad, hi + pad
    elif x.size:
        x_lo, x_hi = float(np.percentile(x, 0.5)), float(np.percentile(x, 99.5))
        if x_hi <= x_lo:
            x_lo, x_hi = x_lo - 1.0, x_hi + 1.0
    else:
        x_lo, x_hi = -1.0, 1.0

    clipped, below, above = clip_with_overflow(x, x_lo, x_hi)
    if clipped.size:
        ax.hist(clipped, bins=bins, range=(x_lo, x_hi),
                color=INK["hist"], edgecolor=INK["surface"], linewidth=0.4,
                label=f"fit window (n={clipped.size:,})")

    if month_values is not None:
        m, _, _ = clip_with_overflow(month_values, x_lo, x_hi)
        if m.size:
            ax.hist(m, bins=bins, range=(x_lo, x_hi), histtype="step",
                    color=INK["month"], linewidth=2.0,
                    label=f"scored month (n={m.size:,})")

    hi_binds = str(band_row.get("hi_binds", ""))
    lo_binds = str(band_row.get("lo_binds", ""))
    _bound_line(ax, float(band_row.get("sigma_hi", np.nan)), INK["sigma"],
                hi_binds == "sigma", "mean+k*sd", 0.98, x_lo, x_hi)
    _bound_line(ax, float(band_row.get("sigma_lo", np.nan)), INK["sigma"],
                lo_binds == "sigma", "mean-k*sd", 0.98, x_lo, x_hi)
    _bound_line(ax, float(band_row.get("p_hi", np.nan)), INK["pct"],
                hi_binds == "percentile", "P-high", 0.72, x_lo, x_hi)
    _bound_line(ax, float(band_row.get("p_lo", np.nan)), INK["pct"],
                lo_binds == "percentile", "P-low", 0.72, x_lo, x_hi)

    for count, xpos, ha in ((below, x_lo, "left"), (above, x_hi, "right")):
        if count:
            ax.annotate(f"{count:,} beyond", xy=(xpos, 0.02),
                        xycoords=("data", "axes fraction"),
                        xytext=(6 if ha == "left" else -6, 0),
                        textcoords="offset points", ha=ha,
                        color=INK["text_secondary"], fontsize=8)

    ax.set_xlim(x_lo, x_hi)
    ax.set_xlabel("performance (spreads)", color=INK["text_secondary"])
    ax.set_ylabel("orders", color=INK["text_secondary"])
    ax.set_title(title, color=INK["text"], fontsize=11, loc="left")

    caption = (f"k={float(band_row.get('k', float('nan'))):.2f}  "
               f"P{float(band_row.get('percentile', float('nan'))):.1f}  "
               f"band [{lo:.2f}, {hi:.2f}]  "
               f"solid = the term that bound (low: {lo_binds or 'none'}, "
               f"high: {hi_binds or 'none'})")
    ax.annotate(caption, xy=(0, -0.16), xycoords="axes fraction",
                color=INK["text_secondary"], fontsize=8)

    handles, _ = ax.get_legend_handles_labels()
    if len(handles) >= 2:
        ax.legend(frameon=False, fontsize=8, labelcolor=INK["text_secondary"])
    return _save(fig, out_path)


def calibration(curve_df: pd.DataFrame, out_path: str, target: int,
                chosen_k: float | None = None) -> str:
    """Out-of-sample flags per month as a function of k."""
    fig, ax = _fig(width=8.0, height=4.5)

    if len(curve_df) == 0:
        ax.annotate("No calibration curve was produced.", xy=(0.5, 0.5),
                    xycoords="axes fraction", ha="center",
                    color=INK["text_secondary"], fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        return _save(fig, out_path)

    c = curve_df.sort_values("k")
    k = c["k"].to_numpy(dtype=float)

    ax.fill_between(k, c["min_flags"], c["max_flags"], color=INK["sigma"],
                    alpha=0.15, linewidth=0, label="month-to-month range")
    ax.plot(k, c["median_flags"], color=INK["sigma"], linewidth=2.0,
            marker="o", markersize=4, label="median flags / month")

    ax.axhline(float(target), color=INK["target"], linewidth=2.0,
               linestyle="--", zorder=2)
    ax.annotate(f"target = {target}", xy=(k.max(), float(target)),
                xytext=(-4, 6), textcoords="offset points", ha="right",
                color=INK["text"], fontsize=8)

    if chosen_k is not None and np.isfinite(chosen_k):
        ax.axvline(float(chosen_k), color=INK["text_secondary"],
                   linewidth=1.5, linestyle=":", zorder=2)
        ax.annotate(f"chosen k = {chosen_k:.2f}",
                    xy=(float(chosen_k), 0.95),
                    xycoords=("data", "axes fraction"),
                    xytext=(5, 0), textcoords="offset points",
                    color=INK["text"], fontsize=8, va="top")

    ax.set_xlabel("k  (sigma multiple)", color=INK["text_secondary"])
    ax.set_ylabel("flags per month (out of sample)",
                  color=INK["text_secondary"])
    ax.set_title("What each k costs in review workload",
                 color=INK["text"], fontsize=11, loc="left")
    # Headroom so the range band does not sit flush against the frame.
    ax.set_ylim(bottom=0, top=float(c["max_flags"].max()) * 1.15 + 1.0)
    ax.legend(frameon=False, fontsize=8, labelcolor=INK["text_secondary"])
    return _save(fig, out_path)
