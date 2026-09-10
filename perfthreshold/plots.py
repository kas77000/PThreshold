"""The two charts, and the reasoning behind how they are drawn.

THE DISTRIBUTION PLOT answers one question: where did the bounds land, and
which of the two candidate terms put them there. So the histogram is a
recessive grey -- it is context -- and the four candidate bounds carry the
colour. The winning candidate on each side is a solid line, the loser dashed,
and every line is labelled with its own value: which term bound must not be
carried by colour alone, and must be readable without the legend.

OVERFLOW BINS RATHER THAN A CLIPPED AXIS. A handful of extreme orders would
otherwise squash the body of the distribution into one bar. The x-range is the
band plus a margin, and everything past it is clamped into the edge bins
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
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
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
    "normal": "#e34948",    # categorical slot 8: the assumed Gaussian
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
    fig.savefig(out_path, facecolor=fig.get_facecolor(),
                bbox_inches="tight")
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


def breach_counts(values, lo: float, hi: float) -> tuple[int, int]:
    """Orders below and above the BAND. Not the same as clip_with_overflow.

    These two counts are what the band puts on a desk, and they sum to the
    figure in the chart's subtitle. `clip_with_overflow` counts something
    different -- observations beyond the drawn AXIS, which is deliberately
    wider than the band, so an order can breach the band and still be on
    scale. Conflating the two makes the numbers on the chart fail to add up.
    """
    a = np.asarray(values, dtype=float).ravel()
    a = a[np.isfinite(a)]
    if a.size == 0 or not (np.isfinite(lo) and np.isfinite(hi)):
        return 0, 0
    return int(np.count_nonzero(a < lo)), int(np.count_nonzero(a > hi))


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
                color=INK["text"], fontsize=8, va="top", zorder=5,
                ha="right" if right_half else "left",
                bbox=dict(facecolor=INK["surface"], edgecolor="none",
                          boxstyle="round,pad=0.2", alpha=0.92))


def distribution(values, band_row, out_path: str, title: str,
                 month_values=None, bins: int = 80,
                 show_normal: bool = True) -> str:
    """Histogram of the metric with all four candidate bounds drawn."""
    x = np.asarray(values, dtype=float).ravel()
    x = x[np.isfinite(x)]

    lo = float(band_row.get("lo", np.nan))
    hi = float(band_row.get("hi", np.nan))
    fig, ax = _fig()

    if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
        pad = 0.16 * (hi - lo)
        x_lo, x_hi = lo - pad, hi + pad
    elif x.size:
        x_lo, x_hi = float(np.percentile(x, 0.5)), float(np.percentile(x, 99.5))
        if x_hi <= x_lo:
            x_lo, x_hi = x_lo - 1.0, x_hi + 1.0
    else:
        x_lo, x_hi = -1.0, 1.0

    n_low, n_high = breach_counts(x, lo, hi)
    if np.isfinite(lo) and np.isfinite(hi) and x.size:
        outside = n_low + n_high
        pct = 100.0 * outside / x.size
        subtitle = (f"{outside:,} of {x.size:,} orders outside the band "
                    f"({n_low:,} low, {n_high:,} high) — {pct:.3f}%")
    else:
        subtitle = "no band fitted for this cell"

    clipped, below, above = clip_with_overflow(x, x_lo, x_hi)
    if clipped.size:
        ax.hist(clipped, bins=bins, range=(x_lo, x_hi),
                color=INK["hist"], edgecolor=INK["surface"], linewidth=0.4)

    # The fitted normal, scaled to the histogram. This is the distribution the
    # coverage claim assumes -- drawing it lets the gap be seen rather than
    # argued about, which is the point of putting it on the chart at all.
    normal_drawn = False
    mu = float(band_row.get("mean", np.nan))
    sd = float(band_row.get("sd", np.nan))
    if show_normal and clipped.size and np.isfinite(mu) and sd > 0:
        grid = np.linspace(x_lo, x_hi, 400)
        pdf = np.exp(-0.5 * ((grid - mu) / sd) ** 2) / (sd * np.sqrt(2 * np.pi))
        ax.plot(grid, pdf * clipped.size * (x_hi - x_lo) / bins,
                color=INK["normal"], linewidth=2.0, zorder=4)
        normal_drawn = True

    month_n = 0
    if month_values is not None:
        m, _, _ = clip_with_overflow(month_values, x_lo, x_hi)
        month_n = int(m.size)
        if m.size:
            ax.hist(m, bins=bins, range=(x_lo, x_hi), histtype="step",
                    color=INK["month"], linewidth=2.0)

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

    # The count that matters, placed against the bound it belongs to and on
    # the OUTSIDE of it, so the two numbers add up to the subtitle. These are
    # breaches of the band -- deliberately not the same quantity as the
    # off-scale counts below, which are an artefact of trimming the axis.
    for count, bound, side in ((n_low, lo, "left"), (n_high, hi, "right")):
        if count and np.isfinite(bound):
            ax.annotate(f"{count:,} outside", xy=(bound, 0.46),
                        xycoords=("data", "axes fraction"),
                        xytext=(-6 if side == "left" else 6, 0),
                        textcoords="offset points",
                        ha="right" if side == "left" else "left",
                        color=INK["text"], fontsize=8.5, fontweight="bold",
                        zorder=6,
                        bbox=dict(facecolor=INK["surface"], edgecolor="none",
                                  boxstyle="round,pad=0.25", alpha=0.92))

    ax.set_xlim(x_lo, x_hi)
    ax.set_xlabel("performance (spreads)", color=INK["text_secondary"])
    ax.set_ylabel("orders", color=INK["text_secondary"])
    ax.set_title(title, color=INK["text"], fontsize=11, loc="left", pad=24)
    # The count is the headline: it is what the band costs to review.
    ax.annotate(subtitle, xy=(0, 1.015), xycoords="axes fraction",
                color=INK["text"], fontsize=9.5, fontweight="bold")

    pctile = float(band_row.get("percentile", float("nan")))
    handles = [Patch(facecolor=INK["hist"], edgecolor="none",
                     label=f"orders (n={x.size:,})")]
    if month_n:
        handles.append(Line2D([0], [0], color=INK["month"], lw=2,
                              label=f"scored month (n={month_n:,})"))
    if normal_drawn:
        handles.append(Line2D([0], [0], color=INK["normal"], lw=2,
                              label="the normal the rule assumes"))
    handles += [
        Line2D([0], [0], color=INK["sigma"], lw=2,
               label="mean ± k·sd"),
        Line2D([0], [0], color=INK["pct"], lw=2,
               label=f"P{pctile:.1f} / P{100 - pctile:.1f}"),
        Line2D([0], [0], color=INK["text_secondary"], lw=2, ls="-",
               label="solid = the bound in force"),
        Line2D([0], [0], color=INK["text_secondary"], lw=1.5, ls="--",
               label="dashed = the other candidate"),
    ]
    ax.legend(handles=handles, frameon=False, fontsize=8, ncol=3,
              loc="upper left", bbox_to_anchor=(0, -0.13),
              labelcolor=INK["text_secondary"], handlelength=1.8,
              columnspacing=1.6)

    spread = float(band_row.get("spread_bps_median", float("nan")))
    in_bps = (f"   =  [{lo * spread:+.0f}, {hi * spread:+.0f}] bps "
              f"at the median spread of {spread:.1f} bps"
              if np.isfinite(spread) and np.isfinite(lo) and np.isfinite(hi)
              else "")
    # The axis is trimmed, so say how many observations were clamped into the
    # end bins. This is NOT the breach count -- the axis is wider than the
    # band, so an order can be outside the band and still on scale.
    off_scale = ("" if not (below or above) else
                 f"   axis trimmed: {below + above:,} order(s) drawn in the "
                 f"end bins ({below:,} left, {above:,} right)")
    caption = (f"k={float(band_row.get('k', float('nan'))):.2f}   "
               f"band [{lo:.2f}, {hi:.2f}] spreads{in_bps}\n"
               f"bound by: low = {lo_binds or 'none'}, "
               f"high = {hi_binds or 'none'}{off_scale}")
    ax.annotate(caption, xy=(0, -0.34), xycoords="axes fraction",
                color=INK["text_secondary"], fontsize=8)
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


def qq(values, out_path: str, title: str, k: float = 4.0,
       max_points: int = 6000) -> str:
    """Normal QQ plot: the single clearest picture of a non-Gaussian tail.

    Both axes are in standard deviations, so the 45-degree line is "the data
    is normal". Fat tails bend AWAY from that line at both ends -- the extreme
    order statistics sit further out than a Gaussian would put them -- and the
    size of that departure at +/- k is exactly the coverage shortfall the band
    suffers. Guides are drawn at +/- k so the reader can see where the rule's
    bound falls relative to where the data actually goes.

    A book with 18,000 orders makes 18,000 markers, which is a heavy PNG and
    an unreadable smear. The body is subsampled; the tails never are, because
    the tails are the entire point.
    """
    a = np.asarray(values, dtype=float).ravel()
    a = np.sort(a[np.isfinite(a)])
    fig, ax = _fig(width=6.4, height=6.0)

    if a.size < 3:
        ax.annotate("too few orders for a QQ plot", xy=(0.5, 0.5),
                    xycoords="axes fraction", ha="center",
                    color=INK["text_secondary"], fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
        return _save(fig, out_path)

    from perfthreshold import normality

    z = (a - a.mean()) / a.std(ddof=1)
    theory = normality.theoretical_quantiles(a.size)

    if a.size > max_points:
        # Keep every point in the tails; thin only the crowded middle.
        edge = max(50, max_points // 10)
        idx = np.unique(np.concatenate([
            np.arange(edge),
            np.linspace(edge, a.size - edge - 1,
                        max_points - 2 * edge).astype(int),
            np.arange(a.size - edge, a.size)]))
    else:
        idx = np.arange(a.size)

    # Framing the view. A single order at -40 sd would otherwise set the axes
    # and squash the informative S-curve into a sliver, so the view is sized
    # to the theoretical range and the bulk of the observed tail. Points past
    # it are counted and named rather than silently dropped -- and the most
    # extreme value is worth printing anyway, being the sharpest single number
    # on the chart.
    lim = float(max(np.abs(theory).max(),
                    np.percentile(np.abs(z), 99.9))) * 1.15
    off = int(np.count_nonzero(np.abs(z) > lim))
    ax.plot([-lim, lim], [-lim, lim], color=INK["normal"], linewidth=2.0,
            zorder=3, label="if the data were normal")
    ax.scatter(theory[idx], z[idx], s=6, color=INK["sigma"], alpha=0.55,
               linewidths=0, zorder=4, label=f"observed (n={a.size:,})")

    for bound in (-k, k):
        ax.axvline(bound, color=INK["text_secondary"], linewidth=1.0,
                   linestyle=":", zorder=2)
    ax.annotate(f"±{k:g} sd", xy=(k, 0.985),
                xycoords=("data", "axes fraction"), xytext=(4, 0),
                textcoords="offset points", va="top",
                color=INK["text_secondary"], fontsize=8)

    s = normality.stats(a, k=k)
    ax.set_title(title, color=INK["text"], fontsize=11, loc="left", pad=24)
    ax.annotate(
        f"beyond ±{k:g} sd: {s['observed_beyond']:,} observed vs "
        f"{s['expected_beyond']:.1f} expected if normal "
        f"({s['tail_ratio']:.0f}x) — {s['verdict']}",
        xy=(0, 1.015), xycoords="axes fraction", color=INK["text"],
        fontsize=9.5, fontweight="bold")

    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("normal quantile (sd)", color=INK["text_secondary"])
    ax.set_ylabel("observed quantile (sd)", color=INK["text_secondary"])
    ax.legend(frameon=False, fontsize=8, loc="lower right",
              labelcolor=INK["text_secondary"])
    if off:
        ax.annotate(
            f"{off:,} order(s) beyond this view; most extreme "
            f"{z[np.argmax(np.abs(z))]:+.1f} sd",
            xy=(0.03, 0.97), xycoords="axes fraction", ha="left", va="top",
            color=INK["text"], fontsize=8,
            bbox=dict(facecolor=INK["surface"], edgecolor="none",
                      boxstyle="round,pad=0.3", alpha=0.92))
    ax.annotate(
        f"skew {s['skew']:+.2f}   excess kurtosis {s['excess_kurtosis']:+.2f}"
        f"   sd/robust {s['sd_over_mad']:.2f}",
        xy=(0, -0.13), xycoords="axes fraction",
        color=INK["text_secondary"], fontsize=8)
    return _save(fig, out_path)
