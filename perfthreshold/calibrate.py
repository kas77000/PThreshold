"""Choosing k against the number of orders it will actually put on a desk.

k = 4 is a statement about a Gaussian. This book is not one, and on a
fat-tailed distribution four sigma might flag forty orders a month or none --
which of those it is cannot be read off the number 4. So k is chosen against a
measured curve rather than assumed, and the curve has to be measured honestly.

LEAVE ONE MONTH OUT. Counting flags on the year the band was fitted on is
circular: those orders shaped the sd that judges them, so the count comes out
flattering, every time, in the same direction. Fitting on the other eleven
months and scoring the twelfth gives twelve genuinely out-of-sample
observations -- and so a RANGE, which is the number that matters. A median of
5 with a worst month of 12 is a different proposition from a median of 5 with
a worst month of 6, and an average alone hides the difference.

The budget is TOTAL across every cell, so k is a single global value. A
badly-behaved group therefore cannot buy itself a wider band; it simply
contributes more of the five, which is the signal you want.

Cost: len(ks) * n_months fits, about 730 for the default grid over a year.
Seconds on a year's file. Deliberately NOT optimised by deriving bounds from
cached per-cell statistics -- that would put the rule's arithmetic in a second
place, and a divergence between the calibrated count and the scored count is
exactly the bug that would be hardest to see.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from perfthreshold import fit, schema

CURVE_COLS = ["k", "n_months", "median_flags", "mean_flags",
              "min_flags", "max_flags", "months_over_target"]


@dataclass
class KChoice:
    k: float | None
    achieved_median: float | None
    reason: str
    reachable: bool


def _month_series(df: pd.DataFrame) -> pd.Series:
    return df[schema.ORDER_DATE].dt.to_period("M").astype(str)


def months(df: pd.DataFrame) -> list[str]:
    """Sorted 'YYYY-MM' labels present in the frame."""
    if len(df) == 0:
        return []
    return sorted(_month_series(df).unique().tolist())


def k_grid(start: float = 2.0, stop: float = 8.0,
           step: float = 0.1) -> list[float]:
    """Inclusive grid, rounded so no value arrives as 3.9999999999."""
    n = int(round((stop - start) / step))
    return [round(start + i * step, 3) for i in range(n + 1)]


def lomo_counts(df: pd.DataFrame, k: float, percentile: float = 99.5,
                min_cell_n: int = 2000) -> dict[str, int]:
    """Flags per month, each month scored by a band that never saw it."""
    labels = months(df)
    if len(labels) < 2:
        raise ValueError(
            "Leave-one-month-out needs at least two months in the fit window; "
            f"found {len(labels)}. Widen the fit file or pass --k explicitly.")

    month = _month_series(df)
    out: dict[str, int] = {}
    for label in labels:
        held = month == label
        trained = fit.fit_cells(df[~held], k=k, percentile=percentile,
                               min_cell_n=min_cell_n)
        out[label] = fit.flag_count(df[held], trained.bands)
    return out


def curve(df: pd.DataFrame, ks: list[float] | None = None,
          percentile: float = 99.5, min_cell_n: int = 2000,
          target: int = 5) -> pd.DataFrame:
    """One row per k: the out-of-sample monthly flag count it implies."""
    ks = k_grid() if ks is None else list(ks)
    rows = []
    for k in ks:
        counts = np.array(list(lomo_counts(
            df, k=k, percentile=percentile, min_cell_n=min_cell_n).values()),
            dtype=float)
        rows.append({
            "k": float(k),
            "n_months": int(counts.size),
            "median_flags": float(np.median(counts)),
            "mean_flags": float(np.mean(counts)),
            "min_flags": int(counts.min()),
            "max_flags": int(counts.max()),
            "months_over_target": int((counts > target).sum()),
        })
    return pd.DataFrame(rows, columns=CURVE_COLS)


def choose_k(curve_df: pd.DataFrame, target: int) -> KChoice:
    """Smallest k whose median out-of-sample monthly count is <= target.

    Smallest rather than largest: every increase in k widens the band and
    therefore hides real orders, so the right k is the least conservative one
    that still fits the review budget.
    """
    if len(curve_df) == 0:
        return KChoice(None, None,
                       "No calibration curve was produced.", False)

    ordered = curve_df.sort_values("k")
    meets = ordered[ordered["median_flags"] <= float(target)]
    if len(meets):
        row = meets.iloc[0]
        return KChoice(
            k=float(row["k"]), achieved_median=float(row["median_flags"]),
            reason=(f"k={row['k']:.2f} gives a median of "
                    f"{row['median_flags']:.1f} flags/month "
                    f"(range {int(row['min_flags'])}-{int(row['max_flags'])} "
                    f"over {int(row['n_months'])} months); target was {target}."),
            reachable=True)

    best = ordered.iloc[-1]
    return KChoice(
        k=None, achieved_median=float(best["median_flags"]),
        reason=(f"No k in the search range reaches {target} flags/month. "
                f"The widest band tried (k={best['k']:.2f}) still gives a "
                f"median of {best['median_flags']:.1f}. Either raise the "
                f"target, widen K_GRID, or accept that this book has more "
                f"genuine outliers than the budget allows."),
        reachable=False)
