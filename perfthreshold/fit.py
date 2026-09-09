"""Fit a band per cell, and apply a band table to orders.

`apply` is the single place a band ever meets an order. Calibration, the split
report and scoring all go through it, which is what stops an in-sample count
and an out-of-sample one from being produced by two subtly different code
paths -- the kind of divergence that shows up as an unexplained change in the
flag rate rather than as a failure.

THE FALLBACK CONTRACT. A cell below min_cell_n cannot support a percentile
estimate: at n = 100 the 99.5th percentile is interpolated between the first
and second worst order, and sd is biased low because the tail has probably not
been sampled yet -- which makes the band too NARROW and the thin cell look
like the worst offender in the book purely because it is small.

So a thin cell copies its benchmark-pooled parent's bounds and records that it
did. If the parent is itself too thin -- always the case under scope=all,
where the cell IS the pooled band -- the cell is left unfitted and its orders
score NO_BAND. Banding on too little evidence is worse than not banding.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from perfthreshold import groups, rule, schema

BAND_LO = "band_lo"
BAND_HI = "band_hi"
ZONE = "zone"

BAND_COLS = [
    "cell_key", schema.BENCHMARK, schema.MARKET_GROUP, "n",
    "spread_bps_median", "spread_bps_mean",
    "mean", "sd", "median", "mad_sigma",
    "sigma_lo", "sigma_hi", "p_lo", "p_hi",
    "lo", "hi", "lo_binds", "hi_binds",
    "k", "percentile", "fitted", "fallback_from",
]


@dataclass
class FitResult:
    bands: pd.DataFrame
    medians: dict[str, dict[str, float]] = field(default_factory=dict)
    k: float = 4.0
    percentile: float = 99.5
    min_cell_n: int = 2000


def _row(cell: str, benchmark: str, market_group: str, b: dict,
         *, fitted: bool, fallback_from: str,
         spread_median: float = float("nan"),
         spread_mean: float = float("nan")) -> dict:
    return {
        "cell_key": cell, schema.BENCHMARK: benchmark,
        schema.MARKET_GROUP: market_group, "n": b["n"],
        # The metric is unitless (spreads). Carrying the cell's own spread is
        # what lets a bound of 4.33 spreads be read back as ~35 bps -- without
        # it the band cannot be translated into money by anyone reading it.
        "spread_bps_median": spread_median, "spread_bps_mean": spread_mean,
        "mean": b["mean"], "sd": b["sd"],
        "median": b["median"], "mad_sigma": b["mad_sigma"],
        "sigma_lo": b["sigma_lo"], "sigma_hi": b["sigma_hi"],
        "p_lo": b["p_lo"], "p_hi": b["p_hi"],
        "lo": b["lo"], "hi": b["hi"],
        "lo_binds": b["lo_binds"], "hi_binds": b["hi_binds"],
        "k": b["k"], "percentile": b["percentile"],
        "fitted": fitted, "fallback_from": fallback_from,
    }


def fit_cells(df: pd.DataFrame, k: float, percentile: float = 99.5,
              min_cell_n: int = 2000) -> FitResult:
    """One band per cell present in `df`, with thin cells handled."""
    if len(df) == 0:
        return FitResult(bands=pd.DataFrame(columns=BAND_COLS), medians={},
                         k=float(k), percentile=float(percentile),
                         min_cell_n=int(min_cell_n))

    # Parent bands first: one per benchmark, over every in-scope row of that
    # benchmark, so a thin cell always has something well-supported to inherit.
    parents: dict[str, dict] = {}
    for bench, g in df.groupby(schema.BENCHMARK, observed=True):
        parents[str(bench)] = rule.bounds(
            g[schema.METRIC].to_numpy(dtype=float), k=k, percentile=percentile)

    rows, medians = [], {}
    for cell, g in df.groupby(schema.CELL_KEY, observed=True):
        cell = str(cell)
        bench, group_name = groups.split_key(cell)
        own = rule.bounds(g[schema.METRIC].to_numpy(dtype=float),
                          k=k, percentile=percentile)

        # Always the cell's OWN spread, even when the band is inherited: the
        # bounds may come from the parent, but the orders are these orders.
        if schema.SPREAD_BPS in g.columns and g[schema.SPREAD_BPS].notna().any():
            spread_median = float(g[schema.SPREAD_BPS].median())
            spread_mean = float(g[schema.SPREAD_BPS].mean())
        else:
            spread_median = spread_mean = float("nan")
        spreads = {"spread_median": spread_median, "spread_mean": spread_mean}

        if own["n"] >= min_cell_n:
            rows.append(_row(cell, bench, group_name, own,
                             fitted=True, fallback_from="", **spreads))
        else:
            parent = parents.get(bench, {})
            pkey = groups.parent_key(cell)
            # A parent that is itself thin is not a rescue. The only honest
            # answer then is no band at all.
            if parent.get("n", 0) >= min_cell_n and parent["n"] > own["n"]:
                inherited = dict(parent)
                inherited["n"] = own["n"]   # the cell's own size, not the pool's
                rows.append(_row(cell, bench, group_name, inherited,
                                 fitted=False, fallback_from=pkey, **spreads))
            else:
                blank = rule.bounds(np.array([]), k=k, percentile=percentile)
                blank["n"] = own["n"]
                rows.append(_row(cell, bench, group_name, blank,
                                 fitted=False, fallback_from="", **spreads))

        medians[cell] = {
            f: (float(g[f].median()) if f in g.columns and g[f].notna().any()
                else float("nan"))
            for f in schema.REFERENCE_FEATURES
        }

    bands = pd.DataFrame(rows, columns=BAND_COLS).sort_values(
        "cell_key").reset_index(drop=True)
    return FitResult(bands=bands, medians=medians, k=float(k),
                     percentile=float(percentile), min_cell_n=int(min_cell_n))


def apply(df: pd.DataFrame, bands: pd.DataFrame) -> pd.DataFrame:
    """Attach each row's bounds and its zone. Unknown cell -> NO_BAND."""
    out = df.copy()
    if len(out) == 0:
        out[BAND_LO] = pd.Series(dtype=float)
        out[BAND_HI] = pd.Series(dtype=float)
        out[ZONE] = pd.Series(dtype=object)
        return out

    lookup = (bands.set_index("cell_key")[["lo", "hi"]]
              if len(bands) else pd.DataFrame(columns=["lo", "hi"]))
    lo = out[schema.CELL_KEY].map(lookup["lo"]) if len(lookup) else np.nan
    hi = out[schema.CELL_KEY].map(lookup["hi"]) if len(lookup) else np.nan
    out[BAND_LO] = pd.to_numeric(pd.Series(lo, index=out.index),
                                 errors="coerce")
    out[BAND_HI] = pd.to_numeric(pd.Series(hi, index=out.index),
                                 errors="coerce")

    v = out[schema.METRIC].to_numpy(dtype=float)
    lo_a = out[BAND_LO].to_numpy(dtype=float)
    hi_a = out[BAND_HI].to_numpy(dtype=float)
    defined = np.isfinite(v) & np.isfinite(lo_a) & np.isfinite(hi_a)

    zone = np.full(len(out), rule.NO_BAND, dtype=object)
    zone[defined & (v < lo_a)] = rule.OUT_LOW
    zone[defined & (v > hi_a)] = rule.OUT_HIGH
    zone[defined & (v >= lo_a) & (v <= hi_a)] = rule.IN_RANGE
    out[ZONE] = zone
    return out


def flag_count(df: pd.DataFrame, bands: pd.DataFrame) -> int:
    """Total orders outside their band. NO_BAND rows are not flags."""
    if len(df) == 0:
        return 0
    return int(apply(df, bands)[ZONE].isin(list(rule.FLAGGED)).sum())
