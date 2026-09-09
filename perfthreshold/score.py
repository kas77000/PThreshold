"""Apply a frozen band to one month, and refuse the ways it can go wrong.

Three refusals, each guarding a failure that produces a clean-looking,
completely meaningless result:

LEAKAGE. Scoring a month the band was fitted on gives a circular flag rate
that looks entirely normal. It is the one mistake that voids the whole
out-of-sample exercise while leaving no trace, so it raises rather than warns.

UNIT MISMATCH. A band frozen in spreads and one frozen in bps would sit side
by side in a summary table looking comparable.

SCOPE MISMATCH. A band fitted per declared group, applied to rows resolved as
one pool, gives every row NO_BAND and reports zero outliers -- a tidy result
that means nothing.

And one diagnostic. When a month flags fifteen instead of five, there are two
entirely different explanations -- the book got harder, or execution got worse
-- and they call for opposite actions. Comparing this month's reference-feature
medians against the ones stamped at fit time is what separates them.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from perfthreshold import fit, rule, schema


class GuardError(ValueError):
    """A refusal. Never downgraded to a warning."""


DRIFT_COLS = ["feature", "fit_median", "month_median", "pct_change", "warn"]

OUTLIER_COLS = [
    schema.ORDER_ID, schema.ORDER_DATE, schema.ALGO, schema.BENCHMARK,
    schema.MARKET, schema.SYMBOL, schema.CELL_KEY, schema.METRIC,
    fit.BAND_LO, fit.BAND_HI, fit.ZONE, "excess",
] + list(schema.DIAGNOSTICS)


@dataclass
class ScoreResult:
    scored: pd.DataFrame
    outliers: pd.DataFrame
    drift: pd.DataFrame
    counts: dict = field(default_factory=dict)
    month: str = ""


def windows_overlap(a_lo, a_hi, b_lo, b_hi) -> bool:
    """Do two date windows intersect? False whenever any bound is unknown.

    False on unknown is deliberate: without dates the guard cannot fire, and
    that is exactly why order_date is effectively required.
    """
    if a_lo is None or a_hi is None or b_lo is None or b_hi is None:
        return False
    a_lo, a_hi = pd.Timestamp(a_lo), pd.Timestamp(a_hi)
    b_lo, b_hi = pd.Timestamp(b_lo), pd.Timestamp(b_hi)
    return bool(a_lo <= b_hi and b_lo <= a_hi)


def month_label(df: pd.DataFrame) -> str:
    if len(df) == 0 or schema.ORDER_DATE not in df.columns:
        return "unknown"
    d = df[schema.ORDER_DATE].dropna()
    if d.empty:
        return "unknown"
    lo, hi = d.min(), d.max()
    if (lo.year, lo.month) == (hi.year, hi.month):
        return lo.strftime("%Y-%m")
    return f"{lo.strftime('%Y-%m')}_{hi.strftime('%Y-%m')}"


def check_guards(band_file, df: pd.DataFrame, scope: str,
                 metric_units: str) -> None:
    if str(metric_units) != str(band_file.metric_units):
        raise GuardError(
            f"Band was fitted in units '{band_file.metric_units}' but is being "
            f"applied to a metric in '{metric_units}'. Bands are not "
            f"interchangeable across units.")

    if str(scope) != str(band_file.scope):
        raise GuardError(
            f"Band was fitted with scope '{band_file.scope}' but is being "
            f"applied with scope '{scope}'. The cell keys would not match and "
            f"every order would score NO_BAND.")

    d = (df[schema.ORDER_DATE].dropna() if schema.ORDER_DATE in df.columns
         else pd.Series(dtype="datetime64[ns]"))
    lo = None if d.empty else d.min()
    hi = None if d.empty else d.max()
    if windows_overlap(band_file.fit_start, band_file.fit_end, lo, hi):
        raise GuardError(
            f"Scored window {lo.date()}..{hi.date()} overlaps the band's fit "
            f"window {band_file.fit_start}..{band_file.fit_end}. Scoring a "
            f"month the band already saw gives a circular flag rate.")


def drift(df: pd.DataFrame, band_file, threshold_pct: float = 25.0
          ) -> pd.DataFrame:
    """Fit-time vs this month's reference-feature medians."""
    # One baseline for the whole month: the medians were stamped per cell, and
    # the simple median across cells that have a value is the natural summary
    # of "what the book looked like when the band was fitted".
    baseline: dict[str, float] = {}
    for feature in schema.REFERENCE_FEATURES:
        vals = [m.get(feature) for m in band_file.medians.values()]
        vals = [v for v in vals if v is not None and np.isfinite(v)]
        baseline[feature] = float(np.median(vals)) if vals else np.nan

    rows = []
    for feature in schema.REFERENCE_FEATURES:
        fit_med = baseline.get(feature, np.nan)
        month_med = (float(df[feature].median())
                     if feature in df.columns and df[feature].notna().any()
                     else np.nan)
        if np.isfinite(fit_med) and fit_med != 0 and np.isfinite(month_med):
            pct = 100.0 * (month_med - fit_med) / abs(fit_med)
        else:
            pct = np.nan
        rows.append({
            "feature": feature, "fit_median": fit_med,
            "month_median": month_med, "pct_change": pct,
            "warn": bool(np.isfinite(pct) and abs(pct) > threshold_pct),
        })
    return pd.DataFrame(rows, columns=DRIFT_COLS)


def score_month(df: pd.DataFrame, band_file, scope: str, metric_units: str,
                drift_threshold_pct: float = 25.0) -> ScoreResult:
    check_guards(band_file, df, scope=scope, metric_units=metric_units)

    scored = fit.apply(df, band_file.bands)

    flagged = scored[scored[fit.ZONE].isin(list(rule.FLAGGED))].copy()
    if len(flagged):
        v = flagged[schema.METRIC].to_numpy(dtype=float)
        hi = flagged[fit.BAND_HI].to_numpy(dtype=float)
        lo = flagged[fit.BAND_LO].to_numpy(dtype=float)
        # How far outside, in the metric's own units. Ranking by this puts the
        # order most worth explaining at the top of the queue.
        flagged["excess"] = np.where(
            flagged[fit.ZONE].to_numpy() == rule.OUT_HIGH, v - hi, lo - v)
        flagged = flagged.sort_values("excess", ascending=False)
    else:
        flagged["excess"] = pd.Series(dtype=float)

    cols = [c for c in OUTLIER_COLS if c in flagged.columns]
    outliers = flagged[cols].reset_index(drop=True)

    by_cell = (flagged[schema.CELL_KEY].value_counts().to_dict()
               if len(flagged) else {})
    counts = {
        "orders": int(len(scored)),
        "flagged": int(len(flagged)),
        "no_band": int((scored[fit.ZONE] == rule.NO_BAND).sum()),
        "out_low": int((scored[fit.ZONE] == rule.OUT_LOW).sum()),
        "out_high": int((scored[fit.ZONE] == rule.OUT_HIGH).sum()),
        "by_cell": {str(k): int(v) for k, v in by_cell.items()},
    }

    return ScoreResult(scored=scored, outliers=outliers,
                       drift=drift(df, band_file, drift_threshold_pct),
                       counts=counts, month=month_label(df))
