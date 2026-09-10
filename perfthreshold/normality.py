"""Is the Gaussian assumption true of this book? Measured, not assumed.

`mean +/- k*sd` carries a coverage claim -- "four sigma covers 99.9937%" --
that is a property of the NORMAL DISTRIBUTION, not of the arithmetic. This
module measures whether that property actually holds, per cell, so the claim
can be supported or withdrawn with evidence rather than argued about.

THE NUMBER THAT COMMUNICATES is not the test statistic. It is
`expected_beyond` against `observed_beyond`: how many orders a normal
distribution says should fall outside mean +/- k*sd, against how many really
do. On a fat-tailed book those differ by one to two orders of magnitude, and
their ratio is the coverage shortfall in a form anyone can read without
knowing what kurtosis is.

WHY THE VERDICT IGNORES THE P-VALUE. Jarque-Bera rejects normality on almost
any large sample -- at n = 18,000 a departure far too small to matter still
produces an overwhelming test statistic. A verdict driven by the p-value would
therefore say "not normal" about a book that is normal enough for the coverage
claim to hold, which is useless. The verdict tracks the tail ratio, which is
the thing the threshold actually depends on. The p-value is reported because
someone will ask for it, not because it decides anything.

No scipy: the normal CDF comes from statistics.NormalDist, and the
Jarque-Bera p-value from the closed form of the chi-square survival function
on two degrees of freedom, exp(-JB/2).
"""

from __future__ import annotations

import math
from statistics import NormalDist

import numpy as np
import pandas as pd

from perfthreshold import schema

_NORM = NormalDist()

CLOSE_TO_NORMAL = "close to normal"
FAT_TAILED = "fat-tailed"
VERY_FAT_TAILED = "very fat-tailed"

# Ratio of observed to normal-expected exceedances beyond mean +/- k*sd.
# Below 2x the coverage claim is roughly honoured; beyond 10x it is not
# meaningfully related to the stated figure at all.
FAT_AT = 2.0
VERY_FAT_AT = 10.0

NORMALITY_COLS = [
    "cell_key", "n", "k",
    "skew", "excess_kurtosis", "sd_over_mad",
    "expected_beyond", "observed_beyond", "tail_ratio",
    "coverage_pct", "coverage_pct_if_normal",
    "jarque_bera", "jb_p_value", "verdict",
]


def _clean(values) -> np.ndarray:
    a = np.asarray(values, dtype=float).ravel()
    return a[np.isfinite(a)]


def normal_two_sided_tail(k: float) -> float:
    """P(|Z| > k) for a standard normal. erfc keeps scipy out of the deps."""
    if not math.isfinite(k):
        return float("nan")
    return math.erfc(abs(float(k)) / math.sqrt(2.0))


def theoretical_quantiles(n: int) -> np.ndarray:
    """Standard-normal order statistics for a QQ plot, n points.

    Uses the (i - 0.5)/n plotting position, which is symmetric and avoids the
    infinite quantile that (i)/n would produce at the last point.
    """
    if n <= 0:
        return np.array([])
    return np.array([_NORM.inv_cdf((i + 0.5) / n) for i in range(n)])


def stats(values, k: float = 4.0) -> dict:
    """Shape diagnostics for one cell, and what they cost in coverage."""
    a = _clean(values)
    n = int(a.size)
    out = {
        "n": n, "k": float(k),
        "skew": np.nan, "excess_kurtosis": np.nan, "sd_over_mad": np.nan,
        "expected_beyond": np.nan, "observed_beyond": 0, "tail_ratio": np.nan,
        "coverage_pct": np.nan,
        "coverage_pct_if_normal": 100.0 * (1.0 - normal_two_sided_tail(k)),
        "jarque_bera": np.nan, "jb_p_value": np.nan,
        "verdict": "too few orders",
    }
    if n < 4:
        return out

    mean = float(np.mean(a))
    sd = float(np.std(a, ddof=1))
    if sd <= 0:
        return out

    z = (a - mean) / sd
    skew = float(np.mean(z ** 3))
    excess_kurtosis = float(np.mean(z ** 4) - 3.0)

    median = float(np.median(a))
    mad_sigma = float(np.median(np.abs(a - median)) * 1.4826)
    sd_over_mad = float(sd / mad_sigma) if mad_sigma > 0 else np.nan

    # The comparison that carries the argument.
    tail_p = normal_two_sided_tail(k)
    expected = n * tail_p
    observed = int(np.count_nonzero(np.abs(z) > k))
    tail_ratio = float(observed / expected) if expected > 0 else np.nan

    # Jarque-Bera; under H0 it is chi-square on 2 df, whose survival function
    # is exactly exp(-x/2).
    jb = n / 6.0 * (skew ** 2 + (excess_kurtosis ** 2) / 4.0)
    jb_p = math.exp(-jb / 2.0) if jb < 1400 else 0.0

    if not np.isfinite(tail_ratio):
        verdict = "undetermined"
    elif tail_ratio >= VERY_FAT_AT:
        verdict = VERY_FAT_TAILED
    elif tail_ratio >= FAT_AT:
        verdict = FAT_TAILED
    else:
        verdict = CLOSE_TO_NORMAL

    out.update({
        "skew": skew, "excess_kurtosis": excess_kurtosis,
        "sd_over_mad": sd_over_mad,
        "expected_beyond": float(expected), "observed_beyond": observed,
        "tail_ratio": tail_ratio,
        "coverage_pct": 100.0 * (1.0 - observed / n),
        "jarque_bera": float(jb), "jb_p_value": float(jb_p),
        "verdict": verdict,
    })
    return out


def report(df: pd.DataFrame, k: float = 4.0) -> pd.DataFrame:
    """One row per cell present in the frame."""
    if len(df) == 0:
        return pd.DataFrame(columns=NORMALITY_COLS)
    rows = []
    for cell, g in df.groupby(schema.CELL_KEY, observed=True):
        row = {"cell_key": str(cell)}
        row.update(stats(g[schema.METRIC].to_numpy(dtype=float), k=k))
        rows.append(row)
    return (pd.DataFrame(rows, columns=NORMALITY_COLS)
            .sort_values("cell_key").reset_index(drop=True))
