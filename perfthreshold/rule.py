"""The band rule, and nothing else.

Deliberately pure: numpy in, dict out. No pandas, no config, no filesystem.
The arithmetic that matters most -- does MAX pick the right term, does the
percentile land where you think -- is the part that must be testable against
hand-computed arrays with no fixtures and no data file, so it lives alone.

The rule, per side:

    hi = MAX(mean + k*sd,  P(percentile))
    lo = MIN(mean - k*sd,  P(100 - percentile))

Two properties are the reason it is written this way.

IT IS PER SIDE. Execution performance is skewed -- a book misses badly far
more often than it beats badly -- so forcing both tails through one candidate
makes the band wrong on at least one of them. Each side takes whichever of
its own two candidates is wider, and the two can bind differently.

THE SIGMA TERM IS LITERAL. `mean + k*sd`, nothing solved, nothing adjusted,
so that "mean plus four sigma" is visibly what it says.

Both `hi_binds` and `lo_binds` record which candidate won. On a fat-tailed
book sd is inflated by the very orders the band exists to catch, so the sigma
term widens to swallow them and the percentile floor may never bind at all --
the rule degenerating quietly to pure k-sigma. That has to be visible in the
output rather than inferred from two near-identical columns.
"""

from __future__ import annotations

import numpy as np

# 1 / Phi^-1(0.75). Makes the scaled MAD a consistent estimator of sigma
# UNDER NORMALITY -- which is the point of reporting it. On genuinely normal
# data the classical and robust scales agree, so any gap between them IS the
# non-normality, expressed in the band's own units.
MAD_TO_SIGMA = 1.4826

IN_RANGE = "IN_RANGE"
OUT_LOW = "OUT_LOW"
OUT_HIGH = "OUT_HIGH"
NO_BAND = "NO_BAND"
FLAGGED = frozenset({OUT_LOW, OUT_HIGH})

SIGMA = "sigma"
PERCENTILE = "percentile"


def _clean(x) -> np.ndarray:
    """Finite values only. NaN and +/-inf are dropped, never propagated."""
    a = np.asarray(x, dtype=float).ravel()
    return a[np.isfinite(a)]


def estimates(x) -> dict:
    """Classical and robust centre/scale for one array.

    Both are always computed. Only the classical pair reaches the band; the
    robust pair is a diagnostic that says how far from normal the cell is.
    """
    a = _clean(x)
    n = int(a.size)
    if n == 0:
        return {"n": 0, "mean": np.nan, "sd": np.nan,
                "median": np.nan, "mad_sigma": np.nan}
    mean = float(np.mean(a))
    # ddof=1: the sample standard deviation. With n == 1 there is no spread to
    # estimate, and 0.0 collapses the band onto the point rather than raising.
    sd = float(np.std(a, ddof=1)) if n > 1 else 0.0
    median = float(np.median(a))
    mad_sigma = float(np.median(np.abs(a - median)) * MAD_TO_SIGMA)
    return {"n": n, "mean": mean, "sd": sd,
            "median": median, "mad_sigma": mad_sigma}


def bounds(x, k: float, percentile: float = 99.5) -> dict:
    """Both candidates per side, the winner, and which one won."""
    a = _clean(x)
    est = estimates(a)
    out = dict(est)
    out.update({"p_hi": np.nan, "p_lo": np.nan,
                "sigma_hi": np.nan, "sigma_lo": np.nan,
                "hi": np.nan, "lo": np.nan,
                "hi_binds": "", "lo_binds": "",
                "k": float(k), "percentile": float(percentile)})
    if est["n"] == 0:
        return out

    k = float(k)
    sigma_hi = est["mean"] + k * est["sd"]
    sigma_lo = est["mean"] - k * est["sd"]
    p_hi = float(np.percentile(a, percentile))
    p_lo = float(np.percentile(a, 100.0 - percentile))

    hi = max(sigma_hi, p_hi)
    lo = min(sigma_lo, p_lo)

    out.update({
        "p_hi": p_hi, "p_lo": p_lo,
        "sigma_hi": float(sigma_hi), "sigma_lo": float(sigma_lo),
        "hi": float(hi), "lo": float(lo),
        "hi_binds": SIGMA if hi == sigma_hi else PERCENTILE,
        "lo_binds": SIGMA if lo == sigma_lo else PERCENTILE,
    })
    return out


def zone(value, lo, hi) -> str:
    """Where one observation sits relative to a band.

    Bounds are INCLUSIVE: an order landing exactly on the edge is in range.
    A strict inequality would make the monthly flag count depend on
    floating-point luck at the boundary.
    """
    if lo is None or hi is None:
        return NO_BAND
    v, lo, hi = float(value), float(lo), float(hi)
    if not np.isfinite(v) or not np.isfinite(lo) or not np.isfinite(hi):
        return NO_BAND
    if v < lo:
        return OUT_LOW
    if v > hi:
        return OUT_HIGH
    return IN_RANGE


def count_flags(x, lo, hi) -> int:
    """How many observations fall outside the band. Undefined band flags none."""
    a = _clean(x)
    if a.size == 0 or lo is None or hi is None:
        return 0
    lo, hi = float(lo), float(hi)
    if not np.isfinite(lo) or not np.isfinite(hi):
        return 0
    return int(np.count_nonzero((a < lo) | (a > hi)))
