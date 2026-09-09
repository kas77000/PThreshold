"""A synthetic order book with the shape the real one has.

Two properties are non-negotiable, because the design is built around them.

FAT TAILS. The metric is drawn from a Student-t, not a normal. On a Gaussian
book `mean + 4*sd` and P99.5 land in almost the same place and the question
the whole calibration exists to answer -- which term binds, and what does k
cost -- never arises. A normal generator would make every test pass for the
wrong reason.

A SKEW. Real execution misses badly more often than it beats badly, so the
low tail is heavier. That is what makes the per-side rule worth having.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Per-benchmark (centre, scale, degrees of freedom). Lower df = fatter tails.
# The two centres differ so that pooling them is visibly wrong, which is what
# makes the benchmark split testable.
_PROFILE = {
    "VWAP": (-0.05, 0.45, 3.5),
    "TWAP": (-0.18, 0.60, 3.0),
}

# Market scale multipliers. Wider, thinner books have a noisier spread, so the
# ratio's tails are fatter even after normalisation -- the residual structure
# that a market split would exist to capture.
_MARKET_SCALE = {"HK": 0.9, "JP": 0.85, "AU": 1.0, "IN": 1.6,
                 "KR": 1.2, "TW": 1.15, "SG": 0.95, "TH": 1.7}


def make_book(n_per_month: int = 400, months: int = 12,
              start: str = "2025-07-01", seed: int = 0,
              benchmarks: tuple[str, ...] = ("VWAP", "TWAP"),
              markets: tuple[str, ...] = ("HK", "JP", "AU", "IN"),
              unmapped_rows: int = 0) -> pd.DataFrame:
    """One frame with RAW extract column names, so it exercises the loader."""
    rng = np.random.default_rng(seed)
    rows = []
    first = pd.Timestamp(start).normalize()
    oid = 0

    for m in range(months):
        month_start = first + pd.DateOffset(months=m)
        days_in_month = month_start.days_in_month
        for _ in range(n_per_month):
            strat = benchmarks[rng.integers(len(benchmarks))]
            mkt = markets[rng.integers(len(markets))]
            centre, scale, df_t = _PROFILE.get(strat, _PROFILE["VWAP"])
            scale = scale * _MARKET_SCALE.get(mkt, 1.0)

            # Student-t for the fat tails; the cubic term adds a left skew so
            # the two sides of the band are genuinely different problems.
            t = float(rng.standard_t(df_t))
            perf = centre + scale * t - 0.03 * (t ** 3) / 10.0

            spread_bps = float(np.abs(rng.normal(8.0, 3.0)) + 1.0)
            oid += 1
            rows.append({
                "aggrTgtId": f"O{oid:07d}",
                "Strategy": strat,
                "Sym": f"{rng.integers(1, 9999):04d} {mkt}",
                "Date": month_start + pd.Timedelta(
                    days=int(rng.integers(0, days_in_month))),
                "ePvwap/Sprd": perf,
                "Pvwap": perf * spread_bps,
                "Sprd": spread_bps,
                "%Adv": float(np.abs(rng.normal(5.0, 4.0))),
                "Vol": float(np.abs(rng.normal(25.0, 8.0))),
                "PR": float(np.clip(rng.normal(12.0, 6.0), 0.1, 90.0)),
                "Dur": float(np.abs(rng.normal(120.0, 60.0)) + 1.0),
                "$Mln": float(np.abs(rng.normal(2.0, 1.5)) + 0.01),
                "#Shares": float(np.abs(rng.normal(50.0, 30.0)) + 1.0),
                "Side": "buy" if rng.random() < 0.5 else "sell",
                "%POST": float(np.clip(rng.normal(55.0, 20.0), 0.0, 100.0)),
                "%OPEN": float(np.clip(rng.normal(3.0, 3.0), 0.0, 100.0)),
                "%CLOSE": float(np.clip(rng.normal(8.0, 7.0), 0.0, 100.0)),
                "Rev30min": float(rng.normal(0.0, 12.0)),
            })

    # Strategies the config does not map. They must be excluded and named, so
    # every loader test needs a way to plant them.
    for _ in range(unmapped_rows):
        oid += 1
        rows.append({
            "aggrTgtId": f"O{oid:07d}", "Strategy": "PART",
            "Sym": f"{rng.integers(1, 9999):04d} HK", "Date": first,
            "ePvwap/Sprd": float(rng.normal()), "Pvwap": 1.0, "Sprd": 8.0,
            "%Adv": 5.0, "Vol": 25.0, "PR": 12.0, "Dur": 120.0,
            "$Mln": 2.0, "#Shares": 50.0, "Side": "buy",
            "%POST": 55.0, "%OPEN": 3.0, "%CLOSE": 8.0, "Rev30min": 0.0,
        })

    return pd.DataFrame(rows)
