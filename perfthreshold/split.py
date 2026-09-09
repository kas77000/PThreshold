"""Evidence for splitting a market out of the pool, stated in review workload.

"This market's distribution differs from pooled by a KS statistic of 0.07" is
a fact nobody can act on. "Pooling costs this market 31 extra reviews a year"
is the same fact in the currency this project is judged in, and it can be held
up against the review budget directly.

So the question asked of every market is: how many orders does the POOLED band
put on a desk that the market's OWN band would not? That difference is the
evidence, and it is reported per market and ranked.

Spread normalisation is the reason pooling is the default at all -- it puts a
wide Indian small cap and a tight Japanese large cap on one scale before the
band is fitted. What survives normalisation is structural: tick-size regimes,
closing-auction dominance, and thin books where the spread itself is noisy and
so fattens the tails of the ratio. This report measures whether any of that
actually bites, rather than assuming it does.

Declaring a grouping (config.MARKET_GROUPS) and testing one (this report) stay
separate on purpose. Config decides what gets fitted; this says whether the
decision was justified.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from perfthreshold import rule, schema

VERDICT_SPLIT = "split"
VERDICT_POOL = "pool"
VERDICT_THIN = "too few orders"

SPLIT_COLS = [
    schema.BENCHMARK, "market", "n",
    "pooled_lo", "pooled_hi", "own_lo", "own_hi",
    "pooled_flags", "own_flags", "excess_flags",
    "pooled_rate_pct", "own_rate_pct", "verdict",
]


def report(df: pd.DataFrame, k: float, percentile: float = 99.5,
           min_market_n: int = 2000, ratio_threshold: float = 2.0,
           min_excess: int = 12) -> pd.DataFrame:
    """One row per (benchmark, market): what pooling costs that market."""
    if len(df) == 0:
        return pd.DataFrame(columns=SPLIT_COLS)

    rows = []
    for bench, bench_rows in df.groupby(schema.BENCHMARK, observed=True):
        pooled = rule.bounds(bench_rows[schema.METRIC].to_numpy(dtype=float),
                             k=k, percentile=percentile)

        for market, g in bench_rows.groupby(schema.MARKET, observed=True):
            x = g[schema.METRIC].to_numpy(dtype=float)
            n = int(np.isfinite(x).sum())
            pooled_flags = rule.count_flags(x, pooled["lo"], pooled["hi"])

            if n < min_market_n:
                rows.append({
                    schema.BENCHMARK: str(bench), "market": str(market),
                    "n": n,
                    "pooled_lo": pooled["lo"], "pooled_hi": pooled["hi"],
                    "own_lo": np.nan, "own_hi": np.nan,
                    "pooled_flags": pooled_flags, "own_flags": np.nan,
                    "excess_flags": np.nan,
                    "pooled_rate_pct": 100.0 * pooled_flags / n if n else np.nan,
                    "own_rate_pct": np.nan,
                    "verdict": VERDICT_THIN,
                })
                continue

            own = rule.bounds(x, k=k, percentile=percentile)
            own_flags = rule.count_flags(x, own["lo"], own["hi"])
            excess = pooled_flags - own_flags

            # Both conditions must hold. The ratio alone would let 3 pooled
            # flags against 1 own flag -- noise -- read as evidence.
            ratio = (pooled_flags / own_flags) if own_flags else float("inf")
            earns_split = (excess >= min_excess) and (ratio >= ratio_threshold)

            rows.append({
                schema.BENCHMARK: str(bench), "market": str(market), "n": n,
                "pooled_lo": pooled["lo"], "pooled_hi": pooled["hi"],
                "own_lo": own["lo"], "own_hi": own["hi"],
                "pooled_flags": pooled_flags, "own_flags": own_flags,
                "excess_flags": excess,
                "pooled_rate_pct": 100.0 * pooled_flags / n,
                "own_rate_pct": 100.0 * own_flags / n,
                "verdict": VERDICT_SPLIT if earns_split else VERDICT_POOL,
            })

    out = pd.DataFrame(rows, columns=SPLIT_COLS)
    # Thin markets last: they carry no evidence either way.
    out["_thin"] = out["verdict"] == VERDICT_THIN
    out = out.sort_values(["_thin", "excess_flags"],
                          ascending=[True, False]).drop(columns="_thin")
    return out.reset_index(drop=True)
