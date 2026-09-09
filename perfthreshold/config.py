"""The only file you edit to point this at real data.

Two settings here decide whether every number downstream is meaningful:

ALGO_BENCHMARK   which benchmark family each strategy belongs to. Get this
                 wrong and the band still fits, the curve still looks like a
                 curve, and nothing ever tells you. Run `check` against the
                 real extract and complete the map from what it prints --
                 never from memory.

MARKET_GROUPS    how markets are pooled. The default pools everything, which
                 is defensible only because the metric is already divided by
                 the spread. See split_report.csv before changing it.
"""

from __future__ import annotations

from perfthreshold import schema

# ---------------------------------------------------------------------------
# 1) Column map: raw extract name -> canonical name
# ---------------------------------------------------------------------------
COLUMN_MAP = {
    "aggrTgtId": schema.ORDER_ID,
    "Strategy": schema.ALGO,
    "Sym": schema.SYMBOL,
    "Date": schema.ORDER_DATE,
    "ePvwap/Sprd": schema.METRIC,
    "Pvwap": schema.SLIPPAGE_BPS,
    "Sprd": schema.SPREAD_BPS,
    "%Adv": schema.PCT_ADV,
    "Vol": schema.VOLATILITY,
    "PR": schema.PARTICIPATION,
    "Dur": schema.DURATION_MIN,
    "$Mln": schema.NOTIONAL,
    "#Shares": schema.QUANTITY,
    "Side": schema.SIDE,
    "%POST": schema.PASSIVE_FILL,
    "%OPEN": schema.OPEN_PCT,
    "%CLOSE": schema.CLOSE_PCT,
    "Rev30min": schema.REVERSION,
}

# The banded metric, taken from the extract already divided by the spread.
# This project never performs the division itself.
METRIC_COLUMN = "ePvwap/Sprd"
METRIC_UNITS = "spreads"

# ---------------------------------------------------------------------------
# 2) Strategy -> benchmark family
# ---------------------------------------------------------------------------
# Keys are matched upper-cased and stripped. A strategy absent from this map
# is EXCLUDED and reported by name -- it is never defaulted into a family.
# The desk calls the TWAP family "TMX"; both spellings map to TWAP so the
# label in the output is the benchmark, not the vendor's product name.
ALGO_BENCHMARK = {
    "VWAP": "VWAP",
    "TWAP": "TWAP",
    "TMX": "TWAP",
}

# ---------------------------------------------------------------------------
# 3) Market grouping
# ---------------------------------------------------------------------------
# "*" is a wildcard: every market falls into that group. The v1 default pools
# the whole book, which the spread-normalised metric makes legal.
MARKET_GROUPS: dict[str, str | list[str]] = {"ALL": "*"}

# Example of a declared split, kept here as documentation:
# MARKET_GROUPS = {
#     "APAC_TIGHT": ["HK", "JP", "AU", "SG"],
#     "APAC_WIDE":  ["IN", "KR", "TW", "TH", "ID", "PH", "MY"],
# }

# ---------------------------------------------------------------------------
# 4) Rule knobs
# ---------------------------------------------------------------------------
PERCENTILE = 99.5        # upper; the lower bound always mirrors to 100 - this
DEFAULT_K = 4.0          # used when --k is given without a value, and as the
                         # fallback if a target search is impossible
DEFAULT_TARGET = 5       # flags per month, TOTAL across all cells
K_GRID = (2.0, 8.0, 0.1)  # (start, stop_inclusive, step) for the target search

# A cell below this cannot support a percentile estimate. Below it, a grouped
# cell inherits its benchmark-pooled parent; an already-pooled cell is
# reported unfittable rather than banded on too little evidence.
MIN_CELL_N = 2000


# ---------------------------------------------------------------------------
# Pure lookups
# ---------------------------------------------------------------------------
def benchmark_for(strategy) -> str | None:
    """Benchmark family for one strategy, or None when it is not mapped."""
    if strategy is None:
        return None
    key = str(strategy).strip().upper()
    if not key or key in {"NAN", "NONE"}:
        return None
    return ALGO_BENCHMARK.get(key)


def market_group_for(market, groups: dict) -> str | None:
    """Which declared group a market belongs to, or None if it belongs to none.

    A "*" member is a wildcard matching every market. Explicit membership is
    checked first so a declared group always wins over a wildcard sharing the
    same config.
    """
    if market is None:
        return None
    key = str(market).strip().upper()
    if not key:
        return None
    wildcard = None
    for name, members in groups.items():
        if members == "*":
            wildcard = wildcard or name
            continue
        if key in {str(m).strip().upper() for m in members}:
            return name
    return wildcard
