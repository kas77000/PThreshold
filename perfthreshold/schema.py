"""Canonical column names used everywhere downstream of the loader.

The raw extract can call its columns whatever it likes; config.COLUMN_MAP
translates those names into these. Nothing outside load.py should ever
reference a vendor column name -- that is what keeps a change to the extract
a one-line change here rather than a search across the codebase.
"""

# --- identity and grouping ------------------------------------------------
ORDER_ID = "order_id"
ALGO = "algo"                 # the raw Strategy value
SYMBOL = "symbol"             # full ticker, e.g. "0700 HK"
MARKET = "market"             # ticker suffix, e.g. "HK"
ORDER_DATE = "order_date"
BENCHMARK = "benchmark"       # derived: TWAP / VWAP, via config.ALGO_BENCHMARK
MARKET_GROUP = "market_group"  # derived: which MARKET_GROUPS entry the row fell in
CELL_KEY = "cell_key"         # derived: "<benchmark>|<market_group>"

# --- the banded metric ----------------------------------------------------
METRIC = "perf_in_spreads"    # from the extract's "ePvwap/Sprd", as supplied

# --- carried for reporting, never banded ----------------------------------
SLIPPAGE_BPS = "slippage_bps"  # Pvwap, shown next to the ratio in outliers.csv
SPREAD_BPS = "spread_bps"      # Sprd, a reference feature and a drift signal

# --- reference features: medians stamped at fit time ----------------------
PCT_ADV = "pct_adv"
VOLATILITY = "volatility"
PARTICIPATION = "participation"
DURATION_MIN = "duration_min"

# --- diagnostics: only ride into outliers.csv -----------------------------
NOTIONAL = "notional"
QUANTITY = "quantity"
SIDE = "side"
PASSIVE_FILL = "passive_fill_pct"
OPEN_PCT = "open_pct"
CLOSE_PCT = "close_pct"
REVERSION = "reversion"

# A row without every one of these cannot be fitted or scored.
REQUIRED = [ORDER_ID, ALGO, SYMBOL, ORDER_DATE, METRIC]

# Medians of these are written into bands.json at fit time and compared at
# score time. They are the only way to tell "the book got harder" apart from
# "execution got worse", and they cannot be reconstructed after the fact.
REFERENCE_FEATURES = [PCT_ADV, VOLATILITY, PARTICIPATION, DURATION_MIN,
                      SPREAD_BPS]

DIAGNOSTICS = [SLIPPAGE_BPS, SPREAD_BPS, NOTIONAL, QUANTITY, SIDE,
               PASSIVE_FILL, OPEN_PCT, CLOSE_PCT, REVERSION, PCT_ADV,
               VOLATILITY, PARTICIPATION, DURATION_MIN]
