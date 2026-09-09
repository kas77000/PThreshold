# PerfThreshold v1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fit two-sided outlier thresholds on spread-normalised algo performance over a year of orders, choosing the sigma multiple against a monthly review budget, and apply the frozen result to a single later month.

**Architecture:** A pure arithmetic core (`rule.py`) wrapped by a pandas pipeline. Loading, cell assignment, fitting, calibration and scoring are separate modules with one job each; the only shared state is a versioned JSON artifact written by `persist.py` and read back by `score.py`. Fit and score are separate CLI commands so a month can never be scored by a band that saw it.

**Tech Stack:** Python 3.13, pandas 2.2, numpy 2.1, matplotlib 3.10, pytest 8.3. No scipy.

## Global Constraints

- The banded metric is the extract column **`ePvwap/Sprd`**, consumed **as supplied**. This project never divides `Pvwap` by `Sprd` itself.
- Metric units are the string `"spreads"` everywhere. `bands.json` records it; `score` refuses a mismatch.
- The rule, per cell per side: `hi = MAX(mean + k*sd, P99.5)`, `lo = MIN(mean - k*sd, P0.5)`.
- `sd` is the sample standard deviation with `ddof=1`. `mean` is the arithmetic mean. The sigma term stays literal — never solved, never adjusted.
- Percentiles use `numpy.percentile` with the default `linear` interpolation. Default upper percentile `99.5`, lower is `100 - 99.5 = 0.5`, always mirrored.
- **k is a single global value** applied to every cell. Default mode is `--target 5`; `--k` overrides. The two are mutually exclusive.
- An unmapped `Strategy` is **excluded, named and counted**. Never defaulted into a benchmark family.
- No silent row drops anywhere. Every exclusion is counted by reason in the cleaning report.
- Dates are parsed to `pandas.Timestamp` normalised to midnight (no time component).
- All file paths use `os.path.join` / `pathlib`. Target platform is Windows; never hardcode `/`.
- Every module gets a module docstring explaining *why* it exists, not what it does.
- Commit after every task.

---

## File Structure

| File | Responsibility |
|---|---|
| `perfthreshold/schema.py` | Canonical internal column names. No logic. |
| `perfthreshold/config.py` | The only file a user edits: column map, `ALGO_BENCHMARK`, `MARKET_GROUPS`, defaults. Plus pure lookup helpers. |
| `perfthreshold/rule.py` | Pure: `numpy` array + k + percentile → bounds and which term bound. No pandas, no config, no I/O. |
| `perfthreshold/load.py` | Read file → validate → rename → derive market/benchmark → tidy frame + `CleaningReport`. |
| `perfthreshold/groups.py` | Resolve a scope string into each row's `cell_key`. |
| `perfthreshold/fit.py` | Fit every cell over a frame; thin-cell fallback. |
| `perfthreshold/persist.py` | Write/read `bands.json` and `bands.csv`; version + provenance. |
| `perfthreshold/calibrate.py` | Leave-one-month-out counts, k-curve, target search. |
| `perfthreshold/split.py` | Per-market pooled-vs-own flag evidence. |
| `perfthreshold/score.py` | Apply a frozen band to one month; leakage/unit/scope guards; drift check. |
| `perfthreshold/plots.py` | Distribution and calibration charts. |
| `perfthreshold/cli.py` | `check` / `fit` / `score`. |
| `tests/synthetic.py` | Fat-tailed synthetic book generator used by every downstream test. |

---

## Task 1: Scaffold, schema, config

**Files:**
- Create: `requirements.txt`, `perfthreshold/__init__.py`, `perfthreshold/schema.py`, `perfthreshold/config.py`, `tests/__init__.py`
- Test: `tests/test_config.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `schema.ORDER_ID`, `schema.ALGO`, `schema.MARKET`, `schema.SYMBOL`, `schema.ORDER_DATE`, `schema.METRIC`, `schema.BENCHMARK`, `schema.CELL_KEY`, `schema.SPREAD_BPS`, `schema.SLIPPAGE_BPS` — all `str` constants.
  - `schema.REQUIRED: list[str]` — canonical names a row cannot be missing.
  - `schema.REFERENCE_FEATURES: list[str]` — canonical names whose medians get stamped.
  - `config.COLUMN_MAP: dict[str, str]` — raw extract name → canonical name.
  - `config.ALGO_BENCHMARK: dict[str, str]` — upper-cased strategy → benchmark family.
  - `config.benchmark_for(strategy: str) -> str | None` — `None` when unmapped.
  - `config.MARKET_GROUPS: dict[str, str | list[str]]`
  - `config.market_group_for(market: str, groups: dict) -> str | None`
  - `config.METRIC_COLUMN: str`, `config.METRIC_UNITS: str`, `config.PERCENTILE: float`, `config.DEFAULT_K: float`, `config.DEFAULT_TARGET: int`, `config.MIN_CELL_N: int`, `config.K_GRID: tuple[float, float, float]`

- [ ] **Step 1: Write the failing test**

Create `tests/test_config.py`:

```python
import pytest

from perfthreshold import config, schema


def test_benchmark_lookup_is_case_insensitive():
    assert config.benchmark_for("vwap") == "VWAP"
    assert config.benchmark_for("VWAP") == "VWAP"
    assert config.benchmark_for("  Vwap  ") == "VWAP"


def test_unmapped_strategy_returns_none_not_a_default():
    # The single most dangerous failure mode: a band fitted on the wrong
    # benchmark still fits and still looks plausible. Never guess.
    assert config.benchmark_for("PART") is None
    assert config.benchmark_for("") is None
    assert config.benchmark_for(None) is None


def test_wildcard_market_group_matches_everything():
    groups = {"ALL": "*"}
    assert config.market_group_for("HK", groups) == "ALL"
    assert config.market_group_for("ZZ", groups) == "ALL"


def test_explicit_market_group_membership():
    groups = {"TIGHT": ["HK", "JP"], "WIDE": ["IN", "TH"]}
    assert config.market_group_for("hk", groups) == "TIGHT"
    assert config.market_group_for("TH", groups) == "WIDE"


def test_market_outside_every_group_is_none():
    groups = {"TIGHT": ["HK", "JP"]}
    assert config.market_group_for("IN", groups) is None


def test_percentile_default_and_metric_contract():
    assert config.PERCENTILE == 99.5
    assert config.METRIC_COLUMN == "ePvwap/Sprd"
    assert config.METRIC_UNITS == "spreads"


def test_column_map_covers_every_required_schema_field():
    mapped = set(config.COLUMN_MAP.values())
    missing = [c for c in schema.REQUIRED if c not in mapped]
    assert missing == [], f"COLUMN_MAP has no source for: {missing}"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_config.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'perfthreshold'`

- [ ] **Step 3: Write `requirements.txt` and the two modules**

`requirements.txt`:

```
pandas>=2.0
numpy>=1.24
matplotlib>=3.7
pytest>=7.0
```

`perfthreshold/__init__.py`:

```python
"""Spread-normalised outlier thresholds for algo execution performance."""

__version__ = "1.0.0"
```

`tests/__init__.py`: empty file.

`perfthreshold/schema.py`:

```python
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
```

`perfthreshold/config.py`:

```python
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

# ---------------------------------------------------------------------------
# 1) Column map: raw extract name -> canonical name
# ---------------------------------------------------------------------------
from perfthreshold import schema

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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_config.py -v`
Expected: PASS — 7 passed

- [ ] **Step 5: Commit**

```bash
git add requirements.txt perfthreshold/ tests/
git commit -m "feat: schema and config contract for PerfThreshold"
```

---

## Task 2: The rule — pure bounds arithmetic

**Files:**
- Create: `perfthreshold/rule.py`
- Test: `tests/test_rule.py`

**Interfaces:**
- Consumes: nothing (deliberately — no pandas, no config, no I/O).
- Produces:
  - `rule.IN_RANGE`, `rule.OUT_LOW`, `rule.OUT_HIGH`, `rule.NO_BAND` — `str` zone labels.
  - `rule.estimates(x) -> dict` with keys `n, mean, sd, median, mad_sigma`.
  - `rule.bounds(x, k: float, percentile: float = 99.5) -> dict` with keys
    `n, mean, sd, median, mad_sigma, p_hi, p_lo, sigma_hi, sigma_lo, hi, lo, hi_binds, lo_binds`.
    `hi_binds` / `lo_binds` are `"sigma"` or `"percentile"`.
  - `rule.zone(value: float, lo: float, hi: float) -> str`
  - `rule.count_flags(x, lo: float, hi: float) -> int`

- [ ] **Step 1: Write the failing test**

Create `tests/test_rule.py`:

```python
import numpy as np
import pytest

from perfthreshold import rule


def test_sigma_term_binds_when_it_is_wider():
    # 0..99 evenly spaced: sd is large relative to the extremes, so
    # mean + 4*sd sits far beyond P99.5 and must win.
    x = np.arange(100, dtype=float)
    b = rule.bounds(x, k=4.0, percentile=99.5)
    assert b["hi_binds"] == "sigma"
    assert b["lo_binds"] == "sigma"
    assert b["hi"] == pytest.approx(b["sigma_hi"])
    assert b["hi"] > b["p_hi"]


def test_percentile_term_binds_when_sigma_is_narrow():
    # k = 0 collapses the sigma term onto the mean, so the percentile must win
    # on both sides. This is the cleanest possible check that MAX/MIN pick the
    # right candidate rather than always taking the same one.
    x = np.arange(100, dtype=float)
    b = rule.bounds(x, k=0.0, percentile=99.5)
    assert b["hi_binds"] == "percentile"
    assert b["lo_binds"] == "percentile"
    assert b["hi"] == pytest.approx(b["p_hi"])
    assert b["lo"] == pytest.approx(b["p_lo"])


def test_bounds_are_hand_computable():
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    b = rule.bounds(x, k=1.0, percentile=99.5)
    assert b["n"] == 5
    assert b["mean"] == pytest.approx(3.0)
    assert b["sd"] == pytest.approx(np.std(x, ddof=1))   # ddof=1, not 0
    assert b["sigma_hi"] == pytest.approx(3.0 + np.std(x, ddof=1))
    assert b["sigma_lo"] == pytest.approx(3.0 - np.std(x, ddof=1))


def test_lower_percentile_mirrors_the_upper_one():
    x = np.arange(1000, dtype=float)
    b = rule.bounds(x, k=0.0, percentile=99.5)
    assert b["p_hi"] == pytest.approx(np.percentile(x, 99.5))
    assert b["p_lo"] == pytest.approx(np.percentile(x, 0.5))


def test_robust_pair_is_computed_but_never_used_for_bounds():
    x = np.array([1.0, 2.0, 3.0, 4.0, 100.0])
    b = rule.bounds(x, k=4.0, percentile=99.5)
    assert b["median"] == pytest.approx(3.0)
    assert b["mad_sigma"] == pytest.approx(1.4826 * 1.0)
    # The active band must come from the classical estimator.
    assert b["hi"] == pytest.approx(max(b["sigma_hi"], b["p_hi"]))


def test_nans_are_dropped_not_propagated():
    x = np.array([1.0, np.nan, 3.0, np.inf, 5.0])
    b = rule.bounds(x, k=1.0, percentile=99.5)
    assert b["n"] == 3
    assert np.isfinite(b["hi"]) and np.isfinite(b["lo"])


def test_empty_input_gives_nan_bounds_and_zero_n():
    b = rule.bounds(np.array([]), k=4.0, percentile=99.5)
    assert b["n"] == 0
    assert np.isnan(b["hi"]) and np.isnan(b["lo"])
    assert b["hi_binds"] == "" and b["lo_binds"] == ""


def test_single_observation_has_zero_sd_not_a_crash():
    b = rule.bounds(np.array([7.0]), k=4.0, percentile=99.5)
    assert b["n"] == 1
    assert b["sd"] == 0.0
    assert b["hi"] == pytest.approx(7.0)


def test_zone_labels_both_tails_and_the_middle():
    assert rule.zone(0.0, lo=-1.0, hi=1.0) == rule.IN_RANGE
    assert rule.zone(-5.0, lo=-1.0, hi=1.0) == rule.OUT_LOW
    assert rule.zone(5.0, lo=-1.0, hi=1.0) == rule.OUT_HIGH
    assert rule.zone(np.nan, lo=-1.0, hi=1.0) == rule.NO_BAND
    assert rule.zone(0.0, lo=np.nan, hi=1.0) == rule.NO_BAND


def test_bounds_are_inclusive_so_a_value_exactly_on_the_edge_is_in_range():
    # An order sitting exactly on the bound should not be flagged: the band is
    # the acceptable region, and a strict inequality would make the flag count
    # depend on floating-point luck.
    assert rule.zone(1.0, lo=-1.0, hi=1.0) == rule.IN_RANGE
    assert rule.zone(-1.0, lo=-1.0, hi=1.0) == rule.IN_RANGE


def test_count_flags_counts_both_tails():
    x = np.array([-10.0, 0.0, 0.5, 10.0, np.nan])
    assert rule.count_flags(x, lo=-1.0, hi=1.0) == 2


def test_count_flags_is_zero_when_the_band_is_undefined():
    x = np.array([-10.0, 0.0, 10.0])
    assert rule.count_flags(x, lo=np.nan, hi=1.0) == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_rule.py -v`
Expected: FAIL — `ImportError: cannot import name 'rule'`

- [ ] **Step 3: Write `perfthreshold/rule.py`**

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_rule.py -v`
Expected: PASS — 12 passed

- [ ] **Step 5: Commit**

```bash
git add perfthreshold/rule.py tests/test_rule.py
git commit -m "feat: pure band rule with per-side binding term"
```

---

## Task 3: Synthetic book generator

**Files:**
- Create: `tests/synthetic.py`
- Test: `tests/test_synthetic.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `synthetic.make_book(n_per_month=400, months=12, start="2025-07-01", seed=0, benchmarks=("VWAP","TWAP"), markets=("HK","JP","AU","IN"), unmapped_rows=0) -> pandas.DataFrame` — a frame with the **raw extract column names**, not canonical ones, so it exercises the loader.

Every downstream task tests against this. It must be fat-tailed, because a Gaussian generator would make the whole calibration exercise look easy and hide the exact behaviour the design is built around.

- [ ] **Step 1: Write the failing test**

Create `tests/test_synthetic.py`:

```python
import numpy as np
import pandas as pd

from tests import synthetic


def test_book_has_the_raw_extract_headers():
    df = synthetic.make_book(n_per_month=10, months=2)
    for col in ["aggrTgtId", "Strategy", "Sym", "Date", "ePvwap/Sprd",
                "Pvwap", "Sprd", "%Adv", "Vol", "PR", "Dur"]:
        assert col in df.columns, col


def test_row_count_is_exact_and_spans_the_requested_months():
    df = synthetic.make_book(n_per_month=50, months=12, start="2025-07-01")
    assert len(df) == 50 * 12
    d = pd.to_datetime(df["Date"])
    assert d.min().strftime("%Y-%m") == "2025-07"
    assert d.max().strftime("%Y-%m") == "2026-06"
    assert d.dt.to_period("M").nunique() == 12


def test_metric_is_fat_tailed_not_gaussian():
    # Excess kurtosis well above 0 is the whole point: on a Gaussian book the
    # percentile floor and the 4-sigma term would agree and the design's
    # central question would never arise.
    df = synthetic.make_book(n_per_month=2000, months=12, seed=1)
    x = df["ePvwap/Sprd"].to_numpy(dtype=float)
    z = (x - x.mean()) / x.std(ddof=1)
    excess_kurtosis = float((z ** 4).mean() - 3.0)
    assert excess_kurtosis > 3.0


def test_seed_makes_it_reproducible():
    a = synthetic.make_book(n_per_month=20, months=2, seed=7)
    b = synthetic.make_book(n_per_month=20, months=2, seed=7)
    pd.testing.assert_frame_equal(a, b)


def test_order_ids_are_unique():
    df = synthetic.make_book(n_per_month=100, months=6)
    assert df["aggrTgtId"].is_unique


def test_unmapped_rows_are_injected_on_request():
    df = synthetic.make_book(n_per_month=10, months=1, unmapped_rows=5)
    assert (df["Strategy"] == "PART").sum() == 5


def test_benchmarks_have_visibly_different_centres():
    # TWAP and VWAP must not be interchangeable, or pooling-vs-splitting
    # tests downstream would pass for the wrong reason.
    df = synthetic.make_book(n_per_month=2000, months=12, seed=2)
    means = df.groupby("Strategy")["ePvwap/Sprd"].mean()
    assert abs(means["VWAP"] - means["TWAP"]) > 0.05
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_synthetic.py -v`
Expected: FAIL — `ImportError: cannot import name 'synthetic' from 'tests'`

- [ ] **Step 3: Write `tests/synthetic.py`**

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_synthetic.py -v`
Expected: PASS — 7 passed

- [ ] **Step 5: Commit**

```bash
git add tests/synthetic.py tests/test_synthetic.py
git commit -m "test: fat-tailed synthetic order book generator"
```

---

## Task 4: Loading, validation and the cleaning report

**Files:**
- Create: `perfthreshold/load.py`
- Test: `tests/test_load.py`

**Interfaces:**
- Consumes: `schema` (Task 1), `config.COLUMN_MAP`, `config.benchmark_for` (Task 1), `tests.synthetic.make_book` (Task 3).
- Produces:
  - `load.CleaningReport` — dataclass with fields `rows_in: int`, `rows_kept: int`, `dropped: dict[str, int]`, `unmapped_strategies: dict[str, int]`, `date_min`, `date_max`; methods `to_frame() -> pandas.DataFrame` and `summary() -> str`.
  - `load.read_any(path: str) -> pandas.DataFrame` — csv / xlsx / parquet by extension.
  - `load.prepare(raw: pandas.DataFrame) -> tuple[pandas.DataFrame, CleaningReport]` — the whole rename/derive/validate pass.
  - `load.load(path: str) -> tuple[pandas.DataFrame, CleaningReport]` — `read_any` then `prepare`.
  - Drop-reason constants: `load.MISSING_ORDER_ID`, `load.MISSING_SYMBOL`, `load.MISSING_DATE`, `load.MISSING_METRIC`, `load.UNMAPPED_STRATEGY`.

The prepared frame carries canonical columns plus `schema.MARKET` and `schema.BENCHMARK`. It does **not** carry `schema.MARKET_GROUP` or `schema.CELL_KEY` — those depend on the scope and belong to Task 5.

- [ ] **Step 1: Write the failing test**

Create `tests/test_load.py`:

```python
import numpy as np
import pandas as pd
import pytest

from perfthreshold import load, schema
from tests import synthetic


def test_raw_headers_become_canonical_names():
    raw = synthetic.make_book(n_per_month=20, months=1)
    df, _ = load.prepare(raw)
    assert schema.METRIC in df.columns
    assert schema.ORDER_ID in df.columns
    assert "ePvwap/Sprd" not in df.columns, "raw name must not survive"
    assert "aggrTgtId" not in df.columns


def test_market_is_the_ticker_suffix():
    raw = pd.DataFrame({
        "aggrTgtId": ["A", "B"], "Strategy": ["VWAP", "VWAP"],
        "Sym": ["0700 HK", "7203 jt"], "Date": ["2025-07-01", "2025-07-02"],
        "ePvwap/Sprd": [0.1, 0.2], "Pvwap": [1.0, 2.0], "Sprd": [8.0, 9.0],
    })
    df, _ = load.prepare(raw)
    assert list(df[schema.MARKET]) == ["HK", "JT"]
    assert list(df[schema.SYMBOL]) == ["0700 HK", "7203 jt"]


def test_benchmark_is_derived_and_tmx_maps_to_twap():
    raw = pd.DataFrame({
        "aggrTgtId": ["A", "B", "C"], "Strategy": ["VWAP", "TMX", "twap"],
        "Sym": ["1 HK"] * 3, "Date": ["2025-07-01"] * 3,
        "ePvwap/Sprd": [0.1, 0.2, 0.3], "Pvwap": [1.0] * 3, "Sprd": [8.0] * 3,
    })
    df, _ = load.prepare(raw)
    assert list(df[schema.BENCHMARK]) == ["VWAP", "TWAP", "TWAP"]


def test_unmapped_strategy_is_excluded_and_named_with_a_count():
    raw = synthetic.make_book(n_per_month=10, months=1, unmapped_rows=4)
    df, rep = load.prepare(raw)
    assert (df[schema.ALGO] == "PART").sum() == 0
    assert rep.unmapped_strategies == {"PART": 4}
    assert rep.dropped[load.UNMAPPED_STRATEGY] == 4


def test_missing_metric_is_dropped_with_a_reason():
    raw = pd.DataFrame({
        "aggrTgtId": ["A", "B"], "Strategy": ["VWAP", "VWAP"],
        "Sym": ["1 HK", "2 HK"], "Date": ["2025-07-01", "2025-07-02"],
        "ePvwap/Sprd": [0.1, np.nan], "Pvwap": [1.0, 2.0], "Sprd": [8.0, 9.0],
    })
    df, rep = load.prepare(raw)
    assert len(df) == 1
    assert rep.dropped[load.MISSING_METRIC] == 1


def test_unparseable_date_is_dropped_with_a_reason():
    raw = pd.DataFrame({
        "aggrTgtId": ["A", "B"], "Strategy": ["VWAP", "VWAP"],
        "Sym": ["1 HK", "2 HK"], "Date": ["2025-07-01", "not a date"],
        "ePvwap/Sprd": [0.1, 0.2], "Pvwap": [1.0, 2.0], "Sprd": [8.0, 9.0],
    })
    df, rep = load.prepare(raw)
    assert len(df) == 1
    assert rep.dropped[load.MISSING_DATE] == 1


def test_dates_are_normalised_to_midnight():
    raw = pd.DataFrame({
        "aggrTgtId": ["A"], "Strategy": ["VWAP"], "Sym": ["1 HK"],
        "Date": ["2025-07-01 14:33:07"], "ePvwap/Sprd": [0.1],
        "Pvwap": [1.0], "Sprd": [8.0],
    })
    df, _ = load.prepare(raw)
    assert df[schema.ORDER_DATE].iloc[0] == pd.Timestamp("2025-07-01")


def test_every_input_row_is_accounted_for():
    # The contract that makes "no silent drops" checkable rather than a claim.
    raw = synthetic.make_book(n_per_month=50, months=3, unmapped_rows=7)
    df, rep = load.prepare(raw)
    assert rep.rows_in == len(raw)
    assert rep.rows_kept == len(df)
    assert rep.rows_kept + sum(rep.dropped.values()) == rep.rows_in


def test_a_row_failing_two_checks_is_counted_once():
    # Counted under its FIRST failing reason, so the drop counts sum to the
    # number of rows actually removed rather than double-counting.
    raw = pd.DataFrame({
        "aggrTgtId": [None], "Strategy": ["PART"], "Sym": ["1 HK"],
        "Date": ["bad"], "ePvwap/Sprd": [np.nan], "Pvwap": [1.0], "Sprd": [8.0],
    })
    df, rep = load.prepare(raw)
    assert len(df) == 0
    assert sum(rep.dropped.values()) == 1


def test_report_records_the_observed_date_window():
    raw = synthetic.make_book(n_per_month=10, months=12, start="2025-07-01")
    _, rep = load.prepare(raw)
    assert rep.date_min.strftime("%Y-%m") == "2025-07"
    assert rep.date_max.strftime("%Y-%m") == "2026-06"


def test_missing_required_source_column_raises_naming_it():
    raw = pd.DataFrame({"aggrTgtId": ["A"], "Strategy": ["VWAP"],
                        "Sym": ["1 HK"], "Date": ["2025-07-01"]})
    with pytest.raises(ValueError, match="ePvwap"):
        load.prepare(raw)


def test_report_to_frame_is_one_row_per_reason():
    raw = synthetic.make_book(n_per_month=10, months=1, unmapped_rows=3)
    _, rep = load.prepare(raw)
    frame = rep.to_frame()
    assert list(frame.columns) == ["reason", "rows"]
    assert (frame["reason"] == load.UNMAPPED_STRATEGY).any()


def test_read_any_round_trips_a_csv(tmp_path):
    raw = synthetic.make_book(n_per_month=10, months=1)
    path = tmp_path / "book.csv"
    raw.to_csv(path, index=False)
    df, rep = load.load(str(path))
    assert rep.rows_kept == len(raw)
    assert schema.METRIC in df.columns


def test_read_any_rejects_an_unknown_extension(tmp_path):
    path = tmp_path / "book.txt"
    path.write_text("nope", encoding="utf-8")
    with pytest.raises(ValueError, match="Unsupported extension"):
        load.read_any(str(path))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_load.py -v`
Expected: FAIL — `ImportError: cannot import name 'load'`

- [ ] **Step 3: Write `perfthreshold/load.py`**

```python
"""Read the extract, translate it, and account for every row that leaves.

The one rule this module exists to enforce: NOTHING DISAPPEARS QUIETLY. Every
row removed is counted under a named reason, and rows_kept plus the sum of
those counts must equal rows_in. That equality is what turns "no silent drops"
from a claim into something a test can check.

The reason it matters more than usual here: an unmapped strategy quietly
defaulted into a benchmark family produces a band that still fits, still
yields a plausible curve, and never announces itself. The failure has to be
caught at the door or it is not caught at all.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from perfthreshold import config, schema

MISSING_ORDER_ID = "missing order id"
MISSING_SYMBOL = "missing symbol"
MISSING_DATE = "missing or unparseable date"
MISSING_METRIC = "missing metric"
UNMAPPED_STRATEGY = "unmapped strategy"

# Checked in this order; a row is counted under the FIRST reason it fails, so
# the counts sum to the number of rows actually removed.
_REASONS = [MISSING_ORDER_ID, MISSING_SYMBOL, MISSING_DATE,
            UNMAPPED_STRATEGY, MISSING_METRIC]


@dataclass
class CleaningReport:
    """What came in, what stayed, and why the rest left."""

    rows_in: int = 0
    rows_kept: int = 0
    dropped: dict[str, int] = field(default_factory=dict)
    unmapped_strategies: dict[str, int] = field(default_factory=dict)
    date_min: pd.Timestamp | None = None
    date_max: pd.Timestamp | None = None

    def to_frame(self) -> pd.DataFrame:
        rows = [{"reason": r, "rows": n} for r, n in self.dropped.items() if n]
        if not rows:
            rows = [{"reason": "none", "rows": 0}]
        return pd.DataFrame(rows, columns=["reason", "rows"])

    def summary(self) -> str:
        lines = [f"rows in  : {self.rows_in}", f"rows kept: {self.rows_kept}"]
        for reason, n in self.dropped.items():
            if n:
                lines.append(f"  dropped ({reason}): {n}")
        if self.unmapped_strategies:
            named = ", ".join(f"{s} ({n})" for s, n in
                              sorted(self.unmapped_strategies.items()))
            lines.append("  UNMAPPED STRATEGIES -- excluded, not defaulted: "
                         + named)
        if self.date_min is not None:
            lines.append(f"date range: {self.date_min.date()} .. "
                         f"{self.date_max.date()}")
        return "\n".join(lines)


def read_any(path: str) -> pd.DataFrame:
    """Read csv, xlsx or parquet by extension."""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".csv":
        return pd.read_csv(path)
    if ext in (".xlsx", ".xls"):
        return pd.read_excel(path)
    if ext == ".parquet":
        return pd.read_parquet(path)
    raise ValueError(
        f"Unsupported extension '{ext}'. Use .csv, .xlsx or .parquet.")


def _require_source_columns(raw: pd.DataFrame) -> None:
    """Fail loudly, naming the RAW column, before anything is renamed."""
    inverse = {v: k for k, v in config.COLUMN_MAP.items()}
    missing = [inverse[c] for c in schema.REQUIRED
               if c in inverse and inverse[c] not in raw.columns]
    if missing:
        raise ValueError(
            "Extract is missing required column(s): " + ", ".join(missing)
            + ". Run `python -m perfthreshold check <file>` to see what it has.")


def prepare(raw: pd.DataFrame) -> tuple[pd.DataFrame, CleaningReport]:
    """Rename, derive, validate. Returns the tidy frame and the report."""
    _require_source_columns(raw)

    rep = CleaningReport(rows_in=len(raw), dropped={r: 0 for r in _REASONS})

    present = {k: v for k, v in config.COLUMN_MAP.items() if k in raw.columns}
    df = raw.rename(columns=present).copy()
    # Anything the map does not name is dropped: an unmapped source column has
    # no canonical meaning, and carrying it invites downstream code to reach
    # for a vendor name.
    df = df[list(present.values())]

    # --- derive -----------------------------------------------------------
    sym = df[schema.SYMBOL].astype(str).str.strip()
    df[schema.MARKET] = sym.str[-2:].str.upper()
    df.loc[sym.isin(["", "nan", "None"]), schema.MARKET] = pd.NA

    df[schema.BENCHMARK] = df[schema.ALGO].map(config.benchmark_for)

    df[schema.ORDER_DATE] = pd.to_datetime(
        df[schema.ORDER_DATE], errors="coerce").dt.normalize()

    for col in [schema.METRIC] + schema.REFERENCE_FEATURES:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    # +/-inf is not a usable metric value and would poison mean and sd.
    df[schema.METRIC] = df[schema.METRIC].replace([np.inf, -np.inf], np.nan)

    # --- validate; first failing reason wins ------------------------------
    oid = df[schema.ORDER_ID].astype(str).str.strip()
    checks = [
        (MISSING_ORDER_ID,
         df[schema.ORDER_ID].isna() | oid.isin(["", "nan", "None"])),
        (MISSING_SYMBOL, df[schema.MARKET].isna()),
        (MISSING_DATE, df[schema.ORDER_DATE].isna()),
        (UNMAPPED_STRATEGY, df[schema.BENCHMARK].isna()),
        (MISSING_METRIC, df[schema.METRIC].isna()),
    ]

    unresolved = pd.Series(True, index=df.index)
    for reason, failed in checks:
        hit = unresolved & failed.fillna(True).astype(bool)
        rep.dropped[reason] = int(hit.sum())
        if reason == UNMAPPED_STRATEGY and bool(hit.any()):
            counts = (df.loc[hit, schema.ALGO].astype(str).str.strip()
                      .value_counts())
            rep.unmapped_strategies = {str(k): int(v)
                                       for k, v in counts.items()}
        unresolved = unresolved & ~hit

    df = df[unresolved].reset_index(drop=True)

    rep.rows_kept = len(df)
    if len(df):
        rep.date_min = df[schema.ORDER_DATE].min()
        rep.date_max = df[schema.ORDER_DATE].max()
    return df, rep


def load(path: str) -> tuple[pd.DataFrame, CleaningReport]:
    return prepare(read_any(path))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_load.py -v`
Expected: PASS — 14 passed

- [ ] **Step 5: Commit**

```bash
git add perfthreshold/load.py tests/test_load.py
git commit -m "feat: extract loader with a fully accounted cleaning report"
```

---

## Task 5: Scope resolution and cell keys

**Files:**
- Create: `perfthreshold/groups.py`
- Test: `tests/test_groups.py`

**Interfaces:**
- Consumes: `schema`, `config.MARKET_GROUPS`, `config.market_group_for` (Task 1); `load.prepare` (Task 4).
- Produces:
  - `groups.SCOPE_ALL = "all"`, `groups.SCOPE_GROUPS = "groups"`, `groups.GROUP_PREFIX = "group:"`
  - `groups.POOLED = "__POOLED__"` — the reserved group name for a benchmark-pooled parent band.
  - `groups.ALL_GROUP = "ALL"` — the group name every row gets under `--scope all`.
  - `groups.parse_scope(scope: str) -> tuple[str, str | None]` — `("all", None)`, `("groups", None)`, `("group", "APAC_TIGHT")`. Raises `ValueError` on anything else.
  - `groups.cell_key(benchmark: str, market_group: str) -> str` — `"VWAP|ALL"`.
  - `groups.split_key(cell: str) -> tuple[str, str]` — the inverse.
  - `groups.parent_key(cell: str) -> str` — `"VWAP|__POOLED__"`.
  - `groups.resolve(df, scope=SCOPE_ALL, market_groups=None, benchmark=None) -> tuple[pandas.DataFrame, dict[str, int]]` — adds `schema.MARKET_GROUP` and `schema.CELL_KEY`, drops out-of-scope rows, returns the frame and a dict of exclusion counts by reason.
  - Exclusion reason constants: `groups.OUT_OF_SCOPE_MARKET`, `groups.OUT_OF_SCOPE_GROUP`, `groups.OUT_OF_SCOPE_BENCHMARK`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_groups.py`:

```python
import pandas as pd
import pytest

from perfthreshold import groups, load, schema
from tests import synthetic


def _prepared(**kw):
    raw = synthetic.make_book(n_per_month=40, months=2, seed=3, **kw)
    df, _ = load.prepare(raw)
    return df


def test_parse_scope_recognises_the_three_forms():
    assert groups.parse_scope("all") == ("all", None)
    assert groups.parse_scope("groups") == ("groups", None)
    assert groups.parse_scope("group:APAC_TIGHT") == ("group", "APAC_TIGHT")


def test_parse_scope_rejects_nonsense_naming_what_it_wanted():
    with pytest.raises(ValueError, match="all"):
        groups.parse_scope("everything")


def test_cell_key_round_trips():
    assert groups.cell_key("VWAP", "ALL") == "VWAP|ALL"
    assert groups.split_key("VWAP|ALL") == ("VWAP", "ALL")


def test_parent_key_is_the_benchmark_pooled_band():
    assert groups.parent_key("VWAP|APAC_TIGHT") == "VWAP|" + groups.POOLED


def test_scope_all_pools_every_market_regardless_of_config():
    df = _prepared()
    out, excl = groups.resolve(
        df, scope="all",
        market_groups={"TIGHT": ["HK"], "WIDE": ["IN"]})
    # The declared groups are ignored under scope=all -- that is the point.
    assert set(out[schema.MARKET_GROUP]) == {"ALL"}
    assert set(out[schema.CELL_KEY]) == {"VWAP|ALL", "TWAP|ALL"}
    assert sum(excl.values()) == 0
    assert len(out) == len(df)


def test_scope_groups_assigns_declared_membership():
    df = _prepared()
    out, excl = groups.resolve(
        df, scope="groups",
        market_groups={"TIGHT": ["HK", "JP"], "WIDE": ["AU", "IN"]})
    assert set(out[schema.MARKET_GROUP]) == {"TIGHT", "WIDE"}
    hk = out[out[schema.MARKET] == "HK"]
    assert set(hk[schema.MARKET_GROUP]) == {"TIGHT"}


def test_scope_groups_excludes_markets_in_no_declared_group():
    df = _prepared()
    out, excl = groups.resolve(
        df, scope="groups", market_groups={"TIGHT": ["HK"]})
    assert set(out[schema.MARKET]) == {"HK"}
    assert excl[groups.OUT_OF_SCOPE_MARKET] == len(df) - len(out)
    assert excl[groups.OUT_OF_SCOPE_MARKET] > 0


def test_scope_single_group_keeps_only_that_group():
    df = _prepared()
    out, excl = groups.resolve(
        df, scope="group:TIGHT",
        market_groups={"TIGHT": ["HK", "JP"], "WIDE": ["AU", "IN"]})
    assert set(out[schema.MARKET]) == {"HK", "JP"}
    assert set(out[schema.MARKET_GROUP]) == {"TIGHT"}
    assert excl[groups.OUT_OF_SCOPE_GROUP] > 0


def test_unknown_group_name_raises_listing_what_is_declared():
    df = _prepared()
    with pytest.raises(ValueError, match="TIGHT"):
        groups.resolve(df, scope="group:NOPE",
                       market_groups={"TIGHT": ["HK"]})


def test_benchmark_filter_narrows_to_one_family():
    df = _prepared()
    out, excl = groups.resolve(df, scope="all", benchmark="VWAP")
    assert set(out[schema.BENCHMARK]) == {"VWAP"}
    assert excl[groups.OUT_OF_SCOPE_BENCHMARK] > 0


def test_benchmark_filter_is_case_insensitive():
    df = _prepared()
    out, _ = groups.resolve(df, scope="all", benchmark="vwap")
    assert set(out[schema.BENCHMARK]) == {"VWAP"}


def test_every_row_is_still_accounted_for_after_resolution():
    df = _prepared()
    out, excl = groups.resolve(
        df, scope="group:TIGHT",
        market_groups={"TIGHT": ["HK"], "WIDE": ["JP"]}, benchmark="VWAP")
    assert len(out) + sum(excl.values()) == len(df)


def test_wildcard_group_under_scope_groups_catches_everything():
    df = _prepared()
    out, excl = groups.resolve(df, scope="groups", market_groups={"ALL": "*"})
    assert len(out) == len(df)
    assert sum(excl.values()) == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_groups.py -v`
Expected: FAIL — `ImportError: cannot import name 'groups'`

- [ ] **Step 3: Write `perfthreshold/groups.py`**

```python
"""Which cell a row belongs to, under a given scope.

Fit and score both derive cells through this module and nothing else. That is
what guarantees the band written for VWAP|APAC_TIGHT is the one score looks up
for those same rows. If the two sides derived cells independently they would
drift apart the first time a strategy name changed case or a group was
renamed, and the mismatch would show up as a flag-rate change rather than as
an error.

Scope is the only thing that varies:

    all              every market pooled; one cell per benchmark
    groups           one cell per (benchmark x declared group)
    group:NAME       that group only; every other market excluded

All three run the same code path -- a scope only changes how a market_group is
assigned -- so a bug cannot exist in one and not the others.
"""

from __future__ import annotations

import pandas as pd

from perfthreshold import config, schema

SCOPE_ALL = "all"
SCOPE_GROUPS = "groups"
GROUP_PREFIX = "group:"

# The group name every row receives under scope=all.
ALL_GROUP = "ALL"

# Reserved name for a benchmark-pooled parent band. Reserved rather than
# reusing "ALL" so that a user who declares a group literally called ALL
# cannot collide with the fallback parent.
POOLED = "__POOLED__"

OUT_OF_SCOPE_MARKET = "market in no declared group"
OUT_OF_SCOPE_GROUP = "market outside the selected group"
OUT_OF_SCOPE_BENCHMARK = "benchmark not selected"

_SEP = "|"


def parse_scope(scope: str) -> tuple[str, str | None]:
    """('all'|'groups'|'group', group_name_or_None)."""
    s = str(scope).strip()
    if s == SCOPE_ALL:
        return SCOPE_ALL, None
    if s == SCOPE_GROUPS:
        return SCOPE_GROUPS, None
    if s.startswith(GROUP_PREFIX):
        name = s[len(GROUP_PREFIX):].strip()
        if not name:
            raise ValueError("scope 'group:' needs a group name after the colon")
        return "group", name
    raise ValueError(
        f"Unknown scope {scope!r}. Use 'all', 'groups', or 'group:NAME'.")


def cell_key(benchmark: str, market_group: str) -> str:
    return f"{benchmark}{_SEP}{market_group}"


def split_key(cell: str) -> tuple[str, str]:
    benchmark, _, market_group = str(cell).partition(_SEP)
    return benchmark, market_group


def parent_key(cell: str) -> str:
    """The benchmark-pooled band a thin cell falls back to."""
    benchmark, _ = split_key(cell)
    return cell_key(benchmark, POOLED)


def resolve(df: pd.DataFrame, scope: str = SCOPE_ALL,
            market_groups: dict | None = None,
            benchmark: str | None = None
            ) -> tuple[pd.DataFrame, dict[str, int]]:
    """Attach market_group and cell_key; drop what the scope excludes."""
    market_groups = (config.MARKET_GROUPS if market_groups is None
                     else market_groups)
    kind, wanted = parse_scope(scope)

    if kind == "group" and wanted not in market_groups:
        raise ValueError(
            f"Group {wanted!r} is not declared. MARKET_GROUPS has: "
            + ", ".join(sorted(market_groups)))

    out = df.copy()
    excluded = {OUT_OF_SCOPE_BENCHMARK: 0, OUT_OF_SCOPE_MARKET: 0,
                OUT_OF_SCOPE_GROUP: 0}

    if benchmark is not None:
        want = str(benchmark).strip().upper()
        keep = out[schema.BENCHMARK].astype(str).str.upper() == want
        excluded[OUT_OF_SCOPE_BENCHMARK] = int((~keep).sum())
        out = out[keep]

    if kind == SCOPE_ALL:
        # Declared groups are deliberately ignored here: scope=all means one
        # pooled cell per benchmark, whatever config happens to declare.
        out[schema.MARKET_GROUP] = ALL_GROUP
    else:
        assigned = out[schema.MARKET].map(
            lambda m: config.market_group_for(m, market_groups))
        out[schema.MARKET_GROUP] = assigned
        keep = assigned.notna()
        excluded[OUT_OF_SCOPE_MARKET] = int((~keep).sum())
        out = out[keep]

        if kind == "group":
            keep = out[schema.MARKET_GROUP] == wanted
            excluded[OUT_OF_SCOPE_GROUP] = int((~keep).sum())
            out = out[keep]

    out[schema.CELL_KEY] = [
        cell_key(b, g) for b, g in
        zip(out[schema.BENCHMARK], out[schema.MARKET_GROUP])
    ]
    return out.reset_index(drop=True), excluded
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_groups.py -v`
Expected: PASS — 13 passed

- [ ] **Step 5: Commit**

```bash
git add perfthreshold/groups.py tests/test_groups.py
git commit -m "feat: scope resolution and cell keys shared by fit and score"
```

---

## Task 6: Fitting cells, thin-cell fallback, and applying a band

**Files:**
- Create: `perfthreshold/fit.py`
- Test: `tests/test_fit.py`

**Interfaces:**
- Consumes: `rule` (Task 2), `schema`/`config` (Task 1), `groups.POOLED`/`groups.cell_key`/`groups.parent_key` (Task 5).
- Produces:
  - `fit.BAND_LO = "band_lo"`, `fit.BAND_HI = "band_hi"`, `fit.ZONE = "zone"` — columns `fit.apply` adds.
  - `fit.BAND_COLS: list[str]` — the band table's column order.
  - `fit.FitResult` — dataclass with `bands: pandas.DataFrame`, `medians: dict[str, dict[str, float]]`, `k: float`, `percentile: float`, `min_cell_n: int`.
  - `fit.fit_cells(df, k, percentile=99.5, min_cell_n=2000) -> FitResult`
  - `fit.apply(df, bands) -> pandas.DataFrame` — adds `band_lo`, `band_hi`, `zone`.
  - `fit.flag_count(df, bands) -> int` — total flagged rows.

`fit.apply` is the single place a band meets an order. Calibration, the split report and scoring all go through it, so an in-sample count and an out-of-sample one can never be computed by two subtly different code paths.

**The fallback contract.** A cell with `n < min_cell_n` copies its benchmark-pooled parent's bounds and records `fallback_from`. If the parent is itself below `min_cell_n` — which is always the case under `--scope all`, where the cell *is* the pooled band — the cell is left unfitted with NaN bounds, and `fit.apply` gives its orders `NO_BAND` rather than banding them on too little evidence.

- [ ] **Step 1: Write the failing test**

Create `tests/test_fit.py`:

```python
import numpy as np
import pandas as pd
import pytest

from perfthreshold import fit, groups, load, rule, schema
from tests import synthetic


def _resolved(scope="all", market_groups=None, n_per_month=400, months=12,
              seed=5, markets=("HK", "JP", "AU", "IN")):
    raw = synthetic.make_book(n_per_month=n_per_month, months=months,
                              seed=seed, markets=markets)
    df, _ = load.prepare(raw)
    out, _ = groups.resolve(df, scope=scope, market_groups=market_groups)
    return out


def test_one_row_per_cell_with_the_declared_columns():
    df = _resolved()
    res = fit.fit_cells(df, k=4.0, percentile=99.5, min_cell_n=100)
    assert list(res.bands.columns) == fit.BAND_COLS
    assert set(res.bands["cell_key"]) == {"VWAP|ALL", "TWAP|ALL"}
    assert len(res.bands) == 2


def test_cell_numbers_match_the_pure_rule_on_the_same_slice():
    df = _resolved()
    res = fit.fit_cells(df, k=4.0, percentile=99.5, min_cell_n=100)
    slice_ = df[df[schema.CELL_KEY] == "VWAP|ALL"][schema.METRIC].to_numpy()
    expected = rule.bounds(slice_, k=4.0, percentile=99.5)
    row = res.bands.set_index("cell_key").loc["VWAP|ALL"]
    assert row["n"] == expected["n"]
    assert row["hi"] == pytest.approx(expected["hi"])
    assert row["lo"] == pytest.approx(expected["lo"])
    assert row["hi_binds"] == expected["hi_binds"]


def test_reference_medians_are_stamped_per_cell():
    df = _resolved()
    res = fit.fit_cells(df, k=4.0, min_cell_n=100)
    med = res.medians["VWAP|ALL"]
    for feature in schema.REFERENCE_FEATURES:
        assert feature in med
        assert np.isfinite(med[feature])


def test_thin_grouped_cell_inherits_its_benchmark_pooled_parent():
    # JP is deliberately tiny; TIGHT/WIDE split makes it its own cell.
    df = _resolved(scope="groups",
                   market_groups={"TIGHT": ["HK"], "WIDE": ["JP"]},
                   n_per_month=300, months=12)
    jp = df[df[schema.MARKET] == "JP"]
    res = fit.fit_cells(df, k=4.0, min_cell_n=len(jp) + 1)
    row = res.bands.set_index("cell_key").loc["VWAP|WIDE"]
    assert not row["fitted"]
    assert row["fallback_from"] == "VWAP|" + groups.POOLED
    # The inherited bounds must equal the pooled fit over that benchmark.
    pooled = df[df[schema.BENCHMARK] == "VWAP"][schema.METRIC].to_numpy()
    expected = rule.bounds(pooled, k=4.0, percentile=99.5)
    assert row["hi"] == pytest.approx(expected["hi"])


def test_pooled_cell_below_min_n_is_left_unfitted_not_inherited():
    # Under scope=all the cell IS the pooled band, so there is no parent.
    df = _resolved(n_per_month=10, months=2)
    res = fit.fit_cells(df, k=4.0, min_cell_n=10_000)
    row = res.bands.set_index("cell_key").loc["VWAP|ALL"]
    assert not row["fitted"]
    assert row["fallback_from"] == ""
    assert np.isnan(row["hi"]) and np.isnan(row["lo"])


def test_apply_adds_bounds_and_a_zone_for_every_row():
    df = _resolved()
    res = fit.fit_cells(df, k=4.0, min_cell_n=100)
    scored = fit.apply(df, res.bands)
    assert len(scored) == len(df)
    for col in (fit.BAND_LO, fit.BAND_HI, fit.ZONE):
        assert col in scored.columns
    assert set(scored[fit.ZONE]) <= {rule.IN_RANGE, rule.OUT_LOW,
                                     rule.OUT_HIGH, rule.NO_BAND}


def test_apply_flags_both_tails():
    df = _resolved()
    res = fit.fit_cells(df, k=1.0, min_cell_n=100)   # deliberately tight
    scored = fit.apply(df, res.bands)
    assert (scored[fit.ZONE] == rule.OUT_LOW).sum() > 0
    assert (scored[fit.ZONE] == rule.OUT_HIGH).sum() > 0


def test_apply_gives_no_band_when_the_cell_was_never_fitted():
    df = _resolved(n_per_month=10, months=2)
    res = fit.fit_cells(df, k=4.0, min_cell_n=10_000)
    scored = fit.apply(df, res.bands)
    assert set(scored[fit.ZONE]) == {rule.NO_BAND}


def test_apply_gives_no_band_for_a_cell_absent_from_the_table():
    df = _resolved()
    res = fit.fit_cells(df, k=4.0, min_cell_n=100)
    only_vwap = res.bands[res.bands["cell_key"] == "VWAP|ALL"]
    scored = fit.apply(df, only_vwap)
    twap = scored[scored[schema.BENCHMARK] == "TWAP"]
    assert set(twap[fit.ZONE]) == {rule.NO_BAND}


def test_larger_k_never_flags_more_orders():
    # Monotonicity is what makes the target search well defined; if it failed,
    # the smallest qualifying k would not be unique.
    df = _resolved()
    counts = []
    for k in (2.0, 3.0, 4.0, 5.0, 6.0):
        res = fit.fit_cells(df, k=k, min_cell_n=100)
        counts.append(fit.flag_count(df, res.bands))
    assert counts == sorted(counts, reverse=True)


def test_flag_count_matches_the_zone_labels():
    df = _resolved()
    res = fit.fit_cells(df, k=3.0, min_cell_n=100)
    scored = fit.apply(df, res.bands)
    assert fit.flag_count(df, res.bands) == int(
        scored[fit.ZONE].isin(list(rule.FLAGGED)).sum())


def test_empty_frame_gives_an_empty_band_table_not_a_crash():
    df = _resolved().iloc[0:0]
    res = fit.fit_cells(df, k=4.0, min_cell_n=100)
    assert len(res.bands) == 0
    assert list(res.bands.columns) == fit.BAND_COLS
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_fit.py -v`
Expected: FAIL — `ImportError: cannot import name 'fit'`

- [ ] **Step 3: Write `perfthreshold/fit.py`**

```python
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
         *, fitted: bool, fallback_from: str) -> dict:
    return {
        "cell_key": cell, schema.BENCHMARK: benchmark,
        schema.MARKET_GROUP: market_group, "n": b["n"],
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

        if own["n"] >= min_cell_n:
            rows.append(_row(cell, bench, group_name, own,
                             fitted=True, fallback_from=""))
        else:
            parent = parents.get(bench, {})
            pkey = groups.parent_key(cell)
            # A parent that is itself thin is not a rescue. The only honest
            # answer then is no band at all.
            if parent.get("n", 0) >= min_cell_n and parent["n"] > own["n"]:
                inherited = dict(parent)
                inherited["n"] = own["n"]   # the cell's own size, not the pool's
                rows.append(_row(cell, bench, group_name, inherited,
                                 fitted=False, fallback_from=pkey))
            else:
                blank = rule.bounds(np.array([]), k=k, percentile=percentile)
                blank["n"] = own["n"]
                rows.append(_row(cell, bench, group_name, blank,
                                 fitted=False, fallback_from=""))

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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_fit.py -v`
Expected: PASS — 12 passed

- [ ] **Step 5: Commit**

```bash
git add perfthreshold/fit.py tests/test_fit.py
git commit -m "feat: per-cell band fitting with thin-cell fallback"
```

---

## Task 7: Calibration — leave-one-month-out counts, the k-curve, and the target search

**Files:**
- Create: `perfthreshold/calibrate.py`
- Test: `tests/test_calibrate.py`

**Interfaces:**
- Consumes: `fit.fit_cells`, `fit.flag_count` (Task 6); `schema` (Task 1).
- Produces:
  - `calibrate.months(df) -> list[str]` — sorted `"YYYY-MM"` labels present.
  - `calibrate.k_grid(start=2.0, stop=8.0, step=0.1) -> list[float]` — inclusive of `stop`, rounded to 3 dp so floating point never produces `3.9999999`.
  - `calibrate.lomo_counts(df, k, percentile=99.5, min_cell_n=2000) -> dict[str, int]` — month label → flags, each month scored by a band fitted on the *other* months.
  - `calibrate.curve(df, ks=None, percentile=99.5, min_cell_n=2000, target=5) -> pandas.DataFrame` — columns `k, n_months, median_flags, mean_flags, min_flags, max_flags, months_over_target`.
  - `calibrate.KChoice` — dataclass `k: float | None`, `achieved_median: float | None`, `reason: str`, `reachable: bool`.
  - `calibrate.choose_k(curve_df, target) -> KChoice` — the smallest k whose median is at or below target.

**Why leave-one-month-out.** Counting flags on the same year the band was fitted on is circular: those orders shaped the sd that judges them, so the in-sample count is always flattering and always wrong in the same direction. Fitting on the other eleven months and scoring the twelfth gives twelve genuinely out-of-sample observations — and therefore a *range*, which is the number that matters. A median of 5 with a worst month of 12 is a different proposition from a median of 5 with a worst month of 6.

**Cost.** This runs `len(ks) × n_months` fits — around 730 for the default grid over a year. Each is two groupby passes, so a year's file takes seconds. It is deliberately not optimised by deriving bounds arithmetically from cached per-cell statistics: that would duplicate the rule's arithmetic in a second place, and a divergence between the calibrated count and the scored count is precisely the bug that would be hardest to notice.

- [ ] **Step 1: Write the failing test**

Create `tests/test_calibrate.py`:

```python
import numpy as np
import pandas as pd
import pytest

from perfthreshold import calibrate, fit, groups, load, schema
from tests import synthetic


def _resolved(n_per_month=400, months=12, seed=11):
    raw = synthetic.make_book(n_per_month=n_per_month, months=months, seed=seed)
    df, _ = load.prepare(raw)
    out, _ = groups.resolve(df, scope="all")
    return out


def _two_month_frame():
    """Month A is all zeros; month B is all tens. Hand-computable on purpose."""
    rows = []
    for i in range(50):
        rows.append({schema.ORDER_ID: f"A{i}", schema.BENCHMARK: "VWAP",
                     schema.MARKET: "HK", schema.MARKET_GROUP: "ALL",
                     schema.CELL_KEY: "VWAP|ALL", schema.METRIC: 0.0,
                     schema.ORDER_DATE: pd.Timestamp("2025-07-05")})
    for i in range(50):
        rows.append({schema.ORDER_ID: f"B{i}", schema.BENCHMARK: "VWAP",
                     schema.MARKET: "HK", schema.MARKET_GROUP: "ALL",
                     schema.CELL_KEY: "VWAP|ALL", schema.METRIC: 10.0,
                     schema.ORDER_DATE: pd.Timestamp("2025-08-05")})
    return pd.DataFrame(rows)


def test_months_lists_every_period_once_in_order():
    df = _resolved(n_per_month=5, months=12, seed=1)
    assert calibrate.months(df)[0] == "2025-07"
    assert calibrate.months(df)[-1] == "2026-06"
    assert len(calibrate.months(df)) == 12


def test_k_grid_includes_both_ends_and_is_not_floating_point_noise():
    g = calibrate.k_grid(2.0, 8.0, 0.1)
    assert g[0] == 2.0 and g[-1] == 8.0
    assert 4.0 in g
    assert all(round(x, 3) == x for x in g)


def test_a_month_is_never_in_its_own_training_set():
    # Trained on the all-zero month, the band collapses to [0, 0], so every
    # order in the all-tens month must flag. If the tens leaked into their own
    # fit the band would be wide and the count would be 0 -- so this number
    # can only come out right if the leave-one-out actually left one out.
    df = _two_month_frame()
    counts = calibrate.lomo_counts(df, k=4.0, min_cell_n=1)
    assert counts["2025-08"] == 50
    assert counts["2025-07"] == 0


def test_lomo_returns_one_count_per_month():
    df = _resolved(n_per_month=200, months=12, seed=2)
    counts = calibrate.lomo_counts(df, k=4.0, min_cell_n=100)
    assert set(counts) == set(calibrate.months(df))
    assert all(isinstance(v, int) for v in counts.values())


def test_lomo_needs_at_least_two_months():
    df = _resolved(n_per_month=50, months=1)
    with pytest.raises(ValueError, match="two months"):
        calibrate.lomo_counts(df, k=4.0, min_cell_n=10)


def test_curve_is_monotone_non_increasing_in_k():
    df = _resolved(n_per_month=300, months=12, seed=3)
    c = calibrate.curve(df, ks=[2.0, 3.0, 4.0, 5.0, 6.0], min_cell_n=100,
                        target=5)
    med = list(c["median_flags"])
    assert med == sorted(med, reverse=True)


def test_curve_reports_the_range_not_just_the_median():
    df = _resolved(n_per_month=300, months=12, seed=4)
    c = calibrate.curve(df, ks=[3.0, 4.0], min_cell_n=100, target=5)
    assert list(c.columns) == ["k", "n_months", "median_flags", "mean_flags",
                               "min_flags", "max_flags", "months_over_target"]
    row = c.iloc[0]
    assert row["min_flags"] <= row["median_flags"] <= row["max_flags"]
    assert row["n_months"] == 12


def test_choose_k_takes_the_smallest_k_meeting_the_target():
    c = pd.DataFrame({
        "k": [3.0, 4.0, 5.0, 6.0],
        "n_months": [12] * 4,
        "median_flags": [14.0, 5.0, 2.0, 1.0],
        "mean_flags": [14.0, 5.0, 2.0, 1.0],
        "min_flags": [6, 1, 0, 0], "max_flags": [31, 12, 5, 3],
        "months_over_target": [12, 4, 0, 0],
    })
    choice = calibrate.choose_k(c, target=5)
    assert choice.reachable is True
    assert choice.k == 4.0
    assert choice.achieved_median == 5.0


def test_choose_k_says_so_when_the_target_is_unreachable():
    c = pd.DataFrame({
        "k": [3.0, 4.0], "n_months": [12, 12],
        "median_flags": [40.0, 30.0], "mean_flags": [40.0, 30.0],
        "min_flags": [20, 15], "max_flags": [60, 45],
        "months_over_target": [12, 12],
    })
    choice = calibrate.choose_k(c, target=5)
    assert choice.reachable is False
    assert choice.k is None
    assert "30" in choice.reason and "5" in choice.reason


def test_choose_k_on_an_empty_curve_is_unreachable_not_a_crash():
    c = pd.DataFrame(columns=["k", "n_months", "median_flags", "mean_flags",
                              "min_flags", "max_flags", "months_over_target"])
    choice = calibrate.choose_k(c, target=5)
    assert choice.reachable is False and choice.k is None


def test_the_calibrated_k_actually_delivers_roughly_the_target():
    # The end-to-end promise of the whole design, checked on synthetic data.
    df = _resolved(n_per_month=500, months=12, seed=9)
    c = calibrate.curve(df, ks=calibrate.k_grid(2.0, 8.0, 0.25),
                        min_cell_n=100, target=5)
    choice = calibrate.choose_k(c, target=5)
    assert choice.reachable
    counts = calibrate.lomo_counts(df, k=choice.k, min_cell_n=100)
    assert float(np.median(list(counts.values()))) <= 5
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_calibrate.py -v`
Expected: FAIL — `ImportError: cannot import name 'calibrate'`

- [ ] **Step 3: Write `perfthreshold/calibrate.py`**

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_calibrate.py -v`
Expected: PASS — 11 passed

- [ ] **Step 5: Commit**

```bash
git add perfthreshold/calibrate.py tests/test_calibrate.py
git commit -m "feat: leave-one-month-out calibration and target-driven k search"
```

---

## Task 8: The market-split evidence report

**Files:**
- Create: `perfthreshold/split.py`
- Test: `tests/test_split.py`

**Interfaces:**
- Consumes: `rule.bounds`, `rule.count_flags` (Task 2); `schema` (Task 1).
- Produces:
  - `split.SPLIT_COLS: list[str]` — the report's column order.
  - `split.VERDICT_SPLIT = "split"`, `split.VERDICT_POOL = "pool"`, `split.VERDICT_THIN = "too few orders"`.
  - `split.report(df, k, percentile=99.5, min_market_n=2000, ratio_threshold=2.0, min_excess=12) -> pandas.DataFrame`

**Why the report is in flags, not in distances.** "This market's distribution differs from pooled by a Kolmogorov–Smirnov statistic of 0.07" is unactionable. "Pooling costs this market 31 extra reviews a year" is the same fact in the currency the project is judged in, and it can be compared directly against the review budget. So the evidence for splitting is expressed as the number of orders pooling puts on a desk that the market's own band would not.

A market earns `split` when **both** hold: its pooled flag rate is at least `ratio_threshold`× its own flag rate, and the excess is at least `min_excess` orders over the fit window. The second condition stops a market with 3 pooled flags and 1 own flag — a ratio of 3.0 on noise — from looking like evidence.

Note the report is deliberately one-directional in interpretation: it measures over-flagging (pooled worse than own). A market that pooling *under*-flags shows a negative excess, which is also worth seeing — it means the pooled band is too wide to catch that market's real problems — so negative values are reported rather than clipped.

- [ ] **Step 1: Write the failing test**

Create `tests/test_split.py`:

```python
import numpy as np
import pandas as pd
import pytest

from perfthreshold import split, schema


def _frame(spec: dict[str, np.ndarray], benchmark="VWAP") -> pd.DataFrame:
    rows = []
    for market, values in spec.items():
        for i, v in enumerate(values):
            rows.append({schema.ORDER_ID: f"{market}{i}",
                         schema.BENCHMARK: benchmark,
                         schema.MARKET: market,
                         schema.METRIC: float(v),
                         schema.ORDER_DATE: pd.Timestamp("2025-07-01")})
    return pd.DataFrame(rows)


def test_identical_markets_are_told_to_stay_pooled():
    rng = np.random.default_rng(0)
    df = _frame({"HK": rng.normal(0, 1, 5000),
                 "JP": rng.normal(0, 1, 5000)})
    rep = split.report(df, k=4.0, min_market_n=100)
    assert set(rep["verdict"]) == {split.VERDICT_POOL}


def test_a_much_tighter_market_is_over_flagged_by_the_pool():
    # HK is tight, IN is wide. The pooled band is dominated by IN, so it is
    # far too wide for HK -- HK is UNDER-flagged, not over-flagged. Reverse it:
    # make the pool tight and one market wide, so the wide market's own band
    # would be far wider than pooled and pooling over-flags it.
    rng = np.random.default_rng(1)
    df = _frame({"HK": rng.normal(0, 1, 20000),
                 "IN": rng.normal(0, 6, 3000)})
    rep = split.report(df, k=4.0, min_market_n=100).set_index("market")
    assert rep.loc["IN", "pooled_flags"] > rep.loc["IN", "own_flags"]
    assert rep.loc["IN", "excess_flags"] > 0
    assert rep.loc["IN", "verdict"] == split.VERDICT_SPLIT


def test_excess_is_exactly_pooled_minus_own():
    rng = np.random.default_rng(2)
    df = _frame({"HK": rng.normal(0, 1, 4000), "IN": rng.normal(0, 5, 4000)})
    rep = split.report(df, k=4.0, min_market_n=100)
    assert (rep["excess_flags"] == rep["pooled_flags"] - rep["own_flags"]).all()


def test_a_thin_market_cannot_earn_its_own_band():
    rng = np.random.default_rng(3)
    df = _frame({"HK": rng.normal(0, 1, 4000), "TH": rng.normal(0, 5, 40)})
    rep = split.report(df, k=4.0, min_market_n=1000).set_index("market")
    assert rep.loc["TH", "verdict"] == split.VERDICT_THIN
    assert np.isnan(rep.loc["TH", "own_hi"])


def test_a_big_ratio_on_tiny_counts_is_not_evidence():
    # 3 pooled flags vs 1 own flag is a ratio of 3.0 on noise. min_excess
    # exists precisely to stop that reading as a reason to split.
    rng = np.random.default_rng(4)
    df = _frame({"HK": rng.normal(0, 1, 5000), "AU": rng.normal(0, 1.05, 3000)})
    rep = split.report(df, k=4.0, min_market_n=100,
                       ratio_threshold=1.01, min_excess=10_000)
    assert set(rep["verdict"]) <= {split.VERDICT_POOL}


def test_every_market_appears_once_per_benchmark():
    rng = np.random.default_rng(5)
    a = _frame({"HK": rng.normal(0, 1, 500), "JP": rng.normal(0, 1, 500)},
               benchmark="VWAP")
    b = _frame({"HK": rng.normal(0, 2, 500), "JP": rng.normal(0, 2, 500)},
               benchmark="TWAP")
    rep = split.report(pd.concat([a, b]), k=4.0, min_market_n=100)
    assert len(rep) == 4
    assert list(rep.columns) == split.SPLIT_COLS
    assert rep.duplicated([schema.BENCHMARK, "market"]).sum() == 0


def test_report_is_ranked_by_the_size_of_the_discrepancy():
    rng = np.random.default_rng(6)
    df = _frame({"HK": rng.normal(0, 1, 8000),
                 "IN": rng.normal(0, 6, 3000),
                 "JP": rng.normal(0, 1.1, 3000)})
    rep = split.report(df, k=4.0, min_market_n=100)
    fitted = rep[rep["verdict"] != split.VERDICT_THIN]
    assert list(fitted["excess_flags"]) == sorted(
        fitted["excess_flags"], reverse=True)


def test_under_flagging_shows_as_a_negative_excess_not_zero():
    # A market far tighter than the pool is under-flagged: the pooled band is
    # too wide to catch its real problems. That is worth seeing, so it is not
    # clipped at zero.
    rng = np.random.default_rng(7)
    df = _frame({"IN": rng.normal(0, 8, 12000), "HK": rng.normal(0, 1, 4000)})
    rep = split.report(df, k=4.0, min_market_n=100).set_index("market")
    assert rep.loc["HK", "excess_flags"] < 0


def test_empty_frame_gives_an_empty_report():
    rep = split.report(pd.DataFrame(columns=[
        schema.BENCHMARK, schema.MARKET, schema.METRIC]), k=4.0)
    assert len(rep) == 0
    assert list(rep.columns) == split.SPLIT_COLS
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_split.py -v`
Expected: FAIL — `ImportError: cannot import name 'split'`

- [ ] **Step 3: Write `perfthreshold/split.py`**

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_split.py -v`
Expected: PASS — 9 passed

- [ ] **Step 5: Commit**

```bash
git add perfthreshold/split.py tests/test_split.py
git commit -m "feat: market-split evidence expressed in review workload"
```

---

## Task 9: Persisting the frozen band artifact

**Files:**
- Create: `perfthreshold/persist.py`
- Test: `tests/test_persist.py`

**Interfaces:**
- Consumes: `fit.BAND_COLS`, `fit.FitResult` (Task 6); `calibrate.CURVE_COLS` (Task 7).
- Produces:
  - `persist.ARTIFACT_VERSION = 1`
  - `persist.BandFile` — dataclass with fields: `version: int`, `created_utc: str`, `metric: str`, `metric_units: str`, `scope: str`, `market_groups: dict`, `k: float`, `percentile: float`, `min_cell_n: int`, `k_mode: str`, `k_reason: str`, `target: int | None`, `fit_start: str | None`, `fit_end: str | None`, `n_orders: int`, `source_file: str`, `source_hash: str`, `bands: pandas.DataFrame`, `medians: dict`, `calibration: pandas.DataFrame`.
  - `persist.file_hash(path: str) -> str` — first 16 hex chars of the file's SHA-256.
  - `persist.write(out_dir: str, band_file: BandFile) -> dict[str, str]` — writes `bands.json` and `bands.csv`, returns `{"json": ..., "csv": ...}`.
  - `persist.read(json_path: str) -> BandFile`

**Why the provenance fields are not optional.** Six months from now someone will ask why an order was flagged. Answering needs the exact k, the exact fit window, the group definitions in force, and which file the numbers came from. None of that can be reconstructed afterwards, and a band file that cannot answer it is a number without a reason. The observed `fit_start`/`fit_end` are the dates actually *seen* in the data, not the window that was requested — a file that was supposed to hold twelve months but holds nine must say nine.

**JSON and NaN.** Python's `json` emits bare `NaN`, which is not valid JSON and breaks every other reader. Non-finite floats are written as `null` and read back as `NaN`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_persist.py`:

```python
import json
import os

import numpy as np
import pandas as pd
import pytest

from perfthreshold import calibrate, fit, groups, load, persist
from tests import synthetic


def _fit_result(tmp_path):
    raw = synthetic.make_book(n_per_month=200, months=12, seed=21)
    csv = tmp_path / "year.csv"
    raw.to_csv(csv, index=False)
    df, rep = load.prepare(raw)
    resolved, _ = groups.resolve(df, scope="all")
    res = fit.fit_cells(resolved, k=4.0, min_cell_n=100)
    curve = calibrate.curve(resolved, ks=[3.0, 4.0], min_cell_n=100, target=5)
    bf = persist.BandFile(
        version=persist.ARTIFACT_VERSION, created_utc="2026-09-09T00:00:00Z",
        metric="ePvwap/Sprd", metric_units="spreads", scope="all",
        market_groups={"ALL": "*"}, k=4.0, percentile=99.5, min_cell_n=100,
        k_mode="target", k_reason="because", target=5,
        fit_start=str(rep.date_min.date()), fit_end=str(rep.date_max.date()),
        n_orders=rep.rows_kept, source_file=str(csv),
        source_hash=persist.file_hash(str(csv)),
        bands=res.bands, medians=res.medians, calibration=curve)
    return bf


def test_write_produces_both_files(tmp_path):
    bf = _fit_result(tmp_path)
    paths = persist.write(str(tmp_path / "out"), bf)
    assert os.path.exists(paths["json"]) and os.path.exists(paths["csv"])
    assert os.path.basename(paths["json"]) == "bands.json"


def test_round_trip_preserves_every_band_number(tmp_path):
    bf = _fit_result(tmp_path)
    paths = persist.write(str(tmp_path / "out"), bf)
    back = persist.read(paths["json"])
    pd.testing.assert_frame_equal(
        back.bands.sort_values("cell_key").reset_index(drop=True),
        bf.bands.sort_values("cell_key").reset_index(drop=True),
        check_dtype=False)


def test_round_trip_preserves_provenance_and_rule_settings(tmp_path):
    bf = _fit_result(tmp_path)
    back = persist.read(persist.write(str(tmp_path / "out"), bf)["json"])
    assert back.k == 4.0
    assert back.percentile == 99.5
    assert back.scope == "all"
    assert back.metric_units == "spreads"
    assert back.market_groups == {"ALL": "*"}
    assert back.fit_start == bf.fit_start and back.fit_end == bf.fit_end
    assert back.source_hash == bf.source_hash
    assert back.n_orders == bf.n_orders
    assert back.k_reason == "because"


def test_round_trip_preserves_reference_medians_and_calibration(tmp_path):
    bf = _fit_result(tmp_path)
    back = persist.read(persist.write(str(tmp_path / "out"), bf)["json"])
    assert set(back.medians) == set(bf.medians)
    assert list(back.calibration.columns) == list(bf.calibration.columns)
    assert len(back.calibration) == len(bf.calibration)


def test_nan_survives_as_nan_and_the_json_stays_valid(tmp_path):
    bf = _fit_result(tmp_path)
    bf.bands.loc[0, "hi"] = np.nan
    paths = persist.write(str(tmp_path / "out"), bf)
    # Must parse with a strict reader: bare NaN is not valid JSON.
    with open(paths["json"], encoding="utf-8") as fh:
        json.loads(fh.read(), parse_constant=_reject)
    back = persist.read(paths["json"])
    assert np.isnan(back.bands.loc[0, "hi"])


def _reject(name):
    raise AssertionError(f"invalid JSON constant emitted: {name}")


def test_reading_a_future_version_is_refused_naming_both_versions(tmp_path):
    bf = _fit_result(tmp_path)
    paths = persist.write(str(tmp_path / "out"), bf)
    with open(paths["json"], encoding="utf-8") as fh:
        blob = json.load(fh)
    blob["version"] = persist.ARTIFACT_VERSION + 99
    with open(paths["json"], "w", encoding="utf-8") as fh:
        json.dump(blob, fh)
    with pytest.raises(ValueError, match=str(persist.ARTIFACT_VERSION)):
        persist.read(paths["json"])


def test_file_hash_is_content_addressed(tmp_path):
    a, b = tmp_path / "a.csv", tmp_path / "b.csv"
    a.write_text("x,y\n1,2\n", encoding="utf-8")
    b.write_text("x,y\n1,2\n", encoding="utf-8")
    assert persist.file_hash(str(a)) == persist.file_hash(str(b))
    b.write_text("x,y\n1,3\n", encoding="utf-8")
    assert persist.file_hash(str(a)) != persist.file_hash(str(b))


def test_csv_carries_the_band_columns_in_order(tmp_path):
    bf = _fit_result(tmp_path)
    paths = persist.write(str(tmp_path / "out"), bf)
    csv = pd.read_csv(paths["csv"])
    assert list(csv.columns) == fit.BAND_COLS


def test_write_creates_a_missing_output_directory(tmp_path):
    bf = _fit_result(tmp_path)
    target = str(tmp_path / "deep" / "nested" / "out")
    paths = persist.write(target, bf)
    assert os.path.exists(paths["json"])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_persist.py -v`
Expected: FAIL — `ImportError: cannot import name 'persist'`

- [ ] **Step 3: Write `perfthreshold/persist.py`**

```python
"""The frozen band artifact: everything needed to score a later month.

Six months from now someone will ask why a particular order was flagged.
Answering needs the exact k, the exact fit window, the group definitions that
were in force, and which file the numbers came from -- and none of it can be
reconstructed afterwards. A band file that cannot answer that question is a
number without a reason, so the provenance fields are not optional.

fit_start and fit_end are the dates ACTUALLY OBSERVED in the data, never the
window that was requested. A file that was meant to hold twelve months but
holds nine has to say nine, because the leakage guard and the drift baseline
both depend on the truth rather than on the intent.

Python's json module emits bare NaN, which is not valid JSON and breaks every
other reader. Non-finite floats go out as null and come back as NaN.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from perfthreshold import calibrate, fit

ARTIFACT_VERSION = 1

BANDS_JSON = "bands.json"
BANDS_CSV = "bands.csv"


@dataclass
class BandFile:
    version: int = ARTIFACT_VERSION
    created_utc: str = ""
    metric: str = ""
    metric_units: str = ""
    scope: str = "all"
    market_groups: dict = field(default_factory=dict)
    k: float = 4.0
    percentile: float = 99.5
    min_cell_n: int = 2000
    k_mode: str = "fixed"          # "fixed" or "target"
    k_reason: str = ""
    target: int | None = None
    fit_start: str | None = None
    fit_end: str | None = None
    n_orders: int = 0
    source_file: str = ""
    source_hash: str = ""
    bands: pd.DataFrame = field(default_factory=lambda:
                                pd.DataFrame(columns=fit.BAND_COLS))
    medians: dict = field(default_factory=dict)
    calibration: pd.DataFrame = field(
        default_factory=lambda: pd.DataFrame(columns=calibrate.CURVE_COLS))


def file_hash(path: str) -> str:
    """First 16 hex chars of the file's SHA-256. Enough to spot a swap."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def _jsonable(value):
    """NaN / inf -> None; numpy scalars -> python scalars."""
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        v = float(value)
        return None if not math.isfinite(v) else v
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if value is pd.NaT or (value is None):
        return None
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value


def _records(df: pd.DataFrame) -> list[dict]:
    return [{k: _jsonable(v) for k, v in row.items()}
            for row in df.to_dict(orient="records")]


def _frame(records: list[dict], columns: list[str]) -> pd.DataFrame:
    out = pd.DataFrame(records, columns=columns)
    if len(out) == 0:
        return out
    for col in columns:
        # null came from NaN; restore it for the numeric columns only, so a
        # genuinely empty string field is not turned into a float.
        if out[col].map(lambda v: isinstance(v, (int, float, type(None)))).all():
            out[col] = pd.to_numeric(out[col], errors="ignore")
    return out


def write(out_dir: str, band_file: BandFile) -> dict[str, str]:
    os.makedirs(out_dir, exist_ok=True)
    json_path = os.path.join(out_dir, BANDS_JSON)
    csv_path = os.path.join(out_dir, BANDS_CSV)

    blob = {
        "version": int(band_file.version),
        "created_utc": band_file.created_utc or datetime.now(
            timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "metric": band_file.metric,
        "metric_units": band_file.metric_units,
        "scope": band_file.scope,
        "market_groups": band_file.market_groups,
        "rule": {
            "k": float(band_file.k),
            "percentile": float(band_file.percentile),
            "min_cell_n": int(band_file.min_cell_n),
            "k_mode": band_file.k_mode,
            "k_reason": band_file.k_reason,
            "target": band_file.target,
        },
        "fit_window": {"start": band_file.fit_start, "end": band_file.fit_end,
                       "n_orders": int(band_file.n_orders)},
        "source": {"file": band_file.source_file,
                   "sha256_16": band_file.source_hash},
        "bands": _records(band_file.bands),
        "reference_medians": {
            cell: {f: _jsonable(v) for f, v in med.items()}
            for cell, med in band_file.medians.items()},
        "calibration": _records(band_file.calibration),
    }

    with open(json_path, "w", encoding="utf-8") as fh:
        # allow_nan=False turns a stray NaN into an error here rather than
        # into a file no other reader can parse.
        json.dump(blob, fh, indent=2, allow_nan=False)

    band_file.bands.to_csv(csv_path, index=False)
    return {"json": json_path, "csv": csv_path}


def read(json_path: str) -> BandFile:
    with open(json_path, encoding="utf-8") as fh:
        blob = json.load(fh)

    version = int(blob.get("version", 0))
    if version != ARTIFACT_VERSION:
        raise ValueError(
            f"Band file {json_path} is artifact version {version}; this build "
            f"reads version {ARTIFACT_VERSION}. Refit rather than guessing at "
            f"the difference.")

    rule_cfg = blob.get("rule", {})
    window = blob.get("fit_window", {})
    source = blob.get("source", {})

    return BandFile(
        version=version,
        created_utc=blob.get("created_utc", ""),
        metric=blob.get("metric", ""),
        metric_units=blob.get("metric_units", ""),
        scope=blob.get("scope", "all"),
        market_groups=blob.get("market_groups", {}),
        k=float(rule_cfg.get("k", 4.0)),
        percentile=float(rule_cfg.get("percentile", 99.5)),
        min_cell_n=int(rule_cfg.get("min_cell_n", 2000)),
        k_mode=rule_cfg.get("k_mode", "fixed"),
        k_reason=rule_cfg.get("k_reason", ""),
        target=rule_cfg.get("target"),
        fit_start=window.get("start"), fit_end=window.get("end"),
        n_orders=int(window.get("n_orders", 0)),
        source_file=source.get("file", ""),
        source_hash=source.get("sha256_16", ""),
        bands=_frame(blob.get("bands", []), fit.BAND_COLS),
        medians={c: {f: (np.nan if v is None else v) for f, v in m.items()}
                 for c, m in blob.get("reference_medians", {}).items()},
        calibration=_frame(blob.get("calibration", []), calibrate.CURVE_COLS),
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_persist.py -v`
Expected: PASS — 9 passed

If `pd.to_numeric(..., errors="ignore")` emits a `FutureWarning` on pandas 2.2, replace that line with an explicit try/except:

```python
        try:
            out[col] = pd.to_numeric(out[col])
        except (TypeError, ValueError):
            pass
```

- [ ] **Step 5: Commit**

```bash
git add perfthreshold/persist.py tests/test_persist.py
git commit -m "feat: versioned band artifact with full provenance"
```

---

## Task 10: Scoring a month — guards, drift, and the review queue

**Files:**
- Create: `perfthreshold/score.py`
- Test: `tests/test_score.py`

**Interfaces:**
- Consumes: `fit.apply`, `fit.BAND_LO/BAND_HI/ZONE` (Task 6); `persist.BandFile` (Task 9); `rule.FLAGGED` (Task 2); `schema` (Task 1).
- Produces:
  - `score.GuardError(ValueError)` — every refusal raises this.
  - `score.OUTLIER_COLS: list[str]`, `score.DRIFT_COLS: list[str]`
  - `score.windows_overlap(a_lo, a_hi, b_lo, b_hi) -> bool` — `False` whenever any bound is `None`.
  - `score.check_guards(band_file, df, scope, metric_units) -> None`
  - `score.drift(df, band_file, threshold_pct=25.0) -> pandas.DataFrame`
  - `score.ScoreResult` — dataclass `scored`, `outliers`, `drift`, `counts: dict`, `month: str`.
  - `score.score_month(df, band_file, scope, metric_units, drift_threshold_pct=25.0) -> ScoreResult`

**The guards, and why each one exists.**

*Leakage.* Scoring a month the band was fitted on produces a flag rate that is circular and looks entirely normal. It is the one mistake that makes the whole out-of-sample exercise meaningless while leaving no trace, so it is refused rather than warned about.

*Unit mismatch.* A band frozen in spreads and a band frozen in bps would sit in one summary table looking comparable. Refused, naming both units.

*Scope mismatch.* A band fitted `--scope groups` has one cell per declared group; applying it to rows resolved `--scope all` would silently give every row `NO_BAND` and report zero outliers — a clean-looking result that means nothing. Refused.

**The drift check.** When a month flags fifteen instead of five there are two entirely different explanations — *the book got harder* or *execution got worse* — and they lead to opposite actions. Comparing the month's reference-feature medians against the medians stamped at fit time separates them. A median that has moved more than the threshold raises a warning that the band was not fitted on orders like these. This is why those medians had to be captured at fit time: they cannot be reconstructed later without refitting the year.

- [ ] **Step 1: Write the failing test**

Create `tests/test_score.py`:

```python
import numpy as np
import pandas as pd
import pytest

from perfthreshold import fit, groups, load, persist, rule, schema, score
from tests import synthetic


def _band_file(scope="all", k=3.0, months=12, seed=31, min_cell_n=100,
               start="2025-07-01"):
    raw = synthetic.make_book(n_per_month=300, months=months, seed=seed,
                              start=start)
    df, rep = load.prepare(raw)
    resolved, _ = groups.resolve(df, scope=scope)
    res = fit.fit_cells(resolved, k=k, min_cell_n=min_cell_n)
    return persist.BandFile(
        metric="ePvwap/Sprd", metric_units="spreads", scope=scope,
        market_groups={"ALL": "*"}, k=k, percentile=99.5,
        min_cell_n=min_cell_n, fit_start=str(rep.date_min.date()),
        fit_end=str(rep.date_max.date()), n_orders=rep.rows_kept,
        bands=res.bands, medians=res.medians)


def _month(scope="all", start="2026-08-01", seed=99, n=400):
    raw = synthetic.make_book(n_per_month=n, months=1, seed=seed, start=start)
    df, _ = load.prepare(raw)
    resolved, _ = groups.resolve(df, scope=scope)
    return resolved


def test_windows_overlap_is_false_when_any_bound_is_unknown():
    assert score.windows_overlap(None, "2026-01-01", "2025-01-01",
                                 "2025-06-01") is False


def test_windows_overlap_detects_a_real_intersection():
    assert score.windows_overlap("2025-07-01", "2026-06-30",
                                 "2026-06-01", "2026-06-30") is True
    assert score.windows_overlap("2025-07-01", "2026-06-30",
                                 "2026-08-01", "2026-08-31") is False


def test_scoring_a_month_inside_the_fit_window_is_refused():
    bf = _band_file()
    inside = _month(start="2026-01-01")   # fit window is 2025-07 .. 2026-06
    with pytest.raises(score.GuardError, match="fit window"):
        score.check_guards(bf, inside, scope="all", metric_units="spreads")


def test_scoring_a_month_after_the_fit_window_is_allowed():
    bf = _band_file()
    after = _month(start="2026-08-01")
    score.check_guards(bf, after, scope="all", metric_units="spreads")


def test_a_unit_mismatch_is_refused_naming_both_units():
    bf = _band_file()
    after = _month(start="2026-08-01")
    with pytest.raises(score.GuardError, match="bps"):
        score.check_guards(bf, after, scope="all", metric_units="bps")


def test_a_scope_mismatch_is_refused_naming_both_scopes():
    bf = _band_file(scope="all")
    after = _month(start="2026-08-01")
    with pytest.raises(score.GuardError, match="groups"):
        score.check_guards(bf, after, scope="groups", metric_units="spreads")


def test_outliers_contain_only_flagged_rows_with_both_tails_possible():
    bf = _band_file(k=1.0)
    after = _month(start="2026-08-01")
    res = score.score_month(after, bf, scope="all",
                            metric_units="spreads")
    assert set(res.outliers[fit.ZONE]) <= {rule.OUT_LOW, rule.OUT_HIGH}
    assert len(res.outliers) == int(
        res.scored[fit.ZONE].isin(list(rule.FLAGGED)).sum())
    assert len(res.outliers) > 0


def test_outliers_are_ranked_by_how_far_outside_the_band_they_are():
    bf = _band_file(k=1.0)
    res = score.score_month(_month(start="2026-08-01"), bf, scope="all",
                            metric_units="spreads")
    assert list(res.outliers["excess"]) == sorted(
        res.outliers["excess"], reverse=True)
    assert (res.outliers["excess"] > 0).all()


def test_outlier_columns_carry_the_diagnostics_needed_to_explain_one():
    bf = _band_file(k=1.0)
    res = score.score_month(_month(start="2026-08-01"), bf, scope="all",
                            metric_units="spreads")
    for col in (schema.ORDER_ID, schema.ORDER_DATE, schema.ALGO,
                schema.BENCHMARK, schema.MARKET, schema.METRIC,
                fit.BAND_LO, fit.BAND_HI, fit.ZONE, "excess",
                schema.SLIPPAGE_BPS, schema.SPREAD_BPS, schema.SIDE):
        assert col in res.outliers.columns, col


def test_counts_report_the_total_and_the_per_cell_split():
    bf = _band_file(k=1.0)
    res = score.score_month(_month(start="2026-08-01"), bf, scope="all",
                            metric_units="spreads")
    assert res.counts["orders"] == len(res.scored)
    assert res.counts["flagged"] == len(res.outliers)
    assert set(res.counts["by_cell"]) <= {"VWAP|ALL", "TWAP|ALL"}
    assert sum(res.counts["by_cell"].values()) == res.counts["flagged"]


def test_month_label_is_recorded():
    bf = _band_file()
    res = score.score_month(_month(start="2026-08-01"), bf, scope="all",
                            metric_units="spreads")
    assert res.month == "2026-08"


def test_drift_is_zero_when_the_month_looks_like_the_fit_window():
    bf = _band_file()
    same = _month(start="2026-08-01", seed=31, n=300)
    d = score.drift(same, bf, threshold_pct=25.0)
    assert list(d.columns) == score.DRIFT_COLS
    assert not d["warn"].any()


def test_drift_warns_when_a_reference_feature_has_moved():
    bf = _band_file()
    harder = _month(start="2026-08-01")
    harder = harder.copy()
    harder[schema.PCT_ADV] = harder[schema.PCT_ADV] * 3.0
    d = score.drift(harder, bf, threshold_pct=25.0).set_index("feature")
    assert bool(d.loc[schema.PCT_ADV, "warn"])
    assert d.loc[schema.PCT_ADV, "pct_change"] > 100.0


def test_orders_in_a_cell_the_band_never_saw_are_no_band_not_flags():
    bf = _band_file(k=1.0)
    bf.bands = bf.bands[bf.bands["cell_key"] == "VWAP|ALL"]
    res = score.score_month(_month(start="2026-08-01"), bf, scope="all",
                            metric_units="spreads")
    twap = res.scored[res.scored[schema.BENCHMARK] == "TWAP"]
    assert set(twap[fit.ZONE]) == {rule.NO_BAND}
    assert res.counts["no_band"] == len(twap)
    assert (res.outliers[schema.BENCHMARK] == "TWAP").sum() == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_score.py -v`
Expected: FAIL — `ImportError: cannot import name 'score'`

- [ ] **Step 3: Write `perfthreshold/score.py`**

```python
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
] + [c for c in schema.DIAGNOSTICS]


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

    d = df[schema.ORDER_DATE].dropna() if schema.ORDER_DATE in df.columns \
        else pd.Series(dtype="datetime64[ns]")
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
    # the order-count-weighted view of "what the book looked like" is the
    # simple median across cells that actually have a value.
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_score.py -v`
Expected: PASS — 14 passed

- [ ] **Step 5: Commit**

```bash
git add perfthreshold/score.py tests/test_score.py
git commit -m "feat: month scoring with leakage, unit and scope guards"
```

---

## Task 11: Plots — the distribution and the calibration curve

**Files:**
- Create: `perfthreshold/plots.py`
- Test: `tests/test_plots.py`

**Interfaces:**
- Consumes: `fit.BAND_COLS` row shape (Task 6); `calibrate.CURVE_COLS` (Task 7).
- Produces:
  - `plots.INK: dict[str, str]` — the design tokens (surface, text, grid, series, status).
  - `plots.clip_with_overflow(values, x_lo, x_hi) -> tuple[numpy.ndarray, int, int]` — values clamped into range, plus the counts that fell below and above.
  - `plots.distribution(values, band_row, out_path, title, month_values=None, bins=80) -> str`
  - `plots.calibration(curve_df, out_path, target, chosen_k=None) -> str`

**Design decisions, and why.**

*The histogram is neutral; the bounds carry the color.* The distribution is context — the question the chart answers is *where do the bounds land and which term put them there*. So the bars are a recessive grey and the two candidate terms take categorical slots 1 and 2.

*Binding solid, losing dashed.* Which term bound is the chart's real payload, and it must not be carried by color alone: the winning candidate is a 2px solid line, the losing one 1.5px dashed. Every line is also directly labelled with its value, so identity never depends on the legend.

*Overflow bins rather than a clipped axis.* A handful of extreme orders would otherwise compress the body of the distribution into a single bar. The x-range is the band plus 10% padding; everything beyond is clamped into the edge bins and the counts are printed there. Clamping and hiding are different things — the count has to be on the chart.

*One axis, always.* No second y-scale on either chart.

Charts render to PNG for a report, so a single light theme is committed to deliberately (the dark blocks a web chart would need do not apply to a raster file). Colors are the validated light-mode slots.

- [ ] **Step 1: Write the failing test**

Create `tests/test_plots.py`:

```python
import os

import matplotlib
matplotlib.use("Agg")            # no display in CI; must precede pyplot import

import numpy as np
import pandas as pd
import pytest

from perfthreshold import calibrate, fit, groups, load, plots, rule
from tests import synthetic


def _band_row(k=4.0):
    raw = synthetic.make_book(n_per_month=300, months=12, seed=41)
    df, _ = load.prepare(raw)
    resolved, _ = groups.resolve(df, scope="all")
    res = fit.fit_cells(resolved, k=k, min_cell_n=100)
    row = res.bands.set_index("cell_key").loc["VWAP|ALL"]
    values = resolved.loc[resolved["cell_key"] == "VWAP|ALL",
                          "perf_in_spreads"].to_numpy(dtype=float)
    return values, row


def test_clip_counts_both_overflows_and_clamps_the_values():
    x = np.array([-100.0, -1.0, 0.0, 1.0, 100.0])
    clipped, below, above = plots.clip_with_overflow(x, -2.0, 2.0)
    assert below == 1 and above == 1
    assert clipped.min() == -2.0 and clipped.max() == 2.0
    assert len(clipped) == len(x)


def test_clip_ignores_non_finite_values():
    x = np.array([np.nan, np.inf, 0.0])
    clipped, below, above = plots.clip_with_overflow(x, -1.0, 1.0)
    assert len(clipped) == 1 and below == 0 and above == 0


def test_clip_with_no_overflow_reports_zeroes():
    x = np.array([-0.5, 0.0, 0.5])
    _, below, above = plots.clip_with_overflow(x, -1.0, 1.0)
    assert below == 0 and above == 0


def test_distribution_writes_a_non_empty_png(tmp_path):
    values, row = _band_row()
    out = plots.distribution(values, row, str(tmp_path / "d.png"),
                             title="VWAP | ALL")
    assert os.path.exists(out)
    assert os.path.getsize(out) > 5_000


def test_distribution_creates_a_missing_directory(tmp_path):
    values, row = _band_row()
    out = plots.distribution(values, row,
                             str(tmp_path / "deep" / "nested" / "d.png"),
                             title="VWAP | ALL")
    assert os.path.exists(out)


def test_distribution_handles_an_unfitted_band_without_crashing(tmp_path):
    values, row = _band_row()
    row = row.copy()
    row["lo"] = np.nan
    row["hi"] = np.nan
    out = plots.distribution(values, row, str(tmp_path / "d.png"),
                             title="unfitted")
    assert os.path.exists(out)


def test_distribution_accepts_a_scored_month_overlay(tmp_path):
    values, row = _band_row()
    month = values[:200] * 1.2
    out = plots.distribution(values, row, str(tmp_path / "d.png"),
                             title="VWAP | ALL", month_values=month)
    assert os.path.getsize(out) > 5_000


def test_distribution_on_an_empty_series_still_writes_a_file(tmp_path):
    _, row = _band_row()
    out = plots.distribution(np.array([]), row, str(tmp_path / "d.png"),
                             title="empty")
    assert os.path.exists(out)


def test_calibration_writes_a_non_empty_png(tmp_path):
    curve = pd.DataFrame({
        "k": [3.0, 4.0, 5.0], "n_months": [12] * 3,
        "median_flags": [14.0, 5.0, 2.0], "mean_flags": [15.0, 6.0, 2.5],
        "min_flags": [6, 1, 0], "max_flags": [31, 12, 5],
        "months_over_target": [12, 4, 0]})
    out = plots.calibration(curve, str(tmp_path / "c.png"), target=5,
                            chosen_k=4.0)
    assert os.path.exists(out)
    assert os.path.getsize(out) > 5_000


def test_calibration_without_a_chosen_k_still_renders(tmp_path):
    curve = pd.DataFrame({
        "k": [3.0, 4.0], "n_months": [12, 12],
        "median_flags": [40.0, 30.0], "mean_flags": [41.0, 31.0],
        "min_flags": [20, 15], "max_flags": [60, 45],
        "months_over_target": [12, 12]})
    out = plots.calibration(curve, str(tmp_path / "c.png"), target=5,
                            chosen_k=None)
    assert os.path.exists(out)


def test_calibration_on_an_empty_curve_writes_a_file_saying_so(tmp_path):
    curve = pd.DataFrame(columns=calibrate.CURVE_COLS)
    out = plots.calibration(curve, str(tmp_path / "c.png"), target=5)
    assert os.path.exists(out)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_plots.py -v`
Expected: FAIL — `ImportError: cannot import name 'plots'`

- [ ] **Step 3: Write `perfthreshold/plots.py`**

```python
"""The two charts, and the reasoning behind how they are drawn.

THE DISTRIBUTION PLOT answers one question: where did the bounds land, and
which of the two candidate terms put them there. So the histogram is a
recessive grey -- it is context -- and the four candidate bounds carry the
colour. The winning candidate on each side is a solid line, the loser dashed,
and every line is labelled with its own value: which term bound must not be
carried by colour alone, and must be readable without the legend.

OVERFLOW BINS RATHER THAN A CLIPPED AXIS. A handful of extreme orders would
otherwise squash the body of the distribution into one bar. The x-range is the
band plus ten percent, and everything past it is clamped into the edge bins
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

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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
}

FONT = {"family": "sans-serif", "size": 9}


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
    fig.tight_layout()
    fig.savefig(out_path, facecolor=fig.get_facecolor())
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


def _bound_line(ax, x, color, solid: bool, label: str, y_frac: float):
    if not np.isfinite(x):
        return
    ax.axvline(x, color=color, linewidth=2.0 if solid else 1.5,
               linestyle="-" if solid else "--", zorder=3)
    ax.annotate(f"{label}\n{x:.2f}", xy=(x, y_frac), xycoords=("data", "axes fraction"),
                xytext=(4, 0), textcoords="offset points",
                color=INK["text"], fontsize=8, va="top", ha="left")


def distribution(values, band_row, out_path: str, title: str,
                 month_values=None, bins: int = 80) -> str:
    """Histogram of the metric with all four candidate bounds drawn."""
    x = np.asarray(values, dtype=float).ravel()
    x = x[np.isfinite(x)]

    lo, hi = float(band_row.get("lo", np.nan)), float(band_row.get("hi", np.nan))
    fig, ax = _fig()

    if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
        pad = 0.10 * (hi - lo)
        x_lo, x_hi = lo - pad, hi + pad
    elif x.size:
        x_lo, x_hi = float(np.percentile(x, 0.5)), float(np.percentile(x, 99.5))
        if x_hi <= x_lo:
            x_lo, x_hi = x_lo - 1.0, x_hi + 1.0
    else:
        x_lo, x_hi = -1.0, 1.0

    clipped, below, above = clip_with_overflow(x, x_lo, x_hi)
    if clipped.size:
        ax.hist(clipped, bins=bins, range=(x_lo, x_hi),
                color=INK["hist"], edgecolor=INK["surface"], linewidth=0.4,
                label=f"fit window (n={clipped.size:,})")

    if month_values is not None:
        m, _, _ = clip_with_overflow(month_values, x_lo, x_hi)
        if m.size:
            ax.hist(m, bins=bins, range=(x_lo, x_hi), histtype="step",
                    color=INK["month"], linewidth=2.0,
                    label=f"scored month (n={m.size:,})")

    hi_binds = str(band_row.get("hi_binds", ""))
    lo_binds = str(band_row.get("lo_binds", ""))
    _bound_line(ax, float(band_row.get("sigma_hi", np.nan)), INK["sigma"],
                hi_binds == "sigma", "mean+k*sd", 0.98)
    _bound_line(ax, float(band_row.get("sigma_lo", np.nan)), INK["sigma"],
                lo_binds == "sigma", "mean-k*sd", 0.98)
    _bound_line(ax, float(band_row.get("p_hi", np.nan)), INK["pct"],
                hi_binds == "percentile", "P-high", 0.72)
    _bound_line(ax, float(band_row.get("p_lo", np.nan)), INK["pct"],
                lo_binds == "percentile", "P-low", 0.72)

    for count, xpos, ha in ((below, x_lo, "left"), (above, x_hi, "right")):
        if count:
            ax.annotate(f"{count:,} beyond", xy=(xpos, 0.02),
                        xycoords=("data", "axes fraction"),
                        xytext=(6 if ha == "left" else -6, 0),
                        textcoords="offset points", ha=ha,
                        color=INK["text_secondary"], fontsize=8)

    ax.set_xlim(x_lo, x_hi)
    ax.set_xlabel("performance (spreads)", color=INK["text_secondary"])
    ax.set_ylabel("orders", color=INK["text_secondary"])
    ax.set_title(title, color=INK["text"], fontsize=11, loc="left")

    caption = (f"k={band_row.get('k', float('nan')):.2f}  "
               f"P{band_row.get('percentile', float('nan')):.1f}  "
               f"band [{lo:.2f}, {hi:.2f}]  "
               f"solid = the term that bound (low: {lo_binds or 'none'}, "
               f"high: {hi_binds or 'none'})")
    ax.annotate(caption, xy=(0, -0.16), xycoords="axes fraction",
                color=INK["text_secondary"], fontsize=8)

    handles, labels = ax.get_legend_handles_labels()
    if len(handles) >= 2:
        ax.legend(frameon=False, fontsize=8, labelcolor=INK["text_secondary"])
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
                    alpha=0.15, linewidth=0,
                    label="month-to-month range")
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
                    xy=(float(chosen_k), 0.95), xycoords=("data", "axes fraction"),
                    xytext=(5, 0), textcoords="offset points",
                    color=INK["text"], fontsize=8, va="top")

    ax.set_xlabel("k  (sigma multiple)", color=INK["text_secondary"])
    ax.set_ylabel("flags per month (out of sample)",
                  color=INK["text_secondary"])
    ax.set_title("What each k costs in review workload",
                 color=INK["text"], fontsize=11, loc="left")
    ax.set_ylim(bottom=0)
    ax.legend(frameon=False, fontsize=8, labelcolor=INK["text_secondary"])
    return _save(fig, out_path)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_plots.py -v`
Expected: PASS — 11 passed

- [ ] **Step 5: Look at the output before believing it**

The validator checks colour, not layout. Render one of each and open them:

```bash
python -c "import matplotlib; matplotlib.use('Agg'); import tests.synthetic as s; from perfthreshold import load, groups, fit, calibrate, plots; raw=s.make_book(n_per_month=400,months=12,seed=1); df,_=load.prepare(raw); r,_=groups.resolve(df,scope='all'); res=fit.fit_cells(r,k=4.0,min_cell_n=100); row=res.bands.set_index('cell_key').loc['VWAP|ALL']; v=r.loc[r['cell_key']=='VWAP|ALL','perf_in_spreads'].to_numpy(); plots.distribution(v,row,'scratch/dist.png',title='VWAP | ALL'); c=calibrate.curve(r,ks=calibrate.k_grid(2.0,8.0,0.5),min_cell_n=100,target=5); plots.calibration(c,'scratch/cal.png',target=5,chosen_k=4.0); print('ok')"
```

Check for label collisions where two bounds land close together, y-axis overflow, and whether the overflow annotations sit on top of bars. Fix any collision by nudging the `y_frac` values in `distribution` before committing.

- [ ] **Step 6: Commit**

```bash
git add perfthreshold/plots.py tests/test_plots.py
git commit -m "feat: distribution and calibration charts"
```

---

## Task 12: The CLI, the reports, and the end-to-end run

**Files:**
- Create: `perfthreshold/cli.py`, `perfthreshold/__main__.py`, `README.md`
- Test: `tests/test_cli.py`

**Interfaces:**
- Consumes: every module from Tasks 1–11.
- Produces:
  - `cli.main(argv: list[str] | None = None) -> int` — `0` on success, `2` on a refused run.
  - `cli.build_parser() -> argparse.ArgumentParser`
  - `cli.safe(name: str) -> str` — a filesystem-safe token (`"VWAP|ALL"` → `"VWAP_ALL"`).
  - `__main__.py` makes `python -m perfthreshold ...` work.

**Commands and their outputs.**

`check --csv FILE` — prints the distinct `Strategy` values with counts and the benchmark each maps to (or **UNMAPPED**), the market suffixes present, per-column missing counts, and the date range. Writes nothing. This is how `ALGO_BENCHMARK` gets completed from evidence rather than from memory, and it is run before trusting any number the other commands produce.

`fit --csv FILE --out DIR` — writes `bands.json`, `bands.csv`, `calibration.csv`, `calibration.png`, `split_report.csv`, `cleaning_report.csv`, `distribution_<cell>.png`, `summary.md`.

`score --csv FILE --bands PATH --out DIR` — writes `scored_orders.csv`, `outliers.csv`, `drift.csv`, `distribution_<cell>.png`, `summary.md`.

`--k` and `--target` are mutually exclusive; with neither, the default is `--target 5`. The calibration curve is computed and printed in both modes — a fixed k still has to show what it costs.

- [ ] **Step 1: Write the failing test**

Create `tests/test_cli.py`:

```python
import json
import os

import matplotlib
matplotlib.use("Agg")

import pandas as pd
import pytest

from perfthreshold import cli, persist
from tests import synthetic


def _year_and_month(tmp_path, n_per_month=250, seed=51):
    """Thirteen months: the first twelve are the fit file, the last is scored."""
    book = synthetic.make_book(n_per_month=n_per_month, months=13,
                               start="2025-07-01", seed=seed)
    period = pd.to_datetime(book["Date"]).dt.to_period("M").astype(str)
    year = book[period < "2026-07"]
    month = book[period == "2026-07"]
    year_path = tmp_path / "year.csv"
    month_path = tmp_path / "month.csv"
    year.to_csv(year_path, index=False)
    month.to_csv(month_path, index=False)
    return str(year_path), str(month_path)


def test_safe_makes_a_cell_key_a_filename():
    assert cli.safe("VWAP|ALL") == "VWAP_ALL"
    assert cli.safe("TWAP|APAC TIGHT") == "TWAP_APAC_TIGHT"


def test_check_runs_and_names_the_unmapped_strategies(tmp_path, capsys):
    book = synthetic.make_book(n_per_month=20, months=2, unmapped_rows=5)
    path = tmp_path / "book.csv"
    book.to_csv(path, index=False)
    assert cli.main(["check", "--csv", str(path)]) == 0
    out = capsys.readouterr().out
    assert "PART" in out and "UNMAPPED" in out
    assert "VWAP" in out and "TWAP" in out


def test_fit_writes_every_declared_artifact(tmp_path):
    year, _ = _year_and_month(tmp_path)
    out = tmp_path / "fitdir"
    rc = cli.main(["fit", "--csv", year, "--out", str(out),
                   "--k", "4", "--min-cell-n", "100",
                   "--k-grid", "2,8,0.5"])
    assert rc == 0
    for name in ("bands.json", "bands.csv", "calibration.csv",
                 "calibration.png", "split_report.csv",
                 "cleaning_report.csv", "summary.md"):
        assert (out / name).exists(), name
    assert (out / "distribution_VWAP_ALL.png").exists()
    assert (out / "distribution_TWAP_ALL.png").exists()


def test_fit_freezes_the_k_it_used(tmp_path):
    year, _ = _year_and_month(tmp_path)
    out = tmp_path / "fitdir"
    cli.main(["fit", "--csv", year, "--out", str(out), "--k", "3.5",
              "--min-cell-n", "100", "--k-grid", "2,8,1"])
    bf = persist.read(str(out / "bands.json"))
    assert bf.k == 3.5
    assert bf.k_mode == "fixed"
    assert bf.metric_units == "spreads"
    assert bf.scope == "all"


def test_fit_in_target_mode_solves_for_k_and_records_why(tmp_path):
    year, _ = _year_and_month(tmp_path, n_per_month=400, seed=52)
    out = tmp_path / "fitdir"
    assert cli.main(["fit", "--csv", year, "--out", str(out),
                     "--target", "5", "--min-cell-n", "100",
                     "--k-grid", "2,8,0.25"]) == 0
    bf = persist.read(str(out / "bands.json"))
    assert bf.k_mode == "target"
    assert bf.target == 5
    assert "flags/month" in bf.k_reason
    assert bf.k >= 2.0


def test_k_and_target_together_are_rejected(tmp_path):
    year, _ = _year_and_month(tmp_path)
    with pytest.raises(SystemExit):
        cli.main(["fit", "--csv", year, "--out", str(tmp_path / "o"),
                  "--k", "4", "--target", "5"])


def test_score_produces_a_review_queue(tmp_path):
    year, month = _year_and_month(tmp_path)
    fitdir, scoredir = tmp_path / "fitdir", tmp_path / "scoredir"
    cli.main(["fit", "--csv", year, "--out", str(fitdir), "--k", "3",
              "--min-cell-n", "100", "--k-grid", "2,8,1"])
    rc = cli.main(["score", "--csv", month, "--bands",
                   str(fitdir / "bands.json"), "--out", str(scoredir)])
    assert rc == 0
    for name in ("scored_orders.csv", "outliers.csv", "drift.csv",
                 "summary.md"):
        assert (scoredir / name).exists(), name
    outliers = pd.read_csv(scoredir / "outliers.csv")
    scored = pd.read_csv(scoredir / "scored_orders.csv")
    assert len(scored) > 0
    assert set(outliers["zone"]) <= {"OUT_LOW", "OUT_HIGH"}


def test_scoring_a_month_inside_the_fit_window_exits_non_zero(tmp_path, capsys):
    year, _ = _year_and_month(tmp_path)
    fitdir = tmp_path / "fitdir"
    cli.main(["fit", "--csv", year, "--out", str(fitdir), "--k", "4",
              "--min-cell-n", "100", "--k-grid", "2,8,1"])
    # Score the fit file against its own band: pure leakage.
    rc = cli.main(["score", "--csv", year, "--bands",
                   str(fitdir / "bands.json"), "--out", str(tmp_path / "s")])
    assert rc == 2
    assert "fit window" in capsys.readouterr().err


def test_end_to_end_the_calibrated_k_lands_near_the_target(tmp_path):
    # The promise of the whole design: fit a year with a review budget, score
    # the next month, and get roughly that many orders to look at.
    year, month = _year_and_month(tmp_path, n_per_month=500, seed=53)
    fitdir, scoredir = tmp_path / "fitdir", tmp_path / "scoredir"
    assert cli.main(["fit", "--csv", year, "--out", str(fitdir),
                     "--target", "5", "--min-cell-n", "100",
                     "--k-grid", "2,8,0.25"]) == 0
    assert cli.main(["score", "--csv", month, "--bands",
                     str(fitdir / "bands.json"),
                     "--out", str(scoredir)]) == 0
    outliers = pd.read_csv(scoredir / "outliers.csv")
    # Generous upper bound: one month is one draw from the range the
    # calibration reported, not the median. The point is the order of
    # magnitude -- single digits, not forty.
    assert len(outliers) <= 20


def test_score_writes_a_distribution_plot_per_cell(tmp_path):
    year, month = _year_and_month(tmp_path)
    fitdir, scoredir = tmp_path / "fitdir", tmp_path / "scoredir"
    cli.main(["fit", "--csv", year, "--out", str(fitdir), "--k", "3",
              "--min-cell-n", "100", "--k-grid", "2,8,1"])
    cli.main(["score", "--csv", month, "--bands", str(fitdir / "bands.json"),
              "--out", str(scoredir)])
    assert (scoredir / "distribution_VWAP_ALL.png").exists()


def test_a_missing_input_file_exits_non_zero_without_a_traceback(tmp_path,
                                                                capsys):
    rc = cli.main(["check", "--csv", str(tmp_path / "nope.csv")])
    assert rc == 2
    assert "nope.csv" in capsys.readouterr().err
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_cli.py -v`
Expected: FAIL — `ImportError: cannot import name 'cli'`

- [ ] **Step 3: Write `perfthreshold/cli.py` and `perfthreshold/__main__.py`**

`perfthreshold/__main__.py`:

```python
import sys

from perfthreshold.cli import main

if __name__ == "__main__":
    sys.exit(main())
```

`perfthreshold/cli.py`:

```python
"""Three commands: check the extract, fit a year, score a month.

Fit and score are separate commands rather than one run with a flag, and that
separation is the whole leakage guard. A single command holding both the year
and the month in memory could always be made to score the month it fitted --
by a bug, a default, or a hurry. Two commands with a file between them cannot.

`check` exists because two things about an extract cannot be read off a column
name: which benchmark family each strategy belongs to, and whether the columns
you need are actually populated. Both are invisible failures -- a band fitted
on the wrong benchmark still fits -- so they get their own command, run first.
"""

from __future__ import annotations

import argparse
import os
import sys

import pandas as pd

from perfthreshold import (calibrate, config, fit, groups, load, persist,
                           plots, schema, score, split)


def safe(name) -> str:
    """A filesystem-safe token. Cell keys contain '|' and group names spaces."""
    cleaned = "".join(c if (c.isalnum() or c in "-_") else "_" for c in str(name))
    return cleaned.strip("_") or "UNKNOWN"


# --------------------------------------------------------------------------
# check
# --------------------------------------------------------------------------
def cmd_check(args) -> int:
    raw = load.read_any(args.csv)
    print(f"file      : {args.csv}")
    print(f"rows      : {len(raw):,}")
    print(f"columns   : {len(raw.columns)}")

    print("\n--- 1) STRATEGY -> BENCHMARK -------------------------------")
    print("Complete config.ALGO_BENCHMARK from this table. An UNMAPPED "
          "strategy is excluded, never defaulted.")
    if "Strategy" in raw.columns:
        counts = raw["Strategy"].astype(str).str.strip().value_counts()
        for strategy, n in counts.items():
            bench = config.benchmark_for(strategy)
            label = bench if bench else "*** UNMAPPED -- WILL BE EXCLUDED ***"
            print(f"  {strategy:<20} {n:>8,}  -> {label}")
    else:
        print("  no 'Strategy' column found")

    print("\n--- 2) MARKETS ---------------------------------------------")
    if "Sym" in raw.columns:
        mkt = raw["Sym"].astype(str).str.strip().str[-2:].str.upper()
        for market, n in mkt.value_counts().items():
            group = config.market_group_for(market, config.MARKET_GROUPS)
            print(f"  {market:<6} {n:>8,}  -> group {group}")
    else:
        print("  no 'Sym' column found")

    print("\n--- 3) COLUMN COMPLETENESS ---------------------------------")
    inverse = {v: k for k, v in config.COLUMN_MAP.items()}
    for canonical in schema.REQUIRED + schema.REFERENCE_FEATURES:
        source = inverse.get(canonical, canonical)
        if source in raw.columns:
            missing = int(pd.to_numeric(raw[source], errors="coerce").isna().sum()
                          if canonical not in (schema.ORDER_ID, schema.ALGO,
                                               schema.SYMBOL, schema.ORDER_DATE)
                          else raw[source].isna().sum())
            flag = "REQUIRED" if canonical in schema.REQUIRED else "reference"
            print(f"  {source:<14} {flag:<10} missing/non-numeric: {missing:,}")
        else:
            flag = "REQUIRED" if canonical in schema.REQUIRED else "reference"
            print(f"  {source:<14} {flag:<10} *** ABSENT ***")

    print("\n--- 4) LOADING IT FOR REAL ---------------------------------")
    df, rep = load.prepare(raw)
    print(rep.summary())
    print(f"\nmetric    : {config.METRIC_COLUMN} ({config.METRIC_UNITS})")
    return 0


# --------------------------------------------------------------------------
# fit
# --------------------------------------------------------------------------
def _parse_grid(text: str) -> list[float]:
    start, stop, step = (float(p) for p in str(text).split(","))
    return calibrate.k_grid(start, stop, step)


def cmd_fit(args) -> int:
    df, clean = load.load(args.csv)
    print(clean.summary())

    resolved, excluded = groups.resolve(
        df, scope=args.scope, market_groups=config.MARKET_GROUPS,
        benchmark=args.benchmark)
    for reason, n in excluded.items():
        if n:
            print(f"  excluded by scope ({reason}): {n:,}")
    if len(resolved) == 0:
        print("No orders left after scope resolution.", file=sys.stderr)
        return 2

    ks = _parse_grid(args.k_grid)
    target = config.DEFAULT_TARGET if (args.k is None and args.target is None) \
        else args.target

    print(f"\nCalibrating over {len(ks)} values of k, "
          f"{len(calibrate.months(resolved))} months, leave-one-month-out ...")
    curve = calibrate.curve(resolved, ks=ks, percentile=args.percentile,
                            min_cell_n=args.min_cell_n,
                            target=target if target is not None else 5)
    print(curve.to_string(index=False))

    if args.k is not None:
        k, k_mode = float(args.k), "fixed"
        near = curve.iloc[(curve["k"] - k).abs().argsort().iloc[0]]
        reason = (f"k={k:.2f} was supplied. Nearest calibrated point "
                  f"(k={near['k']:.2f}) gives a median of "
                  f"{near['median_flags']:.1f} flags/month "
                  f"(range {int(near['min_flags'])}-{int(near['max_flags'])}).")
    else:
        choice = calibrate.choose_k(curve, target=target)
        print("\n" + choice.reason)
        if not choice.reachable:
            # Not an error: the book may genuinely hold more outliers than the
            # budget allows. Fall back to the configured k and say so.
            k, k_mode = config.DEFAULT_K, "target"
            reason = choice.reason + (f" Fell back to the configured "
                                      f"k={k:.2f}.")
        else:
            k, k_mode, reason = choice.k, "target", choice.reason

    result = fit.fit_cells(resolved, k=k, percentile=args.percentile,
                           min_cell_n=args.min_cell_n)
    print("\n" + result.bands.to_string(index=False))

    splits = split.report(resolved, k=k, percentile=args.percentile,
                          min_market_n=args.min_cell_n)

    band_file = persist.BandFile(
        metric=config.METRIC_COLUMN, metric_units=config.METRIC_UNITS,
        scope=args.scope, market_groups=config.MARKET_GROUPS,
        k=float(k), percentile=float(args.percentile),
        min_cell_n=int(args.min_cell_n), k_mode=k_mode, k_reason=reason,
        target=target,
        fit_start=(str(clean.date_min.date()) if clean.date_min is not None
                   else None),
        fit_end=(str(clean.date_max.date()) if clean.date_max is not None
                 else None),
        n_orders=len(resolved), source_file=os.path.abspath(args.csv),
        source_hash=persist.file_hash(args.csv),
        bands=result.bands, medians=result.medians, calibration=curve)

    paths = persist.write(args.out, band_file)
    curve.to_csv(os.path.join(args.out, "calibration.csv"), index=False)
    splits.to_csv(os.path.join(args.out, "split_report.csv"), index=False)
    clean.to_frame().to_csv(os.path.join(args.out, "cleaning_report.csv"),
                            index=False)

    if not args.no_plots:
        plots.calibration(curve, os.path.join(args.out, "calibration.png"),
                          target=target if target is not None else 5,
                          chosen_k=k)
        for _, row in result.bands.iterrows():
            cell = row["cell_key"]
            values = resolved.loc[resolved[schema.CELL_KEY] == cell,
                                  schema.METRIC].to_numpy(dtype=float)
            plots.distribution(
                values, row,
                os.path.join(args.out, f"distribution_{safe(cell)}.png"),
                title=str(cell).replace("|", " | "))

    _write_fit_summary(args.out, band_file, curve, splits, clean, target)
    print(f"\nWrote {paths['json']}")
    return 0


def _write_fit_summary(out_dir, band_file, curve, splits, clean, target):
    lines = [
        "# Fit summary", "",
        f"- metric: `{band_file.metric}` ({band_file.metric_units})",
        f"- scope: `{band_file.scope}`",
        f"- fit window: {band_file.fit_start} .. {band_file.fit_end}",
        f"- orders fitted: {band_file.n_orders:,}",
        f"- rule: `hi = MAX(mean + {band_file.k:g}*sd, "
        f"P{band_file.percentile:g})`, mirrored on the low side",
        f"- k mode: {band_file.k_mode}"
        + (f" (target {target}/month)" if target is not None else ""), "",
        "## Why this k", "", band_file.k_reason, "",
        "## Bands", "", band_file.bands.to_markdown(index=False), "",
        "## Calibration (leave-one-month-out)", "",
        curve.to_markdown(index=False), "",
        "## Market split evidence", "",
        (splits.to_markdown(index=False) if len(splits)
         else "_no markets met the minimum order count_"), "",
        "## Cleaning", "", "```", clean.summary(), "```", "",
    ]
    with open(os.path.join(out_dir, "summary.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))


# --------------------------------------------------------------------------
# score
# --------------------------------------------------------------------------
def cmd_score(args) -> int:
    band_file = persist.read(args.bands)
    df, clean = load.load(args.csv)
    print(clean.summary())

    resolved, excluded = groups.resolve(
        df, scope=band_file.scope, market_groups=band_file.market_groups)
    for reason, n in excluded.items():
        if n:
            print(f"  excluded by scope ({reason}): {n:,}")

    result = score.score_month(resolved, band_file, scope=band_file.scope,
                               metric_units=config.METRIC_UNITS,
                               drift_threshold_pct=args.drift_threshold)

    os.makedirs(args.out, exist_ok=True)
    result.scored.to_csv(os.path.join(args.out, "scored_orders.csv"),
                         index=False)
    result.outliers.to_csv(os.path.join(args.out, "outliers.csv"), index=False)
    result.drift.to_csv(os.path.join(args.out, "drift.csv"), index=False)

    if not args.no_plots:
        for _, row in band_file.bands.iterrows():
            cell = row["cell_key"]
            month_values = resolved.loc[resolved[schema.CELL_KEY] == cell,
                                        schema.METRIC].to_numpy(dtype=float)
            # The scored month against the FROZEN bounds. The band file does
            # not carry the fit year's raw values, so this is the month's own
            # distribution with the year's bounds drawn on it -- which is the
            # comparison a reviewer actually needs.
            plots.distribution(
                month_values, row,
                os.path.join(args.out, f"distribution_{safe(cell)}.png"),
                title=f"{str(cell).replace('|', ' | ')} -- {result.month}")

    _write_score_summary(args.out, band_file, result)
    print(f"\n{result.month}: {result.counts['flagged']} flagged out of "
          f"{result.counts['orders']:,} orders "
          f"({result.counts['no_band']} with no band)")
    for cell, n in sorted(result.counts["by_cell"].items()):
        print(f"  {cell}: {n}")
    if result.drift["warn"].any():
        print("\nDRIFT WARNING -- the book does not look like the fit window:")
        print(result.drift[result.drift["warn"]].to_string(index=False))
    return 0


def _write_score_summary(out_dir, band_file, result):
    lines = [
        f"# Review queue -- {result.month}", "",
        f"- band: fitted {band_file.fit_start} .. {band_file.fit_end}, "
        f"k={band_file.k:g}, P{band_file.percentile:g}",
        f"- orders scored: {result.counts['orders']:,}",
        f"- **flagged: {result.counts['flagged']}** "
        f"({result.counts['out_low']} low, {result.counts['out_high']} high)",
        f"- no band: {result.counts['no_band']}", "",
        "## Expected, from the fit's own calibration", "",
        band_file.k_reason, "",
        "## Outliers", "",
        (result.outliers.to_markdown(index=False) if len(result.outliers)
         else "_nothing outside the band this month_"), "",
        "## Drift against the fit window", "",
        result.drift.to_markdown(index=False), "",
        "A median that has moved more than the threshold means the band was "
        "not fitted on orders like these -- the book got harder, rather than "
        "execution getting worse.", "",
    ]
    with open(os.path.join(out_dir, "summary.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))


# --------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="perfthreshold",
        description="Spread-normalised outlier thresholds for algo execution.")
    sub = p.add_subparsers(dest="command", required=True)

    c = sub.add_parser("check", help="inspect an extract before trusting it")
    c.add_argument("--csv", required=True)
    c.set_defaults(func=cmd_check)

    f = sub.add_parser("fit", help="fit a year and freeze the bands")
    f.add_argument("--csv", required=True)
    f.add_argument("--out", required=True)
    f.add_argument("--scope", default=groups.SCOPE_ALL,
                   help="all | groups | group:NAME")
    f.add_argument("--benchmark", default=None,
                   help="restrict to one benchmark family")
    f.add_argument("--percentile", type=float, default=config.PERCENTILE)
    f.add_argument("--min-cell-n", type=int, default=config.MIN_CELL_N,
                   dest="min_cell_n")
    f.add_argument("--k-grid", default="{},{},{}".format(*config.K_GRID),
                   dest="k_grid", help="start,stop,step for the k search")
    f.add_argument("--no-plots", action="store_true", dest="no_plots")
    mode = f.add_mutually_exclusive_group()
    mode.add_argument("--k", type=float, default=None,
                      help="fix k instead of solving for a target")
    mode.add_argument("--target", type=int, default=None,
                      help="flags per month, total across all cells")
    f.set_defaults(func=cmd_fit)

    s = sub.add_parser("score", help="apply a frozen band to one month")
    s.add_argument("--csv", required=True)
    s.add_argument("--bands", required=True, help="path to bands.json")
    s.add_argument("--out", required=True)
    s.add_argument("--drift-threshold", type=float, default=25.0,
                   dest="drift_threshold")
    s.add_argument("--no-plots", action="store_true", dest="no_plots")
    s.set_defaults(func=cmd_score)

    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except (score.GuardError, ValueError, FileNotFoundError) as exc:
        # A refusal is a result, not a crash: no traceback, exit 2.
        print(str(exc), file=sys.stderr)
        return 2
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_cli.py -v`
Expected: PASS — 11 passed

- [ ] **Step 5: Run the whole suite**

Run: `python -m pytest -q`
Expected: PASS — all tests from Tasks 1–12, no failures.

- [ ] **Step 6: Write `README.md`**

```markdown
# PerfThreshold

Which of last month's algo orders went far enough from normal to be worth a
human's attention -- and few enough of them that each one gets explained.

Performance is `ePvwap/Sprd`, taken from the extract already divided by the
spread. Dividing by the spread is what lets a wide Indian small cap and a tight
Japanese large cap share one band: a 12 bps miss is a rounding error in one and
a disaster in the other.

## The rule

Per cell, per side:

    hi = MAX(mean + k*sd,  P99.5)
    lo = MIN(mean - k*sd,  P0.5)

Two-sided, because a result that looks too *good* is usually a data or
benchmark error and you want to know about those too. Each side takes the wider
of its own two candidates, so the bounds follow the real skew rather than being
forced equidistant from the mean.

A cell is `(benchmark family x market group)`. Algos sharing a benchmark are
measured against the same reference price, so they share a distribution shape --
and pooling them buys the sample size a P99.5 estimate needs.

## k is chosen, not assumed

`k = 4` is a statement about a Gaussian; this book is not one. So `fit` measures
what each k costs, **leave-one-month-out**: for each month in the year, fit on
the other eleven and score that one. Twelve out-of-sample observations, and
therefore a range rather than a single flattering in-sample number.

    --target 5     solve for the smallest k whose median monthly count is <= 5
    --k 4          fix k; the curve is still printed, so you see what it costs

The budget is the total across every cell, and k is one global value -- a
badly-behaved group cannot buy itself a wider band, it just contributes more of
the five.

## Running it

    pip install -r requirements.txt

    python -m perfthreshold check --csv year.csv

Read section 1 of that output closely and complete `ALGO_BENCHMARK` in
`config.py`. An unmapped strategy is excluded and named -- never defaulted --
because a band fitted on the wrong benchmark still fits and never says so.

    python -m perfthreshold fit --csv year.csv --out fits/2025-07_2026-06/
    python -m perfthreshold score --csv aug2026.csv \
        --bands fits/2025-07_2026-06/bands.json --out review/2026-08/

`fit` writes `bands.json` (the frozen artifact -- scoring needs nothing else),
`bands.csv`, `calibration.csv` / `.png`, `split_report.csv`,
`cleaning_report.csv`, a distribution plot per cell, and `summary.md`.

`score` writes `scored_orders.csv`, `outliers.csv` (the queue),
`drift.csv`, a distribution plot per cell, and `summary.md`.

## Scope

    --scope all                 every market pooled; one cell per benchmark
    --scope groups              one cell per declared MARKET_GROUPS entry
    --scope group:APAC_TIGHT    that group only
    --benchmark VWAP            and orthogonally, one family

The scope is stamped into the band file and enforced on apply.

## What it refuses

- Scoring a month that overlaps the fit window. The flag rate would be circular
  and would look entirely normal.
- A band whose metric units differ from what is being scored.
- A band fitted at one scope applied at another -- every order would score
  `NO_BAND` and the run would report zero outliers.
- Banding a cell below `MIN_CELL_N`. A thin cell's sd is biased low because the
  tail has not been sampled yet, which makes the band too *narrow* and the small
  market look like the worst in the book.

## Should markets be pooled?

`split_report.csv` answers it in review workload rather than in distributional
distance: for each market, how many flags does the pooled band produce that the
market's own band would not? A market being handed 31 extra reviews a year by
pooling has earned its own cell. Declaring a grouping (`MARKET_GROUPS`) and
testing one (this report) stay separate -- config decides what gets fitted, the
report says whether that was justified.
```

- [ ] **Step 7: Commit**

```bash
git add perfthreshold/cli.py perfthreshold/__main__.py README.md tests/test_cli.py
git commit -m "feat: check/fit/score CLI, reports and README"
```

---

## Final verification

- [ ] Run the full suite: `python -m pytest -q` — every test passes.
- [ ] Run `check`, `fit` and `score` end to end on a generated synthetic year and confirm by eye that `summary.md`, `calibration.png` and one `distribution_*.png` read correctly.
- [ ] Confirm `git log --oneline` shows one commit per task.
