# PerfThreshold — design

**Date:** 2026-09-09
**Status:** approved, ready for implementation planning

---

## The problem

A year of algo orders executed in the market. Which of next month's orders went
far enough from normal to deserve a human's attention?

The answer has to be a *short* list. A threshold that surfaces forty orders a
month is not a threshold, it is a second job. The target is around **five orders
per month across the whole book** — a number small enough that each one gets
explained properly.

The predecessor project (`Threshold`, `tier5_standalone/` on this machine)
solved a similar problem and its column contract, fit/score split and leakage
guard are reused here. What is rethought: the pooling key, and — the substantive
change — treating the monthly flag count as the objective the fit is *solved
against* rather than a number you discover afterwards.

---

## 1. Inputs

### The extract

Same header as the `Threshold` project:

```
aggrTgtId, Strategy, Sym, Date, ePvwap/Sprd, eIS/Sprd, Pvwap, Sprd,
%Adv, Vol, PR, Dur, $Mln, #Shares, Side, %POST, %OPEN, %CLOSE, Rev30min
```

| Column | Role |
|---|---|
| `aggrTgtId` | order id — required |
| `Strategy` | algo name; maps to a benchmark family — required |
| `Sym` | ticker; last two characters give the market (`"0700 HK"` → `HK`) — required |
| `Date` | order date; drives windowing and the leakage guard — required |
| `ePvwap/Sprd` | **the banded metric**, supplied pre-divided — required |
| `Sprd`, `Pvwap` | carried for the drift report and for readability in `outliers.csv` |
| `%Adv`, `Vol`, `PR`, `Dur` | reference features; medians stamped at fit time |
| `$Mln`, `#Shares`, `Side`, `%POST`, `%OPEN`, `%CLOSE`, `Rev30min` | diagnostics on flagged orders only |

### The metric

`ePvwap/Sprd` — performance versus interval VWAP, already divided by the
spread at source. **Consumed as supplied; this project never performs the
division itself.**

Units are *spreads*, and that is the point. A 12 bps miss is a rounding error in
a wide Indian small cap and a disaster in a tight Japanese large cap. Dividing by
the spread puts every market on one scale, which is what makes pooling markets
legal at all.

The band file records `metric_units = "spreads"`. Scoring refuses a band whose
units differ from what it is scoring — two different units in one summary table
is a silent, plausible-looking error.

### Benchmark families

There is no benchmark column. The mapping lives in `config.py`, keyed on
`Strategy`, matched case-insensitively:

```python
ALGO_BENCHMARK = {
    # "<STRATEGY>": "<benchmark family>"
    # completed from `check` output against the real file
}
```

Two families in v1: **TWAP** (the desk calls it TMX) and **VWAP**.

**An unmapped `Strategy` is excluded and reported by name and row count. It is
never defaulted into a family.** A band fitted on the wrong benchmark still fits,
still produces a plausible curve, and is invisible afterwards — so the failure
mode has to be loud at the point of loading.

`eIS/Sprd` (arrival-benchmarked: PART, POV, IS) is out of scope for v1. Those
algos are excluded like any other unmapped strategy. Arrival slippage carries
the market's drift over the order's life and interval VWAP does not; the two must
never share a band.

---

## 2. The cell

A cell is `(benchmark_family, market_group)`.

### Why benchmark is the pooling key

Algos sharing a benchmark are measured against the same reference price, so their
performance distributions share a shape. Two VWAP-benchmarked algos differ mainly
in aggression, which shifts the *centre* — not the family of the distribution.

Sample size is the reason this matters. A P99.5 estimate is only as good as the
number of orders beyond it: 25,000 orders puts ~125 orders past each percentile
bound, which is enough to estimate one. Per-(market × algo) cells are where
counts get thin and a percentile becomes the average of three orders.

### Market grouping and scope

Groups are declared in config:

```python
MARKET_GROUPS = {"ALL": "*"}                    # v1 default: everything pooled
# MARKET_GROUPS = {
#     "APAC_TIGHT": ["HK", "JP", "AU", "SG"],
#     "APAC_WIDE":  ["IN", "KR", "TW", "TH", "ID", "PH", "MY"],
# }
```

The scope selector decides how those declarations are used at fit time:

| Scope | Cells produced |
|---|---|
| `--scope all` | one cell per benchmark; every market pooled |
| `--scope groups` | one cell per (benchmark × declared group) |
| `--scope group:APAC_TIGHT` | that group only; all other markets excluded |

Orthogonally, `--benchmark VWAP` restricts to a single family.

All three scopes share one code path — a scope only changes how `groups.py`
assigns a row its cell key. The resolved scope, and the group definitions in
force, are stamped into `bands.json`; scoring refuses a band whose scope does not
match the way it is being applied.

A cell with fewer than `MIN_CELL_N` orders cannot support a percentile estimate.
Under `--scope groups` or `--scope group:NAME` it falls back to its
benchmark-pooled parent band, and **the fallback is recorded in the band file** —
a cell must never look fitted when it was inherited. Under `--scope all` a cell
*is* the benchmark-pooled band, so there is no parent to inherit from: a thin
cell there is reported as unfittable and its orders score as `NO_BAND` rather
than being banded on too little evidence.

### Why pooling markets is the right default, and when it stops being

Spread normalisation is the argument for pooling. What survives it — and is
therefore the only legitimate reason to split — is structural: tick-size regimes,
closing-auction dominance in JP and HK, and thin books where the spread itself is
noisy and so inflates the tails of the ratio.

Section 5 measures whether any of that actually bites, in flag counts rather than
in abstract distributional distance.

---

## 3. The rule

Per cell, per side:

```
hi = MAX(mean + k·σ,   P99.5)
lo = MIN(mean − k·σ,   P0.5)
```

Two-sided and symmetric in treatment: both tails are flagged alike, because a
result that looks too *good* is usually a data or benchmark error and you want to
know about those too. The bounds are not forced numerically equidistant from the
mean — each side takes the wider of its own two candidates, so the percentile
terms follow the real skew of the book.

`σ` is the classical sample standard deviation (`ddof=1`) and `mean` the
arithmetic mean. The σ term is kept literal — `mean + k·σ`, nothing solved,
nothing adjusted — so that "mean plus four sigma" is visibly what it says.

### Which term binds, recorded per side

On a fat-tailed book, σ is inflated by the very orders the band exists to catch,
so `mean + kσ` widens to swallow them and the P99.5 floor may never bind — the
rule degenerating silently to pure `kσ`. The fit therefore records, per cell per
side, **which of the two candidates won**. If the percentile is dead weight
everywhere, the report says so in words rather than leaving it to be inferred
from two near-identical columns.

### The robust pair, as a diagnostic only

Alongside the shipped rule, every cell also computes
`centre = median(x)`, `scale = 1.4826 · MAD(x)`. Under normality the two
estimators agree; the gap between them *is* the non-normality, expressed in the
band's own units. It is reported, never active. The shipped rule stays classical
as specified.

---

## 4. Calibration — solving for k against the objective

The success criterion is a count, so the fit solves for it directly.

### Why k cannot be chosen a priori

`k = 4` is a statement about a Gaussian; this book is not one. On a fat-tailed
distribution 4σ might flag forty orders a month or zero, and which of those it is
cannot be read off the number 4. So k is chosen against a measured curve.

### Leave-one-month-out, so the count is honest

Counting flags on the same year that was fitted is circular — those orders shaped
the σ that judges them. Instead, for each of the twelve months in the fit window:
fit the band on the other eleven, score that month. Twelve genuinely
out-of-sample months, each an unbiased sample of what a real month costs.

This yields a *distribution* of monthly counts, not a single rate:

| k | median flags/month | min–max across months | months over target |
|---|---|---|---|
| 3.0 | 14 | 6 – 31 | 12 |
| 3.5 | 8 | 3 – 19 | 10 |
| 4.0 | 5 | 1 – 12 | 4 |
| 4.5 | 3 | 0 – 8 | 1 |
| 5.0 | 2 | 0 – 5 | 0 |

*(illustrative shape; the real table comes from the extract)*

The range column is the one that matters. A median of 5 with a worst month of 12
is a different proposition from a median of 5 with a worst month of 6, and an
average alone hides the difference.

### Two modes, mutually exclusive

| Mode | Behaviour |
|---|---|
| `--target 5` *(default)* | search k for the smallest value whose **median out-of-sample monthly count across all cells combined** is ≤ target; report the k found and what it costs |
| `--k 4` | fix k; the curve is still computed and printed |

The budget is **total across all groups**, per the requirement. k is a **single
global value** applied to every cell — so a badly-behaved group cannot buy itself
a wider band, it simply contributes more of the five, which is the correct
signal. The k actually used is frozen into `bands.json`; scoring can never apply
a different one than the fit chose.

Search is over a bounded grid of k (e.g. 2.0 → 8.0 by 0.1). If no k in range
reaches the target, the fit says so and reports the best achievable count rather
than silently returning the boundary value.

---

## 5. The market-split evidence report

Independent of what config declares, the fit asks of every market **in scope**
with `n ≥ MIN_CELL_N` — so under `--scope group:APAC_TIGHT` the report covers
that group's markets only, since the excluded markets were not loaded:

> How many flags does the **pooled** band give this market, versus how many its
> **own** band would give?

A market where the pooled band flags 5% of orders while its own band flags 0.5%
is being systematically over-flagged by the pool. That is the evidence to split
it — and it is stated in review workload, the currency the project is judged in,
rather than as a distributional distance nobody can act on.

Output is one row per market, ranked by the size of that discrepancy. Declaring a
grouping and testing a grouping stay two separate layers: config decides what
gets fitted, the report says whether the decision was justified.

---

## 6. Outputs

### Fit stage → `fits/<window>/`

**`bands.json`** — the frozen artifact carried to next month. Scoring needs
nothing else. Contents:

- fit window: the min/max date actually observed, not what was requested
- metric name and units; rule parameters (k, percentiles, sidedness)
- resolved scope and the market-group definitions in force
- per cell: `n`, mean, σ, median, 1.4826·MAD, P0.5, P99.5, final `lo`/`hi`,
  and which term bound on each side
- any cell that fell back to its parent, and to which parent
- reference-feature medians: `%Adv`, `Vol`, `PR`, `Dur`, `Sprd`
- the calibration table — the file records what flag rate was expected of it
- provenance: source filename, row count, content hash

**`bands.csv`** — the same per-cell rows, readable.

**`calibration.csv` / `calibration.png`** — k on the x-axis, out-of-sample
flags/month on the y, median line with the min–max month band around it, and a
horizontal line at the target.

**`distribution_<cell>.png`** — histogram of `ePvwap/Sprd` with all four
candidate bounds drawn: the σ bound and the percentile bound on each side, the
binding one solid and the losing candidate dashed. So the plot answers "which
term is doing the work" at a glance. X-axis clipped with explicit overflow bins —
otherwise a handful of extreme orders compress the body of the distribution into
a single bar.

**`split_report.csv`** — the per-market evidence of section 5.

**`cleaning_report.csv`** — every excluded row counted by reason. Printed first,
before any fitted number.

### Score stage → `review/<month>/`

**`scored_orders.csv`** — every order with its cell, the bounds applied, its
value, and its zone (`IN_RANGE` / `OUT_LOW` / `OUT_HIGH` / `NO_BAND`).

**`outliers.csv`** — the short list, with the diagnostic columns that let each
one be explained: `Side`, `%POST`, auction %, `Rev30min`, `$Mln`, `%Adv`, `Vol`,
`PR`, `Dur`, plus `Pvwap` in bps next to the ratio.

**`summary.md`** — count against expectation, the drift table, links to plots.

**`distribution_<cell>.png`** — redrawn with the scored month overlaid on the fit
distribution, flagged orders marked.

### The drift check

When August flags fifteen instead of five there are two entirely different
explanations — *the book got harder* or *execution got worse* — and they lead to
opposite actions. Comparing August's reference-feature medians against the
medians stamped at fit time separates them. A median moving more than 25% raises
a warning that the band was not fitted on orders like these.

That baseline can only be captured at fit time, which is why it is captured
whether or not it was asked for. It cannot be reconstructed later without
refitting the year.

---

## 7. Guards

| Guard | Behaviour |
|---|---|
| **Leakage** | scoring a month that overlaps the band's fit window is refused. A circular flag rate looks entirely normal. |
| **Unit mismatch** | a band whose `metric_units` differ from the scored metric is refused. |
| **Scope mismatch** | a band fitted `--scope groups` cannot be applied as `--scope all`. |
| **Unmapped strategy** | excluded, named, counted. Never defaulted. |
| **Thin cell** | below `MIN_CELL_N`, falls back to parent and records the fallback. |
| **Unreachable target** | if no k in the search range hits the target, say so and report the best achievable. |

---

## 8. Code shape

```
perfthreshold/
  schema.py     canonical column names
  config.py     THE ONLY FILE EDITED — column map, ALGO_BENCHMARK,
                MARKET_GROUPS, k / target defaults, PERCENTILE (99.5, mirrored
                to 0.5 on the low side), MIN_CELL_N, k search grid
  load.py       read → validate → normalise → tidy frame + cleaning report
  groups.py     resolve each row's (benchmark, market_group) cell under a scope
  rule.py       pure: array + k + percentile → bounds. No pandas, no config, no I/O.
  fit.py        fit every cell over the fit window
  calibrate.py  k-curve, leave-one-month-out counts, target search
  split.py      per-market pooled-vs-own flag evidence
  persist.py    write/read bands.json, versioned
  score.py      apply a frozen band to one month + drift check
  plots.py      distribution + calibration charts
  cli.py        check / fit / score
tests/
```

`rule.py` is pure by design: the arithmetic that matters most — does `MAX` pick
the right term, does the percentile land where expected — is testable against
hand-computed arrays with no fixtures and no data file.

### Commands

```bash
python -m perfthreshold check --csv year.csv

python -m perfthreshold fit --csv year.csv --out fits/2025-07_2026-06/ \
    [--scope all|groups|group:NAME] [--benchmark VWAP] [--target 5 | --k 4]

python -m perfthreshold score --csv aug2026.csv \
    --bands fits/2025-07_2026-06/bands.json --out review/2026-08/
```

`check` prints what a column name cannot tell you: distinct `Strategy` values
with counts and the benchmark each maps to (or that it is unmapped), the market
suffixes present, missing-value counts per required column, and the date range.
It is run before trusting any number the other two commands produce, and it is
how `ALGO_BENCHMARK` gets completed from evidence rather than from memory.

---

## 9. Testing

Built test-first. The cases that must hold:

- known array + known k → known bounds; `MAX`/`MIN` select the correct term
- percentile-only and sigma-only cells both bind correctly and are labelled
- an unmapped strategy is excluded, not defaulted
- the leakage guard fires on an overlapping month
- a unit mismatch and a scope mismatch are both refused
- a cell below `MIN_CELL_N` falls back and records the fallback
- leave-one-month-out never lets a scored month enter its own fit
- the target search returns the smallest qualifying k, and reports failure
  honestly when none qualifies
- end-to-end: synthetic year → fit → score a held-out month → outlier count is
  within the calibrated range

---

## 10. Out of scope for v1

No ADV or size bucketing. No regression-based expected-cost model. No per-algo
cells inside a benchmark family. No automatic market clustering. No
arrival-benchmarked (`eIS/Sprd`) families.

Each is a plausible v2. None is needed to answer *which orders do I check this
month*.
