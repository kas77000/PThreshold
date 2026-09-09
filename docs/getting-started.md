# Getting started — your first run on real data

Start to finish: from an extract to a review queue, and what to read on the way.

---

## Step 0 — Export two files, not one

This is the part to get right, because the tool refuses the alternative. `fit`
and `score` are separate commands with a file between them, and scoring a month
the band already saw is blocked outright — a flag rate measured on the same
orders that set the threshold is circular, and it looks entirely normal while
being meaningless.

| File | Contents |
|---|---|
| `year.csv` | ~12 months, e.g. **Jul 2025 → Jun 2026** |
| `month.csv` | **one later month**, e.g. Jul or Aug 2026 — no overlap with the year |

### Minimum header

Same extract as the Threshold project:

```
aggrTgtId, Strategy, Sym, Date, ePvwap/Sprd, Pvwap, Sprd
```

### Worth including, and cheap

| Columns | Why |
|---|---|
| `%Adv, Vol, PR, Dur` | These become the **drift baseline**. Their medians are stamped into `bands.json` at fit time and compared against each scored month, which is what separates *"the book got harder"* from *"execution got worse"*. This can only be captured at fit time — leave them out and you cannot reconstruct it later without refitting the whole year. |
| `$Mln, #Shares, Side, %POST, %OPEN, %CLOSE, Rev30min` | These ride into `outliers.csv` as the diagnostic columns, so each flagged order can actually be explained rather than just named. |

`.csv`, `.xlsx` and `.parquet` all work. Install the dependencies once:

```bash
pip install -r requirements.txt
```

---

## Step 1 — Check the extract before trusting any number it produces

```bash
cd C:\Users\user\Desktop\Projects\PerfThreshold
python -m perfthreshold check --csv year.csv
```

This command exists because two things about an extract cannot be read off a
column name: which benchmark family each strategy belongs to, and whether the
columns you need are actually populated. Both are invisible failures — a band
fitted on the wrong benchmark still fits, still produces a plausible curve, and
never announces itself.

**Read section 1 closely.** It lists every distinct `Strategy` with its row count
and the benchmark it maps to. Anything marked
`*** UNMAPPED -- WILL BE EXCLUDED ***` is dropped from the entire exercise —
excluded and named, never quietly defaulted into a family.

Section 2 gives rows per market. Section 3 gives missing-value counts per
required column. Section 4 loads the file for real and prints the cleaning
report, where `rows kept` plus every drop count always equals `rows in`.

---

## Step 2 — Edit two values in `perfthreshold/config.py`

### `ALGO_BENCHMARK`

Add every real strategy name that section 1 showed you. Your desk's algos are
almost certainly not literally called `VWAP` and `TMX`:

```python
ALGO_BENCHMARK = {
    "VWAP": "VWAP",
    "TWAP": "TWAP",
    "TMX":  "TWAP",     # the desk's name for the TWAP family
    # add the real names here, e.g.:
    # "VWAP_PASSIVE": "VWAP",
    # "TMX_AGGR":     "TWAP",
}
```

Keys are matched upper-cased and stripped, so case and stray whitespace don't
matter. Complete this map from what `check` printed, never from memory.

### `MIN_CELL_N`

Currently `2000`, a placeholder chosen without sight of your order counts. It is
the minimum number of orders a cell needs before a band is fitted on it at all.

It matters because a thin cell fails in a counter-intuitive direction: with few
observations the tail has probably not been sampled yet, so `sd` comes out
**too small**, the band comes out **too narrow**, and the smallest market looks
like the worst offender in the book purely because it is small.

Leave it at 2000 while everything is pooled (`--scope all`). It becomes the
decisive setting the moment you split APAC into groups.

**Re-run `check` until nothing you want to keep is listed as UNMAPPED.**

---

## Step 3 — Fit

```bash
python -m perfthreshold fit --csv year.csv --out fits\2025-07_2026-06\ --k-grid 3,6,0.25
```

Note there is **no `--k` and no `--target`**. That gives you `k = 4` from
`config.DEFAULT_K` — set in advance from policy, not selected from the alert
count. See [`../README.md`](../README.md) for why that ordering is the whole
point.

The narrower `--k-grid` keeps this first run quick. The default grid
(2 → 8 by 0.1) is 61 × 12 = 732 fits; perfectly fine once you know the file
loads, and it takes seconds to low minutes depending on size.

### What `fit` writes

| File | What it is |
|---|---|
| `bands.json` | The frozen artifact. `score` needs nothing else. Carries the rule, the observed fit window, group definitions, reference medians, the calibration table and provenance (source filename + content hash). |
| `bands.csv` | The same per-cell rows, readable. |
| `calibration.csv` / `.png` | What every k costs, leave-one-month-out. |
| `split_report.csv` | Per-market evidence on whether pooling is justified. |
| `cleaning_report.csv` | Every excluded row, counted by reason. |
| `distribution_<cell>.png` | The distribution with all four candidate bounds drawn, a legend, the band in bps, and **the number of orders outside the band** as the subtitle. |
| `summary.md` | All of the above, readable. |

---

## Step 4 — The three numbers to read

### 1. `sd ÷ mad_sigma`, in `bands.csv`

This settles how many alerts to expect, and it is the number to have in hand
before anyone predicts anything. `sd` is the classical scale; `mad_sigma` is the
robust one (1.4826 × MAD). Under normality they are equal by construction, and
the ratio rises with tail weight:

| `sd / mad_sigma` | Regime | Alerts/month per 3,000 orders |
|---|---|---|
| ≈ 1.00 | normal | ~0.2 (about 2 per **year**) |
| 1.10 | mildly fat | ~5 |
| 1.20 | fat | ~11 |
| ≥ 1.30 | very fat | 15+ |

Under normality `mean ± 4σ` covers **99.9937%** of the population — so the
expectation that the rule covers almost the whole spectrum is correct *if* the
book is near-normal. This ratio is what tells you whether it is.

### 1b. `spread_bps_median`, in `bands.csv`

The metric is unitless — performance divided by the spread — so the bounds come
out in *spreads*, not bps. This column is what converts them back:

    band [-4.68, 4.33] spreads  x  median spread 9.0 bps  =  [-42, +39] bps

That translation is printed on every distribution chart, and `spread_bps_mean`
is carried alongside. A cell that inherited its bounds from a pooled parent
still reports **its own** spread — the bounds may be borrowed, but the orders
are not.

### 2. `hi_binds` / `lo_binds`, in `bands.csv`

Expect `sigma` on every row. σ inflates faster than the 99.5th percentile does,
so at k = 4 the `MAX` always picks the sigma term — on every tail thickness from
Gaussian to Student-t with 3 degrees of freedom. In practice the rule is
`mean ± 4σ` and the percentile is a floor that never activates.

Two consequences worth knowing before you are asked:

- You are defending a **4σ rule**, not a two-part rule.
- `MAX` takes the **wider** bound and therefore flags **fewer** orders. If the
  intent behind including P99.5 was *"we always examine at least the worst 0.5%
  of each tail"*, the rule does not do that — that guarantee needs `MIN`, not
  `MAX`. The two are opposite in effect, and the band file records which term
  bound on every cell.

### 3. `median_flags` at k = 4.00, in `calibration.csv`

Together with its `min_flags`–`max_flags` range. This is the honest
out-of-sample estimate of monthly review load: twelve months, each scored by a
band fitted on the other eleven, so no month ever helped set the threshold that
judges it.

The **range** is the number that matters. A median of 5 with a worst month of 12
is a different proposition from a median of 5 with a worst month of 6, and an
average alone hides the difference.

---

## Step 5 — Score the month

```bash
python -m perfthreshold score --csv month.csv ^
    --bands fits\2025-07_2026-06\bands.json --out review\2026-07\
```

| File | What it is |
|---|---|
| `outliers.csv` | The review queue, ranked by `excess` — how far outside the band each order sits, in spreads. Carries the diagnostic columns needed to explain each one. |
| `scored_orders.csv` | Every order with its cell, bounds and zone. |
| `drift.csv` | This month's reference-feature medians against the fit-time baseline. |
| `distribution_<cell>.png` | The month's distribution against the frozen bounds. |
| `summary.md` | The readable version. |

---

## Reading the charts

Every distribution chart carries, without needing the CSVs:

- **The subtitle** — how many orders fall outside the band, split low/high, as a
  count and a percentage. This is the review load that band implies.
- **A legend** — blue is the `mean ± k·sd` term, orange is the percentile term;
  **solid** is the bound actually in force and **dashed** is the candidate that
  lost. Which term bound is the chart's real payload, so it is never carried by
  colour alone.
- **The `N beyond` labels** at each edge — orders clamped into the overflow bins.
  The x-axis is trimmed to the band plus a margin so a handful of extremes
  cannot squash the body of the distribution into one bar; clamping and hiding
  are different things, so the counts stay on the chart.
- **The caption** — k, the band in spreads, the same band in bps at the median
  spread, and which term bound on each side.

After `score`, the count is also the first thing printed:

```
==========================================================
  2026-07   ->   10 ORDERS TO REVIEW
==========================================================
  out of 3,000 scored   (5 low, 5 high, 0 with no band)
    TWAP|ALL                   4
    VWAP|ALL                   6
  queue   : review6-07\outliers.csv
  summary : review6-07\summary.md
```

---

## Why aren't `lo` and `hi` mirror images?

Because `mean ± k·sd` is symmetric about the **mean**, and the mean is not zero.
For a book averaging −0.18 spreads, the whole band sits shifted down by that
amount: `mean − lo` and `hi − mean` are identical, while `|lo| > |hi|`.

Worth knowing what that implies: **a mean-centred band cannot see systematic
underperformance.** If the whole book drifted to −0.5 spreads next year, the band
would slide down with it and flag just as few orders. It answers *"is this order
unusual for this algo family?"* — not *"did this order miss the benchmark?"*
Centring on zero would answer the second question instead. The shipped rule is
mean-centred as specified; `drift.csv` is the compensating control, since it
shows the centre moving even when the flag count does not.

---

## Three ways the first run bites

| Symptom | Cause | Fix |
|---|---|---|
| Every order comes back `NO_BAND` | The cell holds fewer than `MIN_CELL_N` (2000) orders — usual when smoke-testing on a small slice | Add `--min-cell-n 100` for the test, then put it back |
| `exit 2`, *"overlaps the band's fit window"* | The two files share a month | Working as intended. Re-cut the export so they don't overlap |
| *"Extract is missing required column(s): …"* | A column was renamed in the extract | Map the new name in `config.COLUMN_MAP` |

---

## Scope: pooled, grouped, or one group

The first run should be `--scope all` (the default): every market pooled, one
cell per benchmark. That is defensible because the metric is already divided by
the spread, which puts a wide small cap and a tight large cap on one scale
before the band is fitted.

When you want to test grouping, declare the groups in `config.MARKET_GROUPS`
and then choose how they are used:

```bash
--scope all                 every market pooled; one cell per benchmark
--scope markets             one cell per (benchmark x market)
--scope groups              one cell per declared MARKET_GROUPS entry
--scope group:APAC_TIGHT    that group only; other markets excluded entirely
--benchmark VWAP            and orthogonally, restrict to one family
```

`--scope markets` needs no declaration — the market *is* the group — so nothing
can be dropped by forgetting to list it, which is the failure mode of spelling
every market into `MARKET_GROUPS` by hand.

**`MIN_CELL_N` becomes the binding constraint here.** Splitting a year across
markets and benchmarks divides the book many ways; any cell that lands below the
minimum inherits its benchmark-pooled parent's bounds and records
`fallback_from`, so check that column before trusting a per-market band. A cell
showing a `fallback_from` value was not fitted on its own orders.

The scope is stamped into `bands.json` and enforced when the band is applied, so
a band fitted one way cannot be silently scored another.

Before changing the grouping, read `split_report.csv` from the pooled fit. It
answers *should this market be split out?* in review workload rather than in
abstract distributional distance: for each market, how many flags does the
**pooled** band produce that the market's **own** band would not. A market being
handed 31 extra reviews a year by pooling has earned its own cell.

---

## Known gaps to close before this is production

- **Arrival-benchmarked algos (`PART`, `POV`, IS) receive no surveillance.**
  They are excluded at load like any unmapped strategy. Interval VWAP and
  arrival price are different benchmarks and must never share a band, so this
  needs either a second band fitted on `eIS/Sprd` or a documented compensating
  control. An unmonitored population is a harder question to answer than a
  debatable threshold.
- **`MIN_CELL_N` is still a placeholder.** Set it from real counts before
  splitting markets.
- **`bands.json` records provenance but not governance** — who approved the
  parameters, when they take effect, when they are reviewed. That is the
  difference between a script's output and a controlled artifact.
