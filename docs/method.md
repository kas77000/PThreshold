# Method — what this program does to your data, step by step

This document exists so that the method can be attacked on its merits rather
than on its opacity. It states, in order, every operation performed between the
raw extract and the review queue: the arithmetic, where each number comes from,
what is recorded, what is refused, and which choices are judgement calls open to
challenge. Nothing here is a summary of intent — each stage names the function
that implements it, so any claim below can be checked against the code.

Companion documents: [`getting-started.md`](getting-started.md) is the operating
procedure; [`fat-tails.md`](fat-tails.md) is the measured difficulty each choice
below is a response to; [`2026-09-09-perfthreshold-design.md`](2026-09-09-perfthreshold-design.md)
is why each choice was made. This one is *what happens*.

---

## 0. The claim being made

> Given a year of algo orders, produce for each subsequent month a short,
> ranked list of orders whose execution performance fell outside a band fitted
> on that year — where the band's width was fixed by policy before the year was
> looked at, and every order that entered or left the calculation is accounted
> for.

Three things follow from that wording, and they are the method's whole shape:

1. **The band is fitted once, frozen to a file, and applied to months it has
   never seen.** Fit and score are separate commands with a file between them
   (`perfthreshold/cli.py::cmd_fit`, `::cmd_score`).
2. **The threshold parameter `k` is an input, not an output.** It is not
   selected by how many alerts it produces. The alert count is measured, out of
   sample, and reported as a consequence.
3. **Row accounting is an invariant, not an aspiration.** `rows_in` equals
   `rows_kept` plus the sum of named drop reasons, always.

---

## 1. The pipeline in one view

```
  raw extract (csv/xlsx/parquet)
        │
        │  load.read_any        read by extension
        ▼
  ┌───────────────────────────────────────────────────────────┐
  │ 1  load.prepare                                           │
  │    rename to canonical names, derive market/benchmark,    │
  │    coerce numerics, drop invalid rows under a named       │
  │    reason. → CleaningReport (rows_in = kept + dropped)    │
  └───────────────────────────────────────────────────────────┘
        │
        │  groups.resolve(scope)   attach market_group, cell_key
        ▼                          drop what the scope excludes
  ┌─────────────────────────┐        ┌──────────────────────────────┐
  │ FIT  (a year)           │        │ SCORE  (one later month)     │
  │                         │        │                              │
  │ 2 calibrate.curve       │        │ 6 score.check_guards         │
  │   leave-one-month-out   │        │   leakage / units / scope    │
  │   flag count for each k │        │        │                     │
  │        │                │        │        ▼                     │
  │        ▼                │        │ 7 fit.apply                  │
  │ 3 k from policy         │        │   band_lo, band_hi, zone     │
  │   (config.DEFAULT_K)    │        │        │                     │
  │        │                │        │        ▼                     │
  │        ▼                │        │ 8 rank by `excess`           │
  │ 4 fit.fit_cells         │        │   → outliers.csv (the queue) │
  │   one band per cell     │        │        │                     │
  │   thin cells inherit    │        │        ▼                     │
  │        │                │        │ 9 score.drift                │
  │        ▼                │        │   this month's medians vs    │
  │ 5 persist.write ────────┼───────▶│   the ones stamped at fit    │
  │   bands.json (frozen)   │ bands  │                              │
  │   + split.report        │ .json  │                              │
  └─────────────────────────┘        └──────────────────────────────┘
```

Everything below expands one box.

---

## 2. Stage 1 — Loading, and accounting for every row

Implemented in `perfthreshold/load.py`. Called by every command.

**2.1 Required source columns are checked before anything is renamed**
(`_require_source_columns`). The error names the *raw* column
(`aggrTgtId`, not `order_id`), because that is the name the person exporting
the file can act on.

**2.2 Columns are renamed through `config.COLUMN_MAP`, and anything unmapped is
dropped.** An unmapped source column has no canonical meaning; carrying it
would invite later code to reach for a vendor name.

**2.3 Two fields are derived, not read:**

| Derived | From | Rule |
|---|---|---|
| `market` | `Sym` | last two characters, upper-cased (`"0700 HK"` → `HK`) |
| `benchmark` | `Strategy` | `config.ALGO_BENCHMARK` lookup, stripped and upper-cased |

`ALGO_BENCHMARK` is the one mapping a human must supply. A strategy absent from
it is **excluded and named with its row count** — never defaulted into a family.
This is the method's most consequential guard: a band fitted on the wrong
benchmark still fits, still produces a plausible curve, and never announces
itself.

**2.4 Numeric coercion.** The metric and the reference features go through
`pd.to_numeric(errors="coerce")`; `±inf` in the metric becomes `NaN`, because
an infinite value would poison both `mean` and `sd` silently. Dates are parsed
with `errors="coerce"` and normalised to midnight.

**2.5 Rows are dropped under exactly one reason, in this order:**

```
1  missing order id
2  missing symbol            (and therefore no market)
3  missing or unparseable date
4  unmapped strategy         (reported by name, with counts)
5  missing metric
```

A row failing several checks is counted under the **first** one only. That is
what makes the counts sum to the number of rows actually removed, so the
identity

```
rows_in  ==  rows_kept + Σ dropped[reason]
```

holds by construction and is asserted by
`tests/test_load.py::test_every_input_row_is_accounted_for`. The report also
records the **observed** date window (`date_min`, `date_max`) — never the window
that was requested — because the leakage guard and the drift baseline both
depend on what the file actually contains.

Written out as `cleaning_report.csv` and reprinted in `summary.md`.

---

## 3. Stage 1b — Cells: what gets pooled with what

Implemented in `perfthreshold/groups.py::resolve`. **Fit and score both derive
cells through this one function**, which is what guarantees the band written for
`VWAP|APAC_TIGHT` is the band looked up for those same rows later.

A cell key is `"<benchmark>|<market_group>"`. The `market_group` depends only on
`--scope`:

| `--scope` | `market_group` becomes | Cells |
|---|---|---|
| `all` *(default)* | the literal `ALL` for every row; declared groups are ignored | one per benchmark |
| `markets` | the market itself | one per (benchmark × market) |
| `groups` | the matching `config.MARKET_GROUPS` entry; a market in no group is excluded and counted | one per (benchmark × group) |
| `group:NAME` | as `groups`, then every other group is excluded and counted | that group's cells only |
| `--benchmark VWAP` | orthogonal: rows of other families are excluded and counted | narrows any of the above |

Exclusions are returned as counts by reason and printed, so scope narrowing is
visible in the run log rather than implied by a smaller row count.

**Why benchmark is the pooling key.** Algos sharing a benchmark are measured
against the same reference price, so they share a distribution *shape*; pooling
them also buys the sample size a 99.5th-percentile estimate needs.

**Why markets may be pooled at all.** The metric is already divided by the
spread, which puts a wide small cap and a tight large cap on one scale before
anything is fitted. Whether that is *sufficient* is not assumed — Stage 9
measures it.

---

## 4. Stage 4 — The band arithmetic

Implemented in `perfthreshold/rule.py`. Deliberately pure: numpy in, dict out,
no pandas, no config, no filesystem — so the arithmetic can be tested against
hand-computed arrays with no fixtures.

For one cell's metric values `x` (finite values only; `NaN` and `±inf` are
dropped, never propagated):

```
mean      = mean(x)
sd        = std(x, ddof=1)          sample sd; 0.0 when n == 1
median    = median(x)
mad_sigma = 1.4826 · median(|x − median|)      robust scale, diagnostic only

sigma_hi  = mean + k·sd             p_hi = percentile(x, 99.5)
sigma_lo  = mean − k·sd             p_lo = percentile(x,  0.5)

hi = MAX(sigma_hi, p_hi)            lo = MIN(sigma_lo, p_lo)
```

Five properties of this, each deliberate and each checkable:

**4.1 Per side, independently.** Execution performance is skewed — a book
misses badly far more often than it beats badly — so forcing both tails through
one candidate makes the band wrong on at least one of them. `hi` and `lo` can
bind on different terms, and `lo` is not the mirror of `hi`.

**4.2 Which term won is recorded**, per side, as `hi_binds` / `lo_binds`
(`"sigma"` or `"percentile"`). On a fat-tailed book `sd` is inflated by the very
orders the band exists to catch, so the sigma term can widen past the percentile
floor and the rule degenerates to pure k-sigma. That must be visible in the
output, not inferred.

**4.3 `MAX` takes the *wider* bound, and therefore flags *fewer* orders.** State
this plainly, because it is the natural place to attack the rule: a rule
guaranteeing that the worst 0.5% of each tail is always examined would need
`MIN`, not `MAX`. This one guarantees the opposite — the percentile is a floor
on band *width*, not a floor on review volume.

**4.4 At k = 4 the percentile term is effectively inert.** The crossover — the
`k` below which the percentile still binds — is a property of tail thickness
alone, and it sits well under 4 on every regime this book could plausibly be in:

| Sample (n = 20,000) | `sd / mad_sigma` | percentile binds below k ≈ |
|---|---|---|
| Gaussian | 1.005 | 2.59 |
| Student-t, 5 df | 1.193 | 3.08 |
| Student-t, 3 df | 1.503 | 3.24 |

So in production the rule *is* `mean ± 4σ`, with a floor that never activates.
`hi_binds` / `lo_binds` are how you confirm that on your own data instead of
taking this table's word for it.

**4.5 The robust pair never touches the band.** `median` and `mad_sigma` are
computed and reported so that `sd / mad_sigma` can be read as a measure of
non-normality *in the band's own units* — under normality the two scales agree,
so any gap between them **is** the non-normality. It is a diagnostic and a
regime indicator, nothing more.

### A worked example, hand-checkable

`x = [−2, −1, 0, 1, 2]`, `k = 2`, percentile 99.5:

```
mean = 0        sd = 1.5811 (ddof=1)      mad_sigma = 1.4826
sigma_hi = 0 + 2(1.5811) = 3.1623         p_hi = 1.98   (interpolated)
hi = MAX(3.1623, 1.98) = 3.1623           hi_binds = "sigma"
lo = MIN(−3.1623, −1.98) = −3.1623        lo_binds = "sigma"
```

And on 3,000 draws with `k = 4` (seed 7), showing the regime dependence the
alert count actually rests on:

| Sample | `sd` | `mad_sigma` | ratio | `sigma_hi` | `p_hi` | flags |
|---|---|---|---|---|---|---|
| Gaussian | 0.993 | 1.012 | 0.98 | 3.94 | 2.57 | 0 |
| Student-t, 3 df | 1.644 | 1.154 | 1.43 | 6.58 | 5.59 | 21 |

Same rule, same k, same sample size — nought flags or twenty-one, decided
entirely by tail thickness. This is why the fit measures the count instead of
predicting it, and why `sd / mad_sigma` is the first number to read in
`bands.csv`.

### Zone assignment

`rule.zone` / `fit.apply` label each order:

| Label | Condition |
|---|---|
| `OUT_LOW` | `value < lo` |
| `OUT_HIGH` | `value > hi` |
| `IN_RANGE` | `lo ≤ value ≤ hi` |
| `NO_BAND` | the cell has no fitted band, or value/bounds are non-finite |

**Bounds are inclusive.** An order landing exactly on the edge is in range; a
strict inequality would make the monthly count depend on floating-point luck at
the boundary. `NO_BAND` rows are **not** flags and are reported as their own
count — never folded into either the numerator or the "clean" total.

---

## 5. Stage 4b — Thin cells: the fallback contract

Implemented in `perfthreshold/fit.py::fit_cells`.

A cell with fewer than `MIN_CELL_N` (default 2,000) orders cannot support this
rule. At n = 100 the 99.5th percentile is an interpolation between the first and
second worst order, and `sd` is biased low because the tail has probably not been
sampled yet. The failure mode is counter-intuitive and worth stating precisely:
**a thin cell gets a band that is too narrow, and therefore looks like the worst
offender in the book purely because it is small.**

So, per cell:

```
if n >= min_cell_n:
        fit the cell's own band                fitted = True
else:
        parent = band over ALL in-scope rows of that benchmark
        if parent.n >= min_cell_n and parent.n > n:
                copy the parent's lo/hi         fitted = False
                                                fallback_from = "<BENCH>|__POOLED__"
                n stays the CELL's own n, not the pool's
        else:
                no band at all                  fitted = False, lo = hi = NaN
                → those orders score NO_BAND
```

Two consequences a reader should know without digging:

- Under `--scope all` the cell *is* the pooled band, so `parent.n > n` is false
  and a thin cell is simply left unfitted. There is no rescue, by design:
  banding on too little evidence is worse than not banding.
- An inherited band still reports the **cell's own** `n` and its own
  `spread_bps_median` / `spread_bps_mean`. The bounds came from the parent; the
  orders are these orders.

`spread_bps_median` is carried for one reason: the metric is unitless
(spreads), so without the cell's own spread a bound of `4.33 spreads` cannot be
read back as roughly 35 bps by anyone reviewing it.

Per-cell medians of the reference features (`%Adv`, `Vol`, `PR`, `Dur`, `Sprd`)
are stamped here too. They are the drift baseline in Stage 9 and cannot be
reconstructed after the fact.

---

## 6. Stage 2 & 3 — Where `k` comes from, and what the calibration curve is for

This is the part most likely to be challenged, so the ordering is explicit.

### 6.1 `k` is fixed before the data is consulted

`k = config.DEFAULT_K = 4.0` is the production default and comes from policy.
`mean ± 4σ` covers 99.9937% of a normal population and stands on its own stated
rationale. How many orders fall outside it is then a **finding**.

The reason for that ordering is not statistical, it is evidentiary. A threshold
whose value was selected by how few alerts it produced reads, to any reviewer, as
a threshold tuned to suppress alerts — and the calibration table then becomes a
written record of having tried twenty-five values and kept the least demanding
one. The same table with `k` fixed beforehand is the opposite: evidence that the
parameter's consequences were understood before it was set.

| Invocation | `k` | `k_mode` in `bands.json` |
|---|---|---|
| *(no flag)* | `config.DEFAULT_K` = 4.0, from policy | `fixed` |
| `--k 3.5` | fixed explicitly by the operator | `fixed` |
| `--target 5` | **diagnostic only** — solved from the alert count | `target`, plus a warning banner at the top of `summary.md` |

`--k` and `--target` are mutually exclusive at the parser. In `target` mode the
warning travels *inside the artifact*, so the caveat cannot be separated from
the numbers by someone reading the output a year later.

Whichever mode ran, the k actually used is frozen into `bands.json` and read
back from it at score time. Scoring cannot apply a different k than the fit
used.

### 6.2 The curve is computed either way, leave-one-month-out

Implemented in `perfthreshold/calibrate.py`.

```
for each k in K_GRID (default 2.0 → 8.0 step 0.1, 61 values):
    for each month M in the fit window:
        bands = fit_cells(all months EXCEPT M, k)     # M never sees itself
        counts[M] = flag_count(month M, bands)
    record median, mean, min, max, months_over_target
```

Counting flags on the same year the band was fitted on is circular: those orders
shaped the `sd` that judges them, so the count comes out flattering, every time,
in the same direction. Twelve held-out months give twelve genuinely
out-of-sample observations, and therefore a **range**. The range is the column
that matters — a median of 5 with a worst month of 12 is a different proposition
from a median of 5 with a worst month of 6, and an average alone hides the
difference.

Two implementation choices worth stating because they are the ones that could
hide a bug:

- **The count is produced by `fit.apply`**, the same function scoring uses.
  Calibration is deliberately *not* optimised by deriving bounds from cached
  per-cell statistics — that would put the rule's arithmetic in a second place,
  and a divergence between the calibrated count and the scored count is exactly
  the bug that would be hardest to see. Cost is `len(ks) × n_months` fits,
  around 730 for a year: seconds.
- **Leave-one-month-out requires ≥ 2 months** and raises otherwise, naming what
  it found, rather than silently reporting an in-sample number.

In `fixed` mode the summary quotes the nearest calibrated point to the k in
force, labelled as reference only. In `target` mode, `choose_k` takes the
**smallest** k whose median held-out count is ≤ target — smallest, because every
increase in k widens the band and therefore hides real orders. If no k in the
grid reaches the target, that is reported as unreachable with the best
achievable count, and the run falls back to `config.DEFAULT_K` rather than
silently returning the boundary value.

### 6.3 The budget is global

`k` is one value applied to every cell, and the target (when used) is the total
across all cells. A badly-behaved group therefore cannot buy itself a wider
band — it just contributes more of the count, which is the signal you want.

---

## 7. Stage 5 — Freezing the artifact

Implemented in `perfthreshold/persist.py`. `bands.json` must be able to answer,
six months later, why one particular order was flagged. Nothing in this list can
be reconstructed after the fact, so none of it is optional:

| Field | Why it must be in the file |
|---|---|
| `version` | a file from a different artifact version is refused, not guessed at |
| `metric`, `metric_units` | scoring refuses a units mismatch |
| `scope`, `market_groups` | cell keys are only reproducible with the grouping in force at fit time |
| `rule.k`, `percentile`, `min_cell_n` | the exact parameters, not the current config's |
| `rule.k_mode`, `k_reason`, `target` | how k was arrived at, in prose, in the artifact |
| `fit_window.start/end` | the dates **observed**, which the leakage guard depends on |
| `source.file`, `source.sha256_16` | which extract, content-addressed, so a swapped file is detectable |
| `bands` | every column of the band table, including both candidates per side |
| `reference_medians` | the drift baseline, per cell |
| `calibration` | the whole curve, so the sensitivity analysis travels with the band |

Two details: non-finite floats are written as `null` and restored as `NaN`
(Python's `json` emits bare `NaN`, which no other reader parses, and
`allow_nan=False` turns a stray one into an error at write time rather than a
corrupt file). And `fit_start`/`fit_end` are the observed dates — a file meant
to hold twelve months but holding nine must say nine.

`fit` also writes `bands.csv`, `calibration.csv`/`.png`, `split_report.csv`,
`cleaning_report.csv`, one distribution plot per cell, and `summary.md`. Only
`bands.json` is needed to score.

---

## 8. Stages 6–8 — Scoring a month

Implemented in `perfthreshold/score.py`.

### 8.1 Three refusals, before any number is produced

Each guards a failure that produces a clean-looking, completely meaningless
result. All three raise; none is downgraded to a warning
(`GuardError`, exit code 2, no traceback).

| Refusal | The failure it prevents |
|---|---|
| **Leakage** — the scored window intersects the fit window | a circular flag rate that looks entirely normal, and leaves no trace |
| **Unit mismatch** — band units ≠ metric units | a band in spreads and one in bps sitting side by side in a summary table looking comparable |
| **Scope mismatch** — band scope ≠ applied scope | every cell key missing, every order `NO_BAND`, a tidy report of zero outliers |

Window overlap is `a_lo ≤ b_hi and b_lo ≤ a_hi`, and is `False` whenever any
bound is unknown — which is why `order_date` is effectively required.

### 8.2 Apply, then rank

`fit.apply` attaches `band_lo`, `band_hi`, `zone` to every row by cell-key
lookup — the single place a band ever meets an order, shared with calibration
and the split report. Flagged rows then get:

```
excess = value − hi   (OUT_HIGH)      or      lo − value   (OUT_LOW)
```

in the metric's own units, and `outliers.csv` is sorted by it, descending. The
order most worth explaining is at the top. Each row carries the diagnostics
needed to explain one — `Pvwap`, `Sprd`, `$Mln`, `#Shares`, `Side`, `%POST`,
`%OPEN`, `%CLOSE`, `Rev30min`, `%Adv`, `Vol`, `PR`, `Dur` — none of which took
part in the band.

Counts reported: `orders`, `flagged`, `out_low`, `out_high`, `no_band`, and
`flagged` broken down by cell. `no_band` is always visible; it is the number of
orders this run could not judge.

### 8.3 Stage 9 — Drift: "the book got harder" vs "execution got worse"

Fifteen flags instead of five has two entirely different explanations calling
for opposite actions, and the band alone cannot distinguish them. So `score.drift`
compares this month's medians of `%Adv`, `Vol`, `PR`, `Dur` and `Sprd` against
the baseline stamped at fit time:

```
baseline[feature] = median over cells of the per-cell fit-time medians
pct_change        = 100 · (month_median − fit_median) / |fit_median|
warn              = |pct_change| > drift_threshold   (default 25%)
```

A warning means the band was not fitted on orders like these — evidence for
"the book changed", not for "execution degraded". It is a diagnostic printed
alongside the queue; it never alters a band or suppresses a flag.

---

## 9. Stage 9b — Whether pooling markets was justified

Implemented in `perfthreshold/split.py`, run at fit time, and **independent of
what `config.MARKET_GROUPS` declares**. Config decides what gets fitted; this
report says whether that decision was defensible. Keeping them separate is the
point — the report cannot be accused of rubber-stamping its own grouping.

For each (benchmark, market) with `n ≥ MIN_CELL_N`:

```
pooled band  = fitted over every in-scope order of that benchmark
own band     = fitted over that market's orders only
excess_flags = flags(market rows, pooled band) − flags(market rows, own band)
verdict      = "split" if excess_flags ≥ 12 AND pooled/own ratio ≥ 2.0
               else "pool"
markets below MIN_CELL_N            → "too few orders", no verdict
```

The question is asked in review workload, not distributional distance. "This
market's distribution differs from pooled by a KS statistic of 0.07" is a fact
nobody can act on; "pooling costs this market 31 extra reviews a year" is the
same fact in the currency this project is judged in, and can be held against the
review budget directly. Both conditions must hold — the ratio alone would let
3 pooled flags against 1 own flag read as evidence.

---

## 10. What the method does not do

Stated plainly, because these are the honest openings for a challenge and it is
better to name them than to have them found. Each one is measured, with the cost
of the alternative, in [`fat-tails.md`](fat-tails.md):

- **It does not test for normality, and does not need to.** The band is defined
  as `mean ± kσ` by policy, not derived from a distributional assumption. What
  normality *would* buy is the ability to predict the alert count from `k`, and
  that is precisely what the leave-one-month-out curve replaces.
- **It provides no floor on review volume.** `MAX` widens; the 99.5th percentile
  bounds band *width*, never the number of orders examined. A month can
  legitimately return zero.
- **`sd` is not robust, on purpose.** The scale that judges the tail is
  inflated by the tail. `mad_sigma` is reported so the size of that effect is
  visible, but no winsorising, trimming or robust re-fit takes place — the
  band's arithmetic stays literally what it says it is.
- **A single global `k`** means a well-behaved cell and a badly-behaved one get
  the same multiplier. That is a deliberate trade (per-cell k would let a bad
  group widen its own band) and it does mean the flag count concentrates in the
  fattest-tailed cells.
- **Two-sided.** Results that look too *good* are flagged, on the reasoning
  that they are usually a data or benchmark error. They land in the same queue,
  distinguished by `zone`, not in a separate one.
- **The metric is consumed as supplied.** `ePvwap/Sprd` arrives pre-divided;
  this project never performs the division and cannot detect an error in it.
  If the spread used by the source is wrong, every number here inherits it.
- **Drift is descriptive.** Nothing re-fits automatically, and no band is
  adjusted in response to a drift warning. Refitting is a human decision.
- **Cells are re-fitted from scratch each run.** There is no incremental update
  and no smoothing across fit windows, so two adjacent fit windows can produce
  visibly different bounds on the same cell.

---

## 11. Objections, and where to check them

| "But…" | Answer | Where to verify |
|---|---|---|
| "k = 4 was picked to get five alerts." | It was not; it is `config.DEFAULT_K`, fixed by policy, and `bands.json` records `k_mode: "fixed"` with the reason in prose. Solving from the count requires `--target`, which stamps `k_mode: "target"` and a warning banner. | `bands.json` → `rule.k_mode`, `rule.k_reason`; `tests/test_cli.py::test_the_production_default_is_a_fixed_k_of_4` |
| "The flag rate is circular — you fitted and scored the same data." | Scoring a window that intersects the fit window raises. The calibration curve is leave-one-month-out. | `score.check_guards`; `tests/test_calibrate.py::test_a_month_is_never_in_its_own_training_set`, `tests/test_score.py::test_scoring_a_month_inside_the_fit_window_is_refused` |
| "The percentile term is decoration." | At k = 4, yes — and the fit says so per cell rather than leaving it to be inferred. | `bands.csv` → `hi_binds`, `lo_binds`; §4.4 |
| "Pooling markets hides the small ones." | Measured, per market, in extra reviews per year, independently of what config declares. | `split_report.csv` → `excess_flags`, `verdict` |
| "A thin cell will look like the worst in the book." | It cannot get a band at all unless a well-supported parent exists, and inheritance is recorded. | `bands.csv` → `fitted`, `fallback_from`, `n`; `tests/test_fit.py::test_pooled_cell_below_min_n_is_left_unfitted_not_inherited` |
| "Rows were quietly dropped to make the numbers work." | `rows_in == rows_kept + Σ dropped`, every reason named, unmapped strategies listed by name and count. | `cleaning_report.csv`; `tests/test_load.py::test_every_input_row_is_accounted_for` |
| "The band was applied to the wrong things." | Units, scope and window are all checked against the frozen file before a single order is scored; fit and score derive cells through the same function. | `score.check_guards`; `groups.resolve` |
| "Which file did these numbers come from?" | The absolute path and a SHA-256 prefix of the extract are in the artifact. | `bands.json` → `source` |
| "Fewer alerts than last month means the threshold moved." | It cannot have: `k`, the percentile and every bound come from the frozen file, and drift reports whether the book itself changed. | `bands.json` → `rule`; `drift.csv` |

---

## 12. Reproducing any of this

```
pip install -r requirements.txt
python -m pytest -q                      # the invariants above, as tests

python -m perfthreshold check --csv year.csv
python -m perfthreshold fit   --csv year.csv --out fits/2025-07_2026-06/
python -m perfthreshold score --csv aug2026.csv \
      --bands fits/2025-07_2026-06/bands.json --out review/2026-08/
```

Every number in a `summary.md` is also in a csv next to it, and every number in
those csvs is derived by one of the functions named in this document. A refusal
exits 2 with a one-line reason and no traceback — it is a result, not a crash.
