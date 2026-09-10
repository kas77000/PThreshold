# PerfThreshold

Which of last month's algo orders went far enough from normal to be worth a
human's attention — and few enough of them that each one gets explained.

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
measured against the same reference price, so they share a distribution shape —
and pooling them buys the sample size a P99.5 estimate needs.

The fit records, per cell per side, **which of the two terms actually bound**.
On a fat-tailed book `sd` is inflated by the very orders the band exists to
catch, so the sigma term can widen past the percentile floor and the rule
quietly degenerates to pure k-sigma. That has to be visible, not inferred.

## k is set in advance; the alert count is an outcome

**`k = 4` is the production default and comes from policy, not from the alert
count.** That ordering is the point. A threshold whose value was selected by how
few alerts it produced reads, to any reviewer, as a threshold tuned to suppress
alerts — so the default can never be able to look like that. `mean ± 4σ` covers
99.9937% of a normal population and stands on its own rationale; how many orders
fall outside is then a finding, not an input.

`fit` still measures what every k costs, **leave-one-month-out**: for each month
in the year, fit on the other eleven and score that one. Twelve out-of-sample
observations, and therefore a range rather than a single flattering in-sample
number. That curve is sensitivity analysis — evidence that the consequences of
the parameter were understood before it was set.

    (no flag)      k = config.DEFAULT_K (4.0), fixed by policy   <- production
    --k 3.5        fix a different k explicitly
    --target 5     DIAGNOSTIC ONLY: solve for the smallest k whose median
                   monthly count is <= 5. The band file records
                   k_mode: "target" and summary.md carries a warning banner.

The budget is the total across every cell, and k is one global value — a
badly-behaved group cannot buy itself a wider band, it just contributes more of
the count.

### What the rule actually does

At k = 4 the `MAX` is inert: σ inflates faster than the 99.5th percentile does,
so the sigma term wins on every tail thickness from Gaussian to Student-t with
3 degrees of freedom. In practice the rule is `mean ± 4σ`, and the percentile is
a floor that never activates. `hi_binds` / `lo_binds` record which term won on
every cell, so this is measured rather than assumed.

Note the direction: `MAX` takes the **wider** bound and therefore flags **fewer**
orders. A rule guaranteeing that the worst 0.5% of each tail is always examined
would need `MIN`, not `MAX`.

### How many alerts will this give?

It depends only on how fat the tails are, and the fit measures that for you.
`bands.csv` carries `sd` and `mad_sigma` (the robust scale, 1.4826·MAD); their
ratio says which regime the book is in:

| `sd / mad_sigma` | Regime | Alerts/month per 3,000 orders |
|---|---|---|
| ~1.00 | normal | ~0.2 |
| 1.10 | mildly fat | ~5 |
| 1.20 | fat | ~11 |
| >=1.30 | very fat | 15+ |

Run `fit` on one real year and read that ratio before predicting anything.

## Running it

**New here? Start with [`docs/getting-started.md`](docs/getting-started.md)** —
the step-by-step first run on real data, what to edit in `config.py`, and the
three numbers to read afterwards.

    pip install -r requirements.txt

    python -m perfthreshold check --csv year.csv

Read section 1 of that output closely and complete `ALGO_BENCHMARK` in
`config.py`. An unmapped strategy is excluded and named — never defaulted —
because a band fitted on the wrong benchmark still fits and never says so.

    python -m perfthreshold fit --csv year.csv --out fits/2025-07_2026-06/
    python -m perfthreshold score --csv aug2026.csv \
        --bands fits/2025-07_2026-06/bands.json --out review/2026-08/

`fit` writes `bands.json` (the frozen artifact — scoring needs nothing else),
`bands.csv`, `calibration.csv` / `.png`, `split_report.csv`,
`cleaning_report.csv`, a distribution plot per cell, and `summary.md`.

`score` writes `scored_orders.csv`, `outliers.csv` (the queue, ranked by how far
outside the band each order sits), `drift.csv`, a distribution plot per cell,
and `summary.md`.

## Scope

    --scope all                 every market pooled; one cell per benchmark
    --scope markets             one cell per (benchmark x market)
    --scope groups              one cell per declared MARKET_GROUPS entry
    --scope group:APAC_TIGHT    that group only
    --benchmark VWAP            and orthogonally, one family

The scope is stamped into the band file and enforced on apply.

## What it refuses

- Scoring a month that overlaps the fit window. The flag rate would be circular
  and would look entirely normal.
- A band whose metric units differ from what is being scored.
- A band fitted at one scope applied at another — every order would score
  `NO_BAND` and the run would report zero outliers.
- Banding a cell below `MIN_CELL_N`. A thin cell's sd is biased low because the
  tail has not been sampled yet, which makes the band too *narrow* and the small
  market look like the worst in the book.
- Loading a row without accounting for it. `rows_kept` plus the drop counts
  always equals `rows_in`; every exclusion has a named reason.

## Should markets be pooled?

`split_report.csv` answers it in review workload rather than in distributional
distance: for each market, how many flags does the pooled band produce that the
market's own band would not? A market being handed 31 extra reviews a year by
pooling has earned its own cell. Declaring a grouping (`MARKET_GROUPS`) and
testing one (this report) stay separate — config decides what gets fitted, the
report says whether that was justified.

## Why fifteen instead of five?

Two entirely different explanations — the book got harder, or execution got
worse — and they call for opposite actions. `drift.csv` compares this month's
`%Adv`, `Vol`, `PR`, `Dur` and `Sprd` medians against the medians stamped when
the band was fitted. A move past the threshold means the band was not fitted on
orders like these. That baseline can only be captured at fit time.

## Documents

- `docs/method.md` — **how it proceeds**: every operation between the raw
  extract and the review queue, the exact arithmetic, what is refused, what the
  method does *not* do, and where to check each claim
- `docs/fat-tails.md` — **why it is hard**: the tail behaviour every choice in
  the method is a response to, demonstrated with measurements rather than claims
  (`docs/fat_tails_experiments.py` regenerates every table)
- `docs/getting-started.md` — first run on real data, start to finish
- `docs/2026-09-09-perfthreshold-design.md` — the design and the reasoning
  behind each choice
- `docs/2026-09-09-perfthreshold-v1.md` — the implementation plan, task by task
