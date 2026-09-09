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

## k is chosen, not assumed

`k = 4` is a statement about a Gaussian; this book is not one. So `fit` measures
what each k costs, **leave-one-month-out**: for each month in the year, fit on
the other eleven and score that one. Twelve out-of-sample observations, and
therefore a range rather than a single flattering in-sample number.

    --target 5     solve for the smallest k whose median monthly count is <= 5
    --k 4          fix k; the curve is still printed, so you see what it costs

The budget is the total across every cell, and k is one global value — a
badly-behaved group cannot buy itself a wider band, it just contributes more of
the five.

## Running it

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

## Design documents

- `docs/superpowers/specs/2026-09-09-perfthreshold-design.md` — the design and
  the reasoning behind each choice
- `docs/superpowers/plans/2026-09-09-perfthreshold-v1.md` — the implementation
  plan, task by task
