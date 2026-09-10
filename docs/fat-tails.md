# Fat tails — the difficulty this project is actually made of

Every hard decision in PerfThreshold traces back to one property of the data:
the metric's tails are heavier than a normal distribution's. That single fact is
what makes `k = 4` unpredictable, makes `sd` a poor ruler for the thing it is
measuring, makes a small cell dangerous, makes one month's alert count a bad
forecast of the next, and makes pooling markets both necessary and suspect.

This document demonstrates each of those difficulties with numbers rather than
asserting them. Every table below was produced by
[`fat_tails_experiments.py`](fat_tails_experiments.py) calling the project's own
`perfthreshold.rule` — the same function that fits production bands:

```
PYTHONPATH=. python docs/fat_tails_experiments.py
```

Samples are Student-t with the stated degrees of freedom, which is the standard
way to dial tail weight while keeping the centre and shape otherwise ordinary.
Lower df = fatter tails. The project's own synthetic book uses t with 3.0–3.5 df
(`tests/synthetic.py`) because that is the region real execution data sits in.

Related: [`method.md`](method.md) is what the program does;
[`getting-started.md`](getting-started.md) is how to run it. This is *why it is
built the way it is*.

---

## 0. "Fat-tailed" as a measurable, not an adjective

The band file reports two scale estimates for every cell:

| | Definition | Property |
|---|---|---|
| `sd` | `std(x, ddof=1)` | the classical scale — uses every observation, including the tail |
| `mad_sigma` | `1.4826 · median(|x − median|)` | the robust scale — the tail cannot move it |

The constant 1.4826 is `1 / Φ⁻¹(0.75)`, chosen so that **under normality the two
agree**. That is the whole point of reporting both: any gap between them is the
non-normality, expressed in the band's own units.

```
ratio = sd / mad_sigma       1.00 = normal      higher = fatter
```

This ratio is the first number to read in `bands.csv`, and every difficulty
below is a consequence of it exceeding 1.

---

## 1. Difficulty: `k` does not determine the alert count

`mean ± 4σ` covers 99.9937% of a *normal* population — about 2 orders a year in
a 3,000-orders-a-month book. That number is a property of the Gaussian, not of
the rule. Hold `k = 4` fixed and vary only tail weight (n = 400,000 draws each):

| Sample | `sd / mad_sigma` | % outside `mean ± 4σ` | flags per 3,000 orders | `hi` |
|---|---|---|---|---|
| Gaussian | 1.001 | 0.0063% | **0.19** | 3.99 |
| t, 8 df | 1.108 | 0.1703% | **5.11** | 4.62 |
| t, 5 df | 1.199 | 0.3655% | **10.96** | 5.18 |
| t, 4 df | 1.291 | 0.5010% | **15.03** | 5.65 |
| t, 3 df | 1.533 | 0.6190% | **18.57** | 6.99 |
| t, 2.5 df | 2.061 | 0.4810% | **14.43** | 9.60 |

The same rule, the same k, the same sample size: **0.19 alerts a month or 18.6**,
a factor of ninety-eight, decided entirely by something the number 4 says
nothing about. This is the reason the fit measures the count out of sample
instead of predicting it from k, and the reason `k` is fixed by policy rather
than argued from a coverage figure.

**The last row is not a typo, and it is also not a trend.** At t₂.₅ the count
comes out *below* t₃'s in this sample, because `sd` grew faster than the tail
did and the band outran the orders it was meant to catch. But repeat the same
measurement on eight independent samples of 400,000 and the effect turns out to
be sampling noise on top of a plateau:

| Sample | min | median | max (8 samples, per 3,000) |
|---|---|---|---|
| t, 4 df | 14.16 | 14.64 | 14.99 |
| t, 3 df | 18.05 | 19.04 | 19.62 |
| t, 2.5 df | 16.44 | **19.06** | 21.17 |

So the honest reading is: **the count stops rising once tails pass roughly t₃,
and its own estimate becomes unstable** — a ±13% swing at t₂.₅ on 400,000
observations, where t₄ swings ±3%. Two consequences for the regime table in
[`getting-started.md`](getting-started.md#step-4--the-three-numbers-to-read):
its "≥ 1.30 → 15+" is sound in the 1.3–1.6 region and saturates above that, and
a `sd / mad_sigma` of 2.0 measured on one year is itself a noisy number. Read
`median_flags` from your own `calibration.csv`; do not extrapolate from the
ratio.

---

## 2. Difficulty: the ruler is made of the thing being measured

`sd` is computed from every order, including the ones the band exists to catch.
So the outliers set the width of the band that judges them — the estimator is
self-referential in exactly the direction that hurts.

**Trim the worst 0.5% of each tail, refit, and re-score the untrimmed data**
(n = 20,000):

| Sample | `sd` (full) | `hi` (full) | flags | `sd` (trimmed fit) | `hi` | same rows now flagged |
|---|---|---|---|---|---|---|
| Gaussian | 0.999 | 4.00 | 1 | 0.977 | 3.91 | 2 |
| t, 3 df | 1.686 | 6.73 | **119** | 1.456 | 5.82 | **190** |

Removing 0.5% of the observations from the *fit* changes the verdict on the
whole dataset by 60%. On a Gaussian book the same operation moves the count by
one order. The fat tail is not merely present in the data — it is inside the
threshold.

### The sharper version: one order can hide four others

Add a **single** extreme order to a 3,000-order cell and refit:

| Sample | added at | `sd` | `hi` | flags |
|---|---|---|---|---|
| Gaussian | — | 1.005 | 4.04 | 0 |
| | +10 | 1.021 | 4.11 | 1 |
| | +20 | 1.069 | 4.30 | 1 |
| | +40 | 1.242 | 5.00 | 1 |
| t, 3 df | — | 1.758 | 7.02 | 22 |
| | +10 | 1.767 | 7.06 | 23 |
| | +20 | 1.795 | 7.18 | **22** |
| | +40 | 1.903 | 7.62 | **18** |

Read the last row carefully. One catastrophic order was added, and the total
number of flagged orders **fell by four**: it inflated `sd` enough to pull the
band out past four orders that had been outside it. This is the classical
*masking* effect, and it is the single most important thing to understand about
a non-robust threshold. The worst month in a book is precisely the month whose
band is widest, and a genuinely bad order arrives with the ability to excuse its
neighbours.

The program does not fix this — it reports the ingredients (`sd`, `mad_sigma`,
both candidates per side) so the effect is visible, and §7 explains why the fix
was not adopted.

---

## 3. Difficulty: the bound is an estimate, and it converges slowly

Refit the same cell 400 times on fresh samples of n = 3,000 and look at the
spread of `hi`:

| Sample | median `hi` | 5th pct | 95th pct | p95 / p5 |
|---|---|---|---|---|
| Gaussian | 4.00 | 3.92 | 4.08 | 1.04× |
| t, 8 df | 4.61 | 4.49 | 4.77 | 1.06× |
| t, 5 df | 5.15 | 4.95 | 5.39 | 1.09× |
| t, 4 df | 5.61 | 5.36 | 5.97 | 1.11× |
| t, 3 df | 6.72 | 6.24 | 7.92 | **1.27×** |
| t, 2.5 df | 8.02 | 7.25 | 11.23 | **1.55×** |

Two statistically identical fat-tailed years can hand you bands 27% apart — and
the distribution of `hi` is itself right-skewed, which is why the p95 is much
further from the median than the p5 is.

**And more data helps less than you would expect.** Same experiment, varying n:

| Sample | n = 500 | n = 2,000 | n = 8,000 | n = 32,000 |
|---|---|---|---|---|
| Gaussian | 1.113× | 1.057× | 1.026× | 1.014× |
| t, 3 df | 1.487× | 1.268× | 1.174× | 1.102× |

The Gaussian error roughly halves each time n quadruples — the familiar `1/√n`.
The t₃ error does not: at 32,000 orders it is still wider than the Gaussian's at
500. The reason is that `sd`'s own sampling error depends on the fourth moment,
and for t with 4 or fewer degrees of freedom the fourth moment is **infinite** —
the estimator is consistent but its error shrinks at a slower, non-standard rate.
Buying more history is a weaker remedy here than intuition suggests.

Consequence for operations: a refit is expected to move the bounds. `bands.json`
freezes the exact numbers, `source.sha256_16` records which file produced them,
and nothing is smoothed across fit windows — so a band that moved must be
explained by the refit, not discovered later.

---

## 4. Difficulty: the percentile is not a rescue

The obvious answer to "sd is contaminated by the tail" is "use a percentile
instead". The rule does carry `P99.5` as the second candidate per side — but a
percentile deep in the tail is exactly the quantity a fat tail makes hard to
estimate. Estimate `P99.5` 400 times and compare with the truth:

| Sample | n | truth | 5th–95th pct of the estimate | width, as % of truth |
|---|---|---|---|---|
| Gaussian | 250 | 2.57 | 2.04 – 2.94 | 34.9% |
| Gaussian | 1,000 | 2.57 | 2.33 – 2.79 | 17.7% |
| Gaussian | 2,000 | 2.57 | 2.38 – 2.73 | **13.6%** |
| Gaussian | 12,000 | 2.57 | 2.50 – 2.65 | 5.6% |
| t, 3 df | 250 | 5.86 | 3.48 – 8.11 | 79.1% |
| t, 3 df | 1,000 | 5.86 | 4.56 – 7.27 | 46.4% |
| t, 3 df | 2,000 | 5.86 | 4.72 – 7.00 | **38.9%** |
| t, 3 df | 12,000 | 5.86 | 5.39 – 6.32 | 15.8% |

At n = 250 the 99.5th percentile is interpolated between the first and second
worst order in the sample; it is barely an estimate at all. Two things follow:

- **`MIN_CELL_N = 2000` is a floor, not a cure.** It buys a percentile estimate
  with ±39% of uncertainty on a t₃ book, which is enough to be worth having and
  nowhere near enough to be precise. It is set where the *sd* estimate becomes
  tolerable and the percentile becomes meaningful at all — and the fallback
  contract in [`method.md`](method.md#5-stage-4b--thin-cells-the-fallback-contract)
  exists because below it the band comes out too **narrow**, making the smallest
  cell look like the worst in the book.
- **Neither candidate is trustworthy in a thin cell**, which is why the answer
  there is to inherit a well-supported parent band or produce none at all,
  rather than to switch to the "more robust" term.

---

## 5. Difficulty: the alert count is heavy-tailed too

The flag count is a function of a fat-tailed variable, so it inherits the
property. Twelve months of 250 orders, leave-one-month-out, repeated 200 times:

| Sample | typical median month | typical worst month | worst seen |
|---|---|---|---|
| Gaussian | 0.0 | 0.0 | 1 |
| t, 5 df | 1.0 | 3.0 | 6 |
| t, 3 df | 1.5 | 4.0 | 7 |
| t, 2.5 df | 2.0 | 4.0 | 8 |

The worst month runs 2–3× the median month, and on a t₃ book the single worst
month holds a median of **14% of the year's flags** against the 8% an even split
would give. Review capacity has to be planned against the worst month; the
median is a budgeting number, not a staffing one. This is why
`calibration.csv` reports `min_flags` and `max_flags` alongside `median_flags`,
and why `summary.md` quotes the range rather than a single rate.

### The calibration curve is itself an estimate

Five independent "years" from the *same* generator (1,000 orders/month, 12
months), each calibrated leave-one-month-out:

| Year | k = 3.5 | k = 4.0 | k = 4.5 |
|---|---|---|---|
| 1 | median 3.0 (0–11) | **median 2.0** (0–8) | 1.0 (0–6) |
| 2 | median 9.0 (4–10) | **median 5.5** (1–9) | 4.0 (0–7) |
| 3 | median 6.0 (2–8) | **median 3.0** (0–7) | 1.0 (0–3) |
| 4 | median 7.5 (4–13) | **median 6.0** (2–8) | 3.5 (1–5) |
| 5 | median 7.0 (3–14) | **median 6.0** (3–12) | 4.5 (0–9) |

Statistically identical books, and the out-of-sample median at k = 4 ranges from
2.0 to 6.0 — a factor of three. So:

- "The calibration says 5 alerts a month" is a measurement of *one year*, with
  real uncertainty around it, not a forecast. Quote it with its range and its
  window.
- Worth noting what this does to `--target`: a k solved against a noisy median is
  a k fitted to one year's luck. That is a second, independent reason the
  production default is a policy k and `--target` is labelled diagnostic — the
  evidentiary argument in
  [`method.md §6.1`](method.md#61-k-is-fixed-before-the-data-is-consulted) is the
  first.
- The ordering `k ↑ ⇒ flags ↓` is stable and monotone within a year (asserted by
  `tests/test_calibrate.py::test_curve_is_monotone_non_increasing_in_k`); it is
  the *level* that moves between years, not the direction.

---

## 6. Difficulty: a mixture is indistinguishable from a fat tail

This is the trap that connects fat tails to the pooling decision, and it is the
one most likely to be missed.

Take two **perfectly Gaussian** populations — a tight book at σ = 1.0 (85% of
orders) and a wide one at σ = 3.0 (15%) — and pool them:

| Cell | `sd` | `mad_sigma` | ratio | own `hi` | flags/3,000 under its own band | flags/3,000 under the **pooled** band |
|---|---|---|---|---|---|---|
| tight only | 1.004 | 0.999 | **1.005** | 4.01 | 0.29 | **0.00** |
| wide only | 3.016 | 3.011 | **1.002** | 12.03 | 0.00 | **148.00** |
| pooled | 1.491 | 1.125 | **1.325** | 5.95 | 22.20 | 22.20 |

Neither component has a fat tail at all — both ratios are 1.00. The pool's ratio
is 1.325, which is squarely in this project's "very fat" band and
indistinguishable from a genuine t₄. **Some, or all, of an observed fat tail can
be a mixture of well-behaved regimes rather than a tail.**

And the cost lands entirely on one side: under the pooled band the wide
population gets 148 flags per 3,000 orders while the tight one gets **none**.
Pooling did not spread the burden — it moved the whole queue into one market and
gave the other an amnesty.

This is precisely the question `split_report.csv` answers, and why it asks it in
**extra reviews per market** rather than in a distributional distance:

```
excess_flags = flags(market rows, pooled band) − flags(market rows, own band)
verdict = "split" if excess_flags ≥ 12 and pooled/own ratio ≥ 2.0
```

Spread normalisation is what makes pooling defensible in the first place — it puts
a wide small cap and a tight large cap on one scale before anything is fitted.
What survives normalisation is structural: tick-size regimes, closing-auction
dominance, thin books where the spread itself is noisy. The split report measures
whether any of that still bites, and `sd / mad_sigma` alone cannot tell you,
because a mixture and a tail look the same in that one number.

---

## 7. Difficulty: fat tails quietly erase the per-side design

The rule is written per side so that a skewed book gets a band following its real
skew rather than one forced equidistant from the mean. On a fat-tailed book that
design mostly does not get to act. From the project's own synthetic generator
(which carries a deliberate left skew), pooled, k = 4:

| Cell | n | mean | sd | mad_sigma | `sigma_lo` | `p_lo` | `lo` | `sigma_hi` | `p_hi` | `hi` |
|---|---|---|---|---|---|---|---|---|---|---|
| TWAP\|ALL | 7,152 | −0.19 | 1.41 | 0.70 | −5.81 | −3.61 | **−5.81** | 5.43 | 3.21 | **5.43** |
| VWAP\|ALL | 7,248 | −0.05 | 0.80 | 0.53 | −3.24 | −2.42 | **−3.24** | 3.14 | 2.06 | **3.14** |

The skew is real and visible: `p_lo` reaches −3.61 while `p_hi` only reaches
3.21. But sigma wins on both sides, so both bounds land exactly 4·sd from the
mean, and the asymmetry the percentile candidates measured is **discarded**. The
band is symmetric about the mean; only the mean's offset makes `|lo| ≠ |hi|`.

So the honest statement is: the per-side design is dormant at k = 4 and would
activate below the crossover (k ≈ 2.6–3.2 depending on tail weight, see
[`method.md §4.4`](method.md#4-stage-4--the-band-arithmetic)). It costs nothing to
keep, both candidates are recorded per side so the skew stays visible in the
artifact, and if the policy k ever moves down the mechanism is already there.

---

## 8. What the program does about all this

| Difficulty | Mechanism | Where it is visible |
|---|---|---|
| k does not determine the count (§1) | count measured leave-one-month-out over a k grid, never predicted | `calibration.csv`, `calibration.png` |
| tail weight must be quantified (§0, §1) | both scales reported per cell | `bands.csv` → `sd`, `mad_sigma` |
| the ruler is contaminated (§2) | not fixed; made measurable and stated plainly | `sd / mad_sigma`; §2 of this doc |
| bounds are unstable estimates (§3) | bounds frozen with the source hash; no smoothing, no silent refit | `bands.json` → `rule`, `source`, `fit_window` |
| deep percentiles need n (§4) | `MIN_CELL_N`; thin cells inherit a pooled parent or get no band | `bands.csv` → `fitted`, `fallback_from`, `n` |
| the count is heavy-tailed (§5) | range reported, not just the median | `calibration.csv` → `min_flags`, `max_flags`, `months_over_target` |
| calibration is itself noisy (§5) | k set by policy; `--target` labelled diagnostic in the artifact | `bands.json` → `k_mode`, `k_reason`; `summary.md` banner |
| a mixture mimics a tail (§6) | pooling tested in extra reviews per market, independently of config | `split_report.csv` → `excess_flags`, `verdict` |
| which term actually bound (§7) | recorded per cell, per side | `bands.csv` → `hi_binds`, `lo_binds` |
| the tail may have moved, not the desk (§3, §5) | fit-time reference medians vs the scored month | `drift.csv` |

---

## 9. What was deliberately not done, and what it would cost

Each of these is a real answer to a real difficulty above. Each was left out for
a stated reason, and each is a legitimate thing to argue for.

**Robust scale in the band (use `mad_sigma` instead of `sd`).** Fixes the
contamination in §2 and the instability in §3 outright. Not adopted because the
rule then stops being `mean ± 4σ`: the defensible sentence "four standard
deviations" becomes "four robust-scaled median absolute deviations", the
coverage rationale no longer applies, and the review load multiplies at the same
k — the band narrows to a robust scale while the tail stays where it is:

| Sample | `mean ± 4·sd` | `median ± 4·mad_sigma` | multiplier |
|---|---|---|---|
| Gaussian | 0.20 | 0.20 | 1.0× |
| t, 5 df | 10.70 | 22.21 | 2.1× |
| t, 4 df | 14.08 | 34.43 | 2.4× |
| t, 3 df | 18.80 | 59.77 | **3.2×** |
| t, 2.5 df | 18.82 | 83.47 | 4.4× |

(flags per 3,000 orders). `mad_sigma` is therefore reported, never used for
bounds. Switching would be a policy change with a stated cost, not a code
change.

**Winsorising or trimming before the fit.** Same benefit, worse honesty: §2's
experiment shows the trimmed fit flags 60% more orders, so the choice of trim
fraction becomes a second, undeclared threshold parameter chosen by its effect
on the alert count — the exact thing the fixed-k discipline exists to avoid.

**An extreme-value tail model (peaks-over-threshold / generalised Pareto).** The
statistically correct tool for "how far out is the 1-in-2,000 order", and it
would estimate the tail rather than let `sd` proxy for it. Not adopted for v1
because it introduces two fitted parameters plus a threshold choice per cell,
needs its own goodness-of-fit diagnostics, and produces a bound no reviewer can
recompute by hand from two columns. It is the natural v2 if precision in the
tail becomes the binding requirement.

**Per-cell `k`.** Would equalise the alert count across cells, which §6 shows is
currently very unequal. Not adopted because a cell's k would then be set by its
own alert count — a badly-behaved group buying itself a wider band — which
inverts the signal the report exists to produce.

**More history.** §3 shows why this is a weaker fix than it appears: the error
shrinks at a slower-than-`1/√n` rate under infinite kurtosis, and a longer window
makes the drift problem worse by mixing more regimes into one band.

---

## 10. Reading your own book

In order, on one real year:

1. **`sd / mad_sigma` in `bands.csv`** — which regime each cell is in. Near 1.00
   and none of this matters much; 1.2–1.6 and every section above applies; above
   1.7, stop extrapolating from the ratio (§1) and read the measured count.
2. **`median_flags` with `min_flags`–`max_flags` at your k, in
   `calibration.csv`** — the actual review load, out of sample, with its range
   (§5). Staff the max, budget the median.
3. **`hi_binds` / `lo_binds`** — expect `sigma` everywhere at k = 4, which means
   you are defending a 4σ rule and the percentile floor is inert (§7).
4. **`split_report.csv`** — whether the fat tail you are looking at is one
   population or several pooled (§6). A market with a large `excess_flags` is
   carrying the pool's queue.
5. **`drift.csv` each month** — whether the book still looks like the one the
   band was fitted on. A fat tail and a changed book produce the same surprise in
   the count and call for opposite responses.

If (1) says near-normal and (4) says every market pools cleanly, this document
is mostly irrelevant to your data — and that itself is a finding worth writing
down, because it is the condition under which the `k = 4` coverage argument
holds on its own terms.
