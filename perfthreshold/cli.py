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


_TARGET_WARNING = (
    "*** DIAGNOSTIC MODE -- NOT A PRODUCTION BASIS ***\n"
    "k was chosen by the alert count it produces. A threshold selected that "
    "way reads, to a reviewer, as a threshold tuned to suppress alerts. Use "
    "this to understand what a parameter costs; set the production k from "
    "policy (the default, config.DEFAULT_K) and let the count be an outcome."
)


def safe(name) -> str:
    """A filesystem-safe token. Cell keys contain '|' and group names spaces."""
    cleaned = "".join(c if (c.isalnum() or c in "-_") else "_"
                      for c in str(name))
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
    identity_cols = {schema.ORDER_ID, schema.ALGO, schema.SYMBOL,
                     schema.ORDER_DATE}
    for canonical in schema.REQUIRED + schema.REFERENCE_FEATURES:
        source = inverse.get(canonical, canonical)
        flag = "REQUIRED" if canonical in schema.REQUIRED else "reference"
        if source not in raw.columns:
            print(f"  {source:<14} {flag:<10} *** ABSENT ***")
            continue
        if canonical in identity_cols:
            missing = int(raw[source].isna().sum())
        else:
            missing = int(pd.to_numeric(raw[source],
                                        errors="coerce").isna().sum())
        print(f"  {source:<14} {flag:<10} missing/non-numeric: {missing:,}")

    print("\n--- 4) LOADING IT FOR REAL ---------------------------------")
    _, rep = load.prepare(raw)
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

    # --target is the ONLY way k gets chosen by the alert count, and it is a
    # diagnostic. With neither flag, k comes from policy (config.DEFAULT_K):
    # a threshold whose value was selected by how few alerts it produced is,
    # to a reviewer, a threshold tuned to suppress alerts. The default must
    # never be able to look like that.
    solving_for_target = args.target is not None
    # The curve is drawn against a reference count either way -- it is the
    # sensitivity analysis, and it is evidence of a considered choice whether
    # or not it was used to make one.
    reference_target = (args.target if solving_for_target
                        else config.DEFAULT_TARGET)

    print(f"\nCalibrating over {len(ks)} values of k, "
          f"{len(calibrate.months(resolved))} months, leave-one-month-out ...")
    curve = calibrate.curve(resolved, ks=ks, percentile=args.percentile,
                            min_cell_n=args.min_cell_n,
                            target=reference_target)
    print(curve.to_string(index=False))

    if not solving_for_target:
        k, k_mode = (float(args.k) if args.k is not None
                     else float(config.DEFAULT_K)), "fixed"
        source = ("supplied on the command line" if args.k is not None
                  else "the configured default (config.DEFAULT_K)")
        near = curve.iloc[int((curve["k"] - k).abs().argsort().iloc[0])]
        reason = (f"k={k:.2f} was fixed in advance -- {source} -- not selected "
                  f"from the alert count. For reference only, the nearest "
                  f"calibrated point (k={near['k']:.2f}) implies a median of "
                  f"{near['median_flags']:.1f} flags/month "
                  f"(range {int(near['min_flags'])}-{int(near['max_flags'])} "
                  f"over {int(near['n_months'])} out-of-sample months).")
    else:
        choice = calibrate.choose_k(curve, target=args.target)
        print("\n" + choice.reason)
        print(_TARGET_WARNING)
        if not choice.reachable:
            # Not an error: the book may genuinely hold more outliers than the
            # budget allows. Fall back to the configured k and say so.
            k, k_mode = float(config.DEFAULT_K), "target"
            reason = choice.reason + f" Fell back to the configured k={k:.2f}."
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
        target=(args.target if solving_for_target else None),
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
                          target=reference_target, chosen_k=k)
        for _, row in result.bands.iterrows():
            cell = row["cell_key"]
            values = resolved.loc[resolved[schema.CELL_KEY] == cell,
                                  schema.METRIC].to_numpy(dtype=float)
            plots.distribution(
                values, row,
                os.path.join(args.out, f"distribution_{safe(cell)}.png"),
                title=str(cell).replace("|", " | "))

    _write_fit_summary(args.out, band_file, curve, splits, clean,
                       reference_target)
    print(f"\nWrote {paths['json']}")
    return 0


def _write_fit_summary(out_dir, band_file, curve, splits, clean, target):
    # A band whose k came from the alert count says so at the top of its own
    # summary, so the caveat travels with the artifact rather than living in
    # someone's memory of how the run was invoked.
    if band_file.k_mode == "target":
        quoted = _TARGET_WARNING.replace("\n", "  \n> ")
        banner = ["", "> **" + quoted + "**", ""]
    else:
        banner = [""]
    lines = [
        "# Fit summary", *banner,
        f"- metric: `{band_file.metric}` ({band_file.metric_units})",
        f"- scope: `{band_file.scope}`",
        f"- fit window: {band_file.fit_start} .. {band_file.fit_end}",
        f"- orders fitted: {band_file.n_orders:,}",
        f"- rule: `hi = MAX(mean + {band_file.k:g}*sd, "
        f"P{band_file.percentile:g})`, mirrored on the low side",
        f"- k mode: {band_file.k_mode}"
        + (f" (solved against a target of {band_file.target}/month)"
           if band_file.k_mode == "target"
           else " -- set in advance, not from the alert count"), "",
        f"- reference line on the calibration chart: {target} flags/month", "",
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

    # The headline is the number of orders someone now has to explain.
    n = result.counts["flagged"]
    rule_line = "=" * 58
    print(f"\n{rule_line}")
    print(f"  {result.month}   ->   {n} ORDER{'' if n == 1 else 'S'} TO REVIEW")
    print(rule_line)
    print(f"  out of {result.counts['orders']:,} scored   "
          f"({result.counts['out_low']} low, {result.counts['out_high']} high, "
          f"{result.counts['no_band']} with no band)")
    for cell, count in sorted(result.counts["by_cell"].items()):
        print(f"    {cell:<26} {count}")
    print(f"  queue   : {os.path.join(args.out, 'outliers.csv')}")
    print(f"  summary : {os.path.join(args.out, 'summary.md')}")
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
                   help="all | markets | groups | group:NAME")
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
