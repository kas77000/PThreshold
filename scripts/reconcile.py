"""Why does another calculation get a different sigma on the same file?

Same rows, same column, same formula must give the same standard deviation.
So when two calculations disagree, exactly one of those three differs, and
this script says which -- it computes sigma at every stage between "the raw
file, untouched" and "one fitted cell", so the step where the number moves is
visible rather than argued about.

    python scripts/reconcile.py your_year_file.csv

Read the output top to bottom. The line where sigma changes is your answer.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

# Run from anywhere: put the project root on the path rather than requiring
# an install or a particular working directory.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from perfthreshold import config, groups, load, schema  # noqa: E402


def _sd(series) -> tuple[int, float, float]:
    a = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    a = a[np.isfinite(a)]
    if a.size < 2:
        return int(a.size), float("nan"), float("nan")
    return int(a.size), float(np.mean(a)), float(np.std(a, ddof=1))


def _line(label: str, n: int, mean: float, sd: float, k: float = 4.0) -> None:
    if np.isfinite(sd):
        print(f"{label:<46}{n:>9,}{mean:>10.3f}{sd:>10.3f}"
              f"{mean - k * sd:>11.2f}{mean + k * sd:>10.2f}")
    else:
        print(f"{label:<46}{n:>9,}{'':>10}{'':>10}{'':>11}{'':>10}")


def main(path: str) -> int:
    raw = load.read_any(path)
    print(f"file: {path}")
    print(f"raw rows: {len(raw):,}\n")
    print(f"{'stage':<46}{'n':>9}{'mean':>10}{'sigma':>10}"
          f"{'lo(4sd)':>11}{'hi(4sd)':>10}")
    print("-" * 96)

    # ---- 1) the raw file, nothing removed, nothing grouped --------------
    # This is what a one-shot calculation over the whole extract produces.
    metric_col = config.METRIC_COLUMN
    if metric_col in raw.columns:
        _line(f"raw file, all rows, '{metric_col}'", *_sd(raw[metric_col]))
    else:
        print(f"!! '{metric_col}' is not in the file")

    # ---- 2) the same thing, but dividing Pvwap by Sprd ourselves --------
    # If these two lines differ, the two calculations are not banding the
    # same quantity, and everything downstream is incomparable.
    if {"Pvwap", "Sprd"} <= set(raw.columns):
        pv = pd.to_numeric(raw["Pvwap"], errors="coerce")
        sp = pd.to_numeric(raw["Sprd"], errors="coerce").replace(0, np.nan)
        _line("raw file, all rows, Pvwap / Sprd (computed)", *_sd(pv / sp))
        if metric_col in raw.columns:
            supplied = pd.to_numeric(raw[metric_col], errors="coerce")
            ratio = (supplied / (pv / sp)).replace([np.inf, -np.inf], np.nan)
            r = ratio.dropna()
            if len(r):
                print(f"\n    supplied '{metric_col}' divided by computed "
                      f"Pvwap/Sprd:")
                print(f"      median {r.median():.4f}   "
                      f"10th pct {r.quantile(0.10):.4f}   "
                      f"90th pct {r.quantile(0.90):.4f}")
                if abs(r.median() - 1.0) > 0.01:
                    print("      -> THE TWO COLUMNS ARE NOT THE SAME QUANTITY.")
                    print("         A near-constant ratio means a different "
                          "spread convention")
                    print("         (half vs full spread, or different units).")
                else:
                    print("      -> the two agree; the metric is not the "
                          "difference.")
            print()

    # ---- 3) after this project's row filtering --------------------------
    df, clean = load.prepare(raw)
    _line("after filtering (unmapped algos etc.), pooled",
          *_sd(df[schema.METRIC]))
    dropped = {r: n for r, n in clean.dropped.items() if n}
    if dropped:
        print(f"    rows removed: {sum(dropped.values()):,}")
        for reason, n in dropped.items():
            print(f"      {reason}: {n:,}")
        if clean.unmapped_strategies:
            print(f"      strategies excluded: "
                  f"{clean.unmapped_strategies}")
    print()

    # ---- 4) split into cells -------------------------------------------
    for scope in ("all", "markets"):
        resolved, _ = groups.resolve(df, scope=scope)
        for cell, g in resolved.groupby(schema.CELL_KEY, observed=True):
            _line(f"  scope={scope}: {cell}", *_sd(g[schema.METRIC]))

    print("\nThe stage where sigma moves is the reason the two numbers differ.")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__)
        raise SystemExit(2)
    raise SystemExit(main(sys.argv[1]))
