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
