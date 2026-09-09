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
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (np.floating, float)):
        v = float(value)
        return None if not math.isfinite(v) else v
    if isinstance(value, np.bool_):
        return bool(value)
    if value is None or value is pd.NaT:
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
        if out[col].map(
                lambda v: isinstance(v, (int, float, type(None)))).all():
            try:
                out[col] = pd.to_numeric(out[col])
            except (TypeError, ValueError):
                pass
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
