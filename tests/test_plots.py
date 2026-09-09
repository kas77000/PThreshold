import os

import matplotlib
matplotlib.use("Agg")            # no display in CI; must precede pyplot import

import numpy as np
import pandas as pd
import pytest

from perfthreshold import calibrate, fit, groups, load, plots, rule
from tests import synthetic


def _band_row(k=4.0):
    raw = synthetic.make_book(n_per_month=300, months=12, seed=41)
    df, _ = load.prepare(raw)
    resolved, _ = groups.resolve(df, scope="all")
    res = fit.fit_cells(resolved, k=k, min_cell_n=100)
    row = res.bands.set_index("cell_key").loc["VWAP|ALL"]
    values = resolved.loc[resolved["cell_key"] == "VWAP|ALL",
                          "perf_in_spreads"].to_numpy(dtype=float)
    return values, row


def test_clip_counts_both_overflows_and_clamps_the_values():
    x = np.array([-100.0, -1.0, 0.0, 1.0, 100.0])
    clipped, below, above = plots.clip_with_overflow(x, -2.0, 2.0)
    assert below == 1 and above == 1
    assert clipped.min() == -2.0 and clipped.max() == 2.0
    assert len(clipped) == len(x)


def test_clip_ignores_non_finite_values():
    x = np.array([np.nan, np.inf, 0.0])
    clipped, below, above = plots.clip_with_overflow(x, -1.0, 1.0)
    assert len(clipped) == 1 and below == 0 and above == 0


def test_clip_with_no_overflow_reports_zeroes():
    x = np.array([-0.5, 0.0, 0.5])
    _, below, above = plots.clip_with_overflow(x, -1.0, 1.0)
    assert below == 0 and above == 0


def test_distribution_writes_a_non_empty_png(tmp_path):
    values, row = _band_row()
    out = plots.distribution(values, row, str(tmp_path / "d.png"),
                             title="VWAP | ALL")
    assert os.path.exists(out)
    assert os.path.getsize(out) > 5_000


def test_distribution_creates_a_missing_directory(tmp_path):
    values, row = _band_row()
    out = plots.distribution(values, row,
                             str(tmp_path / "deep" / "nested" / "d.png"),
                             title="VWAP | ALL")
    assert os.path.exists(out)


def test_distribution_handles_an_unfitted_band_without_crashing(tmp_path):
    values, row = _band_row()
    row = row.copy()
    row["lo"] = np.nan
    row["hi"] = np.nan
    out = plots.distribution(values, row, str(tmp_path / "d.png"),
                             title="unfitted")
    assert os.path.exists(out)


def test_distribution_accepts_a_scored_month_overlay(tmp_path):
    values, row = _band_row()
    month = values[:200] * 1.2
    out = plots.distribution(values, row, str(tmp_path / "d.png"),
                             title="VWAP | ALL", month_values=month)
    assert os.path.getsize(out) > 5_000


def test_distribution_on_an_empty_series_still_writes_a_file(tmp_path):
    _, row = _band_row()
    out = plots.distribution(np.array([]), row, str(tmp_path / "d.png"),
                             title="empty")
    assert os.path.exists(out)


def test_calibration_writes_a_non_empty_png(tmp_path):
    curve = pd.DataFrame({
        "k": [3.0, 4.0, 5.0], "n_months": [12] * 3,
        "median_flags": [14.0, 5.0, 2.0], "mean_flags": [15.0, 6.0, 2.5],
        "min_flags": [6, 1, 0], "max_flags": [31, 12, 5],
        "months_over_target": [12, 4, 0]})
    out = plots.calibration(curve, str(tmp_path / "c.png"), target=5,
                            chosen_k=4.0)
    assert os.path.exists(out)
    assert os.path.getsize(out) > 5_000


def test_calibration_without_a_chosen_k_still_renders(tmp_path):
    curve = pd.DataFrame({
        "k": [3.0, 4.0], "n_months": [12, 12],
        "median_flags": [40.0, 30.0], "mean_flags": [41.0, 31.0],
        "min_flags": [20, 15], "max_flags": [60, 45],
        "months_over_target": [12, 12]})
    out = plots.calibration(curve, str(tmp_path / "c.png"), target=5,
                            chosen_k=None)
    assert os.path.exists(out)


def test_calibration_on_an_empty_curve_writes_a_file_saying_so(tmp_path):
    curve = pd.DataFrame(columns=calibrate.CURVE_COLS)
    out = plots.calibration(curve, str(tmp_path / "c.png"), target=5)
    assert os.path.exists(out)
