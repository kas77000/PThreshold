import json
import os

import matplotlib
matplotlib.use("Agg")

import pandas as pd
import pytest

from perfthreshold import cli, persist
from tests import synthetic


def _year_and_month(tmp_path, n_per_month=250, seed=51):
    """Thirteen months: the first twelve are the fit file, the last is scored."""
    book = synthetic.make_book(n_per_month=n_per_month, months=13,
                               start="2025-07-01", seed=seed)
    period = pd.to_datetime(book["Date"]).dt.to_period("M").astype(str)
    year = book[period < "2026-07"]
    month = book[period == "2026-07"]
    year_path = tmp_path / "year.csv"
    month_path = tmp_path / "month.csv"
    year.to_csv(year_path, index=False)
    month.to_csv(month_path, index=False)
    return str(year_path), str(month_path)


def test_safe_makes_a_cell_key_a_filename():
    assert cli.safe("VWAP|ALL") == "VWAP_ALL"
    assert cli.safe("TWAP|APAC TIGHT") == "TWAP_APAC_TIGHT"


def test_check_runs_and_names_the_unmapped_strategies(tmp_path, capsys):
    book = synthetic.make_book(n_per_month=20, months=2, unmapped_rows=5)
    path = tmp_path / "book.csv"
    book.to_csv(path, index=False)
    assert cli.main(["check", "--csv", str(path)]) == 0
    out = capsys.readouterr().out
    assert "PART" in out and "UNMAPPED" in out
    assert "VWAP" in out and "TWAP" in out


def test_fit_writes_every_declared_artifact(tmp_path):
    year, _ = _year_and_month(tmp_path)
    out = tmp_path / "fitdir"
    rc = cli.main(["fit", "--csv", year, "--out", str(out),
                   "--k", "4", "--min-cell-n", "100",
                   "--k-grid", "2,8,0.5"])
    assert rc == 0
    for name in ("bands.json", "bands.csv", "calibration.csv",
                 "calibration.png", "split_report.csv",
                 "cleaning_report.csv", "summary.md"):
        assert (out / name).exists(), name
    assert (out / "distribution_VWAP_ALL.png").exists()
    assert (out / "distribution_TWAP_ALL.png").exists()


def test_fit_freezes_the_k_it_used(tmp_path):
    year, _ = _year_and_month(tmp_path)
    out = tmp_path / "fitdir"
    cli.main(["fit", "--csv", year, "--out", str(out), "--k", "3.5",
              "--min-cell-n", "100", "--k-grid", "2,8,1"])
    bf = persist.read(str(out / "bands.json"))
    assert bf.k == 3.5
    assert bf.k_mode == "fixed"
    assert bf.metric_units == "spreads"
    assert bf.scope == "all"


def test_fit_in_target_mode_solves_for_k_and_records_why(tmp_path):
    year, _ = _year_and_month(tmp_path, n_per_month=400, seed=52)
    out = tmp_path / "fitdir"
    assert cli.main(["fit", "--csv", year, "--out", str(out),
                     "--target", "5", "--min-cell-n", "100",
                     "--k-grid", "2,8,0.25"]) == 0
    bf = persist.read(str(out / "bands.json"))
    assert bf.k_mode == "target"
    assert bf.target == 5
    assert "flags/month" in bf.k_reason
    assert bf.k >= 2.0


def test_k_and_target_together_are_rejected(tmp_path):
    year, _ = _year_and_month(tmp_path)
    with pytest.raises(SystemExit):
        cli.main(["fit", "--csv", year, "--out", str(tmp_path / "o"),
                  "--k", "4", "--target", "5"])


def test_score_produces_a_review_queue(tmp_path):
    year, month = _year_and_month(tmp_path)
    fitdir, scoredir = tmp_path / "fitdir", tmp_path / "scoredir"
    cli.main(["fit", "--csv", year, "--out", str(fitdir), "--k", "3",
              "--min-cell-n", "100", "--k-grid", "2,8,1"])
    rc = cli.main(["score", "--csv", month, "--bands",
                   str(fitdir / "bands.json"), "--out", str(scoredir)])
    assert rc == 0
    for name in ("scored_orders.csv", "outliers.csv", "drift.csv",
                 "summary.md"):
        assert (scoredir / name).exists(), name
    outliers = pd.read_csv(scoredir / "outliers.csv")
    scored = pd.read_csv(scoredir / "scored_orders.csv")
    assert len(scored) > 0
    assert set(outliers["zone"]) <= {"OUT_LOW", "OUT_HIGH"}


def test_scoring_a_month_inside_the_fit_window_exits_non_zero(tmp_path, capsys):
    year, _ = _year_and_month(tmp_path)
    fitdir = tmp_path / "fitdir"
    cli.main(["fit", "--csv", year, "--out", str(fitdir), "--k", "4",
              "--min-cell-n", "100", "--k-grid", "2,8,1"])
    # Score the fit file against its own band: pure leakage.
    rc = cli.main(["score", "--csv", year, "--bands",
                   str(fitdir / "bands.json"), "--out", str(tmp_path / "s")])
    assert rc == 2
    assert "fit window" in capsys.readouterr().err


def test_end_to_end_the_calibrated_k_lands_near_the_target(tmp_path):
    # The promise of the whole design: fit a year with a review budget, score
    # the next month, and get roughly that many orders to look at.
    year, month = _year_and_month(tmp_path, n_per_month=500, seed=53)
    fitdir, scoredir = tmp_path / "fitdir", tmp_path / "scoredir"
    assert cli.main(["fit", "--csv", year, "--out", str(fitdir),
                     "--target", "5", "--min-cell-n", "100",
                     "--k-grid", "2,8,0.25"]) == 0
    assert cli.main(["score", "--csv", month, "--bands",
                     str(fitdir / "bands.json"),
                     "--out", str(scoredir)]) == 0
    outliers = pd.read_csv(scoredir / "outliers.csv")
    # Generous upper bound: one month is one draw from the range the
    # calibration reported, not the median. The point is the order of
    # magnitude -- single digits, not forty.
    assert len(outliers) <= 20


def test_score_writes_a_distribution_plot_per_cell(tmp_path):
    year, month = _year_and_month(tmp_path)
    fitdir, scoredir = tmp_path / "fitdir", tmp_path / "scoredir"
    cli.main(["fit", "--csv", year, "--out", str(fitdir), "--k", "3",
              "--min-cell-n", "100", "--k-grid", "2,8,1"])
    cli.main(["score", "--csv", month, "--bands", str(fitdir / "bands.json"),
              "--out", str(scoredir)])
    assert (scoredir / "distribution_VWAP_ALL.png").exists()


def test_a_missing_input_file_exits_non_zero_without_a_traceback(tmp_path,
                                                                capsys):
    rc = cli.main(["check", "--csv", str(tmp_path / "nope.csv")])
    assert rc == 2
    assert "nope.csv" in capsys.readouterr().err
