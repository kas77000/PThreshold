import numpy as np
import pandas as pd
import pytest

from perfthreshold import calibrate, fit, groups, load, schema
from tests import synthetic


def _resolved(n_per_month=400, months=12, seed=11):
    raw = synthetic.make_book(n_per_month=n_per_month, months=months, seed=seed)
    df, _ = load.prepare(raw)
    out, _ = groups.resolve(df, scope="all")
    return out


def _two_month_frame():
    """Month A is all zeros; month B is all tens. Hand-computable on purpose."""
    rows = []
    for i in range(50):
        rows.append({schema.ORDER_ID: f"A{i}", schema.BENCHMARK: "VWAP",
                     schema.MARKET: "HK", schema.MARKET_GROUP: "ALL",
                     schema.CELL_KEY: "VWAP|ALL", schema.METRIC: 0.0,
                     schema.ORDER_DATE: pd.Timestamp("2025-07-05")})
    for i in range(50):
        rows.append({schema.ORDER_ID: f"B{i}", schema.BENCHMARK: "VWAP",
                     schema.MARKET: "HK", schema.MARKET_GROUP: "ALL",
                     schema.CELL_KEY: "VWAP|ALL", schema.METRIC: 10.0,
                     schema.ORDER_DATE: pd.Timestamp("2025-08-05")})
    return pd.DataFrame(rows)


def test_months_lists_every_period_once_in_order():
    df = _resolved(n_per_month=5, months=12, seed=1)
    assert calibrate.months(df)[0] == "2025-07"
    assert calibrate.months(df)[-1] == "2026-06"
    assert len(calibrate.months(df)) == 12


def test_k_grid_includes_both_ends_and_is_not_floating_point_noise():
    g = calibrate.k_grid(2.0, 8.0, 0.1)
    assert g[0] == 2.0 and g[-1] == 8.0
    assert 4.0 in g
    assert all(round(x, 3) == x for x in g)


def test_a_month_is_never_in_its_own_training_set():
    # Each month is trained on the OTHER one, which has a single repeated
    # value: the band collapses to a point (sd = 0) somewhere the held-out
    # month never is, so all 50 of its orders flag. Both months, symmetrically.
    df = _two_month_frame()
    counts = calibrate.lomo_counts(df, k=4.0, min_cell_n=1)
    assert counts["2025-08"] == 50
    assert counts["2025-07"] == 50

    # The contrast that makes the above mean something: fit on BOTH months and
    # the band is mean 5 +/- 4*5.03, which swallows every order. Leakage would
    # therefore show up as zero flags, so 50 can only come from having actually
    # left the month out.
    leaky = fit.fit_cells(df, k=4.0, min_cell_n=1)
    assert fit.flag_count(df, leaky.bands) == 0


def test_lomo_returns_one_count_per_month():
    df = _resolved(n_per_month=200, months=12, seed=2)
    counts = calibrate.lomo_counts(df, k=4.0, min_cell_n=100)
    assert set(counts) == set(calibrate.months(df))
    assert all(isinstance(v, int) for v in counts.values())


def test_lomo_needs_at_least_two_months():
    df = _resolved(n_per_month=50, months=1)
    with pytest.raises(ValueError, match="two months"):
        calibrate.lomo_counts(df, k=4.0, min_cell_n=10)


def test_curve_is_monotone_non_increasing_in_k():
    df = _resolved(n_per_month=300, months=12, seed=3)
    c = calibrate.curve(df, ks=[2.0, 3.0, 4.0, 5.0, 6.0], min_cell_n=100,
                        target=5)
    med = list(c["median_flags"])
    assert med == sorted(med, reverse=True)


def test_curve_reports_the_range_not_just_the_median():
    df = _resolved(n_per_month=300, months=12, seed=4)
    c = calibrate.curve(df, ks=[3.0, 4.0], min_cell_n=100, target=5)
    assert list(c.columns) == ["k", "n_months", "median_flags", "mean_flags",
                               "min_flags", "max_flags", "months_over_target"]
    row = c.iloc[0]
    assert row["min_flags"] <= row["median_flags"] <= row["max_flags"]
    assert row["n_months"] == 12


def test_choose_k_takes_the_smallest_k_meeting_the_target():
    c = pd.DataFrame({
        "k": [3.0, 4.0, 5.0, 6.0],
        "n_months": [12] * 4,
        "median_flags": [14.0, 5.0, 2.0, 1.0],
        "mean_flags": [14.0, 5.0, 2.0, 1.0],
        "min_flags": [6, 1, 0, 0], "max_flags": [31, 12, 5, 3],
        "months_over_target": [12, 4, 0, 0],
    })
    choice = calibrate.choose_k(c, target=5)
    assert choice.reachable is True
    assert choice.k == 4.0
    assert choice.achieved_median == 5.0


def test_choose_k_says_so_when_the_target_is_unreachable():
    c = pd.DataFrame({
        "k": [3.0, 4.0], "n_months": [12, 12],
        "median_flags": [40.0, 30.0], "mean_flags": [40.0, 30.0],
        "min_flags": [20, 15], "max_flags": [60, 45],
        "months_over_target": [12, 12],
    })
    choice = calibrate.choose_k(c, target=5)
    assert choice.reachable is False
    assert choice.k is None
    assert "30" in choice.reason and "5" in choice.reason


def test_choose_k_on_an_empty_curve_is_unreachable_not_a_crash():
    c = pd.DataFrame(columns=["k", "n_months", "median_flags", "mean_flags",
                              "min_flags", "max_flags", "months_over_target"])
    choice = calibrate.choose_k(c, target=5)
    assert choice.reachable is False and choice.k is None


def test_the_calibrated_k_actually_delivers_roughly_the_target():
    # The end-to-end promise of the whole design, checked on synthetic data.
    df = _resolved(n_per_month=500, months=12, seed=9)
    c = calibrate.curve(df, ks=calibrate.k_grid(2.0, 8.0, 0.25),
                        min_cell_n=100, target=5)
    choice = calibrate.choose_k(c, target=5)
    assert choice.reachable
    counts = calibrate.lomo_counts(df, k=choice.k, min_cell_n=100)
    assert float(np.median(list(counts.values()))) <= 5
