import numpy as np
import pandas as pd
import pytest

from perfthreshold import schema, split


def _frame(spec: dict[str, np.ndarray], benchmark="VWAP") -> pd.DataFrame:
    rows = []
    for market, values in spec.items():
        for i, v in enumerate(values):
            rows.append({schema.ORDER_ID: f"{market}{i}",
                         schema.BENCHMARK: benchmark,
                         schema.MARKET: market,
                         schema.METRIC: float(v),
                         schema.ORDER_DATE: pd.Timestamp("2025-07-01")})
    return pd.DataFrame(rows)


def test_identical_markets_are_told_to_stay_pooled():
    rng = np.random.default_rng(0)
    df = _frame({"HK": rng.normal(0, 1, 5000),
                 "JP": rng.normal(0, 1, 5000)})
    rep = split.report(df, k=4.0, min_market_n=100)
    assert set(rep["verdict"]) == {split.VERDICT_POOL}


def test_a_much_wider_market_is_over_flagged_by_the_pool():
    # The pool is dominated by tight HK, so the pooled band is far too narrow
    # for wide IN -- IN is over-flagged, and its own band would flag far fewer.
    rng = np.random.default_rng(1)
    df = _frame({"HK": rng.normal(0, 1, 20000),
                 "IN": rng.normal(0, 6, 3000)})
    rep = split.report(df, k=4.0, min_market_n=100).set_index("market")
    assert rep.loc["IN", "pooled_flags"] > rep.loc["IN", "own_flags"]
    assert rep.loc["IN", "excess_flags"] > 0
    assert rep.loc["IN", "verdict"] == split.VERDICT_SPLIT


def test_excess_is_exactly_pooled_minus_own():
    rng = np.random.default_rng(2)
    df = _frame({"HK": rng.normal(0, 1, 4000), "IN": rng.normal(0, 5, 4000)})
    rep = split.report(df, k=4.0, min_market_n=100)
    assert (rep["excess_flags"] == rep["pooled_flags"] - rep["own_flags"]).all()


def test_a_thin_market_cannot_earn_its_own_band():
    rng = np.random.default_rng(3)
    df = _frame({"HK": rng.normal(0, 1, 4000), "TH": rng.normal(0, 5, 40)})
    rep = split.report(df, k=4.0, min_market_n=1000).set_index("market")
    assert rep.loc["TH", "verdict"] == split.VERDICT_THIN
    assert np.isnan(rep.loc["TH", "own_hi"])


def test_a_big_ratio_on_tiny_counts_is_not_evidence():
    # min_excess exists precisely to stop a large ratio on a handful of
    # orders reading as a reason to split.
    rng = np.random.default_rng(4)
    df = _frame({"HK": rng.normal(0, 1, 5000), "AU": rng.normal(0, 1.05, 3000)})
    rep = split.report(df, k=4.0, min_market_n=100,
                       ratio_threshold=1.01, min_excess=10_000)
    assert set(rep["verdict"]) <= {split.VERDICT_POOL}


def test_every_market_appears_once_per_benchmark():
    rng = np.random.default_rng(5)
    a = _frame({"HK": rng.normal(0, 1, 500), "JP": rng.normal(0, 1, 500)},
               benchmark="VWAP")
    b = _frame({"HK": rng.normal(0, 2, 500), "JP": rng.normal(0, 2, 500)},
               benchmark="TWAP")
    rep = split.report(pd.concat([a, b]), k=4.0, min_market_n=100)
    assert len(rep) == 4
    assert list(rep.columns) == split.SPLIT_COLS
    assert rep.duplicated([schema.BENCHMARK, "market"]).sum() == 0


def test_report_is_ranked_by_the_size_of_the_discrepancy():
    rng = np.random.default_rng(6)
    df = _frame({"HK": rng.normal(0, 1, 8000),
                 "IN": rng.normal(0, 6, 3000),
                 "JP": rng.normal(0, 1.1, 3000)})
    rep = split.report(df, k=4.0, min_market_n=100)
    fitted = rep[rep["verdict"] != split.VERDICT_THIN]
    assert list(fitted["excess_flags"]) == sorted(
        fitted["excess_flags"], reverse=True)


def test_under_flagging_shows_as_a_negative_excess_not_zero():
    # A market far tighter than the pool is under-flagged: the pooled band is
    # too wide to catch its real problems. That is worth seeing, so it is not
    # clipped at zero.
    # k=2.5 so HK's own band sits inside its own data and flags a real number;
    # at k=4 a normal market flags nothing under either band and the excess is
    # trivially 0, which would not exercise the sign at all.
    rng = np.random.default_rng(7)
    df = _frame({"IN": rng.normal(0, 8, 12000), "HK": rng.normal(0, 1, 4000)})
    rep = split.report(df, k=2.5, min_market_n=100).set_index("market")
    assert rep.loc["HK", "own_flags"] > 0
    assert rep.loc["HK", "pooled_flags"] == 0
    assert rep.loc["HK", "excess_flags"] < 0


def test_empty_frame_gives_an_empty_report():
    rep = split.report(pd.DataFrame(columns=[
        schema.BENCHMARK, schema.MARKET, schema.METRIC]), k=4.0)
    assert len(rep) == 0
    assert list(rep.columns) == split.SPLIT_COLS
