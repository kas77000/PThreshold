import numpy as np
import pandas as pd

from tests import synthetic


def test_book_has_the_raw_extract_headers():
    df = synthetic.make_book(n_per_month=10, months=2)
    for col in ["aggrTgtId", "Strategy", "Sym", "Date", "ePvwap/Sprd",
                "Pvwap", "Sprd", "%Adv", "Vol", "PR", "Dur"]:
        assert col in df.columns, col


def test_row_count_is_exact_and_spans_the_requested_months():
    df = synthetic.make_book(n_per_month=50, months=12, start="2025-07-01")
    assert len(df) == 50 * 12
    d = pd.to_datetime(df["Date"])
    assert d.min().strftime("%Y-%m") == "2025-07"
    assert d.max().strftime("%Y-%m") == "2026-06"
    assert d.dt.to_period("M").nunique() == 12


def test_metric_is_fat_tailed_not_gaussian():
    # Excess kurtosis well above 0 is the whole point: on a Gaussian book the
    # percentile floor and the 4-sigma term would agree and the design's
    # central question would never arise.
    df = synthetic.make_book(n_per_month=2000, months=12, seed=1)
    x = df["ePvwap/Sprd"].to_numpy(dtype=float)
    z = (x - x.mean()) / x.std(ddof=1)
    excess_kurtosis = float((z ** 4).mean() - 3.0)
    assert excess_kurtosis > 3.0


def test_seed_makes_it_reproducible():
    a = synthetic.make_book(n_per_month=20, months=2, seed=7)
    b = synthetic.make_book(n_per_month=20, months=2, seed=7)
    pd.testing.assert_frame_equal(a, b)


def test_order_ids_are_unique():
    df = synthetic.make_book(n_per_month=100, months=6)
    assert df["aggrTgtId"].is_unique


def test_unmapped_rows_are_injected_on_request():
    df = synthetic.make_book(n_per_month=10, months=1, unmapped_rows=5)
    assert (df["Strategy"] == "PART").sum() == 5


def test_benchmarks_have_visibly_different_centres():
    # TWAP and VWAP must not be interchangeable, or pooling-vs-splitting
    # tests downstream would pass for the wrong reason.
    df = synthetic.make_book(n_per_month=2000, months=12, seed=2)
    means = df.groupby("Strategy")["ePvwap/Sprd"].mean()
    assert abs(means["VWAP"] - means["TWAP"]) > 0.05
