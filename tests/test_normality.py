import numpy as np
import pandas as pd
import pytest

from perfthreshold import groups, load, normality, schema
from tests import synthetic


def _resolved(n_per_month=1000, months=12, seed=5):
    raw = synthetic.make_book(n_per_month=n_per_month, months=months, seed=seed)
    df, _ = load.prepare(raw)
    out, _ = groups.resolve(df, scope="all")
    return out


def test_a_gaussian_sample_looks_gaussian():
    x = np.random.default_rng(0).normal(0.0, 2.0, 200_000)
    s = normality.stats(x, k=4.0)
    assert abs(s["skew"]) < 0.05
    assert abs(s["excess_kurtosis"]) < 0.1
    assert s["sd_over_mad"] == pytest.approx(1.0, abs=0.02)
    assert s["verdict"] == normality.CLOSE_TO_NORMAL


def test_a_fat_tailed_sample_is_called_out():
    x = np.random.default_rng(1).standard_t(3, 200_000)
    s = normality.stats(x, k=4.0)
    assert s["excess_kurtosis"] > 3.0
    assert s["sd_over_mad"] > 1.3
    assert s["verdict"] != normality.CLOSE_TO_NORMAL


def test_expected_beyond_is_the_normal_tail_count():
    # 4 sigma two-sided is 6.334e-5 of the population.
    x = np.random.default_rng(2).normal(0.0, 1.0, 100_000)
    s = normality.stats(x, k=4.0)
    assert s["expected_beyond"] == pytest.approx(100_000 * 6.334e-5, rel=0.01)


def test_observed_beyond_counts_the_real_exceedances():
    # Built by hand: 996 inside, 4 far outside.
    x = np.concatenate([np.zeros(996), np.array([50.0, -50.0, 60.0, -60.0])])
    s = normality.stats(x, k=4.0)
    assert s["observed_beyond"] == 4


def test_tail_ratio_is_observed_over_expected():
    x = np.random.default_rng(3).standard_t(3, 100_000)
    s = normality.stats(x, k=4.0)
    assert s["tail_ratio"] == pytest.approx(
        s["observed_beyond"] / s["expected_beyond"])
    assert s["tail_ratio"] > 10          # this is the coverage shortfall


def test_the_verdict_uses_effect_size_not_the_p_value():
    # A t with 150 degrees of freedom at n=500k: Jarque-Bera rejects normality
    # at p ~ 1e-12, but only ~1.7x the expected orders fall beyond 4 sigma, so
    # the coverage claim substantially holds. A verdict driven by the p-value
    # would condemn a book that is fine for the purpose; the verdict must
    # track the tail, which is what the threshold depends on.
    x = np.random.default_rng(4).standard_t(150, 500_000)
    s = normality.stats(x, k=4.0)
    assert s["jb_p_value"] < 1e-10        # the test is emphatic...
    assert s["tail_ratio"] < normality.FAT_AT
    assert s["verdict"] == normality.CLOSE_TO_NORMAL   # ...and not the arbiter


def test_jarque_bera_is_near_zero_for_a_gaussian():
    x = np.random.default_rng(5).normal(0.0, 1.0, 50_000)
    s = normality.stats(x, k=4.0)
    assert s["jarque_bera"] < 50
    x_fat = np.random.default_rng(6).standard_t(3, 50_000)
    assert normality.stats(x_fat, k=4.0)["jarque_bera"] > 1_000


def test_report_gives_one_row_per_cell():
    df = _resolved()
    rep = normality.report(df, k=4.0)
    assert list(rep.columns) == normality.NORMALITY_COLS
    assert set(rep["cell_key"]) == {"VWAP|ALL", "TWAP|ALL"}


def test_report_on_an_empty_frame_is_empty_not_a_crash():
    df = _resolved().iloc[0:0]
    rep = normality.report(df, k=4.0)
    assert len(rep) == 0
    assert list(rep.columns) == normality.NORMALITY_COLS


def test_stats_on_too_few_points_does_not_raise():
    s = normality.stats(np.array([1.0]), k=4.0)
    assert s["n"] == 1
    assert np.isnan(s["skew"])
    s0 = normality.stats(np.array([]), k=4.0)
    assert s0["n"] == 0


def test_nonfinite_values_are_dropped():
    x = np.array([np.nan, np.inf, -np.inf, 1.0, 2.0, 3.0])
    assert normality.stats(x, k=4.0)["n"] == 3


def test_theoretical_quantiles_are_symmetric_and_ordered():
    q = normality.theoretical_quantiles(1000)
    assert len(q) == 1000
    assert np.all(np.diff(q) > 0)
    assert q[0] == pytest.approx(-q[-1], abs=1e-9)
