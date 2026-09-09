import numpy as np
import pytest

from perfthreshold import rule


def test_sigma_term_binds_when_it_is_wider():
    # 0..99 evenly spaced: sd is large relative to the extremes, so
    # mean + 4*sd sits far beyond P99.5 and must win.
    x = np.arange(100, dtype=float)
    b = rule.bounds(x, k=4.0, percentile=99.5)
    assert b["hi_binds"] == "sigma"
    assert b["lo_binds"] == "sigma"
    assert b["hi"] == pytest.approx(b["sigma_hi"])
    assert b["hi"] > b["p_hi"]


def test_percentile_term_binds_when_sigma_is_narrow():
    # k = 0 collapses the sigma term onto the mean, so the percentile must win
    # on both sides. This is the cleanest possible check that MAX/MIN pick the
    # right candidate rather than always taking the same one.
    x = np.arange(100, dtype=float)
    b = rule.bounds(x, k=0.0, percentile=99.5)
    assert b["hi_binds"] == "percentile"
    assert b["lo_binds"] == "percentile"
    assert b["hi"] == pytest.approx(b["p_hi"])
    assert b["lo"] == pytest.approx(b["p_lo"])


def test_bounds_are_hand_computable():
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    b = rule.bounds(x, k=1.0, percentile=99.5)
    assert b["n"] == 5
    assert b["mean"] == pytest.approx(3.0)
    assert b["sd"] == pytest.approx(np.std(x, ddof=1))   # ddof=1, not 0
    assert b["sigma_hi"] == pytest.approx(3.0 + np.std(x, ddof=1))
    assert b["sigma_lo"] == pytest.approx(3.0 - np.std(x, ddof=1))


def test_lower_percentile_mirrors_the_upper_one():
    x = np.arange(1000, dtype=float)
    b = rule.bounds(x, k=0.0, percentile=99.5)
    assert b["p_hi"] == pytest.approx(np.percentile(x, 99.5))
    assert b["p_lo"] == pytest.approx(np.percentile(x, 0.5))


def test_robust_pair_is_computed_but_never_used_for_bounds():
    x = np.array([1.0, 2.0, 3.0, 4.0, 100.0])
    b = rule.bounds(x, k=4.0, percentile=99.5)
    assert b["median"] == pytest.approx(3.0)
    assert b["mad_sigma"] == pytest.approx(1.4826 * 1.0)
    # The active band must come from the classical estimator.
    assert b["hi"] == pytest.approx(max(b["sigma_hi"], b["p_hi"]))


def test_nans_are_dropped_not_propagated():
    x = np.array([1.0, np.nan, 3.0, np.inf, 5.0])
    b = rule.bounds(x, k=1.0, percentile=99.5)
    assert b["n"] == 3
    assert np.isfinite(b["hi"]) and np.isfinite(b["lo"])


def test_empty_input_gives_nan_bounds_and_zero_n():
    b = rule.bounds(np.array([]), k=4.0, percentile=99.5)
    assert b["n"] == 0
    assert np.isnan(b["hi"]) and np.isnan(b["lo"])
    assert b["hi_binds"] == "" and b["lo_binds"] == ""


def test_single_observation_has_zero_sd_not_a_crash():
    b = rule.bounds(np.array([7.0]), k=4.0, percentile=99.5)
    assert b["n"] == 1
    assert b["sd"] == 0.0
    assert b["hi"] == pytest.approx(7.0)


def test_zone_labels_both_tails_and_the_middle():
    assert rule.zone(0.0, lo=-1.0, hi=1.0) == rule.IN_RANGE
    assert rule.zone(-5.0, lo=-1.0, hi=1.0) == rule.OUT_LOW
    assert rule.zone(5.0, lo=-1.0, hi=1.0) == rule.OUT_HIGH
    assert rule.zone(np.nan, lo=-1.0, hi=1.0) == rule.NO_BAND
    assert rule.zone(0.0, lo=np.nan, hi=1.0) == rule.NO_BAND


def test_bounds_are_inclusive_so_a_value_exactly_on_the_edge_is_in_range():
    # An order sitting exactly on the bound should not be flagged: the band is
    # the acceptable region, and a strict inequality would make the flag count
    # depend on floating-point luck.
    assert rule.zone(1.0, lo=-1.0, hi=1.0) == rule.IN_RANGE
    assert rule.zone(-1.0, lo=-1.0, hi=1.0) == rule.IN_RANGE


def test_count_flags_counts_both_tails():
    x = np.array([-10.0, 0.0, 0.5, 10.0, np.nan])
    assert rule.count_flags(x, lo=-1.0, hi=1.0) == 2


def test_count_flags_is_zero_when_the_band_is_undefined():
    x = np.array([-10.0, 0.0, 10.0])
    assert rule.count_flags(x, lo=np.nan, hi=1.0) == 0
