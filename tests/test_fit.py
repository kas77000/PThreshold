import numpy as np
import pandas as pd
import pytest

from perfthreshold import fit, groups, load, rule, schema
from tests import synthetic


def _resolved(scope="all", market_groups=None, n_per_month=400, months=12,
              seed=5, markets=("HK", "JP", "AU", "IN")):
    raw = synthetic.make_book(n_per_month=n_per_month, months=months,
                              seed=seed, markets=markets)
    df, _ = load.prepare(raw)
    out, _ = groups.resolve(df, scope=scope, market_groups=market_groups)
    return out


def test_one_row_per_cell_with_the_declared_columns():
    df = _resolved()
    res = fit.fit_cells(df, k=4.0, percentile=99.5, min_cell_n=100)
    assert list(res.bands.columns) == fit.BAND_COLS
    assert set(res.bands["cell_key"]) == {"VWAP|ALL", "TWAP|ALL"}
    assert len(res.bands) == 2


def test_cell_numbers_match_the_pure_rule_on_the_same_slice():
    df = _resolved()
    res = fit.fit_cells(df, k=4.0, percentile=99.5, min_cell_n=100)
    slice_ = df[df[schema.CELL_KEY] == "VWAP|ALL"][schema.METRIC].to_numpy()
    expected = rule.bounds(slice_, k=4.0, percentile=99.5)
    row = res.bands.set_index("cell_key").loc["VWAP|ALL"]
    assert row["n"] == expected["n"]
    assert row["hi"] == pytest.approx(expected["hi"])
    assert row["lo"] == pytest.approx(expected["lo"])
    assert row["hi_binds"] == expected["hi_binds"]


def test_reference_medians_are_stamped_per_cell():
    df = _resolved()
    res = fit.fit_cells(df, k=4.0, min_cell_n=100)
    med = res.medians["VWAP|ALL"]
    for feature in schema.REFERENCE_FEATURES:
        assert feature in med
        assert np.isfinite(med[feature])


def test_thin_grouped_cell_inherits_its_benchmark_pooled_parent():
    # JP is deliberately tiny; TIGHT/WIDE split makes it its own cell.
    df = _resolved(scope="groups",
                   market_groups={"TIGHT": ["HK"], "WIDE": ["JP"]},
                   n_per_month=300, months=12)
    jp = df[df[schema.MARKET] == "JP"]
    res = fit.fit_cells(df, k=4.0, min_cell_n=len(jp) + 1)
    row = res.bands.set_index("cell_key").loc["VWAP|WIDE"]
    assert not row["fitted"]
    assert row["fallback_from"] == "VWAP|" + groups.POOLED
    # The inherited bounds must equal the pooled fit over that benchmark.
    pooled = df[df[schema.BENCHMARK] == "VWAP"][schema.METRIC].to_numpy()
    expected = rule.bounds(pooled, k=4.0, percentile=99.5)
    assert row["hi"] == pytest.approx(expected["hi"])


def test_pooled_cell_below_min_n_is_left_unfitted_not_inherited():
    # Under scope=all the cell IS the pooled band, so there is no parent.
    df = _resolved(n_per_month=10, months=2)
    res = fit.fit_cells(df, k=4.0, min_cell_n=10_000)
    row = res.bands.set_index("cell_key").loc["VWAP|ALL"]
    assert not row["fitted"]
    assert row["fallback_from"] == ""
    assert np.isnan(row["hi"]) and np.isnan(row["lo"])


def test_apply_adds_bounds_and_a_zone_for_every_row():
    df = _resolved()
    res = fit.fit_cells(df, k=4.0, min_cell_n=100)
    scored = fit.apply(df, res.bands)
    assert len(scored) == len(df)
    for col in (fit.BAND_LO, fit.BAND_HI, fit.ZONE):
        assert col in scored.columns
    assert set(scored[fit.ZONE]) <= {rule.IN_RANGE, rule.OUT_LOW,
                                     rule.OUT_HIGH, rule.NO_BAND}


def test_apply_flags_both_tails():
    df = _resolved()
    res = fit.fit_cells(df, k=1.0, min_cell_n=100)   # deliberately tight
    scored = fit.apply(df, res.bands)
    assert (scored[fit.ZONE] == rule.OUT_LOW).sum() > 0
    assert (scored[fit.ZONE] == rule.OUT_HIGH).sum() > 0


def test_apply_gives_no_band_when_the_cell_was_never_fitted():
    df = _resolved(n_per_month=10, months=2)
    res = fit.fit_cells(df, k=4.0, min_cell_n=10_000)
    scored = fit.apply(df, res.bands)
    assert set(scored[fit.ZONE]) == {rule.NO_BAND}


def test_apply_gives_no_band_for_a_cell_absent_from_the_table():
    df = _resolved()
    res = fit.fit_cells(df, k=4.0, min_cell_n=100)
    only_vwap = res.bands[res.bands["cell_key"] == "VWAP|ALL"]
    scored = fit.apply(df, only_vwap)
    twap = scored[scored[schema.BENCHMARK] == "TWAP"]
    assert set(twap[fit.ZONE]) == {rule.NO_BAND}


def test_larger_k_never_flags_more_orders():
    # Monotonicity is what makes the target search well defined; if it failed,
    # the smallest qualifying k would not be unique.
    df = _resolved()
    counts = []
    for k in (2.0, 3.0, 4.0, 5.0, 6.0):
        res = fit.fit_cells(df, k=k, min_cell_n=100)
        counts.append(fit.flag_count(df, res.bands))
    assert counts == sorted(counts, reverse=True)


def test_flag_count_matches_the_zone_labels():
    df = _resolved()
    res = fit.fit_cells(df, k=3.0, min_cell_n=100)
    scored = fit.apply(df, res.bands)
    assert fit.flag_count(df, res.bands) == int(
        scored[fit.ZONE].isin(list(rule.FLAGGED)).sum())


def test_empty_frame_gives_an_empty_band_table_not_a_crash():
    df = _resolved().iloc[0:0]
    res = fit.fit_cells(df, k=4.0, min_cell_n=100)
    assert len(res.bands) == 0
    assert list(res.bands.columns) == fit.BAND_COLS
