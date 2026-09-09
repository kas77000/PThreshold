import numpy as np
import pandas as pd
import pytest

from perfthreshold import fit, groups, load, persist, rule, schema, score
from tests import synthetic


def _band_file(scope="all", k=3.0, months=12, seed=31, min_cell_n=100,
               start="2025-07-01"):
    raw = synthetic.make_book(n_per_month=300, months=months, seed=seed,
                              start=start)
    df, rep = load.prepare(raw)
    resolved, _ = groups.resolve(df, scope=scope)
    res = fit.fit_cells(resolved, k=k, min_cell_n=min_cell_n)
    return persist.BandFile(
        metric="ePvwap/Sprd", metric_units="spreads", scope=scope,
        market_groups={"ALL": "*"}, k=k, percentile=99.5,
        min_cell_n=min_cell_n, fit_start=str(rep.date_min.date()),
        fit_end=str(rep.date_max.date()), n_orders=rep.rows_kept,
        bands=res.bands, medians=res.medians)


def _month(scope="all", start="2026-08-01", seed=99, n=400):
    raw = synthetic.make_book(n_per_month=n, months=1, seed=seed, start=start)
    df, _ = load.prepare(raw)
    resolved, _ = groups.resolve(df, scope=scope)
    return resolved


def test_windows_overlap_is_false_when_any_bound_is_unknown():
    assert score.windows_overlap(None, "2026-01-01", "2025-01-01",
                                 "2025-06-01") is False


def test_windows_overlap_detects_a_real_intersection():
    assert score.windows_overlap("2025-07-01", "2026-06-30",
                                 "2026-06-01", "2026-06-30") is True
    assert score.windows_overlap("2025-07-01", "2026-06-30",
                                 "2026-08-01", "2026-08-31") is False


def test_scoring_a_month_inside_the_fit_window_is_refused():
    bf = _band_file()
    inside = _month(start="2026-01-01")   # fit window is 2025-07 .. 2026-06
    with pytest.raises(score.GuardError, match="fit window"):
        score.check_guards(bf, inside, scope="all", metric_units="spreads")


def test_scoring_a_month_after_the_fit_window_is_allowed():
    bf = _band_file()
    after = _month(start="2026-08-01")
    score.check_guards(bf, after, scope="all", metric_units="spreads")


def test_a_unit_mismatch_is_refused_naming_both_units():
    bf = _band_file()
    after = _month(start="2026-08-01")
    with pytest.raises(score.GuardError, match="bps"):
        score.check_guards(bf, after, scope="all", metric_units="bps")


def test_a_scope_mismatch_is_refused_naming_both_scopes():
    bf = _band_file(scope="all")
    after = _month(start="2026-08-01")
    with pytest.raises(score.GuardError, match="groups"):
        score.check_guards(bf, after, scope="groups", metric_units="spreads")


def test_outliers_contain_only_flagged_rows_with_both_tails_possible():
    bf = _band_file(k=1.0)
    after = _month(start="2026-08-01")
    res = score.score_month(after, bf, scope="all", metric_units="spreads")
    assert set(res.outliers[fit.ZONE]) <= {rule.OUT_LOW, rule.OUT_HIGH}
    assert len(res.outliers) == int(
        res.scored[fit.ZONE].isin(list(rule.FLAGGED)).sum())
    assert len(res.outliers) > 0


def test_outliers_are_ranked_by_how_far_outside_the_band_they_are():
    bf = _band_file(k=1.0)
    res = score.score_month(_month(start="2026-08-01"), bf, scope="all",
                            metric_units="spreads")
    assert list(res.outliers["excess"]) == sorted(
        res.outliers["excess"], reverse=True)
    assert (res.outliers["excess"] > 0).all()


def test_outlier_columns_carry_the_diagnostics_needed_to_explain_one():
    bf = _band_file(k=1.0)
    res = score.score_month(_month(start="2026-08-01"), bf, scope="all",
                            metric_units="spreads")
    for col in (schema.ORDER_ID, schema.ORDER_DATE, schema.ALGO,
                schema.BENCHMARK, schema.MARKET, schema.METRIC,
                fit.BAND_LO, fit.BAND_HI, fit.ZONE, "excess",
                schema.SLIPPAGE_BPS, schema.SPREAD_BPS, schema.SIDE):
        assert col in res.outliers.columns, col


def test_counts_report_the_total_and_the_per_cell_split():
    bf = _band_file(k=1.0)
    res = score.score_month(_month(start="2026-08-01"), bf, scope="all",
                            metric_units="spreads")
    assert res.counts["orders"] == len(res.scored)
    assert res.counts["flagged"] == len(res.outliers)
    assert set(res.counts["by_cell"]) <= {"VWAP|ALL", "TWAP|ALL"}
    assert sum(res.counts["by_cell"].values()) == res.counts["flagged"]


def test_month_label_is_recorded():
    bf = _band_file()
    res = score.score_month(_month(start="2026-08-01"), bf, scope="all",
                            metric_units="spreads")
    assert res.month == "2026-08"


def test_drift_is_zero_when_the_month_looks_like_the_fit_window():
    bf = _band_file()
    same = _month(start="2026-08-01", seed=31, n=300)
    d = score.drift(same, bf, threshold_pct=25.0)
    assert list(d.columns) == score.DRIFT_COLS
    assert not d["warn"].any()


def test_drift_warns_when_a_reference_feature_has_moved():
    bf = _band_file()
    harder = _month(start="2026-08-01").copy()
    harder[schema.PCT_ADV] = harder[schema.PCT_ADV] * 3.0
    d = score.drift(harder, bf, threshold_pct=25.0).set_index("feature")
    assert bool(d.loc[schema.PCT_ADV, "warn"])
    assert d.loc[schema.PCT_ADV, "pct_change"] > 100.0


def test_orders_in_a_cell_the_band_never_saw_are_no_band_not_flags():
    bf = _band_file(k=1.0)
    bf.bands = bf.bands[bf.bands["cell_key"] == "VWAP|ALL"]
    res = score.score_month(_month(start="2026-08-01"), bf, scope="all",
                            metric_units="spreads")
    twap = res.scored[res.scored[schema.BENCHMARK] == "TWAP"]
    assert set(twap[fit.ZONE]) == {rule.NO_BAND}
    assert res.counts["no_band"] == len(twap)
    assert (res.outliers[schema.BENCHMARK] == "TWAP").sum() == 0
