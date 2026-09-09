import numpy as np
import pandas as pd
import pytest

from perfthreshold import load, schema
from tests import synthetic


def test_raw_headers_become_canonical_names():
    raw = synthetic.make_book(n_per_month=20, months=1)
    df, _ = load.prepare(raw)
    assert schema.METRIC in df.columns
    assert schema.ORDER_ID in df.columns
    assert "ePvwap/Sprd" not in df.columns, "raw name must not survive"
    assert "aggrTgtId" not in df.columns


def test_market_is_the_ticker_suffix():
    raw = pd.DataFrame({
        "aggrTgtId": ["A", "B"], "Strategy": ["VWAP", "VWAP"],
        "Sym": ["0700 HK", "7203 jt"], "Date": ["2025-07-01", "2025-07-02"],
        "ePvwap/Sprd": [0.1, 0.2], "Pvwap": [1.0, 2.0], "Sprd": [8.0, 9.0],
    })
    df, _ = load.prepare(raw)
    assert list(df[schema.MARKET]) == ["HK", "JT"]
    assert list(df[schema.SYMBOL]) == ["0700 HK", "7203 jt"]


def test_benchmark_is_derived_and_tmx_maps_to_twap():
    raw = pd.DataFrame({
        "aggrTgtId": ["A", "B", "C"], "Strategy": ["VWAP", "TMX", "twap"],
        "Sym": ["1 HK"] * 3, "Date": ["2025-07-01"] * 3,
        "ePvwap/Sprd": [0.1, 0.2, 0.3], "Pvwap": [1.0] * 3, "Sprd": [8.0] * 3,
    })
    df, _ = load.prepare(raw)
    assert list(df[schema.BENCHMARK]) == ["VWAP", "TWAP", "TWAP"]


def test_unmapped_strategy_is_excluded_and_named_with_a_count():
    raw = synthetic.make_book(n_per_month=10, months=1, unmapped_rows=4)
    df, rep = load.prepare(raw)
    assert (df[schema.ALGO] == "PART").sum() == 0
    assert rep.unmapped_strategies == {"PART": 4}
    assert rep.dropped[load.UNMAPPED_STRATEGY] == 4


def test_missing_metric_is_dropped_with_a_reason():
    raw = pd.DataFrame({
        "aggrTgtId": ["A", "B"], "Strategy": ["VWAP", "VWAP"],
        "Sym": ["1 HK", "2 HK"], "Date": ["2025-07-01", "2025-07-02"],
        "ePvwap/Sprd": [0.1, np.nan], "Pvwap": [1.0, 2.0], "Sprd": [8.0, 9.0],
    })
    df, rep = load.prepare(raw)
    assert len(df) == 1
    assert rep.dropped[load.MISSING_METRIC] == 1


def test_unparseable_date_is_dropped_with_a_reason():
    raw = pd.DataFrame({
        "aggrTgtId": ["A", "B"], "Strategy": ["VWAP", "VWAP"],
        "Sym": ["1 HK", "2 HK"], "Date": ["2025-07-01", "not a date"],
        "ePvwap/Sprd": [0.1, 0.2], "Pvwap": [1.0, 2.0], "Sprd": [8.0, 9.0],
    })
    df, rep = load.prepare(raw)
    assert len(df) == 1
    assert rep.dropped[load.MISSING_DATE] == 1


def test_dates_are_normalised_to_midnight():
    raw = pd.DataFrame({
        "aggrTgtId": ["A"], "Strategy": ["VWAP"], "Sym": ["1 HK"],
        "Date": ["2025-07-01 14:33:07"], "ePvwap/Sprd": [0.1],
        "Pvwap": [1.0], "Sprd": [8.0],
    })
    df, _ = load.prepare(raw)
    assert df[schema.ORDER_DATE].iloc[0] == pd.Timestamp("2025-07-01")


def test_every_input_row_is_accounted_for():
    # The contract that makes "no silent drops" checkable rather than a claim.
    raw = synthetic.make_book(n_per_month=50, months=3, unmapped_rows=7)
    df, rep = load.prepare(raw)
    assert rep.rows_in == len(raw)
    assert rep.rows_kept == len(df)
    assert rep.rows_kept + sum(rep.dropped.values()) == rep.rows_in


def test_a_row_failing_two_checks_is_counted_once():
    # Counted under its FIRST failing reason, so the drop counts sum to the
    # number of rows actually removed rather than double-counting.
    raw = pd.DataFrame({
        "aggrTgtId": [None], "Strategy": ["PART"], "Sym": ["1 HK"],
        "Date": ["bad"], "ePvwap/Sprd": [np.nan], "Pvwap": [1.0], "Sprd": [8.0],
    })
    df, rep = load.prepare(raw)
    assert len(df) == 0
    assert sum(rep.dropped.values()) == 1


def test_report_records_the_observed_date_window():
    raw = synthetic.make_book(n_per_month=10, months=12, start="2025-07-01")
    _, rep = load.prepare(raw)
    assert rep.date_min.strftime("%Y-%m") == "2025-07"
    assert rep.date_max.strftime("%Y-%m") == "2026-06"


def test_missing_required_source_column_raises_naming_it():
    raw = pd.DataFrame({"aggrTgtId": ["A"], "Strategy": ["VWAP"],
                        "Sym": ["1 HK"], "Date": ["2025-07-01"]})
    with pytest.raises(ValueError, match="ePvwap"):
        load.prepare(raw)


def test_report_to_frame_is_one_row_per_reason():
    raw = synthetic.make_book(n_per_month=10, months=1, unmapped_rows=3)
    _, rep = load.prepare(raw)
    frame = rep.to_frame()
    assert list(frame.columns) == ["reason", "rows"]
    assert (frame["reason"] == load.UNMAPPED_STRATEGY).any()


def test_read_any_round_trips_a_csv(tmp_path):
    raw = synthetic.make_book(n_per_month=10, months=1)
    path = tmp_path / "book.csv"
    raw.to_csv(path, index=False)
    df, rep = load.load(str(path))
    assert rep.rows_kept == len(raw)
    assert schema.METRIC in df.columns


def test_read_any_rejects_an_unknown_extension(tmp_path):
    path = tmp_path / "book.txt"
    path.write_text("nope", encoding="utf-8")
    with pytest.raises(ValueError, match="Unsupported extension"):
        load.read_any(str(path))
