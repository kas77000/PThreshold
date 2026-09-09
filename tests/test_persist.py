import json
import os

import numpy as np
import pandas as pd
import pytest

from perfthreshold import calibrate, fit, groups, load, persist
from tests import synthetic


def _reject(name):
    raise AssertionError(f"invalid JSON constant emitted: {name}")


def _fit_result(tmp_path):
    raw = synthetic.make_book(n_per_month=200, months=12, seed=21)
    csv = tmp_path / "year.csv"
    raw.to_csv(csv, index=False)
    df, rep = load.prepare(raw)
    resolved, _ = groups.resolve(df, scope="all")
    res = fit.fit_cells(resolved, k=4.0, min_cell_n=100)
    curve = calibrate.curve(resolved, ks=[3.0, 4.0], min_cell_n=100, target=5)
    return persist.BandFile(
        version=persist.ARTIFACT_VERSION, created_utc="2026-09-09T00:00:00Z",
        metric="ePvwap/Sprd", metric_units="spreads", scope="all",
        market_groups={"ALL": "*"}, k=4.0, percentile=99.5, min_cell_n=100,
        k_mode="target", k_reason="because", target=5,
        fit_start=str(rep.date_min.date()), fit_end=str(rep.date_max.date()),
        n_orders=rep.rows_kept, source_file=str(csv),
        source_hash=persist.file_hash(str(csv)),
        bands=res.bands, medians=res.medians, calibration=curve)


def test_write_produces_both_files(tmp_path):
    bf = _fit_result(tmp_path)
    paths = persist.write(str(tmp_path / "out"), bf)
    assert os.path.exists(paths["json"]) and os.path.exists(paths["csv"])
    assert os.path.basename(paths["json"]) == "bands.json"


def test_round_trip_preserves_every_band_number(tmp_path):
    bf = _fit_result(tmp_path)
    paths = persist.write(str(tmp_path / "out"), bf)
    back = persist.read(paths["json"])
    pd.testing.assert_frame_equal(
        back.bands.sort_values("cell_key").reset_index(drop=True),
        bf.bands.sort_values("cell_key").reset_index(drop=True),
        check_dtype=False)


def test_round_trip_preserves_provenance_and_rule_settings(tmp_path):
    bf = _fit_result(tmp_path)
    back = persist.read(persist.write(str(tmp_path / "out"), bf)["json"])
    assert back.k == 4.0
    assert back.percentile == 99.5
    assert back.scope == "all"
    assert back.metric_units == "spreads"
    assert back.market_groups == {"ALL": "*"}
    assert back.fit_start == bf.fit_start and back.fit_end == bf.fit_end
    assert back.source_hash == bf.source_hash
    assert back.n_orders == bf.n_orders
    assert back.k_reason == "because"


def test_round_trip_preserves_reference_medians_and_calibration(tmp_path):
    bf = _fit_result(tmp_path)
    back = persist.read(persist.write(str(tmp_path / "out"), bf)["json"])
    assert set(back.medians) == set(bf.medians)
    assert list(back.calibration.columns) == list(bf.calibration.columns)
    assert len(back.calibration) == len(bf.calibration)


def test_nan_survives_as_nan_and_the_json_stays_valid(tmp_path):
    bf = _fit_result(tmp_path)
    bf.bands.loc[0, "hi"] = np.nan
    paths = persist.write(str(tmp_path / "out"), bf)
    # Must parse with a strict reader: bare NaN is not valid JSON.
    with open(paths["json"], encoding="utf-8") as fh:
        json.loads(fh.read(), parse_constant=_reject)
    back = persist.read(paths["json"])
    assert np.isnan(back.bands.loc[0, "hi"])


def test_reading_a_future_version_is_refused_naming_both_versions(tmp_path):
    bf = _fit_result(tmp_path)
    paths = persist.write(str(tmp_path / "out"), bf)
    with open(paths["json"], encoding="utf-8") as fh:
        blob = json.load(fh)
    blob["version"] = persist.ARTIFACT_VERSION + 99
    with open(paths["json"], "w", encoding="utf-8") as fh:
        json.dump(blob, fh)
    with pytest.raises(ValueError, match=str(persist.ARTIFACT_VERSION)):
        persist.read(paths["json"])


def test_file_hash_is_content_addressed(tmp_path):
    a, b = tmp_path / "a.csv", tmp_path / "b.csv"
    a.write_text("x,y\n1,2\n", encoding="utf-8")
    b.write_text("x,y\n1,2\n", encoding="utf-8")
    assert persist.file_hash(str(a)) == persist.file_hash(str(b))
    b.write_text("x,y\n1,3\n", encoding="utf-8")
    assert persist.file_hash(str(a)) != persist.file_hash(str(b))


def test_csv_carries_the_band_columns_in_order(tmp_path):
    bf = _fit_result(tmp_path)
    paths = persist.write(str(tmp_path / "out"), bf)
    csv = pd.read_csv(paths["csv"])
    assert list(csv.columns) == fit.BAND_COLS


def test_write_creates_a_missing_output_directory(tmp_path):
    bf = _fit_result(tmp_path)
    target = str(tmp_path / "deep" / "nested" / "out")
    paths = persist.write(target, bf)
    assert os.path.exists(paths["json"])
