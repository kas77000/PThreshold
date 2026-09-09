import pytest

from perfthreshold import config, schema


def test_benchmark_lookup_is_case_insensitive():
    assert config.benchmark_for("vwap") == "VWAP"
    assert config.benchmark_for("VWAP") == "VWAP"
    assert config.benchmark_for("  Vwap  ") == "VWAP"


def test_unmapped_strategy_returns_none_not_a_default():
    # The single most dangerous failure mode: a band fitted on the wrong
    # benchmark still fits and still looks plausible. Never guess.
    assert config.benchmark_for("PART") is None
    assert config.benchmark_for("") is None
    assert config.benchmark_for(None) is None


def test_wildcard_market_group_matches_everything():
    groups = {"ALL": "*"}
    assert config.market_group_for("HK", groups) == "ALL"
    assert config.market_group_for("ZZ", groups) == "ALL"


def test_explicit_market_group_membership():
    groups = {"TIGHT": ["HK", "JP"], "WIDE": ["IN", "TH"]}
    assert config.market_group_for("hk", groups) == "TIGHT"
    assert config.market_group_for("TH", groups) == "WIDE"


def test_market_outside_every_group_is_none():
    groups = {"TIGHT": ["HK", "JP"]}
    assert config.market_group_for("IN", groups) is None


def test_percentile_default_and_metric_contract():
    assert config.PERCENTILE == 99.5
    assert config.METRIC_COLUMN == "ePvwap/Sprd"
    assert config.METRIC_UNITS == "spreads"


def test_column_map_covers_every_required_schema_field():
    mapped = set(config.COLUMN_MAP.values())
    missing = [c for c in schema.REQUIRED if c not in mapped]
    assert missing == [], f"COLUMN_MAP has no source for: {missing}"
