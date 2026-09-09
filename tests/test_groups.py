import pandas as pd
import pytest

from perfthreshold import groups, load, schema
from tests import synthetic


def _prepared(**kw):
    raw = synthetic.make_book(n_per_month=40, months=2, seed=3, **kw)
    df, _ = load.prepare(raw)
    return df


def test_parse_scope_recognises_the_three_forms():
    assert groups.parse_scope("all") == ("all", None)
    assert groups.parse_scope("groups") == ("groups", None)
    assert groups.parse_scope("group:APAC_TIGHT") == ("group", "APAC_TIGHT")


def test_parse_scope_rejects_nonsense_naming_what_it_wanted():
    with pytest.raises(ValueError, match="all"):
        groups.parse_scope("everything")


def test_cell_key_round_trips():
    assert groups.cell_key("VWAP", "ALL") == "VWAP|ALL"
    assert groups.split_key("VWAP|ALL") == ("VWAP", "ALL")


def test_parent_key_is_the_benchmark_pooled_band():
    assert groups.parent_key("VWAP|APAC_TIGHT") == "VWAP|" + groups.POOLED


def test_scope_all_pools_every_market_regardless_of_config():
    df = _prepared()
    out, excl = groups.resolve(
        df, scope="all",
        market_groups={"TIGHT": ["HK"], "WIDE": ["IN"]})
    # The declared groups are ignored under scope=all -- that is the point.
    assert set(out[schema.MARKET_GROUP]) == {"ALL"}
    assert set(out[schema.CELL_KEY]) == {"VWAP|ALL", "TWAP|ALL"}
    assert sum(excl.values()) == 0
    assert len(out) == len(df)


def test_scope_groups_assigns_declared_membership():
    df = _prepared()
    out, excl = groups.resolve(
        df, scope="groups",
        market_groups={"TIGHT": ["HK", "JP"], "WIDE": ["AU", "IN"]})
    assert set(out[schema.MARKET_GROUP]) == {"TIGHT", "WIDE"}
    hk = out[out[schema.MARKET] == "HK"]
    assert set(hk[schema.MARKET_GROUP]) == {"TIGHT"}


def test_scope_groups_excludes_markets_in_no_declared_group():
    df = _prepared()
    out, excl = groups.resolve(
        df, scope="groups", market_groups={"TIGHT": ["HK"]})
    assert set(out[schema.MARKET]) == {"HK"}
    assert excl[groups.OUT_OF_SCOPE_MARKET] == len(df) - len(out)
    assert excl[groups.OUT_OF_SCOPE_MARKET] > 0


def test_scope_single_group_keeps_only_that_group():
    df = _prepared()
    out, excl = groups.resolve(
        df, scope="group:TIGHT",
        market_groups={"TIGHT": ["HK", "JP"], "WIDE": ["AU", "IN"]})
    assert set(out[schema.MARKET]) == {"HK", "JP"}
    assert set(out[schema.MARKET_GROUP]) == {"TIGHT"}
    assert excl[groups.OUT_OF_SCOPE_GROUP] > 0


def test_unknown_group_name_raises_listing_what_is_declared():
    df = _prepared()
    with pytest.raises(ValueError, match="TIGHT"):
        groups.resolve(df, scope="group:NOPE",
                       market_groups={"TIGHT": ["HK"]})


def test_benchmark_filter_narrows_to_one_family():
    df = _prepared()
    out, excl = groups.resolve(df, scope="all", benchmark="VWAP")
    assert set(out[schema.BENCHMARK]) == {"VWAP"}
    assert excl[groups.OUT_OF_SCOPE_BENCHMARK] > 0


def test_benchmark_filter_is_case_insensitive():
    df = _prepared()
    out, _ = groups.resolve(df, scope="all", benchmark="vwap")
    assert set(out[schema.BENCHMARK]) == {"VWAP"}


def test_every_row_is_still_accounted_for_after_resolution():
    df = _prepared()
    out, excl = groups.resolve(
        df, scope="group:TIGHT",
        market_groups={"TIGHT": ["HK"], "WIDE": ["JP"]}, benchmark="VWAP")
    assert len(out) + sum(excl.values()) == len(df)


def test_wildcard_group_under_scope_groups_catches_everything():
    df = _prepared()
    out, excl = groups.resolve(df, scope="groups", market_groups={"ALL": "*"})
    assert len(out) == len(df)
    assert sum(excl.values()) == 0


def test_scope_markets_gives_every_market_its_own_cell():
    # No declaration needed: the market IS the group. Listing markets by hand
    # in MARKET_GROUPS silently drops any you forget to list.
    df = _prepared()
    out, excl = groups.resolve(df, scope="markets")
    assert set(out[schema.MARKET_GROUP]) == set(df[schema.MARKET])
    assert set(out[schema.CELL_KEY]) == {
        f"{b}|{m}" for b, m in zip(df[schema.BENCHMARK], df[schema.MARKET])}
    assert len(out) == len(df)
    assert sum(excl.values()) == 0


def test_scope_markets_ignores_declared_groups():
    df = _prepared()
    out, _ = groups.resolve(df, scope="markets",
                            market_groups={"TIGHT": ["HK"]})
    assert "TIGHT" not in set(out[schema.MARKET_GROUP])
    assert "HK" in set(out[schema.MARKET_GROUP])


def test_scope_markets_parses():
    assert groups.parse_scope("markets") == ("markets", None)


def test_scope_markets_composes_with_a_benchmark_filter():
    df = _prepared()
    out, excl = groups.resolve(df, scope="markets", benchmark="VWAP")
    assert set(out[schema.BENCHMARK]) == {"VWAP"}
    assert all(c.startswith("VWAP|") for c in out[schema.CELL_KEY])
    assert len(out) + sum(excl.values()) == len(df)
