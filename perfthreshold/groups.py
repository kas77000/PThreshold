"""Which cell a row belongs to, under a given scope.

Fit and score both derive cells through this module and nothing else. That is
what guarantees the band written for VWAP|APAC_TIGHT is the one score looks up
for those same rows. If the two sides derived cells independently they would
drift apart the first time a strategy name changed case or a group was
renamed, and the mismatch would show up as a flag-rate change rather than as
an error.

Scope is the only thing that varies:

    all              every market pooled; one cell per benchmark
    groups           one cell per (benchmark x declared group)
    group:NAME       that group only; every other market excluded

All three run the same code path -- a scope only changes how a market_group is
assigned -- so a bug cannot exist in one and not the others.
"""

from __future__ import annotations

import pandas as pd

from perfthreshold import config, schema

SCOPE_ALL = "all"
SCOPE_GROUPS = "groups"
GROUP_PREFIX = "group:"

# The group name every row receives under scope=all.
ALL_GROUP = "ALL"

# Reserved name for a benchmark-pooled parent band. Reserved rather than
# reusing "ALL" so that a user who declares a group literally called ALL
# cannot collide with the fallback parent.
POOLED = "__POOLED__"

OUT_OF_SCOPE_MARKET = "market in no declared group"
OUT_OF_SCOPE_GROUP = "market outside the selected group"
OUT_OF_SCOPE_BENCHMARK = "benchmark not selected"

_SEP = "|"


def parse_scope(scope: str) -> tuple[str, str | None]:
    """('all'|'groups'|'group', group_name_or_None)."""
    s = str(scope).strip()
    if s == SCOPE_ALL:
        return SCOPE_ALL, None
    if s == SCOPE_GROUPS:
        return SCOPE_GROUPS, None
    if s.startswith(GROUP_PREFIX):
        name = s[len(GROUP_PREFIX):].strip()
        if not name:
            raise ValueError("scope 'group:' needs a group name after the colon")
        return "group", name
    raise ValueError(
        f"Unknown scope {scope!r}. Use 'all', 'groups', or 'group:NAME'.")


def cell_key(benchmark: str, market_group: str) -> str:
    return f"{benchmark}{_SEP}{market_group}"


def split_key(cell: str) -> tuple[str, str]:
    benchmark, _, market_group = str(cell).partition(_SEP)
    return benchmark, market_group


def parent_key(cell: str) -> str:
    """The benchmark-pooled band a thin cell falls back to."""
    benchmark, _ = split_key(cell)
    return cell_key(benchmark, POOLED)


def resolve(df: pd.DataFrame, scope: str = SCOPE_ALL,
            market_groups: dict | None = None,
            benchmark: str | None = None
            ) -> tuple[pd.DataFrame, dict[str, int]]:
    """Attach market_group and cell_key; drop what the scope excludes."""
    market_groups = (config.MARKET_GROUPS if market_groups is None
                     else market_groups)
    kind, wanted = parse_scope(scope)

    if kind == "group" and wanted not in market_groups:
        raise ValueError(
            f"Group {wanted!r} is not declared. MARKET_GROUPS has: "
            + ", ".join(sorted(market_groups)))

    out = df.copy()
    excluded = {OUT_OF_SCOPE_BENCHMARK: 0, OUT_OF_SCOPE_MARKET: 0,
                OUT_OF_SCOPE_GROUP: 0}

    if benchmark is not None:
        want = str(benchmark).strip().upper()
        keep = out[schema.BENCHMARK].astype(str).str.upper() == want
        excluded[OUT_OF_SCOPE_BENCHMARK] = int((~keep).sum())
        out = out[keep]

    if kind == SCOPE_ALL:
        # Declared groups are deliberately ignored here: scope=all means one
        # pooled cell per benchmark, whatever config happens to declare.
        out[schema.MARKET_GROUP] = ALL_GROUP
    else:
        assigned = out[schema.MARKET].map(
            lambda m: config.market_group_for(m, market_groups))
        out[schema.MARKET_GROUP] = assigned
        keep = assigned.notna()
        excluded[OUT_OF_SCOPE_MARKET] = int((~keep).sum())
        out = out[keep]

        if kind == "group":
            keep = out[schema.MARKET_GROUP] == wanted
            excluded[OUT_OF_SCOPE_GROUP] = int((~keep).sum())
            out = out[keep]

    out[schema.CELL_KEY] = [
        cell_key(b, g) for b, g in
        zip(out[schema.BENCHMARK], out[schema.MARKET_GROUP])
    ]
    return out.reset_index(drop=True), excluded
