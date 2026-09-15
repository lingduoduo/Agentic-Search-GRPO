"""The domain taxonomy is one registry; tags are derived, not written.

The migration table below is the load-bearing test: it pins every tag the
removed CAPABILITY_ROUTES defined, so this change is provably a refactor.
"""

from __future__ import annotations

import inspect

import pytest

from src.internal.tools.public_data import public_data_tools
from src.internal.tools.search import (
    DomainSearch,
    AVAILABLE_DOMAINS,
    DOMAIN_REGISTRY,
    Capability,
    iter_capabilities,
)

# Exactly what CAPABILITY_ROUTES mapped before this change.
LEGACY_ROUTES = {
    "general.wikipedia": ("search_wikipedia", "query"),
    "academic.arxiv": ("search_arxiv", "query"),
    "resource.wayback": ("search_wayback", "url"),
    "finance.quote": ("get_stock_quote", "symbol"),
    "finance.crypto": ("get_crypto_price", "symbol"),
    "finance.currency": ("convert_currency", "from_currency"),
    "environment.weather": ("get_weather", "location"),
    "travel.location": ("search_location", "query"),
    "travel.nearby": ("search_nearby_places", "query"),
}


def test_every_legacy_tag_still_resolves_identically():
    by_tag = dict(iter_capabilities())
    for tag, (tool_name, query_parameter) in LEGACY_ROUTES.items():
        assert tag in by_tag, f"{tag} disappeared"
        assert by_tag[tag].tool_name == tool_name, tag
        assert by_tag[tag].query_parameter == query_parameter, tag


def test_no_extra_tool_backed_capabilities_were_invented():
    # A refactor adds no routes. `web` is tool-less and excluded.
    tool_backed = {tag for tag, cap in iter_capabilities() if cap.tool_name}
    assert tool_backed == set(LEGACY_ROUTES)


def test_every_domain_declares_a_web_capability():
    for name, domain in DOMAIN_REGISTRY.items():
        names = [c.name for c in domain.capabilities]
        assert "web" in names, name


def test_the_web_capability_is_tool_less():
    for name, domain in DOMAIN_REGISTRY.items():
        web = next(c for c in domain.capabilities if c.name == "web")
        assert web.tool_name is None, name
        assert web.query_parameter == "query"
        assert web.returns == "documents"


def test_registry_keeps_its_seventeen_keys_in_order():
    assert len(DOMAIN_REGISTRY) == 17
    assert list(DOMAIN_REGISTRY) == AVAILABLE_DOMAINS
    assert AVAILABLE_DOMAINS[0] == "general"


def test_returns_is_one_of_two_values():
    for tag, cap in iter_capabilities():
        assert cap.returns in ("documents", "records"), tag


def test_document_and_record_capabilities_are_both_present():
    # A constant field would carry no information; this pins the split.
    kinds = {cap.returns for _tag, cap in iter_capabilities()}
    assert kinds == {"documents", "records"}


def test_capability_names_carry_no_dot():
    # The dot is the tag separator; a dotted name would make tags ambiguous.
    for _tag, cap in iter_capabilities():
        assert "." not in cap.name, cap.name


@pytest.mark.parametrize("tag", sorted(LEGACY_ROUTES))
def test_tags_are_derived_from_their_domain(tag):
    domain, _, capability = tag.partition(".")
    assert domain in DOMAIN_REGISTRY
    assert capability in {c.name for c in DOMAIN_REGISTRY[domain].capabilities}


def test_capability_is_immutable():
    cap = Capability(name="x", query_parameter="query", returns="documents")
    with pytest.raises(Exception):
        cap.name = "y"


def test_routes_match_the_legacy_table():
    service = DomainSearch(tools=public_data_tools())
    resolved = {
        tag: (tool.name, parameter) for tag, (tool, parameter) in service.routes.items()
    }
    assert resolved == LEGACY_ROUTES


def test_a_capability_naming_an_unseeded_tool_is_dropped():
    # DomainSearch must stay usable on a partial tool list; MCP relies on it.
    service = DomainSearch(tools=[])
    assert service.routes == {}


def test_discovery_lists_web_first_then_capabilities():
    service = DomainSearch(tools=public_data_tools())
    entries = service.get_sub_domains(["finance"])["domains"][0]["sub_domains"]
    assert entries[0]["sub_domain"] == "finance.web"
    assert {e["sub_domain"] for e in entries} == {
        "finance.web",
        "finance.quote",
        "finance.crypto",
        "finance.currency",
    }


def test_discovery_reports_what_each_capability_returns():
    service = DomainSearch(tools=public_data_tools())
    entries = service.get_sub_domains(["finance"])["domains"][0]["sub_domains"]
    by_tag = {e["sub_domain"]: e for e in entries}
    assert by_tag["finance.web"]["returns"] == "documents"
    assert by_tag["finance.quote"]["returns"] == "records"


def test_discovery_omits_capabilities_whose_tool_is_absent():
    service = DomainSearch(tools=[])
    entries = service.get_sub_domains(["finance"])["domains"][0]["sub_domains"]
    assert [e["sub_domain"] for e in entries] == ["finance.web"]


def test_discovery_still_rejects_bad_domain_counts():
    service = DomainSearch(tools=[])
    with pytest.raises(ValueError, match="one to five"):
        service.get_sub_domains([])


def test_search_no_longer_accepts_the_aliases():
    parameters = inspect.signature(DomainSearch.search).parameters
    assert "sub_domain" not in parameters
    assert "sub_domain_params" not in parameters
    # The surviving names stay, so callers holding a tag are unaffected.
    for name in ("query", "domain", "tag", "params", "max_results"):
        assert name in parameters


@pytest.mark.asyncio
async def test_passing_a_removed_alias_is_rejected_not_ignored():
    # Silently dropping it would send an unintended query to a provider.
    service = DomainSearch(tools=[])
    with pytest.raises(TypeError):
        await service.search("q", sub_domain="finance.quote")


def test_the_schema_no_longer_advertises_the_aliases():
    from src.internal.tools.search import _search_properties

    properties = _search_properties()
    assert "sub_domain" not in properties
    assert "sub_domain_params" not in properties
    assert set(properties) == {"query", "domain", "tag", "params", "max_results"}


def test_mcp_wrappers_drop_the_aliases():
    from src.internal.mcp_server.tools import search as mcp_search

    for fn in (mcp_search.search_domain, mcp_search.batch_search):
        parameters = inspect.signature(fn).parameters
        assert "sub_domain" not in parameters, fn.__name__
        assert "sub_domain_params" not in parameters, fn.__name__
