"""Contracts for optional search topic domains."""

import pytest

from src.internal.tools.search import (
    AVAILABLE_DOMAINS,
    DOMAIN_REGISTRY,
    normalize_search_domain,
    prepare_domain_query,
    search_domain_parameter,
)

EXPECTED = "web academic code social_media resource news official legal financial patent product business local travel media books education events jobs statistics".split()


def test_registry_contract():
    assert AVAILABLE_DOMAINS == EXPECTED
    assert list(DOMAIN_REGISTRY) == EXPECTED
    assert all(item.description for item in DOMAIN_REGISTRY.values())
    assert DOMAIN_REGISTRY["web"].query_hint == ""
    assert DOMAIN_REGISTRY["patent"].query_hint == "patents intellectual property"


@pytest.mark.parametrize("value", ["Social Media", "social-media", " social_media "])
def test_normalize_alias(value):
    assert normalize_search_domain(value) == "social_media"


@pytest.mark.parametrize("value", [None, 1, [], "", " ", "unknown"])
def test_invalid_domain(value):
    with pytest.raises(ValueError, match="web"):
        normalize_search_domain(value)


def test_prepare_query():
    assert prepare_domain_query("  battery recycling  ") == "  battery recycling  "
    assert (
        prepare_domain_query("battery recycling", "academic")
        == "battery recycling academic research"
    )
    assert prepare_domain_query("  ", "academic") == "  "


def test_schema_is_independent():
    first = search_domain_parameter()
    first["enum"].clear()
    second = search_domain_parameter()
    assert second["enum"] == EXPECTED
    assert second["default"] == "web"
    assert "intellectual property" in second["description"].lower()


@pytest.mark.parametrize(
    "value,expected",
    [
        ("general", "web"),
        ("finance", "financial"),
        ("ip", "patent"),
        ("health", "web"),
        ("security", "web"),
        ("energy", "web"),
        ("environment", "web"),
        ("agriculture", "web"),
        ("gaming", "web"),
        ("film", "media"),
    ],
)
def test_legacy_selector_alias(value, expected):
    assert normalize_search_domain(value) == expected


def test_typical_sources_are_present_and_immutable():
    assert DOMAIN_REGISTRY["web"].typical_sources == ("Web search engines",)
    assert DOMAIN_REGISTRY["academic"].typical_sources == (
        "arXiv",
        "Semantic Scholar",
        "OpenAlex",
    )
    assert all(
        isinstance(entry.typical_sources, tuple) and entry.typical_sources
        for entry in DOMAIN_REGISTRY.values()
    )
