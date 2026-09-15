"""Contracts for optional search topic domains."""

import pytest

from src.internal.tools.search import (
    AVAILABLE_DOMAINS,
    DOMAIN_REGISTRY,
    normalize_search_domain,
    prepare_domain_query,
    search_domain_parameter,
)

EXPECTED = [
    "general",
    "resource",
    "social_media",
    "finance",
    "academic",
    "legal",
    "health",
    "business",
    "security",
    "ip",
    "code",
    "energy",
    "environment",
    "agriculture",
    "travel",
    "film",
    "gaming",
]


def test_registry_contract():
    assert AVAILABLE_DOMAINS == EXPECTED
    assert list(DOMAIN_REGISTRY) == EXPECTED
    assert all(item.description for item in DOMAIN_REGISTRY.values())
    assert DOMAIN_REGISTRY["general"].query_hint == ""
    assert DOMAIN_REGISTRY["ip"].query_hint == "intellectual property"


@pytest.mark.parametrize("value", ["Social Media", "social-media", " social_media "])
def test_normalize_alias(value):
    assert normalize_search_domain(value) == "social_media"


@pytest.mark.parametrize("value", [None, 1, [], "", " ", "unknown"])
def test_invalid_domain(value):
    with pytest.raises(ValueError, match="general"):
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
    assert second["default"] == "general"
    assert "intellectual property" in second["description"].lower()
