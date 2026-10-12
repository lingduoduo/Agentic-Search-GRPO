"""Contracts for optional search topic domains."""

import re

import pytest

from src.internal.tools.search import (
    AVAILABLE_DOMAINS,
    DOMAIN_REGISTRY,
    category_sites_enabled,
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


EXPECTED_SITES = {
    "academic": (
        "arxiv.org",
        "semanticscholar.org",
        "pubmed.ncbi.nlm.nih.gov",
        "openreview.net",
        "aclanthology.org",
    ),
    "code": ("github.com", "gitlab.com", "stackoverflow.com", "pypi.org", "npmjs.com"),
    "social_media": ("reddit.com", "news.ycombinator.com", "x.com"),
    "resource": ("huggingface.co", "kaggle.com", "data.gov", "zenodo.org"),
    "news": ("reuters.com", "apnews.com", "bbc.com", "npr.org"),
    "legal": (
        "law.cornell.edu",
        "courtlistener.com",
        "justia.com",
        "govinfo.gov",
        "eur-lex.europa.eu",
    ),
    "financial": ("sec.gov", "nasdaq.com", "finance.yahoo.com", "marketwatch.com"),
    "patent": ("patents.google.com", "uspto.gov", "wipo.int", "epo.org"),
    "business": ("opencorporates.com", "crunchbase.com"),
    "travel": ("tripadvisor.com", "booking.com", "lonelyplanet.com"),
    "media": (
        "youtube.com",
        "commons.wikimedia.org",
        "vimeo.com",
        "podcasts.apple.com",
    ),
    "books": ("openlibrary.org", "books.google.com", "goodreads.com", "worldcat.org"),
    "education": ("coursera.org", "edx.org", "khanacademy.org", "ocw.mit.edu"),
    "events": ("eventbrite.com", "meetup.com"),
    "jobs": ("indeed.com", "linkedin.com/jobs", "glassdoor.com"),
    "statistics": (
        "census.gov",
        "data.worldbank.org",
        "data.oecd.org",
        "ourworldindata.org",
    ),
}


def test_source_sites_match_the_spec():
    actual = {n: e.source_sites for n, e in DOMAIN_REGISTRY.items() if e.source_sites}
    assert actual == EXPECTED_SITES
    for name in ("web", "official", "product", "local"):
        assert DOMAIN_REGISTRY[name].source_sites == ()


def test_source_sites_are_bare_hosts_capped_at_five():
    shape = re.compile(r"^[a-z0-9.-]+\.[a-z]{2,}(/[a-z0-9_-]+)*$")
    for name, entry in DOMAIN_REGISTRY.items():
        assert len(entry.source_sites) <= 5, name
        for site in entry.source_sites:
            assert shape.match(site), (name, site)


def test_category_sites_setting_defaults_on(monkeypatch):
    monkeypatch.delenv("AGENTIC_SEARCH_CATEGORY_SITES", raising=False)
    assert category_sites_enabled() is True
    monkeypatch.setenv("AGENTIC_SEARCH_CATEGORY_SITES", "0")
    assert category_sites_enabled() is False


def test_discovery_directory_exposes_source_sites():
    from src.internal.tools.search import DomainSearch

    directory = DomainSearch(
        web_search_fn=lambda *a, **k: [], tools=[]
    ).get_sub_domains(["financial", "web"])["domains"]
    assert directory[0]["source_sites"] == list(EXPECTED_SITES["financial"])
    assert directory[1]["source_sites"] == []
