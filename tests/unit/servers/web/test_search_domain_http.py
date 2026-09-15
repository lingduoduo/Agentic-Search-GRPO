"""HTTP surface for the 17-domain search taxonomy.

Covers the request field, its validation, the registry endpoint that feeds the
UI selector, and where the topic hint is applied. The load-bearing rule is that
hints reach web providers only: appending a topic word to a corpus query
upweights documents containing that word rather than focusing the search.
"""

from __future__ import annotations

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from src.internal.servers.web.app import (
    AgentExperienceRequest,
    _reject_domain_for_mode,
    _run_direct_search,
    _run_hybrid_search,
    create_web_app,
)
from src.internal.tools.search import (
    AVAILABLE_DOMAINS,
    DOMAIN_REGISTRY,
    SearchPage,
)


@pytest.fixture
def client():
    with TestClient(create_web_app()) as c:
        yield c


# --- request field and registry endpoint ------------------------------------


def test_domain_defaults_to_general():
    assert AgentExperienceRequest(query="q").domain == "general"


def test_registry_endpoint_lists_all_domains_in_order(client):
    body = client.get("/api/search-domains").json()
    assert [d["name"] for d in body["domains"]] == AVAILABLE_DOMAINS
    assert len(body["domains"]) == 17


def test_registry_endpoint_carries_descriptions(client):
    body = client.get("/api/search-domains").json()
    by_name = {d["name"]: d["description"] for d in body["domains"]}
    assert by_name["finance"] == DOMAIN_REGISTRY["finance"].description


def test_invalid_domain_is_rejected(client):
    r = client.post("/api/agent", json={"query": "q", "domain": "nonsense"})
    assert r.status_code == 400
    assert "domain must be one of" in r.json()["detail"]


# --- direct search ----------------------------------------------------------


async def _passthrough_fetch(pages, **_kwargs):
    return pages


def _capture_search_tool(seen: dict[str, str]):
    async def fake_search_tool(query, *, provider, **_kwargs):
        seen[provider] = query
        return [SearchPage(title="t", url="http://x", summary="c")]

    return fake_search_tool


def _patch_providers(monkeypatch, seen: dict[str, str]) -> None:
    monkeypatch.setattr(
        "src.internal.servers.web.app.search_tool", _capture_search_tool(seen)
    )
    monkeypatch.setattr(
        "src.internal.servers.web.app.fetch_pages_concurrently", _passthrough_fetch
    )


@pytest.mark.asyncio
async def test_direct_search_hints_web_providers(monkeypatch):
    seen: dict[str, str] = {}
    _patch_providers(monkeypatch, seen)
    await _run_direct_search(
        "etf fees",
        source_provider="serpapi",
        search_url="http://x/retrieve",
        top_k=2,
        domain="finance",
    )
    assert seen["serpapi"] == "etf fees finance"


@pytest.mark.asyncio
@pytest.mark.parametrize("provider", ["google", "serper"])
async def test_direct_search_hints_providers_outside_web_provider_set(
    monkeypatch, provider
):
    # _WEB_PROVIDERS is {"serpapi"} and means "needs full-page fetch".
    # google and serper are absent from it but must still be hinted.
    seen: dict[str, str] = {}
    _patch_providers(monkeypatch, seen)
    await _run_direct_search(
        "etf fees",
        source_provider=provider,
        search_url="http://x/retrieve",
        top_k=2,
        domain="finance",
    )
    assert seen[provider] == "etf fees finance"


@pytest.mark.asyncio
async def test_direct_search_leaves_the_corpus_query_raw(monkeypatch):
    seen: dict[str, str] = {}
    _patch_providers(monkeypatch, seen)
    await _run_direct_search(
        "etf fees",
        source_provider="retrieval",
        search_url="http://x/retrieve",
        top_k=2,
        domain="finance",
    )
    assert seen["retrieval"] == "etf fees"


@pytest.mark.asyncio
async def test_direct_search_general_domain_is_a_no_op(monkeypatch):
    seen: dict[str, str] = {}
    _patch_providers(monkeypatch, seen)
    await _run_direct_search(
        "etf fees",
        source_provider="serpapi",
        search_url="http://x/retrieve",
        top_k=2,
        domain="general",
    )
    assert seen["serpapi"] == "etf fees"


# --- hybrid search ----------------------------------------------------------


@pytest.mark.asyncio
async def test_hybrid_hints_each_expanded_query_exactly_once(monkeypatch):
    seen: list[str] = []

    async def fake_search_tool(query, *, provider, **_kwargs):
        if provider != "retrieval":
            seen.append(query)
        return [SearchPage(title="t", url="http://x", summary="c")]

    monkeypatch.setattr("src.internal.servers.web.app.search_tool", fake_search_tool)
    monkeypatch.setattr(
        "src.internal.servers.web.app.fetch_pages_concurrently", _passthrough_fetch
    )
    monkeypatch.setattr(
        "src.internal.servers.web.app._expanded_queries",
        lambda query, llm: [query, f"{query} alt"],
    )
    await _run_hybrid_search(
        "etf fees",
        llm=None,
        search_url="http://x/retrieve",
        top_k=2,
        filters=None,
        source_provider="serpapi",
        domain="finance",
    )
    assert seen == ["etf fees finance", "etf fees alt finance"]
    assert all(q.count("finance") == 1 for q in seen)


@pytest.mark.asyncio
async def test_hybrid_corpus_branch_stays_raw(monkeypatch):
    from src.internal.search.process_search_query import SearchQueryResult

    seen: list[str] = []

    async def fake_expanded(query, **_kwargs):
        seen.append(query)
        return SearchQueryResult(executed_queries=[query], results=[])

    monkeypatch.setattr(
        "src.internal.servers.web.app.run_expanded_search", fake_expanded
    )
    await _run_hybrid_search(
        "etf fees",
        llm=None,
        search_url="http://x/retrieve",
        top_k=2,
        filters=None,
        source_provider="retrieval",
        domain="finance",
    )
    assert seen == ["etf fees"]


# --- mode guard -------------------------------------------------------------

# The guard is tested as a helper rather than by driving /api/agent to
# completion: in this environment search_agent and tool_agent already return
# 400 for want of a local model, and chat_once returns 502 for want of a
# retrieval server, so a status-code assertion would report the environment
# rather than the feature.

_NON_HONORING_MODES = ["chat_once", "chat_loop", "search_agent", "tool_agent"]


@pytest.mark.parametrize("mode", _NON_HONORING_MODES)
def test_non_honoring_mode_rejects_a_domain(mode):
    with pytest.raises(HTTPException) as exc:
        _reject_domain_for_mode("finance", mode)
    assert exc.value.status_code == 400
    assert "does not support" in exc.value.detail


@pytest.mark.parametrize("mode", _NON_HONORING_MODES)
def test_non_honoring_mode_accepts_general(mode):
    _reject_domain_for_mode("general", mode)


@pytest.mark.parametrize("mode", ["search_tool", "hybrid_search", None])
def test_honoring_modes_accept_a_domain(mode):
    _reject_domain_for_mode("finance", mode)


def test_mode_aliases_resolve_before_the_guard():
    # "agentic_rag" aliases to chat_loop, which does not honor domains.
    with pytest.raises(HTTPException):
        _reject_domain_for_mode("finance", "agentic_rag")


def test_rejection_happens_over_http_before_any_dispatch(client):
    # chat_once does no retrieval, so a 400 here cannot come from a provider.
    r = client.post(
        "/api/agent", json={"query": "q", "mode": "chat_once", "domain": "finance"}
    )
    assert r.status_code == 400
    assert "does not support" in r.json()["detail"]
