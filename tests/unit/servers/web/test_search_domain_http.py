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
        return SearchQueryResult(
            original_query=query, executed_queries=[query], results=[]
        )

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


# --- threading from the request body to the leaves --------------------------

# The leaf tests above prove the hint is applied correctly; these prove the
# selected domain actually arrives there from an HTTP request body.


def _capture_kwarg(monkeypatch, target: str, captured: dict) -> None:
    async def fake(query, **kwargs):
        captured["query"] = query
        captured["domain"] = kwargs.get("domain")
        return []

    monkeypatch.setattr(f"src.internal.servers.web.app.{target}", fake)


def test_explicit_search_tool_mode_forwards_the_domain(client, monkeypatch):
    captured: dict = {}
    _capture_kwarg(monkeypatch, "_run_direct_search", captured)
    client.post(
        "/api/agent",
        json={"query": "etf fees", "mode": "search_tool", "domain": "finance"},
    )
    assert captured["domain"] == "finance"
    # The route never rewrites the query: only providers see the hint.
    assert captured["query"] == "etf fees"


def test_explicit_hybrid_mode_forwards_the_domain(client, monkeypatch):
    captured: dict = {}

    async def fake_hybrid(query, **kwargs):
        captured["query"] = query
        captured["domain"] = kwargs.get("domain")
        from src.internal.servers.web.app import _HybridSearchResult

        return _HybridSearchResult(
            executed_queries=[query], documents=[], status="empty", ranking={}
        )

    monkeypatch.setattr("src.internal.servers.web.app._run_hybrid_search", fake_hybrid)
    client.post(
        "/api/agent",
        json={"query": "etf fees", "mode": "hybrid_search", "domain": "finance"},
    )
    assert captured["domain"] == "finance"
    assert captured["query"] == "etf fees"


@pytest.mark.asyncio
async def test_auto_pipeline_forwards_the_domain_to_the_hybrid_leaf(monkeypatch):
    from src.internal.servers.web.app import _auto_search_pipeline, _HybridSearchResult

    captured: dict = {}

    async def fake_hybrid(query, **kwargs):
        captured["domain"] = kwargs.get("domain")
        return _HybridSearchResult(
            executed_queries=[query], documents=[], status="empty", ranking={}
        )

    monkeypatch.setattr("src.internal.servers.web.app._run_hybrid_search", fake_hybrid)
    await _auto_search_pipeline(
        "etf fees",
        llm=None,
        search_url="http://x/retrieve",
        browser_search_url=None,
        rerank_url=None,
        top_k=2,
        filters=None,
        history=[],
        source_provider="auto",
        extra={},
        domain="finance",
    )
    assert captured["domain"] == "finance"


def test_persisted_transcript_keeps_the_unhinted_question(client, monkeypatch):
    captured: dict = {}
    _capture_kwarg(monkeypatch, "_run_direct_search", captured)
    r = client.post(
        "/api/agent",
        json={"query": "etf fees", "mode": "search_tool", "domain": "finance"},
    )
    messages = r.json()["messages"]
    user_turns = [m["content"] for m in messages if m["role"] == "user"]
    assert user_turns == ["etf fees"]


@pytest.mark.asyncio
async def test_direct_or_escalate_forwards_the_domain_to_external_fallback(monkeypatch):
    """The direct-first auto path is the default, and it is its own function.

    Threading `domain` into its body without adding the parameter raised a
    NameError that the broad `except Exception` turned into "internal
    unreachable" rather than a visible failure, so this asserts the parameter
    exists and arrives at the external fallback.
    """
    from src.internal.servers.web.app import _run_search_direct_or_escalate

    seen: list[tuple[str, str | None]] = []

    async def fake_direct(_query, **kwargs):
        seen.append((kwargs["source_provider"], kwargs.get("domain")))
        return []

    async def fake_agent(*_a, **_k):
        return ("answer", [], [], "search", {})

    monkeypatch.setattr("src.internal.servers.web.app._run_direct_search", fake_direct)
    monkeypatch.setattr("src.internal.servers.web.app._run_search_agent", fake_agent)
    await _run_search_direct_or_escalate(
        "etf fees",
        manager=object(),
        tokenizer=object(),
        llm=None,
        search_url="http://x/retrieve",
        browser_search_url=None,
        rerank_url=None,
        top_k=5,
        filters=None,
        history=[],
        source_provider="auto",
        domain="finance",
    )
    assert seen, "the direct-first path never dispatched"
    by_provider = dict(seen)
    # The corpus leg is called without a domain; web legs carry it.
    assert by_provider.get("retrieval") in (None, "general")
    assert any(d == "finance" for p, d in seen if p != "retrieval")
