"""Tests for mid-execution fallbacks in the 3-way agentic router.

The recognizer picks a strategy; dispatch is capability-aware. The
retrieval-first fallback chain (hybrid -> RAG -> raw docs -> 502) lives in
`_auto_search_pipeline`, reached when SEARCH has no local model or
CHAT has no LLM. These tests force a strategy at the recognition boundary and assert
the resulting dispatch / fallback behavior.
"""

from __future__ import annotations

import asyncio

import pytest
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, MagicMock

import src.internal.servers.web.app as web_app
from src.context.models import (
    AnswerGenerationResult,
    SearchContextBundle,
    PromptBundle,
    ContextDocument,
)
from src.internal.servers.web.app import SearchExperienceSettings, create_web_app
from src.internal.servers.web.intent import RouteDecision, RouteStrategy
from src.agents.core.base import AgentLoopOutput


def _make_answer_result(answer: str = "ok") -> AnswerGenerationResult:
    return AnswerGenerationResult(
        answer=answer,
        citations=[],
        context=SearchContextBundle(query="q", documents=[]),
        prompt=PromptBundle(system="", user="", messages=[]),
    )


def _force_route(monkeypatch, strategy: RouteStrategy) -> None:
    monkeypatch.setattr(
        "src.internal.servers.web.app.recognize_intent",
        lambda *a, **k: RouteDecision(strategy),
    )


# --- SEARCH without a local model degrades to the hybrid pipeline ---


def test_search_agent_without_model_runs_hybrid_pipeline(monkeypatch, tmp_path):
    """SEARCH + no local model → hybrid pipeline, intent='search'."""
    from src.internal.servers.web.app import _HybridSearchResult

    async def fake_hybrid(query, **kwargs):
        doc = ContextDocument(id="D1", title="t", content="c", url=None, score=0.0)
        return _HybridSearchResult(executed_queries=[query], documents=[doc])

    _force_route(monkeypatch, RouteStrategy.SEARCH)
    monkeypatch.setattr("src.internal.servers.web.app._run_hybrid_search", fake_hybrid)
    app = create_web_app(SearchExperienceSettings(db_path=tmp_path / "db.sqlite3"))
    with TestClient(app) as client:
        response = client.post("/api/agent", json={"query": "find onboarding doc"})
    assert response.status_code == 200
    assert response.json()["intent"] == "search"


# --- Search fallbacks inside _auto_search_pipeline ---


def test_auto_search_does_not_fall_back_to_rag_without_evidence(monkeypatch, tmp_path):
    """Empty providers terminate as SEARCH without calling the LLM/RAG path."""
    _force_route(monkeypatch, RouteStrategy.SEARCH)
    monkeypatch.setattr(
        "src.internal.servers.web.app._run_hybrid_search",
        AsyncMock(side_effect=ConnectionError("retrieval down")),
    )
    rag = AsyncMock(return_value=_make_answer_result("rag fallback"))
    monkeypatch.setattr("src.internal.servers.web.app.answer_with_retrieval", rag)
    app = create_web_app(SearchExperienceSettings(db_path=tmp_path / "db.sqlite3"))
    with TestClient(app) as client:
        response = client.post("/api/agent", json={"query": "find onboarding doc"})
    assert response.status_code == 200
    data = response.json()
    assert data["intent"] == "search"
    rag.assert_not_awaited()


def test_all_search_providers_unreachable_returns_grounded_200(monkeypatch, tmp_path):
    """Provider failures return an unreachable SEARCH response without LLM fallback."""
    _force_route(monkeypatch, RouteStrategy.SEARCH)
    monkeypatch.setattr(
        "src.internal.servers.web.app._run_hybrid_search",
        AsyncMock(side_effect=ConnectionError("down")),
    )
    monkeypatch.setattr(
        "src.internal.servers.web.app.answer_with_retrieval",
        AsyncMock(side_effect=RuntimeError("llm down")),
    )
    monkeypatch.setattr(
        "src.internal.servers.web.app._run_direct_search",
        AsyncMock(side_effect=ConnectionError("still down")),
    )
    app = create_web_app(
        SearchExperienceSettings(db_path=tmp_path / "db.sqlite3"), llm=MagicMock()
    )
    with TestClient(app) as client:
        response = client.post("/api/agent", json={"query": "find doc"})
    assert response.status_code == 200
    assert "No sources are reachable" in response.json()["answer"]


def test_rag_fails_falls_back_to_raw_docs(monkeypatch, tmp_path):
    """hybrid raises, RAG raises, raw search returns docs → 200 with search intent."""
    _force_route(monkeypatch, RouteStrategy.SEARCH)
    monkeypatch.setattr(
        "src.internal.servers.web.app._run_hybrid_search",
        AsyncMock(side_effect=ConnectionError("down")),
    )
    monkeypatch.setattr(
        "src.internal.servers.web.app.answer_with_retrieval",
        AsyncMock(side_effect=RuntimeError("llm down")),
    )
    raw_doc = ContextDocument(
        id="D1", title="Doc", content="content", url=None, score=0.0, metadata={}
    )
    monkeypatch.setattr(
        "src.internal.servers.web.app._run_direct_search",
        AsyncMock(return_value=[raw_doc, raw_doc, raw_doc]),
    )
    app = create_web_app(
        SearchExperienceSettings(db_path=tmp_path / "db.sqlite3"), llm=MagicMock()
    )
    with TestClient(app) as client:
        response = client.post("/api/agent", json={"query": "find doc"})
    assert response.status_code == 200
    data = response.json()
    assert data["intent"] == "search"
    assert len(data["documents"]) == 3


def test_search_unreachable_returns_clear_message(monkeypatch, tmp_path):
    from src.internal.servers.web.app import _HybridSearchResult

    async def fake_hybrid(query, **kwargs):
        return _HybridSearchResult(
            executed_queries=[query], documents=[], status="unreachable"
        )

    _force_route(monkeypatch, RouteStrategy.SEARCH)
    monkeypatch.setattr("src.internal.servers.web.app._run_hybrid_search", fake_hybrid)
    monkeypatch.setattr(
        "src.internal.servers.web.app._run_direct_search",
        AsyncMock(side_effect=ConnectionError("provider down")),
    )
    app = create_web_app(SearchExperienceSettings(db_path=tmp_path / "db.sqlite3"))
    with TestClient(app) as client:
        response = client.post("/api/agent", json={"query": "find stuff"})
    data = response.json()
    assert data["intent"] == "search"
    assert "No sources are reachable" in data["answer"]
    assert data["documents"] == []


def test_search_empty_uses_no_results_message(monkeypatch, tmp_path):
    from src.internal.servers.web.app import _HybridSearchResult

    async def fake_hybrid(query, **kwargs):
        return _HybridSearchResult(
            executed_queries=[query], documents=[], status="empty"
        )

    _force_route(monkeypatch, RouteStrategy.SEARCH)
    monkeypatch.setattr("src.internal.servers.web.app._run_hybrid_search", fake_hybrid)
    monkeypatch.setattr(
        "src.internal.servers.web.app._run_direct_search",
        AsyncMock(return_value=[]),
    )
    app = create_web_app(SearchExperienceSettings(db_path=tmp_path / "db.sqlite3"))
    with TestClient(app) as client:
        response = client.post("/api/agent", json={"query": "find stuff"})
    data = response.json()
    assert data["intent"] == "search"
    assert "No sources are reachable" not in data["answer"]
    assert "no results" in data["answer"].lower()


# --- TOOL degrade ---


def test_tool_loop_failure_degrades_to_agentic_rag(monkeypatch, tmp_path):
    """TOOL with a model but the loop raises → degrade to CHAT."""
    from src.agents.search import AgenticRAGResult

    _force_route(monkeypatch, RouteStrategy.TOOL)
    monkeypatch.setattr(
        "src.agents.tool.tool_calling.ToolAgentLoop.run",
        AsyncMock(side_effect=RuntimeError("OOM")),
    )
    monkeypatch.setattr(
        "src.internal.servers.web.app.AgenticRAGLoop.run",
        AsyncMock(
            return_value=AgenticRAGResult(
                answer="degraded rag answer",
                citations=[],
                context=SearchContextBundle(query="q", documents=[]),
                rounds_used=1,
            )
        ),
    )
    app = create_web_app(
        SearchExperienceSettings(db_path=tmp_path / "db.sqlite3"), llm=MagicMock()
    )
    with TestClient(app) as client:
        app.state.search_agent_manager = MagicMock()
        app.state.search_agent_tokenizer = MagicMock()
        response = client.post("/api/agent", json={"query": "do something"})
    assert response.status_code == 200
    data = response.json()
    assert data["intent"] == "chat"
    assert data["answer"] == "degraded rag answer"


def test_tool_loop_empty_output_degrades(monkeypatch, tmp_path):
    """TOOL loop returns an empty answer → degrade to CHAT."""
    from src.agents.search import AgenticRAGResult

    _force_route(monkeypatch, RouteStrategy.TOOL)
    monkeypatch.setattr(
        "src.agents.tool.tool_calling.ToolAgentLoop.run",
        AsyncMock(
            return_value=AgentLoopOutput(
                prompt_ids=[],
                response_ids=[],
                response_mask=[],
                num_turns=1,
                final_answer=None,
            )
        ),
    )
    monkeypatch.setattr(
        "src.internal.servers.web.app.AgenticRAGLoop.run",
        AsyncMock(
            return_value=AgenticRAGResult(
                answer="fallback rag",
                citations=[],
                context=SearchContextBundle(query="q", documents=[]),
                rounds_used=1,
            )
        ),
    )
    app = create_web_app(
        SearchExperienceSettings(db_path=tmp_path / "db.sqlite3"), llm=MagicMock()
    )
    with TestClient(app) as client:
        app.state.search_agent_manager = MagicMock()
        app.state.search_agent_tokenizer = MagicMock()
        response = client.post("/api/agent", json={"query": "do something"})
    assert response.status_code == 200
    assert response.json()["answer"] == "fallback rag"


# --- Explicit source selection forces a search against that provider ---


def test_explicit_source_forces_search_against_that_provider(monkeypatch, tmp_path):
    """source_provider='serpapi' → recognition returns SEARCH; the chosen
    provider flows through to hybrid search.
    """
    from src.internal.servers.web.app import _HybridSearchResult

    captured: dict = {}

    async def fake_hybrid(query, **kwargs):
        captured["source_provider"] = kwargs.get("source_provider")
        doc = ContextDocument(id="D1", title="t", content="c", url=None, score=0.0)
        return _HybridSearchResult(executed_queries=[query], documents=[doc])

    # No recognition override: explicit_source must drive SEARCH on its own.
    monkeypatch.setattr("src.internal.servers.web.app._run_hybrid_search", fake_hybrid)
    app = create_web_app(SearchExperienceSettings(db_path=tmp_path / "db.sqlite3"))
    with TestClient(app) as client:
        response = client.post(
            "/api/agent",
            json={"query": "explain how FAISS works", "source_provider": "serpapi"},
        )
    assert response.status_code == 200
    assert response.json()["intent"] == "search"
    assert captured["source_provider"] == "serpapi"


def test_auto_provider_expands_to_internal_and_serpapi():
    from src.internal.servers.web.app import _source_providers_for

    assert _source_providers_for("auto") == ["retrieval", "serpapi"]


# --- SEARCH: direct retrieval first, escalate to the agent loop if weak ---


def _doc(score: float, i: int = 1, title: str | None = None) -> ContextDocument:
    return ContextDocument(
        id=f"D{i}",
        title=title or f"doc{i}",
        content="body",
        url=None,
        score=score,
        metadata={},
    )


def _call_direct_or_escalate(
    monkeypatch,
    direct_docs,
    agent_result=None,
    *,
    source_provider="retrieval",
    provider_docs=None,
    browser_search_url=None,
    query="FAISS",
):
    provider_docs = provider_docs or {}
    provider_calls = []

    async def _fake_direct(*a, **k):
        provider = k["source_provider"]
        provider_calls.append(provider)
        return provider_docs.get(provider, direct_docs)

    called = {
        "agent": False,
        "allow_internal_knowledge_answer": None,
        "providers": provider_calls,
    }

    async def _fake_agent(*a, **k):
        called["agent"] = True
        called["allow_internal_knowledge_answer"] = k.get(
            "allow_internal_knowledge_answer"
        )
        return ("agent answer", ["[D1]"], [_doc(0.9)], "search", {})

    monkeypatch.setattr(web_app, "_run_direct_search", _fake_direct)
    monkeypatch.setattr(web_app, "_run_search_agent", _fake_agent)
    result = asyncio.run(
        web_app._run_search_direct_or_escalate(
            query,
            manager=object(),
            tokenizer=object(),
            llm=None,
            search_url="http://x/retrieve",
            browser_search_url=browser_search_url,
            rerank_url=None,
            top_k=5,
            filters=None,
            history=[],
            source_provider=source_provider,
            on_turn=None,
        )
    )
    return result, called


def test_strong_retrieval_returns_direct_without_agent(monkeypatch):
    # Exact title match ("FAISS" == title) → direct, agent loop NOT called.
    (answer, citations, documents, intent, extra), called = _call_direct_or_escalate(
        monkeypatch, [_doc(0.42, title="FAISS"), _doc(0.1, 2)]
    )
    assert called["agent"] is False
    assert extra["search_mode"] == "direct"
    assert extra["source_provider"] == "retrieval"
    assert extra["retrieval_query"] == "FAISS"
    assert extra["ranking"]["operations"] == ["direct_ranking", "sufficiency_gate"]
    assert extra["ranking"]["candidate_count"] == 2
    assert extra["inference"] == {"mode": "deterministic", "model": None}
    assert extra["tier"] == "exact"
    assert documents[0].score == 0.42
    assert intent == "search"


def test_weak_retrieval_escalates_to_agent(monkeypatch):
    # top score 0.1 < 0.2 → escalate; agent loop IS called.
    (answer, citations, documents, intent, extra), called = _call_direct_or_escalate(
        monkeypatch, [_doc(0.1)]
    )
    assert called["agent"] is True
    assert extra["search_mode"] == "escalated"


def test_empty_retrieval_escalates(monkeypatch):
    (_answer, _c, _d, _i, extra), called = _call_direct_or_escalate(monkeypatch, [])
    assert called["agent"] is True
    assert extra["search_mode"] == "escalated"


def test_auto_routed_weak_internal_retrieval_uses_serpapi_before_agent(monkeypatch):
    (_answer, _c, documents, _i, extra), called = _call_direct_or_escalate(
        monkeypatch,
        [],
        source_provider="auto",
        provider_docs={"retrieval": [], "serpapi": [_doc(0.8)]},
        browser_search_url="http://browser/retrieve",
    )

    assert called["providers"] == ["retrieval", "serpapi"]
    assert called["agent"] is False
    assert documents
    assert extra["search_mode"] == "external_fallback"
    assert extra["source_provider"] == "serpapi"
    assert extra["retrieval_query"] == "FAISS"
    assert extra["ranking"] == {
        "operations": ["direct_ranking"],
        "candidate_count": 1,
    }
    assert extra["inference"] == {"mode": "deterministic", "model": None}


def test_auto_routed_empty_serpapi_uses_browser_before_agent(monkeypatch):
    (_answer, _c, documents, _i, extra), called = _call_direct_or_escalate(
        monkeypatch,
        [],
        source_provider="auto",
        provider_docs={
            "retrieval": [],
            "serpapi": [],
            "browser": [_doc(0.7)],
        },
        browser_search_url="http://browser/retrieve",
    )

    assert called["providers"] == ["retrieval", "serpapi", "browser"]
    assert called["agent"] is False
    assert documents
    assert extra["search_mode"] == "external_fallback"
    assert extra["source_provider"] == "browser"
    assert extra["ranking"]["candidate_count"] == 1
    assert extra["inference"] == {"mode": "deterministic", "model": None}


@pytest.mark.parametrize(
    ("ranking", "expected_operations", "expected_status"),
    [
        (
            {
                "operations": ["deduplicate", "external_rerank", "mmr"],
                "candidate_count": 2,
                "rerank_status": "applied",
                "degraded": False,
            },
            ["deduplicate", "external_rerank", "mmr"],
            "applied",
        ),
        (
            {
                "operations": ["deduplicate", "truncate"],
                "candidate_count": 2,
                "rerank_status": "timeout",
                "degraded": True,
            },
            ["deduplicate", "truncate"],
            "timeout",
        ),
    ],
)
def test_external_fallback_retains_actual_ranking_metadata(
    monkeypatch, ranking, expected_operations, expected_status
):
    ranked_docs = web_app._RankedDocumentList([_doc(0.8), _doc(0.7, 2)], ranking)
    (_answer, _c, _documents, _i, extra), _called = _call_direct_or_escalate(
        monkeypatch,
        [],
        source_provider="auto",
        provider_docs={"retrieval": [], "serpapi": ranked_docs},
    )

    assert extra["ranking"]["operations"] == expected_operations
    assert extra["ranking"]["rerank_status"] == expected_status
    assert extra["ranking"]["candidate_count"] == 2


def test_auto_routed_all_providers_empty_returns_no_evidence_without_agent(monkeypatch):
    (answer, citations, documents, intent, extra), called = _call_direct_or_escalate(
        monkeypatch,
        [],
        source_provider="auto",
        provider_docs={"retrieval": [], "serpapi": [], "browser": []},
        browser_search_url="http://browser/retrieve",
        query="GRPO",
    )

    assert called["providers"] == ["retrieval", "serpapi", "browser"]
    assert called["agent"] is False
    assert answer == "No results found for: GRPO"
    assert citations == []
    assert documents == []
    assert intent == "search"
    assert extra["search_mode"] == "external_empty"
