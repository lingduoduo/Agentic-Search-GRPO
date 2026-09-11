from __future__ import annotations

import json

from fastapi.testclient import TestClient

from src.context.models import AnswerGenerationResult
from src.context.models import ChatMessage
from src.context.models import ContextDocument
from src.context.models import PromptBundle
from src.context.models import SearchContextBundle
from src.context.models import SearchFilters
from src.internal.db import AgenticSearchStore
from src.internal.db import UserRecord
from src.internal.hooks import HookConfig
from src.internal.hooks import HookPoint
from src.internal.hooks import HookRegistry
from src.internal.servers.web.app import SearchExperienceSettings
from src.internal.servers.web.app import create_web_app
from src.internal.servers.web.app import _normalize_agent_mode
from src.internal.servers.web.app import _normalize_source_provider
from src.internal.servers.web.app import _source_providers_for


def _answer_result(question: str) -> AnswerGenerationResult:
    context = SearchContextBundle(
        query=question,
        documents=[
            ContextDocument(
                id="D1",
                title="Deployment Guide",
                content="Use the retrieval server before starting the web app.",
                url="https://example.test/deploy",
                score=0.91,
                metadata={"source_type": "docs"},
            )
        ],
    )
    return AnswerGenerationResult(
        answer="[D1] Start retrieval first, then open the web app.",
        citations=["D1"],
        context=context,
        prompt=PromptBundle(system="", user="", messages=[]),
    )


def test_web_app_serves_browser_experience(tmp_path):
    app = create_web_app(SearchExperienceSettings(db_path=tmp_path / "state.sqlite3"))
    client = TestClient(app)

    response = client.get("/")

    assert response.status_code == 200
    assert "Agentic Search" in response.text
    assert "/api/agent" in client.get("/assets/app.js").text
    assert "text/css" in client.get("/assets/app.css").headers["content-type"]


def test_web_demo_all_sources_includes_browser_excludes_google():
    assert _source_providers_for("all") == ["retrieval", "serpapi", "browser"]


def test_web_demo_rejects_disabled_google_provider():
    try:
        _normalize_source_provider("google")
    except Exception as exc:
        assert "source_provider must be one of" in str(exc)
    else:
        raise AssertionError("google provider should be disabled for the web demo")


def test_agent_endpoint_runs_pipeline_and_persists_chat(monkeypatch, tmp_path):
    async def fake_answer_with_retrieval(
        question: str,
        *,
        llm=None,
        chat_history: list[ChatMessage] | None = None,
        search_url: str,
        top_k: int,
        filters=None,
        user_memory=None,
    ) -> AnswerGenerationResult:
        # Anonymous (no auth) now carries ["public"], not "unfiltered" — a
        # document restricted to another user must not leak to a logged-out
        # caller.
        assert filters == SearchFilters(access_acl=["public"])
        assert question == "How do I deploy?"
        assert chat_history == []
        # The client-supplied search_url below is ignored; the server resolves
        # the retrieval URL from its own settings (SSRF protection).
        assert search_url == "http://server.test/retrieve"
        assert top_k == 3
        return _answer_result(question)

    monkeypatch.setattr(
        "src.internal.servers.web.app.answer_with_retrieval",
        fake_answer_with_retrieval,
    )
    store = AgenticSearchStore(tmp_path / "state.sqlite3")
    app = create_web_app(
        SearchExperienceSettings(search_url="http://server.test/retrieve"),
        store=store,
    )
    client = TestClient(app)

    response = client.post(
        "/api/agent",
        json={
            "query": "How do I deploy?",
            "mode": "chat_once",
            "search_url": "http://search.test/retrieve",
            "top_k": 3,
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert data["answer"] == "[D1] Start retrieval first, then open the web app."
    assert data["documents"][0]["title"] == "Deployment Guide"
    assert [message["role"] for message in data["messages"]] == ["user", "assistant"]

    session = client.get(f"/api/sessions/{data['session_id']}").json()
    assert [message["content"] for message in session["messages"]] == [
        "How do I deploy?",
        "[D1] Start retrieval first, then open the web app.",
    ]
    store.close()


def test_agent_endpoint_labels_source_not_unknown(monkeypatch, tmp_path):
    # The classic answer_with_retrieval path returns documents carrying raw
    # retrieval metadata (no "source" key). The endpoint must stamp a provider
    # label so source cards render "Local Retrieval", never "Unknown".
    async def fake_answer_with_retrieval(question, **kwargs) -> AnswerGenerationResult:
        return _answer_result(question)

    monkeypatch.setattr(
        "src.internal.servers.web.app.answer_with_retrieval",
        fake_answer_with_retrieval,
    )
    app = create_web_app(SearchExperienceSettings())
    response = TestClient(app).post(
        "/api/agent",
        json={"query": "How do I deploy?", "mode": "chat_once"},
    )

    assert response.status_code == 200
    document = response.json()["documents"][0]
    assert document["metadata"]["source"] == "Local Retrieval"
    assert document["metadata"]["source_provider"] == "retrieval"


def test_agent_endpoint_persists_pipeline_stage_summary(monkeypatch, tmp_path):
    async def fake_run_auto_routed(query, **kwargs):
        result = _answer_result(query)
        return (
            result.answer,
            result.citations,
            result.context.documents,
            "search",
            {
                "source_provider": "retrieval",
                "retrieval_query": "Deployment context\nHow do I deploy?",
                "ranking": {
                    "operations": ["weighted_rrf", "truncate"],
                    "candidate_count": 4,
                    "rerank_status": "timeout",
                    "degraded": True,
                },
                "inference": {"mode": "grounded", "model": "local-test"},
            },
        )

    monkeypatch.setattr(
        "src.internal.servers.web.app._run_auto_routed", fake_run_auto_routed
    )
    store = AgenticSearchStore(tmp_path / "stages.sqlite3")
    app = create_web_app(SearchExperienceSettings(), store=store)
    response = TestClient(app).post("/api/agent", json={"query": "How do I deploy?"})

    assert response.status_code == 200
    assistant = store.list_chat_messages(response.json()["session_id"])[-1]
    stages = assistant.metadata["pipeline_stages"]
    assert stages == {
        "retrieval": {
            "query": "Deployment context\nHow do I deploy?",
            "provider": "retrieval",
            "candidate_count": 4,
        },
        "ranking": {
            "operations": ["weighted_rrf", "truncate"],
            "evidence_count": 1,
            "reranker": "external",
            "degradation_reason": "timeout",
        },
        "inference": {"mode": "grounded", "model": "local-test"},
        "answer": {"citations": ["D1"], "document_ids": ["D1"]},
    }
    store.close()


def test_agent_endpoint_persists_stage_metrics_apart_from_pipeline_stages(
    monkeypatch, tmp_path
):
    from src.internal.observability import stage_metrics as sm

    async def fake_run_auto_routed(query, **kwargs):
        # Stand in for the choke points: one retrieval, one generation.
        sm.note_retrieval(elapsed_ms=7.0, docs=3)
        sm.note_generation(
            elapsed_ms=90.0, prompt_tokens=120, completion_tokens=15, kind="answer"
        )
        sm.note_generation(elapsed_ms=5.0, prompt_tokens=30, completion_tokens=2)
        result = _answer_result(query)
        return (result.answer, result.citations, result.context.documents, "search", {})

    monkeypatch.setattr(
        "src.internal.servers.web.app._run_auto_routed", fake_run_auto_routed
    )
    stats = sm.StageLatencyStats()
    monkeypatch.setattr("src.internal.servers.web.app.STAGE_LATENCY", stats)
    store = AgenticSearchStore(tmp_path / "stage-metrics.sqlite3")
    app = create_web_app(SearchExperienceSettings(), store=store)
    response = TestClient(app).post("/api/agent", json={"query": "How do I deploy?"})

    assert response.status_code == 200
    assistant = store.list_chat_messages(response.json()["session_id"])[-1]
    assert assistant.metadata["stage_metrics"] == {
        "retrieval": {"calls": 1, "cache_hits": 0, "ms": 7.0, "docs": 3},
        "generation": {
            "calls": 1,
            "ms": 90.0,
            "prompt_tokens": 120,
            "completion_tokens": 15,
        },
        "auxiliary": {
            "calls": 1,
            "ms": 5.0,
            "prompt_tokens": 30,
            "completion_tokens": 2,
        },
    }
    assert "timing" not in assistant.metadata["pipeline_stages"]
    snap = stats.snapshot()
    assert snap["retrieval"]["count"] == 1 and snap["retrieval"]["avg_docs"] == 3.0
    assert snap["generation"]["count"] == 1
    assert snap["generation"]["avg_completion_tokens"] == 15.0
    assert sm.current() is None  # the request scope was closed
    store.close()


def test_agent_endpoint_persists_inference_fallback_stage(monkeypatch, tmp_path):
    async def fake_run_auto_routed(query, **kwargs):
        result = _answer_result(query)
        return (
            result.answer,
            result.citations,
            result.context.documents,
            "search",
            {"inference_fallback": "synthesis_failed"},
        )

    monkeypatch.setattr(
        "src.internal.servers.web.app._run_auto_routed", fake_run_auto_routed
    )
    store = AgenticSearchStore(tmp_path / "fallback.sqlite3")
    app = create_web_app(SearchExperienceSettings(), store=store)
    response = TestClient(app).post("/api/agent", json={"query": "fallback"})

    assistant = store.list_chat_messages(response.json()["session_id"])[-1]
    stages = assistant.metadata["pipeline_stages"]
    assert stages["inference"] == {
        "mode": "deterministic_fallback",
        "model": None,
    }
    assert stages["retrieval"]["provider"] is None
    assert stages["retrieval"]["candidate_count"] is None
    store.close()


def test_agent_endpoint_reuses_existing_session_history(monkeypatch, tmp_path):
    observed_history: list[ChatMessage] = []

    async def fake_answer_with_retrieval(
        question: str,
        *,
        llm=None,
        chat_history: list[ChatMessage] | None = None,
        search_url: str,
        top_k: int,
        filters=None,
        user_memory=None,
    ) -> AnswerGenerationResult:
        del filters
        observed_history.extend(chat_history or [])
        return _answer_result(question)

    monkeypatch.setattr(
        "src.internal.servers.web.app.answer_with_retrieval",
        fake_answer_with_retrieval,
    )
    app = create_web_app(SearchExperienceSettings(db_path=tmp_path / "state.sqlite3"))
    client = TestClient(app)
    session = client.post("/api/sessions", json={"title": "Deployment"}).json()
    client.post(
        "/api/agent",
        json={
            "query": "First question",
            "session_id": session["id"],
            "mode": "chat_once",
        },
    )

    response = client.post(
        "/api/agent",
        json={
            "query": "Follow up",
            "session_id": session["id"],
            "mode": "chat_once",
        },
    )

    assert response.status_code == 200
    assert [message.role for message in observed_history] == ["user", "assistant"]


def test_agent_endpoint_runs_query_processing_hook(monkeypatch, tmp_path):
    class FakeHookResponse:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            return b'{"query": "rewritten deploy question", "metadata": {"source": "hook"}}'

    async def fake_answer_with_retrieval(
        question: str,
        *,
        llm=None,
        chat_history: list[ChatMessage] | None = None,
        search_url: str,
        top_k: int,
        filters=None,
        user_memory=None,
    ) -> AnswerGenerationResult:
        del llm, chat_history, search_url, top_k, filters
        assert question == "rewritten deploy question"
        return _answer_result(question)

    monkeypatch.setattr(
        "src.internal.servers.web.app.answer_with_retrieval",
        fake_answer_with_retrieval,
    )
    monkeypatch.setattr(
        "src.internal.hooks.executor.urllib.request.urlopen",
        lambda request, timeout: FakeHookResponse(),
    )
    registry = HookRegistry(
        [
            HookConfig(
                hook_point=HookPoint.QUERY_PROCESSING,
                endpoint_url="https://hooks.test/query",
            )
        ]
    )
    app = create_web_app(
        SearchExperienceSettings(db_path=tmp_path / "state.sqlite3"),
        hook_registry=registry,
    )
    client = TestClient(app)

    response = client.post("/api/agent", json={"query": "How do I deploy?"})

    assert response.status_code == 200
    session = client.get(f"/api/sessions/{response.json()['session_id']}").json()
    assert session["messages"][0]["content"] == "rewritten deploy question"
    assert registry.execution_log[-1].is_success is True


def test_direct_search_enriches_web_provider_content(monkeypatch):
    """Content fetching is called for serpapi/google providers, not for retrieval."""
    from src.internal.tools.search import SearchPage
    from src.internal.servers.web.app import _run_direct_search
    import asyncio

    serpapi_pages = [
        SearchPage(title="Result A", summary="snippet A", url="https://a.test"),
    ]
    fetched_pages = [
        SearchPage(
            title="Result A", summary="full article content A", url="https://a.test"
        ),
    ]

    async def _fake_search_tool(query, *, provider, search_url, page_size, **_):
        return serpapi_pages

    async def _fake_fetch_pages(pages, *, max_chars, timeout_seconds=10):
        assert pages == serpapi_pages
        return fetched_pages

    monkeypatch.setattr("src.internal.servers.web.app.search_tool", _fake_search_tool)
    monkeypatch.setattr(
        "src.internal.servers.web.app.fetch_pages_concurrently", _fake_fetch_pages
    )

    docs = asyncio.run(
        _run_direct_search(
            "test query",
            source_provider="serpapi",
            search_url="http://localhost:8000/retrieve",
            top_k=3,
        )
    )
    assert any("full article content A" in doc.content for doc in docs)


def test_direct_search_skips_fetch_for_retrieval_provider(monkeypatch):
    """Content fetching is NOT called for the local retrieval provider."""
    from src.internal.tools.search import SearchPage
    from src.internal.servers.web.app import _run_direct_search
    import asyncio

    fetch_called = []

    async def _fake_search_tool(query, *, provider, search_url, page_size, **_):
        return [SearchPage(title="R", summary="corpus content", url="https://r.test")]

    async def _fake_fetch_pages(pages, **kwargs):
        fetch_called.append(True)
        return pages

    monkeypatch.setattr("src.internal.servers.web.app.search_tool", _fake_search_tool)
    monkeypatch.setattr(
        "src.internal.servers.web.app.fetch_pages_concurrently", _fake_fetch_pages
    )

    asyncio.run(
        _run_direct_search(
            "test query",
            source_provider="retrieval",
            search_url="http://localhost:8000/retrieve",
            top_k=3,
        )
    )
    assert not fetch_called


def test_hybrid_search_enriches_serpapi_provider_content(monkeypatch):
    """Hybrid search fetches full page content for serpapi results."""
    from src.internal.tools.search import SearchPage
    from src.internal.servers.web.app import _run_hybrid_search
    import asyncio

    pages = [SearchPage(title="T", summary="snippet", url="https://t.test")]
    fetched = [SearchPage(title="T", summary="full article body", url="https://t.test")]

    async def _fake_search_tool(query, *, provider, search_url, page_size, **kw):
        return pages

    async def _fake_fetch_pages(pgs, *, max_chars, timeout_seconds=10):
        return fetched

    monkeypatch.setattr("src.internal.servers.web.app.search_tool", _fake_search_tool)
    monkeypatch.setattr(
        "src.internal.servers.web.app.fetch_pages_concurrently", _fake_fetch_pages
    )
    monkeypatch.setattr(
        "src.internal.servers.web.app.expand_keywords",
        lambda query, llm: [],
    )

    result = asyncio.run(
        _run_hybrid_search(
            "latest AI news",
            llm=None,
            search_url="http://localhost:8000/retrieve",
            top_k=3,
            filters=None,
            source_provider="serpapi",
        )
    )
    assert any("full article body" in doc.content for doc in result.documents)


def test_hybrid_search_includes_temporal_variant_for_time_sensitive_query(monkeypatch):
    """Temporal variant is added to executed queries for time-sensitive queries."""
    from src.internal.servers.web.app import _run_hybrid_search
    from src.internal.tools.search import SearchPage
    import asyncio

    executed: list[str] = []

    async def _fake_search_tool(query, *, provider, search_url, page_size, **kw):
        executed.append(query)
        return [SearchPage(title="T", summary="s", url="https://t.test")]

    async def _fake_fetch_pages(pages, **kwargs):
        return pages

    monkeypatch.setattr("src.internal.servers.web.app.search_tool", _fake_search_tool)
    monkeypatch.setattr(
        "src.internal.servers.web.app.fetch_pages_concurrently", _fake_fetch_pages
    )
    monkeypatch.setattr(
        "src.internal.servers.web.app.expand_keywords",
        lambda query, llm: [],
    )

    result = asyncio.run(
        _run_hybrid_search(
            "latest AI models",
            llm=None,
            search_url="http://localhost:8000/retrieve",
            top_k=3,
            filters=None,
            source_provider="serpapi",
        )
    )
    from datetime import datetime

    year = str(datetime.now().year)
    assert any(year in q for q in result.executed_queries)


def test_hybrid_search_runs_search_tool_calls_concurrently(monkeypatch):
    """All search tool calls for expanded queries run concurrently (asyncio.gather)."""
    from src.internal.servers.web.app import _run_hybrid_search
    from src.internal.tools.search import SearchPage
    import asyncio

    call_count = []

    async def _fake_search_tool(query, *, provider, search_url, page_size, **kw):
        call_count.append(query)
        return [SearchPage(title="T", summary="s", url="https://t.test")]

    async def _fake_fetch_pages(pages, **kwargs):
        return pages

    monkeypatch.setattr("src.internal.servers.web.app.search_tool", _fake_search_tool)
    monkeypatch.setattr(
        "src.internal.servers.web.app.fetch_pages_concurrently", _fake_fetch_pages
    )
    monkeypatch.setattr(
        "src.internal.servers.web.app.expand_keywords",
        lambda query, llm: ["AI news expanded"],
    )

    result = asyncio.run(
        _run_hybrid_search(
            "latest AI news",
            llm=object(),  # non-None so expand_keywords is called
            search_url="http://localhost:8000/retrieve",
            top_k=3,
            filters=None,
            source_provider="serpapi",
        )
    )
    # 2 queries: original + 1 expansion (temporal variant added too = 3 total)
    assert len(call_count) >= 2
    assert result.executed_queries is not None


def test_search_agent_mode_is_valid():
    assert _normalize_agent_mode("search_agent") == "search_agent"


def test_search_agent_returns_400_when_not_configured(tmp_path):
    app = create_web_app(SearchExperienceSettings(db_path=tmp_path / "state.sqlite3"))
    client = TestClient(app)
    response = client.post(
        "/api/agent",
        json={"query": "What is FAISS?", "mode": "search_agent"},
    )
    assert response.status_code == 400
    assert "SEARCH_AGENT_MODEL" in response.json()["detail"]


def test_tool_agent_mode_is_valid():
    assert _normalize_agent_mode("tool_agent") == "tool_agent"


def test_tool_agent_returns_400_when_not_configured(tmp_path):
    app = create_web_app(SearchExperienceSettings(db_path=tmp_path / "state.sqlite3"))
    client = TestClient(app)
    response = client.post(
        "/api/agent",
        json={"query": "What is FAISS?", "mode": "tool_agent"},
    )
    assert response.status_code == 400
    assert "SEARCH_AGENT_MODEL" in response.json()["detail"]


def test_web_registry_modes_map_to_classes():
    from src import get_registered_agent_loop, resolve_agent_name
    from src.agents.search import SearchAgentLoop
    from src.agents.tool import ToolAgentLoop

    assert (
        get_registered_agent_loop(resolve_agent_name("search_agent")) is SearchAgentLoop
    )
    assert get_registered_agent_loop(resolve_agent_name("tool_agent")) is ToolAgentLoop


def test_run_agent_search_tool_mode_returns_documents(monkeypatch, tmp_path):
    from src.context.models import ContextDocument

    docs = [
        ContextDocument(
            id="D1",
            title="FAISS Guide",
            content="FAISS is a similarity search library.",
            url="https://example.test/faiss",
            score=0.95,
            # Real `_run_direct_search` documents always carry the metadata
            # `_documents_from_search_pages` attaches (source, provider, the
            # document's own acl). An empty dict is not a shape this route can
            # produce, and `SearchFilters.matches` reads it as "nothing matched"
            # rather than "unrestricted", so the route's access check drops it.
            metadata={"source_provider": "retrieval", "entry_point": "search_tool"},
        )
    ]

    async def fake_run_direct_search(
        query,
        *,
        source_provider,
        search_url,
        top_k,
        browser_search_url=None,
        rerank_url=None,
        filters=None,
    ):
        return docs

    monkeypatch.setattr(
        "src.internal.servers.web.app._run_direct_search",
        fake_run_direct_search,
    )
    app = create_web_app(SearchExperienceSettings(db_path=tmp_path / "state.sqlite3"))
    client = TestClient(app)

    response = client.post(
        "/api/agent",
        json={
            "query": "What is FAISS?",
            "mode": "search_tool",
            "source_provider": "retrieval",
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert len(data["documents"]) == 1
    assert data["documents"][0]["title"] == "FAISS Guide"
    stages = data["messages"][-1]["metadata"]["pipeline_stages"]
    assert stages["retrieval"] == {
        "query": "What is FAISS?",
        "provider": "retrieval",
        "candidate_count": 1,
    }
    assert stages["ranking"]["operations"] == [
        "direct_ranking",
    ]
    assert stages["inference"] == {"mode": "deterministic", "model": None}


def test_hybrid_mode_persists_truthful_direct_stage_metadata(monkeypatch, tmp_path):
    from src.internal.servers.web.app import _HybridSearchResult

    docs = _answer_result("q").context.documents

    async def fake_hybrid(*args, **kwargs):
        return _HybridSearchResult(
            executed_queries=["q", "q expanded"],
            documents=docs,
            status="ok",
            ranking={
                "operations": ["deduplicate", "external_rerank", "mmr"],
                "candidate_count": 2,
                "rerank_status": "applied",
            },
        )

    monkeypatch.setattr("src.internal.servers.web.app._run_hybrid_search", fake_hybrid)
    app = create_web_app(SearchExperienceSettings(db_path=tmp_path / "hybrid.sqlite3"))
    data = (
        TestClient(app)
        .post(
            "/api/agent",
            json={"query": "q", "mode": "hybrid_search", "source_provider": "auto"},
        )
        .json()
    )

    stages = data["messages"][-1]["metadata"]["pipeline_stages"]
    assert stages["retrieval"] == {
        "query": "q",
        "provider": "auto",
        "candidate_count": 2,
    }
    assert stages["ranking"]["operations"] == [
        "deduplicate",
        "external_rerank",
        "mmr",
    ]
    assert stages["ranking"]["reranker"] == "external"
    assert stages["inference"] == {"mode": "deterministic", "model": None}


def test_run_agent_chat_once_mode_returns_answer(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "src.internal.servers.web.app.answer_with_retrieval",
        lambda *a, **kw: __import__("asyncio").coroutine(lambda: _answer_result("q"))(),
    )

    async def fake_answer(
        question,
        *,
        llm=None,
        chat_history=None,
        search_url,
        top_k,
        filters=None,
        user_memory=None,
    ):
        return _answer_result(question)

    monkeypatch.setattr(
        "src.internal.servers.web.app.answer_with_retrieval", fake_answer
    )
    app = create_web_app(SearchExperienceSettings(db_path=tmp_path / "state.sqlite3"))
    client = TestClient(app)

    response = client.post(
        "/api/agent",
        json={"query": "How do I deploy?", "mode": "chat_once"},
    )

    assert response.status_code == 200
    data = response.json()
    assert "Start retrieval first" in data["answer"]


def test_run_agent_chat_once_citations_extracted(monkeypatch, tmp_path):
    """citations in the response match the [Dx] markers extracted from the answer."""

    async def fake_answer(question, *, llm=None, chat_history=None, **kw):
        return _answer_result(question)  # answer contains "[D1]", citations=["D1"]

    monkeypatch.setattr(
        "src.internal.servers.web.app.answer_with_retrieval", fake_answer
    )
    app = create_web_app(SearchExperienceSettings(db_path=tmp_path / "state.sqlite3"))
    client = TestClient(app)

    response = client.post(
        "/api/agent", json={"query": "What is FAISS?", "mode": "chat_once"}
    )

    assert response.status_code == 200
    data = response.json()
    assert data["citations"] == ["D1"]
    assert "[D1]" in data["answer"]


def test_run_agent_trims_long_history(monkeypatch, tmp_path):
    """When session history exceeds MAX_HISTORY_MESSAGES only the tail reaches the LLM."""
    from src.internal.servers.web.app import MAX_HISTORY_MESSAGES

    captured: list = []

    async def fake_answer(question, *, llm=None, chat_history=None, **kw):
        captured.append(list(chat_history or []))
        return _answer_result(question)

    monkeypatch.setattr(
        "src.internal.servers.web.app.answer_with_retrieval", fake_answer
    )

    store = AgenticSearchStore(tmp_path / "state.sqlite3")
    session = store.create_chat_session(title="long")
    for i in range(60):
        role = "user" if i % 2 == 0 else "assistant"
        store.add_chat_message(session.id, role=role, content=f"msg {i}")

    app = create_web_app(
        SearchExperienceSettings(db_path=tmp_path / "state.sqlite3"), store=store
    )
    client = TestClient(app)
    client.post(
        "/api/agent",
        json={"query": "follow up", "mode": "chat_once", "session_id": session.id},
    )

    assert len(captured) == 1
    assert len(captured[0]) <= MAX_HISTORY_MESSAGES


def test_agent_endpoint_returns_intent_field(monkeypatch, tmp_path):
    async def fake_answer(*args, **kwargs):
        return _answer_result("q")

    monkeypatch.setattr(
        "src.internal.servers.web.app.answer_with_retrieval", fake_answer
    )
    app = create_web_app(SearchExperienceSettings(db_path=tmp_path / "db.sqlite3"))
    client = TestClient(app)
    response = client.post(
        "/api/agent", json={"query": "explain FAISS", "mode": "chat_once"}
    )
    assert response.status_code == 200
    data = response.json()
    assert "intent" in data
    assert data["intent"] in ("search", "chat", "tool")


def test_auto_route_agentic_rag_for_chat(monkeypatch, tmp_path):
    """CHAT route → AgenticRAGLoop (decompose + HyDE), intent='chat'."""
    from unittest.mock import AsyncMock, MagicMock
    from src.agents.search import AgenticRAGResult
    from src.context.models import SearchContextBundle
    from src.internal.servers.web.intent import RouteDecision, RouteStrategy

    monkeypatch.setattr(
        "src.internal.servers.web.app.recognize_intent",
        lambda *a, **k: RouteDecision(RouteStrategy.CHAT),
    )
    fake_result = AgenticRAGResult(
        answer="Grounded answer [D1]",
        citations=[],
        context=SearchContextBundle(query="explain FAISS", documents=[]),
        rounds_used=2,
    )
    monkeypatch.setattr(
        "src.internal.servers.web.app.AgenticRAGLoop.run",
        AsyncMock(return_value=fake_result),
    )
    app = create_web_app(
        SearchExperienceSettings(db_path=tmp_path / "db.sqlite3"), llm=MagicMock()
    )
    client = TestClient(app)
    response = client.post("/api/agent", json={"query": "explain FAISS"})
    assert response.status_code == 200
    data = response.json()
    assert data["intent"] == "chat"
    assert data["answer"] == "Grounded answer [D1]"


def test_auto_route_search_uses_direct_provider_order_without_local_model(
    monkeypatch, tmp_path
):
    """SEARCH route with no local model tries internal then SerpAPI directly."""
    called = []

    async def fake_direct(query, *, source_provider, **kwargs):
        called.append(source_provider)
        return []

    from src.internal.servers.web.intent import RouteDecision, RouteStrategy

    monkeypatch.setattr(
        "src.internal.servers.web.app.recognize_intent",
        lambda *a, **k: RouteDecision(RouteStrategy.SEARCH),
    )
    monkeypatch.setattr("src.internal.servers.web.app._run_direct_search", fake_direct)
    app = create_web_app(SearchExperienceSettings(db_path=tmp_path / "db.sqlite3"))
    client = TestClient(app)
    response = client.post("/api/agent", json={"query": "find the onboarding doc"})
    assert response.status_code == 200
    assert called == ["retrieval", "serpapi"]
    data = response.json()
    assert data["intent"] == "search"


def test_explicit_mode_still_works(monkeypatch, tmp_path):
    """Passing explicit mode='chat_once' still routes to answer_with_retrieval."""
    called = {}

    async def fake_answer(
        q, *, llm, chat_history, search_url, top_k, filters, user_memory=None
    ):
        called["answer"] = True
        return _answer_result(q)

    monkeypatch.setattr(
        "src.internal.servers.web.app.answer_with_retrieval", fake_answer
    )
    app = create_web_app(SearchExperienceSettings(db_path=tmp_path / "db.sqlite3"))
    client = TestClient(app)
    response = client.post("/api/agent", json={"query": "hello", "mode": "chat_once"})
    assert response.status_code == 200
    assert called.get("answer") is True
    assert response.json()["intent"] == "chat"


def test_auto_route_tool_agent_runs_tool_loop_when_model_available(
    monkeypatch, tmp_path
):
    """TOOL route with a local model → ToolAgentLoop runs with real tools."""
    from unittest.mock import AsyncMock, MagicMock
    from src.agents.core.base import AgentLoopOutput
    from src.internal.servers.web.intent import RouteDecision, RouteStrategy
    import json

    monkeypatch.setattr(
        "src.internal.servers.web.app.recognize_intent",
        lambda *a, **k: RouteDecision(RouteStrategy.TOOL),
    )
    # A trace that says the corpus search tool was called
    fake_trace = json.dumps(
        {"tool_name": "search", "status": "completed", "result": "[]"}
    )
    fake_output = AgentLoopOutput(
        prompt_ids=[],
        response_ids=[],
        response_mask=[],
        num_turns=1,
        action_trace=fake_trace,
        final_answer="Here are the results.",
    )
    run_mock = AsyncMock(return_value=fake_output)
    monkeypatch.setattr(
        "src.agents.tool.tool_calling.ToolAgentLoop.run",
        run_mock,
    )
    app = create_web_app(SearchExperienceSettings(db_path=tmp_path / "db.sqlite3"))
    with TestClient(app) as client:
        app.state.search_agent_manager = MagicMock()
        app.state.search_agent_tokenizer = MagicMock()
        response = client.post("/api/agent", json={"query": "find the onboarding doc"})
    assert response.status_code == 200
    data = response.json()
    assert data["intent"] == "search"
    assert data["answer"] == "Here are the results."
    assert run_mock.await_args.kwargs["on_approval"] is None


def test_agent_no_llm_chat_degrades_to_pipeline(monkeypatch, tmp_path):
    """No LLM + CHAT route → _auto_search_pipeline (grounded degradation), not a 400.

    Two real-world leak paths are closed so ``llm`` is genuinely None: (1) inject
    empty app_settings so the config loader's GEN_AI key can't set
    ``resolved.llm.api_key``; (2) set OPENAI_API_KEY="" — python-dotenv won't
    override an already-present var, so create_web_app's internal .env reload
    can't repopulate it (delenv alone is insufficient — the reload re-adds it).
    """
    from src.internal.configs import AppSettings
    from src.internal.servers.web.intent import RouteDecision, RouteStrategy

    monkeypatch.setattr("src.internal.servers.web.app.load_dotenv", lambda: None)
    monkeypatch.setenv("OPENAI_API_KEY", "")
    monkeypatch.setattr(
        "src.internal.servers.web.app.recognize_intent",
        lambda *a, **k: RouteDecision(RouteStrategy.CHAT),
    )

    async def fake_pipeline(query, **kw):
        extra = kw.get("extra", {})
        return "extractive answer", ["[D1]"], [], "chat", extra

    monkeypatch.setattr(
        "src.internal.servers.web.app._auto_search_pipeline", fake_pipeline
    )
    app = create_web_app(
        SearchExperienceSettings(db_path=tmp_path / "db.sqlite3"),
        app_settings=AppSettings(),
    )
    client = TestClient(app)
    response = client.post("/api/agent", json={"query": "explain FAISS"})
    assert response.status_code == 200
    body = response.json()
    assert body["intent"] == "chat"
    assert body["answer"] == "extractive answer"


def test_uncertain_query_asks_instead_of_running_an_agent(monkeypatch, tmp_path):
    from src.internal.configs import AppSettings

    def unexpected(*args, **kwargs):
        raise AssertionError("a clarification must not run an agent")

    monkeypatch.setattr("src.internal.servers.web.app._run_agentic_rag", unexpected)
    monkeypatch.setattr(
        "src.internal.servers.web.app._run_search_direct_or_escalate", unexpected
    )
    monkeypatch.setattr("src.internal.servers.web.app._run_tool_agent", unexpected)

    monkeypatch.setattr("src.internal.servers.web.app.load_dotenv", lambda: None)
    monkeypatch.setenv("OPENAI_API_KEY", "")
    app = create_web_app(
        SearchExperienceSettings(db_path=tmp_path / "clarify.sqlite3"),
        app_settings=AppSettings(),
    )
    with TestClient(app) as client:
        response = client.post(
            "/api/agent", json={"query": "Review the vendor renewal terms"}
        )

    assert response.status_code == 200
    body = response.json()
    assert body["intent"] == "clarify"
    assert [option["route"] for option in body["clarification"]["options"]] == [
        "chat",
        "search",
        "tool",
    ]
    assert body["documents"] == []
    assert body["hook_metadata"]["route_mechanism"] == "clarify"


def test_selected_route_dispatches_to_the_matching_runner(monkeypatch, tmp_path):
    calls = []

    async def fake_search(query, **kwargs):
        calls.append(("search", query))
        return "search answer", [], [], "search", {}

    monkeypatch.setattr(
        "src.internal.servers.web.app._run_search_direct_or_escalate", fake_search
    )
    app = create_web_app(SearchExperienceSettings(db_path=tmp_path / "picked.sqlite3"))
    with TestClient(app) as client:
        response = client.post(
            "/api/agent",
            json={"query": "Review the vendor renewal terms", "route": "search"},
        )

    assert response.status_code == 200
    assert calls == [("search", "Review the vendor renewal terms")]
    assert response.json()["hook_metadata"]["route_mechanism"] == "user_selected"


def test_unknown_route_value_is_rejected(tmp_path):
    app = create_web_app(SearchExperienceSettings(db_path=tmp_path / "bad.sqlite3"))
    with TestClient(app) as client:
        response = client.post(
            "/api/agent", json={"query": "anything", "route": "teleport"}
        )

    assert response.status_code == 422


def test_explicit_mode_never_clarifies(monkeypatch, tmp_path):
    called = {}

    async def fake_answer(
        q, *, llm, chat_history, search_url, top_k, filters, user_memory=None
    ):
        called["answer"] = True
        return _answer_result(q)

    monkeypatch.setattr(
        "src.internal.servers.web.app.answer_with_retrieval", fake_answer
    )
    app = create_web_app(
        SearchExperienceSettings(db_path=tmp_path / "explicit.sqlite3")
    )
    with TestClient(app) as client:
        response = client.post(
            "/api/agent",
            json={"query": "Review the vendor renewal terms", "mode": "chat_once"},
        )

    assert response.status_code == 200
    assert called.get("answer") is True
    assert response.json()["intent"] == "chat"
    assert response.json()["clarification"] is None


def test_streaming_done_event_carries_the_clarification(monkeypatch, tmp_path):
    from src.internal.configs import AppSettings

    monkeypatch.setattr("src.internal.servers.web.app.load_dotenv", lambda: None)
    monkeypatch.setenv("OPENAI_API_KEY", "")
    app = create_web_app(
        SearchExperienceSettings(db_path=tmp_path / "stream.sqlite3"),
        app_settings=AppSettings(),
    )
    with TestClient(app) as client:
        response = client.post(
            "/api/agent/stream", json={"query": "Review the vendor renewal terms"}
        )

    assert response.status_code == 200
    done = [
        json.loads(line[len("data: ") :])
        for line in response.text.splitlines()
        if line.startswith("data: ")
    ][-1]
    assert done["type"] == "done"
    assert done["intent"] == "clarify"
    assert done["clarification"]["options"][1]["route"] == "search"


def test_agent_tool_mode_without_model_returns_clear_400(tmp_path):
    """Explicit mode=tool_agent without local model → 400 with 'local model' in detail."""
    app = create_web_app(SearchExperienceSettings(db_path=tmp_path / "db.sqlite3"))
    client = TestClient(app)
    response = client.post(
        "/api/agent", json={"query": "run tool", "mode": "tool_agent"}
    )
    assert response.status_code == 400
    assert "local model" in response.json()["detail"].lower()


def test_tool_approval_decision_requires_authentication(tmp_path):
    app = create_web_app(SearchExperienceSettings(db_path=tmp_path / "db.sqlite3"))
    client = TestClient(app)

    response = client.post("/api/agent/approvals/missing", json={"decision": "approve"})

    assert response.status_code == 401


def test_tool_approval_decision_returns_not_found_for_unknown_id(tmp_path):
    from src.internal.auth import generate_user_jwt_token

    store = AgenticSearchStore(tmp_path / "db.sqlite3")
    store.upsert_user(UserRecord(id="user-1", email="user-1@example.test"))
    app = create_web_app(
        SearchExperienceSettings(db_path=tmp_path / "db.sqlite3"), store=store
    )
    client = TestClient(app)
    token = generate_user_jwt_token(user_id="user-1")

    response = client.post(
        "/api/agent/approvals/missing",
        json={"decision": "deny"},
        headers={"Authorization": f"Bearer {token}"},
    )

    assert response.status_code == 404


def test_agent_other_exception_returns_502_with_message(monkeypatch, tmp_path):
    """Unexpected exception → 502 with the exception message, not 'Agent search failed'."""

    async def explode(*args, **kwargs):
        raise ValueError("bad input format")

    monkeypatch.setattr("src.internal.servers.web.app.answer_with_retrieval", explode)
    app = create_web_app(SearchExperienceSettings(db_path=tmp_path / "db.sqlite3"))
    client = TestClient(app)
    response = client.post(
        "/api/agent", json={"query": "explain FAISS", "mode": "chat_once"}
    )
    assert response.status_code == 502
    assert "bad input format" in response.json()["detail"]
    assert "Agent search failed" not in response.json()["detail"]


def test_hybrid_fanout_merges_real_and_drops_errored_provider(monkeypatch, tmp_path):
    """retrieval returns real pages, serpapi errors → only real docs, status ok."""
    import asyncio
    from src.internal.servers.web.app import _run_hybrid_search
    from src.internal.tools import SearchPage

    async def fake_search_tool(query, *, provider, search_url, page_size, **kw):
        if provider == "retrieval":
            return [SearchPage(title="Real Doc", summary="real", url="http://x/1")]
        return [SearchPage(error="SERPAPI_API_KEY is required.")]

    monkeypatch.setattr("src.internal.servers.web.app.search_tool", fake_search_tool)
    monkeypatch.setattr(
        "src.internal.servers.web.app._expanded_queries", lambda q, llm: [q]
    )
    result = asyncio.run(
        _run_hybrid_search(
            "q",
            llm=None,
            search_url="http://localhost:8001/retrieve",
            top_k=3,
            filters=None,
            source_provider="auto",
        )
    )
    assert result.status == "ok"
    assert [d.title for d in result.documents] == ["Real Doc"]
    assert all(not d.metadata.get("error") for d in result.documents)


def test_hybrid_fanout_all_errored_is_unreachable(monkeypatch, tmp_path):
    import asyncio
    from src.internal.servers.web.app import _run_hybrid_search
    from src.internal.tools import SearchPage

    async def fake_search_tool(query, *, provider, search_url, page_size, **kw):
        return [SearchPage(error="down")]

    monkeypatch.setattr("src.internal.servers.web.app.search_tool", fake_search_tool)
    monkeypatch.setattr(
        "src.internal.servers.web.app._expanded_queries", lambda q, llm: [q]
    )
    result = asyncio.run(
        _run_hybrid_search(
            "q",
            llm=None,
            search_url="http://x/retrieve",
            top_k=3,
            filters=None,
            source_provider="auto",
        )
    )
    assert result.status == "unreachable"
    assert result.documents == []


def test_hybrid_fanout_reachable_but_empty_is_empty(monkeypatch, tmp_path):
    import asyncio
    from src.internal.servers.web.app import _run_hybrid_search

    async def fake_search_tool(query, *, provider, search_url, page_size, **kw):
        return []  # reachable, no hits, no error

    monkeypatch.setattr("src.internal.servers.web.app.search_tool", fake_search_tool)
    monkeypatch.setattr(
        "src.internal.servers.web.app._expanded_queries", lambda q, llm: [q]
    )
    result = asyncio.run(
        _run_hybrid_search(
            "q",
            llm=None,
            search_url="http://x/retrieve",
            top_k=3,
            filters=None,
            source_provider="auto",
        )
    )
    assert result.status == "empty"
    assert result.documents == []


def test_hybrid_fanout_one_provider_raises_does_not_kill_other(monkeypatch, tmp_path):
    import asyncio
    from src.internal.servers.web.app import _run_hybrid_search
    from src.internal.tools import SearchPage

    async def fake_search_tool(query, *, provider, search_url, page_size, **kw):
        if provider == "serpapi":
            raise RuntimeError("boom")
        return [SearchPage(title="Real", summary="r", url="http://x/1")]

    monkeypatch.setattr("src.internal.servers.web.app.search_tool", fake_search_tool)
    monkeypatch.setattr(
        "src.internal.servers.web.app._expanded_queries", lambda q, llm: [q]
    )
    result = asyncio.run(
        _run_hybrid_search(
            "q",
            llm=None,
            search_url="http://x/retrieve",
            top_k=3,
            filters=None,
            source_provider="auto",
        )
    )
    assert result.status == "ok"
    assert [d.title for d in result.documents] == ["Real"]


def test_hybrid_auto_applies_filters_only_to_internal_retrieval(monkeypatch):
    import asyncio

    from src.internal.servers.web.app import _run_hybrid_search
    from src.internal.tools import SearchPage

    provider_filters = {}
    browser_calls = []

    async def fake_search_tool(query, *, provider, search_url, page_size, **kwargs):
        provider_filters[provider] = kwargs.get("filters", "not-passed")
        if provider == "retrieval":
            return [SearchPage(title="Private", summary="allowed", url="http://r/1")]
        return []

    async def fake_browser(query, *, browser_search_url, top_k, existing_count):
        browser_calls.append(query)
        return []

    monkeypatch.setattr("src.internal.servers.web.app.search_tool", fake_search_tool)
    monkeypatch.setattr(
        "src.internal.servers.web.app._run_browser_search", fake_browser
    )
    monkeypatch.setattr(
        "src.internal.servers.web.app._expanded_queries", lambda q, llm: [q]
    )
    filters = {"access": {"user_id": "u1"}}

    asyncio.run(
        _run_hybrid_search(
            "q",
            llm=None,
            search_url="http://x/retrieve",
            browser_search_url="http://browser/retrieve",
            top_k=3,
            filters=filters,
            source_provider="auto",
        )
    )

    assert provider_filters == {"retrieval": filters, "serpapi": "not-passed"}
    assert browser_calls == ["q"]


def test_direct_search_auto_excludes_browser_sidecar(monkeypatch):
    """source_provider='auto' must NOT pull the slow browser sidecar, while
    'all'/'retrieval' still do (regression for the browser-out-of-auto invariant)."""
    import asyncio
    from src.internal.tools.search import SearchPage
    from src.internal.servers.web.app import _run_direct_search

    browser_calls: list[str] = []

    async def _fake_search_tool(query, *, provider, search_url, page_size, **_):
        return [
            SearchPage(title=f"{provider} R", summary="c", url=f"http://{provider}/1")
        ]

    async def _fake_browser(query, *, browser_search_url, top_k, existing_count):
        browser_calls.append(query)
        return []

    monkeypatch.setattr("src.internal.servers.web.app.search_tool", _fake_search_tool)
    monkeypatch.setattr(
        "src.internal.servers.web.app._run_browser_search", _fake_browser
    )

    asyncio.run(
        _run_direct_search(
            "q",
            source_provider="auto",
            search_url="http://localhost:8001/retrieve",
            browser_search_url="http://localhost:9999/retrieve",
            top_k=3,
        )
    )
    assert browser_calls == []  # auto never triggers browser

    asyncio.run(
        _run_direct_search(
            "q",
            source_provider="retrieval",
            search_url="http://localhost:8001/retrieve",
            browser_search_url="http://localhost:9999/retrieve",
            top_k=3,
        )
    )
    assert browser_calls == ["q"]  # non-auto still gets the sidecar


async def test_run_agentic_rag_populates_control_flow_trace():
    """chat_loop runs now emit a control-flow trace (F3) → renders in the F6 waterfall."""
    from unittest.mock import MagicMock, patch

    from src.internal.servers.web.app import _run_agentic_rag

    bundle = SearchContextBundle(
        query="q",
        documents=[
            ContextDocument(id="d1", title="T", content="c about faiss", score=0.9)
        ],
    )
    llm = MagicMock()
    llm.complete.side_effect = [
        "sub-q",
        "hyde text",
        "broader",
        "yes",
        (
            '{"claims":[{"text":"faiss","evidence_ids":["D1"]}],'
            '"missing_information":[],"abstain":false}'
        ),
    ]

    async def _fake_retrieve(queries, **kwargs):
        return [bundle for _ in queries]

    with patch("src.agents.search.agentic_rag.retrieve_contexts", _fake_retrieve):
        _, _, _, intent, extra = await _run_agentic_rag(
            "what is faiss",
            llm=llm,
            search_url="http://x/retrieve",
            top_k=5,
            history=[],
        )

    assert intent == "chat"
    trace = extra["control_flow_trace"]
    assert len(trace) >= 2
    components = [e.component for e in trace]
    assert "query_enhancer" in components
    assert "answer_generator" in components


async def test_agentic_rag_retrieval_propagates_access_filters(monkeypatch):
    from src.agents.search.agentic_rag import AgenticRAGConfig, AgenticRAGLoop

    filters = SearchFilters(access_acl=["user:alice"])
    observed = []

    async def fake_retrieve(queries, **kwargs):
        observed.append(kwargs.get("filters"))
        return [SearchContextBundle(query=q, documents=[]) for q in queries]

    monkeypatch.setattr(
        "src.agents.search.agentic_rag.retrieve_contexts", fake_retrieve
    )
    llm = __import__("unittest.mock").mock.MagicMock()
    llm.complete.side_effect = ["sub-q", "hyde", "broader", "yes", "answer"]
    loop = AgenticRAGLoop(
        AgenticRAGConfig(max_rounds=1, filters=filters),
        llm=llm,
    )

    await loop.run("question")

    assert observed
    assert all(item is filters for item in observed)


def test_generative_query_routes_to_chat_and_dispatches(monkeypatch, tmp_path):
    """A generative ask (former direct_llm) now routes to CHAT → grounded path,
    and dispatches cleanly even when retrieval yields zero documents."""
    dispatched = {}

    async def fake_rag(query, **kw):
        dispatched["query"] = query
        return "here is a haiku", [], [], "chat", {}

    monkeypatch.setattr("src.internal.servers.web.app._run_agentic_rag", fake_rag)

    class _LLM:
        def complete(self, messages, **_):
            return "chat"  # LLM classifier picks the chat label

    app = create_web_app(
        SearchExperienceSettings(db_path=tmp_path / "db.sqlite3"), llm=_LLM()
    )
    client = TestClient(app)
    response = client.post("/api/agent", json={"query": "write a haiku about the sea"})
    assert response.status_code == 200
    body = response.json()
    assert body["intent"] == "chat"
    assert body["documents"] == []  # zero relevant docs, no crash
    assert dispatched["query"] == "write a haiku about the sea"


def _write_dispatch_index(tmp_path):
    import numpy as np

    from src.model.pre_training.intents.model import (
        DEFAULT_ENCODER,
        INDEX_FILENAME,
        CanonicalExample,
        IntentIndex,
    )

    axis = {"search": 0, "chat": 1, "tool": 2}
    module = {"search": "lookup_fact", "chat": "explain", "tool": "schedule"}
    examples, rows = [], []
    for route, index in axis.items():
        for position in range(12):
            examples.append(
                CanonicalExample(
                    f"{route}-{position}",
                    f"{route} {position}",
                    route,
                    (module[route],),
                )
            )
            rows.append(np.eye(3, dtype=np.float32)[index])
    directory = tmp_path / "intent_index"
    IntentIndex(examples, np.stack(rows), DEFAULT_ENCODER, "sha256:x").save(
        directory / INDEX_FILENAME
    )
    return directory


def _stub_encode_texts(monkeypatch, vector_by_query):
    import numpy as np

    from src.internal.servers.web.intent import similarity

    def _fake(texts):
        return np.stack([vector_by_query[text] for text in texts]).astype(np.float32)

    monkeypatch.setattr(similarity, "encode_texts", _fake)


def test_real_intent_index_dispatches_to_each_existing_runner(monkeypatch, tmp_path):
    import numpy as np

    from src.internal.configs import AppSettings

    index_path = _write_dispatch_index(tmp_path)
    _stub_encode_texts(
        monkeypatch,
        {
            "vendor renewal terms archive": np.array([1.0, 0.0, 0.0]),
            "friendly casual discussion response": np.array([0.0, 1.0, 0.0]),
            "workflow approval action request": np.array([0.0, 0.0, 1.0]),
        },
    )
    calls = []

    async def fake_chat(query, **kwargs):
        calls.append(("chat", query))
        return "chat answer", [], [], "chat", {}

    async def fake_search(query, **kwargs):
        calls.append(("search", query))
        return "search answer", [], [], "search", {}

    async def fake_tool(query, **kwargs):
        calls.append(("tool", query))
        return "tool answer", [], [], "tool", {}

    monkeypatch.setattr("src.internal.servers.web.app._run_agentic_rag", fake_chat)
    monkeypatch.setattr(
        "src.internal.servers.web.app._run_search_direct_or_escalate", fake_search
    )
    monkeypatch.setattr("src.internal.servers.web.app._run_tool_agent", fake_tool)

    class _UnexpectedLLM:
        def complete(self, messages, **kwargs):
            raise AssertionError("confident index route consulted the LLM")

    app = create_web_app(
        SearchExperienceSettings(db_path=tmp_path / "dispatch.sqlite3"),
        app_settings=AppSettings(
            intent_index_path=index_path,
        ),
        llm=_UnexpectedLLM(),
    )
    with TestClient(app) as client:
        app.state.search_agent_manager = object()
        app.state.search_agent_tokenizer = object()
        for query in [
            "vendor renewal terms archive",
            "friendly casual discussion response",
            "workflow approval action request",
        ]:
            response = client.post("/api/agent", json={"query": query})
            assert response.status_code == 200

    assert calls == [
        ("search", "vendor renewal terms archive"),
        ("chat", "friendly casual discussion response"),
        ("tool", "workflow approval action request"),
    ]


def test_real_intent_index_abstention_uses_classifier_fallback(monkeypatch, tmp_path):
    import numpy as np

    from src.internal.configs import AppSettings

    index_path = _write_dispatch_index(tmp_path)
    # Off-axis so confidence lands just under 1.0 while the margin to the
    # runner-up route stays wide, isolating the confidence-threshold path.
    near_search = np.array([0.99, 0.1, 0.0])
    near_search = near_search / np.linalg.norm(near_search)
    _stub_encode_texts(monkeypatch, {"vendor renewal discussion request": near_search})
    calls = []

    async def fake_chat(query, **kwargs):
        calls.append(("chat", query))
        return "fallback answer", [], [], "chat", {}

    async def unexpected_search(query, **kwargs):
        raise AssertionError("abstained index prediction dispatched search")

    async def unexpected_tool(query, **kwargs):
        raise AssertionError("abstained index prediction dispatched tool")

    monkeypatch.setattr("src.internal.servers.web.app._run_agentic_rag", fake_chat)
    monkeypatch.setattr(
        "src.internal.servers.web.app._run_search_direct_or_escalate",
        unexpected_search,
    )
    monkeypatch.setattr("src.internal.servers.web.app._run_tool_agent", unexpected_tool)

    class _ClassifierLLM:
        def __init__(self):
            self.calls = 0

        def complete(self, messages, **kwargs):
            self.calls += 1
            return "chat"

    llm = _ClassifierLLM()
    app = create_web_app(
        SearchExperienceSettings(db_path=tmp_path / "fallback.sqlite3"),
        app_settings=AppSettings(
            intent_index_path=index_path,
            # No confidence gate any more, so force abstention via the margin.
            intent_min_route_margin=1.0,
        ),
        llm=llm,
    )
    with TestClient(app) as client:
        response = client.post(
            "/api/agent", json={"query": "vendor renewal discussion request"}
        )

    assert response.status_code == 200
    assert llm.calls == 1
    assert calls == [("chat", "vendor renewal discussion request")]


def test_memory_compression_flag_defaults_off_and_reads_env(monkeypatch):
    monkeypatch.delenv("AGENTIC_SEARCH_MEMORY_COMPRESSION", raising=False)
    assert SearchExperienceSettings.from_app_settings().memory_compression is False
    monkeypatch.setenv("AGENTIC_SEARCH_MEMORY_COMPRESSION", "true")
    assert SearchExperienceSettings.from_app_settings().memory_compression is True


def _seed_long_session(store, n=45):
    session = store.create_chat_session(title="long")
    records = [
        store.add_chat_message(
            session.id, role="user" if i % 2 == 0 else "assistant", content=f"msg {i}"
        )
        for i in range(n)
    ]
    return session.id, records


def test_run_agent_prepends_stored_summary_when_flag_on(monkeypatch, tmp_path):
    from src.internal.cache.interface import InMemoryCache
    from src.internal.memory.working import (
        SUMMARY_PREFIX,
        SessionMemoryState,
        save_state,
    )

    cache = InMemoryCache()
    monkeypatch.setattr("src.internal.cache.interface._default_cache", cache)
    # Never let a developer's OPENAI_API_KEY turn this into a live call.
    monkeypatch.setattr(
        "src.internal.servers.web.app.schedule_compression", lambda wm, **kw: None
    )
    captured: list = []

    async def fake_answer(question, *, llm=None, chat_history=None, **kw):
        captured.append(list(chat_history or []))
        return _answer_result(question)

    monkeypatch.setattr(
        "src.internal.servers.web.app.answer_with_retrieval", fake_answer
    )

    store = AgenticSearchStore(tmp_path / "state.sqlite3")
    session_id, records = _seed_long_session(store)
    save_state(
        cache,
        session_id,
        SessionMemoryState(summary="S", summarized_through=records[4].id),
    )

    app = create_web_app(
        SearchExperienceSettings(
            db_path=tmp_path / "state.sqlite3", memory_compression=True
        ),
        store=store,
    )
    TestClient(app).post(
        "/api/agent",
        json={"query": "follow up", "mode": "chat_once", "session_id": session_id},
    )

    assert len(captured) == 1
    assert captured[0][0].role == "system"
    assert captured[0][0].content == SUMMARY_PREFIX + "S"
    assert len(captured[0]) == 41


def test_run_agent_flag_off_ignores_stored_summary(monkeypatch, tmp_path):
    from src.internal.cache.interface import InMemoryCache
    from src.internal.memory.working import SessionMemoryState, save_state

    cache = InMemoryCache()
    monkeypatch.setattr("src.internal.cache.interface._default_cache", cache)
    captured: list = []

    async def fake_answer(question, *, llm=None, chat_history=None, **kw):
        captured.append(list(chat_history or []))
        return _answer_result(question)

    monkeypatch.setattr(
        "src.internal.servers.web.app.answer_with_retrieval", fake_answer
    )

    store = AgenticSearchStore(tmp_path / "state.sqlite3")
    session_id, records = _seed_long_session(store)
    save_state(
        cache,
        session_id,
        SessionMemoryState(summary="S", summarized_through=records[4].id),
    )

    app = create_web_app(
        SearchExperienceSettings(db_path=tmp_path / "state.sqlite3"), store=store
    )
    TestClient(app).post(
        "/api/agent",
        json={"query": "follow up", "mode": "chat_once", "session_id": session_id},
    )

    # Flag off: no summary system message is injected ahead of the tail.
    assert [m.role for m in captured[0]][:1] != ["system"]
    assert len(captured[0]) == 40


def test_run_agent_schedules_compression_after_reply(monkeypatch, tmp_path):
    from src.internal.cache.interface import InMemoryCache

    cache = InMemoryCache()
    monkeypatch.setattr("src.internal.cache.interface._default_cache", cache)
    scheduled: list = []

    def fake_schedule(wm, **kw):
        scheduled.append((len(wm.pending), kw["enabled"], kw["llm"]))
        return None

    monkeypatch.setattr(
        "src.internal.servers.web.app.schedule_compression", fake_schedule
    )

    async def fake_answer(question, **kw):
        return _answer_result(question)

    monkeypatch.setattr(
        "src.internal.servers.web.app.answer_with_retrieval", fake_answer
    )

    store = AgenticSearchStore(tmp_path / "state.sqlite3")
    session_id, _ = _seed_long_session(store)
    sentinel_llm = object()
    app = create_web_app(
        SearchExperienceSettings(
            db_path=tmp_path / "state.sqlite3", memory_compression=True
        ),
        store=store,
        llm=sentinel_llm,
    )
    TestClient(app).post(
        "/api/agent",
        json={"query": "follow up", "mode": "chat_once", "session_id": session_id},
    )

    assert scheduled == [(5, True, sentinel_llm)]
