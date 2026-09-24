import asyncio

from fastapi.testclient import TestClient
from src.internal.servers.web.app import SearchExperienceSettings, create_web_app

import src.internal.servers.web.app as web_app
from src.shared_configs.intent import RouteStrategy

RESOLVED = "What is FAISS?\nDoes it support GPUs?"


def _search_call(monkeypatch, **kwargs):
    seen = {}

    async def fake_direct(query, **kw):
        seen["direct"] = query
        return []

    def fake_gate(query, docs, **kw):
        seen["gate"] = query
        return False, "weak", 0.0, None

    async def fake_agent(query, **kw):
        seen["agent"] = query
        return "agent answer", [], [], "search", {}

    async def fake_pipeline(query, **kw):
        seen["pipeline"] = (query, kw.get("retrieval_query"))
        return "pipeline answer", [], [], "search", kw.get("extra", {})

    monkeypatch.setattr(web_app, "_run_direct_search", fake_direct)
    monkeypatch.setattr(web_app, "_direct_gate_decision", fake_gate)
    monkeypatch.setattr(web_app, "_run_search_agent", fake_agent)
    monkeypatch.setattr(web_app, "_auto_search_pipeline", fake_pipeline)
    monkeypatch.setattr(web_app, "gate_embedder", lambda: None)
    asyncio.run(
        web_app._run_search_direct_or_escalate(
            "Does it support GPUs?",
            manager=kwargs.get("manager", object()),
            tokenizer=kwargs.get("tokenizer", object()),
            llm=None,
            search_url="http://x/retrieve",
            browser_search_url=None,
            rerank_url=None,
            top_k=5,
            filters=None,
            history=[],
            source_provider="retrieval",
            on_turn=None,
            retrieval_query=kwargs.get("retrieval_query"),
        )
    )
    return seen


def test_direct_search_and_gate_use_the_resolved_query(monkeypatch):
    seen = _search_call(monkeypatch, retrieval_query=RESOLVED)
    assert seen["direct"] == RESOLVED
    assert seen["gate"] == RESOLVED


def test_escalation_to_the_search_agent_keeps_the_raw_message(monkeypatch):
    seen = _search_call(monkeypatch, retrieval_query=RESOLVED)
    assert seen["agent"] == "Does it support GPUs?"


def test_degraded_pipeline_gets_raw_query_and_resolved_retrieval_query(monkeypatch):
    seen = _search_call(
        monkeypatch, retrieval_query=RESOLVED, manager=None, tokenizer=None
    )
    assert seen["pipeline"] == ("Does it support GPUs?", RESOLVED)


def test_search_without_retrieval_query_is_unchanged(monkeypatch):
    seen = _search_call(monkeypatch)
    assert seen["direct"] == seen["gate"] == "Does it support GPUs?"


def test_auto_routed_chat_passes_retrieval_query_to_agentic_rag(monkeypatch):
    seen = {}

    async def fake_rag(query, **kw):
        seen["rag"] = (query, kw.get("retrieval_query"))
        return "answer", [], [], "chat", {}

    monkeypatch.setattr(web_app, "_run_agentic_rag", fake_rag)
    asyncio.run(
        web_app._run_auto_routed(
            "Does it support GPUs?",
            llm=object(),
            manager=None,
            tokenizer=None,
            search_url="http://x/retrieve",
            browser_search_url=None,
            rerank_url=None,
            top_k=5,
            filters=None,
            history=[],
            resolved=None,
            forced_route=RouteStrategy.CHAT,
            retrieval_query=RESOLVED,
        )
    )
    assert seen["rag"] == ("Does it support GPUs?", RESOLVED)


def test_auto_routed_search_passes_retrieval_query(monkeypatch):
    seen = {}

    async def fake_search(query, **kw):
        seen["search"] = (query, kw.get("retrieval_query"))
        return "answer", [], [], "search", {}

    monkeypatch.setattr(web_app, "_run_search_direct_or_escalate", fake_search)
    asyncio.run(
        web_app._run_auto_routed(
            "Does it support GPUs?",
            llm=None,
            manager=None,
            tokenizer=None,
            search_url="http://x/retrieve",
            browser_search_url=None,
            rerank_url=None,
            top_k=5,
            filters=None,
            history=[],
            resolved=None,
            forced_route=RouteStrategy.SEARCH,
            retrieval_query=RESOLVED,
        )
    )
    assert seen["search"] == ("Does it support GPUs?", RESOLVED)


def test_search_pipeline_uses_the_retrieval_query_override():
    from src.internal.search.pipeline import SearchPipeline

    seen = {}

    class Retrieval:
        async def retrieve(self, query, history, filters, top_k):
            seen["retrieve"] = query
            raise RuntimeError("stop after retrieval")

    pipeline = SearchPipeline(Retrieval(), None, None)
    _answer, _c, _d, _i, extra = asyncio.run(
        pipeline.run(
            "Does it support GPUs?", [], None, 5, "retrieval", retrieval_query=RESOLVED
        )
    )
    assert seen["retrieve"] == RESOLVED
    assert extra["retrieval_query"] == RESOLVED


class _ChatLLM:
    def complete(self, messages, **_):
        return "chat"


def _two_turns(
    monkeypatch, tmp_path, *, flag, second="Does it support GPUs?", mode=None
):
    calls = []

    async def fake_rag(query, **kw):
        calls.append((query, kw.get("retrieval_query")))
        return "an answer", [], [], "chat", {}

    monkeypatch.setattr(web_app, "_run_agentic_rag", fake_rag)
    monkeypatch.setattr(web_app, "gate_embedder", lambda: None)
    app = create_web_app(
        SearchExperienceSettings(
            db_path=tmp_path / "db.sqlite3", follow_up_resolution=flag
        ),
        llm=_ChatLLM(),
    )
    client = TestClient(app)
    body = {"query": "What is FAISS?"}
    if mode:
        body["mode"] = mode
    first = client.post("/api/agent", json=body).json()
    second_body = {**body, "query": second, "session_id": first["session_id"]}
    second_response = client.post("/api/agent", json=second_body).json()
    return calls, first, second_response


def test_flag_on_resolves_the_follow_up_for_retrieval(monkeypatch, tmp_path):
    calls, _first, second = _two_turns(monkeypatch, tmp_path, flag=True)
    assert calls[-1] == (
        "Does it support GPUs?",
        "What is FAISS?\nDoes it support GPUs?",
    )
    assert second["hook_metadata"]["follow_up"] == {
        "continuation": True,
        "reason": "reference",
        "query": "What is FAISS?\nDoes it support GPUs?",
    }


def test_flag_on_first_turn_reports_no_history(monkeypatch, tmp_path):
    calls, first, _second = _two_turns(monkeypatch, tmp_path, flag=True)
    assert calls[0] == ("What is FAISS?", "What is FAISS?")
    assert first["hook_metadata"]["follow_up"] == {
        "continuation": False,
        "reason": "no_history",
        "query": "What is FAISS?",
    }


def test_flag_off_changes_nothing(monkeypatch, tmp_path):
    calls, first, second = _two_turns(monkeypatch, tmp_path, flag=False)
    assert calls == [("What is FAISS?", None), ("Does it support GPUs?", None)]
    assert "follow_up" not in first["hook_metadata"]
    assert "follow_up" not in second["hook_metadata"]


def test_explicit_chat_loop_mode_is_resolved_too(monkeypatch, tmp_path):
    calls, _first, _second = _two_turns(
        monkeypatch, tmp_path, flag=True, mode="chat_loop"
    )
    assert calls[-1][1] == "What is FAISS?\nDoes it support GPUs?"


def test_chat_once_mode_passes_retrieval_query(monkeypatch, tmp_path):
    seen = []

    async def fake_awr(question, **kw):
        seen.append((question, kw.get("retrieval_query")))
        from types import SimpleNamespace

        from src.context import SearchContextBundle

        # The chat_once path reads only answer, citations and context.documents.
        return SimpleNamespace(
            answer="an answer",
            citations=[],
            context=SearchContextBundle(query=question, documents=[]),
        )

    monkeypatch.setattr(web_app, "answer_with_retrieval", fake_awr)
    monkeypatch.setattr(web_app, "gate_embedder", lambda: None)
    app = create_web_app(
        SearchExperienceSettings(
            db_path=tmp_path / "db.sqlite3", follow_up_resolution=True
        ),
        llm=_ChatLLM(),
    )
    client = TestClient(app)
    first = client.post(
        "/api/agent", json={"query": "What is FAISS?", "mode": "chat_once"}
    ).json()
    client.post(
        "/api/agent",
        json={
            "query": "Does it support GPUs?",
            "mode": "chat_once",
            "session_id": first["session_id"],
        },
    )
    assert seen[-1] == (
        "Does it support GPUs?",
        "What is FAISS?\nDoes it support GPUs?",
    )
