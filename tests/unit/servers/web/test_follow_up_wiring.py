import asyncio

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
