"""/tool and /chat degrade when the local model is unavailable."""

from __future__ import annotations

import json

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.context.models import ContextDocument, ModelUnavailableError
from src.internal.configs import load_app_settings
from src.internal.db import AgenticSearchStore
from src.internal.servers.query_and_chat.tool_backend import create_tool_router

APP = "src.internal.servers.web.app"
TOOL_RUNNER = "src.internal.servers.web.tool_agent_runner._run_tool_agent"


async def _unavailable(*args, **kwargs):
    raise ModelUnavailableError("Cannot connect to inference server")


def _fake_pipeline(calls: list):
    async def fake(query, **kwargs):
        calls.append({"query": query, **kwargs})
        doc = ContextDocument(id="D1", title="Doc", content="body", score=0.5)
        return "search-only answer", ["D1"], [doc], "search", kwargs["extra"]

    return fake


def _events(text: str) -> list[dict]:
    return [
        json.loads(line[len("data:") :].strip())
        for line in text.splitlines()
        if line.startswith("data:") and line[len("data:") :].strip()
    ]


def _roles(store, session_id) -> list[str]:
    return [m.role for m in store.list_chat_messages(session_id)]


def _tool_app():
    store = AgenticSearchStore(":memory:")
    app = FastAPI()
    app.include_router(
        create_tool_router(
            store, search_url="http://x/retrieve", resolved=load_app_settings()
        )
    )
    app.state.search_agent_manager = object()
    app.state.search_agent_tokenizer = object()
    return app, store


def _send_tool(app, *, stream: bool):
    return TestClient(app).post(
        "/tool/send-tool-message", json={"message": "explain FAISS", "stream": stream}
    )


def test_tool_degrades_to_search_only_answer(monkeypatch):
    monkeypatch.setattr(TOOL_RUNNER, _unavailable)
    monkeypatch.setattr(f"{APP}._auto_search_pipeline", _fake_pipeline([]))
    app, store = _tool_app()

    resp = _send_tool(app, stream=False)

    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["answer"] == "search-only answer"
    assert data["degraded"] == "model_unavailable"
    assert data["error"] is None
    assert data["tool_calls"] == [] and data["num_turns"] == 0
    assert _roles(store, data["session_id"]) == ["user", "assistant"]
    assert store.list_chat_messages(data["session_id"])[-1].content == (
        "search-only answer"
    )


def test_tool_degrade_is_corpus_only_with_caller_acl(monkeypatch):
    monkeypatch.setattr(TOOL_RUNNER, _unavailable)
    calls: list = []
    monkeypatch.setattr(f"{APP}._auto_search_pipeline", _fake_pipeline(calls))
    app, _ = _tool_app()

    assert _send_tool(app, stream=False).status_code == 200

    (call,) = calls
    assert call["query"] == "explain FAISS"
    assert call["llm"] is None
    assert call["source_provider"] == "retrieval"
    assert call["search_url"] == "http://x/retrieve"
    assert call["top_k"] == 5
    assert call["filters"].access_acl == ["public"]  # anonymous caller
    assert call["extra"] == {"route_degraded": "model_unavailable"}


def test_tool_stream_degrades(monkeypatch):
    monkeypatch.setattr(TOOL_RUNNER, _unavailable)
    monkeypatch.setattr(f"{APP}._auto_search_pipeline", _fake_pipeline([]))
    app, store = _tool_app()

    resp = _send_tool(app, stream=True)

    assert resp.status_code == 200
    events = _events(resp.text)
    assert not [e for e in events if e["type"] == "error"]
    assert [e["type"] for e in events] == ["answer", "done"]
    assert events[0]["text"] == "search-only answer"
    done = events[-1]
    assert done["degraded"] == "model_unavailable"
    assert _roles(store, done["session_id"]) == ["user", "assistant"]


@pytest.mark.parametrize("stream", [False, True], ids=["json", "stream"])
def test_tool_other_errors_keep_error(monkeypatch, stream):
    async def explode(*args, **kwargs):
        raise ValueError("bad input format")

    monkeypatch.setattr(TOOL_RUNNER, explode)
    calls: list = []
    monkeypatch.setattr(f"{APP}._auto_search_pipeline", _fake_pipeline(calls))
    app, _ = _tool_app()

    resp = _send_tool(app, stream=stream)

    assert calls == []
    if stream:
        assert _events(resp.text)[-1] == {"type": "error", "detail": "bad input format"}
    else:
        assert resp.json()["answer"] == ""
        assert resp.json()["error"] == "bad input format"
        assert resp.json()["degraded"] is None


@pytest.mark.parametrize("stream", [False, True], ids=["json", "stream"])
def test_tool_fallback_failure_keeps_error(monkeypatch, stream):
    monkeypatch.setattr(TOOL_RUNNER, _unavailable)

    async def pipeline_down(query, **kwargs):
        raise RuntimeError("retrieval server unreachable")

    monkeypatch.setattr(f"{APP}._auto_search_pipeline", pipeline_down)
    app, store = _tool_app()

    resp = _send_tool(app, stream=stream)

    if stream:
        events = _events(resp.text)
        assert events[-1] == {"type": "error", "detail": "retrieval server unreachable"}
    else:
        data = resp.json()
        assert data["answer"] == ""
        assert data["error"] == "retrieval server unreachable"
        assert _roles(store, data["session_id"]) == ["user"]


def test_tool_degrade_acl_filters_private_document(monkeypatch):
    """Real fallback pipeline: a private row never reaches an anonymous caller."""
    from src.context.search import SearchResult

    rows = [
        ("Public", "ok", "http://r/pub", ["public"]),
        ("Private", "secret", "http://r/priv", ["user:alice"]),
    ]

    class _Client:
        def __init__(self, config):
            pass

        async def retrieve_one(self, query, **kwargs):
            return [
                SearchResult(contents=c, title=t, url=u, metadata={"acl": acl})
                for t, c, u, acl in rows
            ]

        async def aclose(self):
            return None

    monkeypatch.setattr(TOOL_RUNNER, _unavailable)
    monkeypatch.setattr("src.context.retrieval.search_runner.SearchClient", _Client)
    app, _ = _tool_app()

    data = _send_tool(app, stream=False).json()

    assert data["degraded"] == "model_unavailable"
    # The answer reports a count, not titles: of the two stubbed rows only the
    # public one may be counted.
    assert "returned 1 result(s)" in data["answer"]
    assert "secret" not in data["answer"]
