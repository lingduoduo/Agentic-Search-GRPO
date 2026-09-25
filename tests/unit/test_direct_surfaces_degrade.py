"""/tool and /chat degrade when the local model is unavailable."""

from __future__ import annotations

import json

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.context.models import ContextDocument, ModelUnavailableError
from src.internal.configs import load_app_settings
from src.internal.db import AgenticSearchStore
from src.internal.servers.query_and_chat import chat_backend
from src.internal.servers.query_and_chat.chat_backend import (
    CHAT_MODEL_UNAVAILABLE_MESSAGE,
    create_chat_router,
)
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


# What the fake pipeline's one document renders to on /tool.
EXPECTED_TOOL_ANSWER = (
    "The tool model is temporarily unavailable, so this answer comes straight "
    "from a search of your documents.\n\n1. Doc\n   body"
)


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
    assert data["answer"] == EXPECTED_TOOL_ANSWER
    assert data["degraded"] == "model_unavailable"
    assert data["error"] is None
    assert data["tool_calls"] == [] and data["num_turns"] == 0
    assert _roles(store, data["session_id"]) == ["user", "assistant"]
    assert (
        store.list_chat_messages(data["session_id"])[-1].content == EXPECTED_TOOL_ANSWER
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
    assert events[0]["text"] == EXPECTED_TOOL_ANSWER
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
    # /tool has no Sources panel and no documents field, so the answer itself
    # must carry what was found -- and only what the caller may read.
    answer = data["answer"]
    assert "Public" in answer and "http://r/pub" in answer
    assert "Private" not in answer and "secret" not in answer
    assert "Sources panel" not in answer


def test_tool_degrade_with_no_documents_says_so(monkeypatch):
    class _Empty:
        def __init__(self, config):
            pass

        async def retrieve_one(self, query, **kwargs):
            return []

        async def aclose(self):
            return None

    monkeypatch.setattr(TOOL_RUNNER, _unavailable)
    monkeypatch.setattr("src.context.retrieval.search_runner.SearchClient", _Empty)
    app, _ = _tool_app()

    data = _send_tool(app, stream=False).json()

    assert data["degraded"] == "model_unavailable"
    assert "no matching documents" in data["answer"]
    assert "Sources panel" not in data["answer"]


def _chat_app():
    store = AgenticSearchStore(":memory:")
    app = FastAPI()
    app.include_router(create_chat_router(store))
    app.state.search_agent_manager = object()
    app.state.search_agent_tokenizer = object()
    return app, store


def _send_chat(app, *, stream: bool):
    return TestClient(app).post(
        "/chat/send-chat-message", json={"message": "hello", "stream": stream}
    )


def test_chat_degrades_with_unavailable_message(monkeypatch):
    monkeypatch.setattr(chat_backend, "_run_plain_chat", _unavailable)
    app, store = _chat_app()

    data = _send_chat(app, stream=False).json()

    assert data["answer"] == CHAT_MODEL_UNAVAILABLE_MESSAGE
    assert data["degraded"] == "model_unavailable"
    assert data["error"] is None
    # The message is not content: only the user turn is stored.
    assert _roles(store, data["session_id"]) == ["user"]


def test_chat_stream_degrades(monkeypatch):
    monkeypatch.setattr(chat_backend, "_run_plain_chat", _unavailable)
    app, store = _chat_app()

    events = _events(_send_chat(app, stream=True).text)

    assert [e["type"] for e in events] == ["answer", "done"]
    assert events[0]["text"] == CHAT_MODEL_UNAVAILABLE_MESSAGE
    assert events[1]["degraded"] == "model_unavailable"
    assert _roles(store, events[1]["session_id"]) == ["user"]


@pytest.mark.parametrize("stream", [False, True], ids=["json", "stream"])
def test_chat_other_errors_keep_error(monkeypatch, stream):
    async def explode(*args, **kwargs):
        raise ValueError("bad input format")

    monkeypatch.setattr(chat_backend, "_run_plain_chat", explode)
    app, _ = _chat_app()

    resp = _send_chat(app, stream=stream)

    if stream:
        assert _events(resp.text)[-1] == {"type": "error", "detail": "bad input format"}
    else:
        assert resp.json()["answer"] == ""
        assert resp.json()["error"] == "bad input format"
        assert resp.json()["degraded"] is None


PLAIN_RUNNER = "src.internal.servers.web.plain_chat_runner._run_plain_chat"


def test_chat_stream_through_the_real_wrapper_streams_tokens(monkeypatch):
    """The wrapper must forward on_token; patching the wrapper itself hid that
    every real streaming /chat request failed with a TypeError."""

    async def runner(message, *, manager, tokenizer, history, on_token=None, **kw):
        for piece in ("hel", "lo"):
            await on_token(piece)
        return "hello"

    monkeypatch.setattr(PLAIN_RUNNER, runner)
    app, _store = _chat_app()

    events = _events(_send_chat(app, stream=True).text)

    assert [e["type"] for e in events] == ["token", "token", "answer", "done"]
    assert events[2]["text"] == "hello"


def test_chat_stream_degrades_through_the_real_wrapper(monkeypatch):
    monkeypatch.setattr(PLAIN_RUNNER, _unavailable)
    app, store = _chat_app()

    events = _events(_send_chat(app, stream=True).text)

    assert [e["type"] for e in events] == ["answer", "done"]
    assert events[1]["degraded"] == "model_unavailable"
    assert _roles(store, events[1]["session_id"]) == ["user"]
