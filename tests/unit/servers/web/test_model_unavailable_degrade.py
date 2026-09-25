"""/api/agent degrades to search-only when the model is unavailable."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest
import requests
from fastapi.testclient import TestClient

from src.context.models import ContextDocument, ModelUnavailableError
from src.internal.servers.web.app import SearchExperienceSettings, create_web_app
from src.internal.servers.web.intent import RouteDecision, RouteStrategy

APP = "src.internal.servers.web.app"


async def _unavailable(*args, **kwargs):
    raise ModelUnavailableError("Cannot connect to inference server")


def _fake_pipeline(calls: list):
    async def fake(query, **kwargs):
        calls.append({"query": query, **kwargs})
        doc = ContextDocument(id="D1", title="Doc", content="body", score=0.5)
        return "search-only answer", ["D1"], [doc], "search", kwargs["extra"]

    return fake


def _post(app, body: dict, *, local_model: bool = False):
    with TestClient(app) as client:
        if local_model:
            app.state.search_agent_manager = MagicMock()
            app.state.search_agent_tokenizer = MagicMock()
        return client.post("/api/agent", json=body)


def _app(tmp_path):
    return create_web_app(
        SearchExperienceSettings(db_path=tmp_path / "db.sqlite3"), llm=MagicMock()
    )


# (mode, runner patched to raise, needs a local model)
MODES = [
    (None, "_run_agentic_rag", False),  # auto, CHAT route
    ("chat_once", "answer_with_retrieval", False),
    ("chat_loop", "_run_agentic_rag", False),
    ("search_agent", "_run_search_agent", True),
    ("tool_agent", "_run_tool_agent", True),
]


@pytest.mark.parametrize(("mode", "runner", "local_model"), MODES)
def test_degrades_per_mode(monkeypatch, tmp_path, mode, runner, local_model):
    monkeypatch.setattr(
        f"{APP}.recognize_intent", lambda *a, **k: RouteDecision(RouteStrategy.CHAT)
    )
    monkeypatch.setattr(f"{APP}.{runner}", _unavailable)
    calls: list = []
    monkeypatch.setattr(f"{APP}._auto_search_pipeline", _fake_pipeline(calls))
    body = {"query": "explain FAISS"}
    if mode:
        body["mode"] = mode

    response = _post(_app(tmp_path), body, local_model=local_model)

    assert response.status_code == 200, response.text
    data = response.json()
    assert data["answer"] == "search-only answer"
    assert data["hook_metadata"]["route_degraded"] == "model_unavailable"
    assert [d["id"] for d in data["documents"]] == ["D1"]
    assert data["messages"][-1]["content"] == "search-only answer"
    assert len(calls) == 1


def test_degrade_passes_request_scope_and_no_llm(monkeypatch, tmp_path):
    monkeypatch.setattr(f"{APP}.answer_with_retrieval", _unavailable)
    calls: list = []
    monkeypatch.setattr(f"{APP}._auto_search_pipeline", _fake_pipeline(calls))

    response = _post(
        _app(tmp_path), {"query": "explain FAISS", "mode": "chat_once", "top_k": 3}
    )

    assert response.status_code == 200
    (call,) = calls
    assert call["query"] == "explain FAISS"
    assert call["llm"] is None
    assert call["top_k"] == 3
    assert call["filters"].access_acl  # anonymous is ["public"], never unfiltered


@pytest.mark.parametrize(
    ("mode", "runner", "local_model", "requested", "expected"),
    [
        # Explicit modes are corpus-only: an outage must not send the query to
        # an external web provider they never contacted.
        ("chat_once", "answer_with_retrieval", False, None, "retrieval"),
        ("chat_loop", "_run_agentic_rag", False, "auto", "retrieval"),
        ("search_agent", "_run_search_agent", True, "serpapi", "retrieval"),
        ("tool_agent", "_run_tool_agent", True, None, "retrieval"),
        # Auto mode keeps the provider the request chose.
        (None, "_run_agentic_rag", False, None, "auto"),
        (None, "_run_agentic_rag", False, "retrieval", "retrieval"),
    ],
)
def test_degrade_source_provider_by_mode(
    monkeypatch, tmp_path, mode, runner, local_model, requested, expected
):
    monkeypatch.setattr(
        f"{APP}.recognize_intent", lambda *a, **k: RouteDecision(RouteStrategy.CHAT)
    )
    monkeypatch.setattr(f"{APP}.{runner}", _unavailable)
    calls: list = []
    monkeypatch.setattr(f"{APP}._auto_search_pipeline", _fake_pipeline(calls))
    body = {"query": "explain FAISS"}
    if mode:
        body["mode"] = mode
    if requested:
        body["source_provider"] = requested

    response = _post(_app(tmp_path), body, local_model=local_model)

    assert response.status_code == 200, response.text
    assert calls[0]["source_provider"] == expected


def test_explicit_mode_with_unknown_source_provider_still_degrades(
    monkeypatch, tmp_path
):
    """Explicit modes never validated source_provider; an outage must not make
    a request that succeeds with a working model return 502."""
    monkeypatch.setattr(f"{APP}.answer_with_retrieval", _unavailable)
    calls: list = []
    monkeypatch.setattr(f"{APP}._auto_search_pipeline", _fake_pipeline(calls))

    response = _post(
        _app(tmp_path),
        {"query": "explain FAISS", "mode": "chat_once", "source_provider": "bogus"},
    )

    assert response.status_code == 200, response.text
    assert calls[0]["source_provider"] == "retrieval"


@pytest.mark.parametrize(
    "error",
    [
        ValueError("bad input format"),
        requests.HTTPError(
            "400 context length exceeded",
            response=MagicMock(status_code=400),
        ),
    ],
    ids=["value_error", "http_4xx"],
)
def test_non_availability_errors_still_502(monkeypatch, tmp_path, error):
    async def explode(*args, **kwargs):
        raise error

    monkeypatch.setattr(f"{APP}.answer_with_retrieval", explode)
    calls: list = []
    monkeypatch.setattr(f"{APP}._auto_search_pipeline", _fake_pipeline(calls))

    response = _post(_app(tmp_path), {"query": "q", "mode": "chat_once"})

    assert response.status_code == 502
    assert str(error) in response.json()["detail"]
    assert calls == []


def test_fallback_failure_still_returns_502(monkeypatch, tmp_path):
    monkeypatch.setattr(f"{APP}.answer_with_retrieval", _unavailable)

    async def pipeline_down(query, **kwargs):
        raise RuntimeError("retrieval server unreachable")

    monkeypatch.setattr(f"{APP}._auto_search_pipeline", pipeline_down)

    response = _post(_app(tmp_path), {"query": "q", "mode": "chat_once"})

    assert response.status_code == 502
    assert "retrieval server unreachable" in response.json()["detail"]


def _stub_corpus_with_private_doc(monkeypatch):
    """Both degrade paths return a public and a private document; only real
    enforcement code may remove the private one."""
    from src.context.search import SearchResult
    from src.internal.tools import SearchPage

    rows = [
        ("Public", "ok", "http://r/pub", ["public"]),
        ("Private", "secret", "http://r/priv", ["user:alice"]),
    ]

    # Auto mode: the hybrid fan-out's retrieval leg goes through search_tool.
    async def fake_search_tool(query, *, provider, search_url, page_size, **kw):
        if provider != "retrieval":
            return []
        return [
            SearchPage(title=t, summary=c, url=u, metadata={"acl": acl})
            for t, c, u, acl in rows
        ]

    # Explicit modes: corpus-only retrieval goes through SearchClient.
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

    async def no_browser(*args, **kwargs):
        return []

    monkeypatch.setattr(f"{APP}.search_tool", fake_search_tool)
    monkeypatch.setattr("src.context.retrieval.search_runner.SearchClient", _Client)
    monkeypatch.setattr(f"{APP}._run_browser_search", no_browser)


@pytest.mark.parametrize(
    ("mode", "runner"),
    [("chat_once", "answer_with_retrieval"), (None, "_run_agentic_rag")],
)
def test_degraded_documents_are_acl_filtered(monkeypatch, tmp_path, mode, runner):
    """Real fallback pipeline: a private document never reaches an anonymous caller."""
    monkeypatch.setattr(
        f"{APP}.recognize_intent", lambda *a, **k: RouteDecision(RouteStrategy.CHAT)
    )
    monkeypatch.setattr(f"{APP}.{runner}", _unavailable)
    _stub_corpus_with_private_doc(monkeypatch)
    body = {"query": "q"}
    if mode:
        body["mode"] = mode

    response = _post(_app(tmp_path), body)

    assert response.status_code == 200, response.text
    data = response.json()
    titles = [d["title"] for d in data["documents"]]
    assert "Public" in titles
    assert "Private" not in titles
    assert "secret" not in data["answer"]
    assert data["hook_metadata"]["route_degraded"] == "model_unavailable"


def test_stream_done_event_reports_model_unavailable(monkeypatch, tmp_path):
    monkeypatch.setattr(
        f"{APP}.recognize_intent", lambda *a, **k: RouteDecision(RouteStrategy.CHAT)
    )
    monkeypatch.setattr(f"{APP}._run_agentic_rag", _unavailable)
    monkeypatch.setattr(f"{APP}._auto_search_pipeline", _fake_pipeline([]))

    with TestClient(_app(tmp_path)) as client:
        response = client.post("/api/agent/stream", json={"query": "explain FAISS"})

    assert response.status_code == 200
    events = [
        json.loads(line[len("data:") :].strip())
        for line in response.text.splitlines()
        if line.startswith("data:") and line[len("data:") :].strip()
    ]
    assert not [e for e in events if e.get("type") == "error"]
    done = next(e for e in events if e["type"] == "done")
    assert done["route_degraded"] == "model_unavailable"
    answer = next(e for e in events if e["type"] == "answer")
    assert answer["text"] == "search-only answer"
