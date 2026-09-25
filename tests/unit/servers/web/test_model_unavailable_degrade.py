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
    assert call["source_provider"] == "auto"


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


def test_degraded_documents_are_acl_filtered(monkeypatch, tmp_path):
    """Real fallback pipeline: a private document never reaches an anonymous caller."""
    from src.internal.tools import SearchPage

    monkeypatch.setattr(f"{APP}.answer_with_retrieval", _unavailable)

    async def fake_search_tool(query, *, provider, search_url, page_size, **kw):
        if provider != "retrieval":
            return []
        return [
            SearchPage(
                title="Public",
                summary="ok",
                url="http://r/pub",
                metadata={"acl": ["public"]},
            ),
            SearchPage(
                title="Private",
                summary="secret",
                url="http://r/priv",
                metadata={"acl": ["user:alice"]},
            ),
        ]

    async def no_browser(*args, **kwargs):
        return []

    monkeypatch.setattr(f"{APP}.search_tool", fake_search_tool)
    monkeypatch.setattr(f"{APP}._run_browser_search", no_browser)

    response = _post(_app(tmp_path), {"query": "q", "mode": "chat_once"})

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
