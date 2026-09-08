"""Tests for ToolCallView trace parsing in _run_auto_routed."""

from __future__ import annotations

import json
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, MagicMock

from src.internal.servers.web.app import SearchExperienceSettings, create_web_app
from src.internal.servers.web.intent import RouteDecision, RouteStrategy
from src.agents.core.base import AgentLoopOutput


def _make_output(action_trace: str) -> AgentLoopOutput:
    return AgentLoopOutput(
        prompt_ids=[],
        response_ids=[],
        response_mask=[],
        num_turns=1,
        final_answer="done",
        action_trace=action_trace,
    )


def _trace_line(
    tool_name, status, result, arguments=None, execution_time=0.123, error_message=None
):
    return json.dumps(
        {
            "tool_name": tool_name,
            "status": status,
            "result": result,
            "arguments": arguments or {},
            "performance": {"execution_time": execution_time},
            "error_message": error_message,
        }
    )


def test_tool_calls_populated_from_action_trace(monkeypatch, tmp_path):
    trace = "\n".join(
        [
            _trace_line(
                "search_routing_tool",
                "TaskStatus.COMPLETED",
                json.dumps([{"title": "t", "content": "c", "url": None}]),
                {"query": "q"},
            ),
            _trace_line(
                "some_other_tool",
                "TaskStatus.COMPLETED",
                "plain result",
                {"x": 1},
                execution_time=0.05,
            ),
        ]
    )
    monkeypatch.setattr(
        "src.internal.servers.web.app.recognize_intent",
        lambda *a, **k: RouteDecision(RouteStrategy.TOOL),
    )
    monkeypatch.setattr(
        "src.agents.tool.tool_calling.ToolAgentLoop.run",
        AsyncMock(return_value=_make_output(trace)),
    )
    app = create_web_app(SearchExperienceSettings(db_path=tmp_path / "db.sqlite3"))
    with TestClient(app) as client:
        app.state.search_agent_manager = MagicMock()
        app.state.search_agent_tokenizer = MagicMock()
        response = client.post("/api/agent", json={"query": "test"})
    assert response.status_code == 200
    data = response.json()
    assert len(data["tool_calls"]) == 2
    assert data["tool_calls"][0]["tool_name"] == "search_routing_tool"
    assert data["tool_calls"][1]["tool_name"] == "some_other_tool"


def test_latency_computed_from_execution_time(monkeypatch, tmp_path):
    trace = _trace_line("my_tool", "TaskStatus.COMPLETED", "ok", execution_time=0.456)
    monkeypatch.setattr(
        "src.internal.servers.web.app.recognize_intent",
        lambda *a, **k: RouteDecision(RouteStrategy.TOOL),
    )
    monkeypatch.setattr(
        "src.agents.tool.tool_calling.ToolAgentLoop.run",
        AsyncMock(return_value=_make_output(trace)),
    )
    app = create_web_app(SearchExperienceSettings(db_path=tmp_path / "db.sqlite3"))
    with TestClient(app) as client:
        app.state.search_agent_manager = MagicMock()
        app.state.search_agent_tokenizer = MagicMock()
        response = client.post("/api/agent", json={"query": "test"})
    assert response.status_code == 200
    assert response.json()["tool_calls"][0]["latency_ms"] == 456


def test_list_result_becomes_n_items(monkeypatch, tmp_path):
    trace = _trace_line(
        "search_routing_tool",
        "TaskStatus.COMPLETED",
        json.dumps([{"title": "a"}, {"title": "b"}]),
    )
    monkeypatch.setattr(
        "src.internal.servers.web.app.recognize_intent",
        lambda *a, **k: RouteDecision(RouteStrategy.TOOL),
    )
    monkeypatch.setattr(
        "src.agents.tool.tool_calling.ToolAgentLoop.run",
        AsyncMock(return_value=_make_output(trace)),
    )
    app = create_web_app(SearchExperienceSettings(db_path=tmp_path / "db.sqlite3"))
    with TestClient(app) as client:
        app.state.search_agent_manager = MagicMock()
        app.state.search_agent_tokenizer = MagicMock()
        response = client.post("/api/agent", json={"query": "test"})
    assert response.status_code == 200
    assert response.json()["tool_calls"][0]["result_summary"] == "2 items"


def test_string_result_truncated_to_200(monkeypatch, tmp_path):
    long_result = "x" * 300
    trace = _trace_line("my_tool", "TaskStatus.COMPLETED", long_result)
    monkeypatch.setattr(
        "src.internal.servers.web.app.recognize_intent",
        lambda *a, **k: RouteDecision(RouteStrategy.TOOL),
    )
    monkeypatch.setattr(
        "src.agents.tool.tool_calling.ToolAgentLoop.run",
        AsyncMock(return_value=_make_output(trace)),
    )
    app = create_web_app(SearchExperienceSettings(db_path=tmp_path / "db.sqlite3"))
    with TestClient(app) as client:
        app.state.search_agent_manager = MagicMock()
        app.state.search_agent_tokenizer = MagicMock()
        response = client.post("/api/agent", json={"query": "test"})
    assert response.status_code == 200
    assert len(response.json()["tool_calls"][0]["result_summary"]) == 200


def test_failed_tool_call_error_message(monkeypatch, tmp_path):
    trace = _trace_line("bad_tool", "TaskStatus.FAILED", None, error_message="timeout")
    monkeypatch.setattr(
        "src.internal.servers.web.app.recognize_intent",
        lambda *a, **k: RouteDecision(RouteStrategy.TOOL),
    )
    monkeypatch.setattr(
        "src.agents.tool.tool_calling.ToolAgentLoop.run",
        AsyncMock(return_value=_make_output(trace)),
    )
    app = create_web_app(SearchExperienceSettings(db_path=tmp_path / "db.sqlite3"))
    with TestClient(app) as client:
        app.state.search_agent_manager = MagicMock()
        app.state.search_agent_tokenizer = MagicMock()
        response = client.post("/api/agent", json={"query": "test"})
    assert response.status_code == 200
    tc = response.json()["tool_calls"][0]
    assert tc["status"] == "failed"
    assert tc["error"] == "timeout"


def test_no_tool_calls_on_chat_path(monkeypatch, tmp_path):
    from src.context.models import (
        AnswerGenerationResult,
        SearchContextBundle,
        PromptBundle,
    )

    monkeypatch.setattr(
        "src.internal.servers.web.app.answer_with_retrieval",
        AsyncMock(
            return_value=AnswerGenerationResult(
                answer="hi",
                citations=[],
                context=SearchContextBundle(query="q", documents=[]),
                prompt=PromptBundle(system="", user="", messages=[]),
            )
        ),
    )
    app = create_web_app(SearchExperienceSettings(db_path=tmp_path / "db.sqlite3"))
    with TestClient(app) as client:
        response = client.post("/api/agent", json={"query": "explain FAISS"})
    assert response.status_code == 200
    assert response.json()["tool_calls"] == []
