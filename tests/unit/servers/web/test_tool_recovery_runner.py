"""Tests for tool-recovery metadata surfacing through the web runners.

Covers `_run_tool_agent`'s note-appending + `extra["tool_recovery"]`, threading
`on_escalation` through `_run_tool_agent`, `_run_auto_routed`, `_run_agent_impl`
and `tool_backend._run`, and the `/tool/send-tool-message` + `/api/agent`
response surfaces (JSON, streaming `done` event, explicit `tool_agent` mode).
"""

from __future__ import annotations

import json
import types
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

import src.internal.servers.web.app as web_app
from src.agents.core.base import AgentLoopOutput
from src.internal.servers.web.app import SearchExperienceSettings, create_web_app


def _output(answer, recovery):
    return AgentLoopOutput(
        prompt_ids=[],
        response_ids=[],
        response_mask=[],
        num_turns=1,
        final_answer=answer,
        action_trace="",
        trajectory_messages=[{"role": "assistant", "content": answer}],
        tool_recovery=recovery,
    )


async def _run(monkeypatch, output, **kwargs):
    run = AsyncMock(return_value=output)
    monkeypatch.setattr("src.agents.tool.tool_calling.ToolAgentLoop.run", run)
    result = await web_app._run_tool_agent(
        "q",
        manager=MagicMock(),
        tokenizer=MagicMock(),
        search_url="http://x/retrieve",
        history=[],
        resolved=types.SimpleNamespace(tool_agent_parser="json"),
        on_turn=None,
        with_search_tool=False,
        **kwargs,
    )
    return result, run


DEGRADED = {
    "outcome": "degraded",
    "needs_user": False,
    "retries": 2,
    "degraded": ["get_weather"],
    "escalations": [],
}

STOPPED = {
    "outcome": "cancelled",
    "needs_user": False,
    "retries": 0,
    "degraded": [],
    "escalations": [
        {"tool": "send", "category": "unknown", "attempts": 1, "decision": "cancel"}
    ],
}
STOPPED_TEXT = "Stopped: send failed (unknown); nothing further was attempted."


@pytest.mark.asyncio
async def test_degraded_answer_gets_the_fixed_note_and_metadata(monkeypatch):
    (answer, _c, _d, _i, extra), _ = await _run(
        monkeypatch, _output("Sunny, probably.", DEGRADED)
    )
    assert answer == (
        "Sunny, probably.\n\nNote: get_weather was unavailable, "
        "so this answer may be incomplete."
    )
    assert extra["tool_recovery"] == DEGRADED


@pytest.mark.asyncio
async def test_empty_degraded_answer_gets_no_note(monkeypatch):
    # An empty answer must stay empty so the auto-route can still fall back.
    (answer, *_), _ = await _run(monkeypatch, _output("", DEGRADED))
    assert answer == ""


@pytest.mark.asyncio
async def test_no_recovery_leaves_answer_and_metadata_alone(monkeypatch):
    (answer, _c, _d, _i, extra), _ = await _run(monkeypatch, _output("fine", None))
    assert answer == "fine" and "tool_recovery" not in extra


@pytest.mark.asyncio
async def test_on_escalation_reaches_the_loop(monkeypatch):
    async def on_escalation(request):
        return None

    _, run = await _run(monkeypatch, _output("fine", None), on_escalation=on_escalation)
    assert run.await_args.kwargs["on_escalation"] is on_escalation


@pytest.mark.asyncio
async def test_stopped_answer_survives_the_explicit_mode_fallback(monkeypatch):
    (answer, _c, _d, _i, extra), _ = await _run(
        monkeypatch, _output(STOPPED_TEXT, STOPPED)
    )
    assert (answer or extra.pop("_assistant_fallback", "")) == STOPPED_TEXT


# ---------------------------------------------------------------------------
# /tool/send-tool-message — JSON and streaming
# ---------------------------------------------------------------------------


def test_send_tool_message_json_carries_tool_recovery(monkeypatch):
    from tests.unit.test_tool_backend import _make_app
    from src.internal.servers.web import tool_agent_runner

    async def fake_run_tool_agent(query, *, on_turn=None, on_approval=None, **kw):
        return (
            STOPPED_TEXT,
            [],
            [],
            "tool",
            {
                "tool_calls": [],
                "num_turns": 1,
                "truncated": False,
                "tool_recovery": STOPPED,
            },
        )

    monkeypatch.setattr(tool_agent_runner, "_run_tool_agent", fake_run_tool_agent)

    client = TestClient(_make_app(with_model=True))
    resp = client.post(
        "/tool/send-tool-message", json={"message": "send it", "stream": False}
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["answer"] == STOPPED_TEXT
    assert body["tool_recovery"] == STOPPED


def test_send_tool_message_stream_done_event_carries_tool_recovery(monkeypatch):
    from tests.unit.test_tool_backend import _make_app
    from src.internal.servers.web import tool_agent_runner

    async def fake_run_tool_agent(query, *, on_turn=None, on_approval=None, **kw):
        return (
            STOPPED_TEXT,
            [],
            [],
            "tool",
            {
                "tool_calls": [],
                "num_turns": 1,
                "truncated": False,
                "tool_recovery": STOPPED,
            },
        )

    monkeypatch.setattr(tool_agent_runner, "_run_tool_agent", fake_run_tool_agent)

    client = TestClient(_make_app(with_model=True))
    with client.stream(
        "POST",
        "/tool/send-tool-message",
        json={"message": "send it", "stream": True},
    ) as resp:
        assert resp.status_code == 200
        events = [
            json.loads(line[len("data:") :].strip())
            for line in resp.iter_lines()
            if line.startswith("data:")
        ]

    done = events[-1]
    assert done["type"] == "done"
    assert done["tool_recovery"] == STOPPED


# ---------------------------------------------------------------------------
# /api/agent — explicit mode="tool_agent"
# ---------------------------------------------------------------------------


def test_explicit_tool_agent_mode_surfaces_tool_recovery(monkeypatch, tmp_path):
    fake_output = AgentLoopOutput(
        prompt_ids=[],
        response_ids=[],
        response_mask=[],
        num_turns=1,
        final_answer=STOPPED_TEXT,
        action_trace="",
        trajectory_messages=[],
        tool_recovery=STOPPED,
    )
    run_mock = AsyncMock(return_value=fake_output)
    monkeypatch.setattr("src.agents.tool.tool_calling.ToolAgentLoop.run", run_mock)

    app = create_web_app(SearchExperienceSettings(db_path=tmp_path / "db.sqlite3"))
    with TestClient(app) as client:
        app.state.search_agent_manager = MagicMock()
        app.state.search_agent_tokenizer = MagicMock()
        response = client.post(
            "/api/agent", json={"query": "send it", "mode": "tool_agent"}
        )

    assert response.status_code == 200
    data = response.json()
    assert data["answer"] == STOPPED_TEXT
    assert data["hook_metadata"]["tool_recovery"] == STOPPED
