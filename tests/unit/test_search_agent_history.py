"""Unit tests for search-agent conversation-history message building."""

from __future__ import annotations

from src.context import ChatMessage
from src.internal.servers.web.app import (
    SEARCH_AGENT_HISTORY_MESSAGES,
    _build_search_agent_messages,
)


def test_empty_history_yields_just_the_query():
    msgs = _build_search_agent_messages("what is FAISS?", [])
    assert msgs == [{"role": "user", "content": "what is FAISS?"}]


def test_short_history_prepended_then_query_last():
    history = [
        ChatMessage(role="user", content="hi"),
        ChatMessage(role="assistant", content="hello"),
    ]
    msgs = _build_search_agent_messages("next question", history)
    assert msgs == [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
        {"role": "user", "content": "next question"},
    ]


def test_long_history_capped_to_last_n_plus_query():
    history = [
        ChatMessage(role=("user" if i % 2 == 0 else "assistant"), content=f"m{i}")
        for i in range(20)
    ]
    msgs = _build_search_agent_messages("q", history)
    # Last SEARCH_AGENT_HISTORY_MESSAGES prior turns, then the query.
    assert len(msgs) == SEARCH_AGENT_HISTORY_MESSAGES + 1
    assert msgs[-1] == {"role": "user", "content": "q"}
    assert [m["content"] for m in msgs[:-1]] == [
        f"m{i}" for i in range(20 - SEARCH_AGENT_HISTORY_MESSAGES, 20)
    ]


def test_cap_default_is_six():
    assert SEARCH_AGENT_HISTORY_MESSAGES == 6
