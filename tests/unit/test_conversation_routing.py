"""Fallback routing distinguishes discussion from requested tool actions."""

import pytest

from src.internal.servers.web.intent.rules import _rule_based_route_or_none
from src.internal.servers.web.intent.types import RouteStrategy


@pytest.mark.parametrize(
    "query",
    [
        "Explain how to send an email",
        "What is required to create a ticket?",
        "Draft an email",
        "Write a guide to schedule meetings",
        "Compare tools that create tickets",
    ],
)
def test_conversation_tool_subject_is_chat(query):
    assert _rule_based_route_or_none(query) is RouteStrategy.CHAT


@pytest.mark.parametrize(
    "query",
    ["Send an email", "Create a ticket", "Can you schedule a meeting?"],
)
def test_requested_action_remains_tool(query):
    assert _rule_based_route_or_none(query) is RouteStrategy.TOOL


def test_search_for_tool_documentation_remains_search():
    assert (
        _rule_based_route_or_none("Find documentation on how to create tickets")
        is RouteStrategy.SEARCH
    )
