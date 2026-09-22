"""The second agent framework and the connector-checkpoint surface stay retired.

``BaseAgent`` (``src/agents/core/graph_base.py``) was a complete Pydantic +
threading + Redis-queue streaming framework with **no subclass anywhere in
src/**. Its only consumers were its own unit test and two docstrings. The live
framework is ``AgentLoopBase`` + the loop registry, which streams through
``on_token``/``on_turn`` callbacks instead.

``ConnectorCheckpoint`` described incremental connector sync. It was reachable
only through package re-exports, and the async worker fleet it belonged to was
removed with the rest of the Onyx heritage.

Both are pinned here because "nothing imports it" is exactly the condition that
let them drift out of reality unnoticed in the first place.
"""

from __future__ import annotations

import importlib

import pytest


@pytest.mark.parametrize(
    "module",
    [
        "src.agents.core.graph_base",
        "src.internal.chat.queue_manager",
    ],
)
def test_the_second_agent_framework_is_gone(module: str) -> None:
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module(module)


def test_connector_checkpoint_is_not_exported() -> None:
    import src
    from src.internal import connectors

    assert not hasattr(src, "ConnectorCheckpoint")
    assert not hasattr(connectors, "ConnectorCheckpoint")

    models = importlib.import_module("src.internal.connectors.models")
    assert not hasattr(models, "ConnectorCheckpoint")


def test_the_live_agent_framework_still_works() -> None:
    """The removals must not touch the framework that is actually used."""
    from src import get_registered_agent_loop, list_registered_agent_loops

    registered = list_registered_agent_loops()
    assert {
        "plain_generation",
        "single_turn_agent",
        "search_agent",
        "tool_agent",
    } <= set(registered)
    assert get_registered_agent_loop("search_agent") is not None


def test_connector_models_that_are_live_survive() -> None:
    from src.internal.connectors import Document, SlimDocument

    assert Document is not None and SlimDocument is not None
