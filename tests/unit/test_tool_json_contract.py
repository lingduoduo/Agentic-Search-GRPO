"""Every registered tool answers the model with a JSON string.

#597 was merged claiming the opposite -- that the public data tools returned
Python reprs and that two downstream consumers were silently failing. They did
not: `guarded` (public_data/_http.py) has always serialized with json.dumps.
That claim was never machine-checked, only asserted in prose, so it survived
review. These tests check it against the real registry.
"""

import asyncio
import json
import os

import pytest

from src.internal.tools.base import FunctionTool
from src.internal.tools.knowledge_base import tool_knowledge_base

# `search` is the one registered tool not wrapped by `guarded`: it is assembled
# in routing_tools.py and relies on ToolRegistry.invoke validating arguments
# against its schema before dispatch, so a bad call never reaches it. Recorded
# here rather than hidden, so adding a second unguarded tool fails loudly.
UNGUARDED: dict[str, str] = {
    "search": "built in routing_tools.py; arguments validated by the registry before dispatch",
}


def _registered() -> list[FunctionTool]:
    return [t for t in tool_knowledge_base() if isinstance(t, FunctionTool)]


def _is_guarded(tool: FunctionTool) -> bool:
    """True when the tool routes through `guarded`.

    Checked by code-object origin rather than by __name__, __qualname__ or
    __annotations__ -- functools.wraps copies all three from the inner function
    onto the wrapper, which is exactly how #597's survey concluded these tools
    returned dicts when the wrapper returns str.
    """
    fn = getattr(tool, "_fn")
    return os.path.basename(fn.__code__.co_filename) == "_http.py"


def _execute(tool: FunctionTool, arguments: dict) -> str:
    async def run():
        instance_id = await tool.create()
        try:
            response, _raw, _meta = await tool.execute(instance_id, arguments)
            return response
        finally:
            await tool.release(instance_id)

    return asyncio.run(run())


@pytest.mark.parametrize("tool", _registered(), ids=lambda t: t.name)
def test_every_guarded_tool_answers_with_json(tool):
    """An unexpected argument lands inside guarded's try, so this needs no network."""
    if tool.name in UNGUARDED:
        pytest.skip(f"{tool.name}: {UNGUARDED[tool.name]}")

    response = _execute(tool, {"__unexpected_argument__": 1})

    assert isinstance(response, str)
    payload = json.loads(response)  # the assertion: it must parse
    assert "error" in payload


def test_the_unguarded_set_is_exactly_what_we_recorded():
    """A new tool added without `guarded` must fail here, not surprise someone later."""
    unguarded = sorted(t.name for t in _registered() if not _is_guarded(t))

    assert unguarded == sorted(UNGUARDED), (
        f"unguarded tools changed: {unguarded}. A tool outside `guarded` does not "
        "serialize its result and does not convert failures into {'error': ...}. "
        "Wrap it with guarded(), or add it to UNGUARDED with the reason."
    )


def test_wraps_makes_annotations_unreliable_for_this_check():
    """Pins the trap that produced #597's false premise.

    The guarded wrapper returns `str`, but functools.wraps copies the inner
    function's `-> dict` annotation onto it. Anything inferring the tool
    contract from annotations gets the wrong answer.
    """
    import typing

    quote = next(t for t in _registered() if t.name == "get_stock_quote")
    fn = getattr(quote, "_fn")

    annotated = typing.get_type_hints(fn).get("return")
    assert annotated is dict, "the wrapper still advertises the inner annotation"

    actual = _execute(quote, {"__unexpected_argument__": 1})
    assert isinstance(actual, str), "but it actually returns a str"
