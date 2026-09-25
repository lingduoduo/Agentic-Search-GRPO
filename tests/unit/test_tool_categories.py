"""Category flags (citeable) on the canonical Tool."""

from __future__ import annotations

from src.internal.tools.base import FunctionTool
from src.internal.tools.registry import ToolRegistry
from src.internal.tools.search import MultiQueryWebSearchTool


def test_functiontool_defaults_false():
    t = FunctionTool(lambda: "ok", name="x")
    assert t.citeable is False


def test_functiontool_flags_settable():
    t = FunctionTool(lambda: "ok", name="x", citeable=True)
    assert t.citeable is True


def test_flags_not_leaked_into_schema():
    t = FunctionTool(lambda: "ok", name="x", citeable=True)
    fn = t.schema.to_dict()["function"]
    assert "citeable" not in fn


def test_tool_decorator_threads_flags():
    reg = ToolRegistry()

    @reg.tool(description="d", citeable=True)
    def search_ish(q: str) -> str:
        return q

    assert reg.get("search_ish").citeable is True


def test_real_search_tools_are_citeable():
    assert MultiQueryWebSearchTool().citeable is True
