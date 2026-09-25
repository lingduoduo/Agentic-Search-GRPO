import importlib

import pytest


@pytest.mark.parametrize(
    "module",
    [
        "src.internal.tools.built_in_tools",
        "src.internal.tools.interface",
        "src.internal.chat.tool_call_args_streaming",
    ],
)
def test_retired_modules_are_gone(module):
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module(module)


def test_stopping_is_retired():
    from src.internal.tools import FunctionTool, Tool

    assert not hasattr(Tool, "stopping")
    with pytest.raises(TypeError):
        FunctionTool(lambda: "x", name="t", stopping=True)


def test_tool_safety_is_tool_effect():
    from src.context import ToolSafety
    from src.internal.tools import ToolEffect

    assert ToolSafety is ToolEffect


def test_admin_surface_counts_citeable_tools_from_the_registry(monkeypatch):
    import src.internal.observability.admin_surface as surface
    from src.internal.tools import FunctionTool, ResultKind, ToolEffect, ToolRegistry

    registry = ToolRegistry()
    registry.register(
        FunctionTool(
            lambda: "[]",
            name="a",
            effect=ToolEffect.READ_ONLY,
            result_kind=ResultKind.DOCUMENTS,
            citeable=True,
        )
    )
    registry.register(
        FunctionTool(
            lambda: "{}",
            name="b",
            effect=ToolEffect.READ_ONLY,
            result_kind=ResultKind.JSON,
        )
    )
    monkeypatch.setattr(surface, "tool_registry", registry)
    assert surface.citeable_tool_count() == 1
