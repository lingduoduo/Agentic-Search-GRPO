"""Pins the consolidated tools package surface at src.internal.tools."""

import importlib

import pytest

_RE_EXPORTS = (
    "Tool",
    "FunctionTool",
    "ToolSchema",
    "ToolEffect",
    "ToolEntry",
    "ToolRegistry",
    "tool",
    "tool_registry",
    "FunctionCall",
    "ToolParser",
    "HermesToolParser",
    "JSONToolParser",
    "Llama3ToolParser",
    "ApiRequestTool",
    "ApiToolError",
    "ApiToolNotFoundError",
    "ApiToolProviderSpec",
    "ApiToolRegistry",
    "ApiToolSpec",
    "SearchPage",
    "fetch_pages_concurrently",
    "fetch_url",
    "format_search_pages",
    "search_tool",
    "MultiQueryWebSearchTool",
    "serper_dev_search",
    "OpenAPISchema",
    "ParameterIn",
    "ParameterType",
    "ParameterTypeMap",
    "build_search_routing_tool",
    "build_rag_routing_tool",
    "ChatTool",
)

_SUBMODULES = (
    "api",
    "base",
    "built_in_tools",
    "html_text",
    "interface",
    "knowledge_base",
    "openapi_schema",
    "parsers",
    "registry",
    "routing_tools",
    "search",
    "semantic_router",
    "validation",
)


def test_framework_surface_re_exported_from_internal_tools():
    mod = importlib.import_module("src.internal.tools")
    missing = [name for name in _RE_EXPORTS if not hasattr(mod, name)]
    assert missing == [], f"missing re-exports: {missing}"


@pytest.mark.parametrize("name", _SUBMODULES)
def test_submodule_lives_under_internal_tools(name):
    module = importlib.import_module(f"src.internal.tools.{name}")
    assert module.__name__ == f"src.internal.tools.{name}"


def test_legacy_src_tools_package_is_gone():
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("src.tools")


_CHAT_MODELS = (
    "ChatFile",
    "SearchToolUsage",
    "ToolCallInfo",
    "ToolCallKickoff",
    "ToolResponse",
)


@pytest.mark.parametrize("name", _CHAT_MODELS)
def test_chat_models_not_re_exported_from_tools(name):
    """These belong to src.internal.chat.tool_models, not the tools package."""
    mod = importlib.import_module("src.internal.tools")
    assert not hasattr(mod, name), f"{name} should not be re-exported here"


@pytest.mark.parametrize("name", _CHAT_MODELS)
def test_chat_models_importable_from_canonical_home(name):
    mod = importlib.import_module("src.internal.chat.tool_models")
    assert hasattr(mod, name)
