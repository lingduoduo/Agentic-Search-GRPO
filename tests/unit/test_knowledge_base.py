from __future__ import annotations

from src.internal.tools.base import Tool
from src.internal.tools.knowledge_base import seed_tools, tool_knowledge_base
from src.internal.tools.registry import ToolRegistry
from src.internal.tools.semantic_router import catalog_from_registry

# One corpus search (`search`), not two over the same corpus, plus the nine
# keyless public data tools that are seeded unconditionally alongside it.
PUBLIC_DATA_NAMES = {
    "search_wikipedia",
    "search_arxiv",
    "search_wayback",
    "get_weather",
    "get_stock_quote",
    "get_crypto_price",
    "convert_currency",
    "search_location",
    "search_nearby_places",
}
DOMAIN_FEATURE_NAMES = {
    "search_domain",
    "get_sub_domains",
    "extract_page",
    "batch_search",
}
BUILTIN_NAMES = {"web_search", "search"} | PUBLIC_DATA_NAMES | DOMAIN_FEATURE_NAMES


def test_knowledge_base_default_has_the_builtin_tools():
    tools = tool_knowledge_base()
    assert all(isinstance(t, Tool) for t in tools)
    assert {t.name for t in tools} == BUILTIN_NAMES


def test_knowledge_base_adds_rag_tool_when_llm_present():
    tools = tool_knowledge_base(llm=object())
    assert {t.name for t in tools} == BUILTIN_NAMES | {"rag_routing_tool"}


def test_seed_tools_registers_into_fresh_registry():
    reg = ToolRegistry()
    count = seed_tools(reg)
    assert count == len(BUILTIN_NAMES)
    assert reg.get("web_search") is not None
    # Built-ins register under source="function".
    assert all(e.source == "function" for e in reg.list())


def test_seeded_registry_surfaces_in_catalog_from_registry():
    reg = ToolRegistry()
    seed_tools(reg)
    catalog = catalog_from_registry(reg)
    # Function tools group into a single "local" server.
    by_name = {s.name: s for s in catalog}
    assert set(by_name) == {"local"}
    assert {t.name for t in by_name["local"].tools} == BUILTIN_NAMES


def test_seed_tools_accepts_explicit_tools():
    reg = ToolRegistry()
    tools = tool_knowledge_base(llm=object())
    assert seed_tools(reg, tools=tools) == len(BUILTIN_NAMES) + 1
    assert reg.get("rag_routing_tool") is not None


def test_the_seeded_corpus_search_is_not_agent_callable():
    # It is built at process start with no request identity, so it cannot carry
    # an ACL. Keeping it out of every agent's tool list makes that structural
    # rather than something each call site has to remember.
    from src.internal.tools.knowledge_base import seed_tools, tool_knowledge_base
    from src.internal.tools.registry import ToolRegistry

    reg = ToolRegistry()
    seed_tools(reg, tools=tool_knowledge_base())

    assert reg.get("search") is not None  # still listed and invocable
    assert "search" not in [t.name for t in reg.agent_tools()]
    assert "web_search" in [t.name for t in reg.agent_tools()]
