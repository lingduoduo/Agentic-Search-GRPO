"""Built-in executable tools that seed the global ToolRegistry.

The ToolRegistry is the single source of truth for this process's runnable
tools. ``tool_knowledge_base()`` is the built-in seed set; ``seed_tools()``
registers it. OpenAPI tools are added separately at runtime via
``register_from_openapi``. MCP-native tools live in the MCP server process and
are not registered here.
"""

from __future__ import annotations

import os

from .base import Tool
from .domain_search import DomainSearch
from .domain_search_tools import build_domain_search_tools
from .public_data import public_data_tools
from .registry import ToolRegistry
from .routing_tools import build_rag_routing_tool, build_search_routing_tool
from .search import MultiQueryWebSearchTool, make_web_cascade_search

DEFAULT_SEARCH_URL = "http://localhost:8000/retrieve"


def tool_knowledge_base(
    *,
    search_url: str = DEFAULT_SEARCH_URL,
    top_k: int = 5,
    llm=None,
) -> list[Tool]:
    """The built-in executable tools that seed the registry.

    ``rag_routing_tool`` is included only when an ``llm`` is supplied (it needs
    a live LLM client).
    """
    # One corpus search, named `search`. It used to be seeded twice — once as a
    # text-returning `search` and once as a JSON-returning `search_routing_tool`
    # — over the same corpus behind the same argument, which left a small model
    # unable to choose between them. The JSON one survives because its result is
    # what becomes source cards.
    web_search_fn = make_web_cascade_search(
        browser_search_url=os.getenv("AGENTIC_SEARCH_BROWSER_SEARCH_URL")
    )
    public_tools = public_data_tools()
    tools: list[Tool] = [
        MultiQueryWebSearchTool(
            search_fn=web_search_fn,
            page_size=top_k,
        ),
        build_search_routing_tool(search_url=search_url, top_k=top_k),
        # Keyless public data sources. They need no configuration, so they are
        # on by default: without them the tool agent has one usable tool and
        # nothing to choose between.
        *public_tools,
        *build_domain_search_tools(
            service=DomainSearch(web_search_fn=web_search_fn, tools=public_tools)
        ),
    ]
    if llm is not None:
        tools.append(
            build_rag_routing_tool(llm=llm, search_url=search_url, top_k=top_k)
        )
    return tools


# Tools no agent loop should be offered, keyed by name at seed time so the
# decision travels with registration instead of being re-derived downstream.
# ``rag_routing_tool`` generates a whole answer rather than returning evidence.
# ``search`` is seeded at process start, where no request identity exists, so
# this instance can carry no ACL; the tool agent builds its own request-bound
# one. This instance stays listed and invocable through /admin/tools.
NOT_AGENT_CALLABLE: frozenset[str] = frozenset({"rag_routing_tool", "search"})


def seed_tools(registry: ToolRegistry, *, tools: list[Tool] | None = None) -> int:
    """Register the built-in tools into *registry*; return the count.

    Uses ``tool_knowledge_base()`` when *tools* is None. Tools register with the
    default ``source="function"``, so the dashboard lists them under "Built-in
    function tools" and ``catalog_from_registry`` groups them into ``local``.
    """
    tools = tool_knowledge_base() if tools is None else tools
    for tool in tools:
        registry.register(tool, agent_callable=tool.name not in NOT_AGENT_CALLABLE)
    return len(tools)
