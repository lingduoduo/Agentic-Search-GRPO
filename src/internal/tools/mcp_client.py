"""MCP as an entrance: make remote MCP servers' tools callable from this process.

The MCP bridge in ``src/internal/mcp_server`` runs the other way — it mirrors
this process's ``tool_registry`` outward so external hosts (Claude Desktop,
Cursor) can call our tools. Nothing pulled the other direction, so the tool
agent could only ever call the retrieval wrappers seeded locally.

This module closes that gap. Configured servers are contacted at startup, their
tools are wrapped as ``FunctionTool``s and registered under ``source="mcp"``,
and from then on they are ordinary registry tools: listed by ``/admin/tools``,
callable by the tool agent, invocable from the Dev Console.

Two hazards this deliberately guards against:

* **Export loop.** ``mcp_server.tools.dynamic`` mirrors the registry back out to
  MCP. Entries registered here carry ``source="mcp"`` so that mirror skips them
  instead of re-exporting a server's own tools to itself.
* **Recursion.** A server may expose a tool that runs an agent
  (``ask_agentic_search`` does). The tool agent's own shadow list excludes it,
  so the agent cannot call itself.

Connections are per call rather than long-lived: discovery opens one session,
and each invocation opens its own. That costs an HTTP round trip per call and
buys a web process that neither babysits a socket nor dies when an MCP server
restarts.
"""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any

from ..configs.timeouts import get_timeout_policies
from .base import (
    FailureCategory,
    FunctionTool,
    ResultKind,
    Tool,
    ToolErrorText,
    ToolFailure,
)
from .registry import ToolRegistry

logger = logging.getLogger(__name__)

MCP_SOURCE = "mcp"

# Remote tools that re-enter an agent. Offering one to the agent lets the agent
# call itself, and the protocol exposes no signal for "this runs an agent", so
# the names have to be configured. Override with AGENTIC_SEARCH_MCP_AGENT_EXCLUDE.
DEFAULT_AGENT_EXCLUDE: frozenset[str] = frozenset({"ask_agentic_search"})

# Remote tools backed by per-user storage. Offered only when a caller has an
# identity to scope them to; anonymously they would write to a shared bucket.
DEFAULT_USER_SCOPED: frozenset[str] = frozenset(
    {
        "save_memory",
        "update_memory_from_conversation",
        "generate_user_profile",
        "get_user_profile",
        "search_memories",
        "consolidate_memories",
    }
)


@dataclass(frozen=True)
class McpServerSpec:
    """A remote MCP server to pull tools from."""

    name: str
    url: str
    headers: dict[str, str] = field(default_factory=dict)
    # Tool names this server exposes that no agent loop may be offered.
    agent_exclude: frozenset[str] = frozenset()
    # Tool names this server exposes that are backed by per-user storage.
    user_scoped: frozenset[str] = frozenset()


def parse_mcp_servers(
    raw: str | None,
    *,
    token: str | None = None,
    agent_exclude: str | None = None,
    user_scoped: str | None = None,
) -> list[McpServerSpec]:
    """Parse ``"name=url, name=url"`` into specs. Empty or unset means disabled.

    Malformed entries are skipped with a warning rather than failing startup —
    one typo in an env var should not take the web process down.
    """
    if not raw or not raw.strip():
        return []
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    excluded = (
        frozenset(n.strip() for n in agent_exclude.split(",") if n.strip())
        if agent_exclude is not None
        else DEFAULT_AGENT_EXCLUDE
    )
    scoped = (
        frozenset(n.strip() for n in user_scoped.split(",") if n.strip())
        if user_scoped is not None
        else DEFAULT_USER_SCOPED
    )
    specs: list[McpServerSpec] = []
    for chunk in raw.split(","):
        entry = chunk.strip()
        if not entry:
            continue
        name, sep, url = entry.partition("=")
        if not sep or not name.strip() or not url.strip():
            logger.warning(
                "Ignoring malformed MCP server entry %r (expected name=url)", entry
            )
            continue
        specs.append(
            McpServerSpec(
                name=name.strip(),
                url=url.strip(),
                headers=dict(headers),
                agent_exclude=excluded,
                user_scoped=scoped,
            )
        )
    return specs


@asynccontextmanager
async def _connect(spec: McpServerSpec):
    """Yield an initialised MCP ClientSession for *spec*.

    Patched wholesale in tests; keep it free of logic worth testing.
    """
    from mcp import ClientSession
    from mcp.client.streamable_http import streamablehttp_client

    policy = get_timeout_policies().tools.mcp
    async with streamablehttp_client(
        spec.url,
        headers=spec.headers or None,
        timeout=policy.timeout_seconds,
        sse_read_timeout=policy.sse_read_timeout_seconds,
    ) as (
        read,
        write,
        _get_session_id,
    ):
        async with ClientSession(read, write) as session:
            await session.initialize()
            yield session


def _result_text(result: Any) -> str:
    """Flatten an MCP CallToolResult into text for the model."""
    parts = [
        getattr(block, "text", "")
        for block in (getattr(result, "content", None) or [])
        if getattr(block, "text", "")
    ]
    text = "\n".join(parts)
    if getattr(result, "isError", False):
        message = f"Error: {text}" if text else "Error: tool call failed."
        return ToolErrorText(
            message,
            ToolFailure(FailureCategory.UNKNOWN, "remote tool reported an error"),
        )
    return text


def _build_tool(spec: McpServerSpec, remote: Any) -> Tool:
    """Wrap one remote MCP tool as a locally callable FunctionTool."""

    async def _call(**arguments: Any) -> str:
        async with _connect(spec) as session:
            result = await session.call_tool(remote.name, arguments)
        return _result_text(result)

    # effect is left UNSPECIFIED on purpose: a remote tool may write, and only
    # READ_ONLY tools bypass the approval gate.
    return FunctionTool(
        _call,
        name=remote.name,
        description=getattr(remote, "description", "") or remote.name,
        parameters=getattr(remote, "inputSchema", None) or {},
        result_kind=ResultKind.TEXT,
    )


async def register_mcp_tools(registry: ToolRegistry, specs: list[McpServerSpec]) -> int:
    """Discover every server's tools and register them; return the count.

    A server that cannot be reached is logged and skipped — MCP is additive, so
    an unreachable one must not stop the web process from starting.
    """

    async def _discover(spec: McpServerSpec) -> list[Any]:
        async with _connect(spec) as session:
            listing = await session.list_tools()
            return list(getattr(listing, "tools", None) or [])

    # Discover every server at once: servers are independent, and a sequential
    # sweep made each unreachable one's connect timeout delay the next server's
    # discovery. Registration below still walks specs in order.
    listings = await asyncio.gather(
        *(_discover(spec) for spec in specs), return_exceptions=True
    )

    registered = 0
    for spec, remote_tools in zip(specs, listings):
        if isinstance(remote_tools, BaseException):
            # any transport failure is non-fatal
            logger.warning(
                "MCP server %r unreachable at %s: %s",
                spec.name,
                spec.url,
                remote_tools,
            )
            continue
        for remote in remote_tools:
            try:
                registry.register(
                    _build_tool(spec, remote),
                    source=MCP_SOURCE,
                    provider_id=spec.name,
                    agent_callable=remote.name not in spec.agent_exclude,
                    user_scoped=remote.name in spec.user_scoped,
                )
            except ValueError as exc:
                logger.warning("skipping MCP tool %s: %s", remote.name, exc)
                continue
            registered += 1
        logger.info(
            "MCP server %r: registered %d tool(s)", spec.name, len(remote_tools)
        )
    return registered


__all__ = [
    "DEFAULT_AGENT_EXCLUDE",
    "DEFAULT_USER_SCOPED",
    "MCP_SOURCE",
    "McpServerSpec",
    "parse_mcp_servers",
    "register_mcp_tools",
]
