"""MCP tools as an entrance: remote MCP servers become callable ToolRegistry tools."""

from __future__ import annotations

from contextlib import asynccontextmanager
from types import SimpleNamespace

import pytest

from src.internal.tools import mcp_client
from src.internal.tools.mcp_client import (
    McpServerSpec,
    parse_mcp_servers,
    register_mcp_tools,
)
from src.internal.tools.registry import ToolRegistry


class _FakeSession:
    """Stands in for mcp.ClientSession."""

    def __init__(self, tools, *, call_result="remote result", fail_call=False):
        self._tools = tools
        self._call_result = call_result
        self._fail_call = fail_call
        self.calls: list[tuple[str, dict]] = []

    async def initialize(self):
        return None

    async def list_tools(self):
        return SimpleNamespace(tools=self._tools)

    async def call_tool(self, name, arguments):
        self.calls.append((name, arguments))
        if self._fail_call:
            return SimpleNamespace(
                isError=True, content=[SimpleNamespace(type="text", text="boom")]
            )
        return SimpleNamespace(
            isError=False,
            content=[SimpleNamespace(type="text", text=self._call_result)],
        )


def _tool(name, description="", schema=None):
    return SimpleNamespace(
        name=name,
        description=description,
        inputSchema=schema or {"type": "object", "properties": {}},
    )


@pytest.fixture
def fake_session(monkeypatch):
    """Patch the session factory; return the single session every connect yields."""
    session = _FakeSession(
        [
            _tool("save_memory", "Save a memory.", {"type": "object"}),
            _tool("ask_agentic_search", "Run the agent."),
        ]
    )

    @asynccontextmanager
    async def _connect(spec):
        yield session

    monkeypatch.setattr(mcp_client, "_connect", _connect)
    return session


# ---------------------------------------------------------------------------
# Config parsing
# ---------------------------------------------------------------------------


def test_parse_mcp_servers_reads_name_url_pairs():
    specs = parse_mcp_servers("local=http://localhost:8090/, gh=http://x:9/mcp")
    assert [(s.name, s.url) for s in specs] == [
        ("local", "http://localhost:8090/"),
        ("gh", "http://x:9/mcp"),
    ]


def test_parse_mcp_servers_empty_is_off():
    assert parse_mcp_servers(None) == []
    assert parse_mcp_servers("   ") == []


def test_parse_mcp_servers_skips_malformed_entries():
    specs = parse_mcp_servers("no-equals-sign, ok=http://x/")
    assert [s.name for s in specs] == ["ok"]


def test_parse_mcp_servers_applies_token():
    specs = parse_mcp_servers("a=http://x/", token="secret")
    assert specs[0].headers == {"Authorization": "Bearer secret"}


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_registers_remote_tools_under_the_mcp_source(fake_session):
    registry = ToolRegistry()
    count = await register_mcp_tools(
        registry, [McpServerSpec(name="local", url="http://x/")]
    )

    assert count == 2
    entry = registry._entries["save_memory"]
    assert entry.source == "mcp"  # so the MCP mirror skips it — no export loop
    assert entry.provider_id == "local"
    assert entry.provider_name == "local"
    assert entry.tool.schema.description == "Save a memory."


@pytest.mark.asyncio
async def test_invoking_a_registered_tool_calls_the_remote_server(fake_session):
    registry = ToolRegistry()
    await register_mcp_tools(registry, [McpServerSpec(name="local", url="http://x/")])

    response, _raw, errors = await registry.invoke("save_memory", {"text": "hello"})

    assert errors == []
    assert response == "remote result"
    assert fake_session.calls == [("save_memory", {"text": "hello"})]


@pytest.mark.asyncio
async def test_remote_tool_error_surfaces_as_text(monkeypatch):
    session = _FakeSession([_tool("save_memory")], fail_call=True)

    @asynccontextmanager
    async def _connect(spec):
        yield session

    monkeypatch.setattr(mcp_client, "_connect", _connect)
    registry = ToolRegistry()
    await register_mcp_tools(registry, [McpServerSpec(name="local", url="http://x/")])

    response, _raw, _errors = await registry.invoke("save_memory", {})
    assert "boom" in response


@pytest.mark.asyncio
async def test_unreachable_server_is_skipped_not_fatal(monkeypatch):
    @asynccontextmanager
    async def _boom(spec):
        raise ConnectionError("no route to host")
        yield  # pragma: no cover

    monkeypatch.setattr(mcp_client, "_connect", _boom)
    registry = ToolRegistry()

    count = await register_mcp_tools(
        registry, [McpServerSpec(name="down", url="http://nope/")]
    )

    assert count == 0
    assert registry.list_tools() == []


@pytest.mark.asyncio
async def test_no_servers_configured_is_a_no_op(monkeypatch):
    registry = ToolRegistry()
    assert await register_mcp_tools(registry, []) == 0


# ---------------------------------------------------------------------------
# Guards
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_mcp_tools_are_not_mirrored_back_out_to_mcp(fake_session):
    # dynamic._sync_all exports the registry to MCP. Re-exporting a server's own
    # tools to itself would duplicate them on every restart.
    from src.internal.mcp_server.tools import dynamic

    registry = ToolRegistry()
    await register_mcp_tools(registry, [McpServerSpec(name="local", url="http://x/")])

    assert [e.tool.name for e in dynamic._exportable_entries(registry)] == []


@pytest.mark.asyncio
async def test_agent_never_sees_the_recursive_mcp_tool(fake_session):
    # ask_agentic_search runs an agent; offering it to the agent lets it call
    # itself. The exclusion is a registration property, not a downstream name
    # match, so a rename cannot silently disable it.
    registry = ToolRegistry()
    await register_mcp_tools(
        registry,
        [
            McpServerSpec(
                name="local",
                url="http://x/",
                agent_exclude=frozenset({"ask_agentic_search"}),
            )
        ],
    )

    assert "ask_agentic_search" not in [t.name for t in registry.agent_tools()]
    assert registry.get("ask_agentic_search") is not None


@pytest.mark.asyncio
async def test_dev_console_groups_mcp_tools_by_server(fake_session):
    from src.internal.tools.semantic_router import catalog_from_registry

    registry = ToolRegistry()
    await register_mcp_tools(
        registry, [McpServerSpec(name="memories", url="http://x/")]
    )

    catalog = catalog_from_registry(registry)
    assert [s.name for s in catalog] == ["memories"]
    assert {t.name for t in catalog[0].tools} == {"save_memory", "ask_agentic_search"}


@pytest.mark.asyncio
async def test_servers_are_discovered_concurrently(monkeypatch):
    """Every server's discovery is in flight at once.

    Guards the fan-out: sweeping servers one await at a time made each
    unreachable server's connect timeout delay the next server's discovery.
    """
    import asyncio

    in_flight = 0
    peak = 0

    @asynccontextmanager
    async def _slow_connect(spec):
        nonlocal in_flight, peak
        in_flight += 1
        peak = max(peak, in_flight)
        await asyncio.sleep(0.02)
        in_flight -= 1
        yield _FakeSession([_tool(f"{spec.name}_tool")])

    monkeypatch.setattr(mcp_client, "_connect", _slow_connect)
    registry = ToolRegistry()
    specs = [McpServerSpec(name=f"s{i}", url=f"http://x{i}/") for i in range(3)]

    count = await register_mcp_tools(registry, specs)

    assert count == 3
    assert peak > 1, f"server discovery was serialized (peak concurrency {peak})"
