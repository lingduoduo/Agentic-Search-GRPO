from __future__ import annotations

import httpx
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.internal.servers.web.debug_router import create_debug_router
from src.internal.tools.base import FunctionTool, ResultKind, ToolEffect
from src.internal.tools.registry import tool_registry


def _client() -> TestClient:
    http_client = httpx.Client(
        transport=httpx.MockTransport(lambda r: httpx.Response(200, json={}))
    )
    app = FastAPI()
    app.include_router(
        create_debug_router(
            search_url="http://retrieval:8001/retrieve", http_client=http_client
        )
    )
    return TestClient(app)


def test_tools_lists_registered_and_catalog():
    stub = FunctionTool(
        lambda query: "ok",
        name="stub_tool_dbg",
        description="a stub",
        parameters={},
        effect=ToolEffect.READ_ONLY,
        result_kind=ResultKind.TEXT,
    )
    tool_registry.register(stub, source="function")
    try:
        resp = _client().get("/api/debug/tools")
        assert resp.status_code == 200
        data = resp.json()
        assert "stub_tool_dbg" in {t["name"] for t in data["registered"]}
        local = next(s for s in data["catalog"] if s["name"] == "local")
        assert any(t["name"] == "stub_tool_dbg" for t in local["tools"])
    finally:
        tool_registry.unregister("stub_tool_dbg")


def test_tools_discover_ranks_relevant_tool():
    stub = FunctionTool(
        lambda query: "ok",
        name="wikipedia_search_dbg",
        description="Search Wikipedia for encyclopedia articles about a topic.",
        parameters={},
        effect=ToolEffect.READ_ONLY,
        result_kind=ResultKind.TEXT,
    )
    tool_registry.register(stub, source="function")
    try:
        resp = _client().post(
            "/api/debug/tools/discover", json={"query": "search wikipedia articles"}
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["request"] == "search wikipedia articles"
        assert "stage1_servers" in data and "final_tools" in data
        assert any(t["name"] == "wikipedia_search_dbg" for t in data["final_tools"])
    finally:
        tool_registry.unregister("wikipedia_search_dbg")


def test_tools_discover_blank_query_is_422():
    resp = _client().post("/api/debug/tools/discover", json={"query": "   "})
    assert resp.status_code == 422
