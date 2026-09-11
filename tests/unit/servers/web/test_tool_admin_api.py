"""HTTP-level tests for /admin/tools/* endpoints."""

from __future__ import annotations

import json

from fastapi.testclient import TestClient

from src.internal.auth import generate_user_jwt_token
from src.internal.configs import AppSettings, AuthSettings
from src.internal.db import UserRecord
from src.internal.servers.web.app import SearchExperienceSettings, create_web_app
from src.internal.tools.registry import tool_registry

_ADMIN = "admin"

_OPENAPI_JSON = json.dumps(
    {
        "openapi": "3.0.0",
        "info": {"title": "Echo API", "description": "Echo words back."},
        "servers": [{"url": "https://echo.test"}],
        "paths": {
            "/echo": {
                "post": {
                    "operationId": "echo_word",
                    "summary": "Echo a word.",
                    "requestBody": {
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {"word": {"type": "string"}},
                                    "required": ["word"],
                                }
                            }
                        }
                    },
                    "responses": {"200": {"description": "OK"}},
                }
            }
        },
    }
)


def _admin_headers() -> dict[str, str]:
    token = generate_user_jwt_token(user_id=_ADMIN)
    return {"Authorization": f"Bearer {token}"}


def _settings() -> AppSettings:
    return AppSettings(auth=AuthSettings(super_users=(_ADMIN,)))


def _make_app(tmp_path):
    app = create_web_app(
        SearchExperienceSettings(db_path=tmp_path / "state.sqlite3"),
        app_settings=_settings(),
    )
    app.state.auth_store.upsert_user(UserRecord(id=_ADMIN))
    return app


def _clear_registry():
    """Snapshot and reset the global registry (both _entries and _api_registry)."""
    from src.internal.tools.api import ApiToolRegistry

    snap_entries = dict(tool_registry._entries)
    snap_api = tool_registry._api_registry
    tool_registry._entries.clear()
    tool_registry._api_registry = ApiToolRegistry()
    return snap_entries, snap_api


def _restore_registry(snapshot):
    snap_entries, snap_api = snapshot
    tool_registry._entries.clear()
    tool_registry._entries.update(snap_entries)
    tool_registry._api_registry = snap_api


# ---------------------------------------------------------------------------
# Auth guard
# ---------------------------------------------------------------------------


def test_list_tools_requires_auth(tmp_path):
    client = TestClient(_make_app(tmp_path))
    assert client.get("/admin/tools").status_code == 401


def test_invoke_tool_requires_auth(tmp_path):
    client = TestClient(_make_app(tmp_path))
    assert (
        client.post("/admin/tools/anything/invoke", json={"arguments": {}}).status_code
        == 401
    )


# ---------------------------------------------------------------------------
# List
# ---------------------------------------------------------------------------


def test_list_tools_empty(tmp_path):
    snap = _clear_registry()
    try:
        client = TestClient(_make_app(tmp_path))
        resp = client.get("/admin/tools", headers=_admin_headers())
        assert resp.status_code == 200
        assert resp.json() == []
    finally:
        _restore_registry(snap)


# ---------------------------------------------------------------------------
# Register OpenAPI
# ---------------------------------------------------------------------------


def test_register_openapi_creates_tools(tmp_path):
    snap = _clear_registry()
    try:
        client = TestClient(_make_app(tmp_path))
        resp = client.post(
            "/admin/tools/openapi",
            json={"name": "Echo API", "openapi_json": _OPENAPI_JSON},
            headers=_admin_headers(),
        )
        assert resp.status_code == 201
        data = resp.json()
        assert data["provider_id"]
        assert "echo_word" in data["tool_names"]

        # Tool appears in list
        names = [
            t["name"]
            for t in client.get("/admin/tools", headers=_admin_headers()).json()
        ]
        assert "echo_word" in names
    finally:
        _restore_registry(snap)


def test_register_openapi_invalid_json_returns_400(tmp_path):
    client = TestClient(_make_app(tmp_path))
    resp = client.post(
        "/admin/tools/openapi",
        json={"name": "Bad", "openapi_json": "not valid json"},
        headers=_admin_headers(),
    )
    assert resp.status_code == 400


# ---------------------------------------------------------------------------
# Get by name
# ---------------------------------------------------------------------------


def test_get_tool_returns_view(tmp_path):
    snap = _clear_registry()
    try:
        client = TestClient(_make_app(tmp_path))
        client.post(
            "/admin/tools/openapi",
            json={"name": "Echo API", "openapi_json": _OPENAPI_JSON},
            headers=_admin_headers(),
        )
        resp = client.get("/admin/tools/echo_word", headers=_admin_headers())
        assert resp.status_code == 200
        assert resp.json()["name"] == "echo_word"
    finally:
        _restore_registry(snap)


def test_get_unknown_tool_returns_404(tmp_path):
    client = TestClient(_make_app(tmp_path))
    assert (
        client.get("/admin/tools/nonexistent", headers=_admin_headers()).status_code
        == 404
    )


# ---------------------------------------------------------------------------
# Invoke
# ---------------------------------------------------------------------------


def test_invoke_function_tool_via_http(tmp_path):
    snap = _clear_registry()
    try:

        def double(n: int) -> int:
            """Double a number."""
            return n * 2

        tool_registry.tool(double)
        client = TestClient(_make_app(tmp_path))
        resp = client.post(
            "/admin/tools/double/invoke",
            json={"arguments": {"n": 7}},
            headers=_admin_headers(),
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["errors"] == []
        assert "14" in data["response"]
    finally:
        _restore_registry(snap)


def test_invoke_unknown_tool_returns_404(tmp_path):
    client = TestClient(_make_app(tmp_path))
    resp = client.post(
        "/admin/tools/ghost/invoke",
        json={"arguments": {}},
        headers=_admin_headers(),
    )
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Delete provider
# ---------------------------------------------------------------------------


def test_delete_openapi_provider(tmp_path):
    snap = _clear_registry()
    try:
        client = TestClient(_make_app(tmp_path))
        reg = client.post(
            "/admin/tools/openapi",
            json={"name": "Echo API", "openapi_json": _OPENAPI_JSON},
            headers=_admin_headers(),
        )
        provider_id = reg.json()["provider_id"]

        assert (
            client.delete(
                f"/admin/tools/openapi/{provider_id}", headers=_admin_headers()
            ).status_code
            == 204
        )

        # Tool is gone
        assert (
            client.get("/admin/tools/echo_word", headers=_admin_headers()).status_code
            == 404
        )

        # Deleting again → 404
        assert (
            client.delete(
                f"/admin/tools/openapi/{provider_id}", headers=_admin_headers()
            ).status_code
            == 404
        )
    finally:
        _restore_registry(snap)


# ---------------------------------------------------------------------------
# Registration flags on the list payload
# ---------------------------------------------------------------------------


def test_list_tools_exposes_agent_callable_and_user_scoped(tmp_path):
    """The two ToolEntry flags must survive the response model.

    They were being dropped: all_summaries() returns them but ToolView did not
    declare them, so a field absent from the model was silently discarded and no
    caller could tell an agent-callable tool from one withheld from agents.
    """
    from src.internal.tools.base import FunctionTool

    snap = _clear_registry()
    try:

        def _plain() -> str:
            return "ok"

        def _hidden() -> str:
            return "ok"

        def _scoped() -> str:
            return "ok"

        tool_registry.register(FunctionTool(_plain, name="plain", description="d"))
        tool_registry.register(
            FunctionTool(_hidden, name="hidden", description="d"),
            agent_callable=False,
        )
        tool_registry.register(
            FunctionTool(_scoped, name="scoped", description="d"),
            user_scoped=True,
        )

        client = TestClient(_make_app(tmp_path))
        by_name = {
            t["name"]: t
            for t in client.get("/admin/tools", headers=_admin_headers()).json()
        }

        assert by_name["plain"]["agent_callable"] is True
        assert by_name["plain"]["user_scoped"] is False
        assert by_name["hidden"]["agent_callable"] is False
        assert by_name["scoped"]["user_scoped"] is True
    finally:
        _restore_registry(snap)


# ---------------------------------------------------------------------------
# Discover
# ---------------------------------------------------------------------------


def test_discover_requires_auth(tmp_path):
    client = TestClient(_make_app(tmp_path))
    assert (
        client.post("/admin/tools/discover", json={"query": "anything"}).status_code
        == 401
    )


def test_discover_ranks_registered_tools(tmp_path):
    snap = _clear_registry()
    try:
        from src.internal.tools.base import FunctionTool

        def _weather() -> str:
            return "sunny"

        def _payroll() -> str:
            return "paid"

        tool_registry.register(
            FunctionTool(
                _weather, name="weather_lookup", description="Look up the weather"
            )
        )
        tool_registry.register(
            FunctionTool(
                _payroll, name="payroll_export", description="Export payroll records"
            )
        )

        client = TestClient(_make_app(tmp_path))
        resp = client.post(
            "/admin/tools/discover",
            json={"query": "what is the weather"},
            headers=_admin_headers(),
        )
        assert resp.status_code == 200
        names = [t["name"] for t in resp.json()["final_tools"]]
        assert "weather_lookup" in names
    finally:
        _restore_registry(snap)


def test_discover_is_not_shadowed_by_the_name_route(tmp_path):
    """POST /admin/tools/discover must not be read as tool name "discover"."""
    snap = _clear_registry()
    try:
        client = TestClient(_make_app(tmp_path))
        resp = client.post(
            "/admin/tools/discover", json={"query": "x"}, headers=_admin_headers()
        )
        # Empty registry still routes to discover (200), not 404-from-get_tool.
        assert resp.status_code == 200
        assert "final_tools" in resp.json()
    finally:
        _restore_registry(snap)
