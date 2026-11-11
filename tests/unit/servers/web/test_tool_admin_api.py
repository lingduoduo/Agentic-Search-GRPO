"""HTTP-level tests for /admin/tools/* endpoints."""

from __future__ import annotations

import json

from fastapi.testclient import TestClient

from src.internal.auth import generate_user_jwt_token
from src.internal.configs import AppSettings, AuthSettings
from src.internal.db import UserRecord
from src.internal.servers.web.app import SearchExperienceSettings, create_web_app
from src.internal.tools.base import ResultKind, ToolEffect
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

        # Tool appears in list, under the provider's registered name
        listed = {
            t["name"]: t
            for t in client.get("/admin/tools", headers=_admin_headers()).json()
        }
        assert listed["echo_word"]["provider_id"] == data["provider_id"]
        assert listed["echo_word"]["provider_name"] == "Echo API"
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

        tool_registry.tool(
            double, effect=ToolEffect.READ_ONLY, result_kind=ResultKind.JSON
        )
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


def test_invoke_reports_a_raising_tool_as_an_error_not_a_500(tmp_path):
    snap = _clear_registry()
    try:

        def flaky() -> str:
            """Fail like an LLM-backed tool whose provider is down."""
            raise RuntimeError("llm provider down")

        tool_registry.tool(
            flaky, effect=ToolEffect.READ_ONLY, result_kind=ResultKind.JSON
        )
        client = TestClient(_make_app(tmp_path))
        resp = client.post(
            "/admin/tools/flaky/invoke",
            json={"arguments": {}},
            headers=_admin_headers(),
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["response"] == ""
        assert data["errors"] == ["unknown: RuntimeError"]
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

        tool_registry.register(
            FunctionTool(
                _plain,
                name="plain",
                description="d",
                effect=ToolEffect.READ_ONLY,
                result_kind=ResultKind.TEXT,
            )
        )
        tool_registry.register(
            FunctionTool(
                _hidden,
                name="hidden",
                description="d",
                effect=ToolEffect.READ_ONLY,
                result_kind=ResultKind.TEXT,
            ),
            agent_callable=False,
        )
        tool_registry.register(
            FunctionTool(
                _scoped,
                name="scoped",
                description="d",
                effect=ToolEffect.READ_ONLY,
                result_kind=ResultKind.TEXT,
            ),
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
                _weather,
                name="weather_lookup",
                description="Look up the weather",
                effect=ToolEffect.READ_ONLY,
                result_kind=ResultKind.TEXT,
            )
        )
        tool_registry.register(
            FunctionTool(
                _payroll,
                name="payroll_export",
                description="Export payroll records",
                effect=ToolEffect.READ_ONLY,
                result_kind=ResultKind.TEXT,
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


# ---------------------------------------------------------------------------
# Providers page, schema validation, provider detail and update
# ---------------------------------------------------------------------------

_TWO_OPS_JSON = json.dumps(
    {
        "openapi": "3.0.0",
        "info": {"title": "Echo API", "description": "Echo words back, twice."},
        "servers": [{"url": "https://echo.test"}],
        "paths": {
            "/echo": {"post": {"operationId": "echo_word", "summary": "Echo a word."}},
            "/shout": {"get": {"operationId": "shout_word"}},
        },
    }
)


def _register(client, name: str, openapi_json: str = _OPENAPI_JSON) -> str:
    resp = client.post(
        "/admin/tools/openapi",
        json={"name": name, "openapi_json": openapi_json},
        headers=_admin_headers(),
    )
    assert resp.status_code == 201, resp.text
    return resp.json()["provider_id"]


def _rename_ops(prefix: str) -> str:
    """A one-operation schema whose tool name is unique to ``prefix``."""
    return _OPENAPI_JSON.replace("echo_word", f"{prefix}_word")


def test_new_endpoints_require_auth(tmp_path):
    client = TestClient(_make_app(tmp_path))
    assert client.get("/admin/tools/providers").status_code == 401
    assert (
        client.post(
            "/admin/tools/openapi/validate", json={"openapi_json": "{}"}
        ).status_code
        == 401
    )
    assert client.get("/admin/tools/openapi/x").status_code == 401
    assert client.put("/admin/tools/openapi/x", json={}).status_code == 401


def test_providers_groups_and_pages(tmp_path):
    snap = _clear_registry()
    try:
        client = TestClient(_make_app(tmp_path))

        def add(a: int, b: int) -> int:
            """Add two ints."""
            return a + b

        tool_registry.tool(
            add, effect=ToolEffect.READ_ONLY, result_kind=ResultKind.JSON
        )

        echo_id = _register(client, "Echo API")
        for i in range(2):
            _register(client, f"Extra {i}", _rename_ops(f"extra{i}"))

        first = client.get(
            "/admin/tools/providers",
            params={"current_page": 1, "page_size": 2},
            headers=_admin_headers(),
        ).json()
        assert first["paginator"] == {
            "current_page": 1,
            "page_size": 2,
            "total_page": 2,
            "total_record": 4,
        }
        local, echo = first["list"]
        assert (local["id"], local["source"]) == ("local", "function")
        assert [t["name"] for t in local["tools"]] == ["add"]
        assert echo == {
            "id": echo_id,
            "name": "Echo API",
            "description": "Echo words back.",
            "source": "openapi",
            "tools": [echo["tools"][0]],
        }
        assert echo["tools"][0]["name"] == "echo_word"

        second = client.get(
            "/admin/tools/providers",
            params={"current_page": 2, "page_size": 2},
            headers=_admin_headers(),
        ).json()
        assert [p["name"] for p in second["list"]] == ["Extra 0", "Extra 1"]

        past_end = client.get(
            "/admin/tools/providers",
            params={"current_page": 3, "page_size": 2},
            headers=_admin_headers(),
        ).json()
        assert past_end["list"] == []
        assert (
            client.get(
                "/admin/tools/providers",
                params={"page_size": 0},
                headers=_admin_headers(),
            ).status_code
            == 422
        )
    finally:
        _restore_registry(snap)


def test_providers_search_matches_provider_and_tool_text(tmp_path):
    snap = _clear_registry()
    try:
        client = TestClient(_make_app(tmp_path))
        _register(client, "Echo API")
        _register(client, "Weather", _rename_ops("forecast"))

        def names(word: str) -> list[str]:
            page = client.get(
                "/admin/tools/providers",
                params={"search_word": word},
                headers=_admin_headers(),
            ).json()
            assert page["paginator"]["total_record"] == len(page["list"])
            return [p["name"] for p in page["list"]]

        assert names("weather") == ["Weather"]  # provider name
        assert names("FORECAST") == ["Weather"]  # tool name, any case
        assert names("echo a word") == ["Echo API", "Weather"]  # tool description
        assert names("nothing") == []
    finally:
        _restore_registry(snap)


def test_validate_previews_tools_without_registering(tmp_path):
    snap = _clear_registry()
    try:
        client = TestClient(_make_app(tmp_path))
        resp = client.post(
            "/admin/tools/openapi/validate",
            json={"openapi_json": _TWO_OPS_JSON},
            headers=_admin_headers(),
        )
        assert resp.status_code == 200
        assert resp.json()["tools"] == [
            {
                "name": "echo_word",
                "description": "Echo a word.",
                "method": "POST",
                "path": "/echo",
            },
            {
                "name": "shout_word",
                "description": "GET /shout",
                "method": "GET",
                "path": "/shout",
            },
        ]
        assert client.get("/admin/tools", headers=_admin_headers()).json() == []

        bad = client.post(
            "/admin/tools/openapi/validate",
            json={"openapi_json": "{not json"},
            headers=_admin_headers(),
        )
        assert bad.status_code == 400
        assert "valid OpenAPI" in bad.json()["detail"]
    finally:
        _restore_registry(snap)


def test_get_openapi_provider_detail(tmp_path):
    snap = _clear_registry()
    try:
        client = TestClient(_make_app(tmp_path))
        resp = client.post(
            "/admin/tools/openapi",
            json={
                "name": "Echo API",
                "openapi_json": _OPENAPI_JSON,
                "headers": {"Authorization": "Bearer x"},
            },
            headers=_admin_headers(),
        )
        provider_id = resp.json()["provider_id"]

        detail = client.get(
            f"/admin/tools/openapi/{provider_id}", headers=_admin_headers()
        ).json()
        assert detail == {
            "provider_id": provider_id,
            "name": "Echo API",
            "description": "Echo words back.",
            "openapi_json": _OPENAPI_JSON,
            "headers": {"Authorization": "Bearer x"},
        }
        for missing in ("not-a-uuid", "00000000-0000-0000-0000-000000000000"):
            assert (
                client.get(
                    f"/admin/tools/openapi/{missing}", headers=_admin_headers()
                ).status_code
                == 404
            )
    finally:
        _restore_registry(snap)


def test_update_openapi_provider_rebuilds_its_tools(tmp_path):
    snap = _clear_registry()
    try:
        client = TestClient(_make_app(tmp_path))
        provider_id = _register(client, "Echo API", _TWO_OPS_JSON)

        resp = client.put(
            f"/admin/tools/openapi/{provider_id}",
            json={
                "name": "Echo v2",
                "openapi_json": _OPENAPI_JSON,
                "headers": {"X-Key": "k"},
            },
            headers=_admin_headers(),
        )
        assert resp.status_code == 200
        assert resp.json() == {"provider_id": provider_id, "tool_names": ["echo_word"]}

        listed = {
            t["name"]: t
            for t in client.get("/admin/tools", headers=_admin_headers()).json()
        }
        # The tool the new schema dropped is gone; the kept one is renamed.
        assert list(listed) == ["echo_word"]
        assert listed["echo_word"]["provider_id"] == provider_id
        assert listed["echo_word"]["provider_name"] == "Echo v2"
        assert listed["echo_word"]["provider_description"] == "Echo words back."
        detail = client.get(
            f"/admin/tools/openapi/{provider_id}", headers=_admin_headers()
        ).json()
        assert detail["headers"] == {"X-Key": "k"}
    finally:
        _restore_registry(snap)


def test_update_with_invalid_schema_changes_nothing(tmp_path):
    snap = _clear_registry()
    try:
        client = TestClient(_make_app(tmp_path))
        provider_id = _register(client, "Echo API")

        resp = client.put(
            f"/admin/tools/openapi/{provider_id}",
            json={"name": "Echo API", "openapi_json": "{not json"},
            headers=_admin_headers(),
        )
        assert resp.status_code == 400
        names = [
            t["name"]
            for t in client.get("/admin/tools", headers=_admin_headers()).json()
        ]
        assert names == ["echo_word"]

        missing = client.put(
            "/admin/tools/openapi/00000000-0000-0000-0000-000000000000",
            json={"name": "x", "openapi_json": _OPENAPI_JSON},
            headers=_admin_headers(),
        )
        assert missing.status_code == 404
    finally:
        _restore_registry(snap)


def test_providers_route_is_not_shadowed_by_the_name_route(tmp_path):
    snap = _clear_registry()
    try:
        client = TestClient(_make_app(tmp_path))
        resp = client.get("/admin/tools/providers", headers=_admin_headers())
        assert resp.status_code == 200
        assert resp.json()["list"] == []
    finally:
        _restore_registry(snap)
