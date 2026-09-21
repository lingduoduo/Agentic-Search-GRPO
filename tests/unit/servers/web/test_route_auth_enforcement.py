"""The startup audit rejects unclassified routes, per route and per method."""

import pytest
from fastapi import Depends, FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.testclient import TestClient

from src.internal.configs import load_app_settings
from src.internal.db import AgenticSearchStore
from src.internal.servers._auth import make_require_admin
from src.internal.servers.web.app import SearchExperienceSettings, create_web_app
from src.internal.servers.web.auth_check import check_router_auth


def test_unguarded_route_fails_with_method_and_path():
    app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)

    @app.post("/private")
    def private():
        return {}

    with pytest.raises(RuntimeError, match="POST /private"):
        check_router_auth(app, [])


@pytest.mark.parametrize("same_route", [False, True])
def test_public_get_does_not_exempt_post(same_route):
    app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)

    def endpoint():
        return {}

    if same_route:
        app.add_api_route("/mixed", endpoint, methods=["GET", "POST"])
    else:
        app.add_api_route("/mixed", endpoint, methods=["GET"])
        app.add_api_route("/mixed", endpoint, methods=["POST"])
    with pytest.raises(RuntimeError, match="POST /mixed"):
        check_router_auth(app, [("/mixed", {"GET"})])


def test_nested_auth_dependency_is_accepted(monkeypatch):
    monkeypatch.setenv("AGENTIC_SEARCH_DEV_ADMIN", "false")
    guard = make_require_admin(load_app_settings())
    app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)

    def parent(user=Depends(guard)):
        return user

    app.add_api_route("/private", lambda: {}, dependencies=[Depends(parent)])
    check_router_auth(app, [])


def test_guard_name_and_comment_do_not_count_as_authentication():
    app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)

    def _require_admin():
        return None

    @app.get("/fake", dependencies=[Depends(_require_admin)])
    def fake():
        # _require_auth(request) is not an actual guard.
        return {"documentation": "resolve_active_user(request, store)"}

    with pytest.raises(RuntimeError, match="GET /fake"):
        check_router_auth(app, [])


def test_public_registration_does_not_hide_an_unguarded_duplicate(monkeypatch):
    monkeypatch.setenv("AGENTIC_SEARCH_DEV_ADMIN", "false")
    app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)
    app.add_api_route("/private", lambda: {})
    app.add_api_route(
        "/private",
        lambda: {},
        dependencies=[Depends(make_require_admin(load_app_settings()))],
    )
    with pytest.raises(RuntimeError, match="GET /private"):
        check_router_auth(app, [])


def test_mounted_app_is_audited_with_its_prefix():
    app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)
    child = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)
    child.add_api_route("/secret", lambda: {})
    app.mount("/nested", child)
    with pytest.raises(RuntimeError, match="GET /nested/secret"):
        check_router_auth(app, [])


def test_static_mount_requires_explicit_public_classification(tmp_path):
    app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)
    app.mount("/files", StaticFiles(directory=tmp_path))
    with pytest.raises(RuntimeError, match="/files"):
        check_router_auth(app, [])
    check_router_auth(app, [("/files", {"GET", "HEAD"})])


def test_inline_session_guard_is_accepted(monkeypatch):
    monkeypatch.setenv("AGENTIC_SEARCH_DEV_ADMIN", "false")
    with AgenticSearchStore(":memory:") as store:
        web = create_web_app(store=store)
        route = next(r for r in web.routes if r.path == "/api/sessions/{session_id}")
        app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)
        app.router.routes.append(route)
        check_router_auth(app, [])


@pytest.mark.parametrize(
    "debug,integration", [(False, False), (True, False), (False, True)]
)
def test_registered_app_is_fully_classified(monkeypatch, debug, integration):
    monkeypatch.setenv("AGENTIC_SEARCH_DEV_ADMIN", "false")
    monkeypatch.setenv("INTEGRATION_TESTS_MODE", str(integration).lower())
    with AgenticSearchStore(":memory:") as store:
        app = create_web_app(SearchExperienceSettings(debug_panels=debug), store=store)
        check_router_auth(app)


def test_lifespan_refuses_unclassified_route(monkeypatch):
    monkeypatch.setenv("AGENTIC_SEARCH_DEV_ADMIN", "false")
    with AgenticSearchStore(":memory:") as store:
        app = create_web_app(store=store)
        app.add_api_route("/forgot-auth", lambda: {})
        with pytest.raises(RuntimeError, match="GET /forgot-auth"):
            with TestClient(app):
                pass


def test_optional_identity_resolution_is_not_a_guard():
    from src.internal.servers.users.api import resolve_request_user

    app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)

    @app.get("/leak")
    def leak(request: Request):
        resolve_request_user(request)
        return {"secret": "visible to everyone"}

    with pytest.raises(RuntimeError, match="GET /leak"):
        check_router_auth(app, [])


def test_unused_nested_guard_is_not_authentication():
    from src.internal.servers.users.api import _require_auth

    app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)

    @app.get("/unused")
    def unused(request: Request):
        def never_called():
            _require_auth(request, None)

        return {}

    with pytest.raises(RuntimeError, match="GET /unused"):
        check_router_auth(app, [])


def test_unguarded_websocket_route_still_fails():
    """The socket branch must not become a hole while becoming reachable."""
    app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)

    @app.websocket("/ws/open")
    async def open_socket(websocket):  # pragma: no cover - never accepted
        await websocket.accept()

    with pytest.raises(RuntimeError, match="WEBSOCKET /ws/open"):
        check_router_auth(app, [])


def test_websocket_route_guarded_inline_passes():
    """A socket authenticates inline; it cannot carry an auth dependency."""
    from src.internal.servers.web.ws_channel import authenticate_ws

    app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)

    @app.websocket("/ws/guarded")
    async def guarded_socket(websocket):  # pragma: no cover - never accepted
        await authenticate_ws(websocket)

    check_router_auth(app, [])
