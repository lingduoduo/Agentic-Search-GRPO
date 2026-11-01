"""Browser account lifecycle and signed-out application entry points."""

import pytest
from fastapi.testclient import TestClient

from src.internal.db import AgenticSearchStore
from src.internal.servers.web.app import create_web_app
from src.internal.servers.web.auth_check import PUBLIC_ENDPOINT_SPECS, check_router_auth


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("AGENTIC_SEARCH_DEV_ADMIN", "false")
    with AgenticSearchStore(":memory:") as store:
        yield TestClient(create_web_app(store=store))


def test_login_logout_clears_browser_identity(client):
    credentials = {"email": "alice@example.test", "password": "test-only-password"}
    assert client.post("/auth/register", json=credentials).status_code == 200
    assert (
        client.post(
            "/auth/login",
            data={
                "username": credentials["email"],
                "password": credentials["password"],
            },
        ).status_code
        == 200
    )
    assert client.get("/me").status_code == 200
    response = client.post("/auth/logout")
    assert response.status_code == 204
    assert response.content == b""
    cookie = response.headers["set-cookie"].lower()
    for attribute in (
        "fastapiusersauth=",
        "httponly",
        "max-age=0",
        "path=/",
        "samesite=lax",
        "expires=",
    ):
        assert attribute in cookie
    assert client.get("/me").status_code == 401


@pytest.mark.parametrize("cookie", [None, "expired-or-invalid-token"])
def test_logout_is_idempotent_without_valid_session(client, cookie):
    if cookie:
        client.cookies.set("fastapiusersauth", cookie)
    for _ in range(2):
        assert client.post("/auth/logout").status_code == 204
    assert client.get("/me").status_code == 401


@pytest.mark.parametrize("path", ["/home", "/account", "/admin"])
def test_account_pages_serve_public_shell_at_exact_paths(client, path):
    response = client.get(path)
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert response.text == client.get("/").text
    assert client.get(path + "/unknown").status_code == 404
    assert (path, {"GET"}) in PUBLIC_ENDPOINT_SPECS
    assert client.post(path).status_code == 405


def test_sign_in_page_shares_its_path_with_the_login_endpoint(client):
    response = client.get("/auth/login")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert response.text == client.get("/").text
    assert ("/auth/login", {"GET", "POST"}) in PUBLIC_ENDPOINT_SPECS
    # POST still reaches the login handler, which rejects bad credentials.
    rejected = client.post(
        "/auth/login", data={"username": "nobody@example.test", "password": "x"}
    )
    assert rejected.status_code == 400
    assert client.get("/login").status_code == 404


def test_account_route_classification_preserves_admin_api_guards(client):
    assert ("/auth/logout", {"POST"}) in PUBLIC_ENDPOINT_SPECS
    check_router_auth(client.app, PUBLIC_ENDPOINT_SPECS)
    assert client.get("/manage/users/accepted").status_code == 401
