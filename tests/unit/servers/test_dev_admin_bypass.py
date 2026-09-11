"""Tests for the dev-only admin-auth bypass (AGENTIC_SEARCH_DEV_ADMIN).

When `AuthSettings.dev_admin_bypass` is on, `make_require_admin` treats every
request as a fixed dev admin so the local admin dashboard loads without a token.
Default off must preserve the normal 401/403 behavior.
"""

from __future__ import annotations

import pytest
from fastapi import FastAPI, HTTPException, Request
from fastapi.testclient import TestClient

from src.internal.configs import AppSettings, AuthSettings
from src.internal.servers._auth import make_require_admin
from src.internal.servers.web.app import SearchExperienceSettings, create_web_app


def _request():
    return Request({"type": "http", "headers": [], "app": FastAPI()})


def test_bypass_returns_dev_admin_without_headers():
    dep = make_require_admin(AppSettings(auth=AuthSettings(dev_admin_bypass=True)))
    user = dep(_request())
    assert user.is_anonymous is False
    assert user.metadata.get("role") == "admin"
    assert user.id == "dev-admin"


def test_default_rejects_anonymous():
    dep = make_require_admin(AppSettings(auth=AuthSettings()))
    with pytest.raises(HTTPException) as exc:
        dep(_request())
    assert exc.value.status_code == 401


def test_admin_endpoint_allows_unauthenticated_when_bypass_on(tmp_path):
    app = create_web_app(
        SearchExperienceSettings(db_path=tmp_path / "state.sqlite3"),
        app_settings=AppSettings(auth=AuthSettings(dev_admin_bypass=True)),
    )
    client = TestClient(app)
    assert client.get("/admin/tools").status_code == 200


def test_admin_endpoint_requires_auth_by_default(tmp_path):
    app = create_web_app(
        SearchExperienceSettings(db_path=tmp_path / "state.sqlite3"),
        app_settings=AppSettings(auth=AuthSettings()),
    )
    client = TestClient(app)
    assert client.get("/admin/tools").status_code == 401
