"""Tests for src/servers/{auth_check,seeding,usage_limits,tenant_usage_limits}."""

from __future__ import annotations

import json

from fastapi import Depends, FastAPI

from src.internal.configs import load_app_settings
from src.internal.servers._auth import make_require_admin

from src.internal.db import AgenticSearchStore
from src.internal.servers.web.auth_check import PUBLIC_ENDPOINT_SPECS
from src.internal.servers.web.auth_check import check_router_auth
from src.internal.servers.web.seeding import SeedConfiguration
from src.internal.servers.web.seeding import get_seed_config
from src.internal.servers.web.seeding import seed_db
from src.internal.servers.limits.tenant_usage_limits import (
    get_tenant_usage_limit_overrides,
)
from src.internal.servers.limits.tenant_usage_limits import unlimited
from src.internal.servers.limits.usage_limits import NO_LIMIT
from src.internal.servers.limits.usage_limits import TenantUsageLimitOverrides
from src.internal.servers.limits.usage_limits import is_tenant_on_trial


# ---------------------------------------------------------------------------
# usage_limits
# ---------------------------------------------------------------------------


def test_no_limit_is_negative_one():
    assert NO_LIMIT == -1


def test_tenant_usage_limit_overrides_is_unlimited_by_default():
    overrides = TenantUsageLimitOverrides(tenant_id="t1")
    assert overrides.is_unlimited()


def test_tenant_usage_limit_overrides_not_unlimited_when_set():
    overrides = TenantUsageLimitOverrides(tenant_id="t1", api_calls_paid=100)
    assert not overrides.is_unlimited()


def test_is_tenant_on_trial_always_false():
    assert is_tenant_on_trial("any-tenant") is False
    assert is_tenant_on_trial("") is False


# ---------------------------------------------------------------------------
# tenant_usage_limits
# ---------------------------------------------------------------------------


def test_unlimited_returns_all_no_limit():
    result = unlimited("t1")
    assert result.tenant_id == "t1"
    assert result.is_unlimited()


def test_get_tenant_usage_limit_overrides_returns_unlimited():
    result = get_tenant_usage_limit_overrides("some-tenant")
    assert result.tenant_id == "some-tenant"
    assert result.is_unlimited()


# ---------------------------------------------------------------------------
# auth_check
# ---------------------------------------------------------------------------


def _make_app() -> FastAPI:
    app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)
    guard = make_require_admin(load_app_settings())

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.get("/api/data", dependencies=[Depends(guard)])
    def data():
        return {}

    @app.post("/api/secret", dependencies=[Depends(guard)])
    def secret():
        return {}

    return app


def test_check_router_auth_runs_without_error():
    app = _make_app()
    check_router_auth(app, PUBLIC_ENDPOINT_SPECS)


def test_check_router_auth_with_custom_public_specs(caplog):
    import logging

    app = _make_app()
    custom_specs = [("/health", {"GET"})]
    with caplog.at_level(logging.INFO):
        check_router_auth(app, custom_specs)
    assert any("/health" in r.message for r in caplog.records)


def test_check_router_auth_warns_for_declared_but_missing_route(caplog):
    import logging

    app = _make_app()
    specs = [("/health", {"GET"}), ("/nonexistent-path", {"GET"})]
    with caplog.at_level(logging.WARNING):
        check_router_auth(app, specs)
    assert any("nonexistent-path" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# seeding
# ---------------------------------------------------------------------------


def test_get_seed_config_returns_none_when_env_not_set(monkeypatch):
    monkeypatch.delenv("ENV_SEED_CONFIGURATION", raising=False)
    assert get_seed_config() is None


def test_get_seed_config_parses_valid_json(monkeypatch):
    payload = {
        "users": [{"id": "u1", "email": "u1@test.com"}],
        "admin_user_ids": ["u1"],
    }
    monkeypatch.setenv("ENV_SEED_CONFIGURATION", json.dumps(payload))
    config = get_seed_config()
    assert config is not None
    assert config.users[0].id == "u1"
    assert config.admin_user_ids == ["u1"]


def test_get_seed_config_returns_none_on_invalid_json(monkeypatch):
    monkeypatch.setenv("ENV_SEED_CONFIGURATION", "not-valid-json")
    assert get_seed_config() is None


def test_seed_db_no_op_when_no_env(monkeypatch, tmp_path):
    monkeypatch.delenv("ENV_SEED_CONFIGURATION", raising=False)
    store = AgenticSearchStore(tmp_path / "seed.sqlite3")
    seed_db(store)  # must not raise
    store.close()


def test_seed_db_inserts_users(monkeypatch, tmp_path):
    payload = {
        "users": [
            {"id": "u1", "email": "alice@example.com", "name": "Alice"},
            {"id": "u2", "email": "bob@example.com"},
        ],
    }
    monkeypatch.setenv("ENV_SEED_CONFIGURATION", json.dumps(payload))

    store = AgenticSearchStore(tmp_path / "seed.sqlite3")
    seed_db(store)

    alice = store.get_user("u1")
    assert alice is not None
    assert alice.email == "alice@example.com"
    assert alice.name == "Alice"

    bob = store.get_user("u2")
    assert bob is not None

    store.close()


def test_seed_db_is_idempotent(monkeypatch, tmp_path):
    payload = {"users": [{"id": "u1", "email": "x@example.com"}]}
    monkeypatch.setenv("ENV_SEED_CONFIGURATION", json.dumps(payload))

    store = AgenticSearchStore(tmp_path / "seed.sqlite3")
    seed_db(store)
    seed_db(store)  # second run must not raise or duplicate

    assert store.get_user("u1") is not None
    store.close()


def test_seed_configuration_model_defaults():
    config = SeedConfiguration()
    assert config.users == []
    assert config.admin_user_ids == []
