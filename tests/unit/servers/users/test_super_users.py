"""Configured super-users get the same active-account permissions on every router."""

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.internal.auth import generate_user_jwt_token
from src.internal.configs import AppSettings, AuthSettings
from src.internal.db import AgenticSearchStore, UserRecord
from src.internal.servers.users.api import create_users_router


@pytest.mark.parametrize("configured", ["alice", "alice@example.test"])
def test_super_user_can_manage_accounts_and_get_admin_permissions(configured):
    with AgenticSearchStore(":memory:") as store:
        store.upsert_user(
            UserRecord(
                id="alice", email="alice@example.test", metadata={"role": "basic"}
            )
        )
        app = FastAPI()
        app.state.auth_store = store
        app.include_router(
            create_users_router(
                store, AppSettings(auth=AuthSettings(super_users=(configured,)))
            )
        )
        client = TestClient(app)
        headers = {
            "Authorization": "Bearer " + generate_user_jwt_token(user_id="alice")
        }
        assert client.get("/manage/users/accepted", headers=headers).status_code == 200
        assert (
            "full_admin_panel_access"
            in client.get("/me/permissions", headers=headers).json()
        )
        store.set_user_active("alice", False)
        assert client.get("/manage/users/accepted", headers=headers).status_code == 401
        assert client.get("/me/permissions", headers=headers).status_code == 401
