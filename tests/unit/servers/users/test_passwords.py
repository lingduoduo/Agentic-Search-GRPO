"""Password storage upgrades preserve existing accounts and fail closed."""

import hashlib

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.internal.configs import AppSettings
from src.internal.db import AgenticSearchStore, UserRecord
from src.internal.servers.users.api import create_users_router


def test_registration_salts_equal_passwords_independently():
    with AgenticSearchStore(":memory:") as store:
        app = FastAPI()
        app.include_router(create_users_router(store, AppSettings()))
        client = TestClient(app)
        for email in ("alice@example.test", "bob@example.test"):
            assert (
                client.post(
                    "/auth/register", json={"email": email, "password": "same password"}
                ).status_code
                == 200
            )
            assert (
                client.post(
                    "/auth/login", data={"username": email, "password": "same password"}
                ).status_code
                == 200
            )
            assert (
                client.post(
                    "/auth/login", data={"username": email, "password": "wrong"}
                ).status_code
                == 400
            )
        hashes = [u.metadata["password_hash"] for u in store.list_users()]
        assert hashes[0] != hashes[1]
        assert all(h.startswith("pbkdf2_sha256$") for h in hashes)


def test_legacy_password_upgrades_only_on_successful_login():
    legacy = hashlib.pbkdf2_hmac(
        "sha256", b"old password", b"agentic-search", 100_000
    ).hex()
    with AgenticSearchStore(":memory:") as store:
        store.upsert_user(
            UserRecord(
                id="alice",
                email="alice@example.test",
                metadata={
                    "password_hash": legacy,
                    "role": "basic",
                    "custom": "preserved",
                },
            )
        )
        app = FastAPI()
        app.include_router(create_users_router(store, AppSettings()))
        client = TestClient(app)
        assert (
            client.post(
                "/auth/login",
                data={"username": "alice@example.test", "password": "wrong"},
            ).status_code
            == 400
        )
        assert store.get_user("alice").metadata["password_hash"] == legacy
        assert (
            client.post(
                "/auth/login",
                data={"username": "alice@example.test", "password": "old password"},
            ).status_code
            == 200
        )
        metadata = store.get_user("alice").metadata
        assert metadata["password_hash"] != legacy
        assert metadata["role"] == "basic"
        assert metadata["custom"] == "preserved"
        assert client.get("/me").status_code == 200
        assert (
            client.post(
                "/auth/login",
                data={"username": "alice@example.test", "password": "old password"},
            ).status_code
            == 200
        )


@pytest.mark.parametrize(
    "stored",
    [
        None,
        {},
        "",
        "broken",
        "pbkdf2_sha256$600000$bad$bad",
        "pbkdf2_sha256$999999999999$00$00",
    ],
)
def test_corrupt_password_hash_refuses_login(stored):
    with AgenticSearchStore(":memory:") as store:
        store.upsert_user(
            UserRecord(
                id="alice",
                email="alice@example.test",
                metadata={"password_hash": stored},
            )
        )
        app = FastAPI()
        app.include_router(create_users_router(store, AppSettings()))
        assert (
            TestClient(app)
            .post(
                "/auth/login",
                data={"username": "alice@example.test", "password": "password"},
            )
            .status_code
            == 400
        )
