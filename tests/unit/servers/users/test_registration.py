"""Registration must atomically assign the bootstrap admin and reject duplicates."""

from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.internal.configs import AppSettings
from src.internal.db import AgenticSearchStore
from src.internal.servers.users import api


@pytest.mark.parametrize("separate_connections", [False, True])
@pytest.mark.parametrize("duplicate", [False, True])
def test_concurrent_registration(
    tmp_path, monkeypatch, separate_connections, duplicate
):
    first = AgenticSearchStore(tmp_path / "users.sqlite")
    second = (
        AgenticSearchStore(tmp_path / "users.sqlite") if separate_connections else first
    )
    barrier = Barrier(2)
    original_hash = api._hash_password

    def synchronized_hash(password):
        result = original_hash(password)
        barrier.wait(timeout=10)
        return result

    monkeypatch.setattr(api, "_hash_password", synchronized_hash)

    def register(args):
        store, email = args
        app = FastAPI()
        app.include_router(api.create_users_router(store, AppSettings()))
        with TestClient(app) as client:
            return client.post(
                "/auth/register", json={"email": email, "password": "password"}
            )

    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            responses = list(
                pool.map(
                    register,
                    [
                        (first, "alice@example.test"),
                        (
                            second,
                            "ALICE@example.test" if duplicate else "bob@example.test",
                        ),
                    ],
                )
            )
        assert sorted(r.status_code for r in responses) == (
            [200, 400] if duplicate else [200, 200]
        )
        users = first.list_users()
        assert len(users) == (1 if duplicate else 2)
        assert sum(u.metadata["role"] == "admin" for u in users) == 1
        # A rejected insert must not leave an open or poisoned transaction.
        assert first.get_user(users[0].id) is not None
    finally:
        if second is not first:
            second.close()
        first.close()


def test_registration_rejects_existing_unicode_email_without_replacing_account():
    from src.internal.db import UserRecord

    with AgenticSearchStore(":memory:") as store:
        existing = store.upsert_user(
            UserRecord(
                id="imported", email="Älice@example.test", metadata={"role": "basic"}
            )
        )
        app = FastAPI()
        app.include_router(api.create_users_router(store, AppSettings()))
        response = TestClient(app).post(
            "/auth/register",
            json={"email": "älice@example.test", "password": "new password"},
        )
        assert response.status_code == 400
        assert store.list_users() == [existing]
