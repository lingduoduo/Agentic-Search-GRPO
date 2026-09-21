from __future__ import annotations

import json
from datetime import datetime, timezone

from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.internal.auth import generate_user_jwt_token
from src.internal.db import AgenticSearchStore
from src.internal.db.models import GroupRecord
from src.internal.db.models import UserRecord
from src.internal.search.process_search_query import SearchQueryResult
from src.internal.servers.query_and_chat.search_backend import create_search_router


def _client(tmp_path) -> tuple[TestClient, AgenticSearchStore]:
    store = AgenticSearchStore(tmp_path / "search.sqlite3")
    app = FastAPI()
    app.include_router(create_search_router(store))
    return TestClient(app), store


def _authenticated_headers() -> dict[str, str]:
    token = generate_user_jwt_token(
        user_id="alice",
        email="alice@example.com",
        group_ids=["engineering"],
    )
    return {"Authorization": f"Bearer {token}"}


def test_search_uses_server_acl_and_preserves_safe_caller_filters(
    tmp_path, monkeypatch
):
    captured = {}

    async def fake_run_expanded_search(query, **kwargs):
        captured["filters"] = kwargs["filters"]
        return SearchQueryResult(
            original_query=query,
            executed_queries=[query],
            results=[],
        )

    monkeypatch.setattr(
        "src.internal.servers.query_and_chat.search_backend.run_expanded_search",
        fake_run_expanded_search,
    )
    client, store = _client(tmp_path)
    store.upsert_user(UserRecord(id="alice", email="alice@example.com"))
    store.upsert_group(
        GroupRecord(id="engineering", name="Engineering", user_ids=["alice"])
    )
    cutoff = "2026-01-02T03:04:05Z"

    response = client.post(
        "/search/send-search-message",
        headers=_authenticated_headers(),
        json={
            "search_query": "deployment guide",
            "stream": False,
            "filters": {
                "source_types": ["file"],
                "document_sets": ["engineering"],
                "tags": {"environment": "production"},
                "time_cutoff": cutoff,
            },
        },
    )

    assert response.status_code == 200
    filters = captured["filters"]
    assert filters.access_acl == [
        "email:alice@example.com",
        "group:engineering",
        "public",
        "user:alice",
    ]
    assert filters.source_types == ["file"]
    assert filters.document_sets == ["engineering"]
    assert filters.tags == {"environment": "production"}
    assert filters.time_cutoff == datetime(2026, 1, 2, 3, 4, 5, tzinfo=timezone.utc)
    store.close()


def test_search_discards_caller_supplied_acl(tmp_path, monkeypatch):
    captured = {}

    async def fake_run_expanded_search(query, **kwargs):
        captured["filters"] = kwargs["filters"]
        return SearchQueryResult(
            original_query=query,
            executed_queries=[query],
            results=[],
        )

    monkeypatch.setattr(
        "src.internal.servers.query_and_chat.search_backend.run_expanded_search",
        fake_run_expanded_search,
    )
    client, store = _client(tmp_path)
    store.upsert_user(UserRecord(id="alice", email="alice@example.com"))

    response = client.post(
        "/search/send-search-message",
        headers=_authenticated_headers(),
        json={
            "search_query": "private roadmap",
            "stream": False,
            "filters": {"access_acl": ["user:other"]},
        },
    )

    assert response.status_code == 200
    assert "user:other" not in captured["filters"].access_acl
    assert "user:alice" in captured["filters"].access_acl
    store.close()


def test_search_requires_authentication(tmp_path, monkeypatch):
    async def fake_run_expanded_search(*args, **kwargs):
        raise AssertionError("unauthenticated search must not execute")

    monkeypatch.setattr(
        "src.internal.servers.query_and_chat.search_backend.run_expanded_search",
        fake_run_expanded_search,
    )
    client, store = _client(tmp_path)

    response = client.post(
        "/search/send-search-message",
        json={"search_query": "private roadmap", "stream": False},
    )

    assert response.status_code == 401
    assert response.json() == {"detail": "Authentication required."}
    store.close()


def test_search_uses_current_store_groups_instead_of_stale_token_claims(
    tmp_path, monkeypatch
):
    captured = {}

    async def fake_run_expanded_search(query, **kwargs):
        captured["filters"] = kwargs["filters"]
        return SearchQueryResult(
            original_query=query,
            executed_queries=[query],
            results=[],
        )

    monkeypatch.setattr(
        "src.internal.servers.query_and_chat.search_backend.run_expanded_search",
        fake_run_expanded_search,
    )
    client, store = _client(tmp_path)
    store.upsert_user(UserRecord(id="alice", email="alice@example.com"))
    monkeypatch.setattr(
        store,
        "list_group_ids_for_user",
        lambda user_id: ["current-group"] if user_id == "alice" else [],
    )

    response = client.post(
        "/search/send-search-message",
        headers=_authenticated_headers(),
        json={"search_query": "private roadmap", "stream": False},
    )

    assert response.status_code == 200
    acl = captured["filters"].access_acl
    assert "group:current-group" in acl
    assert "group:engineering" not in acl
    store.close()


def _stream_lines(client, monkeypatch, result_or_exc, query="deployment guide"):
    """Drive the stream=True branch and return its decoded NDJSON lines."""

    async def fake_run_expanded_search(q, **kwargs):
        if isinstance(result_or_exc, Exception):
            raise result_or_exc
        return result_or_exc

    monkeypatch.setattr(
        "src.internal.servers.query_and_chat.search_backend.run_expanded_search",
        fake_run_expanded_search,
    )
    with client.stream(
        "POST",
        "/search/send-search-message",
        headers=_authenticated_headers(),
        json={"search_query": query, "stream": True},
    ) as response:
        assert response.status_code == 200
        content_type = response.headers["content-type"]
        lines = [line for line in response.iter_lines() if line.strip()]
    return content_type, lines


def test_stream_emits_ndjson_queries_then_docs(tmp_path, monkeypatch):
    client, store = _client(tmp_path)
    store.upsert_user(UserRecord(id="alice", email="alice@example.com"))
    result = SearchQueryResult(
        original_query="deployment guide",
        executed_queries=["deployment guide"],
        results=[],
    )

    content_type, lines = _stream_lines(client, monkeypatch, result)

    assert content_type.startswith("application/x-ndjson")
    payloads = [json.loads(line) for line in lines]
    assert [p["type"] for p in payloads] == ["search_queries", "search_docs"]
    assert payloads[0]["all_executed_queries"] == ["deployment guide"]


def test_stream_is_ndjson_and_not_sse(tmp_path, monkeypatch):
    """Pins the distinction the docstring used to get wrong."""
    client, store = _client(tmp_path)
    store.upsert_user(UserRecord(id="alice", email="alice@example.com"))
    result = SearchQueryResult(
        original_query="deployment guide",
        executed_queries=["deployment guide"],
        results=[],
    )

    _, lines = _stream_lines(client, monkeypatch, result)

    assert lines, "expected at least one frame"
    assert not any(line.startswith("data:") for line in lines)


def test_stream_re_emits_queries_after_expansion(tmp_path, monkeypatch):
    client, store = _client(tmp_path)
    store.upsert_user(UserRecord(id="alice", email="alice@example.com"))
    result = SearchQueryResult(
        original_query="deployment guide",
        executed_queries=["deployment guide", "how to deploy"],
        results=[],
    )

    _, lines = _stream_lines(client, monkeypatch, result)

    payloads = [json.loads(line) for line in lines]
    assert [p["type"] for p in payloads] == [
        "search_queries",
        "search_queries",
        "search_docs",
    ]
    assert payloads[1]["all_executed_queries"] == ["deployment guide", "how to deploy"]


def test_stream_emits_error_packet_when_search_fails(tmp_path, monkeypatch):
    client, store = _client(tmp_path)
    store.upsert_user(UserRecord(id="alice", email="alice@example.com"))

    _, lines = _stream_lines(client, monkeypatch, RuntimeError("index offline"))

    payloads = [json.loads(line) for line in lines]
    assert [p["type"] for p in payloads] == ["search_queries", "search_error"]
    assert "index offline" in payloads[-1]["error"]
