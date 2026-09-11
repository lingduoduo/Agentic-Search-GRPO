"""Session endpoints enforce ownership.

`docs/superpowers/specs/2026-08-14-anonymous-identity-investigation-design.md`
asked whether two signed-out callers can see each other's transcripts. The
answer was broader and worse than the question: **any caller, authenticated or
not, could read any session by id** — including a session owned by a signed-in
user, message bodies included — and could append to it through `/api/agent`.

These began as `xfail(strict=True)` evidence and became assertions once the
guard landed. Strict xfail is what made that transition safe: the moment the
endpoints were fixed the tests XPASSed, which strict mode reports as a failure,
so the markers could not be left behind to silently stop protecting anything.

An *anonymous* session (`user_id IS NULL`) is still readable by anyone holding
its id — see `_caller_may_use_session` for why that is a deliberate narrowing
rather than a gap left open.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src.internal.db.models import UserRecord
from src.internal.db.store import AgenticSearchStore
from src.internal.servers.web.app import SearchExperienceSettings, create_web_app


@pytest.fixture()
def app_and_store():
    directory = Path(tempfile.mkdtemp())
    db_path = directory / "sessions.sqlite3"
    app = create_web_app(SearchExperienceSettings(db_path=db_path))
    # A second handle on the same file: the app owns its own connection, and
    # seeding through the API would require the very auth this is testing.
    return app, AgenticSearchStore(str(db_path))


def _owned_session(store: AgenticSearchStore) -> str:
    owner = store.upsert_user(UserRecord(id="user_alice", email="alice@example.com"))
    session = store.create_chat_session(user_id=owner.id, title="Alice payroll thread")
    store.add_chat_message(
        session.id, role="user", content="my salary is 120k, SSN ends 4471"
    )
    store.add_chat_message(session.id, role="assistant", content="noted")
    return session.id


def test_reading_another_users_session_is_refused(app_and_store):
    """An unauthenticated caller must not read a signed-in user's transcript.

    Session ids are `session_<uuid4hex>`, so this is not brute-forceable — but
    it is a plain IDOR: any id that leaks through a URL, a log line, a referrer
    or a shared link becomes a full transcript read, message bodies included.
    """
    app, store = app_and_store
    session_id = _owned_session(store)

    with TestClient(app) as client:
        response = client.get(f"/api/sessions/{session_id}")

    assert response.status_code == 404, (
        f"anonymous caller received {response.status_code} with "
        f"{len(response.json().get('messages', []))} messages"
    )


def test_posting_to_another_users_session_is_refused(app_and_store):
    """The write half, which is worse than the read half.

    ``_ensure_session`` accepts any *existing* session id with no ownership
    test. The handler then loads that session's history into the model context
    and appends the caller's message to it — so a leaked id is not only a read,
    it is an injection into someone else's conversation.
    """
    app, store = app_and_store
    session_id = _owned_session(store)

    with TestClient(app) as client:
        response = client.post(
            "/api/agent",
            json={"query": "and what was that salary again", "session_id": session_id},
        )

    assert response.status_code == 404, (
        f"anonymous caller wrote to another user's session: {response.status_code}"
    )
