from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.internal.auth import generate_user_jwt_token
from src.internal.db.store import AgenticSearchStore
from src.internal.memory.router import create_memory_router


class _FakeLLM:
    def __init__(self, text="[]"):
        self._text = text

    def complete(self, prompt, **kwargs):
        return self._text


def _client(db, llm=None):
    app = FastAPI()
    app.include_router(create_memory_router(db, llm))
    return TestClient(app)


def test_save_list_search_consolidate_profile_without_llm():
    db = AgenticSearchStore(":memory:")
    c = _client(db)

    r = c.post("/api/memory/save", json={"text": "User enjoys hiking in the mountains"})
    assert r.status_code == 200 and r.json()["memory_id"]

    listed = c.get("/api/memory/list").json()["memories"]
    assert listed[0]["text"] == "User enjoys hiking in the mountains"

    hits = c.post(
        "/api/memory/search", json={"query": "mountain hiking", "max_results": 5}
    ).json()
    assert hits["results"] and "hiking" in hits["results"][0]["text"]

    report = c.post("/api/memory/consolidate", json={"resolve_conflicts": True}).json()[
        "report"
    ]
    assert report["initial"] == 1 and report["final"] == 1

    assert c.get("/api/memory/profile").json()["profile"] == []
    db.close()


def test_llm_endpoints_503_without_llm():
    db = AgenticSearchStore(":memory:")
    c = _client(db, llm=None)
    assert c.post("/api/memory/profile/generate").status_code == 503
    assert c.post("/api/memory/curate", json={}).status_code == 503
    db.close()


def test_generate_profile_with_fake_llm():
    db = AgenticSearchStore(":memory:")
    db.add_user_memory("default_user", "User is a software engineer at TechCorp")
    llm = _FakeLLM(
        '[{"topic":"work","subtopic":"role","content":"Software engineer at TechCorp"}]'
    )
    c = _client(db, llm)
    profile = c.post("/api/memory/profile/generate").json()["profile"]
    assert profile[0]["topic"] == "work"
    assert (
        c.get("/api/memory/profile").json()["profile"][0]["content"]
        == "Software engineer at TechCorp"
    )
    db.close()


# --- Strict mode: AGENTIC_SEARCH_MEMORY_REQUIRE_AUTH ---------------------------
#
# Off by default, so the documented unauthenticated research/CLI flow (every test
# above) keeps working and anonymous callers keep sharing the ``default_user``
# bucket. On, an anonymous caller is refused rather than pooled with every other
# anonymous caller.


def _strict_client(db, llm=None):
    app = FastAPI()
    app.include_router(create_memory_router(db, llm, require_auth=True))
    return TestClient(app)


def _bearer(user_id: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {generate_user_jwt_token(user_id=user_id)}"}


def test_strict_mode_refuses_anonymous_callers():
    db = AgenticSearchStore(":memory:")
    c = _strict_client(db)
    for method, path, body in (
        ("get", "/api/memory/list", None),
        ("post", "/api/memory/save", {"text": "a fact"}),
        ("post", "/api/memory/search", {"query": "fact"}),
        ("post", "/api/memory/consolidate", {}),
        ("get", "/api/memory/profile", None),
    ):
        r = c.get(path) if method == "get" else c.post(path, json=body)
        assert r.status_code == 401, f"{method.upper()} {path} was not refused"
    db.close()


def test_strict_mode_still_serves_an_authenticated_caller_their_own_bucket():
    """The control: strict mode withholds rather than refusing everyone."""
    db = AgenticSearchStore(":memory:")
    db.add_user_memory("default_user", "someone else's memory")
    c = _strict_client(db)

    saved = c.post(
        "/api/memory/save", json={"text": "alice fact"}, headers=_bearer("alice")
    )
    assert saved.status_code == 200

    listed = c.get("/api/memory/list", headers=_bearer("alice")).json()["memories"]
    assert [m["text"] for m in listed] == ["alice fact"]
    db.close()


def test_default_mode_leaves_the_anonymous_bucket_reachable():
    """The July research-use ruling: off by default, behaviour unchanged."""
    db = AgenticSearchStore(":memory:")
    c = _client(db)
    assert c.post("/api/memory/save", json={"text": "anon fact"}).status_code == 200
    assert c.get("/api/memory/list").json()["memories"][0]["text"] == "anon fact"
    db.close()


def test_search_category_filter_and_app_state_encoder(monkeypatch):
    import numpy as np

    import src.internal.memory.service as svc

    def _boom():
        raise AssertionError("maybe_build_encoder must not be called per request")

    monkeypatch.setattr(svc, "maybe_build_encoder", _boom)
    db = AgenticSearchStore(":memory:")
    db.add_user_memory("default_user", "User prefers concise answers")
    db.add_user_memory("default_user", "User is allergic to peanuts")
    calls = []

    def fake_encoder(texts):
        calls.append(texts)
        return np.ones((len(texts), 2), dtype=np.float32)

    app = FastAPI()
    app.state.memory_encoder = fake_encoder
    app.include_router(create_memory_router(db, None))
    c = TestClient(app)
    by_cat = {r.category: r.id for r in db.get_user_memory_records("default_user")}
    assert set(by_cat) == {"preferences", "constraints"}
    res = c.post(
        "/api/memory/search",
        json={"query": "answers peanuts", "category": "preferences"},
    ).json()["results"]
    assert calls
    assert [h["id"] for h in res] == [by_cat["preferences"]]
    db.close()


def test_save_with_category_and_list_filter():
    db = AgenticSearchStore(":memory:")
    c = _client(db)
    c.post("/api/memory/save", json={"text": "Moved to Munich", "category": "profile"})
    c.post("/api/memory/save", json={"text": "User prefers concise answers"})
    listed = c.get("/api/memory/list").json()["memories"]
    assert [(m["text"], m["category"]) for m in listed] == [
        ("Moved to Munich", "profile"),
        ("User prefers concise answers", "preferences"),
    ]
    only = c.get("/api/memory/list", params={"category": "profile"}).json()
    assert [m["text"] for m in only["memories"]] == ["Moved to Munich"]
    db.close()


def test_index_shape_and_order():
    db = AgenticSearchStore(":memory:")
    c = _client(db)
    c.post("/api/memory/save", json={"text": "User prefers concise answers"})
    c.post("/api/memory/save", json={"text": "I'm allergic to peanuts"})
    db.add_user_memory("someone_else", "User prefers long answers")
    body = c.get("/api/memory/index").json()
    assert body["root"] == {
        "summary": "constraints 1 · preferences 1",
        "memory_count": 2,
    }
    assert [cat["name"] for cat in body["categories"]] == [
        "constraints",
        "preferences",
    ]
    first = body["categories"][0]
    assert set(first) == {"name", "summary", "memory_count", "memories"}
    assert first["memory_count"] == 1
    assert [m["text"] for m in first["memories"]] == ["I'm allergic to peanuts"]
    assert set(first["memories"][0]) == {"id", "text", "updated_at"}
    assert all(not n.startswith("__") for n in (x["name"] for x in body["categories"]))
    db.close()


def test_index_uses_stored_summaries_after_consolidate():
    db = AgenticSearchStore(":memory:")
    c = _client(db)
    c.post("/api/memory/save", json={"text": "User prefers concise answers"})
    report = c.post("/api/memory/consolidate", json={}).json()["report"]
    assert {
        "initial",
        "duplicates_removed",
        "conflicts_resolved",
        "merged",
        "reclassified",
        "summaries_rebuilt",
        "final",
    } <= set(report)
    body = c.get("/api/memory/index").json()
    assert body["categories"][0]["summary"].startswith("1 memories; key terms:")
    db.close()


def test_consolidate_passes_llm_and_app_encoder(monkeypatch):
    import src.internal.memory.service as svc

    seen = {}

    def fake(store, user_id, resolve_conflicts=True, *, llm=None, encoder=None):
        seen.update(llm=llm, encoder=encoder, resolve=resolve_conflicts)
        return {"initial": 0}

    monkeypatch.setattr(svc, "consolidate_memories", fake)
    db = AgenticSearchStore(":memory:")
    llm = _FakeLLM()

    def enc(texts):
        return None

    app = FastAPI()
    app.state.memory_encoder = enc
    app.include_router(create_memory_router(db, llm))
    c = TestClient(app)
    c.post("/api/memory/consolidate", json={"resolve_conflicts": False})
    assert seen == {"llm": llm, "encoder": enc, "resolve": False}
    db.close()
