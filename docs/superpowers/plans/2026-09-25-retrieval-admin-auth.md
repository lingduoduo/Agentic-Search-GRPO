# Retrieval Admin Auth Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** The retrieval server's admin, eval and optimize routes require an
admin, and `PATCH /api/admin/retrieval/config` accepts only the one field it
actually applies.

**Architecture:** `create_app` resolves `AppSettings` and builds one
`make_require_admin` dependency. It applies it to the two admin routes and
passes it to the eval and optimize routers. The patch model forbids extra
fields.

**Tech Stack:** FastAPI, pydantic v2, pytest + `TestClient`.

**Spec:** `docs/superpowers/specs/2026-09-25-retrieval-admin-auth-design.md`

## Global Constraints

- `/health` and `/search` stay unauthenticated.
- `create_optimize_router()` with no argument keeps working unauthenticated,
  because its existing unit tests call it that way.
- Error codes: 401 when there is no or an invalid token, 403 for a non-admin,
  422 for an unknown PATCH field.

## Review Focus

- A PATCH with a removed field: it must be 422 with no side effect, never a
  partial apply (Task 1, `test_patch_rejects_unapplied_fields`).
- The `AGENTIC_SEARCH_DEV_ADMIN` bypass still opens the admin routes locally
  (Task 1, `test_dev_bypass_opens_admin_routes`).
- A negative `result_cache_ttl` gives 422 (Task 1).

---

### Task 1: Guard admin routes and make the PATCH honest

**Files:**
- Modify: `src/internal/servers/retrieval/server.py` (`create_app`,
  `RetrievalConfigPatch`, `patch_config`, router includes)
- Modify: `src/internal/servers/retrieval/optimize_router.py`
  (`create_optimize_router` signature)
- Modify: `docs/api-reference.md` (~:68-74), `docs/retrieval.md` (~:560-566)
- Test: `tests/unit/servers/retrieval/test_new_server.py`

**Interfaces:**
- Produces: `create_app(service: RetrievalService | None = None, app_settings:
  AppSettings | None = None) -> FastAPI` and
  `create_optimize_router(require_admin: Callable | None = None) -> APIRouter`.

- [ ] **Step 1: Write the failing tests** (append to `test_new_server.py`)

```python
import os

import pytest

from src.internal.auth.users import generate_user_jwt_token
from src.internal.configs import AppSettings, AuthSettings


def _client(svc=None, *, bypass: bool = False) -> TestClient:
    settings = AppSettings(auth=AuthSettings(dev_admin_bypass=bypass))
    return TestClient(create_app(svc or _make_service([]), app_settings=settings))


def _bearer(**extra) -> dict[str, str]:
    token = generate_user_jwt_token(user_id="someone", extra=extra or None)
    return {"Authorization": f"Bearer {token}"}


@pytest.mark.parametrize(
    ("method", "path", "body"),
    [
        ("get", "/api/admin/retrieval/stats", None),
        ("patch", "/api/admin/retrieval/config", {"result_cache_ttl": 5}),
        ("post", "/internal/search/sparse", {"query": "q", "top_k": 5}),
        ("post", "/internal/optimize/bm25-tune", {"qa_pairs_path": "/etc/passwd"}),
    ],
)
def test_admin_routes_require_a_token(method, path, body):
    kwargs = {"json": body} if body is not None else {}
    assert getattr(_client(), method)(path, **kwargs).status_code == 401


def test_non_admin_token_is_forbidden():
    assert (
        _client().get("/api/admin/retrieval/stats", headers=_bearer()).status_code
        == 403
    )


def test_admin_token_reads_stats():
    resp = _client().get("/api/admin/retrieval/stats", headers=_bearer(role="admin"))
    assert resp.status_code == 200


def test_dev_bypass_opens_admin_routes():
    assert _client(bypass=True).get("/api/admin/retrieval/stats").status_code == 200


def test_patch_applies_result_cache_ttl():
    svc = _make_service([])
    svc._result_cache = MagicMock()
    svc._result_cache._ttl = 300
    resp = _client(svc).patch(
        "/api/admin/retrieval/config",
        json={"result_cache_ttl": 5},
        headers=_bearer(role="admin"),
    )
    assert resp.status_code == 200
    assert resp.json() == {"applied": ["result_cache_ttl"]}
    assert svc._result_cache._ttl == 5


@pytest.mark.parametrize(
    "body", [{"rrf_k": 80}, {"mmr_lambda": 0.4}, {"nprobe": 96}, {"result_cache_ttl": -1}]
)
def test_patch_rejects_unapplied_fields(body, monkeypatch):
    monkeypatch.delenv("RRF_K", raising=False)
    monkeypatch.delenv("MMR_LAMBDA", raising=False)
    resp = _client().patch(
        "/api/admin/retrieval/config", json=body, headers=_bearer(role="admin")
    )
    assert resp.status_code == 422
    assert "RRF_K" not in os.environ and "MMR_LAMBDA" not in os.environ


def test_health_and_search_stay_open():
    client = _client()
    assert client.get("/health").status_code == 200
    assert client.post("/search", json={"query": "q"}).status_code == 200
```

- [ ] **Step 2: Run and verify they fail**

Run: `pytest tests/unit/servers/retrieval/test_new_server.py -q`
Expected: failures, starting with `create_app() got an unexpected keyword
argument 'app_settings'`.

- [ ] **Step 3: Implement**

In `optimize_router.py`:

```python
from collections.abc import Callable

from fastapi import APIRouter, Depends


def create_optimize_router(require_admin: Callable | None = None) -> APIRouter:
    deps = [Depends(require_admin)] if require_admin is not None else []
    router = APIRouter(prefix="/internal/optimize", dependencies=deps)
```

In `server.py`:

```python
from fastapi import Depends, FastAPI
from pydantic import BaseModel, ConfigDict, Field

from src.internal.configs import AppSettings, load_app_settings
from src.internal.servers._auth import make_require_admin


class RetrievalConfigPatch(BaseModel):
    """Only what the running service actually applies. RRF k and MMR lambda
    are fixed when the service is built, so they are not accepted here."""

    model_config = ConfigDict(extra="forbid")

    result_cache_ttl: int | None = Field(default=None, ge=0)


def create_app(
    service: RetrievalService | None = None,
    app_settings: AppSettings | None = None,
) -> FastAPI:
    ...
    require_admin = make_require_admin(app_settings or load_app_settings())
    admin = [Depends(require_admin)]
    ...
    app.include_router(create_eval_router(_service, require_admin=require_admin))
    app.include_router(create_optimize_router(require_admin=require_admin))

    @app.get("/api/admin/retrieval/stats", dependencies=admin)
    ...

    @app.patch("/api/admin/retrieval/config", dependencies=admin)
    def patch_config(patch: RetrievalConfigPatch) -> dict:
        applied: list[str] = []
        if (
            patch.result_cache_ttl is not None
            and getattr(_service, "_result_cache", None) is not None
        ):
            _service._result_cache._ttl = patch.result_cache_ttl
            applied.append("result_cache_ttl")
        return {"applied": applied}
```

Move `RetrievalConfigPatch` to module level, next to the other models, and
delete the "no auth in dev" comment.

- [ ] **Step 4: Fix docs**

In `docs/api-reference.md` and `docs/retrieval.md`, replace the PATCH examples
with this:

```bash
curl -s -X PATCH http://localhost:8001/api/admin/retrieval/config \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $ADMIN_TOKEN" \
  -d '{"result_cache_ttl": 600}'
# → {"applied": ["result_cache_ttl"]}
```

Add a note under the examples: "Admin-only (401/403 otherwise). RRF `k` and
MMR `λ` are fixed when the service is built; unknown fields are rejected
with 422." In `docs/retrieval.md`, the stats example also moves to `:8001`
because that is the server that serves it.

- [ ] **Step 5: Run, mutation-check, full suite, commit**

1. Run `pytest tests/unit/servers/retrieval/ -q` and expect a pass.
2. Mutation check: remove `dependencies=admin` from `stats`, confirm the
   401/403 tests fail, then restore it.
3. Run `pytest -q -p no:cacheprovider` and `ruff check . && ruff format
   --check .`.
4. Commit.
