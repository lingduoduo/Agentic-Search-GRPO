# Retrieval server admin surface: require an admin, drop fake knobs: design

## Problem

`src/internal/servers/retrieval/server.py` (the `RetrievalService` app) has
three problems.

- **Its admin routes have no auth:**
  - `GET /api/admin/retrieval/stats`
  - `PATCH /api/admin/retrieval/config`
  - the eval router (`/internal/search/*`), mounted with `require_admin=None`
    even though its own docstring says to "pass
    require_admin=make_require_admin(app_settings) in production"
  - the optimize router (`/internal/optimize/*`), which reads any
    `qa_pairs_path` the caller names from the server's disk
- **The PATCH reports changes it never makes.** `rrf_k` and `mmr_lambda` are
  written to `os.environ["RRF_K"]` and `os.environ["MMR_LAMBDA"]`, but nothing
  reads either variable. RRF's `k` is the module constant `_RRF_K = 60`
  (`fusion.py:23`, `hybrid_retriever.py:18`). The endpoint still answers
  `{"applied": ["rrf_k", "mmr_lambda"]}`. Only `result_cache_ttl` has an
  effect.
- **The docs promise more than exists.** `docs/api-reference.md:71` and
  `docs/retrieval.md:562` show a PATCH with `nprobe`, which the model does not
  accept. `docs/retrieval.md` also puts the endpoint on `:7860`, the web app,
  which does not mount it. The api-reference example also sends no token.

This is the repo's "advertised capability" defect class: a surface that says
it did something it did not do.

## Decision (agreed with the user: "auth + honest")

1. **Every route except `/health`, `/search` and `/retrieve`-style reads
   requires an admin.** `create_app(service=None, app_settings=None)` resolves
   `app_settings or load_app_settings()` and builds
   `require_admin = make_require_admin(settings)`, the same dependency the web
   app's routers use. It then applies it in three places:
   - `Depends(require_admin)` on `/api/admin/retrieval/stats` and
     `/api/admin/retrieval/config`
   - `create_eval_router(_service, require_admin=require_admin)`
   - `create_optimize_router(require_admin=require_admin)`, which gains an
     optional `require_admin` parameter matching the eval router's
     (`dependencies=[Depends(...)]` when it is given, none when it is `None`,
     so the router's own unit tests keep working unauthenticated).

   `make_require_admin` already handles a standalone service with no user
   store: it accepts a signed JWT with `role: admin`, or a configured super
   user. The `AGENTIC_SEARCH_DEV_ADMIN` bypass works too.
2. **The PATCH accepts only what it applies.** `RetrievalConfigPatch` becomes
   `model_config = ConfigDict(extra="forbid")` with the single field
   `result_cache_ttl: int | None = Field(default=None, ge=0)`. `rrf_k`,
   `mmr_lambda` and `nprobe` now get a 422 instead of a false "applied". The
   `os.environ` writes are deleted.
3. **Docs are corrected.** Both examples show the `:8001` retrieval server, an
   `Authorization: Bearer $ADMIN_TOKEN` header, and `result_cache_ttl` only.
   They state that RRF `k` and MMR `λ` are fixed at build time.

`/search` and `/health` stay open. The data plane is not in scope: the web app
calls `/search` server-to-server, and retrieval ACLs are enforced by
`acl_allows` per document.

## Out of scope

- Making `rrf_k` and `mmr_lambda` runtime-tunable. The user chose "honest"
  over "make knobs real".
- `demo.py` and `hybrid.py`, the servers the compose stack actually runs. They
  mount no admin routes.

## Testing

In `tests/unit/servers/retrieval/test_new_server.py`:

- With `AuthSettings()`, which means no bypass: `stats` returns 401 with no
  token, `PATCH config` returns 401, `POST /internal/search/sparse` returns 401,
  and `POST /internal/optimize/bm25-tune` returns 401.
- A non-admin JWT gets 403 on `stats`.
- An admin JWT (`extra={"role": "admin"}`) gets 200 on `stats`. `PATCH
  {"result_cache_ttl": 5}` returns `{"applied": ["result_cache_ttl"]}` and
  changes `_result_cache._ttl`.
- `PATCH {"rrf_k": 80}` as an admin returns 422, and `os.environ` gains no
  `RRF_K`.
- `/health` and `/search` still answer without a token.
- Mutation check: drop the `Depends` on `stats` and watch the 401 test fail.
