# Serving-path ACL filtering and caching — implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** Close the two serving-side ACL gaps and add a process-local TTL cache at the three request-path choke points, keyed so a hit can never widen access.

**Architecture:** One torch-free `acl_allows` shared by the demo/hybrid servers, the `RetrievalService` local backend, and the pipeline retrieval stage. One `TTLCache` primitive behind a process-level `serving_cache()` that the web lifespan configures from `AGENTIC_SEARCH_SEARCH_CACHE_TTL`.

**Tech Stack:** Python >=3.10, stdlib only for the cache, pytest, Ruff.

**Spec:** `docs/superpowers/specs/2026-09-10-serving-acl-and-cache-design.md`

## Global Constraints

- TDD per task: write the failing test, watch it fail, implement, watch it pass, mutation-check by reverting the implementation.
- No new dependencies. `src/internal/retrieval/acl.py` and `src/internal/cache/ttl_cache.py` import only the stdlib.
- Existing behaviour of `demo.py`, `hybrid.py`, `SearchFilters.matches` and every `_enforce_access` call site is unchanged.
- Branch `feat/serving-acl-and-cache`, worktree `.claude/worktrees/serving-acl-cache`. Never commit to main.
- Local Python: `/Users/linghuang/miniconda3/envs/agentic-search-local/bin/python`.

### Task 1: Shared `acl_allows`

**Files:** create `src/internal/retrieval/acl.py`, `tests/unit/retrieval/test_acl.py`; modify `src/internal/servers/retrieval/demo.py`.

- [ ] Tests: no filters → True; filters without `access_acl` → True; no declared ACL → True; str ACL intersecting → True; list ACL disjoint → False.
- [ ] Implement `acl_allows(metadata, filters)`; make `demo._allowed_by_acl` delegate with `document.get("metadata")`.
- [ ] `pytest tests/unit/retrieval/test_acl.py tests/unit/servers/retrieval/` green.

### Task 2: `RetrievalService` local backend honours `access_acl`

**Files:** `src/internal/retrieval/backends/local.py`, `tests/unit/retrieval/test_retrieval_backend.py`.

- [ ] Tests: sparse search with `{"access_acl": ["user:a"]}` keeps a doc with no ACL, keeps `{"metadata": {"acl": ["user:a"]}}`, keeps top-level `"acl": "user:a"`, drops `{"metadata": {"acl": ["user:b"]}}`; a combined `{"access_acl": [...], "lang": "en"}` still equality-matches `lang`.
- [ ] Implement: split `access_acl` out of the filter dict, apply `acl_allows` on the nested `metadata` dict when present else on `r.metadata`, equality for the rest.

### Task 3: `SearchClientRetrievalStage` enforces

**Files:** `src/internal/search/stages.py`, `tests/unit/search/test_stages.py`.

- [ ] Test: fake client returns one public and one `acl: ["user:b"]` result for filters `{"access_acl": ["user:a"]}`; the `CandidateSet` holds only the public one.
- [ ] Implement the `acl_allows` post-filter before constructing the `CandidateSet`.

### Task 4: `TTLCache` and `serving_cache()`

**Files:** create `src/internal/cache/ttl_cache.py`, `src/internal/cache/serving.py`, `tests/unit/cache/test_ttl_cache.py`, `tests/unit/cache/test_serving_cache.py`.

- [ ] Tests: miss returns None; hit returns the value; entry expires once the injected clock passes `ttl`; a `max_entries=2` cache evicts the least recently used on the third insert; `stats()` counts; `configure_serving_cache(0)` leaves `serving_cache()` None; `configure` then `reset` returns to None.
- [ ] Implement both modules.

### Task 5: `SearchClient.retrieve` caches per query

**Files:** `src/context/retrieval/client.py`, `tests/unit/test_search_client.py`.

- [ ] Tests (with `configure_serving_cache(60)` in a fixture that resets after): second identical `retrieve` makes zero posts; different `filters` posts again; `["a", "b"]` after `["a"]` posts only `["b"]` and returns `[a_rows, b_rows]`; an empty row is not cached; with the cache unconfigured every call posts.
- [ ] Implement the lookup/miss-post/merge with the key from the spec; re-materialise `SearchResult`s from cached raw items.

### Task 6: `search_tool` caches web providers

**Files:** `src/internal/tools/search.py`, `tests/unit/test_search_tools.py`.

- [ ] Tests: `serpapi_search` is invoked once for two identical `search_tool(provider="serpapi")` calls; an error page is not cached; `provider="retrieval"` bypasses this layer (its caching lives in `SearchClient`).
- [ ] Implement a small `_cached_web_search` wrapper around the three web providers.

### Task 7: `RerankHTTPRankingStage` caches the ranked list

**Files:** `src/internal/search/stages.py`, `tests/unit/search/test_stages.py`.

- [ ] Tests: two identical `rank` calls make one HTTP post; changing a candidate's contents posts again.
- [ ] Implement with the key from the spec; cache the decoded `ranked` list.

### Task 8: Settings and lifespan wiring

**Files:** `src/internal/configs/app_configs.py`, `src/internal/servers/web/app.py`, `tests/unit/servers/web/test_serving_cache_lifespan.py`, the existing config test file.

- [ ] Tests: `load_app_settings({"AGENTIC_SEARCH_SEARCH_CACHE_TTL": "0"})` yields `search_cache_ttl_seconds == 0`; default is 300; `SearchExperienceSettings.from_app_settings` carries it; `with TestClient(app)` configures `serving_cache()` with the TTL and resets it on exit; TTL 0 never configures.
- [ ] Implement the field, the env read, and the lifespan calls.

### Task 9: Docs

**Files:** `docs/configuration.md`, `docs/retrieval.md`, `docs/request-routing.md`.

- [ ] Document `AGENTIC_SEARCH_SEARCH_CACHE_TTL`; add a "Serving cache" section to `docs/retrieval.md`; note in `docs/request-routing.md#access-control` that `RetrievalService` and `SearchClientRetrievalStage` now honour/enforce `access_acl`.

### Task 10: Verify and ship

- [ ] `ruff check . --fix && ruff format .`; full `pytest` green (note the pre-existing intent p95 latency bar if it trips).
- [ ] Independent code review of the whole branch; fix findings.
- [ ] Push, open PR with spec + plan.
