# Routing Unbacked-Target Fall-Through Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stop `ROUTING_ENABLED` from returning zero results when the router picks a target the service cannot execute; annotate the search instead.

**Architecture:** `RetrievalService.search` currently early-returns `([], "routed:<target>")` for the SQL, GRAPH, and API targets. That early return becomes a `routed` suffix string that is appended at each of `search`'s two real exits — the result-cache hit and the final fused return — so retrieval runs normally and the routing decision stays visible in the mode.

**Tech Stack:** Python 3, pytest, `src/internal/retrieval/service.py`, `src/internal/routing/`.

**Spec:** `docs/superpowers/specs/2026-09-20-routing-unbacked-target-fallthrough-design.md`

## Global Constraints

- The router's cue lists in `src/internal/routing/router.py` are unchanged. Which queries route where must stay exactly as it is; only the consequence changes.
- The mode marker is spelled as a suffix with a leading `+`: `f"+routed:{decision.retriever.value}"`, matching the existing `+rag_fusion` and `+reranked` convention in the same function.
- `RetrieverTarget` stays imported *inside* `search`, not at module scope. The deferred import keeps `routing/` a lazy dependency of the retrieval service.
- Do not touch `src/internal/routing/construction/`, `src/internal/routing/registry.py`, or `src/internal/retrieval/eval_runner.py`.
- Run `ruff check . --fix && ruff format .` before each commit. A pre-commit `ruff-format` hook aborts the commit if formatting is dirty.

---

### Task 1: Unbacked targets fall through to retrieval

**Files:**
- Modify: `src/internal/retrieval/service.py:190-212` (the routing branch in `search`), plus its two return statements at roughly `:215` (`return cached, "cached"`) and `:329` (`return fused, mode`)
- Test: `tests/unit/routing/test_service_routing.py`

**Interfaces:**
- Consumes: `RetrievalService(backend, router=..., result_cache=...)`; `Router(RouteRegistry(DEFAULT_ROUTES))`; `RetrieverTarget` from `src.internal.routing.route`.
- Produces: `RetrievalService.search(query, top_k) -> tuple[list[RetrievalResult], str]` where the mode string ends with `+routed:sql`, `+routed:graph`, or `+routed:api` when the router picked an unbacked target, and contains no `routed:` substring otherwise.

- [ ] **Step 1: Replace the pinned-bug test and add the new ones**

`test_routing_to_sql_short_circuits_to_empty` asserts `results == []`. That is the defect written down as a contract, so it is deleted rather than kept. Replace the whole file body below the existing `_StubBackend` with this, keeping the existing imports and adding `pytest`:

```python
import pytest

from src.internal.retrieval.backends.base import RetrievalResult
from src.internal.retrieval.service import RetrievalService
from src.internal.routing.registry import DEFAULT_ROUTES, RouteRegistry
from src.internal.routing.router import Router


class _StubBackend:
    def search_sparse(self, query, top_k, filters=None):
        return [RetrievalResult(doc_id="d1", title="t", text="x", url=None, score=1.0)]

    def search_dense(self, query, top_k, filters=None):
        raise NotImplementedError


class _StubCache:
    """A result cache that always hits, so the cached exit is exercised."""

    def __init__(self, results):
        self._results = results

    def get(self, query, filters, top_k):
        return list(self._results)

    def set(self, query, filters, top_k, results):
        pass


def _routed_service(**kwargs):
    return RetrievalService(
        _StubBackend(), router=Router(RouteRegistry(DEFAULT_ROUTES)), **kwargs
    )


def test_routing_disabled_runs_retrieval():
    svc = RetrievalService(_StubBackend())  # no router
    results, mode = svc.search("how many docs are there", top_k=3)
    assert results and results[0].doc_id == "d1"
    assert "routed:" not in mode


def test_routing_to_hybrid_runs_retrieval():
    svc = _routed_service()
    results, mode = svc.search("what is reciprocal rank fusion", top_k=3)
    assert results and results[0].doc_id == "d1"
    assert "routed:" not in mode


@pytest.mark.parametrize(
    "query,target",
    [
        ("how many papers per year", "sql"),
        ("papers related to BM25", "graph"),
        ("latest FAISS release", "api"),
    ],
)
def test_unbacked_target_still_returns_results(query, target):
    """An unbacked route annotates the search; it must not cost the results."""
    svc = _routed_service()
    results, mode = svc.search(query, top_k=3)
    assert results and results[0].doc_id == "d1"
    assert mode.endswith(f"+routed:{target}")


def test_unbacked_target_annotates_a_cached_hit():
    """The cache exit reports the routing decision like the uncached one."""
    cached = [RetrievalResult(doc_id="c1", title="t", text="x", url=None, score=1.0)]
    svc = _routed_service(result_cache=_StubCache(cached))
    results, mode = svc.search("how many papers per year", top_k=3)
    assert results and results[0].doc_id == "c1"
    assert mode == "cached+routed:sql"
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/unit/routing/test_service_routing.py -v`

Expected: the three `test_unbacked_target_still_returns_results` cases FAIL (they get `results == []` and `mode == "routed:sql"`, so both assertions break), and `test_unbacked_target_annotates_a_cached_hit` FAILS (it gets `"routed:sql"`, never reaching the cache). The two pass-through tests PASS — they already hold today and are here as regression cover.

- [ ] **Step 3: Turn the early return into an annotation**

In `src/internal/retrieval/service.py`, inside `search`, replace this block:

```python
        if self._router is not None:
            from src.internal.routing.route import RetrieverTarget

            decision = self._router.route(query)
            if decision.retriever in (
                RetrieverTarget.SQL,
                RetrieverTarget.GRAPH,
                RetrieverTarget.API,
            ):
                # No execution backend for these targets — construct-only.
                # Degrade to empty results so routing never breaks a request.
                return [], f"routed:{decision.retriever.value}"
```

with:

```python
        routed = ""
        if self._router is not None:
            from src.internal.routing.route import RetrieverTarget

            decision = self._router.route(query)
            if decision.retriever in (
                RetrieverTarget.SQL,
                RetrieverTarget.GRAPH,
                RetrieverTarget.API,
            ):
                # No execution backend for these targets — construct-only.
                # Run ordinary retrieval anyway, since an unbacked route must
                # not cost the caller its results, and record the decision.
                routed = f"+routed:{decision.retriever.value}"
```

- [ ] **Step 4: Carry the annotation to both exits**

Still in `search`, the result-cache hit changes from:

```python
                return cached, "cached"
```

to:

```python
                return cached, "cached" + routed
```

and the final return at the end of `search` changes from:

```python
        return fused, mode
```

to:

```python
        return fused, mode + routed
```

Append `routed` last, after `+reranked` is already applied, so the routing
marker is the final segment and `mode.endswith(...)` holds regardless of which
other suffixes the search picked up.

- [ ] **Step 5: Run the tests to verify they pass**

Run: `pytest tests/unit/routing/test_service_routing.py -v`
Expected: all 6 PASS (2 pass-through, 3 parametrized, 1 cached).

- [ ] **Step 6: Mutation-check the new tests**

A test that cannot fail is worse than no test. Prove each new assertion is load-bearing:

```bash
# Revert just the behavior, keeping the variable so the file still parses.
python3 - <<'PY'
import pathlib
p = pathlib.Path("src/internal/retrieval/service.py")
s = p.read_text()
s = s.replace(
    '                routed = f"+routed:{decision.retriever.value}"',
    '                return [], f"routed:{decision.retriever.value}"',
)
p.write_text(s)
PY
pytest tests/unit/routing/test_service_routing.py -v
```

Expected: the 3 parametrized cases and the cached case all go RED. If any stays green, that test is not testing the fix — fix the test before continuing.

Restore with `git checkout src/internal/retrieval/service.py`, then re-run the
file and confirm 6 PASS again. Run `git diff --stat` before moving on to be
sure the restore left no mutated remnant, and if anything looks stale delete
`__pycache__` — a reverted mutation can leave mutated bytecode behind and fake
a red suite.

- [ ] **Step 7: Run the neighbouring suites**

Run: `pytest tests/unit/routing/ tests/unit/test_retrieval_service.py -v`

Expected: all PASS. If `tests/unit/test_retrieval_service.py` does not exist under that name, run `pytest tests/unit -k "retrieval or routing" -q` instead. Nothing outside `tests/unit/routing/test_service_routing.py` greps for `routed:`, so no other suite should need an edit — if one fails, read it before changing it.

- [ ] **Step 8: Commit**

```bash
ruff check . --fix && ruff format .
git add src/internal/retrieval/service.py tests/unit/routing/test_service_routing.py
git commit -m "fix(routing): let an unbacked target annotate the search, not blank it"
```

---

### Task 2: Correct the documented contract

**Files:**
- Modify: `docs/retrieval.md:722`

**Interfaces:**
- Consumes: the mode spelling produced by Task 1 (`hybrid+routed:sql`).
- Produces: nothing other tasks depend on.

- [ ] **Step 1: Rewrite the short-circuit sentence**

`docs/retrieval.md` documents the old behavior as the contract. Find this sentence in the paragraph below the six-constructor table:

```
The three net-new constructors **build and validate but never execute** a query — there is no live SQL/KG/API backend, so `RetrievalService` short-circuits the `sql`/`graph`/`api` targets to `([], "routed:<target>")`. When a real backend is wired later, only the executor changes.
```

Replace it with:

```
The three net-new constructors **build and validate but never execute** a query — there is no live SQL/KG/API backend, so `RetrievalService` runs its ordinary hybrid retrieval for the `sql`/`graph`/`api` targets and records the decision as a mode suffix (`hybrid+routed:sql`). An unbacked route annotates a search rather than emptying it. When a real backend is wired later, only the executor changes.
```

- [ ] **Step 2: Verify no other doc states the old contract**

Run: `grep -rn "routed:" --include='*.md' . | grep -v node_modules | grep -v superpowers/archive`

Expected: only the line just edited in `docs/retrieval.md`. Archived plans under `docs/superpowers/archive/` are history and are left alone. `docs/configuration.md:189` describes the flag but not the short-circuit, so it needs no edit — confirm that by reading it rather than assuming.

- [ ] **Step 3: Commit**

```bash
git add docs/retrieval.md
git commit -m "docs(retrieval): an unbacked route annotates a search, not empties it"
```

---

### Task 3: Verify the whole suite and open the PR

**Files:** none modified.

**Interfaces:**
- Consumes: the commits from Tasks 1 and 2.

- [ ] **Step 1: Run the full default suite**

Run: `pytest -q`

Expected: PASS. `tests/conftest.py` already skips the slow `SEARCH_AGENT_MODEL` load, so the default run is fast. Note the test count — it should be the previous count plus 3: one test removed, four added (three parametrized cases plus the cached one).

- [ ] **Step 2: Confirm the flag is now usable end to end**

Run:

```bash
python3 - <<'PY'
from src.internal.retrieval.backends.base import RetrievalResult
from src.internal.retrieval.service import RetrievalService
from src.internal.routing.registry import DEFAULT_ROUTES, RouteRegistry
from src.internal.routing.router import Router


class B:
    def search_sparse(self, query, top_k, filters=None):
        return [RetrievalResult(doc_id="d1", title="t", text="x", url=None, score=1.0)]

    def search_dense(self, query, top_k, filters=None):
        raise NotImplementedError


svc = RetrievalService(B(), router=Router(RouteRegistry(DEFAULT_ROUTES)))
for q in ["top retrieval papers", "latest FAISS release", "papers related to BM25",
          "how many documents", "what is FAISS"]:
    r, m = svc.search(q, top_k=3)
    print(f"{q!r:28} -> {len(r)} results, mode={m}")
PY
```

Expected: every query returns 1 result. The first three modes end in `+routed:sql`, `+routed:api`, `+routed:graph`; `how many documents` ends in `+routed:sql`; `what is FAISS` is plain `sparse_only` with no `routed:` marker. Zero blank results is the whole point of the change — if any query returns 0, stop.

- [ ] **Step 3: Push and open the PR**

```bash
git push -u origin fix/routing-unbacked-target-fallthrough
gh pr create --title "fix(routing): an unbacked target annotates the search instead of blanking it" --body "$(cat <<'BODY'
## Summary

`ROUTING_ENABLED` could not be turned on. `RetrievalService.search` early-returned `([], "routed:<target>")` whenever the router picked SQL, GRAPH, or API — targets with no execution backend — and the router's cue lists are broad enough (`"top "`, `"most "`, `"per "`, `"latest "`, `"current "`) that this blanked a large slice of ordinary searches.

The unbacked branch now annotates instead of exiting: retrieval runs normally and the decision is recorded as a mode suffix, e.g. `hybrid+routed:sql`, matching the existing `+rag_fusion` / `+reranked` convention.

Measured against the real router before and after:

| query | before | after |
|---|---|---|
| `top retrieval papers` | 0 results, `routed:sql` | 1 result, `sparse_only+routed:sql` |
| `latest FAISS release` | 0 results, `routed:api` | 1 result, `sparse_only+routed:api` |
| `papers related to BM25` | 0 results, `routed:graph` | 1 result, `sparse_only+routed:graph` |
| `what is FAISS` | 1 result, `sparse_only` | unchanged |

## Notes

`test_routing_to_sql_short_circuits_to_empty` asserted `results == []` — the defect written down as a contract — so it was replaced rather than kept. The two pass-through tests had their assertion tightened from `not mode.startswith("routed:")` to `"routed:" not in mode`; under a suffix spelling `startswith` could no longer fail.

The router's cue lists are untouched, so which queries route where is exactly what it was. `docs/retrieval.md` documented the short-circuit as the contract and was corrected.

Out of scope, and each worth its own decision: `routing/construction/` (340 LOC, imported only by tests), and backing the API target with the `DOMAIN_REGISTRY` records capabilities.

Spec: `docs/superpowers/specs/2026-09-20-routing-unbacked-target-fallthrough-design.md`
Plan: `docs/superpowers/plans/2026-09-20-routing-unbacked-target-fallthrough.md`

🤖 Generated with [Claude Code](https://claude.com/claude-code)
BODY
)"
```

- [ ] **Step 4: Confirm the push landed every commit**

Run: `git log --oneline origin/fix/routing-unbacked-target-fallthrough -5`

Expected: the spec commit, the Task 1 commit, and the Task 2 commit are all present. A squash-merge drops commits pushed after the merge starts, so verify before considering the work done.
