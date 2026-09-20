# Retire routing/construction/ Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove the unreachable query-construction layer, in three independently reviewable steps.

**Architecture:** Three deletions, sequenced so each leaves a green tree — wrappers first (they depend on nothing), then the validators together with `base.py` and the package itself, then adaptive MMR and the flag that advertised it.

**Tech Stack:** Python 3, pytest, `src/internal/routing/`, `src/internal/retrieval/fusion_learner.py`.

**Spec:** `docs/superpowers/specs/2026-09-20-retire-routing-construction-design.md`

Closes #591.

## Global Constraints

- **Order is load-bearing.** `base.py` holds `ConstructedQuery`, imported by `sql.py`, `graph.py`, and `api.py`. It cannot go in Task 1.
- **Do not touch** the `RetrieverTarget` enum members, `routing/registry.py`, `routing/router.py`'s cue lists, or `FusionLearner` / `FusionWeights` (live via `servers/retrieval/optimize_router.py`).
- `src/internal/retrieval/query_constructor.QueryConstructor` is a **different, live** class that shares a name with the construction protocol. Never delete or grep-match it away.
- A green suite does not verify a deletion. Every task ends with a grep sweep for the symbols it removed.
- Run `ruff check . --fix && ruff format .` before each commit; the pre-commit `ruff-format` hook aborts otherwise.

---

### Task 1: Remove the wrapper constructors

**Files:**
- Delete: `src/internal/routing/construction/vector.py`, `metadata.py`, `hybrid.py`
- Delete: `tests/unit/routing/test_construction_existing.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: a `construction/` package containing only `__init__.py`, `base.py`, `sql.py`, `graph.py`, `api.py`. Task 2 removes the rest.

- [ ] **Step 1: Confirm nothing outside tests imports them**

```bash
grep -rn --include='*.py' -E "VectorSearchQueryConstructor|MetadataFilterConstructor|HybridRetrievalQueryConstructor" src examples tests
```

Expected: hits only in `src/internal/routing/construction/{vector,metadata,hybrid}.py` and `tests/unit/routing/test_construction_existing.py`. Anything else — stop; the premise is wrong.

- [ ] **Step 2: Delete the four files**

```bash
git rm src/internal/routing/construction/vector.py \
       src/internal/routing/construction/metadata.py \
       src/internal/routing/construction/hybrid.py \
       tests/unit/routing/test_construction_existing.py
```

- [ ] **Step 3: Verify the remaining package still imports**

```bash
python3 -c "
from src.internal.routing.construction.sql import SqlQueryConstructor
from src.internal.routing.construction.graph import validate_cypher
from src.internal.routing.construction.api import ApiRequestConstructor
print('remaining constructors import cleanly')
"
```

Expected: the message prints. `base.py` is still there, which is why this works.

- [ ] **Step 4: Run the routing suite**

Run: `python3 -m pytest tests/unit/routing/ -q`

Expected: PASS, with 4 fewer tests than before (`test_construction_existing.py` held 4).

- [ ] **Step 5: Commit**

```bash
ruff check . --fix && ruff format .
git add -A
git commit -m "refactor(routing): drop the wrapper query constructors"
```

---

### Task 2: Remove the validators, base.py, and construction_target

**Files:**
- Delete: `src/internal/routing/construction/` (the whole directory: `sql.py`, `graph.py`, `api.py`, `base.py`, `__init__.py`)
- Delete: `tests/unit/routing/test_construction_sql.py`, `test_construction_graph.py`, `test_construction_api.py`
- Modify: `src/internal/routing/route.py` (drop `construction_target`)
- Modify: `src/internal/routing/router.py` (drop its assignment)
- Modify: `tests/unit/routing/test_route.py` (drop the keyword)

**Interfaces:**
- Consumes: the package state Task 1 left.
- Produces: `RouteDecision(domain, sources, retriever, confidence, strategy)` — no `construction_target`. Nothing downstream depends on this.

- [ ] **Step 1: Delete the package and its tests**

```bash
git rm -r src/internal/routing/construction
git rm tests/unit/routing/test_construction_sql.py \
       tests/unit/routing/test_construction_graph.py \
       tests/unit/routing/test_construction_api.py
rm -rf src/internal/routing/__pycache__
```

The `__pycache__` removal matters: stale bytecode for a deleted module can keep an import working and hide a missed reference.

- [ ] **Step 2: Drop `construction_target` from RouteDecision**

In `src/internal/routing/route.py`, delete this line from `RouteDecision`:

```python
    construction_target: RetrieverTarget = RetrieverTarget.HYBRID
```

- [ ] **Step 3: Drop its assignment in the router**

In `src/internal/routing/router.py`, inside `Router._decision`, delete:

```python
            construction_target=route.retriever,
```

leaving the other five keyword arguments intact.

- [ ] **Step 4: Update test_route.py**

In `tests/unit/routing/test_route.py::test_route_decision_defaults`, delete the
`construction_target=RetrieverTarget.HYBRID,` line so the call reads:

```python
def test_route_decision_defaults():
    d = RouteDecision(
        domain="docs",
        sources=["local"],
        retriever=RetrieverTarget.HYBRID,
    )
    assert d.confidence == 1.0
    assert d.strategy == "heuristic"
```

- [ ] **Step 5: Prove the package is gone and nothing reaches for it**

```bash
test -d src/internal/routing/construction && echo "STILL PRESENT — stop" || echo "package removed"
grep -rn --include='*.py' -E "routing\.construction|ConstructedQuery|construction_target|validate_sql|validate_cypher|TableSchema|ApiSpec|SqlQueryConstructor|GraphQueryConstructor|ApiRequestConstructor" src examples tests
```

Expected: `package removed`, and the grep returns **nothing**. Note it must not match `src/internal/retrieval/query_constructor.py` — that is a different, live class; the pattern above deliberately does not include the bare word `QueryConstructor`.

- [ ] **Step 6: Run the routing suite and the import guard**

```bash
python3 -m pytest tests/unit/routing/ tests/unit/servers/test_server_modules_import.py -q
```

Expected: PASS. The import guard is the one that would catch a server module that reached into the deleted package.

- [ ] **Step 7: Commit**

```bash
ruff check . --fix && ruff format .
git add -A
git commit -m "refactor(routing): remove the never-executed query constructors"
```

---

### Task 3: Remove adaptive MMR and the flag that advertised it

**Files:**
- Modify: `src/internal/retrieval/fusion_learner.py` (drop `adaptive_mmr_lambda` and its docstring line)
- Modify: `tests/unit/retrieval/test_fusion_learner.py` (drop the import and 4 tests)
- Modify: `src/internal/servers/retrieval/server.py:113` (drop the `adaptive_mmr` stats key)

**Interfaces:**
- Consumes: nothing from earlier tasks; independent of them.
- Produces: `/api/admin/retrieval/stats` without an `adaptive_mmr` key. `FusionLearner` and `FusionWeights` are unchanged and stay exported.

- [ ] **Step 1: Confirm the function has no remaining caller**

```bash
grep -rn --include='*.py' "adaptive_mmr_lambda" src examples tests
```

Expected: only its definition in `fusion_learner.py` and the four tests plus the import in `tests/unit/retrieval/test_fusion_learner.py`. Task 1 removed `hybrid.py`, its only production importer. If `src/` shows anything else, stop.

- [ ] **Step 2: Delete the function**

In `src/internal/retrieval/fusion_learner.py`, delete the whole
`def adaptive_mmr_lambda(query: str) -> float:` function (through its final
`return 0.3`), and delete this line from the module docstring:

```
adaptive_mmr_lambda: returns MMR lambda scaled to query length.
```

- [ ] **Step 3: Delete its four tests**

In `tests/unit/retrieval/test_fusion_learner.py`, remove `adaptive_mmr_lambda`
from the import block, leaving:

```python
from src.internal.retrieval.fusion_learner import (
    FusionLearner,
    FusionWeights,
)
```

then delete these four functions entirely: `test_adaptive_mmr_lambda_short_query`,
`test_adaptive_mmr_lambda_long_query`, `test_adaptive_mmr_lambda_medium_query`,
`test_adaptive_mmr_lambda_tiers`.

- [ ] **Step 4: Drop the self-reporting stats key**

In `src/internal/servers/retrieval/server.py`, inside `retrieval_stats`, delete:

```python
            "adaptive_mmr": os.environ.get("ADAPTIVE_MMR", "false"),
```

Leave `query_expansion_enabled` alone — verify it is genuinely read elsewhere
before assuming it shares the problem; it is not part of this change either way.

- [ ] **Step 5: Verify the sweep is clean**

```bash
grep -rn "adaptive_mmr\|ADAPTIVE_MMR" src examples tests docs --include='*.py' --include='*.md' | grep -v superpowers
```

Expected: nothing. The `superpowers` exclusion keeps archived plans and this
change's own spec out of it — those are history and are left as written.

- [ ] **Step 6: Confirm what stays actually stays**

```bash
python3 -c "
from src.internal.retrieval.fusion_learner import FusionLearner, FusionWeights
from src.internal.servers.retrieval import server
print('FusionLearner and the retrieval server still import')
"
python3 -m pytest tests/unit/retrieval/test_fusion_learner.py -q
```

Expected: the message prints and the remaining fusion-learner tests PASS.

- [ ] **Step 7: Commit**

```bash
ruff check . --fix && ruff format .
git add -A
git commit -m "refactor(retrieval): remove adaptive MMR and the flag that never switched it on"
```

---

### Task 4: Documentation, full verification, and the PR

**Files:**
- Modify: `docs/retrieval.md` (lines ~700-701 and ~714-722)

- [ ] **Step 1: Remove the construction layer from the flow diagram**

In `docs/retrieval.md`, the flow currently reads:

```
query → Router.route() → RouteDecision(domain, sources, retriever, construction_target)
      → QueryConstructor.construct() → ConstructedQuery(target, payload, text)
```

Replace both lines with:

```
query → Router.route() → RouteDecision(domain, sources, retriever)
```

- [ ] **Step 2: Remove the six-constructor table**

Delete the `**Six query constructors**` heading, the whole table beneath it, and
the paragraph that follows describing the three net-new constructors. In its
place put one sentence, so a reader learns the routing layer decides but does
not construct:

```
The router emits a decision only; there is no query-construction layer. Targets
with no execution backend (`sql`, `graph`, `api`) fall through to ordinary
hybrid retrieval and are recorded as a mode suffix — see #590.
```

- [ ] **Step 3: Check no doc still names a removed symbol**

```bash
grep -rn "ConstructedQuery\|construction_target\|construction/\|adaptive_mmr" docs --include='*.md' | grep -v superpowers
```

Expected: nothing.

- [ ] **Step 4: Run the full suite**

Run: `python3 -m pytest -q 2>&1 | tail -3`

Expected: PASS. The count drops by 15: 4 (`test_construction_existing`) + 7
(`test_construction_sql`) + ... — do not predict the exact figure, read it and
confirm the drop is entirely accounted for by the deleted test files. Count
them first with
`git show HEAD~3 --stat | grep test_construction` if unsure.

- [ ] **Step 5: Confirm the deletion is real, not just unexercised**

```bash
python3 -c "
import importlib
try:
    importlib.import_module('src.internal.routing.construction')
except ModuleNotFoundError as exc:
    print('gone as expected:', exc)
else:
    raise SystemExit('construction package still importable')
"
git diff --stat main...HEAD | tail -3
```

Expected: the ModuleNotFoundError message, and a diffstat showing roughly 600
deleted lines.

- [ ] **Step 6: Commit the docs, push, and open the PR**

```bash
ruff check . --fix && ruff format .
git add docs/retrieval.md
git commit -m "docs(retrieval): the router decides, it does not construct"
git push -u origin refactor/retire-routing-construction
gh pr create --title "refactor(routing): retire the query-construction layer nothing reached" --body "$(cat <<'BODY'
Closes #591.

## Summary

`src/internal/routing/construction/` was 340 LOC over seven files with 225 LOC of tests, reachable from nothing in `src/` or `examples/`. `RouteDecision.construction_target` was assigned by the router and read by nobody.

Judged in three parts rather than condemned as one directory, because "no importer" is not sufficient evidence in a repo full of factory-built and default-off features.

**The wrappers earn nothing.** `vector.py` built a dict whose `filters` key was permanently empty; `metadata.py` repackaged the live `QueryConstructor.extract_filters`; `hybrid.py` was three constants and one call. The code `metadata.py` wrapped stays where it lives and stays reachable.

**The validators would have to be rewritten anyway.** The case for keeping 243 LOC of SQL/Cypher/API validation is that it is security-relevant work that would otherwise be rewritten later. Measured, that inverts:

```
validate_sql     SELECT created_at FROM orders                    -> rejected  ("create" ⊂ "created_at")
                 SELECT name FROM users WHERE name = 'delete me'  -> rejected  (keyword in a literal)
validate_cypher  MATCH (p {name:"call center"}) RETURN p          -> rejected  (literal)
```

A column named `created_at` is not an edge case. `validate_sql`'s own docstring concedes the false-positive class and calls it "acceptable — this layer never executes SQL". The validation is sound only because nothing uses it; wiring a backend means replacing substring matching with a parser, not extending it.

**One flag advertised a capability that was never wired.** `ADAPTIVE_MMR` was read exactly once, at `servers/retrieval/server.py:113`, purely to report itself in `/api/admin/retrieval/stats`. Setting it changed nothing — the live path calls `mmr_rerank` with its default `mmr_lambda=0.5`. Same shape as #590, and the worst of the three to leave, because it misinforms rather than idles.

## Notes

Tests prove a function works, not that anything calls it. `adaptive_mmr_lambda` had **four passing tests** and zero reachable callers; `test_construction_sql.py` was 71 green lines for a validator that rejects `SELECT created_at FROM orders`.

Kept deliberately: `FusionLearner` / `FusionWeights` (live via `optimize_router.py`), and the `RetrieverTarget` members `SPARSE` / `DENSE` / `METADATA`, since a `ROUTING_REGISTRY_PATH` JSON can legitimately name them.

Adaptive MMR is removed rather than wired. Wiring it would change live retrieval behavior and needs its own measurement — #588 is the standing reminder that an unmeasured retrieval heuristic can be a null result.

A green suite does not verify a deletion, so this also ran the import-reachability guard and a grep sweep for every removed symbol, taking care not to confuse the removed construction protocol with the live `retrieval.query_constructor.QueryConstructor` that shares its name.

Behavior change, deliberate and singular: `/api/admin/retrieval/stats` no longer reports `adaptive_mmr`.

Spec: `docs/superpowers/specs/2026-09-20-retire-routing-construction-design.md`
Plan: `docs/superpowers/plans/2026-09-20-retire-routing-construction.md`

🤖 Generated with [Claude Code](https://claude.com/claude-code)
BODY
)"
```

- [ ] **Step 7: Confirm every commit reached the remote**

Run: `git log --oneline origin/refactor/retire-routing-construction -6`

Expected: the spec/plan commit and all four task commits.
