# Retire routing/construction/

Closes #591.

## Goal

Remove a query-construction layer that no production path reaches, deciding its
three parts separately rather than condemning a directory wholesale.

## The problem

`src/internal/routing/construction/` is 340 LOC across seven files, with 225
LOC of tests. Nothing in `src/` or `examples/` imports any of it. The single
production consumer of `routing/` is `RetrievalService.search`, which reads a
`RouteDecision`'s `retriever` and never calls `construct()`.
`RouteDecision.construction_target` is assigned by `Router._decision` and read
by nobody outside tests.

The naive argument — no importer, therefore dead — is not good enough here.
This repo has factory-built, registry-dispatched, and default-off features that
all look unreachable to a grep, and #590 was a reminder that default-off is not
the same as inert. So the three parts are judged separately.

### The wrappers earn nothing

`vector.py` builds `{top_k, namespace, filters}` with `filters` permanently
empty. `metadata.py` calls the live `QueryConstructor.extract_filters` and
repackages the result. `hybrid.py` is three constructor arguments and one
function call. Each returns a `ConstructedQuery` that nothing consumes.
Deleting them removes no capability: the code `metadata.py` wraps lives in
`src/internal/retrieval/query_constructor.py` and stays reachable through
`src/context/query_transform.py`.

### The validators would have to be rewritten anyway

This is the part worth getting right, because the obvious argument for keeping
243 LOC of SQL, Cypher, and API validation is that it is security-relevant work
someone thought about, and that deleting it means rewriting it when a backend
arrives. Measured against ordinary inputs, that argument inverts:

```
validate_sql     SELECT created_at FROM orders            -> rejected  ("create" is a substring of "created_at")
                 SELECT name FROM users WHERE name = 'delete me'  -> rejected  (keyword inside a string literal)
validate_cypher  MATCH (p {name:"call center"}) RETURN p  -> rejected  (literal)
                 MATCH (p) WHERE p.name = "Set Phasers" RETURN p   -> rejected  (literal)
```

A column called `created_at` is not an edge case. These validators reject
routine legitimate queries, and `validate_sql`'s own docstring concedes the
class: keyword checks "may false-positive on those words inside string
literals", CTE-shadowed names are unresolved, and both are "acceptable — this
layer never executes SQL".

That is the honest reading. The validation is sound only because nothing uses
it. Wiring a real backend would require replacing substring matching with a
parser, not extending it. So the work is rewrite-either-way, and keeping the
code preserves a false impression that the hard part is done.

### One flag reports a capability that was never wired

`adaptive_mmr_lambda` maps query length to an MMR lambda. Its only importer is
`construction/hybrid.py`. Separately, `ADAPTIVE_MMR` exists as an environment
variable read exactly once, at `servers/retrieval/server.py:113`, purely to
report itself in `/api/admin/retrieval/stats`. Set it and the endpoint says
`"adaptive_mmr": "true"` while nothing changes; the live path calls
`mmr_rerank` with its default `mmr_lambda=0.5`.

That is the #590 shape again — an advertised capability with nothing behind it
— and it is the worst of the three states to leave alone, because it misinforms
rather than merely idles.

## Architecture

Three removals, sequenced so each leaves the tree green.

1. **The wrappers.** `vector.py`, `metadata.py`, `hybrid.py`, and their test
   file. `base.py` survives this step, because `sql.py`, `graph.py`, and
   `api.py` import `ConstructedQuery` from it.
2. **The validators and the package.** `sql.py`, `graph.py`, `api.py`,
   `base.py`, `__init__.py`, and three test files — the directory goes. Then
   `RouteDecision.construction_target` and its assignment in
   `Router._decision`, with `test_route_decision_defaults` dropping the
   keyword.
3. **Adaptive MMR.** `adaptive_mmr_lambda`, its line in the `fusion_learner`
   module docstring, its four tests, and the `"adaptive_mmr"` key in the admin
   stats payload.

`FusionLearner` and `FusionWeights` stay: `servers/retrieval/optimize_router.py`
imports them, so the module remains.

The unused `RetrieverTarget` members `SPARSE`, `DENSE`, and `METADATA` stay
too. A config-driven registry loaded from `ROUTING_REGISTRY_PATH` can name any
of them, so they are not unreachable in the way the constructors are.

## Testing

Deleting code is not verified by a green suite — a suite stays green when the
deleted thing was never exercised. Two checks beyond `pytest`:

- The import-reachability guard (`tests/unit/servers/test_server_modules_import.py`)
  must still pass, proving no server module imported what was removed.
- A grep sweep for every deleted symbol (`ConstructedQuery`, `QueryConstructor`
  as the construction protocol, each constructor class, `validate_sql`,
  `validate_cypher`, `TableSchema`, `ApiSpec`, `construction_target`,
  `adaptive_mmr_lambda`, `ADAPTIVE_MMR`) across `src/`, `examples/`, `tests/`,
  and `docs/`, with only intended residue remaining.

`src/internal/retrieval/query_constructor.QueryConstructor` shares a name with
the construction protocol and is live. The sweep must not confuse them.

## Limits

Roughly 600 LOC removed and no production behavior changed, with one deliberate
exception: `/api/admin/retrieval/stats` stops reporting `adaptive_mmr`, a key
whose value never described anything.

Adaptive MMR is removed rather than wired. Wiring it would be a behavior change
in the live retrieval path and needs its own measurement — #588 is the standing
reminder that an unmeasured retrieval heuristic can turn out to be a null
result.

Out of scope: the `RetrieverTarget` enum members, `routing/registry.py`, the
router's cue lists, and issue #592.
