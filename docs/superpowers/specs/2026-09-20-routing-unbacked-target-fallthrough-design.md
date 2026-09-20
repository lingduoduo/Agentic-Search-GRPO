# Routing: unbacked targets fall through

## Goal

Stop `ROUTING_ENABLED` from silently returning zero results. A routing
decision the service cannot execute should annotate the search, not replace it.

## The problem

`RetrievalService.search` is the only production consumer of
`src/internal/routing/`. When the router picks SQL, GRAPH, or API it returns
early:

```python
# No execution backend for these targets — construct-only.
# Degrade to empty results so routing never breaks a request.
return [], f"routed:{decision.retriever.value}"
```

The comment argues against itself. Returning nothing *is* breaking the request:
the caller asked for documents and got an empty list, with no error to
distinguish "routed away" from "nothing matched".

The three targets are unbacked in fact, not just in principle. The source names
`DEFAULT_ROUTES` assigns them — `analytics_db`, `knowledge_graph`,
`external_api` — appear nowhere else in `src/`. They are labels for backends
that do not exist.

What makes this a trap rather than a dormant branch is the breadth of the
heuristic cues. `_SQL_CUES` includes `"top "`, `"most "`, `"per "`; `_API_CUES`
includes `"latest "` and `"current "`. Run against the real router:

```
'top retrieval papers'    -> sql     (empty)
'latest FAISS release'    -> api     (empty)
'papers related to BM25'  -> graph   (empty)
'how many documents'      -> sql     (empty)
'what is FAISS'           -> hybrid  (served)
```

So setting one environment variable turns a large slice of ordinary searches
into blank results. The flag is default-off, so nothing is broken today — but
it cannot be turned on, which makes the routing layer unreachable in practice.

## Architecture

The unbacked branch stops being an exit and becomes an annotation.

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
        # No execution backend for these targets. Run ordinary retrieval
        # anyway — an unbacked route must not cost the caller its results —
        # and record the decision in the mode.
        routed = f"+routed:{decision.retriever.value}"
```

`search` then carries `routed` to each of its two real exits: the result-cache
hit (`"cached" + routed`) and the final fused return (`mode + routed`). Both
matter; the cache exit is the one easily missed, and a cached hit under an
unbacked route must report the routing decision the same way an uncached one
does.

The `+suffix` spelling is the convention `search` already uses for
`+rag_fusion` and `+reranked`, so `hybrid+routed:sql` reads like the rest of
the mode strings rather than like a new dialect.

The target tuple stays inline rather than being hoisted to a module constant.
`RetrieverTarget` is imported inside the function on purpose, to keep
`routing/` a lazy dependency of the retrieval service; a module-level constant
would force that import at import time to save naming three enum members.

## Testing

Test-driven. `tests/unit/routing/test_service_routing.py` is both the
regression net and the place the bug is currently pinned: the existing
`test_routing_to_sql_short_circuits_to_empty` asserts `results == []`, which is
the defect written down as a contract. It is rewritten, not preserved.

Replacing it: one parametrized test across all three unbacked targets, each
asserting real results come back *and* that the mode ends with
`+routed:<target>`; and one test that a result-cache hit under an unbacked
route reports `cached+routed:sql`, covering the second exit.

The two existing pass-through tests stay, with their assertion tightened from
`not mode.startswith("routed:")` to `"routed:" not in mode`. Under the new
spelling the marker is a suffix, so `startswith` would pass vacuously — it
would no longer be able to fail.

Each new assertion is mutation-checked: revert the fall-through, confirm the
test goes red, restore.

## Limits

This is a behavior fix to one branch. The router's cue lists are unchanged, so
which queries route to SQL, GRAPH, or API is exactly what it was; only the
consequence changes. No backend is added for the unbacked targets, and a query
routed to them still gets hybrid retrieval rather than the retrieval its route
names.

`docs/retrieval.md` documents the short-circuit as the contract and changes
with the code.

Out of scope: `routing/construction/` and its six constructors (unreachable
from `src/`, worth a decision of its own), backing the API target with the
`DOMAIN_REGISTRY` records capabilities, the heuristic cue lists, and
`routing/registry.py`.
