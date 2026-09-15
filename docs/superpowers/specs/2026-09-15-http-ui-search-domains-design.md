# HTTP and UI search domains

## Goal

Make the 17-domain search taxonomy selectable from the web experience. A caller
sends `domain` on `POST /api/agent`; the Assist page offers a selector; the
selected topic hint reaches web search providers and nothing else.

This completes the HTTP/UI increment deferred by
`2026-09-14-search-domains-design.md`, which shipped the taxonomy through
callable tools and MCP and stated that "later HTTP/UI integration should be
planned against the completed shared API". The shared API is complete:
`DOMAIN_REGISTRY`, `normalize_search_domain`, `prepare_domain_query`, and
`search_domain_parameter` in `src/internal/tools/search.py`.

## Why the corpus search page is not the target

The obvious reading of "add a domain selector to the search page" targets
`/search`. That surface is the wrong one, and the reason shapes the whole
design.

`SearchView` calls `POST /search/send-search-message`, which calls
`run_expanded_search` against `search_url` — the local retrieval server. It
searches `data/corpus.jsonl` and never reaches a web provider.

`prepare_domain_query` appends a topic word to the query string: `"ETF fees"`
becomes `"ETF fees finance"`. Against a web provider that is a hint. Against
local TF-IDF it is an additional scored term, upweighting corpus documents that
literally contain "finance". Corpus documents carry only `acl` and `source` in
metadata, so no filter-based route exists either, and the shipped corpus is
entirely information-retrieval material — most of the 17 domains select nothing.

`POST /api/agent` does reach real providers: `_run_direct_search` and
`_run_hybrid_search` dispatch to serpapi, google, serper, and the browser
server. That is where the hints were designed to work, so that is the surface
that gains the control.

## Architecture

One rule: **the topic hint is applied per-provider, at provider dispatch, and
never to the corpus.**

The three honored paths funnel into two leaf helpers, both of which already
iterate over providers. The hint is applied inside that iteration, guarded on
the provider being something other than `retrieval`.

Three properties follow from placing it there rather than at the route:

1. The prior spec requires hints be applied "exactly once, before provider
   dispatch and cache lookup". Provider dispatch is the loop body, and the
   process-local serving cache sits behind the provider calls, so the hinted
   string is both what the provider receives and what keys the cache. Domains
   get distinct cache entries instead of colliding on the raw query.
2. The corpus leg is untouched. `_run_direct_search(query,
   source_provider="retrieval", ...)` on the direct-first path, and the
   `source_provider == "retrieval"` branch of `_run_hybrid_search`, both run the
   raw query.
3. The answer prompt and the persisted transcript keep the user's real
   question. `query` is never rewritten at the route level, so no chat history
   row contains an appended topic word.

In `_run_hybrid_search` the hint is applied inside `_fetch_provider`, after
`_expanded_queries` has run. Hinting before expansion would feed the topic word
to the LLM expander and propagate it into every generated variant; hinting after
gives each dispatched query exactly one hint.

### The provider predicate

`_is_web_provider` must not be reused for this decision. `_WEB_PROVIDERS` is
`{"serpapi"}` and the function's documented meaning is "returns URL snippets
needing full-page fetch" — `google` and `serper` are absent by design. Using it
would silently skip hints for two providers that do want them. The domain rule
uses its own predicate: the provider is not `retrieval`.

## Honored modes

`/api/agent` exposes six modes that consume the query differently. `domain` is
honored where a single caller-supplied query reaches a provider:

| Mode | Query path | Honors `domain` |
|---|---|---|
| auto (`mode=None`) | `_auto_search_pipeline` | yes — the default |
| `search_tool` | `_run_direct_search` | yes |
| `hybrid_search` | `_run_hybrid_search` | yes |
| `chat_once` | no retrieval | no |
| `chat_loop`, `search_agent` | loop generates its own per-round queries | no |
| `tool_agent` | tools already take `domain` | already covered |

Agent loops are excluded deliberately. Their queries are constructed inside
`src/agents`, GRPO training depends on that construction, and "exactly once"
cannot be guaranteed across rounds without changing it.

## Components

- `AgentExperienceRequest` gains `domain: str = "general"`, matching the shape
  of the existing optional `source_provider` and `mode` fields.
- `_run_agent_impl` normalizes `domain` once, before dispatch.
- `_run_direct_search`, `_run_hybrid_search`, `_run_auto_routed`,
  `_auto_search_pipeline`, and `_WebHybridRetrievalStage` gain a `domain`
  keyword defaulting to `"general"`, threading the value to the two leaves.
- `GET /api/search-domains` returns each domain's name and description from
  `DOMAIN_REGISTRY`, in registry order.
- `SearchComposer` renders an always-visible selector; `AssistPage` holds the
  selection in session state and sends it on every submit.

The registry endpoint exists so the taxonomy has one source. A hand-copied
TypeScript array would be a second copy of a list the prior spec requires be
preserved exactly — "the exact 17 identifiers and their supplied order" — and
copies drift.

## Error handling

- An unrecognized `domain` returns 400 before any network call, reusing
  `normalize_search_domain`'s validation.
- An explicit non-honoring `mode` combined with a non-`general` domain returns
  400. The prior spec's rollout boundary exists to avoid "silently accepting
  ignored fields"; dropping the value quietly would reintroduce exactly that.
- `domain="general"` is accepted with every mode and leaves behavior
  byte-identical, because `prepare_domain_query` returns the query unchanged
  when the hint is empty.

## UI

The selector is a normal product control, always visible, defaulting to
`general` — unlike the Source and URL fields, which `SearchComposer` gates
behind `?dev=1` because they are development affordances. The selection persists
for the session so a user researching one topic does not re-pick it every turn.

## Testing

Test-driven. Backend, with mocked providers and no credentials:

1. A non-`general` domain reaches web providers as a hinted query.
2. The corpus leg receives the raw query — asserted positively, not by omission.
3. The hint is applied exactly once along the auto path.
4. `google` and `serper` receive hints, covering the `_is_web_provider` trap.
5. Two domains produce two cache entries rather than one shared entry.
6. An invalid domain returns 400 with no provider call.
7. A non-honoring explicit mode plus a non-`general` domain returns 400.
8. `general` leaves the dispatched query unchanged for every honored mode.
9. The registry endpoint lists all 17 identifiers in registry order.

Frontend: the selector renders options from the endpoint, defaults to `general`,
and the selection appears in the request body.

## Limits

No relevance claim is made or supported here. The prior spec requires comparing
general and domain runs on labeled queries across all 17 categories before
describing domains as improving quality; that evaluation remains outstanding and
this increment does not perform it. What ships is reachability: a caller can
select a topic and the hint arrives at the provider.

Out of scope: the `/search` corpus page, query construction inside `src/agents`,
index schema changes, and new providers.
