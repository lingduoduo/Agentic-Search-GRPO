# Multi-category search

Sub-project 2 of 3 toward search-source agents. Sub-project 1 (category source
sites, #681–#687) made each category reach its own sites and measured that
better on held-out labels (+0.094 authority precision, p = 0.005). Sub-project 3
(automatic category selection) follows this one.

## Intent

A query may use up to three search-source categories. Each category searches in
parallel, and the results are fused with reciprocal rank fusion (RRF), then
reranked and cited by the existing pipeline. This makes the README definition
true: "Queries may use multiple agents, with results fused, reranked, and
synthesized with citations."

Success for this sub-project is correctness across every entry point, with
offline tests. Multi-category relevance is **not measured** here, and the docs
say so. A labelled multi-category evaluation is a follow-up, run when there is
provider quota or on Serper's free tier.

## Inputs, validation and the cap

`normalize_search_domains(values) -> list[str]` lives in
`src/internal/tools/search.py`, beside `normalize_search_domain`. It:

- normalizes each value with `normalize_search_domain`, so legacy aliases apply;
- removes duplicates, keeping the first occurrence;
- raises `InvalidToolInput` for an empty list, an unknown value, or more than
  `MAX_SEARCH_DOMAINS = 3` categories **after** deduplication;
- allows `web`, which counts toward the cap and runs as a plain search with no
  hint and no sites.

All validation happens before any provider call.

| Entry point | Field | Rule |
| --- | --- | --- |
| `POST /api/agent` | new `domains: list[str] \| None` beside `domain` | If `domains` is given, `domain` must be omitted or `web`, otherwise 400. Normalization failures return 400. The response echoes the canonical `domains`. The existing mode rule applies: any non-`web` category needs `search_tool`, `hybrid_search`, or auto. |
| `web_search` tool (`MultiQueryWebSearchTool`) | new `domains` (array, `maxItems` 3) | Mutually exclusive with `domain`. Aliases pass schema validation, as for `domain` today. |
| `search_domain` / `batch_search` (`DomainSearch`) | new `domains` | Only for the untagged web route. A `tag` together with `domains` is rejected, because a capability belongs to one category. A per-item `domains` in a batch follows the same rule. |
| MCP `search_web` | new `domains` | Same rules. Mutually exclusive with `domain`. |
| Assist page | "Search source" becomes a multi-select of up to 3 | It sends `domain` when one category is chosen (unchanged) and `domains` when two or three are. |

Every existing single-`domain` request behaves exactly as today.

## `search_categories`

```python
async def search_categories(
    query: str, *, domains: list[str], page_size: int, search_fn: PageSearch,
    restricted_fn: PageSearch | None = None, time_budget: float | None = None,
) -> CategorySearch
```

- **One category:** returns `search_category(query, domain=domains[0], ...)`
  unchanged. There is no fusion step and no extra metadata.
- **Several categories:** runs one `search_category` per category in parallel
  with `asyncio.gather(..., return_exceptions=True)`. Each gets `page_size` and
  the same `time_budget`.

### Fusion

- Each category contributes its ranked list of usable pages: those with a URL,
  no error, and not blank.
- The lists are scored with `rrf_rank` from `src/internal/retrieval/fusion.py`
  (k = 60), keyed by URL, and truncated to `page_size`.
- Ties keep first-seen order across the lists in category order, so the
  first-named category wins ties. The result is deterministic.
- A page found by several categories accumulates their RRF scores. It keeps the
  `category`, `category_leg` and `category_query` of the first category that
  found it, and gains `categories`: every category that found it, in category
  order. In a multi-category search, a page found by one category gets
  `categories` with one entry. The one-category path adds no `categories` key.
- `executed_queries` concatenates each category's queries in category order.
  `fallback_used` is true if any category used its fallback.

### Failures

- A category whose `search_category` raises is dropped and logged; the others
  still count.
- If every category raises, the first category's exception propagates.
- If no category produced a usable page, the first non-raising category's
  outcome is returned unchanged, error pages included, so callers' existing
  error handling sees what a single-category search returns today.
- Cancellation of the caller cancels every category, which inherits
  `search_category`'s own leg cancellation.

## Where it applies

- **HTTP direct, `web_search` tool, `DomainSearch` web route, MCP `search_web`:**
  each site that calls `search_category` today calls `search_categories` with
  the request's categories. One category goes through the unchanged path.
- **HTTP hybrid, several categories:** categories apply to the **original query
  only**. One `search_categories` call per web provider takes that query; the
  expanded variants run as plain web searches with no category and no `site:`.
- **HTTP hybrid, one category:** unchanged, so the category still applies to
  every expanded variant.
- **Browser legs, several categories:** one call with the plain query, never one
  per category. With one category the browser gets its suffix query as today.
- **Corpus (`retrieval`) legs:** raw query, unchanged.
- After fusion the existing pipeline is unchanged: SerpAPI page fetch, reranker
  when `AGENTIC_SEARCH_RERANK_URL` is set, MMR, and citations.

## Cost

With the cap, per query: at most 6 provider calls (2 per site-listed category)
on direct, tools and MCP; at most 6 plus the plain expanded variants (about 3)
on hybrid. `web` and categories without sites cost 1 call each.

## Assist page

- **Selection:** a multi-select of up to 3 categories. When 3 are selected, the
  remaining options are disabled, with a short hint that the limit is 3.
- **Source cards:** they show a document's `categories`, e.g.
  "financial · news".
- **Discovery failure:** the selector still falls back to `web`.

## Testing

All tests are offline with stub providers. Every new test is mutation-checked:
remove the feature line, see the test fail, restore it.

- **`normalize_search_domains`:** aliases; deduplication before the cap (4
  distinct categories rejected; `["finance", "financial", "news", "code"]`,
  which dedupes to 3, accepted); empty and unknown values rejected; `web`
  allowed.
- **`search_categories`:**
  - one category returns exactly what `search_category` returns;
  - several categories start in parallel;
  - RRF order, including a shared page ranking higher and tie-breaks;
  - `categories` metadata;
  - one category raising, all raising, and no usable pages anywhere;
  - `executed_queries` and `fallback_used`;
  - cancellation.
- **Entry points:**
  - HTTP direct;
  - hybrid with several categories: the original query only, variants plain,
    the browser getting the plain query once;
  - hybrid with one category unchanged;
  - the `web_search` tool;
  - `search_domain` and `batch_search`, including `tag` plus `domains`
    rejected;
  - MCP `search_web`;
  - `domain` plus `domains` returning 400, the mode rule, and the response
    echoing `domains`.
- **Guard:** 4 categories never reach a provider, on any entry point.
- **Frontend:** the selection limit, the payload for one category versus
  several, the discovery fallback, and category chips on source cards.

## Documentation

- **`docs/search-engine.md`:** a "Multi-category search" section covering the
  fields, cap, fusion, failures, per-mode cost, the hybrid rule, and the
  statement that multi-category relevance is unmeasured.
- **`docs/api-reference.md`, `docs/mcp.md`, `docs/frontend.md`:** the new field
  or control.
- **README:** states that queries may use up to three categories, with results
  fused, reranked, and cited.

## Out of scope

- Automatic category selection (sub-project 3).
- A labelled multi-category relevance evaluation.
- A per-request call budget.
- Per-category weights in fusion.
- Any change to single-category behavior.
