# Search engine

[← Back to README](../README.md)

This guide covers the search agent: what it can do and how the web API routes a
request into it. For the authoritative deep dives, see
[API request routing](request-routing.md) and [Retrieval](retrieval.md).

## Capabilities

- **Agentic RAG** — multi-turn search with query enhancement, citations, and
  grounded synthesis.
- **Dense, sparse, and hybrid retrieval** — RRF fusion, reranking, and query
  optimization workflows over the local corpus and indexes.
- **Web search** — Google Custom Search, SerpAPI, and browser automation as
  fall-through sources when internal retrieval is insufficient.

## Request routing

With `mode` omitted, `/api/agent` classifies each request as `chat`, `search`, or
`tool`. An unfiltered auto-routed search tries internal retrieval first; weak or
empty evidence falls through to SerpAPI and then the configured browser-search
service. If no source returns evidence, the API reports that directly instead of
asking a local model to answer from memory. See
[API request routing](request-routing.md) for modes, provider precedence,
access-filter behavior, metadata, and examples.

## Dedicated search surface (`/search/*`)

Beyond the unified `/api/agent`, search has its own retrieval-only surface,
parallel to `/chat/*` and `/tool/*`:

- `POST /search/send-search-message` — runs the (optionally expanded) retrieval
  pipeline and returns the ranked documents with their executed queries. No LLM
  synthesis. Returns JSON, or a newline-delimited JSON stream when `stream:true`.
- `POST /search/search-flow-classification` — keyword-vs-chat routing hint.
- `GET /search/search-history` — past sessions for the caller.

In the web UI, the **Search** tab drives `send-search-message` and renders the
returned documents directly (no answer panel).

Searchable documents are prepared before query time by the existing asynchronous
ingestion and indexing jobs. Filter-aware and degraded search paths use the
shared composed pipeline: bounded session history resolves follow-ups for
retrieval, then candidates are ranked/reranked and used for evidence-grounded
inference. Strong unfiltered auto-search remains a distinct direct-first path: it
queries the original request, applies its direct ranking and sufficiency gate,
and falls through to SerpAPI and browser search when needed. Every path persists
finalized answers, citations, documents, and stage metadata through the same
JSON/SSE response tail. This internal simplification introduces no new public API
and does not change the request or response schemas.


## Search topic domains

The function-calling `search` and `web_search` tools accept an optional `domain`.
MCP's `search_web` accepts it too. The shared taxonomy lives in
`src/internal/tools/search.py`, which exports `AVAILABLE_DOMAINS`.

Single-query tool arguments:

```json
{"query": "battery recycling", "domain": "academic"}
```

Multi-query tool arguments:

```json
{"queries": ["battery recycling", "lithium recovery"], "domain": "academic"}
```

The selected domain applies to every query in that call. To search different
categories, issue separate calls. Omission or `general` preserves existing query
behavior. Domain names accept case, surrounding whitespace, spaces, and hyphens:
`Social Media` and `social-media` normalize to `social_media`. Empty names, unknown
names, and explicit null values are rejected before searching.

For a non-general domain, the tool appends the fixed query hint below exactly
once. For example, `battery recycling` becomes
`battery recycling academic research`. Existing provider selection, fallback
order, and authorization filters continue to apply.

| ID | Meaning and boundaries | Query hint |
| --- | --- | --- |
| general | Broad or mixed-topic search; default | empty string |
| resource | Datasets, reference materials, directories, and reusable tools | resources |
| social_media | Public social posts, communities, and discussions | social media |
| finance | Markets, investments, banking, and financial analysis | finance |
| academic | Scholarly literature, research methods, and publications | academic research |
| legal | Law, regulation, case law, and legal procedure | law |
| health | Medicine, public health, and clinical information | health |
| business | Companies, operations, strategy, and commerce | business |
| security | Cybersecurity, vulnerabilities, and defensive practices | cybersecurity |
| ip | Intellectual property: patents, trademarks, copyright, and licensing | intellectual property |
| code | Source code, programming, APIs, and developer documentation | programming |
| energy | Generation, fuels, storage, and energy systems | energy |
| environment | Climate, ecosystems, conservation, and pollution | environment |
| agriculture | Farming, crops, livestock, and agricultural systems | agriculture |
| travel | Destinations, transport, lodging, and trip planning | travel |
| film | Cinema, films, filmmaking, and the film industry | film |
| gaming | Video games, game development, and gaming communities | video games |

These categories overlap. `resource` describes the purpose of material and
`social_media` describes its source; most other categories describe subjects.
Finance/business, legal/ip, and health/academic can all overlap. Choose the
category that best expresses the question, or use `general` for mixed topics.
`ip` means intellectual property, not Internet Protocol; networking questions
usually fit `security` or `code`.

Domains are query hints, not guaranteed category filters. They do not promise
source authority, freshness, or access to private social content. Added words
can reduce recall; use `general` to search without the hint. Unit tests verify
query handling, not improvements in real-world search relevance; the measured
relevance result is below and found no detectable improvement.

For non-general multi-query calls, tool metadata includes the original sanitized
`queries`, canonical `domain`, and `executed_queries`. Single-query tool output
keeps its existing formatted response. See [MCP](mcp.md#web-search-domain-hints)
for MCP response metadata.

### Does selecting a domain help? Measured, and the answer is: not detectably

The taxonomy shipped with a standing condition — no quality claim without
comparing general and domain runs on labelled queries across every category.
That comparison has now been run, and it did not find an effect.

Run it yourself:

```bash
python -m examples.run_domain_relevance_eval --dry-run   # cost, no calls
python -m examples.run_domain_relevance_eval
```

The labelled set is `data/eval/domain_relevance_queries.json`: 16 topic domains,
three queries each, every query carrying the URL hosts an on-topic result should
come from. `general` is excluded because its hint is empty. Each query runs
twice against SerpAPI — once raw, once hinted — and each arm is scored by
**authority precision**, the share of its top 10 results served by those hosts.
Every response is cached under `data/eval/cache/`, so a re-run costs no quota.

**Result, 48 queries, 37 usable after exclusions:**

| Measure | Value |
|---|---|
| Mean authority-precision delta (domain − general) | **+0.016** |
| Paired permutation p (two-sided, n=37) | **0.33** |
| Cliff's delta | 0.081 (negligible) |
| Empty-result rate, general → domain | 0.125 → 0.146 |
| Mean result overlap (Jaccard) between arms | 0.13 – 0.64 by domain |

The pooled effect is not distinguishable from zero. Per-domain means run from
+0.17 (`security`) to −0.06 (`gaming`), and eight of sixteen are negative —
consistent with noise rather than a signal.

The overlap figures make the null more interesting than a flat "nothing
happened". The hint *substantially changes which results come back* — arms
share as little as 13% of their URLs — but the results it swaps in are not more
likely to come from topic-authoritative sources. The feature does something; it
just does not do the thing it was hoped to do. The domain arm also came back
empty slightly more often, which is the cost side of the same coin.

**How to read this.** Authority precision measures source alignment, not answer
quality: an `arxiv.org` paper is not automatically a better answer than a good
blog post, and this evaluation cannot tell them apart. The unit of analysis is
the query, not the result, so n is 37 rather than 370. Per-domain figures are
descriptive only — three queries cannot support a per-domain claim, and sixteen
tests at that size under multiplicity correction could not reject anything, so
no per-domain p-values are computed. A query whose arm errored or returned
nothing is excluded, because a delta against a failed call measures the failure.

Most importantly, at this sample size a null means **no effect was detected**,
not that no effect exists. The honest summary is that domain hints are not shown
to improve source alignment, and nothing in this repository should claim they
do. A `general` negative control runs alongside the real queries and must show a
zero delta and an overlap of 1.0; if it does not, the harness is mispairing arms
and the run is rejected rather than reported.

### Domains over HTTP and in the web UI

`POST /api/agent` accepts an optional `domain` alongside `query`. It defaults to
`general`, which leaves the query unchanged and keeps behavior identical to
requests that omit the field.

```json
{"query": "etf fees", "domain": "finance"}
```

`GET /api/search-domains` returns each identifier and its description in
registry order. The Assist page uses it to populate the selector next to the
question box, so the 17 identifiers are not copied into the frontend bundle.
The selector is always visible, unlike the Source and retrieval-URL fields,
which appear only under `?dev=1`.

Three modes honor the field, because in each one a single caller-supplied query
reaches a provider: the default auto mode, `search_tool`, and `hybrid_search`.
`chat_once` performs no retrieval, and `chat_loop`, `search_agent`, and
`tool_agent` build their own per-round queries internally, so a non-`general`
domain combined with an explicit one of those modes returns 400 rather than
being dropped without notice. An unrecognized domain also returns 400, before
any provider is called.

The hint reaches web providers only. It is applied inside each helper's
per-provider dispatch, so the local corpus always receives the raw query:
appending a topic word to a corpus query upweights documents that literally
contain that word instead of focusing the search. For the same reason the
answer prompt and the stored transcript keep the user's original question —
only the provider call and its cache key carry the hint. Selecting two
different domains for one query therefore produces two cache entries.

MCP indexed-document search and the vector/web search-agent component still do
not expose a domain selector. No relevance improvement is claimed: the
evaluation above ran the comparison and did not detect one.

The corpus-backed `/search/send-search-message` endpoint deliberately has no
domain field. It queries the local index only, where topic hints do not apply.


## Native domain search features

`DomainSearch` in `src/internal/tools/search.py` combines the existing
web-search flow, public-data tools, and page fetcher. It adds no service
integration, credentials, or CLI. The taxonomy remains in `search.py`.

### One registry, derived tags

A domain and its capabilities are a single declaration. `DOMAIN_REGISTRY` maps
each of the 17 identifiers to a `SearchDomain` carrying its description, its
query hint, and its capabilities:

```python
"finance": SearchDomain(
    "Markets, investments, banking, and financial analysis",
    "finance",
    (
        WEB_CAPABILITY,
        Capability("quote", "symbol", "records", "get_stock_quote"),
        Capability("crypto", "symbol", "records", "get_crypto_price"),
        Capability("currency", "from_currency", "records", "convert_currency"),
    ),
),
```

A tag such as `finance.quote` is **derived** — `f"{domain}.{capability.name}"`
— not a key anyone writes. Previously the domain and its routes were two
structures joined only by a string convention, so `"fnance.quote"` was an
accepted key belonging to no domain, and reading the relationship back out took
`tag.startswith(domain + ".")` and `tag.split(".", 1)[0]`. Neither is possible
now: `iter_capabilities()` yields each `(tag, capability)` pair from the
structure itself.

Every domain declares a `web` capability whose `tool_name` is `None`, meaning
the built-in web cascade rather than a seeded tool. It used to be fabricated
inside `get_sub_domains`; declaring it lets one loop emit both entry kinds.

Each capability states what it `returns`: `documents` for Wikipedia, arXiv,
Wayback, and web search, or `records` for stock quotes, crypto prices, currency
conversion, weather, geocoding, and nearby places. `get_sub_domains` reports it
so a caller can tell titled text from structured fields before choosing. The
field deliberately does not reuse the routing layer's `RetrieverTarget`: every
routed tool here is a remote API, so that enum would be constant, and its
values name local retrieval mechanisms rather than result shape.

The routing layer's own route list in `src/internal/routing/registry.py` stays
separate. Its entries name modalities — docs, structured, graph, live — and a
topic is orthogonal to a modality: a finance question may be an article, a
table, or a live quote.

### One name per concept

`search` accepts `query`, `domain`, `tag`, `params`, and `max_results`. The
aliases `sub_domain` and `sub_domain_params`, along with the equality guards
that reconciled the two spellings, were removed. The dotted tag format is
unchanged, so a client that discovers a tag with `get_sub_domains` and passes it
back is unaffected; only a caller using an alias spelling must rename it.

The tool registry and MCP expose four operations. Three of them —
`get_sub_domains`, `search_domain`, and `batch_search` — are registered but
deliberately **not offered to the agent loop**: every tag below routes to a
public-data tool the agent already holds directly, so putting both on the menu
gives a small model two paths to the same nine tools and a third way to run the
web search it already has in `web_search`. `NOT_AGENT_CALLABLE` in
`knowledge_base.py` withholds them, the same remedy applied to the
`search`/`search_routing_tool` pair, where a system prompt alone was tested and
did not fix selection while the duplicates were present. They stay fully
reachable through `/admin/tools` (always mounted) and MCP's own wrappers, and
through `/api/debug/tools` when `AGENTIC_SEARCH_DEBUG_PANELS` is set.
`extract_page` is agent-callable, because nothing else seeded there fetches a
URL.

| Operation | Behavior |
| --- | --- |
| `get_sub_domains(domains)` | Describe available routes and their real tool schemas for 1–5 domains; no network calls |
| `search_domain(query, domain, tag, params)` | Search the web or invoke an implemented capability |
| `extract_page(url, max_length)` | Reuse the existing HTTP(S) fetcher and HTML-to-text extraction |
| `batch_search(queries, ...)` | Run 1–5 searches concurrently, preserving input order and per-query errors |

All 17 domains offer `<domain>.web`, which appends the existing topic hint and
uses the configured web-search flow. These are broad searches, not guaranteed
category filters. Specialized routes use these existing tools:

| Tag | Existing tool | Supply `query` as |
| --- | --- | --- |
| `general.wikipedia` | `search_wikipedia` | Search text |
| `academic.arxiv` | `search_arxiv` | Paper search text |
| `resource.wayback` | `search_wayback` | Archived page URL |
| `finance.quote` | `get_stock_quote` | Ticker symbol |
| `finance.crypto` | `get_crypto_price` | Cryptocurrency symbol |
| `finance.currency` | `convert_currency` | Source currency; put amount and target currency in params |
| `environment.weather` | `get_weather` | Place name |
| `travel.location` | `search_location` | Place/address query |
| `travel.nearby` | `search_nearby_places` | Place type; put latitude and longitude in params |

Discovery returns each route's `tool_name`, `query_parameter`, `query_format`,
complete `parameters` schema, and additional `params`. It lists only available
read-only public-data implementations, plus web routes. It is a local capability
catalog, not a check of upstream service availability. Private corpus search and
arbitrary registered tools are outside this routing table.

### Examples

Discover first:

```json
{"domains": ["finance", "academic"]}
```

Search with the resulting tag:

```json
{"query": "AAPL", "tag": "finance.quote"}
```

```json
{"query": "retrieval augmented generation", "tag": "academic.arxiv", "params": {"limit": 3}}
```

For an untagged call, `domain` selects its web route. With a tag, the domain is
inferred from the tag or validated against an explicitly supplied domain.
Specialized routes receive the original query without appended topic words.
`sub_domain` and `sub_domain_params` remain aliases for `tag` and `params`.
Unknown tags, conflicting aliases, mismatched prefixes, and unsupported or
missing capability parameters fail before dispatch. Provider-specific options
must be present in the discovered schema; unsupported region/language options
are not silently accepted. Set a supported language through capability params.

`max_results` is capped at 10; a capability's `limit` is bounded by that cap.
Extraction defaults to 5,000 characters with a configurable maximum of 50,000,
and reports fetch failures explicitly. Its text format follows the existing
fetcher; it does not introduce additional file-format support.

Batch requests accept the same search fields per item and shared defaults:

```json
{
  "domain": "academic",
  "queries": [
    {"query": "dense retrieval"},
    {"query": "AAPL", "tag": "finance.quote"}
  ]
}
```

A per-item tag replaces the shared route. A per-item domain without a tag selects
that domain's web route. Per-item parameter objects replace shared parameter
objects. Inputs are not mutated, errors stay with their query, and cancellation
propagates to all outstanding workers.

### Python and result contracts

```python
from src.internal.tools import DomainSearch

async def lookup():
    search = DomainSearch()
    directory = search.get_sub_domains(["finance"])
    result = await search.search("AAPL", tag="finance.quote")
    return directory, result
```

The service returns query/domain/tag metadata and `results`; web searches also
include `executed_query`. Native FunctionTools use the repository's JSON result
conventions: search returns a document array or a facts object, extraction a
document array, discovery a directory object, and batch a `queries` object.
Only document arrays from search and extraction produce citation cards.

The seeded registry reuses the same public-data tool instances and web cascade
as the existing tools. MCP web routes use `MCP_WEB_SEARCH_PROVIDER`; existing
`search_web`, `open_urls`, corpus-search ACLs, and fallback behavior remain intact.
