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
`src/internal/tools/search_domains.py`, which exports `AVAILABLE_DOMAINS`.

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
query handling, not improvements in real-world search relevance.

For non-general multi-query calls, tool metadata includes the original sanitized
`queries`, canonical `domain`, and `executed_queries`. Single-query tool output
keeps its existing formatted response. See [MCP](mcp.md#web-search-domain-hints)
for MCP response metadata.

This release supports function-calling tools and MCP public-web search. The HTTP
search endpoints, web UI, MCP indexed-document search, and vector/web
search-agent component do not expose a domain selector.
