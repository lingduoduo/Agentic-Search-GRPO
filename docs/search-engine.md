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
- **Web search** — SerpAPI, then browser automation (when
  `AGENTIC_SEARCH_BROWSER_SEARCH_URL` is set), as fall-through sources when
  internal retrieval is insufficient. Google Custom Search is available only as
  an explicit `source_provider`.

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

Searchable documents are prepared before query time by the offline
`index_builder`. Filter-aware and degraded search paths use the
shared composed pipeline: bounded session history resolves follow-ups for
retrieval, then candidates are ranked/reranked and used for evidence-grounded
inference. Strong unfiltered auto-search remains a distinct direct-first path: it
queries the original request, applies its direct ranking and sufficiency gate,
and falls through to SerpAPI and browser search when needed. Every path persists
finalized answers, citations, documents, and stage metadata through the same
JSON/SSE response tail. This internal simplification introduces no new public API
and does not change the request or response schemas.


## Search-source agents

The function-calling `search` and `web_search` tools accept an optional `domain`.
MCP's `search_web` accepts it too. The shared taxonomy lives in
`src/internal/tools/search.py`, which exports `AVAILABLE_DOMAINS`. The field keeps
its name, but each value is now a **search-source category** — the kind of source
a query should reach — rather than a subject. Subjects such as health, energy,
security, or gaming are part of the query, and one subject can suit several
sources: a medical-research question fits `academic` or `official`; a
vulnerability fits `code`, `official`, or `news`; company earnings fit
`financial`, `business`, or `news`.

Single-query tool arguments:

```json
{"query": "battery recycling", "domain": "academic"}
```

Multi-query tool arguments:

```json
{"queries": ["battery recycling", "lithium recovery"], "domain": "academic"}
```

The selected category applies to every query in that call. To search different
categories, issue separate calls; nothing fans a query out across several
categories or fuses their results automatically. Omission or `web` preserves
existing query behavior. Names accept case, surrounding whitespace, spaces, and
hyphens: `Social Media` and `social-media` normalize to `social_media`. Empty
names, unknown names, and explicit null values are rejected before searching.

For any category other than `web`, the tool appends the fixed query hint below
exactly once. For example, `battery recycling` becomes
`battery recycling academic research`. Existing provider selection, fallback
order, and authorization filters continue to apply.

| ID | Primary search topics | Typical sources | Query hint |
| --- | --- | --- | --- |
| web | General information, websites, explanations, references | Web search engines | empty string |
| academic | Papers, preprints, citations, conferences, research | arXiv, Semantic Scholar, OpenAlex | academic research |
| code | Repositories, APIs, libraries, examples, issues | GitHub, GitLab, package registries | programming repositories |
| social_media | Discussions, opinions, trends, community experiences | Reddit, Hacker News, public social platforms | social media discussions |
| resource | Datasets, directories, tools, templates, benchmarks | Hugging Face, Kaggle, open-data portals | datasets resources tools |
| news | Breaking news, reporting, investigations, press releases | News publishers, news aggregators | news reporting |
| official | Official announcements, documentation, authoritative statements | Government agencies, organizations, official websites | official documentation announcements |
| legal | Statutes, regulations, case law, legal decisions | Court databases, legal repositories | law statutes case law |
| financial | Market data, filings, earnings, financial statements | SEC EDGAR, exchange data, financial providers | financial market data filings |
| patent | Patents, trademarks, inventions, intellectual property | Google Patents, USPTO, WIPO | patents intellectual property |
| product | Product catalogs, specifications, prices, reviews | Retailers, manufacturers, shopping indexes | product specifications prices reviews |
| business | Company profiles, industry information, organizations | Company registries, corporate websites | company industry information |
| local | Businesses, locations, hours, local services | Maps, business directories | local businesses locations services |
| travel | Flights, hotels, destinations, transportation | Travel providers, tourism websites | travel flights hotels |
| media | Images, videos, audio, podcasts, multimedia | YouTube, Wikimedia Commons, media platforms | images videos audio |
| books | Books, authors, editions, publications, libraries | Open Library, Google Books, library catalogs | books authors libraries |
| education | Courses, tutorials, textbooks, educational programs | University websites, course catalogs, learning platforms | courses tutorials education |
| events | Conferences, concerts, meetups, schedules | Event platforms, official event sites | events conferences schedules |
| jobs | Job openings, careers, salaries, employment opportunities | Employer career pages, job boards | job openings careers |
| statistics | Public statistics, indicators, surveys, census data | Census Bureau, World Bank, OECD | public statistics census indicators |

**Typical sources are descriptive metadata, not integrations.** They name the
source family a category is meant to reach, and they shape the query hint, but
no dedicated connector exists for most of them: every category searches through
the configured web providers, and only the public-data routes listed under
[native domain search](#native-domain-search-features) call a specific service.
Sixteen categories also carry `source_sites`, a separate list of hosts that
their web search is scoped to first; see
[category source sites](#category-source-sites). The rest (`web`, `official`,
`product`, `local`) only append the query hint.
Twenty categories do not mean twenty running agents.

Categories are query hints, not guaranteed category filters. They do not promise
source authority, freshness, or access to private social content. Added words
can reduce recall; use `web` to search without the hint. Unit tests verify
query handling, not improvements in real-world search relevance; the measured
relevance result below predates this taxonomy and found no detectable
improvement from the old topic hints.

### Legacy identifiers

The previous seventeen topic identifiers are still accepted as input, so
existing clients keep working, but discovery, schemas, and responses advertise
and return canonical names only:

| Legacy input | Canonical category |
| --- | --- |
| `general` | `web` |
| `finance` | `financial` |
| `ip` | `patent` |
| `film` | `media` |
| `health`, `security`, `energy`, `environment`, `agriculture`, `gaming` | `web` |

`academic`, `code`, `social_media`, `resource`, `legal`, `business`, and `travel`
were already canonical and keep their names. A legacy topic that maps to `web`
no longer appends its old subject word — the caller's query carries the subject.

For non-`web` multi-query calls, tool metadata includes the original sanitized
`queries`, canonical `domain`, and `executed_queries`. Single-query tool output
keeps its existing formatted response. See [MCP](mcp.md#web-search-domain-hints)
for MCP response metadata.

### Category source sites

Sixteen categories carry `source_sites`: the hosts (or host/path prefixes) a web
search for that category is scoped to first. It is distinct from the descriptive
`typical_sources` column. Each list has at most five entries, is fixed in code
like the rest of the registry, and is not operator-configurable. `web`,
`official`, `product`, and `local` have no list and keep the suffix-only
behavior; `official` has none because `site:gov` would be US-only.

| Category | `source_sites` |
| --- | --- |
| academic | arxiv.org, semanticscholar.org, pubmed.ncbi.nlm.nih.gov, openreview.net, aclanthology.org |
| code | github.com, gitlab.com, stackoverflow.com, pypi.org, npmjs.com |
| social_media | reddit.com, news.ycombinator.com, x.com |
| resource | huggingface.co, kaggle.com, data.gov, zenodo.org |
| news | reuters.com, apnews.com, bbc.com, npr.org |
| legal | law.cornell.edu, courtlistener.com, justia.com, govinfo.gov, eur-lex.europa.eu |
| financial | sec.gov, nasdaq.com, finance.yahoo.com, marketwatch.com |
| patent | patents.google.com, uspto.gov, wipo.int, epo.org |
| business | opencorporates.com, crunchbase.com |
| travel | tripadvisor.com, booking.com, lonelyplanet.com |
| media | youtube.com, commons.wikimedia.org, vimeo.com, podcasts.apple.com |
| books | openlibrary.org, books.google.com, goodreads.com, worldcat.org |
| education | coursera.org, edx.org, khanacademy.org, ocw.mit.edu |
| events | eventbrite.com, meetup.com |
| jobs | indeed.com, linkedin.com/jobs, glassdoor.com |
| statistics | census.gov, data.worldbank.org, data.oecd.org, ourworldindata.org |

`GET /api/search-domains` and the `get_sub_domains` directories expose
`source_sites` as an additive list next to `typical_sources`.

**Restricted first, then fallback.** For a site-listed category, `search_category`
first sends the query with a `site:` clause (`query (site:a OR site:b ...)`, no
suffix words). If that yields at least `min(3, page_size)` usable results (a URL,
no error, not blank), it stops. Otherwise it also runs the suffix query and
merges: restricted results first, then fallback results not already present,
truncated to `page_size`. A restricted leg that raises or times out counts as
zero results, so the fallback runs; a fallback that fails never hides restricted
results. Each page's metadata carries `category`, `category_leg`
(`restricted`, `fallback`, or `hint` for a single suffix call), and
`category_query`; these reach document metadata, including SerpAPI's page-fetch
step, and `executed_queries` lists every query actually sent.

Two paths never restrict. The browser sidecar always gets the suffix query,
because a restricted attempt plus a fallback could double a 30-50 s call; so the
`web_search` tool and `DomainSearch` restrict through SerpAPI only. Corpus
(`retrieval`) legs get the raw query. Categories without a list, and every
category when the setting is off, make the single suffix call they made before.
The same behavior applies on the HTTP direct and hybrid paths
(`_category_search_fns` in `src/internal/servers/web/app.py`), the multi-query
`web_search` tool, `DomainSearch`, and the MCP `search_web` tool.

`AGENTIC_SEARCH_CATEGORY_SITES` controls it. It defaults to on; `0` or `false`
makes every path hint-only, with exactly today's queries.

Known cost: when SerpAPI is down, a site-listed category query spends two
SerpAPI attempts (restricted, then the fallback cascade) before it reaches the
browser.

Whether this improves relevance is measured below.

### Does selecting a domain help? Measured, and the answer is: not detectably

The taxonomy shipped with a standing condition — no quality claim without
comparing general and domain runs on labelled queries across every category.
That comparison has now been run, and it did not find an effect.

**This measures the old topic hints, not the source categories.** The labels
are keyed by the sixteen pre-source topic identifiers, and their authority hosts
were chosen for those subjects. `HISTORICAL_TOPIC_HINTS` in
`src/internal/retrieval/domain_eval.py` freezes those topics' hints, so a rerun
reproduces the original measurement (and hits the same cache) rather than
silently reading `health` or `gaming` as a `web` alias with an empty hint — which
would make both arms identical. A label file keyed by a source category is
rejected. No source-category relevance has been measured; that would need new
labels, not relabelled old ones.

Run it yourself:

```bash
python -m examples.run_domain_relevance_eval --dry-run   # cost, no calls
python -m examples.run_domain_relevance_eval
```

The labelled set is `data/eval/domain_relevance_queries.json`: 16 topic domains
from the pre-source taxonomy,
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

### Does restricting to source sites help?

This repeats the measurement above for the source categories, with a third arm.
Each query runs three ways against SerpAPI: **plain** (raw query), **suffix**
(the old hint-only query), and **category** (`search_category`). Each arm is
scored by authority precision@10, paired category-minus-suffix and
category-minus-plain, one query per unit, with a paired permutation p-value and
Cliff's delta pooled across categories; per-category figures are descriptive
only.

The labels are `data/eval/source_category_queries.json`: 64 queries, four for
each of the sixteen site-listed categories, each with the `authority_hosts` an
on-topic result should come from. They were drafted by a subagent that was not
shown the site lists, spot-checked by the user, and frozen; none were added or
edited after the run.

**Circularity.** A restricted arm scores well by construction if the authority
hosts copy the site lists. 90 of the 245 authority hosts (36.7%) are on their
category's own site list. The report's `on_list_share` field is the per-query
mean, 36.4%, which includes excluded rows; it ranges from 6% (business, travel)
to 69% (patent).

A failed arm is excluded and never cached, and a query whose arm errored or came
back empty is dropped from the paired comparison. `web` control queries must
show a zero delta and an overlap of 1.0; a run without at least one passing
`web` control is rejected and yields no decision.

**Decision rule, fixed before the run.**

- Category significantly worse than suffix (p < 0.05, negative delta): the same
  change flips `AGENTIC_SEARCH_CATEGORY_SITES` to default off and documents why.
- Significantly better: documented as a measured gain.
- Neither: documented as not detectably better; the setting stays on, since the
  fallback keeps results from coming back empty.

Run it yourself (194 live SerpAPI calls; responses are cached under
`data/eval/cache/`, and failed arms are not):

```bash
python -m examples.run_source_category_eval --dry-run   # cost, no calls
python -m examples.run_source_category_eval --control "what is a bond" --control "how do vaccines work"
```

The report is written to `data/eval/source_category_relevance.json`.

The measured result is pending; this section will report it, whichever way it falls.

### Domains over HTTP and in the web UI

`POST /api/agent` accepts an optional `domain` alongside `query`. It defaults to
`web`, which leaves the query unchanged and keeps behavior identical to
requests that omit the field. Legacy identifiers are normalized first, so
`"finance"` arrives as `financial` and `"health"` as `web`.

```json
{"query": "etf fees", "domain": "financial"}
```

`GET /api/search-domains` returns each category's `name`, `description` (its
primary topics), and `typical_sources` array in registry order. The Assist page
uses it to populate the "Search source" selector next to the question box, so
the 20 categories are not copied into the frontend bundle; an option's tooltip
shows its topics and typical sources. If discovery fails, the selector keeps a
single `web` option. The selector is always visible, unlike the Source and retrieval-URL fields,
which appear only under `?dev=1`.

Three modes honor the field, because in each one a single caller-supplied query
reaches a provider: the default auto mode, `search_tool`, and `hybrid_search`.
`chat_once` retrieves from the local corpus only, where the hint never applies,
and `chat_loop`, `search_agent`, and
`tool_agent` build their own per-round queries internally, so a non-`web`
category combined with an explicit one of those modes returns 400 rather than
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
each of the 20 categories to a `SearchDomain` carrying its description, its
query hint, its capabilities, and its typical sources:

```python
"financial": SearchDomain(
    "Market data, filings, earnings, financial statements",
    "financial market data filings",
    (
        WEB_CAPABILITY,
        Capability("quote", "symbol", "records", "get_stock_quote"),
        Capability("crypto", "symbol", "records", "get_crypto_price"),
        Capability("currency", "from_currency", "records", "convert_currency"),
    ),
    ("SEC EDGAR", "exchange data", "financial providers"),
),
```

A tag such as `financial.quote` is **derived** — `f"{domain}.{capability.name}"`
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
source category is orthogonal to a modality: a financial question may be an
article, a table, or a live quote.

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
through `/api/debug/tools` (admin only) when `AGENTIC_SEARCH_DEBUG_PANELS` is set.
`extract_page` is agent-callable, because nothing else seeded there fetches a
URL.

| Operation | Behavior |
| --- | --- |
| `get_sub_domains(domains)` | Describe available routes and their real tool schemas for 1–5 domains; no network calls |
| `search_domain(query, domain, tag, params)` | Search the web or invoke an implemented capability |
| `extract_page(url, max_length)` | Reuse the existing HTTP(S) fetcher and HTML-to-text extraction |
| `batch_search(queries, ...)` | Run 1–5 searches concurrently, preserving input order and per-query errors |

All 20 categories offer `<domain>.web`, which appends the category's query hint and
uses the configured web-search flow. These are broad searches, not guaranteed
category filters. Specialized routes use these existing tools:

| Tag | Existing tool | Supply `query` as |
| --- | --- | --- |
| `web.wikipedia` | `search_wikipedia` | Search text |
| `academic.arxiv` | `search_arxiv` | Paper search text |
| `resource.wayback` | `search_wayback` | Archived page URL |
| `financial.quote` | `get_stock_quote` | Ticker symbol |
| `financial.crypto` | `get_crypto_price` | Cryptocurrency symbol |
| `financial.currency` | `convert_currency` | Source currency; put amount and target currency in params |
| `local.weather` | `get_weather` | Place name |
| `local.location`, `travel.location` | `search_location` | Place/address query |
| `local.nearby`, `travel.nearby` | `search_nearby_places` | Place type; put latitude and longitude in params |

Legacy tags are migrated before the domain/tag agreement check, so they still
reach the same tool with the same arguments: `general.wikipedia` → `web.wikipedia`,
`finance.quote`/`crypto`/`currency` → `financial.*`, and `environment.weather` →
`local.weather`. Any other legacy prefix follows the identifier aliases above, so
`environment.web` is `web.web`. Discovery lists canonical tags only.

Discovery returns each domain's `typical_sources` and, for each route, its
`tool_name`, `query_parameter`, `query_format`, complete `parameters` schema, and
additional `params`. It lists only available
read-only public-data implementations, plus web routes. It is a local capability
catalog, not a check of upstream service availability. Private corpus search and
arbitrary registered tools are outside this routing table.

### Examples

Discover first:

```json
{"domains": ["financial", "academic"]}
```

Search with the resulting tag:

```json
{"query": "AAPL", "tag": "financial.quote"}
```

```json
{"query": "retrieval augmented generation", "tag": "academic.arxiv", "params": {"limit": 3}}
```

For an untagged call, `domain` selects its web route. With a tag, the domain is
inferred from the tag or validated against an explicitly supplied domain.
Specialized routes receive the original query without appended topic words.
Unknown tags, mismatched prefixes, and unsupported or missing capability
parameters fail before dispatch. Provider-specific options
must be present in the discovered schema; unsupported region/language options
are not silently accepted. Set a supported language through capability params.

`max_results` is 1–10: through the tool registry a larger value is rejected by
the tool schema, while the Python service clamps it. A capability's `limit` is
bounded by `max_results`.
Extraction defaults to 5,000 characters with a configurable maximum of 50,000,
and reports fetch failures explicitly. Its text format follows the existing
fetcher; it does not introduce additional file-format support.

Batch requests accept the same search fields per item and shared defaults:

```json
{
  "domain": "academic",
  "queries": [
    {"query": "dense retrieval"},
    {"query": "AAPL", "tag": "financial.quote"}
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
    directory = search.get_sub_domains(["financial"])
    result = await search.search("AAPL", tag="financial.quote")
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
