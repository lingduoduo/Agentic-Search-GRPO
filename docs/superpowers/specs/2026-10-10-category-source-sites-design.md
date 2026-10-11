# Category source sites: real per-category sources

Sub-project 1 of 3 toward multi-category search. Sub-project 2 (multi-category
fan-out with fused results) and sub-project 3 (automatic category selection)
each get their own spec, plan, and PR, in that order, after this one.

## Intent

Today every search-source category reaches the same web providers; the only
difference is a few words appended to the query, and #588 measured no
detectable relevance effect from such hints (p=0.33). This sub-project makes a
category reach its sources: a web query for a category with a site list is
first restricted to those sites, and falls back to today's suffix query when
the restricted search comes back thin. One category per request, as today.

Success means the behavior ships with tests proving each entry point reaches
the category's sites, and a pre-registered measurement on independent labels
reports whether it improves source relevance. The feature ships either way;
the measured result decides its default (see Evaluation).

## Registry

`SearchDomain` gains `source_sites: tuple[str, ...] = ()`: the hosts (or
host/path prefixes) the restricted leg searches. It is distinct from the
descriptive `typical_sources`. Each list has at most five entries, keeping a
restricted query well under Google's 32-word limit. Entries are bare hosts or
host/path, with no scheme and no `site:` prefix. Lists are fixed in code like
the rest of the registry; they are not operator-configurable.

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

`web`, `official`, `product`, and `local` have no list and keep today's
suffix-only behavior. `official` has none because `site:gov` would be US-only.

`GET /api/search-domains` and `get_sub_domains` directories expose
`source_sites` as an additive list field next to `typical_sources`.

## `search_category`

A new function in `src/internal/tools/search.py`, beside `search_tool`, taking
the same provider arguments:

```python
@dataclass(frozen=True)
class CategorySearch:
    pages: list[SearchPage]
    executed_queries: list[str]

async def search_category(
    query: str, *, domain: str, provider: SearchProvider, page_size: int = 5,
    **provider_kwargs,
) -> CategorySearch
```

`domain` is normalized with `normalize_search_domain`. For each provider:

- **`retrieval`** (local corpus): one call with the raw query. No hint, no
  restriction — the existing rule.
- **Browser sidecar**: one call with the suffix query, as today. A restricted
  attempt plus fallback could double a 30–50 s call.
- **Web providers** (`google`, `serpapi`, `serper`) for a category without
  `source_sites`, or with the setting off: one call with the suffix query
  (`prepare_domain_query`), as today.
- **Web providers for a category with `source_sites`**:
  1. Restricted leg: `f"{query} ({' OR '.join('site:' + s for s in sites)})"`.
     No suffix words; the sites already scope the query.
  2. Usable pages are those with a URL, no error, and not `is_blank`. If the
     restricted leg yields at least `min(3, page_size)` usable pages, stop.
  3. Otherwise run the fallback leg — the suffix query. Merge: restricted
     usable pages first, then fallback pages whose URL is not already present,
     truncated to `page_size`.
  4. A restricted leg that raises (including a timeout) counts as zero usable
     pages, so the fallback runs. If the fallback also fails, its outcome is
     returned exactly as a single suffix call's would be today: the same
     exception, error pages, `timed_out` flags, and stale-cache behavior.

Each returned page's `metadata` gains `category` (canonical name) and
`category_leg`: `restricted`, `fallback`, or `hint` (single suffix call).
Pages from `retrieval` are not tagged. `SearchPage.metadata` already flows into
`ContextDocument.metadata`, so source cards and the evaluation see the leg
without new plumbing. `executed_queries` lists every query actually sent, in
order, so tool and MCP metadata stay truthful when the fallback ran.

Each leg is an ordinary `search_tool` call: its query string is its own serving
cache key, and the existing timeout and retry policies apply per leg. In the
hybrid path the outer `provider_wait_seconds` still bounds the whole category
search, so a slow fallback degrades exactly as a slow provider does today.

`prepare_domain_query` stays: it builds the suffix query for the fallback leg
and the hint-only categories.

### Setting

`AGENTIC_SEARCH_CATEGORY_SITES` (default on; `0`/`false` disables). Off, every
path makes exactly the calls it makes today. It is read through the existing
typed configuration, not ad hoc `os.environ` lookups in the call path.

### Wiring

Every site that currently builds a hinted query with `prepare_domain_query`
and then calls a provider switches to `search_category`, so HTTP, tools, and
MCP behave identically:

- `src/internal/servers/web/app.py`: `_run_direct_search` (provider loop and
  browser sidecar) and `_run_hybrid_search` (each expanded query per web
  provider, and the browser leg). Hints are still applied after query
  expansion, once per expanded query. The corpus (`retrieval`) legs keep
  passing the raw query.
- `src/internal/tools/search.py`: the multi-query web search tool and the
  `DomainSearch` `.web` capability.
- `src/internal/mcp_server/tools/search.py`: `search_web`.

Specialized public-data capabilities (`financial.quote`, `academic.arxiv`, …)
are unchanged: they already call a specific service.

## Evaluation

### Labels

A new file `data/eval/source_category_queries.json` with a top-level
`"taxonomy": "source_categories"` marker, four queries for each of the sixteen
categories with `source_sites` (64 queries), each carrying `authority_hosts`.
The loader dispatches on the marker: the marked file takes the three-arm source
path; the unmarked historical file keeps its pinned two-arm topic path and its
existing rejection of source-category keys. Suffix-only categories are not
labelled: their restricted arm would equal their suffix arm.

Circularity guard: a restricted arm scores near-perfect authority precision by
construction if `authority_hosts` copy `source_sites`. Therefore:

- A separate subagent drafts the labels without being shown the `source_sites`
  lists; the user spot-checks them before the run.
- The report states, per category and pooled, the share of authority hosts that
  appear in that category's `source_sites`.

No labels are invented after the run or edited in response to its results.

### Arms and metrics

Per query: **plain** (raw query), **suffix** (today), and **category**
(`search_category`). The fallback leg's query equals the suffix arm's, so it is
a cache hit: at most 64 × 3 = 192 SerpAPI calls. `--dry-run` reports the exact
number of live calls before spending any.

- Primary: authority precision@10, paired category−suffix and category−plain,
  one query per unit. Paired permutation p-value and Cliff's delta, pooled
  across categories; per-category figures descriptive only, no per-category
  p-values. A query whose arm errored or came back empty is excluded from the
  paired comparison, as in #588.
- Secondary: empty-result rate per arm, fallback rate (pooled and per
  category), and the added latency of the fallback leg when it fires.
- Control: plain-query control labels run through the same harness must show a
  zero delta and an overlap of 1.0, or the run is rejected.

### Decision rule (pre-registered)

- Category significantly worse than suffix (p < 0.05, negative delta): the same
  PR flips `AGENTIC_SEARCH_CATEGORY_SITES` to default off and documents why.
- Significantly better: documented as a measured gain.
- Neither: documented as not detectably better; stays on, since the fallback
  keeps results from coming back empty.

The run is a manual step with the user's SerpAPI key, not CI. CI runs only the
offline tests. The report is written to `data/eval/` like the #588 report.

## Testing

All offline with stub providers; every new test is mutation-checked (delete
the feature, see it fail).

- `search_category`: restricted query shape (site clause, no suffix words);
  threshold boundary (2 vs 3 usable, and `page_size` below 3); merge order and
  URL dedupe; `category_leg` tags; restricted exception or timeout triggers the
  fallback; both legs failing propagates the fallback's outcome unchanged
  (`timed_out` preserved); blank pages are not usable; suffix-only categories,
  the browser, and `retrieval` each make exactly one call as today.
- Wiring: each call site through its real entry point — HTTP direct, HTTP
  hybrid with expanded queries, multi-query tool, `DomainSearch` `.web`, MCP
  `search_web` — asserts the restricted query reaches the provider and the
  corpus leg stays raw.
- Setting off: every path issues today's exact queries.
- Registry: entries are bare host or host/path, no scheme or `site:` prefix;
  at most five per category; discovery exposes `source_sites`.
- Evaluation: the marker selects the three-arm path; the historical file still
  loads on its two-arm path; the circularity share is computed correctly.

## Documentation

`docs/search-engine.md` gains the `source_sites` table, the restricted-then-
fallback behavior, the setting, and the measured result in whichever direction
it falls. The README changes only once the measurement exists.

## Out of scope

More than one category per request (sub-project 2); automatic category
selection (sub-project 3); operator-configurable site lists; SerpAPI vertical
engines such as Google News or Scholar; any change to the corpus leg, ranking,
or reranking.
