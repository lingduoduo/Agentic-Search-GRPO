# Category source sites: default off

## Why

The 2026-10-11 pilot of the source-category eval (#682 flags, 16 queries) came
out null: category − suffix +0.022, p = 0.77. Inspecting the restricted arm
showed that the restriction mostly did not happen. Only 38 of 158 restricted
results (24%) came from the category's listed sites, and 8 categories returned
none. Probes found Google, via SerpAPI, ignoring even a single `site:sec.gov`
and SerpAPI's `as_sitesearch`. The parenthesised OR query was also slow: two
probes timed out at 90 s, against a 15 s client timeout in production. The
fallback never fired, because about ten off-list results always came back.

The shipped feature therefore added latency and query noise without delivering
restriction, and the pilot does not measure what it was meant to.

## Change

- `AGENTIC_SEARCH_CATEGORY_SITES` defaults to `false` (`ServiceSettings.category_sites`
  and `load_app_settings`). `true` still opts in, so every path keeps working.
- Tests that exercise the restricted path set the variable explicitly. A new test
  pins that the default issues only the hint query.
- Docs (search-engine, configuration, mcp, README) state the default and the
  pilot finding. The null result is not reported as evidence about restriction.

## Next (separate design)

Enforce restriction on the results, not in the query: a restricted result counts
only if its host is on the list. The cached pilot responses can then be
re-scored at no API cost.
