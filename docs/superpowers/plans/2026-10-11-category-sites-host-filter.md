# Category source sites host filter — plan

**Spec:** `docs/superpowers/specs/2026-10-11-category-sites-host-filter-design.md`. Approved in chat (option a: keep the `site:` clause and filter the results).

- [x] RED: `site_matches` cases (subdomain, dot boundary, host/path, empty or invalid URL); off-list restricted results don't count; no on-list results returns the fallback as today.
- [x] GREEN: `site_matches` added, and `search_category` filters restricted pages through it.
- [x] Mutation check: removing the filter turns 2 tests red.
- [x] `pytest` full suite green.
- [x] Re-score the pilot from cache with the API key unset (0 live calls): +0.126, p=0.064, null. Report in `data/eval/source_category_relevance_hostfilter.json`.
- [x] Docs: search-engine (host filter, re-score table, re-analysis caveat, default stays off) and the configuration row.
