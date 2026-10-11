# Category source sites: enforce restriction on result hosts

## Why

The 2026-10-11 pilot showed that providers relax the `site:` clause. Only 38 of
158 restricted results (24%) came from a category's listed sites, so
"restricted" did not mean restricted and the fallback never fired.

## Change

- `site_matches(url, sites)` in `src/internal/tools/search.py` matches the host
  or a subdomain at a dot boundary (`notsec.gov` and `sec.gov.evil.com` do not
  match `sec.gov`). For a host/path entry such as `linkedin.com/jobs`, the path
  must be that segment or below it.
- `search_category` counts a restricted result only if `site_matches` is true.
  Off-list results are dropped. When fewer than `min(3, page_size)` on-list
  results remain, the existing fallback runs; on-list results stay first.
- The `site:` clause stays in the query. It sometimes works (code and legal were
  10/10 on-list), and keeping it lets the cached pilot be re-scored at no cost.
- `AGENTIC_SEARCH_CATEGORY_SITES` stays off by default (#683).

## Measurement

Re-score the 16-query pilot from cache with the API key unset, so there are 0
live calls. Result: +0.126, p = 0.064, Cliff's delta 0.33, 5 better / 0 worse /
10 unchanged. That is a pre-registered null and a re-analysis, not a
confirmation. A confirmatory run on the 48 unused labels is the next step if
wanted.

## Out of scope

The latency of the `site:` OR query; changing the default; new labels.
