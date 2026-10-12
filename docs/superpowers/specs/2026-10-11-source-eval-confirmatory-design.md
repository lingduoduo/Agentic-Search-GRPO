# Source-category eval: confirmatory run on held-out labels

## Why

The host-filter re-score (#684) trended positive (+0.126, p=0.064), but it
re-analysed the 16 pilot queries after the mechanism changed. A confirmatory run
on fresh data was needed for a verdict.

## Change

- `--skip-per-category K` drops the first K labels of each category, in file
  order, before `--per-category` applies. With `K=1` it holds out exactly the 16
  pilot queries and keeps the other 48. The selection is mechanical and fixed
  before any result exists. The report's `sample` records `skip_per_category`.
- Docs record the confirmatory result, its caveats, and that the default stays
  off.

## Run

`--skip-per-category 1 --no-plain --control "what is a bond"` on SerpAPI: 96
planned calls, 102 used. A temporary operator timeout file
(`AGENTIC_SEARCH_TIMEOUTS_PATH`; 90 s, one attempt) kept slow `site:` queries
from failing or being charged twice. Result: +0.094, p=0.005, Cliff's delta
0.14, n=44 — a pre-registered "better". The gain is concentrated in five
categories, part of it comes from the 35% on-list label overlap, and the
fallback adds about 8.6 s for 83% of queries.

## Decision

Document the gain and keep `AGENTIC_SEARCH_CATEGORY_SITES` off by default until
the latency is reduced.
