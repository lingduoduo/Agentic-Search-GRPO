# Category search: parallel legs with a grace window

## Why

The confirmatory run (#685) measured site restriction as better: +0.094, p=0.005.
With sequential legs, the fallback fired for 83% of queries and added about
8.6 s each time, and a slow `site:` query could add the whole 15 s client
timeout. That latency is why the setting stays off by default.

## Change

- `search_category` starts the restricted and suffix legs together.
- If at least `min(3, page_size)` on-list restricted results arrive, they are
  the result and a suffix leg still in flight is cancelled.
- Once the suffix leg returns, a restricted leg still running gets at most
  `RESTRICT_GRACE_SECONDS` (2.0) more, then is cancelled.
- Merge and failure rules are unchanged: on-list results first, then deduped
  suffix results. A failed suffix leg never hides restricted results. With no
  restricted results, the suffix outcome is returned as is.
- `time_budget` now bounds both legs from the call's start.
- Leaving the function, for any reason including outer cancellation, cancels
  the legs still running.
- `CategorySearch.fallback_used` reports whether suffix pages were used. The
  eval's `fallback_fired` reads it, because `executed_queries` now always lists
  both queries.

## Costs

Always two provider calls per site-listed query, against 1.83 measured: about
9% more quota when the setting is on. Latency becomes about the suffix query's
time plus at most 2 s.

## Not done

Restricting only the categories that gained: that would select on the same
data that measured them. The setting stays off by default; changing that is a
separate decision.
