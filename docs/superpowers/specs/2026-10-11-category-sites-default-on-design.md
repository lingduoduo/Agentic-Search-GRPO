# Category source sites: default on

## Why

The reasons for #683's default-off are resolved:

- **Restriction is enforced:** restricted results are now checked against the
  listed hosts (#684), so off-list results no longer count as restricted.
- **The effect is measured:** a confirmatory run on 48 held-out labels found a
  pre-registered "better" (+0.094 authority precision, p=0.005, #685).
- **The latency is gone:** both legs run in parallel with a 2 s grace window,
  so a category search takes about the suffix query's time plus at most 2 s
  (#686).

The user approved turning it on on 2026-10-11.

## Change

- `AGENTIC_SEARCH_CATEGORY_SITES` defaults to `true` (`ServiceSettings.category_sites`
  and `load_app_settings`). Setting it to `false` restores hint-only queries exactly.
- The default tests now assert on, and the default HTTP path sends both legs.
- Docs: search-engine, configuration, mcp (which was also stale about
  sequential legs) and README.

## Costs accepted

- **Provider calls:** two per site-listed category query.
- **Uneven benefit:** the gain is small and concentrated in five categories,
  and partly inflated by the 35% overlap between label hosts and site lists.
