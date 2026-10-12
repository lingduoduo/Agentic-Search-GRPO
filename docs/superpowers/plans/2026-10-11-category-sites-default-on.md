# Category source sites default on — plan

**Spec:** `docs/superpowers/specs/2026-10-11-category-sites-default-on-design.md`. Approved in chat on 2026-10-11.

- [x] RED: the default tests (configs, `category_sites_enabled`, HTTP default path) expect on; all 3 failed against the old default.
- [x] GREEN: flip the default in `ServiceSettings` and `load_app_settings`.
- [x] Docs: search-engine, configuration row, mcp (also corrected for parallel legs), README.
- [x] `pytest`: 5415 passed, 1 skipped.
