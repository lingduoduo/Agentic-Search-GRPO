# Category source sites default off — plan

**Spec:** `docs/superpowers/specs/2026-10-11-category-sites-default-off-design.md`. A bounded change, approved in chat on 2026-10-11.

- [x] RED: `test_load_app_settings_reads_category_sites` and `test_category_sites_setting_defaults_off` expect the setting off by default.
- [x] GREEN: flip the default in `ServiceSettings` and `load_app_settings`.
- [x] Restricted-path tests (20) set `AGENTIC_SEARCH_CATEGORY_SITES=1` explicitly.
- [x] New test: the default HTTP direct search issues only the hint query. Mutation check: restoring the `True` default turns it red.
- [x] Docs: search-engine (default and pilot finding), configuration row, mcp, README.
- [x] Checks: `pytest` 5394 passed, 1 skipped; docs links and `git diff --check` clean.
