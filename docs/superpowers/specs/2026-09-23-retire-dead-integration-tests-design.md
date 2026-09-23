# Retire the integration tests that reference nothing the app serves

## Goal

Remove the subset of the integration suite that the census in #633 establishes
cannot pass, on the same evidence standard #630 used for the indexing directory.

## The evidence

`examples/audit_integration_endpoints.py` buckets each test file by whether the
endpoints it reaches still exist. Twenty-two files have **no** surviving endpoint
even in the helper-aware mode — that is, even after crediting each test with
every endpoint of every `common_utils` module it imports. That is the strictest
of the two attributions, so 22 is the conservative count, not the generous one.

The paths are not near-misses. Probed by prefix rather than exact match against a
real `create_web_app()`, the families they use return zero routes:

| Endpoint family | routes |
|---|---|
| `/admin/mcp/...` | 0 |
| `/web-search/...`, `/admin/web-search/...` | 0 |
| `/persona` | 0 |
| `/admin/api-key`, `/admin/code-interpreter` | 0 |
| `/admin/image-generation/...` | 0 |

## Why these are rewrites, not repairs

Several of the subsystems still exist — `src/internal/mcp_server/` has five
modules, `src/internal/servers/web_search/` has six. What changed is the shape:
the functionality moved into separate services with different interfaces rather
than admin routes on the web app. So no path correction recovers these tests;
they would have to be rewritten against a different architecture.

That is worth stating plainly because it is the reason deletion is the right call
rather than a lazy one. A test whose endpoint moved is a repair. A test whose
subsystem was re-architected around it is a rewrite, and keeping a broken
placeholder does not make the rewrite more likely.

## What is removed

Twenty-two test files. Eleven directories lose their last test and go entirely,
including their orphaned `conftest.py` fixtures and, in `pruning/`, a 70-file
static website fixture that only `test_pruning.py` used:

```
connector_job_tests/github, connector_job_tests/jira,
tests/code_interpreter, tests/document_set, tests/external_apps,
tests/image_indexing, tests/llm_auto_update, tests/pruning,
tests/search_settings, tests/tags, tests/web_search
```

Eleven further files are removed from directories that keep other tests —
`craft/` (4), `personas/` (4), `discord_bot/`, `mcp/`, `no_vectordb/` — so those
directories shrink rather than disappear.

Total: 94 files, ~36,900 lines, most of it the website fixture.

## Verification

**Collection errors went down, not up.** The remaining integration tree had 16
collection errors on `main` and has 14 here — the deletion removed two and
introduced none. Checking `main` first was the point: a red integration tree in
this repo is usually pre-existing, and attributing it to the change in hand has
wasted time before.

`pytest tests/unit` is unaffected at 4362 passed, 1 skipped: `pyproject.toml`
sets `testpaths = ["tests/unit", "tests/regression"]`, so none of this was ever in
the default run — which is exactly why the drift went unnoticed.

Nothing outside the deleted directories references them, checked across `.py`,
`.toml` and `.yml`.

## What is deliberately left

The ~80 files the census calls "mixed" — reaching both live and dead endpoints.
The tool cannot decide those: a test importing `CCPairManager` is credited with
every endpoint that manager builds, whether or not it calls them. They need
per-test judgement, and guessing at scale is how the census was wrong twice
before it was right.
