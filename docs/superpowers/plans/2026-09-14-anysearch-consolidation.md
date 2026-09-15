# AnySearch Consolidation Implementation Plan

> **For agentic workers:** Use `superpowers:executing-plans` to follow these tasks. The checked steps record the completed implementation. The user's explicit scope excludes a CLI.

**Goal:** Consolidate the supplied sample's operations into shared async library functions and repository tools.

**Architecture:** Keep the taxonomy module pure. Share one AnySearch client across native FunctionTools and the existing search-provider adapter; preserve default routing and tool catalogs.

**Tech Stack:** Python >=3.10, existing aiohttp shim, asyncio, FunctionTool, ToolRegistry, pytest.

**Spec:** [Consolidation design](../specs/2026-09-14-anysearch-consolidation-design.md)

## Global constraints

- No AnySearch CLI or import-time dotenv/stream modifications.
- Preserve the existing taxonomy and default query-hint behavior.
- Reuse the repository's HTTP dependency, tool framework, and search results.
- Keep native tags distinct from general topic hints.
- Native tools require explicit enablement; existing tool selection stays unchanged.
- Do not call the live service for unit tests or claim external API verification.

## Task 1: Shared client and validation

Files: create `src/internal/tools/anysearch.py`, `tests/unit/test_anysearch.py`;
restore the saved taxonomy portion of `src/internal/tools/search_domains.py`.

Produces `AnySearchClient`, `AnySearchError`, `normalize_search_item`, and
`parse_search_params`; consumers use the method signatures specified in the design.

- [x] Back up the supplied sample before edits and identify duplicate definitions/import side effects.
- [x] Add failing tests for native aliases, domain/tag mismatches, config, envelopes, errors, and batches; observe the missing client module failure.
- [x] Implement shared normalization and asynchronous search/discovery/extraction/batch methods.
- [x] Add regression tests for shared batch defaults and cancellation; retain per-item failures and input order.
- [x] Run `python -m pytest tests/unit/test_anysearch.py tests/unit/test_search_domains.py -q`.

## Task 2: Existing search-provider integration

Files: modify `src/internal/tools/search.py`,
`src/internal/mcp_server/tools/search.py`; create `tests/unit/test_anysearch_integration.py`.

Consumes `AnySearchClient`; produces `anysearch_pages`, `anysearch_search`, and
`provider="anysearch"` in existing search dispatch.

- [x] Write failing tests for provider selection, one-time domain hints, error pages, and MCP routing; observe unsupported-provider failures.
- [x] Add the shared SearchPage adapter and optional provider branch.
- [x] Add regression tests for credential-dependent cache bypass and unsupported pagination.
- [x] Run `python -m pytest tests/unit/test_anysearch_integration.py -q`.

## Task 3: Native repository tools

Files: create `src/internal/tools/anysearch_tools.py`, `tests/unit/test_anysearch_tools.py`;
modify `src/internal/tools/knowledge_base.py` and `src/internal/tools/__init__.py`.

Consumes client methods and canonical result adaptation; produces
`build_anysearch_tools(client=None)` and the four tool names listed in the spec.

- [x] Apply the user's correction: expose functions/tools and remove the attempted CLI and CLI-specific tests.
- [x] Add failing registry tests for all four operations, citation output, errors, and opt-in seeding.
- [x] Implement FunctionTool adapters with JSON output and structured provider errors.
- [x] Enable optional seeding through `AGENTIC_SEARCH_ANYSEARCH_ENABLED` and public Python exports.
- [x] Run `python -m pytest tests/unit/test_anysearch_tools.py tests/unit/test_knowledge_base.py -q`.

## Task 4: Documentation and verification

Files: modify `.env.example`, `docs/search-engine.md`, and `docs/mcp.md`;
include this plan and companion design.

- [x] Document environment configuration, native methods, tag semantics, batch behavior, tool registration, and existing MCP provider selection.
- [x] Run the combined affected regression suites and record the result below.
- [x] Run Ruff lint/format and `git diff --check`.
- [x] Review final changes and commit the consolidated implementation on its feature branch.

## Execution notes

The client, provider, and native registry tests were observed failing before their
implementations. Existing query-hint behavior remains covered by the original
search-domain tests. Independent review was attempted but unavailable because the
review agent hit its usage limit; final review is performed locally.

## Verification result

166 tests passed across the affected client, native-tool, provider, search,
domain, cache, fallback, ACL, MCP, and seeding suites. Ruff lint/format and Git
whitespace checks passed. The environment emitted an existing SciPy/NumPy version
compatibility warning; no test failed. No live AnySearch request was made.
