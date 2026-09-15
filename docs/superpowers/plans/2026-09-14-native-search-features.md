# Native Search Features Implementation Plan

> **For agentic workers:** Use `superpowers:executing-plans` to follow this plan. Checked steps record completed work.

**Goal:** Implement the sample's features with existing repository functions and update PR #584.

**Architecture:** One local `DomainSearch` service provides discovery, capability routing, extraction, and batching. Registry and MCP adapters share it; existing provider and public-data code perform the actual work.

**Tech Stack:** Python >=3.10, asyncio, existing FunctionTool, argument validation, search providers, and page fetcher.

**Spec:** [Native search features design](../specs/2026-09-14-native-search-features-design.md)

## Global constraints

- No additional external-service integration, credentials, transport client, or CLI.
- Keep the 17-domain taxonomy and existing topic hints.
- Derive capability parameter schemas from real tool implementations.
- Expose only known read-only public-data routes and existing web routes.
- Keep private corpus access filters and existing search behavior intact.
- Discovery is local; external requests are mocked in tests.

## Task 1: Remove the previous provider integration

- [x] Restore existing provider dispatch, configuration, and public documentation to the PR base before implementing native features.
- [x] Remove the added service client, adapters, tests, and superseded spec/plan.
- [x] Confirm the current source tree contains no references to the removed service.

## Task 2: Local search service

Files: create `src/internal/tools/domain_search.py` and `tests/unit/test_domain_search.py`.

Produces `DomainSearch.get_sub_domains`, `.search`, `.extract`, `.batch_search`,
plus shared parameter parsing and an allowlisted mapping to existing tools.

- [x] Write tests for discovery, all-domain web routes, native routing, invalid arguments, extraction, and batch behavior; observe missing-module failure.
- [x] Implement the service using existing public-data tools, `make_web_cascade_search`, `fetch_url`, and `validate_arguments`.
- [x] Verify actual stock-tool delegation, side-effect/arbitrary-tool exclusion, and cancellation.
- [x] Run `python -m pytest tests/unit/test_domain_search.py -q`.

## Task 3: Registry and MCP integration

Files: create `src/internal/tools/domain_search_tools.py`,
`tests/unit/test_domain_search_tools.py`, and `tests/unit/test_domain_search_mcp.py`;
modify tool exports, knowledge-base seeding, the MCP search module, and
`tests/unit/test_knowledge_base.py`.

Produces the four shared tool names and their registry/MCP result contracts.

- [x] Write failing registry tests for the four operations and citation flags.
- [x] Build native FunctionTools with the existing JSON/error adapter and seed them alongside the existing tools.
- [x] Write failing MCP tests; add typed wrappers that delegate to the same service.
- [x] Preserve MCP web-provider selection and prevent corpus-provider routing through the public web path.
- [x] Run the registry, seeding, and MCP tests.

## Task 4: Documentation, checks, and PR revision

- [x] Rewrite the spec, plan, search-engine docs, and MCP docs around the final native functionality.
- [x] Run the complete affected regression selection: 197 tests passed.
- [x] Run Ruff lint/format checks and Git whitespace checks.
- [x] Commit the revised implementation and push the existing PR branch.
- [x] Rewrite PR #584's title and body to describe only the final implementation.

## Verification notes

The 197-test selection covers native service/registry/MCP behavior, original
search-domain and search tools, cache and fallback behavior, ACLs, tool seeding,
agent-callability, and web loop runners. One existing SciPy/NumPy compatibility
warning appeared; no test failed. External transports were mocked. Local review
checks the routing allowlist, shared callbacks, copied schemas, and removal of the
superseded integration; no independent review is claimed.

## Follow-up: consolidate search modules

- [x] Move the taxonomy, domain service, and FunctionTool builders into `src/internal/tools/search.py`.
- [x] Remove the three former domain modules and update repository imports.
- [x] Update the current design and search documentation.
- [x] Run the affected search, MCP, registry, and agent regression tests.
