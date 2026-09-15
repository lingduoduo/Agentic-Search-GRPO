# Agent Tool Menu Overlap Implementation Plan

**Goal:** Stop offering the tool agent a facade over tools it already holds
directly, without removing the facade from any other surface.

**Spec:** [Agent tool menu overlap](../specs/2026-09-14-agent-tool-menu-overlap-design.md)

**Scope:** Bounded. One existing mechanism (`NOT_AGENT_CALLABLE`), one seeding
module, its tests, and the search documentation. No new module, flag, or
interface.

## Task 1: Flip the agent-callability tests

Files: `tests/unit/test_domain_search_tools.py`.

- [x] Replace `test_seeded_features_are_available_to_agents`, which asserted all
      four tools were on the agent menu, with three tests: the four stay
      registered and invocable; the three facade tools are withheld while
      `extract_page` is offered; and no `CAPABILITY_ROUTES` target is reachable
      two ways from the agent menu.
- [x] Run them and confirm the two behavioral ones fail for the stated reason
      while the registration one already passes.

## Task 2: Withhold the facade tools

Files: `src/internal/tools/knowledge_base.py`.

- [x] Add `search_domain`, `get_sub_domains`, and `batch_search` to
      `NOT_AGENT_CALLABLE`; leave `extract_page` callable.
- [x] Extend the set's comment with the third exclusion reason and the PR #479
      precedent, so the next reader does not re-add them.
- [x] Re-run the flipped tests plus `tests/unit/test_public_data_seeding.py`.
- [x] Mutation-check: remove the three names again and confirm both behavioral
      tests go red; revert and clear stale bytecode.

## Task 3: Documentation

Files: `docs/search-engine.md`.

- [x] State that three of the four operations are registered but not offered to
      the agent loop, why, and which surfaces still reach them.
- [x] Verify the surface claims against the code rather than asserting them:
      `/admin/tools` is mounted unconditionally and lists `all_summaries()`,
      while `/api/debug/tools` is gated on `AGENTIC_SEARCH_DEBUG_PANELS`.

## Task 4: Verification

- [x] `ruff check` and `ruff format` clean.
- [x] Full unit suite: 4138 passed (4136 on the base commit, plus two net new
      tests).

## Verification notes

One run showed `test_mcp_document_tools.py::test_parser_watchdog_terminates_a_process_over_the_rss_limit`
failing. It passed in isolation and on both subsequent full-suite runs with
these changes applied. The test spawns a 384 MB subprocess and asserts the
watchdog kills it for exceeding a 256 MB RSS limit, so it is sensitive to
machine memory pressure. Treated as a pre-existing flake, not a regression from
this change; it is worth stabilizing separately.

The agent menu was inspected directly before and after: 15 tools before, 12
after (11 seeded plus the request-bound corpus search), with `search_domain`,
`get_sub_domains`, and `batch_search` still present in `all_summaries()`.

No independent review is claimed, and no measurement of model tool-selection
behavior was taken — see the spec's evidence note.
