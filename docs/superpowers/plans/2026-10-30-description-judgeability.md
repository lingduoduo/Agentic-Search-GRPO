# Description judgeability plan

**Spec:** ../specs/2026-10-30-description-judgeability-design.md
**Goal:** Measurable routing utility, coverage-aware descriptions with provenance, and bounded ACL-safe preview recovery.
**Architecture:** Independent offline description and evaluation modules; root integrates runtime trace/recovery.
**Constraints:** Existing dependencies; no weights in tests; no membership changes during evaluation; explicit budget/cost reporting.

- [ ] Coverage: tests fail first for tail/language/boundary coverage, real packed-prompt provenance, cache-policy migration and blank artifact validation; implement sampler/adapter/builder integration.
- [ ] Recovery: tests fail first for fixed-beam rescue, budget exhaustion, scan bound, duplicate/weak detection, no-filter and filtered leakage, exact usage and evidence paths; implement runtime flags/API/trace.
- [ ] Evaluation: tests fail first for fixed model/tree/config, missing/pruned/ragged/multiple gold paths, conditional denominators, paired bootstrap, immutable variants, CLI and invalid labels; implement API/CLI.
- [ ] Integration: docs, synthetic end-to-end report, independent review/fixes, focused and full suites, lint and PR.
