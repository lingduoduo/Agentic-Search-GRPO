# Corpus Hierarchy Implementation Plan

> Use superpowers:subagent-driven-development for independent modules and root-owned integration.

**Goal:** Implement and serve persisted structured/clustered corpus hierarchies with incremental maintenance.
**Architecture:** Normalize formats, build/maintain an offline artifact, then navigate it without online embedding calls.
**Tech Stack:** Python, existing NumPy/scikit-learn, BeautifulSoup, SentenceTransformers, Transformers, pytest.
**Spec:** docs/superpowers/specs/2026-10-30-corpus-hierarchy-design.md

## Global constraints
No new dependencies; lazy model imports; no downloaded weights in tests; backward compatible no-artifact navigation. Preserve others' edits.

## Tasks
- [x] Task 1: hierarchy_formats.py and focused test_hierarchy_formats.py. Tests first for metadata/URL paths, HTML/Markdown/layout sections, CJK passage preservation/bounds; implement normalization/extraction. Verify pytest and ruff.
- [x] Task 2: hierarchy.py and test_hierarchy.py. Tests first for deterministic clustering, artifact persistence/validation, incremental changes/deletions/cache reuse, compatible config, descriptions and granularity reports. Implement exact artifact/builder contracts from spec; verify pytest and ruff.
- [x] Task 3: hierarchy_models.py, hierarchy_cli.py and focused tests. Implement local lazy adapters and build/update/inspect with tests for model-independent structure path, model identity, help and local-only generation.
- [x] Task 4: integrate artifact in decision_retriever.py and retrieval/decision.py. Tests first for ragged group depths, description use, ACL-safe descriptions, CJK character bounds, per-kind budgets and hierarchy server wiring. Verify original 61 tests plus new ones.
- [x] Task 5: document both workflows, NQ explicit recipe, limitations and incremental snapshot semantics; run structure CLI smoke, focused/full suites, Ruff, independent review and fixes.

## Review focus
ACL descriptions must not leak hidden titles; malformed artifacts must fail before inference; changed encoder configuration must not reuse incompatible vectors; identical embeddings must not cause infinite splitting; edits/deletions must refresh descriptions without mutating previous artifact.
