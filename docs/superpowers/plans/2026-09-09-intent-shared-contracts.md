# Shared Intent Contracts Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** Share intent vocabulary/defaults, fix CLI scoring parity and remove the remaining unused wrapper.

**Architecture:** A standard-library-only shared_configs module supplies web types, offline labels/scoring defaults, and application configuration. The CLI uses the existing scorer with all configured scoring arguments.

**Tech Stack:** Python >=3.10, enum, dataclasses, NumPy, pytest, Ruff.

**Spec:** `docs/superpowers/specs/2026-09-09-intent-shared-contracts-design.md`

## Global Constraints

- Defaults remain top_k=8, min_route_margin=0.010, min_module_score=0.8215.
- Route values/order remain chat, search, tool.
- Preserve public import locations, API fields, index format, encoder, scoring and evaluation grids.
- No new dependencies or web/model imports in shared configuration.
- Work in `/Users/linghuang/Git/Agentic-Search-intent-parity`, branch `refactor/intent-shared-contracts`.
- Local Python: `/Users/linghuang/miniconda3/bin/python`.

### Task 1: Prove and fix CLI/web scoring parity

**Files:** `tests/unit/test_run_agentic_search.py`, `examples/agentic_search/routing.py`.

**Interface:** `_load_intent_prediction(index_dir, question)` consumes AppSettings;
its call to `IntentIndex.decide` must include `top_k=settings.intent_top_k`.

- [x] Add a real-index regression using search vectors [1,0,0] plus three [0,0,1], chat vectors [0.8,0.6,0], and tool vectors [0,0,1]. Stub only encode_texts with query [1,0,0]. Parameterize environment k=1 and k=4, expecting respectively search/1.0 and chat/0.8 from both CLI and web. Use load_app_settings with an explicit environment mapping patched at the loader boundary to avoid local .env drift.
- [x] Run `python -m pytest -q tests/unit/test_run_agentic_search.py -k scoring_parity` and observe the k=1 assertion failure.
- [x] Pass the missing configured argument:

```python
    decision = index.decide(
        encode_texts([question])[0],
        min_margin=settings.intent_min_route_margin,
        min_module_score=settings.intent_min_module_score,
        top_k=settings.intent_top_k,
    )
```

- [x] Run CLI tests and similarity adapter tests; preserve existing encoder-mismatch/abstention behavior. Commit the fix.

### Task 2: Share vocabulary/defaults and remove the unused wrapper

**Files:** create `src/shared_configs/intent.py`; modify `src/internal/servers/web/intent/types.py`, `rules.py`, `src/internal/configs/{app_configs,default_config}.py`, `src/model/pre_training/intents/{model,evaluation}.py`, relevant router tests and live docs.

**Interfaces:** Shared RouteStrategy retains the existing enum values; existing web import and offline INTENT_LABELS/TOP_K locations re-export the shared definitions.

- [x] Add a subprocess test importing shared contracts with web/model/numpy/torch blocked, asserting RouteStrategy("search") serializes as "search". Add default-case CLI/web scoring parity to Task 1's real-index fixture and retain meaningful heuristic coverage through recognize_intent.
- [x] Create the shared definitions:

```python
from enum import Enum

class RouteStrategy(str, Enum):
    CHAT = "chat"
    SEARCH = "search"
    TOOL = "tool"

INTENT_LABELS = tuple(strategy.value for strategy in RouteStrategy)
DEFAULT_TOP_K = 8
DEFAULT_MIN_ROUTE_MARGIN = 0.010
DEFAULT_MIN_MODULE_SCORE = 0.8215
```

- [x] Replace duplicated definitions with imports; alias DEFAULT_TOP_K as TOP_K in model.py and DEFAULT_MIN_MODULE_SCORE as the evaluation fallback. Re-export RouteStrategy from web types. Change both AppSettings construction defaults and load_app_settings environment fallbacks, plus the DEFAULT_CONFIG mirror, to consume constants.
- [x] Remove `_rule_based_route` and its redundant tests. Preserve `_rule_based_route_or_none` and cover fallback tool/search/no-signal behavior through recognize_intent. Correct CLI top-3 and evaluation serving-k prose without changing grids or metrics.
- [x] Run configuration, taxonomy, model, CLI, similarity and web-router tests, then repository Ruff. Commit the consolidation.

### Task 3: Review, final verification and PR

- [x] Run `python -m pytest -q -o addopts='' tests/unit tests/regression`; record passed/skipped results and skip reasons. Do not claim encoder evaluations when the optional built index is absent.
- [x] Use the requesting-code-review skill for independent review of the branch while preparing PR documentation. Resolve material findings and rerun affected checks if code changes.
- [x] Update this plan with execution evidence, push `refactor/intent-shared-contracts`, and create a PR against main using an exact body file. Retain the worktree for review; do not merge.


## Execution record

- Spec/plan committed before code: `69bdde3a`.
- CLI parity fix: `03cc3d57`; shared contracts and wrapper cleanup: `ce6af390`.
- RED: the k=1 real-index regression failed with CLI chat versus expected search;
  web scoring already selected search. The isolated shared-module import test
  failed before the new module existed.
- GREEN: 39 CLI/adapter tests passed after the fix. After consolidation,
  201 focused tests passed with 3 optional-index skips.
- Final backend run: **3,763 passed, 3 skipped**, 39 existing warnings in 91.69s.
  Skips are the encoder accuracy, out-of-scope separation and latency checks:
  the isolated worktree has no built `data/intent_index/index.npz`.
- Repository-wide Ruff lint and formatting passed (971 files), as did diff
  whitespace checks and commit hooks.
- Independent review approved with no findings; it also verified public enum
  identity, JSON serialization, route order and default consistency.
- Published [PR #570](https://github.com/lingduoduo/Agentic-Search-GRPO/pull/570).
  No merge or deployment was performed.
