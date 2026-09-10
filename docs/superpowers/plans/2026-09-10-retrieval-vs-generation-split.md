# Retrieval vs generation split — implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** Separate retrieval-side from generation-side signals in eval results, per-request telemetry, the human feedback loop, and the reward rollup; remove the dead pieces the investigation found.

**Architecture:** One torch-free metric taxonomy shared by the eval-results endpoint and panel. One ContextVar-scoped `RequestStageMetrics` fed from the two client-side choke points (`SearchClient.retrieve`, the two LLM backends), persisted per turn and rolled into `StageLatencyStats`, exposed on the dev latency endpoint and a new admin metrics endpoint. A `target` field on session feedback carried from the UI to the training loader.

**Tech Stack:** Python >=3.10 stdlib for the new modules, FastAPI, pytest; React 19 + Vitest for the panel and feedback bar.

**Spec:** `docs/superpowers/specs/2026-09-10-retrieval-vs-generation-split-design.md`

## Global Constraints

- TDD per task; mutation-check every new test by deleting the behaviour it pins.
- No torch import on the web request path (`src/internal/observability/` stays stdlib).
- No existing reward value, `dim_*` output, or persisted metadata key changes; additions only.
- Branch `feat/eval-observability-split`, worktree `.claude/worktrees/eval-observability`. Never commit to main.
- Local Python: `/Users/linghuang/miniconda3/envs/agentic-search-local/bin/python`; frontend `cd web && npm run test:ci`.

### Task 1: Metric taxonomy

**Files:** create `src/internal/observability/metric_taxonomy.py`, `tests/unit/observability/test_metric_taxonomy.py`.

- [x] Tests: exemplars per group; parent-path disambiguation; `flatten_metrics` filtering and depth; `group_metrics` omits empties.
- [x] Implement.

### Task 2: Eval results endpoint + panel grouping, eval_runner `--output`, dead workers endpoint

**Files:** `src/internal/servers/web/debug_router.py`, `tests/unit/servers/web/test_debug_router.py`, `web/src/types.ts`, `web/src/components/debug/EvalResultsPanel.tsx`, `web/src/components/debug/__tests__/EvalResultsPanel.test.tsx`, `src/internal/retrieval/eval_runner.py`, `tests/unit/retrieval/test_eval_runner.py`.

- [x] Tests: endpoint returns `groups` and a flattened `metrics` for a Bamboogle summary and a reranked eval_runner file; `--output` writes the printed JSON; the workers endpoint is gone.
- [x] Implement; delete `/workers` and its two tests; panel renders one table per group.

### Task 3: Reward sides + Bamboogle summary

**Files:** `src/model/post_training/reward.py`, `src/model/post_training/eval/bamboogle.py`, `tests/unit/test_reward.py` (or a new `test_reward_sides.py`), `tests/unit/test_bamboogle_eval.py`.

- [x] Tests: `reward_sides` sums equal dimension sums; summary carries `avg_reward_retrieval` / `avg_reward_generation`, `None` without a reward fn, printed when present.
- [x] Implement.

### Task 4: Stage metrics module

**Files:** create `src/internal/observability/stage_metrics.py`, `tests/unit/observability/test_stage_metrics.py`.

- [x] Tests: no-op outside a request; accumulation; snapshot shape; `StageLatencyStats` percentiles, averages, cache-hit rate; bounded window.
- [x] Implement.

### Task 5: Hook the choke points

**Files:** `src/context/retrieval/client.py`, `src/internal/llm/providers.py`, `src/model/serving.py`, `tests/unit/test_search_client_cache.py` (extend), `tests/unit/test_llm_providers*.py` (find the existing provider test), `tests/unit/observability/test_stage_metrics.py`.

- [x] Tests: `SearchClient.retrieve` notes elapsed/docs and `cache_hit` on a full hit; `complete` notes tokens from `usage`; local `generate` notes token counts.
- [x] Implement.

### Task 6: Request wiring, exposure, admin endpoint

**Files:** `src/internal/servers/web/app.py`, `src/internal/servers/web/request_capture.py` (`pipeline_stage_summary` gains `timing`), create `src/internal/servers/web/metrics_router.py`, `src/internal/servers/web/debug_router.py`, tests under `tests/unit/servers/web/`, `web/src/types.ts`, `web/src/api.ts`, `web/src/components/debug/LatencyPanel.tsx` + test.

- [x] Tests: a `/api/agent` turn persists `pipeline_stages.timing` and records into `STAGE_LATENCY`; `/api/debug/latency` returns `stages`; `/api/admin/metrics` 401/403 without admin, three sections with; panel renders the stage table.
- [x] Implement.

### Task 7: Feedback target end to end

**Files:** `src/internal/servers/retrieval/feedback_router.py`, `src/internal/db/store.py`, `src/internal/servers/evals/api.py`, `src/model/post_training/data.py`, tests (`test_feedback_router.py`, `db/test_retrieval_feedback.py`, evals tests, `test_feedback_grpo*.py`), `web/src/api.ts`, `web/src/components/AnswerPanel.tsx`, `web/src/pages/AssistPage.tsx`, `web/src/components/__tests__/AnswerPanel.test.tsx`, `web/src/components/__tests__/ConsoleNav.test.tsx`.

- [x] Tests: `target` validated and persisted; `by_target` rates; loader metadata; panel posts the right pairs and disables; ConsoleNav mock updated.
- [x] Implement.

### Task 8: Docs

**Files:** `docs/configuration.md`, `docs/retrieval.md`, `docs/training-and-evaluation.md`, `docs/api-reference.md` (if it lists admin/debug endpoints).

- [x] Drop the `FUSION_WEIGHTS_PATH` row; document `target`, `by_target`, `/api/admin/metrics`, `stages`, the eval-results grouping, `reward_sides`, `--output`.

### Task 9: Verify and ship

- [ ] `ruff check . --fix && ruff format .`; full `pytest`; `cd web && npm run test:ci`.
- [ ] Independent review of the whole branch; fix findings.
- [ ] Push, open PR with spec + plan.
