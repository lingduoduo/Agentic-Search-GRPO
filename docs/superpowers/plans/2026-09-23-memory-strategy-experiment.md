# Sliding window vs rolling summary — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Measure recall and cost of `window-N` vs `summary-N` vs `full` on seeded synthetic conversations, using the repo's real working-memory code.

**Architecture:** One `examples/` script: a pure generator + scorer (unit-tested), and an async runner that replays each conversation through `AgenticSearchStore(":memory:")`, `load_working_memory` and `compress_session` with an `InMemoryCache`, answering probes via an OpenAI-compatible LLM.

**Tech Stack:** Python, existing `src.internal.memory.working`, `src.internal.db` store, `InMemoryCache`, `OpenAICompatibleLLM`; Ollama `llama3.2:3b`.

**Spec:** `docs/superpowers/specs/2026-09-23-memory-strategy-experiment-design.md`

## Global Constraints

- N ∈ {6, 10, 20, 40}; 12 conversations; 4 facts each; ~80 transcript messages; full history < 4,096 prompt tokens.
- No reimplementation of windowing or summarizing: call `load_working_memory` / `compress_session`.
- The answer prompt is `history + [probe]`, nothing else.
- No defaults or production code change.

## Review Focus

- A fact value appearing in filler or in another fact would make the scorer credit a guess — pinned by `test_values_are_unique_and_absent_from_filler`.
- Summary state leaking between strategies or conversations (shared cache or store) — pinned by `test_runs_are_isolated` in the fake-LLM run.
- `summary-N` silently behaving like `window-N` (summary never produced or never sent) — pinned by `test_summary_strategy_sends_summary`.
- A probe answer that echoes the question's attribute but not the value scored as correct — pinned by scorer tests.
- The in-window/dropped split computed from the wrong index (the probe turns themselves push facts out of the window) — pinned by `test_in_window_is_computed_at_probe_time`.

---

### Task 1: Generator + scorer (TDD)

**Files:** Create `examples/measure_memory_strategies.py`, `tests/unit/test_measure_memory_strategies.py`.

- [ ] Tests first: determinism per seed; 4 facts per conversation at the planned indices; values unique and absent from filler and other conversations' filler; every message ≤ ~60 words; full transcript + probes under a word budget standing in for 4,096 tokens (≤ 2,600 words); scorer normalization (case, `$`, commas, whitespace) and a negative case where the answer names the attribute but not the value.
- [ ] Run → FAIL (module missing). Implement `make_conversation(seed) -> Conversation` (messages, facts with index/attribute/value/question) and `score(answer, value) -> bool`. Run → PASS.

### Task 2: Runner (TDD with a fake LLM)

- [ ] Tests first with a fake LLM whose `complete` returns a fixed summary containing a marker and whose chat call records the messages it was sent:
  - `test_summary_strategy_sends_summary`: under `summary-6`, a probe's messages start with a `system` message beginning `SUMMARY_PREFIX`; under `window-6` none do.
  - `test_runs_are_isolated`: a second strategy/conversation starts with no summary.
  - `test_in_window_is_computed_at_probe_time`.
- [ ] Implement `run_strategy(conversation, strategy, llm, answer_fn) -> list[ProbeResult]` replaying turn by turn: load working memory, (probe turns) answer, append messages, `await compress_session(...)` for summary strategies. Record prompt tokens, latency, summarizer calls/seconds.
- [ ] `aggregate(results) -> dict` and `main()` (`--conversations`, `--windows`, `--llm_base`, `--llm_model`, `--out`).
- [ ] Mutation check: make `summary-N` pass `cache=None` → `test_summary_strategy_sends_summary` goes red; restore.

### Task 3: Run and report

- [ ] Start Ollama; smoke `--conversations 1 --windows 6`; then the full run in the background → `data/eval/memory_strategies.json`.
- [ ] ruff; full `pytest`; commit; PR with the results table.
