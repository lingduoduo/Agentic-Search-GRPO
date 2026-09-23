# AgenticRAG rounds census — plan

Spec: `docs/superpowers/specs/2026-09-23-agentic-rag-rounds-census-design.md`

1. Tests first (`tests/unit/test_measure_agentic_rag_rounds.py`) for the pure
   trace → run-summary function: each stop reason, legacy `status: failed`
   traces, docs-added and post-round-1 timing, and the aggregate
   (failed-open excluded from `one_round_share`).
   → verify: tests fail (module missing).
2. `examples/measure_agentic_rag_rounds.py`: `summarize_trace`, `aggregate`,
   `--history` reader, `--questions` live replay, JSON report.
   → verify: tests pass; mutation-check that deleting the failed-open branch
   turns a test red.
3. Run it: history DB, then the 91 CHAT-labelled eval queries against the demo
   corpus with llama3.1:8b via Ollama. Write the report to `data/eval/`.
   → verify: counts add up to the number of runs.
4. ruff, full `pytest`, PR with the findings.
