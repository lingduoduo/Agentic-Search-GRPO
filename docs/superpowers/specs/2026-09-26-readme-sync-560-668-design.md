# README sync: behavioural delta of #560–#668

**Goal.** Correct or add every README claim that PRs merged since the last full
sync (#559) changed. #602 (search domains) and #667 (diagrams) already did part
of this. No reorganisation and no style pass.

**In scope:** README.md only. The guides under `docs/` are out of scope.

**Changes (each verified against the code on `main` at 4f6894da):**

| Claim | Source |
|---|---|
| Architecture diagram: retrieval `:8000` → `:8001` (the local default in demo.py, hybrid.py and app_configs; only compose uses 8000) | pre-existing |
| Diagram API node: `WS /api/agent/ws` | #611 |
| Install: three companion requirements files | #628 |
| Configure: `AGENTIC_SEARCH_TIMEOUTS_PATH` + timeouts guide | #645 |
| Run locally: hybrid alternative; browser server on :8003, needs `playwright-cli`, host only | #632 |
| Verify: `/ready` (200/503 + checks), opt-in `/metrics`, alert rules | #648, #661, #664 |
| New "Run with Docker": compose, non-root, GHCR sha tags, deploy/rollback link | #622–#631, #662–#668 |
| Chat engine: token-budgeted memory, summarisation defaults; model-unavailable degradation | #653, #654, #656, #658 |
| Tool engine: full JSON Schema validation; WebSocket channel; public-data GET retry | #606, #611, #637 |
| Docs index: timeouts, deploy, metrics, authentication | — |
