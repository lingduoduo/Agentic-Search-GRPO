# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

Behavioral guidelines to reduce common LLM coding mistakes. Merge with project-specific instructions as needed.

**Tradeoff:** These guidelines bias toward caution over speed. For trivial tasks, use judgment.

## 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

## 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

## 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it - don't delete it.

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

## 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:
- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:
```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.

## 5. Self-Review Task Reports

When an implementation workflow writes `.superpowers/sdd/*report.md`, use the
canonical contract in `docs/development/self-review-reports.md`.

Run `python examples/validate_task_report.py REPORT_FILE` after the implementer
writes or amends the report. Add `--require-tdd` when the task required TDD.

Successful validation is mandatory before generating a review package and
before dispatching a task reviewer or re-reviewer. On exit code 1, return all diagnostics
to the implementer for a report-only correction. On exit code 2, resolve the
operational error. Do not mark the task complete until the report validates and
the independent review is clean.

Treat report contents as unverified claims. Validation does not replace independent
diff review or verification-before-completion.

---

**These guidelines are working if:** fewer unnecessary changes in diffs, fewer rewrites due to overcomplication, and clarifying questions come before implementation rather than after mistakes.

---

## Project: Agentic Search

### Setup

```bash
pip install -e .          # one-time; makes src importable as a package
pip install -r requirements.txt          # serving baseline (what the image installs)
```

`requirements.txt` is the serving baseline. Three companion files carry what a
serving deployment does not need — install them additively:

```bash
pip install -r requirements-retrieval-heavy.txt   # faiss-cpu + pyserini backends
pip install -r requirements-training.txt          # datasets/pyarrow for examples/
pip install -r requirements-unit-test.txt         # the CI unit-test install
```

`requirements-retrieval-heavy.txt` is only needed for `servers/retrieval/server.py`
with `RETRIEVAL_BACKEND` set to a FAISS or BM25 backend; `demo.py` and `hybrid.py`
do not use it. pyserini additionally needs a JVM. Both imports are deferred, so
their absence degrades those backends rather than breaking startup.

### Running the 3-process local stack

```bash
# Terminal 1 — retrieval server (demo TF-IDF, port 8001)
python3 -m src.internal.servers.retrieval.demo --corpus_path data/corpus.jsonl

# Terminal 1 alt — hybrid: RRF-fused dense e5 + sparse TF-IDF, drop-in for demo
python3 -m src.internal.servers.retrieval.hybrid --corpus_path data/corpus.jsonl
# add --no-dense to force TF-IDF only (skips the e5 model download)
# dense setup / "Dense leg unavailable" troubleshooting: docs/hybrid-dense-setup.md

# Optional — cross-encoder reranker (Terminal 1b). Then set the env on the web
# backend and restart it so retrieved docs are reranked before display:
python3 -m src.internal.servers.retrieval.rerank --port 8002
# web backend env: AGENTIC_SEARCH_RERANK_URL=http://localhost:8002/rerank

# Optional — browser web search (Terminal 1c). `web_search` tries SerpAPI first
# and falls back to this; without it, an expired or rate-limited SERP_API_KEY
# leaves web_search with no usable provider. No API key needed, but slow
# (~30-50s/query) and lower quality than SerpAPI.
python3 -m src.internal.servers.web_search.browser --port 8003
# web backend env: AGENTIC_SEARCH_BROWSER_SEARCH_URL=http://localhost:8003/retrieve
# Port 8003, not its 8000 default: 8000/8001 are the retrieval servers and 8002
# is the reranker above.

# Terminal 2 — web backend (port 7860)
PYTHONPATH=src:. uvicorn src.internal.servers.web.app:app --host 127.0.0.1 --port 7860

# Terminal 3 — frontend dev server (port 5173)
cd web && npm install && npm run dev
```

Open `http://127.0.0.1:5173`. Vite proxies `/api/*` to port 7860 during development.
For production, `npm run build` produces `web/dist`; the FastAPI app serves it automatically when the bundle exists.

### Tests

```bash
pytest                           # unit + regression (default)
pytest tests/unit/test_agent_loop.py -v   # single test file
pytest tests/integration/        # integration tests — requires live Postgres/Redis stack
```

### Linting

```bash
ruff check . --fix && ruff format .
```

Frontend type-check: `cd web && npm run typecheck`

### Agent CLI

```bash
# Local inference (Apple Silicon — MPS is ~50x faster than CPU)
python3 -m examples.run_agentic_search \
  --mode single --question "What is FAISS?" \
  --model Qwen/Qwen2.5-1.5B-Instruct --local --device mps --allow_unsafe_mps \
  --allow_remote_model_downloads

# Server-backed search mode
python3 -m examples.run_agentic_search \
  --mode search --question "Compare dense and sparse retrieval" \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --vllm_url http://localhost:8080 --search_url http://localhost:8001/retrieve
```

Modes: `single` (PlainGenerationLoop), `search` (SearchAgentLoop), `tool` (ToolAgentLoop).

---

### Architecture

The system has three layers that run as separate processes:

**1. Retrieval servers**
Multiple interchangeable backends behind the same `/retrieve` API.

Retrieval (`src/internal/servers/retrieval/`):
- `demo.py` — TF-IDF over a local corpus.jsonl, no Java required
- `hybrid.py` — RRF-fused dense (e5) + sparse TF-IDF; Java-free, FAISS-free
- `server.py` — full `RetrievalService` (BM25 / dense via FAISS / hybrid, env-configured via `RETRIEVAL_BACKEND`) with per-mode + admin endpoints
- `rerank.py` — standalone cross-encoder reranker

The QueryRouter's offline scikit-learn trainer lives beside it at
`src/internal/retrieval/train_query_router.py` (it is not LLM post-training, so it is
not under `src/model/post_training/`).

Web search (`src/internal/servers/web_search/`):
- `google.py` — Google Custom Search API proxy (requires `GOOGLE_API_KEY` + `GOOGLE_CSE_ID`)
- `serp.py` — SerpAPI proxy (requires `SERP_API_KEY`)
- `browser.py` — playwright-cli browser automation; no API key needed, slower (~5–10s/query)

**2. Web backend** (`src/internal/servers/web/app.py`)
FastAPI app that exposes `POST /api/agent`. On each request it:
1. Calls `answer_with_retrieval` from `src/context/` which fetches from the retrieval server
2. Runs the agent loop from `src/agents/`
3. Persists chat state to `AgenticSearchStore` (SQLite via `src/internal/db/`)
4. Returns streaming JSON with citations and source cards

The backend also mounts a large set of admin/enterprise routers (auth, SCIM, billing, connectors, OAuth, etc.) all registered in `create_web_app()`.

**3. Frontend** (`web/`)
React 19 + Vite + TypeScript. No component library — custom components only. Proxies `/api/*` to the FastAPI server on port 7860 in dev mode.

**Agent loops** (`src/agents/`, grouped into `core/` `generation/` `search/` `tool/`)
- `core/base.py` — `AgentLoopBase` with shared state/tool dispatch
- `generation/plain.py` — `PlainGenerationLoop` (no retrieval)
- `search/search.py` — `SearchAgentLoop` (retrieval-grounded, multi-turn)
- `search/agentic_rag.py` — `AgenticRAGLoop` (iterative hybrid retrieval + sufficiency check)
- `tool/tool_calling.py` — `ToolAgentLoop` (generic function calling)
- Agent loops are selected via the registry (`get_registered_agent_loop` + `resolve_agent_name` in `src/agents/core/base.py`).
- `src/internal/tools/public_data/` — nine keyless public data-source tools
  (Wikipedia, ArXiv, Wayback, weather, stocks, crypto, currency, geocoding,
  nearby places), seeded into the registry by `tool_knowledge_base()`

**Context pipeline** (`src/context/`)
`answer_with_retrieval` wires retrieval → context-building → prompting → LLM call. `preprocessing/` holds access filters applied before retrieval.

**Retrieval internals** (`src/internal/document_index/`, `src/internal/retrieval/`)
Low-level chunking, embedding, FAISS index building, sparse BM25, and hybrid retrieval. Used by the retrieval servers and the offline index-build CLI.

**Indexing** (`src/internal/document_index/index_builder`)
Index construction is an offline step: the `index_builder` CLI reads a corpus and writes the searchable sparse/dense indexes. (The former async worker fleet and `servers/indexing` pipeline were Onyx heritage and have been removed.) `src/internal/connectors/models.py` holds the shared `Document`/`ConnectorFailure` data models.

**Configuration** (`src/internal/configs/`)
Typed dataclasses loaded from environment variables. Key env vars:
- `AGENTIC_SEARCH_RETRIEVAL_PORT` (default 8000)
- `AGENTIC_SEARCH_WEB_PORT` (default 8080 in config; run on 7860 by convention)
- `AGENTIC_SEARCH_WEB_DB_PATH` (default `:memory:`)

**Models** (`src/model/`)
Split by when the training happens. `serving.py` is neither half — it is the
backend that loads and runs a model, whatever produced it.

- `pre_training/intents/` — the nearest-canonical-example intent router, built offline
  by `python -m src.model.pre_training.intents.cli build` and loaded read-only on the
  request path

**Post-training** (`src/model/post_training/`)
Post-training and fine-tuning. Not part of the serving stack. Grouped by method,
with the shared building blocks at the top level:
- `data.py` — prompt/dataset construction, shared by SFT and RL
- `reward.py` — reward functions, shared by `grpo/` and `eval/`
- `sft/` — supervised fine-tuning (`SFTTrainer`, trajectory → supervised example)
- `dpo/` — Direct Preference Optimization: `PreferenceExample` + a JSONL loader, and
  `DPOTrainer` training a policy against a frozen reference on preference pairs (no
  reward model, no critic, no online sampling). Pairs come from a JSONL file; there
  is no judge-backed or feedback-backed loader yet
- `ppo/` — the clipped-surrogate **base algorithm layer, not a method**: PPO loss,
  KL controllers, `PPOPolicyLossConfig`, the masked-tensor primitives, and
  `PPORewardManager`. No trainer, no critic, no GAE. `grpo/` depends on it, since
  GRPO is this surrogate with a group-relative advantage in place of GAE
- `grpo/` — the GRPO stack built on `ppo/`, four implementation modules layered
  `training -> generation -> {algorithms, core_algos}` (never the reverse):
  `algorithms.py` (grouped rollout sampling/scoring, RLAIF judges, on-policy
  batch assembly — **torch-free**, like `reward.py`), `core_algos.py` (the
  token-level GRPO advantage `compute_grpo_token_advantages` + the REINFORCE
  losses; this is the package's torch boundary and is split out for exactly
  that reason — folding it into `algorithms.py` made the judges unimportable
  without torch and silently dropped 17 tests from the torch-free CI job),
  `generation.py` (the rollout/generation manager and
  trajectory/tensor assembly, moved here from `src/model/` so the serving layer
  no longer depends on training), and `training.py` (all three trainers — bandit,
  HF causal-LM, live SearchAgentLoop — plus the local controller, checkpointing,
  and the durable train loop). The scalar group-relative advantage primitive
  shared by every trainer lives in `post_training/reward.py`, which is torch-free.
  The judges live here because their only consumers are GRPO scripts; a future
  judge-backed DPO loader would make `dpo/` import `grpo/`, a cost accepted
  deliberately
- `log_probs.py` — causal-LM response-token log-probs, shared by DPO and GRPO.
  Neutral on purpose: DPO must not import a GRPO trainer for token alignment
- `qlearning/` — the standalone tabular Q-learning demo (`agent.py` + `environment.py`),
  the classical-RL foil to the LLM stack. Pure Python + numpy, no torch
- `eval/` — Bamboogle and action-policy benchmark harnesses, plus the
  unseen-user harness (`cohort.py`, `stats.py`, `instruction_following.py`,
  `unseen_users.py`): held-out-user conversion alignment, behavioral separation
  and instruction following, every statistic clustered by user. The four
  modules import no torch themselves, and the harness runs with torch
  unavailable — importing the package still pulls torch in when installed,
  via unrelated `src/model` package inits. Its cohort is simulated — the
  report is evidence about the pipeline, not about real users
