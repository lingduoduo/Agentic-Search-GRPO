# Architecture

[← Back to README](../README.md)

This guide explains the repository layout, agent families, request routing, and retrieval-grounded agent flow.

## Repository structure

```
src/
├── agents/                      # Agent loops (SearchAgentLoop, ToolAgentLoop, AgenticRAGLoop, …)
├── context/                     # Retrieval-grounded context & prompt builders
├── shared_configs/              # Shared configuration dataclasses
├── model/                       # Split by *when* the training happens
│   ├── serving.py               # ServerManager backends — neither half; runs whatever model exists
│   ├── pre_training/
│   │   └── intents/             # Nearest-canonical-example intent router (built offline, read-only at serve time)
│   └── post_training/           # Not part of the serving stack
│       ├── data.py              # Prompt/dataset construction, shared by SFT and RL
│       ├── reward.py            # Reward functions + the group-relative advantage primitive (torch-free)
│       ├── log_probs.py         # Response-token log-probs, shared by DPO and GRPO
│       ├── sft/                 # Supervised fine-tuning
│       ├── dpo/                 # Direct Preference Optimization (frozen reference, no reward model)
│       ├── ppo/                 # Clipped-surrogate *base algorithm layer* — no trainer, no critic, no GAE
│       ├── grpo/                # The GRPO stack built on ppo/ (training → generation → algorithms/core_algos)
│       ├── qlearning/           # Standalone tabular Q-learning demo (numpy, no torch)
│       └── eval/                # Benchmarks (Bamboogle, action-policy) + the unseen-user harness
└── internal/
    ├── access/                  # Access control & ACL helpers
    ├── auth/                    # Authentication & authorization
    ├── cache/                   # In-memory cache backend (chat session state)
    ├── chat/                    # Chat pipeline (loop, steps, citations, compression)
    ├── configs/                 # Environment-based configuration (AppSettings)
    ├── connectors/              # Connector data models (connector classes removed)
    ├── db/                      # SQLite store (AgenticSearchStore)
    ├── document_index/          # Document index (FAISS / BM25)
    ├── feature_flags/           # Feature-flag providers (env, PostHog, composite)
    ├── feedback/                # Retrieval feedback capture
    ├── file_store/              # In-memory chat file handling
    ├── hooks/                   # Outbound webhook execution
    ├── llm/                     # LLM provider integrations
    ├── mcp_server/              # MCP server (tools, resources, auth)
    ├── memory/                  # Conversation-memory store & curation
    ├── observability/           # Admin surface summary & health score
    ├── prompts/                 # Prompt templates
    ├── retrieval/               # Retrieval core: service, fusion, query transforms, routers
    ├── routing/                 # Routing layer: per-query router + 6 query constructors
    ├── search/                  # Search-vs-chat flow classification
    ├── tools/                   # Internal tool registry
    ├── utils/                   # License, encryption, telemetry utilities
    └── servers/
        ├── app.py               # Shared helpers for the standalone search servers
        ├── admin_surface/       # Admin summary endpoint
        ├── analytics/           # Usage analytics API
        ├── billing/             # Stripe billing proxy
        ├── error_handling/      # Shared exception handlers
        ├── enterprise_settings/ # Enterprise configuration endpoints
        ├── evals/               # Evaluation endpoints
        ├── features/            # Feature-flag endpoints
        ├── license/             # License validation & seat management
        ├── limits/              # Usage limit enforcement
        ├── manage/              # Administrative management endpoints
        ├── middleware/          # License enforcement, tier gate, tenant tracking
        ├── oauth/               # OAuth 2.0 connector authorization
        ├── query_and_chat/      # Search and chat endpoints
        ├── query_history/       # Query history & export
        ├── redis/               # Redis connection helpers
        ├── reporting/           # Usage report ZIP generation
        ├── retrieval/           # Dense/sparse/rerank server entry points
        ├── scim/                # SCIM 2.0 user & group provisioning
        ├── secondary_llm_flows/ # Auxiliary LLM flows (query expansion, …)
        ├── settings/            # Settings endpoints
        ├── tenants/             # Multi-tenant provisioning & management
        ├── token_rate_limits/   # Per-user token rate limiting
        ├── tools/               # Tool-engine HTTP surface (/tool/*)
        ├── user_group/          # Group management
        ├── users/               # User management
        ├── web/                 # FastAPI app assembly
        └── web_search/          # Web search servers (Google, SerpAPI, browser)
bin/                             # Shell helpers (eval, training data generation)
tests/                           # Unit and integration test suites
examples/                        # Runnable CLI examples
```

The FastAPI app is assembled in `src/internal/servers/web/app.py`. Every feature area is a self-contained router factory. `AgenticSearchStore` (SQLite) is the single persistence layer — no Postgres, Redis, or Celery required locally.

**Every public method of `AgenticSearchStore` is serialized on a re-entrant lock.**
A SQLite connection shared across concurrent sessions corrupted state without this;
the store is safe to call from several request handlers at once, but that safety
comes from the lock, not from SQLite. Anything long-running must therefore stay out
of a store call, and code that blocks (model generation, answer synthesis) belongs
on a worker thread — see `asyncio.to_thread` in the agent paths.

**The store's schema is versioned.** `PRAGMA user_version` records the schema
version, and `AgenticSearchStore.SCHEMA_VERSION = 1` is today's schema, which
is everything `_init_schema` creates. `schema_meta.min_reader_version` records
the oldest build that can still read the database.

To change the schema:
- add `_MIGRATIONS[n] = Migration(statements=(...), breaks_older_readers=...)`
  and bump `SCHEMA_VERSION` to `n`;
- each migration applies in its own transaction when a store opens.

Set `breaks_older_readers=True` only when an older build could no longer read
the result correctly, for example after a renamed or dropped column. Additive
changes, such as a new table or a nullable column, leave it `False`.

On rollback, an older build still opens a newer database when its version is
at least the min reader version, and writes nothing to the schema. Otherwise
it refuses with `SchemaVersionError`.

## Agent framework and control flow

The agent layer (`src/agents/`) behind every loop the [Web backend API](api-reference.md#web-backend-api) and [runnable agent examples](training-and-evaluation.md#agent-cli) drive.

### Agent taxonomy — two families

The repo has **two parallel agent designs**; "the agent framework" is the first, and the registry covers only it.

| Family | Members | `run()` contract | LLM access | Registry? | GRPO-trainable? |
|---|---|---|---|---|---|
| **Framework loops** (`AgentLoopBase`) | `plain_generation`, `single_turn_agent`, `search_agent` (the search/tool agents), `tool_agent` | `run(messages, sampling_params, *, on_turn) → AgentLoopOutput` | injected `server_manager` (token-level) | ✅ | ✅ |
| **RAG pipeline** | `AgenticRAGLoop` (web `chat_loop`) | `run(question, *, chat_history) → AgenticRAGResult` | `LLMClient` (chat-level) | ❌ | ❌ |
| **Retrieval pipelines** | `search_tool`, `hybrid_search`, `chat_once` | retrieve → answer functions | — | ❌ | ❌ |

**Tool agents and search agents are members of the framework** — siblings under `AgentLoopBase`, sharing the registry, the `LoopController` + components, and the `server_manager` model boundary. **Agentic RAG sits *beside* the framework, not inside it:** its constructor, `run()` signature, and return type diverge from `AgentLoopBase`, so registering it would break the `dict[str, type[AgentLoopBase]]` contract — it stays a deliberate non-registry loop. A dispatch layer (registry + `resolve_agent_name` + the web intent router) picks one target per request, treating all three families as interchangeable.

**Why they're kept separate (by design).** The framework loops are *token-level* because they're built for GRPO RL training (policy gradients need `prompt_ids`/`response_ids`). `AgenticRAGLoop` is a lighter *chat-level* serving pipeline (`LLMClient`, no tokenizer/`server_manager`) doing query decomposition + HyDE + grounded synthesis. Two simple designs for two purposes beat one contract forced onto both; the boundary is enforced (the registry rejects non-conforming loops) and documented in the [agent invocation consolidation design](superpowers/archive/specs/2026-06-25-agent-invocation-consolidation-design.md) so the families don't quietly drift together. Consolidation *is* feasible — `SearchAgentLoop` already does most of what `AgenticRAGLoop` does (sub-questions, iterative retrieval, evidence gating, citations); express agentic-RAG as a `SearchAgentLoop` config + a HyDE query-transform, bridged via the `ServerManager` protocol, and retire `AgenticRAGLoop`. It's deferred architectural-debt work, not a feature, so the families stay separate for now.

**Loop registry — one source of truth.** Agent loops register by name (`@register`) and are resolved through `get_registered_agent_loop(name)`; `resolve_agent_name` maps CLI/web aliases to the canonical loop. The registry covers the four `AgentLoopBase` loops below. `AgenticRAGLoop` (constructor + `run()` signature diverge from `AgentLoopBase`) and the retrieval pipelines (`search_tool` / `hybrid_search` / `chat_once`) are a distinct, non-registry category — see the [agent invocation consolidation design](superpowers/archive/specs/2026-06-25-agent-invocation-consolidation-design.md).

| Canonical loop | CLI `--mode` | Web `mode` | Purpose |
|---|---|---|---|
| `plain_generation` | `single` | — | one-shot generation, no retrieval |
| `single_turn_agent` | — | — | one-shot RAG |
| `search_agent` | `search` | `search_agent` | multi-turn retrieval QA |
| `tool_agent` | `tool` | `tool_agent` | generic function calling |

```bash
python -c "from src import list_registered_agent_loops, resolve_agent_name; \
print(list_registered_agent_loops()); print(resolve_agent_name('search'))"
  # → ['plain_generation', 'search_agent', 'single_turn_agent', 'tool_agent']
  # → search_agent
```

**LoopController — the search loop's two decisions.** `SearchAgentLoop` consults a stateless `LoopController` (`src/agents/components/loop_controller.py`) for *keep searching?* and *how to answer?*. Four **default-on** behaviors (tunable via `SearchAgentLoopConfig`):

- **Adaptive search budget** — `effective_search_limit` scales rounds by subquestion count: `max_search_limit + search_budget_per_subquestion·(n−1)`, capped at `max_search_limit_cap` (default `10`); single-subquestion runs are unchanged.
- **Plateau early-stop** — stops searching when a round's evidence gain `< evidence_plateau_min_gain` (default `0.05`) **and** evidence is already sufficient (`plateau_requires_sufficient`); never forces a thin answer.
- **Graceful dead-end answer** — at a dead-end / budget-exhaust with evidence collected, one bounded turn yields a best-effort answer instead of returning nothing (`force_answer_on_deadend`); never fabricates when no evidence exists.
- **Smarter answer-gating** — accept / reject (with targeted per-subquestion feedback) / force, decided by the controller.

Each surfaces an additive `metrics` key — `effective_search_limit`, `adaptive_budget_bonus`, `plateau_early_stop`, `forced_final_answer` — the last priced by `SearchRewardConfig.forced_final_answer_penalty` (`-0.05`, mutually exclusive with `answer_when_evidence_insufficient`). Existing reward presets stay byte-stable.

**`run()` control flow** is a linear, append-only turn loop. Each turn: `_generate_turn` (build prompt → generate → decode → parse actions) → action dispatch → `_apply_answer_gate` / `_handle_no_action`, which return a `TurnControl` directive (`CONTINUE` / `BREAK`) the loop acts on → the observation is appended as a `user` message. `_finalize_run_metrics` computes the derived/reward metrics once after the loop.

**Model backend.** Every loop receives an injected `server_manager` satisfying the `ServerManager` protocol (`src/model/serving.py`); `build_server_manager(tokenizer, server_url=…, model=…)` selects the OpenAI-compatible (remote) or in-process HuggingFace (local) backend — shared by the CLI and the web app.

**Drive it from the CLI** (control-flow knobs in `examples/run_agentic_search.py`):
```bash
python -m examples.run_agentic_search --mode search \
  --question "Compare dense and sparse retrieval" \
  --model meta-llama/Llama-3.1-8B-Instruct --vllm_url http://localhost:8080 \
  --search_url http://localhost:8001/retrieve \
  --max_search_limit 5 --max_turns 8 --max_answer_rejections 3
  # --no_evidence_gate disables the require-sufficient-evidence answer gate
```

**Drive it over the API / UI.** `POST /api/agent` picks the loop by `mode`; `POST /api/agent/stream` emits a `progress` SSE event after each turn via the `OnTurnCallback`, so plateau early-stops and forced dead-end answers appear live in the UI progress trace:
```bash
curl -sN -X POST http://localhost:7860/api/agent/stream \
  -H "Content-Type: application/json" \
  -d '{"query": "Compare dense and sparse retrieval", "mode": "search_agent", "top_k": 5}'
  # data: {"type": "progress", "turn": 1, "text": "search_routing_tool · 5 docs"}
  # data: {"type": "progress", "turn": 2, "text": "writing answer…"}
  # data: {"type": "answer", "text": "..."}
  # data: {"type": "done", "intent": "search", "citations": ["[D1]"], "documents": [...]}
```
See the [Web backend API](api-reference.md#web-backend-api) for the full request/response schema.

## Intent routing

The backend auto-classifies every query and dispatches to the right agent without any configuration:

| Intent | Agent loop | Trigger |
|--------|-----------|---------|
| `search` | Direct-first search pipeline (`SearchAgentLoop` only in explicit/escalated paths) | Query needs external retrieval or a bare entity lookup (e.g. `FAISS`) |
| `chat` | `AgenticRAGLoop` | Descriptive/conversational questions and generative asks — grounded synthesis |
| `tool` | `ToolAgentLoop` | Explicit tool use (`search_routing_tool`, custom tools) |

The router is `recognize_intent` (`src/internal/servers/web/intent/recognizer.py`), dispatched by `_run_auto_routed` in `src/internal/servers/web/app.py`. It returns strategy, clarification, and metadata together. Its precedence is explicit source, deterministic regex cues, optional margin-gated canonical similarity, deterministic LLM classifier, then rule-based fallback. Bare terms route to `search`; input with no signal at all triggers a clarification question instead of guessing (see [API request routing](request-routing.md#auto-router-decision-order)).

### End-to-end request flow

```text
offline index_builder (corpus.jsonl)
  → chunk + embed/index documents
  → searchable retrieval indexes

/api/agent or /api/agent/stream
  → query hook + session/history + access filters
  → explicit mode, or recognize_intent(chat | search | tool)
      → chat: AgenticRAGLoop, or the SearchPipeline composition without an LLM
      → tool: ToolAgentLoop, then grounded chat if tools/model are unavailable
      → search: internal retrieval → sufficiency gate
                → SerpAPI → browser-search service
                → deterministic no-evidence response
          escalation without a local model falls back to the same
          SearchPipeline composition:
              follow-up-aware retrieval query
              → candidate retrieval with ACL filters
              → deduplicate + optional rerank + MMR
              → grounded inference only with evidence
  → shared response finalization + persistence + hooks
  → JSON response or SSE answer/done events
```

The normalized `SearchPipeline` stages are internal boundaries, not services that require new deployment units. They adapt the existing retrieval clients, ranking helpers, and inference boundary while preserving `/api/agent`, `/api/agent/stream`, `/retrieve`, `/search`, and `/rerank`. No public API or schema was introduced. Optional reranker failure keeps the best pre-rerank order; retrieval or evidence failure never turns into an ungrounded model answer.

The local policy model is not the fallback for missing evidence on the default unfiltered auto-search path. Strong internal evidence returns directly; weak or empty internal evidence tries external search first. This avoids conflating a model's internal knowledge with retrieved evidence.

There are three independent routing layers: the web request strategy (`chat` / `search` / `tool`), the web source provider (`auto` / `retrieval` / web providers), and the internal retrieval backend router (sparse/dense/hybrid/etc.). See [API request routing](request-routing.md) for the detailed contract and [Retrieval](retrieval.md#routing-and-query-construction) for backend routing.

**RAG-Fusion in tool mode** — `search_routing_tool` aggregates results from all configured retrieval sources (local index, Google, SerpAPI) in a single call, deduplicates by URL, and returns a ranked list with `[D1]`/`[D2]` citation labels.

**SSE streaming with progress events** — All three agent paths emit SSE events:

| Event type | When emitted | Payload |
|------------|-------------|---------|
| `progress` | Each agent turn | `{type, turn, text}` |
| `claim` | Each verified claim, as it is verified | `{type, text}` |
| `trace` | Each control-flow event (Dev Console) | `{type, event}` |
| `approval_required` | An approval-gated tool wants to run | `{type, approval}` |
| `answer` | Answer token chunks | `{type, text}` |
| `done` | Stream complete | `{type, session_id, citations, documents, intent, tool_calls}` |
| `error` | Unhandled exception | `{type, detail}` |

The `on_turn` callback (`OnTurnCallback` in `src/agents/core/base.py`) is the hook that feeds per-turn events into the SSE queue from inside the agent loop.

**`claim` is what makes the grounded answer feel live.** The Assist path does not
stream tokens as the model emits them — its answer *is* the join of the claims it
has verified, so there is nothing to stream until a claim is verified. Each one is
emitted as it passes verification. `on_claim` is called from the answer-generation
worker thread (`generate_answer` runs under `asyncio.to_thread` so synthesis cannot
block the event loop), so it hops back via `loop.call_soon_threadsafe` before
touching the queue.

**The queue is bounded (`maxsize=100`) and drops rather than blocks.** A slow
consumer loses `claim` and `trace` events, never data: the terminal `answer` event
still carries the full text, and the drop counts are logged. Treat `claim` and
`trace` as best-effort progress, and `answer` / `done` as the contract.

## Agentic RAG

`chat_loop` is the web API name for `AgenticRAGLoop` — web modes are named by session behavior, not retrieval strategy. Valid modes: `search_tool`, `hybrid_search`, `chat_once`, `chat_loop`.

```bash
curl -X POST http://localhost:7860/api/agent \
  -H "Content-Type: application/json" \
  -d '{"query": "What is FAISS?", "mode": "chat_loop", "top_k": 5}'
```

Loop flow:

1. **Query enhancement** — decompose into sub-queries; generate HyDE hypothetical answer
2. **Hybrid+rerank retrieval** — retrieve per enhanced query; accumulate unique documents
3. **Sufficiency check** — LLM judges if context is enough; break or continue
4. **Follow-up generation** — LLM proposes targeted follow-up queries if insufficient
5. **Grounded synthesis** — answer from all accumulated evidence with inline citations
