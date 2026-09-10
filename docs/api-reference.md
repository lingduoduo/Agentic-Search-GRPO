# HTTP API reference

[← Back to README](../README.md)

This guide documents the local retrieval, web, chat/session, and health endpoints.

The retrieve → rank/rerank → grounded-inference composition is an internal implementation used by the chat and degraded branches of the endpoints documented here. Strong auto-search keeps its direct-first ranking, sufficiency gate, and provider fallback. These internals add no public endpoint and change no request or response schema. The offline `index_builder` populates the corpus before requests arrive; indexing is not performed by `/api/agent`.

## Retrieval server API

The retrieval server (`src/internal/servers/retrieval/server.py`, examples use `:8001`) exposes the retrieval core over HTTP. The demo server (`demo.py`, TF-IDF) only serves `POST /retrieve`; the full server below adds per-mode and admin endpoints.

**Health:**
```bash
curl -s http://localhost:8001/health
# → {"status": "ok", "backend": "local"}
```

**Hybrid search with metadata filters** (`POST /search` — sparse+dense → RRF → MMR → optional rerank):
```bash
curl -s -X POST http://localhost:8001/search \
  -H "Content-Type: application/json" \
  -d '{"query": "what is FAISS?", "top_k": 5, "filters": {"source": "arxiv"}}'
# → {"results": [{"doc_id": "...", "title": "...", "text": "...", "score": 0.71, ...}],
#    "retrieval_mode": "hybrid", "executed_queries": ["what is FAISS?"], "latency_ms": 41.2}
```

**Per-mode retrieval** (`/internal/search/*` — isolate one retrieval strategy, e.g. for evals):
```bash
# Sparse (BM25) only
curl -s -X POST http://localhost:8001/internal/search/sparse \
  -H "Content-Type: application/json" -d '{"query": "vector database", "top_k": 5}'
# → retrieval_mode: "sparse"

# Dense (embeddings) only
curl -s -X POST http://localhost:8001/internal/search/dense \
  -H "Content-Type: application/json" -d '{"query": "vector database", "top_k": 5}'
# → retrieval_mode: "dense"

# Hybrid with explicit fusion/MMR knobs
curl -s -X POST http://localhost:8001/internal/search/hybrid \
  -H "Content-Type: application/json" \
  -d '{"query": "vector database", "top_k": 5, "over_fetch": 4, "mmr_lambda": 0.5}'
# → retrieval_mode: "hybrid"

# GraphRAG (entity-graph re-ranking)
curl -s -X POST http://localhost:8001/internal/search/graph \
  -H "Content-Type: application/json" -d '{"query": "who founded OpenAI", "top_k": 5}'
# → retrieval_mode: "graph"
```

**Demo server** (`demo.py`, TF-IDF, no Java/embeddings — note `topk`):
```bash
curl -s -X POST http://localhost:8001/retrieve \
  -H "Content-Type: application/json" -d '{"query": "what is FAISS?", "topk": 5}'
```

**Standalone reranker** (`rerank.py` — batch interface: `queries` + per-query `documents` lists):
```bash
curl -s -X POST http://localhost:8001/rerank \
  -H "Content-Type: application/json" \
  -d '{"queries": ["what is FAISS?"],
       "documents": [[{"title": "FAISS", "content": "FAISS is a similarity search library"},
                      {"title": "Cats", "content": "Cats are mammals"}]],
       "rerank_topk": 2}'
```

**Inspect / hot-reload retrieval config** (admin):
```bash
curl -s http://localhost:8001/api/admin/retrieval/stats
curl -s -X PATCH http://localhost:8001/api/admin/retrieval/config \
  -H "Content-Type: application/json" \
  -d '{"rrf_k": 80, "mmr_lambda": 0.4, "nprobe": 96, "result_cache_ttl": 600}'
```

## Web backend API

The FastAPI web backend (`src/internal/servers/web/app.py`, `:7860`) drives the UI and agent loops.

**Run the intent-routed agent** (`POST /api/agent`) — auto-routes search / chat / tool; `response.intent` reflects the chosen path:
```bash
curl -s -X POST http://localhost:7860/api/agent \
  -H "Content-Type: application/json" \
  -d '{"query": "Compare dense and sparse retrieval", "mode": "chat_loop", "top_k": 5}'
# → {"answer": "...", "intent": "chat", "citations": ["[D1]"], "documents": [...], "session_id": "..."}
```

`response.intent` is `"search" | "chat" | "tool" | "clarify"` and is the single field that drives the [intent-adaptive layout](frontend.md#ui-features) (`App.tsx` maps it to a `.results-layout` class). When the auto-router has no signal at all, it returns `intent="clarify"` instead of guessing; no agent runs, and the response carries a `clarification` object — `{"question": "...", "options": [{"route": "chat", "label": "..."}, ...]}` — for the caller to present. Resend the same query with the request field `route` set to `chat`, `search`, or `tool` to skip the router and dispatch through the normal auto path (any other value returns `422`). Read just the intent field:
```bash
curl -s -X POST http://localhost:7860/api/agent \
  -H "Content-Type: application/json" \
  -d '{"query": "find the onboarding checklist", "top_k": 5}' \
  | python -c "import sys, json; print(json.load(sys.stdin)['intent'])"
# → search
```

**Stream the same over SSE** (`POST /api/agent/stream`) — emits one `progress` event after each agent turn (via the `on_turn` callback), a `claim` event for each claim as it is verified, then `answer`, then `done` (which carries `intent`, `citations`, and `documents`; the frontend feeds `intent` to `setIntent`). The non-streaming `/api/agent` is unchanged:
```bash
curl -sN -X POST http://localhost:7860/api/agent/stream \
  -H "Content-Type: application/json" \
  -d '{"query": "Compare dense and sparse retrieval", "top_k": 5}'
# Server-Sent Events (one JSON object per `data:` line):
# data: {"type": "progress", "turn": 1, "text": "search_routing_tool · 5 docs"}
# data: {"type": "progress", "turn": 2, "text": "writing answer…"}
# data: {"type": "claim",    "text": "Dense retrieval embeds the query into a vector space."}
# data: {"type": "answer",   "text": "Dense retrieval embeds the query …"}
# data: {"type": "done",     "session_id": "...", "intent": "chat", "citations": ["[D1]"], "documents": [...]}
```
On failure the stream yields `data: {"type": "error", "detail": "..."}` instead of `done`, which `streamAgent` surfaces as the error banner.

The grounded (Assist) path emits **`claim`** rather than streaming raw tokens,
because its answer *is* the join of the claims it has verified — there is nothing
to stream until a claim passes verification. A Dev Console client additionally
receives `trace` events, and an approval-gated tool produces `approval_required`.

**`claim` and `trace` are best-effort; `answer` and `done` are the contract.** The
SSE queue is bounded and drops these two rather than blocking a slow consumer. No
data is lost when it does — the terminal `answer` event still carries the full
text — so a client must not assume it has seen every `claim`.

**Sessions:**
```bash
curl -s -X POST http://localhost:7860/api/sessions \
  -H "Content-Type: application/json" -d '{"title": "Search session"}'
curl -s http://localhost:7860/api/sessions/{session_id}
```

**A session can only be read or continued by the caller who owns it.** Reading
`/api/sessions/{id}`, continuing an existing session through `/api/agent`, and the
`/chat/*` session endpoints all gate on `caller_may_use_session`
(`src/internal/servers/_auth.py`).

**A caller who does not own the session gets `404`, not `403`** — deliberately. A
`403` would confirm the id exists, which is the one bit an id-guessing caller does
not already have. Do not treat `404` from these routes as proof the session was
deleted; it also means "not yours".

**Submit retrieval feedback** (`POST /api/feedback` — drives the feedback-GRPO training signal):
```bash
curl -s -X POST http://localhost:7860/api/feedback \
  -H "Content-Type: application/json" \
  -d '{"session_id": "sess-123", "signal": "thumbs_down", "target": "retrieval"}'
# → {"ok": true}
```

`target` is `retrieval` (the sources were wrong), `generation` (the answer was
wrong given the sources) or `overall` (default). The answer panel's thumbs bar
posts it; `GET /api/admin/evals/summary` and the training loader read it back
apart (`by_target`, `metadata["human_signal_target"]`).

**Process telemetry** (`GET /api/admin/metrics`, admin only): the per-route
latency window, the same recent `/api/agent` requests split into their
retrieval share (ms, docs, cache-hit rate), their generation share (answer
synthesis only: ms, prompt/completion tokens) and their auxiliary LLM calls
(query transforms, sufficiency checks, intent recognition — kept apart so
they cannot inflate "generation"), and the feedback summary by target.
Recorded in every web process; this is the production-reachable read of it.
Each assistant turn also carries its own split as `metadata.stage_metrics`.
Only `/api/agent` opens the scope; the `/search`, `/chat` and `/tool` direct
surfaces and the `web_search` tool (which does not go through
`SearchClient`) are not counted, and the LiteLLM backend is not hooked.

### Request paths & dispatch

Both `POST /api/agent` and `/api/agent/stream` run the same dispatcher
(`_run_agent_impl`). The optional `mode` field selects the path; **when `mode` is
omitted (the default, and what the bundled UI always sends) the request goes
through the auto-router, which may also ask the user to clarify.** Every path
returns the same shape `(answer, citations, documents, intent, …)` with
`intent ∈ {search, chat, tool, clarify}`; `clarify` is reachable only through
the auto-router and carries no agent output.

All three multi-turn paths are **conversation-aware** — `search_agent`,
`tool_agent`, and `chat_loop` thread bounded prior session turns into the loop.
Search mode uses the tighter `SEARCH_AGENT_HISTORY_MESSAGES` limit because it
also carries long retrieval observations.

For the shared search pipeline, prior session messages are bounded and used to resolve follow-up retrieval queries while the original user query remains the inference question. Candidate retrieval then preserves access filters, ranking deduplicates and optionally reranks/diversifies the candidates, and inference runs only with ranked evidence. Empty evidence produces a deterministic no-results response; a retrieval outage produces a deterministic unavailable-sources response. Finalization persists the answer, citations, documents, and normalized `pipeline_stages` metadata for both JSON and SSE delivery.

| `mode` | Path / loop | `intent` | Requires |
|---|---|---|---|
| _(omitted)_ | auto-router → chat/search/tool strategy; search is direct-first | varies | configured services |
| `search_agent` | `SearchAgentLoop` (multi-turn search) | `search` | local model (else `400`) |
| `tool_agent` | `ToolAgentLoop` (OpenAPI/MCP tools) | `tool`/`search`/`chat` | local model (else `400`) |
| `chat_loop` | `AgenticRAGLoop` (decompose + HyDE) | `chat` | LLM client |
| `hybrid_search` | internal + web fan-out, MMR-merged | `search` | — |
| `search_tool` | raw retrieval, no synthesis | `search` | — |

**Auto-router:** a non-`auto` source forces `search`; otherwise deterministic
regex cues, an optional learned intent model, an optional LLM classifier, and a
rule-based heuristic choose `chat`, `search`, or `tool` whenever any of them has
a signal. Bare terms such as `RAG` and `GRPO` are deterministic searches. When
none of those steps has a signal at all, the router returns `intent="clarify"`
and asks the user instead of guessing (see [Auto-router decision order](request-routing.md#auto-router-decision-order)).

For an unfiltered auto-routed search, `source_provider=auto` means **internal
retrieval → sufficiency gate → SerpAPI → configured browser-search service**.
The first provider with sufficient evidence wins. If none returns evidence, the
response keeps `intent="search"`, has empty citations/documents, and reports
`No results found for: <query>` or that every source is unreachable. It does not
substitute a local-model answer. Signing in does not change this sequence — it
narrows what each provider may return. See
[Access control](request-routing.md#access-control).

`hook_metadata` carries `mode`, `route`, optional `route_degraded`, and search
details such as `search_mode`, `external_provider`, `tier`, and `top_score`.
The complete field-level contract, explicit-source behavior, and worked examples
are in [API request routing](request-routing.md).

## Chat and session API

Chat session management and search-flow routing live on the web backend (`:7860`) under the `/chat`, `/search`, and `/query` routers (`src/internal/servers/query_and_chat/`). The streamed send-message flow itself is `POST /api/agent` / `/api/agent/stream` above; these endpoints manage the sessions and feedback around it.

**Chat sessions** (`/chat`):
```bash
# Create a session
curl -s -X POST http://localhost:7860/chat/create-chat-session \
  -H "Content-Type: application/json" -d '{"title": "Onboarding questions"}'
# → {"chat_session_id": "..."}

# List the user's sessions / fetch one with its messages
curl -s http://localhost:7860/chat/get-user-chat-sessions
curl -s http://localhost:7860/chat/get-chat-session/{session_id}

# Rename / delete
curl -s -X PUT http://localhost:7860/chat/rename-chat-session \
  -H "Content-Type: application/json" \
  -d '{"chat_session_id": "...", "name": "Renamed"}'
curl -s -X DELETE http://localhost:7860/chat/delete-chat-session/{session_id}
```

**Per-message feedback** (`POST /chat/create-chat-message-feedback`):
```bash
curl -s -X POST http://localhost:7860/chat/create-chat-message-feedback \
  -H "Content-Type: application/json" \
  -d '{"chat_message_id": "...", "is_positive": true, "feedback_text": "spot on"}'
```

**Search-flow classification** (`POST /search/search-flow-classification` — keyword-search vs chat routing):
```bash
curl -s -X POST http://localhost:7860/search/search-flow-classification \
  -H "Content-Type: application/json" -d '{"user_query": "find the Q3 onboarding deck"}'
# → {"is_search_flow": true}
```

**Direct search message** (`POST /search/send-search-message` — optional query expansion, streamable):
```bash
curl -s -X POST http://localhost:7860/search/send-search-message \
  -H "Content-Type: application/json" \
  -d '{"search_query": "vector database benchmarks", "run_query_expansion": true, "num_hits": 10, "stream": false}'
```

**Search history** (`GET /search/search-history`):
```bash
curl -s http://localhost:7860/search/search-history
```

`GET /query/standard-answer` exists but is an Enterprise-gated stub — it returns `501` ("Standard Answers is an Enterprise feature … not available in this deployment") in the open-source build.

## API health checks

Web backend: `http://localhost:7860` · Retrieval server: `http://localhost:8001`

**Generate a dev JWT** (required for admin endpoints):

```bash
export TOKEN=$(bin/gen_dev_token.sh)   # or: source bin/gen_dev_token.sh
```

**Core**

```bash
curl -s http://localhost:7860/health                  # web server
curl -s http://localhost:8001/health                  # retrieval server
curl -s http://localhost:7860/settings                # tier / license status (no auth)
```

**Search & chat**

```bash
curl -s -X POST http://localhost:7860/api/agent \
  -H "Content-Type: application/json" \
  -d '{"query": "What is FAISS?", "mode": "search_tool"}'

curl -s http://localhost:7860/api/sessions/SESSION_ID -H "Authorization: Bearer $TOKEN"

curl -s -X POST http://localhost:8001/retrieve \
  -H "Content-Type: application/json" -d '{"query": "dense retrieval", "topk": 3}'
```

**Admin — analytics, billing, reporting**

```bash
curl -s "http://localhost:7860/analytics/query?start=2024-01-01&end=2025-12-31" \
  -H "Authorization: Bearer $TOKEN"
curl -s http://localhost:7860/admin/billing/billing-information -H "Authorization: Bearer $TOKEN"
curl -s http://localhost:7860/admin/usage-report                -H "Authorization: Bearer $TOKEN"
```

**Admin — hooks, rate limits, web search**

```bash
curl -s http://localhost:7860/admin/hooks/specs              -H "Authorization: Bearer $TOKEN"
curl -s http://localhost:7860/admin/hooks                    -H "Authorization: Bearer $TOKEN"
curl -s http://localhost:7860/admin/token-rate-limits/users  -H "Authorization: Bearer $TOKEN"
curl -s http://localhost:7860/admin/web-search/search-providers -H "Authorization: Bearer $TOKEN"
```

**Admin — license**

```bash
curl -s http://localhost:7860/license       -H "Authorization: Bearer $TOKEN"
curl -s http://localhost:7860/license/seats -H "Authorization: Bearer $TOKEN"
```

**SCIM** (uses SCIM bearer token, not a JWT)

```bash
curl -s http://localhost:7860/scim/v2/ServiceProviderConfig  # no auth
curl -s http://localhost:7860/scim/v2/Users  -H "Authorization: Bearer $SCIM_TOKEN"
curl -s http://localhost:7860/scim/v2/Groups -H "Authorization: Bearer $SCIM_TOKEN"
```
