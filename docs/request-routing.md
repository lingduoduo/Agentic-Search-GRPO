# API request routing

[← Back to README](../README.md)

This guide is the source of truth for how the web API turns an agent request into a chat, search, or tool execution. It covers `POST /api/agent` and `POST /api/agent/stream`; both endpoints share the same dispatcher in `src/internal/servers/web/app.py`.

The pipeline described here is query-time orchestration over indexes produced offline by the `index_builder`. It reuses the existing `/api/agent`, `/api/agent/stream`, `/retrieve`, `/search`, and `/rerank` contracts; no public endpoint or schema was added.

## Routing at a glance

```text
request
  ├─ mode is set ───────────────→ run that explicit mode
  └─ mode is omitted
       └─ recognize_intent ──────────→ chat | search | tool | clarify
                                   │
                                   ├─ chat → grounded AgenticRAGLoop
                                   ├─ tool → ToolAgentLoop, or grounded chat fallback
                                   ├─ clarify → ask the user which route they meant; no agent runs
                                   └─ search
                                        1. internal retrieval
                                        2. sufficiency gate
                                        3. SerpAPI
                                        4. browser-search service
                                        5. deterministic no-evidence response
```

Identity does not select a branch here. Every caller — anonymous included —
carries an ACL, and it narrows what each branch may return rather than diverting
the request to a different pipeline. See [Access control](#access-control).

Three separate decisions are involved:

1. **Request strategy** chooses `chat`, `search`, or `tool`.
2. **Source-provider selection** chooses the internal corpus or a web provider.
3. **Retrieval-backend routing** inside `RetrievalService` chooses sparse, dense, hybrid, graph, and query-transformation behavior.

Changing one axis does not directly change the others. For example, `source_provider=retrieval` forces the request strategy to `search`, while `auto` permits intent recognition. The internal retrieval service still selects its own configured retrieval backend.

Filter-aware and degraded search branches use the shared internal `SearchPipeline` composition:

```text
bounded session history
  → resolve follow-up retrieval query
  → retrieve normalized candidates from the selected existing provider
  → deduplicate → optional reranker → MMR/truncation
  → grounded inference when ranked evidence exists
  → persist answer + citations + documents + stage metadata
```

The original query remains the answer question; only retrieval uses the resolved follow-up query. Internal access filters are preserved. If optional reranking fails, the pre-rerank candidate order is retained. If retrieval yields no evidence, model inference is skipped and a deterministic status is returned.

Strong auto-search does not necessarily enter that composition or rewrite its retrieval query. Its existing direct-first path queries internal retrieval with the original request, applies direct ranking plus the sufficiency gate, and then tries SerpAPI and the browser-search service when internal evidence is weak or empty. The provider order below describes that distinct path.

## Request fields

The JSON body uses `AgentExperienceRequest`:

| Field | Default | Meaning |
|---|---:|---|
| `query` | required | Non-empty user request. Query-processing hooks may rewrite it before routing. |
| `session_id` | new session | Existing conversation whose prior messages become bounded history. |
| `user_id` | authenticated user | Development-only identity fallback when no authenticated user is present. It selects which ACL the request carries, not whether it carries one. |
| `search_url` | server setting | Retrieval URL override. Honored only when `AGENTIC_SEARCH_ALLOW_CLIENT_RETRIEVAL_URL=true`. |
| `top_k` | `5` | Requested result count, from 1 through 20. |
| `source_provider` | `auto` | Source policy: `auto`, `retrieval`, `serpapi`, `google`, `browser`, or `all`. |
| `route` | omitted | User-selected `chat`, `search`, or `tool` route; bypasses recognition in auto mode. |
| `mode` | omitted | Explicit dispatch override: `search_tool`, `hybrid_search`, `chat_once`, `chat_loop`, `search_agent`, or `tool_agent`. |

The backend, not the browser client, normally owns service URLs. In production, keep client retrieval URL overrides disabled.

## Explicit modes

Setting `mode` bypasses the three-way intent router.

| Mode | Execution path | Reported intent | Main requirement |
|---|---|---|---|
| `search_tool` | One direct retrieval/search call; deterministic result rendering | `search` | Selected source service |
| `hybrid_search` | Query expansion when an LLM exists, provider search, merge/rerank, deterministic rendering | `search` | Selected source services; LLM is optional |
| `chat_once` | One retrieval-grounded answer call | `chat` | LLM client for synthesis |
| `chat_loop` | `AgenticRAGLoop`: decomposition, HyDE, iterative retrieval, synthesis | `chat` | LLM client |
| `search_agent` | Local policy-model `SearchAgentLoop` | normally `search` | `SEARCH_AGENT_MODEL` or remote model server and tokenizer |
| `tool_agent` | Local policy-model `ToolAgentLoop` with registered tools | varies by tool result | Local or remote policy model and tokenizer |

Explicit policy-agent modes run model **inference** during an API request. They do not update weights, run GRPO, or perform any training step. Training is a separate offline workflow described in [Training and evaluation](training-and-evaluation.md).

## Auto-router decision order

When `mode` is omitted and no user-selected `route` is supplied, `recognize_intent` returns a `RouteDecision` containing the strategy, optional clarification, and production metadata. The dispatcher copies that metadata into `hook_metadata`. The cascade is:

1. A non-`auto` `source_provider` is an explicit search request.
2. Deterministic rules select a route:
   - action commands such as “send”, “deploy”, or “create a ticket” → `tool`;
   - lookup verbs such as “find”, “search for”, or “retrieve” → `search`;
   - bare one-to-three-word terms such as `RAG`, `GRPO`, or `vector database` → `search`;
   - standalone greetings and thanks, such as “hi!” and “thank you” → `chat`;
   - conversational or generative starts normally → `chat`. A current-information cue such as “latest” defers a chat-form question to the next step.
3. If an index is configured, canonical-example similarity selects the highest-scoring route when its gap over the runner-up reaches `AGENTIC_SEARCH_INTENT_MIN_ROUTE_MARGIN` (default `0.010`). Each route scores as its top-8 mean cosine similarity. Margin is the only abstention gate; absolute confidence is diagnostic. See [Training and evaluation](training-and-evaluation.md#intent-routing-by-nearest-canonical-example).
4. An available LLM classifies at temperature 0. A completion must contain exactly one distinct supported whole-word label. A single-label explanation is accepted, but conflicting labels such as “not chat; search” are rejected as `unexpected`.
5. If no LLM is available or its call raises, the last-resort heuristic checks tool, search, and bare-lookup cues. No signal defaults to chat with clarification. An empty or unusable classifier completion also requests clarification. Set `AGENTIC_SEARCH_ROUTE_CLARIFICATION=false` to return the chat default without the question.

Explicit modes, user-selected routes, explicit providers, and deterministic rules take precedence over similarity. A served similarity prediction skips the LLM; an abstention defers to it. The model chooses an execution family, not a specific tool, and cannot bypass authorization. Module labels and the composite-request flag are diagnostics only; no multi-step planner acts on them.

`AGENTIC_SEARCH_INTENT_SHADOW_MODE=true` records similarity predictions for requests that reach the model step, then discards them and continues to the classifier or heuristic. Explicit overrides and deterministic rules still bypass model evaluation. The recognizer resolves omitted settings once so similarity, shadow mode, and clarification use the same configuration.

Every recognition result includes `route_mechanism`. When similarity produces a decision, metadata also contains `route_predicted_intent`, `route_confidence`, `route_abstained`, `route_model_latency_ms`, `route_modules`, and `route_composite`. A model abstention adds `route_fallback_reason="margin_below_threshold"`; shadow evaluation instead adds `route_fallback_reason="shadow_mode"` and the separate `route_shadow_intent`, `route_shadow_abstained`, and `route_shadow_fallback_reason` fields.

Request captures receive exactly one deciding `intent` stage and at most one `intent_model · evaluation` stage. Model capture includes cosine confidence, margin, modules, composite flag, abstention reason, and latency. Classifier capture retains only a supported label or the `empty`/`unexpected` sentinel, never arbitrary completion text or the query. A missing, unreadable, or incompatible index disables similarity safely and leaves the existing fallbacks available.

`route_mechanism` uses the following vocabulary:

| Mechanism | Meaning |
|---|---|
| `explicit_source` | An explicit non-default source provider forced search |
| `rules` | Deterministic high-precision cues decided |
| `model` | The canonical-example similarity match was confident |
| `classifier` | The LLM classifier returned a usable label |
| `heuristic_default` | A fallback cue decided, or clarification was disabled and chat defaulted |
| `clarify` | No signal at all; the user was asked |
| `user_selected` | The user chose the route |

The selected strategy is recorded as `hook_metadata.route`. Capability fallback occurs after classification and may be recorded as `hook_metadata.route_degraded`; the surfaced `intent` describes what actually ran and can differ from the selected route.

When no step in the cascade has a signal, the router asks instead of guessing.
The response carries `intent: "clarify"` and a `clarification` object holding a
question and one option per route; no agent runs. Sending the same query back
with `route` set to `chat`, `search`, or `tool` skips the router and dispatches
through the normal auto path, so the selected agent and its degradation
behavior are identical. Set `AGENTIC_SEARCH_ROUTE_CLARIFICATION=false` to
restore the previous behavior of always choosing a route.

## Auto-routed search provider order

For a request with `source_provider=auto`, search is evidence-first and sequential:

### 1. Internal retrieval

The backend queries the configured internal retrieval URL first. Error documents are excluded from evidence.

### 2. Sufficiency gate

The top internal result is accepted immediately when any tier succeeds:

- normalized query exactly matches the top document title;
- title is within one edit and semantic cosine exceeds `SEARCH_DIRECT_COS_MIN`;
- query-to-passage semantic cosine exceeds `SEARCH_DIRECT_COS_MIN`.

Accepted internal evidence returns `hook_metadata.search_mode="direct"`, plus the gate `tier` and `top_score`. The answer is deterministic result rendering; no answer-generation model is invoked.

### 3. SerpAPI

Weak, empty, or unavailable internal retrieval falls through to SerpAPI using the original query. This occurs before any local policy-model answer path. A successful web result returns `search_mode="external_fallback"` and `external_provider="serpapi"`.

### 4. Browser-search service

If SerpAPI provides no evidence and `browser_search_url` is configured, the backend calls that HTTP service. The service is implemented with `playwright-cli`; the web request handler does not launch Playwright directly. Successful results return `external_provider="browser"`.

### 5. No evidence

If at least one provider was reachable but none found evidence, the API returns:

```text
No results found for: <query>
```

If every attempted provider was unreachable, it returns:

```text
No sources are reachable right now. Please try again shortly.
```

Both cases use `intent="search"`, `search_mode="external_empty"`, and empty `citations` and `documents`. The local model does not replace missing evidence with an internal-knowledge answer.

## Access control

**Signing in narrows results; it does not change the engine.** Anonymous and
authenticated callers take the same route, run the same stages, and differ only
in which documents survive. Requests no longer divert to a separate filter-aware
pipeline when an identity is present — that divert fired on *every* signed-in
request, because an authenticated ACL is never empty, and it silently swapped
one query for an expanded five-query fan-out.

`resolve_capabilities(user, store)` maps the caller to a `RequestCapabilities`
carrying `access_acl`, the memory preamble, and whether user-scoped tools are
offered. **Anonymous is an identity, not the absence of one:** it carries
`["public"]`, so no caller can express "unfiltered" by presenting nothing.

Enforcement is applied where documents are read, not left to the retrieval
backend:

| Path | Where the ACL is applied |
|---|---|
| Direct / degraded search | `_enforce_access` on the returned documents |
| `SearchAgentLoop` | inside the loop, before the documents enter the model's context |
| Tool agent's corpus `search` | in the tool, which is built per request and carries the caller's filters |
| `SearchClientRetrievalStage` (a `SearchPipeline` stage; no route is wired to it today) | in the stage, on the candidates the server returned |

Filters are also sent to the retrieval server, and all three bundled servers
(`demo.py`, `hybrid.py`, and `server.py`'s `RetrievalService`) honour them — but
that is defense in depth. A third-party backend may ignore the field, so the web
layer enforces regardless. Documents that declare no ACL are public. External web
providers receive no internal ACL object; web results carry no ACL to filter on.

The [serving cache](retrieval.md#serving-cache) keys retrieval rows by the
serialised filters, so a cache hit never crosses an ACL boundary, and the
enforcement above runs on hits and misses alike.

Enforcement is a post-filter: a restricted document still consumes retrieval
bandwidth and can displace an accessible one from `top_k` before being dropped.

> Anonymous callers read every document that declares no ACL. A corpus whose
> documents carry no ACL metadata is therefore fully readable anonymously.

## Other source-provider values

The sequential internal → SerpAPI → browser contract above applies specifically to auto-routed search with `source_provider=auto`.

- `retrieval` limits source selection to the internal retrieval service.
- `serpapi`, `google`, and `browser` request an explicit web source and force the `search` strategy.
- `all` is the explicit multi-provider policy used by the hybrid/direct provider helpers.
- `hybrid_search` has its own fan-out, query-expansion, merge, and rerank behavior; do not infer its provider ordering from the auto-routed direct-first path.

An explicit source or explicit mode is an operator override, so its model requirements and fallback behavior follow that path rather than the default auto-search contract.

## Response contract

Both endpoints ultimately produce `AgentExperienceResponse` data:

| Field | Meaning |
|---|---|
| `session_id` | Conversation identifier, created when absent. |
| `answer` | Generated grounded answer or deterministic search status/results. |
| `citations` | Citation labels corresponding to returned evidence. Empty when no evidence exists. |
| `documents` | Normalized source documents with ID, citation, title, content, URL, score, and metadata. |
| `messages` | Persisted conversation messages after the request. |
| `intent` | Actual surfaced path: `search`, `chat`, `tool`, or `clarify` when asking a routing question. |
| `hook_metadata` | Mode, selected route, degradation reason, search mode, provider, and hook data when applicable. |
| `tool_calls` | Structured tool execution records. |
| `control_flow_trace` | Ordered component/action/status events for agent loops. |

Common routing metadata in `hook_metadata`:

| Key | Example | Meaning |
|---|---|---|
| `mode` | `auto` | Dispatcher mode used for the request. |
| `route` | `search` | Strategy selected by the auto-router. |
| `route_degraded` | `no_llm` | Required capability was absent and dispatch used a fallback. |
| `search_mode` | `direct`, `external_fallback`, `external_empty`, `escalated` | Search execution branch. |
| `external_provider` | `serpapi`, `browser` | External provider that supplied evidence. |
| `tier` | `exact`, `fuzzy`, `semantic` | Internal sufficiency tier. |

Persisted assistant-message metadata also contains a normalized `pipeline_stages` summary. It records the retrieval query/provider/candidate count, ranking operations/evidence count/reranker degradation, inference mode/model, and final citation/document IDs. These diagnostics do not add fields to `AgentExperienceResponse`; the same summary is available to request capture/inspection, and JSON and SSE use the same finalization path.

## Streaming events

`POST /api/agent/stream` runs the same dispatcher and sends Server-Sent Events:

1. zero or more `progress` events as agent turns complete;
2. zero or more `claim` events, one per claim as the grounded path verifies it;
3. zero or more `trace` events when control-flow tracing is on;
4. optional `approval_required` events for approval-gated actions;
5. one `answer` event;
6. one `done` event containing the final session, intent, citations, documents, route metadata, tool calls, and trace.

`claim` and `trace` are best-effort — the bounded SSE queue drops them rather than
blocking a slow consumer — while `answer` and `done` are guaranteed.

Failures produce an `error` event instead of `done`. Streaming changes delivery, not routing behavior.

## Why `RAG` and `GRPO` can look different

Both bare queries deterministically route to `search`. Their results can differ because of evidence coverage:

- If the internal corpus contains a strong document titled `RAG`, the sufficiency gate returns internal retrieval immediately.
- If the corpus has no strong `GRPO` result, the same route continues to SerpAPI and then the browser-search service.
- If neither external provider finds evidence, the response reports no results; it does not ask the local model to answer `GRPO` from memory.

This is serving-time routing and inference. It is unrelated to GRPO training, even when the repository also contains GRPO trainers and trainable policy-agent loops.

## Configuration dependencies

- Internal retrieval: `AGENTIC_SEARCH_RETRIEVAL_URL` or the web server's retrieval setting.
- SerpAPI: `SERP_API_KEY` or `SERPAPI_API_KEY` and the SerpAPI integration.
- Browser fallback: `SearchExperienceSettings.browser_search_url` and a running browser-search service. The default `from_app_settings()` construction does not currently populate this URL, so deployments that want browser fallback must wire it into app construction.
- Local policy modes: `SEARCH_AGENT_MODEL` or `SEARCH_AGENT_SERVER_URL`.
- Provider-backed chat and classification: `GEN_AI_MODEL_PROVIDER`, `GEN_AI_MODEL_VERSION`, and provider credentials.
- Optional similarity route: `AGENTIC_SEARCH_INTENT_INDEX_PATH`, `AGENTIC_SEARCH_INTENT_MIN_ROUTE_MARGIN` (default `0.010`), `AGENTIC_SEARCH_INTENT_MIN_MODULE_SCORE` (default `0.8215`, diagnostics only), and `AGENTIC_SEARCH_INTENT_TOP_K` (default `8`).
- Sufficiency threshold: `SEARCH_DIRECT_COS_MIN`.

See [Configuration](configuration.md) for setup details.

## Implementation ownership

| Concern | Code |
|---|---|
| API models and shared dispatcher | `src/internal/servers/web/app.py` |
| Public intent recognition | `src/internal/servers/web/intent/__init__.py` (`recognize_intent`) |
| Cascade, classifier, and routing metadata | `src/internal/servers/web/intent/recognizer.py` |
| Shared vocabulary and scoring defaults | `src/shared_configs/intent.py` |
| Shared decision types and deterministic rules | `src/internal/servers/web/intent/types.py`, `rules.py` |
| Lazy similarity adapter | `src/internal/servers/web/intent/similarity.py` |
| Offline index and evaluation | `src/model/pre_training/intents/` |
| Request capture and inspector metadata | `src/internal/servers/web/request_capture.py` |
| Search, tool, and RAG loops | `src/agents/` |
| Web provider services | `src/internal/servers/web_search/` |
| Internal retrieval backend routing | `src/internal/retrieval/` |
