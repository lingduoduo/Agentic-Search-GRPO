# Configuration

[← Back to README](../README.md)

This guide collects provider, search-agent, application, retrieval, reranking, query-transformation, and routing configuration in one reference.

Copy `.env.example` to `.env`; it is loaded automatically through `python-dotenv`.

See [Authentication](authentication.md) for account state, session ownership,
anonymous-memory behavior and startup route enforcement.

## Model and web-search providers

```bash
# LLM provider (required for agent loops)
GEN_AI_MODEL_PROVIDER=openai       # openai | anthropic | ollama | litellm
GEN_AI_MODEL_VERSION=gpt-4o-mini
GEN_AI_API_KEY=...
GEN_AI_API_BASE=...                # optional override (e.g. http://localhost:11434/v1)

# Web search (pick one or more)
GOOGLE_API_KEY=...
GOOGLE_CSE_ID=...
SERP_API_KEY=...

# Optional
JAVA_HOME=/path/to/java            # for BM25 / pyserini
```

## Local search-agent policy model

Set `SEARCH_AGENT_MODEL` before starting the web API to enable the UI's “Search Agent (Local Model)” mode:

```bash
# 8 GB RAM
SEARCH_AGENT_MODEL=Qwen/Qwen2.5-0.5B-Instruct PYTHONPATH=src:. uvicorn src.internal.servers.web.app:app --host 127.0.0.1 --port 7860

# 16 GB RAM (better quality)
SEARCH_AGENT_MODEL=Qwen/Qwen2.5-1.5B-Instruct PYTHONPATH=src:. uvicorn src.internal.servers.web.app:app --host 127.0.0.1 --port 7860
```

`bin/run_web_stack.sh` reads `SEARCH_AGENT_MODEL` from `.env` and starts all three processes in one command (~30–60 seconds for the first response on MPS).

`SEARCH_AGENT_MODEL` is the policy model used by explicit `search_agent` and `tool_agent` modes. It must emit multi-turn `<search>` / `<answer>` tags. Small models (≤3B, such as `Qwen2.5-0.5B`) often cannot, which can produce an empty answer with zero sources even when routing and retrieval are correct. A capable 7B+ policy model generally needs at least 16 GB; otherwise use the provider-backed `chat_loop` or deterministic `hybrid_search` path. The default auto-routed search path does not need this model: it tries internal retrieval, SerpAPI, and browser search before returning a no-evidence response.

## Web request routing

Routing configuration spans separate capabilities:

| Setting | Role in `/api/agent` routing |
|---|---|
| `AGENTIC_SEARCH_RETRIEVAL_URL` | First source for auto-routed search and internal grounding for chat paths |
| `SERP_API_KEY` / `SERPAPI_API_KEY` | Enables the SerpAPI fallback after weak or empty internal evidence |
| `SearchExperienceSettings.browser_search_url` | Enables the HTTP browser-search fallback after SerpAPI; run `src.internal.servers.web_search.browser` separately and wire its `/retrieve` URL into app construction |
| `SEARCH_AGENT_MODEL` / `SEARCH_AGENT_SERVER_URL` | Enables explicit local/remote policy-agent modes; not required for default auto-search |
| `GEN_AI_MODEL_PROVIDER`, `GEN_AI_MODEL_VERSION`, provider key | Enables grounded chat synthesis and the classifier for ambiguous routes |
| `AGENTIC_SEARCH_INTENT_INDEX_PATH` | Directory holding a canonical-example index (`index.npz`) built by `src.model.pre_training.intents.cli`; unset by default, which disables the similarity route |
| `AGENTIC_SEARCH_INTENT_SHADOW_MODE` | Score requests that reach the similarity step after explicit overrides and deterministic rules and record what the router *would* have decided, without acting on it; `false` by default. Requires an index path. See [promotion](training-and-evaluation.md#promotion-what-would-have-to-be-true) |
| `AGENTIC_SEARCH_INTENT_MIN_ROUTE_MARGIN` | Minimum cosine-similarity gap between the top and runner-up route; defaults to `0.010`, selected on the tuning slice jointly with `top_k` (margins compress as `k` rises, so the two cannot be tuned apart) |
| `AGENTIC_SEARCH_INTENT_MIN_MODULE_SCORE` | Minimum cosine similarity for a module label to be emitted alongside the route; defaults to `0.8215`, derived on the tuning slice at the serving `top_k`. It was `0.45` until #520 — below every score `e5-small-v2` produces, so the gate could not fire at all |
| `AGENTIC_SEARCH_INTENT_TOP_K` | Neighbors averaged per route; defaults to `8`. Re-selected on the wider evaluation instrument; it was `15` on the previous one and an unswept `3` before that — see [the instrument section](training-and-evaluation.md#the-instrument-widened--and-what-that-revealed-about-top_k) |
| `AGENTIC_SEARCH_ROUTE_CLARIFICATION` | Ask the user which route was meant when no step in the cascade has a signal; `true` by default. Set `false` to always choose a route, as before. |
| `AGENTIC_SEARCH_SEARCH_DIRECT_COS_MIN` | Semantic threshold for accepting internal evidence without external fallback |
| `AGENTIC_SEARCH_ALLOW_CLIENT_RETRIEVAL_URL` | Allows a request body to override the server-owned retrieval URL; development only |
| `AGENTIC_SEARCH_SEARCH_CACHE_TTL` | Seconds a retrieval row, web-provider page or rerank score stays in the web app's process-local serving cache; defaults to `300`, `0` disables it. No Redis involved. See [Serving cache](retrieval.md#serving-cache) |
| `AGENTIC_SEARCH_SEARCH_CACHE_STALE_SECONDS` | Seconds past the TTL an expired serving-cache entry is kept to answer when the live call fails (a web provider's error, timeout or open circuit; a retrieval-server error). Such results carry `metadata.stale: true`. Defaults to `3600`, `0` disables the fallback; a negative value is rejected at startup. See [Serving cache](retrieval.md#serving-cache) |

`source_provider=auto` applies the sequential provider order to auto-routed search. Signing in narrows what that search returns rather than selecting a different path. Explicit modes and explicit providers retain their own execution contracts. See [API request routing](request-routing.md).

Intent route values and scoring defaults are defined in
`src/shared_configs/intent.py`. AppSettings, environment fallbacks, the default
configuration map, and the offline scorer consume those definitions. The
example CLI also honors `AGENTIC_SEARCH_INTENT_TOP_K` when `--intent_index` is set.

## Application and authentication

See [Workload identity](workload-identity.md) for JWT subject mappings, production
safeguards, Kubernetes token renewal, and Redis IAM configuration. Existing local
tokens are minted with a one-hour default lifetime; callers must renew them.

| Env var | Default | Description |
|---------|---------|-------------|
| `AGENTIC_SEARCH_AUTH_SECRET` | `agentic-search-dev-secret` | Local JWT signing secret; a non-default secret is required in production |
| `AGENTIC_SEARCH_ENVIRONMENT` | `development` | Set to `production` to reject unsafe signing secrets, admin bypass, and local tokens without expiration |
| `AGENTIC_SEARCH_DEV_ADMIN` | `false` | Development-only admin bypass; forbidden in production |
| `AGENTIC_SEARCH_JWT_PUBLIC_KEY_URL` | — | Operator-configured HTTPS JWKS endpoint for workload authentication |
| `AGENTIC_SEARCH_WORKLOAD_ISSUER` | — | Exact trusted HTTPS issuer; workload authentication is opt-in |
| `AGENTIC_SEARCH_WORKLOAD_AUDIENCE` | — | Expected application audience for workload tokens |
| `AGENTIC_SEARCH_WORKLOAD_SUBJECTS` | `{}` | JSON subject-to-local-identity mapping; see the workload identity guide |
| `AGENTIC_SEARCH_SUPER_USERS` | `[]` | JSON list of admin user IDs or emails |
| `AGENTIC_SEARCH_WEB_DB_PATH` | `:memory:` | SQLite path (`:memory:` for ephemeral) |
| `AGENTIC_SEARCH_METRICS_ENABLED` | `false` | Mount `GET /metrics`, the Prometheus exposition of `agentic_search_http_requests_total`, `agentic_search_http_request_duration_seconds` and `agentic_search_stage_duration_seconds`. The route is **unauthenticated**: restrict it at the network layer (ingress rule, internal-only port). Recording is always on; the flag only controls exposure. Metric definitions, coverage and PromQL for QPS, error, decision-round and tool-timeout rates: [operational metrics](observability-metrics.md). Single-process only (no `PROMETHEUS_MULTIPROC_DIR`); the image runs one uvicorn worker. Separately, `GET /ready` (always mounted, also public) returns 503 unless the store answers `SELECT 1` and the retrieval server's `/health` returns 2xx within `readiness.probe_timeout_seconds` ([timeouts](configuration/timeouts.md)); the compose healthchecks keep using the liveness `GET /health`, so pointing one at `/ready` is an operator choice |
| `AGENTIC_SEARCH_IMAGE` | `agentic-search:local` | **Compose only** (`docker/docker-compose.yml`), not read by the app. The image the `retrieval` and `web` services run. Unset, `up --build` builds and tags `agentic-search:local` from source; set it to a published `ghcr.io/<owner>/agentic-search:sha-<sha>` tag and run `up --no-build` to deploy or roll back ([deploy](deploy.md)) |
| `AGENTIC_SEARCH_MCP_USER_SCOPED` | — | Comma-separated MCP tool names to mark `user_scoped`, so they are withheld from callers with no user |
| `AGENTIC_SEARCH_MEMORY_REQUIRE_AUTH` | `false` | Refuse anonymous memory callers (`401`) instead of pooling them into the shared `default_user` bucket. Governs both `/api/memory/*` and the MCP memory tools |
| `AGENTIC_SEARCH_MEMORY_COMPRESSION` | unset | Summarize turns that fall off the history tail in the background with the configured LLM client and prepend the summary as a system message on the next turn. Three states: **unset** (default) turns it on for `/api/agent` only, which already sends the conversation to that client; **`1`/`true`/`yes`** turns it on for every surface; any other value (`0`/`false`/`no`, …) turns it off everywhere. On `/chat` and `/tool` the answer comes from the local model, so enabling it there sends their older turns to the remote provider for the first time. Inert without an LLM client. Each call summarizes at most ~3000 estimated tokens of the oldest unsummarized turns, so a long backlog drains over several turns. State lives in the cache backend (`CACHE_BACKEND`), in-memory by default |
| `AGENTIC_SEARCH_MEMORY_HISTORY_TOKENS` | `2500` | Estimated-token budget of the session history every surface (`/api/agent`, `/chat`, `/tool`) sends with a turn, on top of the 40-message cap. Estimate: `ceil(characters / 4) + 4` per message. The newest message is always kept, even alone over budget. Older turns are dropped (and summarized when compression is on for that surface). Must be a positive integer; anything else fails at startup |
| `AGENTIC_SEARCH_MEMORY_AUTO_CURATE` | `false` | On each compression event, also curate the summarized turns into the signed-in user's long-term memories (the same `curate` loop the CLI and `/api/memory/curate` run, scoped to that span). Does nothing on a surface where `AGENTIC_SEARCH_MEMORY_COMPRESSION` leaves compression off. Anonymous sessions are never curated. Each overflow costs one curation loop (up to 6 tool-calling turns) against the configured remote LLM client. The curation loop can update and delete existing memories, including ones saved by hand, and judges them from the summarized span alone (not the retained tail) |
| `AGENTIC_SEARCH_MEMORY_SEMANTIC` | `false` | The memory preamble is relevance-aware above the 20-memory cap regardless of this flag (half the slots go to the memories most relevant to the query, half to the most recent); this flag only selects the scorer: e5 embeddings when on, content-word overlap when off. Use the e5 encoder for memory search instead of token overlap: `/api/memory/search`, the MCP `search_memories` tool, and the per-request memory preamble, which above the 20-memory cap fills half its slots with the memories most relevant to the current query. The web app loads the model once at startup; `AGENTIC_SEARCH_MEMORY_EMBED_DEVICE` (default `cpu`) picks the device. With the encoder on, every Assist request above the cap embeds all of that user's memories (no cache); cost grows with memory count. Concurrent requests share one model; torch forward passes on a shared eval-mode model are accepted |
| `AGENTIC_SEARCH_RETRIEVAL_URL` | `http://localhost:8001/retrieve` | Retrieval server URL |
| `AGENTIC_SEARCH_CLOUD_DATA_PLANE_URL` | — | Cloud data plane for billing proxy |
| `AGENTIC_SEARCH_LICENSE_ENFORCEMENT_ENABLED` | `false` | Enable license gating |
| `AGENTIC_SEARCH_DATA_DIR` | `~/.local/share/agentic_search` | License file directory |
| `WEB_DOMAIN` | `http://localhost:8080` | External URL for OAuth redirects |
| `GEN_AI_MODEL_PROVIDER` | `openai` | LLM provider (openai, anthropic, ollama, etc.) |
| `GEN_AI_MODEL_VERSION` | `gpt-4o-mini` | Model name / version |
| `GEN_AI_API_KEY` | — | Provider API key |
| `GEN_AI_API_BASE` | — | Override base URL (e.g. `http://localhost:11434/v1`) |
| `AGENTIC_SEARCH_INTENT_INDEX_PATH` | — | Directory holding a canonical-example index (`index.npz`) built by `src.model.pre_training.intents.cli`; unset keeps similarity routing disabled. **The loaded index is cached by resolved path and never invalidated, and a failed load is cached too** — so build the index *before* starting the process, and restart after changing this variable. Neither direction takes effect at runtime |
| `AGENTIC_SEARCH_INTENT_SHADOW_MODE` | `false` | Score requests that reach the similarity step after explicit overrides and deterministic rules and record `route_shadow_intent` / `route_shadow_abstained` / `route_shadow_fallback_reason` in session telemetry, then discard the decision and fall through to the classifier. Lets the promotion question be measured on real traffic while the router stays dark |
| `AGENTIC_SEARCH_INTENT_MIN_ROUTE_MARGIN` | `0.010` | Minimum cosine-similarity gap between the top and runner-up route; must be finite and between `0.0` and `1.0`. Selected on the tuning slice **jointly with `top_k`** — raising `k` compresses margins, so a margin judged at another `k` is judged at the wrong threshold for itself. Serves 120 of 201 test-slice queries at `0.9667` served accuracy |
| `AGENTIC_SEARCH_INTENT_MIN_MODULE_SCORE` | `0.8215` | Minimum cosine similarity for a module label to be emitted alongside the route; must be finite and between `0.0` and `1.0`. Derived on the tuning slice from a grid **computed at the serving `top_k`**, because module scores fall as `k` rises — a constant derived at one `k` silently collapses emission at another. Diagnostics only; it can never change the route |
| `AGENTIC_SEARCH_INTENT_TOP_K` | `8` | Neighbors averaged per route; must be a positive integer. Chosen on the tuning slice jointly with `min_margin`, under a rule fixed in advance. **Treat it as under-determined rather than optimal**: the same rule picked `15` on the previous, smaller instrument, and `8`/`15`/`25` sit within `0.014` of each other on tuning accuracy — the tie-break toward lower `k` decides among them. See [Training and evaluation](training-and-evaluation.md#top_k-chosen-on-the-split) |
| `AGENTIC_SEARCH_ROUTE_CLARIFICATION` | `true` | Ask the user which route was meant when no step in the cascade has a signal; set `false` to always choose a route, as before |
| `OAUTH_SLACK_CLIENT_ID` | — | Slack OAuth app client ID |
| `OAUTH_CONFLUENCE_CLOUD_CLIENT_ID` | — | Confluence OAuth app client ID |
| `OAUTH_GOOGLE_DRIVE_CLIENT_ID` | — | Google Drive OAuth app client ID |

Set `AGENTIC_SEARCH_INTENT_INDEX_PATH` to a directory containing an
`index.npz` built by `src.model.pre_training.intents.cli` from curated canonical examples.
Leave it unset to keep similarity routing disabled. There is no
promotion gate on this path the way there was for the earlier trained
checkpoint: a missing, unreadable, or encoder-mismatched index simply
disables the similarity route (logged once, then cached) and every request
falls through to the existing LLM/rule fallbacks.

## Neural reranking

| Env var | Default | Description |
|---------|---------|-------------|
| `RERANKER_PROVIDER` | — | `local` or `cohere`; omit to disable neural reranking in `RetrievalService` |
| `RERANKER_MODEL` | `BAAI/bge-reranker-v2-m3` | Cross-encoder model for local reranking |
| `RERANKER_BATCH_SIZE` | `32` | Batch size for local cross-encoder |
| `RERANKER_DEVICE` | `cpu` | Device for local reranker (`cpu`, `mps`, `cuda`) |
| `RERANKER_TOP_K` | same as search `top_k` | Cap returned results after reranking |
| `COHERE_API_KEY` | — | Cohere API key (required when `RERANKER_PROVIDER=cohere`) |
| `RERANKER_ASYNC` | `false` | Wrap reranker in `AsyncReranker` (thread-pool offload) |
| `RERANKER_TIMEOUT_MS` | `500` | Per-query scorer timeout for `AsyncReranker` |
| `RERANKER_MAX_WORKERS` | `4` | Thread pool size for `AsyncReranker` |
| `RERANKER_CACHE_REDIS_URL` | — | Enable `CachedReranker`; set to a Redis URL |
| `RERANKER_CACHE_TTL_SECONDS` | `300` | TTL for cached reranker scores |
| `RERANKER_MAX_TOKENS` | `512` | `PassageTruncator` token limit before scoring (0 = disabled) |
| `RERANKER_TWO_STAGE` | `false` | Enable `TwoStageReranker` (fast pre-filter → heavy scorer) |
| `RERANKER_PRE_FILTER_TOP_N` | `50` | Candidates passed to the heavy scorer in two-stage mode |
| `RERANKER_FAST_MODEL` | inherits `RERANKER_MODEL` | Fast-stage model name in two-stage mode |
| `RERANKER_OVER_FETCH_MULTIPLIER` | `2.0` | Retrieval over-fetch ratio when a reranker is active |

## Retrieval and optimization

| Env var | Default | Description |
|---------|---------|-------------|
| `QUERY_EXPANSION_ENABLED` | `false` | Enable acronym expansion in the BM25 leg, using the bundled default table |
| `ACRONYM_PATH` | — | JSON file of `{"ACRONYM": "expansion"}`. Overrides both the corpus-derived and bundled tables |
| `ACRONYM_CORPUS_PATH` | `BM25_CORPUS_PATH`, else `data/corpus.jsonl` | Corpus scanned for `long form (ABBR)` glosses to build the acronym table |
| `SPELL_CORRECTION_ENABLED` | `false` | Enable `symspellpy` spell correction in BM25 leg. Requires the `symspellpy` package |
| `EXPANSION_MAX_TERMS` | `3` | Max acronym expansions added per query, to prevent BM25 query bloat |
| `RESULT_CACHE_REDIS_URL` | — | Enable `ResultCache`; set to a Redis URL |
| `RESULT_CACHE_TTL` | `300` | TTL in seconds for cached full search responses |

**Measured: expansion did not help.** On the TF-IDF backend over two BEIR
benchmarks, corpus-derived acronym expansion *lowered* NDCG@10 in every
configuration tested — and by more on the queries where it actually fires,
which is the signature of a real effect rather than noise:

| dataset | queries | fires | NDCG@10 base | expanded | delta | delta on firing subset |
|---|---|---|---|---|---|---|
| scifact | 300 | 222 | 0.5811 | 0.5473 | −0.0338 | −0.0456 |
| nfcorpus | 323 | 34 | 0.2936 | 0.2879 | −0.0057 | −0.0538 |

Lowering `EXPANSION_MAX_TERMS` to 1 reduces the harm but does not remove it.
Recall@10 and MRR move the same way. `QUERY_EXPANSION_ENABLED` is off by
default and should stay off unless you have measured a gain on your own corpus.
See #601.

**Query expansion reads the corpus first.** With `QUERY_EXPANSION_ENABLED=true`
the acronym table is built from the corpus at `ACRONYM_CORPUS_PATH` by reading
the glosses it already contains — `ionizing radiation (IR)` in a medical corpus
gives `IR` → ionizing radiation there, rather than this repository's
`information retrieval`. A corpus that defines nothing falls back to the
bundled table at `src/internal/retrieval/acronyms.json`, and setting
`ACRONYM_PATH` overrides both. The startup log names which source was used and
how many acronyms it holds. `EXPANSION_MAX_TERMS` caps how many expansions land
on any one query.

**Not environment variables.** Three knobs that look like settings are CLI
flags instead. The FAISS index type is `--faiss_type` on the index-build CLI
(default `Flat`; it takes any FAISS `index_factory` string, so `HNSW32` and
IVF variants both work), HNSW search breadth is `--hnsw_ef_search` on the same
CLI, and the evaluation latency gate is `--slo-ms`, with `--qt-slo-ms` for the
query-transform leg, on `eval_runner`.

## Query transformation

| Env var | Default | Description |
|---------|---------|-------------|
| `QT_DECOMPOSE` | `false` | Enable query decomposition in `QueryTransformPipeline` |
| `QT_HYDE` | `false` | Enable HyDE (hypothetical document embedding) |
| `QT_STEP_BACK` | `false` | Enable step-back query rephrasing |
| `QT_KEYWORDS` | `false` | Enable keyword expansion for BM25 variants |
| `QT_CONSTRUCT_FILTERS` | `false` | Enable NL → metadata filter extraction |
| `QT_REWRITE` | `false` | Enable canonical query rewrite (`QueryEnhancer.rewrite`); 7th router label |
| `QT_MAX_VARIANTS` | `5` | Max parallel retrieval variants when any `QT_*` is enabled |
| `QT_ASYNC` | `false` | Run the leaf's transform LLM calls in parallel (`AsyncQueryTransformPipeline`) |
| `QT_TRANSFORM_TIMEOUT_MS` | `400` | Per-transform timeout; on exceed that field degrades to its default |
| `QT_MAX_WORKERS` | `5` | Thread-pool size for `AsyncQueryTransformPipeline` |
| `QT_CACHE_REDIS_URL` | — | Enable `CachedQueryTransformPipeline`; set to a Redis URL |
| `QT_CACHE_TTL_SECONDS` | `600` | TTL for cached transform bundles |
| `QT_MULTI_QUERY` | `false` | Enable `MultiQueryGenerator` (N paraphrased query variants) |
| `QT_MULTI_QUERY_N` | `3` | Number of paraphrases generated per query |
| `QT_FUSION_WEIGHTED` | `false` | Use `variant_weighted_rrf_fuse` (original query weighted highest) |
| `QT_SEMANTIC_DEDUP` | `false` | Drop near-duplicate variants before retrieval (needs a backend `embed()`) |
| `QT_SEMANTIC_DEDUP_THRESHOLD` | `0.95` | Cosine cutoff for variant dedup |
| `QT_ROUTER` | `false` | Per-query routing of transforms (`QueryRouter` + heuristic fallback) |
| `QT_ROUTER_MODEL_PATH` | — | Serialized scikit-learn router artifact; heuristic used when unset/missing |
| `QT_CONSTRUCT_OPERATORS` | `false` | Extract numeric range/comparison filters (`rating_gte`/`rating_lte`) |

## Routing and query construction

| Env var | Default | Description |
|---------|---------|-------------|
| `ROUTING_ENABLED` | `false` | Enable the per-query routing layer in `RetrievalService` (domain/source/retriever + query construction); zero overhead when unset |
| `ROUTING_LOGICAL` | `false` | Add the LLM structured-classification router strategy (falls back to heuristic) |
| `ROUTING_SEMANTIC` | `false` | Add the embedding-similarity router strategy (falls back to heuristic) |
| `ROUTING_REGISTRY_PATH` | — | JSON route registry (`{name, description, sources, retriever}`); built-in default used when unset |
