# Retrieval

[← Back to README](../README.md)

This guide covers retrieval services, ranking modes, reranking, and query optimization.

## Grounded answer safety

The shared `src.context` answer pipeline treats retrieval results and approved
read-only tool results as the complete evidence boundary for factual claims.
Retrieved documents are normalized as stable `D*` evidence and successful tool
results as stable `T*` evidence. The model's internal structured draft must cite
known evidence IDs for every claim; malformed drafts, unknown IDs, and claims
without sufficient lexical support fail verification. Evidence can still be
incorrect at its source—the verifier establishes support by the supplied
evidence, not independent truth.

Guarded generation allows at most one corrective retry with the original
evidence and the verifier's findings. After that retry, unsupported claims are
removed. If no supported claim remains, or usable evidence is absent, the result
is exactly `I don't know based on the available evidence.` A supported answer is
never replaced with that canonical abstention: partially supported drafts render
only their supported claims. The no-LLM extractive path also abstains instead of
selecting an unrelated first sentence.

Confidence is deterministic rather than model-reported. It combines the verified
claim fraction, evidence coverage, and an optional evidence-sufficiency signal,
is clamped to `[0.0, 1.0]`, and is `0.0` for abstention. A fully verified answer
can therefore have confidence below `1.0` when its evidence is weak. The safety
guard is enabled by default; direct Python callers can explicitly pass
`GroundedGenerationConfig(enabled=False)` to retain legacy unconstrained
generation during compatibility migration.

### Provider-enforced structured output

Guarded generation always asks for the canonical `AnswerDraft` shape in its
prompt and always runs the local parser and evidence verifier. When the native
OpenAI adapter reports JSON Schema capability, it additionally sends that shape
as a strict Chat Completions `response_format`. Schema enforcement improves the
structural reliability of the provider response; it does not establish that a
claim is factually correct or supported by the retrieved evidence. Local
`parse_answer_draft` and `verify_answer_draft` therefore remain mandatory for
every successful draft, including provider-constrained drafts.

Generic OpenAI-compatible endpoints default to prompt-only generation because
compatibility with the OpenAI request shape does not imply support for Structured
Outputs. Operators may explicitly opt in a compatible endpoint by setting its
provider `custom_config` value `supports_json_schema` to the string `"true"`
(case-insensitive). Other providers remain prompt-only; the implementation does
not probe endpoints dynamically or infer support from model names.

If a schema-enabled request receives a 400 response that explicitly identifies
`response_format` or `json_schema` as unsupported or unknown, that same semantic
attempt is retried once in prompt-only mode. The endpoint remains prompt-only for
any later corrective attempt. This transport-level downgrade does not consume or
increase the single semantic corrective retry. Authentication, rate-limit,
transport, server, and unrelated request failures propagate normally;
they never trigger a schema downgrade.

A provider refusal produces the canonical abstention without exposing refusal
text. An incomplete response, such as a length-truncated completion, is treated
as verifier feedback and may use the one corrective semantic retry. Public
results and traces contain only aggregate booleans and categories, never raw
provider errors, refusals, model output, prompts, schemas, evidence bodies, or
tool arguments.

`structured_output_requested` records whether generation initially asked the
provider to enforce the answer schema. It remains `true` after an explicit
unsupported-schema downgrade, while `structured_output_applied` records whether
the provider actually applied that constraint.

`structured_output_category` is encounter-level aggregate metadata. It records
that a refusal, incomplete structured response, or timeout occurred during
generation, so the value may remain `incomplete` even when a later corrective
draft succeeds and produces a verified answer. A `timeout` category, unlike
`incomplete`, is always terminal: the timeout branch returns immediately, so no
later corrective draft can follow it. Consult `verification_status`,
`abstained`, and the rendered answer for the final answer outcome; do not
interpret `structured_output_category` as final answer verification status.

### Approved tool evidence

Tool evidence is opt-in. A caller supplies a registry and selector, and only
uniquely named tools explicitly classified `read_only` are offered to the
selector or invoked. Unknown, duplicate, side-effecting, and unspecified tools
are rejected. Defaults bound execution to two calls, a five-second timeout, and an
8192-character result size; selection and invocation failures or timeouts degrade
to retrieval-only evidence.
Tool outputs must be JSON-serializable and are normalized as data, never treated
as instructions.

Tool evidence passes through the same mandatory claim-support verifier as
retrieval evidence, and a supported claim citing a `T*` ID renders its `[Tx]`
marker like any `[Dx]`. Note, however, that the *optional* secondary
sentence-level grounding verifier (`src.context.grounding`) matches only `[Dx]`
citations — tool `[Tx]` markers are outside its regex, so that extra pass neither
checks nor strips them.

`ToolRequest` arguments are intended to be JSON-like values built from standard
containers. Mappings are copied and exposed read-only, while nested mappings,
lists/tuples, and sets/frozensets are recursively converted to immutable standard
container snapshots. This does not promise deep immutability for arbitrary
user-defined objects stored inside those containers; registries and selectors
should exchange JSON-like arguments only.

Synchronous selector calls and bounded iteration run through
`asyncio.to_thread` so they do not block the event loop. `asyncio.wait_for`
limits how long the pipeline awaits them, but timing out does not stop the
underlying worker thread, which may continue running. Selectors must therefore
be trusted, independently bounded, and nonblocking; the timeout is a pipeline
latency/failure boundary, not cancellation of synchronous work.

Tool results are serialized incrementally and rejected once the encoding exceeds
`max_result_chars` (default 8192 characters). Every chunk is counted before being
appended and `text` is joined only from counted chunks, so no result over the cap
ever reaches the prompt; total tool evidence is bounded at `max_calls ×
max_result_chars`. Rejection is deliberate rather than truncation: truncated JSON
is not valid JSON, and a truncated result can drop a negation and become evidence
that misleads the verifier. An oversized result is reported with the existing
`failed` status and degrades to retrieval-only answering, exactly like an
invocation failure. Serialization is synchronous and therefore covered by no
timeout. The pure-Python encoder backing this check yields one chunk per
scalar, so the size check can only run between chunks: encoding work is bounded
by the cap plus at most one fully-encoded scalar, and peak memory and event-loop
time are proportional to the largest individual string in the result, not to the
cap — the same species of caveat as the synchronous selector above, whose timeout
bounds pipeline latency but does not stop the underlying thread's work.

### Result and operational metadata

Existing result fields—`answer`, `citations`, `context`, `prompt`, and
`grounding_report`—remain available. The shared result adds defaulted safety
metadata: `confidence`, `verification_status` (`verified`, `partial`, or
`abstained`), `abstained`, summarized `tool_evidence`, and `retry_count`. The MCP
chat adapter preserves its established keys and adds confidence, verification,
abstention, and tool-source summaries.

A timeout on the primary synthesis call returns a degraded answer rather than an
error; a timeout on the schema-downgrade retry call (see Provider-enforced
structured output) is not guarded and still propagates as an error. The degraded
answer is abstention-shaped—`confidence` `0.0`, `verification_status` `abstained`,
no citations—so it flows through the same safe paths as any other abstention, but
it carries its own answer text and the `structured_output_category` `timeout`. The
distinction is deliberate: the canonical abstention asserts a conclusion about the
evidence, which a timeout cannot support, and collapsing the two would make an LLM
outage indistinguishable from a normal low-confidence answer. A timeout does not
consume a generation retry.

Tracing records counts and categories, tool names and statuses, retry count,
verification status, confidence, and abstention. It deliberately excludes
evidence bodies, raw tool output, full prompts, and tool arguments. Tool failures
are operational signals rather than fatal answer errors, because generation can
continue from retrieval evidence.

## Retrieval setup

`src.internal.document_index` is the single indexing entry point — filtering, chunking, embedding, retry-isolated writes, and failure reporting. Query-time retrievers and the retrieval HTTP client live in `src.context`. Reranker utilities live in `src.internal.servers.retrieval`.

Index construction is a separate offline step: the `index_builder` CLI reads a corpus and writes the searchable sparse/dense indexes. Query requests consume those existing indexes; they do not re-ingest or retrain on documents.

## Chunking

The first stage of index construction splits each document into token-budgeted
chunks (`src/internal/document_index/chunking.py`, driven by `ChunkingConfig` in
`models.py`). All three canonical chunking strategies are available:

| Strategy | Status | Where |
|----------|--------|-------|
| Structure-aware packer | **default** | `_split_text_paragraphs` |
| Recursive (heading hierarchy + code/table integrity) | **opt-in** | `_split_text_recursive` |
| Fixed-size + overlap | fallback + config | `_split_token_window` |
| Semantic | **opt-in** | `_split_text_semantic` |

**Structure-aware packer (default).** `_split_text_paragraphs` splits along
natural boundaries in descending order — paragraph/section breaks (a blank line, or
a newline before a Markdown `#` heading) → sentences (`.!?` and CJK `。！？`) — and
greedily packs sentences into a `chunk_size` budget, flushing at a section boundary
once a chunk is ≥ 50% full. A single sentence longer than `chunk_size` falls back to
a fixed-size token window. Overlap between adjacent chunks is sentence-granular
(`_overlap_tail`, `chunk_overlap` tokens). It is *structure-aware but not fully
recursive*: a fixed paragraph→sentence hierarchy, flat heading treatment, and no
code-block/table protection (see the opt-in recursive mode below for those).

**Recursive (opt-in).** Set `ChunkingConfig.recursive_chunking=True` to route to
`_split_text_recursive` — the genuinely recursive, structure-aware splitter. It
first marks fenced code blocks (` ``` `/`~~~`) and Markdown tables as **atomic**
(kept whole, never split internally — an oversized block becomes one chunk rather
than a fragment with a dangling fence), then splits the prose between them along a
coarse→fine hierarchy — Markdown heading levels (`\n# ` → `\n## ` → `\n### `) →
paragraph → line → sentence → word — recursing to a finer separator only when a
piece still exceeds `chunk_size`, and finally merges pieces up to `chunk_size` with
overlap carried from trailing prose only (so an atomic block is never sliced across
chunks). Mutually exclusive with `semantic_chunking`; purely lexical (no embedder).
Scope is Markdown + code/table integrity — **no HTML**; block detection is
heuristic, and a missed block degrades to prose rather than erroring.

**Fixed-size + overlap.** `_split_token_window` is a classic sliding window
(`step = chunk_size − chunk_overlap`). It is not a top-level mode — it is the
fallback the structure-aware and recursive splitters use for a single unit
(sentence or word) that alone exceeds `chunk_size`. Defaults: `chunk_size=900`,
`chunk_overlap=120`.

**Semantic (opt-in).** Set `ChunkingConfig.semantic_chunking=True` to route to
`_split_text_semantic`: it embeds each sentence (via the indexing pipeline's
`embedding_fn`), then places a boundary wherever the cosine distance between
adjacent sentences exceeds this document's `semantic_breakpoint_percentile`
(default `95.0`) — a self-calibrating breakpoint, so no fixed threshold to tune.
`semantic_buffer_size` (default `1`) optionally embeds each sentence with its
neighbors to denoise the signal. Every semantic chunk is still capped at
`chunk_size` (oversized topic regions are re-split with the structure-aware
splitter), and the whole path degrades to `_split_text_paragraphs` when no
embedder is supplied, the document has fewer than two sentences, or embedding
fails — it never blocks indexing. Tradeoff: one extra sentence-embedding pass over
the corpus at index time, which is why it is off by default.

```python
from src.internal.document_index.models import ChunkingConfig

# structure-aware packer (default)
ChunkingConfig()

# recursive: heading hierarchy + code/table integrity
ChunkingConfig(recursive_chunking=True)

# semantic chunking, cut at the 90th-percentile distance breakpoint
ChunkingConfig(semantic_chunking=True, semantic_breakpoint_percentile=90.0)
```

> **Two caveats.** (1) "Tokens" here are whitespace-delimited words
> (`re.findall(r"\S+")`), **not** model/BPE tokens — `chunk_size=900` is ~900 words
> (roughly 1,100–1,300 BPE tokens), so budget against an embedder's context window
> accordingly. (2) The **default** packer is **not** code-block or table aware:
> splitting on blank lines can cut inside a fenced code block or a Markdown table.
> Enable `recursive_chunking` (above) for code/table integrity; the default handles
> prose and heading-structured Markdown well but does not specially protect richly
> structured documents.

**Retrieval servers** (`src/internal/servers/retrieval/`):

| Module | Description |
|--------|-------------|
| `demo.py` | TF-IDF over corpus.jsonl — no Java, no FAISS |
| `hybrid.py` | RRF-fused dense (E5) + sparse TF-IDF; Java-free, FAISS-free — recommended for `AgenticRAGLoop` |
| `server.py` | Full `RetrievalService` (BM25 / dense / hybrid, env-configured via `RETRIEVAL_BACKEND`) with per-mode + admin endpoints |
| `rerank.py` | Standalone cross-encoder reranker (no retrieval) |

> **Two independent local stacks, not one.** `demo.py` and `hybrid.py` are
> self-contained servers — sklearn TF-IDF (plus an in-memory e5 dot-product leg in
> `hybrid.py`, no FAISS) — and expose `POST /retrieve` returning raw document
> dicts. `server.py` is a *different* stack: it wraps `RetrievalService` →
> `LocalBackend` → Pyserini/Lucene BM25 + FAISS/e5 and exposes `POST /search`
> returning `RetrievalResult`-shaped rows. They share neither code path nor API
> shape, so "hybrid" names two different implementations depending on which server
> you run. Everything below — reranking, retrieval optimization, query
> transformation, and routing — applies to the `RetrievalService` stack
> (`server.py` and the web backend), **not** to `demo.py`/`hybrid.py`.

**Web search servers** (`src/internal/servers/web_search/`):

| Module | Description |
|--------|-------------|
| `google.py` | Google Custom Search proxy |
| `serp.py` | SerpAPI proxy |
| `browser.py` | playwright-cli browser automation; no API key, ~5–10s/query |

**Start a retrieval server:**

```bash
# Demo — TF-IDF, no Java/FAISS
python3 -m src.internal.servers.retrieval.demo --corpus_path data/corpus.jsonl

# Hybrid — RRF-fused dense E5 + sparse TF-IDF (add --no-dense for TF-IDF only)
python3 -m src.internal.servers.retrieval.hybrid --corpus_path data/corpus.jsonl
```

`demo.py` also takes `--ignore-acl`, which serves documents regardless of the
request's `access_acl`. It exists so a client's own enforcement can be tested
with nothing behind it — `examples/verify_identity_capabilities.sh` starts the
server this way, because both bundled servers honouring `access_acl` would
otherwise make that script pass even with the web layer's enforcement removed.
Default off; no shipped entry point sets it.

All three bundled retrieval servers apply the same ACL rule,
`src/internal/retrieval/acl.py::acl_allows`: the request's `access_acl` must
intersect the document's declared `acl`, and a document that declares none is
public. `demo.py` and `hybrid.py` apply it per document; `server.py` applies it
inside the `RetrievalService` local backend, where every other filter key is a
metadata equality test.

### Serving cache

The web app keeps one process-local TTL cache
(`src/internal/cache/serving.py`) in front of the three request-path calls
that dominate latency. No Redis is involved; it is configured by the web
app's lifespan from `AGENTIC_SEARCH_SEARCH_CACHE_TTL` (default `300` s, `0`
disables) and cleared on shutdown, so a CLI or test process that never starts
the web app caches nothing.

| Call | Key | Not cached when |
|---|---|---|
| `SearchClient.retrieve` (every `retrieval`-provider caller) | server URL, query, `topk`, serialised filters — per query in a batch; only the misses are posted | the row is empty |
| `search_tool` for `google` / `serpapi` / `serper` | provider, query, page, page size | the result is empty or contains an error page |
| `RerankHTTPRankingStage` | rerank URL, query, `top_k`, the candidate texts | the ranked list is empty |

The serialised access filters are part of the retrieval key, so callers with
different ACLs never share an entry, and every call site still enforces the ACL
on what it gets back — a hit is checked exactly as a miss is. Hits are copied
before they are returned, so a caller mutating a result cannot poison later hits.

**Build indexes:**

```bash
python3 -m src.internal.document_index.index_builder \
  --retrieval_method e5 --model_path intfloat/e5-base-v2 \
  --corpus_path data/corpus.jsonl --faiss_type Flat --save_dir data/indexes/

python3 -m src.internal.document_index.index_builder \
  --retrieval_method bm25 --corpus_path data/corpus.jsonl --save_dir data/indexes/
```

**Web search servers:**

```bash
python3 -m src.internal.servers.web_search.serp \
  --search_url "https://serpapi.com/search" --topk 3 --serp_api_key "$SERP_API_KEY"

python3 -m src.internal.servers.web_search.google \
  --api_key "$GOOGLE_API_KEY" --topk 5 --cse_id "$GOOGLE_CSE_ID" --snippet_only
```

**Health check:**

```bash
curl -i -sS http://127.0.0.1:8001/health
curl -i -sS -X POST http://127.0.0.1:8001/retrieve \
  -H "Content-Type: application/json" -d '{"query":"What is FAISS?","topk":5}'
```

For complete request and response payloads, see the [HTTP API reference](api-reference.md).

## Retrieval in auto-routed API requests

The web API has an evidence-first search path above the retrieval services. For an `/api/agent` request that routes to `search` with `source_provider=auto`, the backend:

1. queries internal retrieval;
2. accepts an exact-title, fuzzy-plus-semantic, or semantic match that clears the direct sufficiency gate;
3. otherwise tries SerpAPI with the original query;
4. otherwise calls the configured browser-search HTTP service;
5. returns a deterministic no-results or sources-unreachable response when no evidence exists.

This sequence is different from explicit `mode=hybrid_search`, whose helper can query internal retrieval in parallel with a cascading web leg and then merge/rerank results. It is also different from the internal retrieval router described below.

Identity does not change which path a request takes. Every caller carries an ACL — anonymous carries `["public"]` — and it narrows what the path returns. Internal retrieval receives the ACL filters and the web layer enforces them on what comes back; external web providers receive no internal document ACL object. See [Access control](request-routing.md#access-control) for where each path applies them.

> **`LocalBackend` filter caveat.** Internal retrieval applies filters *after*
> top-k retrieval as an exact-match over document metadata, and only over
> *non-standard* document keys — the standard fields `id`, `title`, `text`,
> `contents`, and `url` are excluded from the matchable metadata, so a filter on
> one of those keys silently matches nothing. Because filtering is post-hoc,
> aggressive filters can shrink a result set below `top_k`;
> `OVER_FETCH_MULTIPLIER` compensates upstream. (The separate enterprise
> document-index backend has a richer multi-tenant `IndexFilters` model — see
> `src/internal/document_index/FILTER_SEMANTICS.md` — which the local stack does
> not use.)

That pipeline composition uses the same internal stage sequence throughout the web backend:

1. bounded session history resolves continuation-style queries into a retrieval query while retaining the original user question;
2. the selected existing provider returns a normalized candidate set and receives ACL filters when it is internal retrieval;
3. one ranking stage deduplicates candidates, optionally invokes the existing reranker, and applies MMR/truncation;
4. inference synthesizes from ranked evidence, or the pipeline returns deterministic status/results when evidence or synthesis is unavailable;
5. shared response finalization persists citations, documents, and stage metadata.

These stages are internal adapters. Existing `/retrieve`, `/search`, and `/rerank` endpoints remain available with their current payloads, and no new retrieval API was added. Backend RRF inside `RetrievalService` remains distinct from web-layer candidate ranking: RRF fuses backend result lists; the web ranking stage normalizes, deduplicates, optionally reranks, and diversifies the resulting evidence.

### The direct-first sufficiency gate

For an `auto`/`retrieval` request, the backend runs internal retrieval
and compares the query to the **rank-1 result only** through a
backend-independent tiered gate (`_direct_gate_decision`):

| Tier | Condition | Outcome |
|------|-----------|---------|
| `exact` | normalized query == normalized title | direct |
| `fuzzy` | Levenshtein(query, title) < 2 **and** cosine(query, passage) > threshold | direct |
| `semantic` | cosine(query, passage) > threshold | direct |
| `weak` | none of the above | escalate |

A **direct** hit returns the ranked documents with a deterministic non-LLM
summary — no agent loop, no model call. A **weak** result escalates. Note: with
no e5 gate model loaded, `cosine` is `None`, so only `exact`-title queries go
direct and everything else escalates.

## The agentic search loop

Escalation hands off to `SearchAgentLoop` (`src/agents/search/search.py`) — the
"agentic" core, a multi-turn retrieval-grounded agent. (It only runs when a local
model is loaded; without one, the request degrades to the pipeline composition
above.) Unlike `ToolAgentLoop`'s JSON tool calls, this loop is **XML-tag driven**:
the system prompt teaches the model a fixed action vocabulary, and the
environment answers back in the same language by injecting result blocks.

| Tag | Written by | Purpose |
|-----|-----------|---------|
| `<think>` | model | reason about what's known/missing (opens every turn) |
| `<search>` / `<searches>` | model | one query / parallel independent queries |
| `<search retriever="web">` | model | live web vs. internal corpus (`vdb`, default) |
| `<fetch>` | model | pull full page content when snippets are thin |
| `<subquestions>` | model | decompose a multi-facet task |
| `<search_decision>answer` | model | skip search, answer from internal knowledge |
| `<answer>` | model | final grounded answer |
| `<information>` / `<search_evaluation>` | **environment only** | injected results + sufficiency verdict |

**Turn cycle** (`run()`, bounded by `max_turns`): generate → parse XML actions →
register `<subquestions>` as tracks → dedup/budget the requested queries → if
`<answer>` with no new search, apply the answer gate → else run a search round
(retrieve, dedup by source, judge sufficiency, inject `<information>` +
`<search_evaluation>`) → repeat until answered, forced, or budget-exhausted.

**Citations.** Rounds are the citation coordinate system: each result is labeled
`[RxQyDz]` (round, query, doc — all 1-based). The web backend re-enumerates
rounds→queries→docs identically and deliberately skips dedup so every cited label
resolves to its own source card.

### Adaptive budget and the sufficiency control layer

Two decisions — *keep searching?* and *how to answer?* — are computed by three
collaborating pieces, split into a stateless policy over a snapshot of mutable
loop state:

- **`SearchResultEvaluator`** (`src/agents/components/result_evaluation.py`) — the boolean
  threshold gate. A round is sufficient iff total results ≥ `min_total_results`
  and every query individually clears result-count / content-length / score
  thresholds. Emits human-readable reasons that become the `<search_evaluation>`
  weak-query hints. Score thresholds default to `0.0`, so out of the box
  sufficiency is about *presence of substantive results*, not score magnitude.
- **`EvidenceJudge`** (`src/agents/components/evidence_judge.py`) — wraps the
  boolean as a safety rail and adds a continuous `evidence_score ∈ [0,1]`
  (`0.5·frac_sufficient + 0.5·mean(squash(top_score))`). The score term keeps the
  signal monotonic in retrieval quality even after every query clears the boolean
  bar, giving the plateau detector and the GRPO reward a smooth gradient.
- **`LoopController`** (`src/agents/components/loop_controller.py`) — pure policy.
  The search budget is **adaptive**: `effective_search_limit` grows with the
  number of open subquestions, clamped to `max_search_limit_cap` (10). The answer
  gate returns `ACCEPT` (evidence sufficient), `FORCE` (budget reached — best
  effort, flagged), or `REJECT` (insufficient — search again). `FORCE` after
  `max_answer_rejections` is the escape hatch that prevents an infinite
  reject loop.

The loop's large per-run metrics dict (`citation_count`,
`cited_task_coverage_ratio`, `useful_fetched_pages` vs `unnecessary_fetch_count`,
`answer_when_evidence_insufficient`, `budget_used_ratio`, …) is not only
observability — it is the shaped reward signal for GRPO training.

## Neural reranking

`RetrievalService` optionally reranks hybrid-fused results via a layered wrapper chain. Set `RERANKER_PROVIDER` to enable; all wrappers are opt-in via env vars and compose on top of the unchanged `Reranker` leaf.

**Wrapper chain** (outermost → innermost):
```
TwoStageReranker → CachedReranker → AsyncReranker → Reranker (leaf)
```

**Enable local BGE reranking:**
```bash
RERANKER_PROVIDER=local RERANKER_MODEL=BAAI/bge-reranker-v2-m3 \
  PYTHONPATH=src:. uvicorn src.internal.servers.web.app:app --host 127.0.0.1 --port 7860
```

**Enable Cohere reranking:**
```bash
RERANKER_PROVIDER=cohere RERANKER_MODEL=rerank-english-v3.0 COHERE_API_KEY=... \
  PYTHONPATH=src:. uvicorn src.internal.servers.web.app:app --host 127.0.0.1 --port 7860
```

**Enable async + Redis cache wrapper:**
```bash
RERANKER_PROVIDER=local RERANKER_ASYNC=true \
  RERANKER_TIMEOUT_MS=500 RERANKER_CACHE_REDIS_URL=redis://localhost:6379 \
  PYTHONPATH=src:. uvicorn src.internal.servers.web.app:app --host 127.0.0.1 --port 7860
```

**Enable two-stage pipeline** (fast pre-filter → heavy scorer):
```bash
RERANKER_PROVIDER=local RERANKER_TWO_STAGE=true \
  RERANKER_FAST_MODEL=BAAI/bge-reranker-base \
  RERANKER_PRE_FILTER_TOP_N=50 RERANKER_OVER_FETCH_MULTIPLIER=2.0 \
  PYTHONPATH=src:. uvicorn src.internal.servers.web.app:app --host 127.0.0.1 --port 7860
```

**ONNX runtime** (lower latency than PyTorch, requires `pip install optimum[onnxruntime]`):
```bash
RERANKER_PROVIDER=local RERANKER_USE_ONNX=true RERANKER_MODEL=BAAI/bge-reranker-base \
  PYTHONPATH=src:. uvicorn src.internal.servers.web.app:app --host 127.0.0.1 --port 7860
```

**Evaluate reranker quality and latency:**
```bash
# Baseline vs reranked NDCG/MRR + per-query latency
python -m src.internal.retrieval.eval_runner \
  --dataset data/eval/qa_pairs.jsonl --top_k 10 \
  --reranker local --reranker_model BAAI/bge-reranker-v2-m3 \
  --compare-baseline --slo-ms 200 \
  --output data/eval/retrieval_metrics.json   # also written here; the Dev Console reads data/eval/*.json

# Output JSON:
# { "retrieval":  {"ndcg@10": 0.48, "mrr": 0.63},
#   "reranked":   {"ndcg@10": 0.55, "mrr": 0.71, "map@10": 0.52},
#   "latency_ms": {"mean": 312, "p50": 290, "p99": 680, "n": 50},
#   "reranker_improvement_ratio": 0.145 }
```

**Benchmark model configurations offline:**
```bash
python -m src.internal.retrieval.reranker_benchmark \
  --qa-pairs data/eval/qa_pairs.jsonl \
  --models BAAI/bge-reranker-base BAAI/bge-reranker-v2-m3 \
  --batch-sizes 8 16 32 \
  --max-tokens 256 512 \
  --output results/reranker_bench.jsonl
# Prints ranked table sorted by NDCG@10
```

## Retrieval optimization

All optimization components are opt-in; unset env vars = unchanged M1–M4 behavior.

**Tune BM25 parameters against your QA pairs:**
```bash
curl -s -X POST http://localhost:8001/internal/optimize/bm25-tune \
  -H "Content-Type: application/json" \
  -d '{"qa_pairs_path": "data/eval/qa_pairs.jsonl", "k1_range": [0.6, 0.9, 1.2], "b_range": [0.5, 0.75]}' \
  -H "Authorization: Bearer $TOKEN"
# → {"k1": 0.9, "b": 0.6, "score": 0.86}
```

**Learn fusion weights (sparse vs dense RRF weights):**
```bash
curl -s -X POST http://localhost:8001/internal/optimize/fusion-weights \
  -H "Content-Type: application/json" \
  -d '{"qa_pairs_path": "data/eval/qa_pairs.jsonl"}' \
  -H "Authorization: Bearer $TOKEN"
# → {"w_sparse": 0.38, "w_dense": 0.62}
```

Both tuning endpoints score against labelled QA pairs (never user feedback)
and **return** the tuned values; nothing persists them and nothing loads them
at startup. Apply a result by setting the corresponding env vars yourself.

**Tune HNSW ef_search for a recall target:**
```bash
curl -s -X POST http://localhost:8001/internal/optimize/hnsw-tune \
  -H "Content-Type: application/json" \
  -d '{"target_recall": 0.82}' \
  -H "Authorization: Bearer $TOKEN"
# → {"ef_search": 96, "measured_recall": 0.831}
```

**Retrieval stats (cache hit rate, latency, throughput):**
```bash
curl -s http://localhost:7860/api/admin/retrieval/stats \
  -H "Authorization: Bearer $TOKEN"
# → {"result_cache_hit_rate": 0.42, "p99_latency_ms": 112, "throughput_qps": 87, ...}
```

**Hot-reload tunable parameters without restart:**
```bash
curl -s -X PATCH http://localhost:7860/api/admin/retrieval/config \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $TOKEN" \
  -d '{"rrf_k": 80, "mmr_lambda": 0.4, "nprobe": 96, "result_cache_ttl": 600}'
# → {"applied": ["rrf_k", "mmr_lambda", "nprobe", "result_cache_ttl"]}
```

**Enable query expansion and result caching:**
```bash
QUERY_EXPANSION_ENABLED=true SPELL_CORRECTION_ENABLED=true EXPANSION_MAX_TERMS=3 \
  BM25_VARIANT=bm25plus \
  RESULT_CACHE_REDIS_URL=redis://localhost:6379 RESULT_CACHE_TTL=300 \
  ADAPTIVE_MMR=true \
  PYTHONPATH=src:. uvicorn src.internal.servers.web.app:app --host 127.0.0.1 --port 7860
```

## Query transformation optimization

A layered-wrapper optimization stack over `QueryTransformPipeline`, parallel to Neural Reranking. Every layer is opt-in; with all `QT_*` unset, `RetrievalService` runs the single-query path unchanged (`build_query_transform_pipeline_from_env` returns `None`).

**Wrapper chain** (outermost → innermost):
```
RoutedQueryTransformPipeline → CachedQueryTransformPipeline → AsyncQueryTransformPipeline → QueryTransformPipeline (leaf)
```

**Enable parallel transforms + Redis bundle cache:**
```bash
QT_DECOMPOSE=true QT_HYDE=true QT_STEP_BACK=true \
  QT_ASYNC=true QT_TRANSFORM_TIMEOUT_MS=400 \
  QT_CACHE_REDIS_URL=redis://localhost:6379 QT_CACHE_TTL_SECONDS=600 \
  PYTHONPATH=src:. uvicorn src.internal.servers.web.app:app --host 127.0.0.1 --port 7860
```

**Enable Multi-Query + weighted RAG-Fusion:**
```bash
QT_MULTI_QUERY=true QT_MULTI_QUERY_N=3 QT_FUSION_WEIGHTED=true \
  PYTHONPATH=src:. uvicorn src.internal.servers.web.app:app --host 127.0.0.1 --port 7860
```

**Enable per-query learned routing** (heuristic until an artifact exists):
```bash
QT_ROUTER=true QT_ROUTER_MODEL_PATH=data/query_router.joblib \
  PYTHONPATH=src:. uvicorn src.internal.servers.web.app:app --host 127.0.0.1 --port 7860
```
`QT_ROUTER` and `QT_MULTI_QUERY` each activate the pipeline on their own — no other `QT_*` flag is required.

Query transformation is **backend-only** — there is no dedicated HTTP endpoint and no query-transform-specific UI. The pipeline runs inside `RetrievalService.from_env()`, so it applies to **both** the retrieval server's `/search` and the web backend's `/api/agent`. Its observable effect is the `+rag_fusion` suffix on `retrieval_mode`.

**Test it on the retrieval server** (`POST /search` — `retrieval_mode` reflects the transform):
```bash
# Start the retrieval server with QT flags enabled, then:
curl -s -X POST http://localhost:8001/search \
  -H "Content-Type: application/json" \
  -d '{"query": "Compare dense and sparse retrieval", "top_k": 5}' \
  | python -c "import sys, json; print(json.load(sys.stdin)['retrieval_mode'])"
# → hybrid+rag_fusion
```

**Test it on the web backend** (`POST /api/agent`):
```bash
curl -s -X POST http://localhost:7860/api/agent \
  -H "Content-Type: application/json" \
  -d '{"query": "Compare dense and sparse retrieval", "mode": "chat_loop", "top_k": 5}' \
  | python -m json.tool | grep -i retrieval_mode
# → "retrieval_mode": "hybrid+rag_fusion"   (or "hybrid+rag_fusion+reranked" with a reranker)
```

**Extract metadata filters from natural language** (numeric operators behind `QT_CONSTRUCT_OPERATORS`):
```bash
QT_CONSTRUCT_FILTERS=true QT_CONSTRUCT_OPERATORS=true \
  PYTHONPATH=src:. uvicorn src.internal.servers.web.app:app --host 127.0.0.1 --port 7860
# "arxiv papers after 2023 rated above 4" → filters {date_after: "2023-...", rating_gte: 4}
curl -s -X POST http://localhost:7860/api/agent \
  -H "Content-Type: application/json" \
  -d '{"query": "arxiv papers after 2023 rated above 4 on retrieval", "mode": "chat_loop", "top_k": 5}'
```

**Train the learned router offline:**
```bash
python -m src.internal.retrieval.train_query_router --out data/query_router.joblib
# → wrote data/query_router.joblib
# Predicts 7 transform labels: decompose, hyde, step_back, keywords, construct_filters, multi_query, rewrite
```

**Gate transform latency in CI:**
```bash
python -m src.internal.retrieval.eval_runner \
  --dataset data/eval/qa_pairs.jsonl --top_k 10 --qt-slo-ms 300
# Records per-query "qt_latency_ms"; exits non-zero when P99 transform latency > 300ms
```

**Benchmark technique combinations offline** (Python API; the `--dataset` CLI ships a stub `retrieve_fn` to wire to your retriever):
```python
from src.context.query_transform import QueryTransformConfig
from src.internal.retrieval.query_transform_benchmark import run_query_transform_benchmark

dataset = [("what is FAISS", {"doc-1"}), ("compare BM25 and dense", {"doc-2"})]

def retrieve(query, config):
    # build a pipeline from `config`, run RetrievalService.search, return ranked doc_ids
    ...

rows = run_query_transform_benchmark(dataset, retrieve, [
    QueryTransformConfig(),
    QueryTransformConfig(multi_query=True),
    QueryTransformConfig(decompose=True, hyde=True),
], k=10)
# → [{"config_signature": "...", "recall": 0.91, "ndcg": 0.78, "mean_latency_ms": 142.0}, ...]
```

## Routing and query construction

The RAG **Routing → Query Construction** stage (`src/internal/routing/`). It decides **where** a query should go (domain → source → retriever) and **how** to express it for the chosen backend. Distinct from [Intent Routing](architecture.md#intent-routing) (web-level `search`/`chat`/`tool`) and from `QueryRouter` (which picks *transforms*): this layer picks the *retriever/construction target* per query.

**Four routing layers, four jobs.** The word "routing" refers to four independent
mechanisms in this codebase — easy to conflate, so:

| Layer | Where | Decides | Values |
|-------|-------|---------|--------|
| Intent routing | web backend (`recognize_intent`) | which experience to run | `chat` · `search` · `tool` |
| Provider cascade | web auto-search | which evidence source | internal → serpapi → browser |
| Retriever-target routing | `src/internal/routing/` | which retriever / construction target | `sparse·dense·hybrid·metadata·sql·graph·api` |
| Transform routing | `QueryRouter` (`QT_ROUTER`) | which query transforms to apply | `decompose·hyde·step_back·keywords·construct_filters·multi_query·rewrite` |

They run at different stages and compose: intent picks `search`, the provider
cascade sources evidence, and (when enabled) transform routing and
retriever-target routing shape the internal-retrieval leg.

**Backend-only and default-off.** With no `ROUTING_*` env set, `build_router_from_env()` returns `None`, `RetrievalService.search` skips the routing branch entirely, and behavior is byte-identical to today — zero overhead, no frontend change. There is no dedicated HTTP endpoint or UI; routing runs inside `RetrievalService.from_env()`.

**Pipeline:**
```
query → Router.route() → RouteDecision(domain, sources, retriever, construction_target)
      → QueryConstructor.construct() → ConstructedQuery(target, payload, text)
```

**Router strategies** (heuristic default; LLM strategies fall back to it on any failure):

| Strategy | Env | How it routes |
|----------|-----|---------------|
| Heuristic | (default) | Rule-based cue matching → SQL / GRAPH / API / default HYBRID. No LLM; the path the accuracy gate runs against |
| Logical | `ROUTING_LOGICAL=true` | LLM structured-classification into a registered route by name |
| Semantic | `ROUTING_SEMANTIC=true` | Embedding cosine between the query and each route's description |

Routes come from a config-driven registry (`ROUTING_REGISTRY_PATH` → JSON of `{name, description, sources, retriever}`; a built-in default mirrors the local corpus). `RetrieverTarget` ∈ `sparse · dense · hybrid · metadata · sql · graph · api`.

**Six query constructors** (`construction/`, one `construct(query, route) -> ConstructedQuery` interface):

| Constructor | Target | Backing | Output |
|-------------|--------|---------|--------|
| Metadata Filter | `metadata` | wraps `QueryConstructor` | NL → `{filters}` + cleaned query |
| Vector Search | `dense` | params | `{top_k, namespace, filters}` |
| Hybrid Retrieval | `hybrid` | reuses `adaptive_mmr_lambda` | `{rrf_k, w_sparse, w_dense, mmr_lambda}` |
| SQL Generation | `sql` | net-new (no exec) | schema-aware Text-to-SQL, SELECT-only + table allowlist + multi-statement reject |
| Knowledge Graph | `graph` | net-new (no exec) | read-only Cypher (`MATCH…RETURN`), word-boundary write-clause rejection |
| API Request | `api` | net-new (no exec) | `{endpoint, params}` filtered to an `ApiSpec` allowlist |

The three net-new constructors **build and validate but never execute** a query — there is no live SQL/KG/API backend, so `RetrievalService` short-circuits the `sql`/`graph`/`api` targets to `([], "routed:<target>")`. When a real backend is wired later, only the executor changes. Every `route()`/`construct()` degrades to a safe empty/None payload rather than raising.

**Enable per-query routing:**
```bash
ROUTING_ENABLED=true \
  PYTHONPATH=src:. uvicorn src.internal.servers.web.app:app --host 127.0.0.1 --port 7860
# Optional LLM strategies + a custom route registry:
ROUTING_ENABLED=true ROUTING_LOGICAL=true ROUTING_SEMANTIC=true \
  ROUTING_REGISTRY_PATH=data/routes.json  uvicorn ...
```

**Score routing accuracy** (heuristic router; no LLM needed):
```bash
python -m src.internal.retrieval.eval_runner \
  --routing-eval --dataset data/eval/routing_labels.jsonl
# → {"routing_accuracy": 1.0, "num_queries": 12}
```
