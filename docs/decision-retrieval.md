# Retriever-free document navigation

This opt-in backend borrows the hierarchy and judgment strategy from
[Retriever-Free Retrieval: Navigating Documents with Decision Models](https://arxiv.org/abs/2610.35868)
(DecisionRAG, Section 3). It does not replace the default demo or hybrid server.
There is no embedding index, BM25, TF-IDF, or first-stage search at query time.
Optional [offline corpus hierarchy construction](corpus-hierarchy.md) recovers
metadata/HTML/URL structure or builds semantic clusters, persists descriptions,
and maintains incremental snapshots; clustering embeddings are offline only.

## Run locally

With the repository's existing Python dependencies installed:

```bash
python -m src.internal.servers.retrieval.decision \
  --corpus_path data/corpus.jsonl \
  --model /path/to/local/instruct-model \
  --device cpu \
  --beam-width 5 \
  --shortlist-size 20 \
  --batch-size 16 \
  --passage-words 200
```

`--model` accepts a local Transformers causal-language-model directory or a Hugging
Face identifier (which may download weights). Choose an instruction model with a
chat template and distinct single-token `Yes`, `No`, and uppercase-letter labels.
Remote model code is disabled. Scoring runs locally; corpus text is not sent to
a hosted inference provider. `--device mps` or `--device cuda` is optional.
`--corpus demo`, registered comma-lists, and `--corpus all` are also supported.

The server listens on `127.0.0.1:8001`, with `/health` and the existing `/retrieve`
contract. Point the usual local retrieval client at its `/retrieve` endpoint;
no agent or UI changes are needed.

```bash
curl http://127.0.0.1:8001/retrieve \
  -H 'Content-Type: application/json' \
  -d '{"query":"How do I configure search?","topk":5,"return_scores":true,"include_trace":true}'
```

Use exactly one of `query` or `queries` (up to 32 queries per request). A single
query returns a flat `results` list; a query batch returns one list per query. `return_scores` wraps each
document as `{"document": {...}, "score": probability}`. Optional `trace` records
the candidate IDs, scores, and selected IDs at each stage; it is not returned
by default. Invalid queries, excessive `topk`, and context overflow return HTTP
422 rather than an unrelated-retriever fallback. Oversized candidate pools are
adapted to the token budget as described below; an individually oversized
candidate still fails explicitly.

## Corpus structure and navigation

The JSONL schema remains `id`, `title`, `contents` (or `text`), `url`, `metadata`.
An optional `metadata.topic` supplies a topic label. Documents without it belong
to `Documents`. Duplicate document IDs and malformed text/topic fields are errors.

1. Build an in-memory structural tree: **topic -> document -> Markdown section
   -> passage**. No clustering, generated summaries, or vector index is needed.
   Plain text gets one `Body` section. Markdown fenced-code headings are not
   sections. Passages contain at most `--passage-words` whitespace-delimited
   words and `--max-passage-chars` characters (default 1200), supporting no-space
   text too; these are not tokenizer token budgets. With `--hierarchy`, stored
   group structure and normalized sections replace this basic fallback tree.
2. Prune inaccessible documents and empty branches **before any model call**,
   using the existing `filters.access_acl` semantics. This is defense in depth,
   not an authentication boundary: filters must come from a trusted caller.
3. Binary-judge candidate relevance with all candidates in the request visible
   together. Keep the globally best `--beam-width` nodes at each internal level,
   then expand only their children. Split large levels into `--batch-size`
   requests (1-26 candidates); the local adapter splits a request further if its
   full prompts exceed the token budget. Binary probabilities need not sum to one.
4. Binary-screen passages to `--shortlist-size` (1-26), then rank the fitting
   shortlist in **one multiple-choice judgment**. The local adapter keeps the
   largest binary-ranked prefix that fits the token budget, without shortening
   passages. Return up to `topk` passages sorted by this final probability. Ancestor scores are not multiplied into
   leaf scores, and results from different choice pools are never fused.

Passage IDs are `<document-id>::passage-<zero-based-index>`. Original URLs and
metadata are retained; `metadata.document_id`, `metadata.section`, and
`metadata.passage_index` identify the evidence location.

`topk` must be positive and no larger than the shortlist size. A small beam or a
token-limited choice pool may leave fewer than `topk` passages. Ties are stable in corpus order. Empty corpora,
fully filtered corpora, and empty documents return no passages without inference.
Optional `--min-relevance` (default 0, no cutoff) lets binary screening abstain;
thresholds require calibration for the chosen model.

## Model adapter and limits

The paper uses specialized decision models that score many questions against a
shared context in one forward pass. This implementation supplies a **local
causal-LM approximation**, not that specialized architecture:

- Binary judgments use the next-token logits for `Yes` and `No`, normalized only
  over those two options. Each question sees the same candidate list, but its
  context prefill is repeated. `--inference-batch-size` (default 1) bounds the
  number of prompts evaluated together.
- Final choice uses a single prompt's next-token logits for `A` through the
  shortlist's final letter, normalized across the options.
- No `generate()` call, autoregressive rationale, or token sampling is used.
  Model-label tokenization is checked explicitly. Models that explicitly support
  `logits_to_keep` or `num_logits_to_keep` compute only the final position's
  vocabulary logits. Older models use their standard forward pass and may need
  substantially more memory for full-sequence logits.
- Full chat prompts are measured against `--max-context-tokens` (default 4096)
  and the model's declared position limit. Oversized binary pools are recursively
  split into smaller candidate pools; final choice uses the largest fitting
  ranked prefix in one request. Neither passages nor prompts are truncated.
  If a single candidate plus the query/instructions cannot fit, the request is
  rejected; reduce passage words or increase the supported context limit.
  Inference is serialized across concurrent HTTP requests to bound simultaneous
  model use.
- Queries without an active `filters.access_acl` reuse the immutable tree
  directly. ACL-filtered queries still prune the tree before any inference.

The model-independent `DecisionModel` protocol in
[`decision_retriever.py`](../src/internal/document_index/decision_retriever.py)
allows replacing the adapter with a specialized local decision model implementing
`binary(query, candidates)` and `choice(query, candidates)`. Both return one
finite probability per ordered candidate; choice probabilities must sum to one.
Adapters may optionally implement `choice_capacity(query, candidates) -> int`
to limit final choice to a fitting non-empty ranked prefix; adapters without it
retain the configured shortlist. Core navigation has no torch/Transformers
dependency.

This is not a reproduction of the paper's latency, cost, or accuracy. Generic
causal-model label logits are not guaranteed to be calibrated. Title-only
document routing and section-heading routing can prune relevant evidence, and
untitled/unstructured corpora may benefit little from navigation. Weak group
descriptions are a further irreversible-pruning risk; see
[description judgeability](corpus-hierarchy.md#description-judgeability) for
coverage sampling, bounded preview recovery and the paired evaluator. Binary
batch scores can vary with their comparison pool. Evaluate recall, ranking,
memory, and latency on your corpus before enabling it in production.

## Focused tests (no weights or external services)

```bash
python -m pytest tests/unit/test_decision_retriever.py \
  tests/unit/test_local_decision_model.py \
  tests/unit/servers/retrieval/test_decision_retrieval.py
```

Tests cover pruning, shared candidate context, joint reranking, score validation,
ACL-before-inference, passage provenance, context limits, logits-only inference,
adaptive context splitting, whole-passage choice budgeting, final-position-only
logits on supported models, unrestricted-query ACL fast paths, single/batch HTTP
response compatibility, CLI wiring, and the existing search client's response
parser. A tiny randomly initialized Transformers model checks
real forward-pass execution and padding invariance without downloaded weights.
These tests do not measure trained-model retrieval quality.
