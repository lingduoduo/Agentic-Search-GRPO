# Offline corpus hierarchies

The DecisionRAG backend can load a persisted corpus tree instead of treating all
unlabelled documents as one topic. Construction and model inference happen
offline; retrieval still uses decision judgments, without an embedding search.

This pipeline handles two cases:

- **Existing structure:** recover groups from metadata or URL directories and
  sections from explicit sections, HTML or Markdown headings. Conservative
  plain-text layout recovery is opt-in.
- **Missing structure:** embed each document's title and first sentence,
  cluster with deterministic k-means, then describe each cluster. A local LLM
  can generate descriptions; without one, explicitly marked extractive
  descriptions use representative titles/first sentences.

A useful tree is a prerequisite for useful navigation. Size diagnostics and
incremental maintenance reduce operational work; they do not establish that a
tree, its generated descriptions, or navigation budgets are optimal.

## Structured documents and mixed formats

Input is JSONL with a stable nonempty string `id`, `title`, `contents` (or `text`),
`url`, and `metadata`. Binary PDF/Office files must first go through the existing
connector/document extraction pipeline. Supported text payloads are plain text,
Markdown and HTML; no document URLs are fetched.

```json
{"id":"billing-cn","title":"账单说明","contents":"<h1>退款</h1><p>退款将在三个工作日到账。</p>","metadata":{"format":"html","hierarchy_path":["Support","Billing"],"acl":["public"]}}
```

Group path precedence:

1. Explicit `metadata.hierarchy_path`: a list of nonempty labels.
2. Nonempty `metadata.topic`, `metadata.product`, `metadata.category`, in that
   order, combined into a path.
3. URL hostname and directory segments, excluding the final path component.
4. No path: semantic clustering in `auto`, or a labelled `Documents` fallback
   in `structure` mode.

Sections prefer an explicit `sections` list (top-level, then metadata), whose
items contain string `heading` and `text` fields. Otherwise HTML `h1`–`h6`,
Markdown ATX/setext headings, and fenced code blocks are recognized. Set
`metadata.format` or `metadata.content_type` for ambiguous payloads. HTML
comments, noncontent elements and common explicit hiding attributes/styles are
excluded. Static parsing cannot reproduce arbitrary stylesheet visibility or a
browser's layout engine.

Plain text remains one Body section unless `--infer-layout` is set. That option
recognizes conservative uppercase headings separated from surrounding text;
it can miss headings or misclassify text. Every normalized document records
`structure_source` and `section_source` so recovered structure is auditable.

```sh
python -m src.internal.document_index.hierarchy_cli build \
  --corpus-path data/corpus.jsonl \
  --mode structure \
  --output indexes/support-tree.json
```

Structure mode needs no embedding weights. Without `--description-model`, it
also needs no language-model weights. Explicit metadata paths are preserved,
even if a group is larger than the semantic cluster cap; inspect warnings
identify those groups rather than silently rewriting authoritative structure.

## Unstructured corpora and NQ-style construction

`auto` combines path-derived groups with clustering for pathless documents.
A URL prefix alone does not imply a useful hierarchy: Wikipedia's shared
`/wiki/` prefix is not a topic taxonomy. Use `--mode cluster` to ignore such
paths and cluster every document, or supply curated metadata paths.

```sh
python -m src.internal.document_index.hierarchy_cli build \
  --corpus-path data/wikipedia.jsonl \
  --mode cluster \
  --embedding-model intfloat/e5-base-v2 \
  --clusters 1000 \
  --max-cluster-size 1000 \
  --description-model /path/to/local/instruct-model \
  --output indexes/nq-tree.json
```

This follows the paper's **title + first sentence → e5-base-v2 embeddings →
k-means → LLM cluster descriptions** construction strategy. It is not a
reproduction of the published dataset preparation or retrieval results. The
requested cluster count is capped at the document count; overflow splitting
can increase the resulting number of groups beyond that initial count.

Without `--clusters`, the initial count is `ceil(document_count /
target_cluster_size)`. `--target-cluster-size` defaults to 100, and
`--max-cluster-size` to 200. Oversized clusters are divided into deterministic
balanced chunks. This also terminates safely for identical vectors, but such
fallback partitions may have weak semantic coherence; the report warns about
this instead of presenting them as good clusters.

The default encoder is `intfloat/multilingual-e5-base`. Use an encoder suitable
for the corpus languages; English e5-base-v2 is an explicit choice for the NQ
recipe. E5 adapters apply the `passage:` prefix and normalize vectors.
SentenceTransformer and language-model imports/loads are lazy. Model identifiers
may download weights; local model directories avoid that. Remote model code is
disabled, and corpus text is never sent to a hosted inference endpoint.

`--description-model` uses deterministic local generation over a bounded sample
of titles and first sentences. Descriptions record their source, including
`local_llm` or `extractive`; they are not verified factual summaries. Empty model
outputs or samples that cannot fit the context budget fail explicitly.

## Description judgeability

Every group description is a routing input: the decision model judges a branch
from its label plus description, and a pruned branch is never revisited. A
group that is coherent but labelled "various documents" can therefore lose its
gold evidence permanently. Three mechanisms address this; none of them is a
calibrated semantic-quality score.

**Coverage-aware samples with provenance.** Both the local describer and the
extractive fallback draw `--description-sample-size` (default 12) documents
chosen by `select_description_samples`, not an ID prefix: the typical document
nearest the centroid, one representative per language/script stratum, the
boundary document farthest from it, then farthest-first minority coverage.
Stored semantic vectors are used when present; otherwise a content-hashed
lexical proxy is used and labelled `lexical_proxy`. The local prompt asks for
major and minority topics and sibling-distinguishing scope. When the context
budget is tight, fields are shortened before whole samples are dropped, and
every drop is recorded. Extractive descriptions give each sample an equal
character quota, so one long title cannot hide the rest.

Each group stores `description_provenance` (sampled document IDs, their roles,
selection/generation versions, generation settings, warnings such as
`context_dropped:` or `uncovered_language:`) and `description_quality`
(`weak` plus heuristic flags: `missing_or_blank`, `generic_description` for
English/Chinese boilerplate, `low_information`). Validation rejects
whitespace-only descriptions and provenance IDs that are not descendants.
`inspect` reports `weak_description_count`. Older artifacts without current
provenance stay loadable; the next `update` regenerates their descriptions
without re-embedding unchanged documents.

**Bounded child-preview recovery (serving, off by default).** With
`--preview-max-nodes N`, a candidate whose description is flagged weak or is
identical to a sibling's gets a short, explicitly sampled preview of visible
descendant evidence (first sentence plus title) appended before the ordinary
binary judgment. Beam widths and the number of judge questions are unchanged.
Per-query limits: `--preview-max-nodes`, `--preview-total-chars` (1800),
`--preview-chars-per-node` (600), `--preview-children` (4 items) and
`--preview-scan-limit` (64 visited nodes). Under an active `filters.access_acl`
previews read only the already-pruned visible subtree. `include_trace` now
also returns per-query `usage` (protocol calls/judgments and character
proxies — not tokenizer tokens) and per-stage `previewed_ids`.

**Paired routing evaluation.** Measure descriptions by whether they preserve
gold paths under a fixed tree, judge and budget, not by how they read:

```sh
python -m src.internal.document_index.description_eval_cli \
  --hierarchy indexes/nq-tree.json \
  --queries dev-queries.jsonl \
  --model /path/to/local/instruct-model \
  --output reports/descriptions.json \
  --group-beam-width 5 --topk 5
```

Each query line is `{"id", "query", "gold_passage_ids": [...], "filters"?}`;
passage IDs have the form `<document id>::passage-<n>` under the configured
segmentation, and every gold ID must be visible under its filters. Built-in
variants are `stored`, `label_only` and `extractive` (coverage-sampled titles);
`--variants FILE` adds `{name: {group_id: text}}` overrides. All variants reuse
one model instance and the same artifact — memberships are never regenerated,
and gold labels never reach a prompt. The report gives per-stage unconditional
and conditional survival (group depths, document, section, passage, choice)
with explicit denominators, complete ancestor-path survival, final passage
recall@k, cost proxies and latency, plus paired differences against `stored`
with seeded bootstrap 95% intervals. Equal budgets do not imply equal compute,
since branches differ in size. Active ACL filters bypass stored summaries, so
those queries are excluded from the macro, stage and paired comparisons and
reported separately under `acl_bypassed_macro_metrics`. Use held-out
human-checked queries: queries generated from the descriptions themselves
reward matching the generator's wording.

## Updates are full snapshots

```sh
python -m src.internal.document_index.hierarchy_cli update \
  --previous indexes/support-tree.json \
  --corpus-path data/current-corpus.jsonl \
  --output indexes/support-tree.json
```

The supplied corpus is the complete current snapshot, **not just additions**.
Missing IDs are deletions. File replacement is atomic, so updating in place is
supported. Serving processes load snapshots at startup; restart/reload the
service explicitly to use the new file.

For compatible artifacts, `update` adapts structure locally and keeps cost
proportional to change:

- Reuse unchanged document embeddings, assignments and descriptions. Embed only
  new/edited semantic documents; `update_cost.embedded_documents` counts them.
  A changed structural path reroutes the document structurally. Path-derived
  (structural) groups are never restructured.
- Assign new/edited semantic documents to the nearest existing leaf centroid,
  then plan local operations over semantic leaves only, recorded in
  `structure_operations`:
  - `spawn`: new/edited documents, plus earlier poor fits carried in
    `pending_poor_fit_ids`, that fit their leaf poorly form a new leaf when at
    least `min_cluster_size` of them are coherent (mean similarity to their
    own centroid at or above the reference) and their centroid is not close to
    any existing leaf (similarity to the nearest leaf centroid, over that
    leaf's members that are not candidates, below the reference). The second
    test keeps the poor-fit tail of an existing topic, which every batch of
    ordinary on-topic additions produces, from spawning fragments. Established
    members never spawn. Candidates are clustered with increasing cluster
    counts until a bucket passes, so unrelated poor fits do not hide a new
    topic. Poor fits that do not spawn, and spawned members that still fit
    their new leaf below the reference, are carried to the next update; each
    planner run first moves every carried poor fit to its nearest leaf
    centroid (ties broken by leaf ID), and it stops being carried once it fits
    that leaf at or above the reference. Since the reference is a low quantile,
    roughly that fraction of on-topic additions stays carried as tail
    documents.
  - `split`: a leaf above `max_cluster_size`, or one of at least twice the
    minimum size whose poor-fit fraction reaches `split_poor_fit_fraction`
    (default 0.3), is re-clustered into children.
  - `merge`: a leaf below `min_cluster_size` joins the nearest leaf it fits
    (mean member similarity to that centroid at or above the reference) and
    that stays within `max_cluster_size`, preferring siblings; otherwise it
    stays.
  - `remove`: emptied leaves are dropped; when every semantic document is
    deleted, each previous semantic group is reported removed.
  - `insert_level`: with `--max-fanout N` (default off), a parent with more
    than N children gets an intermediate level.
  - `collapse_level`: an intermediate group left with one child or none is
    removed and any child reattached to its parent, whether or not
    `--max-fanout` is set.
- When no semantic document was added, edited or deleted, the planner is
  skipped and the previous semantic groups are kept as they are.

Thresholds are relative, never absolute similarity constants. The reference is
the `fit_quantile` (default 0.05) quantile of member-to-centroid similarity,
recorded at `build` as `structure_reference` (`computed_at: "build"`) and reused
by every update; artifacts written before this field get it computed once from
their stored leaves (`computed_at: "first_update"`). `structure_reference` is
`null` when the tree has no semantic documents.

Descriptions follow a refresh policy:

- **Exempt** (always generated): new groups from spawn, split or level
  insertion, parents refreshed because of a new child, and groups whose stored
  provenance predates the current description policy or generation settings.
- **Capped refresh**: groups whose description is already stale, whose
  sampled documents were deleted, edited or moved out, or whose fresh sample
  overlaps the stored provenance `sample_ids` by less than
  `description_refresh_overlap` (Jaccard, default 0.5). Refreshes run in that
  priority order.
- **Reuse**: everything else keeps its text.

`--max-description-calls N` caps only the capped refreshes; exempt groups are
always described. A group over the cap keeps its previous text, drops removed
documents from its provenance, and is marked `description_stale: true`. At
serving time, with `--preview-max-nodes` above 0, a stale group is eligible for
child-preview recovery like a weak one. The cap binds only with a description
model; extractive descriptions cost no model calls.

`--dry-run` plans the update and prints diagnostics without writing `--output`.
New/edited documents are still embedded, but no description model is called;
`update_cost.exempt_description_calls` and `capped_description_calls` then count
the calls a real run would make, while `description_model_calls` is 0.

`update_cost` reports `embedded_documents`, `description_generations`,
`description_model_calls`, `exempt_description_calls`,
`capped_description_calls`, `deferred_descriptions` and `reused_descriptions`.
`rebuild_recommended` (with a warning) is set when the fraction of semantic
documents that fit their leaf poorly after maintenance reaches
`split_poor_fit_fraction`; `churn` still records the changed fraction. In
practice this signal rarely fires: a leaf whose poor-fit fraction reaches the
same threshold is split first, so the corpus-wide fraction reaches it only when
leaves holding many poor fits could not be split: too small (below twice
`min_cluster_size`), spawned in the same update, or degenerate. Do not rely
on it as the only cue to rebuild; compare `poor_fit_documents` across updates
or rebuild periodically.

Updates inherit the previous configuration. Changing the encoder, description
model, clustering or maintenance settings, or extraction policy requires a
fresh `build`; `update` refuses a changed configuration or embedding dimension.
Local maintenance does not guarantee the result a full k-means rebuild would
produce. There is no background job or automatic global reclustering.

The DecisionRAG paper describes how to construct a tree but has no maintenance
method; these update rules extend its construction rules (title + first
sentence embeddings, k-means, cluster descriptions) to local changes.

`examples/simulate_hierarchy_churn.py` is a manual offline measurement with
real local models: it holds out documents, adds and deletes some each day, and
records per-day and cumulative embedding and description-model call counts for
incremental `update` against a full `build` of every snapshot. With
`--decision-model` and `--queries` it also runs the description evaluator on
both final trees, reporting queries whose gold passages are absent as skipped.
It has no pass threshold.

```sh
python -m examples.simulate_hierarchy_churn \
  --corpus-path data/corpus.jsonl --days 5 --daily-add 20 --daily-delete 10 \
  --embedding-model intfloat/multilingual-e5-base \
  --description-model /path/to/local/instruct-model \
  --max-description-calls 4 --output reports/churn.json --seed 0
```

Artifacts include normalized documents, configuration, membership, fingerprints,
vectors, centroids and descriptions. They contain corpus data and should have
the same access controls as the source corpus. JSON vector storage prioritizes
portability and inspection; large corpora may need substantial disk and memory.
Programmatic injected encoder/describer implementations must represent the
identity declared in the configuration; model adapters with `model_name` are
checked for mismatches.

## Inspect granularity before choosing navigation budgets

```sh
python -m src.internal.document_index.hierarchy_cli inspect indexes/nq-tree.json
```

The report includes group sizes, imbalance, singleton groups, maximum depth,
section/path recovery sources, empty titles, estimated passage fanout, update
reuse/churn, and warnings. Passage estimates declare their word/character
settings; serving overrides may produce different fanout. Estimates and warning
thresholds are operational diagnostics, not calibrated quality measures.

Tune on representative development queries: measure ancestor/leaf recall,
ranking, latency, and memory. In particular, inspect large pages and groups
before lowering navigation budgets. No automatic retrieval-quality tuner or
gold relevance labels are provided by tree construction.

## Serve a saved hierarchy

```sh
python -m src.internal.servers.retrieval.decision \
  --hierarchy indexes/support-tree.json \
  --model /path/to/local/instruct-model \
  --group-beam-width 5 \
  --document-beam-width 10 \
  --section-beam-width 10 \
  --max-passage-chars 1200
```

`--hierarchy` is mutually exclusive with `--corpus`/`--corpus_path`. Corrupt
artifacts fail before decision-model loading. Stored documents and ACL metadata
are the serving source of truth. No builder model or embedding lookup runs
online. Group branches may have different depths; directly attached pages are
retained while deeper groups expand, then documents and passages are judged.

`--beam-width` remains the default at all internal stages; per-kind overrides
allow distinct budgets. Passages are bounded by both `--passage-words` and
`--max-passage-chars`, so no-space multilingual text is bounded too. Interior
newlines/indentation are preserved. Large section fanout is diagnosed rather
than claimed to be eliminated.

Generated descriptions may mix ACL domains. With an active `filters.access_acl`,
the retriever prunes inaccessible passages first and reconstructs group routing
text from visible descendant document titles; it does not expose stored mixed
summaries to the decision model. These caller-supplied filters remain defense
in depth, not an authentication boundary.

## Tests without downloaded weights

```sh
python -m pytest tests/unit/test_hierarchy*.py \
  tests/unit/test_hierarchy_maintenance.py \
  tests/unit/test_hierarchy_adaptive_update.py \
  tests/unit/test_hierarchy_churn.py \
  tests/unit/test_description_*.py \
  tests/unit/test_decision_retriever.py \
  tests/unit/test_local_decision_model.py \
  tests/unit/servers/retrieval/test_decision_retrieval.py
```

Tests exercise actual clustering with injected vectors, format extraction,
artifact corruption and persistence, incremental edits/deletions, local
structure maintenance, multi-day churn costs, description cache invalidation,
lazy model boundaries, CLI workflows, ragged navigation, ACL-safe routing and the existing retrieval API. They do not establish semantic
retrieval quality for a trained production model.
