# Adaptive hierarchy maintenance

## Intent

A corpus hierarchy built once offline (as in the DecisionRAG paper's Natural
Questions tree, §3.4 / §4.1 / App. F) goes stale on a knowledge base that changes
daily. Rebuilding it — embedding every document and having the LLM rewrite every
group description — is a recurring cost that grows with corpus size, not with
the size of the change.

`hierarchy_cli update` (PR #707) already re-embeds only new or edited documents.
Two problems remain:

1. **Description calls cascade.** Any membership change regenerates that group's
   description and every ancestor's (`build_hierarchy`, description loop:
   reuse requires identical descendant membership). Small daily changes spread
   across the tree rewrite most upper-level descriptions every day.
2. **Structure is frozen.** New semantic documents can only join the nearest
   existing leaf centroid. A new topic is absorbed into an unrelated cluster
   (whose description then misroutes it), shrinking clusters never merge,
   oversized clusters are cut into balanced chunks rather than re-clustered, and
   the only remedy is a full rebuild, recommended at 30% churn.

The goal: an `update` whose cost is proportional to what changed, while the tree
adapts its shape locally so full rebuilds stop being routine. The paper describes
no maintenance method; this applies its construction rules (structure first,
recursive k-means where there is none, non-uniform depth with bounded fanout,
typicality-ordered descriptions) locally and repeatedly. That extension is ours,
not the paper's.

## Goals

- **Cost proportional to change.** Embedding calls = new + edited semantic
  documents (unchanged from today). Description-generation calls = groups that
  were restructured or whose content materially changed, not every group with a
  touched descendant. Every update reports its actual counts.
- **Local structural adaptation** for semantic (clustered) groups, using stored
  vectors only:
  - *Spawn:* new documents that fit no existing cluster well form a new cluster
    once enough of them accumulate; until then they join the nearest cluster and
    are reported as poor fits.
  - *Split:* a cluster that is oversized or whose cohesion has degraded is
    re-clustered with local k-means. Balanced chunking remains only as the
    fallback for degenerate vectors.
  - *Merge/retire:* an undersized cluster merges into its nearest sibling when
    the result stays within `max_cluster_size`; empty clusters are removed.
  - *Levels:* when a parent (including the root level of semantic groups) has
    more children than a configured fanout bound *b*, insert an intermediate
    level by k-means over child centroids; when an intermediate group falls to a
    single child, remove it. Bounded fanout keeps depth ≈ log_b N (paper Eq. 4).
- **Bounded, prioritized description refresh.** A group's description is
  regenerated when it is new or restructured, when a document in its stored
  description sample was deleted or edited, or when its recomputed coverage
  sample (the #708 sampler) differs materially from the stored sample. An
  optional per-update cap on description calls, highest-priority groups first;
  groups over the cap keep their old text and are explicitly marked stale.
- **Stability.** Groups untouched by a structural operation keep their IDs,
  labels, descriptions and provenance. Every structural operation is recorded
  with its reason.
- **Dry run.** `update --dry-run` reports planned structural operations and the
  embedding/description calls the update would make, without writing the
  artifact.
- **Evidence.** An offline multi-day churn simulation compares cumulative
  embedding and description calls of repeated incremental updates against
  repeated full rebuilds, and checks routing with the #708 evaluator on the same
  queries.

## Non-goals

- Restructuring path-derived (structural) groups: their shape follows the
  documents' paths, as today. Their descriptions do get the new refresh policy.
- Delta-only input: `update` still takes the full current snapshot (missing IDs
  are deletions).
- Live serving reload, background jobs, or scheduling.
- A global re-clustering inside `update`. Full `build` stays available and
  still is the answer to configuration or embedding-model changes.
- Claims that incremental trees match full-rebuild quality in general.

## Anti-goals (failure even if it "works")

- Routing quality silently degrades: a stale or restructured description that is
  not marked, or a structural change with no recorded reason.
- Untouched groups churn IDs or descriptions (breaks saved evaluation reports
  and caches, and wastes LLM calls).
- Cost claims without measured counts.
- Thresholds that are magic absolute constants tuned on one synthetic corpus.

## Decisions made in conversation

- Structural adaptation runs automatically during `update`; `--dry-run` shows
  the plan first (Claude's call, user approved the overall proposal).
- Untouched group IDs stay stable (Claude's call, approved).
- Follow the paper's construction ideas, applied locally; record that the
  paper itself has no maintenance method.
- **The description cap exempts new groups (user's decision).** Groups created
  by spawn, split or level insertion, and the parents refreshed because of a
  new child, always get a real description. The optional cap only defers
  refreshes of existing descriptions, which keep their old text and are marked
  `description_stale`. On days with many new topics, calls can exceed the cap;
  dry run and the cost report show exempt and capped calls separately.

## Calls made after builder review (Claude's, not the user's)

- **Fanout bound applies to build and update alike, off by default.** A new
  `max_fanout` (*b*) defaults to disabled, so the documented flat
  `--clusters 1000` NQ recipe and existing outputs are unchanged. When set,
  full `build` also produces the multi-level tree (recursive k-means over
  child centroids where a parent exceeds *b*), so incremental and rebuilt trees
  have the same shape rules and the churn simulation compares maintenance, not
  tree shape. With an explicit `--clusters`, *b* bounds the levels above those
  leaf clusters.
- **Old artifacts are not restructured wholesale on first update.** Missing
  statistics are computed at that update, and size-based operations (oversize
  split, undersize merge, empty removal) apply immediately. Cohesion-based
  split and spawn thresholds use a corpus-wide reference (the distribution of
  member-to-own-centroid similarity across all semantic groups), not each
  group's own current value, so an already-degraded cluster is not recorded
  as its own normal. Level insertion happens only if `max_fanout` is set,
  which is a config change and therefore already requires a full `build`.
- **Stale descriptions count as weak at serving time.** The decision retriever
  treats a group marked `description_stale` like a weak description for the
  existing bounded child-preview recovery (`--preview-max-nodes`). Nothing else
  in serving changes.
- **Simulation has two parts.** A CI test with injected embedder, describer
  and deterministic judge asserts call counts, structural operations and a
  well-formed evaluator report. A manual offline script runs the same churn
  with a real local model and writes a report; it has no pass threshold and
  does not gate the work, consistent with the non-goal on quality claims.
- **Parents of a new child are refreshed.** When spawn or level insertion adds
  a new child under a parent, the parent's description is regenerated so the
  new topic is routable from above. Splits and merges among existing children
  re-check the parent only through the sample-change test. Spawns are rare
  relative to additions, so this does not reintroduce the per-addition
  cascade.

## Constraints

- Existing dependencies only (numpy, scikit-learn KMeans already used); no model
  weights or downloads in tests; injected embedders/describers count calls.
- Deterministic for a fixed snapshot sequence and `random_state`.
- Backward compatible: artifacts written before this change load and update
  (new config fields have defaults and must canonicalize so old configs don't
  trigger the "configuration changed" rebuild error; new per-group fields are
  optional). Artifact validation covers all new fields.
- Thresholds are relative to statistics recorded when a group was created
  (e.g. its member-to-centroid similarity distribution), so they adapt to the
  embedding model and corpus; defaults are documented, configurable, and part
  of `HierarchyConfig`.
- Keep the #708 description provenance/quality contract: every description's
  provenance IDs are current descendants; whitespace descriptions rejected.
- Updates of structural-only corpora behave as today apart from the description
  refresh policy.

## Left to the builder

- Exact cohesion statistic, spawn/split/merge thresholds and their defaults, and
  the "material sample change" measure (e.g. sample-set overlap), subject to the
  relative-threshold constraint.
- Priority order for capped description refresh (existing descriptions only).
- How stale descriptions with deleted sampled documents keep valid provenance
  (e.g. drop deleted IDs with a warning) when the cap defers regeneration.
- Representation of the structural change log and cost report in diagnostics.
- Whether the 30% churn warning is replaced by a restructuring-based signal.
- Simulation corpus design and where the script lives.
