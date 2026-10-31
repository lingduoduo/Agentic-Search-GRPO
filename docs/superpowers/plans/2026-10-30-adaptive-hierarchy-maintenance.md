# Adaptive Hierarchy Maintenance Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `hierarchy_cli update` cost proportional to what changed while semantic groups spawn, split, merge and gain/lose levels locally, so full rebuilds stop being routine.

**Architecture:** A new torch-free module `hierarchy_maintenance.py` holds the pure structural planner (k-means partitioning, poor-fit reference, spawn/split/merge, fanout levels) over semantic group dicts and stored vectors. `build_hierarchy` delegates its semantic block to it, then applies a new description-refresh policy (sample-change test, optional cap on existing refreshes, stale marker, cost report). Serving treats stale descriptions as weak for the existing preview recovery.

**Tech Stack:** Python 3.12, numpy, scikit-learn `KMeans` (already used), pytest; no new dependencies.

**Spec:** `docs/superpowers/specs/2026-10-30-adaptive-hierarchy-maintenance-design.md`

## Global Constraints

- Existing dependencies only; no model weights or downloads in tests; injected embedders/describers count calls.
- Deterministic for a fixed snapshot sequence and `random_state` (`KMeans(random_state=config.random_state, n_init=10)`).
- Backward compatible: artifacts from before this change load and update; new `HierarchyConfig` fields have defaults so `asdict(HierarchyConfig(**previous.config))` still equals the current settings; new per-group fields are optional and validated.
- Thresholds are relative (quantiles of recorded similarity distributions), never absolute similarity constants.
- #708 contract holds: provenance IDs are current descendants; whitespace descriptions rejected.
- Path-derived (structural) groups are never restructured.
- `max_fanout` defaults to `None` (disabled): with defaults, `build` output is unchanged.
- New groups (spawn, split, level insertion) and parents refreshed because of a new child are exempt from the description cap.
- Work on branch `feat/adaptive-hierarchy`, created from `feat/description-judgeability` (this depends on #708's sampler, provenance and preview recovery).

## Review Focus

- A #707/#708-era artifact (no `structure_reference`, flat semantic groups) updates without error and without restructuring beyond size rules — test in Task 4.
- Deleting every document of a topic removes its leaf, collapses a now single-child intermediate, and leaves no dangling `parent_id` — tests in Tasks 3 and 4.
- `auto` mode with both structural and semantic documents: structural groups and their IDs are untouched by maintenance — test in Task 4.
- Tiny semantic sets (fewer documents than `min_cluster_size`, singleton groups) neither spawn nor crash — test in Task 2.
- `max_description_calls=0` with a describer: existing refreshes are deferred and marked stale with valid provenance even when sampled documents were deleted, new groups are still described — test in Task 5.

---

### Task 1: Branch, spec commit, configuration

**Files:**
- Modify: `src/internal/document_index/hierarchy.py` (`HierarchyConfig`)
- Modify: `src/internal/document_index/hierarchy_cli.py` (build parser + config construction)
- Test: `tests/unit/test_hierarchy.py`, `tests/unit/test_hierarchy_cli.py`

**Interfaces:**
- Produces: `HierarchyConfig` fields `min_cluster_size: int = 5`, `max_fanout: int | None = None`, `fit_quantile: float = 0.05`, `split_poor_fit_fraction: float = 0.3`, `description_refresh_overlap: float = 0.5`. CLI build flags `--min-cluster-size`, `--max-fanout`.

- [ ] **Step 1: Create the branch and commit the spec and this plan**

```bash
git checkout feat/description-judgeability && git checkout -b feat/adaptive-hierarchy
git add docs/superpowers/specs/2026-10-30-adaptive-hierarchy-maintenance-design.md docs/superpowers/plans/2026-10-30-adaptive-hierarchy-maintenance.md
git commit -m "docs: adaptive hierarchy maintenance design and plan"
```

- [ ] **Step 2: Write failing tests**

```python
@pytest.mark.parametrize("field,value", [
    ("min_cluster_size", 0), ("min_cluster_size", 201),  # must be <= max_cluster_size (200)
    ("max_fanout", 1), ("max_fanout", True),
    ("fit_quantile", 0.0), ("fit_quantile", 0.5),
    ("split_poor_fit_fraction", 0.0), ("split_poor_fit_fraction", 1.5),
    ("description_refresh_overlap", -0.1), ("description_refresh_overlap", 1.1),
])
def test_maintenance_config_validation(field, value):
    with pytest.raises(ValueError, match=field):
        HierarchyConfig(**{field: value})

def test_old_config_without_maintenance_fields_still_updates():
    # build, pop the five new keys from artifact.config, update with previous= : no "configuration changed" error
```

CLI test: `build --min-cluster-size 3 --max-fanout 4` stores both in `artifact.config`.

- [ ] **Step 3: Run** `pytest tests/unit/test_hierarchy.py tests/unit/test_hierarchy_cli.py -q` — Expected: new tests FAIL.
- [ ] **Step 4: Implement** the five fields with validation in `__post_init__` (error messages name the field; `max_fanout` is `None` or a non-bool int `>= 2`; `0 < fit_quantile < 0.5`; `0 < split_poor_fit_fraction <= 1`; `0 <= description_refresh_overlap <= 1`; `1 <= min_cluster_size <= max_cluster_size`) and the two CLI flags (`--max-fanout` default `None`, `--min-cluster-size` default 5).
- [ ] **Step 5: Run** the same command — Expected: PASS.
- [ ] **Step 6: Commit** `feat(hierarchy): maintenance configuration`.

---

### Task 2: Leaf-level structural planner

**Files:**
- Create: `src/internal/document_index/hierarchy_maintenance.py`
- Test: `tests/unit/test_hierarchy_maintenance.py`

**Interfaces:**
- Consumes: Task 1 config fields.
- Produces (all pure; `vectors: dict[str, list[float]]` are unit-normalized; a *leaf* is a dict `{id, parent_id, label, document_ids, centroid}`; inputs are never mutated):
  - `kmeans_partition(member_ids: list[str], vectors, count: int, max_size: int, random_state: int) -> tuple[list[list[str]], bool]` — k-means then deterministic balanced chunking of buckets over `max_size`; bool = degenerate vectors produced fewer distinct labels. Moved out of the `partition` closure in `build_hierarchy`.
  - `centroid(member_ids, vectors) -> list[float]` — normalized mean.
  - `fit_reference(leaves: list[dict], vectors, quantile: float) -> float` — the `quantile` of pooled member-to-own-centroid cosine similarity over all leaf members.
  - `maintain_leaves(leaves: list[dict], vectors, assignments: dict[str, str], config: HierarchyConfig, reference: float) -> MaintenanceResult`
  - `@dataclass MaintenanceResult: leaves: list[dict]; operations: list[dict]; forced_description_ids: set[str]; poor_fit_count: int; degenerate: bool`
  - Operation dict: `{"op": "spawn"|"split"|"merge"|"remove", "group_ids": [new or surviving ids], "source_ids": [affected prior ids], "reason": str}`.
  - New leaf IDs: `"semantic-" + _hash(sorted(member_ids))[:20]` (same scheme as today; import `_hash` from `hierarchy` is circular — copy the 3-line sha256 helper or move `_hash` into this module and import it from `hierarchy`).

`maintain_leaves` algorithm (order matters; `assignments` maps each semantic doc ID to a leaf ID already chosen — Task 4 does the nearest-centroid assignment):
1. Rebuild leaf memberships from `assignments`; recompute centroids.
2. **Spawn:** poor-fit docs = cosine(doc, own leaf centroid) `< reference`. If `>= min_cluster_size` of them: `kmeans_partition(poor, k=ceil(n/target_cluster_size))`; each bucket with size `>= min_cluster_size` and mean member-to-bucket-centroid similarity `>= reference` becomes a new leaf (members removed from their source leaves). Parent = `parent_id` of the existing leaf whose centroid is nearest the bucket centroid; add that parent to `forced_description_ids` when not `None`. Reason names the poor-fit count.
3. **Split:** a leaf with size `> max_cluster_size`, or with size `>= 2*min_cluster_size` and poor-fit fraction `>= split_poor_fit_fraction`, is re-partitioned with `k = max(2, ceil(size/target_cluster_size))`; buckets below `min_cluster_size` join the nearest other bucket. If k-means is degenerate and the leaf is not oversized, skip. Children replace the leaf under the same parent; the old ID is retired.
4. **Merge:** a leaf with `0 < size < min_cluster_size` merges into the nearest other leaf (same parent preferred, then any) if the result is `<= max_cluster_size`; the survivor keeps its ID.
5. **Remove:** empty leaves are dropped.
6. Recompute centroids; `poor_fit_count` is recounted on the final leaves.

- [ ] **Step 1: Write failing tests** (helper `unit(v)`; vectors in 4–8 dims, topics as orthogonal axes plus small fixed offsets)

```python
def test_kmeans_partition_bounds_size_and_is_deterministic(): ...  # max bucket <= max_size; two calls equal
def test_fit_reference_is_quantile_of_member_similarity():
    # two tight leaves + one member at 60° from its centroid -> reference below tight similarities, equals np.quantile of pooled values
def test_new_topic_spawns_leaf_and_forces_parent_refresh():
    # leaves A,B (parent "p"), 6 docs on a new axis assigned to A, min_cluster_size=5
    r = maintain_leaves(...)
    assert [o["op"] for o in r.operations] == ["spawn"]
    spawned = [l for l in r.leaves if l["id"] not in {"A", "B"}]
    assert set(spawned[0]["document_ids"]) == new_ids and spawned[0]["parent_id"] == "p"
    assert r.forced_description_ids == {"p"}
    assert {"A", "B"} <= {l["id"] for l in r.leaves}          # untouched IDs stable
def test_too_few_poor_fits_do_not_spawn():                     # 3 new-axis docs, min 5 -> no ops, poor_fit_count == 3
def test_drifted_leaf_splits_into_coherent_children(): ...     # leaf half axis0 half axis1, both halves >= min -> one "split", old id retired
def test_undersized_leaf_merges_into_nearest_sibling(): ...    # survivor keeps id; op "merge"
def test_emptied_leaf_is_removed(): ...
def test_tiny_semantic_set_is_stable():                        # 2 docs, one singleton leaf each, min 5 -> no spawn/split, no exception
def test_inputs_not_mutated(): ...                             # deepcopy before/after equal
```

- [ ] **Step 2: Run** `pytest tests/unit/test_hierarchy_maintenance.py -q` — Expected: FAIL (module missing).
- [ ] **Step 3: Implement** the functions above in `hierarchy_maintenance.py`. Import `KMeans` lazily inside `kmeans_partition`, as `build_hierarchy` does today.
- [ ] **Step 4: Run** — Expected: PASS.
- [ ] **Step 5: Commit** `feat(hierarchy): local spawn/split/merge planner`.

---

### Task 3: Fanout levels

**Files:**
- Modify: `src/internal/document_index/hierarchy_maintenance.py`
- Test: `tests/unit/test_hierarchy_maintenance.py`

**Interfaces:**
- Consumes: `kmeans_partition`, `centroid` (Task 2).
- Produces: `balance_levels(groups: list[dict], vectors, max_fanout: int | None, random_state: int) -> tuple[list[dict], list[dict], set[str]]` returning `(groups, operations, new_group_ids)`. `groups` are all semantic groups (leaves have `document_ids`; intermediates have `document_ids == []` and a `centroid` of their descendant documents). Intermediate IDs: `"semantic-level-" + hash(sorted(child ids))[:20]`, label `"Topic " + id[-8:]`. Ops: `"insert_level"` (group_ids = new intermediates, source_ids = moved children, reason with child count and bound), `"collapse_level"` (removed intermediate, children reparented to its parent).

Rules: with `max_fanout=None` only collapse/removal runs. Any parent (including the virtual root, `parent_id=None`, counting only semantic groups) with more than `max_fanout` semantic children gets `ceil(n/max_fanout)` new intermediates by k-means over child centroids (`kmeans_partition` with `max_size=max_fanout`); repeat at the new level until every parent is within bound. Intermediates with zero children are removed; with exactly one child are collapsed. Untouched groups keep IDs and parents.

- [ ] **Step 1: Write failing tests**

```python
def test_root_over_fanout_inserts_level_with_bounded_children():
    # 9 leaves at root, max_fanout=3 -> every parent has <= 3 children, all leaves present, ops contain "insert_level"
def test_disabled_fanout_changes_nothing_but_collapses_single_child():
    # intermediate "m" with one child leaf -> child reparented to m's parent, op "collapse_level", m absent
def test_empty_intermediate_removed_and_no_dangling_parents():
    # after removal: every parent_id is None or an existing group id
def test_levels_deterministic(): ...
```

- [ ] **Step 2: Run** `pytest tests/unit/test_hierarchy_maintenance.py -q` — Expected: new tests FAIL.
- [ ] **Step 3: Implement** `balance_levels`.
- [ ] **Step 4: Run** — Expected: PASS.
- [ ] **Step 5: Commit** `feat(hierarchy): bounded-fanout levels`.

---

### Task 4: Integrate the planner into `build_hierarchy`

**Files:**
- Modify: `src/internal/document_index/hierarchy.py` (semantic block ~lines 403–471, diagnostics, `validate`)
- Test: `tests/unit/test_hierarchy_adaptive_update.py` (new)

**Interfaces:**
- Consumes: Tasks 2–3.
- Produces: diagnostics keys `structure_reference: {"poor_fit_similarity": float, "quantile": float, "computed_at": "build"|"first_update"}`, `structure_operations: list[op dict]`, `poor_fit_documents: int`; `rebuild_recommended` redefined. Internal: a `forced_description_ids: set[str]` handed to Task 5's description loop.

Behavior:
- **Build:** `kmeans_partition` replaces the closure (same output for the same input — existing tests must pass unchanged); then `balance_levels(...)` (no-op with `max_fanout=None`); reference = `fit_reference(leaves, vectors, config.fit_quantile)`, `computed_at="build"`. No spawn/split/merge at build.
- **Update:** previous semantic groups = groups with `centroid` (leaves: no semantic children; intermediates: the rest, kept with their `parent_id`). Unchanged docs keep their previous leaf if it still exists; others go to the nearest leaf centroid (as today). Reference = previous `diagnostics["structure_reference"]` if present, else `fit_reference` over the previous leaves with `computed_at="first_update"`. Then `maintain_leaves`, then `balance_levels`. Intermediate centroids recomputed from descendant docs. `forced_description_ids` = `MaintenanceResult.forced_description_ids` plus the non-`None` parents of every group in `balance_levels`' `new_group_ids` (a parent that gained a new child).
- `rebuild_recommended = bool(previous and semantic and poor_fit_documents / len(semantic) >= config.split_poor_fit_fraction)`; replace the 30%-churn warning text with "Many semantic documents fit their cluster poorly after local maintenance; consider a full rebuild." `churn` stays reported.
- `validate`: every non-root `parent_id` exists (already true); intermediates may have empty `document_ids`.

- [ ] **Step 1: Write failing tests** (shared helpers in the test module: `TopicEmbedder` — `model_name` matching config, `calls` counter, maps a doc to a fixed unit axis chosen by its title's first word plus a small deterministic per-ID offset; `docs(topic, n, start=0)`)

```python
def test_build_with_default_config_has_flat_semantic_groups_and_no_ops():
    # default config: every semantic group has parent_id None, structure_operations == [], computed_at == "build"
    # (existing test_hierarchy*.py passing unchanged is the output-compatibility check)
def test_update_spawns_new_topic_without_reembedding_old_docs():
    base = build_hierarchy(docs("alpha",20)+docs("beta",20), HierarchyConfig(mode="cluster", n_clusters=2, min_cluster_size=5), embedder=e)
    e.calls.clear()
    new = build_hierarchy(docs("alpha",20)+docs("beta",20)+docs("gamma",8), base_config, embedder=e, previous=base)
    assert e.calls == [8]
    assert any(o["op"] == "spawn" for o in new.diagnostics["structure_operations"])
    assert len({new.assignments[d["id"]] for d in docs("gamma",8)}) == 1
    assert {g["id"] for g in base.groups} <= {g["id"] for g in new.groups}
def test_deleting_topic_removes_leaf_and_collapses_parent(): ...     # max_fanout=2 build of 3 topics, delete one topic -> no dangling parent, collapse op if single child
def test_auto_mode_structural_groups_untouched(): ...                # structural group ids/document_ids identical before/after semantic spawn
def test_pre_change_artifact_updates_with_first_update_reference():
    # build, then strip structure_reference/structure_operations from diagnostics and the five config keys;
    # update with the same docs -> no ops, computed_at == "first_update"
def test_rebuild_recommended_tracks_poor_fit_not_churn(): ...       # 40% churn within existing topics -> False
```

- [ ] **Step 2: Run** `pytest tests/unit/test_hierarchy_adaptive_update.py -q` — Expected: FAIL.
- [ ] **Step 3: Implement** the integration.
- [ ] **Step 4: Run** `pytest tests/unit/test_hierarchy*.py -q` — Expected: all PASS (existing tests unchanged).
- [ ] **Step 5: Commit** `feat(hierarchy): adaptive semantic structure on update`.

---

### Task 5: Description refresh policy, cap, stale marker, cost report, dry run

**Files:**
- Modify: `src/internal/document_index/hierarchy.py` (description loop, `build_hierarchy` signature, `validate`, diagnostics)
- Modify: `src/internal/document_index/hierarchy_cli.py` (update flags)
- Test: `tests/unit/test_hierarchy_adaptive_update.py`, `tests/unit/test_hierarchy_cli.py`

**Interfaces:**
- Consumes: `forced_description_ids` (Task 4).
- Produces: `build_hierarchy(docs, config=None, *, embedder=None, describer=None, previous=None, max_description_calls: int | None = None, dry_run: bool = False)`; optional group field `description_stale: bool` (validated as bool when present; absent means fresh); diagnostics `update_cost = {"embedded_documents", "description_generations", "description_model_calls", "exempt_description_calls", "capped_description_calls", "deferred_descriptions", "reused_descriptions"}` (ints) and `dry_run: bool`. CLI `update --max-description-calls N --dry-run`.

Policy per group, classified before any generation:
- **exempt** (always generate, outside the cap): new ID; in `forced_description_ids`; or previous provenance missing / old `selection_version` / different `generation_signature`.
- **capped refresh**, priority 0: previously `description_stale`; priority 1: a stored sample document was deleted or edited; priority 2: Jaccard overlap between the recomputed `select_description_samples(...)` IDs and the stored sample IDs `< config.description_refresh_overlap` (ties: lower overlap first, then group ID).
- **reuse** otherwise — membership changes alone do not regenerate (this removes the ancestor cascade).
- The cap counts only `capped refresh` generations and only when a describer is set (extractive generation is free). Over the cap: keep old text, set `description_stale: True`, drop deleted/edited IDs from `description_provenance.document_ids`/`roles`, append warning `"stale_samples_removed"` when any were dropped. A regenerated group has no `description_stale` key.
- `dry_run=True`: same classification and structure, no describer calls (use the extractive text for groups that would be generated), `diagnostics["dry_run"] = True`; CLI prints diagnostics and does not write `--output`. Embedding of new documents still happens (needed for planning); document this in `--help`.

- [ ] **Step 1: Write failing tests** (`CountingDescriber` with `model_name`, `describe_with_provenance` built on `select_description_samples`, `calls` list of labels)

```python
def test_small_additions_do_not_cascade_to_ancestors():
    # max_fanout build; add 1 doc to one existing topic -> describer called for no group whose sample is unchanged; update_cost["reused_descriptions"] > 0
def test_spawn_describes_new_leaf_and_its_parent_even_with_zero_cap():
    new = build_hierarchy(..., describer=d, previous=base, max_description_calls=0)
    assert {spawned_id, parent_id} <= set(d.calls)
    assert new.diagnostics["update_cost"]["exempt_description_calls"] >= 2
def test_zero_cap_defers_existing_refresh_with_valid_provenance():
    # delete a sampled doc of leaf L -> L keeps old text, description_stale is True, deleted id absent from provenance, artifact validates
def test_stale_group_refreshed_first_when_budget_returns(): ...   # next update with cap 1 regenerates the stale group, flag removed
def test_dry_run_makes_no_describer_calls(): ...                  # d.calls == [], diagnostics["dry_run"] is True
def test_invalid_stale_field_rejected(): ...                     # description_stale="yes" -> ValueError on validate
```

CLI: `update --dry-run` leaves the output path absent and prints JSON containing `structure_operations` and `update_cost`.

- [ ] **Step 2: Run** `pytest tests/unit/test_hierarchy_adaptive_update.py tests/unit/test_hierarchy_cli.py -q` — Expected: new tests FAIL.
- [ ] **Step 3: Implement** the policy and flags.
- [ ] **Step 4: Run** `pytest tests/unit/test_hierarchy*.py -q` — Expected: PASS.
- [ ] **Step 5: Commit** `feat(hierarchy): change-proportional description refresh with cap and stale marker`.

---

### Task 6: Stale descriptions are weak at serving time

**Files:**
- Modify: `src/internal/document_index/decision_retriever.py` (`_Node`, `_build_groups.build`, `_visible`, `_with_previews`)
- Test: `tests/unit/test_description_recovery.py`

**Interfaces:**
- Consumes: `description_stale` group field (Task 5).
- Produces: `_Node.stale: bool = False`; set from `group.get("description_stale", False)` unless the group has an evaluator override in `group_descriptions`; `_visible` produces `stale=False` for rebuilt visible-only group text.

- [ ] **Step 1: Write failing tests**

```python
def test_stale_specific_description_gets_preview():
    # artifact whose gold group has a specific, non-weak description but description_stale=True; preview_max_nodes=1
    # -> trace stage previewed_ids contains that group; with stale False -> previewed_ids empty
def test_override_clears_stale(): ...   # group_descriptions override for that group -> not previewed
```

- [ ] **Step 2: Run** `pytest tests/unit/test_description_recovery.py -q` — Expected: FAIL.
- [ ] **Step 3: Implement**: in `_with_previews`, `weak = assess_description(text(node))["weak"] or counts[...] > 1 or node.stale`.
- [ ] **Step 4: Run** `pytest tests/unit/test_description_recovery.py tests/unit/test_decision_retriever.py -q` — Expected: PASS.
- [ ] **Step 5: Commit** `feat(decision): treat stale group descriptions as weak for preview recovery`.

---

### Task 7: Churn simulation, offline script, docs

**Files:**
- Create: `tests/unit/test_hierarchy_churn.py`
- Create: `examples/simulate_hierarchy_churn.py`
- Modify: `docs/corpus-hierarchy.md` ("Updates are full snapshots" section; tests command), `docs/node-description-audit.md` only if a statement there becomes false

**Interfaces:**
- Consumes: everything above; `evaluate_descriptions` (#708).
- Produces: `examples/simulate_hierarchy_churn.py` CLI: `--corpus-path JSONL --days N --daily-add N --daily-delete N --embedding-model ID --description-model PATH [--max-description-calls N] [--max-fanout N] [--decision-model PATH --queries JSONL] --output FILE --seed N`. It holds out documents, adds/deletes per day, runs incremental `update` and a full `build` each day with call-counting adapter wrappers, writes per-day and cumulative embedding/description-call counts for both, and, when `--decision-model` and `--queries` are given, runs `evaluate_descriptions` on the final incremental and rebuilt trees (queries whose gold passages are absent from a tree are reported as skipped). Lazy heavy imports; `--help` loads no model. The report states it has no pass threshold.

- [ ] **Step 1: Write the failing CI simulation test** (reuse `TopicEmbedder`/`CountingDescriber` by importing from `tests/unit/test_hierarchy_adaptive_update.py` or a small `tests/unit/hierarchy_fakes.py`)

```python
def test_five_day_churn_costs_less_than_rebuilds_and_routes_new_topic():
    # day 0: alpha/beta/delta x 30; days 1-5: +5 docs to existing topics, -3 random existing; day 3 also +10 "gamma"
    # incremental embed calls == 90 + added documents; full rebuild embed calls == sum of daily corpus sizes
    # cumulative incremental description_model_calls < cumulative rebuild calls
    # exactly one spawn on day 3 holding all gamma docs; group IDs present on day 0 and untouched are present on day 5
    # evaluate_descriptions on day-5 tree with a keyword judge: gamma query complete_path_survival == 1.0
def test_simulation_script_help_loads_no_models(): ...   # subprocess --help; "torch" not in sys.modules of child (print check)
```

- [ ] **Step 2: Run** `pytest tests/unit/test_hierarchy_churn.py -q` — Expected: FAIL.
- [ ] **Step 3: Implement** the script; fix any integration gaps the simulation exposes (in their owning modules, with a regression test there).
- [ ] **Step 4: Update docs**: replace the "Updates are full snapshots" bullet list with the new behavior (local spawn/split/merge/levels, relative thresholds, description refresh policy and cap, stale marker and its serving effect, `--dry-run`, `update_cost`, new rebuild signal, when a full `build` is still required: config or encoder change); state the paper has no maintenance method and this extends its construction rules; add `test_hierarchy_maintenance.py test_hierarchy_adaptive_update.py test_hierarchy_churn.py` to the tests command.
- [ ] **Step 5: Verify**: `pytest -q` (full suite), `ruff check . && ruff format --check .` — Expected: all pass, clean.
- [ ] **Step 6: Commit** `feat(hierarchy): churn simulation, offline cost script and docs`.
- [ ] **Step 7: Whole-branch independent review**, fix findings with regression tests, re-run Step 5. Do not push or open a PR without the user's go-ahead.
