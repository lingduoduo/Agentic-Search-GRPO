import copy

import numpy as np
import pytest

from src.internal.document_index.hierarchy import (
    HierarchyArtifact,
    HierarchyConfig,
    build_hierarchy,
)
from src.internal.document_index.hierarchy_descriptions import (
    select_description_samples,
)
from tests.unit.hierarchy_fakes import CountingDescriber, TopicEmbedder, docs

MAINTENANCE_FIELDS = (
    "min_cluster_size",
    "max_fanout",
    "fit_quantile",
    "split_poor_fit_fraction",
    "description_refresh_overlap",
)


def two_topics():
    return docs("alpha", 20) + docs("beta", 20)


def ops(artifact, name):
    return [o for o in artifact.diagnostics["structure_operations"] if o["op"] == name]


def assert_tree(artifact):
    ids = {g["id"] for g in artifact.groups}
    assert all(g["parent_id"] is None or g["parent_id"] in ids for g in artifact.groups)
    parents = {g["parent_id"] for g in artifact.groups}
    leaves = {g["id"] for g in artifact.groups if g["id"] not in parents}
    assert set(artifact.assignments.values()) <= leaves


def test_build_with_default_config_has_flat_semantic_groups_and_no_ops():
    artifact = build_hierarchy(
        two_topics(), HierarchyConfig(mode="cluster"), embedder=TopicEmbedder()
    )
    assert all(g["parent_id"] is None for g in artifact.groups)
    diagnostics = artifact.diagnostics
    assert diagnostics["structure_operations"] == []
    assert diagnostics["structure_reference"]["computed_at"] == "build"
    assert diagnostics["structure_reference"]["quantile"] == 0.05
    assert diagnostics["structure_reference"]["poor_fit_similarity"] < 1
    assert diagnostics["pending_poor_fit_ids"] == []
    assert isinstance(diagnostics["poor_fit_documents"], int)
    assert diagnostics["rebuild_recommended"] is False


def test_update_spawns_new_topic_without_reembedding_old_docs():
    embedder = TopicEmbedder()
    config = HierarchyConfig(mode="cluster", n_clusters=2, min_cluster_size=5)
    base = build_hierarchy(two_topics(), config, embedder=embedder)
    snapshot = copy.deepcopy(base)
    embedder.calls.clear()
    new = build_hierarchy(
        two_topics() + docs("gamma", 8), config, embedder=embedder, previous=base
    )
    assert embedder.calls == [8]
    assert ops(new, "spawn")
    assert len({new.assignments[d["id"]] for d in docs("gamma", 8)}) == 1
    assert {g["id"] for g in base.groups} <= {g["id"] for g in new.groups}
    assert (
        new.diagnostics["structure_reference"]
        == (base.diagnostics["structure_reference"])
    )
    assert base == snapshot
    assert_tree(new)


def test_update_preserves_intermediates_and_roundtrips(tmp_path):
    embedder = TopicEmbedder()
    config = HierarchyConfig(mode="cluster", n_clusters=3, max_fanout=2)
    source = two_topics() + docs("gamma", 20)
    base = build_hierarchy(source, config, embedder=embedder)
    intermediates = {g["id"]: g for g in base.groups if not g["document_ids"]}
    assert intermediates and ops(base, "insert_level")
    updated = build_hierarchy(
        source + docs("alpha", 2, start=20), config, embedder=embedder, previous=base
    )
    for gid, old in intermediates.items():
        group = next(g for g in updated.groups if g["id"] == gid)
        assert group["parent_id"] == old["parent_id"]
        assert group["document_ids"] == []
    assert_tree(updated)
    path = tmp_path / "tree.json"
    updated.save(path)
    assert HierarchyArtifact.load(path) == updated


def test_deleting_topic_removes_leaf_and_collapses_parent():
    embedder = TopicEmbedder()
    config = HierarchyConfig(mode="cluster", n_clusters=3, max_fanout=2)
    topics = {t: docs(t, 20) for t in ("alpha", "beta", "gamma")}
    base = build_hierarchy(sum(topics.values(), []), config, embedder=embedder)
    nested = next(g for g in base.groups if g["parent_id"] and g["document_ids"])
    victim = base.documents[
        [d["id"] for d in base.documents].index(nested["document_ids"][0])
    ]["title"].split()[0]
    kept = sum((d for t, d in topics.items() if t != victim), [])
    updated = build_hierarchy(kept, config, embedder=embedder, previous=base)
    assert nested["id"] not in {g["id"] for g in updated.groups}
    assert ops(updated, "remove")
    collapsed = ops(updated, "collapse_level")
    assert collapsed and collapsed[0]["source_ids"] == [nested["parent_id"]]
    assert all(g["parent_id"] is None for g in updated.groups)
    assert_tree(updated)


def test_auto_mode_structural_groups_untouched():
    embedder = TopicEmbedder()
    structural = [
        dict(
            id=f"s{i}",
            title=f"Manual {i}",
            contents="Body.",
            metadata={"hierarchy_path": ["Guides", "Setup"]},
        )
        for i in range(4)
    ]
    config = HierarchyConfig(mode="auto", n_clusters=2)
    base = build_hierarchy(structural + two_topics(), config, embedder=embedder)
    updated = build_hierarchy(
        structural + two_topics() + docs("gamma", 8),
        config,
        embedder=embedder,
        previous=base,
    )
    assert ops(updated, "spawn")

    def structure(artifact):
        return [g for g in artifact.groups if g["id"].startswith("structure-")]

    assert structure(base) and structure(updated) == structure(base)


def test_pre_change_artifact_updates_with_first_update_reference():
    embedder = TopicEmbedder()
    config = HierarchyConfig(mode="cluster", n_clusters=2)
    base = build_hierarchy(two_topics(), config, embedder=embedder)
    for key in ("structure_reference", "structure_operations", "pending_poor_fit_ids"):
        base.diagnostics.pop(key)
    for name in MAINTENANCE_FIELDS:
        base.config.pop(name)
    updated = build_hierarchy(two_topics(), embedder=embedder, previous=base)
    assert updated.diagnostics["structure_operations"] == []
    assert updated.diagnostics["structure_reference"]["computed_at"] == "first_update"
    assert updated.assignments == base.assignments
    grown = build_hierarchy(
        two_topics() + docs("gamma", 8), embedder=embedder, previous=base
    )
    assert ops(grown, "spawn")
    assert grown.diagnostics["structure_reference"]["computed_at"] == "first_update"


def test_rebuild_recommended_tracks_poor_fit_not_churn():
    embedder = TopicEmbedder()
    config = HierarchyConfig(mode="cluster", n_clusters=2)
    base = build_hierarchy(two_topics(), config, embedder=embedder)
    edited = two_topics()
    for doc in edited[:8] + edited[20:28]:
        doc["contents"] = doc["contents"] + " Revised."
    updated = build_hierarchy(edited, config, embedder=embedder, previous=base)
    assert updated.diagnostics["churn"] == 0.4
    assert updated.diagnostics["rebuild_recommended"] is False
    assert not any("churn" in w for w in updated.diagnostics["warnings"])


def test_update_is_deterministic():
    config = HierarchyConfig(mode="cluster", n_clusters=2)
    base = build_hierarchy(two_topics(), config, embedder=TopicEmbedder())
    source = two_topics() + docs("gamma", 8) + docs("alpha", 3, start=20)
    first = build_hierarchy(source, config, embedder=TopicEmbedder(), previous=base)
    second = build_hierarchy(source, config, embedder=TopicEmbedder(), previous=base)
    assert first == second


def test_build_reports_oversize_even_when_kmeans_is_degenerate():
    class Same:
        model_name = HierarchyConfig().embedding_model

        def encode(self, texts):
            return np.ones((len(texts), 3))

    artifact = build_hierarchy(
        docs("alpha", 5),
        HierarchyConfig(mode="cluster", n_clusters=4, max_cluster_size=2),
        embedder=Same(),
    )
    assert len(artifact.groups) == 3
    warnings = artifact.diagnostics["warnings"]
    assert any(w.startswith("Degenerate semantic vectors") for w in warnings)
    assert any(w.startswith("Oversized semantic cluster") for w in warnings)


def test_update_keeps_unrelated_undersized_leaves():
    embedder = TopicEmbedder()
    config = HierarchyConfig(mode="cluster", n_clusters=3, min_cluster_size=5)
    source = two_topics() + docs("gamma", 3)
    base = build_hierarchy(source, config, embedder=embedder)
    small = next(g for g in base.groups if len(g["document_ids"]) == 3)
    updated = build_hierarchy(
        source + docs("alpha", 1, start=20), config, embedder=embedder, previous=base
    )
    kept = next((g for g in updated.groups if g["id"] == small["id"]), None)
    assert kept and kept["document_ids"] == small["document_ids"]
    assert not ops(updated, "merge")


DESCRIBED = dict(description_model=CountingDescriber.model_name)


def three_topics():
    return docs("alpha", 20) + docs("beta", 20) + docs("gamma", 20)


def described_base(source, **config):
    config = HierarchyConfig(mode="cluster", **DESCRIBED, **config)
    describer = CountingDescriber()
    base = build_hierarchy(
        source, config, embedder=TopicEmbedder(), describer=describer
    )
    describer.calls.clear()
    return config, describer, base


def called_ids(artifact, describer):
    by_label = {g["label"]: g["id"] for g in artifact.groups}
    return {by_label[label] for label in describer.calls}


def group(artifact, gid):
    return next(g for g in artifact.groups if g["id"] == gid)


def assert_cost_consistent(artifact, describer=None):
    cost = artifact.diagnostics["update_cost"]
    assert set(cost) == {
        "embedded_documents",
        "description_generations",
        "description_model_calls",
        "exempt_description_calls",
        "capped_description_calls",
        "deferred_descriptions",
        "reused_descriptions",
    }
    assert all(type(v) is int and v >= 0 for v in cost.values())
    assert cost["description_generations"] == (
        cost["exempt_description_calls"] + cost["capped_description_calls"]
    )
    assert cost["description_generations"] + cost["deferred_descriptions"] + cost[
        "reused_descriptions"
    ] == len(artifact.groups)
    if describer is not None:
        assert cost["description_model_calls"] == len(describer.calls)


def delete_sampled(source, artifact, gid):
    sampled = group(artifact, gid)["description_provenance"]["document_ids"][0]
    return sampled, [d for d in source if d["id"] != sampled]


def test_small_additions_do_not_cascade_to_ancestors():
    config, describer, base = described_base(three_topics(), n_clusters=3, max_fanout=2)
    intermediates = {g["id"] for g in base.groups if not g["document_ids"]}
    assert intermediates
    new = build_hierarchy(
        three_topics() + docs("alpha", 1, start=20),
        config,
        embedder=TopicEmbedder(),
        describer=describer,
        previous=base,
    )
    called = called_ids(new, describer)
    assert not called & intermediates
    old = {g["id"]: g for g in base.groups}
    for g in new.groups:
        if g["id"] in called:
            assert (
                g["description_provenance"]["document_ids"]
                != old[g["id"]]["description_provenance"]["document_ids"]
            )
    assert new.diagnostics["update_cost"]["reused_descriptions"] > 0
    assert new.diagnostics["update_cost"]["embedded_documents"] == 1
    assert_cost_consistent(new, describer)


def test_spawn_describes_new_leaf_and_its_parent_even_with_zero_cap():
    config, describer, base = described_base(three_topics(), n_clusters=3, max_fanout=2)
    new = build_hierarchy(
        three_topics() + docs("delta", 8),
        config,
        embedder=TopicEmbedder(),
        describer=describer,
        previous=base,
        max_description_calls=0,
    )
    (spawn,) = ops(new, "spawn")
    (spawned_id,) = spawn["group_ids"]
    parent_id = group(new, spawned_id)["parent_id"]
    assert parent_id is not None
    assert {spawned_id, parent_id} <= called_ids(new, describer)
    cost = new.diagnostics["update_cost"]
    assert cost["exempt_description_calls"] >= 2
    assert cost["capped_description_calls"] == 0
    assert not any(g.get("description_stale") for g in new.groups)
    assert_tree(new)
    assert_cost_consistent(new, describer)


def test_zero_cap_defers_existing_refresh_with_valid_provenance(tmp_path):
    config, describer, base = described_base(two_topics(), n_clusters=2)
    leaf = base.groups[0]
    sampled, source = delete_sampled(two_topics(), base, leaf["id"])
    new = build_hierarchy(
        source,
        config,
        embedder=TopicEmbedder(),
        describer=describer,
        previous=base,
        max_description_calls=0,
    )
    assert describer.calls == []
    stale = group(new, leaf["id"])
    assert stale["description"] == leaf["description"]
    assert stale["description_stale"] is True
    provenance = stale["description_provenance"]
    assert sampled not in provenance["document_ids"]
    assert sampled not in provenance["roles"]
    assert sampled not in provenance["sample_ids"]
    assert "stale_samples_removed" in provenance["warnings"]
    other = group(new, base.groups[1]["id"])
    assert "description_stale" not in other
    assert new.diagnostics["update_cost"]["deferred_descriptions"] == 1
    assert_cost_consistent(new, describer)
    path = tmp_path / "tree.json"
    new.save(path)
    assert HierarchyArtifact.load(path) == new


def test_edited_sample_document_is_refreshed_when_uncapped():
    config, describer, base = described_base(two_topics(), n_clusters=2)
    leaf = base.groups[0]
    sampled = leaf["description_provenance"]["document_ids"][0]
    source = two_topics()
    for doc in source:
        if doc["id"] == sampled:
            doc["contents"] += " Revised."
    new = build_hierarchy(
        source, config, embedder=TopicEmbedder(), describer=describer, previous=base
    )
    assert called_ids(new, describer) == {leaf["id"]}
    assert new.diagnostics["update_cost"]["capped_description_calls"] == 1
    assert_cost_consistent(new, describer)


def test_stale_group_refreshed_first_when_budget_returns():
    config, describer, base = described_base(two_topics(), n_clusters=2)
    first, second = (g["id"] for g in base.groups)
    _, source = delete_sampled(two_topics(), base, first)
    stale = build_hierarchy(
        source,
        config,
        embedder=TopicEmbedder(),
        describer=describer,
        previous=base,
        max_description_calls=0,
    )
    assert group(stale, first)["description_stale"] is True
    # The second leaf now needs a priority-1 refresh too; the stale one wins.
    _, source = delete_sampled(source, stale, second)
    refreshed = build_hierarchy(
        source,
        config,
        embedder=TopicEmbedder(),
        describer=describer,
        previous=stale,
        max_description_calls=1,
    )
    assert called_ids(refreshed, describer) == {first}
    assert "description_stale" not in group(refreshed, first)
    assert group(refreshed, second)["description_stale"] is True
    cost = refreshed.diagnostics["update_cost"]
    assert cost["capped_description_calls"] == 1
    assert cost["deferred_descriptions"] == 1
    assert_cost_consistent(refreshed, describer)


def test_cap_is_free_without_describer():
    config = HierarchyConfig(mode="cluster", n_clusters=2)
    base = build_hierarchy(two_topics(), config, embedder=TopicEmbedder())
    _, source = delete_sampled(two_topics(), base, base.groups[0]["id"])
    new = build_hierarchy(
        source,
        config,
        embedder=TopicEmbedder(),
        previous=base,
        max_description_calls=0,
    )
    assert not any("description_stale" in g for g in new.groups)
    cost = new.diagnostics["update_cost"]
    assert cost["capped_description_calls"] == 1
    assert cost["description_model_calls"] == 0
    assert_cost_consistent(new)


def test_dry_run_makes_no_describer_calls():
    config, describer, base = described_base(three_topics(), n_clusters=3, max_fanout=2)
    snapshot = copy.deepcopy(base)
    embedder = TopicEmbedder()
    new = build_hierarchy(
        three_topics() + docs("delta", 8),
        config,
        embedder=embedder,
        describer=describer,
        previous=base,
        dry_run=True,
    )
    assert describer.calls == []
    assert embedder.calls == [8]
    assert new.diagnostics["dry_run"] is True
    cost = new.diagnostics["update_cost"]
    assert cost["description_model_calls"] == 0
    assert cost["exempt_description_calls"] >= 2
    assert ops(new, "spawn")
    assert base == snapshot
    assert_cost_consistent(new, describer)
    real = build_hierarchy(
        three_topics() + docs("delta", 8),
        config,
        embedder=TopicEmbedder(),
        describer=describer,
        previous=base,
    )
    assert real.diagnostics["dry_run"] is False
    assert (
        real.diagnostics["structure_operations"]
        == (new.diagnostics["structure_operations"])
    )
    assert (
        real.diagnostics["update_cost"]["exempt_description_calls"]
        == (cost["exempt_description_calls"])
    )


def test_invalid_stale_field_rejected():
    artifact = build_hierarchy(
        two_topics(), HierarchyConfig(mode="cluster"), embedder=TopicEmbedder()
    )
    artifact.groups[0]["description_stale"] = "yes"
    with pytest.raises(ValueError, match="description_stale"):
        artifact.validate()
    artifact.groups[0]["description_stale"] = False
    artifact.validate()


class PackingDescriber(CountingDescriber):
    """Keeps only two sample IDs, as context packing can."""

    def describe_with_provenance(self, label, documents, *, vectors=None, sample_size):
        result = super().describe_with_provenance(
            label, documents, vectors=vectors, sample_size=sample_size
        )
        keep = result["document_ids"][:2]
        result["document_ids"] = keep
        result["roles"] = {k: result["roles"][k] for k in keep}
        return result


def test_packed_provenance_does_not_trigger_overlap_refresh():
    config = HierarchyConfig(mode="cluster", n_clusters=3, **DESCRIBED)
    describer = PackingDescriber()
    source = three_topics()
    previous = build_hierarchy(
        source, config, embedder=TopicEmbedder(), describer=describer
    )
    for group_ in previous.groups:
        provenance = group_["description_provenance"]
        assert len(provenance["document_ids"]) == 2
        assert len(provenance["sample_ids"]) == config.description_sample_size
    for step in range(3):
        describer.calls.clear()
        source = source + docs("alpha", 1, start=20 + step)
        previous = build_hierarchy(
            source,
            config,
            embedder=TopicEmbedder(),
            describer=describer,
            previous=previous,
        )
        assert describer.calls == []
        assert previous.diagnostics["update_cost"]["capped_description_calls"] == 0


def test_sample_ids_recorded_on_every_generation_path():
    config = HierarchyConfig(mode="cluster", n_clusters=2)
    extractive = build_hierarchy(two_topics(), config, embedder=TopicEmbedder())

    class Plain:
        def describe(self, label, documents):
            return "Plain summary of " + label

    plain = build_hierarchy(
        two_topics(), config, embedder=TopicEmbedder(), describer=Plain()
    )
    for artifact in (extractive, plain):
        for group_ in artifact.groups:
            provenance = group_["description_provenance"]
            assert len(provenance["sample_ids"]) == config.description_sample_size
            assert set(provenance["document_ids"]) <= set(provenance["sample_ids"])


def test_invalid_sample_ids_rejected():
    artifact = build_hierarchy(
        two_topics(), HierarchyConfig(mode="cluster"), embedder=TopicEmbedder()
    )
    provenance = artifact.groups[0]["description_provenance"]
    for bad in ("x", [1], ["missing-doc"], [provenance["sample_ids"][0]] * 2):
        provenance["sample_ids"] = bad
        with pytest.raises(ValueError, match="provenance"):
            artifact.validate()
    del provenance["sample_ids"]
    artifact.validate()


def test_pre_sample_ids_artifact_falls_back_to_document_ids():
    config, describer, base = described_base(two_topics(), n_clusters=2)
    for group_ in base.groups:
        del group_["description_provenance"]["sample_ids"]
    base.validate()
    new = build_hierarchy(
        two_topics() + docs("alpha", 1, start=20),
        config,
        embedder=TopicEmbedder(),
        describer=describer,
        previous=base,
    )
    assert describer.calls == []
    assert new.diagnostics["update_cost"]["reused_descriptions"] == 2


def test_lowest_overlap_refreshed_first_under_cap():
    config, describer, base = described_base(
        two_topics(), n_clusters=2, description_refresh_overlap=1.0
    )
    source = two_topics() + docs("alpha", 15, start=20) + docs("beta", 15, start=20)
    new = build_hierarchy(
        source,
        config,
        embedder=TopicEmbedder(),
        describer=describer,
        previous=base,
        max_description_calls=1,
    )
    assert not ops(new, "spawn") and not ops(new, "split")
    by_id = {d["id"]: d for d in new.documents}
    overlaps = {}
    for old in base.groups:
        current = group(new, old["id"])
        fresh = {
            s["document_id"]
            for s in select_description_samples(
                [by_id[i] for i in current["document_ids"]],
                vectors=new.vectors,
                limit=config.description_sample_size,
            )
        }
        stored = set(old["description_provenance"]["sample_ids"])
        overlaps[old["id"]] = len(fresh & stored) / len(fresh | stored)
    assert all(v < 1.0 for v in overlaps.values())
    first, second = sorted(overlaps, key=lambda gid: (overlaps[gid], gid))
    assert called_ids(new, describer) == {first}
    assert "description_stale" not in group(new, first)
    assert group(new, second)["description_stale"] is True
    cost = new.diagnostics["update_cost"]
    assert cost["capped_description_calls"] == 1
    assert cost["deferred_descriptions"] == 1
    assert_cost_consistent(new, describer)
    new.validate()


@pytest.mark.parametrize(
    "reference",
    [
        {"quantile": 0.05, "computed_at": "build"},
        {"poor_fit_similarity": "0.9", "quantile": 0.05, "computed_at": "build"},
        {"poor_fit_similarity": float("nan"), "quantile": 0.05, "computed_at": "build"},
        {"poor_fit_similarity": True, "quantile": 0.05, "computed_at": "build"},
        {"poor_fit_similarity": 0.9, "quantile": "low", "computed_at": "build"},
        {"poor_fit_similarity": 0.9, "computed_at": "build"},
        {"poor_fit_similarity": 0.9, "quantile": 0.05, "computed_at": "later"},
        [0.9],
    ],
)
def test_invalid_structure_reference_rejected(reference):
    artifact = build_hierarchy(
        two_topics(), HierarchyConfig(mode="cluster"), embedder=TopicEmbedder()
    )
    artifact.diagnostics["structure_reference"] = reference
    with pytest.raises(ValueError, match="structure_reference"):
        artifact.validate()


def test_absent_or_null_structure_reference_accepted():
    artifact = build_hierarchy(
        two_topics(), HierarchyConfig(mode="cluster"), embedder=TopicEmbedder()
    )
    artifact.diagnostics["structure_reference"] = None
    artifact.validate()
    del artifact.diagnostics["structure_reference"]
    artifact.validate()


@pytest.mark.parametrize(
    "pending",
    ["alpha-000", ["alpha-000", "alpha-000"], [1], ["missing"], ["s-structural"]],
)
def test_invalid_pending_poor_fit_ids_rejected(pending):
    structural = dict(
        id="s-structural",
        title="Manual 0",
        contents="Body.",
        metadata={"hierarchy_path": ["Guides", "Setup"]},
    )
    artifact = build_hierarchy(
        two_topics() + [structural],
        HierarchyConfig(mode="auto", n_clusters=2),
        embedder=TopicEmbedder(),
    )
    assert "centroid" not in next(
        g for g in artifact.groups if "s-structural" in g["document_ids"]
    )
    artifact.diagnostics["pending_poor_fit_ids"] = pending
    with pytest.raises(ValueError, match="pending_poor_fit_ids"):
        artifact.validate()
    artifact.diagnostics["pending_poor_fit_ids"] = ["alpha-000"]
    artifact.validate()


def test_deleting_every_semantic_document_reports_removed_groups():
    embedder = TopicEmbedder()
    config = HierarchyConfig(mode="cluster", n_clusters=3, max_fanout=2)
    base = build_hierarchy(two_topics() + docs("gamma", 20), config, embedder=embedder)
    semantic = sorted(g["id"] for g in base.groups if "centroid" in g)
    updated = build_hierarchy([], config, embedder=embedder, previous=base)
    removed = ops(updated, "remove")
    assert sorted(i for o in removed for i in o["source_ids"]) == semantic
    assert all(o["group_ids"] == [] for o in removed)
