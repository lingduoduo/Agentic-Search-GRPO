import copy

import numpy as np

from src.internal.document_index.hierarchy import (
    HierarchyArtifact,
    HierarchyConfig,
    build_hierarchy,
)
from tests.unit.hierarchy_fakes import TopicEmbedder, docs

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
