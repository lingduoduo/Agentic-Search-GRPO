import copy
import json

import numpy as np
import pytest

from src.internal.document_index.hierarchy import (
    HierarchyArtifact,
    HierarchyConfig,
    build_hierarchy,
)


def docs(n=6, path=None):
    return [
        dict(
            id=str(i),
            title=f"Title {i}",
            contents=f"Text {i}. More.",
            metadata={"hierarchy_path": path} if path else {},
        )
        for i in range(n)
    ]


class Encoder:
    def __init__(self):
        self.calls = []

    def encode(self, texts):
        self.calls.extend(texts)
        return np.array([[1, int(t.split()[1]) % 2] for t in texts], dtype=float)


def test_config_validation():
    for kw in (
        {"mode": "bad"},
        {"max_cluster_size": 0},
        {"n_clusters": 0},
        {"target_cluster_size": True},
    ):
        with pytest.raises(ValueError):
            HierarchyConfig(**kw)


def test_structure_roundtrip_and_corruption(tmp_path):
    artifact = build_hierarchy(docs(path=["A", "B"]), HierarchyConfig(mode="structure"))
    assert len(artifact.groups) == 2
    assert len(artifact.assignments) == 6
    path = tmp_path / "tree.json"
    artifact.save(path)
    assert HierarchyArtifact.load(path) == artifact
    data = json.loads(path.read_text())
    data["groups"][0]["parent_id"] = data["groups"][1]["id"]
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        HierarchyArtifact.load(path)


def test_semantic_incremental_reuse_and_overflow():
    encoder = Encoder()
    config = HierarchyConfig(
        target_cluster_size=2, max_cluster_size=2, min_cluster_size=1
    )
    artifact = build_hierarchy(docs(), config, embedder=encoder)
    assert all(len(g["document_ids"]) <= 2 for g in artifact.groups)
    snapshot = copy.deepcopy(artifact)
    updated = build_hierarchy(docs(), config, embedder=encoder, previous=artifact)
    assert updated.assignments == artifact.assignments
    assert len(encoder.calls) == 6
    assert artifact == snapshot
    updated = build_hierarchy(docs(7)[1:], config, embedder=encoder, previous=artifact)
    assert len(encoder.calls) == 7
    assert "0" not in updated.assignments
    assert updated.diagnostics["deleted_documents"] == 1
    with pytest.raises(ValueError, match="rebuild"):
        build_hierarchy(
            docs(), HierarchyConfig(embedding_model="other"), previous=artifact
        )


def test_descriptions_dirty_only_and_structure_edit():
    class Describer:
        calls = []

        def describe(self, label, documents):
            self.calls.append(label)
            return label + " summary"

    describer = Describer()
    artifact = build_hierarchy(docs(2, ["A"]), describer=describer)
    assert artifact.groups[0]["description_source"] == "llm"
    build_hierarchy(docs(2, ["A"]), describer=describer, previous=artifact)
    assert len(describer.calls) == 1
    changed = docs(2, ["B"])
    updated = build_hierarchy(changed, describer=describer, previous=artifact)
    assert [g["label"] for g in updated.groups] == ["B"]


def test_identical_vectors_hard_bound():
    class Same:
        def encode(self, texts):
            return np.ones((len(texts), 2))

    artifact = build_hierarchy(
        docs(9),
        HierarchyConfig(max_cluster_size=2, min_cluster_size=1),
        embedder=Same(),
    )
    assert max(len(g["document_ids"]) for g in artifact.groups) <= 2
    assert artifact.diagnostics["warnings"]


@pytest.mark.parametrize(
    "corrupt",
    [
        lambda a: a.assignments.update({"0": "missing"}),
        lambda a: a.groups[0]["document_ids"].append("0"),
        lambda a: a.vectors.update({"ghost": [1.0, 0.0]}),
        lambda a: a.groups[0].update(centroid=[float("nan")]),
        lambda a: a.groups[0].update(centroid=[1.0, 2.0, 3.0]),
    ],
)
def test_artifact_rejects_corruption(corrupt):
    artifact = build_hierarchy(docs(2), embedder=Encoder())
    corrupt(artifact)
    with pytest.raises(ValueError):
        artifact.validate()


def test_edit_embeds_only_changed_and_does_not_mutate_inputs():
    source = docs(4)
    original = copy.deepcopy(source)
    encoder = Encoder()
    config = HierarchyConfig(
        max_cluster_size=2, target_cluster_size=2, min_cluster_size=1
    )
    artifact = build_hierarchy(source, config, embedder=encoder)
    assert source == original
    source[0]["contents"] = "Edited content. Extra."
    updated = build_hierarchy(source, config, previous=artifact, embedder=encoder)
    assert len(encoder.calls) == 5
    assert updated.diagnostics["changed_documents"] == 1
    assert updated.diagnostics["reused_documents"] == 3
    assert all(len(g["document_ids"]) <= 2 for g in updated.groups)


def test_invalid_embedding_and_empty_corpus():
    class Bad:
        def encode(self, texts):
            return np.zeros((len(texts), 2))

    with pytest.raises(ValueError, match="zero"):
        build_hierarchy(docs(), embedder=Bad())
    assert build_hierarchy([]).documents == []


def test_flat_descendant_memberships_scan_groups_once():
    from src.internal.document_index.hierarchy import _descendant_memberships

    class CountedGroups(dict):
        visits = 0

        def values(self):
            for value in super().values():
                self.visits += 1
                yield value

    groups = CountedGroups(
        {
            str(i): {"id": str(i), "parent_id": None, "document_ids": [str(i)]}
            for i in range(2000)
        }
    )
    members = _descendant_memberships(groups)
    assert members["100"] == ["100"]
    assert groups.visits <= 2 * len(groups)


def test_nested_descriptions_refresh_only_dirty_ancestors():
    class Describer:
        def __init__(self):
            self.calls = []

        def describe(self, label, documents):
            self.calls.append(label)
            return label

    source = docs(2, ["Root", "Left"]) + [
        dict(
            id="right",
            title="Right",
            contents="Other.",
            metadata={"hierarchy_path": ["Root", "Right"]},
        )
    ]
    describer = Describer()
    artifact = build_hierarchy(source, describer=describer)
    describer.calls.clear()
    source[0]["contents"] = "Changed."
    build_hierarchy(source, previous=artifact, describer=describer)
    assert sorted(describer.calls) == ["Left", "Root"]


def test_injected_declared_model_identity_must_match():
    encoder = Encoder()
    encoder.model_name = "different-encoder"
    with pytest.raises(ValueError, match="embedding_model"):
        build_hierarchy(docs(2), embedder=encoder)


def test_granularity_diagnostics_include_explicit_passage_estimate_settings():
    source = docs(8, ["Large"])
    source[0]["contents"] = " ".join(["paragraph"] * 13000)
    source.append(
        dict(
            id="small",
            title="Small",
            contents="Short.",
            metadata={"hierarchy_path": ["Small"]},
        )
    )
    artifact = build_hierarchy(source, HierarchyConfig(mode="structure"))
    diagnostics = artifact.diagnostics
    assert diagnostics["group_size_min"] == 1
    assert diagnostics["group_size_mean"] == 4.5
    assert diagnostics["group_size_max"] == 8
    assert diagnostics["singleton_group_count"] == 1
    assert diagnostics["imbalance_ratio"] == 8.0
    assert diagnostics["passage_estimate_settings"] == {
        "max_words": 200,
        "max_chars": 1200,
    }
    assert diagnostics["estimated_max_passages_per_document"] > 60
    assert diagnostics["estimated_max_passages_per_section"] > 60
    assert any("60 passages" in warning for warning in diagnostics["warnings"])
    assert any("imbalanced" in warning for warning in diagnostics["warnings"])


def test_old_schema_policy_upgrade_preserves_embeddings_and_validates_provenance():
    import copy
    import pytest
    from src.internal.document_index.hierarchy import HierarchyConfig, build_hierarchy

    docs = [
        {
            "id": "a",
            "title": "Topic A",
            "contents": "Details",
            "structure_path": ["Guide"],
        }
    ]
    artifact = build_hierarchy(docs, HierarchyConfig(mode="structure"))
    old = copy.deepcopy(artifact)
    old.config.pop("description_sample_size")
    for group in old.groups:
        group.pop("description_provenance")
        group.pop("description_quality")
        group["description"] = "Old description"
    upgraded = build_hierarchy(docs, previous=old)
    assert upgraded.groups[0]["description"] != "Old description"
    assert upgraded.groups[0]["description_provenance"]["document_ids"] == ["a"]
    invalid = copy.deepcopy(upgraded)
    invalid.groups[0]["description_provenance"]["document_ids"] = ["missing"]
    with pytest.raises(ValueError, match="provenance"):
        invalid.validate()
    invalid = copy.deepcopy(upgraded)
    invalid.groups[0]["description"] = "   "
    with pytest.raises(ValueError, match="description"):
        invalid.validate()


def test_old_semantic_description_policy_refresh_does_not_reembed():
    encoder = Encoder()
    artifact = build_hierarchy(
        docs(), HierarchyConfig(mode="cluster", n_clusters=1), embedder=encoder
    )
    encoder.calls.clear()
    for group in artifact.groups:
        group.pop("description_provenance")
    refreshed = build_hierarchy(docs(), embedder=encoder, previous=artifact)
    assert encoder.calls == []
    assert refreshed.vectors == artifact.vectors
    assert all("description_provenance" in group for group in refreshed.groups)


def test_generation_settings_record_config_sample_size_and_null_provenance_updates():
    import copy
    from src.internal.document_index.hierarchy import HierarchyConfig, build_hierarchy
    from src.internal.document_index.hierarchy_descriptions import (
        select_description_samples,
    )

    class Describer:
        sample_size = 12
        model_name = "m"

        def describe_with_provenance(self, label, docs, *, vectors=None, sample_size):
            samples = select_description_samples(docs, limit=sample_size)
            return dict(
                text="Topic summary for " + label,
                document_ids=[s["document_id"] for s in samples],
                roles={s["document_id"]: s["roles"] for s in samples},
                selection_version="coverage-v1",
                generation_version="test",
                warnings=[],
            )

    docs = [
        {"id": f"d{i}", "title": f"T{i}", "contents": f"w{i}.", "structure_path": ["A"]}
        for i in range(5)
    ]
    config = HierarchyConfig(
        mode="structure", description_model="m", description_sample_size=3
    )
    artifact = build_hierarchy(docs, config, describer=Describer())
    provenance = artifact.groups[0]["description_provenance"]
    assert len(provenance["document_ids"]) == 3
    assert provenance["generation_settings"]["sample_size"] == 3
    old = copy.deepcopy(artifact)
    for group in old.groups:
        group["description_provenance"] = None
    old.validate()
    refreshed = build_hierarchy(docs, config, describer=Describer(), previous=old)
    assert all(g["description_provenance"] for g in refreshed.groups)


MAINTENANCE_FIELDS = (
    "min_cluster_size",
    "max_fanout",
    "fit_quantile",
    "split_poor_fit_fraction",
    "description_refresh_overlap",
)


@pytest.mark.parametrize(
    "field,value",
    [
        ("min_cluster_size", 0),
        ("min_cluster_size", 201),
        ("max_fanout", 1),
        ("max_fanout", True),
        ("fit_quantile", 0.0),
        ("fit_quantile", 0.5),
        ("split_poor_fit_fraction", 0.0),
        ("split_poor_fit_fraction", 1.5),
        ("description_refresh_overlap", -0.1),
        ("description_refresh_overlap", 1.1),
    ],
)
def test_maintenance_config_validation(field, value):
    with pytest.raises(ValueError, match=field):
        HierarchyConfig(**{field: value})


def test_old_config_without_maintenance_fields_still_updates():
    artifact = build_hierarchy(docs(path=["A", "B"]), HierarchyConfig(mode="structure"))
    for name in MAINTENANCE_FIELDS:
        artifact.config.pop(name)
    updated = build_hierarchy(docs(7, ["A", "B"]), previous=artifact)
    assert len(updated.assignments) == 7
    assert set(MAINTENANCE_FIELDS) <= set(updated.config)
