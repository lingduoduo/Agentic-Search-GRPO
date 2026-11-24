"""Optional E-means boundary tests plus real-backend tests when installed."""

import builtins
import copy
import importlib
import sys
import types

import numpy as np
import pytest

from src.internal.document_index.hierarchy import (
    HierarchyArtifact,
    HierarchyConfig,
    build_hierarchy,
)
from src.internal.document_index.hierarchy.cli import main
from src.internal.document_index.hierarchy import maintenance


@pytest.fixture
def fake_emeans(monkeypatch):
    calls = []

    class Model:
        def __init__(self, d, k, **kwargs):
            self.k = k
            calls.append(dict(d=d, k=k, **kwargs))

        def train(self, x):
            assert x.dtype == np.float32 and x.flags.c_contiguous
            calls[-1]["x"] = x.copy()
            # Deliberately stale training labels: the adapter must use assign.
            return x[: self.k], np.zeros(len(x), dtype=int)

        def assign(self, x):
            calls[-1]["assigned"] = True
            return np.zeros(len(x)), np.arange(len(x)) % self.k

    monkeypatch.setitem(
        sys.modules, "faiss.contrib.e_means", types.SimpleNamespace(NumpyEmeans=Model)
    )
    return calls


def vectors(n=12):
    return {str(i): [1.0, float(i % 3), 0.1] for i in range(n)}


def documents(n=12):
    return [dict(id=str(i), title=str(i), contents="Body.") for i in range(n)]


class Encoder:
    def encode(self, texts):
        return np.array([[1, int(t.split()[0]) % 3, 0.1] for t in texts])


@pytest.mark.parametrize("backend", ["e_means", "spherical_e_means"])
def test_partition_adapter_preserves_members_bounds_and_settings(fake_emeans, backend):
    v = {str(i): [1.0, float(i), 0.1] for i in range(12)}  # all distinct
    result = maintenance.partition_members(
        list(reversed(v)), v, 100, 2, 7, backend=backend, e_means_epochs=2.5
    )
    buckets, degenerate, oversized = result
    assert sorted(i for b in buckets for i in b) == sorted(v)
    assert len(buckets) == len(v)  # count capped at number of documents
    assert not degenerate and not oversized
    call = fake_emeans[-1]
    assert call["k"] == len(v) and call["seed"] == 7
    assert call["nepochs"] == 2.5
    assert call["spherical"] == (backend == "spherical_e_means")
    assert call["assigned"]
    assert np.allclose(call["x"], [v[i] for i in sorted(v)])
    buckets, degenerate, oversized = maintenance.partition_members(
        list(v), v, 2, 2, 7, backend=backend
    )
    assert oversized and not degenerate
    assert max(map(len, buckets)) <= 2


def test_empty_clusters_on_distinct_vectors_are_not_degenerate(
    fake_emeans, monkeypatch
):
    # Matches sklearn: degenerate means fewer distinct vectors than clusters,
    # not merely an empty cluster left by mini-batch training.
    model = sys.modules["faiss.contrib.e_means"].NumpyEmeans
    monkeypatch.setattr(model, "assign", lambda self, x: (None, np.zeros(len(x), int)))
    buckets, degenerate, oversized = maintenance.partition_members(
        list(vectors()), vectors(), 3, 4, 42, backend="e_means"
    )
    assert not degenerate and oversized
    assert len(buckets) == 3


def test_fewer_distinct_vectors_than_clusters_is_degenerate(fake_emeans):
    v = vectors()  # 12 members, only 3 distinct vectors
    _, degenerate, _ = maintenance.partition_members(
        list(v), v, 4, 12, 42, backend="e_means"
    )
    assert degenerate
    # Exactly as many distinct vectors as clusters is not degenerate.
    _, degenerate, _ = maintenance.partition_members(
        list(v), v, 3, 12, 42, backend="e_means"
    )
    assert not degenerate


def test_signed_zeros_are_one_distinct_vector(fake_emeans):
    # Float equality, as np.unique uses: -0.0 == 0.0 despite different bytes.
    v = {"a": [0.0, 1.0], "b": [-0.0, 1.0]}
    _, degenerate, _ = maintenance.partition_members(
        list(v), v, 2, 2, 42, backend="e_means"
    )
    assert degenerate


def test_missing_backend_is_actionable_and_default_stays_lazy(monkeypatch):
    original = builtins.__import__

    def unavailable(name, *args, **kwargs):
        if name.startswith("faiss"):
            raise ImportError("old or missing faiss")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", unavailable)
    v = vectors(4)
    assert maintenance.partition_members([], {}, 2, 2, 42, backend="e_means") == (
        [],
        False,
        False,
    )
    assert maintenance.partition_members(list(v), v, 2, 4, 42)
    with pytest.raises(ValueError, match="faiss.*e_means"):
        maintenance.partition_members(list(v), v, 2, 4, 42, backend="e_means")


@pytest.mark.parametrize(
    "kwargs",
    [
        {"clustering_backend": "unknown"},
        {"clustering_backend": []},
        {"e_means_epochs": 0},
        {"e_means_epochs": -1},
        {"e_means_epochs": True},
        {"e_means_epochs": float("nan")},
        {"e_means_epochs": float("inf")},
        {"e_means_epochs": "5"},
    ],
)
def test_invalid_backend_config(kwargs):
    with pytest.raises(ValueError):
        HierarchyConfig(**kwargs)


def test_build_update_and_levels_use_selected_backend(fake_emeans, tmp_path):
    config = HierarchyConfig(
        mode="cluster",
        clustering_backend="spherical_e_means",
        e_means_epochs=2.5,
        n_clusters=4,
        max_cluster_size=3,
        target_cluster_size=2,
        min_cluster_size=1,
        max_fanout=2,
    )
    artifact = build_hierarchy(documents(), config, embedder=Encoder())
    assert len(fake_emeans) > 1  # initial build plus intermediate levels
    assert all(c["spherical"] and c["nepochs"] == 2.5 for c in fake_emeans)
    path = tmp_path / "tree.json"
    artifact.save(path)
    assert HierarchyArtifact.load(path) == artifact
    before = len(fake_emeans)
    unchanged = build_hierarchy(documents(), config, previous=artifact)
    assert len(fake_emeans) == before
    assert unchanged.assignments == artifact.assignments
    # Force local splitting as well as initial build/level insertion.
    updated = build_hierarchy(
        documents(24), config, embedder=Encoder(), previous=artifact
    )
    assert len(fake_emeans) > before
    assert all(c["spherical"] and c["nepochs"] == 2.5 for c in fake_emeans)
    assert set(updated.assignments) == {str(i) for i in range(24)}
    assert all(len(g["document_ids"]) <= 3 for g in updated.groups)
    with pytest.raises(ValueError, match="rebuild"):
        build_hierarchy(documents(), HierarchyConfig(mode="cluster"), previous=artifact)


def test_old_artifact_defaults_to_kmeans():
    artifact = build_hierarchy(
        documents(3), HierarchyConfig(mode="cluster"), embedder=Encoder()
    )
    old = copy.deepcopy(artifact)
    old.config.pop("clustering_backend")
    old.config.pop("e_means_epochs")
    updated = build_hierarchy(documents(3), previous=old)
    assert updated.config["clustering_backend"] == "kmeans"
    assert updated.assignments == old.assignments


def test_cli_persists_and_inherits_backend(fake_emeans, monkeypatch, tmp_path, capsys):
    import src.internal.document_index.hierarchy.models as models

    monkeypatch.setattr(models, "SentenceEmbedding", lambda *a, **kw: Encoder())
    corpus = tmp_path / "corpus.jsonl"
    output = tmp_path / "tree.json"
    import json

    corpus.write_text("\n".join(json.dumps(d) for d in documents()))
    assert (
        main(
            [
                "build",
                "--corpus-path",
                str(corpus),
                "--output",
                str(output),
                "--mode",
                "cluster",
                "--clustering-backend",
                "e_means",
                "--e-means-epochs",
                "2.5",
                "--clusters",
                "3",
            ]
        )
        == 0
    )
    assert (
        main(
            [
                "update",
                "--corpus-path",
                str(corpus),
                "--output",
                str(output),
                "--previous",
                str(output),
            ]
        )
        == 0
    )
    artifact = HierarchyArtifact.load(output)
    assert artifact.config["clustering_backend"] == "e_means"
    assert artifact.config["e_means_epochs"] == 2.5
    capsys.readouterr()


@pytest.mark.parametrize("backend", ["e_means", "spherical_e_means"])
def test_real_emeans_repeatability_quality_and_duplicates(backend):
    try:
        importlib.import_module("faiss.contrib.e_means")
    except ImportError:
        pytest.skip("optional Faiss E-means installation required")
    rng = np.random.default_rng(3)
    x = np.vstack([np.eye(3)[j] + rng.normal(0, 0.01, (20, 3)) for j in range(3)])
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    v = {str(i): row.tolist() for i, row in enumerate(x)}
    args = (list(v), v, 3, 10, 42)
    first = maintenance.partition_members(*args, backend=backend)
    assert first == maintenance.partition_members(*args, backend=backend)
    buckets, degenerate, oversized = first
    assert not degenerate and oversized
    assert sorted(i for b in buckets for i in b) == sorted(v)
    assert all(len(b) <= 10 for b in buckets)
    assert all(len({int(i) // 20 for i in b}) == 1 for b in buckets)
    duplicate = {str(i): [1.0, 0, 0] for i in range(8)}
    buckets, degenerate, oversized = maintenance.partition_members(
        list(duplicate), duplicate, 3, 2, 42, backend=backend
    )
    assert degenerate and oversized
    assert sorted(i for b in buckets for i in b) == sorted(duplicate)
    assert all(len(b) <= 2 for b in buckets)


@pytest.mark.parametrize("backend", ["e_means", "spherical_e_means"])
def test_real_emeans_tiny_batches_and_snapshot_update(backend):
    try:
        importlib.import_module("faiss.contrib.e_means")
    except ImportError:
        pytest.skip("optional Faiss E-means installation required")
    assert maintenance.partition_members(
        ["a"],
        {"a": [1.0, 0.0, 0.0]},
        10,
        2,
        42,
        backend=backend,
        e_means_epochs=0.5,
    ) == ([["a"]], False, False)
    config = HierarchyConfig(
        mode="cluster",
        clustering_backend=backend,
        n_clusters=3,
        max_cluster_size=4,
        target_cluster_size=3,
        min_cluster_size=2,
        max_fanout=2,
    )
    artifact = build_hierarchy(documents(), config, embedder=Encoder())
    artifact.validate()
    updated = build_hierarchy(
        documents(24), config, embedder=Encoder(), previous=artifact
    )
    updated.validate()
    assert set(updated.assignments) == {str(i) for i in range(24)}
    assert all(len(g["document_ids"]) <= 4 for g in updated.groups)


def test_structure_only_emeans_does_not_import_faiss(monkeypatch):
    original = builtins.__import__

    def forbidden(name, *args, **kwargs):
        if name.startswith("faiss"):
            pytest.fail("structured corpus imported optional Faiss")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", forbidden)
    artifact = build_hierarchy(
        [
            dict(
                id="a",
                title="Guide",
                contents="Text.",
                metadata={"hierarchy_path": ["Guide"]},
            )
        ],
        HierarchyConfig(clustering_backend="e_means"),
    )
    assert artifact.assignments["a"]


def test_fractional_budget_trains_at_least_one_row(fake_emeans):
    maintenance.partition_members(
        ["a", "b"],
        {"a": [1.0, 0.0], "b": [0.0, 1.0]},
        2,
        2,
        42,
        backend="e_means",
        e_means_epochs=0.01,
    )
    assert fake_emeans[-1]["nepochs"] == 0.5


@pytest.mark.parametrize("backend", ["e_means", "spherical_e_means"])
def test_real_emeans_spawn_retries_preserve_backend(backend, monkeypatch):
    try:
        importlib.import_module("faiss.contrib.e_means")
    except ImportError:
        pytest.skip("optional Faiss E-means installation required")
    v = {
        **{f"a{i}": np.eye(8)[0].tolist() for i in range(6)},
        **{f"b{i}": np.eye(8)[1].tolist() for i in range(6)},
    }
    leaves = [
        dict(
            id=prefix,
            parent_id=None,
            label=prefix,
            document_ids=[f"{prefix}{i}" for i in range(6)],
            centroid=np.eye(8)[axis].tolist(),
        )
        for axis, prefix in enumerate(("a", "b"))
    ]
    new = {f"n{i}": np.eye(8)[2].tolist() for i in range(6)}
    scattered = {
        "s0": np.eye(8)[3],
        "s1": (np.eye(8)[0] + np.eye(8)[3]) / np.sqrt(2),
        "s2": (np.eye(8)[1] + np.eye(8)[3]) / np.sqrt(2),
    }
    v.update(new)
    v.update({i: row.tolist() for i, row in scattered.items()})
    candidates = set(new) | set(scattered)
    assignments = {i: g["id"] for g in leaves for i in g["document_ids"]}
    assignments.update({i: "a" for i in candidates})
    calls = []
    original = maintenance.partition_members

    def recording(*args, **kwargs):
        calls.append((args[2], kwargs.copy()))
        return original(*args, **kwargs)

    monkeypatch.setattr(maintenance, "partition_members", recording)
    result = maintenance.maintain_leaves(
        leaves,
        v,
        assignments,
        HierarchyConfig(clustering_backend=backend, min_cluster_size=5),
        0.99,
        candidates,
    )
    spawns = [op for op in result.operations if op["op"] == "spawn"]
    assert spawns
    spawned = next(g for g in result.leaves if g["id"] == spawns[0]["group_ids"][0])
    assert set(spawned["document_ids"]) == set(new)
    assert calls[0][0] == 1 and any(count > 1 for count, _ in calls)
    assert all(
        kw["backend"] == backend and kw["e_means_epochs"] == 5 for _, kw in calls
    )
