"""Backend benchmark harness: routing recall, one trial, and the summary."""

import json

import numpy as np

from examples.benchmark_hierarchy_backends import (
    load_data,
    routing_recall,
    summarize,
    trial,
)
from src.internal.document_index.hierarchy import HierarchyArtifact


def write_data(path, per_topic=30, topics=3, dim=8):
    rng = np.random.default_rng(0)
    ids, vecs, docs = [], [], []
    for t in range(topics):
        for i in range(per_topic):
            doc_id = f"t{t}d{i}"
            v = np.eye(dim)[t] + rng.normal(0, 0.05, dim)
            ids.append(doc_id)
            vecs.append(v / np.linalg.norm(v))
            docs.append(dict(id=doc_id, title=f"Topic {t} doc {i}", contents="Body."))
    q_vecs = np.eye(dim)[:topics]
    np.savez(
        path / "vectors.npz",
        doc_ids=np.array(ids),
        doc_vecs=np.array(vecs, dtype=np.float32),
        qids=np.array([f"q{t}" for t in range(topics)]),
        q_vecs=q_vecs.astype(np.float32),
    )
    (path / "docs.json").write_text(json.dumps(docs))
    (path / "qrels.json").write_text(
        json.dumps({f"q{t}": [f"t{t}d0", f"t{t}d1"] for t in range(topics)})
    )


def test_routing_recall_follows_the_beam():
    vec = {"a": np.array([1.0, 0.0]), "b": np.array([0.0, 1.0])}
    art = HierarchyArtifact(
        groups=[
            dict(id="ga", parent_id=None, document_ids=["a"]),
            dict(id="gb", parent_id=None, document_ids=["b"]),
        ],
        assignments={"a": "ga", "b": "gb"},
    )
    q = np.array([[1.0, 0.0]])
    one = routing_recall(art, vec, q, ["q"], {"q": ["a", "b"]}, beams=(1, 2))
    assert one["recall@b1"] == 0.5 and one["docs_scanned@b1"] == 0.5
    assert one["recall@b2"] == 1.0 and one["docs_scanned@b2"] == 1.0


def test_trial_reports_build_update_and_recall(tmp_path):
    write_data(tmp_path)
    data = load_data(tmp_path)
    result = trial(data, "kmeans", 1, "default", model="fake-model")
    assert result["backend"] == "kmeans" and result["seed"] == 1
    assert result["full_leaves"] >= 1 and result["upd_leaves"] >= 1
    assert result["full_recall@b8"] == 1.0  # well-separated topics
    assert result["cluster_calls"] >= 1
    assert 0 < result["update_added"] < 90
    assert isinstance(result["update_ops"], dict)


def test_summarize_reports_each_backend_and_paired_deltas(tmp_path, capsys):
    rows = []
    for backend, recall in (("kmeans", 0.5), ("e_means", 0.4)):
        for seed in (1, 2):
            row = dict(backend=backend, seed=seed, config="default", update_ops={})
            row.update(
                {
                    m: recall
                    for m in ("full_recall@b1", "full_recall@b4", "upd_recall@b4")
                }
            )
            rows.append(row)
    path = tmp_path / "results.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows))
    summarize(path)
    out = capsys.readouterr().out
    assert "e_means" in out and "kmeans" in out
    assert "-0.1000" in out
