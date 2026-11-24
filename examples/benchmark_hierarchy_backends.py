"""Compare hierarchy clustering backends (k-means, E-means, spherical E-means).

Embeds a BEIR dataset (SciFact by default) once, then clusters the same fixed vectors with each
backend and seed in its own subprocess, recording build/update time, peak
memory, leaf balance, coherence and routing recall. Recall is the fraction of
gold documents reached by a centroid-similarity beam over the tree (no LLM),
together with the fraction of the corpus that beam scanned.

Embedding and clustering run in separate processes on purpose: on macOS the pip
``torch`` and ``faiss-cpu`` wheels each bundle ``libomp.dylib`` and abort when
loaded together (``OMP: Error #15``). Trials never import torch.

This is a manual offline measurement; there is no pass threshold. Results from
2026-11-22 are recorded in docs/corpus-hierarchy.md.

Run (E-means needs faiss-cpu>=1.16.0 importable in the trial processes):

    python -m examples.benchmark_hierarchy_backends embed --data bench/
    python -m examples.benchmark_hierarchy_backends run --data bench/ \\
        --output bench/results.jsonl --seeds 1 2 3 4 5
    python -m examples.benchmark_hierarchy_backends summarize bench/results.jsonl
    # Whole-corpus single-call timing; k-means restarts vs one E-means run:
    python -m examples.benchmark_hierarchy_backends speed --data bench/ --clusters 52
"""

from __future__ import annotations

import argparse
import hashlib
import json
import resource
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

BACKENDS = ("kmeans", "e_means", "spherical_e_means")
CONFIGS = {
    "default": dict(target_cluster_size=100, max_cluster_size=200),
    "fine": dict(target_cluster_size=25, max_cluster_size=50, max_fanout=10),
}
BEAMS = (1, 2, 4, 8)
DEFAULT_MODEL = "intfloat/e5-base-v2"


def embed(data_dir, model, device, dataset="scifact"):
    """Download a BEIR dataset from the Hub and save fixed document/query vectors."""
    from datasets import load_dataset

    from src.internal.document_index.hierarchy.formats import first_sentence
    from src.internal.document_index.hierarchy.models import SentenceEmbedding

    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    corpus = {
        str(r["_id"]): r
        for r in load_dataset(f"BeIR/{dataset}", "corpus", split="corpus")
    }
    queries = {
        str(r["_id"]): r["text"]
        for r in load_dataset(f"BeIR/{dataset}", "queries", split="queries")
    }
    qrels = defaultdict(list)
    for r in load_dataset(f"BeIR/{dataset}-qrels", split="test"):
        if int(r["score"]) > 0:
            qrels[str(r["query-id"])].append(str(r["corpus-id"]))
    ids = sorted(corpus)
    docs = [
        dict(id=i, title=corpus[i]["title"], contents=corpus[i]["text"]) for i in ids
    ]
    embedder = SentenceEmbedding(model, device=device, batch_size=64)
    # The same text build_hierarchy embeds; SentenceEmbedding adds "passage: ".
    doc_vecs = embedder.encode(
        [f"{d['title']}\n{first_sentence(d['contents'])}" for d in docs]
    )
    qids = sorted(q for q in qrels if q in queries)
    query_texts = [queries[q] for q in qids]
    if "e5" in model.lower():
        query_texts = [f"query: {t}" for t in query_texts]
    q_vecs = embedder._model.encode(
        query_texts, normalize_embeddings=True, convert_to_numpy=True
    )
    np.savez(
        data_dir / "vectors.npz",
        doc_ids=np.array(ids),
        doc_vecs=np.asarray(doc_vecs, dtype=np.float32),
        qids=np.array(qids),
        q_vecs=np.asarray(q_vecs, dtype=np.float32),
    )
    (data_dir / "docs.json").write_text(json.dumps(docs))
    (data_dir / "qrels.json").write_text(json.dumps({q: qrels[q] for q in qids}))
    (data_dir / "model.txt").write_text(model)
    print(f"{len(docs)} documents, {len(qids)} queries -> {data_dir}")


def load_data(data_dir):
    data_dir = Path(data_dir)
    arrays = np.load(data_dir / "vectors.npz")
    return dict(
        docs=json.loads((data_dir / "docs.json").read_text()),
        qrels=json.loads((data_dir / "qrels.json").read_text()),
        doc_vec=dict(zip(map(str, arrays["doc_ids"]), arrays["doc_vecs"])),
        qids=list(map(str, arrays["qids"])),
        q_vecs=arrays["q_vecs"],
    )


def routing_recall(artifact, doc_vec, q_vecs, qids, qrels, beams=BEAMS):
    """Beam over member-mean centroids; recall of gold docs and corpus scanned."""
    groups = {g["id"]: g for g in artifact.groups}
    children = defaultdict(list)
    for g in artifact.groups:
        children[g["parent_id"]].append(g["id"])
    members = {}

    def descendants(gid):
        if gid not in members:
            ids = list(groups[gid]["document_ids"])
            for child in children.get(gid, []):
                ids += descendants(child)
            members[gid] = ids
        return members[gid]

    centre = {}
    for gid in groups:
        mean = np.mean([doc_vec[i] for i in descendants(gid)], axis=0)
        centre[gid] = mean / np.linalg.norm(mean)
    n = len(artifact.assignments)
    out = {}
    for beam in beams:
        recalls, scanned = [], []
        for qid, qv in zip(qids, q_vecs):
            frontier, reached = children[None], set()
            while frontier:
                kept = sorted(frontier, key=lambda g: -float(qv @ centre[g]))[:beam]
                frontier = [c for g in kept for c in children.get(g, [])]
                for g in kept:
                    reached.update(groups[g]["document_ids"])
            gold = [d for d in qrels[qid] if d in artifact.assignments]
            recalls.append(sum(d in reached for d in gold) / len(gold))
            scanned.append(len(reached) / n)
        out[f"recall@b{beam}"] = float(np.mean(recalls))
        out[f"docs_scanned@b{beam}"] = float(np.mean(scanned))
    return out


def _tree_metrics(artifact, data):
    doc_vec = data["doc_vec"]
    parents = {g["parent_id"] for g in artifact.groups}
    by_id = {g["id"]: g for g in artifact.groups}
    leaves = [
        g for g in artifact.groups if g["id"] not in parents and g["document_ids"]
    ]
    sizes = np.array([len(g["document_ids"]) for g in leaves])
    coherence = []
    for g in leaves:
        mean = np.mean([doc_vec[i] for i in g["document_ids"]], axis=0)
        mean /= np.linalg.norm(mean)
        coherence += [float(doc_vec[i] @ mean) for i in g["document_ids"]]

    def depth(g):
        d = 0
        while g["parent_id"] is not None:
            g, d = by_id[g["parent_id"]], d + 1
        return d

    metrics = dict(
        leaves=len(leaves),
        max_depth=max(depth(g) for g in leaves),
        size_min=int(sizes.min()),
        size_max=int(sizes.max()),
        size_cv=float(sizes.std() / sizes.mean()),
        coherence=float(np.mean(coherence)),
    )
    metrics.update(
        routing_recall(artifact, doc_vec, data["q_vecs"], data["qids"], data["qrels"])
    )
    return metrics


def trial(data, backend, seed, config_name, model=DEFAULT_MODEL):
    """Full build, then a ~90% build updated with the held-back ~10%."""
    from src.internal.document_index.hierarchy import build as hierarchy
    from src.internal.document_index.hierarchy import (
        maintenance as hierarchy_maintenance,
    )
    from src.internal.document_index.hierarchy.formats import first_sentence

    by_text = {
        f"{d['title']}\n{first_sentence(d['contents'])}": data["doc_vec"][d["id"]]
        for d in data["docs"]
    }

    class Lookup:
        model_name = model

        def encode(self, texts):
            return np.array([by_text[t] for t in texts])

    clock = dict(seconds=0.0, calls=0)
    original = hierarchy_maintenance.partition_members

    def timed(*args, **kwargs):
        start = time.perf_counter()
        try:
            return original(*args, **kwargs)
        finally:
            clock["seconds"] += time.perf_counter() - start
            clock["calls"] += 1

    def config():
        return hierarchy.HierarchyConfig(
            mode="cluster",
            embedding_model=model,
            random_state=seed,
            clustering_backend=backend,
            **CONFIGS[config_name],
        )

    hierarchy_maintenance.partition_members = timed
    hierarchy.partition_members = timed
    try:
        docs = data["docs"]
        result = dict(backend=backend, seed=seed, config=config_name)
        start = time.perf_counter()
        full = hierarchy.build_hierarchy(docs, config(), embedder=Lookup())
        warnings = " ".join(full.diagnostics["warnings"])
        result.update(
            build_s=time.perf_counter() - start,
            cluster_s=clock["seconds"],
            cluster_calls=clock["calls"],
            oversized="Oversized" in warnings,
            degenerate="Degenerate" in warnings,
        )
        result.update({f"full_{k}": v for k, v in _tree_metrics(full, data).items()})

        held = {
            d["id"]
            for d in docs
            if int(hashlib.sha1(d["id"].encode()).hexdigest(), 16) % 10 == 0
        }
        base = hierarchy.build_hierarchy(
            [d for d in docs if d["id"] not in held], config(), embedder=Lookup()
        )
        clock.update(seconds=0.0, calls=0)
        start = time.perf_counter()
        updated = hierarchy.build_hierarchy(
            docs, config(), embedder=Lookup(), previous=base
        )
        ops = defaultdict(int)
        for op in updated.diagnostics["structure_operations"]:
            ops[op["op"]] += 1
        result.update(
            update_s=time.perf_counter() - start,
            update_cluster_s=clock["seconds"],
            update_cluster_calls=clock["calls"],
            update_added=len(held),
            update_ops=dict(ops),
            update_poor_fit=updated.diagnostics["poor_fit_documents"],
        )
        result.update({f"upd_{k}": v for k, v in _tree_metrics(updated, data).items()})
    finally:
        hierarchy_maintenance.partition_members = original
        hierarchy.partition_members = original
    return result


def run_grid(data_dir, output, seeds, configs, backends):
    """One subprocess per trial: clean peak RSS and no shared OpenMP state."""
    with open(output, "a") as sink:
        for config_name in configs:
            for seed in seeds:
                for backend in backends:
                    proc = subprocess.run(
                        [
                            sys.executable,
                            "-m",
                            "examples.benchmark_hierarchy_backends",
                            "trial",
                            "--data",
                            str(data_dir),
                            "--backend",
                            backend,
                            "--seed",
                            str(seed),
                            "--config",
                            config_name,
                        ],
                        capture_output=True,
                        text=True,
                    )
                    if proc.returncode:
                        print(
                            f"FAILED {backend} seed={seed} {config_name}:\n"
                            f"{proc.stderr[-2000:]}",
                            file=sys.stderr,
                        )
                        continue
                    line = proc.stdout.strip().splitlines()[-1]
                    sink.write(line + "\n")
                    sink.flush()
                    print(f"done {backend} seed={seed} {config_name}")


def summarize(path):
    """Mean ± sd per metric and backend, then paired recall deltas vs k-means."""
    rows = [json.loads(line) for line in open(path) if line.strip()]
    by_key = defaultdict(list)
    for row in rows:
        by_key[(row["config"], row["backend"])].append(row)
    configs = sorted({row["config"] for row in rows})
    backends = [b for b in BACKENDS if any(r["backend"] == b for r in rows)]
    skip = {"backend", "seed", "config", "update_ops"}
    metrics = [k for k in rows[0] if k not in skip]
    for config_name in configs:
        print(f"\n## config={config_name}")
        print(f"{'metric':24}" + "".join(f"{b:>26}" for b in backends))
        for metric in metrics:
            cells = []
            for backend in backends:
                v = np.array([r[metric] for r in by_key[(config_name, backend)]], float)
                sd = v.std(ddof=1) if len(v) > 1 else 0.0
                cells.append(f"{v.mean():.4g} ± {sd:.2g} (n={len(v)})")
            print(f"{metric:24}" + "".join(f"{c:>26}" for c in cells))
        for backend in backends:
            ops = defaultdict(int)
            for row in by_key[(config_name, backend)]:
                for name, count in row["update_ops"].items():
                    ops[name] += count
            print(f"update ops {backend}: {dict(ops)}")
    print("\n## paired recall delta vs kmeans (mean over seeds, [min, max])")
    for config_name in configs:
        base = {r["seed"]: r for r in by_key[(config_name, "kmeans")]}
        for backend in backends:
            if backend == "kmeans":
                continue
            for metric in ("full_recall@b1", "full_recall@b4", "upd_recall@b4"):
                deltas = [
                    r[metric] - base[r["seed"]][metric]
                    for r in by_key[(config_name, backend)]
                    if r["seed"] in base and metric in r
                ]
                if deltas:
                    print(
                        f"{config_name:8} {backend:18} {metric:16} "
                        f"{np.mean(deltas):+.4f} [{min(deltas):+.4f}, {max(deltas):+.4f}]"
                    )


def speed(data_dir, clusters, algorithm, seeds=(1, 2, 3)):
    """Time one whole-corpus partition; run each algorithm in its own process."""
    x = np.load(Path(data_dir) / "vectors.npz")["doc_vecs"]
    if algorithm.startswith("kmeans"):
        from sklearn.cluster import KMeans

        n_init = int(algorithm.split("_")[1])
        x = x.astype(np.float64)  # what kmeans_partition passes to sklearn

        def fit(seed):
            return KMeans(clusters, random_state=seed, n_init=n_init).fit_predict(x)
    else:
        from faiss.contrib.e_means import NumpyEmeans

        x = np.ascontiguousarray(x, dtype=np.float32)

        def fit(seed):
            model = NumpyEmeans(
                x.shape[1], clusters, seed=seed, capture_assignments=False
            )
            model.train(x)
            return model.assign(x)[1]

    times, inertia = [], []
    for seed in seeds:
        start = time.perf_counter()
        labels = np.asarray(fit(seed)).ravel()
        times.append(time.perf_counter() - start)
        inertia.append(
            sum(
                ((x[labels == c] - x[labels == c].mean(0)) ** 2).sum()
                for c in np.unique(labels)
            )
        )
    print(
        f"clusters={clusters} {algorithm}: {np.mean(times):.2f}s "
        f"inertia={np.mean(inertia):.1f}"
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("embed")
    p.add_argument("--data", required=True)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--device", default="cpu")
    p.add_argument(
        "--dataset", default="scifact", help="BEIR name on the Hub, e.g. nfcorpus"
    )
    p = sub.add_parser("run")
    p.add_argument("--data", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    p.add_argument("--configs", nargs="+", choices=CONFIGS, default=list(CONFIGS))
    p.add_argument("--backends", nargs="+", choices=BACKENDS, default=list(BACKENDS))
    p = sub.add_parser("trial")
    p.add_argument("--data", required=True)
    p.add_argument("--backend", choices=BACKENDS, required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--config", choices=CONFIGS, required=True)
    p = sub.add_parser("summarize")
    p.add_argument("results")
    p = sub.add_parser("speed")
    p.add_argument("--data", required=True)
    p.add_argument("--clusters", type=int, required=True)
    p.add_argument("--algorithm", choices=("kmeans_10", "kmeans_1", "e_means"))
    args = parser.parse_args(argv)

    if args.command == "embed":
        embed(args.data, args.model, args.device, args.dataset)
    elif args.command == "run":
        run_grid(args.data, args.output, args.seeds, args.configs, args.backends)
    elif args.command == "trial":
        model_file = Path(args.data) / "model.txt"
        model = model_file.read_text().strip() if model_file.exists() else DEFAULT_MODEL
        data = load_data(args.data)
        base_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        result = trial(data, args.backend, args.seed, args.config, model=model)
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        scale = 2**20 if sys.platform == "darwin" else 2**10  # bytes vs KiB
        result["rss_growth_mb"] = (peak - base_rss) / scale
        if "torch" in sys.modules:
            raise RuntimeError("trial imported torch; OpenMP conflict risk")
        print(json.dumps(result))
    elif args.command == "summarize":
        summarize(args.results)
    elif args.command == "speed":
        algorithms = (
            [args.algorithm] if args.algorithm else ["kmeans_10", "kmeans_1", "e_means"]
        )
        if len(algorithms) == 1:
            speed(args.data, args.clusters, algorithms[0])
        else:
            for algorithm in algorithms:
                subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "examples.benchmark_hierarchy_backends",
                        "speed",
                        "--data",
                        args.data,
                        "--clusters",
                        str(args.clusters),
                        "--algorithm",
                        algorithm,
                    ],
                    check=False,
                )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
