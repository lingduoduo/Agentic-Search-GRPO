"""CLI and library for offline retrieval evaluation.

Usage (retrieval only — requires BM25_INDEX_PATH env var):
    python -m src.internal.retrieval.eval_runner \\
        --dataset data/eval/qa_pairs.jsonl --top_k 10

Usage against a running retrieval server (no env vars needed):
    python -m src.internal.retrieval.eval_runner \\
        --dataset data/eval/qa_pairs.jsonl --top_k 10 \\
        --retrieval_url http://localhost:8001/retrieve

Usage (with reranking):
    python -m src.internal.retrieval.eval_runner \\
        --dataset data/eval/qa_pairs.jsonl --top_k 10 \\
        --retrieval_url http://localhost:8001/retrieve \\
        --reranker local --reranker_model BAAI/bge-reranker-v2-m3

QA pairs file format (one JSON object per line):
    {"query": "...", "relevant_doc_ids": ["doc-id-1", "doc-id-2"]}
"""

from __future__ import annotations

import argparse
import json
import math
import time

from .eval_metrics import map_at_k, reranker_improvement_ratio
from .eval_metrics import mrr as mrr_score
from .eval_metrics import ndcg_at_k, recall_at_k
from .service import RetrievalService


class SLOViolationError(RuntimeError):
    """Raised when the P99 reranker latency exceeds the configured SLO."""


def qt_slo_exceeded(latencies_ms: list[float], slo_ms: int) -> bool:
    """True when the P99 transform latency exceeds slo_ms.

    Authoritative SLO gate: uses index int(n*0.99) (distinct from the display
    _percentile which uses math.ceil-based indexing).
    """
    if not latencies_ms:
        return False
    ordered = sorted(latencies_ms)
    idx = min(len(ordered) - 1, int(len(ordered) * 0.99))
    return ordered[idx] > slo_ms


def _percentile(values: list[float], p: int) -> float:
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    idx = min(len(sorted_vals) - 1, max(0, math.ceil(len(sorted_vals) * p / 100) - 1))
    return round(sorted_vals[idx], 1)


def run_eval(
    dataset_path: str,
    *,
    service: RetrievalService | None = None,
    top_k: int = 10,
    reranker=None,  # Reranker | None — avoid circular import at module level
    slo_ms: int | None = None,
    compare_baseline: bool = False,
    qt_slo_ms: int | None = None,
) -> dict:
    """Load QA pairs, run retrieval (and optionally reranking), return metrics.

    Without reranker: returns flat dict {recall@k, ndcg@k, mrr, num_queries}.
    With reranker:    returns {retrieval: {...}, reranked: {...}, latency_ms: {...}}.
    """
    _service = service or RetrievalService.from_env()

    with open(dataset_path) as f:
        qa_pairs = [json.loads(line) for line in f if line.strip()]

    recalls, ndcgs, mrrs = [], [], []
    r_recalls, r_ndcgs, r_mrrs, r_maps, latencies_ms = [], [], [], [], []
    qt_latencies: list[float] = []

    for item in qa_pairs:
        query: str = item["query"]
        relevant: set[str] = set(item["relevant_doc_ids"])
        _qt0 = time.perf_counter()
        results, _ = _service.search(query, top_k=top_k)
        qt_latencies.append((time.perf_counter() - _qt0) * 1000)
        retrieved = [r.doc_id for r in results]

        recalls.append(recall_at_k(retrieved, relevant, top_k))
        ndcgs.append(ndcg_at_k(retrieved, relevant, top_k))
        mrrs.append(mrr_score(retrieved, relevant))

        if reranker is not None:
            t0 = time.monotonic()
            reranked_results = reranker.rerank(query, results, top_k)
            latencies_ms.append((time.monotonic() - t0) * 1000)
            r_retrieved = [r.doc_id for r in reranked_results]
            r_recalls.append(recall_at_k(r_retrieved, relevant, top_k))
            r_ndcgs.append(ndcg_at_k(r_retrieved, relevant, top_k))
            r_mrrs.append(mrr_score(r_retrieved, relevant))
            r_maps.append(map_at_k(r_retrieved, relevant, top_k))

    n = len(qa_pairs)

    def _avg(lst):
        return round(sum(lst) / n, 4) if n else 0.0

    retrieval_metrics = {
        f"recall@{top_k}": _avg(recalls),
        f"ndcg@{top_k}": _avg(ndcgs),
        "mrr": _avg(mrrs),
        "num_queries": n,
    }

    if reranker is None:
        retrieval_metrics["qt_latency_ms"] = {
            "p99": _percentile(qt_latencies, 99),
            "n": n,
        }
        if qt_slo_ms is not None and qt_slo_exceeded(qt_latencies, qt_slo_ms):
            import sys

            print(f"QT SLO breach: P99 > {qt_slo_ms}ms")
            sys.exit(1)
        return retrieval_metrics

    result = {
        "retrieval": retrieval_metrics,
        "reranked": {
            f"recall@{top_k}": _avg(r_recalls),
            f"ndcg@{top_k}": _avg(r_ndcgs),
            f"map@{top_k}": round(sum(r_maps) / n, 4) if n else 0.0,
            "mrr": _avg(r_mrrs),
            "num_queries": n,
        },
        "latency_ms": {
            "mean": round(sum(latencies_ms) / len(latencies_ms), 1)
            if latencies_ms
            else 0.0,
            "p50": _percentile(latencies_ms, 50),
            "p95": _percentile(latencies_ms, 95),
            "p99": _percentile(latencies_ms, 99),
            "n": n,
        },
        "qt_latency_ms": {
            "p99": _percentile(qt_latencies, 99),
            "n": n,
        },
    }

    if compare_baseline:
        result["reranker_improvement_ratio"] = reranker_improvement_ratio(
            retrieval_metrics[f"ndcg@{top_k}"],
            result["reranked"][f"ndcg@{top_k}"],
        )

    if slo_ms is not None and latencies_ms:
        p99 = _percentile(latencies_ms, 99)
        if p99 > slo_ms:
            raise SLOViolationError(
                f"P99 reranker latency {p99}ms exceeds SLO {slo_ms}ms"
            )

    if qt_slo_ms is not None and qt_slo_exceeded(qt_latencies, qt_slo_ms):
        import sys

        print(f"QT SLO breach: P99 > {qt_slo_ms}ms")
        sys.exit(1)

    return result


def run_routing_eval(dataset_path: str, router=None) -> dict:
    """Score the router's top-1 retriever prediction against a labeled set."""
    from .eval_metrics import routing_accuracy

    if router is None:
        from src.internal.routing.registry import DEFAULT_ROUTES, RouteRegistry
        from src.internal.routing.router import Router

        router = Router(RouteRegistry(DEFAULT_ROUTES))

    with open(dataset_path) as f:
        rows = [json.loads(line) for line in f if line.strip()]

    predictions = [router.route(r["query"]).retriever.value for r in rows]
    labels = [str(r["retriever"]) for r in rows]
    return {
        "routing_accuracy": routing_accuracy(predictions, labels),
        "num_queries": len(rows),
    }


class _HttpService:
    """Thin sync wrapper around the retrieval HTTP endpoint for CLI use."""

    def __init__(self, url: str) -> None:
        self._url = url

    def search(
        self, query: str, top_k: int = 10, filters: dict | None = None
    ) -> tuple[list, str]:
        import requests

        from src.internal.retrieval.backends.base import RetrievalResult

        payload: dict = {"queries": [query], "topk": top_k, "return_scores": True}
        if filters:
            payload["filters"] = filters
        resp = requests.post(self._url, json=payload, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        rows = (data.get("results") or data.get("result") or [[]])[0]
        results = []
        for item in rows:
            if "document" in item:
                doc, score = item["document"], float(item.get("score", 0.0))
            else:
                doc, score = item, 0.0
            results.append(
                RetrievalResult(
                    doc_id=str(doc.get("id") or doc.get("doc_id", "")),
                    title=doc.get("title") or "",
                    text=doc.get("contents") or doc.get("text") or "",
                    url=doc.get("url"),
                    score=score,
                )
            )
        return results, "http"


def main(argv: list[str] | None = None) -> None:
    """CLI entry point; ``argv`` defaults to ``sys.argv[1:]``."""
    from dotenv import load_dotenv

    load_dotenv()

    parser = argparse.ArgumentParser(description="Offline retrieval evaluation")
    parser.add_argument("--dataset", required=True, help="Path to qa_pairs.jsonl")
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument(
        "--retrieval_url",
        default=None,
        help="HTTP retrieval endpoint (e.g. http://localhost:8001/retrieve). "
        "When omitted, RetrievalService.from_env() is used (requires BM25_INDEX_PATH).",
    )
    parser.add_argument(
        "--reranker",
        choices=["local", "cohere"],
        default=None,
        help="Provider for reranking (omit to skip reranking)",
    )
    parser.add_argument(
        "--reranker_model",
        default="BAAI/bge-reranker-v2-m3",
        help="Model name for local reranker or Cohere model name",
    )
    parser.add_argument(
        "--slo-ms",
        type=int,
        default=None,
        help="P99 latency SLO in ms. Exits non-zero if exceeded.",
    )
    parser.add_argument(
        "--compare-baseline",
        action="store_true",
        help="Print reranker_improvement_ratio vs retrieval-only NDCG",
    )
    parser.add_argument(
        "--qt-slo-ms",
        type=int,
        default=None,
        help="Fail if P99 query-transform latency exceeds this budget",
    )
    parser.add_argument(
        "--routing-eval",
        action="store_true",
        help="Score the router against a labeled routing set (query, retriever).",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Also write the metrics JSON to this path (the CI eval gate reads it).",
    )
    args = parser.parse_args(argv)

    def _emit(metrics: dict) -> None:
        text = json.dumps(metrics, indent=2)
        print(text)
        if args.output:
            with open(args.output, "w") as handle:
                handle.write(text + "\n")

    if args.routing_eval:
        _emit(run_routing_eval(args.dataset))
        raise SystemExit(0)

    service = _HttpService(args.retrieval_url) if args.retrieval_url else None

    reranker = None
    if args.reranker:
        import os

        from src.internal.retrieval.reranker import Reranker, RerankerConfig

        reranker = Reranker(
            RerankerConfig(
                provider=args.reranker,
                model=args.reranker_model,
                api_key=os.environ.get("COHERE_API_KEY"),
            )
        )

    metrics = run_eval(
        args.dataset,
        top_k=args.top_k,
        service=service,
        reranker=reranker,
        slo_ms=args.slo_ms,
        compare_baseline=args.compare_baseline,
        qt_slo_ms=args.qt_slo_ms,
    )
    _emit(metrics)


if __name__ == "__main__":
    main()
