"""Hybrid retrieval server — RRF-fused dense (e5, in-memory) + sparse (TF-IDF).

Java-free and FAISS-free. Exposes the same /retrieve API as demo.py so the web
backend's "Local Retrieval" provider gets hybrid results with no changes.
"""

from __future__ import annotations

import argparse
import logging
from collections.abc import Callable

import numpy as np

from src.internal.document_index.hybrid_retriever import combine_retrieval_results
from src.internal.servers.app import (
    add_host_port_args,
    create_base_app,
    load_environment,
    run_uvicorn_app,
)
from src.internal.servers.retrieval.demo import (
    DEFAULT_TOPK,
    RetrieveRequest,
    TfidfRetriever,
    _allowed_by_acl,
)


logger = logging.getLogger(__name__)

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8001
DEFAULT_MODEL = "intfloat/e5-base-v2"


def _passage_text(doc: dict) -> str:
    body = doc.get("contents", doc.get("text", ""))
    return f"passage: {doc.get('title', '')} {body}".strip()


def _as_document(doc: dict, index: int) -> dict:
    return {
        "id": doc.get("id", str(index)),
        "title": doc.get("title", ""),
        "text": doc.get("contents", doc.get("text", "")),
        "url": doc.get("url"),
        # Forward corpus metadata (incl. the per-document "source") so result
        # cards can show a real origin instead of the generic provider label.
        "metadata": doc.get("metadata") or {},
    }


class DenseEmbeddingRetriever:
    """Dense retrieval over an in-memory e5 embedding matrix (no FAISS).

    Embeds corpus passages once at construction; retrieve() encodes each query
    as "query: <q>" and ranks documents by dot product (embeddings are
    L2-normalized, so dot product == cosine similarity).
    """

    def __init__(
        self, docs: list[dict], *, encoder: Callable[[list[str]], np.ndarray]
    ) -> None:
        self._docs = docs
        self._encoder = encoder
        if docs:
            self._matrix = encoder([_passage_text(d) for d in docs])
        else:
            self._matrix = np.empty((0, 0), dtype=np.float32)

    def retrieve(self, queries: list[str], topk: int) -> list[list[dict]]:
        if self._matrix.size == 0:
            return [[] for _ in queries]
        query_vecs = self._encoder([f"query: {q}" for q in queries])
        sims = query_vecs @ self._matrix.T
        results: list[list[dict]] = []
        for row in sims:
            ranked = sorted(enumerate(row), key=lambda x: x[1], reverse=True)[:topk]
            results.append(
                [
                    {"document": _as_document(self._docs[i], i), "score": float(score)}
                    for i, score in ranked
                ]
            )
        return results


def build_e5_encoder(
    model_name: str = DEFAULT_MODEL, device: str = "mps"
) -> Callable[[list[str]], np.ndarray]:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name, device=device)

    def encode(texts: list[str]) -> np.ndarray:
        return model.encode(
            texts, normalize_embeddings=True, convert_to_numpy=True
        ).astype(np.float32)

    encode.model_id = model_name  # type: ignore[attr-defined]
    return encode


def _fuse_rows(
    dense_rows: list[list[dict]], sparse_rows: list[list[dict]], topk: int
) -> list[list[dict]]:
    fused: list[list[dict]] = []
    for dense_row, sparse_row in zip(dense_rows, sparse_rows):
        result_sets = [sparse_row] if not dense_row else [dense_row, sparse_row]
        fused.append(combine_retrieval_results(result_sets)[:topk])
    return fused


def create_app(*, dense: object | None, sparse: object):
    app = create_base_app("Hybrid Retrieval Server")

    @app.post("/retrieve")
    def retrieve_endpoint(body: RetrieveRequest):
        queries = body.resolved_queries()
        if not queries:
            return {"results": []}
        fetch_k = body.topk * 2
        sparse_rows = sparse.retrieve(queries, topk=fetch_k)
        if dense is not None:
            dense_rows = dense.retrieve(queries, topk=fetch_k)
        else:
            dense_rows = [[] for _ in queries]
        rows = _fuse_rows(dense_rows, sparse_rows, body.topk)
        rows = [
            [item for item in row if _allowed_by_acl(item["document"], body.filters)]
            for row in rows
        ]
        if not body.return_scores:
            rows = [[item["document"] for item in row] for row in rows]
        if body.query is not None:
            return {"results": rows[0] if rows else []}
        return {"results": rows}

    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Hybrid (dense + sparse) retrieval server"
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--corpus_path", type=str, help="Path to a corpus .jsonl file")
    source.add_argument(
        "--corpus",
        type=str,
        help="Registered corpus name, comma-list, or 'all' (see data/corpora.json)",
    )
    parser.add_argument("--topk", type=int, default=DEFAULT_TOPK)
    parser.add_argument(
        "--device", type=str, default="mps", help="Device for e5 (mps/cpu/cuda)"
    )
    parser.add_argument(
        "--no-dense", action="store_true", help="Disable the dense leg (TF-IDF only)"
    )
    add_host_port_args(
        parser,
        "HYBRID_RETRIEVAL_HOST",
        "HYBRID_RETRIEVAL_PORT",
        default_host=DEFAULT_HOST,
        default_port=DEFAULT_PORT,
    )
    return parser.parse_args()


def _build_dense(docs: list[dict], device: str) -> DenseEmbeddingRetriever | None:
    try:
        encoder = build_e5_encoder(device=device)
        return DenseEmbeddingRetriever(docs, encoder=encoder)
    except Exception as exc:  # missing deps, model download, MPS unavailable
        logger.warning(
            "Dense leg unavailable, serving TF-IDF only. This usually means the "
            "embedding stack is misconfigured: sentence-transformers < 3.0, or a "
            "torch/torchvision version mismatch from a shared env. Install into a "
            "clean venv per docs/hybrid-dense-setup.md. Cause: %s",
            exc,
        )
        return None


def main() -> None:
    load_environment()
    args = parse_args()
    from src.internal.servers.retrieval.corpus_registry import resolve_corpus_docs

    docs = resolve_corpus_docs(args.corpus or args.corpus_path)
    sparse = TfidfRetriever.from_docs(docs)
    dense = None if args.no_dense else _build_dense(docs, args.device)
    app = create_app(dense=dense, sparse=sparse)
    run_uvicorn_app(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
