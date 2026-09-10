"""Lightweight demo retrieval server — no index, no Java, no pyserini.

Loads corpus.jsonl at startup and scores documents with TF-IDF.
Exposes the same /retrieve API as the full retrieval server so the
web frontend works out of the box for local demos.

Usage:
    python3 -m src.internal.servers.demo --corpus_path data/corpus.jsonl
"""

from __future__ import annotations

import argparse
import json

from pydantic import BaseModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.internal.retrieval.acl import acl_allows
from src.internal.servers.app import (
    add_host_port_args,
    create_base_app,
    load_environment,
    run_uvicorn_app,
)

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8001
DEFAULT_TOPK = 5


def _load_corpus(corpus_path: str) -> list[dict]:
    docs = []
    with open(corpus_path) as f:
        for line in f:
            line = line.strip()
            if line:
                docs.append(json.loads(line))
    return docs


class TfidfRetriever:
    def __init__(self, corpus_path: str) -> None:
        self._build(_load_corpus(corpus_path))

    @classmethod
    def from_docs(cls, docs: list[dict]) -> "TfidfRetriever":
        obj = cls.__new__(cls)
        obj._build(docs)
        return obj

    def _build(self, docs: list[dict]) -> None:
        self._docs = docs
        texts = [
            f"{d.get('title', '')} {d.get('contents', d.get('text', ''))}"
            for d in self._docs
        ]
        self._vec = TfidfVectorizer(stop_words="english")
        self._matrix = self._vec.fit_transform(texts)

    def retrieve(self, queries: list[str], topk: int) -> list[list[dict]]:
        results = []
        q_matrix = self._vec.transform(queries)
        scores = cosine_similarity(q_matrix, self._matrix)
        for row_scores in scores:
            # Drop zero-relevance docs (no shared terms) so a query absent from
            # the corpus returns nothing instead of arbitrary top-k filler.
            ranked = sorted(
                (pair for pair in enumerate(row_scores) if pair[1] > 0.0),
                key=lambda x: x[1],
                reverse=True,
            )[:topk]
            results.append(
                [
                    {
                        "document": {
                            "id": self._docs[i].get("id", str(i)),
                            "title": self._docs[i].get("title", ""),
                            "text": self._docs[i].get(
                                "contents", self._docs[i].get("text", "")
                            ),
                            "url": self._docs[i].get("url"),
                            # Forward corpus metadata (incl. the per-document
                            # "source") so result cards can show a real origin.
                            "metadata": self._docs[i].get("metadata") or {},
                        },
                        "score": float(score),
                    }
                    for i, score in ranked
                ]
            )
        return results


class RetrieveRequest(BaseModel):
    queries: list[str] | None = None
    query: str | None = None
    topk: int = DEFAULT_TOPK
    return_scores: bool = False
    filters: dict | None = None

    def resolved_queries(self) -> list[str]:
        if self.queries:
            return self.queries
        if self.query:
            return [self.query]
        return []


def _allowed_by_acl(document: dict, filters: dict | None) -> bool:
    """Whether *document* is readable under *filters* (see ``acl_allows``)."""
    return acl_allows(document.get("metadata"), filters)


def create_app(retriever: TfidfRetriever, *, ignore_acl: bool = False):
    app = create_base_app("Demo Retrieval Server")

    @app.post("/retrieve")
    def retrieve_endpoint(body: RetrieveRequest):
        queries = body.resolved_queries()
        rows = retriever.retrieve(queries, topk=body.topk)
        filters = None if ignore_acl else body.filters
        rows = [
            [item for item in row if _allowed_by_acl(item["document"], filters)]
            for row in rows
        ]
        if not body.return_scores:
            rows = [[item["document"] for item in row] for row in rows]
        if body.query is not None:
            return {"results": rows[0] if rows else []}
        return {"results": rows}

    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Demo TF-IDF retrieval server")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--corpus_path", type=str, help="Path to a corpus .jsonl file")
    source.add_argument(
        "--corpus",
        type=str,
        help="Registered corpus name, comma-list, or 'all' (see data/corpora.json)",
    )
    parser.add_argument("--topk", type=int, default=DEFAULT_TOPK)
    parser.add_argument(
        "--ignore-acl",
        action="store_true",
        help=(
            "Serve documents regardless of the request's access_acl, so a "
            "client's own enforcement can be tested with nothing behind it."
        ),
    )
    add_host_port_args(
        parser,
        "DEMO_RETRIEVAL_HOST",
        "DEMO_RETRIEVAL_PORT",
        default_host=DEFAULT_HOST,
        default_port=DEFAULT_PORT,
    )
    return parser.parse_args()


def main() -> None:
    load_environment()
    args = parse_args()
    # Imported here (not at module top) to avoid a circular import:
    # corpus_registry imports _load_corpus from this module.
    from src.internal.servers.retrieval.corpus_registry import resolve_corpus_docs

    docs = resolve_corpus_docs(args.corpus or args.corpus_path)
    retriever = TfidfRetriever.from_docs(docs)
    app = create_app(retriever, ignore_acl=args.ignore_acl)
    run_uvicorn_app(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
