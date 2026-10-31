"""Opt-in local DecisionRAG-inspired server with the standard /retrieve API."""

from __future__ import annotations

import argparse
import logging
from dataclasses import asdict

from fastapi import HTTPException
from pydantic import BaseModel, Field, model_validator

from src.internal.document_index.decision_retriever import (
    DecisionRetriever,
    DecisionRetrieverConfig,
)
from src.internal.servers.app import (
    add_host_port_args,
    create_base_app,
    load_environment,
    run_uvicorn_app,
)

logger = logging.getLogger(__name__)


class RetrieveRequest(BaseModel):
    query: str | None = Field(default=None, min_length=1)
    queries: list[str] | None = Field(default=None, min_length=1, max_length=32)
    topk: int = Field(default=5, ge=1)
    return_scores: bool = False
    filters: dict | None = None
    include_trace: bool = False

    @model_validator(mode="after")
    def validate_queries(self) -> RetrieveRequest:
        if (self.query is None) == (self.queries is None):
            raise ValueError("Provide exactly one of query or queries")
        if any(not query.strip() for query in self.resolved_queries()):
            raise ValueError("Queries must not be blank")
        return self

    def resolved_queries(self) -> list[str]:
        return [self.query] if self.query is not None else self.queries or []


def create_app(retriever: DecisionRetriever):
    app = create_base_app("Decision Navigation Retrieval Server")

    @app.post("/retrieve")
    def retrieve_endpoint(body: RetrieveRequest):
        if body.topk > retriever.config.shortlist_size:
            raise HTTPException(
                422,
                f"topk cannot exceed shortlist_size ({retriever.config.shortlist_size})",
            )
        try:
            runs = [
                retriever.navigate(query, body.topk, filters=body.filters)
                for query in body.resolved_queries()
            ]
        except ValueError as exc:
            logger.warning("Decision retrieval rejected: %s", exc)
            raise HTTPException(422, str(exc)) from exc
        rows = [run.results for run in runs]
        if not body.return_scores:
            rows = [[item["document"] for item in row] for row in rows]
        response: dict = {"results": rows[0] if body.query is not None else rows}
        if body.include_trace:
            traces = [[asdict(stage) for stage in run.stages] for run in runs]
            response["trace"] = traces[0] if body.query is not None else traces
        return response

    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--corpus_path", help="Path to JSONL corpus")
    source.add_argument("--corpus", help="Registered corpus name, comma-list, or 'all'")
    source.add_argument(
        "--hierarchy", help="Validated offline hierarchy JSON (no online embedding)"
    )
    parser.add_argument(
        "--model", required=True, help="Local model path or Hugging Face causal LM"
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--beam-width", type=int, default=5)
    parser.add_argument("--shortlist-size", type=int, choices=range(1, 27), default=20)
    parser.add_argument("--batch-size", type=int, choices=range(1, 27), default=16)
    parser.add_argument("--passage-words", type=int, default=200)
    parser.add_argument("--max-passage-chars", type=int, default=1200)
    for kind in ("group", "document", "section"):
        parser.add_argument(f"--{kind}-beam-width", type=int, default=None)
    parser.add_argument("--min-relevance", type=float, default=0.0)
    parser.add_argument("--max-context-tokens", type=int, default=4096)
    parser.add_argument("--inference-batch-size", type=int, default=1)
    add_host_port_args(
        parser,
        "DECISION_RETRIEVAL_HOST",
        "DECISION_RETRIEVAL_PORT",
        default_host="127.0.0.1",
        default_port=8001,
    )
    return parser.parse_args()


def main() -> None:
    load_environment()
    args = parse_args()
    from src.internal.document_index.decision_model import LocalDecisionModel
    from src.internal.servers.retrieval.corpus_registry import resolve_corpus_docs

    config = DecisionRetrieverConfig(
        beam_width=args.beam_width,
        shortlist_size=args.shortlist_size,
        batch_size=args.batch_size,
        passage_words=args.passage_words,
        min_relevance=args.min_relevance,
        max_passage_chars=args.max_passage_chars,
        beam_widths={
            kind: value
            for kind in ("group", "document", "section")
            if (value := getattr(args, f"{kind}_beam_width")) is not None
        },
    )
    hierarchy = None
    if args.hierarchy:
        from src.internal.document_index.hierarchy import HierarchyArtifact

        hierarchy = HierarchyArtifact.load(args.hierarchy)
        docs = hierarchy.documents
    else:
        docs = resolve_corpus_docs(args.corpus or args.corpus_path)
    model = LocalDecisionModel.from_pretrained(
        args.model,
        device=args.device,
        max_context_tokens=args.max_context_tokens,
        inference_batch_size=args.inference_batch_size,
    )
    retriever = DecisionRetriever(docs, model, config, hierarchy=hierarchy)
    run_uvicorn_app(create_app(retriever), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
