"""Build, update, and inspect offline corpus hierarchies.

Example: python -m src.internal.document_index.hierarchy_cli build \
    --corpus-path corpus.jsonl --output hierarchy.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .hierarchy import HierarchyArtifact, HierarchyConfig, build_hierarchy
from .hierarchy_formats import normalize_document


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser(
        "build", help="Build a hierarchy from a full corpus snapshot"
    )
    update = commands.add_parser(
        "update", help="Update using a full snapshot; missing IDs are deleted"
    )
    for command in (build, update):
        source = command.add_mutually_exclusive_group(required=True)
        source.add_argument("--corpus", help="Registered corpus spec or JSONL path")
        source.add_argument(
            "--corpus-path",
            "--corpus_path",
            dest="corpus_path",
            help="JSONL corpus path",
        )
        command.add_argument(
            "--output", required=True, help="Destination hierarchy JSON"
        )
        command.add_argument("--device", default="cpu", help="Local model device")
    build.add_argument(
        "--mode", choices=("auto", "structure", "cluster"), default="auto"
    )
    build.add_argument("--embedding-model", default="intfloat/multilingual-e5-base")
    build.add_argument("--description-model")
    build.add_argument("--description-sample-size", type=int, default=12)
    build.add_argument("--target-cluster-size", type=int, default=100)
    build.add_argument("--max-cluster-size", type=int, default=200)
    build.add_argument("--min-cluster-size", type=int, default=5)
    build.add_argument("--max-fanout", type=int, default=None)
    build.add_argument(
        "--clusters",
        type=int,
        help="Explicit semantic cluster count (e.g. 1000 for NQ)",
    )
    build.add_argument("--infer-layout", action="store_true")
    build.add_argument("--random-state", type=int, default=42)
    update.add_argument(
        "--previous",
        required=True,
        help="Previous hierarchy; configuration is inherited",
    )
    inspect = commands.add_parser(
        "inspect", help="Validate an artifact and print its diagnostics"
    )
    inspect.add_argument("artifact", nargs="?", help="Hierarchy JSON")
    inspect.add_argument(
        "--hierarchy",
        dest="hierarchy_path",
        help="Hierarchy JSON (alternative to positional path)",
    )
    return parser


def _load_documents(spec):
    if Path(spec).is_file():
        documents = []
        with open(spec, encoding="utf-8") as stream:
            for number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                try:
                    document = json.loads(line)
                except ValueError as error:
                    raise ValueError(
                        f"{spec}:{number}: invalid JSON: {error}"
                    ) from error
                if not isinstance(document, dict):
                    raise ValueError(f"{spec}:{number}: document must be a JSON object")
                documents.append(document)
        return documents
    from src.internal.servers.retrieval.corpus_registry import resolve_corpus_docs

    return resolve_corpus_docs(spec)


def main(argv=None):
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "inspect":
            path = args.artifact or args.hierarchy_path
            if not path or (args.artifact and args.hierarchy_path):
                raise ValueError("inspect requires one artifact path")
            print(
                json.dumps(
                    HierarchyArtifact.load(path).diagnostics,
                    ensure_ascii=False,
                    sort_keys=True,
                )
            )
            return 0
        previous = (
            HierarchyArtifact.load(args.previous) if args.command == "update" else None
        )
        config = (
            HierarchyConfig(**previous.config)
            if previous
            else HierarchyConfig(
                mode=args.mode,
                embedding_model=args.embedding_model,
                description_model=args.description_model,
                description_sample_size=args.description_sample_size,
                target_cluster_size=args.target_cluster_size,
                max_cluster_size=args.max_cluster_size,
                min_cluster_size=args.min_cluster_size,
                max_fanout=args.max_fanout,
                n_clusters=args.clusters,
                infer_layout=args.infer_layout,
                random_state=args.random_state,
            )
        )
        documents = _load_documents(args.corpus or args.corpus_path)
        # Inspect normalized paths before initializing any optional model adapter.
        semantic_needed = config.mode == "cluster" and bool(documents)
        if config.mode == "auto":
            semantic_needed = any(
                not normalize_document(doc, infer_layout=config.infer_layout)[
                    "structure_path"
                ]
                for doc in documents
            )
        embedder = None
        describer = None
        if semantic_needed:
            from .hierarchy_models import SentenceEmbedding

            embedder = SentenceEmbedding(config.embedding_model, device=args.device)
        if config.description_model:
            from .hierarchy_models import LocalHierarchyDescriber

            describer = LocalHierarchyDescriber(
                config.description_model, device=args.device
            )
        artifact = build_hierarchy(
            documents, config, embedder=embedder, describer=describer, previous=previous
        )
        artifact.save(args.output)
        print(json.dumps(artifact.diagnostics, ensure_ascii=False, sort_keys=True))
        return 0
    except (ValueError, OSError, TypeError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    raise SystemExit(main())
