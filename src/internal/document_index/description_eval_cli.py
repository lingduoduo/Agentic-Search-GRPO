"""Evaluate description variants with an explicit local decision model."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("hierarchy", "queries", "model", "output"):
        parser.add_argument("--" + flag, required=True)
    parser.add_argument(
        "--variants", help="JSON object mapping variant names to group overrides"
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-context-tokens", type=int, default=4096)
    parser.add_argument("--min-relevance", type=float, default=0.0)
    for flag, default in [
        ("topk", 5),
        ("beam-width", 5),
        ("shortlist-size", 20),
        ("passage-words", 200),
        ("max-passage-chars", 1200),
        ("batch-size", 16),
        ("inference-batch-size", 1),
        ("bootstrap-samples", 1000),
        ("seed", 42),
        ("preview-max-nodes", 0),
        ("preview-chars-per-node", 600),
        ("preview-total-chars", 1800),
        ("preview-children", 4),
        ("preview-scan-limit", 64),
    ]:
        parser.add_argument("--" + flag, type=int, default=default)
    for kind in ("group", "document", "section"):
        parser.add_argument(f"--{kind}-beam-width", type=int)
    return parser


def main(argv=None):
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        # Import only after argparse: --help does not import ML libraries.
        from .decision_model import LocalDecisionModel
        from .decision_retriever import DecisionRetrieverConfig
        from .description_eval import builtin_variants, evaluate_descriptions
        from .hierarchy import HierarchyArtifact

        model_path = Path(args.model)
        if not model_path.is_dir():
            raise ValueError("--model must name an existing local model directory")
        artifact = HierarchyArtifact.load(args.hierarchy)
        cases = []
        with open(args.queries, encoding="utf-8") as stream:
            for number, line in enumerate(stream, 1):
                if line.strip():
                    try:
                        cases.append(json.loads(line))
                    except ValueError as error:
                        raise ValueError(f"{args.queries}:{number}: {error}") from error
        variants = builtin_variants(artifact)
        if args.variants:
            with open(args.variants, encoding="utf-8") as stream:
                custom = json.load(stream)
            if not isinstance(custom, dict) or set(custom).intersection(variants):
                raise ValueError(
                    "custom variants must be an object with new names; built-in collisions forbidden"
                )
            variants.update(custom)
        config = DecisionRetrieverConfig(
            beam_width=args.beam_width,
            shortlist_size=args.shortlist_size,
            passage_words=args.passage_words,
            max_passage_chars=args.max_passage_chars,
            batch_size=args.batch_size,
            min_relevance=args.min_relevance,
            beam_widths={
                kind: getattr(args, kind + "_beam_width")
                for kind in ("group", "document", "section")
                if getattr(args, kind + "_beam_width") is not None
            },
            **{
                name: getattr(args, name)
                for name in (
                    "preview_max_nodes",
                    "preview_chars_per_node",
                    "preview_total_chars",
                    "preview_children",
                    "preview_scan_limit",
                )
            },
        )
        model = LocalDecisionModel.from_pretrained(
            str(model_path),
            device=args.device,
            max_context_tokens=args.max_context_tokens,
            inference_batch_size=args.inference_batch_size,
        )
        report = evaluate_descriptions(
            artifact,
            model,
            cases,
            variants,
            config,
            topk=args.topk,
            bootstrap_samples=args.bootstrap_samples,
            seed=args.seed,
        )
        report["model"]["local_path"] = str(model_path.resolve())
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=output.parent,
                suffix=".tmp",
                delete=False,
            ) as stream:
                temporary = stream.name
                json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
                stream.write("\n")
            os.replace(temporary, output)
        finally:
            if temporary and os.path.exists(temporary):
                os.unlink(temporary)
    except (ValueError, OSError, TypeError) as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
