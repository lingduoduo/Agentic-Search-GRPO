"""Measure how routing and retrieval handle follow-up turns and topic switches.

History reaches every chat surface but only the answer prompt reads it: intent
routing and the default retrieval paths see the latest user message alone. This
harness replays hand-written conversations and scores each user turn under four
query conditions:

- ``raw``: the message as typed — today's behaviour;
- ``regex``: ``build_retrieval_context``, the existing deterministic rewriter
  that only the degraded no-LLM paths use;
- ``concat``: the previous user message glued to this one;
- ``gold_rewrite``: the hand-written standalone query — the ceiling a perfect
  resolver reaches.

It reports route accuracy (``recognize_intent`` with no LLM, clarify = miss),
retrieval Hit@5 / MRR@10, and the topic-switch carry-over rate, each difference
against ``raw`` with a 95% CI that resamples conversations.

Run:

    python -m examples.measure_multi_turn_continuity

``--routers rules --retrievers tfidf`` runs without sentence-transformers.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from functools import lru_cache, partial
from pathlib import Path

from src.context import ChatMessage
from src.internal.search.context import (
    Resolution,
    build_retrieval_context,
    resolve_follow_up,
)
from src.model.post_training.eval.stats import cluster_bootstrap_ci

DEFAULT_DATA = Path("data/eval/multi_turn_conversations.jsonl")
DEFAULT_OUT = Path("data/eval/multi_turn_continuity.json")
INTENT_INDEX_DIR = Path("data/intent_index")

ROUTES = ("chat", "search", "tool")
RELATIONS = ("opening", "follow_up", "topic_switch")
KINDS = ("pronoun", "ellipsis", "comparison", "elaboration")
CONDITIONS = ("raw", "regex", "concat", "gold_rewrite")
HIT_K = 5
MRR_K = 10
TAU_GRID = tuple(round(0.70 + 0.01 * i, 2) for i in range(26))
CARRY_OVER_MAX = 0.15
LATENCY_MAX_MS = 30.0

_TURN_KEYS = {
    "text",
    "route",
    "relation",
    "kind",
    "gold_rewrite",
    "corpus",
    "relevant_doc_ids",
    "beir_query_id",
}


@dataclass(frozen=True)
class Turn:
    text: str
    route: str
    relation: str
    gold_rewrite: str
    kind: str | None = None
    corpus: str | None = None
    relevant_doc_ids: tuple[str, ...] = ()
    beir_query_id: str | None = None


@dataclass(frozen=True)
class Conversation:
    id: str
    turns: tuple[Turn, ...]


def _parse_turn(where: str, raw: dict) -> Turn:
    unknown = set(raw) - _TURN_KEYS
    if unknown:
        raise ValueError(f"{where}: unknown keys {sorted(unknown)}")
    for key in ("text", "route", "relation", "gold_rewrite"):
        if not str(raw.get(key, "")).strip():
            raise ValueError(f"{where}: missing {key}")
    turn = Turn(
        text=raw["text"],
        route=raw["route"],
        relation=raw["relation"],
        gold_rewrite=raw["gold_rewrite"],
        kind=raw.get("kind"),
        corpus=raw.get("corpus"),
        relevant_doc_ids=tuple(str(d) for d in raw.get("relevant_doc_ids", ())),
        beir_query_id=raw.get("beir_query_id"),
    )
    if turn.route not in ROUTES:
        raise ValueError(f"{where}: unknown route {turn.route!r}")
    if turn.relation not in RELATIONS:
        raise ValueError(f"{where}: unknown relation {turn.relation!r}")
    if (turn.relation == "follow_up") != (turn.kind is not None):
        raise ValueError(f"{where}: kind is required on follow-ups and only there")
    if turn.kind is not None and turn.kind not in KINDS:
        raise ValueError(f"{where}: unknown kind {turn.kind!r}")
    if turn.relation != "follow_up" and turn.gold_rewrite != turn.text:
        raise ValueError(
            f"{where}: a {turn.relation} turn's gold_rewrite must equal its text"
        )
    has_both = bool(turn.corpus and turn.relevant_doc_ids)
    has_either = bool(turn.corpus or turn.relevant_doc_ids)
    if has_both != (turn.route == "search") or has_either != has_both:
        raise ValueError(
            f"{where}: corpus and relevant_doc_ids are required on search turns "
            "and only there"
        )
    return turn


def load_conversations(path: Path) -> list[Conversation]:
    conversations: list[Conversation] = []
    seen: set[str] = set()
    for line_no, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        raw = json.loads(line)
        conv_id = str(raw.get("id", "")).strip()
        if not conv_id:
            raise ValueError(f"line {line_no}: missing id")
        if conv_id in seen:
            raise ValueError(f"{conv_id}: duplicate id")
        seen.add(conv_id)
        raw_turns = raw.get("turns") or []
        if len(raw_turns) < 2:
            raise ValueError(f"{conv_id}: needs at least 2 turns")
        turns = tuple(
            _parse_turn(f"{conv_id} turn {i}", t) for i, t in enumerate(raw_turns)
        )
        if turns[0].relation != "opening":
            raise ValueError(f"{conv_id} turn 0: first turn must be an opening")
        if any(t.relation == "opening" for t in turns[1:]):
            raise ValueError(f"{conv_id}: only the first turn may be an opening")
        conversations.append(Conversation(conv_id, turns))
    if not conversations:
        raise ValueError(f"{path}: no conversations")
    return conversations


# No condition reads assistant text; the placeholder only keeps the history
# shaped like a real session (user/assistant alternating).
PLACEHOLDER_REPLY = "Here is what I found."


def _history(prior_texts: list[str]) -> list[ChatMessage]:
    history: list[ChatMessage] = []
    for text in prior_texts:
        history.append(ChatMessage(role="user", content=text))
        history.append(ChatMessage(role="assistant", content=PLACEHOLDER_REPLY))
    return history


def build_query(condition: str, conversation: Conversation, index: int) -> str:
    turn = conversation.turns[index]
    prior = [t.text for t in conversation.turns[:index]]
    if condition == "raw":
        return turn.text
    if condition == "gold_rewrite":
        return turn.gold_rewrite
    if condition == "concat":
        return f"{prior[-1]}\n{turn.text}" if prior else turn.text
    if condition == "regex":
        return build_retrieval_context(turn.text, _history(prior)).retrieval_query
    raise ValueError(f"unknown condition {condition!r}")


Router = Callable[[str], str]
Retriever = Callable[[str, str], list[str]]


def score_retrieval(ranked: list[str], relevant: tuple[str, ...]) -> tuple[bool, float]:
    wanted = set(relevant)
    hit = any(doc_id in wanted for doc_id in ranked[:HIT_K])
    rr = next(
        (
            1.0 / rank
            for rank, doc_id in enumerate(ranked[:MRR_K], 1)
            if doc_id in wanted
        ),
        0.0,
    )
    return hit, rr


Resolver = Callable[[str, list[ChatMessage]], Resolution]


def split_of(conversation_id: str) -> str:
    digest = hashlib.sha256(conversation_id.encode()).digest()
    return "dev" if digest[0] % 2 == 0 else "test"


def evaluate(
    conversations: Sequence[Conversation],
    routers: dict[str, Router],
    retrievers: dict[str, Retriever],
    resolvers: dict[str, Resolver] | None = None,
) -> list[dict]:
    resolvers = resolvers or {}
    rows: list[dict] = []
    for conversation in conversations:
        for index, turn in enumerate(conversation.turns):
            for condition in (*CONDITIONS, *resolvers):
                continuation: bool | None = None
                gate_reason: str | None = None
                if condition in resolvers:
                    prior = [t.text for t in conversation.turns[:index]]
                    resolution = resolvers[condition](turn.text, _history(prior))
                    query = resolution.query
                    continuation = resolution.continuation
                    gate_reason = resolution.reason
                else:
                    query = build_query(condition, conversation, index)
                retrieval: dict[str, dict] = {}
                if turn.route == "search":
                    for name, retrieve in retrievers.items():
                        hit, rr = score_retrieval(
                            retrieve(turn.corpus, query), turn.relevant_doc_ids
                        )
                        retrieval[name] = {"hit5": hit, "rr10": rr}
                rows.append(
                    {
                        "conversation_id": conversation.id,
                        "turn_index": index,
                        "relation": turn.relation,
                        "kind": turn.kind,
                        "corpus": turn.corpus,
                        "gold_route": turn.route,
                        "condition": condition,
                        "query": query,
                        # Read on topic-switch turns only, where the gold query is
                        # the text itself: any difference is prior-turn text.
                        "carried": query != turn.text,
                        "route": {
                            name: route(query) for name, route in routers.items()
                        },
                        "retrieval": retrieval,
                        "continuation": continuation,
                        "gate_reason": gate_reason,
                    }
                )
    return rows


ValueOf = Callable[[dict], "float | None"]


def _slices(corpora: Sequence[str]) -> dict[str, Callable[[dict], bool]]:
    slices: dict[str, Callable[[dict], bool]] = {"all": lambda row: True}
    for relation in RELATIONS:
        slices[relation] = lambda row, relation=relation: row["relation"] == relation
    for kind in KINDS:
        slices[f"follow_up/{kind}"] = lambda row, kind=kind: row["kind"] == kind
    for relation in RELATIONS:
        for corpus in corpora:
            slices[f"{relation}/{corpus}"] = (
                lambda row, relation=relation, corpus=corpus: row["relation"]
                == relation
                and row.get("corpus") == corpus
            )
    return slices


def _metrics(routers: Sequence[str], retrievers: Sequence[str]) -> dict[str, ValueOf]:
    metrics: dict[str, ValueOf] = {}
    for name in routers:
        metrics[f"route_acc:{name}"] = lambda row, name=name: float(
            row["route"][name] == row["gold_route"]
        )
    for name in retrievers:
        metrics[f"hit5:{name}"] = lambda row, name=name: (
            float(row["retrieval"][name]["hit5"]) if row["retrieval"] else None
        )
        metrics[f"mrr10:{name}"] = lambda row, name=name: (
            row["retrieval"][name]["rr10"] if row["retrieval"] else None
        )
    metrics["carry_over"] = lambda row: (
        float(row["carried"]) if row["relation"] == "topic_switch" else None
    )
    return metrics


def _by_conversation(
    rows: Sequence[dict],
    condition: str,
    in_slice: Callable[[dict], bool],
    value_of: ValueOf,
) -> dict[str, list[float]]:
    grouped: dict[str, list[float]] = {}
    for row in rows:
        if row["condition"] != condition or not in_slice(row):
            continue
        value = value_of(row)
        if value is not None:
            grouped.setdefault(row["conversation_id"], []).append(value)
    return grouped


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values)


def _paired_difference(units: Sequence[tuple[list[float], list[float]]]) -> float:
    cond_values = [v for unit in units for v in unit[0]]
    raw_values = [v for unit in units for v in unit[1]]
    return _mean(cond_values) - _mean(raw_values)


def summarize(
    rows: Sequence[dict],
    routers: Sequence[str],
    retrievers: Sequence[str],
    *,
    resamples: int,
    seed: int,
) -> dict:
    conditions = list(dict.fromkeys(row["condition"] for row in rows))
    corpora = sorted({row["corpus"] for row in rows if row.get("corpus")})
    summary: dict = {}
    for slice_name, in_slice in _slices(corpora).items():
        for metric_name, value_of in _metrics(routers, retrievers).items():
            raw = _by_conversation(rows, "raw", in_slice, value_of)
            if not raw:
                continue
            per_condition: dict = {}
            for condition in conditions:
                grouped = _by_conversation(rows, condition, in_slice, value_of)
                values = [v for conv_values in grouped.values() for v in conv_values]
                delta = None
                if condition != "raw":
                    # Units are conversations: each carries its (condition, raw)
                    # values for the same turns, so the difference stays paired.
                    units = [(grouped[conv], raw[conv]) for conv in raw]
                    point, low, high = cluster_bootstrap_ci(
                        units, _paired_difference, resamples=resamples, seed=seed
                    )
                    delta = {"point": point, "low": low, "high": high}
                per_condition[condition] = {
                    "mean": _mean(values),
                    "n": len(values),
                    "delta_vs_raw": delta,
                }
            summary.setdefault(slice_name, {})[metric_name] = per_condition
    return summary


def gate_accuracy(rows: Sequence[dict], condition: str) -> dict:
    scored = [
        float(row["continuation"] == (row["relation"] == "follow_up"))
        for row in rows
        if row["condition"] == condition and row["relation"] != "opening"
    ]
    return {"mean": _mean(scored) if scored else None, "n": len(scored)}


def _resolve_all(conversations: Sequence[Conversation], cosine, tau: float):
    for conversation in conversations:
        for index, turn in enumerate(conversation.turns):
            if turn.relation == "opening":
                continue
            history = _history([t.text for t in conversation.turns[:index]])
            yield turn, history, partial(resolve_follow_up, cosine=cosine, tau=tau)


def choose_tau(
    conversations: Sequence[Conversation], cosine, grid: Sequence[float] = TAU_GRID
) -> float:
    best_tau, best_acc = grid[0], -1.0
    for tau in grid:
        results = [
            resolve(turn.text, history).continuation == (turn.relation == "follow_up")
            for turn, history, resolve in _resolve_all(conversations, cosine, tau)
        ]
        accuracy = _mean([float(r) for r in results])
        if accuracy >= best_acc:  # ties resolve to the highest τ
            best_tau, best_acc = tau, accuracy
    return best_tau


def median_latency_ms(
    conversations: Sequence[Conversation], cosine, tau: float
) -> float:
    timings = []
    for turn, history, resolve in _resolve_all(conversations, cosine, tau):
        start = time.perf_counter()
        resolve(turn.text, history)
        timings.append((time.perf_counter() - start) * 1000)
    return statistics.median(timings)


def check_criteria(
    summary: dict, rows: Sequence[dict], latency_ms: float
) -> dict[str, bool]:
    hit = summary["follow_up/scifact"]["hit5:tfidf"]
    gated, concat = hit["gated"], hit["concat"]
    return {
        "gated_beats_raw": gated["delta_vs_raw"]["low"] > 0,
        "carry_over_at_most_0.15": summary["topic_switch"]["carry_over"]["gated"][
            "mean"
        ]
        <= CARRY_OVER_MAX,
        "within_one_hit_of_concat": round(gated["mean"] * gated["n"])
        >= round(concat["mean"] * concat["n"]) - 1,
        "latency_at_most_30ms": latency_ms <= LATENCY_MAX_MS,
    }


def make_router(name: str, index_dir: Path = INTENT_INDEX_DIR) -> Router:
    from src.internal.configs import load_app_settings
    from src.internal.servers.web.intent import similarity
    from src.internal.servers.web.intent.recognizer import recognize_intent

    if name not in ("rules", "knn"):
        raise ValueError(f"unknown router {name!r}")
    settings = replace(
        load_app_settings(),
        intent_index_path=index_dir if name == "knn" else None,
        intent_shadow_mode=False,
        route_clarification=True,
    )
    if name == "knn" and similarity.load_intent_index(settings) is None:
        raise SystemExit(f"knn router: intent index did not load from {index_dir}")
    # The index loads from numpy alone; the encoder is only touched per query and
    # predict_route turns its failure into None, which would route like `rules`.
    if name == "knn" and similarity.predict_route("probe", settings=settings) is None:
        raise SystemExit("knn router: the intent encoder could not predict a route")

    def route(query: str) -> str:
        decision = recognize_intent(
            query, llm=None, explicit_source=False, settings=settings
        )
        if decision.clarification is not None:
            return "clarify"
        return decision.strategy.value

    return route


def load_corpora(names: set[str]) -> dict[str, list[dict]]:
    from src.internal.servers.retrieval.corpus_registry import (
        load_manifest,
        resolve_corpus_docs,
    )

    manifest = load_manifest()
    corpora: dict[str, list[dict]] = {}
    for name in sorted(names):
        entry = manifest.get(name)
        if entry is None:
            raise SystemExit(f"corpus {name!r} is not registered in data/corpora.json")
        path = Path(entry["path"] if isinstance(entry, dict) else entry)
        # resolve_corpus_docs only warns and skips a missing file, which would
        # score fewer turns instead of failing.
        if not path.exists():
            raise SystemExit(f"corpus {name!r} missing: {path}")
        corpora[name] = resolve_corpus_docs(name, manifest)
    return corpora


def check_relevant_ids(
    conversations: Sequence[Conversation], corpora: dict[str, list[dict]]
) -> None:
    known = {name: {str(d.get("id")) for d in docs} for name, docs in corpora.items()}
    missing = sorted(
        f"{conv.id} turn {i}: {doc_id}"
        for conv in conversations
        for i, turn in enumerate(conv.turns)
        if turn.corpus
        for doc_id in turn.relevant_doc_ids
        if doc_id not in known.get(turn.corpus, set())
    )
    if missing:
        raise ValueError("relevant_doc_ids not in corpus: " + "; ".join(missing))


def make_retriever(name: str, corpora: dict[str, list[dict]], device: str) -> Retriever:
    from src.internal.servers.retrieval.demo import TfidfRetriever

    if name not in ("tfidf", "hybrid"):
        raise ValueError(f"unknown retriever {name!r}")
    sparse = {c: TfidfRetriever.from_docs(docs) for c, docs in corpora.items()}
    dense = {}
    if name == "hybrid":
        # Built directly, not via hybrid._build_dense: that one degrades to
        # TF-IDF on failure, which would make `hybrid` silently equal `tfidf`.
        from src.internal.servers.retrieval.hybrid import (
            DenseEmbeddingRetriever,
            build_e5_encoder,
        )

        encoder = build_e5_encoder(device=device)
        dense = {
            c: DenseEmbeddingRetriever(docs, encoder=encoder)
            for c, docs in corpora.items()
        }

    def retrieve(corpus: str, query: str) -> list[str]:
        if name == "hybrid":
            from src.internal.servers.retrieval.hybrid import _fuse_rows

            # Mirrors the hybrid server: each leg fetches 2x, RRF keeps MRR_K.
            fetch_k = MRR_K * 2
            rows = _fuse_rows(
                dense[corpus].retrieve([query], topk=fetch_k),
                sparse[corpus].retrieve([query], topk=fetch_k),
                MRR_K,
            )[0]
        else:
            rows = sparse[corpus].retrieve([query], topk=MRR_K)[0]
        return [str(item["document"]["id"]) for item in rows]

    return retrieve


def format_table(summary: dict) -> str:
    lines: list[str] = []
    for slice_name in (
        "all",
        "follow_up",
        "topic_switch",
        "follow_up/scifact",
        "topic_switch/scifact",
    ):
        for metric_name, per_condition in summary.get(slice_name, {}).items():
            cells = []
            for condition, entry in per_condition.items():
                cell = f"{condition} {entry['mean']:.2f}"
                delta = entry["delta_vs_raw"]
                if delta is not None:
                    cell += (
                        f" ({delta['point']:+.2f} "
                        f"[{delta['low']:+.2f},{delta['high']:+.2f}])"
                    )
                cells.append(cell)
            n = per_condition["raw"]["n"]
            lines.append(
                f"{slice_name:<13} {metric_name:<16} n={n:<4} " + " | ".join(cells)
            )
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument(
        "--routers", nargs="+", default=["rules", "knn"], choices=["rules", "knn"]
    )
    parser.add_argument(
        "--retrievers",
        nargs="+",
        default=["tfidf", "hybrid"],
        choices=["tfidf", "hybrid"],
    )
    parser.add_argument("--device", default="mps")
    parser.add_argument("--resamples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tau", type=float, default=None)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(argv)

    conversations = load_conversations(args.data)
    corpora = load_corpora(
        {t.corpus for c in conversations for t in c.turns if t.corpus}
    )
    check_relevant_ids(conversations, corpora)
    routers = {name: make_router(name) for name in args.routers}
    retrievers = {
        name: make_retriever(name, corpora, args.device) for name in args.retrievers
    }

    from src.internal.utils import embedding_gate

    embedder = embedding_gate.gate_embedder()
    if embedder is None:
        raise SystemExit("gated condition needs the e5 gate embedder; it did not load")
    cosine = embedding_gate.make_cosine_fn(embedder)
    cached = lru_cache(maxsize=None)(cosine)
    dev = [c for c in conversations if split_of(c.id) == "dev"]
    test = [c for c in conversations if split_of(c.id) == "test"]
    # τ is chosen on dev only; every reported number is from the test half.
    tau = args.tau if args.tau is not None else choose_tau(dev, cached)
    resolvers = {
        "gated": partial(resolve_follow_up, cosine=cached, tau=tau),
        "gated_cues": partial(resolve_follow_up, cosine=None, tau=tau),
    }
    rows = evaluate(test, routers, retrievers, resolvers)
    summary = summarize(
        rows, args.routers, args.retrievers, resamples=args.resamples, seed=args.seed
    )
    # Uncached: the serving path encodes on every request.
    latency = median_latency_ms(test, cosine, tau)
    gates = {name: gate_accuracy(rows, name) for name in resolvers}
    criteria = check_criteria(summary, rows, latency)
    print(format_table(summary))
    print(f"tau={tau} (dev conversations={len(dev)}, test conversations={len(test)})")
    print(f"gate accuracy: {gates}")
    print(f"median resolve latency: {latency:.1f} ms")
    print(f"criteria: {criteria}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    config = {key: str(value) for key, value in vars(args).items()}
    config["conversations"] = len(conversations)
    args.out.write_text(
        json.dumps(
            {
                "config": config,
                "tau": tau,
                "split": {
                    "dev": [c.id for c in dev],
                    "test": [c.id for c in test],
                },
                "gate_accuracy": gates,
                "latency_ms": latency,
                "criteria": criteria,
                "summary": summary,
                "rows": rows,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
