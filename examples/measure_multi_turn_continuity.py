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

import json
from dataclasses import dataclass
from pathlib import Path

from src.context import ChatMessage
from src.internal.search.context import build_retrieval_context

DEFAULT_DATA = Path("data/eval/multi_turn_conversations.jsonl")
DEFAULT_OUT = Path("data/eval/multi_turn_continuity.json")
INTENT_INDEX_DIR = Path("data/intent_index")

ROUTES = ("chat", "search", "tool")
RELATIONS = ("opening", "follow_up", "topic_switch")
KINDS = ("pronoun", "ellipsis", "comparison", "elaboration")
CONDITIONS = ("raw", "regex", "concat", "gold_rewrite")
HIT_K = 5
MRR_K = 10

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
