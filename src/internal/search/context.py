"""Deterministic session context construction for document retrieval."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Callable, Iterable

from src.context import ChatMessage


_FOLLOW_UP_PREFIX = re.compile(
    r"^(?:and\b|also\b|but\b|what about\b|how about\b|tell me more\b|"
    r"go on\b|continue\b)",
    re.IGNORECASE,
)
_REFERENCE_FOLLOW_UP = re.compile(
    r"^(?:(?:where|when|what|who|how)\s+"
    r"(?:is|are|was|were|do|does|did|can|could|will|would|has|have|had)\s+"
    r"(?:it|they|them|this|that|these|those|he|him|she|her)\b|"
    r"(?:is|are|was|were|do|does|did|can|could|will|would|has|have|had)\s+"
    r"(?:it|they|this|that|these|those|he|she)\b|"
    r"why\s+(?:is|was|does|did)\s+(?:it|that|this)\b)",
    re.IGNORECASE,
)
_ASSISTANT_INTERNAL_MARKUP = re.compile(
    r"<\s*/?\s*(?:tool(?:_call|_result)?|evidence|search_results?|search|searches|"
    r"fetch|information|search_decision)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class RetrievalContext:
    """Original request plus the bounded session context used for retrieval."""

    query: str
    retrieval_query: str
    history: list[ChatMessage]


def _is_follow_up(query: str) -> bool:
    normalized = query.strip()
    return bool(
        _FOLLOW_UP_PREFIX.search(normalized) or _REFERENCE_FOLLOW_UP.search(normalized)
    )


def _safe_history(history: Iterable[ChatMessage]) -> list[ChatMessage]:
    return [
        message
        for message in history
        if not (
            message.role.lower() == "assistant"
            and _ASSISTANT_INTERNAL_MARKUP.search(message.content)
        )
    ]


def _most_recent_user_topic(history: list[ChatMessage]) -> str | None:
    for message in reversed(history):
        if (
            message.role.lower() == "user"
            and message.content.strip()
            and not _is_follow_up(message.content)
        ):
            return message.content
    return None


def build_retrieval_context(
    query: str,
    history: Iterable[ChatMessage],
    max_messages: int = 6,
) -> RetrievalContext:
    """Build bounded history and resolve simple follow-ups without an LLM."""

    if max_messages < 0:
        raise ValueError("max_messages must be non-negative")

    safe_history = _safe_history(history)
    bounded_history = safe_history[-max_messages:] if max_messages else []
    retrieval_query = query
    if _is_follow_up(query):
        topic = _most_recent_user_topic(bounded_history)
        if topic is not None:
            retrieval_query = f"{topic}\n{query}"

    return RetrievalContext(
        query=query,
        retrieval_query=retrieval_query,
        history=bounded_history,
    )


_REFERENCE = re.compile(
    r"\b(?:it|its|they|them|their|this|that|these|those|one)\b", re.IGNORECASE
)
FRAGMENT_MAX_WORDS = 4
TOPIC_LOOKBACK = 3

Cosine = Callable[[str, str], "float | None"]


@dataclass(frozen=True)
class Resolution:
    """What retrieval searches for, and why."""

    query: str
    continuation: bool
    reason: str  # no_history | reference | fragment | semantic | switch


def _continues(
    message: str, topic: str, cosine: Cosine | None, tau: float
) -> tuple[bool, str]:
    if _REFERENCE.search(message):
        return True, "reference"
    if len(message.split()) <= FRAGMENT_MAX_WORDS:
        return True, "fragment"
    if cosine is not None:
        score = cosine(message, topic)
        if score is not None and score >= tau:
            return True, "semantic"
    return False, "switch"


def resolve_follow_up(
    message: str,
    history: Iterable[ChatMessage],
    *,
    cosine: Cosine | None,
    tau: float,
) -> Resolution:
    """Resolve a follow-up into a standalone retrieval query, without an LLM.

    Cue words ("also", "and", "how about") are deliberately not a signal: alone
    they misfire on topic switches. The topic is the latest earlier user message
    this gate itself judged standalone, within the last TOPIC_LOOKBACK user turns,
    so a chain of follow-ups keeps the original topic and never grows.
    """
    user_texts = [
        m.content
        for m in _safe_history(history)
        if m.role.lower() == "user" and m.content.strip()
    ][-TOPIC_LOOKBACK:]
    topic: str | None = None
    for text in user_texts:
        if topic is None or not _continues(text, topic, cosine, tau)[0]:
            topic = text
    if topic is None:
        return Resolution(message, False, "no_history")
    continues, reason = _continues(message, topic, cosine, tau)
    if continues:
        return Resolution(f"{topic}\n{message}", True, reason)
    return Resolution(message, False, reason)
