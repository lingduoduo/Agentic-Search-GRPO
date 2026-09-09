"""Dependency-free deterministic and fallback intent rules."""

from __future__ import annotations

import re

from .types import RouteStrategy

_SEARCH_RE = re.compile(
    r"\b(find|list|retrieve|search for|show me|pull|get me|look up|fetch)\b",
    re.IGNORECASE,
)
_CHAT_RE = re.compile(
    r"\b(explain|summarize|help me|write|what is|how do|why|difference between|compare|describe)\b",
    re.IGNORECASE,
)
_VERB_RE = re.compile(
    r"\b(is|are|was|were|do|does|did|have|has|can|could|would|should|will)\b",
    re.IGNORECASE,
)
_GENERATIVE_RE = re.compile(
    r"\b(write|translate|rephrase|reword|rewrite|draft|brainstorm|"
    r"joke|poem|haiku)\b",
    re.IGNORECASE,
)
_GREETING_TERM_RE = re.compile(
    r"\b(?:hi|hello|hi there|thanks|thank you)\b", re.IGNORECASE
)
_STANDALONE_GREETING_RE = re.compile(
    r"^(?:hi|hello|hi there|thanks|thank you)[.!?]*$", re.IGNORECASE
)
_TOOL_RE = re.compile(
    r"\b(send|email|create|open (?:a|an) (?:ticket|issue|pr)|file (?:a|an) "
    r"(?:ticket|issue)|schedule|book|call the api|invoke|run the|execute|"
    r"post to|update the|delete the|add to)\b",
    re.IGNORECASE,
)
_TOOL_ACTION_RE = re.compile(
    r"^\s*(send|deploy|assign|notify|remind|invoke|subscribe|unsubscribe)\b",
    re.IGNORECASE,
)
_TOOL_OBJECT_RE = re.compile(
    r"^\s*(?:create|delete|remove|update|add|open|close|file|post|run|execute|"
    r"book|email|schedule|cancel|trigger) "
    r"(?:a |an |the )?"
    r"(?:ticket|issue|pr|pull request|task|event|meeting|reminder|calendar|"
    r"record|entry|api|job|workflow|deployment|message|email)\b",
    re.IGNORECASE,
)
_SEARCH_LOOKUP_RE = re.compile(
    r"^\s*(find|search for|look up|look for|retrieve|fetch|pull|list|locate|"
    r"show me|get me)\b",
    re.IGNORECASE,
)
_CHAT_START_RE = re.compile(
    r"^\s*(what|why|how|explain|describe|summarize|compare|tell me about|"
    r"difference between)\b",
    re.IGNORECASE,
)
_GENERATIVE_START_RE = re.compile(
    r"^\s*(write|draft|translate|rephrase|reword|brainstorm|compose|generate)\b",
    re.IGNORECASE,
)
_CURRENCY_RE = re.compile(
    r"\b(latest|current|recent|news|price|stock|weather|today|now)\b",
    re.IGNORECASE,
)


def _is_standalone_greeting(query: str) -> bool:
    return _STANDALONE_GREETING_RE.fullmatch(query.strip()) is not None


def _is_bare_lookup(query: str) -> bool:
    """Return whether query is a short, verb-less term or entity."""
    q = query.strip()
    if not q or q.endswith("?"):
        return False
    if (
        _TOOL_RE.search(q)
        or _SEARCH_RE.search(q)
        or _GENERATIVE_RE.search(q)
        or _GREETING_TERM_RE.search(q)
        or _CHAT_RE.search(q)
        or _VERB_RE.search(q)
    ):
        return False
    return len(q.split()) <= 3


def _rule_based_route_or_none(query: str) -> RouteStrategy | None:
    """Return a heuristic route, or None when no cue dominates."""
    q = query.strip()
    if not q or _is_standalone_greeting(q):
        return RouteStrategy.CHAT
    if _TOOL_RE.search(q):
        return RouteStrategy.TOOL
    if _SEARCH_RE.search(q):
        return RouteStrategy.SEARCH
    if _is_bare_lookup(q):
        return RouteStrategy.SEARCH
    return None


def _regex_route(query: str) -> RouteStrategy | None:
    """Return a high-precision deterministic route when one is available."""
    q = query.strip()
    if not q:
        return None
    if _is_standalone_greeting(q):
        return RouteStrategy.CHAT
    if _TOOL_ACTION_RE.search(q) or _TOOL_OBJECT_RE.search(q):
        return RouteStrategy.TOOL
    if _is_bare_lookup(q) or _SEARCH_LOOKUP_RE.search(q):
        return RouteStrategy.SEARCH
    if _CHAT_START_RE.search(q) or _GENERATIVE_START_RE.search(q) or q.endswith("?"):
        if _CURRENCY_RE.search(q):
            return None
        return RouteStrategy.CHAT
    return None
