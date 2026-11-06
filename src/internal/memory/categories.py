"""Fixed memory categories and a keyword classifier.

Must not import from ``src.internal.db``: the store imports this module.
"""

from __future__ import annotations

import re

CATEGORIES: tuple[str, ...] = (
    "constraints",
    "preferences",
    "profile",
    "projects",
    "decisions",
    "general",
)
DEFAULT_CATEGORY = "general"
ROOT_NODE = "__root__"

# Order matters: the first rule that matches wins. ``\b`` at the start only, so
# stems match their inflections ("Prefers", "decided").
_RULES: tuple[tuple[str, re.Pattern[str]], ...] = tuple(
    (name, re.compile(pattern, re.IGNORECASE))
    for name, pattern in (
        (
            "constraints",
            r"\b(allerg\w*|intoleran\w*|vegan|never|must not|can'?t eat|cannot eat)\b",
        ),
        ("decisions", r"\b(decided|chose|chosen|going with)\b"),
        ("preferences", r"\b(prefer\w*|like[sd]?|hate[sd]?|favou?rite)\b"),
        ("projects", r"\b(working on|researching|project|goal)"),
        (
            "profile",
            r"\b(i am|i'?m an?|i live|lives? in|my job)\b"
            r"|\bis an? (\w+ )*(engineer|researcher|student)\b",
        ),
    )
)


def classify_memory(text: str) -> str:
    """Return the category of *text*, or ``"general"`` when no rule matches."""
    for name, pattern in _RULES:
        if pattern.search(text):
            return name
    return DEFAULT_CATEGORY


def normalize_category(value: object) -> str:
    """Return *value* as a known category; unknown or empty becomes ``"general"``."""
    if isinstance(value, str):
        cleaned = value.strip().lower()
        if cleaned in CATEGORIES:
            return cleaned
    return DEFAULT_CATEGORY
