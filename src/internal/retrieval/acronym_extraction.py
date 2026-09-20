"""Extract the acronym pairs a corpus defines about itself.

A static acronym table assumes a domain. scifact uses ``IR`` for ionizing
radiation; this repository uses it for information retrieval. Rather than
guess, read the gloss the corpus already writes: ``ionizing radiation (IR)``.

Only the ``long form (ABBR)`` shape is recognised, and only when the
abbreviation's letters match the initials of the trailing words -- without that
check the pattern matches any parenthesised capital, which is most citations.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

logger = logging.getLogger(__name__)

# A short run of words, then a parenthesised candidate abbreviation. The word
# cap bounds the backtracking and reflects that glosses are short.
_GLOSS = re.compile(
    r"([A-Za-z][A-Za-z\-]*(?:\s+[A-Za-z][A-Za-z\-]*){0,4})\s*\(([A-Z][A-Za-z0-9\-]{1,7})\)"
)

_MAX_TEXT_CHARS = 20_000


def extract_acronyms(text: str) -> dict[str, str]:
    """Return ``{ACRONYM: expansion}`` for every verified gloss in *text*."""
    pairs: dict[str, str] = {}
    for long_form, abbreviation in _GLOSS.findall(text[:_MAX_TEXT_CHARS]):
        words = long_form.split()
        target = abbreviation.lower().replace("-", "")
        for count in range(1, len(words) + 1):
            candidate = words[-count:]
            if "".join(w[0].lower() for w in candidate) == target:
                pairs.setdefault(abbreviation.upper(), " ".join(candidate).lower())
                break
    return pairs


def extract_acronyms_from_corpus(path) -> dict[str, str]:
    """Extract from a corpus.jsonl. Never raises: an unusable corpus yields {}."""
    if not path:
        return {}
    corpus = Path(path)
    if not corpus.is_file():
        logger.debug("Acronym extraction: no corpus at %s", corpus)
        return {}

    pairs: dict[str, str] = {}
    try:
        with corpus.open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except ValueError:
                    continue  # one bad line must not lose the whole corpus
                if not isinstance(record, dict):
                    continue
                # `contents` first, `text` second: corpus.jsonl in this repo is
                # {id, title, contents, metadata}, and the retrieval servers read
                # it the same way. Reading only `text` sees titles and nothing else.
                body = record.get("contents") or record.get("text") or ""
                text = f"{record.get('title', '')} {body}"
                for acronym, expansion in extract_acronyms(text).items():
                    pairs.setdefault(acronym, expansion)
    except OSError as exc:
        logger.warning("Acronym extraction failed to read %s: %s", corpus, exc)
        return {}
    return pairs
