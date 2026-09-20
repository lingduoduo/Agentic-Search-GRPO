"""Query expansion: acronym substitution and optional spell correction.

All features are opt-in via env vars. When disabled (default), expand() is a
no-op and adds zero latency. symspellpy is imported lazily so the module can
be imported even when that package is absent.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any
from pathlib import Path

logger = logging.getLogger(__name__)

# Bundled default acronym table. It lives here, not under data/, because that
# directory is gitignored -- a default file there would be invisible to every
# clone. ACRONYM_PATH overrides this file rather than merging with it.
DEFAULT_ACRONYM_PATH = Path(__file__).resolve().parent / "acronyms.json"


def _read_acronym_file(path) -> dict[str, str]:
    """Read an acronym JSON file. Never raises: an unusable file yields {}."""
    try:
        with open(path) as handle:
            return {k.upper(): v for k, v in json.load(handle).items()}
    except FileNotFoundError:
        logger.warning("Acronym file not found: %s", path)
    except (ValueError, TypeError, AttributeError) as exc:
        logger.warning("Acronym file could not be read (%s): %s", path, exc)
    return {}


def _build_symspell() -> Any:
    try:
        from symspellpy import SymSpell  # type: ignore[import]

        sym = SymSpell(max_dictionary_edit_distance=2)
        dict_path = os.path.join(
            os.path.dirname(__file__),
            "data",
            "frequency_dictionary_en_82_765.txt",
        )
        if os.path.exists(dict_path):
            sym.load_dictionary(dict_path, term_index=0, count_index=1)
        return sym
    except ImportError:
        logger.warning("symspellpy not installed; spell correction disabled")
        return None


class QueryOptimizer:
    """Expands a raw query string for the BM25 leg only.

    Args:
        acronym_path: Path to a JSON dict {"ACRONYM": "expansion"}.
                      Pass None to disable acronym expansion.
        max_terms:    Maximum number of extra terms to inject (prevents query bloat).
        spell_enabled: Enable symspellpy spell correction.
    """

    def __init__(
        self,
        acronym_path: str | None,
        *,
        corpus_path=None,
        max_terms: int = 3,
        spell_enabled: bool = False,
    ) -> None:
        # Three sources, strict precedence: an explicit file wins; otherwise the
        # corpus defines its own vocabulary; otherwise the bundled table. A
        # static table assumes a domain -- scifact's IR is ionizing radiation,
        # not information retrieval. See #601.
        self._acronyms: dict[str, str] = {}
        if acronym_path:
            source = str(acronym_path)
            self._acronyms = _read_acronym_file(acronym_path)
        else:
            from .acronym_extraction import extract_acronyms_from_corpus

            derived = extract_acronyms_from_corpus(corpus_path)
            if derived:
                self._acronyms = derived
                source = f"corpus:{corpus_path}"
            else:
                self._acronyms = _read_acronym_file(DEFAULT_ACRONYM_PATH)
                source = f"bundled:{DEFAULT_ACRONYM_PATH.name}"
        if self._acronyms:
            logger.info(
                "Query expansion: %d acronyms from %s", len(self._acronyms), source
            )
        else:
            logger.warning(
                "Query expansion is enabled but no acronyms were loaded from %s; "
                "expansion will have no effect.",
                source,
            )
        self._max_terms = max_terms
        self._sym = _build_symspell() if spell_enabled else None

    @classmethod
    def from_env(cls) -> "QueryOptimizer":
        expansion = os.environ.get("QUERY_EXPANSION_ENABLED", "false").lower() in (
            "1",
            "true",
            "yes",
        )
        spell = os.environ.get("SPELL_CORRECTION_ENABLED", "false").lower() in (
            "1",
            "true",
            "yes",
        )
        if not expansion and not spell:
            return _PassthroughOptimizer()
        return cls(
            os.environ.get("ACRONYM_PATH"),
            corpus_path=(
                os.environ.get("ACRONYM_CORPUS_PATH")
                or os.environ.get("BM25_CORPUS_PATH")
                or "data/corpus.jsonl"
            ),
            max_terms=int(os.environ.get("EXPANSION_MAX_TERMS", "3")),
            spell_enabled=spell,
        )

    def expand(self, query: str) -> str:
        if self._sym is not None:
            suggestions = self._sym.lookup_compound(query, max_edit_distance=2)
            if suggestions:
                query = suggestions[0].term

        extra: list[str] = []
        for token in query.split():
            expansion = self._acronyms.get(token.upper())
            if expansion and expansion.lower() not in query.lower():
                extra.append(expansion)
            if len(extra) >= self._max_terms:
                break

        return (query + " " + " ".join(extra)).strip() if extra else query


class _PassthroughOptimizer(QueryOptimizer):
    """No-op optimizer used when all expansion features are disabled."""

    def __init__(self) -> None:  # type: ignore[override]
        self._acronyms = {}
        self._max_terms = 0
        self._sym = None

    def expand(self, query: str) -> str:
        return query
