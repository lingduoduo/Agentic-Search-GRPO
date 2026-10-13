"""Lazy category-index adapter for ``domains=["auto"]``; degrades to web."""

from __future__ import annotations

import logging
import threading
from pathlib import Path

from src.internal.configs import AppSettings, load_app_settings

logger = logging.getLogger(__name__)

_INDEXES: dict[Path, object | None] = {}
# select_for_query runs in worker threads: one cold load per path, and a failure
# is decided once rather than by whichever racing thread wrote last.
_LOCK = threading.Lock()


def encode_texts(texts: list[str]):
    """Load the optional encoder implementation only when selection needs it."""
    from src.model.pre_training.intents.model import encode_texts as encode

    return encode(texts)


def _load(directory: Path):
    from src.model.pre_training.categories.model import INDEX_FILENAME, CategoryIndex
    from src.model.pre_training.intents.model import DEFAULT_ENCODER

    index = CategoryIndex.load(directory / INDEX_FILENAME)
    if index.encoder != DEFAULT_ENCODER:
        raise ValueError(
            f"category-index built with encoder {index.encoder!r}, "
            f"serving uses {DEFAULT_ENCODER!r}"
        )
    return index


def _index_for(settings: AppSettings) -> object | None:
    """Load and cache the configured index by resolved path; failures cache None."""
    configured = settings.category_index_path
    if configured is None:
        return None
    directory = configured.resolve()
    if directory in _INDEXES:
        return _INDEXES[directory]
    with _LOCK:
        if directory in _INDEXES:
            return _INDEXES[directory]
        try:
            _INDEXES[directory] = _load(directory)
        except Exception:
            logger.exception("category-index: load failed — auto selects web")
            _INDEXES[directory] = None
        return _INDEXES[directory]


def select_for_query(query: str) -> tuple[list[str], str]:
    """Return ``(selected categories, status)``; status is ok or unavailable."""
    settings = load_app_settings()
    index = _index_for(settings)
    if index is None:
        return ["web"], "unavailable"
    try:
        vector = encode_texts([query])[0]
        return index.select(
            vector,
            top_k=settings.category_top_k,
            select_margin=settings.category_select_margin,
        ), "ok"
    except Exception:
        logger.exception("category-index: selection failed — searching web")
        return ["web"], "unavailable"
