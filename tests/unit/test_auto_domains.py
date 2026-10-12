"""resolve_auto_domains: opt-in, falls back to web, never blocks the loop."""

from __future__ import annotations

import asyncio

import numpy as np
import pytest

from src.internal.tools import category_selection
from src.internal.tools.base import InvalidToolInput
from src.internal.tools.search import normalize_search_domain, resolve_auto_domains
from src.model.pre_training.categories.model import CategoryExample, CategoryIndex


@pytest.fixture(autouse=True)
def _clear_cache():
    category_selection._INDEXES.clear()
    yield
    category_selection._INDEXES.clear()


def _saved_index(tmp_path, encoder="intfloat/e5-small-v2"):
    ex = [CategoryExample("n", "news", "n"), CategoryExample("c", "code", "c")]
    idx = CategoryIndex(
        examples=ex,
        vectors=np.eye(2, dtype=np.float32),
        encoder=encoder,
        fingerprint="f",
    )
    idx.save(tmp_path / "index.npz")
    return tmp_path


@pytest.mark.asyncio
@pytest.mark.parametrize("domains", [None, ["news"], ["news", "code"]])
async def test_non_auto_input_is_returned_unchanged(domains):
    assert await resolve_auto_domains("q", domains) == (domains, None)


@pytest.mark.asyncio
@pytest.mark.parametrize("domains", [["auto", "news"], ["news", "auto"]])
async def test_auto_mixed_with_names_is_rejected(domains):
    with pytest.raises(InvalidToolInput):
        await resolve_auto_domains("q", domains)


def test_single_domain_auto_is_still_an_unknown_category():
    with pytest.raises(InvalidToolInput):
        normalize_search_domain("auto")


@pytest.mark.asyncio
async def test_auto_selects_with_the_configured_index(tmp_path, monkeypatch):
    monkeypatch.setenv(
        "AGENTIC_SEARCH_CATEGORY_INDEX_PATH", str(_saved_index(tmp_path))
    )
    monkeypatch.setenv("AGENTIC_SEARCH_CATEGORY_TOP_K", "1")
    monkeypatch.setenv("AGENTIC_SEARCH_CATEGORY_SELECT_MARGIN", "0")
    monkeypatch.setattr(
        category_selection,
        "encode_texts",
        lambda texts: np.array([[0, 1]], dtype=np.float32),
    )
    assert await resolve_auto_domains("q", ["auto"]) == (
        ["code"],
        {"selected": ["code"], "status": "ok"},
    )


@pytest.mark.asyncio
async def test_unset_path_falls_back_to_web(monkeypatch):
    monkeypatch.delenv("AGENTIC_SEARCH_CATEGORY_INDEX_PATH", raising=False)
    assert await resolve_auto_domains("q", ["auto"]) == (
        ["web"],
        {"selected": ["web"], "status": "unavailable"},
    )


@pytest.mark.asyncio
async def test_a_failed_load_is_cached_and_falls_back(tmp_path, monkeypatch):
    monkeypatch.setenv(
        "AGENTIC_SEARCH_CATEGORY_INDEX_PATH", str(tmp_path)
    )  # no index.npz
    loads = []
    real = category_selection._load
    monkeypatch.setattr(
        category_selection, "_load", lambda p: loads.append(p) or real(p)
    )
    for _ in range(2):
        assert (await resolve_auto_domains("q", ["auto"]))[1]["status"] == "unavailable"
    assert len(loads) == 1


@pytest.mark.asyncio
async def test_a_raising_encoder_falls_back(tmp_path, monkeypatch):
    monkeypatch.setenv(
        "AGENTIC_SEARCH_CATEGORY_INDEX_PATH", str(_saved_index(tmp_path))
    )

    def boom(texts):
        raise RuntimeError("no model")

    monkeypatch.setattr(category_selection, "encode_texts", boom)
    assert (await resolve_auto_domains("q", ["auto"]))[0] == ["web"]


@pytest.mark.asyncio
async def test_encoding_runs_off_the_event_loop(tmp_path, monkeypatch):
    monkeypatch.setenv(
        "AGENTIC_SEARCH_CATEGORY_INDEX_PATH", str(_saved_index(tmp_path))
    )
    monkeypatch.setattr(
        category_selection,
        "encode_texts",
        lambda texts: np.array([[1, 0]], dtype=np.float32),
    )
    used = []
    real = asyncio.to_thread

    async def spy(fn, *a, **k):
        used.append(fn)
        return await real(fn, *a, **k)

    monkeypatch.setattr(asyncio, "to_thread", spy)
    await resolve_auto_domains("q", ["auto"])
    assert used == [category_selection.select_for_query]


@pytest.mark.asyncio
async def test_an_index_from_another_encoder_falls_back(tmp_path, monkeypatch):
    monkeypatch.setenv(
        "AGENTIC_SEARCH_CATEGORY_INDEX_PATH",
        str(_saved_index(tmp_path, encoder="other")),
    )
    monkeypatch.setenv("AGENTIC_SEARCH_CATEGORY_TOP_K", "1")
    monkeypatch.setenv("AGENTIC_SEARCH_CATEGORY_SELECT_MARGIN", "0")
    monkeypatch.setattr(
        category_selection,
        "encode_texts",
        lambda texts: np.array([[0, 1]], dtype=np.float32),
    )
    assert await resolve_auto_domains("q", ["auto"]) == (
        ["web"],
        {"selected": ["web"], "status": "unavailable"},
    )
