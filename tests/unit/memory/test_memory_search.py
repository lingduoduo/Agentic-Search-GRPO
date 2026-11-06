import numpy as np

from src.internal.db.store import AgenticSearchStore
from src.internal.memory.search import (
    bm25_scores,
    encoder_identity,
    hybrid_search,
)

TEXTS = ["User is allergic to peanuts", "Prefers arXiv", "Lives in Berlin"]


class FakeEncoder:
    """One-hot on topic: 'peanut'/'nuts' -> axis 0, 'arxiv'/'acl' -> axis 1."""

    def __init__(self, model_id="fake-a"):
        self.model_id = model_id
        self.calls: list[list[str]] = []

    def __call__(self, texts):
        self.calls.append(list(texts))
        out = np.zeros((len(texts), 3), dtype=np.float32)
        for i, t in enumerate(texts):
            low = t.lower()
            if "peanut" in low or "nuts" in low:
                out[i, 0] = 1.0
            elif "arxiv" in low or "acl" in low:
                out[i, 1] = 1.0
            else:
                out[i, 2] = 1.0
        return out


class BrokenEncoder:
    model_id = "broken"

    def __call__(self, texts):
        raise RuntimeError("boom")


def _store_with(texts):
    store = AgenticSearchStore(":memory:")
    for t in texts:
        store.add_user_memory("u1", t)
    return store


def _passage_calls(enc):
    return [c for c in enc.calls if c[0].startswith("passage: ")]


def test_bm25_prefers_rarer_term_and_zero_for_no_overlap():
    s = bm25_scores(
        ["peanut"],
        [["peanut", "allergy"], ["thai", "food"], ["peanut", "butter", "peanut"]],
    )
    assert s[1] == 0.0 and s[0] > 0 and s[2] > 0


def test_encoder_identity():
    assert encoder_identity(FakeEncoder("m1")) == "m1"
    assert "test_memory_search" in encoder_identity(lambda t: t) or "lambda" in (
        encoder_identity(lambda t: t)
    )


def test_lexical_only_ranks_and_filters_zero():
    store = _store_with(TEXTS)
    hits = hybrid_search(store, "u1", "allergic peanuts")
    assert [r.memory_text for r, _ in hits] == ["User is allergic to peanuts"]


def test_category_filter():
    store = _store_with(TEXTS)
    assert hybrid_search(store, "u1", "arXiv", category="constraints") == []


def test_dense_leg_fuses_and_caches():
    store = _store_with(TEXTS)
    enc = FakeEncoder()
    hybrid_search(store, "u1", "nuts", encoder=enc)
    hybrid_search(store, "u1", "nuts", encoder=enc)
    assert len(_passage_calls(enc)) == 1
    assert len(store.get_memory_embeddings("u1")) == 3


def test_dense_only_match_is_returned():
    store = _store_with(TEXTS)
    hits = hybrid_search(store, "u1", "nuts", encoder=FakeEncoder())
    assert hits[0][0].memory_text == "User is allergic to peanuts"


def test_update_invalidates_cached_embedding():
    store = _store_with(TEXTS)
    enc = FakeEncoder()
    hybrid_search(store, "u1", "x", encoder=enc)
    rid = store.get_user_memory_records("u1")[1].id
    store.update_user_memory("u1", rid, "Prefers ACL anthology")
    enc.calls.clear()
    hybrid_search(store, "u1", "x", encoder=enc)
    assert enc.calls[0] == ["passage: Prefers ACL anthology"]


def test_encoder_identity_change_recomputes():
    store = _store_with(TEXTS)
    hybrid_search(store, "u1", "x", encoder=FakeEncoder("fake-a"))
    other = FakeEncoder("fake-b")
    hybrid_search(store, "u1", "x", encoder=other)
    assert len(other.calls[0]) == 3


def test_encoder_failure_falls_back_to_lexical():
    store = _store_with(TEXTS)
    hits = hybrid_search(store, "u1", "allergic peanuts", encoder=BrokenEncoder())
    assert [r.memory_text for r, _ in hits] == ["User is allergic to peanuts"]


def test_no_cross_user_leakage():
    store = _store_with(TEXTS)
    assert hybrid_search(store, "u2", "peanuts", encoder=FakeEncoder()) == []
