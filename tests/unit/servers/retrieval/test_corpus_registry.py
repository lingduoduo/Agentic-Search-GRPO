import json

import pytest

from src.internal.servers.retrieval.corpus_registry import (
    load_manifest,
    register_corpus,
    resolve_corpus_docs,
    unregister_corpus,
)
from src.internal.servers.retrieval.demo import TfidfRetriever


def _write_corpus(path, docs):
    with open(path, "w") as f:
        for d in docs:
            f.write(json.dumps(d) + "\n")


@pytest.fixture
def manifest(tmp_path):
    """Two corpora with disjoint id namespaces (the expected, mergeable case)."""
    _write_corpus(
        tmp_path / "a.jsonl", [{"id": "a1", "title": "A1", "contents": "alpha"}]
    )
    _write_corpus(
        tmp_path / "b.jsonl",
        [
            {"id": "b1", "title": "B1", "contents": "beta"},
            {"id": "b2", "title": "B2", "contents": "gamma"},
        ],
    )
    return {
        "a": {"path": str(tmp_path / "a.jsonl")},
        "b": {"path": str(tmp_path / "b.jsonl")},
    }


def test_resolve_by_name(manifest):
    docs = resolve_corpus_docs("a", manifest)
    assert [d["id"] for d in docs] == ["a1"]


def test_resolve_all_unions_disjoint_corpora(manifest):
    docs = resolve_corpus_docs("all", manifest)
    assert [d["id"] for d in docs] == ["a1", "b1", "b2"]


def test_resolve_comma_list_preserves_order(manifest):
    docs = resolve_corpus_docs("b,a", manifest)
    assert [d["id"] for d in docs] == ["b1", "b2", "a1"]


def test_resolve_same_corpus_repeated_dedupes(manifest):
    # A registered corpus loaded twice collapses its own repeat (no collision).
    docs = resolve_corpus_docs("a,a", manifest)
    assert [d["id"] for d in docs] == ["a1"]


def test_resolve_cross_corpus_id_collision_raises(tmp_path):
    # Two *different* corpora sharing an id must fail loud, not silently drop.
    _write_corpus(tmp_path / "x.jsonl", [{"id": "5", "title": "X", "contents": "x"}])
    _write_corpus(tmp_path / "y.jsonl", [{"id": "5", "title": "Y", "contents": "y"}])
    colliding = {
        "x": {"path": str(tmp_path / "x.jsonl")},
        "y": {"path": str(tmp_path / "y.jsonl")},
    }
    with pytest.raises(ValueError, match="id collision"):
        resolve_corpus_docs("all", colliding)


def test_resolve_path_backcompat(tmp_path, manifest):
    p = tmp_path / "direct.jsonl"
    _write_corpus(p, [{"id": "z", "title": "Z", "contents": "zeta"}])
    docs = resolve_corpus_docs(str(p), manifest)
    assert [d["id"] for d in docs] == ["z"]


def test_resolve_unknown_spec_raises(manifest):
    with pytest.raises(ValueError, match="Unknown corpus spec"):
        resolve_corpus_docs("nope", manifest)


def test_load_manifest_missing_file_returns_empty(tmp_path):
    assert load_manifest(str(tmp_path / "absent.json")) == {}


def test_union_docs_feed_from_docs_retriever(manifest):
    docs = resolve_corpus_docs("all", manifest)
    retriever = TfidfRetriever.from_docs(docs)
    rows = retriever.retrieve(["gamma"], topk=5)
    assert rows[0][0]["document"]["id"] == "b2"


def test_resolve_all_skips_missing_files_and_loads_present(manifest, caplog):
    manifest = dict(manifest)
    manifest["missing"] = {"path": "/nonexistent/path/does-not-exist.jsonl"}
    with caplog.at_level("WARNING"):
        docs = resolve_corpus_docs("all", manifest)
    assert [d["id"] for d in docs] == ["a1", "b1", "b2"]
    assert "missing" in caplog.text
    assert "/nonexistent/path/does-not-exist.jsonl" in caplog.text


def test_resolve_missing_named_corpus_raises(manifest):
    manifest = dict(manifest)
    manifest["missing"] = {"path": "/nonexistent/path/does-not-exist.jsonl"}
    with pytest.raises(ValueError):
        resolve_corpus_docs("missing", manifest)


def test_resolve_empty_spec_raises(manifest):
    with pytest.raises(ValueError):
        resolve_corpus_docs("", manifest)
    with pytest.raises(ValueError):
        resolve_corpus_docs(None, manifest)


def test_manifest_source_label_stamped_on_docs(tmp_path):
    _write_corpus(tmp_path / "a.jsonl", [{"id": "a1", "title": "A1", "contents": "x"}])
    manifest = {"a": {"path": str(tmp_path / "a.jsonl"), "source": "Team Wiki"}}
    docs = resolve_corpus_docs("a", manifest)
    assert docs[0]["metadata"]["source"] == "Team Wiki"


def test_per_doc_source_overrides_corpus_default(tmp_path):
    _write_corpus(
        tmp_path / "a.jsonl",
        [
            {"id": "a1", "title": "A1", "contents": "x", "metadata": {"source": "PDF"}},
            {"id": "a2", "title": "A2", "contents": "y"},
        ],
    )
    manifest = {"a": {"path": str(tmp_path / "a.jsonl"), "source": "Team Wiki"}}
    docs = resolve_corpus_docs("a", manifest)
    # Explicit per-doc source wins; the other doc gets the corpus default.
    assert docs[0]["metadata"]["source"] == "PDF"
    assert docs[1]["metadata"]["source"] == "Team Wiki"


def test_no_source_when_manifest_lacks_one(manifest):
    # Manifest entries without a "source" leave docs unlabeled, so the web
    # backend can fall back to its provider label.
    docs = resolve_corpus_docs("a", manifest)
    assert "source" not in (docs[0].get("metadata") or {})


# --- register_corpus -------------------------------------------------------
#
# The converter writes data/corpus_<dataset>.jsonl but the manifest is what
# makes `--corpus <name>` resolve. These pin the write half of that contract.


def test_register_corpus_creates_a_manifest_that_does_not_exist_yet(tmp_path):
    """A fresh checkout has no data/corpora.json; registering must not require one."""
    path = tmp_path / "corpora.json"

    register_corpus(
        "nfcorpus",
        corpus_path="data/corpus_nfcorpus.jsonl",
        doc_count=3633,
        manifest_path=str(path),
    )

    assert load_manifest(str(path)) == {
        "nfcorpus": {
            "path": "data/corpus_nfcorpus.jsonl",
            "docs": 3633,
            "domain": None,
            "source": "nfcorpus",
        }
    }


def test_register_corpus_keeps_the_entries_already_in_the_manifest(tmp_path):
    """Registering nfcorpus must not cost the user their demo/scifact entries."""
    path = tmp_path / "corpora.json"
    path.write_text(json.dumps({"demo": {"path": "data/corpus.jsonl", "docs": 20}}))

    register_corpus(
        "scifact", corpus_path="s.jsonl", doc_count=5183, manifest_path=str(path)
    )

    manifest = load_manifest(str(path))
    assert manifest["demo"] == {"path": "data/corpus.jsonl", "docs": 20}
    assert manifest["scifact"]["docs"] == 5183


def test_re_registering_updates_the_entry_rather_than_duplicating_it(tmp_path):
    """Re-running the converter (e.g. without --limit) must correct the count."""
    path = tmp_path / "corpora.json"

    register_corpus(
        "nfcorpus", corpus_path="c.jsonl", doc_count=100, manifest_path=str(path)
    )
    register_corpus(
        "nfcorpus", corpus_path="c.jsonl", doc_count=3633, manifest_path=str(path)
    )

    manifest = load_manifest(str(path))
    assert list(manifest) == ["nfcorpus"]
    assert manifest["nfcorpus"]["docs"] == 3633


def test_a_registered_corpus_resolves_by_name(tmp_path):
    """The point of registering: `--corpus <name>` must then load the documents.

    This is the round-trip the two halves exist for -- a manifest that writes
    fields resolve_corpus_docs does not read would pass every test above.
    """
    corpus = tmp_path / "nf.jsonl"
    _write_corpus(corpus, [{"id": "n1", "title": "N1", "contents": "nutrition"}])
    path = tmp_path / "corpora.json"

    register_corpus(
        "nfcorpus",
        corpus_path=str(corpus),
        doc_count=1,
        domain="medical/nutrition",
        source="NFCorpus",
        manifest_path=str(path),
    )

    docs = resolve_corpus_docs("nfcorpus", manifest=load_manifest(str(path)))
    assert [d["id"] for d in docs] == ["n1"]
    # The manifest's source label is what reaches the citation card.
    assert docs[0]["metadata"]["source"] == "NFCorpus"


def test_register_corpus_reports_a_manifest_it_cannot_parse(tmp_path):
    """A hand-edited manifest with a syntax error must not be silently replaced."""
    path = tmp_path / "corpora.json"
    path.write_text("{not json")

    with pytest.raises(ValueError, match="could not be parsed"):
        register_corpus(
            "nfcorpus", corpus_path="c.jsonl", doc_count=1, manifest_path=str(path)
        )

    # The user's file is left exactly as it was, for them to fix.
    assert path.read_text() == "{not json"


def test_unregister_corpus_removes_only_that_entry(tmp_path):
    manifest_path = str(tmp_path / "corpora.json")
    register_corpus(
        "a", corpus_path="a.jsonl", doc_count=1, manifest_path=manifest_path
    )
    register_corpus(
        "b", corpus_path="b.jsonl", doc_count=2, manifest_path=manifest_path
    )

    assert unregister_corpus("a", manifest_path=manifest_path)

    assert set(load_manifest(manifest_path)) == {"b"}
    assert not unregister_corpus("a", manifest_path=manifest_path)


def test_unregister_corpus_without_a_manifest_is_a_no_op(tmp_path):
    assert not unregister_corpus("a", manifest_path=str(tmp_path / "none.json"))


def test_unregister_corpus_reports_a_manifest_it_cannot_parse(tmp_path):
    manifest_path = tmp_path / "corpora.json"
    manifest_path.write_text("{not json")

    with pytest.raises(ValueError, match="could not be parsed"):
        unregister_corpus("a", manifest_path=str(manifest_path))
    assert manifest_path.read_text() == "{not json"
