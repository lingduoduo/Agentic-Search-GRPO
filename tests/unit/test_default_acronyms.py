"""QUERY_EXPANSION_ENABLED must expand something on its own.

Before this, enabling expansion with no ACRONYM_PATH was a silent no-op: the
mechanism worked, the table was empty, and nothing said so. See #599.
"""

import json
import logging

import pytest

from src.internal.retrieval.query_optimizer import (
    DEFAULT_ACRONYM_PATH,
    QueryOptimizer,
)


def test_the_default_table_loads_when_nothing_overrides_it():
    """The whole point: the flag alone now expands."""
    optimizer = QueryOptimizer(None)

    assert (
        optimizer.expand("RAG systems") == "RAG systems retrieval augmented generation"
    )


def test_a_user_file_overrides_the_default_rather_than_merging(tmp_path):
    """A caller who supplies a file gets exactly that file."""
    user_file = tmp_path / "acronyms.json"
    user_file.write_text(json.dumps({"ZZZ": "zulu zulu zulu"}))

    optimizer = QueryOptimizer(str(user_file))

    assert optimizer.expand("ZZZ here") == "ZZZ here zulu zulu zulu"
    # RAG is in the default and absent from the user file: it must NOT expand.
    assert optimizer.expand("RAG here") == "RAG here"


def test_an_empty_table_warns(tmp_path, caplog):
    empty = tmp_path / "empty.json"
    empty.write_text("{}")

    with caplog.at_level(logging.WARNING):
        QueryOptimizer(str(empty))

    assert any("expansion" in r.message.lower() for r in caplog.records)


def test_the_default_table_does_not_warn(caplog):
    with caplog.at_level(logging.WARNING):
        QueryOptimizer(None)

    assert not [r for r in caplog.records if "expansion" in r.message.lower()]


def test_the_bundled_file_is_well_formed():
    payload = json.loads(DEFAULT_ACRONYM_PATH.read_text(encoding="utf-8"))

    assert payload, "the default table must not be empty"
    for key, value in payload.items():
        assert key == key.upper(), f"{key} must be upper-case"
        assert isinstance(value, str) and value.strip(), f"{key} has no expansion"
        assert value.upper() != key, f"{key} expands to itself"


@pytest.mark.parametrize("acronym", ["RAG", "IR", "NDCG", "MMR"])
def test_core_vocabulary_is_present(acronym):
    """These are the system's own terms; losing them is a regression."""
    payload = json.loads(DEFAULT_ACRONYM_PATH.read_text(encoding="utf-8"))

    assert acronym in payload


def _corpus(tmp_path, *texts):
    path = tmp_path / "corpus.jsonl"
    path.write_text("\n".join(json.dumps({"title": "", "text": t}) for t in texts))
    return path


def test_a_corpus_derived_table_beats_the_bundled_one(tmp_path):
    """HCV is not in the bundle; RAG is. The corpus wins outright, not merged."""
    corpus = _corpus(tmp_path, "hepatitis c virus (HCV) prevalence rose")

    optimizer = QueryOptimizer(None, corpus_path=corpus)

    assert optimizer.expand("HCV rates") == "HCV rates hepatitis c virus"
    assert optimizer.expand("RAG systems") == "RAG systems"


def test_an_empty_extraction_falls_back_to_the_bundle(tmp_path):
    corpus = _corpus(tmp_path, "this text defines no acronyms at all")

    optimizer = QueryOptimizer(None, corpus_path=corpus)

    assert (
        optimizer.expand("RAG systems") == "RAG systems retrieval augmented generation"
    )


def test_an_explicit_acronym_path_still_overrides_the_corpus(tmp_path):
    corpus = _corpus(tmp_path, "hepatitis c virus (HCV) prevalence rose")
    user_file = tmp_path / "user.json"
    user_file.write_text(json.dumps({"ZZZ": "zulu zulu"}))

    optimizer = QueryOptimizer(str(user_file), corpus_path=corpus)

    assert optimizer.expand("ZZZ here") == "ZZZ here zulu zulu"
    assert optimizer.expand("HCV rates") == "HCV rates"


def test_the_demo_corpus_uses_the_one_acronym_it_defines():
    """Pins the real-world case: 20 documents define exactly one acronym.

    The corpus glosses "dense passage retrieval (DPR)" and never says what RAG
    means, so DPR expands and RAG does not. Any derived pair beats the bundled
    table -- a corpus that defines its own vocabulary is better evidence than a
    table written elsewhere.
    """
    optimizer = QueryOptimizer(None, corpus_path="data/corpus.jsonl")

    assert optimizer.expand("DPR index") == "DPR index dense passage retrieval"
    assert optimizer.expand("RAG systems") == "RAG systems"
