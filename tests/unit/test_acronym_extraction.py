"""Acronym pairs a corpus defines about itself.

A static table assumes a domain: scifact uses `IR` for ionizing radiation,
this repository uses it for information retrieval. The corpus already carries
the answer in the ordinary gloss `ionizing radiation (IR)`. See #601.
"""

import json

import pytest

from src.internal.retrieval.acronym_extraction import (
    extract_acronyms,
    extract_acronyms_from_corpus,
)


def test_a_glossed_acronym_is_extracted():
    text = "increased sensitivity to ionizing radiation (IR) was observed"

    assert extract_acronyms(text) == {"IR": "ionizing radiation"}


def test_the_initials_must_match():
    """Without this check the pattern matches any parenthesised capital."""
    assert extract_acronyms("some random words (XYZ) appear here") == {}


def test_a_citation_is_not_an_acronym():
    """The shape that makes the naive pattern useless on real prose."""
    assert extract_acronyms("as shown by Pettifor (BBC) in 2012") == {}


def test_multi_word_expansions_are_bounded_to_the_matching_initials():
    text = "we treated acute myeloid leukemia (AML) in the cohort"

    assert extract_acronyms(text) == {"AML": "acute myeloid leukemia"}


def test_the_first_definition_wins():
    text = "ionizing radiation (IR) ... information retrieval (IR)"

    assert extract_acronyms(text) == {"IR": "ionizing radiation"}


def test_a_corpus_yields_its_own_vocabulary(tmp_path):
    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text(
        "\n".join(
            json.dumps(r)
            for r in [
                {"title": "t1", "text": "hepatitis c virus (HCV) prevalence"},
                {"title": "t2", "text": "receptor tyrosine kinase (RTK) signalling"},
            ]
        )
    )

    pairs = extract_acronyms_from_corpus(corpus)

    assert pairs == {"HCV": "hepatitis c virus", "RTK": "receptor tyrosine kinase"}


@pytest.mark.parametrize("bad", ["/nonexistent/corpus.jsonl", None])
def test_an_unusable_corpus_yields_nothing_rather_than_raising(bad):
    assert extract_acronyms_from_corpus(bad) == {}


def test_malformed_lines_are_skipped(tmp_path):
    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text('not json\n{"text": "hepatitis c virus (HCV) here"}\n')

    assert extract_acronyms_from_corpus(corpus) == {"HCV": "hepatitis c virus"}


def test_the_repo_corpus_schema_is_read(tmp_path):
    """corpus.jsonl is {id, title, contents, metadata}, not {title, text}.

    Reading only `text` sees the title and misses every document body -- which
    is what this extractor did before the schema was checked against a real
    corpus file.
    """
    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text(
        json.dumps(
            {
                "id": "doc_001",
                "title": "A study",
                "contents": "we measured hepatitis c virus (HCV) prevalence",
                "metadata": {"acl": ["public"]},
            }
        )
    )

    assert extract_acronyms_from_corpus(corpus) == {"HCV": "hepatitis c virus"}
