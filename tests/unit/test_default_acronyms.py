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
