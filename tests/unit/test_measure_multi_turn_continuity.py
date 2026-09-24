import json
from pathlib import Path

import pytest

from examples.measure_multi_turn_continuity import (
    Conversation,
    Turn,
    build_query,
    load_conversations,
)


def _write(tmp_path: Path, conversations: list[dict]) -> Path:
    path = tmp_path / "conversations.jsonl"
    path.write_text("\n".join(json.dumps(c) for c in conversations) + "\n")
    return path


def _search(text, relation="opening", kind=None, gold=None, ids=("d1",)):
    turn = {
        "text": text,
        "route": "search",
        "relation": relation,
        "gold_rewrite": gold or text,
        "corpus": "demo",
        "relevant_doc_ids": list(ids),
    }
    if kind:
        turn["kind"] = kind
    return turn


def _tool(text, relation="opening", kind=None, gold=None):
    turn = {
        "text": text,
        "route": "tool",
        "relation": relation,
        "gold_rewrite": gold or text,
    }
    if kind:
        turn["kind"] = kind
    return turn


def test_loads_a_valid_conversation(tmp_path):
    path = _write(
        tmp_path,
        [
            {
                "id": "c1",
                "turns": [
                    _tool("weather in Tokyo"),
                    _tool("and in Paris?", "follow_up", "ellipsis", "weather in Paris"),
                ],
            }
        ],
    )
    [conv] = load_conversations(path)
    assert conv == Conversation(
        "c1",
        (
            Turn("weather in Tokyo", "tool", "opening", "weather in Tokyo"),
            Turn(
                "and in Paris?",
                "tool",
                "follow_up",
                "weather in Paris",
                kind="ellipsis",
            ),
        ),
    )


@pytest.mark.parametrize(
    ("turns", "message"),
    [
        (
            [
                _tool("and in Paris?", "follow_up", "ellipsis", "weather in Paris"),
                _tool("x"),
            ],
            "first turn must be an opening",
        ),
        (
            [_tool("weather in Tokyo"), _tool("weather in Oslo", "tangent")],
            "unknown relation",
        ),
        (
            [
                _tool("weather in Tokyo"),
                {**_search("faiss", "topic_switch"), "relevant_doc_ids": []},
            ],
            "relevant_doc_ids",
        ),
        (
            [
                _tool("weather in Tokyo"),
                {**_tool("weather in Oslo", "topic_switch"), "corpus": "demo"},
            ],
            "relevant_doc_ids",
        ),
        (
            [
                _tool("weather in Tokyo"),
                _tool("and Paris?", "follow_up", None, "weather in Paris"),
            ],
            "kind",
        ),
        (
            [
                _tool("weather in Tokyo"),
                _tool("and Paris?", "follow_up", "sarcasm", "weather in Paris"),
            ],
            "unknown kind",
        ),
        (
            [
                _tool("weather in Tokyo"),
                _tool("weather in Oslo", "topic_switch", gold="Oslo weather"),
            ],
            "gold_rewrite must equal",
        ),
        (
            [
                _tool("weather in Tokyo"),
                {**_tool("weather in Oslo", "topic_switch"), "note": "x"},
            ],
            "unknown keys",
        ),
        ([_tool("weather in Tokyo")], "at least 2 turns"),
        (
            [_tool("weather in Tokyo"), _tool("weather in Oslo", "opening")],
            "only the first turn",
        ),
    ],
)
def test_rejects_invalid_conversations(tmp_path, turns, message):
    path = _write(tmp_path, [{"id": "c1", "turns": turns}])
    with pytest.raises(ValueError, match=message):
        load_conversations(path)


def test_rejects_duplicate_ids(tmp_path):
    conv = {
        "id": "c1",
        "turns": [_tool("weather in Tokyo"), _tool("weather in Oslo", "topic_switch")],
    }
    with pytest.raises(ValueError, match="duplicate id"):
        load_conversations(_write(tmp_path, [conv, conv]))


def _conv(*turns: Turn) -> Conversation:
    return Conversation("c", turns)


TOKYO = Turn("weather in Tokyo", "tool", "opening", "weather in Tokyo")
PARIS = Turn("and in Paris?", "tool", "follow_up", "weather in Paris", kind="ellipsis")
BERLIN = Turn("and Berlin?", "tool", "follow_up", "weather in Berlin", kind="ellipsis")
PRICE = Turn(
    "its price in 2020?",
    "tool",
    "follow_up",
    "Tesla stock price in 2020",
    kind="pronoun",
)
CUE_SWITCH = Turn("also, explain BM25", "chat", "topic_switch", "also, explain BM25")


def test_raw_and_gold_rewrite():
    conv = _conv(TOKYO, PARIS)
    assert build_query("raw", conv, 1) == "and in Paris?"
    assert build_query("gold_rewrite", conv, 1) == "weather in Paris"


def test_first_turn_is_unchanged_by_every_condition():
    conv = _conv(TOKYO, PARIS)
    for condition in ("raw", "regex", "concat", "gold_rewrite"):
        assert build_query(condition, conv, 0) == "weather in Tokyo"


def test_concat_uses_the_immediately_previous_user_turn():
    conv = _conv(TOKYO, PARIS, BERLIN)
    assert build_query("concat", conv, 2) == "and in Paris?\nand Berlin?"


def test_regex_prepends_the_most_recent_non_follow_up_turn():
    conv = _conv(TOKYO, PARIS, BERLIN)
    assert build_query("regex", conv, 2) == "weather in Tokyo\nand Berlin?"


def test_regex_leaves_an_uncued_follow_up_alone():
    conv = _conv(TOKYO, PRICE)
    assert build_query("regex", conv, 1) == "its price in 2020?"


def test_regex_false_positive_on_a_cue_word_switch():
    conv = _conv(TOKYO, CUE_SWITCH)
    assert build_query("regex", conv, 1) == "weather in Tokyo\nalso, explain BM25"


def test_unknown_condition_raises():
    with pytest.raises(ValueError, match="unknown condition"):
        build_query("oracle", _conv(TOKYO, PARIS), 1)
