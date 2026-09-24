import json
from pathlib import Path

import pytest

from examples.measure_multi_turn_continuity import (
    Conversation,
    Turn,
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
