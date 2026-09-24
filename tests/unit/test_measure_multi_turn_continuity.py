import json
from pathlib import Path

import pytest

from examples.measure_multi_turn_continuity import (
    Conversation,
    Turn,
    build_query,
    evaluate,
    load_conversations,
    score_retrieval,
    summarize,
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


def test_score_retrieval_hit_and_reciprocal_rank():
    assert score_retrieval(["a", "b", "c"], ("c",)) == (True, pytest.approx(1 / 3))
    assert score_retrieval(["a", "b", "c", "d", "e", "f"], ("f",)) == (
        False,
        pytest.approx(1 / 6),
    )


def test_score_retrieval_handles_short_and_empty_rankings():
    assert score_retrieval([], ("a",)) == (False, 0.0)
    assert score_retrieval(["x"], ("a",)) == (False, 0.0)


def test_score_retrieval_ignores_ranks_past_ten():
    ranked = [f"x{i}" for i in range(10)] + ["a"]
    assert score_retrieval(ranked, ("a",)) == (False, 0.0)


FAISS = Turn(
    "what is FAISS",
    "search",
    "opening",
    "what is FAISS",
    corpus="demo",
    relevant_doc_ids=("d1",),
)
FAISS_TYPES = Turn(
    "which variants does it offer",
    "search",
    "follow_up",
    "FAISS index types",
    kind="pronoun",
    corpus="demo",
    relevant_doc_ids=("d2",),
)


def test_evaluate_rows_route_and_retrieval():
    conv = Conversation("c", (FAISS, FAISS_TYPES))
    routes = {"what is FAISS": "search", "FAISS index types": "search"}

    def router(query):
        return routes.get(query, "clarify")

    def retriever(corpus, query):
        return ["d2"] if "index types" in query else ["d1"]

    rows = evaluate([conv], {"rules": router}, {"tfidf": retriever})

    assert len(rows) == 2 * 4
    raw = next(r for r in rows if r["turn_index"] == 1 and r["condition"] == "raw")
    assert raw["route"] == {"rules": "clarify"}
    assert raw["retrieval"]["tfidf"] == {"hit5": False, "rr10": 0.0}
    assert raw["carried"] is False
    gold = next(
        r for r in rows if r["turn_index"] == 1 and r["condition"] == "gold_rewrite"
    )
    assert gold["route"] == {"rules": "search"}
    assert gold["retrieval"]["tfidf"] == {"hit5": True, "rr10": 1.0}
    concat = next(
        r for r in rows if r["turn_index"] == 1 and r["condition"] == "concat"
    )
    assert concat["carried"] is True


def test_evaluate_skips_retrieval_on_non_search_turns():
    conv = Conversation("c", (TOKYO, PARIS))
    calls = []

    def retriever(corpus, query):
        calls.append(query)
        return []

    rows = evaluate([conv], {"rules": lambda query: "tool"}, {"tfidf": retriever})
    assert calls == []
    assert all(r["retrieval"] == {} for r in rows)


def _row(
    conv,
    index,
    condition,
    *,
    relation="follow_up",
    kind="pronoun",
    predicted="search",
    carried=False,
    hit=None,
):
    return {
        "conversation_id": conv,
        "turn_index": index,
        "relation": relation,
        "kind": kind if relation == "follow_up" else None,
        "gold_route": "search",
        "condition": condition,
        "query": "q",
        "carried": carried,
        "route": {"rules": predicted},
        "retrieval": {}
        if hit is None
        else {"tfidf": {"hit5": hit, "rr10": 1.0 if hit else 0.0}},
    }


def _four_conditions(conv, index, gold_pred, other_pred="clarify", **kwargs):
    rows = []
    for condition in ("raw", "regex", "concat", "gold_rewrite"):
        predicted = gold_pred if condition == "gold_rewrite" else other_pred
        rows.append(_row(conv, index, condition, predicted=predicted, **kwargs))
    return rows


def test_route_accuracy_counts_clarify_as_a_miss():
    rows = _four_conditions("c1", 1, "search")
    summary = summarize(rows, ["rules"], [], resamples=50, seed=0)
    acc = summary["follow_up"]["route_acc:rules"]
    assert acc["raw"]["mean"] == 0.0
    assert acc["gold_rewrite"]["mean"] == 1.0
    assert acc["gold_rewrite"]["delta_vs_raw"]["point"] == 1.0
    assert acc["raw"]["delta_vs_raw"] is None


def test_bootstrap_resamples_conversations_not_turns():
    # One conversation, turns alternating correct/incorrect under gold_rewrite.
    # Resampling the single conversation always reproduces it, so the CI is a
    # point; resampling turns would spread it.
    rows = []
    for index in range(1, 9):
        rows += _four_conditions("c1", index, "search" if index % 2 else "chat")
    summary = summarize(rows, ["rules"], [], resamples=200, seed=0)
    delta = summary["follow_up"]["route_acc:rules"]["gold_rewrite"]["delta_vs_raw"]
    assert delta == {"point": 0.5, "low": 0.5, "high": 0.5}


def test_carry_over_is_scored_on_topic_switches_only():
    conditions = ("raw", "regex", "concat", "gold_rewrite")
    rows = [
        _row("c1", 1, c, relation="topic_switch", carried=(c == "concat"))
        for c in conditions
    ]
    rows += [_row("c1", 2, c, carried=True) for c in conditions]
    summary = summarize(rows, ["rules"], [], resamples=50, seed=0)
    assert summary["topic_switch"]["carry_over"]["concat"]["mean"] == 1.0
    assert summary["topic_switch"]["carry_over"]["raw"]["mean"] == 0.0
    assert "carry_over" not in summary["follow_up"]
    assert summary["all"]["carry_over"]["concat"]["n"] == 1


def test_retrieval_metrics_skip_non_search_turns():
    rows = _four_conditions("c1", 1, "search", hit=True) + _four_conditions(
        "c2", 1, "tool"
    )
    summary = summarize(rows, ["rules"], ["tfidf"], resamples=50, seed=0)
    assert summary["all"]["hit5:tfidf"]["raw"]["n"] == 1
    assert summary["all"]["route_acc:rules"]["raw"]["n"] == 2
