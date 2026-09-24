import re

import pytest

from examples.measure_memory_strategies import (
    Strategy,
    aggregate,
    make_dataset,
    run_strategy,
    score,
)
from src.internal.memory.working import SUMMARY_PREFIX


def _words(text: str) -> int:
    return len(text.split())


def test_dataset_is_deterministic():
    assert make_dataset(3, seed=7) == make_dataset(3, seed=7)
    assert make_dataset(3, seed=7) != make_dataset(3, seed=8)


def test_facts_sit_at_their_planned_indices():
    conv = make_dataset(1, seed=1)[0]
    assert len(conv.facts) == 4
    for fact in conv.facts:
        role, content = conv.messages[fact.index]
        assert role == "user" and fact.value in content


def test_values_are_unique_and_absent_from_filler():
    dataset = make_dataset(12, seed=0)
    values = [f.value for conv in dataset for f in conv.facts]
    assert len(values) == len(set(values))
    for conv in dataset:
        fact_turns = {f.index for f in conv.facts} | {f.index + 1 for f in conv.facts}
        filler = " ".join(
            content
            for i, (_, content) in enumerate(conv.messages)
            if i not in fact_turns
        )
        for value in values:
            assert value.lower() not in filler.lower()


def test_transcript_fits_the_local_context():
    for conv in make_dataset(12, seed=0):
        assert len(conv.messages) == 80
        assert all(_words(c) <= 60 for _, c in conv.messages)
        # ~1.3 tokens per word plus chat-template overhead stays under 4,096.
        assert sum(_words(c) for _, c in conv.messages) <= 2600


@pytest.mark.parametrize(
    "answer,value,ok",
    [
        ("Your budget is $420.", "$420", True),
        ("your budget: 420 dollars", "$420", True),
        ("It's LH452", "LH452", True),
        ("it is lh 452", "LH452", True),
        ("Your dog is called Biscuit!", "Biscuit", True),
        ("You told me your budget earlier.", "$420", False),
        ("I don't know your flight number.", "LH452", False),
        ("$4200", "$420", False),
    ],
)
def test_score(answer, value, ok):
    assert score(answer, value) is ok


class _Summarizer:
    def __init__(self):
        self.calls = 0

    def complete(self, prompt, **kwargs):
        self.calls += 1
        return "SUMMARY-MARKER"


class _Answerer:
    def __init__(self):
        self.sent: list[list[dict]] = []

    def __call__(self, messages):
        self.sent.append(messages)
        return "no idea", 10, 0.0


def _small():
    return make_dataset(1, seed=3, pairs=8, fact_pairs=(0, 3, 5, 7))[0]


@pytest.mark.asyncio
async def test_summary_strategy_sends_summary():
    conv = _small()
    answer = _Answerer()
    await run_strategy(conv, Strategy("summary", 4), answer, _Summarizer())
    assert all(m[0]["role"] == "system" for m in answer.sent)
    assert all(m[0]["content"].startswith(SUMMARY_PREFIX) for m in answer.sent)

    answer = _Answerer()
    await run_strategy(conv, Strategy("window", 4), answer, _Summarizer())
    assert not any(m[0]["role"] == "system" for m in answer.sent)


@pytest.mark.asyncio
async def test_window_strategy_never_calls_the_summarizer():
    summarizer = _Summarizer()
    await run_strategy(_small(), Strategy("window", 4), _Answerer(), summarizer)
    assert summarizer.calls == 0


class _Marker:
    def __init__(self, marker):
        self.marker = marker

    def complete(self, prompt, **kwargs):
        return self.marker


class _Echo:
    """A perfect summarizer: returns everything it was asked to summarize."""

    def complete(self, prompt, **kwargs):
        return prompt[-1]["content"]


@pytest.mark.asyncio
async def test_runs_are_isolated():
    conv = _small()
    await run_strategy(conv, Strategy("summary", 4), _Answerer(), _Marker("RUN-ONE"))
    answer = _Answerer()
    # The echo summarizer copies its prompt -- including any prior summary --
    # into the new one, so a leaked first-run summary would surface here.
    await run_strategy(conv, Strategy("summary", 4), answer, _Echo())
    sent = " ".join(m["content"] for msgs in answer.sent for m in msgs)
    assert "Prior summary" in sent and "RUN-ONE" not in sent


@pytest.mark.asyncio
async def test_summary_prompt_is_summary_then_tail_then_probe():
    conv = _small()
    answer = _Answerer()
    await run_strategy(conv, Strategy("summary", 4), answer, _Marker("S"))
    facts = {f.index: f for f in conv.facts}
    for k, msgs in enumerate(answer.sent):
        question = facts[conv.probe_order[k]].question
        assert msgs[0]["role"] == "system"
        assert len(msgs) == 1 + 4 + 1  # summary, 4-message tail, probe
        assert msgs[-1] == {"role": "user", "content": question}
        assert sum(m["content"] == question for m in msgs) == 1


@pytest.mark.asyncio
async def test_lag_casualties_are_flagged_and_the_rest_reach_the_summary():
    # A fact that leaves the window on the probe turn itself is in neither the
    # tail nor the summary (the summary is built from earlier turns). A perfect
    # summarizer keeps every other dropped fact.
    conv = _small()
    results = await run_strategy(conv, Strategy("summary", 6), _Answerer(), _Echo())
    dropped = [r for r in results if not r.in_window]
    assert dropped
    for r in dropped:
        assert r.in_summary is (not r.dropped_this_turn)


@pytest.mark.asyncio
async def test_summarizer_stats_count_attempts_and_advances():
    class _Failing:
        def complete(self, prompt, **kwargs):
            raise TimeoutError("slow")

    stats = {}
    await run_strategy(
        _small(), Strategy("summary", 4), _Answerer(), _Failing(), stats=stats
    )
    assert stats["summarizer_calls"] > 0 and stats["summarizer_advanced"] == 0

    stats = {}
    await run_strategy(
        _small(), Strategy("summary", 4), _Answerer(), _Marker("S"), stats=stats
    )
    assert stats["summarizer_advanced"] == stats["summarizer_calls"] > 0


@pytest.mark.asyncio
async def test_rows_are_written_as_each_run_finishes(tmp_path):
    from examples.measure_memory_strategies import append_rows

    path = tmp_path / "rows.jsonl"
    results = await run_strategy(_small(), Strategy("window", 4), _Answerer(), None)
    append_rows(path, results)
    append_rows(path, results)
    assert len(path.read_text().splitlines()) == 2 * len(results)


@pytest.mark.asyncio
async def test_in_window_is_computed_at_probe_time():
    conv = _small()  # 16 transcript messages, facts at messages 0, 6, 10, 14
    answer = _Answerer()
    results = await run_strategy(conv, Strategy("window", 6), answer, None)
    by_index = {r.fact_index: r for r in results}
    for k, fact_index in enumerate(conv.probe_order):
        before = 16 + 2 * k
        assert by_index[fact_index].in_window is (fact_index >= before - 6)
        # The probe really was sent with that window: the fact's own message is
        # present in the sent history exactly when it counts as in-window.
        sent = [m["content"] for m in answer.sent[k]]
        stated = conv.messages[fact_index][1]
        assert (stated in sent) is by_index[fact_index].in_window
    assert {r.in_window for r in results} == {True, False}


@pytest.mark.asyncio
async def test_full_strategy_sees_every_message():
    conv = _small()
    answer = _Answerer()
    results = await run_strategy(conv, Strategy("full", None), answer, None)
    assert all(r.in_window for r in results)
    first = answer.sent[0]
    assert len(first) == 16 + 1


def test_aggregate_splits_recall_by_window():
    from examples.measure_memory_strategies import ProbeResult

    rows = [
        ProbeResult("window-6", 0, 0, True, True, False, 100, 0.1),
        ProbeResult("window-6", 0, 1, False, False, False, 90, 0.1),
        ProbeResult("window-6", 0, 2, True, False, False, 90, 0.1),
    ]
    report = aggregate(rows, {"window-6": {"summarizer_calls": 0}})
    s = report["window-6"]
    assert s["recall"] == pytest.approx(2 / 3, abs=1e-3)
    assert s["recall_in_window"] == 1.0
    assert s["recall_dropped"] == 0.5
    assert s["probes"] == 3
    assert s["summary_retention"] is None  # no summary to retain anything


def test_aggregate_excludes_lag_casualties_from_summary_metrics():
    from examples.measure_memory_strategies import ProbeResult

    rows = [
        ProbeResult("summary-6", 0, 0, True, False, True, 100, 0.1),
        ProbeResult(
            "summary-6", 0, 1, False, False, False, 90, 0.1, dropped_this_turn=True
        ),
    ]
    s = aggregate(rows, {})["summary-6"]
    assert s["lag_casualties"] == 1
    assert s["summary_retention"] == 1.0
    assert s["recall_dropped_excl_lag"] == 1.0
    assert s["recall_dropped"] == 0.5


def test_no_fact_value_is_a_substring_of_another():
    values = [f.value for conv in make_dataset(12, seed=0) for f in conv.facts]
    for a in values:
        for b in values:
            if a != b:
                assert not re.search(re.escape(a.lower()), b.lower())
