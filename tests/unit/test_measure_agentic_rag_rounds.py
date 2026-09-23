from examples.measure_agentic_rag_rounds import aggregate, summarize_trace


def _ev(
    component,
    action,
    status="completed",
    ts="2026-01-01T00:00:00.000Z",
    duration_ms=0,
    **details,
):
    return {
        "component": component,
        "action": action,
        "status": status,
        "timestamp": ts,
        "duration_ms": duration_ms,
        "details": details,
    }


def _search(round_, docs, ts="2026-01-01T00:00:01.000Z"):
    return _ev(
        "search_tool",
        "vector_db_search",
        ts=ts,
        search_round=round_,
        document_count=docs,
    )


def _judge(round_, sufficient, status="decided", **extra):
    return _ev(
        "evidence_judge",
        "sufficiency_check",
        status=status,
        search_round=round_,
        sufficient=sufficient,
        **extra,
    )


def _synth(ts="2026-01-01T00:00:05.000Z", duration_ms=1000):
    return _ev("answer_generator", "synthesize", ts=ts, duration_ms=duration_ms)


def test_sufficient_at_round_one():
    run = summarize_trace([_search(1, 4), _judge(1, True), _synth()], rounds_used=1)
    assert run.stop_reason == "sufficient"
    assert run.rounds_used == 1
    assert run.docs_added_after_round_1 == 0


def test_judge_failed_open_is_not_sufficient():
    run = summarize_trace(
        [_search(1, 4), _judge(1, True, status="failed", fallback=True), _synth()],
        rounds_used=1,
    )
    assert run.stop_reason == "judge_failed_open"


def test_legacy_trace_marks_failed_open_by_status_only():
    run = summarize_trace(
        [_search(1, 4), _judge(1, True, status="failed"), _synth()], rounds_used=1
    )
    assert run.stop_reason == "judge_failed_open"


def test_insufficient_then_no_followups():
    run = summarize_trace([_search(1, 2), _judge(1, False), _synth()], rounds_used=1)
    assert run.stop_reason == "no_new_followups"


def test_max_rounds_and_docs_added():
    events = [
        _search(1, 2),
        _judge(1, False),
        _search(2, 3),
        _judge(2, False),
        _search(3, 5),
        _synth(),
    ]
    run = summarize_trace(events, rounds_used=3, max_rounds=3)
    assert run.stop_reason == "max_rounds"
    assert run.docs_after_round_1 == 2
    assert run.docs_final == 5
    assert run.docs_added_after_round_1 == 3


def test_no_queries_when_nothing_was_retrieved():
    run = summarize_trace([_synth()], rounds_used=1)
    assert run.stop_reason == "no_queries"


def test_post_round_1_ms_spans_round_1_end_to_synthesis_start():
    events = [
        _search(1, 2, ts="2026-01-01T00:00:01.000Z"),
        _judge(1, True),
        _synth(ts="2026-01-01T00:00:05.000Z", duration_ms=1500),
    ]
    run = summarize_trace(events, rounds_used=1)
    # synthesis started at 00:00:03.500, round 1 retrieval ended at 00:00:01.000
    assert run.post_round_1_ms == 2500


def test_aggregate_excludes_failed_open_from_one_round_share():
    runs = [
        summarize_trace([_search(1, 2), _judge(1, True), _synth()], rounds_used=1),
        summarize_trace(
            [_search(1, 2), _judge(1, True, status="failed"), _synth()], rounds_used=1
        ),
        summarize_trace(
            [
                _search(1, 2),
                _judge(1, False),
                _search(2, 2),
                _judge(2, False),
                _search(3, 2),
                _synth(),
            ],
            rounds_used=3,
        ),
    ]
    report = aggregate(runs)
    assert report["runs"] == 3
    assert report["stop_reasons"] == {
        "sufficient": 1,
        "judge_failed_open": 1,
        "max_rounds": 1,
    }
    assert report["rounds_used"] == {"1": 2, "3": 1}
    # 1 sufficient-at-round-1 out of 2 runs with a real verdict
    assert report["one_round_share"] == 0.5
    # the one multi-round run added nothing after round 1
    assert report["multi_round_runs"] == 1
    assert report["multi_round_zero_new_docs"] == 1
