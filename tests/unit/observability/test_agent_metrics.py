"""Agent lifecycle metrics must remain independent of transport and concurrency."""

import asyncio
import importlib

import pytest

from src.internal.observability import prometheus as prom


def snapshot(agent, outcome):
    labels = {"agent": agent, "outcome": outcome}
    return tuple(
        prom.REGISTRY.get_sample_value(name, labels) or 0
        for name in (
            "agentic_search_agent_runs_total",
            "agentic_search_agent_decision_rounds_count",
            "agentic_search_agent_decision_rounds_sum",
        )
    )


def delta(before, after):
    return tuple(b - a for a, b in zip(before, after))


def test_metrics_export_counts_rounds_and_bounds_labels():
    before = snapshot("search", "completed")
    prom.observe_agent_run("search", "completed", 3)
    assert delta(before, snapshot("search", "completed")) == (1, 1, 3)
    labels = {"outcome": "timeout"}
    name = "agentic_search_tool_attempts_total"
    before = prom.REGISTRY.get_sample_value(name, labels) or 0
    prom.observe_tool_attempt("timeout")
    assert prom.REGISTRY.get_sample_value(name, labels) == before + 1
    text = prom.render_latest()[0].decode()
    assert 'agentic_search_tool_attempts_total{outcome="timeout"}' in text
    assert 'agent="search",le="3.0",outcome="completed"' in text


@pytest.mark.parametrize(
    "agent,outcome,rounds",
    [
        ("arbitrary", "completed", 1),
        ("search", "other", 1),
        ("tool", "error", -1),
        ("tool", "error", 1.5),
        ("tool", "error", True),
    ],
)
def test_invalid_observation_does_not_create_series(agent, outcome, rounds):
    before = prom.render_latest()[0]
    with pytest.raises(ValueError):
        prom.observe_agent_run(agent, outcome, rounds)
    assert prom.render_latest()[0] == before


@pytest.mark.parametrize(
    "outcome,rounds", [("completed", 3), ("error", 0), ("error", 1), ("cancelled", 1)]
)
async def test_lifecycle_records_once_and_propagates(outcome, rounds):
    metrics = importlib.import_module("src.internal.observability.agent_metrics")
    before = snapshot("search", outcome)

    @metrics.track_agent_run("search")
    async def run():
        for _ in range(rounds):
            metrics.record_decision_round()
        if outcome == "error":
            raise RuntimeError("failed")
        if outcome == "cancelled":
            raise asyncio.CancelledError()
        return "answer"

    if outcome == "completed":
        assert await run() == "answer"
    else:
        with pytest.raises(
            RuntimeError if outcome == "error" else asyncio.CancelledError
        ):
            await run()
    assert delta(before, snapshot("search", outcome)) == (1, 1, rounds)


async def test_concurrent_nested_and_child_tasks_do_not_mix_rounds():
    metrics = importlib.import_module("src.internal.observability.agent_metrics")
    before = snapshot("search", "completed")
    child_before = snapshot("tool", "completed")
    metrics.record_decision_round()  # outside any run

    @metrics.track_agent_run("tool")
    async def nested():
        metrics.record_decision_round()

    @metrics.track_agent_run("search")
    async def run(rounds):
        for _ in range(rounds):
            metrics.record_decision_round()
            await asyncio.sleep(0)
        await nested()

        async def unrelated_child():
            metrics.record_decision_round()

        await asyncio.create_task(unrelated_child())

    await asyncio.gather(run(1), run(3))
    assert delta(before, snapshot("search", "completed")) == (2, 2, 4)
    assert delta(child_before, snapshot("tool", "completed")) == (2, 2, 2)
    metrics.record_decision_round()  # reset after the completed run
    assert delta(before, snapshot("search", "completed")) == (2, 2, 4)
