"""Unit tests for the Bamboogle evaluation module.

Tests run without any network access — dataset loading is mocked.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

pytest.importorskip("torch")
from src.model.post_training.eval.bamboogle import (
    contains_match,
    evaluate_bamboogle,
    exact_match,
    load_bamboogle,
    normalize_text,
    _extract_answer,
    _make_judge_fn,
    _to_loop_output,
)
from src.agents.core.base import AgentLoopOutput


# ---------------------------------------------------------------------------
# normalize / exact_match / contains_match
# ---------------------------------------------------------------------------


def test_normalize_strips_articles():
    assert normalize_text("The quick Brown Fox") == "quick brown fox"


def test_normalize_removes_punctuation():
    assert normalize_text("Hello, World!") == "hello world"


def test_exact_match_hit():
    assert exact_match("Albert Einstein", ["Albert Einstein", "Einstein"]) == 1.0


def test_exact_match_miss():
    assert exact_match("Newton", ["Albert Einstein"]) == 0.0


def test_exact_match_normalises():
    assert exact_match("the eiffel tower", ["Eiffel Tower"]) == 1.0


def test_contains_match_substring():
    assert contains_match("The answer is Paris, France.", ["Paris"]) == 1.0


def test_contains_match_miss():
    assert contains_match("Berlin is the answer.", ["Paris"]) == 0.0


def test_contains_match_multiple_gold():
    assert contains_match("Rome wasn't built in a day.", ["Athens", "Rome"]) == 1.0


# ---------------------------------------------------------------------------
# _extract_answer
# ---------------------------------------------------------------------------


def test_extract_answer_from_string():
    assert _extract_answer("hello") == "hello"


def test_extract_answer_from_dict():
    assert _extract_answer({"answer": "Paris"}) == "Paris"


def test_extract_answer_from_object():
    obj = MagicMock()
    obj.answer = "Berlin"
    assert _extract_answer(obj) == "Berlin"


def test_extract_answer_missing_returns_empty():
    assert _extract_answer({}) == ""


# ---------------------------------------------------------------------------
# _make_judge_fn
# ---------------------------------------------------------------------------


def test_make_judge_fn_match():
    judge = _make_judge_fn(["Paris", "Pairs"])
    assert judge("Paris is the capital", "_ignored_") == 1.0


def test_make_judge_fn_no_match():
    judge = _make_judge_fn(["London"])
    assert judge("Paris is the capital", "_ignored_") == 0.0


# ---------------------------------------------------------------------------
# _to_loop_output
# ---------------------------------------------------------------------------


def test_to_loop_output_stub_from_agent_result():
    obj = MagicMock()
    obj.answer = "some answer"
    del obj.metadata  # no metadata attribute
    out = _to_loop_output(obj)
    assert isinstance(out, AgentLoopOutput)
    assert out.final_answer == "some answer"
    assert out.prompt_ids == []


def test_to_loop_output_rich_path():
    real_output = AgentLoopOutput(
        prompt_ids=[1, 2],
        response_ids=[3, 4],
        response_mask=[1, 1],
        num_turns=2,
        final_answer="rich answer",
        metrics={"rounds_used": 1.0},
    )
    obj = MagicMock()
    obj.metadata = {"loop_output": real_output}
    out = _to_loop_output(obj)
    assert out is real_output
    assert out.metrics["rounds_used"] == 1.0


# ---------------------------------------------------------------------------
# load_bamboogle (mocked network)
# ---------------------------------------------------------------------------

_FAKE_JSONL = "\n".join(
    json.dumps(
        {
            "id": str(i),
            "question": f"Q{i}",
            "golden_answers": [f"A{i}"],
        }
    )
    for i in range(5)
)


@patch("requests.get")
def test_load_bamboogle_all(mock_get):
    mock_get.return_value.text = _FAKE_JSONL
    mock_get.return_value.raise_for_status = MagicMock()
    rows = load_bamboogle(cache_path=None)
    assert len(rows) == 5


@patch("requests.get")
def test_load_bamboogle_limit(mock_get):
    mock_get.return_value.text = _FAKE_JSONL
    mock_get.return_value.raise_for_status = MagicMock()
    rows = load_bamboogle(limit=3, cache_path=None)
    assert len(rows) == 3


@patch("requests.get")
def test_load_bamboogle_writes_cache(mock_get, tmp_path):
    mock_get.return_value.text = _FAKE_JSONL
    mock_get.return_value.raise_for_status = MagicMock()
    cache = tmp_path / "bamboogle_test.jsonl"
    load_bamboogle(cache_path=cache)
    assert cache.exists()
    assert len(cache.read_text().splitlines()) == 5


@patch("requests.get")
def test_load_bamboogle_reads_cache(mock_get, tmp_path):
    """Second call must not hit the network when cache exists."""
    mock_get.return_value.text = _FAKE_JSONL
    mock_get.return_value.raise_for_status = MagicMock()
    cache = tmp_path / "bamboogle_test.jsonl"
    load_bamboogle(cache_path=cache)
    mock_get.reset_mock()
    rows = load_bamboogle(cache_path=cache)
    mock_get.assert_not_called()
    assert len(rows) == 5


@patch("requests.get")
def test_load_bamboogle_cache_none_skips_disk(mock_get):
    """cache_path=None must never touch the filesystem."""
    mock_get.return_value.text = _FAKE_JSONL
    mock_get.return_value.raise_for_status = MagicMock()
    rows = load_bamboogle(cache_path=None)
    assert len(rows) == 5


# ---------------------------------------------------------------------------
# evaluate_bamboogle (fully mocked)
# ---------------------------------------------------------------------------

_FAKE_DATASET = [
    {
        "id": "1",
        "question": "Who invented the telephone?",
        "golden_answers": ["Alexander Graham Bell"],
    },
    {
        "id": "2",
        "question": "What is the capital of France?",
        "golden_answers": ["Paris"],
    },
]


class _PerfectAgent:
    """Returns the first gold answer verbatim."""

    def invoke(self, state: dict) -> MagicMock:
        q = state["messages"][0]["content"]
        answers = {
            "Who invented the telephone?": "Alexander Graham Bell",
            "What is the capital of France?": "Paris",
        }
        result = MagicMock()
        result.answer = answers.get(q, "")
        del result.metadata
        return result


class _WrongAgent:
    def invoke(self, state: dict) -> MagicMock:
        result = MagicMock()
        result.answer = "wrong answer"
        del result.metadata
        return result


@patch(
    "src.model.post_training.eval.bamboogle.load_bamboogle", return_value=_FAKE_DATASET
)
def test_evaluate_perfect_agent(mock_load, tmp_path):
    summary, rows = evaluate_bamboogle(
        _PerfectAgent(),
        limit=2,
        output_path=tmp_path / "out.jsonl",
        verbose=False,
    )
    assert summary.exact_match == pytest.approx(1.0)
    assert summary.contains_match == pytest.approx(1.0)
    assert summary.num_examples == 2
    assert summary.avg_reward is None


@patch(
    "src.model.post_training.eval.bamboogle.load_bamboogle", return_value=_FAKE_DATASET
)
def test_evaluate_wrong_agent(mock_load, tmp_path):
    summary, rows = evaluate_bamboogle(
        _WrongAgent(),
        limit=2,
        output_path=None,
        verbose=False,
    )
    assert summary.exact_match == pytest.approx(0.0)
    assert summary.contains_match == pytest.approx(0.0)


@patch(
    "src.model.post_training.eval.bamboogle.load_bamboogle", return_value=_FAKE_DATASET
)
def test_evaluate_writes_jsonl(mock_load, tmp_path):
    out = tmp_path / "sub" / "out.jsonl"
    evaluate_bamboogle(_PerfectAgent(), limit=2, output_path=out, verbose=False)
    lines = out.read_text().splitlines()
    assert len(lines) == 2
    row = json.loads(lines[0])
    assert "prediction" in row
    assert "exact_match" in row
    assert "reward_components" in row


@patch(
    "src.model.post_training.eval.bamboogle.load_bamboogle", return_value=_FAKE_DATASET
)
def test_evaluate_with_reward_fn(mock_load, tmp_path):
    from src.model.post_training.reward import SearchRewardFunction, SearchRewardConfig

    reward_fn = SearchRewardFunction(SearchRewardConfig.sparse_final_only())
    summary, rows = evaluate_bamboogle(
        _PerfectAgent(),
        reward_fn=reward_fn,
        limit=2,
        output_path=None,
        verbose=False,
    )
    assert summary.avg_reward is not None
    assert summary.avg_reward > 0.0  # perfect agent should score non-zero correctness
    for row in rows:
        assert row.reward_total is not None
        assert "total" in row.reward_components
        assert "correctness" in row.reward_components


@patch(
    "src.model.post_training.eval.bamboogle.load_bamboogle", return_value=_FAKE_DATASET
)
def test_evaluate_parallel_same_results(mock_load, tmp_path):
    """Concurrency > 1 must produce identical results to concurrency=1."""
    serial_summary, serial_rows = evaluate_bamboogle(
        _PerfectAgent(), limit=2, output_path=None, verbose=False, concurrency=1
    )
    parallel_summary, parallel_rows = evaluate_bamboogle(
        _PerfectAgent(), limit=2, output_path=None, verbose=False, concurrency=2
    )
    assert serial_summary.exact_match == parallel_summary.exact_match
    assert serial_summary.contains_match == parallel_summary.contains_match
    assert [r.question for r in serial_rows] == [r.question for r in parallel_rows]


@patch(
    "src.model.post_training.eval.bamboogle.load_bamboogle", return_value=_FAKE_DATASET
)
def test_resume_skips_completed(mock_load, tmp_path):
    """With resume=True, already-completed examples are skipped."""
    out = tmp_path / "out.jsonl"
    out.write_text(
        json.dumps(
            {
                "id": "1",
                "question": "Who invented the telephone?",
                "golden_answers": ["Alexander Graham Bell"],
                "prediction": "Alexander Graham Bell",
                "exact_match": 1.0,
                "contains_match": 1.0,
                "reward_total": None,
                "reward_components": {},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    call_count = 0

    class _CountingAgent:
        def invoke(self, state: dict) -> MagicMock:
            nonlocal call_count
            call_count += 1
            r = MagicMock()
            r.answer = "Paris"
            del r.metadata
            return r

    evaluate_bamboogle(
        _CountingAgent(), limit=2, output_path=out, verbose=False, resume=True
    )
    assert call_count == 1
    lines = out.read_text().splitlines()
    assert len(lines) == 2


@patch(
    "src.model.post_training.eval.bamboogle.load_bamboogle", return_value=_FAKE_DATASET
)
def test_resume_false_reruns_all(mock_load, tmp_path):
    """With resume=False (default), all examples run even if output exists."""
    out = tmp_path / "out.jsonl"
    out.write_text(
        json.dumps(
            {
                "id": "1",
                "question": "Who invented the telephone?",
                "golden_answers": ["Alexander Graham Bell"],
                "prediction": "Alexander Graham Bell",
                "exact_match": 1.0,
                "contains_match": 1.0,
                "reward_total": None,
                "reward_components": {},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    call_count = 0

    class _CountingAgent:
        def invoke(self, state: dict) -> MagicMock:
            nonlocal call_count
            call_count += 1
            r = MagicMock()
            r.answer = "Paris"
            del r.metadata
            return r

    evaluate_bamboogle(
        _CountingAgent(), limit=2, output_path=out, verbose=False, resume=False
    )
    assert call_count == 2


def test_write_summary_json(tmp_path):
    from examples.run_bamboogle_eval import write_summary_json
    from src.model.post_training.eval.bamboogle import BamboogleSummary

    out = tmp_path / "bamboogle_results.jsonl"
    summary = BamboogleSummary(
        num_examples=3, exact_match=0.66, contains_match=1.0, avg_reward=0.5
    )
    path = write_summary_json(str(out), summary)

    assert path == tmp_path / "bamboogle_results.summary.json"
    import json

    data = json.loads(path.read_text())
    assert data == {
        "num_examples": 3,
        "exact_match": 0.66,
        "contains_match": 1.0,
        "avg_reward": 0.5,
        "avg_reward_retrieval": None,
        "avg_reward_generation": None,
    }


@patch(
    "src.model.post_training.eval.bamboogle.load_bamboogle", return_value=_FAKE_DATASET
)
def test_summary_splits_the_reward_by_side(mock_load, tmp_path):
    """The reward's retrieval side and generation side are averaged apart, so
    a change in avg_reward can be attributed to the documents or the answer."""

    class _Agent:
        def invoke(self, state: dict) -> MagicMock:
            r = MagicMock()
            r.answer = "Paris"
            del r.metadata
            return r

    class _RewardFn:
        def reward_components(self, *, output, ground_truth, judge_fn):
            return {
                "correctness": 0.5,  # generation side
                "citation_support": 0.1,  # generation side
                "search_quality": 0.2,  # retrieval side
                "retriever_cost": -0.1,  # retrieval side
                "total": 0.7,
            }

    from src.model.post_training.eval.bamboogle import evaluate_bamboogle

    summary, _ = evaluate_bamboogle(
        _Agent(), limit=2, output_path=None, verbose=False, reward_fn=_RewardFn()
    )
    assert summary.avg_reward == pytest.approx(0.7)
    assert summary.avg_reward_retrieval == pytest.approx(0.1)
    assert summary.avg_reward_generation == pytest.approx(0.6)
    assert "retrieval side  : 0.1000" in str(summary)
    assert "generation side : 0.6000" in str(summary)


@patch(
    "src.model.post_training.eval.bamboogle.load_bamboogle", return_value=_FAKE_DATASET
)
def test_evaluate_reward_uses_gold_list(mock_load):
    """judge_fn must check against all gold answers, not just the first."""
    from src.model.post_training.reward import SearchRewardFunction, SearchRewardConfig

    dataset = [
        {
            "id": "1",
            "question": "Who?",
            "golden_answers": ["Alexander Graham Bell", "Bell"],
        }
    ]
    mock_load.return_value = dataset

    class _SecondGoldAgent:
        def invoke(self, state):
            r = MagicMock()
            r.answer = "Bell"  # matches second gold answer only
            del r.metadata
            return r

    reward_fn = SearchRewardFunction(SearchRewardConfig.sparse_final_only())
    summary, _ = evaluate_bamboogle(
        _SecondGoldAgent(),
        reward_fn=reward_fn,
        limit=1,
        output_path=None,
        verbose=False,
    )
    assert summary.avg_reward > 0.0


# ---------------------------------------------------------------------------
# Shared construction with run_bamboogle_synthetic_grpo
# ---------------------------------------------------------------------------


class _Tokenizer:
    pad_token_id = 0
    eos_token_id = 0

    def decode(self, ids, **kwargs):
        return "x"

    def encode(self, text, **kwargs):
        return [1]


def _eval_args(**overrides):
    import argparse

    defaults = dict(
        model="unused/model",
        local=False,
        device="cpu",
        allow_unsafe_mps=False,
        allow_remote_model_downloads=False,
        server_url="http://localhost:8080",
        search_url="http://localhost:8000/retrieve",
        topk=5,
        max_turns=8,
        generation_timeout_seconds=0.0,
        generation_heartbeat_seconds=10.0,
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def test_server_manager_comes_from_the_serving_factory(monkeypatch):
    """The backend choice belongs to src.model.serving, not to this script.

    The script used to import the manager classes from another example module
    and construct them by hand, so a change to the factory's selection rules
    reached run_agentic_search and skipped this benchmark.
    """
    import src.model.serving as serving
    from examples.run_bamboogle_eval import build_server_manager_from_args

    calls: list[dict] = []

    def _spy(tokenizer, **kwargs):
        calls.append(kwargs)
        return "manager"

    monkeypatch.setattr(serving, "build_server_manager", _spy)

    assert build_server_manager_from_args(_eval_args(), _Tokenizer()) == "manager"
    assert calls[-1]["server_url"] == "http://localhost:8080"
    assert "model" in calls[-1]

    build_server_manager_from_args(_eval_args(local=True), _Tokenizer())
    assert "server_url" not in calls[-1]
    assert calls[-1]["model"] == "unused/model"
    assert calls[-1]["device"] == "cpu"


def test_server_manager_selection_matches_the_cli_flags():
    from examples.run_bamboogle_eval import build_server_manager_from_args
    from src.model.serving import LocalServerManager, OpenAIServerManager

    remote = build_server_manager_from_args(_eval_args(), _Tokenizer())
    assert isinstance(remote, OpenAIServerManager)

    local = build_server_manager_from_args(_eval_args(local=True), _Tokenizer())
    assert isinstance(local, LocalServerManager)
    assert local.model_path == "unused/model"
    # --local loads lazily, so nothing was fetched by constructing it.
    assert local._model is None


def test_search_loop_carries_the_cli_retrieval_settings():
    from examples.run_bamboogle_eval import build_search_loop

    loop = build_search_loop(
        _eval_args(search_url="http://retrieval:9/retrieve", topk=3, max_turns=2),
        _Tokenizer(),
        server_manager=object(),
    )
    config = loop.search_config
    assert config.search_url == "http://retrieval:9/retrieve"
    assert config.topk == 3
    assert config.max_turns == 2


def test_synthetic_script_shares_the_eval_builders(monkeypatch):
    """The synthetic-rollout script must score the agent the eval scores.

    It used to reach for a private ``_build_server_manager`` and repeat the loop
    construction, so a change to either could leave the judge rating a
    different agent than the benchmark measures.
    """
    import examples.run_bamboogle_eval as eval_script
    from examples.run_bamboogle_synthetic_grpo import _build_loop_factory

    monkeypatch.setattr(
        eval_script, "build_server_manager_from_args", lambda a, t: "manager"
    )
    monkeypatch.setattr(eval_script, "build_search_loop", lambda a, t, m: (a, t, m))

    args = _eval_args()
    tokenizer = _Tokenizer()
    factory, server_manager = _build_loop_factory(args, tokenizer)

    assert server_manager == "manager"
    assert factory() == (args, tokenizer, "manager")


def test_search_loop_resolves_through_the_agent_registry(monkeypatch):
    """All three example entrypoints must pick the loop the same way.

    run_agentic_search dispatches through the registry; the bamboogle scripts
    named SearchAgentLoop directly, so a registry change (a new default, a
    swapped implementation behind the ``search`` alias) reached the CLI and
    skipped the benchmark.
    """
    import src.agents.core.base as agent_base
    from examples.run_bamboogle_eval import build_search_loop

    resolved: list[str] = []

    class _Sentinel:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    def _spy(name: str):
        resolved.append(name)
        return _Sentinel

    monkeypatch.setattr(agent_base, "get_registered_agent_loop", _spy)

    loop = build_search_loop(_eval_args(), _Tokenizer(), server_manager=object())

    assert isinstance(loop, _Sentinel)
    assert resolved == [agent_base.resolve_agent_name("search")]
