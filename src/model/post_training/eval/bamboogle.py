"""Bamboogle benchmark evaluation for agentic search.

Bamboogle is a two-hop question-answering benchmark designed to be
difficult for retrieval-augmented systems.  This module downloads the
test split, runs an agent on each question, and reports exact-match,
contains-match, and (optionally) shaped GRPO reward metrics.

Quick start::

    from src.agents.core.graph_base import BaseAgent          # your subclass
    from src.model.post_training.eval.bamboogle import evaluate_bamboogle
    from src.model.post_training.reward import SearchRewardFunction, SearchRewardConfig

    summary, rows = evaluate_bamboogle(
        agent,
        reward_fn=SearchRewardFunction(SearchRewardConfig.second_pass()),
        limit=50,
        output_path="results/bamboogle.jsonl",
    )
    print(summary)

Notes on reward scoring
-----------------------
``reward_fn.reward_components()`` expects an :class:`~src.agents.core.base.AgentLoopOutput`
with populated ``metrics`` and ``context``.  When the agent is a
:class:`~src.agents.core.graph_base.BaseAgent` (invoke-based), only the answer
string is available, so a minimal stub output is used — ``correctness_weight``
and ``format_reward_weight`` fire; search-quality components are zero.

For full shaped-reward scores, pass an agent whose ``invoke()`` stores the
underlying ``AgentLoopOutput`` in ``result.metadata["loop_output"]``.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

from concurrent.futures import ThreadPoolExecutor

from tqdm import tqdm

from src.model.post_training.reward import reward_sides

if TYPE_CHECKING:
    from src.model.post_training.reward import SearchRewardFunction

BAMBOOGLE_URL = (
    "https://huggingface.co/datasets/RUC-NLPIR/FlashRAG_datasets/"
    "resolve/main/bamboogle/test.jsonl"
)

_DEFAULT_CACHE = Path.home() / ".cache" / "agentic_search" / "bamboogle_test.jsonl"

# ---------------------------------------------------------------------------
# Text normalisation and matching
# ---------------------------------------------------------------------------


def normalize_text(s: str) -> str:
    """Lowercase, strip articles, collapse whitespace."""
    s = s.lower()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    return " ".join(s.split())


def exact_match(prediction: str, gold_answers: list[str]) -> float:
    """1.0 if *prediction* exactly equals any gold answer after normalisation."""
    pred = normalize_text(prediction)
    return float(any(pred == normalize_text(g) for g in gold_answers))


def contains_match(prediction: str, gold_answers: list[str]) -> float:
    """1.0 if any gold answer (normalised) is a substring of *prediction*."""
    pred = normalize_text(prediction)
    return float(any(normalize_text(g) in pred for g in gold_answers))


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------


def load_bamboogle(
    limit: int | None = None,
    cache_path: str | Path | None = _DEFAULT_CACHE,
) -> list[dict[str, Any]]:
    """Download and parse the Bamboogle test split from HuggingFace.

    Args:
        limit: Return only the first *limit* examples.  ``None`` returns all
            125 examples in the test split.
        cache_path: Path to cache the raw JSONL locally.  On subsequent calls
            the file is read from disk instead of re-downloading.  Set to
            ``None`` to disable caching.

    Returns:
        List of dicts with ``"question"`` and ``"golden_answers"`` keys.
    """
    if cache_path is not None:
        cache_path = Path(cache_path)
        if cache_path.exists():
            rows = [
                json.loads(line)
                for line in cache_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            return rows[:limit] if limit is not None else rows

    import requests

    resp = requests.get(BAMBOOGLE_URL, timeout=30)
    resp.raise_for_status()
    rows = [json.loads(line) for line in resp.text.splitlines() if line.strip()]

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(resp.text, encoding="utf-8")

    return rows[:limit] if limit is not None else rows


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass
class BamboogleResult:
    """Per-example evaluation result."""

    id: str | None
    question: str
    golden_answers: list[str]
    prediction: str
    exact_match: float
    contains_match: float
    reward_total: float | None = None
    reward_components: dict[str, float] = field(default_factory=dict)


@dataclass
class BamboogleSummary:
    """Aggregate metrics over the evaluated split."""

    num_examples: int
    exact_match: float
    contains_match: float
    avg_reward: float | None = None
    # The reward split by side (see ``reward.reward_sides``): what the
    # documents earned versus what the answer earned. Present whenever
    # ``avg_reward`` is.
    avg_reward_retrieval: float | None = None
    avg_reward_generation: float | None = None

    def __str__(self) -> str:
        lines = [
            f"examples : {self.num_examples}",
            f"exact_match    : {self.exact_match:.3f}",
            f"contains_match : {self.contains_match:.3f}",
        ]
        if self.avg_reward is not None:
            lines.append(f"avg_reward     : {self.avg_reward:.4f}")
        if self.avg_reward_retrieval is not None:
            lines.append(f"  retrieval side  : {self.avg_reward_retrieval:.4f}")
        if self.avg_reward_generation is not None:
            lines.append(f"  generation side : {self.avg_reward_generation:.4f}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _extract_answer(agent_result: Any) -> str:
    """Pull the answer string from whatever the agent returns."""
    if isinstance(agent_result, str):
        return agent_result
    if isinstance(agent_result, dict):
        return agent_result.get("answer", "")
    return getattr(agent_result, "answer", "") or ""


def _make_judge_fn(gold_answers: list[str]) -> Callable[[str, str], float]:
    """Return a judge_fn that checks *prediction* against *gold_answers*.

    The second argument (ground_truth string) is ignored — the gold list is
    captured in the closure.  This matches the ``(str, str) -> float``
    signature expected by :class:`SearchRewardFunction`.
    """

    def _judge(prediction: str, _gt: str) -> float:
        return contains_match(prediction, gold_answers)

    return _judge


def _to_loop_output(agent_result: Any) -> Any:
    """Convert an AgentResult to a minimal AgentLoopOutput for reward scoring.

    If the result already carries a full ``AgentLoopOutput`` in
    ``metadata["loop_output"]``, that is returned directly.  Otherwise a stub
    is built from the answer string alone (search-quality reward components
    will be zero).
    """
    from src.agents.core.base import AgentLoopOutput

    # Rich path: agent stored the real loop output in metadata.
    if hasattr(agent_result, "metadata") and isinstance(agent_result.metadata, dict):
        loop_output = agent_result.metadata.get("loop_output")
        if isinstance(loop_output, AgentLoopOutput):
            return loop_output

    # Stub path: only the answer is available.
    answer = _extract_answer(agent_result)
    return AgentLoopOutput(
        prompt_ids=[],
        response_ids=[],
        response_mask=[],
        num_turns=1,
        final_answer=answer,
    )


# ---------------------------------------------------------------------------
# Resume helpers
# ---------------------------------------------------------------------------


def _load_completed_ids(path: Path) -> set[str]:
    """Return the set of question strings already recorded in *path*.

    Uses ``question`` as the key (always present) rather than ``id`` (may be None).
    """
    if not path.exists():
        return set()
    completed: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            completed.add(row["question"])
        except (json.JSONDecodeError, KeyError):
            pass
    return completed


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------


def evaluate_bamboogle(
    agent: Any,
    *,
    reward_fn: SearchRewardFunction | None = None,
    limit: int | None = 20,
    output_path: str | Path | None = "bamboogle_results.jsonl",
    verbose: bool = True,
    concurrency: int = 1,
    resume: bool = False,
) -> tuple[BamboogleSummary, list[BamboogleResult]]:
    """Run *agent* on the Bamboogle benchmark and report accuracy metrics.

    Args:
        agent: Any object with an ``invoke(state: dict) -> Any`` method.
            Compatible with :class:`~src.agents.core.graph_base.BaseAgent`.
        reward_fn: Optional :class:`~src.model.post_training.reward.SearchRewardFunction`.
            When provided, ``reward_components()`` is called on each result and
            the per-component breakdown is stored alongside accuracy scores.
        limit: Number of examples to evaluate.  ``None`` evaluates all 125.
        output_path: Write per-example results as JSONL here.  ``None``
            skips writing.
        verbose: Show a tqdm progress bar.
        concurrency: Number of questions to evaluate in parallel.  Each thread
            runs one ``agent.invoke()`` call.  Use 1 (default) for serial
            execution.  Values of 4–8 work well with SerpAPI free tier.
        resume: When True and *output_path* already exists, skip examples whose
            questions appear in that file and append new results to it.

    Returns:
        ``(summary, rows)`` — a :class:`BamboogleSummary` and a list of
        :class:`BamboogleResult` objects.
    """
    dataset = load_bamboogle(limit=limit)

    prior_results: list[BamboogleResult] = []
    if resume and output_path is not None:
        out_path = Path(output_path)
        completed_questions = _load_completed_ids(out_path)
        if completed_questions:
            for line in out_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                    prior_results.append(
                        BamboogleResult(
                            id=row.get("id"),
                            question=row["question"],
                            golden_answers=row.get("golden_answers", []),
                            prediction=row.get("prediction", ""),
                            exact_match=row.get("exact_match", 0.0),
                            contains_match=row.get("contains_match", 0.0),
                            reward_total=row.get("reward_total"),
                            reward_components=row.get("reward_components", {}),
                        )
                    )
                except (json.JSONDecodeError, KeyError):
                    pass
            dataset = [
                ex for ex in dataset if ex["question"] not in completed_questions
            ]

    def _run_one(ex: dict[str, Any]) -> BamboogleResult:
        question: str = ex["question"]
        gold_answers: list[str] = ex.get("golden_answers") or ex.get("answers") or []

        agent_result = agent.invoke(
            {"messages": [{"role": "user", "content": question}]}
        )
        answer = _extract_answer(agent_result)

        em = exact_match(answer, gold_answers)
        cm = contains_match(answer, gold_answers)

        reward_total: float | None = None
        components: dict[str, float] = {}
        if reward_fn is not None:
            loop_output = _to_loop_output(agent_result)
            judge_fn = _make_judge_fn(gold_answers)
            components = reward_fn.reward_components(
                output=loop_output,
                ground_truth=gold_answers[0] if gold_answers else "",
                judge_fn=judge_fn,
            )
            reward_total = float(components.get("total", 0.0))

        return BamboogleResult(
            id=ex.get("id"),
            question=question,
            golden_answers=gold_answers,
            prediction=answer,
            exact_match=em,
            contains_match=cm,
            reward_total=reward_total,
            reward_components=components,
        )

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        mapped = pool.map(_run_one, dataset)
        if verbose:
            mapped = tqdm(mapped, total=len(dataset), desc="Bamboogle")
        new_results = list(mapped)

    results = prior_results + new_results

    total_em = sum(r.exact_match for r in results)
    total_contains = sum(r.contains_match for r in results)
    rewarded = [r for r in results if r.reward_total is not None]
    n_reward = len(rewarded)
    total_reward = sum(r.reward_total for r in rewarded)
    sides = [reward_sides(r.reward_components or {}) for r in rewarded]

    n = len(results)
    summary = BamboogleSummary(
        num_examples=n,
        exact_match=total_em / n if n else 0.0,
        contains_match=total_contains / n if n else 0.0,
        avg_reward=total_reward / n_reward if n_reward else None,
        avg_reward_retrieval=(
            sum(s["retrieval"] for s in sides) / n_reward if n_reward else None
        ),
        avg_reward_generation=(
            sum(s["generation"] for s in sides) / n_reward if n_reward else None
        ),
    )

    if output_path is not None:
        if resume and prior_results:
            _append_jsonl(new_results, Path(output_path))
        else:
            _write_jsonl(results, Path(output_path))

    if verbose:
        print(summary)

    return summary, results


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------


def _result_to_dict(r: BamboogleResult) -> dict[str, Any]:
    return {
        "id": r.id,
        "question": r.question,
        "golden_answers": r.golden_answers,
        "prediction": r.prediction,
        "exact_match": r.exact_match,
        "contains_match": r.contains_match,
        "reward_total": r.reward_total,
        "reward_components": r.reward_components,
    }


def _write_jsonl(results: list[BamboogleResult], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(_result_to_dict(r), ensure_ascii=False) + "\n")


def _append_jsonl(results: list[BamboogleResult], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(_result_to_dict(r), ensure_ascii=False) + "\n")
