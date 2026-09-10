"""Reward function and GRPO outcome-advantage computation for SearchAgentLoop rollouts."""

from __future__ import annotations

import math
import re
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from typing import Callable

from ...agents.core.base import AgentLoopOutput
from ...context.search import AgentContext

# Type aliases for judge callables.
JudgeFn = Callable[[str, str], float]
BatchJudgeFn = Callable[[list[str], list[str]], list[float]]
_WHITESPACE_PATTERN = re.compile(r"\s+")
_CITATION_RE = re.compile(r"\[(?:D\d+|R\d+Q\d+D\d+)\]")
_ANSWER_TAG_RE = re.compile(r"<answer>.*?</answer>", re.DOTALL | re.IGNORECASE)

# Four conceptual reward dimensions grouping the fine-grained reward_components
# terms. Each member is a key produced by SearchRewardFunction.reward_components.
REWARD_DIMENSIONS: dict[str, tuple[str, ...]] = {
    "correctness": ("correctness",),
    "citation_support": (
        "citation_support",
        "unsupported_claim_penalty",
        "fetch_usefulness_reward",
        "format_reward",
    ),
    "retrieval_quality": (
        "search_quality",
        "subquestion_coverage",
        "evidence_gain",
        "early_stop_bonus",
        "answer_when_evidence_insufficient_penalty",
        "forced_final_answer_penalty",
        "search_budget_exhausted_without_answer_penalty",
    ),
    "search_efficiency": (
        "per_search_penalty",
        "unnecessary_search_penalty",
        "duplicate_query_penalty",
        "budget_penalty",
        "unnecessary_fetch_penalty",
        "retriever_cost",
        "rerank_cost",
    ),
}

# reward_components keys that are metadata or rollups, not dimension members.
_NON_DIMENSION_KEYS: frozenset[str] = frozenset(
    {"reward_mode", "terminal_reward", "shaping_total", "total", "human_feedback"}
)


def group_reward_components(components: dict[str, float]) -> dict[str, float]:
    """Roll the flat reward_components breakdown up into the 4 reward dimensions.

    Sums each dimension's member terms (missing keys count as 0.0), returning a
    dict with exactly the four keys in :data:`REWARD_DIMENSIONS`. The result is
    the pre-scale decomposition: ``sum(result.values())`` equals
    ``terminal_reward + shaping_total`` for a full components dict.
    """
    return {
        dimension: sum(float(components.get(key, 0.0)) for key in members)
        for dimension, members in REWARD_DIMENSIONS.items()
    }


# Which side of the pipeline each reward dimension scores. ``correctness`` and
# ``citation_support`` judge the answer; ``retrieval_quality`` and
# ``search_efficiency`` judge the documents and the cost of getting them.
REWARD_SIDES: dict[str, tuple[str, ...]] = {
    "retrieval": ("retrieval_quality", "search_efficiency"),
    "generation": ("correctness", "citation_support"),
}


def reward_sides(components: dict[str, float]) -> dict[str, float]:
    """Roll the four dimensions up into a retrieval side and a generation side.

    Purely additive over :func:`group_reward_components`: the two values sum
    to the same pre-scale total the dimensions do, so a reward regression can
    be attributed to the retriever or the generator without changing any
    existing value.
    """
    dims = group_reward_components(components)
    return {
        side: sum(dims[dimension] for dimension in members)
        for side, members in REWARD_SIDES.items()
    }


def normalize_answer_text(text: str) -> str:
    """Normalize an answer string for simple sparse-reward matching."""
    lowered = text.strip().lower()
    lowered = _WHITESPACE_PATTERN.sub(" ", lowered)
    return lowered


def token_f1_score(pred: str, gold: str) -> float:
    """Bag-of-words token F1 between a predicted and gold answer.

    Normalises both strings via :func:`normalize_answer_text`, computes the
    bag-of-words intersection, and returns the F1 (harmonic mean of precision
    and recall).  Useful as a soft alternative to exact-match for NQ-style QA.

    Edge cases:
    - Both empty → 1.0  (both correctly produced nothing)
    - One empty  → 0.0  (total miss)
    - No shared tokens → 0.0
    """
    pred_tokens = normalize_answer_text(pred).split()
    gold_tokens = normalize_answer_text(gold).split()
    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0
    pred_bag = {}
    for t in pred_tokens:
        pred_bag[t] = pred_bag.get(t, 0) + 1
    gold_bag = {}
    for t in gold_tokens:
        gold_bag[t] = gold_bag.get(t, 0) + 1
    overlap = sum(min(pred_bag.get(t, 0), c) for t, c in gold_bag.items())
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_tokens)
    recall = overlap / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def format_compliance_reward(
    response: str,
    *,
    require_citations: bool = True,
    require_answer_tag: bool = False,
) -> float:
    """Reward structural compliance in a search-agent response.

    Checks for:
    - Inline citation markers (``[D1]`` or search-loop ``[R1Q1D1]`` labels)
      when ``require_citations=True``
    - An ``<answer>…</answer>`` block when ``require_answer_tag=True``

    Returns the fraction of required checks that pass (0.0, 0.5, or 1.0 when
    both checks are required; 0.0 or 1.0 when only one is required).
    """
    if not response:
        return 0.0
    checks = 0
    passed = 0
    if require_citations:
        checks += 1
        if _CITATION_RE.search(response):
            passed += 1
    if require_answer_tag:
        checks += 1
        if _ANSWER_TAG_RE.search(response):
            passed += 1
    return passed / checks if checks else 1.0


@dataclass
class CompositeRewardConfig:
    """Blend multiple judge functions into a single scalar reward.

    Each entry in *judges* is a ``(name, weight, judge_fn)`` triple where
    ``judge_fn(pred, gold) -> float`` must return a value in ``[0, 1]``.
    Weights need not sum to 1 — they are applied as literal multipliers.

    Usage::

        cfg = CompositeRewardConfig(judges=[
            ("exact",  0.6, simple_sparse_correctness_reward),
            ("f1",     0.3, token_f1_score),
            ("format", 0.1, lambda p, _: format_compliance_reward(p)),
        ])
        score = cfg.compute("Paris is the capital", "paris")
        fn = cfg.as_judge_fn()   # drop-in replacement for a plain JudgeFn
    """

    judges: list[tuple[str, float, JudgeFn]] = field(default_factory=list)

    def compute(self, pred: str, gold: str) -> float:
        """Return the weighted sum of all judge scores."""
        return sum(w * fn(pred, gold) for _, w, fn in self.judges)

    def compute_breakdown(self, pred: str, gold: str) -> dict[str, float]:
        """Return a per-judge breakdown for logging/debugging."""
        return {name: w * fn(pred, gold) for name, w, fn in self.judges}

    def as_judge_fn(self) -> JudgeFn:
        """Return a single ``(pred, gold) -> float`` callable.

        The returned function is suitable anywhere a plain :data:`JudgeFn` is
        accepted (e.g. :meth:`SearchRewardFunction.compute`).
        """

        def _judge(pred: str, gold: str) -> float:
            return self.compute(pred, gold)

        return _judge


def simple_sparse_correctness_reward(
    pred: str,
    gold: str,
    *,
    partial_match_reward: float = 0.7,
) -> float:
    """Simple first-pass sparse reward for final answers.

    Rules:
    - exact normalized match -> 1.0
    - normalized gold is contained in normalized prediction -> 0.7
    - otherwise -> 0.0
    """
    norm_pred = normalize_answer_text(pred)
    norm_gold = normalize_answer_text(gold)
    if not norm_pred or not norm_gold:
        return 0.0
    if norm_pred == norm_gold:
        return 1.0
    if norm_gold in norm_pred:
        return float(partial_match_reward)
    return 0.0


def _score_answers(
    judge_fn: JudgeFn,
    answers: list[str],
    ground_truths: list[str],
    *,
    batch_judge_fn: BatchJudgeFn | None = None,
) -> list[float]:
    """Score answer/ground-truth pairs, preferring the batch judge when provided.

    Pass `batch_judge_fn` to have an LLM judge score all pairs in a single
    API call instead of making N separate round-trips.
    """
    if batch_judge_fn is not None:
        return [float(s) for s in batch_judge_fn(answers, ground_truths)]
    return [float(judge_fn(a, g)) for a, g in zip(answers, ground_truths)]


@dataclass(frozen=True)
class SearchRewardConfig:
    # Reward aggregation mode:
    # - "shaped": final-answer reward plus process-level shaping terms
    # - "sparse_final_only": only the final-answer judge contributes to total
    reward_mode: str = "shaped"

    # Primary signal — weight applied to the judge score in [0, 1].
    correctness_weight: float = 1.0

    # Fraction of retrieved documents cited in the final answer, weighted by this.
    citation_support_weight: float = 0.3

    # Fraction of declared subquestions whose evidence was marked sufficient.
    subquestion_coverage_weight: float = 0.2

    # Reward for good evaluator verdicts and strong per-query search quality.
    search_quality_weight: float = 0.15

    # Simple first-pass efficiency penalty: subtract this per search round used.
    # Example: -0.02 * num_searches
    per_search_penalty: float = 0.0

    # Applied per search round beyond the first (encourages efficiency).
    unnecessary_search_penalty: float = -0.05

    # Applied per repeated query issued across turns.
    duplicate_query_penalty: float = -0.1

    # Applied once if rounds_used / max_search_rounds exceeds this threshold.
    budget_penalty_threshold: float = 0.8
    budget_penalty: float = -0.1

    # Applied per fetched page that does not contribute to cited evidence.
    unnecessary_fetch_penalty: float = -0.1

    # Applied once when the agent answers with no retrieval citations despite
    # having issued searches that returned results.  A negative value penalises
    # answers that make claims without any retrieved evidence — i.e. the agent
    # searched, got documents, and then ignored them.
    # Phase 2 recommended: -0.1
    unsupported_claim_penalty: float = 0.0

    # Applied once when the agent answers despite insufficient evidence.
    answer_when_evidence_insufficient_penalty: float = -0.2

    # Applied once when the loop forced a best-effort answer at a dead-end/cap.
    # A salvage: milder than the voluntary-insufficient penalty, far better than
    # returning nothing. Mutually exclusive with answer_when_evidence_insufficient.
    forced_final_answer_penalty: float = -0.05

    # Applied once when the search budget is exhausted and no answer is produced.
    search_budget_exhausted_without_answer_penalty: float = -0.2

    # Reward when fetched pages are cited in the final answer.
    fetch_usefulness_reward: float = 0.1

    # Reference budget for the budget-penalty calculation.  Set to the same
    # value as SearchAgentLoopConfig.max_search_limit when known.
    max_search_rounds: int = 5

    # Reward for structural compliance in the answer. Uses
    # :func:`format_compliance_reward` with ``require_citations=True`` by default.
    # Accepts both compact [D1] labels and search-loop [R1Q1D1] labels. Set to
    # 0.0 to disable (the default).
    format_reward_weight: float = 0.0

    # Scale the final total reward before returning.  Useful for aligning
    # reward magnitudes across training phases without changing all weights.
    reward_scale: float = 1.0

    # Human feedback signal weight.  When 0.0 (default) the term is always
    # zero — existing reward presets produce identical scores.
    human_feedback_weight: float = 0.0

    # --- Phase C action-policy terms (all 0.0 by default → presets unchanged) ---
    # Per-round cost of each retriever the policy can choose, so GRPO learns the
    # "search web vs vector-DB" trade-off.  Negative = cost.  Web is typically
    # priced higher (e.g. retriever_cost_web = 5 × retriever_cost_vdb).
    retriever_cost_vdb: float = 0.0
    retriever_cost_web: float = 0.0
    # Flat cost charged per rerank action the policy invokes.
    rerank_cost: float = 0.0
    # Weight on the per-round improvement in evidence_score (Δ summed over the
    # run), rewarding searches that actually make the evidence more sufficient.
    evidence_gain_weight: float = 0.0
    # Flat bonus per round flagged as an evidence plateau (early-stop signal).
    # 0.0 by default → existing presets are byte-stable.
    early_stop_bonus: float = 0.0

    @classmethod
    def _zeroed(
        cls,
        *,
        reward_mode: str = "shaped",
        correctness_weight: float = 1.0,
        max_search_rounds: int = 5,
        reward_scale: float = 1.0,
    ) -> "SearchRewardConfig":
        """All shaping weights and penalties disabled — baseline for presets."""
        return cls(
            reward_mode=reward_mode,
            correctness_weight=correctness_weight,
            citation_support_weight=0.0,
            subquestion_coverage_weight=0.0,
            search_quality_weight=0.0,
            per_search_penalty=0.0,
            unnecessary_search_penalty=0.0,
            duplicate_query_penalty=0.0,
            budget_penalty_threshold=1.0,
            budget_penalty=0.0,
            unnecessary_fetch_penalty=0.0,
            unsupported_claim_penalty=0.0,
            answer_when_evidence_insufficient_penalty=0.0,
            forced_final_answer_penalty=0.0,
            search_budget_exhausted_without_answer_penalty=0.0,
            fetch_usefulness_reward=0.0,
            format_reward_weight=0.0,
            retriever_cost_vdb=0.0,
            retriever_cost_web=0.0,
            rerank_cost=0.0,
            evidence_gain_weight=0.0,
            early_stop_bonus=0.0,
            max_search_rounds=max_search_rounds,
            reward_scale=reward_scale,
        )

    @classmethod
    def sparse_final_only(
        cls,
        *,
        correctness_weight: float = 1.0,
    ) -> "SearchRewardConfig":
        """Build a strict sparse-reward config for agent RL.

        In this mode the rollout gets reward only from the final answer score.
        Search, reasoning, and retrieval traces remain available as labelled
        diagnostics, but they do not contribute to the optimisation target.
        """
        return cls._zeroed(
            reward_mode="sparse_final_only",
            correctness_weight=correctness_weight,
        )

    @classmethod
    def simple_sparse_with_search_penalty(
        cls,
        *,
        correctness_weight: float = 1.0,
        per_search_penalty: float = -0.02,
    ) -> "SearchRewardConfig":
        """Build the recommended first-pass reward for early agent RL.

        Objective:

            reward = correctness_reward - 0.02 * num_searches

        Everything else is disabled so the signal stays easy to reason about.
        Use with :func:`simple_sparse_correctness_reward` as the judge for a
        straightforward first training phase.
        """
        return replace(
            cls._zeroed(correctness_weight=correctness_weight),
            per_search_penalty=per_search_penalty,
        )

    @classmethod
    def second_pass(
        cls,
        *,
        correctness_weight: float = 1.0,
        per_search_penalty: float = -0.02,
        citation_support_weight: float = 0.1,
        unsupported_claim_penalty: float = -0.1,
        duplicate_query_penalty: float = -0.05,
    ) -> "SearchRewardConfig":
        """Phase 2 reward: first-pass formula plus citation and query-quality signals.

        Adds three terms to the Phase 1 (simple sparse + search penalty) objective:

            reward = correctness
                   - 0.02 * num_searches            ← search efficiency
                   + 0.1  * citation_support         ← fraction of retrieved docs cited
                   - 0.1  * unsupported_claim        ← searched but cited nothing
                   - 0.05 * repeated_query_count     ← duplicate search penalty

        Use :func:`simple_sparse_correctness_reward` as the judge unless you
        already have a trained LLM evaluator.  Move to Phase 2 only after the
        model reliably produces answers in Phase 1 (policy has converged enough
        to benefit from the finer-grained signal).
        """
        return replace(
            cls._zeroed(correctness_weight=correctness_weight),
            per_search_penalty=per_search_penalty,
            citation_support_weight=citation_support_weight,
            unsupported_claim_penalty=unsupported_claim_penalty,
            duplicate_query_penalty=duplicate_query_penalty,
        )

    @classmethod
    def third_pass_with_format(
        cls,
        *,
        correctness_weight: float = 1.0,
        per_search_penalty: float = -0.02,
        citation_support_weight: float = 0.1,
        unsupported_claim_penalty: float = -0.1,
        duplicate_query_penalty: float = -0.05,
        format_reward_weight: float = 0.1,
    ) -> "SearchRewardConfig":
        """Phase 3 reward: second-pass formula plus format compliance.

        Adds a structural compliance bonus on top of Phase 2:

            reward = correctness
                   - 0.02 * num_searches
                   + 0.1  * citation_support
                   - 0.1  * unsupported_claim
                   - 0.05 * repeated_query_count
                   + 0.1  * format_compliance   ← inline citation labels present

        Use :func:`token_f1_score` as the judge alongside
        :func:`simple_sparse_correctness_reward` via :class:`CompositeRewardConfig`
        to benefit from both exact-match and partial-match signals in this phase.
        """
        return replace(
            cls._zeroed(correctness_weight=correctness_weight),
            per_search_penalty=per_search_penalty,
            citation_support_weight=citation_support_weight,
            unsupported_claim_penalty=unsupported_claim_penalty,
            duplicate_query_penalty=duplicate_query_penalty,
            format_reward_weight=format_reward_weight,
        )

    @classmethod
    def retriever_aware(
        cls,
        *,
        correctness_weight: float = 1.0,
        retriever_cost_vdb: float = -0.01,
        retriever_cost_web: float = -0.05,
        rerank_cost: float = -0.02,
        evidence_gain_weight: float = 0.1,
        early_stop_bonus: float = 0.0,
    ) -> "SearchRewardConfig":
        """Phase C preset: second-pass shaping plus action-policy costs.

        Adds the retriever/rerank/evidence-gain terms on top of the Phase 2
        objective so GRPO learns the action policy:

            reward = second_pass(...)                      ← correctness + shaping
                   + retriever_cost_vdb * vdb_searches      ← cheap internal search
                   + retriever_cost_web * web_searches      ← web priced 5× the VDB
                   + rerank_cost        * rerank_calls      ← flat cost per rerank
                   + evidence_gain_weight * evidence_gain   ← reward informative rounds

        Existing presets are untouched; this is opt-in for retriever-aware runs.
        """
        return replace(
            cls.second_pass(correctness_weight=correctness_weight),
            retriever_cost_vdb=retriever_cost_vdb,
            retriever_cost_web=retriever_cost_web,
            rerank_cost=rerank_cost,
            evidence_gain_weight=evidence_gain_weight,
            early_stop_bonus=early_stop_bonus,
        )


def group_relative_advantages(
    rewards: Sequence[float], *, normalize: bool = False
) -> list[float]:
    """Mean-center one prompt group's rewards; optionally divide by population std.

    This is the critic-free core signal GRPO trains on:

        advantage_i = reward_i - mean(group_rewards)

    With ``normalize=True`` the centered values are divided by the group's
    population standard deviation, which is what keeps the objective stable
    across groups whose reward scales differ:

        advantage_i = (reward_i - mean) / (std + 1e-8)

    A single-sample group has no relative comparison, so it gets ``0.0``.
    All-equal rewards give a zero std; the epsilon is what keeps the result
    finite rather than NaN.
    """
    n = len(rewards)
    if n <= 1:
        return [0.0] * n
    mean = sum(rewards) / n
    centered = [float(r) - mean for r in rewards]
    if not normalize:
        return centered
    std = math.sqrt(sum(c * c for c in centered) / n)
    return [c / (std + 1e-8) for c in centered]


def grouped_relative_advantages(
    rewards: Sequence[float],
    group_ids: Sequence[str],
    *,
    normalize: bool = False,
) -> list[float]:
    """Apply :func:`group_relative_advantages` per prompt group, in input order.

    Rollouts sharing a ``group_id`` were sampled from the same prompt and are
    the only ones compared against each other -- there is no cross-prompt
    mixing. The returned list is aligned with *rewards*.
    """
    if len(rewards) != len(group_ids):
        raise ValueError("rewards and group_ids must have the same length.")

    groups: dict[str, list[int]] = {}
    for index, group_id in enumerate(group_ids):
        groups.setdefault(group_id, []).append(index)

    advantages = [0.0] * len(rewards)
    for indices in groups.values():
        group_advantages = group_relative_advantages(
            [rewards[i] for i in indices], normalize=normalize
        )
        for index, advantage in zip(indices, group_advantages):
            advantages[index] = advantage
    return advantages


class SearchRewardFunction:
    """Computes per-rollout rewards and GRPO advantages for search agent training.

    Usage::

        reward_fn = SearchRewardFunction()
        rewards = [
            reward_fn.compute(output, ground_truth=gt, judge_fn=exact_match)
            for output, gt in zip(outputs, ground_truths)
        ]
        advantages = reward_fn.compute_batch_advantages(rewards, group_ids)
    """

    def __init__(self, config: SearchRewardConfig | None = None) -> None:
        self.config = config if config is not None else SearchRewardConfig()

    # ------------------------------------------------------------------
    # Per-rollout reward
    # ------------------------------------------------------------------

    def compute(
        self,
        output: AgentLoopOutput,
        ground_truth: str,
        judge_fn: Callable[[str, str], float],
    ) -> float:
        """Return a scalar reward for one rollout.

        Args:
            output: The :class:`AgentLoopOutput` produced by
                :meth:`SearchAgentLoop.run`.
            ground_truth: The reference answer string.
            judge_fn: ``(answer, ground_truth) -> float`` in ``[0, 1]``.
                Common choices: exact-match, token-F1, or an LLM judge.

        Returns:
            A scalar reward (may be negative when penalties dominate).
        """
        return self.reward_components(output, ground_truth, judge_fn)["total"]

    def compute_terminal_reward(
        self,
        output: AgentLoopOutput,
        ground_truth: str,
        judge_fn: Callable[[str, str], float],
    ) -> float:
        """Return the final-answer reward with no process shaping.

        This is the strict sparse-reward signal used in many Agent RL setups:

            reward = judge_fn(final_answer, ground_truth)

        It ignores search, retrieval, reasoning, and fetch behaviour entirely.
        Use this when you want an explicit terminal-only optimisation target
        regardless of the configured reward aggregation mode.
        """
        answer = output.final_answer or ""
        if not answer:
            return 0.0
        return float(self.config.correctness_weight) * judge_fn(answer, ground_truth)

    def compute_token_rewards(
        self,
        output: AgentLoopOutput,
        ground_truth: str,
        judge_fn: Callable[[str, str], float],
    ) -> list[float]:
        """Sparse token-level reward vector for Agent RL training.

        Places the scalar rollout reward at the last model-generated token
        (last position where response_mask == 1) and 0 everywhere else.
        This mirrors the standard sparse-reward formulation: search tokens,
        reasoning tokens, and retrieval tokens receive no reward signal — only
        the final <answer> token does.

        Falls back to the last token when response_mask is absent or all-zero.
        """
        scalar = self.compute(output, ground_truth, judge_fn)
        n = len(output.response_ids)
        if n == 0:
            return []
        token_rewards = [0.0] * n
        token_rewards[self._last_action_idx(output)] = scalar
        return token_rewards

    def compute_sparse_token_rewards(
        self,
        output: AgentLoopOutput,
        ground_truth: str,
        judge_fn: Callable[[str, str], float],
    ) -> list[float]:
        """Strict sparse token rewards using only the terminal answer score.

        Unlike :meth:`compute_token_rewards`, this helper never includes any
        process shaping. The final-answer reward is placed only at the last
        model-generated token and every earlier token gets 0.0.
        """
        scalar = self.compute_terminal_reward(output, ground_truth, judge_fn)
        n = len(output.response_ids)
        if n == 0:
            return []
        token_rewards = [0.0] * n
        token_rewards[self._last_action_idx(output)] = scalar
        return token_rewards

    def compute_batch_sparse_token_rewards(
        self,
        outputs: list[AgentLoopOutput],
        ground_truths: list[str],
        judge_fn: JudgeFn,
        *,
        batch_judge_fn: BatchJudgeFn | None = None,
    ) -> list[list[float]]:
        """Strict sparse token rewards for an entire batch in one call.

        The standard Agent RL sparse-reward loop:

            for each rollout i:
                token_rewards[i][last_action_token] = correctness_weight
                                                      * judge(answer_i, gt_i)
                token_rewards[i][all other tokens]  = 0.0

        Search tokens, reasoning tokens, and retrieval tokens always receive
        zero reward — only the final ``<answer>`` token carries the signal.

        Args:
            outputs: One :class:`AgentLoopOutput` per rollout.
            ground_truths: Reference answer for each rollout.
            judge_fn: ``(answer, ground_truth) -> float`` in ``[0, 1]``.
            batch_judge_fn: Optional batch variant
                ``(list[answers], list[ground_truths]) -> list[float]``.
                Supply this for LLM judges so all N pairs are scored in a
                single API call rather than N separate round-trips.

        Returns:
            One token-level reward vector per rollout.
        """
        if len(outputs) != len(ground_truths):
            raise ValueError("outputs and ground_truths must have the same length.")
        answers = [output.final_answer or "" for output in outputs]
        scores = _score_answers(
            judge_fn, answers, ground_truths, batch_judge_fn=batch_judge_fn
        )
        result: list[list[float]] = []
        for output, score in zip(outputs, scores):
            n = len(output.response_ids)
            if n == 0:
                result.append([])
                continue
            terminal = (
                float(self.config.correctness_weight) * score
                if output.final_answer
                else 0.0
            )
            token_rewards = [0.0] * n
            token_rewards[self._last_action_idx(output)] = terminal
            result.append(token_rewards)
        return result

    def reward_components(
        self,
        output: AgentLoopOutput,
        ground_truth: str,
        judge_fn: Callable[[str, str], float],
    ) -> dict[str, float]:
        """Same as :meth:`compute` but returns a labelled breakdown for logging.

        The ``"total"`` key equals what :meth:`compute` returns.
        """
        answer = output.final_answer or ""
        correctness = judge_fn(answer, ground_truth) if answer else 0.0
        return self._reward_components_from_correctness(output, correctness)

    def reward_dimensions(
        self,
        output: AgentLoopOutput,
        ground_truth: str,
        judge_fn: Callable[[str, str], float],
    ) -> dict[str, float]:
        """Return the four grouped reward dimensions for one rollout.

        A convenience rollup over :meth:`reward_components`: ``correctness``,
        ``citation_support``, ``retrieval_quality``, ``search_efficiency``
        (pre-scale; they sum to ``terminal_reward + shaping_total``).

        Note: this calls :meth:`reward_components`, which invokes ``judge_fn``.
        If you already have a components dict (or want to avoid a second,
        possibly expensive, judge call), use
        :func:`group_reward_components` on it directly instead.
        """
        components = self.reward_components(output, ground_truth, judge_fn)
        return group_reward_components(components)

    def _reward_components_from_correctness(
        self,
        output: AgentLoopOutput,
        correctness: float,
        *,
        human_signal: float | None = None,
    ) -> dict[str, float]:
        """Compute reward breakdown with a pre-scored correctness value.

        Separating the judge call from the arithmetic lets callers batch all
        judge calls upfront (e.g. using a single LLM API request for N rollouts)
        and then compute the full breakdown per sample without re-calling the judge.
        """
        cfg = self.config
        answer = output.final_answer or ""
        metrics = output.metrics
        ctx: AgentContext | None = output.context

        # Compute each dimension's per-term components once. The four helpers
        # partition the shaping terms exactly as REWARD_DIMENSIONS declares, so
        # shaping_total is derived from the grouped subtotals below rather than
        # re-listing every term.
        components: dict[str, float] = {}
        components.update(self._correctness_component(correctness))
        components.update(self._citation_components(answer, ctx, metrics))
        components.update(self._retrieval_components(metrics))
        components.update(self._efficiency_components(metrics))

        dims = group_reward_components(components)
        terminal_reward = dims["correctness"]
        shaping_total = (
            dims["citation_support"]
            + dims["retrieval_quality"]
            + dims["search_efficiency"]
        )
        total = self._aggregate_total_reward(terminal_reward, shaping_total)

        human_feedback = (
            cfg.human_feedback_weight * human_signal
            if human_signal is not None and cfg.human_feedback_weight != 0.0
            else None
        )
        if human_feedback is not None:
            total += human_feedback

        components["reward_mode"] = cfg.reward_mode
        components["terminal_reward"] = terminal_reward
        components["shaping_total"] = shaping_total
        components["total"] = total
        if human_feedback is not None:
            components["human_feedback"] = human_feedback
        components.update({f"dim_{name}": value for name, value in dims.items()})
        return components

    # ------------------------------------------------------------------
    # Per-dimension component helpers (partition the reward_components terms
    # exactly as REWARD_DIMENSIONS declares).
    # ------------------------------------------------------------------

    def _correctness_component(self, correctness: float) -> dict[str, float]:
        """Correctness dimension: the terminal judge-scored reward."""
        return {"correctness": self.config.correctness_weight * correctness}

    _ZERO_CITATION_COMPONENTS: dict[str, float] = {
        "citation_support": 0.0,
        "unsupported_claim_penalty": 0.0,
        "fetch_usefulness_reward": 0.0,
        "format_reward": 0.0,
    }

    def _citation_components(
        self,
        answer: str,
        ctx: AgentContext | None,
        metrics: dict[str, float],
    ) -> dict[str, float]:
        """Citation-support dimension: grounding of claims in retrieved evidence.

        The three grounding terms all ask the same question of the answer text —
        which retrieved results does it cite? — so the citation keys are
        resolved once here and shared, rather than re-parsed by each term.
        """
        cfg = self.config
        cites_weighted = (
            cfg.citation_support_weight != 0.0
            or cfg.fetch_usefulness_reward != 0.0
            or cfg.unsupported_claim_penalty != 0.0
        )
        if not cites_weighted and cfg.format_reward_weight == 0.0:
            # Every term in this dimension is switched off; resolving citations
            # could only produce zeros.
            return dict(self._ZERO_CITATION_COMPONENTS)

        cited_ids: frozenset[str] = frozenset()
        num_results = 0
        if cites_weighted and answer and ctx is not None:
            num_results = ctx.num_results
            if num_results:
                cited_ids = ctx.cited_result_ids(answer)

        return {
            # Fraction of retrieved docs cited in the answer.
            "citation_support": (
                cfg.citation_support_weight * (len(cited_ids) / num_results)
                if cfg.citation_support_weight != 0.0 and num_results
                else 0.0
            ),
            # Fires when the agent searched, got results, and cited none.
            "unsupported_claim_penalty": (
                cfg.unsupported_claim_penalty
                if (
                    cfg.unsupported_claim_penalty != 0.0
                    and num_results
                    and not cited_ids
                    and metrics.get("rounds_used", 0) > 0
                )
                else 0.0
            ),
            # Fetched pages help only when their URLs end up cited.
            "fetch_usefulness_reward": (
                self._fetch_usefulness_reward(answer, ctx, cited_ids)
                if cfg.fetch_usefulness_reward != 0.0
                else 0.0
            ),
            # Reward for inline citation labels in the answer.
            "format_reward": (
                cfg.format_reward_weight * format_compliance_reward(answer)
                if cfg.format_reward_weight != 0.0 and answer
                else 0.0
            ),
        }

    _ZERO_RETRIEVAL_COMPONENTS: dict[str, float] = {
        "search_quality": 0.0,
        "subquestion_coverage": 0.0,
        "evidence_gain": 0.0,
        "early_stop_bonus": 0.0,
        "answer_when_evidence_insufficient_penalty": 0.0,
        "forced_final_answer_penalty": 0.0,
        "search_budget_exhausted_without_answer_penalty": 0.0,
    }

    def _retrieval_components(self, metrics: dict[str, float]) -> dict[str, float]:
        """Retrieval-quality dimension: sufficiency of the gathered evidence."""
        cfg = self.config
        # Ordered so a shaped config exits on its first weighted term; a fully
        # zeroed preset falls through to the constant and skips the arithmetic.
        if (
            cfg.search_quality_weight == 0.0
            and cfg.subquestion_coverage_weight == 0.0
            and cfg.answer_when_evidence_insufficient_penalty == 0.0
            and cfg.forced_final_answer_penalty == 0.0
            and cfg.search_budget_exhausted_without_answer_penalty == 0.0
            and cfg.evidence_gain_weight == 0.0
            and cfg.early_stop_bonus == 0.0
        ):
            return dict(self._ZERO_RETRIEVAL_COMPONENTS)
        return {
            # Evaluator sufficiency verdict blended with per-round query quality.
            "search_quality": cfg.search_quality_weight * self._search_quality(metrics),
            # Fraction of subquestions with sufficient evidence.
            "subquestion_coverage": cfg.subquestion_coverage_weight
            * metrics.get("subquestion_coverage_ratio", 1.0),
            # Reward improvement in evidence_score across rounds.
            "evidence_gain": cfg.evidence_gain_weight
            * metrics.get("evidence_gain_total", 0.0),
            # Reward rounds flagged as an evidence plateau (opt-in metric).
            "early_stop_bonus": cfg.early_stop_bonus * metrics.get("early_stops", 0.0),
            # Penalty for answering despite insufficient evidence.
            "answer_when_evidence_insufficient_penalty": (
                cfg.answer_when_evidence_insufficient_penalty
                * metrics.get("answer_when_evidence_insufficient", 0.0)
            ),
            # Milder penalty for a forced best-effort answer at a dead-end/cap.
            "forced_final_answer_penalty": cfg.forced_final_answer_penalty
            * metrics.get("forced_final_answer", 0.0),
            # Penalty for exhausting the budget without producing an answer.
            "search_budget_exhausted_without_answer_penalty": (
                cfg.search_budget_exhausted_without_answer_penalty
                * metrics.get("search_budget_exhausted_without_answer", 0.0)
            ),
        }

    _ZERO_EFFICIENCY_COMPONENTS: dict[str, float] = {
        "per_search_penalty": 0.0,
        "unnecessary_search_penalty": 0.0,
        "duplicate_query_penalty": 0.0,
        "budget_penalty": 0.0,
        "unnecessary_fetch_penalty": 0.0,
        "retriever_cost": 0.0,
        "rerank_cost": 0.0,
    }

    def _efficiency_components(self, metrics: dict[str, float]) -> dict[str, float]:
        """Search-efficiency dimension: cost of the search/fetch/rerank actions."""
        cfg = self.config
        if (
            cfg.unnecessary_search_penalty == 0.0
            and cfg.duplicate_query_penalty == 0.0
            and cfg.budget_penalty == 0.0
            and cfg.unnecessary_fetch_penalty == 0.0
            and cfg.per_search_penalty == 0.0
            and cfg.retriever_cost_web == 0.0
            and cfg.retriever_cost_vdb == 0.0
            and cfg.rerank_cost == 0.0
        ):
            return dict(self._ZERO_EFFICIENCY_COMPONENTS)
        rounds_used = int(metrics.get("rounds_used", 0))
        budget_fraction = rounds_used / max(cfg.max_search_rounds, 1)
        budget_pen = (
            cfg.budget_penalty
            if budget_fraction >= cfg.budget_penalty_threshold
            else 0.0
        )
        return {
            # Each search round costs a little.
            "per_search_penalty": cfg.per_search_penalty * rounds_used,
            # Each round beyond the first costs more.
            "unnecessary_search_penalty": cfg.unnecessary_search_penalty
            * max(0, rounds_used - 1),
            # Penalty per repeated query across turns.
            "duplicate_query_penalty": cfg.duplicate_query_penalty
            * metrics.get("repeated_search_queries", 0.0),
            # Fired once when rounds consumed exceed the threshold.
            "budget_penalty": budget_pen,
            # Penalty per fetched page that does not contribute cited evidence.
            "unnecessary_fetch_penalty": cfg.unnecessary_fetch_penalty
            * metrics.get("unnecessary_fetch_count", 0.0),
            # Per-round cost differentiated by backend (web priced above vdb).
            "retriever_cost": cfg.retriever_cost_web * metrics.get("web_searches", 0.0)
            + cfg.retriever_cost_vdb * metrics.get("vdb_searches", 0.0),
            # Flat charge per rerank action invoked.
            "rerank_cost": cfg.rerank_cost * metrics.get("rerank_calls", 0.0),
        }

    # ------------------------------------------------------------------
    # GRPO advantage computation
    # ------------------------------------------------------------------

    def assign_grpo_outcome_token_advantages(
        self,
        outputs: list[AgentLoopOutput],
        ground_truths: list[str],
        judge_fn: JudgeFn,
        group_ids: list[str],
        *,
        batch_judge_fn: BatchJudgeFn | None = None,
    ) -> list[list[float]]:
        """End-to-end GRPO outcome advantages as sparse token-level vectors.

        Full pipeline:

            1. Terminal reward per rollout (sparse — no process shaping):
                   r_i = correctness_weight * judge_fn(answer_i, gt_i)

            2. Group-relative centering (no value model):
                   A_i = r_i - mean({r_j : group_j == group_i})

            3. Std normalization within the group:
                   A_i = A_i / (std(group) + ε)

            4. Sparse token assignment — place A_i at the last
               action token (last response_mask == 1 position) and 0
               everywhere else.  All search, reasoning, and retrieval
               tokens carry no gradient signal.

        This is the DeepSeek-R1 / GRPO training loop in one call:
        no separate value model, no token-level critic, no cross-prompt mixing.

        Args:
            outputs: One AgentLoopOutput per rollout.
            ground_truths: Reference answer for each rollout.
            judge_fn: ``(answer, ground_truth) -> float`` in ``[0, 1]``.
            group_ids: Prompt-group identifier.  Rollouts sharing an ID were
                sampled from the same prompt and are normalised together.
            batch_judge_fn: Optional batch variant
                ``(list[answers], list[ground_truths]) -> list[float]``.
                Supply this for LLM judges to score all N rollouts in a
                single API call instead of N separate calls.

        Returns:
            One token-level advantage vector per rollout, aligned with outputs.
            Each vector has length ``len(output.response_ids)`` and is all-zero
            except at the last action token.
        """
        if len(outputs) != len(ground_truths) or len(outputs) != len(group_ids):
            raise ValueError(
                "outputs, ground_truths, and group_ids must have the same length."
            )

        answers = [output.final_answer or "" for output in outputs]
        scores = _score_answers(
            judge_fn, answers, ground_truths, batch_judge_fn=batch_judge_fn
        )
        rewards = [
            float(self.config.correctness_weight) * score
            if output.final_answer
            else 0.0
            for output, score in zip(outputs, scores)
        ]
        scalar_advantages = self.compute_batch_advantages(rewards, group_ids)

        result: list[list[float]] = []
        for output, adv in zip(outputs, scalar_advantages):
            n = len(output.response_ids)
            if n == 0:
                result.append([])
                continue
            token_advs = [0.0] * n
            token_advs[self._last_action_idx(output)] = adv
            result.append(token_advs)
        return result

    def compute_grpo_outcome_advantages(
        self,
        rewards: list[float],
        group_ids: list[str],
    ) -> list[float]:
        """Compute GRPO outcome advantages without a critic value model.

        For each prompt group, advantage is centered by the mean reward of the
        trajectories sampled from that same prompt:

            advantage_i = reward_i - mean(group_rewards)

        This is the core outcome-based GRPO signal: no separate value model, no
        token-level critic targets, and no cross-prompt mixing.
        Single-sample groups get advantage 0.0 because there is no relative
        within-group comparison signal.  The arithmetic itself lives in
        :func:`grouped_relative_advantages`, the one kernel every advantage
        path in the repo shares.
        """
        return grouped_relative_advantages(rewards, group_ids)

    def compute_batch_advantages(
        self,
        rewards: list[float],
        group_ids: list[str],
    ) -> list[float]:
        """Compute std-normalized GRPO advantages within each prompt group.

        For each prompt group:

            advantage_i = (reward_i - mean(group)) / (std(group) + ε)

        Groups with a single sample get advantage 0.0 (no within-group signal).
        Variance uses the population formula (N denominator).  If your trainer
        uses sample variance (N-1), normalise at that layer instead.
        """
        return grouped_relative_advantages(rewards, group_ids, normalize=True)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _last_action_idx(output: AgentLoopOutput) -> int:
        """Index of the last model-generated token in the response sequence.

        Returns the last position where response_mask == 1 (truthy).
        Falls back to ``len(response_ids) - 1`` when the mask is absent or
        all-zero (e.g. padding-only or observation-only sequences).
        """
        n = len(output.response_ids)
        mask = output.response_mask
        return next(
            (i for i in range(n - 1, -1, -1) if i < len(mask) and mask[i]),
            n - 1,
        )

    @staticmethod
    def _search_quality(metrics: dict[str, float]) -> float:
        """Blend final evidence sufficiency with average per-round query quality."""
        final_sufficient = metrics.get("final_evidence_sufficient", 0.0)
        avg_quality = metrics.get("search_quality_score", 0.0)
        search_rounds = metrics.get("search_rounds", 0.0)

        if search_rounds <= 0:
            return metrics.get("answer_allowed", 0.0)
        return (final_sufficient + avg_quality) / 2.0

    def _fetch_usefulness_reward(
        self,
        answer: str,
        ctx: AgentContext | None,
        cited_ids: frozenset[str],
    ) -> float:
        """Reward fetches only when fetched URLs are actually used in cited evidence.

        *cited_ids* is the already-resolved citation set for *answer*.  An answer
        that cites nothing cannot cite a fetched page, so the second context
        traversal is skipped entirely in that case.
        """
        if not answer or not cited_ids or ctx is None or not ctx.fetched_pages:
            return 0.0

        fetched_urls = {page.url for page in ctx.fetched_pages if page.url}
        if not fetched_urls:
            return 0.0

        cited_urls = {result.url for result in ctx.cited_results(answer) if result.url}
        if cited_urls & fetched_urls:
            return self.config.fetch_usefulness_reward
        return 0.0

    def _aggregate_total_reward(
        self,
        terminal_reward: float,
        shaping_total: float,
    ) -> float:
        """Combine terminal reward and shaping according to the configured mode.

        ``reward_scale`` is applied to the combined total so callers can align
        reward magnitudes across training phases without adjusting every weight.
        """
        mode = self.config.reward_mode
        if mode == "shaped":
            raw = terminal_reward + shaping_total
        elif mode == "sparse_final_only":
            raw = terminal_reward
        else:
            raise ValueError(
                f"Unsupported reward_mode: {mode!r}. "
                "Expected 'shaped' or 'sparse_final_only'."
            )
        return raw * float(self.config.reward_scale)
