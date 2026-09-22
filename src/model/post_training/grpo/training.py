"""Consolidated GRPO trainers, local controller, and durable train loop."""

from __future__ import annotations

import asyncio
import copy
import json
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
import torch.nn as nn
import torch.optim as optim

from src.agents.core.base import AgentLoopBase

from ..log_probs import get_response_log_probs
from ..reward import group_relative_advantages
from ..ppo.core_algos import (
    PPOPolicyLossConfig,
    compute_ppo_policy_loss_core,
    kl_penalty,
    masked_mean,
)
from .algorithms import (
    GRPOAdvantageConfig,
    sample_prompt_group,
    score_prompt_group,
)
from .core_algos import compute_grpo_token_advantages
from .generation import (
    GRPORolloutSafetyConfig,
    GRPOTrainingStepResult,
    LogProbCapable,
    _async_collect_grpo_rollouts_core,
    _async_run_grpo_training_step_core,
    _collect_grpo_rollouts_core,
    _run_grpo_training_step_core,
)

if TYPE_CHECKING:
    from src.model.post_training.reward import JudgeFn, SearchRewardFunction

    from .generation import LLMGenerationManager

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tensor utilities (2-D grouped variants — different from the scalar
# masked_mean in core_algos.py which reduces over all positions)
# ---------------------------------------------------------------------------


def _group_masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Row-wise masked mean. Returns shape ``(batch, 1)``."""
    mask = mask.float()
    denom = mask.sum(dim=1, keepdim=True).clamp_min(1e-8)
    return (values * mask).sum(dim=1, keepdim=True) / denom


def compute_group_advantages(
    rewards: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """GRPO group advantage: normalise rewards within each prompt group.

    Args:
        rewards: ``(batch_size, group_size)`` — one scalar reward per rollout.
        mask: ``(batch_size, group_size)`` — 1 for valid rollouts, 0 for padding.

    Returns:
        ``(batch_size, group_size)`` advantages detached from the graph.
    """
    if rewards.ndim != 2 or mask.ndim != 2:
        raise ValueError("rewards and mask must both be 2-D grouped tensors")
    if rewards.shape != mask.shape:
        raise ValueError("rewards and mask must have the same shape")
    if rewards.shape[0] == 0 or rewards.shape[1] == 0:
        raise ValueError("rewards and mask must have non-empty group dimensions")

    mask = mask.to(device=rewards.device, dtype=rewards.dtype)
    if torch.any(mask.sum(dim=1) <= 0):
        raise ValueError("each group must contain at least one valid rollout")

    group_mean = _group_masked_mean(rewards, mask)
    centered = (rewards - group_mean) * mask
    group_var = _group_masked_mean(centered.pow(2), mask)
    group_std = torch.sqrt(group_var + 1e-8)
    return (centered / group_std).detach()


def grpo_clipped_policy_loss(
    new_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    clip_epsilon: float = 0.2,
) -> tuple[torch.Tensor, torch.Tensor]:
    """PPO-style clipped objective for a flat batch of rollouts.

    Returns:
        ``(per_token_loss, ratios)`` — caller applies mask and reduces.
    """
    ratios = torch.exp(new_log_probs - old_log_probs)
    clipped = torch.clamp(ratios, 1.0 - clip_epsilon, 1.0 + clip_epsilon)
    loss = -torch.min(ratios * advantages, clipped * advantages)
    return loss, ratios


def reverse_kl_penalty(
    policy_log_probs: torch.Tensor,
    ref_log_probs: torch.Tensor,
) -> torch.Tensor:
    """Sample-based reverse-KL penalty: ``exp(Δlog p) - Δlog p - 1``."""
    log_ratio = policy_log_probs - ref_log_probs
    return torch.exp(log_ratio) - log_ratio - 1.0


# ---------------------------------------------------------------------------
# Policy network
# ---------------------------------------------------------------------------


class Policy(nn.Module):
    """Simple MLP policy for discrete action spaces."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 64,
        lr: float = 1e-3,
    ) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )
        self.optimizer = optim.Adam(self.parameters(), lr=lr)

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        return self.network(states)

    def get_distribution(self, states: torch.Tensor) -> torch.distributions.Categorical:
        return torch.distributions.Categorical(logits=self.forward(states))

    def get_log_probs(
        self, states: torch.Tensor, actions: torch.Tensor
    ) -> torch.Tensor:
        return self.get_distribution(states).log_prob(actions)


# ---------------------------------------------------------------------------
# GRPO Trainer
# ---------------------------------------------------------------------------


class GRPOTrainer:
    """Self-contained GRPO update loop for grouped rollout environments.

    Args:
        policy: Trainable policy.
        reference_policy: Frozen anchor for the KL penalty.
        clip_epsilon: PPO clip range.
        beta: KL penalty coefficient (``0`` disables the penalty).
        grad_clip: Max-norm for gradient clipping.
    """

    def __init__(
        self,
        policy: Policy,
        reference_policy: Policy,
        clip_epsilon: float = 0.2,
        beta: float = 0.1,
        grad_clip: float = 1.0,
    ) -> None:
        self.policy = policy
        self.reference_policy = reference_policy
        self.clip_epsilon = clip_epsilon
        self.beta = beta
        self.grad_clip = grad_clip

        self.reference_policy.eval()
        for param in self.reference_policy.parameters():
            param.requires_grad = False

    def compute_loss(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        old_log_probs: torch.Tensor,
        rewards: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute the masked GRPO loss with KL penalty.

        Args:
            states: ``(batch_size * group_size, state_dim)``
            actions: ``(batch_size * group_size,)``
            old_log_probs: ``(batch_size * group_size,)``
            rewards: ``(batch_size, group_size)``
            mask: ``(batch_size, group_size)``

        Returns:
            ``(mean_loss, metrics_dict)``
        """
        expected_rollouts = rewards.numel()
        if (
            states.shape[0] != expected_rollouts
            or actions.shape != (expected_rollouts,)
            or old_log_probs.shape != (expected_rollouts,)
        ):
            raise ValueError(
                "states, actions, and old_log_probs must contain "
                "batch_size * group_size rollouts"
            )

        flat_mask = mask.reshape(-1).to(device=rewards.device, dtype=rewards.dtype)
        advantages = compute_group_advantages(rewards, mask).reshape(-1)

        new_log_probs = self.policy.get_log_probs(states, actions)

        policy_loss, ratios = grpo_clipped_policy_loss(
            new_log_probs=new_log_probs,
            old_log_probs=old_log_probs,
            advantages=advantages,
            clip_epsilon=self.clip_epsilon,
        )
        if self.beta == 0:
            kl = torch.zeros_like(new_log_probs)
        else:
            with torch.no_grad():
                ref_log_probs = self.reference_policy.get_log_probs(states, actions)
            kl = reverse_kl_penalty(new_log_probs, ref_log_probs)
        total = policy_loss + self.beta * kl
        normalizer = flat_mask.sum().clamp_min(1e-8)
        mean_loss = (total * flat_mask).sum() / normalizer

        with torch.no_grad():
            metrics: dict[str, float] = {
                "loss": mean_loss.item(),
                "mean_reward": float(
                    (rewards.reshape(-1) * flat_mask).sum() / normalizer
                ),
                "mean_advantage": float((advantages * flat_mask).sum() / normalizer),
                "mean_ratio": float((ratios.detach() * flat_mask).sum() / normalizer),
                "mean_kl": float((kl.detach() * flat_mask).sum() / normalizer),
            }
        return mean_loss, metrics

    def update(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        old_log_probs: torch.Tensor,
        rewards: torch.Tensor,
        mask: torch.Tensor,
    ) -> dict[str, float]:
        """Run one gradient step; returns the metrics dict."""
        loss, metrics = self.compute_loss(states, actions, old_log_probs, rewards, mask)
        self.policy.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.grad_clip)
        self.policy.optimizer.step()
        return metrics


def make_grpo_trainer(
    state_dim: int,
    action_dim: int,
    hidden_dim: int = 64,
    lr: float = 1e-3,
    clip_epsilon: float = 0.2,
    beta: float = 0.1,
    grad_clip: float = 1.0,
) -> GRPOTrainer:
    """Convenience factory that builds policy + reference and returns a trainer."""
    policy = Policy(state_dim, action_dim, hidden_dim=hidden_dim, lr=lr)
    reference_policy = copy.deepcopy(policy)
    return GRPOTrainer(
        policy=policy,
        reference_policy=reference_policy,
        clip_epsilon=clip_epsilon,
        beta=beta,
        grad_clip=grad_clip,
    )


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LLMGRPOConfig:
    """Hyperparameters for one online GRPO training step."""

    num_rollouts: int = 4
    max_new_tokens: int = 256
    temperature: float = 0.7
    temperature_step: float = 0.1
    clip_epsilon: float = 0.2
    kl_beta: float = 0.04
    grad_clip: float = 1.0
    kl_penalty_type: str = "low_var_kl"
    advantage_epsilon: float = 1e-6
    advantage_clip: float | None = None


# ---------------------------------------------------------------------------
# Rollout output
# ---------------------------------------------------------------------------


@dataclass
class LLMRolloutResult:
    """All per-group data produced by one rollout step.

    Shape convention (B = batch_size, G = num_rollouts, T = response_len):

        prompt_ids     (B, P)
        response_ids   (B*G, T)  — flattened; group i occupies rows [i*G : (i+1)*G]
        response_mask  (B*G, T)  — 1 for non-pad response tokens
        old_log_probs  (B*G, T)  — log π_old(token | context) at each response position
        rewards        (B*G,)    — scalar reward per rollout
        advantages     (B*G, T)  — token-expanded group-relative advantage
        group_ids      (B*G,)    — integer group index (same for G rows in a group)
    """

    prompt_ids: torch.Tensor
    response_ids: torch.Tensor
    response_mask: torch.Tensor
    old_log_probs: torch.Tensor
    rewards: torch.Tensor
    advantages: torch.Tensor
    group_ids: torch.Tensor
    metadata: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Rollout builder
# ---------------------------------------------------------------------------


def _build_response_mask(response_ids: torch.Tensor, pad_token_id: int) -> torch.Tensor:
    """1 for every non-pad response token, 0 otherwise."""
    return (response_ids != pad_token_id).long()


def _sparse_reward_at_eos(
    rewards: torch.Tensor,
    response_mask: torch.Tensor,
) -> torch.Tensor:
    """Place each scalar reward at the last valid response token (EOS position).

    This is the standard sparse outcome-reward layout for GRPO: the entire
    reward signal lives at the position where the model stopped generating.

    Args:
        rewards: ``(N,)`` scalar reward per rollout.
        response_mask: ``(N, T)`` — 1 for non-pad positions.

    Returns:
        ``(N, T)`` sparse tensor with reward at the last 1-position per row.
    """
    token_rewards = torch.zeros_like(response_mask, dtype=rewards.dtype)
    eos_positions = (response_mask.sum(dim=1) - 1).clamp(min=0)  # last non-pad idx
    for i, pos in enumerate(eos_positions.tolist()):
        token_rewards[i, int(pos)] = rewards[i]
    return token_rewards


# ---------------------------------------------------------------------------
# Online LLM GRPO Trainer
# ---------------------------------------------------------------------------


class LLMGRPOTrainer:
    """HuggingFace-native online GRPO trainer.

    Mirrors the bandit ``GRPOTrainer`` above but operates on token sequences:

        policy         — AutoModelForCausalLM being optimized
        reference      — frozen SFT copy (deepcopy at init)
        judge_fn       — (response_str, ground_truth_str) → float
        reward_fn      — optional SearchRewardFunction for shaped rewards

    Typical usage::

        trainer = LLMGRPOTrainer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct",
                      judge_fn=exact_match, optimizer=optim.AdamW(policy.parameters()))
        metrics = trainer.step(prompts=["What is FAISS?"], ground_truths=["..."])
    """

    def __init__(
        self,
        policy: nn.Module,
        reference_policy: nn.Module,
        tokenizer: Any,
        optimizer: torch.optim.Optimizer,
        judge_fn: JudgeFn,
        config: LLMGRPOConfig | None = None,
        reward_fn: SearchRewardFunction | None = None,
        device: str | torch.device | None = None,
    ) -> None:
        self.policy = policy
        self.reference = reference_policy
        self.tokenizer = tokenizer
        self.optimizer = optimizer
        self.judge_fn = judge_fn
        self.reward_fn = reward_fn
        self.config = config or LLMGRPOConfig()
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )

        # Freeze reference; it is never updated.
        self.reference.eval()
        for p in self.reference.parameters():
            p.requires_grad = False

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        judge_fn: JudgeFn,
        optimizer_cls: type = torch.optim.AdamW,
        lr: float = 1e-5,
        config: LLMGRPOConfig | None = None,
        reward_fn: SearchRewardFunction | None = None,
        device: str | None = None,
        **hf_kwargs: Any,
    ) -> LLMGRPOTrainer:
        """Load policy + tokenizer from *model_name_or_path* and build trainer."""
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, **hf_kwargs)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id

        policy = AutoModelForCausalLM.from_pretrained(model_name_or_path, **hf_kwargs)
        reference = copy.deepcopy(policy)

        resolved_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        policy = policy.to(resolved_device)
        reference = reference.to(resolved_device)

        optimizer = optimizer_cls(policy.parameters(), lr=lr)
        return cls(
            policy=policy,
            reference_policy=reference,
            tokenizer=tokenizer,
            optimizer=optimizer,
            judge_fn=judge_fn,
            config=config,
            reward_fn=reward_fn,
            device=resolved_device,
        )

    # ------------------------------------------------------------------
    # Rollout  (demo: generate_dummy_group_data)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def rollout(
        self,
        prompts: list[str],
        ground_truths: list[str],
    ) -> LLMRolloutResult:
        """Sample G responses per prompt and compute rewards and advantages.

        Analogous to ``generate_dummy_group_data`` in the bandit demo, but for
        token sequences.  All G rollouts per prompt share the same prompt ids;
        group membership is tracked via ``group_ids``.

        Args:
            prompts: List of B raw text prompts.
            ground_truths: List of B gold answers aligned with *prompts*.

        Returns:
            :class:`LLMRolloutResult` with shape ``(B*G, ...)`` tensors.
        """
        cfg = self.config
        B = len(prompts)
        G = cfg.num_rollouts
        pad_id = self.tokenizer.pad_token_id or 0

        # ── 1. Tokenize prompts ───────────────────────────────────────────
        enc = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        prompt_ids: torch.Tensor = enc["input_ids"].to(self.device)  # (B, P)
        prompt_len = prompt_ids.shape[1]

        # Tile: (B, P) → (B*G, P) — each prompt repeated G times.
        tiled_prompt = prompt_ids.repeat_interleave(G, dim=0)  # (B*G, P)
        tiled_attention = (
            enc["attention_mask"].to(self.device).repeat_interleave(G, dim=0)
        )

        # ── 2. Generate G completions per prompt with varied temperature ──
        all_responses: list[torch.Tensor] = []
        for rollout_idx in range(G):
            temp = cfg.temperature + rollout_idx * cfg.temperature_step
            # Select rows for this rollout index: 0, G, 2G, ... (B rows)
            rows = tiled_prompt[rollout_idx::G]
            attn = tiled_attention[rollout_idx::G]
            out = self.policy.generate(
                input_ids=rows,
                attention_mask=attn,
                do_sample=True,
                temperature=temp,
                max_new_tokens=cfg.max_new_tokens,
                pad_token_id=pad_id,
            )
            # Strip prompt prefix → pure response tokens
            response = out[:, prompt_len:]  # (B, T_i)
            all_responses.append(response)

        # Pad all rollout responses to a common length T.
        max_resp_len = max(r.shape[1] for r in all_responses)

        def _pad(t: torch.Tensor) -> torch.Tensor:
            pad = max_resp_len - t.shape[1]
            if pad > 0:
                t = torch.cat(
                    [t, torch.full((t.shape[0], pad), pad_id, device=t.device)], dim=1
                )
            return t

        # Interleave back to (B*G, T): row order [prompt_0_roll_0, prompt_0_roll_1, ...]
        # i.e. group 0 → rows 0..G-1, group 1 → rows G..2G-1, ...
        padded = [_pad(r) for r in all_responses]  # G tensors of (B, T)
        # Stack to (G, B, T) then transpose to (B, G, T) then reshape to (B*G, T)
        response_ids = (
            torch.stack(padded, dim=0)  # (G, B, T)
            .permute(1, 0, 2)  # (B, G, T)
            .reshape(B * G, max_resp_len)  # (B*G, T)
        )

        response_mask = _build_response_mask(response_ids, pad_id)  # (B*G, T)

        # ── 3. Score responses (demo: reward_function) ────────────────────
        decoded = self.tokenizer.batch_decode(response_ids, skip_special_tokens=True)

        # Align ground truths: ground_truths[i] applies to rows [i*G : (i+1)*G]
        rewards_list: list[float] = []
        for flat_idx, response_str in enumerate(decoded):
            prompt_idx = flat_idx // G
            gt = ground_truths[prompt_idx]
            if self.reward_fn is not None:
                from src.agents.core.base import AgentLoopOutput

                pseudo_output = AgentLoopOutput(
                    prompt_ids=[],
                    response_ids=[],
                    response_mask=[],
                    num_turns=1,
                    metrics={},
                    final_answer=response_str,
                )
                components = self.reward_fn.reward_components(
                    pseudo_output,
                    ground_truth=gt,
                    judge_fn=self.judge_fn,
                )
                rewards_list.append(float(components["total"]))
            else:
                rewards_list.append(float(self.judge_fn(response_str, gt)))

        rewards = torch.tensor(rewards_list, dtype=torch.float32, device=self.device)

        # ── 4. Group advantage (demo: compute_group_advantages) ───────────
        # Group IDs: rollouts from prompt i all share group id i.
        group_ids = torch.arange(B, device=self.device).repeat_interleave(G)

        # Sparse outcome reward: scalar reward at EOS token position.
        token_rewards = _sparse_reward_at_eos(rewards, response_mask.cpu()).to(
            self.device
        )

        advantages, _ = compute_grpo_token_advantages(
            token_rewards,
            response_mask,
            group_ids,
            epsilon=cfg.advantage_epsilon,
            clip_advantages=cfg.advantage_clip,
        )  # (B*G, T)

        # ── 5. Old log-probs under the rollout policy (frozen snapshot) ───
        full_ids = torch.cat([tiled_prompt, response_ids], dim=1)  # (B*G, P+T)
        old_log_probs = get_response_log_probs(
            self.policy, full_ids, prompt_len, response_mask
        )  # (B*G, T)

        return LLMRolloutResult(
            prompt_ids=prompt_ids,
            response_ids=response_ids,
            response_mask=response_mask,
            old_log_probs=old_log_probs,
            rewards=rewards,
            advantages=advantages,
            group_ids=group_ids,
        )

    # ------------------------------------------------------------------
    # Loss  (demo: GRPOTrainer.compute_loss)
    # ------------------------------------------------------------------

    def compute_loss(
        self,
        rollout: LLMRolloutResult,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute GRPO clip loss + KL penalty from a rollout result.

        Analogous to ``GRPOTrainer.compute_loss`` but operating on the
        token-level log-prob tensors in *rollout*.

        Steps:
            new_log_probs  = log π_θ(token | context)             current policy
            ref_log_probs  = log π_ref(token | context)           frozen SFT
            pg_loss        = clipped surrogate (old vs new ratio × advantage)
            kl             = low_var_kl(new ‖ ref) averaged over response tokens
            total          = pg_loss + β · kl
        """
        cfg = self.config
        tiled_prompt = rollout.prompt_ids.repeat_interleave(cfg.num_rollouts, dim=0)
        prompt_len = tiled_prompt.shape[1]
        full_ids = torch.cat([tiled_prompt, rollout.response_ids], dim=1)
        mask = rollout.response_mask

        # Current policy: new_log_probs
        new_log_probs = get_response_log_probs(self.policy, full_ids, prompt_len, mask)

        # Reference policy: ref_log_probs. Frozen, so no grad is recorded.
        # `torch.inference_mode()` was measured here and is deliberately not
        # used: it is faster only on toy models and was repeatably 7-12% slower
        # at realistic hidden/vocab sizes. See docs/benchmarks/.
        with torch.no_grad():
            ref_log_probs = get_response_log_probs(
                self.reference, full_ids, prompt_len, mask
            )

        # PPO-clipped policy loss (demo: grpo_clipped_policy_loss)
        pg_loss, clip_frac, approx_kl, _ = compute_ppo_policy_loss_core(
            old_log_prob=rollout.old_log_probs,
            log_prob=new_log_probs,
            advantages=rollout.advantages,
            eos_mask=mask,
            cliprange=cfg.clip_epsilon,
        )

        # KL penalty vs SFT reference (demo: reverse_kl_penalty → here: low_var_kl)
        kl_per_token = kl_penalty(new_log_probs, ref_log_probs, cfg.kl_penalty_type)
        mean_kl = masked_mean(kl_per_token, mask)
        total_loss = pg_loss + cfg.kl_beta * mean_kl

        with torch.no_grad():
            normalizer = mask.float().sum().clamp_min(1e-8)
            metrics: dict[str, float] = {
                "loss": total_loss.item(),
                "pg_loss": pg_loss.item(),
                "mean_kl": mean_kl.item(),
                "mean_reward": (rollout.rewards.sum() / len(rollout.rewards)).item(),
                "clip_fraction": clip_frac.item(),
                "approx_kl": approx_kl.item(),
                "mean_advantage": (
                    (rollout.advantages * mask.float()).sum() / normalizer
                ).item(),
            }
        return total_loss, metrics

    # ------------------------------------------------------------------
    # Full update step  (demo: GRPOTrainer.update)
    # ------------------------------------------------------------------

    def step(
        self,
        prompts: list[str],
        ground_truths: list[str],
    ) -> dict[str, float]:
        """One complete online GRPO step: rollout → loss → gradient update.

        Analogous to ``GRPOTrainer.update`` above.

        Args:
            prompts: B raw text prompts.
            ground_truths: B gold answers.

        Returns:
            Metrics dict with loss, reward, KL, clip_fraction, etc.
        """
        rollout = self.rollout(prompts, ground_truths)

        self.policy.train()
        loss, metrics = self.compute_loss(rollout)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.config.grad_clip)
        self.optimizer.step()

        return metrics

    def save_checkpoint(self, path: str) -> None:
        """Persist policy weights, tokenizer, and optimizer state for resume.

        Used by ``train_loop`` for periodic checkpointing of long runs. The
        reference policy is frozen and reconstructable, so it is not saved.
        """
        import os

        os.makedirs(path, exist_ok=True)
        self.policy.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        torch.save(self.optimizer.state_dict(), os.path.join(path, "optimizer.pt"))

    def load_checkpoint(self, path: str) -> None:
        """Restore policy weights and optimizer state saved by save_checkpoint."""
        import os

        from transformers import AutoModelForCausalLM

        loaded = AutoModelForCausalLM.from_pretrained(path)
        self.policy.load_state_dict(loaded.state_dict())
        self.policy.to(self.device)
        optimizer_path = os.path.join(path, "optimizer.pt")
        if os.path.exists(optimizer_path):
            self.optimizer.load_state_dict(torch.load(optimizer_path))


# Default ceiling on concurrent agent rollouts. Bounding this (rather than
# leaving it unbounded) keeps a batch of B×G live search rollouts from
# saturating / rate-limiting the retrieval server during long runs.
DEFAULT_MAX_CONCURRENT = 8


def _resolve_max_concurrent(value: int | None) -> int:
    """Always return a positive bound; None falls back to the default ceiling."""
    return DEFAULT_MAX_CONCURRENT if value is None else value


class SearchAgentGRPOTrainer(LLMGRPOTrainer):
    """GRPO trainer that samples via live agent loops for full shaped reward.

    Inherits ``compute_loss()`` and ``step()`` from :class:`LLMGRPOTrainer`
    unchanged — only ``rollout()`` is replaced with an async agent-loop version
    that produces real :class:`~src.agents.core.base.AgentLoopOutput` objects.

    Args:
        loop_factory: Zero-argument callable that returns a fresh
            :class:`~src.agents.core.base.AgentLoopBase` instance per rollout.
            Called ``num_rollouts`` times per prompt group concurrently.
        sampling_params: Default sampling parameters forwarded to
            ``AgentLoopBase.run()``.  Individual rollouts receive per-rollout
            variants (temperature sweep) built by ``build_grpo_sampling_params``
            unless ``sampling_variants`` is provided explicitly.
        sampling_variants: If given, must have length ``num_rollouts``; overrides
            the automatic temperature-sweep logic.
        max_concurrent: Maximum number of agent loops running concurrently
            across the whole batch.  ``None`` (default) launches everything at
            once.  Reduce this if the inference server is capacity-limited.
        advantage_config: Controls within-group advantage normalisation passed
            to ``score_prompt_group``.
    """

    def __init__(
        self,
        policy: nn.Module,
        reference_policy: nn.Module,
        tokenizer: Any,
        optimizer: torch.optim.Optimizer,
        judge_fn: JudgeFn,
        loop_factory: Callable[[], AgentLoopBase],
        *,
        config: LLMGRPOConfig | None = None,
        reward_fn: SearchRewardFunction | None = None,
        sampling_params: dict[str, Any] | None = None,
        sampling_variants: list[dict[str, Any]] | None = None,
        max_concurrent: int | None = None,
        advantage_config: GRPOAdvantageConfig | None = None,
        device: str | torch.device | None = None,
    ) -> None:
        super().__init__(
            policy=policy,
            reference_policy=reference_policy,
            tokenizer=tokenizer,
            optimizer=optimizer,
            judge_fn=judge_fn,
            config=config,
            reward_fn=reward_fn,
            device=device,
        )
        self.loop_factory = loop_factory
        self._sampling_params: dict[str, Any] = sampling_params or {
            "temperature": 0.7,
            "max_tokens": 512,
        }
        self._sampling_variants = sampling_variants
        self._max_concurrent = _resolve_max_concurrent(max_concurrent)
        self._advantage_config = advantage_config or GRPOAdvantageConfig()

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        judge_fn: JudgeFn,
        loop_factory: Callable[[], AgentLoopBase],
        *,
        optimizer_cls: type = torch.optim.AdamW,
        lr: float = 1e-5,
        config: LLMGRPOConfig | None = None,
        reward_fn: SearchRewardFunction | None = None,
        sampling_params: dict[str, Any] | None = None,
        sampling_variants: list[dict[str, Any]] | None = None,
        max_concurrent: int | None = None,
        advantage_config: GRPOAdvantageConfig | None = None,
        device: str | None = None,
        **hf_kwargs: Any,
    ) -> SearchAgentGRPOTrainer:
        """Load policy + tokenizer from *model_name_or_path* and build trainer."""
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, **hf_kwargs)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id

        policy = AutoModelForCausalLM.from_pretrained(model_name_or_path, **hf_kwargs)
        reference = copy.deepcopy(policy)

        resolved_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        policy = policy.to(resolved_device)
        reference = reference.to(resolved_device)

        optimizer = optimizer_cls(policy.parameters(), lr=lr)
        return cls(
            policy=policy,
            reference_policy=reference,
            tokenizer=tokenizer,
            optimizer=optimizer,
            judge_fn=judge_fn,
            loop_factory=loop_factory,
            config=config,
            reward_fn=reward_fn,
            sampling_params=sampling_params,
            sampling_variants=sampling_variants,
            max_concurrent=max_concurrent,
            advantage_config=advantage_config,
            device=resolved_device,
        )

    # ------------------------------------------------------------------
    # Rollout  (overrides LLMGRPOTrainer.rollout)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def rollout(
        self,
        prompts: list[str],
        ground_truths: list[str],
        metadata: list[dict] | None = None,
    ) -> LLMRolloutResult:
        """Sync entry point — runs :meth:`rollout_async` in a new event loop."""
        return asyncio.run(
            self.rollout_async(prompts, ground_truths, metadata=metadata)
        )

    async def rollout_async(
        self,
        prompts: list[str],
        ground_truths: list[str],
        metadata: list[dict] | None = None,
    ) -> LLMRolloutResult:
        """Run live agent loops for each prompt and assemble an LLMRolloutResult.

        Each prompt spawns ``num_rollouts`` concurrent :class:`AgentLoopBase`
        executions, each producing a real :class:`~src.agents.core.base.AgentLoopOutput`
        with populated ``metrics`` and ``context``.  These are scored by
        ``score_prompt_group`` using the full :class:`SearchRewardFunction`
        signal (citations, search quality, etc.).

        Args:
            prompts: List of B raw text prompts.
            ground_truths: List of B gold answers.

        Returns:
            :class:`LLMRolloutResult` with ``(B*G, ...)`` tensors ready for
            :meth:`compute_loss`.
        """
        B = len(prompts)
        G = self.config.num_rollouts
        pad_id = (
            self.tokenizer.pad_token_id
            if hasattr(self.tokenizer, "pad_token_id")
            else 0
        ) or 0

        # ── 1. Run G agent loops per prompt (concurrent, always bounded) ──
        semaphore = asyncio.Semaphore(self._max_concurrent)

        async def _sample_one(messages: list[dict[str, Any]]):
            if semaphore is not None:
                async with semaphore:
                    return await sample_prompt_group(
                        self.loop_factory,
                        messages=messages,
                        sampling_params=self._sampling_params,
                        num_rollouts=G,
                        sampling_variants=self._sampling_variants,
                    )
            return await sample_prompt_group(
                self.loop_factory,
                messages=messages,
                sampling_params=self._sampling_params,
                num_rollouts=G,
                sampling_variants=self._sampling_variants,
            )

        batch_messages = [[{"role": "user", "content": p}] for p in prompts]
        grouped_samples = list(
            await asyncio.gather(*[_sample_one(m) for m in batch_messages])
        )

        # ── 2. Score each group with the full reward function ─────────────────
        # score_prompt_group calls reward_fn._reward_components_from_correctness()
        # with the real AgentLoopOutput — citations and metrics are populated.
        rewards_list: list[float] = []
        all_response_ids: list[list[int]] = []
        all_prompt_ids: list[list[int]] = []  # last-turn prompt per rollout

        for i, (group_samples, gt) in enumerate(zip(grouped_samples, ground_truths)):
            group_metadata = metadata[i] if metadata and i < len(metadata) else None
            scored = score_prompt_group(
                group_samples,
                ground_truth=gt,
                judge_fn=self.judge_fn,
                reward_fn=self.reward_fn,
                advantage_config=self._advantage_config,
                metadata=group_metadata,
            )
            for s in scored:
                rewards_list.append(s.reward)
                all_response_ids.append(s.output.response_ids)
                all_prompt_ids.append(s.output.prompt_ids)

        # ── 3. Pad response_ids to uniform length ─────────────────────────────
        max_resp_len = max(len(r) for r in all_response_ids) if all_response_ids else 1

        response_ids = torch.tensor(
            [r + [pad_id] * (max_resp_len - len(r)) for r in all_response_ids],
            dtype=torch.long,
            device=self.device,
        )  # (B*G, T)

        response_mask = torch.tensor(
            [[1] * len(r) + [0] * (max_resp_len - len(r)) for r in all_response_ids],
            dtype=torch.long,
            device=self.device,
        )  # (B*G, T)

        rewards = torch.tensor(
            rewards_list, dtype=torch.float32, device=self.device
        )  # (B*G,)

        # ── 4. Group IDs and sparse token advantages ──────────────────────────
        group_ids = torch.arange(B, device=self.device).repeat_interleave(G)

        token_rewards = _sparse_reward_at_eos(rewards, response_mask.cpu()).to(
            self.device
        )  # (B*G, T)

        advantages, _ = compute_grpo_token_advantages(
            token_rewards,
            response_mask,
            group_ids,
            epsilon=self.config.advantage_epsilon,
            clip_advantages=self.config.advantage_clip,
        )  # (B*G, T)

        # ── 5. Prompt IDs — one per prompt (not per rollout) ─────────────────
        # Use the first rollout's prompt_ids for each prompt as the representative
        # context.  All rollouts for the same prompt share the same initial prompt.
        prompt_ids_per_prompt = [all_prompt_ids[i * G] for i in range(B)]
        max_prompt_len = max(len(p) for p in prompt_ids_per_prompt)

        prompt_ids = torch.tensor(
            [[pad_id] * (max_prompt_len - len(p)) + p for p in prompt_ids_per_prompt],
            dtype=torch.long,
            device=self.device,
        )  # (B, P)

        # ── 6. Old log-probs under current policy (snapshot before update) ────
        tiled_prompt = prompt_ids.repeat_interleave(G, dim=0)  # (B*G, P)
        prompt_len = tiled_prompt.shape[1]
        full_ids = torch.cat([tiled_prompt, response_ids], dim=1)  # (B*G, P+T)

        old_log_probs = get_response_log_probs(
            self.policy, full_ids, prompt_len, response_mask
        )  # (B*G, T)

        return LLMRolloutResult(
            prompt_ids=prompt_ids,
            response_ids=response_ids,
            response_mask=response_mask,
            old_log_probs=old_log_probs,
            rewards=rewards,
            advantages=advantages,
            group_ids=group_ids,
        )

    # ------------------------------------------------------------------
    # Async step convenience  (compute_loss + gradient update)
    # ------------------------------------------------------------------

    async def step_async(
        self,
        prompts: list[str],
        ground_truths: list[str],
        metadata: list[dict] | None = None,
    ) -> dict[str, float]:
        """Async version of :meth:`step` — avoids nested ``asyncio.run()`` calls.

        Use this when the caller already runs inside an event loop (e.g. a
        training loop driven by ``asyncio.run(train())``).
        """
        rollout = await self.rollout_async(prompts, ground_truths, metadata=metadata)

        self.policy.train()
        loss, metrics = self.compute_loss(rollout)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.config.grad_clip)
        self.optimizer.step()

        return metrics


@dataclass
class RolloutResult:
    """Small generic rollout result for local controller experiments."""

    prompt_id: int
    rollout_id: int
    trajectory: Any
    reward: float = 0.0
    advantage: float = 0.0


class LocalGRPOController:
    """Public training-layer facade over local generation step mechanics."""

    def __init__(
        self,
        manager: LLMGenerationManager,
        *,
        num_rollouts: int = 4,
        max_workers: int | None = None,
    ) -> None:
        self.manager = manager
        self.num_rollouts = int(num_rollouts)
        self.max_workers = max_workers

    @staticmethod
    def assign_group_advantages(group: list[RolloutResult]) -> list[RolloutResult]:
        """Assign std-normalized advantages to a simple rollout group."""
        if not group:
            return group
        advantages = group_relative_advantages(
            [float(item.reward) for item in group], normalize=True
        )
        for item, advantage in zip(group, advantages):
            item.advantage = advantage
        return group

    async def async_collect_rollouts(
        self,
        prompt_batch: Any,
        *,
        search_mode: str,
        sampling_params: dict[str, Any],
        judge_fn: Callable[[str, str], float],
        num_rollouts: int | None = None,
        reward_fn: Any = None,
        advantage_config: Any = None,
        batch_judge_fn: Any = None,
        safety_config: Any = None,
        base_seed: int | None = None,
        current_step: int = 0,
        total_steps: int = 1,
    ) -> tuple[list[Any], list[Any]]:
        """Collect all rollouts for every prompt concurrently.

        All ``N_prompts × N_rollouts`` trajectories run in parallel, overlapping
        HTTP search I/O.  Returns ``(group_results, scored_rollouts)``.
        """
        return await _async_collect_grpo_rollouts_core(
            self.manager,
            prompt_batch,
            search_mode=search_mode,
            sampling_params=sampling_params,
            judge_fn=judge_fn,
            num_rollouts=int(num_rollouts or self.num_rollouts),
            reward_fn=reward_fn,
            advantage_config=advantage_config,
            batch_judge_fn=batch_judge_fn,
            safety_config=safety_config,
            base_seed=base_seed,
            current_step=current_step,
            total_steps=total_steps,
            max_workers=self.max_workers,
        )

    def collect_rollouts(
        self,
        prompt_batch: Any,
        *,
        search_mode: str,
        sampling_params: dict[str, Any],
        judge_fn: Callable[[str, str], float],
        num_rollouts: int | None = None,
        reward_fn: Any = None,
        advantage_config: Any = None,
        batch_judge_fn: Any = None,
        safety_config: Any = None,
        base_seed: int | None = None,
        current_step: int = 0,
        total_steps: int = 1,
    ) -> tuple[list[Any], list[Any]]:
        """Collect rollouts sequentially, one prompt at a time."""
        return _collect_grpo_rollouts_core(
            self.manager,
            prompt_batch,
            search_mode=search_mode,
            sampling_params=sampling_params,
            judge_fn=judge_fn,
            num_rollouts=int(num_rollouts or self.num_rollouts),
            reward_fn=reward_fn,
            advantage_config=advantage_config,
            batch_judge_fn=batch_judge_fn,
            safety_config=safety_config,
            base_seed=base_seed,
            current_step=current_step,
            total_steps=total_steps,
        )

    async def async_training_step(
        self,
        prompt_batch: Any,
        *,
        search_mode: str,
        sampling_params: dict[str, Any],
        judge_fn: Callable[[str, str], float],
        num_rollouts: int | None = None,
        reward_fn: Any = None,
        advantage_config: Any = None,
        batch_judge_fn: Any = None,
        old_backend: Any = None,
        new_backend: Any = None,
        ref_backend: Any = None,
        loss_config: PPOPolicyLossConfig | None = None,
        safety_config: Any = None,
        optimizer: Any = None,
        base_seed: int | None = None,
        current_step: int = 0,
        total_steps: int = 1,
    ) -> Any:
        """Run one GRPO training step with concurrent rollout collection."""
        return await _async_run_grpo_training_step_core(
            self.manager,
            prompt_batch,
            search_mode=search_mode,
            sampling_params=sampling_params,
            judge_fn=judge_fn,
            num_rollouts=int(num_rollouts or self.num_rollouts),
            reward_fn=reward_fn,
            advantage_config=advantage_config,
            batch_judge_fn=batch_judge_fn,
            old_backend=old_backend,
            new_backend=new_backend,
            ref_backend=ref_backend,
            loss_config=loss_config,
            safety_config=safety_config,
            optimizer=optimizer,
            base_seed=base_seed,
            current_step=current_step,
            total_steps=total_steps,
            max_workers=self.max_workers,
        )

    def training_step(
        self,
        prompt_batch: Any,
        *,
        search_mode: str,
        sampling_params: dict[str, Any],
        judge_fn: Callable[[str, str], float],
        num_rollouts: int | None = None,
        reward_fn: Any = None,
        advantage_config: Any = None,
        batch_judge_fn: Any = None,
        old_backend: Any = None,
        new_backend: Any = None,
        ref_backend: Any = None,
        loss_config: PPOPolicyLossConfig | None = None,
        safety_config: Any = None,
        optimizer: Any = None,
        base_seed: int | None = None,
        current_step: int = 0,
        total_steps: int = 1,
    ) -> Any:
        """Run one sequential GRPO training step."""
        return _run_grpo_training_step_core(
            self.manager,
            prompt_batch,
            search_mode=search_mode,
            sampling_params=sampling_params,
            judge_fn=judge_fn,
            num_rollouts=int(num_rollouts or self.num_rollouts),
            reward_fn=reward_fn,
            advantage_config=advantage_config,
            batch_judge_fn=batch_judge_fn,
            old_backend=old_backend,
            new_backend=new_backend,
            ref_backend=ref_backend,
            loss_config=loss_config,
            safety_config=safety_config,
            optimizer=optimizer,
            base_seed=base_seed,
            current_step=current_step,
            total_steps=total_steps,
        )


async def async_run_grpo_training_step(
    manager: LLMGenerationManager,
    prompt_batch: Any,
    *,
    search_mode: str,
    sampling_params: dict[str, Any],
    judge_fn: Callable[[str, str], float],
    num_rollouts: int = 4,
    reward_fn: Any = None,
    advantage_config: Any = None,
    batch_judge_fn: Any = None,
    old_backend: LogProbCapable | None = None,
    new_backend: LogProbCapable | None = None,
    ref_backend: LogProbCapable | None = None,
    loss_config: PPOPolicyLossConfig | None = None,
    safety_config: GRPORolloutSafetyConfig | None = None,
    optimizer: Any = None,
    base_seed: int | None = None,
    current_step: int = 0,
    total_steps: int = 1,
    max_workers: int | None = None,
) -> GRPOTrainingStepResult:
    """Run one GRPO trainer step with concurrent rollout collection.

    Delegates to ``LocalGRPOController.async_training_step`` which runs all
    ``N_prompts × N_rollouts`` trajectories in parallel, overlapping HTTP
    search I/O, then performs one learner-side update.
    """
    return await LocalGRPOController(
        manager, num_rollouts=num_rollouts, max_workers=max_workers
    ).async_training_step(
        prompt_batch,
        search_mode=search_mode,
        sampling_params=sampling_params,
        judge_fn=judge_fn,
        num_rollouts=num_rollouts,
        reward_fn=reward_fn,
        advantage_config=advantage_config,
        batch_judge_fn=batch_judge_fn,
        old_backend=old_backend,
        new_backend=new_backend,
        ref_backend=ref_backend,
        loss_config=loss_config,
        safety_config=safety_config,
        optimizer=optimizer,
        base_seed=base_seed,
        current_step=current_step,
        total_steps=total_steps,
    )


class CheckpointNotSupportedError(RuntimeError):
    """A trainer was asked to checkpoint but implements no hook for it.

    Raised rather than skipped. ``train_loop`` writes its own step manifest
    alongside whatever the trainer persists, so a skipped save leaves a
    directory that looks like a checkpoint and holds no weights, and a skipped
    load restores the step counter onto a freshly initialised policy. Both
    report success while discarding the training they exist to protect --
    the logs resume at step N while the model starts from scratch.

    Trainers that implement the hooks: ``LLMGRPOTrainer`` and its subclass
    ``SearchAgentGRPOTrainer``. Those that do not: ``GRPOTrainer`` (the bandit
    trainer), ``DPOTrainer``, ``SFTTrainer``.
    """


@dataclass
class TrainLoopConfig:
    max_steps: int
    ckpt_dir: str | None = None
    ckpt_every: int = 0  # 0 disables periodic checkpointing
    step_timeout_s: float | None = None  # None disables the per-step timeout


def _require_checkpoint_hook(trainer: Any, hook: str) -> Callable[[str], Any]:
    """Return *trainer*'s checkpoint hook, or refuse the operation outright."""
    fn = getattr(trainer, hook, None)
    if not callable(fn):
        raise CheckpointNotSupportedError(
            f"{type(trainer).__name__} does not implement {hook}(), so this "
            f"checkpoint would carry no model weights -- only a step number. "
            f"Implement {hook}() on the trainer, or run without checkpointing."
        )
    return fn


def save_checkpoint(trainer: Any, path: str, step: int) -> None:
    """Persist trainer state plus the step manifest under *path*.

    Refuses a trainer that cannot save: the manifest alone is not a checkpoint.
    """
    trainer_save = _require_checkpoint_hook(trainer, "save_checkpoint")
    Path(path).mkdir(parents=True, exist_ok=True)
    trainer_save(path)
    (Path(path) / "trainer_state.json").write_text(json.dumps({"step": step}))


def load_checkpoint(trainer: Any, path: str) -> int:
    """Restore trainer state from *path*; return the step to resume at.

    Refuses a trainer that cannot load, rather than resuming the step counter
    onto whatever weights the trainer happens to have been constructed with.
    """
    trainer_load = _require_checkpoint_hook(trainer, "load_checkpoint")
    trainer_load(path)
    state = json.loads((Path(path) / "trainer_state.json").read_text())
    return int(state["step"])


async def train_loop(
    trainer: Any,
    prompts: list[str],
    ground_truths: list[str],
    config: TrainLoopConfig,
    *,
    metadata: list[dict] | None = None,
    resume_from: str | None = None,
    on_metrics: Callable[[dict], None] | None = None,
) -> list[dict]:
    """Run up to ``config.max_steps`` training steps, durably.

    Returns the per-step metrics history (skipped steps are absent).
    """
    import asyncio

    # Check both hooks before the first step, not at the first checkpoint: a run
    # that will not be able to save is better stopped now than after it has
    # spent the hours the checkpoint was meant to protect.
    if config.ckpt_dir is not None and config.ckpt_every:
        _require_checkpoint_hook(trainer, "save_checkpoint")

    start_step = load_checkpoint(trainer, resume_from) if resume_from else 0
    history: list[dict] = []

    for step in range(start_step, config.max_steps):
        try:
            coro = trainer.step_async(prompts, ground_truths, metadata=metadata)
            if config.step_timeout_s is not None:
                metrics = await asyncio.wait_for(coro, config.step_timeout_s)
            else:
                metrics = await coro
        except Exception as exc:  # noqa: BLE001 - a bad step must not abort the run
            logger.warning(
                "Training step %d failed or timed out (%s); skipping.", step, exc
            )
            continue

        record = {**metrics, "step": step}
        history.append(record)
        if on_metrics is not None:
            on_metrics(record)
        if config.ckpt_dir is not None:
            _append_jsonl(Path(config.ckpt_dir) / "metrics.jsonl", record)
        if (
            config.ckpt_dir is not None
            and config.ckpt_every
            and (step + 1) % config.ckpt_every == 0
        ):
            save_checkpoint(
                trainer, str(Path(config.ckpt_dir) / f"step_{step + 1}"), step + 1
            )

    return history


def _append_jsonl(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as fh:
        fh.write(json.dumps(record) + "\n")
