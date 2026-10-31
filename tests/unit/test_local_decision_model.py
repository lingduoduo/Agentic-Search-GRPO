from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from src.internal.document_index.decision_model import LocalDecisionModel
from src.internal.document_index.decision_retriever import DecisionCandidate


class Tokenizer:
    pad_token_id = 0
    eos_token_id = 0
    labels = {"Yes": [1], "No": [2], "A": [3], "B": [4]}

    def __init__(self):
        self.messages = []

    def encode(self, text, **kwargs):
        return self.labels.get(
            text, [5] * (10 if "candidate A relevant" in text else 12)
        )

    def apply_chat_template(self, messages, **kwargs):
        assert kwargs == {"tokenize": False, "add_generation_prompt": True}
        self.messages.append(messages)
        return json.dumps(messages)


class LogitsModel:
    config = SimpleNamespace(max_position_embeddings=32)
    device = "cpu"

    def __init__(self):
        self.calls = []

    def eval(self):
        return self

    def __call__(self, **kwargs):
        import torch

        assert not torch.is_grad_enabled()
        assert kwargs["use_cache"] is False
        self.calls.append(kwargs)
        batch, width = kwargs["input_ids"].shape
        logits = torch.zeros((batch, width, 6))
        logits[:, -1, 1] = 2
        logits[:, -1, 4] = 3
        return SimpleNamespace(logits=logits)


@pytest.fixture
def adapter():
    pytest.importorskip("torch")
    return LocalDecisionModel(LogitsModel(), Tokenizer(), inference_batch_size=2)


CANDIDATES = [
    DecisionCandidate("a", "document", "First document"),
    DecisionCandidate("b", "document", "Second document"),
]


def test_binary_uses_shared_context_without_sampling_and_normalizes_yes_no(adapter):
    scores = adapter.binary("question", CANDIDATES)
    assert scores == pytest.approx([0.880797, 0.880797])
    assert len(adapter.model.calls) == 1
    for messages in adapter.tokenizer.messages:
        assert all(
            candidate.text in messages[-1]["content"] for candidate in CANDIDATES
        )
    call = adapter.model.calls[0]
    assert call["attention_mask"].tolist() == [[0, 0] + [1] * 10, [1] * 12]
    assert call["position_ids"][0].tolist() == [0, 0] + list(range(10))


def test_choice_is_one_joint_distribution_not_independent_binary_scores(adapter):
    scores = adapter.choice("question", CANDIDATES)
    assert sum(scores) == pytest.approx(1)
    assert scores[1] > scores[0]
    assert len(adapter.model.calls) == 1
    assert len(adapter.tokenizer.messages) == 1


def test_inference_batches_are_bounded(adapter):
    adapter.inference_batch_size = 1
    adapter.binary("question", CANDIDATES)
    assert len(adapter.model.calls) == 2


def test_context_overflow_fails_before_model_execution(adapter):
    adapter.max_context_tokens = 8
    with pytest.raises(ValueError, match="context exceeds"):
        adapter.binary("question", CANDIDATES)
    assert adapter.model.calls == []


def test_model_context_limit_is_also_respected(adapter):
    adapter.model.config = SimpleNamespace(max_position_embeddings=8)
    with pytest.raises(ValueError, match="context exceeds 8"):
        adapter.choice("question", CANDIDATES)


def test_multitoken_decision_labels_are_rejected(adapter):
    adapter.tokenizer.labels = {**Tokenizer.labels, "Yes": [1, 2]}
    with pytest.raises(ValueError, match="distinct single tokens"):
        adapter.binary("question", CANDIDATES)
    assert adapter.model.calls == []


@pytest.mark.parametrize("count", [0, 27])
def test_candidate_limit_is_explicit(adapter, count):
    with pytest.raises(ValueError, match="1 to 26"):
        adapter.choice("question", [CANDIDATES[0]] * count)


def test_real_transformer_forward_is_padding_invariant_without_generation():
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    with torch.random.fork_rng():
        torch.manual_seed(7)
        model = transformers.GPT2LMHeadModel(
            transformers.GPT2Config(
                vocab_size=8,
                n_positions=32,
                n_embd=16,
                n_layer=1,
                n_head=2,
                bos_token_id=0,
                eos_token_id=0,
            )
        )

    def forbid_generation(*args, **kwargs):
        raise AssertionError("Decision scoring must not generate tokens")

    model.generate = forbid_generation
    adapter = LocalDecisionModel(model, Tokenizer(), inference_batch_size=2)
    batched = adapter.binary("question", CANDIDATES)
    adapter.inference_batch_size = 1
    separate = adapter.binary("question", CANDIDATES)
    assert batched == pytest.approx(separate, abs=1e-6)
    assert all(0 <= score <= 1 for score in batched)
    assert sum(adapter.choice("question", CANDIDATES)) == pytest.approx(1)


class ContextTokenizer(Tokenizer):
    """Deterministic token budgets without model downloads."""

    def encode(self, text, **kwargs):
        if text in self.labels:
            return self.labels[text]
        messages = json.loads(text)
        context = json.loads(messages[-1]["content"].split("\n\n")[0])
        return [5] * (
            8 + sum(4 * len(c["text"].split()) for c in context["candidates"])
        )


def test_binary_splits_context_without_truncating_candidates():
    pytest.importorskip("torch")
    adapter = LocalDecisionModel(
        LogitsModel(), ContextTokenizer(), max_context_tokens=20
    )
    assert adapter.binary("q", CANDIDATES) == pytest.approx([0.880797, 0.880797])
    assert len(adapter.model.calls) == 2
    assert all(call["input_ids"].shape[1] == 16 for call in adapter.model.calls)


def test_choice_capacity_preserves_ranked_prefix_and_never_runs_model():
    pytest.importorskip("torch")
    adapter = LocalDecisionModel(
        LogitsModel(), ContextTokenizer(), max_context_tokens=20
    )
    candidates = [
        DecisionCandidate("a", "passage", "first"),
        DecisionCandidate("b", "passage", "second"),
        DecisionCandidate("c", "passage", "third long"),
    ]
    assert adapter.choice_capacity("q", candidates) == 2
    assert adapter.model.calls == []
    assert adapter.choice("q", candidates[:2]) == pytest.approx(
        [0.04742587, 0.95257413]
    )


def test_choice_capacity_rejects_oversized_first_candidate():
    adapter = LocalDecisionModel(
        LogitsModel(), ContextTokenizer(), max_context_tokens=10
    )
    with pytest.raises(ValueError, match="context exceeds"):
        adapter.choice_capacity("q", CANDIDATES)
    assert adapter.model.calls == []


def test_real_transformer_computes_only_last_position_logits():
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    with torch.random.fork_rng():
        torch.manual_seed(11)
        model = transformers.GPT2LMHeadModel(
            transformers.GPT2Config(
                vocab_size=8,
                n_positions=32,
                n_embd=16,
                n_layer=1,
                n_head=2,
                eos_token_id=0,
            )
        )
    import inspect

    if "logits_to_keep" not in inspect.signature(model.forward).parameters:
        pytest.skip("Installed Transformers model has no last-logit support")
    widths = []
    hook = model.lm_head.register_forward_pre_hook(
        lambda module, args: widths.append(args[0].shape[1])
    )
    try:
        adapter = LocalDecisionModel(model, Tokenizer(), inference_batch_size=2)
        scores = adapter.binary("q", CANDIDATES)
    finally:
        hook.remove()
    assert len(scores) == 2
    assert widths == [1]


def test_legacy_last_logits_argument_is_supported():
    pytest.importorskip("torch")

    class LegacyModel(LogitsModel):
        def forward(self, *, num_logits_to_keep=0, **kwargs):
            assert num_logits_to_keep == 1
            return super().__call__(**kwargs)

        __call__ = forward

    adapter = LocalDecisionModel(LegacyModel(), Tokenizer())
    assert adapter.binary("q", CANDIDATES) == pytest.approx([0.880797, 0.880797])


def test_navigation_fits_one_final_choice_without_truncating_passages():
    pytest.importorskip("torch")
    from src.internal.document_index.decision_retriever import DecisionRetriever

    adapter = LocalDecisionModel(
        LogitsModel(), ContextTokenizer(), max_context_tokens=24
    )
    run = DecisionRetriever(
        [
            {"id": str(i), "text": word}
            for i, word in enumerate(("alpha", "beta", "gamma"))
        ],
        adapter,
    ).navigate("q", topk=3)
    assert len(run.results) == 2
    assert {item["document"]["text"] for item in run.results} == {"alpha", "beta"}
    assert run.stages[-1].kind == "choice"
    assert len(run.stages[-1].candidate_ids) == 2
    assert sum(run.stages[-1].scores) == pytest.approx(1)
