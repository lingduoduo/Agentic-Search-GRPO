from __future__ import annotations

import sys
from types import SimpleNamespace

import numpy as np
import pytest

from src.internal.document_index.hierarchy_models import (
    LocalHierarchyDescriber,
    SentenceEmbedding,
)


def test_embedding_is_lazy_local_and_prepares_e5_passages(monkeypatch):
    class Encoder:
        def __init__(self, name, **kwargs):
            assert name == "intfloat/multilingual-e5-base"
            assert kwargs == {"device": "cpu", "trust_remote_code": False}

        def encode(self, texts, **kwargs):
            assert texts == ["passage: 中文 title", "passage: already"]
            assert kwargs["normalize_embeddings"] is True
            assert kwargs["batch_size"] == 2
            return np.array([[1.0, 0.0], [0.0, 1.0]])

    adapter = SentenceEmbedding("intfloat/multilingual-e5-base", batch_size=2)
    monkeypatch.setitem(
        sys.modules,
        "sentence_transformers",
        SimpleNamespace(SentenceTransformer=Encoder),
    )
    assert adapter.encode(["中文 title", "passage: already"]).tolist() == [
        [1.0, 0.0],
        [0.0, 1.0],
    ]


def test_non_e5_model_gets_unmodified_text(monkeypatch):
    class Encoder:
        def __init__(self, *args, **kwargs):
            pass

        def encode(self, texts, **kwargs):
            assert texts == ["title"]
            return np.array([[1.0]])

    monkeypatch.setitem(
        sys.modules,
        "sentence_transformers",
        SimpleNamespace(SentenceTransformer=Encoder),
    )
    assert SentenceEmbedding("local-model").encode(["title"]).tolist() == [[1.0]]


def test_empty_embedding_batch_does_not_load_model(monkeypatch):
    monkeypatch.setitem(sys.modules, "sentence_transformers", None)
    assert SentenceEmbedding("local-model").encode([]).shape == (0, 0)


@pytest.mark.parametrize(
    "kwargs", [{"model_name": ""}, {"model_name": "local", "batch_size": 0}]
)
def test_embedding_configuration_is_validated(kwargs):
    with pytest.raises(ValueError):
        SentenceEmbedding(**kwargs)


def test_local_describer_generates_only_offline_with_bounded_context(monkeypatch):
    torch = pytest.importorskip("torch")

    class Tokenizer:
        pad_token_id = None
        eos_token_id = 0

        def apply_chat_template(self, messages, **kwargs):
            assert "untrusted" in messages[0]["content"]
            assert "中文" in messages[1]["content"]
            return messages[1]["content"]

        def __call__(self, text, **kwargs):
            assert kwargs.get("truncation", False) is False
            return {
                "input_ids": torch.tensor([[1, 2, 3]]),
                "attention_mask": torch.ones((1, 3), dtype=torch.long),
            }

        def decode(self, tokens, **kwargs):
            assert tokens.tolist() == [4, 5]
            return "中文知识主题"

    class Model:
        device = "cpu"
        config = SimpleNamespace(max_position_embeddings=128)

        def to(self, device):
            return self

        def eval(self):
            return self

        def generate(self, **kwargs):
            assert not torch.is_grad_enabled()
            assert kwargs["do_sample"] is False
            assert kwargs["max_new_tokens"] == 16
            assert kwargs["pad_token_id"] == 0
            return torch.tensor([[1, 2, 3, 4, 5]])

    class Loader:
        def __init__(self, instance):
            self.instance = instance

        def from_pretrained(self, name, **kwargs):
            assert name == "/local/model"
            assert kwargs["trust_remote_code"] is False
            return self.instance

    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(
            AutoTokenizer=Loader(Tokenizer()), AutoModelForCausalLM=Loader(Model())
        ),
    )
    describer = LocalHierarchyDescriber(
        "/local/model", max_new_tokens=16, max_context_tokens=64
    )
    assert (
        describer.describe(
            "主题", [{"id": "a", "title": "中文", "contents": "第一句。后续文本。"}]
        )
        == "中文知识主题"
    )
