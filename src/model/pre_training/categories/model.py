"""Category index: nearest-example scoring that picks source categories for a query.

Torch-free at import time; the encoder is only loaded when ``build_category_index``
runs without an injected ``encode``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import numpy as np

from src.internal.tools.search import DOMAIN_REGISTRY
from src.model.pre_training.intents.data import fingerprint
from src.model.pre_training.intents.model import DEFAULT_ENCODER, encode_texts
from src.shared_configs.intent import (
    DEFAULT_CATEGORY_SELECT_MARGIN as DEFAULT_SELECT_MARGIN,
)
from src.shared_configs.intent import DEFAULT_CATEGORY_TOP_K as DEFAULT_TOP_K

INDEX_FILENAME = "index.npz"
MAX_SELECTED = 3

__all__ = [
    "DEFAULT_SELECT_MARGIN",
    "DEFAULT_TOP_K",
    "INDEX_FILENAME",
    "MAX_SELECTED",
    "CategoryExample",
    "CategoryIndex",
    "build_category_index",
    "load_category_examples",
]


@dataclass(frozen=True)
class CategoryExample:
    id: str
    category: str
    text: str


def _normalize(text: str) -> str:
    return " ".join(text.lower().split())


def load_category_examples(
    path: Path, *, require_all_categories: bool = True
) -> list[CategoryExample]:
    """Read and validate a JSON list of ``{id, category, text}`` records."""
    records = json.loads(Path(path).read_text())
    if not isinstance(records, list):
        raise ValueError(f"{path}: expected a JSON list of examples")
    examples: list[CategoryExample] = []
    seen_ids: set[str] = set()
    seen_texts: set[str] = set()
    for record in records:
        fields = {}
        for key in ("id", "category", "text"):
            value = record.get(key) if isinstance(record, dict) else None
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{path}: example needs a non-empty {key}: {record!r}")
            fields[key] = value
        if fields["category"] not in DOMAIN_REGISTRY:
            raise ValueError(f"{path}: unknown category {fields['category']!r}")
        if fields["id"] in seen_ids:
            raise ValueError(f"{path}: duplicate id {fields['id']!r}")
        norm = _normalize(fields["text"])
        if norm in seen_texts:
            raise ValueError(f"{path}: duplicate text {fields['text']!r}")
        seen_ids.add(fields["id"])
        seen_texts.add(norm)
        examples.append(CategoryExample(**fields))
    if require_all_categories:
        present = {e.category for e in examples}
        missing = [c for c in DOMAIN_REGISTRY if c not in present]
        if missing:
            raise ValueError(f"{path}: no examples for categories {missing}")
    return examples


class CategoryIndex:
    """Example vectors plus per-category top-k-mean scoring."""

    def __init__(
        self,
        *,
        examples: list[CategoryExample],
        vectors: np.ndarray,
        encoder: str,
        fingerprint: str,
    ) -> None:
        vectors = np.asarray(vectors)
        if vectors.ndim != 2 or vectors.shape[0] != len(examples):
            raise ValueError("vectors must have one row per example")
        if vectors.dtype != np.float32:
            raise ValueError("vectors must be float32")
        if not np.allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-3):
            raise ValueError("vectors must be L2-normalized")
        self._examples = list(examples)
        self._vectors = vectors
        self._encoder = encoder
        self._fingerprint = fingerprint
        self._rows = {
            category: np.array(
                [i for i, e in enumerate(examples) if e.category == category],
                dtype=np.int64,
            )
            for category in DOMAIN_REGISTRY
        }

    @property
    def encoder(self) -> str:
        return self._encoder

    def scores(self, query_vector: np.ndarray, *, top_k: int) -> dict[str, float]:
        sims = self._vectors @ query_vector
        out: dict[str, float] = {}
        for category, rows in self._rows.items():
            if len(rows) == 0:
                out[category] = 0.0
                continue
            best = np.sort(sims[rows])[::-1][: min(top_k, len(rows))]
            out[category] = float(best.mean())
        return out

    def select(
        self, query_vector: np.ndarray, *, top_k: int, select_margin: float
    ) -> list[str]:
        s = self.scores(query_vector, top_k=top_k)
        order = list(DOMAIN_REGISTRY)
        ranked = sorted(s, key=lambda c: (-s[c], order.index(c)))
        best = s[ranked[0]]
        return [c for c in ranked if s[c] >= best - select_margin][:MAX_SELECTED]

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            vectors=self._vectors,
            examples=np.array(
                json.dumps(
                    [
                        {"id": e.id, "category": e.category, "text": e.text}
                        for e in self._examples
                    ]
                )
            ),
            encoder=np.array(self._encoder),
            fingerprint=np.array(self._fingerprint),
        )

    @classmethod
    def load(cls, path: Path) -> "CategoryIndex":
        if not path.exists():
            raise FileNotFoundError(
                f"Category index is missing {path.name}: {path}. Run the "
                "category index build CLI to create it."
            )
        payload = np.load(path, allow_pickle=False)
        examples = [
            CategoryExample(id=r["id"], category=r["category"], text=r["text"])
            for r in json.loads(str(payload["examples"]))
        ]
        return cls(
            examples=examples,
            vectors=payload["vectors"],
            encoder=str(payload["encoder"]),
            fingerprint=str(payload["fingerprint"]),
        )


def build_category_index(
    examples_path: Path,
    *,
    encode: Callable[[Sequence[str]], np.ndarray] | None = None,
) -> CategoryIndex:
    examples = load_category_examples(examples_path)
    encode = encode or (lambda texts: encode_texts(texts))
    vectors = np.asarray(encode([e.text for e in examples]), dtype=np.float32)
    return CategoryIndex(
        examples=examples,
        vectors=vectors,
        encoder=DEFAULT_ENCODER,
        fingerprint=fingerprint(examples_path),
    )
