"""Shared deterministic fakes for corpus-hierarchy tests."""

import numpy as np

from src.internal.document_index.hierarchy import HierarchyConfig
from src.internal.document_index.hierarchy_descriptions import (
    select_description_samples,
)

TOPICS = ("alpha", "beta", "gamma", "delta", "epsilon", "zeta")
DIMS = len(TOPICS) + 3


def docs(topic, n, start=0):
    """``n`` documents about ``topic``; the title's first word selects its axis."""
    return [
        dict(
            id=f"{topic}-{i:03d}",
            title=f"{topic} document {i}",
            contents=f"{topic} notes number {i}. More detail.",
            metadata={},
        )
        for i in range(start, start + n)
    ]


class TopicEmbedder:
    """Unit axis per topic plus a small deterministic offset per document."""

    model_name = HierarchyConfig().embedding_model

    def __init__(self):
        self.calls = []

    def encode(self, texts):
        self.calls.append(len(texts))
        rows = []
        for text in texts:
            words = text.split("\n")[0].split()
            number = int(words[-1])
            row = np.zeros(DIMS)
            row[TOPICS.index(words[0])] = 1.0
            row[-3] = 0.08 * ((number * 37) % 11) / 11
            row[-2] = 0.08 * ((number * 17) % 13) / 13
            row[-1] = 0.08 * ((number * 7) % 5) / 5
            rows.append(row)
        return np.array(rows)


class CountingDescriber:
    """Records the label of every group it describes."""

    model_name = "counting-describer"
    description_source = "llm"

    def __init__(self):
        self.calls = []

    def describe_with_provenance(self, label, documents, *, vectors=None, sample_size):
        self.calls.append(label)
        samples = select_description_samples(
            documents, vectors=vectors, limit=sample_size
        )
        return dict(
            text=f"Summary of {label} ({len(documents)} documents)",
            document_ids=[s["document_id"] for s in samples],
            roles={s["document_id"]: s["roles"] for s in samples},
            selection_version="coverage-v1",
            generation_version="test",
            warnings=[],
        )


class SpreadTopicEmbedder:
    """Random topic centers with wide per-document noise and a shared direction.

    Unlike ``TopicEmbedder``, members spread far from their centroid and every
    vector shares an anisotropic component, as with real embedding models, so
    ordinary on-topic additions produce tail (poor-fit) documents.
    """

    model_name = HierarchyConfig().embedding_model

    def __init__(self, dims=64, spread=0.6, common=2.0, seed=0):
        rng = np.random.default_rng(seed)
        shared = rng.normal(size=dims)
        self.shared = common * shared / np.linalg.norm(shared)
        self.centers = rng.normal(size=(len(TOPICS), dims))
        self.spread = spread
        self.seed = seed
        self.calls = []

    def encode(self, texts):
        self.calls.append(len(texts))
        rows = []
        for text in texts:
            words = text.split("\n")[0].split()
            topic = TOPICS.index(words[0])
            noise = np.random.default_rng([self.seed, topic, int(words[-1])])
            rows.append(
                self.centers[topic]
                + self.spread * noise.normal(size=len(self.shared))
                + self.shared
            )
        return np.array(rows)
