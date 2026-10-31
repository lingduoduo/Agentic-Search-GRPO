"""Shared deterministic fakes for corpus-hierarchy tests."""

import numpy as np

from src.internal.document_index.hierarchy import HierarchyConfig

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
