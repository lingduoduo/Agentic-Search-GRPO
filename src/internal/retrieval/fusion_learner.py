"""Learn per-source RRF weights and provide adaptive MMR lambda.

FusionLearner: grid-searches w_sparse over a QA pairs dataset.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict
from typing import Callable

from .eval_metrics import recall_at_k
from .fusion import FusionWeights

logger = logging.getLogger(__name__)


class FusionLearner:
    """Finds (w_sparse, w_dense) that maximise Recall@10 on a QA pairs dataset.

    Args:
        service_factory: Callable[(w_sparse, w_dense) -> service]. Called once
            per candidate weight pair; the service drives search() internally.
    """

    def __init__(self, service_factory: Callable[[float, float], object]) -> None:
        self._factory = service_factory

    def fit(
        self,
        qa_pairs_path: str,
        *,
        w_sparse_range: list[float] | None = None,
        top_k: int = 10,
        output_path: str | None = None,
    ) -> FusionWeights:
        """Grid-search over w_sparse; w_dense = 1 - w_sparse."""
        w_sparse_range = w_sparse_range or [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

        with open(qa_pairs_path) as f:
            qa_pairs = [json.loads(line) for line in f if line.strip()]

        best = FusionWeights(w_sparse=0.5, w_dense=0.5)
        best_score = -1.0

        for w_s in w_sparse_range:
            w_d = round(1.0 - w_s, 6)
            service = self._factory(w_s, w_d)
            scores: list[float] = []
            for item in qa_pairs:
                results, _ = service.search(item["query"], top_k=top_k)
                retrieved = [r.doc_id for r in results]
                scores.append(
                    recall_at_k(retrieved, set(item["relevant_doc_ids"]), top_k)
                )
            avg = sum(scores) / len(scores) if scores else 0.0
            logger.debug("w_sparse=%.2f w_dense=%.2f recall@10=%.4f", w_s, w_d, avg)
            if avg > best_score:
                best_score = avg
                best = FusionWeights(w_sparse=w_s, w_dense=w_d)

        if output_path:
            with open(output_path, "w") as f:
                json.dump(asdict(best), f, indent=2)
            logger.info("Fusion weights saved to %s", output_path)

        return best
