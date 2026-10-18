from __future__ import annotations

import time

from src.context.query_transform import QueryTransformConfig, QueryTransformPipeline
from src.internal.retrieval.async_query_transform import AsyncQueryTransformPipeline
from unittest.mock import MagicMock


def _fake_llm(text: str = "x") -> MagicMock:
    llm = MagicMock()
    llm.complete.return_value = type("R", (), {"text": text})()
    return llm


def test_slow_transform_times_out_and_degrades():
    leaf = QueryTransformPipeline(
        QueryTransformConfig(step_back=True), _fake_llm("broad")
    )
    # Make step_back sleep past the timeout.
    leaf._enhancer.step_back = lambda q: time.sleep(0.5) or "broad"  # type: ignore
    pipe = AsyncQueryTransformPipeline(leaf, timeout_ms=50, max_workers=2)
    bundle = pipe.transform("q")
    assert bundle.step_back is None  # degraded, no raise
    assert bundle.original == "q"


def test_runs_transforms_and_assembles():
    leaf = QueryTransformPipeline(
        QueryTransformConfig(step_back=True), _fake_llm("broad")
    )
    pipe = AsyncQueryTransformPipeline(leaf, timeout_ms=2000)
    bundle = pipe.transform("q")
    assert bundle.step_back == "broad"


def test_max_variants_and_base_config_delegate():
    leaf = QueryTransformPipeline(QueryTransformConfig(max_variants=7), _fake_llm())
    pipe = AsyncQueryTransformPipeline(leaf)
    assert pipe.max_variants == 7
    assert pipe.base_config is leaf.base_config
