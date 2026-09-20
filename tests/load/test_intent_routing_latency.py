"""Intent routing latency: the serving cost of one route decision.

This is a measurement, not a gate. It lives in tests/load/ because a
wall-clock assertion only means something on a machine that is not also
running 4000 other tests -- see issue #593, and the sibling flake in
test_mcp_document_tools.py.

Run with:
    pytest tests/load/test_intent_routing_latency.py -v -s -m load
"""

import functools
from pathlib import Path
from time import perf_counter

import pytest

pytestmark = pytest.mark.load

DATA = Path(__file__).resolve().parents[2] / "data"

# Re-measured 2026-08-14 on the intfloat/e5-small-v2 index over the 304-example
# canonical set:
#   p95 routing latency  12.20 ms  -> ceiling 25.0 ms
# The ceiling is 2x the measured value. If this fails on a quiet machine, the
# encoder or the scoring path genuinely regressed. Do not raise the number to
# make a loaded machine pass -- that is what moved this test here.
_P95_LATENCY_CEILING_MS = 25.0


def test_routing_one_request_stays_under_the_latency_ceiling():
    """Encode plus decide, the whole serving cost of a route decision."""
    pytest.importorskip("sentence_transformers")

    from src.model.pre_training.intents.model import (
        INDEX_FILENAME,
        IntentIndex,
        encode_texts,
    )

    index_dir = DATA / "intent_index"
    if not (index_dir / INDEX_FILENAME).exists():
        pytest.skip(
            "run `python -m src.model.pre_training.intents.cli build --canonical "
            f"data/intent_canonical.json --output {index_dir}` to measure latency"
        )

    index = IntentIndex.load(index_dir / INDEX_FILENAME)
    query = "book the meeting room for tomorrow afternoon"
    decide = functools.partial(index.decide, min_margin=0.015, min_module_score=0.45)
    for _ in range(5):
        decide(encode_texts([query])[0])

    timings = []
    for _ in range(50):
        start = perf_counter()
        decide(encode_texts([query])[0])
        timings.append((perf_counter() - start) * 1_000)

    p95 = sorted(timings)[int(0.95 * (len(timings) - 1))]
    print(f"\np95 routing latency: {p95:.2f} ms (ceiling {_P95_LATENCY_CEILING_MS})")
    assert p95 <= _P95_LATENCY_CEILING_MS, p95
