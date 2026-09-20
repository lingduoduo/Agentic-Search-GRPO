# Intent Latency Test Determinism Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move the contention-sensitive p95 wall-clock assertion out of the default test run, and replace it with four deterministic assertions that count work instead of measuring elapsed time.

**Architecture:** The timing loop relocates verbatim to `tests/load/`, which is marked `load` and sits outside `testpaths`, so it runs neither in the default suite nor in CI. In its place, `tests/unit/test_intent_evaluation.py` gains four tests that count encoder constructions, encode calls, similarity computations, and disk reads.

**Tech Stack:** Python 3, pytest, `monkeypatch`, numpy, `src/model/pre_training/intents/model.py`.

**Spec:** `docs/superpowers/specs/2026-09-20-intent-latency-test-determinism-design.md`

Closes #593.

## Global Constraints

- **No production code changes.** This plan touches only `tests/`. If a step seems to require editing `src/`, stop — that is a different change.
- `_P95_LATENCY_CEILING_MS = 25.0` keeps its value. The ceiling is not raised, lowered, or re-derived; it only moves file.
- The new unit tests must not call `_report()`. It is `lru_cache`d but runs a full index evaluation, and none of these four tests depend on the index being *accurate* — only on how much work a decision costs.
- `monkeypatch` is the only mechanism for swapping `_MODEL_CACHE` or `SentenceTransformer`, so state is restored automatically and cannot leak into a neighbouring test.
- Run `ruff check . --fix && ruff format .` before each commit; a pre-commit `ruff-format` hook aborts the commit otherwise.

---

### Task 1: Relocate the timing measurement to tests/load/

**Files:**
- Create: `tests/load/test_intent_routing_latency.py`
- Modify: `tests/unit/test_intent_evaluation.py` (delete `test_routing_one_request_stays_under_the_latency_ceiling` and `_P95_LATENCY_CEILING_MS`)

**Interfaces:**
- Consumes: `IntentIndex.load`, `encode_texts` from `src.model.pre_training.intents.model`.
- Produces: nothing other tasks depend on. Task 2 is independent of this one.

- [ ] **Step 1: Create the load test**

`tests/load/` already exists and holds `test_load.py`. Create a sibling:

```python
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
```

- [ ] **Step 2: Delete the original test and its constant**

In `tests/unit/test_intent_evaluation.py`, delete the whole
`test_routing_one_request_stays_under_the_latency_ceiling` function and the
`_P95_LATENCY_CEILING_MS = 25.0` line (around `:668`).

Leave the accuracy and AUC provenance comment block intact. Its latency line
(`#   p95 routing latency       12.20 ms   -> ceiling 25.0 ms`) gains a pointer
so the number is still findable:

```
#   p95 routing latency       12.20 ms                             -> ceiling 25.0 ms
#     (that bar now lives in tests/load/test_intent_routing_latency.py -- see #593)
```

- [ ] **Step 3: Verify the default run no longer collects it**

Run: `python3 -m pytest tests/unit/test_intent_evaluation.py --collect-only -q 2>&1 | grep -c latency`

Expected: `0`. The timing test is gone from the default path.

Then confirm it is still reachable where it now lives:

Run: `python3 -m pytest tests/load/test_intent_routing_latency.py -m load --collect-only -q 2>&1 | tail -3`

Expected: 1 test collected.

- [ ] **Step 4: Confirm tests/load/ is outside the default run**

Run: `python3 -m pytest --collect-only -q 2>&1 | grep -c "tests/load"`

Expected: `0`. `testpaths = ["tests/unit", "tests/regression"]` excludes it, so no `addopts` change is needed. If this returns non-zero, stop — the assumption behind this whole task is wrong.

- [ ] **Step 5: Commit**

```bash
ruff check . --fix && ruff format .
git add tests/load/test_intent_routing_latency.py tests/unit/test_intent_evaluation.py
git commit -m "test(intents): move the p95 latency bar to tests/load/"
```

---

### Task 2: Four deterministic assertions in place of the SLA

**Files:**
- Modify: `tests/unit/test_intent_evaluation.py` (add a helper and four tests)

**Interfaces:**
- Consumes: `IntentIndex`, `INDEX_FILENAME`, `encode_texts`, `_MODEL_CACHE` from `src.model.pre_training.intents.model`; `DATA` and `functools` already imported in the test module.
- Produces: `_index_or_skip() -> IntentIndex`, a light skip guard used by all four tests.

- [ ] **Step 1: Write the four tests**

Append to `tests/unit/test_intent_evaluation.py`. `numpy` and `pytest` are
already imported at module scope; add them only if the file lacks them.

```python
# ---------------------------------------------------------------------------
# Serving-cost invariants. These replace a p95 wall-clock assertion that flaked
# under suite load (#593). What that SLA really guarded is that a route
# decision does not reload the encoder or re-read the index -- both counting
# questions, and counting does not care what else is running.
# ---------------------------------------------------------------------------


def _index_or_skip():
    """Load the index, or skip. Deliberately lighter than ``_report()``.

    These tests count work, not accuracy, so they need the index to exist but
    not to be built with the current encoder -- and they must not pay for a
    full evaluation run.
    """
    from src.model.pre_training.intents.model import INDEX_FILENAME, IntentIndex

    index_dir = DATA / "intent_index"
    if not (index_dir / INDEX_FILENAME).exists():
        pytest.skip(f"intent index is missing: {index_dir / INDEX_FILENAME}")
    return IntentIndex.load(index_dir / INDEX_FILENAME)


def _fake_encoder(dim):
    """A stand-in for SentenceTransformer that records how it was used."""
    state = {"constructions": 0, "encodes": []}

    class _Fake:
        def __init__(self, model_name, device=None):
            state["constructions"] += 1

        def encode(self, texts, **kwargs):
            state["encodes"].append(list(texts))
            rows = np.zeros((len(texts), dim), dtype=np.float32)
            rows[:, 0] = 1.0  # unit vector: the index requires normalized rows
            return rows

    return _Fake, state


def _patch_encoder(monkeypatch, dim):
    """Install the fake and give it an empty cache to fill."""
    sentence_transformers = pytest.importorskip("sentence_transformers")
    from src.model.pre_training.intents import model as model_mod

    fake, state = _fake_encoder(dim)
    monkeypatch.setattr(sentence_transformers, "SentenceTransformer", fake)
    monkeypatch.setattr(model_mod, "_MODEL_CACHE", {})
    return state


def test_the_encoder_is_constructed_once_across_many_decisions(monkeypatch):
    """A cache break would put a multi-second model load on every request.

    This is the regression the old p95 ceiling actually guarded: `_model` says
    "Loading costs seconds; encoding costs ms".
    """
    from src.model.pre_training.intents.model import encode_texts

    index = _index_or_skip()
    state = _patch_encoder(monkeypatch, index._vectors.shape[1])

    for _ in range(10):
        index.decide(
            encode_texts(["book the meeting room"])[0],
            min_margin=0.015,
            min_module_score=0.45,
        )

    assert state["constructions"] == 1


def test_each_decision_encodes_exactly_once(monkeypatch):
    """One forward pass per decision, and one text per pass."""
    from src.model.pre_training.intents.model import encode_texts

    index = _index_or_skip()
    state = _patch_encoder(monkeypatch, index._vectors.shape[1])

    for _ in range(10):
        index.decide(
            encode_texts(["book the meeting room"])[0],
            min_margin=0.015,
            min_module_score=0.45,
        )

    assert len(state["encodes"]) == 10
    assert all(len(batch) == 1 for batch in state["encodes"])


def test_decide_computes_similarities_a_fixed_number_of_times(monkeypatch):
    """Scoring cost per decision must not scale with routes or modules.

    Two per decision today -- ``route_scores`` computes them, then
    ``_emit_modules`` -> ``module_scores`` computes them again. That
    redundancy is pinned here rather than fixed; collapsing it is an
    optimization, and this test is what would tell you it worked.
    """
    from src.model.pre_training.intents.model import IntentIndex

    index = _index_or_skip()
    calls = {"n": 0}
    original = IntentIndex._similarities

    def counting(self, vector):
        calls["n"] += 1
        return original(self, vector)

    monkeypatch.setattr(IntentIndex, "_similarities", counting)

    vector = index._vectors[0]
    for _ in range(10):
        index.decide(vector, min_margin=0.015, min_module_score=0.45)

    assert calls["n"] == 20


def test_decide_never_re_reads_the_index_from_disk(monkeypatch):
    """Break ``np.load`` *after* loading: a decision that works touched no disk."""
    index = _index_or_skip()

    def _exploding_load(*args, **kwargs):
        raise AssertionError("decide() re-read the index from disk")

    monkeypatch.setattr(np, "load", _exploding_load)

    vector = index._vectors[0]
    for _ in range(10):
        decision = index.decide(vector, min_margin=0.015, min_module_score=0.45)
    assert decision.route
```

- [ ] **Step 2: Run them**

Run: `python3 -m pytest tests/unit/test_intent_evaluation.py -k "constructed_once or encodes_exactly_once or fixed_number_of_times or never_re_reads" -v`

Expected: 4 PASS. If any SKIPs, `data/intent_index/index.npz` is missing — build it with the command in `_index_or_skip`'s message before continuing, because a skipped test proves nothing.

- [ ] **Step 3: Mutation-check assertion 1 (encoder constructed once)**

The tests are the deliverable here, so each one must be shown to fail when its invariant breaks. Bypass the cache in `_model`:

```bash
python3 - <<'PY'
import pathlib
p = pathlib.Path("src/model/pre_training/intents/model.py")
s = p.read_text()
old = """    cached = _MODEL_CACHE.get(model_name)
    if isinstance(cached, Exception):"""
new = """    cached = None
    if isinstance(cached, Exception):"""
assert s.count(old) == 1
p.write_text(s.replace(old, new))
PY
python3 -m pytest tests/unit/test_intent_evaluation.py -k "constructed_once" -q
git checkout src/model/pre_training/intents/model.py
```

Expected: RED (`constructions == 10`), then restored.

- [ ] **Step 4: Mutation-check assertion 2 (one encode per decision)**

```bash
python3 - <<'PY'
import pathlib
p = pathlib.Path("src/model/pre_training/intents/model.py")
s = p.read_text()
old = "    prefix = prefix_for(model_name)\n"
new = "    prefix = prefix_for(model_name)\n    _model(model_name).encode([prefix + t for t in texts])  # MUTANT\n"
assert s.count(old) == 1
p.write_text(s.replace(old, new))
PY
python3 -m pytest tests/unit/test_intent_evaluation.py -k "encodes_exactly_once" -q
git checkout src/model/pre_training/intents/model.py
```

Expected: RED (20 encodes, not 10), then restored.

- [ ] **Step 5: Mutation-check assertion 3 (fixed similarity work)**

```bash
python3 - <<'PY'
import pathlib
p = pathlib.Path("src/model/pre_training/intents/model.py")
s = p.read_text()
old = "        routes = self.route_scores(vector, top_k=top_k)\n"
new = "        routes = self.route_scores(vector, top_k=top_k)\n        self._similarities(vector)  # MUTANT\n"
assert s.count(old) == 1
p.write_text(s.replace(old, new))
PY
python3 -m pytest tests/unit/test_intent_evaluation.py -k "fixed_number_of_times" -q
git checkout src/model/pre_training/intents/model.py
```

Expected: RED (30, not 20), then restored.

- [ ] **Step 6: Mutation-check assertion 4 (no disk re-read)**

```bash
python3 - <<'PY'
import pathlib
p = pathlib.Path("src/model/pre_training/intents/model.py")
s = p.read_text()
old = "        routes = self.route_scores(vector, top_k=top_k)\n"
new = "        np.load(self._source_path) if hasattr(self, '_source_path') else np.load('data/intent_index/index.npz')  # MUTANT\n        routes = self.route_scores(vector, top_k=top_k)\n"
assert s.count(old) == 1
p.write_text(s.replace(old, new))
PY
python3 -m pytest tests/unit/test_intent_evaluation.py -k "never_re_reads" -q
git checkout src/model/pre_training/intents/model.py
```

Expected: RED with `decide() re-read the index from disk`, then restored.

- [ ] **Step 7: Confirm the restore is clean**

Run: `git diff --stat src/model/pre_training/intents/model.py`

Expected: empty. Four mutations were applied and reverted; a leftover would poison every later step. If anything looks stale, delete `src/model/pre_training/intents/__pycache__` — reverted source can leave mutated bytecode and fake a red suite.

Then re-run the four tests and confirm 4 PASS.

- [ ] **Step 8: Commit**

```bash
ruff check . --fix && ruff format .
git add tests/unit/test_intent_evaluation.py
git commit -m "test(intents): count the work a route decision does, don't time it"
```

---

### Task 3: Verify and open the PR

**Files:** none modified.

- [ ] **Step 1: Run the full default suite**

Run: `python3 -m pytest -q 2>&1 | tail -3`

Expected: PASS, at 4275 (4272 + 4 added - 1 removed). If the count differs, work out why before proceeding.

- [ ] **Step 2: Run the relocated measurement deliberately**

Run: `python3 -m pytest tests/load/test_intent_routing_latency.py -m load -v -s 2>&1 | tail -6`

Expected: PASS, printing the observed p95. This is the machine it is meant to run on, and it confirms the move did not break the test itself.

- [ ] **Step 3: Push and open the PR**

```bash
git push -u origin test/intent-latency-determinism
gh pr create --title "test(intents): count the work a route decision does instead of timing it" --body "$(cat <<'BODY'
Closes #593.

## Summary

`test_routing_one_request_stays_under_the_latency_ceiling` timed 50 `sentence_transformers` forward passes and asserted a p95 SLA inside the default unit suite. On one commit during #590 it FAILED in a 514-test subset, passed alone in 8.14s, and passed in the full 4272-test run — same code, three outcomes.

Raising the ceiling was never the fix. The file's own provenance comment records p95 **12.20 ms measured, ceiling 25.0 ms** — already 2x headroom, still flaking. The variance is contention for the machine, not a slower encoder, so more headroom only relocates the failure while making the assertion vacuous on a quiet box.

**The measurement moves; it is not lost.** `tests/load/` already exists for exactly this: marked `load`, outside `testpaths`, so it runs in neither the default suite nor CI (`pytest tests/unit/ -q`). The timing loop goes there verbatim, ceiling unchanged, and now prints the observed p95.

**Four deterministic assertions replace it**, counting work instead of elapsed time:

| Assertion | Catches |
|---|---|
| encoder constructed exactly once over 10 decisions | a `_MODEL_CACHE` break putting a multi-second model load on every request |
| one encode per decision, one text per encode | redundant or unbatched encoding |
| `_similarities` computed exactly 20 times over 10 decisions | scoring that scales with route or module count |
| `np.load` never called during `decide` | a decision that re-reads `index.npz` |

All four were mutation-checked: bypassing the cache, double-encoding, an extra `_similarities`, and a re-read each turn the matching test red.

## Notes

The third assertion pins **2** similarity computations per decision — `route_scores` computes them, then `_emit_modules` → `module_scores` computes them again. That redundancy is real and is documented rather than fixed here; collapsing it is an optimization, and this test is what would confirm it worked.

The first two use a fake encoder, so they need no model download and run in milliseconds. `monkeypatch` swaps `_MODEL_CACHE` and restores it, so the guard cannot leak into a neighbouring test.

**No production code changed.** Tests only.

This is one of two instances of the pattern. The RSS watchdog test in `test_mcp_document_tools.py` flakes for the same structural reason — a tight wall-clock budget standing in for resource behavior — and is left for its own change, with PR #504 as the model.

Spec: `docs/superpowers/specs/2026-09-20-intent-latency-test-determinism-design.md`
Plan: `docs/superpowers/plans/2026-09-20-intent-latency-test-determinism.md`

🤖 Generated with [Claude Code](https://claude.com/claude-code)
BODY
)"
```

- [ ] **Step 4: Confirm every commit reached the remote**

Run: `git log --oneline origin/test/intent-latency-determinism -4`

Expected: the spec/plan commit and both task commits. Verify before considering the work done — a squash can drop what was pushed late.
