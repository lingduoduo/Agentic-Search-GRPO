# Default Acronym Expansion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `QUERY_EXPANSION_ENABLED=true` expand something on its own, and log once when it cannot.

**Architecture:** A tracked, packaged default acronym table under `src/`; `ACRONYM_PATH` overrides it rather than merging; a warning when the resulting table is empty.

**Tech Stack:** Python 3, pytest, `src/internal/retrieval/query_optimizer.py`.

**Spec:** `docs/superpowers/specs/2026-09-20-default-acronym-expansion-design.md`

Closes #599.

## Global Constraints

- The default file **must not** live under `data/` — that directory is entirely gitignored, so the file would be invisible to every clone.
- `ACRONYM_PATH` **overrides**, never merges. A caller who supplies a file gets exactly that file.
- Do not change `QUERY_EXPANSION_ENABLED`'s default. It stays `false`.
- Do not touch the spell-correction path; it already behaves correctly.
- Run `ruff check . --fix && ruff format .` before each commit.

---

### Task 1: The default table, loaded when nothing overrides it

**Files:**
- Create: `src/internal/retrieval/acronyms.json`
- Modify: `src/internal/retrieval/query_optimizer.py`
- Modify: `pyproject.toml`
- Test: `tests/unit/test_default_acronyms.py` (create)

**Interfaces:**
- Produces: `QueryOptimizer(acronym_path=None)` loads the bundled default; `DEFAULT_ACRONYM_PATH: Path` module constant.

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_default_acronyms.py`:

```python
"""QUERY_EXPANSION_ENABLED must expand something on its own.

Before this, enabling expansion with no ACRONYM_PATH was a silent no-op: the
mechanism worked, the table was empty, and nothing said so. See #599.
"""

import json
import logging

import pytest

from src.internal.retrieval.query_optimizer import (
    DEFAULT_ACRONYM_PATH,
    QueryOptimizer,
)


def test_the_default_table_loads_when_nothing_overrides_it():
    """The whole point: the flag alone now expands."""
    optimizer = QueryOptimizer(None)

    assert optimizer.expand("RAG systems") == "RAG systems retrieval augmented generation"


def test_a_user_file_overrides_the_default_rather_than_merging(tmp_path):
    """A caller who supplies a file gets exactly that file."""
    user_file = tmp_path / "acronyms.json"
    user_file.write_text(json.dumps({"ZZZ": "zulu zulu zulu"}))

    optimizer = QueryOptimizer(str(user_file))

    assert optimizer.expand("ZZZ here") == "ZZZ here zulu zulu zulu"
    # RAG is in the default and absent from the user file: it must NOT expand.
    assert optimizer.expand("RAG here") == "RAG here"


def test_an_empty_table_warns(tmp_path, caplog):
    empty = tmp_path / "empty.json"
    empty.write_text("{}")

    with caplog.at_level(logging.WARNING):
        QueryOptimizer(str(empty))

    assert any("expansion" in r.message.lower() for r in caplog.records)


def test_the_default_table_does_not_warn(caplog):
    with caplog.at_level(logging.WARNING):
        QueryOptimizer(None)

    assert not [r for r in caplog.records if "expansion" in r.message.lower()]


def test_the_bundled_file_is_well_formed():
    payload = json.loads(DEFAULT_ACRONYM_PATH.read_text(encoding="utf-8"))

    assert payload, "the default table must not be empty"
    for key, value in payload.items():
        assert key == key.upper(), f"{key} must be upper-case"
        assert isinstance(value, str) and value.strip(), f"{key} has no expansion"
        assert value.upper() != key, f"{key} expands to itself"


@pytest.mark.parametrize("acronym", ["RAG", "IR", "NDCG", "MMR"])
def test_core_vocabulary_is_present(acronym):
    """These are the system's own terms; losing them is a regression."""
    payload = json.loads(DEFAULT_ACRONYM_PATH.read_text(encoding="utf-8"))

    assert acronym in payload
```

- [ ] **Step 2: Run them and confirm they fail**

Run: `python3 -m pytest tests/unit/test_default_acronyms.py -q`

Expected: collection fails on `ImportError: cannot import name 'DEFAULT_ACRONYM_PATH'`. That is the correct first failure — the constant does not exist yet.

- [ ] **Step 3: Create the default table**

Create `src/internal/retrieval/acronyms.json`. This is the system's own
vocabulary, where expanding the acronym helps BM25 match this corpus. Generic
terms such as `API` are excluded on purpose: expanding them adds tokens without
making a query match anything it did not already match.

```json
{
  "RAG": "retrieval augmented generation",
  "IR": "information retrieval",
  "NLP": "natural language processing",
  "LLM": "large language model",
  "ANN": "approximate nearest neighbor",
  "KNN": "k nearest neighbors",
  "MRR": "mean reciprocal rank",
  "NDCG": "normalized discounted cumulative gain",
  "TF-IDF": "term frequency inverse document frequency",
  "RRF": "reciprocal rank fusion",
  "MMR": "maximal marginal relevance",
  "SFT": "supervised fine tuning",
  "GRPO": "group relative policy optimization",
  "DPO": "direct preference optimization",
  "PPO": "proximal policy optimization",
  "SERP": "search engine results page"
}
```

- [ ] **Step 4: Load it, and warn on an empty table**

In `src/internal/retrieval/query_optimizer.py`, add `from pathlib import Path`
if absent, and a module constant beside the other module-level names:

```python
# Bundled default acronym table. It lives here, not under data/, because that
# directory is gitignored -- a default file there would be invisible to every
# clone. ACRONYM_PATH overrides this file rather than merging with it.
DEFAULT_ACRONYM_PATH = Path(__file__).resolve().parent / "acronyms.json"
```

Then replace the loading block in `__init__`:

```python
        self._acronyms: dict[str, str] = {}
        if acronym_path:
            try:
                with open(acronym_path) as f:
                    self._acronyms = {k.upper(): v for k, v in json.load(f).items()}
            except FileNotFoundError:
                logger.warning("Acronym file not found: %s", acronym_path)
```

with:

```python
        self._acronyms: dict[str, str] = {}
        source = acronym_path or DEFAULT_ACRONYM_PATH
        try:
            with open(source) as f:
                self._acronyms = {k.upper(): v for k, v in json.load(f).items()}
        except FileNotFoundError:
            logger.warning("Acronym file not found: %s", source)
        except (ValueError, TypeError, AttributeError) as exc:
            # A malformed table must not take the process down; expansion
            # degrades to a no-op, and the warning below says so.
            logger.warning("Acronym file could not be read (%s): %s", source, exc)
        if not self._acronyms:
            logger.warning(
                "Query expansion is enabled but no acronyms were loaded from %s; "
                "expansion will have no effect.",
                source,
            )
```

Note the added `ValueError` arm: the previous code caught only
`FileNotFoundError`, so malformed JSON raised out of the constructor. Now that
a file is always read, a corrupt table must degrade rather than crash.

- [ ] **Step 5: Package the file**

In `pyproject.toml`, after the `[tool.setuptools.packages.find]` block, add:

```toml
[tool.setuptools.package-data]
"*" = ["*.json"]
```

Without this the file is present under `pip install -e .` and absent from a
built wheel, so the default would ship only sometimes and behavior would depend
on install method.

- [ ] **Step 6: Run the tests**

Run: `python3 -m pytest tests/unit/test_default_acronyms.py -v`

Expected: all PASS.

- [ ] **Step 7: Mutation-check each assertion**

Commit nothing yet; use file copies to restore, **not** `git checkout`, which
restores from the index and would discard uncommitted work.

```bash
cp src/internal/retrieval/query_optimizer.py /tmp/qo_good.py
cp src/internal/retrieval/acronyms.json /tmp/acr_good.json

# A: empty the default -> the "loads by default" and "well-formed" tests must fail
echo '{}' > src/internal/retrieval/acronyms.json
python3 -m pytest tests/unit/test_default_acronyms.py -q 2>&1 | tail -4
cp /tmp/acr_good.json src/internal/retrieval/acronyms.json

# B: make the override merge instead of replace -> the override test must fail
python3 - <<'PY'
import pathlib
p = pathlib.Path("src/internal/retrieval/query_optimizer.py")
s = p.read_text()
s = s.replace("        source = acronym_path or DEFAULT_ACRONYM_PATH",
              "        source = acronym_path or DEFAULT_ACRONYM_PATH\n"
              "        import json as _j\n"
              "        self._acronyms.update({k.upper(): v for k, v in _j.loads(DEFAULT_ACRONYM_PATH.read_text()).items()})")
p.write_text(s)
PY
python3 -m pytest tests/unit/test_default_acronyms.py -q 2>&1 | tail -3
cp /tmp/qo_good.py src/internal/retrieval/query_optimizer.py

# C: remove the warning -> the empty-table test must fail
python3 - <<'PY'
import pathlib, re
p = pathlib.Path("src/internal/retrieval/query_optimizer.py")
s = p.read_text()
s = re.sub(r'        if not self\._acronyms:\n(            .*\n)+', '', s)
p.write_text(s)
PY
python3 -m pytest tests/unit/test_default_acronyms.py -q 2>&1 | tail -3
cp /tmp/qo_good.py src/internal/retrieval/query_optimizer.py
rm /tmp/qo_good.py /tmp/acr_good.json

python3 -m pytest tests/unit/test_default_acronyms.py -q 2>&1 | tail -2
```

Expected: each mutation turns its matching test red, and the final run is green
again. A mutation that leaves everything green means that test is not testing
what it claims.

- [ ] **Step 8: Check the behavior change end to end**

```bash
python3 - <<'PY'
import os, importlib
os.environ["QUERY_EXPANSION_ENABLED"] = "true"
os.environ.pop("ACRONYM_PATH", None)
import src.internal.retrieval.query_optimizer as m
importlib.reload(m)
opt = m.QueryOptimizer.from_env()
for q in ["RAG systems", "compare MMR and RRF", "what is FAISS"]:
    print(f"  {q!r:26} -> {opt.expand(q)!r}")
PY
```

Expected: the first two expand, `what is FAISS` is unchanged (FAISS is not an
acronym in the table). This is the change an operator sees.

- [ ] **Step 9: Run the wider suites and commit**

```bash
python3 -m pytest tests/unit -k "optimizer or expansion or retrieval" -q
ruff check . --fix && ruff format .
git add src/internal/retrieval/acronyms.json src/internal/retrieval/query_optimizer.py \
        pyproject.toml tests/unit/test_default_acronyms.py
git commit -m "feat(retrieval): ship a default acronym table and warn when it is empty"
```

---

### Task 2: Documentation, verification, and the PR

**Files:**
- Modify: `docs/configuration.md`

- [ ] **Step 1: Update the rows #598 just corrected**

`docs/configuration.md` currently says `QUERY_EXPANSION_ENABLED` requires
`ACRONYM_PATH` and that `EXPANSION_MAX_TERMS` "only has an effect when
`ACRONYM_PATH` is set". Both are now false. Rewrite the three rows:

```markdown
| `QUERY_EXPANSION_ENABLED` | `false` | Enable acronym expansion in the BM25 leg, using the bundled default table |
| `ACRONYM_PATH` | bundled default | JSON file of `{"ACRONYM": "expansion"}`. Replaces the built-in table rather than extending it |
| `EXPANSION_MAX_TERMS` | `3` | Max acronym expansions added per query, to prevent BM25 query bloat |
```

- [ ] **Step 2: Replace the "needs an acronym file" paragraph**

That paragraph documented the silent no-op as behavior. It is no longer true.
Replace it with:

```markdown
**Query expansion ships with a default table.** `QUERY_EXPANSION_ENABLED=true`
expands acronyms from a table bundled at
`src/internal/retrieval/acronyms.json`, covering this system's own vocabulary
(`RAG`, `IR`, `NDCG`, `MMR`, `GRPO`, and similar). Setting `ACRONYM_PATH`
replaces that table rather than adding to it, so a custom file should include
any built-in entries it still wants. If the resulting table is empty — an
unreadable or empty custom file — expansion logs a warning and becomes a no-op
rather than failing silently.
```

- [ ] **Step 3: Verify the env-var guard still passes**

Run: `python3 -m pytest tests/unit/test_documented_env_vars.py -q`

Expected: 3 PASS. `ACRONYM_PATH` is still read in `src/`, so the guard added in
#598 is satisfied.

- [ ] **Step 4: Full suite**

Run: `python3 -m pytest -q 2>&1 | tail -3`

Expected: PASS at the previous count plus 9 (6 tests, one of them
parametrized 4 ways). Read the number rather than assuming it.

- [ ] **Step 5: Commit, push, open the PR**

```bash
git add docs/configuration.md docs/superpowers/
git commit -m "docs(config): query expansion now ships a default acronym table"
git push -u origin feat/default-acronym-expansion
gh pr create --title "feat(retrieval): give query expansion something to expand" --body "$(cat <<'BODY'
Closes #599.

## Summary

`QUERY_EXPANSION_ENABLED=true` with no `ACRONYM_PATH` was a silent no-op:

```
before: QUERY_EXPANSION_ENABLED=true, ACRONYM_PATH unset -> "ML and IR systems"
after:  QUERY_EXPANSION_ENABLED=true, ACRONYM_PATH unset -> "RAG systems retrieval augmented generation"
```

The mechanism was fine; it had an empty table. An operator had to discover an undocumented setting and author a JSON file before a flag named "enable query expansion" enabled query expansion.

Two changes:

**A bundled default table** at `src/internal/retrieval/acronyms.json` — 16 entries of this system's own vocabulary (`RAG`, `IR`, `NDCG`, `MMR`, `GRPO`, …), where expanding the acronym helps BM25 match this corpus. Generic terms like `API` are excluded on purpose: expanding them adds tokens without making a query match anything new. `EXPANSION_MAX_TERMS=3` caps how many land on any one query.

**A warning when the table is empty anyway**, matching the sibling path's `symspellpy not installed; spell correction disabled`. Two adjacent optional features, configured identically and disabled for the same kind of reason, now behave consistently.

## Two placement decisions worth naming

The file **cannot** live under `data/` — that directory is entirely gitignored, so a default there would be invisible to every clone. That is the same "advertised capability with nothing behind it" shape this change removes.

`pyproject.toml` gains `[tool.setuptools.package-data]`. Without it the file is present under `pip install -e .` (how this repo is developed) and absent from a built wheel, so the default would ship only sometimes and behavior would depend on install method.

`ACRONYM_PATH` **overrides** rather than merges. A caller who supplies a file gets exactly that file; merging would make the effective table depend on content they never wrote and cannot see. A test pins it: an acronym in the default and absent from a user file must not expand.

## Limits, stated

**This changes retrieval behavior** for anyone already running `QUERY_EXPANSION_ENABLED=true` — they move from a no-op to real BM25 query expansion. That is the point, and it is stated rather than buried.

**The default set is a starting point, not a tuned one.** No measurement here claims it improves retrieval; #588 measured a comparable heuristic and reported a null result, which is the standing reason not to assume. What this guarantees is that the flag does what its name says, and announces itself when it cannot. Whether acronym expansion is *worth enabling* is a separate, measurable question.

Loading also gained a `ValueError` arm: the old code caught only `FileNotFoundError`, so malformed JSON raised out of the constructor. Now that a file is always read, a corrupt table degrades to a warned no-op instead of taking the process down.

`docs/configuration.md` is updated — #598 documented the old silent-no-op behavior, which this makes obsolete.

All assertions mutation-checked: emptying the default, making the override merge, and removing the warning each turn the matching test red.

Spec: `docs/superpowers/specs/2026-09-20-default-acronym-expansion-design.md`
Plan: `docs/superpowers/plans/2026-09-20-default-acronym-expansion.md`

🤖 Generated with [Claude Code](https://claude.com/claude-code)
BODY
)"
```
