# Corpus-Derived Acronyms Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Derive the acronym table from the loaded corpus, so expansion is domain-correct instead of assuming this repository's vocabulary.

**Architecture:** A new extractor module, then a three-source precedence in `QueryOptimizer`: explicit `ACRONYM_PATH`, else corpus extraction, else the bundled table.

**Tech Stack:** Python 3, pytest, `src/internal/retrieval/query_optimizer.py`.

**Spec:** `docs/superpowers/specs/2026-09-20-corpus-derived-acronyms-design.md`

Refs #601.

## Global Constraints

- `ACRONYM_PATH` semantics are **unchanged**: when set it is used exactly, overriding both other sources. #600's tests pin this and must stay green.
- The fallback fires **only** on an empty extraction. One extracted pair is enough to prefer the derived table.
- Extraction must never raise. A missing, unreadable or malformed corpus degrades to the bundled table.
- `QUERY_EXPANSION_ENABLED` stays `false` by default, so nothing here runs on the default path.
- Run `ruff check . --fix && ruff format .` before each commit.

---

### Task 1: The extractor

**Files:**
- Create: `src/internal/retrieval/acronym_extraction.py`
- Test: `tests/unit/test_acronym_extraction.py`

**Interfaces:**
- Produces: `extract_acronyms(text: str) -> dict[str, str]` and `extract_acronyms_from_corpus(path) -> dict[str, str]`, both returning `{UPPER_ACRONYM: lower expansion}`.

- [ ] **Step 1: Write the failing tests**

```python
"""Acronym pairs a corpus defines about itself.

A static table assumes a domain: scifact uses `IR` for ionizing radiation,
this repository uses it for information retrieval. The corpus already carries
the answer in the ordinary gloss `ionizing radiation (IR)`. See #601.
"""

import json

import pytest

from src.internal.retrieval.acronym_extraction import (
    extract_acronyms,
    extract_acronyms_from_corpus,
)


def test_a_glossed_acronym_is_extracted():
    text = "increased sensitivity to ionizing radiation (IR) was observed"

    assert extract_acronyms(text) == {"IR": "ionizing radiation"}


def test_the_initials_must_match():
    """Without this check the pattern matches any parenthesised capital."""
    assert extract_acronyms("some random words (XYZ) appear here") == {}


def test_a_citation_is_not_an_acronym():
    """The shape that makes the naive pattern useless on real prose."""
    assert extract_acronyms("as shown by Pettifor (BBC) in 2012") == {}


def test_multi_word_expansions_are_bounded_to_the_matching_initials():
    text = "we treated acute myeloid leukemia (AML) in the cohort"

    assert extract_acronyms(text) == {"AML": "acute myeloid leukemia"}


def test_the_first_definition_wins():
    text = "ionizing radiation (IR) ... information retrieval (IR)"

    assert extract_acronyms(text) == {"IR": "ionizing radiation"}


def test_a_corpus_yields_its_own_vocabulary(tmp_path):
    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text(
        "\n".join(
            json.dumps(r)
            for r in [
                {"title": "t1", "text": "hepatitis c virus (HCV) prevalence"},
                {"title": "t2", "text": "receptor tyrosine kinase (RTK) signalling"},
            ]
        )
    )

    pairs = extract_acronyms_from_corpus(corpus)

    assert pairs == {"HCV": "hepatitis c virus", "RTK": "receptor tyrosine kinase"}


@pytest.mark.parametrize("bad", ["/nonexistent/corpus.jsonl", None])
def test_an_unusable_corpus_yields_nothing_rather_than_raising(bad):
    assert extract_acronyms_from_corpus(bad) == {}


def test_malformed_lines_are_skipped(tmp_path):
    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text('not json\n{"text": "hepatitis c virus (HCV) here"}\n')

    assert extract_acronyms_from_corpus(corpus) == {"HCV": "hepatitis c virus"}
```

- [ ] **Step 2: Run them and confirm they fail**

Run: `python3 -m pytest tests/unit/test_acronym_extraction.py -q`
Expected: collection fails — the module does not exist yet.

- [ ] **Step 3: Write the extractor**

Create `src/internal/retrieval/acronym_extraction.py`:

```python
"""Extract the acronym pairs a corpus defines about itself.

A static acronym table assumes a domain. scifact uses `IR` for ionizing
radiation; this repository uses it for information retrieval. Rather than
guess, read the gloss the corpus already writes: `ionizing radiation (IR)`.

Only the `long form (ABBR)` shape is recognised, and only when the
abbreviation's letters match the initials of the trailing words -- without that
check the pattern matches any parenthesised capital, which is most citations.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

logger = logging.getLogger(__name__)

# A short run of words, then a parenthesised candidate abbreviation. The word
# cap bounds the backtracking and reflects that glosses are short.
_GLOSS = re.compile(
    r"([A-Za-z][A-Za-z\-]*(?:\s+[A-Za-z][A-Za-z\-]*){0,4})\s*\(([A-Z][A-Za-z0-9\-]{1,7})\)"
)

_MAX_TEXT_CHARS = 20_000


def extract_acronyms(text: str) -> dict[str, str]:
    """Return ``{ACRONYM: expansion}`` for every verified gloss in *text*."""
    pairs: dict[str, str] = {}
    for long_form, abbreviation in _GLOSS.findall(text[:_MAX_TEXT_CHARS]):
        words = long_form.split()
        target = abbreviation.lower().replace("-", "")
        for count in range(1, len(words) + 1):
            candidate = words[-count:]
            if "".join(w[0].lower() for w in candidate) == target:
                pairs.setdefault(abbreviation.upper(), " ".join(candidate).lower())
                break
    return pairs


def extract_acronyms_from_corpus(path) -> dict[str, str]:
    """Extract from a corpus.jsonl. Never raises: an unusable corpus yields {}."""
    if not path:
        return {}
    corpus = Path(path)
    if not corpus.is_file():
        logger.debug("Acronym extraction: no corpus at %s", corpus)
        return {}

    pairs: dict[str, str] = {}
    try:
        with corpus.open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except ValueError:
                    continue  # one bad line must not lose the whole corpus
                if not isinstance(record, dict):
                    continue
                text = f"{record.get('title', '')} {record.get('text', '')}"
                for acronym, expansion in extract_acronyms(text).items():
                    pairs.setdefault(acronym, expansion)
    except OSError as exc:
        logger.warning("Acronym extraction failed to read %s: %s", corpus, exc)
        return {}
    return pairs
```

- [ ] **Step 4: Run the tests**

Run: `python3 -m pytest tests/unit/test_acronym_extraction.py -v`
Expected: 9 PASS.

- [ ] **Step 5: Mutation-check the initials verification**

It is the load-bearing part; without it the extractor is noise.

```bash
cp src/internal/retrieval/acronym_extraction.py /tmp/ae_good.py
python3 - <<'PY'
import pathlib
p = pathlib.Path("src/internal/retrieval/acronym_extraction.py")
s = p.read_text()
s = s.replace('if "".join(w[0].lower() for w in candidate) == target:',
              'if True:')
p.write_text(s)
PY
python3 -m pytest tests/unit/test_acronym_extraction.py -q 2>&1 | tail -4
cp /tmp/ae_good.py src/internal/retrieval/acronym_extraction.py && rm /tmp/ae_good.py
```

Expected: `test_the_initials_must_match` and `test_a_citation_is_not_an_acronym` RED.

- [ ] **Step 6: Check it against the real corpus**

```bash
python3 -c "
from src.internal.retrieval.acronym_extraction import extract_acronyms_from_corpus
pairs = extract_acronyms_from_corpus('data/corpus_scifact.jsonl')
print(f'scifact -> {len(pairs)} pairs')
for a, e in list(pairs.items())[:6]:
    print(f'  {a:8} -> {e}')
print('demo ->', len(extract_acronyms_from_corpus('data/corpus.jsonl')), 'pairs')
"
```

Expected: scifact yields around 16 domain-correct biomedical pairs; the demo
corpus yields 0. If scifact yields hundreds, the verification regressed.

- [ ] **Step 7: Commit**

```bash
ruff check . --fix && ruff format .
git add src/internal/retrieval/acronym_extraction.py tests/unit/test_acronym_extraction.py
git commit -m "feat(retrieval): extract the acronyms a corpus defines about itself"
```

---

### Task 2: Wire the precedence into QueryOptimizer

**Files:**
- Modify: `src/internal/retrieval/query_optimizer.py`
- Test: `tests/unit/test_default_acronyms.py` (extend)

**Interfaces:**
- Consumes: `extract_acronyms_from_corpus` from Task 1.
- Produces: `QueryOptimizer(acronym_path=None, corpus_path=...)`; `from_env` resolves the corpus from `ACRONYM_CORPUS_PATH`, then `BM25_CORPUS_PATH`, then `data/corpus.jsonl`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/unit/test_default_acronyms.py`:

```python
def _corpus(tmp_path, *texts):
    import json as _json

    path = tmp_path / "corpus.jsonl"
    path.write_text("\n".join(_json.dumps({"title": "", "text": t}) for t in texts))
    return path


def test_a_corpus_derived_table_beats_the_bundled_one(tmp_path):
    """HCV is not in the bundle; RAG is. The corpus wins outright, not merged."""
    corpus = _corpus(tmp_path, "hepatitis c virus (HCV) prevalence rose")

    optimizer = QueryOptimizer(None, corpus_path=corpus)

    assert optimizer.expand("HCV rates") == "HCV rates hepatitis c virus"
    assert optimizer.expand("RAG systems") == "RAG systems"


def test_an_empty_extraction_falls_back_to_the_bundle(tmp_path):
    corpus = _corpus(tmp_path, "this text defines no acronyms at all")

    optimizer = QueryOptimizer(None, corpus_path=corpus)

    assert optimizer.expand("RAG systems") == "RAG systems retrieval augmented generation"


def test_an_explicit_acronym_path_still_overrides_the_corpus(tmp_path):
    import json as _json

    corpus = _corpus(tmp_path, "hepatitis c virus (HCV) prevalence rose")
    user_file = tmp_path / "user.json"
    user_file.write_text(_json.dumps({"ZZZ": "zulu zulu"}))

    optimizer = QueryOptimizer(str(user_file), corpus_path=corpus)

    assert optimizer.expand("ZZZ here") == "ZZZ here zulu zulu"
    assert optimizer.expand("HCV rates") == "HCV rates"


def test_the_demo_corpus_falls_back(tmp_path):
    """Pins the real-world case: 20 documents define nothing."""
    optimizer = QueryOptimizer(None, corpus_path="data/corpus.jsonl")

    assert optimizer.expand("RAG systems") == "RAG systems retrieval augmented generation"
```

- [ ] **Step 2: Run them and confirm they fail**

Run: `python3 -m pytest tests/unit/test_default_acronyms.py -q`
Expected: the four new tests fail on `TypeError: unexpected keyword argument 'corpus_path'`.

- [ ] **Step 3: Add the precedence**

In `src/internal/retrieval/query_optimizer.py`, extend `__init__`'s signature
with `corpus_path=None` after `acronym_path`, and replace the loading block:

```python
        self._acronyms: dict[str, str] = {}
        source = acronym_path or DEFAULT_ACRONYM_PATH
        try:
            ...
```

with:

```python
        # Three sources, strict precedence: an explicit file wins; otherwise the
        # corpus defines its own vocabulary; otherwise the bundled table. A
        # static table assumes a domain -- scifact's IR is ionizing radiation,
        # not information retrieval. See #601.
        self._acronyms: dict[str, str] = {}
        source = "none"
        if acronym_path:
            source = str(acronym_path)
            self._acronyms = _read_acronym_file(acronym_path)
        else:
            from .acronym_extraction import extract_acronyms_from_corpus

            derived = extract_acronyms_from_corpus(corpus_path)
            if derived:
                self._acronyms = derived
                source = f"corpus:{corpus_path}"
            else:
                self._acronyms = _read_acronym_file(DEFAULT_ACRONYM_PATH)
                source = f"bundled:{DEFAULT_ACRONYM_PATH.name}"
        if self._acronyms:
            logger.info(
                "Query expansion: %d acronyms from %s", len(self._acronyms), source
            )
        else:
            logger.warning(
                "Query expansion is enabled but no acronyms were loaded from %s; "
                "expansion will have no effect.",
                source,
            )
```

and add the helper beside `DEFAULT_ACRONYM_PATH`:

```python
def _read_acronym_file(path) -> dict[str, str]:
    """Read an acronym JSON file. Never raises: an unusable file yields {}."""
    try:
        with open(path) as handle:
            return {k.upper(): v for k, v in json.load(handle).items()}
    except FileNotFoundError:
        logger.warning("Acronym file not found: %s", path)
    except (ValueError, TypeError, AttributeError) as exc:
        logger.warning("Acronym file could not be read (%s): %s", path, exc)
    return {}
```

- [ ] **Step 4: Resolve the corpus in `from_env`**

In `from_env`, pass the corpus path through:

```python
        return cls(
            os.environ.get("ACRONYM_PATH"),
            corpus_path=(
                os.environ.get("ACRONYM_CORPUS_PATH")
                or os.environ.get("BM25_CORPUS_PATH")
                or "data/corpus.jsonl"
            ),
            max_terms=int(os.environ.get("EXPANSION_MAX_TERMS", "3")),
            spell_enabled=spell,
        )
```

- [ ] **Step 5: Run the whole file**

Run: `python3 -m pytest tests/unit/test_default_acronyms.py -v`
Expected: all PASS — the four new tests and #600's five, which pin the explicit
and bundled paths and must not have moved.

- [ ] **Step 6: Mutation-check the precedence**

```bash
cp src/internal/retrieval/query_optimizer.py /tmp/qo_good.py

# A: reverse the precedence -- bundle before corpus
python3 - <<'PY'
import pathlib
p = pathlib.Path("src/internal/retrieval/query_optimizer.py")
s = p.read_text()
s = s.replace("            derived = extract_acronyms_from_corpus(corpus_path)",
              "            derived = {}")
p.write_text(s)
PY
python3 -m pytest tests/unit/test_default_acronyms.py -q 2>&1 | tail -3
cp /tmp/qo_good.py src/internal/retrieval/query_optimizer.py

# B: delete the fallback
python3 - <<'PY'
import pathlib
p = pathlib.Path("src/internal/retrieval/query_optimizer.py")
s = p.read_text()
s = s.replace("                self._acronyms = _read_acronym_file(DEFAULT_ACRONYM_PATH)",
              "                self._acronyms = {}")
p.write_text(s)
PY
python3 -m pytest tests/unit/test_default_acronyms.py -q 2>&1 | tail -3
cp /tmp/qo_good.py src/internal/retrieval/query_optimizer.py && rm /tmp/qo_good.py
python3 -m pytest tests/unit/test_default_acronyms.py -q 2>&1 | tail -2
```

Expected: A turns `test_a_corpus_derived_table_beats_the_bundled_one` red; B
turns the two fallback tests red; the restore is green.

- [ ] **Step 7: Commit**

```bash
ruff check . --fix && ruff format .
git add src/internal/retrieval/query_optimizer.py tests/unit/test_default_acronyms.py
git commit -m "feat(retrieval): prefer the corpus's own acronyms over the bundled table"
```

---

### Task 3: Documentation, verification, and the PR

**Files:**
- Modify: `docs/configuration.md`

- [ ] **Step 1: Document the new source and the precedence**

`configuration.md` currently documents `ACRONYM_PATH` as "bundled default".
Update that row and add `ACRONYM_CORPUS_PATH`:

```markdown
| `ACRONYM_PATH` | — | JSON file of `{"ACRONYM": "expansion"}`. Overrides both the corpus-derived and bundled tables |
| `ACRONYM_CORPUS_PATH` | `BM25_CORPUS_PATH`, else `data/corpus.jsonl` | Corpus scanned for `long form (ABBR)` glosses to build the acronym table |
```

Then replace the "ships with a default table" paragraph:

```markdown
**Query expansion reads the corpus first.** With `QUERY_EXPANSION_ENABLED=true`
the acronym table is built from the corpus at `ACRONYM_CORPUS_PATH` by reading
the glosses it already contains — `ionizing radiation (IR)` in a medical corpus
gives `IR -> ionizing radiation` there, rather than this repository's
`information retrieval`. A corpus that defines nothing falls back to the
bundled table at `src/internal/retrieval/acronyms.json`, and setting
`ACRONYM_PATH` overrides both. The log line at startup names which source was
used and how many acronyms it holds.
```

- [ ] **Step 2: Verify the env-var guard still passes**

Run: `python3 -m pytest tests/unit/test_documented_env_vars.py -q`

Expected: 3 PASS. `ACRONYM_CORPUS_PATH` is read in `from_env`, so the guard
added in #598 is satisfied — and would fail if the row were added without the
code.

- [ ] **Step 3: Full suite**

Run: `python3 -m pytest -q 2>&1 | tail -3`
Expected: PASS at the previous count plus 13 (9 extraction, 4 precedence).

- [ ] **Step 4: Demonstrate the point end to end**

```bash
python3 - <<'PY'
from src.internal.retrieval.query_optimizer import QueryOptimizer
for label, corpus in (("scifact (medical)", "data/corpus_scifact.jsonl"),
                      ("demo (20 docs)", "data/corpus.jsonl")):
    opt = QueryOptimizer(None, corpus_path=corpus)
    print(f"{label:20} {len(opt._acronyms):3} acronyms")
    print(f"   'HCV prevalence' -> {opt.expand('HCV prevalence')!r}")
    print(f"   'RAG systems'    -> {opt.expand('RAG systems')!r}")
PY
```

Expected: scifact expands `HCV` and leaves `RAG` alone; the demo corpus falls
back and expands `RAG`. That contrast is the whole change.

- [ ] **Step 5: Commit, push, open the PR**

```bash
git add docs/configuration.md docs/superpowers/
git commit -m "docs(config): query expansion reads the corpus before the bundle"
git push -u origin feat/corpus-derived-acronyms
gh pr create --title "feat(retrieval): derive the acronym table from the corpus" --body "..."
```
