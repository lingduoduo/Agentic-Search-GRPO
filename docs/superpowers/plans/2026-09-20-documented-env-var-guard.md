# Documented Env Var Guard Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Correct six env vars that `configuration.md` documents but nothing reads, and add a guard so the table cannot drift silently again.

**Architecture:** Write the guard first and confirm it names exactly the six known-bad variables. Then correct the docs and watch it go green. The guard, not the six corrections, is the deliverable.

**Tech Stack:** Python 3, pytest, `docs/configuration.md`.

**Spec:** `docs/superpowers/specs/2026-09-20-documented-env-var-guard-design.md`

## Global Constraints

- **No production code changes.** Only `docs/` and one new test file. If a step seems to need `src/`, stop.
- Do not wire the four real capabilities to their documented env names. Documenting them at their real CLI interface is the fix; adding an env path is a feature and belongs in its own change.
- The allowlist must carry a *reason string* per entry, not a bare name. An exemption has to be argued.
- Run `ruff check . --fix && ruff format .` before each commit.

---

### Task 1: The guard, written against the broken state

**Files:**
- Create: `tests/unit/test_documented_env_vars.py`

**Interfaces:**
- Produces: `_documented_env_vars() -> set[str]` and `_referenced_env_vars() -> set[str]`, plus `ALLOWED_UNREFERENCED: dict[str, str]`.

- [ ] **Step 1: Write the guard**

```python
"""Every env var documented in configuration.md must be read somewhere.

Six documented variables did nothing when set, two of them inside
copy-pasteable startup commands. That fails the way a correct setting looks
when it happens not to matter -- no error, no warning -- so the operator
concludes the knob does not help rather than that they never turned it.
"""

import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
CONFIG_DOC = REPO / "docs" / "configuration.md"

# A documented variable read nowhere in the tree is a bug unless there is a
# reason. The value is that reason -- an exemption must be argued, not just
# added.
ALLOWED_UNREFERENCED: dict[str, str] = {}

_SEARCH_ROOTS = ("src", "examples", "tests", ".github")
_SEARCH_SUFFIXES = {".py", ".ts", ".tsx", ".sh", ".yml", ".yaml", ".go", ".toml"}

_DOC_ROW = re.compile(r"^\|\s*`([A-Z][A-Z0-9_]{3,})`\s*\|", re.M)
_TOKEN = re.compile(r"\b([A-Z][A-Z0-9_]{3,})\b")


def _documented_env_vars() -> set[str]:
    return set(_DOC_ROW.findall(CONFIG_DOC.read_text(encoding="utf-8")))


def _referenced_env_vars() -> set[str]:
    seen: set[str] = set()
    for root in _SEARCH_ROOTS:
        for path in (REPO / root).rglob("*"):
            if not path.is_file() or path.suffix not in _SEARCH_SUFFIXES:
                continue
            if "node_modules" in path.parts or "__pycache__" in path.parts:
                continue
            try:
                seen.update(_TOKEN.findall(path.read_text(encoding="utf-8", errors="ignore")))
            except OSError:
                continue
    return seen


def test_every_documented_env_var_is_read_somewhere():
    """A setting that nothing reads is not a setting."""
    documented = _documented_env_vars()
    assert len(documented) > 50, "the table parser found almost nothing; check the regex"

    unreferenced = sorted(
        name
        for name in documented - _referenced_env_vars()
        if name not in ALLOWED_UNREFERENCED
    )

    assert not unreferenced, (
        "documented in configuration.md but read nowhere in "
        f"{'/, '.join(_SEARCH_ROOTS)}/: {unreferenced}. "
        "Either fix the name, move it to where its real flag is documented, "
        "delete the row, or add it to ALLOWED_UNREFERENCED with a reason."
    )


def test_every_allowlist_entry_carries_a_reason():
    """An exemption must be argued, not just added."""
    for name, reason in ALLOWED_UNREFERENCED.items():
        assert reason.strip(), f"{name} is exempted without a reason"


def test_the_allowlist_has_no_stale_entries():
    """An entry that is now referenced, or no longer documented, must go."""
    documented = _documented_env_vars()
    referenced = _referenced_env_vars()
    for name in ALLOWED_UNREFERENCED:
        assert name in documented, f"{name} is exempted but no longer documented"
        assert name not in referenced, f"{name} is exempted but is now read; drop it"
```

- [ ] **Step 2: Confirm it names exactly the six known-bad variables**

Run: `python3 -m pytest tests/unit/test_documented_env_vars.py -q 2>&1 | tail -20`

Expected: `test_every_documented_env_var_is_read_somewhere` FAILS, and the
message lists exactly:

```
['BM25_VARIANT', 'EF_SEARCH', 'FAISS_INDEX_TYPE', 'LATENCY_SLO_MS',
 'RERANKER_USE_ONNX', 'SEARCH_DIRECT_COS_MIN']
```

Six, no more and no fewer. **This step is the proof the guard works** — a guard
that only passes after the fix has not demonstrated it detects anything. If the
list differs, the regex or the search roots are wrong; fix them before
touching any documentation.

The other two tests PASS trivially against an empty allowlist.

- [ ] **Step 3: Commit the guard, failing**

```bash
ruff check . --fix && ruff format .
git add tests/unit/test_documented_env_vars.py
git commit -m "test(docs): guard that a documented env var is read somewhere"
```

Committing it red is deliberate: the next commit's diff then shows the guard
turning green because the docs were corrected, not because the guard was
weakened.

---

### Task 2: Correct the six

**Files:**
- Modify: `docs/configuration.md`, `docs/retrieval.md`, `docs/request-routing.md`

- [ ] **Step 1: Give `SEARCH_DIRECT_COS_MIN` its prefix**

The code reads `AGENTIC_SEARCH_SEARCH_DIRECT_COS_MIN`
(`src/internal/utils/embedding_gate.py:20`). Rename it in all four places:

- `docs/configuration.md:63` (the table row)
- `docs/request-routing.md:145`, `:146`, `:288`

- [ ] **Step 2: Delete the two fictional rows**

From `docs/configuration.md`, delete these rows entirely:

```
| `RERANKER_USE_ONNX` | `false` | Load reranker via ONNX runtime (`ONNXReranker`) |
| `BM25_VARIANT` | — | Set to `bm25plus` to enable BM25+ lower-bound floor (`δ=1.0`) |
```

There is no ONNX anywhere in the repo, `ONNXReranker` does not exist, and no
BM25 variant selector exists.

- [ ] **Step 3: Remove them from the copy-pasteable commands**

In `docs/retrieval.md`, line ~493 sets `RERANKER_USE_ONNX=true` in a reranker
startup command, and line ~578 sets `BM25_VARIANT=bm25plus` in a query-expansion
command. Remove both assignments, keeping the surrounding commands valid — mind
the trailing backslashes so the shell continuation still parses.

- [ ] **Step 4: Move the three CLI-backed knobs out of the env table**

These describe real capabilities reached through CLI flags, not environment
variables. Delete their rows from `docs/configuration.md`:

```
| `FAISS_INDEX_TYPE` | `hnsw` | `ivfpq` for IVF-PQ quantized index; `hnsw` for original |
| `EF_SEARCH` | — | HNSW `ef_search` override (higher = more recall, slower) |
| `LATENCY_SLO_MS` | `120` | CI SLO gate: P99 above this exits non-zero in `eval_runner` |
```

and add, immediately below that table, a short paragraph naming the real
interface:

```markdown
**Not environment variables.** Three knobs that look like settings are CLI
flags instead. The FAISS index type is `--faiss_type` on the index-build CLI
(default `Flat`; it takes any FAISS `index_factory` string, so `HNSW32` and
IVF variants both work), HNSW search breadth is `--hnsw_ef_search` on the same
CLI, and the evaluation latency gate is `--slo-ms` (with `--qt-slo-ms` for the
query-transform leg) on `eval_runner`.
```

Note the corrected default: `--faiss_type` defaults to `Flat`, not `hnsw` as
the deleted row claimed.

- [ ] **Step 5: Run the guard**

Run: `python3 -m pytest tests/unit/test_documented_env_vars.py -q`

Expected: 3 PASS. If a name still appears, it was missed in `configuration.md` —
correct the doc rather than adding it to the allowlist, which exists for
externally-consumed variables and not for convenience.

- [ ] **Step 6: Mutation-check the guard**

```bash
python3 - <<'PY'
import pathlib
p = pathlib.Path("docs/configuration.md")
s = p.read_text()
s = s.rstrip() + "\n| `TOTALLY_FICTIONAL_SETTING` | `false` | does nothing at all |\n"
p.write_text(s)
PY
python3 -m pytest tests/unit/test_documented_env_vars.py -q 2>&1 | tail -4
git checkout docs/configuration.md
```

Expected: RED, naming `TOTALLY_FICTIONAL_SETTING`. Then restore.

**Careful:** `git checkout docs/configuration.md` restores from the index, so
Step 5's corrections must be staged or committed first, or this discards them.
Stage with `git add docs/` before running the mutation.

- [ ] **Step 7: Commit**

```bash
ruff check . --fix && ruff format .
git add docs/configuration.md docs/retrieval.md docs/request-routing.md
git commit -m "docs(config): six documented settings that did nothing"
```

---

### Task 3: The truncated path, verification, and the PR

**Files:**
- Modify: `docs/training-and-evaluation.md`

- [ ] **Step 1: Fix the truncated path**

`docs/training-and-evaluation.md:13` says the similarity adapter lives in
`web/intent/similarity.py`. It lives in
`src/internal/servers/web/intent/similarity.py`. In this repo `web/` is the
React frontend, so the short form sends a reader to the wrong place entirely.

- [ ] **Step 2: Re-run the qualified-path probe**

```bash
python3 - <<'PY'
import re, pathlib, subprocess, os
tracked = set(subprocess.run(["git","ls-files"],capture_output=True,text=True).stdout.split())
dirs = set()
for p in tracked:
    parts = pathlib.Path(p).parts
    for i in range(1, len(parts)):
        dirs.add(str(pathlib.Path(*parts[:i])))
pat = re.compile(r"`((?:src|tests|web|examples|cli|docs)/[\w./-]+)`")
bad = {}
for md in pathlib.Path("docs").rglob("*.md"):
    if "superpowers" in str(md):
        continue
    for m in pat.finditer(md.read_text(errors="ignore")):
        ref = m.group(1).rstrip("/")
        if ref in tracked or ref in dirs or os.path.exists(ref):
            continue
        bad.setdefault(ref, set()).add(str(md.relative_to("docs")))
for ref, files in sorted(bad.items()):
    print(f"  {ref:56} <- {', '.join(sorted(files))}")
PY
```

Expected: only `src/cli` (which `cli.md` names deliberately as a *deleted*
path) and `tests/integration/.env` (a file the reader is told to create).
Both are correct as written and must not be "fixed".

- [ ] **Step 3: Run the full suite**

Run: `python3 -m pytest -q 2>&1 | tail -3`

Expected: PASS at the previous count plus 3.

- [ ] **Step 4: Commit, push, open the PR**

```bash
git add docs/training-and-evaluation.md
git commit -m "docs(intents): name the similarity adapter's real path"
git push -u origin docs/env-var-accuracy-guard
gh pr create --title "docs(config): six documented settings did nothing, plus the guard that catches the next one" --body "$(cat <<'BODY'
## Summary

`configuration.md` documents 92 environment variables. **Six did nothing when set**, and two of those sat inside copy-pasteable startup commands in `retrieval.md` — so a reader following our own documentation set flags with no effect.

| Documented | Reality |
|---|---|
| `SEARCH_DIRECT_COS_MIN` | real, but the code reads `AGENTIC_SEARCH_SEARCH_DIRECT_COS_MIN` — wrong name |
| `LATENCY_SLO_MS` (default `120`) | the gate is real, but driven by `--slo-ms` / `--qt-slo-ms` — wrong mechanism, and no such default |
| `EF_SEARCH` | real, but the knob is `--hnsw_ef_search` at index-build time |
| `FAISS_INDEX_TYPE` (default `hnsw`) | real, but the flag is `--faiss_type`, the default is `Flat`, and it takes any `index_factory` string |
| `RERANKER_USE_ONNX` | fiction — no ONNX in the repo; the `ONNXReranker` it names does not exist |
| `BM25_VARIANT` | fiction — no BM25+ variant anywhere |

This is the shape #590 and #591 both found: an advertised capability with nothing behind it. It is an expensive kind of error because it fails exactly the way a correct setting looks when it happens not to matter — no error, no warning, no log. The operator concludes the knob does not help, rather than that they never turned it.

Four of the six are worse than fiction: the capability is real, just reached another way. Someone who sets `EF_SEARCH`, measures nothing, and moves on has learned something false about HNSW rather than something true about our docs.

## The guard is the actual deliverable

Correcting six rows without one only resets the clock — this drifted precisely because nothing checked. So the first commit adds a failing test asserting every documented variable is read somewhere in `src/`, `examples/`, `tests/` or `.github/workflows/`, and it names exactly those six. The second commit corrects the docs and turns it green.

The guard is committed **red on purpose**, so the diff shows it going green because the documentation was fixed, not because the guard was weakened.

It carries an allowlist for variables legitimately consumed outside the tree, keyed by name with the **reason** as the value, so an exemption has to be argued. Two further tests keep that honest: every entry must carry a non-empty reason, and an entry that is now read, or no longer documented, fails as stale.

Mutation-checked by adding a fictional row and confirming it goes red.

## Limits, stated rather than papered over

The guard proves a documented variable is *read*. It cannot prove the reading is meaningful — a variable read only to report itself, the `ADAPTIVE_MMR` shape removed in #591, would still pass.

The four real capabilities are documented at their real interface rather than wired to the documented env names. Adding an env path to `--slo-ms` or `--faiss_type` is a feature, and a documentation-accuracy PR should not smuggle one in.

## Also

`training-and-evaluation.md` said the similarity adapter lives in `web/intent/similarity.py`. It lives in `src/internal/servers/web/intent/similarity.py` — and in this repo `web/` is the React frontend, so the short form sent readers somewhere else entirely.

Docs and one test only. No production code, no behavior change.

Spec: `docs/superpowers/specs/2026-09-20-documented-env-var-guard-design.md`
Plan: `docs/superpowers/plans/2026-09-20-documented-env-var-guard.md`

🤖 Generated with [Claude Code](https://claude.com/claude-code)
BODY
)"
```
