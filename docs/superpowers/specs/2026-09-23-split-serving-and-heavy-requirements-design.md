# Split the serving baseline from the backend and training dependencies

## Goal

Stop the container installing dependencies no serving path can use, on a
boundary that reflects what the code actually needs rather than the labels the
file already carried.

## Starting point

`requirements.txt` declared 31 packages and already had section comments —
"Server/runtime dependencies", "Training / data / retrieval dependencies".
Nothing enforced them, and the image had been shipping `faiss-cpu`, `pyserini`,
`datasets`, `mlx-lm`, `playwright` and `pyarrow` regardless.

#627 cut the image from 7.64GB to 2.9GB by installing a CPU-only torch wheel.
This addresses what remains.

## Classifying all 31, and three corrections

An import scan over `src/`, `examples/` and `tests/` — with a pattern that
catches this repo's many function-local imports, because a line-anchored one
reports `pyserini` as unused while missing its deferred import:

| Class | Packages |
|---|---|
| Imported on the serving path | 24, including `scikit-learn`, `sentence-transformers`, `torch`, `transformers` |
| Backend-specific, deferred imports | `faiss-cpu`, `pyserini` |
| Examples-only | `datasets` (three training scripts), `pyarrow` (its dependency) |
| Required, never imported | `python-multipart`, `pytest-asyncio`, `urllib3` |
| Genuinely unimported | `mlx-lm` |
| Wrong package declared | `playwright` |

Three earlier claims of mine were wrong and are corrected here:

**`faiss-cpu` and `pyserini` are not training-only.** Both are reachable from the
serving path: `document_index/_common.py` (faiss) and
`document_index/retrieval.py` (pyserini) → `internal/retrieval/service.py` →
`servers/retrieval/server.py`. They are dependencies of a *backend*, not of
training. **So the axis is deployment shape, not serving-versus-training** — which
is why the split is by what a deployment selects rather than by lifecycle.

**`python-multipart` is required**, though nothing imports it: FastAPI needs it
for the `UploadFile` routes in `servers/enterprise_settings/api.py`. An import
scan calls it dead.

**`peft` and `matplotlib`, which I previously named as unused, are not declared
at all.** They came from a local environment, not from this repo.

## The split

`requirements.txt` becomes the serving baseline and stays the only file the
container installs. Two new companions, matching the existing
`requirements-unit-test.txt` convention:

- **`requirements-retrieval-heavy.txt`** — `faiss-cpu`, `pyserini`. Needed only by
  `servers/retrieval/server.py` under a FAISS or BM25 `RETRIEVAL_BACKEND`. The
  compose stack runs `demo.py` (scikit-learn TF-IDF) or `hybrid.py`
  (sentence-transformers + TF-IDF) and needs neither. pyserini also needs a JVM
  the image does not install, and pins `torch>=2.9`.
- **`requirements-training.txt`** — `datasets`, `pyarrow`.

`mlx-lm` and `playwright` are **dropped, not moved**. `mlx-lm` appears only in a
`serving.py` error string suggesting `mlx_lm.server`, an external process rather
than a library. `playwright` was the wrong package: `web_search/browser.py`
shells out to a `playwright-cli` binary through `subprocess`, which this wheel
does not provide — so declaring it satisfied nothing, and the real dependency
remains undeclared.

## Why this is safe

The removals are deferred imports, and there is already standing evidence:
**`requirements-unit-test.txt` has never carried `faiss-cpu`, `pyserini`,
`datasets`, `mlx-lm` or `playwright`, and the full suite passes on it.** `faiss`
is imported inside `_require_faiss()`; pyserini inside a function at
`retrieval.py:283`. Their absence degrades those backends rather than breaking
startup.

Only the Dockerfile and `CLAUDE.md` read `requirements.txt`; CI's unit-test job
reads `requirements-unit-test.txt`, so the split cannot affect it.

## Testing

`tests/unit/test_requirements_layout.py` pins the split rather than the contents:
the heavy backends and training packages stay out of the baseline and are present
in their companion files, the two dropped packages stay dropped, the three
required-but-never-imported packages stay in, and the serving path's own imports
are still declared. A sixth test records that the unit-test install never carried
the heavy set, which is the evidence the rest rests on.

Mutation-checked: re-adding `faiss-cpu` to the baseline fails the first test.

The image size is measured by the `Docker image (full build)` CI job from #625,
against the 2.9GB baseline #627 established.
