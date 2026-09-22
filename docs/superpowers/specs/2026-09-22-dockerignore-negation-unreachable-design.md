# The corpus fix in #622 did not work; a real build says so

## Goal

Correct the `.dockerignore` shipped in #622, which left the corpus out of the
image it was meant to put there, and fix the test that let it through.

## What shipped, and why it was wrong

#622 pointed the retrieval service at `/app/data/corpus.jsonl` and added
negations so the tracked corpus survives into the image:

```
data
# ...but the demo stack serves these, and they are tracked in git.
!data/corpus.jsonl
!data/corpora.json
```

A real `docker build` reports `CORPUS: MISSING`. **Docker does not descend into
an excluded directory**, so a bare `data` exclusion prunes the walk and the
negations below it are never reached. Excluding the directory's *contents*
instead leaves them reachable:

```
data/*
!data/corpus.jsonl
!data/corpora.json
```

Same build, after: `CORPUS: present (11114 bytes)`, `REGISTRY: present`,
`DATA ENTRIES: corpora.json corpus.jsonl` — the two negated files present, the
rest of `data/` still excluded.

## Why the test passed anyway

This is the more useful half of the lesson. `test_docker_stack_contract.py`
implemented `.dockerignore` matching as "last matching rule wins", which is what
the documentation says and is not what the builder does. The helper was a
restatement of my own assumption, so the test agreed with the code for the same
wrong reason and reported the fix as verified.

A unit test that models an external tool's semantics is only as good as the
model. Two things fix that here:

- `_excluded_by` now implements the pruning rule: an ancestor excluded *as a
  directory* means nothing under it can be re-included, and only then does
  last-match-wins apply to the remainder.
- `test_dockerignore_semantics_match_observed_docker_behaviour` pins the helper
  against both outcomes actually observed from `docker build` — the bare-`data`
  form excludes, the `data/*` form includes — so the model cannot drift back into
  being a guess.

## What the build did confirm

The same probe verified #622's other fix end to end, inside a real image:

```
FINDER MAPPING: dict[str, str] = {'src': '/app/src'}
IMPORT src FROM /: /app/src/__init__.py
```

The editable install now registers the package, and `import src` resolves from a
working directory that is not `/app` — which was the defect. That half of #622
was right.

## Method note

The probe builds on `python:3.11-slim` and skips `requirements.txt`, because the
real image pulls torch, transformers, faiss and pyserini. It still exercises the
repo's actual `.dockerignore` — docker applies that to the build context
regardless of which Dockerfile runs — and the actual `pip install -e .` against
the copied source, which is everything the two fixes turn on.

`.dockerignore` semantics are exactly the kind of thing the contract tests in
#622 cannot settle alone. The PR said as much ("not verified by a real docker
build"); the gap was real and this is what was in it.

## Testing

The corrected corpus test fails against the `.dockerignore` that shipped —
mutation-checked by reverting `data/*` to `data` — and passes against the fix.
The new semantics test grounds the helper in build-observed behaviour in both
directions.
