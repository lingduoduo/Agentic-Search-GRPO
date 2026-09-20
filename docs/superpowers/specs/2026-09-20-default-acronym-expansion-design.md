# Give query expansion something to expand

Closes #599.

## Goal

Make `QUERY_EXPANSION_ENABLED=true` do something on its own, and say so out
loud when it cannot.

## The problem

`QUERY_EXPANSION_ENABLED=true` with no `ACRONYM_PATH` is a silent no-op:

```
QUERY_EXPANSION_ENABLED=true, ACRONYM_PATH unset : "ML and IR systems"
QUERY_EXPANSION_ENABLED=true, ACRONYM_PATH set   : "ML and IR systems machine learning information retrieval"
```

`QueryOptimizer.from_env` sees `expansion=True`, declines to return the
passthrough optimizer, and builds a real one whose `_acronyms` dict is empty
because nothing supplied a file. `expand()` then finds nothing to substitute
and returns the query unchanged. No error, no warning, no log line.

Two things are wrong with that, and they are separable.

**The feature has no content.** The mechanism is fine; it simply has an empty
table. An operator has to discover an undocumented setting and author a JSON
file before a flag named "enable query expansion" enables query expansion.

**The failure is silent, and its sibling is not.** `SPELL_CORRECTION_ENABLED`
logs `symspellpy not installed; spell correction disabled` when its dependency
is missing. Two adjacent optional features, configured the same way and
disabled for the same kind of reason — a missing external resource — behave
inconsistently. The silent one is also the one that misleads: an operator who
enables expansion, measures no improvement, and concludes expansion does not
help has learned something false.

## Architecture

**A bundled default table.** `src/internal/retrieval/acronyms.json`, tracked
and shipped with the package. It cannot live under `data/`, which is entirely
gitignored — a default file there would be invisible to every clone, which is
the same "advertised capability with nothing behind it" shape this change
exists to remove. `src/internal/llm/model_metadata_enrichments.json` is the
precedent for bundled JSON.

`pyproject.toml` gains a `[tool.setuptools.package-data]` entry. Without it the
file is present for `pip install -e .`, which is how this repo is developed, and
absent from a built wheel — a default that ships only sometimes is worse than
no default, because the behavior then depends on install method.

**Override, not merge.** `ACRONYM_PATH` when set replaces the default rather
than extending it. A caller who supplies a file gets exactly that file, which
is predictable and explains itself; merging would make the effective table
depend on content the caller never wrote and cannot see.

**A warning when the table is empty anyway.** After the default, an empty table
means a user file that was missing, unreadable, or contained nothing. That is
worth one log line at construction, matching the spell-correction precedent.

**The default set** is this system's own vocabulary, where expanding the
acronym helps BM25 match this corpus:

```
RAG IR NLP LLM ANN KNN MRR NDCG TF-IDF RRF MMR SFT GRPO DPO PPO SERP
```

Generic terms are deliberately excluded. Expanding `API` to "application
programming interface" adds tokens to any query mentioning an API without
making it match anything it did not already match. `EXPANSION_MAX_TERMS`
defaults to 3, so at most three expansions land on a query regardless.

## Testing

Behavior, not wiring:

- With `ACRONYM_PATH` unset and expansion on, a query containing a default
  acronym comes back expanded. This is the whole point, and it fails today.
- A user-supplied file overrides the default: an acronym present in the default
  and absent from the user file is **not** expanded.
- The warning fires when expansion is on and the table is empty, and does not
  fire when the default loaded.
- The bundled file is valid JSON, every key is upper-case, and every value is
  a non-empty string that differs from its key.

Each is mutation-checked: emptying the default file, making the override merge
instead of replace, and removing the warning must each turn the matching test
red.

## Limits

This changes retrieval behavior for anyone already running
`QUERY_EXPANSION_ENABLED=true`, who moves from a no-op to real BM25 query
expansion. That is the point of the change, and it is stated rather than
buried.

**The default set is a starting point, not a tuned one.** No measurement claims
it improves retrieval. #588 measured a comparable retrieval heuristic and
reported a null result, which is the standing reason not to assume. What this
change guarantees is that the flag does what its name says and announces itself
when it cannot — whether acronym expansion is *worth enabling* is a separate,
measurable question this does not answer.

Out of scope: expanding beyond acronyms (synonyms, WordNet — which never
existed here, see #598), tuning the default set against a benchmark, and
whether `QUERY_EXPANSION_ENABLED` should default to true, which it does not.
