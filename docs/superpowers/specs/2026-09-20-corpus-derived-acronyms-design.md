# Derive the acronym table from the corpus

Refs #601.

## Goal

Make acronym expansion domain-correct for whatever corpus is loaded, instead of
shipping one fixed table that is right for this repository and wrong elsewhere.

## The problem

#600 shipped a default acronym table of this system's own vocabulary — `RAG`,
`IR`, `NDCG`, `MMR`, `GRPO`. Measuring it for #601 produced a better result
than the measurement it was after.

Across roughly 2,700 BEIR queries the table fires three times, and **every
firing is wrong**:

```
scifact  IR  -> "...reveal increased sensitivity to ionizing radiation (IR)..."
arguana  ANN -> "...[1] Pettifor, Ann: 'Greece: The upside of default'..."
```

`IR` is ionizing radiation. `ANN` is a person's name. In both cases the table
would inject "information retrieval" or "approximate nearest neighbor" into a
query about radiation biology or Greek sovereign default.

Three instances is not a harm rate, but it identifies a property that needs no
benchmark: **a static table assumes a domain.** Short acronyms collide across
domains — `MMR` is also the measles-mumps-rubella vaccine, which matters
because one of the supported benchmarks is a nutrition and medical corpus;
`PPO` is also a preferred provider organization; `LLM` is also a Master of Laws.

The deciding observation is that the corpus already contains the answer.
scifact does not merely use `IR` — it *defines* it, in the ordinary academic
gloss `ionizing radiation (IR)`. A table derived from the corpus is correct for
that corpus by construction.

## Architecture

Three sources, in strict precedence:

```
ACRONYM_PATH set          -> exactly that file      (explicit wins; unchanged)
else corpus extraction    -> the derived table      (new)
else                      -> the bundled table      (fallback)
```

**Extraction** is the Schwartz-Hearst shape: find `long form (ABBR)`, then keep
the pair only when the abbreviation's letters match the initials of the
trailing words. That verification is what makes the output usable — over
scifact's 5,183 documents it yields 16 pairs and no garbage:

```
RTK  -> receptor tyrosine kinase        HCV  -> hepatitis c virus
CETP -> cholesteryl ester transfer protein
AML  -> acute myeloid leukemia          IBD  -> inflammatory bowel disease
```

Without the initials check the same pattern matches any parenthesised capital,
which is most citations and many asides.

**The corpus** comes from `ACRONYM_CORPUS_PATH`, falling back to
`BM25_CORPUS_PATH` and then `data/corpus.jsonl` — the path the local backend
already reads. Extraction therefore runs in the same process that builds the
optimizer, with no new endpoint and no coupling to the retrieval server.

**The fallback fires only on an empty extraction.** A corpus that yields even
one pair uses its own, because a pair the corpus defined is better evidence
than a table written for a different domain.

**The source is logged.** A feature that silently selects one of three tables
is the shape this codebase has spent several changes removing, so it says which
one it chose and how many pairs it holds.

## Testing

Behaviour, not wiring:

- extraction finds a real pair from corpus text, and rejects `random words
  (XYZ)` where the initials do not match — the verification is the load-bearing
  part;
- a corpus that yields pairs beats the bundled table, checked by expanding an
  acronym present in the corpus and absent from the bundle;
- an empty extraction falls back to the bundle, which is what the 20-document
  demo corpus does;
- `ACRONYM_PATH` still overrides both, so the explicit case is untouched;
- a missing or unreadable corpus path degrades to the bundle rather than
  raising.

Each is mutation-checked: removing the initials check, reversing the
precedence, and deleting the fallback must each turn the matching test red.

## Limits

**The fallback keeps the collision risk.** A medical corpus that happens to
define no acronyms in the gloss form still gets this repository's ML table, and
`IR` still resolves wrongly there. That is a deliberate choice — the
alternative, expanding nothing, loses the feature on small corpora — and it is
recorded here rather than hidden.

Extraction is linear in corpus size, about 90µs per document: 0.5ms for the
demo corpus, 465ms for scifact, and roughly 35 seconds for a 380,000-document
BEIR corpus. It is paid once when the optimizer is built, and only when
`QUERY_EXPANSION_ENABLED` is set, which is off by default.

Only the gloss form `long form (ABBR)` is recognised. The reverse
`ABBR (long form)`, and acronyms a corpus uses without ever defining, are not
extracted. This does not attempt to disambiguate an acronym a corpus defines
two ways; the first definition wins.

**Nothing here claims expansion improves retrieval.** #601 remains open and
unanswered: the benchmarks fire the table too rarely to measure. This changes
which expansions a corpus gets, not whether expansion helps.
