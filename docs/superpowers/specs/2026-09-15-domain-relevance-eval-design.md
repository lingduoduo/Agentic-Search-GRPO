# Domain relevance evaluation

## Goal

Answer the question the search-domains taxonomy has left open since it shipped:
does selecting a topic domain change what a web search returns, and does it
move results toward that topic's authoritative sources?

`2026-09-14-search-domains-design.md` states the bar: "Before describing the
feature as improving quality, compare general/domain runs on labeled queries
for all 17 categories, including overlapping topics, measuring relevance and
empty-result rates with the same provider configuration." Nothing has met that
bar. Both shipped increments say so explicitly. This one measures it.

## What the environment permits

Three facts, established by probing rather than assumed, constrain every choice
below.

SerpAPI works: a live three-result query returned three usable pages. Google
Custom Search returns 401. One working provider is sufficient, because the bar
asks for *the same* provider configuration, not several.

The OpenAI key has no credits: the API returns 429 "You have no credits
remaining." An LLM-as-judge design, which is the textbook approach to grading
web results, would produce a harness that outputs nothing on this machine. It
is therefore not the design.

SerpAPI has 158 searches left on a 250/month free plan. At two arms per query
this budget, not statistical preference, sets the sample size.

## Measuring relevance without a judge

Each of the 17 domains gets three queries, and each query carries the URL hosts
an on-topic result should come from — `academic` from `arxiv.org`, `*.edu`,
`doi.org`, `pubmed.ncbi.nlm.nih.gov`; `code` from `github.com`,
`stackoverflow.com`, `docs.python.org`. Relevance is then the share of top-k
results drawn from those hosts, compared between the general and domain arms of
the same query.

This is label-once, run-many: the labels are written by hand once, committed,
and every subsequent run is deterministic and free. It needs no LLM and no
per-run human judgment.

It measures topical and source alignment, not answer quality. An `arxiv.org`
paper is not automatically a better answer than a well-written blog post, and
this evaluation cannot tell the difference. The report says so in its own text,
not only here.

## Architecture

Three components, split along the boundary between serving-side code and
training-side code.

`src/internal/retrieval/domain_eval.py` holds the measurement core and stays
torch-free. It runs both arms of a query through `search_tool`, computes the
per-query metrics, and returns records. It imports nothing from
`src/model/post_training`.

`data/eval/domain_relevance_queries.json` holds the label set: 17 domains,
three queries each, with authority hosts per query. `data/` is gitignored, so
this file is force-added, as `data/eval/routing_labels.jsonl` already is.

`examples/run_domain_relevance_eval.py` is the command-line entry point. It
calls the core, applies the paired statistics, and writes the report to
`data/eval/domain_relevance.json`.

The statistics live in the CLI rather than the core for a concrete reason:
importing `src.model.post_training.eval.stats` pulls torch in through package
`__init__` side effects, verified by import. A retrieval-side module that did
that would drop out of the torch-free CI job. Scripts under `examples/` already
import training code, so the CLI is where that dependency belongs, and
`paired_permutation_p` is reused rather than reimplemented.

## Metrics

Top-k is fixed at k=10. SerpAPI bills per search, not per result, so a wider
result list costs no additional quota and gives the precision metric ten
gradations per arm instead of three.

Computed per query, for both the general and the domain arm:

- **authority_precision@k** — the share of the top k results whose URL host
  matches that query's authority hosts. A host matches on exact equality or as
  a suffix following a dot, so `cs.stanford.edu` matches the `stanford.edu`
  pattern and `notstanford.edu` does not.
- **empty** — whether the arm returned no usable results. The taxonomy spec
  names empty-result rate explicitly, and it needs no labels.
- **jaccard** — overlap of result URLs between the two arms of one query. It
  requires no labels and detects the outcome where the hint changes nothing,
  which no relevance metric would distinguish from a hint that changes results
  neutrally.

The per-query delta is the domain arm's authority precision minus the general
arm's.

## Statistics

The unit of analysis is the query. There are 51 of them, paired: the same query
is run with and without its domain hint. The three results within a query are
not three independent observations, and treating them as rows would inflate the
sample roughly threefold and understate every interval.

One pooled two-sided paired sign-flip permutation test runs over the 51 deltas,
reported with Cliff's delta as effect size. Pooling is what the sample supports.

Per-domain results are reported as descriptive statistics and are not tested.
Three queries cannot support an inferential claim about one domain, and running
17 such tests under Benjamini-Hochberg correction at n=3 could not reject
anything at any effect size — the test would be vacuous, and printing 17
p-values would present that vacuum as a finding. The report prints per-domain
means without p-values and states why.

A null pooled result is reported as null. At this sample size a null means the
evaluation did not detect an effect, not that no effect exists, and the report
distinguishes those.

## Cost and reproducibility

Every provider response is cached to `data/eval/cache/` under a key derived
from provider, query, and k. The first full run spends 102 of the 158 available
searches; re-runs read the cache and spend nothing. The CLI reports how many
live calls it intends to make before making them.

Unit tests drive the core with a stub provider and never open a socket, so the
suite stays fast and costs no quota.

## Error handling

A provider error on one arm marks that arm empty and the query is excluded from
the paired relevance comparison, since a delta against a failed call measures
the failure rather than the domain. Excluded queries are counted and reported;
if exclusions exceed a third of the sample the report says the run is not
interpretable rather than presenting a number computed from the remainder.

An unreadable or malformed label file fails immediately with the offending
path, before any provider call, rather than silently evaluating fewer domains.

## Testing

Test-driven, all offline:

1. Host matching accepts a subdomain and rejects a suffix collision
   (`stanford.edu` matches `cs.stanford.edu`, not `notstanford.edu`).
2. authority_precision@k is 0 when no result matches and 1 when all do.
3. A failed arm marks the query excluded, and excluded queries do not enter the
   paired deltas.
4. The cache returns a stored response without calling the provider a second
   time.
5. The label file covers all 17 registry identifiers, so the taxonomy and the
   evaluation cannot drift apart.
6. The report's numeric fields are finite, since the Dev Console reads
   `data/eval/*.json` and a non-finite float reaches the frontend as `null`.
7. Jaccard is 1.0 for identical arms and 0.0 for disjoint ones.

## Limits

The evaluation measures whether domain hints move results toward
topic-appropriate sources. It does not measure answer quality, factual
accuracy, or user satisfaction, and it cannot rank one domain's hint against
another's. With three queries per domain, the only defensible claim is about
the pooled effect across all 17.

Out of scope: changing any hint text in `DOMAIN_REGISTRY` in response to the
results, an LLM judge, additional providers, and corpus-backed retrieval, which
never receives domain hints.
