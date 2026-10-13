# Automatic category selection

Sub-project 3 of 3 toward search-source agents. Sub-project 1 made each category
search its own sites. Sub-project 2 (#688) let a query use up to three
categories, fused with RRF. This sub-project lets the system pick those
categories when the caller asks it to.

## Intent

A caller can send `domains: ["auto"]` instead of naming categories. The system
embeds the query, picks one to three search-source categories by nearest
examples, and runs the normal multi-category search. Auto selection is
**opt-in**: requests that don't ask for it behave exactly as today.

Success is selection accuracy against a fixed bar on held-out labels, measured
with local embeddings only and no search-provider calls. Whether auto-picked
searches give better answers is out of scope.

## Examples, index and scoring

A new package, `src/model/pre_training/categories/`, reuses the intent
router's generic pieces without modifying the intent router:

- `encode_texts` and the e5 `query:` prefix (`intfloat/e5-small-v2`);
- the lazy, cached CPU model load;
- the `.npz` save/load format with a fingerprint of the examples file;
- top-k mean scoring.

It has its own index class over the 20 category labels in `DOMAIN_REGISTRY`.

**Examples.** `data/category_examples.json` is force-added to git like the other
`data/` files. It is a list of `{"id", "category", "text"}` entries, about 8
short example queries per category, all 20 categories present.

- A separate subagent writes them and is never shown
  `data/eval/source_category_queries.json`.
- The loader rejects unknown categories, duplicate ids, duplicate normalized
  texts, and a category with no examples.

**Build.** `python -m src.model.pre_training.categories.cli build --examples
data/category_examples.json --output data/category_index` writes
`data/category_index/index.npz`, which is not tracked in git. It holds:

- `vectors`: L2-normalized float32;
- `examples`: the records, as JSON;
- `encoder`: the model name;
- `fingerprint`: the sha256 of the examples file.

**Scoring.**

1. The query is embedded with the `query:` prefix.
2. Each category scores the mean of its `top_k` highest cosines among its own
   examples.
3. The selection is the best category, plus every other category whose score is
   at least `best − select_margin`, ordered by score, truncated to 3.
4. Ties are broken by `DOMAIN_REGISTRY` order. At least one category is always
   selected.

**Tuning.** `top_k ∈ {1, 2, 3, 5}` and `select_margin ∈ {0.00, 0.01, 0.02, 0.03,
0.05}` are chosen on the **dev split only**. The winner has the highest
coverage, then the fewest mean picks, then the smaller `top_k`, then the
smaller `select_margin`. The chosen values become the code defaults.

## Request flow

`resolve_auto_domains(query, domains) -> (domains, auto_info)` lives in
`src/internal/tools/search.py` and runs before `normalize_search_domains` at
every entry point that accepts `domains`.

- **Input other than `["auto"]`:** returned unchanged, with `auto_info = None`.
- **Input `["auto"]`:**
  - The query is embedded off the event loop, through `asyncio.to_thread`.
  - The categories are selected, and `auto_info = {"selected": [...], "status":
    "ok"}` is returned.
  - The selected list then goes through `normalize_search_domains` and
    `search_categories` as if the caller had named it. One selected category
    takes the existing single-category path unchanged.
- **Rejected input:** `"auto"` mixed with named categories, and `domain="auto"`,
  both get `InvalidToolInput`, which becomes HTTP 400 or a tool error.
- **Fallback:**
  - **When:** `AGENTIC_SEARCH_CATEGORY_INDEX_PATH` is empty (the default), the
    index fails to load, its encoder differs from the default, or encoding
    raises.
  - **Result:** `auto` resolves to `["web"]` with `auto_info = {"selected":
    ["web"], "status": "unavailable"}`, and the search still runs.
  - **No retry:** a failed load is cached and never retried, following the
    intent router's pattern.
- **Settings**, read through the typed app configuration and documented in
  `docs/configuration.md`:
  - `AGENTIC_SEARCH_CATEGORY_INDEX_PATH` (directory; empty disables auto);
  - `AGENTIC_SEARCH_CATEGORY_TOP_K` (a positive integer);
  - `AGENTIC_SEARCH_CATEGORY_SELECT_MARGIN` (in [0, 1]).

## Entry points and responses

| Entry point | `auto` input | Output |
| --- | --- | --- |
| `POST /api/agent` (JSON, SSE, WebSocket) | `domains: ["auto"]` | `domains` echoes the searched categories, including a single one. The new `auto_domains` field (null unless `auto` was requested) is also on the streaming `done` event. The mode rule applies to the resolved categories. |
| `web_search` tool | `domains: ["auto"]` | Metadata gains `domains` and `auto_domains`. |
| `search_domain` / `batch_search` | `domains: ["auto"]` | The result gains `domains` and `auto_domains`. `tag` plus `domains` is still rejected. |
| MCP `search_web` / `search_domain` / `batch_search` | `domains: ["auto"]` | Same fields as the tools. |
| Assist page | an "Auto" checkbox, first in the "Search source" group | Ticking Auto clears the other choices, and ticking a category clears Auto. Auto sends `domains: ["auto"]`. After the response, the page shows "Auto picked: a · b" or "Auto unavailable: searched web". The selection stays on Auto rather than switching to the picks. |

**Cost.** Selection makes no provider calls. Searching costs the same as naming
the categories: at most 6 provider calls per query per provider.

## Evaluation

`python -m src.model.pre_training.categories.cli evaluate`, using local
embeddings only:

- **Dev split:** the first label of each category in
  `data/eval/source_category_queries.json`, in file order (16 queries, the same
  mechanical rule as the source-category pilot). It is used only for the
  tuning grid. Output: `data/eval/category_selection_tuning.json`.
- **Test split:** the other 48 queries, scored **once** with the frozen
  defaults. Output: `data/eval/category_selection.json`, containing:
  - **coverage**: the share of queries whose labelled category is in the
    selected set;
  - mean picks;
  - top-1 accuracy;
  - per-category coverage (descriptive only);
  - a confusion table of labelled category against selected categories.
- **Pre-registered bar:** coverage ≥ 0.80 **and** mean picks ≤ 2.0. If the bar
  is missed, the feature still ships opt-in and the docs say it missed the bar,
  with the numbers.
- **Stated caveats:**
  - the labels were written to fit their category, so the numbers are likely
    optimistic;
  - examples and labels were both drafted by LLM subagents;
  - each label carries one category, so a correct extra pick does not count.
- **Leakage guard:** a test fails if any example's normalized text equals an
  eval query's. When the encoder is available, an eval-gate check also fails
  if any example has e5 cosine > 0.95 to an eval query.
- **CI:** the encoder-dependent build and evaluation run in the existing
  eval-gate workflow, which installs torch. Unit tests stay torch-free, with
  `encode_texts` stubbed.

## Testing

All unit tests run offline with stubbed encoders. Every new test is
mutation-checked.

- **Loader:** rejects unknown categories, duplicate ids and texts, and a
  category with no examples.
- **Selection:** top-k mean; the margin rule; at most 3 and at least 1 picks;
  deterministic ties.
- **Resolver:**
  - non-auto input is unchanged;
  - `["auto"]` resolves;
  - `auto` mixed with named categories, and `domain="auto"`, are rejected;
  - the fallback to `["web"]` / `unavailable` covers an unset path, a failed
    load (cached), an encoder mismatch, and an encoder that raises;
  - encoding runs through `asyncio.to_thread`.
- **Entry points:** HTTP JSON and stream, `web_search`, `search_domain` /
  `batch_search` and MCP each resolve `auto` before any provider call and
  report `auto_domains`.
- **Frontend:**
  - the Auto checkbox clears the other choices and is cleared by them;
  - the request sends `domains: ["auto"]`;
  - "Auto picked" and "Auto unavailable" render;
  - the selection stays on Auto after a search.

## Documentation

- **`docs/search-engine.md`:** a new "Automatic category selection" section
  covering opt-in, method, fallback, the measured result against the bar, and
  the caveats.
- **`docs/configuration.md`:** rows for the three settings.
- **`docs/api-reference.md`, `docs/mcp.md`, `docs/frontend.md`:** the `auto`
  option.
- **README:** notes the opt-in automatic selection.

## Out of scope

- Making `auto` the default.
- An LLM-based or keyword router.
- Labels that allow several correct categories per query.
- Measuring the answer quality of auto-picked searches.
- Any change to the intent router.
