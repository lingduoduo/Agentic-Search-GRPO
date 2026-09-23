# Tighten the built-in tool schemas — design

## Problem

The built-in tools write most of their constraints as prose in the description —
"How many articles to return (1-10)", "One of relevance, lastUpdatedDate,
submittedDate", "Three-letter currency code" — and the code then clamps or
silently substitutes: `limit=500` becomes 10, an unknown `sort_by` becomes
`relevance`, a latitude of 999 goes to the upstream API. No schema sets
`additionalProperties: false`, so a misspelled key is dropped and the default
used. A validator can only enforce what the schema declares.

Companion to PR #637, which makes the validator enforce the full JSON Schema.
This PR declares the constraints; that one makes them binding. They are
independent: each is correct without the other.

## Rule set (enforced for every seeded tool by a guard test)

Every tool from `tool_knowledge_base()` (plus `rag_routing_tool`) and the three
memory tools:

1. The parameters schema passes `Draft202012Validator.check_schema`.
2. Every object schema — root and nested — sets `additionalProperties: false`,
   except open maps listed in the guard with a reason. The only one:
   `params` on `search_domain`/`batch_search`, whose keys depend on `tag`.
3. Every `integer` has both `minimum` and `maximum`.
4. Every `number` has a lower bound (`minimum` or `exclusiveMinimum`).
5. Every `array` has `items` and `maxItems`.
6. Every required `string` has `minLength >= 1`.

## Values

Ranges match the clamps the code already applies, so the schema describes what
the tool actually does; the clamps stay as defence in depth.

| tool.field | constraint |
|---|---|
| search_wikipedia.limit | 1–10 |
| search_wikipedia.language | `^[A-Za-z]+(-[A-Za-z]+)*$`, maxLength 20 |
| search_arxiv.max_results | 1–25 |
| search_arxiv.sort_by | enum = `_ARXIV_SORTS` |
| search_wayback.limit | 1–50 |
| search_wayback.year | 1996–9999 (Wayback began 1996; CDX timestamps carry a 4-digit year) |
| get_weather / search_nearby_places latitude | −90…90 |
| get_weather / search_nearby_places longitude | −180…180 |
| search_nearby_places.radius_meters | 1–10000 |
| search_nearby_places.limit | 1–50 |
| search_location.limit | 1–20 |
| search_location.country_code | `^[A-Za-z]{2}$` |
| convert_currency.amount | exclusiveMinimum 0 |
| convert_currency.from/to_currency | `^[A-Za-z]{3}$` |
| get_crypto_price.vs_currency | `^[A-Za-z]{2,10}$` |
| extract_page.url | `^[Hh][Tt][Tt][Pp][Ss]?://` (the executor rejects anything else) |
| web_search.queries | minItems 1, **maxItems 5** |

`web_search.queries` had no cap at all, and each query is a paid provider call;
5 matches `batch_search`'s existing limit. This is the one new limit rather than
a prose constraint made formal.

Patterns stay ECMA-262-portable (no inline flags), since providers consume these
schemas too.

## Not changed

- `get_weather` keeps `location` required although latitude/longitude skip the
  lookup. Making it `anyOf` puts a combinator at the schema root, which some
  providers reject for function parameters.
- Free-text fields (`query`, `content`, `symbol`) get `minLength` only — no
  guessed formats.
- `tag` stays a string: the executor validates it against live routes with a
  message naming `get_sub_domains`.

## Testing

`tests/unit/test_tool_schema_strictness.py`: the guard (rules 1–6 over every
seeded tool, with the exemption list), plus a table test per row above using
`Draft202012Validator` directly — an in-range value passes, an out-of-range one
fails — so the tests do not depend on #637's validator.
