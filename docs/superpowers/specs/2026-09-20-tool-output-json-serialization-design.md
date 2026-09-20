# Serialize tool output as JSON, not as a Python repr

Refs #596.

> **Correction (2026-09-20, after merge).** The problem statement below is
> **wrong**, and the change it justified fixed nothing that was broken.
>
> Every tool in the registry is wrapped by `guarded`
> (`src/internal/tools/public_data/_http.py:137`), which already serialized:
>
> ```python
> async def _wrapped(**kwargs) -> str:
>     try:
>         return json.dumps(await fn(**kwargs))
> ```
>
> Verified by code-object origin, which `functools.wraps` cannot fake: 13 of 14
> registered tools route through `_http.py:145`; the one that does not
> (`search`) returns `str` anyway. So the nine public data tools **already
> returned JSON strings**. Card recovery already succeeded, and
> `_fit_json_array` already engaged — re-run on a real guarded list tool, it
> produced `6 of 12 results shown, 6 omitted for length`, the exact output the
> text below calls impossible.
>
> Two compounding errors produced this document. The survey read
> `__wrapped__`'s annotation (`-> dict`), but `functools.wraps` copies
> `__annotations__` onto a wrapper that returns `str`. And the "before"
> evidence came from a `FunctionTool` constructed by hand around an *unguarded*
> callable, then reported as the real tools' behaviour.
>
> **What survives:** the change is inert for every existing tool — a `str`
> result passes through untouched — so there was no regression, and the tests
> are valid unit tests of `FunctionTool.execute`'s contract. It stands as a
> guard for any future tool built without `guarded`. **What does not:** that it
> repaired a live defect, and that #596 had a demonstrated consumer. #596's
> original framing — a question with no consumer, not to be fixed on sight —
> was correct.
>
> Kept as written below rather than rewritten, so the reasoning that produced a
> wrong conclusion stays legible. Do not cite the claims in "The problem" as
> fact.


## Goal

Make two pieces of machinery that already exist actually run: the source-card
structure recovery, and the loop's whole-item truncation. Both are defeated by
one line.

## The problem

`FunctionTool.execute` ends:

```python
return str(result), result, {}
```

For a tool returning a `dict` or a `list`, `str()` produces a Python repr —
single-quoted, `None` instead of `null`. It is not JSON, and nothing downstream
can parse it.

Every one of the nine public data tools returns a `dict` or a `list`:

```
search_wikipedia  list    search_arxiv     list    search_wayback       list
get_weather       dict    get_stock_quote  dict    get_crypto_price     dict
convert_currency  dict    search_location  dict    search_nearby_places dict
```

Two consumers are already written against structured output and silently get
nothing.

### The source card builder tries to recover and fails

`tool_agent_runner._extract_tool_calls_and_docs`:

```python
if isinstance(result, str):
    try:
        decoded_result = _json.loads(result)
    except Exception:
        pass
if isinstance(decoded_result, list):
    result_summary = f"{len(decoded_result)} items"
elif result is not None:
    result_summary = str(result)[:200]
```

The `json.loads` is a deliberate attempt to recover the structure the loop
dropped. It raises for every one of the nine tools, the bare `except` swallows
it, and the card falls back to 200 characters of repr instead of "12 items".

### Whole-item truncation never runs for the tools it was written for

`_fit_json_array` trims a JSON array to the leading items that fit, appending a
footer that tells the model what it is not seeing. Its docstring states the
intent: "Ranked tool results are ordered best-first, so dropping items from the
end keeps what matters and leaves the model valid JSON rather than a fragment
that starts mid-object."

It requires text that parses as a JSON list, so it returns `None` for every
repr and the caller falls through to character slicing. Measured on a
12-item ranked list of 3964 characters against the 2048 cap:

```
today  str(list):  CHARACTER SLICE
    ends: ...://example.org/6', 'snippet': 'x...(truncated)
if json.dumps:     whole-item trim
    ends: ...6 of 12 results shown, 6 omitted for length.
```

The three list-returning tools — Wikipedia, ArXiv, Wayback — are exactly the
ranked, multi-item, size-prone ones this was built for. They have never reached
it. The model receives a fragment cut mid-object and is never told that results
were omitted.

## Architecture

One change at the source:

```python
if isinstance(result, str):
    response = result
else:
    try:
        response = json.dumps(result, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        response = str(result)
return response, result, {}
```

A string result passes through untouched — `json.dumps` on a `str` would add
quotes and change every text-returning tool's message for no reason.
`ensure_ascii=False` matches `_fit_json_array`, which documents why: bare
`json.dumps` re-escapes non-ASCII to `\uXXXX`, inflating size accounting and
defeating the trimming it is doing. `default=str` keeps a non-serializable
value from raising, and the `except` is the belt-and-braces fallback to
today's behavior, because a tool returning an unserializable object should
degrade, not fail the call.

This is deliberately *not* the change #596 proposed. Preserving `_raw` on
`ToolExecutionResult` would be larger, would need its own serialization for
the action trace, and would leave the two existing consumers broken anyway,
since both read the string. Fixing the serializer makes the machinery that is
already written start working, which is the smaller and more honest repair.

## Testing

Test-driven, and the tests pin consequences rather than the format:

- A dict-returning tool's response parses as JSON, and `json.loads` round-trips
  it to the original dict — the recovery the card builder attempts.
- A list-returning tool over the cap is trimmed by whole items and carries the
  "N of M results shown" footer, rather than being character-sliced.
- A str-returning tool's response is byte-identical to before, with no added
  quotes. This is the regression that would otherwise slip through.
- A tool returning a non-serializable object still produces a response instead
  of raising.

Each is mutation-checked by reverting the serializer to `str(result)` and
confirming it goes red, so none is a test that cannot fail.

The nine public tools are then checked end to end: every one's response must
parse as JSON.

## Limits

This changes what the model sees for non-string tool results: single-quoted
Python repr becomes JSON. That is a behavior change to tool messages, and it is
the point — the downstream machinery already assumes JSON. JSON is also the
format the model is likelier to have seen for tool output.

Tools returning strings are untouched, which is most of the tool surface
outside `public_data/`.

Out of scope: preserving `_raw` on `ToolExecutionResult` (this change removes
the need, and #596 is updated to say so), the 2048 cap itself, and the source
card rendering beyond the summary line it already computes.
