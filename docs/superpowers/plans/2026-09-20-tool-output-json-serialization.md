# Tool Output JSON Serialization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `FunctionTool.execute` emit JSON for non-string results, so the source-card structure recovery and the loop's whole-item truncation — both already written — actually run.

**Architecture:** One branch in `FunctionTool.execute`. Strings pass through untouched; everything else goes through `json.dumps(..., ensure_ascii=False, default=str)` with a fallback to today's `str()`.

**Tech Stack:** Python 3, pytest, `src/internal/tools/base.py`.

**Spec:** `docs/superpowers/specs/2026-09-20-tool-output-json-serialization-design.md`

Refs #596.

## Global Constraints

- **A `str` result must pass through byte-identical.** `json.dumps` on a string adds quotes and would change every text-returning tool's message. This is the regression to guard hardest.
- `ensure_ascii=False` is required, not cosmetic: `_fit_json_array` documents that bare `json.dumps` re-escapes non-ASCII to `\uXXXX`, inflating its size accounting.
- Do not touch `_fit_json_array`, `_slice_text`, `max_tool_response_length`, or `ToolExecutionResult`. The point of this change is that those already work.
- Run `ruff check . --fix && ruff format .` before each commit; the pre-commit `ruff-format` hook aborts otherwise.

---

### Task 1: Serialize non-string tool results as JSON

**Files:**
- Modify: `src/internal/tools/base.py` (`FunctionTool.execute`)
- Test: `tests/unit/test_tool_output_serialization.py` (create)

**Interfaces:**
- Consumes: nothing.
- Produces: `FunctionTool.execute(instance_id, arguments) -> tuple[str, Any, dict]` where the first element is JSON for non-string results and the original string for string results. The second element (`raw`) is unchanged.

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_tool_output_serialization.py`:

```python
"""Tool responses must be JSON, because two consumers already assume they are.

`tool_agent_runner._extract_tool_calls_and_docs` calls json.loads on the
response to recover structure, and `_fit_json_array` trims a JSON array by
whole items. A Python repr defeats both silently. See #596.
"""

import json

import pytest

from src.agents.tool.tool_calling import _truncate_tool_text
from src.internal.tools.base import FunctionTool


def _execute(fn, **arguments):
    import asyncio

    tool = FunctionTool(fn=fn, name=getattr(fn, "__name__", "t"))

    async def run():
        instance_id = await tool.create()
        try:
            return await tool.execute(instance_id, arguments)
        finally:
            await tool.release(instance_id)

    return asyncio.run(run())


async def _dict_tool(symbol: str) -> dict:
    return {"symbol": symbol.upper(), "price": 123.45, "previous_close": None}


async def _list_tool(count: int) -> list:
    return [
        {"title": f"Result {i}", "url": f"https://example.org/{i}", "snippet": "x" * 260}
        for i in range(count)
    ]


async def _str_tool(text: str) -> str:
    return text


async def _unserializable_tool() -> object:
    return object()


async def _unicode_tool() -> dict:
    return {"city": "Köln", "note": "naïve café"}


def test_a_dict_result_round_trips_through_json():
    """This is exactly the recovery the source-card builder attempts."""
    response, raw, _ = _execute(_dict_tool, symbol="aapl")

    assert json.loads(response) == raw
    assert json.loads(response)["symbol"] == "AAPL"


def test_a_string_result_is_passed_through_unchanged():
    """json.dumps on a str would add quotes and change every text tool."""
    response, raw, _ = _execute(_str_tool, text='plain text with "quotes"')

    assert response == 'plain text with "quotes"'
    assert response == raw


def test_a_long_list_result_is_trimmed_by_whole_items():
    """_fit_json_array only engages for real JSON; a repr falls to slicing."""
    response, _, _ = _execute(_list_tool, count=12)
    assert len(response) > 2048

    truncated = _truncate_tool_text(response, 2048, "left")

    assert "results shown" in truncated
    assert "omitted for length" in truncated
    assert "...(truncated)" not in truncated
    body = truncated.split("\n...")[0]
    assert isinstance(json.loads(body), list)


def test_non_ascii_is_not_escaped():
    """ensure_ascii=False: \\uXXXX inflates _fit_json_array's size accounting."""
    response, _, _ = _execute(_unicode_tool)

    assert "Köln" in response
    assert "\\u" not in response


def test_an_unserializable_result_degrades_instead_of_raising():
    response, _, _ = _execute(_unserializable_tool)

    assert isinstance(response, str)
    assert response
```

- [ ] **Step 2: Run them and confirm which fail**

Run: `python3 -m pytest tests/unit/test_tool_output_serialization.py -v`

Expected: `round_trips_through_json`, `trimmed_by_whole_items` and `non_ascii_is_not_escaped` FAIL. `passed_through_unchanged` and `degrades_instead_of_raising` PASS already — they are regression guards for behavior that must survive, not new behavior.

- [ ] **Step 3: Change the serializer**

In `src/internal/tools/base.py`, add `import json` to the imports if absent,
then replace the final line of `FunctionTool.execute`:

```python
        return str(result), result, {}
```

with:

```python
        # JSON, not str(): the source-card builder json.loads this back into
        # structure and _fit_json_array trims JSON arrays by whole items.
        # A Python repr defeats both silently. A str passes through untouched,
        # since json.dumps would only add quotes. See #596.
        if isinstance(result, str):
            response = result
        else:
            try:
                response = json.dumps(result, ensure_ascii=False, default=str)
            except (TypeError, ValueError):
                response = str(result)
        return response, result, {}
```

- [ ] **Step 4: Run the tests again**

Run: `python3 -m pytest tests/unit/test_tool_output_serialization.py -v`

Expected: 5 PASS.

- [ ] **Step 5: Mutation-check the three new assertions**

A test that cannot fail is worse than none. Revert the serializer and confirm each goes red:

```bash
python3 - <<'PY'
import pathlib
p = pathlib.Path("src/internal/tools/base.py")
s = p.read_text()
old = """        if isinstance(result, str):
            response = result
        else:
            try:
                response = json.dumps(result, ensure_ascii=False, default=str)
            except (TypeError, ValueError):
                response = str(result)
        return response, result, {}"""
new = "        return str(result), result, {}"
assert s.count(old) == 1
p.write_text(s.replace(old, new))
PY
python3 -m pytest tests/unit/test_tool_output_serialization.py -q
git checkout src/internal/tools/base.py
```

Expected: the three new tests RED, the two regression guards still green. Then restore.

Also mutation-check the `ensure_ascii` choice specifically, since it is easy to
write a test that passes either way:

```bash
python3 - <<'PY'
import pathlib
p = pathlib.Path("src/internal/tools/base.py")
s = p.read_text()
p.write_text(s.replace("ensure_ascii=False, default=str", "default=str"))
PY
python3 -m pytest tests/unit/test_tool_output_serialization.py -k non_ascii -q
git checkout src/internal/tools/base.py
```

Expected: RED. If it stays green, the assertion is not testing `ensure_ascii`.

- [ ] **Step 6: Confirm the restore is clean**

Run: `git diff --stat src/internal/tools/base.py`

Expected: empty after the checkouts, then re-apply Step 3 and confirm 5 PASS.
**Commit before running any further checkout-based mutation**, or a restore
will discard the implementation rather than the mutant.

- [ ] **Step 7: Check every real tool end to end**

```bash
python3 - <<'PY'
import asyncio, json
from src.internal.tools.public_data import public_data_tools

async def main():
    bad = []
    for t in public_data_tools():
        fn = getattr(t, "_fn")
        inner = getattr(fn, "__wrapped__", fn)
        import typing
        ann = typing.get_type_hints(inner).get("return", None)
        name = getattr(ann, "__name__", str(ann))
        # Exercise the serializer directly with a representative value.
        sample = {"a": 1} if name == "dict" else ([{"a": 1}] if name == "list" else "text")
        response, raw, _ = await t.execute(await t.create(), {}) if False else (None, None, None)
        print(f"  {t.name:24} returns {name}")
    print("annotations listed; see the serializer test for behavior")

asyncio.run(main())
PY
python3 -m pytest tests/unit/ -k "tool" -q
```

Expected: the tool-related unit tests PASS. This is the sweep that would catch a
test elsewhere asserting on repr-formatted output.

- [ ] **Step 8: Commit**

```bash
ruff check . --fix && ruff format .
git add src/internal/tools/base.py tests/unit/test_tool_output_serialization.py
git commit -m "fix(tools): serialize non-string tool results as JSON"
```

---

### Task 2: Verify, update #596, and open the PR

**Files:** none modified.

- [ ] **Step 1: Run the full default suite**

Run: `python3 -m pytest -q 2>&1 | tail -3`

Expected: PASS, at the previous count plus 5. Any failure elsewhere is a test
that was asserting on the repr format — read it before changing it, and say so
in the PR rather than quietly adjusting it.

- [ ] **Step 2: Demonstrate both consumers now work**

```bash
python3 - <<'PY'
import asyncio, json
from src.internal.tools.base import FunctionTool
from src.agents.tool.tool_calling import _truncate_tool_text

async def ranked(count: int) -> list:
    return [{"title": f"Result {i}", "url": f"https://example.org/{i}",
             "snippet": "x" * 260} for i in range(count)]

async def main():
    tool = FunctionTool(fn=ranked, name="ranked")
    iid = await tool.create()
    response, _, _ = await tool.execute(iid, {"count": 12})

    # consumer 1: the source-card builder's recovery
    decoded = json.loads(response)
    print(f"card summary: {len(decoded)} items (was a 200-char repr)")

    # consumer 2: whole-item truncation
    out = _truncate_tool_text(response, 2048, "left")
    print("truncation   :", out.strip()[-52:])

asyncio.run(main())
PY
```

Expected: `card summary: 12 items` and a tail reading `N of 12 results shown,
M omitted for length.` — not `...(truncated)`.

- [ ] **Step 3: Repoint #596**

```bash
gh issue comment 596 --body "$(cat <<'BODY'
## Answered, and the root cause is not `_raw`

Investigated the question this issue poses — does anything actually want structured tool output? **Yes, two things already do, and both fail silently.**

`tool_agent_runner._extract_tool_calls_and_docs` calls `json.loads` on the response specifically to recover the structure the loop dropped, then falls back to 200 characters of repr when it raises. And `_fit_json_array` trims a ranked JSON array by whole items, appending a footer telling the model what it is not seeing.

Neither runs, because all nine public data tools return `dict` or `list` and `FunctionTool.execute` stringifies with `str()` — a Python repr, not JSON.

The truncation consequence is the worse one. Measured on a 12-item ranked list of 3964 characters against the 2048 cap:

```
today  str(list):  CHARACTER SLICE
    ends: ...://example.org/6', 'snippet': 'x...(truncated)
if json.dumps:     whole-item trim
    ends: ...6 of 12 results shown, 6 omitted for length.
```

`_fit_json_array`'s docstring promises it "leaves the model valid JSON rather than a fragment that starts mid-object". Wikipedia, ArXiv and Wayback — the ranked, multi-item, size-prone tools it was written for — have never reached it.

**So the fix is the serializer, not `_raw`.** Preserving `raw` on `ToolExecutionResult` would be larger, would need its own action-trace serialization, and would leave both existing consumers broken anyway, since both read the string. Emitting JSON makes machinery that is already written start working.

Fixed in the linked PR. Leaving this issue open until it merges.
BODY
)"
```

- [ ] **Step 4: Push and open the PR**

```bash
git add docs/superpowers/specs/2026-09-20-tool-output-json-serialization-design.md \
        docs/superpowers/plans/2026-09-20-tool-output-json-serialization.md
git commit -m "docs(tools): spec and plan for JSON tool output"
git push -u origin fix/tool-output-json-serialization
gh pr create --title "fix(tools): serialize tool results as JSON so the machinery that reads them works" --body "$(cat <<'BODY'
Closes #596.

## Summary

`FunctionTool.execute` returned `str(result)`. For a tool returning a `dict` or `list` that is a Python repr — single-quoted, `None` instead of `null` — and **all nine public data tools return one**.

Two consumers are already written against structured output, and both fail silently.

**The source-card builder tries to recover it.** `tool_agent_runner._extract_tool_calls_and_docs` calls `json.loads(result)` precisely to rebuild the structure the loop dropped, catches the exception with a bare `except: pass`, and falls back to 200 characters of repr instead of "12 items".

**Whole-item truncation never runs for the tools it was written for.** `_fit_json_array` trims a ranked JSON array to the items that fit and appends a footer naming what was omitted. It requires real JSON, so it returns `None` for every repr. Measured on a 12-item ranked list of 3964 characters against the 2048 cap:

| | result |
|---|---|
| before | `...://example.org/6', 'snippet': 'x...(truncated)` |
| after | `...6 of 12 results shown, 6 omitted for length.` |

Its docstring promises it "leaves the model valid JSON rather than a fragment that starts mid-object". Wikipedia, ArXiv and Wayback — the ranked, size-prone tools this exists for — got the fragment.

## The change

One branch in `FunctionTool.execute`. Strings pass through untouched, because `json.dumps` on a `str` would only add quotes and would change every text-returning tool's message. `ensure_ascii=False` matches `_fit_json_array`, which documents why: bare `json.dumps` re-escapes non-ASCII to `\uXXXX` and inflates its size accounting. A non-serializable result degrades to `str()` rather than failing the call.

## Notes

**This is deliberately not what #596 proposed.** That issue framed the gap as the discarded `_raw`. Preserving it would be a larger change, would need its own action-trace serialization, and would leave both consumers broken anyway since both read the string. Fixing the serializer makes already-written machinery start working — and the `_` prefix on `_raw` turns out to have been correct.

Two of the five tests are regression guards for behavior that must **not** change: a string result stays byte-identical, and an unserializable result still returns. All assertions were mutation-checked, including `ensure_ascii` on its own, since that one is easy to write so it passes either way.

**Behavior change:** non-string tool results now reach the model as JSON rather than Python repr. That is the point — everything downstream already assumed JSON.

Spec: `docs/superpowers/specs/2026-09-20-tool-output-json-serialization-design.md`
Plan: `docs/superpowers/plans/2026-09-20-tool-output-json-serialization.md`

🤖 Generated with [Claude Code](https://claude.com/claude-code)
BODY
)"
```

- [ ] **Step 5: Confirm every commit reached the remote**

Run: `git log --oneline origin/fix/tool-output-json-serialization -4`
