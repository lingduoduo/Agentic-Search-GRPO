# Public Data Retry Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Retry transient upstream failures in the shared public-data HTTP layer, bounded by both an attempt cap and an elapsed-time cap.

**Architecture:** One loop inside `_fetch`. Retry only GET, only on 429/502/503/504 and network errors, at most 3 attempts, and never start a retry more than 15s after the first attempt began.

**Tech Stack:** Python 3, pytest, aiohttp, `src/internal/tools/public_data/_http.py`.

**Spec:** `docs/superpowers/specs/2026-09-21-public-data-retry-design.md`

## Global Constraints

- **GET only.** Retrying a POST requires an idempotency guarantee this layer cannot make.
- A status below 400, and any status not in the retryable set, must still be handled in exactly one attempt. A 404 is an answer.
- The elapsed cap is what bounds a caller that passes a long `timeout_seconds` (`search_nearby_places` overrides it for Overpass). Do not replace it with an attempt cap alone.
- `guarded`'s contract is unchanged: a final failure is still `PublicDataError`, which becomes `{"error": ...}`.
- Run `ruff check . --fix && ruff format .` before each commit.

---

### Task 1: Retry transient failures in `_fetch`

**Files:**
- Modify: `src/internal/tools/public_data/_http.py`
- Test: `tests/unit/test_public_data_http.py`

**Interfaces:**
- Produces: `_RETRYABLE_STATUSES`, `_MAX_ATTEMPTS`, `_RETRY_BACKOFF_SECONDS`, `_RETRY_BUDGET_SECONDS` module constants; `_fetch`'s signature is unchanged.

- [ ] **Step 1: Extend the fake session to replay a sequence**

The existing `_FakeSession` replays one canned response. Retry needs a
sequence. In `tests/unit/test_public_data_http.py`, add beside `_install`:

```python
class _SequencedSession:
    """Replays one queued outcome per request, recording each call."""

    calls: list[dict] = []

    def __init__(self, outcomes):
        self._outcomes = list(outcomes)

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    def request(self, method, url, **kwargs):
        _SequencedSession.calls.append({"method": method, "url": url, **kwargs})
        outcome = self._outcomes.pop(0) if self._outcomes else 200
        if isinstance(outcome, Exception):
            raise outcome
        return _FakeResponse(status=outcome, body=json.dumps({"ok": outcome}))


def _install_sequence(monkeypatch, outcomes):
    _SequencedSession.calls = []
    shared = _SequencedSession(outcomes)

    class _Aiohttp:
        @staticmethod
        def ClientTimeout(total=None):
            return total

        @staticmethod
        def ClientSession(timeout=None):
            return shared

    monkeypatch.setattr(_http, "aiohttp", _Aiohttp)
    monkeypatch.setattr(_http.asyncio, "sleep", _no_sleep)
    return _SequencedSession


async def _no_sleep(_seconds):
    """Backoff must not slow the suite; the delay itself is not under test."""
    return None
```

- [ ] **Step 2: Write the failing tests**

```python
def test_a_transient_status_is_retried_and_can_succeed(monkeypatch):
    """The whole point: a 503 that would have worked on the next try."""
    _install_sequence(monkeypatch, [503, 200])

    result = asyncio.run(get_json("https://example.org/x"))

    assert result == {"ok": 200}
    assert len(_SequencedSession.calls) == 2


def test_retries_stop_at_the_attempt_cap(monkeypatch):
    _install_sequence(monkeypatch, [503, 503, 503, 200])

    with pytest.raises(PublicDataError, match="HTTP 503"):
        asyncio.run(get_json("https://example.org/x"))

    assert len(_SequencedSession.calls) == _http._MAX_ATTEMPTS == 3


def test_a_non_transient_status_is_not_retried(monkeypatch):
    """A 404 is an answer, not a blip."""
    _install_sequence(monkeypatch, [404, 200])

    with pytest.raises(PublicDataError, match="HTTP 404"):
        asyncio.run(get_json("https://example.org/x"))

    assert len(_SequencedSession.calls) == 1


def test_a_network_error_is_retried(monkeypatch):
    _install_sequence(monkeypatch, [OSError("connection reset"), 200])

    result = asyncio.run(get_json("https://example.org/x"))

    assert result == {"ok": 200}
    assert len(_SequencedSession.calls) == 2


def test_a_post_is_never_retried(monkeypatch):
    """Retrying a write needs an idempotency guarantee this layer lacks."""
    _install_sequence(monkeypatch, [503, 200])

    with pytest.raises(PublicDataError, match="HTTP 503"):
        asyncio.run(_http._fetch("POST", "https://example.org/x",
                                 timeout_seconds=1.0, as_json=True))

    assert len(_SequencedSession.calls) == 1


def test_no_retry_once_the_elapsed_budget_is_spent(monkeypatch):
    """A caller with a long timeout must not pay it three times over."""
    _install_sequence(monkeypatch, [503, 200])
    clock = iter([0.0, _http._RETRY_BUDGET_SECONDS + 1.0])
    monkeypatch.setattr(_http.time, "monotonic", lambda: next(clock))

    with pytest.raises(PublicDataError, match="HTTP 503"):
        asyncio.run(get_json("https://example.org/x"))

    assert len(_SequencedSession.calls) == 1
```

- [ ] **Step 3: Run them and confirm they fail**

Run: `python3 -m pytest tests/unit/test_public_data_http.py -q`

Expected: the retry tests fail — `_MAX_ATTEMPTS` does not exist, and a single
503 raises immediately. The pre-existing tests in the file must still pass;
they pin the single-attempt behaviour for the success path and must not move.

- [ ] **Step 4: Implement**

In `src/internal/tools/public_data/_http.py`, add `import time` beside the
existing imports (note `asyncio` must also be imported; check before adding),
then the constants beside `DEFAULT_TIMEOUT_SECONDS`:

```python
# Retried because they say "try again", not "no": rate limits and the
# gateway/unavailable family. A 4xx other than 429 is an answer.
_RETRYABLE_STATUSES = frozenset({429, 502, 503, 504})
_MAX_ATTEMPTS = 3
_RETRY_BACKOFF_SECONDS = (0.4, 0.8)
# No retry starts after this much wall time. The per-call timeout is a
# parameter -- search_nearby_places passes a much longer Overpass budget -- so
# an attempt cap alone would let one dead host cost three full timeouts.
_RETRY_BUDGET_SECONDS = 15.0
```

Replace the body of `_fetch` from `timeout = ...` to the end of the request
block with:

```python
    timeout = aiohttp.ClientTimeout(total=timeout_seconds)
    retryable = method.upper() == "GET"
    started = time.monotonic()
    last_error: PublicDataError | None = None

    for attempt in range(_MAX_ATTEMPTS if retryable else 1):
        try:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.request(
                    method, url, params=params, data=data, headers=merged
                ) as response:
                    if response.status >= 400:
                        raise PublicDataError(
                            f"{url} returned HTTP {response.status}",
                        )
                    body = await response.text()
            break
        except PublicDataError as exc:
            status = _status_of(exc)
            if status is not None and status not in _RETRYABLE_STATUSES:
                raise
            last_error = exc
        except Exception as exc:
            logger.debug("public data request to %s failed", url, exc_info=True)
            last_error = PublicDataError(f"request to {url} failed: {exc}")

        if attempt + 1 >= (_MAX_ATTEMPTS if retryable else 1):
            break
        if time.monotonic() - started >= _RETRY_BUDGET_SECONDS:
            break
        await asyncio.sleep(_RETRY_BACKOFF_SECONDS[attempt])
    else:  # pragma: no cover - loop always breaks or exhausts above
        pass

    if last_error is not None and "body" not in dir():
        raise last_error
```

That last condition is fragile. Use an explicit sentinel instead — set
`body = None` before the loop, `break` on success, and after the loop:

```python
    if body is None:
        raise last_error or PublicDataError(f"request to {url} failed")
```

Add the small helper beside `_fetch`:

```python
def _status_of(error: PublicDataError) -> int | None:
    """Recover the HTTP status from the message, or None for a network error."""
    match = re.search(r"returned HTTP (\d{3})", str(error))
    return int(match.group(1)) if match else None
```

and `import re` if absent.

- [ ] **Step 5: Run the tests**

Run: `python3 -m pytest tests/unit/test_public_data_http.py -v`
Expected: all PASS, new and pre-existing.

- [ ] **Step 6: Mutation-check**

Restore with file copies, not `git checkout` — nothing is committed yet.

```bash
cp src/internal/tools/public_data/_http.py /tmp/http_good.py

# A: no retry at all
python3 - <<'PY'
import pathlib
p = pathlib.Path("src/internal/tools/public_data/_http.py"); s = p.read_text()
p.write_text(s.replace("_MAX_ATTEMPTS = 3", "_MAX_ATTEMPTS = 1"))
PY
python3 -m pytest tests/unit/test_public_data_http.py -q 2>&1 | tail -3
cp /tmp/http_good.py src/internal/tools/public_data/_http.py

# B: retry everything, including a 404
python3 - <<'PY'
import pathlib
p = pathlib.Path("src/internal/tools/public_data/_http.py"); s = p.read_text()
p.write_text(s.replace("_RETRYABLE_STATUSES = frozenset({429, 502, 503, 504})",
                       "_RETRYABLE_STATUSES = frozenset(range(400, 600))"))
PY
python3 -m pytest tests/unit/test_public_data_http.py -q 2>&1 | tail -3
cp /tmp/http_good.py src/internal/tools/public_data/_http.py

# C: drop the elapsed cap
python3 - <<'PY'
import pathlib
p = pathlib.Path("src/internal/tools/public_data/_http.py"); s = p.read_text()
p.write_text(s.replace("_RETRY_BUDGET_SECONDS = 15.0", "_RETRY_BUDGET_SECONDS = 1e9"))
PY
python3 -m pytest tests/unit/test_public_data_http.py -q 2>&1 | tail -3
cp /tmp/http_good.py src/internal/tools/public_data/_http.py && rm /tmp/http_good.py
python3 -m pytest tests/unit/test_public_data_http.py -q 2>&1 | tail -2
```

Expected: A turns the retry/success and attempt-cap tests red; B turns the
404 test red; C turns the budget test red. The restore is green.

- [ ] **Step 7: Confirm the live tools still work**

```bash
cat > ./_tools_tmp.py <<'PY'
import asyncio, json
from src.internal.tools.public_data import public_data_tools
ARGS = {"search_wikipedia": {"query": "information retrieval"},
        "get_weather": {"location": "Berlin"},
        "get_stock_quote": {"symbol": "AAPL"},
        "search_wayback": {"url": "example.com"}}
async def main():
    tools = {t.name: t for t in public_data_tools()}
    for name, args in ARGS.items():
        t = tools[name]; iid = await t.create()
        try:
            resp, _, _ = await asyncio.wait_for(t.execute(iid, args), timeout=40)
            p = json.loads(resp)
            bad = isinstance(p, dict) and "error" in p
            print(f"  {name:22} {'ERROR: ' + str(p['error'])[:50] if bad else 'ok'}")
        finally:
            await t.release(iid)
asyncio.run(main())
PY
python3 ./_tools_tmp.py; rm -f ./_tools_tmp.py
```

Expected: the first three succeed. `search_wayback` may still fail — the
upstream fails ~2 of 6 trials even with retry — and that is the honest
outcome, not a regression.

- [ ] **Step 8: Commit**

```bash
ruff check . --fix && ruff format .
git add src/internal/tools/public_data/_http.py tests/unit/test_public_data_http.py
git commit -m "fix(tools): retry a transient upstream in the public-data HTTP layer"
```

---

### Task 2: Verify and open the PR

- [ ] **Step 1: Full suite**

Run: `python3 -m pytest -q 2>&1 | tail -3`
Expected: PASS at the previous count plus 6.

- [ ] **Step 2: Commit docs, push, open the PR**

```bash
git add docs/superpowers/
git commit -m "docs(tools): spec and plan for public-data retry"
git push -u origin fix/public-data-retry
gh pr create --title "fix(tools): retry a transient upstream in the public-data HTTP layer" --body "see spec"
```
