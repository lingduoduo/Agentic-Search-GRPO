# A browser-search server without its binary must say so

## Goal

Make the absence of `playwright-cli` visible. #628 exposed it; this stops it
presenting as "the query found nothing".

## The problem

`web_search/browser.py` shells out to a `playwright-cli` binary via
`subprocess.run`. Nothing declares that binary:

- the `playwright` pip wheel does not provide it — it ships `playwright` — which
  is why #628 removed the declaration as the wrong package
- the container image installs no such tool

So the provider could never work under docker compose, and works on a host only
if the operator happens to have the binary.

That would be a documentation problem if the failure were legible. It is not.
`subprocess.run` raises `FileNotFoundError`, which propagated into
`_search_and_process`'s `except (subprocess.TimeoutExpired, json.JSONDecodeError,
Exception)` and became:

```
logger.warning("browser search failed for %r: %s", query, exc)
return []
```

`web_search` cascades SerpAPI → browser. An empty list from the fallback is
indistinguishable from *the web had no results for this query* — the most
misleading shape a missing dependency can take, because every layer above
behaves correctly on it.

## The change

**`BrowserSearchUnavailableError`**, raised by `_run` when `subprocess.run`
reports `FileNotFoundError`. The message names the binary and says it is not a
pip package and not in the image, because those are the two things someone
debugging it will otherwise assume.

**`_search_and_process` re-raises it** ahead of the broad handler. Degrading
every failure to an empty list is right for a search target that returned
garbage and wrong for a tool that is absent; the distinction is the whole point.
A real failure still degrades — a test pins that, so the tolerance is not traded
away for the signal.

**`main()` refuses to start** when `playwright_cli_available()` is false. A
server whose every request shells out to a missing binary would otherwise start,
report healthy, and answer every query with nothing. The check lives in `main()`
rather than `create_app()` so tests can still build the app — the seam matters,
because a hard failure in the factory would be a worse trade than the bug.

**CLAUDE.md marks the provider host-only**, next to the command that starts it.

## What this does not do

It does not make the provider work in the container. Installing `playwright-cli`
would mean node plus a browser runtime in an image just reduced from 7.64GB to
2.04GB, for a fallback that is slow (~30–50s/query) and lower quality than the
SerpAPI path it backs up. That trade deserves its own decision rather than being
made by a bug fix.

## Testing

Five tests, RED first: the error names the binary; it survives the search path as
a distinct signal; a *real* failure still returns no results; availability is
reportable without running a search; and `main()` exits non-zero.

Mutation-checked: removing the re-raise makes the missing binary look like an
empty result set again, and the second test fails.

One test of mine was wrong before the code was. It asserted the refusal message
appeared on captured stderr, but `SystemExit("...")` carries its message in the
exception and the interpreter is what prints it, so nothing reached `capsys`. The
contract to assert is the exception's payload.
