# Declare the HTTP client src/ actually imports

## Goal

Make the dependency files name the HTTP client this repo imports, so an install
from `requirements.txt` or `pip install .[mcp]` is guaranteed to provide it.

## The problem

Three files declared `httpx2>=0.28.0`:

```
requirements.txt:13
requirements-unit-test.txt:11
pyproject.toml:14        (the `mcp` optional-dependency extra)
```

Nothing in the repository imports `httpx2`. A search for `import httpx2` /
`from httpx2` over `src/`, `tests/` and `examples/` returns nothing.

`httpx` and `httpx2` are unrelated distributions. `httpx2` is pydantic's
successor to httpx (`github.com/pydantic/httpx2`), it installs a package
imported as `httpx2`, and it is at version 2.4.0. So the `>=0.28.0` floor is
httpx's version numbering applied to the wrong distribution name — trivially
satisfied, and constraining nothing. The shape of it (name changed, floor left
behind, imports untouched) reads as a find/replace over the requirement lines.

Meanwhile the code imports `httpx`, at these async call sites:

- `src/internal/mcp_server/utils.py:43` — the shared `AsyncClient` singleton,
  used by `retrieval_client.py`, which also catches `httpx.HTTPStatusError`,
  `httpx.TimeoutException` and `httpx.TransportError`
- `src/internal/search/stages.py:172` — the reranker call
- `src/internal/servers/billing/service.py:61`, `billing/api.py:228`
- `src/internal/servers/web/debug_router.py:65`,
  `src/internal/servers/features/hooks/api.py:177` (sync `httpx.Client`)

`httpx` was declared nowhere. It resolved only because `mcp`, `fastmcp`,
`openai` and `starlette` each depend on it. That works until one of them drops
the dependency or a resolver picks a version outside what these call sites
expect, at which point the failure is an ImportError in the MCP server rather
than anything pointing at the requirement files.

## What this is not

The request that surfaced this was to use aiohttp as the httpx2 transport, for
concurrency. Two findings redirected it:

1. There is no httpx2 code to give a transport to. `httpx2` is declared and
   never imported.
2. `httpx-aiohttp` (the package that provides `AiohttpTransport`, and does
   support httpx2 through its `[httpx2]` extra) requires `aiohttp>=3.10.0,<4`.
   This repo pins `aiohttp==3.9.3`, and that pin is load-bearing: the
   `openai>=1.0.0,<3` cap in `requirements-unit-test.txt:35-41` exists
   *because* openai 3.x vendors `httpx_aiohttp`, whose transport references
   `aiohttp.SocketTimeoutError` — an attribute aiohttp added in 3.10.
   Confirmed locally: `hasattr(aiohttp, "SocketTimeoutError")` is `False` on
   the pinned 3.9.3.

The concurrency premise also points the wrong way. The paths that actually
fan out already use aiohttp directly, not httpx:
`src/context/retrieval/client.py`, `src/internal/tools/search.py`,
`src/internal/tools/public_data/_http.py`, `src/model/serving.py:282`,
`src/internal/servers/web_search/google.py:132`. The httpx sites are MCP,
rerank and billing — none of them concurrency-bound.

So the aiohttp-transport work is deliberately out of scope here. Lifting the
aiohttp pin is a separate decision, and the existing comment asks for it to be
made "deliberately, with aiohttp, not by drift."

## Architecture

Replace the `httpx2` line in all three files with `httpx>=0.28.1,<1.0`.

**The floor.** 0.28.1 is what `fastmcp` already requires (`httpx>=0.28.1,<1.0`)
and what is installed and exercised. Matching it exactly means this declaration
cannot cause a resolution conflict with the dependency that was previously
supplying httpx.

**The cap.** `<1.0` matches `fastmcp` and `openai`, and follows this repo's
habit of capping below a known-breaking major (see the `fastapi<0.137` and
`openai<3` caps in the same files).

**No source changes.** Every `import httpx` is already correct. This is a
declaration fix, not a migration.

## The guard

`tests/unit/test_http_client_declaration.py` holds the invariant in three
parts, so a future find/replace cannot reintroduce the split:

- `test_src_imports_httpx_not_httpx2` parses every `.py` file under `src/` with
  `ast` and asserts the top-level imports contain `httpx` and not `httpx2`.
  This is the premise the other two rest on; if the repo ever genuinely moves
  to httpx2, this test fails first and says so.
- `test_requirements_declare_httpx_and_not_httpx2` asserts both requirements
  files declare `httpx` and neither declares `httpx2`.
- `test_mcp_extra_declares_httpx` asserts the same for the `mcp` extra, which
  is what `pip install .[mcp]` resolves for the MCP server.

Mutation-checked: restoring `httpx2>=0.28.0` in all three files turns the
second and third red. The first stays green, correctly — it describes `src/`,
which the mutation did not touch.
