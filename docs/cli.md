# Command-line tools (`cli/`)

The `cli/` directory holds the project's **Go command-line clients**. They are
the *terminal / scriptable* front door to the running backend — for humans at a
keyboard and for shell scripts / CI. They are **not** driven by the MCP server,
and they hold no logic or data of their own: each command parses flags, calls
the backend's HTTP `/api/*` endpoints, and prints the result.

There are exactly **two** binaries today:

| Binary | Purpose | Source |
|---|---|---|
| `query` | One-shot search + agentic answer (sources + answer + `session_id`) | [cli/cmd/query](../cli/cmd/query/main.go) |
| `memory` | Manage user memory (add / list / search / consolidate / profile / curate) | [cli/cmd/memory](../cli/cmd/memory/main.go) |

> Note: things like **connectors** and **evals** are *not* CLI tools — they live
> in the Web UI and the backend REST API. See [Extending the CLI](#extending-the-cli).

There is also no Python CLI. `src/cli/` used to hold one, but it was a duplicate
of the Go `query` binary — same flags, same `POST /api/agent`, same output — so it
was removed.

`cli/` contains considerably more than these two binaries — including a complete
chat TUI — but none of it has an entry point today. See
[The rest of `cli/`](#the-rest-of-cli-built-tested-not-wired-up) before assuming a
package there is live.

## Three doors to the same backend

The CLI is one of three independent front ends over the same backend and
services — you pick by *who is driving*:

```
   You, in a terminal / a script / CI   ──►  Go CLI (query, memory)   ─┐
   You, in a browser                    ──►  Web UI                   ─┼──►  backend :7860  ──►  services
   An LLM agent                         ──►  MCP tools (:8090)        ─┘
```

They are alternative front doors, not layers that call each other: the CLI is
*your* door, MCP is the *agent's* door ([docs/mcp.md](mcp.md)), the Web UI is the
*browser* door. All three ultimately reach the same routers and services (e.g.
memory converges on `src/internal/memory/service.py`).

## Prerequisites

The CLI talks to a **running backend** (it is a client, not a standalone tool):

```bash
# terminal 1 — retrieval server (see the main README for options)
python3 -m src.internal.servers.retrieval.demo --corpus demo

# terminal 2 — web backend on :7860 (the CLI's default target)
PYTHONPATH=src:. uvicorn src.internal.servers.web.app:app --host 127.0.0.1 --port 7860
```

## Build

```bash
cd cli
go build -o ../bin/query  ./cmd/query
go build -o ../bin/memory ./cmd/memory
# then run ../bin/query and ../bin/memory, or `go run ./cmd/memory -- <args>`
```

Go tooling: `go test ./...`, `go vet ./...`, `go build ./...` (run from `cli/`).
None of the CI workflows (`ci.yml`, `eval-gate.yml`, `publish-image.yml`) has a
`setup-go` step, so **nothing runs these for you** —
run them yourself before pushing Go changes.

## `query` — search + answer

Positional args are the question; flags tune it.

```bash
query "What is FAISS?"
query -top-k 8 "Compare dense and sparse retrieval"
query -session-id <id> "follow-up question"        # resume a chat session
```

Flags: `-url`, `-token`, `-user-id`, `-email`, `-secret` (auth — see below),
`-top-k N` (default 5), `-session-id S`, `-width N` (markdown wrap; 0 = auto).
It hits `POST /api/agent` and renders the source cards, the answer, and the
`session_id`.

## `memory` — manage user memory

```bash
memory add "User prefers window seats"
memory list
memory search -top-k 5 "seating"   # flags BEFORE the query — see the note below
memory consolidate                 # dedup + resolve tagged conflicts (-no-conflict to dedup only)
memory profile                     # show the stored profile
memory profile -generate           # (re)build the profile via the LLM
memory curate                      # LLM reconciles memories from the conversation
memory curate -session-id <id>     # ...restricted to one session
```

Each subcommand maps to an `/api/memory/*` endpoint (see
[docs/api-reference.md](api-reference.md)). `profile -generate` and `curate`
require the backend to have an LLM configured (else they return a 503, surfaced
as a non-zero exit).

> **Flags must come before the positional text.** Go's `flag` package stops
> parsing at the first non-flag argument, and `add`/`search` join everything left
> over into the text or query. So `memory search "seating" -top-k 5` does **not**
> set `-top-k` — it searches for the literal string `seating -top-k 5` and
> silently uses the default. Write `memory search -top-k 5 "seating"`. Same for
> `memory search -json "seat"`. This bites quietly: there is no error, just a
> query with your flags glued onto it.

`-session-id` may only name a session **you own**. Anything else — someone
else's, or one with no owner — yields `session not found, or not readable by
you`. One message covers both causes, so it confirms nothing about another
user's sessions.

> **Curating from conversations now requires signing in.** Sessions started
> without a token are stored with **no owner**, and the no-flag path is scoped by
> `WHERE user_id = ?`, which never matches NULL. So neither `memory curate` nor
> `memory curate -session-id <id>` can reach an unauthenticated caller's own
> conversations — the flag was previously the only route to them, and it is now
> closed too. Pass `-token` or `-user-id` to curate. `memory add` and the other
> subcommands are unaffected.

### JSON output (scripting)

Add `-json` to any `memory` subcommand to print the raw response as JSON instead
of the human-formatted lines — for pipelines and CI:

```bash
memory list -json | jq -r '.memories[].text'
memory search -json "seat" | jq '.results[] | {id, score}'
```

## Authentication

Config comes from a file (`~/.config/agentic-search/config.json`) overridden by
environment variables, overridden by flags:

| Setting | Env var | Flag | Default |
|---|---|---|---|
| Backend URL | `AGENTIC_SEARCH_URL` | `-url` | `http://localhost:7860` |
| Bearer token (PAT/JWT) | `AGENTIC_SEARCH_PAT` | `-token` | — |
| Mint a JWT for a user | — | `-user-id` (+ `-email`) | — |
| JWT signing secret | `AGENTIC_SEARCH_AUTH_SECRET` (then `AUTH_SECRET`) | `-secret` | — |

> **The two binaries disagree about whether auth is optional.** `memory` treats
> a missing token as fine and sends the request anyway
> ([main.go:80](../cli/cmd/memory/main.go#L80) deliberately ignores the resolve
> error). `query` treats it as fatal and exits 1 with
> `provide -token, set AGENTIC_SEARCH_PAT, or pass -user-id to authenticate`
> before contacting the backend at all. So the "unauthenticated is fine" note
> below holds for `memory` but **not** for `query` — run `query` with `-token`,
> `AGENTIC_SEARCH_PAT`, or `-user-id` even against a local backend that would
> have accepted an anonymous caller.

- **Unauthenticated is fine for local/research use** — with no token the backend
  treats the caller as the default user (`default_user`). That bucket is
  **shared**: every anonymous caller reads and writes the same memories, which is
  what makes the single-operator local flow work and what makes it wrong for a
  deployment with more than one person. Set
  `AGENTIC_SEARCH_MEMORY_REQUIRE_AUTH=1` on the backend to refuse anonymous
  memory callers instead; the CLI then needs `-token` or `-user-id`.
- To act as a specific user without a pre-issued token, pass `-user-id alice`;
  the CLI mints an HS256 JWT signed with `AGENTIC_SEARCH_AUTH_SECRET` (falling
  back to `AUTH_SECRET`). That secret **must match** the backend's
  `AGENTIC_SEARCH_AUTH_SECRET` for the token to validate.

## Exit codes

The `memory` binary maps failures to the semantic codes in
[cli/exitcodes](../cli/exitcodes/codes.go), so scripts can branch on them:

| Code | Meaning | Code | Meaning |
|---|---|---|---|
| 0 | success | 5 | backend unreachable |
| 1 | general error | 6 | rate limited (429) |
| 2 | bad request / bad args (400/422) | 7 | timeout (408/504) |
| 3 | not configured | 8 | server error (5xx) |
| 4 | auth failure (401/403) | 9 | not available (404) |

## The rest of `cli/`: built, tested, not wired up

`cli/` was lifted from a larger upstream Go CLI and still carries that CLI's whole
feature set, even though only `query` and `memory` have entry points. A
2026-08-06 audit ([spec](superpowers/specs/2026-08-06-cli-simplification-design.md))
found **4,358 of 6,676 LOC unreachable from either binary**. It is deliberately
kept — it compiles, it is test-covered, and it is the starting point for a richer
client — but **nothing runs it today**, so do not assume a package here is live
just because it exists:

| Package | LOC | Status |
|---|---|---|
| `tui/` (13 files) | ~2,140 | A full bubbletea chat TUI — splash, SSH auth, status bar, scrollback viewport, `/configure`. Complete and tested; **no `main` package launches it.** Needs an entry point (a `cmd/tui` binary, or bare `query` with no args). |
| `parser/`, `api/stream.go`, `models/events.go` | 976 | NDJSON stream-event parsing for `POST /chat/send-chat-message`. Reached only from `tui/`. The endpoint exists but streams **SSE** (`data: {...}` frames), which `ParseStreamLine` rejects as malformed, so the parser needs an SSE reader before the TUI can use it. |
| `overflow/`, `fsutil/`, `browser/` | 458 | Pager writer, skill-file installer, browser opener — TUI support code. |
| `starprompt/` | 83 | One-time "star us on GitHub" prompt, shown before the TUI. |
| `embedded/` + `SKILL.md` | 7 + doc | An agent skill embedded into a binary named `agentic-search`, which this repo does not build. |
| `version/` | 168 | Gates on "backend >= 3.0.0" via `GET /api/version` — **the backend does not serve that endpoint**, so this cannot pass as written. |
| `config/experiments.go` | 46 | Feature-flag registry; its only flag (`stream_markdown`) is read only by `tui/`. |

`ClientAPI` has 18 methods; the two binaries call 8 (`QueryAgent` plus the seven
`/api/memory/*` calls). Of the other 10, most back the TUI, but three target
endpoints that **do not exist** on this backend — `GET /api/version`,
`POST /api/chat/stop-chat-session/`, `POST /api/user/projects/file/upload` — and
two (`GetBackendVersion`, `Search`) have no caller at all, not even in `tui/`.
Anything you wire up will need those three either implemented server-side or
pointed elsewhere.

### `cli/models/events.go` was reconstructed

This file **was not in git history**. The bare `models/` rule in `.gitignore`
(meant for the ML checkpoints in `src/model/models/`) matches *any* directory
named `models` at any depth, so it silently excluded `cli/models/events.go`. The
file existed in exactly one working copy and nowhere else — not on any branch,
not in any worktree. Two consequences:

- **`cli/` had never built from a clone.** `git archive HEAD cli && go build ./...`
  failed with `undefined: models.StreamEvent` and ten more. Neither CI workflow
  has a `setup-go` step, so nothing caught it.
- It was rebuilt from its consumers — `parser/parser.go`'s constructors, the
  assertions in `parser/parser_test.go`, and the field reads in `tui/app.go` —
  which pin every struct field and the `StreamEvent` interface by compilation plus
  a 419-LOC test suite. The `EventType()` constant *values* are **not** pinned,
  because nothing reads them today; they mirror the wire `type` strings. If you
  wire up the TUI and hit an event-name mismatch, that is the one place to look.

`.gitignore` now carries a `!cli/models/` negation, so the next file added under
`cli/models/` cannot vanish the same way.

### Other findings from the audit

- **`src/cli/` was deleted.** It was a duplicate of the `query` binary — same
  flags, same `POST /api/agent`, same output — referenced nowhere and documented
  nowhere. Its four unit tests went with it.
- **Nine files were gofmt-misformatted** on `main`, all from an
  `Assistant`->`Agent` rename that never re-ran `gofmt`. Now formatted. The
  leftover alignment is what revealed that `MessageIDEvent`'s field had been
  `ReservedAssistantMessageID` before that rename.
- **`cli/_version.py`** is an orphan: it derives a version from `GITHUB_REF_NAME`
  for a `[tool.hatch.build.targets.wheel.hooks.custom]` config that does not
  exist, since `pyproject.toml` uses setuptools. Nothing imports it. Left in
  place, flagged here.
- Two behavior quirks are **documented rather than changed**, since fixing either
  alters what a working command does today: `memory`'s flags must precede the
  positional query or they are silently swallowed into it (see
  [the note under `memory`](#memory--manage-user-memory)), and `query` requires a
  token while `memory` does not (see
  [the note under Authentication](#authentication)).

## Extending the CLI

The CLI covers only `query` + `memory` today. Because it is a thin client over
the REST API, adding a command for another capability (connectors, evals,
health, workers, …) is mechanical and self-contained:

1. add a typed method to [cli/api/client.go](../cli/api/client.go) that calls the
   endpoint (and to the `ClientAPI` interface),
2. add request/response structs to [cli/models](../cli/models/models.go),
3. add a subcommand (a new `cli/cmd/<name>` binary, following `cli/cmd/memory`).

`ClientAPI` carries 18 methods but the binaries call only 8, so check
[The rest of `cli/`](#the-rest-of-cli-built-tested-not-wired-up) before adding
one — the method you need may already be there. Note which endpoints do not
exist server-side; a method's presence is not evidence that its endpoint works.

Keep every command a **thin translator** (parse flags → call `/api` → print) —
never let logic or local state live in the CLI; that belongs in the backend
service layer so all three front doors share one source of truth.
