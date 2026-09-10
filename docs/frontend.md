# Frontend development

[← Back to README](../README.md)

This guide covers the React/Vite development workflow and local administration and observability surfaces.

## Development workflow

The `web/` directory contains a React 19 + Vite + TypeScript single-page app. It runs against the FastAPI backend at port 7860 and, in development, Vite proxies the backend routes it calls — `/api/*`, `/health`, `/auth/*`, `/me`, `/admin/*`, `/analytics/*`, `/chat/*`, `/search/*`, `/tool/*` — through to port 7860. A prefix missing from that list is the usual cause of a panel or page that fails only in dev: the request hits the Vite dev server and 404s instead of reaching FastAPI. The `/chat`, `/search` and `/tools` proxies carry a `bypass` so a page navigation (`Accept: text/html`) is served the SPA instead of being forwarded to FastAPI — without it, a refresh on those pages 404s in dev only. `/tools` needs its own entry because proxy keys match by prefix and `/tool` would otherwise swallow it.

### Pages

The four surfaces are routes, not tabs. `web/src/router.tsx` holds the route
list and a ~70-line history router; `App.tsx` is a shell that renders the
topbar, the global panels, and whichever page the route selects. Each page owns
its state, so navigating away discards it.

`ROUTES` in `web/src/router.tsx` is mirrored in two other places, and all three
must be changed together to add a page: the `bypass` entries in
`web/vite.config.ts` (only for prefixes the backend also serves) and the
`spa_page` handler in `src/internal/servers/web/app.py`.

```bash
cd web && npm install && npm run dev   # dev server at http://127.0.0.1:5173
cd web && npm run build                # production bundle → web/dist/ (served by FastAPI)
cd web && npm run typecheck            # TypeScript check
cd web && npm run test -- --run        # Vitest unit tests
```

> **Open the Vite dev server at `http://127.0.0.1:5173`, not `:7860`.** Port 7860
> serves the last `npm run build` bundle from `web/dist` (stale until you rebuild);
> `:5173` is the live source with hot-reload. If a UI change "doesn't show up",
> you're probably looking at `:7860` — rebuild, or use `:5173`.

## Logging in (Search / Chat / Tools pages)

The **Search** page calls `POST /search/send-search-message`, which resolves the
caller with `resolve_active_user` and returns `401 Authentication required` for
anonymous requests — unlike `/api/agent`, this surface requires a real user
rather than falling back to the anonymous `["public"]` identity. `AGENTIC_SEARCH_DEV_ADMIN`
does **not** cover this; that bypass applies only to the admin routers.

There is no login UI in the app. For local dev, open:

```
http://127.0.0.1:5173/dev-login.html
```

and click **Log in**. It registers `dev@localhost` / `devpass` (idempotent), logs
in, and confirms via `/me`. That sets the `fastapiusersauth` cookie on the dev
server's origin, which is where the pages need it. Return to the dashboard and
Search works.

Two caveats:

- `AGENTIC_SEARCH_WEB_DB_PATH` defaults to `:memory:`, so the user disappears when
  the backend restarts — click **Log in** again, or start the backend with a file
  DB (`AGENTIC_SEARCH_WEB_DB_PATH=data/web.db`) to make it stick.
- `dev-login.html` is served by the Vite dev server only. It is not a build input,
  so `npm run build` does not emit it into `web/dist/`. Keep it that way: the first
  registered user becomes an **admin**, so a self-registering page must never be
  reachable from a real deployment.

## Admin dashboard

The web UI includes admin panels — **Connectors, Manage tools, History, Admin
overview, Analytics** (top-bar buttons + the observability panels). The tools
button is labelled *Manage tools*, not *Tools*: the `/tools` nav link goes to the
tool-agent page instead, and the two used to share a label. They call the
`/admin/*` and `/analytics/*` routers, which are **admin-authenticated**. Two
things make them work locally:

1. **Proxy** — the Vite dev server already forwards `/admin` and `/analytics` to
   the backend (above), so requests reach FastAPI on `:7860`.
2. **Auth** — the panels are gated by `make_require_admin`. For local dev, set
   `AGENTIC_SEARCH_DEV_ADMIN=1` (default off) so every admin endpoint accepts the
   request as a dev admin — **no token or cookie needed**:

```bash
AGENTIC_SEARCH_DEV_ADMIN=1 PYTHONPATH=src:. \
  uvicorn src.internal.servers.web.app:app --host 127.0.0.1 --port 7860
```

The startup log prints `ADMIN AUTH BYPASSED …` when it's active. **Dev only —
never set this in production.** Without the flag the endpoints return `401` and
you'd need an admin JWT in the `fastapiusersauth` cookie instead.

Clicking a top-bar button toggles its panel below the observability panels
(scroll down to see it). Empty panels — e.g. **Connectors** showing "No connectors
configured", **Manage tools** empty — are expected when nothing is configured locally,
not an error.

## Dev console

A gated, dev-only console for inspecting the backend servers from the web UI. **Off by default** — enable both flags (never in production):

```bash
  # backend: mount /api/debug/* on the web app
AGENTIC_SEARCH_DEBUG_PANELS=1 PYTHONPATH=src:. \
  uvicorn src.internal.servers.web.app:app --host 127.0.0.1 --port 7860
  # frontend: reveal the "Console" toggle in the top bar
cd web && VITE_DEBUG_PANELS=1 npm run dev
```

Click **Console** in the top bar to open it. **Retrieval Lab** runs a query against each per-mode endpoint (`sparse` / `dense` / `hybrid` / `graph`) via the `POST /api/debug/retrieval/{mode}` proxy and shows results side by side, surfacing **503** (dense not configured → hybrid collapses to sparse) and **404** (endpoint not mounted, e.g. against `demo.py`) explicitly instead of as a generic error. (Health/workers/chat-trace panels land in later phases — see [the plan](superpowers/archive/plans/2026-06-29-backend-observability-uis.md).)

## UI features

**Streaming answers** (`AnswerPanel.tsx` → `ProgressLog`) — every query streams over SSE; `streamAgent` (`web/src/api.ts`) drives the UI from the `progress` / `claim` / `trace` / `approval_required` / `answer` / `done` events (full schema in the [SSE event table](architecture.md#intent-routing)). While the agent runs, a live **Agent reasoning** log renders one row per turn (`⟳ Turn N · writing answer…` active, `✓ Turn N · <tool> · N docs` completed) and answer tokens stream in as markdown; on `done` the log collapses to a one-line summary (`✓ 3 turns`) with a **show reasoning ▸** toggle that re-expands the full trace. Backend side, each turn fires the `on_turn` callback (`OnTurnCallback`) → a `progress` event, while token / tool-call / citation packets originate from `AgentQueueManager` → `Emitter`. The **New** button (`handleNewSession`, on the Assistant page) aborts any in-flight request and clears answer / citations / documents / messages / intent; an in-flight turn is cancellable via the stop-signal fence.

**On the Assistant page the answer arrives claim by claim, not token by token**
(`AssistPage.tsx`). The grounded path's answer *is* the join of the claims it has
verified, so `streamAgent` accumulates each `claim` event onto the running answer
separated by a space and renders it immediately. The terminal `answer` event then
**overwrites** that accumulation rather than appending to it — it is the
authoritative text, which is what makes a dropped `claim` event cosmetic instead of
lossy. `trace` events feed the Dev Console control-flow panel and
`approval_required` events populate the pending-approval list.

**Markdown rendering** — Answers render via `react-markdown`: headings, bold/italic, inline code, code blocks, and ordered/unordered lists. Citation markers (`[D1]`, `[D2]`, …) become anchor links that scroll the page to the matching source card.

**Chat history** — Session timeline renders as a chat bubble layout: user messages right-aligned, assistant messages left-aligned. System messages are filtered out. Keys are stable against message prepend/removal.

**Source cards** (`SourceGrid.tsx`) — `SourceGrid` is a thin mapper over a controlled `SourceCard` (memoised, per-document, owning its own `expanded` / `copied` state). Each card renders one `SourceDocumentView` (`{ id, citation, title, content, url, score, metadata }`) and:
- collapses content to 3 lines by default (`source-content--clamped`); **show more ▾** / **show less ▴** toggles per card.
- a **⎘ copy** button copies the full content and flips to "copied ✓" for 1.5 s.
- carries `id="source-{citation}"` so `[D1]`-style anchor links from the answer scroll to it.
- color-codes the relevance score via `scoreColor()` (green ≥ 0.7, amber ≥ 0.4, orange > 0, grey for 0).
- tags the source provider with a colored pill via `SOURCE_COLORS` (Browser Retrieval, SerpAPI, Local Retrieval, All Active Sources; grey fallback).

Source cards are frontend-only (no dedicated backend endpoint): they are populated from the `documents` array of the `POST /api/agent` response (see the [Web backend API](api-reference.md#web-backend-api)); the retrieval server returns the same fields as `results[]` from `POST /search`. Inspect that backing data with:
```bash
curl -s -X POST http://localhost:7860/api/agent \
  -H "Content-Type: application/json" \
  -d '{"query": "What is FAISS?", "top_k": 3}' \
  | python -c "import sys, json; [print(d['citation'], round(d['score'],2), d['title']) for d in json.load(sys.stdin)['documents']]"
  # → [D1] 0.81 FAISS: A Library for Efficient Similarity Search ...
```

**Tool Call Trace Panel** — When the agent runs in `tool` mode, a panel below the answer shows every tool call: name, status (✓ / ✗), arguments as JSON, result summary (first 200 chars or "N items" for lists), and latency in ms. Failed calls render with a red border and the error message.

**Intent-adaptive layout** — `App.tsx` reads `response.intent` (set from the `done` SSE event via `setIntent`) and applies `intent-${intent}` to the `.results-layout` container; when `intent` is undefined no class is added and the layout falls back to the default single-column stack. The behaviour is **CSS-only** — `styles.css` rules consume the class to reflow the existing panels (no extra components), keyed off stable hooks `.answer-column`, `.sources-panel`, `.session-panel`, and `.tool-trace-panel`:

| Intent | `.results-layout` class | Layout |
|--------|--------|--------|
| `search` | `intent-search` | Single column; `.sources-panel` gets a highlighted border; `.session-panel` dimmed |
| `chat` | `intent-chat` | `.answer-column` + `.session-panel` side-by-side (≥720 px); `.sources-panel` full-width below |
| `tool` | `intent-tool` | `.tool-trace-panel` full-width hero; `.sources-panel` and `.session-panel` side-by-side below |
| narrow (≤720 px) | — | All intents fall back to a single-column grid stack |

The intent itself comes from the backend's routing decision — see the `response.intent` contract under the [Web backend API](api-reference.md#web-backend-api). No new endpoints back this feature; the layout is a pure function of that one field.

`intent` describes what surfaced, while the `done` payload and request inspector expose why and how it ran. Auto requests may include `route`, `route_degraded`, `search_mode`, and `external_provider` metadata. For example, a bare `GRPO` query can display search results from SerpAPI or browser retrieval after the internal sufficiency gate fails; if no provider returns evidence, the sources panel remains empty and the answer reports no results. See [API request routing](request-routing.md).

**Intent badge** (`AnswerPanel.tsx`) — a pill under the answer summarising what ran, derived from `response.intent` + counts: `Searched · 5 sources`, `Answered · 3 citations`, or `Used tools`. Hidden when the answer is empty or the intent is undefined.

**Example-query chips** (`SearchComposer.tsx`) — three chips under the search box, one per routing intent, that populate and run a representative query in a single click so the intent router can be exercised without knowing what triggers each path: 🔍 `find the onboarding checklist` (search), 💬 `explain how FAISS indexing works` (chat), 🛠 `summarize the latest sales figures and chart them` (tool). The chips are hidden while a request is in flight.

**Components** (`web/src/components/`) — each panel is a focused, independently tested unit:

| Component | What it does |
|-----------|--------------|
| `SearchComposer` | Single input box (no mode selector), per-intent example-query chips, source-provider / retrieval-URL / top-K controls, Cmd+Enter submit |
| `AnswerPanel` | Streamed markdown answer + intent badge + `[D1]` citation anchor links |
| `SourceGrid` | Expand/collapse source cards with copy-to-clipboard and citation `id` anchors |
| `SessionTimeline` | Chat-bubble history (user right, assistant left; system filtered) |
| `ToolCallTracePanel` | Per-tool-call trace (name, ✓/✗ status, JSON args, result summary, latency) for `tool` intent |
| `AdminOverview` | Single-call health snapshot — connectors, indexing, users, auth, models, tools, analytics with a composite health score |
| `AnalyticsDashboard` | Usage breakdowns by LLM, persona, and flow (`getAnalyticsBy*`) |
| `ConnectorPanel` | Lists configured connectors and their sync/index status |
| `QueryHistoryPanel` | Per-user query history with CSV export (`getQueryHistory`) |
| `ToolPanel` | Admin view of MCP/OpenAPI tools registered via `tool_registry` |

API client functions live in `web/src/api.ts`: `runAgent` / `streamAgent` (SSE), `createSession` / `getSession`, `getAdminSummary`, `getAnalyticsByLLM` / `getAnalyticsByPersona` / `getAnalyticsByFlow`, `getQueryHistory`, `getAuditSummary`, `submitFeedback`.

**Feedback loop (UI → fine-tuning)** — the thumbs bar under the answer (`AnswerPanel`) posts session thumbs to `POST /api/feedback` via `submitSessionFeedback(sessionId, signal, target)`: a thumbs-up goes as `overall`, a thumbs-down first asks whether the *Sources* (`retrieval`), the *Answer* (`generation`) or *Both* (`overall`) were at fault. `submitFeedback(chatMessageId, isPositive, feedbackText?)` posts per-message like/dislike to `POST /chat/create-chat-message-feedback` (exported; no component calls it today); `QueryHistoryPanel` can filter sessions by `feedback_type` (`like` / `dislike`). These ratings are exactly what `load_feedback_examples` reads back into [feedback-driven GRPO](training-and-evaluation.md#fine-tune-from-user-feedback) — the human-feedback signal that fine-tunes the policy.
