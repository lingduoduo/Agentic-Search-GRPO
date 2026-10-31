# Frontend

[← Back to README](../README.md)

The React web UI lives in its own repository,
[Agentic-Search-UI](https://github.com/lingduoduo/Agentic-Search-UI). Its
[docs/frontend.md](https://github.com/lingduoduo/Agentic-Search-UI/blob/main/docs/frontend.md)
covers pages, components, the dev console, and the UI features. This page
covers only what the backend has to provide for it.

## Running it against this backend

Clone the frontend next to this repository, start the backend on port 7860,
then run the Vite dev server:

```bash
git clone https://github.com/lingduoduo/Agentic-Search-UI ../Agentic-Search-UI
cd ../Agentic-Search-UI && npm install && npm run dev   # http://127.0.0.1:5173
```

Vite proxies the backend routes to `http://127.0.0.1:7860`. Override that with
`BACKEND_URL`. `bin/run_web_stack.sh` starts the frontend from `../Agentic-Search-UI`
(override with `FRONTEND_DIR`).

## Serving it in production

The frontend ships as its own nginx image. nginx serves the bundle and proxies
the API prefixes to this backend, so the browser sees one origin: auth cookies
and SSE need no CORS. The compose stack builds it under an opt-in profile:

```bash
docker compose -f docker/docker-compose.yml --profile frontend up --build   # UI on :8080
```

The build context defaults to `../../Agentic-Search-UI` relative to `docker/`.
Override it with `AGENTIC_SEARCH_FRONTEND_CONTEXT`, or set
`AGENTIC_SEARCH_FRONTEND_IMAGE` to run a published image.

The backend image itself is API-only. On `/`, `/assist`, `/search`, `/chat` and
`/tools` it serves a minimal inline shell, unless `AGENTIC_SEARCH_FRONTEND_DIST`
points at a built `dist/` (with `index.html` and `assets/`). In that case it
serves that bundle itself, which is handy for single-process local use.

The SPA routes are mirrored in four places that must change together:
`ROUTES` in the frontend's `src/router.tsx`, its `vite.config.ts` bypass entries,
its `nginx/default.conf.template`, and `spa_page` in
`src/internal/servers/web/app.py` here.

## Logging in (Search / Chat / Tools pages)

`POST /search/send-search-message` and the other direct-engine pages require
a real user (`resolve_active_user`). Without one they return
`401 Authentication required`. The frontend's dev server serves a `dev-login.html`
helper that registers and logs in `dev@localhost`; see the frontend docs.
The admin panels (`/admin/*`, `/analytics/*`) accept any request as a dev
admin when the backend runs with `AGENTIC_SEARCH_DEV_ADMIN=1`. **Dev only.**

## UI features

The intent-adaptive layout, streaming answers, source cards, and the other UI
behavior are documented in the frontend repository's
[docs/frontend.md](https://github.com/lingduoduo/Agentic-Search-UI/blob/main/docs/frontend.md#ui-features).
The backend contract they consume is in the [Web backend API](api-reference.md#web-backend-api).
