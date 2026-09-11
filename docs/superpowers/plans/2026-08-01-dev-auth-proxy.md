# Plan: dev auth reachability

Spec: [2026-08-01-dev-auth-proxy-design.md](../specs/2026-08-01-dev-auth-proxy-design.md)

## Task 1 — Add `/auth` and `/me` to the Vite dev proxy

Edit `web/vite.config.ts`, adding both prefixes to the existing `server.proxy`
map pointing at `http://127.0.0.1:7860`.

**Verify:** with the backend on `:7860` and the dev server on `:5173`,
`curl -X POST http://127.0.0.1:5173/auth/login -d 'username=…&password=…'`
returns `200` (it returned `404` before). ✅ done

## Task 2 — Add the dev login helper page

Create `web/dev-login.html`: a single button that calls `/auth/register` (ignoring
"already registered"), then `/auth/login`, then `/me` to confirm, printing each
result. Include a comment stating it is dev-server-only and must never be served
in production.

**Verify:** `http://127.0.0.1:5173/dev-login.html` returns `200`; clicking the
button yields a `fastapiusersauth` cookie on the `127.0.0.1:5173` origin, after
which `POST /search/send-search-message` through the proxy returns hits. ✅ done

## Task 3 — Confirm the helper cannot ship to production

Run `npm run build` and inspect `web/dist/`.

**Verify:** `dist/` contains `index.html` and `assets/` only — no
`dev-login.html`. ✅ done

## Task 4 — Update `docs/frontend.md`

Fix the stale proxy prefix list on line 9 (currently missing `/health`,
`/search`, `/tool` as well as the two added here) and document the helper page
in the dev workflow section, next to the existing dev-admin guidance.

**Verify:** the documented prefix list matches `web/vite.config.ts` exactly.

✅ Confirmed on 2026-09-10: the documentation includes the auth and page proxies;
the `/tools` page bypass is also described.

## Task 5 — Regression check

**Verify:** `cd web && npm run typecheck` clean; `npm run test -- --run` green;
`pytest` unaffected (no Python changes, run as a sanity check).

✅ Frontend reverified on 2026-09-10 with `npm run test:ci`: typecheck,
production build and 215 tests pass. `web/dist` contains only `index.html` and
assets, with no `dev-login.html`.
