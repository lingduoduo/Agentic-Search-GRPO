# Authentication and route protection

The web backend accepts HS256 bearer tokens and the `fastapiusersauth` login
cookie. `POST /auth/register` creates an account; the first registered account
becomes admin. Registration selects that role and rejects duplicate emails in
one database write transaction, so concurrent requests cannot both become the
first admin, including across connections to the same SQLite file. Duplicate emails
return 400 without replacing the existing account.

`POST /auth/login` verifies the password and sets the HTTP-only
cookie. The development helper and proxy setup are described in
[Frontend](frontend.md#logging-in-search--chat--tools-pages).

New passwords use PBKDF2-HMAC-SHA256 with 600,000 iterations and an independent
random 16-byte salt. The stored value includes its algorithm, iteration count,
salt and digest; verification compares digests in constant time. Existing
fixed-salt hashes remain usable and are upgraded on successful active-account
login. Failed logins do not alter the hash, and the upgrade preserves other
account fields and never replaces a newer password hash.

## Current account state

On the web app, identity must resolve to an active record in its user store.
Deletion and local or SCIM deactivation invalidate access immediately; admin authorization
uses the current stored role, not an old role claim in a JWT. Configured
`super_users` still grant admin access to active accounts, including user
management and the admin permission returned by `/me/permissions`. `/me` and
`/me/permissions` return 401 for missing or inactive accounts, which also stops
MCP authentication delegated to `/me`.

Standalone service routers without the web app's `auth_store` retain their
configured stateless admin-token support. Tokens alone do not create accounts
in the web app. CLI-minted tokens must identify an existing active web account.

An explicitly supplied Authorization header takes precedence over a login
cookie, including an empty header, malformed Bearer credentials or an
unsupported scheme. Invalid authorization never falls back to the cookie.
Malformed JWT headers, identity claims and numeric dates are rejected; expiry is enforced at
the expiry instant, and a future `nbf` prevents early use.

## Conversations and memory

All session creation attributes ownership to the authenticated caller. A
body-supplied `user_id` cannot create a conversation for someone else. Reading,
renaming, deleting, continuing or rating an owned conversation requires its
owner, including chat and tool streaming endpoints. Refusal returns 404 to
avoid revealing whether another user's session exists.

Anonymous conversations intentionally have `user_id = NULL`. Possession of
their session ID allows access; there is no per-browser anonymous identity.
Anonymous history lists are empty. Retrieval feedback can also refer to an
external session ID, but a matching locally owned session requires its owner.

Memory retains the existing shared `default_user` bucket for anonymous local
research. `AGENTIC_SEARCH_MEMORY_REQUIRE_AUTH=true` disables that anonymous
access through both HTTP and MCP. Web memory requests also check active account
state. Signing in selects the caller's memory bucket and document permissions;
it does not select a different search route.

## Admin and debug surfaces

The debug router is mounted only when debug panels are enabled, and all its web
routes require admin access. The enablement flag alone does not authorize
access to request traces, tool inventories or retrieval diagnostics.
`AGENTIC_SEARCH_DEV_ADMIN=true` remains an explicit development bypass for admin
guards; auth tests set it to `false`, since unsetting it lets `.env` restore it.
SCIM directory operations require SCIM tokens, SCIM token administration
requires admin, and SCIM discovery remains public.

## Adding an endpoint

Before seeding, MCP discovery or model loading, web startup runs
`check_router_auth`. Every path and method must have a recognized authentication
dependency or an inline call to a reviewed guard, or appear in
`PUBLIC_ENDPOINT_SPECS` with its reason. Public GET does not exempt POST;
duplicate registrations and mounted applications are checked individually.
Only explicitly classified static mounts are exempted as static assets.

Use an authentication dependency for new protected endpoints. Inline detection
is a source heuristic, not a control-flow proof: it recognizes reviewed
functions by module and qualified name, ignores comments, optional identity
resolution and uncalled nested definitions, and fails closed when source cannot
be inspected. It cannot prove a guard runs on every branch or its result is
used correctly. Runtime authorization and ownership tests remain required.

The allowlist records deliberate anonymous flows, UI bootstrap data, API docs,
SCIM discovery, integration-mode fixture reset and unimplemented 501 stubs.
Implementing a stub requires revisiting its classification.

Regression coverage lives in `test_route_auth_enforcement.py`,
`test_auth_completion.py`, `test_auth_malformed_credentials.py`, and the existing
session ownership, SCIM, memory, stale-token and MCP authentication tests.
