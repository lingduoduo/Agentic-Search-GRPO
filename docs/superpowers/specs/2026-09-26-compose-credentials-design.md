# Compose credentials from the environment

**Bug.** `docker/docker-compose.yml` hard-coded `GEN_AI_API_KEY: ""` in the
shared `x-app-env` block and declared no `env_file`, even though its comment
said "override with real keys in .env". A literal in `environment:` can't be
overridden by the shell or by `--env-file`, so the containers always ran with
an empty LLM key. `SERP_API_KEY` wasn't passed at all.

**Fix.** Interpolate each operator-supplied key from its own name, keeping the
old values as defaults: `GEN_AI_MODEL_PROVIDER`, `GEN_AI_MODEL_VERSION`,
`GEN_AI_API_KEY`, `GEN_AI_API_BASE`, `GEN_AI_MAX_INPUT_TOKENS`, `SERP_API_KEY`.
The usage is `docker compose --env-file .env -f docker/docker-compose.yml up`.

**Rejected alternative:** `env_file: ../.env`. It would copy every host-oriented
value in `.env` (such as `localhost:8002` rerank URLs) into the containers,
where those hosts do not resolve. Interpolation passes only the keys it names.

An empty value falls back to the code default (`get_env_str` treats "" as unset),
so running without credentials behaves exactly as before.
