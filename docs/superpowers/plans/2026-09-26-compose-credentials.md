# Plan: compose credentials

1. Add a contract test: each operator key is `${KEY...}` on both app services → red, 6 fail
2. Interpolate the keys in `x-app-env` and document `--env-file` in the header → 17 pass
3. `docker compose config`, old file vs new, using an env-file with a key and an unrelated host URL → the old file drops the key; the new one passes the key and nothing else
