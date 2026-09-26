# Plan: compose --env-file in docs

1. README "Run with Docker" and deploy.md "Deploy a tag" use `--env-file .env` → diff
2. `docker compose --env-file <.env.example + key> config` → key reaches both services
