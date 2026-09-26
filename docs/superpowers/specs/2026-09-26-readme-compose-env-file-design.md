# README/deploy: compose credentials via --env-file

Follow-up to #669 and #670. The README pointed at `docker-compose.override.yml` because, before #670, a `.env` could not reach the containers. #670 made the compose file read `GEN_AI_*` and `SERP_API_KEY` by interpolation, so the documented command becomes `docker compose --env-file .env ...`. deploy.md's deploy command also passed no credentials, and gets the same change.
