# CLAUDE.md stale flags

Three claims in `.claude/CLAUDE.md` went stale, found by the #560-#668 docs sync (#671):

- `--vllm_url`: argparse rejects it; the flag is `--server_url` (`examples/agentic_search/parser.py:65`).
- Browser search is "~5-10s/query" in one place and "~30-50s" in another; the measured figure is ~48s.
- `tests/integration/` is described as only needing Postgres/Redis, but most of it targets removed endpoints (see `examples/audit_integration_endpoints.py`).
