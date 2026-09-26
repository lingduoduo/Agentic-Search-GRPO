# Guides sync: behavioural delta of #560–#668

**Goal.** Correct or add every claim in `docs/*.md` that PRs merged since the
last guides sync (#559) changed in a way a reader can observe. It is the
companion to the README sync (#669). No reorganisation and no style pass.

**Method.** Four parallel audits over disjoint file sets:

1. api-reference, request-routing, search-engine
2. architecture, chat-engine, frontend
3. tool-engine, mcp, cli, configuration, configuration/timeouts
4. retrieval, ingestion, training-and-evaluation, testing, authentication, workload-identity

Each claim was verified against code on 4f6894da and then spot-checked
independently.

**Out of scope, recorded in the PR:** drift older than #560 that the audits
noticed, and drift in non-doc files (CLAUDE.md, code docstrings,
tests/integration/README.md).
