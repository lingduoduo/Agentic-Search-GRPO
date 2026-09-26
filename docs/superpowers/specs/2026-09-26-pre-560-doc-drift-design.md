# Fix pre-#560 doc drift

This covers the drift the #560-#668 sync (#671) noticed but left out of scope as older than its range, plus two non-guide text files. Each item was verified against code on fc203c60:

- `done` event field lists (`_terminal_events`)
- the retired `search_routing_tool` (the corpus tool is now `search`), and a false RAG-Fusion claim: `search` is corpus-only
- frontend admin panels (no Connectors, `ToolAdminPanel`) and the intent class being applied in `AssistPage.tsx`
- architecture tree: `search/`, `servers/_auth.py`
- Google listed as an auto fall-through source, when it is explicit `source_provider` only
- cli.md: the endpoint path is `/chat/...`, and the parser (NDJSON) does not match the SSE stream
- garbled merge text from #532, the Bamboogle port note, testing.md commands and the file table, browser timing, and the MCP env vars
- tests/integration/README.md (deleted mock stack) and the measure_multi_turn_continuity docstring
