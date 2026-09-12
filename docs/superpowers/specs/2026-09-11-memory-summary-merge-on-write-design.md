# Merge-on-write for the summary save in `compress_session`

## Origin

`compress_session` (`src/internal/memory/working.py`) reads the session's
memory state, runs the summarizer, and writes the new summary back by
reconstructing the whole `SessionMemoryState` from the snapshot it read
before the LLM call. The lock lease is 120 s, and a slow completion can
outlive it. When it does, another process may have summarized the next span
and advanced `summarized_through`; the first task then overwrites that
newer state with its older summary and cursor, and the next turn
re-summarizes (and, with auto-curation on, re-curates) turns already
covered. #581 closed the same pattern for the curated-cursor write; this
closes it for the summary write, the last instance.

## Design

After the summarizer returns non-empty text, `compress_session` re-reads the
state:

- If the re-read's `summarized_through` differs from the one read before the
  LLM call, another process advanced the summary while this one ran. This
  task's text is stale: log at warning, write nothing, skip curation, return
  `False` (the existing meaning: another task owns the span).
- Otherwise write `replace(current, summary=text, summarized_through=last_id)`,
  which preserves whatever `curated_through` the re-read holds, then proceed
  to curation as today.

A blank re-read on a session that had no summary yet (`summarized_through`
was `None` before and after) is the first-summary case and writes as
before. A blank re-read on a session that did have a cursor is treated as
"changed" and skipped, the same stance #581 takes: better to leave state
alone than to write over a value the read could not see.

The healthy single-process path is unchanged in effect: the cursor never
moves during the LLM call, so the write proceeds.

## Testing

`tests/unit/memory/test_working_memory.py`:

- An LLM whose `complete` writes a newer state (`summary="newer"`,
  `summarized_through="m_newer"`) before returning: `compress_session`
  returns `False`, the newer state is intact, a curate fake was never
  called.
- An LLM whose `complete` writes only `curated_through="moved"` (cursor
  unchanged): returns `True`, summary saved, `curated_through == "moved"`.
- First summary (no prior state) with a cache whose reads fail during the
  LLM call: still writes (cursor `None` before and after).
- Mutation checks: skip the re-read (write from the snapshot) and the first
  test sees `summary == "S"`; drop the `replace` and the second sees
  `curated_through is None`.
