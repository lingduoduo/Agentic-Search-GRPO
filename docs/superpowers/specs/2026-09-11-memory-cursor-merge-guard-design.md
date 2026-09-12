# Guard the curated-cursor merge against a blank re-read

## Origin

PR #579 made `_curate_after_summary` (`src/internal/memory/working.py`)
re-read the session's memory state and merge `curated_through` onto it, so a
summary another process advanced during a long curation is preserved. The
whole-branch re-review parked one residual: `load_state` turns a cache read
failure, or bad JSON, into a blank `SessionMemoryState()`. On such a blip the
merge writes `summary=""`, `summarized_through=None`, and the new cursor,
wiping the summary this very task saved a moment earlier. The outcome is
bounded (the next overflow re-summarizes the whole dropped prefix and may
duplicate memories) but it is a regression the merge introduced, and it is a
one-line guard.

## Design

In `_curate_after_summary`, after `current = load_state(cache, session_id)`:
if `current.summarized_through is None and not current.summary`, the re-read
did not return the state this task just wrote (read failure, malformed
value, or the key expired between the two writes). Log at warning and return
without writing. The summary write already landed; leaving `curated_through`
unmoved is the same accepted outcome as a failed cursor write, and the span
is at worst curated a second time later.

No other behavior changes. The healthy path (state present) merges exactly as
before.

## Testing

`tests/unit/memory/test_working_memory.py`: a cache subclass of
`InMemoryCache` whose `get` raises once a flag is set; the curate fake sets
the flag before returning `True`, so the summary write succeeds and the
cursor re-read fails. After `compress_session` returns `True`, with reads
restored, the state shows `summary == "S"`, `summarized_through ==
pending[-1].id`, `curated_through is None`. Mutation check: remove the guard
and the test sees `summary == ""`.
