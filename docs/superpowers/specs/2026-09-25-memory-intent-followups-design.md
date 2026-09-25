# Memory and intent follow-ups: design

Follow-ups deferred in the #656 (token-budgeted memory) and #657 (routing off
the event loop) reviews. Approved by the user.

## 1. `InMemoryCache` honours expiry

**Problem.** `InMemoryCache.set(key, value, ex=...)`
(`src/internal/cache/interface.py`) ignores `ex`, and `expire()` is a no-op.
Session-memory state (`session_memory:{id}`, set with a 30-day TTL) is
therefore never evicted in-process. Since #656 turned summarization on by
default for `/api/agent`, every overflowing session keeps its summary for the
life of the process.

**Decision.** `InMemoryCache` keeps an optional expiry per key.

- **`set(..., ex=...)`.** It records `now + ex` from an injectable monotonic
  clock, `time.monotonic` by default. `ex=None` means no expiry.
- **`get`, `exists` and `ttl`.** They treat an expired key as missing and
  delete it. `ttl` returns the remaining whole seconds, and keeps the existing
  `TTL_NO_EXPIRY` and `TTL_KEY_NOT_FOUND` sentinels.
- **`expire(key, seconds)`.** It sets the expiry when the key exists.
- **`delete`.** It clears both the value and the expiry.
- **Other operations** (list and lock helpers) are unchanged. The behaviour
  matches Redis semantics for these methods. Anything else that already relies
  on `ex` gets correct behaviour.

## 2. Deleting a chat session deletes its working-memory state

**Problem.** Nothing removes `session_memory:{id}` when a chat session is
deleted, so a deleted conversation's LLM-written summary survives.

**Decision.**
- **The helper.** `src/internal/memory/working.py` gains `forget_session(cache,
  session_id) -> None`. It deletes the state key and never raises; a failure
  is logged at WARNING.
- **Where it runs.** The session-delete route in `chat_backend.py` calls it,
  after `store.delete_chat_session` succeeds, using `get_cache_backend()`.
- **Other routes.** Any other route that deletes chat sessions calls it too.
  The implementer greps for them, for example admin or user-deletion paths,
  and records the list in the plan.

## 3. Intent index and encoder load once under concurrency

**Problem.** Since #657, `recognize_intent` runs in worker threads. The cold
loads in `src/internal/servers/web/intent/similarity.py` (`_INTENT_INDEXES`)
and `src/model/pre_training/intents/model.py` (`_MODEL_CACHE`) are unlocked
check-then-set caches. Two concurrent cold requests can each load the index
and the `SentenceTransformer`. If one of them hits a transient failure, that
failure is cached permanently, which disables similarity routing until
restart.

**Decision.**
- **A lock per cache.** A module-level `threading.Lock` guards each cache,
  with double-checked locking: a fast path without the lock on a hit, then
  load under the lock after checking again.
- **One load.** Only one thread loads, and the others wait and reuse the
  result.
- **Failures.** Failure-caching semantics are unchanged. One load now decides,
  instead of a race.

## Testing

- **`InMemoryCache`** (fake clock):
  - get before and after expiry;
  - `exists` and `ttl` after expiry;
  - `expire` on an existing and a missing key;
  - `ex=None` never expires;
  - `delete` clears the expiry.
- **`forget_session`:**
  - removes the state;
  - a raising cache is swallowed and logged.
- **The delete route:** deleting a session with stored memory state removes
  the state. Deleting a session the caller does not own leaves the state, and
  the existing auth tests still pass.
- **Intent caches:** two threads cold-loading the same path at once call the
  loader exactly once. Use a loader that blocks on an event, and count its
  calls. Test each cache.
- **Mutation checks:**
  - Ignore `ex` again and watch the expiry tests go red.
  - Drop the `forget_session` call and watch the route test go red.
  - Remove each lock and watch its concurrency test go red.
