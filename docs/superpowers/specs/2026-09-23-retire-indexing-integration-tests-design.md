# The repaired indexing tests cannot pass; retire them

## Goal

Answer the question left open in #622 — *verify the mechanically repaired
indexing tests once a live stack exists* — and act on the answer.

## The answer

They cannot be verified, because the subsystem they exercise no longer exists.

#629 gave CI a live Postgres/Redis stack, which was the stated precondition. It
is not the blocker. Both tests drive `CCPairManager` and `IndexAttemptManager`,
which call:

```
/manage/admin/connector/{id}/credential/{id}
/manage/admin/cc-pair/{id}/status
/manage/admin/cc-pair/{id}
/manage/admin/connector/indexing-status
/manage/admin/deletion-attempt
```

Asked directly, the application serves **none** of them. Its only route
containing "connector" is `/oauth/connector/{connector}/callback`, which is
unrelated. Probed for prefixes rather than exact matches, `/manage/admin/cc-pair`
returns zero routes.

This is consistent with the repo's own history: CLAUDE.md records that the async
worker fleet and the `servers/indexing` pipeline were Onyx heritage and have been
removed. The tests outlived the feature.

So no stack, however complete, makes them pass. The mechanical repair in #622 —
substituting `{"has_more": False}` for a deleted mock's payload — made them
*parse*. It could not make them meaningful, and I said at the time it was
unverified rather than working.

## What is removed

The whole of `tests/integration/tests/indexing/`:

- `test_initial_permission_sync.py`, `test_repaired_error_state.py` — the two
  mechanically repaired in #622
- `file_connector/test_file_connector_zip_metadata.py` — same removed API via
  `CCPairManager`, plus its four fixture files
- `conftest.py` — a mock-connector-server client fixture with nothing left to
  serve

`test_checkpointing.py` and `test_polling.py` went in #622, the first as dead
surface and the second because it referenced two further undefined names that a
blanket `# noqa: F821` had hidden.

## The wider finding, not acted on here

The same probe over the whole integration suite: of 180 distinct endpoints
referenced, the application serves **26**. Sixty-nine of 122 test files reference
at least one endpoint that no longer exists — whole families of them, including
`/admin/api-key`, `/admin/code-interpreter`, `/admin/kg/config` and
`/admin/image-generation/config`, none of which match any route even by prefix.

That is a decision about a 122-file suite, not a cleanup to fold into this
change. It is also worth treating carefully: my endpoint extraction is a regex
over the test sources and produced at least one truncated capture, so the exact
count is approximate even though the calibrated spot-checks were unambiguous.
What is certain is the direction — most of this suite describes an application
this repo no longer is.

## Testing

A deletion of never-executed tests, so there is nothing to make pass. The
supporting evidence is the route probe above, run against a real `create_web_app()`
rather than grepped: asking the app was the only way to settle it, exactly as in
#624 where grepping for `@app.` missed a route registered by a shared factory.

`pytest tests/unit` is unaffected — `pyproject.toml` sets
`testpaths = ["tests/unit", "tests/regression"]`, so the integration tree has
never been in the default run.
