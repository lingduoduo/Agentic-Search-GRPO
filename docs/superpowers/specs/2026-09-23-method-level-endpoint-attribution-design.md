# Resolve endpoint attribution to the method, not the module

## Goal

Turn the census's 80-file "needs judgement" bucket into something decidable, and
delete what that precision reveals.

## What the previous attribution could not do

#633 shipped two modes. Both were approximations:

- **direct-only** counted endpoints a file builds itself. Most tests call
  `CCPairManager.delete(...)` rather than constructing a URL, so 67 files came out
  unclassifiable.
- **helper-aware** credited a test with every endpoint of every `common_utils`
  module it imported. Importing a manager is not calling every method on it, so
  "mixed" was inflated to 80.

Two things fix that.

**Resolve to the callable.** `endpoints_by_callable` indexes each manager method
separately — `CCPairManager.pause_cc_pair` builds
`/manage/admin/cc-pair/{}/status` and nothing else — and `callables_invoked` reads
which methods a test actually calls. Attribution follows the call, not the import.

**Follow derived URL constants.** Managers rarely inline a full path:

```python
DISCORD_BOT_API_URL = f"{API_SERVER_URL}/manage/admin/discord-bot"
...
requests.get(f"{DISCORD_BOT_API_URL}/config")
```

The old extractor only recognised f-strings referencing `API_SERVER_URL`
directly, so the second line looked like it built `/config` — or, where the
constant was the whole value, like it built nothing. `_url_prefix_constants`
records those assignments and prepends the prefix, which surfaced **31 endpoints
that were previously invisible**, including every `/build/...` and
`/manage/admin/discord-bot/...` path.

## Result

Measured before #634 merged, so the two attributions are comparable on the same
119 files:

| | helper-aware (#633) | method-level |
|---|---|---|
| Endpoints referenced | 178 | **191** |
| Every endpoint gone | 22 | **25** |
| Mixed | 80 | **66** |
| Fully served | 2 | **11** |
| Unattributable | 15 | **17** |

Fourteen files left "mixed": nine resolved to fully-served, three to wholly-dead,
two to unattributable. Nine of those were being carried as undecided when the
answer was "every endpoint it calls is still there".

## It also cleared #634

The reconciliation mattered more than the counts. Every file #634 deleted is
still wholly dead under this stricter pass — **#634 is a safe subset**, not an
over-deletion.

That was worth checking because I briefly believed the opposite. Comparing the two
wholly-dead sets, I read "in the old set but not the new" as "now mixed" and
nearly pushed a correction to #634 on that basis. The files had moved to
*unattributable*, and a direct probe settled it: the managers behind them build
from `{API_SERVER_URL}/manage/admin/discord-bot` and `{API_SERVER_URL}/build`,
and both return zero routes. The bucket label was mine, not the tool's.

## What is deleted here

The three files this precision reveals, all confirmed by prefix probe against a
real `create_web_app()`:

| File | family | routes |
|---|---|---|
| `tests/cli/test_cli_commands.py` | `/cli` | 0 |
| `tests/projects/test_projects.py` | `/projects` | 0 |
| `tests/craft/test_webapp_proxy.py` | `/build/sessions/...` | 0 |

`cli/` and `projects/` lose their only test and go; `craft/` keeps two.

Afterwards the census reports **zero** wholly-dead files: every remaining
integration test reaches at least one endpoint the application still serves.

## What is still not decidable

Sixty-six files reach both live and dead endpoints. That is now a real finding
rather than an artifact — those tests genuinely mix surviving and removed API —
and no static pass can say whether a given test's dead calls are incidental. They
need reading, one at a time.

Seventeen files remain unattributable: endpoints assembled from variables, or
helpers taking a path as an argument. Those are where to look next if this needs
to be tighter.

## Testing

Collection errors on the remaining tree are **14 before and 14 after** — none
introduced, checked against `origin/main` rather than assumed. `pytest tests/unit`
is unaffected; `testpaths` never covered this tree.

The tool's own correctness was verified by the case that motivated it: the
`/manage/admin/discord-bot/...` and `/build/...` families now appear in the census
where they were previously absent entirely.
