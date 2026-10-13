# Renovate: stop the repository-changed abort, and catch aborts: design

## Problem
Both Renovate runs after #697 aborted right after pushing their first branch
(`renovate/npm-minor-and-patch`). They logged
`Repository has changed during renovation - aborting`, with result
`repository-changed`, and opened no PR. The runs were green.

Cause:
- `minimumReleaseAge` makes Renovate POST a `renovate/stability-days` commit
  status for each branch.
- `RENOVATE_TOKEN` has no Commit statuses permission. The token page grants
  code, issues, pull requests and workflows only.
- Renovate 44's GitHub `setBranchStatus` turns any POST failure into
  `REPOSITORY_CHANGED` (modules/platform/github/index.js).
- The timing fits: PRs opened fine before #697 added the age rule.

The workflow's check step read only WARN and above, but Renovate logs the abort
at INFO, so nothing failed.

## Decision
- `statusCheckNames.minimumReleaseAge: null`. Renovate skips the status
  ("Status check is null ... skipping", update/branch/status-checks.js). With no
  automerge, the status is informational only, and the age rule itself still
  applies to which versions are proposed.
- The check step logs at info. It fails unless exactly one
  `Repository finished` record has `result: "done"`, in addition to the
  existing ERROR and `Error updating branch` conditions. Only WARN and above
  become annotations.
- Not used: Renovate's `exitCodeForErrors`, which is marked experimental.

## Verification
- Strict validator passes.
- Check script, extracted from the workflow:
  - exit 0 on a `done` log;
  - exit 1 on `repository-changed`, on a push failure, on an empty log, and on a
    missing file.
- A github-platform dry run reports `result: "done"`. The lookup-only local
  mode omits the result field, which is not a mode CI uses.
- End to end after merge: the next run opens the npm, python and pre-commit
  PRs and its check step passes.
