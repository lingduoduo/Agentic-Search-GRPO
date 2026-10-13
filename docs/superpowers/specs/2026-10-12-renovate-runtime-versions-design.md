# Renovate: gate workflow runtime versions: design

## Problem
Renovate's github-actions manager tracks `python-version`/`node-version` inputs
of setup-python/setup-node as deps `python`/`node` (packageName
`actions/python-versions`/`actions/node-versions`). The runtime-approval rule
only matched the dockerfile/docker-compose managers, so 3.12 → 3.15 (a *minor*
update) slipped into the routine github-actions group in #694 and failed the
Intent Routing Gate on Python 3.15.

## Decision
- The runtime rule matches managers dockerfile, docker-compose **and**
  github-actions, by `matchDepNames: [python, node]` (not package names, which
  differ per manager).
- It sets its own `groupName: "runtime versions"`. An approval rule does not
  change grouping, so without this the gated bump would stay inside the
  github-actions group and hold back that whole PR.

## Verification
- Strict validator passes.
- Local `--dry-run=full`: the four workflow `python-version: 3.12 → 3.15`
  updates move from `renovate/github-actions` to `renovate/runtime-versions`;
  node 22 → 24 goes to `renovate/major-runtime-versions`.
- The local platform does not evaluate approval. After merge, the next real run
  must show "runtime versions" as awaiting approval on the Dependency Dashboard
  (#693), and #694 must drop its `python-version` changes.
