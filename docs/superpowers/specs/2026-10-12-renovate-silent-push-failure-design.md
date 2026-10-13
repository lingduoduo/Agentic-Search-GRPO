# Renovate: fail the run when branches cannot be pushed: design

## Problem
The first real run (Actions run 38335145281) used a token with Contents
read-only. Every branch push returned 403. Renovate logged one WARN
`Error updating branch` per branch and exited 0, so the workflow was green
while opening nothing. A scheduled run that silently does nothing is
indistinguishable from "no updates this week".

## Decision
- Renovate writes a JSON log (`RENOVATE_LOG_FILE=/tmp/renovate/log.ndjson`,
  level warn). The action mounts `/tmp` into its container by default and
  passes `RENOVATE_*` env through.
- A following step (`if: !cancelled()`) fails on any record with level ≥ 50
  (ERROR/FATAL), on the WARN `Error updating branch`, and on a missing log file.
  Every other WARN becomes a `::warning::` annotation. Lookup warnings are
  routine (transient registry trouble), and failing on them would make the
  weekly run flaky.

## Verification
The check script, extracted from the workflow, passes on a real local log
(one benign WARN) and fails on a synthetic record shaped like the 403 failure
and on a missing file. The end-to-end check is the next real run after merge.
