# Source-category eval: pilot flags

## Intent

The pre-registered source-category evaluation (#681) needs 194 live SerpAPI
calls: 64 labels × three arms plus controls. The user asked for a much smaller
run, one query per category. This adds the flags for it without changing the
labels, the arms, the metrics, or the decision rule.

## Behaviour

- `--per-category N` keeps the first N labels of each category in the frozen
  file's order. The rule is mechanical, so the subset is fixed before any result
  exists and cannot be cherry-picked. With N=1 it keeps 16 queries.
- `--no-plain` skips the plain arm. The pre-registered rule compares only
  category against suffix. A skipped arm is reported as `null` (not run) and is
  never a reason to exclude a query.
- `--dry-run` counts calls for the chosen flags. `--per-category 1 --no-plain`
  with one control needs 33 calls.
- The report gains `sample` (`per_category`, `plain_arm`, `n_labels`,
  `provider`), so a pilot is never read as the full run.

## Interpretation

Sixteen paired queries can detect only a large effect. A pilot can show that
restriction clearly helps or clearly hurts; a clear harm still triggers the
pre-registered default-off rule. A null pilot means the sample was too small to
tell, and the docs say so.

## Out of scope

New labels, new metrics, changes to the decision rule, provider changes
(`--provider serper` already exists).
