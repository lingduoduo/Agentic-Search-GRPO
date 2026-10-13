# Renovate: gate workflow runtime versions: plan

**Spec:** docs/superpowers/specs/2026-10-12-renovate-runtime-versions-design.md

1. Extend the runtime rule in `renovate.json` (github-actions manager,
   `matchDepNames`, `groupName: "runtime versions"`) → verify: strict validator.
2. Local `--dry-run=full` → verify: no `python-version` update left in
   `renovate/github-actions`.
3. Update `docs/development/dependency-updates.md` → verify: wording matches the rule.
4. After merge, trigger the Renovate workflow → verify: dashboard shows the
   runtime group pending approval; #694 no longer changes `python-version`.
