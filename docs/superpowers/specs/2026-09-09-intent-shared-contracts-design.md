# Shared intent vocabulary and scoring defaults

## Context

PR #569 consolidated web intent recognition and is merged with all CI checks
passing. Its review left a private strategy-only wrapper with only test callers.
A follow-up inspection found that RouteStrategy and offline INTENT_LABELS still
define the same vocabulary separately, and scoring defaults are repeated across
AppSettings, environment parsing, the default configuration map and offline code.
The example CLI calls IntentIndex.decide without intent_top_k, silently ignoring
AGENTIC_SEARCH_INTENT_TOP_K even though the web adapter honors that setting.

## Design

Add `src/shared_configs/intent.py`, importing only the standard library. It owns
RouteStrategy, INTENT_LABELS derived in enum order, and DEFAULT_TOP_K=8,
DEFAULT_MIN_ROUTE_MARGIN=0.010, DEFAULT_MIN_MODULE_SCORE=0.8215.

Web decision types re-export the shared enum. Offline model.py imports the
shared labels and aliases DEFAULT_TOP_K as TOP_K so existing imports continue
to work. AppSettings fields, environment-parser fallbacks, DEFAULT_CONFIG, and
the evaluation module's default module score consume the shared constants.
No offline model or web dependency is introduced into the configuration layer.

The CLI passes settings.intent_top_k to the existing index.decide method,
alongside its existing margin and module-score arguments. CLI policy remains
separate from web orchestration: encoder mismatch still raises in the CLI;
the web adapter still defers on unavailable/incompatible models. No web
cascade, shadow mode, provider semantics or tool policy is added to the CLI.

Remove rules._rule_based_route, whose only callers are tests. Preserve meaningful
last-resort heuristic coverage through recognize_intent; retain the private
_rule_based_route_or_none used by the real cascade. Update stale top-3 comments
and evaluation-report prose that incorrectly says serving always uses k=3.

## Constraints

- Python >=3.10; no new dependencies.
- Route values/order remain chat, search, tool.
- Defaults remain top_k=8, min_route_margin=0.010, min_module_score=0.8215.
- Existing web.intent.RouteStrategy and model.INTENT_LABELS/TOP_K imports work.
- Index format, canonical examples, encoder, scoring, evaluation selection grids,
  capture fields, external APIs and authorization remain unchanged.
- This continues the previously authorized spec/plan/code/PR workflow. Create a
  new branch from merged main; do not reopen the merged PR or merge this work.

## Validation

Use a real toy index with one exact search neighbor plus three distant search
neighbors, stable chat similarity 0.8, and distant tool neighbors. For the same
query embedding, k=1 must choose search with score 1.0 and k=4 must choose chat
with score 0.8. Exercise the real CLI loader and web similarity adapter under the
same environment settings; the pre-fix CLI fails k=1 by silently using default k.
Only embedding inference is stubbed; loading, settings, scoring and decisions are
real. Default CLI and web settings must also agree on this fixture.

Verify the shared contracts import in a fresh process with web/model/numpy/torch
imports blocked, and exercise public route parsing/serialization. Existing
configuration, taxonomy, scoring, routing and web tests validate compatibility.
Run focused checks, repository lint/format, then the full backend suite. Obtain
independent review before publishing a follow-up PR; report optional skips.
