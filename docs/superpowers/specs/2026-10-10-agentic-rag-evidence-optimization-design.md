# Agentic RAG evidence optimization

## Intent

Improve retrieval coverage across conversational turns and iterative search rounds while keeping answers grounded in retrieved evidence with stable citations. The user requested multi-turn search, query enhancement, citations, grounded synthesis, Superpowers specs and plans, implementation, and PRs.

## Existing architecture

The web entry point resolves follow-ups into `retrieval_query`. `AgenticRAGLoop.run` enhances that query, batches retrieval, accumulates documents, assigns stable D identifiers, judges sufficiency, generates gap queries, and calls guarded `generate_answer`. History and the original question already reach synthesis. Grounded generation is enabled by default and abstains without evidence.

## Selected approach

Make targeted changes to the existing retrieval loop and query bundle. A replacement controller would duplicate existing behavior and increase compatibility risk. Prompt-only changes would not fix missing original queries or query bookkeeping.

## Requirements

1. `QueryBundle.all_queries()` includes the original query first, followed by decomposition, HyDE, and step-back variants. Ignore whitespace-only entries; deduplicate by lowercase, whitespace-collapsed text, preserving the first raw query. Hypothetical text remains a retrieval input only.
2. Keep the explicit `retrieval_query` authoritative. Regression coverage must show that enhancement and evidence checks use it while synthesis receives the original question and history. Do not add a second conversational resolver to the loop.
3. Replace prefix truncation in evidence checks with a shared bounded preview. Add `evidence_preview_chars=3000` to `AgenticRAGConfig`, validated as positive. Allocate the budget across all document entries, retaining identifiers and document order. Long early documents must not exclude later evidence when the budget can fit all labels and at least one content character per document. Total preview length must not exceed the budget. If labels cannot all fit, deterministically include entries in order within the bound. Do not describe a preview as complete evidence.
4. Sufficiency and gap prompts treat retrieved text as evidence, not instructions. Use the same preview helper for both calls. Preserve existing timeouts and failure behavior.
5. Follow-up candidates are cleaned, blank-filtered, and deduplicated before applying the per-round cap. Record only selected queries as searched; overflow candidates remain eligible in later rounds. Treat standalone `none` in the QUERIES section as no query. Zero follow-up budget ends the loop before another retrieval round.
6. Preserve stable citation identifiers, filters, callbacks, no-LLM operation, guarded synthesis, and existing public result fields. Test accumulated evidence from multiple rounds and verified citations through real answer generation, including empty-evidence abstention.

## Constraints

- Python >=3.10; no new dependencies.
- Preserve explicit retrieval-query precedence and synthesis history.
- Preserve sufficiency timeout and fail-open behavior.
- Preserve guarded generation and stable document identifiers.
- No changes to public HTTP interfaces or retrieval backends.

## Verification

Use deterministic mocked retrieval and LLM responses. Cover normalized duplicates, blank variants, late evidence, tiny preview budgets, explicit follow-up queries, capped overflow, zero follow-up budget, repeated documents, stable citations, and empty-evidence abstention. Run query-enhancer, agentic-RAG, grounded-generation, and grounding unit suites. Live provider benchmarks are outside this change; unit tests establish behavior, not measured answer-quality improvements.

## Delivery

One focused implementation PR contains the spec, plan, code, and regression tests. Separate PRs are justified only if implementation reveals independently shippable changes. Review this design and the accompanying plan before product-code changes.
