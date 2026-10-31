# Node-description judgeability audit

The current hierarchy builder creates valid artifacts, but neither artifact
validity nor nonempty descriptions establish that a deployed decision model can
identify the right branch from them. This audit investigates that boundary.
Recommendations 1, 2 and 4 are now implemented (see
[description judgeability](corpus-hierarchy.md#description-judgeability));
sibling hard-negative validation (3) remains open, and no real-model recall
delta has been measured yet.

## Paper evidence and its limits

The primary [DecisionRAG paper](https://arxiv.org/html/2610.35868v1) makes node
descriptions part of the routing input (§3.1), explicitly notes irreversible
pruning (§3.3), and lists uninformative descriptions as a limitation (§7).
Appendix B.5 supplies up to 300 page titles, ordered by typicality, to Qwen3-8B,
and asks the description to cover less-typical subjects too. This is materially
more coverage-oriented than the current first-12-by-ID implementation.

The inspected paper contains budget/hierarchy experiments (§6.6), but I did not
find a controlled experiment that holds membership, budgets and judge fixed
while varying only descriptions. Architectural dependence and a plausible
causal failure mechanism should not be presented as a measured description-only
effect on the paper's NQ benchmark.

## Concrete implementation findings

| Finding | Evidence | Consequence |
|---|---|---|
| Descriptions are accepted for being nonblank | `hierarchy.py:445–448`; persisted group strings at `153–155` need only be truthy | Generic and duplicate descriptions pass; persisted whitespace-only descriptions also pass. No semantic quality gate. |
| LLM samples are ID prefixes | `hierarchy_models.py:108–115`, default sample size 12; context pressure pops the last sample | No guarantee of major/minority topic or language coverage; changing IDs can change the evidence available to the generator. |
| Extractive descriptions are smaller ID prefixes | `hierarchy.py:452–455`: first eight titles/first sentences, capped at 1600 characters | A tail-topic addition can invalidate the cache and trigger regeneration without changing the actual routing description. This is coverage loss, not a failed cache refresh. |
| Uncertain descriptions still face ordinary pruning | `decision_retriever.py:282–287` exposes group label plus description; `309–313` applies the budget | Pruned descendants never receive a document-level relevance judgment. Default zero cutoff does not prevent budget-based losses. |
| Later levels have the same dependency | `decision_retriever.py:245–254`: sections use title+heading; documents use title or ID | A good group description cannot rescue a gold document pruned under a generic title or an uninformative Body heading. |
| ACL-safe descriptors have a separate coverage limit | `decision_retriever.py:335–349`: visible titles joined and clipped to 1600 characters | The first long visible title can hide later topics even when the filter removes no documents. Reusing mixed-ACL summaries would be unsafe; improve the visible-only representation instead. |
| Diagnostics currently measure structure | `hierarchy.py:503–533` | Size, imbalance and fanout reports are useful, but do not measure coverage, sibling distinguishability or routing recall. |

Description source (`local_llm`, `extractive`) is provenance, not a quality score.
A longer description can also add noise or consume context; text length alone
is not a sufficient acceptance criterion. Similar sibling summaries are not
necessarily wrong if their actual evidence overlaps.

## Reproduced mechanics

These probes were run once with a local script, which is not kept in the
repository, against the code as it stood before #708. It used a transparent
deterministic test judge and prompt capture and loaded no model weights. #708
since changed several of these behaviors: whitespace-only descriptions are
rejected, description samples are coverage-selected rather than ID prefixes, and
extractive and ACL-visible text use per-item quotas. The unit tests for those
changes now cover the same mechanics.

| Probe | Observed result |
|---|---|
| Two sibling groups both described as Various documents | Artifact validation succeeds. |
| Fixed memberships, budget and test judge; change only gold group's description | Generic description retrieves the other document and gold document is never judged; specific description retrieves gold. |
| Add a ninth, later-ID minority-topic document | Membership grows, description is regenerated, but extractive description is unchanged and omits the topic. |
| Actual local-description prompt with 13 documents | The first 12 ID-sorted documents are sampled; the last rare-topic document is absent. No generator inference is run. |
| Two public documents, first title longer than 1600 characters | Unfiltered stored summary mentions the rare topic; ACL-visible descriptor loses it. Both documents remain retrieved in this probe, so this establishes representation loss, not measured ACL recall loss. |
| Persist a whitespace-only description | Validation succeeds. |

These probes demonstrate implementation behavior and a possible irreversible
routing failure. They do not estimate production LLM accuracy, prove that all
generic descriptions fail, or quantify a real-model recall delta. Labels and
model background knowledge can sometimes compensate. Independent review reran
the probes and confirmed this interpretation.

## Recommended investigation and implementation order

1. **Measure routing utility before grading prose.** Keep corpus membership,
   query set, judge and budgets fixed. Compare current descriptions with
   label-only, extractive, diversity-sampled and paper-inspired alternatives.
   Use held-out development queries with gold documents/passages. Generated
   queries can supplement tests, but require independent/human checks and must
   not be the only evidence; matching a generator's own wording is too easy.
2. **Improve coverage selection.** Sample typical documents plus minority,
   boundary and language-diverse examples rather than ID prefixes. Persist
   sampled document IDs, generation version, context/description limits and
   topic-coverage diagnostics. Large/mixed groups may need more than a one-line
   label: bounded routing cards can include scope, entities/tasks, representative
   children and minority topics, grounded in explicit source IDs.
3. **Validate against sibling decisions.** Use positive queries grounded in
   descendants and hard negatives from neighboring branches. Measure coverage
   of supported subtopics, unsupported claims, sibling confusion, and stability
   under paraphrases and irrelevant ID permutations. Do not use the same model
   as both sole author and sole quality judge.
4. **Add bounded recovery for weak routing inputs.** Flag missing/generic
   representations, and evaluate visible child previews, an exploration reserve
   or a bounded deferred-branch queue before irreversible exclusion. Low margins
   alone are not calibrated uncertainty, especially across separate candidate
   batches. Compare recovery at a fixed inference/token budget, not unlimited
   expansion.

Report ancestor survival at each layer, complete gold-path survival, document
and section survival, and final retrieval recall separately. For a fixed gold
passage under one tree path, final retrieval requires ancestor survival;
subsequent ranking cannot recover a path already removed. Separate descriptions
from document-title and section-heading failure when attributing losses.

Evaluate multiple budgets and report paired differences with uncertainty.
Stratify by group size/heterogeneity, rare topics, language, empty/generic titles,
source format, and ACL scope. In particular, test both filters that leave the
visible set unchanged and realistic reduced visible sets. A useful description
is one that preserves relevant paths and distinguishes competing branches under
the deployed judge and budget, rather than one that merely reads fluently.
