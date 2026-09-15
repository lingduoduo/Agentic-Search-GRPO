# Agent tool menu overlap

## Problem

After PR #584 seeded the domain-search features, `tool_knowledge_base()` offered
the tool agent 15 tools. Four of them were a second path to tools already on the
menu:

- Every tag in `CAPABILITY_ROUTES` routes to one of the nine public-data tools
  seeded directly alongside it, so `search_domain` and `batch_search` reach
  nothing the agent could not already reach.
- `batch_search` runs concurrent searches, which `web_search` already does by
  accepting multiple queries.
- `get_sub_domains` exists only to discover tags for the two above.

PR #479 hit this exact shape with `search`, `search_routing_tool`, and
`rag_routing_tool`: the tool agent answered from parametric memory instead of
calling tools, and its commit message records that a system prompt alone was
tested and did not fix selection while the duplicates were present. Withholding
the duplicates did.

The evidence here is that precedent plus the structural redundancy. No
measurement of the 1.5B mis-selecting on the 15-tool set was taken.

## Decision

Withhold `search_domain`, `get_sub_domains`, and `batch_search` from the agent
menu by adding them to the existing `NOT_AGENT_CALLABLE` frozenset in
`knowledge_base.py`. The agent menu goes from 15 to 12 tools (11 seeded plus the
request-bound corpus search that `tool_agent_runner` adds).

`extract_page` stays agent-callable: nothing else seeded there fetches a URL, so
it adds reach rather than a second path to it.

Withholding rather than deleting is deliberate. The three tools stay registered
and invocable through `/admin/tools` (always mounted), `/api/debug/tools` (when
`AGENTIC_SEARCH_DEBUG_PANELS` is set), and MCP, which defines its own wrappers
in its own process and never reads this registry. Nothing PR #584 built is
removed; it is only kept off a small model's menu.

## Alternatives rejected

- **Facade wins** — withhold `web_search` and the nine direct tools instead,
  leaving roughly five. Smallest menu, but it forces two-step discovery
  (`get_sub_domains`, then `search_domain` with a tag) where a 1.5B is likelier
  to fail than when calling `get_weather` directly.
- **Delete the redundancy** — drop `CAPABILITY_ROUTES` and collapse
  `search_domain`/`batch_search` into `web_search`. One honest surface with no
  dead code, but it permanently reverts most of PR #584 and breaks the MCP
  wrappers that depend on the same service.

## Reversal note

`test_seeded_features_are_available_to_agents` asserted the opposite: PR #584
put these on the agent menu on purpose. This spec reverses that decision rather
than correcting an oversight, and the test is replaced accordingly.

## Invariant

The durable guard is that no `CAPABILITY_ROUTES` target is reachable two ways
from the agent menu: every target stays directly callable, and no facade over
those targets is offered alongside them. That assertion, not the name list,
is what catches a future re-introduction.
