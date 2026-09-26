# README Diagrams Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the stale architecture image with three accurate,
renderable Mermaid diagrams in the README.

**Spec:** `docs/superpowers/specs/2026-09-25-readme-diagrams-design.md`

## Global Constraints

- Every node and transition traces to current code.
- The diagrams must render with mermaid-cli before they are committed.

---

- [x] Read `ToolAgentLoop.run`, `_call_tool`, `_escalate`,
  `_request_approval`, `RecoveryPolicy.decide`, the SSE emitters and the
  timeouts.
- [x] Draft the three `.mmd` files. Render them with mermaid-cli 11 and
  inspect the images.
- [x] Rework the state diagram: statuses go inside the state boxes, because
  the notes overlapped the transition labels.
- [x] Insert the diagrams into the README Architecture section. Remove the
  stale PNG and HTML. Link from `docs/tool-engine.md`.
- [x] Run the full unit suite, then ruff, then `git diff --check`. Open the
  PR.
