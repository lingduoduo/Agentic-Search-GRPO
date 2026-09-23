# Enforce the full JSON Schema on tool arguments — design

## Problem

Every runtime tool call — the tool agent, `/tools` REST, MCP dynamic tools, the
memory service — goes through `ToolRegistry.invoke`, which calls
`validate_arguments`. That validator checks only two things: required keys are
present, and each top-level argument's `type`. Everything else a schema declares
is ignored. Measured on `main`:

| call | result |
|---|---|
| `max_length=0` against `minimum: 1` | accepted |
| `domain="bogus"` against an `enum` | accepted |
| `queries=[1, 2]` against `items: {type: string}` | accepted |
| `{"q": "x", "limti": 5}` against `additionalProperties: false` | accepted |

The constraints OpenAPI-imported and MCP tools carry verbatim are therefore
decorative — and those are the tools that may be side-effecting, where an invalid
call is not simply retryable.

## Decision

Keep `validate_arguments(parameters, arguments) -> list[str]` and its callers
unchanged; replace its body with `jsonschema.Draft202012Validator`. Invalid
arguments are **rejected**, not clamped: the error string already reaches the model
as `invalid_arguments` (`ToolAgentLoop._call_tool`), so it can correct itself. The
in-code clamps in the public-data tools stay as defence in depth.

## Behaviour

- `parameters == {}` → no errors (schemaless tools, unchanged).
- Missing required keys keep their existing message, `Missing required argument:
  'x'`, since tests and model prompts see it.
- Other errors: `Argument 'path.to.field': <jsonschema message>`, where the path is
  the error's `absolute_path` joined with `.`; root-level errors (e.g. unexpected
  keys) read `Arguments: <message>`.
- Errors are reported in a stable order (sorted by path, then message) so the
  string the model sees is deterministic.
- **Malformed schema** (`SchemaError` from `check_schema`, or an unresolvable
  `$ref` — OpenAPI parameter schemas are copied without resolving references):
  log a warning and fall back to the current shallow check. A broken remote schema
  must not make every call to that tool fail, and it must not silently skip all
  validation either.
- Type semantics follow JSON Schema: `3.0` is a valid `integer`, `True` is not.
  The shallow check rejected `3.0`; the tools `int()` their counts, so accepting it
  is harmless.

## Dependency

`jsonschema` is installed today only transitively (via `mcp`). Declare
`jsonschema>=4.18,<5` in `requirements.txt` and `requirements-unit-test.txt`.

## Out of scope

Tightening the built-in schemas themselves (ranges, enums, patterns,
`additionalProperties: false`) — the follow-up PR. This PR makes declared
constraints binding; that one declares them.

## Testing

Unit tests in `tests/unit/test_tool_arg_validation.py` for each row of the table
above, nested paths, required-message stability, deterministic order, the
malformed-schema and unresolvable-`$ref` fallbacks, and one end-to-end
`ToolRegistry.invoke` rejection that proves the tool never executes.
