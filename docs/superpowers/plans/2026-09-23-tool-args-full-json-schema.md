# Enforce the full JSON Schema on tool arguments — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make every constraint a tool schema declares binding at `ToolRegistry.invoke`, not just required keys and top-level types.

**Architecture:** `validate_arguments` keeps its signature and callers; its body becomes a `jsonschema.Draft202012Validator` pass with model-readable messages. A malformed schema falls back to the existing shallow check (renamed `_shallow_errors`) with a warning.

**Tech Stack:** Python 3.10+, `jsonschema>=4.18,<5`, pytest.

**Spec:** `docs/superpowers/specs/2026-09-23-tool-args-full-json-schema-design.md`

## Global Constraints

- `validate_arguments(parameters: dict, arguments: dict) -> list[str]` — signature unchanged.
- Missing-required message stays exactly `Missing required argument: 'x'`.
- `parameters == {}` returns `[]`.
- Reject, never clamp.
- `jsonschema>=4.18,<5` in `requirements.txt` and `requirements-unit-test.txt`.

## Review Focus

- A schema with no root `"type": "object"` (several built-ins omit it) must still enforce `properties`/`required` — jsonschema applies them to dict instances regardless; pinned in Task 1 by `test_schema_without_root_type_still_enforced`.
- An OpenAPI parameter schema holding an unresolvable `$ref` must not raise out of `invoke` — pinned by `test_unresolvable_ref_falls_back`.
- A malformed remote (MCP) schema, e.g. `{"type": 5}`, must not raise and must still check required keys — pinned by `test_malformed_schema_falls_back_to_shallow_check`.
- Multiple errors must come back in a deterministic order, so the model sees a stable string — pinned by `test_errors_are_sorted`.
- `3.0` for an `integer` is accepted (JSON Schema semantics); `True` is rejected — pinned by `test_integer_semantics`.

---

### Task 1: Full-schema validation in `validate_arguments`

**Files:**
- Modify: `src/internal/tools/validation.py`
- Modify: `requirements.txt`, `requirements-unit-test.txt`
- Test: `tests/unit/test_tool_arg_validation.py`

**Interfaces:**
- Consumes: nothing new.
- Produces: `validate_arguments(parameters, arguments) -> list[str]` (same contract, stricter), `check_json_type` unchanged (tests import it).

- [ ] **Step 1: Write the failing tests** — append to `tests/unit/test_tool_arg_validation.py`:

```python
_BOUNDED = {
    "type": "object",
    "properties": {
        "max_length": {"type": "integer", "minimum": 1, "maximum": 50000},
        "domain": {"type": "string", "enum": ["general", "finance"]},
        "queries": {"type": "array", "items": {"type": "string"}},
    },
    "additionalProperties": False,
}


def test_range_is_enforced():
    assert validate_arguments(_BOUNDED, {"max_length": 0}) == [
        "Argument 'max_length': 0 is less than the minimum of 1"
    ]


def test_enum_is_enforced():
    (error,) = validate_arguments(_BOUNDED, {"domain": "bogus"})
    assert error.startswith("Argument 'domain': 'bogus' is not one of")


def test_array_items_are_enforced_with_nested_path():
    (error,) = validate_arguments(_BOUNDED, {"queries": ["ok", 2]})
    assert error == "Argument 'queries.1': 2 is not of type 'string'"


def test_unknown_key_is_rejected_when_schema_is_closed():
    (error,) = validate_arguments(_BOUNDED, {"limti": 5})
    assert error.startswith("Arguments: Additional properties are not allowed")
    assert "'limti'" in error


def test_errors_are_sorted():
    errors = validate_arguments(_BOUNDED, {"max_length": 0, "domain": "bogus"})
    assert [e.split(":")[0] for e in errors] == [
        "Argument 'domain'",
        "Argument 'max_length'",
    ]


def test_nested_required_uses_the_same_message():
    schema = {
        "type": "object",
        "properties": {
            "item": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            }
        },
    }
    assert validate_arguments(schema, {"item": {}}) == [
        "Missing required argument: 'item.query'"
    ]


def test_schema_without_root_type_still_enforced():
    params = {"required": ["a"], "properties": {"a": {"type": "integer", "minimum": 1}}}
    assert validate_arguments(params, {}) == ["Missing required argument: 'a'"]
    assert validate_arguments(params, {"a": 0}) != []


def test_integer_semantics():
    assert validate_arguments(_INT_SCHEMA, {"value": 3.0}) == []
    assert validate_arguments(_INT_SCHEMA, {"value": True}) != []


def test_malformed_schema_falls_back_to_shallow_check(caplog):
    params = {"type": 5, "properties": {"a": {"type": "string"}}, "required": ["a"]}
    assert validate_arguments(params, {}) == ["Missing required argument: 'a'"]
    assert "falling back" in caplog.text


def test_unresolvable_ref_falls_back():
    params = {
        "type": "object",
        "properties": {"a": {"$ref": "#/components/schemas/Missing"}},
        "required": ["a"],
    }
    assert validate_arguments(params, {"a": 1}) == []
    assert validate_arguments(params, {}) == ["Missing required argument: 'a'"]


@pytest.mark.asyncio
async def test_registry_invoke_rejects_out_of_range_without_executing():
    from src.internal.tools.registry import ToolRegistry

    executions = []

    @FunctionTool.from_fn(effect=ToolEffect.READ_ONLY, parameters=_BOUNDED)
    def bounded(max_length: int = 10):
        executions.append(max_length)
        return max_length

    registry = ToolRegistry()
    registry.register(bounded)
    _, _, errors = await registry.invoke("bounded", {"max_length": 0})
    assert errors and executions == []
```

- [ ] **Step 2: Run to verify they fail**

Run: `pytest tests/unit/test_tool_arg_validation.py -q`
Expected: the range/enum/items/unknown-key/sorted/nested/fallback tests FAIL (current validator returns `[]`).

- [ ] **Step 3: Implement** — replace `validate_arguments` in `src/internal/tools/validation.py`; keep `check_json_type`:

```python
import logging

from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError

try:  # jsonschema >= 4.18 resolves refs through `referencing`
    from referencing.exceptions import Unresolvable
except ImportError:  # pragma: no cover
    Unresolvable = Exception

logger = logging.getLogger(__name__)


def _path(parts) -> str:
    return ".".join(str(p) for p in parts)


def _shallow_errors(parameters, arguments) -> list[str]:
    # the previous validate_arguments body, unchanged
    ...


def validate_arguments(parameters, arguments) -> list[str]:
    if not parameters:
        return []
    try:
        Draft202012Validator.check_schema(parameters)
        found = list(Draft202012Validator(parameters).iter_errors(arguments))
    except (SchemaError, Unresolvable) as exc:
        logger.warning("Tool schema unusable (%s); falling back to shallow check", exc)
        return _shallow_errors(parameters, arguments)
    errors: list[str] = []
    for error in found:
        where = _path(error.absolute_path)
        if error.validator == "required":
            for name in error.validator_value:
                if isinstance(error.instance, dict) and name not in error.instance:
                    full = f"{where}.{name}" if where else name
                    errors.append(f"Missing required argument: {full!r}")
        elif where:
            errors.append(f"Argument {where!r}: {error.message}")
        else:
            errors.append(f"Arguments: {error.message}")
    return sorted(set(errors))
```

Errors are sorted as whole strings, so `Argument …` lines are ordered by path and precede `Missing …` lines. `set` drops duplicates jsonschema can report for one field.

- [ ] **Step 4: Add the dependency** — add `jsonschema>=4.18,<5` next to `mcp` in `requirements.txt` and in `requirements-unit-test.txt`.

- [ ] **Step 5: Run tests**

Run: `pytest tests/unit/test_tool_arg_validation.py tests/unit/test_tool_registry.py tests/unit/test_public_data_seeding.py tests/unit/test_requirements_layout.py -q`
Expected: PASS.

- [ ] **Step 6: Mutation check** — replace the `iter_errors` pass with `return _shallow_errors(...)`; the range/enum/items/unknown-key tests must go red. Revert, delete `__pycache__` for the module, rerun green.

- [ ] **Step 7: Full suite + lint, commit**

Run: `ruff check . && ruff format --check . && pytest -q`
Expected: only the known pre-existing `test_no_tool_calls_on_chat_path` failure (live OpenAI 429). Any other failure is a seeded tool whose real calls violate its own schema — fix the caller or report it, do not loosen the validator.

```bash
git add src/internal/tools/validation.py requirements.txt requirements-unit-test.txt tests/unit/test_tool_arg_validation.py docs/superpowers
git commit -m "tools: enforce the full JSON Schema on tool arguments"
```
