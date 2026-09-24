"""Validate call arguments against a tool's JSON-schema parameters."""

from __future__ import annotations

import logging
from typing import Any

from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError
from referencing.exceptions import Unresolvable

logger = logging.getLogger(__name__)


def check_json_type(value: Any, json_type: str) -> bool:
    if json_type == "string":
        return isinstance(value, str)
    if json_type == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if json_type == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if json_type == "boolean":
        return isinstance(value, bool)
    if json_type == "array":
        return isinstance(value, list)
    if json_type == "object":
        return isinstance(value, dict)
    if json_type == "null":
        return value is None
    return True  # unknown type — don't reject


def _shallow_errors(parameters: dict[str, Any], arguments: dict[str, Any]) -> list[str]:
    """Required keys and top-level types only: the fallback for unusable schemas."""
    errors: list[str] = []
    required = parameters.get("required", [])
    props: dict[str, Any] = parameters.get("properties", {})

    for req in required:
        if req not in arguments:
            errors.append(f"Missing required argument: {req!r}")

    for key, value in arguments.items():
        if key not in props:
            continue
        expected = props[key].get("type")
        if expected and not check_json_type(value, expected):
            errors.append(
                f"Argument {key!r}: expected {expected!r}, got {type(value).__name__!r}"
            )

    return errors


def _path(parts: Any) -> str:
    return ".".join(str(part) for part in parts)


def validate_arguments(
    parameters: dict[str, Any], arguments: dict[str, Any]
) -> list[str]:
    """Return a list of validation error strings (empty = valid).

    Every constraint the schema declares is enforced -- ranges, enums, array
    items, nested objects, closed objects -- so an invalid call is rejected
    with a message the model can act on, rather than clamped or dropped.

    A schema jsonschema cannot use (malformed, holding a ``$ref`` it cannot
    resolve, as OpenAPI parameter schemas can, or a ``$ref`` cycle) falls back to checking required
    keys and top-level types: one broken remote schema must neither fail every
    call to its tool nor skip validation entirely.
    """
    if not parameters:
        return []
    try:
        Draft202012Validator.check_schema(parameters)
        found = list(Draft202012Validator(parameters).iter_errors(arguments))
    # RecursionError: a $ref cycle with no base case (e.g. a:{$ref:a}) recurses
    # forever; legitimately recursive schemas terminate and never reach here.
    except (SchemaError, Unresolvable, RecursionError) as exc:
        logger.warning(
            "Tool schema unusable (%s); falling back to shallow check",
            getattr(exc, "message", exc),
        )
        return _shallow_errors(parameters, arguments)

    errors: set[str] = set()
    for error in found:
        where = _path(error.absolute_path)
        if error.validator == "required":
            for name in error.validator_value:
                if isinstance(error.instance, dict) and name not in error.instance:
                    full = f"{where}.{name}" if where else name
                    errors.add(f"Missing required argument: {full!r}")
        elif where:
            errors.add(f"Argument {where!r}: {error.message}")
        else:
            errors.add(f"Arguments: {error.message}")
    return sorted(errors)
