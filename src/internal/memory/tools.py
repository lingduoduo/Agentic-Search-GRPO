"""Internal add/update/delete memory Tools dispatched by the curation loop."""

from __future__ import annotations

from typing import Any

from src.internal.tools.base import (
    FailureCategory,
    ResultKind,
    Tool,
    ToolEffect,
    ToolErrorText,
    ToolFailure,
    ToolSchema,
)
from src.internal.tools.registry import ToolRegistry


class _AddMemoryTool(Tool):
    def __init__(self, store, user_id: str, counts: dict[str, int]) -> None:
        self._store, self._user_id, self._counts = store, user_id, counts

    @property
    def name(self) -> str:
        return "add_memory"

    @property
    def effect(self) -> ToolEffect:
        return ToolEffect.SIDE_EFFECTING

    @property
    def result_kind(self) -> ResultKind:
        return ResultKind.TEXT

    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name="add_memory",
            description="Store a new durable memory (one contextual sentence) about the user.",
            parameters={
                "type": "object",
                "properties": {
                    "content": {
                        "type": "string",
                        "minLength": 1,
                        "description": "The memory to store.",
                    }
                },
                "required": ["content"],
                "additionalProperties": False,
            },
        )

    async def execute(self, instance_id: str, arguments: dict[str, Any]):
        del instance_id
        record = self._store.add_user_memory(
            self._user_id, str(arguments.get("content", ""))
        )
        if record is None:
            text = "empty content; nothing added"
            failure = ToolFailure(FailureCategory.INVALID_INPUT, text)
            return ToolErrorText(text, failure), None, {"failure": failure}
        self._counts["add"] += 1
        return f"added memory {record.id}", record, {}


class _UpdateMemoryTool(Tool):
    def __init__(self, store, user_id: str, counts: dict[str, int]) -> None:
        self._store, self._user_id, self._counts = store, user_id, counts

    @property
    def name(self) -> str:
        return "update_memory"

    @property
    def effect(self) -> ToolEffect:
        return ToolEffect.SIDE_EFFECTING

    @property
    def result_kind(self) -> ResultKind:
        return ResultKind.TEXT

    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name="update_memory",
            description="Replace the content of an existing memory identified by memory_id.",
            parameters={
                "type": "object",
                "properties": {
                    "memory_id": {"type": "string", "minLength": 1},
                    "content": {"type": "string", "minLength": 1},
                },
                "required": ["memory_id", "content"],
                "additionalProperties": False,
            },
        )

    async def execute(self, instance_id: str, arguments: dict[str, Any]):
        del instance_id
        updated = self._store.update_user_memory(
            self._user_id,
            str(arguments.get("memory_id", "")),
            str(arguments.get("content", "")),
        )
        if updated is None:
            text = "memory not found"
            failure = ToolFailure(FailureCategory.INVALID_INPUT, text)
            return ToolErrorText(text, failure), None, {"failure": failure}
        self._counts["update"] += 1
        return f"updated memory {updated.id}", updated, {}


class _DeleteMemoryTool(Tool):
    def __init__(self, store, user_id: str, counts: dict[str, int]) -> None:
        self._store, self._user_id, self._counts = store, user_id, counts

    @property
    def name(self) -> str:
        return "delete_memory"

    @property
    def effect(self) -> ToolEffect:
        return ToolEffect.SIDE_EFFECTING

    @property
    def result_kind(self) -> ResultKind:
        return ResultKind.TEXT

    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name="delete_memory",
            description="Delete an outdated or incorrect memory identified by memory_id.",
            parameters={
                "type": "object",
                "properties": {"memory_id": {"type": "string", "minLength": 1}},
                "required": ["memory_id"],
                "additionalProperties": False,
            },
        )

    async def execute(self, instance_id: str, arguments: dict[str, Any]):
        del instance_id
        ok = self._store.delete_user_memory(
            self._user_id, str(arguments.get("memory_id", ""))
        )
        if not ok:
            text = "memory not found"
            failure = ToolFailure(FailureCategory.INVALID_INPUT, text)
            return ToolErrorText(text, failure), None, {"failure": failure}
        self._counts["delete"] += 1
        return "deleted memory", None, {}


def build_memory_registry(
    store, user_id: str
) -> tuple[ToolRegistry, dict[str, int], list[dict]]:
    counts = {"add": 0, "update": 0, "delete": 0}
    tools = [
        _AddMemoryTool(store, user_id, counts),
        _UpdateMemoryTool(store, user_id, counts),
        _DeleteMemoryTool(store, user_id, counts),
    ]
    registry = ToolRegistry()
    for tool in tools:
        registry.register(tool)
    schemas = [tool.schema.to_dict() for tool in tools]
    return registry, counts, schemas
