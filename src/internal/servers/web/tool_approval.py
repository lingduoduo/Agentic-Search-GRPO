"""Process-local coordination for human approval of tool calls."""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from src.agents.tool import ApprovalDecision, ToolApprovalRequest

logger = logging.getLogger(__name__)

_SECRET_KEYS = {
    "password",
    "secret",
    "token",
    "cookie",
    "authorization",
    "headers",
    "apikey",
    "accesstoken",
    "refreshtoken",
    "clientsecret",
    "authtoken",
}
_ELLIPSIS = "…"


class ApprovalNotFound(Exception):
    """The approval ID is not currently pending."""


class ApprovalForbidden(Exception):
    """The approval belongs to another user."""


class ApprovalConflict(Exception):
    """The approval has already received a decision."""


class ApprovalExpired(Exception):
    """The approval deadline has passed."""


@dataclass(frozen=True, slots=True)
class ToolApprovalView:
    id: str
    tool_name: str
    arguments: dict[str, object]
    expires_at: str


@dataclass(slots=True)
class _PendingApproval:
    owner_user_id: str
    future: asyncio.Future[ApprovalDecision]
    expires_at: datetime
    view: ToolApprovalView


def sanitize_tool_arguments(arguments: Mapping[str, object]) -> dict[str, object]:
    """Return a bounded copy of tool arguments suitable for display."""

    sanitized = _sanitize(arguments, depth=0)
    return sanitized if isinstance(sanitized, dict) else {}


def _sanitize(value: object, *, depth: int) -> object:
    if isinstance(value, str):
        return value if len(value) <= 200 else value[:200] + _ELLIPSIS
    if isinstance(value, Mapping):
        if depth >= 2:
            return _ELLIPSIS
        result: dict[str, object] = {}
        for key, item in list(value.items())[:10]:
            key_text = str(key)
            normalized_key = "".join(
                character for character in key_text.casefold() if character.isalnum()
            )
            if normalized_key in _SECRET_KEYS:
                continue
            result[key_text] = _sanitize(item, depth=depth + 1)
        return result
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        if depth >= 2:
            return _ELLIPSIS
        return [_sanitize(item, depth=depth + 1) for item in value[:10]]
    if value is None or isinstance(value, (bool, int, float)):
        return value
    return _sanitize(str(value), depth=depth)


class ToolApprovalBroker:
    """Coordinate approval requests within one web-server process."""

    def __init__(self, timeout_seconds: float = 60.0) -> None:
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        self._timeout_seconds = timeout_seconds
        self._pending: dict[str, _PendingApproval] = {}
        self._lock = asyncio.Lock()
        self.counters = {
            "requested": 0,
            "approved": 0,
            "denied": 0,
            "expired": 0,
            "cancelled": 0,
            "errors": 0,
        }

    @property
    def pending_count(self) -> int:
        return len(self._pending)

    async def request(
        self,
        owner_user_id: str,
        request: ToolApprovalRequest,
        on_registered: Callable[[ToolApprovalView], None] | None = None,
    ) -> ApprovalDecision:
        view = ToolApprovalView(
            id=request.approval_id,
            tool_name=request.tool_name,
            arguments=sanitize_tool_arguments(request.arguments),
            expires_at=request.expires_at.astimezone(timezone.utc)
            .isoformat()
            .replace("+00:00", "Z"),
        )
        future = asyncio.get_running_loop().create_future()
        pending = _PendingApproval(owner_user_id, future, request.expires_at, view)
        started = time.perf_counter()
        completion_label: str | None = None

        async with self._lock:
            if request.approval_id in self._pending:
                self.counters["errors"] += 1
                raise ApprovalConflict(request.approval_id)
            self._pending[request.approval_id] = pending
            self.counters["requested"] += 1

        try:
            if on_registered is not None:
                on_registered(view)
            remaining = (
                request.expires_at - datetime.now(timezone.utc)
            ).total_seconds()
            timeout = max(0.0, min(self._timeout_seconds, remaining))
            try:
                decision = await asyncio.wait_for(asyncio.shield(future), timeout)
            # asyncio.TimeoutError, not the builtin: 3.11 made them the same
            # object, but on 3.10 -- the floor pyproject declares -- they are
            # distinct and the builtin does not catch this.
            except asyncio.TimeoutError:
                async with self._lock:
                    if not future.done():
                        future.set_result(ApprovalDecision.EXPIRED)
                        self.counters["expired"] += 1
                decision = ApprovalDecision.EXPIRED
            return decision
        except asyncio.CancelledError:
            async with self._lock:
                if not future.done():
                    future.cancel()
                    self.counters["cancelled"] += 1
            raise
        except Exception:
            async with self._lock:
                if not future.done():
                    future.cancel()
                self.counters["errors"] += 1
            completion_label = "error"
            raise
        finally:
            async with self._lock:
                if self._pending.get(request.approval_id) is pending:
                    del self._pending[request.approval_id]
            decision_text = completion_label or (
                future.result().value
                if future.done() and not future.cancelled()
                else "cancelled"
            )
            logger.info(
                "Tool approval completed id=%s tool=%s decision=%s duration=%.3f",
                request.approval_id,
                request.tool_name,
                decision_text,
                time.perf_counter() - started,
            )

    async def decide(
        self,
        approval_id: str,
        owner_user_id: str,
        decision: ApprovalDecision,
    ) -> None:
        try:
            async with self._lock:
                pending = self._pending.get(approval_id)
                if pending is None:
                    raise ApprovalNotFound(approval_id)
                if pending.owner_user_id != owner_user_id:
                    raise ApprovalForbidden(approval_id)
                if pending.future.done():
                    raise ApprovalConflict(approval_id)
                if datetime.now(timezone.utc) >= pending.expires_at:
                    pending.future.set_result(ApprovalDecision.EXPIRED)
                    self.counters["expired"] += 1
                    raise ApprovalExpired(approval_id)
                if decision not in (ApprovalDecision.APPROVE, ApprovalDecision.DENY):
                    raise ApprovalConflict(approval_id)
                pending.future.set_result(decision)
                self.counters[
                    "approved" if decision is ApprovalDecision.APPROVE else "denied"
                ] += 1
        except (ApprovalNotFound, ApprovalForbidden, ApprovalConflict, ApprovalExpired):
            async with self._lock:
                self.counters["errors"] += 1
            raise
