"""Process-local coordination for human approval of tool calls."""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from src.agents.tool import (
    ApprovalDecision,
    EscalationDecision,
    ToolApprovalRequest,
    ToolEscalationRequest,
)
from src.internal.configs.timeouts import get_timeout_policies

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


@dataclass(frozen=True, slots=True)
class ToolEscalationView:
    id: str
    tool_name: str
    arguments: dict[str, object]
    category: str
    message: str
    attempts: int
    expires_at: str


@dataclass(slots=True)
class _PendingApproval:
    owner_user_id: str
    future: asyncio.Future[Any]
    expires_at: datetime
    view: Any


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


def _iso(moment: datetime) -> str:
    return moment.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


class DecisionBroker:
    """Process-local request -> human decision -> future, with owner checks and a timeout.

    Approval and escalation are separate subclasses with separate instances:
    they share these mechanics, never state.
    """

    kind = "decision"

    def __init__(
        self,
        timeout_seconds: float,
        *,
        expired: Any,
        allowed: Sequence[Any],
        counter_names: Mapping[Any, str],
    ) -> None:
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        self._timeout_seconds = timeout_seconds
        self._expired = expired
        self._allowed = frozenset(allowed)
        self._counter_names = dict(counter_names)  # decision -> counter key
        self._pending: dict[str, _PendingApproval] = {}
        self._lock = asyncio.Lock()
        self.counters = {
            "requested": 0,
            **{name: 0 for name in self._counter_names.values()},
            "expired": 0,
            "cancelled": 0,
            "errors": 0,
        }

    @property
    def pending_count(self) -> int:
        return len(self._pending)

    async def _request(
        self,
        owner_user_id: str,
        request_id: str,
        tool_name: str,
        expires_at: datetime,
        view: Any,
        on_registered: Callable[[Any], None] | None,
    ) -> Any:
        future = asyncio.get_running_loop().create_future()
        pending = _PendingApproval(owner_user_id, future, expires_at, view)
        started = time.perf_counter()
        completion_label: str | None = None

        async with self._lock:
            if request_id in self._pending:
                self.counters["errors"] += 1
                raise ApprovalConflict(request_id)
            self._pending[request_id] = pending
            self.counters["requested"] += 1

        try:
            if on_registered is not None:
                on_registered(view)
            remaining = (expires_at - datetime.now(timezone.utc)).total_seconds()
            timeout = max(0.0, min(self._timeout_seconds, remaining))
            try:
                decision = await asyncio.wait_for(asyncio.shield(future), timeout)
            # asyncio.TimeoutError, not the builtin: 3.11 made them the same
            # object, but on 3.10 -- the floor pyproject declares -- they are
            # distinct and the builtin does not catch this.
            except asyncio.TimeoutError:
                async with self._lock:
                    if not future.done():
                        future.set_result(self._expired)
                        self.counters["expired"] += 1
                decision = self._expired
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
                if self._pending.get(request_id) is pending:
                    del self._pending[request_id]
            decision_text = completion_label or (
                future.result().value
                if future.done() and not future.cancelled()
                else "cancelled"
            )
            logger.info(
                "Tool %s completed id=%s tool=%s decision=%s duration=%.3f",
                self.kind,
                request_id,
                tool_name,
                decision_text,
                time.perf_counter() - started,
            )

    async def decide(
        self,
        request_id: str,
        owner_user_id: str,
        decision: Any,
    ) -> None:
        try:
            async with self._lock:
                pending = self._pending.get(request_id)
                if pending is None:
                    raise ApprovalNotFound(request_id)
                if pending.owner_user_id != owner_user_id:
                    raise ApprovalForbidden(request_id)
                if pending.future.done():
                    raise ApprovalConflict(request_id)
                if datetime.now(timezone.utc) >= pending.expires_at:
                    pending.future.set_result(self._expired)
                    self.counters["expired"] += 1
                    raise ApprovalExpired(request_id)
                if decision not in self._allowed:
                    raise ApprovalConflict(request_id)
                pending.future.set_result(decision)
                self.counters[self._counter_names[decision]] += 1
        except (ApprovalNotFound, ApprovalForbidden, ApprovalConflict, ApprovalExpired):
            async with self._lock:
                self.counters["errors"] += 1
            raise


class ToolApprovalBroker(DecisionBroker):
    """Coordinate approval requests within one web-server process."""

    kind = "approval"

    def __init__(self, timeout_seconds: float | None = None) -> None:
        if timeout_seconds is None:
            timeout_seconds = get_timeout_policies().tool_loop.approval_timeout_seconds
        super().__init__(
            timeout_seconds,
            expired=ApprovalDecision.EXPIRED,
            allowed=(ApprovalDecision.APPROVE, ApprovalDecision.DENY),
            counter_names={
                ApprovalDecision.APPROVE: "approved",
                ApprovalDecision.DENY: "denied",
            },
        )

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
            expires_at=_iso(request.expires_at),
        )
        return await self._request(
            owner_user_id,
            request.approval_id,
            request.tool_name,
            request.expires_at,
            view,
            on_registered,
        )


class ToolEscalationBroker(DecisionBroker):
    """Coordinate tool-failure escalations within one web-server process."""

    kind = "escalation"

    def __init__(self, timeout_seconds: float | None = None) -> None:
        if timeout_seconds is None:
            timeout_seconds = (
                get_timeout_policies().tool_loop.escalation_timeout_seconds
            )
        super().__init__(
            timeout_seconds,
            expired=EscalationDecision.EXPIRED,
            allowed=(
                EscalationDecision.RETRY,
                EscalationDecision.SKIP,
                EscalationDecision.CANCEL,
            ),
            counter_names={
                EscalationDecision.RETRY: "retry",
                EscalationDecision.SKIP: "skip",
                EscalationDecision.CANCEL: "cancel",
            },
        )

    async def request(
        self,
        owner_user_id: str,
        request: ToolEscalationRequest,
        on_registered: Callable[[ToolEscalationView], None] | None = None,
    ) -> EscalationDecision:
        view = ToolEscalationView(
            id=request.escalation_id,
            tool_name=request.tool_name,
            arguments=sanitize_tool_arguments(request.arguments),
            category=request.category,
            message=request.message,
            attempts=request.attempts,
            expires_at=_iso(request.expires_at),
        )
        return await self._request(
            owner_user_id,
            request.escalation_id,
            request.tool_name,
            request.expires_at,
            view,
            on_registered,
        )
