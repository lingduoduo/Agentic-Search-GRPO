import asyncio
import logging
from datetime import datetime, timedelta, timezone

import pytest

from src.agents.tool import ApprovalDecision, ToolApprovalRequest
from src.internal.servers.web.tool_approval import (
    ApprovalConflict,
    ApprovalExpired,
    ApprovalForbidden,
    ApprovalNotFound,
    ToolApprovalBroker,
    sanitize_tool_arguments,
)


def _request(approval_id: str = "approval-1", expires_in: float = 1.0):
    now = datetime.now(timezone.utc)
    return ToolApprovalRequest(
        approval_id=approval_id,
        tool_name="create_ticket",
        arguments={"title": "example"},
        created_at=now,
        expires_at=now + timedelta(seconds=expires_in),
    )


async def _wait_until_pending(broker: ToolApprovalBroker) -> None:
    for _ in range(100):
        if broker.pending_count == 1:
            return
        await asyncio.sleep(0)
    raise AssertionError("approval was not registered")


def test_sanitizer_bounds_strings_collections_depth_and_secret_keys() -> None:
    result = sanitize_tool_arguments(
        {
            "text": "x" * 201,
            "items": list(range(12)),
            "nested": {"level1": {"level2": {"level3": "hidden"}}},
            "PASSWORD": "hidden",
            "Secret": "hidden",
            "token": "hidden",
            "Cookie": "hidden",
            "AUTHORIZATION": "hidden",
            "headers": {"safe": "not really"},
            "Api_Key": "hidden",
        }
    )

    assert result["text"] == "x" * 200 + "…"
    assert result["items"] == list(range(10))
    assert result["nested"] == {"level1": "…"}
    assert set(result) == {"text", "items", "nested"}


def test_sanitizer_removes_secret_key_variants_at_every_visible_depth() -> None:
    result = sanitize_tool_arguments(
        {
            "access_token": "hidden",
            "refreshToken": "hidden",
            "CLIENT_SECRET": "hidden",
            "auth-token": "hidden",
            "apiKey": "hidden",
            "nested": {
                "AccessToken": "hidden",
                "client_secret": "hidden",
                "safeValue": "visible",
            },
            "tokenizer": "visible",
            "secretariat": "visible",
            "cookie_policy": "visible",
        }
    )

    assert result == {
        "nested": {"safeValue": "visible"},
        "tokenizer": "visible",
        "secretariat": "visible",
        "cookie_policy": "visible",
    }


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("decision", "counter"),
    [(ApprovalDecision.APPROVE, "approved"), (ApprovalDecision.DENY, "denied")],
)
async def test_request_resolves_decision_and_cleans_up(decision, counter) -> None:
    broker = ToolApprovalBroker()
    views = []
    task = asyncio.create_task(broker.request("user-1", _request(), views.append))
    await _wait_until_pending(broker)

    assert views[0].arguments == {"title": "example"}
    await broker.decide("approval-1", "user-1", decision)
    assert await task is decision
    assert broker.pending_count == 0
    assert broker.counters == {
        "requested": 1,
        "approved": int(counter == "approved"),
        "denied": int(counter == "denied"),
        "expired": 0,
        "cancelled": 0,
        "errors": 0,
    }


@pytest.mark.asyncio
async def test_wrong_user_is_forbidden_without_resolving_request() -> None:
    broker = ToolApprovalBroker()
    task = asyncio.create_task(broker.request("owner", _request()))
    await _wait_until_pending(broker)

    with pytest.raises(ApprovalForbidden):
        await broker.decide("approval-1", "intruder", ApprovalDecision.APPROVE)
    assert broker.counters["errors"] == 1
    await broker.decide("approval-1", "owner", ApprovalDecision.DENY)
    assert await task is ApprovalDecision.DENY
    assert broker.pending_count == 0


@pytest.mark.asyncio
async def test_duplicate_decision_conflicts() -> None:
    broker = ToolApprovalBroker()
    task = asyncio.create_task(broker.request("owner", _request()))
    await _wait_until_pending(broker)
    await broker.decide("approval-1", "owner", ApprovalDecision.APPROVE)

    with pytest.raises(ApprovalConflict):
        await broker.decide("approval-1", "owner", ApprovalDecision.DENY)
    assert broker.counters["errors"] == 1
    assert await task is ApprovalDecision.APPROVE
    assert broker.pending_count == 0


@pytest.mark.asyncio
async def test_unknown_id_is_not_found() -> None:
    broker = ToolApprovalBroker()
    with pytest.raises(ApprovalNotFound):
        await broker.decide("missing", "owner", ApprovalDecision.APPROVE)
    assert broker.counters["errors"] == 1
    assert broker.pending_count == 0


@pytest.mark.asyncio
async def test_expiry_returns_expired_and_cleans_up() -> None:
    broker = ToolApprovalBroker()
    assert (
        await broker.request("owner", _request(expires_in=0.01))
        is ApprovalDecision.EXPIRED
    )
    assert broker.counters["expired"] == 1
    assert broker.pending_count == 0


@pytest.mark.asyncio
async def test_decision_after_deadline_reports_expired() -> None:
    broker = ToolApprovalBroker()
    task = asyncio.create_task(broker.request("owner", _request()))
    await _wait_until_pending(broker)
    broker._pending["approval-1"].expires_at = datetime.now(timezone.utc) - timedelta(
        seconds=1
    )

    with pytest.raises(ApprovalExpired):
        await broker.decide("approval-1", "owner", ApprovalDecision.APPROVE)
    assert broker.counters["errors"] == 1
    assert await task is ApprovalDecision.EXPIRED
    assert broker.pending_count == 0


@pytest.mark.asyncio
async def test_cancellation_cleans_up_and_counts_once() -> None:
    broker = ToolApprovalBroker()
    task = asyncio.create_task(broker.request("owner", _request()))
    await _wait_until_pending(broker)
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task
    assert broker.counters["cancelled"] == 1
    assert broker.pending_count == 0


@pytest.mark.asyncio
async def test_registration_callback_failure_cleans_up_and_logs_error(
    caplog,
) -> None:
    broker = ToolApprovalBroker()

    def fail_registration(_view) -> None:
        raise RuntimeError("publication failed")

    with caplog.at_level(logging.INFO, logger="src.internal.servers.web.tool_approval"):
        with pytest.raises(RuntimeError, match="publication failed"):
            await broker.request("owner", _request(), fail_registration)

    assert broker.pending_count == 0
    assert broker.counters["errors"] == 1
    assert broker.counters["cancelled"] == 0
    assert "decision=error" in caplog.text
    assert "decision=cancelled" not in caplog.text
