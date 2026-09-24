import pytest

from src.agents.tool.recovery import Action, RecoveryPolicy
from src.internal.tools import FailureCategory, ToolEffect, ToolFailure

POLICY = RecoveryPolicy()
MID = lambda low, high: 1.0  # noqa: E731 - jitter factor 1.0


def _f(category, *, provider_attempts=0, retry_after=None):
    return ToolFailure(
        category, "m", retry_after=retry_after, provider_attempts=provider_attempts
    )


@pytest.mark.parametrize("effect", list(ToolEffect))
@pytest.mark.parametrize(
    "category", [FailureCategory.INVALID_INPUT, FailureCategory.NOT_FOUND]
)
def test_input_failures_go_back_to_the_model(effect, category):
    assert POLICY.decide(_f(category), effect, 0, 10.0, MID).action is Action.FEED_BACK


@pytest.mark.parametrize("effect", [ToolEffect.SIDE_EFFECTING, ToolEffect.UNSPECIFIED])
@pytest.mark.parametrize(
    "category",
    [FailureCategory.TRANSIENT, FailureCategory.PERMANENT, FailureCategory.UNKNOWN],
)
def test_non_read_only_failures_escalate(effect, category):
    assert POLICY.decide(_f(category), effect, 0, 10.0, MID).action is Action.ESCALATE


def test_transient_read_only_retries_with_backoff():
    first = POLICY.decide(
        _f(FailureCategory.TRANSIENT), ToolEffect.READ_ONLY, 0, 10.0, MID
    )
    second = POLICY.decide(
        _f(FailureCategory.TRANSIENT), ToolEffect.READ_ONLY, 1, 10.0, MID
    )
    assert (first.action, first.delay) == (Action.RETRY, 0.5)
    assert (second.action, second.delay) == (Action.RETRY, 1.0)


def test_retries_run_out():
    got = POLICY.decide(
        _f(FailureCategory.TRANSIENT), ToolEffect.READ_ONLY, 2, 10.0, MID
    )
    assert got.action is Action.UNAVAILABLE


def test_provider_already_retried_is_not_retried_again():
    got = POLICY.decide(
        _f(FailureCategory.TRANSIENT, provider_attempts=3),
        ToolEffect.READ_ONLY,
        0,
        10.0,
        MID,
    )
    assert got.action is Action.UNAVAILABLE


def test_retry_after_is_honoured_up_to_the_cap():
    got = POLICY.decide(
        _f(FailureCategory.TRANSIENT, retry_after=2.0),
        ToolEffect.READ_ONLY,
        0,
        10.0,
        MID,
    )
    assert got.delay == 2.0
    capped = POLICY.decide(
        _f(FailureCategory.TRANSIENT, retry_after=60.0),
        ToolEffect.READ_ONLY,
        0,
        10.0,
        MID,
    )
    assert capped.delay == 4.0


def test_a_delay_that_does_not_fit_the_budget_degrades():
    got = POLICY.decide(
        _f(FailureCategory.TRANSIENT), ToolEffect.READ_ONLY, 0, 0.3, MID
    )
    assert got.action is Action.UNAVAILABLE


@pytest.mark.parametrize(
    "category", [FailureCategory.PERMANENT, FailureCategory.UNKNOWN]
)
def test_permanent_and_unknown_read_only_degrade(category):
    assert (
        POLICY.decide(_f(category), ToolEffect.READ_ONLY, 0, 10.0, MID).action
        is Action.UNAVAILABLE
    )


def test_jitter_scales_the_backoff():
    got = POLICY.decide(
        _f(FailureCategory.TRANSIENT),
        ToolEffect.READ_ONLY,
        0,
        10.0,
        lambda low, high: 1.5,
    )
    assert got.delay == 0.75
