"""Each migrated site reads its value from the timeout policies."""

from contextlib import contextmanager

from src.internal.configs.timeouts import load_timeout_policies, use_timeout_policies


@contextmanager
def overridden(overrides: dict):
    with use_timeout_policies(load_timeout_policies({}, overrides=overrides)) as p:
        yield p
