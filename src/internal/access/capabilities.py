"""What a request's identity is entitled to.

Identity used to be re-derived at each site that cared — filters here, memory
there, tools somewhere else — so a path could enforce the ACL while its
neighbour did not. This is the one place that mapping happens.

Anonymous is an identity, not the absence of one: it carries ``["public"]``.
``access_acl`` is therefore never empty, so no caller can accidentally express
"unfiltered" by passing nothing.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from src.context.preprocessing.access_filters import build_access_filter
from src.internal.access.access import PUBLIC_ACL
from src.internal.memory.service import memory_preamble

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RequestCapabilities:
    """What this caller may see, and what it brings with it."""

    user_id: str | None
    access_acl: list[str]
    memory_preamble: str

    @property
    def user_present(self) -> bool:
        return self.user_id is not None


ANONYMOUS = RequestCapabilities(
    user_id=None, access_acl=[PUBLIC_ACL], memory_preamble=""
)


def resolve_capabilities(
    user, store, *, query: str | None = None, encoder=None
) -> RequestCapabilities:
    """Map a resolved user (or ``None``) to its capabilities.

    ``store`` is required rather than optional because the memory preamble is
    read from it. Keeping it an argument leaves this a plain function with no
    global state, so the agent loops and MCP paths can call it too — something a
    FastAPI dependency could not reach.

    ``query`` and ``encoder`` let the preamble favor memories relevant to the
    current request once the user is above the injection cap; both are
    optional and the anonymous short-circuit runs before either is used.
    """
    if user is None or getattr(user, "is_anonymous", False):
        return ANONYMOUS

    user_id = user.id
    acl = build_access_filter(
        user_id,
        email=getattr(user, "email", None),
        group_ids=getattr(user, "group_ids", None),
    )
    try:
        preamble = memory_preamble(store, user_id, query=query, encoder=encoder)
    except Exception as exc:  # noqa: BLE001 — memory must never fail a request
        logger.warning("memory preamble unavailable for %s: %s", user_id, exc)
        preamble = ""
    return RequestCapabilities(
        user_id=user_id, access_acl=acl, memory_preamble=preamble
    )


__all__ = ["ANONYMOUS", "RequestCapabilities", "resolve_capabilities"]
