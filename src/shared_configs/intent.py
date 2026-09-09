"""Intent vocabulary and scoring defaults shared by serving and offline tools.

Keep this module standard-library-only: configuration readers must not load
web orchestration, NumPy, or an encoder just to resolve routing settings.
"""

from enum import Enum


class RouteStrategy(str, Enum):
    """High-level execution family selected by intent recognition."""

    CHAT = "chat"
    SEARCH = "search"
    TOOL = "tool"


INTENT_LABELS: tuple[str, ...] = tuple(strategy.value for strategy in RouteStrategy)
DEFAULT_TOP_K = 8
DEFAULT_MIN_ROUTE_MARGIN = 0.010
DEFAULT_MIN_MODULE_SCORE = 0.8215
