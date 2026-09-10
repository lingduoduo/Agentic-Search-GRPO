"""Observability: tracer, admin surface, metric taxonomy, stage metrics.

``admin_surface`` is loaded lazily: it imports the store and the tool registry,
and this package is now imported from the retrieval client and the LLM
backends (for ``stage_metrics``), which must stay light and cycle-free.
"""

from __future__ import annotations

from importlib import import_module

from .tracer import NoOpTracer, OtelTracer, get_tracer, set_tracer

_LAZY = {
    "AdminSurfaceCard": ".admin_surface",
    "AdminSurfaceMetric": ".admin_surface",
    "AdminSurfaceSummary": ".admin_surface",
    "build_admin_surface_summary": ".admin_surface",
}

__all__ = [
    "AdminSurfaceCard",
    "AdminSurfaceMetric",
    "AdminSurfaceSummary",
    "build_admin_surface_summary",
    "NoOpTracer",
    "OtelTracer",
    "get_tracer",
    "set_tracer",
]


def __getattr__(name: str):
    module = _LAZY.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(import_module(module, __name__), name)
