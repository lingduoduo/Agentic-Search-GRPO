"""Offline corpus hierarchy: build, maintain, describe and inspect.

Modules: ``build`` (config, artifact, construction), ``maintenance`` (torch-free
structural planner), ``descriptions`` (sample coverage), ``formats`` (text
normalization), ``models`` (lazy local embedding/description models) and
``cli`` (``python -m src.internal.document_index.hierarchy.cli``).
"""

from .build import HierarchyArtifact, HierarchyConfig, build_hierarchy

__all__ = ["HierarchyArtifact", "HierarchyConfig", "build_hierarchy"]
