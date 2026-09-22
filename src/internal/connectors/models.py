"""Shared document models for connector implementations."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ConnectorStopSignal(Exception):
    """Raised to signal a connector run should stop cleanly."""


@dataclass(frozen=True)
class Document:
    """One source document emitted by a connector."""

    id: str
    contents: str
    title: str | None = None
    url: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    permissions: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SlimDocument:
    """Minimal document identity used by lightweight sync passes."""

    id: str
    metadata: dict[str, Any] = field(default_factory=dict)
    permissions: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class HierarchyNode:
    """Optional parent/child node for sources with folder-like structure."""

    id: str
    name: str
    parent_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ConnectorFailure:
    """Structured failure emitted while a connector continues processing."""

    document_id: str | None
    message: str
    exception_type: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Indexing-time section/document types
# ---------------------------------------------------------------------------


class SectionType(str, Enum):
    TEXT = "text"
    IMAGE = "image"
    TABULAR = "tabular"


@dataclass
class Section:
    text: str | None = None
    link: str | None = None
    type: SectionType = SectionType.TEXT
    image_file_id: str | None = None
    heading: str | None = None


@dataclass
class TextSection(Section):
    type: SectionType = SectionType.TEXT


@dataclass
class ImageSection(Section):
    type: SectionType = SectionType.IMAGE


@dataclass
class DocumentFailure:
    document_id: str
    document_link: str | None = None


@dataclass
class EntityFailure:
    entity_id: str


@dataclass
class IndexAttemptMetadata:
    connector_id: int | None = None
    credential_id: int | None = None


@dataclass
class IndexingDocument:
    """Extended document used during indexing with sections and metadata."""

    id: str
    sections: list[Section] = field(default_factory=list)
    title: str | None = None
    url: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    permissions: dict[str, Any] = field(default_factory=dict)
    source: str | None = None  # DocumentSource value
    semantic_identifier: str = ""
    # processed_sections is set by file processors; falls back to sections
    processed_sections: list[Section] = field(default_factory=list)

    def get_title_for_document_index(self) -> str | None:
        return self.title or None

    def get_text_content(self) -> str:
        return " ".join(s.text or "" for s in self.sections if s.text)

    def to_short_descriptor(self) -> str:
        return f"Document ID: {self.id}, Title: {self.title or '(no title)'}"
