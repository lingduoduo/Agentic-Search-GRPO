"""Retriever-free hierarchy navigation inspired by DecisionRAG (arXiv:2610.35868)."""

from __future__ import annotations

import math
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from src.internal.document_index.hierarchy import HierarchyArtifact

from src.internal.retrieval.acl import acl_allows


@dataclass(frozen=True)
class DecisionCandidate:
    id: str
    kind: str
    text: str


class DecisionModel(Protocol):
    """Scores are ordered like candidates; binary scores need not sum to one."""

    def binary(
        self, query: str, candidates: Sequence[DecisionCandidate]
    ) -> Sequence[float]: ...

    def choice(
        self, query: str, candidates: Sequence[DecisionCandidate]
    ) -> Sequence[float]: ...


@dataclass(frozen=True)
class DecisionRetrieverConfig:
    beam_width: int = 5
    shortlist_size: int = 20
    batch_size: int = 16
    passage_words: int = 200
    min_relevance: float = 0.0
    max_passage_chars: int = 1200
    beam_widths: dict[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in (
            "beam_width",
            "shortlist_size",
            "batch_size",
            "passage_words",
            "max_passage_chars",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if not isinstance(self.beam_widths, dict) or any(
            kind not in {"group", "topic", "document", "section"}
            or isinstance(width, bool)
            or not isinstance(width, int)
            or width < 1
            for kind, width in self.beam_widths.items()
        ):
            raise ValueError(
                "beam_widths must map known internal kinds to positive integers"
            )
        if not math.isfinite(self.min_relevance) or not 0 <= self.min_relevance <= 1:
            raise ValueError("min_relevance must be a finite probability")


@dataclass(frozen=True)
class DecisionStage:
    kind: str
    candidate_ids: tuple[str, ...]
    scores: tuple[float, ...]
    selected_ids: tuple[str, ...]


@dataclass
class NavigationResult:
    results: list[dict] = field(default_factory=list)
    stages: list[DecisionStage] = field(default_factory=list)


@dataclass(frozen=True)
class _Node:
    candidate: DecisionCandidate
    children: tuple[_Node, ...] = ()
    document: dict | None = None


def _sections(text: str) -> list[tuple[str, str]]:
    """Use Markdown headings when available; retain the preamble as a section."""
    sections: list[tuple[str, str]] = []
    heading = "Body"
    start = 0
    fenced = False
    fence = ""
    offset = 0
    for line in text.splitlines(keepends=True):
        stripped = line.lstrip()
        if stripped.startswith(("```", "~~~")):
            marker = stripped[:3]
            if not fenced:
                fenced, fence = True, marker
            elif marker == fence:
                fenced = False
        match = None if fenced else re.match(r"^#{1,6}\s+(.+?)\s*#*\s*$", line)
        if match:
            if text[start:offset].strip():
                sections.append((heading, text[start:offset]))
            heading = match.group(1)
            start = offset
        offset += len(line)
    if text[start:].strip():
        sections.append((heading, text[start:]))
    return sections


def _validate_scores(
    values: Sequence[float], count: int, *, choice: bool = False
) -> list[float]:
    if len(values) != count:
        raise ValueError(
            f"Decision model returned {len(values)} scores for {count} candidates"
        )
    scores = [float(value) for value in values]
    if any(not math.isfinite(score) or not 0 <= score <= 1 for score in scores):
        raise ValueError("Decision model scores must be finite probabilities in [0, 1]")
    if choice and not math.isclose(sum(scores), 1.0, abs_tol=1e-5):
        raise ValueError("Multiple-choice probabilities must sum to one")
    return scores


class DecisionRetriever:
    """Navigate structural metadata, never a dense/sparse candidate generator.

    Branch scores are used only for pruning, not multiplied into passage scores.
    All returned scores come from one final shared-context choice judgment.
    """

    def __init__(
        self,
        docs: Sequence[dict],
        model: DecisionModel,
        config: DecisionRetrieverConfig | None = None,
        *,
        hierarchy: HierarchyArtifact | None = None,
    ) -> None:
        self._hierarchy = hierarchy
        if hierarchy is not None:
            hierarchy.validate()
            supplied_ids = [str(doc.get("id", i)) for i, doc in enumerate(docs)]
            if len(set(supplied_ids)) != len(supplied_ids) or set(supplied_ids) != {
                doc["id"] for doc in hierarchy.documents
            }:
                raise ValueError(
                    "Supplied corpus document IDs do not match hierarchy artifact"
                )
            # Persisted content, sections and ACLs are one consistent snapshot.
            docs = hierarchy.documents
        self.model = model
        self.config = config or DecisionRetrieverConfig()
        self._topics = self._build(docs)

    def _build(self, docs: Sequence[dict]) -> tuple[_Node, ...]:
        from src.internal.document_index.hierarchy_formats import split_passages

        topics: dict[str, list[_Node]] = {}
        document_nodes: dict[str, _Node] = {}
        seen: set[str] = set()
        for index, doc in enumerate(docs):
            doc_id = str(doc.get("id", index))
            if doc_id in seen:
                raise ValueError(f"Duplicate document id: {doc_id}")
            seen.add(doc_id)
            text = doc.get("contents", doc.get("text", ""))
            title = doc.get("title", "")
            metadata = doc.get("metadata")
            if metadata is None:
                metadata = {}
            if not isinstance(text, str) or not isinstance(title, str):
                raise ValueError("Document title and contents/text must be strings")
            if not isinstance(metadata, dict):
                raise ValueError("Document metadata must be an object")
            topic = (
                metadata.get("topic", "Documents")
                if self._hierarchy is None
                else "Documents"
            )
            if not isinstance(topic, str) or not topic.strip():
                raise ValueError("metadata.topic must be a non-empty string")
            sections = []
            passage_index = 0
            section_bodies = (
                [
                    (
                        section["heading"],
                        (
                            section["heading"] + "\n"
                            if section["heading"] != "Body"
                            else ""
                        )
                        + section["text"],
                    )
                    for section in doc["sections"]
                ]
                if self._hierarchy is not None
                else _sections(text)
            )
            for section_index, (heading, body) in enumerate(section_bodies):
                passages = []
                for passage in split_passages(
                    body,
                    max_words=self.config.passage_words,
                    max_chars=self.config.max_passage_chars,
                ):
                    key = f"{doc_id}::passage-{passage_index}"
                    document = {
                        "id": key,
                        "title": title,
                        "text": passage,
                        "url": doc.get("url"),
                        "metadata": {
                            **metadata,
                            "document_id": doc_id,
                            "section": heading,
                            "passage_index": passage_index,
                        },
                    }
                    passages.append(
                        _Node(
                            DecisionCandidate(
                                key, "passage", f"{title}\n{heading}\n{passage}"
                            ),
                            document=document,
                        )
                    )
                    passage_index += 1
                if not passages:
                    continue
                sections.append(
                    _Node(
                        DecisionCandidate(
                            f"d{index}/s{section_index}",
                            "section",
                            f"{title}\n{heading}",
                        ),
                        tuple(passages),
                    )
                )
            if sections:
                document_node = _Node(
                    DecisionCandidate(f"d{index}", "document", title or doc_id),
                    tuple(sections),
                )
                topics.setdefault(topic, []).append(document_node)
                document_nodes[doc_id] = document_node
        if self._hierarchy is not None:
            return self._build_groups(document_nodes)
        return tuple(
            _Node(DecisionCandidate(f"t{index}", "topic", topic), tuple(children))
            for index, (topic, children) in enumerate(topics.items())
        )

    def _build_groups(self, document_nodes: dict[str, _Node]) -> tuple[_Node, ...]:
        child_groups: dict[str | None, list[dict]] = {}
        for group in self._hierarchy.groups:
            child_groups.setdefault(group["parent_id"], []).append(group)

        def build(group: dict) -> _Node | None:
            children = [
                document_nodes[i] for i in group["document_ids"] if i in document_nodes
            ]
            children.extend(
                node
                for child in child_groups.get(group["id"], [])
                if (node := build(child)) is not None
            )
            if not children:
                return None
            return _Node(
                DecisionCandidate(
                    f"g/{group['id']}",
                    "group",
                    f"{group['label']}\n{group['description']}",
                ),
                tuple(children),
            )

        return tuple(
            node
            for group in child_groups.get(None, [])
            if (node := build(group)) is not None
        )

    def _screen_nodes(
        self, query: str, candidates: list[_Node], limit: int, result: NavigationResult
    ) -> list[_Node]:
        scores = []
        for start in range(0, len(candidates), self.config.batch_size):
            batch = candidates[start : start + self.config.batch_size]
            scores.extend(
                _validate_scores(
                    self.model.binary(query, [node.candidate for node in batch]),
                    len(batch),
                )
            )
        ranked = sorted(
            (i for i, score in enumerate(scores) if score >= self.config.min_relevance),
            key=lambda i: scores[i],
            reverse=True,
        )[:limit]
        selected = [candidates[i] for i in ranked]
        result.stages.append(
            DecisionStage(
                candidates[0].candidate.kind,
                tuple(node.candidate.id for node in candidates),
                tuple(scores),
                tuple(node.candidate.id for node in selected),
            )
        )
        return selected

    def _visible(self, node: _Node, filters: dict | None) -> _Node | None:
        if node.document is not None:
            return node if acl_allows(node.document["metadata"], filters) else None
        children = tuple(
            child
            for original in node.children
            if (child := self._visible(original, filters)) is not None
        )
        if not children:
            return None
        candidate = node.candidate
        if candidate.kind == "group":
            # Generated summaries may mix ACL domains. Reconstruct from visible
            # document titles only; never send the persisted summary to a judge.
            def titles(nodes):
                for child in nodes:
                    if child.candidate.kind == "document":
                        yield child.candidate.text
                    else:
                        yield from titles(child.children)

            candidate = DecisionCandidate(
                candidate.id,
                "group",
                "Visible documents: " + "; ".join(titles(children))[:1600],
            )
        return _Node(candidate, children)

    def navigate(
        self, query: str, topk: int = 5, *, filters: dict | None = None
    ) -> NavigationResult:
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string")
        if (
            isinstance(topk, bool)
            or not isinstance(topk, int)
            or not 1 <= topk <= self.config.shortlist_size
        ):
            raise ValueError(f"topk must be between 1 and {self.config.shortlist_size}")
        result = NavigationResult()
        if filters and filters.get("access_acl"):
            candidates = [
                node
                for original in self._topics
                if (node := self._visible(original, filters)) is not None
            ]
        else:
            # The tree is immutable; unrestricted queries can reuse it directly.
            candidates = list(self._topics)
        # Different branches can contain documents at different group depths.
        # Park documents while expanding groups so groups and pages never compete
        # in a single relevance pool and shallow evidence is not lost.
        parked_documents = []
        while any(node.candidate.kind == "group" for node in candidates):
            parked_documents.extend(
                node for node in candidates if node.candidate.kind != "group"
            )
            groups = [node for node in candidates if node.candidate.kind == "group"]
            selected = self._screen_nodes(
                query,
                groups,
                self.config.beam_widths.get("group", self.config.beam_width),
                result,
            )
            candidates = [child for node in selected for child in node.children]
        candidates = parked_documents + candidates
        while candidates:
            leaf = candidates[0].document is not None
            limit = (
                self.config.shortlist_size
                if leaf
                else self.config.beam_widths.get(
                    candidates[0].candidate.kind, self.config.beam_width
                )
            )
            selected = self._screen_nodes(query, candidates, limit, result)
            if leaf:
                if selected:
                    capacity = getattr(self.model, "choice_capacity", None)
                    if capacity is not None:
                        count = capacity(query, [node.candidate for node in selected])
                        if (
                            isinstance(count, bool)
                            or not isinstance(count, int)
                            or not 1 <= count <= len(selected)
                        ):
                            raise ValueError("Invalid decision model choice capacity")
                        selected = selected[:count]
                    choice_scores = _validate_scores(
                        self.model.choice(query, [node.candidate for node in selected]),
                        len(selected),
                        choice=True,
                    )
                    order = sorted(
                        range(len(selected)),
                        key=lambda i: choice_scores[i],
                        reverse=True,
                    )[:topk]
                    result.results = [
                        {"document": selected[i].document, "score": choice_scores[i]}
                        for i in order
                    ]
                    result.stages.append(
                        DecisionStage(
                            "choice",
                            tuple(node.candidate.id for node in selected),
                            tuple(choice_scores),
                            tuple(selected[i].candidate.id for i in order),
                        )
                    )
                break
            candidates = [child for node in selected for child in node.children]
        return result

    def retrieve(
        self, queries: list[str], topk: int, *, filters: dict | None = None
    ) -> list[list[dict]]:
        return [
            self.navigate(query, topk, filters=filters).results for query in queries
        ]
