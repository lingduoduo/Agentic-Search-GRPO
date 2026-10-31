"""Deterministic sample coverage and explicitly heuristic description warnings."""

from __future__ import annotations

import hashlib
import re

import numpy as np

DESCRIPTION_POLICY_VERSION = "coverage-v1"


def _language(doc):
    metadata = doc.get("metadata", {})
    language = metadata.get("language") or metadata.get("lang")
    if language:
        return str(language).lower()
    text = doc.get("title", "") + doc.get("contents", "")[:1000]
    for name, pattern in (
        # Kana first: Japanese text almost always also contains Han characters.
        ("kana", r"[\u3040-\u30ff]"),
        ("han", r"[\u3400-\u9fff]"),
        ("arabic", r"[\u0600-\u06ff]"),
        ("cyrillic", r"[\u0400-\u04ff]"),
    ):
        if re.search(pattern, text):
            return name
    return "latin-or-other"


def select_description_samples(documents, *, vectors=None, limit=12):
    """Select centroid, outlier, language and farthest-first coverage examples.

    Roles identify semantic geometry or a content-derived lexical proxy; neither
    is a claim that a sample exhaustively represents its source group.
    """
    if type(limit) is not int or limit < 1:
        raise ValueError("limit must be a positive integer")
    docs = sorted(
        documents, key=lambda d: (d.get("title", ""), d.get("contents", ""), d["id"])
    )
    if len({d["id"] for d in docs}) != len(docs):
        raise ValueError("duplicate document IDs")
    if not docs:
        return []
    semantic = vectors is not None and all(d["id"] in vectors for d in docs)
    if semantic:
        matrix = np.asarray([vectors[d["id"]] for d in docs], dtype=float)
        if matrix.ndim != 2 or not matrix.shape[1] or not np.isfinite(matrix).all():
            raise ValueError("invalid description vectors")
    else:
        matrix = np.zeros((len(docs), 256))
        for i, doc in enumerate(docs):
            text = doc.get("title", "") + " " + doc.get("contents", "")[:1000]
            tokens = re.findall(r"\w+", text.lower()) + list(
                re.sub(r"\s", "", text.lower())
            )
            for token in tokens:
                matrix[
                    i,
                    int.from_bytes(hashlib.sha256(token.encode()).digest()[:2], "big")
                    % 256,
                ] += 1
    norms = np.linalg.norm(matrix, axis=1)
    matrix = matrix / np.where(norms > 0, norms, 1)[:, None]
    center = matrix.mean(axis=0)
    distances = matrix @ center
    selected = {}
    proxy = "semantic_vector" if semantic else "lexical_proxy"

    def add(index, role):
        if index not in selected and len(selected) >= limit:
            return
        selected.setdefault(index, [proxy])
        if role not in selected[index]:
            selected[index].append(role)

    add(int(np.argmax(distances)), "typical")
    strata = {}
    for i, doc in enumerate(docs):
        strata.setdefault(_language(doc), []).append(i)
    for lang, indices in sorted(
        strata.items(), key=lambda item: (len(item[1]), item[0])
    ):
        add(min(indices, key=lambda i: (distances[i], i)), "language:" + lang)
    add(int(np.argmin(distances)), "boundary")
    while len(selected) < min(limit, len(docs)):
        remaining = [i for i in range(len(docs)) if i not in selected]
        chosen = min(
            remaining,
            key=lambda i: (max(float(matrix[i] @ matrix[j]) for j in selected), i),
        )
        add(chosen, "minority")
    uncovered = set(strata) - {_language(docs[i]) for i in selected}
    if uncovered:
        selected[next(iter(selected))].append("coverage_limited:language")
    return [
        dict(document_id=docs[i]["id"], roles=roles) for i, roles in selected.items()
    ]


def assess_description(text):
    """Warning proxies only, not calibrated semantic quality judgments."""
    flags = []
    if not isinstance(text, str) or not text.strip():
        flags.append("missing_or_blank")
    else:
        normalized = text.strip().lower()
        if re.search(
            r"\b(various|miscellaneous|general|assorted)\b.*\b(documents|topics|information)\b",
            normalized,
        ) or re.search(r"(各种|各类|多种|相关).{0,8}(文档|文件|内容|资料)", normalized):
            flags.append("generic_description")
        if (
            len(re.sub(r"\W", "", normalized)) < 12
            or len(set(re.findall(r"\w+", normalized))) < 3
            and not re.search(r"[\u3400-\u9fff]", normalized)
        ):
            flags.append("low_information")
    return dict(weak=bool(flags), flags=flags)
