"""Validated, dependency-light text normalization for offline corpus hierarchies."""

from __future__ import annotations

from copy import deepcopy
import re
from urllib.parse import unquote, urlsplit

from bs4 import (
    BeautifulSoup,
    Comment,
    Declaration,
    Doctype,
    NavigableString,
    ProcessingInstruction,
)


def _string(value, field):
    if not isinstance(value, str):
        raise ValueError(f"{field} must be a string")
    return value


def _sections(value):
    if not isinstance(value, list):
        raise ValueError("sections must be a list")
    result = []
    for section in value:
        if not isinstance(section, dict) or set(section) != {"heading", "text"}:
            raise ValueError("each section requires heading and text")
        result.append(
            {
                key: _string(section[key], f"section.{key}").strip()
                for key in ("heading", "text")
            }
        )
    return result


def _html_sections(text):
    soup = BeautifulSoup(text, "html.parser")
    # Static visibility only: arbitrary stylesheets require browser CSS layout.
    for node in list(soup.descendants):
        if isinstance(node, (Comment, Declaration, Doctype, ProcessingInstruction)):
            node.extract()
    for tag in list(soup.find_all(True)):
        # An earlier excluded ancestor may already have decomposed this tag.
        if tag.decomposed:
            continue
        style = re.sub(r"\s+", "", str(tag.get("style", "")).lower())
        if (
            tag.name in {"head", "script", "style", "template", "input"}
            or tag.has_attr("hidden")
            or str(tag.get("aria-hidden", "")).strip().lower() == "true"
            or re.search(
                r"(?:^|;)(?:display:none|visibility:hidden)(?:!important)?(?:;|$)",
                style,
            )
        ):
            tag.decompose()
    sections = []
    heading, pieces = "Body", []

    def flush():
        body = re.sub(r"\s+", " ", "".join(pieces)).strip()
        if body or heading != "Body":
            sections.append({"heading": heading, "text": body})

    for node in soup.descendants:
        if getattr(node, "name", None) and re.fullmatch(r"h[1-6]", node.name):
            flush()
            heading, pieces = node.get_text(" ", strip=True), []
        elif getattr(node, "name", None) in (
            "p",
            "div",
            "br",
            "li",
            "article",
            "section",
            "tr",
        ):
            pieces.append("\n")
        elif isinstance(node, NavigableString):
            if not any(
                re.fullmatch(r"h[1-6]", parent.name or "") for parent in node.parents
            ):
                pieces.append(str(node))
    flush()
    return sections or [{"heading": "Body", "text": ""}]


def _text_sections(text, infer_layout):
    lines = text.splitlines()
    result, pieces = [], []
    heading, fence = "Body", None

    def flush():
        body = "\n".join(pieces).strip()
        if body or heading != "Body":
            result.append({"heading": heading, "text": body})

    i = 0
    while i < len(lines):
        line = lines[i]
        marker = re.match(r"^\s{0,3}(`{3,}|~{3,})", line)
        if marker:
            run = marker.group(1)
            if fence is None:
                fence = (run[0], len(run))
            elif (
                run[0] == fence[0]
                and len(run) >= fence[1]
                and not line[marker.end() :].strip()
            ):
                fence = None
            pieces.append(line)
            i += 1
            continue
        new_heading = None
        if fence is None:
            atx = re.match(r"^ {0,3}#{1,6}\s+(.+?)\s*#*\s*$", line)
            if atx:
                new_heading = atx.group(1)
            elif (
                line.strip()
                and i + 1 < len(lines)
                and re.fullmatch(r" {0,3}(?:=+|-+)\s*", lines[i + 1])
            ):
                new_heading = line.strip()
                i += 1
            elif (
                infer_layout
                and 3 <= len(line.strip()) <= 70
                and line.strip().isupper()
                and any(c.isalpha() for c in line)
                and (i == 0 or not lines[i - 1].strip())
                and i + 2 < len(lines)
                and not lines[i + 1].strip()
                and lines[i + 2].strip()
                and not re.search(r"[.!?。！？:;{}<>]", line)
            ):
                new_heading = line.strip()
        if new_heading is not None:
            flush()
            heading, pieces = new_heading, []
        else:
            pieces.append(line)
        i += 1
    flush()
    return result or [{"heading": "Body", "text": ""}]


def normalize_document(doc: dict, *, infer_layout: bool = False) -> dict:
    """Copy and validate a source document, recovering sections without fetching URLs."""
    if not isinstance(doc, dict):
        raise ValueError("document must be a dict")
    result = deepcopy(doc)
    identifier = _string(doc.get("id"), "id")
    if not identifier.strip():
        raise ValueError("id must be nonempty")
    contents = _string(doc.get("contents", doc.get("text", "")), "contents")
    title = _string(doc.get("title", ""), "title")
    url = _string("" if doc.get("url") is None else doc["url"], "url")
    metadata = {} if doc.get("metadata") is None else doc["metadata"]
    if not isinstance(metadata, dict):
        raise ValueError("metadata must be a dict")
    for key in ("topic", "product", "category", "format", "content_type"):
        if key in metadata:
            _string(metadata[key], f"metadata.{key}")
    if "hierarchy_path" in metadata:
        path = metadata["hierarchy_path"]
        if not isinstance(path, list) or any(
            not isinstance(p, str) or not p.strip() for p in path
        ):
            raise ValueError(
                "metadata.hierarchy_path must contain nonempty string labels"
            )
    # Validate both sources even when one has precedence.
    for source in (doc, metadata):
        if "sections" in source:
            _sections(source["sections"])
    section_source = "explicit"
    if "section_source" in doc and doc["section_source"] not in (
        "explicit",
        "html",
        "markdown",
        "layout",
        "plain",
    ):
        raise ValueError("invalid section_source")
    if "sections" in doc or "sections" in metadata:
        sections = _sections(
            doc["sections"] if "sections" in doc else metadata["sections"]
        )
        visible = "\n\n".join(
            filter(None, (f"{s['heading']}\n{s['text']}".strip() for s in sections))
        )
        if "section_source" in doc:
            section_source = doc["section_source"]
            visible = contents
    else:
        fmt = metadata.get("format", "").lower()
        mime = metadata.get("content_type", "").lower().split(";")[0].strip()
        html = fmt in ("html", "htm", "text/html") or mime in (
            "text/html",
            "application/xhtml+xml",
        )
        if fmt not in (
            "markdown",
            "md",
            "text",
            "plain",
            "text/plain",
        ) and not mime.startswith("text/plain"):
            # HTML examples inside Markdown fences do not establish HTML format.
            detection_lines = []
            detection_fence = None
            for line in contents.splitlines():
                marker = re.match(r"^\s{0,3}(`{3,}|~{3,})", line)
                if marker:
                    run = marker.group(1)
                    if detection_fence is None:
                        detection_fence = (run[0], len(run))
                    elif (
                        run[0] == detection_fence[0]
                        and len(run) >= detection_fence[1]
                        and not line[marker.end() :].strip()
                    ):
                        detection_fence = None
                elif detection_fence is None:
                    detection_lines.append(line)
            detection_text = "\n".join(detection_lines)
            html = html or bool(
                re.search(
                    r"<(?:html|body|article|section|div|p|h[1-6]|ul|ol|table)(?:\s[^<>]*|)>(?:.|\n)*?</",
                    detection_text,
                    re.I,
                )
            )
        sections = (
            _html_sections(contents) if html else _text_sections(contents, infer_layout)
        )
        if html:
            section_source = "html"
        elif any(s["heading"] != "Body" for s in _text_sections(contents, False)):
            section_source = "markdown"
        elif any(s["heading"] != "Body" for s in sections):
            section_source = "layout"
        else:
            section_source = "plain"
        visible = (
            "\n\n".join(
                (
                    ("" if s["heading"] == "Body" else s["heading"] + "\n") + s["text"]
                ).strip()
                for s in sections
            )
            if html
            else contents.strip()
        )
    if "hierarchy_path" in metadata:
        path, source = (
            [p.strip() for p in metadata["hierarchy_path"]],
            "metadata.hierarchy_path",
        )
    elif any(metadata.get(key, "").strip() for key in ("topic", "product", "category")):
        path, source = (
            [
                metadata[key].strip()
                for key in ("topic", "product", "category")
                if metadata.get(key, "").strip()
            ],
            "metadata",
        )
    else:
        try:
            parsed = urlsplit(url)
            path = (
                [parsed.hostname]
                + [unquote(p) for p in parsed.path.split("/")[1:-1] if p]
                if parsed.hostname
                else []
            )
        except ValueError as exc:
            raise ValueError("invalid document url") from exc
        source = "url" if path else "none"
    result.update(
        id=identifier,
        title=title,
        contents=visible,
        url=url,
        metadata=deepcopy(metadata),
        sections=sections,
        section_source=section_source,
        structure_path=path,
        structure_source=source,
    )
    return result


def first_sentence(text: str) -> str:
    """Return the first multilingual sentence, capped at 500 characters."""
    _string(text, "text")
    text = text.strip()
    match = re.search(r"[.!?。！？]", text[:500])
    return text[: match.end() if match else 500].strip()


def split_passages(text: str, *, max_words: int, max_chars: int) -> list[str]:
    """Bound passages by whitespace words and characters, preserving every token."""
    _string(text, "text")
    for name, limit in (("max_words", max_words), ("max_chars", max_chars)):
        if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
            raise ValueError(f"{name} must be a positive integer")
    chunks, current = [], ""
    previous_end = 0
    for match in re.finditer(r"\S+", text):
        token = match.group()
        gap = text[previous_end : match.start()]
        previous_end = match.end()
        while len(token) > max_chars:
            if current:
                chunks.append(current)
                current = ""
            chunks.append(token[:max_chars])
            token = token[max_chars:]
        if not token:
            continue
        candidate = current + gap + token if current else token
        if len(candidate) > max_chars or len(candidate.split()) > max_words:
            chunks.append(current)
            current = token
        else:
            current = candidate
    if current:
        chunks.append(current)
    return chunks
