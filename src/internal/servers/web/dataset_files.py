"""Turn an uploaded file into corpus documents ({id, title, contents, metadata}).

- ``.jsonl``: one document per line, as the corpus files already are. A line
  needs ``contents`` (or ``text``); ``id``, ``title`` and ``metadata`` are kept.
- ``.txt`` / ``.md``: plain text, split into passages with the indexer's
  paragraph-aware splitter so each document stays a searchable size.

A file is all or nothing: any problem raises :class:`UploadError` and none of
its documents are added.
"""

from __future__ import annotations

import json
import os
import re

from src.internal.document_index.chunking import _split_text

ALLOWED_EXTENSIONS = (".jsonl", ".txt", ".md")
#: Words per passage, and words repeated between neighbouring passages.
PASSAGE_WORDS = 300
PASSAGE_OVERLAP = 30

_HEADING = re.compile(r"^#{1,6}\s+(.+?)\s*#*\s*$", re.MULTILINE)


class UploadError(ValueError):
    """The file cannot become documents; the message says why."""


def extension(filename: str) -> str:
    return os.path.splitext(filename)[1].lower()


def parse_upload(filename: str, data: bytes) -> list[dict]:
    ext = extension(filename)
    if ext not in ALLOWED_EXTENSIONS:
        raise UploadError(
            f"Unsupported file type {ext or '(none)'}; use .jsonl, .txt or .md."
        )
    try:
        text = data.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise UploadError("The file is not UTF-8 text.") from exc
    documents = _parse_jsonl(text) if ext == ".jsonl" else _parse_text(filename, text)
    if not documents:
        raise UploadError("The file has no documents.")
    return documents


def _parse_jsonl(text: str) -> list[dict]:
    documents = []
    for number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise UploadError(f"Line {number}: not valid JSON ({exc.msg}).") from exc
        if not isinstance(row, dict):
            raise UploadError(f"Line {number}: expected a JSON object.")
        contents = row.get("contents", row.get("text"))
        if not isinstance(contents, str) or not contents.strip():
            raise UploadError(f"Line {number}: needs non-empty contents (or text).")
        metadata = row.get("metadata")
        if metadata is None:
            metadata = {}
        elif not isinstance(metadata, dict):
            raise UploadError(f"Line {number}: metadata must be an object.")
        title = row.get("title")
        documents.append(
            {
                "id": None if row.get("id") is None else str(row["id"]),
                "title": None if title is None else str(title),
                "contents": contents,
                "metadata": metadata,
            }
        )
    return documents


def _parse_text(filename: str, text: str) -> list[dict]:
    text = text.strip()
    if not text:
        return []
    stem = os.path.splitext(os.path.basename(filename))[0] or "document"
    heading = _HEADING.search(text) if extension(filename) == ".md" else None
    title = heading.group(1).strip() if heading else stem
    if heading is not None and heading.start() == 0:
        # A leading title heading names the passages; it is not a passage.
        text = text[heading.end() :].strip()
        if not text:
            return []
    passages = _split_text(text, PASSAGE_WORDS, PASSAGE_OVERLAP)
    total = len(passages)
    return [
        {
            "id": f"{stem}-{i}",
            "title": title if total == 1 else f"{title} ({i}/{total})",
            "contents": passage,
            "metadata": {"filename": filename},
        }
        for i, passage in enumerate(passages, start=1)
    ]
