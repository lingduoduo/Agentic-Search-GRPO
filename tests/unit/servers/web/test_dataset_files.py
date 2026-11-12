"""Uploaded files -> corpus documents."""

from __future__ import annotations

import json

import pytest

from src.internal.servers.web.dataset_files import UploadError, parse_upload


def _jsonl(*rows):
    return "\n".join(json.dumps(r) for r in rows).encode()


def test_jsonl_lines_become_documents_as_written():
    docs = parse_upload(
        "corpus.JSONL",
        _jsonl(
            {
                "id": 7,
                "title": "BM25",
                "contents": "Sparse.",
                "metadata": {"acl": ["a"]},
            },
            {"text": "No id or title."},
        )
        + b"\n\n",
    )

    assert docs == [
        {"id": "7", "title": "BM25", "contents": "Sparse.", "metadata": {"acl": ["a"]}},
        {"id": None, "title": None, "contents": "No id or title.", "metadata": {}},
    ]


@pytest.mark.parametrize(
    ("data", "message"),
    [
        (b'{"contents": "ok"}\n{not json', "Line 2: not valid JSON"),
        (b'["a list"]', "Line 1: expected a JSON object"),
        (b'{"title": "no contents"}', "Line 1: needs non-empty contents"),
        (b'{"contents": "  "}', "Line 1: needs non-empty contents"),
        (b'{"contents": "x", "metadata": []}', "Line 1: metadata must be an object"),
        (b"\n\n", "The file has no documents."),
    ],
)
def test_a_bad_jsonl_file_is_rejected_whole(data, message):
    with pytest.raises(UploadError, match=message):
        parse_upload("bad.jsonl", data)


def test_a_short_text_file_is_one_document_titled_by_its_name():
    [doc] = parse_upload("notes/team-notes.txt", "Hello there.\n".encode())

    assert doc == {
        "id": "team-notes-1",
        "title": "team-notes",
        "contents": "Hello there.",
        "metadata": {"filename": "notes/team-notes.txt"},
    }


def test_a_long_markdown_file_is_split_into_titled_passages():
    body = " ".join(f"word{i}" for i in range(700))
    docs = parse_upload("guide.md", f"# Retrieval Guide\n\n{body}".encode())

    assert len(docs) == 3
    assert [d["title"] for d in docs] == [
        "Retrieval Guide (1/3)",
        "Retrieval Guide (2/3)",
        "Retrieval Guide (3/3)",
    ]
    assert [d["id"] for d in docs] == ["guide-1", "guide-2", "guide-3"]
    assert "word699" in docs[-1]["contents"]


def test_a_bom_is_ignored():
    [doc] = parse_upload("a.txt", "﻿Text.".encode())
    assert doc["contents"] == "Text."


@pytest.mark.parametrize(
    ("filename", "data", "message"),
    [
        ("a.pdf", b"%PDF", "Unsupported file type .pdf"),
        ("noext", b"x", r"Unsupported file type \(none\)"),
        ("a.txt", b"\xff\xfe\x00bad", "not UTF-8"),
        ("a.md", b"   \n", "no documents"),
    ],
)
def test_unusable_files_are_rejected(filename, data, message):
    with pytest.raises(UploadError, match=message):
        parse_upload(filename, data)


def test_a_markdown_heading_alone_has_no_documents():
    with pytest.raises(UploadError, match="no documents"):
        parse_upload("empty.md", b"# Just a title\n")


def test_a_later_heading_stays_in_the_text():
    [doc] = parse_upload("a.md", b"Intro.\n\n## Details\n\nMore.")
    assert doc["title"] == "Details"
    assert "## Details" in doc["contents"]
