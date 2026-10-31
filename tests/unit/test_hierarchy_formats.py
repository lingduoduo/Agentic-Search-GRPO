import re

import pytest

from src.internal.document_index.hierarchy_formats import (
    first_sentence,
    normalize_document,
    split_passages,
)


def doc(contents="", **kwargs):
    return {"id": "文書", "contents": contents, **kwargs}


def test_plain_unicode_and_copy():
    original = doc("你好。 世界！", metadata={"extra": ["value"]})
    result = normalize_document(original)
    assert result["sections"] == [{"heading": "Body", "text": "你好。 世界！"}]
    assert result["structure_path"] == []
    result["metadata"]["extra"].append("new")
    assert original == doc("你好。 世界！", metadata={"extra": ["value"]})


@pytest.mark.parametrize(
    "metadata,path,source",
    [
        (
            {"hierarchy_path": [" A ", "中文"], "topic": "other"},
            ["A", "中文"],
            "metadata.hierarchy_path",
        ),
        ({"topic": "T", "product": "P", "category": "C"}, ["T", "P", "C"], "metadata"),
        ({}, ["example.com", "a", "b"], "url"),
    ],
)
def test_paths(metadata, path, source):
    result = normalize_document(
        doc(metadata=metadata, url="https://example.com/a/b/file.html?q=x")
    )
    assert result["structure_path"] == path
    assert result["structure_source"] == source


@pytest.mark.parametrize(
    "patch",
    [
        {"id": ""},
        {"id": 12},
        {"contents": None},
        {"title": []},
        {"url": 1},
        {"metadata": []},
        {"metadata": {"topic": []}},
        {"metadata": {"hierarchy_path": "a"}},
        {"metadata": {"hierarchy_path": [""]}},
        {"metadata": {"format": 1}},
        {"metadata": {"content_type": []}},
        {"sections": None},
        {"sections": [{"heading": "X", "text": 1}]},
        {"metadata": {"sections": [{"text": "missing heading"}]}},
    ],
)
def test_invalid_fields(patch):
    with pytest.raises(ValueError):
        normalize_document(
            doc(**patch) if "contents" not in patch else {"id": "x", **patch}
        )


def test_explicit_sections_precede_parsing():
    sections = [{"heading": "手動", "text": "本文"}]
    assert normalize_document(doc("# Other", sections=sections))["sections"] == sections
    assert (
        normalize_document(doc("x", metadata={"sections": sections}))["sections"]
        == sections
    )


def test_html_visible_text_and_headings():
    result = normalize_document(
        doc(
            "<p>Intro</p><h1>标题</h1><p>Hello <b>world</b>.</p><script>secret</script><style>hidden</style><h2>Next</h2><p>End</p>"
        )
    )
    assert result["sections"] == [
        {"heading": "Body", "text": "Intro"},
        {"heading": "标题", "text": "Hello world."},
        {"heading": "Next", "text": "End"},
    ]
    assert "secret" not in result["contents"] and "hidden" not in result["contents"]


def test_markdown_fences_setext_and_code_angles():
    result = normalize_document(
        doc(
            "Intro\n\nTitle\n=====\nText\n```python\n# not heading\nlist<T>\n```\n## Next\nTail"
        )
    )
    assert [s["heading"] for s in result["sections"]] == ["Body", "Title", "Next"]
    assert "# not heading" in result["sections"][1]["text"]
    code = "Use vector<T> and a < b > c."
    assert normalize_document(doc(code))["contents"] == code


def test_layout_is_opt_in_and_conservative():
    text = "INTRODUCTION\n\nSome prose here.\n\nCONCLUSION\n\nFinal prose."
    assert len(normalize_document(doc(text))["sections"]) == 1
    assert [
        s["heading"]
        for s in normalize_document(doc(text), infer_layout=True)["sections"]
    ] == ["INTRODUCTION", "CONCLUSION"]
    assert (
        len(
            normalize_document(doc("Short line\nAnother line"), infer_layout=True)[
                "sections"
            ]
        )
        == 1
    )


@pytest.mark.parametrize(
    "text,expected",
    [("Hello. Next", "Hello."), ("你好！再见。", "你好！"), ("One? Two", "One?")],
)
def test_first_sentence(text, expected):
    assert first_sentence(text) == expected


def test_first_sentence_bounds_unpunctuated_text():
    assert 0 < len(first_sentence("界" * 10000)) <= 500


@pytest.mark.parametrize(
    "text", ["a bb ccc dddd", "中文字符" * 40, "x" * 100, " one\n\t two  三四五六 七 "]
)
def test_passage_preservation_and_bounds(text):
    chunks = split_passages(text, max_words=2, max_chars=7)
    assert "".join(re.sub(r"\s", "", part) for part in chunks) == re.sub(
        r"\s", "", text
    )
    assert all(len(part) <= 7 and len(part.split()) <= 2 for part in chunks)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_words": 0, "max_chars": 4},
        {"max_words": 2, "max_chars": -1},
        {"max_words": True, "max_chars": 4},
    ],
)
def test_invalid_passage_limits(kwargs):
    with pytest.raises(ValueError):
        split_passages("text", **kwargs)


def test_fenced_html_is_markdown_code_not_visible_html():
    text = "```html\n<h1>Example</h1>\n<p>literal</p>\n```\n\n# Actual\nBody"
    result = normalize_document(doc(text))
    assert [s["heading"] for s in result["sections"]] == ["Body", "Actual"]
    assert "<h1>Example</h1>" in result["contents"]


def test_html_inline_spacing_preserves_words_and_punctuation():
    result = normalize_document(doc("<p>Hello <em>world</em>.</p><p>Next.</p>"))
    assert result["sections"][0]["text"] == "Hello world. Next."


def test_existing_jsonl_text_and_nullable_defaults():
    result = normalize_document(
        {"id": "x", "text": "Legacy text", "url": None, "metadata": None}
    )
    assert result["contents"] == "Legacy text"
    assert result["url"] == "" and result["metadata"] == {}


@pytest.mark.parametrize(
    "text,kwargs,source",
    [
        ("plain", {}, "plain"),
        ("# Title\nText", {}, "markdown"),
        ("<p>Text</p>", {}, "html"),
        ("INTRODUCTION\n\nSome prose.", {"infer_layout": True}, "layout"),
    ],
)
def test_section_provenance_and_canonical_idempotence(text, kwargs, source):
    result = normalize_document(doc(text), **kwargs)
    assert result["section_source"] == source
    assert normalize_document(result, **kwargs) == result


def test_html_excludes_nonvisible_nodes_and_attributes():
    text = """<!DOCTYPE html><html><head><title>Private title</title></head><body>
    <!-- private comment --><template><h1>Private heading</h1>template secret</template>
    <div hidden><h2>Hidden heading</h2>hidden secret</div>
    <span aria-hidden="true">aria secret</span>
    <div style="display: none">display secret</div>
    <span style="visibility: hidden">visibility secret</span>
    <input value="input secret"><h1>Visible</h1><p>Public body.</p>
    </body></html>"""
    result = normalize_document(doc(text))
    assert result["sections"] == [{"heading": "Visible", "text": "Public body."}]
    assert result["contents"] == "Visible\nPublic body."


def test_passages_preserve_interior_code_and_table_whitespace():
    text = "def f():\n    return 1\n\nnext\tline"
    assert split_passages(text, max_words=20, max_chars=100) == [text]
    chunks = split_passages(text, max_words=4, max_chars=100)
    assert chunks == ["def f():\n    return 1", "next\tline"]
