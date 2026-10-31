import copy

from src.internal.document_index.hierarchy_descriptions import (
    assess_description,
    select_description_samples,
)


def test_selection_covers_language_and_semantic_extreme_without_mutation():
    docs = [
        dict(id=str(i), title="Common topic", contents="", metadata={})
        for i in range(8)
    ]
    docs.append(
        dict(id="z", title="少数语言主题", contents="", metadata={"language": "zh"})
    )
    before = copy.deepcopy(docs)
    vectors = {d["id"]: [1, 0] for d in docs}
    vectors["z"] = [0, 1]
    selected = select_description_samples(docs, vectors=vectors, limit=3)
    assert "z" in {s["document_id"] for s in selected}
    assert any("typical" in s["roles"] for s in selected)
    assert docs == before


def test_quality_flags_are_warning_proxies():
    assert assess_description("  ")["weak"]
    assert assess_description("This group contains various documents.")["weak"]
    assert assess_description("本组包含各种文档。")["weak"]
    assert not assess_description(
        "Linux kernel memory allocation and debugging procedures"
    )["weak"]


def test_lexical_selection_has_honest_roles_and_reports_limited_language_coverage():
    docs = [
        dict(id="a", title="English guide", metadata={"lang": "en"}),
        dict(id="b", title="中文教程", metadata={"lang": "zh"}),
    ]
    samples = select_description_samples(docs, limit=1)
    assert "lexical_proxy" in samples[0]["roles"]
    assert "coverage_limited:language" in samples[0]["roles"]


def test_selector_rejects_nonpositive_limit():
    import pytest

    with pytest.raises(ValueError):
        select_description_samples([], limit=0)


def test_japanese_with_kanji_is_not_merged_into_han_stratum():
    from src.internal.document_index.hierarchy_descriptions import _language

    assert _language(dict(title="東京の天気", contents="今日は晴れです")) == "kana"
    assert _language(dict(title="北京天气", contents="今天晴")) == "han"
