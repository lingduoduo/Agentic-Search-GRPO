import pytest

from src.internal.memory.categories import (
    CATEGORIES,
    classify_memory,
    normalize_category,
)


@pytest.mark.parametrize(
    "text,expected",
    [
        ("User is allergic to peanuts", "constraints"),
        ("Never suggest paywalled sources", "constraints"),
        ("Prefers peer-reviewed papers", "preferences"),
        ("Decided to go with FAISS over Annoy", "decisions"),
        ("Is researching hybrid retrieval for a thesis project", "projects"),
        ("I'm a data engineer living in Berlin", "profile"),
        ("The sky was cloudy", "general"),
    ],
)
def test_classify_memory(text, expected):
    assert classify_memory(text) == expected


def test_normalize_category():
    assert normalize_category("Preferences ") == "preferences"
    assert normalize_category("bogus") == "general"
    assert normalize_category(None) == "general"
    assert set(CATEGORIES) == {
        "constraints",
        "preferences",
        "profile",
        "projects",
        "decisions",
        "general",
    }
