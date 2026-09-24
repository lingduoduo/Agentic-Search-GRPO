from src.context import ChatMessage
from src.internal.search.context import Resolution, resolve_follow_up
from src.internal.utils.embedding_gate import follow_up_cos_min


def _history(*user_texts: str) -> list[ChatMessage]:
    messages: list[ChatMessage] = []
    for text in user_texts:
        messages.append(ChatMessage(role="user", content=text))
        messages.append(ChatMessage(role="assistant", content="Here is what I found."))
    return messages


def _no_cosine(message, topic):
    raise AssertionError("cosine must not be consulted")


def test_first_turn_is_standalone():
    assert resolve_follow_up(
        "What is FAISS?", [], cosine=_no_cosine, tau=0.8
    ) == Resolution("What is FAISS?", False, "no_history")


def test_reference_word_makes_a_continuation():
    got = resolve_follow_up(
        "Which index types does it support?",
        _history("What is FAISS?"),
        cosine=_no_cosine,
        tau=0.8,
    )
    assert got == Resolution(
        "What is FAISS?\nWhich index types does it support?", True, "reference"
    )


def test_reference_words_match_whole_words_only():
    # "item" and "Italy" contain "it" but are not references.
    got = resolve_follow_up(
        "Which item ships fastest to Italy by default?",
        _history("What is FAISS?"),
        cosine=None,
        tau=0.8,
    )
    assert got.reason == "switch"


def test_short_fragment_makes_a_continuation():
    got = resolve_follow_up(
        "And cardiac muscle?",
        _history("Tirasemtiv targets fast-twitch muscle."),
        cosine=_no_cosine,
        tau=0.8,
    )
    assert got.continuation and got.reason == "fragment"


def test_cue_word_alone_does_not_make_a_continuation():
    got = resolve_follow_up(
        "Also, what's the weather in Oslo right now?",
        _history("How does dense retrieval with FAISS work?"),
        cosine=lambda message, topic: 0.1,
        tau=0.8,
    )
    assert got == Resolution(
        "Also, what's the weather in Oslo right now?", False, "switch"
    )


def test_semantic_similarity_makes_a_continuation():
    got = resolve_follow_up(
        "Which compounds activate the dephosphorylated form?",
        _history("AMPK activation reduces fibrosis in the lungs."),
        cosine=lambda message, topic: 0.9,
        tau=0.8,
    )
    assert got.continuation and got.reason == "semantic"


def test_below_threshold_is_a_switch():
    got = resolve_follow_up(
        "Which compounds activate the dephosphorylated form?",
        _history("AMPK activation reduces fibrosis in the lungs."),
        cosine=lambda message, topic: 0.79,
        tau=0.8,
    )
    assert got.reason == "switch"


def test_missing_embedder_or_failed_cosine_skips_semantic():
    for cosine in (None, lambda message, topic: None):
        got = resolve_follow_up(
            "Which compounds activate the dephosphorylated form?",
            _history("AMPK activation reduces fibrosis in the lungs."),
            cosine=cosine,
            tau=0.8,
        )
        assert got.reason == "switch"


def test_topic_skips_earlier_continuations():
    got = resolve_follow_up(
        "Berlin?",
        _history("What's the weather in Tokyo?", "And in Paris?"),
        cosine=None,
        tau=0.8,
    )
    assert got.query == "What's the weather in Tokyo?\nBerlin?"


def test_topic_moves_to_the_latest_switch():
    got = resolve_follow_up(
        "Is it open late?",
        _history(
            "What is FAISS?", "Find coffee shops near Union Square, San Francisco."
        ),
        cosine=lambda message, topic: 0.1,
        tau=0.8,
    )
    assert (
        got.query
        == "Find coffee shops near Union Square, San Francisco.\nIs it open late?"
    )


def test_switch_after_a_continuation_passes_through():
    got = resolve_follow_up(
        "Write a haiku about autumn leaves falling slowly.",
        _history("What is RAG?", "Where does reranking fit in that pipeline?"),
        cosine=lambda message, topic: 0.2,
        tau=0.8,
    )
    assert got == Resolution(
        "Write a haiku about autumn leaves falling slowly.", False, "switch"
    )


def test_topic_looks_back_at_most_three_user_turns():
    history = _history(
        "What is FAISS?",
        "Does it use GPUs?",
        "Is it fast?",
        "What about its memory use?",
    )
    got = resolve_follow_up("Is it open source?", history, cosine=None, tau=0.8)
    # Every turn in the 3-turn window continues "What is FAISS?", which has left
    # it. No standalone topic is in reach, so the message is searched as typed
    # rather than glued to another follow-up.
    assert got == Resolution("Is it open source?", False, "no_topic")


def test_oldest_window_turn_is_the_topic_only_if_it_stands_alone():
    history = _history(
        "What is FAISS?",
        "Explain the history of the Roman empire in some depth please.",
        "Is it long?",
        "And its causes?",
    )
    got = resolve_follow_up("Why did it fall?", history, cosine=None, tau=0.8)
    # The Roman-empire turn is classified against the turn before the window
    # (FAISS) and stands alone, so it is the topic.
    assert got.query == (
        "Explain the history of the Roman empire in some depth please.\n"
        "Why did it fall?"
    )


def test_assistant_and_tool_markup_messages_are_ignored():
    history = [
        ChatMessage(role="user", content="What is FAISS?"),
        ChatMessage(
            role="assistant",
            content="<tool_call>search</tool_call> Also, how about BM25?",
        ),
    ]
    got = resolve_follow_up("Does it use GPUs?", history, cosine=None, tau=0.8)
    assert got.query == "What is FAISS?\nDoes it use GPUs?"


def test_follow_up_cos_min_reads_the_environment(monkeypatch):
    monkeypatch.setenv("AGENTIC_SEARCH_FOLLOW_UP_COS_MIN", "0.9")
    assert follow_up_cos_min() == 0.9
    monkeypatch.delenv("AGENTIC_SEARCH_FOLLOW_UP_COS_MIN")
    assert 0.7 <= follow_up_cos_min() <= 0.95
