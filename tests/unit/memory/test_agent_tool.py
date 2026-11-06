from src.internal.db.store import AgenticSearchStore
from src.internal.memory.agent_tool import build_search_memory_tool
from src.internal.tools.base import ToolEffect


async def test_search_memory_tool_is_user_scoped():
    store = AgenticSearchStore(":memory:")
    store.add_user_memory("u1", "Prefers arXiv")
    store.add_user_memory("u2", "Prefers arXiv mirrors")
    tool = build_search_memory_tool(store, "u1")
    text, *_ = await tool.execute("i", {"query": "arXiv"})
    assert "Prefers arXiv" in text and "mirrors" not in text
    assert tool.name == "search_memory"
    assert tool.effect is ToolEffect.READ_ONLY
    assert tool.citeable is False
    store.close()


async def test_search_memory_tool_category_filter():
    store = AgenticSearchStore(":memory:")
    store.add_user_memory("u1", "Prefers arXiv")
    tool = build_search_memory_tool(store, "u1")
    text, *_ = await tool.execute("i", {"query": "arXiv", "category": "constraints"})
    assert text == "no matching memories"
    text, *_ = await tool.execute("i", {"query": "arXiv", "category": "preferences"})
    assert text.startswith("[preferences] ")
    store.close()
