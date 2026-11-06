import sqlite3

from src.internal.db.store import AgenticSearchStore


def test_add_classifies_and_marks_dirty():
    s = AgenticSearchStore(":memory:")
    r = s.add_user_memory("u1", "User is allergic to peanuts")
    assert r.category == "constraints"
    dirty = {n.node for n in s.get_memory_index_nodes("u1") if n.dirty}
    assert dirty == {"constraints", "__root__"}


def test_explicit_category_and_filter():
    s = AgenticSearchStore(":memory:")
    s.add_user_memory("u1", "likes tea", category="profile")
    assert [
        r.category for r in s.get_user_memory_records("u1", category="profile")
    ] == ["profile"]
    assert s.get_user_memory_records("u1", category="preferences") == []


def test_update_keeps_category_and_clears_embedding():
    s = AgenticSearchStore(":memory:")
    r = s.add_user_memory("u1", "Prefers arXiv")
    s.set_memory_embeddings("u1", [(r.id, b"\x00" * 8, "m1")])
    assert s.get_memory_embeddings("u1") == {r.id: (b"\x00" * 8, "m1")}
    s.update_user_memory("u1", r.id, "Prefers arXiv and ACL anthology")
    assert s.get_user_memory_records("u1")[0].category == "preferences"
    assert s.get_memory_embeddings("u1") == {}


def test_embeddings_and_nodes_scoped_by_user():
    s = AgenticSearchStore(":memory:")
    r = s.add_user_memory("u1", "x")
    s.set_memory_embeddings("u2", [(r.id, b"\x00" * 8, "m1")])  # wrong user: ignored
    assert s.get_memory_embeddings("u1") == {}
    assert s.get_memory_index_nodes("u2") == []


def test_upsert_index_node_clears_dirty():
    s = AgenticSearchStore(":memory:")
    s.add_user_memory("u1", "User is allergic to peanuts")
    s.upsert_memory_index_node("u1", "constraints", summary="allergies", memory_count=1)
    node = {n.node: n for n in s.get_memory_index_nodes("u1")}["constraints"]
    assert (node.summary, node.memory_count, node.dirty) == ("allergies", 1, False)


def test_set_category_delete_and_dirty_marking():
    s = AgenticSearchStore(":memory:")
    r = s.add_user_memory("u1", "The sky was cloudy")
    s.upsert_memory_index_node("u1", "general", summary="", memory_count=1)
    s.upsert_memory_index_node("u1", "__root__", summary="", memory_count=1)
    assert s.set_user_memory_category("u1", r.id, "projects") is True
    assert s.set_user_memory_category("u2", r.id, "profile") is False
    assert s.get_user_memory_records("u1")[0].category == "projects"
    dirty = {n.node for n in s.get_memory_index_nodes("u1") if n.dirty}
    assert {"general", "projects", "__root__"} <= dirty
    s.upsert_memory_index_node("u1", "projects", summary="", memory_count=1)
    assert s.delete_user_memory("u1", r.id) is True
    dirty = {n.node for n in s.get_memory_index_nodes("u1") if n.dirty}
    assert {"projects", "__root__"} <= dirty


def test_migrates_legacy_db_and_backfills_categories(tmp_path):
    path = tmp_path / "legacy.db"
    conn = sqlite3.connect(path)
    conn.executescript(
        """CREATE TABLE user_memories (id TEXT PRIMARY KEY, user_id TEXT NOT NULL,
        memory_text TEXT NOT NULL, metadata_json TEXT NOT NULL DEFAULT '{}', is_active INTEGER NOT NULL DEFAULT 1,
        created_at TEXT NOT NULL, updated_at TEXT NOT NULL);
        INSERT INTO user_memories VALUES ('mem_1','u1','User is allergic to peanuts','{}',1,'t','t');"""
    )
    conn.commit()
    conn.close()
    s = AgenticSearchStore(str(path))
    assert s.get_user_memory_records("u1")[0].category == "constraints"
    s2 = AgenticSearchStore(str(path))
    assert s2.get_user_memory_records("u1")[0].category == "constraints"


def test_explicit_general_survives_reopen(tmp_path):
    path = str(tmp_path / "m.db")
    s = AgenticSearchStore(path)
    s.add_user_memory("u1", "User is allergic to peanuts", category="general")
    s._conn.close()
    s2 = AgenticSearchStore(path)
    assert s2.get_user_memory_records("u1")[0].category == "general"
