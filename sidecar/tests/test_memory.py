from __future__ import annotations

from pathlib import Path
import sqlite3
import tempfile
import unittest

from oriens.memory import (
    MemoryCandidate,
    MemoryUnavailableError,
    NullMemoryStore,
    SQLiteMemoryStore,
    _MIGRATIONS,
    extract_explicit_candidates,
)


class NullMemoryStoreTests(unittest.TestCase):
    def test_null_memory_never_creates_or_retains_long_term_data(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            memory_dir = Path(directory) / "memory"
            store = NullMemoryStore()
            store.begin_session("session")
            self.assertEqual(store.recall("玩家偏好" ).items, ())
            store.submit_candidates((MemoryCandidate("喜欢某角色", "test"),))
            store.end_session("session")
            store.set_enabled(True)
            self.assertFalse(store.enabled)
            self.assertEqual(store.list_items(), ())
            self.assertFalse(store.delete("anything"))
            self.assertEqual(store.clear_all(), 0)
            store.close()
            self.assertFalse(memory_dir.exists())


class SQLiteMemoryStoreTests(unittest.TestCase):
    def test_schema_initialization_is_repeatable_and_uses_injected_directory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            memory_dir = Path(directory) / "isolated" / "memory"
            first = SQLiteMemoryStore(memory_dir)
            self.assertEqual(first.database_path.parent, memory_dir.resolve())
            first.close()
            second = SQLiteMemoryStore(memory_dir)
            self.assertEqual(second._connection.execute("PRAGMA foreign_keys").fetchone()[0], 1)
            connection = sqlite3.connect(second.database_path)
            try:
                self.assertEqual(
                    connection.execute("SELECT MAX(version) FROM schema_version").fetchone()[0],
                    len(_MIGRATIONS),
                )
            finally:
                connection.close()
            second.close()

    def test_crud_sources_disable_delete_and_clear(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = SQLiteMemoryStore(Path(directory) / "memory")
            item = store.add(MemoryCandidate(
                "称呼偏好：小明", "玩家明确表达", kind="profile",
                evidence="请叫我小明", source_session_id="session-1", source_run_id="run-1",
                topic_key="profile:preferred_name",
            ))
            self.assertEqual(item.evidence[0].excerpt, "请叫我小明")
            self.assertEqual(item.source_session_id, "session-1")
            corrected = store.update(item.id, content="称呼偏好：小林")
            self.assertEqual(corrected.content, "称呼偏好：小林")
            self.assertGreaterEqual(len(corrected.evidence), 2)
            self.assertTrue(store.set_item_enabled(item.id, False))
            self.assertEqual(store.recall("怎么打这个 Boss").items, ())
            self.assertTrue(store.set_item_enabled(item.id, True))
            self.assertEqual(store.recall("怎么打这个 Boss").items[0].id, item.id)
            self.assertTrue(store.delete(item.id))
            self.assertEqual(store.list_items(), ())
            one = store.add(MemoryCandidate("喜欢以撒", "手动", topic_key="likes:character"))
            two = store.add(MemoryCandidate("提示偏好：解释简短", "手动", kind="guidance_preference"))
            self.assertEqual(store.clear_all(), 2)
            self.assertEqual(store.list_items(), ())
            self.assertEqual(store.list_items(include_deleted=True), ())
            store.close()

    def test_dedup_conflict_and_low_confidence_pending(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = SQLiteMemoryStore(Path(directory) / "memory")
            first = store.add(MemoryCandidate(
                "解释深度偏好：详细", "玩家明确表达", kind="guidance_preference",
                topic_key="guidance:explanation_depth",
            ))
            duplicate = store.add(MemoryCandidate(
                "解释深度偏好：详细", "再次确认", kind="guidance_preference",
                topic_key="guidance:explanation_depth",
            ))
            self.assertEqual(duplicate.id, first.id)
            replacement = store.add(MemoryCandidate(
                "解释深度偏好：简短", "玩家最新纠正", kind="guidance_preference",
                topic_key="guidance:explanation_depth",
            ))
            statuses = {item.id: item.status for item in store.list_items()}
            self.assertEqual(statuses[first.id], "conflicted")
            self.assertEqual(statuses[replacement.id], "active")
            pending = store.add(MemoryCandidate(
                "可能偏好高风险路线", "多局行为推断", confidence=0.6,
                confirmation_level="inferred", topic_key="risk:tendency",
            ))
            self.assertEqual(pending.status, "pending")
            self.assertTrue(store.set_item_enabled(pending.id, True))
            confirmed = next(item for item in store.list_items() if item.id == pending.id)
            self.assertEqual((confirmed.status, confirmed.confirmation_level), ("active", "confirmed"))
            store.set_item_enabled(pending.id, False)
            recalled = store.recall("请深入解释路线", max_items=1, max_chars=30)
            self.assertEqual(tuple(item.id for item in recalled.items), (replacement.id,))
            self.assertLessEqual(recalled.total_chars, 30)
            store.close()

    def test_sensitive_or_unbounded_content_is_not_saved(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = SQLiteMemoryStore(Path(directory) / "memory")
            stored = store.submit_candidates((
                MemoryCandidate("Authorization: Bearer secret", "聊天"),
                MemoryCandidate("A" * 241, "聊天"),
                MemoryCandidate("喜欢犹大", "玩家明确表达"),
            ))
            self.assertEqual(tuple(item.content for item in stored), ("喜欢犹大",))
            self.assertEqual(len(store.list_items()), 1)
            store.close()

    def test_failed_migration_rolls_back_without_damaging_existing_data(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            memory_dir = Path(directory) / "memory"
            store = SQLiteMemoryStore(memory_dir)
            original = store.add(MemoryCandidate("喜欢伊甸", "玩家明确表达"))
            store.close()
            broken = _MIGRATIONS + (("CREATE TABLE migration_probe(value TEXT)", "NOT VALID SQL"),)
            with self.assertRaises(MemoryUnavailableError):
                SQLiteMemoryStore(memory_dir, migrations=broken)
            reopened = SQLiteMemoryStore(memory_dir)
            self.assertEqual(reopened.list_items()[0].id, original.id)
            connection = sqlite3.connect(reopened.database_path)
            try:
                probe = connection.execute(
                    "SELECT 1 FROM sqlite_master WHERE name = 'migration_probe'"
                ).fetchone()
                self.assertIsNone(probe)
            finally:
                connection.close()
            reopened.close()

    def test_corrupt_or_incomplete_database_is_rejected_without_rebuild(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            memory_dir = Path(directory) / "memory"
            memory_dir.mkdir()
            database = memory_dir / "memory.sqlite3"
            database.write_bytes(b"not a sqlite database")
            with self.assertRaises(MemoryUnavailableError):
                SQLiteMemoryStore(memory_dir)
            self.assertEqual(database.read_bytes(), b"not a sqlite database")

    def test_close_is_idempotent_and_releases_database_write_lock(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = SQLiteMemoryStore(Path(directory) / "memory")
            path = store.database_path
            store.close()
            store.close()
            connection = sqlite3.connect(path, timeout=0.2)
            try:
                connection.execute("BEGIN IMMEDIATE")
                connection.execute("ROLLBACK")
            finally:
                connection.close()

    def test_explicit_extraction_is_small_deterministic_and_traceable(self) -> None:
        candidates = extract_explicit_candidates(
            "以后请叫我小林，解释详细一点", session_id="session", run_id="run"
        )
        self.assertEqual([item.kind for item in candidates], ["profile", "guidance_preference"])
        self.assertTrue(all(item.source_session_id == "session" for item in candidates))
        self.assertEqual(extract_explicit_candidates("Authorization: Bearer secret"), ())


if __name__ == "__main__":
    unittest.main()
