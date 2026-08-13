from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from oriens.memory import MemoryCandidate, NullMemoryStore


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
            store.close()
            self.assertFalse(memory_dir.exists())


if __name__ == "__main__":
    unittest.main()

