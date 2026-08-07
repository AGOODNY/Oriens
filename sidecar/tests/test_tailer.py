from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from oriens.tailer import LogTailer


class LogTailerTests(unittest.TestCase):
    def test_reads_appended_complete_lines_only(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "log.txt"
            path.write_text("old\n", encoding="utf-8")
            tailer = LogTailer(path)
            self.assertEqual(tailer.poll().lines, ())
            with path.open("a", encoding="utf-8") as target:
                target.write("new\npartial")
            self.assertEqual(tailer.poll().lines, ("new\n",))
            with path.open("a", encoding="utf-8") as target:
                target.write("-done\n")
            self.assertEqual(tailer.poll().lines, ("partial-done\n",))
            tailer.close()

    def test_recovers_after_truncate(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "log.txt"
            path.write_text("before\n", encoding="utf-8")
            tailer = LogTailer(path, from_start=True)
            self.assertEqual(tailer.poll().lines, ("before\n",))
            path.write_text("after\n", encoding="utf-8")
            poll = tailer.poll()
            self.assertTrue(poll.reopened)
            self.assertEqual(poll.lines, ("after\n",))
            tailer.close()


if __name__ == "__main__":
    unittest.main()

