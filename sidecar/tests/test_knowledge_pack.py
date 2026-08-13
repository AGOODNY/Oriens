from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from oriens.knowledge_pack import KnowledgePackError, KnowledgePackManager


def _write_pack(root: Path, *, full: bool = False) -> None:
    root.mkdir(parents=True)
    contents = {
        "entities.jsonl": b'{"id":"collectible:1"}\n',
        "keyword.sqlite": b"sqlite-test",
    }
    if full:
        contents["vectors.faiss"] = b"faiss-test"
    roles = {
        "entities.jsonl": "entities",
        "keyword.sqlite": "keyword_index",
        "vectors.faiss": "vector_index",
    }
    files = []
    for name, content in contents.items():
        (root / name).write_bytes(content)
        files.append({
            "path": name,
            "size": len(content),
            "sha256": hashlib.sha256(content).hexdigest(),
            "required": True,
            "role": roles[name],
        })
    manifest = {
        "schema_version": 1,
        "pack_id": "isaac-test",
        "display_name": "测试知识包",
        "game": "The Binding of Isaac: Repentance+",
        "game_version": "test",
        "content_version": "1.0.0",
        "created_at": "2026-08-13T00:00:00Z",
        "files": files,
        "vector_dimension": 1024,
        "embedding_model_id": "BAAI/bge-m3",
        "minimum_oriens_version": "0.1.0",
    }
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


class KnowledgePackTests(unittest.TestCase):
    def test_valid_light_and_full_pack_capabilities(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            light = base / "light"
            full = base / "full"
            _write_pack(light)
            _write_pack(full, full=True)
            manager = KnowledgePackManager(base / "installed")
            self.assertEqual(manager.validate(light).capabilities, frozenset({"keyword"}))
            self.assertEqual(manager.validate(full).capabilities, frozenset({"keyword", "vector"}))

    def test_missing_optional_vector_file_is_a_light_pack(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            pack = Path(directory) / "pack"
            _write_pack(pack)
            manifest_path = pack / "manifest.json"
            raw = json.loads(manifest_path.read_text(encoding="utf-8"))
            raw["files"].append({
                "path": "optional.faiss",
                "size": 10,
                "sha256": hashlib.sha256(b"not-present").hexdigest(),
                "required": False,
                "role": "vector_index",
            })
            manifest_path.write_text(json.dumps(raw), encoding="utf-8")
            installed = KnowledgePackManager(Path(directory) / "installed").validate(pack)
            self.assertEqual(installed.capabilities, frozenset({"keyword"}))
            self.assertIsNone(installed.file_for("vector_index"))

    def test_missing_size_hash_and_manifest_errors_are_rejected(self) -> None:
        mutations = ("missing", "size", "hash", "manifest")
        for mutation in mutations:
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as directory:
                pack = Path(directory) / "pack"
                _write_pack(pack)
                if mutation == "missing":
                    raw = json.loads((pack / "manifest.json").read_text(encoding="utf-8"))
                    raw["files"].append({
                        "path": "missing.sqlite",
                        "size": 1,
                        "sha256": hashlib.sha256(b"x").hexdigest(),
                        "required": True,
                        "role": "optional_index",
                    })
                    (pack / "manifest.json").write_text(json.dumps(raw), encoding="utf-8")
                elif mutation == "size":
                    (pack / "keyword.sqlite").write_bytes(b"wrong-size-and-content")
                elif mutation == "hash":
                    (pack / "keyword.sqlite").write_bytes(b"sqlite-tast")
                else:
                    (pack / "manifest.json").write_text("[]", encoding="utf-8")
                with self.assertRaises(KnowledgePackError):
                    KnowledgePackManager(Path(directory) / "installed").validate(pack)

    def test_path_traversal_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            pack = Path(directory) / "pack"
            _write_pack(pack)
            manifest_path = pack / "manifest.json"
            raw = json.loads(manifest_path.read_text(encoding="utf-8"))
            raw["files"][0]["path"] = "../outside.jsonl"
            manifest_path.write_text(json.dumps(raw), encoding="utf-8")
            with self.assertRaisesRegex(KnowledgePackError, "不安全"):
                KnowledgePackManager(Path(directory) / "installed").validate(pack)

    def test_pack_requiring_newer_oriens_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            pack = Path(directory) / "pack"
            _write_pack(pack)
            manifest_path = pack / "manifest.json"
            raw = json.loads(manifest_path.read_text(encoding="utf-8"))
            raw["minimum_oriens_version"] = "99.0.0"
            manifest_path.write_text(json.dumps(raw), encoding="utf-8")
            with self.assertRaisesRegex(KnowledgePackError, "更新版本"):
                KnowledgePackManager(Path(directory) / "installed").validate(pack)

    def test_atomic_install_failure_does_not_change_current_selection(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            installed = base / "installed"
            current_source = base / "current-source"
            _write_pack(current_source)
            manager = KnowledgePackManager(installed)
            manager.install(current_source, select=True)
            self.assertEqual(manager.current().manifest.pack_id, "isaac-test")

            replacement = base / "replacement"
            _write_pack(replacement, full=True)
            raw = json.loads((replacement / "manifest.json").read_text(encoding="utf-8"))
            raw["pack_id"] = "replacement"
            (replacement / "manifest.json").write_text(json.dumps(raw), encoding="utf-8")
            with patch("oriens.knowledge_pack.os.replace", side_effect=OSError("simulated")):
                with self.assertRaises(KnowledgePackError):
                    manager.install(replacement, select=True)
            self.assertEqual(manager.current().manifest.pack_id, "isaac-test")
            self.assertFalse((installed / "replacement").exists())


if __name__ == "__main__":
    unittest.main()
