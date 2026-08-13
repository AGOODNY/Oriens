from __future__ import annotations

from pathlib import Path
import hashlib
import json
import tempfile
import unittest

from oriens.application import (
    LaunchOptions,
    ListeningState,
    OriensApplication,
    _config_for_pack,
)
from oriens.config import load_config
from oriens.knowledge_pack import KnowledgePackManager
from oriens.memory import NullMemoryStore
from oriens.paths import AppPaths


class ApplicationAssemblyTests(unittest.TestCase):
    def test_application_assembles_existing_services_and_closes_idempotently(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            application = OriensApplication.build(
                AppPaths.development(),
                LaunchOptions(
                    config_path=Path("config/rag-v2.1-faiss.toml"),
                    log_path=Path(directory) / "game.log",
                    online=False,
                    enable_vector=False,
                ),
            )
            self.assertIs(application.advice_engine.rag, application.rag)
            self.assertIsNotNone(application.query_engine)
            self.assertIsNotNone(application.router)
            self.assertIsNotNone(application.tailer)
            self.assertIsInstance(application.memory, NullMemoryStore)
            self.assertFalse(application.router.online)
            self.assertFalse(application.memory.enabled)
            application.close()
            application.close()

    def test_pause_discards_new_lines_and_resume_continues_safely(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "game.log"
            log.write_text("", encoding="utf-8")
            application = OriensApplication.build(
                AppPaths.development(),
                LaunchOptions(
                    config_path=Path("config/rag-v2.1-faiss.toml"),
                    log_path=log,
                    from_start=True,
                    online=False,
                    enable_vector=False,
                ),
            )
            event = {
                "schema_version": 1,
                "seq": 1,
                "run_id": "PAUSE TEST:0",
                "type": "room_entered",
                "game_frame": 10,
                "context": {"stage": 1, "room_index": 1, "room_spawn_seed": 1},
                "payload": {},
            }
            application.pause_listening()
            with log.open("a", encoding="utf-8") as target:
                target.write("[ORIENS_EVENT]" + json.dumps(event) + "\n")
            self.assertEqual(application.poll_events(), ())
            self.assertIsNone(application.session.state.run_id)
            application.resume_listening()
            event["seq"] = 2
            event["context"]["room_index"] = 2
            with log.open("a", encoding="utf-8") as target:
                target.write("[ORIENS_EVENT]" + json.dumps(event) + "\n")
            received = application.poll_events()
            self.assertEqual(len(received), 1)
            self.assertEqual(received[0].seq, 2)
            self.assertEqual(
                application.runtime_snapshot().listening, ListeningState.LISTENING
            )
            application.close()

    def test_selected_pack_paths_replace_manual_rag_path_assembly(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "pack"
            root.mkdir()
            contents = {
                "entities.jsonl": b"{}\n",
                "keyword.sqlite": b"keyword",
                "vectors.faiss": b"vectors",
            }
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
            (root / "manifest.json").write_text(json.dumps({
                "schema_version": 1,
                "pack_id": "selected-pack",
                "display_name": "已选知识包",
                "game": "Isaac",
                "game_version": "test-game",
                "content_version": "test-content",
                "created_at": "2026-08-13T00:00:00Z",
                "files": files,
                "vector_dimension": 1024,
                "embedding_model_id": "BAAI/bge-m3",
                "minimum_oriens_version": "0.1.0",
            }), encoding="utf-8")
            pack = KnowledgePackManager(Path(directory) / "installed").validate(root)
            paths = AppPaths.development()
            configured = _config_for_pack(load_config(), pack, paths)
            self.assertEqual(configured.rag.index_path, root / "keyword.sqlite")
            self.assertEqual(configured.rag.vector_index_path, root / "vectors.faiss")
            self.assertEqual(configured.rag.entities_path, root / "entities.jsonl")
            self.assertEqual(configured.rag.content_version, "test-content")


if __name__ == "__main__":
    unittest.main()
