from __future__ import annotations

from pathlib import Path
import hashlib
import json
import tempfile
import unittest
from unittest.mock import patch

from oriens.application import (
    GameSession,
    LaunchOptions,
    ListeningState,
    OriensApplication,
    _config_for_pack,
)
from oriens.config import load_config
from oriens.knowledge_pack import KnowledgePackManager
from oriens.memory import MemoryUnavailableError, NullMemoryStore, SQLiteMemoryStore
from oriens.paths import AppPaths
from oriens.protocol import GameEvent


class CountingMemoryStore(NullMemoryStore):
    enabled = True

    def __init__(self) -> None:
        self.begun: list[str] = []
        self.ended: list[str] = []

    def begin_session(self, session_id: str) -> None:
        self.begun.append(session_id)

    def end_session(self, session_id: str) -> None:
        self.ended.append(session_id)


class FailingLifecycleMemoryStore(NullMemoryStore):
    enabled = True

    def begin_session(self, session_id: str) -> None:
        raise MemoryUnavailableError("test begin failure")

    def end_session(self, session_id: str) -> None:
        raise MemoryUnavailableError("test end failure")


class ApplicationAssemblyTests(unittest.TestCase):
    def test_game_session_notifies_begin_and_end_once_and_close_is_idempotent(self) -> None:
        memory = CountingMemoryStore()
        session = GameSession(memory)
        context = {"stage": 1, "room_index": 1, "room_spawn_seed": 1}
        session.apply(GameEvent(1, 1, "RUN:0", "run_started", 0, context, {}))
        session.apply(GameEvent(1, 2, "RUN:0", "heartbeat", 60, context, {}))
        session.apply(GameEvent(1, 3, "RUN:0", "run_ended", 120, context, {}))
        session.close()
        session.close()
        self.assertEqual(memory.begun, ["RUN:0"])
        self.assertEqual(memory.ended, ["RUN:0"])

    def test_memory_lifecycle_failure_does_not_block_game_state(self) -> None:
        session = GameSession(FailingLifecycleMemoryStore())
        event = GameEvent(
            1, 1, "RUN:FAIL", "run_started", 0,
            {"stage": 1, "room_index": 1, "room_spawn_seed": 1}, {},
        )
        session.apply(event)
        self.assertEqual(session.state.run_id, "RUN:FAIL")
        session.close()

    def test_application_assembles_existing_services_and_closes_idempotently(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths = AppPaths.development(user_data=Path(directory) / "user")
            application = OriensApplication.build(
                paths,
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
            self.assertFalse(paths.memory_dir.exists())
            application.close()
            application.close()

    def test_pause_discards_new_lines_and_resume_continues_safely(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "game.log"
            log.write_text("", encoding="utf-8")
            application = OriensApplication.build(
                AppPaths.development(user_data=Path(directory) / "user"),
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
            paths = AppPaths.development(user_data=Path(directory) / "user")
            configured = _config_for_pack(load_config(paths=paths), pack, paths)
            self.assertEqual(configured.rag.index_path, root / "keyword.sqlite")
            self.assertEqual(configured.rag.vector_index_path, root / "vectors.faiss")
            self.assertEqual(configured.rag.entities_path, root / "entities.jsonl")
            self.assertEqual(configured.rag.content_version, "test-content")

    def test_enabled_memory_uses_one_store_only_in_injected_temp_path(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = AppPaths.development(user_data=root / "user")
            explicit = root / "memory-enabled.toml"
            explicit.write_text("[memory]\nenabled = true\n", encoding="utf-8")
            with patch("oriens.application.SQLiteMemoryStore", wraps=SQLiteMemoryStore) as factory:
                application = OriensApplication.build(
                    paths,
                    LaunchOptions(
                        config_path=explicit,
                        log_path=root / "game.log",
                        online=False,
                        enable_vector=False,
                    ),
                )
            self.assertEqual(factory.call_count, 1)
            self.assertIsInstance(application.memory, SQLiteMemoryStore)
            self.assertEqual(application.memory.database_path.parent, paths.memory_dir)
            self.assertIn("仅保存在本机", application.runtime_snapshot().memory_status)
            application.close()
            self.assertTrue(application.memory.closed)

    def test_memory_initialization_failure_degrades_without_blocking_text_core(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = AppPaths.development(user_data=root / "user")
            explicit = root / "memory-enabled.toml"
            explicit.write_text("[memory]\nenabled = true\n", encoding="utf-8")
            with patch(
                "oriens.application.SQLiteMemoryStore",
                side_effect=MemoryUnavailableError("test failure"),
            ):
                application = OriensApplication.build(
                    paths,
                    LaunchOptions(
                        config_path=explicit,
                        log_path=root / "game.log",
                        online=False,
                        enable_vector=False,
                    ),
                )
            self.assertIsInstance(application.memory, NullMemoryStore)
            self.assertIsNotNone(application.query_engine)
            self.assertIn("安全切换", application.runtime_snapshot().memory_status)
            application.close()

    def test_later_assembly_failure_closes_already_created_memory_store(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = AppPaths.development(user_data=root / "user")
            store = SQLiteMemoryStore(paths.memory_dir)
            with patch("oriens.application.LogTailer", side_effect=RuntimeError("test failure")):
                with self.assertRaisesRegex(RuntimeError, "test failure"):
                    OriensApplication.build(
                        paths,
                        LaunchOptions(
                            config_path=Path("config/rag-v2.1-faiss.toml"),
                            log_path=root / "game.log",
                            online=False,
                            enable_vector=False,
                        ),
                        memory=store,
                    )
            self.assertTrue(store.closed)


if __name__ == "__main__":
    unittest.main()
