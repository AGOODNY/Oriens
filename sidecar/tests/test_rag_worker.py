from __future__ import annotations

from array import array
from pathlib import Path
import sqlite3
import tempfile
import unittest


class RagWorkerTests(unittest.TestCase):
    def test_sqlite_vec_build_table_can_be_copied_and_queried(self) -> None:
        try:
            import sqlite_vec
        except ImportError:
            self.skipTest("sqlite-vec 未安装")
        db = sqlite3.connect(":memory:")
        db.enable_load_extension(True)
        sqlite_vec.load(db)

        db.execute(
            "CREATE VIRTUAL TABLE chunk_vectors_build "
            "USING vec0(embedding float[3] distance_metric=cosine)"
        )
        blob = array("f", (1.0, 0.0, 0.0)).tobytes()
        db.execute(
            "INSERT INTO chunk_vectors_build(rowid, embedding) VALUES(?,?)", (1, blob)
        )
        db.execute(
            "CREATE VIRTUAL TABLE chunk_vectors "
            "USING vec0(embedding float[3] distance_metric=cosine)"
        )
        db.execute(
            "INSERT INTO chunk_vectors(rowid, embedding) "
            "SELECT rowid, embedding FROM chunk_vectors_build"
        )
        db.execute("DROP TABLE chunk_vectors_build")

        self.assertEqual(
            db.execute(
                "SELECT rowid, distance FROM chunk_vectors "
                "WHERE embedding MATCH ? AND k=1 ORDER BY distance",
                (blob,),
            ).fetchone(),
            (1, 0.0),
        )

    def test_faiss_export_reuses_published_sqlite_vectors(self) -> None:
        try:
            import faiss
            import sqlite_vec
        except ImportError:
            self.skipTest("FAISS 或 sqlite-vec 未安装")
        from oriens.rag_worker import convert_sqlite_vectors_to_faiss

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sqlite_path = root / "vectors.sqlite"
            faiss_path = root / "vectors.faiss"
            db = sqlite3.connect(sqlite_path)
            db.enable_load_extension(True)
            sqlite_vec.load(db)
            db.execute("CREATE TABLE metadata(key TEXT PRIMARY KEY, value TEXT NOT NULL)")
            db.executemany(
                "INSERT INTO metadata VALUES(?,?)",
                (("chunk_count", "2"), ("vector_count", "2"), ("vector_backend", "sqlite-vec")),
            )
            db.execute(
                "CREATE VIRTUAL TABLE chunk_vectors "
                "USING vec0(embedding float[3] distance_metric=cosine)"
            )
            db.execute(
                "CREATE TABLE vector_chunk_map(rowid INTEGER PRIMARY KEY, chunk_id TEXT UNIQUE NOT NULL)"
            )
            for rowid, chunk_id, vector in (
                (1, "chunk:a", (1.0, 0.0, 0.0)),
                (2, "chunk:b", (0.0, 1.0, 0.0)),
            ):
                db.execute("INSERT INTO vector_chunk_map VALUES(?,?)", (rowid, chunk_id))
                db.execute(
                    "INSERT INTO chunk_vectors(rowid, embedding) VALUES(?,?)",
                    (rowid, array("f", vector).tobytes()),
                )
            db.commit()
            db.close()

            report = convert_sqlite_vectors_to_faiss(
                sqlite_path, faiss_path, dimension=3, batch_size=1
            )

            self.assertEqual(report["vector_count"], 2)
            self.assertEqual(faiss.read_index(str(faiss_path)).ntotal, 2)
            self.assertTrue(faiss_path.with_suffix(".faiss.ids.json").is_file())
