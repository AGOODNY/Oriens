"""BGE-M3 独立进程 Worker 与可配置的 sqlite-vec/FAISS 后端。

模型加载、嵌入和向量查询均发生在子进程；主 UI 线程只观察就绪状态。
"""

from __future__ import annotations

from array import array
from multiprocessing import get_context
from multiprocessing.connection import Connection
import json
import os
from pathlib import Path
import sqlite3
import tempfile
import time
from typing import Any, Sequence


class VectorWorkerClient:
    def __init__(
        self,
        *,
        index_path: Path,
        model_path: Path,
        backend: str = "sqlite-vec",
        vector_index_path: Path | None = None,
        dimension: int = 1024,
        batch_size: int = 32,
        max_sequence_length: int = 8192,
        device: str = "cpu",
        build_timeout_seconds: float = 7200.0,
        request_timeout_seconds: float = 8.0,
    ) -> None:
        self._index_path = index_path.resolve()
        self._model_path = model_path.resolve()
        self._backend = backend
        self._vector_index_path = (vector_index_path or index_path).resolve()
        self._dimension = dimension
        self._batch_size = batch_size
        self._max_sequence_length = max_sequence_length
        self._device = device
        self._build_timeout_seconds = build_timeout_seconds
        self._timeout = request_timeout_seconds
        self._process: Any = None
        self._connection: Connection | None = None
        self._available = False
        self._reason: str | None = "向量 Worker 正在初始化"
        self._request_id = 0
        self._init_started: float | None = None
        self.init_latency_ms: float | None = None

    @property
    def available(self) -> bool:
        self.poll_status()
        return self._available

    @property
    def unavailable_reason(self) -> str | None:
        self.poll_status()
        return self._reason

    @property
    def pid(self) -> int | None:
        return self._process.pid if self._process is not None else None

    def start(self) -> None:
        if self._process is not None and self._process.is_alive():
            return
        context = get_context("spawn")
        parent, child = context.Pipe()
        self._connection = parent
        self._init_started = time.perf_counter()
        self._process = context.Process(
            target=_worker_main,
            args=(
                child,
                str(self._index_path),
                str(self._model_path),
                self._backend,
                str(self._vector_index_path),
                self._dimension,
                self._batch_size,
                self._max_sequence_length,
                self._device,
            ),
            name="oriens-bge-m3",
            daemon=True,
        )
        self._process.start()

    def poll_status(self) -> None:
        if self._process is None:
            self.start()
        connection = self._connection
        if connection is None:
            return
        while connection.poll(0):
            message = connection.recv()
            kind = message.get("kind")
            if kind == "ready":
                self._available = bool(message.get("vector_index_ready"))
                self._reason = None if self._available else "向量索引尚未构建"
                if self._init_started is not None:
                    self.init_latency_ms = (time.perf_counter() - self._init_started) * 1000
            elif kind == "fatal":
                self._available = False
                self._reason = str(message.get("error") or "向量 Worker 初始化失败")
            else:
                # 搜索/构建响应留给同步请求读取。
                self._pending = message
                break
        if self._process is not None and not self._process.is_alive() and self._reason is None:
            self._available = False
            self._reason = "向量 Worker 已退出"

    def wait_ready(self, timeout_seconds: float = 600.0) -> bool:
        deadline = time.monotonic() + timeout_seconds
        while time.monotonic() < deadline:
            self.poll_status()
            if self._available or self._reason != "向量 Worker 正在初始化":
                return self._available
            time.sleep(0.1)
        self._reason = "向量 Worker 初始化超时"
        return False

    def search(self, query: str, limit: int) -> Sequence[tuple[str, float]]:
        if not self.available:
            return ()
        response = self._request({"action": "search", "query": query, "limit": limit})
        if not response.get("ok"):
            raise RuntimeError("向量查询失败")
        return tuple((str(row[0]), float(row[1])) for row in response["rows"])

    def build(self, chunks: list[dict[str, Any]]) -> dict[str, Any]:
        return self._build_request({"action": "build", "chunks": chunks})

    def build_path(
        self,
        chunks_path: Path,
        *,
        progress: Any | None = None,
        timeout_seconds: float | None = None,
    ) -> dict[str, Any]:
        return self._build_request(
            {"action": "build_path", "chunks_path": str(chunks_path.resolve())},
            progress=progress,
            timeout_seconds=(
                self._build_timeout_seconds
                if timeout_seconds is None
                else timeout_seconds
            ),
        )

    def _build_request(
        self,
        payload: dict[str, Any],
        *,
        progress: Any | None = None,
        timeout_seconds: float = 1800.0,
    ) -> dict[str, Any]:
        self.start()
        # 构建前只要求模型成功加载，不要求已有向量表。
        deadline = time.monotonic() + 600
        while time.monotonic() < deadline:
            connection = self._connection
            if connection is not None and connection.poll(0.1):
                message = connection.recv()
                if message.get("kind") == "ready":
                    if self._init_started is not None:
                        self.init_latency_ms = (time.perf_counter() - self._init_started) * 1000
                    break
                if message.get("kind") == "fatal":
                    self._reason = str(message.get("error"))
                    raise RuntimeError(self._reason)
        else:
            raise TimeoutError("BGE-M3 Worker 初始化超时")
        connection = self._connection
        if connection is None:
            raise RuntimeError("向量 Worker 未启动")
        self._request_id += 1
        payload["request_id"] = self._request_id
        connection.send(payload)
        deadline = time.monotonic() + timeout_seconds
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("向量索引构建超时")
            if not connection.poll(min(1.0, remaining)):
                continue
            response = connection.recv()
            if response.get("request_id") != self._request_id:
                raise RuntimeError("向量 Worker 响应序号不匹配")
            if response.get("kind") == "progress":
                if callable(progress):
                    progress(dict(response))
                continue
            break
        if not response.get("ok"):
            raise RuntimeError(str(response.get("error") or "向量索引构建失败"))
        self._available = True
        self._reason = None
        return response

    def _request(self, payload: dict[str, Any], timeout: float | None = None) -> dict[str, Any]:
        connection = self._connection
        if connection is None:
            raise RuntimeError("向量 Worker 未启动")
        self._request_id += 1
        payload["request_id"] = self._request_id
        connection.send(payload)
        if not connection.poll(timeout or self._timeout):
            raise TimeoutError("向量 Worker 响应超时")
        response = connection.recv()
        if response.get("request_id") != self._request_id:
            raise RuntimeError("向量 Worker 响应序号不匹配")
        return response

    def close(self) -> None:
        connection = self._connection
        process = self._process
        if connection is not None and process is not None and process.is_alive():
            try:
                connection.send({"action": "close", "request_id": -1})
                process.join(timeout=3)
            except (BrokenPipeError, EOFError, OSError):
                pass
            if process.is_alive():
                process.terminate()
                process.join(timeout=3)
        if connection is not None:
            connection.close()
        self._connection = None
        self._process = None
        self._available = False


def _worker_main(
    connection: Connection,
    index_path: str,
    model_path: str,
    backend: str,
    vector_index_path: str,
    dimension: int,
    batch_size: int,
    max_sequence_length: int,
    device: str,
) -> None:
    try:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(model_path, device=device, local_files_only=True)
        model.max_seq_length = max_sequence_length
        db = sqlite3.connect(index_path)
        faiss_module: Any = None
        faiss_index: Any = None
        faiss_ids: list[str] = []
        if backend == "sqlite-vec":
            import sqlite_vec

            db.enable_load_extension(True)
            sqlite_vec.load(db)
            db.enable_load_extension(False)
            vector_ready = _vector_table_exists(db)
        elif backend == "faiss":
            import faiss as faiss_module

            faiss_path = Path(vector_index_path)
            ids_path = _faiss_ids_path(faiss_path)
            files_ready = faiss_path.is_file() and ids_path.is_file()
            if files_ready:
                faiss_index = faiss_module.read_index(str(faiss_path))
                faiss_ids = json.loads(ids_path.read_text(encoding="utf-8"))
                keyword_ids = {
                    str(row[0]) for row in db.execute("SELECT chunk_id FROM chunks")
                }
                vector_ready = (
                    faiss_index.d == dimension
                    and faiss_index.ntotal == len(faiss_ids)
                    and len(faiss_ids) == len(keyword_ids)
                    and set(faiss_ids) == keyword_ids
                )
            else:
                vector_ready = False
        else:
            raise ValueError("不支持的向量后端")
        connection.send(
            {
                "kind": "ready",
                "vector_index_ready": vector_ready,
                "backend": backend,
                "max_sequence_length": max_sequence_length,
                "device": device,
            }
        )
    except Exception:
        connection.send(
            {"kind": "fatal", "error": f"BGE-M3 或 {backend} 初始化失败"}
        )
        connection.close()
        return
    try:
        while True:
            message = connection.recv()
            request_id = message.get("request_id")
            action = message.get("action")
            if action == "close":
                break
            try:
                if action == "search":
                    vector = model.encode([message["query"]], normalize_embeddings=True)[0]
                    if backend == "sqlite-vec":
                        blob = _serialize(vector)
                        rows = db.execute(
                            "SELECT m.chunk_id, v.distance FROM chunk_vectors v "
                            "JOIN vector_chunk_map m ON m.rowid=v.rowid "
                            "WHERE v.embedding MATCH ? AND k=? ORDER BY v.distance",
                            (blob, int(message["limit"])),
                        ).fetchall()
                        # 归一化向量的 cosine distance 越小越好。
                        result = [
                            (row[0], max(0.0, min(1.0, 1.0 - float(row[1]))))
                            for row in rows
                        ]
                    else:
                        import numpy as np

                        query = np.asarray([vector], dtype="float32")
                        distances, indexes = faiss_index.search(
                            query, min(int(message["limit"]), len(faiss_ids))
                        )
                        result = [
                            (faiss_ids[int(index)], max(0.0, min(1.0, float(score))))
                            for score, index in zip(distances[0], indexes[0], strict=True)
                            if int(index) >= 0
                        ]
                    connection.send({"request_id": request_id, "ok": True, "rows": result})
                elif action in {"build", "build_path"}:
                    started = time.perf_counter()
                    batches = (
                        _chunk_batches_from_path(Path(message["chunks_path"]), batch_size)
                        if action == "build_path"
                        else _chunk_batches(iter(message["chunks"]), batch_size)
                    )
                    if backend == "sqlite-vec":
                        db.execute("DROP TABLE IF EXISTS chunk_vectors_build")
                        db.execute("DROP TABLE IF EXISTS vector_chunk_map_build")
                        db.execute(
                            f"CREATE VIRTUAL TABLE chunk_vectors_build USING vec0("
                            f"embedding float[{dimension}] distance_metric=cosine)"
                        )
                        db.execute(
                            "CREATE TABLE vector_chunk_map_build("
                            "rowid INTEGER PRIMARY KEY, chunk_id TEXT UNIQUE NOT NULL)"
                        )
                    else:
                        faiss_index = faiss_module.IndexFlatIP(dimension)
                        faiss_ids = []
                    count = 0
                    for batch in batches:
                        texts = [_embedding_text(chunk) for chunk in batch]
                        vectors = model.encode(
                            texts,
                            normalize_embeddings=True,
                            show_progress_bar=False,
                            batch_size=batch_size,
                        )
                        if backend == "sqlite-vec":
                            for chunk, vector in zip(batch, vectors, strict=True):
                                count += 1
                                db.execute(
                                    "INSERT INTO vector_chunk_map_build VALUES(?,?)",
                                    (count, chunk["chunk_id"]),
                                )
                                db.execute(
                                    "INSERT INTO chunk_vectors_build(rowid, embedding) VALUES(?,?)",
                                    (count, _serialize(vector)),
                                )
                            db.commit()
                        else:
                            import numpy as np

                            matrix = np.asarray(vectors, dtype="float32")
                            faiss_index.add(matrix)
                            faiss_ids.extend(chunk["chunk_id"] for chunk in batch)
                            count += len(batch)
                        connection.send(
                            {
                                "request_id": request_id,
                                "kind": "progress",
                                "processed": count,
                                "elapsed_ms": (time.perf_counter() - started) * 1000,
                            }
                        )
                    if backend == "sqlite-vec":
                        db.execute("DROP TABLE IF EXISTS chunk_vectors")
                        db.execute("DROP TABLE IF EXISTS vector_chunk_map")
                        db.execute(
                            f"CREATE VIRTUAL TABLE chunk_vectors USING vec0("
                            f"embedding float[{dimension}] distance_metric=cosine)"
                        )
                        db.execute(
                            "INSERT INTO chunk_vectors(rowid, embedding) "
                            "SELECT rowid, embedding FROM chunk_vectors_build"
                        )
                        db.execute("DROP TABLE chunk_vectors_build")
                        db.execute(
                            "ALTER TABLE vector_chunk_map_build RENAME TO vector_chunk_map"
                        )
                        db.execute(
                            "INSERT OR REPLACE INTO metadata VALUES('vector_backend',?)",
                            (backend,),
                        )
                        db.execute(
                            "INSERT OR REPLACE INTO metadata VALUES('vector_count',?)",
                            (str(count),),
                        )
                        db.commit()
                    else:
                        _write_faiss_index(
                            faiss_module,
                            faiss_index,
                            faiss_ids,
                            Path(vector_index_path),
                        )
                    elapsed = (time.perf_counter() - started) * 1000
                    connection.send(
                        {
                            "request_id": request_id,
                            "ok": True,
                            "vector_count": count,
                            "build_latency_ms": elapsed,
                            "backend": backend,
                            "max_sequence_length": max_sequence_length,
                            "device": device,
                            "index_bytes": _vector_index_bytes(
                                backend, Path(index_path), Path(vector_index_path)
                            ),
                            "peak_working_set_bytes": _peak_working_set_bytes(),
                        }
                    )
                else:
                    connection.send({"request_id": request_id, "ok": False})
            except Exception as exc:
                connection.send(
                    {
                        "request_id": request_id,
                        "ok": False,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
    except (EOFError, BrokenPipeError):
        pass
    finally:
        db.close()
        connection.close()


def _serialize(vector: Any) -> bytes:
    return array("f", (float(value) for value in vector)).tobytes()


def _vector_table_exists(db: sqlite3.Connection) -> bool:
    row = db.execute(
        "SELECT 1 FROM sqlite_master WHERE name='chunk_vectors'"
    ).fetchone()
    if row is None:
        return False
    metadata = dict(db.execute("SELECT key, value FROM metadata"))
    return (
        metadata.get("vector_backend") == "sqlite-vec"
        and metadata.get("vector_count") == metadata.get("chunk_count")
    )


def _chunk_batches_from_path(path: Path, batch_size: int) -> Any:
    def values() -> Any:
        with path.open("r", encoding="utf-8") as source:
            for line in source:
                if line.strip():
                    yield json.loads(line)

    return _chunk_batches(values(), batch_size)


def _chunk_batches(values: Any, batch_size: int) -> Any:
    batch: list[dict[str, Any]] = []
    for value in values:
        batch.append(value)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def _embedding_text(chunk: dict[str, Any]) -> str:
    return (
        f'{chunk["name_zh"]} {chunk["name_en"]} '
        f'{" ".join(chunk["aliases"])} {chunk["title"]} {chunk["text"]}'
    )


def _faiss_ids_path(index_path: Path) -> Path:
    return index_path.with_suffix(index_path.suffix + ".ids.json")


def _write_faiss_index(module: Any, index: Any, ids: list[str], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        prefix=path.name + ".", suffix=".tmp", dir=path.parent, delete=False
    )
    temporary = Path(handle.name)
    handle.close()
    ids_path = _faiss_ids_path(path)
    ids_handle = tempfile.NamedTemporaryFile(
        prefix=ids_path.name + ".", suffix=".tmp", dir=ids_path.parent, delete=False
    )
    temporary_ids = Path(ids_handle.name)
    ids_handle.close()
    module.write_index(index, str(temporary))
    temporary_ids.write_text(
        json.dumps(ids, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )
    os.replace(temporary, path)
    os.replace(temporary_ids, ids_path)


def convert_sqlite_vectors_to_faiss(
    sqlite_index_path: Path,
    faiss_index_path: Path,
    *,
    dimension: int,
    batch_size: int = 512,
) -> dict[str, Any]:
    """用同一批已验收向量构建 FAISS，避免重复运行 BGE-M3。"""

    import faiss
    import numpy as np
    import sqlite_vec

    started = time.perf_counter()
    db = sqlite3.connect(sqlite_index_path)
    try:
        db.enable_load_extension(True)
        sqlite_vec.load(db)
        db.enable_load_extension(False)
        if not _vector_table_exists(db):
            raise RuntimeError("sqlite-vec 索引尚未完整发布")
        expected = int(dict(db.execute("SELECT key, value FROM metadata"))["vector_count"])
        index = faiss.IndexFlatIP(dimension)
        ids: list[str] = []
        cursor = db.execute(
            "SELECT m.chunk_id, v.embedding FROM chunk_vectors v "
            "JOIN vector_chunk_map m ON m.rowid=v.rowid ORDER BY v.rowid"
        )
        while True:
            rows = cursor.fetchmany(batch_size)
            if not rows:
                break
            vectors = []
            for chunk_id, blob in rows:
                vector = np.frombuffer(blob, dtype="float32")
                if vector.size != dimension:
                    raise RuntimeError(
                        f"向量维度不匹配：{chunk_id}={vector.size}，期望 {dimension}"
                    )
                ids.append(str(chunk_id))
                vectors.append(vector)
            index.add(np.stack(vectors).astype("float32", copy=False))
        if len(ids) != expected:
            raise RuntimeError(f"向量数量不匹配：读取 {len(ids)}，期望 {expected}")
        _write_faiss_index(faiss, index, ids, faiss_index_path)
    finally:
        db.close()
    elapsed_ms = (time.perf_counter() - started) * 1000
    return {
        "backend": "faiss",
        "vector_count": len(ids),
        "build_latency_ms": elapsed_ms,
        "index_bytes": _vector_index_bytes(
            "faiss", sqlite_index_path, faiss_index_path
        ),
        "peak_working_set_bytes": _peak_working_set_bytes(),
        "source_backend": "sqlite-vec",
    }


def _vector_index_bytes(backend: str, keyword_path: Path, vector_path: Path) -> int:
    if backend == "sqlite-vec":
        return keyword_path.stat().st_size
    ids_path = _faiss_ids_path(vector_path)
    return vector_path.stat().st_size + ids_path.stat().st_size


def _peak_working_set_bytes() -> int | None:
    try:
        import psutil

        return int(psutil.Process().memory_info().peak_wset)
    except (ImportError, AttributeError, OSError):
        return None
