"""BGE-M3 独立进程 Worker 与 sqlite-vec 后端。

模型加载、嵌入和向量查询均发生在子进程；主 UI 线程只观察就绪状态。
"""

from __future__ import annotations

from array import array
from multiprocessing import get_context
from multiprocessing.connection import Connection
from pathlib import Path
import sqlite3
import time
from typing import Any, Sequence


class VectorWorkerClient:
    def __init__(
        self,
        *,
        index_path: Path,
        model_path: Path,
        dimension: int = 1024,
        request_timeout_seconds: float = 8.0,
    ) -> None:
        self._index_path = index_path.resolve()
        self._model_path = model_path.resolve()
        self._dimension = dimension
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
            args=(child, str(self._index_path), str(self._model_path), self._dimension),
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
        response = self._request({"action": "build", "chunks": chunks}, timeout=1800.0)
        if not response.get("ok"):
            raise RuntimeError("向量索引构建失败")
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


def _worker_main(connection: Connection, index_path: str, model_path: str, dimension: int) -> None:
    try:
        from sentence_transformers import SentenceTransformer
        import sqlite_vec

        model = SentenceTransformer(model_path, device="cpu", local_files_only=True)
        db = sqlite3.connect(index_path)
        db.enable_load_extension(True)
        sqlite_vec.load(db)
        db.enable_load_extension(False)
        vector_ready = _vector_table_exists(db)
        connection.send({"kind": "ready", "vector_index_ready": vector_ready})
    except Exception:
        connection.send({"kind": "fatal", "error": "BGE-M3 或 sqlite-vec 初始化失败"})
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
                    blob = _serialize(vector)
                    rows = db.execute(
                        "SELECT m.chunk_id, v.distance FROM chunk_vectors v "
                        "JOIN vector_chunk_map m ON m.rowid=v.rowid "
                        "WHERE v.embedding MATCH ? AND k=? ORDER BY v.distance",
                        (blob, int(message["limit"])),
                    ).fetchall()
                    # 归一化向量的 cosine distance 越小越好。
                    result = [(row[0], max(0.0, min(1.0, 1.0 - float(row[1])))) for row in rows]
                    connection.send({"request_id": request_id, "ok": True, "rows": result})
                elif action == "build":
                    chunks = message["chunks"]
                    texts = [f'{chunk["name_zh"]} {chunk["name_en"]} {" ".join(chunk["aliases"])} {chunk["title"]} {chunk["text"]}' for chunk in chunks]
                    started = time.perf_counter()
                    vectors = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
                    db.execute("DROP TABLE IF EXISTS chunk_vectors")
                    db.execute("DROP TABLE IF EXISTS vector_chunk_map")
                    db.execute(
                        f"CREATE VIRTUAL TABLE chunk_vectors USING vec0("
                        f"embedding float[{dimension}] distance_metric=cosine)"
                    )
                    db.execute("CREATE TABLE vector_chunk_map(rowid INTEGER PRIMARY KEY, chunk_id TEXT UNIQUE NOT NULL)")
                    for rowid, (chunk, vector) in enumerate(zip(chunks, vectors, strict=True), start=1):
                        db.execute("INSERT INTO vector_chunk_map VALUES(?,?)", (rowid, chunk["chunk_id"]))
                        db.execute("INSERT INTO chunk_vectors(rowid, embedding) VALUES(?,?)", (rowid, _serialize(vector)))
                    db.commit()
                    elapsed = (time.perf_counter() - started) * 1000
                    connection.send({"request_id": request_id, "ok": True, "vector_count": len(chunks), "build_latency_ms": elapsed})
                else:
                    connection.send({"request_id": request_id, "ok": False})
            except Exception:
                connection.send({"request_id": request_id, "ok": False})
    except (EOFError, BrokenPipeError):
        pass
    finally:
        db.close()
        connection.close()


def _serialize(vector: Any) -> bytes:
    return array("f", (float(value) for value in vector)).tobytes()


def _vector_table_exists(db: sqlite3.Connection) -> bool:
    row = db.execute("SELECT 1 FROM sqlite_master WHERE name='chunk_vectors'").fetchone()
    return row is not None
