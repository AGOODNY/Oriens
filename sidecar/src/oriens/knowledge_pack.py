"""可独立分发的本地知识包格式、校验与原子安装。"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
from typing import Any, Iterable
from uuid import uuid4

from . import __version__


class KnowledgePackError(RuntimeError):
    """可安全展示给用户的知识包错误。"""


@dataclass(frozen=True, slots=True)
class PackFile:
    path: str
    size: int
    sha256: str
    required: bool
    role: str


@dataclass(frozen=True, slots=True)
class KnowledgePackManifest:
    schema_version: int
    pack_id: str
    display_name: str
    game: str
    game_version: str
    content_version: str
    created_at: str
    files: tuple[PackFile, ...]
    vector_dimension: int
    embedding_model_id: str
    minimum_oriens_version: str

    @property
    def capabilities(self) -> frozenset[str]:
        roles = {item.role for item in self.files}
        result = {"keyword"} if "keyword_index" in roles else set()
        if "vector_index" in roles:
            result.add("vector")
        return frozenset(result)


@dataclass(frozen=True, slots=True)
class InstalledKnowledgePack:
    root: Path
    manifest: KnowledgePackManifest

    def file_for(self, role: str) -> Path | None:
        for item in self.manifest.files:
            if item.role == role:
                candidate = self.root / PurePosixPath(item.path)
                return candidate if candidate.is_file() else None
        return None

    @property
    def capabilities(self) -> frozenset[str]:
        result = {"keyword"} if self.file_for("keyword_index") is not None else set()
        if self.file_for("vector_index") is not None:
            result.add("vector")
        return frozenset(result)


class KnowledgePackSource:
    """未来下载器的输入边界；当前仅实现本地目录来源。"""

    def materialize(self) -> Path:
        raise NotImplementedError


@dataclass(frozen=True, slots=True)
class LocalDirectorySource(KnowledgePackSource):
    path: Path

    def materialize(self) -> Path:
        return self.path


class KnowledgePackManager:
    def __init__(self, root: Path, selection_file: Path | None = None) -> None:
        self.root = root.resolve()
        self.selection_file = (selection_file or self.root / "current.json").resolve()

    def enumerate_installed(self) -> tuple[InstalledKnowledgePack, ...]:
        if not self.root.is_dir():
            return ()
        packs: list[InstalledKnowledgePack] = []
        for directory in sorted(self.root.iterdir(), key=lambda item: item.name.casefold()):
            if not directory.is_dir() or directory.name.startswith("."):
                continue
            try:
                packs.append(self.validate(directory))
            except KnowledgePackError:
                continue
        return tuple(packs)

    def validate(self, directory: Path) -> InstalledKnowledgePack:
        root = directory.resolve()
        manifest_path = root / "manifest.json"
        try:
            raw = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise KnowledgePackError("知识包清单无法读取或格式错误。") from exc
        manifest = _parse_manifest(raw)
        for item in manifest.files:
            target = _safe_member(root, item.path)
            if target.is_symlink():
                raise KnowledgePackError("知识包包含不安全的文件链接。")
            if not target.is_file():
                if item.required:
                    raise KnowledgePackError(f"知识包缺少必需文件：{item.path}")
                continue
            try:
                size = target.stat().st_size
            except OSError as exc:
                raise KnowledgePackError(f"知识包文件无法读取：{item.path}") from exc
            if size != item.size:
                raise KnowledgePackError(f"知识包文件大小校验失败：{item.path}")
            if _sha256(target) != item.sha256:
                raise KnowledgePackError(f"知识包文件完整性校验失败：{item.path}")
        installed = InstalledKnowledgePack(root, manifest)
        if "keyword" not in installed.capabilities:
            raise KnowledgePackError("知识包缺少关键词索引能力。")
        return installed

    def current(self) -> InstalledKnowledgePack | None:
        try:
            raw = json.loads(self.selection_file.read_text(encoding="utf-8"))
            pack_id = raw.get("pack_id")
        except FileNotFoundError:
            return None
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise KnowledgePackError("当前知识包选择记录已损坏。") from exc
        if not isinstance(pack_id, str) or not _safe_pack_id(pack_id):
            raise KnowledgePackError("当前知识包选择记录无效。")
        return self.validate(self.root / pack_id)

    def select(self, pack_id: str) -> InstalledKnowledgePack:
        if not _safe_pack_id(pack_id):
            raise KnowledgePackError("知识包 ID 无效。")
        pack = self.validate(self.root / pack_id)
        self.selection_file.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.selection_file.with_name(f".{self.selection_file.name}.{uuid4().hex}.tmp")
        temporary.write_text(
            json.dumps({"pack_id": pack_id}, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, self.selection_file)
        return pack

    def install(
        self,
        source: KnowledgePackSource | Path,
        *,
        select: bool = False,
    ) -> InstalledKnowledgePack:
        source_root = (source.materialize() if isinstance(source, KnowledgePackSource) else source).resolve()
        checked = self.validate(source_root)
        pack_id = checked.manifest.pack_id
        destination = self.root / pack_id
        if destination.exists():
            raise KnowledgePackError("同 ID 的知识包已经安装；本阶段不会自动覆盖现有目录。")
        self.root.mkdir(parents=True, exist_ok=True)
        staging = self.root / f".install-{pack_id}-{uuid4().hex}"
        try:
            staging.mkdir()
            _copy_declared_files(source_root, staging, checked.manifest.files)
            shutil.copy2(source_root / "manifest.json", staging / "manifest.json")
            installed = self.validate(staging)
            os.replace(staging, destination)
            installed = InstalledKnowledgePack(destination, installed.manifest)
            if select:
                self.select(pack_id)
            return installed
        except KnowledgePackError:
            # 为遵守无确认不删除目录的规则，失败的暂存目录保留为隐藏隔离目录。
            raise
        except (OSError, shutil.Error) as exc:
            raise KnowledgePackError("知识包安装失败；当前知识包未被修改。") from exc


def _parse_manifest(raw: Any) -> KnowledgePackManifest:
    if not isinstance(raw, dict):
        raise KnowledgePackError("知识包清单必须是 JSON 对象。")
    required = {
        "schema_version", "pack_id", "display_name", "game", "game_version",
        "content_version", "created_at", "files", "vector_dimension",
        "embedding_model_id", "minimum_oriens_version",
    }
    if set(raw) != required:
        raise KnowledgePackError("知识包清单字段不完整或包含未知字段。")
    if raw["schema_version"] != 1:
        raise KnowledgePackError("不支持此知识包 schema 版本。")
    string_fields = (
        "pack_id", "display_name", "game", "game_version", "content_version",
        "created_at", "embedding_model_id", "minimum_oriens_version",
    )
    if any(not isinstance(raw[name], str) or not raw[name].strip() for name in string_fields):
        raise KnowledgePackError("知识包清单包含无效文本字段。")
    if not _safe_pack_id(raw["pack_id"]):
        raise KnowledgePackError("知识包 ID 无效。")
    try:
        datetime.fromisoformat(raw["created_at"].replace("Z", "+00:00"))
    except ValueError as exc:
        raise KnowledgePackError("知识包创建时间格式无效。") from exc
    if _version_tuple(raw["minimum_oriens_version"]) > _version_tuple(__version__):
        raise KnowledgePackError("此知识包需要更新版本的 Oriens。")
    dimension = raw["vector_dimension"]
    if type(dimension) is not int or dimension <= 0:
        raise KnowledgePackError("知识包向量维度必须是正整数。")
    if not isinstance(raw["files"], list) or not raw["files"]:
        raise KnowledgePackError("知识包文件列表不能为空。")
    files: list[PackFile] = []
    seen: set[str] = set()
    for value in raw["files"]:
        if not isinstance(value, dict) or set(value) != {"path", "size", "sha256", "required", "role"}:
            raise KnowledgePackError("知识包文件记录格式无效。")
        path = value["path"]
        if not isinstance(path, str) or path in seen:
            raise KnowledgePackError("知识包文件路径无效或重复。")
        _validate_relative_path(path)
        seen.add(path)
        size = value["size"]
        digest = value["sha256"]
        required_flag = value["required"]
        role = value["role"]
        if type(size) is not int or size < 0:
            raise KnowledgePackError("知识包文件大小无效。")
        if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise KnowledgePackError("知识包文件 SHA-256 无效。")
        if type(required_flag) is not bool or not isinstance(role, str) or not role.strip():
            raise KnowledgePackError("知识包文件属性无效。")
        files.append(PackFile(path, size, digest, required_flag, role.strip()))
    return KnowledgePackManifest(
        raw["schema_version"], raw["pack_id"], raw["display_name"], raw["game"],
        raw["game_version"], raw["content_version"], raw["created_at"], tuple(files),
        dimension, raw["embedding_model_id"], raw["minimum_oriens_version"],
    )


def _safe_pack_id(value: str) -> bool:
    return bool(value) and all(character.isalnum() or character in "._-" for character in value)


def _version_tuple(value: str) -> tuple[int, int, int]:
    parts = value.split(".")
    if len(parts) != 3 or any(not part.isdigit() for part in parts):
        raise KnowledgePackError("知识包兼容版本格式无效。")
    return tuple(int(part) for part in parts)  # type: ignore[return-value]


def _validate_relative_path(value: str) -> None:
    path = PurePosixPath(value)
    if path.is_absolute() or not path.parts or any(part in {"", ".", ".."} for part in path.parts):
        raise KnowledgePackError("知识包包含不安全的文件路径。")
    if "\\" in value or ":" in value:
        raise KnowledgePackError("知识包包含不安全的文件路径。")


def _safe_member(root: Path, value: str) -> Path:
    _validate_relative_path(value)
    candidate = root / PurePosixPath(value)
    resolved = candidate.resolve()
    if resolved != root and root not in resolved.parents:
        raise KnowledgePackError("知识包包含越界文件路径。")
    return candidate


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _copy_declared_files(source: Path, destination: Path, files: Iterable[PackFile]) -> None:
    for item in files:
        source_file = _safe_member(source, item.path)
        if not source_file.exists() and not item.required:
            continue
        target = _safe_member(destination, item.path)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_file, target)
