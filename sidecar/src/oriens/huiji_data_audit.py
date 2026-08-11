"""离线验收经授权补采的灰机 Wiki ``Data:`` 当前修订快照。

该工具只读取本地 JSONL，不进行任何网络请求，也不执行 Lua 或页面内容。
页面正文逐行进入内存；整份快照不会一次性加载。
"""

from __future__ import annotations

import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping, Sequence


_REQUIRED_RAW_FIELDS = {
    "authorization_ref",
    "content_checksum",
    "content_model",
    "document_id",
    "license_note",
    "namespace",
    "page_id",
    "redirect",
    "retrieved_at",
    "revision_id",
    "revision_timestamp",
    "revision_url",
    "source_title",
    "source_type",
    "source_url",
    "stale",
    "title",
    "wikitext",
}


class DataSnapshotAuditError(RuntimeError):
    """快照格式或内容无法安全验收。"""


def audit_data_snapshot(
    pages_path: Path,
    requirements_path: Path,
    *,
    expected_namespace: int,
) -> dict[str, Any]:
    """流式检查 Data 快照并返回不含页面正文的确定性报告。"""

    requirements = _load_requirements(requirements_path)
    required_titles = set(requirements["required_exact_titles"])
    expected_titles = set(requirements["expected_dependency_titles"])
    watched_titles = required_titles | expected_titles
    found_titles: set[str] = set()
    seen_revisions: set[tuple[int, int]] = set()
    seen_titles: dict[str, tuple[int, int]] = {}
    content_models: Counter[str] = Counter()
    namespace_counts: Counter[int] = Counter()
    input_digest = sha256()
    records = 0
    redirects = 0
    json_pages = 0
    room_pages = 0
    room_stb_pages = 0
    room_prefix = requirements["room_title_prefix"]

    try:
        source = pages_path.open("rb")
    except OSError as exc:
        raise DataSnapshotAuditError(f"无法读取快照：{pages_path}") from exc

    with source:
        for line_number, raw_line in enumerate(source, start=1):
            input_digest.update(raw_line)
            if not raw_line.strip():
                continue
            try:
                record = json.loads(raw_line.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise DataSnapshotAuditError(
                    f"{pages_path.name}:{line_number} 不是有效 UTF-8 JSON"
                ) from exc
            _validate_record(
                record,
                pages_path=pages_path,
                line_number=line_number,
                expected_namespace=expected_namespace,
            )
            records += 1
            namespace_counts[record["namespace"]] += 1
            content_models[record["content_model"] or "unknown"] += 1
            redirects += int(record["redirect"])

            revision_key = (record["page_id"], record["revision_id"])
            if revision_key in seen_revisions:
                raise DataSnapshotAuditError(
                    f"{pages_path.name}:{line_number} 页面/修订组合重复：{revision_key}"
                )
            seen_revisions.add(revision_key)

            title = record["title"]
            previous = seen_titles.get(title)
            if previous is not None and previous != revision_key:
                raise DataSnapshotAuditError(
                    f"{pages_path.name}:{line_number} 当前修订快照包含重复标题：{title}"
                )
            seen_titles[title] = revision_key
            if title in watched_titles:
                found_titles.add(title)

            if title.startswith(room_prefix):
                room_pages += 1

            if record["redirect"] or not _should_parse_json(title, record["content_model"]):
                continue
            try:
                payload = json.loads(record["wikitext"])
            except json.JSONDecodeError as exc:
                raise DataSnapshotAuditError(
                    f"{pages_path.name}:{line_number} Data JSON 内容无效：{title}"
                ) from exc
            json_pages += 1
            if title.startswith(room_prefix) and _contains_room_stb(payload):
                room_stb_pages += 1

    missing_required = sorted(required_titles - found_titles, key=str.casefold)
    missing_expected = sorted(expected_titles - found_titles, key=str.casefold)
    coverage_failures: list[str] = []
    if room_pages < requirements["minimum_room_pages"]:
        coverage_failures.append(
            f"Data:Rooms 页面数 {room_pages} 小于要求 {requirements['minimum_room_pages']}"
        )
    if room_stb_pages < requirements["minimum_room_stb_pages"]:
        coverage_failures.append(
            "未在 Data:Rooms JSON 页面中发现足够的 _type=ROOM_STB 记录"
        )

    status = "ready"
    if missing_required or coverage_failures:
        status = "incomplete"
    return {
        "schema_version": 1,
        "status": status,
        "snapshot_path": str(pages_path.resolve()),
        "snapshot_sha256": "sha256:" + input_digest.hexdigest(),
        "expected_namespace": expected_namespace,
        "records": records,
        "redirects": redirects,
        "namespace_counts": _sorted_counter(namespace_counts),
        "content_models": _sorted_counter(content_models),
        "json_pages": json_pages,
        "room_pages": room_pages,
        "room_stb_pages": room_stb_pages,
        "required_titles_found": sorted(required_titles & found_titles, key=str.casefold),
        "missing_required_titles": missing_required,
        "missing_expected_dependency_titles": missing_expected,
        "coverage_failures": coverage_failures,
        "lua_execution": False,
    }


def _load_requirements(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DataSnapshotAuditError(f"无法读取依赖配置：{path}") from exc
    if not isinstance(value, dict) or value.get("schema_version") != 1:
        raise DataSnapshotAuditError("依赖配置 schema_version 无效")
    for field in ("required_exact_titles", "expected_dependency_titles"):
        items = value.get(field)
        if not isinstance(items, list) or not all(
            isinstance(item, str) and item.startswith("Data:") for item in items
        ):
            raise DataSnapshotAuditError(f"依赖配置字段 {field} 无效")
    prefix = value.get("room_title_prefix")
    if not isinstance(prefix, str) or not prefix.startswith("Data:Rooms/"):
        raise DataSnapshotAuditError("依赖配置字段 room_title_prefix 无效")
    for field in ("minimum_room_pages", "minimum_room_stb_pages"):
        if type(value.get(field)) is not int or value[field] < 0:
            raise DataSnapshotAuditError(f"依赖配置字段 {field} 无效")
    return value


def _validate_record(
    record: Any,
    *,
    pages_path: Path,
    line_number: int,
    expected_namespace: int,
) -> None:
    prefix = f"{pages_path.name}:{line_number}"
    if not isinstance(record, dict) or not _REQUIRED_RAW_FIELDS <= set(record):
        raise DataSnapshotAuditError(f"{prefix} 缺少原始记录字段")
    if record.get("schema_version") != 1:
        raise DataSnapshotAuditError(f"{prefix} schema_version 无效")
    if type(record["page_id"]) is not int or type(record["revision_id"]) is not int:
        raise DataSnapshotAuditError(f"{prefix} 页面/修订 ID 无效")
    if type(record["namespace"]) is not int or record["namespace"] != expected_namespace:
        raise DataSnapshotAuditError(
            f"{prefix} 命名空间 {record.get('namespace')!r} 与预期 {expected_namespace} 不符"
        )
    if type(record["redirect"]) is not bool or type(record["stale"]) is not bool:
        raise DataSnapshotAuditError(f"{prefix} 布尔字段无效")
    if not isinstance(record["title"], str) or not record["title"].startswith("Data:"):
        raise DataSnapshotAuditError(f"{prefix} 标题不是 Data 页面")
    if not isinstance(record["content_model"], str):
        raise DataSnapshotAuditError(f"{prefix} content_model 无效")
    if not isinstance(record["wikitext"], str):
        raise DataSnapshotAuditError(f"{prefix} 页面内容无效")
    expected_checksum = "sha256:" + sha256(record["wikitext"].encode("utf-8")).hexdigest()
    if record["content_checksum"] != expected_checksum:
        raise DataSnapshotAuditError(f"{prefix} 页面内容校验和不匹配")
    for field in ("source_url", "revision_url"):
        if not isinstance(record[field], str) or not record[field].startswith("https://"):
            raise DataSnapshotAuditError(f"{prefix} {field} 必须是 HTTPS URL")
    for field in (
        "authorization_ref",
        "document_id",
        "license_note",
        "retrieved_at",
        "revision_timestamp",
        "source_title",
        "source_type",
    ):
        if not isinstance(record[field], str) or not record[field].strip():
            raise DataSnapshotAuditError(f"{prefix} {field} 不能为空")


def _should_parse_json(title: str, content_model: str) -> bool:
    lowered_title = title.casefold()
    lowered_model = content_model.casefold()
    return (
        lowered_title.endswith((".json", ".tab", ".tabx"))
        or "json" in lowered_model
        or "tabular" in lowered_model
    )


def _contains_room_stb(value: Any) -> bool:
    if isinstance(value, dict):
        if value.get("_type") == "ROOM_STB":
            return True
        return any(_contains_room_stb(item) for item in value.values())
    if isinstance(value, list):
        return any(_contains_room_stb(item) for item in value)
    return False


def _sorted_counter(counter: Mapping[Any, int]) -> dict[str, int]:
    return {
        str(key): counter[key]
        for key in sorted(counter, key=lambda item: str(item).casefold())
    }


def _write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    temporary.replace(path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="离线验收灰机 Wiki Data 命名空间 JSONL；不会访问网络或执行 Lua。"
    )
    parser.add_argument("pages", type=Path, help="补采得到的 pages.jsonl")
    parser.add_argument(
        "--requirements",
        type=Path,
        default=Path("config/huiji-data-snapshot.json"),
        help="关键标题和 ROOM_STB 覆盖要求",
    )
    parser.add_argument(
        "--expected-namespace",
        type=int,
        required=True,
        help="通过站点 siteinfo 确认的 Data 命名空间数字 ID",
    )
    parser.add_argument(
        "--report",
        type=Path,
        help="可选报告路径；建议放在被 Git 忽略的快照目录中",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.expected_namespace < 0:
        print("--expected-namespace 不得为负数", file=sys.stderr)
        return 2
    try:
        report = audit_data_snapshot(
            args.pages,
            args.requirements,
            expected_namespace=args.expected_namespace,
        )
        if args.report:
            _write_json_atomic(args.report, report)
        print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
        return 0 if report["status"] == "ready" else 3
    except DataSnapshotAuditError as exc:
        print(f"Data 快照验收失败：{exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
