from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import tempfile
import unittest

from oriens.huiji_data_audit import DataSnapshotAuditError, audit_data_snapshot


def _record(title: str, page_id: int, content: object, *, namespace: int = 486) -> dict:
    text = json.dumps(content, ensure_ascii=False, separators=(",", ":"))
    revision_id = page_id + 1000
    return {
        "schema_version": 1,
        "document_id": f"huiji:isaac:page:{page_id}:rev:{revision_id}",
        "page_id": page_id,
        "namespace": namespace,
        "title": title,
        "redirect": False,
        "revision_id": revision_id,
        "revision_timestamp": "2026-08-11T00:00:00Z",
        "revision_sha1": "wiki-sha1",
        "content_model": "Json.JsonConfig" if title.endswith(".json") else "Tabular.JsonConfig",
        "wikitext": text,
        "content_checksum": "sha256:" + sha256(text.encode("utf-8")).hexdigest(),
        "source_url": "https://isaac.huijiwiki.com/wiki/" + title,
        "revision_url": "https://isaac.huijiwiki.com/wiki/" + title + "?oldid=1",
        "source_title": "以撒的结合中文 Wiki：" + title,
        "source_type": "community-wiki-authorized-export",
        "retrieved_at": "2026-08-11T00:00:01Z",
        "license_note": "CC BY-NC-SA 3.0；第三方内容逐页核对",
        "authorization_ref": "private-written-authorization",
        "stale": False,
    }


def _requirements(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "required_exact_titles": ["Data:Item.tabx", "Data:Entity.tabx"],
                "expected_dependency_titles": ["Data:Rooms"],
                "required_tabular_tables": {
                    "Data:Item.tabx": {
                        "minimum_rows": 1,
                        "required_fields": ["page", "nameen"],
                    }
                },
                "room_title_prefix": "Data:Rooms/",
                "minimum_room_pages": 1,
                "minimum_room_stb_pages": 1,
                "minimum_distinct_room_files": 1,
                "required_room_fields": ["_type", "_file", "spawns"],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


class HuijiDataAuditTests(unittest.TestCase):
    def test_ready_snapshot_reports_tabular_and_room_stb_coverage(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            requirements = root / "requirements.json"
            pages = root / "pages.jsonl"
            _requirements(requirements)
            records = [
                _record(
                    "Data:Item.tabx",
                    1,
                    {
                        "schema": {"fields": [{"name": "page"}, {"name": "nameen"}]},
                        "data": [["硫磺火", "Brimstone"]],
                    },
                ),
                _record("Data:Entity.tabx", 2, {"schema": {"fields": []}, "data": []}),
                _record("Data:Rooms", 3, {"description": "room index"}),
                _record(
                    "Data:Rooms/test.stb/1.json",
                    4,
                    {
                        "_id": "Data:Rooms/test.stb/1.json",
                        "_type": "ROOM_STB",
                        "_file": "test.stb",
                        "spawns": [],
                    },
                ),
            ]
            pages.write_text(
                "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
                encoding="utf-8",
            )

            report = audit_data_snapshot(pages, requirements, expected_namespace=486)

            self.assertEqual(report["status"], "ready")
            self.assertEqual(report["records"], 4)
            self.assertEqual(report["room_stb_pages"], 1)
            self.assertEqual(report["distinct_room_files"], 1)
            self.assertEqual(report["tabular_tables"]["Data:Item.tabx"]["rows"], 1)
            self.assertEqual(report["missing_required_titles"], [])
            self.assertFalse(report["lua_execution"])

    def test_missing_required_title_is_incomplete_not_format_error(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            requirements = root / "requirements.json"
            pages = root / "pages.jsonl"
            _requirements(requirements)
            records = [
                _record(
                    "Data:Item.tabx",
                    1,
                    {
                        "schema": {"fields": [{"name": "page"}, {"name": "nameen"}]},
                        "data": [["硫磺火", "Brimstone"]],
                    },
                ),
                _record(
                    "Data:Rooms/test.stb/1.json",
                    2,
                    {"_type": "ROOM_STB", "_file": "test.stb", "spawns": []},
                ),
            ]
            pages.write_text(
                "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
                encoding="utf-8",
            )

            report = audit_data_snapshot(pages, requirements, expected_namespace=486)

            self.assertEqual(report["status"], "incomplete")
            self.assertEqual(report["missing_required_titles"], ["Data:Entity.tabx"])

    def test_checksum_mismatch_stops_validation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            requirements = root / "requirements.json"
            pages = root / "pages.jsonl"
            _requirements(requirements)
            record = _record("Data:Item.tabx", 1, {"data": []})
            record["content_checksum"] = "sha256:bad"
            pages.write_text(json.dumps(record, ensure_ascii=False) + "\n", encoding="utf-8")

            with self.assertRaisesRegex(DataSnapshotAuditError, "校验和不匹配"):
                audit_data_snapshot(pages, requirements, expected_namespace=486)

    def test_missing_required_tabular_field_is_incomplete(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            requirements = root / "requirements.json"
            pages = root / "pages.jsonl"
            _requirements(requirements)
            records = [
                _record(
                    "Data:Item.tabx",
                    1,
                    {
                        "schema": {"fields": [{"name": "page"}]},
                        "data": [["硫磺火"]],
                    },
                ),
                _record("Data:Entity.tabx", 2, {"data": []}),
                _record("Data:Rooms", 3, {"description": "room index"}),
                _record(
                    "Data:Rooms/test.stb/1.json",
                    4,
                    {"_type": "ROOM_STB", "_file": "test.stb", "spawns": []},
                ),
            ]
            pages.write_text(
                "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
                encoding="utf-8",
            )

            report = audit_data_snapshot(pages, requirements, expected_namespace=486)

            self.assertEqual(report["status"], "incomplete")
            self.assertIn("Data:Item.tabx 缺少字段：nameen", report["coverage_failures"])

    def test_wrong_namespace_stops_validation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            requirements = root / "requirements.json"
            pages = root / "pages.jsonl"
            _requirements(requirements)
            pages.write_text(
                json.dumps(_record("Data:Item.tabx", 1, {"data": []}, namespace=0)) + "\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(DataSnapshotAuditError, "与预期 486 不符"):
                audit_data_snapshot(pages, requirements, expected_namespace=486)


if __name__ == "__main__":
    unittest.main()
