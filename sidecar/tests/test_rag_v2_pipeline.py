from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import tempfile
import unittest

from oriens.rag import RagService
from oriens.rag_pipeline import build_keyword_index, iter_chunks
from oriens.rag_v2_pipeline import (
    RagV2Paths,
    _title_key,
    build_full_corpus,
    normalize_wikitext_sections,
)


def _record(
    title: str,
    page_id: int,
    revision_id: int,
    wikitext: str,
    *,
    namespace: int = 0,
    redirect: bool = False,
    content_model: str | None = None,
) -> dict[str, object]:
    url_title = title.replace(" ", "_")
    return {
        "schema_version": 1,
        "document_id": f"huiji:isaac:page:{page_id}:rev:{revision_id}",
        "namespace": namespace,
        "page_id": page_id,
        "revision_id": revision_id,
        "parent_revision_id": revision_id - 1,
        "revision_sha1": "0" * 40,
        "revision_timestamp": "2026-08-10T00:00:00Z",
        "retrieved_at": "2026-08-10T01:00:00Z",
        "title": title,
        "redirect": redirect,
        "content_model": content_model or ("wikitext" if namespace != 828 else "Scribunto"),
        "wikitext": wikitext,
        "content_checksum": "sha256:" + sha256(wikitext.encode("utf-8")).hexdigest(),
        "source_url": f"https://isaac.huijiwiki.com/wiki/{url_title}",
        "revision_url": f"https://isaac.huijiwiki.com/wiki/{url_title}?oldid={revision_id}",
        "source_title": f"以撒的结合中文 Wiki：{title}",
        "source_type": "community-wiki-authorized-export",
        "license_note": "CC BY-NC-SA 3.0；第三方内容逐页核对",
        "authorization_ref": "private-written-authorization",
        "stale": False,
    }


class RagV2PipelineTests(unittest.TestCase):
    def test_mediawiki_title_key_normalizes_after_namespace(self) -> None:
        self.assertEqual(_title_key("模板:stage"), _title_key("模板:Stage"))
        self.assertNotEqual(_title_key("Boss"), _title_key("BOSS"))

    def _paths(self, root: Path) -> RagV2Paths:
        output = root / "rag-v2"
        return RagV2Paths(
            raw_paths=(root / "pages.jsonl", root / "dependencies.jsonl"),
            chunks_path=output / "chunks.jsonl",
            manifest_path=output / "manifest.json",
            entities_path=output / "entities.jsonl",
            redirects_path=output / "redirects.jsonl",
            dependency_audit_path=output / "dependency-audit.json",
            lua_facts_path=output / "lua-facts.jsonl",
            overrides_path=Path(__file__).resolve().parents[2]
            / "data"
            / "dictionaries"
            / "rag-v2-overrides.json",
        )

    def _write_fixture(self, root: Path) -> None:
        pages = [
            _record(
                "C118",
                1180,
                2001,
                """{{infobox item}}\n{{ItemSummary}}\n'''硫磺火'''{{En|Brimstone}}是一个道具。\n\n==效果==\n*蓄力发射鲜血激光柱。\n*激光穿透敌人和障碍物。\n\n==协同效应==\n*{{item|妈妈的菜刀}}：会改变攻击方式。\n\n==画廊==\n<gallery>File.png|噪声</gallery>\n{{ItemNav}}""",
            ),
            _record("硫磺火", 1181, 2002, "#重定向[[C118]]", redirect=True),
            _record("Brimstone", 1182, 2003, "#REDIRECT [[硫磺火]]", redirect=True),
            _record("坏别名", 1183, 2004, "#重定向[[不存在页面]]", redirect=True),
            _record("循环甲", 1184, 2005, "#重定向[[循环乙]]", redirect=True),
            _record("循环乙", 1185, 2006, "#重定向[[循环甲]]", redirect=True),
            _record(
                "实体/68",
                1186,
                2007,
                """'''嘘嘘怪'''{{En|Vis}}是一个敌人。\n==不同形态==\n*'''硫磺火'''形态会发射激光，但本页不是道具页面。""",
            ),
        ]
        dependencies = [
            _record(
                "模板:Infobox item",
                3001,
                4001,
                "<includeonly>{{InfoAsk|table=Item.tabx}}</includeonly>",
                namespace=10,
            ),
            _record(
                "模板:ItemSummary",
                3002,
                4002,
                "<includeonly>{{#invoke:ItemQuery|item_summary}}</includeonly>",
                namespace=10,
            ),
            _record(
                "模块:Rooms",
                3003,
                4003,
                'local RoomType = {["ROOM_DEFAULT"] = 1, ["ROOM_SHOP"] = 2}\nreturn p',
                namespace=828,
            ),
            _record(
                "Data:Item.tabx",
                5001,
                6001,
                _tabular(
                    [
                        "page", "namezh", "nameen", "desczh", "descen",
                        "namelist", "id", "type", "effect", "source",
                    ],
                    [[
                        "c118", "硫磺火", "Brimstone", "蓄力激光", "Charged laser",
                        "硫磺火;大眼", 118, "道具", "蓄力发射穿透激光。", "重生",
                    ]],
                ),
                namespace=3500,
                content_model="增强表格",
            ),
            _record(
                "Data:ItemKeywords.tabx",
                5002,
                6002,
                _tabular(
                    ["page", "name_alias", "PinyinIndex", "colors"],
                    [["c118", "大眼;硫磺", "liuhuanghuo;dayan", "1"]],
                ),
                namespace=3500,
                content_model="增强表格",
            ),
            _record(
                "Data:Entity.tabx",
                5003,
                6003,
                _tabular(
                    [
                        "page", "namezh", "nameen", "type", "variant", "subtype",
                        "tag", "tips", "hp", "stagehp", "shieldstrength",
                        "collisionDamage", "source",
                    ],
                    [
                        ["实体/1000#1000.204.0", "召唤五芒星", "Spawn Pentagram", 1000, 204, 0, "效果", "", 0, 0, 0, 0, "忏悔+"],
                        ["实体/1000", "", "Player Creep Yellow", 1000, 204, 0, "效果", "", 0, 0, 0, 0, ""],
                    ],
                ),
                namespace=3500,
                content_model="增强表格",
            ),
            _record(
                "Data:Rooms/test.stb/7.json",
                5004,
                6004,
                json.dumps(
                    {
                        "_type": "ROOM_STB", "_file": "test.stb", "_i": "7",
                        "variant": 2, "name": "测试房间", "type": 13, "subtype": 0,
                        "shape": 1, "width": 15, "height": 9, "difficulty": 1,
                        "weight": 1,
                        "doors": [{"x": 0, "y": 4, "exists": True}],
                        "spawns": [{"x": 7, "y": 4, "entity": [
                            {"type": 5, "variant": 100, "subtype": 0, "weight": 1}
                        ]}],
                    },
                    ensure_ascii=False,
                ),
                namespace=3500,
                content_model="BSON",
            ),
            _record(
                "Data:Anm2/noise.json",
                5005,
                6005,
                json.dumps({"animation": "noise"}),
                namespace=3500,
                content_model="BSON",
            ),
        ]
        for path, values in (
            (root / "pages.jsonl", pages),
            (root / "dependencies.jsonl", dependencies),
        ):
            path.write_text(
                "".join(json.dumps(value, ensure_ascii=False) + "\n" for value in values),
                encoding="utf-8",
                newline="\n",
            )

    def test_streaming_import_redirects_metadata_and_determinism(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._write_fixture(root)
            paths = self._paths(root)
            first = build_full_corpus(paths)
            second = build_full_corpus(paths)
            self.assertEqual(first.corpus_checksum, second.corpus_checksum)
            self.assertEqual(first.raw_record_count, 15)
            self.assertEqual(first.redirect_resolved, 2)
            self.assertEqual(first.redirect_broken, 1)
            self.assertEqual(first.redirect_cycles, 2)
            self.assertGreaterEqual(first.chunk_count, 2)
            self.assertEqual(first.data_chunk_count, 4)
            self.assertEqual(first.room_layout_count, 1)
            chunks = list(iter_chunks(paths.chunks_path))
            required = {
                "page_id",
                "revision_id",
                "revision_timestamp",
                "raw_document_id",
                "redirect_sources",
                "section_path",
                "authorization_ref",
                "source_material_type",
            }
            self.assertTrue(required <= set(chunks[0]))
            self.assertEqual(chunks[0]["entity_id"], "collectible:118")
            aliases = set(chunks[0]["aliases"])
            self.assertIn("硫磺火", aliases)
            self.assertIn("Brimstone", aliases)
            self.assertNotIn("噪声", " ".join(chunk["text"] for chunk in chunks))
            enemy = next(chunk for chunk in chunks if chunk["page_id"] == 1186)
            self.assertNotEqual(enemy["entity_id"], "collectible:118")
            data_item = next(
                chunk for chunk in chunks
                if chunk["source_material_type"] == "wiki-data-tabular-row"
                and chunk["entity_id"] == "collectible:118"
            )
            self.assertIn("大眼", data_item["aliases"])
            self.assertEqual(data_item["name_en"], "Brimstone")
            conflicts = [
                chunk for chunk in chunks
                if chunk["entity_id"] == "game-entity:1000.204.0"
            ]
            self.assertEqual(len(conflicts), 2)
            self.assertTrue(all(chunk["source_conflict"] for chunk in conflicts))
            room = next(chunk for chunk in chunks if chunk["entity_type"] == "room_layout")
            self.assertIn("有效门：(0,4)", room["text"])
            self.assertIn("5.100.0×1", room["text"])
            redirects = [
                json.loads(line)
                for line in paths.redirects_path.read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(
                {row["status"] for row in redirects}, {"resolved", "broken", "cycle"}
            )
            lua_facts = paths.lua_facts_path.read_text(encoding="utf-8")
            self.assertIn("ROOM_SHOP", lua_facts)
            self.assertIn('"executed": false', lua_facts)

    def test_full_chunks_build_streaming_index_and_exact_alias_search(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._write_fixture(root)
            paths = self._paths(root)
            build_full_corpus(paths)
            index = root / "rag-v2.sqlite"
            report = build_keyword_index(iter_chunks(paths.chunks_path), index)
            self.assertEqual(report["chunk_count"], len(list(iter_chunks(paths.chunks_path))))
            service = RagService(index)
            result = service.retrieve("Brimstone")
            self.assertFalse(result.no_answer)
            self.assertEqual(result.hits[0].chunk.entity_id, "collectible:118")
            self.assertEqual(result.corpus_version, "rag-v2-huiji-2026-08-10")
            alias_result = service.retrieve("大眼")
            self.assertEqual(alias_result.hits[0].chunk.entity_id, "collectible:118")
            room_result = service.retrieve("测试房间")
            self.assertEqual(room_result.hits[0].chunk.entity_type, "room_layout")

    def test_semantic_sections_preserve_mechanics_and_drop_noise(self) -> None:
        sections = normalize_wikitext_sections(
            """导语。\n==效果==\n*造成伤害。\n*穿透敌人。\n==注意==\n不要贴墙。\n==轶事==\n与机制无关。""",
            "测试页",
        )
        paths = [path for path, _blocks in sections]
        text = " ".join(block for _path, blocks in sections for block in blocks)
        self.assertIn(("效果",), paths)
        self.assertIn("穿透敌人", text)
        self.assertNotIn("与机制无关", text)


if __name__ == "__main__":
    unittest.main()


def _tabular(fields: list[str], rows: list[list[object]]) -> str:
    return json.dumps(
        {
            "schema": {"fields": [{"name": field} for field in fields]},
            "data": rows,
        },
        ensure_ascii=False,
    )
