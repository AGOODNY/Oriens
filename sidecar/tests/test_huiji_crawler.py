from __future__ import annotations

from io import BytesIO
import json
from pathlib import Path
import tempfile
import unittest
from urllib.error import HTTPError

from oriens.huiji_crawler import (
    CrawlError,
    CrawlOptions,
    HuijiCrawler,
    MediaWikiApiClient,
    RateLimiter,
    _records_from_response,
)


def _page(page_id: int, revision_id: int, title: str) -> dict:
    return {
        "pageid": page_id,
        "ns": 0,
        "title": title,
        "canonicalurl": f"https://isaac.huijiwiki.com/wiki/{title}",
        "revisions": [
            {
                "revid": revision_id,
                "parentid": revision_id - 1,
                "timestamp": "2026-08-10T01:02:03Z",
                "sha1": "wiki-sha1",
                "slots": {
                    "main": {
                        "contentmodel": "wikitext",
                        "content": f"== {title} ==\n测试内容",
                    }
                },
            }
        ],
    }


class _FakeClient:
    def __init__(self, responses: list[dict]) -> None:
        self.responses = responses
        self.parameters: list[dict[str, str]] = []
        self.request_count = 0

    def query(self, parameters):
        self.parameters.append(dict(parameters))
        self.request_count += 1
        return self.responses.pop(0)


class _HttpResponse:
    def __init__(self, value: dict) -> None:
        self.stream = BytesIO(json.dumps(value).encode("utf-8"))

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None

    def read(self, limit: int) -> bytes:
        return self.stream.read(limit)


class HuijiCrawlerTests(unittest.TestCase):
    def test_page_response_becomes_traceable_current_revision_record(self) -> None:
        records = _records_from_response(
            {"query": {"pages": [_page(12, 34, "硫磺火")]}},
            authorization_ref="approval-test",
        )
        self.assertEqual(len(records), 1)
        record = records[0]
        self.assertEqual(record["document_id"], "huiji:isaac:page:12:rev:34")
        self.assertEqual(record["title"], "硫磺火")
        self.assertIn("CC BY-NC-SA 3.0", record["license_note"])
        self.assertIn("逐页核对", record["license_note"])
        self.assertEqual(record["authorization_ref"], "approval-test")
        self.assertTrue(record["content_checksum"].startswith("sha256:"))
        self.assertNotIn("image", record)

    def test_checkpoint_resumes_without_duplicate_page_revision(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            options = CrawlOptions(
                Path(directory), "test@example.invalid", "approval-test", (0,),
                batch_size=1, max_pages=1,
            )
            first = _FakeClient(
                [
                    {
                        "continue": {"gapcontinue": "下一页", "continue": "gapcontinue||"},
                        "query": {"pages": [_page(1, 10, "第一页")]},
                    }
                ]
            )
            report = HuijiCrawler(options, first).run()
            self.assertEqual(report["status"], "paused_at_limit")

            second = _FakeClient(
                [{"query": {"pages": [_page(2, 20, "第二页")]}}]
            )
            report = HuijiCrawler(options, second).run()
            self.assertEqual(report["status"], "complete")
            self.assertEqual(second.parameters[0]["gapcontinue"], "下一页")
            lines = (Path(directory) / "pages.jsonl").read_text(
                encoding="utf-8"
            ).splitlines()
            self.assertEqual(len(lines), 2)

    def test_http_403_stops_without_retry_or_bypass(self) -> None:
        options = CrawlOptions(
            Path("unused"), "test@example.invalid", "approval-test", (0,),
            max_retries=5,
        )
        calls = 0

        def forbidden(_request, **_kwargs):
            nonlocal calls
            calls += 1
            raise HTTPError("https://example.invalid", 403, "Forbidden", {}, None)

        limiter = RateLimiter(3, 0, sleeper=lambda _seconds: None)
        client = MediaWikiApiClient(options, opener=forbidden, limiter=limiter)
        with self.assertRaisesRegex(CrawlError, "不会尝试绕过"):
            client.query({"action": "query"})
        self.assertEqual(calls, 1)

    def test_api_query_uses_contact_user_agent_and_no_write_actions(self) -> None:
        options = CrawlOptions(
            Path("unused"), "test@example.invalid", "approval-test", (0,)
        )
        captured = []

        def opener(request, **_kwargs):
            captured.append(request)
            return _HttpResponse({"query": {"pages": []}})

        limiter = RateLimiter(3, 0, sleeper=lambda _seconds: None)
        client = MediaWikiApiClient(options, opener=opener, limiter=limiter)
        client.query({"action": "query", "generator": "allpages"})
        request = captured[0]
        self.assertIn("test@example.invalid", request.get_header("User-agent"))
        self.assertIn("action=query", request.full_url)
        self.assertNotIn("action=edit", request.full_url)


if __name__ == "__main__":
    unittest.main()
