"""经授权后使用的灰机 Wiki 克制型离线采集器。

只读取以撒中文 Wiki 当前版本的 wikitext。默认单并发、低频率、无图片、
无历史修订，并通过原子检查点支持安全续跑。
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from hashlib import sha256
import json
import os
from pathlib import Path
import random
import socket
import sys
import time
from typing import Any, Callable, Mapping, Protocol, Sequence
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode, urlparse
from urllib.request import Request, urlopen


API_URL = "https://isaac.huijiwiki.com/api.php"
WIKI_BASE_URL = "https://isaac.huijiwiki.com/wiki/"
DEFAULT_LICENSE = (
    "以撒中文 Wiki 原创内容：CC BY-NC-SA 3.0；第三方内容与游戏素材需逐页核对"
)
MIN_DELAY_SECONDS = 3.0
MAX_BATCH_SIZE = 20
MAX_RESPONSE_BYTES = 16 * 1024 * 1024
RETRYABLE_HTTP_STATUS = {429, 500, 502, 503, 504}


class CrawlError(RuntimeError):
    """采集无法安全继续。"""


class QueryClient(Protocol):
    request_count: int

    def query(self, parameters: Mapping[str, str]) -> dict[str, Any]: ...


@dataclass(frozen=True, slots=True)
class CrawlOptions:
    output_dir: Path
    contact: str
    authorization_ref: str
    namespaces: tuple[int, ...]
    batch_size: int = 10
    delay_seconds: float = 5.0
    jitter_seconds: float = 2.0
    timeout_seconds: float = 30.0
    max_retries: int = 5
    max_pages: int = 100


class RateLimiter:
    def __init__(
        self,
        delay_seconds: float,
        jitter_seconds: float,
        *,
        clock: Callable[[], float] = time.monotonic,
        sleeper: Callable[[float], None] = time.sleep,
        random_source: random.Random | random.SystemRandom | None = None,
    ) -> None:
        self.delay_seconds = delay_seconds
        self.jitter_seconds = jitter_seconds
        self._clock = clock
        self._sleeper = sleeper
        self._random = random_source or random.SystemRandom()
        self._last_request_at: float | None = None

    def wait(self) -> None:
        if self._last_request_at is not None:
            target = self.delay_seconds + self._random.uniform(0, self.jitter_seconds)
            remaining = target - (self._clock() - self._last_request_at)
            if remaining > 0:
                self._sleeper(remaining)
        self._last_request_at = self._clock()

    def backoff(self, seconds: float) -> None:
        self._sleeper(max(0.0, seconds))
        self._last_request_at = self._clock()


class MediaWikiApiClient:
    def __init__(
        self,
        options: CrawlOptions,
        *,
        opener: Callable[..., Any] = urlopen,
        limiter: RateLimiter | None = None,
    ) -> None:
        self.options = options
        self._opener = opener
        self._limiter = limiter or RateLimiter(
            options.delay_seconds, options.jitter_seconds
        )
        self.request_count = 0

    def query(self, parameters: Mapping[str, str]) -> dict[str, Any]:
        query = dict(parameters)
        query.update({"format": "json", "formatversion": "2", "maxlag": "5"})
        url = API_URL + "?" + urlencode(query)
        request = Request(
            url,
            headers={
                "Accept": "application/json",
                "User-Agent": (
                    "OriensHuijiCrawler/0.1 "
                    f"(authorized personal research; contact: {self.options.contact})"
                ),
            },
            method="GET",
        )
        last_error: Exception | None = None
        for attempt in range(self.options.max_retries + 1):
            self._limiter.wait()
            self.request_count += 1
            try:
                with self._opener(request, timeout=self.options.timeout_seconds) as response:
                    payload = response.read(MAX_RESPONSE_BYTES + 1)
                    if len(payload) > MAX_RESPONSE_BYTES:
                        raise CrawlError("API 单次响应超过 16 MiB，已停止以避免异常负载")
                value = json.loads(payload.decode("utf-8"))
                if not isinstance(value, dict):
                    raise CrawlError("API 返回的 JSON 根节点不是对象")
                api_error = value.get("error")
                if isinstance(api_error, dict):
                    code = str(api_error.get("code", "unknown"))
                    if code in {"maxlag", "ratelimited"} and attempt < self.options.max_retries:
                        wait_seconds = _api_retry_seconds(api_error, attempt)
                        self._limiter.backoff(wait_seconds)
                        continue
                    raise CrawlError(f"MediaWiki API 拒绝请求：{code}")
                return value
            except HTTPError as exc:
                last_error = exc
                if exc.code in {401, 403}:
                    raise CrawlError(
                        f"服务器返回 HTTP {exc.code}；工具不会尝试绕过访问控制，"
                        "请让管理员确认 API 权限、账号或 IP 白名单"
                    ) from exc
                if exc.code not in RETRYABLE_HTTP_STATUS or attempt >= self.options.max_retries:
                    raise CrawlError(f"API 请求失败：HTTP {exc.code}") from exc
                self._limiter.backoff(_http_retry_seconds(exc, attempt))
            except (URLError, TimeoutError, socket.timeout) as exc:
                last_error = exc
                if attempt >= self.options.max_retries:
                    break
                self._limiter.backoff(_exponential_backoff(attempt))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise CrawlError("API 返回内容不是有效 UTF-8 JSON") from exc
        raise CrawlError("API 多次连接失败，已停止；稍后从检查点续跑") from last_error


class HuijiCrawler:
    def __init__(self, options: CrawlOptions, client: QueryClient | None = None) -> None:
        self.options = options
        self.client = client or MediaWikiApiClient(options)
        self.output_dir = options.output_dir.resolve()
        self.pages_path = self.output_dir / "pages.jsonl"
        self.checkpoint_path = self.output_dir / "checkpoint.json"
        self.run_path = self.output_dir / "run.json"

    def run(self) -> dict[str, Any]:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        state = self._load_checkpoint()
        if state["complete"]:
            return self._report(state, 0, "already_complete")

        known = _load_known_revisions(self.pages_path)
        written_this_run = 0
        started_at = _utc_now()
        with self.pages_path.open("a", encoding="utf-8", newline="\n") as output:
            while state["namespace_index"] < len(self.options.namespaces):
                namespace = self.options.namespaces[state["namespace_index"]]
                response = self.client.query(
                    _query_parameters(
                        namespace,
                        self.options.batch_size,
                        state.get("continuation", {}),
                    )
                )
                records = _records_from_response(
                    response,
                    authorization_ref=self.options.authorization_ref,
                )
                for record in records:
                    key = (record["page_id"], record["revision_id"])
                    if key in known:
                        continue
                    output.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
                    known.add(key)
                    written_this_run += 1
                output.flush()
                os.fsync(output.fileno())

                continuation = response.get("continue")
                if isinstance(continuation, dict) and continuation:
                    state["continuation"] = {
                        str(key): str(value) for key, value in continuation.items()
                    }
                else:
                    state["namespace_index"] += 1
                    state["continuation"] = {}
                state["pages_written_total"] = len(known)
                state["requests_total"] = int(state.get("requests_total", 0)) + 1
                state["updated_at"] = _utc_now()
                state["complete"] = state["namespace_index"] >= len(
                    self.options.namespaces
                )
                _write_json_atomic(self.checkpoint_path, state)

                if self.options.max_pages and written_this_run >= self.options.max_pages:
                    break

        status = "complete" if state["complete"] else "paused_at_limit"
        report = self._report(state, written_this_run, status)
        report["started_at"] = started_at
        _write_json_atomic(self.run_path, report)
        return report

    def _load_checkpoint(self) -> dict[str, Any]:
        if not self.checkpoint_path.exists():
            return {
                "schema_version": 1,
                "api_url": API_URL,
                "namespaces": list(self.options.namespaces),
                "namespace_index": 0,
                "continuation": {},
                "pages_written_total": 0,
                "requests_total": 0,
                "complete": False,
                "created_at": _utc_now(),
                "updated_at": _utc_now(),
            }
        try:
            value = json.loads(self.checkpoint_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise CrawlError("检查点损坏；请保留现场并人工检查 checkpoint.json") from exc
        if (
            not isinstance(value, dict)
            or value.get("schema_version") != 1
            or value.get("api_url") != API_URL
            or value.get("namespaces") != list(self.options.namespaces)
        ):
            raise CrawlError("检查点与当前 API 或命名空间参数不一致，请使用新的输出目录")
        return value

    def _report(
        self, state: Mapping[str, Any], written_this_run: int, status: str
    ) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "status": status,
            "api_url": API_URL,
            "output_dir": str(self.output_dir),
            "namespaces": list(self.options.namespaces),
            "batch_size": self.options.batch_size,
            "delay_seconds": self.options.delay_seconds,
            "jitter_seconds": self.options.jitter_seconds,
            "max_pages_this_run": self.options.max_pages,
            "pages_written_this_run": written_this_run,
            "pages_written_total": int(state.get("pages_written_total", 0)),
            "requests_this_process": self.client.request_count,
            "authorization_ref": self.options.authorization_ref,
            "license_note": DEFAULT_LICENSE,
            "finished_at": _utc_now(),
        }


def _query_parameters(
    namespace: int, batch_size: int, continuation: Mapping[str, str]
) -> dict[str, str]:
    parameters = {
        "action": "query",
        "generator": "allpages",
        "gapnamespace": str(namespace),
        "gaplimit": str(batch_size),
        "prop": "info|revisions",
        "inprop": "url",
        "rvslots": "main",
        "rvprop": "ids|timestamp|sha1|content|contentmodel",
    }
    parameters.update(continuation)
    return parameters


def _records_from_response(
    response: Mapping[str, Any], *, authorization_ref: str
) -> list[dict[str, Any]]:
    query = response.get("query")
    if not isinstance(query, dict):
        return []
    pages = query.get("pages")
    if not isinstance(pages, list):
        return []
    retrieved_at = _utc_now()
    records: list[dict[str, Any]] = []
    for page in pages:
        if not isinstance(page, dict):
            continue
        revisions = page.get("revisions")
        if not isinstance(revisions, list) or not revisions:
            continue
        revision = revisions[0]
        if not isinstance(revision, dict):
            continue
        slots = revision.get("slots")
        main = slots.get("main") if isinstance(slots, dict) else None
        if not isinstance(main, dict):
            continue
        content = main.get("content", "")
        if not isinstance(content, str):
            continue
        title = str(page.get("title", "")).strip()
        page_id = _strict_int(page.get("pageid"), "pageid")
        revision_id = _strict_int(revision.get("revid"), "revid")
        parent_id = _optional_int(revision.get("parentid"))
        source_url = str(page.get("canonicalurl") or _page_url(title))
        records.append(
            {
                "schema_version": 1,
                "document_id": f"huiji:isaac:page:{page_id}:rev:{revision_id}",
                "page_id": page_id,
                "namespace": _strict_int(page.get("ns"), "ns"),
                "title": title,
                "redirect": "redirect" in page,
                "revision_id": revision_id,
                "parent_revision_id": parent_id,
                "revision_timestamp": str(revision.get("timestamp", "")),
                "revision_sha1": str(revision.get("sha1", "")),
                "content_model": str(main.get("contentmodel", "wikitext")),
                "wikitext": content,
                "content_checksum": "sha256:"
                + sha256(content.encode("utf-8")).hexdigest(),
                "source_url": source_url,
                "revision_url": source_url + "?oldid=" + str(revision_id),
                "source_title": "以撒的结合中文 Wiki：" + title,
                "source_type": "community-wiki-authorized-export",
                "retrieved_at": retrieved_at,
                "license_note": DEFAULT_LICENSE,
                "authorization_ref": authorization_ref,
                "stale": False,
            }
        )
    records.sort(key=lambda value: (value["namespace"], value["page_id"]))
    return records


def _load_known_revisions(path: Path) -> set[tuple[int, int]]:
    known: set[tuple[int, int]] = set()
    if not path.exists():
        return known
    with path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
                known.add((int(value["page_id"]), int(value["revision_id"])))
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
                raise CrawlError(
                    f"pages.jsonl 第 {line_number} 行损坏；为避免覆盖数据，已停止"
                ) from exc
    return known


def _write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    temporary.replace(path)


def _http_retry_seconds(error: HTTPError, attempt: int) -> float:
    retry_after = error.headers.get("Retry-After") if error.headers else None
    if retry_after:
        try:
            return min(3600.0, max(0.0, float(retry_after)))
        except ValueError:
            try:
                target = parsedate_to_datetime(retry_after)
                now = datetime.now(target.tzinfo or timezone.utc)
                return min(3600.0, max(0.0, (target - now).total_seconds()))
            except (TypeError, ValueError, OverflowError):
                pass
    return _exponential_backoff(attempt)


def _api_retry_seconds(error: Mapping[str, Any], attempt: int) -> float:
    lag = error.get("lag")
    if type(lag) in {int, float}:
        return min(300.0, max(5.0, float(lag) * 2))
    return _exponential_backoff(attempt)


def _exponential_backoff(attempt: int) -> float:
    return min(300.0, 10.0 * (2**attempt))


def _page_url(title: str) -> str:
    return WIKI_BASE_URL + quote(title.replace(" ", "_"), safe="()/:,_-")


def _strict_int(value: Any, name: str) -> int:
    if type(value) is not int:
        raise CrawlError(f"API 页面字段 {name} 不是整数")
    return value


def _optional_int(value: Any) -> int | None:
    return value if type(value) is int else None


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _parse_namespaces(value: str) -> tuple[int, ...]:
    try:
        namespaces = tuple(dict.fromkeys(int(part.strip()) for part in value.split(",")))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("命名空间必须是逗号分隔的整数") from exc
    if not namespaces or any(namespace < 0 for namespace in namespaces):
        raise argparse.ArgumentTypeError("命名空间不能为空或为负数")
    return namespaces


def _validate_options(args: argparse.Namespace) -> CrawlOptions:
    contact = args.contact.strip()
    if not contact or any(character in contact for character in "\r\n"):
        raise CrawlError("--contact 必须提供有效且不换行的联系信息")
    authorization_ref = args.authorization_ref.strip()
    if not authorization_ref or any(character in authorization_ref for character in "\r\n"):
        raise CrawlError("--authorization-ref 不能为空或包含换行")
    if not 1 <= args.batch_size <= MAX_BATCH_SIZE:
        raise CrawlError(f"--batch-size 必须在 1 到 {MAX_BATCH_SIZE} 之间")
    if args.delay_seconds < MIN_DELAY_SECONDS:
        raise CrawlError(f"--delay-seconds 不得小于 {MIN_DELAY_SECONDS:g}")
    if not 0 <= args.jitter_seconds <= 30:
        raise CrawlError("--jitter-seconds 必须在 0 到 30 之间")
    if not 5 <= args.timeout_seconds <= 120:
        raise CrawlError("--timeout-seconds 必须在 5 到 120 之间")
    if not 0 <= args.max_retries <= 8:
        raise CrawlError("--max-retries 必须在 0 到 8 之间")
    if args.max_pages < 0:
        raise CrawlError("--max-pages 不得小于 0")
    return CrawlOptions(
        output_dir=args.output_dir,
        contact=contact,
        authorization_ref=authorization_ref,
        namespaces=args.namespaces,
        batch_size=args.batch_size,
        delay_seconds=args.delay_seconds,
        jitter_seconds=args.jitter_seconds,
        timeout_seconds=args.timeout_seconds,
        max_retries=args.max_retries,
        max_pages=args.max_pages,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "经授权后，使用灰机官方 MediaWiki API 低速导出以撒中文 Wiki 当前 wikitext。"
            "工具不会抓图片、历史修订或执行并发请求。"
        )
    )
    parser.add_argument(
        "--contact",
        required=True,
        help="写入 User-Agent 的管理员可联系邮箱或账号，不保存到页面数据中",
    )
    parser.add_argument(
        "--authorization-ref",
        default="private-written-authorization",
        help="不含隐私的授权引用，例如 approval-2026-08；不要填写授权原文",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/raw/huijiwiki/isaac"),
        help="原始 JSONL 和检查点目录；默认已被 Git 忽略",
    )
    parser.add_argument(
        "--namespaces",
        type=_parse_namespaces,
        default=(0,),
        help="逗号分隔的命名空间 ID；默认 0（百科正文）",
    )
    parser.add_argument("--batch-size", type=int, default=10, help="每次 API 最多取 10 页，上限 20")
    parser.add_argument(
        "--delay-seconds", type=float, default=5.0, help="请求基础间隔，默认 5 秒，硬下限 3 秒"
    )
    parser.add_argument(
        "--jitter-seconds", type=float, default=2.0, help="额外随机等待上限，默认 2 秒"
    )
    parser.add_argument("--timeout-seconds", type=float, default=30.0, help="单次请求超时")
    parser.add_argument("--max-retries", type=int, default=5, help="临时错误最大重试次数")
    parser.add_argument(
        "--max-pages",
        type=int,
        default=100,
        help="本次最多写入页数，默认 100；0 表示运行到当前范围完成",
    )
    parser.add_argument(
        "--plan",
        action="store_true",
        help="只打印计划并退出，绝不发出网络请求",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    try:
        options = _validate_options(parser.parse_args(argv))
        if urlparse(API_URL).scheme != "https":
            raise CrawlError("API 必须使用 HTTPS")
        if (argv is not None and "--plan" in argv) or (argv is None and "--plan" in sys.argv[1:]):
            print(
                json.dumps(
                    {
                        "mode": "plan-only-no-network",
                        "api_url": API_URL,
                        "output_dir": str(options.output_dir.resolve()),
                        "namespaces": list(options.namespaces),
                        "batch_size": options.batch_size,
                        "delay_seconds": options.delay_seconds,
                        "jitter_seconds": options.jitter_seconds,
                        "max_pages": options.max_pages,
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
            return 0
        report = HuijiCrawler(options).run()
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0
    except KeyboardInterrupt:
        print("采集已由用户暂停；已完成批次的检查点仍然有效，可用相同命令续跑", file=sys.stderr)
        return 130
    except (CrawlError, OSError) as exc:
        print(f"采集已安全停止：{exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
