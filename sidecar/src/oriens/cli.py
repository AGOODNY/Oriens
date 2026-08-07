"""阶段 0 命令行入口。"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
from pathlib import Path
import platform
import sys
import time
from typing import TextIO

from .protocol import EventParseError, parse_event_line
from .state import EventOrderError, StateStore
from .tailer import LogTailer


def default_log_path() -> Path:
    return (
        Path.home()
        / "Documents"
        / "My Games"
        / "Binding of Isaac Repentance+"
        / "log.txt"
    )


def _json_output(value: object) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2))


def run_doctor(args: argparse.Namespace) -> int:
    log_path = args.log.resolve()
    _json_output(
        {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "log_path": str(log_path),
            "log_exists": log_path.exists(),
            "log_size": log_path.stat().st_size if log_path.exists() else None,
        }
    )
    return 0 if log_path.exists() else 1


def _handle_line(
    line: str,
    store: StateStore,
    recorder: TextIO | None,
    *,
    verbose: bool,
) -> None:
    try:
        event = parse_event_line(line)
    except EventParseError as exc:
        store.mark_invalid()
        print(f"协议错误：{exc}", file=sys.stderr)
        return
    if event is None:
        store.mark_ignored()
        return
    try:
        store.apply(event)
    except EventOrderError as exc:
        print(f"顺序错误：{exc}", file=sys.stderr)
        return
    if recorder is not None:
        recorder.write(event.to_json() + "\n")
        recorder.flush()
    if verbose:
        print(
            f"事件 seq={event.seq} run={event.run_id} "
            f"frame={event.game_frame} type={event.type}"
        )


def run_listen(args: argparse.Namespace) -> int:
    log_path = args.log.resolve()
    if args.record is not None:
        args.record.resolve().parent.mkdir(parents=True, exist_ok=True)
        record_context = args.record.resolve().open("a", encoding="utf-8", newline="\n")
    else:
        record_context = nullcontext(None)

    tailer = LogTailer(log_path, from_start=args.from_start)
    store = StateStore()
    deadline = time.monotonic() + args.duration if args.duration > 0 else None
    print(f"正在监听：{log_path}")
    if args.record is not None:
        print(f"正在录制：{args.record.resolve()}")

    try:
        with record_context as recorder:
            while deadline is None or time.monotonic() < deadline:
                poll = tailer.poll()
                if poll.reopened:
                    store.diagnostics.log_reopens += 1
                for line in poll.lines:
                    _handle_line(line, store, recorder, verbose=not args.quiet)
                time.sleep(args.poll_interval)
    except KeyboardInterrupt:
        print("监听已停止。")
    finally:
        tailer.close()

    _json_output(store.snapshot())
    diagnostics = store.diagnostics
    return 2 if diagnostics.invalid_events or diagnostics.out_of_order_events else 0


def run_replay(args: argparse.Namespace) -> int:
    store = StateStore()
    with args.recording.resolve().open("r", encoding="utf-8", errors="replace") as source:
        for line in source:
            _handle_line(line, store, None, verbose=args.verbose)
    _json_output(store.snapshot())
    diagnostics = store.diagnostics
    return 2 if diagnostics.invalid_events or diagnostics.out_of_order_events else 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Oriens 阶段 0 游戏日志技术探针")
    subparsers = parser.add_subparsers(dest="command", required=True)

    doctor = subparsers.add_parser("doctor", help="检查日志路径和 Python 环境")
    doctor.add_argument("--log", type=Path, default=default_log_path())
    doctor.set_defaults(handler=run_doctor)

    listen = subparsers.add_parser("listen", help="监听游戏日志并重建状态")
    listen.add_argument("--log", type=Path, default=default_log_path())
    listen.add_argument("--record", type=Path)
    listen.add_argument("--from-start", action="store_true")
    listen.add_argument("--duration", type=float, default=0.0, help="运行秒数；0 表示持续监听")
    listen.add_argument("--poll-interval", type=float, default=0.1)
    listen.add_argument("--quiet", action="store_true")
    listen.set_defaults(handler=run_listen)

    replay = subparsers.add_parser("replay", help="离线回放 JSONL 录制")
    replay.add_argument("recording", type=Path)
    replay.add_argument("--verbose", action="store_true")
    replay.set_defaults(handler=run_replay)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.handler(args)


if __name__ == "__main__":
    raise SystemExit(main())
