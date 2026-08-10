"""从仓库根目录启动经授权的灰机 Wiki 低速采集器。"""

from __future__ import annotations

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "sidecar" / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from oriens.huiji_crawler import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
