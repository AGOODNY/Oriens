# Oriens: Your In-Game Guide

Oriens 是《以撒的结合：忏悔+》的陪伴型游戏助手。仓库当前完成到“阶段 1：无语音垂直切片”，验证以下闭环：

```text
Repentance+ Lua Mod
  -> [ORIENS_EVENT] 单行 JSON 日志
  -> Python 日志监听、状态重建与本地道具资料
  -> 模拟模型或 Qwen 结构化短建议
  -> PySide6 悬浮窗显示来源、置信度和估算费用
```

阶段 1 不包含语音、截图、长期记忆、向量检索或完整 Wiki。在线模型默认关闭，自动化测试不会产生 API 费用。

## 已确认的本机环境

- 游戏：The Binding of Isaac: Repentance+ `v1.9.7.17.J460`
- 游戏目录：`E:\SteamLibrary\steamapps\common\The Binding of Isaac Rebirth`
- Mod 目录：`E:\SteamLibrary\steamapps\common\The Binding of Isaac Rebirth\mods`
- 日志：`C:\Users\32505\Documents\My Games\Binding of Isaac Repentance+\log.txt`
- Python 3.11：`D:\python\python.exe`

这些路径只作为本轮技术探针记录。命令行默认值会从 Windows 文档目录推导日志路径，游戏安装位置则由安装脚本参数明确指定。

## 安装 Mod

在仓库根目录运行：

```powershell
powershell -ExecutionPolicy Bypass -File .\tools\install_mod.ps1
```

脚本默认安装到本机已确认的游戏目录。如果目标 `mods\oriens` 已存在，脚本会停止，不会静默覆盖；确认是本项目旧版本后可显式加 `-Force`。

## 安装桌面端依赖

使用项目内虚拟环境，避免安装全局软件：

```powershell
D:\python\python.exe -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e .
```

## 运行悬浮窗

默认零费用离线模拟模式：

```powershell
.\.venv\Scripts\oriens.exe ui
```

程序从游戏日志末尾开始监听。缺少 `.env` 或 `DASHSCOPE_API_KEY` 时仍会正常启动，并在悬浮窗中显示简体中文离线提示。阶段 1 固定覆盖道具 ID `1`、`3`、`4`、`12`、`350`。

如需显式启用百炼：

```powershell
.\.venv\Scripts\oriens.exe ui --online
```

模型名称、北京地域端点、超时、重试、单价与本局预算上限位于 `config/default.toml`。真实密钥只放在被 Git 忽略的 `.env` 中，格式参考 `.env.example`。不要把密钥写入命令行参数、配置或日志。

## 阶段 0 监听与回放工具

这部分仍可直接使用 Python 3.11，无需启动 UI：

```powershell
$env:PYTHONPATH = "$PWD\sidecar\src"
D:\python\python.exe -m oriens.cli doctor
D:\python\python.exe -m oriens.cli listen --record data\recordings\live.jsonl
```

监听器默认从日志末尾开始，适合先启动监听器、再启动游戏。需要读取现有日志时加 `--from-start`。按 `Ctrl+C` 安全停止。

离线回放录制：

```powershell
$env:PYTHONPATH = "$PWD\sidecar\src"
D:\python\python.exe -m oriens.cli replay data\recordings\live.jsonl
```

## 自动化测试

```powershell
$env:PYTHONPATH = "$PWD\sidecar\src"
D:\python\python.exe -m unittest discover -s sidecar\tests -v
D:\python\python.exe -m oriens.cli advice-demo 350
```

真实 API 烟雾测试不是常规测试。只有用户确认费用后才执行：

```powershell
D:\python\python.exe -m oriens.cli api-smoke 350 --confirm-charge
```

它最多发起一项短建议任务；在当前配置下通常低于 ¥0.001，实际账单以百炼为准。

## 阶段 0 验收

实机验证时应先启动监听器并录制，再游玩至少 30 分钟。结束后运行回放，确认：

- `out_of_order_events` 为 0；
- `invalid_events` 为 0；
- 若 `sequence_gaps` 非 0，应能用游戏退出、Mod 重载或日志轮转解释；
- 重建的角色、楼层、房间、生命、资源和道具与游戏一致；
- 游戏没有可感知的卡顿。

事件协议见 [docs/event-protocol.md](docs/event-protocol.md)，覆盖范围与待验证项见 [docs/game-state-coverage.md](docs/game-state-coverage.md)。

阶段 1 的[百炼配置依据](docs/qwen-provider.md)与[验收记录](docs/phase1-acceptance.md)单独维护。
