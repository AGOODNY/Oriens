# Oriens: Your In-Game Guide

Oriens 是《以撒的结合：忏悔+》的陪伴型游戏助手。本仓库目前只实现项目计划中的“阶段 0：技术探针”，用于验证以下链路：

```text
Repentance+ Lua Mod
  -> [ORIENS_EVENT] 单行 JSON 日志
  -> Python 日志监听器
  -> 事件校验、状态重建、录制与离线回放
```

阶段 0 不包含 UI、语音、RAG、截图或任何付费模型调用。

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

## 运行监听器

无需安装第三方依赖：

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
```

## 阶段 0 验收

实机验证时应先启动监听器并录制，再游玩至少 30 分钟。结束后运行回放，确认：

- `out_of_order_events` 为 0；
- `invalid_events` 为 0；
- 若 `sequence_gaps` 非 0，应能用游戏退出、Mod 重载或日志轮转解释；
- 重建的角色、楼层、房间、生命、资源和道具与游戏一致；
- 游戏没有可感知的卡顿。

事件协议见 [docs/event-protocol.md](docs/event-protocol.md)，覆盖范围与待验证项见 [docs/game-state-coverage.md](docs/game-state-coverage.md)。

