# Oriens: Your In-Game Guide

Oriens 是面向《以撒的结合：忏悔+》的 Windows 桌面游戏助手。它通过只读桥接 Mod 理解当前对局，在悬浮窗中提供道具建议、游戏问答、一般问答、语音交互和可选的本地长期记忆。

![Oriens 悬浮窗设计预览](docs/ui-mockups/oriens-overlay-expanded-hifi.png)

> 当前仓库提供开发运行版本，目标游戏版本为 Repentance+ `v1.9.7.17.J460`。Oriens 不修改房间、道具、输入或游戏规则。

## 功能

- **实时对局状态**：跟踪楼层、房间、角色、生命、硬币、钥匙、炸弹、持有物和房间道具。
- **局内短建议**：进入宝箱房等关键场景时，结合当前状态和本地资料生成可追溯建议。
- **文字问答**：游戏问题优先使用本地 RAG 与白名单来源；与游戏无关的问题也可以正常交流。
- **链式语音**：按住说话，依次完成实时 ASR、统一问答、流式 TTS 和有界音频播放。
- **桌面产品外壳**：控制中心、游戏悬浮窗、系统托盘和后台核心共享同一个应用实例。
- **本地长期记忆**：可选保存称呼、稳定偏好和提示偏好；默认关闭，可在界面中查看、修改、停用或删除。
- **按需视觉补充**：可选捕获已识别的前台游戏窗口；默认关闭，不回退到全桌面。
- **Realtime 实验模式**：可选使用 Qwen Omni Realtime；默认仍采用更稳定的链式语音。
- **离线降级**：没有密钥、网络、音频设备或向量模型时，日志监听、本地检索和文字能力仍可继续工作。
- **安全交付**：过期状态、非法 JSON、错误引用和不可信游戏事实会被拒绝或降级，不直接展示为可靠建议。

## 工作方式

```text
Oriens Bridge（只读 Lua Mod）
  │
  └─ [ORIENS_EVENT] 单行 JSON 日志
       │
       ▼
Python Sidecar
  ├─ 日志监听与状态重建
  ├─ 本地混合检索（实体 / FTS5 / BGE-M3 / FAISS）
  ├─ 结构化模型路由与来源校验
  ├─ 本地长期记忆
  ├─ ASR / TTS / Realtime / 视觉
  └─ PySide6 控制中心、悬浮窗与托盘
```

桥接 Mod 只读取公开游戏状态并写入游戏日志。特殊房间过渡期间会等待房间和玩家实体稳定后再读取，避免干扰矿层逃脱、妈刀部件等原生流程。

## 运行要求

| 项目 | 要求 |
| --- | --- |
| 操作系统 | Windows 10 或 Windows 11 |
| 游戏 | The Binding of Isaac: Repentance+ |
| 已验证版本 | `1.9.7.17.J460` |
| Python | `>=3.11,<3.14`，推荐 Python 3.11 |
| 桌面 UI | PySide6 6.8+ |
| 完整向量检索 | 可选安装 BGE-M3、FAISS 或 sqlite-vec 依赖 |
| 在线问答与语音 | 可选配置阿里云百炼北京地域凭据 |

## 快速开始

### 1. 创建项目虚拟环境

在仓库根目录运行：

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -e ".[rag]"
```

`.[rag]` 会安装完整向量检索依赖。若只需要基础桌面端和关键词检索，可以改用：

```powershell
.\.venv\Scripts\python.exe -m pip install -e .
```

### 2. 安装 Oriens Bridge

先完全退出游戏，再将桥接 Mod 安装到游戏目录：

```powershell
$gamePath = "C:\Program Files (x86)\Steam\steamapps\common\The Binding of Isaac Rebirth"
powershell -ExecutionPolicy Bypass -File .\tools\install_mod.ps1 -GamePath $gamePath
```

如果目标目录中已经存在本项目的旧版 Oriens，可显式覆盖：

```powershell
powershell -ExecutionPolicy Bypass -File .\tools\install_mod.ps1 -GamePath $gamePath -Force
```

在游戏 Mod 列表中确认已启用 **Oriens Bridge**。当前桥接版本为 `0.2.0`，安装目录固定为 `mods\oriens`。

### 3. 检查环境

```powershell
.\.venv\Scripts\oriens.exe doctor
```

该命令检查 Python 环境和默认游戏日志路径，不启动游戏或联网。

### 4. 启动桌面伴侣

默认零费用离线模式：

```powershell
.\.venv\Scripts\oriens.exe desktop
```

兼容入口 `ui` 与 `desktop` 使用同一个桌面应用：

```powershell
.\.venv\Scripts\oriens.exe ui
```

Oriens 默认从游戏日志末尾开始监听。可以先启动 Oriens，再启动游戏；晚启动时也会通过周期状态快照逐步恢复当前对局。

## 启用在线问答与语音

复制凭据模板：

```powershell
Copy-Item .env.example .env
```

在本地 `.env` 中填写：

```dotenv
DASHSCOPE_API_KEY=
DASHSCOPE_WORKSPACE_ID=
```

然后显式启用在线模式：

```powershell
.\.venv\Scripts\oriens.exe desktop --online
```

- `DASHSCOPE_API_KEY` 用于在线问答和视觉模型。
- `DASHSCOPE_WORKSPACE_ID` 用于北京地域 ASR、TTS 和 Realtime WebSocket。
- 缺少任一凭据时，相关在线能力会显示为不可用或自动降级，不影响本地文字功能。
- `.env` 已被 Git 忽略；不要把真实凭据写进命令行、提交记录、截图或日志。

## 使用方式

### 控制中心

控制中心用于：

- 查看日志连接、模型模式、知识包、语音、记忆、视觉和 Realtime 状态；
- 暂停或恢复日志监听；
- 显示、隐藏或展开悬浮窗；
- 修改用户设置；
- 管理本地长期记忆；
- 完全退出 Oriens。

关闭控制中心窗口只会缩到系统托盘。要停止后台监听、模型任务、音频和 Worker，请使用“完全退出 Oriens”。

### 悬浮窗

悬浮窗会展示当前房间、生命与资源、短建议、来源、置信度、费用估算和检索调试信息。

可以直接输入：

- `这个道具值得拿吗？`
- `硫磺火和妈妈的菜刀如何配合？`
- `我现在的资源适合进商店吗？`
- `你好？`

游戏相关回答必须通过本地状态和来源校验；一般问答不会伪装成本地游戏资料。

### 按住说话

在线链式语音可通过悬浮窗按钮，或在悬浮窗拥有焦点时按住配置键触发。默认按键为 `Space`，也支持配置为 `F8`–`F12`。

- 按住开始录音；
- 松开提交问题；
- 再次开口、换房、取消或退出会中止旧任务；
- 语音失败时，已通过校验的文字回答仍会保留。

当前版本没有系统级全局快捷键，也不会持续监听麦克风。

## 配置

默认配置位于 [config/default.toml](config/default.toml)。开发模式的用户设置保存在 `.oriens-user\config\settings.toml`；安装版使用 `%LOCALAPPDATA%\Oriens\config\settings.toml`。

推荐优先通过设置界面修改开关。以下能力默认值为：

| 能力 | 默认状态 | 说明 |
| --- | --- | --- |
| 链式语音 | 开启 | 只有在线凭据与设备可用时才连接服务 |
| 本地长期记忆 | 关闭 | 启用后仅写入本机 |
| 视觉补充 | 关闭 | 仅捕获已识别的前台游戏窗口 |
| 保存视觉调试截图 | 关闭 | 与视觉开关相互独立 |
| Realtime | 关闭 | 实验能力，需要在线模式 |
| semantic VAD | 关闭 | 仅影响 Realtime |
| 保存 Realtime 调试音频 | 关闭 | 默认不保存原始音频 |
| 单局在线问答预算 | ¥0.20 | 达到上限后使用本地证据摘要 |

保存需要重启的设置后，请完全退出并重新启动 Oriens。

## 本地知识库

仓库随附 `rag-v1` 小型、可回滚基线，可用于离线开发和测试：

```powershell
.\.venv\Scripts\oriens.exe rag-eval --config config\rag-v1.toml
.\.venv\Scripts\oriens.exe rag-query "硫磺火" --config config\rag-v1.toml
```

本机若已准备获授权的 `rag-v2.1 + FAISS` 数据，可启动完整知识库：

```powershell
.\.venv\Scripts\oriens.exe desktop --config config\rag-v2.1-faiss.toml --online
```

完整灰机 Wiki 快照、派生语料和大型索引受授权与体积限制，不随 Git 分发。正常问答不会临时联网搜索；向量 Worker 不可用时会退回实体匹配和 FTS5/BM25。

高级构建与评测：

```powershell
# 导入本地授权快照
.\.venv\Scripts\oriens.exe rag-import --config config\rag-v2.1.toml

# 构建关键词与 sqlite-vec 索引
.\.venv\Scripts\oriens.exe rag-build --config config\rag-v2.1.toml --skip-import --with-vectors

# 从同一批向量导出 FAISS
.\.venv\Scripts\oriens.exe rag-export-faiss --config config\rag-v2.1.toml --output data\indexes\rag-v2.1.faiss

# 离线质量与性能评测
.\.venv\Scripts\oriens.exe rag-eval --config config\rag-v2.1-faiss.toml --with-vectors
.\.venv\Scripts\oriens.exe rag-benchmark --config config\rag-v2.1-faiss.toml
```

数据来源、许可、规范化和索引边界记录在本地 `docs/` 目录中。完整授权语料、模型与生成的索引不随 Git 分发。

## 数据与隐私

Oriens 采用默认最小化策略：

- 桥接 Mod 只写结构化游戏状态日志，不修改游戏逻辑。
- 本地检索不会把整个知识库上传到模型服务。
- 长期记忆默认关闭，不保存 API Key、原始麦克风录音、每帧游戏状态或完整聊天历史。
- 视觉默认关闭；启用后只允许捕获已识别且位于前台的 `isaac-ng.exe` 客户区。
- 截图默认仅存在于内存；只有另行启用调试保存时才落盘。
- Realtime 只发送用户主动提交的音频；调试音频保存默认关闭。
- 普通界面不显示完整端点、业务空间 ID、授权头、原始协议帧或本机绝对隐私路径。

开发模式可写数据位于 `.oriens-user`。安装版可写数据位于：

```text
%LOCALAPPDATA%\Oriens\
├─ config
├─ knowledge
├─ models
├─ cache
├─ logs
└─ memory
```

`memory` 目录只会在启用真实长期记忆后创建。

## 常用诊断与开发命令

```powershell
# 监听日志并录制结构化事件
.\.venv\Scripts\oriens.exe listen --record data\recordings\live.jsonl

# 从日志开头读取
.\.venv\Scripts\oriens.exe listen --from-start

# 离线回放录制
.\.venv\Scripts\oriens.exe replay data\recordings\live.jsonl --verbose

# 固定道具的离线建议演示
.\.venv\Scripts\oriens.exe advice-demo 350

# 零费用语音与 Realtime 基准
.\.venv\Scripts\oriens.exe voice-benchmark --config config\rag-v1.toml
.\.venv\Scripts\oriens.exe realtime-benchmark --config config\rag-v1.toml
```

真实 API 烟雾测试可能产生费用，只有明确接受费用时才运行：

```powershell
.\.venv\Scripts\oriens.exe api-smoke 350 --confirm-charge
.\.venv\Scripts\oriens.exe voice-api-smoke --confirm-charge
```

## 测试

常规测试完全使用临时目录、模拟模型、模拟音频或程序生成数据，不读取真实用户设置，不访问网络，也不产生 API 费用：

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s sidecar\tests
```

桥接 Mod 的名称、版本、特殊房间延迟读取和回调隔离也有静态回归测试。

## 项目结构

```text
OriensYourIn-GameGuide/
├─ assets/                 # UI 字体、图标与装饰资源
├─ config/                 # 默认配置与 RAG 配置
├─ data/                   # 小型知识基线、评测与本地忽略数据
├─ docs/                   # 架构、协议、隐私和验收文档
├─ mod/oriens/             # Oriens Bridge
├─ sidecar/src/oriens/     # Python 应用核心
├─ sidecar/tests/          # 自动化测试
├─ tools/                  # Mod 安装和数据工具
├─ .env.example            # 在线凭据模板
├─ PROJECT_PLAN.md         # 历史计划与验收记录
└─ pyproject.toml          # Python 包与依赖
```

## 故障排查

### 游戏状态一直显示“等待日志”

1. 在游戏 Mod 列表确认 **Oriens Bridge** 已启用。
2. 完全重启游戏，使 Lua Mod 重新加载。
3. 运行 `oriens doctor` 检查日志路径。
4. 确认游戏版本使用 Repentance+ 日志目录。
5. 必要时用 `listen --from-start` 检查是否存在 `[ORIENS_EVENT]`。

### 向量 Worker 不可用

确认已安装 `.[rag]` 依赖，并且配置中的 BGE-M3 模型和向量索引存在。缺失时 Oriens 会继续使用关键词检索，不影响日志监听。

### 在线问答或语音不可用

确认启动命令包含 `--online`，并检查 `.env` 中的 API Key 和 Workspace ID。文字离线能力不依赖语音设备。

### 游戏出现与桥接 Mod 相关的异常

当前 `Oriens Bridge 0.2.0` 已对特殊房间、过场和玩家重建增加稳定等待与回调隔离。若仍可重复出现：

1. 暂时关闭 Oriens Bridge，确认异常是否消失；
2. 保留最新的 Repentance+ `log.txt`；
3. 记录角色、楼层、房间、触发动作和是否能够稳定复现；
4. 提交日志片段和复现步骤，不要包含凭据或其他个人文件。

## 本地设计文档

事件协议、游戏状态覆盖、桌面架构、语音、记忆、视觉、Realtime 与 RAG 数据治理等设计和验收记录保存在 `docs/`。按照当前仓库策略，这些内部文档暂不随远程源码分发。

## 当前限制

- 仅支持 Windows 与《以撒的结合：忏悔+》。
- 当前仓库是开发运行版本，没有安装器和自动更新。
- 完整 `rag-v2.1` 授权语料、BGE-M3 模型和大型索引不随 Git 分发。
- 按住说话快捷键只在悬浮窗获得焦点时工作，不是系统级全局热键。
- 视觉和 Realtime 属于默认关闭的可选能力。
- 模型建议不能替代玩家判断；低置信或资料不足时应以游戏内事实为准。
