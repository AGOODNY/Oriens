# 阶段 0 游戏状态覆盖率报告

报告日期：2026-08-07  
本机版本：The Binding of Isaac: Repentance+ `v1.9.7.17.J460`  
依据：游戏随附 `tools/LuaDocs`、`resources/scripts/enums.lua` 与实机日志

## 当前覆盖

| 决策状态 | 获取方式 | 阶段 0 状态 | 限制 |
|---|---|---|---|
| 角色类型 | `EntityPlayer:GetPlayerType()` | 已实现，待实机核对 | 以数值 ID 表示，名称映射留到后续数据层 |
| 红心/心之容器 | `GetHearts()` / `GetMaxHearts()` | 已实现，待实机核对 | 数值以半心为单位 |
| 魂心/黑心 | `GetSoulHearts()` / `GetBlackHearts()` | 已实现，待实机核对 | 黑心为位掩码，不是简单数量 |
| 永恒心/金心/骨心 | 对应玩家 API | 已实现，待实机核对 | 腐心和碎心未在本机随附文档中确认 |
| 硬币/钥匙/炸弹 | `GetNumCoins/Keys/Bombs()` | 已实现，待实机核对 | 特殊资源尚未覆盖 |
| 被动道具 | `GetCollectibleNum()` 遍历当前配置 | 已实现，待实机核对 | 仅在计数变化或快照时扫描；需验证 Modded ID 上限 |
| 主动道具与充能 | `GetActiveItem/Charge/BatteryCharge()` | 已实现，待实机核对 | 阶段 0 只记录默认主动槽 |
| 饰品/卡牌/胶囊 | `GetTrinket/GetCard/GetPill()` | 已实现，待实机核对 | 记录两个索引槽；特殊多槽角色需单测 |
| 楼层/楼层类型 | `Level:GetStage/GetStageType()` | 已实现，待实机核对 | 数值 ID，路线名称映射留到后续 |
| 房间索引/类型/变体 | Level、Room、RoomDescriptor | 已实现，待实机核对 | 房间几何和门未覆盖 |
| 房间清理状态 | `Room:IsClear()` 与清理奖励回调 | 已实现，待实机核对 | 特殊脚本房需验证 |
| 道具底座生成 | `MC_POST_PICKUP_INIT` | 已实现，待实机核对 | 能读 ID、价格和种子；可见性与选择组未覆盖 |
| 道具获得 | 玩家道具计数差异 | 已实现，待实机核对 | 反映“获得”，不保证能区分所有获得来源 |
| Boss 开始/击败 | Boss 房类型 + 进入/清理回调 | 已实现，待实机核对 | 多阶段 Boss 与脚本 Boss 需要专项验证 |

## 暂不能可靠读取或尚未实现

- 屏幕上所有敌人的语义阶段、攻击意图和精确危险度；
- 所有拾取物未来落点、隐藏房与未揭示房间内容；
- 商店/恶魔房中每个选择的完整交易约束和选择组；
- Modded 道具的稳定名称、说明、版本和机制语义；
- 腐心、碎心等在本机随附 LuaDocs 未确认的方法；
- 道具协同、冲突和攻略结论，这些属于后续本地 RAG，不应由状态探针猜测；
- 游戏画面中的非结构化 UI 状态；阶段 0 不读屏。

## 覆盖率结论（第一版）

阶段 0 所要求的六类核心信息——角色、房间、生命、资源、道具、事件顺序——均已有结构化采集路径。按项目计划中“启用视觉前”的六类决策口径，目前 API 预计可直接覆盖：

- 道具识别：部分覆盖；
- 当前资源和生命状态：核心覆盖；
- 房间类型和清理状态：核心覆盖；
- 商店/交易房选择：部分覆盖；
- Boss 阶段与危险提醒：部分覆盖；
- Modded 道具：仅 ID 级部分覆盖。

因此现阶段没有理由把读屏纳入默认开发范围。最终覆盖结论必须以 30 分钟实机录制与典型房间抽查为准。

## 尚待完成的实机验收

- [x] Mod 在 Repentance+ J460 无 Lua 错误加载；
- [x] `bridge_ready` 可被 Python 解析并录制；
- [x] `run_started` 和首个 `state_snapshot` 可被 Python 解析；
- [ ] 生命、硬币、钥匙、炸弹变化与画面一致；
- [ ] 进房、清房、换层事件与画面一致；
- [ ] 道具底座生成与获得事件可对应；
- [x] 连续游玩 30 分钟，倒序事件为 0；
- [x] 无明显游戏卡顿；
- [x] 录制文件可离线回放并重建最终状态。

### 已完成的实机烟雾验证

2026-08-07 在本机启动 Repentance+ J460，游戏日志确认运行
`mods/oriens/main.lua`，并输出协议 v1 的 `bridge_ready`。Python 监听器录制后离线回放结果为：

- `parsed_events = 1`；
- `invalid_events = 0`；
- `out_of_order_events = 0`；
- `sequence_gaps = 0`；
- `bridge_version = 0.1.0`。

烟雾验证只覆盖 Mod 加载和日志传输，不替代 30 分钟局内验收。

### 已完成的短局验证

2026-08-07 完成一段短局实机录制，游戏帧范围为 0–1239。录制包含 46 条事件：

- 1 条 `bridge_ready`；
- 1 条 `run_started` 和 1 条 `run_ended`；
- 4 条 `room_entered` 和 1 条 `room_cleared`；
- 6 条 `player_state_changed` 和 1 条 `inventory_changed`；
- 21 条 `heartbeat` 和 10 条 `state_snapshot`。

离线回放结果为 `invalid_events = 0`、`out_of_order_events = 0`、
`sequence_gaps = 0`，并重建出角色、生命、资源、库存与最终房间状态。
本次没有覆盖道具拾取、换层或 Boss。30 分钟稳定性验收当时因不具备持续游玩条件而暂缓；
后续已于 2026-08-09 完成，见下一节，短局结果本身不作为长时间验收依据。

### 30 分钟稳定性验证

2026-08-09 完成阶段 0 长时间实机验证。监听器从 10:27:18 录制到 11:12:17，
覆盖约 45 分钟墙钟时间；玩家确认游玩已超过 30 分钟，且未感到明显卡顿。

录制共包含 2776 条事件、7 个 `run_id`，其中 6 局记录到正常结束；最后一局在监听器达到
45 分钟预设上限时仍在进行，因此录制末尾没有 `run_ended`。该情况属于有界监听正常结束，
不是事件丢失或游戏异常。

主要事件覆盖：

- `room_entered` 271 条，`room_cleared` 81 条；
- `floor_changed` 6 条；
- `boss_started` 9 条，`boss_defeated` 7 条；
- `collectible_spawned` 47 条，`collectible_taken` 23 条；
- `player_state_changed` 433 条，`inventory_changed` 182 条；
- `heartbeat` 1180 条，`state_snapshot` 522 条；
- `death` 1 条。

离线回放诊断：

- `parsed_events = 2776`；
- `invalid_events = 0`；
- `out_of_order_events = 0`；
- `sequence_gaps = 0`；
- `log_reopens = 0`。

游戏日志未发现 Oriens Lua 加载或运行错误。回放能够重建角色、生命、资源、库存、楼层与房间状态。
本轮没有在每个检查点把重建数值与游戏画面逐项人工对照，因此“桌面端状态与游戏完全一致”仍保留为
后续人工核对项，不由结构正确性自动推断。
