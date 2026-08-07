# 阶段 0 事件协议

游戏 Mod 使用 `Isaac.DebugString` 输出单行日志：

```text
[ORIENS_EVENT]{"schema_version":1,"seq":184,"run_id":"ABCD EFGH:0","type":"room_entered",...}
```

游戏实际的 `log.txt` 会在前面附加自身日志标签。Python 解析器会在任意位置寻找 `[ORIENS_EVENT]`，因此兼容如下形式：

```text
[INFO] - Lua Debug: [ORIENS_EVENT]{...}
```

## 公共字段

| 字段 | 含义 |
|---|---|
| `schema_version` | 当前固定为 `1` |
| `seq` | 同一 `run_id` 内从 1 开始严格递增 |
| `run_id` | 开局种子字符串与开局帧组成的本局标识 |
| `type` | 语义事件类型 |
| `game_frame` | `Game():GetFrameCount()` |
| `context` | 楼层、房间和清理状态 |
| `payload` | 事件专有内容 |

## 阶段 0 事件

- 生命周期：`bridge_ready`、`run_started`、`run_ended`
- 场景：`floor_changed`、`room_entered`、`room_cleared`
- 玩家：`player_state_changed`、`inventory_changed`、`death`
- 道具：`collectible_spawned`、`collectible_taken`
- Boss：`boss_started`、`boss_defeated`
- 恢复：`heartbeat`、`state_snapshot`

`heartbeat` 默认每 60 游戏帧输出；完整 `state_snapshot` 默认每 300 游戏帧以及开局、进房和换层时输出。生命/资源检查每 6 游戏帧进行一次，只在指纹变化时输出事件。

## 顺序与恢复规则

- Python 按 `run_id + seq` 判定重复和倒序；重复或倒序事件不进入状态。
- 发现序号缺口时仍应用后续事件，同时累计 `sequence_gaps`。
- `state_snapshot` 可在日志截断、sidecar 晚启动或遗漏差异事件后重新建立完整玩家状态。
- 日志文件被截断或替换时，监听器会重新打开并从新文件开头继续。

