"""从有序事件流重建当前游戏状态。"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from .protocol import GameEvent


class EventOrderError(ValueError):
    """事件重复或倒序。"""


@dataclass(slots=True)
class StreamDiagnostics:
    parsed_events: int = 0
    invalid_events: int = 0
    ignored_lines: int = 0
    out_of_order_events: int = 0
    sequence_gaps: int = 0
    log_reopens: int = 0


@dataclass(slots=True)
class GameState:
    run_id: str | None = None
    active: bool = False
    last_seq: int = 0
    game_frame: int = 0
    context: dict[str, Any] = field(default_factory=dict)
    players: dict[str, dict[str, Any]] = field(default_factory=dict)
    room_collectibles: list[dict[str, Any]] = field(default_factory=list)
    last_event_type: str | None = None
    bridge_version: str | None = None


class StateStore:
    def __init__(self) -> None:
        self.state = GameState()
        self.diagnostics = StreamDiagnostics()
        self._last_seq_by_run: dict[str, int] = {}

    def mark_invalid(self) -> None:
        self.diagnostics.invalid_events += 1

    def mark_ignored(self) -> None:
        self.diagnostics.ignored_lines += 1

    def apply(self, event: GameEvent) -> GameState:
        previous_seq = self._last_seq_by_run.get(event.run_id, 0)
        if event.seq <= previous_seq:
            self.diagnostics.out_of_order_events += 1
            raise EventOrderError(
                f"run_id={event.run_id!r} 收到 seq={event.seq}，"
                f"当前已处理到 {previous_seq}"
            )
        if previous_seq and event.seq > previous_seq + 1:
            self.diagnostics.sequence_gaps += event.seq - previous_seq - 1
        self._last_seq_by_run[event.run_id] = event.seq
        self.diagnostics.parsed_events += 1

        state = self.state
        if event.type == "bridge_ready":
            bridge_version = event.payload.get("bridge_version")
            if isinstance(bridge_version, str):
                state.bridge_version = bridge_version
            state.last_event_type = event.type
            return state

        if state.run_id != event.run_id:
            state.run_id = event.run_id
            state.active = False
            state.last_seq = 0
            state.game_frame = 0
            state.context = {}
            state.players = {}
            state.room_collectibles = []

        previous_room = (
            state.context.get("room_index"),
            state.context.get("room_spawn_seed"),
        )
        incoming_room = (
            event.context.get("room_index"),
            event.context.get("room_spawn_seed"),
        )
        if incoming_room != previous_room:
            state.room_collectibles = []
        state.last_seq = event.seq
        state.game_frame = event.game_frame
        state.context = dict(event.context)
        state.last_event_type = event.type

        if event.type == "run_started":
            state.active = True
            self._merge_players(event.payload.get("players"))
        elif event.type == "run_ended":
            state.active = False
        elif event.type == "state_snapshot":
            self._merge_players(event.payload.get("players"))
        elif event.type in {"player_state_changed", "inventory_changed"}:
            self._merge_player(event.payload.get("player"))
        elif event.type == "collectible_spawned":
            self._merge_room_collectible(event.payload)
        elif event.type == "collectible_taken":
            self._mark_collectible_taken(event.payload)

        return state

    def snapshot(self) -> dict[str, Any]:
        return {
            "state": asdict(self.state),
            "diagnostics": asdict(self.diagnostics),
        }

    def _merge_players(self, players: Any) -> None:
        if not isinstance(players, list):
            return
        for player in players:
            self._merge_player(player)

    def _merge_player(self, player: Any) -> None:
        if not isinstance(player, dict):
            return
        controller_index = player.get("controller_index")
        if type(controller_index) is not int:
            return
        key = str(controller_index)
        previous = self.state.players.get(key, {})
        merged = dict(previous)
        merged.update(player)
        self.state.players[key] = merged

    def _merge_room_collectible(self, payload: Any) -> None:
        if not isinstance(payload, dict):
            return
        collectible_id = payload.get("collectible_id")
        if type(collectible_id) is not int or collectible_id < 1:
            return
        init_seed = payload.get("init_seed")
        for existing in self.state.room_collectibles:
            if init_seed is not None and existing.get("init_seed") == init_seed:
                existing.update(payload)
                return
        item = dict(payload)
        item["taken"] = False
        self.state.room_collectibles.append(item)

    def _mark_collectible_taken(self, payload: Any) -> None:
        if not isinstance(payload, dict):
            return
        collectible_id = payload.get("collectible_id")
        for existing in self.state.room_collectibles:
            if existing.get("collectible_id") == collectible_id:
                existing["taken"] = True
