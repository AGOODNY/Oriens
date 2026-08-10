from __future__ import annotations

import unittest

from oriens.protocol import GameEvent
from oriens.state import EventOrderError, StateStore


def event(seq: int, event_type: str, payload: dict | None = None) -> GameEvent:
    return GameEvent(
        schema_version=1,
        seq=seq,
        run_id="AAAA BBBB:0",
        type=event_type,
        game_frame=seq * 10,
        context={"stage": 1, "room_index": 0},
        payload=payload or {},
    )


class StateStoreTests(unittest.TestCase):
    def test_rebuilds_player_state(self) -> None:
        store = StateStore()
        store.apply(event(1, "run_started", {"players": []}))
        store.apply(
            event(
                2,
                "player_state_changed",
                {"player": {"controller_index": 0, "player_type": 0, "resources": {"coins": 7}}},
            )
        )
        store.apply(event(3, "run_ended"))

        self.assertFalse(store.state.active)
        self.assertEqual(store.state.players["0"]["resources"]["coins"], 7)
        self.assertEqual(store.diagnostics.parsed_events, 3)

    def test_counts_sequence_gap(self) -> None:
        store = StateStore()
        store.apply(event(1, "run_started"))
        store.apply(event(4, "heartbeat"))
        self.assertEqual(store.diagnostics.sequence_gaps, 2)

    def test_rejects_duplicate_or_out_of_order_event(self) -> None:
        store = StateStore()
        store.apply(event(2, "heartbeat"))
        with self.assertRaises(EventOrderError):
            store.apply(event(2, "heartbeat"))
        self.assertEqual(store.diagnostics.out_of_order_events, 1)

    def test_tracks_collectibles_in_current_room(self) -> None:
        store = StateStore()
        spawned = event(
            1,
            "collectible_spawned",
            {"collectible_id": 350, "init_seed": 99, "price": 0},
        )
        store.apply(spawned)
        self.assertEqual(store.state.room_collectibles[0]["collectible_id"], 350)
        store.apply(event(2, "collectible_taken", {"collectible_id": 350}))
        self.assertTrue(store.state.room_collectibles[0]["taken"])

        moved = GameEvent(
            schema_version=1,
            seq=3,
            run_id="AAAA BBBB:0",
            type="room_entered",
            game_frame=30,
            context={"stage": 1, "room_index": 1, "room_spawn_seed": 22},
            payload={},
        )
        store.apply(moved)
        self.assertEqual(store.state.room_collectibles, [])


if __name__ == "__main__":
    unittest.main()
