from __future__ import annotations

from pathlib import Path
import unittest
from xml.etree import ElementTree


ROOT = Path(__file__).resolve().parents[2]
MAIN_LUA = ROOT / "mod/oriens/main.lua"
METADATA = ROOT / "mod/oriens/metadata.xml"


class ModBridgeTests(unittest.TestCase):
    def test_bridge_has_stable_product_name_and_version(self) -> None:
        metadata = ElementTree.parse(METADATA).getroot()
        self.assertEqual(metadata.findtext("name"), "Oriens Bridge")
        self.assertEqual(metadata.findtext("id"), "oriens_bridge")
        self.assertEqual(metadata.findtext("version"), "0.2.0")

        source = MAIN_LUA.read_text(encoding="utf-8")
        self.assertIn('RegisterMod("Oriens Bridge", 1)', source)
        self.assertIn('local BRIDGE_VERSION = "0.2.0"', source)
        self.assertNotIn("Phase 0 Probe", source)

    def test_room_callbacks_defer_player_reads_until_entities_are_stable(self) -> None:
        source = MAIN_LUA.read_text(encoding="utf-8")
        new_room = source[source.index("local function onNewRoom()"):
                          source.index("local function onNewLevel()")]

        self.assertIn("pendingRoomEntered = true", new_room)
        self.assertNotIn("emitSnapshot", new_room)
        self.assertNotIn("GetPlayer", new_room)
        self.assertIn("room:GetFrameCount() < ROOM_STABLE_FRAMES", source)
        self.assertIn("player == nil", source)
        self.assertIn("not player:Exists()", source)
        self.assertIn("player.ControllerIndex < 0", source)

    def test_pickups_and_callback_failures_are_isolated_from_gameplay(self) -> None:
        source = MAIN_LUA.read_text(encoding="utf-8")
        pickup = source[source.index("local function onPickupInit"):
                        source.index("local function guarded")]

        self.assertIn("table.insert(pendingCollectibles", pickup)
        self.assertNotIn('emit("collectible_spawned"', pickup)
        self.assertIn("pcall(callback, ...)", source)
        self.assertIn("disabledCallbacks[callbackName] = true", source)


if __name__ == "__main__":
    unittest.main()
