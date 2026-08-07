local Oriens = RegisterMod("Oriens Phase 0 Probe", 1)
local game = Game()
local json = require("json")

local PREFIX = "[ORIENS_EVENT]"
local SCHEMA_VERSION = 1
local BRIDGE_VERSION = "0.1.0"
local HEARTBEAT_FRAMES = 60
local SNAPSHOT_FRAMES = 300
local PLAYER_POLL_FRAMES = 6

local sequence = 0
local runId = "boot"
local activeRun = false
local lastHeartbeatFrame = -1
local lastSnapshotFrame = -1
local lastPlayerPollFrame = -1
local lastPlayerFingerprints = {}
local lastInventoryFingerprints = {}
local lastInventories = {}
local deathReported = {}

local function roomContext()
    if not activeRun then
        return json.EMPTY_OBJECT
    end

    local level = game:GetLevel()
    local room = game:GetRoom()
    local descriptor = level:GetCurrentRoomDesc()
    local roomVariant = -1
    local roomSpawnSeed = 0
    if descriptor ~= nil then
        roomSpawnSeed = descriptor.SpawnSeed or 0
        if descriptor.Data ~= nil then
            roomVariant = descriptor.Data.Variant or -1
        end
    end

    return {
        stage = level:GetStage(),
        stage_type = level:GetStageType(),
        room_index = level:GetCurrentRoomIndex(),
        room_type = room:GetType(),
        room_variant = roomVariant,
        room_spawn_seed = roomSpawnSeed,
        room_clear = room:IsClear()
    }
end

local function emit(eventType, payload)
    local normalizedPayload = payload
    if normalizedPayload == nil or next(normalizedPayload) == nil then
        normalizedPayload = json.EMPTY_OBJECT
    end
    sequence = sequence + 1
    local event = {
        schema_version = SCHEMA_VERSION,
        seq = sequence,
        run_id = runId,
        type = eventType,
        game_frame = game:GetFrameCount(),
        context = roomContext(),
        payload = normalizedPayload
    }
    Isaac.DebugString(PREFIX .. json.encode(event))
end

local function readInventory(player)
    local collectibles = {}
    local collectibleConfig = Isaac.GetItemConfig():GetCollectibles()
    for collectibleId = 1, collectibleConfig.Size - 1 do
        local count = player:GetCollectibleNum(collectibleId)
        if count > 0 then
            table.insert(collectibles, { id = collectibleId, count = count })
        end
    end

    return {
        collectible_count = player:GetCollectibleCount(),
        collectibles = collectibles,
        active_item = player:GetActiveItem(),
        active_charge = player:GetActiveCharge(),
        battery_charge = player:GetBatteryCharge(),
        trinkets = { player:GetTrinket(0), player:GetTrinket(1) },
        cards = { player:GetCard(0), player:GetCard(1) },
        pills = { player:GetPill(0), player:GetPill(1) }
    }
end

local function readPlayer(player, includeInventory)
    local state = {
        controller_index = player.ControllerIndex,
        init_seed = player.InitSeed,
        player_type = player:GetPlayerType(),
        dead = player:IsDead(),
        health = {
            red_hearts = player:GetHearts(),
            max_red_hearts = player:GetMaxHearts(),
            soul_hearts = player:GetSoulHearts(),
            black_heart_mask = player:GetBlackHearts(),
            eternal_hearts = player:GetEternalHearts(),
            golden_hearts = player:GetGoldenHearts(),
            bone_hearts = player:GetBoneHearts()
        },
        resources = {
            coins = player:GetNumCoins(),
            keys = player:GetNumKeys(),
            bombs = player:GetNumBombs()
        }
    }
    if includeInventory then
        state.inventory = readInventory(player)
    end
    return state
end

local function readPlayers(includeInventory)
    local players = {}
    for playerIndex = 0, game:GetNumPlayers() - 1 do
        table.insert(players, readPlayer(Isaac.GetPlayer(playerIndex), includeInventory))
    end
    return players
end

local function playerFingerprint(player)
    return table.concat({
        player:GetPlayerType(),
        player:GetHearts(),
        player:GetMaxHearts(),
        player:GetSoulHearts(),
        player:GetBlackHearts(),
        player:GetEternalHearts(),
        player:GetGoldenHearts(),
        player:GetBoneHearts(),
        player:GetNumCoins(),
        player:GetNumKeys(),
        player:GetNumBombs(),
        player:GetCollectibleCount(),
        player:GetActiveItem(),
        player:GetActiveCharge(),
        player:GetBatteryCharge(),
        player:GetTrinket(0),
        player:GetTrinket(1),
        player:GetCard(0),
        player:GetCard(1),
        player:GetPill(0),
        player:GetPill(1),
        player:IsDead() and 1 or 0
    }, "|")
end

local function inventoryFingerprint(player)
    return table.concat({
        player:GetCollectibleCount(),
        player:GetActiveItem(),
        player:GetActiveCharge(),
        player:GetBatteryCharge(),
        player:GetTrinket(0),
        player:GetTrinket(1),
        player:GetCard(0),
        player:GetCard(1),
        player:GetPill(0),
        player:GetPill(1)
    }, "|")
end

local function collectibleMap(inventory)
    local result = {}
    for _, entry in ipairs(inventory.collectibles) do
        result[entry.id] = entry.count
    end
    return result
end

local function emitInventoryChanges(player, playerIndex, playerState)
    local inventory = playerState.inventory
    local previous = lastInventories[playerIndex] or {}
    local current = collectibleMap(inventory)

    emit("inventory_changed", { player = playerState })
    for collectibleId, count in pairs(current) do
        local previousCount = previous[collectibleId] or 0
        if count > previousCount then
            emit("collectible_taken", {
                controller_index = player.ControllerIndex,
                collectible_id = collectibleId,
                count_added = count - previousCount,
                total_count = count
            })
        end
    end
    lastInventories[playerIndex] = current
    lastInventoryFingerprints[playerIndex] = inventoryFingerprint(player)
end

local function emitSnapshot(reason)
    local players = readPlayers(true)
    emit("state_snapshot", { reason = reason, players = players })
    for playerIndex, playerState in ipairs(players) do
        local zeroBasedIndex = playerIndex - 1
        local player = Isaac.GetPlayer(zeroBasedIndex)
        lastInventories[zeroBasedIndex] = collectibleMap(playerState.inventory)
        lastInventoryFingerprints[zeroBasedIndex] = inventoryFingerprint(player)
        lastPlayerFingerprints[zeroBasedIndex] = playerFingerprint(player)
    end
end

local function pollPlayers()
    for playerIndex = 0, game:GetNumPlayers() - 1 do
        local player = Isaac.GetPlayer(playerIndex)
        local fingerprint = playerFingerprint(player)
        if lastPlayerFingerprints[playerIndex] ~= fingerprint then
            local playerState = readPlayer(player, true)
            lastPlayerFingerprints[playerIndex] = fingerprint
            emit("player_state_changed", { player = playerState })

            if lastInventoryFingerprints[playerIndex] ~= inventoryFingerprint(player) then
                emitInventoryChanges(player, playerIndex, playerState)
            end
        end

        local dead = player:IsDead()
        if dead and not deathReported[playerIndex] then
            deathReported[playerIndex] = true
            emit("death", { controller_index = player.ControllerIndex })
        elseif not dead then
            deathReported[playerIndex] = false
        end
    end
end

local function startRun(continued)
    sequence = 0
    runId = game:GetSeeds():GetStartSeedString() .. ":" .. tostring(game:GetFrameCount())
    activeRun = true
    lastHeartbeatFrame = -1
    lastSnapshotFrame = -1
    lastPlayerPollFrame = -1
    lastPlayerFingerprints = {}
    lastInventoryFingerprints = {}
    lastInventories = {}
    deathReported = {}
    emit("run_started", { continued = continued, players = readPlayers(true) })
    emitSnapshot("run_started")
end

local function endRun(reason, gameOver, shouldSave)
    if not activeRun then
        return
    end
    emit("run_ended", {
        reason = reason,
        game_over = gameOver,
        should_save = shouldSave
    })
    activeRun = false
end

local function onPostUpdate()
    if not activeRun then
        return
    end

    local frame = game:GetFrameCount()
    if lastPlayerPollFrame < 0 or frame - lastPlayerPollFrame >= PLAYER_POLL_FRAMES then
        lastPlayerPollFrame = frame
        pollPlayers()
    end
    if lastHeartbeatFrame < 0 or frame - lastHeartbeatFrame >= HEARTBEAT_FRAMES then
        lastHeartbeatFrame = frame
        emit("heartbeat", {})
    end
    if lastSnapshotFrame < 0 or frame - lastSnapshotFrame >= SNAPSHOT_FRAMES then
        lastSnapshotFrame = frame
        emitSnapshot("periodic")
    end
end

local function onNewRoom()
    if not activeRun then
        return
    end
    emit("room_entered", {})
    if game:GetRoom():GetType() == RoomType.ROOM_BOSS then
        emit("boss_started", {})
    end
    emitSnapshot("room_entered")
end

local function onNewLevel()
    if activeRun then
        emit("floor_changed", {})
        emitSnapshot("floor_changed")
    end
end

local function onRoomCleared()
    if not activeRun then
        return
    end
    emit("room_cleared", {})
    if game:GetRoom():GetType() == RoomType.ROOM_BOSS then
        emit("boss_defeated", {})
    end
end

local function onPickupInit(_, pickup)
    if activeRun and pickup.Variant == PickupVariant.PICKUP_COLLECTIBLE then
        emit("collectible_spawned", {
            collectible_id = pickup.SubType,
            init_seed = pickup.InitSeed,
            price = pickup.Price,
            shop_item_id = pickup.ShopItemId
        })
    end
end

runId = "boot:" .. tostring(game:GetFrameCount())
emit("bridge_ready", {
    bridge_version = BRIDGE_VERSION,
    game_version_probe = "Repentance+ J460"
})

Oriens:AddCallback(ModCallbacks.MC_POST_GAME_STARTED, function(_, continued)
    startRun(continued)
end)
Oriens:AddCallback(ModCallbacks.MC_POST_UPDATE, onPostUpdate)
Oriens:AddCallback(ModCallbacks.MC_POST_NEW_ROOM, onNewRoom)
Oriens:AddCallback(ModCallbacks.MC_POST_NEW_LEVEL, onNewLevel)
Oriens:AddCallback(ModCallbacks.MC_PRE_SPAWN_CLEAN_AWARD, onRoomCleared)
Oriens:AddCallback(ModCallbacks.MC_POST_PICKUP_INIT, onPickupInit)
Oriens:AddCallback(ModCallbacks.MC_POST_GAME_END, function(_, gameOver)
    endRun("game_end", gameOver, nil)
end)
Oriens:AddCallback(ModCallbacks.MC_PRE_GAME_EXIT, function(_, shouldSave)
    endRun("game_exit", nil, shouldSave)
end)
