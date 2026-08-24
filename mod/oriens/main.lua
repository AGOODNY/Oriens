local Oriens = RegisterMod("Oriens Bridge", 1)
local game = Game()
local json = require("json")

local PREFIX = "[ORIENS_EVENT]"
local ERROR_PREFIX = "[ORIENS_BRIDGE_ERROR]"
local SCHEMA_VERSION = 1
local BRIDGE_VERSION = "0.2.0"
local HEARTBEAT_FRAMES = 60
local SNAPSHOT_FRAMES = 300
local PLAYER_POLL_FRAMES = 6
local ROOM_STABLE_FRAMES = 3

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
local pendingRoomEntered = false
local pendingFloorChanged = false
local pendingRoomCleared = false
local pendingSnapshotReason = nil
local pendingCollectibles = {}
local disabledCallbacks = {}

local function roomContext()
    if not activeRun or game:IsPaused() then
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

local function stablePlayers()
    if not activeRun or game:IsPaused() then
        return nil
    end

    local room = game:GetRoom()
    if room == nil or room:GetFrameCount() < ROOM_STABLE_FRAMES then
        return nil
    end

    local playerCount = game:GetNumPlayers()
    if playerCount < 1 then
        return nil
    end

    local players = {}
    for playerIndex = 0, playerCount - 1 do
        local player = game:GetPlayer(playerIndex)
        if player == nil
            or not player:Exists()
            or player.Type ~= EntityType.ENTITY_PLAYER
            or type(player.ControllerIndex) ~= "number"
            or player.ControllerIndex < 0 then
            return nil
        end
        table.insert(players, player)
    end
    return players
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

local function readPlayers(includeInventory, playerEntities)
    local players = {}
    for _, player in ipairs(playerEntities) do
        table.insert(players, readPlayer(player, includeInventory))
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

local function emitSnapshot(reason, playerEntities)
    local players = readPlayers(true, playerEntities)
    emit("state_snapshot", { reason = reason, players = players })
    for playerIndex, playerState in ipairs(players) do
        local zeroBasedIndex = playerIndex - 1
        local player = playerEntities[playerIndex]
        lastInventories[zeroBasedIndex] = collectibleMap(playerState.inventory)
        lastInventoryFingerprints[zeroBasedIndex] = inventoryFingerprint(player)
        lastPlayerFingerprints[zeroBasedIndex] = playerFingerprint(player)
    end
end

local function pollPlayers(playerEntities)
    for oneBasedIndex, player in ipairs(playerEntities) do
        local playerIndex = oneBasedIndex - 1
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
    pendingRoomEntered = false
    pendingFloorChanged = false
    pendingRoomCleared = false
    pendingSnapshotReason = "run_started"
    pendingCollectibles = {}
    disabledCallbacks = {}
    emit("run_started", { continued = continued })
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
    pendingRoomEntered = false
    pendingFloorChanged = false
    pendingRoomCleared = false
    pendingSnapshotReason = nil
    pendingCollectibles = {}
end

local function flushPendingEvents(playerEntities, frame)
    if pendingFloorChanged then
        emit("floor_changed", {})
        pendingFloorChanged = false
    end

    if pendingRoomEntered then
        emit("room_entered", {})
        if game:GetRoom():GetType() == RoomType.ROOM_BOSS then
            emit("boss_started", {})
        end
        pendingRoomEntered = false
    end

    if pendingRoomCleared then
        emit("room_cleared", {})
        if game:GetRoom():GetType() == RoomType.ROOM_BOSS then
            emit("boss_defeated", {})
        end
        pendingRoomCleared = false
    end

    if pendingSnapshotReason ~= nil then
        emitSnapshot(pendingSnapshotReason, playerEntities)
        pendingSnapshotReason = nil
        lastSnapshotFrame = frame
        lastPlayerPollFrame = frame
    end

    for _, payload in ipairs(pendingCollectibles) do
        emit("collectible_spawned", payload)
    end
    pendingCollectibles = {}
end

local function onPostUpdate()
    if not activeRun then
        return
    end

    local frame = game:GetFrameCount()
    local playerEntities = stablePlayers()
    if playerEntities == nil then
        return
    end

    flushPendingEvents(playerEntities, frame)
    if lastPlayerPollFrame < 0 or frame - lastPlayerPollFrame >= PLAYER_POLL_FRAMES then
        lastPlayerPollFrame = frame
        pollPlayers(playerEntities)
    end
    if lastHeartbeatFrame < 0 or frame - lastHeartbeatFrame >= HEARTBEAT_FRAMES then
        lastHeartbeatFrame = frame
        emit("heartbeat", {})
    end
    if lastSnapshotFrame < 0 or frame - lastSnapshotFrame >= SNAPSHOT_FRAMES then
        lastSnapshotFrame = frame
        emitSnapshot("periodic", playerEntities)
    end
end

local function onNewRoom()
    if not activeRun then
        return
    end
    pendingRoomEntered = true
    pendingSnapshotReason = "room_entered"
    lastPlayerPollFrame = -1
end

local function onNewLevel()
    if activeRun then
        pendingFloorChanged = true
        pendingSnapshotReason = "floor_changed"
        lastPlayerPollFrame = -1
    end
end

local function onRoomCleared()
    if not activeRun then
        return
    end
    pendingRoomCleared = true
end

local function onPickupInit(_, pickup)
    if activeRun and pickup.Variant == PickupVariant.PICKUP_COLLECTIBLE then
        table.insert(pendingCollectibles, {
            collectible_id = pickup.SubType,
            init_seed = pickup.InitSeed,
            price = pickup.Price,
            shop_item_id = pickup.ShopItemId
        })
    end
end

local function guarded(callbackName, callback)
    return function(...)
        if disabledCallbacks[callbackName] then
            return
        end
        local ok, err = pcall(callback, ...)
        if not ok then
            disabledCallbacks[callbackName] = true
            Isaac.DebugString(ERROR_PREFIX .. callbackName .. ": " .. tostring(err))
        end
    end
end

runId = "boot:" .. tostring(game:GetFrameCount())
emit("bridge_ready", {
    bridge_version = BRIDGE_VERSION,
    game_version_probe = "Repentance+ J460"
})

Oriens:AddCallback(ModCallbacks.MC_POST_GAME_STARTED, guarded("game_started", function(_, continued)
    startRun(continued)
end))
Oriens:AddCallback(ModCallbacks.MC_POST_UPDATE, guarded("post_update", onPostUpdate))
Oriens:AddCallback(ModCallbacks.MC_POST_NEW_ROOM, guarded("new_room", onNewRoom))
Oriens:AddCallback(ModCallbacks.MC_POST_NEW_LEVEL, guarded("new_level", onNewLevel))
Oriens:AddCallback(ModCallbacks.MC_PRE_SPAWN_CLEAN_AWARD, guarded("room_cleared", onRoomCleared))
Oriens:AddCallback(ModCallbacks.MC_POST_PICKUP_INIT, guarded("pickup_init", onPickupInit))
Oriens:AddCallback(ModCallbacks.MC_POST_GAME_END, guarded("game_end", function(_, gameOver)
    endRun("game_end", gameOver, nil)
end))
Oriens:AddCallback(ModCallbacks.MC_PRE_GAME_EXIT, guarded("game_exit", function(_, shouldSave)
    endRun("game_exit", nil, shouldSave)
end))
