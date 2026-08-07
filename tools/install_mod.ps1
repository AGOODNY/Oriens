param(
    [string]$GamePath = "E:\SteamLibrary\steamapps\common\The Binding of Isaac Rebirth",
    [switch]$Force
)

$ErrorActionPreference = "Stop"
$source = Join-Path (Split-Path -Parent $PSScriptRoot) "mod\oriens"
$modsPath = Join-Path $GamePath "mods"
$destination = Join-Path $modsPath "oriens"

if (-not (Test-Path -LiteralPath $source)) {
    throw "Mod source directory not found: $source"
}
if (-not (Test-Path -LiteralPath $modsPath)) {
    throw "Game mod directory not found: $modsPath"
}
if ((Test-Path -LiteralPath $destination) -and -not $Force) {
    throw "Destination already exists: $destination. Use -Force only for an existing Oriens installation."
}

New-Item -ItemType Directory -Path $destination -Force:$Force | Out-Null
Copy-Item -LiteralPath (Join-Path $source "main.lua") -Destination $destination -Force:$Force
Copy-Item -LiteralPath (Join-Path $source "metadata.xml") -Destination $destination -Force:$Force

Write-Output "Oriens mod installed at: $destination"
