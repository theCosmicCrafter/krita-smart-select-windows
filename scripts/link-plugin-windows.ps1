<#
.SYNOPSIS
    Wire the Smart Select plugin source into a Krita source tree.

.DESCRIPTION
    Krita builds native tool plugins from inside its own source tree. This script
    does what scripts/build-krita-5.3.1.sh does on Linux, without needing
    administrator rights:

      1. creates a directory junction
         <KritaSrc>\plugins\tools\krita-smart-select -> <repo>\plugin
      2. adds `add_subdirectory( krita-smart-select )` after tool_knife in
         <KritaSrc>\plugins\tools\CMakeLists.txt (original saved as
         CMakeLists.txt.orig, line endings preserved)

    Running it again is safe: both steps are skipped when already done.

.PARAMETER KritaSrc
    Path to the Krita source tree.
#>
param(
    [Parameter(Mandatory)]
    [string]$KritaSrc
)

$ErrorActionPreference = "Stop"

$RootDir = (Resolve-Path "$PSScriptRoot\..").Path
$PluginSrc = Join-Path $RootDir "plugin"
$ToolsDir = Join-Path $KritaSrc "plugins\tools"
$ToolsCmake = Join-Path $ToolsDir "CMakeLists.txt"
$Link = Join-Path $ToolsDir "krita-smart-select"

if (-not (Test-Path $ToolsCmake)) {
    throw "Not a Krita source tree (missing $ToolsCmake)"
}

# 1. Junction to the plugin sources
if (Test-Path $Link) {
    $item = Get-Item $Link -Force
    $target = @($item.Target) | Select-Object -First 1
    if ($item.LinkType -ne "Junction" -or -not $target -or
        (Resolve-Path $target).Path -ne $PluginSrc) {
        throw "$Link exists but is not a junction to $PluginSrc"
    }
    Write-Host "## Junction already in place: $Link"
} else {
    New-Item -ItemType Junction -Path $Link -Target $PluginSrc | Out-Null
    Write-Host "## Created junction: $Link -> $PluginSrc"
}

# 2. add_subdirectory in plugins/tools/CMakeLists.txt
$text = [IO.File]::ReadAllText($ToolsCmake)
if ($text -match 'add_subdirectory\(\s*krita-smart-select\s*\)') {
    Write-Host "## CMakeLists.txt already includes krita-smart-select"
} else {
    $pattern = [regex]'(add_subdirectory\( tool_knife \)(\r?\n))'
    if (-not $pattern.IsMatch($text)) {
        throw "Could not find the tool_knife insertion point in $ToolsCmake"
    }
    $backup = "$ToolsCmake.orig"
    if (-not (Test-Path $backup)) { Copy-Item $ToolsCmake $backup }
    $patched = $pattern.Replace($text, '$1add_subdirectory( krita-smart-select )$2', 1)
    [IO.File]::WriteAllText($ToolsCmake, $patched, (New-Object Text.UTF8Encoding $false))
    Write-Host "## Added krita-smart-select to $ToolsCmake"
}
