<#
.SYNOPSIS
    Package the Windows release ZIP for Krita Smart Select.

.DESCRIPTION
    Assembles the layout Krita's "Import Python Plugin from File..." expects, the
    same one scripts/package-linux.sh produces:

      smart_select.desktop
      smart_select.action
      smart_select/
        __init__.py
        extension.py
        worker_launcher.py
        worker_launcher.cmd
        worker/
          *.py
        lib/
          kritasmartselect.dll

    The native plugin resolves libc++.dll, libunwind.dll and the Qt/KF5/Krita DLLs
    from the Krita installation that loads it, so the toolchain runtime is not
    bundled unless -RuntimeDlls is given.

.PARAMETER Version
    Version string for the output archive name (e.g. 0.1.1).

.PARAMETER BuildDir
    CMake build directory that contains bin\kritasmartselect.dll.

.PARAMETER RuntimeDlls
    Optional directory with libc++.dll, libunwind.dll and libwinpthread-1.dll
    (for example <llvm-mingw>\x86_64-w64-mingw32\bin). When set, those DLLs are
    copied next to kritasmartselect.dll.

.EXAMPLE
    .\scripts\package-windows.ps1 -Version 0.1.1 -BuildDir .\build-windows
#>
param(
    [Parameter(Mandatory)]
    [string]$Version,

    [Parameter(Mandatory)]
    [string]$BuildDir,

    [string]$RuntimeDlls
)

$ErrorActionPreference = "Stop"
Add-Type -AssemblyName System.IO.Compression
Add-Type -AssemblyName System.IO.Compression.FileSystem

$RootDir = (Resolve-Path "$PSScriptRoot\..").Path
$DistDir = Join-Path $RootDir "dist"
$StageRoot = Join-Path $DistDir "package-windows"
$StageDir = Join-Path $StageRoot "smart_select"
$LibDir = Join-Path $StageDir "lib"

$NativeDll = Join-Path $BuildDir "bin\kritasmartselect.dll"
if (-not (Test-Path $NativeDll)) {
    throw "Native DLL not found: $NativeDll"
}

# Clean and create staging
if (Test-Path $StageRoot) { Remove-Item -Recurse -Force $StageRoot }
New-Item -ItemType Directory -Path $LibDir -Force | Out-Null
New-Item -ItemType Directory -Path "$StageDir\worker" -Force | Out-Null

# Python loader package and worker
Copy-Item "$RootDir\python\smart_select.desktop" $StageRoot
Copy-Item "$RootDir\python\smart_select\*.py" $StageDir
Copy-Item "$RootDir\python\smart_select\*.cmd" $StageDir
Copy-Item "$RootDir\worker\*.py" "$StageDir\worker"
# Krita's importer copies "<module>.action" from the archive into the user's actions
# directory (the file name only has to match the module, Krita reads every *.action).
Copy-Item "$RootDir\plugin\SmartSelectTool.action" "$StageRoot\smart_select.action"

# Native plugin
Copy-Item $NativeDll $LibDir

# Optional toolchain runtime
if ($RuntimeDlls) {
    foreach ($dll in @("libc++.dll", "libunwind.dll", "libwinpthread-1.dll")) {
        $src = Join-Path $RuntimeDlls $dll
        if (-not (Test-Path $src)) { throw "Runtime DLL not found: $src" }
        Copy-Item $src $LibDir
    }
}

# Create ZIP with forward-slash entry names (Compress-Archive in Windows
# PowerShell 5.1 writes backslashes, which other unzip tools do not understand).
$ZipPath = Join-Path $DistDir "krita-smart-select-windows-x64-$Version.zip"
if (Test-Path $ZipPath) { Remove-Item $ZipPath }

$zip = [System.IO.Compression.ZipFile]::Open($ZipPath, [System.IO.Compression.ZipArchiveMode]::Create)
try {
    # Krita's plugin importer finds the module through its directory entry
    # ("smart_select/"), so the archive must contain explicit directory entries.
    $dirs = Get-ChildItem -Recurse -Directory $StageRoot | Sort-Object FullName
    foreach ($dir in $dirs) {
        $dirEntry = $dir.FullName.Substring($StageRoot.Length + 1).Replace("\", "/") + "/"
        $zip.CreateEntry($dirEntry) | Out-Null
    }
    $files = Get-ChildItem -Recurse -File $StageRoot | Sort-Object FullName
    foreach ($file in $files) {
        $entryName = $file.FullName.Substring($StageRoot.Length + 1).Replace("\", "/")
        [System.IO.Compression.ZipFileExtensions]::CreateEntryFromFile(
            $zip, $file.FullName, $entryName, [System.IO.Compression.CompressionLevel]::Optimal
        ) | Out-Null
    }
} finally {
    $zip.Dispose()
}

# Check the archive the way Krita's plugin importer (plugin_importer.py) reads it:
# the module named by X-KDE-Library must exist as a "<name>/" directory entry that
# contains __init__.py.
$zip = [System.IO.Compression.ZipFile]::OpenRead($ZipPath)
try {
    $names = @($zip.Entries | ForEach-Object { $_.FullName })
    $desktop = $zip.Entries | Where-Object { $_.FullName -like "*.desktop" } | Select-Object -First 1
    if (-not $desktop) { throw "No .desktop file in $ZipPath" }
    $reader = New-Object System.IO.StreamReader($desktop.Open())
    try { $desktopText = $reader.ReadToEnd() } finally { $reader.Dispose() }
    if ($desktopText -notmatch '(?m)^X-KDE-Library=(.+?)\s*$') {
        throw "$($desktop.FullName) has no X-KDE-Library key"
    }
    $module = $Matches[1]
    if ($names -notcontains "$module/") { throw "Archive has no '$module/' directory entry" }
    if ($names -notcontains "$module/__init__.py") { throw "Archive has no '$module/__init__.py'" }
    if ($names -notcontains "$module.action") { throw "Archive has no '$module.action'" }
} finally {
    $zip.Dispose()
}

Write-Host "## Package created: $ZipPath"
Write-Host "## SHA256: $((Get-FileHash $ZipPath -Algorithm SHA256).Hash.ToLower())"
Write-Host "## Contents:"
$zip = [System.IO.Compression.ZipFile]::OpenRead($ZipPath)
try {
    $zip.Entries | ForEach-Object { Write-Host ("   {0,10}  {1}" -f $_.Length, $_.FullName) }
} finally {
    $zip.Dispose()
}
