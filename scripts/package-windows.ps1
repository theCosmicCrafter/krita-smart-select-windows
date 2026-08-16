<#
.SYNOPSIS
    Package the Windows release ZIP for Krita Smart Select.

.DESCRIPTION
    Assembles the installable plugin layout:
      smart_select/
        __init__.py
        extension.py
        worker_launcher.py
        lib/
          kritasmartselect.dll
          libc++.dll
          libunwind.dll
          libwinpthread-1.dll

.PARAMETER Version
    Version string for the output archive name (e.g. 0.1.1).

.PARAMETER BuildDir
    Path to the CMake build directory containing bin\kritasmartselect.dll.

.PARAMETER RuntimeDlls
    Path to the directory containing libc++.dll, libunwind.dll, libwinpthread-1.dll.
    Defaults to the LLVM-MinGW x86_64-w64-mingw32\bin directory.

.EXAMPLE
    .\scripts\package-windows.ps1 -Version 0.1.1 -BuildDir ..\build-windows
#>
param(
    [Parameter(Mandatory)]
    [string]$Version,

    [Parameter(Mandatory)]
    [string]$BuildDir,

    [string]$RuntimeDlls
)

$ErrorActionPreference = "Stop"

$RootDir = Resolve-Path "$PSScriptRoot\.."
$DistDir = Join-Path $RootDir "dist"
$StageDir = Join-Path $DistDir "smart_select"

# Clean and create staging
if (Test-Path $StageDir) { Remove-Item -Recurse -Force $StageDir }
New-Item -ItemType Directory -Path $StageDir -Force | Out-Null
New-Item -ItemType Directory -Path "$StageDir\lib" -Force | Out-Null

# Copy Python files
Copy-Item "$RootDir\python\smart_select\__init__.py" $StageDir
Copy-Item "$RootDir\python\smart_select\extension.py" $StageDir
Copy-Item "$RootDir\python\smart_select\worker_launcher.py" $StageDir

# Copy native DLL
$NativeDll = Join-Path $BuildDir "bin\kritasmartselect.dll"
if (-not (Test-Path $NativeDll)) {
    Write-Error "Native DLL not found: $NativeDll"
    exit 1
}
Copy-Item $NativeDll "$StageDir\lib\"

# Copy Clang runtime DLLs
if (-not $RuntimeDlls) {
    $RuntimeDlls = $env:LLVM_MINGW + "\x86_64-w64-mingw32\bin"
}

foreach ($dll in @("libc++.dll", "libunwind.dll", "libwinpthread-1.dll")) {
    $src = Join-Path $RuntimeDlls $dll
    if (Test-Path $src) {
        Copy-Item $src "$StageDir\lib\"
    } else {
        Write-Warning "Runtime DLL not found: $src"
    }
}

# Copy action file
$ActionDir = Join-Path $DistDir "actions"
New-Item -ItemType Directory -Path $ActionDir -Force | Out-Null
Copy-Item "$RootDir\plugin\SmartSelectTool.action" $ActionDir

# Create ZIP
$ZipName = "krita-smart-select-windows-x64-$Version.zip"
$ZipPath = Join-Path $DistDir $ZipName
if (Test-Path $ZipPath) { Remove-Item $ZipPath }
Compress-Archive -Path "$StageDir\*" -DestinationPath $ZipPath

Write-Host "## Package created: $ZipPath"
Write-Host "## Contents:"
Get-ChildItem -Recurse $StageDir | ForEach-Object { Write-Host "   $($_.FullName.Replace($StageDir, 'smart_select'))" }
