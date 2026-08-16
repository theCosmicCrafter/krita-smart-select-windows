# Building Krita Smart Select on Windows

This document describes how to build the native `kritasmartselect.dll` on Windows
using LLVM-MinGW Clang and KDE's pre-built Qt5 dependency cache.

## Prerequisites

### 1. LLVM-MinGW Clang 21+

The KDE CI dependencies for Krita Windows/Qt5 are built with LLVM-MinGW Clang.
You must use the same compiler ABI to link against them.

Download from [llvm-mingw releases](https://github.com/mstorsjo/llvm-mingw/releases):
- File: `llvm-mingw-<date>-ucrt-x86_64.zip`
- Extract to e.g. `F:\krita-work\llvm-mingw\llvm-mingw-<date>-ucrt-x86_64\`

### 2. KDE Craft + Krita Dependencies

The Krita Windows dependencies (Qt5, KF5, boost, exiv2, etc.) are fetched from
KDE's CI artifact cache using `krita-deps-management`.

```powershell
# Clone krita-deps-management and ci-utilities
cd C:\path\to\krita-source
git clone https://invent.kde.org/packaging/krita-deps-management.git
git clone https://invent.kde.org/packaging/krita-ci-utilities.git krita-deps-management/ci-utilities

# Create a Python venv and install deps
python -m venv venv
venv\Scripts\pip install -r krita-deps-management\requirements.txt

# Generate .kde-ci.yml
venv\Scripts\python krita-deps-management\tools\replace-branch-in-seed-file.py `
    krita-deps-management\latest\krita-deps.yml -p `
    -o branch-corrected-deps.yml -d master
venv\Scripts\python krita-deps-management\tools\generate-deps-file.py `
    -s branch-corrected-deps.yml -o .kde-ci.yml

# Fetch dependencies (downloads ~2GB, unpacks to ~5GB)
$env:KDECI_CACHE_PATH = "F:\krita-cache"
$env:KDECI_WORKDIR_PATH = "F:\krita-work"
$env:KDECI_PACKAGE_PROJECT = "teams/ci-artifacts/krita-windows"
$env:KDECI_GITLAB_SERVER = "https://invent.kde.org/"
$env:KDECI_REPO_METADATA_PATH = "krita-deps-management\repo-metadata"
$env:PYTHONUTF8 = "1"
venv\Scripts\python krita-deps-management\ci-utilities\run-ci-build.py `
    --project krita --branch master --platform "Windows/Qt5/Shared" `
    --only-deps --skip-publishing
```

This installs dependencies to `F:\krita-work\krita\_install\`.

### 3. CMake and Ninja

CMake 3.21+ and Ninja are required. KDE Craft includes both:
- `F:\CraftRoot\dev-utils\cmake-base\bin\cmake.exe`
- `F:\CraftRoot\dev-utils\bin\ninja.exe`

### 4. Krita Source

Download Krita 5.3.x source from
[download.kde.org](https://download.kde.org/stable/krita/). The Smart Select
plugin source goes in `plugins/tools/krita-smart-select/`.

## Building

```powershell
$env:KRITA_SRC = "C:\path\to\krita-source"
$env:KRITA_INSTALL = "F:\krita-work\krita\_install"
$env:LLVM_MINGW = "F:\krita-work\llvm-mingw\llvm-mingw-20251216-ucrt-x86_64"
$env:BUILD_JOBS = "8"

scripts\build-windows.bat
```

This produces `build-windows\bin\kritasmartselect.dll`.

## Packaging

```powershell
$env:LLVM_MINGW = "F:\krita-work\llvm-mingw\llvm-mingw-20251216-ucrt-x86_64"
.\scripts\package-windows.ps1 -Version 0.1.1 -BuildDir .\build-windows
```

This produces `dist\krita-smart-select-windows-x64-0.1.1.zip`.

## Runtime DLLs

The native plugin is built with Clang and links against `libc++.dll`,
`libunwind.dll`, and `libwinpthread-1.dll`. These must be shipped alongside
`kritasmartselect.dll` in the `lib/` directory of the plugin package.

The packaging script copies them from the LLVM-MinGW toolchain automatically.

## Installation

1. Download the Windows release ZIP.
2. In Krita: **Tools → Scripts → Import Python Plugin**.
3. Select the ZIP file.
4. Restart Krita.
5. The Smart Select tool appears in the toolbox under the Select section.
