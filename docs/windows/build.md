# Building Krita Smart Select on Windows

This document describes how to build the native `kritasmartselect.dll` and the
installable release ZIP on Windows, using LLVM-MinGW Clang and KDE's pre-built
Qt5 dependency cache.

The plugin is a native Krita tool, so it is compiled inside a Krita source tree
and must match the Krita build that will load it. Krita for Windows (Qt5) is built
with LLVM-MinGW Clang 21 and the **UCRT** runtime, and ships its own `libc++.dll`.
The plugin has to use the same CRT and compiler generation.

## Prerequisites

### 1. LLVM-MinGW Clang 21, UCRT flavour

Download `llvm-mingw-20251216-ucrt-x86_64.zip` (Clang 21.1.8) from
[llvm-mingw releases](https://github.com/mstorsjo/llvm-mingw/releases) and extract
it into its own directory.

Do not use the `msvcrt` flavour or a newer LLVM: Krita's binaries import
`api-ms-win-crt-*.dll` (UCRT) and the plugin is loaded into that process. You can
check a toolchain with:

```powershell
<llvm-mingw>\bin\clang.exe --version     # clang version 21.x
```

### 2. Krita source tree

Use a Krita source tree that builds as the same version as the installed Krita.
With `BUILD_WITH_QT6=OFF` (the default) the `release/6.0.x` sources build as
Krita 5.3.x; for example the `v6.0.4.1` tag is what an installed Krita
`5.3.4.1 (git bd6e8d0)` was built from. The plugin tool API version is `28`.

The build script wires the plugin into this tree, see
[Building](#building).

### 3. KDE dependencies (Qt5, KF5, boost, ...)

The dependencies are fetched from KDE's CI artifact cache with
`krita-deps-management` and `krita-ci-utilities` (about 83 packages, 1.6 GB
unpacked). Use a separate working directory so Krita's own `.kde-ci.yml` is not
overwritten.

```powershell
cd <krita-build>
git clone https://invent.kde.org/packaging/krita-deps-management.git
git clone https://invent.kde.org/packaging/krita-ci-utilities.git krita-deps-management\ci-utilities

python -m venv venv
venv\Scripts\python -m pip install -r krita-deps-management\requirements.txt

mkdir ci, cache, work
cd ci
$env:PYTHONUTF8 = "1"
..\venv\Scripts\python ..\krita-deps-management\tools\replace-branch-in-seed-file.py `
    ..\krita-deps-management\latest\krita-deps.yml -p -o branch-corrected-deps.yml -d master
..\venv\Scripts\python ..\krita-deps-management\tools\generate-deps-file.py `
    -s branch-corrected-deps.yml -o .kde-ci.yml

$env:KDECI_CACHE_PATH = "<krita-build>\cache"
$env:KDECI_WORKDIR_PATH = "<krita-build>\work"
$env:KDECI_PACKAGE_PROJECT = "teams/ci-artifacts/krita-windows"
$env:KDECI_GITLAB_SERVER = "https://invent.kde.org/"
$env:KDECI_REPO_METADATA_PATH = "<krita-build>\krita-deps-management\repo-metadata"
..\venv\Scripts\python -u ..\krita-ci-utilities\run-ci-build.py `
    --project krita --branch master --platform "Windows/Qt5/Shared" `
    --only-deps --skip-publishing
```

The dependencies are unpacked into `<krita-build>\work\krita\_install`; that
directory is `KRITA_INSTALL` below.

### 4. CMake and Ninja

CMake 3.21+ and Ninja. The Visual Studio Build Tools ship both under
`...\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin` and
`...\CMake\Ninja`; they do not need to be on `PATH` (see `CMAKE_BIN` and
`NINJA_BIN`).

## Building

```powershell
$env:KRITA_SRC = "C:\path\to\krita-source"
$env:KRITA_INSTALL = "<krita-build>\work\krita\_install"
$env:LLVM_MINGW = "<krita-build>\toolchain\llvm-mingw-20251216-ucrt-x86_64"
$env:CMAKE_BIN = "<dir with cmake.exe>"     # optional
$env:NINJA_BIN = "<dir with ninja.exe>"     # optional
$env:BUILD_JOBS = "10"                      # optional

scripts\build-windows.bat
```

The script:

1. runs `scripts\link-plugin-windows.ps1`, which creates the junction
   `<KRITA_SRC>\plugins\tools\krita-smart-select` -> `<repo>\plugin` and adds
   `add_subdirectory( krita-smart-select )` after `tool_knife` in
   `plugins\tools\CMakeLists.txt` (original kept as `CMakeLists.txt.orig`;
   both steps are skipped when already done);
2. configures Krita with Ninja and the KDE dependency prefix;
3. builds only the `kritasmartselect` target.

The target pulls in the Krita libraries it links against, so a clean build is
about 1600 compile steps (roughly 35 minutes on 12 cores). Later builds are
incremental. The output is `build-windows\bin\kritasmartselect.dll`; set
`KRITA_BUILD` to use another build directory.

## Packaging

```powershell
.\scripts\package-windows.ps1 -Version 0.1.2 -BuildDir .\build-windows
```

This produces `dist\krita-smart-select-windows-x64-0.1.2.zip` with the layout
Krita's *Import Python Plugin from File...* expects:

```text
smart_select.desktop
smart_select.action
smart_select/
  __init__.py
  extension.py
  worker_launcher.py
  worker_launcher.cmd
  worker/*.py
  lib/kritasmartselect.dll
```

Krita's importer (`plugins/python/plugin_importer/plugin_importer.py`) finds a
module through its **directory entry** in the archive (`smart_select/`) and requires
`smart_select/__init__.py`; an archive with only file entries is rejected with
"no plugins found". `Compress-Archive` and Python's `ZipFile.write` on files do not
write directory entries, so `package-windows.ps1` writes them explicitly and checks
the result. The importer copies `<module>.action` (here `smart_select.action`, the
same content as `plugin\SmartSelectTool.action`) into the Krita `actions` directory;
Krita reads every `*.action` file there.

## Runtime DLLs

`kritasmartselect.dll` imports `libc++.dll` and `libunwind.dll` (plus Qt5, KF5
and Krita libraries, and UCRT through `api-ms-win-crt-*`). All of them are
already loaded by Krita from its own `bin\`, and Windows resolves imports against
already loaded modules first, so the toolchain runtime is not bundled by default.
Pass `-RuntimeDlls <llvm-mingw>\x86_64-w64-mingw32\bin` to `package-windows.ps1`
to copy `libc++.dll`, `libunwind.dll` and `libwinpthread-1.dll` into `lib\`
anyway.

## Checking a build

List the imports and make sure the CRT is UCRT:

```powershell
<llvm-mingw>\bin\llvm-objdump.exe -p build-windows\bin\kritasmartselect.dll | findstr "DLL Name"
```

Run Krita's own importer over the ZIP (it needs Krita's `i18n` builtin stubbed):

```python
import builtins, sys
builtins.i18n = lambda s: s
sys.path.insert(0, r"<krita-src>\plugins\python\plugin_importer")
from plugin_importer import PluginImporter
importer = PluginImporter(
    r"dist\krita-smart-select-windows-x64-0.1.2.zip", r"<empty dir>", lambda p: True
)
print(importer.get_plugin_info())   # must list smart_select, with 'action' set
```

Load the DLL against an installed Krita to catch missing symbols (Windows reports
error 127 when a Krita export is missing):

```python
import ctypes, os
krita_bin = r"C:\Program Files\Krita (x64)\bin"
os.add_dll_directory(krita_bin)
ctypes.CDLL(r"build-windows\bin\kritasmartselect.dll")
```

## Installation

1. Download the Windows release ZIP.
2. In Krita: **Tools -> Scripts -> Import Python Plugin from File...** and select
   the ZIP.
3. Restart Krita.
4. Enable *Smart Select* in the Python Plugin Manager if needed.
5. The Smart Select tool appears in the toolbox in the Select section.

The importer unpacks `pykrita\smart_select.desktop`, `pykrita\smart_select\` and
`actions\smart_select.action` under `%APPDATA%\krita\`; no manual file copy is
needed.

## Worker launch

The native tool starts the worker with `QProcess` as `<program> <worker script>`
and talks NDJSON over its stdio (see [../worker-protocol.md](../worker-protocol.md)).
On Linux the program is `worker_launcher.py`, which `exec`s
`uv run --no-project --python 3.12 python <script>`. That cannot work on Windows:
`CreateProcess` does not start `.py` files, Krita ships no `python.exe`
(`sys.executable` inside Krita is `krita.exe`) and `os.exec*` there starts a new
process and exits the old one, which `QProcess` would report as the worker
finishing.

So on Windows `extension.py` passes `smart_select\worker_launcher.cmd` as the
program. It runs the same `uv run ...` command with the stdio inherited, so the
protocol on stdout, the logs on stderr and the exit code pass straight through,
and sets `PYTHONUTF8=1` because the worker stdio is UTF-8 JSON (the Windows
default would be the ANSI code page, which breaks non-ASCII paths in requests).
No system Python is needed, only `uv` in the `PATH` of the Krita process; without
it the worker exits with code 127 and logs `error.reason=uv_not_found`.
`worker_launcher.py` also forwards the exit code when it is run on Windows.

The Python version in the `.cmd` must match `RUNTIME_PYTHON_VERSION` in
`worker/runtime_manifest.py`; `tests/worker/test_worker_launch.py` checks it and,
on Windows, starts the batch file with a stub worker the way `QProcess` does.

## Known limitations

The Windows worker has not run inside a live Krita yet. Starting it, the NDJSON
round trip, stderr and exit codes were checked with a Qt 5.15 `QProcess` test
program (from the dependency prefix, mirroring `WorkerClient::start`) and with a
stub worker. The packaged real worker, started through `worker_launcher.cmd` by a
driver that talks NDJSON on stdio like `WorkerClient`, bootstrapped its runtime,
loaded BiRefNet and returned a correct 256x256 soft selection mask on the GPU
(GeForce RTX 5070 Ti, driver 610.88: runtime 7.4 GB with `torch 2.11.0+cu128`,
`device.type=cuda`, about 1.2 s for the 2048x2048 inference, IoU 0.9998 against the
drawn ellipse). Only that single synthetic image was run, not a real Krita session.

- `uv` must be in the `PATH` of the Krita process. The first run may download
  Python 3.12 (progress goes to stderr) before the worker answers.
- Stopping the worker (`QProcess::kill`) ends only `cmd.exe`; `uv` and the worker
  exit when their stdin closes, after any running job has finished.
  `QProcess::terminate` only posts `WM_CLOSE`, which a console process does not
  handle, so `WorkerClient::stop` is expected to wait its 1 s timeout before it
  falls back to `kill`.
- The managed runtime bootstrap (`worker/runtime_env.py`) uses the Windows venv
  layout (`venv\Scripts\python.exe`, `Scripts` on `PATH`). The real PyTorch
  install, `import torch` from the worker, the BiRefNet model download and CUDA
  inference were run on Windows only for the cases listed above (the model
  download in a separate earlier run of the same worker). The deep
  `site-packages` paths under `%APPDATA%\krita\...` may still hit the 260
  character path limit unless long paths are enabled.
- GPU generations: Blackwell GPUs (compute capability 10.0+, e.g. RTX 50xx) get
  the `cu128` wheels, everything else `cu124`, see
  [../runtime-env.md](../runtime-env.md#torch-wheel-index). `cu124` was not
  re-run on Windows with a pre-Blackwell GPU, and the `cu128` wheels (`torch`
  2.11 resolves from the `>=2.5.1,<3` range) only ship kernels for `sm_75` and
  newer, which is why they are not used for older GPUs. A GPU the installed torch
  has no kernels for is reported as `InferenceUnsupportedGpu`; there is no CPU
  fallback (CPU inference of the 2048x2048 model takes about 100 s).
- Cancel: during runtime setup the pip/venv process tree is stopped with
  `taskkill /T /F` and the job ends in about half a second (checked with the real
  worker, a stalled pip and a real venv). Cancel during the model download is not
  acted on until the download finishes, on every platform. If the worker itself is
  killed (`QProcess::kill`, Krita crash) a running pip is not stopped by the worker (by
  code reading, not tested).
- The first selection after a runtime is created (or switched to another torch
  index) downloads several GB (about 2.7 GB of wheels for `cu128`, 7.4 GB runtime
  on disk) before the worker answers.
