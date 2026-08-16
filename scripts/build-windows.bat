@echo off
REM Build script for Krita Smart Select on Windows.
REM
REM Requires:
REM   - LLVM-MinGW Clang 21+ (https://github.com/mstorsjo/llvm-mingw/releases)
REM   - KDE Craft with Qt5 dependencies (krita-deps-management)
REM   - CMake 3.21+ and Ninja
REM
REM Usage:
REM   set KRITA_SRC=C:\path\to\krita-source
REM   set KRITA_INSTALL=F:\krita-work\krita\_install
REM   set LLVM_MINGW=F:\krita-work\llvm-mingw\llvm-mingw-20251216-ucrt-x86_64
REM   scripts\build-windows.bat

setlocal EnableDelayedExpansion

if "%KRITA_SRC%"=="" (
    echo ## ERROR: KRITA_SRC is not set. Point it to the Krita source tree.
    exit /b 1
)
if "%KRITA_INSTALL%"=="" (
    echo ## ERROR: KRITA_INSTALL is not set. Point it to the Krita deps install prefix.
    exit /b 1
)
if "%LLVM_MINGW%"=="" (
    echo ## ERROR: LLVM_MINGW is not set. Point it to the llvm-mingw toolchain root.
    exit /b 1
)

set "BUILD_DIR=%~dp0..\build-windows"

set "PATH=%LLVM_MINGW%\bin;%LLVM_MINGW%\x86_64-w64-mingw32\bin;%PATH%"

set "CC=clang"
set "CXX=clang++"

echo ## Configuring Smart Select for Windows...
cmake -S "%KRITA_SRC%" -B "%BUILD_DIR%" -G Ninja ^
    -DCMAKE_BUILD_TYPE=Release ^
    -DBUILD_WITH_QT6=OFF ^
    -DCMAKE_PREFIX_PATH="%KRITA_INSTALL%" ^
    -DCMAKE_INSTALL_PREFIX="%KRITA_INSTALL%" ^
    -DCMAKE_C_COMPILER=clang ^
    -DCMAKE_CXX_COMPILER=clang++
if errorlevel 1 (
    echo ## CMake configure failed
    exit /b 1
)

echo ## Building kritasmartselect.dll...
cmake --build "%BUILD_DIR%" --target kritasmartselect --parallel %BUILD_JOBS%
if errorlevel 1 (
    echo ## Build failed
    exit /b 1
)

echo ## Build succeeded: %BUILD_DIR%\bin\kritasmartselect.dll
