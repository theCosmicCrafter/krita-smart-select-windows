@echo off
REM Build script for Krita Smart Select on Windows.
REM
REM Builds kritasmartselect.dll inside a Krita 5.3.x (Qt5) source tree against
REM KDE's pre-built dependency cache, using LLVM-MinGW Clang with the UCRT
REM runtime (the same CRT and compiler generation as the official Krita build).
REM
REM Required environment:
REM   KRITA_SRC      Krita source tree (the one that builds as 5.3.x with Qt5)
REM   KRITA_INSTALL  Krita deps prefix, e.g. <work>\krita\_install
REM   LLVM_MINGW     llvm-mingw root, ucrt flavour, Clang 21
REM
REM Optional environment:
REM   CMAKE_BIN      directory with cmake.exe, when it is not in PATH
REM   NINJA_BIN      directory with ninja.exe, when it is not in PATH
REM   KRITA_BUILD    build directory (default: <repo>\build-windows)
REM   BUILD_TYPE     CMake build type (default: Release)
REM   BUILD_JOBS     parallel jobs (default: NUMBER_OF_PROCESSORS)
REM
REM Usage:
REM   set KRITA_SRC=G:\laboratory\krita-6.0.4
REM   set KRITA_INSTALL=G:\laboratory\krita-build\work\krita\_install
REM   set LLVM_MINGW=G:\laboratory\krita-build\toolchain-ucrt\llvm-mingw-20251216-ucrt-x86_64
REM   scripts\build-windows.bat

setlocal EnableExtensions

if "%KRITA_SRC%"=="" goto :missing_src
if "%KRITA_INSTALL%"=="" goto :missing_install
if "%LLVM_MINGW%"=="" goto :missing_llvm

if not exist "%KRITA_SRC%\CMakeLists.txt" goto :bad_src
if not exist "%KRITA_INSTALL%\lib\cmake\Qt5Core" goto :bad_install
if not exist "%LLVM_MINGW%\bin\clang++.exe" goto :bad_llvm

set "ROOT_DIR=%~dp0.."
if "%KRITA_BUILD%"=="" set "KRITA_BUILD=%ROOT_DIR%\build-windows"
if "%BUILD_TYPE%"=="" set "BUILD_TYPE=Release"
if "%BUILD_JOBS%"=="" set "BUILD_JOBS=%NUMBER_OF_PROCESSORS%"

REM Same environment the KDE CI sets up for a dependency prefix.
set "PATH=%LLVM_MINGW%\bin;%KRITA_INSTALL%\bin;%PATH%"
if defined NINJA_BIN set "PATH=%NINJA_BIN%;%PATH%"
if defined CMAKE_BIN set "PATH=%CMAKE_BIN%;%PATH%"
set "CMAKE_PREFIX_PATH=%KRITA_INSTALL%"
set "PKG_CONFIG_PATH=%KRITA_INSTALL%\lib\pkgconfig;%KRITA_INSTALL%\share\pkgconfig"
set "CC=clang"
set "CXX=clang++"

where cmake >nul 2>nul || goto :no_cmake
where ninja >nul 2>nul || goto :no_ninja

echo ## Wiring the plugin into the Krita source tree...
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0link-plugin-windows.ps1" -KritaSrc "%KRITA_SRC%"
if errorlevel 1 goto :link_failed

echo ## Configuring Krita with Smart Select...
cmake -S "%KRITA_SRC%" -B "%KRITA_BUILD%" -G Ninja ^
    -DCMAKE_BUILD_TYPE=%BUILD_TYPE% ^
    -DBUILD_WITH_QT6=OFF ^
    -DBUILD_TESTING=OFF ^
    -DKRITA_ENABLE_PCH=OFF ^
    -DHIDE_SAFE_ASSERTS=ON ^
    -DCMAKE_PREFIX_PATH="%KRITA_INSTALL%" ^
    -DCMAKE_INSTALL_PREFIX="%KRITA_BUILD%\_install" ^
    -DCMAKE_C_COMPILER=clang ^
    -DCMAKE_CXX_COMPILER=clang++
if errorlevel 1 goto :configure_failed

echo ## Building kritasmartselect.dll...
cmake --build "%KRITA_BUILD%" --target kritasmartselect --parallel %BUILD_JOBS%
if errorlevel 1 goto :build_failed

echo ## Build succeeded: %KRITA_BUILD%\bin\kritasmartselect.dll
exit /b 0

:missing_src
echo ## ERROR: KRITA_SRC is not set. Point it to the Krita source tree.
exit /b 1
:missing_install
echo ## ERROR: KRITA_INSTALL is not set. Point it to the Krita deps install prefix.
exit /b 1
:missing_llvm
echo ## ERROR: LLVM_MINGW is not set. Point it to the llvm-mingw toolchain root.
exit /b 1
:bad_src
echo ## ERROR: KRITA_SRC does not look like a Krita source tree: %KRITA_SRC%
exit /b 1
:bad_install
echo ## ERROR: KRITA_INSTALL has no Qt5 CMake config: %KRITA_INSTALL%
exit /b 1
:bad_llvm
echo ## ERROR: clang++.exe not found in LLVM_MINGW\bin: %LLVM_MINGW%
exit /b 1
:no_cmake
echo ## ERROR: cmake.exe not found. Add it to PATH or set CMAKE_BIN.
exit /b 1
:no_ninja
echo ## ERROR: ninja.exe not found. Add it to PATH or set NINJA_BIN.
exit /b 1
:link_failed
echo ## ERROR: could not wire the plugin into the Krita source tree
exit /b 1
:configure_failed
echo ## ERROR: CMake configure failed
exit /b 1
:build_failed
echo ## ERROR: build failed
exit /b 1
