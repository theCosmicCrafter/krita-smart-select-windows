@echo off
rem Windows entry point for the Smart Select worker.
rem
rem The plugin starts the worker with QProcess as "<this file> <worker script>".
rem CreateProcess can run a batch file but not a .py file, and Krita ships no
rem python.exe, so worker_launcher.py cannot be the entry point here. Do what it
rem does on Linux: run the worker on the managed Python through uv. Stdio is
rem inherited (NDJSON protocol on stdout, logs on stderr) and the exit code is
rem passed through.
rem
rem Keep the Python version in sync with RUNTIME_WORKER_PYTHON in
rem worker_launcher.py (checked by tests/worker/test_worker_launch.py).
setlocal

where uv >nul 2>&1
if errorlevel 1 (
    echo [SmartSelectWorker] event.name=worker.launcher.failed event.domain=smart_select severity=error component=worker_launcher error.reason=uv_not_found 1>&2
    exit /b 127
)

rem Worker stdio is UTF-8 JSON; the Windows default would be the ANSI code page.
set "PYTHONUTF8=1"

echo [SmartSelectWorker] event.name=worker.launcher.exec event.domain=smart_select severity=info component=worker_launcher launcher.mode=uv runtime.python.version=3.12 1>&2
uv run --no-project --python 3.12 python %*
exit /b %ERRORLEVEL%
