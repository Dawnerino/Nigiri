@echo off
REM Nigiri entrypoint for Windows.
REM
REM Bootstraps a uv-managed virtual environment on first run, then hands off to
REM run.py which does the first-run wizard and serves the web UI. All arguments
REM are forwarded to run.py (e.g. --host 0.0.0.0, --reset, --no-browser).
setlocal
cd /d "%~dp0"

REM Make sure uv exists. If it doesn't, install it (user-local).
where uv >nul 2>nul
if errorlevel 1 (
  echo [nigiri] uv not found - installing it (https://docs.astral.sh/uv/)...
  powershell -ExecutionPolicy ByPass -NoProfile -Command "irm https://astral.sh/uv/install.ps1 | iex"
  set "PATH=%USERPROFILE%\.local\bin;%PATH%"
)

REM Create the venv and install the orchestrator deps on first run.
if not exist ".venv" (
  echo [nigiri] creating virtual environment...
  uv venv .venv
)
echo [nigiri] syncing base dependencies...
REM --inexact is required: the first-run wizard lazily installs big per-engine
REM runtimes (onnxruntime for Supertonic, torch for Qwen3-TTS) that are not in
REM the base dependency set. A plain `uv sync` uninstalls anything extraneous,
REM which would silently strip the TTS runtime on the very next launch.
uv sync --no-install-project --inexact || uv pip install fastapi uvicorn httpx psutil huggingface-hub python-multipart

set "PY=.venv\Scripts\python.exe"
if not exist "%PY%" set "PY=python"

"%PY%" run.py %*
endlocal
