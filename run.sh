#!/usr/bin/env bash
# Nigiri entrypoint for Linux / macOS.
#
# Bootstraps a uv-managed virtual environment on first run, then hands off to
# run.py which does the first-run wizard and serves the web UI. All arguments
# are forwarded to run.py (e.g. --host 0.0.0.0, --reset, --no-browser).
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

# Make sure uv exists. If it doesn't, install it (user-local, no sudo).
if ! command -v uv >/dev/null 2>&1; then
  echo "[nigiri] uv not found - installing it (https://docs.astral.sh/uv/)..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  # Put uv on PATH for this session.
  export PATH="$HOME/.local/bin:$PATH"
fi

# Create the venv and install the orchestrator deps on first run.
if [ ! -d ".venv" ]; then
  echo "[nigiri] creating virtual environment..."
  uv venv .venv
fi
echo "[nigiri] syncing base dependencies..."
uv sync --no-install-project || uv pip install fastapi uvicorn httpx psutil huggingface-hub python-multipart

# Prefer the venv python; fall back to system python3.
PY=".venv/bin/python"
[ -x "$PY" ] || PY="$(command -v python3 || command -v python)"

exec "$PY" run.py "$@"
