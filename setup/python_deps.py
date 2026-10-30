"""Install the heavy per-engine Python runtimes on demand.

The base venv only carries the orchestrator deps (fastapi/uvicorn/httpx/...). The
TTS runtimes are big - onnxruntime for Supertonic, torch for Qwen3-TTS - so they
are installed lazily, once, when the wizard picks a model that needs them.
Without this the engines still start (weights and imports are lazy) and only fail
at synthesis time with "No module named 'numpy'", which is a confusing place to
discover a missing dependency.

Everything here is safe to call repeatedly: find_spec decides whether anything is
actually missing, and we only shell out to an installer when it is.
"""
from __future__ import annotations

import importlib.util
import shlex
import shutil
import subprocess
import sys
import threading
from pathlib import Path

from setup import catalog

ROOT = Path(__file__).resolve().parent.parent

# Catalog "runtime" field -> what must be importable for it, and the minimal pip
# specs that provide those modules (each pulls its own transitive deps, so we do
# not list onnxruntime/numpy/soundfile for onnx). This mirrors
# [project.optional-dependencies] in pyproject.toml, which stays there for users
# who would rather run `uv sync --extra supertonic` themselves.
_EXTRAS = {
    "onnx": {
        "modules": ["supertonic", "onnxruntime", "numpy", "soundfile"],
        "pip": ["supertonic>=1.1"],
    },
    "torch": {
        "modules": ["qwen_tts", "torch", "torchaudio", "numpy", "soundfile"],
        "pip": ["qwen-tts>=0.1", "torch>=2.4", "torchaudio>=2.4"],
    },
}

# torch is a multi-GB download on a possibly-slow link; give the installer room.
_TIMEOUT_S = 3600

# Progress state the UI polls while an install runs (same shape as downloads.py).
_PROGRESS: dict = {"active": False, "kind": "", "id": "", "percent": 0.0,
                   "detail": "", "error": ""}
_LOCK = threading.Lock()


def _set(**kw) -> None:
    with _LOCK:
        _PROGRESS.update(kw)


def progress() -> dict:
    with _LOCK:
        return dict(_PROGRESS)


def _spec(kind: str, model_id: str) -> dict | None:
    """The install spec for a catalog model, or None if it needs no extra."""
    entry = catalog.by_id(kind, model_id)
    return _EXTRAS.get((entry or {}).get("runtime", ""))


def missing_modules(kind: str, model_id: str) -> list[str]:
    """Modules this model needs that the running interpreter cannot import."""
    spec = _spec(kind, model_id)
    if not spec:
        return []
    return [m for m in spec["modules"]
            if importlib.util.find_spec(m) is None]


def install_command(specs: list[str]) -> list[str]:
    """Installer argv: prefer uv (fast, already required by run.sh), else pip.

    Targets sys.executable explicitly so we install into the venv that is
    actually running the orchestrator, not whatever venv happens to be active.
    """
    python = Path(sys.executable) if sys.executable else ROOT / ".venv/bin/python"
    uv = shutil.which("uv")
    if uv:
        return [uv, "pip", "install", "--python", str(python), *specs]
    return [str(python), "-m", "pip", "install", *specs]


def install_hint(kind: str, model_id: str) -> str:
    """Copy-pasteable command that installs whatever this model is missing.

    Engines surface this in their own error messages, since a missing runtime
    only shows up at synthesis time and "No module named 'numpy'" alone never
    tells the user which command to run.
    """
    spec = _spec(kind, model_id)
    if not spec:
        return ""
    return " ".join(shlex.quote(c) for c in install_command(spec["pip"]))


def ensure(kind: str, model_id: str) -> dict:
    """Install the Python runtime this model needs, if anything is missing.

    Never raises: a failed install is reported so the wizard can show it and the
    user can still proceed (the engine will fail later with its own message).
    """
    spec = _spec(kind, model_id)
    if not spec:
        return {"ok": True, "installed": [],
                "note": f"{kind}/{model_id} needs no python runtime"}

    missing = missing_modules(kind, model_id)
    if not missing:
        return {"ok": True, "installed": [], "already_present": True}

    _set(active=True, kind=kind, id=model_id, percent=0.0,
         detail=f"installing {', '.join(missing)}", error="")
    cmd = install_command(spec["pip"])
    try:
        proc = subprocess.run(cmd, cwd=str(ROOT), capture_output=True,
                              text=True, timeout=_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        msg = f"install timed out after {_TIMEOUT_S}s"
        _set(active=False, percent=0.0, detail="", error=msg)
        return {"ok": False, "installed": [], "error": msg}
    except OSError as exc:
        msg = f"installer failed to start: {exc}"
        _set(active=False, percent=0.0, detail="", error=msg)
        return {"ok": False, "installed": [], "error": msg}

    if proc.returncode != 0:
        tail = "\n".join((proc.stderr or proc.stdout or "").strip()
                         .splitlines()[-6:])
        msg = f"installer exited {proc.returncode}: {tail}"
        _set(active=False, percent=0.0, detail="", error=msg)
        return {"ok": False, "installed": [], "error": msg}

    still = missing_modules(kind, model_id)
    if still:
        msg = f"installed but still cannot import: {', '.join(still)}"
        _set(active=False, percent=0.0, detail="", error=msg)
        return {"ok": False, "installed": [], "error": msg}

    _set(active=False, percent=100.0, detail="installed", error="")
    return {"ok": True, "installed": missing}


def ensure_selection(selection: dict) -> dict:
    """Install runtimes for a {kind: id} selection. Returns per-model results."""
    results = {}
    for kind, model_id in selection.items():
        if not model_id:
            continue
        res = ensure(kind, model_id)
        if res.get("installed") or res.get("error"):
            results[f"{kind}:{model_id}"] = res
    return results


def status(selection: dict) -> dict:
    """What the UI needs to know before offering to run the pipeline."""
    missing = {}
    for kind, model_id in selection.items():
        if not model_id:
            continue
        mods = missing_modules(kind, model_id)
        if mods:
            missing[f"{kind}:{model_id}"] = mods
    return {"missing": missing, "needs_install": bool(missing)}
