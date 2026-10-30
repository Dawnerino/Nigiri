"""STT engine - whisper-cli (whisper.cpp) wrapped in FastAPI.

OpenAI-compatible:
  POST /v1/audio/transcriptions   multipart file/model/language/prompt
  GET  /v1/models                 list the GGUF weights present in ./models
  GET  /health                    ok + available models

whisper-cli holds no VRAM between calls (it shells out per request), so model
switching is free and there is no GPU lock here. The orchestrator passes the
active model in via the STT_MODEL env var; callers may override per-request.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import traceback
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from setup import binaries  # noqa: E402

BIN = binaries.whisper_cli_path()
MODELS = ROOT / "models"
THREADS = os.environ.get("NIGIRI_THREADS", str(max(1, (os.cpu_count() or 4))))

app = FastAPI(title="Nigiri STT")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"],
                   allow_headers=["*"])

_LOCK = threading.Lock()
_state = {"last_exit": 0, "last_ms": 0.0}


def _available_models() -> dict[str, Path]:
    """Map model id -> ggml weight path for every downloaded STT model."""
    found: dict[str, Path] = {}
    for d in MODELS.glob("stt-*"):
        for ggml in d.glob("ggml-*.bin"):
            # id is the dir name minus the "stt-" prefix.
            found[d.name[4:]] = ggml
            break
    return found


def _ffmpeg_to_wav16k(src: Path, dst: Path) -> None:
    """Normalise any input container to 16 kHz mono PCM for whisper.cpp."""
    if not shutil.which("ffmpeg"):
        raise HTTPException(500, "ffmpeg not found on PATH - required for audio decode")
    cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
           "-i", str(src), "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", str(dst)]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
    if proc.returncode != 0:
        raise HTTPException(400, f"ffmpeg decode failed: {proc.stderr.strip()[:500]}")


def _run_whisper(wav: Path, model: Path, lang: str, prompt: str) -> dict:
    out_base = wav.with_suffix("")
    if not BIN.exists():
        raise HTTPException(
            503, f"whisper-cli is not installed at {BIN}. Open the Nigiri setup "
                 "wizard and click 'Install missing runtimes' (on macOS this "
                 "builds whisper.cpp from source and needs Xcode Command Line "
                 "Tools + cmake: `xcode-select --install` and `brew install cmake`).")
    cmd = [str(BIN), "-m", str(model), "-f", str(wav), "-t", THREADS,
           "-oj", "-of", str(out_base), "-np", "--flash-attn"]
    cmd += ["-l", lang if (lang and lang != "auto") else "auto"]
    if prompt:
        cmd += ["-i", prompt]
    started = time.time()
    ld = os.environ.get("LD_LIBRARY_PATH", "")
    lib_dir = str(BIN.parent)
    new_ld = lib_dir if not ld else (lib_dir + os.pathsep + ld)
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=1800,
                          env={**os.environ, "LD_LIBRARY_PATH": new_ld, "GGML_CUDA": "1"})
    _state["last_exit"] = proc.returncode
    _state["last_ms"] = round((time.time() - started) * 1000, 1)
    if proc.returncode != 0:
        raise HTTPException(500, "whisper-cli failed (exit %s): %s" %
                            (proc.returncode, (proc.stderr or proc.stdout).strip()[-800:]))
    out_json = Path(str(out_base) + ".json")
    if not out_json.exists():
        raise HTTPException(500, "whisper-cli produced no output file")
    return json.loads(out_json.read_text())


@app.get("/health")
def health() -> dict:
    return {"ok": BIN.exists(), "bin": str(BIN),
            "models": {k: True for k in _available_models()},
            "state": _state}


@app.get("/v1/models")
def models() -> dict:
    ids = list(_available_models())
    return {"object": "list",
            "data": [{"id": i, "object": "model", "owned_by": "nigiri"} for i in ids]}


@app.post("/v1/audio/transcriptions")
def transcribe(file: UploadFile = File(...), model: str = Form(""),
               language: str = Form("auto"), prompt: str = Form("")) -> dict:
    available = _available_models()
    if not available:
        raise HTTPException(503, "no STT models downloaded")
    model_id = (model or os.environ.get("STT_MODEL", "")).strip()
    if model_id and model_id not in available:
        raise HTTPException(400, f"unknown model '{model_id}', have {list(available)}")
    if not model_id:
        model_id = next(iter(available))
    weight = available[model_id]

    suffix = Path(file.filename or "audio.wav").suffix or ".wav"
    try:
        with tempfile.TemporaryDirectory() as td:
            src = Path(td) / ("in" + suffix)
            src.write_bytes(file.file.read())
            wav = Path(td) / "in16k.wav"
            _ffmpeg_to_wav16k(src, wav)
            with _LOCK:
                data = _run_whisper(wav, weight, language, prompt)
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        # Anything unexpected (decode crash, whisper-cli segfault, bad weights)
        # would otherwise surface as a bare "Internal Server Error". Log the
        # traceback to runtime/stt.log and return the real reason.
        sys.stderr.write("[stt] unhandled error: " + traceback.format_exc() + "\n")
        sys.stderr.flush()
        raise HTTPException(500, f"transcription failed: {exc}")

    # whisper.cpp -oj emits {"transcription":[{"text":...}, ...]}; join segments.
    text = ""
    if isinstance(data, dict):
        segs = data.get("transcription")
        if not isinstance(segs, list):
            segs = (data.get("result", {}) or {}).get("segments") or data.get("segments") or []
        text = " ".join((s.get("text", "") or "").strip() for s in segs if isinstance(s, dict)).strip()
    return {"text": text, "model": model_id, "language": language,
            "duration_ms": _state["last_ms"]}


def main() -> None:
    import uvicorn
    port = int(os.environ.get("STT_PORT", "9101"))
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="info")


if __name__ == "__main__":
    main()
