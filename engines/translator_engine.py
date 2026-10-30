"""Translator engine - llama.cpp's llama-server, the LM Studio replacement.

This shim owns a single llama-server child process that serves the HY-MT1.5
GGUF as an OpenAI-compatible /v1/chat/completions endpoint. The orchestrator
talks to it for translation (and optionally chat). Switching translator models
is a child restart with a new --model path.

  POST /v1/chat/completions   proxied straight to llama-server
  POST /translate             {text, source, target} -> {translated}
  GET  /health                llama-server liveness + loaded model

Device placement (GPU vs CPU) is decided by the orchestrator and passed in via
the NIGIRI_DEVICE env var; we translate that to llama-server's -ngl flag.
"""
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

from fastapi import Body, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from setup import binaries  # noqa: E402

BIN = binaries.llama_server_path()
MODELS = ROOT / "models"
RUNTIME = ROOT / "runtime"
LOG = RUNTIME / "llama-server.log"

PORT = int(os.environ.get("TRANSLATOR_PORT", "9102"))
UPSTREAM = f"http://127.0.0.1:{PORT + 100}"  # llama-server's own port
CTX = os.environ.get("NIGIRI_CTX", "4096")
DEVICE = os.environ.get("NIGIRI_DEVICE", "cpu")  # cpu | gpu
NGL_GPU = os.environ.get("NIGIRI_NGL", "99")     # layers to offload on GPU

app = FastAPI(title="Nigiri Translator")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"],
                   allow_headers=["*"])

_LOCK = threading.Lock()
_proc: subprocess.Popen | None = None
_loaded_model: str = ""


def _model_path(model_id: str) -> Path:
    """Find the GGUF for a translator model id inside ./models."""
    d = MODELS / f"translator-{model_id}"
    for gguf in d.glob("*.gguf"):
        return gguf
    raise HTTPException(500, f"no GGUF found for translator model '{model_id}' in {d}")


def _stop() -> None:
    global _proc
    if _proc and _proc.poll() is None:
        try:
            if os.name == "nt":
                _proc.terminate()
            else:
                _proc.send_signal(signal.SIGTERM)
            _proc.wait(timeout=15)
        except Exception:
            try:
                _proc.kill()
            except Exception:
                pass
    _proc = None


def _start(model_id: str) -> None:
    global _proc, _loaded_model
    gguf = _model_path(model_id)
    RUNTIME.mkdir(parents=True, exist_ok=True)
    cmd = [str(BIN), "-m", str(gguf), "--host", "127.0.0.1", "--port", str(PORT + 100),
           "-c", CTX, "--no-webui"]
    if DEVICE == "gpu":
        cmd += ["-ngl", NGL_GPU]
    else:
        cmd += ["-ngl", "0"]
    log = LOG.open("ab")
    # Point the loader at the isolated llama.cpp library dir.
    ld = os.environ.get("LD_LIBRARY_PATH", "")
    lib_dir = str(BIN.parent)
    new_ld = lib_dir if not ld else (lib_dir + os.pathsep + ld)
    _proc = subprocess.Popen(cmd, stdout=log, stderr=log,
                             env={**os.environ, "LD_LIBRARY_PATH": new_ld,
                                  "GGML_CUDA": "1" if DEVICE == "gpu" else "0"})
    _loaded_model = model_id
    _wait_ready()


def _wait_ready(timeout: int = 180) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(f"{UPSTREAM}/health", timeout=3) as r:
                if r.status == 200:
                    return
        except Exception:
            time.sleep(1.0)
    raise HTTPException(500, "llama-server did not become healthy in time")


def _upstream(path: str, payload: dict) -> dict:
    req = urllib.request.Request(
        f"{UPSTREAM}{path}", data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.loads(r.read())


@app.get("/health")
def health() -> dict:
    alive = bool(_proc and _proc.poll() is None)
    return {"ok": alive, "model": _loaded_model, "device": DEVICE,
            "upstream": UPSTREAM}


@app.post("/v1/chat/completions")
def chat(payload: dict = Body(...)) -> JSONResponse:
    if not (_proc and _proc.poll() is None):
        raise HTTPException(503, "llama-server not running")
    try:
        return JSONResponse(_upstream("/v1/chat/completions", payload))
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(502, f"llama-server error: {exc}")


@app.post("/translate")
def translate(payload: dict = Body(...)) -> dict:
    text = (payload.get("text") or "").strip()
    if not text:
        raise HTTPException(400, "text required")
    source = payload.get("source") or "auto"
    target = payload.get("target") or "English"
    prompt = (f"Translate the following text to {target}. "
              f"Output only the translation.\n\n{text}")
    resp = _upstream("/v1/chat/completions", {
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0, "max_tokens": 1024,
    })
    out = resp["choices"][0]["message"]["content"].strip()
    return {"translated": out, "source": source, "target": target,
            "model": _loaded_model}


def main() -> None:
    import uvicorn
    model_id = os.environ.get("TRANSLATOR_MODEL", "")
    if model_id:
        with _LOCK:
            _start(model_id)
    uvicorn.run(app, host="127.0.0.1", port=PORT, log_level="info")


if __name__ == "__main__":
    main()
