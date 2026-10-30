"""Nigiri orchestrator - the single FastAPI app the user runs.

Responsibilities:
  * First-run wizard: detect specs, check/install llama.cpp + whisper.cpp,
    recommend models, download what the user picks.
  * Supervise the three engine children (STT :9101, translator :9102,
    TTS :9103) and proxy the OpenAI surface + the one-shot pipeline.
  * Bind choice: 127.0.0.1 (localhost) or 0.0.0.0 (LAN), toggleable at runtime.
  * Live status: which components are up, where each model runs (GPU/CPU),
    and current VRAM/RAM usage.

State lives in ./runtime so a restart resumes the same setup. Models live in
./models, runtimes in ./bin - the whole product is one self-contained folder.
"""
from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
import urllib.request
import webbrowser
from pathlib import Path

import httpx
from fastapi import Body, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from setup import catalog, detect, recommend, binaries, downloads  # noqa: E402

RUNTIME = ROOT / "runtime"
WEB = ROOT / "server" / "web"
CONFIG = RUNTIME / "config.json"
CHOSEN = RUNTIME / "chosen.json"
STATE = RUNTIME / "state.json"

STT_PORT = int(os.environ.get("STT_PORT", "9101"))
TRANSLATOR_PORT = int(os.environ.get("TRANSLATOR_PORT", "9102"))
TTS_PORT = int(os.environ.get("TTS_PORT", "9103"))

app = FastAPI(title="Nigiri", version="0.1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"],
                   allow_headers=["*"])

_CHILD_LOCK = threading.Lock()
_CHILDREN: dict[str, subprocess.Popen] = {}
_LOGS: dict[str, object] = {}

# Background runtime-install state (a whisper.cpp source build can take minutes,
# so the wizard kicks it off in a thread and polls rather than blocking a request).
_INSTALL_LOCK = threading.Lock()
_INSTALL: dict[str, object] = {"running": False, "started": False,
                               "result": None, "message": ""}


def _install_worker() -> None:
    """Run binaries.ensure_all() in the background, recording the outcome."""
    try:
        result = binaries.ensure_all()
        ok = all(v.get("ok") for v in result.values() if isinstance(v, dict))
        with _INSTALL_LOCK:
            _INSTALL["result"] = result
            _INSTALL["message"] = "all runtimes ready" if ok else \
                "some runtimes failed to install"
    except Exception as exc:  # noqa: BLE001
        with _INSTALL_LOCK:
            _INSTALL["result"] = None
            _INSTALL["message"] = f"install error: {exc}"
    finally:
        with _INSTALL_LOCK:
            _INSTALL["running"] = False


# --------------------------------------------------------------------- config io
def _read_json(path: Path, default: dict) -> dict:
    if path.exists():
        try:
            return json.loads(path.read_text())
        except json.JSONDecodeError:
            return default
    return default


def _write_json(path: Path, data: dict) -> None:
    RUNTIME.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2))


def current_config() -> dict:
    return _read_json(CONFIG, {"host": "127.0.0.1", "port": 8765, "no_browser": False})


def chosen() -> dict:
    return _read_json(CHOSEN, {})


def save_chosen(data: dict) -> None:
    _write_json(CHOSEN, data)


def is_setup_done() -> bool:
    c = chosen()
    return bool(c.get("stt") and c.get("translator") and c.get("tts"))


# ------------------------------------------------------------- child supervision
def _engine_python() -> str:
    """The python to launch engine children with (the active venv)."""
    return sys.executable


def _start_child(name: str, module: str, env: dict) -> None:
    """Start one engine child, logging to runtime/<name>.log."""
    with _CHILD_LOCK:
        proc = _CHILDREN.get(name)
        if proc and proc.poll() is None:
            return
        RUNTIME.mkdir(parents=True, exist_ok=True)
        log = (RUNTIME / f"{name}.log").open("ab")
        _LOGS[name] = log
        # Each engine child needs its own runtime's library dir on LD_LIBRARY_PATH
        # (llama.cpp and whisper.cpp ship conflicting libggml* versions).
        lib_dir = {"stt": str(ROOT / "bin" / "whisper"),
                   "translator": str(ROOT / "bin" / "llama")}.get(name, "")
        ld = os.environ.get("LD_LIBRARY_PATH", "")
        if lib_dir:
            new_ld = lib_dir + (os.pathsep + ld if ld else "")
            full_env = {**os.environ, **env, "LD_LIBRARY_PATH": new_ld}
        else:
            full_env = {**os.environ, **env}
        _CHILDREN[name] = subprocess.Popen(
            [_engine_python(), "-m", module], cwd=str(ROOT),
            stdout=log, stderr=log, env=full_env)


def _stop_child(name: str) -> None:
    with _CHILD_LOCK:
        proc = _CHILDREN.pop(name, None)
        log = _LOGS.pop(name, None)
    if proc and proc.poll() is None:
        try:
            if os.name == "nt":
                proc.terminate()
            else:
                proc.send_signal(signal.SIGTERM)
            proc.wait(timeout=15)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass
    if log:
        try:
            log.close()
        except Exception:
            pass


def _child_alive(name: str) -> bool:
    proc = _CHILDREN.get(name)
    return bool(proc and proc.poll() is None)


def _health(port: int, path: str = "/health", timeout: float = 3.0) -> dict:
    try:
        r = httpx.get(f"http://127.0.0.1:{port}{path}", timeout=timeout)
        return {"up": r.status_code == 200, **r.json()}
    except Exception as exc:  # noqa: BLE001
        return {"up": False, "error": str(exc)}


def _place_devices(c: dict) -> dict:
    """Decide GPU vs CPU for the whole chosen set, respecting the *combined*
    concurrent VRAM/RAM budget (all three engines run at once).

    Returns {kind: "gpu"|"cpu"}. Models are placed on GPU greedily by quality
    (largest VRAM first) until the budget is exhausted; the rest go to CPU.
    """
    prof = detect.detect()
    free = int(prof.get("vram_free_mib", 0))
    ram = int(prof.get("ram_mib", 0))
    head = recommend.VRAM_HEADROOM_MIB
    placements: dict = {}
    vram_used = 0
    ram_used = 0
    # Order by VRAM footprint descending so the heaviest model gets VRAM first.
    items = []
    for kind in ("stt", "translator", "tts"):
        entry = catalog.by_id(kind, c.get(kind, "")) or {}
        items.append((kind, entry))
    items.sort(key=lambda t: -int(t[1].get("vram_mib", 0)))
    for kind, entry in items:
        need_vram = int(entry.get("vram_mib", 0))
        need_ram = int(entry.get("ram_mib", 0))
        if need_vram <= 0:
            placements[kind] = "cpu"  # CPU-only model (Supertonic)
            ram_used += need_ram
            continue
        if prof.get("has_gpu") and vram_used + need_vram + head <= free:
            placements[kind] = "gpu"
            vram_used += need_vram
        elif ram_used + need_ram <= max(ram, 0):
            placements[kind] = "cpu"
            ram_used += need_ram
        else:
            placements[kind] = "gpu"  # last resort: try GPU anyway
            vram_used += need_vram
    return placements


def start_engines() -> None:
    """Start the engine children for the chosen model set, placing each on the
    best device within the combined budget. STT + TTS always start; translator
    starts if its model exists."""
    c = chosen()
    if not c:
        return
    dev = _place_devices(c)
    # STT
    _start_child("stt", "engines.stt_engine",
                 {"STT_MODEL": c.get("stt", ""), "STT_PORT": str(STT_PORT),
                  "NIGIRI_DEVICE": dev.get("stt", "cpu")})
    # Translator
    if downloads.is_downloaded("translator", c.get("translator", "")):
        d = dev.get("translator", "cpu")
        _start_child("translator", "engines.translator_engine",
                     {"TRANSLATOR_MODEL": c.get("translator", ""),
                      "TRANSLATOR_PORT": str(TRANSLATOR_PORT),
                      "NIGIRI_DEVICE": d,
                      "NIGIRI_NGL": "99" if d == "gpu" else "0"})
    # TTS
    engine = "qwen3" if str(c.get("tts", "")).startswith("qwen3") else "supertonic"
    _start_child("tts", "engines.tts_engine",
                 {"TTS_ENGINE": engine, "TTS_MODEL": c.get("tts", ""),
                  "TTS_PORT": str(TTS_PORT),
                  "NIGIRI_DEVICE": dev.get("tts", "cpu")})



# ------------------------------------------------------------------- wizard API
@app.get("/api/setup/status")
def setup_status() -> dict:
    """Everything the first-run wizard needs: specs, runtimes, recommendations."""
    prof = detect.detect()
    rec = recommend.recommend(prof)
    return {
        "done": is_setup_done(),
        "profile": prof,
        "binaries": binaries.status(),
        "recommendation": rec,
        "catalog": catalog.ALL,
        "downloaded": {
            kind: {e["id"]: downloads.is_downloaded(kind, e["id"])
                   for e in catalog.ALL[kind]}
            for kind in catalog.ALL
        },
    }


@app.post("/api/setup/install_binaries")
def install_binaries() -> dict:
    """Kick off runtime install in the background (a source build is slow).

    The wizard polls /api/setup/install_status until running == False.
    """
    with _INSTALL_LOCK:
        if _INSTALL["running"]:
            return {"started": True, "running": True,
                    "message": "install already in progress"}
        _INSTALL.update({"running": True, "started": True, "result": None,
                         "message": "installing runtimes…"})
    threading.Thread(target=_install_worker, daemon=True).start()
    return {"started": True, "running": True, "message": "installing runtimes…"}


@app.get("/api/setup/install_status")
def install_status() -> dict:
    """Progress of the background runtime install + current binary presence."""
    with _INSTALL_LOCK:
        snap = dict(_INSTALL)
    snap["binaries"] = binaries.status()
    return snap


@app.post("/api/setup/choose")
def choose(payload: dict = Body(...)) -> dict:
    """Persist the user's model selection and kick off downloads in a thread."""
    selection = {k: payload.get(k) for k in ("stt", "translator", "tts")}
    for kind, mid in selection.items():
        if not catalog.by_id(kind, mid or ""):
            raise HTTPException(400, f"unknown {kind} model '{mid}'")
    save_chosen(selection)
    # Downloads run in the background; the UI polls /api/setup/download_status.
    threading.Thread(target=downloads.download_set, args=(selection,),
                     daemon=True).start()
    return {"ok": True, "selection": selection}


@app.get("/api/setup/download_status")
def download_status() -> dict:
    return {"progress": downloads.progress(),
            "downloaded": {
                kind: {e["id"]: downloads.is_downloaded(kind, e["id"])
                       for e in catalog.ALL[kind]}
                for kind in catalog.ALL}}


@app.post("/api/setup/finish")
def finish_setup() -> dict:
    """After downloads complete: start engines and mark setup done."""
    start_engines()
    return {"ok": True, "chosen": chosen()}


# ------------------------------------------------------------------- stack state
def _free_ram_mib() -> int:
    try:
        import psutil
        return int(psutil.virtual_memory().available // (1024 * 1024))
    except Exception:
        return 0


@app.get("/api/stack/state")
def stack_state() -> dict:
    prof = detect.detect()
    c = chosen()
    return {
        "chosen": c,
        "devices": _place_devices(c),
        "stt": _health(STT_PORT),
        "translator": _health(TRANSLATOR_PORT),
        "tts": _health(TTS_PORT),
        "memory": {
            "ram_total_mib": prof["ram_mib"],
            "ram_free_mib": _free_ram_mib(),
            "vram_total_mib": prof["vram_total_mib"],
            "vram_free_mib": prof["vram_free_mib"],
            "gpu_name": prof["gpu_name"],
        },
    }


# ------------------------------------------------------------------- model switch
@app.post("/api/model/switch")
def switch_model(payload: dict = Body(...)) -> dict:
    """Switch one stage's model at runtime (restarts that engine child)."""
    kind = payload.get("kind")
    mid = payload.get("id")
    if kind not in catalog.ALL or not catalog.by_id(kind, mid or ""):
        raise HTTPException(400, f"bad kind/id: {kind}/{mid}")
    if not downloads.is_downloaded(kind, mid):
        raise HTTPException(409, f"model {kind}/{mid} not downloaded")
    c = chosen()
    c[kind] = mid
    save_chosen(c)
    _stop_child(kind)
    start_engines()
    return {"ok": True, "chosen": c}


# ------------------------------------------------------------------- network
@app.get("/api/network")
def network() -> dict:
    return current_config()


@app.post("/api/network")
def set_network(payload: dict = Body(...)) -> dict:
    """Toggle localhost vs LAN. Persisted; full rebind needs a restart."""
    host = payload.get("host", "127.0.0.1")
    if host not in ("127.0.0.1", "0.0.0.0"):
        raise HTTPException(400, "host must be 127.0.0.1 or 0.0.0.0")
    cfg = current_config()
    cfg["host"] = host
    _write_json(CONFIG, cfg)
    os.environ["NIGIRI_HOST"] = host
    return {"ok": True, "host": host, "restart_required": True}



# ------------------------------------------------------------------- OpenAI proxy
@app.post("/v1/audio/transcriptions")
async def transcriptions(file: UploadFile = File(...), model: str = Form(""),
                         language: str = Form("auto"), prompt: str = Form("")) -> Response:
    data = await file.read()
    async with httpx.AsyncClient(timeout=600) as c:
        r = await c.post(f"http://127.0.0.1:{STT_PORT}/v1/audio/transcriptions",
                         files={"file": (file.filename or "audio.wav", data)},
                         data={"model": model, "language": language, "prompt": prompt})
    return Response(content=r.content, status_code=r.status_code,
                    media_type=r.headers.get("content-type", "application/json"))


@app.post("/v1/chat/completions")
async def chat(payload: dict = Body(...)) -> Response:
    async with httpx.AsyncClient(timeout=600) as c:
        r = await c.post(f"http://127.0.0.1:{TRANSLATOR_PORT}/v1/chat/completions",
                         json=payload)
    return Response(content=r.content, status_code=r.status_code,
                    media_type=r.headers.get("content-type", "application/json"))


@app.post("/v1/audio/speech")
async def speech(payload: dict = Body(...)) -> Response:
    async with httpx.AsyncClient(timeout=600) as c:
        r = await c.post(f"http://127.0.0.1:{TTS_PORT}/v1/audio/speech", json=payload)
    return Response(content=r.content, status_code=r.status_code,
                    media_type=r.headers.get("content-type", "application/json"))


# ------------------------------------------------------------------- pipeline
@app.post("/api/pipeline")
async def pipeline(file: UploadFile = File(...), source: str = Form("auto"),
                   target: str = Form("English")) -> Response:
    """One shot: audio -> STT -> translate -> TTS. Returns the spoken translation."""
    import base64
    audio = await file.read()
    async with httpx.AsyncClient(timeout=600) as c:
        stt = await c.post(f"http://127.0.0.1:{STT_PORT}/v1/audio/transcriptions",
                           files={"file": (file.filename or "audio.wav", audio)},
                           data={"language": source})
        if stt.status_code != 200:
            raise HTTPException(502, f"STT failed: {stt.text[:300]}")
        transcript = stt.json().get("text", "").strip()
        if not transcript:
            raise HTTPException(422, "empty transcript")
        tr = await c.post(f"http://127.0.0.1:{TRANSLATOR_PORT}/translate",
                          json={"text": transcript, "source": source, "target": target})
        if tr.status_code != 200:
            raise HTTPException(502, f"translation failed: {tr.text[:300]}")
        translated = tr.json().get("translated", "")
        sp = await c.post(f"http://127.0.0.1:{TTS_PORT}/v1/audio/speech",
                          json={"input": translated, "language": target})
        if sp.status_code != 200:
            raise HTTPException(502, f"TTS failed: {sp.text[:300]}")
    return JSONResponse({"transcript": transcript, "translated": translated,
                         "audio_b64": base64.b64encode(sp.content).decode()})


# ------------------------------------------------------------------- web ui
@app.get("/")
def index() -> FileResponse:
    return FileResponse(WEB / "index.html")


@app.get("/api/shutdown")
def shutdown() -> dict:
    for name in list(_CHILDREN):
        _stop_child(name)
    return {"ok": True}


# --------------------------------------------------------------------- main
def main(host: str = "127.0.0.1", port: int = 8765, no_browser: bool = False,
         reset: bool = False) -> int:
    import uvicorn
    if reset and CHOSEN.exists():
        CHOSEN.unlink()
    # Resume the chosen setup on boot.
    if is_setup_done():
        start_engines()

    url = f"http://{'127.0.0.1' if host == '0.0.0.0' else host}:{port}"
    if not no_browser:
        threading.Timer(1.5, lambda: webbrowser.open(url)).start()
    print(f"[nigiri] serving on {url}  (bind={host})")
    uvicorn.run(app, host=host, port=port, log_level="info")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

