"""Download model weights into ./models using huggingface_hub.

Single-file GGUF/ONNX assets are fetched by repo+filename. Whole-repo models
(Qwen3-TTS, Supertonic) are fetched with allow_patterns so we only grab the
weights, not training junk. Downloads land in ./models/<id>/ and the chosen
placement is recorded in runtime/manifest.json so the engines know where to
find them.
"""
from __future__ import annotations

import json
import threading
from pathlib import Path

from setup import catalog

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "models"
RUNTIME = ROOT / "runtime"
MANIFEST = RUNTIME / "manifest.json"

# Progress state the UI polls while a download runs.
_PROGRESS: dict = {"active": False, "kind": "", "id": "", "percent": 0.0,
                   "detail": "", "error": ""}
_LOCK = threading.Lock()


def _set(**kw) -> None:
    with _LOCK:
        _PROGRESS.update(kw)


def progress() -> dict:
    with _LOCK:
        return dict(_PROGRESS)


def load_manifest() -> dict:
    if MANIFEST.exists():
        try:
            return json.loads(MANIFEST.read_text())
        except json.JSONDecodeError:
            return {}
    return {}


def save_manifest(manifest: dict) -> None:
    RUNTIME.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(manifest, indent=2))


def model_dir(kind: str, model_id: str) -> Path:
    return MODELS / f"{kind}-{model_id}"


def is_downloaded(kind: str, model_id: str) -> bool:
    d = model_dir(kind, model_id)
    return d.exists() and any(d.iterdir())


def _hf_download(repo: str, target: Path, *, file: str | None = None,
                 patterns: list[str] | None = None) -> None:
    from huggingface_hub import hf_hub_download, snapshot_download
    target.mkdir(parents=True, exist_ok=True)
    if file:
        hf_hub_download(repo_id=repo, filename=file, local_dir=str(target))
    else:
        snapshot_download(repo_id=repo, local_dir=str(target),
                          allow_patterns=patterns or [])


def download(kind: str, model_id: str) -> dict:
    """Download one catalog model into ./models. Blocking; run in a thread."""
    entry = catalog.by_id(kind, model_id)
    if not entry:
        _set(active=False, error=f"unknown model {kind}/{model_id}")
        return {"ok": False, "error": f"unknown model {kind}/{model_id}"}

    _set(active=True, kind=kind, id=model_id, percent=0.0,
         detail=f"downloading {entry['label']}", error="")
    target = model_dir(kind, model_id)
    try:
        # Single-file assets (whisper ggml, hy-mt gguf) vs whole-repo (tts).
        file = entry.get("file") or None
        patterns = None
        if not file:
            # TTS repos: grab weights + config, skip training/eval noise.
            patterns = ["*.safetensors", "*.json", "*.model", "*.tiktoken",
                        "*.onnx", "*.bin", "tokenizer*", "*.txt", "*.yaml"]
        _set(percent=10.0, detail=f"fetching from {entry['hf_repo']}")
        _hf_download(entry["hf_repo"], target, file=file, patterns=patterns)
        _set(percent=100.0, detail="done", active=False)
        manifest = load_manifest()
        manifest[f"{kind}:{model_id}"] = {"path": str(target),
                                          "device": None}
        save_manifest(manifest)
        return {"ok": True, "path": str(target)}
    except Exception as exc:  # noqa: BLE001 - surface any download failure to UI
        _set(active=False, error=str(exc))
        return {"ok": False, "error": str(exc)}


def download_set(selection: dict) -> dict:
    """Download a {kind: id} selection sequentially. Returns per-model results."""
    results = {}
    for kind, model_id in selection.items():
        results[f"{kind}:{model_id}"] = download(kind, model_id)
    return results


if __name__ == "__main__":
    import sys
    if len(sys.argv) == 3:
        print(json.dumps(download(sys.argv[1], sys.argv[2]), indent=2))
    else:
        print("usage: downloads.py <kind> <model_id>")
