"""The model catalog: every STT / translation / TTS option Nigiri can install,
with the facts the first-run wizard needs to show trade-offs:

  * download_mib  - weight download size
  * vram_mib      - GPU memory to run it (0 = runs on CPU)
  * ram_mib       - system RAM to run it (CPU inference)
  * speed         - relative speed tier (1 fastest .. 5 slowest) for the UI
  * languages     - language count (None = English only for .en STT)
  * hf_repo/file  - where to download it from

Sizes are the published GGUF/ONNX/safetensors figures. Speed tiers are
qualitative (measured on mid-range hardware) and shown as a 1-5 bar, not a
false-precision tokens/sec.
"""
from __future__ import annotations

# --------------------------------------------------------------------------- STT
# Whisper.cpp GGUF weights (ggerganov/whisper.cpp). .en = English-only.
STT = [
    {"id": "tiny.en", "label": "Whisper tiny.en", "download_mib": 78,
     "vram_mib": 300, "ram_mib": 600, "speed": 1, "languages": None,
     "hf_repo": "ggerganov/whisper.cpp", "file": "ggml-tiny.en.bin",
     "note": "fastest, roughest; English only"},
    {"id": "base.en", "label": "Whisper base.en", "download_mib": 148,
     "vram_mib": 500, "ram_mib": 1000, "speed": 1, "languages": None,
     "hf_repo": "ggerganov/whisper.cpp", "file": "ggml-base.en.bin",
     "note": "very fast, English only"},
    {"id": "small.en", "label": "Whisper small.en", "download_mib": 488,
     "vram_mib": 1000, "ram_mib": 1500, "speed": 2, "languages": None,
     "hf_repo": "ggerganov/whisper.cpp", "file": "ggml-small.en.bin",
     "note": "great CPU default for English"},
    {"id": "small", "label": "Whisper small", "download_mib": 488,
     "vram_mib": 1000, "ram_mib": 1500, "speed": 2, "languages": 99,
     "hf_repo": "ggerganov/whisper.cpp", "file": "ggml-small.bin",
     "note": "multilingual, light"},
    {"id": "medium.en", "label": "Whisper medium.en", "download_mib": 1530,
     "vram_mib": 2500, "ram_mib": 3500, "speed": 3, "languages": None,
     "hf_repo": "ggerganov/whisper.cpp", "file": "ggml-medium.en.bin",
     "note": "accurate English, heavier"},
    {"id": "large-v3-turbo-q5_0", "label": "Whisper large-v3 turbo (q5)",
     "download_mib": 548, "vram_mib": 2000, "ram_mib": 2500, "speed": 2,
     "languages": 99, "hf_repo": "ggerganov/whisper.cpp",
     "file": "ggml-large-v3-turbo-q5_0.bin",
     "note": "recommended: fast + multilingual + accurate"},
    {"id": "large-v3-q5_0", "label": "Whisper large-v3 (q5)", "download_mib": 1080,
     "vram_mib": 3500, "ram_mib": 4000, "speed": 4, "languages": 99,
     "hf_repo": "ggerganov/whisper.cpp", "file": "ggml-large-v3-q5_0.bin",
     "note": "most accurate, multilingual, slowest"},
]

# -------------------------------------------------------------------- Translator
# HY-MT1.5 via llama.cpp (llama-server). 38 languages, OpenAI-compatible.
TRANSLATOR = [
    {"id": "hy-mt1.5-1.8b", "label": "HY-MT1.5 1.8B", "download_mib": 1130,
     "vram_mib": 1600, "ram_mib": 2500, "speed": 2, "languages": 38,
     "hf_repo": "tencent/HY-MT1.5-1.8B-GGUF", "file": "HY-MT1.5-1.8B-Q4_K_M.gguf",
     "note": "fast, fits small GPUs and CPU; 38 languages"},
    {"id": "hy-mt1.5-7b", "label": "HY-MT1.5 7B", "download_mib": 4620,
     "vram_mib": 5400, "ram_mib": 6500, "speed": 4, "languages": 38,
     "hf_repo": "tencent/HY-MT1.5-7B-GGUF", "file": "HY-MT1.5-7B-Q4_K_M.gguf",
     "note": "higher quality, wants a GPU; 38 languages"},
]

# ------------------------------------------------------------------------- TTS
# Supertonic 3 (ONNX, CPU-friendly, 31 langs) + Qwen3-TTS (torch, 10 langs).
# vram_mib/ram_mib are *runtime* footprints (weights + activations + buffers),
# deliberately a little above the raw download so a "fits on GPU" decision does
# not OOM. Qwen3 runs in bf16 on GPU; the 0.6B weights alone are ~2.4 GB.
TTS = [
    {"id": "supertonic-3", "label": "Supertonic 3", "download_mib": 400,
     "vram_mib": 0, "ram_mib": 1200, "speed": 1, "languages": 31,
     "runtime": "onnx", "hf_repo": "supertone-oss-archive/supertonic-3",
     "file": "", "note": "lightning-fast on CPU, no GPU needed; 31 languages"},
    {"id": "qwen3-0.6b", "label": "Qwen3-TTS 0.6B", "download_mib": 2400,
     "vram_mib": 3200, "ram_mib": 4500, "speed": 3, "languages": 10,
     "runtime": "torch", "hf_repo": "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
     "file": "", "note": "lighter neural voice; 10 languages; wants a GPU"},
    {"id": "qwen3-1.7b", "label": "Qwen3-TTS 1.7B", "download_mib": 3900,
     "vram_mib": 5200, "ram_mib": 7000, "speed": 4, "languages": 10,
     "runtime": "torch", "hf_repo": "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
     "file": "", "note": "highest quality + prosody control; 10 languages; GPU"},
]

ALL = {"stt": STT, "translator": TRANSLATOR, "tts": TTS}


def by_id(kind: str, model_id: str) -> dict | None:
    for entry in ALL.get(kind, []):
        if entry["id"] == model_id:
            return entry
    return None


def speed_label(speed: int) -> str:
    return {1: "fastest", 2: "fast", 3: "medium", 4: "slow", 5: "slowest"}.get(
        speed, "medium")
