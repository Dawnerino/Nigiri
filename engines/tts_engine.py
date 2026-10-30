"""TTS engine - Supertonic 3 (ONNX) or Qwen3-TTS (PyTorch), one FastAPI shim.

The orchestrator picks the engine via the TTS_ENGINE env var (supertonic |
qwen3) and the variant via TTS_MODEL. Both engines expose the same OpenAI-ish
contract so the rest of the pipeline cannot tell them apart:

  POST /v1/audio/speech   {input, voice, response_format} -> audio/wav
  GET  /health            engine + model + device + resident flag

Supertonic 3 runs on ONNX Runtime, needs no GPU, and is the default. Qwen3-TTS
runs on torch and prefers CUDA when available. Weights are loaded lazily on the
first synthesis so the shim starts instantly.
"""
from __future__ import annotations

import io
import os
import threading
from pathlib import Path

from fastapi import Body, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "models"

ENGINE = os.environ.get("TTS_ENGINE", "supertonic").strip().lower()
MODEL = os.environ.get("TTS_MODEL", "supertonic-3").strip()
DEVICE = os.environ.get("NIGIRI_DEVICE", "cpu")

app = FastAPI(title="Nigiri TTS")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"],
                   allow_headers=["*"])

_LOCK = threading.Lock()
_resident = {"loaded": False, "device": "", "error": ""}
_tts = None  # cached Supertonic TTS instance

# Supertonic wants ISO codes; the UI/pipeline pass human names. Map the common
# ones and pass through anything that already looks like a 2-3 letter code.
_LANG_CODES = {
    "english": "en", "spanish": "es", "french": "fr", "german": "de",
    "japanese": "ja", "korean": "ko", "chinese": "zh", "mandarin": "zh",
    "italian": "it", "portuguese": "pt", "russian": "ru", "arabic": "ar",
    "hindi": "hi", "dutch": "nl", "polish": "pl", "turkish": "tr",
    "ukrainian": "uk", "vietnamese": "vi", "czech": "cs", "danish": "da",
    "finnish": "fi", "greek": "el", "hungarian": "hu", "indonesian": "id",
    "romanian": "ro", "swedish": "sv", "bulgarian": "bg", "croatian": "hr",
    "estonian": "et", "lithuanian": "lt", "latvian": "lv", "slovak": "sk",
    "slovenian": "sl",
}


def _lang_code(lang: str) -> str:
    if not lang:
        return "en"
    l = lang.strip().lower()
    if l in _LANG_CODES:
        return _LANG_CODES[l]
    if len(l) <= 3 and l.isalpha():
        return l
    return "en"


# Qwen3-TTS wants full language names (or "auto"), not ISO codes.
_QWEN_LANGS = {"auto", "chinese", "english", "french", "german", "italian",
               "japanese", "korean", "portuguese", "russian", "spanish"}


# The languages each TTS model can actually speak, as human-readable names the
# UI shows. The pipeline's target dropdown is populated from the *selected* TTS
# model's list, so a Qwen3 user sees 10 and a Supertonic user sees 31.
_MODEL_LANGUAGES = {
    "qwen3-0.6b": ["English", "Spanish", "French", "German", "Italian",
                   "Japanese", "Korean", "Portuguese", "Russian", "Chinese"],
    "qwen3-1.7b": ["English", "Spanish", "French", "German", "Italian",
                   "Japanese", "Korean", "Portuguese", "Russian", "Chinese"],
    "supertonic-3": ["English", "Spanish", "French", "German", "Italian",
                     "Portuguese", "Russian", "Japanese", "Korean", "Chinese",
                     "Dutch", "Polish", "Turkish", "Ukrainian", "Vietnamese",
                     "Czech", "Danish", "Finnish", "Greek", "Hungarian",
                     "Indonesian", "Romanian", "Swedish", "Bulgarian",
                     "Croatian", "Estonian", "Lithuanian", "Latvian",
                     "Slovak", "Slovenian", "Arabic"],
}


def supported_languages() -> list[str]:
    """Human-readable languages the currently selected TTS model can speak."""
    return list(_MODEL_LANGUAGES.get(MODEL, ["English"]))


def _qwen_lang(lang: str) -> str:
    if not lang:
        return "auto"
    l = lang.strip().lower()
    if l in _QWEN_LANGS:
        return l
    # Map an ISO code (or human name) back to Qwen's full-name vocabulary.
    code = _lang_code(l)
    for name, c in _LANG_CODES.items():
        if c == code and name in _QWEN_LANGS:
            return name
    return "auto"


def _model_dir(kind: str, model_id: str) -> Path:
    return MODELS / f"{kind}-{model_id}"


# --------------------------------------------------------------------- Supertonic
def _synthesize_supertonic(text: str, voice: str, lang: str) -> bytes:
    import numpy as np
    import soundfile as sf
    from supertonic import TTS  # type: ignore

    global _tts
    model_dir = _model_dir("tts", MODEL)
    if _tts is None:
        _tts = TTS(model="supertonic-3", model_dir=str(model_dir), auto_download=False)
        _resident["device"] = "cpu"
        _resident["loaded"] = True

    style = _tts.get_voice_style(voice_name=voice or "M1")
    wav, _dur = _tts.synthesize(text, voice_style=style, lang=_lang_code(lang))
    buf = io.BytesIO()
    sf.write(buf, np.asarray(wav).squeeze(), 44100, format="WAV")
    return buf.getvalue()


# ------------------------------------------------------------------------ Qwen3
_qwen = None  # cached Qwen3TTSModel instance


def _synthesize_qwen3(text: str, voice: str, lang: str) -> bytes:
    import numpy as np
    import soundfile as sf
    import torch
    from qwen_tts import Qwen3TTSModel  # type: ignore

    global _qwen
    model_dir = _model_dir("tts", MODEL)
    device = "cuda" if (DEVICE == "gpu" and torch.cuda.is_available()) else "cpu"
    if device == "cpu" and DEVICE == "gpu":
        # Wizard asked for GPU but CUDA isn't usable right now - bail loudly
        # rather than silently loading a multi-GB torch model into system RAM.
        raise RuntimeError(
            "Qwen3-TTS was set to GPU but CUDA is unavailable; "
            "refusing to load into system RAM. Pick Supertonic or fix CUDA.")
    if _qwen is None:
        _qwen = Qwen3TTSModel.from_pretrained(
            str(model_dir), device_map=device,
            dtype=torch.bfloat16 if device == "cuda" else torch.float32,
            attn_implementation=None,
        )
        _resident["device"] = device
        _resident["loaded"] = True

    # CustomVoice checkpoints synthesize from built-in speakers - no reference
    # audio needed, which is what the pipeline wants. Fall back to voice clone
    # only if a reference wav is supplied via data/voices/<voice>.wav.
    speaker = voice or None
    language = _qwen_lang(lang)
    wavs, sr = _qwen.generate_custom_voice(
        text=text, language=language, speaker=speaker)
    wav = np.asarray(wavs[0]).squeeze()
    buf = io.BytesIO()
    sf.write(buf, wav, int(sr), format="WAV")
    return buf.getvalue()


@app.get("/health")
def health() -> dict:
    return {"ok": True, "engine": ENGINE, "model": MODEL,
            "device": _resident["device"] or DEVICE, "resident": _resident["loaded"],
            "languages": supported_languages(),
            "error": _resident["error"]}


@app.get("/languages")
def languages() -> dict:
    """Languages the selected TTS model can speak (drives the UI target list)."""
    return {"model": MODEL, "languages": supported_languages()}


@app.post("/v1/audio/speech")
def speech(payload: dict = Body(...)) -> Response:
    text = (payload.get("input") or "").strip()
    if not text:
        raise HTTPException(400, "input text required")
    voice = payload.get("voice", "")
    lang = payload.get("language", "")
    try:
        with _LOCK:
            if ENGINE == "qwen3":
                wav = _synthesize_qwen3(text, voice, lang)
            else:
                wav = _synthesize_supertonic(text, voice, lang)
    except ImportError as exc:
        raise HTTPException(500, f"TTS runtime not installed for engine '{ENGINE}': {exc}")
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        _resident["error"] = str(exc)
        raise HTTPException(500, f"synthesis failed: {exc}")
    return Response(content=wav, media_type="audio/wav")


def main() -> None:
    import uvicorn
    port = int(os.environ.get("TTS_PORT", "9103"))
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")


if __name__ == "__main__":
    main()
