# 🍣 Nigiri

**A hand-pressed local voice pipeline.** Speech in → transcribed → translated →
spoken back out, entirely on your own machine. No cloud, no API keys, no data
leaving your computer.

Nigiri wraps three on-device engines behind one web UI and a single command:

| Stage | Engine | Runtime |
|-------|--------|---------|
| **Speech-to-text** | Whisper (tiny → large-v3) | `whisper-cli` (whisper.cpp) |
| **Translation** | HY-MT1.5 (38 languages) | `llama-server` (llama.cpp) |
| **Text-to-speech** | Supertonic 3 (31 langs) · Qwen3-TTS (10 langs) | ONNX Runtime · PyTorch |

Everything runs locally. Models and binaries live inside the project folder, so
you can move the whole thing around as one directory.

---

## Quick start

You need **[uv](https://docs.astral.sh/uv/)** (a fast Python package manager) and
Python 3.10+. That's it - no compiler, no CUDA toolkit, no LM Studio.

```bash
git clone https://github.com/your-name/nigiri.git
cd nigiri

# Linux / macOS
./run.sh

# Windows
run.bat
```

On first run Nigiri:

1. **Reads your system specs** - CPU, system RAM, and GPU VRAM (NVIDIA / AMD /
   Apple Silicon).
2. **Checks llama.cpp** - if it isn't present it offers to download a prebuilt
   binary for your OS (or build it from source if you prefer).
3. **Recommends a model set** that fits your hardware, and shows you, for every
   option, the **download size**, **projected speed**, and **memory it will use
   (VRAM vs system RAM)**. If you have free VRAM, it offers to run the model
   there.
4. Lets you **pick or override** each model in an interactive web wizard, then
   downloads what you chose into `./models/` and starts the pipeline.

The web UI then opens at `http://127.0.0.1:8765`.

---

## Local network or localhost only

By default Nigiri binds to `127.0.0.1` (your machine only). To expose the API
and UI on your local network (e.g. reach it from your phone), pass `--host`:

```bash
./run.sh --host 0.0.0.0        # listen on all interfaces (LAN)
./run.sh --host 127.0.0.1      # localhost only (default)
```

You can also flip this at runtime from the web UI's **Network** card.

---

## What runs where

Nigiri places each model on the best device for your machine:

- **GPU (VRAM)** when there is room - fastest for the translator and Qwen3-TTS.
- **System RAM (CPU)** otherwise - Supertonic 3 is designed to run fast on CPU
  and needs no GPU at all, so it is the default on machines without a GPU.

The first-run wizard shows the projected memory footprint of every choice before
you commit to a download, and the running UI reports live VRAM/RAM usage.

---

## The model menu

### Speech-to-text (Whisper.cpp)
| Model | Download | Languages | Speed | Notes |
|-------|----------|-----------|-------|-------|
| tiny.en | 78 MB | English | fastest | |
| base.en | 148 MB | English | very fast | |
| small.en | 488 MB | English | fast | great CPU default |
| small | 488 MB | multilingual | fast | |
| medium.en | 1.53 GB | English | medium | |
| large-v3-turbo-q5_0 | 548 MB | multilingual | fast | recommended |
| large-v3-q5_0 | 1.08 GB | multilingual | accurate | most accurate |

### Translation (HY-MT1.5, via llama.cpp)
| Model | Download | Languages | Notes |
|-------|----------|-----------|-------|
| HY-MT1.5-1.8B | ~1.9 GB | **38** | fast, fits small GPUs / CPU |
| HY-MT1.5-7B | ~4.6 GB | **38** | higher quality, wants a GPU |

### Text-to-speech
| Model | Download | Languages | Device | Notes |
|-------|----------|-----------|--------|-------|
| **Supertonic 3** | ~0.4 GB | **31** | CPU (or GPU) | lightning-fast, no GPU needed - default |
| Qwen3-TTS 0.6B | ~1.3 GB | 10 | GPU/CPU | lighter neural voice |
| Qwen3-TTS 1.7B | ~3.9 GB | 10 | GPU | highest quality, instruction prosody |

> Recommendations are computed from *your* specs. A laptop with 16 GB RAM and no
> GPU gets small.en + HY-MT1.5-1.8B (CPU) + Supertonic 3; a 10 GB GPU box gets
> large-v3-turbo + HY-MT1.5-1.8B (GPU) + Supertonic 3, and can opt into Qwen3-TTS.

---

## Project layout

```
nigiri/
├── run.py / run.sh / run.bat   one entrypoint: bootstrap, then serve
├── pyproject.toml              uv-managed dependencies
├── setup/                      first-run: detect, catalog, recommend, binaries, download
├── server/                     orchestrator (FastAPI) + web/index.html
├── engines/                    stt / translator / tts service shims
├── models/                     downloaded weights (gitignored)
├── bin/
│   ├── llama/                  llama.cpp runtime + libs (gitignored)
│   └── whisper/                whisper.cpp runtime + libs (gitignored)
├── runtime/                    logs, pids, chosen-model config, job state
└── data/voices/                your reference audio / voice profiles
```

> **Why two `bin/` subfolders?** llama.cpp and whisper.cpp each ship their own
> `libggml*` shared libraries at different versions. Keeping them in separate
> directories (and pointing each child process's `LD_LIBRARY_PATH`/`PATH` at its
> own folder) prevents the two runtimes from clashing over symbols.

## Commands

```bash
./run.sh                      # first-run wizard, then serve
./run.sh --host 0.0.0.0       # expose on LAN
./run.sh --reset              # re-run the first-run wizard
./run.sh --no-browser         # don't auto-open the browser
```

## API

Once running, Nigiri speaks OpenAI where it can:

- `POST /v1/audio/transcriptions` - Whisper STT (OpenAI-compatible)
- `POST /v1/chat/completions` - translation / chat via llama.cpp
- `POST /v1/audio/speech` - TTS
- `POST /api/pipeline` - one-shot speech → text → translate → speech
- `GET  /api/stack/state` - live component + memory status

## License

MIT (see `LICENSE`). Model weights are downloaded separately and remain under
their own licenses - review them before production use.
