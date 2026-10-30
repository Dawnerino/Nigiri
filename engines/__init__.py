"""Engine services: thin FastAPI shims that expose each stage of the pipeline
as an OpenAI-compatible endpoint on its own port. The orchestrator supervises
these as child processes and proxies to them.

  stt_engine.py         whisper-cli            -> :9101  /v1/audio/transcriptions
  translator_engine.py  llama-server (llama.cpp)-> :9102  /v1/chat/completions
  tts_engine.py         Supertonic / Qwen3-TTS  -> :9103  /v1/audio/speech
"""
