"""Pick the best STT / translator / TTS for a given machine.

The rule of thumb, in priority order:

  1. If there is free VRAM, prefer GPU-resident models that fit in it.
  2. If there is no GPU (or no free VRAM), prefer CPU-friendly models: small.en
     for English STT, the 1.8B translator, and Supertonic 3 for TTS (which is
     built for CPU and needs no GPU at all).
  3. Never recommend a model whose runtime memory exceeds what the machine has.

recommend() returns, for each stage, the chosen id plus a short reason, and a
per-model device placement ("gpu" or "cpu") so the wizard can show where each
model will run.
"""
from __future__ import annotations

from setup import catalog

# Keep this much VRAM free as headroom for OS + compute buffers.
VRAM_HEADROOM_MIB = 400


def _fits_gpu(entry: dict, free_vram: int) -> bool:
    if not free_vram:
        return False
    need = entry.get("vram_mib", 0)
    if need <= 0:
        return False  # CPU-only model (Supertonic) - never "on GPU" by choice
    return need + VRAM_HEADROOM_MIB <= free_vram


def _fits_ram(entry: dict, ram: int) -> bool:
    # CPU inference needs the weights + activations in system RAM.
    return entry.get("ram_mib", 0) <= max(ram, 0)


def _pick(stage: list[dict], free_vram: int, ram: int, *, prefer_gpu: bool,
          want_multilingual: bool) -> tuple[dict, str]:
    """Choose one entry from a stage list, returning (entry, reason)."""
    # Candidates that fit somewhere.
    fits = [e for e in stage if _fits_gpu(e, free_vram) or _fits_ram(e, ram)]
    if not fits:
        # Nothing fully fits; take the smallest download as a best-effort.
        fits = sorted(stage, key=lambda e: e["download_mib"])[:1]
    if want_multilingual:
        multi = [e for e in fits if e.get("languages")]
        if multi:
            fits = multi

    if prefer_gpu and free_vram:
        gpu_ok = [e for e in fits if _fits_gpu(e, free_vram)]
        if gpu_ok:
            # Among GPU-fitting models, prefer the most accurate (slowest speed
            # number) that still leaves headroom - quality where we can afford it.
            best = sorted(gpu_ok, key=lambda e: (-e["speed"], e["download_mib"]))[0]
            return best, f"fits in {free_vram} MiB free VRAM - runs on GPU"

    # CPU path: prefer the fastest model that fits in RAM.
    cpu_ok = [e for e in fits if _fits_ram(e, ram)] or fits
    best = sorted(cpu_ok, key=lambda e: (e["speed"], e["download_mib"]))[0]
    return best, "runs on CPU (no spare VRAM)"


def _stage_entry(stage: list[dict], mid: str) -> dict:
    for e in stage:
        if e["id"] == mid:
            return e
    return {}


def recommend(profile: dict) -> dict:
    """Return {prefer_gpu, models, budget}.

    Each model gets {id, device, reason}. The set is chosen so the three
    engines can run *concurrently* within the machine's combined VRAM/RAM
    budget - not just each fitting in isolation.
    """
    free_vram = int(profile.get("vram_free_mib", 0))
    ram = int(profile.get("ram_mib", 0))
    has_gpu = bool(profile.get("has_gpu"))
    prefer_gpu = has_gpu and free_vram >= 1500

    stages = (("stt", catalog.STT), ("translator", catalog.TRANSLATOR),
              ("tts", catalog.TTS))

    # Pass 1: best per-stage pick (quality-first on GPU, speed-first on CPU).
    models: dict = {}
    for stage_name, stage_list in stages:
        entry, reason = _pick(stage_list, free_vram, ram,
                              prefer_gpu=prefer_gpu, want_multilingual=True)
        device = "gpu" if _fits_gpu(entry, free_vram) else "cpu"
        models[stage_name] = {"id": entry["id"], "device": device, "reason": reason}

    # Supertonic is always CPU-friendly; if chosen, force device=cpu.
    if models["tts"]["id"] == "supertonic-3":
        models["tts"]["device"] = "cpu"
        models["tts"]["reason"] = "ONNX, runs fast on CPU - no GPU needed"

    # Pass 2: if the set overflows the *combined* budget, downgrade the
    # heaviest GPU model and retry, until it fits or nothing is left to shed.
    downgraded = False
    stages_map = dict(stages)
    for _ in range(6):
        if _budget(models, free_vram, ram)["ok"]:
            break
        # Find the GPU-resident model with the largest VRAM footprint and
        # replace it with the next-smallest GPU-fitting option (or CPU).
        gpu_items = [(k, int(_stage_entry(stages_map[k], v["id"]).get("vram_mib", 0)))
                     for k, v in models.items() if v["device"] == "gpu"]
        if not gpu_items:
            break
        heaviest = sorted(gpu_items, key=lambda t: -t[1])[0][0]
        stage_list = stages_map[heaviest]
        smaller = [e for e in stage_list
                   if e["id"] != models[heaviest]["id"]
                   and (_fits_gpu(e, free_vram) or _fits_ram(e, ram))]
        if not smaller:
            break
        # Prefer the smallest GPU-fitting replacement, else smallest RAM fit.
        gpu_small = [e for e in smaller if _fits_gpu(e, free_vram)]
        pool = gpu_small or [e for e in smaller if _fits_ram(e, ram)] or smaller
        nxt = sorted(pool, key=lambda e: (e.get("vram_mib", 0), e["download_mib"]))[0]
        device = "gpu" if _fits_gpu(nxt, free_vram) else "cpu"
        models[heaviest] = {
            "id": nxt["id"], "device": device,
            "reason": "downgraded so all three fit in VRAM together",
        }
        downgraded = True

    if downgraded:
        for v in models.values():
            if "downgraded" in v["reason"]:
                continue
            if v["device"] == "gpu":
                v["reason"] = "fits in VRAM alongside the other models"

    out = {"prefer_gpu": prefer_gpu, "models": models,
           "budget": _budget(models, free_vram, ram)}
    return out


def _budget(models: dict, free_vram: int, ram: int) -> dict:
    """Sum the concurrent GPU/RAM footprint of a chosen set and flag overflow.

    `models` maps kind -> {id, device, reason}; the actual memory numbers come
    from the catalog entry for each id.
    """
    vram_total = 0
    ram_total = 0
    for kind, v in models.items():
        entry = catalog.by_id(kind, v.get("id", "")) or {}
        if v.get("device") == "gpu":
            vram_total += int(entry.get("vram_mib", 0))
        else:
            ram_total += int(entry.get("ram_mib", 0))
    vram_ok = (not free_vram) or vram_total + VRAM_HEADROOM_MIB <= free_vram
    ram_ok = ram_total <= max(ram, 0)
    return {
        "vram_total_mib": vram_total,
        "vram_free_mib": free_vram,
        "vram_ok": vram_ok,
        "ram_total_mib": ram_total,
        "ram_total_mib_note": "system RAM for CPU-resident models",
        "ram_ok": ram_ok,
        "ok": vram_ok and ram_ok,
    }


def fits_together(models: dict, profile: dict) -> bool:
    """True if the given {kind: entry} set runs concurrently within budget."""
    free_vram = int(profile.get("vram_free_mib", 0))
    ram = int(profile.get("ram_mib", 0))
    return _budget(models, free_vram, ram)["ok"]


if __name__ == "__main__":
    import json
    from setup import detect
    prof = detect.detect()
    print(json.dumps({"profile": prof, "recommendation": recommend(prof)}, indent=2))
