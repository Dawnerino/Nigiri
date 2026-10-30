"""Read the machine's specs: CPU, system RAM, and GPU VRAM.

Cross-platform and dependency-light. psutil gives us CPU + system RAM
everywhere; GPU detection is best-effort per vendor:

  * NVIDIA  - nvidia-smi (query-gpu), present on Windows + Linux.
  * AMD     - rocm-smi / rsmi where available (Linux); falls back to VRAM=None.
  * Apple   - unified memory: the whole system RAM is usable as VRAM.
  * none    - VRAM=None, everything runs on CPU.

The result is a plain dict so it can be JSON-serialised straight to the UI.
"""
from __future__ import annotations

import os
import platform
import re
import shutil
import subprocess


def _run(cmd: list[str], timeout: int = 8) -> str:
    """Run a command, return stdout text, or '' on any failure."""
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        if proc.returncode != 0:
            return ""
        return proc.stdout or ""
    except (OSError, subprocess.SubprocessError):
        return ""


def _system_ram_mib() -> int:
    try:
        import psutil
        return int(psutil.virtual_memory().total // (1024 * 1024))
    except Exception:
        # Fallbacks without psutil.
        if platform.system() == "Linux":
            try:
                with open("/proc/meminfo") as fh:
                    for line in fh:
                        if line.startswith("MemTotal:"):
                            return int(int(line.split()[1]) * 1024 // (1024 * 1024))
            except OSError:
                pass
        elif platform.system() == "Darwin":
            out = _run(["sysctl", "-n", "hw.memsize"])
            m = re.search(r"(\d+)", out)
            if m:
                return int(int(m.group(1)) // (1024 * 1024))
        return 0


def _cpu_info() -> tuple[int, str]:
    cores = 0
    name = ""
    try:
        import psutil
        cores = psutil.cpu_count(logical=True) or 0
    except Exception:
        cores = os.cpu_count() or 0
    system = platform.system()
    if system == "Linux":
        try:
            with open("/proc/cpuinfo") as fh:
                for line in fh:
                    if line.lower().startswith("model name"):
                        name = line.split(":", 1)[1].strip()
                        break
        except OSError:
            pass
    elif system == "Darwin":
        name = _run(["sysctl", "-n", "machdep.cpu.brand_string"]).strip()
        if not name:
            name = "Apple Silicon"
    elif system == "Windows":
        out = _run(["wmic", "cpu", "get", "name"])
        for line in out.splitlines():
            if line.strip() and line.strip().lower() != "name":
                name = line.strip()
                break
    if not name:
        name = platform.processor() or "unknown CPU"
    return cores, name


class NvidiaGpu:
    def __init__(self, name: str, total_mib: int, free_mib: int):
        self.name = name
        self.total_mib = total_mib
        self.free_mib = free_mib


def _nvidia() -> NvidiaGpu | None:
    exe = shutil.which("nvidia-smi")
    if not exe:
        for cand in (
            "/usr/bin/nvidia-smi",
            "/usr/local/bin/nvidia-smi",
            r"C:\Windows\System32\nvidia-smi.exe",
            r"C:\Program Files\NVIDIA Corporation\NVSMI\nvidia-smi.exe",
        ):
            if os.path.exists(cand):
                exe = cand
                break
    if not exe:
        return None
    out = _run([exe, "--query-gpu=name,memory.total,memory.free",
                "--format=csv,noheader,nounits"])
    if not out.strip():
        return None
    best: NvidiaGpu | None = None
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 3:
            continue
        try:
            total = int(float(parts[1]))
            free = int(float(parts[2]))
        except ValueError:
            continue
        gpu = NvidiaGpu(name=parts[0], total_mib=total, free_mib=free)
        if best is None or free > best.free_mib:
            best = gpu
    return best


def _apple_unified() -> int:
    """On Apple Silicon, system RAM is the VRAM pool (minus OS reserve)."""
    if platform.system() != "Darwin":
        return 0
    return int(_system_ram_mib() * 0.75)


def _amd() -> NvidiaGpu | None:
    """Best-effort AMD VRAM via rocm-smi (Linux). Returns None if unavailable."""
    exe = shutil.which("rocm-smi")
    if not exe:
        return None
    out = _run([exe, "--showmeminfo", "vram", "--csv"])
    if not out.strip():
        return None
    for line in out.strip().splitlines()[1:]:
        parts = line.split(",")
        if len(parts) >= 3:
            try:
                total = int(float(parts[1])) // (1024 * 1024)
                used = int(float(parts[2])) // (1024 * 1024)
                return NvidiaGpu(name="AMD GPU", total_mib=total,
                                 free_mib=max(total - used, 0))
            except ValueError:
                continue
    return None


def detect() -> dict:
    """Return the full machine profile as a JSON-serialisable dict."""
    system = platform.system()  # Linux | Darwin | Windows
    cores, cpu_name = _cpu_info()
    ram_mib = _system_ram_mib()

    gpu: NvidiaGpu | None = _nvidia()
    gpu_vendor = "nvidia"
    if gpu is None and system == "Darwin":
        unified = _apple_unified()
        if unified > 0:
            gpu = NvidiaGpu(name="Apple Silicon (unified memory)",
                            total_mib=unified, free_mib=unified)
            gpu_vendor = "apple"
    if gpu is None:
        amd = _amd()
        if amd is not None:
            gpu = amd
            gpu_vendor = "amd"

    return {
        "os": system.lower(),
        "os_release": platform.release(),
        "arch": platform.machine(),
        "cpu_name": cpu_name,
        "cpu_cores": cores,
        "ram_mib": ram_mib,
        "has_gpu": gpu is not None,
        "gpu_vendor": gpu_vendor if gpu else "none",
        "gpu_name": gpu.name if gpu else "",
        "vram_total_mib": gpu.total_mib if gpu else 0,
        "vram_free_mib": gpu.free_mib if gpu else 0,
    }


if __name__ == "__main__":
    import json
    print(json.dumps(detect(), indent=2))

