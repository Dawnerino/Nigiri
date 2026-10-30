"""Fetch the llama.cpp and whisper.cpp runtimes into ./bin.

Nigiri bundles nothing: on first run it downloads a prebuilt release for the
user's OS/arch, falling back to a source build if no prebuilt matches (or the
download fails). The orchestrator calls ensure() and, if a binary is missing,
the first-run wizard offers to install it - exactly the "ensure llama.cpp is
installed, if not offer it" requirement.

Prebuilt release assets are large zips; we extract only the binaries we need:
  * llama.cpp  -> llama-server  (OpenAI-compatible server for the translator)
  * whisper.cpp -> whisper-cli  (STT)

On Linux, if the project already has a working whisper-cli (e.g. copied in), we
leave it alone.
"""
from __future__ import annotations

import io
import os
import platform
import shutil
import stat
import subprocess
import tarfile
import tempfile
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BIN = ROOT / "bin"
# Each runtime gets its own subdir: llama.cpp and whisper.cpp both ship
# libggml* libraries at different versions, and sharing one flat dir lets the
# loader pick the wrong one (undefined-symbol crashes). Isolating them fixes it.
LLAMA_DIR = BIN / "llama"
WHISPER_DIR = BIN / "whisper"

# Pinned release tags - bump deliberately after testing. The llama.cpp tag must
# be new enough to know the HY-MT (hunyuan-dense) architecture; b12001 does.
LLAMA_TAG = "b12001"
WHISPER_TAG = "b5454"

LLAMA_REPO = "ggml-org/llama.cpp"
WHISPER_REPO = "ggml-org/whisper.cpp"

# Source ref used when we have to build whisper-cli ourselves (macOS publishes
# no CLI prebuilt - only a library xcframework). Kept in sync with WHISPER_TAG.
WHISPER_SRC_REF = WHISPER_TAG


def _exe(name: str, directory: Path) -> Path:
    return directory / (name + (".exe" if platform.system() == "Windows" else ""))


def llama_server_path() -> Path:
    return _exe("llama-server", LLAMA_DIR)


def whisper_cli_path() -> Path:
    return _exe("whisper-cli", WHISPER_DIR)


def _has(path: Path) -> bool:
    return path.exists() and os.access(path, os.X_OK)


def status() -> dict:
    """What runtimes are present. Drives the wizard's install offer."""
    return {
        "llama_server": _has(llama_server_path()),
        "whisper_cli": _has(whisper_cli_path()),
        "ffmpeg": bool(shutil.which("ffmpeg")),
        "bin_dir": str(BIN),
    }


# ------------------------------------------------------------------ asset naming
def _arch64(arch: str) -> str:
    return "arm64" if arch.lower() in ("arm64", "aarch64") else "x64"


def _llama_asset(os_name: str, arch: str) -> str:
    # ggml-org/llama.cpp CI release assets (CPU builds).
    a = _arch64(arch)
    if os_name == "Windows":
        return f"llama-{LLAMA_TAG}-bin-win-cpu-{a}.zip"
    if os_name == "Darwin":
        return f"llama-{LLAMA_TAG}-bin-macos-{a}.tar.gz"
    return f"llama-{LLAMA_TAG}-bin-ubuntu-{a}.tar.gz"


def _whisper_asset(os_name: str, arch: str) -> str:
    # ggml-org/whisper.cpp CI release assets (CPU builds).
    a = _arch64(arch)
    if os_name == "Windows":
        return f"whisper-bin-win-cpu-{a}.zip"
    if os_name == "Darwin":
        return "whisper-bin-macOS.zip"
    return f"whisper-bin-ubuntu-{a}.tar.gz"


def _download(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "nigiri-setup"})
    with urllib.request.urlopen(req, timeout=180) as r:
        return r.read()


def _is_lib(base: str) -> bool:
    return base.endswith((".so", ".dll", ".dylib")) or ".so." in base


def _chmod_x(p: Path) -> None:
    p.chmod(p.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)


def _make_soname_links(dest: Path) -> None:
    """Create missing .so / .so.<major> symlinks for versioned libraries.

    tarballs store libraries as real files (e.g. libllama.so.0.8.0) but the
    binaries link against sonames (libllama.so.0). Zip builds ship symlinks;
    tar builds don't, so we synthesise the short links here.
    """
    import re
    for p in dest.glob("*.so.*"):
        if p.is_symlink():
            continue
        m = re.match(r"^(lib.+?\.so)\.(\d+)", p.name)
        if not m:
            continue
        base, major = m.group(1), m.group(2)
        for link_name in (f"{base}.{major}", base):
            link = dest / link_name
            if not link.exists():
                try:
                    link.symlink_to(p.name)
                except OSError:
                    pass


def _extract_members(members, wanted: list[str], dest: Path) -> bool:
    """members: iterable of (basename, bytes). Extract wanted + all libs.

    llama.cpp / whisper.cpp prebuilts ship the executable plus a set of
    .so/.dll/.dylib siblings; the binary won't load without them. We drop them
    all flat into ./bin and set LD_LIBRARY_PATH (Unix) at launch.
    """
    got = False
    for base, data in members:
        if not base:
            continue
        if base in wanted or _is_lib(base):
            out = dest / base
            out.write_bytes(data)
            if base in wanted:
                _chmod_x(out)
                got = True
    _make_soname_links(dest)
    return got


def _iter_zip(blob: bytes):
    with zipfile.ZipFile(io.BytesIO(blob)) as zf:
        for info in zf.infolist():
            if info.is_dir():
                continue
            yield os.path.basename(info.filename), zf.read(info)


def _iter_targz(blob: bytes):
    with tarfile.open(fileobj=io.BytesIO(blob), mode="r:gz") as tf:
        for m in tf.getmembers():
            if not m.isfile():
                continue
            f = tf.extractfile(m)
            if f is None:
                continue
            yield os.path.basename(m.name), f.read()


def _fetch_release(repo: str, tag: str, asset: str, wanted: list[str],
                   dest: Path) -> bool:
    url = f"https://github.com/{repo}/releases/download/{tag}/{asset}"
    try:
        blob = _download(url)
    except Exception:
        return False
    try:
        members = _iter_zip(blob) if asset.endswith(".zip") else _iter_targz(blob)
        return _extract_members(members, wanted, dest)
    except (zipfile.BadZipFile, tarfile.TarError):
        return False


def _have(tools: list[str]) -> bool:
    return all(shutil.which(t) for t in tools)


def _run(cmd: list[str], cwd: Path, log_path: Path) -> bool:
    """Run a build step, appending stdout+stderr to log_path. True on success."""
    try:
        with log_path.open("ab") as log:
            proc = subprocess.run(cmd, cwd=str(cwd), stdout=log,
                                  stderr=subprocess.STDOUT, timeout=3600)
        return proc.returncode == 0
    except (subprocess.TimeoutExpired, OSError):
        return False


def _find_built(name: str, root: Path) -> Path | None:
    """Locate a freshly built executable by name anywhere under root."""
    for p in root.rglob(name):
        if p.is_file() and os.access(p, os.X_OK):
            return p
    return None


def _fix_macos_rpaths(exe: Path, lib_dir: Path) -> None:
    """Rewrite a Mach-O binary's rpaths to @executable_path (best effort).

    A whisper-cli built by cmake carries an absolute LC_RPATH into the build
    tree (and the temp dir we delete afterwards). macOS resolves @rpath against
    those paths and ignores LD_LIBRARY_PATH, so the copied binary aborts with
    "image not found". We delete the stale rpaths and add @executable_path (plus
    @loader_path) so it finds the dylibs sitting next to it. install_name_tool
    ships with Xcode CLT, which the build already required.
    """
    tool = shutil.which("install_name_tool")
    if not tool:
        return
    try:
        # Parse the binary's load commands with otool to find existing rpaths.
        otool = shutil.which("otool")
        rpaths = []
        if otool:
            listing = subprocess.run([otool, "-l", str(exe)], capture_output=True,
                                     text=True, timeout=60).stdout
            block = False
            for line in listing.splitlines():
                s = line.strip()
                if s.startswith("cmd LC_RPATH"):
                    block = True
                elif block and s.startswith("path "):
                    rpaths.append(s.split(" ", 1)[1].rsplit(" (offset", 1)[0].strip())
                elif block and s.startswith("Section"):
                    block = False
        args = [tool]
        for rp in rpaths:
            args += ["-delete_rpath", rp]
        args += ["-add_rpath", "@executable_path", "-add_rpath", "@loader_path",
                 str(exe)]
        subprocess.run(args, capture_output=True, text=True, timeout=60)
    except Exception:
        # Best effort: if rewriting fails the DYLD_LIBRARY_PATH launch fallback
        # still gives the binary a chance to resolve its libraries.
        pass


def _build_whisper_from_source() -> tuple[bool, str]:
    """Build whisper-cli from source into ./bin/whisper.

    macOS publishes no CLI prebuilt (only a library xcframework), so on Apple
    machines this is the primary install path. It also serves as the fallback
    for any platform whose prebuilt download failed. Requires git + cmake + a C
    compiler; returns (ok, message).
    """
    missing = [t for t in ("git", "cmake", "cc") if not shutil.which(t)]
    if missing:
        return False, ("whisper.cpp must be built from source but "
                       + ", ".join(missing) + " is missing. On macOS run "
                       "`xcode-select --install` (for git + cc) and "
                       "`brew install cmake`, then retry. "
                       "(On Linux: sudo apt install build-essential cmake git)")

    WHISPER_DIR.mkdir(parents=True, exist_ok=True)
    log_path = ROOT / "runtime" / "build-whisper.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix="nigiri-whisper-"))
    src = tmp / "whisper.cpp"
    try:
        if not _run(["git", "clone", "--depth", "1", "--branch", WHISPER_SRC_REF,
                     f"https://github.com/{WHISPER_REPO}.git", str(src)],
                    tmp, log_path):
            return False, "git clone of whisper.cpp failed - check network (see runtime/build-whisper.log)"
        if not _run(["cmake", "-B", "build", "-DCMAKE_BUILD_TYPE=Release"],
                    src, log_path):
            return False, "cmake configure failed for whisper.cpp (see runtime/build-whisper.log)"
        jobs = str(max(1, (os.cpu_count() or 2)))
        if not _run(["cmake", "--build", "build", "--config", "Release",
                     "-j", jobs], src, log_path):
            return False, "whisper.cpp build failed (see runtime/build-whisper.log)"
        built = _find_built("whisper-cli", src / "build")
        if built is None:
            return False, "whisper.cpp built but whisper-cli not found (see runtime/build-whisper.log)"
        shutil.copy2(built, whisper_cli_path())
        _chmod_x(whisper_cli_path())
        # Copy any sibling dylibs the CLI links against so it loads standalone.
        for lib in built.parent.glob("*.dylib"):
            shutil.copy2(lib, WHISPER_DIR / lib.name)
        # macOS: the copied binary keeps an absolute @rpath pointing at the
        # (now-deleted) temp build dir, and macOS ignores LD_LIBRARY_PATH. Point
        # every rpath at @executable_path so the binary finds the dylibs we just
        # copied beside it.
        if platform.system() == "Darwin":
            _fix_macos_rpaths(whisper_cli_path(), WHISPER_DIR)
        return True, "built whisper-cli from source (whisper.cpp)"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def ensure_llama() -> tuple[bool, str]:
    """Make sure llama-server exists. Returns (ok, message)."""
    if _has(llama_server_path()):
        return True, "llama-server already present"
    LLAMA_DIR.mkdir(parents=True, exist_ok=True)
    os_name, arch = platform.system(), platform.machine()
    asset = _llama_asset(os_name, arch)
    if _fetch_release(LLAMA_REPO, LLAMA_TAG, asset,
                      ["llama-server", "llama-server.exe"], LLAMA_DIR):
        return True, f"downloaded llama-server ({asset})"
    return False, (f"could not fetch llama-server prebuilt ({asset}); "
                   "build llama.cpp from source and drop llama-server into ./bin/llama")


def ensure_whisper() -> tuple[bool, str]:
    """Make sure whisper-cli exists. Returns (ok, message).

    macOS ships no CLI prebuilt (only a library xcframework), so there we build
    from source directly. On Linux/Windows we try the prebuilt first and fall
    back to a source build if the download fails.
    """
    if _has(whisper_cli_path()):
        return True, "whisper-cli already present"
    WHISPER_DIR.mkdir(parents=True, exist_ok=True)
    os_name, arch = platform.system(), platform.machine()

    if os_name != "Darwin":
        asset = _whisper_asset(os_name, arch)
        if _fetch_release(WHISPER_REPO, WHISPER_TAG, asset,
                          ["whisper-cli", "whisper-cli.exe", "main", "main.exe"],
                          WHISPER_DIR):
            # Recent whisper.cpp releases ship `main` as a tiny deprecated stub
            # that just prints a warning; the real CLI is `whisper-cli`. If only
            # `main` was present (older builds), promote it.
            stub = WHISPER_DIR / "main"
            if stub.exists() and not whisper_cli_path().exists():
                stub.replace(whisper_cli_path())
            return True, f"downloaded whisper-cli ({asset})"

    # No usable prebuilt (macOS, or a failed download): build from source.
    return _build_whisper_from_source()


def ensure_all() -> dict:
    """Ensure both runtimes; return per-runtime {ok, message} for the wizard."""
    ok_l, msg_l = ensure_llama()
    ok_w, msg_w = ensure_whisper()
    return {
        "llama_server": {"ok": ok_l, "message": msg_l},
        "whisper_cli": {"ok": ok_w, "message": msg_w},
    }


if __name__ == "__main__":
    import json
    print(json.dumps({"status": status(), "ensure": ensure_all()}, indent=2))
