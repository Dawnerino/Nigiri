#!/usr/bin/env python3
"""Nigiri entrypoint.

Parses the command line, records the requested bind host, and launches the
orchestrator (which owns the first-run wizard and the web UI). This file is the
single thing run.sh / run.bat exec, so the whole product is one command.

Arguments
---------
--host 127.0.0.1|0.0.0.0   Interface to bind. Default 127.0.0.1 (localhost only).
--port N                   Web/API port. Default 8765.
--reset                    Re-run the first-run wizard even if already set up.
--no-browser               Do not auto-open the browser.
--headless                 Alias for --no-browser.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RUNTIME = ROOT / "runtime"
CONFIG = RUNTIME / "config.json"


def parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="nigiri",
        description="Nigiri - a hand-pressed local voice pipeline.",
    )
    p.add_argument("--host", default="127.0.0.1",
                   help="bind interface: 127.0.0.1 (localhost, default) or 0.0.0.0 (LAN)")
    p.add_argument("--port", type=int, default=8765, help="web/API port (default 8765)")
    p.add_argument("--reset", action="store_true", help="re-run the first-run wizard")
    p.add_argument("--no-browser", "--headless", dest="no_browser",
                   action="store_true", help="do not auto-open the browser")
    return p.parse_args(argv)


def write_config(args: argparse.Namespace) -> None:
    """Persist the bind choice so the UI can read and toggle it."""
    RUNTIME.mkdir(parents=True, exist_ok=True)
    cfg = {}
    if CONFIG.exists():
        try:
            cfg = json.loads(CONFIG.read_text())
        except json.JSONDecodeError:
            cfg = {}
    cfg["host"] = args.host
    cfg["port"] = args.port
    cfg["no_browser"] = args.no_browser
    CONFIG.write_text(json.dumps(cfg, indent=2))


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    write_config(args)

    # Pass the bind choice to the orchestrator via env so it can re-read it on
    # a runtime toggle without a restart.
    os.environ["NIGIRI_HOST"] = args.host
    os.environ["NIGIRI_PORT"] = str(args.port)
    os.environ["NIGIRI_NO_BROWSER"] = "1" if args.no_browser else ""
    if args.reset:
        os.environ["NIGIRI_RESET"] = "1"

    # Import lazily so the venv bootstrap in run.sh/run.bat stays independent.
    sys.path.insert(0, str(ROOT))
    from server.orchestrator import main as serve  # noqa: E402

    return serve(host=args.host, port=args.port,
                 no_browser=args.no_browser, reset=args.reset)


if __name__ == "__main__":
    raise SystemExit(main())
