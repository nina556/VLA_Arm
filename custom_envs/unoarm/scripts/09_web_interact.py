#!/usr/bin/env python
"""CLI entry: Unoarm ACT web console (pose design + settings).

Minimal launch — configure checkpoint / tasks in the Web UI:

  uv run python custom_envs/unoarm/scripts/09_web_interact.py
"""

from __future__ import annotations

import argparse
import os
import socket
import sys
from pathlib import Path

# Prefer EGL for offscreen rendering in the web server process.
os.environ.setdefault("MUJOCO_GL", "egl")

ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

import uvicorn  # noqa: E402
from webapp.app import build_app  # noqa: E402
from webapp.runner import UnoarmWebRunner, webconfig_from_settings  # noqa: E402
from webapp.settings_store import DEFAULT_SETTINGS_PATH, load_settings  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", type=str, default="0.0.0.0")  # nosec B104 - intentional LAN UI
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument(
        "--settings",
        type=Path,
        default=DEFAULT_SETTINGS_PATH,
        help=f"JSON settings path (default: {DEFAULT_SETTINGS_PATH})",
    )
    parser.add_argument(
        "--remove-sword",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override settings: remove sword from the reach_sword scene.",
    )
    parser.add_argument(
        "--remove-shield",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override settings: remove shield from the reach_sword scene.",
    )
    parser.add_argument(
        "--open",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Auto-open Windows Chrome with WebGL-friendly flags (default: on).",
    )
    return parser.parse_args()


def local_ip() -> str | None:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.connect(("8.8.8.8", 80))
            return sock.getsockname()[1]
    except OSError:
        return None


def _pids_listening_on_port(port: int) -> list[int]:
    """Return PIDs of processes listening on `port` (best-effort, cross-tool).

    Tries `ss -ltnp` first (needs root to see other users' PIDs, but on WSL the
    launcher usually owns the stale process), then falls back to `lsof`/`fuser`.
    """
    import subprocess

    me = os.getpid()
    pids: list[int] = []
    candidates: list[list[str]] = [
        ["ss", "-ltnp"],
        ["/usr/sbin/ss", "-ltnp"],
        ["lsof", f"-iTCP:{port}", "-sTCP:LISTEN", "-t"],
        ["fuser", f"{port}/tcp"],
    ]
    for cmd in candidates:
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=4)
        except (FileNotFoundError, PermissionError, subprocess.SubprocessError):
            continue
        out = (proc.stdout or "") + (proc.stderr or "")
        if not out.strip():
            continue
        # ss output: lines contain "0.0.0.0:7860" then "...users:((\"python\",pid=123,fd=...))"
        import re

        if "ss" in cmd[0]:
            for line in out.splitlines():
                if f":{port}" not in line:
                    continue
                for m in re.finditer(r"pid=(\d+)", line):
                    pid = int(m.group(1))
                    if pid != me:
                        pids.append(pid)
        else:
            for tok in out.split():
                tok = tok.strip()
                if tok.isdigit() and int(tok) != me:
                    pids.append(int(tok))
        if pids:
            break
    # de-dup, keep order
    seen: set[int] = set()
    uniq: list[int] = []
    for p in pids:
        if p not in seen:
            seen.add(p)
            uniq.append(p)
    return uniq


def free_port(port: int, *, timeout: float = 3.0) -> None:
    """Kill any stale process listening on `port` before we bind to it.

    Avoids the recurring ``[Errno 98] address already in use`` when a previous
    uvicorn instance (often a forgotten background launch) is still alive.
    Only acts on WSL/Linux; silently no-ops if nothing is listening.
    """
    import signal
    import time

    pids = _pids_listening_on_port(port)
    if not pids:
        return
    print(f"  Port {port} is in use by PID(s) {pids}; releasing...", flush=True)
    for pid in pids:
        for sig in (signal.SIGTERM, signal.SIGKILL):
            try:
                os.kill(pid, sig)
            except (ProcessLookupError, PermissionError):
                break
            # wait briefly for it to exit
            deadline = time.time() + timeout
            while time.time() < deadline:
                try:
                    os.kill(pid, 0)  # probe: raises if gone
                except (ProcessLookupError, PermissionError):
                    break
                time.sleep(0.1)
            else:
                continue
            break
    # Final settle so the kernel recycles the socket.
    time.sleep(0.5)


def open_windows_chrome(port: int) -> None:
    """Launch Windows Chrome so WebGL works (Cursor/WSL preview often has GL disabled)."""
    script = ROOT / "scripts" / "open_web_chrome.sh"
    if not script.is_file():
        return
    try:
        import subprocess

        subprocess.Popen(  # nosec B607 - invokes the fixed project launcher
            ["bash", str(script), str(port)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        print(f"  Browser: launching Windows Chrome -> http://127.0.0.1:{port}", flush=True)
    except OSError as exc:
        print(f"  Browser: auto-open failed ({exc})", flush=True)


def main() -> None:
    args = parse_args()
    # Release any stale process still bound to the target port (e.g. a forgotten
    # background launch) BEFORE loading the policy/env, so we don't waste a long
    # startup only to hit "[Errno 98] address already in use".
    free_port(args.port)
    settings = load_settings(args.settings)
    if args.remove_sword is not None:
        settings["remove_sword"] = bool(args.remove_sword)
    if args.remove_shield is not None:
        settings["remove_shield"] = bool(args.remove_shield)
    cfg = webconfig_from_settings(
        settings,
        host=args.host,
        port=args.port,
        settings_path=args.settings,
    )
    runner = UnoarmWebRunner(cfg)
    app = build_app(runner)
    ip = local_ip()
    print("\nUnoarm Web ready:", flush=True)
    if cfg.host == "0.0.0.0":  # nosec B104 - display logic for intentional LAN binding
        print(f"  WSL/Linux: http://127.0.0.1:{cfg.port}", flush=True)
        if ip:
            print(f"  Windows host: http://{ip}:{cfg.port}", flush=True)
    else:
        print(f"  URL: http://{cfg.host}:{cfg.port}", flush=True)
    print(f"  Settings: {cfg.settings_path}", flush=True)
    if runner.policy_ready:
        print(f"  Policy: ready ({cfg.checkpoint})", flush=True)
    else:
        print(f"  Policy: not loaded — open 设置 in the UI ({runner.policy_error})", flush=True)
    print(f"  Scene: {cfg.scene}", flush=True)
    print(f"  Remove sword: {cfg.remove_sword}", flush=True)
    print(f"  Remove shield: {cfg.remove_shield}", flush=True)
    if args.open:
        open_windows_chrome(cfg.port)
    else:
        print("  Tip: run scripts/open_web_chrome.sh to open Chrome with WebGL enabled.", flush=True)
    print("", flush=True)
    uvicorn.run(app, host=cfg.host, port=cfg.port, log_level="info")


if __name__ == "__main__":
    main()
