#!/usr/bin/env python
"""Standalone smoke test for the external MuJoCo VLA joint-chunk HTTP bridge.

Examples:

  # Dry-run against the LAN bridge host
  uv run python custom_envs/unoarm/scripts/test_vla_bridge.py

  # Actually execute on the remote MuJoCo driver
  uv run python custom_envs/unoarm/scripts/test_vla_bridge.py --execute --arms right
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from webapp.vla_bridge_client import (  # noqa: E402
    DEFAULT_BRIDGE_BASE_URL,
    BridgeConfig,
    VlaBridgeClient,
    denormalize_actions,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", type=str, default=DEFAULT_BRIDGE_BASE_URL)
    parser.add_argument("--arms", type=str, default="right", choices=["right", "left", "both"])
    parser.add_argument("--fps", type=float, default=20.0)
    parser.add_argument("--execute", action="store_true", help="Set execute=true (default dry-run).")
    parser.add_argument("--result-timeout-sec", type=float, default=5.0)
    parser.add_argument("--http-timeout-sec", type=float, default=8.0)
    parser.add_argument(
        "--joint-index",
        type=int,
        default=8,
        help="Which of the 16 dims to nudge (default: Right_Joint1=8).",
    )
    parser.add_argument("--target-rad", type=float, default=0.01)
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument(
        "--from-normalized",
        action="store_true",
        help="Treat --target-rad as normalized [-1,1] and denorm with --limits-json.",
    )
    parser.add_argument(
        "--limits-json",
        type=Path,
        default=None,
        help="Optional JSON file with shape (16,2) joint limits for --from-normalized.",
    )
    parser.add_argument("--instruction", type=str, default="bridge smoke test")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not (0 <= args.joint_index < 16):
        raise SystemExit("--joint-index must be in [0, 15]")
    if args.steps < 1:
        raise SystemExit("--steps must be >= 1")

    row = np.zeros(16, dtype=np.float64)
    row[args.joint_index] = float(args.target_rad)
    actions = np.repeat(row.reshape(1, 16), args.steps, axis=0)

    if args.from_normalized:
        if args.limits_json is None or not args.limits_json.exists():
            raise SystemExit("--from-normalized requires an existing --limits-json")
        limits = np.asarray(json.loads(args.limits_json.read_text(encoding="utf-8")), dtype=np.float32)
        actions = denormalize_actions(actions.astype(np.float32), limits)

    client = VlaBridgeClient(
        BridgeConfig(
            enabled=True,
            base_url=args.base_url,
            arms=args.arms,
            fps=args.fps,
            execute=args.execute,
            result_timeout_sec=args.result_timeout_sec,
            http_timeout_sec=args.http_timeout_sec,
        )
    )
    print(f"POST {client.endpoint}", flush=True)
    print(f"execute={args.execute} arms={args.arms} steps={args.steps}", flush=True)
    result = client.post_joint_chunk(actions, instruction=args.instruction)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if result.get("ok") is False:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
