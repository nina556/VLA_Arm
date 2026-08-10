#!/usr/bin/env python
"""CLI: generate Unoarm reach-IK training data from handle targets."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data_gen.reach_ik import (  # noqa: E402
    DEFAULT_APPROACH_OFFSET_M,
    DEFAULT_HOLD_STEPS,
    DEFAULT_SEGMENT_STEPS,
    DEFAULT_TASK,
    ReachIkGenConfig,
    generate_reach_ik_dataset,
)

DEFAULT_TARGETS_JSON = ROOT / "data" / "targets_reach_ik_example.json"
DEFAULT_OUTPUT_ROOT = ROOT / "data" / "unoarm_reach_ik"
DEFAULT_REPO_ID = "doki/unoarm_reach_ik"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate LeRobot training data with right-arm MuJoCo DLS IK: "
            "approach → grasp sword handle (grasp-only; no execution point). "
            "TCP orientation is fixed (gripper +X along −Y). "
            "Provide either --targets-json or --bbox-min/--bbox-max/--num-targets."
        ),
    )
    parser.add_argument(
        "--targets-json",
        type=Path,
        default=None,
        help=f"JSON with targets[].xyz. Example: {DEFAULT_TARGETS_JSON}",
    )
    parser.add_argument(
        "--bbox-min",
        type=float,
        nargs=3,
        metavar=("X", "Y", "Z"),
        default=None,
        help="AABB min corner for random handle sampling (meters).",
    )
    parser.add_argument(
        "--bbox-max",
        type=float,
        nargs=3,
        metavar=("X", "Y", "Z"),
        default=None,
        help="AABB max corner for random handle sampling (meters).",
    )
    parser.add_argument(
        "--num-targets",
        type=int,
        default=0,
        help="Number of random targets when using bbox mode.",
    )
    parser.add_argument(
        "--episodes-per-target",
        type=int,
        default=1,
        help="Episodes to record per successful target.",
    )
    parser.add_argument("--segment-steps", type=int, default=DEFAULT_SEGMENT_STEPS)
    parser.add_argument("--hold-steps", type=int, default=DEFAULT_HOLD_STEPS)
    parser.add_argument(
        "--pose-jitter-std",
        type=float,
        default=0.0,
        help="Per-episode joint jitter (rad); default 0 keeps Cartesian accuracy.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--repo-id", type=str, default=DEFAULT_REPO_ID)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--approach-offset",
        type=float,
        default=DEFAULT_APPROACH_OFFSET_M,
        help="Approach TCP offset along +Y before grasp (meters).",
    )
    parser.add_argument(
        "--task",
        type=str,
        default=DEFAULT_TASK,
        help="Language task string (bbox mode; JSON may override).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    has_json = args.targets_json is not None
    has_bbox = args.bbox_min is not None or args.bbox_max is not None
    if has_json and has_bbox:
        print("warning: both --targets-json and bbox flags set; using JSON list", flush=True)
    if not has_json and not (args.bbox_min is not None and args.bbox_max is not None):
        raise SystemExit(
            "Provide --targets-json or both --bbox-min and --bbox-max (with --num-targets)."
        )
    if not has_json and args.num_targets < 1:
        raise SystemExit("bbox mode requires --num-targets >= 1")

    cfg = ReachIkGenConfig(
        targets_json=args.targets_json,
        bbox_min=tuple(args.bbox_min) if args.bbox_min is not None else None,
        bbox_max=tuple(args.bbox_max) if args.bbox_max is not None else None,
        num_targets=int(args.num_targets),
        episodes_per_target=int(args.episodes_per_target),
        segment_steps=int(args.segment_steps),
        hold_steps=int(args.hold_steps),
        pose_jitter_std=float(args.pose_jitter_std),
        seed=int(args.seed),
        output_root=args.output_root,
        repo_id=args.repo_id,
        overwrite=bool(args.overwrite),
        approach_offset_m=float(args.approach_offset),
        task=str(args.task),
    )
    generate_reach_ik_dataset(cfg)


if __name__ == "__main__":
    main()
