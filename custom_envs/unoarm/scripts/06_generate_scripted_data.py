#!/usr/bin/env python
"""CLI: generate scripted Unoarm LeRobot training data from key-pose JSON."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data_gen.scripted import (  # noqa: E402
    DEFAULT_HOLD_NOISE_STD,
    DEFAULT_HOLD_STEPS,
    DEFAULT_IDLE_ACTION_TOLERANCE,
    DEFAULT_MIDPOINT_NOISE_STD,
    DEFAULT_POSE_JITTER_STD,
    DEFAULT_SEGMENT_STEPS,
    DEFAULT_TRIM_LEADING_IDLE,
    ScriptedGenConfig,
    generate_scripted_dataset,
)

DEFAULT_OUTPUT_ROOT = ROOT / "data" / "unoarm_prepare_fight_taunt"
DEFAULT_REPO_ID = "doki/unoarm_prepare_fight_taunt"
DEFAULT_POSES_JSON = ROOT / "data" / "poses_prepare_fight_taunt.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate scripted Unoarm LeRobot training data from key poses. "
            "Diversity comes from per-episode pose_jitter (smooth paths); "
            "avoid midpoint/hold noise which causes frame-level shake."
        ),
    )
    parser.add_argument(
        "--poses-json",
        type=Path,
        default=DEFAULT_POSES_JSON,
        help=(
            "JSON file describing the trajectory. Must contain 'task' (str) and 'poses' "
            "(list of {joint_name: value} dicts in CONTROL_JOINTS order). RAW_ZERO is "
            "automatically prepended/appended as the start/end pose."
        ),
    )
    parser.add_argument("--episodes", type=int, default=30, help="Number of scripted episodes to generate.")
    parser.add_argument(
        "--segment-steps",
        type=int,
        default=DEFAULT_SEGMENT_STEPS,
        help="Interpolation steps between two key poses.",
    )
    parser.add_argument(
        "--hold-steps",
        type=int,
        default=DEFAULT_HOLD_STEPS,
        help="Frames to hold at each key pose, including zero.",
    )
    parser.add_argument(
        "--midpoint-noise-std",
        type=float,
        default=DEFAULT_MIDPOINT_NOISE_STD,
        help="Frame-level noise on interpolation (discouraged; keep 0).",
    )
    parser.add_argument(
        "--hold-noise-std",
        type=float,
        default=DEFAULT_HOLD_NOISE_STD,
        help="Frame-level noise on hold frames (discouraged; keep 0).",
    )
    parser.add_argument(
        "--pose-jitter-std",
        type=float,
        default=DEFAULT_POSE_JITTER_STD,
        help=(
            "Per-episode fixed joint offset (radians) added to all key poses. "
            "Keeps each trajectory smooth while making episodes differ."
        ),
    )
    parser.add_argument(
        "--trim-leading-idle",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_TRIM_LEADING_IDLE,
        help="Drop leading no-motion frames before the first real action.",
    )
    parser.add_argument(
        "--idle-action-tolerance",
        type=float,
        default=DEFAULT_IDLE_ACTION_TOLERANCE,
        help="Raw joint tolerance used to detect leading idle actions.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Base random seed; each episode uses seed + episode_index.")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT, help="Dataset output directory.")
    parser.add_argument("--repo-id", type=str, default=DEFAULT_REPO_ID, help="LeRobot dataset repo id.")
    parser.add_argument("--overwrite", action="store_true", help="Remove an existing dataset at --output-root first.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = ScriptedGenConfig(
        poses_json=args.poses_json,
        episodes=args.episodes,
        segment_steps=args.segment_steps,
        hold_steps=args.hold_steps,
        midpoint_noise_std=args.midpoint_noise_std,
        hold_noise_std=args.hold_noise_std,
        pose_jitter_std=args.pose_jitter_std,
        trim_leading_idle=args.trim_leading_idle,
        idle_action_tolerance=args.idle_action_tolerance,
        seed=args.seed,
        output_root=args.output_root,
        repo_id=args.repo_id,
        overwrite=args.overwrite,
    )
    generate_scripted_dataset(cfg)


if __name__ == "__main__":
    main()
