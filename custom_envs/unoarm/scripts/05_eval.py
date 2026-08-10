"""Evaluate a trained ACT policy on right-arm sword grasping (reach_ik).

Loads a checkpoint, runs N rollouts in the reach_sword scene with the sword
handle placed at random points inside the training AABB, and reports the
**grasp-attach** success rate: the right gripper must actually clamp the sword
handle (``info["right_attached"]``) by the end of the episode. Merely getting
the TCP within 5 cm is NOT enough.

Attach itself still uses the env proximity gate (default 5 cm) + gripper
closed — that is how MuJoCo snap-grasp works, not a separate "near = success"
metric.

The env config MUST match training: scene=reach_sword, remove_shield=True,
enable_execution_point=False. A mismatch silently breaks the observation
distribution.

Usage::

    uv run python custom_envs/unoarm/scripts/05_eval.py \
        --checkpoint data/outputs/020000/pretrained_model \
        --episodes 50
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gym_unoarm.constants import JOINTS, SCENE_REACH_SWORD  # noqa: E402
from gym_unoarm.env import UnoarmEnv  # noqa: E402

from lerobot.policies import make_pre_post_processors  # noqa: E402
from lerobot.policies.act import ACTPolicy  # noqa: E402
from lerobot.policies.utils import build_inference_frame  # noqa: E402

# Training AABB (from unoarm_train/reach_ik_meta.json). Validation targets are
# sampled INSIDE this box so we measure in-distribution fit, not extrapolation.
# NOTE: this dataset covers ONLY the left half of the workspace (X all < -0.2),
# so eval points are restricted to that region. Update if you retrain on a
# different AABB.
TRAIN_AABB_MIN = np.array([-0.401, -0.645, 0.954])
TRAIN_AABB_MAX = np.array([-0.202, -0.439, 1.141])

# Proximity gate used by the env to allow snap-attach (not a success metric).
# Kept for logging / diagnostics only.
REACH_NEAR_M = 0.05
MAX_STEPS = 150  # matches reach_ik episode length (~68 frames + margin)


def dataset_features() -> dict:
    """Schema mirroring the training dataset (top + left/right wrist RGB video)."""
    image_feature = {
        "shape": (480, 640, 3),
        "dtype": "video",
        "names": ["height", "width", "channels"],
        "info": {"is_depth_map": False},
    }
    return {
        "action": {"shape": (16,), "dtype": "float32", "names": list(JOINTS)},
        "observation.state": {"shape": (16,), "dtype": "float32", "names": list(JOINTS)},
        "observation.images.top": image_feature,
        "observation.images.left_wrist": copy.deepcopy(image_feature),
        "observation.images.right_wrist": copy.deepcopy(image_feature),
    }


def obs_to_frame(obs: dict, device: torch.device) -> dict:
    """Build a model-ready inference frame from a raw UnoarmEnv observation."""
    state_np = np.asarray(obs["agent_pos"], dtype=np.float32)
    lerobot_obs: dict = {name: float(state_np[i]) for i, name in enumerate(JOINTS)}
    lerobot_obs["top"] = np.asarray(obs["pixels"]["top"])
    lerobot_obs["left_wrist"] = np.asarray(obs["pixels"]["left_wrist"])
    lerobot_obs["right_wrist"] = np.asarray(obs["pixels"]["right_wrist"])
    return build_inference_frame(
        observation=lerobot_obs,
        ds_features=dataset_features(),
        device=device,
        robot_type="unoarm",
    )


def place_handle(env: UnoarmEnv, handle_xyz: np.ndarray) -> None:
    """Position the sword so its handle world XYZ matches handle_xyz."""
    from data_gen.reach_ik import place_sword_handle

    place_sword_handle(env, handle_xyz)


def run_eval(
    checkpoint: str,
    episodes: int,
    seed: int,
    render: bool,
    from_meta: str | None,
    n_action_steps: int = 10,
) -> dict:
    from lerobot.configs import PreTrainedConfig

    config = PreTrainedConfig.from_pretrained(checkpoint)
    chunk = int(getattr(config, "chunk_size", 50))
    if n_action_steps < 1:
        raise ValueError(f"n_action_steps must be >= 1, got {n_action_steps}")
    if n_action_steps > chunk:
        raise ValueError(f"n_action_steps ({n_action_steps}) cannot exceed chunk_size ({chunk})")
    config.n_action_steps = int(n_action_steps)
    print(f"[info] inference n_action_steps={config.n_action_steps} (chunk_size={chunk})")
    policy = ACTPolicy.from_pretrained(checkpoint, config=config)
    policy.eval()
    policy.config.n_action_steps = int(n_action_steps)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy = policy.to(device)

    preprocess, postprocess = make_pre_post_processors(
        policy.config,
        pretrained_path=checkpoint,
        preprocessor_overrides={"device_processor": {"device": str(device)}},
    )

    # Env config MUST match training (reach_ik grasp_only).
    env = UnoarmEnv(
        obs_type="pixels_agent_pos",
        render_mode="rgb_array",
        scene=SCENE_REACH_SWORD,
        remove_shield=True,
        enable_execution_point=False,
        terminate_on_success=False,
        max_episode_steps=MAX_STEPS + 10,
    )

    rng = np.random.default_rng(seed)
    successes = 0
    min_dists: list[float] = []
    per_episode = []

    # Build the list of validation targets. Two modes:
    # - from_meta: read the EXACT handle positions recorded in a reach_ik_meta.json
    #   (one per unique target_index), so we evaluate on real training-distribution
    #   points rather than random samples inside the AABB. This answers "did the
    #   model memorize/generalize on the actual points it was trained on?".
    # - default: uniform random inside TRAIN_AABB.
    if from_meta:
        import json as _json

        meta = _json.loads(Path(from_meta).read_text(encoding="utf-8"))
        # Deduplicate by target_index — each target has ~3 episodes with the
        # same handle; keep one copy so we test 1 point per unique target.
        seen: set[int] = set()
        unique_handles: list[np.ndarray] = []
        for e in meta["episodes"]:
            ti = e.get("target_index", e["episode_index"])
            if ti in seen:
                continue
            seen.add(ti)
            unique_handles.append(np.asarray(e["handle"], dtype=np.float64))
        rng.shuffle(unique_handles)
        eval_targets = unique_handles[:episodes]
        if len(eval_targets) < episodes:
            print(f"[warn] only {len(eval_targets)} unique targets in meta, requested {episodes}")
        print(f"[from_meta] using {len(eval_targets)} exact training targets from {from_meta}")
    else:
        eval_targets = [rng.uniform(TRAIN_AABB_MIN, TRAIN_AABB_MAX) for _ in range(episodes)]

    for ep, handle in enumerate(eval_targets):
        handle = np.asarray(handle, dtype=np.float64)
        env.reset()
        place_handle(env, handle)

        best_dist = float("inf")
        terminated = False
        truncated = False
        step = 0
        dist_traj: list[float] = []  # per-step reach_distance for failure analysis
        gripper_traj: list[bool] = []  # per-step right-gripper-closed flag
        attached_traj: list[bool] = []
        grip_close_step: int | None = None  # first step the gripper closed
        grip_close_dist_cm: float | None = None  # reach distance at that moment
        attach_step: int | None = None  # first step sword became attached
        attach_dist_cm: float | None = None
        final_attached = False
        last_info: dict = {}
        while not (terminated or truncated) and step < MAX_STEPS:
            obs_frame = obs_to_frame(env._get_obs(), device)
            processed = preprocess(obs_frame)
            with torch.inference_mode():
                action = policy.select_action(processed)
            action = postprocess(action)
            action_np = (
                action.squeeze(0).detach().cpu().numpy()
                if isinstance(action, torch.Tensor)
                else np.asarray(action).squeeze(0)
            )
            action_np = np.asarray(action_np, dtype=np.float32).reshape(-1)[:16]
            obs, _reward, terminated, truncated, info = env.step(action_np)
            last_info = info
            dist = float(info.get("reach_distance") or float("inf"))
            closed = bool(info.get("right_gripper_closed", False))
            attached = bool(info.get("right_attached", False))
            dist_traj.append(dist)
            gripper_traj.append(closed)
            attached_traj.append(attached)
            # Record the FIRST close event — when did the policy decide to grasp,
            # and how far was the TCP from the handle at that instant?
            if closed and grip_close_step is None:
                grip_close_step = step
                grip_close_dist_cm = dist * 100
            if attached and attach_step is None:
                attach_step = step
                attach_dist_cm = dist * 100
            best_dist = min(best_dist, dist)
            final_attached = attached
            step += 1

        # Full grasp success: sword must still be clamped at episode end.
        success = bool(final_attached)
        successes += int(success)
        min_dists.append(best_dist)
        # Downsample trajectories in lockstep so dist and gripper stay aligned.
        ds = max(1, len(dist_traj) // 50)
        per_episode.append(
            {
                "episode": ep,
                "handle_xyz": handle.tolist(),
                "min_dist_cm": best_dist * 100,
                "steps": step,
                "success": success,
                "terminated": terminated,
                "final_attached": final_attached,
                "ever_attached": attach_step is not None,
                "attach_step": attach_step,
                "attach_dist_cm": attach_dist_cm,
                "near_success_legacy": best_dist < REACH_NEAR_M,  # old <5cm metric
                # Gripper timing: when did the policy first close the right
                # gripper, and how far was the TCP from the handle then? A
                # close far from the target means premature grasp; a close at
                # ~0 cm means well-timed grasp; None means it never closed.
                "gripper_first_close_step": grip_close_step,
                "gripper_first_close_dist_cm": grip_close_dist_cm,
                "gripper_ever_closed": grip_close_step is not None,
                # Aligned trajectories (same downsample) for plotting.
                "dist_traj_cm": [d * 100 for d in dist_traj[::ds]],
                "gripper_closed_traj": gripper_traj[::ds],
                "attached_traj": attached_traj[::ds],
                "env_success": bool(last_info.get("success", False)),
            }
        )
        tag = "OK" if success else "FAIL"
        grip_str = (
            f"grip@step{grip_close_step}({grip_close_dist_cm:.0f}cm)"
            if grip_close_step is not None
            else "grip:never"
        )
        attach_str = (
            f"attach@step{attach_step}({attach_dist_cm:.0f}cm)" if attach_step is not None else "attach:never"
        )
        print(
            f"[{tag}] ep{ep + 1:3d}/{episodes} handle=[{handle[0]:+.3f},{handle[1]:+.3f},{handle[2]:+.3f}] "
            f"min_dist={best_dist * 100:.1f}cm {grip_str} {attach_str} "
            f"final_attached={int(final_attached)} steps={step}"
        )

    env.close()

    min_dists_arr = np.array(min_dists)
    rate = successes / episodes
    # Per-axis breakdown: which region of the AABB fails most? Useful for
    # deciding whether to collect more data in a specific sub-region.
    success_arr = np.array([e["success"] for e in per_episode])
    handle_arr = np.array([e["handle_xyz"] for e in per_episode])
    summary = {
        "checkpoint": str(checkpoint),
        "seed": seed,
        "target_source": "from_meta" if from_meta else "uniform_aabb",
        "from_meta_path": str(from_meta) if from_meta else None,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "scene": SCENE_REACH_SWORD,
        "env": {"remove_shield": True, "enable_execution_point": False},
        "success_criterion": "final_right_attached",
        "attach_near_threshold_cm": REACH_NEAR_M * 100,
        "n_action_steps": int(n_action_steps),
        "max_steps": MAX_STEPS,
        "train_aabb_min": TRAIN_AABB_MIN.tolist(),
        "train_aabb_max": TRAIN_AABB_MAX.tolist(),
        "episodes": episodes,
        "successes": successes,
        "success_rate": rate,
        "mean_min_dist_cm": float(min_dists_arr.mean() * 100),
        "median_min_dist_cm": float(np.median(min_dists_arr) * 100),
        "max_min_dist_cm": float(min_dists_arr.max() * 100),
        # Legacy proximity-only rate (old metric) for comparison.
        "legacy_near_success_rate": float(np.mean([e["near_success_legacy"] for e in per_episode])),
        "attach_rate_ever": float(np.mean([e["ever_attached"] for e in per_episode])),
        "attach_rate_final": float(np.mean([e["final_attached"] for e in per_episode])),
        # Success rate when splitting episodes at each axis median — reveals
        # whether one half of the workspace is systematically harder.
        "success_rate_by_half": {
            axis: {
                "low": float(success_arr[handle_arr[:, i] < np.median(handle_arr[:, i])].mean()),
                "high": float(success_arr[handle_arr[:, i] >= np.median(handle_arr[:, i])].mean()),
            }
            for i, axis in enumerate(["x", "y", "z"])
        },
        # Gripper-timing aggregation. "close_dist" is how far the TCP was from
        # the handle when the policy first closed the gripper -- small means a
        # well-timed grasp, large means premature. None-list excluded.
        "gripper": {
            "ever_closed_rate": float(np.mean([e["gripper_ever_closed"] for e in per_episode])),
            "mean_first_close_step": float(
                np.mean(
                    [
                        e["gripper_first_close_step"]
                        for e in per_episode
                        if e["gripper_first_close_step"] is not None
                    ]
                )
            )
            if any(e["gripper_first_close_step"] is not None for e in per_episode)
            else None,
            "mean_first_close_dist_cm": float(
                np.mean(
                    [
                        e["gripper_first_close_dist_cm"]
                        for e in per_episode
                        if e["gripper_first_close_dist_cm"] is not None
                    ]
                )
            )
            if any(e["gripper_first_close_dist_cm"] is not None for e in per_episode)
            else None,
        },
    }

    # Persist results next to the checkpoint: <run>/eval/eval_<timestamp>/
    ckpt_dir = Path(checkpoint).resolve()
    run_root = ckpt_dir.parents[2]  # .../<run>/
    stamp = time.strftime("%Y%m%d_%H%M%S")
    out_dir = run_root / "eval" / f"eval_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    with (out_dir / "episodes.json").open("w", encoding="utf-8") as fh:
        json.dump(per_episode, fh, indent=2, ensure_ascii=False)
    with (out_dir / "episodes.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            [
                "episode",
                "handle_x",
                "handle_y",
                "handle_z",
                "min_dist_cm",
                "steps",
                "success",
                "final_attached",
                "ever_attached",
                "attach_step",
                "attach_dist_cm",
                "near_success_legacy",
                "gripper_ever_closed",
                "gripper_first_close_step",
                "gripper_first_close_dist_cm",
            ]
        )
        for e in per_episode:
            writer.writerow(
                [
                    e["episode"],
                    *e["handle_xyz"],
                    f"{e['min_dist_cm']:.2f}",
                    e["steps"],
                    int(e["success"]),
                    int(e["final_attached"]),
                    int(e["ever_attached"]),
                    e["attach_step"] if e["attach_step"] is not None else "",
                    f"{e['attach_dist_cm']:.2f}" if e["attach_dist_cm"] is not None else "",
                    int(e["near_success_legacy"]),
                    int(e["gripper_ever_closed"]),
                    e["gripper_first_close_step"] if e["gripper_first_close_step"] is not None else "",
                    f"{e['gripper_first_close_dist_cm']:.2f}"
                    if e["gripper_first_close_dist_cm"] is not None
                    else "",
                ]
            )

    print()
    print("=" * 60)
    print(f"SUCCESS RATE (final grasp attach): {successes}/{episodes} = {rate * 100:.1f}%")
    print(
        f"legacy near(<{REACH_NEAR_M * 100:.0f}cm) rate: {summary['legacy_near_success_rate'] * 100:.1f}%  "
        f"ever_attached: {summary['attach_rate_ever'] * 100:.1f}%"
    )
    print(
        f"min reach distance: mean={summary['mean_min_dist_cm']:.1f}cm "
        f"median={summary['median_min_dist_cm']:.1f}cm max={summary['max_min_dist_cm']:.1f}cm"
    )
    print("(success = sword still clamped by right gripper at episode end)")
    sr = summary["success_rate_by_half"]
    print(
        f"success by half  x[low/high]: {sr['x']['low'] * 100:.0f}%/{sr['x']['high'] * 100:.0f}%  "
        f"y: {sr['y']['low'] * 100:.0f}%/{sr['y']['high'] * 100:.0f}%  "
        f"z: {sr['z']['low'] * 100:.0f}%/{sr['z']['high'] * 100:.0f}%"
    )
    g = summary["gripper"]
    grip_rate = g["ever_closed_rate"] * 100
    grip_step = g["mean_first_close_step"]
    grip_dist = g["mean_first_close_dist_cm"]
    grip_step_s = f"{grip_step:.0f}" if grip_step is not None else "n/a"
    grip_dist_s = f"{grip_dist:.1f}cm" if grip_dist is not None else "n/a"
    print(
        f"gripper closed in {grip_rate:.0f}% of episodes; "
        f"avg first close @ step {grip_step_s} (dist {grip_dist_s})"
    )
    print(f"logs saved to: {out_dir}")
    print("=" * 60)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="data/outputs/020000/pretrained_model",
        help="Path to the pretrained_model directory.",
    )
    parser.add_argument("--episodes", type=int, default=50, help="Number of validation rollouts.")
    parser.add_argument("--seed", type=int, default=12345, help="Random seed for target sampling.")
    parser.add_argument(
        "--render", action="store_true", help="Launch MuJoCo viewer (off by default for speed)."
    )
    parser.add_argument(
        "--from_meta",
        type=str,
        default=None,
        help="Path to a reach_ik_meta.json. When set, validation targets are read "
        "from it (one per unique target_index) instead of sampled uniformly. Use "
        "this to evaluate on the EXACT points the model was trained on.",
    )
    parser.add_argument(
        "--n_action_steps",
        type=int,
        default=10,
        help="How many predicted actions to execute before re-querying the policy "
        "(must be <= checkpoint chunk_size). Default 10 ≈ 0.5s at 20 FPS.",
    )
    args = parser.parse_args()
    run_eval(
        args.checkpoint,
        args.episodes,
        args.seed,
        args.render,
        args.from_meta,
        args.n_action_steps,
    )


if __name__ == "__main__":
    main()
