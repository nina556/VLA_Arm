"""Scripted Unoarm LeRobot dataset generation from key-pose JSON files.

Extracted from ``scripts/06_generate_scripted_data.py`` so CLI and Web share
the same implementation.
"""

from __future__ import annotations

import copy
import json
import shutil
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from gym_unoarm.constants import CONTROL_JOINTS, FPS, JOINTS

if TYPE_CHECKING:
    from gym_unoarm.env import UnoarmEnv

EXPECTED_ACTION_ORDER = (
    "Left_Joint1",
    "Left_Joint2",
    "Left_Joint3",
    "Left_Joint4",
    "Left_Joint5",
    "Left_Joint6",
    "Left_Joint7",
    "Left_Gripper_Joint",
    "Right_Joint1",
    "Right_Joint2",
    "Right_Joint3",
    "Right_Joint4",
    "Right_Joint5",
    "Right_Joint6",
    "Right_Joint7",
    "Right_Gripper_Joint",
)

if tuple(CONTROL_JOINTS) != EXPECTED_ACTION_ORDER:
    raise RuntimeError(f"Unexpected Unoarm action order: {CONTROL_JOINTS}")

RAW_ZERO = np.zeros(16, dtype=np.float32)

LogFn = Callable[[str], None]

# Prefer episode-level diversity (pose_jitter) over frame-level noise.
# Midpoint/hold noise makes trajectories shake and teaches the policy to jitter.
DEFAULT_SEGMENT_STEPS = 20
DEFAULT_HOLD_STEPS = 2
DEFAULT_POSE_JITTER_STD = 0.05  # radians, fixed offset per episode
DEFAULT_MIDPOINT_NOISE_STD = 0.0
DEFAULT_HOLD_NOISE_STD = 0.0


@dataclass
class ScriptedGenConfig:
    poses_json: Path
    episodes: int = 30
    segment_steps: int = DEFAULT_SEGMENT_STEPS
    hold_steps: int = DEFAULT_HOLD_STEPS
    midpoint_noise_std: float = DEFAULT_MIDPOINT_NOISE_STD
    hold_noise_std: float = DEFAULT_HOLD_NOISE_STD
    pose_jitter_std: float = DEFAULT_POSE_JITTER_STD
    seed: int = 0
    output_root: Path | None = None
    repo_id: str = "doki/unoarm_scripted"
    overwrite: bool = False


def dataset_features() -> dict[str, Any]:
    # RGB stored as video (dtype "video") so frames are H.264-encoded into mp4
    # instead of living as raw uint8 in the parquet/Arrow table. This is the
    # single biggest memory win: 64 frames of raw RGB eat ~1.4GB in the HF
    # Arrow table, vs ~negligible when streamed from mp4. Requires passing
    # use_videos=True to LeRobotDataset.create.
    image_feature = {
        "dtype": "video",
        "shape": (480, 640, 3),
        "names": ["height", "width", "channels"],
        "info": {"is_depth_map": False},
    }
    return {
        "action": {"dtype": "float32", "shape": (16,), "names": list(JOINTS)},
        "observation.state": {"dtype": "float32", "shape": (16,), "names": list(JOINTS)},
        "observation.images.top": image_feature,
        "observation.images.left_wrist": copy.deepcopy(image_feature),
        "observation.images.right_wrist": copy.deepcopy(image_feature),
    }


def add_frame(dataset: Any, obs: dict, action: np.ndarray, task: str) -> None:
    dataset.add_frame(
        {
            "observation.state": obs["agent_pos"].astype(np.float32),
            "observation.images.top": obs["pixels"]["top"],
            "observation.images.left_wrist": obs["pixels"]["left_wrist"],
            "observation.images.right_wrist": obs["pixels"]["right_wrist"],
            "action": action.astype(np.float32),
            "task": task,
        }
    )


def load_poses_from_json(path: Path) -> tuple[str, list[np.ndarray]]:
    """Load key poses and task description from a JSON file.

    Expected schema:
        {
            "task": "<language instruction>",
            "poses": [
                {"Left_Joint1": 1.0, "Left_Joint2": 0.41, ...},
                ...
            ]
        }

    Joint names must match CONTROL_JOINTS. Missing joints default to 0.0; unknown
    joints raise ValueError. The returned poses are the user-supplied ones only —
    RAW_ZERO is prepended/appended later by build_pose_sequence().
    """
    if not path.exists():
        raise FileNotFoundError(f"Poses JSON not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict) or "task" not in data or "poses" not in data:
        raise ValueError(
            "JSON must be an object with 'task' (str) and 'poses' (list) keys. "
            f"Got: {list(data.keys()) if isinstance(data, dict) else type(data)}"
        )
    task = str(data["task"])
    raw_poses = data["poses"]
    if not isinstance(raw_poses, list) or len(raw_poses) == 0:
        raise ValueError(f"'poses' must be a non-empty list, got: {type(raw_poses)}")

    valid_names = set(CONTROL_JOINTS)
    poses: list[np.ndarray] = []
    for i, pose_dict in enumerate(raw_poses):
        if not isinstance(pose_dict, dict):
            raise ValueError(
                f"pose #{i} must be an object mapping joint names to values, got {type(pose_dict)}"
            )
        unknown = set(pose_dict.keys()) - valid_names
        if unknown:
            raise ValueError(
                f"pose #{i} contains unknown joint names: {sorted(unknown)}. "
                f"Valid names: {list(CONTROL_JOINTS)}"
            )
        vec = np.asarray(
            [float(pose_dict.get(name, 0.0)) for name in CONTROL_JOINTS],
            dtype=np.float32,
        )
        poses.append(vec)
    return task, poses


def build_pose_sequence(user_poses: list[np.ndarray]) -> list[np.ndarray]:
    """Wrap user-supplied key poses with RAW_ZERO at both ends."""
    if len(user_poses) == 0:
        raise ValueError("user_poses must contain at least one key pose")
    return [RAW_ZERO.copy(), *(pose.copy() for pose in user_poses), RAW_ZERO.copy()]


def clip_raw_pose(env: UnoarmEnv, raw_pose: np.ndarray) -> np.ndarray:
    low = env._joint_limits[:, 0]
    high = env._joint_limits[:, 1]
    return np.clip(raw_pose.astype(np.float32, copy=False), low, high).astype(np.float32)


def warn_if_raw_pose_clipped(
    env: UnoarmEnv,
    name: str,
    raw_pose: np.ndarray,
    *,
    log: LogFn = print,
) -> None:
    clipped = clip_raw_pose(env, raw_pose)
    changed = np.flatnonzero(np.abs(clipped - raw_pose) > 1e-6)
    for idx in changed:
        log(
            f"warning: {name}[{idx}] {CONTROL_JOINTS[idx]}={raw_pose[idx]:+.6f} "
            f"is outside the model range and will use {clipped[idx]:+.6f}"
        )


def interpolate(
    start: np.ndarray,
    target: np.ndarray,
    steps: int,
    rng: np.random.Generator,
    midpoint_noise_std: float,
) -> list[np.ndarray]:
    if steps < 1:
        raise ValueError(f"--segment-steps must be >= 1, got {steps}")

    low = np.minimum(start, target)
    high = np.maximum(start, target)
    actions: list[np.ndarray] = []
    for alpha in np.linspace(1.0 / steps, 1.0, steps, dtype=np.float32):
        action = ((1.0 - alpha) * start + alpha * target).astype(np.float32)
        if midpoint_noise_std > 0.0 and alpha < 1.0:
            scale = float(np.sin(np.pi * float(alpha)))
            noise = rng.normal(0.0, midpoint_noise_std * scale, size=action.shape).astype(np.float32)
            action = np.clip(action + noise, low, high).astype(np.float32)
        actions.append(action)
    return actions


def build_episode_actions(
    poses: list[np.ndarray],
    segment_steps: int,
    hold_steps: int,
    rng: np.random.Generator,
    midpoint_noise_std: float,
    hold_noise_std: float = 0.0,
    start: np.ndarray | None = None,
) -> list[np.ndarray]:
    if hold_steps < 0:
        raise ValueError(f"--hold-steps must be >= 0, got {hold_steps}")
    if midpoint_noise_std < 0.0:
        raise ValueError(f"--midpoint-noise-std must be >= 0, got {midpoint_noise_std}")
    if hold_noise_std < 0.0:
        raise ValueError(f"--hold-noise-std must be >= 0, got {hold_noise_std}")

    def hold(target: np.ndarray) -> list[np.ndarray]:
        if hold_noise_std <= 0.0:
            return [target.copy() for _ in range(hold_steps)]
        frames: list[np.ndarray] = []
        for _ in range(hold_steps):
            noise = rng.normal(0.0, hold_noise_std, size=target.shape).astype(np.float32)
            frames.append((target + noise).astype(np.float32))
        return frames

    start_pose = RAW_ZERO.copy() if start is None else np.asarray(start, dtype=np.float32).reshape(16).copy()
    actions: list[np.ndarray] = hold(start_pose)
    current = start_pose.copy()
    for target in poses:
        actions.extend(interpolate(current, target, segment_steps, rng, midpoint_noise_std))
        actions.extend(hold(target))
        current = target.copy()
    return actions


def images_are_black(obs: dict) -> bool:
    return all(np.asarray(image).max(initial=0) == 0 for image in obs["pixels"].values())


def generate_scripted_dataset(config: ScriptedGenConfig, *, log: LogFn = print) -> Path:
    """Generate a LeRobot dataset from a poses JSON file. Returns output_root."""
    if config.episodes < 1:
        raise ValueError(f"--episodes must be >= 1, got {config.episodes}")
    if config.pose_jitter_std < 0.0:
        raise ValueError(f"--pose-jitter-std must be >= 0, got {config.pose_jitter_std}")
    if config.midpoint_noise_std > 0.0 or config.hold_noise_std > 0.0:
        log(
            "warning: midpoint/hold noise adds frame-level shake; prefer pose_jitter_std "
            "for episode diversity (smooth trajectories)."
        )

    output_root = config.output_root
    if output_root is None:
        stem = config.poses_json.stem.removesuffix(".poses")
        output_root = config.poses_json.parent / f"unoarm_{stem}"

    task_desc, user_poses = load_poses_from_json(config.poses_json)
    log(f"loaded {len(user_poses)} key poses and task {task_desc!r} from {config.poses_json}")

    from gym_unoarm.env import UnoarmEnv

    env_for_check = UnoarmEnv(obs_type="pixels_agent_pos", render_mode="rgb_array", max_episode_steps=10_000)
    try:
        for i, pose in enumerate(user_poses):
            warn_if_raw_pose_clipped(env_for_check, f"pose#{i}", pose, log=log)
    finally:
        env_for_check.close()

    if output_root.exists():
        if not config.overwrite:
            raise FileExistsError(f"{output_root} already exists. Pass overwrite=True to replace it.")
        log(f"Removing existing dataset at {output_root}")
        shutil.rmtree(output_root)

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    dataset = LeRobotDataset.create(
        repo_id=config.repo_id,
        fps=FPS,
        features=dataset_features(),
        root=output_root,
        robot_type="unoarm",
        use_videos=True,
    )

    env = UnoarmEnv(
        obs_type="pixels_agent_pos",
        render_mode="rgb_array",
        max_episode_steps=10_000,
        # Three-view observation (top + both wrists); all rendered per frame.
    )
    warned_black_images = False

    try:
        for ep in range(config.episodes):
            rng = np.random.default_rng(config.seed + ep)
            obs, _info = env.reset(options={"state": env._normalize(clip_raw_pose(env, RAW_ZERO))})

            if config.pose_jitter_std > 0.0:
                # One fixed offset per episode: all key poses shift together so the
                # path stays smooth while episodes differ from each other.
                jitter = rng.normal(0.0, config.pose_jitter_std, size=16).astype(np.float32)
                jittered_user_poses = [pose + jitter for pose in user_poses]
            else:
                jitter = None
                jittered_user_poses = user_poses
            full_poses = build_pose_sequence(jittered_user_poses)
            raw_poses = [clip_raw_pose(env, pose) for pose in full_poses]
            raw_actions = build_episode_actions(
                raw_poses,
                config.segment_steps,
                config.hold_steps,
                rng,
                config.midpoint_noise_std,
                config.hold_noise_std,
            )

            for raw_action in raw_actions:
                action = env._normalize(clip_raw_pose(env, raw_action))
                obs, _reward, _terminated, _truncated, _info = env.step(action)
                if not warned_black_images and images_are_black(obs):
                    log("warning: rendered images are all black; check MuJoCo offscreen rendering.")
                    warned_black_images = True
                add_frame(dataset, obs, action.copy(), task_desc)

            dataset.save_episode()
            jitter_msg = f", pose_jitter_l2={float(np.linalg.norm(jitter)):.4f}" if jitter is not None else ""
            log(f"saved episode {ep + 1}/{config.episodes}: {len(raw_actions)} frames{jitter_msg}")
    finally:
        dataset.finalize()
        env.close()
        log(f"dataset finalized at {output_root}")

    return Path(output_root)
