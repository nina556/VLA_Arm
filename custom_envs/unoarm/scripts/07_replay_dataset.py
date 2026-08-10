from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import mujoco.viewer
import numpy as np
import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gym_unoarm.constants import (  # noqa: E402
    FPS,
    SCENE_FREE_SPACE,
    SCENE_REACH_SWORD,
    SCENE_TABLE_PLACE,
)
from gym_unoarm.env import UnoarmEnv, configure_viewer_camera, configure_viewer_theme  # noqa: E402
from data_gen.table_place_ik import episode_peg_xy_from_meta  # noqa: E402

DEFAULT_DATASET_ROOT = ROOT / "data" / "unoarm_prepare_fight_taunt"
DEFAULT_REPO_ID = "doki/unoarm_prepare_fight_taunt"
RAW_ZERO = np.zeros(16, dtype=np.float32)
STATE_LABELS = ("L1", "L2", "L3", "L4", "L5", "L6", "L7", "LG", "R1", "R2", "R3", "R4", "R5", "R6", "R7", "RG")

_SCENE_CHOICES = {
    "free_space": SCENE_FREE_SPACE,
    "reach_sword": SCENE_REACH_SWORD,
    "table_place": SCENE_TABLE_PLACE,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay a scripted Unoarm LeRobot dataset episode.")
    parser.add_argument("--root", type=Path, default=DEFAULT_DATASET_ROOT, help="Local LeRobot dataset root.")
    parser.add_argument("--repo-id", type=str, default=DEFAULT_REPO_ID, help="LeRobot dataset repo id.")
    parser.add_argument("--episode", type=int, default=0, help="Episode index to replay.")
    parser.add_argument("--speed", type=float, default=1.0, help="Playback speed multiplier.")
    parser.add_argument("--loop", action="store_true", help="Loop the selected episode until the viewer closes.")
    parser.add_argument("--dry-run", action="store_true", help="Replay without opening the MuJoCo viewer.")
    parser.add_argument("--print-every", type=int, default=25, help="Print progress every N frames.")
    parser.add_argument("--state-every", type=int, default=1, help="Refresh terminal state display every N frames.")
    parser.add_argument("--no-state-display", action="store_true", help="Disable live terminal state display.")
    parser.add_argument(
        "--state-units",
        choices=("raw", "normalized", "both"),
        default="raw",
        help="Units for live state display. Raw matches MuJoCo Control slider values.",
    )
    parser.add_argument(
        "--show-viewer-ui",
        action="store_true",
        help="Show MuJoCo side UI. Hidden by default so replay controls are read-only.",
    )
    parser.add_argument(
        "--scene",
        type=str,
        default=None,
        help=(
            "Scene to load: free_space | reach_sword | table_place. "
            "If omitted, auto-detected from the dataset's meta json "
            "(table_place_ik_meta.json / reach_ik_meta.json), defaulting to free_space."
        ),
    )
    parser.add_argument(
        "--peg-xy",
        type=float,
        nargs=2,
        default=None,
        metavar=("X", "Y"),
        help=(
            "Override the cylinder (peg) XY position for table_place. "
            "Defaults to the peg_xy recorded in table_place_ik_meta.json."
        ),
    )
    return parser.parse_args()


def to_numpy(value) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=np.float32)


def load_episode(repo_id: str, root: Path, episode: int) -> LeRobotDataset:
    if not root.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {root}")
    return LeRobotDataset(
        repo_id,
        root=root,
        episodes=[episode],
        download_videos=False,
        video_backend="pyav",
        return_uint8=True,
    )


def format_values(values: np.ndarray) -> str:
    left = " ".join(f"{STATE_LABELS[i]}={values[i]:+.2f}" for i in range(8))
    right = " ".join(f"{STATE_LABELS[i]}={values[i]:+.2f}" for i in range(8, 16))
    return f"{left} | {right}"


def format_state_line(
    env: UnoarmEnv,
    frame_idx: int,
    num_frames: int,
    timestamp: float,
    normalized_state: np.ndarray,
    state_units: str,
) -> str:
    raw_state = env._denormalize(normalized_state)
    if state_units == "normalized":
        state_text = f"normalized: {format_values(normalized_state)}"
    elif state_units == "both":
        state_text = f"raw: {format_values(raw_state)} || normalized: {format_values(normalized_state)}"
    else:
        state_text = f"raw: {format_values(raw_state)}"
    return f"frame {frame_idx + 1}/{num_frames} t={timestamp:.3f}s | {state_text}"


def replay_once(
    dataset: LeRobotDataset,
    env: UnoarmEnv,
    viewer=None,
    speed: float = 1.0,
    print_every: int = 25,
    state_every: int = 1,
    state_display: bool = True,
    state_units: str = "raw",
) -> tuple[int, float]:
    env.reset(options={"state": env._normalize(RAW_ZERO)})
    max_state_error = 0.0
    delay = 0.0 if speed <= 0.0 else 1.0 / (FPS * speed)

    for frame_idx in range(dataset.num_frames):
        if viewer is not None and not viewer.is_running():
            return frame_idx, max_state_error

        item = dataset[frame_idx]
        action = np.clip(to_numpy(item["action"]).reshape(16), -1.0, 1.0)
        recorded_state = to_numpy(item["observation.state"]).reshape(16)

        obs, _reward, _terminated, _truncated, _info = env.step(action)
        replay_state = np.asarray(obs["agent_pos"], dtype=np.float32).reshape(16)
        state_error = float(np.max(np.abs(replay_state - recorded_state)))
        max_state_error = max(max_state_error, state_error)
        timestamp = float(to_numpy(item["timestamp"]))

        if viewer is not None:
            viewer.sync()
            if delay > 0.0:
                time.sleep(delay)

        if state_display and state_every > 0 and (frame_idx == 0 or (frame_idx + 1) % state_every == 0):
            print(
                "\r\033[2K"
                + format_state_line(env, frame_idx, dataset.num_frames, timestamp, replay_state, state_units),
                end="",
                flush=True,
            )
        elif print_every > 0 and (frame_idx == 0 or (frame_idx + 1) % print_every == 0):
            print(
                f"frame {frame_idx + 1}/{dataset.num_frames} "
                f"timestamp={timestamp:.3f}s max_state_error={max_state_error:.6f}"
            )

    if state_display:
        print()
    return dataset.num_frames, max_state_error


def resolve_scene_and_peg(
    root: Path,
    scene_override: str | None,
    peg_xy_override: tuple[float, float] | None,
    episode_index: int = 0,
) -> tuple[str, tuple[float, float] | None]:
    """Auto-detect the scene and peg XY from the dataset's meta json.

    For table_place, prefers per-episode ``peg_xy`` in ``table_place_ik_meta.json``,
    then falls back to legacy top-level ``peg_xy``. For reach_sword, reads
    ``reach_ik_meta.json`` (no peg). Falls back to free_space when no meta is found.
    """
    if scene_override is not None:
        scene = _SCENE_CHOICES.get(scene_override)
        if scene is None:
            raise ValueError(
                f"Unknown --scene {scene_override!r}; choose from {list(_SCENE_CHOICES.keys())}"
            )
    else:
        # Auto-detect from meta files.
        if (root / "table_place_ik_meta.json").exists():
            scene = SCENE_TABLE_PLACE
        elif (root / "reach_ik_meta.json").exists():
            scene = SCENE_REACH_SWORD
        else:
            scene = SCENE_FREE_SPACE

    peg_xy: tuple[float, float] | None = None
    if scene == SCENE_TABLE_PLACE:
        if peg_xy_override is not None:
            peg_xy = (float(peg_xy_override[0]), float(peg_xy_override[1]))
        else:
            meta_path = root / "table_place_ik_meta.json"
            if not meta_path.exists():
                raise FileNotFoundError(
                    f"--scene table_place requires table_place_ik_meta.json in {root} "
                    "(or pass --peg-xy X Y)."
                )
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            peg_xy = episode_peg_xy_from_meta(meta, episode_index)
            if peg_xy is None:
                raise KeyError(
                    f"peg_xy not found for episode {episode_index} in {meta_path}"
                )

    return scene, peg_xy


def main() -> None:
    args = parse_args()
    if args.episode < 0:
        raise ValueError(f"--episode must be >= 0, got {args.episode}")

    dataset = load_episode(args.repo_id, args.root, args.episode)
    print(
        f"loaded episode {args.episode} from {args.root}: "
        f"{dataset.num_frames} frames, {dataset.num_episodes} episode"
    )

    scene, peg_xy = resolve_scene_and_peg(
        args.root, args.scene, args.peg_xy, episode_index=args.episode
    )
    print(f"scene: {scene}")
    if peg_xy is not None:
        print(f"peg (cylinder) XY: {peg_xy}  (from dataset meta)")

    env_kwargs: dict = dict(
        obs_type="pixels_agent_pos", render_mode="rgb_array", max_episode_steps=10_000, scene=scene
    )
    if peg_xy is not None:
        env_kwargs["peg_xy"] = peg_xy
    env = UnoarmEnv(**env_kwargs)
    try:
        if args.dry_run:
            frames, max_error = replay_once(
                dataset,
                env,
                speed=0.0,
                print_every=args.print_every,
                state_every=args.state_every,
                state_display=not args.no_state_display,
                state_units=args.state_units,
            )
            print(f"dry run complete: frames={frames}, max_state_error={max_error:.6f}")
            return

        with mujoco.viewer.launch_passive(
            env.model,
            env.data,
            show_left_ui=args.show_viewer_ui,
            show_right_ui=args.show_viewer_ui,
        ) as viewer:
            configure_viewer_camera(viewer)
            configure_viewer_theme(viewer, env.model)
            while viewer.is_running():
                frames, max_error = replay_once(
                    dataset,
                    env,
                    viewer=viewer,
                    speed=args.speed,
                    print_every=args.print_every,
                    state_every=args.state_every,
                    state_display=not args.no_state_display,
                    state_units=args.state_units,
                )
                print(f"replay complete: frames={frames}, max_state_error={max_error:.6f}")
                if not args.loop:
                    break
    finally:
        env.close()


if __name__ == "__main__":
    main()
