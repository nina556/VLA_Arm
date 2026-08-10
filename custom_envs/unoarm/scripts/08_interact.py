#!/usr/bin/env python
"""Interactive ACT inference for the Unoarm simulation.

Loads a trained ACT policy checkpoint and drives the robot in the MuJoCo
viewer. Press Enter at the prompt to run one episode (a fresh rollout from the
rest pose); the policy is purely vision+state, so no language instruction is
needed.

    uv run python custom_envs/unoarm/scripts/08_interact.py \
        --checkpoint outputs/train/.../checkpoints/.../pretrained_model

Notes
-----
- The reset state is the all-zero pose (RAW_ZERO), matching what the data
  generation script (06_generate_scripted_data.py) used, so the policy starts
  from an in-distribution state.
- Three cameras (top + left/right wrist) are fed to the policy, matching the
  training dataset.
"""

from __future__ import annotations

import argparse
import copy
import sys
import time
from pathlib import Path

import mujoco.viewer
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lerobot.policies import make_pre_post_processors  # noqa: E402
from lerobot.policies.act import ACTPolicy  # noqa: E402
from lerobot.policies.utils import build_inference_frame  # noqa: E402

from gym_unoarm.constants import FPS, JOINTS  # noqa: E402
from gym_unoarm.env import (  # noqa: E402
    UnoarmEnv,
    configure_viewer_camera,
    configure_viewer_theme,
)

DEFAULT_CHECKPOINT = (
    ROOT.parent.parent / "data" / "outputs" / "mutiTask" / "10000步权重" / "pretrained_model"
)
RAW_ZERO = np.zeros(16, dtype=np.float32)
STATE_LABELS = (
    "L1",
    "L2",
    "L3",
    "L4",
    "L5",
    "L6",
    "L7",
    "LG",
    "R1",
    "R2",
    "R3",
    "R4",
    "R5",
    "R6",
    "R7",
    "RG",
)
EXIT_COMMANDS = {"", "quit", "exit", "q"}


# ============================ env / policy helpers ============================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
        help="Path to a pretrained_model directory produced by lerobot-train.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=150,
        help="Number of policy steps per episode. Match roughly to generated episode length.",
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=1.0,
        help="Playback speed multiplier. 0 means run as fast as possible (no sleep).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="torch device override, e.g. 'cuda' or 'cpu'. Auto-detected if omitted.",
    )
    parser.add_argument(
        "--show-viewer-ui",
        action="store_true",
        help="Show the MuJoCo side UI (hidden by default so controls stay read-only).",
    )
    return parser.parse_args()


def dataset_features() -> dict:
    """Feature schema mirroring the training dataset (06_generate_scripted_data.py).

    Three-view setup: top RGB camera plus a wrist camera on each arm.
    """
    image_feature = {
        "shape": (480, 640, 3),
        "dtype": "image",
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
    """Convert a raw UnoarmEnv observation into a model-ready inference frame.

    NOTE: ``webapp/runner.py`` imports this function dynamically; keep the
    signature stable (no task/language argument — ACT is vision+state only).
    """
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


def format_state(normalized_state: np.ndarray) -> str:
    left = " ".join(f"{STATE_LABELS[i]}={normalized_state[i]:+.2f}" for i in range(8))
    right = " ".join(f"{STATE_LABELS[i]}={normalized_state[i]:+.2f}" for i in range(8, 16))
    return f"{left} | {right}"


def run_episode(
    env: UnoarmEnv,
    policy: ACTPolicy,
    preprocess,
    postprocess,
    max_steps: int,
    speed: float,
    device: torch.device,
    viewer,
) -> None:
    """Reset to the rest pose and let the policy drive the robot for max_steps."""
    obs, _info = env.reset(options={"state": env._normalize(RAW_ZERO)})
    step_delay = 0.0 if speed <= 0.0 else 1.0 / (FPS * speed)

    for step in range(max_steps):
        if viewer is not None and not viewer.is_running():
            return

        frame = obs_to_frame(obs, device)
        processed = preprocess(frame)
        with torch.inference_mode():
            action = policy.select_action(processed)
        action = postprocess(action)
        if isinstance(action, torch.Tensor):
            action_np = action.squeeze(0).detach().cpu().numpy()
        else:
            action_np = np.asarray(action).squeeze(0)
        action_np = np.asarray(action_np, dtype=np.float32).reshape(-1)
        if action_np.shape[0] > 16:
            action_np = action_np[:16]
        action_np = np.clip(action_np, -1.0, 1.0)

        obs, _reward, _terminated, _truncated, _info = env.step(action_np)

        if viewer is not None:
            viewer.sync()
            if step_delay > 0.0:
                time.sleep(step_delay)

        state = np.asarray(obs["agent_pos"], dtype=np.float32).reshape(16)
        print(
            f"\r\033[2Kstep {step + 1:3d}/{max_steps}  {format_state(state)}",
            end="",
            flush=True,
        )
    print()


# ============================ main loop ============================


def prompt_user() -> str | None:
    """Read a line from stdin. Returns None when the user wants to quit."""
    try:
        raw = input("\n按回车执行一段 episode（输入 quit 退出）> ").strip()
    except (EOFError, KeyboardInterrupt):
        return None
    if raw.lower() in EXIT_COMMANDS:
        return None
    return raw


def main() -> None:
    args = parse_args()

    if not args.checkpoint.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {args.checkpoint}\n"
            "Pass --checkpoint <path> pointing at a pretrained_model directory."
        )
    if args.max_steps < 1:
        raise ValueError(f"--max-steps must be >= 1, got {args.max_steps}")

    device = (
        torch.device(args.device)
        if args.device
        else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"loading policy from {args.checkpoint} (device={device}) ...")
    policy = ACTPolicy.from_pretrained(args.checkpoint)
    policy.eval().to(device)
    preprocess, postprocess = make_pre_post_processors(policy.config, pretrained_path=str(args.checkpoint))

    env = UnoarmEnv(obs_type="pixels_agent_pos", render_mode="rgb_array", max_episode_steps=args.max_steps)
    try:
        with mujoco.viewer.launch_passive(
            env.model,
            env.data,
            show_left_ui=args.show_viewer_ui,
            show_right_ui=args.show_viewer_ui,
        ) as viewer:
            configure_viewer_camera(viewer)
            configure_viewer_theme(viewer, env.model)

            print("\n=== Unoarm ACT 交互式推理 ===")
            print("每次按回车，从静止姿态开始执行一段策略 rollout。")
            print("输入 'quit' / 'exit' 或关闭 viewer 退出。\n")

            while viewer.is_running():
                user_input = prompt_user()
                if user_input is None:
                    break
                run_episode(
                    env,
                    policy,
                    preprocess,
                    postprocess,
                    args.max_steps,
                    args.speed,
                    device,
                    viewer,
                )
    finally:
        env.close()
        print("\nbye.")


if __name__ == "__main__":
    main()
