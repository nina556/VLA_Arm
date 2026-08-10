from __future__ import annotations

import select
import shutil
import sys
import termios
import time
import tty
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gym_unoarm.constants import CONTROL_JOINTS, FPS, JOINTS, MIMIC_JOINTS  # noqa: E402
from gym_unoarm.env import UnoarmEnv, configure_viewer_camera, configure_viewer_theme  # noqa: E402

NUM_EPISODES = 30
STEPS_PER_EP = 200
TASK_DESC = "cross both arms in front of the body"
OUTPUT_ROOT = ROOT / "data" / "unoarm_demo"
DELTA = 0.05
SLIDER_DEADBAND = 0.01

KEY_MAP_LEFT = {
    "1": (0, +1),
    "q": (0, -1),
    "2": (1, +1),
    "w": (1, -1),
    "3": (2, +1),
    "e": (2, -1),
    "4": (3, +1),
    "r": (3, -1),
    "5": (4, +1),
    "t": (4, -1),
    "6": (5, +1),
    "y": (5, -1),
    "7": (6, +1),
    "u": (6, -1),
}

KEY_MAP_RIGHT = {
    "1": (8, +1),
    "q": (8, -1),
    "2": (9, +1),
    "w": (9, -1),
    "3": (10, +1),
    "e": (10, -1),
    "4": (11, +1),
    "r": (11, -1),
    "5": (12, +1),
    "t": (12, -1),
    "6": (13, +1),
    "y": (13, -1),
    "7": (14, +1),
    "u": (14, -1),
}


def get_key_nonblocking() -> str | None:
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        if select.select([sys.stdin], [], [], 0)[0]:
            return sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)
    return None


def build_ctrl_to_qpos_map(model: mujoco.MjModel) -> list[tuple[int, int]]:
    """Map each viewer ctrl index to the qpos address of its target joint."""
    mapping: list[tuple[int, int]] = []
    for ctrl_idx in range(model.nu):
        joint_id = int(model.actuator_trnid[ctrl_idx, 0])
        qpos_addr = int(model.jnt_qposadr[joint_id])
        mapping.append((ctrl_idx, qpos_addr))
    return mapping


def build_mimic_addrs(model: mujoco.MjModel) -> dict[str, dict]:
    """Build a map from mimic joint name to qpos addresses and transform."""
    mimic_addrs: dict[str, dict] = {}
    for mimic_name, (source_name, multiplier, offset) in MIMIC_JOINTS.items():
        mimic_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, mimic_name)
        source_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, source_name)
        if mimic_id < 0 or source_id < 0:
            continue
        mimic_addrs[mimic_name] = {
            "mimic_qposadr": int(model.jnt_qposadr[mimic_id]),
            "source_qposadr": int(model.jnt_qposadr[source_id]),
            "multiplier": float(multiplier),
            "offset": float(offset),
        }
    return mimic_addrs


def apply_ctrl_to_qpos(data: mujoco.MjData, ctrl_to_qpos: list[tuple[int, int]]) -> None:
    """Copy viewer ctrl values (target positions) directly into qpos."""
    for ctrl_idx, qpos_addr in ctrl_to_qpos:
        data.qpos[qpos_addr] = data.ctrl[ctrl_idx]


def apply_mimics(data: mujoco.MjData, mimic_addrs: dict[str, dict]) -> None:
    """Copy source gripper qpos to mimic finger joints."""
    for info in mimic_addrs.values():
        source_value = data.qpos[info["source_qposadr"]]
        data.qpos[info["mimic_qposadr"]] = info["multiplier"] * source_value + info["offset"]


def sync_ctrl_from_qpos(data: mujoco.MjData, ctrl_to_qpos: list[tuple[int, int]]) -> None:
    """Make the viewer control sliders match the current qpos."""
    for ctrl_idx, qpos_addr in ctrl_to_qpos:
        data.ctrl[ctrl_idx] = data.qpos[qpos_addr]


def apply_normalized_action(env: UnoarmEnv, action: np.ndarray) -> None:
    """Apply a normalized 16D action to the MuJoCo data held by env."""
    target_qpos = env._denormalize(action)
    for name, value in zip(CONTROL_JOINTS, target_qpos, strict=True):
        env.data.qpos[env._qpos_addr[name]] = float(value)
    env._apply_mimics()
    env.data.qvel[:] = 0.0
    mujoco.mj_forward(env.model, env.data)


def dataset_features() -> dict:
    image_feature = {
        "dtype": "image",
        "shape": (480, 640, 3),
        "names": ["height", "width", "channels"],
        "info": {"is_depth_map": False},
    }
    return {
        "action": {"dtype": "float32", "shape": (16,), "names": list(JOINTS)},
        "observation.state": {"dtype": "float32", "shape": (16,), "names": list(JOINTS)},
        "observation.images.top": image_feature,
        "observation.images.left_wrist": image_feature,
        "observation.images.right_wrist": image_feature,
    }


def add_frame(dataset: LeRobotDataset, obs: dict, action: np.ndarray) -> None:
    dataset.add_frame(
        {
            "observation.state": obs["agent_pos"].astype(np.float32),
            "observation.images.top": obs["pixels"]["top"],
            "observation.images.left_wrist": obs["pixels"]["left_wrist"],
            "observation.images.right_wrist": obs["pixels"]["right_wrist"],
            "action": action.astype(np.float32),
            "task": TASK_DESC,
        }
    )


def main() -> None:
    env = UnoarmEnv(obs_type="pixels_agent_pos", render_mode="rgb_array")

    if OUTPUT_ROOT.exists():
        print(f"Removing existing dataset at {OUTPUT_ROOT}")
        shutil.rmtree(OUTPUT_ROOT)

    dataset = LeRobotDataset.create(
        repo_id="doki/unoarm_demo",
        fps=FPS,
        features=dataset_features(),
        root=OUTPUT_ROOT,
        robot_type="unoarm",
        use_videos=False,
    )

    ctrl_to_qpos = build_ctrl_to_qpos_map(env.model)
    mimic_addrs = build_mimic_addrs(env.model)

    obs, _ = env.reset()
    # Make viewer sliders match the reset pose.
    sync_ctrl_from_qpos(env.data, ctrl_to_qpos)
    current_target = env._normalize(env._current_control_qpos()).copy()
    gripper_state = {
        "left": float(current_target[7]),
        "right": float(current_target[15]),
    }
    current_arm = "left"

    print(f"Recording {NUM_EPISODES} episodes, {STEPS_PER_EP} steps each.")
    print("space=switch arm, 1..7 increase, q..u decrease, g=gripper, enter=save, r=reset, esc=exit")
    print("You can also drag the MuJoCo Control sliders to move joints directly.")

    try:
        with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
            configure_viewer_camera(viewer)
            configure_viewer_theme(viewer, env.model)
            for ep in range(NUM_EPISODES):
                print(f"=== Episode {ep + 1}/{NUM_EPISODES} ===")
                saved = False
                key_pressed_this_frame = False
                for step in range(STEPS_PER_EP):
                    if not viewer.is_running():
                        dataset.finalize()
                        return

                    # 1. Apply any viewer slider movement to qpos so we can read it.
                    apply_ctrl_to_qpos(env.data, ctrl_to_qpos)
                    apply_mimics(env.data, mimic_addrs)

                    # 2. Read the current pose in normalized space (from sliders or last action).
                    slider_target = env._normalize(env._current_control_qpos())

                    # 3. Handle keyboard input.
                    key = get_key_nonblocking()
                    key_pressed_this_frame = key is not None

                    if key == "\x1b":
                        dataset.finalize()
                        return
                    if key == " ":
                        current_arm = "right" if current_arm == "left" else "left"
                        print(f"selected arm: {current_arm}")
                    elif key == "g":
                        if current_arm == "left":
                            gripper_state["left"] = -1.0 if gripper_state["left"] > 0.0 else 1.0
                            current_target[7] = gripper_state["left"]
                        else:
                            gripper_state["right"] = -1.0 if gripper_state["right"] > 0.0 else 1.0
                            current_target[15] = gripper_state["right"]
                    elif key in ("\r", "\n"):
                        saved = True
                        break
                    elif key == "r":
                        dataset.clear_episode_buffer()
                        obs, _ = env.reset()
                        sync_ctrl_from_qpos(env.data, ctrl_to_qpos)
                        current_target = env._normalize(env._current_control_qpos()).copy()
                        gripper_state["left"] = float(current_target[7])
                        gripper_state["right"] = float(current_target[15])
                        print("episode buffer cleared")
                        break
                    else:
                        keymap = KEY_MAP_LEFT if current_arm == "left" else KEY_MAP_RIGHT
                        if key in keymap:
                            joint_idx, sign = keymap[key]
                            current_target[joint_idx] = float(
                                np.clip(current_target[joint_idx] + sign * DELTA, -1.0, 1.0)
                            )

                    # 4. If no keyboard input this frame and the sliders moved, follow the sliders.
                    if not key_pressed_this_frame and np.linalg.norm(slider_target - current_target) > SLIDER_DEADBAND:
                        current_target = slider_target.copy()
                        gripper_state["left"] = float(current_target[7])
                        gripper_state["right"] = float(current_target[15])

                    # 5. Apply the chosen normalized action to the simulation.
                    apply_normalized_action(env, current_target)
                    sync_ctrl_from_qpos(env.data, ctrl_to_qpos)

                    # 6. Record the frame.
                    obs = env._get_obs()
                    add_frame(dataset, obs, current_target.copy())
                    viewer.sync()
                    time.sleep(1.0 / FPS)

                    if step + 1 >= STEPS_PER_EP:
                        saved = True

                if saved:
                    dataset.save_episode()
                    print(f"saved episode {ep + 1}")

                if ep + 1 < NUM_EPISODES:
                    obs, _ = env.reset()
                    sync_ctrl_from_qpos(env.data, ctrl_to_qpos)
                    current_target = env._normalize(env._current_control_qpos()).copy()
                    gripper_state["left"] = float(current_target[7])
                    gripper_state["right"] = float(current_target[15])
    finally:
        dataset.finalize()
        env.close()
        print(f"dataset finalized at {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
