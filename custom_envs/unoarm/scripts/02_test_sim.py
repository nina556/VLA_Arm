from __future__ import annotations

import sys
import time
from pathlib import Path

import mujoco

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gym_unoarm.constants import FPS, MIMIC_JOINTS, XML_PATH  # noqa: E402
from gym_unoarm.env import configure_viewer_camera, configure_viewer_theme  # noqa: E402


def build_ctrl_to_qpos_map(model: mujoco.MjModel) -> list[tuple[int, int]]:
    """Map each viewer ctrl index to the qpos address of its target joint."""
    mapping: list[tuple[int, int]] = []
    for ctrl_idx in range(model.nu):
        # actuator_trnid[:, 0] is the transmission object id (here, the joint id).
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


def main() -> None:
    import mujoco.viewer

    model = mujoco.MjModel.from_xml_path(str(XML_PATH))
    data = mujoco.MjData(model)

    ctrl_to_qpos = build_ctrl_to_qpos_map(model)
    mimic_addrs = build_mimic_addrs(model)

    # Initialize mimic joints and make ctrl match the current pose so the robot
    # stays still until the user moves a viewer slider.
    apply_mimics(data, mimic_addrs)
    for ctrl_idx, qpos_addr in ctrl_to_qpos:
        data.ctrl[ctrl_idx] = data.qpos[qpos_addr]
    apply_ctrl_to_qpos(data, ctrl_to_qpos)
    apply_mimics(data, mimic_addrs)
    mujoco.mj_forward(model, data)

    print(f"loaded {XML_PATH}")
    print(f"nq={model.nq}, nv={model.nv}, nu={model.nu}, ncam={model.ncam}")
    for i in range(model.njnt):
        print(f"joint {i}: {mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)}")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        configure_viewer_camera(viewer)
        configure_viewer_theme(viewer, model)
        while viewer.is_running():
            # Direct kinematic control: viewer sliders write into data.ctrl;
            # copy those target positions into data.qpos without physics stepping.
            apply_ctrl_to_qpos(data, ctrl_to_qpos)
            apply_mimics(data, mimic_addrs)
            mujoco.mj_forward(model, data)
            viewer.sync()
            time.sleep(1.0 / FPS)


if __name__ == "__main__":
    main()
