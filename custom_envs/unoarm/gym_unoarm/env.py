from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

from .constants import (
    CAMERAS,
    CONTROL_JOINTS,
    DEFAULT_EXECUTION_POINT_POS,
    DT,
    FPS,
    GRASP_CLOSE_THRESHOLD,
    GRASP_RELEASE_THRESHOLD,
    HELMET_HANDLE_LOCAL,
    IMG_H,
    IMG_W,
    LEFT_GRIPPER_INDEX,
    MIMIC_JOINTS,
    PEG_DEFAULT_XY,
    PEG_FALL_MAX_STEPS,
    PEG_GRASP_BETWEEN_HI,
    PEG_GRASP_BETWEEN_LO,
    PEG_GRASP_DIST_MAX,
    PEG_GRASP_SURFACE_TOL,
    PEG_GRAVITY,
    PEG_HALF_HEIGHT,
    PEG_REST_Z_OFFSET,
    PROP_PARK_POS,
    REACH_SUCCESS_THRESHOLD,
    REACH_TCP_SITE,
    RIGHT_GRIPPER_INDEX,
    SCENE_FREE_SPACE,
    SCENE_REACH_SWORD,
    SCENE_TABLE_PLACE,
    START_POSE,
    SWORD_HANDLE_LOCAL,
    TABLE_CIRCLE_RADIUS,
    XML_PATH,
    clip_peg_xy_to_table,
    peg_body_pos_from_xy,
    table_place_top_z,
)
from .reach_scene import ensure_reach_sword_xml, evaluate_reach
from .table_place_scene import ensure_table_place_xml


def _quat_wxyz_to_mat(quat_wxyz: np.ndarray) -> np.ndarray:
    mat = np.zeros(9, dtype=np.float64)
    mujoco.mju_quat2Mat(mat, np.asarray(quat_wxyz, dtype=np.float64).reshape(4))
    return mat.reshape(3, 3)


def _mat_to_quat_wxyz(mat33: np.ndarray) -> np.ndarray:
    quat = np.zeros(4, dtype=np.float64)
    mujoco.mju_mat2Quat(quat, np.asarray(mat33, dtype=np.float64).reshape(9))
    return quat


class UnoarmEnv(gym.Env):
    """Free-space Unoarm simulation for behavior cloning data collection.

    The environment exposes a 16D normalized action/state vector:
    7 left arm joints, 1 left gripper, 7 right arm joints, 1 right gripper.
    Actions are applied kinematically to the MuJoCo qpos array. This keeps the
    first version stable for data collection while preserving the real robot
    joint limits from the URDF/MJCF.
    """

    metadata = {"render_modes": ["rgb_array", "human"], "render_fps": FPS}

    def __init__(
        self,
        task: str = "UnoarmFreeSpace-v0",
        obs_type: str = "pixels_agent_pos",
        render_mode: str = "rgb_array",
        observation_width: int = IMG_W,
        observation_height: int = IMG_H,
        max_episode_steps: int = 300,
        xml_path: str | Path | None = None,
        scene: str = SCENE_FREE_SPACE,
        reach_success_threshold: float = REACH_SUCCESS_THRESHOLD,
        terminate_on_success: bool = False,
        sword_body_pos: tuple[float, float, float] | None = None,
        sword_body_quat: tuple[float, float, float, float] | None = None,
        shield_body_pos: tuple[float, float, float] | None = None,
        shield_body_quat: tuple[float, float, float, float] | None = None,
        remove_sword: bool = False,
        remove_shield: bool = False,
        enable_execution_point: bool = False,
        execution_point_pos: tuple[float, float, float] | None = None,
        peg_xy: tuple[float, float] | None = None,
    ) -> None:
        super().__init__()
        self.task = task
        self.obs_type = obs_type
        self.render_mode = render_mode
        self.obs_width = int(observation_width)
        self.obs_height = int(observation_height)
        self.max_episode_steps = int(max_episode_steps)
        self.scene = str(scene)
        self.reach_success_threshold = float(reach_success_threshold)
        self.terminate_on_success = bool(terminate_on_success)
        self.remove_sword = bool(remove_sword)
        self.remove_shield = bool(remove_shield)
        self.enable_execution_point = bool(enable_execution_point)
        if execution_point_pos is None:
            self.execution_point_pos = tuple(float(x) for x in DEFAULT_EXECUTION_POINT_POS)
        else:
            self.execution_point_pos = tuple(float(x) for x in execution_point_pos)
        self._execution_point_passed = False
        self.sword_body_pos = tuple(float(x) for x in sword_body_pos) if sword_body_pos is not None else None
        self.sword_body_quat = (
            tuple(float(x) for x in sword_body_quat) if sword_body_quat is not None else None
        )
        self.shield_body_pos = (
            tuple(float(x) for x in shield_body_pos) if shield_body_pos is not None else None
        )
        self.shield_body_quat = (
            tuple(float(x) for x in shield_body_quat) if shield_body_quat is not None else None
        )
        if peg_xy is None:
            self.peg_xy = tuple(float(x) for x in PEG_DEFAULT_XY)
        else:
            self.peg_xy = (float(peg_xy[0]), float(peg_xy[1]))

        if xml_path is not None:
            self.xml_path = Path(xml_path)
        elif self.scene == SCENE_REACH_SWORD:
            self.xml_path = ensure_reach_sword_xml(
                body_pos=self.sword_body_pos,
                body_quat=self.sword_body_quat,
                shield_body_pos=self.shield_body_pos,
                shield_body_quat=self.shield_body_quat,
            )
        elif self.scene == SCENE_TABLE_PLACE:
            self.xml_path = ensure_table_place_xml(peg_xy=self.peg_xy)
        elif self.scene == SCENE_FREE_SPACE:
            self.xml_path = XML_PATH
        else:
            raise ValueError(
                f"Unknown scene {self.scene!r}. "
                f"Expected {SCENE_FREE_SPACE!r}, {SCENE_REACH_SWORD!r}, or {SCENE_TABLE_PLACE!r}."
            )

        if not self.xml_path.exists():
            raise FileNotFoundError(
                f"Missing MJCF file: {self.xml_path}. Run "
                "`uv run python custom_envs/unoarm/scripts/01_convert_urdf_to_mjcf.py` first."
            )

        self.model = mujoco.MjModel.from_xml_path(str(self.xml_path))
        self.model.opt.timestep = DT
        self.data = mujoco.MjData(self.model)
        self._elapsed_steps = 0
        self._renderer: mujoco.Renderer | None = None
        self._viewer: Any | None = None
        self._render_failed = False
        self._last_action = np.asarray(START_POSE, dtype=np.float32)

        self._qpos_addr = self._build_qpos_addr()
        self._joint_limits = self._build_joint_limits()
        self._tcp_site_id = -1  # right TCP (default / validate)
        self._left_tcp_site_id = -1
        self._handle_site_id = -1  # sword handle
        self._shield_handle_site_id = -1
        self._sword_body_id = -1
        self._shield_body_id = -1
        self._sword_home_pos = None
        self._sword_home_quat = None
        self._shield_home_pos = None
        self._shield_home_quat = None
        if self.scene == SCENE_REACH_SWORD:
            self._tcp_site_id = int(mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, REACH_TCP_SITE))
            self._left_tcp_site_id = int(mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "left_tcp"))
            self._handle_site_id = int(
                mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "sword_handle")
            )
            self._shield_handle_site_id = int(
                mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "shield_handle")
            )
            self._sword_body_id = int(mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "sword"))
            self._shield_body_id = int(mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "shield"))
            missing = [
                name
                for name, sid in (
                    (REACH_TCP_SITE, self._tcp_site_id),
                    ("left_tcp", self._left_tcp_site_id),
                    ("sword_handle", self._handle_site_id),
                    ("shield_handle", self._shield_handle_site_id),
                    ("sword", self._sword_body_id),
                    ("shield", self._shield_body_id),
                )
                if sid < 0
            ]
            if missing:
                raise ValueError(f"Reach scene XML missing sites/bodies {missing}: {self.xml_path}")
            if self.sword_body_pos is not None:
                self.model.body_pos[self._sword_body_id] = np.asarray(self.sword_body_pos, dtype=np.float64)
            if self.sword_body_quat is not None:
                self.model.body_quat[self._sword_body_id] = np.asarray(self.sword_body_quat, dtype=np.float64)
            if self.shield_body_pos is not None:
                self.model.body_pos[self._shield_body_id] = np.asarray(self.shield_body_pos, dtype=np.float64)
            if self.shield_body_quat is not None:
                self.model.body_quat[self._shield_body_id] = np.asarray(
                    self.shield_body_quat, dtype=np.float64
                )
            if (
                self.sword_body_pos is not None
                or self.sword_body_quat is not None
                or self.shield_body_pos is not None
                or self.shield_body_quat is not None
            ):
                mujoco.mj_forward(self.model, self.data)
            self._sword_home_pos = np.asarray(
                self.model.body_pos[self._sword_body_id], dtype=np.float64
            ).copy()
            self._sword_home_quat = np.asarray(
                self.model.body_quat[self._sword_body_id], dtype=np.float64
            ).copy()
            self._shield_home_pos = np.asarray(
                self.model.body_pos[self._shield_body_id], dtype=np.float64
            ).copy()
            self._shield_home_quat = np.asarray(
                self.model.body_quat[self._shield_body_id], dtype=np.float64
            ).copy()
            # Attach flags must exist before parking / restoring props.
            self._sword_attached = False
            self._shield_attached = False
            self._apply_prop_presence(forward=True)

        self._peg_body_id = -1
        self._peg_grasp_site_id = -1
        self._target_circle_site_id = -1
        self._peg_home_pos = None
        self._peg_attached = False
        self._peg_attach_R_rel = np.eye(3, dtype=np.float64)
        # Offset locked at the moment of grasping so the peg stays where the
        # fingers clamped it (peg_pos = grasp_center + _peg_grasp_offset),
        # instead of snapping to the hand center.
        self._peg_grasp_offset = np.zeros(3, dtype=np.float64)
        # Normalized right-gripper command at attach time; while attached we
        # refuse to close further so the jaw stops at a successful grasp.
        self._peg_attach_gripper_norm: float | None = None
        self._peg_grasp_local = np.array([0.0, 0.0, float(PEG_HALF_HEIGHT)], dtype=np.float64)
        # Right-gripper finger body/geom ids (table_place grasp detection).
        # Body origins sit on the finger hinges and do not translate when the
        # jaw opens/closes — mesh geoms do, so grasp checks use geom distance.
        # Each jaw side uses both primary links (1 + 2); contact = min distance.
        self._rg_left_finger_id = -1
        self._rg_right_finger_id = -1
        self._rg_left_finger_geom_ids: tuple[int, ...] = ()
        self._rg_right_finger_geom_ids: tuple[int, ...] = ()
        self._rg_left_finger_geom_id = -1
        self._rg_right_finger_geom_id = -1
        self._peg_geom_id = -1
        # Peg free-fall state (env-layer vertical integration on release).
        self._peg_falling = False
        self._peg_vz = 0.0  # vertical velocity (m/s); negative = falling
        self._peg_fall_steps = 0
        self._peg_fall_xy = (0.0, 0.0)  # frozen XY during fall
        if self.scene == SCENE_TABLE_PLACE:
            self._tcp_site_id = int(mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, REACH_TCP_SITE))
            self._left_tcp_site_id = int(mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "left_tcp"))
            self._peg_body_id = int(mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "peg"))
            self._peg_grasp_site_id = int(
                mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "peg_grasp")
            )
            self._target_circle_site_id = int(
                mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "target_circle")
            )
            self._peg_geom_id = int(mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "peg_geom"))
            # Right-gripper main finger meshes define the real grasp center
            # (geom midpoint) and the surface-contact check.
            left_bodies = (
                "Right_Gripper_Left_1_Link",
                "Right_Gripper_Left_2_Link",
            )
            right_bodies = (
                "Right_Gripper_Right_1_Link",
                "Right_Gripper_Right_2_Link",
            )
            self._rg_left_finger_id = int(
                mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, left_bodies[0])
            )
            self._rg_right_finger_id = int(
                mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, right_bodies[0])
            )
            left_geoms = []
            for name in left_bodies:
                bid = int(mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name))
                left_geoms.append(self._first_geom_on_body(bid))
            right_geoms = []
            for name in right_bodies:
                bid = int(mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name))
                right_geoms.append(self._first_geom_on_body(bid))
            self._rg_left_finger_geom_ids = tuple(left_geoms)
            self._rg_right_finger_geom_ids = tuple(right_geoms)
            self._rg_left_finger_geom_id = left_geoms[0] if left_geoms else -1
            self._rg_right_finger_geom_id = right_geoms[0] if right_geoms else -1
            missing = [
                name
                for name, sid in (
                    (REACH_TCP_SITE, self._tcp_site_id),
                    ("left_tcp", self._left_tcp_site_id),
                    ("peg", self._peg_body_id),
                    ("peg_geom", self._peg_geom_id),
                    ("peg_grasp", self._peg_grasp_site_id),
                    ("target_circle", self._target_circle_site_id),
                    ("Right_Gripper_Left_1_Link", self._rg_left_finger_id),
                    ("Right_Gripper_Right_1_Link", self._rg_right_finger_id),
                    ("Right_Gripper_Left_1_geom", self._rg_left_finger_geom_id),
                    ("Right_Gripper_Right_1_geom", self._rg_right_finger_geom_id),
                    ("Right_Gripper_Left_2_geom", left_geoms[1] if len(left_geoms) > 1 else -1),
                    ("Right_Gripper_Right_2_geom", right_geoms[1] if len(right_geoms) > 1 else -1),
                )
                if sid < 0
            ]
            if missing:
                raise ValueError(f"Table-place scene XML missing sites/bodies {missing}: {self.xml_path}")
            self.place_peg_xy(self.peg_xy)
            self._peg_home_pos = np.asarray(self.model.body_pos[self._peg_body_id], dtype=np.float64).copy()

        self._sword_attached = False
        self._shield_attached = False
        self._sword_attach_R_rel = np.eye(3, dtype=np.float64)
        self._shield_attach_R_rel = np.eye(3, dtype=np.float64)
        self._grasp_close_threshold = float(GRASP_CLOSE_THRESHOLD)
        self._grasp_release_threshold = float(GRASP_RELEASE_THRESHOLD)
        self._handle_local = np.asarray(SWORD_HANDLE_LOCAL, dtype=np.float64)
        self._shield_handle_local = np.asarray(HELMET_HANDLE_LOCAL, dtype=np.float64)

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(16,), dtype=np.float32)
        image_spaces = {
            camera: spaces.Box(0, 255, (self.obs_height, self.obs_width, 3), dtype=np.uint8)
            for camera in CAMERAS
        }
        obs: dict[str, Any] = {"pixels": spaces.Dict(image_spaces)}
        if obs_type == "pixels":
            self.observation_space = spaces.Dict(obs)
        elif obs_type == "pixels_agent_pos":
            obs["agent_pos"] = spaces.Box(low=-1.0, high=1.0, shape=(16,), dtype=np.float32)
            self.observation_space = spaces.Dict(obs)
        else:
            raise ValueError(f"Unknown obs_type: {obs_type}")

    def _build_qpos_addr(self) -> dict[str, int]:
        names = set(CONTROL_JOINTS) | set(MIMIC_JOINTS)
        qpos_addr: dict[str, int] = {}
        missing: list[str] = []
        for name in names:
            joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if joint_id < 0:
                missing.append(name)
            else:
                qpos_addr[name] = int(self.model.jnt_qposadr[joint_id])
        if missing:
            raise ValueError(f"MJCF is missing expected joints: {missing}")
        return qpos_addr

    def _build_joint_limits(self) -> np.ndarray:
        limits = np.zeros((len(CONTROL_JOINTS), 2), dtype=np.float32)
        for i, name in enumerate(CONTROL_JOINTS):
            joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            low, high = self.model.jnt_range[joint_id]
            if np.isclose(low, high):
                low, high = -1.0, 1.0
            limits[i] = (low, high)
        return limits

    def _denormalize(self, action: np.ndarray) -> np.ndarray:
        action = np.clip(action.astype(np.float32, copy=False), -1.0, 1.0)
        low = self._joint_limits[:, 0]
        high = self._joint_limits[:, 1]
        return low + (action + 1.0) * 0.5 * (high - low)

    def _normalize(self, qpos: np.ndarray) -> np.ndarray:
        low = self._joint_limits[:, 0]
        high = self._joint_limits[:, 1]
        normalized = 2.0 * (qpos - low) / np.maximum(high - low, 1e-6) - 1.0
        return np.clip(normalized, -1.0, 1.0).astype(np.float32)

    def _current_control_qpos(self) -> np.ndarray:
        return np.asarray(
            [self.data.qpos[self._qpos_addr[name]] for name in CONTROL_JOINTS], dtype=np.float32
        )

    def _apply_mimics(self) -> None:
        for mimic_name, (source_name, multiplier, offset) in MIMIC_JOINTS.items():
            source_value = self.data.qpos[self._qpos_addr[source_name]]
            self.data.qpos[self._qpos_addr[mimic_name]] = multiplier * source_value + offset

    def apply_raw_qpos(self, raw_qpos: np.ndarray) -> None:
        """Set control joints from raw MuJoCo qpos values (kinematic, with mimics)."""
        raw_qpos = np.asarray(raw_qpos, dtype=np.float32).reshape(-1)
        if raw_qpos.shape != (16,):
            raise ValueError(f"raw_qpos must have shape (16,), got {raw_qpos.shape}")
        low = self._joint_limits[:, 0]
        high = self._joint_limits[:, 1]
        clipped = np.clip(raw_qpos, low, high).astype(np.float32)
        for name, value in zip(CONTROL_JOINTS, clipped, strict=True):
            self.data.qpos[self._qpos_addr[name]] = float(value)
        self._apply_mimics()
        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)
        self._last_action = self._normalize(clipped)
        self._update_grasps()
        self._clamp_right_gripper_at_attach()

    def _apply_action(self, action: np.ndarray) -> None:
        target_qpos = self._denormalize(action)
        for name, value in zip(CONTROL_JOINTS, target_qpos, strict=True):
            self.data.qpos[self._qpos_addr[name]] = float(value)
        self._apply_mimics()
        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)
        self._update_grasps()
        self._clamp_right_gripper_at_attach()

    def _set_right_gripper_normalized(self, g_norm: float) -> None:
        """Write the right gripper control joint from a normalized [-1, 1] value."""
        raw = self._current_control_qpos().astype(np.float32).copy()
        n = self._normalize(raw)
        n[RIGHT_GRIPPER_INDEX] = float(np.clip(g_norm, -1.0, 1.0))
        raw = self._denormalize(n).astype(np.float32)
        name = CONTROL_JOINTS[RIGHT_GRIPPER_INDEX]
        self.data.qpos[self._qpos_addr[name]] = float(raw[RIGHT_GRIPPER_INDEX])
        self._apply_mimics()
        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)

    def _clamp_right_gripper_at_attach(self) -> None:
        """Stop closing once a table_place grasp has succeeded.

        While the peg is attached, ignore commands that would close the jaw
        past the aperture recorded at attach time (opening to release is OK).
        """
        if self.scene != SCENE_TABLE_PLACE or not self._peg_attached:
            return
        if self._peg_attach_gripper_norm is None:
            return
        cur = self._gripper_normalized(RIGHT_GRIPPER_INDEX)
        if cur < float(self._peg_attach_gripper_norm) - 1e-6:
            self._set_right_gripper_normalized(float(self._peg_attach_gripper_norm))
            self._last_action = self._normalize(self._current_control_qpos())
            # Keep peg stuck after the clamp (finger midpoint may have moved).
            if self._peg_attached:
                self._stick_peg_to_grasp_center()

    def _gripper_normalized(self, index: int) -> float:
        return float(self._normalize(self._current_control_qpos())[index])

    def _gripper_closed(self, index: int) -> bool:
        return self._gripper_normalized(index) <= self._grasp_close_threshold

    def _gripper_open_enough(self, index: int) -> bool:
        return self._gripper_normalized(index) >= self._grasp_release_threshold

    def _right_gripper_closed(self) -> bool:
        return self._gripper_closed(RIGHT_GRIPPER_INDEX)

    def _left_gripper_closed(self) -> bool:
        return self._gripper_closed(LEFT_GRIPPER_INDEX)

    def _set_body_pose_world(self, body_id: int, body_pos: np.ndarray, body_quat_wxyz: np.ndarray) -> None:
        if body_id < 0:
            return
        self.model.body_pos[body_id] = np.asarray(body_pos, dtype=np.float64).reshape(3)
        self.model.body_quat[body_id] = np.asarray(body_quat_wxyz, dtype=np.float64).reshape(4)
        mujoco.mj_forward(self.model, self.data)

    def _attach_body_to_tcp(
        self,
        *,
        body_id: int,
        tcp_site_id: int,
        handle_local: np.ndarray,
        attached_flag: str,
        rel_attr: str,
    ) -> None:
        """Snap/keep relative pose so handle follows TCP while attached."""
        tcp_pos = np.asarray(self.data.site_xpos[tcp_site_id], dtype=np.float64)
        tcp_mat = np.asarray(self.data.site_xmat[tcp_site_id], dtype=np.float64).reshape(3, 3)
        r_body = _quat_wxyz_to_mat(self.model.body_quat[body_id])
        if not bool(getattr(self, attached_flag)):
            setattr(self, rel_attr, tcp_mat.T @ r_body)
            setattr(self, attached_flag, True)
        r_rel = getattr(self, rel_attr)
        r_body = tcp_mat @ r_rel
        body_quat = _mat_to_quat_wxyz(r_body)
        body_pos = tcp_pos - r_body @ np.asarray(handle_local, dtype=np.float64)
        self._set_body_pose_world(body_id, body_pos, body_quat)

    def _update_one_grasp(
        self,
        *,
        body_id: int,
        tcp_site_id: int,
        handle_site_id: int,
        handle_local: np.ndarray,
        gripper_index: int,
        attached_flag: str,
        rel_attr: str,
    ) -> None:
        if body_id < 0 or tcp_site_id < 0 or handle_site_id < 0:
            return
        tcp = np.asarray(self.data.site_xpos[tcp_site_id], dtype=np.float64)
        handle = np.asarray(self.data.site_xpos[handle_site_id], dtype=np.float64)
        dist = float(np.linalg.norm(tcp - handle))
        near = dist < float(self.reach_success_threshold)
        attached = bool(getattr(self, attached_flag))

        if attached:
            if self._gripper_open_enough(gripper_index):
                setattr(self, attached_flag, False)
            else:
                self._attach_body_to_tcp(
                    body_id=body_id,
                    tcp_site_id=tcp_site_id,
                    handle_local=handle_local,
                    attached_flag=attached_flag,
                    rel_attr=rel_attr,
                )
            return

        if near and self._gripper_closed(gripper_index):
            self._attach_body_to_tcp(
                body_id=body_id,
                tcp_site_id=tcp_site_id,
                handle_local=handle_local,
                attached_flag=attached_flag,
                rel_attr=rel_attr,
            )

    def _update_grasps(self) -> None:
        """Kinematic stick for right→sword / left→shield, or right→peg."""
        if self.scene == SCENE_TABLE_PLACE:
            was_attached = bool(self._peg_attached)
            self._update_table_place_grasp()
            if was_attached and not self._peg_attached:
                self._start_peg_fall()
            else:
                self._step_peg_fall()
            return
        if self.scene != SCENE_REACH_SWORD:
            return
        if not self.remove_sword:
            self._update_one_grasp(
                body_id=self._sword_body_id,
                tcp_site_id=self._tcp_site_id,
                handle_site_id=self._handle_site_id,
                handle_local=self._handle_local,
                gripper_index=RIGHT_GRIPPER_INDEX,
                attached_flag="_sword_attached",
                rel_attr="_sword_attach_R_rel",
            )
        else:
            self._sword_attached = False
        if not self.remove_shield:
            self._update_one_grasp(
                body_id=self._shield_body_id,
                tcp_site_id=self._left_tcp_site_id,
                handle_site_id=self._shield_handle_site_id,
                handle_local=self._shield_handle_local,
                gripper_index=LEFT_GRIPPER_INDEX,
                attached_flag="_shield_attached",
                rel_attr="_shield_attach_R_rel",
            )
        else:
            self._shield_attached = False

    def place_peg_xy(self, xy: tuple[float, float] | list[float]) -> None:
        """Place the peg upright on the table at world XY (detaches if needed).

        XY is clamped to the table inset so the cylinder never leaves the work
        surface (accounts for peg radius + edge margin).
        """
        if self.scene != SCENE_TABLE_PLACE or self._peg_body_id < 0:
            raise RuntimeError("place_peg_xy requires scene=table_place")
        self._peg_attached = False
        self._peg_falling = False
        self._peg_vz = 0.0
        self._peg_grasp_offset = np.zeros(3, dtype=np.float64)
        self._peg_attach_gripper_norm = None
        self.peg_xy = clip_peg_xy_to_table(xy)
        pos = peg_body_pos_from_xy(self.peg_xy)
        self.model.body_pos[self._peg_body_id] = np.asarray(pos, dtype=np.float64)
        self.model.body_quat[self._peg_body_id] = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        mujoco.mj_forward(self.model, self.data)

    def _seat_peg_on_table(self) -> None:
        """After release, drop peg upright onto the table at current XY."""
        if self._peg_body_id < 0:
            return
        xy = (
            float(self.model.body_pos[self._peg_body_id][0]),
            float(self.model.body_pos[self._peg_body_id][1]),
        )
        self.place_peg_xy(xy)

    def _start_peg_fall(self) -> None:
        """Begin vertical free-fall from the current peg position (on release).

        XY is frozen at the release point so the peg drops straight down. Each
        subsequent env step advances the fall via ``_step_peg_fall`` until the
        peg contacts the table top.
        """
        if self._peg_body_id < 0:
            return
        self._peg_falling = True
        self._peg_vz = 0.0
        self._peg_fall_steps = 0
        pos = self.model.body_pos[self._peg_body_id]
        self._peg_fall_xy = (float(pos[0]), float(pos[1]))

    def _step_peg_fall(self) -> None:
        """Advance the peg vertical free-fall by one env step; stop on contact."""
        if not self._peg_falling or self._peg_body_id < 0:
            return
        dt = float(self.model.opt.timestep)
        self._peg_vz -= PEG_GRAVITY * dt
        pos = self.model.body_pos[self._peg_body_id]
        new_z = float(pos[2]) + self._peg_vz * dt
        rest_z = table_place_top_z() + float(PEG_HALF_HEIGHT) + PEG_REST_Z_OFFSET
        if new_z <= rest_z:
            # Landed: settle upright on the table at the frozen XY.
            self._peg_falling = False
            self._peg_vz = 0.0
            self.model.body_pos[self._peg_body_id] = np.array(
                [self._peg_fall_xy[0], self._peg_fall_xy[1], rest_z], dtype=np.float64
            )
            self.model.body_quat[self._peg_body_id] = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        else:
            self.model.body_pos[self._peg_body_id, 2] = new_z
        self._peg_fall_steps += 1
        if self._peg_fall_steps >= PEG_FALL_MAX_STEPS:
            self._peg_falling = False
            self._peg_vz = 0.0
        mujoco.mj_forward(self.model, self.data)

    def _first_geom_on_body(self, body_id: int) -> int:
        """Return the first geom id attached to ``body_id``, or -1 if none."""
        if body_id < 0:
            return -1
        for gid in range(int(self.model.ngeom)):
            if int(self.model.geom_bodyid[gid]) == body_id:
                return int(gid)
        return -1

    def _grasp_center_world(self) -> np.ndarray | None:
        """World-space midpoint of the right-gripper main finger meshes.

        Uses mesh geom frames (which swing with jaw open/close), not finger
        body origins at the hinges. This is the real clamping point between
        the two pads, unlike ``right_tcp`` at the gripper root.
        """
        if self._rg_left_finger_geom_id < 0 or self._rg_right_finger_geom_id < 0:
            return None
        lp = np.asarray(self.data.geom_xpos[self._rg_left_finger_geom_id], dtype=np.float64)
        rp = np.asarray(self.data.geom_xpos[self._rg_right_finger_geom_id], dtype=np.float64)
        return (lp + rp) * 0.5

    def _peg_center_world(self) -> np.ndarray | None:
        """World-space center of the peg (cylinder body center = geometric center)."""
        if self._peg_body_id < 0:
            return None
        return np.asarray(self.model.body_pos[self._peg_body_id], dtype=np.float64)

    def _finger_peg_signed_distance(self, finger_geom_id: int) -> float:
        """Signed distance from a finger mesh to the peg cylinder (m).

        Negative means the meshes penetrate / clamp; positive is a gap.
        Returns a large value when ids are invalid.
        """
        if finger_geom_id < 0 or self._peg_geom_id < 0:
            return float(PEG_GRASP_DIST_MAX)
        fromto = np.zeros(6, dtype=np.float64)
        return float(
            mujoco.mj_geomDistance(
                self.model,
                self.data,
                int(finger_geom_id),
                int(self._peg_geom_id),
                float(PEG_GRASP_DIST_MAX),
                fromto,
            )
        )

    def _jaw_side_peg_distance(self, geom_ids: tuple[int, ...]) -> float:
        """Closest signed distance from any geom on one jaw side to the peg."""
        if not geom_ids:
            return float(PEG_GRASP_DIST_MAX)
        return float(min(self._finger_peg_signed_distance(gid) for gid in geom_ids))

    def _peg_between_fingers(self) -> bool:
        """True when the peg center lies between the two main finger mesh frames."""
        if self._peg_body_id < 0 or self._rg_left_finger_geom_id < 0 or self._rg_right_finger_geom_id < 0:
            return False
        left = np.asarray(self.data.geom_xpos[self._rg_left_finger_geom_id], dtype=np.float64)
        right = np.asarray(self.data.geom_xpos[self._rg_right_finger_geom_id], dtype=np.float64)
        peg = np.asarray(self.model.body_pos[self._peg_body_id], dtype=np.float64)
        axis = right - left
        span2 = float(np.dot(axis, axis))
        if span2 < 1e-10:
            return False
        t = float(np.dot(peg - left, axis) / span2)
        return PEG_GRASP_BETWEEN_LO < t < PEG_GRASP_BETWEEN_HI

    def _stick_peg_to_grasp_center(self) -> None:
        """Keep the peg fixed where the fingers clamped it while attached.

        On the first attach we lock the offset between the peg center and the
        grasp center (finger midpoint). Each subsequent step we reapply
        ``peg_pos = grasp_center + locked_offset`` so the peg translates rigidly
        with the hand without snapping into the palm. The peg stays upright.
        """
        if self._peg_body_id < 0:
            return
        gc = self._grasp_center_world()
        if gc is None:
            return
        if not self._peg_attached:
            # Lock the relative position at the instant of grasping.
            pc = self._peg_center_world()
            self._peg_grasp_offset = np.asarray(pc, dtype=np.float64) - np.asarray(gc, dtype=np.float64)
            self._peg_attach_R_rel = np.eye(3, dtype=np.float64)
            self._peg_attached = True
            # Freeze jaw aperture here — further close commands are ignored.
            self._peg_attach_gripper_norm = self._gripper_normalized(RIGHT_GRIPPER_INDEX)
        self.model.body_pos[self._peg_body_id] = np.asarray(gc, dtype=np.float64) + self._peg_grasp_offset
        self.model.body_quat[self._peg_body_id] = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        mujoco.mj_forward(self.model, self.data)

    def _fingers_bracket_peg(self) -> bool:
        """True when both jaw pads nearly touch the peg and the peg is between them.

        Each jaw side (links 1+2) must have signed mesh distance to the peg
        ``<= PEG_GRASP_SURFACE_TOL``. The peg center must also project between
        the two primary finger frames along the jaw axis (rejects same-side
        poke-through when the hand sweeps past the cylinder).
        """
        if (
            self._peg_body_id < 0
            or self._peg_geom_id < 0
            or not self._rg_left_finger_geom_ids
            or not self._rg_right_finger_geom_ids
        ):
            return False
        l_dist = self._jaw_side_peg_distance(self._rg_left_finger_geom_ids)
        r_dist = self._jaw_side_peg_distance(self._rg_right_finger_geom_ids)
        if l_dist > PEG_GRASP_SURFACE_TOL or r_dist > PEG_GRASP_SURFACE_TOL:
            return False
        return self._peg_between_fingers()

    def _update_table_place_grasp(self) -> None:
        """Grasp/release the peg when both right-gripper pads clamp its surface.

        A grasp registers when both jaw sides are nearly touching the peg with
        the peg between the fingers (``_fingers_bracket_peg``) and the gripper
        is closed. Release happens when the gripper opens past the release
        threshold. While attached, the peg stays where it was clamped (locked
        offset) and translates with the hand.
        """
        if self._peg_body_id < 0:
            return

        if self._peg_attached:
            if self._gripper_open_enough(RIGHT_GRIPPER_INDEX):
                self._peg_attached = False
                self._peg_attach_gripper_norm = None
            else:
                self._stick_peg_to_grasp_center()
            return

        if self._fingers_bracket_peg() and self._right_gripper_closed():
            self._stick_peg_to_grasp_center()

    def _apply_prop_presence(self, *, forward: bool = False) -> None:
        """Park removed sword/shield far away so they leave RGB / depth / pointmap."""
        if self.scene != SCENE_REACH_SWORD:
            return
        changed = False
        park = np.asarray(PROP_PARK_POS, dtype=np.float64)
        if self._sword_body_id >= 0:
            if self.remove_sword:
                self.model.body_pos[self._sword_body_id] = park
                self._sword_attached = False
                changed = True
            elif self._sword_home_pos is not None and not self._sword_attached:
                self.model.body_pos[self._sword_body_id] = self._sword_home_pos
                self.model.body_quat[self._sword_body_id] = self._sword_home_quat
                changed = True
        if self._shield_body_id >= 0:
            if self.remove_shield:
                self.model.body_pos[self._shield_body_id] = park
                self._shield_attached = False
                changed = True
            elif self._shield_home_pos is not None and not self._shield_attached:
                self.model.body_pos[self._shield_body_id] = self._shield_home_pos
                self.model.body_quat[self._shield_body_id] = self._shield_home_quat
                changed = True
        if changed and forward:
            mujoco.mj_forward(self.model, self.data)

    def set_prop_removed(
        self, *, remove_sword: bool | None = None, remove_shield: bool | None = None
    ) -> None:
        """Hot-update whether sword/shield are present in the scene."""
        if remove_sword is not None:
            self.remove_sword = bool(remove_sword)
        if remove_shield is not None:
            self.remove_shield = bool(remove_shield)
        if self.scene != SCENE_REACH_SWORD:
            return
        if self._sword_body_id >= 0 and self._sword_home_pos is not None and not self._sword_attached:
            self.model.body_pos[self._sword_body_id] = self._sword_home_pos
            self.model.body_quat[self._sword_body_id] = self._sword_home_quat
        if self._shield_body_id >= 0 and self._shield_home_pos is not None and not self._shield_attached:
            self.model.body_pos[self._shield_body_id] = self._shield_home_pos
            self.model.body_quat[self._shield_body_id] = self._shield_home_quat
        self._apply_prop_presence(forward=True)

    def _render_camera(self, camera: str) -> np.ndarray:
        if self._render_failed:
            return np.zeros((self.obs_height, self.obs_width, 3), dtype=np.uint8)

        try:
            self._renderer.update_scene(self.data, camera=camera)
            image = self._renderer.render()
            return np.asarray(image, dtype=np.uint8).copy()
        except Exception:
            self._render_failed = True
            return np.zeros((self.obs_height, self.obs_width, 3), dtype=np.uint8)

    def _ensure_renderer(self) -> None:
        """Lazily create the MuJoCo renderer on first use."""
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.model, height=self.obs_height, width=self.obs_width)

    def _get_obs(self) -> dict[str, Any]:
        self._ensure_renderer()
        images = {camera: self._render_camera(camera) for camera in CAMERAS}
        obs: dict[str, Any] = {"pixels": images}
        if self.obs_type == "pixels_agent_pos":
            obs["agent_pos"] = self._normalize(self._current_control_qpos())
        return obs

    def set_execution_point(
        self,
        *,
        enable: bool | None = None,
        position: tuple[float, float, float] | list[float] | None = None,
    ) -> None:
        """Hot-update post-grasp execution waypoint settings."""
        if enable is not None:
            self.enable_execution_point = bool(enable)
            if not self.enable_execution_point:
                self._execution_point_passed = False
        if position is not None:
            vals = tuple(float(x) for x in position)
            if len(vals) != 3:
                raise ValueError(f"execution_point_pos must have 3 values, got {len(vals)}")
            self.execution_point_pos = vals

    def _reach_info(self) -> dict[str, Any]:
        if self.scene == SCENE_TABLE_PLACE:
            return self._table_place_info()
        if self.scene != SCENE_REACH_SWORD:
            return {
                "success": False,
                "is_success": False,
                "reach_distance": None,
                "left_reach_distance": None,
                "right_reach_distance": None,
                "left_success": False,
                "right_success": False,
                "attached": False,
                "left_attached": False,
                "right_attached": False,
                "both_attached": False,
                "gripper_closed": False,
                "left_gripper_closed": False,
                "right_gripper_closed": False,
                "execution_point_enabled": False,
                "execution_point_passed": False,
                "execution_point_distance": None,
                "execution_point_success": False,
                "peg_attached": False,
                "place_distance": None,
                "place_fit": None,
            }
        right_tcp = np.asarray(self.data.site_xpos[self._tcp_site_id], dtype=np.float64)
        left_tcp = np.asarray(self.data.site_xpos[self._left_tcp_site_id], dtype=np.float64)
        sword_handle = np.asarray(self.data.site_xpos[self._handle_site_id], dtype=np.float64)
        shield_handle = np.asarray(self.data.site_xpos[self._shield_handle_site_id], dtype=np.float64)
        right_dist, right_ok = evaluate_reach(right_tcp, sword_handle, self.reach_success_threshold)
        left_dist, left_ok = evaluate_reach(left_tcp, shield_handle, self.reach_success_threshold)
        right_attached = bool(self._sword_attached) and not self.remove_sword
        left_attached = bool(self._shield_attached) and not self.remove_shield
        both = left_attached and right_attached
        right_closed = self._right_gripper_closed()
        left_closed = self._left_gripper_closed()

        exec_enabled = bool(self.enable_execution_point) and not self.remove_sword
        exec_dist = None
        exec_now = False
        if exec_enabled:
            exec_pos = np.asarray(self.execution_point_pos, dtype=np.float64).reshape(3)
            exec_dist, exec_now = evaluate_reach(sword_handle, exec_pos, self.reach_success_threshold)
            # Pass-through: once the grasped sword handle enters the radius, latch success.
            if right_attached and exec_now:
                self._execution_point_passed = True
        else:
            self._execution_point_passed = False

        exec_passed = bool(self._execution_point_passed) if exec_enabled else False
        if exec_enabled:
            episode_success = right_attached and exec_passed
        else:
            # Grasp-only: proximity alone is not success — the right gripper must
            # actually clamp the sword handle (kinematic attach).
            episode_success = right_attached

        return {
            # Grasp attach (and optional execution-point pass) — not mere <5cm reach.
            "success": episode_success,
            "is_success": episode_success,
            "reach_distance": right_dist,
            "left_reach_distance": left_dist,
            "right_reach_distance": right_dist,
            "left_success": left_ok,
            "right_success": right_ok,
            "attached": right_attached,
            "grasp_attached": right_attached,
            "left_attached": left_attached,
            "right_attached": right_attached,
            "both_attached": both,
            "gripper_closed": right_closed,
            "left_gripper_closed": left_closed,
            "right_gripper_closed": right_closed,
            "execution_point_enabled": exec_enabled,
            "execution_point_passed": exec_passed,
            "execution_point_distance": exec_dist,
            "execution_point_success": exec_passed,
            "execution_point_pos": list(self.execution_point_pos),
            "peg_attached": False,
            "place_distance": None,
            "place_fit": None,
        }

    def _table_place_info(self) -> dict[str, Any]:
        # Reach metric: grasp center (finger midpoint) to peg center. Falls back
        # to the right_tcp site / peg_grasp site if finger bodies are unavailable.
        grasp_center = self._grasp_center_world()
        right_tcp = (
            grasp_center
            if grasp_center is not None
            else np.asarray(self.data.site_xpos[self._tcp_site_id], dtype=np.float64)
        )
        peg_center = self._peg_center_world()
        peg_grasp = (
            peg_center
            if peg_center is not None
            else np.asarray(self.data.site_xpos[self._peg_grasp_site_id], dtype=np.float64)
        )
        circle = np.asarray(self.data.site_xpos[self._target_circle_site_id], dtype=np.float64)
        reach_dist, near = evaluate_reach(right_tcp, peg_grasp, self.reach_success_threshold)
        peg_attached = bool(self._peg_attached)
        right_closed = self._right_gripper_closed()
        peg_xy = np.asarray(self.model.body_pos[self._peg_body_id][:2], dtype=np.float64)
        circle_xy = np.asarray(circle[:2], dtype=np.float64)
        place_dist = float(np.linalg.norm(peg_xy - circle_xy))
        radius = float(TABLE_CIRCLE_RADIUS)
        place_fit = float(max(0.0, 1.0 - place_dist / max(radius, 1e-6)))
        # Phase-1 success: released on the circle (not attached) and inside radius.
        episode_success = (not peg_attached) and place_dist < radius
        return {
            "success": episode_success,
            "is_success": episode_success,
            "reach_distance": reach_dist,
            "left_reach_distance": None,
            "right_reach_distance": reach_dist,
            "left_success": False,
            "right_success": near,
            "attached": peg_attached,
            "grasp_attached": peg_attached,
            "left_attached": False,
            "right_attached": peg_attached,
            "both_attached": False,
            "gripper_closed": right_closed,
            "left_gripper_closed": self._left_gripper_closed(),
            "right_gripper_closed": right_closed,
            "execution_point_enabled": False,
            "execution_point_passed": False,
            "execution_point_distance": None,
            "execution_point_success": False,
            "peg_attached": peg_attached,
            "peg_falling": bool(self._peg_falling),
            "place_distance": place_dist,
            "place_fit": place_fit,
            "circle_center": [float(circle[0]), float(circle[1]), float(circle[2])],
            "circle_radius": radius,
            "peg_xy": [float(peg_xy[0]), float(peg_xy[1])],
            "table_top_z": float(table_place_top_z()),
        }

    def reset(
        self,
        *,
        seed: int | None = None,
        options: Mapping[str, Any] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        super().reset(seed=seed)
        self._elapsed_steps = 0
        self._sword_attached = False
        self._shield_attached = False
        self._peg_attached = False
        self._peg_falling = False
        self._peg_vz = 0.0
        self._peg_fall_steps = 0
        self._sword_attach_R_rel = np.eye(3, dtype=np.float64)
        self._shield_attach_R_rel = np.eye(3, dtype=np.float64)
        self._peg_attach_R_rel = np.eye(3, dtype=np.float64)
        self._peg_grasp_offset = np.zeros(3, dtype=np.float64)
        self._peg_attach_gripper_norm = None
        self._execution_point_passed = False
        mujoco.mj_resetData(self.model, self.data)

        if self.scene == SCENE_REACH_SWORD:
            if self._sword_body_id >= 0 and self._sword_home_pos is not None:
                self.model.body_pos[self._sword_body_id] = self._sword_home_pos
                self.model.body_quat[self._sword_body_id] = self._sword_home_quat
            if self._shield_body_id >= 0 and self._shield_home_pos is not None:
                self.model.body_pos[self._shield_body_id] = self._shield_home_pos
                self.model.body_quat[self._shield_body_id] = self._shield_home_quat
            self._apply_prop_presence(forward=False)
        elif self.scene == SCENE_TABLE_PLACE:
            peg_xy = self.peg_xy
            if options is not None and "peg_xy" in options:
                raw = options["peg_xy"]
                peg_xy = (float(raw[0]), float(raw[1]))
            self.place_peg_xy(peg_xy)
            self._peg_home_pos = np.asarray(self.model.body_pos[self._peg_body_id], dtype=np.float64).copy()

        start = np.asarray(START_POSE, dtype=np.float32)
        if options is not None and "state" in options:
            start = np.asarray(options["state"], dtype=np.float32)
        if start.shape != (16,):
            raise ValueError(f"Reset state must have shape (16,), got {start.shape}")

        self._last_action = np.clip(start, -1.0, 1.0).astype(np.float32)
        self._apply_action(self._last_action)
        return self._get_obs(), self._reach_info()

    def step(self, action: np.ndarray) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        if action.shape != (16,):
            raise ValueError(f"Action must have shape (16,), got {action.shape}")

        self._elapsed_steps += 1
        self._last_action = np.clip(action, -1.0, 1.0).astype(np.float32)
        self._apply_action(self._last_action)

        obs = self._get_obs()
        reward = 0.0
        info = self._reach_info()
        terminated = bool(self.terminate_on_success and info["success"])
        truncated = self._elapsed_steps >= self.max_episode_steps
        return obs, reward, terminated, truncated, info

    def render(self) -> np.ndarray | None:
        if self.render_mode == "human":
            if self._viewer is None:
                import mujoco.viewer

                self._viewer = mujoco.viewer.launch_passive(self.model, self.data)
                configure_viewer_camera(self._viewer)
            self._viewer.sync()
            return None
        self._ensure_renderer()
        return self._render_camera("top")

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None


def configure_viewer_camera(viewer) -> None:
    """Configure the MuJoCo passive viewer camera to frame the Unoarm robot.

    Call once after `mujoco.viewer.launch_passive(...)` so the whole upper body
    is visible instead of the base being centered and the arms clipped.
    """
    viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    viewer.cam.lookat[:] = [0.0, 0.0, 1.2]
    viewer.cam.distance = 3.5
    viewer.cam.azimuth = 120
    viewer.cam.elevation = -20


def configure_viewer_theme(viewer, model: mujoco.MjModel | None = None) -> None:
    """Make the MuJoCo passive viewer scene brighter and disable harsh shadows."""
    if model is not None:
        # In this MuJoCo binding the background/fog color is exposed as .fog.
        model.vis.rgba.fog[:] = [0.95, 0.95, 0.95, 1.0]

    scene = getattr(viewer, "scn", None)
    if scene is None:
        return

    scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 0
    for i in range(min(2, len(scene.lighting))):
        light = scene.lighting[i]
        if light.active:
            light.ambient[:] = [0.5, 0.5, 0.5]
            light.diffuse[:] = [0.8, 0.8, 0.8]
            light.specular[:] = [0.3, 0.3, 0.3]
