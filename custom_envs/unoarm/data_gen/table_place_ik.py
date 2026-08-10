"""Table-place Reach-IK: pick-and-place dataset generation (multi-point peg).

Samples peg XY inside a reachable on-table workspace (never past the table rim),
plus approach/lift/place trajectory parameters. Solves right-arm IK keyframes
with horizontal or vertical grasp orientation, replays with attach/fall physics,
and writes only successful episodes that keep the right arm out of the table
collision box (lift → XY transport → lower) and land the peg in the fixed circle.
"""

from __future__ import annotations

import json
import shutil
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import mujoco
import numpy as np
from gym_unoarm.constants import (
    FPS,
    GRIPPER_CLOSE,
    GRIPPER_OPEN,
    LEFT_GRIPPER_INDEX,
    PEG_DEFAULT_XY,
    PEG_FALL_MAX_STEPS,
    PEG_HALF_HEIGHT,
    PEG_MIN_DIST_FROM_CIRCLE,
    PEG_SAMPLE_EDGE_MARGIN,
    PEG_WORKSPACE_X_RANGE,
    PEG_WORKSPACE_Y_RANGE,
    RIGHT_GRIPPER_INDEX,
    SCENE_TABLE_PLACE,
    TABLE_CIRCLE_CENTER_XY,
    TABLE_CIRCLE_RADIUS,
    TABLE_PLACE_HALF,
    TABLE_PLACE_POS,
    TASK_TABLE_PLACE,
    clip_peg_xy_to_workspace,
    is_peg_xy_in_workspace,
    peg_workspace_xy_bounds,
    sample_peg_xy,
    table_place_top_z,
)
from gym_unoarm.ik import solve_right_tcp_ik

from .scripted import (
    add_frame,
    build_episode_actions,
    clip_raw_pose,
    dataset_features,
    images_are_black,
)

LogFn = Callable[[str], None]

TABLE_PLACE_IK_META_NAME = "table_place_ik_meta.json"
DEFAULT_TABLE_CLEARANCE_M = 0.01
DEFAULT_TABLE_BOX_MARGIN_M = 0.01
DEFAULT_APPROACH_OFFSET_Y_RANGE = (0.04, 0.10)
DEFAULT_SEGMENT_STEPS = 20
DEFAULT_HOLD_STEPS = 2
DEFAULT_IK_RETRIES = 8
FINGER_TABLE_PENETRATION_TOL = -0.025
ARM_TABLE_PENETRATION_TOL = -1e-3
# Grasp orientation is constrained via axis alignment (radians → cos threshold).
TABLE_PLACE_IK_POS_TOL = 0.02
TABLE_PLACE_IK_ORI_TOL = 0.35
TABLE_PLACE_IK_MAX_ITERS = 500
TABLE_PLACE_TRANSPORT_WAYPOINTS = 5
TABLE_PLACE_XY_HOP_M = 0.06
TABLE_PLACE_LIFT_ORI_TOL = 0.85
# Bias toward side grasps: tip-down is often unreachable after the table shift.
_GRASP_ORI_VERTICAL_PROB = 0.25
# Skip appending a keyframe if arm joints barely moved (avoids long mid-air freezes).
_POSE_ARM_DEDUP_EPS = 0.035
_MAX_TRANSPORT_KEYFRAMES = 6

GraspOrientation = Literal["horizontal", "vertical"]
GRASP_ORI_HORIZONTAL: GraspOrientation = "horizontal"
GRASP_ORI_VERTICAL: GraspOrientation = "vertical"

# Jaw open/close tracks approximately TCP +Z on this gripper. For an upright
# cylinder we always keep +Z nearly horizontal so pads clamp the *side wall*,
# never the top/bottom faces.
#
# Vertical: tip down (+X ≈ −Z), approach from above, jaws along +X.
_vx = np.array([0.0, 0.0, -1.0], dtype=np.float64)
_vz = np.array([1.0, 0.0, 0.0], dtype=np.float64)
_vy = np.cross(_vz, _vx)
_vy = _vy / (np.linalg.norm(_vy) + 1e-12)
_vz = np.cross(_vx, _vy)
_vz = _vz / (np.linalg.norm(_vz) + 1e-12)
VERTICAL_APPROACH_ROT_MAT = np.column_stack([_vx, _vy, _vz])

# Horizontal: reach in along −Y (+X ≈ −Y), jaws along +X (table-parallel).
_hx = np.array([0.0, -1.0, 0.0], dtype=np.float64)
_hz = np.array([1.0, 0.0, 0.0], dtype=np.float64)
_hy = np.cross(_hz, _hx)
_hy = _hy / (np.linalg.norm(_hy) + 1e-12)
_hz = np.cross(_hx, _hy)
_hz = _hz / (np.linalg.norm(_hz) + 1e-12)
HORIZONTAL_APPROACH_ROT_MAT = np.column_stack([_hx, _hy, _hz])

# Reject grasps whose jaw axis is too vertical (would pinch top/bottom caps).
_MAX_JAW_ABS_Z = 0.45


@dataclass
class TablePlaceIkParams:
    approach_offset_y: float
    lift_z: float
    place_z: float
    grasp_orientation: GraspOrientation = GRASP_ORI_HORIZONTAL


@dataclass
class TablePlaceIkGenConfig:
    num_episodes: int = 1
    segment_steps: int = DEFAULT_SEGMENT_STEPS
    hold_steps: int = DEFAULT_HOLD_STEPS
    settle_steps: int = PEG_FALL_MAX_STEPS
    approach_offset_y_range: tuple[float, float] = DEFAULT_APPROACH_OFFSET_Y_RANGE
    lift_z_range: tuple[float, float] | None = None
    place_z_range: tuple[float, float] | None = None
    table_clearance_m: float = DEFAULT_TABLE_CLEARANCE_M
    table_box_margin_m: float = DEFAULT_TABLE_BOX_MARGIN_M
    pose_jitter_std: float = 0.0
    seed: int = 0
    ik_retries: int = DEFAULT_IK_RETRIES
    max_attempts: int | None = None
    output_root: Path | None = None
    repo_id: str = "doki/unoarm_table_place_ik"
    overwrite: bool = False
    append: bool = False  # resume into existing dataset (mutually exclusive with overwrite)
    task: str = TASK_TABLE_PLACE
    # Multi-point peg: sample XY each attempt inside table ∩ reachable workspace.
    sample_peg_xy: bool = True
    peg_xy_fixed: tuple[float, float] | None = None
    peg_workspace_x_range: tuple[float, float] = PEG_WORKSPACE_X_RANGE
    peg_workspace_y_range: tuple[float, float] = PEG_WORKSPACE_Y_RANGE
    peg_edge_margin: float = PEG_SAMPLE_EDGE_MARGIN
    peg_min_dist_from_circle: float = PEG_MIN_DIST_FROM_CIRCLE
    # Prefer multiple successful trajectories per peg XY (vary approach params).
    episodes_per_peg: int = 2
    max_attempts_per_peg: int | None = None


def grasp_orientation_rot_mat(mode: GraspOrientation) -> np.ndarray:
    if mode == GRASP_ORI_VERTICAL:
        return VERTICAL_APPROACH_ROT_MAT.copy()
    return HORIZONTAL_APPROACH_ROT_MAT.copy()


def _jaw_separation_unit(env) -> np.ndarray | None:
    """Unit vector from left→right finger-mesh centroids (jaw open axis)."""
    left_ids = getattr(env, "_rg_left_finger_geom_ids", ()) or ()
    right_ids = getattr(env, "_rg_right_finger_geom_ids", ()) or ()
    if not left_ids or not right_ids:
        return None
    lo = np.mean([env.data.geom_xpos[int(g)] for g in left_ids], axis=0)
    ro = np.mean([env.data.geom_xpos[int(g)] for g in right_ids], axis=0)
    sep = np.asarray(ro, dtype=np.float64) - np.asarray(lo, dtype=np.float64)
    n = float(np.linalg.norm(sep))
    if n < 1e-6:
        return None
    return sep / n


def tcp_orientation_matches(env, mode: GraspOrientation, *, cos_tol: float = 0.85) -> bool:
    """True when TCP pose matches the mode and jaws open horizontally.

    - ``vertical``: TCP +X points down; approach from above.
    - ``horizontal``: TCP +X nearly parallel to the table (side reach).

    Both require the jaw open axis to stay nearly horizontal so pads clamp the
    cylinder *side*, not the top/bottom faces.
    """
    if int(getattr(env, "_tcp_site_id", -1)) < 0:
        return False
    x = np.asarray(env.data.site_xmat[env._tcp_site_id], dtype=np.float64).reshape(3, 3)[:, 0]
    x = x / (np.linalg.norm(x) + 1e-12)
    if mode == GRASP_ORI_VERTICAL:
        if float(-x[2]) < float(cos_tol):
            return False
    else:
        max_abs_z = float(np.sqrt(max(0.0, 1.0 - float(cos_tol) ** 2)))
        if abs(float(x[2])) > max_abs_z + 1e-6:
            return False
    jaw = _jaw_separation_unit(env)
    if jaw is None:
        return True
    return abs(float(jaw[2])) <= _MAX_JAW_ABS_Z


def table_collision_aabb(margin_m: float = DEFAULT_TABLE_BOX_MARGIN_M) -> tuple[np.ndarray, np.ndarray]:
    """Return (center, half_extents) of the inflated table collision box."""
    center = np.asarray(TABLE_PLACE_POS, dtype=np.float64)
    half = np.asarray(TABLE_PLACE_HALF, dtype=np.float64) + float(margin_m)
    return center, half


def right_arm_geom_ids(env) -> list[int]:
    ids: list[int] = []
    for gid in range(int(env.model.ngeom)):
        bid = int(env.model.geom_bodyid[gid])
        name = mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_BODY, bid) or ""
        if name.startswith("Right_"):
            ids.append(int(gid))
    return ids


def _table_geom_id(env) -> int:
    return int(mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_GEOM, "table_geom"))


def _peg_geom_id(env) -> int:
    gid = int(getattr(env, "_peg_geom_id", -1))
    if gid >= 0:
        return gid
    return int(mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_GEOM, "peg_geom"))


def _is_right_finger_pad_body(body_name: str) -> bool:
    """True for jaw pad links that may approach the peg to grasp."""
    if "Gripper" not in body_name:
        return False
    # Primary finger meshes: ..._Left_1/2_Link, ..._Right_1/2_Link
    return any(
        tag in body_name
        for tag in (
            "Left_1_Link",
            "Left_2_Link",
            "Right_1_Link",
            "Right_2_Link",
        )
    )


def _is_pregrasp_peg_collision_body(body_name: str) -> bool:
    """Arm links that must clear the peg before the jaw closes.

    Excludes finger pads and wrist flange (``Right_Link7``), which sit next to
    the hand and naturally come near the cylinder during pregrasp.
    """
    if body_name.startswith("Right_Link"):
        return body_name != "Right_Link7"
    # Gripper support / base (not pads) should also not bat the peg.
    return bool("Gripper" in body_name and not _is_right_finger_pad_body(body_name))


def arm_links_clear_peg(
    env,
    *,
    min_dist: float = 0.002,
) -> bool:
    """True when proximal right-arm geoms do not touch the peg.

    Used **before gripper close**: upper arm / forearm must not hit the
    cylinder. Finger pads and the wrist flange are excluded so a normal
    side-grasp pregrasp remains possible.
    """
    peg_gid = _peg_geom_id(env)
    if peg_gid < 0:
        return True
    fromto = np.zeros(6, dtype=np.float64)
    for gid in right_arm_geom_ids(env):
        bid = int(env.model.geom_bodyid[gid])
        bname = mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_BODY, bid) or ""
        if not _is_pregrasp_peg_collision_body(bname):
            continue
        dist = float(
            mujoco.mj_geomDistance(
                env.model,
                env.data,
                int(gid),
                int(peg_gid),
                1.0,
                fromto,
            )
        )
        if dist < float(min_dist):
            return False
    return True


def _pregrasp_arm_peg_index_range(n_poses: int, segment_steps: int, hold_steps: int) -> tuple[int, int]:
    """Steps before grasp_close where arm links must clear the peg.

    Pose layout: approach(0), pregrasp(1), grasp_close(2), ...
    Check through the end of pregrasp (index 1); closing may contact the peg.
    """
    ends = _pose_phase_end_indices(n_poses, segment_steps, hold_steps)
    if not ends:
        return 0, 0
    return 0, ends[min(1, len(ends) - 1)]


def _geom_origin_in_aabb(env, gid: int, center: np.ndarray, half: np.ndarray) -> bool:
    """True when the geom frame origin lies inside the AABB."""
    xpos = np.asarray(env.data.geom_xpos[int(gid)], dtype=np.float64)
    return bool(np.all(np.abs(xpos - center) <= half + 1e-9))


def arm_clears_table_collision_box(
    env,
    *,
    margin_m: float = DEFAULT_TABLE_BOX_MARGIN_M,
    penetration_tol: float = ARM_TABLE_PENETRATION_TOL,
    require_aabb: bool = True,
) -> bool:
    """True when the right arm does not enter the table collision volume.

    When ``require_aabb`` is True (lift / transport / lower), link geom origins
    must stay outside the inflated table AABB and mesh distance >= 0.

    When False (approach / grasp near the tabletop), only deep mesh penetration
    is rejected so the jaw can work beside the peg on the table.
    """
    table_gid = _table_geom_id(env)
    if table_gid < 0:
        return True
    center, half = table_collision_aabb(margin_m)
    fromto = np.zeros(6, dtype=np.float64)
    for gid in right_arm_geom_ids(env):
        bid = int(env.model.geom_bodyid[gid])
        bname = mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_BODY, bid) or ""
        is_gripper = "Gripper" in bname
        if require_aabb and (not is_gripper) and _geom_origin_in_aabb(env, gid, center, half):
            return False
        dist = float(
            mujoco.mj_geomDistance(
                env.model,
                env.data,
                int(gid),
                int(table_gid),
                1.0,
                fromto,
            )
        )
        if is_gripper:
            min_ok = float(FINGER_TABLE_PENETRATION_TOL)
        elif require_aabb:
            min_ok = max(float(penetration_tol), 0.0)
        else:
            min_ok = -0.015
        if dist < min_ok:
            return False
    return True


def _arm_strict_index_range(n_poses: int, segment_steps: int, hold_steps: int) -> tuple[int, int]:
    """Steps where full table-box clearance is required: after lift through lower."""
    ends = _pose_phase_end_indices(n_poses, segment_steps, hold_steps)
    if len(ends) < 5:
        return ends[min(2, len(ends) - 1)], ends[-1]
    # From end of lift keyframe through end of lower (second-to-last pose).
    return ends[3], ends[-2] if len(ends) >= 2 else ends[-1]


def _default_z_range(table_clearance_m: float, *, low_extra: float, high_extra: float) -> tuple[float, float]:
    floor = table_place_top_z() + float(PEG_HALF_HEIGHT) + float(table_clearance_m)
    return (floor + float(low_extra), floor + float(high_extra))


def resolve_lift_place_ranges(
    config: TablePlaceIkGenConfig,
) -> tuple[tuple[float, float], tuple[float, float]]:
    clearance = float(config.table_clearance_m)
    # Higher default lift so forearm clears the table volume during XY moves.
    lift = config.lift_z_range or _default_z_range(clearance, low_extra=0.05, high_extra=0.12)
    place = config.place_z_range or _default_z_range(clearance, low_extra=0.02, high_extra=0.10)
    return (float(lift[0]), float(lift[1])), (float(place[0]), float(place[1]))


def sample_table_place_ik_params(
    rng: np.random.Generator,
    *,
    approach_offset_y_range: tuple[float, float],
    lift_z_range: tuple[float, float],
    place_z_range: tuple[float, float],
    table_top_z: float,
    peg_half_height: float,
    table_clearance_m: float,
    grasp_orientation: GraspOrientation | None = None,
) -> TablePlaceIkParams:
    floor = float(table_top_z) + float(peg_half_height) + float(table_clearance_m)
    ay_lo, ay_hi = float(approach_offset_y_range[0]), float(approach_offset_y_range[1])
    if ay_hi < ay_lo:
        ay_lo, ay_hi = ay_hi, ay_lo
    lift_lo = max(float(lift_z_range[0]), floor)
    lift_hi = max(float(lift_z_range[1]), floor)
    place_lo = max(float(place_z_range[0]), floor)
    place_hi = max(float(place_z_range[1]), floor)
    if lift_hi < lift_lo:
        lift_hi = lift_lo
    if place_hi < place_lo:
        place_hi = place_lo
    ori: GraspOrientation
    if grasp_orientation is None:
        ori = GRASP_ORI_VERTICAL if float(rng.random()) < _GRASP_ORI_VERTICAL_PROB else GRASP_ORI_HORIZONTAL
    else:
        ori = grasp_orientation
    return TablePlaceIkParams(
        approach_offset_y=float(rng.uniform(ay_lo, ay_hi)),
        lift_z=float(rng.uniform(lift_lo, lift_hi)),
        place_z=float(rng.uniform(place_lo, place_hi)),
        grasp_orientation=ori,
    )


def grasp_center_tcp_offset(env) -> np.ndarray:
    """World offset ``grasp_center - right_tcp`` (meters)."""
    tcp = np.asarray(env.data.site_xpos[env._tcp_site_id], dtype=np.float64)
    gc = env._grasp_center_world()
    if gc is None:
        return np.zeros(3, dtype=np.float64)
    return np.asarray(gc, dtype=np.float64) - tcp


def peg_bottom_clears_table(env, table_clearance_m: float) -> bool:
    peg = env._peg_center_world()
    if peg is None:
        return False
    bottom = float(peg[2]) - float(PEG_HALF_HEIGHT)
    return bottom >= float(table_place_top_z()) + float(table_clearance_m) - 1e-9


def fingers_clear_table(env, *, min_dist: float = FINGER_TABLE_PENETRATION_TOL) -> bool:
    """True when right finger meshes are not deeply penetrating the table geom."""
    table_gid = _table_geom_id(env)
    if table_gid < 0:
        return True
    finger_gids: list[int] = []
    for gids in (
        getattr(env, "_rg_left_finger_geom_ids", ()),
        getattr(env, "_rg_right_finger_geom_ids", ()),
    ):
        finger_gids.extend(int(g) for g in gids if int(g) >= 0)
    if not finger_gids:
        for attr in ("_rg_left_finger_geom_id", "_rg_right_finger_geom_id"):
            gid = int(getattr(env, attr, -1))
            if gid >= 0:
                finger_gids.append(gid)
    fromto = np.zeros(6, dtype=np.float64)
    for gid in finger_gids:
        dist = float(
            mujoco.mj_geomDistance(
                env.model,
                env.data,
                int(gid),
                int(table_gid),
                1.0,
                fromto,
            )
        )
        if dist < float(min_dist):
            return False
    return True


def lift_transport_clears_table(env, table_clearance_m: float) -> bool:
    """True when peg bottom clears the table and fingers are not deeply penetrating it."""
    if not peg_bottom_clears_table(env, table_clearance_m):
        return False
    return fingers_clear_table(env)


def _pose_phase_end_indices(n_poses: int, segment_steps: int, hold_steps: int) -> list[int]:
    """Exclusive end index of each keyframe's interpolate+hold after the start hold."""
    i = int(hold_steps)
    ends: list[int] = []
    for _ in range(n_poses):
        i += int(segment_steps) + int(hold_steps)
        ends.append(i)
    return ends


def _lift_transport_index_range(n_poses: int, segment_steps: int, hold_steps: int) -> tuple[int, int]:
    """Peg-bottom clearance after lift through last XY transport (before lower/release).

    Pose layout:
    approach(0), pregrasp(1), grasp_close(2), lift(3), *xy..., lower(-2), release(-1).
    """
    ends = _pose_phase_end_indices(n_poses, segment_steps, hold_steps)
    if len(ends) < 5:
        return ends[min(2, len(ends) - 1)], ends[-2] if len(ends) >= 2 else ends[-1]
    return ends[3], ends[-3] if len(ends) >= 6 else ends[3]


def _arm_pose_delta(a: np.ndarray, b: np.ndarray) -> float:
    """L2 delta of raw 16-D poses ignoring both gripper joints."""
    da = np.asarray(a, dtype=np.float64).reshape(16).copy()
    db = np.asarray(b, dtype=np.float64).reshape(16).copy()
    da[LEFT_GRIPPER_INDEX] = 0.0
    db[LEFT_GRIPPER_INDEX] = 0.0
    da[RIGHT_GRIPPER_INDEX] = 0.0
    db[RIGHT_GRIPPER_INDEX] = 0.0
    return float(np.linalg.norm(da - db))


def _append_pose(poses: list[np.ndarray], q: np.ndarray, *, force: bool = False) -> None:
    """Append a keyframe unless it is nearly identical to the last arm config."""
    q = np.asarray(q, dtype=np.float32).reshape(16)
    if (
        not force
        and poses
        and _arm_pose_delta(poses[-1], q) < _POSE_ARM_DEDUP_EPS
        and abs(float(poses[-1][RIGHT_GRIPPER_INDEX]) - float(q[RIGHT_GRIPPER_INDEX])) < 1e-4
    ):
        return
    poses.append(q.copy())


def _with_right_gripper(env, q: np.ndarray, *, open_: bool) -> np.ndarray:
    out = np.asarray(q, dtype=np.float32).reshape(16).copy()
    n = env._normalize(out)
    n[RIGHT_GRIPPER_INDEX] = GRIPPER_OPEN if open_ else GRIPPER_CLOSE
    return env._denormalize(n).astype(np.float32)


def _with_right_gripper_norm(env, q: np.ndarray, g_norm: float) -> np.ndarray:
    out = np.asarray(q, dtype=np.float32).reshape(16).copy()
    n = env._normalize(out)
    n[RIGHT_GRIPPER_INDEX] = float(np.clip(g_norm, -1.0, 1.0))
    return env._denormalize(n).astype(np.float32)


def _copy_right_gripper(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Return ``dst`` arm joints with the right-gripper value from ``src``."""
    out = np.asarray(dst, dtype=np.float32).reshape(16).copy()
    out[RIGHT_GRIPPER_INDEX] = float(np.asarray(src, dtype=np.float32).reshape(16)[RIGHT_GRIPPER_INDEX])
    return out


def _close_gripper_until_grasp(env, q_arm: np.ndarray, *, n_steps: int = 28) -> np.ndarray | None:
    """Close the right gripper from open until attach; return that raw pose.

    Stops at the first successful grasp instead of commanding full ``GRIPPER_CLOSE``.
    """
    q_arm = np.asarray(q_arm, dtype=np.float32).reshape(16)
    env.apply_raw_qpos(_with_right_gripper(env, q_arm, open_=True))
    if bool(getattr(env, "_peg_attached", False)):
        return env._current_control_qpos().astype(np.float32).copy()
    for g in np.linspace(float(GRIPPER_OPEN), float(GRIPPER_CLOSE), int(n_steps)):
        env.apply_raw_qpos(_with_right_gripper_norm(env, q_arm, float(g)))
        if bool(getattr(env, "_peg_attached", False)):
            return env._current_control_qpos().astype(np.float32).copy()
    return None


def _solve_tcp(
    env,
    target_pos: np.ndarray,
    *,
    target_rot: np.ndarray | None = None,
    ori_tol: float = TABLE_PLACE_IK_ORI_TOL,
) -> np.ndarray | None:
    """Right TCP IK with axis-based orientation (table_place).

    Uses TCP +X/+Z axis alignment instead of the small-angle SO(3) residual in
    ``solve_right_tcp_ik``, which can report tiny errors for large misalignments.
    """
    if int(getattr(env, "_tcp_site_id", -1)) < 0:
        return None
    pos = np.asarray(target_pos, dtype=np.float64).reshape(3)
    r_tgt = (
        HORIZONTAL_APPROACH_ROT_MAT
        if target_rot is None
        else np.asarray(target_rot, dtype=np.float64).reshape(3, 3)
    )

    joint_ids: list[int] = []
    qadr: list[int] = []
    dof_ids: list[int] = []
    for name in (f"Right_Joint{i}" for i in range(1, 8)):
        jid = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid < 0:
            raise RuntimeError(f"missing joint {name}")
        joint_ids.append(int(jid))
        qadr.append(int(env.model.jnt_qposadr[jid]))
        dof_ids.append(int(env.model.jnt_dofadr[jid]))

    # Warm start: get near the Cartesian target with loose orientation.
    warm = solve_right_tcp_ik(
        env,
        pos,
        target_rot=r_tgt,
        max_iters=250,
        pos_tol=0.04,
        ori_tol=3.0,
    )
    if warm is not None:
        env.apply_raw_qpos(np.asarray(warm, dtype=np.float32))

    damping = 5e-2
    cos_tol = float(np.cos(float(ori_tol)))  # ori_tol in radians ≈ axis angle
    for _ in range(int(TABLE_PLACE_IK_MAX_ITERS)):
        mujoco.mj_forward(env.model, env.data)
        tcp = np.asarray(env.data.site_xpos[env._tcp_site_id], dtype=np.float64)
        r_cur = np.asarray(env.data.site_xmat[env._tcp_site_id], dtype=np.float64).reshape(3, 3)
        pos_e = pos - tcp
        # Align +X and +Z axes via cross-product angular error.
        ori_e = np.cross(r_cur[:, 0], r_tgt[:, 0]) + 0.5 * np.cross(r_cur[:, 2], r_tgt[:, 2])
        x_dot = float(np.dot(r_cur[:, 0], r_tgt[:, 0]))
        if float(np.linalg.norm(pos_e)) < TABLE_PLACE_IK_POS_TOL and x_dot >= cos_tol:
            return env._current_control_qpos().astype(np.float32).copy()

        jacp = np.zeros((3, env.model.nv), dtype=np.float64)
        jacr = np.zeros((3, env.model.nv), dtype=np.float64)
        mujoco.mj_jacSite(env.model, env.data, jacp, jacr, int(env._tcp_site_id))
        J = np.vstack([jacp[:, dof_ids], jacr[:, dof_ids]])
        err = np.concatenate([pos_e, ori_e])
        H = J.T @ J + (float(damping) ** 2) * np.eye(7, dtype=np.float64)
        try:
            dq = np.linalg.solve(H, J.T @ err)
        except np.linalg.LinAlgError:
            return None
        for adr, dqi, jid in zip(qadr, dq, joint_ids, strict=True):
            lo, hi = env.model.jnt_range[jid]
            env.data.qpos[adr] = float(np.clip(env.data.qpos[adr] + dqi, lo, hi))
        env._apply_mimics()

    mujoco.mj_forward(env.model, env.data)
    tcp = np.asarray(env.data.site_xpos[env._tcp_site_id], dtype=np.float64)
    r_cur = np.asarray(env.data.site_xmat[env._tcp_site_id], dtype=np.float64).reshape(3, 3)
    x_dot = float(np.dot(r_cur[:, 0], r_tgt[:, 0]))
    if float(np.linalg.norm(pos - tcp)) < TABLE_PLACE_IK_POS_TOL * 1.5 and x_dot >= cos_tol * 0.95:
        return env._current_control_qpos().astype(np.float32).copy()
    return None


def _align_grasp_center_to_peg(
    env,
    peg: np.ndarray,
    *,
    target_rot: np.ndarray,
    max_iters: int = 20,
) -> np.ndarray | None:
    """Iteratively move TCP so the finger midpoint approaches ``peg``."""
    q: np.ndarray | None = None
    best_q: np.ndarray | None = None
    best_err = float("inf")
    for _ in range(max_iters):
        gc = env._grasp_center_world()
        if gc is None:
            return None
        err_vec = np.asarray(peg, dtype=np.float64) - np.asarray(gc, dtype=np.float64)
        err = float(np.linalg.norm(err_vec))
        q = env._current_control_qpos().astype(np.float32).copy()
        if err < best_err:
            best_err = err
            best_q = q
        if err < 0.015:
            return q
        tcp = np.asarray(env.data.site_xpos[env._tcp_site_id], dtype=np.float64)
        step = min(1.0, 0.04 / max(err, 1e-6))
        target = tcp + step * err_vec
        q_next = _solve_tcp(env, target, target_rot=target_rot, ori_tol=TABLE_PLACE_IK_ORI_TOL)
        if q_next is None:
            q_next = _solve_tcp(env, target, target_rot=target_rot, ori_tol=TABLE_PLACE_LIFT_ORI_TOL)
        if q_next is None:
            q_next = solve_right_tcp_ik(
                env,
                target,
                target_rot=target_rot,
                max_iters=TABLE_PLACE_IK_MAX_ITERS,
                pos_tol=0.03,
                ori_tol=3.0,
            )
        if q_next is None:
            break
        env.apply_raw_qpos(_with_right_gripper(env, q_next, open_=True))
        if not arm_links_clear_peg(env):
            # Arm link poked the peg while approaching — reject this path.
            return None
    if best_q is not None and best_err < 0.03:
        env.apply_raw_qpos(_with_right_gripper(env, best_q, open_=True))
        if not arm_links_clear_peg(env):
            return None
        return best_q
    return None


def _approach_tcp_target(peg: np.ndarray, params: TablePlaceIkParams) -> np.ndarray:
    """Approach TCP position for the selected grasp orientation."""
    off = float(params.approach_offset_y)
    if params.grasp_orientation == GRASP_ORI_VERTICAL:
        # Tip-down is hard at table depth: seed from the robot side (+Y) and above.
        return peg + np.array([0.0, max(off, 0.12), max(off, 0.10)], dtype=np.float64)
    # Side reach: modest +Y backup (large offsets often lose IK after table shift).
    return peg + np.array([0.0, max(min(off, 0.08), 0.04), 0.03], dtype=np.float64)


def _approach_tcp_candidates(peg: np.ndarray, params: TablePlaceIkParams) -> list[np.ndarray]:
    """Primary approach plus a few fallback seeds."""
    primary = _approach_tcp_target(peg, params)
    out = [primary]
    if params.grasp_orientation == GRASP_ORI_VERTICAL:
        for dy, dz in (
            (0.15, 0.10),
            (0.12, 0.12),
            (0.18, 0.10),
            (0.10, 0.14),
            (0.15, 0.15),
            (0.0, 0.12),
            (0.0, 0.16),
        ):
            cand = peg + np.array([0.0, dy, dz], dtype=np.float64)
            if float(np.linalg.norm(cand - primary)) > 1e-3:
                out.append(cand)
    else:
        for dx, dy, dz in (
            (0.0, 0.04, 0.02),
            (0.0, 0.06, 0.04),
            (0.0, 0.10, 0.06),
            (0.0, 0.10, 0.08),
            (0.0, 0.14, 0.05),
            (-0.04, 0.04, 0.02),
            (0.04, 0.04, 0.02),
            (0.0, 0.12, 0.06),
        ):
            cand = peg + np.array([dx, dy, dz], dtype=np.float64)
            if float(np.linalg.norm(cand - primary)) > 1e-3:
                out.append(cand)
    return out


def build_table_place_ik_poses(
    env,
    params: TablePlaceIkParams,
    *,
    table_box_margin_m: float = DEFAULT_TABLE_BOX_MARGIN_M,
) -> list[np.ndarray] | None:
    """Solve pick-and-place keyframes. None on IK / orientation / collision failure.

    Returns raw 16-D poses:
    ``[approach, pregrasp_open, grasp_close, lift, *xy_transport..., lower, release_open]``.
    """
    if getattr(env, "scene", None) != SCENE_TABLE_PLACE:
        raise RuntimeError("build_table_place_ik_poses requires scene=table_place")
    peg0 = env._peg_center_world()
    if peg0 is None or int(getattr(env, "_tcp_site_id", -1)) < 0:
        return None
    peg = np.asarray(peg0, dtype=np.float64).copy()
    circle_xy = np.asarray(TABLE_CIRCLE_CENTER_XY, dtype=np.float64)
    rot = grasp_orientation_rot_mat(params.grasp_orientation)

    _approach_tcp_target(peg, params)
    q_approach = None
    for cand in _approach_tcp_candidates(peg, params):
        q_approach = _solve_tcp(env, cand, target_rot=rot)
        if q_approach is None:
            continue
        env.apply_raw_qpos(_with_right_gripper(env, q_approach, open_=True))
        if (
            tcp_orientation_matches(env, params.grasp_orientation)
            and arm_clears_table_collision_box(env, margin_m=table_box_margin_m, require_aabb=False)
            and arm_links_clear_peg(env)
        ):
            break
        q_approach = None
    if q_approach is None:
        return None
    env.apply_raw_qpos(_with_right_gripper(env, q_approach, open_=True))
    if not tcp_orientation_matches(env, params.grasp_orientation):
        return None
    if not arm_clears_table_collision_box(env, margin_m=table_box_margin_m, require_aabb=False):
        return None
    if not arm_links_clear_peg(env):
        return None

    q_grasp = _align_grasp_center_to_peg(env, peg, target_rot=rot)
    if q_grasp is None:
        return None
    env.apply_raw_qpos(_with_right_gripper(env, q_grasp, open_=True))
    if not tcp_orientation_matches(env, params.grasp_orientation):
        return None
    if not arm_links_clear_peg(env):
        return None

    grasp_open = _with_right_gripper(env, q_grasp, open_=True)
    grasp_close = _close_gripper_until_grasp(env, q_grasp)
    if grasp_close is None:
        return None
    env.apply_raw_qpos(grasp_close)
    if not bool(getattr(env, "_peg_attached", False)):
        return None

    offset = grasp_center_tcp_offset(env)
    lift_tcp = np.array([peg[0], peg[1], float(params.lift_z)], dtype=np.float64) - offset
    q_lift = _solve_tcp(env, lift_tcp, target_rot=rot, ori_tol=TABLE_PLACE_LIFT_ORI_TOL)
    if q_lift is None:
        # Climb in small Z hops from the grasp TCP while keeping orientation.
        tcp = np.asarray(env.data.site_xpos[env._tcp_site_id], dtype=np.float64)
        target_z = float(lift_tcp[2])
        cur = tcp.copy()
        q_lift = None
        for _ in range(16):
            if cur[2] >= target_z - 0.01:
                break
            cur = cur.copy()
            cur[2] = min(target_z, float(cur[2]) + 0.025)
            cur[0], cur[1] = float(lift_tcp[0]), float(lift_tcp[1])
            q_hop = _solve_tcp(env, cur, target_rot=rot, ori_tol=max(TABLE_PLACE_LIFT_ORI_TOL, 1.0))
            if q_hop is None:
                # Position-primary fallback for this hop.
                q_hop = solve_right_tcp_ik(
                    env,
                    cur,
                    target_rot=rot,
                    max_iters=TABLE_PLACE_IK_MAX_ITERS,
                    pos_tol=0.03,
                    ori_tol=3.0,
                )
            if q_hop is None:
                break
            q_lift = q_hop
            env.apply_raw_qpos(_copy_right_gripper(grasp_close, q_hop))
        if (
            q_lift is not None
            and float(np.asarray(env.data.site_xpos[env._tcp_site_id], dtype=np.float64)[2]) < target_z - 0.04
        ):
            # Did not climb high enough — treat as failure.
            q_lift = None
    if q_lift is None:
        return None
    lift_close = clip_raw_pose(env, _copy_right_gripper(grasp_close, q_lift))
    env.apply_raw_qpos(lift_close)
    if not arm_clears_table_collision_box(env, margin_m=table_box_margin_m, require_aabb=True):
        return None
    if not peg_bottom_clears_table(env, DEFAULT_TABLE_CLEARANCE_M):
        return None

    poses: list[np.ndarray] = [
        clip_raw_pose(env, _with_right_gripper(env, q_approach, open_=True)),
        clip_raw_pose(env, grasp_open),
        clip_raw_pose(env, grasp_close),
        clip_raw_pose(env, lift_close),
    ]

    peg_lift = np.asarray(env._peg_center_world(), dtype=np.float64)
    transport_z = max(float(params.lift_z), float(peg_lift[2]))
    place_z = float(params.place_z)
    goal_xy = np.asarray(circle_xy, dtype=np.float64)

    last_q = lift_close
    n_transport = 0
    # Phase A: XY-only hops at constant high Z (no simultaneous descent).
    for _ in range(max(1, int(TABLE_PLACE_TRANSPORT_WAYPOINTS)) * 2):
        if n_transport >= _MAX_TRANSPORT_KEYFRAMES:
            break
        peg_now = np.asarray(env._peg_center_world(), dtype=np.float64)
        err_xy = np.array(
            [goal_xy[0] - peg_now[0], goal_xy[1] - peg_now[1], 0.0],
            dtype=np.float64,
        )
        if float(np.linalg.norm(err_xy[:2])) < 0.025:
            break
        nrm = float(np.linalg.norm(err_xy[:2]))
        if nrm > TABLE_PLACE_XY_HOP_M:
            err_xy[:2] *= TABLE_PLACE_XY_HOP_M / nrm
        offset_now = grasp_center_tcp_offset(env)
        target_gc = peg_now + err_xy
        target_gc[2] = transport_z
        target_tcp = target_gc - offset_now
        q_tp = _solve_tcp(env, target_tcp, target_rot=rot, ori_tol=TABLE_PLACE_LIFT_ORI_TOL)
        if q_tp is None:
            q_tp = _solve_tcp(
                env,
                target_tcp + np.array([0.0, 0.0, 0.03], dtype=np.float64),
                target_rot=rot,
                ori_tol=TABLE_PLACE_LIFT_ORI_TOL + 0.2,
            )
        if q_tp is None:
            break
        cand = clip_raw_pose(env, _copy_right_gripper(grasp_close, q_tp))
        if _arm_pose_delta(last_q, cand) < _POSE_ARM_DEDUP_EPS:
            # IK stalled — stop hopping instead of stacking frozen closed poses.
            break
        last_q = cand
        env.apply_raw_qpos(last_q)
        if not arm_clears_table_collision_box(env, margin_m=table_box_margin_m, require_aabb=True):
            return None
        _append_pose(poses, last_q)
        n_transport += 1

    # Phase B: one lower (closed) then open — never stack duplicate closed hover poses.
    offset_now = grasp_center_tcp_offset(env)
    peg_now = np.asarray(env._peg_center_world(), dtype=np.float64)
    lower_gc = np.array([goal_xy[0], goal_xy[1], place_z], dtype=np.float64)
    if float(np.linalg.norm(peg_now[:2] - goal_xy)) < 0.04:
        lower_gc[0], lower_gc[1] = float(peg_now[0]), float(peg_now[1])
    q_lower = _solve_tcp(env, lower_gc - offset_now, target_rot=rot, ori_tol=TABLE_PLACE_LIFT_ORI_TOL)
    if q_lower is not None:
        lower_close = clip_raw_pose(env, _copy_right_gripper(grasp_close, q_lower))
        env.apply_raw_qpos(lower_close)
        if arm_clears_table_collision_box(env, margin_m=table_box_margin_m, require_aabb=True):
            last_q = lower_close
            _append_pose(poses, last_q)

    release_open = clip_raw_pose(env, _with_right_gripper(env, last_q, open_=True))
    # Always append open even if arm pose matches — gripper change must be visible.
    _append_pose(poses, release_open, force=True)
    if len(poses) < 6:
        return None
    return poses


def resolve_episode_peg_xy(config: TablePlaceIkGenConfig, rng: np.random.Generator) -> tuple[float, float]:
    """Pick peg XY for one attempt: sampled or fixed, always clipped to workspace."""
    xr = config.peg_workspace_x_range
    yr = config.peg_workspace_y_range
    margin = float(config.peg_edge_margin)
    min_d = float(config.peg_min_dist_from_circle)
    if config.sample_peg_xy:
        return sample_peg_xy(
            rng,
            edge_margin=margin,
            x_range=xr,
            y_range=yr,
            min_dist_from_circle=min_d,
        )
    raw = config.peg_xy_fixed if config.peg_xy_fixed is not None else PEG_DEFAULT_XY
    clipped = clip_peg_xy_to_workspace(raw, edge_margin=margin, x_range=xr, y_range=yr)
    if not is_peg_xy_in_workspace(
        clipped,
        edge_margin=margin,
        x_range=xr,
        y_range=yr,
        min_dist_from_circle=min_d,
    ):
        # Fixed peg too close to circle: nudge by resampling once near clipped point.
        return sample_peg_xy(
            rng,
            edge_margin=margin,
            x_range=xr,
            y_range=yr,
            min_dist_from_circle=min_d,
        )
    return clipped


def table_place_ik_config_from_params(
    params: dict[str, Any],
    *,
    output_root: Path,
    repo_id: str | None = None,
) -> TablePlaceIkGenConfig:
    """Build ``TablePlaceIkGenConfig`` from a Web/API params dict."""
    clearance = float(params.get("table_clearance_m", DEFAULT_TABLE_CLEARANCE_M))
    default_lift = _default_z_range(clearance, low_extra=0.05, high_extra=0.12)
    default_place = _default_z_range(clearance, low_extra=0.02, high_extra=0.10)

    def _pair(lo_key: str, hi_key: str, default: tuple[float, float]) -> tuple[float, float]:
        if params.get(lo_key) is None or params.get(hi_key) is None:
            return default
        return (float(params[lo_key]), float(params[hi_key]))

    ay = _pair(
        "approach_offset_y_min",
        "approach_offset_y_max",
        DEFAULT_APPROACH_OFFSET_Y_RANGE,
    )
    out_name_repo = repo_id or str(params.get("repo_id") or "doki/unoarm_table_place_ik")
    max_attempts = params.get("max_attempts")
    sample_peg = params.get("sample_peg_xy")
    sample_peg = True if sample_peg is None else bool(sample_peg)
    peg_fixed = params.get("peg_xy")
    peg_xy_fixed: tuple[float, float] | None = None
    if peg_fixed is not None and len(peg_fixed) >= 2:
        peg_xy_fixed = (float(peg_fixed[0]), float(peg_fixed[1]))
    return TablePlaceIkGenConfig(
        num_episodes=int(params.get("num_episodes", 1)),
        segment_steps=int(params.get("segment_steps", DEFAULT_SEGMENT_STEPS)),
        hold_steps=int(params.get("hold_steps", DEFAULT_HOLD_STEPS)),
        settle_steps=int(params.get("settle_steps", PEG_FALL_MAX_STEPS)),
        approach_offset_y_range=ay,
        lift_z_range=_pair("lift_z_min", "lift_z_max", default_lift),
        place_z_range=_pair("place_z_min", "place_z_max", default_place),
        table_clearance_m=clearance,
        table_box_margin_m=float(params.get("table_box_margin_m", DEFAULT_TABLE_BOX_MARGIN_M)),
        pose_jitter_std=float(params.get("pose_jitter_std", 0.0)),
        seed=int(params.get("seed", 0)),
        ik_retries=int(params.get("ik_retries", DEFAULT_IK_RETRIES)),
        max_attempts=int(max_attempts) if max_attempts is not None else None,
        output_root=Path(output_root),
        repo_id=out_name_repo,
        append=bool(params.get("append", False)),
        # append wins: never wipe an existing dataset when resuming
        overwrite=False if bool(params.get("append", False)) else bool(params.get("overwrite", False)),
        task=str(params.get("task") or TASK_TABLE_PLACE),
        sample_peg_xy=sample_peg,
        peg_xy_fixed=peg_xy_fixed,
        peg_workspace_x_range=_pair("peg_x_min", "peg_x_max", PEG_WORKSPACE_X_RANGE),
        peg_workspace_y_range=_pair("peg_y_min", "peg_y_max", PEG_WORKSPACE_Y_RANGE),
        peg_edge_margin=float(params.get("peg_edge_margin", PEG_SAMPLE_EDGE_MARGIN)),
        peg_min_dist_from_circle=float(params.get("peg_min_dist_from_circle", PEG_MIN_DIST_FROM_CIRCLE)),
        episodes_per_peg=int(params.get("episodes_per_peg", 2)),
        max_attempts_per_peg=(
            int(params["max_attempts_per_peg"]) if params.get("max_attempts_per_peg") is not None else None
        ),
    )


def load_table_place_ik_meta(root: Path) -> dict[str, Any] | None:
    path = Path(root) / TABLE_PLACE_IK_META_NAME
    if not path.is_file():
        return None
    raw = json.loads(path.read_text(encoding="utf-8"))
    return raw if isinstance(raw, dict) else None


def episode_peg_xy_from_meta(meta: dict[str, Any] | None, episode_index: int) -> tuple[float, float] | None:
    """Return peg XY recorded for ``episode_index`` (or legacy top-level peg_xy)."""
    if not isinstance(meta, dict):
        return None
    for row in meta.get("episodes") or []:
        if not isinstance(row, dict):
            continue
        if int(row.get("episode_index", -1)) != int(episode_index):
            continue
        raw = row.get("peg_xy")
        if raw is not None and len(raw) >= 2:
            return (float(raw[0]), float(raw[1]))
    raw = meta.get("peg_xy")
    if raw is not None and len(raw) >= 2:
        return (float(raw[0]), float(raw[1]))
    return None


def generate_table_place_ik_dataset(config: TablePlaceIkGenConfig, *, log: LogFn = print) -> Path:
    """Generate a LeRobot dataset of successful table-place pick-and-place episodes."""
    if config.num_episodes < 1:
        raise ValueError(f"num_episodes must be >= 1, got {config.num_episodes}")
    if config.pose_jitter_std < 0.0:
        raise ValueError(f"pose_jitter_std must be >= 0, got {config.pose_jitter_std}")
    if config.table_clearance_m < 0.0:
        raise ValueError(f"table_clearance_m must be >= 0, got {config.table_clearance_m}")
    if config.table_box_margin_m < 0.0:
        raise ValueError(f"table_box_margin_m must be >= 0, got {config.table_box_margin_m}")
    if config.peg_edge_margin < 0.0:
        raise ValueError(f"peg_edge_margin must be >= 0, got {config.peg_edge_margin}")
    if int(config.episodes_per_peg) < 1:
        raise ValueError(f"episodes_per_peg must be >= 1, got {config.episodes_per_peg}")
    # Validate workspace intersection is non-empty before long generate loops.
    peg_workspace_xy_bounds(
        edge_margin=config.peg_edge_margin,
        x_range=config.peg_workspace_x_range,
        y_range=config.peg_workspace_y_range,
    )

    lift_range, place_range = resolve_lift_place_ranges(config)
    max_attempts = int(
        config.max_attempts
        if config.max_attempts is not None
        else max(config.num_episodes * 40, config.num_episodes + 80)
    )
    episodes_per_peg = int(config.episodes_per_peg)
    max_attempts_per_peg = int(
        config.max_attempts_per_peg
        if config.max_attempts_per_peg is not None
        else max(episodes_per_peg * 40, episodes_per_peg + 40)
    )

    output_root = Path(
        config.output_root or (Path(__file__).resolve().parents[1] / "data" / "unoarm_table_place_ik")
    )
    if config.append and config.overwrite:
        raise ValueError("append and overwrite cannot both be True")
    if config.append:
        if not output_root.is_dir():
            raise FileNotFoundError(f"append=True but dataset root does not exist: {output_root}")
        if not (output_root / "meta" / "info.json").is_file():
            raise FileNotFoundError(
                f"append=True but {output_root} is not a LeRobot dataset (missing meta/info.json)"
            )
    elif output_root.exists():
        if not config.overwrite:
            raise FileExistsError(
                f"{output_root} already exists. Pass overwrite=True to replace it, "
                "or append=True to continue writing into it."
            )
        log(f"Removing existing dataset at {output_root}")
        shutil.rmtree(output_root)

    from gym_unoarm.env import UnoarmEnv

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    existing_meta = load_table_place_ik_meta(output_root) if config.append else None
    base_episode_metas: list[dict[str, Any]] = []
    if isinstance(existing_meta, dict):
        raw_eps = existing_meta.get("episodes") or []
        base_episode_metas = [e for e in raw_eps if isinstance(e, dict)]

    if config.append:
        dataset = LeRobotDataset.resume(
            repo_id=config.repo_id,
            root=output_root,
        )
        base_n = int(dataset.meta.total_episodes)
        log(
            f"Appending to existing dataset at {output_root} "
            f"(already {base_n} episodes, meta_rows={len(base_episode_metas)})"
        )
    else:
        dataset = LeRobotDataset.create(
            repo_id=config.repo_id,
            fps=FPS,
            features=dataset_features(),
            root=output_root,
            robot_type="unoarm",
            use_videos=True,
        )
        base_n = 0

    env = UnoarmEnv(
        obs_type="pixels_agent_pos",
        render_mode="rgb_array",
        max_episode_steps=10_000,
        scene=SCENE_TABLE_PLACE,
        peg_xy=PEG_DEFAULT_XY,
        terminate_on_success=False,
    )

    master_rng = np.random.default_rng(config.seed)
    warned_black = False
    written = 0
    skipped = 0
    attempts = 0
    episode_metas: list[dict[str, Any]] = []
    task_desc = config.task or TASK_TABLE_PLACE
    box_margin = float(config.table_box_margin_m)

    (wx0, wy0), (wx1, wy1) = peg_workspace_xy_bounds(
        edge_margin=config.peg_edge_margin,
        x_range=config.peg_workspace_x_range,
        y_range=config.peg_workspace_y_range,
    )
    log(
        f"table_place IK: need {config.num_episodes} new successes "
        f"(base={base_n}, append={config.append}, "
        f"episodes_per_peg={episodes_per_peg}, "
        f"max_attempts={max_attempts}, max_per_peg={max_attempts_per_peg}) "
        f"task={task_desc!r} sample_peg={config.sample_peg_xy} "
        f"peg_workspace=[{wx0:.3f},{wx1:.3f}]x[{wy0:.3f},{wy1:.3f}]"
    )

    try:
        while written < config.num_episodes and attempts < max_attempts:
            peg_xy = resolve_episode_peg_xy(config, master_rng)
            peg_written = 0
            peg_tries = 0
            reset_opts = {"peg_xy": peg_xy}
            log(
                f"target peg_xy={peg_xy} "
                f"(want {episodes_per_peg} traj, remaining slots "
                f"{config.num_episodes - written})"
            )
            while (
                peg_written < episodes_per_peg
                and written < config.num_episodes
                and attempts < max_attempts
                and peg_tries < max_attempts_per_peg
            ):
                attempts += 1
                peg_tries += 1
                params = sample_table_place_ik_params(
                    master_rng,
                    approach_offset_y_range=config.approach_offset_y_range,
                    lift_z_range=lift_range,
                    place_z_range=place_range,
                    table_top_z=table_place_top_z(),
                    peg_half_height=PEG_HALF_HEIGHT,
                    table_clearance_m=config.table_clearance_m,
                )
                poses: list[np.ndarray] | None = None
                home_q: np.ndarray | None = None
                tried_oris = [params.grasp_orientation]
                alt = (
                    GRASP_ORI_HORIZONTAL
                    if params.grasp_orientation == GRASP_ORI_VERTICAL
                    else GRASP_ORI_VERTICAL
                )
                tried_oris.append(alt)
                for ori_try in tried_oris:
                    params.grasp_orientation = ori_try
                    for _ in range(max(1, int(config.ik_retries))):
                        env.reset(options=reset_opts)
                        home_q = env._current_control_qpos().astype(np.float32).copy()
                        poses = build_table_place_ik_poses(env, params, table_box_margin_m=box_margin)
                        if poses is not None:
                            break
                    if poses is not None:
                        break
                if poses is None or home_q is None:
                    skipped += 1
                    log(
                        f"skip attempt {attempts}: IK failed peg_xy={peg_xy} "
                        f"peg_traj={peg_written}/{episodes_per_peg} params={params}"
                    )
                    continue

                rng = np.random.default_rng(config.seed + 10_000 + attempts)
                env.reset(options=reset_opts)
                start_q = home_q.copy()
                if config.pose_jitter_std > 0.0:
                    jitter = rng.normal(0.0, config.pose_jitter_std, size=16).astype(np.float32)
                    jitter[RIGHT_GRIPPER_INDEX] = 0.0
                    jitter[7] = 0.0
                    ep_poses = [clip_raw_pose(env, pose + jitter) for pose in poses]
                    start_q = clip_raw_pose(env, start_q + jitter)
                else:
                    ep_poses = poses

                raw_actions = build_episode_actions(
                    ep_poses,
                    config.segment_steps,
                    config.hold_steps,
                    rng,
                    midpoint_noise_std=0.0,
                    hold_noise_std=0.0,
                    start=start_q,
                )
                release = ep_poses[-1]
                for _ in range(max(0, int(config.settle_steps))):
                    raw_actions.append(release.copy())

                peg_clear_lo, peg_clear_hi = _lift_transport_index_range(
                    len(ep_poses), config.segment_steps, config.hold_steps
                )
                arm_lo, arm_hi = _arm_strict_index_range(
                    len(ep_poses), config.segment_steps, config.hold_steps
                )
                pre_lo, pre_hi = _pregrasp_arm_peg_index_range(
                    len(ep_poses), config.segment_steps, config.hold_steps
                )

                frame_buf: list[tuple[dict[str, Any], np.ndarray]] = []
                saw_attach = False
                clearance_ok = True
                arm_ok = True
                arm_peg_ok = True
                last_info: dict[str, Any] = {}
                for step_i, raw_action in enumerate(raw_actions):
                    action = env._normalize(clip_raw_pose(env, raw_action))
                    obs, _reward, _terminated, _truncated, last_info = env.step(action)
                    if not warned_black and images_are_black(obs):
                        log("warning: rendered images are all black; check MuJoCo offscreen rendering.")
                        warned_black = True
                    frame_buf.append((obs, action.copy()))
                    if bool(last_info.get("peg_attached", False)):
                        saw_attach = True
                    strict = arm_lo <= step_i < arm_hi
                    if not arm_clears_table_collision_box(env, margin_m=box_margin, require_aabb=strict):
                        arm_ok = False
                    if pre_lo <= step_i < pre_hi and not arm_links_clear_peg(env):
                        arm_peg_ok = False
                    if strict and not fingers_clear_table(env):
                        clearance_ok = False
                    if peg_clear_lo <= step_i < peg_clear_hi:
                        if not peg_bottom_clears_table(env, config.table_clearance_m):
                            clearance_ok = False

                success = bool(last_info.get("success", False))
                falling = bool(last_info.get("peg_falling", False))
                keep = saw_attach and clearance_ok and arm_ok and arm_peg_ok and success and not falling
                if not keep:
                    skipped += 1
                    log(
                        f"skip attempt {attempts}: peg_xy={peg_xy} "
                        f"peg_traj={peg_written}/{episodes_per_peg} "
                        f"saw_attach={saw_attach} clearance_ok={clearance_ok} "
                        f"arm_ok={arm_ok} arm_peg_ok={arm_peg_ok} "
                        f"success={success} falling={falling} "
                        f"place_dist={last_info.get('place_distance')} "
                        f"ori={params.grasp_orientation}"
                    )
                    continue

                for obs, action in frame_buf:
                    add_frame(dataset, obs, action, task_desc)
                dataset.save_episode()
                written += 1
                peg_written += 1
                ep_index = base_n + written - 1
                episode_metas.append(
                    {
                        "episode_index": ep_index,
                        "attempt": attempts,
                        "peg_xy": [float(peg_xy[0]), float(peg_xy[1])],
                        "peg_traj_index": peg_written,
                        "episodes_per_peg": episodes_per_peg,
                        "params": {
                            "approach_offset_y": params.approach_offset_y,
                            "lift_z": params.lift_z,
                            "place_z": params.place_z,
                            "grasp_orientation": params.grasp_orientation,
                        },
                        "n_frames": len(frame_buf),
                        "place_distance": float(last_info.get("place_distance") or 0.0),
                        "success": True,
                    }
                )
                log(
                    f"saved episode {written}/{config.num_episodes} "
                    f"(dataset ep={ep_index}, total={base_n + written}) "
                    f"peg_traj={peg_written}/{episodes_per_peg} "
                    f"({len(frame_buf)} frames) peg_xy={peg_xy} params={params}"
                )
            if peg_written < episodes_per_peg and written < config.num_episodes:
                log(f"abandon peg_xy={peg_xy}: got {peg_written}/{episodes_per_peg} after {peg_tries} tries")
    finally:
        dataset.finalize()
        env.close()

    prev_attempts = int((existing_meta or {}).get("num_attempts") or 0)
    prev_skipped = int((existing_meta or {}).get("num_attempts_skipped") or 0)
    meta = {
        "mode": "table_place_pick_place",
        "task": task_desc,
        "repo_id": config.repo_id,
        "sample_peg_xy": bool(config.sample_peg_xy),
        "episodes_per_peg": episodes_per_peg,
        "append": bool(config.append),
        "peg_workspace": {
            "x_range": [float(wx0), float(wx1)],
            "y_range": [float(wy0), float(wy1)],
            "edge_margin": float(config.peg_edge_margin),
            "min_dist_from_circle": float(config.peg_min_dist_from_circle),
            "requested_x_range": [
                float(config.peg_workspace_x_range[0]),
                float(config.peg_workspace_x_range[1]),
            ],
            "requested_y_range": [
                float(config.peg_workspace_y_range[0]),
                float(config.peg_workspace_y_range[1]),
            ],
        },
        "circle_center": [
            float(TABLE_CIRCLE_CENTER_XY[0]),
            float(TABLE_CIRCLE_CENTER_XY[1]),
            float(table_place_top_z()),
        ],
        "circle_radius": float(TABLE_CIRCLE_RADIUS),
        "table_clearance_m": float(config.table_clearance_m),
        "table_box_margin_m": float(config.table_box_margin_m),
        "num_episodes_written": base_n + written,
        "num_episodes_this_run": written,
        "num_attempts_skipped": prev_skipped + skipped,
        "num_attempts": prev_attempts + attempts,
        "num_attempts_this_run": attempts,
        "num_attempts_skipped_this_run": skipped,
        "episodes": base_episode_metas + episode_metas,
    }
    meta_path = output_root / TABLE_PLACE_IK_META_NAME
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    if written <= 0:
        raise RuntimeError(
            f"No episodes written after {attempts} attempts (skipped={skipped}). "
            "Check IK reachability, orientation, and table collision box."
        )
    log(
        f"done: wrote {written} new episodes "
        f"(dataset total={base_n + written}) to {output_root} "
        f"(skipped_this_run={skipped})"
    )
    return output_root
