"""Tests for table_place scene (table + peg + target circle)."""

from __future__ import annotations

import sys
from pathlib import Path

import mujoco
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gym_unoarm.constants import (  # noqa: E402
    GRIPPER_CLOSE,
    GRIPPER_OPEN,
    PEG_DEFAULT_XY,
    PEG_FALL_MAX_STEPS,
    PEG_HALF_HEIGHT,
    RIGHT_GRIPPER_INDEX,
    SCENE_TABLE_PLACE,
    TABLE_CIRCLE_CENTER_XY,
    TABLE_CIRCLE_RADIUS,
    table_place_top_z,
)
from gym_unoarm.env import UnoarmEnv  # noqa: E402
from gym_unoarm.table_place_scene import ensure_table_place_xml  # noqa: E402


def test_ensure_table_place_xml_loads():
    path = ensure_table_place_xml()
    assert path.is_file()
    model = mujoco.MjModel.from_xml_path(str(path))
    assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "table") >= 0
    assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "peg") >= 0
    assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "peg_grasp") >= 0
    assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "target_circle") >= 0
    assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "right_tcp") >= 0


def test_table_place_env_reset_and_place_metrics():
    env = UnoarmEnv(scene=SCENE_TABLE_PLACE)
    _, info = env.reset()
    assert info["peg_attached"] is False
    assert info["place_distance"] is not None
    assert info["circle_radius"] == pytest.approx(TABLE_CIRCLE_RADIUS)
    assert abs(info["peg_xy"][0] - PEG_DEFAULT_XY[0]) < 1e-6
    # Peg not on circle by default → success false
    assert info["success"] is False
    env.place_peg_xy(TABLE_CIRCLE_CENTER_XY)
    info = env._reach_info()
    assert info["place_distance"] < TABLE_CIRCLE_RADIUS
    assert info["place_fit"] > 0.9
    assert info["success"] is True  # released + inside circle
    env.close()


def _step_until_peg_settled(env: UnoarmEnv, action: np.ndarray, max_steps: int = 200) -> dict:
    """Step the env (holding the given action) until the peg stops falling."""
    info: dict = {}
    for _ in range(max_steps):
        _, _, _, _, info = env.step(action)
        if not env._peg_falling:
            break
    return info


def test_table_place_peg_attach_and_release():
    env = UnoarmEnv(scene=SCENE_TABLE_PLACE)
    env.reset()
    # Align the peg CENTER with the right-gripper grasp center (finger midpoint),
    # then close the gripper. The grasp now keys off the finger midpoint, not the
    # old right_tcp site / peg top point.
    peg_id = env._peg_body_id
    half = float(env._peg_grasp_local[2])
    grasp_center = np.asarray(env._grasp_center_world(), dtype=np.float64).copy()
    assert grasp_center is not None
    env.model.body_pos[peg_id] = grasp_center.astype(np.float64)
    env.model.body_quat[peg_id] = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    mujoco.mj_forward(env.model, env.data)

    info = env._reach_info()
    assert info["reach_distance"] < env.reach_success_threshold
    assert info["peg_attached"] is False

    action = env._last_action.copy()
    action[RIGHT_GRIPPER_INDEX] = GRIPPER_CLOSE
    _, _, _, _, info = env.step(action)
    assert info["peg_attached"] is True

    # Release: peg should START falling (not instantly seated on the table).
    action[RIGHT_GRIPPER_INDEX] = GRIPPER_OPEN
    _, _, _, _, info = env.step(action)
    assert info["peg_attached"] is False
    assert info["peg_falling"] is True
    # Still above the table on the first release step.
    assert float(env.model.body_pos[peg_id][2]) > table_place_top_z() + half - 1e-4

    # Continue stepping (gripper kept open) until the peg settles on the table.
    info = _step_until_peg_settled(env, action)
    assert not env._peg_falling
    # Seated upright on the table top after the fall completes.
    assert abs(float(env.model.body_pos[peg_id][2]) - (table_place_top_z() + half)) < 1e-3
    env.close()


def test_table_place_stops_closing_after_successful_grasp():
    """Once attached, further close commands must not tighten the jaw."""
    env = UnoarmEnv(scene=SCENE_TABLE_PLACE)
    env.reset()
    peg_id = env._peg_body_id
    grasp_center = np.asarray(env._grasp_center_world(), dtype=np.float64)
    env.model.body_pos[peg_id] = grasp_center.astype(np.float64)
    env.model.body_quat[peg_id] = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    mujoco.mj_forward(env.model, env.data)

    # Close gradually until attach (mirrors real closing motion).
    action = env._last_action.copy()
    attached_g = None
    for g in np.linspace(GRIPPER_OPEN, GRIPPER_CLOSE, 24):
        action[RIGHT_GRIPPER_INDEX] = float(g)
        env.step(action)
        if env._peg_attached:
            attached_g = float(env._gripper_normalized(RIGHT_GRIPPER_INDEX))
            break
    assert env._peg_attached is True
    assert attached_g is not None
    assert env._peg_attach_gripper_norm is not None

    # Command full close — jaw must stay at the attach aperture.
    action[RIGHT_GRIPPER_INDEX] = GRIPPER_CLOSE
    for _ in range(5):
        env.step(action)
    assert env._peg_attached is True
    cur = float(env._gripper_normalized(RIGHT_GRIPPER_INDEX))
    assert cur >= attached_g - 0.05
    assert cur > GRIPPER_CLOSE + 0.05
    env.close()


def test_table_place_grasp_uses_finger_midpoint():
    """Grasping keys off the finger-mesh midpoint and locks the peg in place (no snap)."""
    env = UnoarmEnv(scene=SCENE_TABLE_PLACE)
    env.reset()
    peg_id = env._peg_body_id

    # Close first so the mesh midpoint is stable, then seat the peg there.
    action = env._last_action.copy()
    action[RIGHT_GRIPPER_INDEX] = GRIPPER_CLOSE
    for _ in range(15):
        env.step(action)

    grasp_center = np.asarray(env._grasp_center_world(), dtype=np.float64).copy()
    env._peg_attached = False
    env.model.body_pos[peg_id] = grasp_center.astype(np.float64)
    env.model.body_quat[peg_id] = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    mujoco.mj_forward(env.model, env.data)

    env.step(action)

    assert env._peg_attached is True
    # With zero offset at attach, the peg stays at the grasp center (no snap away).
    gc = np.asarray(env._grasp_center_world(), dtype=np.float64)
    peg_center = np.asarray(env.model.body_pos[peg_id], dtype=np.float64)
    assert np.allclose(peg_center, gc, atol=1e-6)
    env.close()


def test_table_place_grasp_locks_offset_no_snap():
    """Grasping must NOT move the peg to the hand; it stays where clamped."""
    env = UnoarmEnv(scene=SCENE_TABLE_PLACE)
    env.reset()
    peg_id = env._peg_body_id

    # Place the peg within grasp radius but OFF the grasp center (e.g. slightly above).
    gc0 = np.asarray(env._grasp_center_world(), dtype=np.float64).copy()
    offset = np.array([0.0, 0.0, 0.03], dtype=np.float64)  # 3 cm above the hand center
    peg_at_attach = gc0 + offset
    env.model.body_pos[peg_id] = peg_at_attach.astype(np.float64)
    env.model.body_quat[peg_id] = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    mujoco.mj_forward(env.model, env.data)

    action = env._last_action.copy()
    action[RIGHT_GRIPPER_INDEX] = GRIPPER_CLOSE
    env.step(action)

    assert env._peg_attached is True
    # The peg must remain where it was clamped, not snap into the hand center.
    peg_after = np.asarray(env.model.body_pos[peg_id], dtype=np.float64)
    assert np.allclose(peg_after, peg_at_attach, atol=1e-6)
    assert not np.allclose(peg_after, gc0, atol=1e-3)
    env.close()


def test_table_place_no_grasp_when_fingers_miss_peg():
    """Closing the gripper away from the peg must NOT attach (no clipping grasp)."""
    env = UnoarmEnv(scene=SCENE_TABLE_PLACE)
    env.reset()
    peg_id = env._peg_body_id
    half = float(env._peg_grasp_local[2])

    # Park the peg on its home spot (resting on the table), far from the fingers.
    env.place_peg_xy(env.peg_xy)
    mujoco.mj_forward(env.model, env.data)
    grasp_center = np.asarray(env._grasp_center_world(), dtype=np.float64)
    peg_center = np.asarray(env.model.body_pos[peg_id], dtype=np.float64)
    # Sanity: at reset pose the fingers are NOT on the peg.
    assert float(np.linalg.norm(grasp_center - peg_center)) > env.reach_success_threshold

    action = env._last_action.copy()
    action[RIGHT_GRIPPER_INDEX] = GRIPPER_CLOSE
    _, _, _, _, info = env.step(action)
    assert info["peg_attached"] is False
    assert env._peg_attached is False
    env.close()


def test_table_place_grasp_requires_both_pads_near_surface():
    """Attach only when both finger meshes are nearly touching the peg surface."""
    env = UnoarmEnv(scene=SCENE_TABLE_PLACE)
    env.reset()
    peg_id = env._peg_body_id

    # Close the jaw first so the finger meshes swing into the clamp pose.
    action = env._last_action.copy()
    action[RIGHT_GRIPPER_INDEX] = GRIPPER_CLOSE
    for _ in range(15):
        env.step(action)

    # Place the peg at the finger-mesh midpoint (real pad clamp point).
    grasp_center = np.asarray(env._grasp_center_world(), dtype=np.float64).copy()
    env._peg_attached = False
    env.model.body_pos[peg_id] = grasp_center.astype(np.float64)
    env.model.body_quat[peg_id] = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    mujoco.mj_forward(env.model, env.data)

    assert env._fingers_bracket_peg() is True
    l_dist = env._jaw_side_peg_distance(env._rg_left_finger_geom_ids)
    r_dist = env._jaw_side_peg_distance(env._rg_right_finger_geom_ids)
    assert l_dist <= 0.015
    assert r_dist <= 0.015

    _, _, _, _, info = env.step(action)
    assert info["peg_attached"] is True
    env.close()


def test_table_place_grasp_allows_slight_peg_offset():
    """Slightly off-center peg still attaches when both pads are near the surface."""
    env = UnoarmEnv(scene=SCENE_TABLE_PLACE)
    env.reset()
    peg_id = env._peg_body_id

    action = env._last_action.copy()
    action[RIGHT_GRIPPER_INDEX] = GRIPPER_CLOSE
    for _ in range(15):
        env.step(action)

    grasp_center = np.asarray(env._grasp_center_world(), dtype=np.float64).copy()
    # 2 cm lateral offset previously failed the opposite-dot check even though
    # both pads were already on the cylinder surface.
    peg_pos = grasp_center + np.array([0.0, 0.02, 0.0], dtype=np.float64)
    env._peg_attached = False
    env.model.body_pos[peg_id] = peg_pos.astype(np.float64)
    env.model.body_quat[peg_id] = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    mujoco.mj_forward(env.model, env.data)

    assert env._fingers_bracket_peg() is True
    assert env._peg_between_fingers() is True
    _, _, _, _, info = env.step(action)
    assert info["peg_attached"] is True
    env.close()


def test_peg_fall_lands_on_table():
    """Directly trigger a fall from above the table and verify the landing."""
    env = UnoarmEnv(scene=SCENE_TABLE_PLACE)
    env.reset()
    peg_id = env._peg_body_id
    half = float(PEG_HALF_HEIGHT)
    drop_xy = (0.05, -0.40)  # arbitrary point above the table
    start_z = table_place_top_z() + half + 0.15  # 15 cm above rest
    env.model.body_pos[peg_id] = np.array(
        [drop_xy[0], drop_xy[1], start_z], dtype=np.float64
    )
    env.model.body_quat[peg_id] = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    mujoco.mj_forward(env.model, env.data)

    env._start_peg_fall()
    assert env._peg_falling is True
    assert env._peg_fall_xy == pytest.approx(drop_xy, abs=1e-9)

    # Hold an idle action; the fall advances inside _update_grasps each step.
    action = env._last_action.copy()
    info = _step_until_peg_settled(env, action)
    assert not env._peg_falling

    rest_z = table_place_top_z() + half
    landed_z = float(env.model.body_pos[peg_id][2])
    assert abs(landed_z - rest_z) < 1e-3  # settled on the table top
    # XY preserved (straight-down fall, no drift).
    assert float(env.model.body_pos[peg_id][0]) == pytest.approx(drop_xy[0], abs=1e-6)
    assert float(env.model.body_pos[peg_id][1]) == pytest.approx(drop_xy[1], abs=1e-6)
    # Upright orientation (identity quaternion).
    quat = np.asarray(env.model.body_quat[peg_id], dtype=np.float64)
    assert np.allclose(quat, [1.0, 0.0, 0.0, 0.0], atol=1e-9)
    assert info["peg_falling"] is False
    env.close()


def test_peg_fall_completes_within_step_budget():
    """The fall must terminate well within the PEG_FALL_MAX_STEPS safety budget."""
    env = UnoarmEnv(scene=SCENE_TABLE_PLACE)
    env.reset()
    peg_id = env._peg_body_id
    half = float(PEG_HALF_HEIGHT)
    # A moderate drop height that should settle in a handful of steps.
    start_z = table_place_top_z() + half + 0.10
    env.model.body_pos[peg_id] = np.array(
        [TABLE_CIRCLE_CENTER_XY[0], TABLE_CIRCLE_CENTER_XY[1], start_z],
        dtype=np.float64,
    )
    env.model.body_quat[peg_id] = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    mujoco.mj_forward(env.model, env.data)
    env._start_peg_fall()

    action = env._last_action.copy()
    steps = 0
    for _ in range(PEG_FALL_MAX_STEPS * 2):
        env.step(action)
        steps += 1
        if not env._peg_falling:
            break
    assert not env._peg_falling
    assert steps <= PEG_FALL_MAX_STEPS
    env.close()
