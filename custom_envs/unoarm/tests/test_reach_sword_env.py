"""Tests for Unoarm reach-sword scene (kinematic success metric)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import gym_unoarm  # noqa: F401,E402
import gymnasium as gym  # noqa: E402
import mujoco  # noqa: E402
from gym_unoarm.constants import (  # noqa: E402
    REACH_SUCCESS_THRESHOLD,
    SCENE_REACH_SWORD,
    SWORD_HANDLE_LOCAL,
    TASK_REACH_SWORD,
)
from gym_unoarm.env import UnoarmEnv  # noqa: E402
from gym_unoarm.reach_scene import ensure_reach_sword_xml, evaluate_reach  # noqa: E402


def test_evaluate_reach_pure():
    tcp = np.array([1.0, 2.0, 3.0])
    handle = np.array([1.0, 2.0, 3.04])
    dist, ok = evaluate_reach(tcp, handle, 0.05)
    assert ok
    assert dist == pytest.approx(0.04, abs=1e-6)
    dist2, ok2 = evaluate_reach(tcp, handle, 0.03)
    assert not ok2
    assert dist2 == pytest.approx(0.04, abs=1e-6)


def test_ensure_reach_sword_xml_contains_sites_and_sword():
    path = ensure_reach_sword_xml()
    assert path.exists()
    text = path.read_text(encoding="utf-8")
    assert 'name="sword"' in text or 'mesh="sword"' in text
    assert 'name="shield"' in text
    assert "helmet.stl" in text
    assert 'name="left_tcp"' in text
    assert 'name="right_tcp"' in text
    assert 'name="sword_handle"' in text
    assert 'name="shield_handle"' in text
    assert "sword.stl" in text


def test_ensure_reach_sword_xml_rewrites_body_pos():
    path = ensure_reach_sword_xml(body_pos=(1.11, 2.22, 3.33))
    text = path.read_text(encoding="utf-8")
    assert 'pos="1.11 2.22 3.33"' in text
    # Second call with different pose must overwrite, not keep stale XML.
    path2 = ensure_reach_sword_xml(body_pos=(0.4, 0.55, 1.74))
    assert path2 == path
    text2 = path.read_text(encoding="utf-8")
    assert 'pos="0.4 0.55 1.74"' in text2
    assert "1.11 2.22 3.33" not in text2


def test_reach_env_applies_sword_body_pos():
    pos = (0.31, 0.62, 1.55)
    env = UnoarmEnv(scene=SCENE_REACH_SWORD, sword_body_pos=pos)
    sword_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, "sword")
    assert sword_id >= 0
    assert np.allclose(env.model.body_pos[sword_id], pos)
    env.close()


def test_resolve_sword_pose_keeps_handle_world():
    from gym_unoarm.sword_pose import handle_world_from_body, resolve_sword_pose

    handle = (-0.1, -0.55, 1.0)
    body, quat, hw = resolve_sword_pose(handle_pos=handle, euler_deg=(0.0, 0.0, 0.0))
    assert np.allclose(hw, handle)
    # Round-trip via MuJoCo-style: body + R @ local ≈ handle
    recovered = handle_world_from_body(body, quat)
    assert np.allclose(recovered, handle, atol=1e-6)

    body2, quat2, hw2 = resolve_sword_pose(handle_pos=handle, euler_deg=(30.0, 15.0, -10.0))
    assert np.allclose(hw2, handle)
    recovered2 = handle_world_from_body(body2, quat2)
    assert np.allclose(recovered2, handle, atol=1e-6)


def test_reach_env_tilted_handle_matches_setting():
    from gym_unoarm.sword_pose import resolve_sword_pose

    handle = (0.2, -0.4, 1.1)
    body, quat, _ = resolve_sword_pose(handle_pos=handle, euler_deg=(45.0, 20.0, 0.0))
    env = UnoarmEnv(
        scene=SCENE_REACH_SWORD,
        sword_body_pos=body,
        sword_body_quat=quat,
    )
    env.reset()
    handle_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_SITE, "sword_handle")
    xpos = np.asarray(env.data.site_xpos[handle_id], dtype=np.float64)
    assert np.allclose(xpos, handle, atol=1e-5)
    env.close()


def test_reach_env_sites_and_far_at_reset():
    env = UnoarmEnv(scene=SCENE_REACH_SWORD, terminate_on_success=False)
    _, info = env.reset()
    assert "reach_distance" in info
    assert info["success"] is False
    assert info["is_success"] is False
    assert info["reach_distance"] > REACH_SUCCESS_THRESHOLD

    tcp_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_SITE, "right_tcp")
    handle_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_SITE, "sword_handle")
    assert tcp_id >= 0
    assert handle_id >= 0
    env.close()


def test_reach_env_success_when_sword_grasped_at_tcp():
    """Success requires kinematic attach (near + gripper closed), not mere proximity."""
    env = UnoarmEnv(
        scene=SCENE_REACH_SWORD,
        reach_success_threshold=REACH_SUCCESS_THRESHOLD,
        terminate_on_success=True,
    )
    env.reset()
    tcp_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_SITE, "right_tcp")
    sword_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, "sword")
    assert tcp_id >= 0 and sword_id >= 0

    tcp = np.asarray(env.data.site_xpos[tcp_id], dtype=np.float64).copy()
    # body quat is 180° about X (wxyz 0,1,0,0): R @ (x,y,z) = (x, -y, -z)
    handle_local = np.asarray(SWORD_HANDLE_LOCAL, dtype=np.float64)
    handle_offset = np.array(
        [handle_local[0], -handle_local[1], -handle_local[2]],
        dtype=np.float64,
    )
    env.model.body_pos[sword_id] = tcp - handle_offset
    mujoco.mj_forward(env.model, env.data)

    # Near but gripper still open → not success.
    info = env._reach_info()
    assert info["reach_distance"] < REACH_SUCCESS_THRESHOLD
    assert info["right_attached"] is False
    assert info["success"] is False

    # Close right gripper while near → attach → success.
    action = env._last_action.copy()
    action[15] = -1.0  # right gripper close (normalized)
    _, _, terminated, _, step_info = env.step(action)
    assert step_info["right_attached"] is True
    assert step_info["success"] is True
    assert terminated is True
    env.close()


def test_execution_point_sticky_pass_requires_attach():
    exec_pos = (-0.15, -0.35, 1.15)
    env = UnoarmEnv(
        scene=SCENE_REACH_SWORD,
        enable_execution_point=True,
        execution_point_pos=exec_pos,
        reach_success_threshold=REACH_SUCCESS_THRESHOLD,
        terminate_on_success=True,
    )
    env.reset()
    info = env._reach_info()
    assert info["execution_point_enabled"] is True
    assert info["success"] is False
    assert info["execution_point_passed"] is False

    # Handle near waypoint but not attached → not success.
    sword_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, "sword")
    handle_local = np.asarray(SWORD_HANDLE_LOCAL, dtype=np.float64)
    handle_offset = np.array(
        [handle_local[0], -handle_local[1], -handle_local[2]],
        dtype=np.float64,
    )
    env.model.body_pos[sword_id] = np.asarray(exec_pos, dtype=np.float64) - handle_offset
    mujoco.mj_forward(env.model, env.data)
    info = env._reach_info()
    assert info["execution_point_distance"] < REACH_SUCCESS_THRESHOLD
    assert info["success"] is False

    # Force kinematic attach then pass → sticky success.
    env._sword_attached = True
    info = env._reach_info()
    assert info["execution_point_passed"] is True
    assert info["success"] is True

    # Move handle far away; sticky flag keeps success.
    env.model.body_pos[sword_id] = np.array([5.0, 5.0, 5.0], dtype=np.float64)
    mujoco.mj_forward(env.model, env.data)
    info = env._reach_info()
    assert info["execution_point_passed"] is True
    assert info["success"] is True

    # Reset clears sticky flag.
    _, info = env.reset()
    assert info["execution_point_passed"] is False
    assert info["success"] is False
    env.close()


def test_gym_register_reach_sword():
    env = gym.make("gym_unoarm/UnoarmReachSword-v0")
    assert isinstance(env.unwrapped, UnoarmEnv)
    assert env.unwrapped.scene == SCENE_REACH_SWORD
    env.close()


def test_task_string_constant():
    assert "left" in TASK_REACH_SWORD.lower() or "shield" in TASK_REACH_SWORD.lower()
    assert "right" in TASK_REACH_SWORD.lower()
    assert "sword" in TASK_REACH_SWORD.lower()


def test_reach_env_has_shield_and_left_metrics():
    env = UnoarmEnv(scene=SCENE_REACH_SWORD)
    _, info = env.reset()
    assert env._shield_body_id >= 0
    assert env._left_tcp_site_id >= 0
    assert info["left_reach_distance"] is not None
    assert info["right_reach_distance"] is not None
    assert info["left_attached"] is False
    assert info["right_attached"] is False
    shield_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, "shield")
    assert shield_id >= 0
    # Shield on +X (robot left), sword on −X (robot right).
    assert env.model.body_pos[shield_id][0] > 0
    assert env.model.body_pos[env._sword_body_id][0] < 0
    env.close()


def _place_sword_at_tcp(env: UnoarmEnv) -> None:
    tcp_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_SITE, "right_tcp")
    sword_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, "sword")
    tcp = np.asarray(env.data.site_xpos[tcp_id], dtype=np.float64).copy()
    handle_local = np.asarray(SWORD_HANDLE_LOCAL, dtype=np.float64)
    # Default body quat is 180° about X: R @ (x,y,z) = (x, -y, -z)
    handle_offset = np.array(
        [handle_local[0], -handle_local[1], -handle_local[2]],
        dtype=np.float64,
    )
    env.model.body_pos[sword_id] = tcp - handle_offset
    mujoco.mj_forward(env.model, env.data)


def test_grasp_attach_follow_and_release():
    """Near handle + close → attach; move arm → follow; open → release."""
    from gym_unoarm.constants import GRIPPER_CLOSE, GRIPPER_OPEN, RIGHT_GRIPPER_INDEX

    env = UnoarmEnv(scene=SCENE_REACH_SWORD, reach_success_threshold=0.05)
    env.reset()
    _place_sword_at_tcp(env)
    near_info = env._reach_info()
    assert near_info["reach_distance"] < 0.05
    assert near_info["success"] is False  # proximity alone is not success
    assert near_info["attached"] is False

    action = env._last_action.copy()
    action[RIGHT_GRIPPER_INDEX] = GRIPPER_CLOSE
    _, _, _, _, info = env.step(action)
    assert info["attached"] is True
    assert info["grasp_attached"] is True
    assert info["gripper_closed"] is True
    assert info["success"] is True

    handle_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_SITE, "sword_handle")
    tcp_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_SITE, "right_tcp")
    handle_before = np.asarray(env.data.site_xpos[handle_id], dtype=np.float64).copy()

    # Nudge a right-arm joint while closed — sword handle must track TCP.
    action2 = action.copy()
    action2[8] = float(np.clip(action2[8] + 0.25, -1.0, 1.0))  # Right_Joint1
    _, _, _, _, info2 = env.step(action2)
    assert info2["attached"] is True
    tcp_after = np.asarray(env.data.site_xpos[tcp_id], dtype=np.float64)
    handle_after = np.asarray(env.data.site_xpos[handle_id], dtype=np.float64)
    assert np.linalg.norm(tcp_after - handle_after) < 1e-4
    assert np.linalg.norm(handle_after - handle_before) > 1e-3

    # Open gripper → release; further arm motion leaves sword behind.
    action3 = action2.copy()
    action3[RIGHT_GRIPPER_INDEX] = GRIPPER_OPEN
    _, _, _, _, info3 = env.step(action3)
    assert info3["attached"] is False
    released_handle = np.asarray(env.data.site_xpos[handle_id], dtype=np.float64).copy()

    action4 = action3.copy()
    action4[8] = float(np.clip(action4[8] - 0.3, -1.0, 1.0))
    _, _, _, _, info4 = env.step(action4)
    assert info4["attached"] is False
    handle_stayed = np.asarray(env.data.site_xpos[handle_id], dtype=np.float64)
    assert np.allclose(handle_stayed, released_handle, atol=1e-5)
    env.close()


def test_grasp_does_not_attach_when_far():
    from gym_unoarm.constants import GRIPPER_CLOSE, RIGHT_GRIPPER_INDEX

    env = UnoarmEnv(scene=SCENE_REACH_SWORD)
    env.reset()
    assert env._reach_info()["reach_distance"] > REACH_SUCCESS_THRESHOLD
    action = env._last_action.copy()
    action[RIGHT_GRIPPER_INDEX] = GRIPPER_CLOSE
    _, _, _, _, info = env.step(action)
    assert info["attached"] is False
    assert info["gripper_closed"] is True
    env.close()


def test_reset_clears_attach_and_restores_sword():
    from gym_unoarm.constants import GRIPPER_CLOSE, RIGHT_GRIPPER_INDEX

    pos = (0.31, 0.62, 1.55)
    env = UnoarmEnv(scene=SCENE_REACH_SWORD, sword_body_pos=pos)
    env.reset()
    _place_sword_at_tcp(env)
    action = env._last_action.copy()
    action[RIGHT_GRIPPER_INDEX] = GRIPPER_CLOSE
    env.step(action)
    assert env._reach_info()["attached"] is True

    _, info = env.reset()
    assert info["attached"] is False
    sword_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, "sword")
    assert np.allclose(env.model.body_pos[sword_id], pos, atol=1e-6)
    env.close()
