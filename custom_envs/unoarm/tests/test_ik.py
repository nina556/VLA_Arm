"""Tests for Unoarm right-arm MuJoCo DLS IK."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gym_unoarm.constants import SCENE_REACH_SWORD
from gym_unoarm.env import UnoarmEnv
from gym_unoarm.ik import FIXED_APPROACH_ROT_MAT, solve_right_tcp_ik


def test_fixed_approach_rot_is_orthonormal():
    R = FIXED_APPROACH_ROT_MAT
    assert R.shape == (3, 3)
    assert np.allclose(R.T @ R, np.eye(3), atol=1e-6)
    assert np.isclose(np.linalg.det(R), 1.0, atol=1e-5)


def test_ik_reaches_in_workspace_handle_default():
    env = UnoarmEnv(scene=SCENE_REACH_SWORD, remove_shield=True)
    env.reset()
    target = np.array([-0.35, -0.55, 1.0], dtype=np.float64)
    q = solve_right_tcp_ik(env, target)
    assert q is not None
    env.apply_raw_qpos(q)
    tcp = np.asarray(env.data.site_xpos[env._tcp_site_id], dtype=np.float64)
    assert np.linalg.norm(tcp - target) < 0.01
    env.close()


def test_ik_returns_none_far_away():
    env = UnoarmEnv(scene=SCENE_REACH_SWORD, remove_shield=True)
    env.reset()
    q = solve_right_tcp_ik(env, np.array([5.0, 5.0, 5.0]))
    assert q is None
    env.close()
