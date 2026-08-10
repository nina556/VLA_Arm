"""Tests for reach-IK target helpers and dataset smoke generation."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data_gen.reach_ik import (
    ReachIkGenConfig,
    build_reach_ik_poses,
    generate_reach_ik_dataset,
    load_targets_json,
    place_sword_handle,
    sample_targets_in_bbox,
)
from gym_unoarm.constants import SCENE_REACH_SWORD
from gym_unoarm.env import UnoarmEnv


def test_load_targets_json(tmp_path: Path):
    p = tmp_path / "t.json"
    p.write_text(
        json.dumps(
            {
                "task": "T",
                "execution_point_pos": [0, 0, 1],
                "targets": [{"xyz": [-0.3, -0.5, 1.0]}],
            }
        ),
        encoding="utf-8",
    )
    task, targets = load_targets_json(p)
    assert task == "T"
    assert len(targets) == 1
    assert np.allclose(targets[0], [-0.3, -0.5, 1.0])


def test_sample_targets_in_bbox():
    rng = np.random.default_rng(0)
    pts = sample_targets_in_bbox([-0.4, -0.6, 0.9], [-0.2, -0.4, 1.1], 20, rng)
    assert len(pts) == 20
    for p in pts:
        assert -0.4 <= p[0] <= -0.2
        assert -0.6 <= p[1] <= -0.4
        assert 0.9 <= p[2] <= 1.1


def test_place_sword_handle_updates_site():
    env = UnoarmEnv(scene=SCENE_REACH_SWORD, remove_shield=True)
    env.reset()
    handle = np.array([-0.28, -0.52, 1.02], dtype=np.float64)
    place_sword_handle(env, handle)
    env.reset()
    xpos = np.asarray(env.data.site_xpos[env._handle_site_id], dtype=np.float64)
    assert np.allclose(xpos, handle, atol=1e-4)
    env.close()


def test_build_reach_ik_poses_grasp_only():
    env = UnoarmEnv(
        scene=SCENE_REACH_SWORD,
        remove_shield=True,
        enable_execution_point=False,
    )
    env.reset()
    handle = np.array([-0.35, -0.55, 1.00], dtype=np.float64)
    place_sword_handle(env, handle)
    poses = build_reach_ik_poses(env, handle)
    assert poses is not None
    assert len(poses) == 3
    for p in poses:
        assert p.shape == (16,)
    env.close()


def test_generate_one_target_smoke(tmp_path: Path):
    p = tmp_path / "targets.json"
    p.write_text(
        json.dumps(
            {
                "task": "Grasp the sword handle",
                "targets": [{"xyz": [-0.35, -0.55, 1.00]}],
            }
        ),
        encoding="utf-8",
    )
    out = tmp_path / "ds"
    cfg = ReachIkGenConfig(
        targets_json=p,
        episodes_per_target=1,
        segment_steps=5,
        hold_steps=1,
        output_root=out,
        overwrite=True,
        repo_id="test/unoarm_reach_ik",
    )
    root = generate_reach_ik_dataset(cfg)
    assert root.exists()
    meta = json.loads((out / "reach_ik_meta.json").read_text(encoding="utf-8"))
    assert meta.get("mode") == "grasp_only"
    assert "execution_point_pos" not in meta
    assert len(meta["episodes"]) == 1
    assert meta["episodes"][0]["handle"] == pytest.approx([-0.35, -0.55, 1.0])


def test_bbox_resamples_until_success(tmp_path: Path):
    """AABB mode should keep drawing until num_targets succeed (not stop after N draws)."""
    out = tmp_path / "ds_bbox"
    cfg = ReachIkGenConfig(
        bbox_min=(-0.40, -0.60, 0.98),
        bbox_max=(-0.25, -0.48, 1.08),
        num_targets=2,
        episodes_per_target=1,
        segment_steps=4,
        hold_steps=1,
        output_root=out,
        overwrite=True,
        repo_id="test/unoarm_reach_ik_bbox",
        seed=1,
    )
    root = generate_reach_ik_dataset(cfg)
    meta = json.loads((root / "reach_ik_meta.json").read_text(encoding="utf-8"))
    assert meta["success_targets"] == 2
    assert len(meta["episodes"]) == 2
    assert meta["episodes"][0]["handle"] != meta["episodes"][1]["handle"]
