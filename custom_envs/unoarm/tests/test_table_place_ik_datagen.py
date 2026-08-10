"""Tests for table-place Reach-IK pick-and-place datagen."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import mujoco
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data_gen.reach_ik import list_reach_ik_datasets  # noqa: E402
from data_gen.table_place_ik import (  # noqa: E402
    TablePlaceIkGenConfig,
    TablePlaceIkParams,
    arm_clears_table_collision_box,
    arm_links_clear_peg,
    build_table_place_ik_poses,
    episode_peg_xy_from_meta,
    generate_table_place_ik_dataset,
    grasp_center_tcp_offset,
    lift_transport_clears_table,
    peg_bottom_clears_table,
    resolve_episode_peg_xy,
    sample_table_place_ik_params,
    table_collision_aabb,
    table_place_ik_config_from_params,
)
from gym_unoarm.constants import (  # noqa: E402
    PEG_DEFAULT_XY,
    PEG_HALF_HEIGHT,
    PEG_WORKSPACE_X_RANGE,
    PEG_WORKSPACE_Y_RANGE,
    RIGHT_GRIPPER_INDEX,
    SCENE_TABLE_PLACE,
    TABLE_PLACE_HALF,
    is_peg_xy_in_workspace,
    peg_table_inset_xy_bounds,
    peg_workspace_xy_bounds,
    sample_peg_xy,
    table_place_top_z,
)
from gym_unoarm.env import UnoarmEnv  # noqa: E402


def test_episode_peg_xy_from_meta_prefers_per_episode():
    meta = {
        "peg_xy": [-0.2, -0.58],
        "episodes": [
            {"episode_index": 0, "peg_xy": [-0.25, -0.60]},
            {"episode_index": 1, "peg_xy": [-0.15, -0.55]},
        ],
    }
    assert episode_peg_xy_from_meta(meta, 0) == (-0.25, -0.60)
    assert episode_peg_xy_from_meta(meta, 1) == (-0.15, -0.55)
    # Unknown episode falls back to legacy top-level peg_xy.
    assert episode_peg_xy_from_meta(meta, 9) == (-0.2, -0.58)
    assert episode_peg_xy_from_meta(None, 0) is None


def test_sample_params_respect_clearance_floor():
    rng = np.random.default_rng(0)
    top = table_place_top_z()
    clearance = 0.01
    floor = top + PEG_HALF_HEIGHT + clearance
    oris = set()
    for _ in range(40):
        p = sample_table_place_ik_params(
            rng,
            approach_offset_y_range=(0.04, 0.10),
            lift_z_range=(floor - 0.05, floor + 0.08),
            place_z_range=(floor - 0.05, floor + 0.08),
            table_top_z=top,
            peg_half_height=PEG_HALF_HEIGHT,
            table_clearance_m=clearance,
        )
        assert p.lift_z >= floor - 1e-9
        assert p.place_z >= floor - 1e-9
        assert 0.04 <= p.approach_offset_y <= 0.10
        assert p.grasp_orientation in ("horizontal", "vertical")
        oris.add(p.grasp_orientation)
    assert oris == {"horizontal", "vertical"}


def test_peg_bottom_clearance_and_tcp_offset():
    env = UnoarmEnv(scene=SCENE_TABLE_PLACE)
    env.reset()
    off = grasp_center_tcp_offset(env)
    assert off.shape == (3,)
    assert peg_bottom_clears_table(env, 0.01) is False
    env.model.body_pos[env._peg_body_id, 2] = table_place_top_z() + PEG_HALF_HEIGHT + 0.05
    mujoco.mj_forward(env.model, env.data)
    assert peg_bottom_clears_table(env, 0.01) is True
    assert lift_transport_clears_table(env, 0.01) is True
    assert arm_clears_table_collision_box(env, margin_m=0.01) is True
    center, half = table_collision_aabb(0.01)
    assert center.shape == (3,) and half.shape == (3,)
    assert half[0] > TABLE_PLACE_HALF[0]
    env.close()


def test_arm_links_clear_peg_at_home():
    env = UnoarmEnv(scene=SCENE_TABLE_PLACE)
    env.reset()
    assert arm_links_clear_peg(env) is True
    # Drive a link geom into the peg AABB by teleporting peg onto the forearm.
    peg_id = env._peg_body_id
    # Pick a right-arm link geom position and seat the peg there.
    from data_gen.table_place_ik import right_arm_geom_ids

    link_gid = None
    for gid in right_arm_geom_ids(env):
        bid = int(env.model.geom_bodyid[gid])
        import mujoco

        name = mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_BODY, bid) or ""
        # Use a proximal link (not Link7 / pads) that the pregrasp check covers.
        if name in {"Right_Link3", "Right_Link4", "Right_Link5"}:
            link_gid = gid
            break
    assert link_gid is not None
    env.model.body_pos[peg_id] = np.asarray(env.data.geom_xpos[link_gid], dtype=np.float64)
    mujoco.mj_forward(env.model, env.data)
    assert arm_links_clear_peg(env) is False
    env.close()


def test_build_table_place_ik_poses_shape():
    env = UnoarmEnv(scene=SCENE_TABLE_PLACE)
    top = table_place_top_z()
    floor = top + PEG_HALF_HEIGHT + 0.01
    built = 0
    for ori in ("horizontal", "vertical"):
        env.reset()
        params = TablePlaceIkParams(
            approach_offset_y=0.08,
            lift_z=floor + 0.12,
            place_z=floor + 0.05,
            grasp_orientation=ori,
        )
        poses = build_table_place_ik_poses(env, params)
        if poses is None:
            continue
        built += 1
        assert len(poses) >= 6
        for p in poses:
            assert p.shape == (16,)
        assert poses[0][RIGHT_GRIPPER_INDEX] > poses[2][RIGHT_GRIPPER_INDEX]
        assert poses[-1][RIGHT_GRIPPER_INDEX] > poses[2][RIGHT_GRIPPER_INDEX]
        # Lift through lower keyframes must clear the table box.
        for p in poses[3:-1]:
            env.apply_raw_qpos(p)
            assert arm_clears_table_collision_box(env, margin_m=0.01, require_aabb=True)
    assert built >= 1
    env.close()


def test_sample_peg_xy_stays_inside_workspace_and_table():
    rng = np.random.default_rng(42)
    (tx0, ty0), (tx1, ty1) = peg_table_inset_xy_bounds()
    (wx0, wy0), (wx1, wy1) = peg_workspace_xy_bounds()
    assert wx0 >= tx0 - 1e-9 and wx1 <= tx1 + 1e-9
    assert wy0 >= ty0 - 1e-9 and wy1 <= ty1 + 1e-9
    xs, ys = [], []
    for _ in range(80):
        xy = sample_peg_xy(rng)
        assert is_peg_xy_in_workspace(xy)
        assert tx0 - 1e-9 <= xy[0] <= tx1 + 1e-9
        assert ty0 - 1e-9 <= xy[1] <= ty1 + 1e-9
        xs.append(xy[0])
        ys.append(xy[1])
    # Multi-point: samples should not all collapse to one location.
    assert max(xs) - min(xs) > 0.05
    assert max(ys) - min(ys) > 0.02


def test_resolve_episode_peg_xy_clips_out_of_bounds():
    cfg = TablePlaceIkGenConfig(
        sample_peg_xy=False,
        peg_xy_fixed=(9.0, 9.0),
        peg_workspace_x_range=PEG_WORKSPACE_X_RANGE,
        peg_workspace_y_range=PEG_WORKSPACE_Y_RANGE,
    )
    rng = np.random.default_rng(0)
    xy = resolve_episode_peg_xy(cfg, rng)
    assert is_peg_xy_in_workspace(xy)
    (x0, y0), (x1, y1) = peg_workspace_xy_bounds()
    assert x0 - 1e-9 <= xy[0] <= x1 + 1e-9
    assert y0 - 1e-9 <= xy[1] <= y1 + 1e-9


def test_table_place_ik_config_from_params(tmp_path: Path):
    cfg = table_place_ik_config_from_params(
        {
            "num_episodes": 3,
            "approach_offset_y_min": 0.05,
            "approach_offset_y_max": 0.08,
            "table_clearance_m": 0.012,
            "seed": 7,
            "sample_peg_xy": True,
            "peg_x_min": -0.25,
            "peg_x_max": 0.05,
            "peg_y_min": -0.68,
            "peg_y_max": -0.52,
            "peg_edge_margin": 0.04,
            "episodes_per_peg": 2,
        },
        output_root=tmp_path / "out",
        repo_id="local/tp",
    )
    assert cfg.num_episodes == 3
    assert cfg.approach_offset_y_range == (0.05, 0.08)
    assert cfg.table_clearance_m == pytest.approx(0.012)
    assert cfg.seed == 7
    assert cfg.lift_z_range is not None
    assert cfg.place_z_range is not None
    assert cfg.sample_peg_xy is True
    assert cfg.peg_workspace_x_range == (-0.25, 0.05)
    assert cfg.peg_workspace_y_range == (-0.68, -0.52)
    assert cfg.peg_edge_margin == pytest.approx(0.04)
    assert cfg.episodes_per_peg == 2


def test_config_episodes_per_peg_from_params(tmp_path: Path):
    cfg = table_place_ik_config_from_params(
        {"episodes_per_peg": 3, "num_episodes": 6},
        output_root=tmp_path / "out2",
    )
    assert cfg.episodes_per_peg == 3
    assert cfg.num_episodes == 6


def test_generate_one_successful_episode(tmp_path: Path):
    top = table_place_top_z()
    floor = top + PEG_HALF_HEIGHT + 0.01
    cfg = TablePlaceIkGenConfig(
        num_episodes=1,
        segment_steps=6,
        hold_steps=1,
        settle_steps=45,
        approach_offset_y_range=(0.05, 0.07),
        lift_z_range=(floor + 0.10, floor + 0.14),
        place_z_range=(floor + 0.04, floor + 0.06),
        table_clearance_m=0.01,
        seed=0,
        output_root=tmp_path / "ds",
        repo_id="local/table_place_ik_test",
        overwrite=True,
        pose_jitter_std=0.0,
        max_attempts=40,
        ik_retries=10,
        # Fixed peg for a stable smoke; multi-point covered by sample unit tests.
        sample_peg_xy=False,
        peg_xy_fixed=PEG_DEFAULT_XY,
        episodes_per_peg=1,
    )
    out = generate_table_place_ik_dataset(cfg)
    assert out.is_dir()
    meta = json.loads((out / "table_place_ik_meta.json").read_text(encoding="utf-8"))
    assert meta["mode"] == "table_place_pick_place"
    assert meta["num_episodes_written"] == 1
    assert len(meta["episodes"]) == 1
    assert meta["episodes"][0]["success"] is True
    assert "peg_xy" in meta["episodes"][0]
    assert is_peg_xy_in_workspace(meta["episodes"][0]["peg_xy"])
    assert "peg_workspace" in meta
    assert meta["sample_peg_xy"] is False


def test_generate_discards_when_clearance_always_fails(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        "data_gen.table_place_ik.peg_bottom_clears_table",
        lambda *_a, **_k: False,
    )
    top = table_place_top_z()
    floor = top + PEG_HALF_HEIGHT + 0.01
    cfg = TablePlaceIkGenConfig(
        num_episodes=1,
        segment_steps=6,
        hold_steps=1,
        settle_steps=20,
        approach_offset_y_range=(0.05, 0.06),
        lift_z_range=(floor + 0.03, floor + 0.04),
        place_z_range=(floor + 0.02, floor + 0.03),
        table_clearance_m=0.01,
        seed=1,
        output_root=tmp_path / "ds_fail",
        repo_id="local/table_place_ik_fail",
        overwrite=True,
        max_attempts=3,
    )
    with pytest.raises(RuntimeError, match="No episodes written"):
        generate_table_place_ik_dataset(cfg)


def test_list_reach_ik_datasets_finds_table_place_meta(tmp_path: Path):
    root = tmp_path / "unoarm_table_place_web"
    root.mkdir()
    (root / "meta").mkdir()
    (root / "meta" / "info.json").write_text(
        json.dumps({"total_episodes": 2, "repo_id": "local/x"}),
        encoding="utf-8",
    )
    (root / "table_place_ik_meta.json").write_text(
        json.dumps(
            {
                "mode": "table_place_pick_place",
                "repo_id": "local/x",
                "num_episodes_written": 2,
                "episodes": [{}, {}],
            }
        ),
        encoding="utf-8",
    )
    items = list_reach_ik_datasets(tmp_path)
    match = [x for x in items if x["name"] == root.name]
    assert match
    assert match[0]["kind"] == "table_place_ik"
    assert match[0]["n_episodes"] == 2
