from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data_gen.scripted import (  # noqa: E402
    RAW_ZERO,
    build_episode_actions,
    build_pose_sequence,
    load_poses_from_json,
    trim_leading_idle_actions,
)
from gym_unoarm.constants import CONTROL_JOINTS  # noqa: E402


def _pose_dict(values: dict[str, float] | None = None) -> dict[str, float]:
    base = dict.fromkeys(CONTROL_JOINTS, 0.0)
    if values:
        base.update(values)
    return base


def test_load_poses_from_json_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "demo.poses.json"
    payload = {
        "task": "Wave hello.",
        "poses": [
            _pose_dict({"Left_Joint1": 0.5}),
            _pose_dict({"Right_Joint1": -0.25}),
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")

    task, poses = load_poses_from_json(path)
    assert task == "Wave hello."
    assert len(poses) == 2
    assert poses[0][0] == pytest.approx(0.5)
    assert poses[1][8] == pytest.approx(-0.25)


def test_load_poses_rejects_unknown_joint(tmp_path: Path) -> None:
    path = tmp_path / "bad.poses.json"
    path.write_text(
        json.dumps({"task": "x", "poses": [{"NotAJoint": 1.0}]}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="unknown joint"):
        load_poses_from_json(path)


def test_build_pose_sequence_wraps_raw_zero() -> None:
    user = [np.ones(16, dtype=np.float32)]
    seq = build_pose_sequence(user)
    assert len(seq) == 3
    assert np.allclose(seq[0], RAW_ZERO)
    assert np.allclose(seq[-1], RAW_ZERO)
    assert np.allclose(seq[1], user[0])


def test_build_episode_actions_length() -> None:
    poses = [np.ones(16, dtype=np.float32), np.full(16, 0.5, dtype=np.float32)]
    rng = np.random.default_rng(0)
    # hold(RAW_ZERO) + interp to p0 + hold(p0) + interp to p1 + hold(p1)
    # Note: build_episode_actions treats `poses` as the full sequence including zeros
    # when called from generate after build_pose_sequence. Here pass two targets only.
    actions = build_episode_actions(poses, segment_steps=4, hold_steps=2, rng=rng, midpoint_noise_std=0.0)
    # hold zero(2) + seg(4) + hold(2) + seg(4) + hold(2) = 14
    assert len(actions) == 14


def test_trim_leading_idle_actions_removes_zero_prefix() -> None:
    poses = build_pose_sequence([np.ones(16, dtype=np.float32)])
    rng = np.random.default_rng(0)
    actions = build_episode_actions(
        poses,
        segment_steps=4,
        hold_steps=2,
        rng=rng,
        midpoint_noise_std=0.0,
    )

    trimmed, skipped = trim_leading_idle_actions(actions, RAW_ZERO, tolerance=1e-6)

    # hold zero(2) + zero->zero segment(4) + hold zero(2)
    assert skipped == 8
    assert len(trimmed) == len(actions) - skipped
    assert not np.allclose(trimmed[0], RAW_ZERO, atol=1e-6)
    assert np.allclose(trimmed[-1], RAW_ZERO, atol=1e-6)


def test_trim_leading_idle_actions_keeps_one_frame_for_empty_motion() -> None:
    actions = [RAW_ZERO.copy() for _ in range(5)]

    trimmed, skipped = trim_leading_idle_actions(actions, RAW_ZERO, tolerance=1e-6)

    assert skipped == 4
    assert len(trimmed) == 1
    assert np.allclose(trimmed[0], RAW_ZERO, atol=1e-6)


def test_pose_jitter_varies_episodes_without_frame_shake() -> None:
    """Episode-level pose jitter should diversify demos while keeping holds still."""
    poses = build_pose_sequence([np.full(16, 0.5, dtype=np.float32)])
    hold_steps = 3
    segment_steps = 5

    def _actions(seed: int, pose_jitter_std: float) -> list[np.ndarray]:
        rng = np.random.default_rng(seed)
        if pose_jitter_std > 0.0:
            jitter = rng.normal(0.0, pose_jitter_std, size=16).astype(np.float32)
            user = [poses[1] + jitter]
            seq = build_pose_sequence(user)
        else:
            seq = poses
        return build_episode_actions(
            seq,
            segment_steps=segment_steps,
            hold_steps=hold_steps,
            rng=rng,
            midpoint_noise_std=0.0,
            hold_noise_std=0.0,
        )

    a0 = np.stack(_actions(0, 0.05))
    a1 = np.stack(_actions(1, 0.05))
    clean = np.stack(_actions(0, 0.0))

    assert a0.shape == a1.shape == clean.shape
    assert not np.allclose(a0, a1, atol=1e-5)
    assert not np.allclose(a0, clean, atol=1e-5)

    # Hold at the start must be identical frames (no frame-level shake).
    assert np.allclose(a0[0], a0[hold_steps - 1], atol=1e-6)

    deltas = np.diff(a0, axis=0)
    sign = np.sign(deltas)
    flips = (sign[1:] * sign[:-1] < 0) & (np.abs(deltas[1:]) > 1e-4) & (np.abs(deltas[:-1]) > 1e-4)
    assert int(flips.sum()) == 0
